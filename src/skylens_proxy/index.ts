// skylens_proxy — KOREN 내부망.
//
// Accepts BOTH shapes of uplink (COMPONENTS.md §2, §3.3):
//   /ingress  게이트웨이 [relay] 가 중계한 연결
//   /direct   드론이 홀펀칭 후 직접 붙는 연결 (webrtc 모드)
//   /signal   게이트웨이와의 홀펀칭 시그널링 제어 채널
//
// and fans everything out over a REDUNDANT set of core paths with health
// checking, automatic failover and failback (corePaths.ts).
//
// The proxy is a transport, not a translator: `payload`, `seq`, `originTs` and
// `from` are forwarded byte-for-byte in meaning. The only thing it adds is its
// own hop stamp (types.ts), which carries `via` so the core can attribute
// latency to the specific KOREN path the frame took.
//
// Run:  npx tsx src/skylens_proxy/index.ts

import http from 'node:http';
import process from 'node:process';
import express from 'express';
import { WebSocketServer } from 'ws';
import type { WebSocket } from 'ws';
import type { Envelope, LinkMode, LinkStatus, UplinkMessage } from '../shared/protocol.ts';
import { loadConfig } from './config.ts';
import { CorePathManager } from './corePaths.ts';
import type { FailoverEvent } from './corePaths.ts';
import { stamp } from './types.ts';
import type { SignalFrame, StampedEnvelope } from './types.ts';

const cfg = loadConfig();
const startedAt = Date.now();

type IngressKind = 'gateway' | 'drone-direct';

interface Ingress {
  ws: WebSocket;
  kind: IngressKind;
  since: number;
  frames: number;
}

const ingresses = new Set<Ingress>();
const signalControls = new Set<WebSocket>();
/** sessionId → the control socket the punch is being brokered over. */
const sessions = new Map<string, WebSocket>();

let seq = 0;
let framesIn = 0;
let framesRejected = 0;
let bytesIn = 0;
let bytesWindow = 0;
let windowStart = Date.now();
let mbps: number | null = null;
let lastSeq = 0;
let signalsIn = 0;
let signalsOut = 0;
let punches = 0;
/** Cumulative per-ingress totals. `open` alone hides a drone that already left,
 *  which is exactly the drone you want to see when debugging a punch. */
const seen: Record<IngressKind, number> = { gateway: 0, 'drone-direct': 0 };
const framesBy: Record<IngressKind, number> = { gateway: 0, 'drone-direct': 0 };

const core = new CorePathManager({
  endpoints: cfg.coreEndpoints,
  failback: cfg.failback,
  reconnectMs: cfg.reconnectMs,
  healthIntervalMs: cfg.healthIntervalMs,
  healthTimeoutMs: cfg.healthTimeoutMs,
  maxQueue: cfg.maxQueue,
  maxAgeMs: cfg.maxQueueAgeMs,
  startupGraceMs: cfg.startupGraceMs,
  // Downlink from the core (mission status, control) goes back to whoever is
  // attached — the gateway fans it out to the drones.
  onMessage: (text) => broadcast(text),
  onActiveChange: (event) => onFailover(event),
});

// ---------------------------------------------------------------------------
// Uplink
// ---------------------------------------------------------------------------

function ingressMode(): LinkMode {
  for (const i of ingresses) if (i.kind === 'drone-direct') return 'webrtc';
  return 'relay';
}

function acceptIngress(ws: WebSocket, kind: IngressKind): void {
  const conn: Ingress = { ws, kind, since: Date.now(), frames: 0 };
  ingresses.add(conn);
  seen[kind] += 1;
  console.log(`[proxy] ${kind} attached (${ingresses.size} ingress open)`);
  sendTo(ws, linkStatusIngress());

  ws.on('message', (data) => {
    const text = data.toString();
    bytesIn += text.length;
    bytesWindow += text.length;
    framesIn += 1;
    framesBy[kind] += 1;
    conn.frames += 1;
    forward(text, kind);
  });

  ws.on('close', () => {
    ingresses.delete(conn);
    console.log(`[proxy] ${kind} detached after ${conn.frames} frame(s)`);
  });

  ws.on('error', (err) => console.warn(`[proxy] ${kind} socket error: ${err.message}`));
}

function forward(text: string, kind: IngressKind): void {
  const rx = Date.now();
  let env: StampedEnvelope<UplinkMessage>;
  try {
    env = JSON.parse(text) as StampedEnvelope<UplinkMessage>;
  } catch {
    framesRejected += 1;
    console.warn(`[proxy] dropped non-JSON frame from ${kind}`);
    return;
  }
  if (typeof env !== 'object' || env === null || typeof env.payload !== 'object') {
    framesRejected += 1;
    console.warn(`[proxy] dropped frame with no Envelope payload from ${kind}`);
    return;
  }
  lastSeq = env.seq;

  const via = core.activeUrl;
  const out = stamp(env, { at: 'proxy', rx, tx: Date.now(), via: via === null ? undefined : via });
  const result = core.send(JSON.stringify(out));
  const payload = env.payload as { kind?: string };
  if (payload.kind === 'video-segment' || payload.kind === 'drone-hello') {
    console.log(
      `[proxy] ${payload.kind} seq=${env.seq} from ${kind} -> ` +
        `${result.ok ? result.via : 'BUFFERED (no healthy core path)'}`,
    );
  }
}

function broadcast(text: string): void {
  for (const i of ingresses) if (i.ws.readyState === 1) i.ws.send(text);
}

// ---------------------------------------------------------------------------
// Hole-punch signalling (webrtc mode)
// ---------------------------------------------------------------------------
//
// The proxy is the answering peer. Real SDP/ICE from the drone is passed through
// untouched; what this build cannot do is terminate a DataChannel in Node, so the
// endpoint handed back with signal-ready is a WebSocket at /direct standing in
// for the negotiated channel. Swapping in node-datachannel replaces exactly this
// function and nothing else — the signalling contract stays the same.

function acceptSignalControl(ws: WebSocket): void {
  signalControls.add(ws);
  console.log(`[proxy] gateway signalling control attached (${signalControls.size})`);

  ws.on('message', (data) => {
    let env: Envelope<SignalFrame>;
    try {
      env = JSON.parse(data.toString()) as Envelope<SignalFrame>;
    } catch {
      return;
    }
    const frame = env === null ? null : env.payload;
    if (!frame || typeof frame.sessionId !== 'string') return;
    signalsIn += 1;
    sessions.set(frame.sessionId, ws);

    if (frame.kind === 'signal-offer') {
      console.log(`[proxy] offer for session ${frame.sessionId} (drone ${frame.droneId})`);
      reply(ws, {
        kind: 'signal-answer',
        sessionId: frame.sessionId,
        droneId: frame.droneId,
        mode: 'webrtc',
        sdp: `v=0\r\no=skylens-proxy ${Date.now()} 1 IN IP4 0.0.0.0\r\ns=skylens-uplink\r\n`,
      });
      reply(ws, {
        kind: 'signal-ice',
        sessionId: frame.sessionId,
        droneId: frame.droneId,
        mode: 'webrtc',
        candidate: { candidate: 'skylens-proxy-host', sdpMid: '0', sdpMLineIndex: 0 },
      });
      punches += 1;
      reply(ws, {
        kind: 'signal-ready',
        sessionId: frame.sessionId,
        droneId: frame.droneId,
        mode: 'webrtc',
        direct: cfg.publicDirectUrl,
      });
      console.log(
        `[proxy] session ${frame.sessionId} ready — drone ${frame.droneId} now sends direct ` +
          `to ${cfg.publicDirectUrl}, gateway is out of the media path`,
      );
      return;
    }

    if (frame.kind === 'signal-bye') {
      sessions.delete(frame.sessionId);
      console.log(`[proxy] session ${frame.sessionId} closed (${frame.reason ?? 'no reason'})`);
    }
    // ICE trickle from the drone: nothing to do beyond acknowledging receipt.
  });

  ws.on('close', () => {
    signalControls.delete(ws);
    for (const [id, sock] of sessions) if (sock === ws) sessions.delete(id);
    console.log(`[proxy] gateway signalling control detached (${signalControls.size})`);
  });

  ws.on('error', (err) => console.warn(`[proxy] signalling socket error: ${err.message}`));
}

function reply(ws: WebSocket, frame: SignalFrame): void {
  if (ws.readyState !== 1) return;
  seq += 1;
  const env: Envelope<SignalFrame> = {
    seq,
    originTs: Date.now(),
    from: 'proxy',
    payload: frame,
  };
  ws.send(JSON.stringify(env));
  signalsOut += 1;
}

// ---------------------------------------------------------------------------
// LinkStatus — the hops this component owns
// ---------------------------------------------------------------------------

function linkStatusIngress(): LinkStatus {
  const mode = ingressMode();
  return {
    kind: 'link-status',
    hop: mode === 'webrtc' ? 'drone→proxy (direct)' : 'gateway→proxy',
    connected: ingresses.size > 0,
    mode,
    latencyMs: null,
    mbps,
  };
}

function linkStatusCore(): LinkStatus {
  const url = core.activeUrl;
  return {
    kind: 'link-status',
    // Free-form by contract — the active KOREN path is named here so the badge
    // can show WHICH line is carrying traffic, not just that one is.
    hop: url === null ? 'proxy→core (no path)' : `proxy→core (${url})`,
    connected: core.connected,
    mode: ingressMode(),
    latencyMs: core.activeLatencyMs,
    mbps,
  };
}

function envelope(status: LinkStatus): Envelope<LinkStatus> {
  seq += 1;
  return { seq, originTs: Date.now(), from: 'proxy', payload: status };
}

function sendTo(ws: WebSocket, status: LinkStatus): void {
  if (ws.readyState === 1) ws.send(JSON.stringify(envelope(status)));
}

function onFailover(event: FailoverEvent): void {
  const text = JSON.stringify(envelope(linkStatusCore()));
  broadcast(text);
  core.send(text);
  console.log(
    `[proxy] link-status: ${event.from ?? 'none'} -> ${event.to ?? 'none'} (${event.reason})`,
  );
}

function tick(): void {
  const now = Date.now();
  const elapsed = now - windowStart;
  if (elapsed >= 1000) {
    mbps = (bytesWindow * 8) / elapsed / 1000;
    bytesWindow = 0;
    windowStart = now;
  }
  const ingressText = JSON.stringify(envelope(linkStatusIngress()));
  const coreText = JSON.stringify(envelope(linkStatusCore()));
  broadcast(ingressText);
  core.send(coreText);
}

// ---------------------------------------------------------------------------
// Servers
// ---------------------------------------------------------------------------

const app = express();
const server = http.createServer(app);
const wssIngress = new WebSocketServer({ noServer: true });
const wssDirect = new WebSocketServer({ noServer: true });
const wssSignal = new WebSocketServer({ noServer: true });

app.get('/health', (_req, res) => {
  res.json({
    component: 'skylens_proxy',
    ok: true,
    uptimeSec: Math.round((Date.now() - startedAt) / 1000),
    listen: { port: cfg.port, ingress: '/ingress', direct: '/direct', signal: '/signal' },
    ingress: {
      mode: ingressMode(),
      open: ingresses.size,
      gateway: [...ingresses].filter((i) => i.kind === 'gateway').length,
      droneDirect: [...ingresses].filter((i) => i.kind === 'drone-direct').length,
      seenGateway: seen.gateway,
      seenDroneDirect: seen['drone-direct'],
      framesFromGateway: framesBy.gateway,
      framesFromDroneDirect: framesBy['drone-direct'],
    },
    signalling: {
      controls: signalControls.size,
      sessions: sessions.size,
      punches,
      signalsIn,
      signalsOut,
    },
    upstream: core.counters(),
    counters: { framesIn, framesRejected, bytesIn, mbps, lastSeq },
  });
});

wssIngress.on('connection', (ws) => acceptIngress(ws, 'gateway'));
wssDirect.on('connection', (ws) => acceptIngress(ws, 'drone-direct'));
wssSignal.on('connection', (ws) => acceptSignalControl(ws));

server.on('upgrade', (req, socket, head) => {
  const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
  const target =
    url.pathname === '/ingress'
      ? wssIngress
      : url.pathname === '/direct'
        ? wssDirect
        : url.pathname === '/signal'
          ? wssSignal
          : null;
  if (target === null) {
    console.warn(`[proxy] rejected upgrade on ${url.pathname}`);
    socket.destroy();
    return;
  }
  target.handleUpgrade(req, socket, head, (ws) => target.emit('connection', ws, req));
});

core.start();
const statusTimer = setInterval(tick, cfg.statusMs);

server.listen(cfg.port, cfg.host, () => {
  console.log(`[proxy] listening on http://${cfg.host}:${cfg.port}`);
  console.log(`[proxy]   ws://${cfg.host}:${cfg.port}/ingress  (gateway relay)`);
  console.log(`[proxy]   ws://${cfg.host}:${cfg.port}/direct   (drone, post hole-punch)`);
  console.log(`[proxy]   ws://${cfg.host}:${cfg.port}/signal   (gateway signalling control)`);
  console.log(`[proxy] health: http://${cfg.host}:${cfg.port}/health`);
});

function shutdown(signal: string): void {
  console.log(`[proxy] ${signal} — shutting down`);
  clearInterval(statusTimer);
  core.stop();
  for (const i of ingresses) i.ws.close();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 1000).unref();
}

process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));
