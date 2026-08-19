// relay mode — the default path.
//
//   드론 ──ws──> 게이트웨이 [relay] ──ws──> 프록시 ──> 코어
//
// The gateway is a transport: it does not read, rewrite or reorder the payload.
// It parses each frame only far enough to (a) confirm it is a JSON Envelope and
// (b) append its own hop stamp, then hands it upstream.
//
// Video is referenced by `uri`, never inlined, so a frame stays small enough that
// the drop-oldest queue in upstream.ts is a meaningful backpressure valve.

import type { WebSocket } from 'ws';
import type { WebSocketServer } from 'ws';
import type { Envelope, LinkStatus, UplinkMessage } from '../shared/protocol.ts';
import type { GatewayConfig } from './config.ts';
import type { StampedEnvelope } from './types.ts';
import { stamp } from './types.ts';
import { UpstreamLink } from './upstream.ts';

export interface RelayCounters {
  dronesConnected: number;
  dronesSeen: number;
  framesIn: number;
  /** Handed to the upstream link without being dropped — queued counts, since a
   *  queued frame still goes out when the proxy returns. Actual writes are in
   *  the `upstream` block of /health. */
  framesAccepted: number;
  framesRejected: number;
  bytesIn: number;
  mbps: number | null;
  lastSeq: number;
  lastOriginTs: number | null;
}

interface DroneConn {
  ws: WebSocket;
  droneId: number | null;
  since: number;
  frames: number;
}

export class RelayGateway {
  private cfg: GatewayConfig;
  private upstream: UpstreamLink;
  private drones = new Set<DroneConn>();
  private dronesSeen = 0;
  private framesIn = 0;
  private framesAccepted = 0;
  private framesRejected = 0;
  private bytesIn = 0;
  private bytesWindow = 0;
  private windowStart = Date.now();
  private mbps: number | null = null;
  private lastSeq = 0;
  private lastOriginTs: number | null = null;
  private seq = 0;
  private statusTimer: ReturnType<typeof setInterval> | null = null;

  constructor(cfg: GatewayConfig) {
    this.cfg = cfg;
    this.upstream = new UpstreamLink({
      url: cfg.proxyIngressUrl,
      label: 'gateway',
      maxQueue: cfg.maxQueue,
      maxAgeMs: cfg.maxQueueAgeMs,
      reconnectMs: cfg.reconnectMs,
      pingMs: cfg.pingMs,
      // The proxy pushes link/mission status back down; fan it out to the drones.
      onMessage: (text) => this.broadcast(text),
    });
  }

  start(wss: WebSocketServer): void {
    this.upstream.start();
    wss.on('connection', (ws) => this.accept(ws));
    this.statusTimer = setInterval(() => this.tick(), this.cfg.statusMs);
  }

  stop(): void {
    if (this.statusTimer) clearInterval(this.statusTimer);
    this.statusTimer = null;
    this.upstream.stop();
    for (const conn of this.drones) conn.ws.close();
  }

  counters(): RelayCounters {
    return {
      dronesConnected: this.drones.size,
      dronesSeen: this.dronesSeen,
      framesIn: this.framesIn,
      framesAccepted: this.framesAccepted,
      framesRejected: this.framesRejected,
      bytesIn: this.bytesIn,
      mbps: this.mbps,
      lastSeq: this.lastSeq,
      lastOriginTs: this.lastOriginTs,
    };
  }

  upstreamCounters() {
    return this.upstream.counters();
  }

  private accept(ws: WebSocket): void {
    const conn: DroneConn = { ws, droneId: null, since: Date.now(), frames: 0 };
    this.drones.add(conn);
    this.dronesSeen += 1;
    console.log(`[gateway] drone connected (${this.drones.size} online)`);
    this.sendLinkStatus(conn);

    ws.on('message', (data) => {
      const text = data.toString();
      this.bytesIn += text.length;
      this.bytesWindow += text.length;
      this.framesIn += 1;
      conn.frames += 1;
      this.forward(conn, text);
    });

    ws.on('close', () => {
      this.drones.delete(conn);
      console.log(
        `[gateway] drone ${conn.droneId ?? '?'} disconnected after ${conn.frames} frame(s) ` +
          `(${this.drones.size} online)`,
      );
    });

    ws.on('error', (err) => console.warn(`[gateway] drone socket error: ${err.message}`));
  }

  private forward(conn: DroneConn, text: string): void {
    const rx = Date.now();
    let env: StampedEnvelope<UplinkMessage>;
    try {
      env = JSON.parse(text) as StampedEnvelope<UplinkMessage>;
    } catch {
      this.framesRejected += 1;
      console.warn('[gateway] dropped non-JSON frame from drone');
      return;
    }
    if (typeof env !== 'object' || env === null || typeof env.payload !== 'object') {
      this.framesRejected += 1;
      console.warn('[gateway] dropped frame with no Envelope payload');
      return;
    }

    const payload = env.payload as UplinkMessage;
    if (conn.droneId === null && 'droneId' in payload) {
      conn.droneId = payload.droneId;
      console.log(`[gateway] drone ${payload.droneId} identified (${payload.kind})`);
    }
    this.lastSeq = env.seq;
    this.lastOriginTs = env.originTs;

    // Transport only: payload, seq, originTs and from are untouched.
    const out = stamp(env, { at: 'gateway', rx, tx: Date.now() });
    const ok = this.upstream.send(JSON.stringify(out));
    if (ok) this.framesAccepted += 1;
  }

  private broadcast(text: string): void {
    for (const conn of this.drones) {
      if (conn.ws.readyState === 1) conn.ws.send(text);
    }
  }

  /** LinkStatus for the hop this component owns: drone→gateway. */
  private linkStatus(): LinkStatus {
    return {
      kind: 'link-status',
      hop: 'drone→gateway',
      connected: this.drones.size > 0,
      mode: 'relay',
      latencyMs: this.upstream.counters().latencyMs,
      mbps: this.mbps,
    };
  }

  private envelope(status: LinkStatus): Envelope<LinkStatus> {
    this.seq += 1;
    return { seq: this.seq, originTs: Date.now(), from: 'gateway', payload: status };
  }

  private sendLinkStatus(conn: DroneConn): void {
    if (conn.ws.readyState === 1) conn.ws.send(JSON.stringify(this.envelope(this.linkStatus())));
  }

  private tick(): void {
    const now = Date.now();
    const elapsed = now - this.windowStart;
    if (elapsed >= 1000) {
      this.mbps = (this.bytesWindow * 8) / elapsed / 1000;
      this.bytesWindow = 0;
      this.windowStart = now;
    }
    const frame = JSON.stringify(this.envelope(this.linkStatus()));
    this.broadcast(frame);
    // The core renders the badges, so the same status rides the uplink too.
    this.upstream.send(frame);
  }
}
