// webrtc mode — hole punching only.
//
//   드론 ──(시그널링만)── 게이트웨이 [webrtc]
//     └──────── WebRTC 직결 ────────> 프록시 → 코어
//
// No media crosses this file. The gateway pairs a drone with the proxy, pumps
// SDP/ICE between them until the punch lands, then hands the drone the proxy's
// direct endpoint and gets out of the way. From that point the drone's frames
// never touch the gateway again — which is the whole point of the mode.
//
// Sessions are multiplexed over ONE gateway↔proxy control socket and demuxed by
// `sessionId`, so the proxy needs a single long-lived connection per gateway.

import type { WebSocket, WebSocketServer } from 'ws';
import type { Envelope, LinkStatus } from '../shared/protocol.ts';
import type { GatewayConfig } from './config.ts';
import type { SignalFrame } from './types.ts';
import { UpstreamLink } from './upstream.ts';

export interface SignallingCounters {
  sessionsOpen: number;
  sessionsTotal: number;
  /** Punches the proxy confirmed with signal-ready. */
  sessionsEstablished: number;
  toProxy: number;
  toDrone: number;
  unmatched: number;
}

interface Session {
  id: string;
  ws: WebSocket;
  droneId: number | null;
  since: number;
  established: boolean;
}

export class SignallingGateway {
  private cfg: GatewayConfig;
  private control: UpstreamLink;
  private sessions = new Map<string, Session>();
  private sessionsTotal = 0;
  private sessionsEstablished = 0;
  private toProxy = 0;
  private toDrone = 0;
  private unmatched = 0;
  private seq = 0;
  private nextId = 1;
  private statusTimer: ReturnType<typeof setInterval> | null = null;

  constructor(cfg: GatewayConfig) {
    this.cfg = cfg;
    this.control = new UpstreamLink({
      url: cfg.proxySignalUrl,
      label: 'gateway/signal',
      maxQueue: cfg.maxQueue,
      maxAgeMs: cfg.maxQueueAgeMs,
      reconnectMs: cfg.reconnectMs,
      pingMs: cfg.pingMs,
      onMessage: (text) => this.fromProxy(text),
    });
  }

  start(wss: WebSocketServer): void {
    this.control.start();
    wss.on('connection', (ws) => this.accept(ws));
    this.statusTimer = setInterval(() => this.tick(), this.cfg.statusMs);
  }

  stop(): void {
    if (this.statusTimer) clearInterval(this.statusTimer);
    this.statusTimer = null;
    this.control.stop();
    for (const s of this.sessions.values()) s.ws.close();
  }

  counters(): SignallingCounters {
    return {
      sessionsOpen: this.sessions.size,
      sessionsTotal: this.sessionsTotal,
      sessionsEstablished: this.sessionsEstablished,
      toProxy: this.toProxy,
      toDrone: this.toDrone,
      unmatched: this.unmatched,
    };
  }

  upstreamCounters() {
    return this.control.counters();
  }

  private accept(ws: WebSocket): void {
    const id = `s${this.nextId++}-${Date.now().toString(36)}`;
    const session: Session = { id, ws, droneId: null, since: Date.now(), established: false };
    this.sessions.set(id, session);
    this.sessionsTotal += 1;
    console.log(`[gateway/signal] session ${id} opened (${this.sessions.size} open)`);

    // Tell the drone which session it is in before it sends an offer.
    this.toSession(session, {
      kind: 'signal-hello',
      sessionId: id,
      droneId: 0,
      mode: 'webrtc',
    });

    ws.on('message', (data) => this.fromDrone(session, data.toString()));

    ws.on('close', () => {
      this.sessions.delete(id);
      console.log(
        `[gateway/signal] session ${id} closed ` +
          `(${session.established ? 'punch established' : 'not established'})`,
      );
      this.toProxyFrame({
        kind: 'signal-bye',
        sessionId: id,
        droneId: session.droneId ?? 0,
        mode: 'webrtc',
        reason: 'drone closed',
      });
    });

    ws.on('error', (err) => console.warn(`[gateway/signal] ${id} socket error: ${err.message}`));
  }

  private fromDrone(session: Session, text: string): void {
    let env: Envelope<SignalFrame>;
    try {
      env = JSON.parse(text) as Envelope<SignalFrame>;
    } catch {
      console.warn(`[gateway/signal] ${session.id} sent non-JSON, ignored`);
      return;
    }
    const frame = env?.payload;
    if (!frame || typeof frame.kind !== 'string') {
      console.warn(`[gateway/signal] ${session.id} sent a frame with no signal payload`);
      return;
    }
    if (session.droneId === null && typeof frame.droneId === 'number') {
      session.droneId = frame.droneId;
    }
    // Force our own session id: a drone cannot address someone else's punch.
    this.toProxyFrame({ ...frame, sessionId: session.id, droneId: session.droneId ?? 0 });
  }

  private fromProxy(text: string): void {
    let env: Envelope<SignalFrame>;
    try {
      env = JSON.parse(text) as Envelope<SignalFrame>;
    } catch {
      return;
    }
    const frame = env?.payload;
    if (!frame || typeof frame.sessionId !== 'string') return;
    const session = this.sessions.get(frame.sessionId);
    if (!session) {
      this.unmatched += 1;
      console.warn(`[gateway/signal] proxy replied for unknown session ${frame.sessionId}`);
      return;
    }
    if (frame.kind === 'signal-ready' && !session.established) {
      session.established = true;
      this.sessionsEstablished += 1;
      console.log(
        `[gateway/signal] ${session.id} punched through — drone ${session.droneId ?? '?'} ` +
          `now talks to ${frame.direct ?? 'the proxy'} directly; gateway carries no media`,
      );
    }
    this.toSession(session, frame);
  }

  private toProxyFrame(frame: SignalFrame): void {
    this.seq += 1;
    const env: Envelope<SignalFrame> = {
      seq: this.seq,
      originTs: Date.now(),
      from: 'gateway',
      payload: frame,
    };
    if (this.control.send(JSON.stringify(env))) this.toProxy += 1;
  }

  private toSession(session: Session, frame: SignalFrame): void {
    if (session.ws.readyState !== 1) return;
    this.seq += 1;
    const env: Envelope<SignalFrame> = {
      seq: this.seq,
      originTs: Date.now(),
      from: 'gateway',
      payload: frame,
    };
    session.ws.send(JSON.stringify(env));
    this.toDrone += 1;
  }

  private linkStatus(): LinkStatus {
    return {
      kind: 'link-status',
      hop: 'drone→gateway (signalling only)',
      connected: this.sessions.size > 0,
      mode: 'webrtc',
      latencyMs: this.control.counters().latencyMs,
      // Media bypasses this hop entirely, so there is no bitrate to report.
      mbps: null,
    };
  }

  private tick(): void {
    this.seq += 1;
    const env: Envelope<LinkStatus> = {
      seq: this.seq,
      originTs: Date.now(),
      from: 'gateway',
      payload: this.linkStatus(),
    };
    const text = JSON.stringify(env);
    for (const s of this.sessions.values()) {
      if (s.ws.readyState === 1) s.ws.send(text);
    }
  }
}
