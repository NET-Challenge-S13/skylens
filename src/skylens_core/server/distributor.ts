// Viewer distribution — and the WebRTC seam.
//
// COMPONENTS.md §3.4-4 says the core distributes scene and markers to situation
// boards over WebRTC. Node has no built-in WebRTC and we are not pulling in a
// native module at this stage, so the core pushes over a plain WebSocket to
// skylens_client, which owns the browser-facing WebRTC relay (COMPONENTS.md
// §3.6). The orchestrator never learns which of the two it is talking to: it only
// ever sees this interface, so a WebRtcDistributor can be dropped in later
// without touching a line of scheduling code.

import type { WebSocket, WebSocketServer } from 'ws';
import type { ControlMessage, Envelope, ViewerMessage } from '../../shared/protocol.ts';

export type ViewerSend = (msg: ViewerMessage) => void;

export interface Distributor {
  /** Shown in /health so an operator can see which transport is live. */
  readonly transport: string;
  viewers(): number;
  broadcast(msg: ViewerMessage): void;
  /** Fired per viewer as it joins, so the core can replay current state. */
  onJoin(fn: (send: ViewerSend) => void): void;
  /** Control tower -> core (assign-route, manual-control). */
  onControl(fn: (msg: ControlMessage, send: ViewerSend) => void): void;
  stop(): void;
}

export interface DistributorCounters {
  transport: string;
  viewers: number;
  viewersSeen: number;
  sent: number;
  controlIn: number;
  rejected: number;
}

export class WsDistributor implements Distributor {
  readonly transport = 'websocket';
  private clients = new Set<WebSocket>();
  private joinHandlers: Array<(send: ViewerSend) => void> = [];
  private controlHandlers: Array<(msg: ControlMessage, send: ViewerSend) => void> = [];
  private seq = 0;
  private viewersSeen = 0;
  private sent = 0;
  private controlIn = 0;
  private rejected = 0;

  start(wss: WebSocketServer): void {
    wss.on('connection', (ws) => this.accept(ws));
  }

  viewers(): number {
    return this.clients.size;
  }

  counters(): DistributorCounters {
    return {
      transport: this.transport,
      viewers: this.clients.size,
      viewersSeen: this.viewersSeen,
      sent: this.sent,
      controlIn: this.controlIn,
      rejected: this.rejected,
    };
  }

  onJoin(fn: (send: ViewerSend) => void): void {
    this.joinHandlers.push(fn);
  }

  onControl(fn: (msg: ControlMessage, send: ViewerSend) => void): void {
    this.controlHandlers.push(fn);
  }

  broadcast(msg: ViewerMessage): void {
    if (this.clients.size === 0) return;
    const text = this.wrap(msg);
    for (const ws of this.clients) this.raw(ws, text);
  }

  stop(): void {
    for (const ws of this.clients) ws.close();
    this.clients.clear();
  }

  private accept(ws: WebSocket): void {
    this.clients.add(ws);
    this.viewersSeen += 1;
    console.log(`[core] viewer connected (${this.clients.size} online)`);

    const send: ViewerSend = (msg) => this.raw(ws, this.wrap(msg));
    for (const fn of this.joinHandlers) fn(send);

    ws.on('message', (data) => this.receive(data.toString(), send));
    ws.on('close', () => {
      this.clients.delete(ws);
      console.log(`[core] viewer disconnected (${this.clients.size} online)`);
    });
    ws.on('error', (err) => console.warn(`[core] viewer socket error: ${err.message}`));
  }

  /** Accepts a bare ControlMessage or one inside an Envelope — the control
   *  tower is a browser, and making it build envelopes buys nothing. */
  private receive(text: string, send: ViewerSend): void {
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      this.rejected += 1;
      console.warn('[core] dropped non-JSON frame from viewer');
      return;
    }
    const obj = parsed as Partial<Envelope<ControlMessage>> & Partial<ControlMessage>;
    const msg = (
      typeof obj?.kind === 'string' ? obj : (obj as Envelope<ControlMessage>).payload
    ) as ControlMessage | undefined;
    if (!msg || typeof msg.kind !== 'string') {
      this.rejected += 1;
      console.warn('[core] dropped viewer frame with no ControlMessage');
      return;
    }
    // Read the tag as a plain string first: the value came off the wire, so it
    // can be anything, but narrowing against the union would leave `never` here.
    const kind: string = msg.kind;
    if (kind !== 'assign-route' && kind !== 'manual-control') {
      this.rejected += 1;
      console.warn(`[core] viewer sent unsupported kind "${kind}"`);
      return;
    }
    this.controlIn += 1;
    for (const fn of this.controlHandlers) fn(msg, send);
  }

  private wrap(msg: ViewerMessage): string {
    this.seq += 1;
    const env: Envelope<ViewerMessage> = {
      seq: this.seq,
      originTs: Date.now(),
      from: 'core',
      payload: msg,
    };
    return JSON.stringify(env);
  }

  private raw(ws: WebSocket, text: string): void {
    // 1 === WebSocket.OPEN; compared numerically so this file needs no ws value import.
    if (ws.readyState !== 1) return;
    try {
      ws.send(text);
      this.sent += 1;
    } catch (err) {
      console.warn(`[core] viewer send failed: ${String(err)}`);
    }
  }
}
