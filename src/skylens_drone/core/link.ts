// The uplink to the gateway.
//
// Both link modes (COMPONENTS.md §3.2) carry exactly the same UplinkMessage
// shapes; only the carrier differs:
//
//   relay  — everything rides the gateway WebSocket at /uplink, which forwards
//            to the proxy. One socket, nothing to negotiate.
//
//   webrtc — the gateway socket at /signal carries hole-punch signalling ONLY.
//            The gateway pairs us with the proxy, pumps SDP/ICE across, and
//            answers with `signal-ready { direct }`. From that moment the uplink
//            moves to `direct` and the gateway never sees another byte of media
//            — which is the entire point of the mode. If the punch never lands
//            we say so and drop frames rather than quietly shovelling media down
//            the signalling socket, which the gateway would not forward anyway.
//
// Runtime-agnostic: the socket is injected (browser `WebSocket`, `ws` in Node /
// the Tauri sidecar), so the very same class runs in the Tauri app, a plain
// browser page, and the headless demo runner.

import type {
  ComponentId,
  ControlMessage,
  Envelope,
  LinkMode,
  UplinkMessage,
} from '../../shared/protocol.ts';

export type LinkPhase = 'offline' | 'connecting' | 'punching' | 'connected' | 'reconnecting';
export type Carrier = 'none' | 'ws' | 'direct';

export interface LinkState {
  phase: LinkPhase;
  carrier: Carrier;
  mode: LinkMode;
  /** Gateway URL — /uplink in relay mode, /signal in webrtc mode. */
  url: string;
  /** Endpoint the proxy handed over with signal-ready (webrtc only). */
  directUrl: string | null;
  /** Punch session id the gateway assigned (webrtc only). */
  sessionId: string | null;
  attempts: number;
  /** Uplink frames handed to a carrier since process start. */
  sent: number;
  /** Frames discarded because no carrier was up. */
  dropped: number;
  lastError: string | null;
}

export interface SocketLike {
  send(data: string): void;
  close(): void;
  onopen: ((ev: unknown) => void) | null;
  onclose: ((ev: unknown) => void) | null;
  onerror: ((ev: unknown) => void) | null;
  onmessage: ((ev: { data: unknown }) => void) | null;
}

export type SocketFactory = (url: string) => SocketLike;

/**
 * Hole-punch signalling frames.
 *
 * These are NOT in shared/protocol.ts on purpose: they describe how one
 * transport hop brokers a direct path, not the cross-component message
 * contract. This declaration mirrors `SignalFrame` in skylens_gateway/types.ts
 * structurally instead of importing it — a component must not reach into
 * another component's internals — so if the gateway changes its dialect this
 * file is the single place the drone has to follow.
 */
export type SignalKind =
  | 'signal-hello'
  | 'signal-offer'
  | 'signal-answer'
  | 'signal-ice'
  | 'signal-ready'
  | 'signal-bye';

export interface SignalFrame {
  kind: SignalKind;
  sessionId: string;
  droneId: number;
  mode: LinkMode;
  sdp?: string;
  candidate?: unknown;
  /** Direct proxy endpoint, present on signal-ready. */
  direct?: string;
  reason?: string;
}

const CONTROL_KINDS = new Set(['assign-route', 'manual-control']);

export interface LinkOptions {
  url: string;
  mode: LinkMode;
  droneId: number;
  socketFactory?: SocketFactory;
  reconnectMinMs?: number;
  reconnectMaxMs?: number;
  /** Warn (and keep waiting) if the punch has not landed after this long. */
  webrtcTimeoutMs?: number;
  onControl: (msg: ControlMessage) => void;
  onState: (state: LinkState) => void;
  onLog?: (line: string) => void;
}

function defaultSocketFactory(url: string): SocketLike {
  const Ctor = (globalThis as Record<string, unknown>).WebSocket as
    | (new (u: string) => SocketLike)
    | undefined;
  if (!Ctor) {
    throw new Error(
      'no global WebSocket; pass socketFactory (e.g. (u) => new (require("ws").WebSocket)(u))',
    );
  }
  return new Ctor(url);
}

export class GatewayLink {
  private opts: LinkOptions;
  private socket: SocketLike | null = null;
  /** webrtc mode: the post-punch socket straight to the proxy. */
  private direct: SocketLike | null = null;
  private seq = 0;
  private closed = false;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private punchTimer: ReturnType<typeof setTimeout> | null = null;
  private state: LinkState;

  constructor(opts: LinkOptions) {
    this.opts = opts;
    this.state = {
      phase: 'offline',
      carrier: 'none',
      mode: opts.mode,
      url: opts.url,
      directUrl: null,
      sessionId: null,
      attempts: 0,
      sent: 0,
      dropped: 0,
      lastError: null,
    };
  }

  get snapshot(): LinkState {
    return { ...this.state };
  }

  /** True only when an uplink frame can actually leave the drone. */
  get connected(): boolean {
    return this.state.phase === 'connected';
  }

  private patch(next: Partial<LinkState>): void {
    this.state = { ...this.state, ...next };
    this.opts.onState(this.snapshot);
  }

  private log(line: string): void {
    this.opts.onLog?.(line);
  }

  connect(): void {
    if (this.closed) return;
    this.patch({
      phase: this.state.attempts === 0 ? 'connecting' : 'reconnecting',
      attempts: this.state.attempts + 1,
    });
    let socket: SocketLike;
    try {
      socket = (this.opts.socketFactory ?? defaultSocketFactory)(this.opts.url);
    } catch (err) {
      this.patch({ phase: 'offline', lastError: String(err) });
      this.scheduleRetry();
      return;
    }
    this.socket = socket;

    socket.onopen = () => {
      if (this.opts.mode === 'relay') {
        this.patch({ phase: 'connected', carrier: 'ws', lastError: null });
        this.log(`gateway socket open (relay) ${this.opts.url} — uplink rides this socket`);
      } else {
        this.patch({ phase: 'punching', carrier: 'none', lastError: null });
        this.log(`signalling socket open (webrtc) ${this.opts.url} — waiting for signal-hello`);
        this.armPunchTimer();
      }
    };
    socket.onmessage = (ev) => this.receive(ev.data);
    socket.onerror = (ev) => {
      this.patch({ lastError: describeError(ev) });
    };
    socket.onclose = () => {
      this.closeDirect('gateway socket closed');
      if (this.closed) return;
      this.patch({ phase: 'offline', carrier: 'none', sessionId: null, directUrl: null });
      this.log('gateway socket closed');
      this.scheduleRetry();
    };
  }

  private scheduleRetry(): void {
    if (this.closed || this.retryTimer) return;
    const min = this.opts.reconnectMinMs ?? 500;
    const max = this.opts.reconnectMaxMs ?? 8000;
    const wait = Math.min(max, min * 2 ** Math.min(this.state.attempts, 5));
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      this.connect();
    }, wait);
  }

  /** Send an uplink message, wrapped in the Envelope every hop propagates. */
  send(payload: UplinkMessage): boolean {
    const target = this.opts.mode === 'webrtc' ? this.direct : this.socket;
    if (!target || !this.connected) {
      this.patch({ dropped: this.state.dropped + 1 });
      return false;
    }
    const env: Envelope<UplinkMessage> = {
      seq: this.seq++,
      originTs: Date.now(),
      from: 'drone' as ComponentId,
      payload,
    };
    try {
      target.send(JSON.stringify(env));
      this.patch({ sent: this.state.sent + 1 });
      return true;
    } catch (err) {
      this.patch({ lastError: String(err), dropped: this.state.dropped + 1 });
      return false;
    }
  }

  private receive(data: unknown): void {
    let parsed: unknown;
    try {
      parsed = JSON.parse(typeof data === 'string' ? data : String(data));
    } catch {
      return;
    }
    // Accept both a bare message and an Envelope-wrapped one: every hop rewraps
    // on the way down and the drone should not care which did.
    const body =
      parsed && typeof parsed === 'object' && 'payload' in (parsed as Record<string, unknown>)
        ? (parsed as Envelope).payload
        : parsed;
    if (!body || typeof body !== 'object') return;
    const kind = (body as { kind?: string }).kind;
    if (typeof kind !== 'string') return;
    if (kind.startsWith('signal-')) {
      this.onSignal(body as SignalFrame);
      return;
    }
    if (CONTROL_KINDS.has(kind)) {
      this.opts.onControl(body as ControlMessage);
    }
    // Anything else (link-status, mission-status) is informational; the app
    // layer does not need it to fly, so it is ignored here rather than leaked.
  }

  // -------------------------------------------------------------------------
  // webrtc mode — hole punching
  // -------------------------------------------------------------------------

  private armPunchTimer(): void {
    if (this.punchTimer) clearTimeout(this.punchTimer);
    const wait = this.opts.webrtcTimeoutMs ?? 8000;
    this.punchTimer = setTimeout(() => {
      this.punchTimer = null;
      if (this.direct) return;
      this.log(
        `no direct path after ${wait} ms — the punch has not landed. Uplink frames are being ` +
          'dropped: the gateway does not relay media in webrtc mode.',
      );
      this.patch({ lastError: 'hole punch not completed' });
    }, wait);
  }

  private onSignal(frame: SignalFrame): void {
    switch (frame.kind) {
      case 'signal-hello': {
        // The gateway names the session; we echo that id on everything after.
        this.patch({ sessionId: frame.sessionId });
        this.log(`punch session ${frame.sessionId} opened — sending offer`);
        this.sendSignal({
          kind: 'signal-offer',
          sessionId: frame.sessionId,
          droneId: this.opts.droneId,
          mode: 'webrtc',
          sdp: this.offerSdp(),
        });
        return;
      }
      case 'signal-answer':
        this.log('signal-answer received from the proxy');
        return;
      case 'signal-ice':
        // Trickle from the answering side. With a real ICE agent this would go
        // to addIceCandidate(); this build has none (see README).
        return;
      case 'signal-ready': {
        if (!frame.direct) {
          this.log('signal-ready without a direct endpoint — ignoring');
          return;
        }
        this.log(`punch landed — media goes direct to ${frame.direct}, gateway carries none of it`);
        this.openDirect(frame.direct);
        return;
      }
      case 'signal-bye':
        this.closeDirect(frame.reason ?? 'peer said bye');
        return;
      default:
        return;
    }
  }

  /**
   * The offer body. There is no ICE agent behind this: the proxy answers with a
   * WebSocket endpoint standing in for the negotiated channel (its own
   * documented limitation — Node has no DataChannel here), so the SDP is a
   * well-formed placeholder that keeps the signalling exchange honest end to
   * end. Swapping in a real RTCPeerConnection replaces this method and
   * openDirect(); nothing else in the drone changes.
   */
  private offerSdp(): string {
    return (
      'v=0\r\n' +
      `o=skylens-drone ${Date.now()} 1 IN IP4 0.0.0.0\r\n` +
      's=skylens-uplink\r\n' +
      't=0 0\r\n' +
      'm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n'
    );
  }

  private sendSignal(frame: SignalFrame): void {
    const env: Envelope<SignalFrame> = {
      seq: this.seq++,
      originTs: Date.now(),
      from: 'drone' as ComponentId,
      payload: frame,
    };
    try {
      this.socket?.send(JSON.stringify(env));
    } catch (err) {
      this.patch({ lastError: String(err) });
    }
  }

  private openDirect(url: string): void {
    if (this.direct) return;
    let socket: SocketLike;
    try {
      socket = (this.opts.socketFactory ?? defaultSocketFactory)(url);
    } catch (err) {
      this.patch({ lastError: String(err) });
      return;
    }
    this.direct = socket;
    this.patch({ directUrl: url });

    socket.onopen = () => {
      if (this.punchTimer) clearTimeout(this.punchTimer);
      this.punchTimer = null;
      this.patch({ phase: 'connected', carrier: 'direct', lastError: null });
      this.log(`direct uplink open: ${url}`);
    };
    socket.onmessage = (ev) => this.receive(ev.data);
    socket.onerror = (ev) => this.patch({ lastError: describeError(ev) });
    socket.onclose = () => {
      if (this.direct !== socket) return;
      this.direct = null;
      if (this.closed) return;
      this.patch({ phase: 'punching', carrier: 'none', directUrl: null });
      this.log('direct uplink closed — re-punching');
      // The signalling socket is still up; ask for a new session.
      const sessionId = this.state.sessionId;
      if (sessionId) {
        this.sendSignal({
          kind: 'signal-offer',
          sessionId,
          droneId: this.opts.droneId,
          mode: 'webrtc',
          sdp: this.offerSdp(),
        });
        this.armPunchTimer();
      }
    };
  }

  private closeDirect(reason: string): void {
    if (this.punchTimer) clearTimeout(this.punchTimer);
    this.punchTimer = null;
    const socket = this.direct;
    this.direct = null;
    if (socket) {
      socket.onclose = null;
      socket.close();
      this.log(`direct uplink torn down (${reason})`);
    }
  }

  close(): void {
    this.closed = true;
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.retryTimer = null;
    if (this.state.sessionId && this.socket) {
      this.sendSignal({
        kind: 'signal-bye',
        sessionId: this.state.sessionId,
        droneId: this.opts.droneId,
        mode: 'webrtc',
        reason: 'drone shutting down',
      });
    }
    this.closeDirect('link closed');
    this.socket?.close();
    this.socket = null;
    this.patch({ phase: 'offline', carrier: 'none', directUrl: null, sessionId: null });
  }
}

function describeError(ev: unknown): string {
  if (ev && typeof ev === 'object' && 'message' in ev) return String((ev as { message: unknown }).message);
  return 'socket error';
}
