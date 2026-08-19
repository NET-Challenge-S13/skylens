// The control tower's ONE upstream link (COMPONENTS.md §2, §8).
//
// The tower does not talk to the situation board, and it does not simulate the
// fleet. It opens `ws://<host>:8080/viewer` on the core and:
//
//   core → tower :  ViewerMessage  (telemetry · mission-status · link-status ·
//                                   server-status · detection · splat-chunk)
//   tower → core :  ControlMessage (assign-route · manual-control)
//
// The core is the source of truth for where the drones are. When it is
// unreachable the tower says so — it does NOT keep flying an imaginary fleet,
// because an operator screen that invents drone positions is worse than one
// that admits it has none.
//
// The core wraps its pushes in an Envelope; control frames may be sent bare
// (distributor.ts accepts either), so that is what we send.

import { CONFIG } from '../shared/viewer/config.ts';
import type {
  CameraFeed,
  ControlMessage,
  DetectionResult,
  DroneTelemetry,
  Envelope,
  LinkStatus,
  MissionStatus,
  ServerStatus,
  SplatChunk,
  ViewerMessage,
} from '../shared/protocol.ts';

export type CoreLinkState = 'connecting' | 'connected' | 'disconnected';

export interface CoreLinkHandlers {
  onState?(state: CoreLinkState, detail: string): void;
  onTelemetry?(t: DroneTelemetry): void;
  onMission?(m: MissionStatus): void;
  onLinkStatus?(l: LinkStatus): void;
  onServerStatus?(s: ServerStatus): void;
  onDetection?(d: DetectionResult): void;
  onSplatChunk?(c: SplatChunk): void;
  onCameraFeed?(f: CameraFeed): void;
}

export interface CoreLink {
  readonly state: CoreLinkState;
  readonly url: string;
  /** Unix ms of the last frame received; 0 before the first one. */
  readonly lastMessageAt: number;
  /** Frames dropped because the socket was down (operator-visible honesty). */
  readonly dropped: number;
  /** Queue-free send: returns false when the core is not connected. */
  send(msg: ControlMessage): boolean;
  start(): void;
  stop(): void;
}

/** `?core=<host[:port]>` overrides the endpoint; otherwise this page's host. */
function resolveCoreUrl(): string {
  const { corePort, corePath } = CONFIG.control;
  if (typeof window === 'undefined') return `ws://localhost:${corePort}${corePath}`;
  const override = new URLSearchParams(window.location.search).get('core');
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  if (override) {
    if (/^wss?:\/\//.test(override)) return override;
    return `${proto}//${override.includes(':') ? override : `${override}:${corePort}`}${corePath}`;
  }
  const host = window.location.hostname || 'localhost';
  return `${proto}//${host}:${corePort}${corePath}`;
}

/** Unwrap an Envelope, or take the frame as-is when it is already a message. */
function unwrap(raw: unknown): ViewerMessage | null {
  const obj = raw as Partial<Envelope<ViewerMessage>> & Partial<ViewerMessage>;
  if (obj && typeof obj.kind === 'string') return obj as ViewerMessage;
  const payload = (obj as Envelope<ViewerMessage> | undefined)?.payload;
  if (payload && typeof payload.kind === 'string') return payload;
  return null;
}

export function createCoreLink(handlers: CoreLinkHandlers): CoreLink {
  const url = resolveCoreUrl();
  let ws: WebSocket | null = null;
  let state: CoreLinkState = 'disconnected';
  // Explicit `number`: CONFIG is `as const`, so this would otherwise infer the
  // literal type of reconnectMinMs and reject the backoff multiply below.
  let backoff: number = CONFIG.control.reconnectMinMs;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  let stopped = true;
  let lastMessageAt = 0;
  let dropped = 0;
  /** Suppress the "연결 끊김" toast-storm while the core is simply not up yet. */
  let everConnected = false;

  const setState = (next: CoreLinkState, detail: string): void => {
    if (state === next) return;
    state = next;
    handlers.onState?.(next, detail);
  };

  const dispatch = (msg: ViewerMessage): void => {
    lastMessageAt = Date.now();
    switch (msg.kind) {
      case 'telemetry':
        handlers.onTelemetry?.(msg);
        break;
      case 'mission-status':
        handlers.onMission?.(msg);
        break;
      case 'link-status':
        handlers.onLinkStatus?.(msg);
        break;
      case 'server-status':
        handlers.onServerStatus?.(msg);
        break;
      case 'detection':
        handlers.onDetection?.(msg);
        break;
      case 'splat-chunk':
        handlers.onSplatChunk?.(msg);
        break;
      case 'camera-feed':
        handlers.onCameraFeed?.(msg);
        break;
    }
  };

  const scheduleRetry = (): void => {
    if (stopped || retryTimer) return;
    const wait = backoff;
    backoff = Math.min(backoff * 2, CONFIG.control.reconnectMaxMs);
    retryTimer = setTimeout(() => {
      retryTimer = null;
      open();
    }, wait);
  };

  function open(): void {
    if (stopped) return;
    setState('connecting', everConnected ? '코어 재연결 시도 중' : '코어 연결 시도 중');
    let socket: WebSocket;
    try {
      socket = new WebSocket(url);
    } catch (err) {
      // Malformed URL (bad ?core=) — retrying will not fix it, but backing off
      // is still better than a tight failure loop.
      setState('disconnected', `코어 주소가 올바르지 않습니다 (${String(err)})`);
      scheduleRetry();
      return;
    }
    ws = socket;

    socket.addEventListener('open', () => {
      if (socket !== ws) return;
      everConnected = true;
      backoff = CONFIG.control.reconnectMinMs;
      setState('connected', '코어 연결됨');
    });

    socket.addEventListener('message', (ev) => {
      if (socket !== ws) return;
      let parsed: unknown;
      try {
        parsed = JSON.parse(String(ev.data));
      } catch {
        console.warn('[coreLink] dropped non-JSON frame');
        return;
      }
      const msg = unwrap(parsed);
      if (msg) dispatch(msg);
    });

    const down = (detail: string): void => {
      if (socket !== ws) return;
      ws = null;
      setState('disconnected', detail);
      scheduleRetry();
    };
    socket.addEventListener('close', () => down('코어 연결 끊김'));
    // 'error' always precedes 'close'; close carries the retry, this only logs.
    socket.addEventListener('error', () => {
      if (socket === ws && state === 'connecting' && !everConnected) {
        setState('disconnected', '코어에 연결할 수 없습니다');
      }
    });
  }

  return {
    get state() {
      return state;
    },
    url,
    get lastMessageAt() {
      return lastMessageAt;
    },
    get dropped() {
      return dropped;
    },

    send(msg: ControlMessage): boolean {
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        dropped += 1;
        return false;
      }
      try {
        ws.send(JSON.stringify(msg));
        return true;
      } catch (err) {
        console.warn('[coreLink] send failed:', err);
        dropped += 1;
        return false;
      }
    },

    start(): void {
      if (!stopped) return;
      stopped = false;
      open();
    },

    stop(): void {
      stopped = true;
      if (retryTimer) clearTimeout(retryTimer);
      retryTimer = null;
      const socket = ws;
      ws = null;
      socket?.close();
      setState('disconnected', '연결 종료');
    },
  };
}
