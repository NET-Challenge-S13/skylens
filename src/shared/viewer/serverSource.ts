// Viewer stream — the browser end of a pipeline push feed.
//
// WHAT CHANGED AND WHY. This module used to be a MOCK that synthesized the
// delay-pattern on client-side timers (it read res/static/demo/segments.json and
// scheduled segment × level chunks itself). COMPONENTS.md §3.4 puts that
// schedule in the core and nowhere else — "클라이언트가 타이머로 스스로 단계를
// 진행시키는 구조는 폐기한다" — so the whole scheduler is gone. What is left is a
// plain subscriber: open a socket, hand every frame to the page, reconnect when
// it drops, and say out loud whether the link is up.
//
// It is deliberately generic over the ENDPOINT, because both viewers need the
// same thing at different addresses:
//   situation board  → ws://<client server>:8090/stream   (relayed by skylens_client)
//   control tower    → ws://<core>:8080/viewer            (direct, plus control uplink)
//
// Frames that are not ViewerMessages (the relay's own relay-hello / relay-status)
// are passed to onOther untouched — this layer does not know that protocol.

import type { ViewerMessage } from '../protocol.ts';
import { isViewerMessageKind } from '../protocol.ts';

/** State of one browser→server socket. */
export type StreamState = 'connecting' | 'online' | 'offline';

export interface ViewerStreamOptions {
  /** ws:// or wss:// endpoint. */
  url: string;
  /** Reconnect backoff bounds, ms. */
  backoffMinMs?: number;
  backoffMaxMs?: number;
  /** Prefix for console diagnostics. */
  label?: string;
}

export interface ViewerStream {
  /** Anything the pipeline pushed that matches shared/protocol.ts §8. */
  onMessage(cb: (msg: ViewerMessage) => void): void;
  /** Frames with a `kind` this layer doesn't know (relay framing, extensions). */
  onOther(cb: (frame: Record<string, unknown>) => void): void;
  onState(cb: (state: StreamState, detail: string) => void): void;
  /** Upstream control (control tower → core). No-op while offline. */
  send(msg: unknown): boolean;
  start(): void;
  stop(): void;
  readonly state: StreamState;
  readonly detail: string;
  /** Frames received since the page loaded. */
  readonly received: number;
}

/**
 * The core may push a bare ViewerMessage or one wrapped in an Envelope (every
 * uplink hop uses envelopes). Accept both so no hop has to guess.
 */
function unwrap(raw: unknown): unknown {
  if (typeof raw !== 'object' || raw === null) return null;
  const obj = raw as Record<string, unknown>;
  if (typeof obj.kind === 'string') return obj;
  if (obj.payload && typeof obj.payload === 'object') return unwrap(obj.payload);
  return null;
}

export function createViewerStream(opts: ViewerStreamOptions): ViewerStream {
  const msgCbs: Array<(m: ViewerMessage) => void> = [];
  const otherCbs: Array<(f: Record<string, unknown>) => void> = [];
  const stateCbs: Array<(s: StreamState, d: string) => void> = [];
  const minMs = opts.backoffMinMs ?? 600;
  const maxMs = opts.backoffMaxMs ?? 8000;
  const label = opts.label ?? 'stream';

  let ws: WebSocket | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let stopped = false;
  let state: StreamState = 'connecting';
  let detail = '연결 시도 중';
  let retries = 0;
  let received = 0;

  function setState(next: StreamState, note: string): void {
    if (state === next && detail === note) return;
    state = next;
    detail = note;
    for (const cb of stateCbs) cb(state, detail);
  }

  function schedule(): void {
    if (stopped || timer) return;
    const raw = Math.min(maxMs, minMs * 2 ** Math.min(retries, 6));
    timer = setTimeout(() => {
      timer = null;
      connect();
    }, Math.round(raw * (0.7 + Math.random() * 0.6)));
  }

  function dispatch(raw: unknown): void {
    const frame = unwrap(raw) as Record<string, unknown> | null;
    if (!frame) return;
    received += 1;
    if (isViewerMessageKind(frame.kind)) {
      const msg = frame as unknown as ViewerMessage;
      for (const cb of msgCbs) cb(msg);
    } else {
      for (const cb of otherCbs) cb(frame);
    }
  }

  function connect(): void {
    if (stopped) return;
    setState('connecting', retries === 0 ? '연결 시도 중' : `재연결 시도 ${retries}회`);
    let sock: WebSocket;
    try {
      sock = new WebSocket(opts.url);
    } catch (e) {
      retries += 1;
      setState('offline', e instanceof Error ? e.message : '연결 실패');
      schedule();
      return;
    }
    ws = sock;

    sock.onopen = () => {
      retries = 0;
      setState('online', '연결됨');
    };
    sock.onmessage = (ev: MessageEvent) => {
      if (typeof ev.data !== 'string') return;
      let parsed: unknown;
      try {
        parsed = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (Array.isArray(parsed)) for (const item of parsed) dispatch(item);
      else dispatch(parsed);
    };
    const down = (why: string): void => {
      if (ws === sock) ws = null;
      if (stopped) return;
      retries += 1;
      setState('offline', why);
      schedule();
    };
    sock.onclose = () => down('연결 끊김 · 재연결 대기');
    sock.onerror = () => {
      console.warn(`[${label}] socket error`, opts.url);
    };
  }

  return {
    onMessage: (cb) => void msgCbs.push(cb),
    onOther: (cb) => void otherCbs.push(cb),
    onState: (cb) => {
      stateCbs.push(cb);
      cb(state, detail);
    },
    send(msg: unknown): boolean {
      if (!ws || ws.readyState !== WebSocket.OPEN) return false;
      ws.send(JSON.stringify(msg));
      return true;
    },
    start(): void {
      stopped = false;
      connect();
    },
    stop(): void {
      stopped = true;
      if (timer) clearTimeout(timer);
      timer = null;
      try {
        ws?.close();
      } catch {
        /* ignore */
      }
      ws = null;
    },
    get state() {
      return state;
    },
    get detail() {
      return detail;
    },
    get received() {
      return received;
    },
  };
}
