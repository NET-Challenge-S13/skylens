/// <reference types="node" />
// UPSTREAM link: client server → core (ws://localhost:8080/viewer).
//
// The core is the only producer of situation data (COMPONENTS.md §3.4); this
// module owns the socket to it and nothing else. It never interprets a message
// beyond unwrapping an optional Envelope — routing and caching live in
// boards.ts, so the relay stays transparent.
//
// Reconnection uses exponential backoff with jitter, and every state change is
// published so boards can show a WAITING state instead of a frozen screen.

import WebSocket from 'ws';
import type { Envelope, ViewerMessage } from '../../shared/protocol.ts';
import type { UpstreamState } from '../relayProtocol.ts';

export interface UpstreamEvents {
  onMessage(msg: ViewerMessage): void;
  onState(state: UpstreamState, detail: string): void;
}

export interface Upstream {
  start(): void;
  stop(): void;
  readonly state: UpstreamState;
  readonly detail: string;
  /** Unix ms the current state began. */
  readonly since: number;
  /** Failed connect attempts since the last successful open. */
  readonly retries: number;
  /** Frames received from the core (before any filtering). */
  readonly received: number;
  /** Frames dropped because they were not parseable JSON objects. */
  readonly malformed: number;
}

const VIEWER_KINDS = new Set([
  'splat-chunk',
  'detection',
  'telemetry',
  'mission-status',
  'server-status',
  'link-status',
]);

/**
 * The core may push a bare ViewerMessage or one wrapped in an Envelope (the
 * uplink hops all use envelopes). Accept both so neither side has to guess.
 */
function unwrap(raw: unknown): ViewerMessage | null {
  if (typeof raw !== 'object' || raw === null) return null;
  const obj = raw as Record<string, unknown>;
  if (typeof obj.kind === 'string' && VIEWER_KINDS.has(obj.kind)) return obj as unknown as ViewerMessage;
  const env = obj as Partial<Envelope<unknown>>;
  if (env.payload && typeof env.payload === 'object') return unwrap(env.payload);
  return null;
}

export function createUpstream(
  url: string,
  backoff: { minMs: number; maxMs: number },
  events: UpstreamEvents,
): Upstream {
  let ws: WebSocket | null = null;
  let timer: NodeJS.Timeout | null = null;
  let heartbeat: NodeJS.Timeout | null = null;
  let stopped = false;
  let state: UpstreamState = 'connecting';
  let detail = '코어 연결 시도 중';
  let since = Date.now();
  let retries = 0;
  let received = 0;
  let malformed = 0;

  function setState(next: UpstreamState, note: string): void {
    if (state === next && detail === note) return;
    state = next;
    detail = note;
    since = Date.now();
    events.onState(state, detail);
  }

  function delay(): number {
    const raw = Math.min(backoff.maxMs, backoff.minMs * 2 ** Math.min(retries, 6));
    // Jitter so several relays restarting together don't hammer the core in lockstep.
    return Math.round(raw * (0.7 + Math.random() * 0.6));
  }

  function schedule(): void {
    if (stopped || timer) return;
    const ms = delay();
    timer = setTimeout(() => {
      timer = null;
      connect();
    }, ms);
  }

  function connect(): void {
    if (stopped) return;
    setState('connecting', `코어 연결 시도 중 (${retries + 1}회)`);
    const sock = new WebSocket(url);
    ws = sock;

    sock.on('open', () => {
      retries = 0;
      setState('online', '코어 연결됨');
      if (heartbeat) clearInterval(heartbeat);
      heartbeat = setInterval(() => {
        if (sock.readyState === WebSocket.OPEN) sock.ping();
      }, 15_000);
    });

    sock.on('message', (data: WebSocket.RawData) => {
      received += 1;
      let parsed: unknown;
      try {
        parsed = JSON.parse(data.toString());
      } catch {
        malformed += 1;
        return;
      }
      const list = Array.isArray(parsed) ? parsed : [parsed];
      for (const item of list) {
        const msg = unwrap(item);
        if (msg) events.onMessage(msg);
        else malformed += 1;
      }
    });

    // A failed connect emits BOTH 'error' and 'close'. Counting each of those as
    // a retry doubles the backoff exponent and makes /health lie about how many
    // attempts have happened, so a socket may take this path only once.
    let settled = false;
    const down = (why: string): void => {
      if (settled) return;
      settled = true;
      if (ws === sock) ws = null;
      if (heartbeat) {
        clearInterval(heartbeat);
        heartbeat = null;
      }
      if (stopped) return;
      retries += 1;
      setState('offline', why);
      schedule();
    };

    sock.on('close', () => down('코어 연결 끊김 · 재연결 대기'));
    sock.on('error', (err: Error & { code?: string }) => {
      const why = err.message || err.code || String(err);
      down(`코어 응답 없음 · ${why}`);
    });
  }

  return {
    start(): void {
      stopped = false;
      connect();
    },
    stop(): void {
      stopped = true;
      if (timer) clearTimeout(timer);
      if (heartbeat) clearInterval(heartbeat);
      timer = null;
      heartbeat = null;
      try {
        ws?.close();
      } catch {
        /* ignore */
      }
    },
    get state() {
      return state;
    },
    get detail() {
      return detail;
    },
    get since() {
      return since;
    },
    get retries() {
      return retries;
    },
    get received() {
      return received;
    },
    get malformed() {
      return malformed;
    },
  };
}
