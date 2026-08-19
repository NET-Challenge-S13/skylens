// Reconnecting WebSocket client with a bounded, drop-oldest send queue.
//
// The rule this file exists to enforce: a drone must NEVER be blocked by the
// state of the KOREN interior. If the proxy is down we hold a short burst of
// frames, then start discarding the oldest ones and count how many we lost.
// The drone keeps flying and keeps sending.

import { WebSocket } from 'ws';

export interface UpstreamCounters {
  url: string;
  connected: boolean;
  queued: number;
  sent: number;
  /** Dropped because the queue was full (backpressure). */
  droppedOverflow: number;
  /** Dropped because they aged out while queued. */
  droppedStale: number;
  reconnects: number;
  latencyMs: number | null;
  lastError: string | null;
}

export interface UpstreamOptions {
  url: string;
  label: string;
  maxQueue: number;
  maxAgeMs: number;
  reconnectMs: number;
  pingMs: number;
  onMessage?: (text: string) => void;
  onOpen?: () => void;
  onClose?: () => void;
}

interface Frame {
  text: string;
  at: number;
}

export class UpstreamLink {
  readonly url: string;
  private opts: UpstreamOptions;
  private ws: WebSocket | null = null;
  private queue: Frame[] = [];
  private timer: ReturnType<typeof setTimeout> | null = null;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private pingSentAt = 0;
  private stopped = false;
  private sent = 0;
  private droppedOverflow = 0;
  private droppedStale = 0;
  private reconnects = 0;
  private latencyMs: number | null = null;
  private lastError: string | null = null;

  constructor(opts: UpstreamOptions) {
    this.opts = opts;
    this.url = opts.url;
  }

  start(): void {
    this.stopped = false;
    this.open();
  }

  stop(): void {
    this.stopped = true;
    if (this.timer) clearTimeout(this.timer);
    if (this.pingTimer) clearInterval(this.pingTimer);
    this.timer = null;
    this.pingTimer = null;
    this.ws?.close();
    this.ws = null;
  }

  get connected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  counters(): UpstreamCounters {
    return {
      url: this.url,
      connected: this.connected,
      queued: this.queue.length,
      sent: this.sent,
      droppedOverflow: this.droppedOverflow,
      droppedStale: this.droppedStale,
      reconnects: this.reconnects,
      latencyMs: this.latencyMs,
      lastError: this.lastError,
    };
  }

  /** Never throws, never blocks. Returns false when the frame was dropped. */
  send(text: string): boolean {
    this.expire();
    // Anything still queued must go first, or the drone's ordering breaks. The
    // drain is attempted here too (not only on reconnect) so a queue can never
    // sit forever behind a link that came back without a fresh open event.
    if (this.connected && this.queue.length > 0) this.drain();
    if (this.connected && this.queue.length === 0) {
      this.write(text);
      return true;
    }
    if (this.queue.length >= this.opts.maxQueue) {
      this.queue.shift();
      this.droppedOverflow += 1;
      if (this.droppedOverflow % 50 === 1) {
        console.warn(
          `[${this.opts.label}] upstream backpressure: queue full (${this.opts.maxQueue}), ` +
            `dropped oldest — ${this.droppedOverflow} total`,
        );
      }
      this.queue.push({ text, at: Date.now() });
      return false;
    }
    this.queue.push({ text, at: Date.now() });
    return true;
  }

  private write(text: string): boolean {
    try {
      this.ws?.send(text);
      this.sent += 1;
      return true;
    } catch (err) {
      this.lastError = String(err);
      return false;
    }
  }

  private expire(): void {
    if (this.queue.length === 0) return;
    const cutoff = Date.now() - this.opts.maxAgeMs;
    let removed = 0;
    while (this.queue.length > 0 && this.queue[0].at < cutoff) {
      this.queue.shift();
      removed += 1;
    }
    if (removed > 0) {
      this.droppedStale += removed;
      console.warn(
        `[${this.opts.label}] dropped ${removed} frame(s) older than ` +
          `${this.opts.maxAgeMs}ms — ${this.droppedStale} total`,
      );
    }
  }

  private drain(): void {
    this.expire();
    while (this.queue.length > 0 && this.connected) {
      const frame = this.queue[0];
      // Only consume the frame once the socket actually took it, otherwise a
      // write that fails mid-drain would silently swallow it.
      if (!this.write(frame.text)) break;
      this.queue.shift();
    }
  }

  private open(): void {
    if (this.stopped) return;
    const ws = new WebSocket(this.url);
    this.ws = ws;

    ws.on('open', () => {
      this.lastError = null;
      console.log(`[${this.opts.label}] upstream connected: ${this.url}`);
      const held = this.queue.length;
      this.drain();
      if (held > 0) console.log(`[${this.opts.label}] flushed ${held} buffered frame(s)`);
      this.pingTimer = setInterval(() => {
        if (!this.connected) return;
        this.pingSentAt = Date.now();
        ws.ping();
      }, this.opts.pingMs);
      this.opts.onOpen?.();
    });

    ws.on('pong', () => {
      if (this.pingSentAt > 0) this.latencyMs = Date.now() - this.pingSentAt;
    });

    ws.on('message', (data) => {
      this.opts.onMessage?.(data.toString());
    });

    ws.on('error', (err) => {
      this.lastError = err.message;
    });

    ws.on('close', () => {
      if (this.pingTimer) clearInterval(this.pingTimer);
      this.pingTimer = null;
      this.latencyMs = null;
      if (this.ws === ws) this.ws = null;
      this.opts.onClose?.();
      if (this.stopped) return;
      this.reconnects += 1;
      console.warn(
        `[${this.opts.label}] upstream down (${this.lastError ?? 'closed'}), ` +
          `retry in ${this.opts.reconnectMs}ms — buffering ${this.queue.length} frame(s)`,
      );
      this.timer = setTimeout(() => this.open(), this.opts.reconnectMs);
    });
  }
}
