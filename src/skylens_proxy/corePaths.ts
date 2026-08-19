// Redundant KOREN paths to the core.
//
// 재난 상황에서 한 경로가 죽어도 서비스가 유지되어야 한다 (COMPONENTS.md §3.3).
//
// Every configured endpoint gets its own socket and is kept WARM — standbys stay
// connected and probed so a failover is a pointer move, not a reconnect. Traffic
// only ever leaves through the active one.
//
// Health is two-sided on purpose:
//   1. the socket is open, AND
//   2. a ping was answered within healthTimeoutMs.
// (2) is what catches the nastier disaster case: the line is nominally up but the
// far end has stopped responding. A close alone would not tell us.

import { WebSocket } from 'ws';

export interface CorePathReport {
  url: string;
  priority: number;
  connected: boolean;
  healthy: boolean;
  active: boolean;
  latencyMs: number | null;
  sent: number;
  reconnects: number;
  /** ms since the last pong, null when never seen. */
  lastPongAgoMs: number | null;
  lastError: string | null;
}

export interface FailoverEvent {
  at: number;
  from: string | null;
  to: string | null;
  reason: string;
}

interface PathOptions {
  reconnectMs: number;
  healthIntervalMs: number;
  healthTimeoutMs: number;
  onDown: (path: CorePath) => void;
  onUp: (path: CorePath) => void;
  onMessage: (text: string) => void;
}

export class CorePath {
  readonly url: string;
  readonly priority: number;
  private opts: PathOptions;
  private ws: WebSocket | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private probe: ReturnType<typeof setInterval> | null = null;
  private stopped = false;
  private pingSentAt = 0;
  private lastPongAt = 0;
  private wasHealthy = false;
  /** True once the first connect attempt has resolved either way. Lets the
   *  manager avoid promoting a standby just because it opened a few ms sooner. */
  settled = false;
  latencyMs: number | null = null;
  sent = 0;
  reconnects = 0;
  lastError: string | null = null;

  constructor(url: string, priority: number, opts: PathOptions) {
    this.url = url;
    this.priority = priority;
    this.opts = opts;
  }

  get connected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  get healthy(): boolean {
    if (!this.connected) return false;
    // Freshly opened: trust the open until the first probe window elapses.
    if (this.lastPongAt === 0) return true;
    return Date.now() - this.lastPongAt <= this.opts.healthTimeoutMs;
  }

  start(): void {
    this.stopped = false;
    this.open();
    this.probe = setInterval(() => this.tick(), this.opts.healthIntervalMs);
  }

  stop(): void {
    this.stopped = true;
    if (this.timer) clearTimeout(this.timer);
    if (this.probe) clearInterval(this.probe);
    this.timer = null;
    this.probe = null;
    this.ws?.close();
    this.ws = null;
  }

  send(text: string): boolean {
    if (!this.connected) return false;
    try {
      this.ws?.send(text);
      this.sent += 1;
      return true;
    } catch (err) {
      this.lastError = String(err);
      return false;
    }
  }

  report(active: boolean): CorePathReport {
    return {
      url: this.url,
      priority: this.priority,
      connected: this.connected,
      healthy: this.healthy,
      active,
      latencyMs: this.latencyMs,
      sent: this.sent,
      reconnects: this.reconnects,
      lastPongAgoMs: this.lastPongAt === 0 ? null : Date.now() - this.lastPongAt,
      lastError: this.lastError,
    };
  }

  private tick(): void {
    if (this.connected) {
      this.pingSentAt = Date.now();
      try {
        this.ws?.ping();
      } catch (err) {
        this.lastError = String(err);
      }
    }
    const healthy = this.healthy;
    if (healthy !== this.wasHealthy) {
      this.wasHealthy = healthy;
      if (healthy) this.opts.onUp(this);
      else this.opts.onDown(this);
    }
  }

  private open(): void {
    if (this.stopped) return;
    const ws = new WebSocket(this.url);
    this.ws = ws;

    ws.on('open', () => {
      this.lastError = null;
      this.lastPongAt = Date.now();
      this.settled = true;
      console.log(`[proxy] core path #${this.priority} connected: ${this.url}`);
      if (!this.wasHealthy) {
        this.wasHealthy = true;
        this.opts.onUp(this);
      }
    });

    ws.on('pong', () => {
      this.lastPongAt = Date.now();
      if (this.pingSentAt > 0) this.latencyMs = this.lastPongAt - this.pingSentAt;
    });

    ws.on('message', (data) => this.opts.onMessage(data.toString()));

    ws.on('error', (err) => {
      this.lastError = err.message;
    });

    ws.on('close', () => {
      if (this.ws === ws) this.ws = null;
      this.latencyMs = null;
      this.lastPongAt = 0;
      this.settled = true;
      if (this.wasHealthy) {
        this.wasHealthy = false;
        this.opts.onDown(this);
      }
      if (this.stopped) return;
      this.reconnects += 1;
      this.timer = setTimeout(() => this.open(), this.opts.reconnectMs);
    });
  }
}

export interface ManagerOptions {
  endpoints: string[];
  failback: boolean;
  reconnectMs: number;
  healthIntervalMs: number;
  healthTimeoutMs: number;
  maxQueue: number;
  maxAgeMs: number;
  /** At boot, give better-priority paths this long to finish connecting before
   *  a standby that happened to open first is promoted. */
  startupGraceMs: number;
  onMessage: (text: string) => void;
  /** Fired whenever the active path changes, so the caller can push LinkStatus. */
  onActiveChange?: (event: FailoverEvent) => void;
}

interface Held {
  text: string;
  at: number;
}

export class CorePathManager {
  private opts: ManagerOptions;
  private paths: CorePath[] = [];
  private active: CorePath | null = null;
  private queue: Held[] = [];
  private droppedOverflow = 0;
  private droppedStale = 0;
  private forwarded = 0;
  private failovers = 0;
  private failbacks = 0;
  private graceUntil = 0;
  private history: FailoverEvent[] = [];

  constructor(opts: ManagerOptions) {
    this.opts = opts;
    this.paths = opts.endpoints.map(
      (url, i) =>
        new CorePath(url, i, {
          reconnectMs: opts.reconnectMs,
          healthIntervalMs: opts.healthIntervalMs,
          healthTimeoutMs: opts.healthTimeoutMs,
          onMessage: opts.onMessage,
          onUp: (p) => this.onUp(p),
          onDown: (p) => this.onDown(p),
        }),
    );
  }

  start(): void {
    const list = this.paths.map((p) => `#${p.priority} ${p.url}`).join(' | ');
    console.log(`[proxy] ${this.paths.length} core path(s), priority order: ${list}`);
    this.graceUntil = Date.now() + this.opts.startupGraceMs;
    for (const p of this.paths) p.start();
    setTimeout(() => {
      this.graceUntil = 0;
      this.select('startup grace elapsed');
    }, this.opts.startupGraceMs).unref();
  }

  stop(): void {
    for (const p of this.paths) p.stop();
  }

  get activeUrl(): string | null {
    return this.active ? this.active.url : null;
  }

  get activeLatencyMs(): number | null {
    return this.active ? this.active.latencyMs : null;
  }

  get connected(): boolean {
    return this.active !== null && this.active.healthy;
  }

  /** Never blocks. Buffers when no path is healthy, drop-oldest past the cap. */
  send(text: string): { ok: boolean; via: string | null } {
    this.expire();
    // Held frames go out before the new one, and the drain is attempted on every
    // send (not only on a failover) so a queue can never outlive the outage that
    // created it.
    if (this.active && this.active.healthy && this.queue.length > 0) this.drain();
    if (this.active && this.active.healthy && this.queue.length === 0) {
      if (this.active.send(text)) {
        this.forwarded += 1;
        return { ok: true, via: this.active.url };
      }
    }
    if (this.queue.length >= this.opts.maxQueue) {
      this.queue.shift();
      this.droppedOverflow += 1;
      if (this.droppedOverflow % 50 === 1) {
        console.warn(
          `[proxy] no healthy core path, queue full (${this.opts.maxQueue}) — dropped oldest, ` +
            `${this.droppedOverflow} total`,
        );
      }
    }
    this.queue.push({ text, at: Date.now() });
    return { ok: false, via: null };
  }

  counters() {
    return {
      forwarded: this.forwarded,
      queued: this.queue.length,
      droppedOverflow: this.droppedOverflow,
      droppedStale: this.droppedStale,
      failovers: this.failovers,
      failbacks: this.failbacks,
      activeUrl: this.activeUrl,
      paths: this.paths.map((p) => p.report(p === this.active)),
      history: this.history.slice(-10),
    };
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
      console.warn(`[proxy] dropped ${removed} frame(s) aged out while all paths were down`);
    }
  }

  private drain(): void {
    this.expire();
    while (this.queue.length > 0 && this.active && this.active.healthy) {
      // Consume only on a successful write: a frame must not be lost because the
      // path we were draining onto died halfway through the queue.
      if (!this.active.send(this.queue[0].text)) break;
      this.queue.shift();
      this.forwarded += 1;
    }
  }

  private onUp(path: CorePath): void {
    console.log(`[proxy] core path #${path.priority} healthy: ${path.url}`);
    this.select(`path #${path.priority} became healthy`);
  }

  private onDown(path: CorePath): void {
    const why = path.lastError === null ? 'no response' : path.lastError;
    console.warn(`[proxy] core path #${path.priority} UNHEALTHY: ${path.url} (${why})`);
    this.select(`path #${path.priority} went unhealthy`);
  }

  /**
   * Pick the path traffic leaves through.
   *   failback=true  — always the healthy path with the best priority.
   *   failback=false — keep the current one while it is healthy (sticky).
   */
  private select(reason: string): void {
    const healthy = this.paths.filter((p) => p.healthy).sort((a, b) => a.priority - b.priority);
    let next: CorePath | null;
    if (!this.opts.failback && this.active && this.active.healthy) next = this.active;
    else next = healthy.length > 0 ? healthy[0] : null;

    if (next === this.active) return;

    // Boot ordering: don't promote a standby just because it opened first while a
    // better path is still dialling.
    const candidate = next;
    if (Date.now() < this.graceUntil && candidate !== null) {
      const better = this.paths.some((p) => p.priority < candidate.priority && !p.settled);
      if (better) return;
    }

    const from = this.active === null ? null : this.active.url;
    const to = next === null ? null : next.url;
    const isFailback = this.active !== null && next !== null && next.priority < this.active.priority;
    this.active = next;
    if (from !== null) {
      if (isFailback) this.failbacks += 1;
      else this.failovers += 1;
    }

    const event: FailoverEvent = { at: Date.now(), from, to, reason };
    this.history.push(event);
    if (to === null) {
      console.error(`[proxy] NO healthy core path (${reason}) — buffering ${this.queue.length}`);
    } else if (from === null) {
      console.log(`[proxy] active core path = ${to} (${reason})`);
    } else if (isFailback) {
      console.warn(`[proxy] FAILBACK ${from} -> ${to} (${reason})`);
    } else {
      console.warn(`[proxy] FAILOVER ${from} -> ${to} (${reason})`);
    }
    if (this.opts.onActiveChange) this.opts.onActiveChange(event);
    this.drain();
  }
}
