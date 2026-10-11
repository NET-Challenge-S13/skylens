// Per-segment REPLACEMENT for anything the delay pattern delivers as files.
//
// Each segment holds at most one displayed object. A message for a higher
// level loads in the background and, once ready, swaps in for the old object
// in the same call: the new one is mounted, then the old one released, so the
// segment never shows both and never shows neither (no flicker). Anything at
// or below what the segment already shows or already wants is stale — the
// relay replays its cache on reconnect, and a reordered or repeated message
// must not downgrade a segment.
//
// A higher level arriving while a lower one is still loading cancels the lower
// load; its result, if it lands anyway, is released without being shown.
//
// Nothing here knows what is being loaded. The point layer plugs in a PLY
// loader; the incremental mesh (S4) is meant to plug in its own and get the
// same ordering guarantees.

export interface SlotMessage {
  segment: number;
  level: number;
}

export interface SlotHooks<M extends SlotMessage, T> {
  /** Fetch and build the object for this message. Honour `signal`. */
  load(msg: M, signal: AbortSignal): Promise<T>;
  /** Put a freshly loaded object on screen for its segment. */
  mount(msg: M, obj: T): void;
  /** Take an object off screen (if it is on) and free it. */
  release(obj: T): void;
  /** A load failed. The segment keeps whatever it was showing. */
  onError?(msg: M, err: unknown): void;
}

export interface SlotState {
  segment: number;
  /** Level on screen, 0 = nothing yet. */
  shown: number;
  /** Level being loaded, 0 = none. */
  loading: number;
  /** Highest level offered, 0 = none. */
  wanted: number;
}

export class SegmentSlots<M extends SlotMessage, T> {
  private readonly hooks: SlotHooks<M, T>;
  private readonly shown = new Map<number, { msg: M; obj: T }>();
  private readonly wanted = new Map<number, M>();
  private readonly inflight = new Map<number, { msg: M; ctrl: AbortController }>();
  private readonly failed = new Map<number, M>();
  private active: boolean;
  private stale = 0;
  private cancelled = 0;

  constructor(hooks: SlotHooks<M, T>, active = true) {
    this.hooks = hooks;
    this.active = active;
  }

  /** Offer a delivered message. 'stale' = the segment already has this level
   *  or better (shown, loading or queued), and nothing changes. */
  offer(msg: M): 'accepted' | 'stale' {
    const seg = msg.segment;
    const best = Math.max(
      this.shown.get(seg)?.msg.level ?? 0,
      this.wanted.get(seg)?.level ?? 0,
      this.inflight.get(seg)?.msg.level ?? 0,
    );
    if (msg.level <= best) {
      this.stale += 1;
      return 'stale';
    }
    this.wanted.set(seg, msg);
    this.failed.delete(seg);
    this.pump(seg);
    return 'accepted';
  }

  /** While inactive, offers are remembered but nothing is fetched (loads in
   *  flight are cancelled and queued again); turning it on loads the highest
   *  offered level of every segment. */
  setActive(on: boolean): void {
    if (this.active === on) return;
    this.active = on;
    if (on) {
      for (const seg of [...this.wanted.keys()]) this.pump(seg);
      return;
    }
    for (const [seg, run] of [...this.inflight]) {
      this.inflight.delete(seg);
      run.ctrl.abort();
      this.cancelled += 1;
      const queued = this.wanted.get(seg);
      if (!queued || queued.level < run.msg.level) this.wanted.set(seg, run.msg);
    }
  }

  get isActive(): boolean {
    return this.active;
  }

  private pump(seg: number): void {
    if (!this.active) return;
    const want = this.wanted.get(seg);
    if (!want) return;
    const showing = this.shown.get(seg)?.msg.level ?? 0;
    if (want.level <= showing) {
      this.wanted.delete(seg);
      return;
    }
    const running = this.inflight.get(seg);
    if (running) {
      if (running.msg.level >= want.level) return;
      running.ctrl.abort();
      this.cancelled += 1;
    }
    this.wanted.delete(seg);
    const ctrl = new AbortController();
    this.inflight.set(seg, { msg: want, ctrl });
    this.hooks.load(want, ctrl.signal).then(
      (obj) => {
        const current = this.inflight.get(seg);
        if (current?.ctrl !== ctrl) {
          // Superseded while loading: never shown.
          this.hooks.release(obj);
          return;
        }
        this.inflight.delete(seg);
        const prev = this.shown.get(seg);
        if (prev && prev.msg.level >= want.level) {
          this.hooks.release(obj);
        } else {
          this.hooks.mount(want, obj);
          this.shown.set(seg, { msg: want, obj });
          if (prev) this.hooks.release(prev.obj);
        }
        this.pump(seg);
      },
      (err: unknown) => {
        if (this.inflight.get(seg)?.ctrl === ctrl) this.inflight.delete(seg);
        if (ctrl.signal.aborted) {
          this.pump(seg);
          return;
        }
        this.failed.set(seg, want);
        this.hooks.onError?.(want, err);
        // A higher level may have been offered while this one was failing.
        this.pump(seg);
      },
    );
  }

  /** What is on screen, segment ascending. */
  shownEntries(): Array<{ msg: M; obj: T }> {
    return [...this.shown.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v);
  }

  states(): SlotState[] {
    const segs = new Set([
      ...this.shown.keys(),
      ...this.wanted.keys(),
      ...this.inflight.keys(),
      ...this.failed.keys(),
    ]);
    return [...segs]
      .sort((a, b) => a - b)
      .map((segment) => ({
        segment,
        shown: this.shown.get(segment)?.msg.level ?? 0,
        loading: this.inflight.get(segment)?.msg.level ?? 0,
        wanted: this.wanted.get(segment)?.level ?? 0,
      }));
  }

  get counters(): { stale: number; cancelled: number; failed: number[] } {
    return { stale: this.stale, cancelled: this.cancelled, failed: [...this.failed.keys()] };
  }

  /** Drop everything (layer teardown). */
  clear(): void {
    for (const { ctrl } of this.inflight.values()) ctrl.abort();
    this.inflight.clear();
    this.wanted.clear();
    for (const { obj } of this.shown.values()) this.hooks.release(obj);
    this.shown.clear();
  }
}
