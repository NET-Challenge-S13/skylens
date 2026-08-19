// 작업 오케스트레이터 — 딜레이 패턴이 사는 곳 (COMPONENTS.md §3.4-3).
//
// 딜레이 패턴의 스케줄 결정권은 코어에만 있다. 클라이언트는 도착한 것을 받을 뿐이다.
// The whole pattern is four rules, and this file is the only place they exist:
//
//   R1  구간이 닫히면 그 구간의 다음 수준이 즉시 나간다.
//       A segment closes when the drone LEFT it (ingest.ts). Its level-1 job is
//       dispatched at once — it never waits behind a refinement, so the
//       commander sees the place he just overflew within one job time.
//
//   R2  정제는 뒤따르고, 다음 구간의 첫 수준과 겹친다.
//       A delivered level L < top immediately queues L+1. With reconConcurrency
//       ≥ 2 that refinement is still in flight when segment k+1 closes and fires
//       its own level 1 — the overlap the report calls 지연 배치.
//
//   R3  추월당한 수준은 실행하지 않고 버린다.
//       A queued job whose level is already delivered for that segment is dead
//       weight: rendering it would be a downgrade. It is dropped, never run.
//
//   R4  첫 복원 잡이 좌표계를 정하고, 이후 모든 잡이 그것을 물려받는다.
//       protocol §7 / API.md §3: gsplat renormalizes per camera set, so the first
//       job's frame id is forced onto every later job. Until that id exists the
//       lane runs one job wide — two "first" jobs would establish two frames.
//
// Detection runs in its own lane so a slow perception job can never stall the
// reconstruction ladder.

import type {
  DetectJobRequest,
  DetectionResult,
  ReconJobRequest,
  SplatChunk,
} from '../../shared/protocol.ts';
import type { QueuedJob, SegmentRecord } from './types.ts';
import type { LadderLevel } from './ladder.ts';
import { rung, topLevel } from './ladder.ts';
import { JobFailed, ModelUnreachable, type ModelClient } from './modelClient.ts';
import type { Store } from './store.ts';

export interface OrchestratorOptions {
  store: Store;
  model: ModelClient;
  ladder: LadderLevel[];
  /** Recon jobs in flight at once. ≥2 is what makes R2 overlap. */
  reconConcurrency: number;
  detectConcurrency: number;
  detect: boolean;
  retryMs: number;
  maxAttempts: number;
  onChunk: (chunk: SplatChunk) => void;
  onDetection: (det: DetectionResult) => void;
}

export interface OrchestratorCounters {
  queued: number;
  reconInFlight: number;
  detectInFlight: number;
  reconQueued: number;
  detectQueued: number;
  dispatched: number;
  dropped: number;
  done: number;
  failed: number;
  retries: number;
  anchorFrame: string | null;
  /** Job ids running right now, for /health. */
  running: string[];
}

export class Orchestrator {
  private opts: OrchestratorOptions;
  private queue: QueuedJob[] = [];
  private running = new Map<string, QueuedJob>();
  private reconInFlight = 0;
  private detectInFlight = 0;
  private jobSeq = 0;
  private dispatched = 0;
  private dropped = 0;
  private done = 0;
  private failed = 0;
  private retries = 0;
  private wake: ReturnType<typeof setTimeout> | null = null;
  private stopped = false;

  constructor(opts: OrchestratorOptions) {
    this.opts = opts;
  }

  get top(): number {
    return topLevel(this.opts.ladder);
  }

  counters(): OrchestratorCounters {
    return {
      queued: this.queue.length,
      reconInFlight: this.reconInFlight,
      detectInFlight: this.detectInFlight,
      reconQueued: this.queue.filter((j) => j.kind === 'recon').length,
      detectQueued: this.queue.filter((j) => j.kind === 'detect').length,
      dispatched: this.dispatched,
      dropped: this.dropped,
      done: this.done,
      failed: this.failed,
      retries: this.retries,
      anchorFrame: this.opts.store.anchorFrame,
      running: [...this.running.keys()],
    };
  }

  /** Pending or running work for this segment, so a re-close is idempotent. */
  private busy(segment: number, kind: 'recon' | 'detect'): boolean {
    if (this.queue.some((j) => j.segment === segment && j.kind === kind)) return true;
    for (const j of this.running.values()) if (j.segment === segment && j.kind === kind) return true;
    return false;
  }

  /**
   * R1. The drone left this segment: take it one rung up the ladder.
   *
   * The rung asked for is `deliveredLevel + 1`, which is what makes a 왕복 route
   * behave: the first close asks for level 1, a later pass over the same place
   * asks for whatever it has not got yet, and a segment already at the top rung
   * asks for nothing at all.
   */
  segmentClosed(seg: SegmentRecord): void {
    if (this.stopped) return;
    if (this.opts.detect && seg.detectGeneration !== seg.generation && seg.sources.length > 0) {
      seg.detectGeneration = seg.generation;
      if (!this.busy(seg.index, 'detect')) this.enqueue(seg, 'detect', 0);
    }
    const next = seg.deliveredLevel + 1;
    if (next > this.top) return;
    if (this.busy(seg.index, 'recon')) return;
    this.enqueue(seg, 'recon', next);
  }

  start(): void {
    this.stopped = false;
    this.pump();
  }

  stop(): void {
    this.stopped = true;
    if (this.wake) clearTimeout(this.wake);
    this.wake = null;
  }

  // -------------------------------------------------------------------------

  private enqueue(seg: SegmentRecord, kind: 'recon' | 'detect', level: number): void {
    this.jobSeq += 1;
    const step = kind === 'recon' ? rung(this.opts.ladder, level) : null;
    const job: QueuedJob = {
      id: `${kind}-s${seg.index}-l${level}-#${this.jobSeq}`,
      kind,
      segment: seg.index,
      level,
      steps: step?.steps ?? 0,
      label: step?.label ?? '탐지',
      generation: seg.generation,
      enqueuedAt: Date.now(),
      attempts: 0,
      notBefore: 0,
    };
    this.queue.push(job);
    if (kind === 'recon') seg.state = 'queued';
    console.log(
      `[core] queue ${job.id}` +
        (kind === 'recon' ? ` steps=${job.steps} "${job.label}"` : '') +
        ` (${this.queue.length} waiting)`,
    );
    this.pump();
  }

  /**
   * Priority, high to low:
   *   1. a segment's FIRST level          (R1 — never made to wait)
   *   2. lower rungs before higher rungs  (cheap wins land first)
   *   3. earlier segments                 (the commander reads front to back)
   */
  private rank(job: QueuedJob): number[] {
    return [job.level === 1 ? 0 : 1, job.level, job.segment, job.enqueuedAt];
  }

  private better(a: QueuedJob, b: QueuedJob): boolean {
    const ra = this.rank(a);
    const rb = this.rank(b);
    for (let i = 0; i < ra.length; i += 1) {
      if (ra[i] !== rb[i]) return ra[i] < rb[i];
    }
    return false;
  }

  private pump(): void {
    if (this.stopped) return;
    for (;;) {
      this.sweep();
      const now = Date.now();
      let pick: QueuedJob | null = null;
      for (const job of this.queue) {
        if (job.notBefore > now) continue;
        if (!this.hasCapacity(job)) continue;
        if (pick === null || this.better(job, pick)) pick = job;
      }
      if (pick === null) break;
      this.queue.splice(this.queue.indexOf(pick), 1);
      void this.run(pick);
    }
    this.rearm();
  }

  /** R3 — drop levels that were overtaken while they sat in the queue. */
  private sweep(): void {
    this.queue = this.queue.filter((job) => {
      if (job.kind !== 'recon') return true;
      const seg = this.opts.store.segments.get(job.segment);
      if (seg === undefined || job.level > seg.deliveredLevel) return true;
      this.dropped += 1;
      this.opts.store.counters.reconJobsDropped += 1;
      console.log(
        `[core] drop ${job.id}: segment ${job.segment} is already at level ${seg.deliveredLevel}`,
      );
      return false;
    });
  }

  private hasCapacity(job: QueuedJob): boolean {
    if (job.kind === 'detect') return this.detectInFlight < this.opts.detectConcurrency;
    // R4: the frame-establishing job runs alone.
    if (this.opts.store.anchorFrame === null && this.reconInFlight > 0) return false;
    // R1: a first level is exempt from the cap; refinements are not.
    return job.level === 1 || this.reconInFlight < this.opts.reconConcurrency;
  }

  private rearm(): void {
    if (this.wake) clearTimeout(this.wake);
    this.wake = null;
    if (this.stopped || this.queue.length === 0) return;
    const now = Date.now();
    let soonest = Number.POSITIVE_INFINITY;
    for (const job of this.queue) soonest = Math.min(soonest, Math.max(job.notBefore - now, 0));
    if (!Number.isFinite(soonest)) return;
    // Backed-off jobs need a wake-up; jobs blocked purely on capacity are pumped
    // again when something finishes, but a slow re-check costs nothing.
    this.wake = setTimeout(() => {
      this.wake = null;
      this.pump();
    }, Math.max(soonest, 250));
  }

  private async run(job: QueuedJob): Promise<void> {
    this.running.set(job.id, job);
    this.dispatched += 1;
    if (job.kind === 'recon') this.reconInFlight += 1;
    else this.detectInFlight += 1;

    const seg = this.opts.store.segment(job.segment);
    if (job.kind === 'recon') seg.state = 'processing';
    const waited = Date.now() - job.enqueuedAt;
    console.log(
      `[core] dispatch ${job.id} after ${waited} ms ` +
        `(recon ${this.reconInFlight}/${this.opts.reconConcurrency} in flight, ` +
        `${this.queue.length} waiting)`,
    );

    try {
      if (job.kind === 'recon') await this.recon(job, seg);
      else await this.detectJob(job, seg);
    } catch (err) {
      this.onError(job, err);
    } finally {
      this.running.delete(job.id);
      if (job.kind === 'recon') this.reconInFlight -= 1;
      else this.detectInFlight -= 1;
      this.pump();
    }
  }

  private async recon(job: QueuedJob, seg: SegmentRecord): Promise<void> {
    const store = this.opts.store;
    const req: ReconJobRequest = {
      segment: job.segment,
      sources: seg.sources.map((s) => ({ uri: s.uri, poses: s.poses })),
      steps: job.steps,
      anchorFrame: store.anchorFrame,
    };
    const started = Date.now();
    const result = await this.opts.model.runRecon(req);

    if (store.anchorFrame === null) {
      store.anchorFrame = result.anchorFrame;
      console.log(`[core] anchorFrame established by ${job.id}: ${result.anchorFrame}`);
    } else if (result.anchorFrame !== store.anchorFrame) {
      // Not fatal — the model adopts an unknown id after a restart (API.md §3) —
      // but it means two spaces could have been mixed, so it is said out loud.
      console.warn(
        `[core] ${job.id} came back on frame ${result.anchorFrame}, expected ${store.anchorFrame}`,
      );
    }

    const chunk: SplatChunk = {
      kind: 'splat-chunk',
      id: `seg${job.segment}-l${job.level}`,
      segment: job.segment,
      level: job.level,
      steps: job.steps,
      label: job.label,
      final: job.level >= this.top,
      url: result.url,
      bytes: result.bytes,
      align: result.align,
    };
    store.putChunk(chunk);
    store.counters.reconJobsDone += 1;
    this.done += 1;
    console.log(
      `[core] segment ${job.segment} level ${job.level}/${this.top} ` +
        `(${job.steps} steps, ${(result.bytes / 1024).toFixed(0)} KiB, ${Date.now() - started} ms) ` +
        `→ viewers${chunk.final ? ' [final]' : ''}`,
    );
    this.opts.onChunk(chunk);

    // R2 — the refinement is queued the instant the level lands, so it is in
    // flight while the next segment is still being flown.
    if (job.level < this.top) this.enqueue(seg, 'recon', job.level + 1);
  }

  private async detectJob(job: QueuedJob, seg: SegmentRecord): Promise<void> {
    const req: DetectJobRequest = {
      segment: job.segment,
      sources: seg.sources.map((s) => ({ uri: s.uri, poses: s.poses })),
    };
    const result = await this.opts.model.runDetect(req);
    let fresh = 0;
    for (const det of result.detections) {
      if (this.opts.store.addDetection(det)) {
        fresh += 1;
        this.opts.onDetection(det);
      }
    }
    this.done += 1;
    console.log(`[core] segment ${job.segment} detection: ${fresh} new marker(s)`);
  }

  /**
   * Degradation, not failure. The model API being absent is a normal state:
   * transport errors go back in the queue with a backoff, and everything else in
   * the core (telemetry, mission, viewer fan-out) keeps running throughout.
   */
  private onError(job: QueuedJob, err: unknown): void {
    const seg = this.opts.store.segments.get(job.segment);
    if (err instanceof ModelUnreachable) {
      job.attempts += 1;
      if (job.attempts < this.opts.maxAttempts) {
        job.notBefore = Date.now() + this.opts.retryMs;
        this.queue.push(job);
        this.retries += 1;
        if (seg && job.kind === 'recon') seg.state = 'queued';
        console.warn(
          `[core] ${job.id} deferred (attempt ${job.attempts}/${this.opts.maxAttempts}): ` +
            `${String(err)}`,
        );
        return;
      }
      this.failed += 1;
      this.opts.store.counters.reconJobsFailed += job.kind === 'recon' ? 1 : 0;
      console.error(`[core] ${job.id} abandoned after ${job.attempts} attempts: ${String(err)}`);
      return;
    }

    // The job RAN and failed (e.g. live mode without the recon deps wired). No
    // amount of retrying fixes that, so the rung is left where it is; a later
    // pass over the same place will ask for it again.
    this.failed += 1;
    if (job.kind === 'recon') this.opts.store.counters.reconJobsFailed += 1;
    if (seg && job.kind === 'recon') seg.state = seg.deliveredLevel > 0 ? 'settled' : 'queued';
    const why = err instanceof JobFailed ? err.message : String(err);
    console.error(`[core] ${job.id} failed: ${why}`);
  }
}
