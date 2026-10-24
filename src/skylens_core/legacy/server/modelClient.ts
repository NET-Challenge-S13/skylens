// The only request/response surface in the system (protocol §7).
//
//   POST /recon/jobs   -> JobAccepted
//   POST /detect/jobs  -> JobAccepted
//   GET  /jobs/{id}    -> JobStatus   (polled until done | failed)
//
// skylens_model may simply not be running. That is a normal state here, not an
// exception: `reachable` goes false, jobs stay queued and are retried, and every
// other part of the core (telemetry, mission, viewer fan-out) keeps working.

import type {
  DetectJobRequest,
  DetectJobResult,
  JobAccepted,
  JobStatus,
  ReconJobRequest,
  ReconJobResult,
} from '../../shared/protocol.ts';

/** Thrown when the model API could not be reached or answered malformed —
 *  retryable. A job that ran and FAILED throws JobFailed instead. */
export class ModelUnreachable extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ModelUnreachable';
  }
}

export class JobFailed extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'JobFailed';
  }
}

export interface ModelClientOptions {
  baseUrl: string;
  pollMs: number;
  jobTimeoutMs: number;
}

export interface ModelHealth {
  url: string;
  reachable: boolean;
  lastOkAt: number | null;
  lastError: string | null;
  submitted: number;
  completed: number;
  failed: number;
  inFlight: number;
}

export class ModelClient {
  private opts: ModelClientOptions;
  private reachable = false;
  private lastOkAt: number | null = null;
  private lastError: string | null = null;
  private submitted = 0;
  private completed = 0;
  private failed = 0;
  private inFlight = 0;
  private warned = false;

  constructor(opts: ModelClientOptions) {
    this.opts = opts;
  }

  health(): ModelHealth {
    return {
      url: this.opts.baseUrl,
      reachable: this.reachable,
      lastOkAt: this.lastOkAt,
      lastError: this.lastError,
      submitted: this.submitted,
      completed: this.completed,
      failed: this.failed,
      inFlight: this.inFlight,
    };
  }

  get isReachable(): boolean {
    return this.reachable;
  }

  async runRecon(req: ReconJobRequest): Promise<ReconJobResult> {
    const result = await this.run('/recon/jobs', req);
    if (result === null || result.kind !== 'recon-result') {
      throw new JobFailed('model returned no recon-result');
    }
    return result;
  }

  async runDetect(req: DetectJobRequest): Promise<DetectJobResult> {
    const result = await this.run('/detect/jobs', req);
    if (result === null || result.kind !== 'detect-result') {
      throw new JobFailed('model returned no detect-result');
    }
    return result;
  }

  /** Cheap liveness probe for /health. Never throws. */
  async probe(): Promise<boolean> {
    try {
      const res = await fetch(`${this.opts.baseUrl}/health`, {
        signal: AbortSignal.timeout(2000),
      });
      this.markUp(res.ok);
      return res.ok;
    } catch (err) {
      this.markDown(String(err));
      return false;
    }
  }

  private async run(
    path: string,
    body: ReconJobRequest | DetectJobRequest,
  ): Promise<ReconJobResult | DetectJobResult | null> {
    const accepted = await this.submit(path, body);
    this.inFlight += 1;
    try {
      return await this.poll(accepted.jobId);
    } finally {
      this.inFlight -= 1;
    }
  }

  private async submit(
    path: string,
    body: ReconJobRequest | DetectJobRequest,
  ): Promise<JobAccepted> {
    let res: Response;
    try {
      res = await fetch(`${this.opts.baseUrl}${path}`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(10_000),
      });
    } catch (err) {
      this.markDown(String(err));
      throw new ModelUnreachable(`POST ${path}: ${String(err)}`);
    }
    if (!res.ok) {
      this.markDown(`HTTP ${res.status}`);
      throw new ModelUnreachable(`POST ${path}: HTTP ${res.status}`);
    }
    let accepted: JobAccepted;
    try {
      accepted = (await res.json()) as JobAccepted;
    } catch (err) {
      this.markDown(String(err));
      throw new ModelUnreachable(`POST ${path}: bad JSON (${String(err)})`);
    }
    if (typeof accepted?.jobId !== 'string') {
      this.markDown('no jobId in response');
      throw new ModelUnreachable(`POST ${path}: response has no jobId`);
    }
    this.markUp(true);
    this.submitted += 1;
    return accepted;
  }

  private async poll(jobId: string): Promise<ReconJobResult | DetectJobResult | null> {
    const deadline = Date.now() + this.opts.jobTimeoutMs;
    for (;;) {
      if (Date.now() > deadline) {
        this.failed += 1;
        throw new JobFailed(`job ${jobId} timed out after ${this.opts.jobTimeoutMs} ms`);
      }
      await sleep(this.opts.pollMs);
      let status: JobStatus;
      let vanished = false;
      try {
        const res = await fetch(`${this.opts.baseUrl}/jobs/${encodeURIComponent(jobId)}`, {
          signal: AbortSignal.timeout(10_000),
        });
        if (res.status === 404) {
          vanished = true;
          status = { jobId, state: 'failed', progress: 0, result: null, error: null };
        } else {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          status = (await res.json()) as JobStatus;
        }
      } catch (err) {
        // A blip mid-job is retryable: keep polling until the job deadline.
        this.markDown(String(err));
        continue;
      }
      this.markUp(true);
      if (vanished) {
        // The model API keeps jobs in memory (API.md §2.1), so a 404 means it
        // restarted under us and this id will never exist again. Polling it to
        // the deadline would burn the whole job timeout on a job nobody is
        // running; hand it back instead, and the orchestrator resubmits it.
        throw new ModelUnreachable(
          `GET /jobs/${jobId}: job is gone (HTTP 404) — the model API restarted`,
        );
      }
      if (status.state === 'done') {
        this.completed += 1;
        return status.result;
      }
      if (status.state === 'failed') {
        this.failed += 1;
        throw new JobFailed(status.error ?? `job ${jobId} failed`);
      }
    }
  }

  private markUp(ok: boolean): void {
    if (!ok) return;
    if (!this.reachable) console.log(`[core] model API reachable at ${this.opts.baseUrl}`);
    this.reachable = true;
    this.warned = false;
    this.lastOkAt = Date.now();
    this.lastError = null;
  }

  private markDown(err: string): void {
    this.lastError = err;
    if (this.reachable || !this.warned) {
      console.warn(
        `[core] model API unreachable at ${this.opts.baseUrl} (${err}) — ` +
          'jobs stay queued, everything else keeps running',
      );
      this.warned = true;
    }
    this.reachable = false;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
