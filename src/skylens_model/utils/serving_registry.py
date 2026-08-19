"""In-memory job registry with a single background worker.

Deliberately has NO database: COMPONENTS.md §3.4/§7 keeps this stage in-memory,
so jobs disappear when the process exits. The core re-issues anything it still
cares about after a restart.

Jobs run STRICTLY ONE AT A TIME. Both real workloads (gsplat training, UNet
inference) want the whole GPU, so overlapping them would only trade throughput
for latency. That also makes ``queued`` in JobAccepted an honest number: it is
how many jobs are ahead of yours.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from .serving_schemas import DetectJobResult, JobState, JobStatus, ReconJobResult

log = logging.getLogger(__name__)

JobKind = Literal["recon", "detect"]
AnyResult = ReconJobResult | DetectJobResult

#: A runner gets a progress sink (0..1) and returns the finished result.
Runner = Callable[[Callable[[float], None]], Awaitable[AnyResult]]


@dataclass
class Job:
    id: str
    kind: JobKind
    state: JobState = "queued"
    progress: float = 0.0
    result: AnyResult | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def status(self) -> JobStatus:
        return JobStatus(
            jobId=self.id,
            state=self.state,
            progress=round(self.progress, 4),
            result=self.result,
            error=self.error,
        )


class JobRegistry:
    """Accepts jobs, runs them one at a time, remembers the recent ones."""

    def __init__(self, history: int = 512) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._runners: dict[str, Runner] = {}
        self._worker: asyncio.Task[None] | None = None
        self._history = history

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run_forever(), name="skylens-job-worker")

    async def stop(self) -> None:
        worker, self._worker = self._worker, None
        if worker is None:
            return
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

    # -- api ---------------------------------------------------------------

    def submit(self, kind: JobKind, runner: Runner) -> tuple[Job, int]:
        """Register a job and hand back its queue position at accept time."""
        ahead = sum(1 for j in self._jobs.values() if j.state in ("queued", "running"))
        job = Job(id=f"{kind}-{uuid.uuid4().hex[:12]}", kind=kind)
        self._jobs[job.id] = job
        self._order.append(job.id)
        self._runners[job.id] = runner
        self._queue.put_nowait(job.id)
        self._evict()
        log.info("job %s accepted (%s), %d ahead", job.id, kind, ahead)
        return job, ahead

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def counts(self) -> dict[str, int]:
        out = {"queued": 0, "running": 0, "done": 0, "failed": 0}
        for job in self._jobs.values():
            out[job.state] += 1
        return out

    # -- worker ------------------------------------------------------------

    async def _run_forever(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            runner = self._runners.pop(job_id, None)
            if job is None or runner is None:  # evicted before it ever ran
                continue
            await self._run_one(job, runner)

    async def _run_one(self, job: Job, runner: Runner) -> None:
        job.state = "running"
        job.started_at = time.time()

        def report(value: float) -> None:
            job.progress = min(1.0, max(0.0, float(value)))

        try:
            job.result = await runner(report)
            job.progress = 1.0
            job.state = "done"
            log.info("job %s done in %.2fs", job.id, time.time() - (job.started_at or 0))
        except asyncio.CancelledError:
            job.state = "failed"
            job.error = "cancelled"
            raise
        except Exception as exc:  # a failed job must not take the worker down
            job.state = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            log.exception("job %s failed", job.id)
        finally:
            job.finished_at = time.time()

    # -- housekeeping ------------------------------------------------------

    def _evict(self) -> None:
        """Drop the oldest FINISHED jobs once the table grows past ``history``."""
        while len(self._order) > self._history:
            for i, job_id in enumerate(self._order):
                job = self._jobs.get(job_id)
                if job is None or job.state in ("done", "failed"):
                    self._order.pop(i)
                    self._jobs.pop(job_id, None)
                    self._runners.pop(job_id, None)
                    break
            else:
                return  # everything still pending; let the table grow
