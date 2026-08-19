"""FastAPI entry point for the ``skylens_model`` component.

The compute API the core talks to. It is the only request/response surface in
SkyLens -- everything else in the system is push (see res/docs/COMPONENTS.md
section 3.5, and src/shared/protocol.ts section 7 for the wire shapes).

    POST /recon/jobs   ReconJobRequest  -> JobAccepted
    POST /detect/jobs  DetectJobRequest -> JobAccepted
    GET  /jobs/{id}                     -> JobStatus
    GET  /health                        -> liveness + torch/CUDA availability

Run it::

    uv run uvicorn skylens_model.app:app --port 8100            # live mode
    SKYLENS_DEMO=1 uv run uvicorn skylens_model.app:app --port 8100

Jobs are held in memory only. A restart loses every job; that is the decision
in COMPONENTS.md section 7, not an oversight.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from . import __version__
from .utils.serving_config import settings
from .utils.serving_pipeline import FRAMES, run_detect, run_recon
from .utils.serving_registry import JobRegistry
from .utils.serving_schemas import (
    DetectJobRequest,
    Health,
    JobAccepted,
    JobStatus,
    ReconJobRequest,
)

log = logging.getLogger("skylens_model.app")

registry = JobRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = settings()
    log.info(
        "skylens_model api up (demo=%s, manifest=%s)",
        cfg.demo,
        cfg.manifest if cfg.demo else "-",
    )
    await registry.start()
    try:
        yield
    finally:
        await registry.stop()


app = FastAPI(
    title="SkyLens model API",
    version=__version__,
    summary="3DGS reconstruction + SkyLensNet perception jobs",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@app.post("/recon/jobs", response_model=JobAccepted, tags=["jobs"])
async def post_recon_job(req: ReconJobRequest) -> JobAccepted:
    """Queue a reconstruction of one capture segment at one refinement level.

    ``anchorFrame`` is null for the FIRST segment of a flight; the result comes
    back with the frame id this job established, and every later job must send
    that id back so the segments land in one space.
    """
    job, ahead = registry.submit("recon", lambda report: run_recon(req, report))
    return JobAccepted(jobId=job.id, queued=ahead)


@app.post("/detect/jobs", response_model=JobAccepted, tags=["jobs"])
async def post_detect_job(req: DetectJobRequest) -> JobAccepted:
    """Queue people / danger-zone detection over one capture segment."""
    job, ahead = registry.submit("detect", lambda report: run_detect(req, report))
    return JobAccepted(jobId=job.id, queued=ahead)


@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["jobs"])
async def get_job(job_id: str) -> JobStatus:
    """Poll a job. ``result`` is filled only once ``state`` is ``done``."""
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
    return job.status()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _gpu_probe() -> tuple[bool, bool, str | None]:
    """(torch importable, cuda usable, device name). Cached -- importing torch
    costs seconds and the answer cannot change while the process lives."""
    try:
        import torch
    except Exception:  # torch is optional for a demo-only host
        return False, False, None
    try:
        if torch.cuda.is_available():
            return True, True, torch.cuda.get_device_name(0)
    except Exception:
        log.exception("cuda probe failed")
    return True, False, None


@app.get("/health", response_model=Health, tags=["ops"])
async def get_health() -> Health:
    """Liveness plus what this process can actually compute."""
    torch_ok, cuda_ok, device = await asyncio.to_thread(_gpu_probe)
    counts = registry.counts()
    counts["frames"] = FRAMES.count()
    return Health(
        version=__version__,
        demo=settings().demo,
        torch=torch_ok,
        cuda=cuda_ok,
        device=device,
        jobs=counts,
    )
