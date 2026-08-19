"""Pydantic mirrors of the model API wire contract.

The single source of truth is ``src/shared/protocol.ts`` §7 (Model API).
Field names here are IDENTICAL to the TypeScript interfaces -- camelCase, no
aliasing -- so a payload serialized by the core deserializes here unchanged and
vice versa. If protocol.ts §7 moves, this file moves with it.

Types pulled in from the surrounding protocol sections because §7 references
them: ``Gps`` (§geo), ``DroneTelemetry`` (§2), ``SplatAlign`` (§4) and
``DetectionResult`` (§5).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Borrowed shapes (geo.ts / protocol.ts §2 §4 §5)
# ---------------------------------------------------------------------------


class Strict(BaseModel):
    """Reject unknown keys: a field the core renamed should fail loudly here."""

    model_config = ConfigDict(extra="forbid")


class Gps(Strict):
    lat: float
    lon: float
    alt: float


class DroneTelemetry(Strict):
    kind: Literal["telemetry"] = "telemetry"
    droneId: int
    #: Formation station the aircraft holds ("left" | "center" | "right").
    #: Mirrors DroneStation in shared/protocol.ts. Optional here so a fix from an
    #: older drone build still validates; the compute side does not use it.
    station: str | None = None
    gps: Gps
    headingDeg: float
    speed: float
    batteryPct: float
    t: int


class SplatAlign(Strict):
    """Where a reconstructed chunk lands in the shared frame."""

    anchor: Gps | None = None
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


IDENTITY_ALIGN = SplatAlign()


class DetectionResult(Strict):
    kind: Literal["detection"] = "detection"
    id: str
    category: Literal["person", "danger"]
    gps: Gps
    confidence: float
    label: str
    segment: int


class JobSource(Strict):
    """One video slice plus the poses covering it."""

    uri: str
    poses: list[DroneTelemetry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# §7 Model API
# ---------------------------------------------------------------------------


class ReconJobRequest(Strict):
    """POST /recon/jobs"""

    segment: int
    sources: list[JobSource] = Field(default_factory=list)
    steps: int
    #: The stretch of route this segment covers, sampled start to end. A
    #: reconstruction from images has no metric scale, so this is what it gets
    #: scaled against. Sent apart from the sources because a segment is closed by
    #: the aircraft leaving it, before its video has finished uploading.
    track: list[Gps] = Field(default_factory=list)
    #: Frame established by the FIRST segment, forced onto every later one.
    #: Null for the first job. See run_recon() for why this exists.
    anchorFrame: str | None = None


class DetectJobRequest(Strict):
    """POST /detect/jobs"""

    segment: int
    sources: list[JobSource] = Field(default_factory=list)


class JobAccepted(Strict):
    jobId: str
    #: Server-side queue position at accept time (0 = starts immediately).
    queued: int


JobState = Literal["queued", "running", "done", "failed"]


class ReconJobResult(Strict):
    kind: Literal["recon-result"] = "recon-result"
    segment: int
    steps: int
    url: str
    bytes: int
    align: SplatAlign = Field(default_factory=SplatAlign)
    #: Frame this job established; later jobs must be given it as anchorFrame.
    anchorFrame: str


class DetectJobResult(Strict):
    kind: Literal["detect-result"] = "detect-result"
    segment: int
    detections: list[DetectionResult] = Field(default_factory=list)


JobResult = Annotated[ReconJobResult | DetectJobResult, Field(discriminator="kind")]


class JobStatus(Strict):
    """GET /jobs/{job_id}"""

    jobId: str
    state: JobState
    progress: float
    result: JobResult | None = None
    error: str | None = None


class Health(Strict):
    """GET /health -- liveness plus what this process can actually compute."""

    status: Literal["ok"] = "ok"
    version: str
    demo: bool
    torch: bool
    cuda: bool
    device: str | None = None
    jobs: dict[str, int] = Field(default_factory=dict)
