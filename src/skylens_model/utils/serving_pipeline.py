"""The compute seam: one entry function per job type.

``run_recon()`` and ``run_detect()`` are the ONLY places the REST layer touches
compute. Swapping the demo path for the real pipeline is an edit inside these
two functions and nothing else.

What is real and what is not, as of this commit:

* demo mode (``SKYLENS_DEMO=1``) -- real in the sense that matters for the demo:
  it resolves an actual prebuilt ``segment x level`` PLY off disk through the
  manifest ``split_segments.py`` writes, checks the file exists, and returns its
  real byte size after a delay proportional to the requested step count. Nothing
  is trained.
* live mode -- an explicit, documented seam. 3DGS reconstruction is a
  minutes-long COLMAP + gsplat optimization needing the ``recon`` dependency
  group and a CUDA toolchain (models/skylens/INSTALL.md); it cannot run inside a
  request handler. SkyLensNet has no trained checkpoint and no 2D->3D projection
  layer yet (skylens_model/README.md, layer 2). Both raise
  ``PipelineUnavailable`` naming the exact missing piece rather than returning
  invented numbers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .geo import Enu, Gps, enu_to_gps
from .serving_config import settings
from .serving_schemas import (
    DetectionResult,
    DetectJobRequest,
    DetectJobResult,
    ReconJobRequest,
    ReconJobResult,
    SplatAlign,
)
from .serving_schemas import Gps as GpsModel

log = logging.getLogger(__name__)

Report = Callable[[float], None]


class PipelineUnavailable(RuntimeError):
    """The requested compute cannot run in this process, and why."""


# ---------------------------------------------------------------------------
# Anchor frames
# ---------------------------------------------------------------------------
#
# Why this exists (interim report III-1-ba, models/skylens/README.md section 5):
# the gsplat loader derives its scene normalization FROM THE CAMERA SET when
# normalize=True. Add the next segment's cameras and that transform is
# recomputed -- measured 87.8 degrees apart between a 75-frame and a 150-frame
# batch. Everything reconstructed earlier then lands in a rotated space.
#
# The fix is not an algorithm change: the FIRST job's transform is stored and
# forced onto every later job. That stored transform is what a frame id names.
# Contract: the first recon job sends anchorFrame=null and gets an id back;
# every later job of the same flight MUST send that id back.


@dataclass
class Frame:
    id: str
    origin_segment: int
    created_at: float = field(default_factory=time.time)
    #: Placement of chunks reconstructed in this frame. Demo assets were cut
    #: from one already-aligned scene, so identity is correct there. The real
    #: pipeline stores the pinned gsplat normalization transform here instead.
    align: SplatAlign = field(default_factory=SplatAlign)


class FrameStore:
    """In-memory frame table. Lost on restart, like everything else here."""

    def __init__(self) -> None:
        self._frames: dict[str, Frame] = {}

    def resolve(self, requested: str | None, segment: int) -> Frame:
        if requested is None:
            frame = Frame(id=f"frame-{uuid.uuid4().hex[:12]}", origin_segment=segment)
            self._frames[frame.id] = frame
            log.info("frame %s established by segment %d", frame.id, segment)
            return frame

        known = self._frames.get(requested)
        if known is not None:
            return known

        # Unknown id: the core outlived this process. Adopting it beats failing
        # the job -- the core is still using one id for the whole flight, and
        # that consistency is the point of the frame.
        adopted = Frame(id=requested, origin_segment=segment)
        self._frames[requested] = adopted
        log.warning("adopting unknown anchorFrame %s (server restart?)", requested)
        return adopted

    def count(self) -> int:
        return len(self._frames)


FRAMES = FrameStore()


# ---------------------------------------------------------------------------
# Demo assets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoAsset:
    level: int
    steps: int
    label: str
    url: str
    path: Path
    bytes: int


def load_manifest() -> dict:
    """Read the segment x level manifest written by split_segments.py."""
    path = settings().manifest
    if not path.exists():
        raise PipelineUnavailable(
            f"demo manifest missing: {path}. Build it with "
            "`uv run python -m skylens_model.models.skylens.split_segments "
            "res/static/demo/step*_light.ply --segments 4`."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_asset(segment: int, steps: int) -> DemoAsset:
    """Pick the prebuilt asset for this segment at the best level <= steps."""
    manifest = load_manifest()
    segments = manifest.get("segments") or []
    if not segments:
        raise PipelineUnavailable(f"demo manifest has no segments: {settings().manifest}")

    # The core may cut a flight into more segments than the demo has assets
    # for; wrap so the stream keeps flowing instead of erroring out mid-demo.
    index = segment % len(segments)
    if index != segment:
        log.warning("segment %d wrapped to demo segment %d", segment, index)
    levels = sorted(segments[index].get("levels") or [], key=lambda level: level["steps"])
    if not levels:
        raise PipelineUnavailable(f"demo segment {index} has no levels")

    # Highest level the requested step count has paid for; if the request asks
    # for fewer steps than the cheapest asset, give the cheapest.
    chosen = levels[0]
    for level in levels:
        if level["steps"] <= steps:
            chosen = level

    rel = str(chosen["url"])
    path = (settings().demo_root() / rel).resolve()
    if not path.exists():
        raise PipelineUnavailable(f"demo asset listed in manifest but missing on disk: {path}")

    return DemoAsset(
        level=int(chosen["level"]),
        steps=int(chosen["steps"]),
        label=str(chosen.get("label", "")),
        url=f"{settings().url_base}/{rel}",
        path=path,
        bytes=int(path.stat().st_size),
    )


def _demo_seconds(steps: int) -> float:
    cfg = settings()
    return max(cfg.min_seconds, min(cfg.max_seconds, steps * cfg.step_seconds))


async def _simulate(seconds: float, report: Report, ticks: int = 20) -> None:
    """Burn the simulated processing time, reporting progress as it goes."""
    for i in range(ticks):
        await asyncio.sleep(seconds / ticks)
        report((i + 1) / ticks)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


async def run_recon(req: ReconJobRequest, report: Report) -> ReconJobResult:
    """Reconstruct one capture segment at one refinement level."""
    frame = FRAMES.resolve(req.anchorFrame, req.segment)
    if settings().demo:
        return await _recon_demo(req, frame, report)
    return await _recon_live(req, frame, report)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _flight(fixes: list[GpsModel]) -> tuple[float, float, GpsModel] | None:
    """Ground distance flown, heading in radians east-of-north, and the middle fix."""
    if len(fixes) < 2:
        return None
    m_per_lat = 111_320.0
    m_per_lon = 111_320.0 * math.cos(math.radians(fixes[0].lat))
    flown = 0.0
    for a, b in zip(fixes, fixes[1:], strict=False):
        flown += math.hypot((b.lon - a.lon) * m_per_lon, (b.lat - a.lat) * m_per_lat)
    east = (fixes[-1].lon - fixes[0].lon) * m_per_lon
    north = (fixes[-1].lat - fixes[0].lat) * m_per_lat
    if flown <= 0.0 or (east == 0.0 and north == 0.0):
        return None
    return flown, math.atan2(east, north), fixes[len(fixes) // 2]


def _demo_align(req: DetectJobRequest | ReconJobRequest, manifest: dict, index: int) -> SplatAlign:
    """
    Put the prebuilt chunk on the ground it stands for.

    A reconstruction built from images has no metric scale — SfM recovers shape,
    not size — so the demo asset's units are whatever its solve settled on. The
    only measured thing in this system is the flight, and this job carries it:
    the poses the segment was filmed from. Measured on the demo asset, a chunk
    was drawn 4.7 m wide for a 40 m stretch of flight, so the board showed a
    scatter of specks with 35 m of empty ground between them.

    So: scale the piece until it covers what was flown, turn its principal axis
    to the heading that was flown, and put its middle where the aircraft was.

    The scale is ONE number for every chunk, taken from the median piece. The
    scene was cut into equal COUNTS of gaussians, not equal lengths, so the
    sparse tail of the scene is a long thin piece; scaling each chunk by its own
    extent would draw the same building at a different size in adjacent chunks.
    A uniform scale keeps the world self-consistent and lets the tail overhang,
    which is what a real reconstruction's tail does anyway.
    """
    # The route stretch the core measured, when it sent one. The poses are the
    # fallback: they only exist once a slice has been uploaded, and the first
    # refinement level is dispatched before that — a segment closes when the
    # aircraft leaves it, not when its video lands.
    track = list(getattr(req, "track", []) or [])
    flight = _flight(track or _segment_fixes(req))
    segments = manifest.get("segments") or []
    if flight is None or not segments:
        # Nothing measured to scale against (a hand-made request). Identity is
        # what the core's own route-anchoring expects to see.
        return SplatAlign()

    flown, heading, middle = flight
    # An asset written in metres (split_segments --length) is placed by a rigid
    # transform: it already IS the size it claims. Scaling it at draw time is
    # what the viewer cannot do convincingly — the splat centres move apart
    # while the surface does not follow, and the board renders a scene that
    # measures correctly and shows nothing.
    if float(manifest.get("metersPerUnit") or 0.0) > 0.0:
        scale = 1.0
    else:
        extents = [float(seg.get("extent") or 0.0) for seg in segments]
        typical = _median([e for e in extents if e > 0.0])
        if typical <= 0.0:
            # An older manifest, from before the splitter recorded piece sizes.
            log.warning("demo manifest has no segment extents — placing chunks unscaled")
            return SplatAlign()
        scale = flown / typical

    # Scene axes are x=East, y=Up, z=-North (shared/geo.ts). The asset's
    # principal axis is horizontal for a corridor sweep, so a yaw about Up is
    # the whole rotation: from the asset axis to the flown heading.
    axis = manifest.get("axis") or [1.0, 0.0, 0.0]
    asset_yaw = math.atan2(float(axis[0]), -float(axis[2]) if len(axis) > 2 else 0.0)
    yaw = heading - asset_yaw
    half = yaw / 2.0
    rotation = (0.0, math.sin(half), 0.0, math.cos(half))

    # Where the piece's own middle sits, once scaled and turned: subtract it, so
    # the piece straddles the anchor instead of hanging off it.
    cx, cy, cz = (float(v) for v in (segments[index].get("centroid") or [0.0, 0.0, 0.0]))
    sx, sy, sz = cx * scale, cy * scale, cz * scale
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    rx = cos_y * sx + sin_y * sz
    rz = -sin_y * sx + cos_y * sz
    position = (-rx, -sy, -rz)

    # Horizontal from the flight, height from the site datum. The middle fix is
    # the AIRCRAFT's altitude — anchoring there would hang the reconstruction 60
    # m up, at the height the camera flew rather than the ground it filmed.
    return SplatAlign(
        anchor=GpsModel(lat=middle.lat, lon=middle.lon, alt=settings().anchor.alt),
        position=position,
        rotation=rotation,
        scale=(scale, scale, scale),
    )


async def _recon_demo(req: ReconJobRequest, frame: Frame, report: Report) -> ReconJobResult:
    asset = resolve_asset(req.segment, req.steps)
    log.info(
        "recon(demo) segment %d steps %d -> level %d (%d steps) %s",
        req.segment,
        req.steps,
        asset.level,
        asset.steps,
        asset.url,
    )
    # Proportional to the REQUESTED steps, not the asset's: the delay pattern is
    # about how long a level makes the operator wait.
    await _simulate(_demo_seconds(req.steps), report)
    manifest = load_manifest()
    index = req.segment % len(manifest.get("segments") or [1])
    return ReconJobResult(
        segment=req.segment,
        steps=asset.steps,
        url=asset.url,
        bytes=asset.bytes,
        # frame.align is identity for the demo assets — they were cut from one
        # already-aligned scene — so what matters is putting the piece on the
        # ground the segment was flown over.
        align=_demo_align(req, manifest, index),
        anchorFrame=frame.id,
    )


async def _recon_live(req: ReconJobRequest, frame: Frame, report: Report) -> ReconJobResult:
    """SEAM: hand the segment to the real 3DGS pipeline.

    What goes here, in order (models/skylens/README.md section 1):

    1. decode ``req.sources[*].uri`` to frames, seeding camera poses from
       ``poses`` so SfM does not re-derive them from scratch;
    2. COLMAP with ALIKED + LightGlue, exhaustive matching (3.1, 3.2);
    3. gsplat for ``req.steps``, with the dataloader normalization PINNED to
       ``frame`` -- see the anchor-frame note above;
    4. export a light PLY (3.5) under the asset root, return its url and bytes.

    Steps 2-3 take minutes and need the ``recon`` group plus a CUDA COLMAP
    build, so they belong in a subprocess/worker host that this coroutine
    awaits -- not inline in the event loop. Until that host exists this raises
    instead of pretending.
    """
    missing = [name for name in ("gsplat", "pycolmap") if not _importable(name)]
    detail = f" (missing deps: {', '.join(missing)}; `uv sync --group recon`)" if missing else ""
    raise PipelineUnavailable(
        "live 3DGS reconstruction is not wired in this process"
        + detail
        + f"; requested segment={req.segment} steps={req.steps} frame={frame.id} "
        f"sources={len(req.sources)}. Run with SKYLENS_DEMO=1 for the prebuilt assets."
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

#: Demo detections, as ENU meters from the AIRCRAFT that filmed the segment.
#: Two per segment so the board always has one person and one danger marker to
#: gate on that segment's arrival.
#:
#: Offsets are small and lateral because that is where a detection can come
#: from: a camera sees the ground it is over. Anchoring these to the configured
#: site instead put every marker in a neat row hundreds of metres from the
#: flight, so the board showed the drones in one corner and the findings in the
#: other — the reconstruction and the track appeared to be in different places.
_DEMO_DETECTIONS: tuple[tuple[str, str, float, float, float, float], ...] = (
    ("person", "구조 대상자 추정", -14.0, 6.0, 1.5, 0.82),
    ("danger", "붕괴 위험 구조물", 9.0, -8.0, 0.0, 0.74),
)


def _segment_fixes(req: DetectJobRequest | ReconJobRequest) -> list[GpsModel]:
    """Every pose the segment was filmed from, in order.

    Both job kinds carry the same sources: what the segment was filmed from is
    what places its reconstruction AND what places what was found in it.
    """
    return [pose.gps for source in req.sources for pose in source.poses]


async def run_detect(req: DetectJobRequest, report: Report) -> DetectJobResult:
    """Detect people and danger zones in one capture segment."""
    if settings().demo:
        return await _detect_demo(req, report)
    return await _detect_live(req, report)


async def _detect_demo(req: DetectJobRequest, report: Report) -> DetectJobResult:
    fixes = _segment_fixes(req)
    detections: list[DetectionResult] = []
    for i, (category, label, e, n, u, confidence) in enumerate(_DEMO_DETECTIONS):
        # Where the aircraft was when this part of the segment was filmed. Two
        # detections, so read a third and two thirds of the way along its track
        # — the markers then spread themselves along the flight instead of
        # needing a per-segment fudge. With no poses (a hand-made request) fall
        # back to the configured site, which is all there is to go on.
        anchor = settings().anchor
        if fixes:
            fix = fixes[min(len(fixes) - 1, (len(fixes) * (i + 1)) // 3)]
            # Horizontal position from the flight; height from the site datum.
            # A detection is on the ground, not at the aircraft's altitude, and
            # nothing here knows the terrain — ARCHITECTURE.md §3-A puts that in
            # the projection layer, which raycasts the depth map onto the DEM.
            origin = Gps(lat=fix.lat, lon=fix.lon, alt=anchor.alt)
        else:
            origin = anchor
        gps: Gps = enu_to_gps(Enu(e=e, n=n, u=u), origin)
        detections.append(
            DetectionResult(
                id=f"det-s{req.segment}-{i}",
                category=category,  # type: ignore[arg-type]
                gps=GpsModel(lat=gps.lat, lon=gps.lon, alt=gps.alt),
                confidence=confidence,
                label=label,
                segment=req.segment,
            )
        )
    await _simulate(max(0.2, settings().min_seconds), report, ticks=4)
    return DetectJobResult(segment=req.segment, detections=detections)


async def _detect_live(req: DetectJobRequest, report: Report) -> DetectJobResult:
    """SEAM: hand the segment to SkyLensNet + the projection layer.

    What goes here (skylens_model/README.md, layers 1-3):

    1. decode ``req.sources[*].uri`` into 4-channel (RGB + thermal) frames;
    2. ``SkyLensForDisasterPerception`` batch inference -- stateless, per frame;
    3. ray-cast each 2D hit through that frame's depth map and the pose in
       ``sources[*].poses`` to world coordinates, then to GPS via utils/geo.py;
    4. landmark fusion (log-odds accumulation) across frames, emitting one
       DetectionResult per surviving landmark.

    Blocked on two things, both stated in the model README: there is no trained
    checkpoint in the repo, and layer 2 (depth ray-casting) is not implemented.
    Made-up markers here would be indistinguishable from real ones on the
    board, so this raises instead.
    """
    torch_note = "" if _importable("torch") else " (torch not importable either)"
    raise PipelineUnavailable(
        "live detection is not wired in this process: no trained SkyLensNet "
        "checkpoint and no depth ray-casting layer yet"
        + torch_note
        + f"; requested segment={req.segment} sources={len(req.sources)}. "
        "Run with SKYLENS_DEMO=1 for fixed demo markers."
    )


# ---------------------------------------------------------------------------


def _importable(name: str) -> bool:
    """True if the module can be found without importing it (torch is slow)."""
    from importlib.util import find_spec

    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False
