"""3D Gaussian Splatting reconstruction interface.

Real pipeline (see ARCHITECTURE.md, Tier 3 "RECON"):

1. GLOMAP — pose estimation from multi-drone video frames.
2. gsplat — GPU 3D Gaussian Splatting training from posed frames.
3. Open3D ICP — fuses per-drone reconstructions into one aligned scene.

The result is exported as splat chunks the Edge VM serves to clients, each
carrying an alignment transform so it lands correctly in the shared ENU/scene
frame (see `skylens_model.utils.geo` and src/skylens_core/geo.ts).

This module is a SCAFFOLD: interface + data shapes only, no gsplat/GLOMAP
calls, no tensor math.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

# GPS as (lat, lon, alt), matching the `Gps` shape in src/skylens_core/geo.ts.
GpsTuple = tuple[float, float, float]


@dataclass(slots=True)
class SplatAlign:
    """Where a splat chunk lands in the shared ENU/scene frame.

    Mirrors `SplatAlign` in src/skylens_core/protocol.ts.
    """

    # Optional GPS anchor to place the chunk at; else it uses the scene origin.
    anchor: Optional[GpsTuple] = None
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # quaternion
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(slots=True)
class SplatChunkSpec:
    """A reconstructed splat chunk ready to hand off to the server/client.

    Mirrors `SplatChunk` in src/skylens_core/protocol.ts: `id`, `url`, `align`.
    """

    id: str
    url: str
    align: SplatAlign = field(default_factory=SplatAlign)


class SplatReconstructor:
    """Interface for the multi-drone 3DGS reconstruction pipeline.

    Real implementations accumulate posed frames from one or more drones,
    run GLOMAP -> gsplat -> Open3D ICP fusion, and export the result as
    `SplatChunkSpec` objects for upload/serving.
    """

    def add_frames(self, frames: Sequence[Any]) -> None:
        """Feed a batch of posed frames into the reconstruction pipeline.

        `frames` is a placeholder for a sequence of `skylens_model.models.
        detection.Frame`-like posed captures. Placeholder only — no
        accumulation happens here.
        """
        raise NotImplementedError

    def export_chunk(self) -> SplatChunkSpec:
        """Export the current reconstruction state as a splat chunk.

        Placeholder — no gsplat training/export is performed here.
        """
        raise NotImplementedError
