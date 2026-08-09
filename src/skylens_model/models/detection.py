"""Human/danger detection interface.

Real design (see IDEA.md / ARCHITECTURE.md): a UNet/TransUNet backbone
consumes 4-channel input (RGB + thermal, aligned) and feeds two heads:

- a segmentation head for danger zones ("stuff": collapse, fire) —
  per-pixel class map, not individual instances.
- an instance head for people — one Detection per person found.

2D detections are then projected into 3D world coordinates via depth-map
raycasting against the reconstructed scene (see `skylens_model.models.splat`),
which is how a `Detection.gps` gets populated.

This module is a SCAFFOLD: it defines the data shapes and the interface a
real detector must implement. No tensor math, no numpy/torch — image data is
typed as `typing.Any` placeholders where real tensors/arrays will go.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

# Mirrors DetectionResult['category'] in src/skylens_core/protocol.ts.
Category = Literal["person", "danger"]

# GPS as (lat, lon, alt), matching the `Gps` shape in src/skylens_core/geo.ts.
GpsTuple = tuple[float, float, float]


@dataclass(slots=True)
class Frame:
    """A single timestamped capture from one drone, ready for AI inference.

    `rgb` and `thermal` are placeholders for the real 4-channel input (RGB
    stacked with a single thermal channel). They are typed `Any` here — in a
    real implementation these would be numpy arrays / torch tensors of shape
    (H, W, 3) and (H, W, 1) respectively.
    """

    rgb: Any
    thermal: Any
    timestamp: float
    # Drone pose at capture time: position (x, y, z) and orientation
    # quaternion (x, y, z, w), in the shared ENU/scene frame.
    drone_pos: tuple[float, float, float]
    drone_quat: tuple[float, float, float, float]
    # GPS of the drone at capture time (lat, lon, alt).
    drone_gps: GpsTuple


@dataclass(slots=True)
class Detection:
    """One detected instance (person) or danger-zone marker, in world GPS.

    Mirrors `DetectionResult` in src/skylens_core/protocol.ts: `category`,
    `gps`, `confidence`, `label`.
    """

    category: Category
    gps: GpsTuple
    confidence: float
    label: str


class HumanDetector:
    """Interface for the UNet-based person/danger detector.

    Real implementations wrap a trained 4-channel UNet/TransUNet with a
    segmentation head (danger zones) and an instance head (people), plus the
    depth-map raycasting step that lifts 2D detections into 3D/GPS using the
    reconstructed scene from `skylens_model.models.splat`.
    """

    def infer(self, frame: Frame) -> list[Detection]:
        """Run detection on a single frame and return world-space Detections.

        Placeholder — no model is loaded or executed here.
        """
        raise NotImplementedError

    def infer_batch(self, frames: Sequence[Frame]) -> list[Detection]:
        """Run detection over multiple frames (e.g. one batch across drones).

        Placeholder — default scaffold behavior chains `infer` per frame so
        subclasses only need to implement the single-frame path.
        """
        raise NotImplementedError
