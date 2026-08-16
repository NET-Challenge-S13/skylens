"""Batch assembly + target encoding.

Turns a list of dataset samples (the dict contract in ``base.py``) into the
tensor dict the SkyLens UNet's ``forward`` expects.

Output keys
-----------
=================== ============================== =================================
key                 shape / dtype                  notes
=================== ============================== =================================
``pixel_values``    (B, C, H, W) float32           C = 4, or 5 with ``validity_channel``
``modality_mask``   (B, 2) bool                    ``[rgb_present, thermal_present]``
``danger_labels``   (B, H, W) int64                255 = ignore; key omitted when no
                                                   sample in the batch has a mask
``person_heatmap``  (B, 1, H/s, W/s) float32       CenterNet gaussian splat
``person_wh``       (B, 2, H/s, W/s) float32       (w, h) in output-stride units
``person_reg_mask`` (B, 1, H/s, W/s) float32       1 at valid centres
=================== ============================== =================================

The three ``person_*`` keys are omitted together when no sample in the batch
carries boxes. That is what makes README §6.3 (per-head training) work: the
training loop just checks key presence to decide which loss to compute.

Channel layout (README §2.1/§2.3)::

    0:3  RGB, scaled to [0, 1]
    3    thermal, min-max normalised into [0.1, 1.0]; exactly 0.0 means "absent"
    4    (optional) validity mask, 1.0 where thermal is real

Reserving 0.0 rather than plain zero-filling is deliberate: FLAME 3's thermal is
radiometric, so a raw 0 is a legitimate temperature and the model could not
otherwise tell "cold" from "no sensor".
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from skylens_model.datasets.base import IGNORE_INDEX

__all__ = [
    "SkyLensCollator",
    "gaussian_radius",
    "gaussian2d",
    "draw_gaussian",
    "THERMAL_ABSENT",
    "THERMAL_MIN",
]

#: Value reserved in the thermal channel to mean "this modality is absent".
THERMAL_ABSENT = 0.0
#: Lower bound of the live thermal range (README §2.3).
THERMAL_MIN = 0.1


# --------------------------------------------------------------------------- #
# CenterNet target encoding
# --------------------------------------------------------------------------- #


def gaussian_radius(det_size: tuple[float, float], min_overlap: float = 0.7) -> float:
    """CornerNet/CenterNet gaussian radius for a box of ``(height, width)``.

    Solves the three "IoU >= ``min_overlap``" cases and returns the smallest
    root, i.e. the largest radius for which a displaced box still overlaps
    enough. Reference: Law & Deng, *CornerNet* (ECCV 2018), ``utils/image.py``.
    """
    height, width = det_size

    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(max(b1 * b1 - 4 * a1 * c1, 0.0))
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4.0
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(max(b2 * b2 - 4 * a2 * c2, 0.0))
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(max(b3 * b3 - 4 * a3 * c3, 0.0))
    r3 = (-b3 + sq3) / (2 * a3)

    return float(min(r1, r2, r3))


def gaussian2d(shape: tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    """Unnormalised 2-D gaussian kernel with peak 1.0 at its centre."""
    m, n = (shape[0] - 1.0) / 2.0, (shape[1] - 1.0) / 2.0
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0.0
    return h


def draw_gaussian(heatmap: np.ndarray, center: tuple[int, int], radius: int,
                  k: float = 1.0) -> np.ndarray:
    """Splat a gaussian onto ``heatmap`` (H, W) in-place, combining with ``max``.

    ``center`` is ``(cx, cy)`` in heatmap coordinates. Overlapping peaks keep the
    stronger response so nearby people do not erase each other.
    """
    radius = int(max(radius, 0))
    diameter = 2 * radius + 1
    gauss = gaussian2d((diameter, diameter), sigma=diameter / 6.0)

    cx, cy = int(center[0]), int(center[1])
    h, w = heatmap.shape[:2]

    left, right = min(cx, radius), min(w - cx, radius + 1)
    top, bottom = min(cy, radius), min(h - cy, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return heatmap

    masked_heatmap = heatmap[cy - top:cy + bottom, cx - left:cx + right]
    masked_gaussian = gauss[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


# --------------------------------------------------------------------------- #
# collator
# --------------------------------------------------------------------------- #


class SkyLensCollator:
    """Collate SkyLens samples into the model's batch dict.

    Parameters
    ----------
    person_head_stride:
        Output stride ``s`` of the point-detection head. Targets are rendered at
        ``(H // s, W // s)``.
    validity_channel:
        Append a 5th channel that is 1.0 where the thermal plane is real
        (README §2.3 "명확" option). ``pixel_values`` then has ``C = 5``.
    min_overlap:
        ``min_overlap`` passed to :func:`gaussian_radius`.
    modality_dropout:
        ``(p_drop_thermal, p_drop_rgb)`` -- symmetric modality dropout from
        README §2.2. Defaults to ``(0.0, 0.0)``; the recommended training value
        is ``(0.25, 0.25)`` (leaving p=0.5 for the full 4-channel mode). Dropout
        is applied *after* stacking, and updates ``modality_mask`` accordingly.
    max_objects:
        Safety cap on boxes encoded per image.
    """

    def __init__(
        self,
        person_head_stride: int = 4,
        validity_channel: bool = False,
        min_overlap: float = 0.7,
        modality_dropout: tuple[float, float] = (0.0, 0.0),
        max_objects: int = 512,
        rng: np.random.Generator | None = None,
    ) -> None:
        if person_head_stride < 1:
            raise ValueError("person_head_stride must be >= 1")
        self.person_head_stride = int(person_head_stride)
        self.validity_channel = bool(validity_channel)
        self.min_overlap = float(min_overlap)
        self.modality_dropout = modality_dropout
        self.max_objects = int(max_objects)
        self._rng = rng if rng is not None else np.random.default_rng()

    # -- public ----------------------------------------------------------

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not samples:
            raise ValueError("SkyLensCollator received an empty batch")

        h, w = self._verify_uniform_size(samples)
        s = self.person_head_stride
        oh, ow = max(h // s, 1), max(w // s, 1)

        pixels, modality = [], []
        for sample in samples:
            px, has_rgb, has_thermal = self._to_pixel_values(sample)
            pixels.append(px)
            modality.append((has_rgb, has_thermal))

        batch: dict[str, torch.Tensor] = {
            "pixel_values": torch.from_numpy(np.stack(pixels)).float(),
            "modality_mask": torch.tensor(modality, dtype=torch.bool),
        }

        # --- segmentation head ---
        if any(s_.get("danger_mask") is not None for s_ in samples):
            labels = np.stack([
                s_["danger_mask"].astype(np.int64)
                if s_.get("danger_mask") is not None
                else np.full((h, w), IGNORE_INDEX, dtype=np.int64)
                for s_ in samples
            ])
            batch["danger_labels"] = torch.from_numpy(labels).long()

        # --- point-detection head ---
        if any(s_.get("person_boxes") is not None for s_ in samples):
            hm = np.zeros((len(samples), 1, oh, ow), dtype=np.float32)
            wh = np.zeros((len(samples), 2, oh, ow), dtype=np.float32)
            reg = np.zeros((len(samples), 1, oh, ow), dtype=np.float32)
            for i, s_ in enumerate(samples):
                boxes = s_.get("person_boxes")
                if boxes is None or len(boxes) == 0:
                    continue
                self._encode_boxes(np.asarray(boxes, np.float32), hm[i, 0], wh[i], reg[i, 0])
            batch["person_heatmap"] = torch.from_numpy(hm)
            batch["person_wh"] = torch.from_numpy(wh)
            batch["person_reg_mask"] = torch.from_numpy(reg)

        return batch

    # -- internals -------------------------------------------------------

    @staticmethod
    def _verify_uniform_size(samples: Sequence[dict[str, Any]]) -> tuple[int, int]:
        shapes = {np.asarray(s["image"]).shape[:2] for s in samples}
        if len(shapes) != 1:
            raise ValueError(
                "SkyLensCollator needs all images in a batch to share (H, W); got "
                f"{sorted(shapes)}. Add a resize/crop to the dataset `transforms`."
            )
        h, w = shapes.pop()
        return int(h), int(w)

    def _to_pixel_values(self, sample: dict[str, Any]) -> tuple[np.ndarray, bool, bool]:
        """Build a (C, H, W) float32 array with the fixed channel layout."""
        image = np.asarray(sample["image"])
        if image.ndim == 2:
            image = image[:, :, None]
        h, w, c = image.shape

        has_rgb = bool(sample.get("has_rgb", False))
        has_thermal = bool(sample.get("has_thermal", False))

        rgb = np.zeros((h, w, 3), dtype=np.float32)
        thermal = np.full((h, w), THERMAL_ABSENT, dtype=np.float32)

        if has_rgb and c >= 3:
            rgb = image[:, :, :3].astype(np.float32)
            if image.dtype == np.uint8:
                rgb = rgb / 255.0
            elif rgb.max() > 1.5:  # float image still in 0..255
                rgb = rgb / 255.0

        if has_thermal:
            plane = image[:, :, 3] if c >= 4 else image[:, :, 0]
            thermal = self._normalize_thermal(plane.astype(np.float32),
                                              already_scaled=(c >= 4))

        # symmetric modality dropout (README §2.2)
        p_thermal, p_rgb = self.modality_dropout
        if has_thermal and p_thermal > 0 and self._rng.random() < p_thermal:
            thermal = np.full((h, w), THERMAL_ABSENT, dtype=np.float32)
            has_thermal = False
        elif has_rgb and p_rgb > 0 and self._rng.random() < p_rgb:
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            has_rgb = False

        planes = [rgb, thermal[:, :, None]]
        if self.validity_channel:
            planes.append(np.full((h, w, 1), 1.0 if has_thermal else 0.0, np.float32))

        stacked = np.concatenate(planes, axis=2)
        return np.ascontiguousarray(stacked.transpose(2, 0, 1)), has_rgb, has_thermal

    @staticmethod
    def _normalize_thermal(plane: np.ndarray, already_scaled: bool) -> np.ndarray:
        """Map a thermal plane into ``[0.1, 1.0]``, keeping 0.0 reserved."""
        if already_scaled and float(plane.min()) >= 0.0 and float(plane.max()) <= 1.0:
            # dataset already applied the reserved-range normalisation
            return plane
        lo, hi = float(plane.min()), float(plane.max())
        norm = np.zeros_like(plane) if hi <= lo else (plane - lo) / (hi - lo)
        return (THERMAL_MIN + (1.0 - THERMAL_MIN) * norm).astype(np.float32)

    def _encode_boxes(self, boxes: np.ndarray, heatmap: np.ndarray,
                      wh: np.ndarray, reg_mask: np.ndarray) -> None:
        """CenterNet encoding of xyxy pixel boxes into the output-stride grids."""
        s = self.person_head_stride
        oh, ow = heatmap.shape

        for x1, y1, x2, y2 in boxes[: self.max_objects]:
            bw, bh = (x2 - x1) / s, (y2 - y1) / s
            if bw <= 0 or bh <= 0:
                continue
            cx, cy = ((x1 + x2) / 2.0) / s, ((y1 + y2) / 2.0) / s
            cxi, cyi = int(cx), int(cy)
            if not (0 <= cxi < ow and 0 <= cyi < oh):
                continue

            radius = max(0, int(gaussian_radius((bh, bw), self.min_overlap)))
            draw_gaussian(heatmap, (cxi, cyi), radius)

            wh[0, cyi, cxi] = bw
            wh[1, cyi, cxi] = bh
            reg_mask[cyi, cxi] = 1.0
