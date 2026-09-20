"""Gaussian radius rounding rule for the person heatmap targets.

v4 truncates the CornerNet radius toward zero, so every box whose radius is
below 1.0 becomes a strictly one-hot target with full-weight negatives right
next to the centre. ``radius_rounding="round"`` rounds instead, so those boxes
keep a minimal soft ring. Only the heatmap changes: reg_mask / wh / offset are
untouched by the rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from skylens_model.utils.collate import SkyLensCollator, gaussian_radius

STRIDE = 4
SIZE = 128


def _sample(boxes):
    return {
        "image": np.zeros((SIZE, SIZE, 3), np.float32),
        "has_rgb": True,
        "has_thermal": False,
        "danger_mask": None,
        "person_boxes": np.asarray(boxes, np.float32),
    }


def _box(w_px: float, h_px: float, cx: float = 64.0, cy: float = 64.0):
    return [cx - w_px / 2, cy - h_px / 2, cx + w_px / 2, cy + h_px / 2]


def _targets(boxes, **kw):
    batch = SkyLensCollator(person_head_stride=STRIDE, **kw)([_sample(boxes)])
    return {k: v.numpy() for k, v in batch.items()}


def _soft_cells(heatmap: np.ndarray) -> int:
    """Cells that are neither a peak nor a full-weight negative."""
    return int(((heatmap > 0.0) & (heatmap < 1.0)).sum())


# A small person: its CornerNet radius is 0.55 cells at min_overlap 0.7, so v4
# truncates it to 0 and the target is strictly one-hot.
SMALL_BOX = _box(20.0, 44.0)
# A large person: radius 3.38 cells at 0.7, 1.63 at 0.85.
LARGE_BOX = _box(120.0, 280.0)


def test_defaults_are_v4_trunc() -> None:
    """The default collator still truncates, and trunc is the explicit default."""
    default = _targets([SMALL_BOX, LARGE_BOX])
    explicit = _targets([SMALL_BOX, LARGE_BOX], radius_rounding="trunc", min_overlap=0.7)
    for key in default:
        np.testing.assert_array_equal(default[key], explicit[key])

    # and it reproduces the v4 formula cell for cell
    for box in (SMALL_BOX, LARGE_BOX):
        bw, bh = (box[2] - box[0]) / STRIDE, (box[3] - box[1]) / STRIDE
        radius = max(0, int(gaussian_radius((bh, bw), 0.7)))
        hm = _targets([box])["person_heatmap"][0, 0]
        assert _soft_cells(hm) == _soft_cells(_reference(radius))


def _reference(radius: int) -> np.ndarray:
    """v4 splat of a single centre, drawn independently of the collator."""
    from skylens_model.utils.collate import draw_gaussian

    hm = np.zeros((SIZE // STRIDE, SIZE // STRIDE), np.float32)
    return draw_gaussian(hm, (SIZE // STRIDE // 2, SIZE // STRIDE // 2), radius)


def test_small_box_is_one_hot_under_v4() -> None:
    hm = _targets([SMALL_BOX])["person_heatmap"][0, 0]
    assert hm.max() == pytest.approx(1.0)
    assert _soft_cells(hm) == 0


def test_round_gives_small_box_a_minimal_ring() -> None:
    """Rounding alone is what rescues a sub-cell box: radius 0.55 -> 1."""
    hm = _targets([SMALL_BOX], radius_rounding="round")["person_heatmap"][0, 0]
    assert hm.max() == pytest.approx(1.0)
    assert _soft_cells(hm) == 8  # the 3x3 neighbourhood minus the peak


def test_round_with_higher_overlap_shrinks_a_large_box_ring() -> None:
    v4 = _targets([LARGE_BOX])["person_heatmap"][0, 0]
    new = _targets([LARGE_BOX], radius_rounding="round", min_overlap=0.85)["person_heatmap"][0, 0]
    assert _soft_cells(v4) > 0
    assert 0 < _soft_cells(new) < _soft_cells(v4)


def test_higher_overlap_never_widens_a_ring_relative_to_v4() -> None:
    """Raising min_overlap to 0.85 more than halves the radius, so rounding on
    top of it can only keep or shrink the v4 ring. It never adds one."""
    for w, h in [(12, 26), (20, 44), (40, 90), (60, 140), (120, 280), (160, 360)]:
        box = _box(float(w), float(h))
        v4 = _targets([box])["person_heatmap"][0, 0]
        new = _targets([box], radius_rounding="round", min_overlap=0.85)["person_heatmap"][0, 0]
        assert _soft_cells(new) <= _soft_cells(v4), f"{w}x{h} grew"


def test_regression_targets_are_untouched() -> None:
    boxes = [SMALL_BOX, LARGE_BOX, _box(20.0, 40.0, cx=30.0, cy=90.0)]
    base = _targets(boxes)
    for kw in (
        {"radius_rounding": "round"},
        {"radius_rounding": "round", "min_overlap": 0.85},
        {"min_overlap": 0.3},
    ):
        other = _targets(boxes, **kw)
        for key in ("person_reg_mask", "person_wh", "person_offset"):
            np.testing.assert_array_equal(base[key], other[key], err_msg=f"{key} changed for {kw}")


@pytest.mark.parametrize("bad", ["nearest", "floor", "", "ROUND"])
def test_invalid_rounding_raises(bad: str) -> None:
    with pytest.raises(ValueError, match="radius_rounding"):
        SkyLensCollator(radius_rounding=bad)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_invalid_min_overlap_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="min_overlap"):
        SkyLensCollator(min_overlap=bad)
