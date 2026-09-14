"""Isotropic (v3) vs anisotropic (TTFNet) person heatmap targets."""

from __future__ import annotations

import numpy as np
import pytest

from skylens_model.utils.collate import SkyLensCollator, draw_gaussian, gaussian_radius

STRIDE = 4


def _sample(boxes, size=256):
    return {"image": np.zeros((size, size, 3), np.float32), "has_rgb": True, "has_thermal": False,
            "danger_mask": None, "person_boxes": np.asarray(boxes, np.float32)}


# tall box: 10 x 40 cells at stride 4, centre cell (30, 30)
TALL = [[100.0, 40.0, 140.0, 200.0]]
MIXED = [[100.0, 40.0, 140.0, 200.0], [10.0, 10.0, 14.0, 13.0], [30.0, 150.0, 90.0, 180.0],
         [128.0, 100.0, 170.0, 130.0], [250.0, 250.0, 262.0, 270.0]]


def _batch(mode, boxes=TALL):
    return SkyLensCollator(person_head_stride=STRIDE, heatmap_gaussian=mode)([_sample(boxes)])


def _v3_heatmap(boxes, size=256):
    hm = np.zeros((size // STRIDE, size // STRIDE), np.float32)
    for x1, y1, x2, y2 in np.asarray(boxes, np.float32):
        bw, bh = (x2 - x1) / STRIDE, (y2 - y1) / STRIDE
        cxi, cyi = int((x1 + x2) / 2 / STRIDE), int((y1 + y2) / 2 / STRIDE)
        if bw <= 0 or bh <= 0 or not (0 <= cxi < hm.shape[1] and 0 <= cyi < hm.shape[0]):
            continue
        draw_gaussian(hm, (cxi, cyi), max(0, int(gaussian_radius((bh, bw), 0.7))))
    return hm


def test_default_is_isotropic_and_matches_v3() -> None:
    assert SkyLensCollator().heatmap_gaussian == "isotropic"
    for boxes in (TALL, MIXED):
        default = SkyLensCollator(person_head_stride=STRIDE)([_sample(boxes)])
        iso = _batch("isotropic", boxes)
        np.testing.assert_array_equal(iso["person_heatmap"][0, 0].numpy(), _v3_heatmap(boxes))
        np.testing.assert_array_equal(default["person_heatmap"].numpy(),
                                      iso["person_heatmap"].numpy())


def test_anisotropic_tall_box_shape() -> None:
    iso = _batch("isotropic")["person_heatmap"][0, 0].numpy()
    ani = _batch("anisotropic")["person_heatmap"][0, 0].numpy()
    cx, cy = 30, 30
    pos = ani > 0
    ys, xs = np.nonzero(pos)
    assert ys.max() - ys.min() > xs.max() - xs.min()
    # v3 isotropic radius for this box is int(gaussian_radius) = 1, a 3x3 kernel
    # (not a plateau), so the elliptical target is wider than it, most of all along y.
    assert int((iso > 0).sum()) == 9
    for k in (1, 2, 3, 4):
        assert ani[cy + k, cx] > ani[cy, cx + k]
        assert ani[cy + k, cx] >= iso[cy + k, cx]
        assert ani[cy - k, cx] == ani[cy + k, cx]


def test_anisotropic_peak_at_centre() -> None:
    for boxes in (TALL, MIXED):
        b = _batch("anisotropic", boxes)
        hm, reg = b["person_heatmap"][0, 0].numpy(), b["person_reg_mask"][0, 0].numpy()
        assert hm.max() == 1.0
        assert np.all(hm[reg == 1.0] == 1.0)
    assert _batch("anisotropic")["person_heatmap"][0, 0, 30, 30] == 1.0


def test_tiny_box_single_positive_cell() -> None:
    hm = _batch("anisotropic", [[40.0, 40.0, 42.0, 43.0]])["person_heatmap"][0, 0].numpy()
    assert int((hm > 0).sum()) == 1 and hm[10, 10] == 1.0


def test_regression_targets_unchanged() -> None:
    iso, ani = _batch("isotropic", MIXED), _batch("anisotropic", MIXED)
    for key in ("person_reg_mask", "person_wh", "person_offset"):
        np.testing.assert_array_equal(iso[key].numpy(), ani[key].numpy())


def test_invalid_value_raises() -> None:
    with pytest.raises(ValueError):
        SkyLensCollator(heatmap_gaussian="elliptic")
