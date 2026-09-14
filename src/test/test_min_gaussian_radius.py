"""Minimum Gaussian radius for person heatmap targets."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from skylens_model.utils.collate import SkyLensCollator, draw_gaussian, gaussian_radius


def _sample(boxes):
    return {"image": np.zeros((64, 64, 3), np.float32), "has_rgb": True, "has_thermal": False,
            "danger_mask": None, "person_boxes": np.asarray(boxes, np.float32)}


# tiny box: 6x6 px -> 1.5x1.5 cells -> CenterNet radius 0; centre cell (5, 5)
TINY = [[19.0, 19.0, 25.0, 25.0]]
BOXES = TINY + [[2.0, 30.0, 40.0, 62.0]]


def test_tiny_box_has_radius_zero() -> None:
    assert int(gaussian_radius((1.5, 1.5), 0.7)) == 0


def test_default_matches_v3_reference() -> None:
    batch = SkyLensCollator(person_head_stride=4)([_sample(BOXES)])
    ref = np.zeros((16, 16), np.float32)
    s = 4
    for x1, y1, x2, y2 in BOXES:
        bw, bh = (x2 - x1) / s, (y2 - y1) / s
        cx, cy = (x1 + x2) / 2 / s, (y1 + y2) / 2 / s
        draw_gaussian(ref, (int(cx), int(cy)), max(0, int(gaussian_radius((bh, bw), 0.7))))
    assert np.array_equal(batch["person_heatmap"][0, 0].numpy(), ref)
    explicit = SkyLensCollator(person_head_stride=4, min_gaussian_radius=0)([_sample(BOXES)])
    for k in batch:
        assert torch.equal(batch[k], explicit[k])


def test_min_radius_one_gives_3x3() -> None:
    b0 = SkyLensCollator(person_head_stride=4)([_sample(TINY)])
    b1 = SkyLensCollator(person_head_stride=4, min_gaussian_radius=1)([_sample(TINY)])
    h0, h1 = b0["person_heatmap"][0, 0], b1["person_heatmap"][0, 0]
    assert int((h0 > 0).sum()) == 1 and h0[5, 5] == 1.0
    patch = h1[4:7, 4:7]
    assert patch[1, 1] == 1.0
    others = torch.cat([patch.flatten()[:4], patch.flatten()[5:]])
    assert bool(((others > 0) & (others < 1)).all())
    assert int((h1 > 0).sum()) == 9
    for k in ("person_reg_mask", "person_wh", "person_offset"):
        assert torch.equal(b0[k], b1[k])


def test_negative_min_radius_rejected() -> None:
    with pytest.raises(ValueError):
        SkyLensCollator(min_gaussian_radius=-1)
