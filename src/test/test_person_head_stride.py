"""Person head output stride: stride 4 unchanged, stride 2 consistent end to end.

v4 runs the point-detection head at output stride 4 (a 4px cell at 512 input).
This experiment runs it at stride 2. The three places that must agree are the
collator's target grid, the model head's output size and the decoder's ``stride``
argument. The segmentation path is independent of all of it.
"""

from __future__ import annotations

import numpy as np
import torch

from skylens_model.models.skylensnet.configuration_skylensnet import SkyLensConfig
from skylens_model.models.skylensnet.modeling_skylensnet import SkyLensForDisasterPerception
from skylens_model.utils.collate import SkyLensCollator
from skylens_model.utils.metrics import decode_gt_boxes, decode_heatmap_peaks

BOX = [6.0, 10.0, 12.0, 19.0]  # centre (9.0, 14.5), w=6, h=9


def _sample(size: int = 64, boxes=(BOX,), with_mask: bool = False) -> dict:
    return {
        "image": np.zeros((size, size, 3), np.float32),
        "has_rgb": True,
        "has_thermal": False,
        "danger_mask": np.zeros((size, size), np.uint8) if with_mask else None,
        "person_boxes": np.asarray(boxes, np.float32),
    }


# --------------------------------------------------------------------------- #
# stride 4 is unchanged versus v4
# --------------------------------------------------------------------------- #


def test_stride4_targets_unchanged() -> None:
    batch = SkyLensCollator(person_head_stride=4)([_sample(32)])
    assert tuple(batch["person_heatmap"].shape) == (1, 1, 8, 8)
    assert tuple(batch["person_wh"].shape) == (1, 2, 8, 8)
    assert batch["person_reg_mask"][0, 0, 3, 2] == 1.0
    assert batch["person_wh"][0, :, 3, 2].tolist() == [1.5, 2.25]
    assert batch["person_offset"][0, :, 3, 2].tolist() == [0.25, 0.625]


def test_stride4_is_still_the_default() -> None:
    assert SkyLensConfig(use_pretrained_backbone=False).person_head_stride == 4
    assert SkyLensCollator().person_head_stride == 4


# --------------------------------------------------------------------------- #
# stride 2: shapes
# --------------------------------------------------------------------------- #


def test_stride2_target_grid_shape() -> None:
    batch = SkyLensCollator(person_head_stride=2)([_sample(64)])
    for key, c in (("person_heatmap", 1), ("person_wh", 2), ("person_offset", 2)):
        assert tuple(batch[key].shape) == (1, c, 32, 32), key
    assert tuple(batch["person_reg_mask"].shape) == (1, 1, 32, 32)


def test_stride2_model_outputs_match_target_grid() -> None:
    cfg = SkyLensConfig(use_pretrained_backbone=False, use_offset_head=True, person_head_stride=2)
    model = SkyLensForDisasterPerception(cfg).eval()
    batch = SkyLensCollator(person_head_stride=2)([_sample(64, with_mask=True)])
    out = model(**batch)
    assert tuple(out.person_heatmap.shape) == tuple(batch["person_heatmap"].shape)
    assert tuple(out.person_wh.shape) == tuple(batch["person_wh"].shape)
    assert tuple(out.person_offset.shape) == tuple(batch["person_offset"].shape)
    assert torch.isfinite(out.loss)


def test_seg_output_unaffected_by_person_head_stride() -> None:
    """The danger head always returns input resolution, whatever the point stride."""
    px = torch.zeros(1, 4, 64, 64)
    shapes = []
    for stride in (4, 2):
        cfg = SkyLensConfig(use_pretrained_backbone=False, person_head_stride=stride)
        model = SkyLensForDisasterPerception(cfg).eval()
        with torch.no_grad():
            out = model(pixel_values=px)
        shapes.append(tuple(out.danger_logits.shape))
        assert out.danger_logits.shape[-2:] == (64, 64)
    assert shapes[0] == shapes[1]


# --------------------------------------------------------------------------- #
# stride 2: encode -> decode round trip
# --------------------------------------------------------------------------- #


def test_stride2_encode_decode_round_trip() -> None:
    """A GT box encoded at stride 2 decodes back to the exact same centre and size."""
    batch = SkyLensCollator(person_head_stride=2)([_sample(64)])
    gt = decode_gt_boxes(
        batch["person_reg_mask"], batch["person_wh"], k=4, stride=2,
        offset=batch["person_offset"],
    )
    assert gt[0, 0].tolist() == [9.0, 14.5, 6.0, 9.0, 1.0]


def test_stride2_peak_decode_uses_stride_units() -> None:
    """wh and offset targets are in grid cells: decoding multiplies both by the stride."""
    batch = SkyLensCollator(person_head_stride=2)([_sample(64)])
    # centre (9.0, 14.5) px / 2 -> grid (4.5, 7.25) -> cell (4, 7), offset (0.5, 0.25)
    assert batch["person_reg_mask"][0, 0, 7, 4] == 1.0
    assert batch["person_wh"][0, :, 7, 4].tolist() == [3.0, 4.5]
    assert batch["person_offset"][0, :, 7, 4].tolist() == [0.5, 0.25]

    hm = torch.zeros_like(batch["person_heatmap"])
    hm[0, 0, 7, 4] = 0.9
    det = decode_heatmap_peaks(
        hm, batch["person_wh"], k=2, threshold=0.3, stride=2, offset=batch["person_offset"],
    )
    assert det[0, 0, :4].tolist() == [9.0, 14.5, 6.0, 9.0]


def test_stride2_separates_neighbours_that_stride4_merges() -> None:
    """Two people 6px apart share a stride-4 cell pair but get distinct stride-2 peaks."""
    boxes = [[7.0, 14.0, 11.0, 24.0], [13.0, 14.0, 17.0, 24.0]]  # centres x=9 and x=15
    b4 = SkyLensCollator(person_head_stride=4)([_sample(64, boxes)])
    b2 = SkyLensCollator(person_head_stride=2)([_sample(64, boxes)])
    assert int(b4["person_reg_mask"].sum()) == 2
    assert int(b2["person_reg_mask"].sum()) == 2

    def peak_cells(batch):
        return sorted(int(x) for x in (batch["person_reg_mask"][0, 0] > 0).nonzero()[:, 1])

    # stride 4: cells 2 and 3 are neighbours, so the 3x3 max-pool NMS in
    # decode_heatmap_peaks can never return both. stride 2: 3 cells apart.
    c4, c2 = peak_cells(b4), peak_cells(b2)
    assert c4[1] - c4[0] == 1
    assert c2[1] - c2[0] >= 2

    # With realistic (non-tied) scores the NMS outcome follows from that spacing.
    def surviving(batch, stride):
        hm = torch.zeros_like(batch["person_heatmap"])
        cells = (batch["person_reg_mask"][0, 0] > 0).nonzero()
        for rank, (y, x) in enumerate(cells.tolist()):
            hm[0, 0, y, x] = 0.9 - 0.1 * rank
        det = decode_heatmap_peaks(hm, batch["person_wh"], k=8, threshold=0.3, stride=stride,
                                   offset=batch["person_offset"])
        return int((det[0, :, 4] > 0).sum())

    assert surviving(b4, 4) == 1
    assert surviving(b2, 2) == 2
