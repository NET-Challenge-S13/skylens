"""CenterNet sub-pixel offset head: targets, config compatibility, decode."""

from __future__ import annotations

import numpy as np
import torch

from skylens_model.models.skylensnet.configuration_skylensnet import SkyLensConfig
from skylens_model.models.skylensnet.modeling_skylensnet import SkyLensForDisasterPerception
from skylens_model.utils.collate import SkyLensCollator
from skylens_model.utils.metrics import decode_gt_boxes, decode_heatmap_peaks


def _sample(boxes):
    return {"image": np.zeros((32, 32, 3), np.float32), "has_rgb": True, "has_thermal": False,
            "danger_mask": None, "person_boxes": np.asarray(boxes, np.float32)}


def test_offset_target_values() -> None:
    # centre (9.0, 14.5) px -> grid (2.25, 3.625) -> cell (2, 3), offset (0.25, 0.625)
    batch = SkyLensCollator(person_head_stride=4)([_sample([[6.0, 10.0, 12.0, 19.0]])])
    off, reg = batch["person_offset"], batch["person_reg_mask"]
    assert tuple(off.shape) == (1, 2, 8, 8)
    assert reg[0, 0, 3, 2] == 1.0
    assert off[0, :, 3, 2].tolist() == [0.25, 0.625]
    assert float((off * (1 - reg)).abs().sum()) == 0.0


def test_old_config_loads_without_head() -> None:
    d = SkyLensConfig(use_pretrained_backbone=False).to_dict()
    d.pop("use_offset_head"), d.pop("offset_loss_weight")
    cfg = SkyLensConfig.from_dict(d)
    assert cfg.use_offset_head is False
    model = SkyLensForDisasterPerception(cfg)
    assert model.offset_head is None
    assert not any("offset" in k for k in model.state_dict())
    out = model.eval()(pixel_values=torch.zeros(1, cfg.in_channels, 64, 64))
    assert out.person_offset is None


def test_offset_head_forward_and_loss() -> None:
    cfg = SkyLensConfig(use_pretrained_backbone=False, use_offset_head=True)
    model = SkyLensForDisasterPerception(cfg)
    boxes = [[6.0, 10.0, 12.0, 19.0]]
    s = _sample(boxes)
    s["image"] = np.zeros((64, 64, 3), np.float32)
    batch = SkyLensCollator(person_head_stride=cfg.person_head_stride)([s])
    out = model(**batch)
    assert out.person_offset.shape[1] == 2
    assert torch.isfinite(out.loss_dict["person_offset"])


def test_decode_with_offset_recovers_subcell_centre() -> None:
    # true centre (9.0, 14.5) px, w=6 h=9 at stride 4
    hm = torch.zeros(1, 1, 8, 8)
    hm[0, 0, 3, 2] = 0.9
    wh = torch.zeros(1, 2, 8, 8)
    wh[0, :, 3, 2] = torch.tensor([1.5, 2.25])
    off = torch.zeros(1, 2, 8, 8)
    off[0, :, 3, 2] = torch.tensor([0.25, 0.625])
    det = decode_heatmap_peaks(hm, wh, k=2, threshold=0.3, stride=4, offset=off)
    assert det[0, 0].tolist() == [9.0, 14.5, 6.0, 9.0, det[0, 0, 4].item()]
    # GT decode with the collator target yields the same continuous box
    batch = SkyLensCollator(person_head_stride=4)([_sample([[6.0, 10.0, 12.0, 19.0]])])
    gt = decode_gt_boxes(batch["person_reg_mask"], batch["person_wh"], k=2, stride=4,
                         offset=batch["person_offset"])
    assert gt[0, 0].tolist() == [9.0, 14.5, 6.0, 9.0, 1.0]
