"""Person heatmap focal loss normalisation: batch vs per-sample."""

from __future__ import annotations

import pytest
import torch

from skylens_model.models.skylensnet.configuration_skylensnet import SkyLensConfig
from skylens_model.models.skylensnet.modeling_skylensnet import centernet_focal_loss


def _old_formula(pred, target, alpha=2.0, beta=4.0):
    eps = 1e-4
    pred = pred.clamp(min=eps, max=1.0 - eps)
    pos = target.eq(1.0).float()
    neg = 1.0 - pos
    pos_loss = (torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos).sum()
    neg_loss = (torch.log(1.0 - pred) * torch.pow(pred, alpha) * torch.pow(1.0 - target, beta) * neg).sum()
    n = pos.sum()
    return -neg_loss if n == 0 else -(pos_loss + neg_loss) / n


def _random_batch(seed=0, b=3, h=16, w=16):
    g = torch.Generator().manual_seed(seed)
    pred = torch.rand(b, 1, h, w, generator=g)
    target = torch.rand(b, 1, h, w, generator=g) * 0.9
    target[target > 0.85] = 1.0
    return pred, target


def test_batch_matches_old_formula() -> None:
    pred, target = _random_batch()
    assert (target == 1.0).sum() > 0
    assert torch.allclose(centernet_focal_loss(pred, target, norm="batch"), _old_formula(pred, target))
    assert torch.allclose(centernet_focal_loss(pred, target), _old_formula(pred, target))


def _image_with_positives(k: int, h=16, w=16):
    """Target with k isolated peaks; pred makes each peak cost the same, negatives free."""
    target = torch.zeros(1, h, w)
    pred = torch.full((1, h, w), 1e-4)
    for i in range(k):
        target[0, i, i] = 1.0
        pred[0, i, i] = 0.1
    return pred, target


def test_per_sample_weights_images_equally() -> None:
    p1, t1 = _image_with_positives(1)
    p10, t10 = _image_with_positives(10)
    pred = torch.stack([p1, p10]).requires_grad_(True)
    target = torch.stack([t1, t10])

    def grad_share(norm):
        pred.grad = None
        centernet_focal_loss(pred, target, norm=norm).backward()
        g = pred.grad.abs()
        return float(g[0].sum()), float(g[1].sum())

    b0, b1 = grad_share("batch")
    assert b1 == pytest.approx(10 * b0, rel=1e-4)  # batch: the 10-person image dominates
    s0, s1 = grad_share("per-sample")
    assert s1 == pytest.approx(s0, rel=1e-4)  # per-sample: equal total weight per image


def test_per_sample_equals_batch_without_positives() -> None:
    pred = torch.rand(2, 1, 8, 8)
    target = torch.zeros(2, 1, 8, 8)
    assert torch.allclose(
        centernet_focal_loss(pred, target, norm="per-sample"), centernet_focal_loss(pred, target)
    )


def test_old_config_loads_as_batch() -> None:
    d = SkyLensConfig(use_pretrained_backbone=False).to_dict()
    d.pop("heatmap_norm")
    assert SkyLensConfig.from_dict(d).heatmap_norm == "batch"


def test_invalid_heatmap_norm_raises() -> None:
    with pytest.raises(ValueError):
        SkyLensConfig(use_pretrained_backbone=False, heatmap_norm="per-image")
    with pytest.raises(ValueError):
        centernet_focal_loss(torch.rand(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), norm="bogus")
