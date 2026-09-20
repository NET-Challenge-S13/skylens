"""히트맵 focal loss 의 positive 항 가중치(`heatmap_pos_weight`) 검증."""

from __future__ import annotations

import pytest
import torch

from skylens_model.models import SkyLensConfig
from skylens_model.models.skylensnet.modeling_skylensnet import centernet_focal_loss


def _reference_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """v4 시점의 식 그대로. 회귀 비교용 기준."""
    eps = 1e-4
    pred = pred.clamp(min=eps, max=1.0 - eps)
    pos_mask = target.eq(1.0).float()
    neg_mask = 1.0 - pos_mask
    pos_loss = (torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos_mask).sum()
    neg_weights = torch.pow(1.0 - target, beta)
    neg_loss = (torch.log(1.0 - pred) * torch.pow(pred, alpha) * neg_weights * neg_mask).sum()
    num_pos = pos_mask.sum()
    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos


def _random_example(seed: int = 0, with_positives: bool = True):
    g = torch.Generator().manual_seed(seed)
    pred = torch.rand(2, 1, 16, 16, generator=g)
    target = torch.rand(2, 1, 16, 16, generator=g) * 0.6
    if with_positives:
        target[0, 0, 4, 4] = 1.0
        target[1, 0, 9, 2] = 1.0
        target[1, 0, 11, 7] = 1.0
    return pred, target


def test_default_matches_v4_formula() -> None:
    pred, target = _random_example()
    assert torch.allclose(centernet_focal_loss(pred, target), _reference_loss(pred, target), atol=0, rtol=0)
    assert torch.allclose(centernet_focal_loss(pred, target, pos_weight=1.0), _reference_loss(pred, target))


def test_pos_weight_scales_only_positive_term() -> None:
    # positive 항과 negative 항을 분리해 뽑아낸다:
    #   target 전체를 0 으로 두고 중심만 1 → neg_weights 는 중심 외 전부 1
    #   pred 를 중심에서만 다르게 주면 두 항의 기여를 각각 계산할 수 있다.
    pred, target = _random_example(seed=3)
    base = centernet_focal_loss(pred, target, pos_weight=1.0)
    w4 = centernet_focal_loss(pred, target, pos_weight=4.0)

    eps = 1e-4
    p = pred.clamp(min=eps, max=1.0 - eps)
    pos_mask = target.eq(1.0).float()
    num_pos = pos_mask.sum()
    pos_term = -(torch.log(p) * torch.pow(1.0 - p, 2.0) * pos_mask).sum() / num_pos
    neg_term = base - pos_term

    assert torch.allclose(w4, 4.0 * pos_term + neg_term, atol=1e-6)
    # negative 항은 건드리지 않았다 (차이가 정확히 positive 항의 3배)
    assert torch.allclose(w4 - base, 3.0 * pos_term, atol=1e-6)
    assert pos_term > 0 and neg_term > 0


def test_no_positive_batch_is_unchanged() -> None:
    pred, target = _random_example(seed=7, with_positives=False)
    ref = _reference_loss(pred, target)
    for w in (1.0, 4.0, 10.0):
        loss = centernet_focal_loss(pred, target, pos_weight=w)
        assert torch.allclose(loss, ref)
        assert torch.isfinite(loss)


def test_old_config_loads_as_one() -> None:
    old = SkyLensConfig(dice_loss_weight=1.0).to_dict()
    old.pop("heatmap_pos_weight", None)
    assert "heatmap_pos_weight" not in old
    restored = SkyLensConfig.from_dict(old)
    assert restored.heatmap_pos_weight == 1.0


def test_config_round_trip_keeps_value() -> None:
    cfg = SkyLensConfig.from_dict(SkyLensConfig(heatmap_pos_weight=4.0).to_dict())
    assert cfg.heatmap_pos_weight == 4.0


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_invalid_values_raise(bad: float) -> None:
    with pytest.raises(ValueError, match="heatmap_pos_weight"):
        SkyLensConfig(heatmap_pos_weight=bad)
