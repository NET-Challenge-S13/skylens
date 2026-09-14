"""사람 박스 크기(wh) loss 형태별 경계 조건과 스케일 민감도 검증."""

from __future__ import annotations

import pytest
import torch

from skylens_model.models.skylensnet.modeling_skylensnet import (
    masked_centered_giou_loss,
    masked_l1_loss,
    masked_log_l1_loss,
)

LOSSES = {
    "l1": masked_l1_loss,
    "log_l1": masked_log_l1_loss,
    "giou": masked_centered_giou_loss,
}


def _single(pred_wh: tuple, gt_wh: tuple) -> tuple:
    pred = torch.zeros(1, 2, 4, 4)
    gt = torch.zeros(1, 2, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    pred[0, :, 1, 2] = torch.tensor(pred_wh)
    gt[0, :, 1, 2] = torch.tensor(gt_wh)
    mask[0, 0, 1, 2] = 1.0
    return pred, gt, mask


@pytest.mark.parametrize("name", list(LOSSES))
def test_no_positives_is_finite(name: str) -> None:
    pred = torch.rand(2, 2, 8, 8, requires_grad=True)
    loss = LOSSES[name](pred, torch.zeros(2, 2, 8, 8), torch.zeros(2, 1, 8, 8))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("name", list(LOSSES))
def test_perfect_prediction_is_near_zero(name: str) -> None:
    pred, gt, mask = _single((5.0, 12.0), (5.0, 12.0))
    pred[0, :, 0, 0] = 99.0  # 마스크 밖 예측은 무관해야 한다
    assert LOSSES[name](pred, gt, mask).item() < 1e-4


@pytest.mark.parametrize("name", list(LOSSES))
def test_grows_with_size_error(name: str) -> None:
    values = [LOSSES[name](*_single((10.0 + d, 10.0 + d), (10.0, 10.0))).item() for d in (0, 1, 3, 8)]
    assert all(a < b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("name", ["log_l1", "giou"])
def test_small_box_error_costs_more(name: str) -> None:
    small = LOSSES[name](*_single((11.0, 11.0), (8.0, 8.0))).item()
    large = LOSSES[name](*_single((83.0, 83.0), (80.0, 80.0))).item()
    assert small > 3 * large


def test_fp16_input_is_finite() -> None:
    pred, gt, mask = _single((300.0, 250.0), (310.0, 240.0))
    for fn in (masked_log_l1_loss, masked_centered_giou_loss):
        assert torch.isfinite(fn(pred.half(), gt.half(), mask.half()))
