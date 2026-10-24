"""세그 soft Dice loss 경계 조건 검증."""

from __future__ import annotations

import torch

from skylens_model.models.skylensnet.modeling_skylensnet import masked_soft_dice_loss


def test_all_ignore_is_finite() -> None:
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    labels = torch.full((2, 8, 8), 255, dtype=torch.long)
    loss = masked_soft_dice_loss(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_missing_class_is_finite() -> None:
    logits = torch.randn(2, 4, 8, 8)
    labels = torch.randint(0, 2, (2, 8, 8))  # 클래스 2·3 GT 없음
    labels[0, 0, :] = 255
    assert torch.isfinite(masked_soft_dice_loss(logits, labels))


def test_perfect_prediction_is_near_zero() -> None:
    labels = torch.randint(0, 3, (2, 16, 16))  # 클래스 3 GT 없음
    logits = torch.nn.functional.one_hot(labels, 4).permute(0, 3, 1, 2).float() * 50.0
    labels[:, :4, :4] = 255
    logits[:, :, :4, :4] = torch.randn(2, 4, 4, 4) * 50.0  # ignore 영역 예측은 무관해야 한다
    assert masked_soft_dice_loss(logits, labels).item() < 1e-3


if __name__ == "__main__":
    test_all_ignore_is_finite()
    test_missing_class_is_finite()
    test_perfect_prediction_is_near_zero()
    print("ok")
