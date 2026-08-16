"""`SkyLensForDisasterPerception` 학습용 HF `Trainer` 서브클래스.

설계 메모
---------
- **loss는 모델이 계산한다.** Trainer는 그걸 받아쓰기만 한다. 배치에 따라
  세그 GT만 있거나 점 GT만 있을 수 있고(README §6.3), 어느 헤드의 loss를 켤지는
  모델이 GT 유무로 판단한다. Trainer가 관여하면 책임이 이중화된다.
- **개별 컴포넌트 로깅이 중요하다.** 총 loss만 보면 이번 스텝이 세그를 학습한
  건지 점 검출을 학습한 건지 알 수 없다. `loss_dict`를 버퍼에 모아
  `log()` 시점에 평균을 흘려보낸다.
- **평가에서 로짓을 다 모으지 않는다.** `(B, num_classes, H, W)` float 로짓을
  평가셋 전체로 누적하면 수 GB다. argmax된 클래스 맵과 디코딩된 점만 남긴다.

이 모듈은 `models/`를 import하지 않는다 (순환 import 회피). 모델 계약은
`SkyLensOutput`의 필드 이름으로만 알고 있으면 충분하다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from transformers import Trainer, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState

from .metrics import decode_heatmap_peaks
from .training_args import SkyLensTrainingArguments

if TYPE_CHECKING:  # 런타임 import 금지 — utils -> models 의존을 만들지 않는다
    from transformers.training_args import TrainingArguments

__all__ = ["SkyLensTrainer", "FreezeBackboneCallback"]

# 백본으로 간주할 속성 이름 후보 (모델 구현이 확정되기 전까지의 완충)
_BACKBONE_ATTRS = ("backbone", "encoder", "feature_extractor")


# --------------------------------------------------------------------------
# 백본 동결 워밍업
# --------------------------------------------------------------------------


class FreezeBackboneCallback(TrainerCallback):
    """학습 초반 `freeze_backbone_epochs` 동안 백본을 얼린다 (README §6.1).

    난수 초기화된 이중 헤드가 만드는 큰 그래디언트가 ImageNet/VisDrone
    프리트레인 백본을 초반에 훼손하는 것을 막는다. 지정 에폭이 지나면
    자동으로 해제된다.

    콜백으로 구현한 이유: 에폭 경계 훅(`on_epoch_begin`)이 이미 있고,
    `Trainer` 본체를 건드리지 않아도 되며, 끄고 싶으면 콜백만 빼면 된다.
    """

    def __init__(self, freeze_epochs: float) -> None:
        self.freeze_epochs = float(freeze_epochs)
        self._frozen = False

    # 내부 헬퍼 -----------------------------------------------------------
    @staticmethod
    def _find_backbone(model: nn.Module) -> nn.Module | None:
        base = getattr(model, "module", model)  # DDP 래핑 해제
        for name in _BACKBONE_ATTRS:
            mod = getattr(base, name, None)
            if isinstance(mod, nn.Module):
                return mod
        return None

    def _set_requires_grad(self, model: nn.Module, flag: bool) -> bool:
        backbone = self._find_backbone(model)
        if backbone is None:
            return False
        for p in backbone.parameters():
            p.requires_grad = flag
        return True

    # 훅 -----------------------------------------------------------------
    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        if model is None or self.freeze_epochs <= 0:
            return
        should_freeze = float(state.epoch or 0.0) < self.freeze_epochs
        if should_freeze and not self._frozen:
            if self._set_requires_grad(model, False):
                self._frozen = True
        elif not should_freeze and self._frozen:
            self._set_requires_grad(model, True)
            self._frozen = False

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        # 학습이 동결 상태로 끝났더라도 모델은 정상 상태로 돌려놓는다.
        if model is not None and self._frozen:
            self._set_requires_grad(model, True)
            self._frozen = False


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------


class SkyLensTrainer(Trainer):
    """이중 헤드 + 결손 GT를 다루는 `Trainer`."""

    args: SkyLensTrainingArguments

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # loss 컴포넌트 누적 버퍼: {이름: [값, ...]}
        self._loss_log_buffer: dict[str, list[float]] = {}

        freeze_epochs = float(getattr(self.args, "freeze_backbone_epochs", 0.0) or 0.0)
        if freeze_epochs > 0 and not any(
            isinstance(cb, FreezeBackboneCallback) for cb in self.callback_handler.callbacks
        ):
            self.add_callback(FreezeBackboneCallback(freeze_epochs))

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------
    def compute_loss(  # type: ignore[override]
        self,
        model: nn.Module,
        inputs: Mapping[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """모델이 계산한 loss를 그대로 쓰되 `loss_dict`를 로깅 버퍼에 쌓는다.

        transformers 4.57의 시그니처는
        `compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None)`
        인데, 이 인자는 버전마다 늘고 줄어서 `**kwargs`로 흡수한다. 우리는
        모델 내부에서 이미 정규화된 loss를 받으므로 `num_items_in_batch`를
        따로 쓸 일이 없다.
        """
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]

        if loss is None:
            # GT가 전무한 배치 — 그래프에 연결된 0을 만들어 backward가 깨지지 않게 한다.
            loss = self._zero_loss(model, outputs)
        else:
            self._accumulate_loss_components(outputs)

        return (loss, outputs) if return_outputs else loss

    def _zero_loss(self, model: nn.Module, outputs: Any) -> torch.Tensor:
        """어떤 파라미터와도 연결된 0 텐서 (optimizer/DDP가 싫어하지 않도록)."""
        ref = None
        for name in ("danger_logits", "person_heatmap", "person_wh"):
            t = getattr(outputs, name, None)
            if isinstance(t, torch.Tensor):
                ref = t
                break
        if ref is not None and ref.requires_grad:
            return ref.sum() * 0.0
        device = next(model.parameters()).device if any(True for _ in model.parameters()) else "cpu"
        dtype = ref.dtype if isinstance(ref, torch.Tensor) else torch.float32
        return torch.zeros((), device=device, dtype=dtype, requires_grad=True)

    def _accumulate_loss_components(self, outputs: Any) -> None:
        if not getattr(self.args, "log_loss_components", True):
            return
        loss_dict = getattr(outputs, "loss_dict", None)
        if not isinstance(loss_dict, Mapping):
            return
        for key, value in loss_dict.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    continue
                value = float(value.detach().float().mean().item())
            self._loss_log_buffer.setdefault(f"loss_{key}", []).append(float(value))

    # ------------------------------------------------------------------
    # 로깅
    # ------------------------------------------------------------------
    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        """누적 loss 컴포넌트의 평균을 병합한 뒤 상위 구현에 넘긴다.

        시그니처가 `log(self, logs)` (≤4.46) → `log(self, logs, start_time=None)`
        (≥4.47) 로 바뀌었으므로 `*args/**kwargs`로 그대로 전달한다.
        """
        if self._loss_log_buffer:
            for key, values in self._loss_log_buffer.items():
                if values:
                    logs[key] = round(sum(values) / len(values), 6)
            self._loss_log_buffer.clear()
        try:
            super().log(logs, *args, **kwargs)
        except TypeError:
            # 구버전: start_time 인자를 받지 않는다.
            super().log(logs)

    # ------------------------------------------------------------------
    # 평가
    # ------------------------------------------------------------------
    def prediction_step(  # type: ignore[override]
        self,
        model: nn.Module,
        inputs: Mapping[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor | None, Any, Any]:
        """평가에 필요한 최소한만 반환한다.

        Returns:
            `(loss, predictions, labels)`
            - `predictions = (seg_pred (B,H,W) int16, detections (B,K,5) float32)`
            - `labels      = (seg_gt (B,H,W) int16, gt_points (B,K,3) float32)`

            `gt_points`의 3번째 성분은 valid 플래그다. 헤드별 분리 학습 때문에
            한쪽 GT가 없는 배치가 정상이므로, 없는 쪽은 seg는 ignore_index로,
            점은 valid=0으로 채워 shape을 항상 일정하게 유지한다
            (Trainer의 배치 누적이 고정 shape을 요구한다).
        """
        has_labels = any(
            inputs.get(k) is not None
            for k in ("danger_labels", "person_heatmap")
        )
        inputs = self._prepare_inputs(dict(inputs))

        with torch.no_grad():
            outputs = model(**inputs)
            loss = getattr(outputs, "loss", None)
            if loss is not None:
                loss = loss.detach()
                self._accumulate_loss_components(outputs)

        if prediction_loss_only or not has_labels:
            return (loss, None, None)

        args = self.args
        stride = int(getattr(args, "person_head_stride", 4))
        k = int(getattr(args, "eval_max_detections", 100))
        thr = float(getattr(args, "eval_score_threshold", 0.3))
        ignore_index = int(getattr(args, "seg_ignore_index", 255))

        # --- 세그: 로짓 대신 argmax된 클래스 맵만 남긴다 (메모리) ---
        danger_gt = inputs.get("danger_labels")
        logits = getattr(outputs, "danger_logits", None)
        if logits is not None and danger_gt is not None:
            seg_pred = logits.argmax(dim=1).to(torch.int16)
            seg_gt = danger_gt.to(torch.int16)
        else:
            ref = logits if logits is not None else danger_gt
            shape = (
                (ref.size(0), ref.size(-2), ref.size(-1))
                if ref is not None
                else (1, 1, 1)
            )
            dev = ref.device if ref is not None else loss.device if loss is not None else "cpu"
            seg_pred = torch.full(shape, ignore_index, dtype=torch.int16, device=dev)
            seg_gt = seg_pred.clone()

        # --- 점 검출: 히트맵을 즉시 디코딩해 (B,K,5)로 축약 ---
        pred_hm = getattr(outputs, "person_heatmap", None)
        gt_hm = inputs.get("person_heatmap")
        if pred_hm is not None and gt_hm is not None:
            det = decode_heatmap_peaks(
                pred_hm,
                getattr(outputs, "person_wh", None),
                k=k,
                threshold=thr,
                stride=stride,
            ).to(torch.float32)
            # GT 히트맵의 정점(가우시안 peak == 1.0)이 곧 GT 중심점이다.
            gt_dec = decode_heatmap_peaks(gt_hm, None, k=k, threshold=0.99, stride=stride)
            gt_points = torch.stack(
                [gt_dec[..., 0], gt_dec[..., 1], (gt_dec[..., 4] > 0).float()], dim=-1
            ).to(torch.float32)
        else:
            b = seg_pred.size(0)
            dev = seg_pred.device
            det = torch.zeros((b, k, 5), dtype=torch.float32, device=dev)
            gt_points = torch.zeros((b, k, 3), dtype=torch.float32, device=dev)

        return (loss, (seg_pred, det), (seg_gt, gt_points))
