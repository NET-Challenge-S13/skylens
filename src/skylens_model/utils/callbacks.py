"""학습 중단·재개를 위한 Trainer 콜백.

SkyLens 학습은 중간에 꺼야 하는 일이 잦다(발표 준비, 공용 GPU 반납 등).
그래서 "언제 꺼도 잃는 게 없다"를 기본값으로 삼는다:

- `GracefulInterruptCallback` — Ctrl+C / SIGTERM 을 받으면 **현재 스텝 경계에서**
  체크포인트를 저장하고 정상 종료한다. 프로세스를 즉시 죽이지 않으므로
  optimizer/scheduler/RNG 상태까지 온전히 남고, 그대로 이어서 재개할 수 있다.
- `find_resume_checkpoint` — output_dir 에서 마지막 체크포인트를 찾아준다.

체크포인트에는 HF Trainer 가 모델 가중치뿐 아니라 optimizer·scheduler·
RNG state·TrainerState(global_step 등)를 함께 저장하므로, 재개하면
데이터 순서와 학습률 스케줄까지 끊긴 지점에서 이어진다.
"""

from __future__ import annotations

import logging
import signal
import threading
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)

__all__ = ["GracefulInterruptCallback", "find_resume_checkpoint"]


class GracefulInterruptCallback(TrainerCallback):
    """종료 시그널을 받으면 체크포인트를 저장하고 안전하게 멈춘다.

    동작
    ----
    1. 첫 번째 Ctrl+C(SIGINT) 또는 SIGTERM: 플래그만 세운다. 학습은 **현재 스텝을
       끝까지 마친 뒤** 저장하고 종료한다 (저장 도중 죽어서 체크포인트가 깨지는 걸 방지).
    2. 두 번째 Ctrl+C: 원래 핸들러로 되돌려 즉시 중단한다 (탈출구).

    노트북 커널 인터럽트도 SIGINT 로 들어오므로 동일하게 처리된다.

    주의: signal 핸들러는 메인 스레드에서만 등록할 수 있다. 서브 스레드에서
    학습을 돌리는 경우 등록을 건너뛰고 경고만 남긴다(학습 자체는 정상 진행).
    """

    def __init__(self, signals: tuple[int, ...] = (signal.SIGINT, signal.SIGTERM)) -> None:
        self._signals = signals
        self._interrupted = False
        self._previous: dict[int, Any] = {}
        self._installed = False

    # -- signal 처리 --------------------------------------------------------

    def _handler(self, signum: int, frame: Any) -> None:
        if self._interrupted:
            # 두 번째 신호 — 원래 핸들러로 되돌리고 즉시 전파한다.
            self._restore()
            signal.raise_signal(signum)
            return
        self._interrupted = True
        logger.warning(
            "[SkyLens] 종료 신호(%s) 수신 — 현재 스텝을 마치고 체크포인트를 저장한 뒤 종료한다. "
            "즉시 중단하려면 한 번 더 누를 것.",
            signal.Signals(signum).name,
        )

    def _install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            logger.warning("[SkyLens] 메인 스레드가 아니라 종료 시그널 훅을 걸 수 없다.")
            return
        for sig in self._signals:
            try:
                self._previous[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handler)
            except (ValueError, OSError):  # 플랫폼이 해당 시그널을 지원하지 않음
                self._previous.pop(sig, None)
        self._installed = True

    def _restore(self) -> None:
        if not self._installed:
            return
        for sig, prev in self._previous.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass
        self._previous.clear()
        self._installed = False

    # -- Trainer 훅 ---------------------------------------------------------

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> TrainerControl:
        self._interrupted = False
        self._install()
        return control

    def _maybe_stop(self, control: TrainerControl) -> TrainerControl:
        if self._interrupted:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> TrainerControl:
        return self._maybe_stop(control)

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> TrainerControl:
        # 평가가 긴 경우에도 끝나는 즉시 반응하도록.
        return self._maybe_stop(control)

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> TrainerControl:
        self._restore()
        if self._interrupted:
            logger.warning(
                "[SkyLens] 사용자 요청으로 중단됨 (global_step=%s). "
                "같은 output_dir 로 resume_from_checkpoint 를 주면 이어서 학습한다.",
                state.global_step,
            )
        return control

    @property
    def interrupted(self) -> bool:
        """중단 신호를 받아 멈췄는지 여부."""
        return self._interrupted


def find_resume_checkpoint(output_dir: str | Path) -> str | None:
    """`output_dir` 에서 마지막 체크포인트 경로를 찾는다. 없으면 None.

    `trainer.train(resume_from_checkpoint=find_resume_checkpoint(out))` 처럼 쓰면
    "있으면 이어서, 없으면 처음부터"가 자동으로 된다.
    """
    path = Path(output_dir)
    if not path.is_dir():
        return None
    # transformers 의 규약(checkpoint-<step>)을 그대로 따른다.
    from transformers.trainer_utils import get_last_checkpoint

    try:
        return get_last_checkpoint(str(path))
    except (FileNotFoundError, ValueError):
        return None
