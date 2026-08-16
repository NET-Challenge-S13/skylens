"""학습 중단·재개용 Trainer 콜백."""

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
    """종료 신호를 받으면 현재 스텝을 마친 뒤 저장하고 멈춘다. 두 번째 신호는 즉시 중단."""

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
        return self._interrupted


def find_resume_checkpoint(output_dir: str | Path) -> str | None:
    """`output_dir` 의 마지막 체크포인트 경로. 없으면 None."""
    path = Path(output_dir)
    if not path.is_dir():
        return None
    from transformers.trainer_utils import get_last_checkpoint

    try:
        return get_last_checkpoint(str(path))
    except (FileNotFoundError, ValueError):
        return None
