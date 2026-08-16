"""Shared utilities for SkyLens model code.

- `geo.py` — GPS <-> local ENU 변환 (`src/skylens_core/geo.ts`의 미러)
- `training_args.py` / `trainer.py` — 이중 헤드 학습용 HF `Trainer` 확장
- `metrics.py` — 세그 mIoU · 점 검출 P/R/F1 · CenterNet 히트맵 디코딩

`geo`는 torch/transformers 없이도 쓸 수 있어야 하므로 (클라이언트 좌표 규약
미러라 무거운 의존성을 끌어오면 곤란하다) 학습 관련 심볼은 **지연 import**한다.
`from skylens_model.utils import SkyLensTrainer` 는 그대로 동작한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .geo import Enu, GeoAnchor, Gps, enu_to_gps, gps_to_enu

if TYPE_CHECKING:
    from .metrics import (
        PointDetectionMetrics,
        SegmentationMetrics,
        build_compute_metrics,
        decode_heatmap_peaks,
    )
    from .callbacks import GracefulInterruptCallback, find_resume_checkpoint
    from .trainer import FreezeBackboneCallback, SkyLensTrainer
    from .training_args import SkyLensTrainingArguments

__all__ = [
    # geo (기존)
    "Gps",
    "GeoAnchor",
    "Enu",
    "gps_to_enu",
    "enu_to_gps",
    # training
    "SkyLensTrainingArguments",
    "SkyLensTrainer",
    "FreezeBackboneCallback",
    # 중단·재개
    "GracefulInterruptCallback",
    "find_resume_checkpoint",
    # metrics
    "decode_heatmap_peaks",
    "SegmentationMetrics",
    "PointDetectionMetrics",
    "build_compute_metrics",
]

_LAZY: dict[str, str] = {
    "SkyLensTrainingArguments": ".training_args",
    "SkyLensTrainer": ".trainer",
    "FreezeBackboneCallback": ".trainer",
    "GracefulInterruptCallback": ".callbacks",
    "find_resume_checkpoint": ".callbacks",
    "decode_heatmap_peaks": ".metrics",
    "SegmentationMetrics": ".metrics",
    "PointDetectionMetrics": ".metrics",
    "build_compute_metrics": ".metrics",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
