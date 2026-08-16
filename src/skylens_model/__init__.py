"""SkyLens model package.

멀티드론 재난 인텔리전스 플랫폼 SkyLens의 AI·복원 모델 코드.
현장을 실시간 3D(Gaussian Splatting)로 복원하고, 그 위에 AI가 탐지한
위험구역·사람을 Depth Map 레이캐스팅으로 투영해 얹는다.

설계 철학과 결정 근거는 `src/skylens_model/README.md` 참조.

구성
----
- ``models``   : SkyLens 인지 모델 (transformers 표준 규약)
                 4채널(RGB+열화상) AutoBackbone 인코더 + UNet 디코더 +
                 이중 헤드(위험구역 세그멘테이션 / 사람 점 검출)
- ``datasets`` : torchvision 스타일 Dataset API + 타겟 인코딩 collator
- ``utils``    : HF Trainer 기반 학습·평가, ENU↔GPS 좌표 변환

무거운 의존성(torch/transformers)을 최상위에서 강제하지 않도록 서브모듈은
지연 로딩한다. ``from skylens_model import SkyLensConfig`` 처럼 바로 쓸 수 있다.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

# 이름 -> 실제 정의된 서브모듈 경로
_LAZY_EXPORTS: dict[str, str] = {
    # models
    "SkyLensConfig": "skylens_model.models.skylens",
    "SkyLensModel": "skylens_model.models.skylens",
    "SkyLensPreTrainedModel": "skylens_model.models.skylens",
    "SkyLensForDisasterPerception": "skylens_model.models.skylens",
    "SkyLensOutput": "skylens_model.models.skylens",
    # utils
    "SkyLensTrainer": "skylens_model.utils.trainer",
    "SkyLensTrainingArguments": "skylens_model.utils.training_args",
    # datasets
    "SkyLensCollator": "skylens_model.utils.collate",
}

__all__ = ["__version__", *sorted(_LAZY_EXPORTS)]


def __getattr__(name: str) -> Any:
    """PEP 562 지연 로딩 — 실제로 참조될 때만 서브모듈을 import한다."""
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # 정적 분석기용 — 런타임에는 실행되지 않는다
    from skylens_model.utils.collate import SkyLensCollator
    from skylens_model.models.skylens import (
        SkyLensConfig,
        SkyLensForDisasterPerception,
        SkyLensModel,
        SkyLensOutput,
        SkyLensPreTrainedModel,
    )
    from skylens_model.utils.trainer import SkyLensTrainer
    from skylens_model.utils.training_args import SkyLensTrainingArguments
