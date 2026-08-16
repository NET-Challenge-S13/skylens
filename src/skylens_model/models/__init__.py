"""SkyLens 모델 패키지 — `skylensnet`(인지) · `skylensgsplat`(3DGS 복원). 심볼은 지연 로딩한다."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_LAZY: dict[str, str] = {
    "SkyLensConfig": ".skylensnet",
    "SkyLensModel": ".skylensnet",
    "SkyLensPreTrainedModel": ".skylensnet",
    "SkyLensForDisasterPerception": ".skylensnet",
    "SkyLensOutput": ".skylensnet",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from .skylensnet import (
        SkyLensConfig,
        SkyLensForDisasterPerception,
        SkyLensModel,
        SkyLensOutput,
        SkyLensPreTrainedModel,
    )
