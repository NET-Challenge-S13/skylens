"""torchvision-style dataset loaders for SkyLens perception training.

No data lives in this repository. Each class points at an already-downloaded
copy under ``root`` and, where the source permits it, can fetch it itself with
``download=True``. See ``README.md`` for the per-dataset availability verdicts
and the exact directory layout each loader expects.

Every ``__getitem__`` returns the same dict::

    {
      "image":        np.ndarray (H, W, C) uint8 | float32,
      "has_rgb":      bool,
      "has_thermal":  bool,
      "danger_mask":  np.ndarray (H, W) uint8 | None,   # 0..3, 255 = ignore
      "person_boxes": np.ndarray (N, 4) float32 | None, # xyxy, pixels
    }

:class:`SkyLensCollator` turns a list of those into the model's batch dict.
"""

from __future__ import annotations

from .airesq import AIResQ
from .base import (
    DANGER_CLASS_NAMES,
    IGNORE_INDEX,
    NUM_DANGER_CLASSES,
    DangerClass,
    Sample,
    SkyLensDatasetBase,
)
from .collate import SkyLensCollator, draw_gaussian, gaussian2d, gaussian_radius
from .download import (
    ManualDownloadRequired,
    download_and_extract,
    download_hf,
    download_kaggle,
    download_url,
)
from .flame import Flame3Pairs, FlameSegmentation
from .llvip import LLVIP
from .rescuenet import RESCUENET_CLASSES, RESCUENET_TO_SKYLENS, RescueNetSegmentation
from .sard import SARD
from .visdrone import VISDRONE_PERSON_IDS, VisDronePerson

__all__ = [
    # schema
    "DangerClass",
    "DANGER_CLASS_NAMES",
    "NUM_DANGER_CLASSES",
    "IGNORE_INDEX",
    "Sample",
    "SkyLensDatasetBase",
    # datasets -- segmentation head
    "FlameSegmentation",
    "RescueNetSegmentation",
    "RESCUENET_CLASSES",
    "RESCUENET_TO_SKYLENS",
    # datasets -- point-detection head
    "SARD",
    "VisDronePerson",
    "VISDRONE_PERSON_IDS",
    "AIResQ",
    # datasets -- RGB+thermal pairs
    "Flame3Pairs",
    "LLVIP",
    # batching
    "SkyLensCollator",
    "gaussian_radius",
    "gaussian2d",
    "draw_gaussian",
    # download utils
    "ManualDownloadRequired",
    "download_and_extract",
    "download_url",
    "download_hf",
    "download_kaggle",
]
