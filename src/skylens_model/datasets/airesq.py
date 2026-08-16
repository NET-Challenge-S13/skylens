"""AIResQ -- thermal-only UAV search-and-rescue dataset.

9,788 drone-view thermal frames (up to 2048x1536) across varied weather and
terrain, including accident-scene / abnormal body postures. Thermal-only, which
is exactly the case symmetric modality dropout was introduced for: without it
this dataset could not be used at all (README §2.2).

Samples come back with ``has_rgb=False, has_thermal=True`` and a single-channel
``image``; the collator places it in channel 3 and zeroes the RGB planes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .base import Sample, SkyLensDatasetBase
from .sard import read_yolo_labels

__all__ = ["AIResQ"]

_IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".PNG")


class AIResQ(SkyLensDatasetBase):
    """AIResQ thermal person detection. ``danger_mask`` is always ``None``."""

    name = "AIResQ (thermal UAV SAR)"
    splits = ("train", "val", "test")
    availability = "account"
    homepage = "https://www.nature.com/articles/s41597-026-07663-9"
    download_note = (
        "AIResQ is a Scientific Data descriptor; the data itself sits in the\n"
        "repository named in the paper's 'Data Records' section (typically Figshare\n"
        "or Zenodo). Open the article, follow the DOI in Data Records, and if it\n"
        "resolves to Zenodo/Figshare the record IS wget-able:\n"
        "    curl -L -o airesq.zip https://zenodo.org/records/<id>/files/<file>?download=1\n"
        "Set that URL on AIResQ.direct_url to enable download=True."
    )
    expected_layout = (
        "<root>/\n"
        "  <split>/images/*.jpg   (thermal frames)\n"
        "  <split>/labels/*.txt   (YOLO: cls cx cy w h, normalised)"
    )

    #: Fill in once the Data Records DOI is resolved; enables ``download=True``.
    direct_url: str | None = None

    _SPLIT_DIRS = {"train": ("train",), "val": ("valid", "val"), "test": ("test",)}

    def _dirs(self) -> tuple[Path, Path]:
        for alias in self._SPLIT_DIRS[self.split]:
            img = self.root / alias / "images"
            if img.is_dir():
                return img, self.root / alias / "labels"
        return self.root / "images", self.root / "labels"

    def _check_exists(self) -> bool:
        img, _ = self._dirs()
        return img.is_dir() and any(p.suffix in _IMG_EXT for p in img.iterdir())

    def download(self) -> None:
        if self._check_exists():
            return
        if self.direct_url:
            from .base import download_and_extract

            download_and_extract(self.direct_url, self.root)
            if self._check_exists():
                return
        super().download()

    def _build_index(self) -> Sequence[Any]:
        img_dir, lab_dir = self._dirs()
        return [(p, lab_dir / f"{p.stem}.txt")
                for p in sorted(q for q in img_dir.iterdir() if q.suffix in _IMG_EXT)]

    def _load_sample(self, record: Any) -> Sample:
        img_path, lab_path = record
        thermal = self._read_thermal(img_path)
        h, w = thermal.shape[:2]
        return {
            "image": self._stack_rgb_thermal(None, thermal),  # (H, W, 1) in [0.1, 1]
            "has_rgb": False,
            "has_thermal": True,
            "danger_mask": None,
            "person_boxes": read_yolo_labels(lab_path, w, h),
        }
