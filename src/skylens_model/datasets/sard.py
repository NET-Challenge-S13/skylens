"""SARD -- Search And Rescue Dataset (UAV person detection).

1,980 drone frames / 6,525 person instances covering standing, sitting, walking,
running and -- crucially -- *exhausted / injured* postures. The median instance
covers under 0.1 % of the frame, which is the scale regime SkyLens actually
operates in (DATASETS.md §3, README §1.5).

Person-only dataset: ``danger_mask`` is always ``None`` (README §6.3).

Annotation formats
------------------
SARD circulates mainly as YOLO-format mirrors (Roboflow Universe, Kaggle). Both
the Roboflow per-split layout and a flat ``images/`` + ``labels/`` layout are
accepted; the loader auto-detects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .base import Sample, SkyLensDatasetBase
from .download import ManualDownloadRequired, download_kaggle

__all__ = ["SARD", "read_yolo_labels"]

_IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")


def read_yolo_labels(path: Path, width: int, height: int,
                     keep_classes: set[int] | None = None) -> np.ndarray:
    """Parse a YOLO ``.txt`` label file into ``(N, 4)`` xyxy pixel boxes.

    Each line is ``cls cx cy w h`` with all geometry normalised to ``[0, 1]``.
    ``keep_classes=None`` keeps every class (SARD is single-class).
    """
    if not path.is_file():
        return np.zeros((0, 4), dtype=np.float32)

    boxes: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if keep_classes is not None and cls not in keep_classes:
            continue
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


class SARD(SkyLensDatasetBase):
    """SARD person detection. Returns xyxy ``person_boxes``, ``danger_mask=None``."""

    name = "SARD (Search And Rescue Dataset)"
    splits = ("train", "val", "test")
    availability = "account"
    homepage = "https://universe.roboflow.com/search?q=SARD+search+and+rescue"
    #: Kaggle mirror slug tried by ``download=True`` (override if yours differs).
    kaggle_slug = "julienmeine/search-and-rescue-dataset"
    download_note = (
        "SARD has no official direct URL. Two automatable routes exist, both\n"
        "needing credentials:\n"
        "  1) Kaggle:   pip install kaggle; put ~/.kaggle/kaggle.json in place, then\n"
        "               kaggle datasets download -d <owner>/<slug> --unzip -p <root>\n"
        "     (pass the slug via SARD.kaggle_slug -- mirrors get renamed often)\n"
        "  2) Roboflow: pip install roboflow; export a YOLOv8 version with your API\n"
        "               key from https://universe.roboflow.com (search 'SARD').\n"
        "The original release is described in Sambolek & Ivasic-Kos, IEEE Access 2021."
    )
    expected_layout = (
        "<root>/\n"
        "  train/images/*.jpg   train/labels/*.txt   (YOLO: cls cx cy w h, normalised)\n"
        "  valid/images/...     valid/labels/...\n"
        "  test/images/...      test/labels/...\n"
        "or flat:  <root>/images/*.jpg  +  <root>/labels/*.txt"
    )

    #: Directory aliases per split, in preference order.
    _SPLIT_DIRS = {"train": ("train",), "val": ("valid", "val"), "test": ("test",)}

    def _dirs(self) -> tuple[Path, Path]:
        for alias in self._SPLIT_DIRS[self.split]:
            img, lab = self.root / alias / "images", self.root / alias / "labels"
            if img.is_dir():
                return img, lab
        return self.root / "images", self.root / "labels"

    def _check_exists(self) -> bool:
        img, _ = self._dirs()
        return img.is_dir() and any(p.suffix in _IMG_EXT for p in img.iterdir())

    def download(self) -> None:
        if self._check_exists():
            return
        try:
            download_kaggle(self.kaggle_slug, self.root)
        except ManualDownloadRequired as exc:
            raise ManualDownloadRequired(
                f"{self.manual_download_message()}\n\n  Kaggle attempt failed: {exc}"
            ) from exc
        if not self._check_exists():
            raise ManualDownloadRequired(self.manual_download_message())

    def _build_index(self) -> Sequence[Any]:
        img_dir, lab_dir = self._dirs()
        records = []
        for img in sorted(p for p in img_dir.iterdir() if p.suffix in _IMG_EXT):
            records.append((img, lab_dir / f"{img.stem}.txt"))
        return records

    def _load_sample(self, record: Any) -> Sample:
        img_path, lab_path = record
        image = self._read_rgb(img_path)
        h, w = image.shape[:2]
        return {
            "image": image,
            "has_rgb": True,
            "has_thermal": False,
            "danger_mask": None,
            "person_boxes": read_yolo_labels(lab_path, w, h),
        }
