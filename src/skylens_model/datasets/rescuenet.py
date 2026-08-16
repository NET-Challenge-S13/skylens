"""RescueNet -- post-disaster UAV semantic segmentation (Hurricane Michael).

4,494 high-resolution drone frames with 11-class pixel masks. This is the only
public source that grades *building damage* and flags *blocked roads* from a
low-altitude aerial viewpoint, which is exactly what the danger-zone head needs
(DATASETS.md §2).

Class mapping to the unified SkyLens schema
-------------------------------------------
=== ============================= ==================
id  RescueNet class               SkyLens
=== ============================= ==================
0   unlabeled                     255 ignore
1   water                         0 normal
2   building no damage            0 normal
3   building minor damage         0 normal
4   building major damage         2 collapse
5   building total destruction    2 collapse
6   vehicle                       0 normal
7   road clear                    0 normal
8   road blocked                  3 road_blocked
9   tree                          0 normal
10  pool                          0 normal
=== ============================= ==================

Rationale: only *major damage* and *total destruction* represent a structure a
commander must treat as a collapse hazard; minor damage is cosmetic and folding
it into class 2 would flood the positive set. "Road blocked" maps straight onto
the access-route question raised in IDEA.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .base import DangerClass, Sample, SkyLensDatasetBase

__all__ = ["RescueNetSegmentation", "RESCUENET_CLASSES", "RESCUENET_TO_SKYLENS"]

RESCUENET_CLASSES = (
    "unlabeled",
    "water",
    "building-no-damage",
    "building-medium-damage",
    "building-major-damage",
    "building-total-destruction",
    "vehicle",
    "road-clear",
    "road-blocked",
    "tree",
    "pool",
)

#: Lookup table indexed by the raw RescueNet label id.
RESCUENET_TO_SKYLENS = np.full(256, DangerClass.NORMAL, dtype=np.uint8)
RESCUENET_TO_SKYLENS[0] = DangerClass.IGNORE  # unlabeled
RESCUENET_TO_SKYLENS[4] = DangerClass.COLLAPSE  # building major damage
RESCUENET_TO_SKYLENS[5] = DangerClass.COLLAPSE  # building total destruction
RESCUENET_TO_SKYLENS[8] = DangerClass.ROAD_BLOCKED  # road blocked
# ids >= 11 are not part of the label set -> treat as ignore
RESCUENET_TO_SKYLENS[len(RESCUENET_CLASSES):] = DangerClass.IGNORE

_IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")


class RescueNetSegmentation(SkyLensDatasetBase):
    """RescueNet danger-zone segmentation. ``person_boxes`` is always ``None``."""

    name = "RescueNet"
    splits = ("train", "val", "test")
    availability = "manual"
    homepage = (
        "https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-"
        "Dataset-for-Semantic-Segmentation"
    )
    download_note = (
        "The authors distribute RescueNet through a Google Drive / Dropbox link\n"
        "published in the GitHub README (and mirrored on the paper's Figshare\n"
        "record). Large Drive files hit the virus-scan interstitial, so `requests`\n"
        "cannot fetch them reliably -- download in a browser, or try:\n"
        "    pip install gdown && gdown --folder <drive-folder-url>"
    )
    expected_layout = (
        "<root>/\n"
        "  train/train-org-img/*.jpg\n"
        "  train/train-label-img/*_lab.png\n"
        "  val/val-org-img/*.jpg      val/val-label-img/*_lab.png\n"
        "  test/test-org-img/*.jpg    test/test-label-img/*_lab.png"
    )

    def _dirs(self) -> tuple[Path, Path]:
        s = self.split
        base = self.root / s
        img, lab = base / f"{s}-org-img", base / f"{s}-label-img"
        if img.is_dir() and lab.is_dir():
            return img, lab
        # flat variant: <root>/<split>-org-img
        return self.root / f"{s}-org-img", self.root / f"{s}-label-img"

    def _check_exists(self) -> bool:
        img, lab = self._dirs()
        return img.is_dir() and lab.is_dir()

    def _build_index(self) -> Sequence[Any]:
        img_dir, lab_dir = self._dirs()
        labels = {p.stem: p for p in lab_dir.iterdir() if p.suffix in _IMG_EXT}
        pairs = []
        for img in sorted(p for p in img_dir.iterdir() if p.suffix in _IMG_EXT):
            lab = labels.get(f"{img.stem}_lab") or labels.get(img.stem)
            if lab is not None:
                pairs.append((img, lab))
        return pairs

    def _load_sample(self, record: Any) -> Sample:
        img_path, lab_path = record
        image = self._read_rgb(img_path)
        raw = self._read_mask(lab_path)
        mask = RESCUENET_TO_SKYLENS[raw]

        if mask.shape != image.shape[:2]:
            import cv2

            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        return {
            "image": image,
            "has_rgb": True,
            "has_thermal": False,
            "danger_mask": mask,
            "person_boxes": None,
        }
