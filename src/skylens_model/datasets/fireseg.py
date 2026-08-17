"""Fire Segmentation -- YOLO-seg polygon annotations for aerial/ground fire.

1,348 RGB frames (1,146 train / 202 val) with Ultralytics YOLO segmentation
polygons for a single class (``fire``). Polygons are rasterised into the unified
danger schema as :attr:`DangerClass.FIRE`, giving the danger-zone head a second
fire source next to FLAME (DATASETS.md §2).

Label files with no rows are *negative* samples (no fire in frame) and yield an
all-``normal`` mask rather than ``None`` -- "there is no fire here" is a training
signal, not missing data.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .base import DangerClass, Sample, SkyLensDatasetBase, download_hf

__all__ = ["FireSegmentation", "read_yolo_polygons"]

_IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")


def read_yolo_polygons(path: str | os.PathLike) -> list[np.ndarray]:
    """YOLO-seg 라벨 파일을 정규화 폴리곤 리스트로 읽는다.

    각 줄은 ``<class> x1 y1 x2 y2 ... xn yn`` 이고 좌표는 [0, 1] 정규화값이다.
    반환값은 ``(n_points, 2)`` float32 배열의 리스트. 파일이 없거나 비어 있으면
    빈 리스트 (음성 샘플).
    """
    path = Path(path)
    if not path.is_file():
        return []

    polys: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:  # class + 최소 3점
            continue
        coords = np.asarray(parts[1:], dtype=np.float32)
        coords = coords[: len(coords) // 2 * 2]
        polys.append(coords.reshape(-1, 2))
    return polys


class FireSegmentation(SkyLensDatasetBase):
    """YOLO-seg 화재 폴리곤 → ``danger_mask`` {0 normal, 1 fire}.

    ``person_boxes`` 는 항상 ``None`` (세그 전용 데이터셋, README §6.3).
    """

    name = "Fire Segmentation (YOLO-seg polygons)"
    splits = ("train", "val")
    availability = "auto"
    homepage = "https://huggingface.co/datasets/sreeharivp23/fire-segmentation-dataset"
    hf_repo_id = "sreeharivp23/fire-segmentation-dataset"
    download_note = (
        "Public, non-gated HuggingFace dataset -- `download=True` fetches it via\n"
        "`huggingface_hub.snapshot_download`. The repo is served through the Xet\n"
        "backend, which throttles aggressively (HTTP 429). Disable it to fall back\n"
        "to plain HTTP range downloads:\n"
        "    HF_HUB_DISABLE_XET=1 python -c '...'\n"
        "`download()` sets that variable itself for the duration of the fetch."
    )
    expected_layout = (
        "<root>/\n"
        "  data.yaml                (names: {0: fire})\n"
        "  images/train/*.jpg       images/val/*.jpg\n"
        "  labels/train/*.txt       labels/val/*.txt\n"
        "                           (YOLO-seg: cls x1 y1 x2 y2 ... normalised;\n"
        "                            an empty file = no fire in that frame)"
    )

    def _dirs(self) -> tuple[Path, Path]:
        return self.root / "images" / self.split, self.root / "labels" / self.split

    def _check_exists(self) -> bool:
        img, _ = self._dirs()
        return img.is_dir() and any(p.suffix in _IMG_EXT for p in img.iterdir())

    def download(self) -> None:
        if self._check_exists():
            return
        prev = os.environ.get("HF_HUB_DISABLE_XET")
        os.environ["HF_HUB_DISABLE_XET"] = "1"  # Xet 백엔드 429 우회
        try:
            download_hf(self.hf_repo_id, self.root)
        finally:
            if prev is None:
                os.environ.pop("HF_HUB_DISABLE_XET", None)
            else:
                os.environ["HF_HUB_DISABLE_XET"] = prev
        if self._check_exists():
            return
        super().download()

    def _build_index(self) -> Sequence[Any]:
        img_dir, lab_dir = self._dirs()
        return [(p, lab_dir / f"{p.stem}.txt")
                for p in sorted(q for q in img_dir.iterdir() if q.suffix in _IMG_EXT)]

    def _load_sample(self, record: Any) -> Sample:
        img_path, lab_path = record
        image = self._read_rgb(img_path)
        h, w = image.shape[:2]

        mask = np.full((h, w), DangerClass.NORMAL, dtype=np.uint8)
        polys = read_yolo_polygons(lab_path)
        if polys:
            import cv2

            pts = [np.round(p * np.array([w, h], np.float32)).astype(np.int32)
                   for p in polys]
            cv2.fillPoly(mask, pts, DangerClass.FIRE)

        return {
            "image": image,
            "has_rgb": True,
            "has_thermal": False,
            "danger_mask": mask,
            "person_boxes": None,
        }
