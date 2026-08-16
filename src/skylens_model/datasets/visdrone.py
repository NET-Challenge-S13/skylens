"""VisDrone2019-DET -- drone-view object detection.

Used as the *backbone warm-up* stage of README §6.1: ImageNet (ground-level
photos) to disaster imagery is too big a domain jump, so VisDrone bridges it
with "drone viewpoint + very small people" at ~70 instances per image.

Only the person-ish categories are kept, per the task spec:

=== ============ ======
id  name         kept
=== ============ ======
0   ignored      no
1   pedestrian   yes
2   people       yes
3   bicycle      no
4   car          no
5   van          no
6   truck        no
7   tricycle     no
8   awning-tric. no
9   bus          no
10  motor        no
11  others       no
=== ============ ======

("pedestrian" is an upright human; "people" covers humans in other poses --
which is precisely the exhausted/prone case SkyLens cares about, so both are
merged into a single ``person`` class.)

Annotation format: one CSV line per object,
``<x>,<y>,<w>,<h>,<score>,<category>,<truncation>,<occlusion>`` with ``x,y`` the
top-left corner in pixels. Rows with ``score == 0`` are ignored regions and are
dropped.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .base import ManualDownloadRequired, Sample, SkyLensDatasetBase, download_hf

__all__ = ["VisDronePerson", "VISDRONE_CATEGORIES", "VISDRONE_PERSON_IDS",
           "read_visdrone_annotation"]

VISDRONE_CATEGORIES = (
    "ignored", "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor", "others",
)
#: Categories folded into the single SkyLens ``person`` class.
VISDRONE_PERSON_IDS = frozenset({1, 2})

_IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")


def read_visdrone_annotation(path: Path,
                             keep: frozenset[int] = VISDRONE_PERSON_IDS,
                             drop_ignored: bool = True) -> np.ndarray:
    """Parse a VisDrone ``.txt`` annotation into ``(N, 4)`` xyxy pixel boxes."""
    if not path.is_file():
        return np.zeros((0, 4), dtype=np.float32)

    boxes: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) < 6:
            continue
        try:
            x, y, w, h = (float(v) for v in parts[:4])
            score, category = int(float(parts[4])), int(float(parts[5]))
        except ValueError:
            continue
        if drop_ignored and score == 0:
            continue
        if category not in keep:
            continue
        if w > 0 and h > 0:
            boxes.append([x, y, x + w, y + h])

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


class VisDronePerson(SkyLensDatasetBase):
    """VisDrone2019-DET restricted to pedestrian/people boxes.

    ``danger_mask`` is always ``None`` -- detection-only dataset.
    """

    name = "VisDrone2019-DET (person subset)"
    splits = ("train", "val", "test")
    availability = "account"
    homepage = "https://github.com/VisDrone/VisDrone-Dataset"
    #: HuggingFace mirror attempted by ``download=True``. VERIFIED to exist as a
    #: dataset repo; it is a FiftyOne export, so the on-disk layout differs from
    #: ``expected_layout`` and may need a one-off reorganisation. Other candidates:
    #: ``lanlanlan23/VisDrone2019``, ``asadistic/VisDrone_2019_coco_format``.
    hf_repo_id = "Voxel51/VisDrone2019-DET"
    download_note = (
        "The official VisDrone release is hosted on Google Drive / OneDrive links\n"
        "from https://github.com/VisDrone/VisDrone-Dataset, which are not reliably\n"
        "wget-able (Drive virus-scan interstitial). Options:\n"
        "  1) HuggingFace mirror -- several exist; set VisDronePerson.hf_repo_id and\n"
        "     run with download=True (may need `huggingface-cli login`).\n"
        "  2) pip install gdown && gdown <drive-file-id>\n"
        "  3) Download the four DET zips manually in a browser."
    )
    expected_layout = (
        "<root>/\n"
        "  VisDrone2019-DET-train/images/*.jpg\n"
        "  VisDrone2019-DET-train/annotations/*.txt\n"
        "  VisDrone2019-DET-val/{images,annotations}/\n"
        "  VisDrone2019-DET-test-dev/{images,annotations}/"
    )

    _SPLIT_DIRS = {
        "train": ("VisDrone2019-DET-train", "train"),
        "val": ("VisDrone2019-DET-val", "val"),
        "test": ("VisDrone2019-DET-test-dev", "VisDrone2019-DET-test-challenge", "test"),
    }

    def _dirs(self) -> tuple[Path, Path]:
        for alias in self._SPLIT_DIRS[self.split]:
            base = self.root / alias
            if (base / "images").is_dir():
                return base / "images", base / "annotations"
        base = self.root / self._SPLIT_DIRS[self.split][0]
        return base / "images", base / "annotations"

    #: FiftyOne export (HF 미러 Voxel51/VisDrone2019-DET) 의 어노테이션 파일.
    _FIFTYONE_MANIFEST = "samples.json"
    #: FiftyOne 라벨 중 사람으로 취급할 것.
    _FIFTYONE_PERSON_LABELS = frozenset({"pedestrians", "people"})

    def _fiftyone_manifest(self) -> Path | None:
        """FiftyOne 배치이면 manifest 경로를, 아니면 None."""
        p = self.root / self._FIFTYONE_MANIFEST
        return p if p.is_file() and (self.root / "data").is_dir() else None

    def _check_exists(self) -> bool:
        if self._fiftyone_manifest() is not None:
            return True
        img, _ = self._dirs()
        return img.is_dir() and any(p.suffix in _IMG_EXT for p in img.iterdir())

    def download(self) -> None:
        if self._check_exists():
            return
        try:
            download_hf(self.hf_repo_id, self.root)
        except Exception as exc:
            raise ManualDownloadRequired(
                f"{self.manual_download_message()}\n\n"
                f"  HuggingFace attempt ({self.hf_repo_id}) failed: {exc}"
            ) from exc
        if not self._check_exists():
            raise ManualDownloadRequired(self.manual_download_message())

    def _build_index_fiftyone(self, manifest: Path) -> Sequence[Any]:
        """FiftyOne export 를 (이미지경로, xyxy박스) 레코드로 편다.

        bounding_box 는 [x, y, w, h] 정규화 좌표(좌상단 기준)이므로
        metadata 의 width/height 를 곱해 절대 xyxy 로 바꾼다.
        스플릿은 sample 의 ``tags`` 로 구분된다.
        """
        import json

        with manifest.open(encoding="utf-8") as fh:
            samples = json.load(fh)["samples"]

        records = []
        for s in samples:
            if self.split not in (s.get("tags") or []):
                continue
            meta = s.get("metadata") or {}
            w, h = meta.get("width"), meta.get("height")
            boxes = []
            for det in (s.get("ground_truth") or {}).get("detections") or []:
                if det.get("label") not in self._FIFTYONE_PERSON_LABELS:
                    continue
                bb = det.get("bounding_box")
                if not bb or not w or not h:
                    continue
                x, y, bw, bh = bb
                boxes.append([x * w, y * h, (x + bw) * w, (y + bh) * h])
            records.append(
                (self.root / s["filepath"], np.asarray(boxes, dtype=np.float32).reshape(-1, 4))
            )
        return records

    def _build_index(self) -> Sequence[Any]:
        manifest = self._fiftyone_manifest()
        if manifest is not None:
            return self._build_index_fiftyone(manifest)
        img_dir, ann_dir = self._dirs()
        return [(p, ann_dir / f"{p.stem}.txt")
                for p in sorted(q for q in img_dir.iterdir() if q.suffix in _IMG_EXT)]

    def _load_sample(self, record: Any) -> Sample:
        img_path, ann = record
        boxes = ann if isinstance(ann, np.ndarray) else read_visdrone_annotation(ann)
        return {
            "image": self._read_rgb(img_path),
            "has_rgb": True,
            "has_thermal": False,
            "danger_mask": None,
            "person_boxes": boxes,
        }
