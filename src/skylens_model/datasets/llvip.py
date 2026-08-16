"""LLVIP -- visible/infrared paired pedestrian dataset (all night-time).

16,836 strictly registered and cropped RGB/IR pairs from fixed cameras, with
VOC-XML person boxes. Two roles in SkyLens:

* the largest clean source for validating that 4-channel early fusion actually
  trains (README §2.1), and
* a night-time person-detection prior -- every frame is dark, which matches the
  night SAR scenario.

Caveat: fixed CCTV viewpoint, not aerial. Treat as structural pre-training only
(DATASETS.md §1).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .base import ManualDownloadRequired, Sample, SkyLensDatasetBase, download_hf

__all__ = ["LLVIP", "read_voc_person_boxes"]

_IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
_PERSON_NAMES = {"person", "people", "pedestrian"}


def read_voc_person_boxes(path: Path,
                          names: set[str] = _PERSON_NAMES) -> np.ndarray:
    """Parse Pascal-VOC XML into ``(N, 4)`` xyxy pixel boxes for person classes."""
    if not path.is_file():
        return np.zeros((0, 4), dtype=np.float32)

    try:
        root = ET.parse(str(path)).getroot()
    except ET.ParseError:
        return np.zeros((0, 4), dtype=np.float32)

    boxes: list[list[float]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if names and name not in names:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            x1 = float(bb.findtext("xmin"))
            y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax"))
            y2 = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


class LLVIP(SkyLensDatasetBase):
    """LLVIP RGB+IR pairs with person boxes.

    ``modality`` selects what ends up in ``image``:

    ``"rgbt"`` (default)
        4-channel stack, ``has_rgb=has_thermal=True``.
    ``"rgb"`` / ``"thermal"``
        single-modality, for feeding the dropout modes of README §2.2 directly.
    """

    name = "LLVIP"
    splits = ("train", "test")
    availability = "auto"
    homepage = "https://bupt-ai-cz.github.io/LLVIP/"
    #: VERIFIED public (non-gated) mirror: a single ~4 GB `LLVIP.zip` plus
    #: `coco_annotations.7z`. Snapshotting it still leaves you to unzip.
    hf_repo_id = "jsonhash/LLVIP"
    download_note = (
        "The project page links Google Drive / Baidu / OneDrive copies, none of\n"
        "which are wget-able. The practical route is the HuggingFace mirror:\n"
        "    huggingface-cli download jsonhash/LLVIP --repo-type dataset \\\n"
        "        --local-dir <root>\n"
        "    unzip <root>/LLVIP.zip -d <root>\n"
        "download=True performs the snapshot but NOT the unzip (the repo ships one\n"
        "big zip). Other mirrors: Frencis/LLVIP_RGBT, hieupth/llvip."
    )
    expected_layout = (
        "<root>/\n"
        "  visible/train/*.jpg     visible/test/*.jpg\n"
        "  infrared/train/*.jpg    infrared/test/*.jpg\n"
        "  Annotations/*.xml       (Pascal VOC, class 'person')"
    )

    def __init__(self, root, split: str = "train", transforms=None,
                 download: bool = False, modality: str = "rgbt") -> None:
        if modality not in ("rgbt", "rgb", "thermal"):
            raise ValueError(f"modality must be rgbt|rgb|thermal, got {modality!r}")
        self.modality = modality
        super().__init__(root, split=split, transforms=transforms, download=download)

    def _dirs(self) -> tuple[Path, Path, Path]:
        return (
            self.root / "visible" / self.split,
            self.root / "infrared" / self.split,
            self.root / "Annotations",
        )

    def _check_exists(self) -> bool:
        vis, ir, ann = self._dirs()
        need_vis = self.modality in ("rgbt", "rgb")
        need_ir = self.modality in ("rgbt", "thermal")
        return ((not need_vis or vis.is_dir())
                and (not need_ir or ir.is_dir())
                and ann.is_dir())

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

    def _build_index(self) -> Sequence[Any]:
        vis, ir, ann = self._dirs()
        primary = vis if self.modality in ("rgbt", "rgb") else ir
        records = []
        for img in sorted(p for p in primary.iterdir() if p.suffix in _IMG_EXT):
            ir_path = next((c for c in (ir / (img.stem + e) for e in _IMG_EXT)
                            if c.is_file()), None)
            vis_path = next((c for c in (vis / (img.stem + e) for e in _IMG_EXT)
                             if c.is_file()), None)
            if self.modality == "rgbt" and (ir_path is None or vis_path is None):
                continue
            records.append((vis_path, ir_path, ann / f"{img.stem}.xml"))
        return records

    def _load_sample(self, record: Any) -> Sample:
        vis_path, ir_path, ann_path = record
        rgb = self._read_rgb(vis_path) if self.modality in ("rgbt", "rgb") else None
        thermal = self._read_thermal(ir_path) if self.modality in ("rgbt", "thermal") else None

        if rgb is not None and thermal is not None and thermal.shape != rgb.shape[:2]:
            import cv2

            thermal = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        image = (rgb if thermal is None
                 else self._stack_rgb_thermal(rgb, thermal))
        return {
            "image": image,
            "has_rgb": rgb is not None,
            "has_thermal": thermal is not None,
            "danger_mask": None,
            "person_boxes": read_voc_person_boxes(ann_path),
        }
