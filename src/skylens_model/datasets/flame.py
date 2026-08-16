"""FLAME / FLAME 3 -- aerial wildfire datasets (IEEE DataPort).

* :class:`FlameSegmentation` -- the FLAME "Fire Segmentation" subset: 2,003
  drone frames with binary fire masks. Feeds the danger-zone head with
  ``fire`` (class 1).
* :class:`Flame3Pairs` -- FLAME 3 image quartets (FOV-corrected RGB +
  radiometric thermal TIFF). The only public source of genuinely registered
  drone RGB+thermal *disaster* pairs, so it is the reference set for validating
  4-channel early fusion (README §2.1, DATASETS.md §1).

Both live behind IEEE DataPort, which requires a (free) account and an
interactive download -- there is no stable direct URL, so ``download=True``
raises with instructions rather than pretending to work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .base import DangerClass, Sample, SkyLensDatasetBase

__all__ = ["FlameSegmentation", "Flame3Pairs"]

_IMG_EXT = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")


def _split_slice(items: list, split: str, val_frac: float = 0.1) -> list:
    """Deterministic train/val partition for datasets shipped without splits."""
    n_val = max(1, int(len(items) * val_frac)) if items else 0
    return items[n_val:] if split == "train" else items[:n_val]


class FlameSegmentation(SkyLensDatasetBase):
    """FLAME fire-segmentation subset -- RGB frames + binary fire masks.

    Yields ``danger_mask`` with ``{0: normal, 1: fire}`` and
    ``person_boxes=None`` (segmentation-only dataset, README §6.3).
    """

    name = "FLAME (fire segmentation)"
    splits = ("train", "val")
    availability = "account"
    homepage = (
        "https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-"
        "burn-detection-using-drones-uavs"
    )
    download_note = (
        "IEEE DataPort requires a free account and an interactive (browser) download;\n"
        "there is no stable direct URL. Grab the 'Images for fire segmentation' and\n"
        "'Masks annotation' items (items 9 and 10 of the FLAME dataset)."
    )
    expected_layout = (
        "<root>/\n"
        "  Images/   frame_0001.jpg ...\n"
        "  Masks/    frame_0001.png ...   (binary: nonzero = fire)"
    )

    def _dirs(self) -> tuple[Path, Path]:
        for img_name, msk_name in (("Images", "Masks"), ("images", "masks")):
            img, msk = self.root / img_name, self.root / msk_name
            if img.is_dir() and msk.is_dir():
                return img, msk
        return self.root / "Images", self.root / "Masks"

    def _check_exists(self) -> bool:
        img, msk = self._dirs()
        return img.is_dir() and msk.is_dir() and any(img.iterdir())

    def _build_index(self) -> Sequence[Any]:
        img_dir, msk_dir = self._dirs()
        pairs = []
        for img in sorted(p for p in img_dir.iterdir() if p.suffix in _IMG_EXT):
            mask = next((m for m in (msk_dir / (img.stem + e) for e in _IMG_EXT)
                         if m.is_file()), None)
            if mask is not None:
                pairs.append((img, mask))
        return _split_slice(pairs, self.split)

    def _load_sample(self, record: Any) -> Sample:
        img_path, mask_path = record
        image = self._read_rgb(img_path)
        raw = self._read_mask(mask_path)
        mask = np.where(raw > 0, DangerClass.FIRE, DangerClass.NORMAL).astype(np.uint8)
        return {
            "image": image,
            "has_rgb": True,
            "has_thermal": False,
            "danger_mask": mask,
            "person_boxes": None,
        }


class Flame3Pairs(SkyLensDatasetBase):
    """FLAME 3 -- FOV-corrected RGB paired with radiometric thermal TIFF.

    Produces a 4-channel ``image`` (RGB in ``[0,1]`` + thermal renormalised into
    ``[0.1, 1.0]`` per README §2.3) with ``has_rgb=has_thermal=True``.

    ``danger_mask`` is ``None``: FLAME 3 ships fire/no-fire *frame* labels rather
    than pixel masks, so it is a fusion-validation set, not a segmentation
    training set (DATASETS.md §1). Frame-level labels, when the folder layout
    provides them, are exposed as the extra key ``frame_label``.
    """

    name = "FLAME 3 (RGB + radiometric thermal)"
    splits = ("train", "val")
    availability = "account"
    homepage = (
        "https://ieee-dataport.org/open-access/flame-3-radiometric-thermal-uav-"
        "imagery-wildfire-management"
    )
    download_note = (
        "IEEE DataPort requires a free account and an interactive download.\n"
        "Use the 'Fire Imagery -- CV Subset' package, which contains the image\n"
        "quartets (raw RGB / raw thermal / FOV-corrected RGB / thermal TIFF)."
    )
    expected_layout = (
        "<root>/\n"
        "  Corrected FOV Images/   <stem>.jpg ...   (FOV-matched RGB)\n"
        "  Thermal TIFFs/          <stem>.tiff ...  (radiometric, per-pixel degC)"
    )

    _RGB_DIRS = ("Corrected FOV Images", "corrected_fov", "rgb", "RGB")
    _THERMAL_DIRS = ("Thermal TIFFs", "thermal_tiff", "thermal", "Thermal")

    def _dirs(self) -> tuple[Path | None, Path | None]:
        def find(names):
            for n in names:
                p = self.root / n
                if p.is_dir():
                    return p
            return None

        return find(self._RGB_DIRS), find(self._THERMAL_DIRS)

    def _check_exists(self) -> bool:
        rgb, thermal = self._dirs()
        return rgb is not None and thermal is not None

    def _build_index(self) -> Sequence[Any]:
        rgb_dir, th_dir = self._dirs()
        thermal_by_stem = {p.stem: p for p in th_dir.rglob("*")
                           if p.suffix.lower() in (".tif", ".tiff")}
        pairs = []
        for img in sorted(p for p in rgb_dir.rglob("*") if p.suffix in _IMG_EXT):
            th = thermal_by_stem.get(img.stem)
            if th is not None:
                pairs.append((img, th))
        return _split_slice(pairs, self.split)

    def _load_sample(self, record: Any) -> Sample:
        rgb_path, th_path = record
        rgb = self._read_rgb(rgb_path)
        thermal = self._read_thermal(th_path)

        if thermal.shape[:2] != rgb.shape[:2]:
            import cv2

            thermal = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        sample: Sample = {
            "image": self._stack_rgb_thermal(rgb, thermal),
            "has_rgb": True,
            "has_thermal": True,
            "danger_mask": None,
            "person_boxes": None,
        }
        parent = rgb_path.parent.name.lower()
        if "no" in parent and "fire" in parent:
            sample["frame_label"] = 0
        elif "fire" in parent:
            sample["frame_label"] = 1
        return sample
