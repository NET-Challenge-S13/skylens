"""Common base class and shared schema for SkyLens datasets.

Design references (``src/skylens_model/README.md``):

* §1.4 -- bbox lives on the *label* side, points on the *projection* side. Every
  person dataset therefore hands back plain xyxy boxes; converting them to a
  CenterNet heatmap is the collator's job (``collate.py``), not the dataset's.
* §2.2/§2.3 -- modality dropout means a sample must declare *which* modalities it
  actually carries (``has_rgb`` / ``has_thermal``) instead of pretending a
  zero-filled channel is real data.
* §6.3 -- heads are trained separately, so a sample legitimately has only one of
  ``danger_mask`` / ``person_boxes``. ``None`` is a normal value, not an error.

Unified label schema (README §9 "클래스 통합 스키마"):

===== ==================== ==============================================
value name                 sourced from
===== ==================== ==============================================
0     normal               everything not otherwise dangerous
1     fire                 FLAME fire masks, FLAME 3
2     collapse             RescueNet building major-damage / total-destruction
3     road_blocked         RescueNet road-blocked
255   ignore               unlabeled / void
===== ==================== ==============================================
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DangerClass",
    "DANGER_CLASS_NAMES",
    "IGNORE_INDEX",
    "NUM_DANGER_CLASSES",
    "Sample",
    "SkyLensDatasetBase",
    "ManualDownloadRequired",
    "download_and_extract",
    "download_url",
    "download_hf",
    "download_kaggle",
    "extract_archive",
    "check_integrity",
]


class DangerClass:
    """Unified danger-zone segmentation labels."""

    NORMAL = 0
    FIRE = 1
    COLLAPSE = 2
    ROAD_BLOCKED = 3
    IGNORE = 255


DANGER_CLASS_NAMES = ("normal", "fire", "collapse", "road_blocked")
NUM_DANGER_CLASSES = len(DANGER_CLASS_NAMES)
IGNORE_INDEX = DangerClass.IGNORE

#: The exact dict contract every ``__getitem__`` must satisfy.
Sample = dict[str, Any]


class SkyLensDatasetBase:
    """Base for all SkyLens datasets -- torchvision-style constructor.

    Subclasses implement :meth:`_check_exists`, :meth:`_build_index` and
    :meth:`_load_sample`, and set the download-metadata class attributes below.

    Parameters
    ----------
    root:
        Directory holding this dataset. The dataset's own files are expected
        directly under ``root`` (torchvision's ``root/<DatasetName>`` nesting is
        *not* used, since these archives already unpack into named folders).
    split:
        One of :attr:`splits`.
    transforms:
        Callable applied to the sample dict before it is returned. An
        albumentations pipeline can be adapted with a small lambda; see
        ``README.md``.
    download:
        Attempt automatic acquisition. Raises
        :class:`ManualDownloadRequired` with
        full instructions when the dataset cannot be fetched programmatically.
    """

    #: Human-facing dataset name used in error messages.
    name: str = "dataset"
    #: Accepted ``split`` values.
    splits: tuple[str, ...] = ("train", "val", "test")
    #: ``"auto"`` | ``"account"`` | ``"manual"`` -- see ``README.md``.
    availability: str = "manual"
    #: Where a human should go to get it.
    homepage: str = ""
    #: Free-form note appended to the manual-download error.
    download_note: str = ""
    #: Directory tree the loader expects once the archive is extracted.
    expected_layout: str = ""

    def __init__(
        self,
        root: str | os.PathLike,
        split: str = "train",
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = self._verify_split(split)
        self.transforms = transforms

        if download:
            self.download()

        if not self._check_exists():
            raise ManualDownloadRequired(self.manual_download_message())

        self._index: list[Any] = list(self._build_index())

    # ------------------------------------------------------------------ #
    # torchvision-style API
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Sample:
        sample = self._load_sample(self._index[idx])
        sample = self._normalize_sample(sample)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n = len(self._index) if hasattr(self, "_index") else "?"
        return f"{type(self).__name__}(root={str(self.root)!r}, split={self.split!r}, n={n})"

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"split={self.split}"

    # ------------------------------------------------------------------ #
    # hooks for subclasses
    # ------------------------------------------------------------------ #

    def _check_exists(self) -> bool:
        """True when the expected files are present under :attr:`root`."""
        raise NotImplementedError

    def _build_index(self) -> Sequence[Any]:
        """Return one opaque record per sample (usually a path or tuple)."""
        raise NotImplementedError

    def _load_sample(self, record: Any) -> Sample:
        """Materialise ``record`` into a sample dict."""
        raise NotImplementedError

    def download(self) -> None:
        """Acquire the dataset. Default: not automatable."""
        if self._check_exists():
            return
        raise ManualDownloadRequired(self.manual_download_message())

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _verify_split(self, split: str) -> str:
        if split not in self.splits:
            raise ValueError(
                f"{type(self).__name__}: unknown split {split!r}; "
                f"expected one of {self.splits}."
            )
        return split

    def manual_download_message(self) -> str:
        """A message a human can actually act on. Never fail silently."""
        verdict = {
            "auto": "This dataset IS auto-downloadable, but the attempt did not "
                    "produce the expected files.",
            "account": "This dataset requires a (free) account, so it cannot be "
                       "fetched without credentials.",
            "manual": "This dataset CANNOT be downloaded programmatically.",
        }.get(self.availability, "")

        lines = [
            f"{self.name}: data not found under {self.root}",
            "",
            verdict,
        ]
        if self.homepage:
            lines += ["", f"  Source: {self.homepage}"]
        if self.download_note:
            lines += ["", "  " + self.download_note.strip().replace("\n", "\n  ")]
        if self.expected_layout:
            lines += [
                "",
                "  After downloading, extract so that the tree looks like:",
                "",
                "  " + self.expected_layout.strip().replace("\n", "\n  "),
            ]
        lines += ["", f"  Then re-run with root={str(self.root)!r} and download=False."]
        return "\n".join(lines)

    # -- sample validation ------------------------------------------------

    @staticmethod
    def _normalize_sample(sample: Sample) -> Sample:
        """Coerce a sample to the fixed contract, and fail loudly if it can't be.

        Contract::

            image         (H, W, C) uint8 | float32   C=3 RGB, 4 RGB+thermal, 1 thermal
            has_rgb       bool
            has_thermal   bool
            danger_mask   (H, W) uint8 | None         0..3, 255 = ignore
            person_boxes  (N, 4) float32 | None       xyxy in pixels
        """
        image = np.asarray(sample["image"])
        if image.ndim == 2:
            image = image[:, :, None]
        if image.ndim != 3:
            raise ValueError(f"image must be (H, W, C); got shape {image.shape}")
        if image.dtype not in (np.uint8, np.float32):
            image = image.astype(np.float32)
        sample["image"] = image

        sample["has_rgb"] = bool(sample.get("has_rgb", False))
        sample["has_thermal"] = bool(sample.get("has_thermal", False))

        mask = sample.get("danger_mask")
        if mask is not None:
            mask = np.asarray(mask)
            if mask.ndim != 2:
                raise ValueError(f"danger_mask must be (H, W); got {mask.shape}")
            sample["danger_mask"] = mask.astype(np.uint8, copy=False)
        else:
            sample["danger_mask"] = None

        boxes = sample.get("person_boxes")
        if boxes is not None:
            boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            sample["person_boxes"] = boxes
        else:
            sample["person_boxes"] = None

        return sample

    # -- io ---------------------------------------------------------------

    @staticmethod
    def _read_rgb(path: str | os.PathLike) -> np.ndarray:
        """Read an 8-bit RGB image as ``(H, W, 3)`` uint8."""
        import cv2

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _read_mask(path: str | os.PathLike) -> np.ndarray:
        """Read a single-channel label image as ``(H, W)`` uint8."""
        import cv2

        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"could not read mask: {path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask.astype(np.uint8, copy=False)

    @staticmethod
    def _read_thermal(path: str | os.PathLike) -> np.ndarray:
        """Read a thermal frame as ``(H, W)`` float32.

        Radiometric TIFFs (FLAME 3) are read with ``tifffile`` when available so
        the per-pixel temperature values survive; everything else falls back to
        OpenCV. ``tifffile`` is imported lazily -- a missing install only breaks
        radiometric reads, not module import.
        """
        path = Path(path)
        if path.suffix.lower() in (".tif", ".tiff"):
            try:
                import tifffile
            except ImportError as exc:
                raise ImportError(
                    "reading radiometric thermal TIFFs needs `tifffile`; "
                    "install it with `pip install tifffile`."
                ) from exc
            return np.asarray(tifffile.imread(str(path))).astype(np.float32)

        import cv2

        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"could not read thermal frame: {path}")
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        return arr.astype(np.float32)

    @staticmethod
    def _stack_rgb_thermal(rgb: np.ndarray | None, thermal: np.ndarray | None) -> np.ndarray:
        """Concatenate RGB (uint8) and thermal (float32) into one array.

        Only used by datasets that ship genuinely registered pairs (FLAME 3,
        KAIST, LLVIP). The thermal plane is min-max normalised into ``[0.1, 1.0]``
        per README §2.3, reserving exactly ``0.0`` for "no thermal".
        """
        parts = []
        if rgb is not None:
            parts.append(rgb.astype(np.float32) / 255.0)
        if thermal is not None:
            t = thermal.astype(np.float32)
            lo, hi = float(t.min()), float(t.max())
            t = np.zeros_like(t) if hi <= lo else (t - lo) / (hi - lo)
            parts.append((0.1 + 0.9 * t)[:, :, None])
        if not parts:
            raise ValueError("at least one of rgb/thermal must be provided")
        return np.concatenate(parts, axis=2).astype(np.float32)

# --------------------------------------------------------------------------- #
# 다운로드 유틸
# --------------------------------------------------------------------------- #


class ManualDownloadRequired(RuntimeError):
    """Raised when a dataset cannot be fetched programmatically.

    Carries a fully actionable message: source URL, the account/agreement
    needed, and the directory layout expected after manual extraction.
    """


def _tv_utils():
    """Lazily import ``torchvision.datasets.utils``."""
    try:
        from torchvision.datasets import utils as tv_utils
    except Exception as exc:  # pragma: no cover - torchvision is a hard dep
        raise RuntimeError(
            "torchvision is required for dataset downloads; install it with "
            "`pip install torchvision`."
        ) from exc
    return tv_utils


# --------------------------------------------------------------------------- #
# direct HTTP
# --------------------------------------------------------------------------- #


def download_url(url: str, root: str | os.PathLike, filename: str | None = None,
                 md5: str | None = None) -> Path:
    """Download ``url`` into ``root``. Returns the path to the file.

    Skips the transfer when the file already exists and the md5 matches.
    """
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    filename = filename or os.path.basename(url.split("?")[0]) or "download.bin"
    dest = root / filename

    if dest.is_file() and (md5 is None or check_integrity(dest, md5)):
        return dest

    _tv_utils().download_url(url, str(root), filename=filename, md5=md5)
    return dest


def extract_archive(archive: str | os.PathLike, to_path: str | os.PathLike | None = None,
                    remove_finished: bool = False) -> Path:
    """Extract a ``.zip``/``.tar*``/``.gz`` archive. Returns the target dir."""
    archive = Path(archive)
    to_path = Path(to_path) if to_path is not None else archive.parent
    to_path.mkdir(parents=True, exist_ok=True)
    _tv_utils().extract_archive(str(archive), str(to_path), remove_finished=remove_finished)
    return to_path


def download_and_extract(url: str, root: str | os.PathLike, md5: str | None = None,
                         filename: str | None = None,
                         extract_root: str | os.PathLike | None = None,
                         remove_finished: bool = False) -> Path:
    """Download ``url`` into ``root`` and extract it. Returns the extract dir.

    ``md5`` is verified when given; a mismatch raises ``RuntimeError`` from
    torchvision rather than silently continuing.
    """
    archive = download_url(url, root, filename=filename, md5=md5)
    return extract_archive(archive, extract_root or root, remove_finished=remove_finished)


def check_integrity(path: str | os.PathLike, md5: str | None = None) -> bool:
    """True when ``path`` exists and (if ``md5`` given) its checksum matches."""
    path = Path(path)
    if not path.is_file():
        return False
    if md5 is None:
        return True
    return _tv_utils().check_md5(str(path), md5)


# --------------------------------------------------------------------------- #
# HuggingFace Hub
# --------------------------------------------------------------------------- #


def download_hf(repo_id: str, root: str | os.PathLike, repo_type: str = "dataset",
                allow_patterns: list[str] | None = None,
                revision: str | None = None) -> Path:
    """Snapshot a HuggingFace repo into ``root``.

    Gated repos raise; the caller is expected to translate that into a
    :class:`ManualDownloadRequired` with login instructions.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ManualDownloadRequired(
            f"`huggingface_hub` is not installed, so {repo_id!r} cannot be fetched "
            "automatically. Run `pip install huggingface_hub` (or download the repo "
            "manually from https://huggingface.co/datasets/" + repo_id + ")."
        ) from exc

    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(root),
        allow_patterns=allow_patterns,
        revision=revision,
    )
    return Path(local)


# --------------------------------------------------------------------------- #
# Kaggle
# --------------------------------------------------------------------------- #


def download_kaggle(slug: str, root: str | os.PathLike, competition: bool = False,
                    unzip: bool = True) -> Path:
    """Download a Kaggle dataset (``owner/name``) via the Kaggle CLI.

    Requires credentials at ``~/.kaggle/kaggle.json`` (or ``KAGGLE_USERNAME`` /
    ``KAGGLE_KEY`` env vars) and acceptance of the dataset's terms on the site.
    """
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    exe = shutil.which("kaggle")
    cmd = [exe] if exe else [sys.executable, "-m", "kaggle"]
    cmd += ["competitions" if competition else "datasets", "download",
            "-c" if competition else "-d", slug, "-p", str(root)]
    if unzip:
        cmd.append("--unzip")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise ManualDownloadRequired(
            f"The Kaggle CLI is unavailable, so {slug!r} cannot be fetched "
            "automatically. Run `pip install kaggle` and place your API token at "
            "~/.kaggle/kaggle.json (Kaggle > Account > Create New API Token)."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ManualDownloadRequired(
            f"`kaggle datasets download -d {slug}` failed (exit {exc.returncode}). "
            "Most often this means the API token is missing/expired, or you have "
            f"not accepted the dataset rules at https://www.kaggle.com/datasets/{slug}"
        ) from exc
    return root
