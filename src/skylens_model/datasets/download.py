"""Download utilities for SkyLens datasets.

Thin wrappers over :mod:`torchvision.datasets.utils` (already a hard dependency)
plus an optional HuggingFace Hub path. Everything third-party is imported
lazily so that merely importing this module never fails.

Three acquisition routes are supported:

``direct``
    A plain HTTP(S) URL that ``requests``/``urllib`` can fetch without auth.
``hf``
    A HuggingFace Hub snapshot (``huggingface_hub.snapshot_download``). Some
    mirrors are gated and need ``huggingface-cli login`` first.
``kaggle``
    The Kaggle CLI/API, which needs ``~/.kaggle/kaggle.json`` credentials.

Anything that requires a browser login, a signed agreement or a Google Drive
"confirm" interstitial is *not* automatable and is surfaced as a
:class:`ManualDownloadRequired` error by the dataset classes (see
``base.py``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = [
    "ManualDownloadRequired",
    "download_and_extract",
    "download_url",
    "download_hf",
    "download_kaggle",
    "extract_archive",
    "check_integrity",
]


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
