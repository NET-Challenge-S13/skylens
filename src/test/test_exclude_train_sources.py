"""`--exclude-train-sources`: 학습 split 만 빼고 평가 split 은 그대로 둔다."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from skylens_model.datasets import (
    LLVIP,
    SARD,
    FireSegmentation,
    RescueNetSegmentation,
    VisDronePerson,
)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_experiment.py"
_ROOT = Path("/data")


def _train_module():
    spec = importlib.util.spec_from_file_location("train_experiment", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _v4(root: Path) -> list:
    """v4 실행이 쓰는 출처 목록(--sard tiles)."""
    return [
        (LLVIP, root / "llvip" / "LLVIP", "train", "test"),
        (VisDronePerson, root / "visdrone", "train", "val"),
        (RescueNetSegmentation, root / "rescuenet", "train", "test"),
        (FireSegmentation, root / "fire_seg", "train", "val"),
        (SARD, root / "sard", "train", None),
    ]


def _eval_splits(sources: list) -> list:
    return [(cls, root, ev) for cls, root, _tr, ev in sources]


def test_default_is_unchanged_from_v4() -> None:
    mod = _train_module()
    v3 = _v4(_ROOT)[:4]
    assert mod.build_sources(_ROOT) == v3
    assert mod.build_sources(_ROOT, "off") == v3
    assert mod.build_sources(_ROOT, "off", []) == v3
    assert mod.build_sources(_ROOT, "tiles") == _v4(_ROOT)
    assert mod.build_sources(_ROOT, "tiles", []) == _v4(_ROOT)
    assert mod.build_sources(_ROOT, "squash") == v3 + [(SARD, _ROOT / "sard", "train", "val")]


def test_exclude_visdrone_drops_only_its_train_split() -> None:
    mod = _train_module()
    base = _v4(_ROOT)
    got = mod.build_sources(_ROOT, "tiles", ["visdrone"])
    assert got == [
        (LLVIP, _ROOT / "llvip" / "LLVIP", "train", "test"),
        (VisDronePerson, _ROOT / "visdrone", None, "val"),
        (RescueNetSegmentation, _ROOT / "rescuenet", "train", "test"),
        (FireSegmentation, _ROOT / "fire_seg", "train", "val"),
        (SARD, _ROOT / "sard", "train", None),
    ]
    # 평가 쪽은 손대지 않는다.
    assert _eval_splits(got) == _eval_splits(base)


def test_exclude_llvip_drops_only_its_train_split() -> None:
    mod = _train_module()
    base = _v4(_ROOT)
    got = mod.build_sources(_ROOT, "tiles", ["llvip"])
    assert got[0] == (LLVIP, _ROOT / "llvip" / "LLVIP", None, "test")
    assert got[1:] == base[1:]
    assert _eval_splits(got) == _eval_splits(base)


def test_exclude_sard_equals_sard_off() -> None:
    mod = _train_module()
    v3 = _v4(_ROOT)[:4]
    assert mod.build_sources(_ROOT, "tiles", ["sard"]) == v3
    assert mod.build_sources(_ROOT, "squash", ["sard"]) == v3


def test_unknown_name_raises() -> None:
    mod = _train_module()
    with pytest.raises(ValueError, match="unknown --exclude-train-sources"):
        mod.build_sources(_ROOT, "tiles", ["visdron"])


def test_excluding_every_person_source_raises() -> None:
    mod = _train_module()
    with pytest.raises(ValueError, match="person-labelled training source"):
        mod.build_sources(_ROOT, "tiles", ["llvip", "visdrone", "sard"])
    # --sard off 면 SARD 는 애초에 학습에 없으므로 둘만 빼도 사람 출처가 없다.
    with pytest.raises(ValueError, match="person-labelled training source"):
        mod.build_sources(_ROOT, "off", ["llvip", "visdrone"])
