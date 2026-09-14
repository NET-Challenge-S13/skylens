"""SARD 옵트인 · SARD 타일 데이터셋 · recall_at_p50."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from skylens_model.datasets import (
    LLVIP,
    SARD,
    FireSegmentation,
    RescueNetSegmentation,
    VisDronePerson,
)
from skylens_model.utils.metrics import PointDetectionMetrics, point_match_flags, recall_sweep
from skylens_model.utils.tiling import ScaledTileCropDataset, build_scaled_cache

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_experiment.py"


def _train_module():
    spec = importlib.util.spec_from_file_location("train_experiment", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sard_off_is_v3_source_list() -> None:
    mod = _train_module()
    root = Path("/data")
    v3 = [
        (LLVIP, root / "llvip" / "LLVIP", "train", "test"),
        (VisDronePerson, root / "visdrone", "train", "val"),
        (RescueNetSegmentation, root / "rescuenet", "train", "test"),
        (FireSegmentation, root / "fire_seg", "train", "val"),
    ]
    assert mod.build_sources(root) == v3
    assert mod.build_sources(root, "off") == v3
    assert mod.build_sources(root, "squash") == v3 + [(SARD, root / "sard", "train", "val")]
    # tiles: 학습에만 들어가고 학습 중 평가 서브셋은 v3 그대로
    assert mod.build_sources(root, "tiles") == v3 + [(SARD, root / "sard", "train", None)]


def _write_sard(root: Path) -> None:
    img_dir, lab_dir = root / "train" / "images", root / "train" / "labels"
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    cv2.imwrite(str(img_dir / "a.jpg"), np.full((1080, 1920, 3), 127, np.uint8))
    # 가로 전체에 걸친 넓은 박스 하나: 어떤 512 크롭에서도 잘린다.
    lines = ["0 0.5 0.5 0.9 0.05", "0 0.1 0.1 0.021 0.05"]
    (lab_dir / "a.txt").write_text("\n".join(lines), encoding="utf-8")


def test_sard_tile_dataset_clips_boxes(tmp_path: Path) -> None:
    _write_sard(tmp_path / "sard")
    raw = SARD(tmp_path / "sard", split="train")
    cache = build_scaled_cache(raw, tmp_path / "SARD_train_short765", 765, workers=1)
    meta = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    assert (meta["samples"][0]["w"], meta["samples"][0]["h"]) == (1360, 765)

    ds = ScaledTileCropDataset(cache, crops_per_image=3, tile=512)
    assert len(ds) == 3
    for k in range(30):
        s = ds[k % 3]
        assert s["image"].shape == (512, 512, 3)
        b = s["person_boxes"]
        assert b is not None and b.ndim == 2 and b.shape[1] == 4
        assert np.all(b >= 0) and np.all(b <= 512)
        # 1224px 폭 박스는 가시율 < 0.5 라 버려진다(같은 규칙: min_visible 0.5).
        assert np.all(b[:, 2] - b[:, 0] < 512)


def test_recall_at_p50_toy_curve() -> None:
    # score 내림차순 TP 패턴: T F F T F T, GT 4개
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    is_tp = np.array([1, 0, 0, 1, 0, 1], bool)
    # thr .9: P1 R.25 / .8: P.5 R.25 / .7: P.33 / .6: P.5 R.5 / .5: P.4 / .4: P.5 R.75
    r = recall_sweep(scores, is_tp, 4, thresholds=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    assert r["recall_at_p50"] == 0.75
    assert r["recall_at_p50_thr"] == 0.4
    assert r["recall_max"] == 0.75
    # 마지막 TP 를 FP 로 바꾸면 0.4 에서 P=2/6 < 0.5 → 0.6 의 0.5 가 최대
    r = recall_sweep(scores, np.array([1, 0, 0, 1, 0, 0], bool), 4,
                     thresholds=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    assert r["recall_at_p50"] == 0.5 and r["recall_max"] == 0.5
    assert recall_sweep(scores, np.zeros(6, bool), 4)["recall_at_p50"] == 0.0


def test_point_match_flags_agree_with_point_metrics() -> None:
    rng = np.random.default_rng(0)
    preds = np.c_[rng.uniform(0, 100, (40, 2)), rng.uniform(0.05, 1, 40)]
    gts = rng.uniform(0, 100, (25, 2))
    flags = point_match_flags(preds, gts, 8.0)
    for t in (0.05, 0.3, 0.6):
        m = PointDetectionMetrics(distance_threshold=8.0)
        m.update(preds[preds[:, 2] >= t], gts)
        assert m.tp == int(flags[preds[:, 2] >= t].sum())
