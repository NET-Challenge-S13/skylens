"""Size-relative point matching radius and the pooled_target pool."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from skylens_model.utils.metrics import point_match_flags, size_relative_radius

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eval_deploy.py"


def _eval_module():
    spec = importlib.util.spec_from_file_location("eval_deploy", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_per_gt_radius_of_8_equals_scalar_matching() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        preds = np.c_[rng.uniform(0, 120, (60, 2)), rng.uniform(0.05, 1, 60)]
        gts = rng.uniform(0, 120, (35, 2))
        old = point_match_flags(preds, gts, 8.0)
        new = point_match_flags(preds, gts, np.full(len(gts), 8.0))
        assert np.array_equal(old, new)


def test_large_gt_matched_at_12px_passes_rel15_only() -> None:
    gt = np.array([[100.0, 100.0, 40.0, 120.0]])  # height 120 -> r = max(8, 18) = 18
    pred = np.array([[112.0, 100.0, 0.9]])
    assert not point_match_flags(pred, gt[:, :2], 8.0)[0]
    assert point_match_flags(pred, gt[:, :2], size_relative_radius(gt[:, 3]))[0]
    # small boxes keep the 8px floor
    assert size_relative_radius(np.array([20.0]))[0] == 8.0


def test_pooled_target_is_exactly_the_three_sources() -> None:
    mod = _eval_module()
    assert mod.POOLS["pooled_target"] == ("llvip_866", "visdrone137", "sard394")
    assert mod.POOLS["pooled137_vd8"] == ("llvip_866", "visdrone137")
    assert mod.TARGET_POOL == "pooled_target" and mod.TARGET_RADIUS == "rel15"


def test_point_curve_matches_recall_sweep_and_targets() -> None:
    mod = _eval_module()
    s = np.array([0.9, 0.8, 0.5, 0.35, 0.2, 0.1])
    f = np.array([True, True, False, True, False, True])
    c = mod.point_curve(s, f, 5, [0.1, 0.3], [0.05, 0.3])
    assert c["sweep"]["0.30"]["point_recall"] == 0.6
    assert abs(c["f1_at_0.30"] - 2 * 0.75 * 0.6 / 1.35) < 1e-12
    assert c["recall_max"] == 0.8
