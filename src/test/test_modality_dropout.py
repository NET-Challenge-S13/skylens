"""Collator modality dropout: v3 default, eligibility, eval collator."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from skylens_model.utils.collate import THERMAL_ABSENT, SkyLensCollator
from skylens_model.utils.trainer import SkyLensTrainer


def _both(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    img = rng.uniform(0.1, 1.0, (32, 32, 4)).astype(np.float32)
    return {"image": img, "has_rgb": True, "has_thermal": True, "danger_mask": None,
            "person_boxes": np.asarray([[4.0, 4.0, 12.0, 16.0]], np.float32)}


def _rgb_only(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    img = rng.integers(1, 255, (32, 32, 3)).astype(np.uint8)
    return {"image": img, "has_rgb": True, "has_thermal": False,
            "danger_mask": np.zeros((32, 32), np.int64), "person_boxes": None}


def _batch() -> list[dict]:
    return [_both(0), _rgb_only(1), _both(2), _rgb_only(3)]


def _v3_reference(samples: list[dict]) -> dict:
    """What develop/v3 produced: no dropout, channels passed through."""
    return SkyLensCollator(person_head_stride=4, modality_dropout=(0.0, 0.0))(samples)


def test_zero_dropout_is_identical_to_v3() -> None:
    samples = _batch()
    ref = _v3_reference(samples)
    for _ in range(5):
        out = SkyLensCollator(person_head_stride=4)(samples)
        assert out.keys() == ref.keys()
        for k in ref:
            assert torch.equal(out[k], ref[k]), k
    assert ref["modality_mask"].tolist() == [[True, True], [True, False]] * 2
    # the thermal channel is the unmodified live plane for RGB+thermal samples
    assert torch.equal(ref["pixel_values"][0, 3], torch.from_numpy(samples[0]["image"][:, :, 3]))


def test_rgb_only_dropout_removes_thermal_only_from_paired_samples() -> None:
    samples = _batch()
    ref = _v3_reference(samples)
    out = SkyLensCollator(person_head_stride=4, modality_dropout=(1.0, 0.0))(samples)
    for i in (0, 2):  # RGB+thermal -> RGB-only
        assert out["modality_mask"][i].tolist() == [True, False]
        assert float(out["pixel_values"][i, 3].abs().max()) == THERMAL_ABSENT
        assert torch.equal(out["pixel_values"][i, :3], ref["pixel_values"][i, :3])
    for i in (1, 3):  # RGB-only: untouched
        assert out["modality_mask"][i].tolist() == [True, False]
        assert torch.equal(out["pixel_values"][i], ref["pixel_values"][i])
    for k in ref:
        if k not in ("pixel_values", "modality_mask"):
            assert torch.equal(out[k], ref[k]), k


def test_thermal_only_dropout_never_removes_the_only_modality() -> None:
    samples = _batch()
    ref = _v3_reference(samples)
    out = SkyLensCollator(person_head_stride=4, modality_dropout=(0.0, 1.0))(samples)
    for i in (0, 2):
        assert out["modality_mask"][i].tolist() == [False, True]
        assert float(out["pixel_values"][i, :3].abs().max()) == 0.0
    for i in (1, 3):
        assert out["modality_mask"][i].tolist() == [True, False]
        assert torch.equal(out["pixel_values"][i], ref["pixel_values"][i])


def test_invalid_probabilities_rejected() -> None:
    with pytest.raises(ValueError):
        SkyLensCollator(modality_dropout=(0.6, 0.6))


def test_eval_collator_never_drops() -> None:
    train_coll = SkyLensCollator(person_head_stride=4, modality_dropout=(0.5, 0.5))
    eval_coll = SkyLensCollator(person_head_stride=4, modality_dropout=(0.0, 0.0))

    class _Stub(SkyLensTrainer):
        def __init__(self) -> None:  # skip HF Trainer setup; only the collator swap is tested
            self.data_collator = train_coll
            self.eval_data_collator = eval_coll

    stub = _Stub()
    seen = stub._with_eval_collator(lambda: stub.data_collator)
    assert seen is eval_coll
    assert stub.data_collator is train_coll  # restored after building the eval loader

    samples = [_both(i) for i in range(64)]
    for _ in range(3):
        assert bool(seen(samples)["modality_mask"].all())
