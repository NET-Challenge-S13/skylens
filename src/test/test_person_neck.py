"""사람 헤드 FPN 넥(person_neck="fpn") 검증: 출력 shape, 구 config 호환, 세그 경로 불변."""

from __future__ import annotations

import pytest
import torch

from skylens_model.models.skylensnet.configuration_skylensnet import SkyLensConfig
from skylens_model.models.skylensnet.modeling_skylensnet import SkyLensForDisasterPerception


def _config(**kw) -> SkyLensConfig:
    return SkyLensConfig(use_pretrained_backbone=False, **kw)


@pytest.mark.parametrize("neck", ["decoder", "fpn"])
def test_output_shapes(neck: str) -> None:
    torch.manual_seed(0)
    model = SkyLensForDisasterPerception(_config(person_neck=neck)).eval()
    x = torch.randn(2, 4, 128, 96)
    with torch.no_grad():
        out = model(x)
    assert out.danger_logits.shape == (2, 4, 128, 96)
    assert out.person_heatmap.shape == (2, 1, 32, 24)
    assert out.person_wh.shape == (2, 2, 32, 24)
    if neck == "fpn":
        # stage1(256) / stage2(512) / stage3(1024) + out_indices 밖의 stage4(2048)
        assert [m.in_channels for m in model.person_neck.lateral] == [256, 512, 1024, 2048]
        assert model.person_stem.block[0].in_channels == 128
    else:
        assert model.person_neck is None
        assert model.person_stem.block[0].in_channels == 32


def test_fpn_training_loss_finite() -> None:
    torch.manual_seed(0)
    model = SkyLensForDisasterPerception(_config(person_neck="fpn", dice_loss_weight=1.0)).train()
    x = torch.randn(2, 4, 64, 64)
    hm = torch.zeros(2, 1, 16, 16)
    hm[:, :, 8, 8] = 1.0
    mask = torch.zeros(2, 1, 16, 16)
    mask[:, :, 8, 8] = 1.0
    out = model(
        x,
        danger_labels=torch.randint(0, 4, (2, 64, 64)),
        person_heatmap=hm,
        person_wh=torch.ones(2, 2, 16, 16),
        person_reg_mask=mask,
    )
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert model.person_neck.lateral[-1].weight.grad is not None


def test_old_config_loads_as_decoder(tmp_path) -> None:
    old = _config().to_dict()
    old.pop("person_neck")
    old.pop("person_neck_channels")
    cfg = SkyLensConfig.from_dict(old)
    assert cfg.person_neck == "decoder" and cfg.person_neck_channels == 128

    torch.manual_seed(0)
    model = SkyLensForDisasterPerception(cfg).eval()
    model.save_pretrained(tmp_path)
    reloaded = SkyLensForDisasterPerception.from_pretrained(tmp_path).eval()
    assert reloaded.person_neck is None
    x = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        a, b = model(x), reloaded(x)
    assert torch.allclose(a.person_heatmap, b.person_heatmap)
    assert torch.allclose(a.danger_logits, b.danger_logits)


def test_invalid_person_neck() -> None:
    with pytest.raises(ValueError):
        _config(person_neck="bogus")


def test_seg_logits_identical_between_modes() -> None:
    torch.manual_seed(0)
    dec = SkyLensForDisasterPerception(_config(person_neck="decoder")).eval()
    fpn = SkyLensForDisasterPerception(_config(person_neck="fpn")).eval()
    # 공유 부분(백본·디코더·세그 헤드)만 옮긴다 — 사람 헤드는 채널이 달라 제외.
    shared = {
        k: v for k, v in dec.state_dict().items() if k.startswith(("skylens.", "danger_head."))
    }
    missing, unexpected = fpn.load_state_dict(shared, strict=False)
    assert not unexpected
    assert all(k.startswith(("person_neck.", "person_stem.", "wh_stem.", "heatmap_head.", "wh_head.")) for k in missing)
    x = torch.randn(2, 4, 96, 96)
    mm = torch.ones(2, 2, dtype=torch.bool)
    with torch.no_grad():
        a = dec(x, modality_mask=mm).danger_logits
        b = fpn(x, modality_mask=mm).danger_logits
    assert torch.equal(a, b)
