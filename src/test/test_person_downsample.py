"""`person_downsample`: bilinear stays exactly v4, conv adds a learned stride-2 stage."""

from __future__ import annotations

import pytest
import torch

from skylens_model.models.skylensnet.configuration_skylensnet import SkyLensConfig
from skylens_model.models.skylensnet.modeling_skylensnet import SkyLensForDisasterPerception


def _cfg(**kw) -> SkyLensConfig:
    return SkyLensConfig(use_pretrained_backbone=False, **kw)


def _model(cfg: SkyLensConfig) -> SkyLensForDisasterPerception:
    torch.manual_seed(0)
    return SkyLensForDisasterPerception(cfg).eval()


def test_default_is_bilinear_and_v4_identical() -> None:
    cfg = _cfg()
    assert cfg.person_downsample == "bilinear"
    model = _model(cfg)
    assert model.person_down_conv is None
    # no new parameters in bilinear mode
    assert not any("person_down_conv" in k for k in model.state_dict())

    x = torch.randn(1, cfg.in_channels, 64, 64)
    with torch.no_grad():
        out = model(pixel_values=x)

    # v4 reference path: decoder features resized bilinearly, then the same heads
    with torch.no_grad():
        feats = model.skylens(x)
        stride = cfg.person_head_stride
        ref_feat = torch.nn.functional.interpolate(
            feats, size=(64 // stride, 64 // stride), mode="bilinear", align_corners=False
        )
        ref_hm = torch.sigmoid(model.heatmap_head(model.person_stem(ref_feat)))
        ref_wh = model.wh_head(model.wh_stem(ref_feat))
    assert torch.equal(out.person_heatmap, ref_hm)
    assert torch.equal(out.person_wh, ref_wh)


def test_conv_mode_shapes() -> None:
    cfg = _cfg(person_downsample="conv", use_offset_head=True)
    model = _model(cfg)
    assert model.person_down_conv is not None

    x = torch.randn(2, cfg.in_channels, 64, 64)
    with torch.no_grad():
        out = model(pixel_values=x)
    grid = 64 // cfg.person_head_stride
    assert tuple(out.person_heatmap.shape) == (2, 1, grid, grid)
    assert tuple(out.person_offset.shape) == (2, 2, grid, grid)
    assert tuple(out.person_wh.shape) == (2, 2, grid, grid)
    assert tuple(out.danger_logits.shape) == (2, cfg.num_danger_classes, 64, 64)
    assert torch.isfinite(out.person_heatmap).all()


def test_heatmap_head_bias_stays_centernet_default() -> None:
    for mode in ("bilinear", "conv"):
        model = _model(_cfg(person_downsample=mode))
        assert torch.allclose(
            model.heatmap_head.bias, torch.full_like(model.heatmap_head.bias, -2.19)
        )


def test_old_config_loads_as_bilinear() -> None:
    d = _cfg().to_dict()
    d.pop("person_downsample")
    cfg = SkyLensConfig.from_dict(d)
    assert cfg.person_downsample == "bilinear"
    model = _model(cfg)
    assert model.person_down_conv is None
    out = model(pixel_values=torch.zeros(1, cfg.in_channels, 64, 64))
    assert out.person_heatmap is not None


def test_invalid_value_raises() -> None:
    with pytest.raises(ValueError, match="person_downsample"):
        _cfg(person_downsample="nearest")


def test_seg_logits_unchanged_between_modes() -> None:
    bil = _model(_cfg())
    conv = _model(_cfg(person_downsample="conv"))
    # share every weight the two modes have in common
    missing = conv.load_state_dict(bil.state_dict(), strict=False)
    assert not missing.unexpected_keys
    assert all("person_down_conv" in k for k in missing.missing_keys)

    x = torch.randn(1, bil.config.in_channels, 64, 64)
    with torch.no_grad():
        a = bil(pixel_values=x).danger_logits
        b = conv(pixel_values=x).danger_logits
    assert torch.equal(a, b)


def test_stride_gap_larger_than_two_falls_back_to_resize() -> None:
    # decoder emits stride 2; a stride-8 head needs one conv plus a 2x resize
    cfg = _cfg(person_downsample="conv", person_head_stride=8)
    model = _model(cfg)
    with torch.no_grad():
        out = model(pixel_values=torch.randn(1, cfg.in_channels, 64, 64))
    assert tuple(out.person_heatmap.shape) == (1, 1, 8, 8)
    assert tuple(out.person_wh.shape) == (1, 2, 8, 8)
