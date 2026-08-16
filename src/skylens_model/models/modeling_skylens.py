"""SkyLens 재난 인지 모델 (PyTorch / HuggingFace transformers 규약).

구조 (README §1, §7):

    pixel_values (B, C, H, W)   C = RGB(3) + thermal(1) [+ validity(1)]
        │  modality dropout (학습 시에만, README §2.2 / §2.3)
        ▼
    AutoBackbone 인코더  ──▶ 멀티스케일 feature (out_indices)
        ▼
    UNet 디코더 (스킵 커넥션)  ──▶ 고해상도 feature map
        ├─▶ 세그멘테이션 헤드 : 위험구역 per-pixel 클래스맵
        └─▶ 점 검출 헤드     : CenterNet 히트맵 + (w,h) 회귀

주의: "transformers를 쓴다"가 "Transformer 모델을 쓴다"는 뜻이 아니다.
`AutoBackbone`으로 ResNet/ConvNeXt 같은 **CNN 인코더**를 불러온다 (README §7.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

from .configuration_skylens import SkyLensConfig


# ---------------------------------------------------------------------------
# 출력 자료구조
# ---------------------------------------------------------------------------


@dataclass
class SkyLensOutput(ModelOutput):
    """SkyLens 이중 헤드 출력.

    Args:
        loss: 가중합된 총 loss. 어떤 GT도 주어지지 않으면 `None`.
        loss_dict: 개별 loss (`danger_seg`, `person_heatmap`, `person_wh`).
        danger_logits: `(B, num_danger_classes, H, W)` — 입력 해상도 위험구역 로짓.
        person_heatmap: `(B, 1, H/s, W/s)` — **sigmoid가 적용된 확률**.
        person_wh: `(B, 2, H/s, W/s)` — 중심점 기준 (w, h) 회귀값.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[dict] = None
    danger_logits: torch.FloatTensor = None
    person_heatmap: torch.FloatTensor = None
    person_wh: torch.FloatTensor = None


# ---------------------------------------------------------------------------
# 빌딩 블록
# ---------------------------------------------------------------------------


class SkyLensConvBlock(nn.Module):
    """Conv-BN-ReLU ×2 (표준 UNet 블록)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_state)


class SkyLensDecoderBlock(nn.Module):
    """업샘플 → (있으면) 스킵 concat → Conv-BN-ReLU ×2."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = SkyLensConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, hidden_state: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skip is not None:
            # 스킵 해상도에 정확히 맞춘다 (홀수 해상도 대비).
            hidden_state = F.interpolate(hidden_state, size=skip.shape[-2:], mode="nearest")
            hidden_state = torch.cat([hidden_state, skip], dim=1)
        else:
            hidden_state = F.interpolate(hidden_state, scale_factor=2.0, mode="nearest")
        return self.conv(hidden_state)


class SkyLensUNetDecoder(nn.Module):
    """백본의 멀티스케일 feature를 받아 고해상도 feature map으로 복원한다.

    스킵 채널 수는 `backbone.channels`에서 받아 **동적으로** 구성한다(하드코딩 금지).
    """

    def __init__(self, encoder_channels: list[int], decoder_channels: tuple):
        super().__init__()
        # encoder_channels 는 얕은→깊은 순서. 가장 깊은 것이 디코더 입력.
        skip_channels = list(encoder_channels[:-1])[::-1]  # 깊은→얕은 순 스킵
        in_ch = encoder_channels[-1]
        blocks = []
        for i, out_ch in enumerate(decoder_channels):
            skip_ch = skip_channels[i] if i < len(skip_channels) else 0
            blocks.append(SkyLensDecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = in_ch

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        skips = list(features[:-1])[::-1]
        hidden_state = features[-1]
        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            hidden_state = block(hidden_state, skip)
        return hidden_state


# ---------------------------------------------------------------------------
# 4채널 inflation (README §6.1)
# ---------------------------------------------------------------------------


def _find_first_conv(module: nn.Module) -> tuple[Optional[str], Optional[nn.Conv2d]]:
    """모듈 트리에서 **첫 번째 Conv2d**를 (이름, 모듈)로 반환한다.

    ResNet / ConvNeXt / timm 등 백본 종류에 상관없이 동작하는 범용 헬퍼.
    `named_modules()`는 정의 순서로 순회하므로 stem conv가 먼저 나온다.
    """
    for name, sub in module.named_modules():
        if isinstance(sub, nn.Conv2d):
            return name, sub
    return None, None


def _set_module_by_name(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def inflate_first_conv(backbone: nn.Module, in_channels: int, pretrained: bool = True) -> None:
    """백본 첫 conv를 `in_channels` 입력으로 확장한다 (in-place).

    README §6.1 공식:
        weight[:, :3]  = 프리트레인 RGB 가중치 (그대로)
        weight[:, 3:]  = 프리트레인 RGB 가중치.mean(dim=1, keepdim=True) (남는 채널마다)

    RGB 평균으로 채우면 초기 활성 스케일이 유지되어 0/랜덤 초기화보다 안정적이다.
    `pretrained=False`면 그냥 채널 수만 맞춘다(기본 초기화).
    """
    name, conv = _find_first_conv(backbone)
    if conv is None:
        raise ValueError("백본에서 Conv2d를 찾지 못했다 — 4채널 inflation 불가.")
    if conv.in_channels == in_channels:
        return

    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    new_conv = new_conv.to(device=conv.weight.device, dtype=conv.weight.dtype)

    if pretrained and conv.in_channels >= 3 and in_channels > 3:
        with torch.no_grad():
            old_w = conv.weight.data
            rgb = old_w[:, :3]
            new_conv.weight.data[:, :3] = rgb
            # 남는 채널(열화상, validity 등)은 전부 RGB 평균으로 채운다.
            mean_w = rgb.mean(dim=1, keepdim=True)
            new_conv.weight.data[:, 3:] = mean_w.repeat(1, in_channels - 3, 1, 1)
            if conv.bias is not None:
                new_conv.bias.data.copy_(conv.bias.data)
    elif pretrained and conv.bias is not None:
        with torch.no_grad():
            new_conv.bias.data.copy_(conv.bias.data)

    # 백본이 PreTrainedModel이면 post_init 시 자기 `_init_weights`로 새 conv를
    # 랜덤 재초기화해버린다. HF 관례대로 "이미 초기화됨" 플래그를 세워 막는다.
    new_conv._is_hf_initialized = True
    new_conv._skylens_skip_init = True

    _set_module_by_name(backbone, name, new_conv)

    # 일부 백본(ResNet/ConvNeXt 등)은 forward에서 채널 수를 검증한다.
    # config 뿐 아니라 embedding 모듈이 캐시한 값도 함께 갱신해야 한다.
    old_in = conv.in_channels
    bb_config = getattr(backbone, "config", None)
    if bb_config is not None:
        for attr in ("num_channels", "in_chans", "input_channels"):
            if hasattr(bb_config, attr):
                setattr(bb_config, attr, in_channels)
    for sub in backbone.modules():
        for attr in ("num_channels", "in_chans"):
            if getattr(sub, attr, None) == old_in:
                setattr(sub, attr, in_channels)


# ---------------------------------------------------------------------------
# loss 함수
# ---------------------------------------------------------------------------


def centernet_focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet penalty-reduced pixel-wise focal loss (Law & Deng, CornerNet).

    `pred`는 **sigmoid가 이미 적용된 확률**, `target`은 가우시안 GT 히트맵.
    중심점(target == 1)만 positive로 보고, 주변 픽셀은 가우시안 값만큼 페널티를 깎는다.
    """
    eps = 1e-4
    pred = pred.clamp(min=eps, max=1.0 - eps)

    pos_mask = target.eq(1.0).float()
    neg_mask = 1.0 - pos_mask

    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos_mask
    neg_weights = torch.pow(1.0 - target, beta)
    neg_loss = torch.log(1.0 - pred) * torch.pow(pred, alpha) * neg_weights * neg_mask

    num_pos = pos_mask.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """유효 중심점 위치에서만 계산하는 L1 회귀 loss (0 나눗셈 방지)."""
    mask = mask.expand_as(pred).float()
    loss = (F.l1_loss(pred * mask, target * mask, reduction="sum")) / mask.sum().clamp(min=1.0)
    return loss


# ---------------------------------------------------------------------------
# PreTrainedModel
# ---------------------------------------------------------------------------


class SkyLensPreTrainedModel(PreTrainedModel):
    config_class = SkyLensConfig
    base_model_prefix = "skylens"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        # 프리트레인 백본은 건드리지 않는다 (post_init 이 가중치를 지우지 않도록).
        if getattr(module, "_skylens_skip_init", False):
            return
        if isinstance(module, nn.Conv2d):
            # 예측 헤드는 kaiming(fan_out)을 쓰면 안 된다. 출력 채널이 1~4로 작아
            # fan_out 이 작고 std = sqrt(2/fan_out) 이 1.4까지 커진다(1채널 히트맵의 경우).
            # 그러면 로짓이 포화되어 heatmap 헤드의 bias=-2.19 (초기 sigmoid≈0.1) 트릭이
            # 무력화되고, CenterNet focal loss 가 초기부터 수천대로 폭주한다.
            # 따라서 출력 헤드는 작은 std 의 정규분포로 따로 초기화한다.
            if getattr(module, "_skylens_head_init", False):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            else:
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                bias_init = getattr(module, "_skylens_bias_init", 0.0)
                nn.init.constant_(module.bias, bias_init)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


# ---------------------------------------------------------------------------
# 인코더 + 디코더 (헤드 없음)
# ---------------------------------------------------------------------------


class SkyLensModel(SkyLensPreTrainedModel):
    """AutoBackbone 인코더 + UNet 디코더 → 고해상도 feature map 반환."""

    def __init__(self, config: SkyLensConfig):
        super().__init__(config)
        self.config = config

        self.backbone = self._build_backbone(config)
        # 4채널(+validity) inflation — README §6.1
        inflate_first_conv(
            self.backbone,
            config.num_input_channels,
            pretrained=config.use_pretrained_backbone,
        )

        encoder_channels = list(self.backbone.channels)
        self.decoder = SkyLensUNetDecoder(encoder_channels, config.decoder_channels)
        self.out_channels = self.decoder.out_channels

        self.post_init()

    @staticmethod
    def _build_backbone(config: SkyLensConfig) -> nn.Module:
        """transformers Backbone API로 인코더를 만든다.

        `use_pretrained_backbone=False`면 config만 가져와 랜덤 초기화한다.
        """
        out_indices = tuple(config.backbone_out_indices)

        if config.use_timm_backbone:
            from transformers import TimmBackbone, TimmBackboneConfig

            if config.use_pretrained_backbone:
                return TimmBackbone.from_pretrained(
                    config.backbone, out_indices=out_indices, use_pretrained_backbone=True
                )
            timm_config = TimmBackboneConfig(
                backbone=config.backbone, out_indices=out_indices, use_pretrained_backbone=False
            )
            return TimmBackbone(timm_config)

        from transformers import AutoBackbone, AutoConfig

        if config.use_pretrained_backbone:
            return AutoBackbone.from_pretrained(
                config.backbone, out_indices=out_indices, use_timm_backbone=False
            )
        backbone_config = AutoConfig.from_pretrained(config.backbone, out_indices=out_indices)
        return AutoBackbone.from_config(backbone_config)

    def _mark_backbone_no_init(self) -> None:
        for module in self.backbone.modules():
            module._skylens_skip_init = True

    def post_init(self):  # type: ignore[override]
        if self.config.use_pretrained_backbone:
            # inflate된 첫 conv 포함, 백본 전체를 재초기화 대상에서 제외한다.
            self._mark_backbone_no_init()
        super().post_init()

    def get_backbone(self) -> nn.Module:
        return self.backbone

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """`(B, C, H, W)` → `(B, decoder_channels[-1], H', W')` 고해상도 feature."""
        outputs = self.backbone(pixel_values)
        features = list(outputs.feature_maps)
        return self.decoder(features)


# ---------------------------------------------------------------------------
# 이중 헤드 모델
# ---------------------------------------------------------------------------


class SkyLensForDisasterPerception(SkyLensPreTrainedModel):
    """SkyLensModel + 이중 헤드(위험구역 세그멘테이션 + 사람 점 검출).

    두 헤드는 공유 인코더를 쓰지만 loss는 분리 계산된다 — 재난과 사람이 동시에
    라벨링된 데이터가 없기 때문(README §6.3). GT가 `None`인 헤드는 loss를 건너뛴다.
    """

    def __init__(self, config: SkyLensConfig):
        super().__init__(config)
        self.config = config

        self.skylens = SkyLensModel(config)
        feat_ch = self.skylens.out_channels

        # 세그멘테이션 헤드 — 1x1 conv
        self.danger_head = nn.Conv2d(feat_ch, config.num_danger_classes, kernel_size=1)

        # 점 검출 헤드 (CenterNet) — 3x3 conv → 1x1 conv
        self.person_stem = SkyLensConvBlock(feat_ch, feat_ch)
        self.heatmap_head = nn.Conv2d(feat_ch, 1, kernel_size=1)
        self.wh_head = nn.Conv2d(feat_ch, 2, kernel_size=1)
        # CenterNet 관례: 초기 sigmoid ≈ 0.1 이 되도록 bias = -2.19
        self.heatmap_head._skylens_bias_init = -2.19
        # 출력 헤드는 kaiming(fan_out) 대상에서 제외한다 (_init_weights 주석 참조).
        for _head in (self.danger_head, self.heatmap_head, self.wh_head):
            _head._skylens_head_init = True

        self.post_init()

    # -- modality dropout ---------------------------------------------------

    def _sample_modality_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """배치마다 [rgb_present, thermal_present] 를 샘플링한다 (README §2.2).

        p=rgb_only  → thermal 제거 / p=thermal_only → RGB 제거 / 나머지 → 둘 다.
        """
        p_rgb_only = self.config.modality_dropout_rgb_only
        p_th_only = self.config.modality_dropout_thermal_only
        u = torch.rand(batch_size, device=device)

        rgb_present = torch.ones(batch_size, dtype=torch.bool, device=device)
        thermal_present = torch.ones(batch_size, dtype=torch.bool, device=device)
        thermal_present[u < p_rgb_only] = False
        rgb_present[(u >= p_rgb_only) & (u < p_rgb_only + p_th_only)] = False
        return torch.stack([rgb_present, thermal_present], dim=1)

    def _apply_modality_mask(
        self, pixel_values: torch.Tensor, modality_mask: torch.Tensor
    ) -> torch.Tensor:
        """결손 모달리티 채널을 0으로 만들고 validity 채널을 기록한다 (README §2.3).

        `use_validity_channel=True`면 마지막 채널이 validity(0/1)이며,
        열화상 결손 시 0으로 세팅해 "없음"과 "차갑다"의 모호성을 없앤다.
        """
        pixel_values = pixel_values.clone()
        rgb_present = modality_mask[:, 0].to(pixel_values.dtype).view(-1, 1, 1, 1)
        thermal_present = modality_mask[:, 1].to(pixel_values.dtype).view(-1, 1, 1, 1)

        n_in = self.config.in_channels
        pixel_values[:, :3] = pixel_values[:, :3] * rgb_present
        pixel_values[:, 3:n_in] = pixel_values[:, 3:n_in] * thermal_present

        if self.config.use_validity_channel and pixel_values.shape[1] > n_in:
            # 마지막 채널 = thermal validity mask
            pixel_values[:, n_in:] = thermal_present.expand(
                -1, pixel_values.shape[1] - n_in, pixel_values.shape[2], pixel_values.shape[3]
            )
        return pixel_values

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        modality_mask: Optional[torch.Tensor] = None,
        danger_labels: Optional[torch.LongTensor] = None,
        person_heatmap: Optional[torch.FloatTensor] = None,
        person_wh: Optional[torch.FloatTensor] = None,
        person_reg_mask: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> SkyLensOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, _, height, width = pixel_values.shape

        # 1) modality dropout — 학습 시에만 샘플링. 명시된 mask는 항상 존중한다.
        if modality_mask is not None:
            pixel_values = self._apply_modality_mask(pixel_values, modality_mask)
        elif self.training:
            sampled = self._sample_modality_mask(batch_size, pixel_values.device)
            pixel_values = self._apply_modality_mask(pixel_values, sampled)

        # 2) 인코더 + UNet 디코더
        features = self.skylens(pixel_values)

        # 3) 세그 헤드 — 입력 해상도로 bilinear 보간
        danger_logits = self.danger_head(features)
        if danger_logits.shape[-2:] != (height, width):
            danger_logits = F.interpolate(
                danger_logits, size=(height, width), mode="bilinear", align_corners=False
            )

        # 4) 점 검출 헤드 — person_head_stride 해상도
        stride = self.config.person_head_stride
        target_hw = (max(height // stride, 1), max(width // stride, 1))
        person_feat = features
        if person_feat.shape[-2:] != target_hw:
            person_feat = F.interpolate(
                person_feat, size=target_hw, mode="bilinear", align_corners=False
            )
        person_feat = self.person_stem(person_feat)
        heatmap_pred = torch.sigmoid(self.heatmap_head(person_feat))
        wh_pred = self.wh_head(person_feat)

        # 5) loss — GT가 없는 헤드는 건너뛴다 (README §6.3 헤드별 분리 학습)
        loss = None
        loss_dict: dict = {}

        if danger_labels is not None:
            seg_loss = F.cross_entropy(
                danger_logits,
                danger_labels.long(),
                ignore_index=self.config.danger_ignore_index,
            )
            loss_dict["danger_seg"] = seg_loss

        if person_heatmap is not None:
            hm_loss = centernet_focal_loss(heatmap_pred, person_heatmap)
            loss_dict["person_heatmap"] = hm_loss

        if person_wh is not None and person_reg_mask is not None:
            wh_loss = masked_l1_loss(wh_pred, person_wh, person_reg_mask)
            loss_dict["person_wh"] = wh_loss

        if loss_dict:
            weights = {
                "danger_seg": self.config.seg_loss_weight,
                "person_heatmap": self.config.heatmap_loss_weight,
                "person_wh": self.config.wh_loss_weight,
            }
            loss = sum(weights[k] * v for k, v in loss_dict.items())
        else:
            loss_dict = None

        if not return_dict:
            output = (danger_logits, heatmap_pred, wh_pred)
            return ((loss, loss_dict) + output) if loss is not None else output

        return SkyLensOutput(
            loss=loss,
            loss_dict=loss_dict,
            danger_logits=danger_logits,
            person_heatmap=heatmap_pred,
            person_wh=wh_pred,
        )


__all__ = [
    "SkyLensOutput",
    "SkyLensPreTrainedModel",
    "SkyLensModel",
    "SkyLensForDisasterPerception",
    "centernet_focal_loss",
    "masked_l1_loss",
    "inflate_first_conv",
]
