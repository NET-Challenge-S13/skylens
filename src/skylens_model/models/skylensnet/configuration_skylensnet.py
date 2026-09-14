"""SkyLens 모델 설정 (HuggingFace `PretrainedConfig` 규약).

설계 근거는 `src/skylens_model/README.md` 참조:
- §1 단일 백본 + 이중 헤드 (세그멘테이션 / 점 검출)
- §2 4채널 early fusion + 대칭 modality dropout + 결손 채널 인코딩
- §6.1 ImageNet 프리트레인 백본 + 첫 conv 4채널 inflation
- §7 `transformers` 인코더(AutoBackbone) + 자체 UNet 디코더/헤드
"""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class SkyLensConfig(PretrainedConfig):
    r"""[`SkyLensModel`] / [`SkyLensForDisasterPerception`] 설정.

    Args:
        backbone (`str`):
            `AutoBackbone`에 넘길 인코더 이름 (예: `"microsoft/resnet-50"`).
        use_timm_backbone (`bool`):
            True면 timm 생태계 백본(`TimmBackbone`)으로 로드한다.
        use_pretrained_backbone (`bool`):
            True면 사전학습 가중치를 로드하고 §6.1의 4채널 inflation을 적용한다.
            False면 구조만 만들고 첫 conv 채널 수만 맞춘다.
        backbone_out_indices (`tuple`):
            멀티스케일 feature를 뽑을 스테이지 인덱스. UNet 스킵 커넥션용.
        in_channels (`int`):
            입력 채널 수. 기본 4 = RGB(3) + thermal(1).
        use_validity_channel (`bool`):
            True면 마지막에 0/1 validity 채널이 하나 더 붙는다(README §2.3).
            즉 실제 입력 채널 수는 `in_channels + 1`.
        decoder_channels (`tuple`):
            UNet 디코더 각 업샘플 블록의 출력 채널 수.
        num_danger_classes (`int`):
            위험구역 클래스 수. 0=정상 1=화재 2=붕괴 3=도로차단.
        person_head_stride (`int`):
            점 검출 헤드(히트맵/wh)가 동작하는 출력 stride.
        modality_dropout_rgb_only (`float`):
            학습 중 thermal을 지워 RGB-only로 만들 확률.
        modality_dropout_thermal_only (`float`):
            학습 중 RGB를 지워 thermal-only로 만들 확률.
        seg_loss_weight / heatmap_loss_weight / wh_loss_weight (`float`):
            총 loss 가중합 계수.
        danger_ignore_index (`int`):
            세그 CrossEntropy에서 무시할 라벨 값.
    """

    model_type = "skylens"

    def __init__(
        self,
        backbone: str = "microsoft/resnet-50",
        use_timm_backbone: bool = False,
        use_pretrained_backbone: bool = True,
        backbone_out_indices: tuple = (0, 1, 2, 3),
        in_channels: int = 4,
        use_validity_channel: bool = False,
        decoder_channels: tuple = (256, 128, 64, 32),
        num_danger_classes: int = 4,
        person_head_stride: int = 4,
        modality_dropout_rgb_only: float = 0.25,
        modality_dropout_thermal_only: float = 0.25,
        seg_loss_weight: float = 1.0,
        heatmap_loss_weight: float = 1.0,
        wh_loss_weight: float = 0.1,
        danger_ignore_index: int = 255,
        **kwargs,
    ):
        self.backbone = backbone
        self.use_timm_backbone = use_timm_backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        # JSON round-trip 시 list로 돌아오므로 항상 tuple로 정규화한다.
        self.backbone_out_indices = tuple(backbone_out_indices)
        self.in_channels = in_channels
        self.use_validity_channel = use_validity_channel
        self.decoder_channels = tuple(decoder_channels)
        self.num_danger_classes = num_danger_classes
        self.person_head_stride = person_head_stride
        self.modality_dropout_rgb_only = modality_dropout_rgb_only
        self.modality_dropout_thermal_only = modality_dropout_thermal_only
        self.seg_loss_weight = seg_loss_weight
        self.heatmap_loss_weight = heatmap_loss_weight
        self.wh_loss_weight = wh_loss_weight
        self.danger_ignore_index = danger_ignore_index

        if self.modality_dropout_rgb_only + self.modality_dropout_thermal_only > 1.0:
            raise ValueError(
                "modality_dropout_rgb_only + modality_dropout_thermal_only 는 1.0 이하여야 한다 "
                f"(현재 {self.modality_dropout_rgb_only} + {self.modality_dropout_thermal_only})."
            )
        if self.in_channels < 4:
            raise ValueError("in_channels 는 최소 4 (RGB 3 + thermal 1) 여야 한다.")

        super().__init__(**kwargs)

    @property
    def num_input_channels(self) -> int:
        """백본 첫 conv가 실제로 받아야 하는 채널 수 (validity 채널 포함)."""
        return self.in_channels + (1 if self.use_validity_channel else 0)


__all__ = ["SkyLensConfig"]
