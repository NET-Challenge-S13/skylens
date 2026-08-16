"""SkyLens 전용 `TrainingArguments`.

README §7.3은 원래 커스텀 학습 루프를 택했지만, 실제로 필요한 커스터마이즈
(헤드별 loss, modality dropout, 컴포넌트 로깅)는 **모델 forward와 Trainer의
몇몇 훅**으로 흡수된다. 그래서 분산·mixed precision·체크포인트·로깅 같은
보일러플레이트는 HF `Trainer`에 맡기고, SkyLens 고유 항목만 여기에 얹는다.

modality dropout은 데이터 파이프라인/모델 쪽 책임이므로 여기에는 없다 (§2.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from transformers import TrainingArguments

__all__ = ["SkyLensTrainingArguments"]


@dataclass
class SkyLensTrainingArguments(TrainingArguments):
    """`TrainingArguments` + SkyLens 이중 헤드 학습 옵션."""

    person_head_stride: int = field(
        default=4,
        metadata={
            "help": (
                "점 검출 히트맵의 stride (입력 해상도 / 히트맵 해상도). "
                "평가 시 히트맵 좌표를 입력 이미지 좌표로 되돌리는 데 쓴다."
            )
        },
    )
    log_loss_components: bool = field(
        default=True,
        metadata={
            "help": (
                "모델이 반환한 loss_dict의 개별 컴포넌트(seg/heatmap/wh)를 "
                "각각 로깅한다. 헤드별 분리 학습(README §6.3)에서는 총합만 보면 "
                "어느 헤드가 학습 중인지 알 수 없어 사실상 필수."
            )
        },
    )
    eval_max_detections: int = field(
        default=100,
        metadata={"help": "평가 시 히트맵에서 뽑을 이미지당 최대 피크 수 (top-k)."},
    )
    eval_score_threshold: float = field(
        default=0.3,
        metadata={"help": "평가 시 검출로 인정할 최소 히트맵 점수."},
    )
    freeze_backbone_epochs: float = field(
        default=0.0,
        metadata={
            "help": (
                "학습 초반 백본을 동결할 에폭 수. 0이면 비활성. "
                "새로 붙인 헤드가 난수 상태로 프리트레인 백본을 망가뜨리는 것을 "
                "막는 워밍업 (README §6.1)."
            )
        },
    )
    point_distance_threshold: float = field(
        default=8.0,
        metadata={
            "help": (
                "점 검출 매칭 허용 거리(입력 해상도 픽셀). bbox IoU가 아니라 "
                "점 거리 기준인 이유는 README §1.4."
            )
        },
    )
    num_danger_classes: int = field(
        default=4,
        metadata={"help": "위험구역 세그 클래스 수 (정상/화재/붕괴/도로차단)."},
    )
    seg_ignore_index: int = field(
        default=255,
        metadata={"help": "세그 라벨의 ignore 인덱스. loss와 지표 모두에서 제외된다."},
    )

    # HF Trainer 의 기본값(True)은 우리 파이프라인과 충돌한다.
    # remove_unused_columns=True 이면 Trainer 가 data_collator 를 RemoveColumnsCollator 로
    # 감싸서, 모델 forward 시그니처에 없는 키를 **collator 호출 전에** 제거한다.
    # 우리 Dataset 은 raw 샘플("image"/"has_rgb"/"danger_mask"/"person_boxes")을 내고
    # SkyLensCollator 가 그것을 모델 입력으로 변환하는 구조라, 켜 두면 빈 dict 가 넘어와
    # KeyError('image') 로 죽는다. 따라서 기본값을 False 로 덮어쓴다.
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "SkyLens 는 collator 가 raw 샘플을 변환하므로 반드시 False 여야 한다."
        },
    )

    def __post_init__(self) -> None:  # noqa: D105
        super().__post_init__()
        if self.remove_unused_columns:
            raise ValueError(
                "SkyLensTrainingArguments 는 remove_unused_columns=False 여야 한다. "
                "True 이면 Trainer 가 'image' 등 raw 키를 SkyLensCollator 이전에 제거해 "
                "KeyError 가 발생한다."
            )
        if self.person_head_stride < 1:
            raise ValueError("person_head_stride >= 1 이어야 한다")
        if self.eval_max_detections < 1:
            raise ValueError("eval_max_detections >= 1 이어야 한다")
        if not 0.0 <= self.eval_score_threshold <= 1.0:
            raise ValueError("eval_score_threshold 는 [0, 1] 범위여야 한다")
        if self.freeze_backbone_epochs < 0:
            raise ValueError("freeze_backbone_epochs >= 0 이어야 한다")
