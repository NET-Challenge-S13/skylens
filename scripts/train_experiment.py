"""Run one SkyLens training experiment from the command line.

`train.ipynb` is for reading and exploring. An experiment needs the same run to
be repeatable from a branch, with its settings written down, so this script
mirrors the notebook and takes the settings that an experiment changes as
arguments. Everything else stays at the notebook's values so a run here is
comparable with the baseline in `runs/skylens/checkpoint-14100`.

Final metrics go to the branch's pull request through `researchtree`, if it is
installed and the branch has an open PR. It never raises into training.

Example:
    uv run python scripts/train_experiment.py \
        --run-name seg-class-weights \
        --epochs 2 \
        --danger-class-weights 1.0 10.19 4.67 7.52
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# 노트북과 같은 값. 실험이 바꾸는 것만 인자로 뺀다.
IMAGE_SIZE = 512
PERSON_HEAD_STRIDE = 4
NUM_DANGER_CLASSES = 4
BACKBONE = "microsoft/resnet-50"
BATCH_SIZE = 8
LR = 1e-4
SAVE_EVERY_STEPS = 300
EVAL_MAX_SAMPLES = 1200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True, help="runs/<name> 아래에 저장한다")
    p.add_argument("--epochs", type=float, default=5.0)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument(
        "--danger-class-weights",
        type=float,
        nargs=NUM_DANGER_CLASSES,
        default=None,
        metavar=("NORMAL", "FIRE", "COLLAPSE", "ROAD"),
        help="세그 CrossEntropy 의 클래스 가중치. 주지 않으면 가중치 없이 학습한다(기준선)",
    )
    p.add_argument("--data-root", type=Path, default=None, help="기본값은 자동 탐색")
    p.add_argument("--eval-max-samples", type=int, default=EVAL_MAX_SAMPLES)
    p.add_argument("--no-resume", action="store_true", help="체크포인트가 있어도 처음부터 학습한다")
    p.add_argument("--no-log-pr", action="store_true", help="결과를 PR 에 적지 않는다")
    return p.parse_args()


def resolve_data_root(explicit: Path | None) -> Path:
    """데이터셋 디렉터리를 찾는다.

    실험은 `.worktrees/<이름>` 워크트리에서 돌지만 `data/` 는 gitignore 대상이라
    메인 체크아웃에만 있다. 워크트리마다 23GB 를 복사할 수는 없으므로, 여기서
    메인 체크아웃을 찾아 그쪽 `data/` 를 함께 쓴다. 리사이즈 캐시(`data/_cache`)도
    같이 공유되어 실험마다 다시 만들지 않는다.
    """
    if explicit is not None:
        return explicit

    here = Path("data")
    if here.exists():
        return here

    # git 공통 디렉터리(.git)의 부모가 메인 체크아웃이다.
    import subprocess

    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        candidate = Path(common).parent / "data"
        if candidate.exists():
            print(f"data/ 가 없어 메인 체크아웃의 데이터를 쓴다: {candidate}")
            return candidate
    except Exception:
        pass

    return here  # 없으면 아래에서 안내 메시지와 함께 실패한다


def main() -> int:
    args = parse_args()

    import numpy as np
    import torch

    print(f"python {sys.version.split()[0]} · torch {torch.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"device {device} ({torch.cuda.get_device_name(0)})")
    else:
        print("device cpu — 학습이 매우 느리다")

    import albumentations as A
    from torch.utils.data import ConcatDataset, Subset

    from skylens_model.datasets import (
        LLVIP,
        SARD,
        FireSegmentation,
        RescueNetSegmentation,
        VisDronePerson,
    )
    from skylens_model.models import SkyLensConfig, SkyLensForDisasterPerception
    from skylens_model.utils import ResizedCache, SkyLensCollator, build_resized_cache
    from skylens_model.utils.callbacks import GracefulInterruptCallback, find_resume_checkpoint
    from skylens_model.utils.metrics import build_compute_metrics
    from skylens_model.utils.trainer import SkyLensTrainer
    from skylens_model.utils.training_args import SkyLensTrainingArguments

    # --- 증강 (노트북 §2) --------------------------------------------------
    tf = A.Compose(
        [A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.2)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["cls"], min_visibility=0.2),
    )

    def train_aug(sample: dict) -> dict:
        boxes = sample.get("person_boxes")
        has_boxes = boxes is not None and len(boxes) > 0
        mask = sample.get("danger_mask")
        kw = {
            "image": sample["image"],
            "bboxes": boxes.tolist() if has_boxes else [],
            "cls": [0] * (len(boxes) if has_boxes else 0),
        }
        if mask is not None:
            kw["mask"] = mask
        out = tf(**kw)
        sample = dict(sample)
        sample["image"] = out["image"]
        sample["danger_mask"] = out["mask"] if mask is not None else None
        if sample.get("person_boxes") is not None:
            b = out["bboxes"]
            sample["person_boxes"] = np.asarray(b, dtype=np.float32) if b else np.zeros((0, 4), np.float32)
        return sample

    # --- 데이터셋 (노트북 §3) ----------------------------------------------
    data_root = resolve_data_root(args.data_root)
    cache_root = data_root / "_cache"
    sources = [
        (LLVIP, data_root / "llvip" / "LLVIP", "train", "test"),
        (VisDronePerson, data_root / "visdrone", "train", "val"),
        (RescueNetSegmentation, data_root / "rescuenet", "train", "test"),
        (FireSegmentation, data_root / "fire_seg", "train", "val"),
        (SARD, data_root / "sard", "train", "val"),
    ]

    def build_split(which: str, augment):
        parts = []
        for cls, root, tr_split, ev_split in sources:
            split = tr_split if which == "train" else ev_split
            if not root.exists():
                print(f"  [없음] {cls.__name__:24s} {root}")
                continue
            try:
                raw = cls(root, split=split)
                cache_dir = cache_root / f"{cls.__name__}_{split}_{IMAGE_SIZE}"
                build_resized_cache(raw, cache_dir, IMAGE_SIZE)
                ds = ResizedCache(cache_dir, transforms=augment)
                print(f"  [ok]   {cls.__name__:24s} {split:5s} {len(ds):>6,}장")
                parts.append(ds)
            except Exception as exc:
                print(f"  [skip] {cls.__name__:24s} {type(exc).__name__}: {str(exc).splitlines()[0][:60]}")
        return ConcatDataset(parts) if parts else None

    print("[train]")
    train_ds = build_split("train", train_aug)
    print("[eval]")
    eval_ds = build_split("eval", None)
    if train_ds is None:
        raise RuntimeError("학습 데이터가 없다. src/skylens_model/datasets/README.md 참조")

    if eval_ds is not None and len(eval_ds) > args.eval_max_samples:
        stride = len(eval_ds) // args.eval_max_samples + 1
        eval_ds = Subset(eval_ds, list(range(0, len(eval_ds), stride)))
    print(f"\ntrain {len(train_ds):,} | eval {len(eval_ds):,}\n")

    # --- 모델 (노트북 §5) --------------------------------------------------
    config = SkyLensConfig(
        backbone=BACKBONE,
        use_pretrained_backbone=True,
        in_channels=4,
        num_danger_classes=NUM_DANGER_CLASSES,
        person_head_stride=PERSON_HEAD_STRIDE,
        modality_dropout_rgb_only=0.25,
        modality_dropout_thermal_only=0.25,
        seg_loss_weight=1.0,
        heatmap_loss_weight=1.0,
        wh_loss_weight=0.1,
        danger_class_weights=args.danger_class_weights,
    )
    model = SkyLensForDisasterPerception(config)
    print(f"파라미터 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"세그 클래스 가중치: {args.danger_class_weights or '없음(기준선)'}\n")

    # --- 학습 (노트북 §6~7) ------------------------------------------------
    output_dir = Path("runs") / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    targs = SkyLensTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4,
        eval_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        logging_steps=25,
        save_strategy="steps",
        save_steps=SAVE_EVERY_STEPS,
        save_total_limit=3,
        eval_strategy="epoch",
        load_best_model_at_end=False,
        report_to=[],
        fp16=(device == "cuda"),
        dataloader_num_workers=0,
        person_head_stride=PERSON_HEAD_STRIDE,
        num_danger_classes=NUM_DANGER_CLASSES,
        freeze_backbone_epochs=0.0,
        eval_score_threshold=0.3,
        point_distance_threshold=8.0,
    )

    trainer = SkyLensTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=SkyLensCollator(
            person_head_stride=PERSON_HEAD_STRIDE,
            validity_channel=False,
            modality_dropout=(0.0, 0.0),
        ),
        # 이것이 없으면 평가가 손실만 내고 mIoU·클래스별 IoU·사람 점지표가 전부
        # 빠진다. 판정 기준이 그 지표들이라 빠지면 실험이 무의미해진다.
        compute_metrics=build_compute_metrics(
            num_classes=NUM_DANGER_CLASSES,
            distance_threshold=targs.point_distance_threshold,
            score_threshold=targs.eval_score_threshold,
        ),
        callbacks=[GracefulInterruptCallback()],
    )

    resume = None if args.no_resume else find_resume_checkpoint(output_dir)
    print(f"재개 지점: {resume}" if resume else "체크포인트 없음, 처음부터 학습")
    result = trainer.train(resume_from_checkpoint=resume)
    print(f"\n[완료] train_loss {result.training_loss:.4f} | steps {result.global_step}")

    metrics = trainer.evaluate()

    # 판정에 쓰는 지표가 실제로 나왔는지 확인한다. 한 번 compute_metrics 를 빠뜨려
    # 90분을 손실만 보고 태운 적이 있다.
    required = ("eval_miou", "eval_iou_class_3", "eval_point_f1")
    missing = [k for k in required if k not in metrics]
    if missing:
        raise RuntimeError(
            "평가에 판정 지표가 없다: " + ", ".join(missing) + ". "
            "compute_metrics 가 Trainer 에 전달됐는지 확인할 것."
        )

    print("\n=== 최종 평가 ===")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:32s} {v:.4f}")

    trainer.save_model(str(output_dir / "final"))
    (output_dir / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not args.no_log_pr:
        log_to_pr(metrics, trainer.state.log_history, args)
    return 0


# 사람이 읽을 때 이 순서로 보는 것이 자연스럽다. 여기 없는 지표도 전부 기록하되 뒤로 밀린다.
METRIC_ORDER = (
    "eval_loss",
    "eval_miou",
    "eval_pixel_accuracy",
    "eval_iou_class_0",
    "eval_iou_class_1",
    "eval_iou_class_2",
    "eval_iou_class_3",
    "eval_point_precision",
    "eval_point_recall",
    "eval_point_f1",
    "eval_point_tp",
    "eval_point_fp",
    "eval_point_fn",
    "loss_danger_seg",
    "loss_person_heatmap",
    "loss_person_wh",
    "eval_runtime",
    "eval_samples_per_second",
)

METRIC_LABEL = {
    "eval_loss": "총 손실",
    "eval_miou": "mIoU",
    "eval_pixel_accuracy": "픽셀 정확도",
    "eval_iou_class_0": "IoU normal",
    "eval_iou_class_1": "IoU fire",
    "eval_iou_class_2": "IoU collapse",
    "eval_iou_class_3": "IoU road_blocked",
    "eval_point_precision": "사람 precision",
    "eval_point_recall": "사람 recall",
    "eval_point_f1": "사람 F1",
    "eval_point_tp": "사람 TP",
    "eval_point_fp": "사람 FP",
    "eval_point_fn": "사람 FN",
    "loss_danger_seg": "손실 세그",
    "loss_person_heatmap": "손실 히트맵",
    "loss_person_wh": "손실 크기",
    "eval_runtime": "평가 시간(초)",
    "eval_samples_per_second": "평가 처리량(장/초)",
}

# 강조해서 보여줄 지표. 판정 기준에 직접 쓰이는 것들이다.
HIGHLIGHT = ("eval_miou", "eval_iou_class_3", "eval_point_f1")


def _numeric(d: dict) -> dict:
    """숫자 지표만 남긴다. nan 은 None 으로 바꿔 '계산 안 됨' 과 0 을 구분한다."""
    out = {}
    for k, v in d.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        out[k] = None if v != v else round(float(v), 4)
    return out


def _ordered(keys) -> list:
    known = [k for k in METRIC_ORDER if k in keys]
    return known + sorted(k for k in keys if k not in METRIC_ORDER)


def _cell(value) -> str:
    if value is None:
        return "없음"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.4f}"


def build_metrics_table(history: list, final: dict) -> str:
    """에폭별 전 지표 표. 최종 수치 하나로는 추세도 과적합도 보이지 않는다."""
    rows_src = [h for h in history if "eval_miou" in h]
    if rows_src:
        cols = [_numeric(h) for h in rows_src]
        labels = [f"epoch {round(float(h.get('epoch', i + 1)), 2)}" for i, h in enumerate(rows_src)]
    else:
        cols, labels = [_numeric(final)], ["최종"]

    keys = _ordered({k for c in cols for k in c})
    lines = [
        "| 지표 | 키 | " + " | ".join(labels) + " |",
        "|---|---|" + "---|" * len(labels),
    ]
    for key in keys:
        label = METRIC_LABEL.get(key, key)
        if key in HIGHLIGHT:
            label = f"**{label}**"
        cells = " | ".join(_cell(c.get(key)) for c in cols)
        lines.append(f"| {label} | `{key}` | {cells} |")
    return "\n".join(lines)


def update_pr_section(heading: str, body_md: str) -> bool:
    """PR 본문의 한 절만 갈아 끼운다. 맨 위 YAML 블록과 다른 절은 건드리지 않는다."""
    import subprocess

    def gh(*argv):
        return subprocess.run(["gh", *argv], capture_output=True, text=True, encoding="utf-8")

    got = gh("pr", "view", "--json", "body", "-q", ".body")
    if got.returncode != 0:
        print("PR 본문을 읽지 못했다: " + got.stderr.strip()[:100])
        return False

    marker = "## " + heading
    block = marker + "\n\n" + body_md + "\n"
    body = got.stdout

    if marker in body:
        before, rest = body.split(marker, 1)
        idx = rest.find("\n## ")
        after = rest[idx + 1 :] if idx != -1 else ""
        body = before + block + ("\n" + after if after else "")
    else:
        body = body.rstrip() + "\n\n" + block

    tmp = Path("runs") / "_pr_body.md"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(body, encoding="utf-8")
    put = gh("pr", "edit", "--body-file", str(tmp))
    tmp.unlink(missing_ok=True)
    if put.returncode != 0:
        print("PR 본문을 쓰지 못했다: " + put.stderr.strip()[:100])
        return False
    return True


def log_to_pr(metrics: dict, history: list, args: argparse.Namespace) -> None:
    """실험 브랜치의 PR 에 지표를 적는다. 실패해도 학습 결과에는 영향이 없다.

    두 곳에 적는다. YAML 블록에는 최종 수치를 전부 넣어 트리에서 실험끼리 비교할 수
    있게 하고, 본문에는 에폭별 전 지표 표를 넣어 추세를 볼 수 있게 한다. F1 하나만
    남기면 precision 과 recall 중 어느 쪽을 버려서 얻은 값인지 알 수 없다.
    """
    payload = {k: v for k, v in _numeric(metrics).items() if v is not None}
    payload["epochs"] = args.epochs

    try:
        import researchtree as rt

        rt.log(**payload)
        print("\nPR YAML 에 지표 " + str(len(payload)) + "개 기록")
    except ImportError:
        print("\nresearchtree 가 없어 YAML 기록을 건너뛴다 (uv add researchtree)")
    except Exception as exc:  # 학습 결과를 잃지 않는다
        print("\nYAML 기록 실패: " + type(exc).__name__ + ": " + str(exc)[:120])

    weights = args.danger_class_weights or "없음(가중치 미적용)"
    note = (
        f"학습 조건: {args.epochs} 에폭 · batch {args.batch_size} · lr {args.lr} · "
        f"세그 클래스 가중치 {weights}\n\n"
        + build_metrics_table(history, metrics)
        + "\n\n`없음` 은 평가셋에 그 클래스가 없어 계산되지 않았다는 뜻이고, 0 과 다르다."
    )
    if update_pr_section("측정값 전체", note):
        print("PR 본문에 에폭별 지표 표 기록")


if __name__ == "__main__":
    raise SystemExit(main())
