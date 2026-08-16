"""평가 지표 — 세그멘테이션(mIoU)과 점 검출(거리 기반 P/R/F1).

두 헤드의 성격이 다르므로 지표도 분리한다 (README §1.2):

- **세그 헤드**는 위험구역(stuff)을 다루므로 confusion matrix 기반 IoU가 표준.
- **점 검출 헤드**는 최종 산출물이 bbox가 아니라 *점*이다 (README §1.4 — 핀홀
  역투영이 `(u, v, Z)`를 받으므로 어차피 점 하나로 줄인다). 따라서 bbox IoU가
  아니라 **픽셀 거리 임계값 기반 그리디 매칭**으로 TP/FP/FN을 센다.

CenterNet 디코딩(`decode_heatmap_peaks`)은 순수 torch로 구현해 GPU에서
그대로 돌아가고 NMS 부속물이 필요 없다 (README §1.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "decode_heatmap_peaks",
    "SegmentationMetrics",
    "PointDetectionMetrics",
    "build_compute_metrics",
]


# --------------------------------------------------------------------------
# CenterNet 히트맵 디코딩
# --------------------------------------------------------------------------


def decode_heatmap_peaks(
    heatmap: torch.Tensor,
    wh: torch.Tensor | None = None,
    k: int = 100,
    threshold: float = 0.3,
    stride: int = 4,
) -> torch.Tensor:
    """CenterNet 스타일 피크 디코딩.

    3x3 max-pool NMS로 로컬 최대점만 남기고 top-k를 뽑아
    `(x, y, w, h, score)`를 입력 이미지 좌표계(= 히트맵 좌표 * stride)로 반환한다.
    별도의 IoU NMS는 쓰지 않는다 — max-pool 자체가 그 역할을 한다.

    Args:
        heatmap: `(B, 1, h, w)` 또는 `(B, h, w)` / `(h, w)`. **sigmoid가 이미
            적용된** 확률 맵을 가정한다 (모델 출력 규약).
        wh: `(B, 2, h, w)` 너비/높이 회귀 맵. 히트맵 격자 단위이며 stride를
            곱해 입력 해상도로 되돌린다. `None`이면 w=h=0으로 채운다.
        k: 배치당 최대 검출 수.
        threshold: 이 값 미만인 피크는 score=0으로 무효화된다.
        stride: 히트맵 → 입력 이미지 해상도 배율.

    Returns:
        `(B, k, 5)` 텐서. 마지막 축은 `(x, y, w, h, score)`이고,
        임계값을 못 넘긴 슬롯은 **score == 0**이므로 소비 측에서
        `score > 0`으로 걸러 쓰면 된다. (가변 길이 대신 고정 K 패딩을 쓰는 이유는
        Trainer의 배치 수집이 고정 shape을 요구하기 때문)
    """
    hm = heatmap
    if hm.dim() == 2:  # (h, w)
        hm = hm[None, None]
    elif hm.dim() == 3:  # (B, h, w)
        hm = hm[:, None]
    if hm.size(1) != 1:
        raise ValueError(f"heatmap 채널은 1이어야 한다 (got {tuple(hm.shape)})")

    b, _, fh, fw = hm.shape
    hm = hm.float()

    # 3x3 max-pool NMS: 자기 자신이 3x3 이웃의 최대값인 픽셀만 살린다.
    pooled = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
    peaks = torch.where(pooled == hm, hm, torch.zeros_like(hm))

    flat = peaks.view(b, -1)
    k_eff = min(int(k), flat.size(1))
    scores, idx = flat.topk(k_eff, dim=1)

    ys = (idx // fw).float()
    xs = (idx % fw).float()

    if wh is not None:
        w_ = wh
        if w_.dim() == 3:
            w_ = w_[None]
        if w_.shape[-2:] != (fh, fw):
            raise ValueError("wh의 공간 크기가 heatmap과 다르다")
        wh_flat = w_.reshape(w_.size(0), 2, -1).float()
        gather_idx = idx.unsqueeze(1).expand(-1, 2, -1)
        picked = wh_flat.gather(2, gather_idx)  # (B, 2, k)
        widths = picked[:, 0] * stride
        heights = picked[:, 1] * stride
    else:
        widths = torch.zeros_like(scores)
        heights = torch.zeros_like(scores)

    keep = scores >= float(threshold)
    scores = torch.where(keep, scores, torch.zeros_like(scores))

    out = torch.stack(
        [xs * stride, ys * stride, widths, heights, scores], dim=-1
    )  # (B, k_eff, 5)
    # 임계값 탈락 슬롯은 좌표까지 0으로 밀어 소비 측 실수를 줄인다.
    out = out * keep.unsqueeze(-1).to(out.dtype)

    if k_eff < k:  # 히트맵이 k보다 작을 때 0 패딩
        pad = out.new_zeros((b, k - k_eff, 5))
        out = torch.cat([out, pad], dim=1)
    return out


# --------------------------------------------------------------------------
# 세그멘테이션 지표
# --------------------------------------------------------------------------


class SegmentationMetrics:
    """confusion matrix를 누적해 mIoU / per-class IoU / pixel accuracy를 낸다.

    배치마다 IoU를 구해 평균내면 클래스가 없는 배치에서 값이 왜곡되므로,
    전체 평가셋의 confusion matrix를 한 번에 쌓은 뒤 마지막에 계산한다.
    `ignore_index` 픽셀은 행렬에 아예 들어가지 않는다.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        if num_classes < 1:
            raise ValueError("num_classes >= 1")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.long
        )

    def reset(self) -> None:
        self.confusion.zero_()

    @staticmethod
    def _as_long_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.detach().reshape(-1).to(torch.long).cpu()
        return torch.as_tensor(np.asarray(x).reshape(-1), dtype=torch.long)

    def update(
        self,
        preds: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
    ) -> None:
        """`preds`/`targets` 모두 클래스 인덱스 맵 `(...,H,W)`. 로짓이 아니다."""
        p = self._as_long_tensor(preds)
        t = self._as_long_tensor(targets)
        if p.numel() != t.numel():
            raise ValueError(
                f"preds/targets 크기 불일치: {p.numel()} vs {t.numel()}"
            )

        valid = (t != self.ignore_index) & (t >= 0) & (t < self.num_classes)
        # 예측이 범위를 벗어나면(방어적) 무시한다.
        valid &= (p >= 0) & (p < self.num_classes)
        if not bool(valid.any()):
            return
        p, t = p[valid], t[valid]

        # 행 = GT, 열 = 예측
        idx = t * self.num_classes + p
        binc = torch.bincount(idx, minlength=self.num_classes**2)
        self.confusion += binc.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float]:
        cm = self.confusion.to(torch.float64)
        tp = cm.diag()
        gt = cm.sum(dim=1)
        pred = cm.sum(dim=0)
        union = gt + pred - tp

        # 등장하지 않은 클래스(union==0)는 mIoU 평균에서 제외한다.
        present = union > 0
        iou = torch.where(present, tp / union.clamp(min=1.0), torch.zeros_like(tp))

        total = cm.sum()
        pixel_acc = float(tp.sum() / total) if float(total) > 0 else 0.0
        miou = float(iou[present].mean()) if bool(present.any()) else 0.0

        out: dict[str, float] = {"miou": miou, "pixel_accuracy": pixel_acc}
        for c in range(self.num_classes):
            out[f"iou_class_{c}"] = float(iou[c]) if bool(present[c]) else float("nan")
        return out


# --------------------------------------------------------------------------
# 점 검출 지표
# --------------------------------------------------------------------------

_Point = Sequence[float]


@dataclass
class PointDetectionMetrics:
    """점 거리 기반 그리디 매칭 P/R/F1.

    bbox IoU를 쓰지 않는 이유는 README §1.4 — 파이프라인의 최종 산출물이
    점이고, 소형 객체에서는 bbox IoU가 몇 픽셀 흔들림에 과도하게 민감하다.

    매칭 규칙: 예측을 score 내림차순으로 정렬 → 각 예측에 대해 아직 안 쓰인 GT 중
    가장 가까운 것을 고른다 → 거리 <= `distance_threshold`면 TP, 아니면 FP.
    남은 GT는 FN.
    """

    distance_threshold: float = 8.0
    tp: int = field(default=0)
    fp: int = field(default=0)
    fn: int = field(default=0)

    def reset(self) -> None:
        self.tp = self.fp = self.fn = 0

    def update(
        self,
        preds: Iterable[_Point] | torch.Tensor | np.ndarray,
        gts: Iterable[_Point] | torch.Tensor | np.ndarray,
    ) -> None:
        """한 이미지 분의 예측/GT를 누적.

        Args:
            preds: `(x, y, score)` 또는 `(x, y)` 시퀀스/배열.
            gts: `(x, y)` (뒤 성분은 무시) 시퀀스/배열.
        """
        p = _to_xy_score(preds)
        g = _to_xy(gts)

        if p.size == 0:
            self.fn += int(g.shape[0])
            return
        if g.size == 0:
            self.fp += int(p.shape[0])
            return

        order = np.argsort(-p[:, 2])
        used = np.zeros(g.shape[0], dtype=bool)
        thr = float(self.distance_threshold)

        for i in order:
            d = np.hypot(g[:, 0] - p[i, 0], g[:, 1] - p[i, 1])
            d[used] = np.inf
            j = int(np.argmin(d))
            if np.isfinite(d[j]) and d[j] <= thr:
                used[j] = True
                self.tp += 1
            else:
                self.fp += 1
        self.fn += int((~used).sum())

    def compute(self) -> dict[str, float]:
        tp, fp, fn = float(self.tp), float(self.fp), float(self.fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {
            "point_precision": precision,
            "point_recall": recall,
            "point_f1": f1,
            "point_tp": tp,
            "point_fp": fp,
            "point_fn": fn,
        }


def _to_xy_score(x: object) -> np.ndarray:
    arr = _to_array(x)
    if arr.size == 0:
        return arr.reshape(0, 3)
    if arr.shape[1] == 2:  # score 없으면 1.0으로 채운다
        arr = np.concatenate([arr, np.ones((arr.shape[0], 1))], axis=1)
    return arr[:, :3].astype(np.float64)


def _to_xy(x: object) -> np.ndarray:
    arr = _to_array(x)
    if arr.size == 0:
        return arr.reshape(0, 2)
    return arr[:, :2].astype(np.float64)


def _to_array(x: object) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy().astype(np.float64)
    elif isinstance(x, np.ndarray):
        arr = x.astype(np.float64)
    else:
        arr = np.asarray(list(x), dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(0, 0)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


# --------------------------------------------------------------------------
# Trainer 연동
# --------------------------------------------------------------------------


def build_compute_metrics(
    num_classes: int,
    ignore_index: int = 255,
    distance_threshold: float = 8.0,
    score_threshold: float = 0.3,
) -> Callable[[object], dict[str, float]]:
    """`Trainer(compute_metrics=...)` 로 넘길 함수를 만든다.

    `SkyLensTrainer.prediction_step`이 내놓는 규약을 소비한다:

    - `predictions = (seg_pred (B,H,W) int, detections (B,K,5))`
    - `label_ids   = (seg_labels (B,H,W) int, gt_points (B,K,3: x,y,valid))`

    한쪽만 존재하는 배치(README §6.3의 헤드별 분리 학습)를 위해, 무효 항목은
    seg는 전부 `ignore_index`, 점은 `valid == 0`으로 들어온다고 가정한다.
    """
    seg_thr_ignore = int(ignore_index)

    def compute_metrics(eval_pred: object) -> dict[str, float]:
        preds = getattr(eval_pred, "predictions", None)
        labels = getattr(eval_pred, "label_ids", None)
        if preds is None and isinstance(eval_pred, (tuple, list)):
            preds, labels = eval_pred[0], eval_pred[1]

        seg_pred, det = _unpack_pair(preds)
        seg_gt, pt_gt = _unpack_pair(labels)

        results: dict[str, float] = {}

        if seg_pred is not None and seg_gt is not None:
            seg = SegmentationMetrics(num_classes, seg_thr_ignore)
            seg.update(torch.as_tensor(np.asarray(seg_pred)), torch.as_tensor(np.asarray(seg_gt)))
            results.update(seg.compute())

        if det is not None and pt_gt is not None:
            pdm = PointDetectionMetrics(distance_threshold=distance_threshold)
            det_a = np.asarray(det, dtype=np.float64)
            gt_a = np.asarray(pt_gt, dtype=np.float64)
            for b in range(det_a.shape[0]):
                d = det_a[b]
                d = d[d[:, 4] >= score_threshold][:, [0, 1, 4]]
                g = gt_a[b]
                g = g[g[:, 2] > 0][:, :2]
                if d.size == 0 and g.size == 0:
                    continue
                pdm.update(d, g)
            results.update(pdm.compute())

        return results

    return compute_metrics


def _unpack_pair(x: object) -> tuple[object | None, object | None]:
    """`(a, b)` 튜플/리스트를 풀되 단일 값·None도 견딘다."""
    if x is None:
        return None, None
    if isinstance(x, (tuple, list)):
        if len(x) >= 2:
            return x[0], x[1]
        if len(x) == 1:
            return x[0], None
        return None, None
    return x, None
