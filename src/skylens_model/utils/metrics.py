"""평가 지표 — 세그멘테이션(mIoU), 점 검출(거리 기반 P/R/F1 · AP), bbox mAP.

두 헤드의 성격이 다르므로 지표도 분리한다 (README §1.2):

- **세그 헤드**는 위험구역(stuff)을 다루므로 confusion matrix 기반 IoU가 표준.
- **점 검출 헤드**는 최종 산출물이 bbox가 아니라 *점*이다 (README §1.4 — 핀홀
  역투영이 `(u, v, Z)`를 받으므로 어차피 점 하나로 줄인다). 따라서 bbox IoU가
  아니라 **픽셀 거리 임계값 기반 그리디 매칭**으로 TP/FP/FN을 센다.
  단일 score 임계값 의존을 없앤 `PointAveragePrecision`(거리 기반 AP)도 함께 낸다.
- **문헌 비교용**으로 `wh` 회귀 헤드가 복원한 박스에 대해 COCO 스타일
  `BoxDetectionMetrics`(mAP@0.5 · mAP@0.5:0.95, 101-point 보간)를 제공한다.

CenterNet 디코딩(`decode_heatmap_peaks`)은 순수 torch로 구현해 GPU에서
그대로 돌아가고 NMS 부속물이 필요 없다 (README §1.4).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "decode_heatmap_peaks",
    "decode_gt_boxes",
    "box_iou_matrix",
    "SegmentationMetrics",
    "PointDetectionMetrics",
    "BoxDetectionMetrics",
    "PointAveragePrecision",
    "build_compute_metrics",
]

#: COCO 규약의 IoU 임계값 (0.50:0.05:0.95, 10단계).
COCO_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


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


def decode_gt_boxes(
    reg_mask: torch.Tensor,
    wh: torch.Tensor,
    k: int = 100,
    stride: int = 4,
) -> torch.Tensor:
    """collator의 CenterNet 타깃에서 GT 박스를 복원한다.

    `reg_mask == 1`인 격자 위치가 객체 중심이고, 같은 위치의 `wh`가 격자 단위
    `(w, h)`다. 둘 다 stride를 곱해 입력 이미지 해상도로 되돌린다.

    Args:
        reg_mask: `(B, 1, h, w)` 또는 `(B, h, w)`. 중심에서 1, 나머지 0.
        wh: `(B, 2, h, w)` 너비/높이 회귀 타깃 (격자 단위).
        k: 이미지당 최대 GT 수 (고정 shape 유지를 위한 패딩 길이).
        stride: 격자 → 입력 이미지 해상도 배율.

    Returns:
        `(B, k, 5)` 텐서, 마지막 축은 `(x, y, w, h, valid)`. 빈 슬롯은 전부 0.
    """
    m = reg_mask
    if m.dim() == 3:
        m = m[:, None]
    if m.size(1) != 1:
        raise ValueError(f"reg_mask 채널은 1이어야 한다 (got {tuple(m.shape)})")
    if wh.dim() == 3:
        wh = wh[None]
    if wh.shape[-2:] != m.shape[-2:]:
        raise ValueError("wh의 공간 크기가 reg_mask와 다르다")

    b, _, _, fw = m.shape
    flat = m.reshape(b, -1).float()
    k_eff = min(int(k), flat.size(1))
    valid, idx = flat.topk(k_eff, dim=1)
    valid = (valid > 0).to(torch.float32)

    ys = (idx // fw).float()
    xs = (idx % fw).float()

    wh_flat = wh.reshape(b, 2, -1).float()
    picked = wh_flat.gather(2, idx.unsqueeze(1).expand(-1, 2, -1))  # (B, 2, k)

    out = torch.stack(
        [
            xs * stride,
            ys * stride,
            picked[:, 0] * stride,
            picked[:, 1] * stride,
            valid,
        ],
        dim=-1,
    )
    out = out * valid.unsqueeze(-1)

    if k_eff < k:
        out = torch.cat([out, out.new_zeros((b, k - k_eff, 5))], dim=1)
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
# Average Precision (COCO 101-point)
# --------------------------------------------------------------------------


def _average_precision(scores: np.ndarray, is_tp: np.ndarray, num_gt: int) -> float:
    """101-point interpolated AP (COCO 규약).

    Args:
        scores: 전체 예측의 confidence `(N,)`.
        is_tp: 같은 순서의 TP 여부 `(N,)` bool.
        num_gt: 전체 GT 개수.
    """
    if num_gt <= 0:
        return 0.0
    if scores.size == 0:
        return 0.0

    order = np.argsort(-scores, kind="stable")
    tp = is_tp[order].astype(np.float64)
    fp = 1.0 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / float(num_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

    # precision envelope: 뒤에서부터 누적 최대 → 단조 감소로 만든다.
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    rec_points = np.linspace(0.0, 1.0, 101)
    idx = np.searchsorted(recall, rec_points, side="left")
    interp = np.where(idx < precision.size, precision[np.minimum(idx, precision.size - 1)], 0.0)
    return float(interp.mean())


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """중심 형식 박스 `(x, y, w, h)` 두 집합의 IoU 행렬 `(len(a), len(b))`."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0] if a.ndim > 1 else 0, b.shape[0] if b.ndim > 1 else 0))

    def to_xyxy(t: np.ndarray) -> np.ndarray:
        x, y, w, h = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
        return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)

    aa, bb = to_xyxy(np.asarray(a, np.float64)), to_xyxy(np.asarray(b, np.float64))

    lt = np.maximum(aa[:, None, :2], bb[None, :, :2])
    rb = np.minimum(aa[:, None, 2:], bb[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]

    area_a = np.clip(aa[:, 2] - aa[:, 0], 0, None) * np.clip(aa[:, 3] - aa[:, 1], 0, None)
    area_b = np.clip(bb[:, 2] - bb[:, 0], 0, None) * np.clip(bb[:, 3] - bb[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, np.finfo(np.float64).eps), 0.0)


class _APAccumulator:
    """여러 이미지의 (score, TP 여부)와 GT 개수를 모아 AP를 낸다."""

    def __init__(self, num_thresholds: int) -> None:
        self._scores: list[np.ndarray] = []
        self._tp: list[list[np.ndarray]] = [[] for _ in range(num_thresholds)]
        self.num_gt = 0

    def reset(self) -> None:
        self._scores.clear()
        for lst in self._tp:
            lst.clear()
        self.num_gt = 0

    def add(self, scores: np.ndarray, tp_per_threshold: Sequence[np.ndarray], n_gt: int) -> None:
        self._scores.append(scores)
        for i, tp in enumerate(tp_per_threshold):
            self._tp[i].append(tp)
        self.num_gt += int(n_gt)

    def ap_per_threshold(self) -> list[float]:
        if not self._scores:
            return [0.0] * len(self._tp)
        scores = np.concatenate(self._scores)
        return [
            _average_precision(scores, np.concatenate(tp) if tp else np.zeros(0, bool), self.num_gt)
            for tp in self._tp
        ]


def _greedy_match(cost_desc: np.ndarray, order: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """score 내림차순 그리디 매칭. `cost_desc`는 (pred, gt) '적합도'(클수록 좋음)."""
    n_pred, n_gt = cost_desc.shape
    tp = np.zeros(n_pred, dtype=bool)
    used = np.zeros(n_gt, dtype=bool)
    for i in order:
        row = np.where(used, -np.inf, cost_desc[i])
        if row.size == 0:
            continue
        j = int(np.argmax(row))
        if np.isfinite(row[j]) and keep[i, j] and row[j] > -np.inf:
            used[j] = True
            tp[i] = True
    return tp


class BoxDetectionMetrics:
    """COCO 스타일 단일 클래스(person) bbox mAP.

    `SegmentationMetrics`와 동일하게 `update()` 누적 → `compute()` 방식이다.
    클래스가 하나뿐이라 클래스 평균 단계는 없고, AP는 **101-point 보간**으로 낸다.
    """

    def __init__(self, iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS) -> None:
        self.iou_thresholds = tuple(float(t) for t in iou_thresholds)
        if not self.iou_thresholds:
            raise ValueError("iou_thresholds가 비어 있다")
        self._acc = _APAccumulator(len(self.iou_thresholds))

    def reset(self) -> None:
        self._acc.reset()

    def update(
        self,
        preds: torch.Tensor | np.ndarray,
        gts: torch.Tensor | np.ndarray,
    ) -> None:
        """한 이미지 분의 예측/GT를 누적.

        Args:
            preds: `(N, 5)` = `(x, y, w, h, score)` 중심 좌표 형식.
            gts: `(M, 4)` = `(x, y, w, h)` (뒤 컬럼은 무시).
        """
        p = _to_array(preds)
        g = _to_array(gts)
        p = p.reshape(0, 5) if p.size == 0 else p
        g = g.reshape(0, 4) if g.size == 0 else g

        n_gt = int(g.shape[0])
        if p.shape[0] == 0:
            self._acc.add(np.zeros(0), [np.zeros(0, bool)] * len(self.iou_thresholds), n_gt)
            return

        scores = p[:, 4].astype(np.float64)
        order = np.argsort(-scores, kind="stable")
        iou = box_iou_matrix(p[:, :4], g[:, :4])

        tps = [
            _greedy_match(iou, order, iou >= thr) if n_gt else np.zeros(p.shape[0], bool)
            for thr in self.iou_thresholds
        ]
        self._acc.add(scores, tps, n_gt)

    def compute(self) -> dict[str, float]:
        aps = self._acc.ap_per_threshold()
        out: dict[str, float] = {"map_50_95": float(np.mean(aps)) if aps else 0.0}
        for thr, ap in zip(self.iou_thresholds, aps, strict=True):
            if abs(thr - 0.5) < 1e-9:
                out["map_50"] = ap
            elif abs(thr - 0.75) < 1e-9:
                out["map_75"] = ap
        out.setdefault("map_50", aps[0] if aps else 0.0)
        return out


class PointAveragePrecision:
    """거리 기반 매칭 AP — `PointDetectionMetrics`의 임계값 비의존 버전.

    매칭 규칙은 `PointDetectionMetrics`와 같지만(기본 8px 그리디), 단일 score
    임계값에서 P/R을 재는 대신 **score 곡선 전체**에 대해 101-point AP를 낸다.
    파이프라인의 최종 산출물이 점이므로 이쪽이 실제 목표에 맞는 지표다.
    """

    def __init__(self, distance_threshold: float = 8.0) -> None:
        self.distance_threshold = float(distance_threshold)
        self._acc = _APAccumulator(1)

    def reset(self) -> None:
        self._acc.reset()

    def update(
        self,
        preds: torch.Tensor | np.ndarray,
        gts: torch.Tensor | np.ndarray,
    ) -> None:
        """`preds`는 `(x, y, score)`, `gts`는 `(x, y)` (뒤 컬럼 무시)."""
        p = _to_xy_score(preds)
        g = _to_xy(gts)

        n_gt = int(g.shape[0])
        if p.shape[0] == 0:
            self._acc.add(np.zeros(0), [np.zeros(0, bool)], n_gt)
            return

        scores = p[:, 2].astype(np.float64)
        order = np.argsort(-scores, kind="stable")
        if n_gt == 0:
            self._acc.add(scores, [np.zeros(p.shape[0], bool)], 0)
            return

        dist = np.hypot(
            p[:, 0][:, None] - g[None, :, 0], p[:, 1][:, None] - g[None, :, 1]
        )
        tp = _greedy_match(-dist, order, dist <= self.distance_threshold)
        self._acc.add(scores, [tp], n_gt)

    def compute(self) -> dict[str, float]:
        return {"point_ap": self._acc.ap_per_threshold()[0]}


# --------------------------------------------------------------------------
# Trainer 연동
# --------------------------------------------------------------------------


def build_compute_metrics(
    num_classes: int,
    ignore_index: int = 255,
    distance_threshold: float = 8.0,
    score_threshold: float = 0.3,
    map_iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
) -> Callable[[object], dict[str, float]]:
    """`Trainer(compute_metrics=...)` 로 넘길 함수를 만든다.

    `SkyLensTrainer.prediction_step`이 내놓는 규약을 소비한다:

    - `predictions = (seg_pred (B,H,W) int, detections (B,K,5: x,y,w,h,score))`
    - `label_ids   = (seg_labels (B,H,W) int, gt_boxes (B,K,5: x,y,w,h,valid))`

    GT 쪽은 구버전 `(x, y, valid)` 3-컬럼 형태도 읽는다 (그 경우 박스 지표는
    건너뛰고 점 지표만 낸다).

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
            det_a = np.asarray(det, dtype=np.float64)
            gt_a = np.asarray(pt_gt, dtype=np.float64)
            # GT 컬럼 수로 형태를 판별한다: 5 = (x,y,w,h,valid), 3 = (x,y,valid).
            has_gt_boxes = gt_a.ndim == 3 and gt_a.shape[-1] >= 5
            valid_col = 4 if has_gt_boxes else 2

            pdm = PointDetectionMetrics(distance_threshold=distance_threshold)
            pap = PointAveragePrecision(distance_threshold=distance_threshold)
            bdm = BoxDetectionMetrics(map_iou_thresholds) if has_gt_boxes else None

            for b in range(det_a.shape[0]):
                d_all = det_a[b][det_a[b][:, 4] > 0]
                g_all = gt_a[b][gt_a[b][:, valid_col] > 0]
                d = d_all[d_all[:, 4] >= score_threshold][:, [0, 1, 4]]
                g = g_all[:, :2]
                if d.size or g.size:
                    pdm.update(d, g)
                if d_all.size or g_all.size:
                    pap.update(d_all[:, [0, 1, 4]], g)
                    if bdm is not None:
                        bdm.update(d_all, g_all[:, :4])

            results.update(pdm.compute())
            results.update(pap.compute())
            if bdm is not None:
                results.update(bdm.compute())

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
