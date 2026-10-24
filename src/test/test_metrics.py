"""AP·IoU 지표의 손계산 검증."""

from __future__ import annotations

import numpy as np
import torch

from skylens_model.utils.metrics import (
    BoxDetectionMetrics,
    PointAveragePrecision,
    box_iou_matrix,
    build_compute_metrics,
    decode_gt_boxes,
)

# 예측 3개(score 0.9/0.8/0.7) 중 1·3번만 GT 2개와 맞는 구성의 101-point AP.
# PR 곡선: (r=0.5, p=1.0) → (0.5, 0.5) → (1.0, 2/3)
# envelope: r<=0.50 → 1.0 (51점), r>0.50 → 2/3 (50점)
_AP_MIXED = (51 * 1.0 + 50 * (2 / 3)) / 101

_PREDS = np.array(
    [[10, 10, 10, 10, 0.9], [200, 200, 10, 10, 0.8], [50, 50, 10, 10, 0.7]], np.float64
)
_GTS = np.array([[10, 10, 10, 10], [50, 50, 10, 10]], np.float64)


def test_box_iou_matrix() -> None:
    a = np.array([[0, 0, 10, 10]], np.float64)
    assert box_iou_matrix(a, a)[0, 0] == 1.0
    assert box_iou_matrix(a, np.array([[100, 100, 10, 10]], np.float64))[0, 0] == 0.0
    # x축으로 절반 어긋남: inter 50 / union 150
    assert box_iou_matrix(a, np.array([[5, 0, 10, 10]], np.float64))[0, 0] == 1 / 3
    # 너비 2배로 감싸는 박스: inter 100 / union 200
    assert box_iou_matrix(a, np.array([[0, 0, 20, 10]], np.float64))[0, 0] == 0.5


def test_box_map_matches_hand_calculation() -> None:
    m = BoxDetectionMetrics()
    m.update(_PREDS, _GTS)
    out = m.compute()
    assert out["map_50"] == out["map_50_95"] == out["map_75"]
    assert abs(out["map_50"] - _AP_MIXED) < 1e-12


def test_box_map_perfect_and_zero() -> None:
    perfect = BoxDetectionMetrics()
    perfect.update(_PREDS[[0, 2]], _GTS)
    assert perfect.compute()["map_50_95"] == 1.0

    miss = BoxDetectionMetrics()
    miss.update(np.array([[900, 900, 10, 10, 0.9]], np.float64), _GTS)
    assert miss.compute()["map_50_95"] == 0.0


def test_box_map_iou_thresholds_discriminate() -> None:
    # IoU = 2/3 → 0.50~0.65 네 임계값에서만 TP → mAP@0.5:0.95 = 0.4
    m = BoxDetectionMetrics()
    m.update(
        np.array([[2, 0, 10, 10, 0.9]], np.float64),
        np.array([[0, 0, 10, 10]], np.float64),
    )
    out = m.compute()
    assert out["map_50"] == 1.0
    assert out["map_75"] == 0.0
    assert abs(out["map_50_95"] - 0.4) < 1e-12


def test_point_average_precision() -> None:
    p = PointAveragePrecision(distance_threshold=8.0)
    p.update(_PREDS[:, [0, 1, 4]], _GTS[:, :2])
    assert abs(p.compute()["point_ap"] - _AP_MIXED) < 1e-12


def test_decode_gt_boxes() -> None:
    reg = torch.zeros(1, 1, 8, 8)
    wh = torch.zeros(1, 2, 8, 8)
    reg[0, 0, 3, 2] = 1.0
    wh[0, 0, 3, 2], wh[0, 1, 3, 2] = 5.0, 7.0

    boxes = decode_gt_boxes(reg, wh, k=4, stride=4)
    assert boxes.shape == (1, 4, 5)
    # 셀 중앙 규약: (x + 0.5) * stride
    assert boxes[0, 0].tolist() == [10.0, 14.0, 20.0, 28.0, 1.0]
    legacy = decode_gt_boxes(reg, wh, k=4, stride=4, legacy_decode=True)
    assert legacy[0, 0].tolist() == [8.0, 12.0, 20.0, 28.0, 1.0]
    assert boxes[0, 1:, 4].sum() == 0.0  # 빈 슬롯은 valid=0


def test_build_compute_metrics_accepts_both_gt_layouts() -> None:
    cm = build_compute_metrics(num_classes=4)
    seg = np.zeros((1, 4, 4), np.int16)
    det = _PREDS[None].astype(np.float32)

    gt_boxes = np.zeros((1, 3, 5), np.float32)
    gt_boxes[0, :2, :4] = _GTS
    gt_boxes[0, :2, 4] = 1.0
    with_boxes = cm(((seg, det), (seg, gt_boxes)))
    assert abs(with_boxes["map_50"] - _AP_MIXED) < 1e-9
    assert abs(with_boxes["point_ap"] - _AP_MIXED) < 1e-9

    # 구버전 (x, y, valid) 레이아웃: 박스 지표는 생략되고 점 지표만 나온다.
    gt_points = np.zeros((1, 3, 3), np.float32)
    gt_points[0, :2, :2] = _GTS[:, :2]
    gt_points[0, :2, 2] = 1.0
    legacy = cm(((seg, det), (seg, gt_points)))
    assert "map_50" not in legacy
    assert abs(legacy["point_ap"] - _AP_MIXED) < 1e-9
    assert legacy["point_f1"] == with_boxes["point_f1"]


def test_decode_heatmap_single_peak_at_cell_centre() -> None:
    from skylens_model.utils.metrics import decode_heatmap_peaks

    hm = torch.zeros(1, 1, 8, 8)
    hm[0, 0, 5, 3] = 0.9  # y=5, x=3
    det = decode_heatmap_peaks(hm, None, k=3, threshold=0.3, stride=4)
    assert det[0, 0, :2].tolist() == [14.0, 22.0]
    assert abs(float(det[0, 0, 4]) - 0.9) < 1e-6
    # 인코딩 int(c/stride)와 왕복: 셀 안의 어떤 연속 중심도 같은 셀로 가고, 오차는 <= stride/2
    for c in (12.0, 13.9, 15.99):
        assert int(c / 4) == 3 and abs(c - float(det[0, 0, 0])) <= 2.0
    legacy = decode_heatmap_peaks(hm, None, k=3, threshold=0.3, stride=4, legacy_decode=True)
    assert legacy[0, 0, :2].tolist() == [12.0, 20.0]


def test_decode_heatmap_offset_overrides_half_cell() -> None:
    from skylens_model.utils.metrics import decode_heatmap_peaks

    hm = torch.zeros(1, 1, 8, 8)
    hm[0, 0, 5, 3] = 0.9
    off = torch.zeros(1, 2, 8, 8)
    off[0, 0, 5, 3], off[0, 1, 5, 3] = 0.25, 0.75
    det = decode_heatmap_peaks(hm, None, k=1, threshold=0.3, stride=4, offset=off)
    assert det[0, 0, :2].tolist() == [13.0, 23.0]  # (3.25*4, 5.75*4), not +0.5


def test_ap_score_floor_does_not_truncate_pr_curve() -> None:
    # 5 GT; 높은 점수 검출 2개는 TP 1/FP 1, 낮은 점수(0.1~0.25) 검출 3개는 전부 TP.
    gts = np.array([[20.0 * i + 10, 10, 8, 8] for i in range(5)])
    preds = np.array(
        [
            [10, 10, 8, 8, 0.9],   # TP
            [300, 300, 8, 8, 0.8],  # FP
            [30, 10, 8, 8, 0.25],  # TP
            [50, 10, 8, 8, 0.2],   # TP
            [70, 10, 8, 8, 0.1],   # TP
        ],
        np.float32,
    )
    seg = np.zeros((1, 4, 4), np.int16)
    gt = np.zeros((1, 5, 5), np.float32)
    gt[0, :, :4] = gts
    gt[0, :, 4] = 1.0
    ev = ((seg, preds[None]), (seg, gt))
    lo = build_compute_metrics(num_classes=4, score_threshold=0.3, ap_score_floor=0.05)(ev)
    hi = build_compute_metrics(num_classes=4, score_threshold=0.3, ap_score_floor=0.3)(ev)
    assert lo["map_50"] >= hi["map_50"] and lo["point_ap"] >= hi["point_ap"]
    assert lo["map_50"] > hi["map_50"] + 0.2
    # 점 P/R/F1은 운영 임계값(0.3)에서 그대로
    assert lo["point_f1"] == hi["point_f1"]
    assert lo["point_tp"] == 1.0 and lo["point_fp"] == 1.0
