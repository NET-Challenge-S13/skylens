"""VisDrone 타일링: 박스 자르기 · 타일 좌표 복원 · 중복 병합."""

from __future__ import annotations

import numpy as np

from skylens_model.utils.tiling import (
    clip_boxes_to_tile,
    merge_tile_detections,
    scale_to_short_side,
    tile_detections_to_full,
    tile_grid,
    tile_starts,
)


def test_tile_starts_cover_with_overlap() -> None:
    assert tile_starts(1360, 512, 64) == [0, 424, 848]
    assert tile_starts(765, 512, 64) == [0, 253]
    assert tile_starts(400, 512, 64) == [0]
    for length in (513, 765, 1020, 1360, 2000):
        s = tile_starts(length, 512, 64)
        assert s[0] == 0 and s[-1] + 512 == length
        assert all(b - a <= 512 - 64 for a, b in zip(s, s[1:], strict=False))
    assert len(tile_grid(1360, 765)) == 6


def test_scale_to_short_side() -> None:
    img = np.zeros((540, 960, 3), np.uint8)
    out, boxes, s = scale_to_short_side(img, np.array([[96, 54, 192, 108]], np.float32), 765)
    assert out.shape[:2] == (765, 1360)
    assert abs(s - 765 / 540) < 1e-9
    np.testing.assert_allclose(boxes[0], [136, 76.5, 272, 153], atol=0.2)


def test_clip_boxes_to_tile() -> None:
    boxes = np.array(
        [
            [110, 110, 120, 130],  # 완전히 안: 이동만
            [95, 100, 115, 110],   # 75% 보임: 잘려서 남음
            [90, 100, 106, 110],   # 37.5% 보임: 버림
            [611, 200, 640, 220],  # 오른쪽 경계에서 1px 폭만 남고 가시율도 낮음: 버림
            [300, 300, 301.5, 320],  # 원래 폭 1.5px: 버림
        ],
        np.float32,
    )
    out = clip_boxes_to_tile(boxes, 100, 100, 512, 512, min_visible=0.5, min_size=2.0)
    np.testing.assert_allclose(out, [[10, 10, 20, 30], [0, 0, 15, 10]])


def test_clip_boxes_small_sliver_kept_if_mostly_visible() -> None:
    # 폭 3px 박스가 경계에서 2px 남으면 가시율 66%, 폭 2px → 남는다.
    out = clip_boxes_to_tile(np.array([[99, 0, 102, 10]], np.float32), 100, 0)
    np.testing.assert_allclose(out, [[0, 0, 2, 10]])


def test_detections_map_back_and_merge_across_tiles() -> None:
    # 같은 사람(전체 좌표 500,300)이 두 타일의 겹침 영역에서 검출된다.
    tile_a = np.array([[500.0, 300.0, 8, 20, 0.9], [100.0, 100.0, 8, 20, 0.5]])
    tile_b = np.array([[76.5, 300.0, 8, 20, 0.7]])  # x0=424 → 전체 x=500.5
    full_a = tile_detections_to_full(tile_a, 0, 0)
    full_b = tile_detections_to_full(tile_b, 424, 0)
    np.testing.assert_allclose(full_b[0, :2], [500.5, 300.0])

    merged = merge_tile_detections([full_a, full_b], iou_threshold=0.5, center_distance=2.0)
    assert merged.shape == (2, 5)
    np.testing.assert_allclose(merged[:, 4], [0.9, 0.5])  # 높은 score 가 남는다
    np.testing.assert_allclose(merged[0, :2], [500.0, 300.0])


def test_merge_keeps_neighbours_within_one_tile() -> None:
    # 같은 타일 안의 붙어 있는 두 사람은 병합하지 않는다.
    same = np.array([[50.0, 50.0, 8, 20, 0.9], [51.0, 50.0, 8, 20, 0.8]])
    assert len(merge_tile_detections([same])) == 2
    assert len(merge_tile_detections([])) == 0
