"""Native-scale tiling for VisDrone.

The default pipeline squashes every image to 512x512 (`cache.resize_sample`).
For VisDrone that shrinks people ~2.7x in width, below what a stride-4 CenterNet
head can resolve. Here images are instead scaled so their short side is ~765px
(native for 1360x765) and the model sees 512x512 windows of that.

- Train: `build_scaled_cache` stores each image once at the target scale, and
  `ScaledTileCropDataset` cuts `crops_per_image` random 512 crops from it per epoch.
- Eval: `tile_starts` lays an overlapping grid, detections are mapped back with
  `tile_detections_to_full` and de-duplicated with `merge_tile_detections`.

Boxes are xyxy pixels (dataset contract); detections are `(x, y, w, h, score)`
centre format (`decode_heatmap_peaks` contract).
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

__all__ = [
    "SHORT_SIDE",
    "TILE_SIZE",
    "scale_to_short_side",
    "tile_starts",
    "tile_grid",
    "clip_boxes_to_tile",
    "tile_detections_to_full",
    "merge_tile_detections",
    "build_scaled_cache",
    "ScaledTileCropDataset",
]

SHORT_SIDE = 765
TILE_SIZE = 512
_JPEG_QUALITY = 95


def scale_to_short_side(
    image: np.ndarray, boxes: np.ndarray | None, short: int = SHORT_SIDE
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Scale keeping aspect ratio so min(h, w) == `short`. Returns (image, boxes, scale)."""
    h, w = image.shape[:2]
    s = short / float(min(h, w))
    nw, nh = int(round(w * s)), int(round(h * s))
    if (nw, nh) != (w, h):
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
        image = cv2.resize(image, (nw, nh), interpolation=interp)
    if boxes is not None:
        b = np.asarray(boxes, np.float32).reshape(-1, 4).copy()
        b[:, [0, 2]] *= nw / w
        b[:, [1, 3]] *= nh / h
        boxes = b
    return image, boxes, s


def tile_starts(length: int, tile: int = TILE_SIZE, min_overlap: int = 64) -> list[int]:
    """Evenly spaced window starts covering [0, length) with >= `min_overlap` overlap."""
    if length <= tile:
        return [0]
    n = math.ceil((length - min_overlap) / (tile - min_overlap))
    n = max(n, 2)
    return [int(round(i * (length - tile) / (n - 1))) for i in range(n)]


def tile_grid(w: int, h: int, tile: int = TILE_SIZE, min_overlap: int = 64) -> list[tuple[int, int]]:
    """(x0, y0) of every tile, row-major."""
    return [(x, y) for y in tile_starts(h, tile, min_overlap) for x in tile_starts(w, tile, min_overlap)]


def clip_boxes_to_tile(
    boxes: np.ndarray | None,
    x0: float,
    y0: float,
    tile_w: float = TILE_SIZE,
    tile_h: float = TILE_SIZE,
    min_visible: float = 0.5,
    min_size: float = 2.0,
) -> np.ndarray:
    """Clip xyxy boxes to a tile and express them in tile coordinates.

    A box is kept when at least `min_visible` of its area lies in the tile and the
    clipped box is at least `min_size` px wide and tall.
    """
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), np.float32)
    b = np.asarray(boxes, np.float32).reshape(-1, 4)
    area = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    c = b - np.array([x0, y0, x0, y0], np.float32)
    c[:, [0, 2]] = np.clip(c[:, [0, 2]], 0, tile_w)
    c[:, [1, 3]] = np.clip(c[:, [1, 3]], 0, tile_h)
    cw, ch = c[:, 2] - c[:, 0], c[:, 3] - c[:, 1]
    vis = np.where(area > 0, cw * ch / np.maximum(area, 1e-9), 0.0)
    keep = (vis >= min_visible) & (cw >= min_size) & (ch >= min_size)
    return c[keep].astype(np.float32)


def tile_detections_to_full(det: np.ndarray, x0: float, y0: float) -> np.ndarray:
    """Shift `(x, y, w, h, score)` detections from tile to full-image coordinates."""
    d = np.asarray(det, np.float64).reshape(-1, 5).copy()
    d[:, 0] += x0
    d[:, 1] += y0
    return d


def merge_tile_detections(
    dets: Sequence[np.ndarray],
    iou_threshold: float = 0.5,
    center_distance: float = 2.0,
) -> np.ndarray:
    """Greedy de-duplication across tiles, highest score wins.

    `dets[i]` are full-image detections from tile i. A detection is suppressed only
    by a higher-scoring one from a *different* tile (within a tile the heatmap
    max-pool already de-duplicates) when their box IoU > `iou_threshold` or their
    centres are within `center_distance` px.
    """
    from .metrics import box_iou_matrix

    parts = [np.asarray(d, np.float64).reshape(-1, 5) for d in dets]
    if not parts or sum(len(p) for p in parts) == 0:
        return np.zeros((0, 5), np.float64)
    all_d = np.concatenate(parts)
    tid = np.concatenate([np.full(len(p), i) for i, p in enumerate(parts)])
    order = np.argsort(-all_d[:, 4], kind="stable")
    all_d, tid = all_d[order], tid[order]

    iou = box_iou_matrix(all_d[:, :4], all_d[:, :4])
    dist = np.hypot(all_d[:, None, 0] - all_d[None, :, 0], all_d[:, None, 1] - all_d[None, :, 1])
    dup = ((iou > iou_threshold) | (dist <= center_distance)) & (tid[:, None] != tid[None, :])

    suppressed = np.zeros(len(all_d), bool)
    keep = []
    for i in range(len(all_d)):
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= dup[i]
    return all_d[keep]


# --------------------------------------------------------------------------
# Train cache: images scaled once to the short side, random crops at load time
# --------------------------------------------------------------------------

_POOL_DATASET = None
_POOL_ARGS: tuple = ()


def _write_one(i: int) -> dict:
    from .cache import _to_uint8

    cache_dir, short = _POOL_ARGS
    s = _POOL_DATASET[i]
    img = _to_uint8(np.asarray(s["image"]))
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    img, boxes, _ = scale_to_short_side(img[:, :, :3], s.get("person_boxes"), short)
    cv2.imwrite(str(Path(cache_dir) / f"{i:06d}.jpg"), np.ascontiguousarray(img[:, :, ::-1]),
                [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    return {
        "w": int(img.shape[1]),
        "h": int(img.shape[0]),
        "boxes": [] if boxes is None else boxes.tolist(),
        "has_boxes": boxes is not None,
    }


def build_scaled_cache(
    dataset: Dataset, cache_dir: str | Path, short: int = SHORT_SIDE, workers: int = 8
) -> Path:
    """Write every RGB sample of `dataset` scaled to `short` px short side.

    `index.json` is written last, so an interrupted build is simply redone.
    Never touches an existing, complete cache.
    """
    global _POOL_DATASET, _POOL_ARGS
    cache_dir = Path(cache_dir)
    index_path = cache_dir / "index.json"
    if index_path.is_file():
        meta = json.loads(index_path.read_text(encoding="utf-8"))
        if meta.get("short_side") == short and len(meta.get("samples", [])) == len(dataset):
            return cache_dir
        raise RuntimeError(f"{cache_dir} holds a different cache; refusing to overwrite it")

    cache_dir.mkdir(parents=True, exist_ok=True)
    _POOL_DATASET, _POOL_ARGS = dataset, (str(cache_dir), short)
    n = len(dataset)
    records: list[dict] = []
    if workers > 1:
        with get_context("fork").Pool(workers) as pool:
            for j, r in enumerate(pool.imap(_write_one, range(n), chunksize=8)):
                records.append(r)
                if (j + 1) % 500 == 0:
                    print(f"    {j + 1}/{n}", flush=True)
    else:
        records = [_write_one(i) for i in range(n)]
    _POOL_DATASET = None

    index_path.write_text(
        json.dumps({"short_side": short, "samples": records}), encoding="utf-8"
    )
    return cache_dir


class ScaledTileCropDataset(Dataset):
    """`crops_per_image` random `tile` x `tile` crops per cached image per epoch.

    Index `k` maps to image `k // crops_per_image`; the crop position is drawn
    fresh on every access (python `random`, which torch re-seeds per worker).
    """

    def __init__(
        self,
        cache_dir: str | Path,
        crops_per_image: int = 3,
        tile: int = TILE_SIZE,
        min_visible: float = 0.5,
        min_size: float = 2.0,
        transforms: Callable | None = None,
    ):
        self.root = Path(cache_dir)
        meta = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        self.records: list[dict] = meta["samples"]
        self.crops = int(crops_per_image)
        self.tile = int(tile)
        self.min_visible = min_visible
        self.min_size = min_size
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.records) * self.crops

    def __getitem__(self, k: int) -> dict:
        i = k // self.crops
        r = self.records[i]
        bgr = cv2.imread(str(self.root / f"{i:06d}.jpg"), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        t = self.tile
        x0 = random.randint(0, max(w - t, 0))
        y0 = random.randint(0, max(h - t, 0))
        crop = bgr[y0:y0 + t, x0:x0 + t, ::-1]
        if crop.shape[:2] != (t, t):  # 짧은 변이 tile 보다 작을 때만
            crop = cv2.copyMakeBorder(np.ascontiguousarray(crop), 0, t - crop.shape[0], 0,
                                      t - crop.shape[1], cv2.BORDER_CONSTANT, value=0)
        boxes = None
        if r["has_boxes"]:
            boxes = clip_boxes_to_tile(np.asarray(r["boxes"], np.float32).reshape(-1, 4),
                                       x0, y0, t, t, self.min_visible, self.min_size)
        sample = {
            "image": np.ascontiguousarray(crop).astype(np.float32) / 255.0,
            "has_rgb": True,
            "has_thermal": False,
            "danger_mask": None,
            "person_boxes": boxes,
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
