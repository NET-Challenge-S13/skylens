"""VisDrone val person metrics, squashed-512 vs native-scale tiled.

Three views of the same checkpoint on the 548 VisDrone val images:

A. `squash512`  — the training eval path (512x512 squash, collator GT with the
   stride-4 quantisation and the k=100 cap). Sanity check: should match the
   per-source VisDrone number of the training eval.
B. `tiled@512`  — tiled inference, merged detections rescaled into the squashed
   512 frame and scored against exactly the GT of A (top-k by score). Same
   yardstick as A; only the inference changes.
C. `tiled@765`  — tiled inference scored at the tiling scale (short side 765)
   against the full, uncapped, unquantised GT. The "native" number.

Works on any checkpoint, tiled-trained or not.

    PYTHONPATH=src .venv/bin/python scripts/eval_visdrone_tiled.py \
        --ckpt runs/<name>/final --data-root /path/to/data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from skylens_model.datasets import VisDronePerson
from skylens_model.models import SkyLensForDisasterPerception
from skylens_model.utils.cache import ResizedCache, resize_sample
from skylens_model.utils.collate import SkyLensCollator
from skylens_model.utils.metrics import (
    BoxDetectionMetrics,
    PointAveragePrecision,
    PointDetectionMetrics,
    build_compute_metrics,
    decode_gt_boxes,
    decode_heatmap_peaks,
)
from skylens_model.utils.tiling import (
    SHORT_SIDE,
    TILE_SIZE,
    merge_tile_detections,
    scale_to_short_side,
    tile_detections_to_full,
    tile_grid,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--short-side", type=int, default=SHORT_SIDE)
    p.add_argument("--tile", type=int, default=TILE_SIZE)
    p.add_argument("--min-overlap", type=int, default=64)
    p.add_argument("--score-threshold", type=float, default=0.3, help="training eval uses 0.3")
    p.add_argument("--k", type=int, default=100, help="max detections per forward (training eval: 100)")
    p.add_argument("--point-distances", type=float, nargs="+", default=[8.0, 16.0])
    p.add_argument("--merge-iou", type=float, default=0.5)
    p.add_argument("--merge-center-dist", type=float, default=2.0)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--out", type=Path, default=None, help="write metrics JSON here")
    return p.parse_args()


@torch.no_grad()
def forward_decode(model, images: list[np.ndarray], stride: int, k: int, thr: float, device) -> list[np.ndarray]:
    """images: HxWx3 float [0,1]. Returns per-image (N,5) detections with score > 0."""
    px = np.stack([np.concatenate([im, np.zeros(im.shape[:2] + (1,), np.float32)], 2).transpose(2, 0, 1)
                   for im in images])
    pv = torch.from_numpy(px).to(device)
    mm = torch.tensor([[True, False]] * len(images), device=device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        out = model(pixel_values=pv, modality_mask=mm)
    off = out.person_offset.float() if getattr(out, "person_offset", None) is not None else None
    det = decode_heatmap_peaks(out.person_heatmap.float(), out.person_wh.float(), k=k, threshold=thr,
                               stride=stride, offset=off).cpu().numpy().astype(np.float64)
    return [d[d[:, 4] > 0] for d in det]


def score(dets: list[np.ndarray], gts: list[np.ndarray], distances, thr: float) -> dict:
    """Mirror of build_compute_metrics' person branch for variable-length lists."""
    res: dict[str, float] = {}
    bdm = BoxDetectionMetrics()
    for d_all, g in zip(dets, gts, strict=True):
        if d_all.size or g.size:
            bdm.update(d_all.reshape(-1, 5), g.reshape(-1, 4))
    res.update(bdm.compute())
    for dist in distances:
        pdm, pap = PointDetectionMetrics(distance_threshold=dist), PointAveragePrecision(dist)
        for d_all, g in zip(dets, gts, strict=True):
            d = d_all[d_all[:, 4] >= thr][:, [0, 1, 4]]
            if d.size or g.size:
                pdm.update(d, g[:, :2])
            if d_all.size or g.size:
                pap.update(d_all[:, [0, 1, 4]], g[:, :2])
        m = {**pdm.compute(), **pap.compute()}
        suffix = "" if dist == distances[0] else f"@{dist:g}px"
        for key in ("point_precision", "point_recall", "point_f1", "point_ap"):
            res[key + suffix] = m[key]
    res["num_gt"] = float(sum(len(g) for g in gts))
    res["num_det"] = float(sum(len(d) for d in dets))
    return res


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkyLensForDisasterPerception.from_pretrained(str(args.ckpt), use_pretrained_backbone=False)
    model.to(device).eval()
    stride = int(model.config.person_head_stride)
    thr, k, bs = args.score_threshold, args.k, args.batch_size
    print(f"ckpt {args.ckpt} | stride {stride} | device {device}", flush=True)

    raw = VisDronePerson(args.data_root / "visdrone", split="val")
    n = len(raw) if args.max_images is None else min(len(raw), args.max_images)
    cache_512 = args.data_root / "_cache" / f"VisDronePerson_val_{TILE_SIZE}"
    cached = None
    if (cache_512 / "index.json").is_file():
        cached = ResizedCache(cache_512)  # read-only
        if len(cached) != len(raw):
            cached = None
    print(f"val images {n} | 512 source: {'cache ' + str(cache_512) if cached else 'in-memory resize'}")

    collator = SkyLensCollator(person_head_stride=stride)
    t0 = time.time()

    # ---- A: squash512 (training-eval path) ----
    det_a, gt_a, sizes = [], [], []
    for start in range(0, n, bs):
        idx = list(range(start, min(start + bs, n)))
        samples = [cached[i] if cached else resize_sample(raw[i], TILE_SIZE) for i in idx]
        batch = collator(samples)
        imgs = [np.asarray(s["image"], np.float32)[:, :, :3] for s in samples]
        if imgs[0].max() > 1.5:
            imgs = [im / 255.0 for im in imgs]
        det_a += forward_decode(model, imgs, stride, k, thr, device)
        g = decode_gt_boxes(batch["person_reg_mask"], batch["person_wh"], k=k, stride=stride,
                            offset=batch.get("person_offset")).numpy()
        gt_a += [x[x[:, 4] > 0][:, :4].astype(np.float64) for x in g]
    print(f"A done {time.time() - t0:.0f}s", flush=True)

    # sanity: the repo's compute_metrics on A (padded) must agree with our scorer
    pad = lambda arrs, c: np.stack([np.pad(a[:k], ((0, k - len(a[:k])), (0, 0)))[:, :c] if len(a) else np.zeros((k, c))  # noqa: E731
                                    for a in arrs])
    gt_pad = pad([np.concatenate([g, np.ones((len(g), 1))], 1) for g in gt_a], 5)
    ref = build_compute_metrics(num_classes=4, distance_threshold=args.point_distances[0],
                                score_threshold=thr)(((None, pad(det_a, 5)), (None, gt_pad)))

    # ---- tiled inference ----
    det_full, gt_full = [], []
    for i in range(n):
        s = raw[i]
        img = np.asarray(s["image"])
        img = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
        oh, ow = img.shape[:2]
        simg, boxes, _ = scale_to_short_side(img[:, :, :3], s["person_boxes"], args.short_side)
        H, W = simg.shape[:2]
        grid = tile_grid(W, H, args.tile, args.min_overlap)
        per_tile = []
        for gs in range(0, len(grid), bs):
            chunk = grid[gs:gs + bs]
            tiles = [np.ascontiguousarray(simg[y:y + args.tile, x:x + args.tile]) for x, y in chunk]
            for (x, y), d in zip(chunk, forward_decode(model, tiles, stride, k, thr, device), strict=True):
                per_tile.append(tile_detections_to_full(d, x, y))
        det_full.append(merge_tile_detections(per_tile, args.merge_iou, args.merge_center_dist))
        b = np.asarray(boxes, np.float64).reshape(-1, 4)
        gt_full.append(np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                                 b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1))
        sizes.append((W, H))
        if (i + 1) % 100 == 0:
            print(f"  tiled {i + 1}/{n} {time.time() - t0:.0f}s", flush=True)

    # ---- B: tiled detections in the squashed 512 frame, GT of A ----
    det_b = []
    for d, (W, H) in zip(det_full, sizes, strict=True):
        d = d.copy()
        d[:, [0, 2]] *= TILE_SIZE / W
        d[:, [1, 3]] *= TILE_SIZE / H
        det_b.append(d[np.argsort(-d[:, 4], kind="stable")[:k]])

    results = {
        "squash512": score(det_a, gt_a, args.point_distances, thr),
        "tiled@512": score(det_b, gt_a, args.point_distances, thr),
        "tiled@765": score(det_full, gt_full, args.point_distances, thr),
    }
    results["squash512_repo_compute_metrics"] = {kk: float(v) for kk, v in ref.items()}
    results["meta"] = {"ckpt": str(args.ckpt), "images": n, "short_side": args.short_side,
                       "tile": args.tile, "min_overlap": args.min_overlap, "k": k,
                       "score_threshold": thr, "merge_iou": args.merge_iou,
                       "merge_center_dist": args.merge_center_dist,
                       "gt_capped_by_k_A": int(sum(len(g) >= k for g in gt_a)),
                       "seconds": round(time.time() - t0, 1)}

    keys = ["map_50", "map_50_95", "point_f1", "point_ap", "point_precision", "point_recall"]
    keys += [f"point_f1@{d:g}px" for d in args.point_distances[1:]]
    keys += [f"point_ap@{d:g}px" for d in args.point_distances[1:]] + ["num_gt", "num_det"]
    print(f"\n{'metric':22s}" + "".join(f"{c:>14s}" for c in ("squash512", "tiled@512", "tiled@765")))
    for key in keys:
        print(f"{key:22s}" + "".join(f"{results[c][key]:14.4f}" for c in ("squash512", "tiled@512", "tiled@765")))
    print("\nrepo compute_metrics on A:", {kk: round(v, 4) for kk, v in ref.items()})
    print("meta:", results["meta"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
