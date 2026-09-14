"""Deployment-style pooled person evaluation.

Pools two sources into one PR curve, each at its deployment inference path:

- LLVIP_test (every-Nth subset of the 1,166-image training eval subset, 866 images):
  full frame squashed to 512, scored against the continuous cache annotations
  (collator filter: w,h>0, centre inside the frame, first 512 boxes).
- VisDrone val: native scale (short side 765), 512 tiles with >=64px overlap,
  detections merged across tiles, scored against uncapped continuous GT.
  `visdrone137` = the images of the training eval subset; `visdrone548` = all.
- SARD val (394 images, 1920x1080): tiled exactly like VisDrone (`sard394_tiled765`).
  It is never part of the training eval subset. `pooled_with_sard` pools
  LLVIP-866 + VisDrone-137 + SARD-394; `pooled137_vd8` stays LLVIP + VisDrone-137.
- Every set also gets `recall_at_p50` (max recall with precision >= 0.5 over a
  0.05..0.95 step-0.01 point sweep) and `recall_max` (recall at 0.05).

Protocol (see src/skylens_model/utils/metrics.py):
- decode centres at (x + 0.5) * stride (or x + person_offset);
- detections kept at score >= --ap-floor (0.05), top-k 100 per image/tile, for
  box mAP and point_ap; point P/R/F1 are reported over an F1 threshold sweep.
- --calib: pick the F1 threshold on a held-out pool (LLVIP_test images not in the
  eval subset, same count, + the 411 VisDrone val images not in visdrone137) and
  apply it to the eval pool ("fair" block).
- --legacy: old protocol (top-left centres, threshold 0.3, grid-decoded LLVIP GT).

    PYTHONPATH=src .venv/bin/python scripts/eval_deploy.py \
        --ckpt runs/<name>/final --data-root /path/to/data --bs 4 [--calib] [--out res.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from skylens_model.datasets import SARD, VisDronePerson
from skylens_model.models import SkyLensForDisasterPerception
from skylens_model.utils import ResizedCache, SkyLensCollator
from skylens_model.utils import tiling as T
from skylens_model.utils.metrics import (
    BoxDetectionMetrics,
    PointAveragePrecision,
    PointDetectionMetrics,
    decode_gt_boxes,
    decode_heatmap_peaks,
    point_match_flags,
    recall_sweep,
)

SIZE, K, SHORT, OVERLAP = 512, 100, 765, 64
# The training eval subset (v3). SARD_val is deliberately absent: a SARD_val_512
# cache (e.g. from a --sard squash run) would otherwise change the stride.
CACHE_NAMES = ["LLVIP_test", "VisDronePerson_val", "RescueNetSegmentation_test", "FireSegmentation_val"]
FINE_SWEEP = [round(t / 100, 2) for t in range(5, 96)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--out", default=None)
    p.add_argument("--ap-floor", type=float, default=0.05)
    p.add_argument("--dist", type=float, default=8.0, help="point match distance (px)")
    p.add_argument("--legacy", action="store_true")
    p.add_argument("--calib", action="store_true")
    p.add_argument("--llvip-modality", choices=["both", "rgb", "thermal"], default="both",
                   help="force the LLVIP input modality (rgb: thermal removed, thermal: RGB removed)")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    legacy = a.legacy
    dec_thr = 0.3 if legacy else a.ap_floor
    sweep = [0.3] if legacy else [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    # --- eval subset: identical to the training eval (every step-th of the concat) ---
    cache = Path(a.data_root) / "_cache"
    parts, src, local = [], [], []
    for n in CACHE_NAMES:
        d = cache / f"{n}_{SIZE}"
        if not (d / "index.json").is_file():
            continue
        ds = ResizedCache(d)
        parts.append(ds)
        src += [n] * len(ds)
        local += list(range(len(ds)))
    if not parts or src[0] != "LLVIP_test":
        raise SystemExit("LLVIP_test_512 cache missing")
    full = ConcatDataset(parts)
    step = len(full) // 1200 + 1
    idx = list(range(0, len(full), step))
    llvip_local = [local[i] for i in idx if src[i] == "LLVIP_test"]  # LLVIP is parts[0]: global == local
    vd137 = [local[i] for i in idx if src[i] == "VisDronePerson_val"]
    llvip_ds = parts[0]
    print(f"subset {len(idx)} | LLVIP {len(llvip_local)} | VisDrone {len(vd137)} | legacy={legacy}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkyLensForDisasterPerception.from_pretrained(a.ckpt, use_pretrained_backbone=False).to(dev).eval()
    stride = int(model.config.person_head_stride)
    amp = dict(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda")
    t0 = time.time()
    centre_mode = {"v": None}

    def decode(out) -> list[np.ndarray]:
        off = None if legacy else getattr(out, "person_offset", None)
        centre_mode["v"] = "legacy_topleft" if legacy else ("offset" if off is not None else "cell_centre")
        det = decode_heatmap_peaks(out.person_heatmap.float(), out.person_wh.float(), k=K, threshold=dec_thr,
                                   stride=stride, offset=off, legacy_decode=legacy)
        return [d[d[:, 4] > 0] for d in det.cpu().numpy().astype(np.float64)]

    def continuous_gt(li: int, max_objects: int = 512) -> np.ndarray:
        r = llvip_ds.records[li]
        if not r["has_boxes"] or not r["boxes"]:
            return np.zeros((0, 4))
        b = np.asarray(r["boxes"], np.float64).reshape(-1, 4)[:max_objects]
        w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        g = SIZE // stride
        ok = (w > 0) & (h > 0) & (np.floor(cx / stride) >= 0) & (np.floor(cx / stride) < g) \
            & (np.floor(cy / stride) >= 0) & (np.floor(cy / stride) < g)
        return np.stack([cx, cy, w, h], 1)[ok]

    coll = SkyLensCollator(person_head_stride=stride, validity_channel=False, modality_dropout=(0.0, 0.0))

    def run_llvip(ids: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        dl = DataLoader(Subset(full, ids), batch_size=a.bs, shuffle=False, collate_fn=coll, num_workers=2)
        det_l, gt_l = [], []
        with torch.no_grad(), torch.autocast(**amp):
            for batch in dl:
                if a.llvip_modality == "rgb":  # remove thermal: channel 3 -> 0.0, mask False
                    batch["pixel_values"][:, 3] = 0.0
                    batch["modality_mask"][:, 1] = False
                elif a.llvip_modality == "thermal":  # remove RGB
                    batch["pixel_values"][:, :3] = 0.0
                    batch["modality_mask"][:, 0] = False
                batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                det_l += decode(model(**batch))
                if legacy:
                    g = decode_gt_boxes(batch["person_reg_mask"], batch["person_wh"], k=4096, stride=stride,
                                        legacy_decode=True)
                    gt_l += [x[x[:, 4] > 0][:, :4] for x in g.cpu().numpy().astype(np.float64)]
        if not legacy:
            gt_l = [continuous_gt(li) for li in ids]
        return det_l, gt_l

    det_l, gt_l = run_llvip(llvip_local)
    print(f"LLVIP done {time.time() - t0:.0f}s", flush=True)

    calib_l = None
    if a.calib and not legacy:
        used = set(llvip_local)
        rest = [i for i in range(len(llvip_ds)) if i not in used]
        hold = rest[:: max(1, len(rest) // len(llvip_local))][: len(llvip_local)]
        calib_l = run_llvip(hold)
        print(f"LLVIP holdout ({len(hold)}) done {time.time() - t0:.0f}s", flush=True)

    @torch.no_grad()
    def forward_tiles(images: list[np.ndarray]) -> list[np.ndarray]:
        px = np.stack([np.concatenate([im, np.zeros(im.shape[:2] + (1,), np.float32)], 2).transpose(2, 0, 1)
                       for im in images])
        mm = torch.tensor([[True, False]] * len(images), device=dev)
        with torch.autocast(**amp):
            return decode(model(pixel_values=torch.from_numpy(px).to(dev), modality_mask=mm))

    def run_tiled(raw, name: str) -> tuple[dict, dict]:
        det_v, gt_v = {}, {}
        for i in range(len(raw)):
            det_v[i], gt_v[i] = tile_one(raw[i])
            if (i + 1) % 100 == 0:
                print(f"  {name} tiled {i + 1}/{len(raw)} {time.time() - t0:.0f}s", flush=True)
        return det_v, gt_v

    def tile_one(s):
        img = np.asarray(s["image"])
        img = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
        simg, boxes, _ = T.scale_to_short_side(img[:, :, :3], s["person_boxes"], SHORT)
        H, W = simg.shape[:2]
        grid = T.tile_grid(W, H, SIZE, OVERLAP)
        per_tile = []
        for gs in range(0, len(grid), a.bs):
            chunk = grid[gs: gs + a.bs]
            tiles = [np.ascontiguousarray(simg[y: y + SIZE, x: x + SIZE]) for x, y in chunk]
            per_tile += [T.tile_detections_to_full(d, x, y) for (x, y), d in zip(chunk, forward_tiles(tiles))]
        b = np.asarray(boxes, np.float64).reshape(-1, 4)
        gt = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
        return T.merge_tile_detections(per_tile, 0.5, 2.0), gt

    raw = VisDronePerson(Path(a.data_root) / "visdrone", split="val")
    det_v, gt_v = run_tiled(raw, "VisDrone")
    sard_raw = SARD(Path(a.data_root) / "sard", split="val")
    det_s, gt_s = run_tiled(sard_raw, "SARD")

    def score(items) -> dict:
        bdm, pap = BoxDetectionMetrics(), PointAveragePrecision(a.dist)
        all_s, all_tp, n_gt = [], [], 0
        pdm = {t: PointDetectionMetrics(distance_threshold=a.dist) for t in sweep}
        for d, g, sc in items:
            d, g = d.reshape(-1, 5), g.reshape(-1, 4)
            if d.size or g.size:
                bdm.update(d, g)
                dp = d[:, [0, 1, 4]].copy()
                dp[:, :2] *= sc
                gp = g[:, :2] * sc
                for t in sweep:
                    pdm[t].update(dp[dp[:, 2] >= t], gp)
                pap.update(dp, gp)
                all_s.append(dp[:, 2])
                all_tp.append(point_match_flags(dp, gp, a.dist))
                n_gt += len(gp)
        r = {k: float(v) for k, v in {**bdm.compute(), **pap.compute()}.items()
             if k in ("map_50", "map_50_95", "point_ap")}
        r["sweep"] = {f"{t:.2f}": {k: float(v) for k, v in pdm[t].compute().items()
                                   if k in ("point_precision", "point_recall", "point_f1")} for t in sweep}
        bt = max(sweep, key=lambda t: r["sweep"][f"{t:.2f}"]["point_f1"])
        r["best_f1"], r["best_thr"] = r["sweep"][f"{bt:.2f}"]["point_f1"], bt
        r["f1_at_0.30"] = r["sweep"]["0.30"]["point_f1"]
        rs = recall_sweep(np.concatenate(all_s) if all_s else np.zeros(0),
                          np.concatenate(all_tp) if all_tp else np.zeros(0, bool), n_gt,
                          [t for t in FINE_SWEEP if t >= dec_thr - 1e-9] or [dec_thr])
        r.update(rs)
        r["num_images"] = len(items)
        r["num_gt"] = int(sum(len(g) for _, g, _ in items))
        r["num_det"] = int(sum(len(d) for d, _, _ in items))
        return r

    L = [(d, g, 1.0) for d, g in zip(det_l, gt_l)]

    def V(ids, sc=1.0):
        return [(det_v[i], gt_v[i], sc) for i in ids]

    def S(ids):
        return [(det_s[i], gt_s[i], 1.0) for i in ids]

    all_vd = range(len(raw))
    all_sard = range(len(sard_raw))
    res = {
        "llvip_866": score(L),
        "visdrone137_tiled765_8px": score(V(vd137)),
        "visdrone548_tiled765_8px": score(V(all_vd)),
        "pooled137_vd8": score(L + V(vd137)),
        "pooled137_vd12": score(L + V(vd137, 8 / 12)),  # VisDrone at a 12px match radius
        "pooled548_vd8": score(L + V(all_vd)),
        "sard394_tiled765": score(S(all_sard)),
        "pooled_with_sard": score(L + V(vd137) + S(all_sard)),
    }
    if calib_l is not None:
        vd411 = [i for i in all_vd if i not in set(vd137)]
        C = [(d, g, 1.0) for d, g in zip(*calib_l)]
        res["calib_llvipHoldout_vd411"] = score(C + V(vd411))
        res["calib_llvipHoldout"] = score(C)
        res["calib_vd411"] = score(V(vd411))
        ct = res["calib_llvipHoldout_vd411"]["best_thr"]
        res["fair"] = {
            "thr_from_calib": ct,
            "pooled137_vd8_f1": res["pooled137_vd8"]["sweep"][f"{ct:.2f}"]["point_f1"],
            "llvip_866_f1": res["llvip_866"]["sweep"][f"{ct:.2f}"]["point_f1"],
            "visdrone137_f1": res["visdrone137_tiled765_8px"]["sweep"][f"{ct:.2f}"]["point_f1"],
        }
    res["meta"] = {"ckpt": a.ckpt, "legacy": legacy, "decode_thr": dec_thr, "topk": K,
                   "centre_mode": centre_mode["v"],
                   "llvip_gt": "grid_decoded" if legacy else "continuous_cache_annotations",
                   "vd137_indices": vd137, "seconds": round(time.time() - t0, 1)}
    if a.llvip_modality != "both":  # absent by default so existing outputs keep their keys
        res["meta"]["llvip_modality"] = a.llvip_modality

    print(f"\n{'set':28s}{'mAP50':>8s}{'mAP5095':>9s}{'ptAP':>8s}{'bestF1':>8s}{'@thr':>6s}{'F1@.3':>8s}{'nGT':>7s}{'nDet':>8s}{'R@P50':>8s}{'Rmax':>8s}")
    for k, v in res.items():
        if isinstance(v, dict) and "map_50" in v:
            print(f"{k:28s}{v['map_50']:8.4f}{v['map_50_95']:9.4f}{v['point_ap']:8.4f}{v['best_f1']:8.4f}"
                  f"{v['best_thr']:6.2f}{v['f1_at_0.30']:8.4f}{v['num_gt']:7d}{v['num_det']:8d}"
                  f"{v['recall_at_p50']:8.4f}{v['recall_max']:8.4f}")
    if "fair" in res:
        print("fair:", res["fair"])
    print("meta:", {k: v for k, v in res["meta"].items() if k != "vd137_indices"})
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
