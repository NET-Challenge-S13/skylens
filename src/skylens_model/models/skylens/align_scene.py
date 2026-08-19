"""Rigidly align one export of a scene onto another export of the SAME scene.

The demo carries two exports of one capture: the level ladder
(``step*_light.ply``, the frame everything downstream uses — the manifest, the
segment cuts, the route placement) and a separately-converted final
(``step30000.compressed.ply``), which came through another tool and no longer
shares that frame. This estimates the rigid transform (level + yaw + scale +
translation) that puts the stray export back onto the reference, without point
correspondences:

1. level the stray cloud (its measured up -> +Y); the REFERENCE is required to
   be leveled already (run ``level_scene`` first);
2. yaw: the horizontal principal axes must coincide — of the two possible
   yaws (φ, φ+180°) keep the one whose along-axis DENSITY profile (splat count
   per slab) cross-correlates with the reference's. Density is the robust
   signature here: percentile spans and medians differ between exports of
   different step counts, but where the buildings stand does not;
3. scale + along-axis shift: joint grid search maximizing that same density
   correlation;
4. remaining translation (cross-axis, height): median difference over matched
   slabs.

Usage::

    uv run python -m skylens_model.models.skylens.align_scene \\
        C:/tmp/step30000_raw.ply res/static/demo/step07000_light.ply \\
        res/static/demo/step30000_final.ply --ref-scale 30.872

``--ref-scale`` is the reference's metres-per-unit (the manifest's) so the
output lands in metres like the segment assets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .level_scene import measure_up, quat_from_to, quat_to_matrix, rotate_ply
from .split_segments import Ply


def _core(centers: np.ndarray) -> np.ndarray:
    lo = np.percentile(centers, 1, axis=0)
    hi = np.percentile(centers, 99, axis=0)
    return centers[np.all((centers >= lo) & (centers <= hi), axis=1)]


def _principal_xz(core: np.ndarray) -> np.ndarray:
    """Horizontal principal direction of a leveled cloud, unit, in the XZ plane."""
    flat = core[:, [0, 2]] - core[:, [0, 2]].mean(axis=0)
    _, _, vh = np.linalg.svd(flat, full_matrices=False)
    d = vh[0]
    return d / np.linalg.norm(d)


def _density(along: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(along, bins)
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def _yaw_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w, x, y, z)."""
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stray", type=Path, help="export to bring into the reference frame")
    ap.add_argument("reference", type=Path, help="LEVELED export already in the wanted frame")
    ap.add_argument("out", type=Path, help="aligned copy to write")
    ap.add_argument(
        "--ref-scale",
        type=float,
        default=1.0,
        help="metres per reference unit (manifest metersPerUnit); output is in metres",
    )
    ap.add_argument(
        "--up",
        type=str,
        default=None,
        help=(
            "the stray cloud's up vector 'x,y,z' in its own frame, overriding "
            "the density measurement — which reads the densest sheet as the "
            "ground and gets the SIGN wrong when a nadir capture reconstructs "
            "the canopy better than the ground under it."
        ),
    )
    ap.add_argument(
        "--crop",
        type=float,
        default=None,
        metavar="PAD",
        help=(
            "drop splats further than PAD metres outside the reference's core "
            "box. A long-trained export carries huge floater gaussians tens of "
            "metres above the site; on screen they read as a tilted second "
            "ground. The reference core box is the honest footprint."
        ),
    )
    args = ap.parse_args()

    stray = Ply.read(args.stray)
    ref = Ply.read(args.reference)
    ref_core = _core(ref.centers())

    # 1. Level the stray cloud.
    if args.up:
        up = np.array([float(v) for v in args.up.split(",")])
        up /= np.linalg.norm(up)
    else:
        up = measure_up(stray.centers())
    q_level = quat_from_to(up, np.array([0.0, 1.0, 0.0]))
    leveled = stray.centers() @ quat_to_matrix(q_level).T

    # 2. Yaw candidates from the horizontal principal axes.
    d_stray = _principal_xz(_core(leveled))
    d_ref = _principal_xz(ref_core)
    base = np.arctan2(d_stray[1], d_stray[0]) - np.arctan2(d_ref[1], d_ref[0])

    cross_ref = np.array([-d_ref[1], d_ref[0]])
    ref_along = ref_core[:, [0, 2]] @ d_ref
    lo, hi = np.percentile(ref_along, [0.5, 99.5])
    slab = (hi - lo) / 40.0
    bins = np.arange(lo - 2, hi + 2 + slab, slab)
    ref_density = _density(ref_along, bins)

    # Joint (yaw, scale, shift) search on the density profile. Nested loops over
    # a coarse grid are plenty: the profile is 40-odd slabs.
    best: tuple[float, np.ndarray, float, float] | None = None
    for extra in (0.0, np.pi):
        rot = _yaw_matrix(base + extra)
        cand_along_raw = _core(leveled @ rot.T)[:, [0, 2]] @ d_ref
        clo, chi = np.percentile(cand_along_raw, [0.5, 99.5])
        span_scale = (hi - lo) / (chi - clo)
        for scale in np.linspace(span_scale * 0.9, span_scale * 1.1, 41):
            scaled = cand_along_raw * scale
            centre_shift = (lo + hi) / 2 - (scaled.min() + scaled.max()) / 2
            for shift in np.arange(-3 * slab, 3 * slab, slab / 4):
                density = _density(scaled + centre_shift + shift, bins)
                score = float(np.corrcoef(ref_density, density)[0, 1])
                if best is None or score > best[0]:
                    best = (score, rot, scale, centre_shift + shift)
        print(f"yaw {np.degrees(base + extra):7.1f} deg searched")

    assert best is not None
    score, rot, scale, along_shift = best
    print(f"best: density corr {score:+.3f}  scale {scale:.4f}  along-shift {along_shift:+.3f}u")
    if score < 0.5:
        print("WARNING: weak density correlation — check the result visually")

    # 4. Remaining translation: cross-axis + height from matched slabs.
    cand_core = _core(leveled @ rot.T) * scale
    cand_along = cand_core[:, [0, 2]] @ d_ref + along_shift
    cand_cross = cand_core[:, [0, 2]] @ cross_ref
    cand_idx = np.digitize(cand_along, bins) - 1
    ref_idx = np.digitize(ref_along, bins) - 1
    ref_cross = ref_core[:, [0, 2]] @ cross_ref
    cross_diffs, y_diffs = [], []
    for b in range(len(bins) - 1):
        in_ref = ref_idx == b
        in_cand = cand_idx == b
        if in_ref.sum() >= 200 and in_cand.sum() >= 200:
            cross_diffs.append(np.median(ref_cross[in_ref]) - np.median(cand_cross[in_cand]))
            y_diffs.append(np.median(ref_core[in_ref, 1]) - np.median(cand_core[in_cand, 1]))
    cross_shift = float(np.median(cross_diffs)) if cross_diffs else 0.0
    y_shift = float(np.median(y_diffs)) if y_diffs else 0.0

    # Compose the world-space translation from the axis-space shifts.
    shift = np.array(
        [
            along_shift * d_ref[0] + cross_shift * cross_ref[0],
            y_shift,
            along_shift * d_ref[1] + cross_shift * cross_ref[1],
        ]
    )

    # 3. Bake, in metres: p' = (R p) * scale * ref_scale + shift * ref_scale.
    q_total = _matrix_to_quat(rot @ quat_to_matrix(q_level))
    rotate_ply(stray, q_total)
    stray.rescale(scale * args.ref_scale)
    data = stray.data.copy()
    for i, axis_name in enumerate(("x", "y", "z")):
        data[axis_name] = data[axis_name] + np.float32(shift[i] * args.ref_scale)
    stray.data = data

    keep = np.ones(len(stray.data), dtype=bool)
    if args.crop is not None:
        box_lo = (np.percentile(ref.centers(), 0.5, axis=0)) * args.ref_scale - args.crop
        box_hi = (np.percentile(ref.centers(), 99.5, axis=0)) * args.ref_scale + args.crop
        pts = stray.centers()
        keep = np.all((pts >= box_lo) & (pts <= box_hi), axis=1)
        print(f"crop to ref core +{args.crop} m: keeping {int(keep.sum())}/{len(keep)} splats")

    stray.write_subset(args.out, keep)
    print(
        f"aligned {args.stray.name} -> {args.out}  "
        f"(scale x{scale * args.ref_scale:.3f}, shift {np.round(shift * args.ref_scale, 2).tolist()} m)"
    )


if __name__ == "__main__":
    main()
