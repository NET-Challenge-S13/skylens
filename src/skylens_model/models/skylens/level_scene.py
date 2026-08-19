"""Stand a 3DGS export upright: bake the leveling rotation into the ``.ply``.

The demo capture's SfM solve settled on a frame where "up" is nowhere near +Y —
measured ~83 degrees off, so the corridor rendered lying on its side. Every
consumer of these assets (the viewer's rigid route placement, the segment
splitter's axis cut, the detection anchors) assumes scene-convention axes:
y up, ground horizontal. Rather than teaching each of them a correction
transform, this bakes ONE rotation into the exports and everything downstream
stays plain.

Up is measured, not guessed: the smallest-variance principal axis of the
outlier-trimmed cloud is the normal of the dominant plane (the ground), and the
sign is chosen so the DENSE side of that axis — the ground sheet — ends up at
the bottom. Both --up and the measurement can be overridden when a scene breaks
the assumption.

Usage::

    uv run python -m skylens_model.models.skylens.level_scene \\
        res/static/demo/step*_light.ply

Files are rewritten in place; originals are kept under ``<dir>/orig/`` (the
solve cannot be re-run casually, so never destroy the only copy).

Run ``split_segments`` again afterwards — the manifest's axis/boundaries
describe the old frame.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from .split_segments import Ply


def measure_up(centers: np.ndarray) -> np.ndarray:
    """Unit up-vector of the scene in its own frame.

    Smallest-variance principal axis of the 1-99 percentile core = normal of
    the dominant plane. Sign: the ground is the densest thing a drone scan
    sees, so the third of the height range holding MORE splats is the bottom.
    """
    lo = np.percentile(centers, 1, axis=0)
    hi = np.percentile(centers, 99, axis=0)
    core = centers[np.all((centers >= lo) & (centers <= hi), axis=1)]
    mean = core.mean(axis=0)
    cov = np.cov((core - mean).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    up = eigenvectors[:, int(np.argmin(eigenvalues))]

    h = (core - mean) @ up
    a, b = np.percentile(h, [1, 99])
    third = (b - a) / 3.0
    bottom = float(((h >= a) & (h < a + third)).mean())
    top = float(((h > b - third) & (h <= b)).mean())
    if top > bottom:
        up = -up
    return up / np.linalg.norm(up)


def quat_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation a -> b, as (w, x, y, z)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 0.0, 1.0]))
        axis /= np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    w = np.sqrt((1.0 + d) * 2.0) / 2.0
    c = np.cross(a, b) / (2.0 * w)
    q = np.array([w, c[0], c[1], c[2]])
    return q / np.linalg.norm(q)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotate_ply(ply: Ply, rotation: np.ndarray) -> None:
    """Apply a rotation (w,x,y,z about the origin) to positions AND gaussians.

    Per-gaussian ``rot_0..3`` is the INRIA-convention quaternion (w first);
    rotating the scene left-multiplies it. Log-scales are axis lengths in the
    gaussian's OWN frame, so they do not change.
    """
    data = ply.data.copy()
    m = quat_to_matrix(rotation)
    p = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    p = p @ m.T
    data["x"], data["y"], data["z"] = (p[:, i].astype(np.float32) for i in range(3))

    if all(f"rot_{i}" in ply.props for i in range(4)):
        w1, x1, y1, z1 = rotation
        w2 = data["rot_0"].astype(np.float64)
        x2 = data["rot_1"].astype(np.float64)
        y2 = data["rot_2"].astype(np.float64)
        z2 = data["rot_3"].astype(np.float64)
        data["rot_0"] = (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2).astype(np.float32)
        data["rot_1"] = (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2).astype(np.float32)
        data["rot_2"] = (w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2).astype(np.float32)
        data["rot_3"] = (w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2).astype(np.float32)

    ply.data = data


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("plys", nargs="+", type=Path, help="exports of ONE scene (all levels)")
    ap.add_argument(
        "--up",
        type=str,
        default=None,
        help="raw-frame up vector 'x,y,z' (default: measured from the largest file)",
    )
    ap.add_argument(
        "--backup",
        type=Path,
        default=None,
        help="where originals go (default <input dir>/orig)",
    )
    args = ap.parse_args()

    backup_dir = args.backup or args.plys[0].parent / "orig"
    backup_dir.mkdir(parents=True, exist_ok=True)

    if args.up:
        up = np.array([float(v) for v in args.up.split(",")], dtype=np.float64)
        up /= np.linalg.norm(up)
    else:
        # The largest file has the most gaussians: the most stable measurement.
        reference = max(args.plys, key=lambda p: p.stat().st_size)
        up = measure_up(Ply.read(reference).centers())
        print(f"measured up (raw frame): {np.round(up, 4).tolist()}  [{reference.name}]")

    rotation = quat_from_to(up, np.array([0.0, 1.0, 0.0]))
    angle = float(2 * np.degrees(np.arccos(np.clip(rotation[0], -1, 1))))
    print(f"leveling rotation: {angle:.1f} deg, quat(w,x,y,z) {np.round(rotation, 6).tolist()}")

    for path in args.plys:
        saved = backup_dir / path.name
        if not saved.exists():
            shutil.copy2(path, saved)
        ply = Ply.read(path)
        rotate_ply(ply, rotation)
        ply.write_subset(path, np.ones(len(ply.data), dtype=bool))
        print(f"  leveled {path.name} ({len(ply.data)} splats)")


if __name__ == "__main__":
    main()
