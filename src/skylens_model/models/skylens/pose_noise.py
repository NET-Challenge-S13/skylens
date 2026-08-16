"""Inject controlled error into COLMAP camera poses.

Answers a design question we cannot answer by reasoning: how accurately does a
drone's pose have to be known before 3DGS quality collapses? Perturb the solved
poses by a known amount, retrain, and read the loss off the PSNR.

Rotation noise is given in degrees and applied about a random axis. Translation
noise is given as a fraction of scene extent, so the same setting means the same
*relative* error whether the scene is a corridor or a city block; it is applied
to the camera centre, then folded back into ``tvec``.

Result of the sweep (``RESULTS.md``): position error hurts far more than
attitude error. 0.2% position costs 2.24 dB, which takes roughly 0.8 deg of
rotation to match. Rotating a camera slides the whole frame and stays partly
correctable; moving one shifts near and far geometry by different amounts, which
breaks triangulation outright.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .colmap_io import (
    copy_model_except,
    qvec_to_rotmat,
    read_images,
    rotmat_to_qvec,
    write_images,
)


def random_rotation(rng: np.random.Generator, degrees: float) -> np.ndarray:
    """Rotation of ``degrees`` about a uniformly random axis (Rodrigues)."""
    if degrees <= 0:
        return np.eye(3)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    theta = np.deg2rad(degrees)
    cross = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(theta) * cross + (1 - np.cos(theta)) * (cross @ cross)


def perturb(
    src: str | Path,
    dst: str | Path,
    rot_deg: float = 0.0,
    trans_ratio: float = 0.0,
    seed: int = 42,
) -> dict[str, float]:
    """Write a perturbed copy of the model. Returns a summary of what moved."""
    src_dir, dst_dir = Path(src), Path(dst)
    copy_model_except(src_dir, dst_dir, skip={"images.bin"})

    images = read_images(src_dir / "images.bin")
    rng = np.random.default_rng(seed)

    centers = np.array([im.center for im in images])
    scene_extent = float(np.linalg.norm(centers.max(0) - centers.min(0)))
    sigma_t = trans_ratio * scene_extent

    displacement = []
    for im in images:
        rot = qvec_to_rotmat(im.qvec)
        center = -rot.T @ im.tvec

        if rot_deg > 0:
            # Draw a per-frame magnitude so the sweep sets a sigma, not a constant.
            rot = rot @ random_rotation(rng, abs(rng.normal(0, rot_deg)))
        new_center = center + rng.normal(0, sigma_t, 3) if sigma_t > 0 else center

        im.qvec = rotmat_to_qvec(rot)
        im.tvec = -rot @ new_center
        displacement.append(float(np.linalg.norm(new_center - center)))

    write_images(dst_dir / "images.bin", images)

    moved = np.array(displacement)
    print(
        f"[pose noise] {len(images)} frames | attitude sigma={rot_deg} deg | "
        f"position sigma={sigma_t:.4f} ({trans_ratio * 100:.2f}% of extent {scene_extent:.2f})"
    )
    print(f"             centre shift: median {np.median(moved):.4f} / max {moved.max():.4f}")
    return {"scene_extent": scene_extent, "sigma_t": sigma_t, "median_shift": float(np.median(moved))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", required=True, help="source sparse model dir")
    parser.add_argument("--dst", required=True, help="output sparse model dir")
    parser.add_argument("--rot-deg", type=float, default=0.0, help="attitude noise sigma, degrees")
    parser.add_argument("--trans-ratio", type=float, default=0.0, help="position noise, fraction of extent")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    perturb(args.src, args.dst, args.rot_deg, args.trans_ratio, args.seed)


if __name__ == "__main__":
    main()
