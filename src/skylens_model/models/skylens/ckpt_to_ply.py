"""Export a gsplat checkpoint to a standard 3DGS ``.ply``.

The file is a PLY *container*, not a point cloud. Each element carries 59
properties, against the 6 (xyz + rgb) a point cloud would have::

    x, y, z                position                     3
    f_dc_0..2              base colour, SH degree 0      3
    f_rest_0..44           view-dependent colour, SH 1-3 45
    opacity                                              1
    scale_0..2             per-axis extent               3
    rot_0..3               orientation quaternion        4

``scale`` and ``rot`` are what make a gaussian a squashed, oriented ellipsoid
rather than a dot, which is how flat surfaces form. Generic point-cloud viewers
(CloudCompare, MeshLab) read only the first three and silently drop the rest,
so the model appears as loose dots -- the file is fine, the viewer is not.
Use SuperSplat, the gsplat viewer, or the RECON viewer in this repo instead.

Values are written as the raw optimised parameters: opacity is a logit, scale is
logarithmic. That is what the INRIA reference format stores, and what every 3DGS
viewer expects to invert on load.

``--light`` drops the 45 ``f_rest`` properties, shrinking the file ~4x with the
geometry untouched. Only the angle-dependent colour shift is lost. Worth it for
anything crossing a network.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

SH_REST_COUNT = 45


def export(ckpt_path: str | Path, out_path: str | Path, light: bool = False) -> Path:
    """Convert one checkpoint. Returns the written path."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    splats = ckpt["splats"]

    xyz = splats["means"].float().numpy()
    count = xyz.shape[0]

    columns = [xyz, splats["sh0"].float().numpy().reshape(count, 3)]
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]

    if not light:
        # (N, 15, 3) -> channel-major, matching the reference layout:
        # 15 coefficients for R, then G, then B.
        rest = splats["shN"].float().numpy().transpose(0, 2, 1).reshape(count, -1)
        columns.append(rest)
        names += [f"f_rest_{i}" for i in range(rest.shape[1])]

    columns += [
        splats["opacities"].float().numpy().reshape(count, 1),
        splats["scales"].float().numpy(),
        splats["quats"].float().numpy(),
    ]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]

    data = np.concatenate(columns, axis=1).astype(np.float32)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {count}\n".encode())
        for name in names:
            f.write(f"property float {name}\n".encode())
        f.write(b"end_header\n")
        f.write(data.tobytes())

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"  {out.name:<28} {count:>9,} GS  {len(names):>2} props  {size_mb:6.1f} MB")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt", required=True, help="gsplat checkpoint (.pt)")
    parser.add_argument("--out", required=True, help="output .ply path")
    parser.add_argument("--light", action="store_true", help="drop SH rest (~4x smaller)")
    args = parser.parse_args()
    export(args.ckpt, args.out, args.light)


if __name__ == "__main__":
    main()
