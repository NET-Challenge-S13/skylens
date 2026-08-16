"""Cut a COLMAP model down to the frames from selected captures.

Used by the formation experiment. Taking subsets of one joint reconstruction --
rather than reconstructing each capture separately -- keeps every condition in
the *same* coordinate frame, so their outputs are directly comparable. Separate
reconstructions would each get their own arbitrary scale and orientation.

Filtering ``images.bin`` alone is not enough. Tracks in ``points3D.bin`` still
reference the image ids that were dropped, and gsplat's loader raises
``KeyError`` on the first stale id. Tracks have to be pruned in step, and points
whose track falls below two observations are removed outright -- a point seen
once cannot be triangulated, so keeping it only adds noise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .colmap_io import (
    copy_model_except,
    read_images,
    read_points3d,
    write_images,
    write_points3d,
)

MIN_TRACK_LENGTH = 2


def subset(src: str | Path, dst: str | Path, keep_prefixes: list[str]) -> tuple[int, int]:
    """Keep frames whose filename starts with any prefix. Returns (images, points)."""
    src_dir, dst_dir = Path(src), Path(dst)
    copy_model_except(src_dir, dst_dir, skip={"images.bin", "points3D.bin"})

    images = read_images(src_dir / "images.bin")
    kept = [im for im in images if im.name.decode().startswith(tuple(keep_prefixes))]
    kept_ids = {im.image_id for im in kept}
    write_images(dst_dir / "images.bin", kept)

    points = read_points3d(src_dir / "points3D.bin")
    kept_points = []
    for pt in points:
        pt.track = [(i, idx) for i, idx in pt.track if i in kept_ids]
        if len(pt.track) >= MIN_TRACK_LENGTH:
            kept_points.append(pt)
    write_points3d(dst_dir / "points3D.bin", kept_points)

    print(
        f"[subset] images {len(images)} -> {len(kept)} | "
        f"points {len(points):,} -> {len(kept_points):,}  ({', '.join(keep_prefixes)})"
    )
    return len(kept), len(kept_points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", required=True, help="source sparse model dir")
    parser.add_argument("--dst", required=True, help="output sparse model dir")
    parser.add_argument("--keep", nargs="+", required=True, help="image name prefixes to keep")
    args = parser.parse_args()
    subset(args.src, args.dst, args.keep)


if __name__ == "__main__":
    main()
