"""Split 3DGS ``.ply`` files into capture segments for the delay-pattern stream.

The reconstruction pipeline is online: the drone keeps flying, so a segment of
the flight is reconstructed as soon as its frames land, at a LOW training step
count first, and refined afterwards while the NEXT segment starts its own low
level. That stagger is the delay pattern (see the interim report, Ⅱ-3-다).

A gsplat run exports one ``.ply`` per training step for the WHOLE scene, so the
demo assets carry the *level* axis (250 / 1,000 / 3,500 / 7,000 steps) but not
the *segment* axis. This script adds it: it cuts every level file along the same
spatial boundaries, so segment ``k`` at any level covers the same piece of the
scene and a higher level can replace a lower one in place.

The cut axis is the scene's principal axis, which for a corridor sweep is the
direction the drone travelled. Boundaries are quantiles of the reference (the
highest-step) file, so segments hold a comparable number of gaussians, and the
SAME boundary values are applied to every level.

Segments partition the scene: no point is duplicated, and the union of all
segments at one level is that level's original file.

Usage::

    uv run python -m skylens_model.models.skylens.split_segments \\
        res/static/demo/step*_light.ply --segments 4

Writes ``<out>/seg<k>_step<NNNNN>.ply`` plus a ``segments.json`` manifest the
status-board client streams from.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# Reconstruction quality per training step, measured in RESULTS.md and quoted in
# the interim report (표 8). Shown on the status board as each level lands.
STEP_LABELS = {
    250: "형상 윤곽만 식별",
    1000: "공간 구조 식별 가능",
    3500: "표면 형성",
    7000: "실용 품질",
    15000: "형상 개수 고정",
    30000: "최종",
}

_STEP_RE = re.compile(r"step0*(\d+)")


def steps_from_name(path: Path) -> int:
    """Training-step count encoded in a checkpoint export's filename."""
    m = _STEP_RE.search(path.stem)
    if not m:
        raise ValueError(f"no step count in filename: {path.name}")
    return int(m.group(1))


class Ply:
    """A binary-little-endian PLY of all-float vertex properties (3DGS export)."""

    def __init__(self, header: list[str], props: list[str], data: np.ndarray) -> None:
        self.header = header
        self.props = props
        self.data = data

    @classmethod
    def read(cls, path: Path) -> Ply:
        raw = path.read_bytes()
        end = raw.find(b"end_header\n")
        if end < 0:
            raise ValueError(f"{path.name}: no end_header")
        header = raw[:end].decode("ascii").splitlines()
        body = raw[end + len(b"end_header\n") :]

        if not any(line.startswith("format binary_little_endian") for line in header):
            raise ValueError(f"{path.name}: only binary_little_endian is supported")

        count = 0
        props: list[str] = []
        for line in header:
            if line.startswith("element vertex"):
                count = int(line.split()[2])
            elif line.startswith("property"):
                kind, name = line.split()[1:3]
                if kind != "float":
                    raise ValueError(f"{path.name}: non-float property {name} ({kind})")
                props.append(name)

        dtype = np.dtype([(p, "<f4") for p in props])
        data = np.frombuffer(body, dtype=dtype, count=count)
        return cls(header, props, data)

    def rescale(self, factor: float) -> None:
        """Resize the scene in place, gaussians and all.

        A 3DGS export carries no unit: the solve fixes shape, not size. Every
        other coordinate in this system is metres, so the asset is brought into
        metres here rather than being stretched by a transform when it is drawn
        — a viewer-side scale moves the splat centres apart without the surface
        following convincingly, and the board rendered a scene that measured
        correctly and showed nothing.

        Positions scale directly; ``scale_0..2`` are LOG standard deviations, so
        they take the logarithm of the factor instead.
        """
        if factor <= 0.0 or factor == 1.0:
            return
        editable = self.data.copy()
        for axis in ("x", "y", "z"):
            editable[axis] = editable[axis] * factor
        for axis in ("scale_0", "scale_1", "scale_2"):
            if axis in self.props:
                editable[axis] = editable[axis] + np.log(factor)
        self.data = editable

    def centers(self) -> np.ndarray:
        return np.stack([self.data["x"], self.data["y"], self.data["z"]], axis=1).astype(np.float64)

    def write_subset(self, path: Path, mask: np.ndarray) -> int:
        """Write the masked vertices out with this file's property layout."""
        subset = self.data[mask]
        lines = []
        for line in self.header:
            if line.startswith("element vertex"):
                lines.append(f"element vertex {len(subset)}")
            else:
                lines.append(line)
        blob = ("\n".join(lines) + "\nend_header\n").encode("ascii")
        path.write_bytes(blob + subset.tobytes())
        return len(subset)


def principal_axis(centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dominant direction of the scene, ignoring outlier floaters.

    SfM scenes carry stray gaussians far outside the captured space; they would
    drag a plain PCA toward themselves, so the axis is fit on the 1-99 percentile
    core only.
    """
    lo = np.percentile(centers, 1, axis=0)
    hi = np.percentile(centers, 99, axis=0)
    core = centers[np.all((centers >= lo) & (centers <= hi), axis=1)]
    if len(core) < 3:
        core = centers
    mean = core.mean(axis=0)
    _, _, vh = np.linalg.svd(core - mean, full_matrices=False)
    axis = vh[0]
    # Sign is arbitrary out of SVD; pin it so repeated runs number segments the
    # same way.
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    return axis, mean


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plys", nargs="+", type=Path, help="level exports, one per training step")
    ap.add_argument("--segments", type=int, default=4, help="number of capture segments (default 4)")
    ap.add_argument(
        "--length",
        type=float,
        default=None,
        help=(
            "ground distance in metres the scene spans along its principal axis "
            "(the flight it was captured on). Given, the export is rewritten in "
            "metres so it can be placed by a rigid transform."
        ),
    )
    ap.add_argument("--out", type=Path, default=None, help="output dir (default <input dir>/segments)")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest path (default <input dir>/segments.json)",
    )
    args = ap.parse_args()

    if args.segments < 1:
        ap.error("--segments must be >= 1")

    levels = sorted(args.plys, key=steps_from_name)
    src_dir = levels[0].parent
    out_dir = args.out or src_dir / "segments"
    manifest_path = args.manifest or src_dir / "segments.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The highest-step file is the reference: it has the most gaussians, so its
    # quantiles are the most stable estimate of where the scene should be cut.
    reference = Ply.read(levels[-1])
    ref_centers = reference.centers()
    axis, origin = principal_axis(ref_centers)

    # Bring the whole scene into metres before anything is cut, so every level
    # and every segment shares one unit.
    meters_per_unit = 1.0
    if args.length is not None:
        ref_proj_raw = (ref_centers - origin) @ axis
        span = float(np.diff(np.percentile(ref_proj_raw, [1, 99]))[0])
        if span <= 0:
            ap.error("the reference cloud has no extent along its principal axis")
        meters_per_unit = args.length / span
        print(f"scene spans {span:.3f} units -> {args.length} m  ({meters_per_unit:.3f} m/unit)")
        reference.rescale(meters_per_unit)
        ref_centers = reference.centers()
        origin = origin * meters_per_unit
    ref_proj = (ref_centers - origin) @ axis
    qs = np.linspace(0, 100, args.segments + 1)[1:-1]
    boundaries = np.percentile(ref_proj, qs) if len(qs) else np.array([])

    print(f"axis {np.round(axis, 4).tolist()}  segments {args.segments}")

    # A reconstruction from images has no metric scale: its units are whatever
    # the SfM solve settled on. Whoever places it has to resolve that against
    # something measured, and the only measured thing in this system is the
    # flight. So record how big each piece IS, in the scene's own units, and let
    # the serving side map that onto the ground the segment was flown over.
    # Without it the demo drew a 4.7 m speck for every 40 m of flight.
    ref_assign = (
        np.searchsorted(boundaries, ref_proj) if len(boundaries) else np.zeros(len(ref_proj), int)
    )
    segment_shape: list[dict] = []
    for k in range(args.segments):
        own = ref_centers[ref_assign == k]
        if len(own) == 0:
            segment_shape.append({"extent": 0.0, "centroid": [0.0, 0.0, 0.0]})
            continue
        own_proj = (own - origin) @ axis
        # 1-99 percentile, the same trim the axis fit uses: a handful of stray
        # gaussians well outside the scene would otherwise set the size.
        lo, hi = np.percentile(own_proj, [1, 99])
        segment_shape.append(
            {
                "extent": float(hi - lo),
                "centroid": np.percentile(own, 50, axis=0).tolist(),
            }
        )

    segments: list[dict] = [
        {"index": k, "levels": [], **segment_shape[k]} for k in range(args.segments)
    ]

    for level_index, ply_path in enumerate(levels, start=1):
        ply = Ply.read(ply_path)
        ply.rescale(meters_per_unit)
        steps = steps_from_name(ply_path)
        proj = (ply.centers() - origin) @ axis
        assign = np.searchsorted(boundaries, proj) if len(boundaries) else np.zeros(len(proj), int)

        for k in range(args.segments):
            out_path = out_dir / f"seg{k}_step{steps:05d}.ply"
            splats = ply.write_subset(out_path, assign == k)
            segments[k]["levels"].append(
                {
                    "level": level_index,
                    "steps": steps,
                    "label": STEP_LABELS.get(steps, ""),
                    "url": f"{out_dir.name}/{out_path.name}",
                    "splats": splats,
                    "bytes": out_path.stat().st_size,
                }
            )
            print(f"  seg{k} step{steps:05d}: {splats:>7} splats  {out_path.stat().st_size / 1e6:.2f} MB")

    manifest = {
        "source": [p.name for p in levels],
        "reference": levels[-1].name,
        # The split geometry, so a viewer could re-derive the same cut.
        "axis": axis.tolist(),
        "origin": origin.tolist(),
        "boundaries": boundaries.tolist(),
        # 1.0 once the export has been rewritten in metres (--length); the
        # serving side then places a chunk with a rigid transform.
        "metersPerUnit": meters_per_unit,
        # Full extent along the axis (1-99 percentile), for anyone placing the
        # scene as one piece rather than segment by segment.
        "extent": float(np.diff(np.percentile(ref_proj, [1, 99]))[0]),
        # Cheapest level of the whole scene: the client loads it once to derive
        # the shared fit transform + the "not yet scanned" ghost cloud.
        "preview": levels[0].name,
        "segments": segments,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
