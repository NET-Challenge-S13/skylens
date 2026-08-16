"""SkyLensRecon -- 3D Gaussian Splatting reconstruction pipeline.

Turns multi-drone footage into a single 3D scene: COLMAP solves where every
frame was shot from, then gsplat fits gaussians to those camera poses.

The pipeline itself lives in ``pipeline/`` as shell scripts because it
orchestrates external binaries (ffmpeg, colmap) rather than Python libraries.
This package holds the Python tools those scripts call, plus the tools used to
run the design experiments in ``RESULTS.md``.

See ``README.md`` for why each stage is built the way it is.
"""

from __future__ import annotations

__all__: list[str] = []
