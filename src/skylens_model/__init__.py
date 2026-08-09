"""SkyLens model package.

Houses the AI/reconstruction model code for SkyLens: a multi-drone disaster
intelligence platform that reconstructs a scene in real-time 3D (Gaussian
Splatting) and overlays AI-detected danger zones and people, projected from
2D detections into 3D via depth-map raycasting.

This package is a SCAFFOLD: interfaces and data shapes only, no trained
models or inference logic yet. See `skylens_model/README.md` for the
folder map and how this plugs into the server/client wire protocol.
"""

from __future__ import annotations

__version__ = "0.1.0"
