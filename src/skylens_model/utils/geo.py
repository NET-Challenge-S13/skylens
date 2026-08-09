"""GPS <-> local ENU (East/North/Up) coordinate conversion.

Pure-Python mirror of `src/skylens_core/geo.ts`. Detections and splat chunks
produced by this package must line up with the client's rendered scene, and
the client operates in a local ENU meter frame anchored at a reference GPS
point (1 scene unit = 1 meter).

IMPORTANT: this module must stay numerically in sync with geo.ts. Same
equirectangular small-area approximation, same Earth radius constant, and
the same scene axis convention: x = East, y = Up, z = -North. If you change
the formulas here, change geo.ts too (and vice versa).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_R = 6378137.0  # Earth radius (m), matches geo.ts
_DEG = math.pi / 180.0


@dataclass(slots=True)
class Gps:
    """Matches the `Gps` shape in src/skylens_core/geo.ts."""

    lat: float
    lon: float
    alt: float  # meters, ellipsoidal/relative — used as-is for Up


# Reference point that defines the local ENU origin. Alias for clarity,
# matching `GeoAnchor` in geo.ts.
GeoAnchor = Gps


@dataclass(slots=True)
class Enu:
    """Local East/North/Up offset in meters from a `GeoAnchor`."""

    e: float
    n: float
    u: float


def gps_to_enu(gps: Gps, anchor: GeoAnchor) -> Enu:
    """GPS -> local ENU meters (equirectangular small-area approximation)."""
    d_lat = (gps.lat - anchor.lat) * _DEG
    d_lon = (gps.lon - anchor.lon) * _DEG
    return Enu(
        e=d_lon * _R * math.cos(anchor.lat * _DEG),
        n=d_lat * _R,
        u=gps.alt - anchor.alt,
    )


def enu_to_gps(enu: Enu, anchor: GeoAnchor) -> Gps:
    """Local ENU meters -> GPS (inverse of `gps_to_enu`)."""
    return Gps(
        lat=anchor.lat + (enu.n / _R) / _DEG,
        lon=anchor.lon + (enu.e / (_R * math.cos(anchor.lat * _DEG))) / _DEG,
        alt=anchor.alt + enu.u,
    )


# Scene convention (matches the client's Three.js right-handed frame):
# x = East, y = Up, z = -North.
SceneVec3 = tuple[float, float, float]


def enu_to_scene(enu: Enu) -> SceneVec3:
    return (enu.e, enu.u, -enu.n)


def scene_to_enu(v: SceneVec3) -> Enu:
    return Enu(e=v[0], u=v[1], n=-v[2])


def gps_to_scene(gps: Gps, anchor: GeoAnchor) -> SceneVec3:
    return enu_to_scene(gps_to_enu(gps, anchor))


def scene_to_gps(v: SceneVec3, anchor: GeoAnchor) -> Gps:
    return enu_to_gps(scene_to_enu(v), anchor)
