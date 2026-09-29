"""Fan one planned route into a three-aircraft formation of DJI Fly missions.

SkyLens flies three aircraft one after another to stand in for a simultaneous
formation (skylens_model/models/skylens/route_gen.py explains why one drone
cannot fly three legs at once). Doing that by hand meant computing three offset
waypoint sets in a spreadsheet and typing each into DJI Fly. This module does
the same arithmetic and writes the three KMZ missions directly.

Geometry. The planned route is the formation CENTROID track. The three members
sit at the vertices of an equilateral triangle of side ``spacing_m`` in the
horizontal plane, translated rigidly with the route:

    front-left  (station 'left')   forward +R/2, lateral -s/2
    front-right (station 'right')  forward +R/2, lateral +s/2
    rear-centre (station 'center') forward -R,   lateral  0        R = s/sqrt(3)

Lateral separation across the front pair is exactly ``s``; the rear aircraft
trails the front midline by sqrt(3)/2 * s. That is the shape measured off the
hand-built standard2-*.lchz missions (side ~10 m), and the same triangle
route_gen.py sweeps. Forward is the route bearing (first waypoint -> last);
lateral is positive to the right of travel.

Altitude is left untouched: every member keeps the planned per-waypoint GPS
altitude, so mission.py derives the same relative heights against the shared
take-off point. Only the horizontal position shifts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from mission import MissionError, MissionResult, build_mission

# Metres per degree of latitude; longitude is scaled by cos(lat) at use. Flat
# approximation, matching route_gen.py — error is sub-centimetre over the tens
# of metres a formation offset spans.
M_PER_DEG_LAT = 111_320.0

# Order matters only for readability; each is (forward_m_factor, lateral_m_factor)
# in units where spacing_m == s. R = s / sqrt(3).
_SQRT3 = math.sqrt(3.0)


def member_offsets(spacing_m: float) -> dict[str, tuple[float, float]]:
    """Return {station: (forward_m, lateral_m)} for an equilateral triangle."""
    if not math.isfinite(spacing_m) or spacing_m <= 0:
        raise MissionError("formation.spacingM must be a positive number")
    r = spacing_m / _SQRT3
    return {
        "left": (r / 2.0, -spacing_m / 2.0),   # front-left
        "right": (r / 2.0, spacing_m / 2.0),   # front-right
        "center": (-r, 0.0),                   # rear-centre
    }


def route_bearing_deg(waypoints: list[dict[str, Any]]) -> float:
    """Bearing (deg from north) of the straight line first -> last waypoint."""
    first, last = waypoints[0], waypoints[-1]
    lat0 = math.radians(float(first["lat"]))
    east = (float(last["lon"]) - float(first["lon"])) * M_PER_DEG_LAT * math.cos(lat0)
    north = (float(last["lat"]) - float(first["lat"])) * M_PER_DEG_LAT
    if east == 0.0 and north == 0.0:
        raise MissionError("route start and end coincide; cannot derive a heading")
    return math.degrees(math.atan2(east, north))


def _shift(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    dlat = north_m / M_PER_DEG_LAT
    dlon = east_m / (M_PER_DEG_LAT * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def offset_waypoints(
    waypoints: list[dict[str, Any]], heading_deg: float, forward_m: float, lateral_m: float
) -> list[dict[str, Any]]:
    """Rigidly translate every waypoint by a (forward, lateral) offset."""
    h = math.radians(heading_deg)
    east = forward_m * math.sin(h) + lateral_m * math.cos(h)
    north = forward_m * math.cos(h) - lateral_m * math.sin(h)
    out = []
    for wp in waypoints:
        lat, lon = _shift(float(wp["lat"]), float(wp["lon"]), east, north)
        out.append({"lat": lat, "lon": lon, "alt": float(wp["alt"])})
    return out


def build_formation(
    payload: dict[str, Any],
    template_path: Path,
    output_dir: Path,
    base_name: str,
    stamp: str,
    limits: dict[str, float] | None = None,
) -> list[tuple[str, MissionResult]]:
    """Write one KMZ per formation member. Returns [(station, result), ...]."""
    raw = payload.get("waypoints")
    if not isinstance(raw, list) or len(raw) < 2:
        raise MissionError("waypoints must contain at least two points")

    formation = payload.get("formation")
    spacing = 10.0
    if isinstance(formation, dict) and formation.get("spacingM") is not None:
        spacing = float(formation["spacingM"])
    offsets = member_offsets(spacing)
    heading = route_bearing_deg(raw)

    flight = payload.get("flight") if isinstance(payload.get("flight"), dict) else {}
    results: list[tuple[str, MissionResult]] = []
    for station, (forward_m, lateral_m) in offsets.items():
        member_payload = {
            "name": f"{base_name}-{station}",
            "loop": bool(payload.get("loop")),
            "flight": flight,
            "waypoints": offset_waypoints(raw, heading, forward_m, lateral_m),
        }
        output_path = output_dir / f"{stamp}-{base_name}-{station}.kmz"
        results.append((station, build_mission(member_payload, template_path, output_path, limits)))
    return results
