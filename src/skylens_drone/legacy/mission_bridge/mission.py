"""Build a DJI Fly KMZ mission from a SkyLens GPS route.

DJI does not publish the consumer-aircraft enum values used by DJI Fly. A KMZ
created by the actual Lito X1 is therefore mandatory: this module reads the
aircraft and payload identifiers from that file and refuses to fall back to an
enterprise model. The generated dialect follows the KMZ/WPML shape used by
DJI Fly/Litchi imports; ground-test it after DJI Fly or firmware updates.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WPML_NS = "http://www.dji.com/wpmz/1.0.2"
KML_NS = "http://www.opengis.net/kml/2.2"


class MissionError(ValueError):
    """Input or template cannot safely produce a mission."""


@dataclass(frozen=True)
class Waypoint:
    lat: float
    lon: float
    absolute_alt_msl: float
    relative_height: float


@dataclass(frozen=True)
class MissionResult:
    path: Path
    sha256: str
    waypoint_count: int
    route_distance_m: float
    takeoff_alt_msl: float
    warnings: tuple[str, ...]


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise MissionError(f"{label} must be a number")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise MissionError(f"{label} must be a number") from exc
    if not math.isfinite(out):
        raise MissionError(f"{label} must be finite")
    return out


def _distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6_371_000.0 * 2 * math.asin(min(1.0, math.sqrt(h)))


def _read_template_ids(template_path: Path) -> dict[str, int | None]:
    if not template_path.is_file():
        raise MissionError(
            f"Lito X1 template KMZ not found: {template_path}. "
            "Create and save a dummy waypoint mission in DJI Fly first."
        )
    try:
        with zipfile.ZipFile(template_path) as archive:
            names = archive.namelist()
            source_name = next(
                (name for name in names if name.endswith("waylines.wpml")),
                next((name for name in names if name.endswith("template.kml")), None),
            )
            if source_name is None:
                raise MissionError("template KMZ has no template.kml or waylines.wpml")
            text = archive.read(source_name).decode("utf-8", "replace")
    except zipfile.BadZipFile as exc:
        raise MissionError("template is not a valid KMZ/ZIP file") from exc

    def read(tag: str) -> int | None:
        found = re.search(rf"<wpml:{tag}>(-?\d+)</wpml:{tag}>", text)
        return int(found.group(1)) if found else None

    def required(tag: str) -> int:
        value = read(tag)
        if value is None:
            raise MissionError(f"template KMZ has no wpml:{tag}")
        return value

    # The consumer DJI Fly / Litchi KMZ for an integrated-camera aircraft (Lito
    # X1) carries no <wpml:payloadInfo>: there is no removable payload to
    # identify. Enterprise templates do. Treat payload as optional and mirror the
    # template — emitting a payloadEnumValue the template never had would be a
    # guess, and the enum table has no consumer entry to guess from.
    return {
        "drone_enum": required("droneEnumValue"),
        "drone_sub_enum": required("droneSubEnumValue"),
        "payload_enum": read("payloadEnumValue"),
    }


def _prepare_waypoints(
    payload: dict[str, Any], limits: dict[str, float]
) -> tuple[list[Waypoint], float, float, list[str]]:
    raw = payload.get("waypoints")
    if not isinstance(raw, list) or len(raw) < 2:
        raise MissionError("waypoints must contain at least two points")
    if len(raw) > int(limits["max_waypoints"]):
        raise MissionError(f"waypoint count exceeds {int(limits['max_waypoints'])}")

    flight = payload.get("flight") if isinstance(payload.get("flight"), dict) else {}
    agl = _number(flight.get("aglM", 30), "flight.aglM")
    if not limits["min_height_m"] <= agl <= limits["max_height_m"]:
        raise MissionError(
            f"flight.aglM must be within {limits['min_height_m']:.0f}-"
            f"{limits['max_height_m']:.0f} m"
        )

    parsed: list[tuple[float, float, float]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise MissionError(f"waypoint {index} must be an object")
        lat = _number(item.get("lat"), f"waypoint {index}.lat")
        lon = _number(item.get("lon"), f"waypoint {index}.lon")
        alt = _number(item.get("alt"), f"waypoint {index}.alt")
        if not -85 <= lat <= 85 or not -180 <= lon <= 180:
            raise MissionError(f"waypoint {index} is outside valid WGS84 bounds")
        parsed.append((lat, lon, alt))

    warnings: list[str] = []
    raw_takeoff = flight.get("takeoffAltMsl")
    if raw_takeoff is None:
        takeoff_alt = parsed[0][2] - agl
        warnings.append(
            "takeoffAltMsl was inferred from waypoint 1 and AGL; confirm the actual "
            "takeoff elevation in DJI Fly"
        )
    else:
        takeoff_alt = _number(raw_takeoff, "flight.takeoffAltMsl")

    points = [
        Waypoint(lat, lon, absolute_alt, absolute_alt - takeoff_alt)
        for lat, lon, absolute_alt in parsed
    ]
    for index, point in enumerate(points):
        if not limits["min_height_m"] <= point.relative_height <= limits["max_height_m"]:
            raise MissionError(
                f"waypoint {index} relative height {point.relative_height:.1f} m is outside "
                f"{limits['min_height_m']:.0f}-{limits['max_height_m']:.0f} m"
            )

    distance = sum(
        _distance_m((a.lat, a.lon), (b.lat, b.lon))
        for a, b in zip(points, points[1:], strict=False)
    )
    if distance > limits["max_route_m"]:
        raise MissionError(
            f"route distance {distance:.0f} m exceeds configured limit "
            f"{limits['max_route_m']:.0f} m"
        )

    if bool(payload.get("loop")):
        # DJI Fly has no infinite-loop contract. Make one inspectable out-and-back
        # mission and let the operator explicitly start another sortie if needed.
        points = points + list(reversed(points[:-1]))
        distance *= 2
        warnings.append(
            "loop was converted to one out-and-back pass; DJI Fly will not repeat indefinitely"
        )
        if len(points) > int(limits["max_waypoints"]):
            raise MissionError("out-and-back conversion exceeds the waypoint limit")

    return points, takeoff_alt, distance, warnings


def _mission_config(ids: dict[str, int | None], speed: float, rth_height: float) -> str:
    payload = ""
    if ids.get("payload_enum") is not None:
        payload = f"""
      <wpml:payloadInfo>
        <wpml:payloadEnumValue>{ids['payload_enum']}</wpml:payloadEnumValue>
        <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
      </wpml:payloadInfo>"""
    return f"""    <wpml:missionConfig>
      <wpml:flyToWaylineMode>safely</wpml:flyToWaylineMode>
      <wpml:finishAction>goHome</wpml:finishAction>
      <wpml:exitOnRCLost>executeLostAction</wpml:exitOnRCLost>
      <wpml:executeRCLostAction>hover</wpml:executeRCLostAction>
      <wpml:takeOffSecurityHeight>{rth_height:.2f}</wpml:takeOffSecurityHeight>
      <wpml:globalTransitionalSpeed>{speed:.2f}</wpml:globalTransitionalSpeed>
      <wpml:globalRTHHeight>{rth_height:.2f}</wpml:globalRTHHeight>
      <wpml:droneInfo>
        <wpml:droneEnumValue>{ids['drone_enum']}</wpml:droneEnumValue>
        <wpml:droneSubEnumValue>{ids['drone_sub_enum']}</wpml:droneSubEnumValue>
      </wpml:droneInfo>{payload}
    </wpml:missionConfig>"""


def _actions(index: int, last: int, gimbal_pitch: float) -> str:
    action_rows = [f"""          <wpml:action>
            <wpml:actionId>0</wpml:actionId>
            <wpml:actionActuatorFunc>gimbalRotate</wpml:actionActuatorFunc>
            <wpml:actionActuatorFuncParam>
              <wpml:gimbalRotateMode>absoluteAngle</wpml:gimbalRotateMode>
              <wpml:gimbalPitchRotateEnable>1</wpml:gimbalPitchRotateEnable>
              <wpml:gimbalPitchRotateAngle>{gimbal_pitch:.2f}</wpml:gimbalPitchRotateAngle>
              <wpml:gimbalRollRotateEnable>0</wpml:gimbalRollRotateEnable>
              <wpml:gimbalYawRotateEnable>0</wpml:gimbalYawRotateEnable>
              <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
            </wpml:actionActuatorFuncParam>
          </wpml:action>"""]
    if index == 0 or index == last:
        func = "startRecord" if index == 0 else "stopRecord"
        action_rows.append(f"""          <wpml:action>
            <wpml:actionId>1</wpml:actionId>
            <wpml:actionActuatorFunc>{func}</wpml:actionActuatorFunc>
            <wpml:actionActuatorFuncParam>
              <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
            </wpml:actionActuatorFuncParam>
          </wpml:action>""")
    return f"""        <wpml:actionGroup>
          <wpml:actionGroupId>{index}</wpml:actionGroupId>
          <wpml:actionGroupStartIndex>{index}</wpml:actionGroupStartIndex>
          <wpml:actionGroupEndIndex>{index}</wpml:actionGroupEndIndex>
          <wpml:actionGroupMode>sequence</wpml:actionGroupMode>
          <wpml:actionTrigger><wpml:actionTriggerType>reachPoint</wpml:actionTriggerType></wpml:actionTrigger>
{chr(10).join(action_rows)}
        </wpml:actionGroup>"""


def _template_kml(
    points: list[Waypoint], ids: dict[str, int | None], speed: float, rth: float, pitch: float
) -> str:
    timestamp = int(time.time() * 1000)
    marks = []
    for index, point in enumerate(points):
        marks.append(f"""      <Placemark>
        <Point><coordinates>{point.lon:.8f},{point.lat:.8f}</coordinates></Point>
        <wpml:index>{index}</wpml:index>
        <wpml:height>{point.relative_height:.2f}</wpml:height>
        <wpml:ellipsoidHeight>{point.absolute_alt_msl:.2f}</wpml:ellipsoidHeight>
        <wpml:useGlobalHeight>0</wpml:useGlobalHeight>
        <wpml:useGlobalSpeed>1</wpml:useGlobalSpeed>
        <wpml:useGlobalHeadingParam>1</wpml:useGlobalHeadingParam>
        <wpml:useGlobalTurnParam>1</wpml:useGlobalTurnParam>
        <wpml:gimbalPitchAngle>{pitch:.2f}</wpml:gimbalPitchAngle>
      </Placemark>""")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="{KML_NS}" xmlns:wpml="{WPML_NS}">
  <Document>
    <wpml:author>SkyLens</wpml:author>
    <wpml:createTime>{timestamp}</wpml:createTime>
    <wpml:updateTime>{timestamp}</wpml:updateTime>
{_mission_config(ids, speed, rth)}
    <Folder>
      <wpml:templateType>waypoint</wpml:templateType>
      <wpml:templateId>0</wpml:templateId>
      <wpml:waylineCoordinateSysParam>
        <wpml:coordinateMode>WGS84</wpml:coordinateMode>
        <wpml:heightMode>relativeToStartPoint</wpml:heightMode>
        <wpml:positioningType>GPS</wpml:positioningType>
      </wpml:waylineCoordinateSysParam>
      <wpml:autoFlightSpeed>{speed:.2f}</wpml:autoFlightSpeed>
      <wpml:gimbalPitchMode>usePointSetting</wpml:gimbalPitchMode>
      <wpml:globalWaypointHeadingParam><wpml:waypointHeadingMode>followWayline</wpml:waypointHeadingMode></wpml:globalWaypointHeadingParam>
      <wpml:globalWaypointTurnMode>toPointAndStopWithContinuityCurvature</wpml:globalWaypointTurnMode>
      <wpml:globalUseStraightLine>1</wpml:globalUseStraightLine>
{chr(10).join(marks)}
    </Folder>
  </Document>
</kml>
"""


def _waylines_wpml(
    points: list[Waypoint], ids: dict[str, int | None], speed: float, rth: float, pitch: float
) -> str:
    marks = []
    last = len(points) - 1
    for index, point in enumerate(points):
        marks.append(f"""      <Placemark>
        <Point><coordinates>{point.lon:.8f},{point.lat:.8f}</coordinates></Point>
        <wpml:index>{index}</wpml:index>
        <wpml:executeHeight>{point.relative_height:.2f}</wpml:executeHeight>
        <wpml:waypointSpeed>{speed:.2f}</wpml:waypointSpeed>
        <wpml:waypointHeadingParam><wpml:waypointHeadingMode>followWayline</wpml:waypointHeadingMode></wpml:waypointHeadingParam>
        <wpml:waypointTurnParam>
          <wpml:waypointTurnMode>toPointAndStopWithContinuityCurvature</wpml:waypointTurnMode>
          <wpml:waypointTurnDampingDist>0.20</wpml:waypointTurnDampingDist>
        </wpml:waypointTurnParam>
        <wpml:useStraightLine>1</wpml:useStraightLine>
{_actions(index, last, pitch)}
      </Placemark>""")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="{KML_NS}" xmlns:wpml="{WPML_NS}">
  <Document>
{_mission_config(ids, speed, rth)}
    <Folder>
      <wpml:templateId>0</wpml:templateId>
      <wpml:waylineId>0</wpml:waylineId>
      <wpml:autoFlightSpeed>{speed:.2f}</wpml:autoFlightSpeed>
      <wpml:executeHeightMode>relativeToStartPoint</wpml:executeHeightMode>
{chr(10).join(marks)}
    </Folder>
  </Document>
</kml>
"""


def build_mission(
    payload: dict[str, Any],
    template_path: Path,
    output_path: Path,
    limits: dict[str, float] | None = None,
) -> MissionResult:
    """Validate a SkyLens route and write a Lito-identified DJI Fly KMZ."""
    configured = {
        "min_height_m": 5.0,
        "max_height_m": 120.0,
        "max_route_m": 5000.0,
        "max_waypoints": 200.0,
    }
    if limits:
        configured.update(limits)
    ids = _read_template_ids(template_path)
    points, takeoff_alt, distance, warnings = _prepare_waypoints(payload, configured)
    flight = payload.get("flight") if isinstance(payload.get("flight"), dict) else {}
    speed = _number(flight.get("speedMps", 2.0), "flight.speedMps")
    pitch = _number(flight.get("gimbalPitchDeg", -30.0), "flight.gimbalPitchDeg")
    rth = _number(flight.get("rthHeightM", 30.0), "flight.rthHeightM")
    if not 0.5 <= speed <= 10:
        raise MissionError("flight.speedMps must be within 0.5-10 m/s")
    if not -90 <= pitch <= 30:
        raise MissionError("flight.gimbalPitchDeg must be within -90 to 30 degrees")
    if not configured["min_height_m"] <= rth <= configured["max_height_m"]:
        raise MissionError("flight.rthHeightM is outside the configured height limits")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("wpmz/template.kml", _template_kml(points, ids, speed, rth, pitch))
        archive.writestr("wpmz/waylines.wpml", _waylines_wpml(points, ids, speed, rth, pitch))
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return MissionResult(
        path=output_path,
        sha256=digest,
        waypoint_count=len(points),
        route_distance_m=distance,
        takeoff_alt_msl=takeoff_alt,
        warnings=tuple(warnings),
    )
