"""Generate DJI waypoint routes (KMZ/WPML) for repeatable capture geometry.

One drone cannot fly three formation legs at once, so it flies them one after
another. Doing that by hand is what produced the capture geometry measured in
``RESULTS.md``: triangle side ratios of 2.0-7.8x instead of equal, and camera
pitch of -2.3 / +14.3 / +1.7 degrees when the intent was to look down. This tool
removes that variance by writing the legs out as waypoint missions.

What a route file can and cannot do:

* CAN fix position, altitude, heading, gimbal pitch, speed and camera actions,
  so the same geometry is repeatable across flights.
* CANNOT supply camera pose for 3DGS. Experiment 3 measured a 2.24 dB loss from
  0.2% position error; GNSS horizontal accuracy is 0.5-1.5 m, far coarser. Poses
  still come from COLMAP.
* CANNOT fly indoors. Waypoints need satellite positioning.
* CANNOT make the legs simultaneous. Anything that moves between legs (people,
  fire) breaks the static-scene assumption 3DGS rests on.

Geometry. The three legs start at the vertices of an equilateral triangle lying
in the horizontal plane and translate together along ``--heading``, sweeping a
prism over the ground below. Lateral separation across the legs is what creates
parallax, and parallax is what resolves depth — see README section 4.2.

File format. A KMZ is a zip containing ``wpmz/template.kml`` (editable business
layer) and ``wpmz/waylines.wpml`` (execution layer). Both are written here so
the file works whether the app regenerates the wayline or executes it directly.

Aircraft identity, and the reason ``--from-existing`` is required. Checked
against DJI's published WPML reference (dji-sdk/Cloud-API-Doc), the spec covers
five enterprise aircraft and nothing else::

    89  M350 RTK      60  M300 RTK      67  M30/M30T
    77  M3E/M3T/M3M   91  M3D/M3TD

Every page names DJI Pilot 2 and FlightHub 2 as the consuming apps; DJI Fly is
never mentioned, and no consumer model appears in the enum table. So the Lito
X1's ``droneEnumValue`` cannot be looked up — it has to be lifted from a route
the app itself wrote. Point ``--from-existing`` at one. The built-in defaults
are the documented M30 values, kept only so the writer has something valid to
emit, and they will not match a consumer aircraft.

By the same token the rest of this file follows the enterprise dialect. DJI Fly
writes KMZ too, but its variant is undocumented, which is why third-party tools
had to reverse-engineer it. Treat a generated route as untested until it loads
and flies correctly.

Loading. DJI Fly has no import button. Create a placeholder route in the app,
find its KMZ on the controller, and overwrite it with the generated file.
"""

from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path

WPML_NS = "http://www.dji.com/wpmz/1.0.2"
KML_NS = "http://www.opengis.net/kml/2.2"

# Metres per degree of latitude. Longitude is scaled by cos(latitude) at use.
# Flat-earth approximation; error stays well under a centimetre over the
# hundred-metre scale these routes cover.
M_PER_DEG_LAT = 111320.0

DEFAULT_DRONE_ENUM = 67
DEFAULT_DRONE_SUB_ENUM = 0
DEFAULT_PAYLOAD_ENUM = 52


def offset_latlon(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    """Shift a coordinate by a metre offset in the east/north directions."""
    dlat = north_m / M_PER_DEG_LAT
    dlon = east_m / (M_PER_DEG_LAT * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def triangle_vertices(side_m: float, heading_deg: float) -> list[tuple[float, float]]:
    """Equilateral triangle vertices as (east, north) offsets from its centre.

    Vertex 0 leads along ``heading``; the other two trail, spread laterally.
    Returned in metres, already rotated into the heading frame.
    """
    radius = side_m / math.sqrt(3.0)
    out = []
    for k in range(3):
        # 90 deg puts vertex 0 ahead of centre before the heading rotation.
        theta = math.radians(90.0 + 120.0 * k)
        fwd = radius * math.sin(theta)
        lat_off = radius * math.cos(theta)
        # Rotate (forward, lateral) into (east, north) for this heading.
        h = math.radians(heading_deg)
        east = fwd * math.sin(h) + lat_off * math.cos(h)
        north = fwd * math.cos(h) - lat_off * math.sin(h)
        out.append((east, north))
    return out


def leg_waypoints(
    center_lat: float,
    center_lon: float,
    vertex: tuple[float, float],
    heading_deg: float,
    length_m: float,
    count: int,
) -> list[tuple[float, float]]:
    """Straight leg from one vertex, ``count`` evenly spaced points."""
    east0, north0 = vertex
    h = math.radians(heading_deg)
    points = []
    for i in range(count):
        travelled = length_m * i / max(1, count - 1)
        east = east0 + travelled * math.sin(h)
        north = north0 + travelled * math.cos(h)
        points.append(offset_latlon(center_lat, center_lon, east, north))
    return points


def _mission_config(cfg: dict) -> str:
    return f"""    <wpml:missionConfig>
      <wpml:flyToWaylineMode>safely</wpml:flyToWaylineMode>
      <wpml:finishAction>goHome</wpml:finishAction>
      <wpml:exitOnRCLost>executeLostAction</wpml:exitOnRCLost>
      <wpml:executeRCLostAction>hover</wpml:executeRCLostAction>
      <wpml:takeOffSecurityHeight>{cfg["takeoff_height"]}</wpml:takeOffSecurityHeight>
      <wpml:globalTransitionalSpeed>{cfg["speed"]}</wpml:globalTransitionalSpeed>
      <wpml:globalRTHHeight>{cfg["rth_height"]}</wpml:globalRTHHeight>
      <wpml:droneInfo>
        <wpml:droneEnumValue>{cfg["drone_enum"]}</wpml:droneEnumValue>
        <wpml:droneSubEnumValue>{cfg["drone_sub_enum"]}</wpml:droneSubEnumValue>
      </wpml:droneInfo>
      <wpml:payloadInfo>
        <wpml:payloadEnumValue>{cfg["payload_enum"]}</wpml:payloadEnumValue>
        <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
      </wpml:payloadInfo>
    </wpml:missionConfig>"""


def build_template_kml(points: list[tuple[float, float]], cfg: dict) -> str:
    """The editable business layer the app reads."""
    marks = []
    for i, (lat, lon) in enumerate(points):
        marks.append(f"""      <Placemark>
        <Point>
          <coordinates>{lon:.8f},{lat:.8f}</coordinates>
        </Point>
        <wpml:index>{i}</wpml:index>
        <wpml:height>{cfg["altitude"]}</wpml:height>
        <wpml:ellipsoidHeight>{cfg["altitude"]}</wpml:ellipsoidHeight>
        <wpml:useGlobalHeight>1</wpml:useGlobalHeight>
        <wpml:useGlobalSpeed>1</wpml:useGlobalSpeed>
        <wpml:useGlobalHeadingParam>1</wpml:useGlobalHeadingParam>
        <wpml:useGlobalTurnParam>1</wpml:useGlobalTurnParam>
        <wpml:gimbalPitchAngle>{cfg["gimbal_pitch"]}</wpml:gimbalPitchAngle>
      </Placemark>""")

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="{KML_NS}" xmlns:wpml="{WPML_NS}">
  <Document>
    <wpml:author>skylens</wpml:author>
    <wpml:createTime>{cfg["timestamp_ms"]}</wpml:createTime>
    <wpml:updateTime>{cfg["timestamp_ms"]}</wpml:updateTime>
{_mission_config(cfg)}
    <Folder>
      <wpml:templateType>waypoint</wpml:templateType>
      <wpml:templateId>0</wpml:templateId>
      <wpml:waylineCoordinateSysParam>
        <wpml:coordinateMode>WGS84</wpml:coordinateMode>
        <wpml:heightMode>relativeToStartPoint</wpml:heightMode>
        <wpml:positioningType>GPS</wpml:positioningType>
      </wpml:waylineCoordinateSysParam>
      <wpml:autoFlightSpeed>{cfg["speed"]}</wpml:autoFlightSpeed>
      <wpml:gimbalPitchMode>usePointSetting</wpml:gimbalPitchMode>
      <wpml:globalWaypointHeadingParam>
        <wpml:waypointHeadingMode>followWayline</wpml:waypointHeadingMode>
      </wpml:globalWaypointHeadingParam>
      <wpml:globalWaypointTurnMode>toPointAndStopWithContinuityCurvature</wpml:globalWaypointTurnMode>
      <wpml:globalUseStraightLine>1</wpml:globalUseStraightLine>
{chr(10).join(marks)}
    </Folder>
  </Document>
</kml>
"""


def _gimbal_action(action_id: int, pitch: float) -> str:
    return f"""          <wpml:action>
            <wpml:actionId>{action_id}</wpml:actionId>
            <wpml:actionActuatorFunc>gimbalRotate</wpml:actionActuatorFunc>
            <wpml:actionActuatorFuncParam>
              <wpml:gimbalRotateMode>absoluteAngle</wpml:gimbalRotateMode>
              <wpml:gimbalPitchRotateEnable>1</wpml:gimbalPitchRotateEnable>
              <wpml:gimbalPitchRotateAngle>{pitch}</wpml:gimbalPitchRotateAngle>
              <wpml:gimbalRollRotateEnable>0</wpml:gimbalRollRotateEnable>
              <wpml:gimbalRollRotateAngle>0</wpml:gimbalRollRotateAngle>
              <wpml:gimbalYawRotateEnable>0</wpml:gimbalYawRotateEnable>
              <wpml:gimbalYawRotateAngle>0</wpml:gimbalYawRotateAngle>
              <wpml:gimbalRotateTimeEnable>0</wpml:gimbalRotateTimeEnable>
              <wpml:gimbalRotateTime>0</wpml:gimbalRotateTime>
              <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
            </wpml:actionActuatorFuncParam>
          </wpml:action>"""


def _record_action(action_id: int, func: str) -> str:
    return f"""          <wpml:action>
            <wpml:actionId>{action_id}</wpml:actionId>
            <wpml:actionActuatorFunc>{func}</wpml:actionActuatorFunc>
            <wpml:actionActuatorFuncParam>
              <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
            </wpml:actionActuatorFuncParam>
          </wpml:action>"""


def _action_group(group_id: int, index: int, actions: list[str]) -> str:
    return f"""        <wpml:actionGroup>
          <wpml:actionGroupId>{group_id}</wpml:actionGroupId>
          <wpml:actionGroupStartIndex>{index}</wpml:actionGroupStartIndex>
          <wpml:actionGroupEndIndex>{index}</wpml:actionGroupEndIndex>
          <wpml:actionGroupMode>sequence</wpml:actionGroupMode>
          <wpml:actionTrigger>
            <wpml:actionTriggerType>reachPoint</wpml:actionTriggerType>
          </wpml:actionTrigger>
{chr(10).join(actions)}
        </wpml:actionGroup>"""


def build_waylines_wpml(points: list[tuple[float, float]], cfg: dict) -> str:
    """The execution layer. Gimbal is set at every point; recording brackets the leg."""
    last = len(points) - 1
    marks = []
    for i, (lat, lon) in enumerate(points):
        actions = [_gimbal_action(0, cfg["gimbal_pitch"])]
        if i == 0:
            actions.append(_record_action(1, "startRecord"))
        elif i == last:
            actions.append(_record_action(1, "stopRecord"))

        marks.append(f"""      <Placemark>
        <Point>
          <coordinates>{lon:.8f},{lat:.8f}</coordinates>
        </Point>
        <wpml:index>{i}</wpml:index>
        <wpml:executeHeight>{cfg["altitude"]}</wpml:executeHeight>
        <wpml:waypointSpeed>{cfg["speed"]}</wpml:waypointSpeed>
        <wpml:waypointHeadingParam>
          <wpml:waypointHeadingMode>followWayline</wpml:waypointHeadingMode>
        </wpml:waypointHeadingParam>
        <wpml:waypointTurnParam>
          <wpml:waypointTurnMode>toPointAndStopWithContinuityCurvature</wpml:waypointTurnMode>
          <wpml:waypointTurnDampingDist>0.2</wpml:waypointTurnDampingDist>
        </wpml:waypointTurnParam>
        <wpml:useStraightLine>1</wpml:useStraightLine>
{_action_group(i, i, actions)}
      </Placemark>""")

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="{KML_NS}" xmlns:wpml="{WPML_NS}">
  <Document>
{_mission_config(cfg)}
    <Folder>
      <wpml:templateId>0</wpml:templateId>
      <wpml:waylineId>0</wpml:waylineId>
      <wpml:autoFlightSpeed>{cfg["speed"]}</wpml:autoFlightSpeed>
      <wpml:executeHeightMode>relativeToStartPoint</wpml:executeHeightMode>
{chr(10).join(marks)}
    </Folder>
  </Document>
</kml>
"""


def write_kmz(path: Path, template: str, waylines: str) -> None:
    """Pack both layers into the wpmz/ folder a KMZ expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("wpmz/template.kml", template)
        z.writestr("wpmz/waylines.wpml", waylines)


def read_aircraft_ids(kmz_path: str | Path) -> dict:
    """Lift droneInfo / payloadInfo out of a KMZ the DJI app produced."""
    import re

    with zipfile.ZipFile(kmz_path) as z:
        name = next((n for n in z.namelist() if n.endswith(("template.kml", "waylines.wpml"))), None)
        if name is None:
            raise ValueError(f"{kmz_path} has no template.kml or waylines.wpml")
        text = z.read(name).decode("utf-8", "replace")

    def grab(tag: str, default: int) -> int:
        found = re.search(rf"<wpml:{tag}>(-?\d+)</wpml:{tag}>", text)
        return int(found.group(1)) if found else default

    ids = {
        "drone_enum": grab("droneEnumValue", DEFAULT_DRONE_ENUM),
        "drone_sub_enum": grab("droneSubEnumValue", DEFAULT_DRONE_SUB_ENUM),
        "payload_enum": grab("payloadEnumValue", DEFAULT_PAYLOAD_ENUM),
    }
    print(f"[aircraft] read from {Path(kmz_path).name}: {ids}")
    return ids


def generate(args) -> list[Path]:
    center_lat, center_lon = (float(v) for v in args.center.split(","))

    cfg = {
        "altitude": args.altitude,
        "speed": args.speed,
        "gimbal_pitch": args.gimbal_pitch,
        "takeoff_height": args.takeoff_height,
        "rth_height": args.rth_height,
        "timestamp_ms": args.timestamp_ms,
        "drone_enum": DEFAULT_DRONE_ENUM,
        "drone_sub_enum": DEFAULT_DRONE_SUB_ENUM,
        "payload_enum": DEFAULT_PAYLOAD_ENUM,
    }
    if args.from_existing:
        cfg.update(read_aircraft_ids(args.from_existing))
    else:
        print("[aircraft] WARNING: writing documented M30 values (drone 67, payload 52).")
        print("           DJI publishes enum values for enterprise aircraft only, so there")
        print("           is no correct value to use for a consumer model. Re-run with")
        print("           --from-existing <app-made route.kmz> before flying this.")

    vertices = triangle_vertices(args.spacing, args.heading)
    out_dir = Path(args.out)
    written = []

    print(f"\n[formation] equilateral triangle, side {args.spacing} m, "
          f"heading {args.heading} deg, leg length {args.length} m")
    for name, vertex in zip("ABC", vertices, strict=True):
        points = leg_waypoints(
            center_lat, center_lon, vertex, args.heading, args.length, args.waypoints
        )
        path = out_dir / f"leg_{name}.kmz"
        write_kmz(
            path,
            build_template_kml(points, cfg),
            build_waylines_wpml(points, cfg),
        )
        east, north = vertex
        print(f"  leg {name}  start offset E{east:+6.2f} N{north:+6.2f} m  "
              f"{len(points)} waypoints  ->  {path}")
        written.append(path)

    lateral = max(abs(e) for e, _ in vertices) + max(abs(n) for _, n in vertices)
    print(f"\n[geometry] gimbal pitch {args.gimbal_pitch} deg, altitude {args.altitude} m")
    print(f"           max vertex separation ~{lateral:.2f} m — this is the parallax budget")
    print("\n[load] DJI Fly has no import. Make a placeholder route in the app, locate its")
    print("       KMZ on the controller, and overwrite it with one of these. One leg per flight.")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate three DJI waypoint legs for repeatable capture geometry."
    )
    parser.add_argument("--center", required=True, help="formation centre as 'lat,lon'")
    parser.add_argument("--heading", type=float, required=True, help="travel direction, deg from north")
    parser.add_argument("--length", type=float, default=40.0, help="leg length in metres")
    parser.add_argument("--spacing", type=float, default=3.0, help="triangle side in metres")
    parser.add_argument("--altitude", type=float, default=5.0, help="height above start point, m")
    parser.add_argument("--gimbal-pitch", type=float, default=-30.0, help="camera pitch, deg (down is negative)")
    parser.add_argument("--speed", type=float, default=1.5, help="flight speed, m/s")
    parser.add_argument("--waypoints", type=int, default=5, help="waypoints per leg")
    parser.add_argument("--takeoff-height", type=float, default=5.0)
    parser.add_argument("--rth-height", type=float, default=30.0)
    parser.add_argument("--timestamp-ms", type=int, default=0, help="createTime/updateTime value")
    parser.add_argument("--from-existing", help="KMZ made by DJI Fly, to copy aircraft ids from")
    parser.add_argument("--out", default="routes", help="output directory")
    args = parser.parse_args()

    if args.waypoints < 2:
        parser.error("--waypoints must be at least 2")
    if not -90.0 <= args.gimbal_pitch <= 30.0:
        parser.error("--gimbal-pitch outside the plausible range [-90, 30]")

    generate(args)


if __name__ == "__main__":
    main()


__all__ = [
    "build_template_kml",
    "build_waylines_wpml",
    "leg_waypoints",
    "offset_latlon",
    "read_aircraft_ids",
    "triangle_vertices",
    "write_kmz",
]
