from __future__ import annotations

import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from skylens_mission_bridge.mission import MissionError, build_mission


def write_lito_template(path: Path) -> None:
    xml = """<?xml version="1.0"?>
<kml xmlns:wpml="http://www.dji.com/wpmz/1.0.2">
  <wpml:droneEnumValue>123</wpml:droneEnumValue>
  <wpml:droneSubEnumValue>7</wpml:droneSubEnumValue>
  <wpml:payloadEnumValue>456</wpml:payloadEnumValue>
</kml>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("wpmz/template.kml", xml)


def payload(loop: bool = False) -> dict:
    return {
        "name": "test-route",
        "droneId": 1,
        "loop": loop,
        "flight": {
            "aglM": 30,
            "takeoffAltMsl": 50,
            "speedMps": 2,
            "gimbalPitchDeg": -35,
            "rthHeightM": 40,
        },
        "waypoints": [
            {"lat": 36.3685, "lon": 127.3475, "alt": 80},
            {"lat": 36.3687, "lon": 127.3477, "alt": 85},
        ],
    }


class MissionBridgeTest(TestCase):
    def test_builds_dji_kmz_with_template_aircraft_ids(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            template = tmp_path / "lito-template.kmz"
            output = tmp_path / "route.kmz"
            write_lito_template(template)

            result = build_mission(payload(), template, output)

            self.assertEqual(result.waypoint_count, 2)
            self.assertEqual(result.takeoff_alt_msl, 50)
            self.assertTrue(result.sha256)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(
                    set(archive.namelist()), {"wpmz/template.kml", "wpmz/waylines.wpml"}
                )
                waylines = archive.read("wpmz/waylines.wpml").decode()
            self.assertIn("<wpml:droneEnumValue>123</wpml:droneEnumValue>", waylines)
            self.assertIn("<wpml:payloadEnumValue>456</wpml:payloadEnumValue>", waylines)
            self.assertIn("<wpml:executeHeight>30.00</wpml:executeHeight>", waylines)
            self.assertIn("<wpml:executeHeight>35.00</wpml:executeHeight>", waylines)
            self.assertIn("<wpml:actionActuatorFunc>startRecord</wpml:actionActuatorFunc>", waylines)
            self.assertIn("<wpml:actionActuatorFunc>stopRecord</wpml:actionActuatorFunc>", waylines)

    def test_loop_becomes_one_out_and_back_pass(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            template = tmp_path / "lito-template.kmz"
            output = tmp_path / "route.kmz"
            write_lito_template(template)

            result = build_mission(payload(loop=True), template, output)

            self.assertEqual(result.waypoint_count, 3)
            self.assertTrue(any("out-and-back" in warning for warning in result.warnings))
            with zipfile.ZipFile(output) as archive:
                waylines = archive.read("wpmz/waylines.wpml").decode()
            self.assertEqual(waylines.count("<wpml:index>"), 3)

    def test_refuses_generation_without_real_template(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            with self.assertRaisesRegex(MissionError, "template KMZ not found"):
                build_mission(payload(), tmp_path / "missing.kmz", tmp_path / "route.kmz")

    def test_rejects_height_outside_safety_limits(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            template = tmp_path / "lito-template.kmz"
            write_lito_template(template)
            bad = payload()
            bad["waypoints"][1]["alt"] = 250

            with self.assertRaisesRegex(MissionError, "relative height"):
                build_mission(bad, template, tmp_path / "route.kmz")
