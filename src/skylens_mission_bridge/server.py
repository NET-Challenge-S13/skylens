"""Dependency-free localhost HTTP bridge for SkyLens -> DJI Fly missions."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from cloud import CloudConfig, CloudError, delete_mission, list_missions, upload_mission_file
from formation import build_formation
from mission import MissionError, build_mission

ROOT = Path(__file__).resolve().parents[2]
HOST = os.getenv("SKYLENS_MISSION_HOST", "127.0.0.1")
PORT = int(os.getenv("SKYLENS_MISSION_PORT", "8091"))
TEMPLATE = Path(
    os.getenv("SKYLENS_DJI_TEMPLATE_KMZ", ROOT / "config/lito-x1-template.kmz")
).expanduser()
OUTPUT = Path(os.getenv("SKYLENS_MISSION_OUTPUT", ROOT / "output/missions")).expanduser()
INSTALL_TARGET = os.getenv("SKYLENS_DJI_INSTALL_TARGET", "").strip()
CLOUD = CloudConfig.from_env()
ALLOWED_ORIGINS = {
    value.strip()
    for value in os.getenv(
        "SKYLENS_MISSION_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:8080,http://127.0.0.1:8080",
    ).split(",")
    if value.strip()
}
MAX_BODY = 1024 * 1024


def _safe_name(value: Any) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "route")).strip("-._")
    return clean[:60] or "route"


def _wants_cloud(payload: dict[str, Any]) -> bool:
    return str(payload.get("deliver", "")).strip().lower() == "cloud"


def _require_cloud() -> CloudConfig:
    if CLOUD is None:
        raise MissionError(
            "cloud delivery requested but not configured: set SKYLENS_DJI_MC_TOKEN and "
            "SKYLENS_DJI_WK_KEY"
        )
    return CLOUD


def _install(source: Path) -> tuple[bool, str | None]:
    if not INSTALL_TARGET:
        return False, None
    target = Path(INSTALL_TARGET).expanduser()
    if target.suffix.lower() != ".kmz":
        raise MissionError("SKYLENS_DJI_INSTALL_TARGET must name one explicit .kmz file")
    if not target.parent.is_dir():
        raise MissionError(f"install target directory is not available: {target.parent}")
    if target.exists():
        backup_dir = OUTPUT / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"{target.stem}-{int(time.time())}.kmz"
        shutil.copy2(target, backup)
    shutil.copy2(source, target)
    return True, str(target)


class Handler(BaseHTTPRequestHandler):
    server_version = "SkyLensMissionBridge/0.1"

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or origin in ALLOWED_ORIGINS

    def _cors(self) -> None:
        origin = self.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        if not self._origin_allowed():
            self._json(HTTPStatus.FORBIDDEN, {"ok": False, "error": "origin not allowed"})
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "templateReady": TEMPLATE.is_file(),
                    "template": str(TEMPLATE),
                    "output": str(OUTPUT),
                    "installConfigured": bool(INSTALL_TARGET),
                    "cloudConfigured": CLOUD is not None,
                },
            )
            return
        if parsed.path == "/cloud/missions":
            if CLOUD is None:
                self._json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"ok": False, "error": "cloud not configured"},
                )
                return
            try:
                items = list_missions(CLOUD)
            except CloudError as exc:
                self._json(HTTPStatus.BAD_GATEWAY, {"ok": False, "error": str(exc)})
                return
            self._json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "missions": [
                        {
                            "uuid": it.get("uuid"),
                            "name": it.get("name"),
                            "waypointCount": it.get("point_num"),
                            "distanceM": it.get("distance"),
                        }
                        for it in items
                    ],
                },
            )
            return
        prefix = "/missions/"
        if parsed.path.startswith(prefix):
            name = Path(unquote(parsed.path[len(prefix) :])).name
            path = OUTPUT / name
            if path.parent != OUTPUT or not path.is_file() or path.suffix.lower() != ".kmz":
                self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "mission not found"})
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self._cors()
            self.send_header("Content-Type", "application/vnd.google-earth.kmz")
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/missions", "/formation", "/cloud/delete"):
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        if not self._origin_allowed():
            self._json(HTTPStatus.FORBIDDEN, {"ok": False, "error": "origin not allowed"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY:
                raise MissionError("request body is empty or too large")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise MissionError("request body must be a JSON object")
            if self.path == "/cloud/delete":
                self._handle_cloud_delete(payload)
                return
            name = _safe_name(payload.get("name"))
            stamp = time.strftime("%Y%m%d-%H%M%S")
            if self.path == "/formation":
                self._handle_formation(payload, name, stamp)
            else:
                self._handle_single(payload, name, stamp)
        except (MissionError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
        except CloudError as exc:  # DJI cloud rejected or unreachable
            self._json(HTTPStatus.BAD_GATEWAY, {"ok": False, "error": str(exc)})
        except Exception as exc:  # keep localhost service alive, but surface the failure
            print(f"[mission-bridge] unexpected error: {exc}", file=sys.stderr)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": "internal error"})

    def _handle_cloud_delete(self, payload: dict[str, Any]) -> None:
        mission_uuid = str(payload.get("missionUuid") or payload.get("uuid") or "").strip()
        if not mission_uuid:
            raise MissionError("missionUuid is required")
        removed = delete_mission(_require_cloud(), mission_uuid)
        self._json(HTTPStatus.OK, {"ok": True, "missionUuid": mission_uuid, "removed": removed})

    def _handle_single(self, payload: dict[str, Any], name: str, stamp: str) -> None:
        output_path = OUTPUT / f"{stamp}-{name}.kmz"
        result = build_mission(payload, TEMPLATE, output_path)
        installed, target = _install(result.path)
        body = {
            "ok": True,
            "fileName": result.path.name,
            "downloadUrl": f"http://{HOST}:{PORT}/missions/{result.path.name}",
            "sha256": result.sha256,
            "waypointCount": result.waypoint_count,
            "routeDistanceM": round(result.route_distance_m, 1),
            "takeoffAltMsl": round(result.takeoff_alt_msl, 1),
            "installed": installed,
            "installTarget": target,
            "warnings": list(result.warnings),
        }
        if _wants_cloud(payload):
            cloud = upload_mission_file(
                _require_cloud(), result.path, result.path.stem, result.route_distance_m
            )
            body["cloud"] = {"missionUuid": cloud.mission_uuid, "verified": cloud.verified}
        self._json(HTTPStatus.CREATED, body)

    def _handle_formation(self, payload: dict[str, Any], name: str, stamp: str) -> None:
        # Three files to three aircraft: auto-install (one configured target)
        # cannot mean three things, so a formation is always download-only.
        members = build_formation(payload, TEMPLATE, OUTPUT, name, stamp)
        want_cloud = _wants_cloud(payload)
        cfg = _require_cloud() if want_cloud else None
        rows = []
        for station, result in members:
            row = {
                "station": station,
                "fileName": result.path.name,
                "downloadUrl": f"http://{HOST}:{PORT}/missions/{result.path.name}",
                "sha256": result.sha256,
                "waypointCount": result.waypoint_count,
                "routeDistanceM": round(result.route_distance_m, 1),
                "takeoffAltMsl": round(result.takeoff_alt_msl, 1),
                "warnings": list(result.warnings),
            }
            if cfg is not None:
                cloud = upload_mission_file(
                    cfg, result.path, result.path.stem, result.route_distance_m
                )
                row["cloud"] = {"missionUuid": cloud.mission_uuid, "verified": cloud.verified}
            rows.append(row)
        self._json(HTTPStatus.CREATED, {"ok": True, "installed": False, "members": rows})

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[mission-bridge] {self.address_string()} {fmt % args}")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[mission-bridge] listening on http://{HOST}:{PORT}")
    print(
        f"[mission-bridge] Lito template: {TEMPLATE} "
        f"({'ready' if TEMPLATE.is_file() else 'MISSING'})"
    )
    print(f"[mission-bridge] output: {OUTPUT}")
    if INSTALL_TARGET:
        print(f"[mission-bridge] install target: {INSTALL_TARGET}")
    else:
        print("[mission-bridge] install target: not configured (download-only mode)")
    if CLOUD is not None:
        print("[mission-bridge] DJI cloud: configured (deliver:'cloud' enabled)")
    else:
        print(
            "[mission-bridge] DJI cloud: not configured "
            "(set SKYLENS_DJI_MC_TOKEN + SKYLENS_DJI_WK_KEY)"
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
