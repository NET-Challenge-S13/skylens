"""Offline unit tests for the DJI cloud delivery module (no network)."""

from __future__ import annotations

import io
import json
import zipfile

from skylens_mission_bridge.cloud import (
    CloudConfig,
    sign_request,
    wrap_mission_package,
)


def _raw_kmz() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("wpmz/template.kml", "<kml/>")
        archive.writestr(
            "wpmz/waylines.wpml",
            "<kml><coordinates>127.1,36.1</coordinates>"
            "<coordinates>127.2,36.2</coordinates></kml>",
        )
    return buffer.getvalue()


def test_sign_request_matches_captured_vector() -> None:
    # Captured from the live service: this exact canonical string + uav key
    # produced this signature. Pins the algorithm so a refactor cannot drift.
    cfg = CloudConfig(token="t", wk_key="d62db06b88a3efe3eb62a50cf5a25569", secret_id="uav")
    sign = sign_request(
        "/api/waypoint/mission/compare",
        nonce=1309760617,
        timestamp_ms=1790707096180,
        cfg=cfg,
    )
    assert sign == "HgkYpTnkrCcsWg8INZ3pe2N+0aVfisOXtOSZSl23Ex8="


def test_wrap_raw_kmz_into_cloud_package() -> None:
    raw = _raw_kmz()
    package = wrap_mission_package(raw, "mission-uuid-1")
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        names = set(archive.namelist())
        assert "mission-uuid-1.kmz" in names
        assert "image/ShotSnap.json" in names
        # inner kmz must be the untouched raw mission
        assert archive.read("mission-uuid-1.kmz") == raw
        shot = json.loads(archive.read("image/ShotSnap.json"))
        assert shot == {"WAY_POINT": {}, "POI_POINT": {}}


def test_wrap_is_idempotent_for_already_wrapped() -> None:
    raw = _raw_kmz()
    package = wrap_mission_package(raw, "mission-uuid-1")
    again = wrap_mission_package(package, "mission-uuid-2")
    assert again == package  # already a wrapper, returned unchanged


def test_config_from_env_requires_both_secrets(monkeypatch) -> None:
    monkeypatch.delenv("SKYLENS_DJI_MC_TOKEN", raising=False)
    monkeypatch.delenv("SKYLENS_DJI_WK_KEY", raising=False)
    assert CloudConfig.from_env() is None

    monkeypatch.setenv("SKYLENS_DJI_MC_TOKEN", "token")
    assert CloudConfig.from_env() is None  # key still missing

    monkeypatch.setenv("SKYLENS_DJI_WK_KEY", "key")
    cfg = CloudConfig.from_env()
    assert cfg is not None
    assert cfg.token == "token"
    assert cfg.secret_id == "uav"
