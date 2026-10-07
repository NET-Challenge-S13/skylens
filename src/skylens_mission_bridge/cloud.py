"""Deliver a SkyLens KMZ mission to the DJI Fly consumer cloud.

DJI Fly syncs waypoint missions through an undocumented account-scoped cloud so
an RC 2 can pull them over the air. This module speaks that protocol with the
Python standard library only (no requests, no boto3), matching the bridge's
dependency-free rule: urllib for HTTPS, hmac/hashlib for both the DJI request
signature and the AWS SigV4 upload.

Secrets never live in source. The account token and the signing key are read
from the environment; without them the cloud path stays disabled and the bridge
keeps working in download / local-install mode.

Flow (verified against the live service):
  1. POST /api/waypoint/mission/compare   announce the mission uuid
  2. POST /api/waypoint/file/upload/sts/init   get AWS STS creds per file
  3. PUT  s3://<bucket>/<object_key>   upload mission_file + mission_cover
The cloud mission is created by the two S3 objects; no extra register call.

The cloud "mission file" is NOT the raw wpmz KMZ. It is a wrapper zip holding
<mission_uuid>.kmz (the raw wpmz KMZ) plus image/ShotSnap.json. Uploading a raw
KMZ makes DJI Fly refuse to open it.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import io
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_URL = "https://flyapi.djigate.com"

# 1x1 PNG placeholder cover. A mission needs a mission_cover object to bind; a
# real waypoint thumbnail can be passed to upload_mission() instead.
_PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


# Short cache for URL-sourced tokens so each request does not re-fetch: {url: (ts, value)}.
_TOKEN_URL_CACHE: dict[str, tuple[float, str]] = {}
_TOKEN_URL_TTL = 15.0


def _fetch_token_url(url: str, force: bool = False) -> str:
    cached = _TOKEN_URL_CACHE.get(url)
    if cached and not force and (time.time() - cached[0]) < _TOKEN_URL_TTL:
        return cached[1]
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            value = resp.read().decode("utf-8", "replace").strip()
    except (urllib.error.URLError, OSError):
        return cached[1] if cached else ""
    _TOKEN_URL_CACHE[url] = (time.time(), value)
    return value


class CloudError(RuntimeError):
    """The DJI cloud rejected a request or is not configured."""


class CloudAuthError(CloudError):
    """The DJI cloud rejected the request for an auth/token reason (retryable)."""


@dataclass(frozen=True)
class CloudConfig:
    """DJI cloud credentials and signing parameters, all from the environment.

    The account token expires. ``token_file`` lets an external refresher keep a
    fresh token on disk; it is re-read on every request so a refresh takes effect
    without restarting the bridge. ``token`` is the static fallback.
    """

    token: str
    wk_key: str
    secret_id: str = "uav"
    app_info: str = "Android-SkyLens-13-KR"
    token_file: str | None = None
    token_url: str | None = None

    @classmethod
    def from_env(cls) -> CloudConfig | None:
        token = os.getenv("SKYLENS_DJI_MC_TOKEN", "").strip()
        wk_key = os.getenv("SKYLENS_DJI_WK_KEY", "").strip()
        token_file = os.getenv("SKYLENS_DJI_TOKEN_FILE", "").strip() or None
        token_url = os.getenv("SKYLENS_DJI_TOKEN_URL", "").strip() or None
        if not wk_key:
            return None
        # A token file / URL counts as a configured token source even when empty
        # now; a central refresher may populate it before the first upload.
        if not token and not token_file and not token_url:
            return None
        return cls(
            token=token,
            wk_key=wk_key,
            secret_id=os.getenv("SKYLENS_DJI_WK_SECRETID", "uav").strip() or "uav",
            app_info=os.getenv("SKYLENS_DJI_APP_INFO", "Android-SkyLens-13-KR").strip()
            or "Android-SkyLens-13-KR",
            token_file=token_file,
            token_url=token_url,
        )

    def resolve_token(self, force: bool = False) -> str:
        """Current token.

        A WSA-less PC points SKYLENS_DJI_TOKEN_URL (or _FILE) at a central source
        that a refresher keeps fresh, so no device is needed here. Priority:
        URL (cached briefly) -> file -> static env token. ``force`` bypasses the
        URL cache, used by the one auth-error retry.
        """
        if self.token_url:
            value = _fetch_token_url(self.token_url, force=force)
            if value:
                return value
        if self.token_file:
            try:
                value = Path(self.token_file).expanduser().read_text(encoding="utf-8").strip()
                if value:
                    return value
            except OSError:
                pass
        return self.token


# ── DJI request signing (x-wk-sign) ─────────────────────────────────────
def sign_request(path_with_query: str, nonce: int, timestamp_ms: int, cfg: CloudConfig) -> str:
    """base64(HMAC-SHA256) of the DJI canonical string. Pure and deterministic."""
    signed = (
        f"{path_with_query}&X-Wk-Nonce={nonce}&X-Wk-SecretId={cfg.secret_id}"
        f"&X-Wk-Signature-Method=HmacSHA256&X-Wk-Timestamp={timestamp_ms}"
    )
    return base64.b64encode(
        hmac.new(cfg.wk_key.encode(), signed.encode(), hashlib.sha256).digest()
    ).decode()


def _wk_headers(path_with_query: str, cfg: CloudConfig, force_token: bool = False) -> dict[str, str]:
    nonce = random.randint(-(2**31), 2**31 - 1)
    ts = int(time.time() * 1000)
    sign = sign_request(path_with_query, nonce, ts, cfg)
    return {
        "x-wk-nonce": str(nonce),
        "x-wk-secretid": cfg.secret_id,
        "x-wk-timestamp": str(ts),
        "x-wk-sign": sign,
        "x-mc-token": cfg.resolve_token(force=force_token),
        "x-request-id": str(uuid.uuid4()),
        "x-app-info": cfg.app_info,
        "content-type": "application/json; charset=UTF-8",
        "user-agent": "okhttp/4.12.0",
    }


_AUTH_HINTS = ("token", "auth", "login", "expire", "unauthor", "登录", "鉴权")


def _looks_like_auth_error(message: str) -> bool:
    low = message.lower()
    return any(hint in low for hint in _AUTH_HINTS)


def _api_once(
    method: str, path: str, cfg: CloudConfig, body: Any, force_token: bool = False
) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE_URL + path, data=data, method=method, headers=_wk_headers(path, cfg, force_token)
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            parsed = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise CloudAuthError(f"DJI {method} {path} -> HTTP {exc.code}") from exc
        raise CloudError(f"DJI {method} {path} -> HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise CloudError(f"DJI {method} {path} unreachable: {exc.reason}") from exc
    result = parsed.get("result") if isinstance(parsed, dict) else None
    if not isinstance(result, dict) or result.get("code") != 0:
        msg = str(result.get("msg") if isinstance(result, dict) else "unknown error")
        if _looks_like_auth_error(msg):
            raise CloudAuthError(f"DJI {method} {path} rejected: {msg}")
        raise CloudError(f"DJI {method} {path} rejected: {msg}")
    return parsed


def _api(method: str, path: str, cfg: CloudConfig, body: Any = None) -> dict[str, Any]:
    """Call the DJI API; retry once on an auth error after re-reading the token.

    The retry matters only when a token file is configured and an external
    refresher has written a newer token since the first attempt.
    """
    try:
        return _api_once(method, path, cfg, body)
    except CloudAuthError:
        if not cfg.token_file and not cfg.token_url:
            raise
        time.sleep(0.5)
        # force_token bypasses the URL cache so we pick up a freshly refreshed token.
        return _api_once(method, path, cfg, body, force_token=True)


def compare(cfg: CloudConfig, missions: list[dict] | None = None, start_time: int = 0) -> dict:
    return _api(
        "POST",
        "/api/waypoint/mission/compare",
        cfg,
        {"missions": missions or [], "start_time": start_time},
    )


def check(cfg: CloudConfig, uuids: list[str]) -> list[dict]:
    data = _api("POST", "/api/waypoint/mission/check", cfg, {"mission_uuid_list": uuids})
    return data.get("data", {}).get("items", []) or []


def list_missions(cfg: CloudConfig) -> list[dict]:
    uuids = compare(cfg).get("data", {}).get("mission_download_list") or []
    return check(cfg, uuids) if uuids else []


def delete_mission(cfg: CloudConfig, mission_uuid: str) -> bool:
    now = int(time.time() * 1000)
    compare(
        cfg,
        missions=[
            {
                "uuid": mission_uuid,
                "name": "",
                "md5": "",
                "point_gps_list": [],
                "app_create_time": now - 1000,
                "app_update_time": now,
                "app_delete_time": now,
            }
        ],
    )
    return not any(it.get("uuid") == mission_uuid for it in list_missions(cfg))


# ── KMZ helpers ─────────────────────────────────────────────────────────
def _kmz_points(kmz_bytes: bytes) -> tuple[list[dict[str, float]], float, float]:
    lats: list[float] = []
    lons: list[float] = []
    try:
        with zipfile.ZipFile(io.BytesIO(kmz_bytes)) as archive:
            names = archive.namelist()
            name = next((n for n in names if n.endswith("waylines.wpml")), None) or next(
                (n for n in names if n.endswith(".kml")), None
            )
            if name:
                xml = archive.read(name).decode("utf-8", "replace")
                for match in re.finditer(r"<coordinates>\s*([-\d.]+),([-\d.]+)", xml):
                    lons.append(float(match.group(1)))
                    lats.append(float(match.group(2)))
    except (zipfile.BadZipFile, ValueError):
        pass
    gps = [{"latitude": la, "longitude": lo} for la, lo in zip(lats, lons, strict=False)]
    center_lat = sum(lats) / len(lats) if lats else 0.0
    center_lon = sum(lons) / len(lons) if lons else 0.0
    return gps, center_lat, center_lon


def wrap_mission_package(kmz_bytes: bytes, mission_uuid: str) -> bytes:
    """Wrap a raw wpmz KMZ as the cloud mission-file zip (<uuid>.kmz + ShotSnap)."""
    try:
        probe = zipfile.ZipFile(io.BytesIO(kmz_bytes))
        names = probe.namelist()
        is_raw = any(n.endswith(("template.kml", "waylines.wpml")) for n in names)
        is_wrapped = any(n.endswith(".kmz") for n in names)
        if is_wrapped and not is_raw:
            return kmz_bytes
        if not is_raw:
            return kmz_bytes
    except zipfile.BadZipFile:
        return kmz_bytes
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as out:
        out.writestr(f"{mission_uuid}.kmz", kmz_bytes)
        out.writestr("image/ShotSnap.json", json.dumps({"WAY_POINT": {}, "POI_POINT": {}}))
    return buffer.getvalue()


# ── AWS S3 upload (manual SigV4, stdlib only) ───────────────────────────
def _s3_put(file_meta: dict[str, Any], body: bytes) -> None:
    auth = file_meta["authorization"]
    bucket = file_meta["cloud_bucket_name"]
    region = file_meta["region"]
    key = file_meta["object_key"]
    host = f"{bucket}.s3.{region}.amazonaws.com"
    now = datetime.datetime.now(datetime.UTC)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = now.strftime("%Y%m%d")
    canonical_uri = "/" + "/".join(urllib.parse.quote(seg, safe="~") for seg in key.split("/"))
    payload_hash = hashlib.sha256(body).hexdigest()
    token = auth["session_token"]
    canonical_headers = (
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
        f"x-amz-security-token:{token}\n"
    )
    signed_headers = "host;x-amz-content-sha256;x-amz-date;x-amz-security-token"
    canonical_request = "\n".join(
        ["PUT", canonical_uri, "", canonical_headers, signed_headers, payload_hash]
    )
    scope = f"{datestamp}/{region}/s3/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        ]
    )

    def _sign(key_bytes: bytes, msg: str) -> bytes:
        return hmac.new(key_bytes, msg.encode(), hashlib.sha256).digest()

    k_date = _sign(("AWS4" + auth["sk"]).encode(), datestamp)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, "s3")
    k_signing = _sign(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode(), hashlib.sha256).hexdigest()
    authorization = (
        f"AWS4-HMAC-SHA256 Credential={auth['ak']}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    req = urllib.request.Request(
        f"https://{host}{canonical_uri}",
        data=body,
        method="PUT",
        headers={
            "Authorization": authorization,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
            "x-amz-security-token": token,
            "Content-Length": str(len(body)),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status not in (200, 204):
                raise CloudError(f"S3 upload failed: HTTP {resp.status}")
    except urllib.error.HTTPError as exc:
        raise CloudError(f"S3 upload failed: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise CloudError(f"S3 upload unreachable: {exc.reason}") from exc


# ── public: upload one mission to the DJI cloud ─────────────────────────
@dataclass(frozen=True)
class CloudResult:
    mission_uuid: str
    name: str
    verified: bool


def upload_mission(
    cfg: CloudConfig,
    kmz_bytes: bytes,
    name: str,
    distance_m: float = 0.0,
    cover_png: bytes | None = None,
) -> CloudResult:
    """Push one KMZ to the account cloud so an RC 2 can download it over the air."""
    mission_uuid = str(uuid.uuid4())
    now = int(time.time() * 1000)
    gps, lat, lon = _kmz_points(kmz_bytes)
    package = wrap_mission_package(kmz_bytes, mission_uuid)
    cover = cover_png if cover_png is not None else _PLACEHOLDER_PNG

    base = {
        "app_create_time": now,
        "app_update_time": now,
        "biz_version": "1",
        "distance": float(distance_m),
        "drone_list": [],
        "duration": 0,
        "latitude": lat,
        "longitude": lon,
        "mission_name": name,
        "mission_uuid": mission_uuid,
        "point_gps_list": gps,
        "point_num": len(gps),
    }
    files = [
        dict(
            base,
            biz_type="mission_file",
            md5=hashlib.md5(package).hexdigest(),  # noqa: S324 - server contract, not security
            name=f"{mission_uuid}.kmz",
            uuid=str(uuid.uuid4()),
        ),
        dict(
            base,
            biz_type="mission_cover",
            md5=hashlib.md5(cover).hexdigest(),  # noqa: S324 - server contract, not security
            name=f"{name}.png",
            uuid=str(uuid.uuid4()),
        ),
    ]
    compare(
        cfg,
        missions=[
            {
                "uuid": mission_uuid,
                "name": name,
                "md5": hashlib.md5(package).hexdigest(),  # noqa: S324
                "point_gps_list": gps,
                "app_create_time": now,
                "app_update_time": now,
                "app_delete_time": 0,
            }
        ],
    )
    sts = _api("POST", "/api/waypoint/file/upload/sts/init", cfg, {"files": files})
    for file_meta in sts["data"]["files"]:
        payload = package if file_meta["biz_type"] == "mission_file" else cover
        _s3_put(file_meta, payload)

    verified = False
    for _ in range(6):
        time.sleep(2)
        if any(it.get("uuid") == mission_uuid for it in check(cfg, [mission_uuid])):
            verified = True
            break
    return CloudResult(mission_uuid=mission_uuid, name=name, verified=verified)


def upload_mission_file(
    cfg: CloudConfig, kmz_path: Path, name: str, distance_m: float = 0.0
) -> CloudResult:
    return upload_mission(cfg, Path(kmz_path).read_bytes(), name, distance_m)
