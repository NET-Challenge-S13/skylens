"""Server settings, all from the environment so nothing is hardcoded per host.

| env | default | meaning |
|---|---|---|
| ``SKYLENS_DEMO``            | ``0``                | 1 = demo mode: serve prebuilt assets, train nothing |
| ``SKYLENS_DEMO_MANIFEST``   | ``<repo>/res/static/demo/segments.json`` | segment x level manifest |
| ``SKYLENS_DEMO_URL_BASE``   | ``/res/static/demo``  | url prefix the clients fetch assets under |
| ``SKYLENS_DEMO_STEP_SECONDS`` | ``0.0006``          | simulated seconds per training step |
| ``SKYLENS_DEMO_MIN_SECONDS``  | ``0.3``             | floor on the simulated delay |
| ``SKYLENS_DEMO_MAX_SECONDS``  | ``12.0``            | cap on the simulated delay |
| ``SKYLENS_ANCHOR``          | ``36.3685,127.3475,30`` | ENU origin, matches shared/viewer/config.ts |
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .geo import Gps

#: skylens_model/serving/config.py -> skylens_model/serving -> skylens_model -> src -> repo
REPO_ROOT = Path(__file__).resolve().parents[3]


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _num(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    demo: bool
    manifest: Path
    url_base: str
    step_seconds: float
    min_seconds: float
    max_seconds: float
    anchor: Gps

    def demo_root(self) -> Path:
        """Directory the manifest's relative urls resolve against."""
        return self.manifest.parent


def _anchor() -> Gps:
    raw = os.environ.get("SKYLENS_ANCHOR", "36.3685,127.3475,30")
    try:
        lat, lon, alt = (float(p) for p in raw.split(",")[:3])
    except ValueError:
        lat, lon, alt = 36.3685, 127.3475, 30.0
    return Gps(lat=lat, lon=lon, alt=alt)


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Read once per process. Restart the server to change these."""
    manifest = Path(
        os.environ.get("SKYLENS_DEMO_MANIFEST", REPO_ROOT / "res" / "static" / "demo" / "segments.json")
    ).resolve()
    return Settings(
        demo=_flag("SKYLENS_DEMO"),
        manifest=manifest,
        url_base=os.environ.get("SKYLENS_DEMO_URL_BASE", "/res/static/demo").rstrip("/"),
        step_seconds=_num("SKYLENS_DEMO_STEP_SECONDS", 0.0006),
        min_seconds=_num("SKYLENS_DEMO_MIN_SECONDS", 0.3),
        max_seconds=_num("SKYLENS_DEMO_MAX_SECONDS", 12.0),
        anchor=_anchor(),
    )
