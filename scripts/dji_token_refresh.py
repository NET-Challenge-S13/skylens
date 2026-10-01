#!/usr/bin/env python3
"""Refresh the DJI Fly account token the mission bridge uses for cloud delivery.

The ``x-mc-token`` the bridge sends expires. A logged-in DJI Fly session already
holds a fresh one and renews it on its own. This tool reads that live token out
of the running app and writes it to the file the bridge watches
(``SKYLENS_DJI_TOKEN_FILE``), so cloud uploads keep working without hand-copying.

This is a rig tool, NOT part of the dependency-free bridge. It needs the DFIR
extraction setup, not a field laptop:

  - Windows Subsystem for Android (or any rooted Android) running DJI Fly, logged
    into the account, reachable over adb.
  - frida-server running on that device, and ``pip install frida`` here.
  - adb on PATH, or SKYLENS_ADB pointing at it.

It attaches to dji.go.v5, hooks the OkHttp header calls, triggers one waypoint
sync so a request goes out, captures the ``x-mc-token`` header value, and writes
it. Run it once, or on a schedule (Task Scheduler / cron) to keep the file fresh.

Usage:
  python scripts/dji_token_refresh.py                 # write to default file
  python scripts/dji_token_refresh.py --out path.txt  # explicit output
  SKYLENS_ADB=C:\\platform-tools\\adb.exe python scripts/dji_token_refresh.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PACKAGE = "dji.go.v5"
MISSION_ACTIVITY = "dji.go.v5/com.uav.waypoint.missionlib.MissionLibActivity"
ADB = os.getenv("SKYLENS_ADB", "adb")
SERIAL = os.getenv("SKYLENS_WSA_SERIAL", "127.0.0.1:58526")

# Hook the OkHttp header setters (okhttp3 package names survive R8) and report any
# value whose header name is x-mc-token.
HOOK_JS = r"""
Java.perform(function () {
    function report(name, value) {
        if (name && name.toLowerCase() === 'x-mc-token' && value) {
            send({ token: value });
        }
    }
    var targets = [
        ['okhttp3.Request$Builder', 'header'],
        ['okhttp3.Request$Builder', 'addHeader'],
        ['okhttp3.Headers$Builder', 'add'],
        ['okhttp3.Headers$Builder', 'set'],
    ];
    targets.forEach(function (pair) {
        try {
            var cls = Java.use(pair[0]);
            cls[pair[1]].overload('java.lang.String', 'java.lang.String').implementation =
                function (name, value) {
                    report(name, value);
                    return this[pair[1]](name, value);
                };
        } catch (e) {
            /* overload absent in this build; the others still cover it */
        }
    });
    send({ ready: true });
});
"""


def _adb(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [ADB, "-s", SERIAL, *args], capture_output=True, text=True, timeout=30
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the DJI Fly cloud token.")
    default_out = os.getenv("SKYLENS_DJI_TOKEN_FILE") or str(
        Path(__file__).resolve().parents[1] / "output" / ".dji_token"
    )
    parser.add_argument("--out", default=default_out, help="token output file")
    args = parser.parse_args()

    try:
        import frida
    except ImportError:
        print("frida is required: pip install frida", file=sys.stderr)
        return 2

    _adb("connect", SERIAL)
    try:
        device = frida.get_usb_device(timeout=10)
    except Exception as exc:  # noqa: BLE001 - surface any attach failure plainly
        print(f"cannot reach frida device ({SERIAL}): {exc}", file=sys.stderr)
        return 2

    pid = next(
        (p.pid for p in device.enumerate_processes() if p.name == PACKAGE),
        None,
    )
    if pid is None:
        print(f"{PACKAGE} is not running on {SERIAL}; open DJI Fly first", file=sys.stderr)
        return 2

    captured: dict[str, str] = {}
    session = device.attach(pid)
    script = session.create_script(HOOK_JS)

    def on_message(message: dict, _data: object) -> None:
        if message.get("type") == "send":
            payload = message.get("payload") or {}
            if payload.get("token"):
                captured["token"] = payload["token"]

    script.on("message", on_message)
    script.load()

    # Trigger a request so the interceptor runs (opening the mission library syncs).
    _adb("shell", "su", "-c", f"am start -n {MISSION_ACTIVITY}")

    for _ in range(20):
        if "token" in captured:
            break
        time.sleep(0.5)
    session.detach()

    token = captured.get("token", "").strip()
    if not token:
        print("no x-mc-token seen; is the account logged in?", file=sys.stderr)
        return 1

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(token, encoding="utf-8")
    print(f"token refreshed -> {out_path} ({len(token)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
