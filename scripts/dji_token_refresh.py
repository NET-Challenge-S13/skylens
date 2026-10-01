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

It attaches to dji.go.v5 and scans process memory for the resident account token
(it looks like ``US_<base64url>``), then writes it. A memory scan is used on
purpose: DJI Fly minifies OkHttp method names and ships anti-frida guards, so
hooking Java methods by name is brittle, while the token sits in memory as plain
UTF-8. Run it once, or on a schedule (Task Scheduler / cron) to keep it fresh.

Usage:
  python scripts/dji_token_refresh.py                 # write to default file
  python scripts/dji_token_refresh.py --out path.txt  # explicit output
  SKYLENS_ADB=C:\\platform-tools\\adb.exe python scripts/dji_token_refresh.py
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PACKAGE = "dji.go.v5"
ADB = os.getenv("SKYLENS_ADB", "adb")
SERIAL = os.getenv("SKYLENS_WSA_SERIAL", "127.0.0.1:58526")
# Token prefix to anchor the memory scan on. DJI Fly tokens observed as "US_...".
TOKEN_PREFIX = os.getenv("SKYLENS_DJI_TOKEN_PREFIX", "US_")
TOKEN_RE = re.compile(rf"{re.escape(TOKEN_PREFIX)}[A-Za-z0-9_-]{{80,240}}")

# Scan readable memory for the token prefix and send a window around each hit;
# Python extracts and validates the token. No Java-method or OkHttp dependency.
SCAN_JS = r"""
(function () {
    var prefixBytes = %s;  // hex byte pattern for the token prefix
    var ranges = Process.enumerateRanges('r--');
    ranges.forEach(function (r) {
        try {
            Memory.scan(r.base, r.size, prefixBytes, {
                onMatch: function (addr) {
                    for (var len = 260; len >= 80; len -= 60) {
                        try {
                            var s = addr.readUtf8String(len);
                            if (s) { send({ s: s }); break; }
                        } catch (e) { /* page edge; try a shorter read */ }
                    }
                },
                onError: function () {},
                onComplete: function () {},
            });
        } catch (e) { /* unreadable range */ }
    });
    send({ done: true });
})();
"""


def _adb(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [ADB, "-s", SERIAL, *args], capture_output=True, text=True, timeout=30
    )


def _resolve_pid(device: object) -> int | None:
    """Find the app's main pid. frida's name lookup is flaky on WSA, so ask adb
    first (largest RSS = the UI process), then fall back to frida enumeration."""
    out = _adb("shell", "su", "-c", "ps -A -o PID,RSS,NAME").stdout
    best_pid, best_rss = None, -1
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1] == PACKAGE:
            try:
                pid_i, rss_i = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if rss_i > best_rss:
                best_pid, best_rss = pid_i, rss_i
    if best_pid is not None:
        return best_pid
    return next(
        (p.pid for p in device.enumerate_processes() if p.name == PACKAGE),  # type: ignore[attr-defined]
        None,
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

    pid = _resolve_pid(device)
    if pid is None:
        print(f"{PACKAGE} is not running on {SERIAL}; open DJI Fly first", file=sys.stderr)
        return 2

    prefix_hex = " ".join(f"{b:02x}" for b in TOKEN_PREFIX.encode())
    done = {"flag": False}
    candidates: set[str] = set()
    session = device.attach(pid)
    script = session.create_script(SCAN_JS % repr(prefix_hex))

    def on_message(message: dict, _data: object) -> None:
        if message.get("type") != "send":
            return
        payload = message.get("payload") or {}
        text = payload.get("s")
        if isinstance(text, str):
            match = TOKEN_RE.match(text)
            if match:
                candidates.add(match.group(0))
        if payload.get("done"):
            done["flag"] = True

    script.on("message", on_message)
    script.load()

    for _ in range(40):
        if done["flag"]:
            break
        time.sleep(0.5)
    session.detach()

    if not candidates:
        print(
            f"no {TOKEN_PREFIX!r} token found in memory; is the account logged in?",
            file=sys.stderr,
        )
        return 1
    # The full token is the longest run that starts with the prefix.
    token = max(candidates, key=len).strip()

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(token, encoding="utf-8")
    print(f"token refreshed -> {out_path} ({len(token)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
