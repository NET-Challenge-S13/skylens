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
# Opening the mission library makes the app issue an authenticated cloud call,
# which pulls the token into heap memory where the scan can see it.
MISSION_ACTIVITY = "dji.go.v5/com.uav.waypoint.missionlib.MissionLibActivity"
ADB = os.getenv("SKYLENS_ADB", "adb")
SERIAL = os.getenv("SKYLENS_WSA_SERIAL", "127.0.0.1:58526")
# Token prefix to anchor the memory scan on. DJI Fly tokens observed as "US_...".
TOKEN_PREFIX = os.getenv("SKYLENS_DJI_TOKEN_PREFIX", "US_")
TOKEN_RE = re.compile(rf"{re.escape(TOKEN_PREFIX)}[A-Za-z0-9_-]{{90,200}}")


def _looks_like_token(s: str) -> bool:
    """Tell the real account token from same-prefixed native symbols.

    The mc-token is a high-entropy ~131-char base64url string: many digits, both
    letter cases, almost no underscores. Mangled C++ symbols that also start with
    the prefix (e.g. US_TO_APP_req...) are longer, underscore-heavy, digit-poor.
    """
    if not (120 <= len(s) <= 160):
        return False
    digits = sum(c.isdigit() for c in s)
    has_lower = any(c.islower() for c in s)
    has_upper = any(c.isupper() for c in s)
    return digits >= 18 and has_lower and has_upper and s.count("_") <= 4

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


def _get_device(timeout: int = 10):  # noqa: ANN202 - frida.core.Device, imported lazily
    import frida

    _adb("connect", SERIAL)
    return frida.get_usb_device(timeout=timeout)


def capture_token(device) -> str | None:  # noqa: ANN001 - frida device
    """Scan the running DJI Fly's memory and return the live account token, or None."""
    from collections import Counter

    pid = _resolve_pid(device)
    if pid is None:
        return None

    prefix_hex = " ".join(f"{b:02x}" for b in TOKEN_PREFIX.encode())
    done = {"flag": False}
    counts: Counter[str] = Counter()
    session = device.attach(pid)

    # Nudge the app to make an authenticated request so the token is resident.
    _adb("shell", "monkey", "-p", PACKAGE, "-c", "android.intent.category.LAUNCHER", "1")
    time.sleep(2)
    _adb("shell", "su", "-c", f"am start -n {MISSION_ACTIVITY}")
    time.sleep(4)

    script = session.create_script(SCAN_JS % repr(prefix_hex))

    def on_message(message: dict, _data: object) -> None:
        if message.get("type") != "send":
            return
        payload = message.get("payload") or {}
        text = payload.get("s")
        if isinstance(text, str):
            match = TOKEN_RE.match(text)
            if match:
                counts[match.group(0)] += 1
        if payload.get("done"):
            done["flag"] = True

    script.on("message", on_message)
    script.load()
    for _ in range(40):
        if done["flag"]:
            break
        time.sleep(0.5)
    try:
        session.detach()
    except Exception:  # noqa: BLE001 - detaching a dead session is fine
        pass

    # Keep only token-shaped candidates, then take the most frequently resident
    # one (the live token is referenced from many places; a stray symbol is not).
    ranked = [s for s, _n in counts.most_common() if _looks_like_token(s)]
    return ranked[0].strip() if ranked else None


def _serve(port: int, interval: int, out_path: Path) -> int:
    """Rig mode: refresh the token on a timer and serve it over HTTP so WSA-less
    field PCs can point SKYLENS_DJI_TOKEN_URL here. Run on a trusted network only;
    the token is a credential."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = {"token": ""}
    if out_path.exists():
        state["token"] = out_path.read_text(encoding="utf-8").strip()

    def refresh_loop() -> None:
        while True:
            try:
                token = capture_token(_get_device())
                if token:
                    state["token"] = token
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(token, encoding="utf-8")
                    print(f"[refresh] token updated ({len(token)} chars)")
                else:
                    print("[refresh] no token found (is DJI Fly logged in?)")
            except Exception as exc:  # noqa: BLE001 - keep the server alive
                print(f"[refresh] error: {exc}")
            time.sleep(interval)

    threading.Thread(target=refresh_loop, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            token = state["token"].encode()
            if not token:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(token)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(token)

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"serving DJI token on http://0.0.0.0:{port}/ (refresh every {interval}s)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the DJI Fly cloud token.")
    default_out = os.getenv("SKYLENS_DJI_TOKEN_FILE") or str(
        Path(__file__).resolve().parents[1] / "output" / ".dji_token"
    )
    parser.add_argument("--out", default=default_out, help="token output file")
    parser.add_argument(
        "--serve", type=int, metavar="PORT",
        help="run as a token server on PORT, refreshing on a timer (rig mode)",
    )
    parser.add_argument(
        "--interval", type=int, default=1800, help="refresh interval seconds for --serve"
    )
    args = parser.parse_args()

    try:
        import frida  # noqa: F401 - probe the dependency early
    except ImportError:
        print("frida is required: pip install frida", file=sys.stderr)
        return 2

    out_path = Path(args.out).expanduser()
    if args.serve:
        return _serve(args.serve, args.interval, out_path)

    try:
        device = _get_device()
    except Exception as exc:  # noqa: BLE001 - surface any attach failure plainly
        print(f"cannot reach frida device ({SERIAL}): {exc}", file=sys.stderr)
        return 2
    token = capture_token(device)
    if not token:
        print(
            f"no {TOKEN_PREFIX!r} account token found in memory; is DJI Fly logged in?",
            file=sys.stderr,
        )
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(token, encoding="utf-8")
    print(f"token refreshed -> {out_path} ({len(token)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
