#!/usr/bin/env python3
"""The KOREN comparison probe, v2.

One run prints one JSON line on stdout. cron runs it every 10 minutes and the
results pile up as JSONL for aggregation later.

What it measures
  1) TCP connect round trip - ICMP is blocked, so the handshake stands in for it.
  2) Burst completion time  - how long one .spz chunk takes to arrive. The real KPI here.
  3) Parallel throughput    - the same volume split across N streams, which tells
                              whether a single stream is bound by TCP slow start
                              and the bandwidth-delay product.
  4) CPU usage              - if the CPU saturates mid-transfer, the number is the
                              encryption limit, not the line. (5) A public baseline:
both nodes fetch the same remote file to compare egress. No dependencies, stdlib only.
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def cpu_snapshot() -> tuple[int, int]:
    """(idle, total) cumulative ticks, jitter-free."""
    with open("/proc/stat", encoding="ascii") as fh:
        parts = [int(x) for x in fh.readline().split()[1:]]
    idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
    return idle, sum(parts)


def cpu_busy_pct(before: tuple[int, int], after: tuple[int, int]) -> float:
    d_idle = after[0] - before[0]
    d_total = after[1] - before[1]
    if d_total <= 0:
        return 0.0
    return round(100.0 * (1 - d_idle / d_total), 1)


def tcp_rtt(host: str, port: int, n: int = 20, timeout: float = 5.0) -> dict:
    """TCP handshake round trip, the stand-in where ICMP is blocked."""
    samples: list[float] = []
    fails = 0
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        t0 = time.perf_counter()
        try:
            s.connect((host, port))
            samples.append((time.perf_counter() - t0) * 1000.0)
        except OSError:
            fails += 1
        finally:
            s.close()
        time.sleep(0.05)
    if not samples:
        return {"ok": False, "fails": fails, "n": n}
    ordered = sorted(samples)
    return {
        "ok": True,
        "n": n,
        "fails": fails,
        "min_ms": round(ordered[0], 3),
        "med_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3),
        "max_ms": round(ordered[-1], 3),
        # Jitter is the mean absolute difference between consecutive samples in arrival order.
        "jitter_ms": round(
            statistics.fmean(abs(b - a) for a, b in zip(samples, samples[1:])), 3
        )
        if len(samples) > 1
        else 0.0,
    }


def banner_rtt(host: str, port: int, n: int = 10, timeout: float = 15.0) -> dict:
    """Time until the far sshd answers with its banner.

    Unlike tcp_rtt, which times the handshake only, this really goes to the far
    server and back, so it stays correct on a locally forwarded port.
    """
    samples: list[float] = []
    fails = 0
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        t0 = time.perf_counter()
        try:
            s.connect((host, port))
            data = s.recv(64)
            if data:
                samples.append((time.perf_counter() - t0) * 1000.0)
            else:
                fails += 1
        except OSError:
            fails += 1
        finally:
            s.close()
        time.sleep(0.2)
    if not samples:
        return {"ok": False, "fails": fails, "n": n}
    ordered = sorted(samples)
    return {
        "ok": True,
        "n": n,
        "fails": fails,
        "min_ms": round(ordered[0], 3),
        "med_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3),
        "max_ms": round(ordered[-1], 3),
        "jitter_ms": round(
            statistics.fmean(abs(b - a) for a, b in zip(samples, samples[1:])), 3
        )
        if len(samples) > 1
        else 0.0,
    }


def sh(cmd: str, timeout: int) -> tuple[int, str, str]:
    p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


SSH_OPTS = "-o BatchMode=yes -o Compression=no -o ConnectTimeout=15"


def _stream_cmd(ssh_host: str, blocks: int, direction: str) -> str:
    if direction == "up":
        return (f"dd if=/dev/zero bs=65536 count={blocks} status=none | "
                f"ssh {SSH_OPTS} {ssh_host} 'cat > /dev/null'")
    return (f"ssh {SSH_OPTS} {ssh_host} "
            f"'dd if=/dev/zero bs=65536 count={blocks} status=none' > /dev/null")


def ssh_burst(ssh_host: str, size_mb: int, direction: str, streams: int = 1,
              timeout: int = 240) -> dict:
    """Send or receive size_mb over SSH and time it to completion.

    In our deployment the .spz chunks and markers really do cross an SSH tunnel, so
    this is the operational path rather than a synthetic benchmark. Compression is
    off (Compression=no): a zero-filled payload would compress and inflate the rate.
    """
    per = max(1, (size_mb * 16) // streams)
    cmds = [_stream_cmd(ssh_host, per, direction) for _ in range(streams)]
    c0 = cpu_snapshot()
    t0 = time.perf_counter()
    if streams == 1:
        rc, _, err = sh(cmds[0], timeout)
        rcs = [rc]
        errs = [err]
    else:
        with ThreadPoolExecutor(max_workers=streams) as pool:
            out = list(pool.map(lambda c: sh(c, timeout), cmds))
        rcs = [o[0] for o in out]
        errs = [o[2] for o in out]
    dt = time.perf_counter() - t0
    c1 = cpu_snapshot()
    sent_mb = per * streams / 16
    if any(rcs):
        return {"ok": False, "dir": direction, "streams": streams,
                "err": next((e for e in errs if e), "")[:200]}
    return {
        "ok": True,
        "size_mb": round(sent_mb, 1),
        "dir": direction,
        "streams": streams,
        "seconds": round(dt, 3),
        "mbps": round(sent_mb * 8 / dt, 1),
        "cpu_busy_pct": cpu_busy_pct(c0, c1),
    }


def ssh_burst_hosts(hosts: list[str], size_mb: int, direction: str,
                    streams: int = 4, timeout: int = 240) -> dict:
    """Spread the streams over several hosts, meaning several independent connections.

    Several streams to one host only split one SSH session into channels, which still
    share a single TCP connection, so the parallelism is not real. This matters most
    when measuring Pangyo behind the tunnel.
    """
    per = max(1, (size_mb * 16) // streams)
    cmds = [_stream_cmd(hosts[i % len(hosts)], per, direction) for i in range(streams)]
    c0 = cpu_snapshot()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=streams) as pool:
        out = list(pool.map(lambda c: sh(c, timeout), cmds))
    dt = time.perf_counter() - t0
    c1 = cpu_snapshot()
    sent_mb = per * streams / 16
    if any(o[0] for o in out):
        return {"ok": False, "dir": direction, "streams": streams,
                "err": next((o[2] for o in out if o[2]), "")[:200]}
    return {
        "ok": True,
        "size_mb": round(sent_mb, 1),
        "dir": direction,
        "streams": streams,
        "hosts": len(set(hosts[:streams])),
        "seconds": round(dt, 3),
        "mbps": round(sent_mb * 8 / dt, 1),
        "cpu_busy_pct": cpu_busy_pct(c0, c1),
    }


def http_download(name: str, url: str, mb: int = 50, timeout: int = 120) -> dict:
    """The public baseline: both nodes fetch the same byte range of the same remote
    file, so egress performance can be compared.
    A Range request pins the volume, so the comparison holds even if the file changes size.
    """
    c0 = cpu_snapshot()
    last_byte = mb * 1024 * 1024 - 1
    rc, out, err = sh(
        f'curl -sS -o /dev/null -m {timeout} -r 0-{last_byte} '
        f'-w "%{{speed_download}} %{{time_total}} %{{size_download}} %{{http_code}}" '
        f'"{url}"',
        timeout + 15,
    )
    c1 = cpu_snapshot()
    if rc != 0:
        return {"ok": False, "name": name, "url": url, "err": err[:160]}
    try:
        speed, total, size, code = out.split()
        got = int(float(size))
        if int(code) not in (200, 206) or got < 1024 * 1024:
            return {"ok": False, "name": name, "url": url,
                    "http": int(code), "bytes": got}
        return {
            "ok": True,
            "name": name,
            "url": url,
            "mbps": round(float(speed) * 8 / 1e6, 1),
            "seconds": round(float(total), 3),
            "bytes": got,
            "http": int(code),
            "cpu_busy_pct": cpu_busy_pct(c0, c1),
        }
    except ValueError:
        return {"ok": False, "name": name, "url": url, "raw": out[:160]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="이 프로브가 서 있는 지점 이름")
    ap.add_argument("--rtt", action="append", default=[],
                    help="host:port:name 형식, 여러 번 가능")
    ap.add_argument("--banner", action="append", default=[],
                    help="host:port:name. 상대 sshd 배너까지의 왕복시간")
    ap.add_argument("--ssh-host", help="SSH 버스트를 잴 상대(ssh config 이름)")
    ap.add_argument("--burst-mb", default="4,30",
                    help="버스트 크기 목록 MB. 기본 4(.spz 1장), 30(.ply 1장)")
    ap.add_argument("--parallel-host", action="append", default=[],
                    help="병렬 측정을 분산할 호스트 목록. 각각 독립 연결이어야 공정하다")
    ap.add_argument("--parallel", type=int, default=4,
                    help="병렬 스트림 수. 단일과 비교해 TCP 시작 구간 영향을 가른다")
    ap.add_argument("--parallel-mb", type=int, default=100,
                    help="병렬 측정에 쓸 총량 MB")
    ap.add_argument("--http", action="append", default=[],
                    help="name=url 형식. 여러 번 주면 각각 측정한다")
    ap.add_argument("--http-mb", type=int, default=50, help="기준선에서 받을 양 MB")
    ap.add_argument("--skip-burst", action="store_true")
    ap.add_argument("--skip-parallel", action="store_true")
    args = ap.parse_args()

    rec: dict = {"ts": now_iso(), "label": args.label, "rtt": {}, "burst": [], "http": []}

    for spec in args.rtt:
        parts = spec.split(":")
        host, port = parts[0], int(parts[1])
        name = parts[2] if len(parts) > 2 else f"{host}:{port}"
        rec["rtt"][name] = tcp_rtt(host, port)

    for spec in args.banner:
        parts = spec.split(":")
        host, port = parts[0], int(parts[1])
        name = parts[2] if len(parts) > 2 else f"banner {host}:{port}"
        rec["rtt"][name] = banner_rtt(host, port)

    if args.ssh_host and not args.skip_burst:
        for mb in [int(x) for x in args.burst_mb.split(",") if x]:
            for direction in ("up", "down"):
                rec["burst"].append(ssh_burst(args.ssh_host, mb, direction))
        if not args.skip_parallel and args.parallel > 1:
            hosts = args.parallel_host or [args.ssh_host]
            for direction in ("up", "down"):
                rec["burst"].append(
                    ssh_burst_hosts(hosts, args.parallel_mb, direction,
                                    streams=args.parallel)
                )

    for spec in args.http:
        name, _, url = spec.partition("=")
        rec["http"].append(http_download(name or url, url or name, mb=args.http_mb))

    print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
