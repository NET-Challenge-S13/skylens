#!/usr/bin/env python3
"""The deeper probe: the long experiment that runs once an hour.

Where the 10-minute probe (`probe.py`) watches the trend, this one draws the
curves. One point a day is not enough; several days of curves answer these:

  Q1. How large does a transfer have to be before it reaches line speed (where is
      the knee of TCP slow start)
  Q2. Where does adding streams saturate (is a single stream bound)
  Q3. Does a long send hold up, or does it collapse partway
  Q4. How many retransmits appear (is loss eating the performance)
Usage:
    python3 deep.py --label koren-internal --ssh-host pangyo
    python3 deep.py --label pi-175        --ssh-host mnd
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import probe


def tcp_counters() -> dict[str, int]:
    """The kernel's cumulative TCP counters; the before/after difference gives this transfer's retransmits."""
    want = {"OutSegs", "RetransSegs", "InSegs"}
    out: dict[str, int] = {}
    try:
        with open("/proc/net/snmp", encoding="ascii") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return out
    for i, line in enumerate(lines):
        if not line.startswith("Tcp:"):
            continue
        keys = line.split()[1:]
        if i + 1 >= len(lines):
            break
        vals = lines[i + 1].split()[1:]
        for k, v in zip(keys, vals):
            if k in want:
                out[k] = int(v)
        break
    return out


def retrans_delta(before: dict[str, int], after: dict[str, int]) -> dict:
    if not before or not after:
        return {}
    sent = after.get("OutSegs", 0) - before.get("OutSegs", 0)
    retx = after.get("RetransSegs", 0) - before.get("RetransSegs", 0)
    return {
        "out_segs": sent,
        "retrans_segs": retx,
        "retrans_pct": round(100.0 * retx / sent, 4) if sent > 0 else 0.0,
    }


def measured(fn, *args, **kwargs) -> dict:
    """Run one transfer and capture the retransmit counters alongside it."""
    c0 = tcp_counters()
    res = fn(*args, **kwargs)
    c1 = tcp_counters()
    if res.get("ok"):
        res["tcp"] = retrans_delta(c0, c1)
    return res


def sustained(ssh_host: str, seconds: int = 30) -> dict:
    """Send for a long stretch and see whether it holds: this times the sustainable rate, not completion."""
    # Offer plenty and cut it off on the clock, then compute from what went out by then.
    blocks = 20 * 1024 * 16  # 20GB cap; the clock ends it
    cmd = (f"timeout {seconds} bash -c 'dd if=/dev/zero bs=65536 count={blocks} status=none "
           f"| ssh {probe.SSH_OPTS} {ssh_host} \"cat > /dev/null\"' ; true")
    c0 = tcp_counters()
    t0 = time.perf_counter()
    probe.sh(cmd, seconds + 60)
    dt = time.perf_counter() - t0
    c1 = tcp_counters()
    d = retrans_delta(c0, c1)
    # The volume is not back-computed from kernel counters: segment counts are not
    # bytes, so the sustained rate is only a secondary figure next to the retransmit ratio.
    return {"ok": True, "seconds": round(dt, 1), "tcp": d}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--ssh-host", required=True)
    ap.add_argument("--sizes", default="1,4,16,64,128",
                    help="단일 스트림으로 재는 전송 크기 MB 목록")
    ap.add_argument("--streams", default="1,2,4,8",
                    help="같은 총량을 몇 개로 나눠 보낼지 목록")
    # Splitting a fixed total makes each transfer smaller as streams grow, which runs
    # straight back into slow start, so what is fixed here is the volume per stream.
    ap.add_argument("--stream-each", type=int, default=50,
                    help="스트림 하나가 보낼 양 MB")
    ap.add_argument("--sustain-sec", type=int, default=30)
    ap.add_argument("--repeat", type=int, default=2, help="각 점을 몇 번 반복할지")
    ap.add_argument("--parallel-host", action="append", default=[],
                    help="스트림 곡선을 분산할 호스트 목록. 독립 연결이어야 공정하다")
    args = ap.parse_args()

    rec: dict = {
        "ts": probe.now_iso(),
        "label": args.label,
        "kind": "deep",
        "target": args.ssh_host,
        "size_sweep": [],
        "stream_sweep": [],
        "sustained": {},
    }

    # Q1 - grow the size and see where throughput flattens
    for mb in [int(x) for x in args.sizes.split(",") if x]:
        runs = [measured(probe.ssh_burst, args.ssh_host, mb, "up")
                for _ in range(args.repeat)]
        ok = [r for r in runs if r.get("ok")]
        if ok:
            rec["size_sweep"].append({
                "size_mb": mb,
                "mbps_med": round(statistics.median(r["mbps"] for r in ok), 1),
                "seconds_med": round(statistics.median(r["seconds"] for r in ok), 3),
                "runs": len(ok),
                "retrans_pct": round(
                    statistics.fmean(r.get("tcp", {}).get("retrans_pct", 0.0) for r in ok), 4
                ),
            })

    # Q2 - add streams and see where it saturates
    # With a host list the streams spread over it. When measuring Pangyo behind the
    # tunnel, piling them on one host only splits one SSH session into channels.
    hosts = args.parallel_host or [args.ssh_host]
    for n in [int(x) for x in args.streams.split(",") if x]:
        runs = [measured(probe.ssh_burst_hosts, hosts, args.stream_each * n, "up", streams=n)
                for _ in range(args.repeat)]
        ok = [r for r in runs if r.get("ok")]
        if ok:
            rec["stream_sweep"].append({
                "streams": n,
                "hosts": len(set(hosts[:n])),
                "mbps_med": round(statistics.median(r["mbps"] for r in ok), 1),
                "cpu_med": round(statistics.median(r["cpu_busy_pct"] for r in ok), 1),
                "runs": len(ok),
            })

    # Q3 and Q4 - does a long send hold up, and how many retransmits appear
    if args.sustain_sec > 0:
        rec["sustained"] = sustained(args.ssh_host, args.sustain_sec)

    print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
