#!/usr/bin/env python3
"""심화 프로브 — 시간마다 한 번 도는 긴 실험.

10분 주기 프로브(`probe.py`)가 추세를 보는 것이라면, 이쪽은 **곡선**을 그린다.
하루 한 점이 아니라 며칠치 곡선이 쌓여야 아래 질문에 답할 수 있다.

  Q1. 전송이 얼마나 커져야 회선 속도가 나오나 (TCP 시작 구간의 무릎이 어디인가)
  Q2. 스트림을 늘리면 어디서 포화하나 (단일 스트림이 묶여 있는가)
  Q3. 오래 보내면 유지되나, 아니면 중간에 무너지나
  Q4. 재전송이 얼마나 나나 (손실이 성능을 깎고 있는가)

사용:
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
    """커널의 TCP 누적 카운터. 전후 차이로 이번 전송의 재전송을 본다."""
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
    """전송 하나를 돌리면서 재전송 카운터를 함께 잡는다."""
    c0 = tcp_counters()
    res = fn(*args, **kwargs)
    c1 = tcp_counters()
    if res.get("ok"):
        res["tcp"] = retrans_delta(c0, c1)
    return res


def sustained(ssh_host: str, seconds: int = 30) -> dict:
    """오래 보내면서 유지되는지 본다. 완료 시간이 아니라 **지속 가능한 속도**를 잰다."""
    # 넉넉한 양을 보내고 제한 시간에 끊는다. 끊긴 시점까지 보낸 양으로 계산한다.
    blocks = 20 * 1024 * 16  # 20GB 상한. 시간으로 끊는다.
    cmd = (f"timeout {seconds} bash -c 'dd if=/dev/zero bs=65536 count={blocks} status=none "
           f"| ssh {probe.SSH_OPTS} {ssh_host} \"cat > /dev/null\"' ; true")
    c0 = tcp_counters()
    t0 = time.perf_counter()
    probe.sh(cmd, seconds + 60)
    dt = time.perf_counter() - t0
    c1 = tcp_counters()
    d = retrans_delta(c0, c1)
    # 보낸 양은 커널 카운터로 역산하지 않고, 세그먼트 수가 아니라 실제 전송량을 모르므로
    # 지속 속도는 재전송 비율과 함께 보조 지표로만 쓴다.
    return {"ok": True, "seconds": round(dt, 1), "tcp": d}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--ssh-host", required=True)
    ap.add_argument("--sizes", default="1,4,16,64,128",
                    help="단일 스트림으로 재는 전송 크기 MB 목록")
    ap.add_argument("--streams", default="1,2,4,8",
                    help="같은 총량을 몇 개로 나눠 보낼지 목록")
    # 총량을 나누면 스트림이 늘수록 각 전송이 작아져 TCP 시작 구간에 걸린다.
    # 그래서 스트림 1개가 보낼 양을 고정한다.
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

    # Q1 — 크기를 키우며 처리량이 어디서 평평해지는지
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

    # Q2 — 스트림을 늘리면 어디서 포화하는지
    # 호스트 목록이 주어지면 스트림을 거기에 나눠 보낸다. 터널 뒤의 판교를 잴 때
    # 한 호스트에 몰아주면 SSH 세션 하나의 채널로 쪼개져 병렬이 성립하지 않는다.
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

    # Q3·Q4 — 오래 보내면 유지되는가, 재전송은 얼마나 나는가
    if args.sustain_sec > 0:
        rec["sustained"] = sustained(args.ssh_host, args.sustain_sec)

    print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
