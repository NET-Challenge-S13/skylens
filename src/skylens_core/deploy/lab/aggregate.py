#!/usr/bin/env python3
"""Aggregate the measurements: read the JSONL and table the distribution per node and metric.

No single average. What matters in disaster response is not the mean but the bad
tail (p95) and the wobble (jitter).

Usage:
    python3 aggregate.py ~/lab/results/*.jsonl              # 전체
    python3 aggregate.py --since 2026-10-05 ~/lab/results/*.jsonl
    python3 aggregate.py --md ~/lab/results/*.jsonl         # 마크다운 표로
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * p))
    return ordered[idx]


def summarize(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "n": len(values),
        "min": round(min(values), 1),
        "med": round(statistics.median(values), 1),
        "p95": round(pct(values, 0.95), 1),
        "max": round(max(values), 1),
        "cv": round(statistics.pstdev(values) / statistics.fmean(values), 3)
        if statistics.fmean(values)
        else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--since", help="YYYY-MM-DD 이후만")
    ap.add_argument("--md", action="store_true", help="마크다운 표로 출력")
    args = ap.parse_args()

    # (label, metric) -> list of values
    rtt: dict[tuple[str, str], list[float]] = defaultdict(list)
    jit: dict[tuple[str, str], list[float]] = defaultdict(list)
    burst: dict[tuple[str, str], list[float]] = defaultdict(list)
    bsec: dict[tuple[str, str], list[float]] = defaultdict(list)
    http: dict[tuple[str, str], list[float]] = defaultdict(list)
    span: list[str] = []

    for path in args.files:
        try:
            fh = open(path, encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = rec.get("ts", "")
                if args.since and ts[:10] < args.since:
                    continue
                span.append(ts)
                label = rec.get("label", "?")
                for name, v in (rec.get("rtt") or {}).items():
                    if v.get("ok"):
                        rtt[(label, name)].append(v["med_ms"])
                        jit[(label, name)].append(v.get("jitter_ms", 0.0))
                for b in rec.get("burst") or []:
                    if b.get("ok"):
                        key = (label, f'{b["size_mb"]}MB {b["dir"]} x{b.get("streams",1)}')
                        burst[key].append(b["mbps"])
                        bsec[key].append(b["seconds"])
                for h in rec.get("http") or []:
                    if h.get("ok"):
                        http[(label, h.get("name", h.get("url", "?")))].append(h["mbps"])

    if not span:
        print("표본이 없다.")
        return

    span.sort()
    bar = "|" if args.md else ""

    def row(cells: list[str]) -> str:
        if args.md:
            return "| " + " | ".join(cells) + " |"
        return "  ".join(c.ljust(w) for c, w in zip(cells, [28, 8, 8, 8, 8, 8, 6]))

    print(f"\n표본 {len(span)}개 · {span[0]} ~ {span[-1]}\n")

    def block(title: str, data: dict, unit: str, lower_is_better: bool) -> None:
        if not data:
            return
        print(f"### {title} ({unit})\n")
        head = ["지점 · 항목", "표본", "최소", "중앙", "p95", "최대", "변동"]
        print(row(head))
        if args.md:
            print("|" + "|".join(["---"] * len(head)) + "|")
        for (label, name), values in sorted(data.items()):
            s = summarize(values)
            print(row([f"{label} · {name}", str(s["n"]), str(s["min"]),
                       str(s["med"]), str(s["p95"]), str(s["max"]), str(s["cv"])]))
        note = "작을수록 좋다" if lower_is_better else "클수록 좋다"
        print(f"\n({note}. 변동은 표준편차/평균)\n")

    block("왕복시간", rtt, "ms", True)
    block("지터", jit, "ms", True)
    block("SSH 버스트 처리량", burst, "Mbps", False)
    block("SSH 버스트 완료 시간", bsec, "s", True)
    block("공개 기준선 처리량", http, "Mbps", False)


if __name__ == "__main__":
    main()
