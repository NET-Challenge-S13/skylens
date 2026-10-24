#!/usr/bin/env python3
"""Pull out what the aggregate tables do not show, from eleven days of measurements."""
import json, glob, statistics as st
from collections import defaultdict

recs = defaultdict(list)
for path in sorted(glob.glob("results/*.jsonl")):
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        recs[d.get("label", "?")].append(d)

def med(xs):
    return round(st.median(xs), 1) if xs else None

def pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(len(s) * p / 100))], 1)

def http_of(d, name):
    for h in d.get("http", []):
        if h.get("name") == name and h.get("ok"):
            return h
    return None

def hour(d):
    return int(d["ts"][11:13])

def day(d):
    return d["ts"][:10]

print("=" * 66)
print("1. 시간대별 국내 처리량 중앙값 (Mbps)")
print("=" * 66)
print("  시 | 대전(KOREN)  파이(상용) | 배수 | 대전 CPU%")
for h in range(24):
    dj = [x["mbps"] for d in recs["mnd-koren"] if hour(d) == h
          for x in [http_of(d, "domestic_kakao")] if x]
    pi = [x["mbps"] for d in recs["pi-175"] if hour(d) == h
          for x in [http_of(d, "domestic_kakao")] if x]
    cp = [x["cpu_busy_pct"] for d in recs["mnd-koren"] if hour(d) == h
          for x in [http_of(d, "domestic_kakao")] if x and x.get("cpu_busy_pct") is not None]
    if dj and pi:
        print("  %2d | %10s  %9s | %4.1f | %5s" %
              (h, med(dj), med(pi), med(dj) / med(pi), med(cp)))

print()
print("=" * 66)
print("2. 파이 국내 처리량 분포 — 천장인가")
print("=" * 66)
pi = [x["mbps"] for d in recs["pi-175"] for x in [http_of(d, "domestic_kakao")] if x]
dj = [x["mbps"] for d in recs["mnd-koren"] for x in [http_of(d, "domestic_kakao")] if x]
for name, xs in (("파이(상용)", pi), ("대전(KOREN)", dj)):
    print("  %-12s n=%d  p5=%s p25=%s 중앙=%s p75=%s p95=%s 최대=%s" %
          (name, len(xs), pct(xs, 5), pct(xs, 25), med(xs), pct(xs, 75), pct(xs, 95), round(max(xs), 1)))
    band = [x for x in xs if abs(x - st.median(xs)) / st.median(xs) <= 0.05]
    print("               중앙값 ±5%% 안에 든 표본: %.1f%%" % (100 * len(band) / len(xs)))

print()
print("=" * 66)
print("3. 두 회선이 같이 떨어지는가 (같은 10분 버킷)")
print("=" * 66)
bd, bp = {}, {}
for d in recs["mnd-koren"]:
    h = http_of(d, "domestic_kakao")
    if h:
        bd[d["ts"][:15]] = h["mbps"]
for d in recs["pi-175"]:
    h = http_of(d, "domestic_kakao")
    if h:
        bp[d["ts"][:15]] = h["mbps"]
both = sorted(set(bd) & set(bp))
print("  같은 시각 표본: %d쌍" % len(both))
if len(both) > 10:
    xs = [bd[k] for k in both]
    ys = [bp[k] for k in both]
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    print("  상관계수: %.3f" % (num / den if den else 0))
    dlow = {k for k in both if bd[k] < pct(xs, 10)}
    plow = {k for k in both if bp[k] < pct(ys, 10)}
    print("  대전 하위10%% 사건 %d개, 파이 하위10%% 사건 %d개, 동시 발생 %d개"
          % (len(dlow), len(plow), len(dlow & plow)))
    print("  (무관하면 기대 동시 발생은 약 %.1f개)" % (len(dlow) * len(plow) / len(both)))

print()
print("=" * 66)
print("4. 지터 꼬리 — 100ms 넘는 사건은 망의 일인가 측정기의 일인가")
print("=" * 66)
spikes = defaultdict(set)
for label in ("mnd-koren", "pi-175", "koren-internal"):
    for d in recs[label]:
        for k, v in d.get("rtt", {}).items():
            if isinstance(v, dict) and (v.get("jitter_ms") or 0) > 50:
                spikes[label].add(d["ts"][:16])
for label, ts in spikes.items():
    print("  %-16s 50ms 초과 지터가 난 시각 %d개" % (label, len(ts)))
allts = set().union(*spikes.values()) if spikes else set()
multi = [t for t in allts if sum(t in s for s in spikes.values()) >= 2]
print("  두 노드 이상에서 같은 분에 동시 발생: %d개 / 전체 %d개" % (len(multi), len(allts)))
if multi:
    print("  예시:", sorted(multi)[:6])

print()
print("=" * 66)
print("5. 작은 전송의 한계 — 바이트인가 왕복인가")
print("=" * 66)
for label, rttkey in (("koren-internal", "pangyo_via_proxy"), ("pi-175", "daejeon_ssh")):
    rtts = [v["med_ms"] for d in recs[label] for k, v in d.get("rtt", {}).items()
            if k == rttkey and isinstance(v, dict) and v.get("med_ms")]
    if not rtts:
        continue
    r = st.median(rtts)
    print("  [%s] 왕복시간 중앙 %.1f ms" % (label, r))
    for size in (4.0, 30.0, 100.0):
        for dirn in ("up", "down"):
            sec = [b["seconds"] for d in recs[label] for b in d.get("burst", [])
                   if b.get("ok") and b.get("size_mb") == size and b.get("dir") == dirn]
            mbps = [b["mbps"] for d in recs[label] for b in d.get("burst", [])
                    if b.get("ok") and b.get("size_mb") == size and b.get("dir") == dirn]
            if sec:
                s = st.median(sec)
                print("    %6.1fMB %-4s 완료 %.2f s · %7.1f Mbps · 왕복이 차지하는 비중 %.0f%%"
                      % (size, dirn, s, st.median(mbps), 100 * (r / 1000) / s))

print()
print("=" * 66)
print("6. 요일별 — 업무일에 좁혀지는가")
print("=" * 66)
import datetime
wk = defaultdict(lambda: ([], []))
for d in recs["mnd-koren"]:
    h = http_of(d, "domestic_kakao")
    if h:
        wk[datetime.date.fromisoformat(day(d)).weekday()][0].append(h["mbps"])
for d in recs["pi-175"]:
    h = http_of(d, "domestic_kakao")
    if h:
        wk[datetime.date.fromisoformat(day(d)).weekday()][1].append(h["mbps"])
names = "월화수목금토일"
for i in range(7):
    a, b = wk[i]
    if a and b:
        print("  %s  대전 %7s   파이 %6s   배수 %.1f  (표본 %d/%d)"
              % (names[i], med(a), med(b), med(a) / med(b), len(a), len(b)))

print()
print("=" * 66)
print("7. 해외 구간 — 불리한 숫자의 안정성")
print("=" * 66)
for label in ("mnd-koren", "pi-175"):
    xs = [x["mbps"] for d in recs[label] for x in [http_of(d, "overseas_ovh")] if x]
    fails = sum(1 for d in recs[label] for h in d.get("http", [])
                if h.get("name") == "overseas_ovh" and not h.get("ok"))
    if xs:
        print("  %-12s 중앙 %7s  p5 %6s  p95 %7s  실패 %d회 / %d회"
              % (label, med(xs), pct(xs, 5), pct(xs, 95), fails, len(xs) + fails))
