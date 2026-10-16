#!/bin/bash
# Measure the Daejeon to Pangyo leg. Run it on Daejeon.
#
# This leg is the heart of the KOREN case. Frames pulled from the drone video go
# to Pangyo and the reconstruction (.spz) plus the markers come back. That round
# trip decides how long a commander waits for new 3D.
#
# Two prerequisites, each a one-time human step; unattended afterwards:
#   1) ~/.koren_portal holds the portal account, two lines, mode 600:
#        ID=<portal id>
#        PW=<portal password>
#   2) Pangyo's authorized_keys holds the Daejeon public key:
#        on Daejeon:  cat ~/.ssh/id_ed25519.pub
#        on Pangyo :  append that line to ~/.ssh/authorized_keys
#
# Careful: today this path does not ride the KOREN backbone directly, it goes
# through a SOCKS proxy (116.89.174.24:1080). So these figures are not the
# backbone ceiling; they are the KOREN internal path we can actually use today.
# Once a direct Daejeon-Pangyo port opens, re-measure with the same script.

set -u
cd "$(dirname "$0")"

PANGYO_HOST=10.246.246.9
PANGYO_PORT=11000
HIRUN=${HIRUN:-$HOME/hirun}
OUT=${OUT:-~/lab/results/koren-$(date +%Y%m%d).jsonl}
mkdir -p ~/lab/results

if [ ! -f ~/.koren_portal ]; then
  echo "~/.koren_portal 이 없다. 포털 계정을 넣어야 무인 측정이 된다." >&2
  exit 1
fi
# shellcheck disable=SC1090
. ~/.koren_portal

# The Pangyo login is the same account as the portal. Read it from the file so
# the repository carries no account name (it is public). Override with PANGYO_USER.
PANGYO_USER=${PANGYO_USER:-$ID}

ts() { date -Is; }

# Run one command on Pangyo and hand back its stdout.
pangyo() {
  "$HIRUN" -u "$ID" -p "$PW" -c \
    "ssh -p $PANGYO_PORT -o BatchMode=yes -o StrictHostKeyChecking=no \
         -o UserKnownHostsFile=/dev/null $PANGYO_USER@$PANGYO_HOST '$1'" 2>/dev/null
}

# --- 1) round trip: how long one command takes to go and come back ---------
rtt_samples=()
for _ in $(seq 1 5); do
  t0=$(date +%s%N)
  out=$(pangyo "echo ok")
  t1=$(date +%s%N)
  [ "${out:-}" = "ok" ] && rtt_samples+=("$(( (t1 - t0) / 1000000 ))")
done

# --- 2) throughput: Pangyo -> Daejeon, Daejeon -> Pangyo ------------------
# Move as much as one .spz chunk and time it to completion.
down_mb=${1:-30}
t0=$(date +%s%N)
pangyo "dd if=/dev/zero bs=65536 count=$((down_mb * 16)) status=none" > /dev/null
t1=$(date +%s%N)
down_s=$(awk "BEGIN{printf \"%.3f\", ($t1 - $t0)/1000000000}")

t0=$(date +%s%N)
dd if=/dev/zero bs=65536 count=$((down_mb * 16)) status=none \
  | "$HIRUN" -u "$ID" -p "$PW" -c \
      "ssh -p $PANGYO_PORT -o BatchMode=yes -o StrictHostKeyChecking=no \
           -o UserKnownHostsFile=/dev/null $PANGYO_USER@$PANGYO_HOST 'cat > /dev/null'" \
      >/dev/null 2>&1
t1=$(date +%s%N)
up_s=$(awk "BEGIN{printf \"%.3f\", ($t1 - $t0)/1000000000}")

# --- 3) record ------------------------------------------------------------
python3 - "$(ts)" "${rtt_samples[*]:-}" "$down_mb" "$down_s" "$up_s" <<'PY' >> "$OUT"
import json, statistics, sys
ts, rtts, mb, down_s, up_s = sys.argv[1:6]
vals = [float(x) for x in rtts.split()] if rtts.strip() else []
mb, down_s, up_s = float(mb), float(down_s), float(up_s)
rec = {
    "ts": ts,
    "label": "koren-internal",
    "note": "대전↔판교, 현재는 SOCKS 프록시 경유",
    "rtt": {},
    "burst": [],
}
if vals:
    rec["rtt"]["pangyo_cmd_roundtrip"] = {
        "ok": True, "n": len(vals),
        "min_ms": round(min(vals), 1),
        "med_ms": round(statistics.median(vals), 1),
        "max_ms": round(max(vals), 1),
        "jitter_ms": round(statistics.pstdev(vals), 1) if len(vals) > 1 else 0.0,
    }
for direction, sec in (("down", down_s), ("up", up_s)):
    if sec > 0:
        rec["burst"].append({
            "ok": True, "size_mb": mb, "dir": direction, "streams": 1,
            "seconds": round(sec, 3), "mbps": round(mb * 8 / sec, 1),
        })
print(json.dumps(rec, ensure_ascii=False))
PY

tail -1 "$OUT"
