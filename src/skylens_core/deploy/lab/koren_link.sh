#!/bin/bash
# 대전 ↔ 판교 구간 측정. 대전에서 실행한다.
#
# 이 구간이 KOREN 정당성의 본체다. 드론 영상에서 뽑은 프레임이 판교로 가고
# 복원 결과(.spz)와 마커가 대전으로 돌아온다. 그 왕복 시간이 지휘관이
# 새 3D 를 보기까지의 시간을 정한다.
#
# 전제 두 가지(각각 한 번만 사람이 해 주면 그 뒤로는 무인 실행)
#   1) ~/.koren_portal 에 포털 계정이 있다. 두 줄, 권한 600.
#        ID=<포털 아이디>
#        PW=<포털 비밀번호>
#   2) 판교의 authorized_keys 에 대전의 공개키가 들어 있다.
#        대전에서:  cat ~/.ssh/id_ed25519.pub
#        판교에서:  그 줄을 ~/.ssh/authorized_keys 에 추가
#
# 주의: 현재 이 경로는 KOREN 백본을 직접 타지 않고 SOCKS 프록시
# (116.89.174.24:1080)를 거친다. 따라서 여기서 나온 수치는 백본의 상한이
# 아니라 "오늘 우리가 실제로 쓸 수 있는 KOREN 내부 경로"의 값이다.
# 대전↔판교 직결 포트가 열리면 같은 스크립트로 다시 재서 비교한다.

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

# 판교 로그인 계정은 포털 계정과 같다. 저장소에 계정명을 남기지 않기 위해
# 여기서 받아온다(공개 저장소다). 다르면 PANGYO_USER 를 환경변수로 넘긴다.
PANGYO_USER=${PANGYO_USER:-$ID}

ts() { date -Is; }

# 판교에서 명령 하나를 돌리고 표준출력을 돌려준다.
pangyo() {
  "$HIRUN" -u "$ID" -p "$PW" -c \
    "ssh -p $PANGYO_PORT -o BatchMode=yes -o StrictHostKeyChecking=no \
         -o UserKnownHostsFile=/dev/null $PANGYO_USER@$PANGYO_HOST '$1'" 2>/dev/null
}

# --- 1) 왕복시간: 명령 하나를 왕복시키는 데 걸리는 시간 -------------------
rtt_samples=()
for _ in $(seq 1 5); do
  t0=$(date +%s%N)
  out=$(pangyo "echo ok")
  t1=$(date +%s%N)
  [ "${out:-}" = "ok" ] && rtt_samples+=("$(( (t1 - t0) / 1000000 ))")
done

# --- 2) 처리량: 판교 → 대전, 대전 → 판교 --------------------------------
# .spz 한 덩어리에 해당하는 양을 실제로 옮겨 완료 시간을 잰다.
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

# --- 3) 기록 -------------------------------------------------------------
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
