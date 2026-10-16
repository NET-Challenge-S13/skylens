#!/bin/bash
# 연결 재사용이 TCP 시작 구간 비용을 없애는가.
#
# 배경
#   같은 회선에서 4MB 는 50 Mbps, 30MB 는 181 Mbps 가 나왔다(10-05 실측). 작은 전송은
#   회선이 아니라 TCP 시작 구간이 지배한다는 뜻이다. `.spz` 한 장이 4MB 급이므로,
#   구간마다 새 연결로 보내면 이 비용을 매번 치른다.
#
# 묻는 것
#   연결을 살려 두고 조각만 이어 보내면 그 비용이 사라지는가.
#
# 세 조건, 같은 총량(4MB × N)
#   A. 매번 새 연결      — 조각마다 TCP·SSH 핸드셰이크를 새로 한다
#   B. 연결 재사용        — 하나의 TCP 연결 위에 채널만 새로 연다(HTTP keep-alive 와 같은 꼴)
#   C. 한 번에 전송       — 같은 총량을 한 스트림으로. 이론적 상한
#
# A 와 B 의 차이가 "연결 비용", B 와 C 의 차이가 "조각으로 나눈 비용"이다.

set -u
HOST=${1:-mnd}
MB=${2:-4}
N=${3:-10}
CM=/tmp/cm-reuse-$$

opts="-o BatchMode=yes -o Compression=no"
blocks=$((MB * 16))
total=$((MB * N))

say() { printf "  %-22s %7.2f초  %7.1f Mbps  조각당 %6.0f ms\n" "$1" "$2" "$3" "$4"; }

echo "===== $(date -Is) · 연결 재사용 시험 ($HOST, ${MB}MB × ${N}회) ====="

# --- A. 매번 새 연결 -----------------------------------------------------
S=$(date +%s%N)
for _ in $(seq 1 "$N"); do
  dd if=/dev/zero bs=65536 count=$blocks status=none \
    | ssh $opts "$HOST" "cat > /dev/null" 2>/dev/null
done
E=$(date +%s%N)
A=$(awk -v s=$S -v e=$E 'BEGIN{printf "%.3f", (e-s)/1e9}')
say "A 매번 새 연결" "$A" "$(awk -v t=$total -v d=$A 'BEGIN{printf "%.1f", t*8/d}')" \
    "$(awk -v d=$A -v n=$N 'BEGIN{printf "%.0f", d*1000/n}')"

# --- B. 연결 재사용 ------------------------------------------------------
# 먼저 마스터 연결을 띄워 둔다. 이후 ssh 호출은 이 TCP 연결에 채널만 추가한다.
ssh $opts -o ControlMaster=yes -o ControlPath=$CM -o ControlPersist=120 -N -f "$HOST" 2>/dev/null
sleep 2
ssh $opts -o ControlPath=$CM "$HOST" "true" 2>/dev/null   # 예열: 창을 미리 키운다
S=$(date +%s%N)
for _ in $(seq 1 "$N"); do
  dd if=/dev/zero bs=65536 count=$blocks status=none \
    | ssh $opts -o ControlPath=$CM "$HOST" "cat > /dev/null" 2>/dev/null
done
E=$(date +%s%N)
B=$(awk -v s=$S -v e=$E 'BEGIN{printf "%.3f", (e-s)/1e9}')
say "B 연결 재사용" "$B" "$(awk -v t=$total -v d=$B 'BEGIN{printf "%.1f", t*8/d}')" \
    "$(awk -v d=$B -v n=$N 'BEGIN{printf "%.0f", d*1000/n}')"
ssh $opts -o ControlPath=$CM -O exit "$HOST" 2>/dev/null

# --- C. 한 번에 전송 -----------------------------------------------------
S=$(date +%s%N)
dd if=/dev/zero bs=65536 count=$((blocks * N)) status=none \
  | ssh $opts "$HOST" "cat > /dev/null" 2>/dev/null
E=$(date +%s%N)
C=$(awk -v s=$S -v e=$E 'BEGIN{printf "%.3f", (e-s)/1e9}')
say "C 한 번에 전송" "$C" "$(awk -v t=$total -v d=$C 'BEGIN{printf "%.1f", t*8/d}')" \
    "$(awk -v d=$C -v n=$N 'BEGIN{printf "%.0f", d*1000/n}')"

awk -v a="$A" -v b="$B" -v c="$C" 'BEGIN{
  printf "  → 재사용으로 줄어든 시간 %.1f%%, 한 번에 보내면 %.1f%%\n", (a-b)/a*100, (a-c)/a*100
}'
