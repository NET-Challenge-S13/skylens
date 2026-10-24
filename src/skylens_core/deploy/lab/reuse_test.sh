#!/bin/bash
# Does reusing a connection remove the cost of TCP's opening phase?
#
# Background
#   On the same line 4MB came out at 50 Mbps and 30MB at 181 Mbps (measured
#   10-05). Small transfers are dominated by the TCP opening phase, not the line.
#   One .spz is about 4MB, so a fresh connection per chunk pays that every time.
#
# The question
#   If the connection stays up and only the chunks follow, does the cost vanish?
#
# Three conditions, same total (4MB x N)
#   A. a new connection each time - a TCP and SSH handshake per chunk
#   B. reuse the connection      - new channels over one TCP connection, the
#                                  shape HTTP keep-alive has
#   C. one transfer              - the same total in a single stream, the ceiling
#
# A vs B is the cost of the connection; B vs C is the cost of chunking.

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

# --- A. a new connection each time ---------------------------------------
S=$(date +%s%N)
for _ in $(seq 1 "$N"); do
  dd if=/dev/zero bs=65536 count=$blocks status=none \
    | ssh $opts "$HOST" "cat > /dev/null" 2>/dev/null
done
E=$(date +%s%N)
A=$(awk -v s=$S -v e=$E 'BEGIN{printf "%.3f", (e-s)/1e9}')
say "A 매번 새 연결" "$A" "$(awk -v t=$total -v d=$A 'BEGIN{printf "%.1f", t*8/d}')" \
    "$(awk -v d=$A -v n=$N 'BEGIN{printf "%.0f", d*1000/n}')"

# --- B. reuse the connection ---------------------------------------------
# Bring up a master connection first; later ssh calls only add a channel to it.
ssh $opts -o ControlMaster=yes -o ControlPath=$CM -o ControlPersist=120 -N -f "$HOST" 2>/dev/null
sleep 2
ssh $opts -o ControlPath=$CM "$HOST" "true" 2>/dev/null   # warm-up: grow the window first
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

# --- C. one transfer -----------------------------------------------------
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
