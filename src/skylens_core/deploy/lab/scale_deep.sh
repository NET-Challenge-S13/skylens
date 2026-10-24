#!/bin/bash
# Find the saturation point: raise the number of independent connections from 1
# to N and watch where the summed throughput bends.
# Why this exists
#   The six-hourly run only goes to 8 connections, and the KOREN path has not
#   bent by then - it is still climbing nearly linearly. To know whether the
#   "2.6x" we quote is a floor, we have to push further.
# Careful
#   Each tunnel makes hirun authenticate over HTTP once, so bringing them all up
#   at once leans on the portal; space them out. Everything temporary comes down
#   at the end (the standing tunnels 19022-19025 are left alone). CPU usage is
#   recorded too: if the CPU is saturated the number is the VM, not the line.

set -u
cd "$(dirname "$0")" || exit 1
. ~/.koren_portal

BASE=19100
MAXN=${1:-16}
MB=${2:-50}          # per connection, not the total
STEPS=${3:-"1 2 4 8 16"}

echo "===== $(date -Is) · 포화점 탐색 (최대 ${MAXN}연결, 연결당 ${MB}MB) ====="

# --- bring the tunnels up (spaced out) -----------------------------------
started=0
for i in $(seq 0 $((MAXN-1))); do
  p=$((BASE+i))
  if ! ss -ltn | grep -q "127.0.0.1:$p "; then
    LOCAL_PORT=$p setsid nohup ./pangyo_tunnel.sh < /dev/null > /tmp/tun-$p.log 2>&1 &
    started=$((started+1))
    sleep 0.4
  fi
done
sleep 30
up=$(ss -ltn | grep -cE "127.0.0.1:19(1[0-9][0-9])")
echo "터널 ${up}/${MAXN} 준비 (새로 올린 것 ${started}개)"
[ "$up" -lt 2 ] && { echo "터널이 모자라 중단한다"; exit 1; }

cpu() { awk '/^cpu /{print $2+$3+$4+$6+$7+$8, $5}' /proc/stat; }

for N in $STEPS; do
  [ "$N" -gt "$up" ] && continue
  C0=$(cpu); S=$(date +%s%N)
  pids=""
  for i in $(seq 0 $((N-1))); do
    p=$((BASE+i))
    ( dd if=/dev/zero bs=65536 count=$((MB*16)) status=none \
        | ssh -o BatchMode=yes -o Compression=no -o StrictHostKeyChecking=no \
              -o UserKnownHostsFile=/dev/null -p $p "${PANGYO_USER:-$ID}@127.0.0.1" "cat > /dev/null" 2>/dev/null ) &
    pids="$pids $!"
  done
  # The tunnels are children of this shell too, so a bare wait never returns; wait on the transfer PIDs only.
  for q in $pids; do wait "$q"; done
  E=$(date +%s%N); C1=$(cpu)
  awk -v s="$S" -v e="$E" -v t=$((MB*N)) -v n="$N" -v c0="$C0" -v c1="$C1" 'BEGIN{
    split(c0,a," "); split(c1,b," ");
    busy=b[1]-a[1]; idle=b[2]-a[2]; tot=busy+idle; d=(e-s)/1e9;
    printf "  %2d연결  %6.2f초  합계 %7.1f Mbps  연결당 %6.1f  CPU %4.1f%%\n", n, d, t*8/d, t*8/d/n, (tot>0? 100*busy/tot : 0)
  }'
  sleep 3
done

# --- take the tunnels down -----------------------------------------------
for i in $(seq 0 $((MAXN-1))); do pkill -f "L $((BASE+i)):" 2>/dev/null; done
sleep 2
echo "임시 터널 정리 완료 (상시 터널 $(ss -ltn | grep -cE "127.0.0.1:1902[2-5]")/4 유지)"
