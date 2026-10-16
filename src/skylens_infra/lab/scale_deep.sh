#!/bin/bash
# 포화점 탐색. 독립 연결 수를 1~N 까지 늘리며 합계 처리량이 어디서 꺾이는지 본다.
#
# 왜 필요한가
#   6시간 주기 시험은 8연결까지만 본다. 그런데 KOREN 경로는 8에서도 꺾이지 않고
#   선형으로 늘어난다. 지금 쓰는 "2.6배"가 하한인지 아닌지 알려면 더 밀어봐야 한다.
#
# 주의
#   터널 하나당 hirun 이 HTTP 인증을 한 번 한다. 한꺼번에 띄우면 포털에 부담이 되므로
#   간격을 두고 올린다. 측정이 끝나면 전부 내린다(상시 터널 19022~19025 는 건드리지 않는다).
#   CPU 점유율을 함께 기록한다. CPU 가 포화면 그 숫자는 회선이 아니라 VM 의 한계다.

set -u
cd "$(dirname "$0")" || exit 1
. ~/.koren_portal

BASE=19100
MAXN=${1:-16}
MB=${2:-50}          # 연결 하나가 보낼 양. 총량이 아니라 연결당으로 고정한다.
STEPS=${3:-"1 2 4 8 16"}

echo "===== $(date -Is) · 포화점 탐색 (최대 ${MAXN}연결, 연결당 ${MB}MB) ====="

# --- 터널 올리기 (간격을 두고) -------------------------------------------
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
  # 터널도 이 셸의 자식이라 인자 없는 wait 는 영원히 멈춘다. 전송 PID 만 기다린다.
  for q in $pids; do wait "$q"; done
  E=$(date +%s%N); C1=$(cpu)
  awk -v s="$S" -v e="$E" -v t=$((MB*N)) -v n="$N" -v c0="$C0" -v c1="$C1" 'BEGIN{
    split(c0,a," "); split(c1,b," ");
    busy=b[1]-a[1]; idle=b[2]-a[2]; tot=busy+idle; d=(e-s)/1e9;
    printf "  %2d연결  %6.2f초  합계 %7.1f Mbps  연결당 %6.1f  CPU %4.1f%%\n", n, d, t*8/d, t*8/d/n, (tot>0? 100*busy/tot : 0)
  }'
  sleep 3
done

# --- 터널 내리기 ---------------------------------------------------------
for i in $(seq 0 $((MAXN-1))); do pkill -f "L $((BASE+i)):" 2>/dev/null; done
sleep 2
echo "임시 터널 정리 완료 (상시 터널 $(ss -ltn | grep -cE "127.0.0.1:1902[2-5]")/4 유지)"
