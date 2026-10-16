#!/bin/bash
# 독립 터널 수를 늘리며 합계 처리량이 어디서 평평해지는지 본다.
# 각 터널은 프록시를 통과하는 별개 연결이다. 한 터널에 여러 스트림을 태우면
# 결국 한 연결 안에서 나눠 쓰는 것이라 의미가 다르다.
set -u
. ~/.koren_portal
BASE=19030
MAXN=${1:-8}
MB=${2:-50}

for i in $(seq 0 $((MAXN-1))); do
  p=$((BASE+i))
  pgrep -f "L ${p}:" >/dev/null || (LOCAL_PORT=$p setsid nohup ~/lab/pangyo_tunnel.sh < /dev/null > /tmp/tun-$p.log 2>&1 &)
done
sleep 25
UP=$(ss -ltn | grep -cE "127.0.0.1:190[3-9][0-9]")
echo "터널 $UP 개 준비"

for N in 1 2 4 8; do
  [ "$N" -gt "$MAXN" ] && continue
  C0=$(awk '/^cpu /{print $2+$3+$4+$6+$7+$8, $5}' /proc/stat)
  S=$(date +%s%N)
  for i in $(seq 0 $((N-1))); do
    p=$((BASE+i))
    ( dd if=/dev/zero bs=65536 count=$((MB*16)) status=none \
        | ssh -o BatchMode=yes -o Compression=no -o StrictHostKeyChecking=no \
              -o UserKnownHostsFile=/dev/null -p $p "${PANGYO_USER:-$ID}@127.0.0.1" "cat > /dev/null" 2>/dev/null ) &
  done
  wait
  E=$(date +%s%N)
  C1=$(awk '/^cpu /{print $2+$3+$4+$6+$7+$8, $5}' /proc/stat)
  awk -v s=$S -v e=$E -v t=$((MB*N)) -v n=$N -v c0="$C0" -v c1="$C1" 'BEGIN{
    split(c0,a," "); split(c1,b," ");
    busy=b[1]-a[1]; idle=b[2]-a[2]; tot=busy+idle;
    d=(e-s)/1e9;
    printf "  터널 %d개  %6.2f초  합계 %7.1f Mbps  (터널당 %6.1f)  CPU %4.1f%%\n", n, d, t*8/d, t*8/d/n, (tot>0? 100*busy/tot : 0)
  }'
done

# 측정이 끝나면 임시 터널을 내린다. 예약된 10분 프로브와 섞이지 않게 한다.
for i in $(seq 0 $((MAXN-1))); do pkill -f "L $((BASE+i)):" 2>/dev/null; done
