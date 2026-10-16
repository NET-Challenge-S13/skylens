#!/bin/bash
# Raise the number of independent tunnels and watch where the summed throughput
# flattens. Each tunnel is its own connection through the proxy; several streams
# inside one tunnel would only share that one connection, which measures something else.
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

# Take the temporary tunnels down so they do not mix with the scheduled 10-minute probe.
for i in $(seq 0 $((MAXN-1))); do pkill -f "L $((BASE+i)):" 2>/dev/null; done
