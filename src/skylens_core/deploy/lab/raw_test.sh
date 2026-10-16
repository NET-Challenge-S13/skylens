#!/bin/bash
# hirun 으로 암호화 없는 TCP 전송을 재서, SSH 측정에 섞인 암호화 비용을 분리한다.
set -u
. ~/.koren_portal
MB=${1:-200}
N=${2:-1}
echo "=== 연결 확인 ==="
timeout 20 ~/hirun -u "$ID" -p "$PW" -c "nc -z -w 8 10.246.246.9 15201" >/dev/null 2>&1
echo "  종료코드 $?"
echo "=== ${N} 스트림 × ${MB}MB, 암호화 없음 ==="
S=$(date +%s%N)
for i in $(seq 1 "$N"); do
  ( dd if=/dev/zero bs=65536 count=$((MB*16)) status=none \
      | timeout 300 ~/hirun -u "$ID" -p "$PW" -c "nc -N 10.246.246.9 15201" >/dev/null 2>&1 ) &
done
wait
E=$(date +%s%N)
awk -v s=$S -v e=$E -v t=$((MB*N)) 'BEGIN{d=(e-s)/1e9; printf "  %.2f초  합계 %.1f Mbps\n", d, t*8/d}'
