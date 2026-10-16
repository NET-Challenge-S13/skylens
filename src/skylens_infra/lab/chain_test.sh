#!/bin/bash
# 구간별 측정과 전체 경로 측정을 나란히 놓는다.
#   A: 파이 → 대전        (상용 회선 구간)
#   B: 대전 → 판교        (KOREN 내부 구간, 프록시 경유)
#   A+B: 파이 → 대전 → 판교 (실제 파이프라인 경로)
# 전체가 둘 중 약한 쪽과 같은지, 아니면 이어붙이며 더 깎이는지 본다.
set -u
MB=${1:-50}
CNT=$((MB*16))

echo "=== A) 파이 → 대전 ==="
S=$(date +%s%N)
dd if=/dev/zero bs=65536 count=$CNT status=none | ssh -o BatchMode=yes -o Compression=no mnd "cat > /dev/null"
E=$(date +%s%N)
awk -v s=$S -v e=$E -v m=$MB 'BEGIN{d=(e-s)/1e9; printf "  %.2f초  %.1f Mbps\n", d, m*8/d}'

echo "=== B) 대전 → 판교 (대전에서 실행) ==="
ssh -o BatchMode=yes mnd "S=\$(date +%s%N); dd if=/dev/zero bs=65536 count=$CNT status=none | ssh -o BatchMode=yes -o Compression=no pangyo 'cat > /dev/null'; E=\$(date +%s%N); awk -v s=\$S -v e=\$E -v m=$MB 'BEGIN{d=(e-s)/1e9; printf \"  %.2f초  %.1f Mbps\n\", d, m*8/d}'" 2>/dev/null

echo "=== A+B) 파이 → 대전 → 판교 (이어붙인 실제 경로) ==="
S=$(date +%s%N)
dd if=/dev/zero bs=65536 count=$CNT status=none \
  | ssh -o BatchMode=yes -o Compression=no mnd "ssh -o BatchMode=yes -o Compression=no pangyo 'cat > /dev/null'" 2>/dev/null
E=$(date +%s%N)
awk -v s=$S -v e=$E -v m=$MB 'BEGIN{d=(e-s)/1e9; printf "  %.2f초  %.1f Mbps\n", d, m*8/d}'
