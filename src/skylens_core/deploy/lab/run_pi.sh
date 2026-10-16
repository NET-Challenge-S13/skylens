#!/bin/bash
# 파이(상용 회선 · KOREN 등록 IP)에서 도는 10분 주기 프로브.
cd "$(dirname "$0")"
OUT=~/lab/results/pi-$(date +%Y%m%d).jsonl
exec >> "$OUT" 2>> ~/lab/results/pi-error.log
timeout 900 python3 probe.py \
  --label pi-175 \
  --rtt 116.89.187.181:26022:daejeon_ssh \
  --rtt 1.1.1.1:443:cloudflare_anycast \
  --rtt 8.8.8.8:443:google_anycast \
  --ssh-host mnd \
  --burst-mb 4,30 \
  --parallel 4 --parallel-mb 100 \
  --http-mb 50 \
  --http domestic_kakao=http://mirror.kakao.com/ubuntu/ls-lR.gz \
  --http overseas_ovh=https://proof.ovh.net/files/100Mb.dat
