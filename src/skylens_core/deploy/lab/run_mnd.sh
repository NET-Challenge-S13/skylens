#!/bin/bash
# 대전(KOREN 내부)에서 도는 10분 주기 프로브.
cd "$(dirname "$0")"
OUT=~/lab/results/mnd-$(date +%Y%m%d).jsonl
exec >> "$OUT" 2>> ~/lab/results/mnd-error.log
timeout 900 python3 probe.py \
  --label mnd-koren \
  --rtt 1.1.1.1:443:cloudflare_anycast \
  --rtt 8.8.8.8:443:google_anycast \
  --rtt 175.126.98.44:80:bridge_http \
  --skip-burst \
  --http-mb 50 \
  --http domestic_kakao=http://mirror.kakao.com/ubuntu/ls-lR.gz \
  --http overseas_ovh=https://proof.ovh.net/files/100Mb.dat
