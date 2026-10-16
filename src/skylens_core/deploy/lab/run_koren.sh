#!/bin/bash
# 대전 ↔ 판교(KOREN 내부) 10분 주기 프로브. 상시 터널 위에서 돈다.
# 병렬 측정은 터널 4개에 나눠 보낸다. 한 터널에 스트림을 여러 개 주면
# SSH 세션 하나 안에서 채널만 나뉘어 TCP 연결 하나를 공유하므로 병렬이 아니다.
cd "$(dirname "$0")"
exec >> ~/lab/results/koren-$(date +%Y%m%d).jsonl 2>> ~/lab/results/koren-error.log
ss -ltn | grep -q 19022 || { echo "{\"ts\": \"$(date -Is)\", \"label\": \"koren-internal\", \"error\": \"tunnel down\"}"; exit 0; }
timeout 600 python3 probe.py \
  --label koren-internal \
  --banner 127.0.0.1:19022:pangyo_via_proxy \
  --rtt 116.89.174.24:1080:socks_proxy_tcp \
  --ssh-host pangyo \
  --burst-mb 4,30 \
  --parallel 4 --parallel-mb 100 \
  --parallel-host pangyo --parallel-host pangyo2 \
  --parallel-host pangyo3 --parallel-host pangyo4
