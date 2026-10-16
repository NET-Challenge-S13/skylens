#!/bin/bash
# The 10-minute Daejeon-to-Pangyo probe (inside KOREN), over the standing tunnels.
# Parallel measurements are spread over four tunnels. Several streams inside one
# tunnel share a single TCP connection as channels, so that is not parallel at all.
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
