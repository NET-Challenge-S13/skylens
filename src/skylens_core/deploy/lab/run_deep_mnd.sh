#!/bin/bash
# 대전 → 판교(KOREN 내부) 심화 측정. 시간마다 한 번.
# 스트림 곡선은 터널 8개에 하나씩 분산한다. 한 터널에 둘 이상 태우면 그 둘은 독립 연결이 아니다.
cd /home/ubuntu/lab || exit 1
ss -ltn | grep -q 19022 || exit 0
exec >> /home/ubuntu/lab/results/deep-koren-$(date +%Y%m%d).jsonl 2>> /home/ubuntu/lab/results/deep-error.log
timeout 1800 python3 deep.py --label koren-internal --ssh-host pangyo \
  --parallel-host pangyo --parallel-host pangyo2 --parallel-host pangyo3 --parallel-host pangyo4 \
  --parallel-host pangyo5 --parallel-host pangyo6 --parallel-host pangyo7 --parallel-host pangyo8
