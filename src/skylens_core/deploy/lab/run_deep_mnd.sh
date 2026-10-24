#!/bin/bash
# The deeper Daejeon-to-Pangyo measurement (inside KOREN), once an hour.
# The stream curve is spread one per tunnel across eight; two in one tunnel would not be independent.
cd /home/ubuntu/lab || exit 1
ss -ltn | grep -q 19022 || exit 0
exec >> /home/ubuntu/lab/results/deep-koren-$(date +%Y%m%d).jsonl 2>> /home/ubuntu/lab/results/deep-error.log
timeout 1800 python3 deep.py --label koren-internal --ssh-host pangyo \
  --parallel-host pangyo --parallel-host pangyo2 --parallel-host pangyo3 --parallel-host pangyo4 \
  --parallel-host pangyo5 --parallel-host pangyo6 --parallel-host pangyo7 --parallel-host pangyo8
