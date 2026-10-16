#!/bin/bash
# 파이 → 대전(상용 회선) 심화 측정. 시간마다 한 번. 대전 쪽과 30분 어긋나게 돈다.
cd "$(dirname "$0")"
exec >> ~/lab/results/deep-pi-$(date +%Y%m%d).jsonl 2>> ~/lab/results/deep-error.log
timeout 1500 python3 deep.py --label pi-175 --ssh-host mnd
