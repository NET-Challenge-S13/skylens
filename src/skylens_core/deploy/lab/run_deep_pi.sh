#!/bin/bash
# The deeper pi-to-Daejeon measurement (consumer line), once an hour, offset 30 minutes from the Daejeon one.
cd "$(dirname "$0")"
exec >> ~/lab/results/deep-pi-$(date +%Y%m%d).jsonl 2>> ~/lab/results/deep-error.log
timeout 1500 python3 deep.py --label pi-175 --ssh-host mnd
