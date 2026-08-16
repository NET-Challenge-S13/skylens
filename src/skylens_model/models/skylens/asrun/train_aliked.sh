#!/usr/bin/env bash
D=/root/skylens/data/corridor_aliked
R=/root/skylens/results/corridor_aliked
L=/root/skylens/logs/corridor_aliked_train.log
cd /root/skylens/examples || exit 1
{
echo "======== ALIKED 3DGS 30,000 step $(date +%T) ========"
date +%s > "$D/.t_train_start"
python3 incr_trainer.py default \
  --data_dir "$D" --result_dir "$R" \
  --data_factor 1 --max_steps 30000 \
  --eval_steps 250 500 1000 2000 3500 5000 7000 10000 15000 20000 30000 \
  --ply_steps 250 500 1000 2000 3500 5000 7000 10000 15000 20000 30000 \
  --save_steps 30000 --save_ply --antialiased --disable_viewer
echo "--- rc=$? $(date +%T) ---"
ls --time-style=+%s -la "$R"/ply/*.ply 2>/dev/null | awk '{print $6,$5,$7}'
for f in "$R"/stats/val_step*.json; do [ -f "$f" ] && echo "$(basename $f): $(cat $f)"; done
echo "======== 종료 $(date +%T) ========"
} >> "$L" 2>&1
