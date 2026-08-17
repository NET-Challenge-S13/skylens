#!/usr/bin/env bash
# ALIKED COLMAP 완료 대기 → 검증 → 3DGS 30,000 step 학습 (단계별 산출)
A=/root/skylens/logs/corridor_aliked.log
D=/root/skylens/data/corridor_aliked
R=/root/skylens/results/corridor_aliked
L=/root/skylens/logs/corridor_aliked_train.log
mkdir -p /root/skylens/logs
{
echo "======== ALIKED 완료 대기 $(date +%T) ========"
for i in $(seq 1 360); do
  if grep -q "종료" "$A" 2>/dev/null; then echo "COLMAP 완료 감지 ($((i*20))초 대기)"; break; fi
  if ! pgrep -f "colmap-cuda.*mapper" >/dev/null; then echo "mapper 소실 — 확인 필요"; break; fi
  sleep 20
done

N=$(ls "$D/sparse/" 2>/dev/null | wc -l)
echo "sparse 모델 개수: $N"
if [ ! -d "$D/sparse/0" ]; then echo "sparse/0 없음 — 학습 중단"; exit 1; fi
/root/miniconda3/envs/colmap-cuda/bin/colmap model_analyzer --path "$D/sparse/0" 2>&1 \
  | grep -viE '^I2026|^W2026' | grep -E "Registered|Points:|Mean track|reprojection"

echo "======== 3DGS 학습 30,000 step (단계별 산출) $(date +%T) ========"
date +%s > "$D/.t_train_start"
cd /root/skylens/examples || exit 1
python3 incr_trainer.py default \
  --data_dir "$D" --result_dir "$R" \
  --data_factor 1 --max_steps 30000 \
  --eval_steps 250 500 1000 2000 3500 5000 7000 10000 15000 20000 30000 \
  --ply_steps 250 500 1000 2000 3500 5000 7000 10000 15000 20000 30000 \
  --save_steps 30000 --save_ply --antialiased --disable_viewer
echo "--- 학습 rc=$? $(date +%T) ---"

echo "--- 단계별 산출물 (시각, 크기) ---"
ls --time-style=+%s -la "$R"/ply/*.ply 2>/dev/null | awk '{print $6, $5, $7}'
echo "--- 단계별 지표 ---"
for f in "$R"/stats/val_step*.json; do [ -f "$f" ] && echo "$(basename "$f"): $(cat "$f")"; done
echo "======== 전체 종료 $(date +%T) ========"
} >> "$L" 2>&1
