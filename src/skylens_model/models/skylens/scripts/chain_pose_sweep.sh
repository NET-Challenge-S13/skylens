#!/usr/bin/env bash
# ALIKED 30k 학습 완료 대기 → 포즈 정확도 민감도 8조건 순차 실행
SRC=/root/skylens/data/corridor_aliked/sparse/0
IMG=/root/skylens/data/corridor_hq/images
TL=/root/skylens/logs/corridor_aliked_train.log
L=/root/skylens/logs/pose_sweep_local.log
mkdir -p /root/skylens/logs
{
echo "======== ALIKED 학습 완료 대기 $(date +%T) ========"
for i in $(seq 1 240); do
  if grep -q "======== 종료" "$TL" 2>/dev/null; then echo "학습 완료 감지 ($((i*15))초)"; break; fi
  if ! pgrep -f "incr_trainer.*corridor_aliked" >/dev/null; then echo "학습 프로세스 소실"; break; fi
  sleep 15
done

echo "======== 포즈 민감도 8조건 순차 실행 $(date +%T) ========"
run(){
  NAME=$1; ROT=$2; TR=$3
  D=/root/skylens/data/pose_$NAME
  R=/root/skylens/results/pose_$NAME
  rm -rf "$D" "$R"; mkdir -p "$D"
  ln -sfn "$IMG" "$D/images"
  python3 /root/skylens/scripts/pose_noise_local.py \
    --src "$SRC" --dst "$D/sparse/0" --rot_deg "$ROT" --trans_ratio "$TR" --seed 42
  cd /root/skylens/examples || return 1
  python3 incr_trainer.py default \
    --data_dir "$D" --result_dir "$R" \
    --data_factor 1 --max_steps 7000 \
    --eval_steps 7000 --save_steps 7000 --disable_viewer \
    > /root/skylens/logs/pose_${NAME}.log 2>&1
  RC=$?
  F="$R/stats/val_step6999.json"
  if [ -f "$F" ]; then
    echo "  [$NAME] 자세 ${ROT}도 · 위치 ${TR} → $(cat "$F")"
  else
    echo "  [$NAME] 실패 (rc=$RC)"
  fi
}

run base 0.0 0.000
run r0p1 0.1 0.000
run r0p3 0.3 0.000
run r0p5 0.5 0.000
run r1p0 1.0 0.000
run t0p2 0.0 0.002
run t0p5 0.0 0.005
run mix  0.3 0.003

echo "======== 결과 요약 ========"
for n in base r0p1 r0p3 r0p5 r1p0 t0p2 t0p5 mix; do
  f=/root/skylens/results/pose_$n/stats/val_step6999.json
  if [ -f "$f" ]; then
    python3 -c "
import json;d=json.load(open('$f'))
print(f\"  {'$n':>5}: PSNR {d['psnr']:6.2f} | SSIM {d['ssim']:.3f} | {d['num_GS']:>9,} GS\")"
  fi
done
echo "======== 종료 $(date +%F\ %T) ========"
} >> "$L" 2>&1
