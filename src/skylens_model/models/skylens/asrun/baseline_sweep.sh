#!/usr/bin/env bash
# 드론 편대 설계 실험: 베이스라인(간격) vs 시선 각도 다양성
# 같은 540장 통합 모델의 부분집합으로 학습해 두 요인의 기여를 분리한다.
SRC=/root/skylens/data/corridor_aliked
IMG=/root/skylens/data/corridor_hq/images
L=/root/skylens/logs/baseline_sweep.log
mkdir -p /root/skylens/logs
{
echo "======== 편대 설계 실험 $(date +%F\ %T) ========"

run(){
  NAME=$1; shift
  SRCS="$@"
  D=/root/skylens/data/bl_$NAME
  R=/root/skylens/results/bl_$NAME
  rm -rf "$D" "$R"; mkdir -p "$D"
  ln -sfn "$IMG" "$D/images"
  python3 /root/skylens/scripts/subset_model.py --src "$SRC/sparse/0" --dst "$D/sparse/0" --keep $SRCS
  cd /root/skylens/examples || return 1
  python3 incr_trainer.py default \
    --data_dir "$D" --result_dir "$R" \
    --data_factor 1 --max_steps 7000 \
    --eval_steps 7000 --save_steps 7000 --disable_viewer \
    > /root/skylens/logs/bl_${NAME}.log 2>&1
  F="$R/stats/val_step6999.json"
  if [ -f "$F" ]; then echo "  [$NAME] $(cat "$F")"; else echo "  [$NAME] 실패"; fi
}

run solo37   cam2037
run solo39   cam2039
run pair3739 cam2037 cam2039
run pair3839 cam2038 cam2039
run pair3738 cam2037 cam2038
run all3     cam2037 cam2038 cam2039

echo "======== 결과 요약 ========"
for n in solo37 solo39 pair3739 pair3839 pair3738 all3; do
  f=/root/skylens/results/bl_$n/stats/val_step6999.json
  [ -f "$f" ] && python3 -c "
import json;d=json.load(open('$f'))
print(f\"  {'$n':>9}: PSNR {d['psnr']:6.2f} | SSIM {d['ssim']:.3f} | {d['num_GS']:>9,} GS\")"
done
echo "======== 종료 $(date +%F\ %T) ========"
} >> "$L" 2>&1
