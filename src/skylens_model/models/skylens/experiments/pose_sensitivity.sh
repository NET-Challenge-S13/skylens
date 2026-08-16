#!/usr/bin/env bash
# Experiment 3 -- how accurately must a drone's pose be known?
#
#   ./pose_sensitivity.sh <scene>
#
# Perturbs the solved poses by a known amount, retrains each condition for 7,000
# steps, and reads the quality loss off the PSNR. Eight conditions: a clean
# baseline, four attitude levels, two position levels, and one mixed.
#
# Measured result (RESULTS.md):
#
#   attitude 0.1 deg   -0.08 dB      position 0.2%   -2.24 dB
#   attitude 0.3 deg   -1.26 dB      position 0.5%   -3.60 dB
#   attitude 0.5 deg   -1.49 dB      mixed           -2.99 dB
#   attitude 1.0 deg   -2.70 dB
#
# Position error dominates. Matching the 0.2% position loss takes roughly 0.8 deg
# of attitude error. The practical consequence: GPS horizontal accuracy is
# 0.5-1.5 m, far coarser than this, so drone position telemetry cannot be used
# as camera pose directly -- pose still has to be solved from the imagery.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../pipeline/config.sh"

SCENE="${1:?usage: pose_sensitivity.sh <scene>}"
SRC="$DATA_DIR/$SCENE/sparse/0"
IMAGES="$DATA_DIR/$SCENE/images"

run_condition() {
  local name="$1" rot="$2" trans="$3"
  local work="$DATA_DIR/pose_$name"
  local out="$RESULT_DIR/pose_$name"

  rm -rf "$work" "$out"
  mkdir -p "$work"
  ln -sfn "$IMAGES" "$work/images"

  python3 -m skylens_model.models.skylens.pose_noise \
    --src "$SRC" --dst "$work/sparse/0" \
    --rot-deg "$rot" --trans-ratio "$trans" --seed 42

  ( cd "$GSPLAT_EXAMPLES" && python3 simple_trainer.py default \
      --data_dir "$work" --result_dir "$out" \
      --data_factor 1 --max_steps 7000 \
      --eval_steps 7000 --save_steps 7000 --disable_viewer \
      > "$LOG_DIR/pose_${name}.log" 2>&1 ) || log "  [$name] FAILED -- see $LOG_DIR/pose_${name}.log"

  local stats="$out/stats/val_step6999.json"
  [ -f "$stats" ] && log "  [$name] $(cat "$stats")"
}

log "pose sensitivity sweep on $SCENE"
run_condition base 0.0 0.000
run_condition r0p1 0.1 0.000
run_condition r0p3 0.3 0.000
run_condition r0p5 0.5 0.000
run_condition r1p0 1.0 0.000
run_condition t0p2 0.0 0.002
run_condition t0p5 0.0 0.005
run_condition mix  0.3 0.003

log "summary"
for n in base r0p1 r0p3 r0p5 r1p0 t0p2 t0p5 mix; do
  f="$RESULT_DIR/pose_$n/stats/val_step6999.json"
  [ -f "$f" ] && python3 -c "
import json
d = json.load(open('$f'))
print(f\"  {'$n':>5}: PSNR {d['psnr']:6.2f} | SSIM {d['ssim']:.3f} | {d['num_GS']:>9,} GS\")"
done
