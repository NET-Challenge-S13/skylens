#!/usr/bin/env bash
# Experiment 4 -- drone formation: does spacing or viewing-angle diversity matter more?
#
#   ./formation_sweep.sh <scene>
#
# Trains six subsets of one joint reconstruction. Because every condition is a
# slice of the same solved model, they share a coordinate frame and are directly
# comparable -- reconstructing each capture separately would not be.
#
# ---------------------------------------------------------------------------
# KNOWN LIMITATION -- READ BEFORE QUOTING THE PSNR
#
# Each condition holds out every 8th frame *of its own training videos*, so the
# conditions are not graded on the same test set. A single-capture condition
# ends up interpolating between frames 1/30 s apart, which is close to copying,
# and scores far higher than a condition covering the whole space.
#
# That is why the measured PSNR falls monotonically as frames are added
# (solo37 33.68 -> all3 21.62). It is an artifact of the split, not a finding.
#
# To make this decisive, hold out a FIXED set of frames drawn from all captures
# before subsetting, exclude them from every condition's training set, and
# evaluate all six against it. Not yet implemented.
#
# The gaussian counts are unaffected by the split and are worth reading: the
# wide-spacing pair produced 472,103 gaussians from 270 frames, while the
# wide-angle pair produced 322,791 from 405. Parallax, not frame count, is what
# lets density control subdivide -- suggestive that spacing matters more, but
# not yet proof.
# ---------------------------------------------------------------------------

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../pipeline/config.sh"

SCENE="${1:?usage: formation_sweep.sh <scene>}"
SRC="$DATA_DIR/$SCENE/sparse/0"
IMAGES="$DATA_DIR/$SCENE/images"

run_condition() {
  local name="$1"
  shift
  local work="$DATA_DIR/formation_$name"
  local out="$RESULT_DIR/formation_$name"

  rm -rf "$work" "$out"
  mkdir -p "$work"
  ln -sfn "$IMAGES" "$work/images"

  python3 -m skylens_recon.subset_model --src "$SRC" --dst "$work/sparse/0" --keep "$@"

  ( cd "$GSPLAT_EXAMPLES" && python3 simple_trainer.py default \
      --data_dir "$work" --result_dir "$out" \
      --data_factor 1 --max_steps 7000 \
      --eval_steps 7000 --save_steps 7000 --disable_viewer \
      > "$LOG_DIR/formation_${name}.log" 2>&1 ) || log "  [$name] FAILED -- see $LOG_DIR/formation_${name}.log"

  local stats="$out/stats/val_step6999.json"
  [ -f "$stats" ] && log "  [$name] $(cat "$stats")"
}

log "formation sweep on $SCENE"
run_condition solo37   cam2037
run_condition solo39   cam2039
run_condition pair3739 cam2037 cam2039
run_condition pair3839 cam2038 cam2039
run_condition pair3738 cam2037 cam2038
run_condition all3     cam2037 cam2038 cam2039

log "summary (see limitation above before comparing PSNR)"
for n in solo37 solo39 pair3739 pair3839 pair3738 all3; do
  f="$RESULT_DIR/formation_$n/stats/val_step6999.json"
  [ -f "$f" ] && python3 -c "
import json
d = json.load(open('$f'))
print(f\"  {'$n':>9}: PSNR {d['psnr']:6.2f} | SSIM {d['ssim']:.3f} | {d['num_GS']:>9,} GS\")"
done
