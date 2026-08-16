#!/usr/bin/env bash
# Stage 3 -- fit gaussians to the solved cameras.
#
#   ./03_train.sh <scene> [max_steps]
#
# This is per-scene optimisation, not inference. No pretrained model is involved:
# every gaussian's position, scale, orientation, opacity and colour is fitted
# from scratch by rendering the current guess, comparing it against the real
# frame, and pushing the error back through. Nothing appears that the footage
# did not contain.
#
# Intermediate checkpoints are written at the eval steps so a viewer can show
# progressively better geometry rather than waiting for the full run. The quality
# curve is logarithmic (RESULTS.md, experiment 2): 250 -> 1,000 steps gains
# 2.38 dB, while 10,000 -> 30,000 gains only 1.60 dB. This is what makes a
# "first frame within 30 seconds, keep refining" display loop workable.
#
# Gaussian growth stops around step 15,000, after which the file size is fixed
# and only the existing gaussians are polished -- relevant when budgeting
# transfer over the network.

set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh

SCENE="${1:?usage: 03_train.sh <scene> [max_steps]}"
MAX_STEPS="${2:-30000}"

DIR="$DATA_DIR/$SCENE"
OUT="$RESULT_DIR/$SCENE"

[ -d "$DIR/sparse/0" ] || { echo "no sparse model at $DIR/sparse/0 -- run 02_reconstruct.sh first" >&2; exit 1; }

STEPS="250 500 1000 2000 3500 5000 7000 10000 15000 20000 30000"
KEEP=""
for s in $STEPS; do [ "$s" -le "$MAX_STEPS" ] && KEEP="$KEEP $s"; done
[ -n "$KEEP" ] || KEEP="$MAX_STEPS"

log "training $SCENE for $MAX_STEPS steps (checkpoints at:$KEEP)"

cd "$GSPLAT_EXAMPLES"
python3 simple_trainer.py default \
  --data_dir "$DIR" \
  --result_dir "$OUT" \
  --data_factor 1 \
  --max_steps "$MAX_STEPS" \
  --eval_steps $KEEP \
  --ply_steps $KEEP \
  --save_steps "$MAX_STEPS" \
  --save_ply \
  --antialiased \
  --disable_viewer

log "done -- results in $OUT"
for f in "$OUT"/stats/val_step*.json; do
  [ -f "$f" ] && echo "  $(basename "$f"): $(cat "$f")"
done
