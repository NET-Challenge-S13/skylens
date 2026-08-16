#!/usr/bin/env bash
# Stage 2 -- COLMAP structure-from-motion. Solves where every frame was shot from.
#
#   ./02_reconstruct.sh <scene>
#
# Three steps: find features in each frame, match them across frame pairs, then
# alternate PnP / triangulation / bundle adjustment until poses and 3D points
# agree.
#
# Feature type is ALIKED_N16ROT with the LightGlue matcher, not SIFT. On the
# indoor corridor capture this was not a marginal win (RESULTS.md, experiment 1):
#
#   geometric verification   SIFT 56%          ALIKED 91%
#   registration             3 split models    540/540 in one model
#   distortion k1            +0.4482 diverged  +0.0778
#
# SIFT finds 3.3x more features on blank walls, but most fail verification, and
# enough bad matches survive to send self-calibration into a physically
# impossible focal length. ALIKED was also faster at every stage.
#
# Exhaustive matching, not sequential: merging several captures into one
# coordinate frame means finding which frame of capture A overlaps which frame
# of capture C with no temporal ordering to lean on. Sequential matching only
# links time-adjacent frames and cannot bridge separate videos.

set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh

SCENE="${1:?usage: 02_reconstruct.sh <scene>}"
DIR="$DATA_DIR/$SCENE"
DB="$DIR/database.db"

[ -d "$DIR/images" ] || { echo "no frames at $DIR/images -- run 01_extract_frames.sh first" >&2; exit 1; }

rm -f "$DB"
rm -rf "$DIR/sparse"
mkdir -p "$DIR/sparse"

log "[1/3] ALIKED feature extraction"
"$COLMAP_BIN" feature_extractor \
  --database_path "$DB" --image_path "$DIR/images" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV \
  --ImageReader.camera_params "$CAMERA_PARAMS" \
  --FeatureExtraction.type ALIKED_N16ROT \
  --FeatureExtraction.use_gpu 1 \
  --FeatureExtraction.gpu_index 0 \
  --AlikedExtraction.max_num_features 8192

log "[2/3] LightGlue exhaustive matching"
"$COLMAP_BIN" exhaustive_matcher \
  --database_path "$DB" \
  --FeatureMatching.type ALIKED_LIGHTGLUE \
  --FeatureMatching.use_gpu 1 \
  --FeatureMatching.gpu_index 0

# Refinement stays ON. Intrinsics were supplied above, so the mapper starts from
# a sane focal length and only adjusts it -- that is the safe configuration.
log "[3/3] mapper"
"$COLMAP_BIN" mapper \
  --database_path "$DB" --image_path "$DIR/images" \
  --output_path "$DIR/sparse" \
  --Mapper.ba_refine_focal_length 1 \
  --Mapper.ba_refine_extra_params 1 \
  --Mapper.filter_max_reproj_error 2

# The mapper can emit several disconnected models. Training reads sparse/0, so
# if a later model registered more frames, promote it -- otherwise a run can
# silently train on a handful of images.
BEST=""
BEST_N=0
for model in "$DIR"/sparse/*/; do
  N="$(python3 -c "
import struct,sys
with open('$model/images.bin','rb') as f:
    print(struct.unpack('<Q', f.read(8))[0])
" 2>/dev/null || echo 0)"
  log "  $(basename "$model"): $N images"
  if [ "$N" -gt "$BEST_N" ]; then BEST_N="$N"; BEST="$model"; fi
done

if [ -n "$BEST" ] && [ "$(basename "$BEST")" != "0" ]; then
  log "promoting $(basename "$BEST") ($BEST_N images) to sparse/0"
  rm -rf "$DIR/sparse/0.bak"
  mv "$DIR/sparse/0" "$DIR/sparse/0.bak"
  mv "$BEST" "$DIR/sparse/0"
fi

log "done -- sparse/0 has $BEST_N registered images"
"$COLMAP_BIN" model_analyzer --path "$DIR/sparse/0" 2>&1 | grep -viE '^[IW][0-9]{4}' | head -10
