#!/usr/bin/env bash
C=/root/miniconda3/envs/colmap-cuda/bin/colmap
S=/root/skylens/data/corridor_hq          # 프레임 재사용 (동일 540장·2048px)
D=/root/skylens/data/corridor_aliked
L=/root/skylens/logs/corridor_aliked.log
{
echo "======== ALIKED + LightGlue 비교 실험 $(date +%F\ %T) ========"
rm -rf $D; mkdir -p $D/sparse
ln -sfn $S/images $D/images
DB=$D/database.db

echo "[1/3] ALIKED 특징점 추출 (GPU) $(date +%T)"
$C feature_extractor --database_path $DB --image_path $D/images \
  --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
  --ImageReader.camera_params "1683.5,1640.3,1024,576,0.0365,-0.0001,-0.0003,0.0009" \
  --FeatureExtraction.type ALIKED_N16ROT \
  --FeatureExtraction.use_gpu 1 --FeatureExtraction.gpu_index 0 \
  --AlikedExtraction.max_num_features 8192 2>&1 | tail -4

echo "[2/3] LightGlue 매칭 (GPU) $(date +%T)"
$C exhaustive_matcher --database_path $DB \
  --FeatureMatching.type ALIKED_LIGHTGLUE \
  --FeatureMatching.use_gpu 1 --FeatureMatching.gpu_index 0 2>&1 | tail -4

echo "[3/3] mapper $(date +%T)"
$C mapper --database_path $DB --image_path $D/images --output_path $D/sparse \
  --Mapper.ba_refine_focal_length 1 --Mapper.ba_refine_extra_params 1 \
  --Mapper.filter_max_reproj_error 2 2>&1 | tail -4

echo "--- 결과 ---"; ls $D/sparse
for m in $D/sparse/*/; do echo "=== $m ==="; $C model_analyzer --path "$m" 2>&1 | grep -viE '^I2026|^W2026' | head -8; done
echo "======== 종료 $(date +%F\ %T) ========"
} >> "$L" 2>&1
