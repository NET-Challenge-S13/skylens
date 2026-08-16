#!/usr/bin/env bash
D=/root/skylens/data/corridor_hq
L=/root/skylens/logs/corridor_hq.log
mkdir -p /root/skylens/logs
{
echo "======== 복도 고품질 재복원 (로컬) $(date +%F\ %T) ========"
rm -rf $D; mkdir -p $D/images
echo "[1/4] 프레임 추출 — 540장 · 2048px"
declare -A N=( [2037]=135 [2038]=270 [2039]=135 )
for f in 2037 2038 2039; do
  V=/root/skylens/IMG_$f.MP4
  DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $V)
  FPS=$(python3 -c "print(round(${N[$f]}/$DUR,4))")
  ffmpeg -hide_banner -loglevel error -i $V \
    -vf "fps=$FPS,scale=w=2048:h=2048:force_original_aspect_ratio=decrease:flags=lanczos" \
    -q:v 2 -start_number 0 $D/images/cam${f}_%04d.jpg
  echo "  IMG_$f → $(ls $D/images/cam${f}_* | wc -l)장"
done
echo "  총 $(ls $D/images | wc -l)장 · $(ffprobe -v error -show_entries stream=width,height -of csv=p=0 $D/images/cam2037_0000.jpg) · $(du -sh $D/images|cut -f1)"

DB=$D/database.db; mkdir -p $D/sparse
echo "[2/4] feature extraction $(date +%T)"
colmap feature_extractor --database_path $DB --image_path $D/images \
  --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
  --SiftExtraction.use_gpu 0 --SiftExtraction.num_threads 24 \
  --SiftExtraction.max_num_features 16384 --SiftExtraction.peak_threshold 0.004 2>&1 | tail -3
echo "[3/4] matching $(date +%T)"
colmap sequential_matcher --database_path $DB \
  --SiftMatching.use_gpu 0 --SiftMatching.num_threads 24 \
  --SequentialMatching.overlap 25 --SequentialMatching.quadratic_overlap 1 \
  --SequentialMatching.loop_detection 0 2>&1 | tail -3
echo "  + 교차 촬영본 매칭 보강 $(date +%T)"
colmap exhaustive_matcher --database_path $DB \
  --SiftMatching.use_gpu 0 --SiftMatching.num_threads 24 \
  --ExhaustiveMatching.block_size 40 2>&1 | tail -3
echo "[4/4] mapper $(date +%T)"
colmap mapper --database_path $DB --image_path $D/images --output_path $D/sparse \
  --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_extra_params 0 2>&1 | tail -4
echo "--- 모델 수 ---"; ls $D/sparse
for m in $D/sparse/*/; do echo "=== $m ==="; colmap model_analyzer --path "$m" 2>&1 | grep -viE '^I2026|^W2026' | head -8; done
echo "======== COLMAP 종료 $(date +%F\ %T) ========"
} >> "$L" 2>&1
