/*
 * densewrap.rs
 *
 * Copyright (c) 2026 ParkSangWoo
 *
 * Author(s):
 *
 *      ParkSangWoo <dev.parksangwoo@gmail.com>
 *
 *
 * This file is part of cumulus3d.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//! 조밀화 공용 경로: 구역 밖 영상 제외 → 왜곡 보정 → 장면 변환 → densify.
//! 스트림과 `densify` 하위 명령이 함께 쓴다.

use cumulus3d_core::{ImageId, Reconstruction};
use cumulus3d_dense::{
    undistort, DenseOutput, DenseScene, DensifyOptions, DepthMapCache, ImageBuffer, MvsProfile, PatchMatchBackend, SceneOptions,
    UndistortCache, UndistortOptions,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// PatchMatch 백엔드 선택. 지금은 `cuda` 만 있다. 장치가 없으면 오류 문자열.
pub fn make_pm_backend(name: &str) -> Result<Arc<dyn PatchMatchBackend>, String> {
    match name {
        "cuda" => cumulus3d_cuda::CudaPatchMatch::try_default()
            .map(|b| Arc::new(b) as Arc<dyn PatchMatchBackend>)
            .map_err(|e| format!("--pm-backend cuda: {e} (조밀화에는 CUDA 장치가 필요)")),
        other => Err(format!("알 수 없는 --pm-backend {other} (cuda)")),
    }
}

/// `--mvs-profile` 해석.
pub fn parse_profile(s: &str) -> Result<MvsProfile, String> {
    MvsProfile::parse(s).ok_or_else(|| format!("알 수 없는 --mvs-profile {s} (fast|quality)"))
}

/// SIFT 백엔드 선택(cpu|cuda).
pub fn make_sift_backend(name: &str) -> Result<Arc<dyn cumulus3d_features::SiftEngine>, String> {
    match name {
        "cpu" => Ok(Arc::new(cumulus3d_features::CpuSift::default())),
        "cuda" => cumulus3d_cuda::CudaSift::try_default()
            .map(|b| Arc::new(b) as Arc<dyn cumulus3d_features::SiftEngine>)
            .map_err(|e| format!("--sift-backend cuda: {e}")),
        other => Err(format!("알 수 없는 --sift-backend {other} (cpu|cuda)")),
    }
}

/// 기술자 매칭 백엔드 선택(cpu|cuda).
pub fn make_match_backend(name: &str) -> Result<Box<dyn cumulus3d_matching::MatcherBackend>, String> {
    match name {
        "cpu" => Ok(Box::new(cumulus3d_matching::CpuMatcher::default())),
        "cuda" => cumulus3d_cuda::CudaMatcher::try_default()
            .map(|b| Box::new(b) as Box<dyn cumulus3d_matching::MatcherBackend>)
            .map_err(|e| format!("--match-backend cuda: {e}")),
        other => Err(format!("알 수 없는 --match-backend {other} (cpu|cuda)")),
    }
}

/// 조밀화 설정.
#[derive(Clone)]
pub struct DenseConfig {
    /// 왜곡 보정 옵션.
    pub undistort: UndistortOptions,
    /// 조밀 장면 변환 옵션.
    pub scene: SceneOptions,
    /// 조밀화(PatchMatch·필터·융합) 옵션.
    pub densify: DensifyOptions,
    /// 점수 융합 설정(있으면 `densify.fusion.mode` 대신 점수 융합).
    pub score: Option<cumulus3d_dense::fusion_score::ScoreFusionOptions>,
    /// 백엔드(만들지 못했으면 그 오류; 조밀화 단계에서 보고).
    pub backend: Result<Arc<dyn PatchMatchBackend>, String>,
    /// `--serialize-dense`: 백엔드 호출을 한 번에 하나로.
    pub lock: Option<Arc<Mutex<()>>>,
    /// 깊이맵 캐시(`--depth-cache`).
    pub depth_cache: Option<Arc<DepthMapCache>>,
    /// 왜곡 보정 결과 캐시.
    pub undistort_cache: Arc<UndistortCache>,
}

impl DenseConfig {
    /// 왜곡 보정 max_image_size 960, 프로파일 기본값.
    pub fn new(backend: Result<Arc<dyn PatchMatchBackend>, String>, profile: MvsProfile) -> Self {
        Self {
            undistort: UndistortOptions::pipeline(),
            scene: SceneOptions::default(),
            densify: DensifyOptions::with_profile(profile),
            score: None,
            backend,
            lock: None,
            depth_cache: None,
            undistort_cache: Arc::new(UndistortCache::new()),
        }
    }
}

/// 조밀화 결과 요약.
pub struct DenseRun {
    /// 조밀화에 쓴 영상 수.
    pub frames: usize,
    /// 조밀화 입력 장면.
    pub scene: DenseScene,
    /// 조밀화 결과(점군 등).
    pub output: DenseOutput,
    /// 왜곡 보정 시간.
    pub undistort_time: Duration,
    /// 조밀화 시간.
    pub densify_time: Duration,
    /// 백엔드 직렬화 잠금 대기 시간.
    pub lock_wait: Duration,
}

/// `keep` 이 거짓인 등록 영상을 빼고(image_deleter) 조밀화한다. 보정 전 입력 영상은 `image_root/이름`.
pub fn dense_model(model: &Reconstruction, keep: impl Fn(&str) -> bool, image_root: &Path, cfg: &DenseConfig) -> Result<DenseRun, String> {
    let backend = cfg.backend.clone()?;
    let t0 = Instant::now();
    let mut m = model.clone();
    let del: Vec<String> =
        m.registered_images().into_iter().filter_map(|id| m.image(id).map(|im| im.name.clone())).filter(|n| !keep(n)).collect();
    let del_ref: Vec<&str> = del.iter().map(|s| s.as_str()).collect();
    m.deregister_images_by_name(&del_ref);
    let root: PathBuf = image_root.to_path_buf();
    let und = undistort(&m, &cfg.undistort, &cfg.undistort_cache, |im| ImageBuffer::load(root.join(&im.name)).ok().map(Arc::new))
        .map_err(|e| format!("왜곡 보정 실패: {e}"))?;
    let images: BTreeMap<ImageId, Arc<ImageBuffer>> = und.images;
    let frames = images.len();
    let scene = DenseScene::from_reconstruction(&und.reconstruction, &images, &cfg.scene).map_err(|e| format!("장면 변환 실패: {e}"))?;
    drop(images);
    let undistort_time = t0.elapsed();
    let tw = Instant::now();
    let guard = cfg.lock.as_ref().map(|l| l.lock().unwrap_or_else(|p| p.into_inner()));
    let lock_wait = tw.elapsed();
    let t1 = Instant::now();
    let output =
        cumulus3d_dense::densify::densify_with(&scene, &cfg.densify, cfg.score.as_ref(), backend.as_ref(), cfg.depth_cache.as_deref())
            .map_err(|e| format!("조밀화 실패: {e}"));
    drop(guard);
    let output = output?;
    Ok(DenseRun { frames, scene, output, undistort_time, densify_time: t1.elapsed(), lock_wait })
}
