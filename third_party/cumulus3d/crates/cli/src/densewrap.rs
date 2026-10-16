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

//! Shared densification path: exclude images outside the zone -> undistortion -> scene conversion -> densify.
//! Used by both the stream and the `densify` subcommand.

use cumulus3d_core::{ImageId, Reconstruction};
use cumulus3d_dense::{
    undistort_with, DenseOutput, DenseScene, DensifyOptions, DepthMapCache, ImageBuffer, MvsProfile, PatchMatchBackend, SceneOptions,
    UndistortBackend, UndistortCache, UndistortOptions,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// PatchMatch backend selection. Currently only `cuda` exists. Without a device, an error string.
pub fn make_pm_backend(name: &str) -> Result<Arc<dyn PatchMatchBackend>, String> {
    match name {
        "cuda" => cumulus3d_cuda::CudaPatchMatch::try_default()
            .map(|b| Arc::new(b) as Arc<dyn PatchMatchBackend>)
            .map_err(|e| format!("--pm-backend cuda: {e} (조밀화에는 CUDA 장치가 필요)")),
        other => Err(format!("알 수 없는 --pm-backend {other} (cuda)")),
    }
}

/// Image undistortion backend selection (cpu|cuda). `None` = the CPU path.
pub fn make_undistort_backend(name: &str) -> Result<Option<Arc<dyn UndistortBackend>>, String> {
    match name {
        "cpu" => Ok(None),
        "cuda" => cumulus3d_cuda::CudaUndistort::try_default()
            .map(|b| Some(Arc::new(b) as Arc<dyn UndistortBackend>))
            .map_err(|e| format!("--undistort-backend cuda: {e}")),
        other => Err(format!("알 수 없는 --undistort-backend {other} (cpu|cuda)")),
    }
}

/// Interpretation of `--mvs-profile`.
pub fn parse_profile(s: &str) -> Result<MvsProfile, String> {
    MvsProfile::parse(s).ok_or_else(|| format!("알 수 없는 --mvs-profile {s} (fast|quality)"))
}

<<<<<<< skylens (vendored)
/// 번들 조정 백엔드 선택(cpu|cuda). cuda 는 실행 중 실패하면 경고를 찍고 CPU 로 다시 푼다.
pub fn make_ba_backend(name: &str) -> Result<Arc<dyn cumulus3d_ba::BaBackend>, String> {
    match name {
        "cpu" => Ok(Arc::new(cumulus3d_ba::CpuBaBackend)),
        "cuda" => cumulus3d_cuda::CudaBundleAdjuster::try_default()
            .map(|b| Arc::new(b) as Arc<dyn cumulus3d_ba::BaBackend>)
            .map_err(|e| format!("--ba-backend cuda: {e}")),
        other => Err(format!("알 수 없는 --ba-backend {other} (cpu|cuda)")),
    }
}

/// SIFT 백엔드 선택(cpu|cuda).
||||||| base b9f4a9bda
/// SIFT 백엔드 선택(cpu|cuda).
=======
/// SIFT backend selection (cpu|cuda).
>>>>>>> upstream v0.5.0
pub fn make_sift_backend(name: &str) -> Result<Arc<dyn cumulus3d_features::SiftEngine>, String> {
    match name {
        "cpu" => Ok(Arc::new(cumulus3d_features::CpuSift::default())),
        "cuda" => cumulus3d_cuda::CudaSift::try_default()
            .map(|b| Arc::new(b) as Arc<dyn cumulus3d_features::SiftEngine>)
            .map_err(|e| format!("--sift-backend cuda: {e}")),
        other => Err(format!("알 수 없는 --sift-backend {other} (cpu|cuda)")),
    }
}

/// Descriptor matching backend selection (cpu|cuda).
pub fn make_match_backend(name: &str) -> Result<Box<dyn cumulus3d_matching::MatcherBackend>, String> {
    match name {
        "cpu" => Ok(Box::new(cumulus3d_matching::CpuMatcher::default())),
        "cuda" => cumulus3d_cuda::CudaMatcher::try_default()
            .map(|b| Box::new(b) as Box<dyn cumulus3d_matching::MatcherBackend>)
            .map_err(|e| format!("--match-backend cuda: {e}")),
        other => Err(format!("알 수 없는 --match-backend {other} (cpu|cuda)")),
    }
}

<<<<<<< skylens (vendored)
/// 희소 단계(두 뷰 기하 검증·등록·삼각측량) 백엔드 선택(cpu|cuda). cuda 는 장치 하나를 세 단계가 함께 쓴다.
pub fn make_sparse_backend(name: &str) -> Result<crate::session::SparseBackends, String> {
    match name {
        "cpu" => Ok(crate::session::SparseBackends::default()),
        "cuda" => cumulus3d_cuda::CudaSparse::try_default()
            .map(|b| {
                let b = Arc::new(b);
                crate::session::SparseBackends { two_view: b.clone(), abs_pose: b.clone(), triangulation: b }
            })
            .map_err(|e| format!("--sparse-backend cuda: {e}")),
        other => Err(format!("알 수 없는 --sparse-backend {other} (cpu|cuda)")),
    }
}

/// 조밀화 설정.
||||||| base b9f4a9bda
/// 조밀화 설정.
=======
/// Densification settings.
>>>>>>> upstream v0.5.0
#[derive(Clone)]
pub struct DenseConfig {
    /// Undistortion options.
    pub undistort: UndistortOptions,
    /// Dense scene conversion options.
    pub scene: SceneOptions,
    /// Densification (PatchMatch, filter, fusion) options.
    pub densify: DensifyOptions,
    /// Score fusion settings (if present, score fusion is used instead of `densify.fusion.mode`).
    pub score: Option<cumulus3d_dense::fusion_score::ScoreFusionOptions>,
    /// Backend (if it could not be created, that error; reported in the densification stage).
    pub backend: Result<Arc<dyn PatchMatchBackend>, String>,
    /// `--serialize-dense`: one backend call at a time.
    pub lock: Option<Arc<Mutex<()>>>,
    /// Depth map cache (`--depth-cache`).
    pub depth_cache: Option<Arc<DepthMapCache>>,
    /// Undistortion result cache.
    pub undistort_cache: Arc<UndistortCache>,
    /// Image undistortion backend (None = CPU).
    pub undistort_backend: Option<Arc<dyn UndistortBackend>>,
}

impl DenseConfig {
    /// Undistortion max_image_size 960, profile defaults.
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
            undistort_backend: None,
        }
    }
}

/// Densification result summary.
pub struct DenseRun {
    /// Number of images used for densification.
    pub frames: usize,
    /// Densification input scene.
    pub scene: DenseScene,
    /// Densification result (point cloud etc.).
    pub output: DenseOutput,
    /// Undistortion time.
    pub undistort_time: Duration,
    /// Densification time.
    pub densify_time: Duration,
    /// Time spent waiting for the backend serialization lock.
    pub lock_wait: Duration,
}

/// Densify after removing registered images for which `keep` is false (image_deleter). The input image before undistortion is
/// `image_root/name`.
pub fn dense_model(model: &Reconstruction, keep: impl Fn(&str) -> bool, image_root: &Path, cfg: &DenseConfig) -> Result<DenseRun, String> {
    let backend = cfg.backend.clone()?;
    let t0 = Instant::now();
    let mut m = model.clone();
    let del: Vec<String> =
        m.registered_images().into_iter().filter_map(|id| m.image(id).map(|im| im.name.clone())).filter(|n| !keep(n)).collect();
    let del_ref: Vec<&str> = del.iter().map(|s| s.as_str()).collect();
    m.deregister_images_by_name(&del_ref);
    let root: PathBuf = image_root.to_path_buf();
    let und = undistort_with(
        &m,
        &cfg.undistort,
        &cfg.undistort_cache,
        |im| ImageBuffer::load(root.join(&im.name)).ok().map(Arc::new),
        cfg.undistort_backend.as_deref(),
    )
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
