/*
 * lib.rs
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

//! 왜곡 보정과 다시점 조밀화.
//!
//! - [`undistort`](mod@undistort): 왜곡 있는 카메라 → PINHOLE 카메라 계산, 카메라별 재표본 맵 캐시, 영상 재표본, 희소 모델 변환.
//! - [`image`]: 영상 버퍼와 재표본 도구.
//! - [`scene`]: 조밀화 장면(뷰 K·자세·영상, 희소점).
//! - [`neighbors`]: 이웃 뷰 선택과 깊이 범위.
//! - [`kernel`]: PatchMatch 백엔드 경계(한 스케일의 전파·뷰 선택·정제는 백엔드가 수행).
//! - [`densify`](mod@densify): 다중 스케일 진행, 세부 복원, 필터, 융합 연결.
//! - [`fusion`], [`fusion_score`]: 깊이맵 융합(일치·확장 중앙값·점수 융합).
//! - [`postproc`], [`upsample`], [`math`], [`params`]: 후처리, 상향 표본, 수치 도구, 설정.
//! - [`stats`]: 점군 품질 통계.
//! - [`cache`]: 겹치는 구역용 깊이맵 캐시.
//! - [`synthetic`]: 시험용 합성 장면.
//!
//! 이 크레이트에는 PatchMatch 백엔드 구현이 없다. [`densify()`] 에는 [`PatchMatchBackend`] 구현
//! (예: `cumulus3d-cuda` 의 GPU 백엔드 `CudaPatchMatch`)을 넘겨야 한다.
//!
//! # 사용 예
//!
//! 합성 장면의 참 깊이맵을 CPU 에서 융합한다(백엔드 불필요).
//!
//! ```
//! use cumulus3d_dense::neighbors::PairStats;
//! use cumulus3d_dense::synthetic::{make_scene, SynthConfig};
//! use cumulus3d_dense::{fuse, FusionInput, FusionParams};
//!
//! let cfg = SynthConfig { width: 96, height: 72, focal: 75.0, num_points: 500, ..SynthConfig::default() };
//! let s = make_scene(&cfg);
//! let stats = PairStats::new(&s.scene);
//! let overlap: Vec<Vec<usize>> = (0..s.scene.views.len()).map(|v| stats.select(v, 50, 0.0)).collect();
//! let inputs: Vec<Option<FusionInput>> =
//!     s.depth.iter().zip(&s.normal).map(|(d, n)| Some(FusionInput::plain(d, n))).collect();
//! let out = fuse(&s.scene, &inputs, &overlap, &FusionParams::default(), 1);
//! assert!(out.cloud.len() > 0);
//! ```
//!
//! 실제 조밀화(왜곡 보정 → 장면 → 깊이맵 → 융합 → PLY). 백엔드는 GPU 구현을 넘긴다.
//!
//! ```no_run
//! use cumulus3d_dense::{densify, undistort_from_dir, DenseScene, DensifyOptions, PatchMatchBackend, SceneOptions, UndistortCache, UndistortOptions};
//!
//! fn run(backend: &dyn PatchMatchBackend) -> cumulus3d_core::Result<()> {
//!     let rec = cumulus3d_core::interop::read_model("sparse/0")?;
//!     let und = undistort_from_dir(&rec, "images", &UndistortOptions::pipeline(), &UndistortCache::new())?;
//!     let scene = DenseScene::from_reconstruction(&und.reconstruction, &und.images, &SceneOptions::default())?;
//!     let out = densify(&scene, &DensifyOptions::default(), backend, None)?;
//!     out.write_ply("dense/fused.ply")?;
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]

pub mod cache;
pub mod densify;
pub mod fusion;
pub mod fusion_score;
pub mod image;
pub mod kernel;
pub mod math;
pub mod neighbors;
pub mod params;
pub mod postproc;
pub mod scene;
pub mod stats;
pub mod synthetic;
pub mod undistort;
pub mod upsample;

pub use cache::{CachedDepth, DepthMapCache};
pub use densify::{compute_depth_maps, densify, fuse_depth_maps, pair_geometry, DenseOutput, DenseTimings, DepthMapResult, DepthMapSet};
pub use fusion::{fuse, FusionInput, FusionOutput};
pub use image::{GrayImage, ImageBuffer};
pub use kernel::{
    DepthSnapshot, KernelInput, KernelView, LevelImage, PairGeometry, PatchMatchBackend, PatchMatchSession, RunParams, ViewState,
};
pub use params::{
    DensifyOptions, FilterParams, FusionMode, FusionParams, FusionResidual, LevelSchedule, MvsProfile, NeighborParams, PmParams,
    ResidualParams,
};
pub use postproc::PostParams;
pub use scene::{DenseScene, DenseView, SceneOptions, ScenePoint};
pub use stats::{cloud_stats, CloudStats};
pub use undistort::{
    undistort, undistort_from_dir, undistort_reconstruction, undistorted_camera, write_undistorted_workspace, CameraUndistortion,
    UndistortCache, UndistortOptions, UndistortResult,
};
