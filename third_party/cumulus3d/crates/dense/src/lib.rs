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

//! Undistortion and multi-view densification.
//!
//! - [`undistort`](mod@undistort): distorted camera → PINHOLE camera computation, per-camera resampling-map cache, image resampling, sparse
//!   model conversion.
//! - [`image`]: image buffers and resampling tools.
//! - [`scene`]: densification scene (view K, pose, image; sparse points).
//! - [`neighbors`]: neighbor view selection and depth ranges.
//! - [`kernel`]: PatchMatch backend boundary (propagation, view selection and refinement at one scale are done by the backend).
//! - [`densify`](mod@densify): multi-scale progression, detail restoration, filters, fusion hookup.
//! - [`fusion`], [`fusion_score`]: depth map fusion (agreement, extended median, score fusion).
//! - [`smooth`]: normal-direction bilateral smoothing of the fused cloud.
//! - [`visibility`]: free-space (visibility) filter that removes fused points in front of surfaces other views observed.
//! - [`postproc`], [`upsample`], [`math`], [`params`]: post-processing, upsampling, numeric tools, settings.
//! - [`stats`]: point cloud quality statistics.
//! - [`cache`]: depth map cache for overlapping zones.
//! - [`synthetic`]: synthetic scenes for tests.
//!
//! This crate has no PatchMatch backend implementation. [`densify()`] needs a [`PatchMatchBackend`] implementation
//! (e.g. the GPU backend `CudaPatchMatch` of `cumulus3d-cuda`).
//!
//! # Usage example
//!
//! Fuse the ground-truth depth maps of a synthetic scene on the CPU (no backend needed).
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
//! Real densification (undistortion → scene → depth maps → fusion → PLY). Pass a GPU implementation as the backend.
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
pub mod smooth;
pub mod stats;
pub mod synthetic;
pub mod undistort;
pub mod upsample;
pub mod visibility;

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
pub use smooth::{smooth_along_normals, SmoothParams};
pub use stats::{cloud_stats, CloudStats};
pub use undistort::{
    undistort, undistort_from_dir, undistort_reconstruction, undistorted_camera, write_undistorted_workspace, CameraUndistortion,
    UndistortCache, UndistortOptions, UndistortResult,
};
pub use visibility::{filter_visibility, VisibilityParams};
