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

//! GPS→ENU, Sim3 alignment, shared 3D point alignment, point cloud post-processing.
//!
//! - [`geodesy`]: WGS84 LLA ↔ ECEF ↔ ENU.
//! - [`umeyama`](mod@umeyama): Umeyama Sim3, LO-RANSAC robust estimation, iterative reweighting (robust Umeyama).
//! - [`model_aligner`]: alignment of registered image projection centers to GPS ENU (robust Sim3 estimation).
//! - [`shared`]: shared 3D point correspondences between two reconstructions and their alignment.
//! - [`kdtree`], [`cloud`]: point cloud transform, proximity masking, thinning, snapshot composition.
//! - [`reanchor`]: chained Sim3 that reconnects the coordinate frames of earlier zones to the new refined model.
//!
//! # Usage example
//!
//! ```
//! use cumulus3d_align::{compose_snapshot, umeyama, EnuFrame, SnapshotOptions};
//! use cumulus3d_core::io::PointCloud;
//! use cumulus3d_core::{Quat, Sim3, Vec3};
//!
//! // GPS (WGS84) ↔ ENU: east/north/up meter coordinates relative to the origin.
//! let enu = EnuFrame::new(37.5, 127.0, 10.0);
//! let p = enu.lla_to_enu(37.5001, 127.0001, 10.0);
//! let (lat, lon, _alt) = enu.enu_to_lla(&p);
//! assert!((lat - 37.5001).abs() < 1e-9 && (lon - 127.0001).abs() < 1e-9);
//!
//! // Estimate the Sim3 (src → dst) between sets of corresponding points.
//! let truth = Sim3::new(2.0, Quat::from_axis_angle(&Vec3::z(), 0.3), Vec3::new(1.0, 2.0, 3.0));
//! let src: Vec<Vec3> = (0..10).map(|i| Vec3::new(i as f64, (i * i) as f64 * 0.1, (i as f64).sin())).collect();
//! let dst: Vec<Vec3> = src.iter().map(|x| truth.transform_point(x)).collect();
//! let est = umeyama(&src, &dst, true).unwrap();
//! assert!((est.scale - 2.0).abs() < 1e-9);
//!
//! // Snapshot: remove preview points within 1.5 m of refined points, merge, then thin by 1/stride.
//! let fine = PointCloud { positions: vec![[0.0, 0.0, 0.0]], ..Default::default() };
//! let coarse = PointCloud { positions: vec![[0.5, 0.0, 0.0], [10.0, 0.0, 0.0]], ..Default::default() };
//! let snap = compose_snapshot(&[&fine], &[&coarse], &SnapshotOptions { mask_radius: 1.5, stride: 1 });
//! assert_eq!(snap.len(), 2);
//! ```
//!
//! ```no_run
//! use cumulus3d_align::{align_to_gps_file, ModelAlignerOptions};
//! let mut rec = cumulus3d_core::interop::read_model("model/0").unwrap();
//! let a = align_to_gps_file(&mut rec, "gps_ref.txt", &ModelAlignerOptions::default()).unwrap();
//! println!("인라이어 {}/{}, 중앙 오차 {:.2} m", a.num_inliers, a.common.len(), a.median_error);
//! ```
#![warn(missing_docs)]

pub mod cloud;
pub mod error;
pub mod geodesy;
pub mod kdtree;
pub mod model_aligner;
pub mod reanchor;
pub mod shared;
pub mod umeyama;

pub use cloud::{compose_snapshot, decimate, merge_clouds, remove_near, select, transform_cloud, SnapshotOptions};
pub use error::{AlignError, Result};
pub use geodesy::{ecef_to_lla, lla_to_ecef, EnuFrame, WGS84_A, WGS84_B, WGS84_E2, WGS84_F};
pub use kdtree::KdTree;
pub use model_aligner::{
    align_to_gps, align_to_gps_file, estimate_gps_alignment, gps_to_enu, EnuOrigin, GpsAlignment, ModelAlignerOptions,
};
pub use reanchor::AnchorChain;
pub use shared::{align_reconstructions, position_index_from_name, shared_point_correspondences, SharedPointOptions};
pub use umeyama::{estimate_sim3_ransac, robust_umeyama, umeyama, RankCheck, RobustUmeyamaOptions, RobustUmeyamaResult, Sim3Estimator};
