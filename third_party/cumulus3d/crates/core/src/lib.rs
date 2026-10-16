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

//! Common foundation: camera models, geometry, reconstruction data structures, correspondence graph, feature store, RANSAC, I/O.
//!
//! Convention summary (public items per module are in the crate `README.md`):
//! - Poses are `world_to_cam` (world -> camera); quaternions are Hamilton, stored in w,x,y,z order.
//! - Camera frame: x right, y down, z forward. Pixel coordinates: top-left pixel center = (0.5, 0.5).
//! - ids: camera/rig/frame/image = u32, 3D point = u64, pair = u64. The invalid value is the maximum of each type.
//!
//! # Usage example
//!
//! Builds a reconstruction from two images and one 3D point, computes the reprojection error, then saves and loads it as a model file.
//!
//! ```
//! use cumulus3d_core::analyzer::ModelStats;
//! use cumulus3d_core::interop::{read_model, write_model_binary, ImageOrder};
//! use cumulus3d_core::{Camera, CameraModelKind, Image, Quat, Reconstruction, Rigid3, TrackEntry, Vec3};
//!
//! let mut cam = Camera::from_focal(CameraModelKind::Pinhole, 1000.0, 1920, 1080);
//! cam.camera_id = 1;
//! let mut rec = Reconstruction::new();
//! rec.add_camera_own_rig(cam.clone())?;
//!
//! // Project one world point into the two cameras (1 m baseline) to produce 2D observations.
//! let x = Vec3::new(0.2, -0.1, 5.0);
//! for (id, tx) in [(1u32, 0.0), (2, -1.0)] {
//!     let world_to_cam = Rigid3::new(Quat::IDENTITY, Vec3::new(tx, 0.0, 0.0));
//!     let xy = cam.cam_to_img(&world_to_cam.transform_point(&x)).unwrap();
//!     rec.add_image_own_frame(Image::new(id, format!("img{id}.jpg"), 1, [xy]), Some(world_to_cam))?;
//!     rec.register_image(id)?;
//! }
//! let pid = rec.add_point3d(x, vec![TrackEntry::new(1, 0), TrackEntry::new(2, 0)], [255, 0, 0])?;
//! rec.update_point3d_errors();
//! assert!(rec.point3d(pid).unwrap().error < 1e-9);
//! rec.check_invariants()?;
//!
//! let dir = std::env::temp_dir().join("cumulus3d_core_doc_example");
//! std::fs::create_dir_all(&dir)?;
//! write_model_binary(&rec, &dir, ImageOrder::default())?;
//! let back = read_model(&dir)?;
//! let stats = ModelStats::compute(&back);
//! assert_eq!((stats.registered_image_count, stats.num_points3d), (2, 1));
//! # Ok::<(), cumulus3d_core::Error>(())
//! ```

#![warn(missing_docs)]

pub mod analyzer;
pub mod camera;
pub mod error;
pub mod features;
pub mod geometry;
pub mod graph;
pub mod ids;
pub mod interop;
pub mod io;
pub mod linalg;
pub mod ransac;
pub mod reconstruction;
pub mod store;

pub use camera::{Camera, CameraModelKind};
pub use error::{Error, Result};
pub use features::{Descriptors, FeatureMatch, Keypoint, TwoViewGeometry, TwoViewGeometryConfig, DESCRIPTOR_DIM};
pub use geometry::{Mat3, Mat3x4, Quat, Rigid3, Sim3, Vec2, Vec3};
pub use graph::{MatchGraph, MatchGraphOptions};
pub use ids::*;
pub use reconstruction::{Frame, Image, Point2D, Point3D, Reconstruction, Rig, TrackEntry};
pub use store::FeatureStore;
