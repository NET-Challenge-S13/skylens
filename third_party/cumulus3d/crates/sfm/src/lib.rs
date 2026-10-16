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

//! Sparse reconstruction: global SfM (rotation averaging + position estimation), registering new images to an existing model,
//! and triangulation.
//!
//! The inputs are the [`cumulus3d_core::FeatureStore`] (features and two-view geometry) of `cumulus3d-core` and the
//! [`cumulus3d_core::MatchGraph`] (correspondence graph); the output is a [`cumulus3d_core::Reconstruction`].
//! The entry points are [`global_mapper()`] (first model), [`register_images`] (register new images) and
//! [`triangulate_points`] (create/extend points). A per-module table is in the crate README.
//!
//! # Usage example
//!
//! Build the first model from the first 5 positions of a synthetic scene (3 drones, 3 images per position),
//! register the images of the 6th position, then triangulate only the new images.
//!
//! ```
//! use cumulus3d_core::{MatchGraph, MatchGraphOptions};
//! use cumulus3d_sfm::synthetic::{generate, SceneConfig};
//! use cumulus3d_sfm::{
//!     global_mapper, register_images, triangulate_points, GlobalSfmOptions, PointTriangulatorOptions,
//!     RegistrationOptions, TriangulationScope,
//! };
//!
//! let scene = generate(&SceneConfig { num_positions: 6, num_points: 1500, ..Default::default() });
//! let graph_up_to = |n: usize| {
//!     let opts = MatchGraphOptions { image_names: scene.names_up_to(n), ..Default::default() };
//!     MatchGraph::from_store(&scene.store, &opts)
//! };
//!
//! // 1) First model with global SfM.
//! let out = global_mapper(&scene.store, &graph_up_to(5), &GlobalSfmOptions::script())?;
//! assert!(out.failure.is_none());
//! let mut rec = out.reconstruction;
//!
//! // 2) A new position arrives: registration, then triangulate only the new images.
//! let graph = graph_up_to(6);
//! let report = register_images(&mut rec, &scene.store, &graph, &RegistrationOptions::default())?;
//! let new_images = report.registered();
//! let opts = PointTriangulatorOptions { scope: TriangulationScope::Images(new_images), ..Default::default() };
//! let tri = triangulate_points(&mut rec, &graph, &opts)?;
//! println!("등록 {} 장, 새 점 {}", rec.registered_image_count(), tri.num_created);
//! # Ok::<(), cumulus3d_core::Error>(())
//! ```

#![warn(missing_docs)]

pub mod absolute_pose;
pub mod global_mapper;
pub mod math;
pub mod positioning;
pub mod registration;
pub mod rotation_averaging;
#[doc(hidden)]
pub mod synthetic;
pub mod tracks;
pub mod triangulation;
pub mod triangulator;

pub use absolute_pose::{solve_abs_pose_scored, AbsPoseBackend, CpuAbsPose, CpuPoseScorer, PoseScorer};
pub use global_mapper::{global_mapper, GlobalMapperOutput, GlobalSfmOptions};
pub use registration::{register_images, register_images_with, RegistrationOptions, RegistrationOrder, RegistrationReport};
pub use triangulation::{CpuTriangulation, ObsCheck, ObsQuery, TriProblem, TriangulationBackend};
pub use triangulator::{triangulate_points, triangulate_points_with, PointRefiner, PointTriangulatorOptions, TriangulationScope};
