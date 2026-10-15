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

//! 희소 재구성: 전역 SfM(회전 평균 + 위치 추정), 기존 모델에 새 영상 등록, 삼각측량.
//!
//! 입력은 `cumulus3d-core` 의 [`cumulus3d_core::FeatureStore`] (특징·두 뷰 기하)와
//! [`cumulus3d_core::MatchGraph`] (대응 그래프), 출력은 [`cumulus3d_core::Reconstruction`].
//! 진입점은 [`global_mapper()`] (첫 모델), [`register_images`] (새 영상 등록),
//! [`triangulate_points`] (점 생성·연장)이다. 모듈별 표는 크레이트 README 에 있다.
//!
//! # 사용 예
//!
//! 합성 장면(드론 3대, 위치마다 3장)의 앞 5개 위치로 첫 모델을 만들고,
//! 6번째 위치의 영상을 등록한 뒤 새 영상만 삼각측량한다.
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
//! // 1) 전역 SfM 으로 첫 모델.
//! let out = global_mapper(&scene.store, &graph_up_to(5), &GlobalSfmOptions::script())?;
//! assert!(out.failure.is_none());
//! let mut rec = out.reconstruction;
//!
//! // 2) 새 위치 도착: 등록 → 새 영상만 삼각측량.
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

pub use global_mapper::{global_mapper, GlobalMapperOutput, GlobalSfmOptions};
pub use registration::{register_images, RegistrationOptions, RegistrationOrder, RegistrationReport};
pub use triangulator::{triangulate_points, PointRefiner, PointTriangulatorOptions, TriangulationScope};
