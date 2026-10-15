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

//! 공통 기반: 카메라 모델, 기하, 재구성 자료 구조, 대응 그래프, 특징 저장소, RANSAC, 입출력.
//!
//! 규약 요약(모듈별 공개 항목은 크레이트 `README.md`):
//! - 자세는 `world_to_cam`(세계 → 카메라), 쿼터니언은 해밀턴, 저장 순서 w,x,y,z.
//! - 카메라 좌표계: x 오른쪽, y 아래, z 앞. 픽셀 좌표: 좌상단 화소 중심 = (0.5, 0.5).
//! - id: 카메라/rig/프레임/영상 = u32, 3D 점 = u64, 짝 = u64. 무효값은 각 자료형의 최댓값.
//!
//! # 사용 예
//!
//! 두 영상과 3D 점 하나로 재구성을 만들고, 재투영 오차를 계산한 뒤 모델 파일로 저장·적재한다.
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
//! // 세계 점 하나를 두 카메라(기선 1 m)에 투영해 2D 관측을 만든다.
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
