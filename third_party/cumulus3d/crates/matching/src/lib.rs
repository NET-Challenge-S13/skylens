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

//! Descriptor matching and two-view geometry verification.
//!
<<<<<<< skylens (vendored)
//! - [`descriptor`]: 정수 내적 무차별 매칭(비율·거리·교차 검사), [`MatcherBackend`] 로 GPU 교체 가능.
//! - [`estimators`], [`essential`]: 5점 E, 7/8점 F, DLT H, 평행이동 해법과 잔차(core RANSAC 추정기).
//! - [`two_view`]: E/F/H LO-RANSAC 과 구성 판정, 워터마크.
//! - [`batched`]: 짝 묶음 검증 백엔드([`TwoViewBackend`])와 가설 일괄 채점(GPU 교체용).
//! - [`pose`]: E/H 분해, 중점 삼각측량 cheirality, 상대 자세(sfm 공용).
//! - [`pairs`], [`pipeline`]: 짝 목록 파일과 FeatureStore 기록.
||||||| base b9f4a9bda
//! - [`descriptor`]: 정수 내적 무차별 매칭(비율·거리·교차 검사), [`MatcherBackend`] 로 GPU 교체 가능.
//! - [`estimators`], [`essential`]: 5점 E, 7/8점 F, DLT H, 평행이동 해법과 잔차(core RANSAC 추정기).
//! - [`two_view`]: E/F/H LO-RANSAC 과 구성 판정, 워터마크.
//! - [`pose`]: E/H 분해, 중점 삼각측량 cheirality, 상대 자세(sfm 공용).
//! - [`pairs`], [`pipeline`]: 짝 목록 파일과 FeatureStore 기록.
=======
//! - [`descriptor`]: brute-force integer dot-product matching (ratio, distance, cross-check); the GPU can be swapped
//!   in via [`MatcherBackend`].
//! - [`estimators`], [`essential`]: 5-point E, 7/8-point F, DLT H, translation solver and residuals (core RANSAC estimators).
//! - [`two_view`]: E/F/H LO-RANSAC, configuration decision, watermark.
//! - [`pose`]: E/H decomposition, midpoint triangulation cheirality, relative pose (shared with sfm).
//! - [`pairs`], [`pipeline`]: pair list file and FeatureStore writes.
>>>>>>> upstream v0.5.0
//!
//! # Usage example
//!
//! ```
//! use cumulus3d_core::{Camera, CameraModelKind, Descriptors, FeatureMatch, Keypoint, TwoViewGeometryConfig};
//! use cumulus3d_matching::{estimate_two_view, CpuMatcher, DescriptorMatchOptions, MatcherBackend, TwoViewOptions};
//!
//! // 1) Descriptor matching: descriptor i is a mutually orthogonal vector with only components 4i..4i+4 set to 255.
//! let make = |order: &[usize]| {
//!     let mut d = Descriptors::new();
//!     for &i in order {
//!         let mut row = [0u8; 128];
//!         row[4 * i..4 * i + 4].fill(255);
//!         d.push(&row);
//!     }
//!     d
//! };
//! let d1 = make(&(0..32).collect::<Vec<_>>());
//! let d2 = make(&(0..32).rev().collect::<Vec<_>>());
//! let m = CpuMatcher::default().match_descriptors(&d1, &d2, &DescriptorMatchOptions::default(), 1000);
//! assert_eq!(m.len(), 32);
//! assert!(m.iter().all(|x| x.idx2 == 31 - x.idx1));
//!
//! // 2) Two-view geometry: two images of the same pinhole camera translated by 1 along the x axis.
//! let mut cam = Camera::from_focal(CameraModelKind::Pinhole, 500.0, 640, 480);
//! cam.focal_from_prior = true; // if the focal length is known, use the calibrated (E) path.
//! let (mut kps1, mut kps2, mut matches) = (Vec::new(), Vec::new(), Vec::new());
//! for i in 0..60u32 {
//!     let (x, y, z) = (((i * 37) % 23) as f64 * 0.2 - 2.2, ((i * 17) % 19) as f64 * 0.2 - 1.8, 4.0 + ((i * 7) % 11) as f64 * 0.3);
//!     let px = |cx: f64| Keypoint::new((500.0 * cx / z + 320.0) as f32, (500.0 * y / z + 240.0) as f32);
//!     kps1.push(px(x));
//!     kps2.push(px(x - 1.0));
//!     matches.push(FeatureMatch::new(i, i));
//! }
//! let tvg = estimate_two_view(&cam, &kps1, &cam, &kps2, &matches, &TwoViewOptions::default());
//! assert_eq!(tvg.config, TwoViewGeometryConfig::Calibrated);
//! assert!(tvg.inlier_matches.len() >= 50);
//! ```

#![warn(missing_docs)]

pub mod batched;
pub mod descriptor;
pub mod essential;
pub mod estimators;
pub mod linalg;
pub mod pairs;
pub mod pipeline;
pub mod poly;
pub mod pose;
pub mod two_view;

pub use batched::{verify_pairs_batched, BatchScorer, CpuBatchScorer, CpuTwoView, ResidualSet, ScoreRequest, TwoViewBackend, VerifyJob};
pub use descriptor::{AcceptRule, CpuMatcher, DescriptorMatchOptions, MatcherBackend, Top2};
pub use essential::{essential_eight_point, essential_five_point};
pub use estimators::{
    fundamental_eight_point, fundamental_seven_point, homography_dlt, homography_transfer_error_sq, sampson_error_sq,
    EssentialFivePointEstimator, Fundamental7PtEstimator, FundamentalEightPointEstimator, HomographyEstimator, TranslationEstimator,
};
pub use pairs::{parse_pair_list, read_pair_list, PairList};
pub use pipeline::{match_pair_list_file, match_pairs, match_pairs_with, verify_pair, MatchingStats, PairMatchingOptions};
pub use pose::{
    decompose_essential, decompose_homography, pose_from_essential, pose_from_homography, recover_two_view_pose,
    refit_and_estimate_relative_pose, triangulate_midpoint,
};
pub use two_view::{
    decide_calibrated, decide_uncalibrated, estimate_two_view, finalize_geometry, is_watermark, Decision, MaskChoice, ModelOutcome,
    TwoViewOptions,
};
