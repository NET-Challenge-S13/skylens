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

//! Image reading and SIFT feature extraction.
//!
//! - [`gray`]: decoding, grayscale conversion, downscaling, 90° rotation.
//! - [`exif_info`], [`camera_init`]: EXIF focal-length rules and initial camera values.
//! - [`sift`]: deterministic SIFT (feature-count limit per level); the backend can be swapped via [`sift::SiftEngine`].
//! - [`extract`]: camera grouping + incremental `FeatureStore` writes.
//!
//! # Usage example
//!
//! ```
//! use cumulus3d_core::FeatureStore;
//! use cumulus3d_features::{
//!     CpuSift, ExifInfo, ExtractionOptions, FeatureExtractor, GrayImage, ImageSource, ImageStatus, SiftEngine,
//!     SiftOptions,
//! };
//!
//! // Synthetic image: three bright blobs.
//! let blob = |x: usize, y: usize, cx: f32, cy: f32, s: f32| {
//!     let (dx, dy) = (x as f32 - cx, y as f32 - cy);
//!     (-(dx * dx + dy * dy) / (2.0 * s * s)).exp()
//! };
//! let img = GrayImage::from_f32(320, 240, |x, y| {
//!     0.2 + 0.6 * (blob(x, y, 80.0, 60.0, 6.0) + blob(x, y, 200.0, 150.0, 9.0) + blob(x, y, 260.0, 70.0, 5.0))
//! });
//!
//! // 1) Call the SIFT backend directly: feature i <-> descriptor row i.
//! let out = CpuSift::new().extract(&img, &SiftOptions::default())?;
//! assert!(!out.is_empty());
//! assert_eq!(out.features.len(), out.descriptors.len());
//!
//! // 2) Write to the store (camera assignment + keypoints and descriptors).
//! let store = FeatureStore::new();
//! let src = ImageSource { name: "camF/0001.jpg".into(), gray: img, exif: ExifInfo::default() };
//! let reports = FeatureExtractor::new().extract_inputs(&store, vec![src], &ExtractionOptions::default())?;
//! assert!(matches!(reports[0].status, ImageStatus::Extracted { .. }));
//! assert_eq!(store.num_images(), 1);
//! # Ok::<(), cumulus3d_core::Error>(())
//! ```

#![warn(missing_docs)]

pub mod camera_init;
pub mod exif_info;
pub mod extract;
pub mod gray;
pub mod sift;

pub use camera_init::{infer_focal, init_camera, lookup_sensor_width, FocalEstimate};
pub use exif_info::ExifInfo;
pub use extract::{
    extract_for_camera, image_folder, read_image_list, CameraMode, ExtractionOptions, FeatureExtractor, ImageReport, ImageSource,
    ImageStatus, ReaderOptions,
};
pub use gray::{limited_size, read_gray, rgb_to_gray, rotate_keypoint_ccw, GrayImage};
pub use sift::{CpuSift, DescriptorNormalization, FeatureSelection, SiftEngine, SiftFeature, SiftOptions, SiftOutput};
