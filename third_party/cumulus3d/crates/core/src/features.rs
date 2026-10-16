/*
 * features.rs
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
 * Third-party notices: parts of the algorithms, default parameters and data
 * formats in this file follow other open-source projects. Their copyright
 * notices and licenses are reproduced in THIRD_PARTY_NOTICES.md.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//! Feature and matching data types.

use crate::error::{Error, Result};
use crate::geometry::{Mat3, Rigid3};

/// SIFT descriptor dimension.
pub const DESCRIPTOR_DIM: usize = 128;

/// Keypoint: pixel coordinates (x, y) at input image resolution + affine shape A = [[a11, a12], [a21, a22]] (f32).
///
/// The pixel origin is the top-left corner; the top-left pixel center = (0.5, 0.5).
/// The columns of A are the keypoint's local coordinate axes mapped into image coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Keypoint {
    /// x pixel coordinate.
    pub x: f32,
    /// y pixel coordinate.
    pub y: f32,
    /// The (1,1) element of the affine shape A.
    pub a11: f32,
    /// The (1,2) element of the affine shape A.
    pub a12: f32,
    /// The (2,1) element of the affine shape A.
    pub a21: f32,
    /// The (2,2) element of the affine shape A.
    pub a22: f32,
}

impl Keypoint {
    /// A = identity matrix.
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y, a11: 1.0, a12: 0.0, a21: 0.0, a22: 1.0 }
    }
    /// (scale, orientation) → A = s·[[cos, −sin], [sin, cos]].
    pub fn from_scale_orientation(x: f32, y: f32, scale: f32, orientation: f32) -> Self {
        let (s, c) = orientation.sin_cos();
        Self { x, y, a11: scale * c, a12: -scale * s, a21: scale * s, a22: scale * c }
    }
    /// Interprets a DB row (2, 4 or 6 columns).
    pub fn from_row(row: &[f32]) -> Result<Self> {
        match row.len() {
            2 => Ok(Self::new(row[0], row[1])),
            4 => Ok(Self::from_scale_orientation(row[0], row[1], row[2], row[3])),
            6 => Ok(Self { x: row[0], y: row[1], a11: row[2], a12: row[3], a21: row[4], a22: row[5] }),
            n => Err(Error::Format(format!("키포인트 열 수 {n} 는 지원하지 않음"))),
        }
    }
    /// 6-column row `[x, y, a11, a12, a21, a22]`.
    pub fn to_row(&self) -> [f32; 6] {
        [self.x, self.y, self.a11, self.a12, self.a21, self.a22]
    }
    /// Scale of the first axis ‖(a11, a21)‖.
    pub fn scale_x(&self) -> f32 {
        (self.a11 * self.a11 + self.a21 * self.a21).sqrt()
    }
    /// Scale of the second axis ‖(a12, a22)‖.
    pub fn scale_y(&self) -> f32 {
        (self.a12 * self.a12 + self.a22 * self.a22).sqrt()
    }
    /// Mean scale.
    pub fn scale(&self) -> f32 {
        0.5 * (self.scale_x() + self.scale_y())
    }
    /// Orientation (radians).
    pub fn orientation(&self) -> f32 {
        self.a21.atan2(self.a11)
    }
    /// Shear angle (radians).
    pub fn shear(&self) -> f32 {
        (-self.a12).atan2(self.a22) - self.orientation()
    }
    /// Resolution conversion: x·sx, y·sy, first column (a11,a21)·sx, second column (a12,a22)·sy.
    pub fn rescale(&mut self, sx: f32, sy: f32) {
        self.x *= sx;
        self.y *= sy;
        self.a11 *= sx;
        self.a21 *= sx;
        self.a12 *= sy;
        self.a22 *= sy;
    }
}

/// uint8 descriptor set of one image (N × 128, row-major).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Descriptors {
    data: Vec<u8>,
}

impl Descriptors {
    /// Empty set.
    pub fn new() -> Self {
        Self::default()
    }
    /// Creates with capacity for n descriptors.
    pub fn with_capacity(n: usize) -> Self {
        Self { data: Vec::with_capacity(n * DESCRIPTOR_DIM) }
    }
    /// The length must be a multiple of 128.
    pub fn from_vec(data: Vec<u8>) -> Result<Self> {
        if !data.len().is_multiple_of(DESCRIPTOR_DIM) {
            return Err(Error::Format(format!("기술자 바이트 수 {} 가 128 의 배수가 아님", data.len())));
        }
        Ok(Self { data })
    }
    /// Number of descriptors.
    pub fn len(&self) -> usize {
        self.data.len() / DESCRIPTOR_DIM
    }
    /// Whether it is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    /// The i-th descriptor (128 bytes).
    pub fn row(&self, i: usize) -> &[u8] {
        &self.data[i * DESCRIPTOR_DIM..(i + 1) * DESCRIPTOR_DIM]
    }
    /// The i-th descriptor (mutable).
    pub fn row_mut(&mut self, i: usize) -> &mut [u8] {
        &mut self.data[i * DESCRIPTOR_DIM..(i + 1) * DESCRIPTOR_DIM]
    }
    /// Appends one descriptor.
    pub fn push(&mut self, d: &[u8; DESCRIPTOR_DIM]) {
        self.data.extend_from_slice(d);
    }
    /// All bytes, row-major.
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
    /// Converts to a byte vector.
    pub fn into_vec(self) -> Vec<u8> {
        self.data
    }
    /// Keeps only the first n.
    pub fn truncate(&mut self, n: usize) {
        self.data.truncate(n * DESCRIPTOR_DIM);
    }
}

/// Pair of keypoint indices in two images (image1 idx, image2 idx).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct FeatureMatch {
    /// Keypoint index in image 1.
    pub idx1: u32,
    /// Keypoint index in image 2.
    pub idx2: u32,
}

impl FeatureMatch {
    /// Creates from an index pair.
    pub fn new(idx1: u32, idx2: u32) -> Self {
        Self { idx1, idx2 }
    }
    /// Match with the two indices swapped.
    pub fn swapped(&self) -> Self {
        Self { idx1: self.idx2, idx2: self.idx1 }
    }
}

/// Kind of two-view geometry configuration (DB config value).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum TwoViewGeometryConfig {
    #[default]
    /// Undefined (not verified or failed).
    Undefined = 0,
    /// Degenerate (unusable).
    Degenerate = 1,
    /// Calibrated (E model).
    Calibrated = 2,
    /// Uncalibrated (F model).
    Uncalibrated = 3,
    /// Planar (H model, with baseline).
    Planar = 4,
    /// Pure rotation (H model, no baseline).
    Panoramic = 5,
    /// Planar or pure rotation.
    PlanarOrRotation = 6,
    /// Watermark (fixed pattern at the image border).
    Watermark = 7,
    /// Multiple models.
    Multiple = 8,
    /// Rig calibrated.
    CalibratedRig = 9,
}

impl TwoViewGeometryConfig {
    /// Integer value -> kind.
    pub fn from_i32(v: i32) -> Option<Self> {
        use TwoViewGeometryConfig::*;
        Some(match v {
            0 => Undefined,
            1 => Degenerate,
            2 => Calibrated,
            3 => Uncalibrated,
            4 => Planar,
            5 => Panoramic,
            6 => PlanarOrRotation,
            7 => Watermark,
            8 => Multiple,
            9 => CalibratedRig,
            _ => return None,
        })
    }
    /// Integer value used for storage.
    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

/// Two-view geometry of one pair. Matrices are in the "image1 -> image2" direction (x2ᵀ F x1 = 0, x2 ~ H x1).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TwoViewGeometry {
    /// Configuration kind.
    pub config: TwoViewGeometryConfig,
    /// Essential matrix E.
    pub e: Option<Mat3>,
    /// Fundamental matrix F.
    pub f: Option<Mat3>,
    /// Homography H.
    pub h: Option<Mat3>,
    /// Relative pose (image1 camera -> image2 camera).
    pub cam1_to_cam2: Option<Rigid3>,
    /// Inlier matches.
    pub inlier_matches: Vec<FeatureMatch>,
    /// Median triangulation angle (radians). Auxiliary information that is not stored.
    pub tri_angle: Option<f64>,
}

impl TwoViewGeometry {
    /// Direction reversal: F -> Fᵀ, E -> Eᵀ, H -> H⁻¹, pose -> inverse, inlier columns swapped.
    pub fn inverted(&self) -> Self {
        Self {
            config: self.config,
            e: self.e.map(|m| m.transpose()),
            f: self.f.map(|m| m.transpose()),
            h: self.h.and_then(|m| m.try_inverse()),
            cam1_to_cam2: self.cam1_to_cam2.map(|p| p.inverse()),
            inlier_matches: self.inlier_matches.iter().map(|m| m.swapped()).collect(),
            tri_angle: self.tri_angle,
        }
    }
    /// In-place direction reversal (see `inverted`).
    pub fn invert(&mut self) {
        *self = self.inverted();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keypoint_shape() {
        let k = Keypoint::from_scale_orientation(1600.5, 1200.25, 2.0, 0.3);
        assert!((k.scale() - 2.0).abs() < 1e-6);
        assert!((k.orientation() - 0.3).abs() < 1e-6);
        assert!(k.shear().abs() < 1e-6);
        let mut k2 = k;
        k2.rescale(4000.0 / 3200.0, 3000.0 / 2400.0);
        assert!((k2.x - 2000.625).abs() < 1e-5 * 2000.0);
        assert!((k2.y - 1500.3125).abs() < 1e-5 * 1500.0);
        assert!((k2.scale() - 2.5).abs() < 1e-5);
        assert!((k2.orientation() - 0.3).abs() < 1e-5);
        assert_eq!(Keypoint::from_row(&k.to_row()).unwrap(), k);
        assert!(Keypoint::from_row(&[1.0; 3]).is_err());
    }

    #[test]
    fn tvg_invert() {
        let h = Mat3::new(1.0, 0.1, 5.0, 0.0, 1.2, -3.0, 0.001, 0.0, 1.0);
        let g = TwoViewGeometry {
            config: TwoViewGeometryConfig::Calibrated,
            h: Some(h),
            f: Some(h),
            inlier_matches: vec![FeatureMatch::new(1, 2)],
            ..Default::default()
        };
        let gi = g.inverted();
        assert!((gi.h.unwrap() * h - Mat3::identity()).norm() < 1e-12);
        assert_eq!(gi.f.unwrap(), h.transpose());
        assert_eq!(gi.inlier_matches[0], FeatureMatch::new(2, 1));
        let gii = gi.inverted();
        assert!((gii.h.unwrap() - h).norm() < 1e-12);
    }
}
