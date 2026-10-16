/*
 * scene.rs
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

//! Densification scene: undistorted model + images → views (K, pose, gray/color image) and sparse points (list of observing views).
//!
//! Pixel convention: inside a scene, an integer pixel index is the pixel center coordinate. The input model (top-left pixel center = (0.5,
//! 0.5))
//! is shifted by subtracting 0.5 from the principal point.

use crate::image::ImageBuffer;
use cumulus3d_core::{CameraModelKind, Error, ImageId, Reconstruction, Result};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;

/// Scene construction options.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneOptions {
    /// Upper bound on the long side (unchanged if ≤ 0). Applied only when shrinking.
    pub max_image_size: i64,
}

impl Default for SceneOptions {
    fn default() -> Self {
        Self { max_image_size: -1 }
    }
}

/// One densification view.
#[derive(Clone, Debug)]
pub struct DenseView {
    /// Image id in the model.
    pub image_id: ImageId,
    /// Image name (relative path).
    pub name: String,
    /// Width (pixels).
    pub width: usize,
    /// Height (pixels).
    pub height: usize,
    /// fx, fy, cx, cy (integer-center convention).
    pub k: [f64; 4],
    /// World → camera rotation (row-major) and translation.
    pub r: [[f64; 3]; 3],
    /// World → camera translation.
    pub t: [f64; 3],
    /// 8-bit gray (0.2126R + 0.7152G + 0.0722B, rounded).
    pub gray: Arc<Vec<u8>>,
    /// Color (fusion color).
    pub color: Arc<ImageBuffer>,
}

impl DenseView {
    /// Camera center (world).
    pub fn center(&self) -> [f64; 3] {
        let (r, t) = (&self.r, &self.t);
        [
            -(r[0][0] * t[0] + r[1][0] * t[1] + r[2][0] * t[2]),
            -(r[0][1] * t[0] + r[1][1] * t[1] + r[2][1] * t[2]),
            -(r[0][2] * t[0] + r[1][2] * t[1] + r[2][2] * t[2]),
        ]
    }
    /// World point → camera coordinates.
    pub fn to_cam(&self, x: &[f64; 3]) -> [f64; 3] {
        let r = &self.r;
        [
            r[0][0] * x[0] + r[0][1] * x[1] + r[0][2] * x[2] + self.t[0],
            r[1][0] * x[0] + r[1][1] * x[1] + r[1][2] * x[2] + self.t[1],
            r[2][0] * x[0] + r[2][1] * x[1] + r[2][2] * x[2] + self.t[2],
        ]
    }
    /// Camera coordinates → world.
    pub fn to_world(&self, c: &[f64; 3]) -> [f64; 3] {
        let r = &self.r;
        let d = [c[0] - self.t[0], c[1] - self.t[1], c[2] - self.t[2]];
        [
            r[0][0] * d[0] + r[1][0] * d[1] + r[2][0] * d[2],
            r[0][1] * d[0] + r[1][1] * d[1] + r[2][1] * d[2],
            r[0][2] * d[0] + r[1][2] * d[1] + r[2][2] * d[2],
        ]
    }
    /// Camera direction vector → world direction.
    pub fn dir_to_world(&self, n: &[f64; 3]) -> [f64; 3] {
        let r = &self.r;
        [
            r[0][0] * n[0] + r[1][0] * n[1] + r[2][0] * n[2],
            r[0][1] * n[0] + r[1][1] * n[1] + r[2][1] * n[2],
            r[0][2] * n[0] + r[1][2] * n[1] + r[2][2] * n[2],
        ]
    }
    /// World direction vector → camera direction.
    pub fn dir_to_cam(&self, d: &[f64; 3]) -> [f64; 3] {
        let r = &self.r;
        [0, 1, 2].map(|i| r[i][0] * d[0] + r[i][1] * d[1] + r[i][2] * d[2])
    }
    /// Camera coordinates → pixel (x, y, depth).
    pub fn project_cam(&self, c: &[f64; 3]) -> (f64, f64, f64) {
        (self.k[0] * c[0] / c[2] + self.k[2], self.k[1] * c[1] / c[2] + self.k[3], c[2])
    }
    /// Normalized ray K⁻¹[x, y, 1].
    pub fn ray(&self, x: f64, y: f64) -> [f64; 3] {
        [(x - self.k[2]) / self.k[0], (y - self.k[3]) / self.k[1], 1.0]
    }
    /// Hash of pose, intrinsics and size (depth map cache key).
    pub fn geometry_hash(&self) -> u64 {
        let mut b = Vec::with_capacity(160);
        b.extend_from_slice(self.name.as_bytes());
        for v in self.k.iter().chain(self.r.iter().flatten()).chain(self.t.iter()) {
            b.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        b.extend_from_slice(&(self.width as u64).to_le_bytes());
        b.extend_from_slice(&(self.height as u64).to_le_bytes());
        crate::math::hash_bytes(&b)
    }
}

/// One sparse point (world coordinates and observing view indices; the same view may appear twice).
#[derive(Clone, Debug)]
pub struct ScenePoint {
    /// World coordinates.
    pub xyz: [f64; 3],
    /// Observing view indices.
    pub views: Vec<u32>,
}

/// Densification scene.
#[derive(Clone, Debug, Default)]
pub struct DenseScene {
    /// Views (in registered image order).
    pub views: Vec<DenseView>,
    /// Sparse points.
    pub points: Vec<ScenePoint>,
}

/// RGB → 8-bit gray.
pub fn gray_of(img: &ImageBuffer) -> Vec<u8> {
    let n = img.width * img.height;
    if img.channels < 3 {
        return (0..n).map(|i| img.data[i * img.channels]).collect();
    }
    (0..n)
        .map(|i| {
            let p = &img.data[i * img.channels..i * img.channels + 3];
            (0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64).round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

impl DenseScene {
    /// Builds the views in registered image order. Images not in `images` are dropped. Cameras must be distortion-free pinhole.
    pub fn from_reconstruction(rec: &Reconstruction, images: &BTreeMap<ImageId, Arc<ImageBuffer>>, opts: &SceneOptions) -> Result<Self> {
        let mut views = Vec::new();
        let mut index: HashMap<ImageId, u32> = HashMap::new();
        for iid in rec.registered_images() {
            let Some(im) = rec.image(iid) else { continue };
            let Some(buf) = images.get(&iid) else { continue };
            let cam = rec.camera(im.camera_id).ok_or_else(|| Error::NotFound(format!("카메라 {}", im.camera_id)))?;
            if !matches!(cam.model, CameraModelKind::Pinhole | CameraModelKind::SingleFocalPinhole) {
                return Err(Error::InvalidArgument(format!("조밀화: 영상 {} 의 카메라가 핀홀이 아님(왜곡 보정 필요)", im.name)));
            }
            if buf.width as u64 != cam.width || buf.height as u64 != cam.height {
                return Err(Error::InvalidArgument(format!(
                    "조밀화: 영상 {} 크기 {}x{} != 카메라 {}x{}",
                    im.name, buf.width, buf.height, cam.width, cam.height
                )));
            }
            let Some(pose) = rec.world_to_cam(iid) else { continue };
            let rm = pose.rotation_matrix();
            let mut k = [cam.focal_length_x(), cam.focal_length_y(), cam.principal_point_x(), cam.principal_point_y()];
            let mut color = buf.clone();
            if opts.max_image_size > 0 {
                let m = opts.max_image_size as f64;
                let s = (m / buf.width as f64).min(m / buf.height as f64);
                if s < 1.0 {
                    let nw = ((buf.width as f64 * s).round() as usize).max(1);
                    let nh = ((buf.height as f64 * s).round() as usize).max(1);
                    let (sx, sy) = (nw as f64 / buf.width as f64, nh as f64 / buf.height as f64);
                    k = [k[0] * sx, k[1] * sy, k[2] * sx, k[3] * sy];
                    color = Arc::new(buf.resize_area(nw, nh));
                }
            }
            k[2] -= 0.5;
            k[3] -= 0.5;
            index.insert(iid, views.len() as u32);
            views.push(DenseView {
                image_id: iid,
                name: im.name.clone(),
                width: color.width,
                height: color.height,
                k,
                r: [[rm[(0, 0)], rm[(0, 1)], rm[(0, 2)]], [rm[(1, 0)], rm[(1, 1)], rm[(1, 2)]], [rm[(2, 0)], rm[(2, 1)], rm[(2, 2)]]],
                t: [pose.translation.x, pose.translation.y, pose.translation.z],
                gray: Arc::new(gray_of(&color)),
                color,
            });
        }
        let mut points = Vec::new();
        for (_, p) in rec.points3d() {
            let vs: Vec<u32> = p.track.iter().filter_map(|t| index.get(&t.image_id).copied()).collect();
            if vs.is_empty() {
                continue;
            }
            points.push(ScenePoint { xyz: [p.xyz.x, p.xyz.y, p.xyz.z], views: vs });
        }
        Ok(Self { views, points })
    }

    /// Reads from the undistortion working folder (`sparse/`, `images/`).
    pub fn from_workspace_dir(dir: impl AsRef<Path>, opts: &SceneOptions) -> Result<Self> {
        let dir = dir.as_ref();
        let rec = cumulus3d_core::interop::read_model(dir.join("sparse"))?;
        let mut images = BTreeMap::new();
        for iid in rec.registered_images() {
            if let Some(im) = rec.image(iid) {
                if let Ok(b) = ImageBuffer::load(dir.join("images").join(&im.name)) {
                    images.insert(iid, Arc::new(b));
                }
            }
        }
        Self::from_reconstruction(&rec, &images, opts)
    }
}
