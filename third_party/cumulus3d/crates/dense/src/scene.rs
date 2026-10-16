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

//! 조밀화 장면: 왜곡 보정된 모델 + 영상 → 뷰(K·자세·회색/컬러 영상)와 희소점(관측 뷰 목록).
//!
//! 픽셀 규약: 장면 안에서는 정수 픽셀 인덱스가 곧 픽셀 중심 좌표다. 입력 모델(좌상단 화소 중심 = (0.5, 0.5))의
//! 주점에서 0.5 를 빼서 옮긴다.

use crate::image::ImageBuffer;
use cumulus3d_core::{CameraModelKind, Error, ImageId, Reconstruction, Result};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;

/// 장면 구성 옵션.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneOptions {
    /// 긴 변 상한(≤ 0 이면 그대로). 줄일 때만 적용한다.
    pub max_image_size: i64,
}

impl Default for SceneOptions {
    fn default() -> Self {
        Self { max_image_size: -1 }
    }
}

/// 조밀화 뷰 하나.
#[derive(Clone, Debug)]
pub struct DenseView {
    /// 모델의 영상 id.
    pub image_id: ImageId,
    /// 영상 이름(상대 경로).
    pub name: String,
    /// 너비(픽셀).
    pub width: usize,
    /// 높이(픽셀).
    pub height: usize,
    /// fx, fy, cx, cy (정수 중심 규약).
    pub k: [f64; 4],
    /// 세계 → 카메라 회전(행 우선)과 이동.
    pub r: [[f64; 3]; 3],
    /// 세계 → 카메라 이동.
    pub t: [f64; 3],
    /// 회색 8비트(0.2126R + 0.7152G + 0.0722B 반올림).
    pub gray: Arc<Vec<u8>>,
    /// 컬러(융합 색).
    pub color: Arc<ImageBuffer>,
}

impl DenseView {
    /// 카메라 중심(세계).
    pub fn center(&self) -> [f64; 3] {
        let (r, t) = (&self.r, &self.t);
        [
            -(r[0][0] * t[0] + r[1][0] * t[1] + r[2][0] * t[2]),
            -(r[0][1] * t[0] + r[1][1] * t[1] + r[2][1] * t[2]),
            -(r[0][2] * t[0] + r[1][2] * t[1] + r[2][2] * t[2]),
        ]
    }
    /// 세계 점 → 카메라 좌표.
    pub fn to_cam(&self, x: &[f64; 3]) -> [f64; 3] {
        let r = &self.r;
        [
            r[0][0] * x[0] + r[0][1] * x[1] + r[0][2] * x[2] + self.t[0],
            r[1][0] * x[0] + r[1][1] * x[1] + r[1][2] * x[2] + self.t[1],
            r[2][0] * x[0] + r[2][1] * x[1] + r[2][2] * x[2] + self.t[2],
        ]
    }
    /// 카메라 좌표 → 세계.
    pub fn to_world(&self, c: &[f64; 3]) -> [f64; 3] {
        let r = &self.r;
        let d = [c[0] - self.t[0], c[1] - self.t[1], c[2] - self.t[2]];
        [
            r[0][0] * d[0] + r[1][0] * d[1] + r[2][0] * d[2],
            r[0][1] * d[0] + r[1][1] * d[1] + r[2][1] * d[2],
            r[0][2] * d[0] + r[1][2] * d[1] + r[2][2] * d[2],
        ]
    }
    /// 카메라 방향 벡터 → 세계 방향.
    pub fn dir_to_world(&self, n: &[f64; 3]) -> [f64; 3] {
        let r = &self.r;
        [
            r[0][0] * n[0] + r[1][0] * n[1] + r[2][0] * n[2],
            r[0][1] * n[0] + r[1][1] * n[1] + r[2][1] * n[2],
            r[0][2] * n[0] + r[1][2] * n[1] + r[2][2] * n[2],
        ]
    }
    /// 세계 방향 벡터 → 카메라 방향.
    pub fn dir_to_cam(&self, d: &[f64; 3]) -> [f64; 3] {
        let r = &self.r;
        [0, 1, 2].map(|i| r[i][0] * d[0] + r[i][1] * d[1] + r[i][2] * d[2])
    }
    /// 카메라 좌표 → 픽셀 (x, y, 깊이).
    pub fn project_cam(&self, c: &[f64; 3]) -> (f64, f64, f64) {
        (self.k[0] * c[0] / c[2] + self.k[2], self.k[1] * c[1] / c[2] + self.k[3], c[2])
    }
    /// 정규 광선 K⁻¹[x, y, 1].
    pub fn ray(&self, x: f64, y: f64) -> [f64; 3] {
        [(x - self.k[2]) / self.k[0], (y - self.k[3]) / self.k[1], 1.0]
    }
    /// 자세·내부 매개변수·크기 해시(깊이맵 캐시 열쇠).
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

/// 희소점 하나(세계 좌표와 관측 뷰 색인; 같은 뷰가 두 번 나올 수 있다).
#[derive(Clone, Debug)]
pub struct ScenePoint {
    /// 세계 좌표.
    pub xyz: [f64; 3],
    /// 관측 뷰 색인.
    pub views: Vec<u32>,
}

/// 조밀화 장면.
#[derive(Clone, Debug, Default)]
pub struct DenseScene {
    /// 뷰(등록 영상 순서).
    pub views: Vec<DenseView>,
    /// 희소점.
    pub points: Vec<ScenePoint>,
}

/// RGB → 회색 8비트.
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
    /// 등록 영상 순서로 뷰를 만든다. `images` 에 없는 영상은 빠진다. 카메라는 왜곡 없는 핀홀이어야 한다.
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

    /// 왜곡 보정 작업 폴더(`sparse/`, `images/`)에서 읽는다.
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
