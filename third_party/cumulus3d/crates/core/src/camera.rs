/*
 * camera.rs
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

//! 카메라 모델.
//!
//! 좌표 규약: 카메라 좌표 (X, Y, Z) → 정규화 평면 (u, v) = (X/Z, Y/Z) → 왜곡 → 픽셀.
//! 픽셀 원점은 영상 좌상단 모서리, 좌상단 화소 중심 = (0.5, 0.5).

use crate::error::{Error, Result};
use crate::geometry::{Mat3, Vec2, Vec3};
use crate::ids::{CameraId, INVALID_CAMERA_ID};
use nalgebra::{Matrix2, Matrix2x3};

/// 지원 카메라 모델. 숫자 id 는 모델 파일 형식의 모델 번호.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CameraModelKind {
    /// 단일 초점 핀홀(f, cx, cy).
    SingleFocalPinhole = 0,
    /// 핀홀(fx, fy, cx, cy).
    Pinhole = 1,
    /// 단일 초점 + 방사 왜곡 1항(f, cx, cy, k).
    SingleFocalRadial = 2,
    /// 단일 초점 + 방사 왜곡 2항(f, cx, cy, k1, k2).
    Radial = 3,
    /// 방사 2항 + 접선 2항(fx, fy, cx, cy, k1, k2, p1, p2).
    OpenCv = 4,
}

/// 왜곡 제거 뉴턴 반복 최대 횟수.
pub const UNDISTORT_MAX_ITERS: usize = 100;
/// 수렴 판정: ‖step‖² < 1e−10.
pub const UNDISTORT_STEP_SQ_TOL: f64 = 1e-10;

/// 지원 모델의 파일 형식 이름(id 순서).
const MODEL_NAMES: [&str; 5] = ["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"];

impl CameraModelKind {
    /// 모델 번호 → 종류.
    pub fn from_id(id: i32) -> Result<Self> {
        match id {
            0 => Ok(Self::SingleFocalPinhole),
            1 => Ok(Self::Pinhole),
            2 => Ok(Self::SingleFocalRadial),
            3 => Ok(Self::Radial),
            4 => Ok(Self::OpenCv),
            _ => Err(Error::Unsupported(format!("카메라 모델 id {id}"))),
        }
    }
    /// 모델 번호.
    pub fn id(self) -> i32 {
        self as i32
    }
    /// 파일 형식 모델 이름(예: `PINHOLE`).
    pub fn name(self) -> &'static str {
        MODEL_NAMES[self as usize]
    }
    /// 파일 형식 모델 이름 → 종류.
    pub fn from_name(name: &str) -> Result<Self> {
        match MODEL_NAMES.iter().position(|n| *n == name) {
            Some(i) => Self::from_id(i as i32),
            None => Err(Error::Format(format!("알 수 없는 카메라 모델 이름 {name}"))),
        }
    }
    /// 파라미터 개수.
    pub fn num_params(self) -> usize {
        match self {
            Self::SingleFocalPinhole => 3,
            Self::Pinhole => 4,
            Self::SingleFocalRadial => 4,
            Self::Radial => 5,
            Self::OpenCv => 8,
        }
    }
    /// 파라미터 이름(쉼표 구분).
    pub fn params_info(self) -> &'static str {
        match self {
            Self::SingleFocalPinhole => "f, cx, cy",
            Self::Pinhole => "fx, fy, cx, cy",
            Self::SingleFocalRadial => "f, cx, cy, k",
            Self::Radial => "f, cx, cy, k1, k2",
            Self::OpenCv => "fx, fy, cx, cy, k1, k2, p1, p2",
        }
    }
    /// `params` 안 초점 파라미터 위치.
    pub fn focal_slots(self) -> &'static [usize] {
        match self {
            Self::SingleFocalPinhole | Self::SingleFocalRadial | Self::Radial => &[0],
            Self::Pinhole | Self::OpenCv => &[0, 1],
        }
    }
    /// `params` 안 주점 파라미터 위치.
    pub fn pp_slots(self) -> &'static [usize] {
        match self {
            Self::SingleFocalPinhole | Self::SingleFocalRadial | Self::Radial => &[1, 2],
            Self::Pinhole | Self::OpenCv => &[2, 3],
        }
    }
    /// `params` 안 왜곡(추가) 파라미터 위치.
    pub fn extra_param_slots(self) -> &'static [usize] {
        match self {
            Self::SingleFocalPinhole | Self::Pinhole => &[],
            Self::SingleFocalRadial => &[3],
            Self::Radial => &[3, 4],
            Self::OpenCv => &[4, 5, 6, 7],
        }
    }
}

/// 카메라(내부 파라미터). `focal_from_prior` 는 메모리 전용(모델 파일에 저장 안 됨).
#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    /// 카메라 id.
    pub camera_id: CameraId,
    /// 카메라 모델.
    pub model: CameraModelKind,
    /// 영상 너비(픽셀).
    pub width: u64,
    /// 영상 높이(픽셀).
    pub height: u64,
    /// 모델 파라미터(순서는 `CameraModelKind::params_info`).
    pub params: Vec<f64>,
    /// 초점이 EXIF 등 사전 정보에서 왔는지(보정 경로 선택용).
    pub focal_from_prior: bool,
}

impl Camera {
    /// 파라미터 개수를 검증해 생성.
    pub fn new(camera_id: CameraId, model: CameraModelKind, width: u64, height: u64, params: Vec<f64>) -> Result<Self> {
        if params.len() != model.num_params() {
            return Err(Error::Format(format!(
                "카메라 {camera_id}: 모델 {} 의 파라미터 개수 {} != {}",
                model.name(),
                params.len(),
                model.num_params()
            )));
        }
        Ok(Self { camera_id, model, width, height, params, focal_from_prior: false })
    }

    /// 초점 f 로 초기화: 초점 = f, 주점 = (W/2, H/2), 왜곡 0.
    pub fn from_focal(model: CameraModelKind, focal: f64, width: u64, height: u64) -> Self {
        let mut params = vec![0.0; model.num_params()];
        for &i in model.focal_slots() {
            params[i] = focal;
        }
        let pp = model.pp_slots();
        params[pp[0]] = width as f64 / 2.0;
        params[pp[1]] = height as f64 / 2.0;
        Self { camera_id: INVALID_CAMERA_ID, model, width, height, params, focal_from_prior: false }
    }

    /// 파라미터 개수가 모델과 맞는지.
    pub fn verify_params(&self) -> bool {
        self.params.len() == self.model.num_params()
    }

    /// 평균 초점거리.
    pub fn mean_focal(&self) -> f64 {
        let idx = self.model.focal_slots();
        idx.iter().map(|&i| self.params[i]).sum::<f64>() / idx.len() as f64
    }
    /// x 초점거리.
    pub fn focal_length_x(&self) -> f64 {
        self.params[self.model.focal_slots()[0]]
    }
    /// y 초점거리(단일 초점 모델이면 x 와 같음).
    pub fn focal_length_y(&self) -> f64 {
        let idx = self.model.focal_slots();
        self.params[idx[idx.len() - 1]]
    }
    /// 모든 초점 파라미터를 f 로 설정.
    pub fn set_focal_length(&mut self, f: f64) {
        for &i in self.model.focal_slots() {
            self.params[i] = f;
        }
    }
    /// 주점 x.
    pub fn principal_point_x(&self) -> f64 {
        self.params[self.model.pp_slots()[0]]
    }
    /// 주점 y.
    pub fn principal_point_y(&self) -> f64 {
        self.params[self.model.pp_slots()[1]]
    }
    /// 주점 설정.
    pub fn set_principal_point(&mut self, cx: f64, cy: f64) {
        let pp = self.model.pp_slots();
        self.params[pp[0]] = cx;
        self.params[pp[1]] = cy;
    }
    /// 왜곡(추가) 파라미터 복사본.
    pub fn extra_params(&self) -> Vec<f64> {
        self.model.extra_param_slots().iter().map(|&i| self.params[i]).collect()
    }
    /// 왜곡 없음 판정: 추가 파라미터가 모두 0.
    pub fn is_undistorted(&self) -> bool {
        self.model.extra_param_slots().iter().all(|&i| self.params[i] == 0.0)
    }
    /// 보정 행렬 K (왜곡 무시).
    pub fn calibration_matrix(&self) -> Mat3 {
        Mat3::new(self.focal_length_x(), 0.0, self.principal_point_x(), 0.0, self.focal_length_y(), self.principal_point_y(), 0.0, 0.0, 1.0)
    }
    /// 비정상 파라미터 판정.
    pub fn has_implausible_params(&self, min_focal_ratio: f64, max_focal_ratio: f64, max_extra_param: f64) -> bool {
        let (w, h) = (self.width as f64, self.height as f64);
        let (cx, cy) = (self.principal_point_x(), self.principal_point_y());
        if cx < 0.0 || cx > w || cy < 0.0 || cy > h {
            return true;
        }
        let m = w.max(h);
        for &i in self.model.focal_slots() {
            let r = self.params[i] / m;
            if r < min_focal_ratio || r > max_focal_ratio {
                return true;
            }
        }
        self.model.extra_param_slots().iter().any(|&i| self.params[i].abs() > max_extra_param)
    }

    /// 균일 크기 조정: 초점·주점에 s 곱, 크기는 반올림. 왜곡 계수 불변.
    pub fn rescale(&mut self, scale: f64) {
        let nw = (self.width as f64 * scale).round().max(1.0) as u64;
        let nh = (self.height as f64 * scale).round().max(1.0) as u64;
        self.scale_params(scale, scale);
        self.width = nw;
        self.height = nh;
    }

    /// 새 크기로 조정: 축별 sx = W'/W, sy = H'/H 를 초점·주점에 곱함.
    pub fn rescale_to(&mut self, new_width: u64, new_height: u64) {
        let sx = new_width as f64 / self.width as f64;
        let sy = new_height as f64 / self.height as f64;
        self.scale_params(sx, sy);
        self.width = new_width;
        self.height = new_height;
    }

    fn scale_params(&mut self, sx: f64, sy: f64) {
        let f = self.model.focal_slots();
        if f.len() == 1 {
            // 설계 결정: 단일 초점 모델의 비등방 축척은 두 축 평균을 사용.
            self.params[f[0]] *= 0.5 * (sx + sy);
        } else {
            self.params[f[0]] *= sx;
            self.params[f[1]] *= sy;
        }
        let pp = self.model.pp_slots();
        self.params[pp[0]] *= sx;
        self.params[pp[1]] *= sy;
    }

    /// 픽셀 오차 임계를 정규화 평면 임계로: τ / 평균 초점.
    pub fn img_to_cam_threshold(&self, threshold_px: f64) -> f64 {
        threshold_px / self.mean_focal()
    }

    /// 정규화 평면 왜곡량 Δ(u,v) 와 ∂Δ/∂(u,v).
    fn distortion(&self, u: f64, v: f64) -> (f64, f64, Matrix2<f64>) {
        let p = &self.params;
        match self.model {
            CameraModelKind::SingleFocalPinhole | CameraModelKind::Pinhole => (0.0, 0.0, Matrix2::zeros()),
            CameraModelKind::SingleFocalRadial | CameraModelKind::Radial => {
                let (k1, k2) = if self.model == CameraModelKind::SingleFocalRadial { (p[3], 0.0) } else { (p[3], p[4]) };
                let r2 = u * u + v * v;
                let radial = k1 * r2 + k2 * r2 * r2;
                let g = 2.0 * k1 + 4.0 * k2 * r2; // ∂radial/∂u = g·u
                let du = u * radial;
                let dv = v * radial;
                let j = Matrix2::new(radial + u * g * u, u * g * v, v * g * u, radial + v * g * v);
                (du, dv, j)
            }
            CameraModelKind::OpenCv => {
                let (k1, k2, p1, p2) = (p[4], p[5], p[6], p[7]);
                let r2 = u * u + v * v;
                let radial = k1 * r2 + k2 * r2 * r2;
                let du = u * radial + 2.0 * p1 * u * v + p2 * (r2 + 2.0 * u * u);
                let dv = v * radial + 2.0 * p2 * u * v + p1 * (r2 + 2.0 * v * v);
                let du_du = radial + u * (2.0 * k1 * u + 4.0 * k2 * r2 * u) + 2.0 * p1 * v + 6.0 * p2 * u;
                let du_dv = u * (2.0 * k1 * v + 4.0 * k2 * r2 * v) + 2.0 * p1 * u + 2.0 * p2 * v;
                let dv_du = v * (2.0 * k1 * u + 4.0 * k2 * r2 * u) + 2.0 * p2 * v + 2.0 * p1 * u;
                let dv_dv = radial + v * (2.0 * k1 * v + 4.0 * k2 * r2 * v) + 2.0 * p2 * u + 6.0 * p1 * v;
                (du, dv, Matrix2::new(du_du, du_dv, dv_du, dv_dv))
            }
        }
    }

    fn fxfycxcy(&self) -> (f64, f64, f64, f64) {
        (self.focal_length_x(), self.focal_length_y(), self.principal_point_x(), self.principal_point_y())
    }

    /// 정규화 평면 (u, v) → 픽셀 (왜곡 포함).
    pub fn normalized_to_img(&self, uv: &Vec2) -> Vec2 {
        let (fx, fy, cx, cy) = self.fxfycxcy();
        let (du, dv, _) = self.distortion(uv.x, uv.y);
        Vec2::new(fx * (uv.x + du) + cx, fy * (uv.y + dv) + cy)
    }

    /// 카메라 좌표 3D 점 → 픽셀. Z < f64 ε 이면 None.
    pub fn cam_to_img(&self, xc: &Vec3) -> Option<Vec2> {
        if xc.z < f64::EPSILON {
            return None;
        }
        Some(self.normalized_to_img(&Vec2::new(xc.x / xc.z, xc.y / xc.z)))
    }

    /// 픽셀 → 정규화 평면(왜곡 제거). 뉴턴 100회 내 미수렴이면 None.
    pub fn img_to_normalized(&self, xy: &Vec2) -> Option<Vec2> {
        let (fx, fy, cx, cy) = self.fxfycxcy();
        let p0 = Vec2::new((xy.x - cx) / fx, (xy.y - cy) / fy);
        if self.model.extra_param_slots().is_empty() || self.is_undistorted() {
            return if p0.x.is_finite() && p0.y.is_finite() { Some(p0) } else { None };
        }
        let mut p = p0;
        for _ in 0..UNDISTORT_MAX_ITERS {
            let (du, dv, jd) = self.distortion(p.x, p.y);
            let j = Matrix2::identity() + jd;
            let res = Vec2::new(p.x + du - p0.x, p.y + dv - p0.y);
            let mut step = j.lu().solve(&res)?;
            if !step.x.is_finite() || !step.y.is_finite() {
                return None;
            }
            let rho2 = (p.norm_squared() * 0.01).max(0.01);
            let s2 = step.norm_squared();
            if s2 > rho2 {
                step *= (rho2 / s2).sqrt();
            }
            p -= step;
            if step.norm_squared() < UNDISTORT_STEP_SQ_TOL {
                return Some(p);
            }
        }
        None
    }

    /// 픽셀 → 단위 광선 (u, v, 1)/‖·‖.
    pub fn img_to_ray(&self, xy: &Vec2) -> Option<Vec3> {
        self.img_to_normalized(xy).map(|uv| Vec3::new(uv.x, uv.y, 1.0).normalize())
    }

    /// 투영과 해석적 야코비안.
    ///
    /// 반환: (픽셀, ∂픽셀/∂카메라좌표 2×3). `d_params` 가 주어지면 행 우선 2×num_params
    /// (첫 행 = x, 둘째 행 = y, 열 = 파라미터 순서)로 ∂픽셀/∂파라미터를 채운다.
    /// Z < ε 이면 None (BA 는 이때 잔차·야코비안 0 으로 둔다).
    pub fn cam_to_img_with_jacobian(&self, xc: &Vec3, d_params: Option<&mut [f64]>) -> Option<(Vec2, Matrix2x3<f64>)> {
        if xc.z < f64::EPSILON {
            return None;
        }
        let iz = 1.0 / xc.z;
        let u = xc.x * iz;
        let v = xc.y * iz;
        let (fx, fy, cx, cy) = self.fxfycxcy();
        let (du, dv, jd) = self.distortion(u, v);
        let xd = u + du;
        let yd = v + dv;
        let pix = Vec2::new(fx * xd + cx, fy * yd + cy);
        let d = Matrix2::identity() + jd;
        let nm = Matrix2x3::new(iz, 0.0, -xc.x * iz * iz, 0.0, iz, -xc.y * iz * iz);
        let fd = Matrix2::new(fx, 0.0, 0.0, fy);
        let j_point = fd * d * nm;
        if let Some(out) = d_params {
            let n = self.model.num_params();
            assert!(out.len() >= 2 * n, "d_params 길이 부족");
            out[..2 * n].iter_mut().for_each(|x| *x = 0.0);
            let r2 = u * u + v * v;
            let (rx, ry) = out.split_at_mut(n);
            match self.model {
                CameraModelKind::SingleFocalPinhole => {
                    rx[0] = u;
                    rx[1] = 1.0;
                    ry[0] = v;
                    ry[2] = 1.0;
                }
                CameraModelKind::Pinhole => {
                    rx[0] = u;
                    rx[2] = 1.0;
                    ry[1] = v;
                    ry[3] = 1.0;
                }
                CameraModelKind::SingleFocalRadial | CameraModelKind::Radial => {
                    let f = fx;
                    rx[0] = xd;
                    rx[1] = 1.0;
                    rx[3] = f * u * r2;
                    ry[0] = yd;
                    ry[2] = 1.0;
                    ry[3] = f * v * r2;
                    if self.model == CameraModelKind::Radial {
                        rx[4] = f * u * r2 * r2;
                        ry[4] = f * v * r2 * r2;
                    }
                }
                CameraModelKind::OpenCv => {
                    rx[0] = xd;
                    rx[2] = 1.0;
                    rx[4] = fx * u * r2;
                    rx[5] = fx * u * r2 * r2;
                    rx[6] = fx * 2.0 * u * v;
                    rx[7] = fx * (r2 + 2.0 * u * u);
                    ry[1] = yd;
                    ry[3] = 1.0;
                    ry[4] = fy * v * r2;
                    ry[5] = fy * v * r2 * r2;
                    ry[6] = fy * (r2 + 2.0 * v * v);
                    ry[7] = fy * 2.0 * u * v;
                }
            }
        }
        Some((pix, j_point))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_pcg::Pcg64;

    fn opencv(params: [f64; 8]) -> Camera {
        Camera::new(1, CameraModelKind::OpenCv, 1000, 800, params.to_vec()).unwrap()
    }

    #[test]
    fn opencv_hand_computed() {
        let cam = opencv([1000.0, 1000.0, 500.0, 400.0, 0.1, 0.01, 0.001, -0.002]);
        let p = cam.cam_to_img(&Vec3::new(0.2, -0.1, 1.0)).unwrap();
        assert!((p.x - 700.705).abs() < 1e-9, "{}", p.x);
        assert!((p.y - 299.6475).abs() < 1e-9, "{}", p.y);
        assert!(cam.cam_to_img(&Vec3::new(0.0, 0.0, 0.0)).is_none());
        assert!(cam.cam_to_img(&Vec3::new(0.0, 0.0, -1.0)).is_none());
    }

    #[test]
    fn roundtrip_all_models() {
        let mut rng = Pcg64::seed_from_u64(42);
        for model in [
            CameraModelKind::SingleFocalPinhole,
            CameraModelKind::Pinhole,
            CameraModelKind::SingleFocalRadial,
            CameraModelKind::Radial,
            CameraModelKind::OpenCv,
        ] {
            for _ in 0..300 {
                let mut cam = Camera::from_focal(model, 900.0 + rng.random_range(0.0..200.0), 1000, 800);
                let extra = model.extra_param_slots();
                let ranges = [0.3, 0.1, 0.01, 0.01];
                for (k, &i) in extra.iter().enumerate() {
                    cam.params[i] = rng.random_range(-ranges[k]..ranges[k]);
                }
                if model == CameraModelKind::OpenCv || model == CameraModelKind::Pinhole {
                    cam.params[1] = cam.params[0] * 1.01;
                }
                let uv = Vec2::new(rng.random_range(-0.8..0.8), rng.random_range(-0.8..0.8));
                // 왜곡 사상이 국소적으로 가역이 아닌 표본(야코비안 행렬식 ≤ 0.2)은 해가 유일하지 않아 제외.
                if (Matrix2::identity() + cam.distortion(uv.x, uv.y).2).determinant() <= 0.2 {
                    continue;
                }
                let px = cam.normalized_to_img(&uv);
                let uv2 = cam.img_to_normalized(&px).expect("수렴해야 함");
                assert!((uv2 - uv).norm() <= 1e-8, "{model:?} {uv:?} {uv2:?}");
                let px2 = cam.normalized_to_img(&uv2);
                assert!((px2 - px).norm() <= 1e-6);
            }
        }
    }

    #[test]
    fn undistort_failure_no_nan() {
        let cam = opencv([1000.0, 1000.0, 500.0, 400.0, -5.0, 0.0, 0.0, 0.0]);
        // 큰 반경: k1 = −5 이면 F(p) 의 최대값(r≈0.26 에서 ~0.17)을 넘는 p0 는 해가 없다.
        let r = cam.img_to_normalized(&Vec2::new(500.0 + 900.0, 400.0 + 900.0));
        assert!(r.is_none());
    }

    #[test]
    fn distortion_jacobian_numeric() {
        let mut rng = Pcg64::seed_from_u64(7);
        for _ in 0..100 {
            let cam = opencv([
                1000.0,
                1000.0,
                500.0,
                400.0,
                rng.random_range(-0.3..0.3),
                rng.random_range(-0.1..0.1),
                rng.random_range(-0.01..0.01),
                rng.random_range(-0.01..0.01),
            ]);
            let (u, v) = (rng.random_range(-0.8..0.8), rng.random_range(-0.8..0.8));
            let (_, _, j) = cam.distortion(u, v);
            let h = 1e-7;
            let (a1, b1, _) = cam.distortion(u + h, v);
            let (a0, b0, _) = cam.distortion(u - h, v);
            let (c1, d1, _) = cam.distortion(u, v + h);
            let (c0, d0, _) = cam.distortion(u, v - h);
            let num = Matrix2::new((a1 - a0) / (2.0 * h), (c1 - c0) / (2.0 * h), (b1 - b0) / (2.0 * h), (d1 - d0) / (2.0 * h));
            assert!((num - j).abs().max() <= 1e-6);
        }
    }

    #[test]
    fn projection_jacobian_numeric() {
        let mut rng = Pcg64::seed_from_u64(9);
        for model in [
            CameraModelKind::SingleFocalPinhole,
            CameraModelKind::Pinhole,
            CameraModelKind::SingleFocalRadial,
            CameraModelKind::Radial,
            CameraModelKind::OpenCv,
        ] {
            for _ in 0..50 {
                let mut cam = Camera::from_focal(model, 1200.0, 2000, 1500);
                for &i in model.extra_param_slots() {
                    cam.params[i] = rng.random_range(-0.2..0.2);
                }
                if model == CameraModelKind::OpenCv {
                    cam.params[6] = rng.random_range(-0.01..0.01);
                    cam.params[7] = rng.random_range(-0.01..0.01);
                }
                let xc = Vec3::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), rng.random_range(2.0..5.0));
                let n = model.num_params();
                let mut dp = vec![0.0; 2 * n];
                let (pix, jp) = cam.cam_to_img_with_jacobian(&xc, Some(&mut dp)).unwrap();
                assert!((pix - cam.cam_to_img(&xc).unwrap()).norm() < 1e-12);
                // point
                for k in 0..3 {
                    let h = 1e-6 * xc[k].abs().max(1.0);
                    let mut a = xc;
                    a[k] += h;
                    let mut b = xc;
                    b[k] -= h;
                    let num = (cam.cam_to_img(&a).unwrap() - cam.cam_to_img(&b).unwrap()) / (2.0 * h);
                    for r in 0..2 {
                        let an = jp[(r, k)];
                        assert!((num[r] - an).abs() <= 1e-6 * an.abs().max(1.0), "{model:?} pt {r},{k}: {} vs {}", num[r], an);
                    }
                }
                // params
                for k in 0..n {
                    let h = 1e-6 * cam.params[k].abs().max(1.0);
                    let mut a = cam.clone();
                    a.params[k] += h;
                    let mut b = cam.clone();
                    b.params[k] -= h;
                    let num = (a.cam_to_img(&xc).unwrap() - b.cam_to_img(&xc).unwrap()) / (2.0 * h);
                    for r in 0..2 {
                        let an = dp[r * n + k];
                        assert!((num[r] - an).abs() <= 1e-6 * an.abs().max(1.0), "{model:?} param {r},{k}: {} vs {}", num[r], an);
                    }
                }
            }
        }
        let cam = Camera::from_focal(CameraModelKind::OpenCv, 1000.0, 100, 100);
        assert!(cam.cam_to_img_with_jacobian(&Vec3::new(0.0, 0.0, -1.0), None).is_none());
    }

    #[test]
    fn model_tables_and_rescale() {
        let cam = Camera::from_focal(CameraModelKind::OpenCv, 4800.0, 4000, 3000);
        assert_eq!(cam.params, vec![4800.0, 4800.0, 2000.0, 1500.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(CameraModelKind::from_name("OPENCV").unwrap(), CameraModelKind::OpenCv);
        assert!(CameraModelKind::from_id(6).is_err());
        assert!(Camera::new(1, CameraModelKind::OpenCv, 1, 1, vec![0.0; 7]).is_err());
        let mut c = cam.clone();
        c.rescale_to(960, 720);
        assert!((c.params[0] - 4800.0 * 0.24).abs() < 1e-9);
        assert!((c.params[3] - 1500.0 * 0.24).abs() < 1e-9);
        assert!((cam.img_to_cam_threshold(4.0) - 4.0 / 4800.0).abs() < 1e-15);
        assert!(!cam.has_implausible_params(0.1, 10.0, 1.0));
        let mut b = cam.clone();
        b.params[4] = 1.5;
        assert!(b.has_implausible_params(0.1, 10.0, 1.0));
    }
}
