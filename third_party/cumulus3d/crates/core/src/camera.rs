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

//! Camera models.
//!
//! Coordinate convention: camera coordinates (X, Y, Z) -> normalized plane (u, v) = (X/Z, Y/Z) -> distortion -> pixel.
//! The pixel origin is the top-left corner of the image; the top-left pixel center = (0.5, 0.5).

use crate::error::{Error, Result};
use crate::geometry::{Mat3, Vec2, Vec3};
use crate::ids::{CameraId, INVALID_CAMERA_ID};
use nalgebra::{Matrix2, Matrix2x3};

/// Supported camera models. The numeric id is the model number in the model file format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CameraModelKind {
    /// Single-focal pinhole (f, cx, cy).
    SingleFocalPinhole = 0,
    /// Pinhole (fx, fy, cx, cy).
    Pinhole = 1,
    /// Single focal + 1 radial distortion term (f, cx, cy, k).
    SingleFocalRadial = 2,
    /// Single focal + 2 radial distortion terms (f, cx, cy, k1, k2).
    Radial = 3,
    /// 2 radial terms + 2 tangential terms (fx, fy, cx, cy, k1, k2, p1, p2).
    OpenCv = 4,
}

/// Maximum number of Newton iterations for undistortion.
pub const UNDISTORT_MAX_ITERS: usize = 100;
/// Convergence test: ‖step‖² < 1e−10.
pub const UNDISTORT_STEP_SQ_TOL: f64 = 1e-10;

/// File-format names of the supported models (in id order).
const MODEL_NAMES: [&str; 5] = ["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"];

impl CameraModelKind {
    /// Model number -> kind.
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
    /// Model number.
    pub fn id(self) -> i32 {
        self as i32
    }
    /// File-format model name (e.g. `PINHOLE`).
    pub fn name(self) -> &'static str {
        MODEL_NAMES[self as usize]
    }
    /// File-format model name -> kind.
    pub fn from_name(name: &str) -> Result<Self> {
        match MODEL_NAMES.iter().position(|n| *n == name) {
            Some(i) => Self::from_id(i as i32),
            None => Err(Error::Format(format!("알 수 없는 카메라 모델 이름 {name}"))),
        }
    }
    /// Number of parameters.
    pub fn num_params(self) -> usize {
        match self {
            Self::SingleFocalPinhole => 3,
            Self::Pinhole => 4,
            Self::SingleFocalRadial => 4,
            Self::Radial => 5,
            Self::OpenCv => 8,
        }
    }
    /// Parameter names (comma-separated).
    pub fn params_info(self) -> &'static str {
        match self {
            Self::SingleFocalPinhole => "f, cx, cy",
            Self::Pinhole => "fx, fy, cx, cy",
            Self::SingleFocalRadial => "f, cx, cy, k",
            Self::Radial => "f, cx, cy, k1, k2",
            Self::OpenCv => "fx, fy, cx, cy, k1, k2, p1, p2",
        }
    }
    /// Position of the focal parameters within `params`.
    pub fn focal_slots(self) -> &'static [usize] {
        match self {
            Self::SingleFocalPinhole | Self::SingleFocalRadial | Self::Radial => &[0],
            Self::Pinhole | Self::OpenCv => &[0, 1],
        }
    }
    /// Position of the principal point parameters within `params`.
    pub fn pp_slots(self) -> &'static [usize] {
        match self {
            Self::SingleFocalPinhole | Self::SingleFocalRadial | Self::Radial => &[1, 2],
            Self::Pinhole | Self::OpenCv => &[2, 3],
        }
    }
    /// Position of the distortion (extra) parameters within `params`.
    pub fn extra_param_slots(self) -> &'static [usize] {
        match self {
            Self::SingleFocalPinhole | Self::Pinhole => &[],
            Self::SingleFocalRadial => &[3],
            Self::Radial => &[3, 4],
            Self::OpenCv => &[4, 5, 6, 7],
        }
    }
}

/// Camera (intrinsic parameters). `focal_from_prior` is memory-only (not stored in the model file).
#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    /// Camera id.
    pub camera_id: CameraId,
    /// Camera model.
    pub model: CameraModelKind,
    /// Image width (pixels).
    pub width: u64,
    /// Image height (pixels).
    pub height: u64,
    /// Model parameters (order per `CameraModelKind::params_info`).
    pub params: Vec<f64>,
    /// Whether the focal length came from prior information such as EXIF (for choosing the calibration path).
    pub focal_from_prior: bool,
}

impl Camera {
    /// Creates after validating the parameter count.
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

    /// Initializes with focal length f: focal = f, principal point = (W/2, H/2), distortion 0.
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

    /// Whether the parameter count matches the model.
    pub fn verify_params(&self) -> bool {
        self.params.len() == self.model.num_params()
    }

    /// Mean focal length.
    pub fn mean_focal(&self) -> f64 {
        let idx = self.model.focal_slots();
        idx.iter().map(|&i| self.params[i]).sum::<f64>() / idx.len() as f64
    }
    /// x focal length.
    pub fn focal_length_x(&self) -> f64 {
        self.params[self.model.focal_slots()[0]]
    }
    /// y focal length (same as x for single-focal models).
    pub fn focal_length_y(&self) -> f64 {
        let idx = self.model.focal_slots();
        self.params[idx[idx.len() - 1]]
    }
    /// Sets all focal parameters to f.
    pub fn set_focal_length(&mut self, f: f64) {
        for &i in self.model.focal_slots() {
            self.params[i] = f;
        }
    }
    /// Principal point x.
    pub fn principal_point_x(&self) -> f64 {
        self.params[self.model.pp_slots()[0]]
    }
    /// Principal point y.
    pub fn principal_point_y(&self) -> f64 {
        self.params[self.model.pp_slots()[1]]
    }
    /// Sets the principal point.
    pub fn set_principal_point(&mut self, cx: f64, cy: f64) {
        let pp = self.model.pp_slots();
        self.params[pp[0]] = cx;
        self.params[pp[1]] = cy;
    }
    /// Copy of the distortion (extra) parameters.
    pub fn extra_params(&self) -> Vec<f64> {
        self.model.extra_param_slots().iter().map(|&i| self.params[i]).collect()
    }
    /// No-distortion test: all extra parameters are 0.
    pub fn is_undistorted(&self) -> bool {
        self.model.extra_param_slots().iter().all(|&i| self.params[i] == 0.0)
    }
    /// Calibration matrix K (distortion ignored).
    pub fn calibration_matrix(&self) -> Mat3 {
        Mat3::new(self.focal_length_x(), 0.0, self.principal_point_x(), 0.0, self.focal_length_y(), self.principal_point_y(), 0.0, 0.0, 1.0)
    }
    /// Abnormal-parameter test.
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

    /// Uniform resize: multiplies focal and principal point by s, size is rounded. Distortion coefficients unchanged.
    pub fn rescale(&mut self, scale: f64) {
        let nw = (self.width as f64 * scale).round().max(1.0) as u64;
        let nh = (self.height as f64 * scale).round().max(1.0) as u64;
        self.scale_params(scale, scale);
        self.width = nw;
        self.height = nh;
    }

    /// Resize to a new size: multiplies focal and principal point by per-axis sx = W'/W, sy = H'/H.
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
            // Design decision: for single-focal models, the anisotropic scale uses the mean of the two axes.
            self.params[f[0]] *= 0.5 * (sx + sy);
        } else {
            self.params[f[0]] *= sx;
            self.params[f[1]] *= sy;
        }
        let pp = self.model.pp_slots();
        self.params[pp[0]] *= sx;
        self.params[pp[1]] *= sy;
    }

    /// Converts a pixel error threshold to a normalized-plane threshold: τ / mean focal length.
    pub fn img_to_cam_threshold(&self, threshold_px: f64) -> f64 {
        threshold_px / self.mean_focal()
    }

    /// Normalized-plane distortion Δ(u,v) and ∂Δ/∂(u,v).
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

    /// Normalized plane (u, v) -> pixel (with distortion).
    pub fn normalized_to_img(&self, uv: &Vec2) -> Vec2 {
        let (fx, fy, cx, cy) = self.fxfycxcy();
        let (du, dv, _) = self.distortion(uv.x, uv.y);
        Vec2::new(fx * (uv.x + du) + cx, fy * (uv.y + dv) + cy)
    }

    /// 3D point in camera coordinates -> pixel. None if Z < f64 ε.
    pub fn cam_to_img(&self, xc: &Vec3) -> Option<Vec2> {
        if xc.z < f64::EPSILON {
            return None;
        }
        Some(self.normalized_to_img(&Vec2::new(xc.x / xc.z, xc.y / xc.z)))
    }

    /// Pixel -> normalized plane (undistortion). None if Newton does not converge within 100 iterations.
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

    /// Pixel -> unit ray (u, v, 1)/‖·‖.
    pub fn img_to_ray(&self, xy: &Vec2) -> Option<Vec3> {
        self.img_to_normalized(xy).map(|uv| Vec3::new(uv.x, uv.y, 1.0).normalize())
    }

    /// Projection and analytic Jacobian.
    ///
    /// Returns: (pixel, ∂pixel/∂camera-coordinates 2×3). If `d_params` is given, fills ∂pixel/∂parameters as row-major 2×num_params
    /// (first row = x, second row = y, columns = parameter order).
    /// None if Z < ε (BA then sets the residual and Jacobian to 0).
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
                // Samples where the distortion map is not locally invertible (Jacobian determinant ≤ 0.2) are excluded
                // since the solution is not unique.
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
        // Large radius: with k1 = −5, a p0 beyond the maximum of F(p) (~0.17 at r≈0.26) has no solution.
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
