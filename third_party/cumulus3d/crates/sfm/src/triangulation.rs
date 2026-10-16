/*
 * triangulation.rs
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

//! Triangulation formulas: two-view DLT, multi-view, midpoint method, angle and depth checks, RANSAC triangulation.

use cumulus3d_core::geometry::triangulation_angle;
use cumulus3d_core::ransac::{n_choose_k, ransac_with_sampler, Estimator, ExhaustiveSampler, RansacParams};
use cumulus3d_core::{Camera, Mat3x4, Rigid3, Vec2, Vec3};
use nalgebra::{Matrix4, SymmetricEigen, SVD};

/// Two-view DLT. `x1`, `x2` are normalized-plane coordinates, `p1`, `p2` are [R|t].
pub fn triangulate_dlt(p1: &Mat3x4, p2: &Mat3x4, x1: &Vec2, x2: &Vec2) -> Option<Vec3> {
    let mut a = Matrix4::<f64>::zeros();
    a.set_row(0, &(x1.x * p1.row(2) - p1.row(0)));
    a.set_row(1, &(x1.y * p1.row(2) - p1.row(1)));
    a.set_row(2, &(x2.x * p2.row(2) - p2.row(0)));
    a.set_row(3, &(x2.y * p2.row(2) - p2.row(1)));
    let svd = SVD::new(a, false, true);
    let vt = svd.v_t?;
    // Singular values in descending order, so the last row has the smallest singular value.
    let h = vt.row(3);
    if h[3] == 0.0 {
        return None;
    }
    let x = Vec3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3]);
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Multi-view triangulation: minimizes the algebraic error of the component perpendicular to the rays.
/// `rays` are unit rays in camera coordinates.
pub fn triangulate_multi_view(poses: &[Mat3x4], rays: &[Vec3]) -> Option<Vec3> {
    if poses.len() < 2 || poses.len() != rays.len() {
        return None;
    }
    let mut m = Matrix4::<f64>::zeros();
    for (p, b) in poses.iter().zip(rays) {
        let q: Mat3x4 = p - b * (b.transpose() * p);
        m += q.transpose() * q;
    }
    let eig = SymmetricEigen::new(m);
    let mut k = 0;
    for i in 1..4 {
        if eig.eigenvalues[i] < eig.eigenvalues[k] {
            k = i;
        }
    }
    let h = eig.eigenvectors.column(k);
    if h[3] == 0.0 {
        return None;
    }
    let x = Vec3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3]);
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Midpoint method. The result is in camera 1 coordinates. None if behind a camera.
pub fn triangulate_midpoint(cam1_to_cam2: &Rigid3, ray1: &Vec3, ray2: &Vec3) -> Option<Vec3> {
    let rt = cam1_to_cam2.rotation_matrix().transpose();
    let r2 = rt * ray2;
    let c2 = -(rt * cam1_to_cam2.translation);
    let a = cumulus3d_core::Mat3::from_columns(&[*ray1, -r2, -c2]);
    let (_, _, vm) = crate::math::svd3(&a)?;
    let v = vm.column(2);
    if v[2] == 0.0 {
        return None;
    }
    let (l1, l2) = (v[0] / v[2], v[1] / v[2]);
    if !(l1 > f64::EPSILON && l2 > f64::EPSILON) {
        return None;
    }
    Some(0.5 * (l1 * ray1 + c2 + l2 * r2))
}

/// Positive depth check: third row of P · (X, 1) ≥ ε.
pub fn has_positive_depth(p: &Mat3x4, x: &Vec3) -> bool {
    p[(2, 0)] * x.x + p[(2, 1)] * x.y + p[(2, 2)] * x.z + p[(2, 3)] >= f64::EPSILON
}

/// Angular error (radians): between the observation ray and the direction to the point in camera coordinates.
pub fn angular_error(p: &Mat3x4, ray: &Vec3, x: &Vec3) -> f64 {
    let xc = p * x.push(1.0);
    let n = xc.norm();
    if n == 0.0 {
        return std::f64::consts::PI;
    }
    (ray.dot(&xc) / n).clamp(-1.0, 1.0).acos()
}

/// One observation input for RANSAC triangulation.
#[derive(Clone, Debug)]
pub struct TriObservation {
    /// world_to_cam [R|t].
    pub proj: Mat3x4,
    /// Projection center.
    pub center: Vec3,
    /// Unit ray in camera coordinates.
    pub ray: Vec3,
}

impl TriObservation {
    /// Builds an observation from a world_to_cam pose and a camera-coordinate ray.
    pub fn new(pose: &Rigid3, ray: Vec3) -> Self {
        Self { proj: pose.matrix(), center: pose.center(), ray }
    }
}

/// RANSAC triangulation options.
#[derive(Clone, Debug)]
pub struct TriangulationRansacParams {
    /// Maximum angular error (degrees).
    pub max_angle_error_deg: f64,
    /// Minimum triangulation angle (degrees).
    pub min_tri_angle_deg: f64,
    /// RANSAC confidence.
    pub confidence: f64,
    /// Minimum inlier ratio.
    pub min_inlier_ratio: f64,
    /// Maximum number of iterations.
    pub max_trials: usize,
    /// If the number of observations is at most this, exhaustively search all 2-point combinations.
    pub exhaustive_max_num_obs: usize,
}

impl Default for TriangulationRansacParams {
    fn default() -> Self {
        Self {
            max_angle_error_deg: 2.0,
            min_tri_angle_deg: 1.5,
            confidence: 0.9999,
            min_inlier_ratio: 0.02,
            max_trials: 10000,
            exhaustive_max_num_obs: 15,
        }
    }
}

/// Model validation: all view depths positive + maximum angle over view pairs ≥ minimum angle.
fn valid_point(obs: &[TriObservation], x: &Vec3, min_angle_rad: f64) -> bool {
    if !obs.iter().all(|o| has_positive_depth(&o.proj, x)) {
        return false;
    }
    for i in 0..obs.len() {
        for j in 0..i {
            if triangulation_angle(&obs[i].center, &obs[j].center, x) >= min_angle_rad {
                return true;
            }
        }
    }
    false
}

struct TriEstimator {
    min_angle_rad: f64,
    multi_view: bool,
}

impl Estimator for TriEstimator {
    type X = TriObservation;
    type Y = ();
    type Model = Vec3;
    fn min_num_samples(&self) -> usize {
        2
    }
    fn estimate(&self, x: &[TriObservation], _y: &[()], models: &mut Vec<Vec3>) {
        let p = if self.multi_view || x.len() != 2 {
            let poses: Vec<Mat3x4> = x.iter().map(|o| o.proj).collect();
            let rays: Vec<Vec3> = x.iter().map(|o| o.ray).collect();
            triangulate_multi_view(&poses, &rays)
        } else {
            let (a, b) = (&x[0], &x[1]);
            if a.ray.z <= 0.0 || b.ray.z <= 0.0 {
                None
            } else {
                triangulate_dlt(&a.proj, &b.proj, &(a.ray.xy() / a.ray.z), &(b.ray.xy() / b.ray.z))
            }
        };
        if let Some(p) = p {
            if valid_point(x, &p, self.min_angle_rad) {
                models.push(p);
            }
        }
    }
    fn residuals(&self, x: &[TriObservation], _y: &[()], model: &Vec3, residuals: &mut Vec<f64>) {
        residuals.clear();
        residuals.extend(x.iter().map(|o| angular_error(&o.proj, &o.ray, model).powi(2)));
    }
}

/// RANSAC multi-view triangulation. On success (point, inlier mask).
pub fn estimate_triangulation(obs: &[TriObservation], opts: &TriangulationRansacParams) -> Option<(Vec3, Vec<bool>)> {
    let n = obs.len();
    if n < 2 {
        return None;
    }
    let min_angle_rad = opts.min_tri_angle_deg.to_radians();
    let est = TriEstimator { min_angle_rad, multi_view: false };
    let local = TriEstimator { min_angle_rad, multi_view: true };
    let num_pairs = n_choose_k(n, 2);
    let ropts = RansacParams {
        max_error: opts.max_angle_error_deg.to_radians(),
        min_inlier_ratio: opts.min_inlier_ratio,
        confidence: opts.confidence,
        dyn_trials_factor: 3.0,
        min_trials: if n <= opts.exhaustive_max_num_obs { num_pairs } else { 0 },
        max_trials: opts.max_trials.min(num_pairs),
        random_seed: Some(0),
    };
    let y = vec![(); n];
    let rep = ransac_with_sampler(&est, Some(&local), ExhaustiveSampler::new(2), &ropts, obs, &y);
    if !rep.success || rep.support.num_inliers < 2 {
        return None;
    }
    rep.model.map(|m| (m, rep.inlier_mask))
}

/// RANSAC 삼각측량 문제 하나.
#[derive(Clone, Debug)]
pub struct TriProblem<'a> {
    /// 관측.
    pub obs: &'a [TriObservation],
    /// 옵션.
    pub params: TriangulationRansacParams,
}

/// 관측 검사 하나: 등록 영상의 관측(픽셀)과 3D 점.
#[derive(Clone, Copy, Debug)]
pub struct ObsQuery {
    /// `cameras` 안 번호.
    pub camera: u32,
    /// `poses` 안 번호(world_to_cam).
    pub pose: u32,
    /// 관측 픽셀.
    pub xy: Vec2,
    /// 3D 점(세계 좌표).
    pub xyz: Vec3,
}

/// 관측 검사 결과.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObsCheck {
    /// 픽셀 재투영 제곱 오차(`Reconstruction::squared_reprojection_error` 와 같음, 투영 실패면 f64::MAX).
    pub sq_reproj_error: f64,
    /// 카메라 깊이 = P 의 셋째 행 · (X, 1) ([`has_positive_depth`] 는 이 값 ≥ ε).
    pub depth: f64,
}

/// 삼각측량 일괄 계산 백엔드. CPU 기본 구현은 [`CpuTriangulation`].
pub trait TriangulationBackend: Send + Sync {
    /// RANSAC 삼각측량 여러 문제. 결과는 문제마다 [`estimate_triangulation`] 과 같은 규칙, 순서는 입력 순서.
    fn estimate_batch(&self, problems: &[TriProblem<'_>]) -> Vec<Option<(Vec3, Vec<bool>)>>;
    /// 관측 여러 개의 재투영 오차와 깊이.
    fn check_observations(&self, cameras: &[Camera], poses: &[Rigid3], queries: &[ObsQuery]) -> Vec<ObsCheck>;
}

/// CPU 기본 구현.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuTriangulation;

/// 관측 검사 하나(CPU).
pub fn check_observation(camera: &Camera, pose: &Rigid3, xy: &Vec2, xyz: &Vec3) -> ObsCheck {
    let p = pose.matrix();
    let depth = p[(2, 0)] * xyz.x + p[(2, 1)] * xyz.y + p[(2, 2)] * xyz.z + p[(2, 3)];
    ObsCheck { sq_reproj_error: cumulus3d_core::Reconstruction::squared_reprojection_error(pose, camera, xy, xyz), depth }
}

impl TriangulationBackend for CpuTriangulation {
    fn estimate_batch(&self, problems: &[TriProblem<'_>]) -> Vec<Option<(Vec3, Vec<bool>)>> {
        use rayon::prelude::*;
        problems.par_iter().with_min_len(16).map(|p| estimate_triangulation(p.obs, &p.params)).collect()
    }
    fn check_observations(&self, cameras: &[Camera], poses: &[Rigid3], queries: &[ObsQuery]) -> Vec<ObsCheck> {
        use rayon::prelude::*;
        queries
            .par_iter()
            .with_min_len(256)
            .map(|q| check_observation(&cameras[q.camera as usize], &poses[q.pose as usize], &q.xy, &q.xyz))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::so3_exp;
    use rand::{RngExt, SeedableRng};

    fn pose_looking_at(c: Vec3, target: Vec3) -> Rigid3 {
        let z = (target - c).normalize();
        let x = Vec3::new(1.0, 0.0, 0.0).cross(&z).normalize();
        let x = if x.norm() > 0.5 { x } else { Vec3::y().cross(&z).normalize() };
        let y = z.cross(&x);
        let r = cumulus3d_core::Mat3::from_rows(&[x.transpose(), y.transpose(), z.transpose()]);
        Rigid3::from_rotation_matrix(&r, -(r * c))
    }

    #[test]
    fn dlt_noise_free() {
        let mut rng = rand_pcg::Pcg64::seed_from_u64(1);
        for _ in 0..100 {
            let x = Vec3::new(rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0), rng.random_range(0.0..3.0));
            let p1 = pose_looking_at(Vec3::new(-2.0, 0.0, 20.0), Vec3::zeros());
            let p2 = pose_looking_at(Vec3::new(3.0, 1.0, 22.0), Vec3::zeros());
            let (a, b) = (p1 * x, p2 * x);
            let y = triangulate_dlt(&p1.matrix(), &p2.matrix(), &(a.xy() / a.z), &(b.xy() / b.z)).unwrap();
            assert!((y - x).norm() < 1e-9 * 10.0, "{}", (y - x).norm());
            let m = triangulate_multi_view(&[p1.matrix(), p2.matrix()], &[a.normalize(), b.normalize()]).unwrap();
            assert!((m - x).norm() < 1e-8);
            let rel = p2 * p1.inverse();
            let mp = triangulate_midpoint(&rel, &a.normalize(), &b.normalize()).unwrap();
            assert!((mp - a).norm() < 1e-8);
        }
    }

    #[test]
    fn multi_view_noise_within_3sigma() {
        // 10 views, 0.5 px noise (f=1000): 3D error within roughly 3σ of the covariance.
        let mut rng = rand_pcg::Pcg64::seed_from_u64(2);
        let f = 1000.0;
        let x = Vec3::new(0.5, -0.3, 1.0);
        let mut poses = Vec::new();
        let mut rays = Vec::new();
        let mut info = nalgebra::Matrix3::<f64>::zeros();
        for k in 0..10 {
            let c = Vec3::new(-10.0 + 2.0 * k as f64, rng.random_range(-2.0..2.0), 30.0);
            let p = pose_looking_at(c, Vec3::zeros());
            let xc = p * x;
            let uv = xc.xy() / xc.z + Vec2::new(crate::math::gaussian(&mut rng), crate::math::gaussian(&mut rng)) * (0.5 / f);
            rays.push(uv.push(1.0).normalize());
            poses.push(p.matrix());
            // Information matrix approximation: two directions perpendicular to the ray, variance (0.5/f · depth)².
            let d = xc.norm();
            let sig = 0.5 / f * d;
            let b = (p.rotation_matrix().transpose() * xc).normalize();
            info += (nalgebra::Matrix3::identity() - b * b.transpose()) / (sig * sig);
        }
        let y = triangulate_multi_view(&poses, &rays).unwrap();
        let cov = info.try_inverse().unwrap();
        let e = y - x;
        let mahal = (e.transpose() * cov.try_inverse().unwrap() * e)[0];
        // 3σ-equivalent value of χ² with 3 degrees of freedom (≈ 14.2)
        assert!(mahal < 14.2, "mahalanobis² = {mahal}");
    }

    #[test]
    fn positive_depth_rejects_behind() {
        let p = Rigid3::identity().matrix();
        assert!(has_positive_depth(&p, &Vec3::new(0.0, 0.0, 1.0)));
        assert!(!has_positive_depth(&p, &Vec3::new(0.0, 0.0, -1.0)));
        // RANSAC model validation: points behind both cameras are rejected.
        let p1 = Rigid3::identity();
        let p2 = Rigid3::new(cumulus3d_core::Quat::IDENTITY, Vec3::new(-1.0, 0.0, 0.0));
        let x = Vec3::new(0.3, 0.1, -5.0);
        let o = [TriObservation::new(&p1, -(p1 * x).normalize()), TriObservation::new(&p2, -(p2 * x).normalize())];
        // Flipping the rays makes the DLT solution land behind the cameras.
        assert!(estimate_triangulation(&o, &TriangulationRansacParams::default()).is_none());
    }

    #[test]
    fn ransac_angle_thresholds() {
        // Angular error boundary: 1.99° is an inlier (residual ≤ (2°)²), 2.01° is an outlier.
        let x = Vec3::new(0.0, 0.0, 10.0);
        let p = Rigid3::identity().matrix();
        let max2 = 2f64.to_radians().powi(2);
        for (deg, inl) in [(1.99f64, true), (2.01, false)] {
            let r = so3_exp(&Vec3::new(deg.to_radians(), 0.0, 0.0)) * x.normalize();
            assert_eq!(angular_error(&p, &r, &x).powi(2) <= max2, inl);
        }
        // RANSAC: 5 correct views + a view off by 1.9° (inlier) + a view off by 10° (outlier).
        let mut o = Vec::new();
        for k in 0..5 {
            let pk = Rigid3::new(cumulus3d_core::Quat::IDENTITY, Vec3::new(-4.0 + 2.0 * k as f64, 0.5 * k as f64, 0.0));
            o.push(TriObservation::new(&pk, (pk * x).normalize()));
        }
        for deg in [1.9f64, 10.0] {
            let pk = Rigid3::new(cumulus3d_core::Quat::IDENTITY, Vec3::new(0.0, -3.0, 0.0));
            let r = so3_exp(&Vec3::new(0.0, deg.to_radians(), 0.0)) * (pk * x).normalize();
            o.push(TriObservation::new(&pk, r));
        }
        let (y, mask) = estimate_triangulation(&o, &TriangulationRansacParams::default()).unwrap();
        assert_eq!(mask, vec![true, true, true, true, true, true, false]);
        assert!((y - x).norm() < 0.2);
    }

    #[test]
    fn min_tri_angle_boundary() {
        // Two symmetric centers, parallax angle 1.49° so not created, 1.51° so created.
        for (deg, ok) in [(1.49f64, false), (1.51, true)] {
            let half = (deg / 2.0).to_radians();
            let z = 1.0 / half.tan();
            let x = Vec3::new(0.0, 0.0, z);
            let p1 = Rigid3::new(cumulus3d_core::Quat::IDENTITY, Vec3::new(1.0, 0.0, 0.0));
            let p2 = Rigid3::new(cumulus3d_core::Quat::IDENTITY, Vec3::new(-1.0, 0.0, 0.0));
            let o = vec![TriObservation::new(&p1, (p1 * x).normalize()), TriObservation::new(&p2, (p2 * x).normalize())];
            assert_eq!(estimate_triangulation(&o, &TriangulationRansacParams::default()).is_some(), ok);
        }
    }

    #[test]
    fn tri_angle_symmetric_analytic() {
        let c1 = Vec3::new(-1.0, 0.0, 0.0);
        let c2 = Vec3::new(1.0, 0.0, 0.0);
        for z in [0.1, 0.5, 1.0, 3.0, 100.0] {
            let x = Vec3::new(0.0, 0.0, z);
            let theta = 2.0 * (1.0f64 / z).atan();
            let expect = theta.min(std::f64::consts::PI - theta);
            assert!((triangulation_angle(&c1, &c2, &x) - expect).abs() < 1e-12);
        }
    }
}
