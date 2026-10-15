/*
 * absolute_pose.rs
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

//! 절대 자세: P3P 최소해법, EPnP 비최소해법, LO-RANSAC.

use crate::math::{adjugate3, kabsch, solve_cubic_real, solve_quadratic_real};
use cumulus3d_core::ransac::{lo_ransac, Estimator, RansacParams};
use cumulus3d_core::{Camera, Mat3, Rigid3, Vec2, Vec3};
use nalgebra::{Matrix3, SMatrix, SVector, SymmetricEigen, SVD};

/// P3P. 단위 광선 3개와 월드점 3개 → 최대 4개 world_to_cam.
///
/// 거리 보존 이차식 3개에서 두 퇴화 원뿔 조합 D1 + γ D2 의 행렬식 3차식을 풀고,
/// 퇴화 원뿔을 직선 쌍으로 분해해 깊이 λ 를 구한다(Lambda Twist 계열). 이후 가우스-뉴턴 보정.
pub fn p3p(rays: &[Vec3; 3], points: &[Vec3; 3]) -> Vec<Rigid3> {
    let y = rays;
    let x = points;
    if y.iter().any(|r| r.norm_squared() < 0.5) {
        return Vec::new();
    }
    let b12 = y[0].dot(&y[1]);
    let b13 = y[0].dot(&y[2]);
    let b23 = y[1].dot(&y[2]);
    let a12 = (x[0] - x[1]).norm_squared();
    let a13 = (x[0] - x[2]).norm_squared();
    let a23 = (x[1] - x[2]).norm_squared();
    if a12 < 1e-24 || a13 < 1e-24 || a23 < 1e-24 {
        return Vec::new();
    }
    let m12 = Matrix3::new(1.0, -b12, 0.0, -b12, 1.0, 0.0, 0.0, 0.0, 0.0);
    let m13 = Matrix3::new(1.0, 0.0, -b13, 0.0, 0.0, 0.0, -b13, 0.0, 1.0);
    let m23 = Matrix3::new(0.0, 0.0, 0.0, 0.0, 1.0, -b23, 0.0, -b23, 1.0);
    let d1 = m12 * a13 - m13 * a12;
    let d2 = m23 * a13 - m13 * a23;

    // det(D1 + γ D2) = c3 γ³ + c2 γ² + c1 γ + c0
    let c0 = d1.determinant();
    let c3 = d2.determinant();
    let fp = (d1 + d2).determinant();
    let fm = (d1 - d2).determinant();
    let c2 = 0.5 * (fp + fm) - c0;
    let c1 = 0.5 * (fp - fm) - c3;
    let mut gammas = solve_cubic_real(c3, c2, c1, c0);
    // 큰 근부터(수치적으로 D1 의 기여가 작아지는 쪽) 시도.
    gammas.sort_by(|a, b| b.abs().total_cmp(&a.abs()));

    let mut sols: Vec<[f64; 3]> = Vec::new();
    let eqs = |l: &[f64; 3]| -> [f64; 3] {
        [
            l[0] * l[0] + l[1] * l[1] - 2.0 * b12 * l[0] * l[1] - a12,
            l[0] * l[0] + l[2] * l[2] - 2.0 * b13 * l[0] * l[2] - a13,
            l[1] * l[1] + l[2] * l[2] - 2.0 * b23 * l[1] * l[2] - a23,
        ]
    };
    for &g in &gammas {
        let d0 = d1 + d2 * g;
        let dq = if g.abs() <= 1.0 { d2 } else { d1 };
        let lines = split_degenerate_conic(&d0);
        if lines.is_empty() {
            continue;
        }
        let before = sols.len();
        for l in lines {
            // λ_k 를 나머지 둘로 표현: 가장 큰 계수 성분 k.
            let mut k = 0;
            for i in 1..3 {
                if l[i].abs() > l[k].abs() {
                    k = i;
                }
            }
            if l[k] == 0.0 {
                continue;
            }
            let (ia, ib) = match k {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            let mut u0 = Vec3::zeros();
            u0[ia] = 1.0;
            u0[k] = -l[ia] / l[k];
            let mut u1 = Vec3::zeros();
            u1[ib] = 1.0;
            u1[k] = -l[ib] / l[k];
            let qa = (u1.transpose() * dq * u1)[0];
            let qb = 2.0 * (u0.transpose() * dq * u1)[0];
            let qc = (u0.transpose() * dq * u0)[0];
            for tau in solve_quadratic_real(qa, qb, qc) {
                let u = u0 + u1 * tau;
                let den = (u.transpose() * m12 * u)[0];
                if den <= 0.0 {
                    continue;
                }
                let mut s = (a12 / den).sqrt();
                if u.sum() < 0.0 {
                    s = -s;
                }
                let lam = u * s;
                if lam.iter().any(|v| *v <= 0.0) {
                    continue;
                }
                let mut lam = [lam[0], lam[1], lam[2]];
                // 가우스-뉴턴 보정.
                for _ in 0..5 {
                    let r = eqs(&lam);
                    let j = Matrix3::new(
                        2.0 * lam[0] - 2.0 * b12 * lam[1],
                        2.0 * lam[1] - 2.0 * b12 * lam[0],
                        0.0,
                        2.0 * lam[0] - 2.0 * b13 * lam[2],
                        0.0,
                        2.0 * lam[2] - 2.0 * b13 * lam[0],
                        0.0,
                        2.0 * lam[1] - 2.0 * b23 * lam[2],
                        2.0 * lam[2] - 2.0 * b23 * lam[1],
                    );
                    let Some(ji) = j.try_inverse() else { break };
                    let d = ji * Vec3::new(r[0], r[1], r[2]);
                    let nl = [lam[0] - d[0], lam[1] - d[1], lam[2] - d[2]];
                    if nl.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                        break;
                    }
                    lam = nl;
                }
                // 중복 해 제거.
                if sols.iter().any(|s| (0..3).all(|i| (s[i] - lam[i]).abs() <= 1e-9 * (1.0 + lam[i].abs()))) {
                    continue;
                }
                sols.push(lam);
            }
        }
        if sols.len() > before {
            break;
        }
    }

    let mut out = Vec::with_capacity(sols.len());
    for lam in sols.into_iter().take(4) {
        let pc = [y[0] * lam[0], y[1] * lam[1], y[2] * lam[2]];
        if let Some((r, t)) = kabsch(x, &pc) {
            out.push(Rigid3::from_rotation_matrix(&r, t));
        }
    }
    out
}

/// 계수 2 대칭 3×3 원뿔을 두 직선으로 분해(실수 직선이 아니면 빈 목록).
fn split_degenerate_conic(c: &Matrix3<f64>) -> Vec<Vec3> {
    let b = adjugate3(c);
    let mut i = 0;
    for k in 1..3 {
        if b[(k, k)].abs() > b[(i, i)].abs() {
            i = k;
        }
    }
    let scale = c.abs().max().powi(2).max(1e-300);
    let p = if b[(i, i)] < 0.0 {
        let beta = (-b[(i, i)]).sqrt();
        b.column(i) / beta
    } else if b[(i, i)].abs() <= 1e-12 * scale {
        // 이중 직선(교점 없음): 원뿔이 계수 1 → 그대로 분해.
        Vec3::zeros()
    } else {
        return Vec::new();
    };
    let a = c + cumulus3d_core::geometry::skew(&p);
    let (mut r, mut cc) = (0, 0);
    let mut best = 0.0;
    for rr in 0..3 {
        for c2 in 0..3 {
            if a[(rr, c2)].abs() > best {
                best = a[(rr, c2)].abs();
                r = rr;
                cc = c2;
            }
        }
    }
    if best == 0.0 {
        return Vec::new();
    }
    let g: Vec3 = a.row(r).transpose();
    let h: Vec3 = a.column(cc).into_owned();
    vec![g, h]
}

/// EPnP. 4점 이상. `camera` 가 주어지면 세 근사 중 픽셀 오차 합으로 선택,
/// 없으면 각도 오차 합.
pub fn epnp(rays: &[Vec3], points: &[Vec3], camera: Option<&Camera>, points2d: Option<&[Vec2]>) -> Option<Rigid3> {
    let n = points.len();
    if n < 4 || rays.len() != n {
        return None;
    }
    // 제어점.
    let c0 = points.iter().sum::<Vec3>() / n as f64;
    let mut cov = Mat3::zeros();
    for p in points {
        let d = p - c0;
        cov += d * d.transpose();
    }
    let (u, sv, _) = crate::math::svd3(&cov)?;
    let mut cws = [c0; 4];
    for k in 0..3 {
        let s = (sv[k] / n as f64).sqrt();
        cws[k + 1] = c0 + u.column(k) * s;
    }
    let basis = Mat3::from_columns(&[cws[1] - c0, cws[2] - c0, cws[3] - c0]);
    let binv = basis.try_inverse()?;
    if !binv.iter().all(|v| v.is_finite()) {
        return None;
    }
    let alphas: Vec<[f64; 4]> = points
        .iter()
        .map(|p| {
            let a = binv * (p - c0);
            [1.0 - a.x - a.y - a.z, a.x, a.y, a.z]
        })
        .collect();
    // MᵀM (12×12) 누적: 점마다 [b]× Σ α_j c_j = 0.
    let mut mtm = SMatrix::<f64, 12, 12>::zeros();
    for (b, al) in rays.iter().zip(&alphas) {
        if b.norm_squared() < 0.5 {
            continue;
        }
        let s = cumulus3d_core::geometry::skew(b);
        let mut blk = SMatrix::<f64, 3, 12>::zeros();
        for (j, a) in al.iter().enumerate() {
            blk.fixed_view_mut::<3, 3>(0, 3 * j).copy_from(&(s * *a));
        }
        mtm += blk.transpose() * blk;
    }
    let eig = SymmetricEigen::new(mtm);
    let mut order: Vec<usize> = (0..12).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
    // v[0] = 최소 고유값.
    let v: Vec<SVector<f64, 12>> = order[..4].iter().map(|&k| eig.eigenvectors.column(k).into_owned()).collect();

    // L (6×10), ρ (6).
    let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let mut l = SMatrix::<f64, 6, 10>::zeros();
    let mut rho = SVector::<f64, 6>::zeros();
    for (row, &(a, b)) in pairs.iter().enumerate() {
        let dv: Vec<Vec3> = (0..4)
            .map(|k| Vec3::new(v[k][3 * a] - v[k][3 * b], v[k][3 * a + 1] - v[k][3 * b + 1], v[k][3 * a + 2] - v[k][3 * b + 2]))
            .collect();
        let vals = [
            dv[0].dot(&dv[0]),
            2.0 * dv[0].dot(&dv[1]),
            dv[1].dot(&dv[1]),
            2.0 * dv[0].dot(&dv[2]),
            2.0 * dv[1].dot(&dv[2]),
            dv[2].dot(&dv[2]),
            2.0 * dv[0].dot(&dv[3]),
            2.0 * dv[1].dot(&dv[3]),
            2.0 * dv[2].dot(&dv[3]),
            dv[3].dot(&dv[3]),
        ];
        for (c, val) in vals.iter().enumerate() {
            l[(row, c)] = *val;
        }
        rho[row] = (cws[a] - cws[b]).norm_squared();
    }
    let lsq = |cols: &[usize]| -> Option<Vec<f64>> {
        let mut a = nalgebra::DMatrix::<f64>::zeros(6, cols.len());
        for (j, &c) in cols.iter().enumerate() {
            for i in 0..6 {
                a[(i, j)] = l[(i, c)];
            }
        }
        let b = nalgebra::DVector::from_iterator(6, rho.iter().copied());
        let svd = SVD::new(a, true, true);
        let x = svd.solve(&b, 1e-12).ok()?;
        Some(x.iter().copied().collect())
    };
    let mut candidates: Vec<[f64; 4]> = Vec::new();
    // (a) β11, β12, β13, β14
    if let Some(b4) = lsq(&[0, 1, 3, 6]) {
        let beta = if b4[0] < 0.0 {
            let b1 = (-b4[0]).sqrt();
            [b1, -b4[1] / b1, -b4[2] / b1, -b4[3] / b1]
        } else {
            let b1 = b4[0].sqrt();
            [b1, b4[1] / b1, b4[2] / b1, b4[3] / b1]
        };
        candidates.push(beta);
    }
    // (b) β11, β12, β22
    if let Some(b3) = lsq(&[0, 1, 2]) {
        let (mut b1, b2) = if b3[0] < 0.0 {
            ((-b3[0]).sqrt(), if b3[2] < 0.0 { (-b3[2]).sqrt() } else { 0.0 })
        } else {
            (b3[0].sqrt(), if b3[2] > 0.0 { b3[2].sqrt() } else { 0.0 })
        };
        if b3[1] < 0.0 {
            b1 = -b1;
        }
        candidates.push([b1, b2, 0.0, 0.0]);
    }
    // (c) β11, β12, β22, β13, β23
    if let Some(b5) = lsq(&[0, 1, 2, 3, 4]) {
        let (mut b1, b2) = if b5[0] < 0.0 {
            ((-b5[0]).sqrt(), if b5[2] < 0.0 { (-b5[2]).sqrt() } else { 0.0 })
        } else {
            (b5[0].sqrt(), if b5[2] > 0.0 { b5[2].sqrt() } else { 0.0 })
        };
        if b5[1] < 0.0 {
            b1 = -b1;
        }
        let b3 = if b1 != 0.0 { b5[3] / b1 } else { 0.0 };
        candidates.push([b1, b2, b3, 0.0]);
    }

    let mut best: Option<(f64, Rigid3)> = None;
    for mut beta in candidates {
        if beta.iter().any(|b| !b.is_finite()) {
            continue;
        }
        gauss_newton_beta(&l, &rho, &mut beta);
        let Some(pose) = pose_from_betas(&v, &beta, &alphas, points) else { continue };
        let err: f64 = match (camera, points2d) {
            (Some(cam), Some(p2)) => points
                .iter()
                .zip(p2)
                .map(|(x, xy)| match cam.cam_to_img(&(pose * *x)) {
                    Some(p) => (p - xy).norm(),
                    None => 1e10,
                })
                .sum(),
            _ => points
                .iter()
                .zip(rays)
                .map(|(x, b)| {
                    let xc = pose * *x;
                    (b.dot(&xc) / xc.norm().max(1e-300)).clamp(-1.0, 1.0).acos()
                })
                .sum(),
        };
        if best.as_ref().is_none_or(|(e, _)| err < *e) {
            best = Some((err, pose));
        }
    }
    best.map(|(_, p)| p)
}

fn gauss_newton_beta(l: &SMatrix<f64, 6, 10>, rho: &SVector<f64, 6>, beta: &mut [f64; 4]) {
    for _ in 0..5 {
        let b = *beta;
        let bt = [
            b[0] * b[0],
            b[0] * b[1],
            b[1] * b[1],
            b[0] * b[2],
            b[1] * b[2],
            b[2] * b[2],
            b[0] * b[3],
            b[1] * b[3],
            b[2] * b[3],
            b[3] * b[3],
        ];
        let mut j = SMatrix::<f64, 6, 4>::zeros();
        let mut r = SVector::<f64, 6>::zeros();
        for i in 0..6 {
            let li = |c: usize| l[(i, c)];
            let lb: f64 = (0..10).map(|c| li(c) * bt[c]).sum();
            r[i] = rho[i] - lb;
            j[(i, 0)] = 2.0 * li(0) * b[0] + li(1) * b[1] + li(3) * b[2] + li(6) * b[3];
            j[(i, 1)] = li(1) * b[0] + 2.0 * li(2) * b[1] + li(4) * b[2] + li(7) * b[3];
            j[(i, 2)] = li(3) * b[0] + li(4) * b[1] + 2.0 * li(5) * b[2] + li(8) * b[3];
            j[(i, 3)] = li(6) * b[0] + li(7) * b[1] + li(8) * b[2] + 2.0 * li(9) * b[3];
        }
        // J δ = r (최소제곱, QR).
        let qr = j.qr();
        let qtr = qr.q().transpose() * r;
        let Some(d) = qr.r().solve_upper_triangular(&qtr) else { return };
        if d.iter().any(|v| !v.is_finite()) {
            return;
        }
        for k in 0..4 {
            beta[k] += d[k];
        }
    }
}

fn pose_from_betas(v: &[SVector<f64, 12>], beta: &[f64; 4], alphas: &[[f64; 4]], points: &[Vec3]) -> Option<Rigid3> {
    let mut x = SVector::<f64, 12>::zeros();
    for k in 0..4 {
        x += v[k] * beta[k];
    }
    let cc: Vec<Vec3> = (0..4).map(|j| Vec3::new(x[3 * j], x[3 * j + 1], x[3 * j + 2])).collect();
    let mut pcs: Vec<Vec3> = alphas.iter().map(|a| cc[0] * a[0] + cc[1] * a[1] + cc[2] * a[2] + cc[3] * a[3]).collect();
    if pcs[0].z < 0.0 {
        for p in pcs.iter_mut() {
            *p = -*p;
        }
    }
    let (r, t) = kabsch(points, &pcs)?;
    Some(Rigid3::from_rotation_matrix(&r, t))
}

/// 2D-3D 대응 하나(RANSAC 자료).
#[derive(Clone, Debug)]
pub struct Corr2D {
    /// 픽셀 좌표.
    pub xy: Vec2,
    /// 역투영 단위 광선(실패 시 영벡터).
    pub ray: Vec3,
}

struct P3PEstimator<'a> {
    camera: &'a Camera,
}

fn reproj_residuals(camera: &Camera, x: &[Corr2D], y: &[Vec3], pose: &Rigid3, out: &mut Vec<f64>) {
    out.clear();
    let r = pose.rotation_matrix();
    out.extend(x.iter().zip(y).map(|(c, p)| match camera.cam_to_img(&(r * p + pose.translation)) {
        Some(q) => (q - c.xy).norm_squared(),
        None => f64::MAX,
    }));
}

impl Estimator for P3PEstimator<'_> {
    type X = Corr2D;
    type Y = Vec3;
    type Model = Rigid3;
    fn min_num_samples(&self) -> usize {
        3
    }
    fn estimate(&self, x: &[Corr2D], y: &[Vec3], models: &mut Vec<Rigid3>) {
        models.extend(p3p(&[x[0].ray, x[1].ray, x[2].ray], &[y[0], y[1], y[2]]));
    }
    fn residuals(&self, x: &[Corr2D], y: &[Vec3], model: &Rigid3, out: &mut Vec<f64>) {
        reproj_residuals(self.camera, x, y, model, out);
    }
}

struct EPnPEstimator<'a> {
    camera: &'a Camera,
}

impl Estimator for EPnPEstimator<'_> {
    type X = Corr2D;
    type Y = Vec3;
    type Model = Rigid3;
    fn min_num_samples(&self) -> usize {
        4
    }
    fn estimate(&self, x: &[Corr2D], y: &[Vec3], models: &mut Vec<Rigid3>) {
        let rays: Vec<Vec3> = x.iter().map(|c| c.ray).collect();
        let xy: Vec<Vec2> = x.iter().map(|c| c.xy).collect();
        if let Some(p) = epnp(&rays, y, Some(self.camera), Some(&xy)) {
            models.push(p);
        }
    }
    fn residuals(&self, x: &[Corr2D], y: &[Vec3], model: &Rigid3, out: &mut Vec<f64>) {
        reproj_residuals(self.camera, x, y, model, out);
    }
}

/// 절대 자세 RANSAC 옵션.
#[derive(Clone, Debug)]
pub struct AbsolutePoseOptions {
    /// 최대 재투영 오차(픽셀).
    pub max_error: f64,
    /// 최소 인라이어 비율(동적 반복 횟수 계산용).
    pub min_inlier_ratio: f64,
    /// RANSAC 신뢰도(동적 반복 횟수 계산용).
    pub confidence: f64,
    /// 최소 반복 횟수.
    pub min_trials: usize,
    /// 최대 반복 횟수.
    pub max_trials: usize,
    /// 동적 반복 횟수에 곱하는 배율.
    pub dyn_trials_factor: f64,
    /// 난수 시드. `None` 이면 비결정적.
    pub random_seed: Option<u64>,
}

impl Default for AbsolutePoseOptions {
    fn default() -> Self {
        Self {
            max_error: 12.0,
            min_inlier_ratio: 0.25,
            confidence: 0.99999,
            min_trials: 100,
            max_trials: 10000,
            dyn_trials_factor: 3.0,
            random_seed: Some(0),
        }
    }
}

/// 절대 자세 추정 결과.
#[derive(Clone, Debug)]
pub struct AbsolutePoseResult {
    /// 추정한 세계 → 카메라 자세.
    pub world_to_cam: Rigid3,
    /// 입력 대응별 인라이어 여부.
    pub inlier_mask: Vec<bool>,
    /// 인라이어 수.
    pub num_inliers: usize,
    /// 수행한 RANSAC 반복 횟수.
    pub num_trials: usize,
}

/// P3P + EPnP LO-RANSAC (초점 고정 경로). 인라이어 < 3 이면 None.
pub fn solve_abs_pose(camera: &Camera, points2d: &[Vec2], points3d: &[Vec3], opts: &AbsolutePoseOptions) -> Option<AbsolutePoseResult> {
    if points2d.len() != points3d.len() || points2d.len() < 3 {
        return None;
    }
    let corrs: Vec<Corr2D> = points2d.iter().map(|xy| Corr2D { xy: *xy, ray: camera.img_to_ray(xy).unwrap_or_else(Vec3::zeros) }).collect();
    let ropts = RansacParams {
        max_error: opts.max_error,
        min_inlier_ratio: opts.min_inlier_ratio,
        confidence: opts.confidence,
        dyn_trials_factor: opts.dyn_trials_factor,
        min_trials: opts.min_trials,
        max_trials: opts.max_trials,
        random_seed: opts.random_seed,
    };
    let rep = lo_ransac(&P3PEstimator { camera }, &EPnPEstimator { camera }, &ropts, &corrs, points3d);
    if !rep.success {
        return None;
    }
    let num_inliers = rep.inlier_mask.iter().filter(|b| **b).count();
    if num_inliers < 3 {
        return None;
    }
    Some(AbsolutePoseResult { world_to_cam: rep.model?, inlier_mask: rep.inlier_mask, num_inliers, num_trials: rep.num_trials })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::so3_exp;
    use cumulus3d_core::CameraModelKind;
    use rand::{RngExt, SeedableRng};

    fn random_pose(rng: &mut rand_pcg::Pcg64) -> Rigid3 {
        let w = Vec3::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));
        let r = so3_exp(&(w * 1.5));
        let t = Vec3::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), rng.random_range(4.0..8.0));
        Rigid3::from_rotation_matrix(&r, t)
    }

    fn random_points(rng: &mut rand_pcg::Pcg64, pose: &Rigid3, n: usize) -> Vec<Vec3> {
        // 카메라 앞 점 → 월드로.
        let inv = pose.inverse();
        (0..n)
            .map(|_| {
                let xc = Vec3::new(rng.random_range(-2.0..2.0), rng.random_range(-1.5..1.5), rng.random_range(3.0..10.0));
                inv * xc
            })
            .collect()
    }

    fn pose_err(a: &Rigid3, b: &Rigid3) -> (f64, f64) {
        let r = crate::math::rotation_angle_between(&a.rotation_matrix(), &b.rotation_matrix());
        let t = (a.translation - b.translation).norm() / b.translation.norm();
        (r, t)
    }

    #[test]
    fn p3p_noise_free() {
        let mut rng = rand_pcg::Pcg64::seed_from_u64(7);
        let mut worst = (0.0f64, 0.0f64);
        for _ in 0..500 {
            let pose = random_pose(&mut rng);
            let pts = random_points(&mut rng, &pose, 3);
            let rays = [0, 1, 2].map(|i| (pose * pts[i]).normalize());
            let sols = p3p(&rays, &[pts[0], pts[1], pts[2]]);
            assert!(!sols.is_empty() && sols.len() <= 4, "해 {}", sols.len());
            let best = sols.iter().map(|s| pose_err(s, &pose)).min_by(|a, b| (a.0 + a.1).total_cmp(&(b.0 + b.1))).unwrap();
            worst.0 = worst.0.max(best.0);
            worst.1 = worst.1.max(best.1);
        }
        assert!(worst.0 < 1e-8 && worst.1 < 1e-8, "{worst:?}");
    }

    #[test]
    fn epnp_noise_free() {
        let mut rng = rand_pcg::Pcg64::seed_from_u64(8);
        for n in [6usize, 50, 500] {
            for _ in 0..20 {
                let pose = random_pose(&mut rng);
                let pts = random_points(&mut rng, &pose, n);
                let rays: Vec<Vec3> = pts.iter().map(|p| (pose * *p).normalize()).collect();
                let est = epnp(&rays, &pts, None, None).unwrap();
                let (r, t) = pose_err(&est, &pose);
                assert!(r < 1e-6 && t < 1e-6, "n={n} r={r} t={t}");
            }
        }
    }

    fn opencv_cam() -> Camera {
        Camera::new(1, CameraModelKind::OpenCv, 1920, 1080, vec![1500.0, 1500.0, 960.0, 540.0, -0.1, 0.02, 0.001, 0.001]).unwrap()
    }

    #[test]
    fn lo_ransac_with_outliers() {
        let cam = opencv_cam();
        let mut rng = rand_pcg::Pcg64::seed_from_u64(9);
        let mut successes = 0;
        for trial in 0..100 {
            let pose = random_pose(&mut rng);
            let inv = pose.inverse();
            let mut p2 = Vec::new();
            let mut p3 = Vec::new();
            let mut truth = Vec::new();
            while p2.len() < 300 {
                let xy = Vec2::new(rng.random_range(0.0..1920.0), rng.random_range(0.0..1080.0));
                let depth = rng.random_range(3.0..10.0);
                let ray = cam.img_to_ray(&xy).unwrap();
                let x = inv * (ray / ray.z * depth);
                let outlier = p2.len() < 90;
                if outlier {
                    p2.push(Vec2::new(rng.random_range(0.0..1920.0), rng.random_range(0.0..1080.0)));
                } else {
                    let g = crate::math::gaussian;
                    p2.push(xy + Vec2::new(g(&mut rng), g(&mut rng)) * 0.5);
                }
                p3.push(x);
                truth.push(!outlier);
            }
            let opts = AbsolutePoseOptions { random_seed: Some(trial), ..Default::default() };
            let Some(res) = solve_abs_pose(&cam, &p2, &p3, &opts) else { continue };
            // 등록 경로에서는 RANSAC 뒤 비선형 정제(refine_abs_pose)가 따른다. 여기서는
            // 그 대신 인라이어 전체 EPnP 로 다듬어 평가한다(LO 는 인라이어 수 우선이라
            // 경계 이상치 1개를 품은 P3P 모델이 남을 수 있음).
            let (ir, (ip2, ip3)): (Vec<Vec3>, (Vec<Vec2>, Vec<Vec3>)) =
                (0..p2.len()).filter(|&i| res.inlier_mask[i]).map(|i| (cam.img_to_ray(&p2[i]).unwrap(), (p2[i], p3[i]))).unzip();
            let polished = epnp(&ir, &ip3, Some(&cam), Some(&ip2)).unwrap();
            let rot = crate::math::rotation_angle_between(&polished.rotation_matrix(), &pose.rotation_matrix()).to_degrees();
            let cerr = (polished.center() - pose.center()).norm();
            let tp = truth.iter().zip(&res.inlier_mask).filter(|(a, b)| **a && **b).count();
            let recall = tp as f64 / 210.0;
            if rot < 0.1 && cerr < 0.005 * 6.5 && recall > 0.95 {
                successes += 1;
            } else {
                eprintln!("trial {trial}: rot {rot} cerr {cerr} recall {recall} inl {}", res.num_inliers);
            }
        }
        assert_eq!(successes, 100);
    }
}
