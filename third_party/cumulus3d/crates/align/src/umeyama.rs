/*
 * umeyama.rs
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

//! Umeyama Sim3 and robust estimation.

use cumulus3d_core::ransac::{lo_ransac, Estimator, RansacParams, RansacReport};
use cumulus3d_core::{Mat3, Mat3x4, Sim3, Vec3};
use nalgebra::Matrix3xX;

/// Degeneracy test method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RankCheck {
    /// Default behavior: reject if the rank of the uncentered 3×n coordinate matrix is < 3
    /// (points on a plane through the origin may also be rejected).
    #[default]
    Uncentered,
    /// Improved: reject if the rank of the centered coordinates is < 2 (rejects only collinear or single-point configurations).
    Centered,
    /// No test (only checks the SVD solution).
    None,
}

fn numerical_rank(m: &Matrix3xX<f64>) -> usize {
    let n = m.ncols();
    if n == 0 {
        return 0;
    }
    let sv = m.clone().svd(false, false).singular_values;
    let max = sv.iter().cloned().fold(0.0f64, f64::max);
    if max <= 0.0 {
        return 0;
    }
    // Design decision: rank decision threshold. Uses the standard numerical rank criterion σ > σ_max·ε·max(3, n).
    let tol = max * f64::EPSILON * (n.max(3) as f64);
    sv.iter().filter(|s| **s > tol).count()
}

fn passes_rank(pts: &[Vec3], check: RankCheck) -> bool {
    match check {
        RankCheck::None => true,
        RankCheck::Uncentered => numerical_rank(&Matrix3xX::from_columns(pts)) >= 3,
        RankCheck::Centered => {
            let mu = pts.iter().sum::<Vec3>() / pts.len() as f64;
            let c: Vec<Vec3> = pts.iter().map(|p| p - mu).collect();
            numerical_rank(&Matrix3xX::from_columns(&c)) >= 2
        }
    }
}

/// Umeyama solver (unweighted). Sim3 with `dst ≈ s R src + t` (new_from_old = src_to_dst).
/// s = 1 if `estimate_scale = false`. None if there are fewer than 3 points or the solution is NaN/degenerate.
pub fn umeyama(src: &[Vec3], dst: &[Vec3], estimate_scale: bool) -> Option<Sim3> {
    let n = src.len();
    if n < 3 || n != dst.len() {
        return None;
    }
    let inv_n = 1.0 / n as f64;
    let mx = src.iter().sum::<Vec3>() * inv_n;
    let my = dst.iter().sum::<Vec3>() * inv_n;
    let mut var_x = 0.0;
    let mut cov = Mat3::zeros();
    for (x, y) in src.iter().zip(dst) {
        let dx = x - mx;
        var_x += dx.norm_squared();
        cov += (y - my) * dx.transpose();
    }
    var_x *= inv_n;
    cov *= inv_n;
    // The nalgebra 3×3 SVD can be wrong for repeated singular values, so the one-sided Jacobi SVD from core is used.
    let (u, d, v) = cumulus3d_core::linalg::svd3(&cov)?;
    let vt = v.transpose();
    let mut s_diag = Vec3::new(1.0, 1.0, 1.0);
    if u.determinant() * vt.determinant() < 0.0 {
        s_diag.z = -1.0;
    }
    let r = u * Mat3::from_diagonal(&s_diag) * vt;
    let scale = if estimate_scale {
        if var_x <= 0.0 {
            return None;
        }
        d.dot(&s_diag) / var_x
    } else {
        1.0
    };
    let t = my - scale * r * mx;
    let mut m = Mat3x4::zeros();
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(&(scale * r));
    m.set_column(3, &t);
    if m.iter().any(|v| !v.is_finite()) || scale <= 0.0 {
        return None;
    }
    // Convert the [sR|t] matrix to a Sim3 (s = norm of the first column, normalize the quaternion).
    Some(Sim3::from_matrix(&m))
}

/// Sim3 estimator for RANSAC (at least 3 pairs, residual = ‖y − T x‖²).
#[derive(Clone, Copy, Debug, Default)]
pub struct Sim3Estimator {
    /// Minimal-sample degeneracy test method.
    pub rank_check: RankCheck,
}

impl Estimator for Sim3Estimator {
    type X = Vec3;
    type Y = Vec3;
    type Model = Sim3;
    fn min_num_samples(&self) -> usize {
        3
    }
    fn estimate(&self, x: &[Vec3], y: &[Vec3], models: &mut Vec<Sim3>) {
        if !passes_rank(x, self.rank_check) || !passes_rank(y, self.rank_check) {
            return;
        }
        if let Some(m) = umeyama(x, y, true) {
            models.push(m);
        }
    }
    fn residuals(&self, x: &[Vec3], y: &[Vec3], model: &Sim3, residuals: &mut Vec<f64>) {
        let sr = model.rotation.to_rotation_matrix() * model.scale;
        residuals.clear();
        residuals.extend(x.iter().zip(y).map(|(a, b)| (b - (sr * a + model.translation)).norm_squared()));
    }
}

/// LO-RANSAC robust Sim3 (src → dst). `opts.max_error` is the distance threshold (before squaring).
pub fn estimate_sim3_ransac(src: &[Vec3], dst: &[Vec3], opts: &RansacParams, rank_check: RankCheck) -> RansacReport<Sim3> {
    let est = Sim3Estimator { rank_check };
    lo_ransac(&est, &est, opts, src, dst)
}

/// Robust Umeyama (user post-processing) options.
#[derive(Clone, Debug, PartialEq)]
pub struct RobustUmeyamaOptions {
    /// Number of refit iterations.
    pub iterations: usize,
    /// Threshold = `max(factor · median(r[keep]), min_threshold)`.
    pub factor: f64,
    /// Lower bound of the inlier threshold (distance).
    pub min_threshold: f64,
    /// Whether to estimate scale (if false, a rigid transform).
    pub estimate_scale: bool,
}

impl Default for RobustUmeyamaOptions {
    fn default() -> Self {
        Self { iterations: 5, factor: 3.0, min_threshold: 0.3, estimate_scale: true }
    }
}

/// Robust Umeyama result.
#[derive(Clone, Debug)]
pub struct RobustUmeyamaResult {
    /// src_to_dst.
    pub sim3: Sim3,
    /// Median of the residuals (distances) of the final inliers.
    pub median_residual: f64,
    /// Number of final inliers.
    pub num_inliers: usize,
    /// Per-pair inlier flags.
    pub inlier_mask: Vec<bool>,
    /// Final residuals (distances) of all pairs.
    pub residuals: Vec<f64>,
}

/// Median (mean of the middle two for an even count). NaN for empty input.
pub(crate) fn median(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    v.sort_unstable_by(|a, b| a.total_cmp(b));
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Iterative refit: each round Umeyama on keep, then all residuals r, then `keep = r < max(factor·median(r[keep]), min_threshold)`.
/// The returned Sim3 is the last fit; inliers and median are based on the last updated keep.
pub fn robust_umeyama(src: &[Vec3], dst: &[Vec3], opts: &RobustUmeyamaOptions) -> Option<RobustUmeyamaResult> {
    let n = src.len();
    if n < 3 || n != dst.len() {
        return None;
    }
    let mut keep = vec![true; n];
    let mut best: Option<(Sim3, Vec<f64>)> = None;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..opts.iterations.max(1) {
        xs.clear();
        ys.clear();
        for i in 0..n {
            if keep[i] {
                xs.push(src[i]);
                ys.push(dst[i]);
            }
        }
        // Design decision: if inliers drop below 3, keep the previous fit and stop.
        let Some(sim) = umeyama(&xs, &ys, opts.estimate_scale) else { break };
        let r: Vec<f64> = src.iter().zip(dst).map(|(a, b)| (b - sim.transform_point(a)).norm()).collect();
        let mut rk: Vec<f64> = r.iter().zip(&keep).filter(|(_, k)| **k).map(|(v, _)| *v).collect();
        let thr = (opts.factor * median(&mut rk)).max(opts.min_threshold);
        let new_keep: Vec<bool> = r.iter().map(|v| *v < thr).collect();
        best = Some((sim, r));
        if new_keep.iter().filter(|k| **k).count() < 3 {
            break;
        }
        keep = new_keep;
    }
    let (sim3, residuals) = best?;
    let mut rk: Vec<f64> = residuals.iter().zip(&keep).filter(|(_, k)| **k).map(|(v, _)| *v).collect();
    Some(RobustUmeyamaResult {
        sim3,
        median_residual: median(&mut rk),
        num_inliers: keep.iter().filter(|k| **k).count(),
        inlier_mask: keep,
        residuals,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use cumulus3d_core::Quat;
    use rand::RngExt;
    use rand_pcg::Pcg64;

    pub fn random_sim3(rng: &mut Pcg64) -> Sim3 {
        let axis = Vec3::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));
        let s = 10f64.powf(rng.random_range(-1.0..1.0));
        Sim3::new(
            s,
            Quat::from_axis_angle(&axis.normalize(), rng.random_range(-3.0..3.0)),
            Vec3::new(rng.random_range(-100.0..100.0), rng.random_range(-100.0..100.0), rng.random_range(-100.0..100.0)),
        )
    }
    pub fn rand_vec(rng: &mut Pcg64, r: f64) -> Vec3 {
        Vec3::new(rng.random_range(-r..r), rng.random_range(-r..r), rng.random_range(-r..r))
    }
    pub fn gauss(rng: &mut Pcg64) -> f64 {
        let u1: f64 = rng.random_range(1e-12..1.0);
        let u2: f64 = rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn sim_close(a: &Sim3, b: &Sim3, tol: f64) {
        assert!((a.scale - b.scale).abs() / b.scale < tol, "s {} vs {}", a.scale, b.scale);
        let ra = a.rotation.to_rotation_matrix();
        let rb = b.rotation.to_rotation_matrix();
        assert!((ra - rb).norm() < tol, "R");
        assert!((a.translation - b.translation).norm() / b.translation.norm().max(1.0) < tol, "t");
    }

    #[test]
    fn umeyama_exact() {
        let mut rng = cumulus3d_core::ransac::make_rng(Some(1));
        for _ in 0..50 {
            let t = random_sim3(&mut rng);
            let src: Vec<Vec3> = (0..10).map(|_| rand_vec(&mut rng, 10.0)).collect();
            let dst: Vec<Vec3> = src.iter().map(|p| t.transform_point(p)).collect();
            let e = umeyama(&src, &dst, true).unwrap();
            sim_close(&e, &t, 1e-9);
            assert!((e.rotation.to_rotation_matrix().determinant() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn umeyama_isotropic_points() {
        // Octahedron vertices: all three covariance singular values are equal (the case where the nalgebra 3×3 SVD was wrong).
        let mut rng = cumulus3d_core::ransac::make_rng(Some(7));
        let src: Vec<Vec3> = [Vec3::x(), -Vec3::x(), Vec3::y(), -Vec3::y(), Vec3::z(), -Vec3::z()].into();
        for _ in 0..50 {
            let t = random_sim3(&mut rng);
            let dst: Vec<Vec3> = src.iter().map(|p| t.transform_point(p)).collect();
            sim_close(&umeyama(&src, &dst, true).unwrap(), &t, 1e-9);
        }
    }

    #[test]
    fn umeyama_reflection_gives_rotation() {
        // Target is a mirror image: the optimal solution must be a rotation, not a reflection.
        let mut rng = cumulus3d_core::ransac::make_rng(Some(2));
        let src: Vec<Vec3> = (0..10).map(|_| rand_vec(&mut rng, 10.0)).collect();
        let dst: Vec<Vec3> = src.iter().map(|p| Vec3::new(p.x, p.y, -p.z)).collect();
        let e = umeyama(&src, &dst, true).unwrap();
        assert!((e.rotation.to_rotation_matrix().determinant() - 1.0).abs() < 1e-9);
        assert!(e.scale > 0.0);
    }

    #[test]
    fn rank_checks() {
        // Points on a plane through the origin (z=0): the uncentered test rejects, the centered test passes.
        let p = vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 1.0, 0.0)];
        assert!(!passes_rank(&p, RankCheck::Uncentered));
        assert!(passes_rank(&p, RankCheck::Centered));
        let q: Vec<Vec3> = p.iter().map(|v| v + Vec3::new(0.0, 0.0, 5.0)).collect();
        assert!(passes_rank(&q, RankCheck::Uncentered));
        let line = vec![Vec3::new(1.0, 1.0, 1.0), Vec3::new(2.0, 2.0, 2.0), Vec3::new(3.0, 3.0, 3.0)];
        assert!(!passes_rank(&line, RankCheck::Centered));
    }

    /// 30% synthetic outliers: both LO-RANSAC and robust Umeyama recover the original Sim3.
    #[test]
    fn sim3_recovery_with_outliers() {
        let mut rng = cumulus3d_core::ransac::make_rng(Some(3));
        let truth = random_sim3(&mut rng);
        let n = 200;
        let src: Vec<Vec3> = (0..n).map(|_| rand_vec(&mut rng, 50.0)).collect();
        let mut dst: Vec<Vec3> = src.iter().map(|p| truth.transform_point(p)).collect();
        let mut is_out = vec![false; n];
        for i in 0..n {
            if i % 10 < 3 {
                is_out[i] = true;
                dst[i] += rand_vec(&mut rng, 1.0).normalize() * (truth.scale * rng.random_range(30.0..200.0));
            }
        }
        let opts = RansacParams { max_error: 1e-3 * truth.scale, random_seed: Some(9), ..Default::default() };
        let rep = estimate_sim3_ransac(&src, &dst, &opts, RankCheck::Uncentered);
        assert!(rep.success);
        sim_close(rep.model.as_ref().unwrap(), &truth, 1e-8);
        let want: Vec<bool> = is_out.iter().map(|o| !o).collect();
        assert_eq!(rep.inlier_mask, want);
        let r = robust_umeyama(&src, &dst, &RobustUmeyamaOptions { min_threshold: 1e-6, ..Default::default() }).unwrap();
        sim_close(&r.sim3, &truth, 1e-8);
        assert_eq!(r.num_inliers, n - is_out.iter().filter(|b| **b).count());
        assert!(r.median_residual < 1e-6);
    }

    #[test]
    fn median_even_odd() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 2.0, 3.0]), 2.5);
    }
}
