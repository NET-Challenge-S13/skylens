/*
 * math.rs
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

//! 내부 수치 유틸: SO(3) 지수/로그, 점 정렬(Kabsch/Umeyama), 다항식 근, 희소 행렬.

use cumulus3d_core::{Mat3, Quat, Sim3, Vec3};
use nalgebra::{DMatrix, DVector, Matrix3};

/// 회전 벡터 → 회전 행렬.
pub(crate) fn so3_exp(w: &Vec3) -> Mat3 {
    Quat::from_rotation_vector(w).to_rotation_matrix()
}

/// 회전 행렬 → 회전 벡터(각 ∈ [0, π]).
pub(crate) fn so3_log(r: &Mat3) -> Vec3 {
    Quat::from_rotation_matrix(r).to_rotation_vector()
}

/// 두 회전 행렬 사이 각(라디안).
#[allow(dead_code)]
pub(crate) fn rotation_angle_between(a: &Mat3, b: &Mat3) -> f64 {
    so3_log(&(a.transpose() * b)).norm()
}

/// 3×3 SVD (U, σ 내림차순, V). nalgebra 일반 SVD 는 중복 특이값 3×3 에서 가끔 틀려
/// matching 의 단측 야코비 구현을 쓴다.
pub(crate) fn svd3(m: &Mat3) -> Option<(Mat3, Vec3, Mat3)> {
    cumulus3d_matching::linalg::svd3(m)
}

/// 3×3 행렬을 가장 가까운 회전으로(SVD).
pub(crate) fn nearest_rotation(m: &Mat3) -> Mat3 {
    let Some((u, _, v)) = svd3(m) else { return Mat3::identity() };
    let vt = v.transpose();
    let mut d = Mat3::identity();
    if (u * vt).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    u * d * vt
}

/// dst ≈ R·src + t 의 최소제곱 강체 정렬(Kabsch). 점 3개 이상.
pub(crate) fn kabsch(src: &[Vec3], dst: &[Vec3]) -> Option<(Mat3, Vec3)> {
    let n = src.len();
    if n < 3 || dst.len() != n {
        return None;
    }
    let cs = src.iter().sum::<Vec3>() / n as f64;
    let cd = dst.iter().sum::<Vec3>() / n as f64;
    let mut h = Mat3::zeros();
    for (s, d) in src.iter().zip(dst) {
        h += (s - cs) * (d - cd).transpose();
    }
    let (u, _, v) = svd3(&h)?;
    // 특이값은 내림차순: 반사 보정은 가장 작은(마지막) 성분에.
    let mut d = Mat3::identity();
    if (v * u.transpose()).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let r = v * d * u.transpose();
    if !r.iter().all(|x| x.is_finite()) {
        return None;
    }
    Some((r, cd - r * cs))
}

/// dst ≈ s·R·src + t 상사 정렬(Umeyama). 평가·테스트용.
pub fn umeyama(src: &[Vec3], dst: &[Vec3]) -> Option<Sim3> {
    let n = src.len();
    if n < 3 || dst.len() != n {
        return None;
    }
    let cs = src.iter().sum::<Vec3>() / n as f64;
    let cd = dst.iter().sum::<Vec3>() / n as f64;
    let mut h = Mat3::zeros();
    let mut var = 0.0;
    for (s, d) in src.iter().zip(dst) {
        h += (d - cd) * (s - cs).transpose();
        var += (s - cs).norm_squared();
    }
    h /= n as f64;
    var /= n as f64;
    if var <= 0.0 {
        return None;
    }
    let (u, sv, v) = svd3(&h)?;
    let vt = v.transpose();
    let mut d = Mat3::identity();
    if (u * vt).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let r = u * d * vt;
    let s = (sv[0] * d[(0, 0)] + sv[1] * d[(1, 1)] + sv[2] * d[(2, 2)]) / var;
    let t = cd - s * (r * cs);
    Some(Sim3::new(s, Quat::from_rotation_matrix(&r), t))
}

/// 표준 정규 난수(박스-뮬러). 합성 자료·테스트용.
pub fn gaussian<R: rand::Rng + ?Sized>(rng: &mut R) -> f64 {
    use rand::RngExt;
    let u1: f64 = rng.random_range(1e-12..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// 실근: a x² + b x + c = 0 (a≈0 이면 일차).
pub(crate) fn solve_quadratic_real(a: f64, b: f64, c: f64) -> Vec<f64> {
    let scale = a.abs().max(b.abs()).max(c.abs());
    if scale == 0.0 {
        return Vec::new();
    }
    if a.abs() <= 1e-14 * scale {
        if b.abs() <= 1e-14 * scale {
            return Vec::new();
        }
        return vec![-c / b];
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        // 수치 잡음으로 살짝 음수면 중근 처리.
        if disc > -1e-12 * b * b {
            return vec![-b / (2.0 * a)];
        }
        return Vec::new();
    }
    let sq = disc.sqrt();
    // 소거 오차를 피하는 형태.
    let q = -0.5 * (b + b.signum() * sq);
    if q == 0.0 {
        return vec![0.0];
    }
    vec![q / a, c / q]
}

/// 실근: c3 x³ + c2 x² + c1 x + c0 = 0, 뉴턴 보정 포함.
pub(crate) fn solve_cubic_real(c3: f64, c2: f64, c1: f64, c0: f64) -> Vec<f64> {
    let scale = c3.abs().max(c2.abs()).max(c1.abs()).max(c0.abs());
    if scale == 0.0 {
        return Vec::new();
    }
    if c3.abs() <= 1e-14 * scale {
        return solve_quadratic_real(c2, c1, c0);
    }
    let (a, b, c) = (c2 / c3, c1 / c3, c0 / c3);
    let p = b - a * a / 3.0;
    let q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c;
    let disc = (q / 2.0).powi(2) + (p / 3.0).powi(3);
    let mut roots = if disc > 0.0 {
        let s = disc.sqrt();
        vec![(-q / 2.0 + s).cbrt() + (-q / 2.0 - s).cbrt() - a / 3.0]
    } else if p.abs() < 1e-300 {
        vec![-a / 3.0]
    } else {
        let r = 2.0 * (-p / 3.0).sqrt();
        let arg = (3.0 * q / (2.0 * p) * (-3.0 / p).sqrt()).clamp(-1.0, 1.0);
        let phi = arg.acos() / 3.0;
        (0..3).map(|k| r * (phi - 2.0 * std::f64::consts::PI * k as f64 / 3.0).cos() - a / 3.0).collect()
    };
    for x in roots.iter_mut() {
        for _ in 0..4 {
            let f = ((c3 * *x + c2) * *x + c1) * *x + c0;
            let df = (3.0 * c3 * *x + 2.0 * c2) * *x + c1;
            if df.abs() < 1e-300 {
                break;
            }
            let nx = *x - f / df;
            if !nx.is_finite() {
                break;
            }
            *x = nx;
        }
    }
    roots
}

/// 3×3 대칭 행렬의 수반 행렬(adjugate).
pub(crate) fn adjugate3(m: &Matrix3<f64>) -> Matrix3<f64> {
    let c = |r0: usize, r1: usize, c0: usize, c1: usize| m[(r0, c0)] * m[(r1, c1)] - m[(r0, c1)] * m[(r1, c0)];
    // 여인자 행렬의 전치.
    Matrix3::new(
        c(1, 2, 1, 2),
        -c(0, 2, 1, 2),
        c(0, 1, 1, 2),
        -c(1, 2, 0, 2),
        c(0, 2, 0, 2),
        -c(0, 1, 0, 2),
        c(1, 2, 0, 1),
        -c(0, 2, 0, 1),
        c(0, 1, 0, 1),
    )
}

/// 행 압축(CSR) 희소 행렬. 회전 평균의 계수 행렬처럼 행마다 원소가 몇 개뿐인 경우용.
#[derive(Clone, Debug, Default)]
pub struct CsrMatrix {
    /// 행 수.
    pub nrows: usize,
    /// 열 수.
    pub ncols: usize,
    /// 행 시작 위치(길이 nrows+1).
    pub row_ptr: Vec<usize>,
    /// 원소별 열 인덱스.
    pub col_idx: Vec<usize>,
    /// 원소 값.
    pub values: Vec<f64>,
}

impl CsrMatrix {
    /// 열 수만 정한 빈 행렬.
    pub fn new(ncols: usize) -> Self {
        Self { nrows: 0, ncols, row_ptr: vec![0], col_idx: Vec::new(), values: Vec::new() }
    }
    /// 행 하나 추가: (열, 값) 목록.
    pub fn push_row(&mut self, entries: &[(usize, f64)]) {
        for &(c, v) in entries {
            debug_assert!(c < self.ncols);
            self.col_idx.push(c);
            self.values.push(v);
        }
        self.row_ptr.push(self.col_idx.len());
        self.nrows += 1;
    }
    /// 밀집 행렬에서 0 아닌 원소만 담아 만든다.
    pub fn from_dense(m: &DMatrix<f64>) -> Self {
        let mut s = Self::new(m.ncols());
        for r in 0..m.nrows() {
            let e: Vec<(usize, f64)> = (0..m.ncols()).filter(|&c| m[(r, c)] != 0.0).map(|c| (c, m[(r, c)])).collect();
            s.push_row(&e);
        }
        s
    }
    /// y = A x.
    pub fn mul_vec(&self, x: &DVector<f64>) -> DVector<f64> {
        let mut y = DVector::zeros(self.nrows);
        for r in 0..self.nrows {
            let mut s = 0.0;
            for k in self.row_ptr[r]..self.row_ptr[r + 1] {
                s += self.values[k] * x[self.col_idx[k]];
            }
            y[r] = s;
        }
        y
    }
    /// x = Aᵀ y.
    pub fn tr_mul_vec(&self, y: &DVector<f64>) -> DVector<f64> {
        let mut x = DVector::zeros(self.ncols);
        for r in 0..self.nrows {
            let yr = y[r];
            if yr == 0.0 {
                continue;
            }
            for k in self.row_ptr[r]..self.row_ptr[r + 1] {
                x[self.col_idx[k]] += self.values[k] * yr;
            }
        }
        x
    }
    /// Aᵀ W A + λ I (밀집). `w` 는 행 가중(None = 1).
    pub fn weighted_normal(&self, w: Option<&[f64]>, lambda: f64) -> DMatrix<f64> {
        let mut n = DMatrix::zeros(self.ncols, self.ncols);
        for r in 0..self.nrows {
            let wr = w.map_or(1.0, |w| w[r]);
            if wr == 0.0 {
                continue;
            }
            let (a, b) = (self.row_ptr[r], self.row_ptr[r + 1]);
            for i in a..b {
                let vi = self.values[i] * wr;
                let ci = self.col_idx[i];
                for j in a..b {
                    n[(ci, self.col_idx[j])] += vi * self.values[j];
                }
            }
        }
        for i in 0..self.ncols {
            n[(i, i)] += lambda;
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_roots() {
        // (x-1)(x-2)(x+3) = x³ - 7x + 6
        let mut r = solve_cubic_real(1.0, 0.0, -7.0, 6.0);
        r.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(r.len(), 3);
        for (a, b) in r.iter().zip([-3.0, 1.0, 2.0]) {
            assert!((a - b).abs() < 1e-12);
        }
        let r = solve_cubic_real(2.0, 0.0, 0.0, -16.0);
        assert_eq!(r.len(), 1);
        assert!((r[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn kabsch_and_umeyama() {
        let r = so3_exp(&Vec3::new(0.3, -0.2, 0.5));
        let t = Vec3::new(1.0, 2.0, -3.0);
        let src = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 2.0, 0.5)];
        let dst: Vec<Vec3> = src.iter().map(|p| r * p + t).collect();
        let (r2, t2) = kabsch(&src, &dst).unwrap();
        assert!((r2 - r).norm() < 1e-12 && (t2 - t).norm() < 1e-12);
        let dst2: Vec<Vec3> = src.iter().map(|p| 2.5 * (r * p) + t).collect();
        let s = umeyama(&src, &dst2).unwrap();
        assert!((s.scale - 2.5).abs() < 1e-12);
        for (a, b) in src.iter().zip(&dst2) {
            assert!((s.transform_point(a) - b).norm() < 1e-10);
        }
    }
}
