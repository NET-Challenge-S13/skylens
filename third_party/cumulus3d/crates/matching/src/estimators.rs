/*
 * estimators.rs
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

//! 최소/비최소 해법과 잔차, core RANSAC `Estimator` 구현.

use crate::essential::{epipolar_row, essential_five_point};
use crate::linalg::{hartley_normalize, mat3_from_row_major, null_space_9, singular_values_9, solve8, svd3};
use crate::poly::solve_cubic_monic;
use cumulus3d_core::ransac::Estimator;
use cumulus3d_core::{Mat3, Vec2, Vec3};
use nalgebra::SMatrix;

/// Sampson 제곱 오차. l = M a, 분자 = (bᵀ l)², 분모 = (bᵀM)_x² + (bᵀM)_y² + l_x² + l_y². 분모 0 → +∞.
#[inline]
pub fn sampson_error_sq(m: &Mat3, a: &Vec3, b: &Vec3) -> f64 {
    let l = m * a;
    let bm = m.tr_mul(b); // (bᵀ M)ᵀ
    let num = b.dot(&l);
    let den = bm.x * bm.x + bm.y * bm.y + l.x * l.x + l.y * l.y;
    if den == 0.0 {
        f64::INFINITY
    } else {
        num * num / den
    }
}

/// 호모그래피 단방향 전이 제곱 오차 ‖x2 − π(H x1)‖². 유한하지 않으면 +∞.
#[inline]
pub fn homography_transfer_error_sq(h: &Mat3, x1: &Vec2, x2: &Vec2) -> f64 {
    let p = h * Vec3::new(x1.x, x1.y, 1.0);
    let dx = x2.x - p.x / p.z;
    let dy = x2.y - p.y / p.z;
    let r = dx * dx + dy * dy;
    if r.is_finite() {
        r
    } else {
        f64::INFINITY
    }
}

#[inline]
fn homog(p: &Vec2) -> Vec3 {
    Vec3::new(p.x, p.y, 1.0)
}

// ---------------- Fundamental ----------------

/// 7점 fundamental (정규화 없음). 최대 3개, 각 프로베니우스 노름 1.
pub fn fundamental_seven_point(x1: &[Vec2], x2: &[Vec2]) -> Vec<Mat3> {
    if x1.len() != 7 || x2.len() != 7 {
        return Vec::new();
    }
    let rows: Vec<[f64; 9]> = x1.iter().zip(x2).map(|(a, b)| epipolar_row(&homog(a), &homog(b))).collect();
    let Some(ns) = null_space_9(&rows, 2) else { return Vec::new() };
    let fa = mat3_from_row_major(&ns[0]);
    let fb = mat3_from_row_major(&ns[1]);
    // det(λ D + B), D = fa − fb, B = fb: 다중선형성으로 열 혼합 행렬식.
    let d = fa - fb;
    let b = fb;
    let det3 = |c0: Vec3, c1: Vec3, c2: Vec3| c0.dot(&c1.cross(&c2));
    let (d0, d1, d2) = (d.column(0).into_owned(), d.column(1).into_owned(), d.column(2).into_owned());
    let (b0, b1, b2) = (b.column(0).into_owned(), b.column(1).into_owned(), b.column(2).into_owned());
    let c3 = det3(d0, d1, d2);
    let c2 = det3(d0, d1, b2) + det3(d0, b1, d2) + det3(b0, d1, d2);
    let c1 = det3(d0, b1, b2) + det3(b0, d1, b2) + det3(b0, b1, d2);
    let c0 = det3(b0, b1, b2);
    if c3.abs() < 1e-16 {
        return Vec::new();
    }
    solve_cubic_monic(c2 / c3, c1 / c3, c0 / c3)
        .into_iter()
        .filter_map(|l| {
            let f = d * l + b;
            let n = f.norm();
            (n > 0.0 && n.is_finite()).then(|| f / n)
        })
        .collect()
}

/// 정규화 8점(이상) fundamental. rank-2 강제 후 역정규화. 모델 1개.
pub fn fundamental_eight_point(x1: &[Vec2], x2: &[Vec2]) -> Option<Mat3> {
    if x1.len() < 8 || x1.len() != x2.len() {
        return None;
    }
    let (n1, t1) = hartley_normalize(x1);
    let (n2, t2) = hartley_normalize(x2);
    let rows: Vec<[f64; 9]> = n1.iter().zip(&n2).map(|(a, b)| epipolar_row(&homog(a), &homog(b))).collect();
    let (_, v) = singular_values_9(&rows)?;
    let fh = mat3_from_row_major(&v);
    let (u, s, vv) = svd3(&fh)?;
    let fr = u * Mat3::from_diagonal(&Vec3::new(s[0], s[1], 0.0)) * vv.transpose();
    let f = t2.transpose() * fr * t1;
    f.iter().all(|x| x.is_finite()).then_some(f)
}

// ---------------- Homography ----------------

/// DLT 호모그래피. N = 4: h33 = 1 고정 8×8 LU, N > 4: 2N×9 SVD(수치 랭크 < 8 이면 실패).
/// `normalize` 가 참이면 하틀리 정규화 후 역정규화(개선, 기본 끔). |det H| < 1e-8 이면 실패
/// (N점 SVD 해는 단위 노름이라 평행이동 성분이 큰 픽셀 H 는 det 가 작아 거부될 수 있다 — 기본 동작).
pub fn homography_dlt(x1: &[Vec2], x2: &[Vec2], normalize: bool) -> Option<Mat3> {
    let n = x1.len();
    if n < 4 || x2.len() != n {
        return None;
    }
    let (p1, p2, t1, t2) = if normalize {
        let (a, ta) = hartley_normalize(x1);
        let (b, tb) = hartley_normalize(x2);
        (a, b, ta, tb)
    } else {
        (x1.to_vec(), x2.to_vec(), Mat3::identity(), Mat3::identity())
    };
    let hn = if n == 4 {
        let mut a = SMatrix::<f64, 8, 8>::zeros();
        let mut rhs = SMatrix::<f64, 8, 1>::zeros();
        for i in 0..4 {
            let (u, v) = (p1[i].x, p1[i].y);
            let (s, t) = (p2[i].x, p2[i].y);
            let r0 = [u, v, 1.0, 0.0, 0.0, 0.0, -s * u, -s * v];
            let r1 = [0.0, 0.0, 0.0, u, v, 1.0, -t * u, -t * v];
            for j in 0..8 {
                a[(2 * i, j)] = r0[j];
                a[(2 * i + 1, j)] = r1[j];
            }
            rhs[2 * i] = s;
            rhs[2 * i + 1] = t;
        }
        let h = solve8(a, rhs)?;
        if h.iter().any(|v| v.is_nan()) {
            return None;
        }
        Mat3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0)
    } else {
        let mut rows = Vec::with_capacity(2 * n);
        for i in 0..n {
            let (u, v) = (p1[i].x, p1[i].y);
            let (s, t) = (p2[i].x, p2[i].y);
            rows.push([u, v, 1.0, 0.0, 0.0, 0.0, -s * u, -s * v, -s]);
            rows.push([0.0, 0.0, 0.0, u, v, 1.0, -t * u, -t * v, -t]);
        }
        let (sv, v) = singular_values_9(&rows)?;
        // 설계 결정: 수치 랭크 허용오차. 표준 max(행,열)·ε·σ_max 사용.
        let tol = (2 * n).max(9) as f64 * f64::EPSILON * sv[0];
        let rank = sv.iter().filter(|&&s| s > tol).count();
        if rank < 8 {
            return None;
        }
        mat3_from_row_major(&v)
    };
    let mut h = t2.try_inverse()? * hn * t1;
    // det 임계의 스케일 기준: 4점 = h33 = 1, N점 = 프로베니우스 노름 1.
    if normalize {
        h = if n == 4 { h / h[(2, 2)] } else { h / h.norm() };
    }
    if !h.iter().all(|x| x.is_finite()) || h.determinant().abs() < 1e-8 {
        return None;
    }
    Some(h)
}

// ---------------- Estimator 구현 ----------------

/// 5점 essential(최소) 겸 N점 국소 추정기. 입력은 단위 광선.
#[derive(Clone, Copy, Debug, Default)]
pub struct EssentialFivePointEstimator;

impl Estimator for EssentialFivePointEstimator {
    type X = Vec3;
    type Y = Vec3;
    type Model = Mat3;
    fn min_num_samples(&self) -> usize {
        5
    }
    fn estimate(&self, x: &[Vec3], y: &[Vec3], models: &mut Vec<Mat3>) {
        models.extend(essential_five_point(x, y));
    }
    fn residuals(&self, x: &[Vec3], y: &[Vec3], m: &Mat3, r: &mut Vec<f64>) {
        r.clear();
        r.extend(x.iter().zip(y).map(|(a, b)| sampson_error_sq(m, a, b)));
    }
}

fn fundamental_residuals(x: &[Vec2], y: &[Vec2], m: &Mat3, r: &mut Vec<f64>) {
    r.clear();
    r.extend(x.iter().zip(y).map(|(a, b)| sampson_error_sq(m, &homog(a), &homog(b))));
}

/// 7점 fundamental(최소 해법). 픽셀 좌표.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fundamental7PtEstimator;

impl Estimator for Fundamental7PtEstimator {
    type X = Vec2;
    type Y = Vec2;
    type Model = Mat3;
    fn min_num_samples(&self) -> usize {
        7
    }
    fn estimate(&self, x: &[Vec2], y: &[Vec2], models: &mut Vec<Mat3>) {
        models.extend(fundamental_seven_point(x, y));
    }
    fn residuals(&self, x: &[Vec2], y: &[Vec2], m: &Mat3, r: &mut Vec<f64>) {
        fundamental_residuals(x, y, m, r);
    }
}

/// 정규화 8점 fundamental(국소 추정기). 픽셀 좌표.
#[derive(Clone, Copy, Debug, Default)]
pub struct FundamentalEightPointEstimator;

impl Estimator for FundamentalEightPointEstimator {
    type X = Vec2;
    type Y = Vec2;
    type Model = Mat3;
    fn min_num_samples(&self) -> usize {
        8
    }
    fn estimate(&self, x: &[Vec2], y: &[Vec2], models: &mut Vec<Mat3>) {
        models.extend(fundamental_eight_point(x, y));
    }
    fn residuals(&self, x: &[Vec2], y: &[Vec2], m: &Mat3, r: &mut Vec<f64>) {
        fundamental_residuals(x, y, m, r);
    }
}

/// DLT 호모그래피(최소 4점 겸 N점). 잔차 = 단방향 전이 제곱 오차.
#[derive(Clone, Copy, Debug, Default)]
pub struct HomographyEstimator {
    /// 하틀리 정규화(개선, 기본 끔).
    pub normalize: bool,
}

impl Estimator for HomographyEstimator {
    type X = Vec2;
    type Y = Vec2;
    type Model = Mat3;
    fn min_num_samples(&self) -> usize {
        4
    }
    fn estimate(&self, x: &[Vec2], y: &[Vec2], models: &mut Vec<Mat3>) {
        models.extend(homography_dlt(x, y, self.normalize));
    }
    fn residuals(&self, x: &[Vec2], y: &[Vec2], m: &Mat3, r: &mut Vec<f64>) {
        r.clear();
        r.extend(x.iter().zip(y).map(|(a, b)| homography_transfer_error_sq(m, a, b)));
    }
}

/// 2D 평행이동 t = 평균(x2) − 평균(x1) (워터마크 검출용).
#[derive(Clone, Copy, Debug, Default)]
pub struct TranslationEstimator;

impl Estimator for TranslationEstimator {
    type X = Vec2;
    type Y = Vec2;
    type Model = Vec2;
    fn min_num_samples(&self) -> usize {
        1
    }
    fn estimate(&self, x: &[Vec2], y: &[Vec2], models: &mut Vec<Vec2>) {
        if x.is_empty() {
            return;
        }
        let n = x.len() as f64;
        let mx = x.iter().fold(Vec2::zeros(), |a, p| a + p) / n;
        let my = y.iter().fold(Vec2::zeros(), |a, p| a + p) / n;
        models.push(my - mx);
    }
    fn residuals(&self, x: &[Vec2], y: &[Vec2], t: &Vec2, r: &mut Vec<f64>) {
        r.clear();
        r.extend(x.iter().zip(y).map(|(a, b)| (b - a - t).norm_squared()));
    }
}
