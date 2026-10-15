/*
 * linalg.rs
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

//! 작은 선형대수 도우미(영공간, 3×3 SVD, 하틀리 정규화).

use cumulus3d_core::{Mat3, Vec2, Vec3};
use nalgebra::{DMatrix, SMatrix};

/// 행 우선 N×9 제약 행렬의 영공간 기저(가장 작은 특이값 순서대로 `k` 개).
///
/// N < 9 이면 영행으로 채워 9×9 로 만든 뒤 SVD 한다(기저가 생성하는 공간은 QR 판과 같다).
pub fn null_space_9(rows: &[[f64; 9]], k: usize) -> Option<Vec<[f64; 9]>> {
    let (sv, vt) = svd_rows_9(rows)?;
    let mut order: Vec<usize> = (0..9).collect();
    order.sort_by(|&x, &y| sv[y].total_cmp(&sv[x]));
    let mut out = Vec::with_capacity(k);
    for idx in 0..k {
        let row = order[8 - idx];
        let mut v = [0.0; 9];
        for j in 0..9 {
            v[j] = vt[(row, j)];
        }
        out.push(v);
    }
    Some(out)
}

/// 특이값(정렬 안 됨)과 Vᵀ(9×9). 9행 이하는 고정 크기 행렬로 할당 없이 처리.
fn svd_rows_9(rows: &[[f64; 9]]) -> Option<([f64; 9], SMatrix<f64, 9, 9>)> {
    let mut sv = [0.0; 9];
    if rows.len() <= 9 {
        let mut a = SMatrix::<f64, 9, 9>::zeros();
        for (i, r) in rows.iter().enumerate() {
            for j in 0..9 {
                a[(i, j)] = r[j];
            }
        }
        let svd = a.try_svd(false, true, f64::EPSILON, 1000)?;
        sv.copy_from_slice(svd.singular_values.as_slice());
        Some((sv, svd.v_t?))
    } else {
        // 큰 N: QᵀQ 대신 직접 SVD(조건수 제곱 방지).
        let mut a = DMatrix::<f64>::zeros(rows.len(), 9);
        for (i, r) in rows.iter().enumerate() {
            for j in 0..9 {
                a[(i, j)] = r[j];
            }
        }
        let svd = a.try_svd(false, true, f64::EPSILON, 1000)?;
        sv.copy_from_slice(svd.singular_values.as_slice());
        let vt = svd.v_t?;
        Some((sv, SMatrix::<f64, 9, 9>::from_fn(|i, j| vt[(i, j)])))
    }
}

/// 9열 행렬 SVD 의 특이값(내림차순)과 최소 우특이벡터.
pub fn singular_values_9(rows: &[[f64; 9]]) -> Option<(Vec<f64>, [f64; 9])> {
    let (sv, vt) = svd_rows_9(rows)?;
    let mut order: Vec<usize> = (0..9).collect();
    order.sort_by(|&x, &y| sv[y].total_cmp(&sv[x]));
    let s: Vec<f64> = order.iter().map(|&i| sv[i]).collect();
    let mut v = [0.0; 9];
    for j in 0..9 {
        v[j] = vt[(order[8], j)];
    }
    Some((s, v))
}

/// 행 우선 9-벡터 → 3×3.
pub fn mat3_from_row_major(e: &[f64; 9]) -> Mat3 {
    Mat3::new(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8])
}

pub use cumulus3d_core::linalg::svd3;

/// 3×3 의 (근사) 영벡터: 행 쌍 외적 중 가장 큰 것.
pub fn null_vector3(m: &Mat3) -> Vec3 {
    let r0 = m.row(0).transpose();
    let r1 = m.row(1).transpose();
    let r2 = m.row(2).transpose();
    let c = [r0.cross(&r1), r0.cross(&r2), r1.cross(&r2)];
    let mut best = c[0];
    for v in &c[1..] {
        if v.norm_squared() > best.norm_squared() {
            best = *v;
        }
    }
    let n = best.norm();
    if n > 0.0 {
        best / n
    } else {
        Vec3::zeros()
    }
}

/// 하틀리 정규화: 무게중심 c, 무게중심까지 거리 RMS = s, k = √2/s.
/// `T = [[k,0,−k cx],[0,k,−k cy],[0,0,1]]`. 정규화 점과 T 를 돌려준다.
pub fn hartley_normalize(pts: &[Vec2]) -> (Vec<Vec2>, Mat3) {
    let n = pts.len().max(1) as f64;
    let c = pts.iter().fold(Vec2::zeros(), |a, p| a + p) / n;
    let ms = pts.iter().map(|p| (p - c).norm_squared()).sum::<f64>() / n;
    let rms = ms.sqrt();
    let k = if rms > 0.0 { std::f64::consts::SQRT_2 / rms } else { 1.0 };
    let t = Mat3::new(k, 0.0, -k * c.x, 0.0, k, -k * c.y, 0.0, 0.0, 1.0);
    (pts.iter().map(|p| (p - c) * k).collect(), t)
}

/// 행 우선 8×8 선형계 부분 피벗 LU. 특이하면 None.
pub fn solve8(a: SMatrix<f64, 8, 8>, b: SMatrix<f64, 8, 1>) -> Option<SMatrix<f64, 8, 1>> {
    a.lu().solve(&b)
}
