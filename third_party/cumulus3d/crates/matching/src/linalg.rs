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

//! Small linear-algebra helpers (null space, 3×3 SVD, Hartley normalization).

use cumulus3d_core::{Mat3, Vec2, Vec3};
use nalgebra::{DMatrix, SMatrix};

/// Null-space basis of a row-major N×9 constraint matrix (`k` vectors in order of smallest singular value).
///
/// If N < 9, pad with zero rows to make it 9×9 and then run SVD (the space spanned by the basis is the same as the QR version).
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

/// Singular values (unsorted) and Vᵀ (9×9). Up to 9 rows are handled with a fixed-size matrix without allocation.
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
        // Large N: direct SVD instead of QᵀQ (avoids squaring the condition number).
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

/// Singular values (descending) and smallest right singular vector of the SVD of a 9-column matrix.
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

/// Row-major 9-vector -> 3×3.
pub fn mat3_from_row_major(e: &[f64; 9]) -> Mat3 {
    Mat3::new(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8])
}

pub use cumulus3d_core::linalg::svd3;

/// (Approximate) null vector of a 3×3: the largest of the row-pair cross products.
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

/// Hartley normalization: centroid c, RMS distance to the centroid = s, k = √2/s.
/// `T = [[k,0,-k cx],[0,k,-k cy],[0,0,1]]`. Returns the normalized points and T.
pub fn hartley_normalize(pts: &[Vec2]) -> (Vec<Vec2>, Mat3) {
    let n = pts.len().max(1) as f64;
    let c = pts.iter().fold(Vec2::zeros(), |a, p| a + p) / n;
    let ms = pts.iter().map(|p| (p - c).norm_squared()).sum::<f64>() / n;
    let rms = ms.sqrt();
    let k = if rms > 0.0 { std::f64::consts::SQRT_2 / rms } else { 1.0 };
    let t = Mat3::new(k, 0.0, -k * c.x, 0.0, k, -k * c.y, 0.0, 0.0, 1.0);
    (pts.iter().map(|p| (p - c) * k).collect(), t)
}

/// Row-major 8×8 linear system partial-pivot LU. None if singular.
pub fn solve8(a: SMatrix<f64, 8, 8>, b: SMatrix<f64, 8, 1>) -> Option<SMatrix<f64, 8, 1>> {
    a.lu().solve(&b)
}
