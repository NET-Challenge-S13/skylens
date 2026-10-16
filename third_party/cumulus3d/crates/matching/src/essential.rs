/*
 * essential.rs
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

//! 5-point essential (Nistér method) and 8-point essential (for re-fitting).

use crate::linalg::{mat3_from_row_major, null_space_9, null_vector3, singular_values_9, svd3};
use crate::poly::{poly_add, poly_mul, roots_companion};
use cumulus3d_core::{Mat3, Vec3};
use nalgebra::SMatrix;

/// Epipolar constraint row bᵀ M a = 0 (row-major 9-vector).
#[inline]
pub fn epipolar_row(a: &Vec3, b: &Vec3) -> [f64; 9] {
    [b.x * a.x, b.x * a.y, b.x * a.z, b.y * a.x, b.y * a.y, b.y * a.z, b.z * a.x, b.z * a.y, b.z * a.z]
}

// ---- x, y, z polynomials (degree 3 or less) ----
type P1 = [f64; 4]; // x, y, z, 1
type P2 = [f64; 10];
type P3 = [f64; 20];

const M1: [[u8; 3]; 4] = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]];
const M2: [[u8; 3]; 10] = [[2, 0, 0], [1, 1, 0], [0, 2, 0], [1, 0, 1], [0, 1, 1], [0, 0, 2], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]];
/// Column order of A: first 10 (containing x or y) + last 10 [xz², xz, x, yz², yz, y, z³, z², z, 1].
/// The last 6 of the first group form B(z) from the pairs (x²z, x²), (y²z, y²), (xyz, xy).
const M3: [[u8; 3]; 20] = [
    [3, 0, 0],
    [0, 3, 0],
    [2, 1, 0],
    [1, 2, 0],
    [2, 0, 1],
    [2, 0, 0],
    [0, 2, 1],
    [0, 2, 0],
    [1, 1, 1],
    [1, 1, 0],
    [1, 0, 2],
    [1, 0, 1],
    [1, 0, 0],
    [0, 1, 2],
    [0, 1, 1],
    [0, 1, 0],
    [0, 0, 3],
    [0, 0, 2],
    [0, 0, 1],
    [0, 0, 0],
];

const fn find<const N: usize>(list: &[[u8; 3]; N], e: [u8; 3]) -> usize {
    let mut i = 0;
    while i < N {
        if list[i][0] == e[0] && list[i][1] == e[1] && list[i][2] == e[2] {
            return i;
        }
        i += 1;
    }
    usize::MAX
}

const fn table11() -> [[usize; 4]; 4] {
    let mut t = [[0usize; 4]; 4];
    let mut i = 0;
    while i < 4 {
        let mut j = 0;
        while j < 4 {
            t[i][j] = find(&M2, [M1[i][0] + M1[j][0], M1[i][1] + M1[j][1], M1[i][2] + M1[j][2]]);
            j += 1;
        }
        i += 1;
    }
    t
}

const fn table21() -> [[usize; 4]; 10] {
    let mut t = [[0usize; 4]; 10];
    let mut i = 0;
    while i < 10 {
        let mut j = 0;
        while j < 4 {
            t[i][j] = find(&M3, [M2[i][0] + M1[j][0], M2[i][1] + M1[j][1], M2[i][2] + M1[j][2]]);
            j += 1;
        }
        i += 1;
    }
    t
}

const T11: [[usize; 4]; 4] = table11();
const T21: [[usize; 4]; 10] = table21();

fn mul11(a: &P1, b: &P1) -> P2 {
    let mut r = [0.0; 10];
    for i in 0..4 {
        for j in 0..4 {
            r[T11[i][j]] += a[i] * b[j];
        }
    }
    r
}

fn mul21(a: &P2, b: &P1) -> P3 {
    let mut r = [0.0; 20];
    for i in 0..10 {
        for j in 0..4 {
            r[T21[i][j]] += a[i] * b[j];
        }
    }
    r
}

fn add2(a: &mut P2, b: &P2, s: f64) {
    for i in 0..10 {
        a[i] += s * b[i];
    }
}

fn add3(a: &mut P3, b: &P3, s: f64) {
    for i in 0..20 {
        a[i] += s * b[i];
    }
}

/// 5-point (or N >= 5 points) essential solutions (Frobenius norm 1, at most 10).
/// `rays1`, `rays2` are unit rays (image 1, image 2). Constraint: r2ᵀ E r1 = 0.
pub fn essential_five_point(rays1: &[Vec3], rays2: &[Vec3]) -> Vec<Mat3> {
    let n = rays1.len();
    if n < 5 || rays2.len() != n {
        return Vec::new();
    }
    let rows: Vec<[f64; 9]> = rays1.iter().zip(rays2).map(|(a, b)| epipolar_row(a, b)).collect();
    let Some(basis) = null_space_9(&rows, 4) else { return Vec::new() };
    // basis[0] has the smallest singular value. The order X, Y, Z, W does not affect the solution set.
    let (bx, by, bz, bw) = (basis[3], basis[2], basis[1], basis[0]);

    // Degree-1 polynomial per E element.
    let mut e = [[0.0f64; 4]; 9];
    for k in 0..9 {
        e[k] = [bx[k], by[k], bz[k], bw[k]];
    }
    let em = |r: usize, c: usize| -> &P1 { &e[r * 3 + c] };

    // det(E)
    let mut det: P3 = [0.0; 20];
    {
        let mut m0 = mul11(em(1, 1), em(2, 2));
        add2(&mut m0, &mul11(em(1, 2), em(2, 1)), -1.0);
        let mut m1 = mul11(em(1, 0), em(2, 2));
        add2(&mut m1, &mul11(em(1, 2), em(2, 0)), -1.0);
        let mut m2 = mul11(em(1, 0), em(2, 1));
        add2(&mut m2, &mul11(em(1, 1), em(2, 0)), -1.0);
        add3(&mut det, &mul21(&m0, em(0, 0)), 1.0);
        add3(&mut det, &mul21(&m1, em(0, 1)), -1.0);
        add3(&mut det, &mul21(&m2, em(0, 2)), 1.0);
    }
    // E Eᵀ
    let mut eet = [[0.0f64; 10]; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = [0.0; 10];
            for k in 0..3 {
                add2(&mut s, &mul11(em(i, k), em(j, k)), 1.0);
            }
            eet[i * 3 + j] = s;
        }
    }
    let mut tr = [0.0; 10];
    for i in 0..3 {
        add2(&mut tr, &eet[i * 3 + i], 1.0);
    }
    // 2 E Eᵀ E − tr(E Eᵀ) E
    let mut a = SMatrix::<f64, 10, 20>::zeros();
    for c in 0..20 {
        a[(0, c)] = det[c];
    }
    for i in 0..3 {
        for j in 0..3 {
            let mut p: P3 = [0.0; 20];
            for k in 0..3 {
                add3(&mut p, &mul21(&eet[i * 3 + k], em(k, j)), 2.0);
            }
            add3(&mut p, &mul21(&tr, em(i, j)), -1.0);
            for c in 0..20 {
                a[(1 + i * 3 + j, c)] = p[c];
            }
        }
    }
    let af: SMatrix<f64, 10, 10> = a.fixed_view::<10, 10>(0, 0).into_owned();
    let ab: SMatrix<f64, 10, 10> = a.fixed_view::<10, 10>(0, 10).into_owned();
    let Some(g) = af.lu().solve(&ab) else { return Vec::new() };
    if g.iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }

    // B(z): row pairs (4,5), (6,7), (8,9). first + G·last = 0.
    // Each element = ascending z coefficients. x coefficient (cubic), y coefficient (cubic), constant (quartic).
    let mut bmat: [[Vec<f64>; 3]; 3] = Default::default();
    for (r, (ei, fi)) in [(4usize, 5usize), (6, 7), (8, 9)].into_iter().enumerate() {
        let ge = |c: usize| g[(ei, c)];
        let gf = |c: usize| g[(fi, c)];
        for (col, base) in [(0usize, 0usize), (1, 3)] {
            // e: [z², z, 1] at base..base+3; minus z·f
            bmat[r][col] = vec![ge(base + 2), ge(base + 1) - gf(base + 2), ge(base) - gf(base + 1), -gf(base)];
        }
        bmat[r][2] = vec![ge(9), ge(8) - gf(9), ge(7) - gf(8), ge(6) - gf(7), -gf(6)];
    }
    let b = &bmat;
    let cof = |r1: usize, r2: usize, c1: usize, c2: usize| -> Vec<f64> {
        poly_add(&poly_mul(&b[r1][c1], &b[r2][c2]), &poly_mul(&b[r1][c2], &b[r2][c1]), -1.0)
    };
    let d0 = poly_mul(&b[0][0], &cof(1, 2, 1, 2));
    let d1 = poly_mul(&b[0][1], &cof(1, 2, 0, 2));
    let d2 = poly_mul(&b[0][2], &cof(1, 2, 0, 1));
    let detb = poly_add(&poly_add(&d0, &d1, -1.0), &d2, 1.0);

    let mut out = Vec::new();
    for (re, im) in roots_companion(&detb) {
        if im.abs() > 1e-10 {
            continue;
        }
        let z = re;
        let ev = |p: &Vec<f64>| crate::poly::poly_eval(p, z);
        let bmz = Mat3::new(
            ev(&b[0][0]),
            ev(&b[0][1]),
            ev(&b[0][2]),
            ev(&b[1][0]),
            ev(&b[1][1]),
            ev(&b[1][2]),
            ev(&b[2][0]),
            ev(&b[2][1]),
            ev(&b[2][2]),
        );
        let v = smallest_right_singular(&bmz);
        if v.z.abs() < 1e-10 {
            continue;
        }
        let x = v.x / v.z;
        let y = v.y / v.z;
        let mut ev9 = [0.0; 9];
        for k in 0..9 {
            ev9[k] = x * bx[k] + y * by[k] + z * bz[k] + bw[k];
        }
        let m = mat3_from_row_major(&ev9);
        let nrm = m.norm();
        if nrm > 0.0 && nrm.is_finite() {
            out.push(m / nrm);
        }
    }
    out
}

/// Smallest right singular vector of a 3×3 (general SVD).
fn smallest_right_singular(m: &Mat3) -> Vec3 {
    match svd3(m) {
        Some((_, _, v)) => v.column(2).into_owned(),
        None => null_vector3(m),
    }
}

/// 8-point (or more) essential: unit rays without normalization, smallest singular vector, then enforce rank 2
/// (without equalizing the two singular values).
pub fn essential_eight_point(rays1: &[Vec3], rays2: &[Vec3]) -> Option<Mat3> {
    if rays1.len() < 8 || rays1.len() != rays2.len() {
        return None;
    }
    let rows: Vec<[f64; 9]> = rays1.iter().zip(rays2).map(|(a, b)| epipolar_row(a, b)).collect();
    let (_, v) = singular_values_9(&rows)?;
    let e = mat3_from_row_major(&v);
    let (u, s, vv) = svd3(&e)?;
    let r = u * Mat3::from_diagonal(&Vec3::new(s[0], s[1], 0.0)) * vv.transpose();
    r.iter().all(|x| x.is_finite()).then_some(r)
}
