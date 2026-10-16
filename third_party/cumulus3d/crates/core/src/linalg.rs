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

//! 작은 고정 크기 선형대수. nalgebra 0.35 의 3×3 SVD 는 특이값이 겹칠 때 틀린 값을 내므로
//! 단측 야코비 구현을 여기 둔다(matching·sfm·align 공용).

use crate::geometry::{Mat3, Vec3};

/// 3×3 SVD (U, σ 내림차순, V), M = U diag(σ) Vᵀ, det(U) = +1.
/// 단측 야코비 회전으로 열을 직교화한다(중복 특이값에서도 정확).
pub fn svd3(m: &Mat3) -> Option<(Mat3, Vec3, Mat3)> {
    if !m.iter().all(|x| x.is_finite()) {
        return None;
    }
    let mut a = *m;
    let mut v = Mat3::identity();
    for _sweep in 0..30 {
        let mut off = 0.0f64;
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            let ap = a.column(p).into_owned();
            let aq = a.column(q).into_owned();
            let alpha = ap.norm_squared();
            let beta = aq.norm_squared();
            let gamma = ap.dot(&aq);
            if gamma == 0.0 {
                continue;
            }
            let rel = gamma.abs() / (alpha * beta).sqrt();
            off = off.max(rel);
            if rel < 1e-15 {
                continue;
            }
            let zeta = (beta - alpha) / (2.0 * gamma);
            let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
            let t = if zeta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = c * t;
            for k in 0..3 {
                let x = a[(k, p)];
                let y = a[(k, q)];
                a[(k, p)] = c * x - s * y;
                a[(k, q)] = s * x + c * y;
                let x = v[(k, p)];
                let y = v[(k, q)];
                v[(k, p)] = c * x - s * y;
                v[(k, q)] = s * x + c * y;
            }
        }
        if off < 1e-15 {
            break;
        }
    }
    let norms = [a.column(0).norm(), a.column(1).norm(), a.column(2).norm()];
    let mut order = [0usize, 1, 2];
    order.sort_by(|&x, &y| norms[y].total_cmp(&norms[x]));
    let s = Vec3::new(norms[order[0]], norms[order[1]], norms[order[2]]);
    let mut vs = Mat3::zeros();
    let mut cols = [Vec3::zeros(); 3];
    for (c, &o) in order.iter().enumerate() {
        vs.set_column(c, &v.column(o));
        cols[c] = a.column(o).into_owned();
    }
    if s[0] == 0.0 {
        return Some((Mat3::identity(), s, vs));
    }
    let u1 = cols[0] / s[0];
    let u2 = if s[1] > s[0] * 1e-300 && s[1] > 0.0 { cols[1] / s[1] } else { any_orthogonal(&u1) };
    let u2 = (u2 - u1 * u1.dot(&u2)).normalize();
    let u3 = u1.cross(&u2);
    // σ3 의 부호: M v3 = σ3 u3 가 되도록(음이면 v3 뒤집음).
    if s[2] > 0.0 && cols[2].dot(&u3) < 0.0 {
        let c = -vs.column(2);
        vs.set_column(2, &c);
    }
    Some((Mat3::from_columns(&[u1, u2, u3]), s, vs))
}

fn any_orthogonal(v: &Vec3) -> Vec3 {
    let a = if v.x.abs() < 0.9 { Vec3::x() } else { Vec3::y() };
    v.cross(&a).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(m: &Mat3) {
        let (u, s, v) = svd3(m).unwrap();
        let r = u * Mat3::from_diagonal(&s) * v.transpose();
        assert!((r - m).norm() < 1e-12 * (1.0 + m.norm()), "{r} vs {m}");
        assert!((u.transpose() * u - Mat3::identity()).norm() < 1e-12);
        assert!((v.transpose() * v - Mat3::identity()).norm() < 1e-12);
        assert!((u.determinant() - 1.0).abs() < 1e-12);
        assert!(s[0] >= s[1] && s[1] >= s[2] && s[2] >= 0.0);
    }

    #[test]
    fn svd3_repeated_singular_values() {
        let r = crate::geometry::Quat::from_axis_angle(&Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7).to_rotation_matrix();
        let r2 = crate::geometry::Quat::from_axis_angle(&Vec3::new(-2.0, 0.5, 1.0).normalize(), 1.9).to_rotation_matrix();
        check(&Mat3::identity());
        check(&r);
        check(&(r * Mat3::from_diagonal(&Vec3::new(2.0, 2.0, 1.0)) * r2));
        check(&(r * Mat3::from_diagonal(&Vec3::new(3.0, 1.0, 1.0)) * r2));
        check(&(r * Mat3::from_diagonal(&Vec3::new(1.0, 1.0, 0.0)) * r2));
        check(&(r * Mat3::from_diagonal(&Vec3::new(1.0, -1.0, 1.0)) * r2));
        check(&Mat3::zeros());
    }
}
