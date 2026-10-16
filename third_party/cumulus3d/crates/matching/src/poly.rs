/*
 * poly.rs
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

//! Polynomial roots.

use nalgebra::DMatrix;

/// Complex roots (real part, imaginary part) of a polynomial given by coefficients (ascending: `c[0] + c[1] z + …`).
/// Eigenvalues of the companion matrix normalized by the leading coefficient. If the leading coefficient is 0, the degree is lowered.
pub fn roots_companion(coeffs: &[f64]) -> Vec<(f64, f64)> {
    let mut deg = coeffs.len().saturating_sub(1);
    while deg > 0 && coeffs[deg] == 0.0 {
        deg -= 1;
    }
    if deg == 0 {
        return Vec::new();
    }
    let lead = coeffs[deg];
    if !lead.is_finite() || coeffs[..deg].iter().any(|c| !c.is_finite()) {
        return Vec::new();
    }
    if deg == 1 {
        return vec![(-coeffs[0] / lead, 0.0)];
    }
    let mut c = DMatrix::<f64>::zeros(deg, deg);
    for i in 0..deg {
        c[(0, i)] = -coeffs[deg - 1 - i] / lead;
    }
    for i in 1..deg {
        c[(i, i - 1)] = 1.0;
    }
    match c.try_schur(f64::EPSILON, 1000) {
        Some(s) => s.complex_eigenvalues().iter().map(|z| (z.re, z.im)).collect(),
        None => Vec::new(),
    }
}

/// Real roots (1 or 3) of the monic cubic x³ + a x² + b x + c. Analytic solution.
pub fn solve_cubic_monic(a: f64, b: f64, c: f64) -> Vec<f64> {
    let a3 = a / 3.0;
    let p = b - a * a3;
    let q = 2.0 * a3 * a3 * a3 - a3 * b + c;
    // t³ + p t + q = 0, x = t − a/3
    let disc = (q / 2.0) * (q / 2.0) + (p / 3.0) * (p / 3.0) * (p / 3.0);
    if disc > 0.0 {
        let sq = disc.sqrt();
        let u = (-q / 2.0 + sq).cbrt();
        let v = (-q / 2.0 - sq).cbrt();
        vec![u + v - a3]
    } else if p == 0.0 {
        vec![-a3]
    } else {
        // Three real roots (including a repeated root): trigonometric solution.
        let m = 2.0 * (-p / 3.0).sqrt();
        let arg = (3.0 * q / (p * m)).clamp(-1.0, 1.0);
        let theta = arg.acos() / 3.0;
        let tau = 2.0 * std::f64::consts::PI / 3.0;
        vec![m * theta.cos() - a3, m * (theta - tau).cos() - a3, m * (theta - 2.0 * tau).cos() - a3]
    }
}

/// Polynomial product (ascending coefficients).
pub fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut r = vec![0.0; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            r[i + j] += x * y;
        }
    }
    r
}

/// Polynomial sum/difference.
pub fn poly_add(a: &[f64], b: &[f64], sign: f64) -> Vec<f64> {
    let mut r = vec![0.0; a.len().max(b.len())];
    for (i, x) in a.iter().enumerate() {
        r[i] += x;
    }
    for (i, y) in b.iter().enumerate() {
        r[i] += sign * y;
    }
    r
}

/// Polynomial value.
pub fn poly_eval(c: &[f64], z: f64) -> f64 {
    c.iter().rev().fold(0.0, |acc, &v| acc * z + v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_and_companion() {
        // (x-1)(x-2)(x+3) = x³ − 7x + 6
        let mut r = solve_cubic_monic(0.0, -7.0, 6.0);
        r.sort_by(f64::total_cmp);
        assert!((r[0] + 3.0).abs() < 1e-12 && (r[1] - 1.0).abs() < 1e-12 && (r[2] - 2.0).abs() < 1e-12);
        // x³ + x + 1: one real root
        let r = solve_cubic_monic(0.0, 1.0, 1.0);
        assert_eq!(r.len(), 1);
        assert!(poly_eval(&[1.0, 1.0, 0.0, 1.0], r[0]).abs() < 1e-12);
        let roots = roots_companion(&[6.0, -7.0, 0.0, 1.0]);
        let mut re: Vec<f64> = roots.iter().map(|z| z.0).collect();
        re.sort_by(f64::total_cmp);
        assert!((re[0] + 3.0).abs() < 1e-10 && (re[2] - 2.0).abs() < 1e-10);
    }
}
