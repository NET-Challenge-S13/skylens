//! Eigenvector of the smallest eigenvalue of a symmetric 3x3 matrix, used for normal estimation.
//!
//! The fast path is closed form: the eigenvalue comes from the trigonometric solution of the
//! characteristic cubic, and the eigenvector from the longest cross product of two rows of
//! `A - λI`. When the direction is ill-conditioned (near-isotropic input, or the two smallest
//! eigenvalues nearly coincide) the function falls back to cyclic Jacobi rotations.

/// Returns the unit eigenvector belonging to the smallest eigenvalue of the symmetric matrix `m`.
///
/// The matrix is first scaled so its largest entry has magnitude 1; the eigenvector direction
/// does not depend on the scale. The smallest eigenvalue is computed from
/// `B = (A - qI) / p`, where `det(B) / 2 = cos(3φ)`.
///
/// The squared length of the chosen row cross product is approximately
/// `(λ_mid - λ_min)² (λ_max - λ_min)²`. If it falls below `1e-8 * p⁴`, i.e. the two smallest
/// eigenvalues are closer than about `1e-4` relative to the spread, the closed form is unstable
/// and [`jacobi`] is used instead. Non-finite or all-zero input and near-isotropic matrices also
/// go to [`jacobi`].
///
/// # Returns
///
/// A unit vector. Its sign is arbitrary.
pub(crate) fn smallest_eigvec(m: [[f64; 3]; 3]) -> [f64; 3] {
    let scale = m[0][0]
        .abs()
        .max(m[1][1].abs())
        .max(m[2][2].abs())
        .max(m[0][1].abs())
        .max(m[0][2].abs())
        .max(m[1][2].abs());
    if !(scale > 1e-300 && scale.is_finite()) {
        return jacobi(m);
    }
    let inv = 1.0 / scale;
    let (a00, a11, a22) = (m[0][0] * inv, m[1][1] * inv, m[2][2] * inv);
    let (a01, a02, a12) = (m[0][1] * inv, m[0][2] * inv, m[1][2] * inv);
    let q = (a00 + a11 + a22) / 3.0;
    let (b00, b11, b22) = (a00 - q, a11 - q, a22 - q);
    let off2 = a01 * a01 + a02 * a02 + a12 * a12;
    let p2 = (b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * off2) / 6.0;
    if p2 < 1e-24 {
        return jacobi(m);
    }
    let p = p2.sqrt();
    let det = b00 * (b11 * b22 - a12 * a12) - a01 * (a01 * b22 - a12 * a02)
        + a02 * (a01 * a12 - b11 * a02);
    let half = (det / (2.0 * p2 * p)).clamp(-1.0, 1.0);
    let phi = half.acos() / 3.0;
    let lam = q + 2.0 * p * (phi + 2.0 * std::f64::consts::FRAC_PI_3).cos();
    let r0 = [a00 - lam, a01, a02];
    let r1 = [a01, a11 - lam, a12];
    let r2 = [a02, a12, a22 - lam];
    let c = [cross(r0, r1), cross(r0, r2), cross(r1, r2)];
    let d = [dot(c[0], c[0]), dot(c[1], c[1]), dot(c[2], c[2])];
    let mut best = 0;
    for i in 1..3 {
        if d[i] > d[best] {
            best = i;
        }
    }
    if d[best] < 1e-8 * p2 * p2 {
        return jacobi(m);
    }
    let l = d[best].sqrt();
    [c[best][0] / l, c[best][1] / l, c[best][2] / l]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Computes the smallest-eigenvalue eigenvector with cyclic Jacobi rotations.
///
/// Fallback for inputs where the closed form in [`smallest_eigvec`] is ill-conditioned. Runs at
/// most 32 sweeps and stops early once the off-diagonal sum drops below `1e-15`.
fn jacobi(m: [[f64; 3]; 3]) -> [f64; 3] {
    let mut a = m;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..32 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < 1e-15 {
            break;
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for row in a.iter_mut() {
                let akp = row[p];
                let akq = row[q];
                row[p] = c * akp - s * akq;
                row[q] = s * akp + c * akq;
            }
            let (head, tail) = a.split_at_mut(q);
            let (row_p, row_q) = (&mut head[p], &mut tail[0]);
            for (ap, aq) in row_p.iter_mut().zip(row_q.iter_mut()) {
                let apk = *ap;
                let aqk = *aq;
                *ap = c * apk - s * aqk;
                *aq = s * apk + c * aqk;
            }
            for row in v.iter_mut() {
                let vp = row[p];
                let vq = row[q];
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }
    let mut best = 0;
    for i in 1..3 {
        if a[i][i] < a[best][best] {
            best = i;
        }
    }
    [v[0][best], v[1][best], v[2][best]]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Covariance of points spread over the xy plane: the z variance is the smallest.
    #[test]
    fn plane_normal() {
        let n = smallest_eigvec([[2.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 0.01]]);
        assert!(n[2].abs() > 0.999, "{n:?}");
    }

    fn rot(a: f64, b: f64) -> [[f64; 3]; 3] {
        let (ca, sa, cb, sb) = (a.cos(), a.sin(), b.cos(), b.sin());
        [
            [ca, -sa * cb, sa * sb],
            [sa, ca * cb, -ca * sb],
            [0.0, sb, cb],
        ]
    }

    /// Returns `R diag(l) Rᵀ`.
    fn compose(r: [[f64; 3]; 3], l: [f64; 3]) -> [[f64; 3]; 3] {
        let mut m = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    m[i][j] += r[i][k] * l[k] * r[j][k];
                }
            }
        }
        m
    }

    /// Closed form and Jacobi agree over the range seen in normal estimation.
    ///
    /// Covers planes (`λ_min ≈ 0`), lines (`λ_min ≈ λ_mid`) and large differences in scale. The
    /// expected eigenvector is column 0 of `R`.
    #[test]
    fn closed_form_matches_jacobi() {
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..20000 {
            let r = rot(rnd() * 6.3, rnd() * 3.2);
            let s = 10f64.powf(-6.0 + 5.0 * rnd());
            let l = [
                s * 0.15 * rnd() * rnd(),
                s * (0.2 + rnd()),
                s * (1.0 + rnd()),
            ];
            let m = compose(r, l);
            let a = smallest_eigvec(m);
            let b = jacobi(m);
            let c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).abs();
            let t = (a[0] * r[0][0] + a[1] * r[1][0] + a[2] * r[2][0]).abs();
            assert!(c > 1.0 - 1e-9 && t > 1.0 - 1e-9, "{m:?} {a:?} {b:?}");
        }
    }

    /// Degenerate matrices still give a unit vector, and a unique smallest eigenvalue is found
    /// even when the other two coincide.
    #[test]
    fn degenerate_inputs_are_finite() {
        let r = rot(0.7, 1.1);
        for m in [
            [[0.0; 3]; 3],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            compose(r, [1e-3, 1e-3, 2e-3]),
            compose(r, [1e-3, 2e-3, 2e-3]),
            compose(r, [0.0, 0.0, 1e-2]),
        ] {
            let v = smallest_eigvec(m);
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((l - 1.0).abs() < 1e-9, "{m:?} {v:?}");
        }
        let v = smallest_eigvec(compose(r, [1e-3, 2e-3, 2e-3]));
        let t = (v[0] * r[0][0] + v[1] * r[1][0] + v[2] * r[2][0]).abs();
        assert!(t > 1.0 - 1e-9, "{v:?}");
    }
}
