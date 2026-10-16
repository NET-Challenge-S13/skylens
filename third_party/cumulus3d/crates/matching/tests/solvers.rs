/*
 * solvers.rs
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

//! 해법·잔차·분해 검증.

use cumulus3d_core::ransac::{static_max_num_trials, RansacParams};
use cumulus3d_core::{Mat3, Quat, Vec2, Vec3};
use cumulus3d_matching::*;
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

fn rng(s: u64) -> Pcg64 {
    Pcg64::seed_from_u64(s)
}

fn rand_rot(r: &mut Pcg64, max_angle: f64) -> Mat3 {
    let axis = Vec3::new(r.random_range(-1.0..1.0), r.random_range(-1.0..1.0), r.random_range(-1.0..1.0)).normalize();
    Quat::from_axis_angle(&axis, r.random_range(-max_angle..max_angle)).to_rotation_matrix()
}

fn rand_unit(r: &mut Pcg64) -> Vec3 {
    Vec3::new(r.random_range(-1.0..1.0), r.random_range(-1.0..1.0), r.random_range(-1.0..1.0)).normalize()
}

fn skew(v: &Vec3) -> Mat3 {
    Mat3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

fn normalize_sign(m: &Mat3) -> Mat3 {
    let m = m / m.norm();
    // 가장 큰 절댓값 원소를 양수로.
    let mut k = 0;
    for i in 0..9 {
        if m[i].abs() > m[k].abs() {
            k = i;
        }
    }
    if m[k] < 0.0 {
        -m
    } else {
        m
    }
}

fn gauss(r: &mut Pcg64) -> f64 {
    let u1: f64 = r.random_range(1e-12..1.0);
    let u2: f64 = r.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// 점 n 개(깊이 2..10, 카메라1 좌표)와 카메라2 기준 점.
fn scene(r: &mut Pcg64, n: usize, rot: &Mat3, t: &Vec3) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut a = Vec::new();
    let mut b = Vec::new();
    while a.len() < n {
        let x = Vec3::new(r.random_range(-2.0..2.0), r.random_range(-2.0..2.0), r.random_range(2.0..10.0));
        let x2 = rot * x + t;
        if x2.z > 0.1 {
            a.push(x);
            b.push(x2);
        }
    }
    (a, b)
}

#[test]
fn ransac_trial_bounds() {
    let mk = |ratio: f64| RansacParams { min_inlier_ratio: ratio, confidence: 0.999, max_trials: usize::MAX, ..Default::default() };
    assert_eq!(static_max_num_trials(&mk(0.25), 4), 5296);
    assert_eq!(static_max_num_trials(&mk(0.25), 5), 21217);
    assert_eq!(static_max_num_trials(&mk(0.25), 7), 339734);
    assert_eq!(static_max_num_trials(&mk(0.7), 1), 18);
    // 기본 옵션 적용 시 상한 10000.
    let o = TwoViewOptions::default().ransac;
    assert_eq!(static_max_num_trials(&o, 5), 10000);
    assert_eq!(static_max_num_trials(&o, 4), 5296);
}

#[test]
fn sampson_first_order() {
    let mut r = rng(1);
    let k = Mat3::new(1000.0, 0.0, 500.0, 0.0, 1000.0, 400.0, 0.0, 0.0, 1.0);
    let rot = rand_rot(&mut r, 0.2);
    let t = Vec3::new(1.0, 0.2, 0.1);
    let ki = k.try_inverse().unwrap();
    let f = ki.transpose() * skew(&t) * rot * ki;
    let x1 = Vec3::new(420.0, 310.0, 1.0);
    let l = f * x1;
    // 에피폴라선 위의 점 + 법선 방향 d.
    let p0 = Vec2::new(600.0, -(l.x * 600.0 + l.z) / l.y);
    let nrm = Vec2::new(l.x, l.y).normalize();
    for d in [1e-3, 1e-2, 0.1] {
        let p = p0 + nrm * d;
        let e = sampson_error_sq(&f, &x1, &Vec3::new(p.x, p.y, 1.0));
        // Sampson 은 두 영상 오차를 합친 근사 → d² 보다 작거나 같음. 영상2 만 움직였으므로
        // 분모의 영상1 항 비율만큼 작다. 순수 영상2 거리 = 분자/(l_x²+l_y²).
        let lb = f.transpose() * Vec3::new(p.x, p.y, 1.0);
        let pure = e * (lb.x * lb.x + lb.y * lb.y + l.x * l.x + l.y * l.y) / (l.x * l.x + l.y * l.y);
        assert!(((pure - d * d) / (d * d)).abs() < 1e-6, "d={d} pure={pure}");
    }
    // F 가 영행렬이면 분모 0 → ∞.
    assert_eq!(sampson_error_sq(&Mat3::zeros(), &x1, &x1), f64::INFINITY);
}

#[test]
fn five_point_essential() {
    let mut r = rng(2);
    for trial in 0..20 {
        let rot = rand_rot(&mut r, 0.5);
        let t = rand_unit(&mut r);
        let e_true = normalize_sign(&(skew(&t) * rot));
        let (p1, p2) = scene(&mut r, 12, &rot, &t);
        let r1: Vec<Vec3> = p1.iter().map(|p| p.normalize()).collect();
        let r2: Vec<Vec3> = p2.iter().map(|p| p.normalize()).collect();
        let sols = essential_five_point(&r1[..5], &r2[..5]);
        assert!(sols.len() <= 10);
        let best = sols.iter().map(|e| (normalize_sign(e) - e_true).norm()).fold(f64::INFINITY, f64::min);
        assert!(best < 1e-8, "trial {trial}: {best}, n={}", sols.len());
        let sols_n = essential_five_point(&r1, &r2);
        let best = sols_n.iter().map(|e| (normalize_sign(e) - e_true).norm()).fold(f64::INFINITY, f64::min);
        assert!(best < 1e-8, "N점 trial {trial}: {best}");
        // 8점 E 도 같은 E.
        let e8 = essential_eight_point(&r1, &r2).unwrap();
        let d8 = (normalize_sign(&e8) - e_true).norm();
        assert!(d8 < 1e-8, "8점 {d8}");
    }
}

fn synth_f(r: &mut Pcg64) -> (Mat3, Mat3, Mat3, Vec3) {
    let k = Mat3::new(1000.0, 0.0, 1000.0, 0.0, 1000.0, 750.0, 0.0, 0.0, 1.0);
    let rot = rand_rot(r, 0.3);
    let t = rand_unit(r);
    let ki = k.try_inverse().unwrap();
    (ki.transpose() * skew(&t) * rot * ki, k, rot, t)
}

fn project(k: &Mat3, x: &Vec3) -> Vec2 {
    let p = k * x;
    Vec2::new(p.x / p.z, p.y / p.z)
}

#[test]
fn seven_and_eight_point_fundamental() {
    let mut r = rng(3);
    for _ in 0..20 {
        let (f_true, k, rot, t) = synth_f(&mut r);
        let f_true = normalize_sign(&f_true);
        let (a, b) = scene(&mut r, 50, &rot, &t);
        let x1: Vec<Vec2> = a.iter().map(|p| project(&k, p)).collect();
        let x2: Vec<Vec2> = b.iter().map(|p| project(&k, p)).collect();
        let sols = fundamental_seven_point(&x1[..7], &x2[..7]);
        assert!(!sols.is_empty() && sols.len() <= 3);
        let (best, fb) = sols.iter().map(|f| ((normalize_sign(f) - f_true).norm(), *f)).fold((f64::INFINITY, Mat3::zeros()), |acc, v| {
            if v.0 < acc.0 {
                v
            } else {
                acc
            }
        });
        assert!(best < 1e-6, "{best}");
        let sv = fb.singular_values();
        let (mx, mn) = (sv.max(), sv.min());
        assert!(mn / mx < 1e-10);

        // 잡음 0.5 px 8점 이상.
        let n1: Vec<Vec2> = x1.iter().map(|p| p + Vec2::new(0.5 * gauss(&mut r), 0.5 * gauss(&mut r))).collect();
        let n2: Vec<Vec2> = x2.iter().map(|p| p + Vec2::new(0.5 * gauss(&mut r), 0.5 * gauss(&mut r))).collect();
        let f8 = fundamental_eight_point(&n1, &n2).unwrap();
        let mean: f64 =
            n1.iter().zip(&n2).map(|(a, b)| sampson_error_sq(&f8, &Vec3::new(a.x, a.y, 1.0), &Vec3::new(b.x, b.y, 1.0))).sum::<f64>()
                / n1.len() as f64;
        assert!(mean < 1.0, "평균 Sampson {mean}");
    }
}

#[test]
fn homography_dlt_exact() {
    let mut r = rng(4);
    let mut n_checked = 0;
    for _ in 0..40 {
        let k = Mat3::new(1000.0, 0.0, 1000.0, 0.0, 1000.0, 750.0, 0.0, 0.0, 1.0);
        let rot = rand_rot(&mut r, 0.3);
        let t = Vec3::new(r.random_range(-0.2..0.2), r.random_range(-0.2..0.2), r.random_range(-0.2..0.2));
        let n = Vec3::new(r.random_range(-0.3..0.3), r.random_range(-0.3..0.3), 1.0).normalize();
        let d = 5.0;
        let h_true = k * (rot - t * n.transpose() / d) * k.try_inverse().unwrap();
        let h_true = h_true / h_true[(2, 2)];
        let x1: Vec<Vec2> = (0..4).map(|_| Vec2::new(r.random_range(0.0..2000.0), r.random_range(0.0..1500.0))).collect();
        let x2: Vec<Vec2> = x1
            .iter()
            .map(|p| {
                let q = h_true * Vec3::new(p.x, p.y, 1.0);
                Vec2::new(q.x / q.z, q.y / q.z)
            })
            .collect();
        for normalize in [false, true] {
            let h = homography_dlt(&x1, &x2, normalize).unwrap();
            let h = h / h[(2, 2)];
            assert!((h - h_true).norm() / h_true.norm() < 1e-8);
        }
        // N점(잡음 없음)
        let x1n: Vec<Vec2> = (0..30).map(|_| Vec2::new(r.random_range(0.0..2000.0), r.random_range(0.0..1500.0))).collect();
        let x2n: Vec<Vec2> = x1n
            .iter()
            .map(|p| {
                let q = h_true * Vec3::new(p.x, p.y, 1.0);
                Vec2::new(q.x / q.z, q.y / q.z)
            })
            .collect();
        // det 스케일 규칙: N점 해는 단위 노름이므로 det(H/‖H‖) 가 1e-8 미만이면 거부된다.
        let det_unit = (h_true / h_true.norm()).determinant().abs();
        for normalize in [false, true] {
            let res = homography_dlt(&x1n, &x2n, normalize);
            if det_unit > 2e-8 {
                let h = res.unwrap();
                let h = h / h[(2, 2)];
                assert!((h - h_true).norm() / h_true.norm() < 1e-7);
                n_checked += 1;
            } else if det_unit < 5e-9 {
                assert!(res.is_none());
            }
        }
    }
    assert!(n_checked > 0);
    // 퇴화: 한 점에 몰린 대응 → det 작음/특이 → None.
    let p = vec![Vec2::new(1.0, 1.0); 4];
    assert!(homography_dlt(&p, &p, false).is_none());
    // 크기 1e-3 스케일 축소: det = 1e-9 < 1e-8 → None.
    let x1 = vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0)];
    let x2: Vec<Vec2> = x1.iter().map(|p| p * 1e-3 + Vec2::new(0.0, 0.0)).collect();
    // H = diag(1e-3,1e-3,1) → det 1e-6 (통과), diag(1e-5,1e-5,1) → 1e-10 (거부)
    assert!(homography_dlt(&x1, &x2, false).is_some());
    let x3: Vec<Vec2> = x1.iter().map(|p| p * 1e-5).collect();
    assert!(homography_dlt(&x1, &x3, false).is_none());
    // 전이 오차의 NaN/∞ 은 ∞.
    assert_eq!(homography_transfer_error_sq(&Mat3::zeros(), &Vec2::new(1.0, 1.0), &Vec2::new(1.0, 1.0)), f64::INFINITY);
}

#[test]
fn essential_decomposition_cheirality() {
    let mut r = rng(5);
    for _ in 0..20 {
        let rot = rand_rot(&mut r, 0.5);
        let t = rand_unit(&mut r);
        let e = skew(&t) * rot;
        let (p1, p2) = scene(&mut r, 30, &rot, &t);
        let r1: Vec<Vec3> = p1.iter().map(|p| p.normalize()).collect();
        let r2: Vec<Vec3> = p2.iter().map(|p| p.normalize()).collect();
        let (pose, pts) = pose_from_essential(&(e * 3.7), &r1, &r2).unwrap();
        assert_eq!(pts.len(), 30);
        let dr = Quat::from_rotation_matrix(&rot).angular_distance(&pose.rotation);
        assert!(dr < 1e-6, "rot err {dr}");
        let ang = pose.translation.normalize().dot(&t).clamp(-1.0, 1.0).acos();
        assert!(ang < 1e-6, "t err {ang}");
        // 4후보 중 정답만 모든 점 통과.
        let cands = decompose_essential(&e).unwrap();
        let full: Vec<usize> =
            cands.iter().map(|(rr, tt)| r1.iter().zip(&r2).filter(|(a, b)| triangulate_midpoint(rr, tt, a, b).is_some()).count()).collect();
        assert_eq!(full.iter().filter(|&&c| c == 30).count(), 1, "{full:?}");
    }
}

#[test]
fn homography_decomposition() {
    let mut r = rng(6);
    let k1 = Mat3::new(1000.0, 0.0, 1000.0, 0.0, 1000.0, 750.0, 0.0, 0.0, 1.0);
    let k2 = Mat3::new(1100.0, 0.0, 950.0, 0.0, 1100.0, 760.0, 0.0, 0.0, 1.0);
    for _ in 0..20 {
        let rot = rand_rot(&mut r, 0.3);
        let t = Vec3::new(r.random_range(-1.0..1.0), r.random_range(-1.0..1.0), r.random_range(-0.3..0.3));
        let n = Vec3::new(r.random_range(-0.3..0.3), r.random_range(-0.3..0.3), 1.0).normalize();
        let d = 4.0;
        let h = k2 * (rot - t * n.transpose() / d) * k1.try_inverse().unwrap() * 2.5;
        let cands = decompose_homography(&h, &k1, &k2);
        assert_eq!(cands.len(), 4);
        let td = t / d;
        let ok = cands.iter().any(|(rr, tt, nn)| {
            let dr = Quat::from_rotation_matrix(&rot).angular_distance(&Quat::from_rotation_matrix(rr));
            dr < 1e-6 && (tt - td).norm() < 1e-6 * (1.0 + td.norm()) && (nn - n).norm() < 1e-6
        });
        assert!(ok, "{cands:?} vs R t/d={td:?} n={n:?}");
    }
    // 순수 회전 → 후보 1개, t = 0.
    let rot = rand_rot(&mut r, 0.3);
    let h = k2 * rot * k1.try_inverse().unwrap();
    let c = decompose_homography(&h, &k1, &k2);
    assert_eq!(c.len(), 1);
    assert!(c[0].1.norm() == 0.0);
    assert!((c[0].0 - rot).norm() < 1e-9);
}
