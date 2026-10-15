/*
 * two_view.rs
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

//! 두 뷰 기하 통합 검증.

use cumulus3d_core::{Camera, CameraModelKind, FeatureMatch, Keypoint, Mat3, Quat, TwoViewGeometryConfig as C, Vec2, Vec3};
use cumulus3d_matching::*;
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

const W: u64 = 2000;
const H: u64 = 1500;

fn camera(prior: bool) -> Camera {
    let mut c = Camera::new(1, CameraModelKind::OpenCv, W, H, vec![1000.0, 1010.0, 1000.0, 750.0, -0.02, 0.003, 0.0005, -0.0003]).unwrap();
    c.focal_from_prior = prior;
    c
}

fn gauss(r: &mut Pcg64) -> f64 {
    let u1: f64 = r.random_range(1e-12..1.0);
    let u2: f64 = r.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn in_image(p: &Vec2) -> bool {
    p.x > 0.0 && p.y > 0.0 && p.x < W as f64 && p.y < H as f64
}

/// 장면 종류.
enum Scene {
    General,
    Planar,
    Rotation,
}

struct Data {
    kps1: Vec<Keypoint>,
    kps2: Vec<Keypoint>,
    matches: Vec<FeatureMatch>,
    truth: Vec<bool>,
}

fn make(seed: u64, scene: Scene, n_in: usize, n_out: usize, noise: f64) -> (Data, Mat3, Vec3) {
    let mut r = Pcg64::seed_from_u64(seed);
    let cam = camera(true);
    let rot = Quat::from_axis_angle(&Vec3::new(0.1, 1.0, 0.05).normalize(), 0.08).to_rotation_matrix();
    let t = match scene {
        Scene::Rotation => Vec3::zeros(),
        _ => Vec3::new(-1.0, 0.1, 0.15),
    };
    let mut d = Data { kps1: vec![], kps2: vec![], matches: vec![], truth: vec![] };
    let mut items: Vec<(Vec2, Vec2, bool)> = Vec::new();
    while items.iter().filter(|x| x.2).count() < n_in {
        let x = match scene {
            Scene::Planar => {
                let (u, v) = (r.random_range(-6.0..6.0), r.random_range(-5.0..5.0));
                Vec3::new(u, v, 10.0 + 0.2 * u + 0.1 * v)
            }
            _ => Vec3::new(r.random_range(-6.0..6.0), r.random_range(-5.0..5.0), r.random_range(5.0..15.0)),
        };
        let (Some(p1), Some(p2)) = (cam.cam_to_img(&x), cam.cam_to_img(&(rot * x + t))) else { continue };
        if !in_image(&p1) || !in_image(&p2) {
            continue;
        }
        let n1 = p1 + Vec2::new(noise * gauss(&mut r), noise * gauss(&mut r));
        let n2 = p2 + Vec2::new(noise * gauss(&mut r), noise * gauss(&mut r));
        items.push((n1, n2, true));
    }
    for _ in 0..n_out {
        let p1 = Vec2::new(r.random_range(0.0..W as f64), r.random_range(0.0..H as f64));
        let p2 = Vec2::new(r.random_range(0.0..W as f64), r.random_range(0.0..H as f64));
        items.push((p1, p2, false));
    }
    // 섞기
    for i in (1..items.len()).rev() {
        let j = r.random_range(0..=i);
        items.swap(i, j);
    }
    for (i, (a, b, tr)) in items.into_iter().enumerate() {
        d.kps1.push(Keypoint::new(a.x as f32, a.y as f32));
        d.kps2.push(Keypoint::new(b.x as f32, b.y as f32));
        d.matches.push(FeatureMatch::new(i as u32, i as u32));
        d.truth.push(tr);
    }
    (d, rot, t)
}

fn opts() -> TwoViewOptions {
    let mut o = TwoViewOptions::default();
    o.ransac.random_seed = Some(7);
    o
}

#[test]
fn calibrated_with_outliers() {
    let (d, rot, t) = make(1, Scene::General, 600, 400, 0.5);
    let cam = camera(true);
    let g = estimate_two_view(&cam, &d.kps1, &cam, &d.kps2, &d.matches, &opts());
    assert_eq!(g.config, C::Calibrated);
    assert!(g.e.is_some() && g.f.is_some());
    let inl: std::collections::HashSet<u32> = g.inlier_matches.iter().map(|m| m.idx1).collect();
    let tp = d.truth.iter().enumerate().filter(|(i, t)| **t && inl.contains(&(*i as u32))).count();
    let recall = tp as f64 / 600.0;
    let precision = tp as f64 / inl.len() as f64;
    assert!(recall > 0.95 && precision > 0.98, "recall {recall} precision {precision}");

    // 상대 자세(옵션) 복원.
    let mut o = opts();
    o.compute_relative_pose = true;
    let g = estimate_two_view(&cam, &d.kps1, &cam, &d.kps2, &d.matches, &o);
    let p = g.cam1_to_cam2.expect("자세");
    assert!(Quat::from_rotation_matrix(&rot).angular_distance(&p.rotation) < 0.01);
    assert!(p.translation.normalize().dot(&t.normalize()) > 0.999);
    assert!(g.tri_angle.unwrap() > 0.0);
    // 일괄 재맞춤 경로(행렬 없이).
    let mut g2 = g.clone();
    g2.e = None;
    g2.cam1_to_cam2 = None;
    assert!(refit_and_estimate_relative_pose(&cam, &cam, &d.kps1, &d.kps2, &mut g2));
    let p2 = g2.cam1_to_cam2.unwrap();
    assert!((p2.translation.norm() - 1.0).abs() < 1e-9);
    assert!(Quat::from_rotation_matrix(&rot).angular_distance(&p2.rotation) < 0.01);
}

#[test]
fn uncalibrated_path() {
    let (d, _, _) = make(2, Scene::General, 300, 100, 0.5);
    let c1 = camera(true);
    let c2 = camera(false);
    let g = estimate_two_view(&c1, &d.kps1, &c2, &d.kps2, &d.matches, &opts());
    assert_eq!(g.config, C::Uncalibrated);
    assert!(g.e.is_none());
    assert!(g.inlier_matches.len() >= 285);
    // UNCALIBRATED 자세 분해(E = K2ᵀ F K1, 왜곡 무시 → 대략적).
    let mut g2 = g.clone();
    assert!(recover_two_view_pose(&c1, &c2, &d.kps1, &d.kps2, &mut g2));
}

#[test]
fn planar_and_panoramic() {
    let cam = camera(true);
    let (d, _, _) = make(3, Scene::Planar, 400, 50, 0.5);
    let g = estimate_two_view(&cam, &d.kps1, &cam, &d.kps2, &d.matches, &opts());
    assert_eq!(g.config, C::PlanarOrRotation);
    let mut o = opts();
    o.compute_relative_pose = true;
    let g = estimate_two_view(&cam, &d.kps1, &cam, &d.kps2, &d.matches, &o);
    // 왜곡이 있는 카메라라 H 는 근사 → PLANAR 확정만 확인.
    assert_eq!(g.config, C::Planar);

    let (d, _, _) = make(4, Scene::Rotation, 400, 50, 0.3);
    let g = estimate_two_view(&cam, &d.kps1, &cam, &d.kps2, &d.matches, &opts());
    assert_eq!(g.config, C::PlanarOrRotation);
}

#[test]
fn panoramic_pinhole_pose() {
    // 왜곡 없는 카메라, 잡음 없는 순수 회전 → 분해 시 PANORAMIC(‖t‖² < 1e-12).
    let mut cam = Camera::new(1, CameraModelKind::Pinhole, W, H, vec![1000.0, 1000.0, 1000.0, 750.0]).unwrap();
    cam.focal_from_prior = true;
    let rot = Quat::from_axis_angle(&Vec3::new(0.2, 1.0, 0.0).normalize(), 0.1).to_rotation_matrix();
    let mut r = Pcg64::seed_from_u64(9);
    let (mut k1, mut k2, mut m) = (vec![], vec![], vec![]);
    while k1.len() < 200 {
        let x = Vec3::new(r.random_range(-6.0..6.0), r.random_range(-5.0..5.0), r.random_range(5.0..15.0));
        let (Some(a), Some(b)) = (cam.cam_to_img(&x), cam.cam_to_img(&(rot * x))) else { continue };
        if !in_image(&a) || !in_image(&b) {
            continue;
        }
        m.push(FeatureMatch::new(k1.len() as u32, k1.len() as u32));
        k1.push(Keypoint::new(a.x as f32, a.y as f32));
        k2.push(Keypoint::new(b.x as f32, b.y as f32));
    }
    let mut o = opts();
    o.compute_relative_pose = true;
    let g = estimate_two_view(&cam, &k1, &cam, &k2, &m, &o);
    assert_eq!(g.config, C::Panoramic, "{:?}", g.cam1_to_cam2);
    assert_eq!(g.tri_angle, Some(0.0));
}

#[test]
fn degenerate_and_finalize() {
    let (d, _, _) = make(5, Scene::General, 14, 0, 0.1);
    let cam = camera(true);
    let g = estimate_two_view(&cam, &d.kps1, &cam, &d.kps2, &d.matches, &opts());
    assert_eq!(g.config, C::Degenerate);
    let f = finalize_geometry(g, 15);
    assert_eq!(f.config, C::Undefined);
    assert!(f.inlier_matches.is_empty());
}

#[test]
fn decision_boundaries() {
    let o = TwoViewOptions::default();
    let m = |n: usize| ModelOutcome { success: true, num_inliers: n };
    // r_EF = 0.95 정확히 → (가) 거부 → F 경로. r_HF = 0.5 → UNCALIBRATED.
    let d = decide_calibrated(m(95), m(100), m(50), &o);
    assert_eq!(d, Decision { config: C::Uncalibrated, mask: Some(MaskChoice::F) });
    // r_EF 살짝 큼 → CALIBRATED. r_HE = 0.8 정확히 → CALIBRATED 유지.
    let d = decide_calibrated(m(100), m(100), m(80), &o);
    assert_eq!(d, Decision { config: C::Calibrated, mask: Some(MaskChoice::E) });
    // n_E < n_F → F 마스크.
    let d = decide_calibrated(m(99), m(100), m(10), &o);
    assert_eq!(d, Decision { config: C::Calibrated, mask: Some(MaskChoice::F) });
    // 보정 경로에서 n_H = 선택 수 → 교체 안 함, PoP.
    let d = decide_calibrated(m(100), m(100), m(100), &o);
    assert_eq!(d, Decision { config: C::PlanarOrRotation, mask: Some(MaskChoice::E) });
    let d = decide_calibrated(m(100), m(100), m(101), &o);
    assert_eq!(d.mask, Some(MaskChoice::H));
    // 비보정 경로에서 n_H = n_F → H 마스크.
    let d = decide_uncalibrated(m(100), m(100), &o);
    assert_eq!(d, Decision { config: C::PlanarOrRotation, mask: Some(MaskChoice::H) });
    // F 실패 + H 성공 → PoP(H).
    let fail = ModelOutcome { success: false, num_inliers: 0 };
    assert_eq!(decide_uncalibrated(fail, m(30), &o).mask, Some(MaskChoice::H));
    // 모두 15 미만 → DEGENERATE.
    assert_eq!(decide_uncalibrated(m(14), m(14), &o).config, C::Degenerate);
    assert_eq!(decide_calibrated(m(14), m(14), m(14), &o).config, C::Degenerate);
    // E 만 실패, H 만 15 이상 → (다).
    assert_eq!(decide_calibrated(fail, m(10), m(20), &o), Decision { config: C::PlanarOrRotation, mask: Some(MaskChoice::H) });
}

#[test]
fn watermark_detection() {
    let cam = camera(true);
    let mut r = Pcg64::seed_from_u64(11);
    let b = 0.1 * ((W * W + H * H) as f64).sqrt(); // 250
    let border_pt = |r: &mut Pcg64| -> Vec2 {
        // 왼쪽/오른쪽 테두리 띠 안
        let x = if r.random_range(0.0..1.0) < 0.5 { r.random_range(0.0..b - 5.0) } else { r.random_range(W as f64 - b + 5.0..W as f64) };
        Vec2::new(x, r.random_range(0.0..H as f64))
    };
    let make = |r: &mut Pcg64, n_border: usize, n_inner: usize| {
        let (mut k1, mut k2, mut m) = (vec![], vec![], vec![]);
        for i in 0..(n_border + n_inner) {
            let p = if i < n_border {
                border_pt(r)
            } else {
                Vec2::new(r.random_range(b + 10.0..W as f64 - b - 10.0), r.random_range(b + 10.0..H as f64 - b - 10.0))
            };
            let q = p + Vec2::new(3.0 + r.random_range(-1.0..1.0), -2.0 + r.random_range(-1.0..1.0));
            k1.push(Keypoint::new(p.x as f32, p.y as f32));
            k2.push(Keypoint::new(q.x as f32, q.y as f32));
            m.push(FeatureMatch::new(i as u32, i as u32));
        }
        (k1, k2, m)
    };
    let o = opts();
    let (k1, k2, m) = make(&mut r, 100, 0);
    assert!(is_watermark(&cam, &k1, &cam, &k2, &m, &o));
    // 전체 추정도 WATERMARK.
    let g = estimate_two_view(&cam, &k1, &cam, &k2, &m, &o);
    assert_eq!(g.config, C::Watermark);
    // 테두리 비율 69% → 아님.
    let (k1, k2, m) = make(&mut r, 69, 31);
    assert!(!is_watermark(&cam, &k1, &cam, &k2, &m, &o));
}

#[test]
fn deterministic_with_seed() {
    let (d, _, _) = make(6, Scene::General, 300, 200, 0.5);
    let cam = camera(true);
    let a = estimate_two_view(&cam, &d.kps1, &cam, &d.kps2, &d.matches, &opts());
    let b = estimate_two_view(&cam, &d.kps1, &cam, &d.kps2, &d.matches, &opts());
    assert_eq!(a, b);
}
