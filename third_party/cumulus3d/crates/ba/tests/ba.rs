/*
 * ba.rs
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

mod common;

use common::*;
use cumulus3d_ba::*;
use cumulus3d_core::{Quat, Reconstruction, Rigid3, TrackEntry, Vec2, Vec3};
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

fn gauge_info(rec: &Reconstruction) -> (Rigid3, Rigid3, usize) {
    // 정규화된 초기 자세 기준으로 영상1(id 1), 영상2(id 2) 와 고정 축 d.
    let norm = |p: Rigid3| Rigid3::new(p.rotation.normalized(), p.translation);
    let p1 = norm(rec.world_to_cam(1).unwrap());
    let p2 = norm(rec.world_to_cam(2).unwrap());
    let b = p1.compose(&p2.inverse()).translation;
    let d = (0..3).max_by(|&a, &c| b[a].abs().total_cmp(&b[c].abs())).unwrap();
    (p1, p2, d)
}

#[test]
fn recovers_synthetic_scene_default_config() {
    // 검증 장면: 16 위치 × 3 대 = 48 장, 점 1만.
    let scene = make_scene(&SceneOpts { num_positions: 16, num_points: 10000, ..Default::default() });
    let truth = &scene.truth;
    assert!(truth.registered_image_count() == 48 && truth.num_points3d() == 10000);
    let mut rec = perturb(truth, &Perturb::default(), 7);
    let (p1, p2, d) = gauge_info(&rec);
    let cams_before = rec.cameras().clone();
    let s = bundle_adjust(&mut rec, &BaConfig::default()).unwrap();
    assert!(s.is_usable(), "{s:?}");
    assert_eq!(s.linear_solver, LinearSolverType::DenseSchur);
    let rms = rms_per_coord(&rec);
    eprintln!("{s:?} rms/coord={rms}");
    assert!((0.45..0.55).contains(&rms), "rms {rms}");
    // 게이지: 영상1 자세 비트 단위 동일, 영상2 의 고정 이동 성분 동일.
    assert_eq!(rec.world_to_cam(1).unwrap(), p1);
    assert_eq!(rec.world_to_cam(2).unwrap().translation[d].to_bits(), p2.translation[d].to_bits());
    assert_ne!(rec.world_to_cam(2).unwrap().rotation, p2.rotation);
    // 주점 고정, 초점·왜곡은 정제됨.
    for (id, c) in rec.cameras() {
        let b = &cams_before[id];
        assert_eq!(c.params[2].to_bits(), b.params[2].to_bits());
        assert_eq!(c.params[3].to_bits(), b.params[3].to_bits());
        assert_ne!(c.params[0], b.params[0]);
        let t = truth.camera(*id).unwrap();
        assert!((c.params[0] - t.params[0]).abs() / t.params[0] < 2e-3, "fx {} vs {}", c.params[0], t.params[0]);
    }
    let (er, ec) = pose_errors(&rec, truth);
    eprintln!("max rot err {er} deg, rms center err {ec} m");
    assert!(er < 0.05 && ec < 0.05, "{er} {ec}");
    // 같은 관측의 최대우도 해(정답에서 출발)와 게이지만 다르고 일치해야 한다.
    let mut ml = truth.clone();
    bundle_adjust(&mut ml, &BaConfig { gradient_tolerance: 1e-8, ..Default::default() }).unwrap();
    let (mr, mc) = pose_errors(&rec, &ml);
    eprintln!("vs ML: rot {mr} deg, center {mc} m");
    assert!(mr < 1e-3 && mc < 1e-3, "{mr} {mc}");
    // 점 오차 갱신됨.
    assert!(rec.points3d().all(|(_, p)| p.error >= 0.0));
}

#[test]
fn three_solvers_agree() {
    let scene = make_scene(&SceneOpts { num_positions: 8, num_points: 1500, seed: 3, ..Default::default() });
    let init = perturb(&scene.truth, &Perturb::default(), 11);
    let mut results = Vec::new();
    for solver in [LinearSolverType::DenseSchur, LinearSolverType::SparseSchur, LinearSolverType::IterativeSchur] {
        let mut rec = init.clone();
        let cfg = BaConfig { linear_solver: solver, ..Default::default() };
        let s = bundle_adjust(&mut rec, &cfg).unwrap();
        assert_eq!(s.linear_solver, solver);
        assert!(s.is_usable());
        eprintln!("{solver:?}: {s:?}");
        results.push((s, rec));
    }
    let (s0, r0) = &results[0];
    // 직접 풀이기끼리는 거의 같은 해, PCG 는 부정확 단계라 약하게 정해지는 방향(깊이)에서 조금 다르다.
    for ((s, r), tol) in results[1..].iter().zip([1e-6, 1e-3]) {
        assert!((s.final_cost - s0.final_cost).abs() / s0.final_cost < 1e-8, "{} vs {}", s.final_cost, s0.final_cost);
        for pid in r0.point3d_ids() {
            let d = (r.point3d(pid).unwrap().xyz - r0.point3d(pid).unwrap().xyz).norm();
            assert!(d < tol, "{:?} point {pid} differs {d}", s.linear_solver);
        }
    }
    // 직접 풀이기 둘은 같은 경로를 밟는다.
    assert_eq!(results[0].0.num_iterations, results[1].0.num_iterations);
}

#[test]
fn solver_selection_rule() {
    let c = BaConfig::default();
    assert_eq!(select_linear_solver(50, &c), LinearSolverType::DenseSchur);
    assert_eq!(select_linear_solver(51, &c), LinearSolverType::SparseSchur);
    assert_eq!(select_linear_solver(1000, &c), LinearSolverType::SparseSchur);
    assert_eq!(select_linear_solver(1001, &c), LinearSolverType::IterativeSchur);
}

/// 관측 일부를 큰 오차로 오염.
fn corrupt(rec: &mut Reconstruction, frac: f64, seed: u64) -> std::collections::HashSet<(u32, u32)> {
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut bad = std::collections::HashSet::new();
    let mut out = rec.clone();
    // 2D 점을 바꾸려면 영상을 새로 만들어야 하므로 재구성을 다시 조립한다.
    let mut fresh = Reconstruction::new();
    for c in rec.cameras().values() {
        fresh.add_camera_own_rig(c.clone()).unwrap();
    }
    for id in rec.registered_images() {
        let im = rec.image(id).unwrap();
        let pts: Vec<Vec2> = im
            .points2d()
            .iter()
            .enumerate()
            .map(|(k, p)| {
                if p.has_point3d() && rng.random_range(0.0..1.0) < frac {
                    bad.insert((id, k as u32));
                    let a = rng.random_range(0.0..std::f64::consts::TAU);
                    p.xy + Vec2::new(a.cos(), a.sin()) * rng.random_range(20.0..60.0)
                } else {
                    p.xy
                }
            })
            .collect();
        let new = cumulus3d_core::Image::new(id, im.name.clone(), im.camera_id, pts);
        fresh.add_image_own_frame(new, rec.world_to_cam(id)).unwrap();
        fresh.register_image(id).unwrap();
    }
    for (pid, p) in rec.points3d() {
        fresh
            .add_point3d_with_id(pid, cumulus3d_core::Point3D { xyz: p.xyz, color: p.color, error: -1.0, track: p.track.clone() })
            .unwrap();
    }
    std::mem::swap(&mut out, &mut fresh);
    *rec = out;
    bad
}

fn clean_rms(rec: &Reconstruction, bad: &std::collections::HashSet<(u32, u32)>) -> f64 {
    let mut s = 0.0;
    let mut n = 0;
    for (_, p) in rec.points3d() {
        for e in &p.track {
            if bad.contains(&(e.image_id, e.point2d_idx)) {
                continue;
            }
            let pose = rec.world_to_cam(e.image_id).unwrap();
            let im = rec.image(e.image_id).unwrap();
            let cam = rec.camera(im.camera_id).unwrap();
            s += Reconstruction::squared_reprojection_error(&pose, cam, &im.point2d(e.point2d_idx).xy, &p.xyz);
            n += 1;
        }
    }
    (s / (2 * n) as f64).sqrt()
}

#[test]
fn robust_losses_resist_outliers() {
    let scene = make_scene(&SceneOpts { num_positions: 8, num_points: 2000, seed: 5, ..Default::default() });
    let mut init = perturb(&scene.truth, &Perturb { rot_deg: 0.2, pos_m: 0.2, point_m: 0.2, focal_frac: 0.01 }, 2);
    let bad = corrupt(&mut init, 0.05, 9);
    let run = |loss: Loss| {
        let mut rec = init.clone();
        let s = bundle_adjust(&mut rec, &BaConfig { loss, ..Default::default() }).unwrap();
        assert!(s.is_usable());
        let (er, ec) = pose_errors(&rec, &scene.truth);
        let rms = clean_rms(&rec, &bad);
        eprintln!("{loss:?}: clean rms {rms:.3} px, rot {er:.4} deg, center {ec:.4} m, iters {}", s.num_iterations);
        (rms, ec)
    };
    let (rms_t, ec_t) = run(Loss::Trivial);
    for loss in [Loss::Cauchy(1.0), Loss::Huber(1.0), Loss::SoftL1(1.0)] {
        let (rms, ec) = run(loss);
        assert!(rms < 0.6, "{loss:?} rms {rms}");
        assert!(rms < 0.5 * rms_t, "{loss:?}: {rms} vs trivial {rms_t}");
        assert!(ec < 0.25 * ec_t, "{loss:?}: {ec} vs trivial {ec_t}");
    }
}

#[test]
fn constant_flags_respected() {
    let scene = make_scene(&SceneOpts { num_positions: 6, num_points: 1200, seed: 8, ..Default::default() });
    let mut rec = perturb(&scene.truth, &Perturb::default(), 4);
    let before = rec.clone();
    let const_pts: Vec<u64> = rec.point3d_ids().into_iter().step_by(10).collect();
    let cfg = BaConfig {
        constant_poses: [3u32, 7].into_iter().collect(),
        constant_cameras: [2u32].into_iter().collect(),
        constant_points: const_pts.iter().copied().collect(),
        ..Default::default()
    };
    let s = bundle_adjust(&mut rec, &cfg).unwrap();
    assert!(s.is_usable());
    // 상수 프레임이 2개 이상이므로 자동 게이지는 아무것도 하지 않는다 → 영상 1 은 움직인다.
    for id in [3u32, 7] {
        assert_eq!(rec.world_to_cam(id).unwrap(), before.world_to_cam(id).unwrap());
    }
    assert_ne!(rec.world_to_cam(1).unwrap(), before.world_to_cam(1).unwrap());
    assert_eq!(rec.camera(2).unwrap().params, before.camera(2).unwrap().params);
    assert_ne!(rec.camera(1).unwrap().params, before.camera(1).unwrap().params);
    for pid in &const_pts {
        assert_eq!(rec.point3d(*pid).unwrap().xyz, before.point3d(*pid).unwrap().xyz);
    }

    // 점 고정 + 자세·내부 정제 끔 → 아무것도 안 변함.
    let mut rec2 = before.clone();
    let cfg2 = BaConfig {
        refine_points: false,
        refine_poses: false,
        refine_focal_length: false,
        refine_extra_params: false,
        ..Default::default()
    };
    bundle_adjust(&mut rec2, &cfg2).unwrap();
    for id in before.registered_images() {
        assert_eq!(rec2.world_to_cam(id).unwrap(), before.world_to_cam(id).unwrap());
    }
    for (pid, p) in before.points3d() {
        assert_eq!(rec2.point3d(pid).unwrap().xyz, p.xyz);
    }
}

#[test]
fn points_only_ba_with_fixed_cameras() {
    // point_triangulator 의 전역 BA 와 같은 설정: 자세·내부 고정, 점만.
    let scene = make_scene(&SceneOpts { num_positions: 6, num_points: 1500, seed: 12, ..Default::default() });
    let mut rec = scene.truth.clone();
    let mut rng = Pcg64::seed_from_u64(1);
    for pid in rec.point3d_ids() {
        let x = rec.point3d(pid).unwrap().xyz + Vec3::new(gauss(&mut rng), gauss(&mut rng), gauss(&mut rng)) * 0.3;
        rec.set_point3d_xyz(pid, x).unwrap();
    }
    let cfg = BaConfig {
        refine_poses: false,
        refine_focal_length: false,
        refine_extra_params: false,
        max_num_iterations: 50,
        gradient_tolerance: 1.0,
        ..Default::default()
    };
    let s = bundle_adjust(&mut rec, &cfg).unwrap();
    assert!(s.is_usable());
    let mut max_err: f64 = 0.0;
    let mut sum = 0.0;
    for (pid, p) in scene.truth.points3d() {
        let e = (rec.point3d(pid).unwrap().xyz - p.xyz).norm();
        max_err = max_err.max(e);
        sum += e;
    }
    let mean = sum / scene.truth.num_points3d() as f64;
    eprintln!("{s:?} mean point err {mean} max {max_err}");
    assert!(mean < 0.1, "mean {mean}");
    assert!((0.4..0.6).contains(&rms_per_coord(&rec)));
}

#[test]
fn negative_depth_observations_prefiltered() {
    let scene = make_scene(&SceneOpts { num_positions: 4, num_points: 800, seed: 21, ..Default::default() });
    let mut rec = scene.truth.clone();
    // 드론 위(카메라 뒤) 점, 여분 2D 점에 연결.
    let spare = |rec: &Reconstruction, id: u32| (rec.image(id).unwrap().num_points2d() - 1) as u32;
    let track = vec![TrackEntry::new(1, spare(&rec, 1)), TrackEntry::new(2, spare(&rec, 2)), TrackEntry::new(3, spare(&rec, 3))];
    let pid = rec.add_point3d(Vec3::new(5.0, 0.0, 120.0), track, [0, 0, 0]).unwrap();
    let s = bundle_adjust(&mut rec, &BaConfig::default()).unwrap();
    assert!(s.dropped_observations >= 2);
    assert!(rec.point3d(pid).is_none());
    rec.check_invariants().unwrap();
}

#[test]
fn refine_abs_pose_converges() {
    let cam = make_cameras().remove(0);
    let mut rng = Pcg64::seed_from_u64(3);
    let r = Quat::from_rotation_vector(&Vec3::new(0.1, -0.2, 0.05)).to_rotation_matrix();
    let truth = Rigid3::from_rotation_matrix(&r, Vec3::new(0.3, -0.2, 1.0));
    let (mut x2, mut x3, mut mask) = (Vec::new(), Vec::new(), Vec::new());
    while x2.len() < 300 {
        let xc = Vec3::new(rng.random_range(-6.0..6.0), rng.random_range(-4.0..4.0), rng.random_range(8.0..20.0));
        let px = cam.cam_to_img(&xc).unwrap();
        if px.x < 0.0 || px.y < 0.0 || px.x > 1000.0 || px.y > 750.0 {
            continue;
        }
        let xw = truth.inverse().transform_point(&xc);
        let k = x2.len();
        let mut obs = px + Vec2::new(gauss(&mut rng), gauss(&mut rng)) * 0.5;
        // 10% 마스크된 이상치, 5% 마스크 안 된 이상치(Cauchy 가 견딘다).
        if k % 10 == 0 {
            obs += Vec2::new(80.0, -60.0);
            mask.push(false);
        } else {
            if k % 20 == 1 {
                obs += Vec2::new(25.0, 30.0);
            }
            mask.push(true);
        }
        x2.push(obs);
        x3.push(xw);
    }
    let dq = Quat::from_rotation_vector(&Vec3::new(0.02, -0.03, 0.01));
    let mut pose = Rigid3::new(dq.hamilton(&truth.rotation), truth.translation + Vec3::new(0.4, -0.3, 0.5));
    let s = refine_abs_pose(&cam, &x2, &x3, &mask, &mut pose, Loss::Cauchy(1.0), 100).unwrap();
    let er = pose.rotation.angular_distance(&truth.rotation).to_degrees();
    let ec = (pose.center() - truth.center()).norm();
    eprintln!("{s:?} rot {er} deg center {ec}");
    assert!(s.is_usable() && s.final_cost < s.initial_cost);
    assert!(er < 0.05 && ec < 0.02, "{er} {ec}");
    // 입력 검증
    assert!(refine_abs_pose(&cam, &x2, &x3[..5], &mask, &mut pose, Loss::Trivial, 10).is_err());
    assert!(refine_abs_pose(&cam, &x2, &x3, &vec![false; x2.len()], &mut pose, Loss::Trivial, 10).is_err());
}

#[test]
fn window_ba_keeps_outside_fixed() {
    // 일부 영상만 BA: 밖과 공유하는 점은 상수, 밖 영상 자세는 그대로.
    let scene = make_scene(&SceneOpts { num_positions: 8, num_points: 1500, seed: 31, ..Default::default() });
    let mut rec = perturb(&scene.truth, &Perturb { rot_deg: 0.1, pos_m: 0.1, point_m: 0.1, focal_frac: 0.0 }, 5);
    let before = rec.clone();
    let window: std::collections::HashSet<u32> = (13..=24).collect();
    let cfg = BaConfig { images: window.clone(), refine_focal_length: false, refine_extra_params: false, ..Default::default() };
    let s = bundle_adjust(&mut rec, &cfg).unwrap();
    assert!(s.is_usable() && s.num_images == 12);
    for id in before.registered_images() {
        if !window.contains(&id) {
            assert_eq!(rec.world_to_cam(id).unwrap(), before.world_to_cam(id).unwrap());
        }
    }
    for (pid, p) in before.points3d() {
        if p.track.iter().any(|t| !window.contains(&t.image_id)) {
            assert_eq!(rec.point3d(pid).unwrap().xyz, p.xyz);
        }
    }
}

#[test]
fn single_thread_matches_multi_thread() {
    let scene = make_scene(&SceneOpts { num_positions: 5, num_points: 1000, seed: 41, ..Default::default() });
    let init = perturb(&scene.truth, &Perturb::default(), 3);
    let mut a = init.clone();
    let mut b = init.clone();
    let sa = bundle_adjust(&mut a, &BaConfig { num_threads: 1, ..Default::default() }).unwrap();
    let sb = bundle_adjust(&mut b, &BaConfig::default()).unwrap();
    assert_eq!(sa.final_cost.to_bits(), sb.final_cost.to_bits());
    for (pid, p) in a.points3d() {
        assert_eq!(p.xyz, b.point3d(pid).unwrap().xyz);
    }
}
