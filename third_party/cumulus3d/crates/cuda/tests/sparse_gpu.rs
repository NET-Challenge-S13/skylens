/*
 * sparse_gpu.rs
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

//! 희소 단계 CUDA 백엔드와 CPU 경로 대조(장치가 없으면 건너뜀).

use cumulus3d_core::{MatchGraph, MatchGraphOptions, Rigid3, Vec2, Vec3};
use cumulus3d_cuda::{is_available, CudaSparse};
use cumulus3d_matching::batched::synthetic_pairs;
use cumulus3d_matching::{CpuTwoView, TwoViewBackend, TwoViewOptions, VerifyJob};
use cumulus3d_sfm::absolute_pose::{solve_abs_pose, AbsolutePoseOptions};
use cumulus3d_sfm::global_mapper::{global_mapper, GlobalSfmOptions};
use cumulus3d_sfm::synthetic::{generate, opencv_camera, SceneConfig};
use cumulus3d_sfm::triangulation::{estimate_triangulation, TriObservation, TriangulationRansacParams};
use cumulus3d_sfm::{
    register_images_with, triangulate_points_with, AbsPoseBackend, CpuAbsPose, CpuTriangulation, ObsQuery, PointTriangulatorOptions,
    RegistrationOptions, TriProblem, TriangulationBackend, TriangulationScope,
};
use rand::{RngExt, SeedableRng};
use std::collections::HashSet;
use std::time::Instant;

fn device() -> Option<CudaSparse> {
    if !is_available() {
        eprintln!("CUDA 장치 없음: 건너뜀");
        return None;
    }
    Some(CudaSparse::try_default().expect("CUDA 희소 백엔드"))
}

#[test]
fn two_view_gpu_equals_cpu() {
    let Some(gpu) = device() else { return };
    let (store, pairs) = synthetic_pairs(21, 5, 1500, 0.4);
    let jobs: Vec<VerifyJob<'_>> = pairs.iter().map(|(a, b, m)| VerifyJob { id1: *a, id2: *b, matches: m }).collect();
    for seed in [1u64, 2, 3] {
        let mut opts = TwoViewOptions::default();
        opts.ransac.random_seed = Some(seed);
        let t = Instant::now();
        let want = CpuTwoView.verify_pairs(&store, &jobs, &opts);
        let tc = t.elapsed();
        let t = Instant::now();
        let got = gpu.verify_pairs(&store, &jobs, &opts);
        let tg = t.elapsed();
        eprintln!("짝 {}: CPU {tc:.2?}, GPU {tg:.2?}", jobs.len());
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(g.config, w.config);
            // 채점은 CPU 와 같은 순서로 더하므로 인라이어도 같아야 한다(수치 잡음 여유 1%).
            let a: HashSet<_> = g.inlier_matches.iter().copied().collect();
            let b: HashSet<_> = w.inlier_matches.iter().copied().collect();
            let diff = a.symmetric_difference(&b).count();
            assert!(diff * 100 <= b.len().max(1), "인라이어 차이 {diff} / {}", b.len());
        }
    }
}

#[test]
fn abs_pose_gpu_equals_cpu() {
    let Some(gpu) = device() else { return };
    let cam = opencv_camera(1);
    let mut rng = rand_pcg::Pcg64::seed_from_u64(31);
    for trial in 0..10u64 {
        let pose = Rigid3::new(
            cumulus3d_core::Quat::from_axis_angle(&Vec3::new(0.3, -0.5, 0.8).normalize(), rng.random_range(-1.0..1.0)),
            Vec3::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), rng.random_range(4.0..8.0)),
        );
        let inv = pose.inverse();
        let (mut p2, mut p3) = (Vec::new(), Vec::new());
        while p2.len() < 2000 {
            let xy = Vec2::new(rng.random_range(0.0..1920.0), rng.random_range(0.0..1080.0));
            let ray = cam.img_to_ray(&xy).unwrap();
            p3.push(inv * (ray / ray.z * rng.random_range(3.0..10.0)));
            if p2.len() % 3 == 0 {
                p2.push(Vec2::new(rng.random_range(0.0..1920.0), rng.random_range(0.0..1080.0)));
            } else {
                p2.push(xy + Vec2::new(rng.random_range(-0.5..0.5), rng.random_range(-0.5..0.5)));
            }
        }
        let opts = AbsolutePoseOptions { random_seed: Some(trial), ..Default::default() };
        let t = Instant::now();
        let want = solve_abs_pose(&cam, &p2, &p3, &opts).expect("CPU 자세");
        let tc = t.elapsed();
        let t = Instant::now();
        let got = gpu.solve(&cam, &p2, &p3, &opts).expect("GPU 자세");
        let tg = t.elapsed();
        eprintln!("자세 {trial}: CPU {tc:.2?}, GPU {tg:.2?}, 반복 {} / {}", want.num_trials, got.num_trials);
        let diff = got.inlier_mask.iter().zip(&want.inlier_mask).filter(|(a, b)| a != b).count();
        assert!(diff * 100 <= want.num_inliers, "인라이어 차이 {diff}");
        assert!((got.world_to_cam.translation - want.world_to_cam.translation).norm() < 1e-6);
        assert!(got.world_to_cam.rotation.angular_distance(&want.world_to_cam.rotation) < 1e-6);
        // CPU 백엔드 트레이트도 같은 값.
        let cpu = CpuAbsPose.solve(&cam, &p2, &p3, &opts).unwrap();
        assert_eq!(cpu.inlier_mask, want.inlier_mask);
    }
}

fn look_at(c: Vec3, target: Vec3) -> Rigid3 {
    let z = (target - c).normalize();
    let x = Vec3::new(1.0, 0.0, 0.0).cross(&z).normalize();
    let y = z.cross(&x);
    let r = cumulus3d_core::Mat3::from_rows(&[x.transpose(), y.transpose(), z.transpose()]);
    Rigid3::from_rotation_matrix(&r, -(r * c))
}

#[test]
fn triangulation_gpu_matches_cpu() {
    let Some(gpu) = device() else { return };
    let mut rng = rand_pcg::Pcg64::seed_from_u64(41);
    // 문제마다 뷰 2..20 개, 일부 관측은 크게 틀어진 광선.
    let mut all: Vec<Vec<TriObservation>> = Vec::new();
    for _ in 0..3000 {
        let x = Vec3::new(rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0), rng.random_range(0.0..3.0));
        let n = rng.random_range(2..20usize);
        let mut obs = Vec::new();
        for _ in 0..n {
            let c = Vec3::new(rng.random_range(-8.0..8.0), rng.random_range(-8.0..8.0), rng.random_range(18.0..25.0));
            let p = look_at(c, Vec3::zeros());
            let mut r = (p * x).normalize();
            if rng.random_range(0.0..1.0) < 0.2 {
                r = (r + Vec3::new(rng.random_range(-0.2..0.2), rng.random_range(-0.2..0.2), 0.0)).normalize();
            } else {
                r = (r + Vec3::new(rng.random_range(-1e-4..1e-4), rng.random_range(-1e-4..1e-4), 0.0)).normalize();
            }
            obs.push(TriObservation::new(&p, r));
        }
        all.push(obs);
    }
    let params = TriangulationRansacParams::default();
    let problems: Vec<TriProblem<'_>> = all.iter().map(|o| TriProblem { obs: o, params: params.clone() }).collect();
    let t = Instant::now();
    let want: Vec<_> = all.iter().map(|o| estimate_triangulation(o, &params)).collect();
    let tc = t.elapsed();
    let t = Instant::now();
    let got = gpu.estimate_batch_gpu(&problems).expect("GPU 삼각측량");
    let tg = t.elapsed();
    eprintln!("삼각측량 {}: CPU(직렬) {tc:.2?}, GPU {tg:.2?}", problems.len());
    let mut same_mask = 0;
    for (g, w) in got.iter().zip(&want) {
        match (g, w) {
            (Some((gx, gm)), Some((wx, wm))) => {
                assert!((gx - wx).norm() <= 1e-6 * (1.0 + wx.norm()), "{gx:?} {wx:?}");
                if gm == wm {
                    same_mask += 1;
                }
            }
            (None, None) => same_mask += 1,
            _ => {}
        }
    }
    assert!(same_mask * 1000 >= 999 * problems.len(), "같은 결과 {same_mask} / {}", problems.len());
    // CPU 백엔드 트레이트는 그대로 같다.
    assert_eq!(CpuTriangulation.estimate_batch(&problems), want);
}

#[test]
fn observation_checks_gpu_equal_cpu() {
    let Some(gpu) = device() else { return };
    let mut rng = rand_pcg::Pcg64::seed_from_u64(51);
    let cams: Vec<_> = (1..=3).map(opencv_camera).collect();
    let poses: Vec<Rigid3> =
        (0..10).map(|k| look_at(Vec3::new(k as f64 - 5.0, rng.random_range(-2.0..2.0), 20.0), Vec3::zeros())).collect();
    let queries: Vec<ObsQuery> = (0..20000)
        .map(|_| ObsQuery {
            camera: rng.random_range(0..3u32),
            pose: rng.random_range(0..10u32),
            xy: Vec2::new(rng.random_range(0.0..1920.0), rng.random_range(0.0..1080.0)),
            xyz: Vec3::new(rng.random_range(-6.0..6.0), rng.random_range(-6.0..6.0), rng.random_range(-3.0..30.0)),
        })
        .collect();
    let want = CpuTriangulation.check_observations(&cams, &poses, &queries);
    let got = gpu.check_observations_gpu(&cams, &poses, &queries).expect("GPU 관측 검사");
    assert_eq!(got.len(), want.len());
    // 같은 순서로 계산하므로 보통 비트 단위로 같다(여유: 상대 1e-12).
    let close = |a: f64, b: f64| a == b || (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0);
    for (g, w) in got.iter().zip(&want) {
        assert!(close(g.sq_reproj_error, w.sq_reproj_error), "{g:?} {w:?}");
        assert!(close(g.depth, w.depth), "{g:?} {w:?}");
    }
}

#[test]
fn register_and_triangulate_gpu_close_to_cpu() {
    let Some(gpu) = device() else { return };
    let scene = generate(&SceneConfig::default());
    let graph_up_to = |n: usize| {
        let opts = MatchGraphOptions { image_names: scene.names_up_to(n), ..Default::default() };
        MatchGraph::from_store(&scene.store, &opts)
    };
    let out = global_mapper(&scene.store, &graph_up_to(13), &GlobalSfmOptions::script()).unwrap();
    let base = out.reconstruction;
    let mut counts = Vec::new();
    for use_gpu in [false, true] {
        let mut rec = base.clone();
        let t = Instant::now();
        let pose_backend: &dyn AbsPoseBackend = if use_gpu { &gpu } else { &CpuAbsPose };
        let tri_backend: &dyn TriangulationBackend = if use_gpu { &gpu } else { &CpuTriangulation };
        for pos in 13..16 {
            let graph = graph_up_to(pos + 1);
            let rep = register_images_with(&mut rec, &scene.store, &graph, &RegistrationOptions::default(), pose_backend).unwrap();
            let topts = PointTriangulatorOptions { scope: TriangulationScope::Images(rep.registered()), ..Default::default() };
            triangulate_points_with(&mut rec, &graph, &topts, tri_backend).unwrap();
            rec.check_invariants().unwrap();
        }
        eprintln!("gpu {use_gpu}: {:.2?}, 등록 {} 점 {}", t.elapsed(), rec.registered_image_count(), rec.num_points3d());
        counts.push((rec.registered_image_count(), rec.num_points3d()));
    }
    assert_eq!(counts[0].0, counts[1].0);
    let (a, b) = (counts[0].1 as f64, counts[1].1 as f64);
    assert!((a - b).abs() <= 0.01 * a, "점 수 CPU {a} GPU {b}");
}
