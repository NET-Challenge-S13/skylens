/*
 * ba_gpu.rs
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

//! CUDA 번들 조정이 CPU 백엔드와 같은 해에 닿는지(합성 문제). 장치가 없으면 건너뛴다.
//!
//! 문제: OPENCV·RADIAL 카메라 두 대(초점·왜곡 정제), 일부 영상은 비기준 rig 센서, 자세 0 고정과
//! 자세 1 의 이동 x 성분 고정(게이지), 일부 점 고정, 0.3px 잡음.

use cumulus3d_ba::{BaBackend, BaSolution, CpuBaBackend, LinearSolverType, Loss, ProblemInput, Termination, TrOptions};
use cumulus3d_core::{Camera, CameraModelKind, Quat, Rigid3, Vec2, Vec3};
use cumulus3d_cuda::{is_available, CudaBundleAdjuster, CPU_CHOLESKY_N};
use std::time::Instant;

struct Rng(u64);

impl Rng {
    /// [−0.5, 0.5) 균등.
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }
    /// 대략 표준정규(균등 12개 합).
    fn g(&mut self) -> f64 {
        (0..12).map(|_| self.u()).sum()
    }
}

fn make_problem(np: usize, npt: usize, loss: Loss, seed: u64) -> ProblemInput {
    let mut rng = Rng(seed);
    let cams = vec![
        Camera::new(1, CameraModelKind::OpenCv, 1000, 800, vec![900.0, 880.0, 500.0, 400.0, 0.05, -0.02, 0.001, 0.002]).unwrap(),
        Camera::new(2, CameraModelKind::Radial, 1000, 800, vec![850.0, 500.0, 400.0, 0.03, -0.01]).unwrap(),
    ];
    let truth: Vec<Rigid3> = (0..np)
        .map(|i| {
            let rv = Vec3::new(0.05 * rng.u(), 0.05 * rng.u(), 0.05 * rng.u());
            Rigid3::new(Quat::from_rotation_vector(&rv), Vec3::new(-(i as f64) * 0.5, 0.4 * rng.u(), 0.0))
        })
        .collect();
    let sensor = Rigid3::new(Quat::from_rotation_vector(&Vec3::new(0.02, -0.03, 0.01)), Vec3::new(-0.3, 0.05, 0.0));
    // 영상: 자세마다 하나(카메라 i%2), 셋째 자세마다 rig 센서 영상 하나 더(카메라 0).
    let (mut img_pose, mut img_cam, mut img_sensor) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..np {
        img_pose.push(i as u32);
        img_cam.push((i % 2) as u32);
        img_sensor.push(None);
        if i % 3 == 0 {
            img_pose.push(i as u32);
            img_cam.push(0);
            img_sensor.push(Some(sensor));
        }
    }
    let len = np as f64 * 0.5;
    let pts_true: Vec<Vec3> = (0..npt).map(|_| Vec3::new((len + 4.0) * (rng.u() + 0.5) - 2.0, 6.0 * rng.u(), 9.0 + 2.0 * rng.u())).collect();
    let (mut obs_img, mut obs_pt, mut obs_xy) = (Vec::new(), Vec::new(), Vec::new());
    let mut points = Vec::new();
    let mut point_const = Vec::new();
    for x in &pts_true {
        let mut vis: Vec<(u32, Vec2)> = Vec::new();
        for (im, (&p, &c)) in img_pose.iter().zip(&img_cam).enumerate() {
            let mut xc = truth[p as usize].transform_point(x);
            if let Some(s) = &img_sensor[im] {
                xc = s.transform_point(&xc);
            }
            let Some(px) = cams[c as usize].cam_to_img(&xc) else { continue };
            if px.x < 0.0 || px.y < 0.0 || px.x > 1000.0 || px.y > 800.0 {
                continue;
            }
            vis.push((im as u32, px + Vec2::new(0.3 * rng.g(), 0.3 * rng.g())));
        }
        if vis.len() < 2 {
            continue;
        }
        let j = points.len() as u32;
        let fixed = j % 13 == 0;
        let n = if fixed { Vec3::zeros() } else { Vec3::new(rng.g(), rng.g(), rng.g()) * 0.05 };
        points.push(x + n);
        point_const.push(fixed);
        for (im, px) in vis {
            obs_img.push(im);
            obs_pt.push(j);
            obs_xy.push(px);
        }
    }
    let poses: Vec<Rigid3> = truth
        .iter()
        .enumerate()
        .map(|(i, t)| {
            if i == 0 {
                return *t;
            }
            let dq = Quat::from_rotation_vector(&Vec3::new(0.004 * rng.g(), 0.004 * rng.g(), 0.004 * rng.g()));
            Rigid3::new(dq.hamilton(&t.rotation).normalized(), t.translation + Vec3::new(rng.g(), rng.g(), rng.g()) * 0.02)
        })
        .collect();
    let mut cams0 = cams.clone();
    cams0[0].params[0] += 4.0;
    cams0[1].params[0] -= 3.0;
    let mut pose_mask = vec![[false; 6]; np];
    pose_mask[1][3] = true;
    ProblemInput {
        poses,
        pose_const: (0..np).map(|i| i == 0).collect(),
        pose_mask,
        cams: cams0,
        cam_var: vec![vec![0, 1, 4, 5, 6, 7], vec![0, 3, 4]],
        img_pose,
        img_cam,
        img_sensor,
        points,
        point_const,
        obs_img,
        obs_pt,
        obs_xy,
        pose_priors: Vec::new(),
        cam_priors: Vec::new(),
        loss,
        solver: LinearSolverType::DenseSchur,
        max_linear_solver_iterations: 500,
    }
}

fn rms(s: &BaSolution, nobs: usize) -> f64 {
    (2.0 * s.result.final_cost / nobs as f64).sqrt()
}

fn compare(input: &ProblemInput, cpu: &BaSolution, gpu: &BaSolution, cost_rel: f64, pose_tol: f64, pt_tol: f64) {
    assert!(gpu.result.termination != Termination::Failure, "{:?}", gpu.result);
    let (cc, gc) = (cpu.result.final_cost, gpu.result.final_cost);
    assert!((cc - gc).abs() <= cost_rel * cc.max(1e-12), "비용 CPU {cc} GPU {gc}");
    assert!(((cpu.result.initial_cost - gpu.result.initial_cost) / cpu.result.initial_cost).abs() < 1e-9, "초기 비용");
    // 게이지: 고정 자세는 비트 단위로 같고, 고정 성분도 그대로.
    assert_eq!(gpu.poses[0], input.poses[0]);
    assert_eq!(gpu.poses[1].translation.x.to_bits(), input.poses[1].translation.x.to_bits());
    for (a, b) in cpu.poses.iter().zip(&gpu.poses) {
        assert!((a.translation - b.translation).norm() < pose_tol, "{a:?} vs {b:?}");
        assert!(a.rotation.angular_distance(&b.rotation) < pose_tol, "{a:?} vs {b:?}");
    }
    for (a, b) in cpu.cams.iter().zip(&gpu.cams) {
        for (x, y) in a.params.iter().zip(&b.params) {
            assert!((x - y).abs() <= 1e3 * pose_tol * x.abs().max(1.0), "카메라 {:?} vs {:?}", a.params, b.params);
        }
    }
    for (j, (a, b)) in cpu.points.iter().zip(&gpu.points).enumerate() {
        if input.point_const[j] {
            assert_eq!(a, b);
            assert_eq!(*b, input.points[j]);
        } else {
            assert!((a - b).norm() < pt_tol, "점 {j}: {a:?} vs {b:?}");
        }
    }
}

#[test]
fn gpu_ba_matches_cpu_dense_path() {
    if !is_available() {
        eprintln!("CUDA 장치 없음: 건너뜀");
        return;
    }
    let gpu = CudaBundleAdjuster::try_default().expect("CUDA BA");
    let opts = TrOptions::new(200, 1e-12, 1e-10, 1e-14);
    for loss in [Loss::Trivial, Loss::Cauchy(2.0), Loss::Huber(1.0), Loss::SoftL1(1.5)] {
        let input = make_problem(12, 1500, loss, 11);
        let nobs = input.obs_img.len();
        let t = Instant::now();
        let c = CpuBaBackend.solve(&input, &opts).unwrap();
        let tc = t.elapsed();
        let t = Instant::now();
        let g = gpu.solve_gpu(&input, &opts).expect("GPU BA");
        let tg = t.elapsed();
        eprintln!(
            "{loss:?}: 관측 {nobs} CPU {tc:.2?} ({} 반복, RMS {:.4}) GPU {tg:.2?} ({} 반복, RMS {:.4}, {:?})",
            c.result.num_iterations,
            rms(&c, nobs),
            g.result.num_iterations,
            rms(&g, nobs),
            g.linear_solver
        );
        assert_eq!(g.linear_solver, LinearSolverType::DenseSchur);
        if loss == Loss::Trivial {
            assert!((rms(&c, nobs) - rms(&g, nobs)).abs() < 1e-6);
        }
        compare(&input, &c, &g, 1e-7, 1e-6, 1e-5);
    }
}

#[test]
fn gpu_ba_matches_cpu_pcg_path() {
    if !is_available() {
        eprintln!("CUDA 장치 없음: 건너뜀");
        return;
    }
    let gpu = CudaBundleAdjuster::try_default().expect("CUDA BA");
    // 자세 100 개 → 축소 계통 차원 > CPU_CHOLESKY_N 이라 GPU PCG 경로.
    let input = make_problem(100, 6000, Loss::Trivial, 5);
    assert!(6 * 99 + 9 > CPU_CHOLESKY_N);
    let nobs = input.obs_img.len();
    let opts = TrOptions::new(300, 1e-12, 1e-10, 1e-14);
    let t = Instant::now();
    let c = CpuBaBackend.solve(&input, &opts).unwrap();
    let tc = t.elapsed();
    let t = Instant::now();
    let g = gpu.solve_gpu(&input, &opts).expect("GPU BA");
    let tg = t.elapsed();
    eprintln!(
        "PCG: 관측 {nobs} CPU {tc:.2?} ({} 반복, RMS {:.4}) GPU {tg:.2?} ({} 반복, RMS {:.4})",
        c.result.num_iterations,
        rms(&c, nobs),
        g.result.num_iterations,
        rms(&g, nobs)
    );
    assert_eq!(g.linear_solver, LinearSolverType::IterativeSchur);
    // 부정확한 단계(PCG)라 LM 경로가 다르다. 같은 최소점에 닿았는지만 본다.
    assert!((rms(&c, nobs) - rms(&g, nobs)).abs() < 1e-4, "RMS");
    compare(&input, &c, &g, 1e-5, 1e-4, 1e-3);
}

#[test]
fn backend_trait_runs_on_gpu() {
    if !is_available() {
        eprintln!("CUDA 장치 없음: 건너뜀");
        return;
    }
    let gpu = CudaBundleAdjuster::try_default().expect("CUDA BA");
    assert_eq!(gpu.name(), "cuda");
    let input = make_problem(6, 400, Loss::Cauchy(1.0), 3);
    let opts = TrOptions::new(100, 0.0, 1e-4, 0.0);
    let s = gpu.solve(&input, &opts).unwrap();
    assert!(s.result.termination != Termination::Failure);
    assert!(s.result.final_cost < s.result.initial_cost);
}
