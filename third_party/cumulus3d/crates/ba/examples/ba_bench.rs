/*
 * ba_bench.rs
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

//! 규모 벤치마크: 80 위치 × 3 대 = 240 장, 3D 점 20만, OPENCV, 잡음 0.5 px, 초기 섭동.
//! 실행: cargo run --release -p cumulus3d-ba --example ba_bench [-- 위치수 점수 풀이기(auto|dense|sparse|iterative)]

#[path = "../tests/common/mod.rs"]
mod common;

use common::*;
use cumulus3d_ba::{bundle_adjust, BaConfig, LinearSolverType};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let positions: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(80);
    let points: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let solver = match args.get(3).map(String::as_str) {
        Some("dense") => LinearSolverType::DenseSchur,
        Some("sparse") => LinearSolverType::SparseSchur,
        Some("iterative") => LinearSolverType::IterativeSchur,
        _ => LinearSolverType::Auto,
    };
    let t = Instant::now();
    let scene = make_scene(&SceneOpts { num_positions: positions, num_points: points, spacing: 12.0, seed: 99, ..Default::default() });
    let mut rec = perturb(&scene.truth, &Perturb::default(), 5);
    let nobs = rec.total_observations();
    println!(
        "장면: 영상 {} 점 {} 관측 {} (평균 트랙 {:.2}) 생성 {:.2}s",
        rec.registered_image_count(),
        rec.num_points3d(),
        nobs,
        rec.mean_track_len(),
        t.elapsed().as_secs_f64()
    );
    let cfg = BaConfig { linear_solver: solver, ..Default::default() };
    let t = Instant::now();
    let s = bundle_adjust(&mut rec, &cfg).expect("BA");
    let dt = t.elapsed().as_secs_f64();
    println!("{s:?}");
    println!(
        "BA 1회: {:.2}s, 반복 {} (성공 {}), 반복당 {:.3}s, 풀이기 {:?}, RMS/축 {:.3} px",
        dt,
        s.num_iterations,
        s.num_successful_steps,
        dt / s.num_iterations.max(1) as f64,
        s.linear_solver,
        rms_per_coord(&rec)
    );
    let (er, ec) = pose_errors(&rec, &scene.truth);
    println!("정답 대비(상사 정렬 후): 최대 회전 {er:.4}°, RMS 중심 {ec:.4} m");
}
