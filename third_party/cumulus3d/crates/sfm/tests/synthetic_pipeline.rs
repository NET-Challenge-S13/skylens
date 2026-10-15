/*
 * synthetic_pipeline.rs
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

//! 합성 장면 통합 테스트: 전역 매퍼(BA 0회) → 위치별 등록 + 삼각측량.

use cumulus3d_core::{ImageId, MatchGraph, MatchGraphOptions, Reconstruction, Vec3};
use cumulus3d_sfm::global_mapper::{global_mapper, GlobalSfmOptions};
use cumulus3d_sfm::math::umeyama;
use cumulus3d_sfm::registration::{register_images, RegistrationOptions};
use cumulus3d_sfm::synthetic::{generate, Scene, SceneConfig};
use cumulus3d_sfm::triangulator::{triangulate_points, PointRefiner, PointTriangulatorOptions, TriangulationScope};

fn graph_up_to(scene: &Scene, n: usize) -> MatchGraph {
    let opts = MatchGraphOptions { image_names: scene.names_up_to(n), ..Default::default() };
    MatchGraph::from_store(&scene.store, &opts)
}

/// 상사 정렬 후 (중심 오차 중앙값/장면 범위, 회전 오차 중앙값 도).
fn pose_errors(rec: &Reconstruction, scene: &Scene) -> (f64, f64) {
    let ids: Vec<ImageId> = rec.registered_images();
    let est: Vec<Vec3> = ids.iter().map(|i| rec.projection_center(*i).unwrap()).collect();
    let tru: Vec<Vec3> = ids.iter().map(|i| scene.truth[i].center()).collect();
    let sim = umeyama(&est, &tru).unwrap();
    let mut ce: Vec<f64> = est.iter().zip(&tru).map(|(a, b)| (sim.transform_point(a) - b).norm()).collect();
    ce.sort_by(|a, b| a.total_cmp(b));
    let extent = {
        let (mut lo, mut hi) = (tru[0], tru[0]);
        for t in &tru {
            lo = lo.inf(t);
            hi = hi.sup(t);
        }
        (hi - lo).norm()
    };
    // 회전은 전역 회전 하나(R_est ≈ R_true·G)로 따로 정렬.
    let mut sum = cumulus3d_core::Mat3::zeros();
    for i in &ids {
        sum += scene.truth[i].rotation_matrix().transpose() * rec.world_to_cam(*i).unwrap().rotation_matrix();
    }
    let svd = sum.svd(true, true);
    let g = svd.u.unwrap() * svd.v_t.unwrap();
    let mut re: Vec<f64> = ids
        .iter()
        .map(|i| {
            let d = rec.world_to_cam(*i).unwrap().rotation_matrix().transpose() * scene.truth[i].rotation_matrix() * g;
            ((d.trace() - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
        })
        .collect();
    re.sort_by(|a, b| a.total_cmp(b));
    (ce[ce.len() / 2] / extent, re[re.len() / 2])
}

#[test]
fn global_mapper_script_settings() {
    let scene = generate(&SceneConfig::default());
    let graph = graph_up_to(&scene, 13);
    let t = std::time::Instant::now();
    let out = global_mapper(&scene.store, &graph, &GlobalSfmOptions::script()).unwrap();
    eprintln!("global_mapper {:?}: {:?}", t.elapsed(), out.summary);
    assert!(out.failure.is_none(), "{:?}", out.failure);
    let rec = &out.reconstruction;
    rec.check_invariants().unwrap();
    let rate = rec.registered_image_count() as f64 / 39.0;
    assert!(rate >= 0.95, "등록률 {rate}");
    let (c, r) = pose_errors(rec, &scene);
    eprintln!("center rel {c} rot {r}° points {} mean reproj {}", rec.num_points3d(), rec.mean_reproj_error());
    assert!(c < 0.01 && r < 0.5, "center {c} rot {r}");
    assert!(rec.num_points3d() > 500);
}

fn incremental(refiner: PointRefiner, scope_new: bool, refine_pose: bool) -> (Reconstruction, Scene) {
    let scene = generate(&SceneConfig::default());
    let out = global_mapper(&scene.store, &graph_up_to(&scene, 13), &GlobalSfmOptions::script()).unwrap();
    let mut rec = out.reconstruction;
    for pos in 13..16 {
        let graph = graph_up_to(&scene, pos + 1);
        let ropts = RegistrationOptions { refine_pose, ..Default::default() };
        let rep = register_images(&mut rec, &scene.store, &graph, &ropts).unwrap();
        eprintln!("pos {pos}: {:?}", rep.attempts);
        for id in &scene.images_by_position[pos] {
            assert!(rec.is_image_registered(*id), "영상 {id} 등록 실패");
        }
        let scope =
            if scope_new { TriangulationScope::Images(scene.images_by_position[pos].clone()) } else { TriangulationScope::AllRegistered };
        let topts = PointTriangulatorOptions { refiner, scope, ..Default::default() };
        let before = rec.num_points3d();
        let t = std::time::Instant::now();
        let r = triangulate_points(&mut rec, &graph, &topts).unwrap();
        eprintln!("tri {:?} {:?} points {before} → {}", t.elapsed(), r, rec.num_points3d());
        rec.check_invariants().unwrap();
    }
    (rec, scene)
}

#[test]
fn register_and_triangulate_per_position() {
    let (rec, scene) = incremental(PointRefiner::PerPoint, false, false);
    assert_eq!(rec.registered_image_count(), 48);
    let (c, r) = pose_errors(&rec, &scene);
    eprintln!("center rel {c} rot {r} reproj {}", rec.mean_reproj_error());
    assert!(c < 0.01 && r < 0.5);
    assert!(rec.mean_reproj_error() < 2.0);
}

#[test]
fn incremental_scope_new_images_only() {
    let (rec, scene) = incremental(PointRefiner::PerPoint, true, false);
    assert_eq!(rec.registered_image_count(), 48);
    let (c, r) = pose_errors(&rec, &scene);
    assert!(c < 0.01 && r < 0.5);
}

#[test]
fn register_with_ba_pose_refinement() {
    let (rec, scene) = incremental(PointRefiner::BundleAdjuster, false, true);
    assert_eq!(rec.registered_image_count(), 48);
    let (c, r) = pose_errors(&rec, &scene);
    assert!(c < 0.01 && r < 0.5);
}

#[test]
fn triangulation_accuracy_with_true_poses() {
    let scene = generate(&SceneConfig { num_positions: 8, ..Default::default() });
    let graph = graph_up_to(&scene, 8);
    let mut rec = cumulus3d_sfm::global_mapper::init_reconstruction(&scene.store, &graph).unwrap();
    for (id, pose) in &scene.truth {
        rec.set_world_to_cam(*id, *pose).unwrap();
        rec.register_image(*id).unwrap();
    }
    let opts = PointTriangulatorOptions { refiner: PointRefiner::PerPoint, clear_points: true, ..Default::default() };
    triangulate_points(&mut rec, &graph, &opts).unwrap();
    let mut errs = Vec::new();
    let mut wrong = 0;
    for (_, p) in rec.points3d() {
        let t = &p.track[0];
        let Some(&j) = scene.kp_point.get(&(t.image_id, t.point2d_idx)) else { continue };
        let e = (p.xyz - scene.points[j]).norm();
        if e > 1.0 {
            wrong += 1;
        }
        errs.push(e);
    }
    errs.sort_by(|a, b| a.total_cmp(b));
    eprintln!("points {} median err {} m, >1m {}", errs.len(), errs[errs.len() / 2], wrong);
    assert!(errs.len() > 1000);
    assert!(errs[errs.len() / 2] < 0.05);
    assert!((wrong as f64) < 0.01 * errs.len() as f64);
}

#[test]
fn global_mapper_default_with_ba() {
    let scene = generate(&SceneConfig::default());
    let out = global_mapper(&scene.store, &graph_up_to(&scene, 13), &GlobalSfmOptions::default()).unwrap();
    assert!(out.failure.is_none(), "{:?}", out.failure);
    let (c, r) = pose_errors(&out.reconstruction, &scene);
    let rec = &out.reconstruction;
    eprintln!("BA 3회: center {c} rot {r} points {} reproj {}", rec.num_points3d(), rec.mean_reproj_error());
    assert_eq!(rec.registered_image_count(), 39);
    assert!(c < 0.01 && r < 0.5);
    assert!(rec.mean_reproj_error() < 1.0);
}
