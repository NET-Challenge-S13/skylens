/*
 * matching.rs
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

//! 기술자 매칭·짝 목록·저장소 파이프라인 검증.

use cumulus3d_core::{
    images_of_pair, pair_id_of, Camera, CameraModelKind, Descriptors, FeatureMatch, FeatureStore, Keypoint, TwoViewGeometryConfig as C,
    Vec2, Vec3, DESCRIPTOR_DIM,
};
use cumulus3d_matching::descriptor::{apply_tests, dot_to_angle, top2_naive};
use cumulus3d_matching::*;
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

fn quantize(v: &[f64]) -> [u8; 128] {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut out = [0u8; 128];
    for k in 0..128 {
        out[k] = (512.0 * v[k] / n).round().clamp(0.0, 255.0) as u8;
    }
    out
}

fn rand_desc(r: &mut Pcg64) -> Vec<f64> {
    // SIFT 처럼 음이 아니고 성긴 벡터.
    (0..128).map(|_| if r.random_range(0.0..1.0) < 0.4 { r.random_range(0.0..1.0f64).powi(2) } else { 0.0 }).collect()
}

fn perturb(r: &mut Pcg64, v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| (x + r.random_range(-s..s)).max(0.0)).collect()
}

#[test]
fn pair_ids() {
    assert_eq!(pair_id_of(1, 2).unwrap(), 2147483649);
    assert_eq!(pair_id_of(2, 1).unwrap(), 2147483649);
    assert_eq!(images_of_pair(2147483649), (1, 2));
}

#[test]
fn angle_conversion() {
    assert_eq!(dot_to_angle(262144), 0.0);
    assert_eq!(dot_to_angle(300000), 0.0);
    assert!((dot_to_angle(0) - std::f32::consts::FRAC_PI_2).abs() < 1e-7);
}

#[test]
fn descriptor_matching() {
    let mut r = Pcg64::seed_from_u64(1);
    let base: Vec<Vec<f64>> = (0..300).map(|_| rand_desc(&mut r)).collect();
    let mut d1 = Descriptors::new();
    for b in &base {
        d1.push(&quantize(b));
    }
    // 영상2: 무작위 200 + 잡음 복제 99(영상1 의 3k) + 영상1[21] 의 동일 사본 2개.
    let mut d2 = Descriptors::new();
    let mut expected = Vec::new();
    let mut idx2 = 0u32;
    for _ in 0..200 {
        d2.push(&quantize(&rand_desc(&mut r)));
        idx2 += 1;
    }
    for k in 0..100u32 {
        let src = (k * 3) as usize;
        if src == 21 {
            continue; // 아래에서 동일 사본 2개로 사용
        }
        d2.push(&quantize(&perturb(&mut r, &base[src], 0.01)));
        expected.push(FeatureMatch::new(src as u32, idx2));
        idx2 += 1;
    }
    let dup = quantize(&base[21]);
    d2.push(&dup);
    d2.push(&dup);
    expected.sort();

    let cpu = CpuMatcher::default();
    let opts = DescriptorMatchOptions::default();
    let m = cpu.match_descriptors(&d1, &d2, &opts, 8192);
    assert_eq!(m, expected, "복제 쌍만 정확히 매칭, 동일 사본 2개는 비율 검사로 거부");

    // 타일 커널 = 순차 기준 구현(비트 일치).
    let (n1, n2) = (d1.len(), d2.len());
    let (rc, cc) = cpu.top2(d1.as_slice(), n1, d2.as_slice(), n2);
    let (rn, cn) = top2_naive(d1.as_slice(), n1, d2.as_slice(), n2);
    assert_eq!(rc, rn);
    assert_eq!(cc, cn);
    // 작은 타일·홀수 블록에서도 동일.
    let odd = CpuMatcher { row_block: 7, col_tile: 5 };
    let (ro, co) = odd.top2(d1.as_slice(), n1, d2.as_slice(), n2);
    assert_eq!(ro, rn);
    assert_eq!(co, cn);

    // GPU 규칙 vs CPU 무차별 규칙: 경계값 없는 자료에서 완전 일치.
    let cpu_rule = DescriptorMatchOptions { rule: AcceptRule::CpuBruteForce, ..Default::default() };
    assert_eq!(apply_tests(&rn, &cn, &opts, 8192), apply_tests(&rn, &cn, &cpu_rule, 8192));

    // 교차 검사 끔 → 매칭이 같거나 많다.
    let no_cc = DescriptorMatchOptions { cross_check: false, ..Default::default() };
    assert!(cpu.match_descriptors(&d1, &d2, &no_cc, 8192).len() >= m.len());
    // 최대 매칭 수 자르기(앞쪽 기술자만 사용 + 출력 자르기).
    let few = cpu.match_descriptors(&d1, &d2, &opts, 250);
    assert!(few.len() <= 250);
    assert!(few.iter().all(|f| f.idx1 < 250 && f.idx2 < 250));
}

#[test]
fn ratio_boundary_rules() {
    use cumulus3d_matching::Top2;
    let opts = DescriptorMatchOptions { cross_check: false, ..Default::default() };
    let cpu_rule = DescriptorMatchOptions { rule: AcceptRule::CpuBruteForce, ..opts.clone() };
    // θ1 = 0.7 정확히 근처: GPU (<) 와 CPU (≤) 가 다를 수 있는 경계는 내적 정수라 정확히 맞추기 어려움 →
    // 동일 내적(θ1 = θ2)은 둘 다 거부.
    let t = Top2 { best: 250000, idx: 0, second: 250000 };
    assert!(apply_tests(&[t], &[], &opts, 10).is_empty());
    assert!(apply_tests(&[t], &[], &cpu_rule, 10).is_empty());
    let t = Top2 { best: 260000, idx: 3, second: 100000 };
    assert_eq!(apply_tests(&[t], &[], &opts, 10), vec![FeatureMatch::new(0, 3)]);
}

#[test]
fn pair_list_parser() {
    let names = ["a.jpg", "camF/0001.jpg", "camF/0002.jpg", "c.jpg"];
    let lookup = |n: &str| names.iter().position(|x| *x == n).map(|i| i as u32 + 1);
    let text = "# 주석\n\n  a.jpg camF/0001.jpg  \ncamF/0001.jpg a.jpg\nc.jpg c.jpg\nmissing.jpg a.jpg\ncamF/0002.jpg c.jpg extra stuff\nc.jpg camF/0002.jpg\na.jpg\tc.jpg\nc.jpg a.jpg\n";
    let l = parse_pair_list(text, lookup);
    assert_eq!(l.pairs, vec![(1, 2), (3, 4), (4, 1)]);
    assert!(l.missing_names.contains(&"missing.jpg".to_string()));
    assert!(l.missing_names.iter().any(|n| n.contains('\t')));
}

/// 저장소에 영상 3장 + 공통 3D 점의 키포인트/기술자.
fn build_store() -> FeatureStore {
    let store = FeatureStore::new();
    let mut cam = Camera::new(0, CameraModelKind::OpenCv, 2000, 1500, vec![1000.0, 1000.0, 1000.0, 750.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
    cam.focal_from_prior = true;
    cam.camera_id = cumulus3d_core::INVALID_CAMERA_ID;
    let cid = store.add_camera(cam.clone()).unwrap();
    let mut r = Pcg64::seed_from_u64(42);
    let pts: Vec<Vec3> =
        (0..800).map(|_| Vec3::new(r.random_range(-6.0..6.0), r.random_range(-5.0..5.0), r.random_range(6.0..15.0))).collect();
    let descs: Vec<Vec<f64>> = (0..800).map(|_| rand_desc(&mut r)).collect();
    for (k, cx) in [0.0f64, 0.6, 1.2].iter().enumerate() {
        let id = store.add_image(&format!("cam/{k}.jpg"), cid).unwrap();
        let mut kps = Vec::new();
        let mut d = Descriptors::new();
        // 영상마다 순서를 섞어 인덱스가 다르게.
        let mut order: Vec<usize> = (0..800).collect();
        for i in (1..800).rev() {
            let j = r.random_range(0..=i);
            order.swap(i, j);
        }
        for &i in &order {
            let xc = pts[i] - Vec3::new(*cx, 0.0, 0.0);
            let p = cam.cam_to_img(&xc).unwrap_or(Vec2::new(-1.0, -1.0));
            if p.x < 0.0 || p.y < 0.0 || p.x > 2000.0 || p.y > 1500.0 {
                continue;
            }
            kps.push(Keypoint::new((p.x + 0.3 * r.random_range(-1.0..1.0)) as f32, (p.y + 0.3 * r.random_range(-1.0..1.0)) as f32));
            d.push(&quantize(&perturb(&mut r, &descs[i], 0.01)));
        }
        // 잡음 특징 200개.
        for _ in 0..200 {
            kps.push(Keypoint::new(r.random_range(0.0..2000.0f32), r.random_range(0.0..1500.0f32)));
            d.push(&quantize(&rand_desc(&mut r)));
        }
        assert_eq!(d.as_slice().len(), kps.len() * DESCRIPTOR_DIM);
        store.set_keypoints(id, kps);
        store.set_descriptors(id, d);
    }
    store
}

#[test]
fn store_pipeline_incremental() {
    let store = build_store();
    let mut opts = PairMatchingOptions::default();
    opts.geometry.ransac.random_seed = Some(3);
    let cpu = CpuMatcher::default();
    // 첫 위치: (1,2). id1 > id2 방향으로도 넣는다: (3,1).
    let s = match_pairs(&store, &[(1, 2), (2, 1), (3, 1)], &opts, &cpu).unwrap();
    assert_eq!(s.num_matched, 2);
    assert_eq!(s.num_valid_geometries, 2);
    let g12 = store.get_two_view(1, 2).unwrap();
    assert_eq!(g12.config, C::Calibrated);
    assert!(g12.inlier_matches.len() > 300);
    // (3,1) 은 작은 id → 큰 id 로 반전 저장: 읽을 때 (1,3) 방향은 열 교환, F 전치.
    let g31 = store.get_two_view(3, 1).unwrap();
    let g13 = store.get_two_view(1, 3).unwrap();
    assert_eq!(g13.inlier_matches[0], g31.inlier_matches[0].swapped());
    assert!((g13.f.unwrap() - g31.f.unwrap().transpose()).norm() < 1e-12);
    // 원시 매칭도 같은 규칙.
    let m31 = store.read_matches(3, 1).unwrap();
    let m13 = store.read_matches(1, 3).unwrap();
    assert_eq!(m13[0], m31[0].swapped());

    // 누적 목록 재투입: 기존 짝은 건너뜀, 새 짝 (2,3) 만 처리.
    let s = match_pairs(&store, &[(1, 2), (3, 1), (2, 3)], &opts, &cpu).unwrap();
    assert_eq!(s.num_skipped_existing, 2);
    assert_eq!(s.num_matched, 1);

    // 규칙 6: 원시 매칭만 있음 → 검증만.
    store.remove_two_view(1, 2);
    let s = match_pairs(&store, &[(1, 2)], &opts, &cpu).unwrap();
    assert_eq!(s.num_verified_only, 1);
    assert!(store.contains_two_view(1, 2));
    // 규칙 5: 기하만 있음 → 다시 매칭.
    store.delete_matches(1, 2);
    let s = match_pairs(&store, &[(1, 2)], &opts, &cpu).unwrap();
    assert_eq!(s.num_matched, 1);
}

#[test]
fn pair_list_file_roundtrip() {
    let store = build_store();
    let dir = std::env::temp_dir().join(format!("cumulus3d_matching_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("pairs.txt");
    std::fs::write(&path, "cam/0.jpg cam/1.jpg\ncam/2.jpg cam/1.jpg\nnone.jpg cam/1.jpg\n").unwrap();
    let mut opts = PairMatchingOptions::default();
    opts.geometry.ransac.random_seed = Some(1);
    let (list, stats) = match_pair_list_file(&store, &path, &opts, &CpuMatcher::default()).unwrap();
    assert_eq!(list.pairs, vec![(1, 2), (3, 2)]);
    assert_eq!(list.missing_names, vec!["none.jpg".to_string()]);
    assert_eq!(stats.num_matched, 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn few_matches_store_empty() {
    // 매칭 15 미만이면 원시 매칭은 빈 목록, 기하는 UNDEFINED 로 "항상 한 줄" 기록.
    let store = FeatureStore::new();
    let mut cam = Camera::new(0, CameraModelKind::Pinhole, 100, 100, vec![100.0, 100.0, 50.0, 50.0]).unwrap();
    cam.camera_id = cumulus3d_core::INVALID_CAMERA_ID;
    let cid = store.add_camera(cam).unwrap();
    let mut r = Pcg64::seed_from_u64(5);
    let shared: Vec<Vec<f64>> = (0..10).map(|_| rand_desc(&mut r)).collect();
    for k in 0..2 {
        let id = store.add_image(&format!("{k}.jpg"), cid).unwrap();
        let mut d = Descriptors::new();
        let mut kps = vec![];
        for s in &shared {
            d.push(&quantize(&perturb(&mut r, s, 0.01)));
            kps.push(Keypoint::new(r.random_range(0.0..100.0f32), r.random_range(0.0..100.0f32)));
        }
        store.set_descriptors(id, d);
        store.set_keypoints(id, kps);
    }
    let s = match_pairs(&store, &[(1, 2)], &PairMatchingOptions::default(), &CpuMatcher::default()).unwrap();
    assert_eq!(s.num_matched, 1);
    assert!(store.read_matches(1, 2).unwrap().is_empty());
    let g = store.get_two_view(1, 2).unwrap();
    assert_eq!(g.config, C::Undefined);
    assert!(g.inlier_matches.is_empty());
}
