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

//! Verification of descriptor matching, pair list, and store pipeline.

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
    // Non-negative sparse vector like SIFT.
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
    // Image 2: 200 random + 99 noisy copies (3k of image 1) + 2 identical copies of image 1[21].
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
            continue; // used as the 2 identical copies below
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

    // Tile kernel = sequential reference implementation (bit-identical).
    let (n1, n2) = (d1.len(), d2.len());
    let (rc, cc) = cpu.top2(d1.as_slice(), n1, d2.as_slice(), n2);
    let (rn, cn) = top2_naive(d1.as_slice(), n1, d2.as_slice(), n2);
    assert_eq!(rc, rn);
    assert_eq!(cc, cn);
    // Same for small tiles and odd blocks.
    let odd = CpuMatcher { row_block: 7, col_tile: 5 };
    let (ro, co) = odd.top2(d1.as_slice(), n1, d2.as_slice(), n2);
    assert_eq!(ro, rn);
    assert_eq!(co, cn);

    // GPU rule vs CPU brute-force rule: exact match on data without boundary values.
    let cpu_rule = DescriptorMatchOptions { rule: AcceptRule::CpuBruteForce, ..Default::default() };
    assert_eq!(apply_tests(&rn, &cn, &opts, 8192), apply_tests(&rn, &cn, &cpu_rule, 8192));

    // Cross-check off -> the number of matches is the same or larger.
    let no_cc = DescriptorMatchOptions { cross_check: false, ..Default::default() };
    assert!(cpu.match_descriptors(&d1, &d2, &no_cc, 8192).len() >= m.len());
    // Truncation of the maximum match count (use only the leading descriptors + truncate the output).
    let few = cpu.match_descriptors(&d1, &d2, &opts, 250);
    assert!(few.len() <= 250);
    assert!(few.iter().all(|f| f.idx1 < 250 && f.idx2 < 250));
}

#[test]
fn ratio_boundary_rules() {
    use cumulus3d_matching::Top2;
    let opts = DescriptorMatchOptions { cross_check: false, ..Default::default() };
    let cpu_rule = DescriptorMatchOptions { rule: AcceptRule::CpuBruteForce, ..opts.clone() };
    // Near exactly θ1 = 0.7: the boundary where GPU (<) and CPU (≤) may differ is hard to hit exactly since dot products are integers ->
    // identical dot products (θ1 = θ2) are rejected by both.
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

/// 3 images in the store + keypoints/descriptors of common 3D points.
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
        // Shuffle the order per image so the indices differ.
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
        // 200 noise features.
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
    // First position: (1,2). Also insert in the id1 > id2 direction: (3,1).
    let s = match_pairs(&store, &[(1, 2), (2, 1), (3, 1)], &opts, &cpu).unwrap();
    assert_eq!(s.num_matched, 2);
    assert_eq!(s.num_valid_geometries, 2);
    let g12 = store.get_two_view(1, 2).unwrap();
    assert_eq!(g12.config, C::Calibrated);
    assert!(g12.inlier_matches.len() > 300);
    // (3,1) is stored reversed as small id -> large id: when reading, the (1,3) direction swaps columns and transposes F.
    let g31 = store.get_two_view(3, 1).unwrap();
    let g13 = store.get_two_view(1, 3).unwrap();
    assert_eq!(g13.inlier_matches[0], g31.inlier_matches[0].swapped());
    assert!((g13.f.unwrap() - g31.f.unwrap().transpose()).norm() < 1e-12);
    // Raw matches follow the same rule.
    let m31 = store.read_matches(3, 1).unwrap();
    let m13 = store.read_matches(1, 3).unwrap();
    assert_eq!(m13[0], m31[0].swapped());

    // Re-feeding the accumulated list: existing pairs are skipped, only the new pair (2,3) is processed.
    let s = match_pairs(&store, &[(1, 2), (3, 1), (2, 3)], &opts, &cpu).unwrap();
    assert_eq!(s.num_skipped_existing, 2);
    assert_eq!(s.num_matched, 1);

    // Rule 6: only raw matches exist -> verification only.
    store.remove_two_view(1, 2);
    let s = match_pairs(&store, &[(1, 2)], &opts, &cpu).unwrap();
    assert_eq!(s.num_verified_only, 1);
    assert!(store.contains_two_view(1, 2));
    // Rule 5: only geometry exists -> match again.
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
    // If matches are below 15, the raw matches are an empty list and geometry is UNDEFINED, recorded as "always one row".
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
