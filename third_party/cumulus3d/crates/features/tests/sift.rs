/*
 * sift.rs
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

//! SIFT 검증 + 실제 사진(있을 때만).

use cumulus3d_features::sift::pyramid::{self, ScaleSpace};
use cumulus3d_features::*;
use std::f32::consts::PI;

fn noise_field(w: usize, h: usize, seed: u64, sigma: f32, contrast: f32) -> GrayImage {
    let mut s = seed;
    let mut noise = vec![0f32; w * h];
    for v in noise.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = (s >> 40) as f32 / (1u64 << 24) as f32;
    }
    let mut out = vec![0f32; w * h];
    let mut tmp = vec![0f32; w * h];
    pyramid::gaussian_blur(&noise, &mut out, &mut tmp, w, h, sigma);
    GrayImage::from_f32(w, h, |x, y| (out[y * w + x] - 0.5) * contrast + 0.5)
}

fn unlimited() -> SiftOptions {
    SiftOptions { max_num_features: 0, ..Default::default() }
}

fn l2u8(a: &[u8], b: &[u8]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| (x as f32 - y as f32).powi(2)).sum::<f32>().sqrt()
}

#[test]
fn scale_space_table() {
    let ss = ScaleSpace::new(3);
    let expect = [1.6000, 2.0159, 2.5398, 3.2000, 4.0317, 5.0797];
    for (i, e) in expect.iter().enumerate() {
        assert!((ss.sigma(i as f32 - 1.0) - e).abs() < 1e-4, "l={}", i as i32 - 1);
    }
    let inc = [1.2263, 1.5450, 1.9466, 2.4525, 3.0900];
    for (l, e) in inc.iter().enumerate() {
        assert!((ss.sigma_inc(l as i32) - e).abs() < 1e-4, "l={l}");
    }
    let init = (1.6f32 * 1.6 - 1.0).sqrt();
    assert!((init - 1.249).abs() < 1e-3);
    let widths: Vec<usize> = [init, inc[0], inc[1], inc[2], inc[3], inc[4]].iter().map(|&s| pyramid::gaussian_kernel(s).len()).collect();
    assert_eq!(widths, vec![11, 11, 13, 17, 21, 25]);
    assert_eq!(pyramid::auto_num_octaves(6400, 4800), 9);
    let k = pyramid::gaussian_kernel(2.0);
    assert!((k.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

#[test]
fn upsample_rule() {
    // 입력 c ↔ 업샘플 2c, 홀수 위치는 평균.
    let src = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]; // 3x2
    let mut dst = vec![0f32; 6 * 4];
    pyramid::upsample2(&src, 3, 2, &mut dst);
    assert_eq!(&dst[0..6], &[0.0, 0.5, 1.0, 1.5, 2.0, 2.0]);
    assert_eq!(&dst[6..12], &[1.5, 2.0, 2.5, 3.0, 3.5, 3.5]);
    assert_eq!(&dst[12..18], &[3.0, 3.5, 4.0, 4.5, 5.0, 5.0]);
}

#[test]
fn gaussian_blob_position_and_scale() {
    let sb = 8.0f32;
    let img = GrayImage::from_f32(512, 512, |x, y| {
        let dx = x as f32 + 0.5 - 256.5;
        let dy = y as f32 + 0.5 - 256.5;
        0.5 + 0.4 * (-(dx * dx + dy * dy) / (2.0 * sb * sb)).exp()
    });
    let out = CpuSift::new().extract(&img, &unlimited()).unwrap();
    let best = out
        .features
        .iter()
        .min_by(|a, b| {
            let da = (a.x - 256.5).hypot(a.y - 256.5);
            let db = (b.x - 256.5).hypot(b.y - 256.5);
            da.total_cmp(&db)
        })
        .expect("블롭 검출");
    eprintln!("blob kp: {best:?}, total {}", out.len());
    assert!((best.x - 256.5).abs() < 0.3 && (best.y - 256.5).abs() < 0.3, "{best:?}");
    assert!(best.scale > 0.75 * sb && best.scale < 1.25 * sb, "scale {}", best.scale);
}

#[test]
fn translation_covariance() {
    let big = noise_field(560, 560, 11, 2.5, 2.5);
    let crop = |ox: usize, oy: usize| GrayImage::from_f32(512, 512, |x, y| big.get(x + ox, y + oy) as f32 / 255.0);
    let (dx, dy) = (16usize, 32usize);
    let a = crop(dx, dy);
    let b = crop(0, 0);
    let sift = CpuSift::new();
    let fa = sift.extract(&a, &unlimited()).unwrap();
    let fb = sift.extract(&b, &unlimited()).unwrap();
    let mut checked = 0;
    let mut bad = 0;
    for (i, f) in fa.features.iter().enumerate() {
        // 이동이 정수인 옥타브(16 = 2^4 → o ≤ 3), 경계에서 먼 작은 스케일만.
        if f.octave > 2 || f.scale > 5.0 || f.x < 130.0 || f.y < 130.0 || f.x > 382.0 || f.y > 382.0 {
            continue;
        }
        let (tx, ty) = (f.x + dx as f32, f.y + dy as f32);
        let m = fb.features.iter().position(|g| {
            (g.x - tx).abs() < 0.05 && (g.y - ty).abs() < 0.05 && (g.scale - f.scale).abs() < 1e-3 && g.orientation == f.orientation
        });
        checked += 1;
        match m {
            Some(j) if fa.descriptors.row(i) == fb.descriptors.row(j) => {}
            Some(j) => {
                bad += 1;
                eprintln!(
                    "desc diff o={} j={} x={} y={} d={}",
                    f.octave,
                    f.level,
                    f.x,
                    f.y,
                    l2u8(fa.descriptors.row(i), fb.descriptors.row(j))
                );
            }
            None => {
                bad += 1;
                let near = fb.features.iter().map(|g| (g.x - tx).hypot(g.y - ty)).fold(1e9f32, f32::min);
                eprintln!("missing o={} j={} x={} y={} near={near}", f.octave, f.level, f.x, f.y);
            }
        }
    }
    eprintln!("translation: checked {checked}, mismatched {bad}");
    assert!(checked > 50);
    assert_eq!(bad, 0);
}

#[test]
fn rotation_invariance_90() {
    let img = noise_field(384, 320, 5, 3.0, 3.0);
    let rot = img.rotate_ccw(1);
    let sift = CpuSift::new();
    let fa = sift.extract(&img, &unlimited()).unwrap();
    let fb = sift.extract(&rot, &unlimited()).unwrap();
    let w = img.width as f32;
    let mut dists = Vec::new();
    let mut ori_err = Vec::new();
    for (i, f) in fa.features.iter().enumerate() {
        if f.x < 40.0 || f.y < 40.0 || f.x > w - 40.0 || f.y > img.height as f32 - 40.0 || f.scale > 8.0 {
            continue;
        }
        let (tx, ty) = (f.y, w - f.x);
        // 같은 위치·스케일 중 방향이 가장 가까운 것.
        let target = f.orientation + PI / 2.0;
        let best = fb
            .features
            .iter()
            .enumerate()
            .filter(|(_, g)| (g.x - tx).hypot(g.y - ty) < 0.5 && (g.scale / f.scale - 1.0).abs() < 0.1)
            .map(|(j, g)| {
                let d = (g.orientation - target).rem_euclid(2.0 * PI);
                (j, d.min(2.0 * PI - d))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((j, d)) = best {
            dists.push(l2u8(fa.descriptors.row(i), fb.descriptors.row(j)));
            ori_err.push(d);
        }
    }
    let n = dists.len();
    let mean = dists.iter().sum::<f32>() / n as f32;
    let mut oe = ori_err.clone();
    oe.sort_by(|a, b| a.total_cmp(b));
    eprintln!("rotation: matched {n} of {} , mean desc dist {mean}, median ori err {}", fa.len(), oe[n / 2]);
    assert!(n > 100);
    assert!(mean < 60.0, "{mean}");
    assert!(oe[n / 2] < 0.1);
}

#[test]
fn descriptor_norms() {
    let img = noise_field(256, 256, 3, 2.0, 2.0);
    let out = CpuSift::new().extract(&img, &unlimited()).unwrap();
    assert!(out.len() > 100);
    for i in 0..out.len() {
        let n = out.descriptors.row(i).iter().map(|&v| (v as f32).powi(2)).sum::<f32>().sqrt();
        assert!((n - 512.0).abs() <= 10.0, "norm {n}");
    }
    assert_eq!(out.descriptors.len(), out.features.len());
    // L2 정규화 옵션도 512 근처
    let o2 = SiftOptions { normalization: DescriptorNormalization::L2, ..unlimited() };
    let out = CpuSift::new().extract(&img, &o2).unwrap();
    for i in 0..out.len() {
        let n = out.descriptors.row(i).iter().map(|&v| (v as f32).powi(2)).sum::<f32>().sqrt();
        assert!((n - 512.0).abs() <= 10.0, "norm {n}");
    }
}

#[test]
fn edge_suppression() {
    // 세로 직선 에지(가장자리까지 이어짐).
    let img = GrayImage::from_f32(256, 256, |x, _| if x < 128 { 0.2 } else { 0.8 });
    let out = CpuSift::new().extract(&img, &unlimited()).unwrap();
    let on_edge = out
        .features
        .iter()
        .filter(|f| (f.x - 128.0).abs() < 2.0 * f.scale + 2.0 && f.y > 3.0 * f.scale + 4.0 && f.y < 256.0 - 3.0 * f.scale - 4.0)
        .count();
    eprintln!("edge: total {}, on edge {on_edge}", out.len());
    assert_eq!(on_edge, 0);
}

fn level_counts(out: &SiftOutput) -> Vec<((i32, u32), usize)> {
    let mut m = std::collections::BTreeMap::new();
    for f in &out.features {
        *m.entry((f.octave, f.level)).or_insert(0usize) += 1;
    }
    m.into_iter().collect()
}

#[test]
fn max_features_rules() {
    let img = noise_field(1024, 768, 9, 1.5, 3.0);
    let sift = CpuSift::new();
    let all = sift.extract(&img, &unlimited()).unwrap();
    eprintln!("unlimited: {}", all.len());
    assert!(all.len() > 8192);
    let compat = sift.extract(&img, &SiftOptions::default()).unwrap();
    let counts = level_counts(&compat);
    let n = compat.len();
    // 가장 미세한 남은 레벨 = (옥타브, 레벨) 사전순 최소.
    let finest = counts.first().unwrap().1;
    eprintln!("compat: {n}, levels {counts:?}");
    assert!(n >= 8192);
    assert!(n - finest <= 8192);
    // 레벨은 통째로: 남은 레벨의 개수는 무제한 결과와 같다.
    let all_counts: std::collections::BTreeMap<_, _> = level_counts(&all).into_iter().collect();
    for (lv, c) in &counts {
        assert_eq!(all_counts[lv], *c, "레벨 {lv:?}");
    }
    let topk = sift.extract(&img, &SiftOptions { selection: FeatureSelection::TopK, ..Default::default() }).unwrap();
    assert_eq!(topk.len(), 8192);
    assert_eq!(topk.descriptors.len(), 8192);
    let grid = sift.extract(&img, &SiftOptions { selection: FeatureSelection::SpatialGrid { cells: 8 }, ..Default::default() }).unwrap();
    assert_eq!(grid.len(), 8192);
}

#[test]
fn improvement_options_run() {
    let img = noise_field(300, 200, 21, 2.0, 2.5);
    let opts = SiftOptions {
        refinement_iterations: 5,
        reject_singular_refinement: true,
        orientation_bin_interpolation: true,
        truncate_width_to_4: false,
        upright: false,
        ..unlimited()
    };
    let out = CpuSift::new().extract(&img, &opts).unwrap();
    assert!(out.len() > 50);
    for f in &out.features {
        assert!(f.x >= 0.0 && f.x <= 300.0 && f.y >= 0.0 && f.y <= 200.0);
    }
    let up = CpuSift::new().extract(&img, &SiftOptions { upright: true, ..unlimited() }).unwrap();
    assert!(up.features.iter().all(|f| f.orientation == 0.0));
}

#[test]
fn deterministic() {
    let img = noise_field(320, 240, 2, 2.0, 2.5);
    let s = CpuSift::new();
    let a = s.extract(&img, &SiftOptions::default()).unwrap();
    let b = s.extract(&img, &SiftOptions::default()).unwrap();
    assert_eq!(a.features, b.features);
    assert_eq!(a.descriptors, b.descriptors);
}

#[test]
fn real_photo_if_available() {
    // CUMULUS3D_TEST_PHOTO=<jpg/png 경로> 가 있을 때만 실행.
    let Some(path) = std::env::var_os("CUMULUS3D_TEST_PHOTO").map(std::path::PathBuf::from).filter(|p| p.exists()) else {
        eprintln!("CUMULUS3D_TEST_PHOTO 없음 → 건너뜀");
        return;
    };
    let path = path.as_path();
    let g = read_gray(path).unwrap();
    let out = CpuSift::new().extract(&g, &SiftOptions::default()).unwrap();
    eprintln!("{}: {}x{} → {} features", path.display(), g.width, g.height, out.len());
    assert!(out.len() > 100);
    for f in &out.features {
        assert!(f.x >= 0.0 && f.x <= g.width as f32 && f.y >= 0.0 && f.y <= g.height as f32);
    }
}
