/*
 * host_parts.rs
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

//! 커널 바깥 부분: 이웃 선택, 깊이 범위, 융합(참 깊이맵 입력), PLY, 상대 기하, 캐시.

use cumulus3d_dense::fusion::{fuse, FusionInput};
use cumulus3d_dense::neighbors::{depth_ranges, PairStats};
use cumulus3d_dense::synthetic::{make_scene, SynthConfig, SynthScene};
use cumulus3d_dense::{
    pair_geometry, CachedDepth, DenseOutput, DenseScene, DenseView, DepthMapCache, FusionMode, FusionParams, ImageBuffer, ScenePoint,
};
use std::sync::{Arc, OnceLock};

fn synth() -> &'static SynthScene {
    static S: OnceLock<SynthScene> = OnceLock::new();
    S.get_or_init(|| make_scene(&SynthConfig::default()))
}

fn bare_view(c: [f64; 3]) -> DenseView {
    // 아래를 보는 카메라(세계 z 위).
    let r = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
    let t = [-(r[0][0] * c[0]), -(r[1][1] * c[1]), -(r[2][2] * c[2])];
    let img = Arc::new(ImageBuffer::new(4, 4, 3));
    DenseView {
        image_id: 0,
        name: String::new(),
        width: 4,
        height: 4,
        k: [100.0, 100.0, 1.5, 1.5],
        r,
        t,
        gray: Arc::new(vec![0; 16]),
        color: img,
    }
}

#[test]
fn neighbor_selection_by_shared_count_and_angle() {
    // 뷰 0 (원점 위 10 m), 1 (옆 3 m: 큰 각), 2 (옆 0.05 m: 각 < 1°), 3 (옆 −3 m).
    let views = vec![bare_view([0.0, 0.0, 10.0]), bare_view([3.0, 0.0, 10.0]), bare_view([0.05, 0.0, 10.0]), bare_view([-3.0, 0.0, 10.0])];
    let mut points = Vec::new();
    let add = |pts: &mut Vec<ScenePoint>, n: usize, a: u32, b: u32| {
        for k in 0..n {
            pts.push(ScenePoint { xyz: [k as f64 * 0.1, 0.0, 0.0], views: vec![a, b] });
        }
    };
    add(&mut points, 5, 0, 1);
    add(&mut points, 8, 0, 2);
    add(&mut points, 3, 0, 3);
    add(&mut points, 1, 1, 3);
    let scene = DenseScene { views, points };
    let st = PairStats::new(&scene);
    let a1 = 1f64.to_radians();
    assert_eq!(st.select(0, 10, a1), vec![1, 3]);
    assert_eq!(st.select(0, 1, a1), vec![1]);
    assert_eq!(st.select(0, 10, 0.0), vec![2, 1, 3]);
    assert_eq!(st.select(1, 10, a1), vec![0, 3]);
    let r = depth_ranges(&scene);
    assert!((r[0].unwrap().0 - 7.5).abs() < 1e-9 && (r[0].unwrap().1 - 12.5).abs() < 1e-9);
}

#[test]
fn pair_geometry_maps_points() {
    let s = synth();
    let sc = &s.scene;
    let g = pair_geometry(sc, 0, 4);
    let x = [1.3, -0.7, 0.2];
    let a = sc.views[0].to_cam(&x);
    let b = sc.views[4].to_cam(&x);
    for (i, bi) in b.iter().enumerate() {
        let p = (0..3).map(|k| g.r[i * 3 + k] as f64 * a[k]).sum::<f64>() + g.t[i] as f64;
        assert!((p - bi).abs() < 1e-4, "{p} {bi}");
    }
    let c4 = sc.views[0].to_cam(&sc.views[4].center());
    for (gc, c) in g.center.iter().zip(c4) {
        assert!((*gc as f64 - c).abs() < 1e-4);
    }
}

#[test]
fn synthetic_scene_is_consistent() {
    let s = synth();
    assert_eq!(s.scene.views.len(), 9);
    assert!(s.scene.points.len() > 1000);
    // 참 깊이에서 세계로 되돌린 점은 장면 표면 위.
    let v = &s.scene.views[4];
    let (w, h) = (v.width, v.height);
    for (x, y) in [(10, 10), (w / 2, h / 2), (w - 5, h - 7), (100, 200)] {
        let d = s.depth[4][y * w + x] as f64;
        let p = v.to_world(&[(x as f64 - v.k[2]) / v.k[0] * d, (y as f64 - v.k[3]) / v.k[1] * d, d]);
        assert!(s.surface_distance(p) < 1e-3, "{p:?}");
        let n = s.normal[4][y * w + x];
        let ray = v.ray(x as f64, y as f64);
        assert!(n[0] as f64 * ray[0] + n[1] as f64 * ray[1] + n[2] as f64 * ray[2] < 0.0);
    }
}

fn overlap(scene: &DenseScene) -> Vec<Vec<usize>> {
    let st = PairStats::new(scene);
    (0..scene.views.len()).map(|v| st.select(v, 50, 0.0)).collect()
}

fn run_fusion(depth: &[Vec<f32>], normal: &[Vec<[f32; 3]>], p: &FusionParams, threads: usize) -> cumulus3d_dense::FusionOutput {
    let s = synth();
    let inputs: Vec<Option<FusionInput>> = depth.iter().zip(normal).map(|(d, n)| Some(FusionInput::plain(d, n))).collect();
    fuse(&s.scene, &inputs, &overlap(&s.scene), p, threads)
}

#[test]
fn fusion_of_true_maps_lies_on_surfaces() {
    let s = synth();
    let p = FusionParams { mode: FusionMode::Traversal, ..FusionParams::default() };
    let out = run_fusion(&s.depth, &s.normal, &p, 1);
    let npix = s.config.width * s.config.height;
    let n = out.cloud.len();
    assert!(n > npix / 10 && n < 9 * npix, "점 {n}");
    let mut far = 0;
    for x in &out.cloud.positions {
        if s.surface_distance([x[0] as f64, x[1] as f64, x[2] as f64]) > 1e-4 * s.config.altitude {
            far += 1;
        }
    }
    assert!(far * 1000 < n, "표면 밖 {far}/{n}");
    let multi = out.visibility.iter().filter(|v| v.len() >= 2).count();
    assert!(multi * 100 >= 95 * n, "가시 뷰 ≥ 2: {multi}/{n}");
    // 병렬(원자적 점유)도 같은 규칙: 점 수가 거의 같고 모두 표면 위.
    let par = run_fusion(&s.depth, &s.normal, &p, 0);
    let d = (par.cloud.len() as f64 - n as f64).abs() / n as f64;
    assert!(d < 0.05, "병렬 점 수 {} vs {n}", par.cloud.len());
    // 순차는 결정적.
    let again = run_fusion(&s.depth, &s.normal, &p, 1);
    assert_eq!(again.cloud, out.cloud);
}

fn merged_with_others(out: &cumulus3d_dense::FusionOutput, v: u32) -> usize {
    out.visibility.iter().filter(|vis| vis.contains(&v) && vis.len() > 1).count()
}

#[test]
fn fusion_depth_and_normal_tolerances() {
    let s = synth();
    let p = FusionParams { mode: FusionMode::Traversal, ..FusionParams::default() };
    // 뷰 4 깊이 2% 키움 → 1% 허용치에 걸려 다른 뷰와 합쳐지지 않음.
    let mut d = s.depth.clone();
    for x in d[4].iter_mut() {
        *x *= 1.02;
    }
    let base = merged_with_others(&run_fusion(&s.depth, &s.normal, &p, 1), 4);
    let scaled = merged_with_others(&run_fusion(&d, &s.normal, &p, 1), 4);
    assert!(scaled * 50 < base, "깊이 2%: {scaled} / {base}");
    // 법선 회전: 15° 는 합쳐지지 않고 8° 는 합쳐진다.
    let rot = |deg: f64| {
        let mut n = s.normal.clone();
        let (sa, ca) = deg.to_radians().sin_cos();
        for v in n[4].iter_mut() {
            let (x, z) = (v[0] as f64, v[2] as f64);
            *v = [(ca * x + sa * z) as f32, v[1], (-sa * x + ca * z) as f32];
        }
        n
    };
    // (카메라 y 축 회전이라 y 축에 가까운 법선(상자 옆면)은 덜 돈다.)
    assert!(merged_with_others(&run_fusion(&s.depth, &rot(15.0), &p, 1), 4) * 20 < base);
    assert!(merged_with_others(&run_fusion(&s.depth, &rot(8.0), &p, 1), 4) * 2 > base);
    // 최소 픽셀 수: 기여 4개인 점은 5 에서 나오지 않는다.
    let out = run_fusion(&s.depth, &s.normal, &FusionParams { min_num_pixels: 1, ..p.clone() }, 1);
    let mut counts = [0usize; 3];
    let strict = run_fusion(&s.depth, &s.normal, &p, 1);
    counts[0] = out.cloud.len();
    counts[1] = strict.cloud.len();
    counts[2] = strict.visibility.iter().filter(|v| v.len() < 2).count();
    assert!(counts[1] < counts[0], "{counts:?}");
}

#[test]
fn ply_output_layout() {
    let mut out = DenseOutput::default();
    out.cloud.positions = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    out.cloud.normals = vec![[0.0, 0.0, 1.0]; 2];
    out.cloud.colors = vec![[1, 2, 3], [4, 5, 6]];
    let p = std::env::temp_dir().join(format!("cumulus3d_dense_ply_{}/a.ply", std::process::id()));
    out.write_ply(&p).unwrap();
    let b = std::fs::read(&p).unwrap();
    let hdr = "ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    assert_eq!(&b[..hdr.len()], hdr.as_bytes());
    assert_eq!(b.len() - hdr.len(), 54);
    std::fs::remove_dir_all(p.parent().unwrap()).ok();
}

#[test]
fn depth_cache_capacity() {
    let c = DepthMapCache::with_capacity(2);
    let e = Arc::new(CachedDepth {
        width: 1,
        height: 1,
        raw_depth: vec![1.0],
        depth: vec![1.0],
        normal: vec![[0.0, 0.0, -1.0]],
        cost: vec![0.0],
    });
    for k in 0..3u64 {
        c.insert((k, 0), e.clone());
    }
    assert_eq!(c.len(), 2);
    assert!(c.get((0, 0)).is_none() && c.get((2, 0)).is_some());
}

#[test]
fn consistency_fusion_of_true_maps() {
    let s = synth();
    let p = FusionParams::default();
    assert_eq!(p.mode, FusionMode::Consistency);
    let out = run_fusion(&s.depth, &s.normal, &p, 0);
    let n = out.cloud.len();
    let npix = s.config.width * s.config.height;
    assert!(n > npix && n < 9 * npix, "점 {n}");
    let far = out.cloud.positions.iter().filter(|x| s.surface_distance([x[0] as f64, x[1] as f64, x[2] as f64]) > 2e-3).count();
    assert!(far * 100 < n, "표면 밖 {far}/{n}");
    assert!(out.visibility.iter().all(|v| v.len() >= 3));
    // 결정적(스레드 수와 무관).
    let one = run_fusion(&s.depth, &s.normal, &p, 1);
    assert_eq!(one.cloud, out.cloud);
    // 표시를 끄면 점이 더 많다(같은 표면을 뷰마다 다시 낸다).
    let nomark = run_fusion(&s.depth, &s.normal, &FusionParams { mark_used: false, ..p.clone() }, 0);
    assert!(nomark.cloud.len() > n);
}
