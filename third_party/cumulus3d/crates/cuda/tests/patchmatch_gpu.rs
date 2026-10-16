/*
 * patchmatch_gpu.rs
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

//! GPU 조밀화 정확도: 합성 장면(무늬 바닥 + 상자, 3×3 카메라)의 참 깊이·법선과 직접 비교. 장치가 없으면 건너뛴다.

use cumulus3d_cuda::{is_available, CudaDevice, CudaPatchMatch, CudaPatchMatchOptions};
use cumulus3d_dense::synthetic::{make_scene, SynthConfig, SynthScene};
use cumulus3d_dense::{compute_depth_maps, fuse_depth_maps, DensifyOptions, DepthMapSet, MvsProfile};
use std::sync::OnceLock;
use std::time::Instant;

fn synth() -> &'static SynthScene {
    static S: OnceLock<SynthScene> = OnceLock::new();
    S.get_or_init(|| make_scene(&SynthConfig::default()))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

struct Acc {
    valid: f64,
    depth_med: f64,
    normal_med_deg: f64,
    depth_p90: f64,
}

fn accuracy(s: &SynthScene, set: &DepthMapSet) -> Acc {
    let mut de = Vec::new();
    let mut ne = Vec::new();
    let mut total = 0usize;
    for (v, m) in set.maps.iter().enumerate() {
        let m = m.as_ref().expect("깊이맵");
        for i in 0..m.depth.len() {
            total += 1;
            let d = m.depth[i] as f64;
            let t = s.depth[v][i] as f64;
            if d <= 0.0 || t <= 0.0 {
                continue;
            }
            de.push((d - t).abs() / t);
            let (a, b) = (m.normal[i], s.normal[v][i]);
            let c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) as f64;
            ne.push(c.clamp(-1.0, 1.0).acos().to_degrees());
        }
    }
    let valid = de.len() as f64 / total as f64;
    let dm = median(&mut de);
    let p90 = de[de.len() * 9 / 10];
    Acc { valid, depth_med: dm, normal_med_deg: median(&mut ne), depth_p90: p90 }
}

fn backend(hw: bool) -> CudaPatchMatch {
    CudaPatchMatch::new(CudaDevice::new(0).unwrap(), CudaPatchMatchOptions { hw_interp: hw, ..Default::default() }).unwrap()
}

fn check(profile: MvsProfile, hw: bool) -> DepthMapSet {
    let s = synth();
    let be = backend(hw);
    let opts = DensifyOptions::with_profile(profile);
    let t = Instant::now();
    let set = compute_depth_maps(&s.scene, &opts, &be, None).unwrap();
    let el = t.elapsed();
    assert_eq!(set.num_levels, 3);
    let a = accuracy(s, &set);
    eprintln!(
        "{profile:?} hw={hw}: {:.2?} 유효 {:.3} 깊이 상대 오차 중앙 {:.2e} (90% {:.2e}) 법선 중앙 {:.2}° 스케일별 {:?}",
        el, a.valid, a.depth_med, a.depth_p90, a.normal_med_deg, set.timings.levels
    );
    assert!(a.depth_med < 0.005, "깊이 중앙 상대 오차 {}", a.depth_med);
    assert!(a.normal_med_deg < 3.0, "법선 중앙 오차 {}", a.normal_med_deg);
    assert!(a.valid > 0.5, "유효 비율 {}", a.valid);
    set
}

#[test]
fn gpu_depth_accuracy_fast_and_quality() {
    if !is_available() {
        eprintln!("CUDA 장치 없음: 건너뜀");
        return;
    }
    let s = synth();
    let fast = check(MvsProfile::Fast, true);
    check(MvsProfile::Quality, true);
    check(MvsProfile::Fast, false);
    // 결정성: 같은 입력이면 비트 단위로 같다.
    let again = compute_depth_maps(&s.scene, &DensifyOptions::with_profile(MvsProfile::Fast), &backend(true), None).unwrap();
    for (a, b) in fast.maps.iter().zip(&again.maps) {
        let (a, b) = (a.as_ref().unwrap(), b.as_ref().unwrap());
        assert!(a.raw_depth == b.raw_depth && a.normal == b.normal, "결정성");
    }
    // 융합 점은 장면 표면 위.
    let opts = DensifyOptions::default();
    for mode in [cumulus3d_dense::FusionMode::Consistency, cumulus3d_dense::FusionMode::Traversal] {
        let mut o = opts.clone();
        o.fusion.mode = mode;
        let f = fuse_depth_maps(&s.scene, &fast, &o, 0);
        let mut d: Vec<f64> = f.cloud.positions.iter().map(|p| s.surface_distance([p[0] as f64, p[1] as f64, p[2] as f64])).collect();
        let n = d.len();
        let far = d.iter().filter(|&&x| x > 0.1).count();
        let med = median(&mut d);
        eprintln!("{mode:?}: 점 {n}, 표면 거리 중앙 {med:.4} m, 0.1 m 초과 {:.3}%", 100.0 * far as f64 / n as f64);
        assert!(n > 10_000);
        assert!(med < 0.02, "표면 거리 중앙 {med}");
        assert!(far * 50 < n, "0.1 m 초과 {far}/{n}");
    }
}

#[test]
fn gpu_depth_cache_reuses_views() {
    if !is_available() {
        eprintln!("CUDA 장치 없음: 건너뜀");
        return;
    }
    let s = synth();
    let be = backend(true);
    let cache = cumulus3d_dense::DepthMapCache::new();
    let opts = DensifyOptions::default();
    let a = compute_depth_maps(&s.scene, &opts, &be, Some(&cache)).unwrap();
    assert_eq!(a.cache_hits, 0);
    let b = compute_depth_maps(&s.scene, &opts, &be, Some(&cache)).unwrap();
    assert_eq!(b.cache_hits, 9);
    for (x, y) in a.maps.iter().zip(&b.maps) {
        assert_eq!(x.as_ref().unwrap().depth, y.as_ref().unwrap().depth);
    }
}

/// 측정용 보고(단언 없음): 창 조합·선택 요소별 합성 장면 정확도와 깊이 시간.
#[test]
fn synthetic_variants_report() {
    if !is_available() {
        eprintln!("CUDA 장치 없음: 건너뜀");
        return;
    }
    let s = synth();
    let be = backend(true);
    let mut cases: Vec<(String, DensifyOptions)> = Vec::new();
    for (r, st) in [(5, 1), (5, 2), (4, 2), (3, 1), (3, 2), (2, 1)] {
        let mut o = DensifyOptions::with_profile(MvsProfile::Fast);
        o.pm.window_radius = r;
        o.pm.window_step = st;
        cases.push((format!("창 R{r} S{st}"), o));
    }
    let mut o = DensifyOptions::with_profile(MvsProfile::Fast);
    o.pm.weak_texture_var = 0.0;
    cases.push(("거친 후보 끔".into(), o));
    let mut o = DensifyOptions::with_profile(MvsProfile::Fast);
    o.neighbors.diversity_decay = 0.5;
    cases.push(("방향 다양성 0.5".into(), o));
    let mut o = DensifyOptions::with_profile(MvsProfile::Fast);
    o.filter.median_filter = false;
    cases.push(("중앙값 필터 끔".into(), o));
    let mut o = DensifyOptions::with_profile(MvsProfile::Fast);
    o.geometric_iters_override = Some(3);
    cases.push(("기하 반복 3".into(), o));
    cases.push(("quality".into(), DensifyOptions::with_profile(MvsProfile::Quality)));
    for (name, o) in cases {
        let set = compute_depth_maps(&s.scene, &o, &be, None).unwrap();
        let a = accuracy(s, &set);
        eprintln!(
            "| {name} | {:.3} | {:.2e} | {:.2e} | {:.2}° | {:.0} ms |",
            a.valid,
            a.depth_med,
            a.depth_p90,
            a.normal_med_deg,
            set.timings.depth().as_secs_f64() * 1e3
        );
    }
}
