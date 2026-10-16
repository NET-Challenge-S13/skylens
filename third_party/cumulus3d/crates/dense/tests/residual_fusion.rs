/*
 * residual_fusion.rs
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

//! 일치 융합의 남은 픽셀 처리(none / release / second-pass): 합성 장면의 참 깊이맵 입력.

use cumulus3d_dense::fusion::{fuse, FusionInput, FusionOutput};
use cumulus3d_dense::neighbors::PairStats;
use cumulus3d_dense::synthetic::{make_scene, SynthConfig, SynthScene};
use cumulus3d_dense::{DenseScene, FusionParams, FusionResidual};
use std::collections::HashMap;
use std::sync::OnceLock;

fn synth() -> &'static SynthScene {
    static S: OnceLock<SynthScene> = OnceLock::new();
    S.get_or_init(|| make_scene(&SynthConfig::default()))
}

fn overlap(scene: &DenseScene) -> Vec<Vec<usize>> {
    let st = PairStats::new(scene);
    (0..scene.views.len()).map(|v| st.select(v, 50, 0.0)).collect()
}

/// 참 깊이에 픽셀별 결정적 잡음(상대 ±amp)을 넣어, 엄격한 1차 융합에서 떨어지는 픽셀이 생기게 한다.
fn noisy_depth(amp: f32) -> Vec<Vec<f32>> {
    synth()
        .depth
        .iter()
        .enumerate()
        .map(|(v, d)| {
            d.iter()
                .enumerate()
                .map(|(i, &x)| {
                    let h = (i as u64 ^ ((v as u64) << 40)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                    let u = ((h >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
                    x * (1.0 + amp * u)
                })
                .collect()
        })
        .collect()
}

fn run(depth: &[Vec<f32>], p: &FusionParams, threads: usize) -> FusionOutput {
    let s = synth();
    let inputs: Vec<Option<FusionInput>> = depth.iter().zip(&s.normal).map(|(d, n)| Some(FusionInput::plain(d, n))).collect();
    fuse(&s.scene, &inputs, &overlap(&s.scene), p, threads)
}

fn params(residual: FusionResidual) -> FusionParams {
    FusionParams { residual, min_consistent_views: 4, max_depth_error: 0.004, ..FusionParams::default() }
}

fn gsd(depth: &[Vec<f32>]) -> f64 {
    let s = synth();
    let mut g: Vec<f64> = depth
        .iter()
        .enumerate()
        .map(|(v, d)| {
            let mut x: Vec<f64> = d.iter().filter(|&&a| a > 0.0).step_by(97).map(|&a| a as f64).collect();
            x.sort_by(|a, b| a.total_cmp(b));
            x[x.len() / 2] / s.scene.views[v].k[0]
        })
        .collect();
    g.sort_by(|a, b| a.total_cmp(b));
    g[g.len() / 2]
}

#[test]
fn none_and_release_are_identical_and_untagged() {
    let d = noisy_depth(0.003);
    let none = run(&d, &params(FusionResidual::None), 0);
    let rel = run(&d, &params(FusionResidual::Release), 0);
    assert!(none.cloud.len() > 1000);
    // 채택이 확정된 뒤에만 표시를 남기므로, 버려진 후보의 점유를 되돌려도 결과가 같다.
    assert_eq!(none.cloud, rel.cloud);
    assert_eq!(none.visibility, rel.visibility);
    assert!(none.residual.is_empty() && rel.residual.is_empty());
    assert_eq!(none.num_residual(), 0);
    // 기본값(none)은 이전 동작과 같다: 표시를 끄지 않은 일치 융합의 결정성.
    assert_eq!(run(&d, &params(FusionResidual::Release), 1).cloud, rel.cloud);
}

#[test]
fn second_pass_adds_points_away_from_first_pass() {
    let s = synth();
    let d = noisy_depth(0.003);
    let none = run(&d, &params(FusionResidual::None), 0);
    let sp = run(&d, &params(FusionResidual::SecondPass), 0);
    let n1 = none.cloud.len();
    // 1차 결과는 그대로이고, 뒤에 2차 점이 꼬리표와 함께 붙는다.
    assert_eq!(sp.residual.len(), sp.cloud.len());
    assert_eq!(&sp.cloud.positions[..n1], &none.cloud.positions[..]);
    assert!(sp.residual[..n1].iter().all(|&r| !r));
    assert!(sp.residual[n1..].iter().all(|&r| r));
    let nr = sp.num_residual();
    assert!(nr > 0, "2차 점 없음");
    assert_eq!(sp.residual_cloud().len(), nr);
    assert!(sp.visibility[n1..].iter().all(|v| v.len() > FusionParams::default().residual_params.min_views));
    // 2차 점은 표면 위(깊이 잡음 ±0.3%·고도 20 m ≈ ±6 cm 안).
    let far = sp.cloud.positions[n1..].iter().filter(|x| s.surface_distance([x[0] as f64, x[1] as f64, x[2] as f64]) > 0.07).count();
    assert!(far * 50 < nr, "표면 밖 2차 점 {far}/{nr}");
    // 1차 점과 min-dist 이내로 겹치지 않는다.
    let md = FusionParams::default().residual_params.min_dist_gsd * gsd(&d);
    assert!(md > 0.0);
    let key = |x: &[f32; 3]| x.map(|a| (a as f64 / md).floor() as i64);
    let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (i, x) in sp.cloud.positions[..n1].iter().enumerate() {
        grid.entry(key(x)).or_default().push(i);
    }
    for x in &sp.cloud.positions[n1..] {
        let c = key(x);
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    for &i in grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]).map(|v| v.as_slice()).unwrap_or(&[]) {
                        let q = sp.cloud.positions[i];
                        let d2: f64 = (0..3).map(|a| (q[a] as f64 - x[a] as f64).powi(2)).sum();
                        assert!(d2.sqrt() > md, "2차 점이 1차 점과 {} < {md}", d2.sqrt());
                    }
                }
            }
        }
    }
    // min-dist 를 0 으로 하면 2차 점이 더 많거나 같다.
    let mut p0 = params(FusionResidual::SecondPass);
    p0.residual_params.min_dist_gsd = 0.0;
    assert!(run(&d, &p0, 0).num_residual() >= nr);
}

#[test]
fn second_pass_is_deterministic() {
    let d = noisy_depth(0.003);
    let p = params(FusionResidual::SecondPass);
    let a = run(&d, &p, 0);
    for t in [1, 3] {
        let b = run(&d, &p, t);
        assert_eq!(a.cloud, b.cloud, "스레드 {t}");
        assert_eq!(a.visibility, b.visibility);
        assert_eq!(a.residual, b.residual);
    }
    assert!(a.pass1_time.as_nanos() > 0 && a.pass2_time.as_nanos() > 0);
}
