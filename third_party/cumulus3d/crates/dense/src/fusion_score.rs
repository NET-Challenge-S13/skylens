/*
 * fusion_score.rs
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

//! 점수 기반 융합: 기준 픽셀의 3D 점을 이웃 뷰에 투영해 일치 뷰의 기여와 자유공간 위반 벌점을 합한 점수로 채택한다.
//!
//! 기준 픽셀 x 를 점 X 로 올리고 이웃 뷰 j 마다
//! - 깊이·법선이 일치하면 기여 s_j = c_j · exp(−e_j²/(2σ_e²)) · (1 − exp(−θ_j²/(2σ_θ²))),
//!   c_j = 그 픽셀의 광도 신뢰도 clamp(1 − 비용, 0, 1)(비용이 없으면 1), e_j = 왕복 재투영 오차(px),
//!   θ_j = 기준·j 시선 사잇각;
//! - 불일치이면서 X 가 j 의 관측 표면보다 카메라에 가까우면(자유공간 위반) 벌점 p_j = c_j.
//!
//! 기준 자신은 s_ref = c_ref. 점수 S = Σs − λΣp 가 τ 이상이면 후보가 되고, 위치·법선·색은 s(역분산 가중이
//! 켜져 있으면 곱함)를 가중치로 평균한다.
//!
//! 병렬·결정성: 1단계는 기준 영상 단위 병렬로 후보를 만든다(공유 상태 없음). 2단계(`mark_used`)는 각 후보가 쓰는
//! 픽셀(기준 + 일치 픽셀) 집합이 겹치면 우선순위(점수 내림차순, 같으면 (뷰, 픽셀) 오름차순)가 높은 후보만 남긴다.
//! 라운드마다 남은 후보가 자기 픽셀을 원자적 최소로 점유하고, 모든 픽셀을 이긴 후보를 채택해 그 픽셀을 잠근다.
//! 결과는 우선순위 순 탐욕 선택과 같아 실행 순서·스레드 수와 무관하다. 출력은 (뷰, 픽셀) 순.

use crate::fusion::{FusionInput, FusionOutput};
use crate::params::FusionParams;
use crate::scene::DenseScene;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

/// 점수 융합 옵션.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreFusionOptions {
    /// 재투영 오차 가중의 σ_e(픽셀).
    pub sigma_e_px: f64,
    /// 시선 사잇각 가중의 σ_θ(도).
    pub sigma_theta_deg: f64,
    /// 자유공간 위반 벌점 계수 λ.
    pub lambda: f64,
    /// 채택 문턱 τ.
    pub tau: f64,
    /// 상대 깊이 허용치(일치 판정과 자유공간 위반의 여유).
    pub depth_error: f64,
    /// 법선 허용 각(도).
    pub normal_error_deg: f64,
    /// 검사할 겹침 뷰 수(겹침 목록 앞에서부터).
    pub num_neighbors: usize,
    /// 채택된 점에 쓰인 픽셀을 다른 점에 다시 쓰지 않음.
    pub mark_used: bool,
    /// 깊이 불확실성 역분산 가중을 기여 가중치에 곱함.
    pub inverse_variance: bool,
    /// 역분산 가중의 정합 불확실성 σ_px = sigma_px0 + sigma_px_slope·비용 (픽셀).
    pub sigma_px0: f64,
    /// 역분산 가중 σ_px 의 비용 기울기.
    pub sigma_px_slope: f64,
}

impl Default for ScoreFusionOptions {
    fn default() -> Self {
        Self {
            sigma_e_px: 1.0,
            sigma_theta_deg: 2.0,
            lambda: 1.0,
            tau: 2.0,
            depth_error: 0.01,
            normal_error_deg: 10.0,
            num_neighbors: 12,
            mark_used: true,
            inverse_variance: true,
            sigma_px0: 0.25,
            sigma_px_slope: 1.0,
        }
    }
}

impl ScoreFusionOptions {
    /// 기존 융합 허용치(깊이·법선 허용치, 겹침 뷰 수, 사용 표시, 역분산 가중)를 이어받고 점수 매개변수는 기본값.
    pub fn from_fusion(p: &FusionParams) -> Self {
        Self {
            depth_error: p.max_depth_error,
            normal_error_deg: p.max_normal_error_deg,
            num_neighbors: p.consistency_num_images,
            mark_used: p.mark_used,
            inverse_variance: p.inverse_variance,
            sigma_px0: p.sigma_px0,
            sigma_px_slope: p.sigma_px_slope,
            ..Self::default()
        }
    }
}

struct Cam {
    k: [f64; 4],
    r: [[f64; 3]; 3],
    t: [f64; 3],
    c: [f64; 3],
    w: usize,
    h: usize,
}

impl Cam {
    #[inline]
    fn project(&self, x: &[f64; 3]) -> Option<(f64, f64, f64)> {
        let r = &self.r;
        let cz = r[2][0] * x[0] + r[2][1] * x[1] + r[2][2] * x[2] + self.t[2];
        if cz <= 0.0 {
            return None;
        }
        let cx = r[0][0] * x[0] + r[0][1] * x[1] + r[0][2] * x[2] + self.t[0];
        let cy = r[1][0] * x[0] + r[1][1] * x[1] + r[1][2] * x[2] + self.t[1];
        Some((self.k[0] * cx / cz + self.k[2], self.k[1] * cy / cz + self.k[3], cz))
    }
    #[inline]
    fn backproject(&self, col: f64, row: f64, d: f64) -> [f64; 3] {
        let c = [(col - self.k[2]) / self.k[0] * d, (row - self.k[3]) / self.k[1] * d, d];
        self.dir_world(&[c[0] - self.t[0], c[1] - self.t[1], c[2] - self.t[2]])
    }
    #[inline]
    fn dir_world(&self, p: &[f64; 3]) -> [f64; 3] {
        let r = &self.r;
        [0, 1, 2].map(|i| r[0][i] * p[0] + r[1][i] * p[1] + r[2][i] * p[2])
    }
    #[inline]
    fn normal_world(&self, n: &[f32; 3]) -> [f64; 3] {
        self.dir_world(&[n[0] as f64, n[1] as f64, n[2] as f64])
    }
}

#[inline]
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// 광도 신뢰도 [0,1]: clamp(1 − 비용, 0, 1), 비용이 없으면 1.
#[inline]
fn confidence(inp: &FusionInput<'_>, pix: usize) -> f64 {
    inp.cost.map_or(1.0, |c| (1.0 - c[pix] as f64).clamp(0.0, 1.0))
}

#[inline]
fn inv_var(o: &ScoreFusionOptions, inp: &FusionInput<'_>, f: f64, pix: usize, d: f64) -> f64 {
    match (o.inverse_variance, inp.cost) {
        (true, Some(c)) if inp.baseline > 0.0 => {
            let spx = o.sigma_px0 + o.sigma_px_slope * c[pix].clamp(0.0, 2.0) as f64;
            let sd = d * d * spx / (f * inp.baseline);
            1.0 / (sd * sd).max(1e-300)
        }
        _ => 1.0,
    }
}

/// 채택 후보(S ≥ τ).
struct Cand {
    pix: u32,
    score: f32,
    pos: [f32; 3],
    nrm: [f32; 3],
    col: [u8; 3],
    /// 일치 픽셀 (뷰, 픽셀) — 뷰 후보 묶음의 `hits` 안 구간.
    hit0: u32,
    nhit: u32,
}

#[derive(Default)]
struct ViewCands {
    cands: Vec<Cand>,
    hits: Vec<(u32, u32)>,
}

/// 기준 뷰 v 의 후보 계산(공유 상태 없음).
fn view_candidates(
    scene: &DenseScene,
    cams: &[Cam],
    inputs: &[Option<FusionInput<'_>>],
    ov: &[usize],
    v: usize,
    o: &ScoreFusionOptions,
) -> ViewCands {
    let Some(inp) = inputs[v] else { return ViewCands::default() };
    let cam = &cams[v];
    let (w, h) = (cam.w, cam.h);
    let cos_n = o.normal_error_deg.to_radians().cos();
    let k_e = 1.0 / (2.0 * o.sigma_e_px * o.sigma_e_px);
    let st = o.sigma_theta_deg.to_radians();
    let k_t = 1.0 / (2.0 * st * st);
    let color = &scene.views[v].color;
    let mut out = ViewCands::default();
    let mut hits: Vec<(u32, u32)> = Vec::new();
    for row in 0..h {
        for col in 0..w {
            let pix = row * w + col;
            let d = inp.depth[pix] as f64;
            if d <= 0.0 {
                continue;
            }
            let x = cam.backproject(col as f64, row as f64, d);
            let nw = cam.normal_world(&inp.normal[pix]);
            let ray_ref = [x[0] - cam.c[0], x[1] - cam.c[1], x[2] - cam.c[2]];
            let l_ref = dot(&ray_ref, &ray_ref).sqrt();
            let s_ref = confidence(&inp, pix);
            let w0 = s_ref * inv_var(o, &inp, cam.k[0], pix, d);
            let mut sum_s = s_ref;
            let mut sum_p = 0.0;
            let mut sw = w0;
            let mut sx = x.map(|a| a * w0);
            let mut sn = nw.map(|a| a * w0);
            let c0 = color.rgb(col.min(color.width - 1), row.min(color.height - 1));
            let mut sc = c0.map(|a| a as f64 * w0);
            hits.clear();
            for &j in ov {
                let oc = &cams[j];
                let Some((u, vv, z)) = oc.project(&x) else { continue };
                let (cu, rv) = ((u + 0.5).floor(), (vv + 0.5).floor());
                if cu < 0.0 || rv < 0.0 || cu >= oc.w as f64 || rv >= oc.h as f64 {
                    continue;
                }
                let opix = rv as usize * oc.w + cu as usize;
                let oin = inputs[j].expect("사용 뷰");
                let od = oin.depth[opix] as f64;
                if od <= 0.0 {
                    continue;
                }
                let cj = confidence(&oin, opix);
                if (z - od).abs() / od > o.depth_error {
                    if z < od * (1.0 - o.depth_error) {
                        sum_p += cj;
                    }
                    continue;
                }
                let no = oc.normal_world(&oin.normal[opix]);
                if dot(&no, &nw) < cos_n {
                    continue;
                }
                let xo = oc.backproject(cu, rv, od);
                let Some((bu, bv, _)) = cam.project(&xo) else { continue };
                let e2 = (bu - col as f64).powi(2) + (bv - row as f64).powi(2);
                let ray_j = [x[0] - oc.c[0], x[1] - oc.c[1], x[2] - oc.c[2]];
                let cos_t = (dot(&ray_ref, &ray_j) / (l_ref * dot(&ray_j, &ray_j).sqrt())).clamp(-1.0, 1.0);
                let th = cos_t.acos();
                let s = cj * (-e2 * k_e).exp() * (1.0 - (-th * th * k_t).exp());
                sum_s += s;
                if s <= 0.0 {
                    continue;
                }
                let wo = s * inv_var(o, &oin, oc.k[0], opix, od);
                let oimg = &scene.views[j].color;
                let co = oimg.rgb((cu as usize).min(oimg.width - 1), (rv as usize).min(oimg.height - 1));
                sw += wo;
                for k in 0..3 {
                    sx[k] += wo * xo[k];
                    sn[k] += wo * no[k];
                    sc[k] += wo * co[k] as f64;
                }
                hits.push((j as u32, opix as u32));
            }
            let score = sum_s - o.lambda * sum_p;
            if score < o.tau || sw <= 0.0 || !sw.is_finite() {
                continue;
            }
            let l = dot(&sn, &sn).sqrt();
            if l < f32::EPSILON as f64 {
                continue;
            }
            out.cands.push(Cand {
                pix: pix as u32,
                score: score as f32,
                pos: sx.map(|a| (a / sw) as f32),
                nrm: sn.map(|a| (a / l) as f32),
                col: sc.map(|a| (a / sw).round().clamp(0.0, 255.0) as u8),
                hit0: out.hits.len() as u32,
                nhit: hits.len() as u32,
            });
            out.hits.extend_from_slice(&hits);
        }
    }
    out
}

fn install<R: Send>(pool: &Option<rayon::ThreadPool>, f: impl FnOnce() -> R + Send) -> R {
    match pool {
        Some(pl) => pl.install(f),
        None => f(),
    }
}

const FREE: u32 = u32::MAX;
const LOCKED: u32 = 0;

/// 점수 기반 융합. 입력은 [`crate::fusion::fuse_consistency`] 와 같다(`inputs[v]` 가 None 이면 뷰 v 미사용,
/// `overlap[v]` 는 검사할 겹침 뷰 순서).
///
/// 연결: densify 의 `fuse_depth_maps` 에서 새 융합 모드일 때 `fuse(..)` 대신
/// `fuse_scored(scene, &inputs, &overlap, &ScoreFusionOptions::from_fusion(&opts.fusion), threads)` 를 호출한다.
pub fn fuse_scored(
    scene: &DenseScene,
    inputs: &[Option<FusionInput<'_>>],
    overlap: &[Vec<usize>],
    opts: &ScoreFusionOptions,
    threads: usize,
) -> FusionOutput {
    let n = scene.views.len();
    let cams: Vec<Cam> = scene.views.iter().map(|v| Cam { k: v.k, r: v.r, t: v.t, c: v.center(), w: v.width, h: v.height }).collect();
    let used: Vec<bool> = (0..n).map(|v| inputs.get(v).is_some_and(|i| i.is_some())).collect();
    let ovs: Vec<Vec<usize>> = (0..n)
        .map(|v| {
            overlap
                .get(v)
                .map_or_else(Vec::new, |l| l.iter().copied().filter(|&o| o != v && o < n && used[o]).take(opts.num_neighbors).collect())
        })
        .collect();
    let pool = if threads > 0 { rayon::ThreadPoolBuilder::new().num_threads(threads).build().ok() } else { None };

    // 1단계: 기준 영상 단위 병렬 후보 계산.
    let per_view: Vec<ViewCands> = install(&pool, || {
        (0..n)
            .into_par_iter()
            .map(|v| if used[v] { view_candidates(scene, &cams, inputs, &ovs[v], v, opts) } else { ViewCands::default() })
            .collect()
    });

    // 전역 후보 목록 (뷰, 후보 색인) — (뷰, 픽셀) 순.
    let ids: Vec<(u32, u32)> =
        per_view.iter().enumerate().flat_map(|(v, vc)| (0..vc.cands.len() as u32).map(move |i| (v as u32, i))).collect();
    let mut accepted = vec![!opts.mark_used; ids.len()];

    if opts.mark_used && !ids.is_empty() {
        assert!(ids.len() < (u32::MAX - 1) as usize, "융합 후보 수가 너무 많음");
        // 우선순위 순위(1 부터; 작을수록 우선): 점수 내림차순, 같으면 (뷰, 픽셀) 순.
        let mut order: Vec<u32> = (0..ids.len() as u32).collect();
        let score = |g: u32| {
            let (v, i) = ids[g as usize];
            per_view[v as usize].cands[i as usize].score
        };
        order.par_sort_unstable_by(|&a, &b| score(b).total_cmp(&score(a)).then(a.cmp(&b)));
        let mut rank = vec![0u32; ids.len()];
        for (r, &g) in order.iter().enumerate() {
            rank[g as usize] = r as u32 + 1;
        }
        let claim: Vec<Vec<AtomicU32>> =
            (0..n).map(|v| if used[v] { (0..cams[v].w * cams[v].h).map(|_| AtomicU32::new(FREE)).collect() } else { Vec::new() }).collect();
        let pixels = |g: usize| {
            let (v, i) = ids[g];
            let vc = &per_view[v as usize];
            let c = &vc.cands[i as usize];
            std::iter::once((v, c.pix)).chain(vc.hits[c.hit0 as usize..(c.hit0 + c.nhit) as usize].iter().copied())
        };
        let cell = |(v, p): (u32, u32)| &claim[v as usize][p as usize];
        // 0 = 미정, 1 = 채택, 2 = 기각.
        let state: Vec<std::sync::atomic::AtomicU8> = (0..ids.len()).map(|_| std::sync::atomic::AtomicU8::new(0)).collect();
        let mut active: Vec<usize> = (0..ids.len()).collect();
        while !active.is_empty() {
            install(&pool, || {
                // 점유 초기화(잠긴 픽셀 제외).
                active.par_iter().for_each(|&g| {
                    for q in pixels(g) {
                        let c = cell(q);
                        if c.load(Ordering::Relaxed) != LOCKED {
                            c.store(FREE, Ordering::Relaxed);
                        }
                    }
                });
                // 잠긴 픽셀이 있으면 기각, 아니면 순위로 점유.
                active.par_iter().for_each(|&g| {
                    if pixels(g).any(|q| cell(q).load(Ordering::Relaxed) == LOCKED) {
                        state[g].store(2, Ordering::Relaxed);
                        return;
                    }
                    for q in pixels(g) {
                        cell(q).fetch_min(rank[g], Ordering::Relaxed);
                    }
                });
                // 모든 픽셀을 이긴 후보 채택.
                active.par_iter().for_each(|&g| {
                    if state[g].load(Ordering::Relaxed) == 0 && pixels(g).all(|q| cell(q).load(Ordering::Relaxed) == rank[g]) {
                        state[g].store(1, Ordering::Relaxed);
                    }
                });
                // 채택 후보의 픽셀 잠금.
                active.par_iter().for_each(|&g| {
                    if state[g].load(Ordering::Relaxed) == 1 {
                        for q in pixels(g) {
                            cell(q).store(LOCKED, Ordering::Relaxed);
                        }
                    }
                });
            });
            active.retain(|&g| state[g].load(Ordering::Relaxed) == 0);
        }
        for (g, a) in accepted.iter_mut().enumerate() {
            *a = state[g].load(Ordering::Relaxed) == 1;
        }
    }

    let mut out = FusionOutput::default();
    for (g, &(v, i)) in ids.iter().enumerate() {
        if !accepted[g] {
            continue;
        }
        let vc = &per_view[v as usize];
        let c = &vc.cands[i as usize];
        out.cloud.positions.push(c.pos);
        out.cloud.normals.push(c.nrm);
        out.cloud.colors.push(c.col);
        let mut vis: Vec<u32> =
            std::iter::once(v).chain(vc.hits[c.hit0 as usize..(c.hit0 + c.nhit) as usize].iter().map(|&(o, _)| o)).collect();
        vis.sort_unstable();
        vis.dedup();
        out.visibility.push(vis);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::{make_scene, SynthConfig, SynthScene};

    fn synth() -> SynthScene {
        make_scene(&SynthConfig { width: 160, height: 120, focal: 125.0, num_points: 0, ..SynthConfig::default() })
    }

    fn all_overlap(n: usize) -> Vec<Vec<usize>> {
        (0..n).map(|v| (0..n).filter(|&o| o != v).collect()).collect()
    }

    fn run(s: &SynthScene, depth: &[Vec<f32>], o: &ScoreFusionOptions, threads: usize) -> FusionOutput {
        let inputs: Vec<Option<FusionInput>> = depth.iter().zip(&s.normal).map(|(d, n)| Some(FusionInput::plain(d, n))).collect();
        fuse_scored(&s.scene, &inputs, &all_overlap(s.scene.views.len()), o, threads)
    }

    #[test]
    fn fuses_synthetic_scene_accurately() {
        let s = synth();
        let out = run(&s, &s.depth, &ScoreFusionOptions::default(), 0);
        let np = out.cloud.positions.len();
        assert!(np > 5000, "점 수 {np}");
        assert_eq!(out.visibility.len(), np);
        let mut d: Vec<f64> = out.cloud.positions.iter().map(|p| s.surface_distance([p[0] as f64, p[1] as f64, p[2] as f64])).collect();
        d.sort_by(|a, b| a.total_cmp(b));
        let med = d[d.len() / 2];
        let p99 = d[d.len() * 99 / 100];
        assert!(med < 0.02 && p99 < 0.2, "표면 거리 중앙값 {med}, 99% {p99}");
        assert!(out.visibility.iter().all(|v| v.len() >= 2));
    }

    #[test]
    fn mark_used_reduces_duplicates() {
        let s = synth();
        let with = run(&s, &s.depth, &ScoreFusionOptions::default(), 0).cloud.positions.len();
        let without = run(&s, &s.depth, &ScoreFusionOptions { mark_used: false, ..Default::default() }, 0).cloud.positions.len();
        assert!(with < without, "{with} vs {without}");
    }

    #[test]
    fn point_count_monotone_in_tau() {
        let s = synth();
        let mut prev = usize::MAX;
        for tau in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8.5] {
            let c = run(&s, &s.depth, &ScoreFusionOptions { tau, ..Default::default() }, 0).cloud.positions.len();
            assert!(c <= prev, "τ={tau}: {c} > {prev}");
            prev = c;
        }
        assert!(prev < run(&s, &s.depth, &ScoreFusionOptions { tau: 0.5, ..Default::default() }, 0).cloud.positions.len());
    }

    #[test]
    fn free_space_violations_are_rejected() {
        let s = synth();
        let mut depth = s.depth.clone();
        let w = s.config.width;
        // 뷰 4(가운데)의 한 구역을 카메라 쪽으로 당긴 떠 있는 점으로 만든다.
        for row in 40..80 {
            for col in 60..100 {
                depth[4][row * w + col] *= 0.6;
            }
        }
        let floating = |out: &FusionOutput| {
            out.cloud.positions.iter().filter(|p| s.surface_distance([p[0] as f64, p[1] as f64, p[2] as f64]) > 1.0).count()
        };
        // 기준 혼자서도 채택되는 낮은 문턱: 벌점이 없으면 떠 있는 점이 남는다.
        let base = ScoreFusionOptions { tau: 0.5, ..Default::default() };
        let no_pen = run(&s, &depth, &ScoreFusionOptions { lambda: 0.0, ..base.clone() }, 0);
        let pen = run(&s, &depth, &base, 0);
        assert!(floating(&no_pen) > 500, "벌점 없음: {}", floating(&no_pen));
        assert_eq!(floating(&pen), 0);
    }

    #[test]
    fn deterministic_across_threads() {
        let s = synth();
        // 결정적 잡음으로 충돌을 늘린다.
        let depth: Vec<Vec<f32>> = s
            .depth
            .iter()
            .enumerate()
            .map(|(v, d)| {
                d.iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        x * (1.0 + 0.004 * ((crate::math::mix64((v * 1_000_003 + i) as u64) >> 40) as f32 / (1u64 << 24) as f32 - 0.5))
                    })
                    .collect()
            })
            .collect();
        let o = ScoreFusionOptions { tau: 1.5, ..Default::default() };
        let a = run(&s, &depth, &o, 1);
        for t in [2, 4, 0] {
            let b = run(&s, &depth, &o, t);
            assert_eq!(a.cloud.positions, b.cloud.positions);
            assert_eq!(a.cloud.normals, b.cloud.normals);
            assert_eq!(a.cloud.colors, b.cloud.colors);
            assert_eq!(a.visibility, b.visibility);
        }
    }
}
