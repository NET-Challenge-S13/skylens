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

//! Score-based fusion: projects the 3D point of a reference pixel onto neighbor views and adopts it by a score summing contributions of
//! agreeing views and free-space violation penalties.
//!
//! Lift the reference pixel x to a point X and, for each neighbor view j:
//! - if depth and normal agree, contribution s_j = c_j · exp(−e_j²/(2σ_e²)) · (1 − exp(−θ_j²/(2σ_θ²))), with c_j = photometric confidence
//!   of that pixel clamp(1 − cost, 0, 1) (1 if there is no cost), e_j = round-trip reprojection error (px), θ_j = angle between the
//!   reference and j viewing rays;
//! - if it disagrees and X is closer to the camera than the observed surface of j (free-space violation), penalty p_j = c_j.
//!
//! The reference itself has s_ref = c_ref. If the score S = Σs − λΣp is at least τ it becomes a candidate, and position, normal and color
//! are averaged with s as weight (multiplied by the inverse-variance weight
//! if it is on).
//!
//! Parallelism and determinism: stage 1 builds candidates in parallel per reference image (no shared state). Stage 2 (`mark_used`): if the
//! pixel sets (reference + agreeing pixels) used by candidates overlap,
//! only the candidate with higher priority (score descending, ties by (view, pixel) ascending) remains.
//! In each round the remaining candidates atomically occupy their pixels by minimum, and a candidate that wins all its pixels is adopted
//! and locks those pixels.
//! The result equals greedy selection in priority order, so it is independent of execution order and thread count. Output is in (view,
//! pixel) order.

use crate::fusion::{FusionInput, FusionOutput};
use crate::params::FusionParams;
use crate::scene::DenseScene;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

/// Score fusion options.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreFusionOptions {
    /// σ_e (pixels) of the reprojection error weight.
    pub sigma_e_px: f64,
    /// σ_θ (degrees) of the viewing-ray angle weight.
    pub sigma_theta_deg: f64,
    /// Free-space violation penalty coefficient λ.
    pub lambda: f64,
    /// Adoption threshold τ.
    pub tau: f64,
    /// Relative depth tolerance (margin for agreement judgment and free-space violation).
    pub depth_error: f64,
    /// Normal tolerance angle (degrees).
    pub normal_error_deg: f64,
    /// Number of overlapping views to check (from the front of the overlap list).
    pub num_neighbors: usize,
    /// Pixels used by an adopted point are not reused for other points.
    pub mark_used: bool,
    /// Multiply the depth-uncertainty inverse-variance weight into the contribution weight.
    pub inverse_variance: bool,
    /// Registration uncertainty σ_px = sigma_px0 + sigma_px_slope·cost (pixels) of the inverse-variance weight.
    pub sigma_px0: f64,
    /// Cost slope of the inverse-variance weight σ_px.
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
    /// Inherits the existing fusion tolerances (depth/normal tolerances, overlap view count, used mark, inverse-variance weight); score
    /// parameters take defaults.
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

/// Photometric confidence [0,1]: clamp(1 − cost, 0, 1), 1 if there is no cost.
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

/// Adopted candidate (S ≥ τ).
struct Cand {
    pix: u32,
    score: f32,
    pos: [f32; 3],
    nrm: [f32; 3],
    col: [u8; 3],
    /// Agreeing pixels (view, pixel): a range within `hits` of the view's candidate bundle.
    hit0: u32,
    nhit: u32,
}

#[derive(Default)]
struct ViewCands {
    cands: Vec<Cand>,
    hits: Vec<(u32, u32)>,
}

/// Candidate computation for reference view v (no shared state).
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

/// Score-based fusion. Input is the same as [`crate::fusion::fuse_consistency`] (view v is unused if `inputs[v]` is None,
/// `overlap[v]` is the order of overlap views to check).
///
/// Hookup: in densify's `fuse_depth_maps`, in the new fusion mode, call
/// `fuse_scored(scene, &inputs, &overlap, &ScoreFusionOptions::from_fusion(&opts.fusion), threads)` instead of `fuse(..)`.
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

    // Stage 1: candidate computation, parallel per reference image.
    let per_view: Vec<ViewCands> = install(&pool, || {
        (0..n)
            .into_par_iter()
            .map(|v| if used[v] { view_candidates(scene, &cams, inputs, &ovs[v], v, opts) } else { ViewCands::default() })
            .collect()
    });

    // Global candidate list (view, candidate index), in (view, pixel) order.
    let ids: Vec<(u32, u32)> =
        per_view.iter().enumerate().flat_map(|(v, vc)| (0..vc.cands.len() as u32).map(move |i| (v as u32, i))).collect();
    let mut accepted = vec![!opts.mark_used; ids.len()];

    if opts.mark_used && !ids.is_empty() {
        assert!(ids.len() < (u32::MAX - 1) as usize, "융합 후보 수가 너무 많음");
        // Priority rank (from 1; smaller is higher priority): score descending, ties in (view, pixel) order.
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
        // 0 = undecided, 1 = adopted, 2 = rejected.
        let state: Vec<std::sync::atomic::AtomicU8> = (0..ids.len()).map(|_| std::sync::atomic::AtomicU8::new(0)).collect();
        let mut active: Vec<usize> = (0..ids.len()).collect();
        while !active.is_empty() {
            install(&pool, || {
                // Reset occupation (excluding locked pixels).
                active.par_iter().for_each(|&g| {
                    for q in pixels(g) {
                        let c = cell(q);
                        if c.load(Ordering::Relaxed) != LOCKED {
                            c.store(FREE, Ordering::Relaxed);
                        }
                    }
                });
                // Reject if there is a locked pixel, otherwise occupy by rank.
                active.par_iter().for_each(|&g| {
                    if pixels(g).any(|q| cell(q).load(Ordering::Relaxed) == LOCKED) {
                        state[g].store(2, Ordering::Relaxed);
                        return;
                    }
                    for q in pixels(g) {
                        cell(q).fetch_min(rank[g], Ordering::Relaxed);
                    }
                });
                // Adopt candidates that won all their pixels.
                active.par_iter().for_each(|&g| {
                    if state[g].load(Ordering::Relaxed) == 0 && pixels(g).all(|q| cell(q).load(Ordering::Relaxed) == rank[g]) {
                        state[g].store(1, Ordering::Relaxed);
                    }
                });
                // Lock the pixels of adopted candidates.
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
        out.cloud.keys.push(crate::fusion::point_key(
            scene.views[v as usize].image_id,
            c.pix as usize % cams[v as usize].w,
            c.pix as usize / cams[v as usize].w,
        ));
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
        // Make a region of view 4 (the middle) into floating points pulled toward the camera.
        for row in 40..80 {
            for col in 60..100 {
                depth[4][row * w + col] *= 0.6;
            }
        }
        let floating = |out: &FusionOutput| {
            out.cloud.positions.iter().filter(|p| s.surface_distance([p[0] as f64, p[1] as f64, p[2] as f64]) > 1.0).count()
        };
        // Low threshold at which the reference alone is adopted: without penalty the floating points remain.
        let base = ScoreFusionOptions { tau: 0.5, ..Default::default() };
        let no_pen = run(&s, &depth, &ScoreFusionOptions { lambda: 0.0, ..base.clone() }, 0);
        let pen = run(&s, &depth, &base, 0);
        assert!(floating(&no_pen) > 500, "벌점 없음: {}", floating(&no_pen));
        assert_eq!(floating(&pen), 0);
    }

    #[test]
    fn deterministic_across_threads() {
        let s = synth();
        // Increase collisions with deterministic noise.
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
