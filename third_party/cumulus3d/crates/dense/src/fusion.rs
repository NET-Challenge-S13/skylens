/*
 * fusion.rs
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
 * Third-party notices: parts of the algorithms, default parameters and data
 * formats in this file follow other open-source projects. Their copyright
 * notices and licenses are reproduced in THIRD_PARTY_NOTICES.md.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//! Depth map fusion: gathers agreeing pixels by depth-first expansion through neighbors of neighbors and builds per-component median
//! points.
//!
//! Image order: start from the first used image, then move to the first unfinished image in the overlap list of the finished image (if
//! none, the unfinished image with the smallest
//! index). Start pixels within one image are processed in parallel by row bands, and the "fused" mark is
//! taken atomically (the expansion that occupies a pixel first owns it). With `threads == 1` this equals row-major sequential processing.

use crate::params::{FusionMode, FusionParams, FusionResidual};
use crate::scene::DenseScene;
use cumulus3d_core::io::PointCloud;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::time::{Duration, Instant};

/// Identifying key of a fused point: `(image_id << 32) | (row << 16) | col`. (image_id, col, row) are the reference view that made the
/// point and the
/// integer pixel of its depth map (depth map resolution). For agreement fusion it is the pixel where a candidate starts as reference; for
/// extended fusion it is the seed pixel that started the expansion.
/// It is unique within a point cloud and the same when the same depth map is fused again (the same pixel between re-densifications = the
/// same point).
#[inline]
pub fn point_key(image_id: u32, col: usize, row: usize) -> u64 {
    ((image_id as u64) << 32) | ((row as u64 & 0xffff) << 16) | (col as u64 & 0xffff)
}

/// Fusion input: per-view-size depth (0 = invalid) and normals in reference camera coordinates.
#[derive(Clone, Copy, Debug)]
pub struct FusionInput<'a> {
    /// Depth of view size (0 = invalid).
    pub depth: &'a [f32],
    /// Normal in the reference camera coordinates.
    pub normal: &'a [[f32; 3]],
    /// Per-pixel final registration cost (used for inverse-variance weighting if present).
    pub cost: Option<&'a [f32]>,
    /// Representative baseline of the view (median center distance to source views). 0 means no weighting.
    pub baseline: f64,
}

impl<'a> FusionInput<'a> {
    /// Input without weighting information.
    pub fn plain(depth: &'a [f32], normal: &'a [[f32; 3]]) -> Self {
        Self { depth, normal, cost: None, baseline: 0.0 }
    }
}

/// Inverse of the depth variance from stereo error propagation: σ_d = d²·σ_px/(f·b), σ_px = σ0 + s·cost.
#[inline]
fn inv_var(p: &FusionParams, inp: &FusionInput<'_>, f: f64, pix: usize, d: f64) -> f64 {
    match (p.inverse_variance, inp.cost) {
        (true, Some(c)) if inp.baseline > 0.0 => {
            let spx = p.sigma_px0 + p.sigma_px_slope * c[pix].clamp(0.0, 2.0) as f64;
            let sd = d * d * spx / (f * inp.baseline);
            1.0 / (sd * sd).max(1e-300)
        }
        _ => 1.0,
    }
}

/// Fusion result (points, with contributing view indices per point in ascending order).
#[derive(Clone, Debug, Default)]
pub struct FusionOutput {
    /// Fused point cloud.
    pub cloud: PointCloud,
    /// Contributing view indices per point (ascending).
    pub visibility: Vec<Vec<u32>>,
    /// Whether each point is a second-fusion point (empty if no second fusion was done).
    pub residual: Vec<bool>,
    /// First fusion time (recorded only for agreement fusion).
    pub pass1_time: Duration,
    /// Second fusion time (only when the second fusion ran).
    pub pass2_time: Duration,
    /// Normal-direction smoothing time (only when smoothing ran).
    pub smooth_time: Duration,
    /// Visibility filter time (only when the filter ran).
    pub visibility_time: Duration,
    /// Points removed by the visibility filter.
    pub visibility_removed: usize,
}

impl FusionOutput {
    /// Number of second-fusion points.
    pub fn num_residual(&self) -> usize {
        self.residual.iter().filter(|&&r| r).count()
    }
    /// Point cloud of only the second-fusion points.
    pub fn residual_cloud(&self) -> PointCloud {
        let mut pc = PointCloud::default();
        for (i, _) in self.residual.iter().enumerate().filter(|(_, &r)| r) {
            pc.positions.push(self.cloud.positions[i]);
            if let Some(nn) = self.cloud.normals.get(i) {
                pc.normals.push(*nn);
            }
            if let Some(c) = self.cloud.colors.get(i) {
                pc.colors.push(*c);
            }
            if let Some(k) = self.cloud.keys.get(i) {
                pc.keys.push(*k);
            }
        }
        pc
    }
}

struct Cam {
    k: [f64; 4],
    r: [[f64; 3]; 3],
    t: [f64; 3],
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
        let r = &self.r;
        let p = [c[0] - self.t[0], c[1] - self.t[1], c[2] - self.t[2]];
        [
            r[0][0] * p[0] + r[1][0] * p[1] + r[2][0] * p[2],
            r[0][1] * p[0] + r[1][1] * p[1] + r[2][1] * p[2],
            r[0][2] * p[0] + r[1][2] * p[1] + r[2][2] * p[2],
        ]
    }
    #[inline]
    fn normal_world(&self, n: &[f32; 3]) -> [f64; 3] {
        let r = &self.r;
        let n = [n[0] as f64, n[1] as f64, n[2] as f64];
        [
            r[0][0] * n[0] + r[1][0] * n[1] + r[2][0] * n[2],
            r[0][1] * n[0] + r[1][1] * n[1] + r[2][1] * n[2],
            r[0][2] * n[0] + r[1][2] * n[1] + r[2][2] * n[2],
        ]
    }
}

#[derive(Default)]
struct Scratch {
    stack: Vec<(u32, u32, u32, u32)>,
    xs: [Vec<f64>; 3],
    ns: [Vec<f64>; 3],
    cs: [Vec<f64>; 3],
    vis: Vec<u32>,
}

/// Depth map fusion. If `inputs[v]` is None, view v is not used. `overlap[v]` is the order of views to expand into.
pub fn fuse(
    scene: &DenseScene,
    inputs: &[Option<FusionInput<'_>>],
    overlap: &[Vec<usize>],
    p: &FusionParams,
    threads: usize,
) -> FusionOutput {
    match p.mode {
        FusionMode::Traversal => fuse_traversal(scene, inputs, overlap, p, threads),
        FusionMode::Consistency => fuse_consistency(scene, inputs, overlap, p, threads),
    }
}

/// Agreement fusion: views in index order, rows within a view in parallel. The 3D point of a reference pixel is projected onto the overlap
/// views (first `consistency_num_images`);
/// it agrees if the nearest-pixel depth is within the relative tolerance, the error of reprojecting that pixel's point back to the
/// reference is within tolerance, and the normal angle is within tolerance.
/// If the number of agreeing views is at least `min_consistent_views`, the mean position, normal and color of the reference and the
/// agreeing points are emitted as a point.
/// With `mark_used`, pixels of other views used for agreement are not reused as reference pixels of that view (earlier views decide, so it
/// is deterministic).
///
/// Leftover pixel handling (`residual`):
/// - `Release`: pixels (reference and neighbor) occupied by a candidate during checking are given back if adoption fails. This
///   implementation leaves marks only after adoption is confirmed, so the result equals `None` (a path that explicitly guarantees discarded
///   candidates leave no marks).
/// - `SecondPass`: after the first pass ends, only valid-depth pixels that were not reference or agreeing pixels of first-pass points are
///   used as references and rechecked with the second-pass tolerances. Parallel per image; a second-pass candidate atomically occupies
///   (`fetch_min`) its own reference pixel and agreeing pixels with the minimum candidate number, and only candidates that own their
///   reference pixel remain (independent of execution order). Candidates within `min_dist_gsd`·GSD of a first-pass point are discarded.
pub fn fuse_consistency(
    scene: &DenseScene,
    inputs: &[Option<FusionInput<'_>>],
    overlap: &[Vec<usize>],
    p: &FusionParams,
    threads: usize,
) -> FusionOutput {
    let t1 = Instant::now();
    let n = scene.views.len();
    let cams: Vec<Cam> = scene.views.iter().map(|v| Cam { k: v.k, r: v.r, t: v.t, w: v.width, h: v.height }).collect();
    let used: Vec<bool> = (0..n).map(|v| inputs.get(v).is_some_and(|i| i.is_some())).collect();
    let alloc_u8 = |on: bool| -> Vec<Vec<AtomicU8>> {
        (0..n).map(|v| if on && used[v] { (0..cams[v].w * cams[v].h).map(|_| AtomicU8::new(0)).collect() } else { Vec::new() }).collect()
    };
    let marks = alloc_u8(true);
    let second = p.residual == FusionResidual::SecondPass;
    let release = p.residual == FusionResidual::Release;
    // Reference and agreeing pixels of first-pass points (to exclude them from the second pass).
    let consumed = alloc_u8(second);
    let tol =
        Tol { depth: p.max_depth_error, cos_n: p.max_normal_error_deg.to_radians().cos(), r2: p.max_reproj_error * p.max_reproj_error };
    let pool = if threads > 0 { rayon::ThreadPoolBuilder::new().num_threads(threads).build().ok() } else { None };
    let ctx = ConsCtx { scene, cams: &cams, inputs };
    let ovs: Vec<Vec<usize>> =
        (0..n).map(|v| overlap[v].iter().copied().filter(|&o| used[o]).take(p.consistency_num_images).collect()).collect();
    let mut out = FusionOutput::default();
    for v in 0..n {
        let Some(inp) = inputs[v] else { continue };
        let cam = &cams[v];
        let (w, h) = (cam.w, cam.h);
        let ov = &ovs[v];
        let work = || -> Vec<(PointCloud, Vec<Vec<u32>>)> {
            (0..h)
                .into_par_iter()
                .map(|row| {
                    let mut pc = PointCloud::default();
                    let mut vis = Vec::new();
                    let mut hits: Vec<(usize, usize)> = Vec::new();
                    let mut claimed: Vec<(usize, usize)> = Vec::new();
                    for col in 0..w {
                        let pix = row * w + col;
                        if inp.depth[pix] <= 0.0 || (p.mark_used && marks[v][pix].load(Ordering::Relaxed) != 0) {
                            continue;
                        }
                        let pt = ctx.evaluate(p, v, &inp, row, col, ov, &tol, &mut hits);
                        let ok = hits.len() >= p.min_consistent_views && pt.is_some();
                        if release && p.mark_used {
                            // Temporarily occupy (2) the pixels of the checked candidate; on adoption confirm (1), otherwise give back only
                            // what this candidate occupied.
                            // Confirmed marks (1) are not given back, so the adoption marks of other candidates that shared the same pixels
                            // are not erased.
                            claimed.clear();
                            for &(o, q) in std::iter::once(&(v, pix)).chain(hits.iter()) {
                                if marks[o][q].compare_exchange(0, TENTATIVE, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                                    claimed.push((o, q));
                                }
                            }
                            for &(o, q) in &claimed {
                                if ok {
                                    marks[o][q].store(1, Ordering::Release);
                                } else {
                                    let _ = marks[o][q].compare_exchange(TENTATIVE, 0, Ordering::AcqRel, Ordering::Acquire);
                                }
                            }
                        }
                        let Some((x, nrm, c)) = pt.filter(|_| ok) else { continue };
                        if p.mark_used {
                            mark_hits(p, &cams, inputs, &marks, &hits);
                        }
                        if second {
                            consumed[v][pix].store(1, Ordering::Relaxed);
                            for &(o, q) in &hits {
                                consumed[o][q].store(1, Ordering::Relaxed);
                            }
                        }
                        pc.positions.push(x);
                        pc.normals.push(nrm);
                        pc.colors.push(c);
                        pc.keys.push(point_key(scene.views[v].image_id, col, row));
                        vis.push(sorted_vis(v, &hits));
                    }
                    (pc, vis)
                })
                .collect()
        };
        let parts = match &pool {
            Some(pl) => pl.install(work),
            None => work(),
        };
        for (pc, vis) in parts {
            out.cloud.positions.extend(pc.positions);
            out.cloud.normals.extend(pc.normals);
            out.cloud.colors.extend(pc.colors);
            out.cloud.keys.extend(pc.keys);
            out.visibility.extend(vis);
        }
    }
    out.pass1_time = t1.elapsed();
    if second {
        let t2 = Instant::now();
        let res = match &pool {
            Some(pl) => pl.install(|| second_pass(&ctx, p, &used, &ovs, &consumed, &out.cloud)),
            None => second_pass(&ctx, p, &used, &ovs, &consumed, &out.cloud),
        };
        let n1 = out.cloud.len();
        out.residual = vec![false; n1];
        for (pc, vis) in res {
            out.residual.extend(std::iter::repeat_n(true, pc.len()));
            out.cloud.positions.extend(pc.positions);
            out.cloud.normals.extend(pc.normals);
            out.cloud.colors.extend(pc.colors);
            out.cloud.keys.extend(pc.keys);
            out.visibility.extend(vis);
        }
        out.pass2_time = t2.elapsed();
    }
    out
}

/// Temporary occupation mark value for leftover-pixel `Release` handling.
const TENTATIVE: u8 = 2;

/// Marks the agreeing pixels of an adopted point as "used" (including surrounding pixels with matching depth if there is a radius).
fn mark_hits(p: &FusionParams, cams: &[Cam], inputs: &[Option<FusionInput<'_>>], marks: &[Vec<AtomicU8>], hits: &[(usize, usize)]) {
    let r = p.mark_radius as i64;
    for &(o, opix) in hits {
        if r == 0 {
            marks[o][opix].store(1, Ordering::Relaxed);
            continue;
        }
        // Also mark pixels within the radius whose depth agrees (suppresses duplicate reference pixels of the same surface patch).
        let oc = &cams[o];
        let oin = inputs[o].expect("사용 뷰");
        let (cx0, cy0) = ((opix % oc.w) as i64, (opix / oc.w) as i64);
        let d0 = oin.depth[opix] as f64;
        for yy in (cy0 - r).max(0)..=(cy0 + r).min(oc.h as i64 - 1) {
            for xx in (cx0 - r).max(0)..=(cx0 + r).min(oc.w as i64 - 1) {
                let q = yy as usize * oc.w + xx as usize;
                let dq = oin.depth[q] as f64;
                if dq > 0.0 && (dq - d0).abs() <= p.max_depth_error * d0 {
                    marks[o][q].store(1, Ordering::Relaxed);
                }
            }
        }
    }
}

fn sorted_vis(v: usize, hits: &[(usize, usize)]) -> Vec<u32> {
    let mut vv: Vec<u32> = std::iter::once(v as u32).chain(hits.iter().map(|&(o, _)| o as u32)).collect();
    vv.sort_unstable();
    vv.dedup();
    vv
}

/// Agreement check tolerances.
struct Tol {
    depth: f64,
    cos_n: f64,
    r2: f64,
}

struct ConsCtx<'a> {
    scene: &'a DenseScene,
    cams: &'a [Cam],
    inputs: &'a [Option<FusionInput<'a>>],
}

type Point = ([f32; 3], [f32; 3], [u8; 3]);

impl ConsCtx<'_> {
    /// Agreement check for one reference pixel: gathers the agreeing (view, pixel) into `hits` and yields the weighted average point (None
    /// if the normal sum is 0).
    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        &self,
        p: &FusionParams,
        v: usize,
        inp: &FusionInput<'_>,
        row: usize,
        col: usize,
        ov: &[usize],
        tol: &Tol,
        hits: &mut Vec<(usize, usize)>,
    ) -> Option<Point> {
        let cam = &self.cams[v];
        let pix = row * cam.w + col;
        let d = inp.depth[pix] as f64;
        let color = &self.scene.views[v].color;
        let x = cam.backproject(col as f64, row as f64, d);
        let nw = cam.normal_world(&inp.normal[pix]);
        let w0 = inv_var(p, inp, cam.k[0], pix, d);
        let mut sw = w0;
        let mut sx = x.map(|a| a * w0);
        let mut sn = nw.map(|a| a * w0);
        let c0 = color.rgb(col.min(color.width - 1), row.min(color.height - 1));
        let mut sc = [c0[0] as f64 * w0, c0[1] as f64 * w0, c0[2] as f64 * w0];
        hits.clear();
        for &o in ov {
            let oc = &self.cams[o];
            let Some((u, vv, z)) = oc.project(&x) else { continue };
            let (cu, rv) = ((u + 0.5).floor(), (vv + 0.5).floor());
            if cu < 0.0 || rv < 0.0 || cu >= oc.w as f64 || rv >= oc.h as f64 {
                continue;
            }
            let opix = rv as usize * oc.w + cu as usize;
            let oin = self.inputs[o].expect("사용 뷰");
            let od = oin.depth[opix] as f64;
            if od <= 0.0 || (z - od).abs() / od > tol.depth {
                continue;
            }
            let xo = oc.backproject(cu, rv, od);
            let Some((bu, bv, _)) = cam.project(&xo) else { continue };
            if (bu - col as f64).powi(2) + (bv - row as f64).powi(2) > tol.r2 {
                continue;
            }
            let no = oc.normal_world(&oin.normal[opix]);
            if no[0] * nw[0] + no[1] * nw[1] + no[2] * nw[2] < tol.cos_n {
                continue;
            }
            let oc_img = &self.scene.views[o].color;
            let co = oc_img.rgb((cu as usize).min(oc_img.width - 1), (rv as usize).min(oc_img.height - 1));
            let wo = inv_var(p, &oin, oc.k[0], opix, od);
            sw += wo;
            for k in 0..3 {
                sx[k] += wo * xo[k];
                sn[k] += wo * no[k];
                sc[k] += wo * co[k] as f64;
            }
            hits.push((o, opix));
        }
        let m = sw;
        let l = (sn[0] * sn[0] + sn[1] * sn[1] + sn[2] * sn[2]).sqrt();
        if l < f32::EPSILON as f64 {
            return None;
        }
        Some((
            [(sx[0] / m) as f32, (sx[1] / m) as f32, (sx[2] / m) as f32],
            [(sn[0] / l) as f32, (sn[1] / l) as f32, (sn[2] / l) as f32],
            [(sc[0] / m).round() as u8, (sc[1] / m).round() as u8, (sc[2] / m).round() as u8],
        ))
    }
}

/// Ground sample distance of the depth maps (median over views of median valid depth / focal length).
pub fn scene_gsd(scene: &DenseScene, inputs: &[Option<FusionInput<'_>>]) -> f64 {
    let cams: Vec<Cam> = scene.views.iter().map(|v| Cam { k: v.k, r: v.r, t: v.t, w: v.width, h: v.height }).collect();
    depth_gsd(&cams, inputs)
}

/// Ground sample distance of a depth map (median over views of the median valid depth / focal length).
fn depth_gsd(cams: &[Cam], inputs: &[Option<FusionInput<'_>>]) -> f64 {
    let mut gs: Vec<f64> = inputs
        .iter()
        .enumerate()
        .filter_map(|(v, i)| {
            let i = i.as_ref()?;
            let mut d: Vec<f64> = i.depth.iter().filter(|&&d| d > 0.0).step_by(97).map(|&d| d as f64).collect();
            if d.is_empty() {
                return None;
            }
            Some(crate::math::median_in_place(&mut d) / cams[v].k[0])
        })
        .collect();
    if gs.is_empty() {
        0.0
    } else {
        crate::math::median_in_place(&mut gs)
    }
}

/// Grid hash of point positions (cell size `cell`): sorted (cell key, point index).
struct GridIndex {
    cell: f64,
    keys: Vec<(u64, u32)>,
}

impl GridIndex {
    fn cell_key(c: [i64; 3]) -> u64 {
        let h = (c[0] as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (c[1] as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
            ^ (c[2] as u64).wrapping_mul(0x1656_67b1_9e37_79f9);
        crate::math::mix64(h)
    }
    fn cell_of(&self, x: &[f32; 3]) -> [i64; 3] {
        x.map(|a| (a as f64 / self.cell).floor() as i64)
    }
    fn new(cloud: &PointCloud, cell: f64) -> Self {
        let mut g = Self { cell, keys: Vec::new() };
        let keys: Vec<(u64, u32)> = cloud.positions.par_iter().enumerate().map(|(i, x)| (Self::cell_key(g.cell_of(x)), i as u32)).collect();
        g.keys = keys;
        g.keys.par_sort_unstable();
        g
    }
    /// Whether there is a point within distance `r` (≤ cell size) of `x`.
    fn any_within(&self, cloud: &PointCloud, x: &[f32; 3], r: f64) -> bool {
        let c = self.cell_of(x);
        let r2 = r * r;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let k = Self::cell_key([c[0] + dx, c[1] + dy, c[2] + dz]);
                    let lo = self.keys.partition_point(|e| e.0 < k);
                    for &(kk, i) in &self.keys[lo..] {
                        if kk != k {
                            break;
                        }
                        let q = cloud.positions[i as usize];
                        let d2: f64 = (0..3).map(|a| (q[a] as f64 - x[a] as f64).powi(2)).sum();
                        if d2 <= r2 {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

/// Second fusion: uses only valid-depth pixels not used in the first pass as references, checked with the second-pass tolerances. Returns
/// per-view (points, visibility) in view order.
fn second_pass(
    ctx: &ConsCtx<'_>,
    p: &FusionParams,
    used: &[bool],
    ovs: &[Vec<usize>],
    consumed: &[Vec<AtomicU8>],
    first: &PointCloud,
) -> Vec<(PointCloud, Vec<Vec<u32>>)> {
    let rp = &p.residual_params;
    let cams = ctx.cams;
    let n = cams.len();
    let tol = Tol {
        depth: rp.depth_error.unwrap_or(0.5 * p.max_depth_error),
        cos_n: rp.normal_error_deg.unwrap_or(0.67 * p.max_normal_error_deg).to_radians().cos(),
        r2: p.max_reproj_error * p.max_reproj_error,
    };
    let min_dist = rp.min_dist_gsd * depth_gsd(cams, ctx.inputs);
    let grid = (min_dist > 0.0).then(|| GridIndex::new(first, min_dist));
    // Pixel index: view start offset + pixel within the view.
    let mut offset = vec![0u64; n + 1];
    for v in 0..n {
        offset[v + 1] = offset[v] + if used[v] { (cams[v].w * cams[v].h) as u64 } else { 0 };
    }
    assert!(offset[n] < u32::MAX as u64, "2차 융합: 픽셀 수가 u32 범위를 넘음");
    let owner: Vec<Vec<AtomicU32>> =
        (0..n).map(|v| if used[v] { (0..cams[v].w * cams[v].h).map(|_| AtomicU32::new(u32::MAX)).collect() } else { Vec::new() }).collect();
    // Stage 1: build and occupy candidates in parallel per view.
    struct Cand {
        pix: usize,
        pt: Point,
        vis: Vec<u32>,
    }
    let cands: Vec<Vec<Cand>> = (0..n)
        .into_par_iter()
        .map(|v| {
            let mut out = Vec::new();
            let Some(inp) = ctx.inputs[v] else { return out };
            let (w, h) = (cams[v].w, cams[v].h);
            let mut hits = Vec::new();
            for row in 0..h {
                for col in 0..w {
                    let pix = row * w + col;
                    if inp.depth[pix] <= 0.0 || consumed[v][pix].load(Ordering::Relaxed) != 0 {
                        continue;
                    }
                    let Some(pt) = ctx.evaluate(p, v, &inp, row, col, &ovs[v], &tol, &mut hits) else { continue };
                    if hits.len() < rp.min_views {
                        continue;
                    }
                    if let Some(g) = &grid {
                        if g.any_within(first, &pt.0, min_dist) {
                            continue;
                        }
                    }
                    let id = (offset[v] + pix as u64) as u32;
                    owner[v][pix].fetch_min(id, Ordering::AcqRel);
                    for &(o, q) in &hits {
                        owner[o][q].fetch_min(id, Ordering::AcqRel);
                    }
                    out.push(Cand { pix, pt, vis: sorted_vis(v, &hits) });
                }
            }
            out
        })
        .collect();
    // Stage 2: keep only candidates that own their reference pixel (independent of execution order since it is the occupation minimum).
    cands
        .into_par_iter()
        .enumerate()
        .map(|(v, cs)| {
            let mut pc = PointCloud::default();
            let mut vis = Vec::new();
            for c in cs {
                let id = (offset[v] + c.pix as u64) as u32;
                if owner[v][c.pix].load(Ordering::Acquire) != id {
                    continue;
                }
                pc.positions.push(c.pt.0);
                pc.normals.push(c.pt.1);
                pc.colors.push(c.pt.2);
                pc.keys.push(point_key(ctx.scene.views[v].image_id, c.pix % ctx.cams[v].w, c.pix / ctx.cams[v].w));
                vis.push(c.vis);
            }
            (pc, vis)
        })
        .collect()
}

/// Extended median fusion.
pub fn fuse_traversal(
    scene: &DenseScene,
    inputs: &[Option<FusionInput<'_>>],
    overlap: &[Vec<usize>],
    p: &FusionParams,
    threads: usize,
) -> FusionOutput {
    let n = scene.views.len();
    let cams: Vec<Cam> = scene.views.iter().map(|v| Cam { k: v.k, r: v.r, t: v.t, w: v.width, h: v.height }).collect();
    let used: Vec<bool> = (0..n).map(|v| inputs.get(v).is_some_and(|i| i.is_some())).collect();
    let fused: Vec<Vec<AtomicU8>> =
        (0..n).map(|v| if used[v] { (0..cams[v].w * cams[v].h).map(|_| AtomicU8::new(0)).collect() } else { Vec::new() }).collect();
    let mut done = vec![false; n];
    let mut out = FusionOutput::default();
    let cos_max_normal = p.max_normal_error_deg.to_radians().cos();
    let pool = if threads > 0 { rayon::ThreadPoolBuilder::new().num_threads(threads).build().ok() } else { None };
    let mut cur = (0..n).find(|&v| used[v]);
    while let Some(v) = cur {
        let ctx = Ctx { scene, cams: &cams, inputs, overlap, used: &used, done: &done, fused: &fused, p, cos_max_normal };
        let (w, h) = (cams[v].w, cams[v].h);
        const BAND: usize = 8;
        let nb = h.div_ceil(BAND);
        let work = || -> Vec<(PointCloud, Vec<Vec<u32>>)> {
            (0..nb)
                .into_par_iter()
                .map_init(Scratch::default, |sc, b| {
                    let mut pc = PointCloud::default();
                    let mut vis = Vec::new();
                    for row in b * BAND..((b + 1) * BAND).min(h) {
                        for col in 0..w {
                            if fused[v][row * w + col].load(Ordering::Relaxed) != 0 {
                                continue;
                            }
                            if let Some((x, nrm, c, vv)) = ctx.grow(v, row, col, sc) {
                                pc.positions.push(x);
                                pc.normals.push(nrm);
                                pc.colors.push(c);
                                pc.keys.push(point_key(scene.views[v].image_id, col, row));
                                vis.push(vv);
                            }
                        }
                    }
                    (pc, vis)
                })
                .collect()
        };
        let parts = match &pool {
            Some(pl) => pl.install(work),
            None => work(),
        };
        for (pc, vis) in parts {
            out.cloud.positions.extend(pc.positions);
            out.cloud.normals.extend(pc.normals);
            out.cloud.colors.extend(pc.colors);
            out.cloud.keys.extend(pc.keys);
            out.visibility.extend(vis);
        }
        done[v] = true;
        cur = overlap[v].iter().copied().find(|&o| used[o] && !done[o]).or_else(|| (0..n).find(|&o| used[o] && !done[o]));
    }
    out
}

struct Ctx<'a> {
    scene: &'a DenseScene,
    cams: &'a [Cam],
    inputs: &'a [Option<FusionInput<'a>>],
    overlap: &'a [Vec<usize>],
    used: &'a [bool],
    done: &'a [bool],
    fused: &'a [Vec<AtomicU8>],
    p: &'a FusionParams,
    cos_max_normal: f64,
}

type Fused = ([f32; 3], [f32; 3], [u8; 3], Vec<u32>);

impl Ctx<'_> {
    fn grow(&self, v0: usize, row0: usize, col0: usize, sc: &mut Scratch) -> Option<Fused> {
        sc.stack.clear();
        for a in sc.xs.iter_mut().chain(sc.ns.iter_mut()).chain(sc.cs.iter_mut()) {
            a.clear();
        }
        sc.vis.clear();
        sc.stack.push((v0 as u32, row0 as u32, col0 as u32, 0));
        let mut xref = [0.0f64; 3];
        let mut nref = [0.0f64; 3];
        let mut count = 0usize;
        while let Some((v, row, col, depth)) = sc.stack.pop() {
            let (v, row, col) = (v as usize, row as usize, col as usize);
            let cam = &self.cams[v];
            let pix = row * cam.w + col;
            let flag = &self.fused[v][pix];
            if flag.load(Ordering::Relaxed) != 0 {
                continue;
            }
            let inp = self.inputs[v].as_ref().expect("사용 뷰");
            let d = inp.depth[pix] as f64;
            if d <= 0.0 {
                continue;
            }
            if depth > 0 {
                let Some((px, py, pz)) = cam.project(&xref) else { continue };
                if (pz - d).abs() / d > self.p.max_depth_error {
                    continue;
                }
                let e2 = (px - col as f64).powi(2) + (py - row as f64).powi(2);
                if e2 > self.p.max_reproj_error * self.p.max_reproj_error {
                    continue;
                }
            }
            let nw = cam.normal_world(&inp.normal[pix]);
            if depth > 0 && nref[0] * nw[0] + nref[1] * nw[1] + nref[2] * nw[2] < self.cos_max_normal {
                continue;
            }
            if flag.swap(1, Ordering::AcqRel) != 0 {
                continue;
            }
            let x = cam.backproject(col as f64, row as f64, d);
            let rgb = self.color(v, row, col);
            for c in 0..3 {
                sc.xs[c].push(x[c]);
                sc.ns[c].push(nw[c]);
                sc.cs[c].push(rgb[c] as f64);
            }
            sc.vis.push(v as u32);
            if depth == 0 {
                xref = x;
                nref = nw;
            }
            count += 1;
            if count >= self.p.max_num_pixels {
                break;
            }
            if depth as usize >= self.p.max_traversal_depth.saturating_sub(1) {
                continue;
            }
            for &o in &self.overlap[v] {
                if !self.used[o] || self.done[o] {
                    continue;
                }
                let oc = &self.cams[o];
                let Some((u, vv, _)) = oc.project(&x) else { continue };
                let (cu, rv) = ((u + 0.5).floor(), (vv + 0.5).floor());
                if cu < 0.0 || rv < 0.0 || cu >= oc.w as f64 || rv >= oc.h as f64 {
                    continue;
                }
                sc.stack.push((o as u32, rv as u32, cu as u32, depth + 1));
            }
        }
        if count < self.p.min_num_pixels.max(1) {
            return None;
        }
        let med = |a: &mut Vec<f64>| crate::math::median_in_place(a);
        let x = [med(&mut sc.xs[0]), med(&mut sc.xs[1]), med(&mut sc.xs[2])];
        let n = [med(&mut sc.ns[0]), med(&mut sc.ns[1]), med(&mut sc.ns[2])];
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if l < f32::EPSILON as f64 {
            return None;
        }
        let c = [med(&mut sc.cs[0]), med(&mut sc.cs[1]), med(&mut sc.cs[2])];
        let mut vis = sc.vis.clone();
        vis.sort_unstable();
        vis.dedup();
        Some((
            [x[0] as f32, x[1] as f32, x[2] as f32],
            [(n[0] / l) as f32, (n[1] / l) as f32, (n[2] / l) as f32],
            [c[0].round().clamp(0.0, 255.0) as u8, c[1].round().clamp(0.0, 255.0) as u8, c[2].round().clamp(0.0, 255.0) as u8],
            vis,
        ))
    }

    fn color(&self, v: usize, row: usize, col: usize) -> [u8; 3] {
        let c = &self.scene.views[v].color;
        if col < c.width && row < c.height {
            c.rgb(col, row)
        } else {
            [0, 0, 0]
        }
    }
}
