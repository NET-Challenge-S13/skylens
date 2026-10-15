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

//! 깊이맵 융합: 이웃의 이웃으로 깊이 우선 확장하며 일치 픽셀을 모아 성분별 중앙값 점을 만든다.
//!
//! 영상 순서: 첫 사용 영상에서 시작해, 끝낸 영상의 겹침 목록에서 처음 나오는 미완료 영상(없으면 색인이 가장 작은
//! 미완료 영상)으로 넘어간다. 한 영상 안의 시작 픽셀은 행 띠 단위로 병렬 처리하며, "융합됨" 표시는 원자적으로
//! 점유한다(먼저 점유한 확장이 픽셀을 가진다). `threads == 1` 이면 행 우선 순차 처리와 같다.

use crate::params::{FusionMode, FusionParams, FusionResidual};
use crate::scene::DenseScene;
use cumulus3d_core::io::PointCloud;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::time::{Duration, Instant};

/// 융합 입력: 뷰 크기의 깊이(0 = 무효)와 기준 카메라 좌표 법선.
#[derive(Clone, Copy, Debug)]
pub struct FusionInput<'a> {
    /// 뷰 크기의 깊이(0 = 무효).
    pub depth: &'a [f32],
    /// 기준 카메라 좌표 법선.
    pub normal: &'a [[f32; 3]],
    /// 픽셀별 최종 정합 비용(있으면 역분산 가중에 쓴다).
    pub cost: Option<&'a [f32]>,
    /// 뷰의 대표 기준선(원천 뷰와의 중심 거리 중앙값). 0 이면 가중 없음.
    pub baseline: f64,
}

impl<'a> FusionInput<'a> {
    /// 가중 정보 없는 입력.
    pub fn plain(depth: &'a [f32], normal: &'a [[f32; 3]]) -> Self {
        Self { depth, normal, cost: None, baseline: 0.0 }
    }
}

/// 스테레오 오차 전파 깊이 분산의 역수: σ_d = d²·σ_px/(f·b), σ_px = σ0 + s·비용.
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

/// 융합 결과(점, 점마다 기여 뷰 색인 오름차순).
#[derive(Clone, Debug, Default)]
pub struct FusionOutput {
    /// 융합된 점군.
    pub cloud: PointCloud,
    /// 점마다 기여 뷰 색인(오름차순).
    pub visibility: Vec<Vec<u32>>,
    /// 점마다 2차 융합 점 여부(2차 융합을 하지 않았으면 비어 있다).
    pub residual: Vec<bool>,
    /// 1차 융합 시간(일치 융합만 기록).
    pub pass1_time: Duration,
    /// 2차 융합 시간(2차 융합을 했을 때만).
    pub pass2_time: Duration,
}

impl FusionOutput {
    /// 2차 융합 점 수.
    pub fn num_residual(&self) -> usize {
        self.residual.iter().filter(|&&r| r).count()
    }
    /// 2차 융합 점만 모은 점군.
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

/// 깊이맵 융합. `inputs[v]` 가 None 이면 뷰 v 는 쓰지 않는다. `overlap[v]` 는 확장 대상 뷰 순서.
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

/// 일치 융합: 뷰를 색인 순서로, 뷰 안의 행은 병렬. 기준 픽셀의 3D 점을 겹침 뷰(앞 `consistency_num_images` 개)에 투영해
/// 최근접 픽셀 깊이가 상대 허용치 안이고, 그 픽셀 점을 기준에 되투영한 오차가 허용치 안이며, 법선 각이 허용치 안이면 일치.
/// 일치 뷰가 `min_consistent_views` 이상이면 기준과 일치 점들의 평균 위치·법선·색을 점으로 낸다.
/// `mark_used` 면 일치에 쓰인 다른 뷰 픽셀은 그 뷰의 기준 픽셀로 다시 쓰지 않는다(앞 뷰가 정하므로 결정적).
///
/// 남은 픽셀 처리(`residual`):
/// - `Release`: 후보가 검사 중 점유한 픽셀(기준·이웃)을 채택 실패 시 되돌린다. 이 구현은 표시를 채택이 확정된 뒤에만
///   남기므로 결과는 `None` 과 같다(버려진 후보가 표시를 남기지 않음을 명시적으로 보장하는 경로).
/// - `SecondPass`: 1차가 끝난 뒤, 1차 점의 기준·일치 픽셀이 아니었던 유효 깊이 픽셀만 기준으로 삼아 2차 허용치로 다시 검사한다.
///   영상 단위 병렬이며, 2차 후보는 자기 기준 픽셀과 일치 픽셀을 후보 번호의 최솟값으로 원자적 점유(`fetch_min`)하고
///   자기 기준 픽셀의 점유자가 자신인 후보만 남는다(실행 순서와 무관). 1차 점과 `min_dist_gsd`·GSD 이내인 후보는 버린다.
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
    // 1차 점의 기준·일치 픽셀(2차 대상 제외용).
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
                            // 검사한 후보의 픽셀을 임시 점유(2)했다가, 채택되면 확정(1), 아니면 이 후보가 점유한 것만 되돌린다.
                            // 확정 표시(1)는 되돌리지 않으므로 같은 픽셀을 함께 쓴 다른 후보의 채택 표시는 지워지지 않는다.
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
            out.visibility.extend(vis);
        }
        out.pass2_time = t2.elapsed();
    }
    out
}

/// 남은 픽셀 `Release` 처리의 임시 점유 표시 값.
const TENTATIVE: u8 = 2;

/// 채택된 점의 일치 픽셀을 "사용됨"으로 표시(반경이 있으면 깊이가 일치하는 주변 픽셀까지).
fn mark_hits(p: &FusionParams, cams: &[Cam], inputs: &[Option<FusionInput<'_>>], marks: &[Vec<AtomicU8>], hits: &[(usize, usize)]) {
    let r = p.mark_radius as i64;
    for &(o, opix) in hits {
        if r == 0 {
            marks[o][opix].store(1, Ordering::Relaxed);
            continue;
        }
        // 반경 안에서 깊이가 일치하는 픽셀까지 표시(같은 표면 조각의 중복 기준 픽셀 억제).
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

/// 일치 검사 허용치.
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
    /// 기준 픽셀 하나의 일치 검사: 일치한 (뷰, 픽셀)을 `hits` 에 모으고 가중 평균 점을 낸다(법선 합이 0 이면 None).
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

/// 깊이맵의 지상 표본 간격(뷰별 유효 깊이 중앙값/초점의 중앙값).
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

/// 점 위치의 격자 해시(칸 크기 `cell`): 정렬된 (칸 열쇠, 점 번호).
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
    /// `x` 에서 거리 `r`(≤ 칸 크기) 이내 점이 있는지.
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

/// 2차 융합: 1차에서 쓰이지 않은 유효 깊이 픽셀만 기준으로, 2차 허용치로 검사. 뷰별 (점, 가시성) 을 뷰 순서로 돌려준다.
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
    // 픽셀 번호: 뷰 시작 오프셋 + 뷰 안 픽셀.
    let mut offset = vec![0u64; n + 1];
    for v in 0..n {
        offset[v + 1] = offset[v] + if used[v] { (cams[v].w * cams[v].h) as u64 } else { 0 };
    }
    assert!(offset[n] < u32::MAX as u64, "2차 융합: 픽셀 수가 u32 범위를 넘음");
    let owner: Vec<Vec<AtomicU32>> =
        (0..n).map(|v| if used[v] { (0..cams[v].w * cams[v].h).map(|_| AtomicU32::new(u32::MAX)).collect() } else { Vec::new() }).collect();
    // 1단계: 뷰 단위 병렬로 후보를 만들고 점유.
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
    // 2단계: 자기 기준 픽셀의 점유자가 자신인 후보만 남긴다(점유 최솟값이라 실행 순서와 무관).
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
                vis.push(c.vis);
            }
            (pc, vis)
        })
        .collect()
}

/// 확장 중앙값 융합.
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
