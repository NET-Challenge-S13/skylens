/*
 * problem.rs
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

//! 번들 조정 문제: 관측별 잔차·야코비안, 정규방정식 블록, Schur 보수 조립과 역대입.
//!
//! 변수 배치: E 블록 = [가변 자세(6) …, 가변 카메라(정제 파라미터 수) …], 점 블록(3) 은 소거.
//! 자세 국소 매개화: 접공간 (δ회전 3, δ이동 3). 회전 갱신 q ← Exp_q(δ) ⊗ q,
//! Exp_q(δ) = [cos‖δ‖, sin‖δ‖ δ/‖δ‖] (라이브러리 쿼터니언 다양체 관례 → 실제 회전각 2‖δ‖).

use crate::linsolve::{BlockSym, SchurSolver};
use crate::tr::{StepInfo, TrProblem};
use crate::{LinearSolverType, Loss};
use cumulus3d_core::geometry::skew;
use cumulus3d_core::{Camera, Mat3, Quat, Rigid3, Vec2, Vec3};
use nalgebra::{Matrix2x3, Matrix3};
use rayon::prelude::*;

pub(crate) const NONE: u32 = u32::MAX;
/// 관측 결과를 결정적으로 합칠 때의 조각 크기.
const CHUNK: usize = 8192;
const MAX_CAM_DIM: usize = 16;

/// 문제 입력(lib.rs 가 재구성에서 만든다). 관측은 점 순서로 정렬돼 있어야 한다.
pub(crate) struct ProblemInput {
    pub poses: Vec<Rigid3>,
    pub pose_const: Vec<bool>,
    /// true = 그 접공간 성분 상수(회전 0..3, 이동 3..6).
    pub pose_mask: Vec<[bool; 6]>,
    pub cams: Vec<Camera>,
    /// 정제할 파라미터 인덱스(비면 상수 카메라).
    pub cam_var: Vec<Vec<usize>>,
    pub img_pose: Vec<u32>,
    pub img_cam: Vec<u32>,
    /// 비기준 센서면 rig_to_sensor.
    pub img_sensor: Vec<Option<Rigid3>>,
    pub points: Vec<Vec3>,
    pub point_const: Vec<bool>,
    pub obs_img: Vec<u32>,
    pub obs_pt: Vec<u32>,
    pub obs_xy: Vec<Vec2>,
    pub loss: Loss,
    pub solver: LinearSolverType,
    pub max_linear_solver_iterations: usize,
}

#[derive(Clone)]
pub(crate) struct State {
    pub poses: Vec<Rigid3>,
    pub rot: Vec<Mat3>,
    pub cams: Vec<Camera>,
    pub pts: Vec<Vec3>,
}

pub(crate) struct BaProblem {
    loss: Loss,
    pub cur: State,
    cand: State,
    pub pose_e: Vec<u32>,
    pose_mask: Vec<[bool; 6]>,
    pub cam_e: Vec<u32>,
    cam_var: Vec<Vec<usize>>,
    img_pose: Vec<u32>,
    img_cam: Vec<u32>,
    img_sensor: Vec<Option<(Mat3, Vec3)>>,
    pub pt_v: Vec<u32>,
    v_pt: Vec<u32>,
    obs_img: Vec<u32>,
    obs_pt: Vec<u32>,
    obs_xy: Vec<Vec2>,
    pt_obs: Vec<usize>,
    // E 배치
    e_dim: Vec<usize>,
    e_off: Vec<usize>,
    n_e: usize,
    n_pose_e: usize,
    e_src: Vec<u32>,
    row_points: Vec<Vec<u32>>,
    b_cross: Vec<Vec<(u32, usize)>>,
    cross_vals: Vec<f64>,
    bdiag_off: Vec<usize>,
    bdiag: Vec<f64>,
    s: BlockSym,
    solver: SchurSolver,
    pub solver_kind: LinearSolverType,
    // 관측별 저장(로버스트·야코비 배율 적용 후)
    cs: usize,
    jp: Vec<[f64; 12]>,
    jx: Vec<[f64; 6]>,
    jc: Vec<f64>,
    res: Vec<[f64; 2]>,
    ocost: Vec<f64>,
    scaled: bool,
    scale_e: Vec<f64>,
    scale_p: Vec<[f64; 3]>,
    g_e: Vec<f64>,
    g_p: Vec<[f64; 3]>,
    v: Vec<[f64; 9]>,
    vinv: Vec<[f64; 9]>,
    hdiag_e: Vec<f64>,
    /// 가변 점별 (카메라 E 블록, W_cj = Σ Jxᵀ Jc 의 pcw_vals 오프셋), 3×nc 행 우선.
    pcw_ptr: Vec<usize>,
    pcw_idx: Vec<(u32, usize)>,
    pcw_vals: Vec<f64>,
    /// 점 우선 Schur 조립 조각(가변 점 범위). 비면 행 우선 조립.
    s_chunks: Vec<(usize, usize)>,
    s_acc: Vec<Vec<f64>>,
    r_acc: Vec<Vec<f64>>,
    /// 점 우선 B 누적 조각(관측 범위)과 누적기 [bdiag | cross | g].
    b_chunks: Vec<(usize, usize)>,
    b_acc: Vec<Vec<f64>>,
    delta_e: Vec<f64>,
    delta_p: Vec<[f64; 3]>,
    rhs: Vec<f64>,
}

/// 라이브러리 쿼터니언 다양체의 ⊞: Exp_q(δ) ⊗ q.
pub(crate) fn quat_plus(q: &Quat, d: &[f64]) -> Quat {
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if n == 0.0 {
        return *q;
    }
    let s = n.sin() / n;
    let dq = Quat::new(n.cos(), s * d[0], s * d[1], s * d[2]);
    dq.hamilton(q)
}

fn det_sum(v: &[f64]) -> f64 {
    let parts: Vec<f64> = v.par_chunks(CHUNK).map(|c| c.iter().sum::<f64>()).collect();
    parts.iter().sum()
}

/// 3×3 대칭 양정치 역행렬(촐레스키). 실패면 None.
fn inv3(m: &[f64; 9]) -> Option<[f64; 9]> {
    let mm = Matrix3::from_row_slice(m);
    let c = mm.cholesky()?;
    let i = c.inverse();
    let mut o = [0.0; 9];
    for r in 0..3 {
        for q in 0..3 {
            o[r * 3 + q] = i[(r, q)];
        }
    }
    o.iter().all(|x| x.is_finite()).then_some(o)
}

/// 가변 슬라이스를 주어진 길이들로 쪼갠다.
fn split_lens(mut s: &mut [f64], lens: impl Iterator<Item = usize>) -> Vec<&mut [f64]> {
    let mut out = Vec::new();
    for l in lens {
        let (a, b) = std::mem::take(&mut s).split_at_mut(l);
        out.push(a);
        s = b;
    }
    out
}

impl BaProblem {
    pub fn new(inp: ProblemInput) -> Option<Self> {
        let np = inp.poses.len();
        let nc = inp.cams.len();
        let npt = inp.points.len();
        let nobs = inp.obs_img.len();
        // E 블록 배치
        let mut e_dim = Vec::new();
        let mut e_src = Vec::new();
        let mut pose_e = vec![NONE; np];
        for p in 0..np {
            if !inp.pose_const[p] {
                pose_e[p] = e_dim.len() as u32;
                e_dim.push(6);
                e_src.push(p);
            }
        }
        let n_pose_e = e_dim.len();
        let mut cam_e = vec![NONE; nc];
        let mut maxc = 1;
        for c in 0..nc {
            if !inp.cam_var[c].is_empty() {
                cam_e[c] = e_dim.len() as u32;
                e_dim.push(inp.cam_var[c].len());
                maxc = maxc.max(inp.cam_var[c].len());
            }
        }
        assert!(maxc <= MAX_CAM_DIM / 2, "카메라 파라미터 수 초과");
        let mut e_off = Vec::with_capacity(e_dim.len());
        let mut n_e = 0;
        for &d in &e_dim {
            e_off.push(n_e);
            n_e += d;
        }
        let ne = e_dim.len();
        // 점
        let mut pt_v = vec![NONE; npt];
        let mut v_pt = Vec::new();
        for j in 0..npt {
            if !inp.point_const[j] {
                pt_v[j] = v_pt.len() as u32;
                v_pt.push(j as u32);
            }
        }
        let mut pt_obs = vec![0usize; npt + 1];
        for k in 0..nobs {
            debug_assert!(k == 0 || inp.obs_pt[k - 1] <= inp.obs_pt[k]);
            pt_obs[inp.obs_pt[k] as usize + 1] += 1;
        }
        for j in 0..npt {
            pt_obs[j + 1] += pt_obs[j];
        }
        let block_pose = |k: usize| pose_e[inp.img_pose[inp.obs_img[k] as usize] as usize];
        let block_cam = |k: usize| cam_e[inp.img_cam[inp.obs_img[k] as usize] as usize];
        // E 블록별 관측
        let mut e_obs: Vec<Vec<u32>> = vec![Vec::new(); ne];
        for k in 0..nobs {
            let pe = block_pose(k);
            if pe != NONE {
                e_obs[pe as usize].push(k as u32);
            }
            let ce = block_cam(k);
            if ce != NONE {
                e_obs[ce as usize].push(k as u32);
            }
        }
        let row_points: Vec<Vec<u32>> = e_obs
            .par_iter()
            .map(|ks| {
                let mut v: Vec<u32> = ks.iter().map(|&k| inp.obs_pt[k as usize]).filter(|&j| pt_v[j as usize] != NONE).collect();
                v.dedup();
                v
            })
            .collect();
        // 자세-카메라 교차 블록
        let mut b_cross: Vec<Vec<(u32, usize)>> = vec![Vec::new(); n_pose_e];
        let mut ncross = 0;
        for (a, bc) in b_cross.iter_mut().enumerate() {
            let mut cams: Vec<u32> = e_obs[a].iter().map(|&k| block_cam(k as usize)).filter(|&c| c != NONE).collect();
            cams.sort_unstable();
            cams.dedup();
            for c in cams {
                bc.push((c, ncross));
                ncross += 6 * e_dim[c as usize];
            }
        }
        let mut bdiag_off = Vec::with_capacity(ne);
        let mut nbd = 0;
        for &d in &e_dim {
            bdiag_off.push(nbd);
            nbd += d * d;
        }
        // S 구조
        let row_cols: Vec<Vec<u32>> = (0..ne)
            .into_par_iter()
            .map(|a| {
                let mut cols = vec![a as u32];
                if a < n_pose_e {
                    cols.extend(b_cross[a].iter().map(|x| x.0));
                }
                for &j in &row_points[a] {
                    for k in pt_obs[j as usize]..pt_obs[j as usize + 1] {
                        let pe = block_pose(k);
                        if pe != NONE && pe as usize > a {
                            cols.push(pe);
                        }
                        let ce = block_cam(k);
                        if ce != NONE && ce as usize > a {
                            cols.push(ce);
                        }
                    }
                }
                cols.sort_unstable();
                cols.dedup();
                cols
            })
            .collect();
        let s = BlockSym::new(e_dim.clone(), row_cols);
        let mut pcw_ptr = vec![0usize];
        let mut pcw_idx = Vec::new();
        let mut npcw = 0;
        for &j in &v_pt {
            let j = j as usize;
            let mut cs: Vec<u32> = (pt_obs[j]..pt_obs[j + 1]).map(block_cam).filter(|&c| c != NONE).collect();
            cs.sort_unstable();
            cs.dedup();
            for c in cs {
                pcw_idx.push((c, npcw));
                npcw += 3 * e_dim[c as usize];
            }
            pcw_ptr.push(pcw_idx.len());
        }
        let solver = SchurSolver::new(inp.solver, &s, inp.max_linear_solver_iterations)?;
        // 점 우선 조립 조각: 조각마다 S 사본을 두므로 메모리 상한(256MB)으로 개수를 정한다.
        // 개수는 문제 크기로만 정해 스레드 수와 무관하게 결정적이다.
        let s_bytes = 8 * s.vals.len().max(1);
        let nch = (256usize << 20) / s_bytes;
        let nch = nch.min(16);
        let mut s_chunks = Vec::new();
        if nch >= 4 && !v_pt.is_empty() && ne > 0 {
            let w: Vec<f64> = v_pt
                .iter()
                .map(|&j| {
                    let l = (pt_obs[j as usize + 1] - pt_obs[j as usize]) as f64;
                    l * l + 1.0
                })
                .collect();
            let total: f64 = w.iter().sum();
            let mut acc = 0.0;
            let mut start = 0;
            for (i, wi) in w.iter().enumerate() {
                acc += wi;
                if acc >= total * (s_chunks.len() + 1) as f64 / nch as f64 && s_chunks.len() + 1 < nch {
                    s_chunks.push((start, i + 1));
                    start = i + 1;
                }
            }
            s_chunks.push((start, v_pt.len()));
        }
        let s_acc = vec![vec![0.0; s.vals.len()]; s_chunks.len()];
        let r_acc = vec![vec![0.0; n_e]; s_chunks.len()];
        let nbch = 32.min(nobs.max(1));
        let b_chunks: Vec<(usize, usize)> = (0..nbch).map(|c| (c * nobs / nbch, (c + 1) * nobs / nbch)).collect();
        let rot: Vec<Mat3> = inp.poses.iter().map(|p| p.rotation_matrix()).collect();
        let cur = State { poses: inp.poses, rot, cams: inp.cams, pts: inp.points };
        let nv = v_pt.len();
        let cs = 2 * maxc;
        Some(Self {
            loss: inp.loss,
            cand: cur.clone(),
            cur,
            pose_e,
            pose_mask: inp.pose_mask,
            cam_e,
            cam_var: inp.cam_var,
            img_pose: inp.img_pose,
            img_cam: inp.img_cam,
            img_sensor: inp.img_sensor.iter().map(|s| s.map(|r| (r.rotation_matrix(), r.translation))).collect(),
            pt_v,
            v_pt,
            obs_img: inp.obs_img,
            obs_pt: inp.obs_pt,
            obs_xy: inp.obs_xy,
            pt_obs,
            e_dim,
            e_off,
            n_e,
            n_pose_e,
            e_src: e_src.iter().map(|&x| x as u32).collect(),
            row_points,
            b_cross,
            cross_vals: vec![0.0; ncross],
            bdiag_off,
            bdiag: vec![0.0; nbd],
            s,
            solver,
            solver_kind: inp.solver,
            cs,
            jp: vec![[0.0; 12]; nobs],
            jx: vec![[0.0; 6]; nobs],
            jc: vec![0.0; nobs * cs],
            res: vec![[0.0; 2]; nobs],
            ocost: vec![0.0; nobs],
            scaled: false,
            scale_e: vec![1.0; n_e],
            scale_p: vec![[1.0; 3]; nv],
            g_e: vec![0.0; n_e],
            g_p: vec![[0.0; 3]; nv],
            v: vec![[0.0; 9]; nv],
            vinv: vec![[0.0; 9]; nv],
            hdiag_e: vec![0.0; n_e],
            pcw_ptr,
            pcw_idx,
            pcw_vals: vec![0.0; npcw],
            s_chunks,
            s_acc,
            r_acc,
            b_acc: vec![Vec::new(); b_chunks.len()],
            b_chunks,
            delta_e: vec![0.0; n_e],
            delta_p: vec![[0.0; 3]; nv],
            rhs: vec![0.0; n_e],
        })
    }

    pub fn num_obs(&self) -> usize {
        self.obs_img.len()
    }
    pub fn num_var_points(&self) -> usize {
        self.v_pt.len()
    }

    /// 관측 k 의 (보정) 잔차와 ρ. `jac` 가 있으면 보정·배율 적용 야코비안을 채운다.
    #[inline]
    pub(crate) fn eval_obs(&self, st: &State, k: usize, jac: Option<(&mut [f64; 12], &mut [f64; 6], &mut [f64])>) -> ([f64; 2], f64) {
        let i = self.obs_img[k] as usize;
        let p = self.img_pose[i] as usize;
        let c = self.img_cam[i] as usize;
        let j = self.obs_pt[k] as usize;
        let rf = &st.rot[p];
        let xw = &st.pts[j];
        let xr_rot = rf * xw;
        let xr = xr_rot + st.poses[p].translation;
        let sensor = self.img_sensor[i];
        let xc = match &sensor {
            Some((rs, ts)) => rs * xr + ts,
            None => xr,
        };
        let cam = &st.cams[c];
        let want = jac.is_some();
        let mut dp = [0.0; MAX_CAM_DIM];
        let n = cam.model.num_params();
        let proj = cam.cam_to_img_with_jacobian(&xc, if want { Some(&mut dp[..2 * n]) } else { None });
        let Some((pix, jxc)) = proj else {
            // 투영 실패: 잔차·야코비안 0.
            if let Some((jp, jx, jc)) = jac {
                *jp = [0.0; 12];
                *jx = [0.0; 6];
                jc.iter_mut().for_each(|x| *x = 0.0);
            }
            return ([0.0; 2], 0.0);
        };
        let r = pix - self.obs_xy[k];
        let (rho, w) = self.loss.weight(r.norm_squared());
        let rr = [r.x * w, r.y * w];
        if let Some((jp, jx, jc)) = jac {
            let m: Matrix2x3<f64> = match &sensor {
                Some((rs, _)) => jxc * rs,
                None => jxc,
            };
            let pe = self.pose_e[p];
            if pe != NONE {
                let jrot = m * skew(&xr_rot) * (-2.0);
                let mask = &self.pose_mask[p];
                let off = self.e_off[pe as usize];
                for row in 0..2 {
                    for q in 0..3 {
                        jp[row * 6 + q] = if mask[q] { 0.0 } else { jrot[(row, q)] * w };
                        jp[row * 6 + 3 + q] = if mask[3 + q] { 0.0 } else { m[(row, q)] * w };
                    }
                }
                if self.scaled {
                    for q in 0..6 {
                        let s = self.scale_e[off + q];
                        jp[q] *= s;
                        jp[6 + q] *= s;
                    }
                }
            } else {
                *jp = [0.0; 12];
            }
            let v = self.pt_v[j];
            if v != NONE {
                let jpt = m * rf;
                let s = if self.scaled { self.scale_p[v as usize] } else { [1.0; 3] };
                for row in 0..2 {
                    for q in 0..3 {
                        jx[row * 3 + q] = jpt[(row, q)] * w * s[q];
                    }
                }
            } else {
                *jx = [0.0; 6];
            }
            let ce = self.cam_e[c];
            if ce != NONE {
                let var = &self.cam_var[c];
                let nv = var.len();
                let off = self.e_off[ce as usize];
                for (q, &pi) in var.iter().enumerate() {
                    let s = if self.scaled { self.scale_e[off + q] } else { 1.0 };
                    jc[q] = dp[pi] * w * s;
                    jc[nv + q] = dp[n + pi] * w * s;
                }
            }
        }
        (rr, rho)
    }

    /// 상태 st 에서의 비용(½Σρ).
    pub(crate) fn cost_at(&self, st: &State) -> f64 {
        let costs: Vec<f64> = (0..self.num_obs()).into_par_iter().with_min_len(1024).map(|k| self.eval_obs(st, k, None).1).collect();
        0.5 * det_sum(&costs)
    }

    fn evaluate_jacobians(&mut self) -> Option<f64> {
        let cs = self.cs;
        let mut jp = std::mem::take(&mut self.jp);
        let mut jx = std::mem::take(&mut self.jx);
        let mut jc = std::mem::take(&mut self.jc);
        let mut res = std::mem::take(&mut self.res);
        let mut ocost = std::mem::take(&mut self.ocost);
        {
            let this = &*self;
            jp.par_iter_mut()
                .zip(jx.par_iter_mut())
                .zip(jc.par_chunks_mut(cs))
                .zip(res.par_iter_mut())
                .zip(ocost.par_iter_mut())
                .enumerate()
                .with_min_len(512)
                .for_each(|(k, ((((a, b), c), r), oc))| {
                    let (rr, rho) = this.eval_obs(&this.cur, k, Some((a, b, c)));
                    *r = rr;
                    *oc = rho;
                });
        }
        self.jp = jp;
        self.jx = jx;
        self.jc = jc;
        self.res = res;
        self.ocost = ocost;
        let cost = 0.5 * det_sum(&self.ocost);
        cost.is_finite().then_some(cost)
    }

    /// B 블록, 교차 블록, g_E, 점별 V·g_p.
    fn build_normal(&mut self) {
        let cs = self.cs;
        let nbd = self.bdiag.len();
        let ncr = self.cross_vals.len();
        let ne = self.n_e;
        let len = nbd + ncr + ne;
        // 점 우선(관측 순서) 누적: 관측을 연속으로 읽고, 조각별 누적기를 정해진 순서로 합친다.
        let mut b_acc = std::mem::take(&mut self.b_acc);
        b_acc.par_iter_mut().zip(self.b_chunks.par_iter()).for_each(|(acc, &(k0, k1))| {
            acc.clear();
            acc.resize(len, 0.0);
            let (bd, rest) = acc.split_at_mut(nbd);
            let (cr, g) = rest.split_at_mut(ncr);
            for k in k0..k1 {
                let im = self.obs_img[k] as usize;
                let r = &self.res[k];
                let pe = self.pose_e[self.img_pose[im] as usize];
                let ce = self.cam_e[self.img_cam[im] as usize];
                if pe != NONE {
                    let a = pe as usize;
                    let j = &self.jp[k];
                    let o = self.bdiag_off[a];
                    let og = self.e_off[a];
                    for p in 0..6 {
                        g[og + p] += j[p] * r[0] + j[6 + p] * r[1];
                        for q in p..6 {
                            bd[o + p * 6 + q] += j[p] * j[q] + j[6 + p] * j[6 + q];
                        }
                    }
                    if ce != NONE {
                        let nv = self.e_dim[ce as usize];
                        let off = self.b_cross[a].iter().find(|x| x.0 == ce).expect("교차 구조").1;
                        let jcm = &self.jc[k * cs..k * cs + 2 * nv];
                        for p in 0..6 {
                            for q in 0..nv {
                                cr[off + p * nv + q] += j[p] * jcm[q] + j[6 + p] * jcm[nv + q];
                            }
                        }
                    }
                }
                if ce != NONE {
                    let a = ce as usize;
                    let d = self.e_dim[a];
                    let o = self.bdiag_off[a];
                    let og = self.e_off[a];
                    let jcm = &self.jc[k * cs..k * cs + 2 * d];
                    for p in 0..d {
                        g[og + p] += jcm[p] * r[0] + jcm[d + p] * r[1];
                        for q in p..d {
                            bd[o + p * d + q] += jcm[p] * jcm[q] + jcm[d + p] * jcm[d + q];
                        }
                    }
                }
            }
        });
        {
            let mut out = vec![0.0; len];
            out.par_chunks_mut(4096).enumerate().for_each(|(ci, o)| {
                let base = ci * 4096;
                for acc in &b_acc {
                    let n = o.len();
                    for (x, y) in o.iter_mut().zip(&acc[base..base + n]) {
                        *x += y;
                    }
                }
            });
            self.bdiag.copy_from_slice(&out[..nbd]);
            self.cross_vals.copy_from_slice(&out[nbd..nbd + ncr]);
            self.g_e.copy_from_slice(&out[nbd + ncr..]);
        }
        self.b_acc = b_acc;
        for a in 0..self.e_dim.len() {
            let d = self.e_dim[a];
            let o = self.bdiag_off[a];
            for p in 0..d {
                for q in 0..p {
                    self.bdiag[o + p * d + q] = self.bdiag[o + q * d + p];
                }
                self.hdiag_e[self.e_off[a] + p] = self.bdiag[o + p * d + p];
            }
        }
        // 점: V, g_p, 카메라 교차 W_cj
        let mut v = std::mem::take(&mut self.v);
        let mut gp = std::mem::take(&mut self.g_p);
        let mut pcw = std::mem::take(&mut self.pcw_vals);
        {
            let lens = (0..self.v_pt.len()).map(|vi| {
                let r = self.pcw_ptr[vi]..self.pcw_ptr[vi + 1];
                self.pcw_idx[r].iter().map(|x| 3 * self.e_dim[x.0 as usize]).sum::<usize>()
            });
            let slices = split_lens(&mut pcw, lens);
            v.par_iter_mut().zip(gp.par_iter_mut()).zip(slices.into_par_iter()).enumerate().with_min_len(256).for_each(
                |(vi, ((vm, g), wc))| {
                    let j = self.v_pt[vi] as usize;
                    let ents = &self.pcw_idx[self.pcw_ptr[vi]..self.pcw_ptr[vi + 1]];
                    let base = ents.first().map(|x| x.1).unwrap_or(0);
                    wc.iter_mut().for_each(|x| *x = 0.0);
                    let mut m = [0.0; 9];
                    let mut gg = [0.0; 3];
                    for k in self.pt_obs[j]..self.pt_obs[j + 1] {
                        let jx = &self.jx[k];
                        let r = &self.res[k];
                        for p in 0..3 {
                            gg[p] += jx[p] * r[0] + jx[3 + p] * r[1];
                            for q in 0..3 {
                                m[p * 3 + q] += jx[p] * jx[q] + jx[3 + p] * jx[3 + q];
                            }
                        }
                        let ce = self.cam_e[self.img_cam[self.obs_img[k] as usize] as usize];
                        if ce != NONE {
                            let off = ents.iter().find(|x| x.0 == ce).map(|x| x.1).unwrap_or(base) - base;
                            let nv = self.e_dim[ce as usize];
                            let jcm = &self.jc[k * cs..k * cs + 2 * nv];
                            for p in 0..3 {
                                for q in 0..nv {
                                    wc[off + p * nv + q] += jx[p] * jcm[q] + jx[3 + p] * jcm[nv + q];
                                }
                            }
                        }
                    }
                    *vm = m;
                    *g = gg;
                },
            );
        }
        self.pcw_vals = pcw;
        self.v = v;
        self.g_p = gp;
    }

    /// Schur 보수 한 블록 행(a) 조립. `vals` = 행 a 의 값, `rhs` = 행 a 의 우변.
    ///
    /// 행 단위로 쓰기가 겹치지 않아 행 병렬·결정적이다. 자세-자세 항은 관측별 6×2·2×6,
    /// 자세-카메라·카메라-카메라 항은 점별로 미리 합친 W_cj(3×nc) 로 계산한다.
    fn assemble_row(&self, a: usize, vals: &mut [f64], rhs: &mut [f64], damp_e: &[f64], slot: &mut [usize]) {
        let da = self.e_dim[a];
        let oa = self.e_off[a];
        let rv0 = self.s.row_val[a];
        for e in self.s.row_ptr[a]..self.s.row_ptr[a + 1] {
            slot[self.s.cols[e] as usize] = self.s.val_off[e] - rv0;
        }
        vals.iter_mut().for_each(|x| *x = 0.0);
        let bd = &self.bdiag[self.bdiag_off[a]..self.bdiag_off[a] + da * da];
        vals[..da * da].copy_from_slice(bd);
        for i in 0..da {
            vals[i * da + i] += damp_e[oa + i];
            rhs[i] = -self.g_e[oa + i];
        }
        let au = a as u32;
        if a < self.n_pose_e {
            for &(ce, off) in &self.b_cross[a] {
                let o = slot[ce as usize];
                let n = 6 * self.e_dim[ce as usize];
                vals[o..o + n].iter_mut().zip(&self.cross_vals[off..off + n]).for_each(|(x, y)| *x += y);
            }
            let mut rhs6 = [0.0; 6];
            for &j in &self.row_points[a] {
                let j = j as usize;
                let vi = self.pt_v[j] as usize;
                let vinv = &self.vinv[vi];
                let gj = &self.g_p[vi];
                let range = self.pt_obs[j]..self.pt_obs[j + 1];
                let mut w = [0.0; 18];
                for k in range.clone() {
                    if self.pose_e[self.img_pose[self.obs_img[k] as usize] as usize] != au {
                        continue;
                    }
                    let (ja, jx) = (&self.jp[k], &self.jx[k]);
                    for r in 0..6 {
                        for c in 0..3 {
                            w[r * 3 + c] += ja[r] * jx[c] + ja[6 + r] * jx[3 + c];
                        }
                    }
                }
                let mut t = [0.0; 18];
                for r in 0..6 {
                    for c in 0..3 {
                        t[r * 3 + c] = w[r * 3] * vinv[c] + w[r * 3 + 1] * vinv[3 + c] + w[r * 3 + 2] * vinv[6 + c];
                    }
                    rhs6[r] += t[r * 3] * gj[0] + t[r * 3 + 1] * gj[1] + t[r * 3 + 2] * gj[2];
                }
                for k in range {
                    let pe = self.pose_e[self.img_pose[self.obs_img[k] as usize] as usize];
                    if pe == NONE || pe < au {
                        continue;
                    }
                    let jx = &self.jx[k];
                    let mut u = [0.0; 12];
                    for r in 0..6 {
                        for row in 0..2 {
                            u[r * 2 + row] = t[r * 3] * jx[row * 3] + t[r * 3 + 1] * jx[row * 3 + 1] + t[r * 3 + 2] * jx[row * 3 + 2];
                        }
                    }
                    let jp = &self.jp[k];
                    let o = slot[pe as usize];
                    let blk: &mut [f64; 36] = (&mut vals[o..o + 36]).try_into().expect("6x6");
                    for r in 0..6 {
                        let (u0, u1) = (u[r * 2], u[r * 2 + 1]);
                        for c in 0..6 {
                            blk[r * 6 + c] -= u0 * jp[c] + u1 * jp[6 + c];
                        }
                    }
                }
                for q in self.pcw_ptr[vi]..self.pcw_ptr[vi + 1] {
                    let (ce, off) = self.pcw_idx[q];
                    let nv = self.e_dim[ce as usize];
                    let wc = &self.pcw_vals[off..off + 3 * nv];
                    let o = slot[ce as usize];
                    let blk = &mut vals[o..o + 6 * nv];
                    for r in 0..6 {
                        let (t0, t1, t2) = (t[r * 3], t[r * 3 + 1], t[r * 3 + 2]);
                        for c in 0..nv {
                            blk[r * nv + c] -= t0 * wc[c] + t1 * wc[nv + c] + t2 * wc[2 * nv + c];
                        }
                    }
                }
            }
            for i in 0..6 {
                rhs[i] += rhs6[i];
            }
            // 상수 성분: 야코비안 열이 0 이라 행·열이 0 → 대각 1, 우변 0 으로 두면 δ = 0.
            let p = self.e_src[a] as usize;
            for (i, &m) in self.pose_mask[p].iter().enumerate() {
                if m {
                    vals[i * da + i] = 1.0;
                    rhs[i] = 0.0;
                }
            }
        } else {
            let mut t = [0.0; 3 * MAX_CAM_DIM];
            for &j in &self.row_points[a] {
                let j = j as usize;
                let vi = self.pt_v[j] as usize;
                let vinv = &self.vinv[vi];
                let gj = &self.g_p[vi];
                let ents = &self.pcw_idx[self.pcw_ptr[vi]..self.pcw_ptr[vi + 1]];
                let Some(&(_, own)) = ents.iter().find(|x| x.0 == au) else { continue };
                let wa = &self.pcw_vals[own..own + 3 * da];
                // t = W_aᵀ V⁻¹ (da×3)
                for r in 0..da {
                    let mut acc = 0.0;
                    for c in 0..3 {
                        let x = wa[r] * vinv[c] + wa[da + r] * vinv[3 + c] + wa[2 * da + r] * vinv[6 + c];
                        t[r * 3 + c] = x;
                        acc += x * gj[c];
                    }
                    rhs[r] += acc;
                }
                for &(ce, off) in ents {
                    if ce < au {
                        continue;
                    }
                    let nv = self.e_dim[ce as usize];
                    let wc = &self.pcw_vals[off..off + 3 * nv];
                    let o = slot[ce as usize];
                    let blk = &mut vals[o..o + da * nv];
                    for r in 0..da {
                        let (t0, t1, t2) = (t[r * 3], t[r * 3 + 1], t[r * 3 + 2]);
                        for c in 0..nv {
                            blk[r * nv + c] -= t0 * wc[c] + t1 * wc[nv + c] + t2 * wc[2 * nv + c];
                        }
                    }
                }
            }
        }
    }

    fn x_norm(&self) -> f64 {
        let mut s = 0.0;
        for p in &self.cur.poses {
            let q = p.rotation;
            s += q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z + p.translation.norm_squared();
        }
        for c in &self.cur.cams {
            s += c.params.iter().map(|x| x * x).sum::<f64>();
        }
        let parts: Vec<f64> = self.cur.pts.par_chunks(CHUNK).map(|ch| ch.iter().map(|x| x.norm_squared()).sum::<f64>()).collect();
        s += parts.iter().sum::<f64>();
        s.sqrt()
    }

    /// 후보 상태 = cur ⊞ δ (배율 해제). 반환: ‖x_후보 − x‖².
    fn make_candidate(&mut self) -> f64 {
        let mut sq = 0.0;
        for p in 0..self.cur.poses.len() {
            let pe = self.pose_e[p];
            let cur = self.cur.poses[p];
            if pe == NONE {
                self.cand.poses[p] = cur;
                self.cand.rot[p] = self.cur.rot[p];
                continue;
            }
            let o = self.e_off[pe as usize];
            let d: [f64; 6] = std::array::from_fn(|i| self.delta_e[o + i] * self.scale_e[o + i]);
            let q = quat_plus(&cur.rotation, &d[0..3]).normalized();
            let t = cur.translation + Vec3::new(d[3], d[4], d[5]);
            let np = Rigid3::new(q, t);
            let (a, b) = (cur.rotation, q);
            sq += (a.w - b.w).powi(2) + (a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2);
            sq += (t - cur.translation).norm_squared();
            self.cand.rot[p] = np.rotation_matrix();
            self.cand.poses[p] = np;
        }
        for c in 0..self.cur.cams.len() {
            self.cand.cams[c].params.clone_from(&self.cur.cams[c].params);
            let ce = self.cam_e[c];
            if ce == NONE {
                continue;
            }
            let o = self.e_off[ce as usize];
            for (q, &pi) in self.cam_var[c].iter().enumerate() {
                let d = self.delta_e[o + q] * self.scale_e[o + q];
                self.cand.cams[c].params[pi] += d;
                sq += d * d;
            }
        }
        let mut pts = std::mem::take(&mut self.cand.pts);
        let parts: Vec<f64> = pts
            .par_chunks_mut(CHUNK)
            .enumerate()
            .map(|(ci, ch)| {
                let mut s = 0.0;
                for (i, x) in ch.iter_mut().enumerate() {
                    let j = ci * CHUNK + i;
                    let v = self.pt_v[j];
                    *x = self.cur.pts[j];
                    if v != NONE {
                        let dp = &self.delta_p[v as usize];
                        let sc = &self.scale_p[v as usize];
                        let d = Vec3::new(dp[0] * sc[0], dp[1] * sc[1], dp[2] * sc[2]);
                        *x += d;
                        s += d.norm_squared();
                    }
                }
                s
            })
            .collect();
        self.cand.pts = pts;
        sq + parts.iter().sum::<f64>()
    }
}

impl TrProblem for BaProblem {
    fn linearize(&mut self, first: bool) -> Option<f64> {
        let cost = self.evaluate_jacobians()?;
        self.build_normal();
        if first {
            // 야코비 열 배율 1/(1+‖열‖), 첫 평가에서 한 번 정한다(라이브러리 관례).
            for i in 0..self.n_e {
                self.scale_e[i] = 1.0 / (1.0 + self.hdiag_e[i].sqrt());
            }
            for (s, v) in self.scale_p.iter_mut().zip(&self.v) {
                for q in 0..3 {
                    s[q] = 1.0 / (1.0 + v[q * 4].sqrt());
                }
            }
            self.scaled = true;
            self.evaluate_jacobians()?;
            self.build_normal();
        }
        Some(cost)
    }

    fn grad_inf_norm(&self) -> f64 {
        let mut m: f64 = 0.0;
        for p in 0..self.cur.poses.len() {
            let pe = self.pose_e[p];
            if pe == NONE {
                continue;
            }
            let o = self.e_off[pe as usize];
            let g: [f64; 6] = std::array::from_fn(|i| self.g_e[o + i] / self.scale_e[o + i]);
            let q = self.cur.poses[p].rotation;
            let qn = quat_plus(&q, &[-g[0], -g[1], -g[2]]);
            m = m.max((q.w - qn.w).abs()).max((q.x - qn.x).abs()).max((q.y - qn.y).abs()).max((q.z - qn.z).abs());
            for &gi in &g[3..6] {
                m = m.max(gi.abs());
            }
        }
        for i in self.n_pose_e..self.e_dim.len() {
            let o = self.e_off[i];
            for q in 0..self.e_dim[i] {
                m = m.max((self.g_e[o + q] / self.scale_e[o + q]).abs());
            }
        }
        let pm = self
            .g_p
            .par_iter()
            .zip(self.scale_p.par_iter())
            .map(|(g, s)| (g[0] / s[0]).abs().max((g[1] / s[1]).abs()).max((g[2] / s[2]).abs()))
            .reduce(|| 0.0, f64::max);
        m.max(pm)
    }

    fn compute_step(&mut self, radius: f64) -> Option<StepInfo> {
        const MIN_D: f64 = 1e-6;
        const MAX_D: f64 = 1e32;
        let damp_e: Vec<f64> = self.hdiag_e.iter().map(|&h| h.clamp(MIN_D, MAX_D) / radius).collect();
        // 점 블록 역행렬(감쇠 포함)
        let mut vinv = std::mem::take(&mut self.vinv);
        let ok = vinv
            .par_iter_mut()
            .zip(self.v.par_iter())
            .with_min_len(1024)
            .map(|(out, v)| {
                let mut m = *v;
                for q in 0..3 {
                    m[q * 4] += v[q * 4].clamp(MIN_D, MAX_D) / radius;
                }
                match inv3(&m) {
                    Some(i) => {
                        *out = i;
                        true
                    }
                    None => false,
                }
            })
            .all(|x| x);
        self.vinv = vinv;
        if !ok {
            return None;
        }
        if !self.s_chunks.is_empty() {
            self.assemble_point_major(&damp_e);
        } else {
            self.assemble_row_major(&damp_e);
        }
        let de = self.solver.solve(&self.s, &self.rhs)?;
        self.delta_e = de;
        self.back_substitute_and_model()
    }

    fn candidate_cost(&mut self) -> Option<f64> {
        let c = self.cost_at(&self.cand);
        c.is_finite().then_some(c)
    }

    fn accept_candidate(&mut self) {
        std::mem::swap(&mut self.cur, &mut self.cand);
    }
}

impl BaProblem {
    /// 행 우선 조립(블록 행 병렬, 행마다 독립 쓰기). S 가 커서 조각 사본을 못 둘 때.
    fn assemble_row_major(&mut self, damp_e: &[f64]) {
        let mut vals = std::mem::take(&mut self.s.vals);
        let mut rhs = std::mem::take(&mut self.rhs);
        {
            let nrow = self.e_dim.len();
            let row_lens = (0..nrow).map(|a| self.s.row_val[a + 1] - self.s.row_val[a]);
            let vrows = split_lens(&mut vals, row_lens);
            let rrows = split_lens(&mut rhs, self.e_dim.iter().copied());
            let this = &*self;
            vrows
                .into_par_iter()
                .zip(rrows.into_par_iter())
                .enumerate()
                .for_each_init(|| vec![0usize; nrow], |slot, (a, (v, r))| this.assemble_row(a, v, r, damp_e, slot));
        }
        self.s.vals = vals;
        self.rhs = rhs;
    }

    /// 점 우선 조립: 점 범위 조각마다 S·우변 사본에 누적 → 조각 순서대로 합산 → 행별 B·감쇠 더함.
    fn assemble_point_major(&mut self, damp_e: &[f64]) {
        let mut s_acc = std::mem::take(&mut self.s_acc);
        let mut r_acc = std::mem::take(&mut self.r_acc);
        s_acc.par_iter_mut().zip(r_acc.par_iter_mut()).zip(self.s_chunks.par_iter()).for_each(|((acc, racc), &(v0, v1))| {
            acc.iter_mut().for_each(|x| *x = 0.0);
            racc.iter_mut().for_each(|x| *x = 0.0);
            self.schur_points(v0, v1, acc, racc);
        });
        let mut vals = std::mem::take(&mut self.s.vals);
        vals.par_chunks_mut(4096).enumerate().for_each(|(ci, o)| {
            let base = ci * 4096;
            o.iter_mut().for_each(|x| *x = 0.0);
            for acc in &s_acc {
                let n = o.len();
                for (x, y) in o.iter_mut().zip(&acc[base..base + n]) {
                    *x += y;
                }
            }
        });
        let mut rhs = std::mem::take(&mut self.rhs);
        rhs.iter_mut().for_each(|x| *x = 0.0);
        for racc in &r_acc {
            rhs.iter_mut().zip(racc).for_each(|(x, y)| *x += y);
        }
        self.s_acc = s_acc;
        self.r_acc = r_acc;
        // 행별 B·감쇠·교차·−g·상수 성분
        {
            let nrow = self.e_dim.len();
            let row_lens = (0..nrow).map(|a| self.s.row_val[a + 1] - self.s.row_val[a]);
            let vrows = split_lens(&mut vals, row_lens);
            let rrows = split_lens(&mut rhs, self.e_dim.iter().copied());
            let this = &*self;
            vrows.into_par_iter().zip(rrows.into_par_iter()).enumerate().for_each(|(a, (v, r))| {
                let da = this.e_dim[a];
                let oa = this.e_off[a];
                let bd = &this.bdiag[this.bdiag_off[a]..this.bdiag_off[a] + da * da];
                for (x, y) in v[..da * da].iter_mut().zip(bd) {
                    *x += y;
                }
                for i in 0..da {
                    v[i * da + i] += damp_e[oa + i];
                    r[i] -= this.g_e[oa + i];
                }
                if a < this.n_pose_e {
                    let rv0 = this.s.row_val[a];
                    for &(ce, off) in &this.b_cross[a] {
                        let o = this.s.offset(a, ce) - rv0;
                        let n = 6 * this.e_dim[ce as usize];
                        v[o..o + n].iter_mut().zip(&this.cross_vals[off..off + n]).for_each(|(x, y)| *x += y);
                    }
                    for (i, &m) in this.pose_mask[this.e_src[a] as usize].iter().enumerate() {
                        if m {
                            v[i * da + i] = 1.0;
                            r[i] = 0.0;
                        }
                    }
                }
            });
        }
        self.s.vals = vals;
        self.rhs = rhs;
    }

    /// 가변 점 v0..v1 의 Schur 기여(−W V⁻¹ Wᵀ, +W V⁻¹ g_p)를 acc·racc 에 더한다.
    fn schur_points(&self, v0: usize, v1: usize, acc: &mut [f64], racc: &mut [f64]) {
        let mut loc: Vec<(u32, usize, [f64; 18])> = Vec::new();
        let mut tc = [0.0; 3 * MAX_CAM_DIM];
        for vi in v0..v1 {
            let j = self.v_pt[vi] as usize;
            let vinv = &self.vinv[vi];
            let gj = &self.g_p[vi];
            loc.clear();
            for k in self.pt_obs[j]..self.pt_obs[j + 1] {
                let pe = self.pose_e[self.img_pose[self.obs_img[k] as usize] as usize];
                if pe == NONE {
                    continue;
                }
                let (ja, jx) = (&self.jp[k], &self.jx[k]);
                let mut w = [0.0; 18];
                for r in 0..6 {
                    for c in 0..3 {
                        w[r * 3 + c] = ja[r] * jx[c] + ja[6 + r] * jx[3 + c];
                    }
                }
                let mut t = [0.0; 18];
                let og = self.e_off[pe as usize];
                for r in 0..6 {
                    for c in 0..3 {
                        t[r * 3 + c] = w[r * 3] * vinv[c] + w[r * 3 + 1] * vinv[3 + c] + w[r * 3 + 2] * vinv[6 + c];
                    }
                    racc[og + r] += t[r * 3] * gj[0] + t[r * 3 + 1] * gj[1] + t[r * 3 + 2] * gj[2];
                }
                loc.push((pe, k, t));
            }
            let ents = &self.pcw_idx[self.pcw_ptr[vi]..self.pcw_ptr[vi + 1]];
            for (pa, _, t) in loc.iter() {
                let a = *pa as usize;
                for (pb, kb, _) in loc.iter() {
                    // 위삼각 블록만. 같은 블록 안의 관측 쌍은 두 순서 모두 더한다(W_a 합과 같음).
                    if *pb < *pa {
                        continue;
                    }
                    let jx = &self.jx[*kb];
                    let mut u = [0.0; 12];
                    for r in 0..6 {
                        for row in 0..2 {
                            u[r * 2 + row] = t[r * 3] * jx[row * 3] + t[r * 3 + 1] * jx[row * 3 + 1] + t[r * 3 + 2] * jx[row * 3 + 2];
                        }
                    }
                    let jp = &self.jp[*kb];
                    let o = self.s.offset(a, *pb);
                    let blk: &mut [f64; 36] = (&mut acc[o..o + 36]).try_into().expect("6x6");
                    for r in 0..6 {
                        let (u0, u1) = (u[r * 2], u[r * 2 + 1]);
                        for c in 0..6 {
                            blk[r * 6 + c] -= u0 * jp[c] + u1 * jp[6 + c];
                        }
                    }
                }
                for &(ce, off) in ents {
                    let nv = self.e_dim[ce as usize];
                    let wc = &self.pcw_vals[off..off + 3 * nv];
                    let o = self.s.offset(a, ce);
                    let blk = &mut acc[o..o + 6 * nv];
                    for r in 0..6 {
                        let (t0, t1, t2) = (t[r * 3], t[r * 3 + 1], t[r * 3 + 2]);
                        for c in 0..nv {
                            blk[r * nv + c] -= t0 * wc[c] + t1 * wc[nv + c] + t2 * wc[2 * nv + c];
                        }
                    }
                }
            }
            for &(ca, offa) in ents {
                let da = self.e_dim[ca as usize];
                let wa = &self.pcw_vals[offa..offa + 3 * da];
                let og = self.e_off[ca as usize];
                for r in 0..da {
                    let mut s = 0.0;
                    for c in 0..3 {
                        let x = wa[r] * vinv[c] + wa[da + r] * vinv[3 + c] + wa[2 * da + r] * vinv[6 + c];
                        tc[r * 3 + c] = x;
                        s += x * gj[c];
                    }
                    racc[og + r] += s;
                }
                for &(cb, offb) in ents {
                    if cb < ca {
                        continue;
                    }
                    let nv = self.e_dim[cb as usize];
                    let wc = &self.pcw_vals[offb..offb + 3 * nv];
                    let o = self.s.offset(ca as usize, cb);
                    let blk = &mut acc[o..o + da * nv];
                    for r in 0..da {
                        let (t0, t1, t2) = (tc[r * 3], tc[r * 3 + 1], tc[r * 3 + 2]);
                        for c in 0..nv {
                            blk[r * nv + c] -= t0 * wc[c] + t1 * wc[nv + c] + t2 * wc[2 * nv + c];
                        }
                    }
                }
            }
        }
    }

    fn back_substitute_and_model(&mut self) -> Option<StepInfo> {
        // 점 역대입: δp = V⁻¹(−g_p − Wᵀ δE)
        let cs = self.cs;
        let mut dp = std::mem::take(&mut self.delta_p);
        dp.par_iter_mut().enumerate().with_min_len(256).for_each(|(vi, out)| {
            let j = self.v_pt[vi] as usize;
            let g = &self.g_p[vi];
            let mut b = [-g[0], -g[1], -g[2]];
            for k in self.pt_obs[j]..self.pt_obs[j + 1] {
                let y = self.e_product(k, cs);
                let jx = &self.jx[k];
                for q in 0..3 {
                    b[q] -= jx[q] * y[0] + jx[3 + q] * y[1];
                }
            }
            let vi_m = &self.vinv[vi];
            for q in 0..3 {
                out[q] = vi_m[q * 3] * b[0] + vi_m[q * 3 + 1] * b[1] + vi_m[q * 3 + 2] * b[2];
            }
        });
        self.delta_p = dp;
        if !self.delta_p.iter().all(|d| d.iter().all(|x| x.is_finite())) {
            return None;
        }
        // 모형 비용 변화 −(rᵀJδ + ½‖Jδ‖²)
        let mc: Vec<f64> = (0..self.num_obs())
            .into_par_iter()
            .with_min_len(1024)
            .map(|k| {
                let mut y = self.e_product(k, cs);
                let v = self.pt_v[self.obs_pt[k] as usize];
                if v != NONE {
                    let d = &self.delta_p[v as usize];
                    let jx = &self.jx[k];
                    y[0] += jx[0] * d[0] + jx[1] * d[1] + jx[2] * d[2];
                    y[1] += jx[3] * d[0] + jx[4] * d[1] + jx[5] * d[2];
                }
                let r = &self.res[k];
                -(r[0] * y[0] + r[1] * y[1] + 0.5 * (y[0] * y[0] + y[1] * y[1]))
            })
            .collect();
        let model_cost_change = det_sum(&mc);
        let x_norm = self.x_norm();
        let step_norm = self.make_candidate().sqrt();
        Some(StepInfo { model_cost_change, step_norm, x_norm })
    }
}

impl BaProblem {
    /// J_E,k δ_E (2).
    #[inline]
    fn e_product(&self, k: usize, cs: usize) -> [f64; 2] {
        let im = self.obs_img[k] as usize;
        let mut y = [0.0; 2];
        let pe = self.pose_e[self.img_pose[im] as usize];
        if pe != NONE {
            let o = self.e_off[pe as usize];
            let jp = &self.jp[k];
            for c in 0..6 {
                y[0] += jp[c] * self.delta_e[o + c];
                y[1] += jp[6 + c] * self.delta_e[o + c];
            }
        }
        let ce = self.cam_e[self.img_cam[im] as usize];
        if ce != NONE {
            let o = self.e_off[ce as usize];
            let nv = self.e_dim[ce as usize];
            let jcm = &self.jc[k * cs..k * cs + 2 * nv];
            for c in 0..nv {
                y[0] += jcm[c] * self.delta_e[o + c];
                y[1] += jcm[nv + c] * self.delta_e[o + c];
            }
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cumulus3d_core::CameraModelKind;

    fn one_obs_problem(pose: Rigid3, sensor: Option<Rigid3>, x: Vec3, xy: Vec2) -> BaProblem {
        let cam = Camera::new(1, CameraModelKind::OpenCv, 1000, 800, vec![900.0, 880.0, 510.0, 395.0, 0.15, -0.12, 0.008, -0.006]).unwrap();
        BaProblem::new(ProblemInput {
            poses: vec![pose],
            pose_const: vec![false],
            pose_mask: vec![[false; 6]],
            cams: vec![cam],
            cam_var: vec![vec![0, 1, 4, 5, 6, 7]],
            img_pose: vec![0],
            img_cam: vec![0],
            img_sensor: vec![sensor],
            points: vec![x],
            point_const: vec![false],
            obs_img: vec![0],
            obs_pt: vec![0],
            obs_xy: vec![xy],
            loss: Loss::Trivial,
            solver: LinearSolverType::DenseSchur,
            max_linear_solver_iterations: 10,
        })
        .unwrap()
    }

    fn residual(p: &BaProblem, st: &State) -> [f64; 2] {
        p.eval_obs(st, 0, None).0
    }

    fn with_pose(st: &State, pose: Rigid3) -> State {
        let mut s = st.clone();
        s.rot[0] = pose.rotation_matrix();
        s.poses[0] = pose;
        s
    }

    #[test]
    fn analytic_jacobian_matches_central_difference() {
        let pose = Rigid3::new(Quat::from_rotation_vector(&Vec3::new(0.2, -0.3, 0.1)), Vec3::new(0.4, -0.2, 1.5));
        let sensor = Rigid3::new(Quat::from_rotation_vector(&Vec3::new(-0.05, 0.1, 0.02)), Vec3::new(0.3, 0.0, -0.1));
        for sensor in [None, Some(sensor)] {
            let x = Vec3::new(1.2, -0.8, 6.0);
            let p = one_obs_problem(pose, sensor, x, Vec2::new(500.0, 400.0));
            let (mut jp, mut jx) = ([0.0; 12], [0.0; 6]);
            let mut jc = vec![0.0; p.cs];
            p.eval_obs(&p.cur, 0, Some((&mut jp, &mut jx, &mut jc)));
            let h = 1e-6;
            let check = |num: [f64; 2], ana: [f64; 2], what: &str| {
                for r in 0..2 {
                    let scale = ana[r].abs().max(1.0);
                    assert!((num[r] - ana[r]).abs() / scale < 1e-6, "{what}: {num:?} vs {ana:?}");
                }
            };
            for c in 0..3 {
                let mut d = [0.0; 3];
                d[c] = h;
                let qp = quat_plus(&pose.rotation, &d);
                d[c] = -h;
                let qm = quat_plus(&pose.rotation, &d);
                let rp = residual(&p, &with_pose(&p.cur, Rigid3::new(qp, pose.translation)));
                let rm = residual(&p, &with_pose(&p.cur, Rigid3::new(qm, pose.translation)));
                check([(rp[0] - rm[0]) / (2.0 * h), (rp[1] - rm[1]) / (2.0 * h)], [jp[c], jp[6 + c]], "rot");
                let mut tp = pose.translation;
                tp[c] += h;
                let mut tm = pose.translation;
                tm[c] -= h;
                let rp = residual(&p, &with_pose(&p.cur, Rigid3::new(pose.rotation, tp)));
                let rm = residual(&p, &with_pose(&p.cur, Rigid3::new(pose.rotation, tm)));
                check([(rp[0] - rm[0]) / (2.0 * h), (rp[1] - rm[1]) / (2.0 * h)], [jp[3 + c], jp[9 + c]], "trans");
                let hx = h * x.norm();
                let mut sp = p.cur.clone();
                sp.pts[0][c] += hx;
                let mut sm = p.cur.clone();
                sm.pts[0][c] -= hx;
                let (rp, rm) = (residual(&p, &sp), residual(&p, &sm));
                check([(rp[0] - rm[0]) / (2.0 * hx), (rp[1] - rm[1]) / (2.0 * hx)], [jx[c], jx[3 + c]], "point");
            }
            let var = [0usize, 1, 4, 5, 6, 7];
            for (q, &pi) in var.iter().enumerate() {
                let v0 = p.cur.cams[0].params[pi];
                let hp = h * v0.abs().max(1e-2);
                let mut sp = p.cur.clone();
                sp.cams[0].params[pi] += hp;
                let mut sm = p.cur.clone();
                sm.cams[0].params[pi] -= hp;
                let (rp, rm) = (residual(&p, &sp), residual(&p, &sm));
                check([(rp[0] - rm[0]) / (2.0 * hp), (rp[1] - rm[1]) / (2.0 * hp)], [jc[q], jc[6 + q]], "cam");
            }
        }
    }

    #[test]
    fn projection_failure_gives_zero_residual_and_jacobian() {
        let p = one_obs_problem(Rigid3::identity(), None, Vec3::new(0.1, 0.2, -3.0), Vec2::new(10.0, 20.0));
        let (mut jp, mut jx) = ([1.0; 12], [1.0; 6]);
        let mut jc = vec![1.0; p.cs];
        let (r, rho) = p.eval_obs(&p.cur, 0, Some((&mut jp, &mut jx, &mut jc)));
        assert_eq!(r, [0.0, 0.0]);
        assert_eq!(rho, 0.0);
        assert!(jp.iter().chain(jx.iter()).chain(jc.iter()).all(|&v| v == 0.0));
    }

    #[test]
    fn point_major_and_row_major_schur_agree() {
        // 무작위 다중 영상 문제(일부 자세 성분 고정, 상수 점·카메라 섞음).
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let cams: Vec<Camera> = (0..2)
            .map(|i| {
                Camera::new(i + 1, CameraModelKind::OpenCv, 1000, 800, vec![900.0, 880.0, 500.0, 400.0, 0.1, -0.05, 0.001, 0.002]).unwrap()
            })
            .collect();
        let np = 6;
        let poses: Vec<Rigid3> = (0..np)
            .map(|i| {
                Rigid3::new(
                    Quat::from_rotation_vector(&Vec3::new(0.05 * rnd(), 0.05 * rnd(), 0.05 * rnd())),
                    Vec3::new(i as f64 * 0.5, 0.0, 0.0),
                )
            })
            .collect();
        let npt = 60;
        let points: Vec<Vec3> = (0..npt).map(|_| Vec3::new(4.0 * rnd() + 1.5, 3.0 * rnd(), 8.0 + rnd())).collect();
        let (mut oi, mut op, mut ox) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..npt {
            for i in 0..np {
                if (i + j) % 3 != 0 || j % 7 == 0 {
                    // 영상 i (자세 i, 카메라 i%2), 같은 자세 다른 영상도 섞이도록 영상 = 자세.
                    oi.push(i as u32);
                    op.push(j as u32);
                    ox.push(Vec2::new(500.0 + 300.0 * rnd(), 400.0 + 300.0 * rnd()));
                }
            }
        }
        let mut mask = vec![[false; 6]; np];
        mask[1][4] = true;
        let mk = || ProblemInput {
            poses: poses.clone(),
            pose_const: (0..np).map(|i| i == 0).collect(),
            pose_mask: mask.clone(),
            cams: cams.clone(),
            cam_var: vec![vec![0, 1, 4, 5, 6, 7], vec![0, 1]],
            img_pose: (0..np as u32).collect(),
            img_cam: (0..np as u32).map(|i| i % 2).collect(),
            img_sensor: vec![None; np],
            points: points.clone(),
            point_const: (0..npt).map(|j| j % 11 == 0).collect(),
            obs_img: oi.clone(),
            obs_pt: op.clone(),
            obs_xy: ox.clone(),
            loss: Loss::Cauchy(2.0),
            solver: LinearSolverType::DenseSchur,
            max_linear_solver_iterations: 10,
        };
        let mut p = BaProblem::new(mk()).unwrap();
        assert!(!p.s_chunks.is_empty());
        p.linearize(true).unwrap();
        let damp: Vec<f64> = p.hdiag_e.iter().map(|h| h.clamp(1e-6, 1e32) / 1e3).collect();
        for (vi, v) in p.v.iter().enumerate() {
            let mut m = *v;
            for q in 0..3 {
                m[q * 4] += v[q * 4].clamp(1e-6, 1e32) / 1e3;
            }
            p.vinv[vi] = inv3(&m).unwrap();
        }
        p.assemble_row_major(&damp);
        let (s1, r1) = (p.s.vals.clone(), p.rhs.clone());
        p.assemble_point_major(&damp);
        let scale = s1.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        for (a, b) in s1.iter().zip(&p.s.vals) {
            assert!((a - b).abs() <= 1e-12 * scale, "{a} vs {b}");
        }
        let rs = r1.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        for (a, b) in r1.iter().zip(&p.rhs) {
            assert!((a - b).abs() <= 1e-12 * rs, "{a} vs {b}");
        }
    }

    #[test]
    fn quat_plus_rotates_by_twice_delta() {
        let q = Quat::from_rotation_vector(&Vec3::new(0.3, 0.1, -0.2));
        let d = [0.01, -0.02, 0.005];
        let qp = quat_plus(&q, &d);
        let expect = Quat::from_rotation_vector(&(Vec3::new(d[0], d[1], d[2]) * 2.0)).hamilton(&q);
        assert!(qp.angular_distance(&expect) < 1e-12);
    }
}
