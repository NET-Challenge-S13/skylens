/*
 * ba.rs
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

//! `cumulus3d_ba::BaBackend` 의 CUDA 구현.
//!
//! 반복마다 무거운 일은 GPU 가 한다: 관측별 잔차·견고 손실·야코비안, 정규방정식(H, g_E, 점별 V·g_p·W),
//! 점 블록 역행렬, Schur 보수 축소 계통 조립(밀집 n_E×n_E), 축소 계통 풀이(크면 블록 야코비 PCG),
//! 점 역대입, 모형 비용 변화, 후보 비용. 신뢰 영역 LM 바깥 루프(`cumulus3d_ba::minimize`)와
//! 자세·카메라 후보 만들기(수십 개 값)는 CPU 가 한다. 축소 계통이 작으면(n_E ≤ [`CPU_CHOLESKY_N`])
//! 내려받아 CPU 밀집 촐레스키로 푼다(CPU 백엔드의 밀집 풀이기와 같은 정확한 해).
//!
//! 정밀도: 전부 f64. 잔차는 픽셀 좌표(~10³)의 차이라 f32 로는 반올림이 1e−4 px 수준이 되고, 축소 계통은
//! 초점·깊이 결합과 게이지 때문에 조건수가 커서 f32 누적이 LM 수락 판정을 흔든다.
//! H·S·우변 누적은 원자적 덧셈이라 합산 순서가 실행마다 달라 마지막 자리 수준의 차이가 날 수 있다.

use crate::device::{CudaDevice, GpuError};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg};
use cumulus3d_ba::{
    minimize, quat_plus, BaBackend, BaSolution, CpuBaBackend, LinearSolverType, Loss, ProblemInput, StepInfo, TrOptions, TrProblem,
};
use cumulus3d_core::{Camera, Rigid3, Vec3};
use nalgebra::{DMatrix, DVector};
use std::sync::{Arc, Mutex};

pub(crate) const SRC: &str = include_str!("kernels/ba.cu");
const BS: u32 = 256;
/// 격자 상한(부분합 배열 길이).
const MAX_GRID: usize = 1024;
/// 카메라 정제 파라미터 상한(커널 MAXC 와 같다).
const MAXC: usize = 8;
/// 밀집 축소 계통 최대 크기. H·S 두 벌 f64 = 2·n²·8 바이트(6000 이면 약 576MB).
pub const MAX_DENSE_N: usize = 6000;
/// 이 크기 이하면 축소 계통을 내려받아 CPU 밀집 촐레스키로 푼다(50 영상 ≈ 306).
pub const CPU_CHOLESKY_N: usize = 512;
/// PCG 이차 모형 상대 감소 기준(CPU 반복 풀이기와 같다).
const PCG_ETA: f64 = 0.1;

/// CUDA 번들 조정 오류.
#[derive(Debug, thiserror::Error)]
pub enum CudaBaError {
    /// 장치·컴파일 오류.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// 드라이버 호출 오류.
    #[error("CUDA 드라이버 오류: {0}")]
    Driver(#[from] DriverError),
    /// 이 백엔드가 다루지 않는 문제(너무 큰 축소 계통 등).
    #[error("{0}")]
    Unsupported(String),
}

struct Kernels {
    eval: CudaFunction,
    img_normal: CudaFunction,
    point_normal: CudaFunction,
    set_scales: CudaFunction,
    vinv: CudaFunction,
    s_init: CudaFunction,
    schur: CudaFunction,
    symmetrize: CudaFunction,
    fix_mask: CudaFunction,
    diag_blocks: CudaFunction,
    matvec: CudaFunction,
    precond: CudaFunction,
    dot: CudaFunction,
    dot_xbr: CudaFunction,
    axpy2: CudaFunction,
    xpby: CudaFunction,
    backsub: CudaFunction,
    model_cost: CudaFunction,
    gp_max: CudaFunction,
}

/// CUDA 번들 조정 백엔드. 실패하면 경고를 찍고 CPU 백엔드로 같은 문제를 다시 푼다.
pub struct CudaBundleAdjuster {
    dev: Arc<CudaDevice>,
    k: Kernels,
    /// 한 번에 하나만 장치를 쓴다(스트림 공유).
    lock: Mutex<()>,
}

impl CudaBundleAdjuster {
    /// 장치 `dev` 에 BA 커널을 컴파일해 만든다.
    pub fn new(dev: Arc<CudaDevice>) -> Result<Self, GpuError> {
        let m = dev.compile(SRC, true, &[])?;
        let f = |name: &str| m.load_function(name);
        let k = Kernels {
            eval: f("ba_eval")?,
            img_normal: f("ba_img_normal")?,
            point_normal: f("ba_point_normal")?,
            set_scales: f("ba_set_scales")?,
            vinv: f("ba_vinv")?,
            s_init: f("ba_s_init")?,
            schur: f("ba_schur")?,
            symmetrize: f("ba_symmetrize")?,
            fix_mask: f("ba_fix_mask")?,
            diag_blocks: f("ba_diag_blocks")?,
            matvec: f("ba_matvec")?,
            precond: f("ba_precond")?,
            dot: f("ba_dot")?,
            dot_xbr: f("ba_dot_xbr")?,
            axpy2: f("ba_axpy2")?,
            xpby: f("ba_xpby")?,
            backsub: f("ba_backsub")?,
            model_cost: f("ba_model_cost")?,
            gp_max: f("ba_gp_max")?,
        };
        Ok(Self { dev, k, lock: Mutex::new(()) })
    }

    /// 장치 0 으로 만든다.
    pub fn try_default() -> Result<Self, GpuError> {
        Self::new(CudaDevice::new(0)?)
    }

    /// GPU 로만 푼다(대체 경로 없음). 테스트·비교용.
    pub fn solve_gpu(&self, input: &ProblemInput, opts: &TrOptions) -> Result<BaSolution, CudaBaError> {
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut prob = GpuProblem::new(self.dev.stream.clone(), &self.k, input)?;
        let result = minimize(&mut prob, opts);
        if let Some(e) = prob.err.take() {
            return Err(e);
        }
        let points = prob.download_points()?;
        let linear_solver = if prob.dense_cpu { LinearSolverType::DenseSchur } else { LinearSolverType::IterativeSchur };
        Ok(BaSolution { poses: prob.cur.poses, cams: prob.cur.cams, points, result, linear_solver })
    }
}

impl BaBackend for CudaBundleAdjuster {
    fn name(&self) -> &str {
        "cuda"
    }

    fn solve(&self, input: &ProblemInput, opts: &TrOptions) -> cumulus3d_core::Result<BaSolution> {
        let t0 = std::time::Instant::now();
        match self.solve_gpu(input, opts) {
            Ok(s) => {
                if crate::sparse::compare_enabled() {
                    let g = t0.elapsed();
                    let t1 = std::time::Instant::now();
                    let _ = CpuBaBackend.solve(input, opts);
                    crate::sparse::compare_line("bundle-adjust", input_size(input), g, Some(t1.elapsed()));
                }
                Ok(s)
            }
            Err(e) => {
                eprintln!("경고: CUDA 번들 조정 실패({e}), CPU 로 계산");
                CpuBaBackend.solve(input, opts)
            }
        }
    }
}

fn launch_cfg(grid: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (grid as u32, 1, 1), block_dim: (BS, 1, 1), shared_mem_bytes: 0 }
}

/// 격자 스트라이드 커널의 블록 수(부분합 길이). 크기로만 정해 결정적이다.
fn grid_for(n: usize) -> usize {
    n.div_ceil(BS as usize).clamp(1, MAX_GRID)
}

fn up_i32(s: &Arc<CudaStream>, v: &[i32]) -> Result<CudaSlice<i32>, DriverError> {
    if v.is_empty() {
        s.alloc_zeros::<i32>(1)
    } else {
        s.clone_htod(v)
    }
}

fn up_f64(s: &Arc<CudaStream>, v: &[f64]) -> Result<CudaSlice<f64>, DriverError> {
    if v.is_empty() {
        s.alloc_zeros::<f64>(1)
    } else {
        s.clone_htod(v)
    }
}

fn to_i32(v: &[u32]) -> Vec<i32> {
    v.iter().map(|&x| x as i32).collect()
}

fn zeros(s: &Arc<CudaStream>, n: usize) -> Result<CudaSlice<f64>, DriverError> {
    s.alloc_zeros::<f64>(n.max(1))
}

/// 부분합 앞 g 개를 내려받아 순서대로 더한다.
fn sum_partial(s: &Arc<CudaStream>, p: &CudaSlice<f64>, g: usize) -> Result<f64, DriverError> {
    let mut h = vec![0.0f64; g];
    s.memcpy_dtoh(&p.slice(0..g), &mut h)?;
    s.synchronize()?;
    Ok(h.iter().sum())
}

fn max_partial(s: &Arc<CudaStream>, p: &CudaSlice<f64>, g: usize) -> Result<f64, DriverError> {
    let mut h = vec![0.0f64; g];
    s.memcpy_dtoh(&p.slice(0..g), &mut h)?;
    s.synchronize()?;
    Ok(h.iter().fold(0.0, |m, &x| m.max(x)))
}

fn read_flag(s: &Arc<CudaStream>, f: &CudaSlice<i32>) -> Result<bool, DriverError> {
    let v = s.clone_dtoh(f)?;
    s.synchronize()?;
    Ok(v[0] != 0)
}

fn pose_arrays(poses: &[Rigid3]) -> (Vec<f64>, Vec<f64>) {
    let mut rot = Vec::with_capacity(9 * poses.len());
    let mut tr = Vec::with_capacity(3 * poses.len());
    for p in poses {
        let m = p.rotation_matrix();
        for r in 0..3 {
            for c in 0..3 {
                rot.push(m[(r, c)]);
            }
        }
        tr.extend_from_slice(&[p.translation.x, p.translation.y, p.translation.z]);
    }
    (rot, tr)
}

fn cam_array(cams: &[Camera]) -> Vec<f64> {
    let mut v = vec![0.0; MAXC * cams.len()];
    for (c, cam) in cams.iter().enumerate() {
        v[MAXC * c..MAXC * c + cam.params.len()].copy_from_slice(&cam.params);
    }
    v
}

/// 장치 버퍼. 구조 배열은 한 번 올리고, 상태는 현재·후보 두 벌.
struct Dev {
    obs_img: CudaSlice<i32>,
    obs_pt: CudaSlice<i32>,
    obs_xy: CudaSlice<f64>,
    img_pose: CudaSlice<i32>,
    img_cam: CudaSlice<i32>,
    img_hs: CudaSlice<i32>,
    img_sr: CudaSlice<f64>,
    img_st: CudaSlice<f64>,
    pose_eoff: CudaSlice<i32>,
    pose_mask: CudaSlice<i32>,
    cam_model: CudaSlice<i32>,
    cam_np: CudaSlice<i32>,
    cam_eoff: CudaSlice<i32>,
    cam_nv: CudaSlice<i32>,
    cam_var: CudaSlice<i32>,
    pt_v: CudaSlice<i32>,
    v_pt: CudaSlice<i32>,
    pt_obs: CudaSlice<i32>,
    img_obs_ptr: CudaSlice<i32>,
    img_obs_idx: CudaSlice<i32>,
    obs_qp: CudaSlice<i32>,
    obs_qc: CudaSlice<i32>,
    pw_ptr: CudaSlice<i32>,
    pw_pt: CudaSlice<i32>,
    pw_eoff: CudaSlice<i32>,
    pw_dim: CudaSlice<i32>,
    pw_voff: CudaSlice<i32>,
    row_blk: CudaSlice<i32>,
    mask_idx: CudaSlice<i32>,
    blk_off: CudaSlice<i32>,
    blk_dim: CudaSlice<i32>,
    blk_poff: CudaSlice<i32>,
    row_off: CudaSlice<i32>,
    row_dim: CudaSlice<i32>,
    row_poff: CudaSlice<i32>,
    // 상태(현재 / 후보)
    rot: CudaSlice<f64>,
    trans: CudaSlice<f64>,
    cams: CudaSlice<f64>,
    pts: CudaSlice<f64>,
    rot_c: CudaSlice<f64>,
    trans_c: CudaSlice<f64>,
    cams_c: CudaSlice<f64>,
    pts_c: CudaSlice<f64>,
    scale_e: CudaSlice<f64>,
    scale_p: CudaSlice<f64>,
    // 관측별(배율·견고 보정 적용 후)
    jp: CudaSlice<f64>,
    jx: CudaSlice<f64>,
    jc: CudaSlice<f64>,
    res: CudaSlice<f64>,
    // 정규방정식·축소 계통
    h: CudaSlice<f64>,
    g_e: CudaSlice<f64>,
    v: CudaSlice<f64>,
    gp: CudaSlice<f64>,
    pw_vals: CudaSlice<f64>,
    vinv: CudaSlice<f64>,
    s: CudaSlice<f64>,
    rhs: CudaSlice<f64>,
    delta_e: CudaSlice<f64>,
    delta_p: CudaSlice<f64>,
    diag: CudaSlice<f64>,
    pinv: CudaSlice<f64>,
    // PCG
    x: CudaSlice<f64>,
    r: CudaSlice<f64>,
    z: CudaSlice<f64>,
    p: CudaSlice<f64>,
    ap: CudaSlice<f64>,
    partial: CudaSlice<f64>,
    flag: CudaSlice<i32>,
}

struct HostState {
    poses: Vec<Rigid3>,
    cams: Vec<Camera>,
}

struct GpuProblem<'a> {
    s: Arc<CudaStream>,
    k: &'a Kernels,
    d: Dev,
    npt: usize,
    nobs: usize,
    nimg: usize,
    nvp: usize,
    n_e: usize,
    cs: usize,
    nent: usize,
    nmask: usize,
    /// E 블록 (스칼라 오프셋, 크기).
    blocks: Vec<(usize, usize)>,
    pinv_len: usize,
    pose_eoff: Vec<i32>,
    cam_eoff: Vec<i32>,
    cam_var: Vec<Vec<usize>>,
    loss: (i32, f64),
    max_lin_iter: usize,
    dense_cpu: bool,
    cur: HostState,
    cand: HostState,
    scale_e: Vec<f64>,
    g_e: Vec<f64>,
    gp_max: f64,
    delta_e: Vec<f64>,
    err: Option<CudaBaError>,
}

impl<'a> GpuProblem<'a> {
    fn new(s: Arc<CudaStream>, k: &'a Kernels, inp: &ProblemInput) -> Result<Self, CudaBaError> {
        let np = inp.poses.len();
        let nc = inp.cams.len();
        let npt = inp.points.len();
        let nobs = inp.obs_img.len();
        let nimg = inp.img_pose.len();
        let lens_ok = inp.pose_const.len() == np
            && inp.pose_mask.len() == np
            && inp.cam_var.len() == nc
            && inp.img_cam.len() == nimg
            && inp.img_sensor.len() == nimg
            && inp.point_const.len() == npt
            && inp.obs_pt.len() == nobs
            && inp.obs_xy.len() == nobs;
        let idx_ok = inp.img_pose.iter().all(|&p| (p as usize) < np) && inp.img_cam.iter().all(|&c| (c as usize) < nc);
        if !lens_ok || !idx_ok {
            return Err(CudaBaError::Unsupported("BA 입력 길이·번호 불일치".into()));
        }
        if nobs >= (i32::MAX as usize) / 16 || npt >= (i32::MAX as usize) / 4 {
            return Err(CudaBaError::Unsupported(format!("관측 {nobs}·점 {npt} 이 너무 많음")));
        }
        // E 배치: 가변 자세(6) 다음 가변 카메라(정제 파라미터 수). CPU 와 같은 순서.
        let mut blocks: Vec<(usize, usize)> = Vec::new();
        let mut n_e = 0usize;
        let mut pose_eoff = vec![-1i32; np];
        for p in 0..np {
            if !inp.pose_const[p] {
                pose_eoff[p] = n_e as i32;
                blocks.push((n_e, 6));
                n_e += 6;
            }
        }
        let mut cam_eoff = vec![-1i32; nc];
        let mut maxc = 1usize;
        for c in 0..nc {
            let nv = inp.cam_var[c].len();
            if nv > MAXC {
                return Err(CudaBaError::Unsupported(format!("카메라 정제 파라미터 {nv} 개(최대 {MAXC})")));
            }
            if nv > 0 {
                cam_eoff[c] = n_e as i32;
                blocks.push((n_e, nv));
                n_e += nv;
                maxc = maxc.max(nv);
            }
            if inp.cams[c].params.len() > MAXC || inp.cams[c].params.len() != inp.cams[c].model.num_params() {
                return Err(CudaBaError::Unsupported("카메라 파라미터 수".into()));
            }
            if inp.cam_var[c].iter().any(|&pi| pi >= inp.cams[c].params.len()) {
                return Err(CudaBaError::Unsupported("카메라 정제 파라미터 번호".into()));
            }
        }
        if n_e > MAX_DENSE_N {
            return Err(CudaBaError::Unsupported(format!("축소 계통 {n_e} 차원(밀집 GPU 상한 {MAX_DENSE_N})")));
        }
        let cs = 2 * maxc;
        // 점
        let mut pt_v = vec![-1i32; npt];
        let mut v_pt: Vec<i32> = Vec::new();
        for j in 0..npt {
            if !inp.point_const[j] {
                pt_v[j] = v_pt.len() as i32;
                v_pt.push(j as i32);
            }
        }
        let nvp = v_pt.len();
        let mut pt_obs = vec![0i32; npt + 1];
        for k in 0..nobs {
            let j = inp.obs_pt[k] as usize;
            let i = inp.obs_img[k] as usize;
            if j >= npt || i >= nimg || (k > 0 && inp.obs_pt[k - 1] > inp.obs_pt[k]) {
                return Err(CudaBaError::Unsupported("관측이 점 순서로 정렬돼 있지 않거나 범위 밖".into()));
            }
            pt_obs[j + 1] += 1;
        }
        for j in 0..npt {
            pt_obs[j + 1] += pt_obs[j];
        }
        // 영상별 관측
        let mut img_obs_ptr = vec![0i32; nimg + 1];
        for &i in &inp.obs_img {
            img_obs_ptr[i as usize + 1] += 1;
        }
        for i in 0..nimg {
            img_obs_ptr[i + 1] += img_obs_ptr[i];
        }
        let mut fill: Vec<i32> = img_obs_ptr[..nimg].to_vec();
        let mut img_obs_idx = vec![0i32; nobs];
        for (k, &i) in inp.obs_img.iter().enumerate() {
            let f = &mut fill[i as usize];
            img_obs_idx[*f as usize] = k as i32;
            *f += 1;
        }
        // 점-E 블록 항목(점마다 서로 다른 블록)
        let block_dim = |off: i32| -> usize {
            // 오프셋으로 블록 크기 찾기(블록 수가 적어 선형 탐색 대신 이분 탐색).
            let i = blocks.binary_search_by_key(&(off as usize), |b| b.0).expect("E 블록");
            blocks[i].1
        };
        let mut pw_ptr = vec![0i32];
        let (mut pw_pt, mut pw_eoff, mut pw_dim, mut pw_voff) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut obs_qp = vec![-1i32; nobs];
        let mut obs_qc = vec![-1i32; nobs];
        let mut nvals = 0usize;
        for (vi, &j) in v_pt.iter().enumerate() {
            let j = j as usize;
            let start = pw_eoff.len();
            for k in pt_obs[j] as usize..pt_obs[j + 1] as usize {
                let i = inp.obs_img[k] as usize;
                let pe = pose_eoff[inp.img_pose[i] as usize];
                let ce = cam_eoff[inp.img_cam[i] as usize];
                for (off, slot) in [(pe, &mut obs_qp[k]), (ce, &mut obs_qc[k])] {
                    if off < 0 {
                        continue;
                    }
                    let q = match pw_eoff[start..].iter().position(|&o| o == off) {
                        Some(x) => start + x,
                        None => {
                            let d = block_dim(off);
                            pw_pt.push(vi as i32);
                            pw_eoff.push(off);
                            pw_dim.push(d as i32);
                            pw_voff.push(nvals as i32);
                            nvals += 3 * d;
                            pw_eoff.len() - 1
                        }
                    };
                    *slot = q as i32;
                }
            }
            pw_ptr.push(pw_eoff.len() as i32);
        }
        if nvals >= i32::MAX as usize {
            return Err(CudaBaError::Unsupported("점-E 블록 값이 너무 많음".into()));
        }
        let nent = pw_eoff.len();
        // 스칼라 행 → 블록, 상수 성분, 전처리 블록 배치
        let mut row_blk = vec![0i32; n_e];
        let (mut row_off, mut row_dim, mut row_poff) = (vec![0i32; n_e], vec![0i32; n_e], vec![0i32; n_e]);
        let (mut blk_off, mut blk_dim, mut blk_poff) = (Vec::new(), Vec::new(), Vec::new());
        let mut pinv_len = 0usize;
        for (b, &(o, d)) in blocks.iter().enumerate() {
            blk_off.push(o as i32);
            blk_dim.push(d as i32);
            blk_poff.push(pinv_len as i32);
            for r in 0..d {
                row_blk[o + r] = b as i32;
                row_off[o + r] = o as i32;
                row_dim[o + r] = d as i32;
                row_poff[o + r] = pinv_len as i32;
            }
            pinv_len += d * d;
        }
        let mut mask_idx = Vec::new();
        let mut pose_mask = vec![0i32; np];
        for p in 0..np {
            let mut bits = 0;
            for (q, &m) in inp.pose_mask[p].iter().enumerate() {
                if m {
                    bits |= 1 << q;
                    if pose_eoff[p] >= 0 {
                        mask_idx.push(pose_eoff[p] + q as i32);
                    }
                }
            }
            pose_mask[p] = bits;
        }
        let nmask = mask_idx.len();
        // 영상
        let mut img_hs = vec![0i32; nimg];
        let mut img_sr = vec![0.0f64; 9 * nimg];
        let mut img_st = vec![0.0f64; 3 * nimg];
        for (i, sens) in inp.img_sensor.iter().enumerate() {
            if let Some(r) = sens {
                img_hs[i] = 1;
                let m = r.rotation_matrix();
                for a in 0..3 {
                    for b in 0..3 {
                        img_sr[9 * i + 3 * a + b] = m[(a, b)];
                    }
                }
                img_st[3 * i..3 * i + 3].copy_from_slice(&[r.translation.x, r.translation.y, r.translation.z]);
            }
        }
        // 카메라
        let cam_model: Vec<i32> = inp.cams.iter().map(|c| c.model.id()).collect();
        let cam_np: Vec<i32> = inp.cams.iter().map(|c| c.model.num_params() as i32).collect();
        let cam_nv: Vec<i32> = inp.cam_var.iter().map(|v| v.len() as i32).collect();
        let mut cam_var = vec![0i32; MAXC * nc];
        for (c, v) in inp.cam_var.iter().enumerate() {
            for (q, &pi) in v.iter().enumerate() {
                cam_var[MAXC * c + q] = pi as i32;
            }
        }
        let obs_xy: Vec<f64> = inp.obs_xy.iter().flat_map(|p| [p.x, p.y]).collect();
        let pts: Vec<f64> = inp.points.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let (rot, trans) = pose_arrays(&inp.poses);
        let cams = cam_array(&inp.cams);
        let dense_cpu = n_e <= CPU_CHOLESKY_N;
        let nn = n_e * n_e;
        let d = Dev {
            obs_img: up_i32(&s, &to_i32(&inp.obs_img))?,
            obs_pt: up_i32(&s, &to_i32(&inp.obs_pt))?,
            obs_xy: up_f64(&s, &obs_xy)?,
            img_pose: up_i32(&s, &to_i32(&inp.img_pose))?,
            img_cam: up_i32(&s, &to_i32(&inp.img_cam))?,
            img_hs: up_i32(&s, &img_hs)?,
            img_sr: up_f64(&s, &img_sr)?,
            img_st: up_f64(&s, &img_st)?,
            pose_eoff: up_i32(&s, &pose_eoff)?,
            pose_mask: up_i32(&s, &pose_mask)?,
            cam_model: up_i32(&s, &cam_model)?,
            cam_np: up_i32(&s, &cam_np)?,
            cam_eoff: up_i32(&s, &cam_eoff)?,
            cam_nv: up_i32(&s, &cam_nv)?,
            cam_var: up_i32(&s, &cam_var)?,
            pt_v: up_i32(&s, &pt_v)?,
            v_pt: up_i32(&s, &v_pt)?,
            pt_obs: up_i32(&s, &pt_obs)?,
            img_obs_ptr: up_i32(&s, &img_obs_ptr)?,
            img_obs_idx: up_i32(&s, &img_obs_idx)?,
            obs_qp: up_i32(&s, &obs_qp)?,
            obs_qc: up_i32(&s, &obs_qc)?,
            pw_ptr: up_i32(&s, &pw_ptr)?,
            pw_pt: up_i32(&s, &pw_pt)?,
            pw_eoff: up_i32(&s, &pw_eoff)?,
            pw_dim: up_i32(&s, &pw_dim)?,
            pw_voff: up_i32(&s, &pw_voff)?,
            row_blk: up_i32(&s, &row_blk)?,
            mask_idx: up_i32(&s, &mask_idx)?,
            blk_off: up_i32(&s, &blk_off)?,
            blk_dim: up_i32(&s, &blk_dim)?,
            blk_poff: up_i32(&s, &blk_poff)?,
            row_off: up_i32(&s, &row_off)?,
            row_dim: up_i32(&s, &row_dim)?,
            row_poff: up_i32(&s, &row_poff)?,
            rot: up_f64(&s, &rot)?,
            trans: up_f64(&s, &trans)?,
            cams: up_f64(&s, &cams)?,
            pts: up_f64(&s, &pts)?,
            rot_c: up_f64(&s, &rot)?,
            trans_c: up_f64(&s, &trans)?,
            cams_c: up_f64(&s, &cams)?,
            pts_c: up_f64(&s, &pts)?,
            scale_e: up_f64(&s, &vec![1.0; n_e])?,
            scale_p: up_f64(&s, &vec![1.0; 3 * nvp])?,
            jp: zeros(&s, 12 * nobs)?,
            jx: zeros(&s, 6 * nobs)?,
            jc: zeros(&s, cs * nobs)?,
            res: zeros(&s, 2 * nobs)?,
            h: zeros(&s, nn)?,
            g_e: zeros(&s, n_e)?,
            v: zeros(&s, 9 * nvp)?,
            gp: zeros(&s, 3 * nvp)?,
            pw_vals: zeros(&s, nvals)?,
            vinv: zeros(&s, 9 * nvp)?,
            s: zeros(&s, nn)?,
            rhs: zeros(&s, n_e)?,
            delta_e: zeros(&s, n_e)?,
            delta_p: zeros(&s, 3 * nvp)?,
            diag: zeros(&s, pinv_len)?,
            pinv: zeros(&s, pinv_len)?,
            x: zeros(&s, n_e)?,
            r: zeros(&s, n_e)?,
            z: zeros(&s, n_e)?,
            p: zeros(&s, n_e)?,
            ap: zeros(&s, n_e)?,
            partial: zeros(&s, MAX_GRID)?,
            flag: s.alloc_zeros::<i32>(1)?,
        };
        let loss = match inp.loss {
            Loss::Trivial => (0, 0.0),
            Loss::SoftL1(a) => (1, a),
            Loss::Cauchy(a) => (2, a),
            Loss::Huber(a) => (3, a),
        };
        let cur = HostState { poses: inp.poses.clone(), cams: inp.cams.clone() };
        let cand = HostState { poses: inp.poses.clone(), cams: inp.cams.clone() };
        Ok(Self {
            s,
            k,
            d,
            npt,
            nobs,
            nimg,
            nvp,
            n_e,
            cs,
            nent,
            nmask,
            blocks,
            pinv_len,
            pose_eoff,
            cam_eoff,
            cam_var: inp.cam_var.clone(),
            loss,
            max_lin_iter: inp.max_linear_solver_iterations,
            dense_cpu,
            cur,
            cand,
            scale_e: vec![1.0; n_e],
            g_e: vec![0.0; n_e],
            gp_max: 0.0,
            delta_e: vec![0.0; n_e],
            err: None,
        })
    }

    /// 관측 평가. `cand` 면 후보 상태, `want_jac` 면 잔차·야코비안 저장. 반환: ½Σρ.
    fn eval(&mut self, cand: bool, want_jac: bool) -> Result<f64, CudaBaError> {
        if self.nobs == 0 {
            return Ok(0.0);
        }
        let (s, k, d) = (&self.s, self.k, &mut self.d);
        let g = grid_for(self.nobs);
        let (n, wj, cs) = (self.nobs as i32, want_jac as i32, self.cs as i32);
        let (lk, la) = self.loss;
        let (rot, trans, cams, pts) = if cand { (&d.rot_c, &d.trans_c, &d.cams_c, &d.pts_c) } else { (&d.rot, &d.trans, &d.cams, &d.pts) };
        let mut l = s.launch_builder(&k.eval);
        l.arg(&n).arg(&wj).arg(&cs);
        l.arg(&d.obs_img).arg(&d.obs_pt).arg(&d.obs_xy);
        l.arg(&d.img_pose).arg(&d.img_cam).arg(&d.img_hs).arg(&d.img_sr).arg(&d.img_st);
        l.arg(rot).arg(trans);
        l.arg(&d.pose_eoff).arg(&d.pose_mask);
        l.arg(&d.cam_model).arg(&d.cam_np).arg(cams).arg(&d.cam_eoff).arg(&d.cam_nv).arg(&d.cam_var);
        l.arg(pts).arg(&d.pt_v).arg(&d.scale_e).arg(&d.scale_p);
        l.arg(&lk).arg(&la);
        l.arg(&mut d.jp).arg(&mut d.jx).arg(&mut d.jc).arg(&mut d.res).arg(&mut d.partial);
        // SAFETY: ba_eval 의 인자 순서·형과 같고, 배열 크기는 new 에서 관측·영상·자세·카메라·점 수로 잡았다.
        unsafe { l.launch(launch_cfg(g)) }?;
        Ok(0.5 * sum_partial(s, &d.partial, g)?)
    }

    /// H·g_E (영상별), V·g_p·W (점별), 점 기울기 최대.
    fn build_normal(&mut self) -> Result<(), CudaBaError> {
        let (s, k, d) = (&self.s, self.k, &mut self.d);
        let (n_e, cs) = (self.n_e as i32, self.cs as i32);
        s.memset_zeros(&mut d.h)?;
        s.memset_zeros(&mut d.g_e)?;
        if self.n_e > 0 && self.nimg > 0 {
            let mut l = s.launch_builder(&k.img_normal);
            l.arg(&n_e).arg(&cs).arg(&d.img_obs_ptr).arg(&d.img_obs_idx).arg(&d.img_pose).arg(&d.img_cam);
            l.arg(&d.pose_eoff).arg(&d.cam_eoff).arg(&d.cam_nv).arg(&d.jp).arg(&d.jc).arg(&d.res);
            l.arg(&mut d.h).arg(&mut d.g_e);
            // SAFETY: 블록 하나 = 영상 하나(격자 = 영상 수), 인자는 ba_img_normal 과 같다.
            unsafe { l.launch(LaunchConfig { grid_dim: (self.nimg as u32, 1, 1), block_dim: (BS, 1, 1), shared_mem_bytes: 0 }) }?;
        }
        self.gp_max = 0.0;
        if self.nvp > 0 {
            let nvp = self.nvp as i32;
            let mut l = s.launch_builder(&k.point_normal);
            l.arg(&nvp).arg(&cs).arg(&d.v_pt).arg(&d.pt_obs).arg(&d.obs_qp).arg(&d.obs_qc);
            l.arg(&d.pw_ptr).arg(&d.pw_dim).arg(&d.pw_voff).arg(&d.jp).arg(&d.jx).arg(&d.jc).arg(&d.res);
            l.arg(&mut d.v).arg(&mut d.gp).arg(&mut d.pw_vals);
            // SAFETY: ba_point_normal 과 같은 인자, 가변 점 수만큼.
            unsafe { l.launch(launch_cfg(grid_for(self.nvp))) }?;
        }
        if self.n_e > 0 {
            s.memcpy_dtoh(&d.g_e.slice(0..self.n_e), &mut self.g_e)?;
            s.synchronize()?;
        }
        if self.nvp > 0 {
            let nvp = self.nvp as i32;
            let g = grid_for(3 * self.nvp);
            let mut l = s.launch_builder(&k.gp_max);
            l.arg(&nvp).arg(&d.gp).arg(&d.scale_p).arg(&mut d.partial);
            // SAFETY: ba_gp_max(nvp, gp[3·nvp], scale_p[3·nvp], partial).
            unsafe { l.launch(launch_cfg(g)) }?;
            self.gp_max = max_partial(s, &d.partial, g)?;
        }
        Ok(())
    }

    /// 첫 선형화 뒤 야코비 열 배율을 정한다.
    fn set_scales(&mut self) -> Result<(), CudaBaError> {
        let (s, k, d) = (&self.s, self.k, &mut self.d);
        let total = self.n_e + self.nvp;
        if total > 0 {
            let (n_e, nvp) = (self.n_e as i32, self.nvp as i32);
            let mut l = s.launch_builder(&k.set_scales);
            l.arg(&n_e).arg(&nvp).arg(&d.h).arg(&d.v).arg(&mut d.scale_e).arg(&mut d.scale_p);
            // SAFETY: ba_set_scales(n_e, nvp, H[n_e²], V[9·nvp], scale_e[n_e], scale_p[3·nvp]).
            unsafe { l.launch(launch_cfg(grid_for(total))) }?;
        }
        if self.n_e > 0 {
            s.memcpy_dtoh(&d.scale_e.slice(0..self.n_e), &mut self.scale_e)?;
        }
        s.synchronize()?;
        Ok(())
    }

    fn linearize_impl(&mut self, first: bool) -> Result<Option<f64>, CudaBaError> {
        let cost = self.eval(false, true)?;
        if !cost.is_finite() {
            return Ok(None);
        }
        self.build_normal()?;
        if first {
            self.set_scales()?;
            let c2 = self.eval(false, true)?;
            if !c2.is_finite() {
                return Ok(None);
            }
            self.build_normal()?;
        }
        Ok(Some(cost))
    }

    /// 축소 계통 S·우변 조립. 점 블록이 양정치가 아니면 Ok(false).
    fn assemble(&mut self, radius: f64) -> Result<bool, CudaBaError> {
        let (s, k, d) = (&self.s, self.k, &mut self.d);
        s.memset_zeros(&mut d.flag)?;
        if self.nvp > 0 {
            let nvp = self.nvp as i32;
            let mut l = s.launch_builder(&k.vinv);
            l.arg(&nvp).arg(&radius).arg(&d.v).arg(&mut d.vinv).arg(&mut d.flag);
            // SAFETY: ba_vinv(nvp, radius, V[9·nvp], vinv[9·nvp], fail[1]).
            unsafe { l.launch(launch_cfg(grid_for(self.nvp))) }?;
        }
        if read_flag(s, &d.flag)? {
            return Ok(false);
        }
        if self.n_e == 0 {
            return Ok(true);
        }
        let n_e = self.n_e as i32;
        {
            let mut l = s.launch_builder(&k.s_init);
            l.arg(&n_e).arg(&radius).arg(&d.h).arg(&d.g_e).arg(&mut d.s).arg(&mut d.rhs);
            // SAFETY: ba_s_init(n_e, radius, H[n_e²], g_e[n_e], S[n_e²], rhs[n_e]).
            unsafe { l.launch(launch_cfg(grid_for(self.n_e * self.n_e))) }?;
        }
        if self.nent > 0 {
            let nent = self.nent as i32;
            let mut l = s.launch_builder(&k.schur);
            l.arg(&nent).arg(&n_e).arg(&d.pw_pt).arg(&d.pw_ptr).arg(&d.pw_eoff).arg(&d.pw_dim).arg(&d.pw_voff);
            l.arg(&d.pw_vals).arg(&d.vinv).arg(&d.gp).arg(&mut d.s).arg(&mut d.rhs);
            // SAFETY: ba_schur 와 같은 인자, 항목 수만큼.
            unsafe { l.launch(launch_cfg(grid_for(self.nent))) }?;
        }
        {
            let mut l = s.launch_builder(&k.symmetrize);
            l.arg(&n_e).arg(&d.row_blk).arg(&mut d.s);
            // SAFETY: ba_symmetrize(n_e, row_blk[n_e], S[n_e²]).
            unsafe { l.launch(launch_cfg(grid_for(self.n_e * self.n_e))) }?;
        }
        if self.nmask > 0 {
            let nm = self.nmask as i32;
            let mut l = s.launch_builder(&k.fix_mask);
            l.arg(&nm).arg(&n_e).arg(&d.mask_idx).arg(&mut d.s).arg(&mut d.rhs);
            // SAFETY: ba_fix_mask(nm, n_e, idx[nm], S, rhs).
            unsafe { l.launch(launch_cfg(grid_for(self.nmask))) }?;
        }
        Ok(true)
    }

    /// S δE = rhs. 작으면 CPU 촐레스키, 크면 GPU PCG. 해는 d.delta_e 와 self.delta_e. 실패면 Ok(false).
    fn solve_reduced(&mut self) -> Result<bool, CudaBaError> {
        let n = self.n_e;
        if n == 0 {
            return Ok(true);
        }
        if self.dense_cpu {
            let (s, d) = (&self.s, &mut self.d);
            let sv = s.clone_dtoh(&d.s)?;
            let mut rhs = vec![0.0; n];
            s.memcpy_dtoh(&d.rhs.slice(0..n), &mut rhs)?;
            s.synchronize()?;
            let m = DMatrix::from_row_slice(n, n, &sv[..n * n]);
            let Some(ch) = m.cholesky() else { return Ok(false) };
            let x = ch.solve(&DVector::from_column_slice(&rhs));
            if !x.iter().all(|v| v.is_finite()) {
                return Ok(false);
            }
            self.delta_e.copy_from_slice(x.as_slice());
            s.memcpy_htod(&self.delta_e, &mut d.delta_e)?;
            return Ok(true);
        }
        if !self.pcg()? {
            return Ok(false);
        }
        let (s, d) = (&self.s, &mut self.d);
        s.memcpy_dtoh(&d.x.slice(0..n), &mut self.delta_e)?;
        s.synchronize()?;
        if !self.delta_e.iter().all(|v| v.is_finite()) {
            return Ok(false);
        }
        s.memcpy_dtod(&d.x, &mut d.delta_e)?;
        Ok(true)
    }

    fn dot(&mut self, which: (Vec2Ref, Vec2Ref)) -> Result<f64, CudaBaError> {
        let (s, k, d) = (&self.s, self.k, &mut self.d);
        let n = self.n_e as i32;
        let g = grid_for(self.n_e);
        // 필드별로 빌려야 partial 을 가변으로 함께 넘길 수 있다.
        let a = match which.0 {
            Vec2Ref::Rhs => &d.rhs,
            Vec2Ref::R => &d.r,
            Vec2Ref::Z => &d.z,
            Vec2Ref::P => &d.p,
            Vec2Ref::Ap => &d.ap,
        };
        let b = match which.1 {
            Vec2Ref::Rhs => &d.rhs,
            Vec2Ref::R => &d.r,
            Vec2Ref::Z => &d.z,
            Vec2Ref::P => &d.p,
            Vec2Ref::Ap => &d.ap,
        };
        let mut l = s.launch_builder(&k.dot);
        l.arg(&n).arg(a).arg(b).arg(&mut d.partial);
        // SAFETY: ba_dot(n, a[n], b[n], partial).
        unsafe { l.launch(launch_cfg(g)) }?;
        Ok(sum_partial(s, &d.partial, g)?)
    }

    fn precond(&mut self) -> Result<(), CudaBaError> {
        let (s, k, d) = (&self.s, self.k, &mut self.d);
        let n = self.n_e as i32;
        let mut l = s.launch_builder(&k.precond);
        l.arg(&n).arg(&d.row_off).arg(&d.row_dim).arg(&d.row_poff).arg(&d.pinv).arg(&d.r).arg(&mut d.z);
        // SAFETY: ba_precond(n, row_off, row_dim, row_poff, pinv, r, z).
        unsafe { l.launch(launch_cfg(grid_for(self.n_e))) }?;
        Ok(())
    }

    /// 블록 야코비 PCG(CPU 반복 풀이기와 같은 종료 규칙). 해는 d.x. 첫 반복에서 곡률이 양수가 아니면 Ok(false).
    fn pcg(&mut self) -> Result<bool, CudaBaError> {
        let n = self.n_e;
        // 대각 블록 역행렬(촐레스키, 실패면 단위행렬: CPU 와 같다).
        {
            let (s, k, d) = (&self.s, self.k, &mut self.d);
            let nb = self.blocks.len() as i32;
            let n_e = n as i32;
            let mut l = s.launch_builder(&k.diag_blocks);
            l.arg(&nb).arg(&n_e).arg(&d.blk_off).arg(&d.blk_dim).arg(&d.blk_poff).arg(&d.s).arg(&mut d.diag);
            // SAFETY: ba_diag_blocks(nb, n_e, blk_off, blk_dim, blk_poff, S, out[Σd²]).
            unsafe { l.launch(launch_cfg(grid_for(self.blocks.len()))) }?;
            let mut diag = vec![0.0; self.pinv_len];
            s.memcpy_dtoh(&d.diag.slice(0..self.pinv_len), &mut diag)?;
            s.synchronize()?;
            let mut pinv = vec![0.0; self.pinv_len];
            let mut po = 0;
            for &(_, bd) in &self.blocks {
                let m = DMatrix::from_row_slice(bd, bd, &diag[po..po + bd * bd]);
                let inv = m.cholesky().map(|c| c.inverse()).unwrap_or_else(|| DMatrix::identity(bd, bd));
                for r in 0..bd {
                    for c in 0..bd {
                        pinv[po + r * bd + c] = inv[(r, c)];
                    }
                }
                po += bd * bd;
            }
            s.memcpy_htod(&pinv, &mut d.pinv)?;
            s.memset_zeros(&mut d.x)?;
        }
        let bnorm = self.dot((Vec2Ref::Rhs, Vec2Ref::Rhs))?.sqrt();
        if bnorm == 0.0 {
            return Ok(true);
        }
        {
            let (s, d) = (&self.s, &mut self.d);
            s.memcpy_dtod(&d.rhs, &mut d.r)?;
        }
        self.precond()?;
        {
            let (s, d) = (&self.s, &mut self.d);
            s.memcpy_dtod(&d.z, &mut d.p)?;
        }
        let mut rz = self.dot((Vec2Ref::R, Vec2Ref::Z))?;
        let mut q0 = 0.0;
        let ni = n as i32;
        for it in 1..=self.max_lin_iter.max(1) {
            {
                let (s, k, d) = (&self.s, self.k, &mut self.d);
                let g = (n * 32).div_ceil(BS as usize).clamp(1, 4096);
                let mut l = s.launch_builder(&k.matvec);
                l.arg(&ni).arg(&d.s).arg(&d.p).arg(&mut d.ap);
                // SAFETY: ba_matvec(n, S[n²], x[n], y[n]), 워프 하나가 행 하나.
                unsafe { l.launch(launch_cfg(g)) }?;
            }
            let pap = self.dot((Vec2Ref::P, Vec2Ref::Ap))?;
            if pap.is_nan() || pap <= 0.0 || !rz.is_finite() {
                if it == 1 {
                    return Ok(false);
                }
                break;
            }
            let alpha = rz / pap;
            {
                let (s, k, d) = (&self.s, self.k, &mut self.d);
                let mut l = s.launch_builder(&k.axpy2);
                l.arg(&ni).arg(&alpha).arg(&d.p).arg(&d.ap).arg(&mut d.x).arg(&mut d.r);
                // SAFETY: ba_axpy2(n, α, p, ap, x, r).
                unsafe { l.launch(launch_cfg(grid_for(n))) }?;
            }
            let rnorm = self.dot((Vec2Ref::R, Vec2Ref::R))?.sqrt();
            if rnorm <= 1e-12 * bnorm {
                break;
            }
            let xbr = {
                let (s, k, d) = (&self.s, self.k, &mut self.d);
                let g = grid_for(n);
                let mut l = s.launch_builder(&k.dot_xbr);
                l.arg(&ni).arg(&d.x).arg(&d.rhs).arg(&d.r).arg(&mut d.partial);
                // SAFETY: ba_dot_xbr(n, x, b, r, partial).
                unsafe { l.launch(launch_cfg(g)) }?;
                sum_partial(s, &d.partial, g)?
            };
            let q1 = -0.5 * xbr;
            if q1 < 0.0 && (it as f64) * (q1 - q0) / q1 < PCG_ETA {
                break;
            }
            q0 = q1;
            self.precond()?;
            let rz_new = self.dot((Vec2Ref::R, Vec2Ref::Z))?;
            let beta = rz_new / rz;
            rz = rz_new;
            {
                let (s, k, d) = (&self.s, self.k, &mut self.d);
                let mut l = s.launch_builder(&k.xpby);
                l.arg(&ni).arg(&beta).arg(&d.z).arg(&mut d.p);
                // SAFETY: ba_xpby(n, β, z, p).
                unsafe { l.launch(launch_cfg(grid_for(n))) }?;
            }
        }
        Ok(true)
    }

    /// 점 역대입·후보 점·모형 비용 변화. 반환: (모형 비용 변화, ‖점 단계‖²), 비유한이면 None.
    fn back_substitute(&mut self) -> Result<Option<(f64, f64)>, CudaBaError> {
        let (s, k, d) = (&self.s, self.k, &mut self.d);
        let cs = self.cs as i32;
        let mut step_pts = 0.0;
        if self.nvp > 0 {
            s.memset_zeros(&mut d.flag)?;
            let nvp = self.nvp as i32;
            let g = grid_for(self.nvp);
            let mut l = s.launch_builder(&k.backsub);
            l.arg(&nvp).arg(&cs).arg(&d.v_pt).arg(&d.pt_obs).arg(&d.obs_img).arg(&d.img_pose).arg(&d.img_cam);
            l.arg(&d.pose_eoff).arg(&d.cam_eoff).arg(&d.cam_nv).arg(&d.jp).arg(&d.jx).arg(&d.jc);
            l.arg(&d.gp).arg(&d.vinv).arg(&d.delta_e).arg(&d.scale_p).arg(&d.pts).arg(&mut d.pts_c);
            l.arg(&mut d.delta_p).arg(&mut d.partial).arg(&mut d.flag);
            // SAFETY: ba_backsub 와 같은 인자, 가변 점 수만큼.
            unsafe { l.launch(launch_cfg(g)) }?;
            step_pts = sum_partial(s, &d.partial, g)?;
            if read_flag(s, &d.flag)? {
                return Ok(None);
            }
        }
        let mut mc = 0.0;
        if self.nobs > 0 {
            let n = self.nobs as i32;
            let g = grid_for(self.nobs);
            let mut l = s.launch_builder(&k.model_cost);
            l.arg(&n).arg(&cs).arg(&d.obs_img).arg(&d.obs_pt).arg(&d.img_pose).arg(&d.img_cam);
            l.arg(&d.pose_eoff).arg(&d.cam_eoff).arg(&d.cam_nv).arg(&d.pt_v).arg(&d.jp).arg(&d.jx).arg(&d.jc);
            l.arg(&d.res).arg(&d.delta_e).arg(&d.delta_p).arg(&mut d.partial);
            // SAFETY: ba_model_cost 와 같은 인자, 관측 수만큼.
            unsafe { l.launch(launch_cfg(g)) }?;
            mc = sum_partial(s, &d.partial, g)?;
        }
        Ok(Some((mc, step_pts)))
    }

    /// ‖x‖ (자세 쿼터니언·이동, 카메라 파라미터, 점).
    fn x_norm(&mut self) -> Result<f64, CudaBaError> {
        let mut sq = 0.0;
        for p in &self.cur.poses {
            let q = p.rotation;
            sq += q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z + p.translation.norm_squared();
        }
        for c in &self.cur.cams {
            sq += c.params.iter().map(|x| x * x).sum::<f64>();
        }
        if self.npt > 0 {
            let (s, k, d) = (&self.s, self.k, &mut self.d);
            let n = (3 * self.npt) as i32;
            let g = grid_for(3 * self.npt);
            let mut l = s.launch_builder(&k.dot);
            l.arg(&n).arg(&d.pts).arg(&d.pts).arg(&mut d.partial);
            // SAFETY: ba_dot(3·npt, pts, pts, partial).
            unsafe { l.launch(launch_cfg(g)) }?;
            sq += sum_partial(s, &d.partial, g)?;
        }
        Ok(sq.sqrt())
    }

    /// 자세·카메라 후보(호스트)를 만들어 올린다. 반환: ‖후보 − 현재‖² 의 자세·카메라 부분.
    fn make_candidate(&mut self) -> Result<f64, CudaBaError> {
        let mut sq = 0.0;
        for p in 0..self.cur.poses.len() {
            let pe = self.pose_eoff[p];
            let cur = self.cur.poses[p];
            if pe < 0 {
                self.cand.poses[p] = cur;
                continue;
            }
            let o = pe as usize;
            let dl: [f64; 6] = std::array::from_fn(|i| self.delta_e[o + i] * self.scale_e[o + i]);
            let q = quat_plus(&cur.rotation, &dl[0..3]).normalized();
            let t = cur.translation + Vec3::new(dl[3], dl[4], dl[5]);
            let (a, b) = (cur.rotation, q);
            sq += (a.w - b.w).powi(2) + (a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2);
            sq += (t - cur.translation).norm_squared();
            self.cand.poses[p] = Rigid3::new(q, t);
        }
        for c in 0..self.cur.cams.len() {
            self.cand.cams[c].params.clone_from(&self.cur.cams[c].params);
            let ce = self.cam_eoff[c];
            if ce < 0 {
                continue;
            }
            let o = ce as usize;
            for (q, &pi) in self.cam_var[c].iter().enumerate() {
                let dl = self.delta_e[o + q] * self.scale_e[o + q];
                self.cand.cams[c].params[pi] += dl;
                sq += dl * dl;
            }
        }
        let (rot, tr) = pose_arrays(&self.cand.poses);
        let cams = cam_array(&self.cand.cams);
        let (s, d) = (&self.s, &mut self.d);
        if !rot.is_empty() {
            s.memcpy_htod(&rot, &mut d.rot_c)?;
            s.memcpy_htod(&tr, &mut d.trans_c)?;
        }
        if !cams.is_empty() {
            s.memcpy_htod(&cams, &mut d.cams_c)?;
        }
        Ok(sq)
    }

    fn compute_step_impl(&mut self, radius: f64) -> Result<Option<StepInfo>, CudaBaError> {
        if !self.assemble(radius)? {
            return Ok(None);
        }
        if !self.solve_reduced()? {
            return Ok(None);
        }
        let Some((model_cost_change, step_pts)) = self.back_substitute()? else {
            return Ok(None);
        };
        let x_norm = self.x_norm()?;
        let sq = self.make_candidate()?;
        Ok(Some(StepInfo { model_cost_change, step_norm: (sq + step_pts).sqrt(), x_norm }))
    }

    fn download_points(&self) -> Result<Vec<Vec3>, CudaBaError> {
        if self.npt == 0 {
            return Ok(Vec::new());
        }
        let mut h = vec![0.0; 3 * self.npt];
        self.s.memcpy_dtoh(&self.d.pts.slice(0..3 * self.npt), &mut h)?;
        self.s.synchronize()?;
        Ok(h.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect())
    }

    fn record<T>(&mut self, r: Result<Option<T>, CudaBaError>) -> Option<T> {
        match r {
            Ok(v) => v,
            Err(e) => {
                self.err = Some(e);
                None
            }
        }
    }
}

/// PCG 내적에 쓰는 장치 벡터 이름.
#[derive(Clone, Copy)]
enum Vec2Ref {
    Rhs,
    R,
    Z,
    P,
    Ap,
}

impl TrProblem for GpuProblem<'_> {
    fn linearize(&mut self, first: bool) -> Option<f64> {
        if self.err.is_some() {
            return None;
        }
        let r = self.linearize_impl(first);
        self.record(r)
    }

    fn grad_inf_norm(&self) -> f64 {
        let mut m: f64 = 0.0;
        for (p, &pe) in self.pose_eoff.iter().enumerate() {
            if pe < 0 {
                continue;
            }
            let o = pe as usize;
            let g: [f64; 6] = std::array::from_fn(|i| self.g_e[o + i] / self.scale_e[o + i]);
            let q = self.cur.poses[p].rotation;
            let qn = quat_plus(&q, &[-g[0], -g[1], -g[2]]);
            m = m.max((q.w - qn.w).abs()).max((q.x - qn.x).abs()).max((q.y - qn.y).abs()).max((q.z - qn.z).abs());
            for &gi in &g[3..6] {
                m = m.max(gi.abs());
            }
        }
        for &ce in &self.cam_eoff {
            if ce < 0 {
                continue;
            }
            let o = ce as usize;
            let dim = self.blocks.iter().find(|b| b.0 == o).map_or(0, |b| b.1);
            for q in 0..dim {
                m = m.max((self.g_e[o + q] / self.scale_e[o + q]).abs());
            }
        }
        m.max(self.gp_max)
    }

    fn compute_step(&mut self, radius: f64) -> Option<StepInfo> {
        if self.err.is_some() {
            return None;
        }
        let r = self.compute_step_impl(radius);
        self.record(r)
    }

    fn candidate_cost(&mut self) -> Option<f64> {
        if self.err.is_some() {
            return None;
        }
        let r = self.eval(true, false).map(|c| c.is_finite().then_some(c));
        self.record(r)
    }

    fn accept_candidate(&mut self) {
        std::mem::swap(&mut self.cur, &mut self.cand);
        let d = &mut self.d;
        std::mem::swap(&mut d.rot, &mut d.rot_c);
        std::mem::swap(&mut d.trans, &mut d.trans_c);
        std::mem::swap(&mut d.cams, &mut d.cams_c);
        std::mem::swap(&mut d.pts, &mut d.pts_c);
    }
}

/// 계측용 크기: 관측 수.
fn input_size(input: &ProblemInput) -> usize {
    input.obs_img.len()
}
