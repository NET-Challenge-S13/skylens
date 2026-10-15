/*
 * sparse.rs
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

//! 희소 단계의 CUDA 백엔드([`CudaSparse`]).
//!
//! - 두 뷰 기하 검증(`TwoViewBackend`): 짝 묶음의 E/F/H LO-RANSAC 가설을 라운드마다 커널 한 번으로 채점.
//! - 절대 자세(`AbsPoseBackend`): P3P 가설을 묶음으로 채점.
//! - 삼각측량(`TriangulationBackend`): 후보 트랙마다 스레드 하나가 RANSAC 삼각측량(2점 DLT, 다중 뷰 LO,
//!   깊이·각도 검사)을 하고, 관측 재투영 오차·깊이를 한꺼번에 계산.
//!
//! 가설 생성(최소 해법)·LO·모델 갱신은 CPU 가 그대로 한다. 채점 커널은 곱셈-덧셈 융합을 끄고 CPU 와
//! 같은 순서로 계산하므로 같은 시드에서 CPU 경로와 같은 인라이어를 낸다. 삼각측량 커널은 특이값 분해
//! 방식이 달라 수치 잡음 수준의 차이가 날 수 있다. 장치 오류는 경고 후 CPU 로 계산한다.

use crate::device::{CudaDevice, GpuError};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use cumulus3d_core::ransac::{n_choose_k, static_max_num_trials, RansacParams, Support};
use cumulus3d_core::{Camera, CameraModelKind, FeatureStore, Rigid3, TwoViewGeometry, Vec2, Vec3};
use cumulus3d_matching::{verify_pairs_batched, BatchScorer, CpuTwoView, ResidualSet, ScoreRequest, TwoViewBackend, TwoViewOptions, VerifyJob};
use cumulus3d_sfm::absolute_pose::{solve_abs_pose, AbsolutePoseOptions, AbsolutePoseResult};
use cumulus3d_sfm::triangulation::estimate_triangulation;
use cumulus3d_sfm::{solve_abs_pose_scored, AbsPoseBackend, CpuTriangulation, ObsCheck, ObsQuery, PoseScorer, TriProblem, TriangulationBackend};
use rayon::prelude::*;
use std::sync::{Arc, Mutex, MutexGuard};

pub(crate) const SRC: &str = include_str!("kernels/sparse.cu");
/// 채점 커널 블록 크기(커널의 BLOCK 과 같아야 함).
const BLOCK: u32 = 128;
/// GPU 삼각측량 문제 하나의 최대 관측 수(커널의 MAX_TRI_OBS, 인라이어 비트 마스크 64비트).
const MAX_TRI_OBS: usize = 64;
/// 관측 하나의 double 수(커널의 OBS_STRIDE).
const OBS_STRIDE: usize = 18;
/// 이보다 적은 삼각측량 문제는 CPU 가 더 빠르다.
const TRI_MIN_GPU: usize = 64;
/// 이보다 적은 관측 검사는 CPU 가 더 빠르다.
const OBS_MIN_GPU: usize = 4096;

/// 희소 단계 CUDA 백엔드. 두 뷰 기하 검증·절대 자세·삼각측량 백엔드 트레이트를 모두 구현한다.
pub struct CudaSparse {
    dev: Arc<CudaDevice>,
    f_two_view: CudaFunction,
    f_pose: CudaFunction,
    f_tri: CudaFunction,
    f_obs: CudaFunction,
    /// 한 번에 하나만 장치를 쓴다(스트림 공유).
    lock: Mutex<()>,
}

/// 카메라를 커널 형식으로: [모델 번호, fx, fy, cx, cy, k1, k2, p1, p2].
fn pack_camera(cam: &Camera) -> [f64; 9] {
    let p = &cam.params;
    let (k1, k2, p1, p2) = match cam.model {
        CameraModelKind::SingleFocalPinhole | CameraModelKind::Pinhole => (0.0, 0.0, 0.0, 0.0),
        CameraModelKind::SingleFocalRadial => (p[3], 0.0, 0.0, 0.0),
        CameraModelKind::Radial => (p[3], p[4], 0.0, 0.0),
        CameraModelKind::OpenCv => (p[4], p[5], p[6], p[7]),
    };
    [
        cam.model.id() as f64,
        cam.focal_length_x(),
        cam.focal_length_y(),
        cam.principal_point_x(),
        cam.principal_point_y(),
        k1,
        k2,
        p1,
        p2,
    ]
}

/// 자세를 커널 형식으로: [R 행 우선 9, t 3]. R 은 CPU 와 같은 `rotation_matrix()`.
fn pack_pose(pose: &Rigid3, out: &mut Vec<f64>) {
    let r = pose.rotation_matrix();
    for row in 0..3 {
        for col in 0..3 {
            out.push(r[(row, col)]);
        }
    }
    out.extend_from_slice(&[pose.translation.x, pose.translation.y, pose.translation.z]);
}

fn supports(cnt: &[u32], sum: &[f64]) -> Vec<Support> {
    cnt.iter().zip(sum).map(|(&n, &s)| Support { num_inliers: n as usize, residual_sum: s }).collect()
}

impl CudaSparse {
    /// 장치 `dev` 에 희소 단계 커널을 컴파일해 만든다(fmad 끔).
    pub fn new(dev: Arc<CudaDevice>) -> Result<Self, GpuError> {
        let module = dev.compile(SRC, false, &[])?;
        Ok(Self {
            f_two_view: module.load_function("score_two_view")?,
            f_pose: module.load_function("score_pose")?,
            f_tri: module.load_function("tri_ransac")?,
            f_obs: module.load_function("check_obs")?,
            dev,
            lock: Mutex::new(()),
        })
    }

    /// 장치 0 으로 만든다.
    pub fn try_default() -> Result<Self, GpuError> {
        Self::new(CudaDevice::new(0)?)
    }

    fn guard(&self) -> MutexGuard<'_, ()> {
        self.lock.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// GPU RANSAC 삼각측량(오류를 그대로 돌려줌; 트레이트 메서드는 오류 시 경고 후 CPU 로 계산).
    /// 관측이 2개 미만이거나 64개를 넘는 문제는 CPU 로 푼다.
    pub fn estimate_batch_gpu(&self, problems: &[TriProblem<'_>]) -> Result<Vec<Option<(Vec3, Vec<bool>)>>, GpuError> {
        let mut out: Vec<Option<(Vec3, Vec<bool>)>> = (0..problems.len()).map(|_| None).collect();
        let (gpu, cpu): (Vec<usize>, Vec<usize>) =
            (0..problems.len()).partition(|&i| (2..=MAX_TRI_OBS).contains(&problems[i].obs.len()));
        let cpu_res: Vec<(usize, Option<(Vec3, Vec<bool>)>)> =
            cpu.par_iter().map(|&i| (i, estimate_triangulation(problems[i].obs, &problems[i].params))).collect();
        for (i, r) in cpu_res {
            out[i] = r;
        }
        if gpu.is_empty() {
            return Ok(out);
        }
        let mut obs_host: Vec<f64> = Vec::new();
        let mut desc_i: Vec<i64> = Vec::with_capacity(4 * gpu.len());
        let mut desc_d: Vec<f64> = Vec::with_capacity(3 * gpu.len());
        let mut start = 0usize;
        for &i in &gpu {
            let p = &problems[i];
            let n = p.obs.len();
            // estimate_triangulation 과 같은 RANSAC 설정.
            let num_pairs = n_choose_k(n, 2);
            let ro = RansacParams {
                max_error: p.params.max_angle_error_deg.to_radians(),
                min_inlier_ratio: p.params.min_inlier_ratio,
                confidence: p.params.confidence,
                dyn_trials_factor: 3.0,
                min_trials: if n <= p.params.exhaustive_max_num_obs { num_pairs } else { 0 },
                max_trials: p.params.max_trials.min(num_pairs),
                random_seed: Some(0),
            };
            let max_trials = static_max_num_trials(&ro, 2).min(num_pairs);
            desc_i.extend_from_slice(&[start as i64, n as i64, ro.min_trials as i64, max_trials as i64]);
            desc_d.extend_from_slice(&[ro.max_error * ro.max_error, p.params.min_tri_angle_deg.to_radians(), ro.confidence]);
            for o in p.obs {
                for r in 0..3 {
                    for c in 0..4 {
                        obs_host.push(o.proj[(r, c)]);
                    }
                }
                obs_host.extend_from_slice(&[o.center.x, o.center.y, o.center.z, o.ray.x, o.ray.y, o.ray.z]);
            }
            start += n;
        }
        debug_assert_eq!(obs_host.len(), start * OBS_STRIDE);
        let np = gpu.len();
        let (x, mask, ok) = {
            let _g = self.guard();
            let s = &self.dev.stream;
            let obs = s.clone_htod(&obs_host)?;
            let di = s.clone_htod(&desc_i)?;
            let dd = s.clone_htod(&desc_d)?;
            let mut ox = s.alloc_zeros::<f64>(3 * np)?;
            let mut om = s.alloc_zeros::<u64>(np)?;
            let mut ook = s.alloc_zeros::<i32>(np)?;
            let np_i = np as i32;
            let cfg = LaunchConfig { grid_dim: ((np as u32).div_ceil(64), 1, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 };
            let mut l = s.launch_builder(&self.f_tri);
            l.arg(&obs).arg(&di).arg(&dd).arg(&np_i).arg(&mut ox).arg(&mut om).arg(&mut ook);
            // SAFETY: tri_ransac(obs[18·Σn], desc_i[4·np], desc_d[3·np], np, out_x[3·np], out_mask[np], out_ok[np]) 와 같은 인자·크기.
            unsafe { l.launch(cfg) }?;
            let x = s.clone_dtoh(&ox)?;
            let mask = s.clone_dtoh(&om)?;
            let ok = s.clone_dtoh(&ook)?;
            s.synchronize()?;
            (x, mask, ok)
        };
        for (k, &i) in gpu.iter().enumerate() {
            if ok[k] == 0 {
                continue;
            }
            let n = problems[i].obs.len();
            let m: Vec<bool> = (0..n).map(|b| (mask[k] >> b) & 1 == 1).collect();
            out[i] = Some((Vec3::new(x[3 * k], x[3 * k + 1], x[3 * k + 2]), m));
        }
        Ok(out)
    }

    /// GPU 관측 검사(오류를 그대로 돌려줌).
    pub fn check_observations_gpu(&self, cameras: &[Camera], poses: &[Rigid3], queries: &[ObsQuery]) -> Result<Vec<ObsCheck>, GpuError> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let cams: Vec<f64> = cameras.iter().flat_map(pack_camera).chain(std::iter::once(0.0)).collect();
        let mut ps: Vec<f64> = Vec::with_capacity(12 * poses.len() + 1);
        for p in poses {
            pack_pose(p, &mut ps);
        }
        ps.push(0.0);
        let mut qi: Vec<i32> = Vec::with_capacity(2 * queries.len());
        let mut qd: Vec<f64> = Vec::with_capacity(5 * queries.len());
        for q in queries {
            if q.camera as usize >= cameras.len() || q.pose as usize >= poses.len() {
                return Err(GpuError::NotAvailable(format!("관측 검사: 잘못된 번호 (카메라 {}, 자세 {})", q.camera, q.pose)));
            }
            qi.extend_from_slice(&[q.camera as i32, q.pose as i32]);
            qd.extend_from_slice(&[q.xy.x, q.xy.y, q.xyz.x, q.xyz.y, q.xyz.z]);
        }
        let n = queries.len();
        let out = {
            let _g = self.guard();
            let s = &self.dev.stream;
            let dqi = s.clone_htod(&qi)?;
            let dqd = s.clone_htod(&qd)?;
            let dc = s.clone_htod(&cams)?;
            let dp = s.clone_htod(&ps)?;
            let mut o = s.alloc_zeros::<f64>(2 * n)?;
            let n_i = n as i32;
            let cfg = LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let mut l = s.launch_builder(&self.f_obs);
            l.arg(&dqi).arg(&dqd).arg(&n_i).arg(&dc).arg(&dp).arg(&mut o);
            // SAFETY: check_obs(q_i[2n], q_d[5n], n, cams[9·nc], poses[12·np], out[2n]) 와 같은 인자·크기.
            unsafe { l.launch(cfg) }?;
            let out = s.clone_dtoh(&o)?;
            s.synchronize()?;
            out
        };
        Ok((0..n).map(|k| ObsCheck { sq_reproj_error: out[2 * k], depth: out[2 * k + 1] }).collect())
    }
}

/// 두 뷰 기하 가설 채점기(자료는 `load` 때 한 번 올린다).
struct GpuTwoViewScorer<'a> {
    sp: &'a CudaSparse,
    data: Option<CudaSlice<f64>>,
    /// 슬롯마다 (자료 시작 오프셋, 자료 수, 종류).
    slots: Vec<(i64, i64, i64)>,
}

impl GpuTwoViewScorer<'_> {
    fn load_gpu(&mut self, sets: &[ResidualSet<'_>]) -> Result<(), GpuError> {
        let mut host: Vec<f64> = Vec::new();
        self.slots.clear();
        for s in sets {
            let off = host.len() as i64;
            match *s {
                ResidualSet::EssentialRays(a, b) => {
                    for (p, q) in a.iter().zip(b) {
                        host.extend_from_slice(&[p.x, p.y, p.z, q.x, q.y, q.z]);
                    }
                    self.slots.push((off, a.len().min(b.len()) as i64, 0));
                }
                ResidualSet::FundamentalPx(a, b) => {
                    for (p, q) in a.iter().zip(b) {
                        host.extend_from_slice(&[p.x, p.y, q.x, q.y]);
                    }
                    self.slots.push((off, a.len().min(b.len()) as i64, 1));
                }
                ResidualSet::HomographyPx(a, b) => {
                    for (p, q) in a.iter().zip(b) {
                        host.extend_from_slice(&[p.x, p.y, q.x, q.y]);
                    }
                    self.slots.push((off, a.len().min(b.len()) as i64, 2));
                }
            }
        }
        if host.is_empty() {
            host.push(0.0);
        }
        let _g = self.sp.guard();
        self.data = Some(self.sp.dev.stream.clone_htod(&host)?);
        Ok(())
    }

    fn score_gpu(&mut self, reqs: &[ScoreRequest<'_>]) -> Result<Vec<Vec<Support>>, String> {
        let total: usize = reqs.iter().map(|r| r.models.len()).sum();
        if total == 0 {
            return Ok(reqs.iter().map(|_| Vec::new()).collect());
        }
        let data = self.data.as_ref().ok_or_else(|| "채점 전에 자료를 올려야 함".to_string())?;
        let mut ent_i: Vec<i64> = Vec::with_capacity(3 * total);
        let mut ent_m: Vec<f64> = Vec::with_capacity(10 * total);
        for r in reqs {
            let &(off, n, kind) = self.slots.get(r.slot).ok_or_else(|| format!("없는 슬롯 {}", r.slot))?;
            for m in r.models {
                ent_i.extend_from_slice(&[off, n, kind]);
                for row in 0..3 {
                    for col in 0..3 {
                        ent_m.push(m[(row, col)]);
                    }
                }
                ent_m.push(r.max_residual);
            }
        }
        let launch = || -> Result<(Vec<u32>, Vec<f64>), GpuError> {
            let _g = self.sp.guard();
            let s = &self.sp.dev.stream;
            let ei = s.clone_htod(&ent_i)?;
            let em = s.clone_htod(&ent_m)?;
            let mut on = s.alloc_zeros::<u32>(total)?;
            let mut os = s.alloc_zeros::<f64>(total)?;
            let cfg = LaunchConfig { grid_dim: (total as u32, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
            let mut l = s.launch_builder(&self.sp.f_two_view);
            l.arg(data).arg(&ei).arg(&em).arg(&mut on).arg(&mut os);
            // SAFETY: score_two_view(data, ent_i[3·total], ent_m[10·total], out_n[total], out_s[total]) 와 같은 인자·크기,
            // 오프셋·자료 수는 load 에서 올린 자료 안.
            unsafe { l.launch(cfg) }?;
            let cnt = s.clone_dtoh(&on)?;
            let sum = s.clone_dtoh(&os)?;
            s.synchronize()?;
            Ok((cnt, sum))
        };
        let (cnt, sum) = launch().map_err(|e| e.to_string())?;
        let mut out = Vec::with_capacity(reqs.len());
        let mut k = 0;
        for r in reqs {
            let m = r.models.len();
            out.push(supports(&cnt[k..k + m], &sum[k..k + m]));
            k += m;
        }
        Ok(out)
    }
}

impl BatchScorer for GpuTwoViewScorer<'_> {
    fn load(&mut self, sets: &[ResidualSet<'_>]) -> Result<(), String> {
        self.load_gpu(sets).map_err(|e| e.to_string())
    }
    fn score(&mut self, reqs: &[ScoreRequest<'_>]) -> Result<Vec<Vec<Support>>, String> {
        self.score_gpu(reqs)
    }
}

/// 절대 자세 가설 채점기(대응·카메라는 만들 때 한 번 올린다).
struct GpuPoseScorer<'a> {
    sp: &'a CudaSparse,
    data: CudaSlice<f64>,
    cam: CudaSlice<f64>,
    n: i32,
}

impl<'a> GpuPoseScorer<'a> {
    fn new(sp: &'a CudaSparse, camera: &Camera, points2d: &[Vec2], points3d: &[Vec3]) -> Result<Self, GpuError> {
        let mut host = Vec::with_capacity(5 * points2d.len() + 1);
        for (p, x) in points2d.iter().zip(points3d) {
            host.extend_from_slice(&[p.x, p.y, x.x, x.y, x.z]);
        }
        let n = points2d.len().min(points3d.len()) as i32;
        if host.is_empty() {
            host.push(0.0);
        }
        let cam = pack_camera(camera);
        let _g = sp.guard();
        let s = &sp.dev.stream;
        Ok(Self { sp, data: s.clone_htod(&host)?, cam: s.clone_htod(&cam.to_vec())?, n })
    }

    fn score_gpu(&mut self, poses: &[Rigid3], max_residual: f64) -> Result<Vec<Support>, GpuError> {
        if poses.is_empty() {
            return Ok(Vec::new());
        }
        let mut ph = Vec::with_capacity(12 * poses.len());
        for p in poses {
            pack_pose(p, &mut ph);
        }
        let np = poses.len();
        let _g = self.sp.guard();
        let s = &self.sp.dev.stream;
        let dp = s.clone_htod(&ph)?;
        let mut on = s.alloc_zeros::<u32>(np)?;
        let mut os = s.alloc_zeros::<f64>(np)?;
        let cfg = LaunchConfig { grid_dim: (np as u32, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        let mut l = s.launch_builder(&self.sp.f_pose);
        l.arg(&self.data).arg(&self.n).arg(&self.cam).arg(&dp).arg(&max_residual).arg(&mut on).arg(&mut os);
        // SAFETY: score_pose(data[5n], n, cam[9], poses[12·np], thr, out_n[np], out_s[np]) 와 같은 인자·크기.
        unsafe { l.launch(cfg) }?;
        let cnt = s.clone_dtoh(&on)?;
        let sum = s.clone_dtoh(&os)?;
        s.synchronize()?;
        Ok(supports(&cnt, &sum))
    }
}

impl PoseScorer for GpuPoseScorer<'_> {
    fn score(&mut self, poses: &[Rigid3], max_residual: f64) -> Result<Vec<Support>, String> {
        self.score_gpu(poses, max_residual).map_err(|e| e.to_string())
    }
}

impl TwoViewBackend for CudaSparse {
    fn verify_pairs(&self, store: &FeatureStore, jobs: &[VerifyJob<'_>], opts: &TwoViewOptions) -> Vec<TwoViewGeometry> {
        let mut sc = GpuTwoViewScorer { sp: self, data: None, slots: Vec::new() };
        match verify_pairs_batched(store, jobs, opts, &mut sc) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("경고: CUDA 기하 검증 실패({e}), CPU 로 계산");
                CpuTwoView.verify_pairs(store, jobs, opts)
            }
        }
    }
}

impl AbsPoseBackend for CudaSparse {
    fn solve(&self, camera: &Camera, points2d: &[Vec2], points3d: &[Vec3], opts: &AbsolutePoseOptions) -> Option<AbsolutePoseResult> {
        if points2d.len() != points3d.len() || points2d.len() < 3 {
            return solve_abs_pose(camera, points2d, points3d, opts);
        }
        let r = GpuPoseScorer::new(self, camera, points2d, points3d)
            .map_err(|e| e.to_string())
            .and_then(|mut sc| solve_abs_pose_scored(camera, points2d, points3d, opts, &mut sc));
        match r {
            Ok(v) => v,
            Err(e) => {
                eprintln!("경고: CUDA 절대 자세 실패({e}), CPU 로 계산");
                solve_abs_pose(camera, points2d, points3d, opts)
            }
        }
    }
}

impl TriangulationBackend for CudaSparse {
    fn estimate_batch(&self, problems: &[TriProblem<'_>]) -> Vec<Option<(Vec3, Vec<bool>)>> {
        if problems.len() < TRI_MIN_GPU {
            return CpuTriangulation.estimate_batch(problems);
        }
        match self.estimate_batch_gpu(problems) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("경고: CUDA 삼각측량 실패({e}), CPU 로 계산");
                CpuTriangulation.estimate_batch(problems)
            }
        }
    }

    fn check_observations(&self, cameras: &[Camera], poses: &[Rigid3], queries: &[ObsQuery]) -> Vec<ObsCheck> {
        if queries.len() < OBS_MIN_GPU {
            return CpuTriangulation.check_observations(cameras, poses, queries);
        }
        match self.check_observations_gpu(cameras, poses, queries) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("경고: CUDA 관측 검사 실패({e}), CPU 로 계산");
                CpuTriangulation.check_observations(cameras, poses, queries)
            }
        }
    }
}
