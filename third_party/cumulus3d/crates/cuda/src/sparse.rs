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
//! 가설 생성(최소 해법)·LO·모델 갱신은 CPU 가 그대로 한다. 채점 커널은 곱셈-덧셈 융합을 끄고 잔차를
//! CPU 와 같은 순서로 계산하므로 인라이어 수가 같다. 잔차 합은 더하는 순서가 달라 마지막 자리가 다를 수
//! 있어, 동점 가설 판정이 드물게 갈린다. 삼각측량 커널은 특이값 분해 방식이 달라 수치 잡음 수준의 차이가
//! 날 수 있다. 장치 오류는 경고 후 CPU 로 계산한다.
//!
//! 스트림은 가장 높은 우선순위로 만든다. 이 단계의 커널은 짧고 결과를 곧바로 기다리므로, 같은 GPU 의
//! 긴 작업(조밀화·UNet) 뒤에 줄 서면 계산보다 기다리는 시간이 길어진다.

use crate::device::{CudaDevice, GpuError, SyncedCopy};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PinnedHostSlice, PushKernelArg, ValidAsZeroBits};
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
    /// 한 번에 하나만 장치를 쓴다(스트림·버퍼 공유).
    lock: Mutex<Bufs>,
}

/// 고정(page-locked) 호스트 메모리를 거쳐 올리고 내린다. 일반 메모리에서 올리면 드라이버의 공용 중계
/// 버퍼를 거치느라 같은 GPU 의 다른 전송(영상 업로드 등) 뒤에 수백 ms 씩 줄을 선다. 버퍼는 늘기만 하고
/// 다시 쓴다. 잠금 구간 하나 안에서 올리는 자료는 `up` 의 서로 다른 구간을 쓰고, 다음 잠금 때 처음부터 다시 쓴다.
#[derive(Default)]
struct Staging {
    up: Option<PinnedHostSlice<u8>>,
    off: usize,
    down: Option<PinnedHostSlice<u8>>,
}

/// 스테이징 안 자료 시작 정렬(바이트).
const STAGE_ALIGN: usize = 256;

fn grow_pinned(ctx: &Arc<CudaContext>, slot: &mut Option<PinnedHostSlice<u8>>, bytes: usize, flags: u32) -> Result<(), GpuError> {
    if slot.as_ref().is_none_or(|b| b.len() < bytes) {
        let cap = (bytes + bytes / 4).max(1 << 16);
        *slot = None;
        // SAFETY: 내용은 쓰기 전에 읽지 않는다(올릴 때는 먼저 채우고, 내릴 때는 장치가 먼저 쓴다).
        *slot = Some(unsafe { ctx.alloc_pinned_with_flags::<u8>(cap, flags) }?);
    }
    Ok(())
}

/// 장치 버퍼를 최소 `n` 개로(늘기만 함).
fn grow_dev<T: DeviceRepr + ValidAsZeroBits>(s: &Arc<CudaStream>, slot: &mut Option<CudaSlice<T>>, n: usize) -> Result<(), GpuError> {
    let n = n.max(1);
    if slot.as_ref().is_none_or(|b| b.len() < n) {
        *slot = None;
        *slot = Some(s.alloc_zeros::<T>(n + n / 4)?);
    }
    Ok(())
}

fn dev_ref<T>(slot: &Option<CudaSlice<T>>) -> &CudaSlice<T> {
    slot.as_ref().expect("grow_dev 뒤에 씀")
}

fn bytes_of<T>(v: &[T]) -> usize {
    std::mem::size_of_val(v)
}

impl Staging {
    /// 잠금 구간 시작: 이번에 올릴 총 바이트(`parts` 개 자료, 정렬 여유 포함)를 받을 만큼 늘린다.
    fn begin(&mut self, ctx: &Arc<CudaContext>, bytes: usize, parts: usize) -> Result<(), GpuError> {
        self.off = 0;
        // 올리는 쪽은 호스트가 쓰기만 하므로 쓰기 결합 메모리.
        grow_pinned(ctx, &mut self.up, bytes + parts * STAGE_ALIGN, cudarc::driver::sys::CU_MEMHOSTALLOC_WRITECOMBINED)
    }

    /// `data` 를 스테이징에 복사하고 장치 버퍼 `dst`(필요하면 늘림) 앞부분으로 비동기 전송한다.
    /// 전송이 끝나기 전에는 같은 잠금 구간 안에서 그 구간을 다시 쓰지 않는다.
    fn upload<T: DeviceRepr + ValidAsZeroBits + Copy>(
        &mut self,
        s: &Arc<CudaStream>,
        data: &[T],
        dst: &mut Option<CudaSlice<T>>,
    ) -> Result<(), GpuError> {
        grow_dev(s, dst, data.len())?;
        if data.is_empty() {
            return Ok(());
        }
        let bytes = bytes_of(data);
        let up = self.up.as_mut().ok_or_else(|| GpuError::NotAvailable("스테이징 준비 전 업로드".into()))?;
        let buf = up.as_mut_slice()?;
        let start = self.off.next_multiple_of(STAGE_ALIGN);
        if start + bytes > buf.len() {
            return Err(GpuError::NotAvailable(format!("스테이징 부족: {start} + {bytes} > {}", buf.len())));
        }
        let region = &mut buf[start..start + bytes];
        // SAFETY: 고정 메모리는 페이지 정렬이고 start 는 256 의 배수라 T 정렬을 만족한다. 길이는 bytes 와 같다.
        let typed = unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, region.as_mut_ptr(), bytes);
            std::slice::from_raw_parts(region.as_ptr() as *const T, data.len())
        };
        self.off = start + bytes;
        let d = dst.as_mut().expect("grow_dev 뒤");
        s.memcpy_htod(typed, &mut d.slice_mut(0..data.len()))?;
        Ok(())
    }

    /// 장치 버퍼 앞 `n` 개를 고정 메모리로 내려 받아 돌려준다(스트림 동기화 포함).
    fn download<T: DeviceRepr + Copy>(&mut self, s: &Arc<CudaStream>, src: &CudaSlice<T>, n: usize) -> Result<Vec<T>, GpuError> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let bytes = n * std::mem::size_of::<T>();
        grow_pinned(s.context(), &mut self.down, bytes, 0)?;
        let down = self.down.as_mut().expect("grow_pinned 뒤");
        let buf = down.as_mut_slice()?;
        // SAFETY: 페이지 정렬된 고정 메모리, 길이 bytes 이상.
        let typed = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut T, n) };
        s.memcpy_dtoh_synced(&src.slice(0..n), typed)?;
        s.synchronize()?;
        Ok(typed.to_vec())
    }
}

/// 다시 쓰는 장치 버퍼와 스테이징(잠금 안에 둔다).
#[derive(Default)]
struct Bufs {
    st: Staging,
    // 관측 검사
    oq_i: Option<CudaSlice<i32>>,
    oq_d: Option<CudaSlice<f64>>,
    o_cam: Option<CudaSlice<f64>>,
    o_pose: Option<CudaSlice<f64>>,
    o_out: Option<CudaSlice<f64>>,
    // 삼각측량
    t_obs: Option<CudaSlice<f64>>,
    t_di: Option<CudaSlice<i64>>,
    t_dd: Option<CudaSlice<f64>>,
    t_x: Option<CudaSlice<f64>>,
    t_m: Option<CudaSlice<u64>>,
    t_ok: Option<CudaSlice<i32>>,
    // 채점 라운드(두 뷰·절대 자세)
    r_ei: Option<CudaSlice<i64>>,
    r_em: Option<CudaSlice<f64>>,
    r_n: Option<CudaSlice<u32>>,
    r_s: Option<CudaSlice<f64>>,
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
            lock: Mutex::new(Bufs::default()),
        })
    }

    /// 장치 0 으로 만든다.
    pub fn try_default() -> Result<Self, GpuError> {
        Self::new(CudaDevice::new_high_priority(0)?)
    }

    fn guard(&self) -> MutexGuard<'_, Bufs> {
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
        // 관측 수가 비슷한 문제끼리 한 워프에 들도록 정렬한다(스레드 하나 = 문제 하나라 갈라짐이 곧 대기).
        let mut gpu = gpu;
        gpu.sort_by_key(|&i| std::cmp::Reverse(problems[i].obs.len()));
        let mut starts = Vec::with_capacity(gpu.len());
        let mut start = 0usize;
        for &i in &gpu {
            starts.push(start);
            start += problems[i].obs.len();
        }
        let mut obs_host = vec![0.0f64; start * OBS_STRIDE];
        let mut desc_i = vec![0i64; 4 * gpu.len()];
        let mut desc_d = vec![0.0f64; 3 * gpu.len()];
        {
            // 문제마다 관측 구간이 겹치지 않으므로 구간별로 나눠 병렬로 채운다.
            let mut chunks: Vec<&mut [f64]> = Vec::with_capacity(gpu.len());
            let mut rest = &mut obs_host[..];
            for &i in &gpu {
                let (a, b) = rest.split_at_mut(problems[i].obs.len() * OBS_STRIDE);
                chunks.push(a);
                rest = b;
            }
            chunks
                .into_par_iter()
                .zip(desc_i.par_chunks_mut(4))
                .zip(desc_d.par_chunks_mut(3))
                .zip(gpu.par_iter().zip(starts.par_iter()))
                .for_each(|(((dst, di), dd), (&i, &st))| {
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
                    di.copy_from_slice(&[st as i64, n as i64, ro.min_trials as i64, max_trials as i64]);
                    dd.copy_from_slice(&[ro.max_error * ro.max_error, p.params.min_tri_angle_deg.to_radians(), ro.confidence]);
                    for (o, d) in p.obs.iter().zip(dst.chunks_exact_mut(OBS_STRIDE)) {
                        for r in 0..3 {
                            for c in 0..4 {
                                d[4 * r + c] = o.proj[(r, c)];
                            }
                        }
                        d[12..].copy_from_slice(&[o.center.x, o.center.y, o.center.z, o.ray.x, o.ray.y, o.ray.z]);
                    }
                });
        }
        debug_assert_eq!(obs_host.len(), start * OBS_STRIDE);
        let np = gpu.len();
        let (x, mask, ok) = {
            let mut g = self.guard();
            let b = &mut *g;
            let s = &self.dev.stream;
            b.st.begin(&self.dev.ctx, bytes_of(&obs_host) + bytes_of(&desc_i) + bytes_of(&desc_d), 3)?;
            b.st.upload(s, &obs_host, &mut b.t_obs)?;
            b.st.upload(s, &desc_i, &mut b.t_di)?;
            b.st.upload(s, &desc_d, &mut b.t_dd)?;
            grow_dev(s, &mut b.t_x, 3 * np)?;
            grow_dev(s, &mut b.t_m, np)?;
            grow_dev(s, &mut b.t_ok, np)?;
            let np_i = np as i32;
            let cfg = LaunchConfig { grid_dim: ((np as u32).div_ceil(64), 1, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 };
            let mut l = s.launch_builder(&self.f_tri);
            let (ox, om, ook) = (b.t_x.as_mut().expect("t_x"), b.t_m.as_mut().expect("t_m"), b.t_ok.as_mut().expect("t_ok"));
            l.arg(dev_ref(&b.t_obs)).arg(dev_ref(&b.t_di)).arg(dev_ref(&b.t_dd)).arg(&np_i).arg(ox).arg(om).arg(ook);
            // SAFETY: tri_ransac(obs[18·Σn], desc_i[4·np], desc_d[3·np], np, out_x[3·np], out_mask[np], out_ok[np]) 와 같은 인자,
            // 버퍼는 모두 그 크기 이상. 커널은 np 개만 읽고 쓴다.
            unsafe { l.launch(cfg) }?;
            let x = b.st.download(s, dev_ref(&b.t_x), 3 * np)?;
            let mask = b.st.download(s, dev_ref(&b.t_m), np)?;
            let ok = b.st.download(s, dev_ref(&b.t_ok), np)?;
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
        if let Some(q) = queries.par_iter().find_any(|q| q.camera as usize >= cameras.len() || q.pose as usize >= poses.len()) {
            return Err(GpuError::NotAvailable(format!("관측 검사: 잘못된 번호 (카메라 {}, 자세 {})", q.camera, q.pose)));
        }
        let mut qi = vec![0i32; 2 * queries.len()];
        let mut qd = vec![0.0f64; 5 * queries.len()];
        qi.par_chunks_mut(2).zip(qd.par_chunks_mut(5)).zip(queries.par_iter()).for_each(|((i, d), q)| {
            i.copy_from_slice(&[q.camera as i32, q.pose as i32]);
            d.copy_from_slice(&[q.xy.x, q.xy.y, q.xyz.x, q.xyz.y, q.xyz.z]);
        });
        let n = queries.len();
        let out = {
            let mut g = self.guard();
            let b = &mut *g;
            let s = &self.dev.stream;
            b.st.begin(&self.dev.ctx, bytes_of(&qi) + bytes_of(&qd) + bytes_of(&cams) + bytes_of(&ps), 4)?;
            b.st.upload(s, &qi, &mut b.oq_i)?;
            b.st.upload(s, &qd, &mut b.oq_d)?;
            b.st.upload(s, &cams, &mut b.o_cam)?;
            b.st.upload(s, &ps, &mut b.o_pose)?;
            grow_dev(s, &mut b.o_out, 2 * n)?;
            let n_i = n as i32;
            let cfg = LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let mut l = s.launch_builder(&self.f_obs);
            let o = b.o_out.as_mut().expect("o_out");
            l.arg(dev_ref(&b.oq_i)).arg(dev_ref(&b.oq_d)).arg(&n_i).arg(dev_ref(&b.o_cam)).arg(dev_ref(&b.o_pose)).arg(o);
            // SAFETY: check_obs(q_i[2n], q_d[5n], n, cams[9·nc], poses[12·np], out[2n]) 와 같은 인자, 버퍼는 그 크기 이상.
            unsafe { l.launch(cfg) }?;
            b.st.download(s, dev_ref(&b.o_out), 2 * n)?
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
    /// 계측: 채점 라운드 수, 장치에서 보낸 시간, 가설 수.
    rounds: usize,
    dev_time: std::time::Duration,
    hyps: usize,
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
        let mut g = self.sp.guard();
        let s = &self.sp.dev.stream;
        g.st.begin(&self.sp.dev.ctx, bytes_of(&host), 1)?;
        let mut data = None;
        g.st.upload(s, &host, &mut data)?;
        s.synchronize()?;
        self.data = data;
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
            let mut g = self.sp.guard();
            let b = &mut *g;
            let s = &self.sp.dev.stream;
            b.st.begin(&self.sp.dev.ctx, bytes_of(&ent_i) + bytes_of(&ent_m), 2)?;
            b.st.upload(s, &ent_i, &mut b.r_ei)?;
            b.st.upload(s, &ent_m, &mut b.r_em)?;
            grow_dev(s, &mut b.r_n, total)?;
            grow_dev(s, &mut b.r_s, total)?;
            let cfg = LaunchConfig { grid_dim: (total as u32, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
            let mut l = s.launch_builder(&self.sp.f_two_view);
            let (on, os) = (b.r_n.as_mut().expect("r_n"), b.r_s.as_mut().expect("r_s"));
            l.arg(data).arg(dev_ref(&b.r_ei)).arg(dev_ref(&b.r_em)).arg(on).arg(os);
            // SAFETY: score_two_view(data, ent_i[3·total], ent_m[10·total], out_n[total], out_s[total]) 와 같은 인자,
            // 버퍼는 그 크기 이상, 오프셋·자료 수는 load 에서 올린 자료 안.
            unsafe { l.launch(cfg) }?;
            let cnt = b.st.download(s, dev_ref(&b.r_n), total)?;
            let sum = b.st.download(s, dev_ref(&b.r_s), total)?;
            Ok((cnt, sum))
        };
        let t = std::time::Instant::now();
        let (cnt, sum) = launch().map_err(|e| e.to_string())?;
        self.rounds += 1;
        self.dev_time += t.elapsed();
        self.hyps += total;
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
        let mut g = sp.guard();
        let s = &sp.dev.stream;
        g.st.begin(&sp.dev.ctx, bytes_of(&host) + bytes_of(&cam), 2)?;
        let (mut data, mut dcam) = (None, None);
        g.st.upload(s, &host, &mut data)?;
        g.st.upload(s, &cam, &mut dcam)?;
        s.synchronize()?;
        drop(g);
        Ok(Self { sp, data: data.expect("업로드 뒤"), cam: dcam.expect("업로드 뒤"), n })
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
        let mut g = self.sp.guard();
        let b = &mut *g;
        let s = &self.sp.dev.stream;
        b.st.begin(&self.sp.dev.ctx, bytes_of(&ph), 1)?;
        b.st.upload(s, &ph, &mut b.r_em)?;
        grow_dev(s, &mut b.r_n, np)?;
        grow_dev(s, &mut b.r_s, np)?;
        let cfg = LaunchConfig { grid_dim: (np as u32, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
        let mut l = s.launch_builder(&self.sp.f_pose);
        let (on, os) = (b.r_n.as_mut().expect("r_n"), b.r_s.as_mut().expect("r_s"));
        l.arg(&self.data).arg(&self.n).arg(&self.cam).arg(dev_ref(&b.r_em)).arg(&max_residual).arg(on).arg(os);
        // SAFETY: score_pose(data[5n], n, cam[9], poses[12·np], thr, out_n[np], out_s[np]) 와 같은 인자, 버퍼는 그 크기 이상.
        unsafe { l.launch(cfg) }?;
        let cnt = b.st.download(s, dev_ref(&b.r_n), np)?;
        let sum = b.st.download(s, dev_ref(&b.r_s), np)?;
        Ok(supports(&cnt, &sum))
    }
}

impl PoseScorer for GpuPoseScorer<'_> {
    fn score(&mut self, poses: &[Rigid3], max_residual: f64) -> Result<Vec<Support>, String> {
        self.score_gpu(poses, max_residual).map_err(|e| e.to_string())
    }
}

/// `CUMULUS3D_CUDA_COMPARE=1` 이면 백엔드 호출마다 GPU 시간과 같은 입력의 CPU 시간을 함께 찍는다(계측용, 일을 두 번 한다).
pub(crate) fn compare_enabled() -> bool {
    std::env::var("CUMULUS3D_CUDA_COMPARE").is_ok_and(|v| v != "0")
}

/// 계측 한 줄: `[cuda-compare] <단계> n=<크기> gpu=<ms> cpu=<ms>`.
pub(crate) fn compare_line(stage: &str, n: usize, gpu: std::time::Duration, cpu: Option<std::time::Duration>) {
    match cpu {
        Some(c) => eprintln!("[cuda-compare] {stage} n={n} gpu={:.1}ms cpu={:.1}ms", gpu.as_secs_f64() * 1e3, c.as_secs_f64() * 1e3),
        None => eprintln!("[cuda-compare] {stage} n={n} gpu={:.1}ms", gpu.as_secs_f64() * 1e3),
    }
}

impl TwoViewBackend for CudaSparse {
    fn verify_pairs(&self, store: &FeatureStore, jobs: &[VerifyJob<'_>], opts: &TwoViewOptions) -> Vec<TwoViewGeometry> {
        let t0 = std::time::Instant::now();
        let mut sc = GpuTwoViewScorer { sp: self, data: None, slots: Vec::new(), rounds: 0, dev_time: Default::default(), hyps: 0 };
        match verify_pairs_batched(store, jobs, opts, &mut sc) {
            Ok(v) => {
                if compare_enabled() {
                    eprintln!(
                        "[cuda-compare] two-view phases rounds={} hyps={} dev={:.1}ms",
                        sc.rounds,
                        sc.hyps,
                        sc.dev_time.as_secs_f64() * 1e3
                    );
                    let g = t0.elapsed();
                    let t1 = std::time::Instant::now();
                    let _ = CpuTwoView.verify_pairs(store, jobs, opts);
                    compare_line("two-view", jobs.len(), g, Some(t1.elapsed()));
                }
                v
            }
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
        let t0 = std::time::Instant::now();
        let r = GpuPoseScorer::new(self, camera, points2d, points3d)
            .map_err(|e| e.to_string())
            .and_then(|mut sc| solve_abs_pose_scored(camera, points2d, points3d, opts, &mut sc));
        match r {
            Ok(v) => {
                if compare_enabled() {
                    let g = t0.elapsed();
                    let t1 = std::time::Instant::now();
                    let _ = solve_abs_pose(camera, points2d, points3d, opts);
                    compare_line("abs-pose", points2d.len(), g, Some(t1.elapsed()));
                }
                v
            }
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
            let t0 = std::time::Instant::now();
            let r = CpuTriangulation.estimate_batch(problems);
            if compare_enabled() {
                eprintln!("[cuda-compare] tri-ransac(cpu, small) n={} cpu={:.1}ms", problems.len(), t0.elapsed().as_secs_f64() * 1e3);
            }
            return r;
        }
        let t0 = std::time::Instant::now();
        match self.estimate_batch_gpu(problems) {
            Ok(r) => {
                if compare_enabled() {
                    let g = t0.elapsed();
                    let t1 = std::time::Instant::now();
                    let _ = CpuTriangulation.estimate_batch(problems);
                    compare_line("tri-ransac", problems.len(), g, Some(t1.elapsed()));
                }
                r
            }
            Err(e) => {
                eprintln!("경고: CUDA 삼각측량 실패({e}), CPU 로 계산");
                CpuTriangulation.estimate_batch(problems)
            }
        }
    }

    fn check_observations(&self, cameras: &[Camera], poses: &[Rigid3], queries: &[ObsQuery]) -> Vec<ObsCheck> {
        if queries.len() < OBS_MIN_GPU {
            let t0 = std::time::Instant::now();
            let r = CpuTriangulation.check_observations(cameras, poses, queries);
            if compare_enabled() {
                eprintln!("[cuda-compare] check-obs(cpu, small) n={} cpu={:.1}ms", queries.len(), t0.elapsed().as_secs_f64() * 1e3);
            }
            return r;
        }
        let t0 = std::time::Instant::now();
        match self.check_observations_gpu(cameras, poses, queries) {
            Ok(r) => {
                if compare_enabled() {
                    let g = t0.elapsed();
                    let t1 = std::time::Instant::now();
                    let _ = CpuTriangulation.check_observations(cameras, poses, queries);
                    compare_line("check-obs", queries.len(), g, Some(t1.elapsed()));
                }
                r
            }
            Err(e) => {
                eprintln!("경고: CUDA 관측 검사 실패({e}), CPU 로 계산");
                CpuTriangulation.check_observations(cameras, poses, queries)
            }
        }
    }
}
