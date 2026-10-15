/*
 * patchmatch.rs
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

//! `cumulus3d_dense::PatchMatchBackend` 의 CUDA 구현.
//!
//! - 세션 시작 때 모든 뷰·스케일의 회색 영상을 피치 정렬 버퍼 하나(기준 패치 읽기)와 뷰·스케일별 블록 선형 배열 +
//!   텍스처 객체(원천 표본, 하드웨어 쌍선형)로 올린다(상주).
//! - 반쪽 단계 커널은 활성 색 픽셀 하나당 스레드 하나(반쪽 격자), 블록 32×8 스레드 = 64×8 픽셀, 기준 패치는 공유 메모리 타일.
//! - 여러 기준 뷰를 그리드 z 축으로 묶어 한 번에 발사한다(픽셀 예산 안에서).
//! - 기하 실행의 원천 깊이 스냅숏은 스냅숏 번호가 바뀔 때만 다시 올린다.

use crate::device::{CudaDevice, GpuError};
use cudarc::driver::sys;
use cudarc::driver::{CudaFunction, CudaSlice, DevicePtr, DeviceRepr, LaunchConfig, PinnedHostSlice, PushKernelArg};
use cumulus3d_core::{Error, Result};
use cumulus3d_dense::kernel::{DepthSnapshot, KernelInput, PatchMatchBackend, PatchMatchSession, RunParams, ViewState};
use cumulus3d_dense::math::emission;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

pub(crate) const SRC: &str = include_str!("kernels/patchmatch.cu");
const MAXSRC: usize = 32;
const BX: u32 = 32;
const BY: u32 = 8;

/// CUDA PatchMatch 옵션.
#[derive(Clone, Debug, PartialEq)]
pub struct CudaPatchMatchOptions {
    /// 원천 표본을 텍스처 하드웨어 쌍선형으로(거짓이면 소프트웨어 쌍선형).
    pub hw_interp: bool,
    /// 한 번 발사에 묶는 픽셀 수 상한(여러 기준 뷰 동시 처리).
    pub batch_pixels: usize,
    /// 한 번 발사에 묶는 뷰 수 상한.
    pub max_batch: usize,
    /// 커널 `__launch_bounds__` 의 SM 당 최소 블록 수(레지스터 상한을 정한다).
    pub min_blocks: u32,
    /// 근사 나눗셈·제곱근(속도 우선).
    pub fast_math: bool,
}

impl Default for CudaPatchMatchOptions {
    fn default() -> Self {
        Self {
            hw_interp: true,
            batch_pixels: 2_200_000,
            max_batch: 64,
            min_blocks: std::env::var("CUMULUS3D_PM_MIN_BLOCKS").ok().and_then(|v| v.parse().ok()).unwrap_or(2),
            fast_math: std::env::var("CUMULUS3D_PM_FAST_MATH").map(|v| v != "0").unwrap_or(true),
        }
    }
}

struct Funcs {
    half: CudaFunction,
    eval: CudaFunction,
    filter: CudaFunction,
    upsample: CudaFunction,
    median: CudaFunction,
}

/// CUDA PatchMatch 백엔드.
pub struct CudaPatchMatch {
    dev: Arc<CudaDevice>,
    opts: CudaPatchMatchOptions,
    funcs: Mutex<HashMap<(i32, i32, bool), Arc<Funcs>>>,
    /// 장치 사용 직렬화(세션 하나씩).
    lock: Mutex<()>,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct LevelDesc {
    tex: u64,
    off: u64,
    pitch: i32,
    w: i32,
    h: i32,
    pad: i32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PairRec {
    src: i32,
    r: [f32; 9],
    t: [f32; 3],
    c: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct LevelPair {
    a: [f32; 9],
    b: [f32; 3],
}

/// A = K_j R K⁻¹, b = K_j t (스케일별).
fn level_pair(kr: [f32; 4], kj: [f32; 4], r: &[f32; 9], t: &[f32; 3]) -> LevelPair {
    let (fx, fy, cx, cy) = (kr[0] as f64, kr[1] as f64, kr[2] as f64, kr[3] as f64);
    let kinv = [[1.0 / fx, 0.0, -cx / fx], [0.0, 1.0 / fy, -cy / fy], [0.0, 0.0, 1.0]];
    let kjm = [[kj[0] as f64, 0.0, kj[2] as f64], [0.0, kj[1] as f64, kj[3] as f64], [0.0, 0.0, 1.0]];
    let rm: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|k| r[i * 3 + k] as f64));
    let mut kr_ = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for k in 0..3 {
            kr_[i][k] = (0..3).map(|m| kjm[i][m] * rm[m][k]).sum();
        }
    }
    let mut a = [0f32; 9];
    for i in 0..3 {
        for k in 0..3 {
            a[i * 3 + k] = (0..3).map(|m| kr_[i][m] * kinv[m][k]).sum::<f64>() as f32;
        }
    }
    let b: [f32; 3] = std::array::from_fn(|i| (0..3).map(|m| kjm[i][m] * t[m] as f64).sum::<f64>() as f32);
    LevelPair { a, b }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ViewDesc {
    key: u64,
    dmin: f32,
    dmax: f32,
    nsrc: i32,
    pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct BatchItem {
    off: i64,
    view: i32,
    w: i32,
    h: i32,
    pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct KParams {
    seed: u64,
    img: u64,
    lv: u64,
    pairs: u64,
    views: u64,
    batch: u64,
    snap: u64,
    snap_off: u64,
    plane: u64,
    cost: u64,
    bview: u64,
    count: u64,
    nlevels: i32,
    level: i32,
    geometric: i32,
    run_id: i32,
    t: i32,
    color: i32,
    random_init: i32,
    topk: i32,
    tau_good: f32,
    tau_bad: f32,
    inv2beta2: f32,
    prev_w: f32,
    n1: i32,
    n2: i32,
    inv2s: f32,
    inv2c: f32,
    cos_tri_min: f32,
    inv2inc: f32,
    lambda: f32,
    gmax: f32,
    eps: f32,
    phi: f32,
    cos_filter_tri: f32,
    q_min: f32,
    ncc_norm: f32,
    inv2ncc: f32,
    filter_gmax: f32,
    pad0: f32,
    prior: u64,
    use_prior: i32,
    weak_var: f32,
    lpairs: u64,
}

// SAFETY: 위 구조체들은 repr(C) 평범한 값이고 커널 쪽 선언과 배치가 같다.
unsafe impl DeviceRepr for KParams {}

fn bytes_of<T: Copy>(v: &[T]) -> Vec<u8> {
    // SAFETY: T 는 repr(C) Copy 값(패딩 없는 배치로 선언), 바이트로만 읽는다.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }.to_vec()
}

fn trace() -> bool {
    std::env::var("CUMULUS3D_CUDA_TRACE").is_ok_and(|v| v != "0")
}

fn gerr(e: impl std::fmt::Display) -> Error {
    Error::Invariant(format!("CUDA: {e}"))
}

impl CudaPatchMatch {
    /// 장치 `dev` 와 옵션으로 만든다(커널은 첫 사용 때 컴파일).
    pub fn new(dev: Arc<CudaDevice>, opts: CudaPatchMatchOptions) -> std::result::Result<Self, GpuError> {
        Ok(Self { dev, opts, funcs: Mutex::new(HashMap::new()), lock: Mutex::new(()) })
    }
    /// 장치 0, 기본 옵션.
    pub fn try_default() -> std::result::Result<Self, GpuError> {
        Self::new(CudaDevice::new(0)?, CudaPatchMatchOptions::default())
    }
    /// 사용하는 장치.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.dev
    }
    /// 현재 옵션.
    pub fn options(&self) -> &CudaPatchMatchOptions {
        &self.opts
    }
    fn funcs(&self, wr: i32, step: i32) -> std::result::Result<Arc<Funcs>, GpuError> {
        let key = (wr, step, self.opts.hw_interp);
        let _ = self.opts.fast_math;
        let mut g = self.funcs.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(f) = g.get(&key) {
            return Ok(f.clone());
        }
        let defs = defines(wr, step, self.opts.hw_interp, self.opts.min_blocks);
        let m = self.dev.compile_with(SRC, true, self.opts.fast_math, &defs)?;
        let f = Arc::new(Funcs {
            half: m.load_function("pm_half")?,
            eval: m.load_function("pm_eval")?,
            filter: m.load_function("pm_filter")?,
            upsample: m.load_function("pm_upsample")?,
            median: m.load_function("pm_median")?,
        });
        if trace() {
            use cudarc::driver::sys::CUfunction_attribute_enum as A;
            for (n, k) in [("pm_half", &f.half), ("pm_eval", &f.eval), ("pm_filter", &f.filter)] {
                let regs = k.num_regs().unwrap_or(-1);
                let local = k.get_attribute(A::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES).unwrap_or(-1);
                let occ = k.occupancy_max_active_blocks_per_multiprocessor(BX * BY, 0, None).unwrap_or(0);
                eprintln!("[cuda] {n} WR={wr} WSTEP={step}: 레지스터 {regs}, 지역 메모리 {local} B, SM 당 블록 {occ}");
            }
        }
        g.insert(key, f.clone());
        Ok(f)
    }
}

impl PatchMatchBackend for CudaPatchMatch {
    fn name(&self) -> String {
        format!("cuda({})", self.dev.name())
    }
    fn begin<'a>(&'a self, input: &'a KernelInput) -> Result<Box<dyn PatchMatchSession + 'a>> {
        let guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let pm = &input.pm;
        if !(1..=20).contains(&pm.window_radius) || !(1..=2).contains(&pm.window_step) {
            return Err(Error::InvalidArgument(format!("창 반경 {} (1..=20), 간격 {} (1|2)", pm.window_radius, pm.window_step)));
        }
        let funcs = self.funcs(pm.window_radius, pm.window_step).map_err(gerr)?;
        Session::new(self, guard, funcs, input).map(|s| Box::new(s) as Box<dyn PatchMatchSession + 'a>)
    }
}

struct Session<'a> {
    be: &'a CudaPatchMatch,
    _guard: MutexGuard<'a, ()>,
    f: Arc<Funcs>,
    input: &'a KernelInput,
    img: CudaSlice<u8>,
    img_shift: u64,
    texs: Vec<sys::CUtexObject>,
    arrays: Vec<sys::CUarray>,
    lv: CudaSlice<u8>,
    pairs: CudaSlice<u8>,
    lpairs: CudaSlice<u8>,
    views: CudaSlice<u8>,
    snap_id: Option<u64>,
    snap: CudaSlice<f32>,
    snap_off: CudaSlice<i64>,
    plane: CudaSlice<f32>,
    prior: CudaSlice<f32>,
    cost: CudaSlice<f32>,
    bview: CudaSlice<u8>,
    count: CudaSlice<u8>,
    batch: CudaSlice<u8>,
    /// 고정 호스트 버퍼(0: 평면 올림, 1: 사전·입력 평면 올림, 2: 평면 내림).
    stages: [Option<PinnedHostSlice<f32>>; 3],
    base: KParams,
}

/// 상태들을 (nx, ny, nz, d) 평면 배열로(병렬).
fn pack_planes(host: &mut [f32], states: &[&ViewState], offs: &[usize]) {
    use rayon::prelude::*;
    for (st, &o) in states.iter().zip(offs) {
        let n = st.depth.len();
        host[4 * o..4 * (o + n)].par_chunks_mut(4).zip(st.depth.par_iter().zip(st.normal.par_iter())).for_each(|(q, (d, nm))| {
            q.copy_from_slice(&[nm[0], nm[1], nm[2], *d]);
        });
    }
}

/// 평면 배열 → 상태(병렬).
fn unpack_planes(p: &[f32], st: &mut ViewState, o: usize) {
    use rayon::prelude::*;
    let n = st.width * st.height;
    st.depth.resize(n, 0.0);
    st.normal.resize(n, [0.0; 3]);
    st.depth.par_iter_mut().zip(st.normal.par_iter_mut()).zip(p[4 * o..4 * (o + n)].par_chunks(4)).for_each(|((d, nm), q)| {
        *nm = [q[0], q[1], q[2]];
        *d = q[3];
    });
}

fn round_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

impl<'a> Session<'a> {
    fn new(be: &'a CudaPatchMatch, guard: MutexGuard<'a, ()>, f: Arc<Funcs>, input: &'a KernelInput) -> Result<Self> {
        let dev = &be.dev;
        let s = &dev.stream;
        let nl = input.num_levels;
        let nv = input.views.len();
        // 영상 배치: 뷰·스케일마다 피치 정렬, 시작 주소 텍스처 정렬.
        let ta = dev.tex_align.max(256);
        let pa = dev.pitch_align.max(32);
        let mut layout = Vec::with_capacity(nv * nl);
        let mut total = 0usize;
        for v in &input.views {
            if v.levels.len() != nl {
                return Err(Error::InvalidArgument("스케일 수 불일치".into()));
            }
            if v.sources.len() > MAXSRC {
                return Err(Error::InvalidArgument(format!("원천 뷰 {} > {MAXSRC}", v.sources.len())));
            }
            for l in &v.levels {
                let pitch = round_up(l.width.max(1), pa);
                total = round_up(total, ta);
                layout.push((total, pitch));
                total += pitch * l.height;
            }
        }
        let alloc = total + ta;
        let mut img = s.alloc_zeros::<u8>(alloc.max(1)).map_err(gerr)?;
        let base_ptr = {
            let (p, _g) = img.device_ptr(s);
            p
        };
        let shift = ((ta as u64 - base_ptr % ta as u64) % ta as u64) as usize;
        let mut host = vec![0u8; alloc];
        let mut i = 0;
        for v in &input.views {
            for l in &v.levels {
                let (off, pitch) = layout[i];
                for y in 0..l.height {
                    let d = shift + off + y * pitch;
                    host[d..d + l.width].copy_from_slice(&l.gray[y * l.width..(y + 1) * l.width]);
                }
                i += 1;
            }
        }
        s.memcpy_htod(&host, &mut img).map_err(gerr)?;
        drop(host);
        dev.ctx.bind_to_thread().map_err(gerr)?;
        let mut texs = Vec::with_capacity(nv * nl);
        let mut arrays: Vec<sys::CUarray> = Vec::with_capacity(nv * nl);
        let mut lvd = Vec::with_capacity(nv * nl);
        let mut i = 0;
        for v in &input.views {
            for l in &v.levels {
                let (off, pitch) = layout[i];
                i += 1;
                // 블록 선형 배열(2D 지역성이 좋은 텍스처 배치)에 복사.
                let ad = sys::CUDA_ARRAY_DESCRIPTOR {
                    Width: l.width,
                    Height: l.height,
                    Format: sys::CUarray_format::CU_AD_FORMAT_UNSIGNED_INT8,
                    NumChannels: 1,
                };
                let mut arr: sys::CUarray = std::ptr::null_mut();
                // SAFETY: 유효한 서술자. 배열은 Drop 에서 해제한다.
                let r = unsafe { sys::cuArrayCreate_v2(&mut arr, &ad) };
                if r != sys::CUresult::CUDA_SUCCESS {
                    return Err(gerr(format!("배열 생성 실패 {r:?}")));
                }
                arrays.push(arr);
                let cp = sys::CUDA_MEMCPY2D {
                    srcXInBytes: 0,
                    srcY: 0,
                    srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_HOST,
                    srcHost: l.gray.as_ptr() as *const std::ffi::c_void,
                    srcDevice: 0,
                    srcArray: std::ptr::null_mut(),
                    srcPitch: l.width,
                    dstXInBytes: 0,
                    dstY: 0,
                    dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_ARRAY,
                    dstHost: std::ptr::null_mut(),
                    dstDevice: 0,
                    dstArray: arr,
                    dstPitch: 0,
                    WidthInBytes: l.width,
                    Height: l.height,
                };
                // SAFETY: 입력 호스트 버퍼는 width·height 바이트, 대상 배열은 같은 크기.
                let r = unsafe { sys::cuMemcpy2D_v2(&cp) };
                if r != sys::CUresult::CUDA_SUCCESS {
                    return Err(gerr(format!("배열 복사 실패 {r:?}")));
                }
                // SAFETY: 0 으로 채운 뒤 필요한 필드만 쓴다.
                let mut res: sys::CUDA_RESOURCE_DESC = unsafe { std::mem::zeroed() };
                res.resType = sys::CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
                res.res.array = sys::CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 { hArray: arr };
                // SAFETY: 위와 같음.
                let mut td: sys::CUDA_TEXTURE_DESC = unsafe { std::mem::zeroed() };
                td.addressMode = [sys::CUaddress_mode::CU_TR_ADDRESS_MODE_BORDER; 3];
                td.filterMode = sys::CUfilter_mode::CU_TR_FILTER_MODE_LINEAR;
                td.flags = 0;
                let mut tex: sys::CUtexObject = 0;
                // SAFETY: 유효한 서술자, 버퍼는 세션 동안 살아 있다(Drop 에서 객체를 먼저 해제).
                let r = unsafe { sys::cuTexObjectCreate(&mut tex, &res, &td, std::ptr::null()) };
                if r != sys::CUresult::CUDA_SUCCESS {
                    return Err(gerr(format!("텍스처 생성 실패 {r:?}")));
                }
                texs.push(tex);
                lvd.push(LevelDesc {
                    tex,
                    off: (shift + off) as u64,
                    pitch: pitch as i32,
                    w: l.width as i32,
                    h: l.height as i32,
                    pad: 0,
                    fx: l.k[0],
                    fy: l.k[1],
                    cx: l.k[2],
                    cy: l.k[3],
                });
            }
        }
        let mut pr = vec![PairRec::default(); nv * MAXSRC];
        let mut vd = Vec::with_capacity(nv);
        for (vi, v) in input.views.iter().enumerate() {
            for (j, (&src, g)) in v.sources.iter().zip(&v.pairs).enumerate() {
                pr[vi * MAXSRC + j] = PairRec { src: src as i32, r: g.r, t: g.t, c: g.center };
            }
            vd.push(ViewDesc { key: v.key, dmin: v.depth_min, dmax: v.depth_max, nsrc: v.sources.len() as i32, pad: 0 });
        }
        let mut lpv = vec![LevelPair::default(); nv * nl * MAXSRC];
        for (vi, v) in input.views.iter().enumerate() {
            for l in 0..nl {
                for (j, (&src, g)) in v.sources.iter().zip(&v.pairs).enumerate() {
                    lpv[(vi * nl + l) * MAXSRC + j] = level_pair(v.levels[l].k, input.views[src].levels[l].k, &g.r, &g.t);
                }
            }
        }
        let lpairs = s.clone_htod(&bytes_of(&lpv)).map_err(gerr)?;
        let lv = s.clone_htod(&bytes_of(&lvd)).map_err(gerr)?;
        let pairs = s.clone_htod(&bytes_of(&pr)).map_err(gerr)?;
        let views = s.clone_htod(&bytes_of(&vd)).map_err(gerr)?;
        let p = &input.pm;
        let fl = &input.filter;
        let sc = p.sigma_s();
        let base = KParams {
            seed: input.seed,
            nlevels: nl as i32,
            topk: p.top_k as i32,
            tau_bad: p.tau1,
            inv2beta2: 1.0 / (2.0 * p.beta * p.beta),
            prev_w: p.prev_view_weight,
            n1: p.n1 as i32,
            n2: p.n2 as i32,
            inv2s: 1.0 / (2.0 * sc * sc),
            inv2c: 1.0 / (2.0 * p.sigma_color * p.sigma_color),
            cos_tri_min: p.min_triangulation_angle_deg.to_radians().cos(),
            inv2inc: 1.0 / (2.0 * p.incident_angle_sigma * p.incident_angle_sigma),
            lambda: p.geom_lambda,
            gmax: p.geom_max_cost,
            cos_filter_tri: fl.min_triangulation_angle_deg.to_radians().cos(),
            q_min: emission(1.0 - fl.min_ncc as f64, fl.ncc_sigma as f64) as f32,
            ncc_norm: cumulus3d_dense::math::emission_norm(fl.ncc_sigma as f64) as f32,
            inv2ncc: 1.0 / (2.0 * fl.ncc_sigma * fl.ncc_sigma),
            filter_gmax: fl.geom_max_cost,
            weak_var: p.weak_texture_var,
            ..Default::default()
        };
        Ok(Self {
            be,
            _guard: guard,
            f,
            input,
            snap: s.alloc_zeros::<f32>(1).map_err(gerr)?,
            snap_off: s.clone_htod(&vec![-1i64; nv.max(1)]).map_err(gerr)?,
            img,
            img_shift: shift as u64,
            texs,
            arrays,
            lv,
            pairs,
            lpairs,
            views,
            snap_id: None,
            plane: s.alloc_zeros::<f32>(4).map_err(gerr)?,
            prior: s.alloc_zeros::<f32>(4).map_err(gerr)?,
            cost: s.alloc_zeros::<f32>(1).map_err(gerr)?,
            bview: s.alloc_zeros::<u8>(1).map_err(gerr)?,
            count: s.alloc_zeros::<u8>(1).map_err(gerr)?,
            batch: s.alloc_zeros::<u8>(1).map_err(gerr)?,
            stages: [None, None, None],
            base,
        })
    }

    fn take_stage(&mut self, which: usize, n: usize) -> Result<PinnedHostSlice<f32>> {
        match self.stages[which].take() {
            Some(b) if b.len() >= n => Ok(b),
            _ => {
                drop(self.stages[which].take());
                // SAFETY: 고정 메모리 할당; 쓰기 전에 읽지 않는다(필요 구간을 먼저 채운다).
                unsafe { self.be.dev.ctx.alloc_pinned::<f32>(n.max(1) + n / 4) }.map_err(gerr)
            }
        }
    }

    fn ensure_capacity(&mut self, npix: usize) -> Result<()> {
        let s = &self.be.dev.stream;
        if self.cost.len() < npix {
            let n = npix.max(1);
            self.plane = s.alloc_zeros::<f32>(4 * n).map_err(gerr)?;
            self.prior = s.alloc_zeros::<f32>(4 * n).map_err(gerr)?;
            self.cost = s.alloc_zeros::<f32>(n).map_err(gerr)?;
            self.bview = s.alloc_zeros::<u8>(n).map_err(gerr)?;
            self.count = s.alloc_zeros::<u8>(n).map_err(gerr)?;
        }
        Ok(())
    }

    fn ensure_snapshot(&mut self, snap: &DepthSnapshot) -> Result<()> {
        if self.snap_id == Some(snap.id) {
            return Ok(());
        }
        let s = &self.be.dev.stream;
        let nv = self.input.views.len();
        let mut offs = vec![-1i64; nv.max(1)];
        let mut total = 0usize;
        for (v, off) in offs.iter_mut().enumerate().take(nv) {
            if let Some(Some(m)) = snap.maps.get(v) {
                let l = &self.input.views[v].levels[snap.level];
                if m.len() == l.width * l.height {
                    *off = total as i64;
                    total += m.len();
                }
            }
        }
        let mut host = vec![0f32; total.max(1)];
        for (v, &o) in offs.iter().enumerate().take(nv) {
            if o >= 0 {
                let m = snap.maps[v].as_ref().expect("있음");
                host[o as usize..o as usize + m.len()].copy_from_slice(m);
            }
        }
        if self.snap.len() < host.len() {
            self.snap = s.alloc_zeros::<f32>(host.len()).map_err(gerr)?;
        }
        s.memcpy_htod(&host, &mut self.snap).map_err(gerr)?;
        s.memcpy_htod(&offs, &mut self.snap_off).map_err(gerr)?;
        self.snap_id = Some(snap.id);
        Ok(())
    }

    /// 뷰 목록을 픽셀 예산으로 묶는다(반환: 시작·끝 색인).
    fn batches(&self, views: &[usize], level: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut start = 0;
        let mut pix = 0usize;
        for (i, &v) in views.iter().enumerate() {
            let l = &self.input.views[v].levels[level];
            let p = l.width * l.height;
            if i > start && (pix + p > self.be.opts.batch_pixels || i - start >= self.be.opts.max_batch) {
                out.push((start, i));
                start = i;
                pix = 0;
            }
            pix += p;
        }
        if start < views.len() {
            out.push((start, views.len()));
        }
        out
    }

    /// 묶음 하나의 상태를 올리고 매개변수를 채운다. 반환: (픽셀 수, 최대 폭, 최대 높이, 항목들).
    fn upload_batch(
        &mut self,
        views: &[usize],
        states: &[ViewState],
        level: usize,
        with_plane: bool,
        with_prior: bool,
    ) -> Result<(usize, usize, usize, Vec<BatchItem>)> {
        let mut items = Vec::with_capacity(views.len());
        let (mut off, mut mw, mut mh) = (0usize, 0usize, 0usize);
        for (&v, st) in views.iter().zip(states) {
            let l = &self.input.views[v].levels[level];
            if st.width != l.width || st.height != l.height {
                return Err(Error::InvalidArgument(format!("상태 크기 {}x{} != 스케일 {}x{}", st.width, st.height, l.width, l.height)));
            }
            items.push(BatchItem { off: off as i64, view: v as i32, w: l.width as i32, h: l.height as i32, pad: 0 });
            off += l.width * l.height;
            mw = mw.max(l.width);
            mh = mh.max(l.height);
        }
        self.ensure_capacity(off)?;
        let s = self.be.dev.stream.clone();
        if with_plane {
            let mut st = self.take_stage(0, 4 * off)?;
            let refs: Vec<&ViewState> = states.iter().collect();
            let offs: Vec<usize> = items.iter().map(|it| it.off as usize).collect();
            pack_planes(&mut st.as_mut_slice().map_err(gerr)?[..4 * off], &refs, &offs);
            s.memcpy_htod(&st.as_slice().map_err(gerr)?[..4 * off], &mut self.plane.slice_mut(0..4 * off)).map_err(gerr)?;
            self.stages[0] = Some(st);
        }
        if with_prior && states.iter().all(|s| s.prior.len() == s.width * s.height) {
            let mut pb = self.take_stage(1, 4 * off)?;
            {
                let host = pb.as_mut_slice().map_err(gerr)?;
                for (it, st) in items.iter().zip(states) {
                    let o = it.off as usize;
                    host[4 * o..4 * o + 4 * st.prior.len()].copy_from_slice(st.prior.as_flattened());
                }
            }
            s.memcpy_htod(&pb.as_slice().map_err(gerr)?[..4 * off], &mut self.prior.slice_mut(0..4 * off)).map_err(gerr)?;
            self.stages[1] = Some(pb);
        }
        let b = bytes_of(&items);
        if self.batch.len() < b.len() {
            self.batch = s.alloc_zeros::<u8>(b.len()).map_err(gerr)?;
        }
        s.memcpy_htod(&b, &mut self.batch.slice_mut(0..b.len())).map_err(gerr)?;
        Ok((off, mw, mh, items))
    }

    fn params(&self, level: usize, geometric: bool) -> KParams {
        let s = &self.be.dev.stream;
        let ptr8 = |x: &CudaSlice<u8>| x.device_ptr(s).0;
        let ptrf = |x: &CudaSlice<f32>| x.device_ptr(s).0;
        let mut k = self.base;
        k.img = ptr8(&self.img);
        k.lv = ptr8(&self.lv);
        k.pairs = ptr8(&self.pairs);
        k.lpairs = ptr8(&self.lpairs);
        k.views = ptr8(&self.views);
        k.batch = ptr8(&self.batch);
        k.snap = ptrf(&self.snap);
        k.snap_off = self.snap_off.device_ptr(s).0;
        k.plane = ptrf(&self.plane);
        k.prior = ptrf(&self.prior);
        k.cost = ptrf(&self.cost);
        k.bview = ptr8(&self.bview);
        k.count = ptr8(&self.count);
        k.level = level as i32;
        k.geometric = geometric as i32;
        k
    }

    fn launch_eval(&self, k: &KParams, mode: i32, mw: usize, mh: usize, nb: usize) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: ((mw as u32).div_ceil(BX), (mh as u32).div_ceil(BY), nb as u32),
            block_dim: (BX, BY, 1),
            shared_mem_bytes: 0,
        };
        let s = &self.be.dev.stream;
        let mut l = s.launch_builder(&self.f.eval);
        l.arg(k).arg(&mode);
        // SAFETY: pm_eval(KParams, int); 버퍼 크기는 upload_batch 가 맞춘다.
        unsafe { l.launch(cfg) }.map_err(gerr)?;
        Ok(())
    }

    fn launch_half(&self, k: &KParams, mw: usize, mh: usize, nb: usize) -> Result<()> {
        let hw = mw.div_ceil(2) as u32;
        let cfg =
            LaunchConfig { grid_dim: (hw.div_ceil(BX), (mh as u32).div_ceil(BY), nb as u32), block_dim: (BX, BY, 1), shared_mem_bytes: 0 };
        let s = &self.be.dev.stream;
        let mut l = s.launch_builder(&self.f.half);
        l.arg(k);
        // SAFETY: pm_half(KParams); 정적 공유 메모리.
        unsafe { l.launch(cfg) }.map_err(gerr)?;
        Ok(())
    }

    fn download_plane_cost(&mut self, items: &[BatchItem], npix: usize, states: &mut [ViewState], plane: bool) -> Result<()> {
        let s = self.be.dev.stream.clone();
        let mut c = vec![0f32; npix];
        s.memcpy_dtoh(&self.cost.slice(0..npix), &mut c).map_err(gerr)?;
        let mut pb = self.take_stage(2, if plane { 4 * npix } else { 1 })?;
        if plane {
            s.memcpy_dtoh(&self.plane.slice(0..4 * npix), &mut pb.as_mut_slice().map_err(gerr)?[..4 * npix]).map_err(gerr)?;
        }
        s.synchronize().map_err(gerr)?;
        let p = pb.as_slice().map_err(gerr)?;
        for (it, st) in items.iter().zip(states.iter_mut()) {
            let o = it.off as usize;
            let n = st.width * st.height;
            st.cost.resize(n, 0.0);
            st.cost.copy_from_slice(&c[o..o + n]);
            if plane {
                unpack_planes(p, st, o);
            }
        }
        self.stages[2] = Some(pb);
        Ok(())
    }
}

impl PatchMatchSession for Session<'_> {
    fn run(&mut self, views: &[usize], states: &mut [ViewState], p: &RunParams, snapshot: Option<&DepthSnapshot>) -> Result<()> {
        if p.geometric {
            let snap = snapshot.ok_or_else(|| Error::InvalidArgument("기하 실행에 스냅숏 필요".into()))?;
            self.ensure_snapshot(snap)?;
        }
        let pm = self.input.pm.clone();
        let t0 = std::time::Instant::now();
        let mut tk = std::time::Duration::ZERO;
        for (a, b) in self.batches(views, p.level) {
            let (npix, mw, mh, items) = self.upload_batch(&views[a..b], &states[a..b], p.level, !p.random_init, p.use_prior)?;
            let mut k = self.params(p.level, p.geometric);
            k.run_id = p.run_id as i32;
            k.random_init = p.random_init as i32;
            k.use_prior = (p.use_prior && states[a..b].iter().all(|s| s.prior.len() == s.width * s.height)) as i32;
            self.launch_eval(&k, 1, mw, mh, b - a)?;
            let (eps0, phi0) =
                if p.geometric { (pm.eps0_geometric, pm.phi0_geometric_deg) } else { (pm.eps0_photometric, pm.phi0_photometric_deg) };
            for t in 0..p.iterations {
                let tf = t as f32;
                k.t = t as i32;
                k.tau_good = pm.tau0 * (-tf * tf / pm.tau_alpha).exp();
                k.eps = (eps0 * 0.5f32.powi(t as i32)).max(pm.eps_min);
                k.phi = (phi0 * 0.5f32.powi(t as i32)).max(pm.phi_min_deg).to_radians();
                for color in 0..2 {
                    k.color = color;
                    self.launch_half(&k, mw, mh, b - a)?;
                }
            }
            if trace() {
                let tt = std::time::Instant::now();
                self.be.dev.stream.synchronize().map_err(gerr)?;
                tk += tt.elapsed();
            }
            self.download_plane_cost(&items, npix, &mut states[a..b], true)?;
        }
        if trace() {
            eprintln!(
                "[cuda] run 스케일 {} {} 반복 {} 뷰 {}: {:.3}s (커널 대기 {:.3}s)",
                p.level,
                if p.geometric { "기하" } else { "광도" },
                p.iterations,
                views.len(),
                t0.elapsed().as_secs_f64(),
                tk.as_secs_f64()
            );
        }
        Ok(())
    }

    fn evaluate(
        &mut self,
        level: usize,
        views: &[usize],
        states: &[ViewState],
        geometric: bool,
        snapshot: Option<&DepthSnapshot>,
    ) -> Result<Vec<Vec<f32>>> {
        if geometric {
            let snap = snapshot.ok_or_else(|| Error::InvalidArgument("기하 평가에 스냅숏 필요".into()))?;
            self.ensure_snapshot(snap)?;
        }
        let mut out = Vec::with_capacity(views.len());
        for (a, b) in self.batches(views, level) {
            let (npix, mw, mh, items) = self.upload_batch(&views[a..b], &states[a..b], level, true, false)?;
            let k = self.params(level, geometric);
            self.launch_eval(&k, 0, mw, mh, b - a)?;
            let mut tmp: Vec<ViewState> =
                states[a..b].iter().map(|s| ViewState { width: s.width, height: s.height, ..Default::default() }).collect();
            self.download_plane_cost(&items, npix, &mut tmp, false)?;
            out.extend(tmp.into_iter().map(|s| s.cost));
        }
        Ok(out)
    }

    fn upsample(
        &mut self,
        _input: &KernelInput,
        level: usize,
        views: &[usize],
        low: &[ViewState],
        sigma_s: f32,
        sigma_c: f32,
    ) -> Result<Vec<ViewState>> {
        self.upsample_gpu(level, views, low, sigma_s, sigma_c)
    }

    fn median_filter(&mut self, _input: &KernelInput, level: usize, views: &[usize], states: &[ViewState]) -> Result<Vec<ViewState>> {
        self.median_gpu(level, views, states)
    }

    fn filter(&mut self, views: &[usize], states: &[ViewState], snapshot: &DepthSnapshot) -> Result<Vec<Vec<u8>>> {
        self.ensure_snapshot(snapshot)?;
        let level = snapshot.level;
        let mut out = Vec::with_capacity(views.len());
        for (a, b) in self.batches(views, level) {
            let (npix, mw, mh, items) = self.upload_batch(&views[a..b], &states[a..b], level, true, false)?;
            let k = self.params(level, true);
            let cfg = LaunchConfig {
                grid_dim: ((mw as u32).div_ceil(BX), (mh as u32).div_ceil(BY), (b - a) as u32),
                block_dim: (BX, BY, 1),
                shared_mem_bytes: 0,
            };
            let s = self.be.dev.stream.clone();
            let mut l = s.launch_builder(&self.f.filter);
            l.arg(&k);
            // SAFETY: pm_filter(KParams).
            unsafe { l.launch(cfg) }.map_err(gerr)?;
            let mut c = vec![0u8; npix];
            s.memcpy_dtoh(&self.count.slice(0..npix), &mut c).map_err(gerr)?;
            s.synchronize().map_err(gerr)?;
            for it in &items {
                let o = it.off as usize;
                out.push(c[o..o + (it.w * it.h) as usize].to_vec());
            }
        }
        Ok(out)
    }
}

impl Session<'_> {
    /// 평면 묶음 올리기(prior 버퍼, 항목별 시작 `offs`).
    fn upload_planes_to_prior(&mut self, states: &[&ViewState], offs: &[usize], total: usize) -> Result<()> {
        let s = self.be.dev.stream.clone();
        if self.prior.len() < 4 * total.max(1) {
            self.prior = s.alloc_zeros::<f32>(4 * total.max(1)).map_err(gerr)?;
        }
        let mut pb = self.take_stage(1, 4 * total)?;
        pack_planes(&mut pb.as_mut_slice().map_err(gerr)?[..4 * total], states, offs);
        s.memcpy_htod(&pb.as_slice().map_err(gerr)?[..4 * total], &mut self.prior.slice_mut(0..4 * total)).map_err(gerr)?;
        self.stages[1] = Some(pb);
        Ok(())
    }

    fn set_batch(&mut self, items: &[BatchItem]) -> Result<()> {
        let s = self.be.dev.stream.clone();
        let b = bytes_of(items);
        if self.batch.len() < b.len() {
            self.batch = s.alloc_zeros::<u8>(b.len()).map_err(gerr)?;
        }
        s.memcpy_htod(&b, &mut self.batch.slice_mut(0..b.len())).map_err(gerr)?;
        Ok(())
    }

    fn download_planes(&mut self, items: &[BatchItem], npix: usize) -> Result<Vec<ViewState>> {
        let s = self.be.dev.stream.clone();
        let mut pb = self.take_stage(2, 4 * npix)?;
        s.memcpy_dtoh(&self.plane.slice(0..4 * npix), &mut pb.as_mut_slice().map_err(gerr)?[..4 * npix]).map_err(gerr)?;
        s.synchronize().map_err(gerr)?;
        let p = pb.as_slice().map_err(gerr)?;
        let out = items
            .iter()
            .map(|it| {
                let mut st = ViewState::new(it.w as usize, it.h as usize);
                unpack_planes(p, &mut st, it.off as usize);
                st
            })
            .collect();
        self.stages[2] = Some(pb);
        Ok(out)
    }

    fn launch_simple(&self, f: &CudaFunction, k: &KParams, extra: Option<(f32, f32)>, mw: usize, mh: usize, nb: usize) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: ((mw as u32).div_ceil(BX), (mh as u32).div_ceil(BY), nb as u32),
            block_dim: (BX, BY, 1),
            shared_mem_bytes: 0,
        };
        let s = &self.be.dev.stream;
        let mut l = s.launch_builder(f);
        l.arg(k);
        let (a, b) = extra.unwrap_or((0.0, 0.0));
        if extra.is_some() {
            l.arg(&a).arg(&b);
        }
        // SAFETY: pm_upsample(KParams, float, float) / pm_median(KParams); 버퍼 크기는 호출자가 맞춘다.
        unsafe { l.launch(cfg) }.map_err(gerr)?;
        Ok(())
    }

    fn upsample_gpu(&mut self, level: usize, views: &[usize], low: &[ViewState], ss: f32, sc: f32) -> Result<Vec<ViewState>> {
        let mut out = Vec::with_capacity(views.len());
        for (a, b) in self.batches(views, level) {
            let mut items = Vec::new();
            let mut lowoffs = Vec::new();
            let (mut off, mut loff, mut mw, mut mh) = (0usize, 0usize, 0usize, 0usize);
            for (&v, st) in views[a..b].iter().zip(&low[a..b]) {
                let l = &self.input.views[v].levels[level];
                let ll = &self.input.views[v].levels[level - 1];
                if st.width != ll.width || st.height != ll.height {
                    return Err(Error::InvalidArgument("상향 표본 입력 크기".into()));
                }
                items.push(BatchItem { off: off as i64, view: v as i32, w: l.width as i32, h: l.height as i32, pad: loff as i32 });
                lowoffs.push(loff);
                off += l.width * l.height;
                loff += ll.width * ll.height;
                mw = mw.max(l.width);
                mh = mh.max(l.height);
            }
            self.ensure_capacity(off)?;
            let refs: Vec<&ViewState> = low[a..b].iter().collect();
            self.upload_planes_to_prior(&refs, &lowoffs, loff)?;
            self.set_batch(&items)?;
            let k = self.params(level, false);
            let f = self.f.upsample.clone();
            self.launch_simple(&f, &k, Some((1.0 / (2.0 * ss * ss), 1.0 / (2.0 * sc * sc))), mw, mh, b - a)?;
            out.extend(self.download_planes(&items, off)?);
        }
        Ok(out)
    }

    fn median_gpu(&mut self, level: usize, views: &[usize], states: &[ViewState]) -> Result<Vec<ViewState>> {
        let mut out = Vec::with_capacity(views.len());
        for (a, b) in self.batches(views, level) {
            let mut items = Vec::new();
            let mut offs = Vec::new();
            let (mut off, mut mw, mut mh) = (0usize, 0usize, 0usize);
            for (&v, st) in views[a..b].iter().zip(&states[a..b]) {
                items.push(BatchItem { off: off as i64, view: v as i32, w: st.width as i32, h: st.height as i32, pad: 0 });
                offs.push(off);
                off += st.width * st.height;
                mw = mw.max(st.width);
                mh = mh.max(st.height);
            }
            self.ensure_capacity(off)?;
            let refs: Vec<&ViewState> = states[a..b].iter().collect();
            self.upload_planes_to_prior(&refs, &offs, off)?;
            self.set_batch(&items)?;
            let k = self.params(level, false);
            let f = self.f.median.clone();
            self.launch_simple(&f, &k, None, mw, mh, b - a)?;
            let mut res = self.download_planes(&items, off)?;
            for (r, st) in res.iter_mut().zip(&states[a..b]) {
                r.cost = st.cost.clone();
            }
            out.extend(res);
        }
        Ok(out)
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        let _ = self.be.dev.stream.synchronize();
        let _ = self.img_shift;
        for &t in &self.texs {
            // SAFETY: 이 세션이 만든 텍스처 객체.
            unsafe {
                let _ = sys::cuTexObjectDestroy(t);
            }
        }
        for &a in &self.arrays {
            // SAFETY: 이 세션이 만든 배열(텍스처를 먼저 해제했다).
            unsafe {
                let _ = sys::cuArrayDestroy(a);
            }
        }
    }
}

/// 커널 컴파일 정의(창 반경·간격·하드웨어 보간·블록 수).
pub(crate) fn defines(wr: i32, step: i32, hw_interp: bool, min_blocks: u32) -> Vec<String> {
    vec![
        format!("WR={wr}"),
        format!("WSTEP={step}"),
        format!("HW_INTERP={}", hw_interp as i32),
        format!("MIN_BLOCKS={}", min_blocks.max(1)),
    ]
}
