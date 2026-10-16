/*
 * sift.rs
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

//! CUDA implementation of `SiftEngine`. The whole SIFT runs on the GPU.
//!
//! 1. Gaussian scale space (normalization, 2x upsampling, separable blur, downscaling): bit-identical to the CPU pyramid.
//! 2. DoG, extremum detection + edge and contrast suppression + sub-pixel refinement: one thread per pixel, candidates are
//!    appended atomically.
//! 3. Orientation assignment (including multiple orientations per keypoint): one warp per candidate.
//! 4. Maximum feature count selection (`FeatureSelection`): only the per-level counts are downloaded to plan with the same
//!    rules as the CPU, and the rank within a level (TopK) or within a grid cell (SpatialGrid) is counted on the GPU.
//! 5. 128-dimensional descriptors (normalize -> clip at 0.2 -> renormalize -> L1-ROOT or L2 -> ×512 quantization): one warp
//!    per kept feature.
//!
//! Only the per-level counts (small arrays) and the final features and descriptors are downloaded. The formulas and
//! parameters, coordinate rules (octave, level, pixel center 0.5) and output order (octave -> level -> detection pixel ->
//! orientation slot) are the same as `CpuSift`. However, the histogram summation order and the transcendental function
//! implementations (exp, atan2, sin/cos, pow) differ, so the result is not bit-identical. Positions, orientations and
//! descriptors are practically the same and the feature count is nearly the same (see the tests). The summation order is
//! fixed, so the same input gives the same result on every run.

use crate::device::{CudaDevice, GpuError, SyncedCopy};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, DriverError, LaunchConfig, PushKernelArg};
use cumulus3d_core::{Descriptors, Error, Result};
use cumulus3d_features::sift::pyramid::{self, ScaleSpace};
use cumulus3d_features::{CpuSift, DescriptorNormalization, FeatureSelection, GrayImage, SiftEngine, SiftFeature, SiftOptions, SiftOutput};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub(crate) const SRC: &str = include_str!("kernels/sift.cu");
const TILE_W: u32 = 128;
/// Number of orientation histogram bins (ORI_NB in sift.cu).
const ORI_NB: usize = 36;
/// Warps per block of the orientation kernel (ORI_WARPS in sift.cu).
const ORI_WARPS: u32 = 4;
/// Warps per block of the descriptor kernel (DESC_WARPS in sift.cu).
const DESC_WARPS: u32 = 2;
/// Block size of the rank kernel (RANK_T in sift.cu).
const RANK_T: u32 = 256;
/// Initial capacity of the candidate buffer.
const MIN_CAND_CAP: usize = 1 << 14;

/// CUDA SIFT backend.
pub struct CudaSift {
    dev: Arc<CudaDevice>,
    f_norm: CudaFunction,
    f_up: CudaFunction,
    f_bh: CudaFunction,
    f_bv: CudaFunction,
    f_dec: CudaFunction,
    f_dog: CudaFunction,
    f_detect: CudaFunction,
    f_orient: CudaFunction,
    f_group: CudaFunction,
    f_rank: CudaFunction,
    f_keep: CudaFunction,
    f_desc: CudaFunction,
    /// Device use serialization (stream shared) + state carried over between images.
    lock: Mutex<State>,
}

/// State carried over between images.
#[derive(Default)]
struct State {
    /// Candidate buffer capacity that sufficed last time (remembered because an overflow means detecting again).
    cand_cap: usize,
}

/// Octave on the device.
struct DevOctave {
    o: i32,
    w: usize,
    h: usize,
    gauss: Vec<CudaSlice<f32>>,
}

fn cfg2(w: usize, h: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((w as u32).div_ceil(32), (h as u32).div_ceil(8), 1), block_dim: (32, 8, 1), shared_mem_bytes: 0 }
}

fn cfg1(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

/// Separates input errors (returned as is) from device errors (computed on the CPU instead).
enum Fail {
    Input(Error),
    Device(GpuError),
}

impl From<GpuError> for Fail {
    fn from(e: GpuError) -> Self {
        Fail::Device(e)
    }
}

impl From<DriverError> for Fail {
    fn from(e: DriverError) -> Self {
        Fail::Device(GpuError::from(e))
    }
}

/// Maximum feature count selection plan. Level index = octave index · S + j (octave index 0 is the finest).
#[derive(Clone, Debug, PartialEq)]
struct Plan {
    /// Per-level state: 0 dropped (including not processed), 1 kept whole, 2 picked by rank.
    state: Vec<i32>,
    /// Rank selection mode: 0 none, 1 TopK (rank < `limit` in state-2 levels), 2 SpatialGrid (round robin over grid cells).
    mode: i32,
    /// Number to keep in the TopK boundary level.
    limit: usize,
    /// Number of remaining features.
    n_out: usize,
}

/// Reproduces the level processing order and early stop of `CpuSift::extract` and the `select` rules from per-level counts
/// only. `det[l]` = number of detected candidates at level l, `items[l]` = number of features with orientations attached.
fn plan_selection(det: &[usize], items: &[usize], n_oct: usize, s: usize, mode: FeatureSelection, k: usize) -> Plan {
    let nl = n_oct * s;
    // From coarse octaves and high levels first. The CPU skips detection of levels that get cut, so stop under the same condition.
    let mut order: Vec<usize> = Vec::with_capacity(nl);
    let (mut cum_det, mut cum_or) = (0usize, 0usize);
    'outer: for oi in (0..n_oct).rev() {
        for j in (0..s).rev() {
            let stop = match mode {
                FeatureSelection::CompatLevels => cum_det > k,
                FeatureSelection::TopK => cum_or >= k,
                FeatureSelection::SpatialGrid { .. } => false,
            };
            if stop {
                break 'outer;
            }
            let l = oi * s + j;
            cum_det += det[l];
            cum_or += items[l];
            order.push(l);
        }
    }
    let mut state = vec![0i32; nl];
    for &l in &order {
        state[l] = 1;
    }
    let total: usize = order.iter().map(|&l| items[l]).sum();
    let mut plan = Plan { state, mode: 0, limit: 0, n_out: total };
    if total <= k {
        return plan;
    }
    match mode {
        FeatureSelection::CompatLevels => {
            // From the finest level, remove whole levels while "total - that level > K".
            let mut t = total;
            for &l in order.iter().rev() {
                if t - items[l] > k {
                    t -= items[l];
                    plan.state[l] = 0;
                } else {
                    break;
                }
            }
            plan.n_out = t;
        }
        FeatureSelection::TopK => {
            // Coarse levels whole first, the first overflowing level by |response| for the remaining count, then drop the rest.
            let mut rem = k;
            for &l in &order {
                if items[l] <= rem {
                    rem -= items[l];
                } else {
                    if rem > 0 {
                        plan.state[l] = 2;
                        plan.mode = 1;
                        plan.limit = rem;
                    } else {
                        plan.state[l] = 0;
                    }
                    rem = 0;
                }
            }
            plan.n_out = k;
        }
        FeatureSelection::SpatialGrid { .. } => {
            for &l in &order {
                plan.state[l] = 2;
            }
            plan.mode = 2;
            plan.n_out = k;
        }
    }
    plan
}

/// Boundary of the grid-cell round robin (in cell order, the next rank from each cell, stopping at K items).
/// The item of rank r in cell b is kept when `r < rcut || (r == rcut && b < bcut)`. `sizes[b]` = number of items in cell b.
fn grid_cut(sizes: &[usize], k: usize) -> (usize, usize) {
    let mut sorted = sizes.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let (mut taken, mut r, mut p) = (0usize, 0usize, 0usize);
    loop {
        while p < n && sorted[p] <= r {
            p += 1;
        }
        let avail = n - p;
        if avail == 0 {
            return (r, n);
        }
        if taken + avail <= k {
            taken += avail;
            r += 1;
            if taken == k {
                return (r, 0);
            }
            continue;
        }
        // This round covers only the first `k - taken` cells.
        let left = k - taken;
        let mut seen = 0usize;
        for (b, &sz) in sizes.iter().enumerate() {
            if sz > r {
                if seen == left {
                    return (r, b);
                }
                seen += 1;
            }
        }
        return (r, sizes.len());
    }
}

impl CudaSift {
    /// Compiles the SIFT kernels on device `dev` and builds it.
    pub fn new(dev: Arc<CudaDevice>) -> std::result::Result<Self, GpuError> {
        // Turn off fused multiply-add and use IEEE division: the pyramid is bit-identical to the CPU, and detection rounds the
        // same way as the CPU formulas.
        let m = dev.compile(SRC, false, &[])?;
        Ok(Self {
            f_norm: m.load_function("sift_normalize")?,
            f_up: m.load_function("sift_upsample2")?,
            f_bh: m.load_function("sift_blur_h")?,
            f_bv: m.load_function("sift_blur_v")?,
            f_dec: m.load_function("sift_decimate2")?,
            f_dog: m.load_function("sift_dog")?,
            f_detect: m.load_function("sift_detect")?,
            f_orient: m.load_function("sift_orient")?,
            f_group: m.load_function("sift_group")?,
            f_rank: m.load_function("sift_rank")?,
            f_keep: m.load_function("sift_keep")?,
            f_desc: m.load_function("sift_descriptor")?,
            dev,
            lock: Mutex::new(State::default()),
        })
    }

    /// Builds with device 0.
    pub fn try_default() -> std::result::Result<Self, GpuError> {
        Self::new(CudaDevice::new(0)?)
    }

    fn blur(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        tmp: &mut CudaSlice<f32>,
        w: usize,
        h: usize,
        sigma: f32,
    ) -> std::result::Result<(), GpuError> {
        let s = &self.dev.stream;
        let k = pyramid::gaussian_kernel(sigma);
        let r = (k.len() / 2) as i32;
        let kd = s.clone_htod(&k)?;
        let (wi, hi) = (w as i32, h as i32);
        {
            let cfg = LaunchConfig { grid_dim: ((w as u32).div_ceil(TILE_W), h as u32, 1), block_dim: (TILE_W, 1, 1), shared_mem_bytes: 0 };
            let mut l = s.launch_builder(&self.f_bh);
            l.arg(src).arg(&mut *tmp).arg(&wi).arg(&hi).arg(&kd).arg(&r);
            // SAFETY: sift_blur_h(src[w·h], tmp[w·h], w, h, k[2r+1], r), r ≤ 16.
            unsafe { l.launch(cfg) }?;
        }
        let mut l = s.launch_builder(&self.f_bv);
        l.arg(&*tmp).arg(dst).arg(&wi).arg(&hi).arg(&kd).arg(&r);
        // SAFETY: same type as sift_blur_v.
        unsafe { l.launch(cfg2(w, h)) }?;
        Ok(())
    }

    /// GPU pyramid (same rules as CPU `pyramid::build_pyramid`).
    fn build(
        &self,
        s: &Arc<CudaStream>,
        image: &GrayImage,
        w: usize,
        h: usize,
        opts: &SiftOptions,
        ss: ScaleSpace,
    ) -> std::result::Result<Vec<DevOctave>, GpuError> {
        let src = s.clone_htod(&image.data)?;
        let mut base = s.alloc_zeros::<f32>(w * h)?;
        {
            let (sw, wi, hi) = (image.width as i32, w as i32, h as i32);
            let mut l = s.launch_builder(&self.f_norm);
            l.arg(&src).arg(&sw).arg(&wi).arg(&hi).arg(&mut base);
            // SAFETY: sift_normalize(src[sw·h], sw, w, h, dst[w·h]).
            unsafe { l.launch(cfg2(w, h)) }?;
        }
        let o_min = opts.first_octave;
        let (w0, h0, src0) = if o_min == -1 {
            let mut up = s.alloc_zeros::<f32>(4 * w * h)?;
            let (wi, hi) = (w as i32, h as i32);
            let mut l = s.launch_builder(&self.f_up);
            l.arg(&base).arg(&wi).arg(&hi).arg(&mut up);
            // SAFETY: sift_upsample2(src[w·h], w, h, dst[4·w·h]).
            unsafe { l.launch(cfg2(w, 2 * h)) }?;
            (2 * w, 2 * h, up)
        } else {
            (w, h, base)
        };
        let n_oct = opts.num_octaves.unwrap_or_else(|| pyramid::auto_num_octaves(w0, h0)).max(1);
        let mut tmp = s.alloc_zeros::<f32>(w0 * h0)?;
        let mut octs: Vec<DevOctave> = Vec::with_capacity(n_oct);
        let (mut w, mut h) = (w0, h0);
        for oi in 0..n_oct {
            let o = o_min + oi as i32;
            let mut gauss: Vec<CudaSlice<f32>> = Vec::with_capacity(ss.s + 3);
            if oi == 0 {
                let s_first = ss.sigma(-1.0);
                let sn = 0.5 / 2f32.powi(o_min);
                if s_first > sn {
                    let mut g0 = s.alloc_zeros::<f32>(w * h)?;
                    self.blur(&src0, &mut g0, &mut tmp, w, h, (s_first * s_first - sn * sn).sqrt())?;
                    gauss.push(g0);
                } else {
                    gauss.push(s.clone_dtod(&src0)?);
                }
            } else {
                let prev = &octs[oi - 1];
                let (pw, ph) = (prev.w, prev.h);
                let (nw, nh) = (pw / 2, ph / 2);
                let mut g0 = s.alloc_zeros::<f32>(nw * nh)?;
                let (pwi, nwi, nhi) = (pw as i32, nw as i32, nh as i32);
                let mut l = s.launch_builder(&self.f_dec);
                // Previous octave level S−1 (gauss index S).
                l.arg(&prev.gauss[ss.s]).arg(&pwi).arg(&mut g0).arg(&nwi).arg(&nhi);
                // SAFETY: sift_decimate2(src[pw·ph], pw, dst[nw·nh], nw, nh).
                unsafe { l.launch(cfg2(nw, nh)) }?;
                w = nw;
                h = nh;
                gauss.push(g0);
            }
            for l in 0..=(ss.s as i32 + 1) {
                let mut g = s.alloc_zeros::<f32>(w * h)?;
                let last = gauss.last().expect("레벨 있음");
                self.blur(last, &mut g, &mut tmp, w, h, ss.sigma_inc(l))?;
                gauss.push(g);
            }
            octs.push(DevOctave { o, w, h, gauss });
            if w / 2 < 4 || h / 2 < 4 {
                break;
            }
        }
        Ok(octs)
    }

    fn extract_gpu(&self, image: &GrayImage, opts: &SiftOptions) -> std::result::Result<SiftOutput, Fail> {
        if opts.octave_resolution == 0 {
            return Err(Fail::Input(Error::InvalidArgument("octave_resolution 은 1 이상".into())));
        }
        if opts.first_octave != -1 && opts.first_octave != 0 {
            return Err(Fail::Input(Error::Unsupported(format!("first_octave {} (−1, 0 만 지원)", opts.first_octave))));
        }
        let w = if opts.truncate_width_to_4 { image.width - image.width % 4 } else { image.width };
        let h = image.height;
        if w < 8 || h < 8 {
            return Ok(SiftOutput::default());
        }
        let ss = ScaleSpace::new(opts.octave_resolution);
        let mut state = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let st = self.dev.stream.clone();
        let t0 = Instant::now();

        // 1. Scale space.
        let pyr = self.build(&st, image, w, h, opts, ss)?;
        let s = ss.s;
        let n_oct = pyr.len();
        let nl = n_oct * s;
        // Level table: device address of the Gaussian (gauss[j + 1]) of level l = oi·S + j, and (w, h, octave, j).
        // These are raw addresses, so `pyr` is released only after the last synchronization.
        let mut lv_ptr: Vec<u64> = Vec::with_capacity(nl);
        let mut lv_i: Vec<i32> = Vec::with_capacity(4 * nl);
        for oct in &pyr {
            for j in 0..s {
                lv_ptr.push(oct.gauss[j + 1].device_ptr(&st).0);
                lv_i.extend_from_slice(&[oct.w as i32, oct.h as i32, oct.o, j as i32]);
            }
        }
        let lv_ptr_d = st.clone_htod(&lv_ptr)?;
        let lv_i_d = st.clone_htod(&lv_i)?;

        // 2. DoG + detection (all levels). If the candidates exceed the capacity, grow it and redo.
        let (w0, h0) = (pyr[0].w, pyr[0].h);
        let mut dog = st.alloc_zeros::<f32>((s + 2) * w0 * h0)?;
        let iters = opts.refinement_iterations.max(1) as i32;
        let reject = opts.reject_singular_refinement as i32;
        let mut cap = state.cand_cap.max(MIN_CAND_CAP);
        let (cf, ci, counts) = loop {
            let mut cf = st.alloc_zeros::<f32>(4 * cap)?;
            let mut ci = st.alloc_zeros::<i32>(2 * cap)?;
            let mut cnt = st.alloc_zeros::<i32>(1 + nl)?;
            for (oi, oct) in pyr.iter().enumerate() {
                let n = oct.w * oct.h;
                let ni = n as i32;
                for d in 0..s + 2 {
                    let di = d as i32;
                    let mut l = st.launch_builder(&self.f_dog);
                    l.arg(&oct.gauss[d + 1]).arg(&oct.gauss[d]).arg(&mut dog).arg(&di).arg(&ni);
                    // SAFETY: sift_dog(hi[n], lo[n], dog[at least (S+2)·n], d < S+2, n).
                    unsafe { l.launch(cfg1(n)) }?;
                }
                if oct.w < 3 || oct.h < 3 {
                    continue;
                }
                let cfg = LaunchConfig {
                    grid_dim: ((oct.w as u32).div_ceil(32), (oct.h as u32).div_ceil(8), s as u32),
                    block_dim: (32, 8, 1),
                    shared_mem_bytes: 0,
                };
                let (wi, hi, base, capi) = (oct.w as i32, oct.h as i32, (oi * s) as i32, cap as i32);
                let mut l = st.launch_builder(&self.f_detect);
                l.arg(&dog)
                    .arg(&wi)
                    .arg(&hi)
                    .arg(&base)
                    .arg(&opts.peak_threshold)
                    .arg(&opts.edge_threshold)
                    .arg(&iters)
                    .arg(&reject)
                    .arg(&ss.sigma0)
                    .arg(&ss.k)
                    .arg(&mut cf)
                    .arg(&mut ci)
                    .arg(&capi)
                    .arg(&mut cnt);
                // SAFETY: sift_detect(dog[(S+2)·w·h], w, h, level base, …, cf[4·cap], ci[2·cap], cap, counters[1 + level count]),
                // grid z = S and level index base + j < level count.
                unsafe { l.launch(cfg) }?;
            }
            let counts = st.clone_dtoh_synced(&cnt)?;
            st.synchronize()?;
            let total = counts[0].max(0) as usize;
            if total <= cap {
                break (cf, ci, counts);
            }
            cap = total.next_power_of_two();
        };
        state.cand_cap = cap;
        drop(dog);
        let ncand = counts[0].max(0) as usize;
        let det: Vec<usize> = counts[1..].iter().map(|&c| c.max(0) as usize).collect();
        let t_detect = t0.elapsed();
        if ncand == 0 {
            return Ok(SiftOutput::default());
        }

        // 3. Orientation assignment.
        let per = if opts.upright { 1 } else { opts.max_num_orientations.clamp(1, ORI_NB) };
        let icap = ncand * per;
        let mut items = st.alloc_zeros::<i32>(3 * icap)?;
        let mut icnt = st.alloc_zeros::<i32>(1 + nl)?;
        {
            let (nc, mo, interp, up, ic) =
                (ncand as i32, per as i32, opts.orientation_bin_interpolation as i32, opts.upright as i32, icap as i32);
            let cfg = LaunchConfig {
                grid_dim: ((ncand as u32).div_ceil(ORI_WARPS), 1, 1),
                block_dim: (32 * ORI_WARPS, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = st.launch_builder(&self.f_orient);
            l.arg(&cf).arg(&ci).arg(&nc).arg(&lv_ptr_d).arg(&lv_i_d).arg(&mo).arg(&interp).arg(&up).arg(&mut items).arg(&ic).arg(&mut icnt);
            // SAFETY: sift_orient(cf[4·ncand], ci[2·ncand], ncand, lv_ptr[level count], lv_i[4·level count], …,
            // items[3·icap], icap, counters[1 + level count]). One candidate emits at most `per` items, so icap is not exceeded.
            unsafe { l.launch(cfg) }?;
        }
        let icounts = st.clone_dtoh_synced(&icnt)?;
        st.synchronize()?;
        let nitems = (icounts[0].max(0) as usize).min(icap);
        let lv_items: Vec<usize> = icounts[1..].iter().map(|&c| c.max(0) as usize).collect();
        let t_orient = t0.elapsed();

        // 4. Maximum feature count selection.
        let k_max = if opts.max_num_features == 0 { usize::MAX } else { opts.max_num_features };
        let plan = plan_selection(&det, &lv_items, n_oct, s, opts.selection, k_max);
        let n_out = plan.n_out.min(nitems);
        if n_out == 0 {
            return Ok(SiftOutput::default());
        }
        let state_d = st.clone_htod(&plan.state)?;
        let ni = nitems as i32;
        let mut group = st.alloc_zeros::<i32>(nitems)?;
        let mut rank = st.alloc_zeros::<i32>(nitems)?;
        let (mut rcut, mut bcut) = (0i32, 0i32);
        if plan.mode != 0 {
            let cells = match opts.selection {
                FeatureSelection::SpatialGrid { cells } => cells.max(1) as usize,
                _ => 1,
            };
            let mut bcount = st.alloc_zeros::<i32>(cells * cells)?;
            // Input image size ~ first octave size x 2^o_min (same formula as the CPU).
            let sc = 2f32.powi(pyr[0].o);
            let (iw, ih) = (pyr[0].w as f32 * sc, pyr[0].h as f32 * sc);
            let (gridi, cellsi) = ((plan.mode == 2) as i32, cells as i32);
            {
                let mut l = st.launch_builder(&self.f_group);
                l.arg(&items)
                    .arg(&ni)
                    .arg(&cf)
                    .arg(&ci)
                    .arg(&lv_i_d)
                    .arg(&state_d)
                    .arg(&gridi)
                    .arg(&cellsi)
                    .arg(&iw)
                    .arg(&ih)
                    .arg(&mut group)
                    .arg(&mut bcount);
                // SAFETY: sift_group(items[3·n], n, cf, ci, lv_i, lv_state[level count], …, group[n], bucket_count[cells²]).
                unsafe { l.launch(cfg1(nitems)) }?;
            }
            if plan.mode == 2 {
                let raw = st.clone_dtoh_synced(&bcount)?;
                st.synchronize()?;
                let sizes: Vec<usize> = raw.iter().map(|&c| c.max(0) as usize).collect();
                let (r, b) = grid_cut(&sizes, k_max);
                rcut = r as i32;
                bcut = b as i32;
            }
            let cfg = LaunchConfig { grid_dim: ((nitems as u32).div_ceil(RANK_T), 1, 1), block_dim: (RANK_T, 1, 1), shared_mem_bytes: 0 };
            let mut l = st.launch_builder(&self.f_rank);
            l.arg(&items).arg(&ni).arg(&cf).arg(&ci).arg(&group).arg(&mut rank);
            // SAFETY: sift_rank(items[3·n], n, cf, ci, group[n], rank[n]), block size = RANK_T.
            unsafe { l.launch(cfg) }?;
        }
        let mut keep = st.alloc_zeros::<i32>(n_out)?;
        let mut kcnt = st.alloc_zeros::<i32>(1)?;
        let kcap = n_out as i32;
        {
            let limit = plan.limit.min(i32::MAX as usize) as i32;
            let mut l = st.launch_builder(&self.f_keep);
            l.arg(&items)
                .arg(&ni)
                .arg(&ci)
                .arg(&state_d)
                .arg(&group)
                .arg(&rank)
                .arg(&plan.mode)
                .arg(&limit)
                .arg(&rcut)
                .arg(&bcut)
                .arg(&mut keep)
                .arg(&kcap)
                .arg(&mut kcnt);
            // SAFETY: sift_keep(items[3·n], n, ci, lv_state, group[n], rank[n], …, keep[cap], cap, counter[1]).
            unsafe { l.launch(cfg1(nitems)) }?;
        }

        // 5. Descriptors + output features.
        let mut fo = st.alloc_zeros::<f32>(5 * n_out)?;
        let mut io = st.alloc_zeros::<i32>(4 * n_out)?;
        let mut dd = st.alloc_zeros::<u8>(128 * n_out)?;
        {
            let l1 = (opts.normalization == DescriptorNormalization::L1Root) as i32;
            let cfg = LaunchConfig {
                grid_dim: ((n_out as u32).div_ceil(DESC_WARPS), 1, 1),
                block_dim: (32 * DESC_WARPS, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = st.launch_builder(&self.f_desc);
            l.arg(&keep)
                .arg(&kcnt)
                .arg(&kcap)
                .arg(&items)
                .arg(&cf)
                .arg(&ci)
                .arg(&lv_ptr_d)
                .arg(&lv_i_d)
                .arg(&l1)
                .arg(&mut fo)
                .arg(&mut io)
                .arg(&mut dd);
            // SAFETY: sift_descriptor(keep[cap], kcount[1], cap, items, cf, ci, lv_ptr, lv_i, l1root,
            // fo[5·cap], io[4·cap], desc[128·cap]); processed count min(*kcount, cap), shared memory is static (32 KiB).
            unsafe { l.launch(cfg) }?;
        }
        let kc = st.clone_dtoh_synced(&kcnt)?;
        let fo_h = st.clone_dtoh_synced(&fo)?;
        let io_h = st.clone_dtoh_synced(&io)?;
        let dd_h = st.clone_dtoh_synced(&dd)?;
        st.synchronize()?;
        // Release here the pyramid that the raw addresses of the level table pointed to.
        drop(pyr);
        let m = (kc[0].max(0) as usize).min(n_out);

        // Same output order as CpuSift: octave ascending, level ascending, detection pixel order, orientation slot.
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_unstable_by_key(|&i| (io_h[4 * i], io_h[4 * i + 1], io_h[4 * i + 2], io_h[4 * i + 3]));
        let mut features = Vec::with_capacity(m);
        let mut desc = Descriptors::with_capacity(m);
        for &i in &order {
            features.push(SiftFeature {
                x: fo_h[5 * i],
                y: fo_h[5 * i + 1],
                scale: fo_h[5 * i + 2],
                orientation: fo_h[5 * i + 3],
                octave: io_h[4 * i],
                level: io_h[4 * i + 1] as u32,
                response: fo_h[5 * i + 4],
            });
            let row: &[u8; 128] = dd_h[128 * i..128 * (i + 1)].try_into().expect("기술자 128바이트");
            desc.push(row);
        }
        if std::env::var_os("CUMULUS3D_CUDA_TRACE").is_some() {
            eprintln!(
                "cuda sift: 피라미드·검출 {t_detect:.2?} (후보 {ncand}), 방향 {:.2?} (항목 {nitems}), 선택·기술자 {:.2?} (특징 {m})",
                t_orient - t_detect,
                t0.elapsed() - t_orient
            );
        }
        Ok(SiftOutput { features, descriptors: desc })
    }
}

impl SiftEngine for CudaSift {
    fn name(&self) -> &str {
        "cuda"
    }
    fn extract(&self, image: &GrayImage, options: &SiftOptions) -> Result<SiftOutput> {
        match self.extract_gpu(image, options) {
            Ok(o) => Ok(o),
            Err(Fail::Input(e)) => Err(e),
            Err(Fail::Device(e)) => {
                eprintln!("경고: CUDA SIFT 실패({e}), CPU SIFT 로 계산");
                CpuSift::new().extract(image, options)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- Selection plan (without a device) -----

    /// Reference from running the cell-order round robin literally: number kept per cell.
    fn round_robin(sizes: &[usize], k: usize) -> Vec<usize> {
        let mut kept = vec![0usize; sizes.len()];
        let (mut taken, mut r) = (0usize, 0usize);
        while taken < k {
            let mut any = false;
            for (b, &sz) in sizes.iter().enumerate() {
                if sz > r {
                    any = true;
                    if taken < k {
                        kept[b] += 1;
                        taken += 1;
                    }
                }
            }
            if !any {
                break;
            }
            r += 1;
        }
        kept
    }

    #[test]
    fn grid_cut_matches_round_robin() {
        let cases: [&[usize]; 6] = [&[5, 0, 3, 8, 1], &[1, 1, 1, 1], &[10], &[0, 0, 4], &[2, 7, 7, 2, 9, 0, 3], &[3, 3, 3]];
        for sizes in cases {
            let total: usize = sizes.iter().sum();
            for k in 0..=total + 1 {
                let (r, b) = grid_cut(sizes, k);
                let got: Vec<usize> =
                    sizes.iter().enumerate().map(|(bi, &sz)| (0..sz).filter(|&rk| rk < r || (rk == r && bi < b)).count()).collect();
                assert_eq!(got, round_robin(sizes, k), "{sizes:?} k={k}");
            }
        }
    }

    #[test]
    fn plan_follows_cpu_rules() {
        // 2 octaves × S = 2 -> levels 0..4, coarse order 3, 2, 1, 0.
        let det = [10, 10, 10, 10];
        let items = [12, 11, 10, 9];
        // CompatLevels: levels 3, 2, 1 processed (stops at cumulative detections 30 > 25), 30 - 11 > 25 does not hold, so
        // nothing is removed.
        let p = plan_selection(&det, &items, 2, 2, FeatureSelection::CompatLevels, 25);
        assert_eq!(p, Plan { state: vec![0, 1, 1, 1], mode: 0, limit: 0, n_out: 30 });
        // K = 15: level 3 processed since it is within 15 (cumulative detections 10), level 2 processed (20 > 15), stop at
        // level 1. 19 - 10 > 15 does not hold.
        let p = plan_selection(&det, &items, 2, 2, FeatureSelection::CompatLevels, 15);
        assert_eq!(p.state, vec![0, 0, 1, 1]);
        assert_eq!(p.n_out, 19);
        // K = 5: levels 3, 2 processed (cumulative detections 20 > 5 only after level 2), 19 - 10 = 9 > 5, so level 2 is
        // removed.
        let p = plan_selection(&det, &items, 2, 2, FeatureSelection::CompatLevels, 5);
        assert_eq!(p.state, vec![0, 0, 0, 1]);
        assert_eq!(p.n_out, 9);
        // TopK: level 3 (9), level 2 (cumulative 9 < 15 -> processed), stop at level 1 (19 ≥ 15). Only 6 from level 2.
        let p = plan_selection(&det, &items, 2, 2, FeatureSelection::TopK, 15);
        assert_eq!(p, Plan { state: vec![0, 0, 2, 1], mode: 1, limit: 6, n_out: 15 });
        // SpatialGrid: all processed, all selected by rank.
        let p = plan_selection(&det, &items, 2, 2, FeatureSelection::SpatialGrid { cells: 4 }, 15);
        assert_eq!(p, Plan { state: vec![2, 2, 2, 2], mode: 2, limit: 0, n_out: 15 });
        // Within the cap, everything is kept.
        let p = plan_selection(&det, &items, 2, 2, FeatureSelection::TopK, 100);
        assert_eq!(p, Plan { state: vec![1, 1, 1, 1], mode: 0, limit: 0, n_out: 42 });
    }

    // ----- GPU vs CPU (skipped without a device) -----

    fn gpu() -> Option<CudaSift> {
        if !crate::is_available() {
            eprintln!("CUDA 장치 없음: 건너뜀");
            return None;
        }
        Some(CudaSift::try_default().expect("CUDA SIFT"))
    }

    /// Gaussian blobs (center x, center y, σ, amplitude) (deterministic).
    fn blobs(seed: u64, n: u64, w: f64, h: f64) -> Vec<(f64, f64, f64, f64)> {
        (0..n)
            .map(|i| {
                let mut z = seed.wrapping_add(i).wrapping_mul(0x9e3779b97f4a7c15);
                let mut r = || {
                    z ^= z >> 29;
                    z = z.wrapping_mul(0xbf58476d1ce4e5b9);
                    z ^= z >> 32;
                    (z >> 11) as f64 / (1u64 << 53) as f64
                };
                (r() * w, r() * h, 2.0 + 14.0 * r(), r() - 0.5)
            })
            .collect()
    }

    /// `base` + sum of blobs at the center (i + 0.5, j + 0.5) of pixel (i, j), clamped to [0, 1] and converted to 8 bits.
    fn render(w: usize, h: usize, blobs: &[(f64, f64, f64, f64)], base: impl Fn(f64, f64) -> f64) -> GrayImage {
        let mut acc: Vec<f64> = (0..w * h).map(|i| base((i % w) as f64 + 0.5, (i / w) as f64 + 0.5)).collect();
        for &(cx, cy, s, a) in blobs {
            let r = 3.0 * s;
            let x0 = ((cx - r - 0.5).floor().max(0.0)) as usize;
            let x1 = ((cx + r - 0.5).ceil().min(w as f64 - 1.0)).max(-1.0);
            let y0 = ((cy - r - 0.5).floor().max(0.0)) as usize;
            let y1 = ((cy + r - 0.5).ceil().min(h as f64 - 1.0)).max(-1.0);
            if x1 < 0.0 || y1 < 0.0 {
                continue;
            }
            for y in y0..=y1 as usize {
                for x in x0..=x1 as usize {
                    let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
                    acc[y * w + x] += a * (-(dx * dx + dy * dy) / (2.0 * s * s)).exp();
                }
            }
        }
        GrayImage { width: w, height: h, data: acc.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect() }
    }

    fn angle_diff(a: f32, b: f32) -> f32 {
        let d = (a - b).rem_euclid(std::f32::consts::TAU);
        d.min(std::f32::consts::TAU - d)
    }

    fn desc_dist(a: &[u8], b: &[u8]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| (x as f32 - y as f32).powi(2)).sum::<f32>().sqrt()
    }

    /// Whether most GPU features have a CPU feature in the same octave, within 1 pixel, with a similar orientation, and with a
    /// close descriptor.
    fn assert_equivalent(cpu: &SiftOutput, gpu: &SiftOutput, label: &str) {
        let (nc, ng) = (cpu.len(), gpu.len());
        eprintln!("{label}: CPU {nc} GPU {ng} 특징");
        assert!(nc > 20, "{label}: CPU 특징이 너무 적음({nc})");
        let ratio = ng as f64 / nc as f64;
        assert!((0.9..=1.1).contains(&ratio), "{label}: 특징 수 CPU {nc}, GPU {ng}");
        assert_eq!(gpu.descriptors.len(), ng);
        let mut dists = Vec::new();
        for (gi, g) in gpu.features.iter().enumerate() {
            let mut best: Option<(f32, usize)> = None;
            for (ci, c) in cpu.features.iter().enumerate() {
                if c.octave != g.octave {
                    continue;
                }
                let (dx, dy) = (c.x - g.x, c.y - g.y);
                if dx * dx + dy * dy > 1.0 {
                    continue;
                }
                let da = angle_diff(c.orientation, g.orientation);
                if best.is_none_or(|(b, _)| da < b) {
                    best = Some((da, ci));
                }
            }
            if let Some((da, ci)) = best {
                if da < 0.1 {
                    dists.push(desc_dist(gpu.descriptors.row(gi), cpu.descriptors.row(ci)));
                }
            }
        }
        let frac = dists.len() as f64 / ng.max(1) as f64;
        assert!(frac >= 0.9, "{label}: CPU 짝이 있는 GPU 특징 비율 {frac:.3}");
        dists.sort_by(f32::total_cmp);
        let (med, p90) = (dists[dists.len() / 2], dists[dists.len() * 9 / 10]);
        eprintln!("{label}: 짝 비율 {frac:.3}, 기술자 거리 중앙값 {med:.2}, 90% {p90:.2} (기술자 노름 약 512)");
        assert!(med < 10.0 && p90 < 60.0, "{label}: 기술자 거리 중앙값 {med}, 90% {p90}");
    }

    #[test]
    fn gpu_sift_equivalent_to_cpu() {
        let Some(gpu) = gpu() else { return };
        let cpu = CpuSift::new();
        let blob_img = |w: usize, h: usize, seed: u64| render(w, h, &blobs(seed, 400, w as f64, h as f64), |_, _| 0.5);
        // Checkerboard rotated by 17° (period 24 pixels, smoothed with tanh) + blobs.
        let checker = |w: usize, h: usize| {
            let (c, s) = (0.3f64.cos(), 0.3f64.sin());
            let k = std::f64::consts::PI / 24.0;
            render(w, h, &blobs(99, 60, w as f64, h as f64), move |x, y| {
                let (u, v) = ((c * x + s * y) * k, (-s * x + c * y) * k);
                0.5 + 0.3 * (3.0 * u.sin() * v.sin()).tanh()
            })
        };
        let cases: Vec<(&str, GrayImage, SiftOptions)> = vec![
            ("얼룩 643×417 기본", blob_img(643, 417, 1), SiftOptions::default()),
            ("얼룩 512×384 첫 옥타브 0", blob_img(512, 384, 2), SiftOptions { first_octave: 0, ..SiftOptions::default() }),
            ("돌린 체커보드 640×480", checker(640, 480), SiftOptions::default()),
            (
                "얼룩 640×480 L2·보간·반복 보정",
                blob_img(640, 480, 3),
                SiftOptions {
                    normalization: DescriptorNormalization::L2,
                    orientation_bin_interpolation: true,
                    refinement_iterations: 5,
                    ..SiftOptions::default()
                },
            ),
            (
                "얼룩 640×480 TopK 300",
                blob_img(640, 480, 4),
                SiftOptions { max_num_features: 300, selection: FeatureSelection::TopK, ..SiftOptions::default() },
            ),
            ("얼룩 640×480 CompatLevels 400", blob_img(640, 480, 5), SiftOptions { max_num_features: 400, ..SiftOptions::default() }),
            (
                "얼룩 640×480 격자 500 직립",
                blob_img(640, 480, 6),
                SiftOptions {
                    max_num_features: 500,
                    selection: FeatureSelection::SpatialGrid { cells: 4 },
                    upright: true,
                    ..SiftOptions::default()
                },
            ),
        ];
        for (label, img, opts) in &cases {
            let c = cpu.extract(img, opts).unwrap();
            let g = gpu.extract(img, opts).unwrap();
            assert_equivalent(&c, &g, label);
            if opts.max_num_features > 0 && opts.selection != FeatureSelection::CompatLevels {
                assert!(g.len() <= opts.max_num_features, "{label}: 상한 {} 초과 {}", opts.max_num_features, g.len());
            }
            // The summation order is fixed, so running again gives the same result.
            let g2 = gpu.extract(img, opts).unwrap();
            assert_eq!(g.features, g2.features, "{label}: 실행마다 다름");
            assert_eq!(g.descriptors.as_slice(), g2.descriptors.as_slice(), "{label}: 기술자가 실행마다 다름");
        }
    }

    #[test]
    fn gpu_sift_matches_rotated_scaled_copy() {
        let Some(gpu) = gpu() else { return };
        let (w, h) = (640usize, 480usize);
        let b = blobs(7, 500, w as f64, h as f64);
        // T p = s·R(θ)·(p − c) + c. A Gaussian blob is still a blob after a similarity transform, so only the center and σ
        // are moved.
        let (ang, sc) = (0.5f64, 0.8f64);
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        let (cs, sn) = (ang.cos(), ang.sin());
        let fwd = move |x: f64, y: f64| {
            let (dx, dy) = (x - cx, y - cy);
            (sc * (cs * dx - sn * dy) + cx, sc * (sn * dx + cs * dy) + cy)
        };
        let bt: Vec<_> = b
            .iter()
            .map(|&(x, y, s, a)| {
                let (u, v) = fwd(x, y);
                (u, v, s * sc, a)
            })
            .collect();
        let img_a = render(w, h, &b, |_, _| 0.5);
        let img_b = render(w, h, &bt, |_, _| 0.5);
        let fa = gpu.extract(&img_a, &SiftOptions::default()).unwrap();
        let fb = gpu.extract(&img_b, &SiftOptions::default()).unwrap();
        eprintln!("회전·축소 사본: A {} B {} 특징", fa.len(), fb.len());
        assert!(fa.len() > 50 && fb.len() > 50);
        // Nearest neighbor + ratio test (0.8).
        let (mut n_match, mut n_ok) = (0usize, 0usize);
        for (i, f) in fa.features.iter().enumerate() {
            let da = fa.descriptors.row(i);
            let (mut b1, mut b2, mut bi) = (f32::INFINITY, f32::INFINITY, 0usize);
            for k in 0..fb.len() {
                let d = desc_dist(da, fb.descriptors.row(k));
                if d < b1 {
                    b2 = b1;
                    b1 = d;
                    bi = k;
                } else if d < b2 {
                    b2 = d;
                }
            }
            if b1 < 0.8 * b2 {
                n_match += 1;
                let (ex, ey) = fwd(f.x as f64, f.y as f64);
                let g = &fb.features[bi];
                if (ex - g.x as f64).hypot(ey - g.y as f64) < 3.0 {
                    n_ok += 1;
                }
            }
        }
        let inl = n_ok as f64 / n_match.max(1) as f64;
        eprintln!("회전·축소 사본: 매칭 {n_match}, 기하 일치 {n_ok} ({inl:.3})");
        assert!(n_ok >= 40, "기하 일치 매칭이 너무 적음({n_ok})");
        assert!(inl >= 0.8, "기하 일치 비율 {inl:.3}");
    }
}
