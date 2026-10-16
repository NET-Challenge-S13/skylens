/*
 * pyramid.rs
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

//! Gaussian scale space. Separable f32 blur, rayon-parallel per row.

use rayon::prelude::*;
use std::sync::Mutex;

/// f32 buffer reuse pool (reuses pyramid allocations across images).
#[derive(Default)]
pub struct BufferPool {
    bufs: Mutex<Vec<Vec<f32>>>,
}

impl BufferPool {
    /// Buffer of length `len` (contents assumed to be overwritten). Prefers reusing one with matching capacity.
    pub fn take(&self, len: usize) -> Vec<f32> {
        let mut g = self.bufs.lock().unwrap_or_else(|e| e.into_inner());
        let pos = g.iter().enumerate().filter(|(_, b)| b.capacity() >= len).min_by_key(|(_, b)| b.capacity()).map(|(i, _)| i);
        let mut v = match pos {
            Some(i) => g.swap_remove(i),
            None => g.pop().unwrap_or_default(),
        };
        drop(g);
        v.clear();
        v.resize(len, 0.0);
        v
    }

    /// Returns a buffer to the pool.
    pub fn give(&self, v: Vec<f32>) {
        if v.capacity() == 0 {
            return;
        }
        let mut g = self.bufs.lock().unwrap_or_else(|e| e.into_inner());
        // Prevent unbounded growth: discard old ones.
        if g.len() < 64 {
            g.push(v);
        }
    }
}

/// 1D Gaussian kernel: half-width r = ceil(4σ - 0.5), width 2r+1 clamped to [5, 33], normalized to sum 1.
pub fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let r = ((4.0 * sigma - 0.5).ceil() as i64).clamp(2, 16) as usize;
    let mut k: Vec<f32> = (0..=2 * r)
        .map(|i| {
            let d = i as f32 - r as f32;
            (-(d * d) / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k
}

/// Separable Gaussian blur (edge replication). `tmp` is a work buffer of the same size.
#[allow(clippy::needless_range_loop)]
pub fn gaussian_blur(src: &[f32], dst: &mut [f32], tmp: &mut [f32], w: usize, h: usize, sigma: f32) {
    let k = gaussian_kernel(sigma);
    let r = k.len() / 2;
    // Horizontal: for a padded row, out[x] += k_i · pad[x + i] (vectorizable form).
    tmp.par_chunks_mut(w).zip(src.par_chunks(w)).for_each_init(
        || vec![0f32; w + 2 * r],
        |pad, (out, row)| {
            let first = row[0];
            let last = row[w - 1];
            pad[..r].fill(first);
            pad[r..r + w].copy_from_slice(row);
            pad[r + w..].fill(last);
            let kc = k[r];
            for (o, p) in out.iter_mut().zip(&pad[r..r + w]) {
                *o = kc * p;
            }
            for i in 0..r {
                let ki = k[i];
                let a = &pad[i..i + w];
                let b = &pad[2 * r - i..2 * r - i + w];
                for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
                    *o += ki * (x + y);
                }
            }
        },
    );
    // Vertical: out row y = Σ k_i · tmp row clamp(y + i - r).
    let tmp: &[f32] = tmp;
    dst.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let row = |yy: isize| -> &[f32] {
            let yy = yy.clamp(0, h as isize - 1) as usize;
            &tmp[yy * w..(yy + 1) * w]
        };
        let kc = k[r];
        for (o, p) in out.iter_mut().zip(row(y as isize)) {
            *o = kc * p;
        }
        for i in 0..r {
            let ki = k[i];
            let a = row(y as isize + i as isize - r as isize);
            let b = row(y as isize + r as isize - i as isize);
            for ((o, x), z) in out.iter_mut().zip(a).zip(b) {
                *o += ki * (x + z);
            }
        }
    });
}

/// 2x upsample: output (2r+a, 2c+b) = bilinear around input (r,c) (weights a/2, b/2), edge replication.
// Design decision: out-of-range neighbors of the last row/column use edge replication.
pub fn upsample2(src: &[f32], w: usize, h: usize, dst: &mut [f32]) {
    let w2 = 2 * w;
    dst.par_chunks_mut(w2).enumerate().for_each(|(yy, out)| {
        let r = yy / 2;
        let a = (yy % 2) as f32 * 0.5;
        let r1 = (r + 1).min(h - 1);
        let row0 = &src[r * w..(r + 1) * w];
        let row1 = &src[r1 * w..(r1 + 1) * w];
        for c in 0..w {
            let c1 = (c + 1).min(w - 1);
            let top0 = row0[c];
            let bot0 = row1[c];
            let v0 = top0 + a * (bot0 - top0);
            let v1 = row0[c1] + a * (row1[c1] - row0[c1]);
            out[2 * c] = v0;
            out[2 * c + 1] = v0 + 0.5 * (v1 - v0);
        }
    });
}

/// Takes only even rows/columns for 1/2 downscale (no averaging). Output size (w/2, h/2).
// Design decision: odd sizes use the rounded-down size.
pub fn decimate2(src: &[f32], w: usize, dst: &mut [f32], w2: usize) {
    dst.par_chunks_mut(w2).enumerate().for_each(|(y, out)| {
        let row = &src[2 * y * w..];
        for (x, o) in out.iter_mut().enumerate() {
            *o = row[2 * x];
        }
    });
}

/// One octave: Gaussian levels l = -1..=S+1 (S+3 images).
pub struct Octave {
    /// Octave index o (starts from -1 if the first octave is -1).
    pub o: i32,
    /// Octave width.
    pub w: usize,
    /// Octave height.
    pub h: usize,
    /// Gaussian level images (in order l = -1..=S+1).
    pub gauss: Vec<Vec<f32>>,
}

impl Octave {
    /// Gaussian level l (-1..=S+1).
    #[inline]
    pub fn level(&self, l: i32) -> &[f32] {
        &self.gauss[(l + 1) as usize]
    }
}

/// Scale-space constants.
#[derive(Clone, Copy, Debug)]
pub struct ScaleSpace {
    /// Number of detection levels per octave S.
    pub s: usize,
    /// Inter-level factor k = 2^(1/S).
    pub k: f32,
    /// Base σ₀.
    pub sigma0: f32,
}

impl ScaleSpace {
    /// Computes the constants from the detection level count S.
    pub fn new(s: usize) -> Self {
        let k = 2f32.powf(1.0 / s as f32);
        Self { s, k, sigma0: 1.6 * k }
    }
    /// Relative σ(l) = σ₀ k^l within an octave.
    pub fn sigma(&self, l: f32) -> f32 {
        self.sigma0 * self.k.powf(l)
    }
    /// Increment σ from level l-1 -> l.
    pub fn sigma_inc(&self, l: i32) -> f32 {
        let a = self.sigma(l as f32);
        let b = self.sigma(l as f32 - 1.0);
        (a * a - b * b).sqrt()
    }
}

/// Automatic octave count: max(1, floor(log2(min(W₀,H₀))) - 3).
pub fn auto_num_octaves(w0: usize, h0: usize) -> usize {
    let m = w0.min(h0).max(1);
    let l = (usize::BITS - 1 - m.leading_zeros()) as i64;
    (l - 3).max(1) as usize
}

/// Builds the pyramid. `base` is the pre-blur image at first-octave resolution (upsampling done).
#[allow(clippy::too_many_arguments)]
pub fn build_pyramid(
    base: Vec<f32>,
    w0: usize,
    h0: usize,
    o_min: i32,
    num_octaves: usize,
    ss: ScaleSpace,
    sigma_nominal: f32,
    pool: &BufferPool,
) -> Vec<Octave> {
    let n_levels = ss.s + 3;
    let mut octaves: Vec<Octave> = Vec::with_capacity(num_octaves);
    let mut tmp = pool.take(w0 * h0);
    let mut src = base;
    let (mut w, mut h) = (w0, h0);
    for oi in 0..num_octaves {
        let o = o_min + oi as i32;
        let mut gauss = Vec::with_capacity(n_levels);
        if oi == 0 {
            let s_first = ss.sigma(-1.0);
            let sn = sigma_nominal / 2f32.powi(o_min);
            let mut g0 = pool.take(w * h);
            if s_first > sn {
                gaussian_blur(&src, &mut g0, &mut tmp[..w * h], w, h, (s_first * s_first - sn * sn).sqrt());
                pool.give(std::mem::take(&mut src));
            } else {
                g0.copy_from_slice(&src);
            }
            gauss.push(g0);
        } else {
            // Decimate level S-1 (σ = 2·1.6) of the previous octave.
            let prev = &octaves[oi - 1];
            let (pw, ph) = (prev.w, prev.h);
            let (nw, nh) = (pw / 2, ph / 2);
            let mut g0 = pool.take(nw * nh);
            decimate2(prev.level(ss.s as i32 - 1), pw, &mut g0, nw);
            w = nw;
            h = nh;
            gauss.push(g0);
        }
        for l in 0..=(ss.s as i32 + 1) {
            let mut g = pool.take(w * h);
            gaussian_blur(gauss.last().expect("레벨 있음"), &mut g, &mut tmp[..w * h], w, h, ss.sigma_inc(l));
            gauss.push(g);
        }
        octaves.push(Octave { o, w, h, gauss });
        if w / 2 < 4 || h / 2 < 4 {
            break;
        }
    }
    pool.give(tmp);
    octaves
}
