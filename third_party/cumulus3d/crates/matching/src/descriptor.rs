/*
 * descriptor.rs
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

//! Brute-force descriptor matching.
//!
//! Computes the integer dot product d = Σ a_k b_k (u8×u8->u32) like a block GEMM, and reduces the per-row and per-column top-2
//! directly inside each tile (the full dot-product matrix is not stored). Distance = arccos(min(d/512², 1)).

use cumulus3d_core::{Descriptors, FeatureMatch, DESCRIPTOR_DIM};
use rayon::prelude::*;

/// 512² - the squared norm of a normalized descriptor.
pub const DOT_NORM: f32 = 262144.0;

/// SIFT descriptor matching options.
#[derive(Clone, Debug, PartialEq)]
pub struct DescriptorMatchOptions {
    /// Upper bound on the nearest/second-nearest angular distance ratio (Lowe ratio test).
    pub max_ratio: f64,
    /// Upper bound on the nearest angular distance (radians).
    pub max_distance: f64,
    /// Whether to use the cross-check (mutual nearest-neighbor agreement).
    pub cross_check: bool,
    /// Boundary rule: GPU (θ1 < max_dist && θ1 < r θ2) or CPU brute-force (θ1 ≤ max_dist && θ1 < r θ2).
    pub rule: AcceptRule,
}

impl Default for DescriptorMatchOptions {
    fn default() -> Self {
        Self { max_ratio: 0.8, max_distance: 0.7, cross_check: true, rule: AcceptRule::Gpu }
    }
}

/// Boundary handling of the acceptance condition (differs only at the boundary value).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AcceptRule {
    #[default]
    /// GPU rule: θ1 < max_distance (strict).
    Gpu,
    /// CPU brute-force rule: θ1 ≤ max_distance.
    CpuBruteForce,
}

/// Top-2 of one row (or column): maximum dot product, its index, and the second maximum.
/// Initial value 0, updated only on "strictly greater", so ties keep the earlier index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Top2 {
    /// Maximum dot product.
    pub best: u32,
    /// Relative index of the maximum (`u32::MAX` if none).
    pub idx: u32,
    /// Second maximum dot product.
    pub second: u32,
}

impl Default for Top2 {
    fn default() -> Self {
        Self { best: 0, idx: u32::MAX, second: 0 }
    }
}

impl Top2 {
    #[inline]
    /// Updates the top-2 with dot product `d` (index `idx`).
    pub fn push(&mut self, d: u32, idx: u32) {
        if d > self.best {
            self.second = self.best;
            self.best = d;
            self.idx = idx;
        } else if d > self.second {
            self.second = d;
        }
    }
    /// Merges the result of the earlier segment (self) and the later segment (o). Same result as sequential updating (associative).
    #[inline]
    pub fn merge(&mut self, o: &Top2) {
        if o.best > self.best {
            self.second = self.best.max(o.second);
            self.best = o.best;
            self.idx = o.idx;
        } else {
            self.second = self.second.max(o.best);
        }
    }
}

/// Matching kernel backend. A GPU implementation only needs to swap in `top2` (fused GEMM + top-2 epilogue).
pub trait MatcherBackend: Send + Sync {
    /// Integer dot-product top-2 per row (each image-1 descriptor) and per column (each image-2 descriptor).
    fn top2(&self, d1: &[u8], n1: usize, d2: &[u8], n2: usize) -> (Vec<Top2>, Vec<Top2>);

    /// Matching with the ratio, distance, and cross checks applied. Ascending image-1 index, at most `max_num_matches`.
    /// If the descriptor count exceeds `max_num_matches`, only the leading ones are used (GPU behavior).
    fn match_descriptors(
        &self,
        d1: &Descriptors,
        d2: &Descriptors,
        opts: &DescriptorMatchOptions,
        max_num_matches: usize,
    ) -> Vec<FeatureMatch> {
        let n1 = d1.len().min(max_num_matches);
        let n2 = d2.len().min(max_num_matches);
        if n1 == 0 || n2 == 0 {
            return Vec::new();
        }
        let (rows, cols) = self.top2(&d1.as_slice()[..n1 * DESCRIPTOR_DIM], n1, &d2.as_slice()[..n2 * DESCRIPTOR_DIM], n2);
        apply_tests(&rows, &cols, opts, max_num_matches)
    }
}

/// Dot product -> angle (radians). arccos(min(d/262144, 1)).
// Design decision: compute in f32 to match the GPU path's results.
#[inline]
pub fn dot_to_angle(d: u32) -> f32 {
    (d as f32 / DOT_NORM).min(1.0).acos()
}

#[inline]
fn accept(t: &Top2, opts: &DescriptorMatchOptions) -> bool {
    if t.idx == u32::MAX {
        return false;
    }
    let th1 = dot_to_angle(t.best);
    let th2 = dot_to_angle(t.second);
    let maxd = opts.max_distance as f32;
    let ratio = opts.max_ratio as f32;
    match opts.rule {
        AcceptRule::Gpu => th1 < maxd && th1 < ratio * th2,
        AcceptRule::CpuBruteForce => !(th1 > maxd || th1 >= ratio * th2),
    }
}

/// Applies the checks to the top-2 results.
pub fn apply_tests(rows: &[Top2], cols: &[Top2], opts: &DescriptorMatchOptions, max_num_matches: usize) -> Vec<FeatureMatch> {
    let mut out = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        if !accept(r, opts) {
            continue;
        }
        if opts.cross_check {
            let c = &cols[r.idx as usize];
            if c.idx != i as u32 || !accept(c, opts) {
                continue;
            }
        }
        out.push(FeatureMatch::new(i as u32, r.idx));
        if out.len() >= max_num_matches {
            break;
        }
    }
    out
}

/// CPU backend: rayon-parallel + cache tiling (row blocks × column tiles).
#[derive(Clone, Debug)]
pub struct CpuMatcher {
    /// Number of rows per parallel work unit.
    pub row_block: usize,
    /// Column tile size that stays in L1.
    pub col_tile: usize,
}

impl Default for CpuMatcher {
    fn default() -> Self {
        Self { row_block: 128, col_tile: 64 }
    }
}

#[inline(always)]
fn dot128(a: &[u8], b: &[u8]) -> u32 {
    let a: &[u8; DESCRIPTOR_DIM] = a.try_into().expect("128");
    let b: &[u8; DESCRIPTOR_DIM] = b.try_into().expect("128");
    let mut s = 0u32;
    for k in 0..DESCRIPTOR_DIM {
        s += a[k] as u16 as u32 * b[k] as u16 as u32;
    }
    s
}

/// 4-row × 1-column micro-kernel: reuses one line of b four times.
#[inline(always)]
fn dot4x1(a0: &[u8], a1: &[u8], a2: &[u8], a3: &[u8], b: &[u8]) -> [u32; 4] {
    let a0: &[u8; DESCRIPTOR_DIM] = a0.try_into().expect("128");
    let a1: &[u8; DESCRIPTOR_DIM] = a1.try_into().expect("128");
    let a2: &[u8; DESCRIPTOR_DIM] = a2.try_into().expect("128");
    let a3: &[u8; DESCRIPTOR_DIM] = a3.try_into().expect("128");
    let b: &[u8; DESCRIPTOR_DIM] = b.try_into().expect("128");
    let (mut s0, mut s1, mut s2, mut s3) = (0u32, 0u32, 0u32, 0u32);
    for k in 0..DESCRIPTOR_DIM {
        let bk = b[k] as u32;
        s0 += a0[k] as u32 * bk;
        s1 += a1[k] as u32 * bk;
        s2 += a2[k] as u32 * bk;
        s3 += a3[k] as u32 * bk;
    }
    [s0, s1, s2, s3]
}

impl MatcherBackend for CpuMatcher {
    fn top2(&self, d1: &[u8], n1: usize, d2: &[u8], n2: usize) -> (Vec<Top2>, Vec<Top2>) {
        const D: usize = DESCRIPTOR_DIM;
        let rb = self.row_block.max(4);
        let ct = self.col_tile.max(1);
        // Per row block: that block's row top-2 and column top-2 (restricted to the rows of that block).
        let parts: Vec<(Vec<Top2>, Vec<Top2>)> = (0..n1.div_ceil(rb))
            .into_par_iter()
            .map(|blk| {
                let r0 = blk * rb;
                let r1 = (r0 + rb).min(n1);
                let mut rows = vec![Top2::default(); r1 - r0];
                let mut cols = vec![Top2::default(); n2];
                let mut tile = vec![0u32; (r1 - r0) * ct];
                let mut c0 = 0;
                while c0 < n2 {
                    let c1 = (c0 + ct).min(n2);
                    let w = c1 - c0;
                    // Compute the dot-product tile (4 rows at a time).
                    let mut i = r0;
                    while i + 4 <= r1 {
                        let a = |k: usize| &d1[(i + k) * D..(i + k + 1) * D];
                        let (a0, a1, a2, a3) = (a(0), a(1), a(2), a(3));
                        for j in c0..c1 {
                            let s = dot4x1(a0, a1, a2, a3, &d2[j * D..(j + 1) * D]);
                            for k in 0..4 {
                                tile[(i + k - r0) * ct + (j - c0)] = s[k];
                            }
                        }
                        i += 4;
                    }
                    while i < r1 {
                        let a = &d1[i * D..(i + 1) * D];
                        for j in c0..c1 {
                            tile[(i - r0) * ct + (j - c0)] = dot128(a, &d2[j * D..(j + 1) * D]);
                        }
                        i += 1;
                    }
                    // Row/column reduction (traversal in ascending index order -> same as the sequential rule).
                    for li in 0..(r1 - r0) {
                        let row = &tile[li * ct..li * ct + w];
                        let gi = (r0 + li) as u32;
                        let rt = &mut rows[li];
                        for (lj, &d) in row.iter().enumerate() {
                            rt.push(d, (c0 + lj) as u32);
                            cols[c0 + lj].push(d, gi);
                        }
                    }
                    c0 = c1;
                }
                (rows, cols)
            })
            .collect();
        let mut rows = Vec::with_capacity(n1);
        let mut cols = vec![Top2::default(); n2];
        for (r, c) in parts {
            rows.extend(r);
            for (acc, p) in cols.iter_mut().zip(&c) {
                acc.merge(p);
            }
        }
        (rows, cols)
    }
}

/// Reference implementation (for tests): double-loop sequential top-2.
pub fn top2_naive(d1: &[u8], n1: usize, d2: &[u8], n2: usize) -> (Vec<Top2>, Vec<Top2>) {
    const D: usize = DESCRIPTOR_DIM;
    let mut rows = vec![Top2::default(); n1];
    let mut cols = vec![Top2::default(); n2];
    for i in 0..n1 {
        for j in 0..n2 {
            let d = dot128(&d1[i * D..(i + 1) * D], &d2[j * D..(j + 1) * D]);
            rows[i].push(d, j as u32);
        }
    }
    for j in 0..n2 {
        for i in 0..n1 {
            let d = dot128(&d1[i * D..(i + 1) * D], &d2[j * D..(j + 1) * D]);
            cols[j].push(d, i as u32);
        }
    }
    (rows, cols)
}
