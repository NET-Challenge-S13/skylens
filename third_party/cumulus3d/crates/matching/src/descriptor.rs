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

//! 기술자 무차별 매칭.
//!
//! 정수 내적 d = Σ a_k b_k (u8×u8→u32) 를 블록 GEMM 처럼 계산하고, 행·열별 top-2 를
//! 타일 안에서 바로 축약한다(내적 행렬 전체를 저장하지 않음). 거리 = arccos(min(d/512², 1)).

use cumulus3d_core::{Descriptors, FeatureMatch, DESCRIPTOR_DIM};
use rayon::prelude::*;

/// 512² — 정규화 기술자 노름의 제곱.
pub const DOT_NORM: f32 = 262144.0;

/// SIFT 기술자 매칭 옵션.
#[derive(Clone, Debug, PartialEq)]
pub struct DescriptorMatchOptions {
    /// 최근접/차근접 각 거리 비율 상한(Lowe 비율 검사).
    pub max_ratio: f64,
    /// 최근접 각 거리 상한(라디안).
    pub max_distance: f64,
    /// 교차 검사(양방향 최근접 일치) 사용 여부.
    pub cross_check: bool,
    /// 경계 규칙: GPU(θ1 < max_dist && θ1 < r θ2) 또는 CPU 무차별(θ1 ≤ max_dist && θ1 < r θ2).
    pub rule: AcceptRule,
}

impl Default for DescriptorMatchOptions {
    fn default() -> Self {
        Self { max_ratio: 0.8, max_distance: 0.7, cross_check: true, rule: AcceptRule::Gpu }
    }
}

/// 채택 조건의 경계 처리(경계값에서만 차이).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AcceptRule {
    #[default]
    /// GPU 규칙: θ1 < max_distance (엄격).
    Gpu,
    /// CPU 무차별 규칙: θ1 ≤ max_distance.
    CpuBruteForce,
}

/// 한 행(또는 열)의 top-2: 최댓값 내적, 그 인덱스, 두 번째 최댓값.
/// 초기값 0, "엄격히 큼"으로 갱신하므로 동점이면 앞선 인덱스 유지.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Top2 {
    /// 최댓값 내적.
    pub best: u32,
    /// 최댓값의 상대 인덱스(없으면 `u32::MAX`).
    pub idx: u32,
    /// 두 번째 최댓값 내적.
    pub second: u32,
}

impl Default for Top2 {
    fn default() -> Self {
        Self { best: 0, idx: u32::MAX, second: 0 }
    }
}

impl Top2 {
    #[inline]
    /// 내적 `d`(인덱스 `idx`)로 top-2 를 갱신한다.
    pub fn push(&mut self, d: u32, idx: u32) {
        if d > self.best {
            self.second = self.best;
            self.best = d;
            self.idx = idx;
        } else if d > self.second {
            self.second = d;
        }
    }
    /// 앞 구간(self)과 뒤 구간(o)의 결과 병합. 순차 갱신과 같은 결과(결합적).
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

/// 매칭 커널 백엔드. GPU 구현은 `top2` 만 바꿔 끼우면 된다(융합 GEMM + top-2 에필로그).
pub trait MatcherBackend: Send + Sync {
    /// 행(영상1 각 기술자)별, 열(영상2 각 기술자)별 정수 내적 top-2.
    fn top2(&self, d1: &[u8], n1: usize, d2: &[u8], n2: usize) -> (Vec<Top2>, Vec<Top2>);

    /// 비율·거리·교차 검사를 적용한 매칭. 영상1 인덱스 오름차순, 최대 `max_num_matches`.
    /// 기술자 수가 `max_num_matches` 를 넘으면 앞쪽만 쓴다(GPU 동작).
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

/// 내적 → 각도(라디안). arccos(min(d/262144, 1)).
// 설계 결정: GPU 경로와 결과를 맞추려고 f32 로 계산한다.
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

/// top-2 결과에 검사들을 적용.
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

/// CPU 백엔드: rayon 병렬 + 캐시 타일링(행 블록 × 열 타일).
#[derive(Clone, Debug)]
pub struct CpuMatcher {
    /// 병렬 작업 단위 행 수.
    pub row_block: usize,
    /// L1 에 머무는 열 타일 크기.
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

/// 4행 × 1열 마이크로 커널: b 한 줄을 네 번 재사용.
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
        // 행 블록마다: 그 블록의 행 top-2 와 (그 블록 행들에 한정한) 열 top-2.
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
                    // 내적 타일 계산(행 4개씩).
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
                    // 행·열 축약(인덱스 오름차순 순회 → 순차 규칙과 동일).
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

/// 기준 구현(테스트용): 이중 루프 순차 top-2.
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
