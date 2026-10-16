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

//! 가우시안 스케일 공간. 분리형 f32 블러, 행 단위 rayon 병렬.

use rayon::prelude::*;
use std::sync::Mutex;

/// f32 버퍼 재사용 풀(영상 간 피라미드 할당 재사용).
#[derive(Default)]
pub struct BufferPool {
    bufs: Mutex<Vec<Vec<f32>>>,
}

impl BufferPool {
    /// 길이 `len` 버퍼(내용은 덮어쓰기 전제). 용량이 맞는 것을 우선 재사용.
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

    /// 버퍼를 풀에 돌려준다.
    pub fn give(&self, v: Vec<f32>) {
        if v.capacity() == 0 {
            return;
        }
        let mut g = self.bufs.lock().unwrap_or_else(|e| e.into_inner());
        // 무한 증가 방지: 오래된 것은 버림.
        if g.len() < 64 {
            g.push(v);
        }
    }
}

/// 1D 가우시안 커널: 반폭 r = ceil(4σ − 0.5), 폭 2r+1 을 [5, 33] 로 제한, 합 1 정규화.
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

/// 분리형 가우시안 블러(가장자리 복제). `tmp` 는 같은 크기 작업 버퍼.
#[allow(clippy::needless_range_loop)]
pub fn gaussian_blur(src: &[f32], dst: &mut [f32], tmp: &mut [f32], w: usize, h: usize, sigma: f32) {
    let k = gaussian_kernel(sigma);
    let r = k.len() / 2;
    // 가로: 패딩한 행에 대해 out[x] += k_i · pad[x + i] (벡터화되는 형태).
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
    // 세로: out 행 y = Σ k_i · tmp 행 clamp(y + i − r).
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

/// 2배 업샘플: 출력 (2r+a, 2c+b) = 입력 (r,c) 주변 쌍선형(가중치 a/2, b/2), 가장자리 복제.
// 설계 결정: 마지막 행/열의 범위 밖 이웃은 가장자리 복제.
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

/// 짝수 행·열만 취해 1/2 축소(평균 없음). 출력 크기 (w/2, h/2).
// 설계 결정: 홀수 크기는 내림 크기 사용.
pub fn decimate2(src: &[f32], w: usize, dst: &mut [f32], w2: usize) {
    dst.par_chunks_mut(w2).enumerate().for_each(|(y, out)| {
        let row = &src[2 * y * w..];
        for (x, o) in out.iter_mut().enumerate() {
            *o = row[2 * x];
        }
    });
}

/// 옥타브 하나: 가우시안 레벨 l = −1..=S+1 (S+3 장).
pub struct Octave {
    /// 옥타브 번호 o (첫 옥타브 −1 이면 −1부터).
    pub o: i32,
    /// 옥타브 너비.
    pub w: usize,
    /// 옥타브 높이.
    pub h: usize,
    /// 가우시안 레벨 영상들(l = −1..=S+1 순).
    pub gauss: Vec<Vec<f32>>,
}

impl Octave {
    /// 가우시안 레벨 l (−1..=S+1).
    #[inline]
    pub fn level(&self, l: i32) -> &[f32] {
        &self.gauss[(l + 1) as usize]
    }
}

/// 스케일 공간 상수.
#[derive(Clone, Copy, Debug)]
pub struct ScaleSpace {
    /// 옥타브당 검출 레벨 수 S.
    pub s: usize,
    /// 레벨 간 배율 k = 2^(1/S).
    pub k: f32,
    /// 기준 σ₀.
    pub sigma0: f32,
}

impl ScaleSpace {
    /// 검출 레벨 수 S 로 상수를 계산.
    pub fn new(s: usize) -> Self {
        let k = 2f32.powf(1.0 / s as f32);
        Self { s, k, sigma0: 1.6 * k }
    }
    /// 옥타브 내 상대 σ(l) = σ₀ k^l.
    pub fn sigma(&self, l: f32) -> f32 {
        self.sigma0 * self.k.powf(l)
    }
    /// 레벨 l−1 → l 증분 σ.
    pub fn sigma_inc(&self, l: i32) -> f32 {
        let a = self.sigma(l as f32);
        let b = self.sigma(l as f32 - 1.0);
        (a * a - b * b).sqrt()
    }
}

/// 옥타브 수 자동 결정: max(1, floor(log2(min(W₀,H₀))) − 3).
pub fn auto_num_octaves(w0: usize, h0: usize) -> usize {
    let m = w0.min(h0).max(1);
    let l = (usize::BITS - 1 - m.leading_zeros()) as i64;
    (l - 3).max(1) as usize
}

/// 피라미드 구성. `base` 는 첫 옥타브 해상도의 블러 전 영상(업샘플 완료).
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
            // 이전 옥타브 레벨 S−1 (σ = 2·1.6) 을 데시메이트.
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
