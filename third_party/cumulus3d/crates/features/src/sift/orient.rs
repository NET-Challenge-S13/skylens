/*
 * orient.rs
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

//! 방향 할당·128차원 기술자.

use rayon::prelude::*;
use std::f32::consts::PI;

const TWO_PI: f32 = 2.0 * PI;

/// 가우시안 레벨의 기울기(크기 m = ½‖∇‖, 각 θ = atan2(dy, dx)). 미리 계산하거나 즉석 계산.
pub struct Gradient<'a> {
    g: &'a [f32],
    w: usize,
    /// 영상 높이.
    pub h: usize,
    pre: Option<(Vec<f32>, Vec<f32>)>,
}

impl<'a> Gradient<'a> {
    /// 즉석 계산(키포인트가 적은 레벨).
    pub fn lazy(g: &'a [f32], w: usize, h: usize) -> Self {
        Self { g, w, h, pre: None }
    }

    /// 전 영상 미리 계산(키포인트가 많은 레벨). 결과는 즉석 계산과 비트 단위 동일.
    pub fn precomputed(g: &'a [f32], w: usize, h: usize) -> Self {
        let mut mag = vec![0f32; w * h];
        let mut ang = vec![0f32; w * h];
        mag.par_chunks_mut(w).zip(ang.par_chunks_mut(w)).enumerate().for_each(|(y, (m, a))| {
            if y == 0 || y + 1 >= h {
                return;
            }
            for x in 1..w.saturating_sub(1) {
                let (mm, aa) = grad_at(g, w, x, y);
                m[x] = mm;
                a[x] = aa;
            }
        });
        Self { g, w, h, pre: Some((mag, ang)) }
    }

    /// 영상 너비.
    pub fn width(&self) -> usize {
        self.w
    }

    /// 화소 (x, y) 의 (m, θ). 1 ≤ x ≤ w−2, 1 ≤ y ≤ h−2 전제.
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> (f32, f32) {
        match &self.pre {
            Some((m, a)) => (m[y * self.w + x], a[y * self.w + x]),
            None => grad_at(self.g, self.w, x, y),
        }
    }
}

#[inline]
fn grad_at(g: &[f32], w: usize, x: usize, y: usize) -> (f32, f32) {
    let i = y * w + x;
    let dx = g[i + 1] - g[i - 1];
    let dy = g[i + w] - g[i - w];
    (0.5 * (dx * dx + dy * dy).sqrt(), dy.atan2(dx))
}

/// 방향 할당 매개변수.
#[derive(Clone, Copy, Debug)]
pub struct OrientParams {
    /// 키포인트당 최대 방향 수.
    pub max_num_orientations: usize,
    /// 인접 빈 선형 보간(개선 옵션).
    pub bin_interpolation: bool,
}

/// 16비트 양자화된 내부 각 θ_int ([0, 2π)).
#[inline]
pub fn quantize_angle(theta: f32) -> u16 {
    let f = (theta / TWO_PI).rem_euclid(1.0);
    ((f * 65535.0).floor() as u32).min(65534) as u16
}

// 설계 결정: 16비트 값의 복원 배율은 저장과 같은 65535 사용.
#[inline]
/// 양자화 각 → 라디안([`quantize_angle`] 의 역).
pub fn dequantize_angle(q: u16) -> f32 {
    q as f32 / 65535.0 * TWO_PI
}

/// 키포인트 (x, y, σ) 의 주 방향들(양자화된 θ_int). 피크가 없으면 빈 목록.
pub fn orientations(grad: &Gradient, x: f32, y: f32, sigma: f32, p: &OrientParams) -> Vec<u16> {
    const NB: usize = 36;
    let w = grad.width();
    let h = grad.h;
    let sw = 1.5 * sigma;
    let rad = 2.0 * sw;
    let x0 = ((x - rad).floor() as i64).max(1);
    let x1 = ((x + rad).floor() as i64).min(w as i64 - 2);
    let y0 = ((y - rad).floor() as i64).max(1);
    let y1 = ((y + rad).floor() as i64).min(h as i64 - 2);
    let r2 = rad * rad + 0.5;
    let inv = 1.0 / (2.0 * sw * sw);
    let mut hist = [0f32; NB];
    let bscale = NB as f32 / TWO_PI;
    for yi in y0..=y1 {
        let dy = yi as f32 + 0.5 - y;
        for xi in x0..=x1 {
            let dx = xi as f32 + 0.5 - x;
            let d2 = dx * dx + dy * dy;
            if d2 >= r2 {
                continue;
            }
            let (m, th) = grad.at(xi as usize, yi as usize);
            let wgt = m * (-d2 * inv).exp();
            if p.bin_interpolation {
                let t = th * bscale - 0.5;
                let b0 = t.floor();
                let f = t - b0;
                let b0 = (b0 as i64).rem_euclid(NB as i64) as usize;
                hist[b0] += wgt * (1.0 - f);
                hist[(b0 + 1) % NB] += wgt * f;
            } else {
                let mut b = (th * bscale).floor() as i64;
                if b < 0 {
                    b += NB as i64;
                }
                hist[(b as usize).min(NB - 1)] += wgt;
            }
        }
    }
    // 원형 3탭 평균 6회.
    for _ in 0..6 {
        let prev = hist;
        for i in 0..NB {
            hist[i] = (prev[(i + NB - 1) % NB] + prev[i] + prev[(i + 1) % NB]) / 3.0;
        }
    }
    let hmax = hist.iter().cloned().fold(0f32, f32::max);
    let mut peaks: Vec<(f32, usize)> = Vec::new();
    for i in 0..NB {
        let (hl, hc, hr) = (hist[(i + NB - 1) % NB], hist[i], hist[(i + 1) % NB]);
        if hc > 0.8 * hmax && hc > hl && hc > hr {
            peaks.push((hc, i));
        }
    }
    // 가중치 내림차순(동률은 빈 번호 오름차순으로 결정적).
    peaks.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut out: Vec<u16> = Vec::with_capacity(p.max_num_orientations);
    for &(hc, i) in peaks.iter().take(p.max_num_orientations.max(1)) {
        let (hl, hr) = (hist[(i + NB - 1) % NB], hist[(i + 1) % NB]);
        let di = 0.5 * (hr - hl) / (2.0 * hc - hr - hl);
        let th = (i as f32 + di + 0.5) / bscale;
        let q = quantize_angle(th);
        if !out.contains(&q) {
            out.push(q);
        }
    }
    out
}

/// 기술자 정규화 방식.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DescriptorNormalization {
    /// L1 정규화 후 제곱근(RootSIFT). 기본.
    #[default]
    L1Root,
    /// L2 정규화.
    L2,
}

/// 128차원 기술자(uint8). θ 는 내부 각(θ_int).
pub fn descriptor(grad: &Gradient, x: f32, y: f32, sigma: f32, theta: f32, norm: DescriptorNormalization) -> [u8; 128] {
    let w = grad.width();
    let h = grad.h;
    let s = 3.0 * sigma;
    let (sn, c) = theta.sin_cos();
    let rad = s * 2.5 * std::f32::consts::SQRT_2 + 1.0;
    let x0 = ((x - rad).floor() as i64).max(1);
    let x1 = ((x + rad).ceil() as i64).min(w as i64 - 2);
    let y0 = ((y - rad).floor() as i64).max(1);
    let y1 = ((y + rad).ceil() as i64).min(h as i64 - 2);
    let inv_s = 1.0 / s;
    let bins_per_rad = 4.0 / PI;
    let mut d = [0f32; 128];
    for yi in y0..=y1 {
        let dy = yi as f32 + 0.5 - y;
        for xi in x0..=x1 {
            let dx = xi as f32 + 0.5 - x;
            // 회전된 셀 좌표(키포인트 중심 기준).
            let rx = (c * dx + sn * dy) * inv_s;
            let ry = (c * dy - sn * dx) * inv_s;
            let fx = rx + 1.5;
            let fy = ry + 1.5;
            if fx <= -1.0 || fx >= 4.0 || fy <= -1.0 || fy >= 4.0 {
                continue;
            }
            let (m, th) = grad.at(xi as usize, yi as usize);
            let gw = (-(rx * rx + ry * ry) / 8.0).exp() * m;
            let t = ((theta - th) * bins_per_rad).rem_euclid(8.0);
            let tb = t.floor();
            let tf = t - tb;
            let b0 = (tb as usize) % 8;
            let b1 = (b0 + 1) % 8;
            let ix0 = fx.floor() as i64;
            let iy0 = fy.floor() as i64;
            let wx1 = fx - ix0 as f32;
            let wy1 = fy - iy0 as f32;
            for (iy, wy) in [(iy0, 1.0 - wy1), (iy0 + 1, wy1)] {
                if !(0..4).contains(&iy) || wy <= 0.0 {
                    continue;
                }
                for (ix, wx) in [(ix0, 1.0 - wx1), (ix0 + 1, wx1)] {
                    if !(0..4).contains(&ix) || wx <= 0.0 {
                        continue;
                    }
                    let base = ((iy * 4 + ix) * 8) as usize;
                    let ww = gw * wx * wy;
                    d[base + b0] += ww * (1.0 - tf);
                    d[base + b1] += ww * tf;
                }
            }
        }
    }
    normalize_quantize(&mut d, norm)
}

/// L2 → 0.2 절단 → L2, 이어서 L1-ROOT(또는 L2), ×512 반올림 후 `[0,255]` 포화.
pub fn normalize_quantize(d: &mut [f32; 128], norm: DescriptorNormalization) -> [u8; 128] {
    let l2 = |d: &mut [f32; 128]| {
        let n = d.iter().map(|v| v * v).sum::<f32>().sqrt();
        if n > 0.0 {
            d.iter_mut().for_each(|v| *v /= n);
        }
    };
    l2(d);
    d.iter_mut().for_each(|v| *v = v.min(0.2));
    l2(d);
    match norm {
        DescriptorNormalization::L1Root => {
            let s: f32 = d.iter().sum();
            if s > 0.0 {
                d.iter_mut().for_each(|v| *v = (*v / s).sqrt());
            }
        }
        DescriptorNormalization::L2 => l2(d),
    }
    let mut q = [0u8; 128];
    for (o, v) in q.iter_mut().zip(d.iter()) {
        *o = (512.0 * v).round().clamp(0.0, 255.0) as u8;
    }
    q
}
