/*
 * upsample.rs
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
 * SPDX-License-Identifier: Apache-2.0
 */

//! 영상 피라미드, 스케일 사이 깊이·법선 전달(결합 양방향 상향 표본), 5×5 중앙값 평면 필터, 깊이 축소.

use crate::kernel::{LevelImage, ViewState};
use crate::math::plane_transfer;
use rayon::prelude::*;
use std::sync::Arc;

/// 스케일 수: 짧은 변을 절반씩 줄여 `min_size` 이상으로 남는 만큼(최대 `max_levels`, 최소 1).
pub fn num_levels(min_side: usize, max_levels: usize, min_size: usize) -> usize {
    let mut l = 1;
    let mut s = min_side;
    while l < max_levels && s / 2 >= min_size {
        s /= 2;
        l += 1;
    }
    l
}

/// 2×2 평균 축소 전 가우스(σ = 0.5 상위 픽셀) 평활 커널(반경 1).
fn blur3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let g = [(-1.0f32 / (2.0 * 0.25)).exp(), 1.0, (-1.0f32 / (2.0 * 0.25)).exp()];
    let s: f32 = g.iter().sum();
    let g = [g[0] / s, g[1] / s, g[2] / s];
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut a = 0.0;
            for (k, gk) in g.iter().enumerate() {
                let xx = (x as i64 + k as i64 - 1).clamp(0, w as i64 - 1) as usize;
                a += gk * src[y * w + xx];
            }
            tmp[y * w + x] = a;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut a = 0.0;
            for (k, gk) in g.iter().enumerate() {
                let yy = (y as i64 + k as i64 - 1).clamp(0, h as i64 - 1) as usize;
                a += gk * tmp[yy * w + x];
            }
            out[y * w + x] = a;
        }
    }
    out
}

/// 회색 피라미드(0 = 최저). 각 단계는 평활 후 2×2 평균, 8비트로 반올림 저장.
/// 내부 행렬: f·0.5, c ← (c + 0.5)·0.5 − 0.5.
pub fn build_pyramid(gray: &Arc<Vec<u8>>, width: usize, height: usize, k: [f64; 4], levels: usize) -> Vec<LevelImage> {
    let mut out = vec![LevelImage { width, height, gray: gray.clone(), k: [k[0] as f32, k[1] as f32, k[2] as f32, k[3] as f32] }];
    let (mut w, mut h) = (width, height);
    let mut kk = k;
    let mut cur: Vec<f32> = gray.iter().map(|&v| v as f32).collect();
    for _ in 1..levels {
        let b = blur3(&cur, w, h);
        let (nw, nh) = (w / 2, h / 2);
        let mut next = vec![0.0f32; nw * nh];
        for y in 0..nh {
            for x in 0..nw {
                let i = 2 * y * w + 2 * x;
                next[y * nw + x] = 0.25 * (b[i] + b[i + 1] + b[i + w] + b[i + w + 1]);
            }
        }
        let q: Vec<u8> = next.iter().map(|v| v.round().clamp(0.0, 255.0) as u8).collect();
        cur = q.iter().map(|&v| v as f32).collect();
        kk = [kk[0] * 0.5, kk[1] * 0.5, (kk[2] + 0.5) * 0.5 - 0.5, (kk[3] + 0.5) * 0.5 - 0.5];
        w = nw;
        h = nh;
        out.push(LevelImage { width: w, height: h, gray: Arc::new(q), k: [kk[0] as f32, kk[1] as f32, kk[2] as f32, kk[3] as f32] });
    }
    out.reverse();
    out
}

#[inline]
fn ray(k: &[f32; 4], x: f64, y: f64) -> [f64; 3] {
    [(x - k[2] as f64) / k[0] as f64, (y - k[3] as f64) / k[1] as f64, 1.0]
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// 결합 양방향 상향 표본: 저해상 5×5 창의 평면들을 고해상 광선에 옮겨 공간·밝기 가중 평균.
/// 법선은 같은 가중치의 성분 평균을 정규화하고 카메라 쪽으로 뒤집는다.
pub fn joint_bilateral_upsample(low: &ViewState, low_img: &LevelImage, high_img: &LevelImage, sigma_s: f32, sigma_c: f32) -> ViewState {
    let (w, h) = (high_img.width, high_img.height);
    let mut out = ViewState::new(w, h);
    let is2 = 1.0 / (2.0 * sigma_s as f64 * sigma_s as f64);
    let ic2 = 1.0 / (2.0 * sigma_c as f64 * sigma_c as f64);
    let rows: Vec<(Vec<f32>, Vec<[f32; 3]>)> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut dr = vec![0.0f32; w];
            let mut nr = vec![[0.0f32; 3]; w];
            for x in 0..w {
                let rp = ray(&high_img.k, x as f64, y as f64);
                let ip = high_img.gray[y * w + x] as f64 / 255.0;
                let (px, py) = ((x as f64 - 0.5) * 0.5, (y as f64 - 0.5) * 0.5);
                let (qx0, qy0) = (px.round() as i64, py.round() as i64);
                let (mut sw, mut sd) = (0.0f64, 0.0f64);
                let mut sn = [0.0f64; 3];
                let mut nearest: Option<(f64, f64, [f64; 3])> = None;
                for qy in qy0 - 2..=qy0 + 2 {
                    if qy < 0 || qy >= low.height as i64 {
                        continue;
                    }
                    for qx in qx0 - 2..=qx0 + 2 {
                        if qx < 0 || qx >= low.width as i64 {
                            continue;
                        }
                        let qi = qy as usize * low.width + qx as usize;
                        let dq = low.depth[qi] as f64;
                        if dq <= 0.0 {
                            continue;
                        }
                        let nq = low.normal[qi];
                        let n = [nq[0] as f64, nq[1] as f64, nq[2] as f64];
                        let Some(d) = plane_transfer(dq, n, ray(&low_img.k, qx as f64, qy as f64), rp) else { continue };
                        let ds = (qx as f64 - px).powi(2) + (qy as f64 - py).powi(2);
                        if nearest.is_none_or(|v| ds < v.0) {
                            nearest = Some((ds, d, n));
                        }
                        let iq = low_img.gray[qi] as f64 / 255.0;
                        let wgt = (-ds * is2 - (ip - iq) * (ip - iq) * ic2).exp();
                        sw += wgt;
                        sd += wgt * d;
                        for c in 0..3 {
                            sn[c] += wgt * n[c];
                        }
                    }
                }
                let (d, mut n) = if sw > 1e-12 {
                    (sd / sw, sn)
                } else if let Some((_, d, n)) = nearest {
                    (d, n)
                } else {
                    continue;
                };
                let l = dot(n, n).sqrt();
                if l < 1e-12 {
                    continue;
                }
                n = [n[0] / l, n[1] / l, n[2] / l];
                if dot(n, rp) >= 0.0 {
                    n = [-n[0], -n[1], -n[2]];
                }
                dr[x] = d as f32;
                nr[x] = [n[0] as f32, n[1] as f32, n[2] as f32];
            }
            (dr, nr)
        })
        .collect();
    for (y, (dr, nr)) in rows.into_iter().enumerate() {
        out.depth[y * w..(y + 1) * w].copy_from_slice(&dr);
        out.normal[y * w..(y + 1) * w].copy_from_slice(&nr);
    }
    out
}

/// 5×5 중앙값 평면 필터: 창 안 이웃 평면을 자기 광선에 옮긴 깊이들의 중앙(아래쪽 가운데) 평면을 채택.
pub fn median_plane_filter(s: &ViewState, img: &LevelImage) -> ViewState {
    let (w, h) = (s.width, s.height);
    let mut out = s.clone();
    let rows: Vec<(Vec<f32>, Vec<[f32; 3]>)> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut dr = s.depth[y * w..(y + 1) * w].to_vec();
            let mut nr = s.normal[y * w..(y + 1) * w].to_vec();
            let mut c: Vec<(f64, usize)> = Vec::with_capacity(25);
            for x in 0..w {
                if s.depth[y * w + x] <= 0.0 {
                    continue;
                }
                let rp = ray(&img.k, x as f64, y as f64);
                c.clear();
                for qy in y.saturating_sub(2)..(y + 3).min(h) {
                    for qx in x.saturating_sub(2)..(x + 3).min(w) {
                        let qi = qy * w + qx;
                        if s.depth[qi] <= 0.0 {
                            continue;
                        }
                        let nq = s.normal[qi];
                        let n = [nq[0] as f64, nq[1] as f64, nq[2] as f64];
                        if let Some(d) = plane_transfer(s.depth[qi] as f64, n, ray(&img.k, qx as f64, qy as f64), rp) {
                            c.push((d, qi));
                        }
                    }
                }
                if c.is_empty() {
                    continue;
                }
                c.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let (d, qi) = c[(c.len() - 1) / 2];
                dr[x] = d as f32;
                nr[x] = s.normal[qi];
            }
            (dr, nr)
        })
        .collect();
    for (y, (dr, nr)) in rows.into_iter().enumerate() {
        out.depth[y * w..(y + 1) * w].copy_from_slice(&dr);
        out.normal[y * w..(y + 1) * w].copy_from_slice(&nr);
    }
    out
}

/// 깊이맵 2배 축소(2×2 안의 양수 평균). 크기는 내림.
pub fn downsample_depth(d: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (nw, nh) = (w / 2, h / 2);
    let mut out = vec![0.0f32; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let (mut s, mut c) = (0.0f32, 0);
            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let v = d[(2 * y + dy) * w + 2 * x + dx];
                if v > 0.0 {
                    s += v;
                    c += 1;
                }
            }
            if c > 0 {
                out[y * nw + x] = s / c as f32;
            }
        }
    }
    (out, nw, nh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_and_intrinsics() {
        assert_eq!(num_levels(539, 3, 64), 3);
        assert_eq!(num_levels(240, 3, 64), 2);
        assert_eq!(num_levels(90, 3, 64), 1);
        let g = Arc::new(vec![100u8; 9 * 7]);
        let p = build_pyramid(&g, 9, 7, [10.0, 10.0, 4.0, 3.0], 2);
        assert_eq!((p[0].width, p[0].height, p[1].width), (4, 3, 9));
        assert!((p[0].k[2] - 1.75).abs() < 1e-6 && (p[0].k[0] - 5.0).abs() < 1e-6);
        assert!(p[0].gray.iter().all(|&v| v == 100));
    }

    #[test]
    fn upsample_reproduces_plane() {
        // 정면 평면 깊이 5 + 기울어진 평면: 상향 표본 깊이가 고해상 광선 교점과 일치.
        let hk = [40.0f32, 40.0, 15.5, 11.5];
        let lk = [20.0f32, 20.0, (15.5 + 0.5) * 0.5 - 0.5, (11.5 + 0.5) * 0.5 - 0.5];
        let n = {
            let v = [0.3f64, -0.2, -1.0];
            let l = dot(v, v).sqrt();
            [v[0] / l, v[1] / l, v[2] / l]
        };
        let hit = |r: [f64; 3]| -5.0 / dot(n, r); // n·X = −5
        let mut low = ViewState::new(16, 12);
        for y in 0..12 {
            for x in 0..16 {
                low.depth[y * 16 + x] = hit(ray(&lk, x as f64, y as f64)) as f32;
                low.normal[y * 16 + x] = [n[0] as f32, n[1] as f32, n[2] as f32];
            }
        }
        let li = LevelImage { width: 16, height: 12, gray: Arc::new(vec![50; 192]), k: lk };
        let hi = LevelImage { width: 32, height: 24, gray: Arc::new(vec![50; 768]), k: hk };
        let up = joint_bilateral_upsample(&low, &li, &hi, 1.0, 0.04);
        for y in 0..24 {
            for x in 0..32 {
                let t = hit(ray(&hk, x as f64, y as f64));
                assert!(((up.depth[y * 32 + x] as f64 - t) / t).abs() < 1e-5);
            }
        }
        let m = median_plane_filter(&up, &hi);
        assert!(((m.depth[100] as f64 - up.depth[100] as f64) / up.depth[100] as f64).abs() < 1e-5);
    }
}
