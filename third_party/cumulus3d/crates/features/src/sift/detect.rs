/*
 * detect.rs
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

//! DoG 극값 검출·엣지 억제·부분화소 보정.

use rayon::prelude::*;

/// 검출된 점(옥타브 화소 단위).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// 옥타브 내 좌표(화소 중심 .5 규칙 포함): c + 0.5 + δx.
    pub x: f32,
    /// 옥타브 내 y 좌표(x 와 같은 규칙).
    pub y: f32,
    /// 옥타브 내 σ_o = σ₀ k^(j + δs).
    pub sigma: f32,
    /// 보정된 DoG 값 v + ½ g·δ (특이 행렬이면 v).
    pub response: f32,
}

/// 검출 매개변수.
#[derive(Clone, Copy, Debug)]
pub struct DetectParams {
    /// DoG 극값 임계값.
    pub peak_threshold: f32,
    /// 엣지 억제 임계값.
    pub edge_threshold: f32,
    /// 1 = 1회(이동 없음). >1 이면 |δ|>0.6 일 때 화소 이동하며 재보정.
    pub refinement_iterations: usize,
    /// 특이 헤시안일 때 δ=0 통과 대신 버림.
    pub reject_singular: bool,
    /// 기준 σ₀.
    pub sigma0: f32,
    /// 레벨 간 배율 k = 2^(1/S).
    pub k: f32,
}

#[allow(clippy::needless_range_loop)]
/// 3×3 부분 피벗 가우스 소거. 피벗 크기 < 1e−10 이면 None.
fn solve3(mut a: [[f32; 3]; 3], mut b: [f32; 3]) -> Option<[f32; 3]> {
    for col in 0..3 {
        let mut piv = col;
        for r in col + 1..3 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-10 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in col + 1..3 {
            let f = a[r][col] / a[col][col];
            for c in col..3 {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0f32; 3];
    for r in (0..3).rev() {
        let mut s = b[r];
        for c in r + 1..3 {
            s -= a[r][c] * x[c];
        }
        x[r] = s / a[r][r];
    }
    Some(x)
}

/// 검출 레벨 j 에서 극값 검출. `dm, d0, dp` = D_j, D_{j+1}, D_{j+2} (아래·현재·위).
/// 결과는 행 우선(결정적) 순서.
#[allow(clippy::needless_range_loop)]
pub fn detect_level(dm: &[f32], d0: &[f32], dp: &[f32], w: usize, h: usize, j: usize, p: &DetectParams) -> Vec<Candidate> {
    if w < 3 || h < 3 {
        return Vec::new();
    }
    let pre = 0.8 * p.peak_threshold;
    let e = p.edge_threshold;
    let edge_ratio = (e + 1.0) * (e + 1.0) / e;
    (1..h - 1)
        .into_par_iter()
        .flat_map_iter(|r| {
            let mut out = Vec::new();
            let row = &d0[r * w..(r + 1) * w];
            for c in 1..w - 1 {
                let v = row[c];
                if v.abs() <= pre {
                    continue;
                }
                if !is_extremum(dm, d0, dp, w, r, c, v) {
                    continue;
                }
                let i = r * w + c;
                // 엣지 억제(같은 레벨 3×3).
                let fxx = d0[i - 1] + d0[i + 1] - 2.0 * v;
                let fyy = d0[i - w] + d0[i + w] - 2.0 * v;
                let fxy = 0.25 * (d0[i + w + 1] + d0[i - w - 1] - d0[i + w - 1] - d0[i - w + 1]);
                let det = fxx * fyy - fxy * fxy;
                let tr = fxx + fyy;
                if det <= 0.0 || tr * tr > edge_ratio * det {
                    continue;
                }
                if let Some(cand) = refine(dm, d0, dp, w, h, r, c, j, p) {
                    out.push(cand);
                }
            }
            out
        })
        .collect()
}

#[inline]
fn is_extremum(dm: &[f32], d0: &[f32], dp: &[f32], w: usize, r: usize, c: usize, v: f32) -> bool {
    let i = r * w + c;
    let (l, rt) = (d0[i - 1], d0[i + 1]);
    let is_max = if v > l.max(rt) {
        true
    } else if v < l.min(rt) {
        false
    } else {
        return false;
    };
    // 나머지 24 이웃: 같은 값 허용.
    let others = |img: &[f32], skip_lr: bool| -> bool {
        for dy in [-1isize, 0, 1] {
            let base = (i as isize + dy * w as isize) as usize;
            for dx in [-1isize, 0, 1] {
                if skip_lr && dy == 0 {
                    continue;
                }
                let n = img[(base as isize + dx) as usize];
                if is_max {
                    if v < n {
                        return false;
                    }
                } else if v > n {
                    return false;
                }
            }
        }
        true
    };
    others(d0, true) && others(dm, false) && others(dp, false)
}

#[allow(clippy::too_many_arguments)]
fn refine(dm: &[f32], d0: &[f32], dp: &[f32], w: usize, h: usize, r0: usize, c0: usize, j: usize, p: &DetectParams) -> Option<Candidate> {
    let (mut r, mut c) = (r0, c0);
    let iters = p.refinement_iterations.max(1);
    for it in 0..iters {
        let i = r * w + c;
        let v = d0[i];
        let fx = 0.5 * (d0[i + 1] - d0[i - 1]);
        let fy = 0.5 * (d0[i + w] - d0[i - w]);
        let fs = 0.5 * (dp[i] - dm[i]);
        let fxx = d0[i - 1] + d0[i + 1] - 2.0 * v;
        let fyy = d0[i - w] + d0[i + w] - 2.0 * v;
        let fss = dp[i] + dm[i] - 2.0 * v;
        let fxy = 0.25 * (d0[i + w + 1] + d0[i - w - 1] - d0[i + w - 1] - d0[i - w + 1]);
        let fxs = 0.25 * (dp[i + 1] + dm[i - 1] - dp[i - 1] - dm[i + 1]);
        let fys = 0.25 * (dp[i + w] + dm[i - w] - dp[i - w] - dm[i + w]);
        let hm = [[fxx, fxy, fxs], [fxy, fyy, fys], [fxs, fys, fss]];
        let g = [fx, fy, fs];
        let sol = solve3(hm, [-fx, -fy, -fs]);
        let (delta, resp) = match sol {
            None => {
                if p.reject_singular {
                    return None;
                }
                ([0f32; 3], v)
            }
            Some(d) => {
                // 반복 보정(개선 옵션): 화소 이동 후 다시 푼다.
                if it + 1 < iters {
                    let mut moved = false;
                    if d[0] > 0.6 && c + 2 < w {
                        c += 1;
                        moved = true;
                    } else if d[0] < -0.6 && c > 1 {
                        c -= 1;
                        moved = true;
                    }
                    if d[1] > 0.6 && r + 2 < h {
                        r += 1;
                        moved = true;
                    } else if d[1] < -0.6 && r > 1 {
                        r -= 1;
                        moved = true;
                    }
                    if moved {
                        continue;
                    }
                }
                let resp = v + 0.5 * (g[0] * d[0] + g[1] * d[1] + g[2] * d[2]);
                if resp.abs() <= p.peak_threshold || d[0].abs() >= 1.0 || d[1].abs() >= 1.0 || d[2].abs() >= 1.0 {
                    return None;
                }
                (d, resp)
            }
        };
        return Some(Candidate {
            x: c as f32 + 0.5 + delta[0],
            y: r as f32 + 0.5 + delta[1],
            sigma: p.sigma0 * p.k.powf(j as f32 + delta[2]),
            response: resp,
        });
    }
    None
}

/// DoG D_d = G_{d} − G_{d−1} (배열 색인: `gauss[d+1] − gauss[d]`).
pub fn dog(a: &[f32], b: &[f32], out: &mut [f32]) {
    out.par_chunks_mut(4096).zip(a.par_chunks(4096).zip(b.par_chunks(4096))).for_each(|(o, (hi, lo))| {
        for ((o, x), y) in o.iter_mut().zip(hi).zip(lo) {
            *o = x - y;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve3_basic() {
        let a = [[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]];
        let x = [1.0f32, -2.0, 0.5];
        let b = [2.0 * 1.0 - 2.0, 1.0 - 6.0 + 0.5, -2.0 + 2.0];
        let s = solve3(a, b).unwrap();
        for i in 0..3 {
            assert!((s[i] - x[i]).abs() < 1e-5);
        }
        assert!(solve3([[0.0; 3]; 3], [1.0; 3]).is_none());
    }
}
