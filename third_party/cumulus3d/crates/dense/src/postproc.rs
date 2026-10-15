/*
 * postproc.rs
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

//! 필터 뒤 깊이맵 후처리(신뢰도 조건부).
//!
//! - **작은 조각 제거**: 4-이웃 사이 상대 깊이 차가 문턱 이하면 같은 성분. 넓이가 영상 넓이 비율보다 작고
//!   비용 중앙값이 신뢰 문턱보다 큰 성분을 지운다(작아도 확실한 조각은 남긴다).
//! - **경계 인식 틈 메우기**: 무효 픽셀에서 8방향으로 최대 `대각선 × 비율` 까지 걷다가 밝기 차가 문턱을 넘으면(영상 경계) 멈춘다.
//!   처음 만나는 유효 픽셀들 중 밝기가 가장 비슷한 것의 평면을 자기 광선에 옮겨 채운다. 자기 PatchMatch 비용이
//!   문턱 이하인 픽셀만 채우고, 원래 유효했던 픽셀만 출처로 쓴다(연쇄 채움 없음).

use crate::math::plane_transfer;

/// 후처리 매개변수.
#[derive(Clone, Debug, PartialEq)]
pub struct PostParams {
    /// 작은 조각 제거를 할지.
    pub remove_speckles: bool,
    /// 같은 성분으로 볼 상대 깊이 차.
    pub speckle_rel_depth: f32,
    /// 최소 성분 넓이(영상 넓이 비율).
    pub speckle_min_area_frac: f64,
    /// 비용 중앙값이 이 값 이하인 성분은 작아도 남긴다.
    pub speckle_keep_cost: f32,
    /// 경계 인식 틈 메우기를 할지.
    pub fill_holes: bool,
    /// 최대 채움 거리(영상 대각선 비율).
    pub fill_max_frac: f64,
    /// 걷기를 멈추는 인접 밝기 차(\[0,1\]).
    pub fill_edge_step: f32,
    /// 채울 픽셀의 자기 비용 상한.
    pub fill_max_cost: f32,
}

impl Default for PostParams {
    fn default() -> Self {
        Self {
            remove_speckles: true,
            speckle_rel_depth: 0.02,
            speckle_min_area_frac: 2e-4,
            speckle_keep_cost: 0.2,
            fill_holes: false,
            fill_max_frac: 0.01,
            fill_edge_step: 0.06,
            fill_max_cost: 1.0,
        }
    }
}

/// 작은 조각 제거. 지운 픽셀 수를 돌려준다.
pub fn remove_speckles(depth: &mut [f32], normal: &mut [[f32; 3]], cost: &[f32], w: usize, h: usize, p: &PostParams) -> usize {
    let min_area = ((w * h) as f64 * p.speckle_min_area_frac).ceil() as usize;
    if min_area <= 1 {
        return 0;
    }
    let mut label = vec![u32::MAX; w * h];
    let mut stack = Vec::new();
    let mut comp = Vec::new();
    let mut costs = Vec::new();
    let mut removed = 0;
    for start in 0..w * h {
        if depth[start] <= 0.0 || label[start] != u32::MAX {
            continue;
        }
        comp.clear();
        stack.push(start);
        label[start] = start as u32;
        while let Some(i) = stack.pop() {
            comp.push(i);
            let (x, y) = (i % w, i / w);
            let di = depth[i];
            let mut nb = [usize::MAX; 4];
            if x > 0 {
                nb[0] = i - 1;
            }
            if x + 1 < w {
                nb[1] = i + 1;
            }
            if y > 0 {
                nb[2] = i - w;
            }
            if y + 1 < h {
                nb[3] = i + w;
            }
            for j in nb {
                if j == usize::MAX || label[j] != u32::MAX || depth[j] <= 0.0 {
                    continue;
                }
                if (depth[j] - di).abs() <= p.speckle_rel_depth * di.min(depth[j]) {
                    label[j] = start as u32;
                    stack.push(j);
                }
            }
        }
        if comp.len() < min_area {
            costs.clear();
            costs.extend(comp.iter().map(|&i| cost[i]));
            costs.sort_by(|a, b| a.total_cmp(b));
            if costs[costs.len() / 2] > p.speckle_keep_cost {
                for &i in &comp {
                    depth[i] = 0.0;
                    normal[i] = [0.0; 3];
                }
                removed += comp.len();
            }
        }
    }
    removed
}

/// 경계 인식 틈 메우기. 채운 픽셀 수를 돌려준다.
#[allow(clippy::too_many_arguments)]
pub fn fill_holes(
    depth: &mut [f32],
    normal: &mut [[f32; 3]],
    cost: &mut [f32],
    raw_cost: &[f32],
    gray: &[u8],
    k: [f64; 4],
    w: usize,
    h: usize,
    p: &PostParams,
) -> usize {
    let maxd = ((((w * w + h * h) as f64).sqrt() * p.fill_max_frac).round() as i64).max(1);
    let valid: Vec<bool> = depth.iter().map(|&d| d > 0.0).collect();
    let ray = |x: f64, y: f64| [(x - k[2]) / k[0], (y - k[3]) / k[1], 1.0];
    let step = (p.fill_edge_step * 255.0) as i32;
    let mut fills = Vec::new();
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let i = (y as usize) * w + x as usize;
            if valid[i] || raw_cost[i] > p.fill_max_cost {
                continue;
            }
            let ip = gray[i] as i32;
            let mut best: Option<(i32, i64, usize)> = None;
            for (dx, dy) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)] {
                let mut prev = ip;
                for s in 1..=maxd {
                    let (qx, qy) = (x + dx * s, y + dy * s);
                    if qx < 0 || qy < 0 || qx >= w as i64 || qy >= h as i64 {
                        break;
                    }
                    let q = qy as usize * w + qx as usize;
                    let iq = gray[q] as i32;
                    if (iq - prev).abs() > step {
                        break;
                    }
                    prev = iq;
                    if valid[q] {
                        let key = ((iq - ip).abs(), s, q);
                        if best.is_none_or(|b| (key.0, key.1) < (b.0, b.1)) {
                            best = Some(key);
                        }
                        break;
                    }
                }
            }
            if let Some((_, _, q)) = best {
                let n = normal[q];
                let nd = [n[0] as f64, n[1] as f64, n[2] as f64];
                let (qx, qy) = ((q % w) as f64, (q / w) as f64);
                if let Some(d) = plane_transfer(depth[q] as f64, nd, ray(qx, qy), ray(x as f64, y as f64)) {
                    fills.push((i, d as f32, n, cost[q]));
                }
            }
        }
    }
    for &(i, d, n, c) in &fills {
        depth[i] = d;
        normal[i] = n;
        cost[i] = c;
    }
    fills.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speckle_and_fill() {
        let (w, h) = (40, 30);
        let k = [50.0, 50.0, 19.5, 14.5];
        let mut d = vec![10.0f32; w * h];
        let mut n = vec![[0.0f32, 0.0, -1.0]; w * h];
        let mut c = vec![0.5f32; w * h];
        // 떠 있는 작은 조각(깊이 5) 2×2, 구멍 3×3.
        for (x, y) in [(5, 5), (6, 5), (5, 6), (6, 6)] {
            d[y * w + x] = 5.0;
        }
        for y in 20..23 {
            for x in 20..23 {
                d[y * w + x] = 0.0;
            }
        }
        let p = PostParams { speckle_min_area_frac: 10.0 / (w * h) as f64, fill_holes: true, fill_max_frac: 0.1, ..Default::default() };
        assert_eq!(remove_speckles(&mut d, &mut n, &c, w, h, &p), 4);
        assert_eq!(d[5 * w + 5], 0.0);
        let gray = vec![100u8; w * h];
        let raw = vec![0.3f32; w * h];
        let filled = fill_holes(&mut d, &mut n, &mut c, &raw, &gray, k, w, h, &p);
        assert_eq!(filled, 13);
        assert!((d[21 * w + 21] - 10.0).abs() < 1e-4);
    }
}
