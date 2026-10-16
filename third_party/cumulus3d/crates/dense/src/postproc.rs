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

//! Post-processing of depth maps after the filter (confidence-conditional).
//!
//! - **Small component removal**: pixels whose relative depth difference between 4-neighbors is at or below a threshold form one component.
//!   Components whose area is below an image-area ratio and whose median cost is above a confidence threshold are erased (small but certain
//!   pieces are kept).
//! - **Boundary-aware gap filling**: from an invalid pixel, walk in 8 directions up to `diagonal × ratio`; stop when the brightness
//!   difference exceeds a threshold (image boundary). The plane of the valid pixel with the most similar brightness among the first ones
//!   met is moved onto the own ray to fill. Only pixels whose own PatchMatch cost is at or below a threshold are filled, and only
//!   originally valid pixels are used as sources (no chained filling).

use crate::math::plane_transfer;

/// Post-processing parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct PostParams {
    /// Whether to remove small components.
    pub remove_speckles: bool,
    /// Relative depth difference to regard as the same component.
    pub speckle_rel_depth: f32,
    /// Minimum component area (fraction of image area).
    pub speckle_min_area_frac: f64,
    /// Components whose median cost is at or below this value are kept even if small.
    pub speckle_keep_cost: f32,
    /// Whether to do boundary-aware gap filling.
    pub fill_holes: bool,
    /// Maximum fill distance (fraction of image diagonal).
    pub fill_max_frac: f64,
    /// Adjacent brightness difference at which walking stops (\[0,1\]).
    pub fill_edge_step: f32,
    /// Upper bound of the own cost of a pixel to fill.
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

/// Small component removal. Returns the number of erased pixels.
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

/// Boundary-aware gap filling. Returns the number of filled pixels.
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
        // Floating small component (depth 5) 2×2, hole 3×3.
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
