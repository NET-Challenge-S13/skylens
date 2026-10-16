/*
 * visibility.rs
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

//! Visibility (free-space) filter for a fused point cloud.
//!
//! A fused point that lies clearly in front of a surface another view observed along the same ray would
//! block that view's line of sight to a surface it actually saw, so it violates free space. Floating
//! blobs under objects and halos around edges are typically such points. For every point the filter
//! projects it into up to `check_views` neighbouring views and compares its depth `d_x` with the final
//! (filtered) depth `d_v` of the nearest pixel:
//!
//! - violation: `d_x < d_v · (1 − tau)` (the point is in front of the observed surface),
//! - support: `|d_x − d_v| ≤ tau · d_v`.
//!
//! A point is removed when it has at least `min_violations` violations and more violations than
//! supports. Views where the point falls outside the image, behind the camera or on an invalid depth
//! pixel do not vote. Decisions only read the depth maps, so the result does not depend on thread
//! scheduling.

use crate::fusion::{FusionInput, FusionOutput};
use crate::scene::{DenseScene, DenseView};
use rayon::prelude::*;

/// Visibility filter parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct VisibilityParams {
    /// Maximum number of neighbouring views checked per point (0 = off).
    pub check_views: usize,
    /// Relative depth tolerance `tau`.
    pub tau: f64,
    /// Minimum number of violating views for a point to be removed.
    pub min_violations: usize,
}

impl Default for VisibilityParams {
    fn default() -> Self {
        Self { check_views: 0, tau: 0.01, min_violations: 2 }
    }
}

impl VisibilityParams {
    /// Whether the filter runs.
    pub fn enabled(&self) -> bool {
        self.check_views > 0 && self.tau > 0.0 && self.min_violations > 0
    }
}

/// Depth of `x` in `view` and the nearest pixel index, or `None` when behind the camera or outside the image.
#[inline]
fn project_pixel(view: &DenseView, x: &[f64; 3]) -> Option<(f64, usize)> {
    let c = view.to_cam(x);
    if c[2] <= 0.0 {
        return None;
    }
    let col = (view.k[0] * c[0] / c[2] + view.k[2]).round();
    let row = (view.k[1] * c[1] / c[2] + view.k[3]).round();
    if col < 0.0 || row < 0.0 || col >= view.width as f64 || row >= view.height as f64 {
        return None;
    }
    Some((c[2], row as usize * view.width + col as usize))
}

/// Candidate views of one point: neighbour lists of its contributing views (first view first), deduplicated, at most `limit`.
fn candidates(vis: &[u32], overlap: &[Vec<usize>], limit: usize, out: &mut Vec<usize>) {
    out.clear();
    for &v in vis {
        let Some(list) = overlap.get(v as usize) else { continue };
        for &o in list {
            if out.len() >= limit {
                return;
            }
            if !out.contains(&o) {
                out.push(o);
            }
        }
    }
}

/// Removes free-space violating points from `out` (cloud, per-point visibility and residual flags stay aligned) and
/// returns the number removed. `overlap[v]` is the neighbour list of view `v` (the one fusion used). Does nothing when
/// the filter is off or the output has no per-point view lists.
pub fn filter_visibility(
    scene: &DenseScene,
    inputs: &[Option<FusionInput<'_>>],
    overlap: &[Vec<usize>],
    out: &mut FusionOutput,
    p: &VisibilityParams,
) -> usize {
    let n = out.cloud.len();
    if !p.enabled() || n == 0 || out.visibility.len() != n {
        return 0;
    }
    let keep: Vec<bool> = (0..n)
        .into_par_iter()
        .map_init(Vec::new, |cand, i| {
            let q = out.cloud.positions[i];
            let x = [q[0] as f64, q[1] as f64, q[2] as f64];
            candidates(&out.visibility[i], overlap, p.check_views, cand);
            let (mut violations, mut supports) = (0usize, 0usize);
            for &v in cand.iter() {
                let (Some(view), Some(Some(inp))) = (scene.views.get(v), inputs.get(v)) else { continue };
                let Some((dx, pix)) = project_pixel(view, &x) else { continue };
                let dv = inp.depth.get(pix).copied().unwrap_or(0.0) as f64;
                if dv <= 0.0 {
                    continue;
                }
                if dx < dv * (1.0 - p.tau) {
                    violations += 1;
                } else if (dx - dv).abs() <= p.tau * dv {
                    supports += 1;
                }
            }
            !(violations >= p.min_violations && violations > supports)
        })
        .collect();
    let removed = keep.iter().filter(|&&k| !k).count();
    if removed > 0 {
        retain_mask(&mut out.cloud.positions, &keep);
        retain_mask(&mut out.cloud.normals, &keep);
        retain_mask(&mut out.cloud.colors, &keep);
        retain_mask(&mut out.cloud.keys, &keep);
        retain_mask(&mut out.visibility, &keep);
        retain_mask(&mut out.residual, &keep);
    }
    removed
}

/// Keeps the entries whose mask is set; leaves vectors that are not point-aligned (empty or other length) untouched.
fn retain_mask<T>(v: &mut Vec<T>, keep: &[bool]) {
    if v.len() != keep.len() {
        return;
    }
    let mut it = keep.iter();
    v.retain(|_| it.next().copied().unwrap_or(true));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::{make_scene, SynthBox, SynthConfig, SynthScene};

    const PLANE_POINTS: usize = 5;

    /// A flat slab (top at z = 0) seen by a 3x3 camera grid. The first `PLANE_POINTS` points lie on the plane
    /// (from view 0's true depth), the rest float halfway between view 0 and the plane.
    fn setup() -> (SynthScene, FusionOutput, Vec<Vec<usize>>) {
        let cfg = SynthConfig { boxes: vec![SynthBox { min: [-30.0, -30.0, -1.0], max: [30.0, 30.0, 0.0] }], ..SynthConfig::default() };
        let s = make_scene(&cfg);
        let v0 = &s.scene.views[0];
        let mut out = FusionOutput::default();
        let pixels = [(150usize, 120usize), (190, 140), (230, 160), (170, 100), (210, 180)];
        for frac in [1.0, 0.5] {
            for &(col, row) in &pixels {
                let d = s.depth[0][row * v0.width + col] as f64;
                assert!(d > 0.0);
                let c = [(col as f64 - v0.k[2]) / v0.k[0] * d * frac, (row as f64 - v0.k[3]) / v0.k[1] * d * frac, d * frac];
                let x = v0.to_world(&c);
                out.cloud.positions.push([x[0] as f32, x[1] as f32, x[2] as f32]);
                out.cloud.colors.push([128, 128, 128]);
                out.visibility.push(vec![0]);
            }
        }
        let nv = s.scene.views.len();
        let overlap = (0..nv).map(|v| (0..nv).filter(|&o| o != v).collect()).collect();
        (s, out, overlap)
    }

    fn inputs(s: &SynthScene) -> Vec<Option<FusionInput<'_>>> {
        s.depth.iter().zip(&s.normal).map(|(d, n)| Some(FusionInput::plain(d, n))).collect()
    }

    #[test]
    fn removes_floaters_and_keeps_plane() {
        let (s, mut out, overlap) = setup();
        let inp = inputs(&s);
        let p = VisibilityParams { check_views: 8, tau: 0.01, min_violations: 2 };
        assert!(p.enabled());
        let removed = filter_visibility(&s.scene, &inp, &overlap, &mut out, &p);
        assert_eq!(removed, 5);
        assert_eq!(out.cloud.len(), PLANE_POINTS);
        assert_eq!(out.cloud.colors.len(), PLANE_POINTS);
        assert_eq!(out.visibility.len(), PLANE_POINTS);
        // Every survivor is a plane point (z ≈ 0).
        assert!(out.cloud.positions.iter().all(|q| q[2].abs() < 0.05), "{:?}", out.cloud.positions);
    }

    #[test]
    fn default_params_leave_cloud_unchanged() {
        let (s, mut out, overlap) = setup();
        let inp = inputs(&s);
        let before = out.cloud.positions.clone();
        assert!(!VisibilityParams::default().enabled());
        assert_eq!(filter_visibility(&s.scene, &inp, &overlap, &mut out, &VisibilityParams::default()), 0);
        assert_eq!(out.cloud.positions, before);
        assert_eq!(out.visibility.len(), before.len());
    }
}
