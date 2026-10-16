/*
 * stats.rs
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

//! Point cloud quality statistics: neighbor point spacing, outlier ratio (reprojection mismatch against other views' depth maps), ground
//! sample distance.

use crate::densify::DepthMapSet;
use crate::neighbors::PairStats;
use crate::scene::DenseScene;
use cumulus3d_core::io::PointCloud;
use rayon::prelude::*;
use std::collections::HashMap;

/// Point cloud statistics.
#[derive(Clone, Debug, Default)]
pub struct CloudStats {
    /// Number of points.
    pub points: usize,
    /// Number of sampled points.
    pub sampled: usize,
    /// Median nearest-neighbor distance (m, scene units).
    pub nn_spacing_median: f64,
    /// Ground sample distance (size covered by one pixel, median depth / focal length).
    pub gsd: f64,
    /// Outlier ratio: points supported by fewer than 2 of the overlapping views' depth maps with reprojection error ≤ tolerance.
    pub outlier_ratio: f64,
    /// Duplicate ratio: fraction of points that have another point within radius GSD/2.
    pub duplicate_ratio: f64,
    /// Local plane deviation ratio: points with fewer than 6 neighbors within radius 4·GSD (isolated) or with distance to the neighbors'
    /// principal plane > GSD.
    pub plane_outlier_ratio: f64,
    /// Median local plane distance (points with enough neighbors).
    pub plane_residual_median: f64,
}

/// Computes statistics. `vis` holds the contributing views per point (the first view is the reference view); `max_samples` points are
/// picked evenly.
pub fn cloud_stats(
    scene: &DenseScene,
    maps: &DepthMapSet,
    cloud: &PointCloud,
    vis: &[Vec<u32>],
    max_samples: usize,
    reproj_px: f64,
) -> CloudStats {
    let n = cloud.len();
    let mut st = CloudStats { points: n, ..Default::default() };
    if n == 0 {
        return st;
    }
    let mut gs: Vec<f64> = maps
        .maps
        .iter()
        .enumerate()
        .filter_map(|(v, m)| {
            let m = m.as_ref()?;
            let mut d: Vec<f64> = m.depth.iter().filter(|&&d| d > 0.0).step_by(97).map(|&d| d as f64).collect();
            if d.is_empty() {
                return None;
            }
            Some(crate::math::median_in_place(&mut d) / scene.views[v].k[0])
        })
        .collect();
    st.gsd = if gs.is_empty() { 0.0 } else { crate::math::median_in_place(&mut gs) };
    let step = (n / max_samples.max(1)).max(1);
    let idx: Vec<usize> = (0..n).step_by(step).collect();
    st.sampled = idx.len();
    // Nearest spacing (grid hash, cell = 2·GSD).
    let cell = (2.0 * st.gsd).max(1e-9);
    let key =
        |p: &[f32; 3]| ((p[0] as f64 / cell).floor() as i64, (p[1] as f64 / cell).floor() as i64, (p[2] as f64 / cell).floor() as i64);
    let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
    for (i, p) in cloud.positions.iter().enumerate() {
        grid.entry(key(p)).or_default().push(i as u32);
    }
    let nn: Vec<f64> = idx
        .par_iter()
        .filter_map(|&i| {
            let p = cloud.positions[i];
            let (a, b, c) = key(&p);
            let mut best = f64::INFINITY;
            for da in -1..=1 {
                for db in -1..=1 {
                    for dc in -1..=1 {
                        if let Some(v) = grid.get(&(a + da, b + db, c + dc)) {
                            for &j in v {
                                if j as usize == i {
                                    continue;
                                }
                                let q = cloud.positions[j as usize];
                                let d = ((p[0] - q[0]) as f64).powi(2) + ((p[1] - q[1]) as f64).powi(2) + ((p[2] - q[2]) as f64).powi(2);
                                best = best.min(d);
                            }
                        }
                    }
                }
            }
            best.is_finite().then(|| best.sqrt())
        })
        .collect();
    if !nn.is_empty() {
        let mut v = nn.clone();
        st.nn_spacing_median = crate::math::median_in_place(&mut v);
        st.duplicate_ratio = nn.iter().filter(|&&d| d < 0.5 * st.gsd).count() as f64 / idx.len() as f64;
    }
    // Local plane deviation (independent metric using only the point cloud).
    let cell4 = (4.0 * st.gsd).max(1e-9);
    let key4 =
        |p: &[f32; 3]| ((p[0] as f64 / cell4).floor() as i64, (p[1] as f64 / cell4).floor() as i64, (p[2] as f64 / cell4).floor() as i64);
    let mut grid4: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
    for (i, p) in cloud.positions.iter().enumerate() {
        grid4.entry(key4(p)).or_default().push(i as u32);
    }
    let r2 = cell4 * cell4;
    let res: Vec<Option<f64>> = idx
        .par_iter()
        .map(|&i| {
            let p = cloud.positions[i];
            let (a, b, c) = key4(&p);
            let mut nb: Vec<[f64; 3]> = Vec::new();
            for da in -1..=1 {
                for db in -1..=1 {
                    for dc in -1..=1 {
                        if let Some(v) = grid4.get(&(a + da, b + db, c + dc)) {
                            for &j in v {
                                if j as usize == i {
                                    continue;
                                }
                                let q = cloud.positions[j as usize];
                                let d = [(q[0] - p[0]) as f64, (q[1] - p[1]) as f64, (q[2] - p[2]) as f64];
                                if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= r2 {
                                    nb.push(d);
                                }
                            }
                        }
                    }
                }
            }
            if nb.len() < 6 {
                return None;
            }
            let n = nb.len() as f64;
            let mut m = [0.0; 3];
            for d in &nb {
                for k in 0..3 {
                    m[k] += d[k] / n;
                }
            }
            let mut cov = nalgebra::Matrix3::<f64>::zeros();
            for d in &nb {
                let v = nalgebra::Vector3::new(d[0] - m[0], d[1] - m[1], d[2] - m[2]);
                cov += v * v.transpose();
            }
            let e = nalgebra::SymmetricEigen::new(cov);
            let k = (0..3).min_by(|&x, &y| e.eigenvalues[x].total_cmp(&e.eigenvalues[y])).unwrap_or(0);
            let nrm = e.eigenvectors.column(k);
            // Distance from the point (origin) to the plane through the neighbor mean.
            Some((nrm[0] * m[0] + nrm[1] * m[1] + nrm[2] * m[2]).abs())
        })
        .collect();
    let mut resid: Vec<f64> = res.iter().flatten().copied().collect();
    let bad = res.iter().filter(|r| r.is_none_or(|v| v > st.gsd)).count();
    st.plane_outlier_ratio = bad as f64 / idx.len() as f64;
    if !resid.is_empty() {
        st.plane_residual_median = crate::math::median_in_place(&mut resid);
    }
    // Outliers.
    let stats = PairStats::new(scene);
    let overlap: Vec<Vec<usize>> = (0..scene.views.len()).map(|v| stats.select(v, 12, 0.0)).collect();
    let outl = idx
        .par_iter()
        .filter(|&&i| {
            let Some(&a) = vis.get(i).and_then(|v| v.first()) else { return false };
            let a = a as usize;
            let p = cloud.positions[i];
            let x = [p[0] as f64, p[1] as f64, p[2] as f64];
            let va = &scene.views[a];
            let ca = va.to_cam(&x);
            if ca[2] <= 0.0 {
                return true;
            }
            let (ua, wa, _) = va.project_cam(&ca);
            let mut support = 0;
            for &o in &overlap[a] {
                let Some(m) = maps.maps[o].as_ref() else { continue };
                let vo = &scene.views[o];
                let c = vo.to_cam(&x);
                if c[2] <= 0.0 {
                    continue;
                }
                let (u, v, _) = vo.project_cam(&c);
                let (ui, vi) = ((u + 0.5).floor(), (v + 0.5).floor());
                if ui < 0.0 || vi < 0.0 || ui >= vo.width as f64 || vi >= vo.height as f64 {
                    continue;
                }
                let d = m.depth[vi as usize * vo.width + ui as usize] as f64;
                if d <= 0.0 {
                    continue;
                }
                let r = vo.ray(ui, vi);
                let q = vo.to_world(&[r[0] * d, r[1] * d, d]);
                let cq = va.to_cam(&q);
                if cq[2] <= 0.0 {
                    continue;
                }
                let (uq, wq, _) = va.project_cam(&cq);
                if (uq - ua).powi(2) + (wq - wa).powi(2) <= reproj_px * reproj_px {
                    support += 1;
                }
            }
            support < 2
        })
        .count();
    st.outlier_ratio = outl as f64 / idx.len() as f64;
    st
}
