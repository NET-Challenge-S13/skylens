/*
 * smooth.rs
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

//! Normal-direction bilateral smoothing of a fused point cloud (surface consolidation).
//!
//! Fused points of one surface come from different reference pixels whose depths carry independent
//! noise, so a wall shows up as a slab a few millimetres thick. Each point is moved only along its own
//! normal by a bilateral average of its neighbours' signed offsets along that normal
//! (point-set version of the bilateral surface filter: Fleishman et al. 2003, Jones et al. 2003;
//! Digne & de Franchis 2017 for point clouds):
//!
//! `p' = p + n · Σ w_j d_j / Σ w_j`, `d_j = n·(p_j − p)`,
//! `w_j = exp(−|p_j − p|²/(2σ_s²)) · exp(−d_j²/(2σ_r²))`, neighbours within radius `r` whose normal is
//! within `normal_deg` of `n` (the point itself enters with `d = 0`, `w = 1`).
//!
//! The spatial term keeps the average local, the range term keeps points on the other side of a step
//! from being pulled across it, and the normal gate keeps corners and edges. Points never move
//! tangentially, so coverage and point count are unchanged. Updates are Jacobi style (all points read
//! the previous iteration), so the result does not depend on thread scheduling.

use cumulus3d_core::io::PointCloud;
use rayon::prelude::*;
use std::collections::HashMap;

/// Smoothing parameters. Distances are multiples of the depth-map ground sample distance (GSD).
#[derive(Clone, Debug, PartialEq)]
pub struct SmoothParams {
    /// Number of iterations (0 = off).
    pub iterations: usize,
    /// Neighbour radius `r` (GSD multiple). The spatial σ is `r / 2`.
    pub radius_gsd: f64,
    /// Range σ along the normal (GSD multiple).
    pub sigma_range_gsd: f64,
    /// Neighbours whose normal deviates more than this (degrees) are ignored.
    pub normal_deg: f64,
}

impl Default for SmoothParams {
    fn default() -> Self {
        Self { iterations: 0, radius_gsd: 1.0, sigma_range_gsd: 0.25, normal_deg: 30.0 }
    }
}

impl SmoothParams {
    /// Whether smoothing runs.
    pub fn enabled(&self) -> bool {
        self.iterations > 0 && self.radius_gsd > 0.0 && self.sigma_range_gsd > 0.0
    }
}

/// Hasher for packed cell keys (keys are already mixed, so the hash is the key itself).
#[derive(Default, Clone, Copy)]
struct KeyHasher(u64);

impl std::hash::Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 << 8) ^ b as u64;
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
}

type KeyMap = HashMap<u64, (u32, u32), std::hash::BuildHasherDefault<KeyHasher>>;

fn cell_key(c: [i32; 3]) -> u64 {
    let k = ((c[0] as u32 as u64 & 0x1f_ffff) << 42) | ((c[1] as u32 as u64 & 0x1f_ffff) << 21) | (c[2] as u32 as u64 & 0x1f_ffff);
    crate::math::mix64(k)
}

/// Points reordered by grid cell (side = neighbour radius) for cache-friendly neighbour scans.
struct Grid {
    /// Original index of each sorted point.
    order: Vec<u32>,
    /// Sorted positions (f64) and unit normals.
    pos: Vec<[f64; 3]>,
    nrm: Vec<[f64; 3]>,
    /// Cell of each run of sorted points: (cell, start, end).
    runs: Vec<([i32; 3], u32, u32)>,
    /// Cell key → range of the sorted list.
    map: KeyMap,
}

impl Grid {
    fn new(pos: &[[f32; 3]], nrm: &[[f64; 3]], cell: f64) -> Self {
        let cell_of = |x: &[f32; 3]| x.map(|a| (a as f64 / cell).floor() as i32);
        let mut keyed: Vec<([i32; 3], u32)> = pos.par_iter().enumerate().map(|(i, x)| (cell_of(x), i as u32)).collect();
        keyed.par_sort_unstable();
        let mut runs = Vec::new();
        let mut s = 0usize;
        while s < keyed.len() {
            let k = keyed[s].0;
            let mut e = s + 1;
            while e < keyed.len() && keyed[e].0 == k {
                e += 1;
            }
            runs.push((k, s as u32, e as u32));
            s = e;
        }
        let mut map = KeyMap::with_capacity_and_hasher(runs.len(), Default::default());
        for &(c, s, e) in &runs {
            map.insert(cell_key(c), (s, e));
        }
        let order: Vec<u32> = keyed.into_iter().map(|(_, i)| i).collect();
        let spos = order.par_iter().map(|&i| pos[i as usize].map(|a| a as f64)).collect();
        let snrm = order.par_iter().map(|&i| nrm[i as usize]).collect();
        Self { order, pos: spos, nrm: snrm, runs, map }
    }
}

/// Smooths `cloud` in place along its normals. `gsd` is the depth-map ground sample distance in scene
/// units. Returns the median absolute displacement of the last iteration (scene units), or 0 when
/// nothing ran (smoothing off, no normals, or `gsd` ≤ 0).
pub fn smooth_along_normals(cloud: &mut PointCloud, gsd: f64, p: &SmoothParams) -> f64 {
    let n = cloud.positions.len();
    if !p.enabled() || gsd <= 0.0 || cloud.normals.len() != n || n == 0 {
        return 0.0;
    }
    assert!(n < u32::MAX as usize, "smoothing: too many points for u32 indices");
    let r = p.radius_gsd * gsd;
    let r2 = r * r;
    let inv_2ss = 1.0 / (2.0 * (0.5 * r).powi(2));
    let inv_2sr = 1.0 / (2.0 * (p.sigma_range_gsd * gsd).powi(2));
    let cos_n = p.normal_deg.to_radians().cos();
    let normals: Vec<[f64; 3]> = cloud
        .normals
        .par_iter()
        .map(|m| {
            let m = [m[0] as f64, m[1] as f64, m[2] as f64];
            let l = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
            if l > 1e-12 {
                m.map(|a| a / l)
            } else {
                [0.0; 3]
            }
        })
        .collect();
    let mut last_disp = 0.0;
    for _ in 0..p.iterations {
        let g = Grid::new(&cloud.positions, &normals, r);
        // Per cell run: gather the 27 neighbour ranges once, then scan them for every point of the cell.
        let shifts: Vec<Vec<f64>> = g
            .runs
            .par_iter()
            .map(|&(c, s, e)| {
                let mut nb: Vec<(u32, u32)> = Vec::with_capacity(27);
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if let Some(&rg) = g.map.get(&cell_key([c[0] + dx, c[1] + dy, c[2] + dz])) {
                                nb.push(rg);
                            }
                        }
                    }
                }
                (s..e)
                    .map(|i| {
                        let i = i as usize;
                        let (pi, ni) = (g.pos[i], g.nrm[i]);
                        if ni == [0.0; 3] {
                            return 0.0;
                        }
                        let (mut sw, mut swd) = (1.0f64, 0.0f64);
                        for &(a, b) in &nb {
                            for j in a as usize..b as usize {
                                let q = g.pos[j];
                                let dv = [q[0] - pi[0], q[1] - pi[1], q[2] - pi[2]];
                                let d2 = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
                                if d2 > r2 || j == i {
                                    continue;
                                }
                                let nj = g.nrm[j];
                                if ni[0] * nj[0] + ni[1] * nj[1] + ni[2] * nj[2] < cos_n {
                                    continue;
                                }
                                let dn = ni[0] * dv[0] + ni[1] * dv[1] + ni[2] * dv[2];
                                let w = (-d2 * inv_2ss - dn * dn * inv_2sr).exp();
                                sw += w;
                                swd += w * dn;
                            }
                        }
                        swd / sw
                    })
                    .collect()
            })
            .collect();
        let mut disp: Vec<f64> = Vec::with_capacity(n);
        for (k, t) in shifts.into_iter().flatten().enumerate() {
            let (pi, ni) = (g.pos[k], g.nrm[k]);
            cloud.positions[g.order[k] as usize] = [(pi[0] + t * ni[0]) as f32, (pi[1] + t * ni[1]) as f32, (pi[2] + t * ni[2]) as f32];
            disp.push(t.abs());
        }
        last_disp = crate::math::median_in_place(&mut disp);
    }
    last_disp
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noisy_plane(n: usize, noise: f64) -> PointCloud {
        let mut pc = PointCloud::default();
        let mut s = 0x1234_5678u64;
        let mut rnd = || {
            s = crate::math::mix64(s.wrapping_add(0x9e37_79b9_7f4a_7c15));
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..n {
            let (x, y) = (rnd() * 4.0, rnd() * 4.0);
            let z = (rnd() - 0.5) * 2.0 * noise;
            pc.positions.push([x as f32, y as f32, z as f32]);
            pc.normals.push([0.0, 0.0, 1.0]);
            pc.colors.push([0, 0, 0]);
        }
        pc
    }

    fn rms_z(pc: &PointCloud) -> f64 {
        (pc.positions.iter().map(|p| (p[2] as f64).powi(2)).sum::<f64>() / pc.len() as f64).sqrt()
    }

    #[test]
    fn thins_a_noisy_plane_without_moving_tangentially() {
        let mut pc = noisy_plane(20_000, 0.01);
        let before = pc.clone();
        let p = SmoothParams { iterations: 1, radius_gsd: 4.0, sigma_range_gsd: 1.0, normal_deg: 30.0 };
        let d = smooth_along_normals(&mut pc, 0.05, &p);
        assert!(d > 0.0);
        assert!(rms_z(&pc) < 0.5 * rms_z(&before), "{} vs {}", rms_z(&pc), rms_z(&before));
        for (a, b) in pc.positions.iter().zip(&before.positions) {
            assert_eq!((a[0], a[1]), (b[0], b[1]));
        }
    }

    #[test]
    fn keeps_a_step_and_respects_normals() {
        // Two parallel sheets 0.5 apart: the range term must keep them apart.
        let mut pc = noisy_plane(10_000, 0.0);
        let mut upper = noisy_plane(10_000, 0.0);
        for q in upper.positions.iter_mut() {
            q[2] = 0.5;
        }
        pc.positions.extend(upper.positions);
        pc.normals.extend(upper.normals);
        pc.colors.extend(upper.colors);
        let p = SmoothParams { iterations: 2, radius_gsd: 12.0, sigma_range_gsd: 0.5, normal_deg: 30.0 };
        smooth_along_normals(&mut pc, 0.05, &p);
        assert!(pc.positions[..10_000].iter().all(|q| q[2].abs() < 1e-4));
        assert!(pc.positions[10_000..].iter().all(|q| (q[2] - 0.5).abs() < 1e-4));
        // Off: untouched.
        let mut c = noisy_plane(100, 0.01);
        let b = c.clone();
        assert_eq!(smooth_along_normals(&mut c, 0.05, &SmoothParams::default()), 0.0);
        assert_eq!(c.positions, b.positions);
    }
}
