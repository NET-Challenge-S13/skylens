/*
 * densify.rs
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

//! Densification progression: neighbors and depth range → pyramid → per-scale runs (photometric, detail restoration, geometric rounds) →
//! median filter → readout and filter → fusion.

use crate::cache::{CachedDepth, DepthMapCache};
use crate::fusion::{fuse, FusionInput, FusionOutput};
use crate::fusion_score::{fuse_scored, ScoreFusionOptions};
use crate::kernel::{DepthSnapshot, KernelInput, KernelView, PairGeometry, PatchMatchBackend, RunParams, ViewState};
use crate::neighbors::{depth_ranges, PairStats};
use crate::params::DensifyOptions;
use crate::scene::DenseScene;
use crate::upsample::{build_pyramid, downsample_depth, num_levels};
use cumulus3d_core::io::{write_ply, PlyLayout, PointCloud};
use cumulus3d_core::{Error, Result};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Per-stage timings.
#[derive(Clone, Debug, Default)]
pub struct DenseTimings {
    /// Neighbor selection + depth range.
    pub neighbors: Duration,
    /// Pyramid construction + backend session start (upload).
    pub prepare: Duration,
    /// Per-scale depth estimation (kernel runs + upsampling + detail-restoration decision). 0 = lowest scale.
    pub levels: Vec<Duration>,
    /// Of that, the total CPU-side upsampling and detail decision time.
    pub upsample: Duration,
    /// Median filter + readout and filter.
    pub filter: Duration,
    /// Fusion.
    pub fusion: Duration,
    /// Of that, agreement fusion first and second pass times (the second is nonzero only when the second fusion ran).
    pub fusion_pass1: Duration,
    /// Agreement fusion second pass time.
    pub fusion_pass2: Duration,
    /// Of the fusion time, normal-direction smoothing of the fused cloud (0 when off).
    pub fusion_smooth: Duration,
    /// Of the fusion time, free-space visibility filtering of the fused cloud (0 when off).
    pub fusion_visibility: Duration,
    /// Points removed by the visibility filter.
    pub fusion_visibility_removed: usize,
}

impl DenseTimings {
    /// Total depth estimation.
    pub fn depth(&self) -> Duration {
        self.levels.iter().sum()
    }
    /// Total time (neighbors + preparation + depth + filter + fusion).
    pub fn total(&self) -> Duration {
        self.neighbors + self.prepare + self.depth() + self.filter + self.fusion
    }
}

/// One final depth map (view resolution).
#[derive(Clone, Debug)]
pub struct DepthMapResult {
    /// Width (pixels).
    pub width: usize,
    /// Height (pixels).
    pub height: usize,
    /// Final depth before filtering.
    pub raw_depth: Vec<f32>,
    /// Depth after the filter (0 = invalid).
    pub depth: Vec<f32>,
    /// Normal in the reference camera coordinates.
    pub normal: Vec<[f32; 3]>,
    /// Final aggregated cost (confidence).
    pub cost: Vec<f32>,
    /// Whether it came from the depth map cache.
    pub from_cache: bool,
}

/// Depth map computation result.
#[derive(Clone, Debug, Default)]
pub struct DepthMapSet {
    /// Depth map per view (None for views that failed to produce one).
    pub maps: Vec<Option<Arc<DepthMapResult>>>,
    /// Source views per view.
    pub sources: Vec<Vec<usize>>,
    /// Representative baseline per view (median source center distance).
    pub baselines: Vec<f64>,
    /// Number of cache-hit views.
    pub cache_hits: usize,
    /// Number of scales used.
    pub num_levels: usize,
    /// Per-stage timings.
    pub timings: DenseTimings,
}

/// Densification result.
#[derive(Clone, Debug, Default)]
pub struct DenseOutput {
    /// Dense point cloud (position, normal, color).
    pub cloud: PointCloud,
    /// Contributing view indices per point.
    pub visibility: Vec<Vec<u32>>,
    /// Per-stage timings.
    pub timings: DenseTimings,
    /// Number of cache-hit views.
    pub cache_hits: usize,
    /// Number of views for which a depth map was built.
    pub depth_views: usize,
    /// Depth maps (for statistics and diagnostics).
    pub depth_maps: DepthMapSet,
    /// Whether each point is a second-fusion point (empty if no second fusion was done).
    pub residual: Vec<bool>,
}

impl DenseOutput {
    /// Writes PLY (x y z nx ny nz red green blue). Creates the parent folder.
    pub fn write_ply(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(p) = path.parent() {
            if !p.as_os_str().is_empty() {
                std::fs::create_dir_all(p)?;
            }
        }
        write_ply(path, &self.cloud, PlyLayout::XyzNormalRgb)
    }

    /// Number of second-fusion points.
    pub fn num_residual(&self) -> usize {
        self.residual.iter().filter(|&&r| r).count()
    }

    /// Writes only the second-fusion points as PLY (returns false without writing if there are none).
    pub fn write_residual_ply(&self, path: impl AsRef<Path>) -> Result<bool> {
        if self.num_residual() == 0 {
            return Ok(false);
        }
        let mut pc = PointCloud::default();
        for (i, _) in self.residual.iter().enumerate().filter(|(_, &r)| r) {
            pc.positions.push(self.cloud.positions[i]);
            if let Some(n) = self.cloud.normals.get(i) {
                pc.normals.push(*n);
            }
            if let Some(c) = self.cloud.colors.get(i) {
                pc.colors.push(*c);
            }
            if let Some(k) = self.cloud.keys.get(i) {
                pc.keys.push(*k);
            }
        }
        write_ply(path.as_ref(), &pc, PlyLayout::XyzNormalRgb)?;
        Ok(true)
    }
}

fn mat_mul_t(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // a · bᵀ
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = (0..3).map(|k| a[i][k] * b[j][k]).sum();
        }
    }
    m
}

/// Reference → source relative geometry.
pub fn pair_geometry(scene: &DenseScene, r: usize, s: usize) -> PairGeometry {
    let (vr, vs) = (&scene.views[r], &scene.views[s]);
    let rr = mat_mul_t(&vs.r, &vr.r);
    let t: [f64; 3] = std::array::from_fn(|i| vs.t[i] - (0..3).map(|k| rr[i][k] * vr.t[k]).sum::<f64>());
    let c: [f64; 3] = std::array::from_fn(|i| -(0..3).map(|k| rr[k][i] * t[k]).sum::<f64>());
    PairGeometry {
        r: std::array::from_fn(|i| rr[i / 3][i % 3] as f32),
        t: [t[0] as f32, t[1] as f32, t[2] as f32],
        center: [c[0] as f32, c[1] as f32, c[2] as f32],
    }
}

fn cache_key(scene: &DenseScene, v: usize, fp: u64) -> (u64, u64) {
    (scene.views[v].geometry_hash(), fp)
}

/// Computes the final depth maps of all views.
pub fn compute_depth_maps(
    scene: &DenseScene,
    opts: &DensifyOptions,
    backend: &dyn PatchMatchBackend,
    cache: Option<&DepthMapCache>,
) -> Result<DepthMapSet> {
    let n = scene.views.len();
    let mut tm = DenseTimings::default();
    let t0 = Instant::now();
    if opts.neighbors.num_views == 0 || opts.neighbors.num_views > 32 {
        return Err(Error::InvalidArgument(format!("원천 뷰 수 {} (1..=32)", opts.neighbors.num_views)));
    }
    let stats = PairStats::new(scene);
    let sources: Vec<Vec<usize>> = if opts.neighbors.diversity_decay < 1.0 {
        (0..n)
            .map(|v| {
                stats.select_diverse(
                    scene,
                    v,
                    opts.neighbors.num_views,
                    opts.neighbors.min_triangulation_angle_deg.to_radians(),
                    opts.neighbors.direction_bins,
                    opts.neighbors.diversity_decay,
                )
            })
            .collect()
    } else {
        stats.select_all(opts.neighbors.num_views, opts.neighbors.min_triangulation_angle_deg)
    };
    let ranges = depth_ranges(scene);
    let valid: Vec<bool> = (0..n).map(|v| ranges[v].is_some() && !sources[v].is_empty()).collect();
    tm.neighbors = t0.elapsed();

    let t1 = Instant::now();
    let fp = opts.fingerprint();
    let cached: Vec<Option<Arc<CachedDepth>>> = (0..n)
        .map(|v| {
            if valid[v] {
                cache
                    .and_then(|c| c.get(cache_key(scene, v, fp)))
                    .filter(|c| c.width == scene.views[v].width && c.height == scene.views[v].height)
            } else {
                None
            }
        })
        .collect();
    let cache_hits = cached.iter().filter(|c| c.is_some()).count();
    let active: Vec<usize> = (0..n).filter(|&v| valid[v] && cached[v].is_none()).collect();
    let baselines: Vec<f64> = (0..n)
        .map(|v| {
            let c = scene.views[v].center();
            let mut b: Vec<f64> = sources[v]
                .iter()
                .map(|&s| {
                    let o = scene.views[s].center();
                    ((c[0] - o[0]).powi(2) + (c[1] - o[1]).powi(2) + (c[2] - o[2]).powi(2)).sqrt()
                })
                .collect();
            if b.is_empty() {
                0.0
            } else {
                crate::math::median_in_place(&mut b)
            }
        })
        .collect();
    let mut set = DepthMapSet {
        maps: vec![None; n],
        sources: sources.clone(),
        baselines,
        cache_hits,
        num_levels: 0,
        timings: DenseTimings::default(),
    };
    if active.is_empty() && cache_hits == 0 {
        tm.prepare = t1.elapsed();
        set.timings = tm;
        return Ok(set);
    }
    let min_side = scene.views.iter().map(|v| v.width.min(v.height)).min().unwrap_or(1);
    let nl = num_levels(min_side, opts.max_levels.max(1), opts.min_level_size.max(8));
    set.num_levels = nl;
    let top = nl - 1;
    let pyramids: Vec<_> = scene.views.par_iter().map(|v| build_pyramid(&v.gray, v.width, v.height, v.k, nl)).collect();
    let views: Vec<KernelView> = pyramids
        .into_iter()
        .enumerate()
        .map(|(v, levels)| {
            let (dmin, dmax) = ranges[v].unwrap_or((1.0, 2.0));
            KernelView {
                key: crate::math::hash_bytes(scene.views[v].name.as_bytes()),
                levels,
                sources: sources[v].clone(),
                pairs: sources[v].iter().map(|&s| pair_geometry(scene, v, s)).collect(),
                depth_min: dmin as f32,
                depth_max: dmax as f32,
            }
        })
        .collect();
    let mut pm = opts.pm.clone();
    pm.sigma_spatial = pm.sigma_s();
    let input = KernelInput { views, num_levels: nl, pm, filter: opts.filter.clone(), seed: opts.seed };
    // Per-scale depth of cached views (used only as the source depth for geometric runs).
    let cached_levels: Vec<Option<Vec<Arc<Vec<f32>>>>> = cached
        .iter()
        .map(|c| {
            c.as_ref().map(|c| {
                let mut lv = vec![Arc::new(c.raw_depth.clone())];
                let (mut d, mut w, mut h) = (c.raw_depth.clone(), c.width, c.height);
                for _ in 1..nl {
                    let (nd, nw, nh) = downsample_depth(&d, w, h);
                    lv.push(Arc::new(nd.clone()));
                    d = nd;
                    w = nw;
                    h = nh;
                }
                lv.reverse();
                lv
            })
        })
        .collect();
    let mut session = backend.begin(&input)?;
    tm.prepare = t1.elapsed();

    let mut snap_id = 0u64;
    let mut run_id = 0u32;
    let mut snapshot = |level: usize, states: &[ViewState]| -> DepthSnapshot {
        snap_id += 1;
        let mut maps: Vec<Option<Arc<Vec<f32>>>> = cached_levels.iter().map(|c| c.as_ref().map(|l| l[level].clone())).collect();
        for (i, &v) in active.iter().enumerate() {
            maps[v] = Some(Arc::new(states[i].depth.clone()));
        }
        DepthSnapshot { id: snap_id, level, maps }
    };
    let mut next_run = || {
        run_id += 1;
        run_id
    };

    let mut states: Vec<ViewState> = Vec::new();
    if !active.is_empty() {
        for level in 0..nl {
            let tl = Instant::now();
            let sch = opts.schedule(level, top);
            let use_prior = level > 0 && opts.pm.weak_texture_var > 0.0;
            let li = |v: usize| &input.views[v].levels[level];
            if level == 0 {
                states = active.iter().map(|&v| ViewState::new(li(v).width, li(v).height)).collect();
                let p = RunParams {
                    level,
                    geometric: false,
                    random_init: true,
                    iterations: sch.photometric_iters,
                    run_id: next_run(),
                    use_prior: false,
                };
                session.run(&active, &mut states, &p, None)?;
            } else {
                let tu = Instant::now();
                let mut up = session.upsample(&input, level, &active, &states, opts.jbu_sigma_spatial, opts.jbu_sigma_color)?;
                if use_prior {
                    for u in up.iter_mut() {
                        u.prior = u.depth.iter().zip(&u.normal).map(|(&d, n)| [n[0], n[1], n[2], d]).collect();
                    }
                }
                tm.upsample += tu.elapsed();
                let c_init = session.evaluate(level, &active, &up, false, None)?;
                let mut photo: Vec<ViewState> = if sch.restorer_random_init {
                    up.iter().map(|s| ViewState { prior: s.prior.clone(), ..ViewState::new(s.width, s.height) }).collect()
                } else {
                    up.clone()
                };
                let p = RunParams {
                    level,
                    geometric: false,
                    random_init: sch.restorer_random_init,
                    iterations: sch.photometric_iters,
                    run_id: next_run(),
                    use_prior,
                };
                session.run(&active, &mut photo, &p, None)?;
                let c_photo = session.evaluate(level, &active, &photo, false, None)?;
                let tu = Instant::now();
                let xi = opts.restorer_threshold;
                states = up
                    .into_par_iter()
                    .zip(photo.into_par_iter())
                    .enumerate()
                    .map(|(i, (mut u, ph))| {
                        for px in 0..u.depth.len() {
                            if u.depth[px] <= 0.0 || c_init[i][px] - c_photo[i][px] > xi {
                                u.depth[px] = ph.depth[px];
                                u.normal[px] = ph.normal[px];
                            }
                        }
                        u
                    })
                    .collect();
                tm.upsample += tu.elapsed();
            }
            for _ in 0..sch.geometric_rounds {
                let snap = snapshot(level, &states);
                let p = RunParams {
                    level,
                    geometric: true,
                    random_init: false,
                    iterations: sch.geometric_iters,
                    run_id: next_run(),
                    use_prior,
                };
                session.run(&active, &mut states, &p, Some(&snap))?;
            }
            tm.levels.push(tl.elapsed());
        }
    }

    let tf = Instant::now();
    for s in states.iter_mut() {
        s.prior = Vec::new();
    }
    if opts.filter.median_filter {
        states = session.median_filter(&input, top, &active, &states)?;
    }
    let counts = if active.is_empty() {
        Vec::new()
    } else {
        let snap = snapshot(top, &states);
        session.filter(&active, &states, &snap)?
    };
    drop(session);
    let finals: Vec<(usize, DepthMapResult)> = active
        .par_iter()
        .zip(states.into_par_iter())
        .zip(counts.into_par_iter())
        .map(|((&v, s), cnt)| {
            let need = ((opts.filter.min_num_consistent as usize).min(sources[v].len()) as u8).max(1);
            let mut depth = s.depth.clone();
            let mut normal = s.normal;
            for px in 0..depth.len() {
                if !(depth[px] > 0.0 && cnt[px] >= need) {
                    depth[px] = 0.0;
                    normal[px] = [0.0; 3];
                }
            }
            let mut cost = s.cost.clone();
            if opts.post.remove_speckles {
                crate::postproc::remove_speckles(&mut depth, &mut normal, &cost, s.width, s.height, &opts.post);
            }
            if opts.post.fill_holes {
                let view = &scene.views[v];
                crate::postproc::fill_holes(&mut depth, &mut normal, &mut cost, &s.cost, &view.gray, view.k, s.width, s.height, &opts.post);
            }
            (v, DepthMapResult { width: s.width, height: s.height, raw_depth: s.depth, depth, normal, cost, from_cache: false })
        })
        .collect();
    for (v, r) in finals {
        if let Some(c) = cache {
            c.insert(
                cache_key(scene, v, fp),
                Arc::new(CachedDepth {
                    width: r.width,
                    height: r.height,
                    raw_depth: r.raw_depth.clone(),
                    depth: r.depth.clone(),
                    normal: r.normal.clone(),
                    cost: r.cost.clone(),
                }),
            );
        }
        set.maps[v] = Some(Arc::new(r));
    }
    for (v, c) in cached.iter().enumerate() {
        if let Some(c) = c {
            set.maps[v] = Some(Arc::new(DepthMapResult {
                width: c.width,
                height: c.height,
                raw_depth: c.raw_depth.clone(),
                depth: c.depth.clone(),
                normal: c.normal.clone(),
                cost: c.cost.clone(),
                from_cache: true,
            }));
        }
    }
    tm.filter = tf.elapsed();
    set.timings = tm;
    Ok(set)
}

/// Depth map fusion (overlap list is at most `check_num_images`, in shared-point-count order).
pub fn fuse_depth_maps(scene: &DenseScene, maps: &DepthMapSet, opts: &DensifyOptions, threads: usize) -> FusionOutput {
    fuse_depth_maps_with(scene, maps, opts, None, threads)
}

/// Depth map fusion. With `score`, score fusion (overlap list length = `score.num_neighbors`); otherwise `opts.fusion.mode`.
pub fn fuse_depth_maps_with(
    scene: &DenseScene,
    maps: &DepthMapSet,
    opts: &DensifyOptions,
    score: Option<&ScoreFusionOptions>,
    threads: usize,
) -> FusionOutput {
    let stats = PairStats::new(scene);
    let lim = match (score, opts.fusion.mode) {
        (Some(s), _) => s.num_neighbors,
        (None, crate::params::FusionMode::Traversal) => opts.fusion.check_num_images,
        (None, crate::params::FusionMode::Consistency) => opts.fusion.consistency_num_images,
    };
    let overlap: Vec<Vec<usize>> = (0..scene.views.len()).map(|v| stats.select(v, lim, 0.0)).collect();
    let inputs: Vec<Option<FusionInput>> = maps
        .maps
        .iter()
        .enumerate()
        .map(|(v, m)| {
            m.as_ref().map(|m| FusionInput {
                depth: &m.depth,
                normal: &m.normal,
                cost: Some(&m.cost),
                baseline: maps.baselines.get(v).copied().unwrap_or(0.0),
            })
        })
        .collect();
    let mut out = match score {
        Some(s) => fuse_scored(scene, &inputs, &overlap, s, threads),
        None => fuse(scene, &inputs, &overlap, &opts.fusion, threads),
    };
    let pool = (threads > 0).then(|| rayon::ThreadPoolBuilder::new().num_threads(threads).build().ok()).flatten();
    if opts.fusion.visibility.enabled() {
        let t = Instant::now();
        let vp = &opts.fusion.visibility;
        out.visibility_removed = match &pool {
            Some(pool) => pool.install(|| crate::visibility::filter_visibility(scene, &inputs, &overlap, &mut out, vp)),
            None => crate::visibility::filter_visibility(scene, &inputs, &overlap, &mut out, vp),
        };
        out.visibility_time = t.elapsed();
    }
    if opts.fusion.smooth.enabled() {
        let t = Instant::now();
        let gsd = crate::fusion::scene_gsd(scene, &inputs);
        let sp = &opts.fusion.smooth;
        match &pool {
            Some(pool) => pool.install(|| crate::smooth::smooth_along_normals(&mut out.cloud, gsd, sp)),
            None => crate::smooth::smooth_along_normals(&mut out.cloud, gsd, sp),
        };
        out.smooth_time = t.elapsed();
    }
    out
}

/// Reruns only fusion with the retained depth maps (no depth estimation or filtering). From `opts` only the fusion settings are used,
/// and score fusion is used if `score` is present. The result's `timings` are the depth map timings with this fusion time put in.
pub fn fuse_output(scene: &DenseScene, maps: &DepthMapSet, opts: &DensifyOptions, score: Option<&ScoreFusionOptions>) -> DenseOutput {
    let tf = Instant::now();
    let f = fuse_depth_maps_with(scene, maps, opts, score, 0);
    let mut timings = maps.timings.clone();
    timings.fusion = tf.elapsed();
    timings.fusion_pass1 = f.pass1_time;
    timings.fusion_pass2 = f.pass2_time;
    timings.fusion_smooth = f.smooth_time;
    timings.fusion_visibility = f.visibility_time;
    timings.fusion_visibility_removed = f.visibility_removed;
    let depth_views = maps.maps.iter().filter(|m| m.is_some()).count();
    DenseOutput {
        cloud: f.cloud,
        visibility: f.visibility,
        timings,
        cache_hits: maps.cache_hits,
        depth_views,
        depth_maps: maps.clone(),
        residual: f.residual,
    }
}

/// Full densification: depth maps → filter → fusion.
pub fn densify(
    scene: &DenseScene,
    opts: &DensifyOptions,
    backend: &dyn PatchMatchBackend,
    cache: Option<&DepthMapCache>,
) -> Result<DenseOutput> {
    densify_with(scene, opts, None, backend, cache)
}

/// Same as [`densify`], but with score fusion if `score` is present.
pub fn densify_with(
    scene: &DenseScene,
    opts: &DensifyOptions,
    score: Option<&ScoreFusionOptions>,
    backend: &dyn PatchMatchBackend,
    cache: Option<&DepthMapCache>,
) -> Result<DenseOutput> {
    let maps = compute_depth_maps(scene, opts, backend, cache)?;
    Ok(fuse_output(scene, &maps, opts, score))
}
