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

//! 조밀화 진행: 이웃·깊이 범위 → 피라미드 → 스케일별 실행(광도, 세부 복원, 기하 회차) → 중앙값 필터 → 판독·필터 → 융합.

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

/// 단계별 시간.
#[derive(Clone, Debug, Default)]
pub struct DenseTimings {
    /// 이웃 선택 + 깊이 범위.
    pub neighbors: Duration,
    /// 피라미드 구성 + 백엔드 세션 시작(업로드).
    pub prepare: Duration,
    /// 스케일별 깊이 추정(커널 실행 + 상향 표본 + 세부 복원 판정). 0 = 최저 스케일.
    pub levels: Vec<Duration>,
    /// 그중 CPU 쪽 상향 표본·세부 판정 시간 합.
    pub upsample: Duration,
    /// 중앙값 필터 + 판독·필터.
    pub filter: Duration,
    /// 융합.
    pub fusion: Duration,
    /// 그중 일치 융합 1차·2차 시간(2차 융합을 했을 때만 2차가 0 이 아니다).
    pub fusion_pass1: Duration,
    /// 일치 융합 2차 시간.
    pub fusion_pass2: Duration,
}

impl DenseTimings {
    /// 깊이 추정 합.
    pub fn depth(&self) -> Duration {
        self.levels.iter().sum()
    }
    /// 전체 시간(이웃 + 준비 + 깊이 + 필터 + 융합).
    pub fn total(&self) -> Duration {
        self.neighbors + self.prepare + self.depth() + self.filter + self.fusion
    }
}

/// 최종 깊이맵 한 장(뷰 해상도).
#[derive(Clone, Debug)]
pub struct DepthMapResult {
    /// 너비(픽셀).
    pub width: usize,
    /// 높이(픽셀).
    pub height: usize,
    /// 필터 전 최종 깊이.
    pub raw_depth: Vec<f32>,
    /// 필터 통과 깊이(무효 0).
    pub depth: Vec<f32>,
    /// 기준 카메라 좌표 법선.
    pub normal: Vec<[f32; 3]>,
    /// 최종 집계 비용(신뢰도).
    pub cost: Vec<f32>,
    /// 깊이맵 캐시에서 왔는지.
    pub from_cache: bool,
}

/// 깊이맵 계산 결과.
#[derive(Clone, Debug, Default)]
pub struct DepthMapSet {
    /// 뷰별 깊이맵(깊이맵을 만들지 못한 뷰는 None).
    pub maps: Vec<Option<Arc<DepthMapResult>>>,
    /// 뷰별 원천 뷰.
    pub sources: Vec<Vec<usize>>,
    /// 뷰별 대표 기준선(원천 중심 거리 중앙값).
    pub baselines: Vec<f64>,
    /// 캐시 적중 뷰 수.
    pub cache_hits: usize,
    /// 사용한 스케일 수.
    pub num_levels: usize,
    /// 단계별 시간.
    pub timings: DenseTimings,
}

/// 조밀화 결과.
#[derive(Clone, Debug, Default)]
pub struct DenseOutput {
    /// 조밀 점군(위치·법선·색).
    pub cloud: PointCloud,
    /// 점마다 기여 뷰 색인.
    pub visibility: Vec<Vec<u32>>,
    /// 단계별 시간.
    pub timings: DenseTimings,
    /// 캐시 적중 뷰 수.
    pub cache_hits: usize,
    /// 깊이맵을 만든 뷰 수.
    pub depth_views: usize,
    /// 깊이맵(통계·진단용).
    pub depth_maps: DepthMapSet,
    /// 점마다 2차 융합 점 여부(2차 융합을 하지 않았으면 비어 있다).
    pub residual: Vec<bool>,
}

impl DenseOutput {
    /// PLY(x y z nx ny nz red green blue) 쓰기. 부모 폴더를 만든다.
    pub fn write_ply(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(p) = path.parent() {
            if !p.as_os_str().is_empty() {
                std::fs::create_dir_all(p)?;
            }
        }
        write_ply(path, &self.cloud, PlyLayout::XyzNormalRgb)
    }

    /// 2차 융합 점 수.
    pub fn num_residual(&self) -> usize {
        self.residual.iter().filter(|&&r| r).count()
    }

    /// 2차 융합 점만 PLY 로 쓴다(없으면 쓰지 않고 거짓).
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

/// 기준 → 원천 상대 기하.
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

/// 모든 뷰의 최종 깊이맵을 계산한다.
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
    // 캐시 뷰의 스케일별 깊이(기하 실행의 원천 깊이로만 쓴다).
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

/// 깊이맵 융합(겹침 목록은 공유 점 수 순 최대 `check_num_images`).
pub fn fuse_depth_maps(scene: &DenseScene, maps: &DepthMapSet, opts: &DensifyOptions, threads: usize) -> FusionOutput {
    fuse_depth_maps_with(scene, maps, opts, None, threads)
}

/// 깊이맵 융합. `score` 가 있으면 점수 융합(겹침 목록 길이 = `score.num_neighbors`), 없으면 `opts.fusion.mode`.
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
    match score {
        Some(s) => fuse_scored(scene, &inputs, &overlap, s, threads),
        None => fuse(scene, &inputs, &overlap, &opts.fusion, threads),
    }
}

/// 보관한 깊이맵으로 융합만 다시 실행한다(깊이 추정·필터는 하지 않는다). `opts` 에서는 융합 설정만 쓰고,
/// `score` 가 있으면 점수 융합을 쓴다. 결과의 `timings` 는 깊이맵 쪽 시간에 이번 융합 시간을 넣은 것이다.
pub fn fuse_output(scene: &DenseScene, maps: &DepthMapSet, opts: &DensifyOptions, score: Option<&ScoreFusionOptions>) -> DenseOutput {
    let tf = Instant::now();
    let f = fuse_depth_maps_with(scene, maps, opts, score, 0);
    let mut timings = maps.timings.clone();
    timings.fusion = tf.elapsed();
    timings.fusion_pass1 = f.pass1_time;
    timings.fusion_pass2 = f.pass2_time;
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

/// 조밀화 전체: 깊이맵 → 필터 → 융합.
pub fn densify(
    scene: &DenseScene,
    opts: &DensifyOptions,
    backend: &dyn PatchMatchBackend,
    cache: Option<&DepthMapCache>,
) -> Result<DenseOutput> {
    densify_with(scene, opts, None, backend, cache)
}

/// [`densify`] 와 같되 `score` 가 있으면 점수 융합.
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
