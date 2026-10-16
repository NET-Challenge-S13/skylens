/*
 * params.rs
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

//! Densification parameters: profiles, PatchMatch constants, filter and fusion tolerances.

/// Densification profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MvsProfile {
    /// Short iteration schedule (default). The detail-restoration stage starts from the upsampled hypothesis.
    #[default]
    Fast,
    /// Long iteration schedule. The detail-restoration stage starts from random initialization.
    Quality,
}

impl MvsProfile {
    /// Looks up by name ("fast" / "quality", case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "fast" => Some(Self::Fast),
            "quality" => Some(Self::Quality),
            _ => None,
        }
    }
    /// Profile name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Quality => "quality",
        }
    }
}

/// Iteration schedule of one scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LevelSchedule {
    /// Number of red-black iterations of a photometric run (lowest scale: random initialization; above it: detail-restoration run).
    pub photometric_iters: u32,
    /// Whether the detail-restoration run starts from random initialization (if false, it starts from the upsampled hypothesis).
    pub restorer_random_init: bool,
    /// Number of geometric-consistency run rounds (synchronized over all reference views).
    pub geometric_rounds: u32,
    /// Number of iterations in one geometric run round.
    pub geometric_iters: u32,
}

/// PatchMatch constants (cost, view selection, refinement, geometric term).
#[derive(Clone, Debug, PartialEq)]
pub struct PmParams {
    /// Window radius (pixels).
    pub window_radius: i32,
    /// Sample spacing inside the window (1 = all, 2 = every other row and column).
    pub window_step: i32,
    /// Bilateral weight spatial σ (pixels). Window radius if ≤ 0.
    pub sigma_spatial: f32,
    /// Bilateral weight brightness σ (brightness in \[0,1\]).
    pub sigma_color: f32,
    /// Good-cost boundary τ(t) = tau0·exp(−t²/tau_alpha).
    pub tau0: f32,
    /// Decay constant α of the good-cost boundary.
    pub tau_alpha: f32,
    /// Bad-cost boundary.
    pub tau1: f32,
    /// Cost confidence exp(−m²/(2β²)).
    pub beta: f32,
    /// View adoption: number of good costs > n1 and number of bad costs < n2.
    pub n1: u32,
    /// Upper bound n2 on the number of bad costs for view adoption.
    pub n2: u32,
    /// Weight of the previous most important view when it is not adopted.
    pub prev_view_weight: f32,
    /// Triangulation prior minimum angle (degrees).
    pub min_triangulation_angle_deg: f32,
    /// Incidence angle prior σ.
    pub incident_angle_sigma: f32,
    /// Geometric term weight λ and upper bound δ (pixels).
    pub geom_lambda: f32,
    /// Geometric term upper bound δ (pixels).
    pub geom_max_cost: f32,
    /// K of the top-K mean used when the weight sum is 0.
    pub top_k: u32,
    /// Initial refinement perturbation size (relative inverse depth, normal angle in degrees): photometric/geometric.
    pub eps0_photometric: f32,
    /// Initial refinement perturbation size (relative inverse depth): geometric run.
    pub eps0_geometric: f32,
    /// Initial refinement normal perturbation angle (degrees): photometric run.
    pub phi0_photometric_deg: f32,
    /// Initial refinement normal perturbation angle (degrees): geometric run.
    pub phi0_geometric_deg: f32,
    /// Lower bound of the inverse-depth perturbation.
    pub eps_min: f32,
    /// Lower bound of the normal perturbation angle (degrees).
    pub phi_min_deg: f32,
    /// Reference patch variance threshold (brightness² in \[0,1\]) for adding the coarse-scale hypothesis as a candidate. 0 turns it off.
    pub weak_texture_var: f32,
}

impl Default for PmParams {
    fn default() -> Self {
        Self {
            window_radius: 5,
            window_step: 2,
            sigma_spatial: -1.0,
            sigma_color: 0.2,
            tau0: 0.8,
            tau_alpha: 90.0,
            tau1: 1.2,
            beta: 0.3,
            n1: 2,
            n2: 3,
            prev_view_weight: 0.2,
            min_triangulation_angle_deg: 1.0,
            incident_angle_sigma: 0.9,
            geom_lambda: 0.2,
            geom_max_cost: 3.0,
            top_k: 3,
            eps0_photometric: 0.2,
            eps0_geometric: 0.05,
            phi0_photometric_deg: 30.0,
            phi0_geometric_deg: 10.0,
            eps_min: 0.005,
            phi_min_deg: 1.0,
            weak_texture_var: 0.0,
        }
    }
}

impl PmParams {
    /// Effective spatial σ.
    pub fn sigma_s(&self) -> f32 {
        if self.sigma_spatial > 0.0 {
            self.sigma_spatial
        } else {
            self.window_radius as f32
        }
    }
    /// Window sample offsets (−R..R, spacing step).
    pub fn window_offsets(&self) -> Vec<i32> {
        let r = self.window_radius;
        let s = self.window_step.max(1);
        let mut v = Vec::new();
        let mut o = -r;
        while o <= r {
            v.push(o);
            o += s;
        }
        v
    }
}

/// Depth map filter tolerances.
#[derive(Clone, Debug, PartialEq)]
pub struct FilterParams {
    /// Photometric condition: visibility probability ≥ E(1 − min_ncc).
    pub min_ncc: f32,
    /// Views with a triangulation angle smaller than this (degrees) are not counted.
    pub min_triangulation_angle_deg: f32,
    /// Minimum number of agreeing views (effective value is min(source count, this value)).
    pub min_num_consistent: u32,
    /// Forward/backward reprojection error tolerance (pixels).
    pub geom_max_cost: f32,
    /// Probability emission σ.
    pub ncc_sigma: f32,
    /// 5×5 median plane filter before readout.
    pub median_filter: bool,
}

impl Default for FilterParams {
    fn default() -> Self {
        Self {
            min_ncc: 0.1,
            min_triangulation_angle_deg: 3.0,
            min_num_consistent: 2,
            geom_max_cost: 1.0,
            ncc_sigma: 0.6,
            median_filter: true,
        }
    }
}

/// Fusion method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum FusionMode {
    /// Per-component median point of the pixels gathered by depth-first expansion through neighbors of neighbors (close to one point per
    /// surface).
    Traversal,
    /// For each reference pixel, count the neighbor views whose depth, normal and reprojection agree; if enough, the mean point of the
    /// agreeing points.
    #[default]
    Consistency,
}

impl FusionMode {
    /// Looks up by name ("traversal"/"median", "consistency"/"average").
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "traversal" | "median" => Some(Self::Traversal),
            "consistency" | "average" => Some(Self::Consistency),
            _ => None,
        }
    }
}

/// Handling of below-criterion (leftover) pixels in agreement fusion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum FusionResidual {
    /// No handling (default).
    #[default]
    None,
    /// Gives back the pixels (reference and neighbor) temporarily occupied by discarded candidates so later reference images can use them
    /// again.
    Release,
    /// After the first fusion, a second fusion with stricter tolerances using only valid-depth pixels that did not become a point.
    SecondPass,
}

impl FusionResidual {
    /// Looks up by name ("none", "release", "second-pass").
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().replace('_', "-").as_str() {
            "none" => Some(Self::None),
            "release" => Some(Self::Release),
            "second-pass" | "secondpass" => Some(Self::SecondPass),
            _ => None,
        }
    }
    /// Handling method name.
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Release => "release",
            Self::SecondPass => "second-pass",
        }
    }
}

/// Second fusion tolerances.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidualParams {
    /// Minimum number of agreeing views other than the reference.
    pub min_views: usize,
    /// Relative depth tolerance (None = 0.5 times the first pass).
    pub depth_error: Option<f64>,
    /// Normal tolerance angle (degrees, None = 0.67 times the first pass).
    pub normal_error_deg: Option<f64>,
    /// Second-pass points within this distance (multiple of GSD) of a first-pass point are discarded.
    pub min_dist_gsd: f64,
}

impl Default for ResidualParams {
    fn default() -> Self {
        Self { min_views: 2, depth_error: None, normal_error_deg: None, min_dist_gsd: 0.5 }
    }
}

/// Fusion tolerances.
#[derive(Clone, Debug, PartialEq)]
pub struct FusionParams {
    /// Fusion method.
    pub mode: FusionMode,
    /// Agreement fusion: handling of leftover pixels.
    pub residual: FusionResidual,
    /// Second fusion tolerances.
    pub residual_params: ResidualParams,
    /// Agreement fusion: minimum number of agreeing views other than the reference.
    pub min_consistent_views: usize,
    /// Agreement fusion: a pixel already used by the point of another reference pixel is not reused as a reference.
    pub mark_used: bool,
    /// Agreement fusion: marking radius (pixels, 0 = projected pixel only).
    pub mark_radius: usize,
    /// Agreement fusion: inverse-variance weighted average of depth uncertainty (simple average if off).
    pub inverse_variance: bool,
    /// Registration uncertainty σ_px = sigma_px0 + sigma_px_slope·cost (pixels).
    pub sigma_px0: f64,
    /// Cost slope of σ_px.
    pub sigma_px_slope: f64,
    /// Agreement fusion: number of overlapping views to check (in shared-point order).
    pub consistency_num_images: usize,
    /// Extended fusion: minimum number of pixels required for a point.
    pub min_num_pixels: usize,
    /// Extended fusion: maximum number of pixels gathered into a point.
    pub max_num_pixels: usize,
    /// Extended fusion: maximum expansion depth.
    pub max_traversal_depth: usize,
    /// Reprojection error tolerance (pixels).
    pub max_reproj_error: f64,
    /// Relative depth tolerance.
    pub max_depth_error: f64,
    /// Normal tolerance angle (degrees).
    pub max_normal_error_deg: f64,
    /// Extended fusion: number of overlapping views to check.
    pub check_num_images: usize,
    /// Normal-direction bilateral smoothing applied to the fused cloud (iterations 0 = off).
    pub smooth: crate::smooth::SmoothParams,
    /// Free-space visibility filter applied to the fused cloud before smoothing (check_views 0 = off).
    pub visibility: crate::visibility::VisibilityParams,
}

impl Default for FusionParams {
    fn default() -> Self {
        Self {
            mode: FusionMode::Consistency,
            residual: FusionResidual::None,
            residual_params: ResidualParams::default(),
            min_consistent_views: 5,
            mark_used: true,
            mark_radius: 0,
            inverse_variance: true,
            sigma_px0: 0.25,
            sigma_px_slope: 1.0,
            consistency_num_images: 12,
            min_num_pixels: 5,
            max_num_pixels: 10000,
            max_traversal_depth: 100,
            max_reproj_error: 2.0,
            max_depth_error: 0.01,
            max_normal_error_deg: 10.0,
            check_num_images: 50,
            smooth: crate::smooth::SmoothParams::default(),
            visibility: crate::visibility::VisibilityParams::default(),
        }
    }
}

/// Neighbor view selection.
#[derive(Clone, Debug, PartialEq)]
pub struct NeighborParams {
    /// Maximum number of source views per reference view (≤ 32).
    pub num_views: usize,
    /// Lower bound of the 75th-percentile triangulation angle (degrees).
    pub min_triangulation_angle_deg: f64,
    /// Direction diversity: number of baseline azimuth sectors and the decay for repeated selection in the same sector (1 = off).
    pub direction_bins: usize,
    /// Decay for repeated selection in the same sector (1 = direction diversity off).
    pub diversity_decay: f64,
}

impl Default for NeighborParams {
    fn default() -> Self {
        Self { num_views: 10, min_triangulation_angle_deg: 1.0, direction_bins: 8, diversity_decay: 1.0 }
    }
}

/// `densify` options.
#[derive(Clone, Debug, PartialEq)]
pub struct DensifyOptions {
    /// Iteration schedule profile.
    pub profile: MvsProfile,
    /// Neighbor view selection.
    pub neighbors: NeighborParams,
    /// PatchMatch constants.
    pub pm: PmParams,
    /// Depth map filter tolerances.
    pub filter: FilterParams,
    /// Fusion tolerances.
    pub fusion: FusionParams,
    /// Post-processing after the filter.
    pub post: crate::postproc::PostParams,
    /// Upper bound on the number of scales (downscale ratio 0.5).
    pub max_levels: usize,
    /// Lower bound of the short side at the lowest scale (the scale count is reduced if it would become smaller).
    pub min_level_size: usize,
    /// Joint bilateral upsampling parameters: spatial σ (low-res pixels), brightness σ.
    pub jbu_sigma_spatial: f32,
    /// Joint bilateral upsampling brightness σ.
    pub jbu_sigma_color: f32,
    /// Detail restorer replacement threshold ξ.
    pub restorer_threshold: f32,
    /// Random seed.
    pub seed: u64,
    /// Schedule override (for measurement): geometric run iteration count, round count.
    pub geometric_iters_override: Option<u32>,
    /// Schedule override: geometric run round count.
    pub geometric_rounds_override: Option<u32>,
}

impl Default for DensifyOptions {
    /// Same as [`DensifyOptions::with_profile`]`(MvsProfile::Fast)`.
    fn default() -> Self {
        Self::with_profile(MvsProfile::Fast)
    }
}

impl DensifyOptions {
    /// Untuned base values shared by all profiles (component defaults).
    fn base() -> Self {
        Self {
            profile: MvsProfile::Fast,
            neighbors: NeighborParams::default(),
            pm: PmParams::default(),
            filter: FilterParams::default(),
            fusion: FusionParams::default(),
            post: crate::postproc::PostParams::default(),
            max_levels: 3,
            min_level_size: 64,
            jbu_sigma_spatial: 1.0,
            jbu_sigma_color: 0.04,
            restorer_threshold: 0.1,
            seed: 0x005e_ed0f_de75,
            geometric_iters_override: None,
            geometric_rounds_override: None,
        }
    }
}

impl DensifyOptions {
    /// Profile defaults: fast uses window radius 5 with spacing 2 (36 samples), quality uses radius 5 with spacing 1 (121 samples).
    ///
    /// The fast profile also carries the filter/fusion values tuned on 240 aerial views at 960 px
    /// (see `docs/tuning/`): stricter depth-map filter, no median filter, hole filling on, 3-view
    /// consistency fusion with a tight depth tolerance, and normal-direction smoothing of the fused cloud.
    pub fn with_profile(profile: MvsProfile) -> Self {
        let mut o = Self { profile, ..Self::base() };
        o.pm.window_step = match profile {
            MvsProfile::Fast => 2,
            MvsProfile::Quality => 1,
        };
        if profile == MvsProfile::Fast {
            o.apply_fast_tuning();
        }
        o
    }

    /// Tuned fast-profile filter, post-processing and fusion values.
    fn apply_fast_tuning(&mut self) {
        let f = &mut self.filter;
        (f.min_ncc, f.geom_max_cost, f.min_num_consistent, f.median_filter) = (0.2, 0.5, 3, false);
        self.post.fill_holes = true;
        let u = &mut self.fusion;
        (u.min_consistent_views, u.max_depth_error, u.max_normal_error_deg, u.max_reproj_error) = (3, 0.005, 23.0, 100.0);
    }

    /// Schedule of scale `level` (0 = lowest). `top` is the highest scale number.
    pub fn schedule(&self, level: usize, top: usize) -> LevelSchedule {
        let mut s = self.base_schedule(level, top);
        if let Some(v) = self.geometric_iters_override {
            s.geometric_iters = v;
        }
        if let Some(v) = self.geometric_rounds_override {
            s.geometric_rounds = v;
        }
        s
    }

    fn base_schedule(&self, level: usize, top: usize) -> LevelSchedule {
        match self.profile {
            MvsProfile::Quality => LevelSchedule {
                photometric_iters: if level == 0 { 7 } else { 6 },
                restorer_random_init: true,
                geometric_rounds: 2,
                geometric_iters: 6,
            },
            MvsProfile::Fast => {
                let _ = top;
                LevelSchedule {
                    photometric_iters: if level == 0 { 6 } else { 3 },
                    restorer_random_init: false,
                    geometric_rounds: 2,
                    geometric_iters: 2,
                }
            }
        }
    }

    /// Settings hash (for the depth map cache key).
    pub fn fingerprint(&self) -> u64 {
        let s = format!(
            "{:?}",
            (
                self.profile,
                &self.neighbors,
                &self.pm,
                &self.filter,
                &self.post,
                self.max_levels,
                self.min_level_size,
                self.jbu_sigma_spatial,
                self.jbu_sigma_color,
                (self.restorer_threshold, self.seed, self.geometric_iters_override, self.geometric_rounds_override)
            )
        );
        crate::math::hash_bytes(s.as_bytes())
    }
}
