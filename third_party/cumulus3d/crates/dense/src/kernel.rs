/*
 * kernel.rs
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

//! PatchMatch kernel boundary. The backend handles propagation, view selection and refinement at one scale (red-black iteration runs),
//! single-hypothesis cost evaluation and the final readout;
//! this crate handles multi-scale progression, upsampling, detail-restoration decisions, filter thresholds and fusion.
//!
//! # Rules the kernel must follow
//! Coordinates: integer pixel = pixel center. Ray `ray(x, y) = ((x−cx)/fx, (y−cy)/fy, 1)`, depth is camera z,
//! the normal is a unit vector in reference camera coordinates with `n·ray < 0`.
//!
//! - **Cost** `m = 1 − ρ`: bilateral-weighted NCC using only the reference image color. The window is the square grid of
//!   `PmParams::window_offsets()`, weights `exp(−(dx²+dy²)/(2σs²) − (I_c − I_k)²/(2σc²))`, brightness outside the image is 0. Source
//!   brightness is the bilinear value (0 outside) at the position moved by the plane-induced homography `H = K_j (R + t nᵀ/δ) K⁻¹`, `δ =
//!   n·(d·ray)`. 2 if variance < 1e−5, otherwise `clamp(1 − cov/√(var_r var_s), 0, 2)`. 2 if the center projection is behind the camera or
//!   outside the source image.
//! - **Geometric error** `Δe`: pixel distance after projecting the reference point into the source, going back to the reference through the
//!   nearest value of the source depth map and reprojecting. Infinite on projection failure, outside the image or depth 0. In the cost it
//!   is `min(Δe, geom_max_cost)`.
//! - **Priors** (once per view, using the current hypothesis at the start of a half step): product of the triangulation, incidence-angle
//!   and resolution priors ([`crate::math`]).
//! - **Iteration**: red-black half steps. Color `(x+y) mod 2`. An active pixel evaluates 8 hypotheses plus the current one: for each of 8
//!   regions (per up/right/down/left direction, 7 V-shaped offsets `(0,−1),(±1,−2),(±2,−3),(±3,−4)` and 11 band offsets
//!   `(0,−3),(0,−5),…,(0,−23)`, rotated by 90° each), the plane of the neighbor with the smallest stored cost (the first one found on ties)
//!   moved onto the own ray. If a region has no valid neighbor or the move fails, the current hypothesis is used instead.
//! - **View selection**: vote with the costs of the 8 propagated hypotheses (good `m < τ(t) = τ0·exp(−t²/α)`, bad `m > τ1`; adopted if the
//!   good count exceeds n1 and the bad count is below n2). Weight of an adopted view `w = mean of exp(−m²/(2β²)) over the good costs`,
//!   doubled if it was the most important view of the previous iteration. Even if not adopted, `prev_view_weight` if it was the previous
//!   most important view. Multiplying by the priors gives `w''`, used for a per-hypothesis weighted average (for a geometric run, `m +
//!   λ·min(Δe, δ)`). If the weight sum is 0, the prior-weighted average gathered in the same pass is used instead. Views whose prior is 0
//!   are not evaluated. The minimum-cost hypothesis is adopted (ties go to the earlier index). Most important view = maximum `w''` (ties go
//!   to the earlier view).
//! - **Refinement** (same half step): random depth (uniform in inverse depth `[1/dmax, 1/dmin]`), random normal (uniform on the sphere,
//!   flipped toward the camera), perturbed depth `1/d' = (1/d)(1 + u·ε_t)`, perturbed normal (rotate by `u·φ_t` about a random
//!   perpendicular axis, on failure retry up to 3 times halving the angle), with `ε_t = max(ε_min, ε0·2^−t)`, `φ_t = max(φ_min, φ0·2^−t)`,
//!   forming 6 combinations `(d_r,n)(d,n_r)(d_r,n_r)(d_p,n)(d,n_p)(d_p,n_p)`. Aggregated with the same weights (positive views only);
//!   adopted if smaller.
//! - **Random numbers**: per-pixel `splitmix64` sequence. Start state `mix64(seed ^ mix64(view_key ^ mix64(level·2⁴⁸ + run·2³² + t·2¹⁶ +
//!   half)) ^ pixel)` (initialization uses t = 0xFFFF). A uniform (0,1] value is `((x >> 40) + 1)·2⁻²⁴`. Deterministic regardless of thread
//!   layout.
//! - **Coarse-scale candidate**: if `use_prior` is set and the reference patch variance is below the threshold, the state's `prior` plane
//!   is refined and evaluated as one more candidate.
//! - **Run start**: after random initialization (if requested), fill the stored cost with the top-K mean cost of the current hypothesis and
//!   clear the most important view.
//! - **Numerics**: the kernel may use approximate division and exponential (speed first). Results are bit-reproducible on the same device
//!   with the same input.

use crate::params::{FilterParams, PmParams};
use cumulus3d_core::Result;
use std::sync::Arc;

/// View image at one scale.
#[derive(Clone, Debug)]
pub struct LevelImage {
    /// Width (pixels).
    pub width: usize,
    /// Height (pixels).
    pub height: usize,
    /// 8-bit gray (row-major).
    pub gray: Arc<Vec<u8>>,
    /// fx, fy, cx, cy (integer-center convention).
    pub k: [f32; 4],
}

/// Reference → source relative geometry (source camera coordinates = r·X + t, source center in reference coordinates `center`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct PairGeometry {
    /// Reference → source rotation (row-major 3×3).
    pub r: [f32; 9],
    /// Reference → source translation.
    pub t: [f32; 3],
    /// Source camera center (reference camera coordinates).
    pub center: [f32; 3],
}

/// Kernel input view.
#[derive(Clone, Debug)]
pub struct KernelView {
    /// Random number key (image name hash).
    pub key: u64,
    /// Per-scale images (0 = lowest resolution).
    pub levels: Vec<LevelImage>,
    /// Source view indices (≤ 32).
    pub sources: Vec<usize>,
    /// Relative geometry per source.
    pub pairs: Vec<PairGeometry>,
    /// Lower bound of the random initialization depth.
    pub depth_min: f32,
    /// Upper bound of the random initialization depth.
    pub depth_max: f32,
}

/// Whole kernel input (fixed during one densification call).
#[derive(Clone, Debug)]
pub struct KernelInput {
    /// All kernel input views.
    pub views: Vec<KernelView>,
    /// Number of scales.
    pub num_levels: usize,
    /// PatchMatch constants.
    pub pm: PmParams,
    /// Readout and filter tolerances.
    pub filter: FilterParams,
    /// Random seed.
    pub seed: u64,
}

/// Pixel state of one reference view (one scale).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ViewState {
    /// Width (pixels).
    pub width: usize,
    /// Height (pixels).
    pub height: usize,
    /// Per-pixel depth (camera z).
    pub depth: Vec<f32>,
    /// Per-pixel normal (reference camera coordinates).
    pub normal: Vec<[f32; 3]>,
    /// Per-pixel stored cost.
    pub cost: Vec<f32>,
    /// Coarse-scale hypothesis (nx, ny, nz, d). If non-empty, used as a candidate at weakly textured pixels.
    pub prior: Vec<[f32; 4]>,
}

impl ViewState {
    /// State filled with depth and normal 0 and cost 2 (maximum).
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        Self { width, height, depth: vec![0.0; n], normal: vec![[0.0; 3]; n], cost: vec![2.0; n], prior: Vec::new() }
    }
}

/// Depth snapshot read by geometric runs (view index → depth map at that scale). Equal `id` implies equal content.
#[derive(Clone, Debug)]
pub struct DepthSnapshot {
    /// Snapshot identifier.
    pub id: u64,
    /// Scale of the depth maps.
    pub level: usize,
    /// Depth map per view index (None if absent).
    pub maps: Vec<Option<Arc<Vec<f32>>>>,
}

/// Parameters of one run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunParams {
    /// Scale number (0 = lowest).
    pub level: usize,
    /// Whether this is a geometric-consistency run.
    pub geometric: bool,
    /// Whether to randomly initialize before starting.
    pub random_init: bool,
    /// Number of red-black iterations.
    pub iterations: u32,
    /// Run number for the random number key.
    pub run_id: u32,
    /// Use the state's `prior` hypothesis as an extra candidate at weakly textured pixels (reference patch variance <
    /// `PmParams::weak_texture_var`).
    pub use_prior: bool,
}

/// Backend state during one densification call (device-resident data, etc.).
pub trait PatchMatchSession {
    /// Runs red-black iterations on state `states[i]` of `views[i]`. Updates the state in place.
    fn run(&mut self, views: &[usize], states: &mut [ViewState], params: &RunParams, snapshot: Option<&DepthSnapshot>) -> Result<()>;
    /// Top-K mean cost of the current hypothesis (for geometric, `m + λ·min(Δe, δ)`).
    fn evaluate(
        &mut self,
        level: usize,
        views: &[usize],
        states: &[ViewState],
        geometric: bool,
        snapshot: Option<&DepthSnapshot>,
    ) -> Result<Vec<Vec<f32>>>;
    /// Final-scale readout: per pixel, the number of source views satisfying all photometric and geometric conditions.
    /// Conditions: triangulation angle ≥ filter minimum angle, incidence cos > 0, emission visibility probability ≥ E(1 − min_ncc),
    /// non-truncated Δe ≤ tolerance.
    fn filter(&mut self, views: &[usize], states: &[ViewState], snapshot: &DepthSnapshot) -> Result<Vec<Vec<u8>>>;
    /// Joint bilateral upsampling of the scale `level − 1` state to `level` (same rule as [`crate::upsample::joint_bilateral_upsample`]).
    /// The default implementation computes on the host.
    fn upsample(
        &mut self,
        input: &KernelInput,
        level: usize,
        views: &[usize],
        low: &[ViewState],
        sigma_s: f32,
        sigma_c: f32,
    ) -> Result<Vec<ViewState>> {
        Ok(views
            .iter()
            .zip(low)
            .map(|(&v, s)| {
                crate::upsample::joint_bilateral_upsample(
                    s,
                    &input.views[v].levels[level - 1],
                    &input.views[v].levels[level],
                    sigma_s,
                    sigma_c,
                )
            })
            .collect())
    }
    /// 5×5 median plane filter (same rule as [`crate::upsample::median_plane_filter`]). The default implementation computes on the host.
    fn median_filter(&mut self, input: &KernelInput, level: usize, views: &[usize], states: &[ViewState]) -> Result<Vec<ViewState>> {
        Ok(views.iter().zip(states).map(|(&v, s)| crate::upsample::median_plane_filter(s, &input.views[v].levels[level])).collect())
    }
}

/// PatchMatch backend.
pub trait PatchMatchBackend: Send + Sync {
    /// Backend name (for logs).
    fn name(&self) -> String;
    /// Receives the input and opens a session (image upload, etc.).
    fn begin<'a>(&'a self, input: &'a KernelInput) -> Result<Box<dyn PatchMatchSession + 'a>>;
}

/// Pixel random-number start state (same definition as the kernel).
pub fn rng_state(seed: u64, view_key: u64, level: u32, run: u32, t: u32, half: u32, pixel: u64) -> u64 {
    use crate::math::mix64;
    let tag = ((level as u64) << 48) ^ ((run as u64) << 32) ^ ((t as u64) << 16) ^ half as u64;
    mix64(seed ^ mix64(view_key ^ mix64(tag)) ^ pixel)
}

/// One splitmix64 step and a uniform (0,1] value.
pub fn rng_next(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let z = crate::math::mix64(*state);
    ((z >> 40) as f32 + 1.0) * (1.0 / 16_777_216.0)
}
