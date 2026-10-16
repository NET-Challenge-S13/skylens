/*
 * declare.rs
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

//! Declarative builder layer: record a plan first and run it all at once later.
//!
//! The methods of the [`ReconBuilder`] returned by [`Recon::declare`] only **record a plan** ("this is what to do") and execute
//! nothing (they neither create nor delete files or folders, and do not open devices). [`ReconBuilder::build`] validates the plan
//! and builds an executable reconstruction ([`Recon`]); [`Recon::run`] assembles the session reducer + pipeline + hooks + default output
//! hook and
//! runs the position loop, the wait for background jobs and the shutdown in one go (deferred execution).
//!
//! The plan itself is a value ([`Plan`]), so it can be compared, cloned and saved as TOML ([`Plan::to_toml`] / [`Plan::from_toml`]).
//! Hooks (closures) are kept in the builder, not in the plan.
//!
//! # Usage example
//!
//! ```no_run
//! use cumulus3d_cli::declare::{Dense, Recon, Sinks};
//! use cumulus3d_cli::events::{Event, EventKind};
//!
//! fn main() -> Result<(), String> {
//!     let recon = Recon::declare()
//!         .input("data")                      // data/images/<cam>/, data/gps_ref.txt
//!         .preset("aerial-formation")         // camF/camR/camL, formation pairing rule, stride 3
//!         .stride(3)
//!         .zones(12, 2)
//!         .dense(Dense::profile("fast").fusion_min_views(3))
//!         .sinks(Sinks::default_files("out").models(true))
//!         .seed(7)
//!         .on(EventKind::PositionDone, |e: &Event| {
//!             if let Some(t) = e.timeline_text() {
//!                 println!("{t}");
//!             }
//!         })
//!         .on_zone_preview(|range, cloud| println!("preview {}: {} points", range.zone, cloud.len()))
//!         .build()                            // validation only (errors are collected and reported at once)
//!         .map_err(|e| e.to_string())?;
//!     println!("{}", recon.plan().to_toml());
//!     let summary = recon.run()?;              // actual execution happens here
//!     println!("{} events", summary.events);
//!     Ok(())
//! }
//! ```
//!
//! Run from a TOML plan file: `cumulus3d run plan.toml`. The default plan is `cumulus3d plan --print-default`.
//! Relative paths in a plan file are relative to the working location (current folder).

use crate::blocks::LiveStream;
use crate::densewrap::{
    make_ba_backend, make_match_backend, make_pm_backend, make_sift_backend, make_sparse_backend, make_undistort_backend, parse_profile,
    DenseConfig,
};
use crate::events::{EventKind, Frame, ZoneRange};
use crate::intrinsics::IntrinsicsPrior;
use crate::pipeline::{Pipeline, QueuePolicy, Summary};
use crate::session::{zones_upto, Input, Session, SessionConfig};
use crate::sinks::{DefaultSinks, SinkSet};
use crate::stream::{frame_set_of, Layout, StreamConfig, CAMS};
use cumulus3d_core::io::ply::PointCloud;
use cumulus3d_core::io::{read_gps_file, GpsRecord};
use cumulus3d_dense::{DensifyOptions, DepthMapCache, FusionMode, FusionResidual};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Default preset name.
pub const DEFAULT_PRESET: &str = "aerial-formation";

/// Known preset names.
pub const PRESETS: [&str; 1] = [DEFAULT_PRESET];

// ============================================================================================
// Plan values
// ============================================================================================

/// Reconstruction plan (a value). It runs nothing and only holds "what to do and how".
///
/// The default ([`Plan::default`]) is the `aerial-formation` preset + the `cumulus3d stream` defaults,
/// and the paths are placeholders (`data/images`, `data/gps_ref.txt`, `out`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Plan {
    /// RANSAC seed for two-view geometry and GPS alignment (if unset, differs from run to run).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Number of rayon threads (0 = number of cores).
    pub threads: usize,
    /// GPS record file (`name latitude longitude altitude` lines). Without it, alignment is impossible.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gps: Option<PathBuf>,
    /// Pairing strategy.
    pub pairing: Pairing,
    /// Input images.
    pub input: Source,
    /// Zone splitting.
    pub zones: Zones,
    /// Coordinate alignment.
    pub align: Align,
    /// Feature extraction.
    pub features: Features,
    /// Descriptor matching.
    pub matching: Matching,
    /// Sparse reconstruction.
    pub sparse: Sparse,
    /// Densification.
    pub dense: Dense,
    /// Output hooks.
    pub sinks: Sinks,
}

impl Default for Plan {
    fn default() -> Self {
        Self {
            seed: None,
            threads: 0,
            gps: Some(PathBuf::from("data/gps_ref.txt")),
            pairing: Pairing::Formation,
            input: Source::default(),
            zones: Zones::default(),
            align: Align::default(),
            features: Features::default(),
            matching: Matching::default(),
            sparse: Sparse::default(),
            dense: Dense::default(),
            sinks: Sinks::default_files("out"),
        }
    }
}

/// Input images: `<images>/<camera>/<camera>_NNNN.jpg`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Source {
    /// Image folder (parent of the camera folders).
    pub images: PathBuf,
    /// List of camera folders (frame-bundle order for one position). The position list is taken from the first camera folder.
    pub cameras: Vec<String>,
    /// File number interval: only frames with number % stride == 0 are used. Position = index in the selected frame list.
    pub stride: usize,
    /// Only the first N positions (all if unset).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub positions: Option<usize>,
}

impl Default for Source {
    fn default() -> Self {
        Self { images: PathBuf::from("data/images"), cameras: CAMS.iter().map(|c| c.to_string()).collect(), stride: 3, positions: None }
    }
}

/// Pairing strategy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Pairing {
    /// Formation rule: same camera at position gaps 1..5, 8, 16 / different cameras at position difference 0..4.
    #[default]
    Formation,
}

/// Zones (in units of positions).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Zones {
    /// Zone size.
    pub span: usize,
    /// Zone overlap (< span).
    pub overlap: usize,
}

impl Default for Zones {
    fn default() -> Self {
        Self { span: 12, overlap: 2 }
    }
}

/// Alignment coordinate frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AlignFrame {
    /// Align to the ENU (east, north, up) frame using GPS records (requires the GPS file).
    #[default]
    Enu,
    /// No alignment (the GPS file is not used).
    None,
}

/// Coordinate alignment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Align {
    /// Coordinate frame.
    pub frame: AlignFrame,
    /// Fix the session ENU origin to the first GPS line (if off, the first common record at each alignment).
    pub fixed_origin: bool,
}

/// Feature extraction.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Features {
    /// SIFT backend (cpu|cuda). The result is the same as cpu.
    pub backend: String,
    /// Maximum number of features.
    pub max_num_features: usize,
    /// Maximum input image size.
    pub max_image_size: usize,
}

impl Default for Features {
    fn default() -> Self {
        Self { backend: "cpu".into(), max_num_features: 8192, max_image_size: 3200 }
    }
}

/// Descriptor matching.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Matching {
    /// Matching backend (cpu|cuda). The result is the same as cpu.
    pub backend: String,
}

impl Default for Matching {
    fn default() -> Self {
        Self { backend: "cpu".into() }
    }
}

/// Sparse reconstruction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Sparse {
    /// Triangulate only newly registered images (an improvement that can change the result).
    pub incremental_triangulation: bool,
    /// Two-view geometric verification, registration and triangulation backend (cpu|cuda).
    pub backend: String,
    /// Bundle adjustment backend (cpu|cuda). cuda falls back to the CPU with a warning if it fails at run time.
    pub ba_backend: String,
    /// Stable display (see [`SessionConfig::stable_display`]).
    pub stable_display: bool,
    /// Stable display: settle tolerance (see [`SessionConfig::settle_tol`]).
    pub settle_tol: f64,
    /// Stable display: last zone that may still settle (see [`SessionConfig::settle_max_zone`]).
    pub settle_max_zone: usize,
    /// Soft intrinsics prior file (see [`crate::intrinsics`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intrinsics_prior: Option<PathBuf>,
    /// Intrinsics prior: relative σ of the focal lengths.
    pub intrinsics_prior_sigma: f64,
    /// Intrinsics prior: absolute σ of the distortion parameters.
    pub intrinsics_prior_dist_sigma: f64,
}

impl Default for Sparse {
    fn default() -> Self {
        Self {
            incremental_triangulation: false,
            backend: "cpu".into(),
            ba_backend: "cpu".into(),
            stable_display: false,
            settle_tol: 0.01,
            settle_max_zone: 3,
            intrinsics_prior: None,
            intrinsics_prior_sigma: crate::intrinsics::DEFAULT_FOCAL_SIGMA,
            intrinsics_prior_dist_sigma: crate::intrinsics::DEFAULT_DIST_SIGMA,
        }
    }
}

/// Densification plan. Start from [`Dense::profile`] and add to it with methods. Unset fusion/filter values use the profile defaults.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Dense {
    /// Whether to densify (if off, there is no post-processing point cloud).
    pub enabled: bool,
    /// PatchMatch backend (cuda).
    pub backend: String,
    /// Image undistortion backend (cpu|cuda). Same result as cpu up to rounding.
    pub undistort_backend: String,
    /// Profile (fast|quality).
    pub profile: String,
    /// Maximum undistorted image size.
    pub max_image_size: i64,
    /// Number of source views (1..=32).
    pub views: usize,
    /// Upper bound on the number of scales (>= 1; 1 = a single scale at full resolution). If unset, the profile default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_levels: Option<usize>,
    /// Serialize preview/refined densification backend calls.
    pub serialize: bool,
    /// Reuse depth maps of overlapping images.
    pub depth_cache: bool,
    /// Fusion overrides.
    pub fusion: Fusion,
    /// Depth map filter overrides.
    pub filter: Filter,
}

impl Default for Dense {
    fn default() -> Self {
        Self {
            enabled: true,
            backend: "cuda".into(),
            undistort_backend: "cpu".into(),
            profile: "fast".into(),
            max_image_size: 960,
            views: 10,
            max_levels: None,
            serialize: false,
            depth_cache: false,
            fusion: Fusion::default(),
            filter: Filter::default(),
        }
    }
}

/// Fusion overrides (same meaning as `--fusion-*` of the `densify` subcommand). If unset, the profile default.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Fusion {
    /// Fusion mode (consistency|traversal).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Consistent fusion: minimum number of agreeing views besides the reference (>= 1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_views: Option<usize>,
    /// Relative depth tolerance (> 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_error: Option<f64>,
    /// Normal tolerance angle (degrees, (0, 180]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normal_error: Option<f64>,
    /// Reprojection tolerance (pixels, > 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reproj_error: Option<f64>,
    /// Handling of remaining pixels (none|release|second-pass).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residual: Option<String>,
    /// Smoothing of the fused cloud along normals: iterations (0 = off).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub smooth_iters: Option<usize>,
    /// Smoothing neighbour radius (multiple of the depth-map ground sample distance, > 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub smooth_radius: Option<f64>,
    /// Smoothing range sigma along the normal (ground-sample-distance multiple, > 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub smooth_range: Option<f64>,
    /// Smoothing neighbour normal gate (degrees, (0, 180]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub smooth_normal: Option<f64>,
    /// Free-space visibility filter: neighbouring views checked per fused point (0 = off).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility_views: Option<usize>,
    /// Free-space visibility filter: relative depth tolerance (> 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility_tau: Option<f64>,
    /// Free-space visibility filter: minimum violating views to remove a point.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility_min_violations: Option<usize>,
}

/// Depth map filter overrides (same meaning as `--filter-*` and `--median-filter` of the `densify` subcommand). If unset, the profile
/// default.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Filter {
    /// Minimum number of agreeing views.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_views: Option<u32>,
    /// Forward/backward reprojection tolerance (pixels, > 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geom_error: Option<f32>,
    /// Minimum NCC ([-1, 1]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_ncc: Option<f32>,
    /// 5x5 median plane filter before readout.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub median: Option<bool>,
    /// Edge-aware hole filling of the filtered depth maps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fill_holes: Option<bool>,
}

impl Dense {
    /// Start from a profile (fast|quality) and the defaults (on, backend cuda).
    pub fn profile(name: &str) -> Self {
        Self { profile: name.to_string(), ..Self::default() }
    }
    /// Densification turned off.
    pub fn off() -> Self {
        Self { enabled: false, ..Self::default() }
    }
    /// PatchMatch backend name.
    pub fn backend(mut self, name: &str) -> Self {
        self.backend = name.to_string();
        self
    }
    /// Image undistortion backend name (cpu|cuda).
    pub fn undistort_backend(mut self, name: &str) -> Self {
        self.undistort_backend = name.to_string();
        self
    }
    /// Maximum undistorted image size.
    pub fn max_image_size(mut self, n: i64) -> Self {
        self.max_image_size = n;
        self
    }
    /// Number of source views.
    pub fn views(mut self, n: usize) -> Self {
        self.views = n;
        self
    }
    /// Serialize preview/refined backend calls.
    pub fn serialize(mut self, on: bool) -> Self {
        self.serialize = on;
        self
    }
    /// Reuse depth maps of overlapping images.
    pub fn depth_cache(mut self, on: bool) -> Self {
        self.depth_cache = on;
        self
    }
    /// Fusion mode (consistency|traversal).
    pub fn fusion_mode(mut self, mode: &str) -> Self {
        self.fusion.mode = Some(mode.to_string());
        self
    }
    /// Consistent fusion: minimum number of agreeing views besides the reference.
    pub fn fusion_min_views(mut self, n: usize) -> Self {
        self.fusion.min_views = Some(n);
        self
    }
    /// Fusion relative depth tolerance.
    pub fn fusion_depth_error(mut self, v: f64) -> Self {
        self.fusion.depth_error = Some(v);
        self
    }
    /// Fusion normal tolerance angle (degrees).
    pub fn fusion_normal_error(mut self, deg: f64) -> Self {
        self.fusion.normal_error = Some(deg);
        self
    }
    /// Fusion reprojection tolerance (pixels).
    pub fn fusion_reproj_error(mut self, px: f64) -> Self {
        self.fusion.reproj_error = Some(px);
        self
    }
    /// Handling of remaining pixels in consistent fusion (none|release|second-pass).
    pub fn fusion_residual(mut self, mode: &str) -> Self {
        self.fusion.residual = Some(mode.to_string());
        self
    }
    /// Smoothing of the fused cloud along normals: iterations (0 = off).
    pub fn fusion_smooth_iters(mut self, n: usize) -> Self {
        self.fusion.smooth_iters = Some(n);
        self
    }
    /// Smoothing neighbour radius (ground-sample-distance multiple).
    pub fn fusion_smooth_radius(mut self, r: f64) -> Self {
        self.fusion.smooth_radius = Some(r);
        self
    }
    /// Smoothing range sigma along the normal (ground-sample-distance multiple).
    pub fn fusion_smooth_range(mut self, r: f64) -> Self {
        self.fusion.smooth_range = Some(r);
        self
    }
    /// Smoothing neighbour normal gate (degrees).
    pub fn fusion_smooth_normal(mut self, deg: f64) -> Self {
        self.fusion.smooth_normal = Some(deg);
        self
    }
    /// Upper bound on the number of scales (1 = a single scale at full resolution).
    pub fn max_levels(mut self, n: usize) -> Self {
        self.max_levels = Some(n);
        self
    }
    /// Edge-aware hole filling of the filtered depth maps.
    pub fn fill_holes(mut self, on: bool) -> Self {
        self.filter.fill_holes = Some(on);
        self
    }
    /// Depth map filter: minimum number of agreeing views.
    pub fn filter_min_views(mut self, n: u32) -> Self {
        self.filter.min_views = Some(n);
        self
    }
    /// Depth map filter: forward/backward reprojection tolerance (pixels).
    pub fn filter_geom_error(mut self, px: f32) -> Self {
        self.filter.geom_error = Some(px);
        self
    }
    /// Depth map filter: minimum NCC.
    pub fn filter_min_ncc(mut self, v: f32) -> Self {
        self.filter.min_ncc = Some(v);
        self
    }
    /// Median plane filter before readout.
    pub fn median_filter(mut self, on: bool) -> Self {
        self.filter.median = Some(on);
        self
    }

    /// Apply the overrides to the densification options (an unknown name is an error).
    pub fn apply(&self, o: &mut DensifyOptions) -> Result<(), String> {
        let f = &self.fusion;
        if let Some(v) = self.max_levels {
            o.max_levels = v;
        }
        if let Some(m) = &f.mode {
            o.fusion.mode = parse_fusion_mode(m)?;
        }
        if let Some(v) = f.min_views {
            o.fusion.min_consistent_views = v;
        }
        if let Some(v) = f.depth_error {
            o.fusion.max_depth_error = v;
        }
        if let Some(v) = f.normal_error {
            o.fusion.max_normal_error_deg = v;
        }
        if let Some(v) = f.reproj_error {
            o.fusion.max_reproj_error = v;
        }
        if let Some(v) = f.smooth_iters {
            o.fusion.smooth.iterations = v;
        }
        if let Some(v) = f.smooth_radius {
            o.fusion.smooth.radius_gsd = v;
        }
        if let Some(v) = f.smooth_range {
            o.fusion.smooth.sigma_range_gsd = v;
        }
        if let Some(v) = f.smooth_normal {
            o.fusion.smooth.normal_deg = v;
        }
        if let Some(v) = f.visibility_views {
            o.fusion.visibility.check_views = v;
        }
        if let Some(v) = f.visibility_tau {
            o.fusion.visibility.tau = v;
        }
        if let Some(v) = f.visibility_min_violations {
            o.fusion.visibility.min_violations = v;
        }
        if let Some(m) = &f.residual {
            o.fusion.residual =
                FusionResidual::parse(m).ok_or_else(|| format!("알 수 없는 융합 남은 픽셀 처리 {m} (none|release|second-pass)"))?;
        }
        let g = &self.filter;
        if let Some(v) = g.min_views {
            o.filter.min_num_consistent = v;
        }
        if let Some(v) = g.geom_error {
            o.filter.geom_max_cost = v;
        }
        if let Some(v) = g.min_ncc {
            o.filter.min_ncc = v;
        }
        if let Some(v) = g.median {
            o.filter.median_filter = v;
        }
        if let Some(v) = g.fill_holes {
            o.post.fill_holes = v;
        }
        Ok(())
    }

    /// Densification settings for running. The backend is created here (if there is no device, that error is reported in the
    /// densification stage).
    pub fn config(&self) -> Result<DenseConfig, String> {
        let mut d = DenseConfig::new(make_pm_backend(&self.backend), parse_profile(&self.profile)?);
        d.undistort_backend = make_undistort_backend(&self.undistort_backend)?;
        d.undistort.max_image_size = self.max_image_size;
        d.densify.neighbors.num_views = self.views;
        self.apply(&mut d.densify)?;
        if self.serialize {
            d.lock = Some(Arc::new(Mutex::new(())));
        }
        if self.depth_cache {
            d.depth_cache = Some(Arc::new(DepthMapCache::new()));
        }
        Ok(d)
    }
}

fn parse_fusion_mode(m: &str) -> Result<FusionMode, String> {
    FusionMode::parse(m).ok_or_else(|| format!("알 수 없는 융합 방식 {m} (consistency|traversal)"))
}

/// Output hook plan. If `out` is unset, no files are written (enabled file items are ignored, only hooks remain).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Sinks {
    /// Output folder.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out: Option<PathBuf>,
    /// Delete and recreate the output folder when the run starts.
    pub clean: bool,
    /// Also write the run.log contents to standard output.
    pub echo: bool,
    /// `timeline.txt`.
    pub timeline: bool,
    /// `run.log`, `DONE`.
    pub run_log: bool,
    /// Zone PLY (`full/preview`, `full/refined`).
    pub zone_ply: bool,
    /// Alignment, snapshots, manifest, final_frame.
    pub snapshots: bool,
    /// Per-zone sparse models (`work/models/`).
    pub models: bool,
    /// Live display thinning: keep 1 of `live_stride` points by key hash (1 = keep all; ≥ 1).
    pub live_stride: usize,
}

impl Default for Sinks {
    /// [`Sinks::default_files`] without an output folder (the remaining values when only part of `[sinks]` is written in TOML).
    fn default() -> Self {
        Self { out: None, ..Self::default_files("") }
    }
}

impl Sinks {
    /// The same file output as `cumulus3d stream` (excluding sparse models, also logged to standard output, folder recreated at start).
    pub fn default_files(out: impl Into<PathBuf>) -> Self {
        Self {
            out: Some(out.into()),
            clean: true,
            echo: true,
            timeline: true,
            run_log: true,
            zone_ply: true,
            snapshots: true,
            models: false,
            live_stride: crate::post::LiveOptions::default().stride,
        }
    }
    /// No file output (no output folder, no standard output logging).
    pub fn none() -> Self {
        Self {
            out: None,
            clean: false,
            echo: false,
            timeline: false,
            run_log: false,
            zone_ply: false,
            snapshots: false,
            models: false,
            live_stride: crate::post::LiveOptions::default().stride,
        }
    }
    /// Whether to delete the output folder at start.
    pub fn clean(mut self, on: bool) -> Self {
        self.clean = on;
        self
    }
    /// Whether to log to standard output.
    pub fn echo(mut self, on: bool) -> Self {
        self.echo = on;
        self
    }
    /// Whether to write `timeline.txt`.
    pub fn timeline(mut self, on: bool) -> Self {
        self.timeline = on;
        self
    }
    /// Whether to write `run.log`.
    pub fn run_log(mut self, on: bool) -> Self {
        self.run_log = on;
        self
    }
    /// Whether to write zone PLY.
    pub fn zone_ply(mut self, on: bool) -> Self {
        self.zone_ply = on;
        self
    }
    /// Whether to write snapshot/alignment output.
    pub fn snapshots(mut self, on: bool) -> Self {
        self.snapshots = on;
        self
    }
    /// Whether to write per-zone sparse models.
    pub fn models(mut self, on: bool) -> Self {
        self.models = on;
        self
    }
    /// Live display thinning stride (1 = keep all).
    pub fn live_stride(mut self, stride: usize) -> Self {
        self.live_stride = stride;
        self
    }
    fn set(&self) -> SinkSet {
        SinkSet {
            timeline: self.timeline,
            run_log: self.run_log,
            zone_ply: self.zone_ply,
            models: self.models,
            snapshots: self.snapshots,
            live_stream: None,
            live_stride: self.live_stride,
        }
    }
}

impl Plan {
    /// As a TOML string.
    pub fn to_toml(&self) -> String {
        toml::to_string(self).expect("계획은 항상 TOML 로 바꿀 수 있다")
    }

    /// TOML with the profile's effective dense defaults written as comments under empty
    /// `[dense.fusion]` / `[dense.filter]` tables (what `cumulus3d plan --print-default` prints).
    /// Comments do not change the plan: `from_toml` of the result equals `self`.
    pub fn to_toml_annotated(&self) -> String {
        let s = self.to_toml();
        let Ok(profile) = parse_profile(&self.dense.profile) else { return s };
        let o = DensifyOptions::with_profile(profile);
        let (f, g) = (&o.fusion, &o.filter);
        let mode = match f.mode {
            FusionMode::Consistency => "consistency",
            FusionMode::Traversal => "traversal",
        };
        let fusion = format!(
            "# {} profile defaults:\n# mode = \"{mode}\"\n# min_views = {}\n# depth_error = {}\n# normal_error = {}\n# reproj_error = {}\n# residual = \"{}\"\n# smooth_iters = {}\n# smooth_radius = {}\n# smooth_range = {}\n# smooth_normal = {}\n",
            self.dense.profile, f.min_consistent_views, f.max_depth_error, f.max_normal_error_deg, f.max_reproj_error, f.residual.name(), f.smooth.iterations, f.smooth.radius_gsd, f.smooth.sigma_range_gsd, f.smooth.normal_deg
        );
        let filter = format!(
            "# {} profile defaults:\n# min_views = {}\n# geom_error = {}\n# min_ncc = {}\n# median = {}\n# fill_holes = {}\n",
            self.dense.profile, g.min_num_consistent, g.geom_max_cost, g.min_ncc, g.median_filter, o.post.fill_holes
        );
        let mut out = String::with_capacity(s.len() + fusion.len() + filter.len());
        let mut lines = s.lines().peekable();
        while let Some(l) = lines.next() {
            out.push_str(l);
            out.push('\n');
            let empty = lines.peek().is_none_or(|n| n.trim().is_empty() || n.starts_with('['));
            if empty && l == "[dense.fusion]" {
                out.push_str(&fusion);
            } else if empty && l == "[dense.filter]" {
                out.push_str(&filter);
            }
        }
        out
    }

    /// From a TOML string (missing items take defaults, unknown items are an error).
    pub fn from_toml(s: &str) -> Result<Self, String> {
        toml::from_str(s).map_err(|e| format!("계획 TOML: {e}"))
    }

    /// From a TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let p = path.as_ref();
        let s = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
        Self::from_toml(&s).map_err(|e| format!("{}: {e}", p.display()))
    }

    /// The plan equivalent to the `cumulus3d stream` settings (`<src>/images`, `<src>/gps_ref.txt`, aerial-formation cameras).
    pub fn from_stream(c: &StreamConfig) -> Self {
        Self {
            seed: c.seed,
            threads: c.threads,
            gps: Some(c.src.join("gps_ref.txt")),
            pairing: Pairing::Formation,
            input: Source {
                images: c.src.join("images"),
                cameras: CAMS.iter().map(|s| s.to_string()).collect(),
                stride: c.stride,
                positions: c.max_positions,
            },
            zones: Zones { span: c.span, overlap: c.overlap },
            align: Align { frame: AlignFrame::Enu, fixed_origin: c.fixed_enu_origin },
            features: Features {
                backend: c.sift_backend.clone(),
                max_num_features: c.max_num_features,
                max_image_size: c.sift_max_image_size,
            },
            matching: Matching { backend: c.match_backend.clone() },
            sparse: Sparse {
                incremental_triangulation: c.incremental_triangulation,
                stable_display: c.stable_display,
                settle_tol: c.settle_tol,
                settle_max_zone: c.settle_max_zone,
                intrinsics_prior: c.intrinsics_prior.clone(),
                intrinsics_prior_sigma: c.intrinsics_prior_sigma,
                intrinsics_prior_dist_sigma: c.intrinsics_prior_dist_sigma,
                ..Default::default()
            },
            dense: Dense {
                enabled: !c.no_dense,
                backend: c.pm_backend.clone(),
                undistort_backend: c.undistort_backend.clone(),
                profile: c.mvs_profile.clone(),
                max_image_size: c.dense_max_image_size,
                views: c.number_views,
                max_levels: c.max_levels,
                serialize: c.serialize_dense,
                depth_cache: c.depth_cache,
                fusion: c.fusion.clone(),
                filter: Filter::default(),
            },
            sinks: Sinks { models: c.save_models, echo: c.echo, live_stride: c.live_stride, ..Sinks::default_files(&c.out) },
        }
    }
}

// ============================================================================================
// Validation errors
// ============================================================================================

/// One plan validation problem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    /// Plan item (TOML path, e.g. `zones.overlap`).
    pub field: String,
    /// Description.
    pub message: String,
}

/// Validation error of [`ReconBuilder::build`]. It holds all the problems found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanError {
    /// List of problems (in the order found).
    pub problems: Vec<Problem>,
}

impl PlanError {
    /// Whether there is a problem for the given item.
    pub fn has(&self, field: &str) -> bool {
        self.problems.iter().any(|p| p.field == field)
    }
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "계획 검사 실패({} 건)", self.problems.len())?;
        for p in &self.problems {
            write!(f, "\n  - {}: {}", p.field, p.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for PlanError {}

#[derive(Default)]
struct Problems(Vec<Problem>);

impl Problems {
    fn add(&mut self, field: &str, message: impl Into<String>) {
        self.0.push(Problem { field: field.to_string(), message: message.into() });
    }
    fn check(&mut self, ok: bool, field: &str, message: impl FnOnce() -> String) {
        if !ok {
            self.add(field, message());
        }
    }
}

// ============================================================================================
// Builder
// ============================================================================================

type Attach = Box<dyn FnOnce(Pipeline<Session>) -> Pipeline<Session>>;

/// Plan recorder. Methods only record and do not run anything. Validate with [`ReconBuilder::build`].
pub struct ReconBuilder {
    plan: Plan,
    attach: Vec<Attach>,
    pending: Vec<Problem>,
    frames: bool,
}

impl fmt::Debug for ReconBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReconBuilder").field("plan", &self.plan).field("hooks", &self.attach.len()).field("pending", &self.pending).finish()
    }
}

impl ReconBuilder {
    /// The plan recorded so far.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Input folder: images `<src>/images/<camera>/`, GPS `<src>/gps_ref.txt`.
    pub fn input(mut self, src: impl AsRef<Path>) -> Self {
        let src = src.as_ref();
        self.plan.input.images = src.join("images");
        self.plan.gps = Some(src.join("gps_ref.txt"));
        self
    }

    /// Only specify the image folder (parent of the camera folders).
    pub fn images(mut self, dir: impl Into<PathBuf>) -> Self {
        self.plan.input.images = dir.into();
        self
    }

    /// List of camera folders (frame-bundle order).
    pub fn cameras<S: Into<String>>(mut self, cams: impl IntoIterator<Item = S>) -> Self {
        self.plan.input.cameras = cams.into_iter().map(Into::into).collect();
        self
    }

    /// Apply a preset (cameras, pairing strategy, stride). An unknown name is an error in `build`.
    /// `aerial-formation`: camF/camR/camL, formation pairing rule, stride 3.
    pub fn preset(mut self, name: &str) -> Self {
        match name {
            DEFAULT_PRESET => {
                self.plan.input.cameras = CAMS.iter().map(|c| c.to_string()).collect();
                self.plan.input.stride = 3;
                self.plan.pairing = Pairing::Formation;
            }
            other => self
                .pending
                .push(Problem { field: "preset".into(), message: format!("알 수 없는 프리셋 {other} ({})", PRESETS.join("|")) }),
        }
        self
    }

    /// File number interval.
    pub fn stride(mut self, n: usize) -> Self {
        self.plan.input.stride = n;
        self
    }

    /// Only the first N positions.
    pub fn positions(mut self, n: usize) -> Self {
        self.plan.input.positions = Some(n);
        self
    }

    /// Pairing strategy.
    pub fn pairing(mut self, p: Pairing) -> Self {
        self.plan.pairing = p;
        self
    }

    /// Zone size and overlap (number of positions).
    pub fn zones(mut self, span: usize, overlap: usize) -> Self {
        self.plan.zones = Zones { span, overlap };
        self
    }

    /// GPS record file.
    pub fn gps(mut self, path: impl Into<PathBuf>) -> Self {
        self.plan.gps = Some(path.into());
        self
    }

    /// No GPS file (to turn off alignment too, pass [`AlignFrame::None`] to [`ReconBuilder::align`]).
    pub fn no_gps(mut self) -> Self {
        self.plan.gps = None;
        self
    }

    /// Alignment coordinate frame.
    pub fn align(mut self, frame: AlignFrame) -> Self {
        self.plan.align.frame = frame;
        self
    }

    /// Fix the session ENU origin to the first GPS line.
    pub fn fixed_enu_origin(mut self, on: bool) -> Self {
        self.plan.align.fixed_origin = on;
        self
    }

    /// Feature extraction settings.
    pub fn features(mut self, f: Features) -> Self {
        self.plan.features = f;
        self
    }

    /// Matching backend (cpu|cuda).
    pub fn match_backend(mut self, name: &str) -> Self {
        self.plan.matching.backend = name.to_string();
        self
    }

    /// Set the SIFT, matching, sparse-stage, PatchMatch, undistortion and bundle adjustment backends all to cuda.
    pub fn gpu(mut self) -> Self {
        self.plan.features.backend = "cuda".into();
        self.plan.matching.backend = "cuda".into();
        self.plan.sparse.backend = "cuda".into();
        self.plan.dense.backend = "cuda".into();
        self.plan.sparse.ba_backend = "cuda".into();
        self.plan.dense.undistort_backend = "cuda".into();
        self
    }

    /// Triangulate only newly registered images.
    pub fn incremental_triangulation(mut self, on: bool) -> Self {
        self.plan.sparse.incremental_triangulation = on;
        self
    }

    /// Stable display (see [`SessionConfig::stable_display`]).
    pub fn stable_display(mut self, on: bool) -> Self {
        self.plan.sparse.stable_display = on;
        self
    }

    /// Stable display settling: intrinsics change tolerance and the last zone that may still settle
    /// (see [`SessionConfig::settle_tol`], [`SessionConfig::settle_max_zone`]).
    pub fn settle(mut self, tol: f64, max_zone: usize) -> Self {
        self.plan.sparse.settle_tol = tol;
        self.plan.sparse.settle_max_zone = max_zone;
        self
    }

    /// Densification plan (on).
    pub fn dense(mut self, d: Dense) -> Self {
        self.plan.dense = Dense { enabled: true, ..d };
        self
    }

    /// Densification off.
    pub fn no_dense(mut self) -> Self {
        self.plan.dense.enabled = false;
        self
    }

    /// Output hook plan.
    pub fn sinks(mut self, s: Sinks) -> Self {
        self.plan.sinks = s;
        self
    }

    /// RANSAC seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.plan.seed = Some(seed);
        self
    }

    /// Number of rayon threads (0 = number of cores).
    pub fn threads(mut self, n: usize) -> Self {
        self.plan.threads = n;
        self
    }

    fn hook(mut self, f: impl FnOnce(Pipeline<Session>) -> Pipeline<Session> + 'static) -> Self {
        self.attach.push(Box::new(f));
        self
    }

    /// A hook for a specific kind ([`Pipeline::on`]).
    /// Choosing `EventKind::FrameDecoded` turns frame decoding on.
    pub fn on<F>(mut self, kind: EventKind, f: F) -> Self
    where
        F: FnMut(&crate::events::Event) + Send + 'static,
    {
        self.frames |= kind == EventKind::FrameDecoded;
        self.hook(move |p| p.on(kind, f))
    }

    /// Frame hook ([`Pipeline::on_frame`]): receives each decoded input frame (RGB), one camera at a time, before the position is
    /// processed.
    /// Registering one turns frame decoding on.
    pub fn on_frame<F>(mut self, f: F) -> Self
    where
        F: FnMut(&Arc<Frame>) + Send + 'static,
    {
        self.frames = true;
        self.hook(move |p| p.on_frame(f))
    }

    /// Hook for all events ([`Pipeline::on_any`]).
    pub fn on_any<F>(self, f: F) -> Self
    where
        F: FnMut(&crate::events::Event) + Send + 'static,
    {
        self.hook(move |p| p.on_any(f))
    }

    /// Zone preview point-cloud hook ([`Pipeline::on_zone_preview`]).
    pub fn on_zone_preview<F>(self, f: F) -> Self
    where
        F: FnMut(ZoneRange, &Arc<PointCloud>) + Send + 'static,
    {
        self.hook(move |p| p.on_zone_preview(f))
    }

    /// Zone refined point-cloud hook ([`Pipeline::on_zone_refined`]).
    pub fn on_zone_refined<F>(self, f: F) -> Self
    where
        F: FnMut(ZoneRange, &Arc<PointCloud>) + Send + 'static,
    {
        self.hook(move |p| p.on_zone_refined(f))
    }

    /// Snapshot hook ([`Pipeline::on_snapshot`]).
    pub fn on_snapshot<F>(self, f: F) -> Self
    where
        F: FnMut(usize, &Arc<PointCloud>) + Send + 'static,
    {
        self.hook(move |p| p.on_snapshot(f))
    }

    /// Position-done hook ([`Pipeline::on_position_done`]).
    pub fn on_position_done<F>(self, f: F) -> Self
    where
        F: FnMut(usize, usize, usize) + Send + 'static,
    {
        self.hook(move |p| p.on_position_done(f))
    }

    /// Warning/error message hook ([`Pipeline::on_message`]).
    pub fn on_message<F>(self, f: F) -> Self
    where
        F: FnMut(EventKind, &str) + Send + 'static,
    {
        self.hook(move |p| p.on_message(f))
    }

    /// Per-kind queue policy ([`Pipeline::policy`]).
    pub fn policy(self, kind: EventKind, p: QueuePolicy) -> Self {
        self.hook(move |pl| pl.policy(kind, p))
    }

    /// Validate the plan -> an executable [`Recon`]. It neither creates nor deletes files or folders (it only reads the input folder
    /// listing and the GPS file).
    ///
    /// Checks: input and camera folders exist, number of selected positions, alignment and GPS file, backend names and device (when CUDA
    /// is requested),
    /// output folder writable, option value ranges. All problems are collected and returned at once as a [`PlanError`].
    pub fn build(self) -> Result<Recon, PlanError> {
        let ReconBuilder { plan, attach, pending, frames } = self;
        let mut pr = Problems(pending);
        let (layout, gps) = check(&plan, &mut pr);
        if !pr.0.is_empty() {
            return Err(PlanError { problems: pr.0 });
        }
        let (Some(layout), Some(gps)) = (layout, gps) else {
            unreachable!("문제가 없으면 입력과 GPS 를 읽었다");
        };
        Ok(Recon { plan, attach, layout, gps, frames })
    }
}

fn backend_ok(pr: &mut Problems, field: &str, name: &str, allowed: &[&str]) {
    if !allowed.contains(&name) {
        pr.add(field, format!("알 수 없는 백엔드 {name} ({})", allowed.join("|")));
    } else if name == "cuda" && !cumulus3d_cuda::is_available() {
        pr.add(field, "CUDA 장치를 찾지 못함 — cuda 백엔드를 쓸 수 없다");
    }
}

/// Whether the nearest existing ancestor folder is writable (checked without creating anything).
fn writable_target(p: &Path) -> Result<(), String> {
    if p.exists() {
        let m = std::fs::metadata(p).map_err(|e| format!("{}: {e}", p.display()))?;
        if !m.is_dir() {
            return Err(format!("{} 는 폴더가 아님", p.display()));
        }
        if m.permissions().readonly() {
            return Err(format!("{} 에 쓸 수 없음", p.display()));
        }
        return Ok(());
    }
    let mut a = p.parent();
    while let Some(d) = a {
        let d = if d.as_os_str().is_empty() { Path::new(".") } else { d };
        if d.exists() {
            let m = std::fs::metadata(d).map_err(|e| format!("{}: {e}", d.display()))?;
            if !m.is_dir() {
                return Err(format!("{} 는 폴더가 아님", d.display()));
            }
            if m.permissions().readonly() {
                return Err(format!("{} 에 쓸 수 없음", d.display()));
            }
            return Ok(());
        }
        a = d.parent();
    }
    Ok(())
}

fn check(plan: &Plan, pr: &mut Problems) -> (Option<Layout>, Option<Vec<GpsRecord>>) {
    // Value ranges.
    let inp = &plan.input;
    pr.check(inp.stride >= 1, "input.stride", || "1 이상이어야 함".into());
    pr.check(inp.positions != Some(0), "input.positions", || "1 이상이어야 함".into());
    pr.check(!inp.cameras.is_empty(), "input.cameras", || "카메라 폴더 목록이 비어 있음".into());
    let mut seen = std::collections::BTreeSet::new();
    for c in &inp.cameras {
        if c.is_empty() || c.contains(['/', '\\']) {
            pr.add("input.cameras", format!("잘못된 카메라 폴더 이름 {c:?}"));
        } else if !seen.insert(c.as_str()) {
            pr.add("input.cameras", format!("카메라 {c} 중복"));
        }
    }
    let z = plan.zones;
    pr.check(z.span >= 1, "zones.span", || "1 이상이어야 함".into());
    pr.check(z.overlap < z.span.max(1), "zones.overlap", || format!("구역 크기({})보다 작아야 함 (지금 {})", z.span, z.overlap));
    let f = &plan.features;
    backend_ok(pr, "features.backend", &f.backend, &["cpu", "cuda"]);
    pr.check(f.max_num_features >= 1, "features.max_num_features", || "1 이상이어야 함".into());
    pr.check(f.max_image_size >= 16, "features.max_image_size", || "16 이상이어야 함".into());
    backend_ok(pr, "matching.backend", &plan.matching.backend, &["cpu", "cuda"]);
    backend_ok(pr, "sparse.backend", &plan.sparse.backend, &["cpu", "cuda"]);
    backend_ok(pr, "sparse.ba_backend", &plan.sparse.ba_backend, &["cpu", "cuda"]);
    backend_ok(pr, "dense.undistort_backend", &plan.dense.undistort_backend, &["cpu", "cuda"]);

    let d = &plan.dense;
    if d.enabled {
        if d.backend != "cuda" {
            pr.add("dense.backend", format!("알 수 없는 백엔드 {} (cuda)", d.backend));
        } else if !cumulus3d_cuda::is_available() {
            pr.add("dense.backend", "조밀화에는 CUDA 장치가 필요한데 장치를 찾지 못함 — `.no_dense()` 또는 [dense] enabled = false");
        }
        if let Err(e) = parse_profile(&d.profile) {
            pr.add("dense.profile", e);
        }
        pr.check(d.max_image_size >= 16, "dense.max_image_size", || "16 이상이어야 함".into());
        pr.check((1..=32).contains(&d.views), "dense.views", || format!("1..=32 이어야 함 (지금 {})", d.views));
        if let Some(v) = d.max_levels {
            pr.check(v >= 1, "dense.max_levels", || "1 이상이어야 함".into());
        }
        let fu = &d.fusion;
        if let Some(m) = &fu.mode {
            if let Err(e) = parse_fusion_mode(m) {
                pr.add("dense.fusion.mode", e);
            }
        }
        if let Some(m) = &fu.residual {
            pr.check(FusionResidual::parse(m).is_some(), "dense.fusion.residual", || {
                format!("알 수 없는 처리 {m} (none|release|second-pass)")
            });
        }
        if let Some(v) = fu.min_views {
            pr.check(v >= 1, "dense.fusion.min_views", || "1 이상이어야 함".into());
        }
        if let Some(v) = fu.depth_error {
            pr.check(v > 0.0 && v.is_finite(), "dense.fusion.depth_error", || format!("양수여야 함 (지금 {v})"));
        }
        if let Some(v) = fu.normal_error {
            pr.check(v > 0.0 && v <= 180.0, "dense.fusion.normal_error", || format!("(0, 180] 이어야 함 (지금 {v})"));
        }
        if let Some(v) = fu.reproj_error {
            pr.check(v > 0.0 && v.is_finite(), "dense.fusion.reproj_error", || format!("양수여야 함 (지금 {v})"));
        }
        if let Some(v) = fu.visibility_tau {
            pr.check(v > 0.0 && v < 1.0, "dense.fusion.visibility_tau", || format!("(0, 1) 이어야 함 (지금 {v})"));
        }
        if let Some(v) = fu.smooth_radius {
            pr.check(v > 0.0 && v.is_finite(), "dense.fusion.smooth_radius", || format!("양수여야 함 (지금 {v})"));
        }
        if let Some(v) = fu.smooth_range {
            pr.check(v > 0.0 && v.is_finite(), "dense.fusion.smooth_range", || format!("양수여야 함 (지금 {v})"));
        }
        if let Some(v) = fu.smooth_normal {
            pr.check(v > 0.0 && v <= 180.0, "dense.fusion.smooth_normal", || format!("(0, 180] 이어야 함 (지금 {v})"));
        }
        let fi = &d.filter;
        if let Some(v) = fi.geom_error {
            pr.check(v > 0.0 && v.is_finite(), "dense.filter.geom_error", || format!("양수여야 함 (지금 {v})"));
        }
        if let Some(v) = fi.min_ncc {
            pr.check((-1.0..=1.0).contains(&v), "dense.filter.min_ncc", || format!("[-1, 1] 이어야 함 (지금 {v})"));
        }
    }

    // Intrinsics prior file.
    if let Some(p) = &plan.sparse.intrinsics_prior {
        if let Err(e) = IntrinsicsPrior::load(p, plan.sparse.intrinsics_prior_sigma, plan.sparse.intrinsics_prior_dist_sigma) {
            pr.add("sparse.intrinsics_prior", e);
        }
    }

    // Output.
    let s = &plan.sinks;
    pr.check(s.live_stride >= 1, "sinks.live_stride", || "must be >= 1 (1 = keep all)".into());
    if let Some(out) = &s.out {
        if let Err(e) = writable_target(out) {
            pr.add("sinks.out", e);
        }
        if s.clean && (inp.images.starts_with(out) || plan.gps.as_ref().is_some_and(|g| g.starts_with(out))) {
            pr.add("sinks.out", format!("시작 때 지울 출력 폴더 {} 안에 입력이 있음", out.display()));
        }
    }

    // Input folders.
    let mut layout = None;
    if !inp.images.is_dir() {
        pr.add("input.images", format!("영상 폴더 없음: {}", inp.images.display()));
    } else {
        let missing: Vec<&String> = inp.cameras.iter().filter(|c| !inp.images.join(c).is_dir()).collect();
        for c in &missing {
            pr.add("input.cameras", format!("카메라 폴더 없음: {}", inp.images.join(c).display()));
        }
        if missing.is_empty() && !inp.cameras.is_empty() && inp.stride >= 1 {
            match Layout::discover_in(&inp.images, &inp.cameras, inp.stride) {
                Ok(mut l) => {
                    if let Some(n) = inp.positions {
                        l.truncate(n);
                    }
                    if l.frames.is_empty() {
                        pr.add(
                            "input",
                            format!(
                                "선택된 영상 없음: {}/{c}/{c}_NNNN.jpg 중 번호 % {} == 0 인 파일",
                                inp.images.display(),
                                inp.stride,
                                c = inp.cameras[0]
                            ),
                        );
                    } else {
                        layout = Some(l);
                    }
                }
                Err(e) => pr.add("input.images", e),
            }
        }
    }

    // GPS and alignment.
    let mut gps = None;
    match (plan.align.frame, &plan.gps) {
        (AlignFrame::None, _) => gps = Some(Vec::new()),
        (AlignFrame::Enu, None) => pr.add("gps", "ENU 정렬에는 GPS 파일이 필요함 — `.gps(경로)` 또는 정렬 끔([align] frame = \"none\")"),
        (AlignFrame::Enu, Some(p)) => match read_gps_file(p) {
            Err(e) => pr.add("gps", format!("{}: {e}", p.display())),
            Ok(all) => {
                if let Some(l) = &layout {
                    let sel: Vec<GpsRecord> = all.into_iter().filter(|g| l.pos_of.contains_key(&g.name)).collect();
                    if sel.is_empty() {
                        pr.add("gps", format!("{}: 선택된 영상의 GPS 기록이 하나도 없음", p.display()));
                    } else {
                        gps = Some(sel);
                    }
                } else {
                    gps = Some(Vec::new());
                }
            }
        },
    }
    (layout, gps)
}

// ============================================================================================
// Execution
// ============================================================================================

/// An executable reconstruction that has passed validation. Run it with [`Recon::run`].
pub struct Recon {
    plan: Plan,
    attach: Vec<Attach>,
    layout: Layout,
    gps: Vec<GpsRecord>,
    frames: bool,
}

impl fmt::Debug for Recon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Recon")
            .field("plan", &self.plan)
            .field("hooks", &self.attach.len())
            .field("positions", &self.layout.frames.len())
            .finish()
    }
}

impl Recon {
    /// Start recording with the default plan ([`Plan::default`]).
    pub fn declare() -> ReconBuilder {
        Self::from_plan(Plan::default())
    }

    /// Start recording with the given plan (e.g. a plan read from TOML).
    pub fn from_plan(plan: Plan) -> ReconBuilder {
        ReconBuilder { plan, attach: Vec::new(), pending: Vec::new(), frames: false }
    }

    /// The plan.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Number of selected positions.
    pub fn positions(&self) -> usize {
        self.layout.frames.len()
    }

    /// Position <-> image name layout.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Run: prepare the output folder -> session + pipeline + default output hook + user hooks -> position loop -> wait for background
    /// jobs and shut down.
    /// On a fatal error (session failure) it returns that error without post-processing.
    pub fn run(self) -> Result<Summary, String> {
        let Recon { plan, attach, layout, gps, frames } = self;
        if plan.threads > 0 {
            // Ignore if it was already created (tests etc.).
            let _ = rayon::ThreadPoolBuilder::new().num_threads(plan.threads).build_global();
        }
        // Without a backend, record an error in the densification stage.
        let dense = if plan.dense.enabled { Some(plan.dense.config()?) } else { None };
        let prior = match &plan.sparse.intrinsics_prior {
            Some(p) => {
                Some(Arc::new(IntrinsicsPrior::load(p, plan.sparse.intrinsics_prior_sigma, plan.sparse.intrinsics_prior_dist_sigma)?))
            }
            None => None,
        };
        let s = &plan.sinks;
        let sinks = match &s.out {
            Some(out) => {
                if s.clean && out.exists() {
                    std::fs::remove_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
                }
                let mut set = s.set();
                // Live diff stream: stable display only. In stable display the display frame is the ENU frame of the first
                // GPS record (zone 0 is brought into ENU and every later zone keeps that frame through the GPS priors).
                if plan.sparse.stable_display {
                    let origin = gps.first().map(|g| [g.lat, g.lon, g.alt]);
                    set.live_stream = Some(LiveStream { enu: origin.is_some(), origin_gps: origin });
                }
                Some(DefaultSinks::with_set(out, s.echo, set).map_err(|e| e.to_string())?)
            }
            None => None,
        };
        let npos = layout.frames.len();
        let mut sc = SessionConfig::new(plan.input.images.clone());
        sc.gps = gps;
        sc.span = plan.zones.span;
        sc.overlap = plan.zones.overlap;
        sc.total_positions = Some(npos);
        sc.incremental_triangulation = plan.sparse.incremental_triangulation;
        sc.stable_display = plan.sparse.stable_display;
        sc.settle_tol = plan.sparse.settle_tol;
        sc.settle_max_zone = plan.sparse.settle_max_zone;
        sc.intrinsics_prior = prior;
        sc.fixed_enu_origin = plan.align.fixed_origin;
        sc.seed = plan.seed;
        sc.extraction.sift.max_num_features = plan.features.max_num_features;
        sc.extraction.reader.max_image_size = plan.features.max_image_size;
        sc.sift = make_sift_backend(&plan.features.backend)?;
        sc.matcher = Arc::from(make_match_backend(&plan.matching.backend)?);
        sc.sparse = make_sparse_backend(&plan.sparse.backend)?;
        sc.ba_backend = make_ba_backend(&plan.sparse.ba_backend)?;
        sc.dense = dense;
        // Do not rewind (saves memory).
        sc.history = 0;
        sc.decode_frames = frames;

        let mut pl = Pipeline::new(Session::new(sc));
        if let Some(sinks) = sinks {
            // There is a single output hook (on_any), so even if asynchronous, events are processed in order.
            pl = pl.on_any(sinks.into_hook());
        }
        for a in attach {
            pl = a(pl);
        }

        let (span, ovl) = (plan.zones.span, plan.zones.overlap);
        let mut reg_end: BTreeMap<usize, Vec<ZoneRange>> = BTreeMap::new();
        for z in zones_upto(npos, span, ovl) {
            reg_end.entry(z.hi - 1).or_default().push(z);
        }
        pl.push(Input::Log(format!(
            "regions: {}",
            reg_end
                .iter()
                .map(|(e, v)| format!("{e} ->{}", v.iter().map(|z| format!(" {}:{}:{}", z.zone, z.lo, z.hi)).collect::<String>()))
                .collect::<Vec<_>>()
                .join(", ")
        )));
        let d = &plan.dense;
        pl.push(Input::Log(format!(
            "설정: span {span} overlap {ovl} stride {} 증분삼각측량 {} 깊이캐시 {} 고정원점 {} pm {}({})/sift {}/match {}/ba {} 직렬조밀화 {} 왜곡보정최대 {} 뷰 {} 스레드 {}",
            plan.input.stride,
            plan.sparse.incremental_triangulation,
            d.depth_cache,
            plan.align.fixed_origin,
            d.backend,
            d.profile,
            plan.features.backend,
            plan.matching.backend,
            plan.sparse.ba_backend,
            d.serialize,
            d.max_image_size,
            d.views,
            rayon::current_num_threads()
        )));
        for p in 0..npos {
            pl.push(frame_set_of(&layout, &plan.input.cameras, p).into());
            if let Some(e) = pl.reducer().failed() {
                // Fatal error: stop without post-processing (the hook queue is drained when the pipeline is dropped).
                return Err(e.to_string());
            }
        }
        let (_session, summary) = pl.finish();
        Ok(summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_plan_toml_roundtrip() {
        let p = Plan::default();
        let s = p.to_toml();
        assert_eq!(Plan::from_toml(&s).unwrap(), p, "{s}");
        // Empty document = default plan.
        assert_eq!(Plan::from_toml("").unwrap(), p);
    }

    #[test]
    fn full_plan_toml_roundtrip() {
        let b = Recon::declare()
            .input("some/where")
            .cameras(["a", "b"])
            .stride(2)
            .positions(5)
            .zones(6, 1)
            .fixed_enu_origin(true)
            .incremental_triangulation(true)
            .dense(
                Dense::profile("quality")
                    .fusion_mode("traversal")
                    .fusion_min_views(3)
                    .fusion_depth_error(0.02)
                    .fusion_normal_error(12.5)
                    .fusion_reproj_error(1.5)
                    .fusion_residual("second-pass")
                    .fusion_smooth_normal(30.0)
                    .max_levels(1)
                    .filter_min_views(2)
                    .filter_geom_error(1.25)
                    .filter_min_ncc(0.1)
                    .median_filter(true)
                    .depth_cache(true)
                    .serialize(true),
            )
            .sinks(Sinks::default_files("o").models(true).echo(false))
            .seed(42)
            .threads(3);
        let p = b.plan().clone();
        let s = p.to_toml();
        assert_eq!(Plan::from_toml(&s).unwrap(), p, "{s}");
        assert!(s.contains("[dense.fusion]") && s.contains("min_views = 3"), "{s}");
        assert!(s.contains("smooth_normal = 30.0") && s.contains("max_levels = 1"), "{s}");
    }

    #[test]
    fn example_plan_parses() {
        let s = include_str!("../../../examples/plans/aerial-formation.toml");
        let p = Plan::from_toml(s).unwrap();
        assert_eq!(p.input.cameras, ["camF", "camR", "camL"]);
        assert_eq!(p.pairing, Pairing::Formation);
        assert_eq!(Plan::from_toml(&p.to_toml()).unwrap(), p);
    }

    #[test]
    fn unknown_field_rejected() {
        assert!(Plan::from_toml("[zones]\nspam = 3\n").is_err());
        assert!(Plan::from_toml("pairing = \"grid\"\n").is_err());
    }

    #[test]
    fn stream_plan_matches_builder() {
        let mut c = StreamConfig::new("src".into(), "dst".into());
        (c.stride, c.span, c.overlap, c.seed, c.save_models, c.echo) = (1, 4, 1, Some(7), true, false);
        let b = Recon::declare()
            .input("src")
            .preset(DEFAULT_PRESET)
            .stride(1)
            .zones(4, 1)
            .seed(7)
            .sinks(Sinks::default_files("dst").models(true).echo(false));
        assert_eq!(Plan::from_stream(&c), *b.plan());
    }

    #[test]
    fn build_collects_all_problems() {
        let tmp = std::env::temp_dir().join(format!("cumulus3d_declare_missing_{}", std::process::id()));
        let e = Recon::declare()
            .input(tmp.join("in"))
            .preset("grid")
            .stride(0)
            .zones(4, 4)
            .dense(Dense::profile("slow").views(40).max_levels(0).fusion_smooth_normal(0.0).fusion_mode("magic").filter_min_ncc(2.0))
            .features(Features { backend: "fpga".into(), ..Features::default() })
            .sinks(Sinks::default_files(tmp.join("in")).clean(true))
            .build()
            .unwrap_err();
        for f in [
            "preset",
            "input.stride",
            "zones.overlap",
            "features.backend",
            "dense.profile",
            "dense.views",
            "dense.fusion.mode",
            "dense.max_levels",
            "dense.fusion.smooth_normal",
            "dense.filter.min_ncc",
            "sinks.out",
            "input.images",
            "gps",
        ] {
            assert!(e.has(f), "{f} 없음:\n{e}");
        }
        assert_eq!(e.has("dense.backend"), !cumulus3d_cuda::is_available());
        assert!(!tmp.exists());
        assert!(e.to_string().starts_with("계획 검사 실패("));
    }

    #[test]
    fn live_stride_roundtrip_and_validation() {
        assert_eq!(Sinks::default().live_stride, 6);
        let p = Plan { sinks: Sinks::default_files("o").live_stride(3), ..Plan::default() };
        assert_eq!(Plan::from_toml(&p.to_toml()).unwrap().sinks.live_stride, 3);
        let e = Recon::declare().sinks(Sinks::default_files("o").live_stride(0)).no_dense().build().unwrap_err();
        assert!(e.has("sinks.live_stride"), "{e}");
    }

    #[test]
    fn enu_without_gps_is_error() {
        let e = Recon::declare().no_gps().no_dense().build().unwrap_err();
        assert!(e.has("gps"), "{e}");
        let e = Recon::declare().no_gps().align(AlignFrame::None).no_dense().build().unwrap_err();
        assert!(!e.has("gps"), "{e}");
    }
}
