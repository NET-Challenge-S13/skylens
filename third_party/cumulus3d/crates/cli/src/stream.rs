/*
 * stream.rs
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

//! `cumulus3d stream`: feed the positions of the input folder one by one into the [`Session`](crate::session::Session) reducer
//! ([`Pipeline`](crate::pipeline::Pipeline)),
//! and the default output hooks ([`crate::sinks`]) produce timeline.txt, run.log, point clouds and snapshot files.
//! The computation (features -> pair matching -> first model / continued registration -> per-zone preview and refinement) lives in
//! [`crate::session`].
//! [`run_stream`] converts the settings into a plan ([`crate::declare::Plan`]) and runs it through the declarative layer
//! ([`crate::declare`]).

use crate::declare::{Fusion, Plan, Recon};
use crate::densewrap::{make_match_backend, make_sift_backend, DenseConfig};
use crate::intrinsics::IntrinsicsPrior;
use crate::session::{pairs_for, zones_upto, FrameImage, FrameSet, SessionConfig};
use cumulus3d_core::io::GpsRecord;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Camera folders (script order).
pub const CAMS: [&str; 3] = ["camF", "camR", "camL"];

/// Stream settings. Defaults = script behavior.
#[derive(Clone, Debug)]
pub struct StreamConfig {
    /// Input folder (`images/<cam>/`, `gps_ref.txt`).
    pub src: PathBuf,
    /// Output folder.
    pub out: PathBuf,
    /// Zone size (number of positions).
    pub span: usize,
    /// Zone overlap (number of positions).
    pub overlap: usize,
    /// Frame interval (only number % stride == 0 is used).
    pub stride: usize,
    /// Triangulate only newly registered images (improvement, off by default).
    pub incremental_triangulation: bool,
    /// Reuse depth maps of overlapping images (improvement, off by default).
    pub depth_cache: bool,
    /// Fix the session ENU origin to the first GPS line (improvement, off by default = the first common record at each alignment).
    pub fixed_enu_origin: bool,
    /// PatchMatch backend name (cuda).
    pub pm_backend: String,
    /// Densification profile (fast|quality).
    pub mvs_profile: String,
    /// SIFT backend (cpu|cuda).
    pub sift_backend: String,
    /// Descriptor matching backend (cpu|cuda).
    pub match_backend: String,
    /// Number of rayon threads (0 = number of cores).
    pub threads: usize,
    /// Process only the first N positions.
    pub max_positions: Option<usize>,
    /// Save per-zone sparse models.
    pub save_models: bool,
    /// Serialize preview/refined densification backend calls.
    pub serialize_dense: bool,
    /// Maximum undistortion size (script 960).
    pub dense_max_image_size: i64,
    /// Number of densification source views.
    pub number_views: usize,
    /// Maximum number of SIFT features (script 8192).
    pub max_num_features: usize,
    /// Maximum SIFT input size (default 3200).
    pub sift_max_image_size: usize,
    /// Also show the run.log contents on standard output.
    pub echo: bool,
    /// Skip densification (for development/tests).
    pub no_dense: bool,
    /// Stable display (see [`crate::session::SessionConfig::stable_display`]).
    pub stable_display: bool,
    /// Stable display: intrinsics change below which the display settles (see [`crate::session::SessionConfig::settle_tol`]).
    pub settle_tol: f64,
    /// Stable display: last zone that may still settle (see [`crate::session::SessionConfig::settle_max_zone`]).
    pub settle_max_zone: usize,
    /// Soft intrinsics prior file (see [`crate::intrinsics`]).
    pub intrinsics_prior: Option<PathBuf>,
    /// Intrinsics prior: relative σ of the focal lengths.
    pub intrinsics_prior_sigma: f64,
    /// Intrinsics prior: absolute σ of the distortion parameters.
    pub intrinsics_prior_dist_sigma: f64,
    /// RANSAC seed for two-view geometry and GPS alignment (None = differs from run to run).
    pub seed: Option<u64>,
    /// Upper bound on the number of densification scales (None = profile default; 1 = a single scale at full resolution).
    pub max_levels: Option<usize>,
    /// Fusion overrides (smoothing, visibility; unset values use the profile default).
    pub fusion: Fusion,
    /// Live display thinning: keep 1 of N points by key hash (1 = keep all; ≥ 1).
    pub live_stride: usize,
}

impl StreamConfig {
    /// Default settings (zone 12, overlap 2, stride 3).
    pub fn new(src: PathBuf, out: PathBuf) -> Self {
        Self {
            src,
            out,
            span: 12,
            overlap: 2,
            stride: 3,
            incremental_triangulation: false,
            depth_cache: false,
            fixed_enu_origin: false,
            pm_backend: "cuda".into(),
            mvs_profile: "fast".into(),
            sift_backend: "cpu".into(),
            match_backend: "cpu".into(),
            threads: 0,
            max_positions: None,
            save_models: false,
            serialize_dense: false,
            dense_max_image_size: 960,
            number_views: 10,
            max_num_features: 8192,
            sift_max_image_size: 3200,
            echo: true,
            no_dense: false,
            stable_display: false,
            settle_tol: 0.01,
            settle_max_zone: 3,
            intrinsics_prior: None,
            intrinsics_prior_sigma: crate::intrinsics::DEFAULT_FOCAL_SIGMA,
            intrinsics_prior_dist_sigma: crate::intrinsics::DEFAULT_DIST_SIGMA,
            seed: None,
            max_levels: None,
            fusion: Fusion::default(),
            live_stride: crate::post::LiveOptions::default().stride,
        }
    }
}

/// Position <-> image name. Position = index in the selected frame list (not file number / stride).
#[derive(Clone, Debug)]
pub struct Layout {
    /// Frame number of the input file per position.
    pub frames: Vec<u32>,
    /// Image name -> position.
    pub pos_of: HashMap<String, usize>,
}

impl Layout {
    /// Those of `images/camF/camF_NNNN.jpg` with number % stride == 0 (ascending number).
    pub fn discover(src: &Path, stride: usize) -> Result<Self, String> {
        Self::discover_in(&src.join("images"), &CAMS, stride)
    }

    /// Those of the first camera folder of the image folder `images` (`<cams[0]>/<cams[0]>_NNNN.jpg`) with number % stride == 0
    /// (ascending number).
    /// The name -> position table is built for all cameras.
    pub fn discover_in<S: AsRef<str>>(images: &Path, cams: &[S], stride: usize) -> Result<Self, String> {
        let first = cams.first().ok_or("카메라 폴더 목록이 비어 있음")?.as_ref();
        let dir = images.join(first);
        let rd = std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let mut frames = Vec::new();
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            let Some(stem) = n.strip_prefix(&format!("{first}_")).and_then(|s| s.strip_suffix(".jpg")) else { continue };
            if let Ok(i) = stem.parse::<u32>() {
                if (i as usize).is_multiple_of(stride.max(1)) {
                    frames.push(i);
                }
            }
        }
        frames.sort_unstable();
        frames.dedup();
        let mut l = Self { frames, pos_of: HashMap::new() };
        for p in 0..l.frames.len() {
            for c in cams {
                let n = l.name(c.as_ref(), p);
                l.pos_of.insert(n, p);
            }
        }
        Ok(l)
    }

    /// Keep only the first `n` positions.
    pub fn truncate(&mut self, n: usize) {
        self.frames.truncate(n);
        let frames = self.frames.len();
        self.pos_of.retain(|_, p| *p < frames);
    }

    /// Image name of a camera and position (`cam/cam_NNNN.jpg`).
    pub fn name(&self, cam: &str, p: usize) -> String {
        format!("{cam}/{cam}_{:04}.jpg", self.frames[p])
    }

    /// Position of an image name.
    pub fn position(&self, name: &str) -> Option<usize> {
        self.pos_of.get(name).copied()
    }
}

/// Frame bundle of position p (camera order camF, camR, camL).
pub fn frame_set(layout: &Layout, p: usize) -> FrameSet {
    frame_set_of(layout, &CAMS, p)
}

/// Frame bundle of position p (camera order = `cams`).
pub fn frame_set_of<S: AsRef<str>>(layout: &Layout, cams: &[S], p: usize) -> FrameSet {
    FrameSet::new(cams.iter().map(|c| (c.as_ref().to_string(), layout.name(c.as_ref(), p))))
}

/// Same pairing rule as the script: same camera gaps 1..5, 8, 16 / different cameras at position difference 0..4. Sorted name pairs.
pub fn pairs_for_position(layout: &Layout, p: usize) -> Vec<(String, String)> {
    let frames: Vec<Vec<FrameImage>> = (0..=p).map(|q| frame_set(layout, q).images).collect();
    pairs_for(&frames, p)
}

/// Zones: start = 0, SPAN, ...; [start-OVL, start+SPAN+OVL) intersect [0, NPOS). Completion position = hi-1.
pub fn regions(npos: usize, span: usize, ovl: usize) -> Vec<(usize, usize, usize)> {
    zones_upto(npos, span, ovl).into_iter().map(|z| (z.zone, z.lo, z.hi)).collect()
}

/// Stream settings -> session settings. `gps` = GPS records of the selected images (file order), `dense` = None skips densification.
pub fn session_config(cfg: &StreamConfig, npos: usize, gps: Vec<GpsRecord>, dense: Option<DenseConfig>) -> Result<SessionConfig, String> {
    let mut s = SessionConfig::new(cfg.src.join("images"));
    s.gps = gps;
    s.span = cfg.span;
    s.overlap = cfg.overlap;
    s.total_positions = Some(npos);
    s.incremental_triangulation = cfg.incremental_triangulation;
    s.stable_display = cfg.stable_display;
    s.settle_tol = cfg.settle_tol;
    s.settle_max_zone = cfg.settle_max_zone;
    if let Some(p) = &cfg.intrinsics_prior {
        s.intrinsics_prior = Some(Arc::new(IntrinsicsPrior::load(p, cfg.intrinsics_prior_sigma, cfg.intrinsics_prior_dist_sigma)?));
    }
    s.fixed_enu_origin = cfg.fixed_enu_origin;
    s.seed = cfg.seed;
    s.extraction.sift.max_num_features = cfg.max_num_features;
    s.extraction.reader.max_image_size = cfg.sift_max_image_size;
    s.sift = make_sift_backend(&cfg.sift_backend)?;
    s.matcher = Arc::from(make_match_backend(&cfg.match_backend)?);
    s.dense = dense;
    // The CLI does not rewind (saves memory).
    s.history = 0;
    Ok(s)
}

/// Run the stream: convert the settings into a plan ([`Plan::from_stream`]) and validate and run it through the declarative layer
/// ([`Recon`]).
/// Recreates the output folder. The output is the same as when the session reducer + pipeline + default output hooks are assembled
/// directly.
pub fn run_stream(cfg: StreamConfig) -> Result<(), String> {
    let recon = Recon::from_plan(Plan::from_stream(&cfg)).build().map_err(|e| e.to_string())?;
    recon.run().map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_match_script() {
        // NPOS=26, SPAN 12, OVL 2 → [0,14), [10,26), [22,26)
        assert_eq!(regions(26, 12, 2), vec![(0, 0, 14), (1, 10, 26), (2, 22, 26)]);
        assert_eq!(regions(40, 12, 2), vec![(0, 0, 14), (1, 10, 26), (2, 22, 38), (3, 34, 40)]);
    }

    #[test]
    fn pairs_rule() {
        let frames: Vec<u32> = (0..20).map(|i| i * 3).collect();
        let mut l = Layout { frames, pos_of: HashMap::new() };
        l.pos_of.insert("x".into(), 0);
        let p0 = pairs_for_position(&l, 0);
        // Position 0: 3 pairs between different cameras.
        assert_eq!(p0.len(), 3);
        let p17 = pairs_for_position(&l, 17);
        // Same camera at gap 7 x 3 + different cameras: 6 ordered pairs x 5 positions, of which q=p is a duplicate -> 3 + 6x4 = 27.
        assert_eq!(p17.len(), 21 + 27);
        assert!(p17.contains(&("camF/camF_0003.jpg".to_string(), "camF/camF_0051.jpg".to_string())));
    }
}
