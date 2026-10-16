/*
 * main.rs
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

//! The `cumulus3d` executable.

use clap::{Args, Parser, Subcommand};
use cumulus3d_cli::declare::{Plan, Recon};
use cumulus3d_cli::interop::{self, InteropCmd};
use cumulus3d_cli::stream::{run_stream, StreamConfig};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "cumulus3d", version, about = "한 프로세스 점진 재구성 + 단계별 하위 명령")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// incremental_stream.sh as a single process: position-arrival loop + per-zone preview/refined densification + post-processing.
    Stream(Box<StreamArgs>),
    /// Run from a plan TOML file: validate the plan (reporting every problem) -> position loop, wait for background jobs,
    /// post-processing, all in one go.
    Run(RunArgs),
    /// Plan tools: print the default plan, validate a plan.
    Plan(PlanArgs),
    #[command(flatten)]
    Interop(InteropCmd),
}

#[derive(Args, Debug)]
struct StreamArgs {
    /// Input folder (images/camF|camR|camL/*.jpg, gps_ref.txt).
    #[arg(long)]
    src: PathBuf,
    /// Output folder (deleted and recreated).
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 12)]
    span: usize,
    #[arg(long, default_value_t = 2)]
    overlap: usize,
    /// Only frames with file number % stride == 0. Position = index in the selected frame list.
    #[arg(long, default_value_t = 3)]
    stride: usize,
    /// Triangulate only newly registered images (an improvement that can change the result).
    #[arg(long)]
    incremental_triangulation: bool,
    /// Reuse depth maps of overlapping images.
    #[arg(long)]
    depth_cache: bool,
    /// Fix the session ENU origin to the first GPS line.
    #[arg(long)]
    fixed_enu_origin: bool,
    /// PatchMatch backend (cuda). Without a device, densification fails with an error.
    #[arg(long, default_value = "cuda")]
    pm_backend: String,
    /// Densification profile (fast|quality).
    #[arg(long, default_value = "fast")]
    mvs_profile: String,
    /// SIFT backend (cpu|cuda). The result is the same as cpu.
    #[arg(long, default_value = "cpu")]
    sift_backend: String,
    /// Descriptor matching backend (cpu|cuda). The result is the same as cpu.
    #[arg(long, default_value = "cpu")]
    match_backend: String,
    /// Set --pm-backend/--sift-backend/--match-backend all to cuda.
    #[arg(long)]
    gpu: bool,
    /// Number of rayon threads (0 = number of cores).
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Only the first N positions (for quick tests).
    #[arg(long)]
    max_positions: Option<usize>,
    /// Save the per-zone sparse model (work/models/*).
    #[arg(long)]
    save_models: bool,
    /// Serialize the preview/refined densification backend calls.
    #[arg(long)]
    serialize_dense: bool,
    /// Maximum undistorted image size (script: 960). Lowering it speeds up densification.
    #[arg(long, default_value_t = 960)]
    dense_max_image_size: i64,
    /// Number of densification source views (<= 32).
    #[arg(long, default_value_t = 10)]
    number_views: usize,
    /// Maximum number of SIFT features (script: 8192).
    #[arg(long, default_value_t = 8192)]
    max_num_features: usize,
    /// Maximum SIFT input image size (default 3200).
    #[arg(long, default_value_t = 3200)]
    sift_max_image_size: usize,
    /// Do not show progress on standard output (only in run.log).
    #[arg(long)]
    quiet: bool,
    /// Skip densification (for development/tests; no post-processing output).
    #[arg(long)]
    no_dense: bool,
    /// Stable display: no preview, each refined zone is adjusted with already shown cameras held fixed and stays where
    /// it was shown (local frame); GPS/ENU is applied once to the final outputs.
    #[arg(long)]
    stable_display: bool,
    /// Stable display: the display settles (shown cameras and intrinsics frozen) once the relative focal change between
    /// consecutive refined zones is below this (default 0.01 = 1 %).
    #[arg(long, default_value_t = 0.01)]
    settle_tol: f64,
    /// Stable display: settle by force at this zone at the latest (default 3).
    #[arg(long, default_value_t = 3)]
    settle_max_zone: usize,
    /// Soft intrinsics prior: JSON `{"cameras": {"<camera_id>": [params in the camera model's order]}}` (OPENCV:
    /// fx fy cx cy k1 k2 p1 p2). Cameras start from these values and the bundle adjustment is pulled towards them.
    #[arg(long)]
    intrinsics_prior: Option<PathBuf>,
    /// Intrinsics prior: relative sigma of the focal lengths (residual (f - f0)/(sigma*f0)).
    #[arg(long, default_value_t = 0.005)]
    intrinsics_prior_sigma: f64,
    /// Intrinsics prior: absolute sigma of the distortion parameters.
    #[arg(long, default_value_t = 0.01)]
    intrinsics_prior_dist_sigma: f64,
    /// Fix the RANSAC seed (two-view geometry, GPS alignment). Without it, it differs from run to run.
    #[arg(long)]
    seed: Option<u64>,
    /// Live display thinning: keep 1 of N points (key hash); 1 = keep all. Not applied to merged.ply.
    #[arg(long, default_value_t = 6, value_parser = parse_live_stride)]
    live_stride: usize,
    /// Upper bound on the number of densification scales (1 = a single scale at full resolution; unset = profile default).
    #[arg(long)]
    max_levels: Option<usize>,
    /// Fused-cloud smoothing along normals: iterations (0 = off).
    #[arg(long)]
    fusion_smooth_iters: Option<usize>,
    /// Fused-cloud smoothing: neighbour radius (GSD multiple).
    #[arg(long)]
    fusion_smooth_radius: Option<f64>,
    /// Fused-cloud smoothing: range sigma along the normal (GSD multiple).
    #[arg(long)]
    fusion_smooth_range: Option<f64>,
    /// Fused-cloud smoothing: neighbour normal gate (degrees).
    #[arg(long)]
    fusion_smooth_normal: Option<f64>,
    /// Free-space visibility filter: neighbouring views checked per fused point (0 = off).
    #[arg(long)]
    visibility_views: Option<usize>,
    /// Free-space visibility filter: relative depth tolerance.
    #[arg(long)]
    visibility_tau: Option<f64>,
    /// Free-space visibility filter: minimum violating views to remove a point.
    #[arg(long)]
    visibility_min_violations: Option<usize>,
}

#[derive(Args, Debug)]
struct RunArgs {
    /// Plan file (TOML). Relative paths are relative to the working location.
    plan: PathBuf,
    /// Only validate; do not run.
    #[arg(long)]
    check: bool,
}

#[derive(Args, Debug)]
struct PlanArgs {
    /// Print the default plan (aerial-formation preset) as TOML to standard output.
    #[arg(long)]
    print_default: bool,
    /// Validate the plan file (do not run).
    #[arg(long)]
    check: Option<PathBuf>,
}

fn parse_live_stride(s: &str) -> Result<usize, String> {
    match s.parse::<usize>() {
        Ok(n) if n >= 1 => Ok(n),
        _ => Err("must be an integer >= 1".into()),
    }
}

fn check_plan(path: &PathBuf) -> Result<Recon, String> {
    Recon::from_plan(Plan::load(path)?).build().map_err(|e| e.to_string())
}

fn main() {
    let cli = Cli::parse();
    let r = match cli.cmd {
        Cmd::Stream(a) => {
            let a = *a;
            let mut c = StreamConfig::new(a.src, a.out);
            c.span = a.span;
            c.overlap = a.overlap;
            c.stride = a.stride;
            c.incremental_triangulation = a.incremental_triangulation;
            c.depth_cache = a.depth_cache;
            c.fixed_enu_origin = a.fixed_enu_origin;
            c.mvs_profile = a.mvs_profile;
            c.pm_backend = a.pm_backend;
            c.sift_backend = a.sift_backend;
            c.match_backend = a.match_backend;
            if a.gpu {
                c.pm_backend = "cuda".into();
                c.sift_backend = "cuda".into();
                c.match_backend = "cuda".into();
            }
            c.threads = a.threads;
            c.max_positions = a.max_positions;
            c.save_models = a.save_models;
            c.serialize_dense = a.serialize_dense;
            c.dense_max_image_size = a.dense_max_image_size;
            c.number_views = a.number_views;
            c.max_num_features = a.max_num_features;
            c.sift_max_image_size = a.sift_max_image_size;
            c.echo = !a.quiet;
            c.no_dense = a.no_dense;
            c.stable_display = a.stable_display;
            c.settle_tol = a.settle_tol;
            c.settle_max_zone = a.settle_max_zone;
            c.intrinsics_prior = a.intrinsics_prior;
            c.intrinsics_prior_sigma = a.intrinsics_prior_sigma;
            c.intrinsics_prior_dist_sigma = a.intrinsics_prior_dist_sigma;
            c.seed = a.seed;
            c.max_levels = a.max_levels;
            c.live_stride = a.live_stride;
            c.fusion.smooth_iters = a.fusion_smooth_iters;
            c.fusion.smooth_radius = a.fusion_smooth_radius;
            c.fusion.smooth_range = a.fusion_smooth_range;
            c.fusion.smooth_normal = a.fusion_smooth_normal;
            c.fusion.visibility_views = a.visibility_views;
            c.fusion.visibility_tau = a.visibility_tau;
            c.fusion.visibility_min_violations = a.visibility_min_violations;
            run_stream(c)
        }
        Cmd::Run(a) => check_plan(&a.plan).and_then(|r| if a.check { Ok(()) } else { r.run().map(|_| ()) }),
        Cmd::Plan(a) => {
            if a.print_default {
                print!("{}", Plan::default().to_toml_annotated());
                Ok(())
            } else if let Some(p) = a.check {
                check_plan(&p).map(|r| println!("계획 검사 통과: 위치 {}", r.positions()))
            } else {
                Err("--print-default 또는 --check <plan.toml> 중 하나가 필요함".into())
            }
        }
        Cmd::Interop(c) => interop::run(c),
    };
    if let Err(e) = r {
        eprintln!("오류: {e}");
        std::process::exit(1);
    }
}
