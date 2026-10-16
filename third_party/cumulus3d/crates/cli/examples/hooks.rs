/*
 * hooks.rs
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

//! Closure hook example: push a synthetic scene (3 drones x 8 positions) in position by position, print the events and the
//! preview/refined point counts, and use the default output hook to produce the same files as `cumulus3d stream` (timeline.txt, run.log,
//! snapshots/ ...).
//!
//! ```bash
//! cargo run --release -p cumulus3d-cli --example hooks -- [output folder]
//! ```
//! With a CUDA device it densifies as well; without one it runs with no densification (point cloud 0, sparse point count only).

#[path = "../tests/common/mod.rs"]
mod common;

use cumulus3d_cli::densewrap::{make_pm_backend, parse_profile, DenseConfig};
use cumulus3d_cli::events::{Event, EventKind};
use cumulus3d_cli::pipeline::Pipeline;
use cumulus3d_cli::session::{FrameSet, Input, Session, SessionConfig};
use cumulus3d_cli::sinks::{self, SinkOptions};
use cumulus3d_cli::stream::{Layout, CAMS};
use cumulus3d_core::io::read_gps_file;
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let out = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("cumulus3d_hooks"));
    let src = out.with_extension("src");
    if !src.join("gps_ref.txt").exists() {
        common::make_dataset(&src);
    }
    if out.exists() {
        std::fs::remove_dir_all(&out).map_err(|e| e.to_string())?;
    }
    let layout = Layout::discover(&src, 1)?;
    let npos = layout.frames.len();

    let mut cfg = SessionConfig::new(src.join("images"));
    cfg.gps = read_gps_file(src.join("gps_ref.txt")).map_err(|e| e.to_string())?;
    (cfg.span, cfg.overlap, cfg.total_positions, cfg.seed) = (4, 1, Some(npos), Some(7));
    if cumulus3d_cuda::is_available() {
        let mut d = DenseConfig::new(make_pm_backend("cuda"), parse_profile("fast")?);
        d.undistort.max_image_size = 160;
        cfg.dense = Some(d);
    }

    // Hook: print event text, zone point-cloud point counts, warnings and errors; plus the default output hook (files).
    let pipeline = Pipeline::new(Session::new(cfg))
        .on_any(|e: &Event| {
            if let Some(t) = e.timeline_text() {
                println!("[event v{}] {t}", e.meta().version);
            }
        })
        .on(EventKind::ZonePreview, |e: &Event| {
            if let Event::ZonePreview { range, cloud, model, .. } = e {
                println!("  초벌 {}: 조밀 점 {} / 희소 점 {}", range.zone, cloud.len(), model.num_points3d());
            }
        })
        .on(EventKind::ZoneRefined, |e: &Event| {
            if let Event::ZoneRefined { range, cloud, model, .. } = e {
                println!("  정밀 {}: 조밀 점 {} / 희소 점 {}", range.zone, cloud.len(), model.num_points3d());
            }
        })
        .on_message(|k, m| eprintln!("  {k:?}: {m}"));
    let mut pipeline = sinks::attach(pipeline, &out, &SinkOptions::default()).map_err(|e| e.to_string())?;

    for p in 0..npos {
        pipeline.push(Input::Frames(FrameSet::new(CAMS.iter().map(|c| (*c, layout.name(c, p))))));
    }
    let (_, summary) = pipeline.finish();
    println!("이벤트 {} 개, 훅 호출 {} 번, 훅 패닉 {}", summary.events, summary.hook_calls, summary.hook_panics);

    for f in ["timeline.txt", "run.log", "snapshots/manifest.json", "snapshots/timeline.txt", "DONE"] {
        let p = out.join(f);
        println!("{} {}", if p.exists() { "있음" } else { "없음" }, p.display());
    }
    Ok(())
}
