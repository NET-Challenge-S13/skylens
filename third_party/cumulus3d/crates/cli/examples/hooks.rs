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

//! 람다 훅 예: 합성 장면(드론 3대 × 8위치)을 위치 단위로 밀어 넣고, 이벤트·초벌/정밀 점 수를 출력하며
//! 기본 출력 훅으로 `cumulus3d stream` 과 같은 파일(timeline.txt, run.log, snapshots/ …)을 만든다.
//!
//! ```bash
//! cargo run --release -p cumulus3d-cli --example hooks -- [출력 폴더]
//! ```
//! CUDA 장치가 있으면 조밀화까지 하고, 없으면 조밀화 없이(점군 0, 희소 점 수만) 돈다.

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

    // 훅: 사건 문구 출력, 구역 점군 점 수, 경고·오류. 그리고 기본 출력 훅(파일).
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
