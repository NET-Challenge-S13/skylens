/*
 * frame_hook_plan.rs
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

//! Runs a TOML plan (same as `cumulus3d run <plan.toml>`) with a frame hook attached, and reports
//! what the hook saw: frame count per camera, image sizes, and the time from start to each frame.
//!
//! ```bash
//! cargo run --release -p cumulus3d-cli --example frame_hook_plan -- plan.toml
//! ```
//! Exits with an error if the hook saw no frames or any frame buffer has the wrong size.

use cumulus3d_cli::declare::{Plan, Recon};
use cumulus3d_cli::events::Frame;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Default)]
struct Seen {
    per_camera: BTreeMap<String, usize>,
    bad: usize,
    bytes: usize,
    sizes: BTreeMap<(u32, u32), usize>,
    first_s: Option<f64>,
    last_s: f64,
    positions: usize,
}

fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).ok_or("usage: frame_hook_plan <plan.toml>")?;
    let plan = Plan::load(&path)?;
    let seen = Arc::new(Mutex::new(Seen::default()));
    let refined = Arc::new(Mutex::new(Vec::new()));
    let t0 = Instant::now();
    let (s, r) = (seen.clone(), refined.clone());
    let recon = Recon::from_plan(plan)
        .on_frame(move |f: &Arc<Frame>| {
            let mut s = s.lock().unwrap();
            let t = t0.elapsed().as_secs_f64();
            s.first_s.get_or_insert(t);
            s.last_s = t;
            *s.per_camera.entry(f.camera.clone()).or_default() += 1;
            *s.sizes.entry((f.width, f.height)).or_default() += 1;
            s.bytes += f.rgb.len();
            s.positions = s.positions.max(f.position + 1);
            if f.rgb.len() != (f.width * f.height * 3) as usize || f.width == 0 {
                s.bad += 1;
            }
        })
        .on_zone_refined(move |z, cloud| r.lock().unwrap().push((z.zone, cloud.len())))
        .build()
        .map_err(|e| e.to_string())?;
    let summary = recon.run()?;
    let s = seen.lock().unwrap();
    let total: usize = s.per_camera.values().sum();
    println!(
        "frame hook: {total} frames over {} positions, per camera {:?}, sizes {:?}, {:.1} MB RGB, bad {}",
        s.positions,
        s.per_camera,
        s.sizes,
        s.bytes as f64 / 1e6,
        s.bad
    );
    println!(
        "frame hook: first frame at {:.2}s, last at {:.2}s; run total {:.1}s",
        s.first_s.unwrap_or(0.0),
        s.last_s,
        t0.elapsed().as_secs_f64()
    );
    println!("refined zones (zone, dense points): {:?}", refined.lock().unwrap());
    println!("events {}, hook calls {}, hook panics {}", summary.events, summary.hook_calls, summary.hook_panics);
    if total == 0 || s.bad > 0 || summary.hook_panics > 0 {
        return Err("frame hook check failed".into());
    }
    Ok(())
}
