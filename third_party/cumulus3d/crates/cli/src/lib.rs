/*
 * lib.rs
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

//! Library part of the cumulus3d executable: the incremental stream pipeline and the stage-by-stage compatible subcommands ([`interop`]).
//!
//! - [`session`]: the state value [`session::Session`] and the reducer functions [`session::step`] / [`session::poll`] /
//!   [`session::command`] / [`session::finish`].
//! - [`pipeline`]: [`pipeline::Pipeline`], which wraps the reducer and provides hooks, queue policy, panic isolation and subscription
//!   channels.
//! - [`events`]: per-stage results [`events::Event`] and commands [`events::Command`].
//! - [`sinks`]: default hooks that produce the file output of `cumulus3d stream` (timeline.txt, run.log, PLY, snapshots).
//! - [`stream`]: input folder -> session -> pipeline assembly ([`stream::run_stream`]).
//! - [`compose`]: a persistent node for live input. Declare the current input state ([`compose::FrameState`]) and it is compared with the
//!   accepted record to process incrementally.
//! - [`declare`]: the declarative builder layer. It only records a plan ([`declare::Plan`]); `build` validates it and `run` executes it
//!   in one go
//!   ([`declare::Recon`]). Plans are saved and read as TOML (`cumulus3d run plan.toml`).
//!
//! # Usage example
//!
//! Feed an image folder (`images/camF|camR|camL/*.jpg`, `gps_ref.txt`) position by position for incremental reconstruction,
//! and use the default output hook to produce the same files as `cumulus3d stream`.
//!
//! ```no_run
//! use cumulus3d_cli::events::{Event, EventKind};
//! use cumulus3d_cli::pipeline::Pipeline;
//! use cumulus3d_cli::session::{Input, Session, SessionConfig};
//! use cumulus3d_cli::sinks::{self, SinkOptions};
//! use cumulus3d_cli::stream::{frame_set, Layout};
//! use cumulus3d_core::io::read_gps_file;
//! use std::path::Path;
//!
//! fn main() -> Result<(), String> {
//!     let src = Path::new("data");
//!     let layout = Layout::discover(src, 3)?;
//!     let mut cfg = SessionConfig::new(src.join("images"));
//!     cfg.gps = read_gps_file(src.join("gps_ref.txt")).map_err(|e| e.to_string())?;
//!     cfg.total_positions = Some(layout.frames.len());
//!
//!     let pipeline = Pipeline::new(Session::new(cfg)).on(EventKind::ZonePreview, |e: &Event| {
//!         if let Event::ZonePreview { range, cloud, .. } = e {
//!             println!("preview zone {}: {} points", range.zone, cloud.len());
//!         }
//!     });
//!     let mut pipeline = sinks::attach(pipeline, Path::new("out"), &SinkOptions::default()).map_err(|e| e.to_string())?;
//!     for p in 0..layout.frames.len() {
//!         pipeline.push(Input::Frames(frame_set(&layout, p)));
//!     }
//!     let (_session, summary) = pipeline.finish();
//!     println!("{} events", summary.events);
//!     Ok(())
//! }
//! ```
//!
//! Declarative (record a plan -> validate -> run):
//!
//! ```no_run
//! use cumulus3d_cli::declare::{Dense, Recon, Sinks};
//!
//! let recon = Recon::declare()
//!     .input("data")
//!     .preset("aerial-formation")
//!     .zones(12, 2)
//!     .dense(Dense::profile("fast").fusion_min_views(3))
//!     .sinks(Sinks::default_files("out"))
//!     .seed(7)
//!     .build()
//!     .unwrap_or_else(|e| panic!("{e}"));
//! let summary = recon.run().unwrap();
//! println!("{} events", summary.events);
//! ```
//!
//! Functional use that only passes state values around, without hooks:
//!
//! ```no_run
//! use cumulus3d_cli::session::{finish, step, FrameSet, Session, SessionConfig};
//!
//! let s0 = Session::new(SessionConfig::new("data/images"));
//! let frames = FrameSet::new([("camF", "camF/camF_0000.jpg"), ("camR", "camR/camR_0000.jpg")]);
//! let (s1, events) = step(&s0, frames);
//! for e in &events {
//!     if let Some(t) = e.timeline_text() {
//!         println!("{t}");
//!     }
//! }
//! let (_done, _summary_events) = finish(&s1);
//! ```

#![warn(missing_docs)]

pub mod blocks;
pub mod compose;
pub mod declare;
pub mod densewrap;
pub mod events;
pub mod fusion_variants;
pub mod interop;
pub mod intrinsics;
pub mod pipeline;
pub mod post;
pub mod session;
pub mod sinks;
pub mod stream;
pub mod util;
