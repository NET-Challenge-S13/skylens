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

//! cumulus3d 실행 파일의 라이브러리 부분: 점진 스트림 파이프라인과 단계별 호환 하위 명령([`interop`]).
//!
//! - [`session`]: 상태 값 [`session::Session`] 과 리듀서 함수 [`session::step`] / [`session::poll`] /
//!   [`session::command`] / [`session::finish`].
//! - [`pipeline`]: 리듀서를 감싸 훅·큐 정책·패닉 격리·구독 채널을 제공하는 [`pipeline::Pipeline`].
//! - [`events`]: 단계별 결과 [`events::Event`] 와 명령 [`events::Command`].
//! - [`sinks`]: `cumulus3d stream` 의 파일 출력(timeline.txt, run.log, PLY, 스냅샷)을 만드는 기본 훅.
//! - [`stream`]: 입력 폴더 → 세션 → 파이프라인 조립([`stream::run_stream`]).
//! - [`compose`]: 실시간 입력용 유지 노드. 지금의 입력 상태([`compose::FrameState`])를 선언하면 받아들인 기록과 비교해 증분 처리한다.
//! - [`declare`]: 선언형 빌더 층. 계획([`declare::Plan`])을 기록만 하고, `build` 에서 검사, `run` 에서 한 번에 실행
//!   ([`declare::Recon`]). 계획은 TOML 로 저장·읽기(`cumulus3d run plan.toml`).
//!
//! # 사용 예
//!
//! 영상 폴더(`images/camF|camR|camL/*.jpg`, `gps_ref.txt`)를 위치 단위로 넣어 점진 재구성하고,
//! 기본 출력 훅으로 `cumulus3d stream` 과 같은 파일을 만든다.
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
//!             println!("초벌 구역 {}: 점 {}", range.zone, cloud.len());
//!         }
//!     });
//!     let mut pipeline = sinks::attach(pipeline, Path::new("out"), &SinkOptions::default()).map_err(|e| e.to_string())?;
//!     for p in 0..layout.frames.len() {
//!         pipeline.push(Input::Frames(frame_set(&layout, p)));
//!     }
//!     let (_session, summary) = pipeline.finish();
//!     println!("이벤트 {} 개", summary.events);
//!     Ok(())
//! }
//! ```
//!
//! 선언형(계획 기록 → 검사 → 실행):
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
//! println!("이벤트 {} 개", summary.events);
//! ```
//!
//! 훅 없이 상태 값만 주고받는 함수형 사용:
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

pub mod compose;
pub mod declare;
pub mod densewrap;
pub mod events;
pub mod fusion_variants;
pub mod interop;
pub mod pipeline;
pub mod post;
pub mod session;
pub mod sinks;
pub mod stream;
pub mod util;
