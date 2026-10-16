/*
 * sinks.rs
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

//! Default output hooks: receive events and produce the file output of `cumulus3d stream`.
//!
//! | Hook | Events received | Output |
//! |---|---|---|
//! | [`TimelineSink`] | events that have text (table below) | `timeline.txt` (`epoch seconds.nanoseconds text`) |
//! | [`RunLogSink`] | events that have text, `Log`, `Finished` | `run.log` (+ standard output), `DONE` |
//! | [`ZonePlySink`] | `ZonePreview`, `ZoneRefined` (`dense` true) | `full/preview/preview_KK_posLO-HI.ply`, `full/refined/refined_KK_posLO-HI.ply` |
//! | [`ModelSink`] | `ZoneArrived`, `ZonePreview`, `ZoneAdjusted`, `ZoneRefinedPose`, `AllRefinedDone` | `work/models/{snap,preview,ba,refined}_K/`, `work/models/chain/` |
//! | [`SnapshotSink`] | zone events, `AllRefinedDone` | `aligned/`, `snapshots/event_NN_TTTT.Ts_KIND_K.ply`, `snapshots/manifest.json`, `snapshots/timeline.txt`, `final_frame/`, `live/`, run.log summary |
//!
//! Event -> timeline text conversion table ([`Event::timeline_text`], [`TIMELINE_TABLE`]):
//!
//! | Event | Text |
//! |---|---|
//! | `Started { positions: Some(N) }` | `start NPOS=N` |
//! | `ModelInitialized { registered: N }` | `init_model Registered images: N` |
//! | `BaseAdopted { zone: K }` | `adopt refined K as base` |
//! | `PositionDone { position: P, registered: R, expected: E }` | `pos P registered R/E` |
//! | `ZoneArrived { range: K [LO,HI) }` | `region K arrived pos[LO,HI)` |
//! | `ZonePreview { range: K }` | `preview K ready` |
//! | `RefineStarted { zone: K }` | `refined K ba_start` |
//! | `ZoneRefinedPose { range: K, registered: N, mean_reproj_px: X }` | `refined K pose_ready Registered images: N Mean reprojection error: X(6 decimal places)px ` (including the trailing space) |
//! | `ZoneRefined { range: K }` | `refined K ready` |
//! | `AllPositionsDone` | `all positions done` |
//! | `AllRefinedDone` | `all refined done` |
//!
//! Ordering: the hooks of one output folder must be bundled in a single [`DefaultSinks`] and handled in event order.
//! In an asynchronous pipeline the order between per-kind lanes is not guaranteed, so use [`attach`] (a single `on_any`).
//! [`default_sinks`] (a per-kind list) is for synchronous mode (`Pipeline::sync(true)`) or direct calls.

use crate::blocks::{DiffStream, Heartbeat, LiveStream};
use crate::events::{Event, EventKind, ZoneRange};
use crate::pipeline::{Pipeline, Reducer};
use crate::post::{self, PostZones, ZoneCloud, ZoneTimes};
use crate::util::Logger;
use cumulus3d_core::interop::{write_model_binary, ImageOrder};
use cumulus3d_core::io::{write_ply, PlyLayout, PointCloud};
use cumulus3d_core::Reconstruction;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// One hook (the shape passed to the pipeline's `.on`/`.on_any`).
pub type Hook = Box<dyn FnMut(&Event) + Send>;

/// Event kinds that produce timeline text (conversion rules in [`Event::timeline_text`]).
pub const TIMELINE_TABLE: [(EventKind, &str); 11] = [
    (EventKind::Started, "start NPOS={positions}"),
    (EventKind::ModelInitialized, "init_model Registered images: {registered}"),
    (EventKind::BaseAdopted, "adopt refined {zone} as base"),
    (EventKind::PositionDone, "pos {position} registered {registered}/{expected}"),
    (EventKind::ZoneArrived, "region {zone} arrived pos[{lo},{hi})"),
    (EventKind::ZonePreview, "preview {zone} ready"),
    (EventKind::RefineStarted, "refined {zone} ba_start"),
    (
        EventKind::ZoneRefinedPose,
        "refined {zone} pose_ready Registered images: {registered} Mean reprojection error: {mean_reproj_px:.6}px ",
    ),
    (EventKind::ZoneRefined, "refined {zone} ready"),
    (EventKind::AllPositionsDone, "all positions done"),
    (EventKind::AllRefinedDone, "all refined done"),
];

/// A hook that receives events and produces output.
pub trait Sink: Send {
    /// Event kinds it handles.
    fn kinds(&self) -> Vec<EventKind>;
    /// Handle one event. A failure is recorded in run.log and does not panic.
    fn handle(&mut self, e: &Event);
}

/// One hook as a per-kind hook list (sharing the same state). Use it only in synchronous mode if ordering between kinds is needed.
pub fn hooks<S: Sink + 'static>(sink: S) -> Vec<(EventKind, Hook)> {
    let kinds = sink.kinds();
    let shared = Arc::new(Mutex::new(sink));
    kinds
        .into_iter()
        .map(|k| {
            let s = Arc::clone(&shared);
            let h: Hook = Box::new(move |e: &Event| s.lock().unwrap_or_else(|p| p.into_inner()).handle(e));
            (k, h)
        })
        .collect()
}

/// One hook as an all-event hook (`on_any`). It filters and handles only the kinds it accepts.
pub fn any_hook<S: Sink + 'static>(mut sink: S) -> Hook {
    let kinds = sink.kinds();
    Box::new(move |e: &Event| {
        if kinds.contains(&e.kind()) {
            sink.handle(e)
        }
    })
}

fn epoch_secs(t: SystemTime) -> f64 {
    t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64()
}

fn zone_file(kind: &str, r: &ZoneRange) -> String {
    format!("{kind}_{:02}_pos{}-{}.ply", r.zone, r.lo, r.hi)
}

// ---------------------------------------------------------------------------------------------

/// Shared handle to run.log. Several hooks (such as [`SnapshotSink`], which writes a summary) write to the same file.
#[derive(Clone)]
pub struct RunLog {
    inner: Arc<RunLogInner>,
}

struct RunLogInner {
    logger: Logger,
    /// Stage times measured in hooks (e.g. `post`). They are added to the per-stage times of `Finished`.
    extra_times: Mutex<Vec<(String, Duration)>>,
}

impl RunLog {
    /// Create `path` anew (if `None`, no file and only standard output, when `echo`).
    pub fn open(path: Option<&Path>, echo: bool) -> std::io::Result<Self> {
        Ok(Self { inner: Arc::new(RunLogInner { logger: Logger::new(path, echo)?, extra_times: Mutex::new(Vec::new()) }) })
    }
    /// Internal recorder.
    pub fn logger(&self) -> &Logger {
        &self.inner.logger
    }
    /// Record one line.
    pub fn line(&self, s: &str) {
        self.inner.logger.line(s)
    }
    /// Record with the `[HH:MM:SS] ` prefix of the given time.
    pub fn stamped_at(&self, at: SystemTime, s: &str) {
        self.inner.logger.stamped_at(at, s)
    }
    /// Add stage time (same names accumulate).
    pub fn add_time(&self, stage: &str, d: Duration) {
        if let Ok(mut v) = self.inner.extra_times.lock() {
            v.push((stage.to_string(), d));
        }
    }
    fn take_times(&self) -> Vec<(String, Duration)> {
        self.inner.extra_times.lock().map(|mut v| std::mem::take(&mut *v)).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------------------------

/// `timeline.txt`: for every event that has text, one line `epoch seconds.nanoseconds text` (time = `meta.at`).
pub struct TimelineSink {
    file: File,
}

impl TimelineSink {
    /// Create timeline.txt anew at `path`.
    pub fn create(path: &Path) -> std::io::Result<Self> {
        Ok(Self { file: File::create(path)? })
    }

    /// One event line (None if there is no text).
    pub fn line(e: &Event) -> Option<String> {
        let s = e.timeline_text()?;
        let t = e.meta().at.duration_since(UNIX_EPOCH).unwrap_or_default();
        Some(format!("{}.{:09} {s}", t.as_secs(), t.subsec_nanos()))
    }
}

impl Sink for TimelineSink {
    fn kinds(&self) -> Vec<EventKind> {
        TIMELINE_TABLE.iter().map(|(k, _)| *k).collect()
    }
    fn handle(&mut self, e: &Event) {
        if let Some(l) = Self::line(e) {
            let _ = writeln!(self.file, "{l}");
            let _ = self.file.flush();
        }
    }
}

// ---------------------------------------------------------------------------------------------

/// `run.log`: timeline text is `[HH:MM:SS] text`, `Log` as is (with the time prefix if `stamped`),
/// `Finished` is followed by the `DONE` file after the per-stage times (including the hook-measured `post`) and the final chain model
/// summary.
/// `Warning`/`Error` come together with a `Log` of the same content, so they are not written separately.
pub struct RunLogSink {
    log: RunLog,
    done: Option<PathBuf>,
}

impl RunLogSink {
    /// `done` = the empty file to create on `Finished` (usually `<out>/DONE`).
    pub fn new(log: RunLog, done: Option<PathBuf>) -> Self {
        Self { log, done }
    }
}

impl Sink for RunLogSink {
    fn kinds(&self) -> Vec<EventKind> {
        let mut v: Vec<EventKind> = TIMELINE_TABLE.iter().map(|(k, _)| *k).collect();
        v.extend([EventKind::Log, EventKind::Finished]);
        v
    }
    fn handle(&mut self, e: &Event) {
        if let Some(s) = e.timeline_text() {
            self.log.stamped_at(e.meta().at, &s);
            return;
        }
        match e {
            Event::Log { meta, line, stamped } => {
                if *stamped {
                    self.log.stamped_at(meta.at, line)
                } else {
                    self.log.line(line)
                }
            }
            Event::Finished { pipeline_secs, stage_times, chain_stats, .. } => {
                let mut m: BTreeMap<String, (Duration, usize)> = BTreeMap::new();
                for (s, d, n) in stage_times {
                    let x = m.entry(s.clone()).or_default();
                    x.0 += *d;
                    x.1 += n;
                }
                for (s, d) in self.log.take_times() {
                    let x = m.entry(s).or_default();
                    x.0 += d;
                    x.1 += 1;
                }
                self.log.line(&format!("== 단계별 시간 (파이프라인 {pipeline_secs:.1}s, 배경 스레드와 겹친 누적)"));
                for (stage, (d, n)) in m {
                    self.log.line(&format!("{stage:<14} {:>9.2}s  ({n}회)", d.as_secs_f64()));
                }
                if !chain_stats.is_empty() {
                    self.log.line("== 최종 체인 모델");
                    for l in chain_stats {
                        self.log.line(l);
                    }
                }
                if let Some(p) = &self.done {
                    if let Err(err) = std::fs::write(p, b"") {
                        self.log.line(&format!("{}: {err}", p.display()));
                    }
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------------------------

/// `full/preview/preview_KK_posLO-HI.ply`, `full/refined/refined_KK_posLO-HI.ply`
/// (x y z nx ny nz red green blue). Events whose `dense` is false (no densification or failed) are not written.
pub struct ZonePlySink {
    full: PathBuf,
    log: Option<RunLog>,
}

impl ZonePlySink {
    /// `full` = `full/` of the output folder.
    pub fn new(full: PathBuf, log: Option<RunLog>) -> Self {
        Self { full, log }
    }

    fn write(&self, kind: &str, r: &ZoneRange, c: &PointCloud) {
        let path = self.full.join(kind).join(zone_file(kind, r));
        let res = path
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .map_err(|e| e.to_string())
            .and_then(|_| write_ply(&path, c, PlyLayout::XyzNormalRgb).map_err(|e| e.to_string()));
        if let (Err(e), Some(l)) = (res, &self.log) {
            l.line(&format!("[dense] {kind} {} PLY 쓰기 실패: {e}", r.zone));
        }
    }
}

impl Sink for ZonePlySink {
    fn kinds(&self) -> Vec<EventKind> {
        vec![EventKind::ZonePreview, EventKind::ZoneRefined]
    }
    fn handle(&mut self, e: &Event) {
        match e {
            Event::ZonePreview { range, cloud, dense: true, .. } => self.write("preview", range, cloud),
            Event::ZoneRefined { range, cloud, dense: true, .. } => self.write("refined", range, cloud),
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------------------------

/// `--save-models`: per-zone sparse models under `work/models/`.
/// `ZoneArrived` → `snap_K`, `ZonePreview` → `preview_K`, `ZoneAdjusted` → `ba_K`, `ZoneRefinedPose` → `refined_K`,
/// `AllRefinedDone` → `chain`.
pub struct ModelSink {
    dir: PathBuf,
    log: Option<RunLog>,
}

impl ModelSink {
    /// `dir` = usually `<out>/work/models`.
    pub fn new(dir: PathBuf, log: Option<RunLog>) -> Self {
        Self { dir, log }
    }

    fn save(&self, rec: &Reconstruction, name: &str) {
        let d = self.dir.join(name);
        let _ = std::fs::create_dir_all(&d);
        if let (Err(e), Some(l)) = (write_model_binary(rec, &d, ImageOrder::Registration), &self.log) {
            l.line(&format!("모델 저장 실패 {}: {e}", d.display()));
        }
    }
}

impl Sink for ModelSink {
    fn kinds(&self) -> Vec<EventKind> {
        vec![EventKind::ZoneArrived, EventKind::ZonePreview, EventKind::ZoneAdjusted, EventKind::ZoneRefinedPose, EventKind::AllRefinedDone]
    }
    fn handle(&mut self, e: &Event) {
        match e {
            Event::ZoneArrived { range, model, .. } => self.save(model, &format!("snap_{}", range.zone)),
            Event::ZonePreview { range, model, .. } => self.save(model, &format!("preview_{}", range.zone)),
            Event::ZoneAdjusted { zone, model, .. } => self.save(model, &format!("ba_{zone}")),
            Event::ZoneRefinedPose { range, model, .. } => self.save(model, &format!("refined_{}", range.zone)),
            Event::AllRefinedDone { chain: Some(c), .. } => self.save(c, "chain"),
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------------------------

/// Hook for alignment, masking, per-event snapshots, manifest and re-anchoring (final_frame). It collects zone results and
/// builds them all at once with [`post::run_zones`] at `AllRefinedDone`. Times are seconds relative to `Started` (or the first received
/// event if absent).
///
/// - Alignment range of preview K (>= 1) = [lo of zone K, hi of zone K-1) (the overlap with the previous zone).
/// - Image name -> position is collected from `FrameIngested` and `FeaturesExtracted`.
/// - `Reset { position }` drops zones with hi > position, and `ZoneInvalidated` drops that zone.
pub struct SnapshotSink {
    out: PathBuf,
    log: RunLog,
    t0: Option<SystemTime>,
    ranges: BTreeMap<usize, ZoneRange>,
    times: BTreeMap<usize, Stamps>,
    preview: BTreeMap<usize, Arc<PointCloud>>,
    refined: BTreeMap<usize, Arc<PointCloud>>,
    preview_models: BTreeMap<usize, Arc<Reconstruction>>,
    refined_models: BTreeMap<usize, Arc<Reconstruction>>,
    positions: HashMap<String, usize>,
    live: post::LiveDisplay,
    live_count: usize,
    /// Live diff stream of the live display (`live/stream/`; stable display only) and the display it last wrote.
    diff: Option<DiffStream>,
    last_display: Option<(PointCloud, Vec<usize>)>,
    /// Progress heartbeat (`live/stream/status.json`), on with the block stream.
    beat: Option<Heartbeat>,
}

/// Zone event time (the first one received).
#[derive(Clone, Copy, Default)]
struct Stamps {
    arrived: Option<SystemTime>,
    preview: Option<SystemTime>,
    pose: Option<SystemTime>,
    refined: Option<SystemTime>,
}

impl SnapshotSink {
    /// Create from the output folder and run.log.
    pub fn new(out: PathBuf, log: RunLog) -> Self {
        Self {
            out,
            log,
            t0: None,
            ranges: BTreeMap::new(),
            times: BTreeMap::new(),
            preview: BTreeMap::new(),
            refined: BTreeMap::new(),
            preview_models: BTreeMap::new(),
            refined_models: BTreeMap::new(),
            positions: HashMap::new(),
            live: post::LiveDisplay::default(),
            live_count: 0,
            diff: None,
            last_display: None,
            beat: None,
        }
    }

    /// Updates the heartbeat state through `f` and rewrites `status.json`; a failure is only logged.
    fn beat(&mut self, at: SystemTime, f: impl FnOnce(&mut Heartbeat)) {
        let Some(b) = self.beat.as_mut() else { return };
        f(b);
        let t = epoch_secs(at) - epoch_secs(self.t0.unwrap_or(at));
        if let Err(e) = b.write((t * 10.0).round() / 10.0) {
            self.log.line(&format!("[live] status.json 실패: {e}"));
        }
    }

    /// Sets the live display thinning stride (1 = keep all).
    pub fn with_live_stride(mut self, stride: usize) -> Self {
        self.live = post::LiveDisplay::new(post::LiveOptions { stride, ..post::LiveOptions::default() });
        self
    }

    /// Turns the live diff stream (`live/stream/`) on.
    pub fn with_live_stream(mut self, cfg: LiveStream) -> Self {
        let dir = self.out.join("live").join("stream");
        self.beat = Some(Heartbeat { dir: dir.clone(), stage: "registration", ..Default::default() });
        self.diff = Some(DiffStream::new(dir, cfg));
        self
    }

    fn drop_zone(&mut self, k: usize) {
        self.ranges.remove(&k);
        self.times.remove(&k);
        self.preview.remove(&k);
        self.refined.remove(&k);
        self.preview_models.remove(&k);
        self.refined_models.remove(&k);
        self.live.forget(k);
    }

    /// Writes the display cloud for the refined zones received so far (`live/display_NN_TTTT.Ts_refined_K.ply`, and
    /// `live/latest.ply` replaced atomically), every zone placed in the newest refined model's frame.
    fn write_live(&mut self, k: usize, at: SystemTime, kind: &str) {
        let t = Instant::now();
        let clouds: BTreeMap<usize, &PointCloud> = self.refined.iter().map(|(j, c)| (*j, &**c)).collect();
        let models: BTreeMap<usize, &Reconstruction> = self.refined_models.iter().map(|(j, m)| (*j, &**m)).collect();
        // Stable display (no preview ever received; zone 0's preview always precedes zone 1's refined result): all zone
        // clouds already share one frame, so they are not relocated.
        self.live.set_relocate(!self.preview_models.is_empty());
        let (cloud, zones) = self.live.compose_zoned(&clouds, &models);
        let dir = self.out.join("live");
        let rel = epoch_secs(at) - epoch_secs(self.t0.unwrap_or(at));
        self.live_count += 1;
        let name = format!("display_{:02}_{rel:06.1}s_refined_{k}.ply", self.live_count);
        let tmp = dir.join(".latest.ply.tmp");
        let r = std::fs::create_dir_all(&dir)
            .map_err(|e| e.to_string())
            .and_then(|_| write_ply(dir.join(&name), &cloud, PlyLayout::XyzRgbNormal).map_err(|e| e.to_string()))
            .and_then(|_| write_ply(&tmp, &cloud, PlyLayout::XyzRgbNormal).map_err(|e| e.to_string()))
            .and_then(|_| std::fs::rename(&tmp, dir.join("latest.ply")).map_err(|e| e.to_string()));
        self.log.add_time("live", t.elapsed());
        match r {
            Ok(()) => {
                self.log.line(&format!("[live] {name}: 구역 {} 개, 점 {}, {:.2}s", clouds.len(), cloud.len(), t.elapsed().as_secs_f64()))
            }
            Err(e) => self.log.line(&format!("[live] {name} 실패: {e}")),
        }
        if self.diff.is_some() {
            self.write_stream(k, rel, kind, cloud, zones);
        }
    }

    /// One event of the live diff stream (`kind`: refined | replace | final); logs the time and the change counts.
    fn write_stream(&mut self, zone: usize, t_rel: f64, kind: &str, cloud: PointCloud, zones: Vec<usize>) {
        let t = Instant::now();
        let Some(bs) = self.diff.as_mut() else { return };
        let r = bs.write_event(t_rel, zone, kind, &cloud, &zones);
        self.log.add_time("live_stream", t.elapsed());
        match r {
            Ok(st) => self.log.line(&format!(
                "[live] stream seq {}: 추가 {} 이동 {} 제거 {}, 점 {}, {:.2} MB, {:.2}s",
                bs.events_written(),
                st.n_add,
                st.n_move,
                st.n_remove,
                st.points,
                st.bytes as f64 / 1e6,
                t.elapsed().as_secs_f64()
            )),
            Err(e) => self.log.line(&format!("[live] stream {kind} 구역 {zone} 실패: {e}")),
        }
        self.last_display = Some((cloud, zones));
    }

    /// Post-process with the results collected so far. When finished, copy `timeline.txt` to `snapshots/`.
    /// `final_model` = `AllRefinedDone::chain` (in stable display mode the GPS-aligned final model).
    pub fn finish_with(&mut self, final_model: Option<&Reconstruction>) {
        if self.preview_models.is_empty() && !self.refined.is_empty() {
            let t0 = epoch_secs(self.t0.unwrap_or(UNIX_EPOCH));
            let rel = |t: Option<SystemTime>| t.map(|t| epoch_secs(t) - t0);
            let times: BTreeMap<usize, ZoneTimes> = self
                .times
                .iter()
                .map(|(k, s)| {
                    (
                        *k,
                        ZoneTimes {
                            arrived: rel(s.arrived),
                            preview_ready: None,
                            refined_pose: rel(s.pose),
                            refined_ready: rel(s.refined),
                        },
                    )
                })
                .collect();
            let files: BTreeMap<usize, String> =
                self.refined.keys().filter_map(|k| self.ranges.get(k).map(|r| (*k, zone_file("refined", r)))).collect();
            let rf = self.refined.iter().filter_map(|(k, c)| files.get(k).map(|f| (*k, ZoneCloud { file: f.clone(), cloud: c }))).collect();
            let display = self.refined_models.values().next_back().map(|m| &**m);
            let t = Instant::now();
            let models: BTreeMap<usize, &Reconstruction> = self.refined_models.iter().map(|(k, m)| (*k, &**m)).collect();
            let r = post::run_stable(&self.out, &rf, &models, &times, display, final_model, self.log.logger());
            self.log.add_time("post", t.elapsed());
            if let Err(e) = r {
                self.log.line(&format!("[post] 실패: {e}"));
            }
            let _ = std::fs::copy(self.out.join("timeline.txt"), self.out.join("snapshots").join("timeline.txt"));
            return;
        }
        self.finish();
    }

    /// Post-process with the results collected so far (preview/refined alignment, per-event snapshots, re-anchoring). When finished, copy
    /// `timeline.txt` to `snapshots/`.
    pub fn finish(&mut self) {
        let t0 = epoch_secs(self.t0.unwrap_or(UNIX_EPOCH));
        let rel = |t: Option<SystemTime>| t.map(|t| epoch_secs(t) - t0);
        let times: BTreeMap<usize, ZoneTimes> = self
            .times
            .iter()
            .map(|(k, s)| {
                (
                    *k,
                    ZoneTimes {
                        arrived: rel(s.arrived),
                        preview_ready: rel(s.preview),
                        refined_pose: rel(s.pose),
                        refined_ready: rel(s.refined),
                    },
                )
            })
            .collect();
        let windows = self
            .ranges
            .iter()
            .map(|(k, r)| {
                let hi = k.checked_sub(1).and_then(|j| self.ranges.get(&j)).map_or(r.hi, |p| p.hi);
                (*k, (r.lo, hi))
            })
            .collect();
        let zc = |kind: &str, m: &BTreeMap<usize, Arc<PointCloud>>| -> BTreeMap<usize, (String, Arc<PointCloud>)> {
            m.iter().filter_map(|(k, c)| self.ranges.get(k).map(|r| (*k, (zone_file(kind, r), Arc::clone(c))))).collect()
        };
        let (pv, rf) = (zc("preview", &self.preview), zc("refined", &self.refined));
        let positions = &self.positions;
        let pos = move |n: &str| positions.get(n).copied();
        let z = PostZones {
            out: &self.out,
            preview: pv.iter().map(|(k, (f, c))| (*k, ZoneCloud { file: f.clone(), cloud: c })).collect(),
            refined: rf.iter().map(|(k, (f, c))| (*k, ZoneCloud { file: f.clone(), cloud: c })).collect(),
            preview_models: self.preview_models.iter().map(|(k, m)| (*k, &**m)).collect(),
            refined_models: self.refined_models.iter().map(|(k, m)| (*k, &**m)).collect(),
            times,
            windows,
            position: &pos,
        };
        let t = Instant::now();
        let r = post::run_zones(&z, self.log.logger());
        self.log.add_time("post", t.elapsed());
        if let Err(e) = r {
            self.log.line(&format!("[post] 실패: {e}"));
        }
        let _ = std::fs::copy(self.out.join("timeline.txt"), self.out.join("snapshots").join("timeline.txt"));
    }
}

impl Sink for SnapshotSink {
    fn kinds(&self) -> Vec<EventKind> {
        vec![
            EventKind::Started,
            EventKind::FrameIngested,
            EventKind::FeaturesExtracted,
            EventKind::ZoneArrived,
            EventKind::ZonePreview,
            EventKind::PositionDone,
            EventKind::RefineStarted,
            EventKind::ZoneRefinedPose,
            EventKind::ZoneRefined,
            EventKind::Reset,
            EventKind::ZoneInvalidated,
            EventKind::AllRefinedDone,
        ]
    }

    fn handle(&mut self, e: &Event) {
        let at = e.meta().at;
        if matches!(e, Event::Started { .. }) || self.t0.is_none() {
            self.t0 = Some(at);
        }
        match e {
            Event::Started { positions, .. } => self.beat(at, |b| b.positions_total = *positions),
            Event::PositionDone { position, registered, .. } => {
                let (p, r) = (*position + 1, *registered);
                self.beat(at, |b| {
                    b.stage = "triangulation";
                    b.positions = p;
                    b.images = r;
                });
            }
            Event::RefineStarted { zone, .. } => self.beat(at, |b| {
                b.stage = "ba";
                b.zone = *zone;
            }),
            Event::FrameIngested { position, images, .. } => {
                for n in images {
                    self.positions.insert(n.clone(), *position);
                }
            }
            Event::FeaturesExtracted { image, position, .. } => {
                self.positions.insert(image.clone(), *position);
            }
            Event::ZoneArrived { range, .. } => {
                self.ranges.insert(range.zone, *range);
                self.times.entry(range.zone).or_default().arrived.get_or_insert(at);
            }
            Event::ZonePreview { range, cloud, model, dense, .. } => {
                self.ranges.entry(range.zone).or_insert(*range);
                self.times.entry(range.zone).or_default().preview.get_or_insert(at);
                self.preview_models.insert(range.zone, Arc::clone(model));
                if *dense {
                    self.preview.insert(range.zone, Arc::clone(cloud));
                }
            }
            Event::ZoneRefinedPose { range, model, .. } => {
                self.beat(at, |b| {
                    b.stage = "dense";
                    b.zone = range.zone;
                });
                self.ranges.entry(range.zone).or_insert(*range);
                self.times.entry(range.zone).or_default().pose.get_or_insert(at);
                self.refined_models.insert(range.zone, Arc::clone(model));
                self.live.forget(range.zone);
            }
            Event::ZoneRefined { range, cloud, dense, .. } => {
                self.ranges.entry(range.zone).or_insert(*range);
                self.times.entry(range.zone).or_default().refined.get_or_insert(at);
                if *dense {
                    // A zone that already has a cloud is a shown zone being re-densified or moved (`replace_shown`).
                    let kind = if self.refined.contains_key(&range.zone) { "replace" } else { "refined" };
                    self.refined.insert(range.zone, Arc::clone(cloud));
                    self.write_live(range.zone, at, kind);
                }
            }
            Event::Reset { position, .. } => {
                let ks: Vec<usize> = self.ranges.values().filter(|r| r.hi > *position).map(|r| r.zone).collect();
                for k in ks {
                    self.drop_zone(k);
                }
            }
            Event::ZoneInvalidated { zone, .. } => self.drop_zone(*zone),
            Event::AllRefinedDone { chain, .. } => {
                self.beat(at, |b| b.stage = "done");
                // Final stream event: the display does not change (the ENU alignment of the final outputs is separate).
                if let Some((cloud, zones)) = self.last_display.take() {
                    let rel = epoch_secs(at) - epoch_secs(self.t0.unwrap_or(at));
                    let zone = self.refined.keys().next_back().copied().unwrap_or(0);
                    self.write_stream(zone, rel, "final", cloud, zones);
                }
                self.finish_with(chain.as_deref())
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------------------------

/// Default hook settings.
#[derive(Clone, Debug, Default)]
pub struct SinkOptions {
    /// Also write the run.log contents to standard output.
    pub echo: bool,
    /// Save per-zone sparse models (`work/models/`).
    pub save_models: bool,
}

/// Which default hooks to enable ([`DefaultSinks::with_set`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SinkSet {
    /// `timeline.txt`([`TimelineSink`]).
    pub timeline: bool,
    /// `run.log`·`DONE`([`RunLogSink`]).
    pub run_log: bool,
    /// Zone PLY `full/{preview,refined}/` ([`ZonePlySink`]).
    pub zone_ply: bool,
    /// Per-zone sparse models `work/models/` ([`ModelSink`]).
    pub models: bool,
    /// Alignment, snapshots, manifest, final_frame ([`SnapshotSink`]).
    pub snapshots: bool,
    /// Live diff stream `live/stream/` of the live display (needs `snapshots`; stable display only).
    pub live_stream: Option<LiveStream>,
    /// Live display thinning stride (1 = keep all).
    pub live_stride: usize,
}

impl SinkSet {
    /// Enable all.
    pub fn all() -> Self {
        Self {
            timeline: true,
            run_log: true,
            zone_ply: true,
            models: true,
            snapshots: true,
            live_stream: None,
            live_stride: post::LiveOptions::default().stride,
        }
    }
}

/// The default hook bundle of one output folder (handled in the order timeline -> run.log -> full PLY -> models -> snapshots).
pub struct DefaultSinks {
    sinks: Vec<Box<dyn Sink>>,
    log: RunLog,
}

impl DefaultSinks {
    /// Create `full/preview`, `full/refined` and `work` under `out`, and open `timeline.txt` and `run.log` anew.
    /// It does not delete the folder.
    pub fn new(out: &Path, opts: &SinkOptions) -> std::io::Result<Self> {
        Self::with_set(out, opts.echo, SinkSet { models: opts.save_models, ..SinkSet::all() })
    }

    /// Enable only the chosen hooks. `work` is always created; `full/preview` and `full/refined` only when zone PLY is enabled.
    /// If run.log is off, records go only to standard output when `echo` is set, and `DONE` is not created either. It does not delete the
    /// folder.
    pub fn with_set(out: &Path, echo: bool, set: SinkSet) -> std::io::Result<Self> {
        let dirs: &[&str] = if set.zone_ply { &["full/preview", "full/refined", "work"] } else { &["work"] };
        for d in dirs {
            std::fs::create_dir_all(out.join(d))?;
        }
        let run_log = out.join("run.log");
        let log = RunLog::open(set.run_log.then_some(run_log.as_path()), echo)?;
        let mut sinks: Vec<Box<dyn Sink>> = Vec::new();
        if set.timeline {
            sinks.push(Box::new(TimelineSink::create(&out.join("timeline.txt"))?));
        }
        sinks.push(Box::new(RunLogSink::new(log.clone(), set.run_log.then(|| out.join("DONE")))));
        if set.zone_ply {
            sinks.push(Box::new(ZonePlySink::new(out.join("full"), Some(log.clone()))));
        }
        if set.models {
            sinks.push(Box::new(ModelSink::new(out.join("work").join("models"), Some(log.clone()))));
        }
        if set.snapshots {
            let mut snap = SnapshotSink::new(out.to_path_buf(), log.clone()).with_live_stride(set.live_stride);
            if let Some(cfg) = set.live_stream {
                snap = snap.with_live_stream(cfg);
            }
            sinks.push(Box::new(snap));
        }
        Ok(Self { sinks, log })
    }

    /// The run.log handle (when other hooks write to the same file).
    pub fn run_log(&self) -> RunLog {
        self.log.clone()
    }

    /// An all-event hook (`Pipeline::on_any`) that keeps the event order.
    pub fn into_hook(self) -> Hook {
        any_hook(self)
    }
}

impl Sink for DefaultSinks {
    fn kinds(&self) -> Vec<EventKind> {
        let mut v: Vec<EventKind> = Vec::new();
        for s in &self.sinks {
            for k in s.kinds() {
                if !v.contains(&k) {
                    v.push(k);
                }
            }
        }
        v
    }
    fn handle(&mut self, e: &Event) {
        let k = e.kind();
        for s in &mut self.sinks {
            if s.kinds().contains(&k) {
                s.handle(e);
            }
        }
    }
}

/// The default hooks as a per-kind list (sharing state). For a synchronous pipeline (`sync(true)`) or direct calls.
pub fn default_sinks(out: &Path, opts: &SinkOptions) -> std::io::Result<Vec<(EventKind, Hook)>> {
    Ok(hooks(DefaultSinks::new(out, opts)?))
}

/// Attach the default hooks to a pipeline (a single `on_any`, so the order is preserved even in asynchronous mode).
pub fn attach<R: Reducer>(p: Pipeline<R>, out: &Path, opts: &SinkOptions) -> std::io::Result<Pipeline<R>> {
    Ok(p.on_any(DefaultSinks::new(out, opts)?.into_hook()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::Meta;

    fn meta(s: u64, n: u32) -> Meta {
        Meta { version: 0, at: UNIX_EPOCH + Duration::new(s, n) }
    }
    fn r(zone: usize, lo: usize, hi: usize) -> ZoneRange {
        ZoneRange { zone, lo, hi }
    }

    #[test]
    fn heartbeat_status_json_follows_events() {
        let dir = std::env::temp_dir().join(format!("c3d_status_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let log = RunLog::open(None, false).unwrap();
        let cfg = LiveStream { enu: false, origin_gps: None };
        let mut s = SnapshotSink::new(dir.clone(), log).with_live_stream(cfg);
        let read = || crate::util::Json::parse(&std::fs::read_to_string(dir.join("live/stream/status.json")).unwrap()).unwrap();
        s.handle(&Event::Started { meta: meta(100, 0), positions: Some(26) });
        s.handle(&Event::PositionDone { meta: meta(102, 500_000_000), position: 14, registered: 44, expected: 45 });
        let j = read();
        assert_eq!(
            j.dump(),
            crate::util::Json::obj(vec![
                ("t", crate::util::Json::Num("2.5".into())),
                ("stage", crate::util::Json::Str("triangulation".into())),
                ("zone", crate::util::Json::Int(0)),
                ("positions", crate::util::Json::Int(15)),
                ("positions_total", crate::util::Json::Int(26)),
                ("images", crate::util::Json::Int(44)),
            ])
            .dump()
        );
        s.handle(&Event::RefineStarted { meta: meta(103, 0), zone: 1 });
        let j = read();
        assert_eq!(j.get("stage").map(|v| v.dump()), Some("\"ba\"".into()));
        assert_eq!(j.get("zone").map(|v| v.dump()), Some("1".into()));
        assert!(!dir.join("live/stream/status.json.tmp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn timeline_table() {
        let m = || meta(1_700_000_000, 5);
        let model = Arc::new(Reconstruction::new());
        let cloud = Arc::new(PointCloud::default());
        let cases: Vec<(Event, &str)> = vec![
            (Event::Started { meta: m(), positions: Some(26) }, "start NPOS=26"),
            (Event::ModelInitialized { meta: m(), registered: 42, position: 13, model: model.clone() }, "init_model Registered images: 42"),
            (Event::BaseAdopted { meta: m(), zone: 0 }, "adopt refined 0 as base"),
            (Event::PositionDone { meta: m(), position: 14, registered: 44, expected: 45 }, "pos 14 registered 44/45"),
            (Event::ZoneArrived { meta: m(), range: r(1, 10, 26), model: model.clone() }, "region 1 arrived pos[10,26)"),
            (
                Event::ZonePreview {
                    meta: m(),
                    range: r(1, 10, 26),
                    cloud: cloud.clone(),
                    model: model.clone(),
                    dense: true,
                    frame: "gps".into(),
                },
                "preview 1 ready",
            ),
            (Event::RefineStarted { meta: m(), zone: 1 }, "refined 1 ba_start"),
            (
                Event::ZoneRefinedPose { meta: m(), range: r(1, 10, 26), registered: 48, mean_reproj_px: 0.25095, model: model.clone() },
                "refined 1 pose_ready Registered images: 48 Mean reprojection error: 0.250950px ",
            ),
            (Event::ZoneRefined { meta: m(), range: r(1, 10, 26), cloud, model, dense: true }, "refined 1 ready"),
            (Event::AllPositionsDone { meta: m() }, "all positions done"),
            (Event::AllRefinedDone { meta: m(), chain: None }, "all refined done"),
        ];
        assert_eq!(cases.len(), TIMELINE_TABLE.len());
        for ((e, want), (k, _)) in cases.iter().zip(TIMELINE_TABLE.iter()) {
            assert_eq!(e.kind(), *k);
            assert_eq!(TimelineSink::line(e).unwrap(), format!("1700000000.000000005 {want}"));
        }
        assert!(TimelineSink::line(&Event::Warning { meta: m(), message: "x".into() }).is_none());
    }
}
