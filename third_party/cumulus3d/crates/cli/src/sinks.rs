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

//! 기본 출력 훅: 이벤트를 받아 `cumulus3d stream` 의 파일 출력을 만든다.
//!
//! | 훅 | 받는 이벤트 | 출력 |
//! |---|---|---|
//! | [`TimelineSink`] | 문구가 있는 이벤트(아래 표) | `timeline.txt` (`epoch초.나노 문구`) |
//! | [`RunLogSink`] | 문구가 있는 이벤트, `Log`, `Finished` | `run.log`(+표준 출력), `DONE` |
//! | [`ZonePlySink`] | `ZonePreview`, `ZoneRefined` (`dense` 참) | `full/preview/preview_KK_posLO-HI.ply`, `full/refined/refined_KK_posLO-HI.ply` |
//! | [`ModelSink`] | `ZoneArrived`, `ZonePreview`, `ZoneAdjusted`, `ZoneRefinedPose`, `AllRefinedDone` | `work/models/{snap,preview,ba,refined}_K/`, `work/models/chain/` |
//! | [`SnapshotSink`] | 구역 이벤트들, `AllRefinedDone` | `aligned/`, `snapshots/event_NN_TTTT.Ts_KIND_K.ply`, `snapshots/manifest.json`, `snapshots/timeline.txt`, `final_frame/`, run.log 요약 |
//!
//! 이벤트 → timeline 문구 변환표([`Event::timeline_text`], [`TIMELINE_TABLE`]):
//!
//! | 이벤트 | 문구 |
//! |---|---|
//! | `Started { positions: Some(N) }` | `start NPOS=N` |
//! | `ModelInitialized { registered: N }` | `init_model Registered images: N` |
//! | `BaseAdopted { zone: K }` | `adopt refined K as base` |
//! | `PositionDone { position: P, registered: R, expected: E }` | `pos P registered R/E` |
//! | `ZoneArrived { range: K [LO,HI) }` | `region K arrived pos[LO,HI)` |
//! | `ZonePreview { range: K }` | `preview K ready` |
//! | `RefineStarted { zone: K }` | `refined K ba_start` |
//! | `ZoneRefinedPose { range: K, registered: N, mean_reproj_px: X }` | `refined K pose_ready Registered images: N Mean reprojection error: X(소수 6자리)px ` (끝 공백 포함) |
//! | `ZoneRefined { range: K }` | `refined K ready` |
//! | `AllPositionsDone` | `all positions done` |
//! | `AllRefinedDone` | `all refined done` |
//!
//! 순서: 같은 출력 폴더의 훅들은 [`DefaultSinks`] 하나로 묶여 이벤트 순서대로 처리돼야 한다.
//! 비동기 파이프라인에서는 종류별 레인 사이 순서가 보장되지 않으므로 [`attach`](`on_any` 한 개)를 쓴다.
//! [`default_sinks`](종류별 목록)는 동기 모드(`Pipeline::sync(true)`)나 직접 호출용이다.

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

/// 훅 하나(파이프라인 `.on`/`.on_any` 에 넘기는 모양).
pub type Hook = Box<dyn FnMut(&Event) + Send>;

/// timeline 문구가 나오는 이벤트 종류(변환 규칙은 [`Event::timeline_text`]).
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

/// 이벤트를 받아 출력을 만드는 훅.
pub trait Sink: Send {
    /// 처리하는 이벤트 종류.
    fn kinds(&self) -> Vec<EventKind>;
    /// 이벤트 하나 처리. 실패는 run.log 에 남기고 패닉하지 않는다.
    fn handle(&mut self, e: &Event);
}

/// 훅 하나를 종류별 훅 목록으로(같은 상태를 공유). 종류 사이 순서가 필요하면 동기 모드에서만 쓴다.
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

/// 훅 하나를 모든 이벤트용 훅(`on_any`)으로. 받는 종류만 걸러 처리한다.
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

/// run.log 공유 손잡이. 여러 훅(요약을 쓰는 [`SnapshotSink`] 등)이 같은 파일에 쓴다.
#[derive(Clone)]
pub struct RunLog {
    inner: Arc<RunLogInner>,
}

struct RunLogInner {
    logger: Logger,
    /// 훅에서 잰 단계 시간(예: `post`). `Finished` 의 단계별 시간에 합쳐 쓴다.
    extra_times: Mutex<Vec<(String, Duration)>>,
}

impl RunLog {
    /// `path` 를 새로 만든다(`None` 이면 파일 없이 표준 출력만, `echo` 일 때).
    pub fn open(path: Option<&Path>, echo: bool) -> std::io::Result<Self> {
        Ok(Self { inner: Arc::new(RunLogInner { logger: Logger::new(path, echo)?, extra_times: Mutex::new(Vec::new()) }) })
    }
    /// 내부 기록기.
    pub fn logger(&self) -> &Logger {
        &self.inner.logger
    }
    /// 한 줄 기록.
    pub fn line(&self, s: &str) {
        self.inner.logger.line(s)
    }
    /// 주어진 시각의 `[HH:MM:SS] ` 머리를 붙여 기록.
    pub fn stamped_at(&self, at: SystemTime, s: &str) {
        self.inner.logger.stamped_at(at, s)
    }
    /// 단계 시간 추가(같은 이름은 누적).
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

/// `timeline.txt`: 문구가 있는 이벤트마다 `epoch초.나노 문구` 한 줄(시각 = `meta.at`).
pub struct TimelineSink {
    file: File,
}

impl TimelineSink {
    /// `path` 에 timeline.txt 를 새로 만든다.
    pub fn create(path: &Path) -> std::io::Result<Self> {
        Ok(Self { file: File::create(path)? })
    }

    /// 이벤트 한 줄(문구 없으면 None).
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

/// `run.log`: timeline 문구는 `[HH:MM:SS] 문구`, `Log` 는 그대로(`stamped` 면 시각 머리),
/// `Finished` 는 단계별 시간(훅이 잰 `post` 포함)·최종 체인 모델 요약 뒤 `DONE` 파일.
/// `Warning`/`Error` 는 같은 내용의 `Log` 와 함께 나오므로 따로 쓰지 않는다.
pub struct RunLogSink {
    log: RunLog,
    done: Option<PathBuf>,
}

impl RunLogSink {
    /// `done` = `Finished` 때 만들 빈 파일(보통 `<out>/DONE`).
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
/// (x y z nx ny nz red green blue). `dense` 가 거짓인 이벤트(조밀화 없음·실패)는 쓰지 않는다.
pub struct ZonePlySink {
    full: PathBuf,
    log: Option<RunLog>,
}

impl ZonePlySink {
    /// `full` = 출력 폴더의 `full/`.
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

/// `--save-models`: `work/models/` 아래 구역별 희소 모델.
/// `ZoneArrived` → `snap_K`, `ZonePreview` → `preview_K`, `ZoneAdjusted` → `ba_K`, `ZoneRefinedPose` → `refined_K`,
/// `AllRefinedDone` → `chain`.
pub struct ModelSink {
    dir: PathBuf,
    log: Option<RunLog>,
}

impl ModelSink {
    /// `dir` = 보통 `<out>/work/models`.
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

/// 정렬·마스킹·사건별 스냅샷·manifest·재고정(final_frame) 훅. 구역 결과를 모아 두었다가
/// `AllRefinedDone` 에서 [`post::run_zones`] 로 한 번에 만든다. 시각은 `Started`(없으면 처음 받은 이벤트) 기준 초.
///
/// - 초벌 K(≥ 1) 정렬 구간 = [구역 K 의 lo, 구역 K−1 의 hi) (앞 구역과의 겹침).
/// - 영상 이름 → 위치는 `FrameIngested`·`FeaturesExtracted` 로 모은다.
/// - `Reset { position }` 은 hi > position 인 구역을, `ZoneInvalidated` 는 그 구역을 버린다.
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
}

/// 구역 사건 시각(처음 받은 것).
#[derive(Clone, Copy, Default)]
struct Stamps {
    arrived: Option<SystemTime>,
    preview: Option<SystemTime>,
    pose: Option<SystemTime>,
    refined: Option<SystemTime>,
}

impl SnapshotSink {
    /// 출력 폴더와 run.log 로 만든다.
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
        }
    }

    fn drop_zone(&mut self, k: usize) {
        self.ranges.remove(&k);
        self.times.remove(&k);
        self.preview.remove(&k);
        self.refined.remove(&k);
        self.preview_models.remove(&k);
        self.refined_models.remove(&k);
    }

    /// 지금까지 모은 결과로 후처리. 끝나면 `timeline.txt` 를 `snapshots/` 에 복사한다.
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
                self.ranges.entry(range.zone).or_insert(*range);
                self.times.entry(range.zone).or_default().pose.get_or_insert(at);
                self.refined_models.insert(range.zone, Arc::clone(model));
            }
            Event::ZoneRefined { range, cloud, dense, .. } => {
                self.ranges.entry(range.zone).or_insert(*range);
                self.times.entry(range.zone).or_default().refined.get_or_insert(at);
                if *dense {
                    self.refined.insert(range.zone, Arc::clone(cloud));
                }
            }
            Event::Reset { position, .. } => {
                let ks: Vec<usize> = self.ranges.values().filter(|r| r.hi > *position).map(|r| r.zone).collect();
                for k in ks {
                    self.drop_zone(k);
                }
            }
            Event::ZoneInvalidated { zone, .. } => self.drop_zone(*zone),
            Event::AllRefinedDone { .. } => self.finish(),
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------------------------

/// 기본 훅 설정.
#[derive(Clone, Debug, Default)]
pub struct SinkOptions {
    /// run.log 내용을 표준 출력에도.
    pub echo: bool,
    /// 구역별 희소 모델(`work/models/`) 저장.
    pub save_models: bool,
}

/// 기본 훅 중 켤 것([`DefaultSinks::with_set`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SinkSet {
    /// `timeline.txt`([`TimelineSink`]).
    pub timeline: bool,
    /// `run.log`·`DONE`([`RunLogSink`]).
    pub run_log: bool,
    /// 구역 PLY `full/{preview,refined}/`([`ZonePlySink`]).
    pub zone_ply: bool,
    /// 구역별 희소 모델 `work/models/`([`ModelSink`]).
    pub models: bool,
    /// 정렬·스냅샷·manifest·final_frame([`SnapshotSink`]).
    pub snapshots: bool,
}

impl SinkSet {
    /// 모두 켬.
    pub fn all() -> Self {
        Self { timeline: true, run_log: true, zone_ply: true, models: true, snapshots: true }
    }
}

/// 출력 폴더 하나의 기본 훅 묶음(timeline → run.log → full PLY → 모델 → 스냅샷 순서로 처리).
pub struct DefaultSinks {
    sinks: Vec<Box<dyn Sink>>,
    log: RunLog,
}

impl DefaultSinks {
    /// `out` 아래 `full/preview`, `full/refined`, `work` 를 만들고 `timeline.txt`, `run.log` 를 새로 연다.
    /// 폴더를 지우지는 않는다.
    pub fn new(out: &Path, opts: &SinkOptions) -> std::io::Result<Self> {
        Self::with_set(out, opts.echo, SinkSet { models: opts.save_models, ..SinkSet::all() })
    }

    /// 고른 훅만 켠다. `work` 는 늘 만들고, `full/preview`·`full/refined` 는 구역 PLY 를 켤 때만 만든다.
    /// run.log 를 끄면 기록은 `echo` 일 때 표준 출력에만 가고 `DONE` 도 만들지 않는다. 폴더를 지우지는 않는다.
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
            sinks.push(Box::new(SnapshotSink::new(out.to_path_buf(), log.clone())));
        }
        Ok(Self { sinks, log })
    }

    /// run.log 손잡이(다른 훅이 같은 파일에 쓸 때).
    pub fn run_log(&self) -> RunLog {
        self.log.clone()
    }

    /// 이벤트 순서를 지키는 모든 이벤트용 훅(`Pipeline::on_any`).
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

/// 기본 훅을 종류별 목록으로(상태 공유). 동기 파이프라인(`sync(true)`) 또는 직접 호출용.
pub fn default_sinks(out: &Path, opts: &SinkOptions) -> std::io::Result<Vec<(EventKind, Hook)>> {
    Ok(hooks(DefaultSinks::new(out, opts)?))
}

/// 파이프라인에 기본 훅을 붙인다(`on_any` 하나라 비동기 모드에서도 순서 보존).
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
