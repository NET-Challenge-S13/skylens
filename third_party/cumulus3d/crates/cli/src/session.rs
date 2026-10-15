/*
 * session.rs
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

//! 상태 값과 리듀서: [`Session`] + [`step`] / [`poll`] / [`command`] / [`finish`].
//!
//! 위치 하나의 프레임 묶음([`FrameSet`])이 들어올 때마다 특징 → 새 짝 매칭 → (첫 모델 / 이어 등록 + 삼각측량)
//! → 구역 완료 시 초벌(전경) 조밀화 + 정밀(배경 스레드: BA → GPS ENU 정렬 → 조밀화)을 수행하고,
//! 그 과정을 단계별 [`Event`] 로 내보낸다. 파일·로그는 쓰지 않는다(run.log 줄도 [`Event::Log`] 로 나간다).
//!
//! # 상태와 공유
//! `Session` 은 값이다. `step(&s, frames)` 는 새 `Session` 을 돌려주고 `s` 는 그대로 쓸 수 있다.
//! 모델·영상 목록·대응 그래프 등은 `Arc`/쓰기 시 복사로 공유한다. 단, 특징 저장소는 추가 전용 공유 자료라
//! 같은 이전 값에서 서로 다른 프레임으로 두 번 `step` 하면 저장소가 섞인다(되감기는 [`Command::ResetFrom`] 으로).
//! 배경 정밀 작업 결과(정밀 모델·점군)와 이벤트 버전 번호도 세션 계열 전체가 공유한다.
//!
//! # 이벤트 전달
//! - 기본: 전경 이벤트는 그 호출의 반환 벡터로, 배경 정밀 작업 이벤트는 내부 수신함에 쌓였다가 다음
//!   `step`/`poll`/`command`/`finish` 반환 벡터 앞쪽으로 나간다.
//! - [`Session::set_sink`] 로 수신기를 두면 **모든** 이벤트(전경·배경)가 낸 즉시 버전 순으로 수신기에 전달된다.
//!   이때 반환 벡터의 전경 이벤트는 이미 전달된 것의 사본이다(다시 전달하지 말 것). 수신기는 막히면 안 된다
//!   (내부 잠금 안에서 불린다).
//! - `meta.version` 은 세션 계열 전체에서 단조 증가하며 낸 순서(= timeline 순서)를 준다.
//!
//! # run.log·timeline 재현 규칙(출력 훅용)
//! - timeline 문구는 [`Event::timeline_text`] 가 주며, run.log 에도 `[HH:MM:SS] 문구` 로 같은 자리에 쓴다.
//! - [`Event::Log`] 는 그 자리의 run.log 한 줄(`stamped` 이면 `[HH:MM:SS] ` 머리).
//! - [`Event::Warning`]/[`Event::Error`] 는 같은 내용의 `Log` 와 함께 나가므로 run.log 에 따로 쓰지 않는다.
//! - [`Event::Finished`] 의 단계 시간에는 후처리("post") 시간이 없다(후처리는 출력 훅이 한다).

use crate::densewrap::{dense_model, DenseConfig};
use crate::events::{Command, Event, Frame, Meta, ZoneRange};
use crate::util::StageTimes;
use cumulus3d_align::{align_to_gps, EnuOrigin, ModelAlignerOptions};
use cumulus3d_ba::{bundle_adjust, BaConfig};
use cumulus3d_core::analyzer::ModelStats;
use cumulus3d_core::io::ply::PointCloud;
use cumulus3d_core::io::GpsRecord;
use cumulus3d_core::{CameraId, FeatureStore, ImageId, MatchGraph, MatchGraphOptions, Reconstruction};
use cumulus3d_features::{CameraMode, ExtractionOptions, FeatureExtractor, ImageReport, ImageStatus, SiftEngine};
use cumulus3d_matching::{match_pairs, MatcherBackend, PairMatchingOptions};
use cumulus3d_sfm::{
    global_mapper, register_images, triangulate_points, GlobalSfmOptions, PointTriangulatorOptions, RegistrationOptions, TriangulationScope,
};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Instant, SystemTime};

/// 위치 하나의 영상 하나.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameImage {
    /// 카메라 이름(예: `camF`). 같은 이름은 같은 카메라 내부값을 쓴다.
    pub camera: String,
    /// 영상 이름(`image_root` 기준 상대 경로, 예: `camF/camF_0000.jpg`).
    pub name: String,
}

/// 위치 하나의 프레임 묶음(카메라마다 한 장).
#[derive(Clone, Debug, Default)]
pub struct FrameSet {
    /// 카메라별 영상.
    pub images: Vec<FrameImage>,
    /// 이 묶음과 함께 들어온 GPS 기록(설정의 `gps` 뒤에 덧붙는다).
    pub gps: Vec<GpsRecord>,
}

impl FrameSet {
    /// (카메라, 이름) 목록으로.
    pub fn new<C: Into<String>, N: Into<String>>(images: impl IntoIterator<Item = (C, N)>) -> Self {
        Self { images: images.into_iter().map(|(c, n)| FrameImage { camera: c.into(), name: n.into() }).collect(), gps: Vec::new() }
    }
}

/// 세션 설정.
#[derive(Clone)]
pub struct SessionConfig {
    /// 입력 영상 폴더(영상 이름의 기준).
    pub image_root: PathBuf,
    /// GPS 기록(정렬 순서 = 이 순서).
    pub gps: Vec<GpsRecord>,
    /// 구역 크기·겹침(위치 단위).
    pub span: usize,
    /// 구역 겹침(위치 수).
    pub overlap: usize,
    /// 알려진 전체 위치 수. 있으면 마지막 구역들이 마지막 위치에서 닫히고, 없으면 [`finish`] 에서 닫힌다.
    pub total_positions: Option<usize>,
    /// 새로 등록된 영상만 삼각측량.
    pub incremental_triangulation: bool,
    /// 첫 GPS 기록으로 ENU 원점 고정(기본: 정렬마다 첫 공통 기록).
    pub fixed_enu_origin: bool,
    /// 두 뷰 기하·GPS 정렬 RANSAC 시드.
    pub seed: Option<u64>,
    /// 특징 추출 옵션.
    pub extraction: ExtractionOptions,
    /// SIFT 백엔드.
    pub sift: Arc<dyn SiftEngine>,
    /// 기술자 매칭 백엔드.
    pub matcher: Arc<dyn MatcherBackend>,
    /// 조밀화 설정(None = 조밀화 생략, 점군 이벤트는 `dense: false`).
    pub dense: Option<DenseConfig>,
    /// [`Command::ResetFrom`] 용으로 보관할 위치별 되감기 지점 수(0 = 되감기 불가, 메모리 절약).
    pub history: usize,
    /// 입력 프레임을 RGB 로 디코딩해 처리 전에 [`Event::FrameDecoded`] 로 낸다(기본 꺼짐).
    pub decode_frames: bool,
}

impl SessionConfig {
    /// 기본값: 구역 12 / 겹침 2, SIFT 최대 8192 특징·입력 3200, CPU SIFT·매칭, 조밀화 없음, 되감기 8.
    pub fn new(image_root: impl Into<PathBuf>) -> Self {
        let mut extraction = ExtractionOptions::default();
        extraction.sift.max_num_features = 8192;
        extraction.reader.max_image_size = 3200;
        Self {
            image_root: image_root.into(),
            gps: Vec::new(),
            span: 12,
            overlap: 2,
            total_positions: None,
            incremental_triangulation: false,
            fixed_enu_origin: false,
            seed: None,
            extraction,
            sift: Arc::new(cumulus3d_features::CpuSift::default()),
            matcher: Arc::new(cumulus3d_matching::CpuMatcher::default()),
            dense: None,
            history: 8,
            decode_frames: false,
        }
    }
}

/// 리듀서 입력.
#[derive(Clone, Debug)]
pub enum Input {
    /// 다음 위치의 프레임 묶음.
    Frames(FrameSet),
    /// 상태를 바꾸는 명령.
    Command(Command),
    /// 배경 이벤트 수거만.
    Poll,
    /// run.log 에 한 줄(`Event::Log`).
    Log(String),
    /// 남은 구역 닫기 → 배경 작업 대기 → 요약. 이후 입력은 경고만 낸다.
    Finish,
}

impl From<FrameSet> for Input {
    fn from(f: FrameSet) -> Self {
        Input::Frames(f)
    }
}

impl From<Command> for Input {
    fn from(c: Command) -> Self {
        Input::Command(c)
    }
}

/// [`crate::pipeline::Pipeline`] 이 구동하는 리듀서로서의 세션. `finish` 는 남은 구역을 닫고 배경 정밀 작업을 모두 기다린다.
impl crate::pipeline::Reducer for Session {
    type Input = Input;
    fn step(&mut self, input: Input) -> Vec<Event> {
        self.apply(input)
    }
    fn prelude(&mut self, input: &Input) -> Vec<Event> {
        let Input::Frames(fs) = input else { return Vec::new() };
        let sh = self.shared.clone();
        let mut evs = Vec::new();
        {
            let mut out = Out { sh: &sh, fg: Some(&mut evs), guard: None };
            self.announce(fs, &mut out);
        }
        evs
    }
    fn poll(&mut self) -> Vec<Event> {
        self.apply(Input::Poll)
    }
    fn command(&mut self, c: Command) -> Vec<Event> {
        self.apply(Input::Command(c))
    }
    fn finish(&mut self) -> Vec<Event> {
        self.apply(Input::Finish)
    }
}

/// 즉시 전달 수신기.
pub type EventSink = Arc<dyn Fn(Event) + Send + Sync>;

/// 구역 초벌 결과.
#[derive(Clone, Debug)]
pub struct PreviewZone {
    /// 구역 범위.
    pub range: ZoneRange,
    /// 초벌 모델(구역 0 은 GPS 정렬본).
    pub model: Arc<Reconstruction>,
    /// 초벌 조밀 점군(조밀화를 안 했으면 None).
    pub cloud: Option<Arc<PointCloud>>,
    /// 점군 좌표계("gps" | "refined_K" | "unaligned").
    pub frame: String,
}

/// 구역 정밀 결과(자세 확정 시점부터 있다; 점군은 조밀화가 끝나야 생긴다).
#[derive(Clone, Debug)]
pub struct RefinedZone {
    /// 구역 범위.
    pub range: ZoneRange,
    /// BA + GPS 정렬된 모델.
    pub model: Arc<Reconstruction>,
    /// 정밀 조밀 점군(조밀화가 끝나기 전·생략 시 None).
    pub cloud: Option<Arc<PointCloud>>,
    /// 정밀 조밀화까지 끝났는지.
    pub done: bool,
    generation: u64,
}

// ---------------------------------------------------------------- 이벤트 발행

struct EmitState {
    version: u64,
    sink: Option<EventSink>,
    inbox: Vec<Event>,
}

struct Emitter(Mutex<EmitState>);

impl Emitter {
    fn lock(&self) -> MutexGuard<'_, EmitState> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// 버전·시각을 붙여 낸다. `fg` 가 있으면 반환 벡터에도 넣는다.
    fn emit(&self, make: impl FnOnce(Meta) -> Event, fg: Option<&mut Vec<Event>>) {
        let mut st = self.lock();
        st.version += 1;
        let ev = make(Meta { version: st.version, at: SystemTime::now() });
        match (&st.sink, fg) {
            (Some(s), fg) => {
                let s = s.clone();
                if let Some(v) = fg {
                    v.push(ev.clone());
                }
                s(ev);
            }
            (None, Some(v)) => v.push(ev),
            (None, None) => st.inbox.push(ev),
        }
    }
}

/// 세션 계열이 공유하는 불변 설정 + 배경 상태.
struct Shared {
    cfg: SessionConfig,
    extractor: FeatureExtractor,
    emitter: Emitter,
    times: StageTimes,
    bg: Mutex<Background>,
    start: Mutex<Option<Instant>>,
}

#[derive(Default)]
struct Background {
    jobs: Vec<JoinHandle<()>>,
    ready: BTreeMap<usize, RefinedZone>,
    generation: BTreeMap<usize, u64>,
}

impl Shared {
    fn bg(&self) -> MutexGuard<'_, Background> {
        self.bg.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn generation(&self, zone: usize) -> u64 {
        self.bg().generation.get(&zone).copied().unwrap_or(0)
    }

    /// 구역 결과 폐기: 세대를 올려 진행 중 작업의 이후 결과·이벤트를 버린다.
    fn invalidate(&self, zone: usize) {
        let mut bg = self.bg();
        *bg.generation.entry(zone).or_insert(0) += 1;
        bg.ready.remove(&zone);
    }

    fn aligner_options(&self, gps: &[GpsRecord]) -> ModelAlignerOptions {
        let mut o = ModelAlignerOptions { max_error: 3.0, ..Default::default() };
        o.ransac.random_seed = self.cfg.seed;
        if self.cfg.fixed_enu_origin {
            if let Some(g) = gps.first() {
                o.origin = EnuOrigin::Explicit { lat: g.lat, lon: g.lon, alt: g.alt };
            }
        }
        o
    }

    /// 등록 영상의 GPS 로 ENU 정렬. 실패하면 모델 그대로 두고 경고(설계 결정: 멈추지 않음).
    fn align_enu(&self, rec: &mut Reconstruction, what: &str, gps_all: &[GpsRecord], out: &mut Out) -> bool {
        let names: BTreeSet<String> = rec.registered_images().into_iter().filter_map(|i| rec.image(i).map(|im| im.name.clone())).collect();
        let gps: Vec<GpsRecord> = gps_all.iter().filter(|g| names.contains(&g.name)).cloned().collect();
        let opts = self.aligner_options(gps_all);
        let t0 = Instant::now();
        let r = align_to_gps(rec, &gps, &opts);
        out.stage("align", what, t0);
        match r {
            Ok(a) => {
                out.log(format!(
                    "[align] {what}: 공통 {} 인라이어 {} 오차 평균 {:.3}m 중앙 {:.3}m 축척 {:.4}",
                    a.common.len(),
                    a.num_inliers,
                    a.mean_error,
                    a.median_error,
                    a.sim3.scale
                ));
                true
            }
            Err(e) => {
                out.warn(format!("[align] {what}: 정렬 실패({e}) — 정렬 없이 계속"));
                false
            }
        }
    }

    /// 조밀화(구역 밖 영상 제외). 조밀화가 꺼져 있거나 실패하면 None.
    fn dense_region(&self, kind: &str, k: usize, model: &Reconstruction, keep: &HashSet<String>, out: &mut Out) -> Option<Arc<PointCloud>> {
        let dense = self.cfg.dense.as_ref()?;
        let what = format!("{kind} {k}");
        let t = Instant::now();
        let r = dense_model(model, |n| keep.contains(n), &self.cfg.image_root, dense);
        let total = t.elapsed();
        match r {
            Ok(run) => {
                self.times.add("undistort", run.undistort_time);
                self.times.add("densify", run.densify_time);
                let tm = &run.output.timings;
                out.stamped(format!(
                    "[time] dense {what}: 전체 {:.1}s (왜곡 보정+장면 {:.1}s, 대기 {:.1}s, 조밀화 {:.1}s = 이웃 {:.1} 준비 {:.1} 깊이 {:.1} 필터 {:.1} 융합 {:.1}, 캐시 적중 {})",
                    total.as_secs_f64(),
                    run.undistort_time.as_secs_f64(),
                    run.lock_wait.as_secs_f64(),
                    run.densify_time.as_secs_f64(),
                    tm.neighbors.as_secs_f64(),
                    tm.prepare.as_secs_f64(),
                    tm.depth().as_secs_f64(),
                    tm.filter.as_secs_f64(),
                    tm.fusion.as_secs_f64(),
                    run.output.cache_hits
                ));
                out.log(format!("[dense] {kind} {k} frames {} points {}", run.frames, run.output.cloud.len()));
                let mut cloud = run.output.cloud;
                // PLY 왕복과 같은 값: 색이 없으면 흰색.
                if !cloud.is_empty() && cloud.colors.is_empty() {
                    cloud.colors = vec![[255, 255, 255]; cloud.len()];
                }
                Some(Arc::new(cloud))
            }
            Err(e) => {
                out.warn(format!("[dense] {kind} {k} 실패: {e}"));
                None
            }
        }
    }

    /// 배경 정밀 작업: 사본 → BA → ENU 정렬 → (채택 가능) → 조밀화.
    fn refine_job(
        self: Arc<Self>,
        range: ZoneRange,
        mut rec: Reconstruction,
        generation: u64,
        gps: Arc<Vec<GpsRecord>>,
        keep: Arc<HashSet<String>>,
    ) {
        let k = range.zone;
        let sh = &*self;
        let mut out = Out { sh, fg: None, guard: Some((k, generation)) };
        out.emit(|meta| Event::RefineStarted { meta, zone: k });
        let t0 = Instant::now();
        let r = bundle_adjust(&mut rec, &BaConfig::default());
        out.stage("ba", &format!("refined {k}"), t0);
        match r {
            Ok(s) => out.log(format!(
                "[ba] refined {k}: 반복 {} 비용 {:.4e} → {:.4e} ({:?}), RMS {:.3}px",
                s.num_iterations,
                s.initial_cost,
                s.final_cost,
                s.termination,
                s.rms_reprojection_error()
            )),
            Err(e) => out.warn(format!("[ba] refined {k} 실패: {e}")),
        }
        let adjusted = Arc::new(rec.clone());
        out.emit(|meta| Event::ZoneAdjusted { meta, zone: k, model: adjusted });
        sh.align_enu(&mut rec, &format!("refined {k}"), &gps, &mut out);
        let st = ModelStats::compute(&rec);
        let model = Arc::new(rec);
        {
            let mut bg = sh.bg();
            if bg.generation.get(&k).copied().unwrap_or(0) == generation {
                bg.ready.insert(k, RefinedZone { range, model: model.clone(), cloud: None, done: false, generation });
            }
        }
        let m = model.clone();
        out.emit(|meta| Event::ZoneRefinedPose {
            meta,
            range,
            registered: st.registered_image_count,
            mean_reproj_px: st.mean_reprojection_error,
            model: m,
        });
        let cloud = sh.dense_region("refined", k, &model, &keep, &mut out);
        if let Some(z) = sh.bg().ready.get_mut(&k).filter(|z| z.generation == generation) {
            z.cloud = cloud.clone();
            z.done = true;
        }
        out.emit(|meta| Event::ZoneRefined { meta, range, dense: cloud.is_some(), cloud: cloud.unwrap_or_default(), model });
    }
}

/// 이벤트 출구(전경: 반환 벡터, 배경: 수신기/수신함). 배경은 구역 세대가 바뀌면 이벤트를 버린다.
struct Out<'a> {
    sh: &'a Shared,
    fg: Option<&'a mut Vec<Event>>,
    guard: Option<(usize, u64)>,
}

impl Out<'_> {
    fn emit(&mut self, make: impl FnOnce(Meta) -> Event) {
        if let Some((k, g)) = self.guard {
            if self.sh.generation(k) != g {
                return;
            }
        }
        self.sh.emitter.emit(make, self.fg.as_deref_mut());
    }

    fn log(&mut self, line: impl Into<String>) {
        let line = line.into();
        self.emit(|meta| Event::Log { meta, line, stamped: false });
    }

    fn stamped(&mut self, line: impl Into<String>) {
        let line = line.into();
        self.emit(|meta| Event::Log { meta, line, stamped: true });
    }

    /// run.log 줄 + 같은 내용의 경고.
    fn warn(&mut self, line: String) {
        self.log(line.clone());
        self.emit(|meta| Event::Warning { meta, message: line });
    }

    /// 단계 시간 누적 + `[time] 단계 대상: 초` 줄.
    fn stage(&mut self, stage: &str, what: &str, t0: Instant) {
        let d = t0.elapsed();
        self.sh.times.add(stage, d);
        self.stamped(format!("[time] {stage} {what}: {:.3}s", d.as_secs_f64()));
    }
}

// ---------------------------------------------------------------- 상태 값

#[derive(Clone)]
struct Checkpoint {
    /// 이 위치까지 처리한 직후의 상태.
    position: usize,
    chain: Option<Reconstruction>,
    adopted: Option<usize>,
    cam_ids: Arc<BTreeMap<String, CameraId>>,
    graph: Arc<MatchGraph>,
    gps_len: usize,
}

/// 점진 재구성 상태 값.
#[derive(Clone)]
pub struct Session {
    shared: Arc<Shared>,
    store: Arc<FeatureStore>,
    graph: Arc<MatchGraph>,
    frames: Arc<Vec<Vec<FrameImage>>>,
    position_of: Arc<HashMap<String, usize>>,
    ingested: usize,
    cam_ids: Arc<BTreeMap<String, CameraId>>,
    gps: Arc<Vec<GpsRecord>>,
    chain: Option<Reconstruction>,
    adopted: Option<usize>,
    arrived: BTreeMap<usize, ZoneRange>,
    previews: Arc<BTreeMap<usize, PreviewZone>>,
    checkpoints: VecDeque<Checkpoint>,
    started: bool,
    finished: bool,
    failed: Option<String>,
    /// 프레임을 이미 먼저 낸 위치(같은 묶음을 두 번 내지 않게).
    announced: Option<usize>,
}

/// 다음 프레임 묶음을 처리한 새 상태와 이벤트.
pub fn step(session: &Session, frames: FrameSet) -> (Session, Vec<Event>) {
    session.clone().advance(Input::Frames(frames))
}

/// 배경 정밀 작업 이벤트 수거(수신기가 없을 때). 상태는 바뀌지 않는다.
pub fn poll(session: &Session) -> (Session, Vec<Event>) {
    session.clone().advance(Input::Poll)
}

/// 명령 처리.
pub fn command(session: &Session, cmd: Command) -> (Session, Vec<Event>) {
    session.clone().advance(Input::Command(cmd))
}

/// 남은 구역 닫기 → 배경 작업 대기 → 요약.
pub fn finish(session: &Session) -> (Session, Vec<Event>) {
    session.clone().advance(Input::Finish)
}

impl Session {
    /// 설정으로 빈 세션을 만든다.
    pub fn new(cfg: SessionConfig) -> Self {
        let extractor = FeatureExtractor::with_backend(cfg.sift.clone());
        let gps = Arc::new(cfg.gps.clone());
        let shared = Arc::new(Shared {
            cfg,
            extractor,
            emitter: Emitter(Mutex::new(EmitState { version: 0, sink: None, inbox: Vec::new() })),
            times: StageTimes::default(),
            bg: Mutex::new(Background::default()),
            start: Mutex::new(None),
        });
        Self {
            shared,
            store: Arc::new(FeatureStore::new()),
            graph: Arc::new(MatchGraph::new()),
            frames: Arc::new(Vec::new()),
            position_of: Arc::new(HashMap::new()),
            ingested: 0,
            cam_ids: Arc::new(BTreeMap::new()),
            gps,
            chain: None,
            adopted: None,
            arrived: BTreeMap::new(),
            previews: Arc::new(BTreeMap::new()),
            checkpoints: VecDeque::new(),
            started: false,
            announced: None,
            finished: false,
            failed: None,
        }
    }

    /// 즉시 전달 수신기 설정/해제(세션 계열 공유). 설정 시 쌓여 있던 배경 이벤트를 먼저 전달한다.
    pub fn set_sink(&self, sink: Option<EventSink>) {
        let mut st = self.shared.emitter.lock();
        if let Some(s) = &sink {
            for ev in std::mem::take(&mut st.inbox) {
                s(ev);
            }
        }
        st.sink = sink;
    }

    /// 외부(예: 훅 실행기)에서 이 세션의 버전 순서로 이벤트를 낸다. 수신기가 없으면 수신함에 쌓인다.
    pub fn emit_external(&self, make: impl FnOnce(Meta) -> Event) {
        self.shared.emitter.emit(make, None);
    }

    /// 상태 전이(소유 버전, 복사 없음). `step`/`poll`/`command`/`finish` 는 이것의 얇은 감쌈.
    pub fn advance(mut self, input: Input) -> (Session, Vec<Event>) {
        let evs = self.apply(input);
        (self, evs)
    }

    /// 제자리 상태 전이. 반환 이벤트는 버전 순(그동안 배경 작업이 낸 이벤트도 제자리에 끼워 넣는다).
    pub fn apply(&mut self, input: Input) -> Vec<Event> {
        let sh = self.shared.clone();
        let mut evs: Vec<Event> = Vec::new();
        {
            let mut out = Out { sh: &sh, fg: Some(&mut evs), guard: None };
            if self.finished {
                if !matches!(input, Input::Poll) {
                    out.emit(|meta| Event::Warning { meta, message: "세션이 이미 끝남 — 입력 무시".into() });
                }
            } else {
                if !matches!(input, Input::Poll) {
                    self.ensure_started(&mut out);
                }
                match input {
                    Input::Frames(fs) => {
                        self.announce(&fs, &mut out);
                        if let Some(e) = &self.failed {
                            let message = format!("이전 오류로 중단된 세션 — 프레임 무시 ({e})");
                            out.emit(|meta| Event::Warning { meta, message });
                        } else if let Err(e) = self.process_frames(fs, &mut out) {
                            self.failed = Some(e.clone());
                            out.emit(|meta| Event::Error { meta, message: e });
                        }
                    }
                    Input::Command(c) => self.apply_command(c, &mut out),
                    Input::Poll => {}
                    Input::Log(line) => out.log(line),
                    Input::Finish => {
                        if self.failed.is_none() {
                            self.finish_session(&mut out);
                        }
                        self.finished = true;
                    }
                }
            }
        }
        // 이 호출 동안(과 그 전에) 배경 작업이 수신함에 낸 이벤트를 버전 순으로 합친다.
        let bg = std::mem::take(&mut sh.emitter.lock().inbox);
        if !bg.is_empty() {
            evs.extend(bg);
            evs.sort_by_key(|e| e.meta().version);
        }
        evs
    }

    // ------------------------------------------------------------ 조회

    /// 처리한 위치 수.
    pub fn positions(&self) -> usize {
        self.frames.len()
    }
    /// 위치별 프레임 묶음.
    pub fn frames(&self) -> &[Vec<FrameImage>] {
        &self.frames
    }
    /// 영상 이름의 위치.
    pub fn position_of(&self, name: &str) -> Option<usize> {
        self.position_of.get(name).copied()
    }
    /// 현재 체인 모델(첫 모델 전에는 None).
    pub fn model(&self) -> Option<&Reconstruction> {
        self.chain.as_ref()
    }
    /// 체인의 기준으로 채택된 마지막 정밀 구역.
    pub fn adopted(&self) -> Option<usize> {
        self.adopted
    }
    /// 특징 저장소.
    pub fn store(&self) -> &FeatureStore {
        &self.store
    }
    /// 대응 그래프.
    pub fn graph(&self) -> &MatchGraph {
        &self.graph
    }
    /// 세션 설정.
    pub fn config(&self) -> &SessionConfig {
        &self.shared.cfg
    }
    /// 도착(처리)한 구역들.
    pub fn arrived_zones(&self) -> Vec<ZoneRange> {
        self.arrived.values().copied().collect()
    }
    /// 구역별 초벌 결과.
    pub fn previews(&self) -> &BTreeMap<usize, PreviewZone> {
        &self.previews
    }
    /// 구역별 정밀 결과(자세 확정된 것, 배경 작업과 공유되는 현재 값).
    pub fn refined(&self) -> BTreeMap<usize, RefinedZone> {
        self.shared.bg().ready.clone()
    }
    /// 아직 끝나지 않은 배경 정밀 작업 수.
    pub fn pending_jobs(&self) -> usize {
        self.shared.bg().jobs.iter().filter(|j| !j.is_finished()).count()
    }
    /// 마지막으로 발급한 이벤트 버전.
    pub fn version(&self) -> u64 {
        self.shared.emitter.lock().version
    }
    /// 치명 오류(특징 추출·매칭·첫 모델 실패). 있으면 이후 프레임은 무시된다.
    pub fn failed(&self) -> Option<&str> {
        self.failed.as_deref()
    }
    /// `Finish` 를 처리했는지.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    // ------------------------------------------------------------ 처리

    fn ensure_started(&mut self, out: &mut Out) {
        if self.started {
            return;
        }
        self.started = true;
        *self.shared.start.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
        let positions = self.shared.cfg.total_positions;
        out.emit(|meta| Event::Started { meta, positions });
    }

    /// 다음 위치의 프레임을 RGB 로 디코딩해 [`Event::FrameDecoded`] 로 낸다(위치당 한 번, 카메라 병렬).
    fn announce(&mut self, fs: &FrameSet, out: &mut Out) {
        let sh = self.shared.clone();
        let p = self.frames.len();
        if !sh.cfg.decode_frames || self.finished || self.failed.is_some() || self.announced == Some(p) {
            return;
        }
        self.ensure_started(out);
        self.announced = Some(p);
        let decoded: Vec<Result<Frame, String>> = fs
            .images
            .par_iter()
            .map(|im| {
                let path = sh.cfg.image_root.join(&im.name);
                let img = image::ImageReader::open(&path)
                    .and_then(|r| r.with_guessed_format())
                    .map_err(|e| format!("{}: {e}", path.display()))?
                    .decode()
                    .map_err(|e| format!("{}: 디코딩 실패: {e}", path.display()))?
                    .into_rgb8();
                let gps = fs.gps.iter().chain(self.gps.iter()).find(|g| g.name == im.name).cloned();
                let (width, height) = img.dimensions();
                Ok(Frame {
                    position: p,
                    camera: im.camera.clone(),
                    name: im.name.clone(),
                    path,
                    width,
                    height,
                    rgb: Arc::from(img.into_raw()),
                    gps,
                })
            })
            .collect();
        for d in decoded {
            match d {
                Ok(f) => {
                    let frame = Arc::new(f);
                    out.emit(|meta| Event::FrameDecoded { meta, frame });
                }
                Err(e) => out.emit(|meta| Event::Warning { meta, message: format!("[frame] {e}") }),
            }
        }
    }

    fn process_frames(&mut self, fs: FrameSet, out: &mut Out) -> Result<(), String> {
        let sh = self.shared.clone();
        let cfg = &sh.cfg;
        let p = self.frames.len();
        let names: Vec<String> = fs.images.iter().map(|i| i.name.clone()).collect();
        {
            let pos = Arc::make_mut(&mut self.position_of);
            for n in &names {
                pos.insert(n.clone(), p);
            }
        }
        Arc::make_mut(&mut self.frames).push(fs.images.clone());
        self.ingested += fs.images.len();
        if !fs.gps.is_empty() {
            Arc::make_mut(&mut self.gps).extend(fs.gps);
        }
        out.emit(|meta| Event::FrameIngested { meta, position: p, images: names });

        // 1) 특징: 첫 묶음은 폴더당 카메라, 이후 같은 카메라 id.
        let t0 = Instant::now();
        let r = self.extract(p, &fs.images, out);
        out.stage("feature", &format!("pos {p}"), t0);
        r?;

        // 2) 새 영상이 낀 짝만 매칭.
        let pairs: Vec<(ImageId, ImageId)> = pairs_for(&self.frames, p)
            .into_iter()
            .filter_map(|(a, b)| Some((self.store.image_id_by_name(&a)?, self.store.image_id_by_name(&b)?)))
            .collect();
        let mut mopts = PairMatchingOptions::default();
        mopts.geometry.ransac.random_seed = cfg.seed;
        let t0 = Instant::now();
        let st = match_pairs(&self.store, &pairs, &mopts, cfg.matcher.as_ref());
        out.stage("matching", &format!("pos {p} ({} pairs)", pairs.len()), t0);
        let st = st.map_err(|e| e.to_string())?;
        Arc::make_mut(&mut self.graph).update_from_store(&self.store, &MatchGraphOptions::default());
        out.log(format!(
            "[match] pos {p}: 짝 {} 매칭 {} 유효 기하 {} (그래프 영상 {})",
            pairs.len(),
            st.num_matched,
            st.num_valid_geometries,
            self.graph.image_ids().len()
        ));
        let (np, nv, nm) = (pairs.len(), st.num_valid_geometries, st.num_matched);
        out.emit(|meta| Event::PairsMatched { meta, pairs: np, verified: nv, position: p, matched: nm });

        // 3) 등록.
        let (span, ovl) = (cfg.span, cfg.overlap);
        if p + 1 < span + ovl {
            self.checkpoint(p);
            return Ok(());
        }
        if p + 1 == span + ovl {
            let t0 = Instant::now();
            let o = global_mapper(&self.store, &self.graph, &GlobalSfmOptions::script());
            out.stage("mapper", &format!("pos {p}"), t0);
            let o = o.map_err(|e| e.to_string())?;
            if let Some(f) = &o.failure {
                out.warn(format!("[mapper] 실패 보고: {f}"));
            }
            out.log(format!("[mapper] {:?}", o.summary));
            let rec = o.reconstruction;
            let model = Arc::new(rec.clone());
            let registered = rec.registered_image_count();
            out.emit(|meta| Event::ModelInitialized { meta, registered, position: p, model });
            let ids = rec.registered_images();
            self.emit_registered(&rec, &ids, out);
            self.chain = Some(rec);
        } else {
            // 새 정밀 모델이 나왔으면 그 위에서 이어 등록.
            let mut k = self.adopted.map_or(0, |a| a + 1);
            loop {
                let r = sh.bg().ready.get(&k).map(|z| z.model.clone());
                let Some(r) = r else { break };
                self.chain = Some((*r).clone());
                self.adopted = Some(k);
                out.emit(|meta| Event::BaseAdopted { meta, zone: k });
                k += 1;
            }
            if let Some(base) = &self.chain {
                let mut reg = base.clone();
                let t0 = Instant::now();
                let rr = register_images(&mut reg, &self.store, &self.graph, &RegistrationOptions::default());
                out.stage("registration", &format!("pos {p}"), t0);
                match rr {
                    Ok(rep) => {
                        let new_ids = rep.registered();
                        out.log(format!("[register] pos {p}: 시도 {} 등록 {}", rep.attempts.len(), new_ids.len()));
                        let scope = if cfg.incremental_triangulation {
                            TriangulationScope::Images(new_ids.clone())
                        } else {
                            TriangulationScope::AllRegistered
                        };
                        let topts = PointTriangulatorOptions { clear_points: false, scope, ..Default::default() };
                        let t0 = Instant::now();
                        let tr = triangulate_points(&mut reg, &self.graph, &topts);
                        out.stage("triangulation", &format!("pos {p}"), t0);
                        match tr {
                            Ok(t) => {
                                out.log(format!(
                                    "[triangulate] pos {p}: 생성 {} 연장 {} 완성 {} 병합 {} 필터 {} (점 {})",
                                    t.num_created,
                                    t.num_continued,
                                    t.num_completed,
                                    t.num_merged,
                                    t.num_filtered,
                                    reg.num_points3d()
                                ));
                                self.emit_registered(&reg, &new_ids, out);
                                self.chain = Some(reg);
                            }
                            Err(e) => out.warn(format!("[triangulate] pos {p} 실패: {e} (이전 모델 유지)")),
                        }
                    }
                    Err(e) => out.warn(format!("[register] pos {p} 실패: {e}")),
                }
            }
        }
        let registered = self.chain.as_ref().map_or(0, |r| r.registered_image_count());
        let expected = self.ingested;
        out.emit(|meta| Event::PositionDone { meta, position: p, registered, expected });

        // 4) 구역 완료 → 초벌(전경) + 정밀(배경).
        for z in zones_ending_at(p, span, ovl, cfg.total_positions) {
            if !self.arrived.contains_key(&z.zone) {
                self.process_zone(z, out);
            }
        }
        self.checkpoint(p);
        Ok(())
    }

    fn extract(&mut self, p: usize, images: &[FrameImage], out: &mut Out) -> Result<(), String> {
        let sh = self.shared.clone();
        let cfg = &sh.cfg;
        if self.cam_ids.is_empty() {
            let names: Vec<String> = images.iter().map(|i| i.name.clone()).collect();
            let mut o = cfg.extraction.clone();
            o.reader.camera_mode = CameraMode::PerFolder;
            let rep = sh.extractor.extract_files(&self.store, &cfg.image_root, &names, &o).map_err(|e| e.to_string())?;
            report_extraction(&rep, p, out);
            let cams = Arc::make_mut(&mut self.cam_ids);
            for im in images {
                let s = self.store.image_by_name(&im.name).ok_or_else(|| format!("{} 첫 영상 특징 추출 실패", im.camera))?;
                cams.insert(im.camera.clone(), s.camera_id);
            }
        } else {
            for im in images {
                let mut o = cfg.extraction.clone();
                let known = self.cam_ids.get(&im.camera).copied();
                o.reader.camera_mode = known.map_or(CameraMode::PerFolder, CameraMode::Existing);
                let rep = sh
                    .extractor
                    .extract_files(&self.store, &cfg.image_root, std::slice::from_ref(&im.name), &o)
                    .map_err(|e| e.to_string())?;
                report_extraction(&rep, p, out);
                if known.is_none() {
                    if let Some(s) = self.store.image_by_name(&im.name) {
                        Arc::make_mut(&mut self.cam_ids).insert(im.camera.clone(), s.camera_id);
                    }
                }
            }
        }
        Ok(())
    }

    fn emit_registered(&self, rec: &Reconstruction, ids: &[ImageId], out: &mut Out) {
        for &id in ids {
            let (Some(im), Some(pose)) = (rec.image(id), rec.world_to_cam(id)) else { continue };
            let image = im.name.clone();
            let position = self.position_of(&image).unwrap_or(usize::MAX);
            out.emit(|meta| Event::FrameRegistered { meta, position, image, image_id: id, cam_from_world: pose });
        }
    }

    /// 구역 도착: 초벌(전경, 구역 0 만 GPS 정렬) + 정밀 작업(배경) 시작.
    fn process_zone(&mut self, range: ZoneRange, out: &mut Out) {
        let sh = self.shared.clone();
        let k = range.zone;
        let snap = self.chain.clone().unwrap_or_default();
        let m = Arc::new(snap.clone());
        out.emit(|meta| Event::ZoneArrived { meta, range, model: m });
        let mut preview = snap.clone();
        let mut frame = match self.adopted {
            Some(j) => format!("refined_{j}"),
            None => "unaligned".to_string(),
        };
        if k == 0 && sh.align_enu(&mut preview, "preview 0", &self.gps, out) {
            frame = "gps".to_string();
        }
        let keep: Arc<HashSet<String>> =
            Arc::new(self.position_of.iter().filter(|(_, p)| **p >= range.lo && **p < range.hi).map(|(n, _)| n.clone()).collect());
        let generation = sh.generation(k);
        {
            let (sh2, gps, keep) = (sh.clone(), self.gps.clone(), keep.clone());
            let h = std::thread::spawn(move || sh2.refine_job(range, snap, generation, gps, keep));
            sh.bg().jobs.push(h);
        }
        let cloud = sh.dense_region("preview", k, &preview, &keep, out);
        let model = Arc::new(preview);
        self.arrived.insert(k, range);
        Arc::make_mut(&mut self.previews)
            .insert(k, PreviewZone { range, model: model.clone(), cloud: cloud.clone(), frame: frame.clone() });
        out.emit(|meta| Event::ZonePreview { meta, range, dense: cloud.is_some(), cloud: cloud.unwrap_or_default(), model, frame });
    }

    fn checkpoint(&mut self, position: usize) {
        let n = self.shared.cfg.history;
        if n == 0 {
            return;
        }
        self.checkpoints.push_back(Checkpoint {
            position,
            chain: self.chain.clone(),
            adopted: self.adopted,
            cam_ids: self.cam_ids.clone(),
            graph: self.graph.clone(),
            gps_len: self.gps.len(),
        });
        while self.checkpoints.len() > n {
            self.checkpoints.pop_front();
        }
    }

    fn drop_zone(&mut self, k: usize, out: &mut Out) {
        self.arrived.remove(&k);
        Arc::make_mut(&mut self.previews).remove(&k);
        self.shared.invalidate(k);
        out.emit(|meta| Event::ZoneInvalidated { meta, zone: k });
    }

    fn apply_command(&mut self, cmd: Command, out: &mut Out) {
        match cmd {
            Command::ResetFrom(p) => self.reset_from(p, out),
            Command::InvalidateZone(k) => {
                let Some(range) = self.arrived.get(&k).copied() else {
                    out.emit(|meta| Event::Warning {
                        meta, message: format!("구역 {k} 은 아직 도착하지 않음 — 무효화할 것 없음")
                    });
                    return;
                };
                self.drop_zone(k, out);
                // 현재 체인으로 다시 만든다.
                self.process_zone(range, out);
            }
        }
    }

    /// 위치 p 이후 상태 폐기: 저장소·그래프·체인을 위치 p−1 직후로 되돌리고 그 뒤에 닫힌 구역을 무효화.
    fn reset_from(&mut self, p: usize, out: &mut Out) {
        let npos = self.frames.len();
        if p > npos {
            out.emit(|meta| Event::Warning { meta, message: format!("ResetFrom({p}): 처리한 위치 {npos} 보다 큼 — 무시") });
            return;
        }
        let cp = if p == 0 {
            None
        } else {
            match self.checkpoints.iter().find(|c| c.position + 1 == p) {
                Some(c) => Some(c.clone()),
                None => {
                    out.emit(|meta| Event::Warning {
                        meta, message: format!("ResetFrom({p}): 되감기 지점 없음(history 부족) — 무시")
                    });
                    return;
                }
            }
        };
        let gone: Vec<usize> = self.arrived.iter().filter(|(_, r)| r.hi.saturating_sub(1) >= p).map(|(k, _)| *k).collect();
        for k in gone {
            self.drop_zone(k, out);
        }
        let pos = self.position_of.clone();
        self.store = Arc::new(self.store.retain_copy(|im| pos.get(&im.name).is_some_and(|q| *q < p)));
        Arc::make_mut(&mut self.frames).truncate(p);
        Arc::make_mut(&mut self.position_of).retain(|_, q| *q < p);
        self.ingested = self.frames.iter().map(|f| f.len()).sum();
        self.checkpoints.retain(|c| c.position < p);
        match cp {
            Some(c) => {
                self.chain = c.chain;
                self.adopted = c.adopted;
                self.cam_ids = c.cam_ids;
                self.graph = c.graph;
                Arc::make_mut(&mut self.gps).truncate(c.gps_len);
            }
            None => {
                self.chain = None;
                self.adopted = None;
                self.cam_ids = Arc::new(BTreeMap::new());
                self.graph = Arc::new(MatchGraph::new());
                self.gps = Arc::new(self.shared.cfg.gps.clone());
            }
        }
        out.emit(|meta| Event::Reset { meta, position: p });
    }

    fn finish_session(&mut self, out: &mut Out) {
        let sh = self.shared.clone();
        let (span, ovl) = (sh.cfg.span, sh.cfg.overlap);
        // 전체 수를 몰랐으면 남은 구역을 마지막 위치에서 닫는다.
        let npos = self.frames.len();
        if npos > 0 && npos >= span + ovl {
            for z in zones_upto(npos, span, ovl) {
                if !self.arrived.contains_key(&z.zone) {
                    self.process_zone(z, out);
                }
            }
        }
        out.emit(|meta| Event::AllPositionsDone { meta });
        loop {
            let jobs = std::mem::take(&mut sh.bg().jobs);
            if jobs.is_empty() {
                break;
            }
            for j in jobs {
                let _ = j.join();
            }
        }
        let chain = self.chain.clone().map(Arc::new);
        out.emit(|meta| Event::AllRefinedDone { meta, chain });
        let pipeline_secs = sh.start.lock().unwrap_or_else(|e| e.into_inner()).map_or(0.0, |t| t.elapsed().as_secs_f64());
        let stage_times = sh.times.snapshot().into_iter().map(|(k, (d, n))| (k, d, n)).collect();
        let chain_stats = self.chain.as_ref().map(|c| ModelStats::compute(c).lines()).unwrap_or_default();
        out.emit(|meta| Event::Finished { meta, pipeline_secs, stage_times, chain_stats });
    }
}

fn report_extraction(rep: &[ImageReport], position: usize, out: &mut Out) {
    for r in rep {
        match &r.status {
            ImageStatus::Extracted { image_id, camera_id, num_features } => {
                out.log(format!("[feature] {} id {image_id} cam {camera_id} 특징 {num_features}", r.name));
                let (image, image_id, count) = (r.name.clone(), *image_id, *num_features);
                out.emit(|meta| Event::FeaturesExtracted { meta, image, count, position, image_id });
            }
            ImageStatus::AlreadyExists { image_id } => out.log(format!("[feature] {} id {image_id} 이미 있음", r.name)),
            ImageStatus::Failed { error } => out.warn(format!("[feature] {} 실패: {error}", r.name)),
        }
    }
}

/// 짝 규칙: 같은 카메라 위치 간격 1..5, 8, 16 / 다른 카메라 위치 차 0..4. 위치 p 의 영상이 낀 짝만, 정렬된 이름 짝.
pub fn pairs_for(frames: &[Vec<FrameImage>], p: usize) -> Vec<(String, String)> {
    let mut pr = BTreeSet::new();
    let Some(cur) = frames.get(p) else { return Vec::new() };
    for im in cur {
        for d in [1usize, 2, 3, 4, 5, 8, 16] {
            if p >= d {
                if let Some(o) = frames[p - d].iter().find(|x| x.camera == im.camera) {
                    pr.insert((o.name.clone(), im.name.clone()));
                }
            }
        }
    }
    for a in cur {
        for fq in &frames[p.saturating_sub(4)..=p] {
            for b in fq {
                if a.camera == b.camera {
                    continue;
                }
                let (x, y) = (a.name.clone(), b.name.clone());
                pr.insert(if x <= y { (x, y) } else { (y, x) });
            }
        }
    }
    pr.into_iter().collect()
}

/// 구역: start = 0, span, …; [start−ovl, start+span+ovl) ∩ [0, total). 완료 위치 = hi−1.
pub fn zones_upto(total: usize, span: usize, ovl: usize) -> Vec<ZoneRange> {
    let mut v = Vec::new();
    let mut start = 0;
    let mut k = 0;
    while start < total {
        v.push(ZoneRange { zone: k, lo: start.saturating_sub(ovl), hi: (start + span + ovl).min(total) });
        k += 1;
        start += span.max(1);
    }
    v
}

/// 위치 p 에서 닫히는 구역(전체 수를 모르면 잘리지 않은 구역만).
fn zones_ending_at(p: usize, span: usize, ovl: usize, total: Option<usize>) -> Vec<ZoneRange> {
    let mut v = Vec::new();
    let mut k = 0;
    loop {
        let start = k * span.max(1);
        if start > p || total.is_some_and(|n| start >= n) {
            break;
        }
        let hi = total.map_or(start + span + ovl, |n| (start + span + ovl).min(n));
        if hi == p + 1 {
            v.push(ZoneRange { zone: k, lo: start.saturating_sub(ovl), hi });
        }
        k += 1;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(n: usize) -> Vec<Vec<FrameImage>> {
        (0..n)
            .map(|p| {
                ["camF", "camR", "camL"]
                    .iter()
                    .map(|c| FrameImage { camera: c.to_string(), name: format!("{c}/{c}_{:04}.jpg", p * 3) })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn zones_match_regions() {
        let z = |v: Vec<ZoneRange>| v.into_iter().map(|r| (r.zone, r.lo, r.hi)).collect::<Vec<_>>();
        assert_eq!(z(zones_upto(26, 12, 2)), vec![(0, 0, 14), (1, 10, 26), (2, 22, 26)]);
        // 위치별로 닫히는 구역을 모으면 같은 목록.
        let mut all = Vec::new();
        for p in 0..26 {
            all.extend(zones_ending_at(p, 12, 2, Some(26)));
        }
        all.sort_by_key(|r| r.zone);
        assert_eq!(z(all), z(zones_upto(26, 12, 2)));
        // 전체 수를 모르면 잘리지 않은 구역만 위치에서 닫힌다.
        assert_eq!(z(zones_ending_at(13, 12, 2, None)), vec![(0, 0, 14)]);
        assert_eq!(z(zones_ending_at(25, 12, 2, None)), vec![(1, 10, 26)]);
        assert!(zones_ending_at(25, 12, 2, Some(40)).iter().any(|r| r.zone == 1));
    }

    #[test]
    fn pair_rule() {
        let f = frames(20);
        assert_eq!(pairs_for(&f, 0).len(), 3);
        let p17 = pairs_for(&f, 17);
        assert_eq!(p17.len(), 21 + 27);
        assert!(p17.contains(&("camF/camF_0003.jpg".to_string(), "camF/camF_0051.jpg".to_string())));
    }
}
