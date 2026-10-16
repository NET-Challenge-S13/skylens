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

//! State value and reducer: [`Session`] + [`step`] / [`poll`] / [`command`] / [`finish`].
//!
//! Each time the frame bundle of one position ([`FrameSet`]) arrives, it performs features -> matching of new pairs -> (first model /
//! continued registration + triangulation)
//! -> on zone completion, preview (foreground) densification + refinement (background thread: BA -> GPS ENU alignment -> densification),
//! and emits that process as per-stage [`Event`]s. It writes no files or logs (run.log lines also go out as [`Event::Log`]).
//!
//! # State and sharing
//! `Session` is a value. `step(&s, frames)` returns a new `Session` and `s` can still be used.
//! Models, image lists, correspondence graphs etc. are shared via `Arc` / copy on write. However, the feature store is append-only shared
//! data,
//! so calling `step` twice from the same earlier value with different frames mixes up the store (rewind with [`Command::ResetFrom`]).
//! Background refinement results (refined models and point clouds) and the event version number are also shared by the whole session
//! lineage.
//!
//! # Event delivery
//! - Default: foreground events go out in the return vector of that call; background refinement events accumulate in an internal inbox
//!   and go out
//!   at the front of the return vector of the next `step`/`poll`/`command`/`finish`.
//! - If a receiver is set with [`Session::set_sink`], **all** events (foreground and background) are delivered to it in version order as
//!   soon as they are emitted.
//!   In that case the foreground events in the return vector are copies of what was already delivered (do not deliver them again). The
//!   receiver must not block
//!   (it is called inside an internal lock).
//! - `meta.version` increases monotonically across the whole session lineage and gives the emission order (= timeline order).
//!
//! # Rules for reproducing run.log and timeline (for output hooks)
//! - The timeline text is given by [`Event::timeline_text`], and it is also written to run.log in the same place as `[HH:MM:SS] text`.
//! - [`Event::Log`] is one run.log line at that place (with a `[HH:MM:SS] ` prefix if `stamped`).
//! - [`Event::Warning`]/[`Event::Error`] go out together with a `Log` of the same content, so do not write them to run.log separately.
//! - The stage times of [`Event::Finished`] have no post-processing ("post") time (post-processing is done by the output hook).

use crate::densewrap::{dense_model, DenseConfig};
use crate::events::{Command, Event, Frame, Meta, ZoneRange};
use crate::intrinsics::IntrinsicsPrior;
use crate::post::fit_shared_points;
use crate::util::StageTimes;
use cumulus3d_align::shared::NameFilter;
use cumulus3d_align::{align_to_gps, EnuOrigin, ModelAlignerOptions};
use cumulus3d_ba::{bundle_adjust_with, BaBackend, BaBackendRef, BaConfig};
use cumulus3d_core::analyzer::ModelStats;
use cumulus3d_core::io::ply::PointCloud;
use cumulus3d_core::io::GpsRecord;
use cumulus3d_core::{Camera, CameraId, FeatureStore, ImageId, MatchGraph, MatchGraphOptions, Reconstruction, Rigid3};
use cumulus3d_features::{CameraMode, ExtractionOptions, FeatureExtractor, ImageReport, ImageStatus, SiftEngine};
use cumulus3d_matching::{match_pairs_with, MatcherBackend, PairMatchingOptions, TwoViewBackend};
use cumulus3d_sfm::{
    global_mapper, register_images_with, triangulate_points_with, AbsPoseBackend, GlobalSfmOptions, PointTriangulatorOptions,
    RegistrationOptions, TriangulationBackend, TriangulationScope,
};
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Instant, SystemTime};

/// One image of one position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameImage {
    /// Camera name (e.g. `camF`). The same name uses the same camera intrinsics.
    pub camera: String,
    /// Image name (path relative to `image_root`, e.g. `camF/camF_0000.jpg`).
    pub name: String,
}

/// Frame bundle of one position (one image per camera).
#[derive(Clone, Debug, Default)]
pub struct FrameSet {
    /// Images per camera.
    pub images: Vec<FrameImage>,
    /// GPS records that came with this bundle (appended after `gps` in the settings).
    pub gps: Vec<GpsRecord>,
}

impl FrameSet {
    /// As a (camera, name) list.
    pub fn new<C: Into<String>, N: Into<String>>(images: impl IntoIterator<Item = (C, N)>) -> Self {
        Self { images: images.into_iter().map(|(c, n)| FrameImage { camera: c.into(), name: n.into() }).collect(), gps: Vec::new() }
    }
}

<<<<<<< skylens (vendored)
/// 위치마다 도는 희소 단계의 백엔드 묶음. 기본은 모두 CPU.
#[derive(Clone)]
pub struct SparseBackends {
    /// 두 뷰 기하 검증.
    pub two_view: Arc<dyn TwoViewBackend>,
    /// 등록의 절대 자세 RANSAC.
    pub abs_pose: Arc<dyn AbsPoseBackend>,
    /// 삼각측량(생성 RANSAC·깊이·재투영 검사).
    pub triangulation: Arc<dyn TriangulationBackend>,
}

impl Default for SparseBackends {
    fn default() -> Self {
        Self {
            two_view: Arc::new(cumulus3d_matching::CpuTwoView),
            abs_pose: Arc::new(cumulus3d_sfm::CpuAbsPose),
            triangulation: Arc::new(cumulus3d_sfm::CpuTriangulation),
        }
    }
}

/// 세션 설정.
||||||| base b9f4a9bda
/// 세션 설정.
=======
/// Session settings.
>>>>>>> upstream v0.5.0
#[derive(Clone)]
pub struct SessionConfig {
    /// Input image folder (base of the image names).
    pub image_root: PathBuf,
    /// GPS records (alignment order = this order).
    pub gps: Vec<GpsRecord>,
    /// Zone size and overlap (in units of positions).
    pub span: usize,
    /// Zone overlap (number of positions).
    pub overlap: usize,
    /// Total number of known positions. If set, the last zones are closed at the last position; if not, they are closed in [`finish`].
    pub total_positions: Option<usize>,
    /// Triangulate only newly registered images.
    pub incremental_triangulation: bool,
    /// Fix the ENU origin to the first GPS record (default: the first common record at each alignment).
    pub fixed_enu_origin: bool,
    /// RANSAC seed for two-view geometry and GPS alignment.
    pub seed: Option<u64>,
    /// Feature extraction options.
    pub extraction: ExtractionOptions,
    /// SIFT backend.
    pub sift: Arc<dyn SiftEngine>,
    /// Descriptor matching backend.
    pub matcher: Arc<dyn MatcherBackend>,
<<<<<<< skylens (vendored)
    /// 번들 조정 백엔드(구역 정밀 BA, 첫 모델 전역 매퍼의 BA).
    pub ba_backend: Arc<dyn BaBackend>,
    /// 희소 단계(기하 검증·등록·삼각측량) 백엔드.
    pub sparse: SparseBackends,
    /// 조밀화 설정(None = 조밀화 생략, 점군 이벤트는 `dense: false`).
||||||| base b9f4a9bda
    /// 조밀화 설정(None = 조밀화 생략, 점군 이벤트는 `dense: false`).
=======
    /// Densification settings (None = skip densification, point-cloud events carry `dense: false`).
>>>>>>> upstream v0.5.0
    pub dense: Option<DenseConfig>,
    /// Number of per-position rewind points kept for [`Command::ResetFrom`] (0 = no rewind, saves memory).
    pub history: usize,
    /// Decode input frames to RGB and emit them as [`Event::FrameDecoded`] before processing (off by default).
    pub decode_frames: bool,
    /// Stable display (default off). No preview densification; the refinement of zone K waits for zone K−1, takes the
    /// cameras already shown by it (pose and intrinsics, copied exactly) as fixed in its bundle adjustment, and is not
    /// re-aligned to GPS, so a zone never moves after it is shown and the space only grows. The coordinate frame is
    /// the first refined zone's (local). `AllRefinedDone::chain` is the final model aligned to GPS (ENU) once.
    pub stable_display: bool,
    /// Stable display: standard deviation (m) of the GPS position prior on each newly placed camera. ≤ 0 disables
    /// the prior (then the shown frame is local and drifts with the chain).
    pub gps_prior_sigma: f64,
    /// Stable display: until settled, every refined zone re-adjusts all earlier cameras and the intrinsics (shown zones are
    /// re-placed). Once the largest relative focal change between two consecutive refined models is below this (default
    /// 0.01 = 1 %), the display settles: from then on shown cameras and intrinsics are frozen.
    pub settle_tol: f64,
    /// Stable display: the display settles by force at this zone at the latest (default 3).
    pub settle_max_zone: usize,
    /// Soft intrinsics prior (see [`crate::intrinsics`]): every camera starts from the prior values as soon as it exists and
    /// each refinement pulls its refined parameters back towards them. None = off.
    pub intrinsics_prior: Option<Arc<IntrinsicsPrior>>,
}

impl SessionConfig {
    /// Defaults: zone 12 / overlap 2, SIFT up to 8192 features and input 3200, CPU SIFT and matching, no densification, rewind 8.
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
            ba_backend: Arc::new(cumulus3d_ba::CpuBaBackend),
            sparse: SparseBackends::default(),
            dense: None,
            history: 8,
            decode_frames: false,
            stable_display: false,
            gps_prior_sigma: 1.0,
            settle_tol: 0.01,
            settle_max_zone: 3,
            intrinsics_prior: None,
        }
    }
}

/// Reducer input.
#[derive(Clone, Debug)]
pub enum Input {
    /// Frame bundle of the next position.
    Frames(FrameSet),
    /// A command that changes the state.
    Command(Command),
    /// Only collect background events.
    Poll,
    /// One line to run.log (`Event::Log`).
    Log(String),
    /// Close the remaining zones -> wait for background jobs -> summary. Later inputs only produce warnings.
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

/// The session as a reducer driven by [`crate::pipeline::Pipeline`]. `finish` closes the remaining zones and waits for all background
/// refinement jobs.
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

/// Immediate-delivery receiver.
pub type EventSink = Arc<dyn Fn(Event) + Send + Sync>;

/// Zone preview result.
#[derive(Clone, Debug)]
pub struct PreviewZone {
    /// Zone range.
    pub range: ZoneRange,
    /// Preview model (zone 0 is the GPS-aligned one).
    pub model: Arc<Reconstruction>,
    /// Preview dense point cloud (None if densification was not run).
    pub cloud: Option<Arc<PointCloud>>,
    /// Point-cloud coordinate frame ("gps" | "refined_K" | "unaligned").
    pub frame: String,
}

/// Zone refined result (present from the time the pose is finalized; the point cloud exists once densification finishes).
#[derive(Clone, Debug)]
pub struct RefinedZone {
    /// Zone range.
    pub range: ZoneRange,
    /// Model after BA + GPS alignment.
    pub model: Arc<Reconstruction>,
    /// Refined dense point cloud (None before densification finishes or if it is skipped).
    pub cloud: Option<Arc<PointCloud>>,
    /// Whether refined densification has finished.
    pub done: bool,
    generation: u64,
}

// ---------------------------------------------------------------- Event emission

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

    /// Attaches version and time and emits it. If `fg` is present, also puts it in the return vector.
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

/// Immutable settings + background state shared by the session lineage.
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
    /// Stable display: the zone at which the display settled (shown cameras and intrinsics frozen from the next zone on).
    settled: Option<usize>,
    /// Stable display: image-name set each zone was densified with (to densify it again while settling).
    zone_keep: BTreeMap<usize, Arc<HashSet<String>>>,
    /// Stable display: the model each zone's cloud was densified with (its intrinsics decide whether to densify again).
    dense_basis: BTreeMap<usize, Arc<Reconstruction>>,
    /// Stable display: zone -> generation of the last refine job of that zone that ran to its end.
    finished: BTreeMap<usize, u64>,
}

/// Marks a stable-display refine job as finished (also on panic) so the next zone's settle work can proceed.
struct FinishMark<'a> {
    sh: &'a Shared,
    zone: usize,
    generation: u64,
}

impl Drop for FinishMark<'_> {
    fn drop(&mut self) {
        self.sh.bg().finished.insert(self.zone, self.generation);
    }
}

impl Shared {
    fn bg(&self) -> MutexGuard<'_, Background> {
        self.bg.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn generation(&self, zone: usize) -> u64 {
        self.bg().generation.get(&zone).copied().unwrap_or(0)
    }

    /// Discard a zone result: raise the generation so that later results and events of in-flight jobs are dropped.
    fn invalidate(&self, zone: usize) {
        let mut bg = self.bg();
        *bg.generation.entry(zone).or_insert(0) += 1;
        bg.ready.remove(&zone);
        bg.zone_keep.remove(&zone);
        bg.dense_basis.remove(&zone);
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

    /// ENU alignment using the GPS of registered images. On failure, keep the model as is and warn (design decision: do not stop).
    /// Waits for the refined pose of zone `k` (stable display orders refinements). None if it does not arrive
    /// (zone invalidated or refinement failed) within the time limit, or the generation changed.
    fn wait_refined(&self, k: usize, generation: u64) -> Option<Arc<Reconstruction>> {
        let t0 = Instant::now();
        let me = k + 1;
        loop {
            if let Some(z) = self.bg().ready.get(&k) {
                return Some(z.model.clone());
            }
            if self.generation(me) != generation || t0.elapsed().as_secs() > 1800 {
                return None;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    /// Waits until the refine job of zone `k` has run to its end (stable display, settling: earlier zones are only
    /// re-placed once nothing else writes them). False on timeout or when zone `me` (generation `generation`) was invalidated.
    fn wait_finished(&self, k: usize, me: usize, generation: u64) -> bool {
        let t0 = Instant::now();
        loop {
            {
                let bg = self.bg();
                if bg.finished.get(&k).is_some_and(|g| *g == bg.generation.get(&k).copied().unwrap_or(0)) {
                    return true;
                }
            }
            if self.generation(me) != generation || t0.elapsed().as_secs() > 1800 {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    /// Stable display, not yet settled: zone `k`'s adjustment moved the earlier cameras, so every shown zone `j < k` is
    /// placed into `model` again. A zone whose cloud was made with intrinsics differing by `settle_tol` or more from
    /// `model`'s is densified again; otherwise its cloud is moved with the Sim3 from its previous model to `model`.
    /// The zone is emitted again (`ZoneRefinedPose` + `ZoneRefined`) under its own generation.
    fn replace_shown(&self, k: usize, generation: u64, model: &Arc<Reconstruction>, st: &ModelStats) {
        if !self.wait_finished(k - 1, k, generation) {
            return;
        }
        for j in 0..k {
            let snap = {
                let bg = self.bg();
                let gj = bg.generation.get(&j).copied().unwrap_or(0);
                bg.ready
                    .get(&j)
                    .filter(|z| z.generation == gj)
                    .map(|z| (z.clone(), gj, bg.zone_keep.get(&j).cloned(), bg.dense_basis.get(&j).cloned()))
            };
            let Some((z, gj, keep, basis)) = snap else { continue };
            let mut out = Out { sh: self, fg: None, guard: Some((j, gj)) };
            let change = basis.as_ref().map_or(0.0, |b| intrinsics_change(b.cameras(), model.cameras()));
            let mut basis_new = None;
            let cloud = z.cloud.as_ref().map(|old| {
                if change >= self.cfg.settle_tol {
                    if let Some(c) = keep.as_ref().and_then(|keep| self.dense_region("refined", j, model, keep, &mut out)) {
                        out.log(format!("[settle] 구역 {j} 다시 조밀화 (내부값 변화 {:.2}%)", change * 100.0));
                        basis_new = Some(model.clone());
                        return c;
                    }
                }
                // Shared 3D points seen by zone j's images first, then all shared points, then the camera-center fit.
                let by_points = keep
                    .as_ref()
                    .and_then(|keep| {
                        let keep = keep.clone();
                        let filter: NameFilter = Arc::new(move |n: &str| keep.contains(n));
                        fit_shared_points(&z.model, model, Some(filter), MIN_SHARED_PAIRS)
                    })
                    .or_else(|| fit_shared_points(&z.model, model, None, MIN_SHARED_PAIRS));
                let mut c = (**old).clone();
                if let Some((f, n)) = by_points {
                    let t = f.sim3;
                    cumulus3d_align::transform_cloud(&mut c, &t);
                    out.log(format!(
                        "[settle] 구역 {j} 점군 이동: 공유 점 {n}, 잔차 중앙 {:.3}m, 축척 {:.4}, 이동 {:.2}m",
                        f.median,
                        t.scale,
                        t.translation.norm()
                    ));
                } else {
                    let t = keep.as_ref().and_then(|keep| sim3_between(&z.model, model, |n| keep.contains(n)));
                    match t {
                        Some(t) => {
                            cumulus3d_align::transform_cloud(&mut c, &t);
                            out.log(format!(
                                "[settle] 구역 {j} 점군 이동(카메라 중심 대체): 이동 {:.3}m 축척 {:.4}",
                                t.translation.norm(),
                                t.scale
                            ));
                        }
                        None => out.log(format!("[settle] 구역 {j} 점군 이동 생략: 공통 영상 부족")),
                    }
                }
                Arc::new(c)
            });
            {
                let mut bg = self.bg();
                if bg.generation.get(&j).copied().unwrap_or(0) != gj {
                    continue;
                }
                let Some(r) = bg.ready.get_mut(&j).filter(|r| r.generation == gj) else { continue };
                r.model = model.clone();
                r.cloud = cloud.clone();
                r.done = true;
                if let Some(b) = basis_new {
                    bg.dense_basis.insert(j, b);
                }
            }
            let (range, m) = (z.range, model.clone());
            let (registered, mean_reproj_px) = (st.registered_image_count, st.mean_reprojection_error);
            out.emit(|meta| Event::ZoneRefinedPose { meta, range, registered, mean_reproj_px, model: m });
            let m = model.clone();
            out.emit(|meta| Event::ZoneRefined { meta, range, dense: cloud.is_some(), cloud: cloud.unwrap_or_default(), model: m });
        }
    }

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

    /// Densification (excluding images outside the zone). None if densification is off or fails.
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
                // Same value as the PLY round trip: white if there is no color.
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

    /// Background refinement job: copy -> BA -> ENU alignment -> (adoptable) -> densification.
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
        let _finish = FinishMark { sh, zone: k, generation };
        out.emit(|meta| Event::RefineStarted { meta, zone: k });
        let mut ba = BaConfig::default();
        // Stable display: the previous zone's model, and whether the display is still settling (intrinsics and shown
        // cameras free) in this zone.
        let mut prev: Option<Arc<Reconstruction>> = None;
        let mut settling = false;
        if sh.cfg.stable_display {
            let mut bg = sh.bg();
            if bg.generation.get(&k).copied().unwrap_or(0) == generation {
                bg.zone_keep.insert(k, keep.clone());
            }
        }
        if sh.cfg.stable_display {
            let targets = if sh.cfg.gps_prior_sigma > 0.0 { enu_targets(&rec, &gps) } else { HashMap::new() };
            if k == 0 {
                // Bring the first model into ENU once; every later zone is placed in this frame.
                let (src, dst): (Vec<_>, Vec<_>) = targets.iter().filter_map(|(id, g)| rec.projection_center(*id).map(|c| (c, *g))).unzip();
                if src.len() >= 3 {
                    if let Some(t) = cumulus3d_align::umeyama(&src, &dst, true) {
                        rec.transform(&t);
                        out.log(format!("[stable] refined 0: ENU 초기 정렬 GPS {} 개, 축척 {:.4}", src.len(), t.scale));
                    }
                }
                settling = true;
            } else {
                prev = sh.wait_refined(k - 1, generation);
                settling = sh.bg().settled.is_none();
                match (&prev, settling) {
                    (Some(p), false) => {
                        let (n, scale) = anchor_to_shown(&mut rec, p, &mut ba);
                        out.log(format!("[stable] refined {k}: 고정 영상 {n}, 이전 구역에 맞춘 축척 {scale:.4}"));
                    }
                    (Some(p), true) => {
                        let scale = sim3_onto_shown(&mut rec, p).map_or(1.0, |(t, _)| t.scale);
                        out.log(format!("[stable] refined {k}: 정착 전 — 모든 자세·내부값 자유, 이전 구역에 맞춘 축척 {scale:.4}"));
                    }
                    (None, _) => out.warn(format!("[stable] refined {k}: 앞 구역 결과 없음 — 고정 없이 조정")),
                }
            }
            let sigma = sh.cfg.gps_prior_sigma;
            for (id, g) in &targets {
                if !ba.constant_poses.contains(id) {
                    ba.position_priors.insert(*id, (*g, sigma));
                }
            }
            if !ba.position_priors.is_empty() {
                out.log(format!("[stable] refined {k}: GPS 위치 제약 {} 개 (σ {sigma} m)", ba.position_priors.len()));
            }
        }
        if let Some(pr) = &sh.cfg.intrinsics_prior {
            for (cid, cam) in rec.cameras() {
                if !ba.constant_cameras.contains(cid) {
                    if let Some(p) = pr.ba_prior(cam) {
                        ba.intrinsics_priors.insert(*cid, p);
                    }
                }
            }
            if !ba.intrinsics_priors.is_empty() {
                out.log(format!(
                    "[prior] refined {k}: 내부값 사전 카메라 {} 개 (초점 σ {} 상대, 왜곡 σ {})",
                    ba.intrinsics_priors.len(),
                    pr.focal_sigma,
                    pr.dist_sigma
                ));
            }
        }
        let t0 = Instant::now();
<<<<<<< skylens (vendored)
        let r = bundle_adjust_with(&mut rec, &BaConfig::default(), &*sh.cfg.ba_backend);
||||||| base b9f4a9bda
        let r = bundle_adjust(&mut rec, &BaConfig::default());
=======
        let r = bundle_adjust(&mut rec, &ba);
>>>>>>> upstream v0.5.0
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
        // Settling BA moved the whole model (gauge freedom); bring it back into the previously shown frame.
        if let (true, Some(p)) = (settling && k > 0, &prev) {
            match fit_shared_points(&rec, p, None, MIN_SHARED_PAIRS) {
                Some((f, n)) => {
                    let t = f.sim3;
                    rec.transform(&t);
                    out.log(format!(
                        "[settle] refined {k}: 화면 좌표계 재정렬 공유 점 {n}, 잔차 중앙 {:.3}m, 축척 {:.4}, 이동 {:.2}m",
                        f.median,
                        t.scale,
                        t.translation.norm()
                    ));
                }
                None => match sim3_onto_shown(&mut rec, p) {
                    Some((t, _)) => out.log(format!(
                        "[settle] refined {k}: 화면 좌표계 재정렬 (카메라 중심 대체) 축척 {:.4}, 이동 {:.2}m",
                        t.scale,
                        t.translation.norm()
                    )),
                    None => out.warn(format!("[settle] refined {k}: 화면 좌표계 재정렬 생략: 공통 영상 부족")),
                },
            }
        }
        let adjusted = Arc::new(rec.clone());
        out.emit(|meta| Event::ZoneAdjusted { meta, zone: k, model: adjusted });
        if !sh.cfg.stable_display {
            sh.align_enu(&mut rec, &format!("refined {k}"), &gps, &mut out);
        }
        let st = ModelStats::compute(&rec);
        // Settle decision (before the pose is published, so the next zone sees it).
        let mut settle_now = false;
        if settling {
            let change = prev.as_ref().map(|p| intrinsics_change(p.cameras(), rec.cameras()));
            let forced = k >= sh.cfg.settle_max_zone;
            settle_now = forced || change.is_some_and(|c| c < sh.cfg.settle_tol);
            let verdict = match (settle_now, forced) {
                (true, true) => "강제 고정",
                (true, false) => "고정",
                _ => "정착 중",
            };
            match change {
                Some(c) => out.log(format!("[settle] refined {k}: 내부값 변화 {:.2}% → {verdict}", c * 100.0)),
                None => out.log(format!("[settle] refined {k}: → {verdict}")),
            }
        }
        let model = Arc::new(rec);
        {
            let mut bg = sh.bg();
            if bg.generation.get(&k).copied().unwrap_or(0) == generation {
                if settle_now {
                    bg.settled = Some(k);
                }
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
        {
            let mut bg = sh.bg();
            if bg.generation.get(&k).copied().unwrap_or(0) == generation {
                if let Some(z) = bg.ready.get_mut(&k).filter(|z| z.generation == generation) {
                    z.cloud = cloud.clone();
                    z.done = true;
                }
                bg.dense_basis.insert(k, model.clone());
            }
        }
        let m = model.clone();
        out.emit(|meta| Event::ZoneRefined { meta, range, dense: cloud.is_some(), cloud: cloud.unwrap_or_default(), model: m });
        // Still settling: the earlier cameras moved with this adjustment, so the zones already shown are placed again.
        if settling && k > 0 {
            sh.replace_shown(k, generation, &model, &st);
        }
    }
}

/// Event exit (foreground: return vector, background: receiver/inbox). The background drops events when the zone generation changes.
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

    /// run.log line + a warning with the same content.
    fn warn(&mut self, line: String) {
        self.log(line.clone());
        self.emit(|meta| Event::Warning { meta, message: line });
    }

    /// Accumulate stage time + a `[time] stage target: seconds` line.
    fn stage(&mut self, stage: &str, what: &str, t0: Instant) {
        let d = t0.elapsed();
        self.sh.times.add(stage, d);
        self.stamped(format!("[time] {stage} {what}: {:.3}s", d.as_secs_f64()));
    }
}

// ---------------------------------------------------------------- State value

#[derive(Clone)]
struct Checkpoint {
    /// The state right after processing up to this position.
    position: usize,
    chain: Option<Reconstruction>,
    adopted: Option<usize>,
    cam_ids: Arc<BTreeMap<String, CameraId>>,
    graph: Arc<MatchGraph>,
    gps_len: usize,
}

/// Incremental reconstruction state value.
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
    /// Positions whose frames were already emitted (so the same bundle is not emitted twice).
    announced: Option<usize>,
}

/// The new state and events after processing the next frame bundle.
pub fn step(session: &Session, frames: FrameSet) -> (Session, Vec<Event>) {
    session.clone().advance(Input::Frames(frames))
}

/// Collect background refinement events (when there is no receiver). The state does not change.
pub fn poll(session: &Session) -> (Session, Vec<Event>) {
    session.clone().advance(Input::Poll)
}

/// Command handling.
pub fn command(session: &Session, cmd: Command) -> (Session, Vec<Event>) {
    session.clone().advance(Input::Command(cmd))
}

/// Close the remaining zones -> wait for background jobs -> summary.
pub fn finish(session: &Session) -> (Session, Vec<Event>) {
    session.clone().advance(Input::Finish)
}

impl Session {
    /// Create an empty session from the settings.
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

    /// Set/clear the immediate-delivery receiver (shared by the session lineage). When set, background events accumulated so far are
    /// delivered first.
    pub fn set_sink(&self, sink: Option<EventSink>) {
        let mut st = self.shared.emitter.lock();
        if let Some(s) = &sink {
            for ev in std::mem::take(&mut st.inbox) {
                s(ev);
            }
        }
        st.sink = sink;
    }

    /// Emit an event from outside (e.g. a hook runner) in this session's version order. If there is no receiver, it accumulates in the
    /// inbox.
    pub fn emit_external(&self, make: impl FnOnce(Meta) -> Event) {
        self.shared.emitter.emit(make, None);
    }

    /// State transition (owned version, no copying). `step`/`poll`/`command`/`finish` are thin wrappers around this.
    pub fn advance(mut self, input: Input) -> (Session, Vec<Event>) {
        let evs = self.apply(input);
        (self, evs)
    }

    /// In-place state transition. Returned events are in version order (events emitted by background jobs in the meantime are also
    /// inserted in place).
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
        // Merge, in version order, the events that background jobs put into the inbox during (and before) this call.
        let bg = std::mem::take(&mut sh.emitter.lock().inbox);
        if !bg.is_empty() {
            evs.extend(bg);
            evs.sort_by_key(|e| e.meta().version);
        }
        evs
    }

    // ------------------------------------------------------------ Queries

    /// Number of processed positions.
    pub fn positions(&self) -> usize {
        self.frames.len()
    }
    /// Frame bundle per position.
    pub fn frames(&self) -> &[Vec<FrameImage>] {
        &self.frames
    }
    /// Position of an image name.
    pub fn position_of(&self, name: &str) -> Option<usize> {
        self.position_of.get(name).copied()
    }
    /// Current chain model (None before the first model).
    pub fn model(&self) -> Option<&Reconstruction> {
        self.chain.as_ref()
    }
    /// The last refined zone adopted as the chain reference.
    pub fn adopted(&self) -> Option<usize> {
        self.adopted
    }
    /// Feature store.
    pub fn store(&self) -> &FeatureStore {
        &self.store
    }
    /// Correspondence graph.
    pub fn graph(&self) -> &MatchGraph {
        &self.graph
    }
    /// Session settings.
    pub fn config(&self) -> &SessionConfig {
        &self.shared.cfg
    }
    /// Zones that have arrived (been processed).
    pub fn arrived_zones(&self) -> Vec<ZoneRange> {
        self.arrived.values().copied().collect()
    }
    /// Per-zone preview results.
    pub fn previews(&self) -> &BTreeMap<usize, PreviewZone> {
        &self.previews
    }
    /// Per-zone refined results (those with finalized poses, the current value shared with background jobs).
    pub fn refined(&self) -> BTreeMap<usize, RefinedZone> {
        self.shared.bg().ready.clone()
    }
    /// Number of background refinement jobs not yet finished.
    pub fn pending_jobs(&self) -> usize {
        self.shared.bg().jobs.iter().filter(|j| !j.is_finished()).count()
    }
    /// The last event version issued.
    pub fn version(&self) -> u64 {
        self.shared.emitter.lock().version
    }
    /// Fatal error (feature extraction, matching, first model failure). If set, later frames are ignored.
    pub fn failed(&self) -> Option<&str> {
        self.failed.as_deref()
    }
    /// Whether `Finish` was handled.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    // ------------------------------------------------------------ Processing

    fn ensure_started(&mut self, out: &mut Out) {
        if self.started {
            return;
        }
        self.started = true;
        *self.shared.start.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
        let positions = self.shared.cfg.total_positions;
        out.emit(|meta| Event::Started { meta, positions });
    }

    /// Decode the next position's frames to RGB and emit them as [`Event::FrameDecoded`] (once per position, cameras in parallel).
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

        // 1) Features: for the first bundle, one camera per folder; afterwards the same camera id.
        let t0 = Instant::now();
        let r = self.extract(p, &fs.images, out);
        out.stage("feature", &format!("pos {p}"), t0);
        r?;

        // 2) Match only pairs that involve a new image.
        let pairs: Vec<(ImageId, ImageId)> = pairs_for(&self.frames, p)
            .into_iter()
            .filter_map(|(a, b)| Some((self.store.image_id_by_name(&a)?, self.store.image_id_by_name(&b)?)))
            .collect();
        let mut mopts = PairMatchingOptions::default();
        mopts.geometry.ransac.random_seed = cfg.seed;
        let t0 = Instant::now();
        let st = match_pairs_with(&self.store, &pairs, &mopts, cfg.matcher.as_ref(), cfg.sparse.two_view.as_ref());
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

        // 3) Registration.
        let (span, ovl) = (cfg.span, cfg.overlap);
        if p + 1 < span + ovl {
            self.checkpoint(p);
            return Ok(());
        }
        if p + 1 == span + ovl {
            let t0 = Instant::now();
            let gopts = GlobalSfmOptions { ba_backend: BaBackendRef::new(cfg.ba_backend.clone()), ..GlobalSfmOptions::script() };
            let o = global_mapper(&self.store, &self.graph, &gopts);
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
            // If a new refined model has appeared, continue registering on top of it.
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
                let rr = register_images_with(&mut reg, &self.store, &self.graph, &RegistrationOptions::default(), cfg.sparse.abs_pose.as_ref());
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
                        let tr = triangulate_points_with(&mut reg, &self.graph, &topts, cfg.sparse.triangulation.as_ref());
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

        // 4) Zone completion -> preview (foreground) + refinement (background).
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
            self.apply_intrinsics_prior(out)?;
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
                    self.apply_intrinsics_prior(out)?;
                }
            }
        }
        Ok(())
    }

    /// Soft intrinsics prior: sets the store's cameras to the prior values (before any registration or model uses them).
    /// A camera without an entry is left alone; an entry whose length differs from the camera model's is an error.
    fn apply_intrinsics_prior(&self, out: &mut Out) -> Result<(), String> {
        let Some(pr) = &self.shared.cfg.intrinsics_prior else { return Ok(()) };
        for mut cam in self.store.cameras() {
            let before = cam.params.clone();
            if pr.apply(&mut cam)? && cam.params != before {
                out.log(format!("[prior] 카메라 {}: 내부값 초기화 {} → {}", cam.camera_id, fmt_params(&before), fmt_params(&cam.params)));
                self.store.update_camera(cam).map_err(|e| e.to_string())?;
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

    /// Zone arrival: preview (foreground, only zone 0 is GPS-aligned) + start the refinement job (background).
    fn process_zone(&mut self, range: ZoneRange, out: &mut Out) {
        let sh = self.shared.clone();
        let k = range.zone;
        let snap = self.chain.clone().unwrap_or_default();
        let m = Arc::new(snap.clone());
        out.emit(|meta| Event::ZoneArrived { meta, range, model: m });
        if sh.cfg.stable_display {
            let keep: Arc<HashSet<String>> =
                Arc::new(self.position_of.iter().filter(|(_, p)| **p >= range.lo && **p < range.hi).map(|(n, _)| n.clone()).collect());
            let generation = sh.generation(k);
            let (sh2, gps) = (sh.clone(), self.gps.clone());
            let h = std::thread::spawn(move || sh2.refine_job(range, snap, generation, gps, keep));
            sh.bg().jobs.push(h);
            self.arrived.insert(k, range);
            return;
        }
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
                // Rebuild with the current chain.
                self.process_zone(range, out);
            }
        }
    }

    /// Discard the state after position p: roll the store, graph and chain back to right after position p-1, and invalidate the zones
    /// closed after that.
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
        // If the total count was unknown, close the remaining zones at the last position.
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
        let chain = match (&self.chain, sh.cfg.stable_display) {
            (Some(c), true) => {
                // Final outputs in ENU through one GPS alignment of a copy of the whole model (with or without GPS priors the
                // display frame is the first shown zone's frame); the display chain itself stays untouched. On failure the
                // display-frame chain is handed on.
                let mut enu = c.clone();
                if sh.align_enu(&mut enu, "final (stable display)", &self.gps, out) {
                    // Display → ENU scale on the camera centers: close to the zone-0 initial ENU fit (1.0 when the display frame
                    // stayed ENU through the GPS priors); a large gap means the frame drifted.
                    match display_to_enu_scale(c, &enu) {
                        Some(s) => out.log(format!("[stable] 표시→ENU 축척 {s:.4} (1.0 이면 표시 좌표계가 ENU 와 같음)")),
                        None => out.log("[stable] 표시→ENU 축척: 공통 영상 부족".to_string()),
                    }
                    Some(Arc::new(enu))
                } else {
                    out.warn(
                        "[align] final (stable display): GPS 정렬 실패 — 표시 좌표계 모델을 그대로 넘김(final_frame/ 은 표시 좌표계)"
                            .to_string(),
                    );
                    Some(Arc::new(c.clone()))
                }
            }
            (c, _) => c.clone().map(Arc::new),
        };
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

fn fmt_params(p: &[f64]) -> String {
    format!("[{}]", p.iter().map(|x| format!("{x:.5}")).collect::<Vec<_>>().join(" "))
}

/// Scale of the Sim3 (Umeyama on the projection centers of the images registered in both) taking `display` to `enu`.
fn display_to_enu_scale(display: &Reconstruction, enu: &Reconstruction) -> Option<f64> {
    let (mut src, mut dst) = (Vec::new(), Vec::new());
    for id in display.registered_images() {
        let name = display.image(id)?.name.clone();
        let Some(eid) = enu.image_by_name(&name).map(|im| im.image_id) else { continue };
        if let (Some(a), Some(b)) = (display.projection_center(id), enu.projection_center(eid)) {
            src.push(a);
            dst.push(b);
        }
    }
    if src.len() < 3 {
        return None;
    }
    cumulus3d_align::umeyama(&src, &dst, true).map(|t| t.scale)
}

/// ENU positions (origin = first GPS record) of the registered images of `rec` that have a GPS record.
fn enu_targets(rec: &Reconstruction, gps: &[GpsRecord]) -> HashMap<ImageId, cumulus3d_core::Vec3> {
    let Some(o) = gps.first() else { return HashMap::new() };
    let frame = cumulus3d_align::EnuFrame::new(o.lat, o.lon, o.alt);
    let by_name: HashMap<&str, &GpsRecord> = gps.iter().map(|g| (g.name.as_str(), g)).collect();
    rec.registered_images()
        .into_iter()
        .filter_map(|id| {
            let g = by_name.get(rec.image(id)?.name.as_str())?;
            Some((id, frame.lla_to_enu(g.lat, g.lon, g.alt)))
        })
        .collect()
}

/// Brings `rec` into the frame of the already shown model `shown` (Sim3 on the projection centers of the images registered
/// in both). Returns the Sim3 and, for each shared image, its pose in `shown`.
fn sim3_onto_shown(rec: &mut Reconstruction, shown: &Reconstruction) -> Option<(cumulus3d_core::Sim3, Vec<(ImageId, Rigid3)>)> {
    let mut pairs = Vec::new();
    for id in rec.registered_images() {
        let Some(name) = rec.image(id).map(|im| im.name.clone()) else { continue };
        let Some(sid) = shown.image_by_name(&name).map(|im| im.image_id) else { continue };
        if let (Some(a), Some(b), Some(pose)) = (rec.projection_center(id), shown.projection_center(sid), shown.world_to_cam(sid)) {
            pairs.push((id, a, b, pose));
        }
    }
    let (src, dst): (Vec<_>, Vec<_>) = pairs.iter().map(|(_, a, b, _)| (*a, *b)).unzip();
    let t = cumulus3d_align::umeyama(&src, &dst, true)?;
    rec.transform(&t);
    Some((t, pairs.into_iter().map(|(id, _, _, pose)| (id, pose)).collect()))
}

/// Stable display, settled: brings `rec` into the frame of the already shown model `shown`, then copies the shown poses and
/// the camera intrinsics exactly and marks them constant in `ba`. Returns (fixed image count, Sim3 scale).
fn anchor_to_shown(rec: &mut Reconstruction, shown: &Reconstruction, ba: &mut BaConfig) -> (usize, f64) {
    let Some((t, poses)) = sim3_onto_shown(rec, shown) else { return (0, 1.0) };
    for (id, pose) in &poses {
        if rec.set_world_to_cam(*id, *pose).is_ok() {
            ba.constant_poses.insert(*id);
        }
    }
    for (cid, cam) in shown.cameras() {
        if let Some(c) = rec.camera_mut(*cid) {
            *c = cam.clone();
            ba.constant_cameras.insert(*cid);
        }
    }
    (ba.constant_poses.len(), t.scale)
}

/// Fewest shared 3D points for which a point-based Sim3 fit is trusted over the camera-center fit.
const MIN_SHARED_PAIRS: usize = 20;

/// Sim3 taking `from` into the frame of `to` (umeyama with scale on the projection centers of the images, accepted by
/// `keep`, registered in both). None with fewer than 3 common images.
fn sim3_between(from: &Reconstruction, to: &Reconstruction, keep: impl Fn(&str) -> bool) -> Option<cumulus3d_core::Sim3> {
    let (mut src, mut dst) = (Vec::new(), Vec::new());
    for id in from.registered_images() {
        let Some(name) = from.image(id).map(|im| im.name.as_str()).filter(|n| keep(n)) else { continue };
        let Some(tid) = to.image_by_name(name).map(|im| im.image_id) else { continue };
        if let (Some(a), Some(b)) = (from.projection_center(id), to.projection_center(tid)) {
            src.push(a);
            dst.push(b);
        }
    }
    if src.len() < 3 {
        return None;
    }
    cumulus3d_align::umeyama(&src, &dst, true)
}

/// Largest relative change of a focal parameter (fx, fy; just f for single-focal models) over the cameras present in both
/// `a` and `b`, as a fraction of `a`'s value. Infinite when a camera changed its model kind; 0 when no camera is shared.
pub fn intrinsics_change(a: &BTreeMap<CameraId, Camera>, b: &BTreeMap<CameraId, Camera>) -> f64 {
    let mut worst = 0.0f64;
    for (id, ca) in a {
        let Some(cb) = b.get(id) else { continue };
        if ca.model != cb.model {
            return f64::INFINITY;
        }
        for &s in ca.model.focal_slots() {
            let (fa, fb) = (ca.params[s], cb.params[s]);
            worst = worst.max(if fa != 0.0 {
                ((fb - fa) / fa).abs()
            } else if fb != 0.0 {
                f64::INFINITY
            } else {
                0.0
            });
        }
    }
    worst
}

/// Pairing rule: same camera at position gaps 1..5, 8, 16 / different cameras at position difference 0..4. Only pairs involving an image
/// of position p, as sorted name pairs.
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

/// Zones: start = 0, span, ...; [start-ovl, start+span+ovl) intersect [0, total). Completion position = hi-1.
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

/// Zones closed at position p (if the total count is unknown, only zones that are not truncated).
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
        // Collecting the zones closed at each position gives the same list.
        let mut all = Vec::new();
        for p in 0..26 {
            all.extend(zones_ending_at(p, 12, 2, Some(26)));
        }
        all.sort_by_key(|r| r.zone);
        assert_eq!(z(all), z(zones_upto(26, 12, 2)));
        // If the total count is unknown, only non-truncated zones are closed at a position.
        assert_eq!(z(zones_ending_at(13, 12, 2, None)), vec![(0, 0, 14)]);
        assert_eq!(z(zones_ending_at(25, 12, 2, None)), vec![(1, 10, 26)]);
        assert!(zones_ending_at(25, 12, 2, Some(40)).iter().any(|r| r.zone == 1));
    }

    #[test]
    fn intrinsics_change_is_max_relative_focal_change() {
        use cumulus3d_core::CameraModelKind;
        let cam = |id: CameraId, model: CameraModelKind, params: Vec<f64>| Camera::new(id, model, 4000, 3000, params).unwrap();
        let single = |id, f: f64| cam(id, CameraModelKind::SingleFocalRadial, vec![f, 2000.0, 1500.0, 0.1]);
        let pinhole = |id, fx: f64, fy: f64| cam(id, CameraModelKind::Pinhole, vec![fx, fy, 2000.0, 1500.0]);
        let map = |v: Vec<Camera>| v.into_iter().map(|c| (c.camera_id, c)).collect::<BTreeMap<_, _>>();
        let a = map(vec![single(1, 1000.0), pinhole(2, 2000.0, 2000.0)]);
        assert_eq!(intrinsics_change(&a, &a), 0.0);
        // Principal point / distortion changes do not count; the worst focal parameter of any camera does.
        let b = map(vec![single(1, 1005.0), pinhole(2, 2000.0, 2040.0), single(3, 1.0)]);
        assert!((intrinsics_change(&a, &b) - 0.02).abs() < 1e-12);
        let c = map(vec![single(1, 990.0), pinhole(2, 2000.0, 2000.0)]);
        assert!((intrinsics_change(&a, &c) - 0.01).abs() < 1e-12);
        // No shared camera: nothing changed; model kind change: infinite.
        assert_eq!(intrinsics_change(&a, &map(vec![single(9, 5.0)])), 0.0);
        assert!(intrinsics_change(&a, &map(vec![pinhole(1, 1000.0, 1000.0)])).is_infinite());
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
