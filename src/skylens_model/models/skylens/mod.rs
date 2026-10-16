//! Live reconstruction engine (replaces the 3DGS pipeline that lived here).
//!
//! cumulus3d `CompositionHost` grows the point cloud from live frame sets, the
//! UNet runs as the `on_frame` hook, deltamesh turns zone clouds into mesh
//! blocks, and everything leaves as `Chunk`s.
//!
//! The signatures below are the contract `app.rs` codes against.
//!
//! - [`frames`]: proto `FrameSet` -> `LiveFrameSet`, per-camera assembly, the
//!   anchor record that pins the ENU origin.
//! - [`live`]: the threads behind one live node (detector worker, event
//!   forwarder, chunk consumer, refinement poller).
//! - [`fusion`]: projection of frame detections onto the zone points and the
//!   confidence-weighted fusion across frames (pure math).
//! - [`chunks`]: pointblitz/MeshBlock/DetectionBatch encoding and the replay log.
//! - [`slotstream`]: the follower of the live diff stream (stable display): each event of
//!   `<run>/live/stream/` becomes one point slot chunk.
//! - [`jobs`] and [`sources`]: the REST/gRPC job API (`steps` -> level, `uri`
//!   decoding).
//!
//! Coordinates. Every chunk and detection is in ENU metres about
//! `EngineConfig::anchor` (x = East, y = North, z = Up; cumulus3d's ENU is the
//! WGS84 tangent plane at the anchor). Detections carry GPS converted with
//! `skylens_shared::geo::enu_to_gps`.
//!
//! Stable display (`EngineConfig::stable_display`, the default). The live node
//! runs cumulus3d's stable display: no zone previews, and a zone keeps its place
//! once shown (while the intrinsics settle, up to zone [`SETTLE_MAX_ZONE`], the
//! earlier zones are placed again). The points leave as the live diff stream
//! (point slot chunks, written by the node's file sinks under
//! `<work_dir>/runs/<run>/` and followed by [`slotstream`]); mesh blocks and
//! detections still come from the refined zone events. Without densification
//! (no CUDA) the stream stays empty, so the refined zone points go out as point
//! chunks as before.

pub mod chunks;
pub mod frames;
pub mod fusion;
pub mod jobs;
pub mod live;
pub mod recon;
pub mod slotstream;
pub mod sources;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use anyhow::{Context, anyhow, ensure};
use cumulus3d_cli::compose::{
    CompositionHost, FrameKey as LiveKey, FrameState, Outcome, ReconNode, ReconcilePolicy,
};
use cumulus3d_cli::declare::AlignFrame;
use cumulus3d_cli::sinks::SinkOptions;
use prost::Message;
use skylens_shared::proto::model::v1::{
    Align, Chunk, DetectJobRequest, DetectJobResult, FrameSet, Level, ReconJobRequest, ReconJobResult,
};

use crate::detect::Detector;

use frames::{Assembled, Assembler};
use fusion::Fuser;
use live::{Consumer, FrameQueue, Msg};

/// Id of the live node on the engine's host.
pub const LIVE_NODE: &str = "live";
/// Folder under `work_dir` with job results; `ReconJobResult.url` is
/// `/<RESULTS_DIR>/<file>`, so the HTTP server serves this folder at that path.
pub const RESULTS_DIR: &str = "results";
/// Folder under `work_dir` with one output folder per live run (stable display:
/// run.log, zone PLYs, the live display and its diff stream `live/stream/`).
pub const RUNS_DIR: &str = "runs";
/// Live run folders kept under `RUNS_DIR`, the new one included; older ones are removed at
/// engine start.
pub const KEEP_RUNS: usize = 3;
/// Stable display: the intrinsics count as settled once the focal length of two
/// refined zones in a row differs by less than this (cumulus3d's default, 1 %).
pub const SETTLE_TOL: f64 = 0.01;
/// Stable display: the display settles at this zone at the latest (cumulus3d's
/// default). Zones up to it may be placed again while the intrinsics settle.
pub const SETTLE_MAX_ZONE: usize = 3;
/// Live display thinning of the diff stream: 1 of this many points (cumulus3d's
/// default; the decision 0052 measurements use it).
pub const DEFAULT_LIVE_STRIDE: usize = 6;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// ENU origin; every zone and chunk is expressed about it.
    pub anchor: skylens_shared::geo::GeoAnchor,
    /// Camera names in a frame set, e.g. ["left", "center", "right"].
    pub cameras: Vec<String>,
    /// cumulus3d zone span and overlap, in frame sets.
    pub zone_span: usize,
    pub zone_overlap: usize,
    /// Working directory for cumulus3d (models, snapshots).
    pub work_dir: std::path::PathBuf,
    /// Use the GPU backends of cumulus3d and deltamesh.
    pub gpu: bool,
    /// cumulus3d stable display with the live diff stream (point slot chunks).
    /// Off: zone previews and refined zones as point chunks per segment.
    pub stable_display: bool,
    /// Live display thinning of the diff stream (1 = every point).
    pub live_stride: usize,
}

/// Result of declaring one frame set to the reconstruction node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushOutcome {
    /// New set, ingested once.
    Ingested,
    /// Already accepted or superseded; nothing happened.
    Ignored,
    /// Waiting for cameras that have not arrived yet.
    Pending,
}

impl PushOutcome {
    fn from_compose(o: &Outcome) -> Self {
        match o {
            Outcome::Appended { .. } | Outcome::Replayed { .. } | Outcome::Invalidated { .. } => PushOutcome::Ingested,
            Outcome::Waiting { .. } => PushOutcome::Pending,
            Outcome::Duplicate { .. } | Outcome::Ignored(_) | Outcome::Ended => PushOutcome::Ignored,
        }
    }
}

pub struct Engine {
    cfg: EngineConfig,
    /// Keeps the live node registered for the engine's lifetime.
    _host: CompositionHost,
    node: ReconNode,
    detector: Option<Arc<dyn Detector>>,
    assembler: Mutex<Assembler>,
    log: Arc<chunks::ChunkLog>,
    frame_queue: Option<Arc<FrameQueue>>,
    stop: Arc<AtomicBool>,
    poller: Mutex<Option<JoinHandle<()>>>,
    /// Follower of the live diff stream (stable display).
    follower: Mutex<Option<JoinHandle<()>>>,
    /// Output folder of this run (stable display).
    run_dir: Option<std::path::PathBuf>,
    /// Jobs run one at a time.
    job_lock: Mutex<()>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

impl Engine {
    pub fn new(cfg: EngineConfig, detector: Option<Arc<dyn Detector>>) -> anyhow::Result<Self> {
        ensure!(!cfg.cameras.is_empty(), "engine needs at least one camera");
        let live_dir = cfg.work_dir.join("live");
        std::fs::create_dir_all(&live_dir).with_context(|| format!("create {}", live_dir.display()))?;

        let host = CompositionHost::new();
        // One id per model run, higher than any earlier run's: it names the run folder, is the slot
        // stream id and sets the chunk sequence base, so a core that resumes after a restart of this
        // server gets every chunk of the new run.
        let runs = cfg.work_dir.join(RUNS_DIR);
        let stream = slotstream::next_stream_id(&runs, std::time::SystemTime::now());
        let log = chunks::ChunkLog::starting_after(slotstream::sequence_base(stream));
        let run_dir = cfg.stable_display.then(|| runs.join(slotstream::run_folder_name(stream, std::process::id())));
        if run_dir.is_some() {
            let removed = slotstream::prune_runs(&runs, KEEP_RUNS - 1);
            if removed > 0 {
                tracing::info!("removed {removed} old live run folder(s) under {}", runs.display());
            }
        }
        let mut b = ReconNode::declare()
            .id(LIVE_NODE)
            .images(&live_dir)
            .cameras(cfg.cameras.clone())
            .zones(cfg.zone_span, cfg.zone_overlap)
            .align(AlignFrame::Enu)
            .fixed_enu_origin(true)
            // A refined zone can be meshed only once (deltamesh), and late or corrected
            // sets would rewind zones already streamed; append only.
            .reconcile(ReconcilePolicy::AppendOnly)
            // Triangulate only the images registered at this position. The default re-triangulates
            // every registered image on each set, which grew to 7.5 s per set in a 5-minute run.
            .incremental_triangulation(true);
        if let Some(dir) = &run_dir {
            ensure!(cfg.live_stride >= 1, "live stride must be at least 1");
            // The diff stream is written by the node's file sinks; its origin is the anchor
            // (the first GPS record of every set, frames::ANCHOR_RECORD).
            b = b
                .stable_display(true)
                .settle(SETTLE_TOL, SETTLE_MAX_ZONE)
                .live_stride(cfg.live_stride)
                .stream_origin(cfg.anchor.lat, cfg.anchor.lon, cfg.anchor.alt)
                .output(dir, SinkOptions::default())
                // Only run.log, the timeline and the diff stream: no zone, display or model files
                // (about 0.8 GB per run otherwise, written before each stream pack).
                .stream_only(true);
        }
        if cfg.gpu {
            b = b.gpu();
        }
        b = match jobs::dense_for(cfg.gpu) {
            Some(d) => b.dense(d),
            None => b.no_dense(),
        };
        let frame_queue = detector.as_ref().map(|_| FrameQueue::new());
        if let Some(q) = &frame_queue {
            let q = Arc::clone(q);
            // The hook only enqueues; the UNet runs on its own worker.
            b = b.on_frame(move |f| q.push(Arc::clone(f)));
        }
        let node = b.build(&host).map_err(|e| anyhow!("live node: {e}"))?;
        let events = node.subscribe().context("live node closed before start")?;

        let (tx, rx) = std::sync::mpsc::channel::<Msg>();
        let mut consumer = Consumer::new(Arc::clone(&log), cfg.anchor, cfg.gpu, detector.is_some());
        if cfg.stable_display {
            // With densification the points leave through the diff stream; without it the stream
            // stays empty and the refined zone points go out as point chunks.
            consumer = consumer.stable_display(SETTLE_MAX_ZONE, jobs::dense_for(cfg.gpu).is_some());
        }
        consumer.spawn(rx)?;
        if let (Some(q), Some(det)) = (&frame_queue, &detector) {
            live::spawn_detector(Arc::clone(q), Arc::clone(det), tx.clone())?;
        }
        live::spawn_forwarder(events, tx)?;
        let stop = Arc::new(AtomicBool::new(false));
        let poller = live::spawn_poller(node.clone(), Arc::clone(&stop))?;
        let follower = match &run_dir {
            Some(dir) => {
                tracing::info!("live run output in {} (stream id {stream})", dir.display());
                let log = Arc::clone(&log);
                let publish = move |chunks: Vec<Chunk>| {
                    log.publish(chunks);
                };
                Some(slotstream::spawn(slotstream::stream_dir(dir), stream, cfg.anchor, publish, Arc::clone(&stop))?)
            }
            None => None,
        };

        Ok(Self {
            assembler: Mutex::new(Assembler::new(cfg.cameras.clone())),
            cfg,
            _host: host,
            node,
            detector,
            log,
            frame_queue,
            stop,
            poller: Mutex::new(Some(poller)),
            follower: Mutex::new(follower),
            run_dir,
            job_lock: Mutex::new(()),
        })
    }

    /// Output folder of this live run (stable display), `<work_dir>/runs/live-<stream id>-<pid>`.
    pub fn run_dir(&self) -> Option<&std::path::Path> {
        self.run_dir.as_deref()
    }

    /// Declare one frame set (gRPC PushFrames).
    ///
    /// Blocks while cumulus3d ingests the position (features, matching,
    /// registration, and the zone preview when a zone closes); call it from a
    /// blocking context. Cameras may arrive in separate pushes with the same key:
    /// the set is declared once all of them are there.
    pub fn push(&self, set: FrameSet) -> anyhow::Result<PushOutcome> {
        let assembled = lock(&self.assembler).offer(set)?;
        let state = match assembled {
            Assembled::Repeat => return Ok(PushOutcome::Ignored),
            Assembled::Pending { key, missing } => {
                FrameState::pending(LiveKey::new(key.source_id, key.sequence, key.revision), missing)
            }
            Assembled::Complete(set) => FrameState::ready(frames::to_live_set(&set, &self.cfg.cameras, self.cfg.anchor)?),
        };
        let outcome = self.node.compose(state).map_err(|e| anyhow!("compose: {e}"))?;
        Ok(PushOutcome::from_compose(&outcome))
    }

    /// Chunks with a sequence above `after_sequence`, then every new one as it is
    /// produced (gRPC SubscribeChunks). Already-produced chunks are replayed first.
    pub fn subscribe(&self, after_sequence: u64) -> tokio::sync::mpsc::Receiver<Chunk> {
        self.log.subscribe(after_sequence)
    }

    /// REST/gRPC recon job. `steps` selects the level (preview or refined).
    /// `progress` receives 0.0..=1.0.
    ///
    /// Levels: `steps < jobs::REFINED_MIN_STEPS` is the preview, otherwise the
    /// refined cloud. The cloud is written to `<work_dir>/results/` as PLY and
    /// `url` points at `/results/<file>`. The cloud is already in ENU about the
    /// anchor, so `align` is the identity at the anchor.
    pub fn run_recon_job(
        &self,
        req: &ReconJobRequest,
        progress: &(dyn Fn(f64) + Send + Sync),
    ) -> anyhow::Result<ReconJobResult> {
        let _one = lock(&self.job_lock);
        let level = jobs::level_for_steps(req.steps);
        let dir = jobs::job_dir(&self.cfg.work_dir, "recon", req.segment);
        let result: anyhow::Result<ReconJobResult> = (|| {
            let out = jobs::run(&self.cfg, &dir, &req.sources, &req.track, level, None, progress)?;
            let suffix = if level == Level::Refined { "refined" } else { "preview" };
            let file = format!("segment-{:03}-{suffix}.ply", req.segment);
            let bytes = jobs::write_cloud(&self.cfg.work_dir.join(RESULTS_DIR).join(&file), &out)?;
            Ok(ReconJobResult {
                segment: req.segment,
                steps: req.steps,
                url: format!("/{RESULTS_DIR}/{file}"),
                bytes,
                align: Some(self.identity_align()),
                anchor_frame: req.anchor_frame.clone().unwrap_or_else(|| self.anchor_frame()),
            })
        })();
        let _ = std::fs::remove_dir_all(&dir);
        if result.is_ok() {
            progress(1.0);
        }
        result
    }

    /// REST/gRPC detect job over the segment's sources.
    ///
    /// Detections need camera poses and surface points, so the job reconstructs
    /// the segment at preview level with the UNet on the frame hook and fuses the
    /// result exactly like the live path. `detections` is an encoded
    /// `payload.v1.DetectionBatch`.
    pub fn run_detect_job(
        &self,
        req: &DetectJobRequest,
        progress: &(dyn Fn(f64) + Send + Sync),
    ) -> anyhow::Result<DetectJobResult> {
        let detector = self.detector.clone().context("detect job: no UNet loaded")?;
        let _one = lock(&self.job_lock);
        let dir = jobs::job_dir(&self.cfg.work_dir, "detect", req.segment);
        let out = jobs::run(&self.cfg, &dir, &req.sources, &[], Level::Preview, Some(detector), progress);
        let _ = std::fs::remove_dir_all(&dir);
        let out = out?;
        let mut fuser = Fuser::new(self.cfg.anchor, out.observations.len().max(1));
        fuser.set_zone(req.segment, &recon::positions_f64(&out.points), out.views);
        for (name, obs) in out.observations {
            fuser.add_observation(&name, obs);
        }
        let batch = fuser.batch(req.segment);
        progress(1.0);
        Ok(DetectJobResult { segment: req.segment, detections: batch.encode_to_vec() })
    }

    /// Clouds are produced in ENU about the anchor: identity transform.
    fn identity_align(&self) -> Align {
        Align {
            anchor: Some(self.cfg.anchor.into()),
            position: vec![0.0, 0.0, 0.0],
            rotation: vec![0.0, 0.0, 0.0, 1.0],
            scale: vec![1.0, 1.0, 1.0],
        }
    }

    /// The coordinate frame is fixed by the anchor, so its id is derived from it.
    fn anchor_frame(&self) -> String {
        let a = self.cfg.anchor;
        format!("enu-{:.7}-{:.7}-{:.2}", a.lat, a.lon, a.alt)
    }
}

impl Drop for Engine {
    /// Stops the poller and the detector worker and ends every subscription. The
    /// node itself is dropped with the engine without waiting for background
    /// refinement; the forwarder and consumer threads end once its event channel
    /// closes.
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(q) = &self.frame_queue {
            q.close();
        }
        self.log.close();
        if let Some(h) = lock(&self.poller).take() {
            let _ = h.join();
        }
        if let Some(h) = lock(&self.follower).take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cumulus3d_cli::compose::IgnoreReason;

    #[test]
    fn compose_outcomes_map_to_push_outcomes() {
        assert_eq!(PushOutcome::from_compose(&Outcome::Appended { position: 3, skipped: 0 }), PushOutcome::Ingested);
        assert_eq!(PushOutcome::from_compose(&Outcome::Replayed { from: 1, positions: 2 }), PushOutcome::Ingested);
        assert_eq!(PushOutcome::from_compose(&Outcome::Waiting { missing: vec!["left".into()] }), PushOutcome::Pending);
        assert_eq!(PushOutcome::from_compose(&Outcome::Duplicate { position: 0 }), PushOutcome::Ignored);
        assert_eq!(PushOutcome::from_compose(&Outcome::Ignored(IgnoreReason::Late)), PushOutcome::Ignored);
        assert_eq!(PushOutcome::from_compose(&Outcome::Ended), PushOutcome::Ignored);
    }

    fn engine_config(dir: &str) -> EngineConfig {
        EngineConfig {
            anchor: skylens_shared::geo::Gps { lat: 36.3685, lon: 127.3475, alt: 30.0 },
            cameras: ["left", "center", "right"].map(String::from).to_vec(),
            zone_span: 12,
            zone_overlap: 2,
            work_dir: std::env::temp_dir().join(format!("skylens-engine-{dir}-{}", std::process::id())),
            gpu: false,
            stable_display: true,
            live_stride: DEFAULT_LIVE_STRIDE,
        }
    }

    #[test]
    fn engine_starts_assembles_and_replays_nothing() {
        use skylens_shared::proto::model::v1::{FrameKey, ImageFormat, LiveImage};
        let cfg = engine_config("start");
        let engine = Engine::new(cfg.clone(), None).unwrap();
        let partial = FrameSet {
            key: Some(FrameKey { source_id: "formation".into(), sequence: 0, revision: 0 }),
            images: vec![LiveImage {
                camera: "left".into(),
                format: ImageFormat::Jpeg as i32,
                data: vec![0xFF, 0xD8, 0xFF],
                ..Default::default()
            }],
            captured_at_ms: 0,
        };
        assert_eq!(engine.push(partial).unwrap(), PushOutcome::Pending);
        let bad = FrameSet { key: None, images: vec![], captured_at_ms: 0 };
        assert!(engine.push(bad).is_err());
        assert!(engine.log.since(0).is_empty());
        assert_eq!(engine.anchor_frame(), "enu-36.3685000-127.3475000-30.00");
        // Stable display: the run has its own output folder under runs/.
        let run = engine.run_dir().expect("stable display gives a run folder").to_path_buf();
        assert!(run.starts_with(cfg.work_dir.join(RUNS_DIR)));
        // The run's stream id sets the chunk sequence base.
        let id = slotstream::run_folder_stream(run.file_name().unwrap().to_str().unwrap()).unwrap();
        assert_eq!(engine.log.last_sequence(), slotstream::sequence_base(id));
        drop(engine);
        let _ = std::fs::remove_dir_all(&cfg.work_dir);
    }

    #[test]
    fn a_restarted_engine_gets_a_higher_stream_and_sequences_and_keeps_few_runs() {
        let cfg = engine_config("restart");
        let _ = std::fs::remove_dir_all(&cfg.work_dir);
        let mut last = (0u32, 0u64);
        for _ in 0..KEEP_RUNS + 2 {
            // Restarts within the same second: the id still grows.
            let engine = Engine::new(cfg.clone(), None).unwrap();
            let run = engine.run_dir().unwrap().to_path_buf();
            let id = slotstream::run_folder_stream(run.file_name().unwrap().to_str().unwrap()).unwrap();
            let base = engine.log.last_sequence();
            assert!(id > last.0 && base > last.1 + u64::from(u32::MAX), "{id} {base} after {last:?}");
            last = (id, base);
            std::fs::create_dir_all(&run).unwrap();
            drop(engine);
        }
        let runs = std::fs::read_dir(cfg.work_dir.join(RUNS_DIR)).unwrap().count();
        assert_eq!(runs, KEEP_RUNS);
        let _ = std::fs::remove_dir_all(&cfg.work_dir);
    }

    #[test]
    fn without_stable_display_there_is_no_run_folder() {
        let cfg = EngineConfig { stable_display: false, ..engine_config("plain") };
        let engine = Engine::new(cfg.clone(), None).unwrap();
        assert!(engine.run_dir().is_none());
        drop(engine);
        let _ = std::fs::remove_dir_all(&cfg.work_dir);
    }

    #[test]
    fn detect_job_without_a_detector_is_an_error() {
        let cfg = engine_config("nodet");
        let engine = Engine::new(cfg.clone(), None).unwrap();
        let req = DetectJobRequest { segment: 0, sources: vec![] };
        assert!(engine.run_detect_job(&req, &|_| {}).is_err());
        drop(engine);
        let _ = std::fs::remove_dir_all(&cfg.work_dir);
    }
}
