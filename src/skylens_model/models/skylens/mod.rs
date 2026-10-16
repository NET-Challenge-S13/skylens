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
//! - [`replay`]: recorded cumulus3d display streams fed back as events (tools, no GPU).
//! - [`slotstream`]: the display deltas of stable display as point slot chunks (pointblitz
//!   slot messages).
//! - [`jobs`] and [`sources`]: the REST/gRPC job API (`steps` -> level, `uri`
//!   decoding).
//!
//! Coordinates. Every chunk and detection is in ENU metres about
//! `EngineConfig::anchor` (x = East, y = North, z = Up; cumulus3d's ENU is the
//! WGS84 tangent plane at the anchor). Detections carry GPS converted with
//! `skylens_shared::geo::enu_to_gps`.
//!
//! Stable display (`EngineConfig::stable_display`, the default). The live node
//! runs cumulus3d's stable display with its in-memory display deltas
//! (`NodeBuilder::display_deltas`): no zone previews, and a zone keeps its place
//! once shown (while the intrinsics settle, up to zone [`SETTLE_MAX_ZONE`], the
//! earlier zones are placed again). cumulus3d composes the display (thinning,
//! overlap ownership, dense slot ids) and the consumer writes each
//! `Event::DisplayDelta` as one point slot chunk ([`slotstream`]); the mesh is
//! built from that composed display, so overlapping zones give one surface.
//! Detections come from the refined zone events. Without densification (no CUDA)
//! there are no deltas, so the refined zone points go out as point chunks.
//!
//! The reconstruction settings equal the V100 live runs (`cumulus3d stream`):
//! see [`jobs::live_plan`] and [`jobs::dense_for`].

pub mod chunks;
pub mod frames;
pub mod fusion;
pub mod jobs;
pub mod live;
pub mod recon;
pub mod replay;
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
use cumulus3d_cli::post::LiveOptions;
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
/// Stable display: the intrinsics count as settled once the focal length of two
/// refined zones in a row differs by less than this (`--settle-tol`, cumulus3d's default, 1 %).
pub const SETTLE_TOL: f64 = 0.01;
/// Stable display: the display settles at this zone at the latest (`--settle-max-zone`,
/// cumulus3d's default). Zones up to it may be placed again while the intrinsics settle.
pub const SETTLE_MAX_ZONE: usize = 3;
/// RANSAC seed of the live node (the V100 runs used `--seed 1`).
pub const LIVE_SEED: u64 = 1;
/// Live display thinning: 1 of this many points (the V100 live runs' `--live-stride 6`, also
/// cumulus3d's default; the decision 0052 measurements use it).
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
    /// cumulus3d stable display with display deltas (point slot chunks).
    /// Off: zone previews and refined zones as point chunks per segment.
    pub stable_display: bool,
    /// Live display thinning (1 of this many points, 1 = every point).
    pub live_stride: usize,
    /// Camera intrinsics estimate (`data/demo/camera.json`) for the intrinsics prior of the live
    /// node; `None` or a missing file means no prior.
    pub camera_json: Option<std::path::PathBuf>,
    /// Width of the pushed frames; the prior is scaled to it (`None` = the file's `video_width`).
    pub frame_width: Option<u32>,
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
    /// Stream id of this model run (see [`Engine::stream_id`]).
    stream: u32,
    /// Sender into the consumer thread for [`Engine::inject_event`]; dropped in `Drop` so the
    /// consumer ends with the engine.
    inject: Mutex<Option<std::sync::mpsc::Sender<live::Msg>>>,
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
        // One id per model run, higher than any earlier run's: it is the slot stream id and sets the
        // chunk sequence base, so a core that resumes after a restart of this server gets every chunk
        // of the new run and the screens start the new mission from an empty point buffer.
        let stream = slotstream::next_stream_id(std::time::SystemTime::now());
        let log = chunks::ChunkLog::starting_after(slotstream::sequence_base(stream));
        // Display deltas need dense clouds with point keys.
        let dense = jobs::dense_for(cfg.gpu);
        let deltas = cfg.stable_display && dense.is_some();
        let mut b = ReconNode::declare()
            // The plan comes first (it replaces every setting); the builder calls below refine it.
            .plan(jobs::live_plan())
            .settle(SETTLE_TOL, SETTLE_MAX_ZONE)
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
            .incremental_triangulation(true)
            // Same RANSAC seed as the V100 live runs (`--seed 1`).
            .seed(LIVE_SEED);
        if cfg.stable_display {
            ensure!(cfg.live_stride >= 1, "live stride must be at least 1");
            // Settle-then-freeze display: shown zones are only refined, never jump.
            b = b.stable_display(true);
        }
        if let Some(path) = &cfg.camera_json {
            match jobs::intrinsics_prior(path, &cfg.cameras, cfg.frame_width) {
                Ok(Some(p)) => {
                    tracing::info!("intrinsics prior from {} for {:?}", path.display(), cfg.cameras);
                    b = b.intrinsics_prior(p);
                }
                Ok(None) => tracing::info!("no {}: the live node runs without an intrinsics prior", path.display()),
                Err(e) => tracing::warn!("{}: {e:#}; the live node runs without an intrinsics prior", path.display()),
            }
        }
        if deltas {
            b = b.display_deltas(LiveOptions { stride: cfg.live_stride, ..Default::default() });
        }
        if cfg.gpu {
            b = b.gpu();
        }
        b = match dense {
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
            // With densification the points leave as display deltas; without it there are none (and
            // the refined zone points go out as point chunks). The consumer also takes deltas that
            // `inject_event` hands it (recorded streams), so it is set up either way.
            tracing::info!("live display: stable, 1 of {} points, slot stream {stream}, deltas {deltas}", cfg.live_stride);
            consumer = consumer.display_stream(stream);
        }
        consumer.spawn(rx)?;
        if let (Some(q), Some(det)) = (&frame_queue, &detector) {
            live::spawn_detector(Arc::clone(q), Arc::clone(det), tx.clone())?;
        }
        live::spawn_forwarder(events, tx.clone())?;
        let stop = Arc::new(AtomicBool::new(false));
        let poller = live::spawn_poller(node.clone(), Arc::clone(&stop))?;

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
            stream,
            inject: Mutex::new(Some(tx)),
            job_lock: Mutex::new(()),
        })
    }

    /// Stream id of this model run: the `generation` of its point slot chunks, and the upper 32 bits
    /// of every chunk sequence.
    pub fn stream_id(&self) -> u32 {
        self.stream
    }

    /// Warns once when pushed frames carry no GPS. With stable display the zone clouds are placed in
    /// ENU only through the per-image GPS fixes (the anchor record alone pins the origin, not the
    /// orientation), so without them the display delta frame is the reconstruction's own, not ENU,
    /// although cumulus3d still labels it ENU about the anchor.
    fn note_gps(&self, set: &FrameSet) {
        use std::sync::atomic::AtomicU8;
        // 0 = nothing seen, 1 = warned (no GPS), 2 = GPS seen.
        static SEEN: AtomicU8 = AtomicU8::new(0);
        let has = set.images.iter().any(|i| i.gps.is_some());
        if has {
            SEEN.store(2, Ordering::Relaxed);
        } else if SEEN.compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
            tracing::warn!(
                "frame set {:?} carries no GPS: the reconstruction cannot be placed in ENU (the points will be \
                 rotated and scaled arbitrarily on the board); check that the core sends the drone GPS with the frames",
                set.key
            );
        }
    }

    /// Hook for replay tools and tests: hands a cumulus3d event straight to the chunk consumer, as
    /// if the live node had emitted it (see [`replay`]). Fails once the engine is shutting down.
    pub fn inject_event(&self, ev: cumulus3d_cli::events::Event) -> anyhow::Result<()> {
        let guard = lock(&self.inject);
        let tx = guard.as_ref().context("engine is shut down")?;
        tx.send(live::Msg::Event(ev)).map_err(|_| anyhow!("chunk consumer has stopped"))
    }

    /// Declare one frame set (gRPC PushFrames).
    ///
    /// Blocks while cumulus3d ingests the position (features, matching,
    /// registration, and the zone preview when a zone closes); call it from a
    /// blocking context. Cameras may arrive in separate pushes with the same key:
    /// the set is declared once all of them are there.
    pub fn push(&self, set: FrameSet) -> anyhow::Result<PushOutcome> {
        self.note_gps(&set);
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
        lock(&self.inject).take();
        if let Some(h) = lock(&self.poller).take() {
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
            camera_json: None,
            frame_width: None,
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
        // The run's stream id sets the chunk sequence base.
        assert_eq!(engine.log.last_sequence(), slotstream::sequence_base(engine.stream_id()));
        drop(engine);
        let _ = std::fs::remove_dir_all(&cfg.work_dir);
    }

    #[test]
    fn a_restarted_engine_gets_a_higher_stream_and_sequences() {
        let cfg = engine_config("restart");
        let mut last = (0u32, 0u64);
        for _ in 0..3 {
            // Restarts within the same second: the id still grows.
            let engine = Engine::new(cfg.clone(), None).unwrap();
            let (id, base) = (engine.stream_id(), engine.log.last_sequence());
            assert!(id > last.0 && base > last.1 + u64::from(u32::MAX), "{id} {base} after {last:?}");
            last = (id, base);
            drop(engine);
        }
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
