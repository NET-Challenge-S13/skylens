//! Threads of the live path.
//!
//! ```text
//! compose (caller thread) ── cumulus3d pipeline ──┬─ on_frame lane ─> FrameQueue ─> detector worker ─┐
//!                                                 └─ subscription ──> forwarder ───────────────────────┤
//!                                                                                                     v
//!                                    consumer: points, deltamesh blocks, detection fusion ─> ChunkLog
//! poller: collects finished background refinement (cumulus3d only hands it out on push/poll)
//! ```
//!
//! Nothing on this side holds the node's lock, so reconstruction never waits on
//! the UNet, the mesher or a subscriber.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use cumulus3d_cli::compose::ReconNode;
use cumulus3d_cli::events::{Event, Frame, ZoneRange};
use cumulus3d_core::io::ply::PointCloud;
use cumulus3d_core::Reconstruction;
use deltamesh::Mesher;
use skylens_shared::geo::GeoAnchor;
use skylens_shared::proto::model::v1::Level;

use super::chunks::{ChunkLog, detections_chunk, mesh_chunk, points_delivery};
use super::fusion::{FrameObservation, Fuser};
use super::recon::{positions_f64, zone_points, zone_views};
use crate::detect::Detector;

/// Decoded frames waiting for the UNet. When the UNet falls behind, the oldest
/// frames are dropped: fusion interpolates across frames, so a gap only lowers
/// the frame count, while an unbounded queue of full-resolution RGB would not.
pub const DETECT_QUEUE_CAP: usize = 24;
/// Frame observations kept for re-fusion when a zone is rebuilt.
pub const MAX_OBSERVATIONS: usize = 4096;
/// How often finished background refinement is collected.
pub const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Input of the consumer thread.
pub enum Msg {
    Event(Event),
    Observation { name: String, obs: FrameObservation },
}

#[derive(Default)]
struct QueueState {
    frames: VecDeque<Arc<Frame>>,
    closed: bool,
}

/// Bounded hand-off from the frame hook to the detector worker.
#[derive(Default)]
pub struct FrameQueue {
    st: Mutex<QueueState>,
    cv: Condvar,
    dropped: AtomicU64,
}

impl FrameQueue {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Never blocks (called from the cumulus3d hook lane).
    pub fn push(&self, f: Arc<Frame>) {
        let mut st = self.st.lock().unwrap_or_else(|p| p.into_inner());
        if st.closed {
            return;
        }
        while st.frames.len() >= DETECT_QUEUE_CAP {
            st.frames.pop_front();
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        st.frames.push_back(f);
        drop(st);
        self.cv.notify_one();
    }

    fn pop(&self) -> Option<Arc<Frame>> {
        let mut st = self.st.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if let Some(f) = st.frames.pop_front() {
                return Some(f);
            }
            if st.closed {
                return None;
            }
            st = self.cv.wait(st).unwrap_or_else(|p| p.into_inner());
        }
    }

    pub fn close(&self) {
        let mut st = self.st.lock().unwrap_or_else(|p| p.into_inner());
        st.closed = true;
        st.frames.clear();
        drop(st);
        self.cv.notify_all();
    }

    /// Frames dropped because the UNet fell behind.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

fn spawn(name: &str, f: impl FnOnce() + Send + 'static) -> anyhow::Result<JoinHandle<()>> {
    Ok(std::thread::Builder::new().name(name.into()).spawn(f)?)
}

/// Runs the UNet on queued frames and hands the observations to the consumer.
pub fn spawn_detector(queue: Arc<FrameQueue>, det: Arc<dyn Detector>, tx: Sender<Msg>) -> anyhow::Result<JoinHandle<()>> {
    spawn("skylens-detect", move || {
        while let Some(f) = queue.pop() {
            match catch_unwind(AssertUnwindSafe(|| det.detect(&f.rgb, None, f.width, f.height))) {
                Ok(Ok(d)) => {
                    let obs = FrameObservation::from_detections(&d);
                    if tx.send(Msg::Observation { name: f.name.clone(), obs }).is_err() {
                        return;
                    }
                }
                Ok(Err(e)) => tracing::warn!("detector failed on {}: {e:#}", f.name),
                Err(_) => tracing::error!("detector panicked on {}", f.name),
            }
        }
    })
}

/// Moves pipeline events to the consumer, dropping decoded frames (the
/// detector gets them through its own queue).
pub fn spawn_forwarder(events: Receiver<Event>, tx: Sender<Msg>) -> anyhow::Result<JoinHandle<()>> {
    spawn("skylens-events", move || {
        for ev in events {
            if matches!(ev, Event::FrameDecoded { .. } | Event::Log { .. }) {
                continue;
            }
            if tx.send(Msg::Event(ev)).is_err() {
                return;
            }
        }
    })
}

/// Collects finished background refinement until `stop`.
pub fn spawn_poller(node: ReconNode, stop: Arc<AtomicBool>) -> anyhow::Result<JoinHandle<()>> {
    spawn("skylens-poll", move || {
        while !stop.load(Ordering::Relaxed) {
            std::thread::sleep(POLL_INTERVAL);
            node.poll();
        }
    })
}

/// Creates the mesher: deltamesh's GPU mesher when asked and built with the
/// `gpu-mesh` feature, the CPU `SdfMesher` otherwise (or when no GPU is found).
fn new_mesher(gpu: bool) -> Box<dyn Mesher> {
    #[cfg(feature = "gpu-mesh")]
    {
        if gpu {
            let made = deltamesh::gpu::GpuCtx::shared()
                .and_then(|ctx| deltamesh::gpu_mesher::GpuMesher::new(ctx, deltamesh::Config::default()));
            match made {
                Ok(m) => return Box::new(m),
                Err(e) => tracing::warn!("deltamesh GPU mesher unavailable ({e}); using the CPU mesher"),
            }
        }
    }
    #[cfg(not(feature = "gpu-mesh"))]
    {
        if gpu {
            tracing::info!("built without the gpu-mesh feature; deltamesh runs on the CPU");
        }
    }
    Box::new(deltamesh::sdf::SdfMesher::new(deltamesh::Config::default()))
}

/// State of the consumer thread.
pub struct Consumer {
    log: Arc<ChunkLog>,
    gpu: bool,
    detecting: bool,
    positions: HashMap<String, usize>,
    refined_generation: HashMap<u32, u32>,
    detection_version: HashMap<u32, u32>,
    fuser: Fuser,
}

impl Consumer {
    pub fn new(log: Arc<ChunkLog>, anchor: GeoAnchor, gpu: bool, detecting: bool) -> Self {
        Self {
            log,
            gpu,
            detecting,
            positions: HashMap::new(),
            refined_generation: HashMap::new(),
            detection_version: HashMap::new(),
            fuser: Fuser::new(anchor, MAX_OBSERVATIONS),
        }
    }

    /// Runs until every sender is gone. Batches what is already queued so a
    /// burst of observations yields one detection batch per segment.
    pub fn spawn(mut self, rx: Receiver<Msg>) -> anyhow::Result<JoinHandle<()>> {
        spawn("skylens-chunks", move || {
            let mut mesher = new_mesher(self.gpu);
            while let Ok(first) = rx.recv() {
                let mut dirty = BTreeSet::new();
                let mut next = Some(first);
                while let Some(msg) = next.take().or_else(|| rx.try_recv().ok()) {
                    let r = catch_unwind(AssertUnwindSafe(|| self.handle(msg, mesher.as_mut(), &mut dirty)));
                    if r.is_err() {
                        tracing::error!("chunk consumer panicked on a message; continuing");
                    }
                }
                self.flush_detections(dirty);
            }
        })
    }

    fn handle(&mut self, msg: Msg, mesher: &mut dyn Mesher, dirty: &mut BTreeSet<u32>) {
        match msg {
            Msg::Observation { name, obs } => {
                dirty.extend(self.fuser.add_observation(&name, obs));
            }
            Msg::Event(ev) => match ev {
                Event::FrameIngested { position, images, .. } => {
                    for im in images {
                        self.positions.insert(im, position);
                    }
                }
                Event::FrameRegistered { position, image, .. } => {
                    // Poses here are in the chain frame, which is not ENU until a refined
                    // zone is adopted; projection uses the zone model's poses instead.
                    self.positions.entry(image).or_insert(position);
                }
                Event::ZonePreview { range, cloud, model, frame, .. } => {
                    if frame == "unaligned" {
                        tracing::debug!("zone {} preview is not GPS-aligned yet; waiting for the refined zone", range.zone);
                        return;
                    }
                    self.zone(range, &cloud, &model, Level::Preview, mesher, dirty);
                }
                Event::ZoneRefined { range, cloud, model, .. } => {
                    self.zone(range, &cloud, &model, Level::Refined, mesher, dirty);
                }
                Event::Warning { message, .. } => tracing::warn!("recon: {message}"),
                Event::Error { message, .. } => tracing::error!("recon: {message}"),
                _ => {}
            },
        }
    }

    fn zone(
        &mut self,
        range: ZoneRange,
        cloud: &PointCloud,
        model: &Reconstruction,
        level: Level,
        mesher: &mut dyn Mesher,
        dirty: &mut BTreeSet<u32>,
    ) {
        let seg = range.zone as u32;
        let points = zone_points(cloud, model, range, &self.positions);
        if points.is_empty() {
            tracing::debug!("zone {seg} {level:?}: no points");
            return;
        }
        let generation = if level == Level::Refined {
            let g = self.refined_generation.entry(seg).or_insert(0);
            *g += 1;
            *g
        } else {
            0
        };
        self.log.publish(points_delivery(seg, level, generation, &points));

        let dm_level = if level == Level::Refined { deltamesh::Level::Refined } else { deltamesh::Level::Preview };
        let mpts: Vec<deltamesh::Point> =
            points.iter().map(|p| deltamesh::Point { pos: p.pos, rgb: p.rgb, normal: p.normal }).collect();
        match mesher.ingest(seg, dm_level, &mpts) {
            Ok(_) => {
                let changed = mesher.extract();
                let blocks: Vec<_> =
                    changed.iter().map(|id| mesh_chunk(seg, level, *id, mesher.version(id), mesher.mesh(id))).collect();
                self.log.publish(blocks);
            }
            Err(e) => tracing::warn!("deltamesh zone {seg}: {e}"),
        }

        if self.detecting {
            self.fuser.set_zone(seg, &positions_f64(&points), zone_views(model, range, &self.positions));
            dirty.insert(seg);
        }
    }

    fn flush_detections(&mut self, dirty: BTreeSet<u32>) {
        let mut out = Vec::with_capacity(dirty.len());
        for seg in dirty {
            let v = self.detection_version.entry(seg).or_insert(0);
            *v += 1;
            out.push(detections_chunk(seg, *v, &self.fuser.batch(seg)));
        }
        if !out.is_empty() {
            self.log.publish(out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cumulus3d_cli::events::Meta;
    use skylens_shared::proto::model::v1::ChunkKind;
    use std::time::SystemTime;

    const ANCHOR: GeoAnchor = skylens_shared::geo::Gps { lat: 36.3685, lon: 127.3475, alt: 30.0 };

    fn frame(i: usize) -> Arc<Frame> {
        Arc::new(Frame {
            position: i,
            camera: "left".into(),
            name: format!("left/{i:06}.png"),
            path: Default::default(),
            width: 1,
            height: 1,
            rgb: Arc::from(vec![0u8; 3]),
            gps: None,
        })
    }

    #[test]
    fn frame_queue_drops_the_oldest_beyond_its_cap() {
        let q = FrameQueue::new();
        for i in 0..DETECT_QUEUE_CAP + 5 {
            q.push(frame(i));
        }
        assert_eq!(q.dropped(), 5);
        assert_eq!(q.pop().unwrap().position, 5);
        q.close();
        assert!(q.pop().is_none());
        q.push(frame(99));
        assert!(q.pop().is_none(), "closed queue accepts nothing");
    }

    /// A flat 5 m x 5 m patch sampled every 5 cm (the deltamesh example).
    fn patch() -> Arc<PointCloud> {
        let mut c = PointCloud::default();
        for i in 0..100 {
            for j in 0..100 {
                c.positions.push([i as f32 * 0.05, j as f32 * 0.05, 0.0]);
                c.colors.push([180, 180, 180]);
            }
        }
        Arc::new(c)
    }

    fn zone_event(refined: bool, frame: &str) -> Msg {
        let meta = Meta { version: 1, at: SystemTime::now() };
        let range = ZoneRange { zone: 2, lo: 20, hi: 34 };
        let model = Arc::new(Reconstruction::new());
        Msg::Event(if refined {
            Event::ZoneRefined { meta, range, cloud: patch(), model, dense: true }
        } else {
            Event::ZonePreview { meta, range, cloud: patch(), model, dense: true, frame: frame.into() }
        })
    }

    #[test]
    fn zones_become_point_and_mesh_chunks() {
        let log = ChunkLog::new();
        let mut c = Consumer::new(log.clone(), ANCHOR, false, false);
        let mut mesher = new_mesher(false);
        let mut dirty = BTreeSet::new();

        c.handle(zone_event(false, "unaligned"), mesher.as_mut(), &mut dirty);
        assert_eq!(log.last_sequence(), 0, "unaligned previews are not published");

        c.handle(zone_event(false, "gps"), mesher.as_mut(), &mut dirty);
        let all = log.since(0);
        let points: Vec<_> = all.iter().filter(|x| x.kind == ChunkKind::Points as i32).collect();
        let blocks: Vec<_> = all.iter().filter(|x| x.kind == ChunkKind::MeshBlock as i32).collect();
        assert!(!points.is_empty());
        assert!(points.iter().all(|x| x.segment == 2 && x.level == Level::Preview as i32 && x.generation == 0));
        assert!(!blocks.is_empty(), "the patch meshes into at least one block");

        c.handle(zone_event(true, ""), mesher.as_mut(), &mut dirty);
        let points: Vec<_> =
            log.since(0).iter().filter(|x| x.kind == ChunkKind::Points as i32).map(|x| (x.level, x.generation)).collect();
        assert!(points.iter().all(|p| *p == (Level::Refined as i32, 1)), "{points:?}");
        assert!(dirty.is_empty(), "no detector, no detection batches");
    }
}
