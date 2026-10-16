//! Threads of the live path.
//!
//! ```text
//! compose (caller thread) ── cumulus3d pipeline ──┬─ on_frame lane ─> FrameQueue ─> detector worker ─┐
//!                                                 └─ subscription ──> forwarder ───────────────────────┤
//!                                                                                                     v
//!                                    consumer: points, deltamesh blocks, detection fusion ─> ChunkLog
//! poller: collects finished background refinement (cumulus3d only hands it out on push/poll)
//! slots follower (stable display): <run>/live/stream/ ─> point slot chunks ─> ChunkLog
//! ```
//!
//! Stable display. Zones arrive refined only (no previews). Their points leave through the diff
//! stream (see `slotstream`), so the consumer publishes point chunks only for zones the stream
//! does not carry: all of them when there is no densification, and a zone whose densification
//! failed (`ZoneRefined { dense: false }`, cumulus3d writes stream events for dense zones only).
//! A refined zone can be meshed once (deltamesh), but while the intrinsics settle cumulus3d places
//! the zones up to `settle_max_zone` again and emits them again; those zones are meshed at the
//! preview level (replaceable) until a zone past `settle_max_zone` arrives, which is the point
//! after which no shown zone moves. Then their latest points are meshed at the refined level. A
//! run that never gets past `settle_max_zone` (at most `settle_max_zone + 1` zones, 4 by default)
//! keeps those meshes at the preview level.
//!
//! Nothing on this side holds the node's lock, so reconstruction never waits on
//! the UNet, the mesher or a subscriber.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
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

use super::chunks::{ChunkLog, ZonePoint, detections_chunk, mesh_chunk, points_delivery};
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
            match &ev {
                Event::FrameDecoded { .. } => continue,
                // cumulus3d's run log: per-stage lines (features, matching, registration, dense).
                Event::Log { line, .. } => {
                    tracing::debug!(target: "cumulus3d", "{line}");
                    continue;
                }
                _ => {}
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

/// One zone result as the consumer handles it.
struct ZoneIn<'a> {
    range: ZoneRange,
    cloud: &'a PointCloud,
    model: &'a Reconstruction,
    level: Level,
    /// The zone's points are dense, so the diff stream carries them (stable display).
    dense: bool,
}

/// Stable display state of the consumer.
#[derive(Debug, Default)]
struct Stable {
    /// Zones up to this one may be placed again while the intrinsics settle.
    settle_max_zone: u32,
    /// The diff stream carries the points (densification on): no point chunks.
    stream_points: bool,
    /// Latest points of the zones meshed provisionally (preview level), until `committed`.
    provisional: BTreeMap<u32, Vec<ZonePoint>>,
    /// A zone past `settle_max_zone` arrived: no shown zone moves any more.
    committed: bool,
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
    stable: Option<Stable>,
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
            stable: None,
        }
    }

    /// cumulus3d stable display: zones up to `settle_max_zone` may come again while the
    /// intrinsics settle; `stream_points` = the diff stream carries the points.
    pub fn stable_display(mut self, settle_max_zone: usize, stream_points: bool) -> Self {
        self.stable =
            Some(Stable { settle_max_zone: settle_max_zone as u32, stream_points, provisional: BTreeMap::new(), committed: false });
        self
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
                    let z = ZoneIn { range, cloud: &cloud, model: &model, level: Level::Preview, dense: false };
                    self.zone(z, mesher, dirty);
                }
                Event::ZoneRefined { range, cloud, model, dense, .. } => {
                    let z = ZoneIn { range, cloud: &cloud, model: &model, level: Level::Refined, dense };
                    self.zone(z, mesher, dirty);
                }
                Event::Warning { message, .. } => tracing::warn!("recon: {message}"),
                Event::Error { message, .. } => tracing::error!("recon: {message}"),
                _ => {}
            },
        }
    }

    fn zone(&mut self, z: ZoneIn<'_>, mesher: &mut dyn Mesher, dirty: &mut BTreeSet<u32>) {
        let ZoneIn { range, cloud, model, level, dense } = z;
        let seg = range.zone as u32;
        let points = zone_points(cloud, model, range, &self.positions);
        if points.is_empty() {
            tracing::debug!("zone {seg} {level:?}: no points");
            return;
        }
        let streamed = dense && self.stable.as_ref().is_some_and(|s| s.stream_points);
        if !streamed {
            if self.stable.as_ref().is_some_and(|s| s.stream_points) {
                tracing::warn!("zone {seg}: densification failed; its sparse points go out as point chunks");
            }
            let generation = if level == Level::Refined {
                let g = self.refined_generation.entry(seg).or_insert(0);
                *g += 1;
                *g
            } else {
                0
            };
            self.log.publish(points_delivery(seg, level, generation, &points));
        }

        match self.stable.as_mut() {
            None => self.mesh_zone(seg, level, &points, mesher),
            Some(st) if !st.committed && seg <= st.settle_max_zone => {
                // May be placed again while the intrinsics settle: replaceable for now.
                st.provisional.insert(seg, points.clone());
                self.mesh_zone(seg, Level::Preview, &points, mesher);
            }
            Some(st) => {
                let ready = if st.committed {
                    Vec::new()
                } else {
                    st.committed = true;
                    std::mem::take(&mut st.provisional).into_iter().collect()
                };
                for (z, pts) in ready {
                    self.mesh_zone(z, Level::Refined, &pts, mesher);
                }
                self.mesh_zone(seg, Level::Refined, &points, mesher);
            }
        }

        if self.detecting {
            self.fuser.set_zone(seg, &positions_f64(&points), zone_views(model, range, &self.positions));
            dirty.insert(seg);
        }
    }

    /// Meshes one zone's points at `level` and publishes the blocks that changed.
    fn mesh_zone(&mut self, seg: u32, level: Level, points: &[ZonePoint], mesher: &mut dyn Mesher) {
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

    /// A flat 5 m x 5 m patch sampled every 5 cm (the deltamesh example), starting at x = `x0`.
    fn patch_at(x0: f32) -> Arc<PointCloud> {
        let mut c = PointCloud::default();
        for i in 0..100 {
            for j in 0..100 {
                c.positions.push([x0 + i as f32 * 0.05, j as f32 * 0.05, 0.0]);
                c.colors.push([180, 180, 180]);
            }
        }
        Arc::new(c)
    }

    fn zone_event(refined: bool, frame: &str) -> Msg {
        zone_event_at(2, refined, frame)
    }

    fn zone_event_at(zone: usize, refined: bool, frame: &str) -> Msg {
        zone_event_with(zone, refined, frame, true)
    }

    /// A zone event; each zone has its own patch (100 m apart), so zones never share mesh blocks.
    fn zone_event_with(zone: usize, refined: bool, frame: &str, dense: bool) -> Msg {
        let meta = Meta { version: 1, at: SystemTime::now() };
        let range = ZoneRange { zone, lo: 10 * zone, hi: 10 * zone + 14 };
        let model = Arc::new(Reconstruction::new());
        let cloud = patch_at(100.0 * zone as f32);
        Msg::Event(if refined {
            Event::ZoneRefined { meta, range, cloud, model, dense }
        } else {
            Event::ZonePreview { meta, range, cloud, model, dense, frame: frame.into() }
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

    /// (segment, level) of the mesh blocks published after `after`.
    fn mesh_levels(log: &ChunkLog, after: u64) -> BTreeSet<(u32, i32)> {
        log.since(after).iter().filter(|x| x.kind == ChunkKind::MeshBlock as i32).map(|x| (x.segment, x.level)).collect()
    }

    #[test]
    fn stable_display_streams_points_and_meshes_settling_zones_provisionally() {
        let log = ChunkLog::new();
        let mut c = Consumer::new(log.clone(), ANCHOR, false, false).stable_display(1, true);
        let mut mesher = new_mesher(false);
        let mut dirty = BTreeSet::new();
        let pre = Level::Preview as i32;
        let fin = Level::Refined as i32;

        // Zones 0 and 1 may still move: meshed at the preview level, no point chunks.
        c.handle(zone_event_at(0, true, ""), mesher.as_mut(), &mut dirty);
        c.handle(zone_event_at(1, true, ""), mesher.as_mut(), &mut dirty);
        let s = log.last_sequence();
        assert!(log.since(0).iter().all(|x| x.kind == ChunkKind::MeshBlock as i32), "points travel in the diff stream");
        assert_eq!(mesh_levels(&log, 0), BTreeSet::from([(0, pre), (1, pre)]));
        // Zone 0 placed again (settling, here 1 m east): replaced, still provisional.
        let moved = Msg::Event(Event::ZoneRefined {
            meta: Meta { version: 1, at: SystemTime::now() },
            range: ZoneRange { zone: 0, lo: 0, hi: 14 },
            cloud: patch_at(1.0),
            model: Arc::new(Reconstruction::new()),
            dense: true,
        });
        c.handle(moved, mesher.as_mut(), &mut dirty);
        let replaced = mesh_levels(&log, s);
        assert_eq!(replaced, BTreeSet::from([(0, pre)]), "the moved zone 0 is re-meshed provisionally");

        // Zone 2 is past the settle limit: zones 0 and 1 are committed at the refined level (their
        // latest points, so their mesh does not change and deltamesh publishes no new block), then
        // zone 2 is meshed refined.
        let s = log.last_sequence();
        c.handle(zone_event_at(2, true, ""), mesher.as_mut(), &mut dirty);
        assert_eq!(mesh_levels(&log, s), BTreeSet::from([(2, fin)]));
        let one = [deltamesh::Point { pos: [0.0; 3], rgb: [0; 3], normal: [0.0; 3] }];
        for z in [0, 1, 2] {
            assert!(
                matches!(mesher.ingest(z, deltamesh::Level::Refined, &one), Err(deltamesh::IngestError::AlreadyRefined(_))),
                "zone {z} is refined in deltamesh"
            );
        }
        let s = log.last_sequence();
        c.handle(zone_event_at(3, true, ""), mesher.as_mut(), &mut dirty);
        assert_eq!(mesh_levels(&log, s), BTreeSet::from([(3, fin)]));
    }

    #[test]
    fn stable_display_without_densification_sends_point_chunks() {
        let log = ChunkLog::new();
        let mut c = Consumer::new(log.clone(), ANCHOR, false, false).stable_display(3, false);
        let mut mesher = new_mesher(false);
        let mut dirty = BTreeSet::new();
        c.handle(zone_event_at(0, true, ""), mesher.as_mut(), &mut dirty);
        let points: Vec<_> = log.since(0).iter().filter(|x| x.kind == ChunkKind::Points as i32).map(|x| (x.segment, x.level)).collect();
        assert!(!points.is_empty());
        assert!(points.iter().all(|p| *p == (0, Level::Refined as i32)));
    }

    #[test]
    fn stable_display_sends_point_chunks_for_a_zone_whose_densification_failed() {
        let log = ChunkLog::new();
        let mut c = Consumer::new(log.clone(), ANCHOR, false, false).stable_display(3, true);
        let mut mesher = new_mesher(false);
        let mut dirty = BTreeSet::new();
        let points = |log: &ChunkLog| -> BTreeSet<u32> {
            log.since(0).iter().filter(|x| x.kind == ChunkKind::Points as i32).map(|x| x.segment).collect()
        };
        c.handle(zone_event_with(0, true, "", true), mesher.as_mut(), &mut dirty);
        assert!(points(&log).is_empty(), "a dense zone travels in the diff stream");
        // Zone 1's densification failed: the stream has nothing of it, so its points go as chunks.
        c.handle(zone_event_with(1, true, "", false), mesher.as_mut(), &mut dirty);
        assert_eq!(points(&log), BTreeSet::from([1]));
        assert!(log.since(0).iter().any(|x| x.kind == ChunkKind::MeshBlock as i32 && x.segment == 1), "and it is meshed");
        c.handle(zone_event_with(2, true, "", true), mesher.as_mut(), &mut dirty);
        assert_eq!(points(&log), BTreeSet::from([1]));
    }
}
