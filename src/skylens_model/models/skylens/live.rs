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
//! Stable display. Zones arrive refined only (no previews). cumulus3d composes the display in
//! memory (`NodeBuilder::display_deltas`) and delivers an `Event::DisplayDelta` after every dense
//! zone result. The consumer writes each delta as one point slot chunk (`slotstream`, in delivery
//! order), applies it to a model-side mirror of the slot table, and meshes the mirror's points of
//! the zones the delta touched (as repeatable previews): the display has already dropped the
//! overlap between zones, so the mesh is one surface, not one per raw zone cloud. Zone clouds
//! themselves publish no point chunks and no mesh, except a zone whose densification failed
//! (`ZoneRefined { dense: false }`: it has no delta) and a run without densification: those
//! go out as point chunks and are meshed from the zone cloud. Detection fusion always uses the
//! zone clouds.
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
use cumulus3d_cli::live::{DeltaKind, DisplayDelta, DisplayFrame, DisplayMirror};
use cumulus3d_core::io::ply::PointCloud;
use cumulus3d_core::Reconstruction;
use deltamesh::Mesher;
use skylens_shared::geo::GeoAnchor;
use skylens_shared::proto::model::v1::Level;

use super::chunks::{ChunkLog, detections_chunk, mesh_chunk, points_delivery};
use super::fusion::{FrameObservation, Fuser};
use super::recon::{positions_f64, zone_points, zone_views};
use super::slotstream::{SlotEncoder, align_delta};
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
    /// The zone's points are dense: with a display stream their points leave as display deltas.
    dense: bool,
}

/// The display stream of stable display: the slot messages and the mirror the mesh is built from.
struct DisplayOut {
    anchor: GeoAnchor,
    encoder: SlotEncoder,
    mirror: DisplayMirror,
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
    anchor: GeoAnchor,
    display: Option<DisplayOut>,
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
            anchor,
            display: None,
        }
    }

    /// The node delivers display deltas (`display_deltas`): they leave as point slot chunks of the
    /// stream `stream`, and the mesh comes from the composed display instead of the zone clouds.
    pub fn display_stream(mut self, stream: u32) -> Self {
        self.display = Some(DisplayOut { anchor: self.anchor, encoder: SlotEncoder::new(stream), mirror: DisplayMirror::new() });
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
                    if frame == "display" {
                        // Stable display: the zone's preview leaves as a `Preview` display delta.
                        return;
                    }
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
                Event::DisplayDelta { delta, zone, kind, t, frame, .. } => {
                    self.display_delta(&delta, zone, kind, t, &frame, mesher)
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
        // With a display stream the dense zones' points and mesh come from the display deltas.
        let streamed = dense && self.display.is_some();
        if !streamed {
            if self.display.is_some() {
                tracing::info!("zone {seg}: not densified; its sparse points go out as point chunks");
            }
            let generation = if level == Level::Refined {
                let g = self.refined_generation.entry(seg).or_insert(0);
                *g += 1;
                *g
            } else {
                0
            };
            self.log.publish(points_delivery(seg, level, generation, &points));
            let mpts: Vec<deltamesh::Point> =
                points.iter().map(|p| deltamesh::Point { pos: p.pos, rgb: p.rgb, normal: p.normal }).collect();
            self.mesh(seg, level, &mpts, mesher);
        }
        if self.detecting {
            self.fuser.set_zone(seg, &positions_f64(&points), zone_views(model, range, &self.positions));
            dirty.insert(seg);
        }
    }

    /// Hands `points` to the mesher as segment `seg` at `level` and publishes the blocks that changed.
    fn mesh(&mut self, seg: u32, level: Level, points: &[deltamesh::Point], mesher: &mut dyn Mesher) {
        let dm_level = if level == Level::Refined { deltamesh::Level::Refined } else { deltamesh::Level::Preview };
        match mesher.ingest(seg, dm_level, points) {
            Ok(_) => {
                let changed = mesher.extract();
                let blocks: Vec<_> =
                    changed.iter().map(|id| mesh_chunk(seg, level, *id, mesher.version(id), mesher.mesh(id))).collect();
                self.log.publish(blocks);
            }
            Err(e) => tracing::warn!("deltamesh zone {seg}: {e}"),
        }
    }

    /// Publishes one display delta as a point slot chunk and re-meshes the zones it touched.
    fn display_delta(
        &mut self,
        delta: &DisplayDelta,
        zone: usize,
        kind: DeltaKind,
        t: f64,
        frame: &DisplayFrame,
        mesher: &mut dyn Mesher,
    ) {
        let Some(out) = self.display.as_mut() else { return };
        let Some(delta) = align_delta(delta, frame, out.anchor) else {
            tracing::warn!("display delta {} skipped: its points are not in an ENU frame", delta.seq);
            return;
        };
        // Zones whose display points change: those of the adds, of the moved points and of the
        // removed ids. The last two are only known from the mirror before the delta.
        let mut touched: BTreeSet<u8> = delta.adds.iter().map(|a| a.zone).collect();
        let zone_of = |id: u32| out.mirror.slots.get(id as usize).copied().flatten().map(|p| p.zone);
        touched.extend(delta.moves.iter().filter_map(|m| zone_of(m.id)));
        touched.extend(delta.removes.iter().filter_map(|&id| zone_of(id)));
        if let Err(e) = out.mirror.apply(&delta) {
            tracing::warn!("display delta {} does not fit the model mirror: {e}", delta.seq);
        }
        let chunk = match out.encoder.encode(&delta, zone, kind, t) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("display delta {} skipped: {e:#}", delta.seq);
                return;
            }
        };
        tracing::info!(
            "display delta {} (zone {zone}): +{} ~{} -{}, {} points, {:.1} MB",
            chunk.version,
            delta.adds.len(),
            delta.moves.len(),
            delta.removes.len(),
            delta.total_points,
            chunk.payload.len() as f64 / 1e6
        );
        self.log.publish([chunk]);
        self.mesh_display(&touched, u32::try_from(zone).unwrap_or(u32::MAX), mesher);
    }

    /// Meshes the composed display: for each touched zone the mirror's points of that zone go to
    /// the mesher as a (re-sent) preview of the segment, and the changed blocks are published with
    /// preview semantics. A refined ingest is accepted once per segment, so it is never used here.
    /// The display dedupes the overlap between zones, so the mesh is one surface.
    fn mesh_display(&mut self, touched: &BTreeSet<u8>, seg: u32, mesher: &mut dyn Mesher) {
        let Some(out) = self.display.as_ref() else { return };
        if touched.is_empty() {
            return;
        }
        let mut by_zone: HashMap<u8, Vec<deltamesh::Point>> = touched.iter().map(|z| (*z, Vec::new())).collect();
        for (_, p) in out.mirror.points() {
            if let Some(v) = by_zone.get_mut(&p.zone) {
                v.push(deltamesh::Point { pos: p.xyz, rgb: p.rgb, normal: [0.0; 3] });
            }
        }
        for z in touched {
            if let Err(e) = mesher.ingest(u32::from(*z), deltamesh::Level::Preview, &by_zone[z]) {
                tracing::warn!("deltamesh display zone {z}: {e}");
            }
        }
        let changed = mesher.extract();
        let blocks: Vec<_> =
            changed.iter().map(|id| mesh_chunk(seg, Level::Preview, *id, mesher.version(id), mesher.mesh(id))).collect();
        self.log.publish(blocks);
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
    use skylens_shared::proto::model::v1::{Chunk, ChunkKind};
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

    use cumulus3d_cli::live::{PointAdd, PointMove};
    use pointblitz_io::slots::Fold;

    fn blocks_since(log: &ChunkLog, after: u64) -> Vec<(u32, i32)> {
        log.since(after).iter().filter(|x| x.kind == ChunkKind::MeshBlock as i32).map(|x| (x.segment, x.level)).collect()
    }

    fn slots_in(log: &ChunkLog) -> Vec<Arc<Chunk>> {
        log.since(0).into_iter().filter(|x| x.kind == ChunkKind::PointSlots as i32).collect()
    }

    /// The adds of a flat patch of `n` x `n` points (5 cm apart) at x = `x0`, ids from `id0`, owned by `zone`.
    fn patch_adds(x0: f32, n: u32, id0: u32, zone: u8) -> Vec<PointAdd> {
        (0..n * n)
            .map(|i| PointAdd { id: id0 + i, xyz: [x0 + (i / n) as f32 * 0.05, (i % n) as f32 * 0.05, 0.0], rgb: [180; 3], zone })
            .collect()
    }

    fn delta_event(zone: usize, delta: DisplayDelta, enu: bool) -> Msg {
        delta_event_of(zone, delta, enu, DeltaKind::Refined, 1.0)
    }

    fn delta_event_of(zone: usize, delta: DisplayDelta, enu: bool, kind: DeltaKind, t: f64) -> Msg {
        Msg::Event(Event::DisplayDelta {
            meta: Meta { version: 1, at: SystemTime::now() },
            delta: Arc::new(delta),
            zone,
            kind,
            t,
            frame: DisplayFrame { enu, origin_gps: Some([ANCHOR.lat, ANCHOR.lon, ANCHOR.alt]) },
        })
    }

    fn delta(adds: Vec<PointAdd>, moves: Vec<PointMove>, removes: Vec<u32>, total: usize, capacity: u32) -> DisplayDelta {
        DisplayDelta { seq: 1, adds, moves, removes, total_points: total, capacity }
    }

    #[test]
    fn display_deltas_become_ordered_slot_chunks_and_a_mesh_from_the_display() {
        let log = ChunkLog::new();
        let mut c = Consumer::new(log.clone(), ANCHOR, false, false).display_stream(5);
        let mut mesher = new_mesher(false);
        let mut dirty = BTreeSet::new();
        let (pre, fin) = (Level::Preview as i32, Level::Refined as i32);

        // The dense zone clouds themselves publish nothing: their points leave as deltas.
        c.handle(zone_event_at(0, true, ""), mesher.as_mut(), &mut dirty);
        assert_eq!(log.last_sequence(), 0);

        c.handle(delta_event(0, delta(patch_adds(0.0, 100, 0, 0), vec![], vec![], 10_000, 10_000), true), mesher.as_mut(), &mut dirty);
        let s = log.last_sequence();
        assert!(blocks_since(&log, 0).iter().all(|b| *b == (0, pre)), "display meshes are previews of the delta's zone");
        assert!(!blocks_since(&log, 0).is_empty());
        // Zone 1 owns the next patch (200 m away); zone 0's point 0 moves and a point of zone 0 leaves.
        let d2 = delta(
            patch_adds(200.0, 100, 10_000, 1),
            vec![PointMove { id: 0, xyz: [0.0, 0.0, 0.2], rgb: [1, 2, 3] }],
            vec![1],
            19_999,
            20_000,
        );
        c.handle(delta_event(1, d2, true), mesher.as_mut(), &mut dirty);
        assert!(blocks_since(&log, s).iter().any(|b| b.0 == 1), "the zone 1 delta meshes zone 1");

        // The run ends with an empty final message carrying kind and time.
        let end = delta(vec![], vec![], vec![], 19_999, 20_000);
        c.handle(delta_event_of(1, end, true, DeltaKind::Final, 7.25), mesher.as_mut(), &mut dirty);
        let slots = slots_in(&log);
        let last = slots.last().unwrap();
        assert_eq!((last.version, last.slot_kind, last.slot_t_ms), (3, 3, 7250));
        let h = pointblitz_io::slots::validate(&last.payload).unwrap();
        assert_eq!((h.n_add, h.n_move, h.n_remove), (0, 0, 0));
        assert_eq!(slots.iter().map(|x| (x.slot_kind, x.slot_t_ms)).collect::<Vec<_>>()[..2], [(1, 1000), (1, 1000)]);
        let slots = &slots[..2];
        assert_eq!(slots.iter().map(|x| (x.version, x.generation, x.segment, x.level)).collect::<Vec<_>>(), vec![(1, 5, 0, fin), (2, 5, 1, fin)]);
        let mut fold = Fold::new();
        for x in slots {
            fold.apply(&x.payload).unwrap();
        }
        assert_eq!((fold.seq(), fold.points(), fold.capacity()), (2, 19_999, 20_000));
        assert_eq!(c.display.as_ref().unwrap().mirror.len(), 19_999, "the mirror follows the same deltas");
        assert!(log.since(0).iter().all(|x| x.kind != ChunkKind::Points as i32));
    }

    #[test]
    fn a_delta_outside_the_enu_frame_is_skipped_without_using_a_seq() {
        let log = ChunkLog::new();
        let mut c = Consumer::new(log.clone(), ANCHOR, false, false).display_stream(5);
        let mut mesher = new_mesher(false);
        let mut dirty = BTreeSet::new();
        c.handle(delta_event(0, delta(patch_adds(0.0, 10, 0, 0), vec![], vec![], 100, 100), false), mesher.as_mut(), &mut dirty);
        assert_eq!(log.last_sequence(), 0);
        c.handle(delta_event(0, delta(patch_adds(0.0, 10, 0, 0), vec![], vec![], 100, 100), true), mesher.as_mut(), &mut dirty);
        assert_eq!(slots_in(&log).iter().map(|x| x.version).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn a_zone_whose_densification_failed_goes_out_as_point_chunks() {
        let log = ChunkLog::new();
        let mut c = Consumer::new(log.clone(), ANCHOR, false, false).display_stream(5);
        let mut mesher = new_mesher(false);
        let mut dirty = BTreeSet::new();
        let points = |log: &ChunkLog| -> BTreeSet<u32> {
            log.since(0).iter().filter(|x| x.kind == ChunkKind::Points as i32).map(|x| x.segment).collect()
        };
        c.handle(zone_event_with(0, true, "", true), mesher.as_mut(), &mut dirty);
        assert!(points(&log).is_empty(), "a dense zone travels as display deltas");
        // Zone 1's densification failed: it has no delta, so its points go as chunks, meshed from the zone.
        c.handle(zone_event_with(1, true, "", false), mesher.as_mut(), &mut dirty);
        assert_eq!(points(&log), BTreeSet::from([1]));
        assert!(log.since(0).iter().any(|x| x.kind == ChunkKind::MeshBlock as i32 && x.segment == 1));
        c.handle(zone_event_with(2, true, "", true), mesher.as_mut(), &mut dirty);
        assert_eq!(points(&log), BTreeSet::from([1]));
    }

    #[test]
    fn without_a_display_stream_dense_zones_go_out_as_point_chunks() {
        let log = ChunkLog::new();
        let mut c = Consumer::new(log.clone(), ANCHOR, false, false);
        let mut mesher = new_mesher(false);
        let mut dirty = BTreeSet::new();
        c.handle(zone_event_at(0, true, ""), mesher.as_mut(), &mut dirty);
        let points: Vec<_> =
            log.since(0).iter().filter(|x| x.kind == ChunkKind::Points as i32).map(|x| (x.segment, x.level)).collect();
        assert!(!points.is_empty());
        assert!(points.iter().all(|p| *p == (0, Level::Refined as i32)));
    }
}
