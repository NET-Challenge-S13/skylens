//! Threads of the live path.
//!
//! ```text
//! compose (caller thread) ── cumulus3d pipeline ──┬─ on_frame lane ─> FrameQueue ─> detector worker ─┐
//!                                                 └─ subscription ──> forwarder ───────────────────────┤
//!                                                                                                     v
//!                                    consumer: display deltas, points, deltamesh blocks, detection fusion ─> ChunkLog
//! poller: collects finished background refinement (cumulus3d only hands it out on push/poll)
//! ```
//!
//! Display deltas. In `PointsMode::Delta`/`Both` the consumer turns every
//! `Event::DisplayDelta` into a `POINTS_DELTA` chunk. Coordinates are ENU metres
//! about the engine anchor exactly like the old per-zone path: the anchor record
//! is the first GPS fix, so cumulus3d reports it as `DisplayFrame::origin_gps`;
//! [`align_delta`] translates a delta whose origin differs by more than 1 cm and
//! drops one that is not in an ENU frame at all. A model-side `DisplayMirror`
//! follows every delta; every `snapshot_every` deltas a snapshot chunk (the whole
//! mirror, same `seq` as the last delta, generation + 1) starts a new generation,
//! which the `ChunkLog` and the core store use to forget the older deltas. The
//! zone clouds still feed meshing and detection fusion in every mode.
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
use cumulus3d_cli::live::{DeltaKind as LiveKind, DisplayDelta, DisplayFrame, DisplayMirror};
use cumulus3d_core::io::ply::PointCloud;
use cumulus3d_core::Reconstruction;
use deltamesh::Mesher;
use skylens_shared::delta::{AddRec, MoveRec, pack_adds, pack_moves, pack_removes};
use skylens_shared::geo::{GeoAnchor, Gps, gps_to_enu};
use skylens_shared::proto::model::v1::Level;
use skylens_shared::proto::payload::v1::{DeltaKind, PointsDelta};

use super::PointsMode;
use super::chunks::{ChunkLog, detections_chunk, mesh_chunk, points_delivery, points_delta_delivery};
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

/// Offsets of the display origin from the anchor below this (metres) are ignored.
pub const ORIGIN_TOLERANCE_M: f64 = 0.01;

/// A display delta expressed in ENU metres about `anchor`.
///
/// `None` when the points are not in an ENU frame (`frame.enu == false`): they
/// would be unaligned with everything else. When the frame origin differs from
/// the anchor by more than [`ORIGIN_TOLERANCE_M`], every add and move is
/// translated by the ENU offset of the origin about the anchor. A frame without a
/// known origin is taken as the anchor frame.
pub fn align_delta(d: &DisplayDelta, frame: &DisplayFrame, anchor: GeoAnchor) -> Option<DisplayDelta> {
    if !frame.enu {
        return None;
    }
    let mut out = d.clone();
    let Some([lat, lon, alt]) = frame.origin_gps else { return Some(out) };
    let o = gps_to_enu(Gps { lat, lon, alt }, anchor);
    if (o.e * o.e + o.n * o.n + o.u * o.u).sqrt() <= ORIGIN_TOLERANCE_M {
        return Some(out);
    }
    let off = [o.e as f32, o.n as f32, o.u as f32];
    let shift = |p: &mut [f32; 3]| {
        for (v, o) in p.iter_mut().zip(off) {
            *v += o;
        }
    };
    out.adds.iter_mut().for_each(|a| shift(&mut a.xyz));
    out.moves.iter_mut().for_each(|m| shift(&mut m.xyz));
    Some(out)
}

/// State of the consumer thread.
pub struct Consumer {
    log: Arc<ChunkLog>,
    anchor: GeoAnchor,
    mode: PointsMode,
    snapshot_every: usize,
    mirror: DisplayMirror,
    /// Snapshot epoch of the delta stream.
    delta_generation: u32,
    deltas_since_snapshot: usize,
    /// `PointsDelta.run`: start time of this model run, unix seconds.
    run: u32,
    /// Frame positions ingested so far (highest position + 1).
    ingested: u32,
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
            anchor,
            mode: PointsMode::default(),
            snapshot_every: super::DEFAULT_SNAPSHOT_EVERY,
            mirror: DisplayMirror::new(),
            delta_generation: 0,
            deltas_since_snapshot: 0,
            run: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(1, |d| u32::try_from(d.as_secs()).unwrap_or(u32::MAX).max(1)),
            ingested: 0,
            gpu,
            detecting,
            positions: HashMap::new(),
            refined_generation: HashMap::new(),
            detection_version: HashMap::new(),
            fuser: Fuser::new(anchor, MAX_OBSERVATIONS),
        }
    }

    /// How points are published and how often a delta snapshot is cut (0 = never).
    pub fn points(mut self, mode: PointsMode, snapshot_every: usize) -> Self {
        self.mode = mode;
        self.snapshot_every = snapshot_every;
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
                    self.ingested = self.ingested.max(u32::try_from(position + 1).unwrap_or(u32::MAX));
                }
                Event::DisplayDelta { delta, zone, kind, t, frame, .. } => {
                    self.display_delta(&delta, zone, kind, t, &frame);
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
        if self.mode.segment() {
            self.log.publish(points_delivery(seg, level, generation, &points));
        }

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

    /// Publishes one cumulus3d display delta as a `POINTS_DELTA` chunk and, on
    /// the snapshot cadence, the snapshot that starts the next generation.
    fn display_delta(&mut self, delta: &DisplayDelta, zone: usize, kind: LiveKind, t: f64, frame: &DisplayFrame) {
        if !self.mode.delta() {
            return;
        }
        let kind = match kind {
            LiveKind::Refined => DeltaKind::Refined,
            LiveKind::Replace => DeltaKind::Replace,
            LiveKind::Final => DeltaKind::Final,
            other => {
                tracing::warn!("display delta {} of unknown kind {other:?} skipped", delta.seq);
                return;
            }
        };
        let Some(delta) = align_delta(delta, frame, self.anchor) else {
            tracing::warn!("display delta {} skipped: its points are not in an ENU frame", delta.seq);
            return;
        };
        if let Err(e) = self.mirror.apply(&delta) {
            tracing::warn!("display delta {} does not fit the model mirror: {e}", delta.seq);
        }
        let adds: Vec<AddRec> =
            delta.adds.iter().map(|a| AddRec { id: a.id, xyz: a.xyz, rgb: a.rgb, zone: a.zone }).collect();
        let moves: Vec<MoveRec> = delta.moves.iter().map(|m| MoveRec { id: m.id, xyz: m.xyz, rgb: m.rgb }).collect();
        let msg = PointsDelta {
            seq: u32::try_from(delta.seq).unwrap_or(u32::MAX),
            kind: kind as i32,
            zone: u32::try_from(zone).unwrap_or(u32::MAX),
            capacity: delta.capacity,
            total_points: u32::try_from(delta.total_points).unwrap_or(u32::MAX),
            snapshot: false,
            t_s: t,
            positions: self.ingested,
            positions_total: 0,
            adds: pack_adds(&adds),
            moves: pack_moves(&moves),
            removes: pack_removes(&delta.removes),
            run: self.run,
        };
        let mut out = points_delta_delivery(&msg, self.delta_generation);
        self.deltas_since_snapshot += 1;
        if self.snapshot_every > 0 && self.deltas_since_snapshot >= self.snapshot_every {
            self.deltas_since_snapshot = 0;
            self.delta_generation += 1;
            out.extend(points_delta_delivery(&self.snapshot(&msg), self.delta_generation));
        }
        self.log.publish(out);
    }

    /// The whole mirror as a snapshot carrying the header of `last`.
    fn snapshot(&self, last: &PointsDelta) -> PointsDelta {
        let adds: Vec<AddRec> = self
            .mirror
            .points()
            .map(|(id, p)| AddRec { id, xyz: p.xyz, rgb: p.rgb, zone: p.zone })
            .collect();
        PointsDelta {
            snapshot: true,
            capacity: last.capacity.max(u32::try_from(self.mirror.slots.len()).unwrap_or(u32::MAX)),
            total_points: u32::try_from(adds.len()).unwrap_or(u32::MAX),
            adds: pack_adds(&adds),
            moves: Vec::new(),
            removes: Vec::new(),
            ..last.clone()
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
        let mut c = Consumer::new(log.clone(), ANCHOR, false, false).points(PointsMode::Segment, 0);
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

    // ---- display deltas ---------------------------------------------------

    use cumulus3d_cli::live::KeyedDiff;
    use skylens_shared::delta::{from_chunk, iter_adds, iter_moves, iter_removes};
    use skylens_shared::proto::model::v1::Chunk;

    const ENU_FRAME: DisplayFrame = DisplayFrame { enu: true, origin_gps: Some([36.3685, 127.3475, 30.0]) };

    /// n points with unique keys; point i belongs to zone `i % 2`.
    fn keyed(n: usize, dz: f32) -> (PointCloud, Vec<usize>) {
        let mut c = PointCloud::default();
        let mut z = Vec::new();
        for i in 0..n {
            c.positions.push([i as f32, 2.0, 1.0 + dz]);
            c.colors.push([i as u8, 10, 20]);
            c.keys.push(500 + i as u64);
            z.push(i % 2);
        }
        (c, z)
    }

    fn delta_event(d: DisplayDelta, kind: LiveKind, zone: usize, t: f64, frame: DisplayFrame) -> Msg {
        let meta = Meta { version: 1, at: SystemTime::now() };
        Msg::Event(Event::DisplayDelta { meta, delta: Arc::new(d), zone, kind, t, frame })
    }

    fn deltas_of(log: &ChunkLog) -> Vec<Arc<Chunk>> {
        log.since(0).into_iter().filter(|c| c.kind == ChunkKind::PointsDelta as i32).collect()
    }

    /// Viewer-side reference applier over the wire records (the contract's rules).
    type Slot = Option<([f32; 3], [u8; 3], u8)>;

    #[derive(Default)]
    struct Viewer {
        slots: Vec<Slot>,
        applied: u32,
    }

    impl Viewer {
        fn apply(&mut self, c: &Chunk) {
            let d = from_chunk(c).unwrap();
            if d.snapshot {
                self.slots.clear();
            } else if d.seq <= self.applied {
                return;
            }
            if self.slots.len() < d.capacity as usize {
                self.slots.resize(d.capacity as usize, None);
            }
            for id in iter_removes(&d.removes).unwrap() {
                self.slots[id as usize] = None;
            }
            for a in iter_adds(&d.adds).unwrap() {
                self.slots[a.id as usize] = Some((a.xyz, a.rgb, a.zone));
            }
            for m in iter_moves(&d.moves).unwrap() {
                let s = self.slots[m.id as usize].as_mut().unwrap();
                s.0 = m.xyz;
                s.1 = m.rgb;
            }
            self.applied = d.seq;
        }
    }

    fn same_as_mirror(v: &Viewer, m: &DisplayMirror) {
        assert_eq!(v.slots.iter().flatten().count(), m.len());
        for (id, p) in m.points() {
            assert_eq!(v.slots[id as usize], Some((p.xyz, p.rgb, p.zone)), "slot {id}");
        }
    }

    fn consumer(mode: PointsMode, every: usize) -> (Arc<ChunkLog>, Consumer, Box<dyn Mesher>) {
        let log = ChunkLog::new();
        (log.clone(), Consumer::new(log, ANCHOR, false, false).points(mode, every), new_mesher(false))
    }

    #[test]
    fn deltas_become_chunks_that_replay_to_the_mirror() {
        let (log, mut c, mut mesher) = consumer(PointsMode::Delta, 0);
        let mut dirty = BTreeSet::new();
        let mut diff = KeyedDiff::new();
        let mut reference = DisplayMirror::new();
        let (c1, z1) = keyed(10, 0.0);
        let (mut c2, z2) = keyed(10, 0.0);
        c2.positions[3][2] += 0.5; // a move
        c2.keys[7] = 999; // a remove + an add
        let (c3, z3) = keyed(10, 0.0);
        c.handle(Msg::Event(Event::FrameIngested { meta: Meta { version: 1, at: SystemTime::now() }, position: 4, images: vec![] }), mesher.as_mut(), &mut dirty);
        for (i, (cloud, zones, kind)) in
            [(&c1, &z1, LiveKind::Refined), (&c2, &z2, LiveKind::Replace), (&c3, &z3, LiveKind::Replace)].into_iter().enumerate()
        {
            let d = diff.apply(cloud, zones).unwrap();
            reference.apply(&d).unwrap();
            c.handle(delta_event(d, kind, i, 1.5 + i as f64, ENU_FRAME), mesher.as_mut(), &mut dirty);
        }
        let fin = diff.unchanged();
        reference.apply(&fin).unwrap();
        c.handle(delta_event(fin, LiveKind::Final, 2, 9.0, ENU_FRAME), mesher.as_mut(), &mut dirty);

        let chunks = deltas_of(&log);
        assert_eq!(chunks.len(), 4);
        assert_eq!(log.since(0).len(), 4, "delta mode publishes nothing but deltas here");
        for (i, ch) in chunks.iter().enumerate() {
            assert_eq!((ch.segment, ch.level, ch.version, ch.generation), (i.min(2) as u32, Level::Refined as i32, i as u32 + 1, 0));
            assert!(ch.block_id.is_empty());
        }
        let first = from_chunk(&chunks[0]).unwrap();
        assert_eq!((first.kind, first.t_s, first.positions, first.positions_total), (DeltaKind::Refined as i32, 1.5, 5, 0));
        assert_eq!((first.capacity, first.total_points, first.snapshot), (10, 10, false));
        let last = from_chunk(&chunks[3]).unwrap();
        assert_eq!(last.kind, DeltaKind::Final as i32);
        assert!(last.adds.is_empty() && last.moves.is_empty() && last.removes.is_empty());
        assert_eq!(last.total_points, 10);

        let mut v = Viewer::default();
        chunks.iter().for_each(|ch| v.apply(ch));
        same_as_mirror(&v, &reference);
        same_as_mirror(&v, &c.mirror);
        // Duplicates are ignored.
        chunks.iter().for_each(|ch| v.apply(ch));
        same_as_mirror(&v, &reference);
    }

    #[test]
    fn snapshots_bump_the_generation_and_bound_the_log() {
        let (log, mut c, mut mesher) = consumer(PointsMode::Delta, 3);
        let mut dirty = BTreeSet::new();
        let mut diff = KeyedDiff::new();
        let mut reference = DisplayMirror::new();
        for i in 0..7 {
            let (cloud, zones) = keyed(10 + i, i as f32 * 0.1);
            let d = diff.apply(&cloud, &zones).unwrap();
            reference.apply(&d).unwrap();
            c.handle(delta_event(d, LiveKind::Replace, 0, i as f64, ENU_FRAME), mesher.as_mut(), &mut dirty);
        }
        // Snapshots after deltas 3 and 6: the log keeps the last snapshot (gen 2) and delta 7 (gen 2).
        let kept = deltas_of(&log);
        let meta: Vec<_> = kept.iter().map(|ch| (ch.version, ch.generation)).collect();
        assert_eq!(meta, vec![(6, 2), (7, 2)]);
        let snap = from_chunk(&kept[0]).unwrap();
        assert!(snap.snapshot && snap.moves.is_empty() && snap.removes.is_empty());
        assert_eq!(snap.seq, 6);
        assert_eq!(snap.total_points as usize, snap.adds.len() / 20);
        let mut v = Viewer::default();
        kept.iter().for_each(|ch| v.apply(ch));
        same_as_mirror(&v, &reference);
        // A resuming subscriber sees the same coherent list.
        let mut rx = log.subscribe(0);
        assert_eq!(rx.blocking_recv().unwrap().generation, 2);

        // 0 = never.
        let (log, mut c, mut mesher) = consumer(PointsMode::Delta, 0);
        let mut diff = KeyedDiff::new();
        for i in 0..5 {
            let (cloud, zones) = keyed(4 + i, 0.0);
            c.handle(delta_event(diff.apply(&cloud, &zones).unwrap(), LiveKind::Refined, 0, 0.0, ENU_FRAME), mesher.as_mut(), &mut dirty);
        }
        assert_eq!(deltas_of(&log).len(), 5);
        assert!(deltas_of(&log).iter().all(|ch| ch.generation == 0));
    }

    #[test]
    fn frame_alignment_translates_or_drops() {
        let (cloud, zones) = keyed(4, 0.0);
        let d = KeyedDiff::new().apply(&cloud, &zones).unwrap();
        assert!(align_delta(&d, &DisplayFrame { enu: false, origin_gps: None }, ANCHOR).is_none());
        // Same origin (and a sub-centimetre offset): untouched.
        assert_eq!(align_delta(&d, &ENU_FRAME, ANCHOR).unwrap(), d);
        let near = DisplayFrame { enu: true, origin_gps: Some([36.3685 + 5e-9, 127.3475, 30.0]) };
        assert_eq!(align_delta(&d, &near, ANCHOR).unwrap(), d);
        assert_eq!(align_delta(&d, &DisplayFrame { enu: true, origin_gps: None }, ANCHOR).unwrap(), d);
        // Origin 10 m up and ~111 m north of the anchor.
        let origin = Gps { lat: 36.3685 + 0.001, lon: 127.3475, alt: 40.0 };
        let off = gps_to_enu(origin, ANCHOR);
        assert!((off.n - 111.3).abs() < 0.5 && (off.u - 10.0).abs() < 1e-9, "{off:?}");
        let far = DisplayFrame { enu: true, origin_gps: Some([origin.lat, origin.lon, origin.alt]) };
        let mut with_move = d.clone();
        with_move.moves.push(cumulus3d_cli::live::PointMove { id: 1, xyz: [1.0, 1.0, 1.0], rgb: [1, 2, 3] });
        let t = align_delta(&with_move, &far, ANCHOR).unwrap();
        for (a, b) in t.adds.iter().zip(&with_move.adds) {
            assert!((a.xyz[0] - b.xyz[0] - off.e as f32).abs() < 1e-3);
            assert!((a.xyz[1] - b.xyz[1] - off.n as f32).abs() < 1e-3);
            assert!((a.xyz[2] - b.xyz[2] - 10.0).abs() < 1e-3);
            assert_eq!((a.id, a.rgb, a.zone), (b.id, b.rgb, b.zone));
        }
        assert!((t.moves[0].xyz[1] - 1.0 - off.n as f32).abs() < 1e-3);

        // Through the consumer: not-ENU is skipped, a shifted origin is translated.
        let (log, mut c, mut mesher) = consumer(PointsMode::Delta, 0);
        let mut dirty = BTreeSet::new();
        c.handle(delta_event(d.clone(), LiveKind::Refined, 0, 0.0, DisplayFrame { enu: false, origin_gps: None }), mesher.as_mut(), &mut dirty);
        assert!(deltas_of(&log).is_empty());
        c.handle(delta_event(d.clone(), LiveKind::Refined, 0, 0.0, far), mesher.as_mut(), &mut dirty);
        let got = from_chunk(&deltas_of(&log)[0]).unwrap();
        let first = iter_adds(&got.adds).unwrap().next().unwrap();
        assert!((first.xyz[1] - (2.0 + off.n as f32)).abs() < 1e-3);
    }

    #[test]
    fn modes_select_the_point_paths_and_meshing_always_runs() {
        for (mode, want_segment, want_delta) in
            [(PointsMode::Delta, false, true), (PointsMode::Segment, true, false), (PointsMode::Both, true, true)]
        {
            let (log, mut c, mut mesher) = consumer(mode, 0);
            let mut dirty = BTreeSet::new();
            let (cloud, zones) = keyed(6, 0.0);
            let d = KeyedDiff::new().apply(&cloud, &zones).unwrap();
            c.handle(delta_event(d, LiveKind::Refined, 2, 0.0, ENU_FRAME), mesher.as_mut(), &mut dirty);
            c.handle(zone_event(true, ""), mesher.as_mut(), &mut dirty);
            let all = log.since(0);
            let has = |k: ChunkKind| all.iter().any(|x| x.kind == k as i32);
            assert_eq!(has(ChunkKind::Points), want_segment, "{mode:?}");
            assert_eq!(has(ChunkKind::PointsDelta), want_delta, "{mode:?}");
            assert!(has(ChunkKind::MeshBlock), "{mode:?} must still mesh the zone");
        }
    }
}
