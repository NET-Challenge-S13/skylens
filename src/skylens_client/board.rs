//! What the board knows, shared by the DOM side (`app.rs`) and the two 3D views (`scene.rs`,
//! `inset.rs`) as `Rc<RefCell<Board>>`.
//!
//! Everything here arrived from the core over `ws /viewer`: the site (whose anchor is the ENU
//! origin), the reconstruction chunks (through the shared chunk store and its arrival rules),
//! telemetry, the route, the mission line and the server status. What the board derives from it
//! (per-segment samples, the framing, the interior and corridor analysis) is cached here and
//! refreshed when a segment's chunks change.
//!
//! Points come either as point chunks per segment or as the point slot stream (stable display,
//! decision 0052); a segment of the slot stream is the zone its slots belong to, read off the
//! store's slot state the same way (samples, height range, box, point count).

use crate::delta_log::{DeltaLog, LayerOp, SlotMeta};
use crate::follow::{self, CorridorPath, DronePose, Framing, Interior};
use crate::panels_live::LiveSettings;
use crate::glam::DVec3;
use crate::markers;
use crate::points;
use skylens_shared::proto::model::v1 as model;
use skylens_shared::proto::payload::v1 as payload;
use skylens_shared::proto::viewer::v1 as viewer;
use skylens_viewer::camera::CameraSync;
use skylens_viewer::{ChunkStore, SiteFrame};
use std::cell::RefCell;
use std::rc::Rc;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Samples kept per segment for heights, framing and the corridor analysis.
const SAMPLES_PER_SEGMENT: usize = 2_000;
/// Samples the interior analysis reads at most (legacy `sampleCenters(9000)`).
const INTERIOR_SAMPLES: usize = 9_000;
/// Samples per segment the point panel frames on (legacy `samplePoints(400)`).
pub const FIT_SAMPLES_PER_SEGMENT: usize = 400;
/// Second half of a [`SegmentInfo`] key read off the slot stream (the first half is the zone's
/// stamp); point chunk keys count chunks and never get this high.
const SLOT_KEY: usize = usize::MAX;

/// Slot messages kept for the live log while they wait for a gap to fill.
const MAX_PENDING_SLOT_MESSAGES: usize = 128;

/// Which view drains the live log's ops (each keeps its own cursor).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveView {
    Main,
    Inset,
}

/// A drone the pipeline reported (telemetry only; the board simulates nothing).
#[derive(Debug, Clone, PartialEq)]
pub struct Drone {
    pub id: u32,
    pub pos: DVec3,
    pub forward: DVec3,
}

/// What the board read off one segment's point chunks (or its slots).
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentInfo {
    key: (u64, usize),
    pub samples: Vec<DVec3>,
    pub range: Option<(f64, f64)>,
    pub core: Option<(DVec3, DVec3)>,
    pub center: Option<DVec3>,
    pub points: u64,
    pub refined: bool,
}

#[derive(Debug, Clone)]
struct MeshCenter {
    stamp: u64,
    segment: u32,
    center: DVec3,
}

pub struct Board {
    pub store: ChunkStore,
    pub site: Option<SiteFrame>,
    pub drones: Vec<Drone>,
    pub active_drone: Option<u32>,
    /// The assigned route in ENU (two waypoints or more).
    pub route: Option<Vec<DVec3>>,
    pub mission: Option<viewer::MissionStatus>,
    pub core_status: Option<viewer::ServerStatus>,
    /// Reconstruction chunks and detections this board received on the current connection.
    pub local_chunks: u64,
    pub local_detections: u64,
    /// The last reconstruction chunk could not be used.
    pub geometry_failed: bool,
    /// Detections revealed (their segment's final level arrived), in reveal order. Kept across
    /// reconnects so a replay does not toast them again.
    pub revealed: Vec<String>,
    pub confirmed: BTreeSet<String>,
    /// Camera state machine (SYNCED / FOCUSING / LOCKED / RETURNING). Its own orbit pose is not
    /// used: like the legacy board, the camera eases toward the focus or follow pose.
    pub cam: CameraSync,
    pub segments: BTreeMap<u32, SegmentInfo>,
    mesh_centers: HashMap<[i32; 3], MeshCenter>,
    /// Framing from the first geometry (legacy: from the placeholder cloud).
    pub framing: Option<Framing>,
    pub floor_z: f64,
    pub interior: Option<Interior>,
    pub corridor: Option<CorridorPath>,
    pub point_size_m: f32,
    /// The first segment's reveal progress (written by the main view).
    pub first_progress: Option<f32>,
    pub point_layer_on: bool,
    /// Seconds since the board started (marker pulse).
    pub time: f64,
    /// The live point stream as events: timeline rows, seek, undo (the point layers drain it).
    pub live: DeltaLog,
    /// The controls bar's choices (point size, colour mode, speed, fit), written by the panels.
    pub settings: Rc<RefCell<LiveSettings>>,
    /// Slot messages seen but not yet applied by the store: seq -> (zone, message).
    pending_slots: BTreeMap<u32, (u32, SlotMeta, Vec<u8>)>,
    /// Stream id the live log follows.
    live_stream: u32,
    cursors: [Option<u64>; 2],
}

impl Board {
    pub fn new(point_layer_on: bool) -> Board {
        Board {
            store: ChunkStore::new(),
            site: None,
            drones: Vec::new(),
            active_drone: None,
            route: None,
            mission: None,
            core_status: None,
            local_chunks: 0,
            local_detections: 0,
            geometry_failed: false,
            revealed: Vec::new(),
            confirmed: BTreeSet::new(),
            cam: CameraSync::new(DVec3::splat(-1.0), DVec3::splat(1.0)),
            segments: BTreeMap::new(),
            mesh_centers: HashMap::new(),
            framing: None,
            floor_z: 0.0,
            interior: None,
            corridor: None,
            point_size_m: 0.15,
            first_progress: None,
            point_layer_on,
            time: 0.0,
            live: DeltaLog::new(),
            settings: Rc::new(RefCell::new(LiveSettings::default())),
            pending_slots: BTreeMap::new(),
            live_stream: 0,
            cursors: [None, None],
        }
    }

    /// The link reconnected: the core replays what it holds.
    pub fn reconnected(&mut self) {
        self.store.clear();
        self.live.reset();
        self.pending_slots.clear();
        self.live_stream = 0;
        self.cursors = [None, None];
        self.local_chunks = 0;
        self.local_detections = 0;
        self.geometry_failed = false;
    }

    // ---- telemetry and plan -----------------------------------------------------------------

    /// A telemetry frame: the drone appears the first time (the first one becomes active).
    pub fn telemetry(&mut self, t: &model::DroneTelemetry) {
        let (Some(site), Some(gps)) = (&self.site, &t.gps) else { return };
        let pos = DVec3::from(site.enu_of(gps));
        let forward = follow::drone_forward(t.heading_deg);
        match self.drones.iter_mut().find(|d| d.id == t.drone_id) {
            Some(d) => {
                d.pos = pos;
                d.forward = forward;
            }
            None => {
                self.drones.push(Drone { id: t.drone_id, pos, forward });
                if self.drones.len() == 1 {
                    self.active_drone = Some(t.drone_id);
                }
            }
        }
    }

    pub fn assign_route(&mut self, r: &viewer::AssignRoute) {
        let Some(site) = &self.site else { return };
        self.route = (r.waypoints.len() >= 2).then(|| r.waypoints.iter().map(|w| DVec3::from(site.enu_of(w))).collect());
    }

    pub fn active(&self) -> Option<DronePose> {
        let d = self.drones.iter().find(|d| Some(d.id) == self.active_drone).or(self.drones.first())?;
        Some(DronePose { pos: d.pos, forward: d.forward })
    }

    // ---- reconstruction ---------------------------------------------------------------------

    /// Applies a reconstruction chunk. Terrain and buildings are the tower's base map and never
    /// shown here.
    pub fn chunk(&mut self, chunk: &model::Chunk) -> skylens_viewer::Applied {
        self.note_slot_message(chunk);
        let applied = self.store.apply(chunk);
        use skylens_viewer::Change as C;
        if let C::SlotsApplied { seqs, .. } = &applied.change {
            self.feed_live(seqs);
        } else if chunk.kind() == model::ChunkKind::PointSlots && !matches!(applied.change, C::SlotsHeld { .. }) {
            // Stale, invalid or from an older stream: never applied, so never logged.
            self.pending_slots.remove(&chunk.version);
        }
        match (&applied.change, chunk.kind()) {
            (C::Invalid(_), model::ChunkKind::Points | model::ChunkKind::MeshBlock | model::ChunkKind::PointSlots) => {
                self.geometry_failed = true
            }
            (C::PointsAppended { .. } | C::PointsReplaced { .. } | C::PointsPending { .. }, _)
            | (C::SlotsApplied { .. } | C::SlotsHeld { .. }, _)
            | (C::MeshUpserted { .. } | C::MeshRemoved { .. }, _) => {
                self.geometry_failed = false;
                self.local_chunks += 1;
            }
            (C::Detections { ids }, _) => self.local_detections += ids.len() as u64,
            _ => {}
        }
        applied
    }

    /// Keeps a slot message until the store has applied it (it may be held behind a gap), and
    /// starts a new live run when the stream id grows.
    fn note_slot_message(&mut self, chunk: &model::Chunk) {
        if chunk.kind() != model::ChunkKind::PointSlots {
            return;
        }
        let Ok(h) = skylens_viewer::pointblitz::io::slots::validate(&chunk.payload) else { return };
        if chunk.generation < self.live_stream {
            return;
        }
        if chunk.generation > self.live_stream {
            self.live_stream = chunk.generation;
            self.pending_slots.clear();
            self.live.reset();
            self.cursors = [None, None];
        }
        if self.pending_slots.len() >= MAX_PENDING_SLOT_MESSAGES {
            self.pending_slots.clear();
        }
        self.pending_slots.insert(h.seq, (chunk.segment, SlotMeta { kind: chunk.slot_kind, t_ms: chunk.slot_t_ms }, chunk.payload.clone()));
    }

    /// Logs the messages the store applied, in order, with the board clock.
    fn feed_live(&mut self, seqs: &[u32]) {
        let t_ms = skylens_viewer::now_ms();
        for seq in seqs {
            if let Some((zone, meta, msg)) = self.pending_slots.remove(seq) {
                self.live.push_meta(self.live_stream, zone, &msg, t_ms, meta);
            }
        }
    }

    /// What a point layer must do since it last asked (see [`DeltaLog::ops_since`]).
    pub fn live_ops(&mut self, who: LiveView) -> Vec<LayerOp> {
        let i = who as usize;
        let ops = self.live.ops_since(&mut self.cursors[i]);
        // A layer without a cursor starts from the present state, so it needs no old ops.
        if let Some(low) = self.cursors.iter().flatten().min().copied() {
            self.live.trim_ops(low);
        }
        ops
    }

    /// The layer lost its GPU state: it starts over from the present on its next ask.
    pub fn live_ops_reset(&mut self, who: LiveView) {
        self.cursors[who as usize] = None;
    }

    /// Whether any reconstruction has arrived.
    pub fn has_geometry(&self) -> bool {
        !self.store.arrived().is_empty()
    }

    /// Segments that hold points: point chunk segments and slot zones with shown slots.
    pub fn point_segments(&self) -> BTreeSet<u32> {
        let mut out: BTreeSet<u32> = self.store.segments().map(|(s, _)| s).collect();
        out.extend(self.store.slots().zones().filter(|(_, z)| z.points > 0).map(|(z, _)| z));
        out
    }

    /// Re-reads the segments whose chunks or slots changed, then the framing and the interior
    /// analysis.
    pub fn refresh(&mut self) {
        let mut changed = false;
        let live = self.point_segments();
        let before = self.segments.len();
        self.segments.retain(|s, _| live.contains(s));
        changed |= before != self.segments.len();
        for (seg, pts) in self.store.segments() {
            let key = (pts.epoch(), pts.payloads().len());
            if self.segments.get(&seg).is_some_and(|i| i.key == key) {
                continue;
            }
            let payloads = pts.payloads();
            let samples = points::sample_chunks(payloads.iter().map(|p| &p[..]), SAMPLES_PER_SEGMENT);
            let info = SegmentInfo {
                key,
                range: points::height_range(&samples),
                core: points::core_box(&samples),
                center: pts.center().map(DVec3::from),
                points: pts.point_count(),
                refined: pts.is_refined(),
                samples,
            };
            self.segments.insert(seg, info);
            changed = true;
        }
        changed |= self.refresh_slot_zones();
        // Mesh block centres, per stored version.
        let ids: BTreeSet<[i32; 3]> = self.store.mesh_blocks().map(|(id, _)| id).collect();
        self.mesh_centers.retain(|id, _| ids.contains(id));
        for (id, e) in self.store.mesh_blocks() {
            if self.mesh_centers.get(&id).is_some_and(|m| m.stamp == e.stamp) {
                continue;
            }
            if let Some(center) = mesh_center(&e.block) {
                self.mesh_centers.insert(id, MeshCenter { stamp: e.stamp, segment: e.segment, center });
            }
        }
        if changed {
            self.analyse();
        }
    }

    /// Re-reads the slot zones whose slots changed (one pass over the slots for all of them).
    /// A segment that also has point chunks is read off its chunks.
    fn refresh_slot_zones(&mut self) -> bool {
        let sp = self.store.slots();
        let stale: Vec<(u32, u64)> = sp
            .zones()
            .filter(|(z, s)| s.points > 0 && self.store.segment(*z).is_none())
            .filter(|(z, s)| self.segments.get(z).is_none_or(|i| i.key != (s.stamp, SLOT_KEY)))
            .map(|(z, s)| (z, s.stamp))
            .collect();
        if stale.is_empty() {
            return false;
        }
        let mut samples = sp.samples(SAMPLES_PER_SEGMENT);
        for (zone, stamp) in stale {
            let Some(z) = sp.zone(zone) else { continue };
            let samples: Vec<DVec3> = samples.remove(&zone).unwrap_or_default().into_iter().map(DVec3::from).collect();
            let center = (z.lo[0] <= z.hi[0]).then(|| (DVec3::from(z.lo.map(f64::from)) + DVec3::from(z.hi.map(f64::from))) / 2.0);
            let info = SegmentInfo {
                key: (stamp, SLOT_KEY),
                range: points::height_range(&samples),
                core: points::core_box(&samples),
                center,
                points: z.points,
                refined: true,
                samples,
            };
            self.segments.insert(zone, info);
        }
        true
    }

    fn analyse(&mut self) {
        let all: Vec<DVec3> = self.segments.values().flat_map(|i| i.samples.iter().copied()).collect();
        if self.framing.is_none() {
            self.framing = Framing::from_samples(&all);
        }
        let mut z: Vec<f64> = all.iter().map(|p| p.z).collect();
        z.sort_by(f64::total_cmp);
        if self.interior.is_none() {
            self.floor_z = points::quantile(&z, 0.02).unwrap_or(0.0);
        }
        let step = (all.len() / INTERIOR_SAMPLES).max(1);
        let thin: Vec<DVec3> = all.iter().step_by(step).copied().collect();
        if self.interior.is_none() {
            self.interior = follow::detect_interior(&thin);
            if let Some(i) = &self.interior {
                self.floor_z = i.floor_z;
            }
        }
        if self.interior.is_some() {
            // The strip grows; the centreline is re-measured on what has arrived.
            self.corridor = follow::build_corridor_path(&thin);
        }
        let payloads: Vec<&[u8]> = self.store.segments().flat_map(|(_, p)| p.payloads().iter().map(|c| &c[..])).collect();
        let mut spacing = points::chunk_spacings(payloads);
        // Slot zones: the footprint of their core box over their point count.
        spacing.extend(
            self.segments
                .values()
                .filter(|i| i.key.1 == SLOT_KEY)
                .filter_map(|i| points::spacing(i.core?, i.points)),
        );
        self.point_size_m = points::point_size_from_spacings(spacing);
    }

    /// Centres of the arrived segments: their points, else their mesh blocks.
    pub fn segment_centers(&self) -> Vec<(u32, DVec3)> {
        let mut out: BTreeMap<u32, DVec3> = self.segments.iter().filter_map(|(s, i)| Some((*s, i.center?))).collect();
        let mut mesh: BTreeMap<u32, (DVec3, usize)> = BTreeMap::new();
        for m in self.mesh_centers.values() {
            let e = mesh.entry(m.segment).or_insert((DVec3::ZERO, 0));
            e.0 += m.center;
            e.1 += 1;
        }
        for (s, (sum, n)) in mesh {
            out.entry(s).or_insert(sum / n as f64);
        }
        out.into_iter().collect()
    }

    /// Height range of the points on screen: robust 2..98 % of all samples, at least
    /// 0.5 m tall (legacy `recomputeRange`).
    pub fn height_range(&self) -> Option<(f64, f64)> {
        let all: Vec<DVec3> = self.segments.values().flat_map(|i| i.samples.iter().copied()).collect();
        let n = all.len().div_ceil(points::MAX_BOUNDS_SAMPLES).max(1);
        let mut z: Vec<f64> = all.iter().step_by(n).map(|p| p.z).filter(|v| v.is_finite()).collect();
        z.sort_by(f64::total_cmp);
        let (lo, hi) = (points::quantile(&z, 0.02)?, points::quantile(&z, 0.98)?);
        Some((lo, hi.max(lo + 0.5)))
    }

    /// Box of every segment's core (the point panel's framing), or None.
    pub fn points_box(&self) -> Option<(DVec3, DVec3)> {
        self.segments.values().filter_map(|i| i.core).reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)))
    }

    /// An even sample of what is on screen, `per` points per segment.
    pub fn fit_samples(&self, per: usize) -> Vec<DVec3> {
        self.segments.values().flat_map(|i| i.samples.iter().take(per).copied()).collect()
    }

    pub fn total_points(&self) -> u64 {
        self.segments.values().map(|i| i.points).sum()
    }

    // ---- detections -------------------------------------------------------------------------

    /// Whether a segment's final (refined) level has arrived: detections wait for it, so a
    /// finding surfaces when its segment finishes processing, not while it is a rough shell.
    pub fn segment_final(&self, segment: u32) -> bool {
        self.store.levels().get(&segment) == Some(&model::Level::Refined)
    }

    pub fn detection_pos(&self, d: &payload::Detection) -> Option<DVec3> {
        Some(DVec3::from(self.site.as_ref()?.enu_of(d.gps.as_ref()?)))
    }

    /// Detections that just became revealable (final level in, placeable), in arrival order;
    /// each is reported once for the board's lifetime.
    pub fn reveal_new(&mut self) -> Vec<payload::Detection> {
        let mut out = Vec::new();
        for d in self.store.detections() {
            if self.revealed.contains(&d.id) || !self.segment_final(d.segment) || self.detection_pos(d).is_none() {
                continue;
            }
            out.push(d.clone());
        }
        self.revealed.extend(out.iter().map(|d| d.id.clone()));
        out
    }

    /// Markers on screen: revealed, still delivered, final level present. `(id, feet, person)`.
    pub fn markers(&self) -> Vec<(String, DVec3, bool)> {
        self.store
            .detections()
            .filter(|d| self.revealed.contains(&d.id) && self.segment_final(d.segment))
            .filter_map(|d| {
                let person = d.category() == payload::DetectionCategory::Person;
                Some((d.id.clone(), self.detection_pos(d)?, person))
            })
            .collect()
    }

    /// Markers to draw: [`Self::markers`] with same-kind detections within 3 m merged.
    pub fn marker_view(&self) -> Vec<(String, DVec3, bool)> {
        let sightings: Vec<markers::Sighting> = self
            .store
            .detections()
            .filter(|d| self.revealed.contains(&d.id) && self.segment_final(d.segment))
            .filter_map(|d| {
                let person = d.category() == payload::DetectionCategory::Person;
                Some((d.id.clone(), self.detection_pos(d)?, person, d.confidence))
            })
            .collect();
        markers::dedupe(&sightings)
    }

    /// The next detection for the camera: shown and not yet confirmed (legacy candidate rule).
    pub fn focus_candidate(&self) -> Option<(String, DVec3)> {
        self.markers().into_iter().find(|(id, _, _)| !self.confirmed.contains(id)).map(|(id, p, _)| (id, p))
    }

    /// The detection the camera is focusing or locked on, with its feet.
    pub fn focused(&self) -> Option<(String, DVec3)> {
        use skylens_viewer::camera::SyncState as S;
        if !matches!(self.cam.state, S::Focusing | S::Locked) {
            return None;
        }
        let id = self.cam.focused.as_ref()?;
        let d = self.store.detection(id)?;
        Some((id.clone(), self.detection_pos(d)?))
    }
}

/// Bounding-box centre of a mesh block.
fn mesh_center(b: &payload::MeshBlock) -> Option<DVec3> {
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for p in b.positions.chunks_exact(3) {
        let v = DVec3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
        lo = lo.min(v);
        hi = hi.max(v);
    }
    lo.x.is_finite().then(|| (lo + hi) / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pointblitz::io::Point;
    use crate::pointblitz::io::chunk::{FLAG_LAST_IN_GENERATION, encode_with};
    use prost::Message as _;

    fn site() -> viewer::Site {
        viewer::Site {
            anchor: Some(model::Gps { lat: 36.3685, lon: 127.3475, alt: 30.0 }),
            ..Default::default()
        }
    }

    fn points_chunk(seq: u64, segment: u32, level: model::Level, z: f32) -> model::Chunk {
        let pts: Vec<Point> = (0..100).map(|i| Point { position: [i as f32, (i % 7) as f32, z], color: [1, 2, 3] }).collect();
        model::Chunk {
            sequence: seq,
            kind: model::ChunkKind::Points as i32,
            segment,
            level: level as i32,
            generation: if level == model::Level::Refined { 1 } else { 0 },
            payload: encode_with(0, 0, FLAG_LAST_IN_GENERATION, &pts, 12),
            ..Default::default()
        }
    }

    fn detections_chunk(seq: u64, segment: u32, id: &str, person: bool) -> model::Chunk {
        let batch = payload::DetectionBatch {
            detections: vec![payload::Detection {
                id: id.into(),
                category: if person { payload::DetectionCategory::Person } else { payload::DetectionCategory::Fire } as i32,
                gps: Some(model::Gps { lat: 36.3686, lon: 127.3476, alt: 30.0 }),
                confidence: 0.8,
                label: "생존자".into(),
                segment,
                ..Default::default()
            }],
        };
        model::Chunk {
            sequence: seq,
            kind: model::ChunkKind::Detections as i32,
            segment,
            payload: batch.encode_to_vec(),
            ..Default::default()
        }
    }

    fn board() -> Board {
        let mut b = Board::new(true);
        b.site = SiteFrame::from_site(&site());
        b
    }

    #[test]
    fn detections_wait_for_the_final_level_and_reveal_once() {
        let mut b = board();
        b.chunk(&detections_chunk(1, 0, "p1", true));
        assert!(b.reveal_new().is_empty());
        b.chunk(&points_chunk(2, 0, model::Level::Preview, 0.0));
        assert!(b.reveal_new().is_empty(), "a preview is not the final level");
        assert!(b.markers().is_empty());
        b.chunk(&points_chunk(3, 0, model::Level::Refined, 0.0));
        let r = b.reveal_new();
        assert_eq!(r.len(), 1);
        assert_eq!(b.markers().len(), 1);
        assert!(b.reveal_new().is_empty(), "once");
        assert_eq!(b.focus_candidate().map(|(id, _)| id), Some("p1".into()));
        b.confirmed.insert("p1".into());
        assert!(b.focus_candidate().is_none());
        // A reconnect replays; the reveal is not reported again and the marker waits again.
        b.reconnected();
        b.chunk(&detections_chunk(1, 0, "p1", true));
        assert!(b.markers().is_empty());
        b.chunk(&points_chunk(3, 0, model::Level::Refined, 0.0));
        assert!(b.reveal_new().is_empty());
        assert_eq!(b.markers().len(), 1);
    }

    #[test]
    fn segments_are_sampled_and_framed() {
        let mut b = board();
        assert!(!b.has_geometry());
        b.chunk(&points_chunk(1, 0, model::Level::Preview, 0.0));
        b.chunk(&points_chunk(2, 1, model::Level::Preview, 4.0));
        b.refresh();
        assert!(b.has_geometry());
        assert_eq!(b.segments.len(), 2);
        assert_eq!(b.total_points(), 200);
        let (lo, hi) = b.height_range().unwrap();
        assert!(lo.abs() < 1e-3 && (hi - 4.0).abs() < 1e-3);
        assert!(b.framing.is_some());
        assert_eq!(b.segment_centers().len(), 2);
        assert_eq!(b.fit_samples(10).len(), 20);
        assert!(b.points_box().is_some());
        assert_eq!(b.local_chunks, 2);
    }

    /// A slot message adding `n` points per zone (zone z at height 2z), ids from `first`.
    fn slots_chunk(seq: u32, first: u32, zones: &[u8], n: u32) -> model::Chunk {
        use crate::pointblitz::io::slots;
        let mut pack = Vec::new();
        let mut id = first;
        for &z in zones {
            for i in 0..n {
                let position = [i as f32 * 0.5, (i % 9) as f32, 2.0 * f32::from(z)];
                slots::Record { id, position, color: [7, 7, 7], zone: z }.write(&mut pack);
                id += 1;
            }
        }
        let header = slots::SlotHeader { seq, n_add: id - first, n_move: 0, n_remove: 0, capacity: id };
        model::Chunk {
            sequence: u64::from(seq),
            kind: model::ChunkKind::PointSlots as i32,
            segment: u32::from(*zones.last().unwrap()),
            level: model::Level::Refined as i32,
            version: seq,
            generation: 1,
            payload: slots::encode(&header, &pack).unwrap(),
            ..Default::default()
        }
    }

    #[test]
    fn slot_zones_are_read_like_segments() {
        let mut b = board();
        b.chunk(&detections_chunk(1, 1, "p1", true));
        b.chunk(&slots_chunk(1, 0, &[0], 200));
        b.refresh();
        assert!(b.has_geometry());
        assert_eq!(b.point_segments().into_iter().collect::<Vec<_>>(), vec![0]);
        assert_eq!(b.total_points(), 200);
        assert!(b.reveal_new().is_empty(), "zone 1 has not arrived");
        b.chunk(&slots_chunk(2, 200, &[1], 100));
        b.refresh();
        assert_eq!(b.segments.len(), 2);
        assert_eq!(b.total_points(), 300);
        assert!(b.segments.values().all(|i| i.refined));
        let (lo, hi) = b.height_range().unwrap();
        assert!(lo.abs() < 1e-6 && (hi - 2.0).abs() < 1e-6, "{lo} {hi}");
        assert_eq!(b.segment_centers().len(), 2);
        assert_eq!(b.reveal_new().len(), 1, "a slot zone arrives at the final level");
        assert!(b.framing.is_some());
        assert!(b.point_size_m >= 0.05 && b.point_size_m <= 1.0);
        assert_eq!(b.local_chunks, 2);
        // Nothing changed: the cached samples stay.
        let before = b.segments.clone();
        b.refresh();
        assert_eq!(b.segments, before);
    }

    #[test]
    fn slot_chunks_feed_the_live_log_in_order_and_start_a_new_run_on_a_new_stream() {
        let mut b = board();
        let in_stream = |mut c: model::Chunk, stream: u32| {
            c.generation = stream;
            c
        };
        // Seq 2 first: the store holds it behind the gap, the log gets both once seq 1 lands.
        b.chunk(&in_stream(slots_chunk(2, 10, &[1], 5), 5));
        assert_eq!(b.live.len(), 0);
        b.chunk(&in_stream(slots_chunk(1, 0, &[0], 10), 5));
        assert_eq!((b.live.len(), b.live.applied(), b.live.run_id()), (2, 2, Some(5)));
        let e = b.live.entries();
        assert_eq!((e[0].seq, e[0].zone, e[0].n_add), (1, 0, 10));
        assert_eq!((e[1].seq, e[1].zone, e[1].total_points), (2, 1, 15));
        // A repeat is not logged twice.
        b.chunk(&in_stream(slots_chunk(2, 10, &[1], 5), 5));
        assert_eq!(b.live.len(), 2);
        // The layers drain the log through their own cursors.
        let ops = b.live_ops(LiveView::Main);
        assert_eq!(ops.len(), 2);
        assert!(b.live_ops(LiveView::Main).is_empty());
        // A newer stream (a new model run) empties the log and tells the layer to reset.
        b.chunk(&in_stream(slots_chunk(1, 0, &[0], 3), 6));
        assert_eq!((b.live.len(), b.live.run_id(), b.live.table().len()), (1, Some(6), 3));
        assert!(matches!(b.live_ops(LiveView::Main)[0], crate::delta_log::LayerOp::Reset));
        // An older stream is dropped.
        b.chunk(&in_stream(slots_chunk(2, 3, &[0], 3), 5));
        assert_eq!(b.live.len(), 1);
        // A reconnect: the core replays from the start.
        b.reconnected();
        assert_eq!(b.live.len(), 0);
        b.chunk(&in_stream(slots_chunk(1, 0, &[0], 4), 6));
        assert_eq!(b.live.len(), 1);
    }

    #[test]
    fn the_chunk_fields_set_the_row_kind_and_clock() {
        let mut b = board();
        let mut c = slots_chunk(1, 0, &[0], 5);
        c.slot_kind = 1;
        c.slot_t_ms = 1500;
        b.chunk(&c);
        let header = crate::pointblitz::io::slots::SlotHeader { seq: 2, n_add: 0, n_move: 0, n_remove: 0, capacity: 5 };
        let mut f = model::Chunk { sequence: 2, version: 2, generation: 1, payload: crate::pointblitz::io::slots::encode(&header, &[]).unwrap(), ..c.clone() };
        f.slot_kind = 3;
        f.slot_t_ms = 9000;
        b.chunk(&f);
        let e = b.live.entries();
        assert_eq!((e[0].kind, e[0].t_ms), (crate::delta_log::Kind::Refined, 1500.0));
        assert_eq!((e[1].kind, e[1].t_ms), (crate::delta_log::Kind::Final, 9000.0));
        assert!(b.live.is_final());
    }

    #[test]
    fn telemetry_needs_the_site_and_the_first_drone_is_active() {
        let mut b = Board::new(true);
        let t = model::DroneTelemetry { drone_id: 2, gps: Some(model::Gps { lat: 36.3685, lon: 127.3475, alt: 40.0 }), ..Default::default() };
        b.telemetry(&t);
        assert!(b.drones.is_empty(), "no frame before the site");
        b.site = SiteFrame::from_site(&site());
        b.telemetry(&t);
        b.telemetry(&model::DroneTelemetry { drone_id: 3, ..t.clone() });
        assert_eq!(b.active_drone, Some(2));
        let a = b.active().unwrap();
        assert!((a.pos.z - 10.0).abs() < 1e-6);
        b.assign_route(&viewer::AssignRoute { drone_id: 2, waypoints: vec![model::Gps::default()], r#loop: false });
        assert!(b.route.is_none(), "one waypoint is no route");
    }
}
