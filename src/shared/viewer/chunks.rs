//! Chunk store: what the core has delivered, with the replacement rules applied (SPEC §5).
//!
//! - **Points** (`CHUNK_KIND_POINTS`, pointblitz chunk v2) are kept per reconstruction segment.
//!   Preview chunks only add points. A refined chunk replaces the preview of its segment, and a
//!   newer refined generation replaces an older one. Like pointblitz's own `Scene`, a replacing
//!   generation stays pending until the chunk that completes its coarse first pass (or its last
//!   chunk) arrives, so the segment never goes blank in between. Every replacement gives the
//!   segment a new `epoch`, which is how the GPU side knows to rebuild instead of append.
//! - **Mesh blocks** are kept per block id; only a newer `version` replaces a block and a block
//!   with zero triangles removes it (the removal is remembered, so an older version cannot
//!   resurrect it).
//! - **Detections** are kept per id (an update replaces). They are visible only once the segment
//!   they were found in has arrived (SPEC §7); [`Applied::revealed`] lists the ids that just
//!   became visible.
//! - **Terrain tiles** and **building sets** are kept per cell; a later delivery replaces the
//!   cell unless it carries an older non-zero `version`.
//! - A segment has *arrived* once any reconstruction chunk (points or mesh) of it was received.
//!
//! The store keeps payloads so the 3D view can be rebuilt (GPU recreated, display change) without
//! asking the core again. Call [`ChunkStore::clear`] when the link reconnects: the core replays.

use crate::protocol;
use pointblitz::io::chunk::{FLAG_FIRST_PASS_COMPLETE, FLAG_LAST_IN_GENERATION, decode_header};
use skylens_shared::proto::model::v1 as model;
use skylens_shared::proto::payload::v1 as payload;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;

/// Chunk bytes, shared between the store and the GPU uploads.
pub type Payload = Rc<[u8]>;

/// Ordering key of a delivery within a segment: previews rank below every refined generation,
/// and all previews share one key because they only add points.
type Key = (u8, u32);

fn key(level: model::Level, generation: u32) -> Key {
    match level {
        model::Level::Refined => (1, generation),
        _ => (0, 0),
    }
}

fn level_rank(level: model::Level) -> u8 {
    match level {
        model::Level::Unspecified => 0,
        model::Level::Preview => 1,
        model::Level::Refined => 2,
    }
}

/// The point chunks of one reconstruction segment.
#[derive(Debug, Clone)]
pub struct SegmentPoints {
    shown_key: Option<Key>,
    shown: Vec<Payload>,
    pending: Option<(Key, Vec<Payload>)>,
    epoch: u64,
}

impl SegmentPoints {
    /// Chunks on screen, in arrival order.
    pub fn payloads(&self) -> &[Payload] {
        &self.shown
    }

    /// Changes whenever the shown set is replaced (not when chunks are appended).
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Whether the shown chunks are a refined generation.
    pub fn is_refined(&self) -> bool {
        self.shown_key.is_some_and(|(r, _)| r == 1)
    }

    /// Points on screen, from the chunk headers.
    pub fn point_count(&self) -> u64 {
        self.shown
            .iter()
            .filter_map(|p| decode_header(p).ok())
            .map(|h| u64::from(h.point_count))
            .sum()
    }

    /// Centre of the shown chunks' bounding boxes in ENU, if any.
    pub fn center(&self) -> Option<[f64; 3]> {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for h in self.shown.iter().filter_map(|p| decode_header(p).ok()) {
            for i in 0..3 {
                lo[i] = lo[i].min(h.origin[i] + f64::from(h.bbox_min[i]));
                hi[i] = hi[i].max(h.origin[i] + f64::from(h.bbox_max[i]));
            }
        }
        lo[0].is_finite().then(|| [0, 1, 2].map(|i| (lo[i] + hi[i]) / 2.0))
    }
}

/// One mesh block as delivered.
#[derive(Debug, Clone)]
pub struct MeshEntry {
    pub version: u32,
    pub segment: u32,
    pub level: model::Level,
    pub block: Rc<payload::MeshBlock>,
    /// Unique per stored entry; the GPU side re-uploads when it changes.
    pub stamp: u64,
}

#[derive(Debug, Clone)]
pub struct TerrainEntry {
    pub version: u32,
    pub tile: Rc<payload::TerrainTile>,
    pub stamp: u64,
}

#[derive(Debug, Clone)]
pub struct BuildingEntry {
    pub version: u32,
    pub set: Rc<payload::BuildingSet>,
    pub stamp: u64,
}

/// What one chunk did.
#[derive(Debug, Clone, PartialEq)]
pub enum Change {
    /// Points added to what the segment shows.
    PointsAppended { segment: u32 },
    /// The segment now shows a newer delivery (new epoch).
    PointsReplaced { segment: u32 },
    /// Stored for a newer delivery that is not complete yet.
    PointsPending { segment: u32 },
    MeshUpserted { id: [i32; 3] },
    MeshRemoved { id: [i32; 3] },
    /// Detection ids received (new or updated), visible or not yet.
    Detections { ids: Vec<String> },
    Terrain { cell: [i32; 2] },
    Buildings { cell: [i32; 2] },
    /// A live point delta of the current run, in seq order (see [`Applied::delta`]). `new_run`:
    /// it started a new run, so the legacy point state was dropped.
    PointsDelta { seq: u32, zone: u32, new_run: bool },
    /// Older than what is stored; dropped.
    Stale,
    /// Undecodable or of an unknown kind; counted in [`ChunkStore::invalid`].
    Invalid(String),
}

/// Result of [`ChunkStore::apply`].
#[derive(Debug, Clone, PartialEq)]
pub struct Applied {
    pub change: Change,
    /// The segment arrived with this chunk (its first reconstruction chunk).
    pub newly_arrived: Option<u32>,
    /// Detections that just became visible (their segment arrived, or they arrived for an
    /// already arrived segment), in arrival order. Each id is reported once.
    pub revealed: Vec<String>,
    /// The decoded delta of a [`Change::PointsDelta`], for the board's delta log.
    pub delta: Option<Rc<payload::PointsDelta>>,
}

impl Applied {
    fn of(change: Change) -> Self {
        Applied { change, newly_arrived: None, revealed: Vec::new(), delta: None }
    }
}

#[derive(Debug, Default)]
pub struct ChunkStore {
    points: BTreeMap<u32, SegmentPoints>,
    meshes: BTreeMap<[i32; 3], MeshEntry>,
    /// Version at which a block was removed.
    mesh_removed: HashMap<[i32; 3], u32>,
    detections: BTreeMap<String, payload::Detection>,
    detection_order: Vec<String>,
    revealed: BTreeSet<String>,
    terrain: BTreeMap<[i32; 2], TerrainEntry>,
    buildings: BTreeMap<[i32; 2], BuildingEntry>,
    arrived: BTreeSet<u32>,
    levels: BTreeMap<u32, model::Level>,
    /// Run id and last seq of the point deltas seen. Once a delta has arrived, the per-segment
    /// preview/refined point chunks are ignored: the delta stream is the one source of points.
    delta_run: Option<u32>,
    delta_seq: u32,
    /// Never reset, so a segment recreated after `clear` gets an epoch the GPU has not seen.
    next_stamp: u64,
    pub chunks_received: u64,
    pub invalid: u64,
    /// Highest `Chunk.sequence` seen.
    pub last_sequence: u64,
}

impl ChunkStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn stamp(&mut self) -> u64 {
        self.next_stamp += 1;
        self.next_stamp
    }

    /// Forgets everything (the link reconnected and the core replays). Stamps keep counting.
    pub fn clear(&mut self) {
        let next_stamp = self.next_stamp;
        *self = ChunkStore { next_stamp, ..Default::default() };
    }

    pub fn apply(&mut self, chunk: &model::Chunk) -> Applied {
        self.chunks_received += 1;
        self.last_sequence = self.last_sequence.max(chunk.sequence);
        let mut out = match chunk.kind() {
            model::ChunkKind::Points => self.apply_points(chunk),
            model::ChunkKind::MeshBlock => self.apply_mesh(chunk),
            model::ChunkKind::Detections => self.apply_detections(chunk),
            model::ChunkKind::Terrain => self.apply_terrain(chunk),
            model::ChunkKind::Buildings => self.apply_buildings(chunk),
            model::ChunkKind::PointsDelta => self.apply_points_delta(chunk),
            model::ChunkKind::Unspecified => Applied::of(Change::Invalid(format!("unknown chunk kind {}", chunk.kind))),
        };
        if matches!(out.change, Change::Invalid(_)) {
            self.invalid += 1;
        }
        if let Some(seg) = out.newly_arrived {
            out.revealed.extend(self.reveal_segment(seg));
        }
        out
    }

    /// A live point delta: the first of a new run drops the old state, an older or repeated seq is
    /// stale, and a zone's first refined/replace delta marks that zone as arrived (its detections
    /// reveal). The delta itself goes to the board's log through [`Applied::delta`].
    fn apply_points_delta(&mut self, chunk: &model::Chunk) -> Applied {
        let d = match skylens_shared::delta::from_chunk(chunk) {
            Ok(d) => d,
            Err(e) => return Applied::of(Change::Invalid(e.to_string())),
        };
        let new_run = self.delta_run.is_some_and(|r| r != d.run);
        let first = self.delta_run.is_none();
        if !new_run && !first && d.seq <= self.delta_seq && !d.snapshot {
            return Applied::of(Change::Stale);
        }
        if new_run || first {
            // The legacy per-segment clouds (if any came first) make way for the delta stream.
            self.points.clear();
            self.arrived.clear();
            self.levels.clear();
            self.revealed.clear();
        }
        self.delta_run = Some(d.run);
        self.delta_seq = d.seq;
        let mut out = Applied::of(Change::PointsDelta { seq: d.seq, zone: d.zone, new_run });
        if matches!(d.kind(), payload::DeltaKind::Refined | payload::DeltaKind::Replace) && !d.adds.is_empty() {
            out.newly_arrived = self.mark_arrived(d.zone, model::Level::Refined);
        }
        out.delta = Some(Rc::new(d));
        out
    }

    /// Whether the board is fed by point deltas (the legacy point chunks are ignored).
    pub fn delta_mode(&self) -> bool {
        self.delta_run.is_some()
    }

    fn apply_points(&mut self, chunk: &model::Chunk) -> Applied {
        if self.delta_mode() {
            return Applied::of(Change::Stale);
        }
        let header = match decode_header(&chunk.payload) {
            Ok(h) => h,
            Err(e) => return Applied::of(Change::Invalid(e.to_string())),
        };
        let completes = header.flags & (FLAG_LAST_IN_GENERATION | FLAG_FIRST_PASS_COMPLETE) != 0;
        let payload: Payload = Rc::from(chunk.payload.as_slice());
        let k = key(chunk.level(), chunk.generation);
        let segment = chunk.segment;
        let fresh_epoch = self.stamp();
        let seg = self.points.entry(segment).or_insert_with(|| SegmentPoints {
            shown_key: None,
            shown: Vec::new(),
            pending: None,
            epoch: fresh_epoch,
        });
        let change = match seg.shown_key {
            None => {
                seg.shown_key = Some(k);
                seg.shown.push(payload);
                Change::PointsAppended { segment }
            }
            Some(cur) if k < cur => Change::Stale,
            Some(cur) if k == cur => {
                seg.shown.push(payload);
                Change::PointsAppended { segment }
            }
            Some(_) => {
                let mut stale = false;
                match &mut seg.pending {
                    Some((pk, v)) if *pk == k => v.push(payload),
                    Some((pk, _)) if *pk > k => stale = true,
                    _ => seg.pending = Some((k, vec![payload])),
                }
                if stale {
                    Change::Stale
                } else if completes {
                    let (pk, v) = seg.pending.take().expect("pending set above");
                    seg.shown_key = Some(pk);
                    seg.shown = v;
                    seg.epoch = fresh_epoch;
                    Change::PointsReplaced { segment }
                } else {
                    Change::PointsPending { segment }
                }
            }
        };
        let mut out = Applied::of(change);
        if matches!(out.change, Change::PointsAppended { .. } | Change::PointsReplaced { .. }) {
            out.newly_arrived = self.mark_arrived(segment, chunk.level());
        }
        out
    }

    fn apply_mesh(&mut self, chunk: &model::Chunk) -> Applied {
        let block = match protocol::decode_mesh_block(&chunk.payload) {
            Ok(b) => b,
            Err(e) => return Applied::of(Change::Invalid(e.to_string())),
        };
        let id_src = if chunk.block_id.len() == 3 { &chunk.block_id } else { &block.id };
        let Ok(id) = <[i32; 3]>::try_from(id_src.as_slice()) else {
            return Applied::of(Change::Invalid("mesh block id is not [x, y, z]".into()));
        };
        let version = if chunk.version != 0 { chunk.version } else { block.version };
        let known = self
            .meshes
            .get(&id)
            .map(|e| e.version)
            .or_else(|| self.mesh_removed.get(&id).copied());
        if known.is_some_and(|v| v >= version) {
            return Applied::of(Change::Stale);
        }
        let change = if block.indices.is_empty() {
            self.meshes.remove(&id);
            self.mesh_removed.insert(id, version);
            Change::MeshRemoved { id }
        } else {
            let stamp = self.stamp();
            self.mesh_removed.remove(&id);
            self.meshes.insert(
                id,
                MeshEntry { version, segment: chunk.segment, level: chunk.level(), block: Rc::new(block), stamp },
            );
            Change::MeshUpserted { id }
        };
        let mut out = Applied::of(change);
        out.newly_arrived = self.mark_arrived(chunk.segment, chunk.level());
        out
    }

    fn apply_detections(&mut self, chunk: &model::Chunk) -> Applied {
        let batch = match protocol::decode_detections(&chunk.payload) {
            Ok(b) => b,
            Err(e) => return Applied::of(Change::Invalid(e.to_string())),
        };
        let mut ids = Vec::new();
        let mut revealed = Vec::new();
        for d in batch.detections {
            if d.id.is_empty() {
                continue;
            }
            let id = d.id.clone();
            if !self.detections.contains_key(&id) {
                self.detection_order.push(id.clone());
            }
            let segment = d.segment;
            self.detections.insert(id.clone(), d);
            if self.arrived.contains(&segment) && self.revealed.insert(id.clone()) {
                revealed.push(id.clone());
            }
            ids.push(id);
        }
        let mut out = Applied::of(Change::Detections { ids });
        out.revealed = revealed;
        out
    }

    fn cell_of(block_id: &[i32], payload_cell: &[i32]) -> Option<[i32; 2]> {
        let src = if block_id.len() >= 2 { &block_id[..2] } else { payload_cell.get(..2)? };
        <[i32; 2]>::try_from(src).ok()
    }

    fn apply_terrain(&mut self, chunk: &model::Chunk) -> Applied {
        let tile = match protocol::decode_terrain_tile(&chunk.payload) {
            Ok(t) => t,
            Err(e) => return Applied::of(Change::Invalid(e.to_string())),
        };
        let Some(cell) = Self::cell_of(&chunk.block_id, &tile.cell) else {
            return Applied::of(Change::Invalid("terrain tile has no [x, y] cell".into()));
        };
        if self.terrain.get(&cell).is_some_and(|e| chunk.version != 0 && e.version > chunk.version) {
            return Applied::of(Change::Stale);
        }
        let stamp = self.stamp();
        self.terrain.insert(cell, TerrainEntry { version: chunk.version, tile: Rc::new(tile), stamp });
        Applied::of(Change::Terrain { cell })
    }

    fn apply_buildings(&mut self, chunk: &model::Chunk) -> Applied {
        let set = match protocol::decode_building_set(&chunk.payload) {
            Ok(s) => s,
            Err(e) => return Applied::of(Change::Invalid(e.to_string())),
        };
        let Some(cell) = Self::cell_of(&chunk.block_id, &set.cell) else {
            return Applied::of(Change::Invalid("building set has no [x, y] cell".into()));
        };
        if self.buildings.get(&cell).is_some_and(|e| chunk.version != 0 && e.version > chunk.version) {
            return Applied::of(Change::Stale);
        }
        let stamp = self.stamp();
        self.buildings.insert(cell, BuildingEntry { version: chunk.version, set: Rc::new(set), stamp });
        Applied::of(Change::Buildings { cell })
    }

    fn mark_arrived(&mut self, segment: u32, level: model::Level) -> Option<u32> {
        let entry = self.levels.entry(segment).or_insert(model::Level::Unspecified);
        if level_rank(level) > level_rank(*entry) {
            *entry = level;
        }
        self.arrived.insert(segment).then_some(segment)
    }

    /// Detections of a segment that just arrived, in arrival order.
    fn reveal_segment(&mut self, segment: u32) -> Vec<String> {
        let mut out = Vec::new();
        for id in &self.detection_order {
            if self.detections.get(id).is_some_and(|d| d.segment == segment) && self.revealed.insert(id.clone()) {
                out.push(id.clone());
            }
        }
        out
    }

    // ---- points ---------------------------------------------------------------------------

    /// Point chunks per segment, ordered by segment.
    pub fn segments(&self) -> impl Iterator<Item = (u32, &SegmentPoints)> {
        self.points.iter().map(|(s, p)| (*s, p))
    }

    pub fn segment(&self, segment: u32) -> Option<&SegmentPoints> {
        self.points.get(&segment)
    }

    // ---- meshes ---------------------------------------------------------------------------

    pub fn mesh_blocks(&self) -> impl Iterator<Item = ([i32; 3], &MeshEntry)> {
        self.meshes.iter().map(|(id, e)| (*id, e))
    }

    pub fn mesh_block(&self, id: [i32; 3]) -> Option<&MeshEntry> {
        self.meshes.get(&id)
    }

    pub fn mesh_block_count(&self) -> usize {
        self.meshes.len()
    }

    // ---- detections -----------------------------------------------------------------------

    /// Every detection received, in first-arrival order, visible or not.
    pub fn detections(&self) -> impl Iterator<Item = &payload::Detection> {
        self.detection_order.iter().filter_map(|id| self.detections.get(id))
    }

    /// Detections whose segment has arrived, in first-arrival order.
    pub fn visible_detections(&self) -> impl Iterator<Item = &payload::Detection> {
        self.detections().filter(|d| self.arrived.contains(&d.segment))
    }

    pub fn detection(&self, id: &str) -> Option<&payload::Detection> {
        self.detections.get(id)
    }

    pub fn is_detection_visible(&self, id: &str) -> bool {
        self.detections.get(id).is_some_and(|d| self.arrived.contains(&d.segment))
    }

    // ---- terrain and buildings ------------------------------------------------------------

    pub fn terrain_tiles(&self) -> impl Iterator<Item = ([i32; 2], &TerrainEntry)> {
        self.terrain.iter().map(|(c, e)| (*c, e))
    }

    pub fn terrain_tile(&self, cell: [i32; 2]) -> Option<&TerrainEntry> {
        self.terrain.get(&cell)
    }

    pub fn building_sets(&self) -> impl Iterator<Item = ([i32; 2], &BuildingEntry)> {
        self.buildings.iter().map(|(c, e)| (*c, e))
    }

    pub fn building_set(&self, cell: [i32; 2]) -> Option<&BuildingEntry> {
        self.buildings.get(&cell)
    }

    // ---- arrival --------------------------------------------------------------------------

    pub fn arrived(&self) -> &BTreeSet<u32> {
        &self.arrived
    }

    pub fn has_arrived(&self, segment: u32) -> bool {
        self.arrived.contains(&segment)
    }

    /// Highest reconstruction level delivered per segment (points or mesh).
    pub fn levels(&self) -> &BTreeMap<u32, model::Level> {
        &self.levels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pointblitz::io::Point;
    use pointblitz::io::chunk::encode_with;
    use prost::Message as _;

    fn pts(n: usize) -> Vec<Point> {
        (0..n).map(|i| Point { position: [i as f32, 0.0, 0.0], color: [10, 20, 30] }).collect()
    }

    fn recon(segment: u32, level: model::Level, generation: u32, flags: u32, n: usize) -> model::Chunk {
        model::Chunk {
            kind: model::ChunkKind::Points as i32,
            segment,
            level: level as i32,
            generation,
            payload: encode_with(generation, 0, flags, &pts(n), 12),
            ..Default::default()
        }
    }

    fn mesh(id: [i32; 3], version: u32, triangles: usize, segment: u32) -> model::Chunk {
        let block = payload::MeshBlock {
            id: id.to_vec(),
            version,
            positions: vec![0.0; 9],
            colors: vec![255; 9],
            indices: (0..triangles as u32 * 3).map(|i| i % 3).collect(),
        };
        model::Chunk {
            kind: model::ChunkKind::MeshBlock as i32,
            segment,
            level: model::Level::Preview as i32,
            block_id: id.to_vec(),
            version,
            payload: block.encode_to_vec(),
            ..Default::default()
        }
    }

    fn detections(list: &[(&str, u32, payload::DetectionCategory)]) -> model::Chunk {
        let batch = payload::DetectionBatch {
            detections: list
                .iter()
                .map(|(id, segment, cat)| payload::Detection {
                    id: (*id).into(),
                    segment: *segment,
                    category: *cat as i32,
                    ..Default::default()
                })
                .collect(),
        };
        model::Chunk {
            kind: model::ChunkKind::Detections as i32,
            payload: batch.encode_to_vec(),
            ..Default::default()
        }
    }

    fn shown(store: &ChunkStore, segment: u32) -> usize {
        store.segment(segment).map_or(0, |s| s.payloads().len())
    }

    #[test]
    fn previews_append_and_keep_the_epoch() {
        let mut s = ChunkStore::new();
        let first = s.apply(&recon(0, model::Level::Preview, 0, 0, 4));
        assert_eq!(first.change, Change::PointsAppended { segment: 0 });
        assert_eq!(first.newly_arrived, Some(0));
        let epoch = s.segment(0).unwrap().epoch();
        for _ in 0..2 {
            let a = s.apply(&recon(0, model::Level::Preview, 0, 0, 4));
            assert_eq!(a.change, Change::PointsAppended { segment: 0 });
            assert_eq!(a.newly_arrived, None);
        }
        assert_eq!(shown(&s, 0), 3);
        assert_eq!(s.segment(0).unwrap().epoch(), epoch);
        assert_eq!(s.segment(0).unwrap().point_count(), 12);
        assert!(!s.segment(0).unwrap().is_refined());
    }

    #[test]
    fn refined_replaces_the_preview_of_its_segment_only() {
        let mut s = ChunkStore::new();
        s.apply(&recon(0, model::Level::Preview, 0, 0, 4));
        s.apply(&recon(0, model::Level::Preview, 0, 0, 4));
        s.apply(&recon(1, model::Level::Preview, 0, 0, 4));
        let epoch = s.segment(0).unwrap().epoch();

        let a = s.apply(&recon(0, model::Level::Refined, 1, 0, 8));
        assert_eq!(a.change, Change::PointsPending { segment: 0 });
        assert_eq!(shown(&s, 0), 2);

        let a = s.apply(&recon(0, model::Level::Refined, 1, FLAG_LAST_IN_GENERATION, 8));
        assert_eq!(a.change, Change::PointsReplaced { segment: 0 });
        assert_eq!(shown(&s, 0), 2);
        assert_ne!(s.segment(0).unwrap().epoch(), epoch);
        assert!(s.segment(0).unwrap().is_refined());
        assert_eq!(shown(&s, 1), 1);

        // Late previews of a refined segment are stale.
        assert_eq!(s.apply(&recon(0, model::Level::Preview, 0, 0, 4)).change, Change::Stale);
        assert_eq!(s.levels()[&0], model::Level::Refined);
        assert_eq!(s.levels()[&1], model::Level::Preview);
    }

    #[test]
    fn newer_generation_replaces_and_older_is_stale() {
        let mut s = ChunkStore::new();
        s.apply(&recon(2, model::Level::Refined, 5, FLAG_LAST_IN_GENERATION, 4));
        assert_eq!(s.apply(&recon(2, model::Level::Refined, 5, 0, 4)).change, Change::PointsAppended { segment: 2 });
        assert_eq!(s.apply(&recon(2, model::Level::Refined, 4, FLAG_LAST_IN_GENERATION, 4)).change, Change::Stale);
        assert_eq!(
            s.apply(&recon(2, model::Level::Refined, 6, FLAG_FIRST_PASS_COMPLETE, 4)).change,
            Change::PointsReplaced { segment: 2 }
        );
        assert_eq!(shown(&s, 2), 1);
    }

    #[test]
    fn a_newer_pending_generation_drops_an_older_pending_one() {
        let mut s = ChunkStore::new();
        s.apply(&recon(0, model::Level::Preview, 0, 0, 4));
        s.apply(&recon(0, model::Level::Refined, 1, 0, 4));
        s.apply(&recon(0, model::Level::Refined, 2, 0, 4));
        assert_eq!(s.apply(&recon(0, model::Level::Refined, 1, FLAG_LAST_IN_GENERATION, 4)).change, Change::Stale);
        assert_eq!(
            s.apply(&recon(0, model::Level::Refined, 2, FLAG_LAST_IN_GENERATION, 4)).change,
            Change::PointsReplaced { segment: 0 }
        );
        assert_eq!(shown(&s, 0), 2);
    }

    #[test]
    fn invalid_points_are_counted_not_applied() {
        let mut s = ChunkStore::new();
        let mut c = recon(0, model::Level::Preview, 0, 0, 4);
        c.payload.truncate(10);
        assert!(matches!(s.apply(&c).change, Change::Invalid(_)));
        assert_eq!(s.invalid, 1);
        assert!(s.arrived().is_empty());
        let unknown = model::Chunk::default();
        assert!(matches!(s.apply(&unknown).change, Change::Invalid(_)));
        assert_eq!(s.invalid, 2);
    }

    #[test]
    fn mesh_blocks_keep_only_newer_versions_and_remove_empty_ones() {
        let mut s = ChunkStore::new();
        assert_eq!(s.apply(&mesh([0, 0, 0], 2, 1, 4)).change, Change::MeshUpserted { id: [0, 0, 0] });
        assert_eq!(s.apply(&mesh([0, 0, 0], 2, 1, 4)).change, Change::Stale);
        assert_eq!(s.apply(&mesh([0, 0, 0], 1, 1, 4)).change, Change::Stale);
        s.apply(&mesh([1, 0, 0], 1, 1, 4));
        assert_eq!(s.mesh_block_count(), 2);
        let stamp = s.mesh_block([0, 0, 0]).unwrap().stamp;

        assert_eq!(s.apply(&mesh([0, 0, 0], 3, 0, 4)).change, Change::MeshRemoved { id: [0, 0, 0] });
        assert_eq!(s.mesh_block_count(), 1);
        // An older version cannot bring a removed block back; a newer one can.
        assert_eq!(s.apply(&mesh([0, 0, 0], 2, 1, 4)).change, Change::Stale);
        assert_eq!(s.apply(&mesh([0, 0, 0], 4, 1, 4)).change, Change::MeshUpserted { id: [0, 0, 0] });
        assert_ne!(s.mesh_block([0, 0, 0]).unwrap().stamp, stamp);
        assert!(s.has_arrived(4));
    }

    #[test]
    fn detections_show_only_for_arrived_segments_and_reveal_once() {
        use payload::DetectionCategory::{Fire, Person};
        let mut s = ChunkStore::new();
        let a = s.apply(&detections(&[("p1", 3, Person), ("f1", 3, Fire), ("p2", 5, Person)]));
        assert_eq!(a.change, Change::Detections { ids: vec!["p1".into(), "f1".into(), "p2".into()] });
        assert!(a.revealed.is_empty());
        assert_eq!(s.visible_detections().count(), 0);
        assert!(!s.has_arrived(3));

        // Segment 3 arrives: its two detections are revealed, in arrival order.
        let a = s.apply(&recon(3, model::Level::Preview, 0, 0, 1));
        assert_eq!(a.newly_arrived, Some(3));
        assert_eq!(a.revealed, vec!["p1".to_string(), "f1".to_string()]);
        assert!(s.is_detection_visible("p1"));
        assert!(!s.is_detection_visible("p2"));

        // An update of a visible detection replaces it without revealing it again.
        let a = s.apply(&detections(&[("p1", 3, Person)]));
        assert!(a.revealed.is_empty());
        assert_eq!(s.detections().count(), 3);

        // A detection for an arrived segment is revealed on arrival.
        let a = s.apply(&detections(&[("p3", 3, Person)]));
        assert_eq!(a.revealed, vec!["p3".to_string()]);
        let visible: Vec<&str> = s.visible_detections().map(|d| d.id.as_str()).collect();
        assert_eq!(visible, vec!["p1", "f1", "p3"]);
    }

    #[test]
    fn terrain_and_buildings_are_kept_per_cell() {
        let tile = |version: u32, cols: u32| model::Chunk {
            kind: model::ChunkKind::Terrain as i32,
            block_id: vec![2, -1],
            version,
            payload: payload::TerrainTile { cell: vec![2, -1], cols, rows: 1, heights: vec![0.0; cols as usize], ..Default::default() }
                .encode_to_vec(),
            ..Default::default()
        };
        let mut s = ChunkStore::new();
        assert_eq!(s.apply(&tile(2, 2)).change, Change::Terrain { cell: [2, -1] });
        assert_eq!(s.apply(&tile(1, 3)).change, Change::Stale);
        assert_eq!(s.apply(&tile(0, 4)).change, Change::Terrain { cell: [2, -1] });
        assert_eq!(s.terrain_tile([2, -1]).unwrap().tile.cols, 4);
        // Terrain does not make a segment arrive.
        assert!(s.arrived().is_empty());

        let set = model::Chunk {
            kind: model::ChunkKind::Buildings as i32,
            payload: payload::BuildingSet { cell: vec![0, 1], buildings: vec![] }.encode_to_vec(),
            ..Default::default()
        };
        assert_eq!(s.apply(&set).change, Change::Buildings { cell: [0, 1] });
        assert_eq!(s.building_sets().count(), 1);
    }

    #[test]
    fn clear_forgets_but_epochs_keep_increasing() {
        let mut s = ChunkStore::new();
        s.apply(&recon(0, model::Level::Preview, 0, 0, 1));
        let before = s.segment(0).unwrap().epoch();
        s.clear();
        assert!(s.segment(0).is_none());
        assert!(s.arrived().is_empty());
        s.apply(&recon(0, model::Level::Preview, 0, 0, 1));
        assert!(s.segment(0).unwrap().epoch() > before);
    }

    #[test]
    fn segment_center_is_the_middle_of_its_boxes() {
        let mut s = ChunkStore::new();
        let pts = vec![
            Point { position: [100.0, 200.0, 10.0], color: [0; 3] },
            Point { position: [110.0, 220.0, 30.0], color: [0; 3] },
        ];
        let c = model::Chunk {
            kind: model::ChunkKind::Points as i32,
            level: model::Level::Preview as i32,
            payload: encode_with(0, 0, 0, &pts, 12),
            ..Default::default()
        };
        s.apply(&c);
        let m = s.segment(0).unwrap().center().unwrap();
        assert!((m[0] - 105.0).abs() < 1e-3 && (m[1] - 210.0).abs() < 1e-3 && (m[2] - 20.0).abs() < 1e-3);
    }
}
