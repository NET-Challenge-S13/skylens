//! Chunk bookkeeping: which chunk payloads are on screen.
//!
//! Replacement rules (SPEC §5, COMPONENTS §3.6):
//!
//! - Points, terrain and buildings are kept per stream `(layer, segment)`.
//!   Preview chunks only add points. A refined chunk replaces the preview of its
//!   segment, and a newer refined generation replaces an older one. As in
//!   pointblitz's own `Scene`, a replacing generation stays pending until the
//!   chunk that completes its coarse first pass (or its last chunk) arrives, so
//!   the segment never goes blank in between.
//! - Mesh blocks are kept per block id; only a newer `version` replaces the block
//!   and a block with zero triangles is removed.
//! - A segment has *arrived* once any reconstruction chunk (points or mesh) of it
//!   was received. Detections are shown only for arrived segments (SPEC §7).
//!
//! The book keeps the payload bytes so the 3D view can be rebuilt (display mode
//! change, GPU device recreated) without asking the core again.

use crate::display::Layer;
use crate::protocol;
use pointblitz::io::chunk::{decode_header, FLAG_FIRST_PASS_COMPLETE, FLAG_LAST_IN_GENERATION};
use skylens_shared::proto::model::v1 as model;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

pub type Payload = Rc<[u8]>;

/// Work for the 3D view, in arrival order.
#[derive(Debug, Clone)]
pub enum SceneOp {
    /// Add a chunk to what is on screen.
    InsertPoints(Layer, Payload),
    /// Replace every point chunk on screen with this list.
    RebuildPoints(Vec<(Layer, Payload)>),
    /// Hand a mesh block to `MeshPass::apply`.
    ApplyMesh(Payload),
    /// `MeshPass::clear`, then apply every block in the list.
    ResetMesh(Vec<Payload>),
}

/// What a point chunk did to its stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PointOutcome {
    Appended,
    Pending,
    Swapped,
    Stale,
    Invalid(String),
}

/// Ordering key of a delivery within a stream: previews rank below every refined
/// generation, and all previews share one key because they only add points.
type Key = (u8, u32);

fn key(level: model::Level, generation: u32) -> Key {
    match level {
        model::Level::Refined => (1, generation),
        _ => (0, 0),
    }
}

#[derive(Debug, Default)]
struct Stream {
    shown_key: Option<Key>,
    shown: Vec<Payload>,
    pending: Option<(Key, Vec<Payload>)>,
}

impl Stream {
    fn insert(&mut self, k: Key, payload: Payload, completes: bool) -> PointOutcome {
        match self.shown_key {
            None => {
                self.shown_key = Some(k);
                self.shown.push(payload);
                PointOutcome::Appended
            }
            Some(cur) if k < cur => PointOutcome::Stale,
            Some(cur) if k == cur => {
                self.shown.push(payload);
                PointOutcome::Appended
            }
            Some(_) => {
                match &mut self.pending {
                    Some((pk, v)) if *pk == k => v.push(payload),
                    Some((pk, _)) if *pk > k => return PointOutcome::Stale,
                    _ => self.pending = Some((k, vec![payload])),
                }
                if completes {
                    let (pk, v) = self.pending.take().expect("pending set above");
                    self.shown_key = Some(pk);
                    self.shown = v;
                    PointOutcome::Swapped
                } else {
                    PointOutcome::Pending
                }
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct ChunkBook {
    streams: BTreeMap<(Layer, u32), Stream>,
    meshes: BTreeMap<[i32; 3], (u32, Payload)>,
    arrived: BTreeSet<u32>,
    /// Highest level delivered per reconstruction segment (points or mesh).
    levels: BTreeMap<u32, model::Level>,
    pub chunks_received: u64,
    pub invalid: u64,
}

/// Result of [`ChunkBook::apply`].
#[derive(Debug, Default)]
pub struct Applied {
    pub ops: Vec<SceneOp>,
    /// Detections carried by a `CHUNK_KIND_DETECTIONS` chunk.
    pub detections: Option<Vec<skylens_shared::proto::payload::v1::Detection>>,
    /// The segment arrived with this chunk (first reconstruction chunk of it).
    pub newly_arrived: Option<u32>,
    pub outcome: Option<PointOutcome>,
}

impl ChunkBook {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply(&mut self, chunk: &model::Chunk) -> Applied {
        self.chunks_received += 1;
        let mut out = Applied::default();
        let kind = chunk.kind();
        let layer = match kind {
            model::ChunkKind::Points => Some(Layer::Recon),
            model::ChunkKind::Terrain => Some(Layer::Terrain),
            model::ChunkKind::Buildings => Some(Layer::Buildings),
            _ => None,
        };

        if let Some(layer) = layer {
            let header = match decode_header(&chunk.payload) {
                Ok(h) => h,
                Err(e) => {
                    self.invalid += 1;
                    out.outcome = Some(PointOutcome::Invalid(e.to_string()));
                    return out;
                }
            };
            let completes =
                header.flags & (FLAG_LAST_IN_GENERATION | FLAG_FIRST_PASS_COMPLETE) != 0;
            let payload: Payload = Rc::from(chunk.payload.as_slice());
            let stream = self.streams.entry((layer, chunk.segment)).or_default();
            let outcome = stream.insert(
                key(chunk.level(), chunk.generation),
                payload.clone(),
                completes,
            );
            match outcome {
                PointOutcome::Appended => out.ops.push(SceneOp::InsertPoints(layer, payload)),
                PointOutcome::Swapped => out.ops.push(SceneOp::RebuildPoints(self.visible_points())),
                _ => {}
            }
            if layer == Layer::Recon && matches!(outcome, PointOutcome::Appended | PointOutcome::Swapped) {
                out.newly_arrived = self.mark_arrived(chunk.segment, chunk.level());
            }
            out.outcome = Some(outcome);
            return out;
        }

        match kind {
            model::ChunkKind::MeshBlock => {
                let block = match protocol::decode_mesh_block(&chunk.payload) {
                    Ok(b) => b,
                    Err(e) => {
                        self.invalid += 1;
                        out.outcome = Some(PointOutcome::Invalid(e.to_string()));
                        return out;
                    }
                };
                let id_src = if chunk.block_id.len() == 3 { &chunk.block_id } else { &block.id };
                let Ok(id) = <[i32; 3]>::try_from(id_src.as_slice()) else {
                    self.invalid += 1;
                    out.outcome = Some(PointOutcome::Invalid("mesh block id is not [x, y, z]".into()));
                    return out;
                };
                let version = if chunk.version != 0 { chunk.version } else { block.version };
                if self.meshes.get(&id).is_some_and(|(v, _)| *v >= version) {
                    out.outcome = Some(PointOutcome::Stale);
                    return out;
                }
                let payload: Payload = Rc::from(chunk.payload.as_slice());
                if block.indices.is_empty() {
                    self.meshes.remove(&id);
                } else {
                    self.meshes.insert(id, (version, payload.clone()));
                }
                out.ops.push(SceneOp::ApplyMesh(payload));
                out.newly_arrived = self.mark_arrived(chunk.segment, chunk.level());
                out.outcome = Some(PointOutcome::Appended);
            }
            model::ChunkKind::Detections => match protocol::decode_detections(&chunk.payload) {
                Ok(batch) => out.detections = Some(batch.detections),
                Err(e) => {
                    self.invalid += 1;
                    out.outcome = Some(PointOutcome::Invalid(e.to_string()));
                }
            },
            _ => {
                self.invalid += 1;
                out.outcome = Some(PointOutcome::Invalid(format!("unknown chunk kind {}", chunk.kind)));
            }
        }
        out
    }

    fn mark_arrived(&mut self, segment: u32, level: model::Level) -> Option<u32> {
        let entry = self.levels.entry(segment).or_insert(model::Level::Unspecified);
        if level_rank(level) > level_rank(*entry) {
            *entry = level;
        }
        self.arrived.insert(segment).then_some(segment)
    }

    /// Every point chunk on screen, in a stable order (layer, segment, arrival).
    pub fn visible_points(&self) -> Vec<(Layer, Payload)> {
        self.streams
            .iter()
            .flat_map(|((layer, _), s)| s.shown.iter().map(move |p| (*layer, p.clone())))
            .collect()
    }

    /// Current mesh blocks, for replaying into a fresh `MeshPass`.
    pub fn mesh_payloads(&self) -> Vec<Payload> {
        self.meshes.values().map(|(_, p)| p.clone()).collect()
    }

    pub fn arrived(&self) -> &BTreeSet<u32> {
        &self.arrived
    }

    pub fn has_arrived(&self, segment: u32) -> bool {
        self.arrived.contains(&segment)
    }

    /// Highest reconstruction level seen per segment.
    pub fn levels(&self) -> &BTreeMap<u32, model::Level> {
        &self.levels
    }

    pub fn mesh_blocks(&self) -> usize {
        self.meshes.len()
    }

    /// Ops that recreate the whole scene from what the book holds.
    pub fn full_resync(&self) -> Vec<SceneOp> {
        vec![
            SceneOp::RebuildPoints(self.visible_points()),
            SceneOp::ResetMesh(self.mesh_payloads()),
        ]
    }
}

fn level_rank(level: model::Level) -> u8 {
    match level {
        model::Level::Unspecified => 0,
        model::Level::Preview => 1,
        model::Level::Refined => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pointblitz::io::chunk::encode_with;
    use pointblitz::io::Point;
    use prost::Message as _;
    use skylens_shared::proto::payload::v1 as payload;

    fn pts(n: usize) -> Vec<Point> {
        (0..n)
            .map(|i| Point { position: [i as f32, 0.0, 0.0], color: [10, 20, 30] })
            .collect()
    }

    fn point_chunk(
        kind: model::ChunkKind,
        segment: u32,
        level: model::Level,
        generation: u32,
        flags: u32,
        n: usize,
    ) -> model::Chunk {
        model::Chunk {
            sequence: 0,
            kind: kind as i32,
            segment,
            level: level as i32,
            block_id: vec![],
            version: 0,
            generation,
            payload: encode_with(generation, 0, flags, &pts(n), 12),
        }
    }

    fn recon(segment: u32, level: model::Level, generation: u32, flags: u32, n: usize) -> model::Chunk {
        point_chunk(model::ChunkKind::Points, segment, level, generation, flags, n)
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
            sequence: 0,
            kind: model::ChunkKind::MeshBlock as i32,
            segment,
            level: model::Level::Preview as i32,
            block_id: id.to_vec(),
            version,
            generation: 0,
            payload: block.encode_to_vec(),
        }
    }

    fn visible_counts(book: &ChunkBook) -> usize {
        book.visible_points().len()
    }

    #[test]
    fn previews_append() {
        let mut book = ChunkBook::new();
        for _ in 0..3 {
            let a = book.apply(&recon(0, model::Level::Preview, 0, 0, 4));
            assert_eq!(a.outcome, Some(PointOutcome::Appended));
            assert!(matches!(a.ops[..], [SceneOp::InsertPoints(Layer::Recon, _)]));
        }
        assert_eq!(visible_counts(&book), 3);
    }

    #[test]
    fn refined_replaces_preview_of_its_segment_only() {
        let mut book = ChunkBook::new();
        book.apply(&recon(0, model::Level::Preview, 0, 0, 4));
        book.apply(&recon(0, model::Level::Preview, 0, 0, 4));
        book.apply(&recon(1, model::Level::Preview, 0, 0, 4));

        // First refined chunk without a completing flag stays pending.
        let a = book.apply(&recon(0, model::Level::Refined, 1, 0, 8));
        assert_eq!(a.outcome, Some(PointOutcome::Pending));
        assert!(a.ops.is_empty());
        assert_eq!(visible_counts(&book), 3);

        // Completing chunk swaps segment 0; segment 1 keeps its preview.
        let a = book.apply(&recon(0, model::Level::Refined, 1, FLAG_LAST_IN_GENERATION, 8));
        assert_eq!(a.outcome, Some(PointOutcome::Swapped));
        match &a.ops[..] {
            [SceneOp::RebuildPoints(list)] => assert_eq!(list.len(), 3), // 2 refined + 1 preview
            other => panic!("unexpected ops {other:?}"),
        }

        // Late previews of a refined segment are stale.
        let a = book.apply(&recon(0, model::Level::Preview, 0, 0, 4));
        assert_eq!(a.outcome, Some(PointOutcome::Stale));
        assert_eq!(visible_counts(&book), 3);
    }

    #[test]
    fn newer_generation_replaces_and_older_is_stale() {
        let mut book = ChunkBook::new();
        book.apply(&recon(2, model::Level::Refined, 5, FLAG_LAST_IN_GENERATION, 4));
        assert_eq!(visible_counts(&book), 1);

        // Same generation appends (fine pass after the coarse pass).
        let a = book.apply(&recon(2, model::Level::Refined, 5, 0, 4));
        assert_eq!(a.outcome, Some(PointOutcome::Appended));

        let a = book.apply(&recon(2, model::Level::Refined, 4, FLAG_LAST_IN_GENERATION, 4));
        assert_eq!(a.outcome, Some(PointOutcome::Stale));

        let a = book.apply(&recon(2, model::Level::Refined, 6, FLAG_FIRST_PASS_COMPLETE, 4));
        assert_eq!(a.outcome, Some(PointOutcome::Swapped));
        assert_eq!(visible_counts(&book), 1);
    }

    #[test]
    fn a_newer_pending_generation_drops_an_older_pending_one() {
        let mut book = ChunkBook::new();
        book.apply(&recon(0, model::Level::Preview, 0, 0, 4));
        book.apply(&recon(0, model::Level::Refined, 1, 0, 4));
        book.apply(&recon(0, model::Level::Refined, 2, 0, 4));
        let a = book.apply(&recon(0, model::Level::Refined, 1, FLAG_LAST_IN_GENERATION, 4));
        assert_eq!(a.outcome, Some(PointOutcome::Stale));
        let a = book.apply(&recon(0, model::Level::Refined, 2, FLAG_LAST_IN_GENERATION, 4));
        assert_eq!(a.outcome, Some(PointOutcome::Swapped));
        assert_eq!(visible_counts(&book), 2);
    }

    #[test]
    fn terrain_and_buildings_are_separate_streams() {
        let mut book = ChunkBook::new();
        book.apply(&point_chunk(model::ChunkKind::Terrain, 0, model::Level::Unspecified, 0, 0, 4));
        book.apply(&point_chunk(model::ChunkKind::Buildings, 0, model::Level::Unspecified, 0, 0, 4));
        book.apply(&recon(0, model::Level::Refined, 1, FLAG_LAST_IN_GENERATION, 4));
        let layers: Vec<Layer> = book.visible_points().iter().map(|(l, _)| *l).collect();
        assert_eq!(layers, vec![Layer::Recon, Layer::Terrain, Layer::Buildings]);
        // Terrain does not make a reconstruction segment arrive.
        assert_eq!(book.arrived().iter().copied().collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn invalid_point_payload_is_counted_not_applied() {
        let mut book = ChunkBook::new();
        let mut c = recon(0, model::Level::Preview, 0, 0, 4);
        c.payload.truncate(10);
        let a = book.apply(&c);
        assert!(matches!(a.outcome, Some(PointOutcome::Invalid(_))));
        assert_eq!(book.invalid, 1);
        assert!(book.arrived().is_empty());
    }

    #[test]
    fn mesh_blocks_keep_only_newer_versions_and_drop_empty_ones() {
        let mut book = ChunkBook::new();
        assert_eq!(book.apply(&mesh([0, 0, 0], 2, 1, 4)).ops.len(), 1);
        assert_eq!(book.apply(&mesh([0, 0, 0], 2, 1, 4)).outcome, Some(PointOutcome::Stale));
        assert_eq!(book.apply(&mesh([0, 0, 0], 1, 1, 4)).outcome, Some(PointOutcome::Stale));
        assert_eq!(book.apply(&mesh([1, 0, 0], 1, 1, 4)).ops.len(), 1);
        assert_eq!(book.mesh_blocks(), 2);

        // Zero triangles removes the block, and the op still goes to the view.
        let a = book.apply(&mesh([0, 0, 0], 3, 0, 4));
        assert!(matches!(a.ops[..], [SceneOp::ApplyMesh(_)]));
        assert_eq!(book.mesh_blocks(), 1);
        assert!(book.has_arrived(4));
    }

    #[test]
    fn segment_arrival_is_reported_once_and_levels_rise() {
        let mut book = ChunkBook::new();
        assert_eq!(book.apply(&recon(3, model::Level::Preview, 0, 0, 1)).newly_arrived, Some(3));
        assert_eq!(book.apply(&recon(3, model::Level::Preview, 0, 0, 1)).newly_arrived, None);
        assert_eq!(book.levels()[&3], model::Level::Preview);
        book.apply(&recon(3, model::Level::Refined, 1, FLAG_LAST_IN_GENERATION, 1));
        assert_eq!(book.levels()[&3], model::Level::Refined);
    }

    #[test]
    fn detection_chunks_are_decoded() {
        let batch = payload::DetectionBatch {
            detections: vec![payload::Detection { id: "a".into(), segment: 1, ..Default::default() }],
        };
        let chunk = model::Chunk {
            kind: model::ChunkKind::Detections as i32,
            segment: 1,
            payload: batch.encode_to_vec(),
            ..Default::default()
        };
        let mut book = ChunkBook::new();
        let a = book.apply(&chunk);
        assert_eq!(a.detections.unwrap().len(), 1);
        assert!(!book.has_arrived(1));
    }

    #[test]
    fn full_resync_replays_points_and_meshes() {
        let mut book = ChunkBook::new();
        book.apply(&recon(0, model::Level::Preview, 0, 0, 2));
        book.apply(&mesh([0, 0, 0], 1, 1, 0));
        match &book.full_resync()[..] {
            [SceneOp::RebuildPoints(p), SceneOp::ResetMesh(m)] => {
                assert_eq!(p.len(), 1);
                assert_eq!(m.len(), 1);
            }
            other => panic!("unexpected {other:?}"),
        }
    }
}
