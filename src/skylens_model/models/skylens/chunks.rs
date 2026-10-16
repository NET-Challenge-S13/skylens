//! Chunk production and the replay log behind `SubscribeChunks`.
//!
//! - [`point_chunks`]: a zone cloud as pointblitz chunk format v2 bytes. Refined
//!   deliveries go coarse first (every 8th point, then the rest) with
//!   `FLAG_FIRST_PASS_COMPLETE` on the last chunk of the first pass.
//! - [`mesh_chunk`]: one deltamesh block as `payload.v1.MeshBlock`.
//! - [`ChunkLog`]: assigns the monotonic `sequence`, keeps what a late subscriber
//!   still needs, and feeds every subscriber from its own thread so a slow one
//!   never blocks the producer.
//!
//! Compaction (what the log forgets): a mesh block is superseded by a newer
//! version of the same block id, a detection batch by a newer batch of the same
//! segment, and the points of a segment by a delivery of a newer generation.
//! A subscriber resuming at `after_sequence` therefore receives the current
//! state, not every intermediate one.

use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

use pointblitz_io::Point as PbPoint;
use pointblitz_io::chunk::encode;
use pointblitz_io::convert::{MAX_POINTS, piece_plan};
use prost::Message;
use skylens_shared::proto::model::v1::{Chunk, ChunkKind, Level};
use skylens_shared::proto::payload::v1::{DetectionBatch, MeshBlock};

/// Channel depth of one subscriber.
pub const SUBSCRIBER_BUFFER: usize = 256;

/// One point of a zone cloud, ENU metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZonePoint {
    pub pos: [f32; 3],
    pub rgb: [u8; 3],
    /// Zero when unknown.
    pub normal: [f32; 3],
}

/// Encodes a zone cloud as pointblitz chunks (at most `max_points` each).
/// `coarse` sends every 8th point first (refined snapshots). Always at least one
/// chunk; the last carries `FLAG_LAST_IN_GENERATION`.
pub fn point_chunks(points: &[ZonePoint], generation: u32, coarse: bool, max_points: usize) -> Vec<Vec<u8>> {
    piece_plan(points.len(), coarse, max_points.max(1))
        .iter()
        .enumerate()
        .map(|(i, piece)| {
            let pts: Vec<PbPoint> = (0..piece.count)
                .map(|j| {
                    let p = &points[piece.start + j * piece.step];
                    PbPoint { position: p.pos, color: p.rgb }
                })
                .collect();
            encode(generation, i as u32, piece.flags, &pts)
        })
        .collect()
}

/// Point chunks of one zone delivery, ready for the log.
pub fn points_delivery(segment: u32, level: Level, generation: u32, points: &[ZonePoint]) -> Vec<Chunk> {
    let coarse = level == Level::Refined;
    point_chunks(points, generation, coarse, MAX_POINTS)
        .into_iter()
        .map(|payload| Chunk {
            kind: ChunkKind::Points as i32,
            segment,
            level: level as i32,
            generation,
            payload,
            ..Default::default()
        })
        .collect()
}

/// One mesh block. `None` (block emptied) becomes a zero-triangle removal.
pub fn mesh_chunk(segment: u32, level: Level, id: [i32; 3], version: u32, mesh: Option<&deltamesh::BlockMesh>) -> Chunk {
    let block = match mesh {
        Some(m) => MeshBlock {
            id: m.id.to_vec(),
            version: m.version,
            positions: m.positions.clone(),
            colors: m.colors.clone(),
            indices: m.indices.clone(),
        },
        None => MeshBlock { id: id.to_vec(), version, ..Default::default() },
    };
    Chunk {
        kind: ChunkKind::MeshBlock as i32,
        segment,
        level: level as i32,
        block_id: block.id.clone(),
        version: block.version,
        payload: block.encode_to_vec(),
        ..Default::default()
    }
}

/// The full current detection set of one segment. `version` counts batches of
/// the segment; a newer batch replaces the older one.
pub fn detections_chunk(segment: u32, version: u32, batch: &DetectionBatch) -> Chunk {
    Chunk {
        kind: ChunkKind::Detections as i32,
        segment,
        version,
        payload: batch.encode_to_vec(),
        ..Default::default()
    }
}

#[derive(Default)]
struct LogState {
    last: u64,
    chunks: BTreeMap<u64, Arc<Chunk>>,
    closed: bool,
}

/// Sequenced, compacted chunk log with replaying subscribers.
#[derive(Default)]
pub struct ChunkLog {
    st: Mutex<LogState>,
    cv: Condvar,
}

impl ChunkLog {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// A log whose first chunk gets sequence `base + 1` (see `slotstream::sequence_base`: the
    /// live engine starts each model run above every sequence of the runs before it).
    pub fn starting_after(base: u64) -> Arc<Self> {
        let log = Self::default();
        log.lock().last = base;
        Arc::new(log)
    }

    fn lock(&self) -> MutexGuard<'_, LogState> {
        self.st.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Appends chunks in order, assigning sequences. Returns the last sequence.
    pub fn publish(&self, chunks: impl IntoIterator<Item = Chunk>) -> u64 {
        let mut st = self.lock();
        for mut c in chunks {
            st.last += 1;
            c.sequence = st.last;
            compact(&mut st.chunks, &c);
            let seq = c.sequence;
            st.chunks.insert(seq, Arc::new(c));
        }
        let last = st.last;
        drop(st);
        self.cv.notify_all();
        last
    }

    /// Highest sequence assigned so far.
    pub fn last_sequence(&self) -> u64 {
        self.lock().last
    }

    /// Number of chunks retained for replay.
    pub fn retained(&self) -> usize {
        self.lock().chunks.len()
    }

    /// Chunks retained with a sequence above `after`, in order.
    pub fn since(&self, after: u64) -> Vec<Arc<Chunk>> {
        self.lock().chunks.range(after.saturating_add(1)..).map(|(_, c)| c.clone()).collect()
    }

    /// Replays retained chunks above `after_sequence`, then follows new ones.
    /// The channel closes when the log closes; dropping the receiver ends the
    /// feeder thread within a second.
    pub fn subscribe(self: &Arc<Self>, after_sequence: u64) -> tokio::sync::mpsc::Receiver<Chunk> {
        let (tx, rx) = tokio::sync::mpsc::channel(SUBSCRIBER_BUFFER);
        let log = Arc::clone(self);
        let spawned = std::thread::Builder::new().name("chunk-subscriber".into()).spawn(move || {
            let mut cursor = after_sequence;
            loop {
                let batch = {
                    let mut st = log.lock();
                    loop {
                        let next: Vec<Arc<Chunk>> =
                            st.chunks.range(cursor.saturating_add(1)..).map(|(_, c)| c.clone()).collect();
                        if !next.is_empty() {
                            break next;
                        }
                        if st.closed || tx.is_closed() {
                            return;
                        }
                        st = log.cv.wait_timeout(st, Duration::from_secs(1)).unwrap_or_else(|p| p.into_inner()).0;
                    }
                };
                for c in batch {
                    cursor = c.sequence;
                    if tx.blocking_send((*c).clone()).is_err() {
                        return;
                    }
                }
            }
        });
        if let Err(e) = spawned {
            tracing::error!("chunk subscriber thread: {e}");
        }
        rx
    }

    /// Ends every subscription after it has drained what is retained.
    pub fn close(&self) {
        self.lock().closed = true;
        self.cv.notify_all();
    }
}

/// Drops retained chunks the new one supersedes.
fn compact(chunks: &mut BTreeMap<u64, Arc<Chunk>>, new: &Chunk) {
    let kind = new.kind;
    if kind == ChunkKind::MeshBlock as i32 {
        chunks.retain(|_, c| !(c.kind == kind && c.block_id == new.block_id && c.version <= new.version));
    } else if kind == ChunkKind::Detections as i32 {
        chunks.retain(|_, c| !(c.kind == kind && c.segment == new.segment && c.version <= new.version));
    } else if kind == ChunkKind::Points as i32 {
        chunks.retain(|_, c| !(c.kind == kind && c.segment == new.segment && c.generation < new.generation));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pointblitz_io::chunk::{FLAG_FIRST_PASS_COMPLETE, FLAG_LAST_IN_GENERATION, decode_header};

    fn cloud(n: usize) -> Vec<ZonePoint> {
        (0..n)
            .map(|i| ZonePoint { pos: [i as f32, 0.5 * i as f32, 10.0], rgb: [i as u8, 1, 2], normal: [0.0; 3] })
            .collect()
    }

    #[test]
    fn preview_points_keep_order_and_close_the_delivery() {
        let c = point_chunks(&cloud(10), 0, false, 4);
        let h: Vec<_> = c.iter().map(|b| decode_header(b).unwrap()).collect();
        assert_eq!(h.iter().map(|h| h.point_count).collect::<Vec<_>>(), [4, 4, 2]);
        assert_eq!(h.last().unwrap().flags, FLAG_LAST_IN_GENERATION);
        assert!(h.iter().all(|h| h.flags & FLAG_FIRST_PASS_COMPLETE == 0));
    }

    #[test]
    fn refined_points_go_coarse_first() {
        let c = point_chunks(&cloud(64), 3, true, 1024);
        let h: Vec<_> = c.iter().map(|b| decode_header(b).unwrap()).collect();
        assert!(h.iter().all(|h| h.generation == 3));
        // First pass: every 8th point.
        assert_eq!(h[0].point_count, 8);
        assert_ne!(h[0].flags & FLAG_FIRST_PASS_COMPLETE, 0);
        assert_ne!(h.last().unwrap().flags & FLAG_LAST_IN_GENERATION, 0);
        assert_eq!(h.iter().map(|h| h.point_count).sum::<u32>(), 64);
    }

    #[test]
    fn empty_cloud_is_one_closing_chunk() {
        let c = point_chunks(&[], 1, true, 16);
        assert_eq!(c.len(), 1);
        let h = decode_header(&c[0]).unwrap();
        assert_eq!(h.point_count, 0);
        assert_ne!(h.flags & FLAG_LAST_IN_GENERATION, 0);
    }

    #[test]
    fn mesh_block_round_trips_and_removal_is_empty() {
        let m = deltamesh::BlockMesh {
            id: [1, -2, 3],
            version: 4,
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            colors: vec![9; 9],
            indices: vec![0, 1, 2],
        };
        let c = mesh_chunk(5, Level::Preview, m.id, m.version, Some(&m));
        assert_eq!(c.kind, ChunkKind::MeshBlock as i32);
        assert_eq!((c.segment, c.version, c.block_id.clone()), (5, 4, vec![1, -2, 3]));
        let back = MeshBlock::decode(c.payload.as_slice()).unwrap();
        assert_eq!(back.indices, vec![0, 1, 2]);
        assert_eq!(back.positions.len(), 9);

        let gone = mesh_chunk(5, Level::Refined, [1, -2, 3], 5, None);
        let back = MeshBlock::decode(gone.payload.as_slice()).unwrap();
        assert!(back.indices.is_empty() && back.positions.is_empty());
        assert_eq!((back.version, back.id), (5, vec![1, -2, 3]));
    }

    #[test]
    fn sequences_are_monotonic_and_compaction_keeps_the_newest() {
        let log = ChunkLog::new();
        let mesh = |v| mesh_chunk(0, Level::Preview, [0, 0, 0], v, None);
        log.publish(points_delivery(0, Level::Preview, 0, &cloud(3)));
        log.publish([mesh(1), mesh(2)]);
        log.publish([detections_chunk(0, 1, &DetectionBatch::default())]);
        log.publish([detections_chunk(0, 2, &DetectionBatch::default())]);
        assert_eq!(log.last_sequence(), 5);
        let kept: Vec<_> = log.since(0).iter().map(|c| (c.sequence, c.kind, c.version)).collect();
        assert_eq!(
            kept,
            vec![
                (1, ChunkKind::Points as i32, 0),
                (3, ChunkKind::MeshBlock as i32, 2),
                (5, ChunkKind::Detections as i32, 2),
            ]
        );
        // A refined generation replaces the preview points of the same segment only.
        log.publish(points_delivery(1, Level::Preview, 0, &cloud(2)));
        log.publish(points_delivery(0, Level::Refined, 1, &cloud(2)));
        let points: Vec<_> = log
            .since(0)
            .iter()
            .filter(|c| c.kind == ChunkKind::Points as i32)
            .map(|c| (c.segment, c.generation))
            .collect();
        // Refined points go out coarse-first, so one generation may span several chunks.
        assert!(points.contains(&(1, 0)), "segment 1 preview must stay: {points:?}");
        assert!(points.contains(&(0, 1)), "segment 0 refined must arrive: {points:?}");
        assert!(!points.contains(&(0, 0)), "segment 0 preview must be replaced: {points:?}");
    }

    #[test]
    fn a_log_with_a_base_numbers_its_chunks_above_it() {
        let base = 7u64 << 32;
        let log = ChunkLog::starting_after(base);
        assert_eq!(log.last_sequence(), base);
        assert!(log.since(0).is_empty());
        log.publish([detections_chunk(0, 1, &DetectionBatch::default())]);
        log.publish([detections_chunk(1, 1, &DetectionBatch::default())]);
        let seqs: Vec<u64> = log.since(0).iter().map(|c| c.sequence).collect();
        assert_eq!(seqs, vec![base + 1, base + 2]);
        // A core that stored a lower (earlier run's) sequence gets every chunk of this run.
        assert_eq!(log.since((6u64 << 32) + 900).len(), 2);
    }

    #[test]
    fn subscriber_replays_then_follows() {
        let log = ChunkLog::new();
        log.publish([detections_chunk(0, 1, &DetectionBatch::default())]);
        log.publish([detections_chunk(1, 1, &DetectionBatch::default())]);
        let mut rx = log.subscribe(1);
        let first = rx.blocking_recv().unwrap();
        assert_eq!((first.sequence, first.segment), (2, 1));
        log.publish([detections_chunk(2, 1, &DetectionBatch::default())]);
        assert_eq!(rx.blocking_recv().unwrap().sequence, 3);
        log.close();
        assert!(rx.blocking_recv().is_none());
    }
}
