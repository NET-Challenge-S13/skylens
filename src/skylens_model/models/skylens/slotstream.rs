//! The live display as point slot messages.
//!
//! cumulus3d composes the display in memory (`NodeBuilder::display_deltas`): after every dense zone
//! result it delivers an `Event::DisplayDelta`, the point-keyed difference between the previous and
//! the new display cloud (adds, moves and removes with dense slot ids). [`SlotEncoder`] writes each
//! delta as one pointblitz slot message, the 32-byte "PBSD" header in front of one v4 pack (the
//! layout of cumulus3d's live diff stream files, decision 0052 of pointblitz), and
//! [`slots_chunk`] wraps it in one `CHUNK_KIND_POINT_SLOTS` chunk:
//!
//! | chunk field | value |
//! |---|---|
//! | `segment` | the zone that caused the delta |
//! | `level` | refined (stable display has no preview) |
//! | `version` | the message seq (1, 2, ... in delivery order, no gaps) |
//! | `generation` | the stream id of this model run ([`next_stream_id`]) |
//!
//! | pack record | bytes (little endian) |
//! |---|---|
//! | add | `u32 id, 3 x f32 xyz, 3 x u8 rgb, u8 zone` (20) |
//! | move | `u32 id, 3 x f32 xyz, 3 x u8 rgb, u8 0` (20) |
//! | remove | `u32 id` (4) |
//!
//! Positions are ENU metres about the engine anchor ([`align_delta`]).
//!
//! The module needs nothing from the rest of the model server (no libtorch), so the cross-component
//! tests in `src/test` compile it on its own and run its tests on machines without CUDA.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use cumulus3d_cli::live::{DisplayDelta, DisplayFrame};
use pointblitz_io::slots::{self, Record, SlotHeader};
use skylens_shared::geo::{GeoAnchor, Gps, gps_to_enu};
use skylens_shared::proto::model::v1::{Chunk, ChunkKind, Level};

/// Offsets of the display origin from the anchor below this (metres) are ignored.
pub const ORIGIN_TOLERANCE_M: f64 = 0.01;

/// Whole seconds since the Unix epoch at `at` (wrapping in 2106), at least 1.
pub fn stream_id(at: SystemTime) -> u32 {
    at.duration_since(UNIX_EPOCH).map(|d| d.as_secs() as u32).unwrap_or(1).max(1)
}

static LAST_STREAM: AtomicU32 = AtomicU32::new(0);

/// Stream id of a new model run: the wall clock second, and higher than every id this process handed
/// out before, so a restart within the same second or after the clock went back still gets a higher
/// id (the core and the screens drop a stream whose id is not higher than the current one).
pub fn next_stream_id(now: SystemTime) -> u32 {
    let want = stream_id(now);
    let mut id = want;
    let _ = LAST_STREAM.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |last| {
        id = want.max(last.saturating_add(1));
        Some(id)
    });
    id
}

/// First chunk sequence of a model run minus one: the stream id in the upper 32 bits. The core
/// resumes `SubscribeChunks` after the last sequence it stored; with sequences unique per run a
/// restarted model's chunks are all above that, so none of them is skipped.
pub fn sequence_base(stream: u32) -> u64 {
    u64::from(stream) << 32
}

/// One slot message as a chunk.
pub fn slots_chunk(stream: u32, zone: u32, seq: u32, message: Vec<u8>) -> Chunk {
    Chunk {
        kind: ChunkKind::PointSlots as i32,
        segment: zone,
        level: Level::Refined as i32,
        version: seq,
        generation: stream,
        payload: message,
        ..Default::default()
    }
}

/// A display delta expressed in ENU metres about `anchor`.
///
/// `None` when the points are not in an ENU frame (`frame.enu == false`): they would be unaligned
/// with everything else. When the frame origin differs from the anchor by more than
/// [`ORIGIN_TOLERANCE_M`], every add and move is translated by the ENU offset of the origin about the
/// anchor. A frame without a known origin is taken as the anchor frame.
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

/// The v4 pack of a delta: adds, then moves, then removes.
pub fn pack(d: &DisplayDelta) -> Vec<u8> {
    let mut out = Vec::with_capacity((d.adds.len() + d.moves.len()) * slots::ADD_BYTES + d.removes.len() * slots::REMOVE_BYTES);
    for a in &d.adds {
        Record { id: a.id, position: a.xyz, color: a.rgb, zone: a.zone }.write(&mut out);
    }
    for m in &d.moves {
        Record { id: m.id, position: m.xyz, color: m.rgb, zone: 0 }.write(&mut out);
    }
    for id in &d.removes {
        out.extend(id.to_le_bytes());
    }
    out
}

/// Numbers the slot messages of one model run and writes them.
#[derive(Debug)]
pub struct SlotEncoder {
    stream: u32,
    next: u32,
}

impl SlotEncoder {
    /// An encoder for the run with stream id `stream`; its first message is seq 1.
    pub fn new(stream: u32) -> Self {
        Self { stream, next: 1 }
    }

    /// The stream id of the run.
    pub fn stream(&self) -> u32 {
        self.stream
    }

    /// Seq of the next message (1 before any).
    pub fn next_seq(&self) -> u32 {
        self.next
    }

    /// The chunk for `delta` (already aligned to the anchor, see [`align_delta`]), caused by `zone`.
    /// The message seq is this encoder's own count, not the delta's: cumulus3d's seq also counts
    /// unchanged clouds and can skip, and the core holds a stream with a gap.
    pub fn encode(&mut self, delta: &DisplayDelta, zone: usize) -> anyhow::Result<Chunk> {
        let header = SlotHeader {
            seq: self.next,
            n_add: u32::try_from(delta.adds.len())?,
            n_move: u32::try_from(delta.moves.len())?,
            n_remove: u32::try_from(delta.removes.len())?,
            capacity: delta.capacity,
        };
        let message = slots::encode(&header, &pack(delta))?;
        slots::validate(&message)?;
        let chunk = slots_chunk(self.stream, u32::try_from(zone).unwrap_or(u32::MAX), self.next, message);
        self.next += 1;
        Ok(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cumulus3d_cli::live::{PointAdd, PointMove};
    use pointblitz_io::slots::{Diff, Fold, HEADER_LEN};

    const ANCHOR: GeoAnchor = Gps { lat: 36.3685, lon: 127.3475, alt: 30.0 };

    fn add(id: u32, x: f32, zone: u8) -> PointAdd {
        PointAdd { id, xyz: [x, -x, 1.0], rgb: [10, 20, 30], zone }
    }

    fn delta(adds: Vec<PointAdd>, moves: Vec<PointMove>, removes: Vec<u32>, capacity: u32) -> DisplayDelta {
        DisplayDelta { seq: 1, adds, moves, removes, total_points: 0, capacity }
    }

    fn enu_frame(origin: Option<[f64; 3]>) -> DisplayFrame {
        DisplayFrame { enu: true, origin_gps: origin }
    }

    #[test]
    fn stream_ids_only_grow_within_a_process() {
        let now = SystemTime::now();
        let a = next_stream_id(now);
        let b = next_stream_id(now);
        let c = next_stream_id(now - std::time::Duration::from_secs(3600));
        assert!(a < b && b < c, "{a} {b} {c}");
        assert_eq!(sequence_base(7), 7 << 32);
    }

    #[test]
    fn a_delta_becomes_a_validated_message_with_the_pack_bytes() {
        let d = delta(
            vec![add(0, 1.0, 0), add(1, 2.0, 1)],
            vec![PointMove { id: 0, xyz: [5.0, 6.0, 7.0], rgb: [1, 2, 3] }],
            vec![],
            2,
        );
        let mut e = SlotEncoder::new(42);
        let c = e.encode(&d, 1).unwrap();
        assert_eq!((c.kind, c.segment, c.level, c.version, c.generation), (ChunkKind::PointSlots as i32, 1, Level::Refined as i32, 1, 42));
        let h = slots::validate(&c.payload).unwrap();
        assert_eq!((h.seq, h.n_add, h.n_move, h.n_remove, h.capacity), (1, 2, 1, 0, 2));
        assert_eq!(c.payload.len(), HEADER_LEN + 3 * 20);
        // Byte layout of the first add: id, xyz, rgb, zone (little endian).
        let r = &c.payload[HEADER_LEN..HEADER_LEN + 20];
        assert_eq!(r[..4], 0u32.to_le_bytes());
        assert_eq!(r[4..8], 1.0f32.to_le_bytes());
        assert_eq!(r[8..12], (-1.0f32).to_le_bytes());
        assert_eq!(r[12..16], 1.0f32.to_le_bytes());
        assert_eq!(r[16..], [10, 20, 30, 0]);
        // The move record carries zone 0.
        let diff = Diff::parse(&c.payload).unwrap();
        assert_eq!(diff.move_records().next().unwrap().position, [5.0, 6.0, 7.0]);
        assert_eq!(e.next_seq(), 2);
    }

    #[test]
    fn messages_are_numbered_by_the_encoder_and_fold_to_the_display() {
        let mut e = SlotEncoder::new(9);
        let mut fold = Fold::new();
        let deltas = [
            delta(vec![add(0, 1.0, 0), add(1, 2.0, 0), add(2, 3.0, 0)], vec![], vec![], 3),
            // cumulus3d's own seq may skip (unchanged clouds); the message seq does not.
            DisplayDelta { seq: 5, ..delta(vec![add(3, 4.0, 1)], vec![PointMove { id: 0, xyz: [0.0; 3], rgb: [9, 9, 9] }], vec![2], 4) },
            delta(vec![], vec![], vec![], 4),
        ];
        for (i, d) in deltas.iter().enumerate() {
            let c = e.encode(d, i).unwrap();
            assert_eq!(c.version, i as u32 + 1);
            fold.apply(&c.payload).unwrap();
        }
        assert_eq!((fold.seq(), fold.points(), fold.capacity()), (3, 3, 4));
    }

    #[test]
    fn frame_alignment_translates_or_drops() {
        let d = delta(vec![add(0, 1.0, 0)], vec![PointMove { id: 0, xyz: [2.0, 2.0, 2.0], rgb: [0; 3] }], vec![], 1);
        // Not an ENU frame: dropped.
        assert!(align_delta(&d, &DisplayFrame { enu: false, origin_gps: None }, ANCHOR).is_none());
        // Origin = anchor, or unknown: unchanged.
        for f in [enu_frame(Some([ANCHOR.lat, ANCHOR.lon, ANCHOR.alt])), enu_frame(None)] {
            assert_eq!(align_delta(&d, &f, ANCHOR).unwrap(), d);
        }
        // An origin 0.001 degrees north of the anchor: points shift by about 111 m north.
        let shifted = align_delta(&d, &enu_frame(Some([ANCHOR.lat + 0.001, ANCHOR.lon, ANCHOR.alt])), ANCHOR).unwrap();
        assert!((shifted.adds[0].xyz[1] - (-1.0 + 111.2)).abs() < 0.5, "{:?}", shifted.adds[0].xyz);
        assert!((shifted.moves[0].xyz[0] - 2.0).abs() < 0.01);
    }
}
