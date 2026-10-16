//! Encoding helpers for `CHUNK_KIND_POINTS_DELTA` (`skylens.payload.v1.PointsDelta`).
//!
//! The `adds`, `moves` and `removes` fields are packed little-endian records: adds and moves are
//! 20 bytes (`id u32, x f32, y f32, z f32, r u8, g u8, b u8, zone/pad u8`), removes are 4 bytes
//! (`id u32`). Per delta a viewer applies removes, then adds, then moves.

use std::fmt;

use prost::Message as _;

use crate::proto::model::v1 as model;
use crate::proto::payload::v1::PointsDelta;

/// Bytes per add or move record.
pub const RECORD: usize = 20;
/// Bytes per remove record.
pub const REMOVE_RECORD: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeltaError {
    /// A packed buffer whose length is not a multiple of its record size.
    BadLength { field: &'static str, len: usize, record: usize },
    /// A record id at or beyond `capacity`.
    IdOutOfRange { field: &'static str, id: u32, capacity: u32 },
    /// The chunk is not a points delta.
    WrongKind(i32),
    Decode(String),
}

impl fmt::Display for DeltaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeltaError::BadLength { field, len, record } => {
                write!(f, "{field}: length {len} is not a multiple of {record}")
            }
            DeltaError::IdOutOfRange { field, id, capacity } => {
                write!(f, "{field}: id {id} >= capacity {capacity}")
            }
            DeltaError::WrongKind(k) => write!(f, "chunk kind {k} is not a points delta"),
            DeltaError::Decode(e) => write!(f, "points delta decode: {e}"),
        }
    }
}

impl std::error::Error for DeltaError {}

/// A new or replaced point: slot id, ENU position, colour and the zone that produced it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AddRec {
    pub id: u32,
    pub xyz: [f32; 3],
    pub rgb: [u8; 3],
    pub zone: u8,
}

/// A moved/recoloured point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveRec {
    pub id: u32,
    pub xyz: [f32; 3],
    pub rgb: [u8; 3],
}

fn put_record(out: &mut Vec<u8>, id: u32, xyz: [f32; 3], rgb: [u8; 3], last: u8) {
    out.extend_from_slice(&id.to_le_bytes());
    for v in xyz {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&rgb);
    out.push(last);
}

fn get_record(r: &[u8]) -> (u32, [f32; 3], [u8; 3], u8) {
    let u = |i: usize| [r[i], r[i + 1], r[i + 2], r[i + 3]];
    (
        u32::from_le_bytes(u(0)),
        [f32::from_le_bytes(u(4)), f32::from_le_bytes(u(8)), f32::from_le_bytes(u(12))],
        [r[16], r[17], r[18]],
        r[19],
    )
}

fn check(field: &'static str, bytes: &[u8], record: usize) -> Result<(), DeltaError> {
    if bytes.len() % record != 0 {
        return Err(DeltaError::BadLength { field, len: bytes.len(), record });
    }
    Ok(())
}

pub fn pack_adds(adds: &[AddRec]) -> Vec<u8> {
    let mut out = Vec::with_capacity(adds.len() * RECORD);
    for a in adds {
        put_record(&mut out, a.id, a.xyz, a.rgb, a.zone);
    }
    out
}

pub fn iter_adds(bytes: &[u8]) -> Result<impl Iterator<Item = AddRec> + '_, DeltaError> {
    check("adds", bytes, RECORD)?;
    Ok(bytes.chunks_exact(RECORD).map(|r| {
        let (id, xyz, rgb, zone) = get_record(r);
        AddRec { id, xyz, rgb, zone }
    }))
}

pub fn pack_moves(moves: &[MoveRec]) -> Vec<u8> {
    let mut out = Vec::with_capacity(moves.len() * RECORD);
    for m in moves {
        put_record(&mut out, m.id, m.xyz, m.rgb, 0);
    }
    out
}

pub fn iter_moves(bytes: &[u8]) -> Result<impl Iterator<Item = MoveRec> + '_, DeltaError> {
    check("moves", bytes, RECORD)?;
    Ok(bytes.chunks_exact(RECORD).map(|r| {
        let (id, xyz, rgb, _) = get_record(r);
        MoveRec { id, xyz, rgb }
    }))
}

pub fn pack_removes(ids: &[u32]) -> Vec<u8> {
    ids.iter().flat_map(|id| id.to_le_bytes()).collect()
}

pub fn iter_removes(bytes: &[u8]) -> Result<impl Iterator<Item = u32> + '_, DeltaError> {
    check("removes", bytes, REMOVE_RECORD)?;
    Ok(bytes.chunks_exact(REMOVE_RECORD).map(|r| u32::from_le_bytes([r[0], r[1], r[2], r[3]])))
}

/// Checks buffer lengths and that every id is below `capacity`.
pub fn validate(d: &PointsDelta) -> Result<(), DeltaError> {
    let range = |field: &'static str, id: u32| {
        if id >= d.capacity {
            Err(DeltaError::IdOutOfRange { field, id, capacity: d.capacity })
        } else {
            Ok(())
        }
    };
    for a in iter_adds(&d.adds)? {
        range("adds", a.id)?;
    }
    for m in iter_moves(&d.moves)? {
        range("moves", m.id)?;
    }
    for id in iter_removes(&d.removes)? {
        range("removes", id)?;
    }
    Ok(())
}

/// Wraps a delta in a chunk: segment = zone, level = refined, version = seq,
/// generation = snapshot epoch, block id empty.
pub fn to_chunk(sequence: u64, d: &PointsDelta, generation: u32) -> model::Chunk {
    model::Chunk {
        sequence,
        kind: model::ChunkKind::PointsDelta as i32,
        segment: d.zone,
        level: model::Level::Refined as i32,
        block_id: Vec::new(),
        version: d.seq,
        generation,
        payload: d.encode_to_vec(),
        ..Default::default()
    }
}

/// Decodes the payload of a `POINTS_DELTA` chunk and validates it.
pub fn from_chunk(c: &model::Chunk) -> Result<PointsDelta, DeltaError> {
    if c.kind() != model::ChunkKind::PointsDelta {
        return Err(DeltaError::WrongKind(c.kind));
    }
    let d = PointsDelta::decode(c.payload.as_slice()).map_err(|e| DeltaError::Decode(e.to_string()))?;
    validate(&d)?;
    Ok(d)
}

impl PointsDelta {
    /// Checks buffer lengths and that every id is below `capacity`.
    pub fn validate(&self) -> Result<(), DeltaError> {
        validate(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> PointsDelta {
        PointsDelta {
            seq: 7,
            zone: 3,
            capacity: 10,
            total_points: 2,
            adds: pack_adds(&[
                AddRec { id: 0, xyz: [1.0, -2.5, 3.25], rgb: [1, 2, 3], zone: 3 },
                AddRec { id: 9, xyz: [0.0, 0.0, 0.0], rgb: [255, 0, 128], zone: 1 },
            ]),
            moves: pack_moves(&[MoveRec { id: 4, xyz: [5.0, 6.0, 7.0], rgb: [9, 8, 7] }]),
            removes: pack_removes(&[2, 5]),
            ..Default::default()
        }
    }

    #[test]
    fn round_trip() {
        let d = sample();
        d.validate().unwrap();
        assert_eq!(d.adds.len(), 2 * RECORD);
        let adds: Vec<_> = iter_adds(&d.adds).unwrap().collect();
        assert_eq!(adds[0], AddRec { id: 0, xyz: [1.0, -2.5, 3.25], rgb: [1, 2, 3], zone: 3 });
        assert_eq!(adds[1].id, 9);
        let moves: Vec<_> = iter_moves(&d.moves).unwrap().collect();
        assert_eq!(moves, vec![MoveRec { id: 4, xyz: [5.0, 6.0, 7.0], rgb: [9, 8, 7] }]);
        assert_eq!(iter_removes(&d.removes).unwrap().collect::<Vec<_>>(), vec![2, 5]);
    }

    #[test]
    fn bad_lengths_are_rejected() {
        assert!(matches!(iter_adds(&[0u8; 19]), Err(DeltaError::BadLength { field: "adds", .. })));
        assert!(matches!(iter_moves(&[0u8; 21]), Err(DeltaError::BadLength { field: "moves", .. })));
        assert!(matches!(iter_removes(&[0u8; 5]), Err(DeltaError::BadLength { field: "removes", .. })));
        let d = PointsDelta { capacity: 1, removes: vec![0; 3], ..Default::default() };
        assert!(d.validate().is_err());
    }

    #[test]
    fn id_at_or_above_capacity_is_rejected() {
        let mut d = sample();
        d.capacity = 9; // add id 9 is now out of range
        assert!(matches!(d.validate(), Err(DeltaError::IdOutOfRange { field: "adds", id: 9, capacity: 9 })));
        let d = PointsDelta { capacity: 3, removes: pack_removes(&[3]), ..Default::default() };
        assert!(matches!(d.validate(), Err(DeltaError::IdOutOfRange { field: "removes", .. })));
    }

    #[test]
    fn chunk_metadata_mapping() {
        let d = sample();
        let c = to_chunk(42, &d, 5);
        assert_eq!(c.sequence, 42);
        assert_eq!(c.kind(), model::ChunkKind::PointsDelta);
        assert_eq!(c.kind, 6);
        assert_eq!(c.segment, 3);
        assert_eq!(c.level(), model::Level::Refined);
        assert_eq!(c.version, 7);
        assert_eq!(c.generation, 5);
        assert!(c.block_id.is_empty());
        assert_eq!(from_chunk(&c).unwrap(), d);
    }

    #[test]
    fn from_chunk_rejects_wrong_kind_and_garbage() {
        let mut c = to_chunk(1, &sample(), 0);
        c.kind = model::ChunkKind::Points as i32;
        assert!(matches!(from_chunk(&c), Err(DeltaError::WrongKind(_))));
        c.kind = model::ChunkKind::PointsDelta as i32;
        c.payload = vec![0xff, 0xff];
        assert!(matches!(from_chunk(&c), Err(DeltaError::Decode(_))));
    }
}
