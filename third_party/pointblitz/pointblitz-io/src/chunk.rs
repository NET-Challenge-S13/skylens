//! PointBlitz chunk format v2 (decisions 0008, 0022, 0051): what the server sends and the GPU consumes.
//!
//! ```text
//! offset size  field
//!      0    4  magic "PBCK"
//!      4    2  version = 2            (u16 LE)
//!      6    2  header length = 80     (u16 LE)
//!      8    4  generation             (u32 LE)  refined snapshots start a new generation
//!     12    4  index within generation (u32 LE)
//!     16    4  point count            (u32 LE)
//!     20    4  flags                  (u32 LE)  bit 0: last chunk of its generation
//!                                               bit 1: the coarse first pass is complete
//!     24   24  origin x, y, z         (f64 LE, ENU metres)
//!     48   12  bbox min x, y, z       (f32 LE, relative to origin)
//!     60   12  bbox max x, y, z       (f32 LE, relative to origin)
//!     72    2  point stride = 12 or 8 (u16 LE)
//!     74    6  reserved, zero
//!     80  stride·n points:
//!          stride 12: position u16×3, zero u16, color u8×4 (sRGB, a = 255)
//!          stride 8:  position u16×3, color RGB565 (u16)
//! ```
//!
//! Positions are quantised to the chunk's bounding box: `p = origin + bbox_min + q / 65535 ×
//! (bbox_max − bbox_min)`, per axis. The body after byte 80 is a wgpu vertex buffer as it is
//! (`Uint16x4` at 0, and for stride 12 `Unorm8x4` at 8); the GPU undoes the quantisation, so a
//! client still copies it without parsing. Version 1 (f32 positions, stride 16) is not read any more.

use crate::ply::Point;

pub const MAGIC: [u8; 4] = *b"PBCK";
pub const VERSION: u16 = 2;
pub const HEADER_LEN: usize = 80;
/// Set on the last chunk of a generation: the client may swap generations after it.
pub const FLAG_LAST_IN_GENERATION: u32 = 1;
/// Set on the chunk that completes the coarse first pass of a full delivery (decision 0051): a
/// client may swap to the new generation there and fill the rest in as it arrives.
pub const FLAG_FIRST_PASS_COMPLETE: u32 = 2;
/// Point strides of format v2 (decision 0051): RGBA8 colour, or RGB565 colour.
pub const STRIDES: [usize; 2] = [12, 8];
/// The stride new chunks are written with, unless [`set_stride`] chose the other one.
pub const DEFAULT_STRIDE: usize = 12;

static STRIDE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(DEFAULT_STRIDE);

/// Chooses the stride [`encode`] writes (12 or 8) for this process — to compare the two
/// (decision 0051). Other values are ignored.
pub fn set_stride(stride: usize) {
    if STRIDES.contains(&stride) {
        STRIDE.store(stride, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The stride [`encode`] writes.
pub fn stride() -> usize {
    STRIDE.load(std::sync::atomic::Ordering::Relaxed)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChunkHeader {
    pub generation: u32,
    pub index: u32,
    pub point_count: u32,
    pub flags: u32,
    pub origin: [f64; 3],
    pub bbox_min: [f32; 3],
    pub bbox_max: [f32; 3],
    /// Bytes per point: 12 or 8.
    pub stride: u16,
}

impl ChunkHeader {
    /// Position of quantised value `q` (0 … 65535 per axis), relative to the origin.
    pub fn dequantise(&self, q: [u16; 3]) -> [f32; 3] {
        [0, 1, 2].map(|i| {
            self.bbox_min[i] + f32::from(q[i]) / 65535.0 * (self.bbox_max[i] - self.bbox_min[i])
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkError {
    TooShort,
    BadMagic,
    UnsupportedVersion(u16),
    BadHeaderLength(u16),
    UnknownFlags(u32),
    ReservedNotZero,
    /// A point stride other than 12 or 8.
    BadStride(u16),
    /// Body length does not match the point count.
    BadLength {
        expected: usize,
        actual: usize,
    },
}

impl std::fmt::Display for ChunkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid chunk: {self:?}")
    }
}

impl std::error::Error for ChunkError {}

/// Encodes points into one chunk with the process stride ([`stride`]).
pub fn encode(generation: u32, index: u32, flags: u32, points: &[Point]) -> Vec<u8> {
    encode_with(generation, index, flags, points, stride())
}

/// Encodes points into one chunk with point stride 12 or 8 (decision 0051). The origin is chosen per
/// chunk (decision 0013) and positions are quantised to the chunk's bounding box.
pub fn encode_with(
    generation: u32,
    index: u32,
    flags: u32,
    points: &[Point],
    stride: usize,
) -> Vec<u8> {
    assert!(STRIDES.contains(&stride), "stride 12 or 8");
    let (min, max) = bounds(points);
    // Origin: bbox centre rounded to whole metres (stable, readable, exact in f64).
    let origin = [0, 1, 2].map(|i| ((min[i] + max[i]) / 2.0).round());
    let rel = |p: [f64; 3]| [0, 1, 2].map(|i| (p[i] - origin[i]) as f32);
    let mut out = Vec::with_capacity(HEADER_LEN + stride * points.len());
    out.extend(MAGIC);
    out.extend(VERSION.to_le_bytes());
    out.extend((HEADER_LEN as u16).to_le_bytes());
    out.extend(generation.to_le_bytes());
    out.extend(index.to_le_bytes());
    out.extend(
        u32::try_from(points.len())
            .expect("chunk holds < 2^32 points")
            .to_le_bytes(),
    );
    out.extend(flags.to_le_bytes());
    for v in origin {
        out.extend(v.to_le_bytes());
    }
    let (bmin, bmax) = if points.is_empty() {
        ([0.0; 3], [0.0; 3])
    } else {
        (rel(min), rel(max))
    };
    for v in bmin.into_iter().chain(bmax) {
        out.extend(v.to_le_bytes());
    }
    out.extend((stride as u16).to_le_bytes());
    out.extend([0u8; 6]);
    debug_assert_eq!(out.len(), HEADER_LEN);
    // Quantise against the stored f32 box, exactly as the GPU will undo it.
    let scale = [0, 1, 2].map(|i| {
        let ext = f64::from(bmax[i]) - f64::from(bmin[i]);
        if ext > 0.0 { 65535.0 / ext } else { 0.0 }
    });
    for p in points {
        let r = rel(p.position.map(f64::from));
        for i in 0..3 {
            let q = ((f64::from(r[i]) - f64::from(bmin[i])) * scale[i]).round();
            out.extend((q.clamp(0.0, 65535.0) as u16).to_le_bytes());
        }
        let [cr, cg, cb] = p.color;
        if stride == 12 {
            out.extend([0, 0, cr, cg, cb, 255]);
        } else {
            out.extend(rgb565(cr, cg, cb).to_le_bytes());
        }
    }
    out
}

/// sRGB bytes → RGB565 (rounded).
pub fn rgb565(r: u8, g: u8, b: u8) -> u16 {
    let q = |v: u8, bits: u32| ((u32::from(v) * ((1 << bits) - 1) + 127) / 255) as u16;
    (q(r, 5) << 11) | (q(g, 6) << 5) | q(b, 5)
}

fn bounds(points: &[Point]) -> ([f64; 3], [f64; 3]) {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for p in points {
        for i in 0..3 {
            let v = f64::from(p.position[i]);
            min[i] = min[i].min(v);
            max[i] = max[i].max(v);
        }
    }
    if points.is_empty() {
        ([0.0; 3], [0.0; 3])
    } else {
        (min, max)
    }
}

/// Validates a chunk and returns its header.
pub fn decode_header(bytes: &[u8]) -> Result<ChunkHeader, ChunkError> {
    if bytes.len() < HEADER_LEN {
        return Err(ChunkError::TooShort);
    }
    if bytes[..4] != MAGIC {
        return Err(ChunkError::BadMagic);
    }
    let u16_at = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let f32_at = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let f64_at = |o: usize| f64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    let version = u16_at(4);
    if version != VERSION {
        return Err(ChunkError::UnsupportedVersion(version));
    }
    if usize::from(u16_at(6)) != HEADER_LEN {
        return Err(ChunkError::BadHeaderLength(u16_at(6)));
    }
    let point_count = u32_at(16);
    let stride = u16_at(72);
    if !STRIDES.contains(&usize::from(stride)) {
        return Err(ChunkError::BadStride(stride));
    }
    let expected = (point_count as usize)
        .checked_mul(usize::from(stride))
        .and_then(|b| b.checked_add(HEADER_LEN))
        .ok_or(ChunkError::BadLength {
            expected: usize::MAX,
            actual: bytes.len(),
        })?;
    if bytes.len() != expected {
        return Err(ChunkError::BadLength {
            expected,
            actual: bytes.len(),
        });
    }
    // v2 defines two flag bits and six zero reserved bytes; anything else is not a v2 chunk.
    let flags = u32_at(20);
    if flags & !(FLAG_LAST_IN_GENERATION | FLAG_FIRST_PASS_COMPLETE) != 0 {
        return Err(ChunkError::UnknownFlags(flags));
    }
    if bytes[74..80].iter().any(|b| *b != 0) {
        return Err(ChunkError::ReservedNotZero);
    }
    Ok(ChunkHeader {
        generation: u32_at(8),
        index: u32_at(12),
        point_count,
        flags: u32_at(20),
        origin: [f64_at(24), f64_at(32), f64_at(40)],
        bbox_min: [f32_at(48), f32_at(52), f32_at(56)],
        bbox_max: [f32_at(60), f32_at(64), f32_at(68)],
        stride,
    })
}

/// The vertex data of a validated chunk (stride 12 or 8), ready to copy into a GPU buffer.
pub fn vertex_bytes(bytes: &[u8]) -> &[u8] {
    &bytes[HEADER_LEN..]
}

/// Splits a complete PLY file into chunks of at most `max_points` points for one generation.
/// The last chunk carries [`FLAG_LAST_IN_GENERATION`].
pub fn ply_to_chunks(
    ply: &[u8],
    generation: u32,
    max_points: usize,
) -> Result<Vec<Vec<u8>>, crate::PlyError> {
    let mut out = Vec::new();
    ply_tail_to_chunks(ply, generation, 0, max_points, |c| {
        out.push(c);
        true
    })?;
    Ok(out)
}

/// Encodes the points of `ply` from record `skip` on, at most `max_points` per chunk, and hands
/// each chunk to `sink` as soon as it is encoded so a server can stream it (P1.4). Only one chunk's
/// worth of points is held at a time (PR #8 review); skipped records are not decoded.
///
/// The last chunk of the delivery carries [`FLAG_LAST_IN_GENERATION`], and at least one chunk is
/// produced (empty when nothing is left), so a client always learns that a delivery is complete.
/// `index` counts chunks within the delivery. Returns the number of chunks handed over; stops early
/// when `sink` returns false.
pub fn ply_tail_to_chunks(
    ply: &[u8],
    generation: u32,
    skip: usize,
    max_points: usize,
    mut sink: impl FnMut(Vec<u8>) -> bool,
) -> Result<u32, crate::PlyError> {
    assert!(max_points > 0);
    let h = crate::ply::parse_header(ply)?;
    let n = h.body_len()?;
    let body = &ply[h.header_len..];
    if body.len() < n {
        return Err(crate::PlyError::Truncated {
            expected: n,
            actual: body.len(),
        });
    }
    let skip = skip.min(h.vertex_count);
    let records = &body[skip * h.stride..n];
    let count = chunk_count(h.vertex_count - skip, max_points);
    for i in 0..count {
        let flags = if i + 1 == count {
            FLAG_LAST_IN_GENERATION
        } else {
            0
        };
        let part = chunk_records(&h, records, max_points, i);
        if !sink(encode_records(&h, part, generation, i as u32, flags)) {
            return Ok(i as u32 + 1);
        }
    }
    Ok(count as u32)
}

/// Chunks needed for `points` points: at least one, since an empty delivery still closes.
pub fn chunk_count(points: usize, max_points: usize) -> usize {
    assert!(max_points > 0);
    points.div_ceil(max_points).max(1)
}

/// The records of chunk `i` when `records` is split into chunks of `max_points` points.
pub fn chunk_records<'a>(
    h: &crate::Header,
    records: &'a [u8],
    max_points: usize,
    i: usize,
) -> &'a [u8] {
    let start = (i * max_points * h.stride).min(records.len());
    let end = ((i + 1) * max_points * h.stride).min(records.len());
    &records[start..end]
}

/// Encodes whole vertex records (`h.stride` bytes each) as one chunk. Chunks are independent, so a
/// server can encode the chunks of one delivery in parallel and still send them in order.
pub fn encode_records(
    h: &crate::Header,
    records: &[u8],
    generation: u32,
    index: u32,
    flags: u32,
) -> Vec<u8> {
    let points: Vec<Point> = records.chunks_exact(h.stride).map(|r| h.point(r)).collect();
    encode(generation, index, flags, &points)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(n: usize, offset: f32) -> Vec<Point> {
        (0..n)
            .map(|i| Point {
                position: [offset + i as f32 * 0.25, -(i as f32), 10.0],
                color: [i as u8, 2, 3],
            })
            .collect()
    }

    #[test]
    fn layout_is_header_80_plus_stride_per_point() {
        for stride in STRIDES {
            let c = encode_with(3, 1, FLAG_LAST_IN_GENERATION, &pts(5, 0.0), stride);
            assert_eq!(c.len(), 80 + stride * 5);
            assert_eq!(&c[..4], b"PBCK");
            let h = decode_header(&c).unwrap();
            assert_eq!(
                (h.generation, h.index, h.point_count, h.flags, h.stride),
                (3, 1, 5, 1, stride as u16)
            );
            // First point: x is the box minimum (q = 0), then the colour.
            let v = vertex_bytes(&c);
            assert_eq!(&v[0..2], &[0, 0]);
            if stride == 12 {
                assert_eq!(&v[6..12], &[0, 0, 0, 2, 3, 255]);
            } else {
                assert_eq!(&v[6..8], &rgb565(0, 2, 3).to_le_bytes());
            }
        }
    }

    #[test]
    fn positions_round_trip_within_half_a_step() {
        // 5 km from the dataset origin; the chunk spans 24.75 m in x, 99 m in y (decision 0051).
        let input = pts(100, 5000.0);
        for stride in STRIDES {
            let c = encode_with(0, 0, 0, &input, stride);
            let h = decode_header(&c).unwrap();
            let step = [0, 1, 2].map(|k| (h.bbox_max[k] - h.bbox_min[k]) / 65535.0);
            for (i, rec) in vertex_bytes(&c).chunks_exact(stride).enumerate() {
                let q = [0, 1, 2].map(|k| u16::from_le_bytes([rec[2 * k], rec[2 * k + 1]]));
                let back = h.dequantise(q);
                for k in 0..3 {
                    let want = f64::from(input[i].position[k]) - h.origin[k];
                    let err = (f64::from(back[k]) - want).abs();
                    assert!(err <= f64::from(step[k]) / 2.0 + 1e-4, "axis {k}: {err} m");
                }
            }
            assert!(h.bbox_min[0] <= 0.0 && h.bbox_max[0] >= 0.0);
        }
    }

    #[test]
    fn rgb565_keeps_the_ends() {
        assert_eq!(rgb565(0, 0, 0), 0);
        assert_eq!(rgb565(255, 255, 255), 0xffff);
        assert_eq!(rgb565(255, 0, 0), 0xf800);
        assert_eq!(rgb565(0, 255, 0), 0x07e0);
        assert_eq!(rgb565(0, 0, 255), 0x001f);
    }

    #[test]
    fn other_strides_and_flags_are_rejected() {
        let mut c = encode_with(0, 0, 0, &pts(2, 0.0), 12);
        c[72] = 16;
        assert_eq!(decode_header(&c), Err(ChunkError::BadStride(16)));
        let mut c = encode_with(0, 0, 0, &pts(2, 0.0), 12);
        c[20] = 4;
        assert_eq!(decode_header(&c), Err(ChunkError::UnknownFlags(4)));
        let mut c = encode_with(0, 0, 0, &pts(2, 0.0), 12);
        c[4] = 1; // version 1
        assert_eq!(decode_header(&c), Err(ChunkError::UnsupportedVersion(1)));
    }

    #[test]
    fn tail_skips_records_and_closes_the_delivery() {
        let mut ply = b"ply\nformat binary_little_endian 1.0\nelement vertex 5\n\
            property float x\nproperty float y\nproperty float z\n\
            property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
            .to_vec();
        for i in 0..5u8 {
            for v in [f32::from(i), 0.0, 0.0] {
                ply.extend(v.to_le_bytes());
            }
            ply.extend([i, 0, 0]);
        }
        let collect = |skip: usize, max: usize| {
            let mut got = Vec::new();
            let n = ply_tail_to_chunks(&ply, 7, skip, max, |c| {
                got.push(c);
                true
            })
            .unwrap();
            assert_eq!(n as usize, got.len());
            got
        };
        let got = collect(3, 1);
        let h: Vec<_> = got
            .iter()
            .map(|c| {
                let h = decode_header(c).unwrap();
                (h.generation, h.index, h.point_count, h.flags)
            })
            .collect();
        assert_eq!(h, [(7, 0, 1, 0), (7, 1, 1, FLAG_LAST_IN_GENERATION)]);
        assert_eq!(&vertex_bytes(&got[0])[8..12], &[3, 0, 0, 255]);
        // Nothing left: one empty chunk still closes the delivery.
        let empty = collect(9, 4);
        assert_eq!(empty.len(), 1);
        let e = decode_header(&empty[0]).unwrap();
        assert_eq!((e.point_count, e.flags), (0, FLAG_LAST_IN_GENERATION));
        // The sink can stop the delivery.
        let mut seen = 0;
        let n = ply_tail_to_chunks(&ply, 0, 0, 1, |_| {
            seen += 1;
            seen < 2
        })
        .unwrap();
        assert_eq!((n, seen), (2, 2));
    }

    #[test]
    fn rejects_bad_chunks() {
        let c = encode(0, 0, 0, &pts(2, 0.0));
        assert_eq!(decode_header(&c[..79]), Err(ChunkError::TooShort));
        let mut bad = c.clone();
        bad[0] = b'X';
        assert_eq!(decode_header(&bad), Err(ChunkError::BadMagic));
        let mut v3 = c.clone();
        v3[4] = 3;
        assert_eq!(decode_header(&v3), Err(ChunkError::UnsupportedVersion(3)));
        assert!(matches!(
            decode_header(&c[..c.len() - 1]),
            Err(ChunkError::BadLength { .. })
        ));
        let mut flags = c.clone();
        flags[20] = 0b100;
        assert_eq!(decode_header(&flags), Err(ChunkError::UnknownFlags(4)));
        let mut reserved = c.clone();
        reserved[75] = 1;
        assert_eq!(decode_header(&reserved), Err(ChunkError::ReservedNotZero));
    }

    #[test]
    fn empty_chunk_is_valid() {
        let c = encode(9, 0, FLAG_LAST_IN_GENERATION, &[]);
        assert_eq!(decode_header(&c).unwrap().point_count, 0);
        assert_eq!(c.len(), 80);
    }

    #[test]
    fn ply_splits_into_flagged_chunks() {
        let mut ply = b"ply\nformat binary_little_endian 1.0\nelement vertex 10\n\
             property float x\nproperty float y\nproperty float z\n\
             property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
            .to_vec();
        for i in 0..10u8 {
            for v in [f32::from(i), 0.0, 0.0] {
                ply.extend(v.to_le_bytes());
            }
            ply.extend([i, i, i]);
        }
        let chunks = ply_to_chunks(&ply, 4, 4).unwrap();
        let heads: Vec<_> = chunks.iter().map(|c| decode_header(c).unwrap()).collect();
        assert_eq!(
            heads.iter().map(|h| h.point_count).collect::<Vec<_>>(),
            [4, 4, 2]
        );
        assert_eq!(heads.iter().map(|h| h.flags).collect::<Vec<_>>(), [0, 0, 1]);
        assert!(heads.iter().all(|h| h.generation == 4));
    }
}
