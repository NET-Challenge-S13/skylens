//! Incremental PLY reading: bytes arrive in pieces (network, file reads), points come out as soon
//! as their record is complete. Nothing already seen is scanned twice (PR #3 review).

use crate::ply::{Header, PlyError, Point, find_end_header, parse_header};

/// Longest prefix that can still turn into `end_header\r\n` once more bytes arrive.
const END_HEADER_TAIL: usize = b"end_header\r\n".len();
/// Headers longer than this are rejected rather than buffered forever.
const MAX_HEADER: usize = 64 * 1024;

#[derive(Debug, Default)]
pub struct PlyStream {
    /// Header bytes until `end_header`, afterwards a partial vertex record.
    buf: Vec<u8>,
    /// How far `buf` has been searched for `end_header`.
    scanned: usize,
    header: Option<Header>,
    /// Vertex records delivered so far.
    delivered: usize,
}

impl PlyStream {
    pub fn new() -> Self {
        Self::default()
    }

    /// The parsed header, once `end_header` has arrived.
    pub fn header(&self) -> Option<&Header> {
        self.header.as_ref()
    }

    /// Points delivered so far.
    pub fn delivered(&self) -> usize {
        self.delivered
    }

    /// True when every vertex announced by the header has been delivered.
    pub fn is_complete(&self) -> bool {
        self.header
            .as_ref()
            .is_some_and(|h| self.delivered == h.vertex_count)
    }

    /// Feeds the next piece of the file and calls `sink` for every point it completes.
    /// Bytes after the last vertex (other elements) are ignored.
    pub fn push(&mut self, mut bytes: &[u8], mut sink: impl FnMut(Point)) -> Result<(), PlyError> {
        if self.header.is_none() {
            self.buf.extend_from_slice(bytes);
            if self.buf.len() >= 4 && &self.buf[..4] != b"ply\n" && &self.buf[..4] != b"ply\r" {
                return Err(PlyError::NotPly);
            }
            let Some((_, header_len)) = find_end_header(&self.buf, self.scanned) else {
                if self.buf.len() > MAX_HEADER {
                    return Err(PlyError::BadHeader("header longer than 64 KiB".into()));
                }
                self.scanned = self.buf.len().saturating_sub(END_HEADER_TAIL);
                return Ok(());
            };
            let header = parse_header(&self.buf[..header_len])?;
            header.body_len()?;
            let rest = self.buf.split_off(header_len);
            self.buf.clear();
            self.header = Some(header);
            return self.push_body(&rest, &mut sink);
        }
        if self.is_complete() {
            bytes = &[];
        }
        self.push_body(bytes, &mut sink)
    }

    fn push_body(
        &mut self,
        mut bytes: &[u8],
        sink: &mut impl FnMut(Point),
    ) -> Result<(), PlyError> {
        let h = self.header.as_ref().expect("header parsed");
        let stride = h.stride;
        let remaining = |delivered: usize| h.vertex_count - delivered;

        // Complete a record split across pieces.
        if !self.buf.is_empty() {
            let need = stride - self.buf.len();
            let take = need.min(bytes.len());
            self.buf.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if self.buf.len() < stride {
                return Ok(());
            }
            if remaining(self.delivered) > 0 {
                sink(h.point(&self.buf));
                self.delivered += 1;
            }
            self.buf.clear();
        }

        let whole = (bytes.len() / stride).min(remaining(self.delivered));
        for rec in bytes[..whole * stride].chunks_exact(stride) {
            sink(h.point(rec));
        }
        self.delivered += whole;
        if remaining(self.delivered) > 0 {
            self.buf.extend_from_slice(&bytes[whole * stride..]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ply::points;

    fn sample(n: usize) -> Vec<u8> {
        let mut out = format!(
            "ply\r\nformat binary_little_endian 1.0\r\nelement vertex {n}\r\n\
             property float32 x\r\nproperty float32 y\r\nproperty float32 z\r\n\
             property uint8 red\r\nproperty uint8 green\r\nproperty uint8 blue\r\n\
             property float32 nx\r\nproperty float32 ny\r\nproperty float32 nz\r\nend_header\r\n"
        )
        .into_bytes();
        for i in 0..n {
            for v in [i as f32, -(i as f32), 0.5 * i as f32] {
                out.extend(v.to_le_bytes());
            }
            out.extend([(i % 256) as u8, 7, 9]);
            for v in [0.0f32, 0.0, 1.0] {
                out.extend(v.to_le_bytes());
            }
        }
        out.extend(b"trailing element bytes");
        out
    }

    fn feed(bytes: &[u8], piece: usize) -> Vec<Point> {
        let mut s = PlyStream::new();
        let mut got = Vec::new();
        for p in bytes.chunks(piece) {
            s.push(p, |pt| got.push(pt)).unwrap();
        }
        assert!(s.is_complete());
        got
    }

    #[test]
    fn same_points_for_every_piece_size() {
        let bytes = sample(300);
        let whole: Vec<Point> = points(&bytes).unwrap().collect();
        assert_eq!(whole.len(), 300);
        for piece in [1, 2, 3, 7, 26, 27, 28, 100, 4096, bytes.len()] {
            assert_eq!(feed(&bytes, piece), whole, "piece size {piece}");
        }
    }

    #[test]
    fn trailing_bytes_after_vertices_are_ignored() {
        let bytes = sample(5);
        let mut s = PlyStream::new();
        let mut n = 0;
        s.push(&bytes, |_| n += 1).unwrap();
        s.push(b"more junk", |_| n += 1).unwrap();
        assert_eq!(n, 5);
    }

    #[test]
    fn header_search_resumes_instead_of_rescanning() {
        let bytes = sample(1);
        let mut s = PlyStream::new();
        for b in bytes.chunks(1).take(40) {
            s.push(b, |_| {}).unwrap();
        }
        assert!(s.header().is_none());
        assert!(s.scanned > 0 && s.scanned <= s.buf.len());
        assert!(s.buf.len() - s.scanned <= END_HEADER_TAIL);
    }

    #[test]
    fn rejects_non_ply_early_and_huge_headers() {
        let mut s = PlyStream::new();
        assert_eq!(s.push(b"PK\x03\x04", |_| {}), Err(PlyError::NotPly));
        let mut s = PlyStream::new();
        let mut junk = b"ply\n".to_vec();
        junk.extend(std::iter::repeat_n(b'x', MAX_HEADER + 1));
        assert!(matches!(s.push(&junk, |_| {}), Err(PlyError::BadHeader(_))));
    }
}
