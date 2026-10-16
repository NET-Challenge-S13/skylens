//! Binary little-endian PLY reading for point clouds.

use std::fmt;

/// A PLY scalar type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl ScalarType {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            _ => return None,
        })
    }

    /// Size in bytes.
    pub fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

/// Errors from PLY parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlyError {
    /// `end_header` has not arrived yet; more bytes are needed.
    Incomplete,
    NotPly,
    UnsupportedFormat(String),
    UnsupportedProperty(String),
    MissingProperty(&'static str),
    BadHeader(String),
    /// The body is shorter than `vertex_count * stride`.
    Truncated {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for PlyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Incomplete => write!(f, "PLY header is incomplete"),
            Self::NotPly => write!(f, "not a PLY file"),
            Self::UnsupportedFormat(s) => write!(f, "unsupported PLY format: {s}"),
            Self::UnsupportedProperty(s) => write!(f, "unsupported PLY property: {s}"),
            Self::MissingProperty(s) => write!(f, "missing PLY property: {s}"),
            Self::BadHeader(s) => write!(f, "bad PLY header: {s}"),
            Self::Truncated { expected, actual } => {
                write!(
                    f,
                    "PLY body truncated: expected {expected} bytes, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for PlyError {}

/// Byte offsets of the properties a point cloud needs, within one vertex record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// Length of the header including the `end_header\n` line.
    pub header_len: usize,
    pub vertex_count: usize,
    /// Bytes per vertex record.
    pub stride: usize,
    /// Offsets of x, y, z (float32).
    pub position: [usize; 3],
    /// Offsets of red, green, blue (uint8), if present.
    pub color: Option<[usize; 3]>,
    /// Offsets of nx, ny, nz (float32), if present.
    pub normal: Option<[usize; 3]>,
}

impl Header {
    /// Bytes of vertex data, checked against overflow (hostile headers, 32-bit wasm).
    pub fn body_len(&self) -> Result<usize, PlyError> {
        self.vertex_count
            .checked_mul(self.stride)
            .ok_or_else(|| PlyError::BadHeader("vertex count × stride overflows".into()))
    }

    /// Decodes one vertex record of `self.stride` bytes.
    pub fn point(&self, rec: &[u8]) -> Point {
        let f = |o: usize| f32::from_le_bytes([rec[o], rec[o + 1], rec[o + 2], rec[o + 3]]);
        Point {
            position: [
                f(self.position[0]),
                f(self.position[1]),
                f(self.position[2]),
            ],
            color: self
                .color
                .map_or([255, 255, 255], |c| [rec[c[0]], rec[c[1]], rec[c[2]]]),
        }
    }
}

/// One point as read from the file.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub position: [f32; 3],
    /// sRGB 8-bit color; white when the file has no color.
    pub color: [u8; 3],
}

const END_HEADER: &[u8] = b"end_header";

/// Finds the `end_header` line, accepting `\n` or `\r\n` line endings, starting the search at
/// `from` so incremental callers do not rescan. Returns (start of the keyword, header length).
pub(crate) fn find_end_header(bytes: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut at = from.min(bytes.len());
    while let Some(i) = find(&bytes[at..], END_HEADER) {
        let start = at + i;
        let after = start + END_HEADER.len();
        match &bytes[after..] {
            [b'\n', ..] => return Some((start, after + 1)),
            [b'\r', b'\n', ..] => return Some((start, after + 2)),
            // The line ending has not arrived yet: the caller needs more bytes.
            [] | [b'\r'] => return None,
            _ => at = after,
        }
    }
    None
}

/// Parses the header at the start of `bytes`.
///
/// Returns [`PlyError::Incomplete`] if `end_header` has not been seen yet, so callers that
/// receive the file in pieces can retry with more bytes.
pub fn parse_header(bytes: &[u8]) -> Result<Header, PlyError> {
    if bytes.len() >= 4 && &bytes[..4] != b"ply\n" && &bytes[..4] != b"ply\r" {
        return Err(PlyError::NotPly);
    }
    let (end, header_len) = find_end_header(bytes, 0).ok_or(PlyError::Incomplete)?;
    let text = std::str::from_utf8(&bytes[..end])
        .map_err(|_| PlyError::BadHeader("header is not UTF-8".into()))?;

    let mut vertex_count = None;
    let mut in_vertex = false;
    let mut stride = 0usize;
    let mut props: Vec<(String, ScalarType, usize)> = Vec::new();

    for line in text.lines().skip(1) {
        let line = line.trim_end_matches('\r');
        let mut it = line.split_whitespace();
        match it.next() {
            Some("format") => {
                let fmt = it.next().unwrap_or("");
                if fmt != "binary_little_endian" {
                    return Err(PlyError::UnsupportedFormat(fmt.to_string()));
                }
            }
            Some("element") => {
                let name = it.next().unwrap_or("");
                let count: usize = it
                    .next()
                    .and_then(|c| c.parse().ok())
                    .ok_or_else(|| PlyError::BadHeader(line.to_string()))?;
                in_vertex = name == "vertex";
                if in_vertex {
                    vertex_count = Some(count);
                } else if vertex_count.is_none() && count > 0 {
                    return Err(PlyError::BadHeader(
                        "elements before vertex are not supported".into(),
                    ));
                }
            }
            Some("property") if in_vertex => {
                let ty = it.next().unwrap_or("");
                if ty == "list" {
                    return Err(PlyError::UnsupportedProperty(line.to_string()));
                }
                let ty = ScalarType::parse(ty)
                    .ok_or_else(|| PlyError::UnsupportedProperty(line.to_string()))?;
                let name = it
                    .next()
                    .ok_or_else(|| PlyError::BadHeader(line.to_string()))?;
                props.push((name.to_string(), ty, stride));
                stride += ty.size();
            }
            _ => {}
        }
    }

    let vertex_count = vertex_count.ok_or(PlyError::MissingProperty("element vertex"))?;
    let find_prop = |names: &[&str], ty: ScalarType| {
        props
            .iter()
            .find(|(n, t, _)| names.contains(&n.as_str()) && *t == ty)
            .map(|(_, _, off)| *off)
    };
    let pos =
        |n: &'static str| find_prop(&[n], ScalarType::F32).ok_or(PlyError::MissingProperty(n));
    let position = [pos("x")?, pos("y")?, pos("z")?];
    // Colors must be uint8. Other types (e.g. float 0..1) are reported instead of silently turning
    // white (PR #3 review).
    if let Some((name, ty, _)) = props.iter().find(|(n, t, _)| {
        ["red", "green", "blue", "r", "g", "b"].contains(&n.as_str()) && *t != ScalarType::U8
    }) {
        return Err(PlyError::UnsupportedProperty(format!(
            "{name}: color must be uint8, got {ty:?}"
        )));
    }
    let color = match (
        find_prop(&["red", "r"], ScalarType::U8),
        find_prop(&["green", "g"], ScalarType::U8),
        find_prop(&["blue", "b"], ScalarType::U8),
    ) {
        (Some(r), Some(g), Some(b)) => Some([r, g, b]),
        _ => None,
    };
    let normal = match (
        find_prop(&["nx"], ScalarType::F32),
        find_prop(&["ny"], ScalarType::F32),
        find_prop(&["nz"], ScalarType::F32),
    ) {
        (Some(x), Some(y), Some(z)) => Some([x, y, z]),
        _ => None,
    };

    Ok(Header {
        header_len,
        vertex_count,
        stride,
        position,
        color,
        normal,
    })
}

/// Iterates the points of a complete PLY file.
pub fn points(bytes: &[u8]) -> Result<impl Iterator<Item = Point> + '_, PlyError> {
    let h = parse_header(bytes)?;
    let body = &bytes[h.header_len..];
    let expected = h.body_len()?;
    if body.len() < expected {
        return Err(PlyError::Truncated {
            expected,
            actual: body.len(),
        });
    }
    Ok(body[..expected]
        .chunks_exact(h.stride)
        .map(move |rec| h.point(rec)))
}

/// True when `next` starts with every vertex record of `prev`, unchanged and in order, with the same
/// property layout — a preview that only appends points (docs/data/flight-01.md §2).
pub fn is_prefix(prev: &[u8], next: &[u8]) -> Result<bool, PlyError> {
    let a = parse_header(prev)?;
    let b = parse_header(next)?;
    let same_layout =
        (a.stride, a.position, a.color, a.normal) == (b.stride, b.position, b.color, b.normal);
    if !same_layout || b.vertex_count < a.vertex_count {
        return Ok(false);
    }
    let n = a.body_len()?;
    let (pa, pb) = (&prev[a.header_len..], &next[b.header_len..]);
    if pa.len() < n || pb.len() < n {
        return Err(PlyError::Truncated {
            expected: n,
            actual: pa.len().min(pb.len()),
        });
    }
    Ok(pa[..n] == pb[..n])
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ply(header_props: &str, records: &[Vec<u8>]) -> Vec<u8> {
        let mut out = format!(
            "ply\nformat binary_little_endian 1.0\nelement vertex {}\n{header_props}end_header\n",
            records.len()
        )
        .into_bytes();
        for r in records {
            out.extend_from_slice(r);
        }
        out
    }

    fn f(v: f32) -> [u8; 4] {
        v.to_le_bytes()
    }

    #[test]
    fn prefix_means_same_records_at_the_start() {
        let props = "property float x\nproperty float y\nproperty float z\n";
        let rec = |v: f32| [f(v), f(0.0), f(0.0)].concat();
        let a = ply(props, &[rec(1.0), rec(2.0)]);
        let b = ply(props, &[rec(1.0), rec(2.0), rec(3.0)]);
        let c = ply(props, &[rec(1.0), rec(9.0), rec(3.0)]);
        assert_eq!(is_prefix(&a, &b), Ok(true));
        assert_eq!(is_prefix(&a, &a), Ok(true));
        assert_eq!(is_prefix(&a, &c), Ok(false)); // a record changed
        assert_eq!(is_prefix(&b, &a), Ok(false)); // fewer points
        let other = ply(
            "property float y\nproperty float x\nproperty float z\n",
            &[rec(1.0), rec(2.0), rec(3.0)],
        );
        assert_eq!(is_prefix(&a, &other), Ok(false)); // same bytes, different layout
    }

    /// skyrecon `XyzRgbNormal`: x y z, red green blue, nx ny nz — 27 bytes.
    #[test]
    fn reads_xyz_rgb_normal() {
        let props = "property float32 x\nproperty float32 y\nproperty float32 z\n\
            property uint8 red\nproperty uint8 green\nproperty uint8 blue\n\
            property float32 nx\nproperty float32 ny\nproperty float32 nz\n";
        let mut rec = Vec::new();
        rec.extend(f(1.0));
        rec.extend(f(-2.5));
        rec.extend(f(3.25));
        rec.extend([10, 20, 30]);
        rec.extend(f(0.0));
        rec.extend(f(0.0));
        rec.extend(f(1.0));
        let bytes = ply(props, &[rec]);
        let h = parse_header(&bytes).unwrap();
        assert_eq!(h.stride, 27);
        assert_eq!(h.position, [0, 4, 8]);
        assert_eq!(h.color, Some([12, 13, 14]));
        assert_eq!(h.normal, Some([15, 19, 23]));
        let pts: Vec<_> = points(&bytes).unwrap().collect();
        assert_eq!(
            pts,
            vec![Point {
                position: [1.0, -2.5, 3.25],
                color: [10, 20, 30]
            }]
        );
    }

    /// skyrecon `XyzNormalRgb`: x y z, nx ny nz, red green blue with float/uchar names.
    #[test]
    fn reads_xyz_normal_rgb_by_name() {
        let props = "property float x\nproperty float y\nproperty float z\n\
            property float nx\nproperty float ny\nproperty float nz\n\
            property uchar red\nproperty uchar green\nproperty uchar blue\n";
        let mut rec = Vec::new();
        for v in [4.0, 5.0, 6.0, 0.0, 1.0, 0.0] {
            rec.extend(f(v));
        }
        rec.extend([1, 2, 3]);
        let bytes = ply(props, &[rec]);
        let pts: Vec<_> = points(&bytes).unwrap().collect();
        assert_eq!(pts[0].position, [4.0, 5.0, 6.0]);
        assert_eq!(pts[0].color, [1, 2, 3]);
    }

    /// 15-byte layout without normals.
    #[test]
    fn reads_without_normals() {
        let props = "property float32 x\nproperty float32 y\nproperty float32 z\n\
            property uint8 red\nproperty uint8 green\nproperty uint8 blue\n";
        let mut rec = Vec::new();
        for v in [7.0, 8.0, 9.0] {
            rec.extend(f(v));
        }
        rec.extend([4, 5, 6]);
        let bytes = ply(props, &[rec.clone(), rec]);
        let h = parse_header(&bytes).unwrap();
        assert_eq!((h.stride, h.normal), (15, None));
        assert_eq!(points(&bytes).unwrap().count(), 2);
    }

    #[test]
    fn missing_color_is_white() {
        let props = "property float32 x\nproperty float32 y\nproperty float32 z\n";
        let bytes = ply(props, &[[f(0.0), f(0.0), f(0.0)].concat()]);
        assert_eq!(
            points(&bytes).unwrap().next().unwrap().color,
            [255, 255, 255]
        );
    }

    #[test]
    fn accepts_crlf_header() {
        let bytes = b"ply\r\nformat binary_little_endian 1.0\r\nelement vertex 1\r\n\
            property float x\r\nproperty float y\r\nproperty float z\r\nend_header\r\n\
            \0\0\x80\x3f\0\0\0\x40\0\0\x40\x40";
        let h = parse_header(bytes).unwrap();
        assert_eq!(h.header_len, bytes.len() - 12);
        assert_eq!(
            points(bytes).unwrap().next().unwrap().position,
            [1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn end_header_split_across_pieces_is_incomplete_not_wrong() {
        let full = ply(
            "property float32 x\nproperty float32 y\nproperty float32 z\n",
            &[],
        );
        let cut = full.len() - 1; // "end_header" without its newline
        assert_eq!(parse_header(&full[..cut]), Err(PlyError::Incomplete));
        let crlf = b"ply\r\nformat binary_little_endian 1.0\r\nelement vertex 0\r\nend_header\r";
        assert_eq!(parse_header(crlf), Err(PlyError::Incomplete));
    }

    #[test]
    fn rejects_non_uint8_color() {
        let props = "property float32 x\nproperty float32 y\nproperty float32 z\n\
            property float red\nproperty float green\nproperty float blue\n";
        assert!(matches!(
            parse_header(&ply(props, &[])),
            Err(PlyError::UnsupportedProperty(_))
        ));
    }

    #[test]
    fn overflowing_vertex_count_is_an_error() {
        let bytes = format!(
            "ply\nformat binary_little_endian 1.0\nelement vertex {}\n\
             property float x\nproperty float y\nproperty float z\nend_header\n",
            usize::MAX / 4
        );
        assert!(matches!(
            points(bytes.as_bytes()).err(),
            Some(PlyError::BadHeader(_))
        ));
    }

    #[test]
    fn incomplete_header_asks_for_more_bytes() {
        let bytes = b"ply\nformat binary_little_endian 1.0\nelement vertex 3\n";
        assert_eq!(parse_header(bytes), Err(PlyError::Incomplete));
    }

    #[test]
    fn rejects_ascii_and_big_endian() {
        for fmt in ["ascii", "binary_big_endian"] {
            let bytes =
                format!("ply\nformat {fmt} 1.0\nelement vertex 0\nproperty float x\nend_header\n");
            assert_eq!(
                parse_header(bytes.as_bytes()),
                Err(PlyError::UnsupportedFormat(fmt.to_string()))
            );
        }
    }

    #[test]
    fn rejects_truncated_body() {
        let props = "property float32 x\nproperty float32 y\nproperty float32 z\n";
        let mut bytes = ply(props, &[[f(0.0), f(0.0), f(0.0)].concat()]);
        bytes.pop();
        assert_eq!(
            points(&bytes).err(),
            Some(PlyError::Truncated {
                expected: 12,
                actual: 11
            })
        );
    }

    #[test]
    fn rejects_non_ply_and_missing_position() {
        assert_eq!(parse_header(b"abcd"), Err(PlyError::NotPly));
        let bytes = ply("property float32 x\nproperty float32 y\n", &[]);
        assert_eq!(parse_header(&bytes), Err(PlyError::MissingProperty("z")));
    }
}
