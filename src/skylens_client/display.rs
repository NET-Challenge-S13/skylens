//! Display options of the control tower and the byte-level edits the 3D view makes
//! to pointblitz chunks before upload.
//!
//! SPEC §6: buildings have three display options over one scene, 점 / 검정 텍스처
//! (default) / 실사 항공뷰. Terrain and buildings arrive as pointblitz chunk v2
//! payloads coloured with the aerial drape, so the options are colour rewrites of
//! the chunk body, done on a copy before it goes to the GPU. Reconstruction points
//! are never recoloured.

use pointblitz::io::chunk::HEADER_LEN;
use std::borrow::Cow;

/// What a pointblitz chunk shows; decides how a display mode recolours it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Layer {
    /// Reconstruction points from skylens_model (`CHUNK_KIND_POINTS`).
    Recon,
    /// Terrain with the aerial drape (`CHUNK_KIND_TERRAIN`).
    Terrain,
    /// Buildings (`CHUNK_KIND_BUILDINGS`).
    Buildings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DisplayMode {
    /// 점: buildings as cyan points over a graded, dimmed drape.
    Points,
    /// 검정 텍스처 건물 (default): dark buildings over the graded drape.
    #[default]
    Black,
    /// 실사 항공뷰: the aerial colours as delivered, terrain and buildings alike.
    Aerial,
}

impl DisplayMode {
    pub const ALL: [DisplayMode; 3] = [DisplayMode::Points, DisplayMode::Black, DisplayMode::Aerial];

    pub fn id(self) -> &'static str {
        match self {
            DisplayMode::Points => "points",
            DisplayMode::Black => "black",
            DisplayMode::Aerial => "aerial",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "points" => Some(DisplayMode::Points),
            "black" => Some(DisplayMode::Black),
            "aerial" => Some(DisplayMode::Aerial),
            _ => None,
        }
    }
}

/// Point stride of a v2 chunk (12 or 8), or None if the header is too short.
fn stride(bytes: &[u8]) -> Option<usize> {
    let s = u16::from_le_bytes([*bytes.get(72)?, *bytes.get(73)?]);
    matches!(s, 8 | 12).then_some(usize::from(s))
}

/// Returns the chunk with its header generation and flags replaced. The 3D view
/// merges many independent streams into one pointblitz `Scene`, whose generation
/// rule would otherwise let one segment's refined snapshot swap out every other
/// segment; the view therefore renumbers chunks before inserting them.
pub fn retag(bytes: &[u8], generation: u32, flags: u32) -> Vec<u8> {
    let mut out = bytes.to_vec();
    if out.len() >= HEADER_LEN {
        out[8..12].copy_from_slice(&generation.to_le_bytes());
        out[20..24].copy_from_slice(&flags.to_le_bytes());
    }
    out
}

/// Rewrites the colours of a chunk for a display mode, in place.
pub fn recolor_in_place(layer: Layer, mode: DisplayMode, bytes: &mut [u8]) {
    let Some(f) = color_fn(layer, mode) else {
        return;
    };
    let Some(stride) = stride(bytes) else {
        return;
    };
    let body = &mut bytes[HEADER_LEN.min(bytes.len())..];
    for point in body.chunks_exact_mut(stride) {
        if stride == 12 {
            let [r, g, b] = f([point[8], point[9], point[10]]);
            point[8] = r;
            point[9] = g;
            point[10] = b;
        } else {
            let c = u16::from_le_bytes([point[6], point[7]]);
            let rgb = f(from_565(c));
            point[6..8].copy_from_slice(&pointblitz::io::chunk::rgb565(rgb[0], rgb[1], rgb[2]).to_le_bytes());
        }
    }
}

/// Returns the bytes to upload for a layer under a display mode (borrowed when unchanged).
pub fn recolor(layer: Layer, mode: DisplayMode, bytes: &[u8]) -> Cow<'_, [u8]> {
    if color_fn(layer, mode).is_none() {
        return Cow::Borrowed(bytes);
    }
    let mut out = bytes.to_vec();
    recolor_in_place(layer, mode, &mut out);
    Cow::Owned(out)
}

type ColorFn = fn([u8; 3]) -> [u8; 3];

fn color_fn(layer: Layer, mode: DisplayMode) -> Option<ColorFn> {
    match (layer, mode) {
        (Layer::Recon, _) | (_, DisplayMode::Aerial) => None,
        (Layer::Terrain, _) => Some(graded),
        (Layer::Buildings, DisplayMode::Black) => Some(black_texture),
        (Layer::Buildings, DisplayMode::Points) => Some(tactical_cyan),
    }
}

fn luma(c: [u8; 3]) -> f32 {
    (0.2126 * f32::from(c[0]) + 0.7152 * f32::from(c[1]) + 0.0722 * f32::from(c[2])) / 255.0
}

fn scale(tint: [f32; 3], k: f32) -> [u8; 3] {
    tint.map(|t| (t * k * 255.0).round().clamp(0.0, 255.0) as u8)
}

/// Cold, dim grade of the drape so drones and overlays own the brightest values.
fn graded(c: [u8; 3]) -> [u8; 3] {
    scale([0.42, 0.56, 0.66], 0.12 + 0.48 * luma(c))
}

/// Near-black buildings that keep a trace of the drape's shading.
fn black_texture(c: [u8; 3]) -> [u8; 3] {
    scale([0.55, 0.62, 0.72], 0.10 + 0.12 * luma(c))
}

/// Cyan building points (legacy tactical mode).
fn tactical_cyan(c: [u8; 3]) -> [u8; 3] {
    scale([0.37, 0.78, 1.0], 0.55 + 0.45 * luma(c))
}

fn from_565(c: u16) -> [u8; 3] {
    let r = ((c >> 11) & 31) as u32;
    let g = ((c >> 5) & 63) as u32;
    let b = (c & 31) as u32;
    [(r * 255 / 31) as u8, (g * 255 / 63) as u8, (b * 255 / 31) as u8]
}

#[cfg(test)]
mod tests {
    use super::*;
    use pointblitz::io::chunk::{decode_header, encode_with, FLAG_LAST_IN_GENERATION};
    use pointblitz::io::Point;

    fn points() -> Vec<Point> {
        vec![
            Point { position: [0.0, 0.0, 0.0], color: [200, 180, 160] },
            Point { position: [10.0, 5.0, 2.0], color: [20, 40, 60] },
        ]
    }

    #[test]
    fn retag_rewrites_generation_and_flags_only() {
        let bytes = encode_with(3, 7, 0, &points(), 12);
        let out = retag(&bytes, 42, FLAG_LAST_IN_GENERATION);
        let h = decode_header(&out).unwrap();
        assert_eq!(h.generation, 42);
        assert_eq!(h.flags, FLAG_LAST_IN_GENERATION);
        assert_eq!(h.index, 7);
        assert_eq!(&out[24..], &bytes[24..]);
    }

    #[test]
    fn recon_and_aerial_are_untouched() {
        let bytes = encode_with(0, 0, 0, &points(), 12);
        assert!(matches!(recolor(Layer::Recon, DisplayMode::Black, &bytes), Cow::Borrowed(_)));
        assert!(matches!(recolor(Layer::Buildings, DisplayMode::Aerial, &bytes), Cow::Borrowed(_)));
        assert!(matches!(recolor(Layer::Terrain, DisplayMode::Aerial, &bytes), Cow::Borrowed(_)));
    }

    #[test]
    fn black_mode_darkens_buildings_stride_12() {
        let bytes = encode_with(0, 0, 0, &points(), 12);
        let out = recolor(Layer::Buildings, DisplayMode::Black, &bytes);
        // Header and positions unchanged, still a valid chunk.
        assert_eq!(&out[..HEADER_LEN], &bytes[..HEADER_LEN]);
        decode_header(&out).unwrap();
        for p in out[HEADER_LEN..].chunks_exact(12) {
            assert!(p[8].max(p[9]).max(p[10]) < 60, "dark: {:?}", &p[8..11]);
            assert_eq!(p[11], 255);
        }
        for (a, b) in out[HEADER_LEN..].chunks_exact(12).zip(bytes[HEADER_LEN..].chunks_exact(12)) {
            assert_eq!(&a[..8], &b[..8]);
        }
    }

    #[test]
    fn points_mode_turns_buildings_cyan_stride_8() {
        let bytes = encode_with(0, 0, 0, &points(), 8);
        let out = recolor(Layer::Buildings, DisplayMode::Points, &bytes);
        decode_header(&out).unwrap();
        for p in out[HEADER_LEN..].chunks_exact(8) {
            let [r, g, b] = from_565(u16::from_le_bytes([p[6], p[7]]));
            assert!(b > r && g > r, "cyan-ish: {r} {g} {b}");
        }
    }

    #[test]
    fn display_mode_ids_round_trip() {
        for m in DisplayMode::ALL {
            assert_eq!(DisplayMode::parse(m.id()), Some(m));
        }
        assert_eq!(DisplayMode::default(), DisplayMode::Black);
        assert_eq!(DisplayMode::parse("nope"), None);
    }
}
