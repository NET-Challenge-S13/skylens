//! Single-line text for the canvases the tower paints itself (planner map labels). There is no 2D
//! canvas API in Blitz, so text is laid out here from the bundled fonts: one glyph per character
//! through the font's character map and advance widths, no shaping (labels are ASCII digits and
//! letters).

use skrifa::MetadataProvider as _;
use skrifa::instance::{LocationRef, Size};
use skylens_viewer::anyrender::{Glyph, PaintScene as _, Scene};
use skylens_viewer::peniko::kurbo::{Affine, Vec2};
use skylens_viewer::peniko::{Blob, Color, Fill, FontData};
use std::sync::{Arc, OnceLock};

/// Which bundled face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Face {
    Regular,
    Bold,
}

fn bytes(face: Face) -> &'static [u8] {
    let want = match face {
        Face::Regular => 1,
        Face::Bold => 3,
    };
    // BUNDLED_FONTS lists Barlow Regular, Medium, SemiBold as entries 1..=3.
    skylens_viewer::shell::BUNDLED_FONTS.get(want).map(|(_, b)| *b).unwrap_or(skylens_viewer::shell::BUNDLED_FONTS[0].1)
}

fn font_data(face: Face) -> &'static FontData {
    static REGULAR: OnceLock<FontData> = OnceLock::new();
    static BOLD: OnceLock<FontData> = OnceLock::new();
    let cell = match face {
        Face::Regular => &REGULAR,
        Face::Bold => &BOLD,
    };
    cell.get_or_init(|| {
        let b: &'static [u8] = bytes(face);
        FontData::new(Blob::new(Arc::new(b) as Arc<dyn AsRef<[u8]> + Send + Sync>), 0)
    })
}

/// Horizontal anchor of the text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Start,
    Center,
}

/// Vertical anchor of the text (canvas `textBaseline`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Baseline {
    Alphabetic,
    Middle,
}

/// Lays out `text` at `size` px: glyph ids and pen positions (x along the baseline), and the
/// total advance.
pub fn layout(face: Face, size: f32, text: &str) -> (Vec<Glyph>, f32, f32, f32) {
    let Ok(font) = skrifa::FontRef::new(bytes(face)) else { return (Vec::new(), 0.0, 0.0, 0.0) };
    let charmap = font.charmap();
    let metrics = font.glyph_metrics(Size::new(size), LocationRef::default());
    let line = font.metrics(Size::new(size), LocationRef::default());
    let mut x = 0.0f32;
    let mut glyphs = Vec::new();
    for ch in text.chars() {
        let gid = charmap.map(ch).unwrap_or_default();
        glyphs.push(Glyph { id: gid.to_u32(), x, y: 0.0 });
        x += metrics.advance_width(gid).unwrap_or(size * 0.5);
    }
    (glyphs, x, line.ascent, line.descent)
}

/// Draws one line of text in `transform`'s space with the given anchors.
#[allow(clippy::too_many_arguments)]
pub fn draw_text(scene: &mut Scene, transform: Affine, face: Face, size: f32, text: &str, x: f64, y: f64, align: Align, baseline: Baseline, color: Color) {
    let (glyphs, width, ascent, descent) = layout(face, size, text);
    if glyphs.is_empty() {
        return;
    }
    let dx = match align {
        Align::Start => 0.0,
        Align::Center => -f64::from(width) / 2.0,
    };
    let dy = match baseline {
        Baseline::Alphabetic => 0.0,
        // Canvas "middle" is the middle of the em box; ascent/descent approximate it.
        Baseline::Middle => f64::from(ascent + descent) / 2.0,
    };
    let t = transform * Affine::translate(Vec2::new(x + dx, y + dy));
    scene.draw_glyphs(font_data(face), size, false, &[], Vec2::ZERO, Fill::NonZero, color, 1.0, t, None, glyphs.into_iter());
}
