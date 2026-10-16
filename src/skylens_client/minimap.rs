//! Minimap "MINIMAP · TOP-DOWN" (legacy `ui/minimap.ts`), top-right: live drones, the route, the
//! reconstructed segments and the detections, top-down in the ENU frame (north up).
//!
//! The extent follows the content: a square, padded box around the drones, detections and route,
//! eased toward changes (0.12 per 60 fps frame), so a drone flying off the edge pulls the view with
//! it. Drawn by a Blitz custom widget mounted on the legacy `canvas.minimap__canvas`, in CSS pixels
//! like the legacy 2D canvas.

use crate::anyrender::{self, PaintScene as _};
use crate::peniko::kurbo::{Affine, BezPath, Circle, Rect, Stroke};
use crate::peniko::{Blob, Color, Fill, FontData};
use blitz_dom::Widget;
use blitz_dom::node::ComputedStyles;
use skylens_viewer::camera::per_frame;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

/// Map extent in ENU metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Extent {
    pub min_e: f64,
    pub max_e: f64,
    pub min_n: f64,
    pub max_n: f64,
}

/// Before anything live arrives (the legacy fallback when the scene box was degenerate).
pub const EMPTY_EXTENT: Extent = Extent { min_e: -20.0, max_e: 20.0, min_n: -20.0, max_n: 20.0 };

/// Pads 15 % per side (10 m when the span is zero).
pub fn pad(x: Extent) -> Extent {
    let pe = (x.max_e - x.min_e) * 0.15;
    let pn = (x.max_n - x.min_n) * 0.15;
    let pe = if pe == 0.0 { 10.0 } else { pe };
    let pn = if pn == 0.0 { 10.0 } else { pn };
    Extent { min_e: x.min_e - pe, max_e: x.max_e + pe, min_n: x.min_n - pn, max_n: x.max_n + pn }
}

/// A square, padded box around the points, at least 120 m across (a single parked drone must not be
/// magnified until noise reads as movement). Square, so a straight flight never looks bent.
pub fn target_extent(points: &[[f64; 2]]) -> Extent {
    if points.is_empty() {
        return pad(EMPTY_EXTENT);
    }
    let (mut e0, mut e1, mut n0, mut n1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for [e, n] in points {
        e0 = e0.min(*e);
        e1 = e1.max(*e);
        n0 = n0.min(*n);
        n1 = n1.max(*n);
    }
    let (cx, cy) = ((e0 + e1) / 2.0, (n0 + n1) / 2.0);
    let half = ((e1 - e0) / 2.0).max((n1 - n0) / 2.0).max(60.0);
    pad(Extent { min_e: cx - half, max_e: cx + half, min_n: cy - half, max_n: cy + half })
}

/// Eases `view` toward `to` by `k`.
pub fn ease(view: Extent, to: Extent, k: f64) -> Extent {
    Extent {
        min_e: view.min_e + (to.min_e - view.min_e) * k,
        max_e: view.max_e + (to.max_e - view.max_e) * k,
        min_n: view.min_n + (to.min_n - view.min_n) * k,
        max_n: view.max_n + (to.max_n - view.max_n) * k,
    }
}

/// ENU to map pixels (north up).
pub fn project(view: &Extent, e: f64, n: f64, w: f64, h: f64) -> (f64, f64) {
    let span_e = if view.max_e - view.min_e == 0.0 { 1.0 } else { view.max_e - view.min_e };
    let span_n = if view.max_n - view.min_n == 0.0 { 1.0 } else { view.max_n - view.min_n };
    ((e - view.min_e) / span_e * w, h - (n - view.min_n) / span_n * h)
}

/// The scale bar: a round number of metres about 28 % of the width, its length in pixels, and its
/// label (`"50 m"`, `"2 km"`).
pub fn scale_bar(view: &Extent, w: f64) -> (f64, f64, String) {
    let metres_per_px = (view.max_e - view.min_e) / w.max(1.0);
    let rough = metres_per_px * (w * 0.28);
    let pow = 10f64.powf(rough.max(1.0).log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0].iter().map(|m| m * pow).find(|v| *v >= rough).unwrap_or(pow * 10.0);
    let px = step / metres_per_px;
    let label = if step >= 1000.0 {
        format!("{} km", crate::points::js_number(step / 1000.0))
    } else {
        format!("{} m", crate::points::js_number(step))
    };
    (step, px, label)
}

/// What the map shows, refreshed by the board every poll.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MapData {
    /// `[e, n]`, and whether it is the active drone.
    pub drones: Vec<([f64; 2], bool)>,
    /// `[e, n]`, and whether it is a person (else a hazard).
    pub detections: Vec<([f64; 2], bool)>,
    pub route: Vec<[f64; 2]>,
    /// Centres of the reconstructed segments.
    pub chunks: Vec<[f64; 2]>,
}

impl MapData {
    /// What the extent fits: drones, detections and the route (the reconstruction is backdrop).
    pub fn fit_points(&self) -> Vec<[f64; 2]> {
        let mut v: Vec<[f64; 2]> = self.drones.iter().map(|(p, _)| *p).collect();
        v.extend(self.detections.iter().map(|(p, _)| *p));
        v.extend(self.route.iter().copied());
        v
    }
}

/// The widget.
pub struct MinimapWidget {
    data: Rc<RefCell<MapData>>,
    view: Extent,
    last_ms: Option<f64>,
    font: Option<(FontData, Vec<u8>)>,
}

impl MinimapWidget {
    pub fn new(data: Rc<RefCell<MapData>>) -> Self {
        let font = skylens_viewer::shell::BUNDLED_FONTS
            .iter()
            .find(|(name, _)| *name == "IBM Plex Mono")
            .map(|(_, bytes)| {
                let blob: Blob<u8> = Blob::new(Arc::new(*bytes) as Arc<dyn AsRef<[u8]> + Send + Sync>);
                (FontData::new(blob, 0), bytes.to_vec())
            });
        MinimapWidget { data, view: pad(EMPTY_EXTENT), last_ms: None, font }
    }

    pub fn boxed(data: Rc<RefCell<MapData>>) -> Box<dyn Widget> {
        Box::new(Self::new(data))
    }

    fn draw(&mut self, scene: &mut anyrender::Scene, w: f64, h: f64, scale: f64) {
        let data = self.data.borrow();
        let now = skylens_viewer::now_ms();
        let dt = self.last_ms.map_or(1.0 / 60.0, |l| ((now - l) / 1000.0).clamp(0.0, 0.25));
        self.last_ms = Some(now);
        self.view = ease(self.view, target_extent(&data.fit_points()), per_frame(0.12, dt));
        let view = self.view;
        let t = Affine::scale(scale);
        let proj = |p: [f64; 2]| project(&view, p[0], p[1], w, h);

        // Frame and scale bar.
        let frame = Rect::new(0.5, 0.5, w - 0.5, h - 0.5);
        scene.stroke(&Stroke::new(1.0), t, rgba(140, 200, 255, 0.45), None, &frame);
        self.draw_scale(scene, t, w, h);

        // Reconstructed ground under everything else.
        for c in &data.chunks {
            let (x, y) = proj(*c);
            let circle = Circle::new((x, y), 7.0);
            scene.fill(Fill::NonZero, t, rgba(120, 200, 255, 0.20), None, &circle);
            scene.stroke(&Stroke::new(1.0), t, rgba(120, 200, 255, 0.55), None, &circle);
        }

        // Route.
        if data.route.len() > 1 {
            let mut path = BezPath::new();
            for (i, p) in data.route.iter().enumerate() {
                let (x, y) = proj(*p);
                if i == 0 {
                    path.move_to((x, y));
                } else {
                    path.line_to((x, y));
                }
            }
            scene.stroke(&Stroke::new(1.5), t, rgba(255, 210, 127, 0.85), None, &path);
        }

        // Detection pins.
        for (p, person) in &data.detections {
            let (x, y) = proj(*p);
            let c = if *person { rgba(0x39, 0xd9, 0x8a, 1.0) } else { rgba(0xff, 0x4d, 0x4d, 1.0) };
            scene.fill(Fill::NonZero, t, c, None, &Circle::new((x, y), 3.0));
        }

        // Drones.
        for (p, active) in &data.drones {
            let (x, y) = proj(*p);
            let (c, r) = if *active { (rgba(0x9f, 0xe8, 0xff, 1.0), 4.0) } else { (rgba(0x4a, 0x90, 0xd9, 1.0), 3.0) };
            scene.fill(Fill::NonZero, t, c, None, &Circle::new((x, y), r));
            if *active {
                scene.stroke(&Stroke::new(1.0), t, rgba(159, 232, 255, 0.6), None, &Circle::new((x, y), 6.5));
            }
        }
    }

    fn draw_scale(&self, scene: &mut anyrender::Scene, t: Affine, w: f64, h: f64) {
        let (_, px, label) = scale_bar(&self.view, w);
        let (x0, y0) = (10.0, h - 12.0);
        let mut bar = BezPath::new();
        bar.move_to((x0, y0 - 3.0));
        bar.line_to((x0, y0));
        bar.line_to((x0 + px, y0));
        bar.line_to((x0 + px, y0 - 3.0));
        scene.stroke(&Stroke::new(1.0), t, rgba(200, 230, 255, 0.7), None, &bar);
        let Some((font, bytes)) = &self.font else { return };
        let glyphs = layout_text(bytes, &label, 9.0, x0 + px + 5.0, y0 + 1.0);
        if glyphs.is_empty() {
            return;
        }
        scene.draw_glyphs(
            font,
            9.0,
            false,
            &[],
            crate::peniko::kurbo::Vec2::ZERO,
            Fill::NonZero,
            rgba(200, 230, 255, 0.75),
            1.0,
            t,
            None,
            glyphs.into_iter(),
        );
    }
}

fn rgba(r: u8, g: u8, b: u8, a: f32) -> Color {
    Color::new([f32::from(r) / 255.0, f32::from(g) / 255.0, f32::from(b) / 255.0, a])
}

/// Left-to-right glyph run of `text` with its baseline at `(x, y)` (no shaping: the label is
/// digits, a space and Latin letters in a monospace font).
fn layout_text(font: &[u8], text: &str, size: f32, x: f64, y: f64) -> Vec<anyrender::Glyph> {
    use skrifa::MetadataProvider as _;
    use skrifa::instance::{LocationRef, Size};
    let Ok(f) = skrifa::FontRef::new(font) else { return Vec::new() };
    let charmap = f.charmap();
    let metrics = f.glyph_metrics(Size::new(size), LocationRef::default());
    let mut pen = x as f32;
    let mut out = Vec::new();
    for ch in text.chars() {
        let Some(gid) = charmap.map(ch) else { continue };
        out.push(anyrender::Glyph { id: gid.to_u32(), x: pen, y: y as f32 });
        pen += metrics.advance_width(gid).unwrap_or(size * 0.6);
    }
    out
}

impl Widget for MinimapWidget {
    fn requires_redraw(&self) -> bool {
        true
    }

    fn paint(
        &mut self,
        _render_ctx: &mut dyn anyrender::RenderContext,
        _styles: &ComputedStyles,
        width: u32,
        height: u32,
        scale: f64,
    ) -> anyrender::Scene {
        let mut scene = anyrender::Scene::new();
        let scale = if scale > 0.0 { scale } else { 1.0 };
        let (w, h) = (f64::from(width) / scale, f64::from(height) / scale);
        if w >= 1.0 && h >= 1.0 {
            self.draw(&mut scene, w, h, scale);
        }
        scene
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_extent_is_square_padded_and_at_least_120_m() {
        let e = target_extent(&[[0.0, 0.0]]);
        assert!((e.max_e - e.min_e - 120.0 * 1.3).abs() < 1e-9);
        assert!((e.max_n - e.min_n - 120.0 * 1.3).abs() < 1e-9);
        let e = target_extent(&[[0.0, 0.0], [400.0, 100.0]]);
        assert!((e.max_e - e.min_e - (e.max_n - e.min_n)).abs() < 1e-9, "square");
        assert!((e.min_e - (0.0 - 400.0 * 0.15)).abs() < 1e-9);
        assert_eq!(target_extent(&[]), pad(EMPTY_EXTENT));
    }

    #[test]
    fn north_is_up() {
        let v = Extent { min_e: 0.0, max_e: 100.0, min_n: 0.0, max_n: 100.0 };
        assert_eq!(project(&v, 0.0, 0.0, 200.0, 200.0), (0.0, 200.0));
        assert_eq!(project(&v, 100.0, 100.0, 200.0, 200.0), (200.0, 0.0));
    }

    #[test]
    fn easing_moves_part_of_the_way() {
        let a = Extent { min_e: 0.0, max_e: 10.0, min_n: 0.0, max_n: 10.0 };
        let b = Extent { min_e: 100.0, max_e: 110.0, min_n: 0.0, max_n: 10.0 };
        let c = ease(a, b, 0.12);
        assert!((c.min_e - 12.0).abs() < 1e-9);
    }

    #[test]
    fn scale_bar_is_a_round_distance() {
        // 156 m over 198 px: rough ~43.7 m -> 50 m.
        let v = Extent { min_e: 0.0, max_e: 156.0, min_n: 0.0, max_n: 156.0 };
        let (step, px, label) = scale_bar(&v, 198.0);
        assert_eq!(step, 50.0);
        assert!((px - 50.0 / (156.0 / 198.0)).abs() < 1e-9);
        assert_eq!(label, "50 m");
        let v = Extent { min_e: 0.0, max_e: 6000.0, min_n: 0.0, max_n: 6000.0 };
        assert_eq!(scale_bar(&v, 198.0).2, "2 km");
    }

    #[test]
    fn the_fit_ignores_the_reconstruction() {
        let d = MapData {
            drones: vec![([1.0, 2.0], true)],
            detections: vec![([3.0, 4.0], false)],
            route: vec![[5.0, 6.0]],
            chunks: vec![[1000.0, 1000.0]],
        };
        assert_eq!(d.fit_points(), vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    }
}
