//! Route planner map (legacy `control/routeModal.ts`): a top-down satellite map in the modal
//! canvas. Clicks drop GPS waypoints, a click on a marker removes it, a drag (beyond 4 px) pans.
//!
//! [`Planner`] is the state and the map geometry (a linear lon/lat frame around the map centre,
//! exactly the legacy one, so waypoint coordinates do not depend on whether imagery loaded).
//! [`PlannerWidget`] is the Blitz custom widget on `canvas.route-modal__canvas`: it draws the map
//! (tiles or the schematic grid, faint grid, scale bar, N compass, anchor cross, green polyline,
//! numbered markers) into a 460 x 460 buffer scaled to the element, and turns pointer input into
//! [`PlannerEvent`]s that the screen applies (it owns the terrain needed to turn a click into an
//! absolute altitude).

use crate::glyphs::{self, Align, Baseline, Face};
use blitz_dom::Widget;
use blitz_dom::node::ComputedStyles;
use blitz_traits::events::UiEvent;
use skylens_shared::proto::model::v1 as model;
use skylens_viewer::anyrender::{self, PaintScene as _, Scene};
use skylens_viewer::peniko::kurbo::{Affine, BezPath, Circle, Point, Rect, Stroke};
use skylens_viewer::peniko::{Blob, Color, Fill, ImageAlphaType, ImageBrush, ImageData, ImageFormat};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

/// Map drawing buffer (device px of the legacy canvas). CSS scales it to the element.
pub const MAP_PX: f64 = 460.0;
/// Selectable ground spans (metres, full map width).
pub const SPANS: [u32; 4] = [300, 600, 1200, 2500];
/// Click within this many map px of a marker removes it.
pub const HIT_RADIUS: f64 = 12.0;
/// Under this much pointer travel a press is a click, over it a pan (CSS px).
pub const DRAG_SLOP: f64 = 4.0;
pub const SAT_MAX_ZOOM: u32 = 17;
/// VWorld serves tilematrix 6..19 only.
pub const SAT_MIN_ZOOM: u32 = 6;
/// Metres per degree of latitude in the planner frame.
const M_PER_DEG: f64 = 111_320.0;

// --- Web Mercator tile math (legacy, same as terrainSource) ---
pub fn lon_to_tile_x(lon: f64, z: u32) -> f64 {
    (lon + 180.0) / 360.0 * 2f64.powi(z as i32)
}
pub fn lat_to_tile_y(lat: f64, z: u32) -> f64 {
    let r = lat.to_radians();
    (1.0 - (r.tan() + 1.0 / r.cos()).ln() / std::f64::consts::PI) / 2.0 * 2f64.powi(z as i32)
}
pub fn tile_x_to_lon(x: f64, z: u32) -> f64 {
    x / 2f64.powi(z as i32) * 360.0 - 180.0
}
pub fn tile_y_to_lat(y: f64, z: u32) -> f64 {
    let n = std::f64::consts::PI - 2.0 * std::f64::consts::PI * y / 2f64.powi(z as i32);
    (0.5 * (n.exp() - (-n).exp())).atan().to_degrees()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub west: f64,
    pub east: f64,
    pub south: f64,
    pub north: f64,
}

/// A planner point (absolute GPS altitude).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Waypoint {
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
}

impl Waypoint {
    pub fn gps(&self) -> model::Gps {
        model::Gps { lat: self.lat, lon: self.lon, alt: self.alt }
    }
}

/// One satellite tile in the map.
#[derive(Debug, Clone)]
pub struct MapTile {
    pub z: u32,
    pub tx: u32,
    pub ty: u32,
    pub image: ImageData,
}

/// What the widget asks the screen to do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlannerEvent {
    /// A click at map px (add a waypoint or remove the marker under it).
    Click { x: f64, y: f64 },
    /// A pan started (`is-panning` on) or ended (off; save the centre, reload tiles).
    PanStarted,
    PanEnded,
}

#[derive(Debug, Clone, Copy)]
struct Drag {
    x: f32,
    y: f32,
    moved: f64,
    panning: bool,
}

/// Planner state shared by the screen and the widget.
#[derive(Debug)]
pub struct Planner {
    pub center: Waypoint,
    /// The operator moved the map or a stored centre was restored: the core's suggestion must
    /// not move it.
    pub pinned: bool,
    pub span_m: f64,
    /// Height above ground (m).
    pub agl: f64,
    pub waypoints: Vec<Waypoint>,
    pub looped: bool,
    pub formation: bool,
    pub cloud: bool,
    pub spacing_m: f64,
    /// The ENU origin (map cross).
    pub anchor: Option<Waypoint>,
    pub tiles: Vec<MapTile>,
    /// Invalidates tile loads of a previous view.
    pub token: u64,
    pub events: Vec<PlannerEvent>,
    /// Element size in CSS px at the last paint.
    pub css_size: (f64, f64),
    drag: Option<Drag>,
    dirty: bool,
}

impl Planner {
    pub fn new(center: Waypoint, pinned: bool) -> Planner {
        Planner {
            center,
            pinned,
            span_m: f64::from(SPANS[1]),
            agl: 60.0,
            waypoints: Vec::new(),
            looped: true,
            formation: false,
            cloud: false,
            spacing_m: 10.0,
            anchor: None,
            tiles: Vec::new(),
            token: 0,
            events: Vec::new(),
            css_size: (MAP_PX, MAP_PX),
            drag: None,
            dirty: true,
        }
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn bounds(&self) -> Bounds {
        bounds(self.center.lat, self.center.lon, self.span_m)
    }

    pub fn gps_to_px(&self, lat: f64, lon: f64) -> (f64, f64) {
        let b = self.bounds();
        ((lon - b.west) / (b.east - b.west) * MAP_PX, (b.north - lat) / (b.north - b.south) * MAP_PX)
    }

    /// Map px to lat/lon.
    pub fn px_to_latlon(&self, x: f64, y: f64) -> (f64, f64) {
        let b = self.bounds();
        (b.north - y / MAP_PX * (b.north - b.south), b.west + x / MAP_PX * (b.east - b.west))
    }

    /// Index of the marker within the hit radius of a map px, first match.
    pub fn hit(&self, x: f64, y: f64) -> Option<usize> {
        self.waypoints.iter().position(|w| {
            let (wx, wy) = self.gps_to_px(w.lat, w.lon);
            (wx - x).hypot(wy - y) <= HIT_RADIUS
        })
    }

    /// Tiles for the current view: zoom and the inclusive tile range.
    pub fn tile_plan(&self) -> (u32, Vec<(u32, u32)>) {
        tile_plan(&self.bounds())
    }

    /// Starts a new view: drops tiles and invalidates loads in flight.
    pub fn new_view(&mut self) -> u64 {
        self.token += 1;
        self.tiles.clear();
        self.dirty = true;
        self.token
    }

    /// Pointer input on the canvas, in CSS px relative to it.
    pub fn pointer(&mut self, ev: &UiEvent) {
        match ev {
            UiEvent::PointerDown(e) => {
                self.drag = Some(Drag { x: e.coords.client_x, y: e.coords.client_y, moved: 0.0, panning: false });
            }
            UiEvent::PointerMove(e) => {
                let Some(mut d) = self.drag else { return };
                if e.buttons.is_empty() {
                    // Released outside the canvas: the gesture is over.
                    self.end_drag(None);
                    return;
                }
                let (x, y) = (e.coords.client_x, e.coords.client_y);
                let (dx, dy) = (f64::from(x - d.x), f64::from(y - d.y));
                d.moved += dx.hypot(dy);
                if d.moved <= DRAG_SLOP {
                    self.drag = Some(d);
                    return;
                }
                d.x = x;
                d.y = y;
                let b = self.bounds();
                let (w, h) = (self.css_size.0.max(1.0), self.css_size.1.max(1.0));
                let lon_per_px = (b.east - b.west) / w;
                let lat_per_px = (b.north - b.south) / h;
                self.center.lat = (self.center.lat + dy * lat_per_px).clamp(-85.0, 85.0);
                self.center.lon -= dx * lon_per_px;
                self.pinned = true;
                if !d.panning {
                    d.panning = true;
                    self.events.push(PlannerEvent::PanStarted);
                }
                self.drag = Some(d);
                self.dirty = true;
            }
            UiEvent::PointerUp(e) => self.end_drag(Some((e.coords.client_x, e.coords.client_y))),
            UiEvent::PointerCancel(_) => self.end_drag(None),
            _ => {}
        }
    }

    fn end_drag(&mut self, release: Option<(f32, f32)>) {
        let Some(d) = self.drag.take() else { return };
        if d.moved > DRAG_SLOP {
            self.events.push(PlannerEvent::PanEnded);
            return;
        }
        if let Some((x, y)) = release {
            let (w, h) = (self.css_size.0.max(1.0), self.css_size.1.max(1.0));
            self.events.push(PlannerEvent::Click { x: f64::from(x) / w * MAP_PX, y: f64::from(y) / h * MAP_PX });
        }
    }

    // ---------------------------------------------------------------------------------------
    // Drawing (legacy draw(): grid, satellite, faint grid, scale and compass, route)
    // ---------------------------------------------------------------------------------------

    pub fn draw(&self, scene: &mut Scene, base: Affine) {
        let clip = Rect::new(0.0, 0.0, MAP_PX, MAP_PX);
        scene.push_clip_layer(Fill::NonZero, base, &clip);

        // Schematic grid (always under everything).
        scene.fill(Fill::NonZero, base, Color::from_rgb8(0x0d, 0x15, 0x22), None, &clip);
        grid_lines(scene, base, Color::from_rgba8(120, 170, 220, alpha8(0.14)));

        if !self.tiles.is_empty() {
            for t in &self.tiles {
                let nw_lon = tile_x_to_lon(f64::from(t.tx), t.z);
                let nw_lat = tile_y_to_lat(f64::from(t.ty), t.z);
                let se_lon = tile_x_to_lon(f64::from(t.tx + 1), t.z);
                let se_lat = tile_y_to_lat(f64::from(t.ty + 1), t.z);
                let (x0, y0) = self.gps_to_px(nw_lat, nw_lon);
                let (x1, y1) = self.gps_to_px(se_lat, se_lon);
                let sx = (x1 - x0) / f64::from(t.image.width.max(1));
                let sy = (y1 - y0) / f64::from(t.image.height.max(1));
                let brush = ImageBrush { image: &t.image, sampler: Default::default() };
                scene.draw_image(brush, base * Affine::translate((x0, y0)) * Affine::scale_non_uniform(sx, sy));
            }
            grid_lines(scene, base, Color::from_rgba8(255, 255, 255, alpha8(0.08)));
        }

        self.draw_scale_and_compass(scene, base);
        self.draw_route(scene, base);
        scene.pop_layer();
    }

    fn draw_scale_and_compass(&self, scene: &mut Scene, base: Affine) {
        let white = Color::from_rgba8(255, 255, 255, alpha8(0.85));
        let bar = MAP_PX / 4.0;
        let (x, y) = (14.0, MAP_PX - 18.0);
        let mut p = BezPath::new();
        p.move_to((x, y));
        p.line_to((x + bar, y));
        p.move_to((x, y - 4.0));
        p.line_to((x, y + 4.0));
        p.move_to((x + bar, y - 4.0));
        p.line_to((x + bar, y + 4.0));
        scene.stroke(&Stroke::new(2.0), base, white, None, &p);
        glyphs::draw_text(scene, base, Face::Regular, 11.0, &crate::labels::scale_label(self.span_m), x, y - 7.0, Align::Start, Baseline::Alphabetic, white);
        glyphs::draw_text(scene, base, Face::Regular, 11.0, "N", MAP_PX - 18.0, 18.0, Align::Start, Baseline::Alphabetic, white);
        let mut n = BezPath::new();
        n.move_to((MAP_PX - 14.0, 22.0));
        n.line_to((MAP_PX - 14.0, 34.0));
        scene.stroke(&Stroke::new(2.0), base, white, None, &n);
    }

    fn draw_route(&self, scene: &mut Scene, base: Affine) {
        if let Some(a) = self.anchor {
            let (ax, ay) = self.gps_to_px(a.lat, a.lon);
            let mut p = BezPath::new();
            p.move_to((ax - 6.0, ay));
            p.line_to((ax + 6.0, ay));
            p.move_to((ax, ay - 6.0));
            p.line_to((ax, ay + 6.0));
            scene.stroke(&Stroke::new(1.0), base, Color::from_rgba8(120, 200, 255, alpha8(0.6)), None, &p);
        }
        if self.waypoints.len() > 1 {
            let mut p = BezPath::new();
            for (i, w) in self.waypoints.iter().enumerate() {
                let pt = self.gps_to_px(w.lat, w.lon);
                if i == 0 {
                    p.move_to(pt);
                } else {
                    p.line_to(pt);
                }
            }
            scene.stroke(&Stroke::new(2.0), base, Color::from_rgba8(74, 222, 128, alpha8(0.9)), None, &p);
        }
        for (i, w) in self.waypoints.iter().enumerate() {
            let (x, y) = self.gps_to_px(w.lat, w.lon);
            let fill = if i == 0 { Color::from_rgb8(0x4a, 0xde, 0x80) } else { Color::from_rgb8(0xff, 0xcc, 0x55) };
            let c = Circle::new(Point::new(x, y), 8.0);
            scene.fill(Fill::NonZero, base, fill, None, &c);
            scene.stroke(&Stroke::new(2.0), base, Color::from_rgba8(0, 0, 0, alpha8(0.7)), None, &c);
            glyphs::draw_text(scene, base, Face::Bold, 11.0, &(i + 1).to_string(), x, y, Align::Center, Baseline::Middle, Color::from_rgb8(0x0a, 0x0f, 0x18));
        }
    }
}

fn alpha8(a: f64) -> u8 {
    (a * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Seven inner lines each way (legacy 8 x 8 grid).
fn grid_lines(scene: &mut Scene, base: Affine, color: Color) {
    let step = MAP_PX / 8.0;
    let mut p = BezPath::new();
    for i in 1..8 {
        let v = f64::from(i) * step;
        p.move_to((v, 0.0));
        p.line_to((v, MAP_PX));
        p.move_to((0.0, v));
        p.line_to((MAP_PX, v));
    }
    scene.stroke(&Stroke::new(1.0), base, color, None, &p);
}

/// The map's ground box around a centre for a span (legacy `bounds()`).
pub fn bounds(lat: f64, lon: f64, span_m: f64) -> Bounds {
    let cos = lat.to_radians().cos();
    let cos = if cos == 0.0 { 1.0 } else { cos };
    let d_lat = span_m / M_PER_DEG;
    let d_lon = span_m / (M_PER_DEG * cos);
    Bounds { west: lon - d_lon / 2.0, east: lon + d_lon / 2.0, south: lat - d_lat / 2.0, north: lat + d_lat / 2.0 }
}

/// Deepest zoom whose tiles over the box are at most 12 (legacy `pickZoom`), and the tiles.
pub fn tile_plan(b: &Bounds) -> (u32, Vec<(u32, u32)>) {
    let mut z = SAT_MIN_ZOOM;
    for cand in (SAT_MIN_ZOOM + 1..=SAT_MAX_ZOOM).rev() {
        let cols = lon_to_tile_x(b.east, cand).floor() - lon_to_tile_x(b.west, cand).floor() + 1.0;
        let rows = lat_to_tile_y(b.south, cand).floor() - lat_to_tile_y(b.north, cand).floor() + 1.0;
        if cols * rows <= 12.0 {
            z = cand;
            break;
        }
    }
    let tx0 = lon_to_tile_x(b.west, z).floor() as u32;
    let tx1 = lon_to_tile_x(b.east, z).floor() as u32;
    let ty0 = lat_to_tile_y(b.north, z).floor() as u32;
    let ty1 = lat_to_tile_y(b.south, z).floor() as u32;
    let mut tiles = Vec::new();
    for tx in tx0..=tx1 {
        for ty in ty0..=ty1 {
            tiles.push((tx, ty));
        }
    }
    (z, tiles)
}

/// The core's tile proxy path for one tile.
pub fn tile_path(z: u32, tx: u32, ty: u32) -> String {
    format!("/vworld/wmts/{z}/{ty}/{tx}.jpeg")
}

/// Decodes a tile JPEG for the map (None for anything that is not an image).
pub fn decode_tile(content_type: &str, bytes: &[u8]) -> Option<ImageData> {
    if !content_type.is_empty() && !content_type.contains("image") {
        return None;
    }
    let img = image::load_from_memory(bytes).ok()?.into_rgba8();
    let (width, height) = img.dimensions();
    Some(ImageData {
        data: Blob::new(Arc::new(img.into_raw()) as Arc<dyn AsRef<[u8]> + Send + Sync>),
        format: ImageFormat::Rgba8,
        alpha_type: ImageAlphaType::Alpha,
        width,
        height,
    })
}

/// A stored map centre (legacy `restoreCenter`), None when absent or implausible.
pub fn parse_center(raw: Option<&str>) -> Option<Waypoint> {
    let v: serde_json::Value = serde_json::from_str(raw?).ok()?;
    let lat = v.get("lat")?.as_f64()?;
    let lon = v.get("lon")?.as_f64()?;
    if lat.abs() > 85.0 || lon.abs() > 180.0 {
        return None;
    }
    let alt = v.get("alt").and_then(|a| a.as_f64()).unwrap_or(0.0);
    Some(Waypoint { lat, lon, alt })
}

pub fn stored_center(c: &Waypoint) -> String {
    serde_json::json!({ "lat": c.lat, "lon": c.lon, "alt": c.alt }).to_string()
}

// ---------------------------------------------------------------------------------------------
// Widget
// ---------------------------------------------------------------------------------------------

/// The custom widget on `canvas.route-modal__canvas`.
pub struct PlannerWidget {
    state: Rc<RefCell<Planner>>,
}

impl PlannerWidget {
    pub fn boxed(state: Rc<RefCell<Planner>>) -> Box<dyn Widget> {
        Box::new(PlannerWidget { state })
    }
}

impl Widget for PlannerWidget {
    fn requires_redraw(&self) -> bool {
        self.state.borrow().dirty
    }

    fn handle_event(&mut self, event: &UiEvent) {
        self.state.borrow_mut().pointer(event);
    }

    fn paint(&mut self, _ctx: &mut dyn anyrender::RenderContext, _styles: &ComputedStyles, width: u32, height: u32, scale: f64) -> Scene {
        let mut scene = Scene::new();
        let mut s = self.state.borrow_mut();
        s.dirty = false;
        if width == 0 || height == 0 {
            return scene;
        }
        let scale = if scale > 0.0 { scale } else { 1.0 };
        s.css_size = (f64::from(width) / scale, f64::from(height) / scale);
        let base = Affine::scale_non_uniform(f64::from(width) / MAP_PX, f64::from(height) / MAP_PX);
        s.draw(&mut scene, base);
        scene
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planner() -> Planner {
        Planner::new(Waypoint { lat: 36.3685, lon: 127.3475, alt: 30.0 }, false)
    }

    #[test]
    fn px_and_gps_are_inverse() {
        let p = planner();
        let (x, y) = p.gps_to_px(p.center.lat, p.center.lon);
        assert!((x - 230.0).abs() < 1e-9 && (y - 230.0).abs() < 1e-9);
        let (lat, lon) = p.px_to_latlon(100.0, 300.0);
        let (x, y) = p.gps_to_px(lat, lon);
        assert!((x - 100.0).abs() < 1e-9 && (y - 300.0).abs() < 1e-9);
        let b = p.bounds();
        assert!(((b.north - b.south) * M_PER_DEG - 600.0).abs() < 1e-6);
    }

    #[test]
    fn marker_hits_within_twelve_px() {
        let mut p = planner();
        let (lat, lon) = p.px_to_latlon(200.0, 200.0);
        p.waypoints.push(Waypoint { lat, lon, alt: 0.0 });
        assert_eq!(p.hit(210.0, 205.0), Some(0));
        assert_eq!(p.hit(215.0, 210.0), None);
    }

    #[test]
    fn zoom_keeps_at_most_twelve_tiles() {
        for span in SPANS {
            let b = bounds(36.3685, 127.3475, f64::from(span));
            let (z, tiles) = tile_plan(&b);
            assert!(tiles.len() <= 12 || z == SAT_MIN_ZOOM, "span {span}: {} tiles at z{z}", tiles.len());
            assert!((SAT_MIN_ZOOM..=SAT_MAX_ZOOM).contains(&z));
        }
        // Tile corners round-trip.
        let x = lon_to_tile_x(127.3475, 15);
        assert!((tile_x_to_lon(x, 15) - 127.3475).abs() < 1e-9);
        let y = lat_to_tile_y(36.3685, 15);
        assert!((tile_y_to_lat(y, 15) - 36.3685).abs() < 1e-9);
        assert_eq!(tile_path(17, 111, 222), "/vworld/wmts/17/222/111.jpeg");
    }

    #[test]
    fn stored_centre_round_trips_and_rejects_nonsense() {
        let c = Waypoint { lat: 36.1, lon: 127.2, alt: 3.0 };
        assert_eq!(parse_center(Some(&stored_center(&c))), Some(c));
        assert_eq!(parse_center(Some("{\"lat\":91,\"lon\":0}")), None);
        assert_eq!(parse_center(Some("nope")), None);
        assert_eq!(parse_center(None), None);
    }
}
