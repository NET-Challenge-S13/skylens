//! The minimap (the point panel, legacy `pointInset.ts`), mounted on `.point-inset__view`.
//!
//! A separate scene and camera: every delivered point segment (and the point slot stream)
//! coloured by height over the range the points actually span, on `#06090e`, with the visible
//! detections as pins above the points and the drones that are moving as diamonds.
//!
//! The camera is a turntable: it orbits the centre of the delivered points' box, far enough that
//! the whole box fits, at 35 degrees above the horizon, one turn every 40 s, so positions read in
//! 3D. It refits (easing) only when the box grows by more than 10 % or the panel changes shape,
//! and never follows drones or detections. A drag turns and tilts it (the turn pauses while the
//! pointer is down), the wheel zooms, a double click returns to the first view.

use crate::board::{Board, LiveView};
use crate::glam::DVec3;
use crate::live_layer::{LiveLayer, Paint};
use crate::markers;
use crate::pointblitz::core::{FrameStyle, PointSize};
use crate::points;
use skylens_shared::proto::payload::v1 as payload;
use skylens_viewer::camera::{Pose, damp_factor};
use skylens_viewer::config::{self, color};
use skylens_viewer::view3d::{SEGMENT_CLOUDS, cloud_segment, retag, segment_cloud};
use skylens_viewer::{FrameInfo, FrameSpec, Gpu3d, ViewInput, ViewScreen};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::f64::consts::TAU;
use std::rc::Rc;

pub const FOV_DEG: f64 = 45.0;
const BACKGROUND: u32 = 0x06090e;
const BACKGROUND_RGB: [u8; 3] = [0x06, 0x09, 0x0e];
/// One turntable turn takes this long (s).
pub const TURN_SECONDS: f64 = 40.0;
/// The turntable's height above the horizon (degrees) and the range a drag may tilt it to.
pub const ELEVATION_DEG: f64 = 35.0;
const ELEVATION_MIN_DEG: f64 = 10.0;
const ELEVATION_MAX_DEG: f64 = 80.0;
/// Range of the operator's distance factor (wheel).
const ZOOM_MIN: f64 = 0.4;
const ZOOM_MAX: f64 = 2.5;
/// Distance factor per wheel step (like the orbit's `0.95`).
const ZOOM_STEP: f64 = 0.95;
/// Point size (CSS px), independent of the main view's setting.
const POINT_PX: f32 = 1.5;
/// Ids of the minimap's own meshes: detection pins (pin, ring = base + 1) and drones
/// (`DRONE_IDS + drone id`), below the shared sync namespaces.
const PIN_IDS: u64 = 1 << 21;
const DRONE_IDS: u64 = 1 << 32;

/// The turntable camera: looking at `target` from `distance` away, `yaw` radians counter-clockwise
/// from east and `elevation` radians above the horizon (z up).
pub fn turntable_pose(target: DVec3, distance: f64, yaw: f64, elevation: f64) -> Pose {
    let (se, ce) = elevation.sin_cos();
    let (sy, cy) = yaw.sin_cos();
    Pose::new(target + DVec3::new(ce * cy, ce * sy, se) * distance, target)
}

/// Distance at which a turntable at `elevation` (radians) sees the whole box from any yaw, with a
/// 5 % margin. The box is taken as the upright cylinder around it: across, its bounding sphere
/// must fit the horizontal field; up, each rim point (near or far side, top or bottom) must fit
/// the vertical field at its own depth.
pub fn turntable_distance(bmin: DVec3, bmax: DVec3, fov_deg: f64, aspect: f64, elevation: f64) -> f64 {
    let size = (bmax - bmin).max(DVec3::ZERO);
    let r_h = (size.x.hypot(size.y) / 2.0).max(1.0);
    let half_z = size.z / 2.0;
    let r = r_h.hypot(half_z);
    let v_tan = (fov_deg.to_radians() / 2.0).tan();
    let h_half = (v_tan * aspect.max(0.1)).atan();
    let (se, ce) = elevation.sin_cos();
    let across = r / h_half.sin();
    let mut up: f64 = 0.0;
    for h in [-r_h, r_h] {
        for z in [-half_z, half_z] {
            // Offset toward the camera (h along the ground toward it, z up) and up on screen.
            let toward = h * ce + z * se;
            let screen_up = z * ce - h * se;
            up = up.max(screen_up.abs() / v_tan + toward);
        }
    }
    across.max(up) * 1.05
}

/// What a segment's cloud was built from.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Uploaded {
    epoch: u64,
    chunks: usize,
    range: (f64, f64),
}

/// A detection pin as last uploaded.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Pin {
    base: u64,
    feet: DVec3,
    person: bool,
    scale: f64,
}

/// The operator's adjustments on top of the turntable.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Adjust {
    /// Added to the automatic yaw (radians).
    yaw: f64,
    /// Height above the horizon (radians).
    elevation: f64,
    /// Distance factor.
    zoom: f64,
}

impl Adjust {
    const HOME: Adjust = Adjust { yaw: 0.0, elevation: ELEVATION_DEG.to_radians(), zoom: 1.0 };
}

pub struct InsetScene {
    board: Rc<RefCell<Board>>,
    /// Jump (not ease) to the next framing: nothing was framed yet.
    snap: bool,
    fit_key: Option<(DVec3, DVec3, i64)>,
    /// Where the framing is heading (target, distance) and where it is now.
    goal: Option<(DVec3, f64)>,
    target: DVec3,
    distance: f64,
    /// The automatic turn (radians).
    yaw: f64,
    /// The operator's adjustments: asked for, and as shown (eased toward the asked ones).
    want: Adjust,
    shown: Adjust,
    /// Last pointer position while the pointer is down.
    drag: Option<(f32, f32)>,
    near: f64,
    uploaded: BTreeMap<u32, Uploaded>,
    css_height: f64,
    point_size: Option<f32>,
    layer: LiveLayer,
    pins: BTreeMap<String, Pin>,
    next_pin: u64,
    /// Drone id -> (position, active, radius) as last uploaded.
    drones: BTreeMap<u32, (DVec3, bool, f64)>,
}

impl InsetScene {
    pub fn new(board: Rc<RefCell<Board>>) -> InsetScene {
        InsetScene {
            board,
            snap: true,
            fit_key: None,
            goal: None,
            target: DVec3::ZERO,
            distance: 50.0,
            yaw: -TAU / 8.0,
            want: Adjust::HOME,
            shown: Adjust::HOME,
            drag: None,
            near: 0.5,
            uploaded: BTreeMap::new(),
            css_height: 1.0,
            point_size: None,
            layer: LiveLayer::new(LiveView::Inset),
            pins: BTreeMap::new(),
            next_pin: 0,
            drones: BTreeMap::new(),
        }
    }

    /// Double click: back to the turntable's own tilt and distance (the turn goes on).
    pub fn reframe(&mut self) {
        self.want = Adjust::HOME;
    }

    fn sync_clouds(&mut self, gpu: &mut Gpu3d, board: &mut Board, now_s: f64) {
        let Some(range) = board.height_range() else { return };
        // The slot stream: the same live layer as the main view, always coloured by height.
        self.layer.pump(gpu, board, Paint::Height(range.0, range.1), BACKGROUND_RGB, now_s);
        for (seg, pts) in board.store.segments() {
            let id = segment_cloud(seg);
            let payloads = pts.payloads();
            let want = Uploaded { epoch: pts.epoch(), chunks: payloads.len(), range };
            match self.uploaded.get(&seg) {
                Some(have) if *have == want => {}
                Some(have) if have.epoch == want.epoch && have.range == range && have.chunks < want.chunks => {
                    for p in &payloads[have.chunks..] {
                        let colored = points::recolor_by_height(p, range.0, range.1);
                        if let Err(e) = gpu.clouds.insert_chunk(&gpu.device, id, &retag(&colored, 0, 0)) {
                            skylens_viewer::log(&format!("[inset] dropped point chunk of segment {seg}: {e}"));
                        }
                    }
                    self.uploaded.insert(seg, want);
                }
                _ => {
                    let colored: Vec<Vec<u8>> = payloads.iter().map(|p| points::recolor_by_height(p, range.0, range.1)).collect();
                    let refs: Vec<&[u8]> = colored.iter().map(Vec::as_slice).collect();
                    gpu.clouds.replace(&gpu.device, id, &refs);
                    self.uploaded.insert(seg, want);
                }
            }
        }
        let stale: Vec<u64> = gpu.clouds.ids().filter(|id| cloud_segment(*id).is_some_and(|s| board.store.segment(s).is_none())).collect();
        for id in stale {
            gpu.clouds.remove(id);
            if let Some(s) = cloud_segment(id) {
                self.uploaded.remove(&s);
            }
        }
    }

    /// Refits the framing goal when the box really grew or the panel changed shape.
    fn refit(&mut self, board: &Board, aspect: f64) {
        let Some((bmin, bmax)) = board.points_box() else { return };
        // Only a box that really grew (or a reshaped panel) refits: a point count that changes
        // with every event must not pump the zoom.
        let key = (bmin, bmax, (aspect * 1000.0).round() as i64);
        let grew = self.fit_key.is_none_or(|(lo, hi, a)| {
            let span = (hi - lo).max_element().max(1.0);
            a != key.2 || (bmin.cmplt(lo - DVec3::splat(span * 0.1)).any()) || (bmax.cmpgt(hi + DVec3::splat(span * 0.1)).any())
        });
        if !grew {
            return;
        }
        self.fit_key = Some(key);
        let distance = turntable_distance(bmin, bmax, FOV_DEG, aspect, ELEVATION_DEG.to_radians());
        let r = ((bmax - bmin).length() / 2.0).max(1.0);
        self.near = (r / 2000.0).max(0.1);
        self.goal = Some(((bmin + bmax) / 2.0, distance));
    }

    /// Eases the framing toward its goal and the shown adjustments toward the asked ones, and
    /// advances the turn.
    fn step(&mut self, dt: f64) {
        if let Some((target, distance)) = self.goal {
            if self.snap {
                self.snap = false;
                self.target = target;
                self.distance = distance;
            } else {
                let k = damp_factor(2.5, dt);
                self.target = self.target.lerp(target, k);
                self.distance += (distance - self.distance) * k;
            }
        }
        if self.drag.is_none() {
            self.yaw = (self.yaw + TAU / TURN_SECONDS * dt).rem_euclid(TAU);
        }
        let k = damp_factor(10.0, dt);
        self.shown.yaw += (self.want.yaw - self.shown.yaw) * k;
        self.shown.elevation += (self.want.elevation - self.shown.elevation) * k;
        self.shown.zoom += (self.want.zoom - self.shown.zoom) * k;
    }

    fn pose(&self) -> Pose {
        turntable_pose(self.target, self.distance * self.shown.zoom, self.yaw + self.shown.yaw, self.shown.elevation)
    }

    /// Pin size factor for a box `extent` metres across: big enough to find on the minimap.
    fn pin_scale(extent: f64) -> f64 {
        if extent.is_finite() { (extent * 0.05).clamp(1.0, 12.0) } else { 1.0 }
    }

    /// The visible detections as pins (same-kind sightings within 3 m merged) with a ground ring.
    fn sync_pins(&mut self, gpu: &mut Gpu3d, board: &Board) {
        let sightings: Vec<markers::Sighting> = board
            .store
            .visible_detections()
            .filter_map(|d| {
                let person = d.category() == payload::DetectionCategory::Person;
                Some((d.id.clone(), board.detection_pos(d)?, person, d.confidence))
            })
            .collect();
        let shown = markers::dedupe(&sightings);
        let scale = Self::pin_scale(board.points_box().map_or(f64::NAN, |(lo, hi)| (hi - lo).max_element()));
        let gone: Vec<String> = self.pins.keys().filter(|id| !shown.iter().any(|(s, _, _)| s == *id)).cloned().collect();
        for id in gone {
            if let Some(p) = self.pins.remove(&id) {
                gpu.meshes.remove(p.base);
                gpu.meshes.remove(p.base + 1);
            }
        }
        for (id, feet, person) in shown {
            let base = match self.pins.get(&id) {
                Some(p) => {
                    // Keep the size unless the box changed it by more than 10 %.
                    let same_scale = (p.scale - scale).abs() <= p.scale * 0.1;
                    if p.feet == feet && p.person == person && same_scale {
                        continue;
                    }
                    p.base
                }
                None => {
                    let b = PIN_IDS + 2 * self.next_pin;
                    self.next_pin += 1;
                    debug_assert!(b + 1 < DRONE_IDS);
                    b
                }
            };
            gpu.meshes.upsert(&gpu.device, &gpu.queue, base, &markers::pin_mesh(feet, scale), None, markers::flat_material(markers::marker_color(person)));
            gpu.meshes.upsert(&gpu.device, &gpu.queue, base + 1, &markers::ring_mesh(feet, scale * 1.5), None, markers::ring_material(person));
            self.pins.insert(id, Pin { base, feet, person, scale });
        }
    }

    /// The drones that are moving, as diamonds; a drone that stops is taken off.
    fn sync_drones(&mut self, gpu: &mut Gpu3d, board: &Board) {
        let radius = Self::pin_scale(board.points_box().map_or(f64::NAN, |(lo, hi)| (hi - lo).max_element()));
        let moving: BTreeMap<u32, (DVec3, bool)> = board.moving_drones().map(|d| (d.id, (d.pos, Some(d.id) == board.active_drone))).collect();
        let gone: Vec<u32> = self.drones.keys().filter(|id| !moving.contains_key(id)).copied().collect();
        for id in gone {
            self.drones.remove(&id);
            gpu.meshes.remove(DRONE_IDS + u64::from(id));
        }
        for (id, (pos, active)) in moving {
            let want = (pos, active, radius);
            if self.drones.get(&id) == Some(&want) {
                continue;
            }
            let mesh_id = DRONE_IDS + u64::from(id);
            debug_assert!(mesh_id < SEGMENT_CLOUDS);
            gpu.meshes.upsert(&gpu.device, &gpu.queue, mesh_id, &markers::drone_mesh(pos, radius), None, markers::flat_material(if active { color::DRONE_CORE } else { color::DRONE_TRAIL }));
            self.drones.insert(id, want);
        }
    }
}

impl ViewScreen for InsetScene {
    fn gpu_lost(&mut self) {
        self.uploaded.clear();
        self.pins.clear();
        self.drones.clear();
        let board = self.board.clone();
        if let Ok(mut b) = board.try_borrow_mut() {
            self.layer.lost(&mut b);
        }
    }

    fn frame(&mut self, gpu: &mut Gpu3d, info: &FrameInfo) -> FrameSpec {
        self.css_height = info.css_height();
        let board_rc = self.board.clone();
        let mut board = board_rc.borrow_mut();
        self.sync_clouds(gpu, &mut board, info.now_ms / 1000.0);
        self.sync_pins(gpu, &board);
        self.sync_drones(gpu, &board);

        let aspect = info.css_width() / info.css_height().max(1.0);
        self.refit(&board, aspect);
        self.step(info.dt);

        let size = POINT_PX * info.scale as f32;
        let point_size = (self.point_size != Some(size)).then(|| {
            self.point_size = Some(size);
            PointSize::Pixels(size)
        });
        FrameSpec {
            camera: self.pose().camera(FOV_DEG, self.near),
            style: FrameStyle { clear: config::clear(BACKGROUND), fog: None },
            point_size,
        }
    }

    fn input(&mut self, input: &ViewInput) {
        match *input {
            ViewInput::PointerDown { x, y, .. } => self.drag = Some((x, y)),
            ViewInput::PointerMove { x, y, pressed } => {
                if let (true, Some((lx, ly))) = (pressed, self.drag) {
                    let h = self.css_height.max(1.0);
                    let (dx, dy) = (f64::from(x - lx), f64::from(y - ly));
                    // Dragging right turns the scene right; dragging down looks from higher up.
                    self.want.yaw -= TAU * dx / h * 0.5;
                    self.want.elevation = (self.want.elevation + std::f64::consts::PI * dy / h * 0.5)
                        .clamp(ELEVATION_MIN_DEG.to_radians(), ELEVATION_MAX_DEG.to_radians());
                    self.drag = Some((x, y));
                }
            }
            ViewInput::PointerUp { .. } => self.drag = None,
            ViewInput::Wheel { delta_y, .. } => {
                // `delta_y > 0` zooms out.
                if delta_y > 0.0 {
                    self.want.zoom /= ZOOM_STEP;
                } else if delta_y < 0.0 {
                    self.want.zoom *= ZOOM_STEP;
                }
                self.want.zoom = self.want.zoom.clamp(ZOOM_MIN, ZOOM_MAX);
            }
            ViewInput::Click { .. } => {}
        }
    }

    fn needs_redraw(&self) -> bool {
        // Drawn only while the panel is open and holds points (it is hidden otherwise); the
        // turntable keeps it moving.
        self.board.try_borrow().map_or(true, |b| b.point_layer_on && b.total_points() > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_turntable_orbits_at_its_elevation_and_distance() {
        let t = DVec3::new(100.0, 50.0, 2.0);
        let el = ELEVATION_DEG.to_radians();
        let p = turntable_pose(t, 80.0, 0.0, el);
        assert_eq!(p.target, t);
        assert!((p.distance() - 80.0).abs() < 1e-9);
        let d = p.eye - t;
        assert!(d.x > 0.0 && d.y.abs() < 1e-9, "yaw 0 looks from the east");
        assert!((d.z / 80.0 - el.sin()).abs() < 1e-12);
        // A quarter turn later it looks from the north, at the same height.
        let q = turntable_pose(t, 80.0, TAU / 4.0, el);
        assert!((q.eye - t).y > 79.0 * el.cos() && (q.eye.z - p.eye.z).abs() < 1e-9);
    }

    #[test]
    fn the_turntable_sees_the_whole_box_from_every_side() {
        let bmin = DVec3::new(70.0, 20.0, 0.0);
        let bmax = DVec3::new(160.0, 80.0, 12.0);
        let target = (bmin + bmax) / 2.0;
        for aspect in [0.6, 1.0, 1.6] {
            let dist = turntable_distance(bmin, bmax, FOV_DEG, aspect, ELEVATION_DEG.to_radians());
            let (w, h) = (200.0 * aspect, 200.0);
            for k in 0..16 {
                let yaw = TAU * f64::from(k) / 16.0;
                let pose = turntable_pose(target, dist, yaw, ELEVATION_DEG.to_radians());
                let p = skylens_viewer::Projector::new(pose.camera(FOV_DEG, 0.1), w, h);
                for i in 0..8 {
                    let c = DVec3::new(
                        if i & 1 == 0 { bmin.x } else { bmax.x },
                        if i & 2 == 0 { bmin.y } else { bmax.y },
                        if i & 4 == 0 { bmin.z } else { bmax.z },
                    );
                    assert!(p.project_visible(c.to_array()).is_some(), "aspect {aspect} yaw {k}/16 corner {i}");
                }
            }
            // Not absurdly far: the box's horizontal radius is about 54 m.
            assert!(dist < 400.0, "{dist}");
        }
    }

    #[test]
    fn pins_grow_with_the_box_within_bounds() {
        assert_eq!(InsetScene::pin_scale(10.0), 1.0);
        assert!((InsetScene::pin_scale(100.0) - 5.0).abs() < 1e-12);
        assert_eq!(InsetScene::pin_scale(1.0e5), 12.0);
        assert_eq!(InsetScene::pin_scale(f64::NAN), 1.0);
    }
}
