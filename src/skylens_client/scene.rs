//! The main 3D view (legacy `StatusViewer`), mounted on the legacy `canvas#status-view`.
//!
//! Draws only what has arrived: one pointblitz scene per reconstruction segment with its arrival
//! fade (or, for the point slot stream of stable display, one slot scene drawn opaque: its points
//! appear as each event lands, without a per-segment fade), the deltamesh blocks under the legacy
//! scene's warm ambient and key light, and the detection markers once their segment's final
//! level is in. Background `#0d0f14` with distance fog in the same colour, re-derived from the
//! follow distance as the reconstruction grows.
//!
//! Camera: the operator orbits with damping; while they drag and for 3.5 s after, the automatic
//! camera waits. Otherwise the camera eases (`1 - exp(-2.5 dt)`) toward the focused detection
//! (FOCUSING / LOCKED) or the follow pose of the active drone.

use crate::board::{Board, FIT_SAMPLES_PER_SEGMENT, LiveView};
use crate::inset;
use crate::live_layer::{LiveLayer, Paint};
use crate::follow::{self, FollowInput, Follower};
use crate::glam::DVec3;
use crate::markers;
use crate::pointblitz::core::{Blend, Fog, FrameStyle, Material, PointSize, Shading};
use crate::reveal::RevealFade;
use skylens_viewer::camera::{UserOrbit, damp_factor};
use skylens_viewer::config::{self, color};
use skylens_viewer::view3d::{SEGMENT_CLOUDS, segment_cloud};
use skylens_viewer::{FrameInfo, FrameSpec, Gpu3d, ViewInput, ViewScreen};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

/// Vertical field of view of the board camera (legacy `PerspectiveCamera(50, ..., 0.05, ...)`).
pub const FOV_DEG: f64 = 50.0;
const NEAR: f64 = 0.05;
/// Ids of the board's own meshes (markers), below the shared sync namespaces.
const MARKER_IDS: u64 = 1 << 20;
/// `#0d0f14`: what the point fades run to.
const BG_RGB: [u8; 3] = [0x0d, 0x0f, 0x14];

/// The legacy scene light on a mesh: `AmbientLight(0xffe8cf, 0.35)` (a hemisphere light with the
/// same sky and ground colour is the same thing) and `DirectionalLight(0xfff1d8, 0.5)` at
/// three.js (10, 20, 8), i.e. ENU (10, -8, 20). Opaque at `alpha` 1, blended below.
pub fn lambert(alpha: f32) -> Material {
    let ambient = config::rgb(0xffe8cf);
    let d = DVec3::new(10.0, -8.0, 20.0).normalize();
    Material {
        shading: Shading::Lambert {
            sky: ambient,
            ground: ambient,
            hemi_intensity: 0.35,
            sun_dir: [d.x as f32, d.y as f32, d.z as f32],
            sun_color: config::rgb(0xfff1d8),
            sun_intensity: 0.5,
            emissive: [0.0; 3],
        },
        tint: [1.0, 1.0, 1.0, alpha],
        blend: if alpha >= 1.0 { Blend::Opaque } else { Blend::Alpha },
        double_sided: true,
    }
}

pub struct StatusScene {
    board: Rc<RefCell<Board>>,
    fade: RevealFade,
    orbit: UserOrbit,
    follower: Follower,
    framed: bool,
    fog: (f64, f64),
    css_height: f64,
    point_size: Option<f32>,
    /// The animated live point layer.
    layer: LiveLayer,
    /// The operator asked to fit the view: the automatic camera waits until then (ms).
    fit_hold_until: f64,
    /// Mesh alpha per segment as last applied.
    mesh_alpha: BTreeMap<u32, f32>,
    /// Marker id -> mesh id base (pin, ring = base + 1).
    marker_ids: BTreeMap<String, u64>,
    next_marker: u64,
}

impl StatusScene {
    pub fn new(board: Rc<RefCell<Board>>) -> StatusScene {
        let framing = follow::Framing::DEFAULT;
        let mut orbit = UserOrbit::new(framing.opening_pose(), FOV_DEG);
        let (lo, hi) = framing.zoom_limits();
        orbit.min_distance = lo;
        orbit.max_distance = hi;
        StatusScene {
            board,
            fade: RevealFade::new(),
            orbit,
            follower: Follower::default(),
            framed: false,
            fog: framing.base_fog(),
            css_height: 1.0,
            point_size: None,
            layer: LiveLayer::new(LiveView::Main),
            fit_hold_until: 0.0,
            mesh_alpha: BTreeMap::new(),
            marker_ids: BTreeMap::new(),
            next_marker: 0,
        }
    }

    fn sync_markers(&mut self, gpu: &mut Gpu3d, board: &Board) {
        let markers = board.markers();
        let live: Vec<&String> = markers.iter().map(|(id, _, _)| id).collect();
        let gone: Vec<String> = self.marker_ids.keys().filter(|id| !live.contains(id)).cloned().collect();
        for id in gone {
            if let Some(base) = self.marker_ids.remove(&id) {
                gpu.meshes.remove(base);
                gpu.meshes.remove(base + 1);
            }
        }
        for (id, feet, person) in markers {
            let base = match self.marker_ids.get(&id) {
                Some(b) => *b,
                None => {
                    let b = MARKER_IDS + 2 * self.next_marker;
                    self.next_marker += 1;
                    debug_assert!(b + 1 < SEGMENT_CLOUDS);
                    self.marker_ids.insert(id.clone(), b);
                    gpu.meshes.upsert(&gpu.device, &gpu.queue, b, &markers::pin_mesh(feet), None, markers::pin_material(person));
                    b
                }
            };
            let ring = markers::ring_mesh(feet, markers::pulse(board.time, feet.x));
            gpu.meshes.upsert(&gpu.device, &gpu.queue, base + 1, &ring, None, markers::ring_material(person));
        }
    }

    /// "화면 맞춤": frame everything delivered, obliquely, and keep the automatic camera off for a while.
    fn fit_view(&mut self, board: &Board, info: &FrameInfo) {
        let Some((lo, hi)) = board.points_box() else { return };
        let aspect = info.css_width() / info.css_height().max(1.0);
        if let Some(f) = inset::fit(&board.fit_samples(FIT_SAMPLES_PER_SEGMENT), lo, hi, FOV_DEG, aspect) {
            self.orbit.set_pose(f.pose);
            self.orbit.max_distance = self.orbit.max_distance.max(f.max_distance);
            self.fit_hold_until = info.now_ms + 30_000.0;
        }
    }

    fn camera(&mut self, board: &Board, info: &FrameInfo) {
        if !self.framed && let Some(f) = board.framing {
            // First geometry: frame on it (legacy: on the placeholder cloud at load).
            self.framed = true;
            let (lo, hi) = f.zoom_limits();
            self.orbit.min_distance = lo;
            self.orbit.max_distance = hi;
            self.fog = f.base_fog();
            if board.active().is_none() && !self.orbit.auto_paused(info.now_ms) {
                self.orbit.set_pose(f.opening_pose());
            }
        }
        let framing = board.framing.unwrap_or(follow::Framing::DEFAULT);
        let centers = board.segment_centers();
        let desired = match board.focused() {
            Some((_, feet)) => Some(follow::focus_pose(
                feet,
                board.interior.as_ref(),
                board.corridor.as_ref(),
                &centers,
                board.active().as_ref(),
            )),
            None => board.active().map(|drone| {
                let chunk_centers: Vec<DVec3> = centers.iter().map(|(_, c)| *c).collect();
                let input = FollowInput {
                    framing,
                    floor_z: board.floor_z,
                    interior: board.interior.as_ref(),
                    path: board.corridor.as_ref(),
                    chunk_centers: &chunk_centers,
                };
                let out = self.follower.pose(&drone, &input, info.dt);
                self.fog = out.fog;
                out.pose
            }),
        };
        if let Some(d) = desired
            && !self.orbit.auto_paused(info.now_ms)
            && info.now_ms >= self.fit_hold_until
        {
            self.orbit.approach(&d, damp_factor(2.5, info.dt));
        }
        self.orbit.update(info.dt);
    }
}

impl ViewScreen for StatusScene {
    fn gpu_lost(&mut self) {
        self.mesh_alpha.clear();
        let board = self.board.clone();
        if let Ok(mut b) = board.try_borrow_mut() {
            self.layer.lost(&mut b);
        }
    }

    fn frame(&mut self, gpu: &mut Gpu3d, info: &FrameInfo) -> FrameSpec {
        self.css_height = info.css_height();
        let board_rc = self.board.clone();
        let mut board = board_rc.borrow_mut();
        board.time += info.dt;

        // Points: one scene per segment, faded in on arrival; the slot stream as the animated
        // live layer (fade and glide per point), coloured as the controls bar says.
        gpu.sync_segments(&board.store, 0.0);
        let (mode, scale, fit) = {
            let mut st = board.settings.borrow_mut();
            let fit = std::mem::take(&mut st.fit_requested);
            (st.color_mode, st.point_size / 2.0, fit)
        };
        let paint = Paint::of(mode, board.height_range());
        self.layer.pump(gpu, &mut board, paint, BG_RGB, info.now_ms / 1000.0);
        if fit {
            self.fit_view(&board, info);
        }
        for (seg, level) in board.store.levels() {
            self.fade.note_arrival(*seg, *level);
        }
        self.fade.update(info.dt);
        board.first_progress = self.fade.first_progress();
        for (seg, a) in self.fade.segments() {
            gpu.clouds.set_alpha(segment_cloud(seg), a);
        }

        // Mesh blocks, lit like the legacy scene, faded with their segment.
        let fade = &self.fade;
        gpu.sync_mesh_blocks(&board.store, |e| lambert(fade.alpha(e.segment)));
        let alphas: BTreeMap<u32, f32> = fade.segments().collect();
        if alphas != self.mesh_alpha {
            gpu.restyle_mesh_blocks(&board.store, |e| lambert(fade.alpha(e.segment)));
            self.mesh_alpha = alphas;
        }

        self.sync_markers(gpu, &board);
        self.camera(&board, info);

        let size = board.point_size_m * scale;
        let point_size = (self.point_size != Some(size)).then(|| {
            self.point_size = Some(size);
            PointSize::Metres(size)
        });
        FrameSpec {
            camera: self.orbit.pose().camera(FOV_DEG, NEAR),
            style: FrameStyle {
                clear: config::clear(color::STATUS_BG),
                fog: Some(Fog { color: config::rgb(color::STATUS_BG), near: self.fog.0 as f32, far: self.fog.1 as f32 }),
            },
            point_size,
        }
    }

    fn input(&mut self, input: &ViewInput) {
        let now = skylens_viewer::now_ms();
        match *input {
            ViewInput::PointerDown { x, y, button, shift } => self.orbit.pointer_down(x, y, button == 2, button == 1, shift),
            ViewInput::PointerMove { x, y, pressed } => {
                if pressed {
                    self.orbit.pointer_move(x, y, self.css_height);
                }
            }
            ViewInput::PointerUp { .. } => self.orbit.pointer_up(now),
            ViewInput::Wheel { delta_y, .. } => self.orbit.wheel(delta_y, now),
            ViewInput::Click { .. } => {}
        }
    }

    fn needs_redraw(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_light_is_the_legacy_scene_light() {
        let m = lambert(1.0);
        assert_eq!(m.blend, Blend::Opaque);
        let Shading::Lambert { sky, ground, hemi_intensity, sun_intensity, sun_dir, .. } = m.shading else {
            panic!("lit")
        };
        assert_eq!(sky, ground);
        assert_eq!((hemi_intensity, sun_intensity), (0.35, 0.5));
        // The key light is above and to the south-east.
        assert!(sun_dir[2] > 0.0 && sun_dir[0] > 0.0 && sun_dir[1] < 0.0);
        assert_eq!(lambert(0.5).blend, Blend::Alpha);
        assert_eq!(lambert(0.5).tint[3], 0.5);
    }
}
