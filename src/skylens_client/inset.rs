//! The point panel's own view (legacy `pointInset.ts`), mounted on `.point-inset__view`.
//!
//! A separate scene and camera: every delivered point segment (and the point slot stream)
//! coloured by height over the range the points actually span, on `#06090e`. The camera frames
//! everything delivered obliquely from the south-east and widens as segments arrive, until the
//! operator drags or zooms inside the panel; a double click hands the framing back.

use crate::board::{Board, FIT_SAMPLES_PER_SEGMENT};
use crate::glam::DVec3;
use crate::pointblitz::core::{FrameStyle, PointSize};
use crate::points;
use skylens_viewer::camera::{Pose, UserOrbit, damp_factor};
use skylens_viewer::config;
use skylens_viewer::view3d::{cloud_segment, retag, segment_cloud};
use skylens_viewer::{FrameInfo, FrameSpec, Gpu3d, ViewInput, ViewScreen};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

pub const FOV_DEG: f64 = 45.0;
const BACKGROUND: u32 = 0x06090e;

/// Camera basis (right, up, forward) looking from `eye` at `target`, z up.
fn basis(eye: DVec3, target: DVec3) -> (DVec3, DVec3, DVec3) {
    let f = (target - eye).normalize_or(DVec3::Y);
    let s = f.cross(DVec3::Z).normalize_or(DVec3::X);
    (s, s.cross(f), f)
}

/// A framing of the panel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fit {
    pub pose: Pose,
    pub near: f64,
    pub max_distance: f64,
}

/// A view that holds every delivered point, obliquely from the south-east (legacy `pose()`):
/// an even sample is projected and the aim and distance corrected until its 2nd..98th percentile
/// sits centred in the panel (a round capture leaves its box mostly empty, and the near side of an
/// oblique view projects larger than the far side).
pub fn fit(samples: &[DVec3], bmin: DVec3, bmax: DVec3, fov_deg: f64, aspect: f64) -> Option<Fit> {
    if samples.is_empty() || !(bmin.x <= bmax.x && bmin.y <= bmax.y && bmin.z <= bmax.z) {
        return None;
    }
    let r = ((bmax - bmin).length() / 2.0).max(1.0);
    let v_half = fov_deg.to_radians() / 2.0;
    let tan = v_half.tan();
    // three.js (0.45, 0.7, 0.6) in its y-up, z-south frame.
    let dir = DVec3::new(0.45, -0.6, 0.7).normalize();
    let mut target = (bmin + bmax) / 2.0;
    let mut dist = r / v_half.sin();
    let pct = |v: &[f64], q: f64| v[((q * v.len() as f64).floor() as usize).min(v.len() - 1)];
    for _ in 0..4 {
        let eye = target + dir * dist;
        let (s, u, f) = basis(eye, target);
        let mut xs = Vec::with_capacity(samples.len());
        let mut ys = Vec::with_capacity(samples.len());
        for p in samples {
            let rel = *p - eye;
            let depth = rel.dot(f);
            if depth.abs() < 1e-9 {
                continue;
            }
            xs.push(rel.dot(s) / (depth * tan * aspect));
            ys.push(rel.dot(u) / (depth * tan));
        }
        if xs.is_empty() {
            return None;
        }
        xs.sort_by(f64::total_cmp);
        ys.sort_by(f64::total_cmp);
        let (x0, x1, y0, y1) = (pct(&xs, 0.02), pct(&xs, 0.98), pct(&ys, 0.02), pct(&ys, 0.98));
        let half_h = tan * dist;
        let half_w = half_h * aspect;
        target += s * ((x0 + x1) / 2.0 * half_w) + u * ((y0 + y1) / 2.0 * half_h);
        let extent = ((x1 - x0) / 2.0).max((y1 - y0) / 2.0);
        dist *= (extent / 0.9).max(0.2);
    }
    Some(Fit {
        pose: Pose::new(target + dir * dist, target),
        near: (r / 2000.0).max(0.1),
        max_distance: (dist * 3.0).max(10.0),
    })
}

/// What a segment's cloud was built from.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Uploaded {
    epoch: u64,
    chunks: usize,
    range: (f64, f64),
}

pub struct InsetScene {
    board: Rc<RefCell<Board>>,
    orbit: UserOrbit,
    /// Follow the delivered bounds until the operator drags in the panel.
    auto: bool,
    /// Jump (not ease) to the next framing: nothing was framed yet.
    snap: bool,
    fit_key: Option<(DVec3, DVec3, u64, i64)>,
    fit: Option<Fit>,
    near: f64,
    uploaded: BTreeMap<u32, Uploaded>,
    css_height: f64,
    point_size: Option<f32>,
}

impl InsetScene {
    pub fn new(board: Rc<RefCell<Board>>) -> InsetScene {
        let mut orbit = UserOrbit::new(Pose::new([30.0, -40.0, 45.0], [0.0, 0.0, 0.0]), FOV_DEG);
        orbit.damping = 0.12;
        InsetScene {
            board,
            orbit,
            auto: true,
            snap: true,
            fit_key: None,
            fit: None,
            near: 0.5,
            uploaded: BTreeMap::new(),
            css_height: 1.0,
            point_size: None,
        }
    }

    /// Double click: frame everything delivered and keep following new segments.
    pub fn reframe(&mut self) {
        self.auto = true;
        self.snap = true;
    }

    fn sync_clouds(&mut self, gpu: &mut Gpu3d, board: &Board) {
        let Some(range) = board.height_range() else { return };
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
        // The slot stream: recoloured message by message; a new range rebuilds it.
        let key = range.0.to_bits().rotate_left(17) ^ range.1.to_bits();
        let (device, queue) = (&gpu.device, &gpu.queue);
        gpu.clouds.sync_slots_with(device, queue, &board.store, key, |m| points::recolor_slots_by_height(m, range.0, range.1));
    }

    fn desired(&mut self, board: &Board, aspect: f64) -> Option<Pose> {
        let (bmin, bmax) = board.points_box()?;
        // Refit only when what is shown or the panel's shape changes.
        let key = (bmin, bmax, board.total_points(), (aspect * 1000.0).round() as i64);
        if self.fit_key != Some(key) {
            let samples = board.fit_samples(FIT_SAMPLES_PER_SEGMENT);
            self.fit = fit(&samples, bmin, bmax, FOV_DEG, aspect);
            self.fit_key = Some(key);
            if let Some(f) = self.fit {
                self.near = f.near;
                self.orbit.max_distance = f.max_distance;
            }
        }
        self.fit.map(|f| f.pose)
    }
}

impl ViewScreen for InsetScene {
    fn gpu_lost(&mut self) {
        self.uploaded.clear();
    }

    fn frame(&mut self, gpu: &mut Gpu3d, info: &FrameInfo) -> FrameSpec {
        self.css_height = info.css_height();
        let board_rc = self.board.clone();
        let board = board_rc.borrow();
        self.sync_clouds(gpu, &board);

        let aspect = info.css_width() / info.css_height().max(1.0);
        if self.auto
            && let Some(d) = self.desired(&board, aspect)
        {
            if self.snap {
                self.snap = false;
                self.orbit.set_pose(d);
            } else {
                self.orbit.approach(&d, damp_factor(2.5, info.dt));
            }
        }
        self.orbit.update(info.dt);

        let size = board.point_size_m;
        let point_size = (self.point_size != Some(size)).then(|| {
            self.point_size = Some(size);
            PointSize::Metres(size)
        });
        FrameSpec {
            camera: self.orbit.pose().camera(FOV_DEG, self.near),
            style: FrameStyle { clear: config::clear(BACKGROUND), fog: None },
            point_size,
        }
    }

    fn input(&mut self, input: &ViewInput) {
        let now = skylens_viewer::now_ms();
        match *input {
            ViewInput::PointerDown { x, y, button, shift } => {
                self.auto = false;
                self.orbit.pointer_down(x, y, button == 2, button == 1, shift);
            }
            ViewInput::PointerMove { x, y, pressed } => {
                if pressed {
                    self.orbit.pointer_move(x, y, self.css_height);
                }
            }
            ViewInput::PointerUp { .. } => self.orbit.pointer_up(now),
            ViewInput::Wheel { delta_y, .. } => {
                self.auto = false;
                self.orbit.wheel(delta_y, now);
            }
            ViewInput::Click { .. } => {}
        }
    }

    fn needs_redraw(&self) -> bool {
        // Drawn only while the panel is open and holds points (it is hidden otherwise).
        self.board.try_borrow().map_or(true, |b| b.point_layer_on && b.total_points() > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fit_centres_the_points_and_sees_them_all() {
        let samples: Vec<DVec3> = (0..400)
            .map(|i| {
                let a = f64::from(i) * 0.1;
                DVec3::new(100.0 + 30.0 * a.cos(), 50.0 + 30.0 * a.sin(), f64::from(i % 5))
            })
            .collect();
        let bmin = DVec3::new(70.0, 20.0, 0.0);
        let bmax = DVec3::new(130.0, 80.0, 4.0);
        let f = fit(&samples, bmin, bmax, FOV_DEG, 1.5).unwrap();
        // From the south-east and above.
        let d = f.pose.eye - f.pose.target;
        assert!(d.x > 0.0 && d.y < 0.0 && d.z > 0.0);
        assert!(f.pose.target.distance(DVec3::new(100.0, 50.0, 2.0)) < 10.0);
        // The 2nd..98th percentile is fitted into 90 % of the panel: practically all is in view.
        let p = skylens_viewer::Projector::new(f.pose.camera(FOV_DEG, f.near), 300.0, 200.0);
        let seen = samples.iter().filter(|s| p.project_visible(s.to_array()).is_some()).count();
        assert!(seen >= 390, "{seen}");
        assert!(f.max_distance >= 10.0);
        assert!(fit(&[], bmin, bmax, FOV_DEG, 1.5).is_none());
    }
}
