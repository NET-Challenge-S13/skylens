//! Camera controllers ported from the legacy three.js viewers, in pointblitz ENU (x east,
//! y north, z up), and the projection used to place DOM overlays and pick the map.
//!
//! Each controller produces a [`Pose`]; the screen picks which one drives the frame:
//!
//! | legacy | here |
//! |---|---|
//! | `towerViewer.ts` `updateOverviewCamera` (no drone: slow spin around the area) | [`OverviewOrbit`] |
//! | `towerViewer.ts` `updateChaseCamera` (behind and above the active drone, lerp 0.06) | [`ChaseCamera`] |
//! | `cameraSync.ts` SYNCED / FOCUSING / LOCKED / RETURNING with eased tweens | [`CameraSync`], [`Tween`] |
//! | `statusViewer.ts` follow (`1 - exp(-2.5 dt)` toward the desired pose) | [`DampedFollow`] |
//! | `statusViewer.ts` `OrbitControls` (damping 0.08, auto camera paused while dragging and 3.5 s after) | [`UserOrbit`] |
//!
//! The legacy per-frame lerps assumed 60 fps; here they are made frame-rate independent with
//! [`per_frame`], which gives the same motion at 60 fps.

use glam::DVec3;
use pointblitz::Camera;

const UP: DVec3 = DVec3::Z;

/// Smoothstep ease-in-out on [0, 1] (legacy `math.ts` `easeInOut`).
pub fn ease_in_out(t: f64) -> f64 {
    let c = t.clamp(0.0, 1.0);
    c * c * (3.0 - 2.0 * c)
}

/// Frame-rate independent damping factor; `lambda` is the per-second rate (legacy `dampFactor`).
pub fn damp_factor(lambda: f64, dt: f64) -> f64 {
    1.0 - (-lambda * dt.max(0.0)).exp()
}

/// A per-frame lerp factor tuned at 60 fps, converted for a frame of `dt` seconds.
pub fn per_frame(factor_at_60fps: f64, dt: f64) -> f64 {
    1.0 - (1.0 - factor_at_60fps.clamp(0.0, 1.0)).powf(dt.max(0.0) * 60.0)
}

/// Where the camera is and what it looks at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub eye: DVec3,
    pub target: DVec3,
}

impl Pose {
    pub fn new(eye: impl Into<DVec3>, target: impl Into<DVec3>) -> Pose {
        Pose { eye: eye.into(), target: target.into() }
    }

    /// pointblitz camera with z up.
    pub fn camera(&self, fov_y_deg: f64, near: f64) -> Camera {
        let mut c = Camera::new(self.eye.to_array(), self.target.to_array(), UP.to_array(), fov_y_deg);
        c.near = near;
        c
    }

    pub fn lerp(&self, to: &Pose, t: f64) -> Pose {
        Pose { eye: self.eye.lerp(to.eye, t), target: self.target.lerp(to.target, t) }
    }

    pub fn distance(&self) -> f64 {
        self.eye.distance(self.target)
    }
}

// ---------------------------------------------------------------------------------------------
// Overview orbit (tower, no drone connected)
// ---------------------------------------------------------------------------------------------

/// Slow spin around the operating area while no drone is connected (legacy tower standby).
#[derive(Debug, Clone, PartialEq)]
pub struct OverviewOrbit {
    pub center: DVec3,
    pub radius: f64,
    /// Radians; the legacy angle ran counter-clockwise in the three.js x/z plane.
    pub angle: f64,
    pub spin_deg_per_s: f64,
    /// Eye height above the centre as a fraction of the radius.
    pub height_ratio: f64,
}

impl Default for OverviewOrbit {
    /// The tower's standby values: radius 40 until geometry loads, 4 degrees per second, eye at
    /// half the radius.
    fn default() -> Self {
        Self { center: DVec3::ZERO, radius: 40.0, angle: 0.6, spin_deg_per_s: 4.0, height_ratio: 0.5 }
    }
}

impl OverviewOrbit {
    /// Frames the orbit on loaded geometry: radius `max(width, depth) * 0.62 + 8` (legacy
    /// `frameOverview`).
    pub fn frame_bounds(&mut self, min: DVec3, max: DVec3) {
        self.center = (min + max) / 2.0;
        let size = max - min;
        self.radius = size.x.max(size.y) * 0.62 + 8.0;
    }

    pub fn pose(&self) -> Pose {
        let (s, c) = self.angle.sin_cos();
        let r = self.radius;
        // three.js placed the eye at (cos a, sin a) in its x/z plane, and three z is -north.
        let eye = self.center + DVec3::new(c * r, -s * r, r * self.height_ratio);
        Pose { eye, target: self.center }
    }

    /// Advances the spin and returns the pose.
    pub fn step(&mut self, dt: f64) -> Pose {
        self.angle += dt * self.spin_deg_per_s.to_radians();
        self.pose()
    }
}

// ---------------------------------------------------------------------------------------------
// Chase camera (tower, active drone)
// ---------------------------------------------------------------------------------------------

/// Behind and above the active drone, eased toward it (legacy `updateChaseCamera`).
#[derive(Debug, Clone, PartialEq)]
pub struct ChaseCamera {
    /// Metres behind and above the drone before `scale`.
    pub back: f64,
    pub up: f64,
    /// Legacy `droneViewScale()` (0.15 on map scenes).
    pub scale: f64,
    /// Lerp per frame at 60 fps (legacy `CAMERA_LERP`).
    pub lerp: f64,
    pose: Option<Pose>,
}

impl Default for ChaseCamera {
    fn default() -> Self {
        Self { back: 9.0, up: 4.0, scale: 0.15, lerp: 0.06, pose: None }
    }
}

impl ChaseCamera {
    /// The next drone is eased in from wherever it is, not from a stale pose (the overview calls
    /// this every frame it runs).
    pub fn reset(&mut self) {
        self.pose = None;
    }

    pub fn desired(&self, drone: DVec3, forward: DVec3) -> Pose {
        let f = if forward.length_squared() > 1e-4 { forward } else { DVec3::Y };
        let eye = drone - f * (self.back * self.scale) + UP * (self.up * self.scale);
        Pose { eye, target: drone }
    }

    /// Eases toward the pose behind `drone` looking along `forward` (ENU, need not be unit).
    pub fn step(&mut self, drone: DVec3, forward: DVec3, dt: f64) -> Pose {
        let desired = self.desired(drone, forward);
        let next = match self.pose {
            None => desired,
            Some(p) => p.lerp(&desired, per_frame(self.lerp, dt)),
        };
        self.pose = Some(next);
        next
    }

    pub fn pose(&self) -> Option<Pose> {
        self.pose
    }
}

/// Unit ENU forward vector of a compass heading (degrees clockwise from north), level.
pub fn heading_forward(heading_deg: f64) -> DVec3 {
    let h = heading_deg.to_radians();
    DVec3::new(h.sin(), h.cos(), 0.0)
}

// ---------------------------------------------------------------------------------------------
// Tween and camera sync (status board)
// ---------------------------------------------------------------------------------------------

/// Eased move from one pose to another over a fixed time (real seconds, not the sim clock).
#[derive(Debug, Clone, PartialEq)]
pub struct Tween {
    pub from: Pose,
    pub to: Pose,
    pub seconds: f64,
    elapsed: f64,
}

impl Tween {
    pub fn new(from: Pose, to: Pose, seconds: f64) -> Tween {
        Tween { from, to, seconds, elapsed: 0.0 }
    }

    /// Advances and returns the pose and whether the tween has finished.
    pub fn step(&mut self, dt: f64) -> (Pose, bool) {
        self.elapsed += dt.max(0.0);
        let t = self.elapsed / self.seconds.max(1e-6);
        (self.from.lerp(&self.to, ease_in_out(t)), t >= 1.0)
    }

    pub fn progress(&self) -> f64 {
        (self.elapsed / self.seconds.max(1e-6)).min(1.0)
    }
}

/// Camera sync states (legacy §8.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncState {
    Synced,
    Focusing,
    Locked,
    Returning,
}

impl SyncState {
    /// The `.cam-state` chip text.
    pub fn label(self) -> &'static str {
        match self {
            SyncState::Synced => "CAM · SYNCED",
            SyncState::Focusing => "CAM · FOCUSING",
            SyncState::Locked => "CAM · LOCKED",
            SyncState::Returning => "CAM · RETURNING",
        }
    }
}

/// What [`CameraSync::step`] reports.
#[derive(Debug, Clone, PartialEq)]
pub enum SyncEvent {
    StateChanged(SyncState),
    /// The focus finished returning; the detection is no longer focused.
    Released(String),
}

/// Legacy `cameraSync.ts`: SYNCED is a slow overview orbit around the reconstruction; a newly
/// revealed detection tweens the camera in (FOCUSING), holds (LOCKED) until it is confirmed, then
/// tweens back (RETURNING) to where the orbit resumes.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraSync {
    pub state: SyncState,
    /// Detection being focused.
    pub focused: Option<String>,
    center: DVec3,
    dist: f64,
    elevation: f64,
    angle: f64,
    /// Rad/s of the SYNCED orbit.
    pub orbit_speed: f64,
    pub tween_seconds: f64,
    pub focus_distance: f64,
    pose: Pose,
    initialized: bool,
    tween: Option<Tween>,
}

impl CameraSync {
    /// Frames the orbit on the scene bounds: distance fits the bounding sphere in a ~50 degree
    /// view with headroom (`radius * 2.4`), elevation `0.35` of that.
    pub fn new(min: DVec3, max: DVec3) -> CameraSync {
        let center = (min + max) / 2.0;
        let radius = ((max - min).length() / 2.0).max(1.0);
        let dist = radius * 2.4;
        let mut s = CameraSync {
            state: SyncState::Synced,
            focused: None,
            center,
            dist,
            elevation: dist * 0.35,
            angle: std::f64::consts::FRAC_PI_4,
            orbit_speed: 0.12,
            tween_seconds: crate::config::camera::TWEEN_SECONDS,
            focus_distance: crate::config::camera::FOCUS_DISTANCE,
            pose: Pose { eye: DVec3::ZERO, target: center },
            initialized: false,
            tween: None,
        };
        s.pose = s.orbit_pose();
        s
    }

    /// Re-frames the orbit (the reconstruction grew); the angle is kept.
    pub fn reframe(&mut self, min: DVec3, max: DVec3) {
        let angle = self.angle;
        let (state, focused, pose, initialized, tween) =
            (self.state, self.focused.take(), self.pose, self.initialized, self.tween.take());
        *self = CameraSync { angle, state, focused, pose, initialized, tween, ..CameraSync::new(min, max) };
    }

    fn orbit_pose(&self) -> Pose {
        let (s, c) = self.angle.sin_cos();
        // three.js: (cos a * d, elevation, sin a * d) with z = -north.
        let eye = self.center + DVec3::new(c * self.dist, -s * self.dist, self.elevation);
        Pose { eye, target: self.center }
    }

    pub fn pose(&self) -> Pose {
        self.pose
    }

    /// Starts focusing a detection at `at` (ENU), approached from the current direction so it is
    /// not occluded. Ignored unless SYNCED with nothing focused.
    pub fn focus(&mut self, id: impl Into<String>, at: DVec3) -> Option<SyncEvent> {
        if self.state != SyncState::Synced || self.focused.is_some() {
            return None;
        }
        if !self.initialized {
            self.pose = self.orbit_pose();
            self.initialized = true;
        }
        let mut dir = at - self.pose.eye;
        if dir.length_squared() < 1e-6 {
            dir = DVec3::NEG_Y;
        }
        let dir = dir.normalize();
        let mut eye = at - dir * self.focus_distance;
        eye.z += self.focus_distance * 0.35;
        self.tween = Some(Tween::new(self.pose, Pose { eye, target: at }, self.tween_seconds));
        self.focused = Some(id.into());
        self.state = SyncState::Focusing;
        Some(SyncEvent::StateChanged(SyncState::Focusing))
    }

    /// The focused detection was confirmed: tween back to the orbit.
    pub fn confirm(&mut self) -> Option<SyncEvent> {
        if !matches!(self.state, SyncState::Focusing | SyncState::Locked) {
            return None;
        }
        self.tween = Some(Tween::new(self.pose, self.orbit_pose(), self.tween_seconds));
        self.state = SyncState::Returning;
        Some(SyncEvent::StateChanged(SyncState::Returning))
    }

    pub fn step(&mut self, dt: f64) -> Vec<SyncEvent> {
        let mut events = Vec::new();
        match self.state {
            SyncState::Synced => {
                // The angle only advances here, so a return lands where the orbit resumes.
                self.angle += self.orbit_speed * dt;
                self.pose = self.orbit_pose();
                self.initialized = true;
            }
            SyncState::Locked => {}
            SyncState::Focusing | SyncState::Returning => {
                let (pose, done) = match self.tween.as_mut() {
                    Some(t) => t.step(dt),
                    None => (self.pose, true),
                };
                self.pose = pose;
                if done {
                    self.tween = None;
                    if self.state == SyncState::Focusing {
                        self.state = SyncState::Locked;
                        events.push(SyncEvent::StateChanged(SyncState::Locked));
                    } else {
                        self.state = SyncState::Synced;
                        if let Some(id) = self.focused.take() {
                            events.push(SyncEvent::Released(id));
                        }
                        events.push(SyncEvent::StateChanged(SyncState::Synced));
                    }
                }
            }
        }
        events
    }
}

// ---------------------------------------------------------------------------------------------
// Damped follow
// ---------------------------------------------------------------------------------------------

/// Eases a pose toward a desired one at `rate` per second (legacy status follow: 2.5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DampedFollow {
    pub rate: f64,
}

impl Default for DampedFollow {
    fn default() -> Self {
        Self { rate: 2.5 }
    }
}

impl DampedFollow {
    pub fn step(&self, current: &Pose, desired: &Pose, dt: f64) -> Pose {
        current.lerp(desired, damp_factor(self.rate, dt))
    }
}

// ---------------------------------------------------------------------------------------------
// User orbit (three.js OrbitControls with damping)
// ---------------------------------------------------------------------------------------------

/// Which drag the pointer is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DragKind {
    Rotate,
    Pan,
    Dolly,
}

/// Operator navigation like three.js `OrbitControls` (left drag rotates, right or shift drag pans,
/// wheel zooms, `enableDamping` with `dampingFactor` 0.08), plus the status board's rule that the
/// automatic camera pauses while dragging and for 3.5 s after.
#[derive(Debug, Clone, PartialEq)]
pub struct UserOrbit {
    pub eye: DVec3,
    pub target: DVec3,
    pub fov_y_deg: f64,
    pub min_distance: f64,
    pub max_distance: f64,
    pub damping: f64,
    pub rotate_speed: f64,
    pub zoom_speed: f64,
    /// How long the automatic camera stays paused after a drag ends (ms).
    pub grace_ms: f64,
    /// The operator owns the camera for good (legacy `freeCamera`).
    pub held: bool,
    delta_theta: f64,
    delta_phi: f64,
    pan_offset: DVec3,
    scale: f64,
    drag: Option<(DragKind, f32, f32)>,
    grace_until_ms: f64,
}

impl UserOrbit {
    pub fn new(pose: Pose, fov_y_deg: f64) -> UserOrbit {
        UserOrbit {
            eye: pose.eye,
            target: pose.target,
            fov_y_deg,
            min_distance: 0.0,
            max_distance: f64::INFINITY,
            damping: 0.08,
            rotate_speed: 1.0,
            zoom_speed: 1.0,
            grace_ms: 3500.0,
            held: false,
            delta_theta: 0.0,
            delta_phi: 0.0,
            pan_offset: DVec3::ZERO,
            scale: 1.0,
            drag: None,
            grace_until_ms: 0.0,
        }
    }

    pub fn pose(&self) -> Pose {
        Pose { eye: self.eye, target: self.target }
    }

    /// Whether the operator is steering (dragging, within the grace period, or holding the camera).
    pub fn auto_paused(&self, now_ms: f64) -> bool {
        self.held || self.drag.is_some() || now_ms < self.grace_until_ms
    }

    pub fn dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// Moves eye and target toward an automatic pose (call before [`Self::update`], only when not
    /// [`Self::auto_paused`]).
    pub fn approach(&mut self, desired: &Pose, k: f64) {
        self.eye = self.eye.lerp(desired.eye, k);
        self.target = self.target.lerp(desired.target, k);
    }

    pub fn set_pose(&mut self, pose: Pose) {
        self.eye = pose.eye;
        self.target = pose.target;
        self.delta_theta = 0.0;
        self.delta_phi = 0.0;
        self.pan_offset = DVec3::ZERO;
        self.scale = 1.0;
    }

    /// Pointer pressed on the view (CSS px). `secondary` is the right button; `shift` turns a
    /// left drag into a pan, `middle` dollies.
    pub fn pointer_down(&mut self, x: f32, y: f32, secondary: bool, middle: bool, shift: bool) {
        let kind = if middle {
            DragKind::Dolly
        } else if secondary || shift {
            DragKind::Pan
        } else {
            DragKind::Rotate
        };
        self.drag = Some((kind, x, y));
    }

    /// Pointer moved while pressed. `view_height` is the view's CSS height.
    pub fn pointer_move(&mut self, x: f32, y: f32, view_height: f64) {
        let Some((kind, lx, ly)) = self.drag else { return };
        let (dx, dy) = (f64::from(x - lx), f64::from(y - ly));
        let h = view_height.max(1.0);
        match kind {
            DragKind::Rotate => {
                self.delta_theta -= std::f64::consts::TAU * dx / h * self.rotate_speed;
                self.delta_phi -= std::f64::consts::TAU * dy / h * self.rotate_speed;
            }
            DragKind::Pan => self.pan(dx, dy, h),
            DragKind::Dolly => {
                if dy > 0.0 {
                    self.scale /= self.zoom_scale();
                } else if dy < 0.0 {
                    self.scale *= self.zoom_scale();
                }
            }
        }
        self.drag = Some((kind, x, y));
    }

    /// Pointer released (or cancelled): the automatic camera stays paused for the grace period.
    pub fn pointer_up(&mut self, now_ms: f64) {
        if self.drag.take().is_some() {
            self.grace_until_ms = now_ms + self.grace_ms;
        }
    }

    /// Wheel; `delta_y > 0` zooms out. Like OrbitControls, a wheel turn counts as a drag that
    /// started and ended now.
    pub fn wheel(&mut self, delta_y: f64, now_ms: f64) {
        if delta_y > 0.0 {
            self.scale /= self.zoom_scale();
        } else if delta_y < 0.0 {
            self.scale *= self.zoom_scale();
        }
        self.grace_until_ms = now_ms + self.grace_ms;
    }

    fn zoom_scale(&self) -> f64 {
        0.95f64.powf(self.zoom_speed)
    }

    fn pan(&mut self, dx: f64, dy: f64, h: f64) {
        let offset = self.eye - self.target;
        let target_distance = offset.length() * (self.fov_y_deg.to_radians() / 2.0).tan();
        let (right, cam_up) = camera_axes(self.eye, self.target);
        // Dragging right moves the scene right (target left); dragging down moves it down.
        self.pan_offset += right * (-2.0 * dx * target_distance / h);
        self.pan_offset += cam_up * (2.0 * dy * target_distance / h);
    }

    /// Applies the damped rotation and pan and the zoom (call once per frame).
    pub fn update(&mut self, dt: f64) {
        let f = per_frame(self.damping, dt);
        let offset = self.eye - self.target;
        let mut r = offset.length().max(1e-9);
        // Spherical about z, measured like three.js (theta from -north toward east, phi from up).
        let mut theta = offset.x.atan2(-offset.y);
        let mut phi = (offset.z / r).clamp(-1.0, 1.0).acos();

        theta += self.delta_theta * f;
        phi += self.delta_phi * f;
        phi = phi.clamp(1e-6, std::f64::consts::PI - 1e-6);
        r = (r * self.scale).clamp(self.min_distance.max(1e-6), self.max_distance);

        self.target += self.pan_offset * f;
        let (st, ct) = theta.sin_cos();
        let (sp, cp) = phi.sin_cos();
        self.eye = self.target + DVec3::new(r * sp * st, -r * sp * ct, r * cp);

        self.delta_theta *= 1.0 - f;
        self.delta_phi *= 1.0 - f;
        self.pan_offset *= 1.0 - f;
        self.scale = 1.0;
    }

    /// Whether damping is still moving the camera.
    pub fn settling(&self) -> bool {
        self.delta_theta.abs() > 1e-5 || self.delta_phi.abs() > 1e-5 || self.pan_offset.length() > 1e-4
    }
}

/// Camera right and up vectors (like pointblitz's view matrix, up = z).
fn camera_axes(eye: DVec3, target: DVec3) -> (DVec3, DVec3) {
    let f = (target - eye).normalize_or(DVec3::Y);
    let s = f.cross(UP).normalize_or(DVec3::X);
    let u = s.cross(f);
    (s, u)
}

// ---------------------------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------------------------

/// Maps ENU positions to CSS pixels of the 3D view and back to rays, for DOM-positioned overlays
/// (labels, cards) and map picking (route planning clicks).
#[derive(Debug, Clone, PartialEq)]
pub struct Projector {
    pub camera: Camera,
    /// The view's size in CSS pixels.
    pub width: f64,
    pub height: f64,
}

impl Projector {
    pub fn new(camera: Camera, width: f64, height: f64) -> Projector {
        Projector { camera, width: width.max(1.0), height: height.max(1.0) }
    }

    /// CSS pixel position of an ENU point (x right, y down), or None behind the camera.
    pub fn project(&self, p: [f64; 3]) -> Option<[f64; 2]> {
        // project_px works in whole pixels of a target; scale up for sub-pixel accuracy.
        const K: f64 = 16.0;
        let w = (self.width * K).round().max(1.0) as u32;
        let h = (self.height * K).round().max(1.0) as u32;
        self.camera
            .project_px(p, w, h)
            .map(|[x, y]| [x * self.width / f64::from(w), y * self.height / f64::from(h)])
    }

    /// Like [`Self::project`] but also None outside the view.
    pub fn project_visible(&self, p: [f64; 3]) -> Option<[f64; 2]> {
        self.project(p)
            .filter(|[x, y]| (0.0..=self.width).contains(x) && (0.0..=self.height).contains(y))
    }

    /// The ray through a CSS pixel: origin at the eye and a unit direction.
    pub fn ray(&self, x: f64, y: f64) -> (DVec3, DVec3) {
        let eye = self.camera.eye;
        let f = (self.camera.target - eye).normalize_or(DVec3::Y);
        let s = f.cross(self.camera.up).normalize_or(DVec3::X);
        let u = s.cross(f);
        let tan_half = (self.camera.fov_y_deg.to_radians() / 2.0).tan();
        let aspect = self.width / self.height;
        let ndc_x = 2.0 * x / self.width - 1.0;
        let ndc_y = 1.0 - 2.0 * y / self.height;
        let dir = f + s * (ndc_x * tan_half * aspect) + u * (ndc_y * tan_half);
        (eye, dir.normalize())
    }

    /// Where the ray through a CSS pixel meets the horizontal plane `z = up`, if in front.
    pub fn pick_plane(&self, x: f64, y: f64, up: f64) -> Option<DVec3> {
        let (o, d) = self.ray(x, y);
        if d.z.abs() < 1e-9 {
            return None;
        }
        let t = (up - o.z) / d.z;
        (t > 0.0).then(|| o + d * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: DVec3, b: DVec3, eps: f64) -> bool {
        a.distance(b) < eps
    }

    #[test]
    fn easing_and_damping() {
        assert_eq!(ease_in_out(-1.0), 0.0);
        assert_eq!(ease_in_out(0.5), 0.5);
        assert_eq!(ease_in_out(2.0), 1.0);
        assert!((damp_factor(2.5, 1.0) - (1.0 - (-2.5f64).exp())).abs() < 1e-12);
        assert!((per_frame(0.06, 1.0 / 60.0) - 0.06).abs() < 1e-12);
        // One 30 fps frame moves as far as two 60 fps frames.
        let two_60 = 1.0 - (1.0 - 0.06f64).powi(2);
        assert!((per_frame(0.06, 1.0 / 30.0) - two_60).abs() < 1e-12);
    }

    #[test]
    fn overview_orbits_the_framed_area() {
        let mut o = OverviewOrbit::default();
        o.frame_bounds(DVec3::new(-100.0, -50.0, 0.0), DVec3::new(100.0, 50.0, 20.0));
        assert!((o.radius - (200.0 * 0.62 + 8.0)).abs() < 1e-9);
        let a = o.pose();
        assert!(close(a.target, DVec3::new(0.0, 0.0, 10.0), 1e-9));
        assert!((a.eye.z - (10.0 + o.radius * 0.5)).abs() < 1e-9);
        let horizontal = (a.eye - a.target).truncate().length();
        assert!((horizontal - o.radius).abs() < 1e-9);
        let b = o.step(1.0);
        assert!((o.angle - (0.6 + 4f64.to_radians())).abs() < 1e-12);
        assert!(!close(a.eye, b.eye, 1e-6));
    }

    #[test]
    fn chase_sits_behind_and_above_and_eases() {
        let mut c = ChaseCamera::default();
        let drone = DVec3::new(10.0, 20.0, 60.0);
        let north = heading_forward(0.0);
        let first = c.step(drone, north, 1.0 / 60.0);
        // Snaps on the first frame: 9 * 0.15 m south, 4 * 0.15 m up.
        assert!(close(first.eye, DVec3::new(10.0, 20.0 - 1.35, 60.6), 1e-9));
        assert_eq!(first.target, drone);
        // Then eases 6 % per 60 fps frame.
        let moved = drone + DVec3::new(10.0, 0.0, 0.0);
        let next = c.step(moved, north, 1.0 / 60.0);
        assert!((next.target.x - (10.0 + 0.6)).abs() < 1e-9);
        c.reset();
        assert!(c.pose().is_none());
        let east = heading_forward(90.0);
        assert!(close(east, DVec3::X, 1e-12));
    }

    #[test]
    fn camera_sync_focuses_locks_and_returns() {
        let mut s = CameraSync::new(DVec3::splat(-10.0), DVec3::splat(10.0));
        assert_eq!(s.state.label(), "CAM · SYNCED");
        let before = s.pose();
        s.step(1.0);
        assert!(!close(before.eye, s.pose().eye, 1e-6), "orbit advances while synced");

        let at = DVec3::new(3.0, 4.0, 0.0);
        assert_eq!(s.focus("p1", at), Some(SyncEvent::StateChanged(SyncState::Focusing)));
        assert_eq!(s.focus("p2", at), None, "one focus at a time");
        let mut events = Vec::new();
        for _ in 0..200 {
            events.extend(s.step(0.01));
        }
        assert_eq!(s.state, SyncState::Locked);
        assert_eq!(events, vec![SyncEvent::StateChanged(SyncState::Locked)]);
        assert!(close(s.pose().target, at, 1e-9));
        assert!(s.pose().eye.z > at.z, "approached from above");

        let resume = s.orbit_pose();
        assert_eq!(s.confirm(), Some(SyncEvent::StateChanged(SyncState::Returning)));
        let mut events = Vec::new();
        for _ in 0..200 {
            events.extend(s.step(0.01));
        }
        assert_eq!(s.state, SyncState::Synced);
        assert_eq!(events[0], SyncEvent::Released("p1".into()));
        assert!(s.focused.is_none());
        // Returned to where the orbit resumes (plus the one synced step after landing).
        assert!(s.pose().target.distance(resume.target) < 1e-9);
    }

    #[test]
    fn user_orbit_rotates_with_damping_and_pauses_the_auto_camera() {
        let pose = Pose::new([0.0, -100.0, 50.0], [0.0, 0.0, 0.0]);
        let mut o = UserOrbit::new(pose, 50.0);
        assert!(!o.auto_paused(0.0));
        o.pointer_down(100.0, 100.0, false, false, false);
        assert!(o.auto_paused(0.0));
        o.pointer_move(150.0, 100.0, 600.0);
        let r0 = o.pose().distance();
        o.update(1.0 / 60.0);
        // Rotation keeps the distance, applied only partly (damping).
        assert!((o.pose().distance() - r0).abs() < 1e-6);
        assert!(o.settling());
        for _ in 0..600 {
            o.update(1.0 / 60.0);
        }
        assert!(!o.settling());
        // Dragging right by 50 of 600 px turns the eye by 2 pi * 50 / 600 around z.
        let turned = (o.eye.x).atan2(-o.eye.y);
        assert!((turned + std::f64::consts::TAU * 50.0 / 600.0).abs() < 1e-3, "{turned}");

        o.pointer_up(1_000.0);
        assert!(o.auto_paused(4_499.0));
        assert!(!o.auto_paused(4_501.0));
        o.wheel(1.0, 10_000.0);
        assert!(o.auto_paused(13_000.0));
        o.update(1.0 / 60.0);
        assert!((o.pose().distance() - r0 / 0.95).abs() < 1e-6);
    }

    #[test]
    fn user_orbit_clamps_distance_and_pans_the_target() {
        let pose = Pose::new([0.0, -10.0, 0.0], [0.0, 0.0, 0.0]);
        let mut o = UserOrbit::new(pose, 50.0);
        o.min_distance = 5.0;
        for _ in 0..50 {
            o.wheel(-1.0, 0.0);
        }
        o.update(1.0 / 60.0);
        assert!((o.pose().distance() - 5.0).abs() < 1e-9);

        o.pointer_down(0.0, 0.0, true, false, false);
        o.pointer_move(30.0, 0.0, 300.0);
        o.pointer_up(0.0);
        for _ in 0..600 {
            o.update(1.0 / 60.0);
        }
        // Looking north, a right drag moves the target west.
        assert!(o.target.x < -0.1);
        assert!(o.target.y.abs() < 1e-6);
    }

    #[test]
    fn projector_round_trips_through_rays() {
        let cam = Pose::new([100.0, -200.0, 150.0], [120.0, 0.0, 0.0]).camera(50.0, 0.1);
        let p = Projector::new(cam, 800.0, 450.0);
        let c = p.project([120.0, 0.0, 0.0]).unwrap();
        assert!((c[0] - 400.0).abs() < 0.05 && (c[1] - 225.0).abs() < 0.05);
        let world = DVec3::new(150.0, 30.0, 0.0);
        let px = p.project_visible(world.to_array()).unwrap();
        let hit = p.pick_plane(px[0], px[1], 0.0).unwrap();
        assert!(hit.distance(world) < 0.5, "{hit:?}");
        assert!(p.project_visible([120.0, 0.0, 1e6]).is_none());
        assert!(p.project([100.0, -400.0, 150.0]).is_none(), "behind the camera");
    }
}
