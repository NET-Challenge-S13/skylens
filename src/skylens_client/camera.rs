//! Orbit camera for the 3D view, in pointblitz's ENU frame (x east, y north, z up).

use pointblitz::Camera;

#[derive(Debug, Clone, PartialEq)]
pub struct OrbitCamera {
    pub target: [f64; 3],
    pub distance: f64,
    /// Radians clockwise from north; 0 looks north.
    pub yaw: f64,
    /// Radians above the horizon.
    pub pitch: f64,
    pub fov_y_deg: f64,
    /// The operator moved the camera; automatic framing stops.
    pub user_moved: bool,
}

pub const MIN_DISTANCE: f64 = 5.0;
pub const MAX_DISTANCE: f64 = 20_000.0;
const MIN_PITCH: f64 = 0.05;
const MAX_PITCH: f64 = 1.55;

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: [0.0, 0.0, 0.0],
            distance: 600.0,
            yaw: 0.0,
            pitch: 0.75,
            fov_y_deg: 50.0,
            user_moved: false,
        }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> [f64; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let d = self.distance;
        [
            self.target[0] - sy * cp * d,
            self.target[1] - cy * cp * d,
            self.target[2] + sp * d,
        ]
    }

    pub fn camera(&self) -> Camera {
        let mut c = Camera::new(self.eye(), self.target, [0.0, 0.0, 1.0], self.fov_y_deg);
        // Kilometre-scale scenes: a slightly farther near plane keeps reversed-Z precise.
        c.near = (self.distance * 0.001).clamp(0.05, 2.0);
        c
    }

    /// Drag with the primary button: turn around the target.
    pub fn orbit(&mut self, dx_px: f64, dy_px: f64) {
        self.yaw += dx_px * 0.005;
        self.pitch = (self.pitch + dy_px * 0.005).clamp(MIN_PITCH, MAX_PITCH);
        self.user_moved = true;
    }

    /// Drag with the secondary button (or shift): move the target over the ground.
    pub fn pan(&mut self, dx_px: f64, dy_px: f64) {
        let (sy, cy) = self.yaw.sin_cos();
        let k = self.distance * 0.0015;
        let right = [cy, -sy];
        let forward = [sy, cy];
        self.target[0] += (-right[0] * dx_px + forward[0] * dy_px) * k;
        self.target[1] += (-right[1] * dx_px + forward[1] * dy_px) * k;
        self.user_moved = true;
    }

    /// Wheel: positive steps move away.
    pub fn zoom(&mut self, steps: f64) {
        self.distance = (self.distance * (steps * 0.12).exp()).clamp(MIN_DISTANCE, MAX_DISTANCE);
        self.user_moved = true;
    }

    /// Re-centres without counting as an operator move.
    pub fn frame(&mut self, target: [f64; 3], distance: Option<f64>) {
        self.target = target;
        if let Some(d) = distance {
            self.distance = d.clamp(MIN_DISTANCE, MAX_DISTANCE);
        }
    }

    /// View-projection that maps absolute ENU metres to clip space, for passes that
    /// are not camera-relative (the mesh pass gets absolute block positions).
    pub fn absolute_view_proj(&self, aspect: f64) -> [[f32; 4]; 4] {
        let cam = self.camera();
        let relative = cam.view_proj(aspect).as_dmat4();
        let to_eye = glam::DMat4::from_translation(-cam.eye);
        (relative * to_eye).as_mat4().to_cols_array_2d()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_camera_looks_north_from_the_south() {
        let c = OrbitCamera::default();
        let e = c.eye();
        assert!(e[1] < 0.0 && e[2] > 0.0);
        assert!(e[0].abs() < 1e-9);
        let px = c.camera().project_px(c.target, 800, 600).unwrap();
        assert!((px[0] - 400.0).abs() < 1e-6 && (px[1] - 300.0).abs() < 1e-6);
    }

    #[test]
    fn pitch_is_clamped_and_moves_count() {
        let mut c = OrbitCamera::default();
        c.orbit(0.0, 10_000.0);
        assert!(c.pitch <= MAX_PITCH);
        c.orbit(0.0, -10_000.0);
        assert!(c.pitch >= MIN_PITCH);
        assert!(c.user_moved);
    }

    #[test]
    fn zoom_is_clamped() {
        let mut c = OrbitCamera::default();
        c.zoom(-1_000.0);
        assert_eq!(c.distance, MIN_DISTANCE);
        c.zoom(1_000.0);
        assert_eq!(c.distance, MAX_DISTANCE);
    }

    #[test]
    fn absolute_view_proj_matches_pointblitz_projection() {
        let mut c = OrbitCamera::default();
        c.frame([1200.0, -800.0, 40.0], Some(300.0));
        let vp = glam::Mat4::from_cols_array_2d(&c.absolute_view_proj(800.0 / 600.0));
        let p = glam::Vec4::new(1210.0, -790.0, 45.0, 1.0);
        let clip = vp * p;
        let ndc = clip / clip.w;
        let px = [(ndc.x as f64 + 1.0) / 2.0 * 800.0, (1.0 - ndc.y as f64) / 2.0 * 600.0];
        let reference = c.camera().project_px([1210.0, -790.0, 45.0], 800, 600).unwrap();
        assert!((px[0] - reference[0]).abs() < 0.05, "{px:?} vs {reference:?}");
        assert!((px[1] - reference[1]).abs() < 0.05, "{px:?} vs {reference:?}");
    }
}
