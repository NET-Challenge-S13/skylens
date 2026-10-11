//! Camera in ENU metres (decision 0013): x east, y north, z up.
//!
//! Rendering is camera-relative: the view matrix places the eye at the origin and every chunk is
//! shifted by `chunk.origin − eye`, computed in f64, so f32 precision is spent near the viewer.
//! Projection is reversed-Z with an infinite far plane (depth 1 at the near plane, 0 at infinity).

use glam::{DMat4, DVec3, DVec4, Mat4};

/// Right-handed view matrix for an eye at the origin looking along `dir`
/// (view space: x right, y up, −z forward).
fn look_dir_rh(dir: DVec3, up: DVec3) -> DMat4 {
    let f = dir.normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);
    DMat4::from_cols(
        DVec4::new(s.x, u.x, -f.x, 0.0),
        DVec4::new(s.y, u.y, -f.y, 0.0),
        DVec4::new(s.z, u.z, -f.z, 0.0),
        DVec4::new(0.0, 0.0, 0.0, 1.0),
    )
}

/// Reversed-Z, infinite far plane: depth = near / distance (1 at the near plane, → 0 far away).
fn perspective_infinite_reverse_rh(fov_y: f64, aspect: f64, near: f64) -> DMat4 {
    let f = 1.0 / (fov_y / 2.0).tan();
    DMat4::from_cols(
        DVec4::new(f / aspect, 0.0, 0.0, 0.0),
        DVec4::new(0.0, f, 0.0, 0.0),
        DVec4::new(0.0, 0.0, 0.0, -1.0),
        DVec4::new(0.0, 0.0, near, 0.0),
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    pub eye: DVec3,
    pub target: DVec3,
    pub up: DVec3,
    pub fov_y_deg: f64,
    pub near: f64,
}

impl Camera {
    pub fn new(eye: [f64; 3], target: [f64; 3], up: [f64; 3], fov_y_deg: f64) -> Self {
        Self {
            eye: eye.into(),
            target: target.into(),
            up: up.into(),
            fov_y_deg,
            near: 0.05,
        }
    }

    /// View (eye at origin) × projection, as f32 for the GPU.
    pub fn view_proj(&self, aspect: f64) -> Mat4 {
        let view = look_dir_rh(self.target - self.eye, self.up);
        let proj = perspective_infinite_reverse_rh(self.fov_y_deg.to_radians(), aspect, self.near);
        (proj * view).as_mat4()
    }

    /// Where an ENU point lands in pixels (x right, y down, pixel centres at i + 0.5), or None if
    /// it is behind the camera. CPU reference used by tests and culling checks.
    pub fn project_px(&self, p: [f64; 3], width: u32, height: u32) -> Option<[f64; 2]> {
        let aspect = f64::from(width) / f64::from(height);
        let view = look_dir_rh(self.target - self.eye, self.up);
        let proj = perspective_infinite_reverse_rh(self.fov_y_deg.to_radians(), aspect, self.near);
        let clip = proj * view * (DVec3::from(p) - self.eye).extend(1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some([
            (ndc.x + 1.0) / 2.0 * f64::from(width),
            (1.0 - ndc.y) / 2.0 * f64::from(height),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_projects_to_centre() {
        let c = Camera::new([0.0, -10.0, 5.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 50.0);
        let p = c.project_px([0.0, 0.0, 0.0], 1920, 1080).unwrap();
        assert!((p[0] - 960.0).abs() < 1e-9 && (p[1] - 540.0).abs() < 1e-9);
    }

    #[test]
    fn east_is_right_and_up_is_up_when_looking_north() {
        let c = Camera::new([0.0, -10.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 50.0);
        let east = c.project_px([1.0, 0.0, 0.0], 100, 100).unwrap();
        let up = c.project_px([0.0, 0.0, 1.0], 100, 100).unwrap();
        assert!(east[0] > 50.0);
        assert!(up[1] < 50.0);
    }

    #[test]
    fn behind_camera_is_none() {
        let c = Camera::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 50.0);
        assert_eq!(c.project_px([0.0, -1.0, 0.0], 100, 100), None);
    }

    #[test]
    fn far_coordinates_keep_precision() {
        // 5 km from the ENU origin, 1 cm apart: still distinct pixels when viewed from 1 m.
        let base = [5000.0, 5000.0, 100.0];
        let c = Camera::new(
            [base[0], base[1] - 1.0, base[2]],
            base,
            [0.0, 0.0, 1.0],
            50.0,
        );
        let a = c.project_px(base, 1920, 1080).unwrap();
        let b = c
            .project_px([base[0] + 0.01, base[1], base[2]], 1920, 1080)
            .unwrap();
        assert!(b[0] - a[0] > 10.0);
    }
}
