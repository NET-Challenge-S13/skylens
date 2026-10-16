//! The animated point waves of the live display, painted as vector shapes (no second GPU pass):
//! the waiting screen's wave field over the 3D view and the monitor panel's small dot wave.
//!
//! Both are ports of the reference viewer's wave heights (`waitMat` vertex shader and
//! `drawMonitor`), in the SkyLens blues. Blitz paints no additive blending, so the dots are plain
//! alpha rects.

use crate::anyrender::{self, PaintScene as _};
use crate::peniko::kurbo::{Affine, Rect};
use crate::peniko::{Color, Fill};
use blitz_dom::Widget;
use blitz_dom::node::ComputedStyles;

/// Grid of the waiting field (the reference uses 190 x 190 WebGL points; shapes cost more).
pub const WAIT_GRID: usize = 84;
/// Half extent of the waiting field.
const WAIT_EXT: f64 = 1.3;

/// Wave height of the waiting field at `(x, y)` and time `t`.
pub fn wait_height(x: f64, y: f64, t: f64) -> f64 {
    let r = x.hypot(y);
    0.11 * (r * 15.0 - t * 2.4).sin() / (1.0 + r * 2.2) + 0.05 * (r * 6.5 - t * 1.2 + 1.3).sin() + 0.035 * (x * 6.0 + t * 0.9).sin() * (y * 5.0 - t * 0.7).sin()
}

/// Height to a 0..1 ramp position.
pub fn height_k(h: f64) -> f64 {
    (h * 4.5 + 0.45).clamp(0.0, 1.0)
}

/// The SkyLens blue ramp: deep blue, the board's info blue, near white.
pub fn ramp(k: f64) -> [f64; 3] {
    let (lo, mid, hi) = ([0.05, 0.28, 0.85], [0.0, 0.78, 1.0], [0.85, 1.0, 1.0]);
    let mix = |a: [f64; 3], b: [f64; 3], f: f64| [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
    if k < 0.6 { mix(lo, mid, k / 0.6) } else { mix(mid, hi, (k - 0.6) / 0.4) }
}

/// Deterministic jitter in 0..1 for grid cell `(i, j)`.
fn jitter(i: usize, j: usize, salt: u32) -> f64 {
    let mut h = (i as u32).wrapping_mul(73856093) ^ (j as u32).wrapping_mul(19349663) ^ salt.wrapping_mul(83492791);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    f64::from(h & 0xffff) / 65535.0
}

/// A camera orbiting the field: position, and the basis (right, up, forward).
struct Cam {
    eye: [f64; 3],
    right: [f64; 3],
    up: [f64; 3],
    fwd: [f64; 3],
}

fn norm(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn wait_cam(t: f64) -> Cam {
    let a = t * 0.07;
    let eye = [a.sin() * 2.3, -a.cos() * 2.3, 1.15 + 0.12 * (t * 0.2).sin()];
    let fwd = norm([-eye[0], -eye[1], -0.05 - eye[2]]);
    let right = norm(cross(fwd, [0.0, 0.0, 1.0]));
    let up = cross(right, fwd);
    Cam { eye, right, up, fwd }
}

/// Screen position (pixels), depth and ramp position of the field point `(x, y)`, or None when
/// it is behind the camera.
fn project_wait(cam: &Cam, x: f64, y: f64, t: f64, w: f64, h: f64) -> Option<(f64, f64, f64, f64)> {
    let height = wait_height(x, y, t);
    let rel = [x - cam.eye[0], y - cam.eye[1], height - cam.eye[2]];
    let depth = dot(rel, cam.fwd);
    if depth < 0.1 {
        return None;
    }
    let focal = h / 2.0 / (25f64).to_radians().tan();
    Some((w / 2.0 + dot(rel, cam.right) / depth * focal, h / 2.0 - dot(rel, cam.up) / depth * focal, depth, height_k(height)))
}

fn color(rgb: [f64; 3], a: f64) -> Color {
    Color::new([rgb[0] as f32, rgb[1] as f32, rgb[2] as f32, a.clamp(0.0, 1.0) as f32])
}

/// Paints the waiting wave over the whole view. `w`, `h` are physical pixels, `scale` the device
/// pixel ratio, `t` seconds since the board opened, `opacity` the fade (1 full).
pub fn paint_wait(scene: &mut anyrender::Scene, w: f64, h: f64, scale: f64, t: f64, opacity: f64) {
    if opacity <= 0.0 || w < 1.0 || h < 1.0 {
        return;
    }
    let cam = wait_cam(t);
    let px = scale.min(2.0) * (h / scale / 520.0).max(1.0) * 2.2 * 2.2;
    let pulse = 0.85 + 0.15 * (t * 1.7).sin();
    let n = WAIT_GRID;
    for i in 0..n {
        for j in 0..n {
            let x = ((i as f64 + jitter(i, j, 1) * 0.5) / (n - 1) as f64 - 0.5) * 2.0 * WAIT_EXT;
            let y = ((j as f64 + jitter(i, j, 2) * 0.5) / (n - 1) as f64 - 0.5) * 2.0 * WAIT_EXT;
            let r = x.hypot(y);
            let edge = 1.0 - smoothstep(0.75, 1.25, r);
            if edge <= 0.01 {
                continue;
            }
            let Some((sx, sy, depth, k)) = project_wait(&cam, x, y, t, w, h) else { continue };
            let size = px * (0.8 + 1.3 * k) * 2.3 / depth;
            if sx < -size || sy < -size || sx > w + size || sy > h + size {
                continue;
            }
            let c = ramp(k);
            let alpha = (0.35 + 0.65 * k) * edge * opacity * 0.9 * pulse;
            let half = size / 2.0;
            scene.fill(
                Fill::NonZero,
                Affine::IDENTITY,
                color([c[0] * pulse, c[1] * pulse, c[2] * pulse], alpha),
                None,
                &Rect::new(sx - half, sy - half, sx + half, sy + half),
            );
        }
    }
}

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------------------------
// Monitor
// ---------------------------------------------------------------------------------------------

const MON_COLS: usize = 50;
const MON_ROWS: usize = 30;

/// Wave height of the monitor dots at `(x, y)` in -1..1; zero when `calm`.
pub fn monitor_height(x: f64, y: f64, t: f64, calm: bool) -> f64 {
    if calm {
        return 0.0;
    }
    let r = x.hypot(y);
    0.16 * (r * 9.0 - t * 3.2).sin() / (1.0 + r * 2.0) + 0.05 * (r * 4.0 - t * 1.4 + 1.0).sin()
}

/// The monitor's dot wave on `canvas#live-monitor-canvas`; reads the clock itself.
pub struct MonitorWave {
    calm: std::rc::Rc<std::cell::Cell<bool>>,
}

impl MonitorWave {
    pub fn boxed(calm: std::rc::Rc<std::cell::Cell<bool>>) -> Box<dyn Widget> {
        Box::new(MonitorWave { calm })
    }

    fn draw(&self, scene: &mut anyrender::Scene, w: f64, h: f64, scale: f64) {
        let t = skylens_viewer::now_ms() / 1000.0;
        let calm = self.calm.get();
        let (ca, sa) = ((t * 0.18).cos(), (t * 0.18).sin());
        let (cx, cy) = (w / 2.0, h * 0.66);
        let sx_k = w / 220.0 * 104.0;
        let sy_k = h / 140.0 * 26.0;
        let sz_k = h / 140.0;
        let tf = Affine::scale(scale);
        for i in 0..MON_COLS {
            for j in 0..MON_ROWS {
                let x = (i as f64 / (MON_COLS - 1) as f64 - 0.5) * 2.0;
                let y = (j as f64 / (MON_ROWS - 1) as f64 - 0.5) * 2.0;
                let hh = monitor_height(x, y, t, calm);
                let k = height_k(hh);
                let (xr, yr) = (x * ca - y * sa, x * sa + y * ca);
                let sc = 1.0 / (1.0 + yr * 0.22);
                let px = cx + xr * sc * sx_k;
                let py = cy + yr * sc * sy_k - hh * sc * 80.0 * sz_k;
                let s = (1.7 * sc + 0.3) * sz_k;
                let c = ramp(k);
                scene.fill(Fill::NonZero, tf, color(c, 0.35 + 0.65 * k), None, &Rect::new(px - 0.8, py - 0.8, px - 0.8 + s, py - 0.8 + s));
            }
        }
    }
}

impl Widget for MonitorWave {
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
    fn the_calm_monitor_is_flat() {
        assert_eq!(monitor_height(0.3, -0.2, 5.0, true), 0.0);
        assert!(monitor_height(0.3, -0.2, 5.0, false).abs() > 0.0);
    }

    #[test]
    fn heights_map_into_the_ramp() {
        assert_eq!(height_k(-10.0), 0.0);
        assert_eq!(height_k(10.0), 1.0);
        let lo = ramp(0.0);
        let hi = ramp(1.0);
        assert!(lo[2] > lo[0] && hi[0] > 0.8);
        // The ramp is continuous at its knee.
        let a = ramp(0.6 - 1e-9);
        let b = ramp(0.6 + 1e-9);
        assert!((a[1] - b[1]).abs() < 1e-6);
    }

    #[test]
    fn the_wait_field_is_in_front_of_the_camera_and_on_screen() {
        let cam = wait_cam(0.0);
        let (sx, sy, depth, k) = project_wait(&cam, 0.0, 0.0, 0.0, 1500.0, 950.0).expect("centre is visible");
        assert!(depth > 1.0 && depth < 4.0);
        assert!((0.0..=1.0).contains(&k));
        assert!(sx > 600.0 && sx < 900.0, "{sx}");
        assert!(sy > 100.0 && sy < 850.0, "{sy}");
        // A point behind the eye is dropped.
        assert!(project_wait(&cam, cam.eye[0] * 4.0, cam.eye[1] * 4.0, 0.0, 1500.0, 950.0).is_none());
    }

    #[test]
    fn painting_adds_shapes_only_while_visible() {
        let mut s = anyrender::Scene::new();
        paint_wait(&mut s, 800.0, 600.0, 1.0, 1.0, 0.0);
        assert_eq!(s.commands.len(), 0);
        paint_wait(&mut s, 800.0, 600.0, 1.0, 1.0, 1.0);
        assert!(!s.commands.is_empty());
    }
}
