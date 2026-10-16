//! Detection markers (legacy `statusViewer.ts` `buildMarker`): a cone pin floating well above the
//! feet, point down, and a pulsing ring on the ground, green for people and red for hazards.
//!
//! The position is the feet. The pin hangs at 8.5 m (a cone at body height reads as stabbing the
//! person, and the reconstruction smears people taller than life); the ring lies 5 cm up.

use crate::glam::DVec3;
use crate::pointblitz::core::{Blend, Material, MeshData, Shading};
use skylens_viewer::config::{self, color};

/// Cone radius, height and segments (three.js `ConeGeometry(0.5, 1.6, 12)`).
const PIN_RADIUS: f64 = 0.5;
const PIN_HEIGHT: f64 = 1.6;
const PIN_SEGMENTS: usize = 12;
/// Height of the pin's centre above the feet.
pub const PIN_LIFT: f64 = 8.5;
/// Ring radii and segments (three.js `RingGeometry(0.8, 1.0, 32)`).
const RING_INNER: f64 = 0.8;
const RING_OUTER: f64 = 1.0;
const RING_SEGMENTS: usize = 32;
const RING_LIFT: f64 = 0.05;

/// The marker colour of a detection kind.
pub fn marker_color(person: bool) -> u32 {
    if person { color::MARKER_PERSON } else { color::MARKER_DANGER }
}

/// Ring scale at board time `t` seconds for a marker at `east` (legacy `0.8 + 0.4 sin(3t + x)`).
pub fn pulse(t: f64, east: f64) -> f64 {
    0.8 + 0.4 * (t * 3.0 + east).sin()
}

/// The pin: a cone with its apex pointing down at the feet, centred `PIN_LIFT` above them.
pub fn pin_mesh(feet: DVec3) -> MeshData {
    let centre = feet + DVec3::Z * PIN_LIFT;
    let apex = centre - DVec3::Z * (PIN_HEIGHT / 2.0);
    let base_z = centre.z + PIN_HEIGHT / 2.0;
    let mut positions = vec![apex.to_array(), [centre.x, centre.y, base_z]];
    for i in 0..PIN_SEGMENTS {
        let a = i as f64 / PIN_SEGMENTS as f64 * std::f64::consts::TAU;
        positions.push([centre.x + PIN_RADIUS * a.cos(), centre.y + PIN_RADIUS * a.sin(), base_z]);
    }
    let mut indices = Vec::new();
    for i in 0..PIN_SEGMENTS as u32 {
        let a = 2 + i;
        let b = 2 + (i + 1) % PIN_SEGMENTS as u32;
        // Side (outward, counter-clockwise seen from outside) and the top cap.
        indices.extend([0, b, a]);
        indices.extend([1, a, b]);
    }
    MeshData { positions, indices, flat: false, ..Default::default() }
}

/// The ground ring at `scale`.
pub fn ring_mesh(feet: DVec3, scale: f64) -> MeshData {
    let z = feet.z + RING_LIFT;
    let mut positions = Vec::with_capacity(RING_SEGMENTS * 2);
    for i in 0..RING_SEGMENTS {
        let a = i as f64 / RING_SEGMENTS as f64 * std::f64::consts::TAU;
        let (s, c) = a.sin_cos();
        positions.push([feet.x + RING_INNER * scale * c, feet.y + RING_INNER * scale * s, z]);
        positions.push([feet.x + RING_OUTER * scale * c, feet.y + RING_OUTER * scale * s, z]);
    }
    let n = RING_SEGMENTS as u32;
    let mut indices = Vec::with_capacity(RING_SEGMENTS * 6);
    for i in 0..n {
        let (i0, o0) = (2 * i, 2 * i + 1);
        let (i1, o1) = (2 * ((i + 1) % n), 2 * ((i + 1) % n) + 1);
        indices.extend([i0, o0, o1, i0, o1, i1]);
    }
    MeshData { positions, indices, flat: true, normals: Some(vec![[0.0, 0.0, 1.0]; RING_SEGMENTS * 2]), ..Default::default() }
}

/// The pin's look: lit like the legacy `MeshStandardMaterial` with the colour as emissive at 0.6.
pub fn pin_material(person: bool) -> Material {
    let c = config::rgb(marker_color(person));
    let mut m = crate::scene::lambert(1.0);
    if let Shading::Lambert { emissive, .. } = &mut m.shading {
        *emissive = c.map(|v| v * 0.6);
    }
    m.tint = [c[0], c[1], c[2], 1.0];
    m.double_sided = false;
    m
}

/// The ring's look: flat colour at 0.8 opacity, both sides (legacy `MeshBasicMaterial`).
pub fn ring_material(person: bool) -> Material {
    Material {
        shading: Shading::Unlit,
        tint: config::rgba(marker_color(person), 0.8),
        blend: Blend::Alpha,
        double_sided: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pin_hangs_point_down_above_the_feet() {
        let feet = DVec3::new(10.0, 20.0, 3.0);
        let m = pin_mesh(feet);
        let lowest = m.positions.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min);
        let highest = m.positions.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max);
        assert!((lowest - (3.0 + 8.5 - 0.8)).abs() < 1e-9);
        assert!((highest - (3.0 + 8.5 + 0.8)).abs() < 1e-9);
        assert_eq!(m.positions[0][..2], [10.0, 20.0]);
        assert_eq!(m.indices.len(), 12 * 6);
        assert!(m.indices.iter().all(|i| (*i as usize) < m.positions.len()));
    }

    #[test]
    fn the_ring_pulses_on_the_ground() {
        let feet = DVec3::new(0.0, 0.0, 1.0);
        let m = ring_mesh(feet, 1.2);
        assert!(m.positions.iter().all(|p| (p[2] - 1.05).abs() < 1e-12));
        let r = (m.positions[1][0].powi(2) + m.positions[1][1].powi(2)).sqrt();
        assert!((r - 1.2).abs() < 1e-9);
        assert!(m.indices.iter().all(|i| (*i as usize) < m.positions.len()));
        assert!((pulse(0.0, 0.0) - 0.8).abs() < 1e-12);
        assert!(pulse(0.5, 0.0) > 0.8 && pulse(0.5, 0.0) <= 1.2);
    }

    #[test]
    fn colours_follow_the_kind() {
        assert_eq!(marker_color(true), 0x39d98a);
        assert_eq!(marker_color(false), 0xff4d4d);
        assert_eq!(ring_material(true).tint[3], 0.8);
        assert_eq!(ring_material(false).blend, Blend::Alpha);
    }
}
