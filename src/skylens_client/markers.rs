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

/// Detections of the same kind closer than this (metres, ground plane) share one marker.
pub const MERGE_RADIUS: f64 = 3.0;

/// A detection as the marker layer sees it: `(id, feet, person, confidence)`, oldest first.
pub type Sighting = (String, DVec3, bool, f64);

/// Merges sightings of the same kind within [`MERGE_RADIUS`] into one marker each: the highest
/// confidence wins, a tie goes to the later one. Returns `(id, feet, person)` of the winners.
pub fn dedupe(sightings: &[Sighting]) -> Vec<(String, DVec3, bool)> {
    let mut kept: Vec<&Sighting> = Vec::new();
    for s in sightings {
        let near = kept
            .iter()
            .position(|k| k.2 == s.2 && (k.1.x - s.1.x).hypot(k.1.y - s.1.y) <= MERGE_RADIUS);
        match near {
            Some(i) if s.3 >= kept[i].3 => kept[i] = s,
            Some(_) => {}
            None => kept.push(s),
        }
    }
    kept.into_iter().map(|k| (k.0.clone(), k.1, k.2)).collect()
}

/// Size factor of the pins for a scene `extent` metres across: small scenes keep the legacy
/// size, big ones shrink it so a handful of pins never hides the view.
pub fn pin_scale(extent: f64) -> f64 {
    if extent.is_finite() { (extent * 0.002).clamp(0.25, 0.6) } else { 0.6 }
}

/// The marker colour of a detection kind.
pub fn marker_color(person: bool) -> u32 {
    if person { color::MARKER_PERSON } else { color::MARKER_DANGER }
}

/// Ring scale at board time `t` seconds for a marker at `east` (legacy `0.8 + 0.4 sin(3t + x)`).
pub fn pulse(t: f64, east: f64) -> f64 {
    0.8 + 0.4 * (t * 3.0 + east).sin()
}

/// The pin: a cone with its apex pointing down at the feet, centred `PIN_LIFT` above them.
pub fn pin_mesh(feet: DVec3, scale: f64) -> MeshData {
    let (radius, height) = (PIN_RADIUS * scale, PIN_HEIGHT * scale);
    let centre = feet + DVec3::Z * (PIN_LIFT * scale);
    let apex = centre - DVec3::Z * (height / 2.0);
    let base_z = centre.z + height / 2.0;
    let mut positions = vec![apex.to_array(), [centre.x, centre.y, base_z]];
    for i in 0..PIN_SEGMENTS {
        let a = i as f64 / PIN_SEGMENTS as f64 * std::f64::consts::TAU;
        positions.push([centre.x + radius * a.cos(), centre.y + radius * a.sin(), base_z]);
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

/// Adds an axis-aligned box from `lo` to `hi` to a mesh being built.
fn push_box(positions: &mut Vec<[f64; 3]>, indices: &mut Vec<u32>, lo: DVec3, hi: DVec3) {
    let base = positions.len() as u32;
    for k in 0..8u32 {
        let x = if k & 1 == 0 { lo.x } else { hi.x };
        let y = if k & 2 == 0 { lo.y } else { hi.y };
        let z = if k & 4 == 0 { lo.z } else { hi.z };
        positions.push([x, y, z]);
    }
    // Two triangles per face (both sides are drawn, so the winding only needs to be consistent).
    for f in [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]] {
        indices.extend([base + f[0], base + f[1], base + f[2], base + f[0], base + f[2], base + f[3]]);
    }
}

/// Adds a flat ring (rotor guard) around `centre` in the horizontal plane.
fn push_ring(positions: &mut Vec<[f64; 3]>, indices: &mut Vec<u32>, centre: DVec3, inner: f64, outer: f64) {
    const SEGMENTS: u32 = 16;
    let base = positions.len() as u32;
    for i in 0..SEGMENTS {
        let a = std::f64::consts::TAU * f64::from(i) / f64::from(SEGMENTS);
        let d = DVec3::new(a.cos(), a.sin(), 0.0);
        positions.push((centre + d * inner).to_array());
        positions.push((centre + d * outer).to_array());
    }
    for i in 0..SEGMENTS {
        let (j, k) = (base + 2 * i, base + 2 * ((i + 1) % SEGMENTS));
        indices.extend([j, j + 1, k + 1, j, k + 1, k]);
    }
}

/// A drone on the minimap: a diamond (octahedron of half-height `radius`, faceted) circled by a
/// level ring like Saturn's, so it reads as a drone rather than another marker, and a thin stem
/// down to `ground_z` (when finite and below) so its height reads at a glance.
pub fn drone_mesh(pos: DVec3, radius: f64, ground_z: f64) -> MeshData {
    let r = radius;
    let mut positions = vec![
        (pos + DVec3::X * r).to_array(),
        (pos + DVec3::Y * r).to_array(),
        (pos - DVec3::X * r).to_array(),
        (pos - DVec3::Y * r).to_array(),
        (pos + DVec3::Z * r).to_array(),
        (pos - DVec3::Z * r).to_array(),
    ];
    let mut indices = Vec::with_capacity(24);
    for i in 0..4u32 {
        let j = (i + 1) % 4;
        // Upper and lower faces, counter-clockwise seen from outside.
        indices.extend([i, j, 4]);
        indices.extend([j, i, 5]);
    }
    push_ring(&mut positions, &mut indices, pos, r * 1.45, r * 1.75);
    if ground_z.is_finite() && ground_z < pos.z - r {
        let s = r * 0.06;
        push_box(&mut positions, &mut indices, DVec3::new(pos.x - s, pos.y - s, ground_z), DVec3::new(pos.x + s, pos.y + s, pos.z - r));
    }
    MeshData { positions, indices, flat: true, ..Default::default() }
}

/// A marker drawn flat in full colour, unlit: on the minimap a lit pin a few pixels tall reads as
/// a dark speck against the height colours.
pub fn flat_material(rgb: u32) -> Material {
    Material { shading: Shading::Unlit, tint: config::rgba(rgb, 1.0), blend: Blend::Opaque, double_sided: true }
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
        let m = pin_mesh(feet, 1.0);
        let lowest = m.positions.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min);
        let highest = m.positions.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max);
        assert!((lowest - (3.0 + 8.5 - 0.8)).abs() < 1e-9);
        assert!((highest - (3.0 + 8.5 + 0.8)).abs() < 1e-9);
        assert_eq!(m.positions[0][..2], [10.0, 20.0]);
        assert_eq!(m.indices.len(), 12 * 6);
        assert!(m.indices.iter().all(|i| (*i as usize) < m.positions.len()));
    }

    #[test]
    fn nearby_same_kind_sightings_merge_to_the_most_confident() {
        let at = |x: f64| DVec3::new(x, 0.0, 0.0);
        let s = |id: &str, x: f64, person: bool, c: f64| (id.to_string(), at(x), person, c);
        let v = vec![
            s("a", 0.0, true, 0.6),
            s("b", 2.0, true, 0.9),
            s("c", 1.0, true, 0.7),
            s("d", 1.0, false, 0.5),
            s("e", 10.0, true, 0.4),
            s("f", 11.0, true, 0.4),
        ];
        let ids: Vec<String> = dedupe(&v).into_iter().map(|m| m.0).collect();
        assert_eq!(ids, ["b", "d", "f"]);
    }

    #[test]
    fn pins_shrink_with_the_scene() {
        assert!(pin_scale(100.0) < 1.0 && pin_scale(100.0) >= 0.25);
        assert_eq!(pin_scale(1.0e6), 0.6);
        let big = pin_mesh(DVec3::ZERO, 0.3);
        let h = |m: &MeshData| m.positions.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max);
        assert!(h(&big) < h(&pin_mesh(DVec3::ZERO, 1.0)));
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
    fn the_drone_is_a_ringed_diamond_with_a_stem_to_the_ground() {
        let c = DVec3::new(5.0, -3.0, 40.0);
        let m = drone_mesh(c, 2.0, 10.0);
        assert!(m.indices.iter().all(|i| (*i as usize) < m.positions.len()));
        // The diamond's faces point outward (its first 24 indices).
        for f in m.indices[..24].chunks_exact(3) {
            let [a, b, d] = [f[0], f[1], f[2]].map(|i| DVec3::from(m.positions[i as usize]));
            assert!((b - a).cross(d - a).dot((a + b + d) / 3.0 - c) > 0.0);
        }
        // The ring is level with the drone and wider than the diamond.
        let ring: Vec<DVec3> = m.positions[6..6 + 32].iter().map(|p| DVec3::from(*p)).collect();
        assert!(ring.iter().all(|p| (p.z - c.z).abs() < 1e-9));
        let widest = ring.iter().map(|p| (*p - c).length()).fold(0.0, f64::max);
        assert!((widest - 3.5).abs() < 1e-9);
        // The stem reaches the ground; without ground below there is none.
        assert_eq!(m.positions.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min), 10.0);
        let floating = drone_mesh(c, 2.0, f64::NEG_INFINITY);
        assert!(floating.positions.iter().all(|p| p[2] >= 38.0));
    }

    #[test]
    fn colours_follow_the_kind() {
        assert_eq!(marker_color(true), 0x39d98a);
        assert_eq!(marker_color(false), 0xff4d4d);
        assert_eq!(ring_material(true).tint[3], 0.8);
        assert_eq!(ring_material(false).blend, Blend::Alpha);
        assert_eq!(flat_material(marker_color(true)).tint, [0x39 as f32 / 255.0, 0xd9 as f32 / 255.0, 0x8a as f32 / 255.0, 1.0]);
    }
}
