//! Building footprints (`payload.v1.BuildingSet`) turned into the legacy looks
//! (`buildingSource.ts`, `towerViewer.ts`):
//!
//! - prisms: one wall quad per footprint edge plus a triangulated roof, per-vertex wall/roof
//!   colours, optional top-down UVs onto the cell's satellite image (실사 항공뷰);
//! - linework: roof outline, base outline and a few uprights per footprint (검정 텍스처 건물);
//! - points: the sampled point cloud of the 점 option (roof, wall and ground-ring samples from a
//!   seeded PRNG), coloured with the height ramp.
//!
//! Positions are scene metres: ENU with heights multiplied by [`EXAGGERATION`]. `unit` is the
//! number of metres one legacy scene unit stood for, used where the legacy code offset things by
//! fixed scene units (the base sunk 0.03 under the ground, the base line 0.035 above it).

use crate::terrain::{Dem, EXAGGERATION};
use skylens_shared::proto::payload::v1 as payload;
use skylens_viewer::pointblitz::core::MeshData;

/// Wall and roof colours baked per vertex (legacy `0x9aa2ab`, `0xdde2e7`).
pub const WALL_RGB: [u8; 3] = [0x9a, 0xa2, 0xab];
pub const ROOF_RGB: [u8; 3] = [0xdd, 0xe2, 0xe7];

/// The legacy point budget of the 점 option over the whole operating area.
pub const POINT_BUDGET: f64 = 60_000.0;

/// Seeded PRNG of the legacy layers (`mulberry32`), bit for bit.
#[derive(Debug, Clone)]
pub struct Mulberry32 {
    a: u32,
}

impl Mulberry32 {
    pub fn new(seed: u32) -> Mulberry32 {
        Mulberry32 { a: seed }
    }

    pub fn next_f64(&mut self) -> f64 {
        self.a = self.a.wrapping_add(0x6d2b_79f5);
        let a = self.a;
        let mut t = (a ^ (a >> 15)).wrapping_mul(1 | a);
        t = (t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t))) ^ t;
        f64::from(t ^ (t >> 14)) / 4_294_967_296.0
    }
}

/// The footprint ring as `[e, n]` pairs, as sent (closed: the first vertex repeated at the end).
pub fn ring(b: &payload::Building) -> Vec<[f64; 2]> {
    b.outline.chunks_exact(2).map(|p| [p[0], p[1]]).collect()
}

/// The ring without the closing duplicate (for triangulation and area).
pub fn open_ring(r: &[[f64; 2]]) -> &[[f64; 2]] {
    if r.len() > 1 && r[0] == r[r.len() - 1] { &r[..r.len() - 1] } else { r }
}

/// Shoelace area and perimeter in metres (legacy `ringAreaAndPerimeter`, closing edge included).
pub fn area_and_perimeter(r: &[[f64; 2]]) -> (f64, f64) {
    let mut area = 0.0;
    let mut perim = 0.0;
    for i in 0..r.len() {
        let [x1, y1] = r[i];
        let [x2, y2] = r[(i + 1) % r.len()];
        area += x1 * y2 - x2 * y1;
        perim += (x2 - x1).hypot(y2 - y1);
    }
    (area.abs() / 2.0, perim)
}

/// Even-odd point in polygon (legacy `pointInRing`).
pub fn point_in_ring(x: f64, y: f64, r: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let mut j = r.len().wrapping_sub(1);
    for i in 0..r.len() {
        let [xi, yi] = r[i];
        let [xj, yj] = r[j];
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// The legacy cartographic minimum height: buildings shorter than `min(12, span / 1200)` metres
/// are raised toward it, at most 8x (`buildingSource.ts`).
pub fn display_height(height_m: f64, span_m: f64) -> f64 {
    let min_h = (span_m / 1200.0).min(12.0);
    if height_m > 0.0 && height_m < min_h { height_m * (min_h / height_m).min(8.0) } else { height_m }
}

/// Ear-clipping triangulation of a simple polygon (either winding). Returns index triples into
/// `poly`; an empty result means the ring was degenerate (the walls still stand, as before).
pub fn triangulate(poly: &[[f64; 2]]) -> Vec<[usize; 3]> {
    let n = poly.len();
    if n < 3 {
        return Vec::new();
    }
    let signed: f64 = (0..n)
        .map(|i| {
            let [x1, y1] = poly[i];
            let [x2, y2] = poly[(i + 1) % n];
            x1 * y2 - x2 * y1
        })
        .sum();
    if signed.abs() < 1e-12 {
        return Vec::new();
    }
    let ccw = signed > 0.0;
    let mut idx: Vec<usize> = (0..n).collect();
    let mut out = Vec::with_capacity(n - 2);
    let cross = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    let mut guard = 0;
    while idx.len() > 3 && guard < n * n {
        guard += 1;
        let m = idx.len();
        let mut clipped = false;
        for k in 0..m {
            let (ia, ib, ic) = (idx[(k + m - 1) % m], idx[k], idx[(k + 1) % m]);
            let (a, b, c) = (poly[ia], poly[ib], poly[ic]);
            let turn = cross(a, b, c);
            let convex = if ccw { turn > 1e-12 } else { turn < -1e-12 };
            if !convex {
                continue;
            }
            let contains = idx.iter().any(|&j| {
                if j == ia || j == ib || j == ic {
                    return false;
                }
                let p = poly[j];
                let d1 = cross(a, b, p);
                let d2 = cross(b, c, p);
                let d3 = cross(c, a, p);
                if ccw { d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0 } else { d1 <= 0.0 && d2 <= 0.0 && d3 <= 0.0 }
            });
            if contains {
                continue;
            }
            out.push([ia, ib, ic]);
            idx.remove(k);
            clipped = true;
            break;
        }
        if !clipped {
            // Self-intersecting or collinear leftovers: drop a vertex rather than loop forever.
            let k = idx.iter().position(|_| true).unwrap_or(0);
            idx.remove(k);
        }
    }
    if idx.len() == 3 {
        out.push([idx[0], idx[1], idx[2]]);
    }
    out
}

/// Top-down projection of an ENU point onto a cell image (`[min_e, min_n, max_e, max_n]`, north
/// up), clamped like the legacy `uvAt`.
pub fn uv_at(rect: [f64; 4], e: f64, n: f64) -> [f32; 2] {
    let u = ((e - rect[0]) / (rect[2] - rect[0])).clamp(0.0, 1.0);
    let v = ((rect[3] - n) / (rect[3] - rect[1])).clamp(0.0, 1.0);
    [u as f32, v as f32]
}

/// Prisms and linework of one building set.
#[derive(Debug, Default, Clone)]
pub struct Prisms {
    pub mesh: MeshData,
    /// Line list: pairs of points.
    pub edges: Vec<[f64; 3]>,
    pub footprints: usize,
}

/// Extrudes a building set. `uv_rect` projects the cell's satellite image onto the prisms;
/// `z_offset` lowers the ring cells like their terrain (scene metres); `unit` is metres per
/// legacy scene unit; `span_m` the operating area span (minimum height rule).
pub fn extrude(set: &payload::BuildingSet, uv_rect: Option<[f64; 4]>, z_offset: f64, unit: f64, span_m: f64) -> Prisms {
    let mut positions: Vec<[f64; 3]> = Vec::new();
    let mut colors: Vec<[u8; 4]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut edges: Vec<[f64; 3]> = Vec::new();
    let wall = [WALL_RGB[0], WALL_RGB[1], WALL_RGB[2], 255];
    let roof = [ROOF_RGB[0], ROOF_RGB[1], ROOF_RGB[2], 255];
    let mut footprints = 0;

    let tri = |pts: [[f64; 3]; 3], c: [u8; 4], positions: &mut Vec<[f64; 3]>, colors: &mut Vec<[u8; 4]>, uvs: &mut Vec<[f32; 2]>| {
        for p in pts {
            positions.push(p);
            colors.push(c);
            if let Some(rect) = uv_rect {
                uvs.push(uv_at(rect, p[0], p[1]));
            }
        }
    };

    for b in &set.buildings {
        let outer = ring(b);
        if outer.len() < 3 {
            continue;
        }
        footprints += 1;
        let base_z = b.base * EXAGGERATION + z_offset;
        let h = display_height(b.height, span_m) * EXAGGERATION;
        let yb = base_z - 0.03 * unit;
        let yt = base_z + h;
        let stride = outer.len().div_ceil(4).max(1);
        for i in 0..outer.len() {
            let j = (i + 1) % outer.len();
            let [x1, y1] = outer[i];
            let [x2, y2] = outer[j];
            tri([[x1, y1, yb], [x2, y2, yb], [x2, y2, yt]], wall, &mut positions, &mut colors, &mut uvs);
            tri([[x1, y1, yb], [x2, y2, yt], [x1, y1, yt]], wall, &mut positions, &mut colors, &mut uvs);
            edges.push([x1, y1, yt]);
            edges.push([x2, y2, yt]);
            let lift = yb + 0.035 * unit;
            edges.push([x1, y1, lift]);
            edges.push([x2, y2, lift]);
            if i % stride == 0 {
                edges.push([x1, y1, yb]);
                edges.push([x1, y1, yt]);
            }
        }
        let open = open_ring(&outer);
        for [a, bb, c] in triangulate(open) {
            let p = |k: usize| [open[k][0], open[k][1], yt];
            tri([p(a), p(bb), p(c)], roof, &mut positions, &mut colors, &mut uvs);
        }
    }

    let indices = (0..positions.len() as u32).collect();
    let mesh = MeshData {
        positions,
        normals: None,
        colors: Some(colors),
        uvs: uv_rect.map(|_| uvs),
        indices,
        // Faces separate by shading: each triangle has its own vertices.
        flat: true,
    };
    Prisms { mesh, edges, footprints }
}

/// One sampled building point: scene position and the legacy brightness jitter (unused by the
/// tactical ramp, kept so the PRNG stream matches the legacy draw order).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampledPoint {
    pub pos: [f64; 3],
}

/// Samples the 점 point cloud over the given buildings (legacy `loadBuildings` point path).
/// `budget_per_weight` is points per unit of `perimeter * height + area`.
pub fn sample_points(buildings: &[&payload::Building], dem: &Dem, budget_per_weight: f64, span_m: f64, rng: &mut Mulberry32) -> Vec<SampledPoint> {
    let mut out = Vec::new();
    for b in buildings {
        let outer = ring(b);
        if outer.len() < 3 {
            continue;
        }
        let (area, perim_all) = area_and_perimeter(&outer);
        let weight = perim_all * b.height + area;
        let base_z = b.base * EXAGGERATION;
        let h = display_height(b.height, span_m) * EXAGGERATION;
        let n = ((budget_per_weight * weight).round() as i64).max(6) as usize;
        let (mut min_x, mut max_x, mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for [x, y] in &outer {
            min_x = min_x.min(*x);
            max_x = max_x.max(*x);
            min_y = min_y.min(*y);
            max_y = max_y.max(*y);
        }
        let n_roof = ((n as f64 * 0.35).floor() as usize).max(2);
        let n_base = ((n as f64 * 0.25).floor() as usize).max(2);
        let n_wall = n.saturating_sub(n_roof + n_base).max(2);

        let mut placed = 0;
        let mut tries = 0;
        while placed < n_roof && tries < n_roof * 8 {
            tries += 1;
            let x = min_x + rng.next_f64() * (max_x - min_x);
            let y = min_y + rng.next_f64() * (max_y - min_y);
            if !point_in_ring(x, y, &outer) {
                continue;
            }
            out.push(SampledPoint { pos: [x, y, base_z + h] });
            let _dim = rng.next_f64();
            placed += 1;
        }

        let lens: Vec<f64> = (0..outer.len())
            .map(|i| {
                let [x1, y1] = outer[i];
                let [x2, y2] = outer[(i + 1) % outer.len()];
                (x2 - x1).hypot(y2 - y1)
            })
            .collect();
        let perim: f64 = lens.iter().sum();
        let on_ring = |rng: &mut Mulberry32| -> [f64; 2] {
            let mut target = rng.next_f64() * perim;
            let mut edge = 0;
            while edge < lens.len() - 1 && target > lens[edge] {
                target -= lens[edge];
                edge += 1;
            }
            let t = if lens[edge] > 0.0 { target / lens[edge] } else { 0.0 };
            let [x1, y1] = outer[edge];
            let [x2, y2] = outer[(edge + 1) % outer.len()];
            [x1 + (x2 - x1) * t, y1 + (y2 - y1) * t]
        };
        for _ in 0..n_wall {
            let [x, y] = on_ring(rng);
            let z = base_z + rng.next_f64() * h;
            out.push(SampledPoint { pos: [x, y, z] });
            let _dim = rng.next_f64();
        }
        for _ in 0..n_base {
            let [x, y] = on_ring(rng);
            let z = dem.ground(x, y) * EXAGGERATION;
            out.push(SampledPoint { pos: [x, y, z] });
            let _dim = rng.next_f64();
        }
    }
    out
}

/// Total weight of a building list (the legacy budget split denominator).
pub fn total_weight(buildings: &[&payload::Building]) -> f64 {
    buildings
        .iter()
        .map(|b| {
            let r = ring(b);
            let (area, perim) = area_and_perimeter(&r);
            perim * b.height + area
        })
        .sum()
}

/// sRGB byte to linear light.
pub fn srgb_to_linear(c: u8) -> f64 {
    let c = f64::from(c) / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// Linear light to an sRGB byte.
pub fn linear_to_srgb(v: f64) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0).round().clamp(0.0, 255.0) as u8
}

/// The 점 height ramp (legacy `tacticalColors`): far colour at the lowest point, near colour at
/// the highest, mixed in linear light like three.js `Color.lerp`, written as sRGB bytes.
pub fn ramp_color(t: f64, far: u32, near: u32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let ch = |hex: u32, shift: u32| srgb_to_linear(((hex >> shift) & 0xff) as u8);
    let mut out = [0u8; 3];
    for (k, shift) in [16u32, 8, 0].into_iter().enumerate() {
        let a = ch(far, shift);
        let b = ch(near, shift);
        out[k] = linear_to_srgb(a + (b - a) * t);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(base: f64, height: f64) -> payload::Building {
        payload::Building {
            id: "b".into(),
            outline: vec![0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0, 0.0, 0.0],
            base,
            height,
        }
    }

    #[test]
    fn mulberry32_matches_the_javascript_sequence() {
        let mut r = Mulberry32::new(1);
        let a = r.next_f64();
        let b = r.next_f64();
        assert!((0.0..1.0).contains(&a) && (0.0..1.0).contains(&b) && a != b);
        let mut again = Mulberry32::new(1);
        assert_eq!(again.next_f64(), a);
        // Known value: mulberry32(1)() === 0.6270739405881613 in JavaScript.
        assert!((a - 0.627_073_940_588_161_3).abs() < 1e-15);
        assert!((b - 0.002_735_721_180_215_478).abs() < 1e-15);
    }

    #[test]
    fn ring_geometry() {
        let b = square(0.0, 9.0);
        let r = ring(&b);
        assert_eq!(r.len(), 5);
        assert_eq!(open_ring(&r).len(), 4);
        let (area, perim) = area_and_perimeter(&r);
        assert_eq!((area, perim), (100.0, 40.0));
        assert!(point_in_ring(5.0, 5.0, &r));
        assert!(!point_in_ring(15.0, 5.0, &r));
    }

    #[test]
    fn triangulates_both_windings_and_concave_rings() {
        let ccw = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        assert_eq!(triangulate(&ccw).len(), 2);
        let cw: Vec<[f64; 2]> = ccw.iter().rev().copied().collect();
        assert_eq!(triangulate(&cw).len(), 2);
        // An L shape: 6 vertices, 4 triangles, total area 75.
        let l = [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [5.0, 5.0], [5.0, 10.0], [0.0, 10.0]];
        let tris = triangulate(&l);
        assert_eq!(tris.len(), 4);
        let area: f64 = tris
            .iter()
            .map(|[a, b, c]| {
                let (a, b, c) = (l[*a], l[*b], l[*c]);
                ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs() / 2.0
            })
            .sum();
        assert!((area - 75.0).abs() < 1e-9);
        assert!(triangulate(&[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]).is_empty());
    }

    #[test]
    fn minimum_height_rule() {
        // 3 km span: minimum 2.5 m, so a 2 m shed is raised and a 6 m house is not.
        assert!((display_height(2.0, 3000.0) - 2.5).abs() < 1e-9);
        assert_eq!(display_height(6.0, 3000.0), 6.0);
        // Never more than 8x.
        assert!((display_height(0.5, 20_000.0) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn extrusion_counts_and_heights() {
        let set = payload::BuildingSet { cell: vec![0, 0], buildings: vec![square(2.0, 10.0)] };
        let p = extrude(&set, Some([0.0, 0.0, 400.0, 400.0]), 0.0, 70.0, 3000.0);
        // 5 ring vertices (closed) -> 5 walls of 2 triangles, plus 2 roof triangles.
        assert_eq!(p.mesh.positions.len(), (5 * 2 + 2) * 3);
        let top = p.mesh.positions.iter().map(|q| q[2]).fold(f64::NEG_INFINITY, f64::max);
        assert!((top - (2.0 + 10.0) * 1.5).abs() < 1e-9);
        let bottom = p.mesh.positions.iter().map(|q| q[2]).fold(f64::INFINITY, f64::min);
        assert!((bottom - (3.0 - 0.03 * 70.0)).abs() < 1e-9);
        assert_eq!(p.mesh.uvs.as_ref().unwrap().len(), p.mesh.positions.len());
        assert_eq!(p.edges.len() % 2, 0);
        assert_eq!(p.footprints, 1);
        let plain = extrude(&set, None, 0.0, 70.0, 3000.0);
        assert!(plain.mesh.uvs.is_none());
    }

    #[test]
    fn uv_projection_is_north_up_and_clamped() {
        let rect = [0.0, 0.0, 400.0, 400.0];
        assert_eq!(uv_at(rect, 0.0, 400.0), [0.0, 0.0]);
        assert_eq!(uv_at(rect, 400.0, 0.0), [1.0, 1.0]);
        assert_eq!(uv_at(rect, 500.0, -5.0), [1.0, 1.0]);
    }

    #[test]
    fn points_are_on_the_building() {
        let b = square(0.0, 10.0);
        let mut rng = Mulberry32::new(0xb17d_5eed);
        let pts = sample_points(&[&b], &Dem::new(), 0.05, 3000.0, &mut rng);
        assert!(pts.len() >= 6);
        for p in &pts {
            assert!((-1e-9..=10.0 + 1e-9).contains(&p.pos[0]) && (-1e-9..=10.0 + 1e-9).contains(&p.pos[1]));
            assert!(p.pos[2] >= -1e-9 && p.pos[2] <= 15.0 + 1e-9);
        }
    }

    #[test]
    fn ramp_ends_are_the_legacy_colours() {
        assert_eq!(ramp_color(0.0, 0x2a4a8a, 0x3fd0e0), [0x2a, 0x4a, 0x8a]);
        assert_eq!(ramp_color(1.0, 0x2a4a8a, 0x3fd0e0), [0x3f, 0xd0, 0xe0]);
        assert_eq!(linear_to_srgb(srgb_to_linear(0x80)), 0x80);
    }
}
