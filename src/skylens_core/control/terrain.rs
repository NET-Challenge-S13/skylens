//! Terrain tiles (`payload.v1.TerrainTile`) as the tower draws and samples them.
//!
//! - [`Dem`]: bilinear ground height under any ENU point, from the tiles received so far
//!   (route planner altitudes, route pads, drone ground clamp, building points).
//! - [`grade`]: the legacy cold grade of the satellite image for the tactical display modes
//!   (`terrainSource.ts` `applyCyberGrade`): a luminance LUT into a navy/teal ramp and a faint
//!   graticule every 250 m / 1 km.
//! - [`terrain_mesh`]: the height grid as a textured mesh, vertically exaggerated like the legacy
//!   scene.
//!
//! Scene coordinates are ENU metres with the vertical multiplied by [`EXAGGERATION`] (legacy
//! `EXAGGERATION = 1.5`, applied to terrain, buildings, drones and the route alike).

use skylens_shared::proto::payload::v1 as payload;
use skylens_viewer::pointblitz::core::{MeshData, TextureImage};
use std::collections::HashMap;
use std::rc::Rc;

/// Legacy vertical exaggeration ("real relief reads nearly flat at map scale").
pub const EXAGGERATION: f64 = 1.5;

/// Scene height of a height in metres above the anchor altitude.
pub fn scene_z(height_m: f64) -> f64 {
    height_m * EXAGGERATION
}

// ---------------------------------------------------------------------------------------------
// DEM sampling
// ---------------------------------------------------------------------------------------------

/// Bilinear height of one tile at an ENU point inside it (metres above the anchor altitude).
pub fn tile_height_at(tile: &payload::TerrainTile, e: f64, n: f64) -> Option<f64> {
    let (cols, rows) = (tile.cols as usize, tile.rows as usize);
    if cols < 2 || rows < 2 || tile.heights.len() < cols * rows {
        return None;
    }
    let (w, h) = (tile.max_e - tile.min_e, tile.max_n - tile.min_n);
    if !(w > 0.0 && h > 0.0) || e < tile.min_e || e > tile.max_e || n < tile.min_n || n > tile.max_n {
        return None;
    }
    let gx = (e - tile.min_e) / w * (cols - 1) as f64;
    let gy = (n - tile.min_n) / h * (rows - 1) as f64;
    let x0 = (gx.floor() as usize).min(cols - 2);
    let y0 = (gy.floor() as usize).min(rows - 2);
    let fx = gx - x0 as f64;
    let fy = gy - y0 as f64;
    let at = |x: usize, y: usize| f64::from(tile.heights[y * cols + x]);
    let a = at(x0, y0);
    let b = at(x0 + 1, y0);
    let c = at(x0, y0 + 1);
    let d = at(x0 + 1, y0 + 1);
    Some(a * (1.0 - fx) * (1.0 - fy) + b * fx * (1.0 - fy) + c * (1.0 - fx) * fy + d * fx * fy)
}

/// Ground heights from every terrain tile received.
#[derive(Debug, Default, Clone)]
pub struct Dem {
    tiles: HashMap<[i32; 2], Rc<payload::TerrainTile>>,
    /// Cell edge in metres, from the first tile (the core's grid is regular).
    cell_m: Option<f64>,
    /// Lowest height of the fine tiles: the legacy answered `minE` outside its grid.
    floor: Option<f64>,
}

impl Dem {
    pub fn new() -> Dem {
        Dem::default()
    }

    /// Rebuilds from the stored tiles.
    pub fn rebuild<'a>(&mut self, tiles: impl Iterator<Item = ([i32; 2], &'a Rc<payload::TerrainTile>)>) {
        self.tiles.clear();
        self.cell_m = None;
        self.floor = None;
        for (cell, tile) in tiles {
            if self.cell_m.is_none() && tile.max_e > tile.min_e {
                self.cell_m = Some(tile.max_e - tile.min_e);
            }
            if !tile.ring {
                if let Some(lo) = tile.heights.iter().copied().map(f64::from).filter(|h| h.is_finite()).reduce(f64::min) {
                    self.floor = Some(self.floor.map_or(lo, |f| f.min(lo)));
                }
            }
            self.tiles.insert(cell, tile.clone());
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// Height under (e, n) where a tile covers it.
    pub fn height_at(&self, e: f64, n: f64) -> Option<f64> {
        if let Some(size) = self.cell_m {
            let cell = [(e / size).floor() as i32, (n / size).floor() as i32];
            if let Some(h) = self.tiles.get(&cell).and_then(|t| tile_height_at(t, e, n)) {
                return Some(h);
            }
        }
        // A point on a shared edge, or a grid that is not the expected one.
        self.tiles.values().find_map(|t| tile_height_at(t, e, n))
    }

    /// Height under (e, n), or the legacy fallback (the lowest ground of the area, else 0 =
    /// anchor altitude) where nothing is loaded.
    pub fn ground(&self, e: f64, n: f64) -> f64 {
        self.height_at(e, n).unwrap_or_else(|| self.floor.unwrap_or(0.0))
    }
}

// ---------------------------------------------------------------------------------------------
// Cold grade (tactical drape)
// ---------------------------------------------------------------------------------------------

/// Luminance to navy/teal ramp with the legacy midtone lift baked in (`cyberLut`).
pub fn cyber_lut() -> [u8; 256 * 3] {
    const STOPS: [(f64, [f64; 3]); 5] = [
        (0.0, [4.0, 9.0, 18.0]),
        (0.38, [8.0, 27.0, 42.0]),
        (0.7, [15.0, 57.0, 72.0]),
        (0.9, [30.0, 87.0, 104.0]),
        (1.0, [58.0, 126.0, 142.0]),
    ];
    let mut lut = [0u8; 256 * 3];
    for i in 0..256 {
        let t = (i as f64 / 255.0).powf(0.88);
        let mut k = 0;
        while k < STOPS.len() - 2 && t > STOPS[k + 1].0 {
            k += 1;
        }
        let (t0, c0) = STOPS[k];
        let (t1, c1) = STOPS[k + 1];
        let f = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
        for ch in 0..3 {
            lut[i * 3 + ch] = clamped_u8(c0[ch] + (c1[ch] - c0[ch]) * f);
        }
    }
    lut
}

/// `Uint8ClampedArray` assignment: clamp, then round half to even.
fn clamped_u8(v: f64) -> u8 {
    let v = v.clamp(0.0, 255.0);
    let r = v.round();
    let r = if (v - v.floor() - 0.5).abs() < 1e-12 && r % 2.0 != 0.0 { r - 1.0 } else { r };
    r as u8
}

/// Rec. 601-ish luminance the legacy grade used, in 0..255.
fn luminance(r: u8, g: u8, b: u8) -> usize {
    ((u32::from(r) * 54 + u32::from(g) * 183 + u32::from(b) * 19) >> 8).min(255) as usize
}

/// Graticule spacing in metres for an image of `m_per_px` (legacy: at least 48 px, 250 m minor,
/// 1 km major, both measured on the ground).
pub fn graticule_steps(m_per_px: f64) -> (f64, f64) {
    let minor_px = (250.0 / m_per_px).round().max(48.0);
    let major_px = (1000.0 / m_per_px).round().max(minor_px * 2.0);
    (minor_px * m_per_px, major_px * m_per_px)
}

/// Canvas `source-over` of one colour with alpha onto an sRGB byte (the 2D canvas blends the
/// encoded values).
fn over(dst: u8, src: u8, alpha: f64) -> u8 {
    (f64::from(src) * alpha + f64::from(dst) * (1.0 - alpha)).round().clamp(0.0, 255.0) as u8
}

/// Grades an RGBA image of a tile in place. `min_e` / `max_n` are the image's west and north edges
/// in ENU metres and `m_per_px` its ground resolution; graticule lines sit on ENU multiples of the
/// step so neighbouring tiles line up.
pub fn grade(rgba: &mut [u8], width: u32, height: u32, min_e: f64, max_n: f64, m_per_px_x: f64, m_per_px_y: f64) {
    let lut = cyber_lut();
    for px in rgba.chunks_exact_mut(4) {
        let l = luminance(px[0], px[1], px[2]);
        px[0] = lut[l * 3];
        px[1] = lut[l * 3 + 1];
        px[2] = lut[l * 3 + 2];
    }
    let (w, h) = (width as usize, height as usize);
    let mpp = (m_per_px_x + m_per_px_y) / 2.0;
    let (minor, major) = graticule_steps(mpp);
    let mut draw = |step: f64, color: [u8; 3], alpha: f64| {
        // Columns whose pixel holds a multiple of `step` (east), rows likewise (north).
        let mut cols = Vec::new();
        let first = (min_e / step).ceil();
        let mut k = first;
        loop {
            let x = ((k * step - min_e) / m_per_px_x).floor();
            if x >= w as f64 {
                break;
            }
            if x >= 0.0 {
                cols.push(x as usize);
            }
            k += 1.0;
        }
        let mut rows = Vec::new();
        let mut k = (max_n / step).floor();
        loop {
            let y = ((max_n - k * step) / m_per_px_y).floor();
            if y >= h as f64 {
                break;
            }
            if y >= 0.0 {
                rows.push(y as usize);
            }
            k -= 1.0;
        }
        // One stroke per pass: a pixel where a column and a row cross is painted once, as the
        // canvas path does.
        let mut painted = vec![false; w * h];
        for &x in &cols {
            for y in 0..h {
                painted[y * w + x] = true;
            }
        }
        for &y in &rows {
            for x in 0..w {
                painted[y * w + x] = true;
            }
        }
        for (i, p) in painted.iter().enumerate() {
            if *p {
                for ch in 0..3 {
                    rgba[i * 4 + ch] = over(rgba[i * 4 + ch], color[ch], alpha);
                }
            }
        }
    };
    draw(minor, [92, 190, 208], 0.035);
    draw(major, [102, 210, 226], 0.085);
}

/// Decodes a tile's JPEG to RGBA, or None when it is empty or undecodable.
pub fn decode_jpeg(bytes: &[u8]) -> Option<TextureImage> {
    if bytes.is_empty() {
        return None;
    }
    let img = image::load_from_memory_with_format(bytes, image::ImageFormat::Jpeg).ok()?.into_rgba8();
    let (width, height) = img.dimensions();
    Some(TextureImage { width, height, rgba: img.into_raw() })
}

/// The tile's image, raw (`graded == false`) or graded for the tactical modes.
pub fn tile_texture(tile: &payload::TerrainTile, graded: bool) -> Option<TextureImage> {
    let mut img = decode_jpeg(&tile.imagery_jpeg)?;
    if graded {
        let mx = (tile.max_e - tile.min_e) / f64::from(img.width.max(1));
        let my = (tile.max_n - tile.min_n) / f64::from(img.height.max(1));
        grade(&mut img.rgba, img.width, img.height, tile.min_e, tile.max_n, mx, my);
    }
    Some(img)
}

// ---------------------------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------------------------

/// The height grid as a mesh in scene coordinates, UVs over the north-up image. `z_offset` is in
/// scene metres (the ring sits a little lower so its seam hides under the fine tiles).
pub fn terrain_mesh(tile: &payload::TerrainTile, z_offset: f64) -> Option<MeshData> {
    let (cols, rows) = (tile.cols as usize, tile.rows as usize);
    if cols < 2 || rows < 2 || tile.heights.len() < cols * rows {
        return None;
    }
    let (w, h) = (tile.max_e - tile.min_e, tile.max_n - tile.min_n);
    let mut positions = Vec::with_capacity(cols * rows);
    let mut uvs = Vec::with_capacity(cols * rows);
    for j in 0..rows {
        let fy = j as f64 / (rows - 1) as f64;
        for i in 0..cols {
            let fx = i as f64 / (cols - 1) as f64;
            let hgt = f64::from(tile.heights[j * cols + i]);
            positions.push([tile.min_e + fx * w, tile.min_n + fy * h, scene_z(hgt) + z_offset]);
            // Row 0 is the south edge; image row 0 is north.
            uvs.push([fx as f32, (1.0 - fy) as f32]);
        }
    }
    let mut indices = Vec::with_capacity((cols - 1) * (rows - 1) * 6);
    for j in 0..rows - 1 {
        for i in 0..cols - 1 {
            let a = (j * cols + i) as u32;
            let b = a + 1;
            let c = a + cols as u32;
            let d = c + 1;
            // Counter-clockwise seen from above (+z).
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    Some(MeshData { positions, normals: None, colors: None, uvs: Some(uvs), indices, flat: false })
}

/// Scene-space bounds of a tile's surface: `(min, max)`.
pub fn tile_bounds(tile: &payload::TerrainTile) -> ([f64; 3], [f64; 3]) {
    let (lo, hi) = tile
        .heights
        .iter()
        .map(|h| f64::from(*h))
        .filter(|h| h.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), h| (lo.min(h), hi.max(h)));
    let (lo, hi) = if lo.is_finite() { (lo, hi) } else { (0.0, 0.0) };
    ([tile.min_e, tile.min_n, scene_z(lo)], [tile.max_e, tile.max_n, scene_z(hi)])
}

/// Cells of the core's fixed grid touching the operating area (Go `cellRange`), so the loading
/// screen knows how many fine tiles make up the area.
pub fn core_cell_range(min: [f64; 2], max: [f64; 2], cell_m: f64) -> ([i32; 2], [i32; 2]) {
    let x0 = (min[0] / cell_m).floor() as i32;
    let y0 = (min[1] / cell_m).floor() as i32;
    let x1 = x0.max((max[0] / cell_m).ceil() as i32 - 1);
    let y1 = y0.max((max[1] / cell_m).ceil() as i32 - 1);
    ([x0, y0], [x1, y1])
}

/// The core's terrain cell edge (Go `terrain.CellSizeM`).
pub const CORE_CELL_M: f64 = 400.0;

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(cell: [i32; 2], heights: Vec<f32>, cols: u32, rows: u32) -> payload::TerrainTile {
        payload::TerrainTile {
            cell: cell.to_vec(),
            min_e: f64::from(cell[0]) * 400.0,
            min_n: f64::from(cell[1]) * 400.0,
            max_e: f64::from(cell[0] + 1) * 400.0,
            max_n: f64::from(cell[1] + 1) * 400.0,
            cols,
            rows,
            heights,
            imagery_jpeg: vec![],
            ring: false,
        }
    }

    #[test]
    fn bilinear_heights_follow_row_zero_south() {
        // Row 0 (south) is 0 m, row 1 (north) is 10 m.
        let t = tile([0, 0], vec![0.0, 0.0, 10.0, 10.0], 2, 2);
        assert_eq!(tile_height_at(&t, 0.0, 0.0), Some(0.0));
        assert_eq!(tile_height_at(&t, 200.0, 400.0), Some(10.0));
        assert!((tile_height_at(&t, 100.0, 100.0).unwrap() - 2.5).abs() < 1e-9);
        assert_eq!(tile_height_at(&t, -1.0, 0.0), None);
    }

    #[test]
    fn dem_finds_the_cell_and_falls_back_to_the_floor() {
        let a = Rc::new(tile([0, 0], vec![5.0; 4], 2, 2));
        let b = Rc::new(tile([-1, 0], vec![20.0; 4], 2, 2));
        let mut dem = Dem::new();
        dem.rebuild([([0, 0], &a), ([-1, 0], &b)].into_iter());
        assert_eq!(dem.height_at(10.0, 10.0), Some(5.0));
        assert_eq!(dem.height_at(-10.0, 10.0), Some(20.0));
        assert_eq!(dem.height_at(10.0, -10.0), None);
        assert_eq!(dem.ground(10.0, -10.0), 5.0);
        assert_eq!(Dem::new().ground(0.0, 0.0), 0.0);
    }

    #[test]
    fn lut_matches_the_legacy_ramp_ends() {
        let lut = cyber_lut();
        assert_eq!(&lut[0..3], &[4, 9, 18]);
        assert_eq!(&lut[255 * 3..256 * 3], &[58, 126, 142]);
        // Monotonic in every channel.
        for i in 1..256 {
            for ch in 0..3 {
                assert!(lut[i * 3 + ch] >= lut[(i - 1) * 3 + ch]);
            }
        }
    }

    #[test]
    fn graticule_steps_keep_ground_distance() {
        let (minor, major) = graticule_steps(0.78125);
        assert!((minor - 250.0).abs() < 1.0 && (major - 1000.0).abs() < 1.0);
        // Coarse imagery: never closer than 48 px.
        let (minor, _) = graticule_steps(10.0);
        assert_eq!(minor, 480.0);
    }

    #[test]
    fn grade_draws_lines_on_enu_multiples() {
        let (w, h) = (8u32, 8u32);
        let mut img = vec![255u8; (w * h * 4) as usize];
        // 50 m per pixel, west edge at 0 m: column 0 sits on the 1000 m grid (0 m).
        grade(&mut img, w, h, 0.0, 400.0, 50.0, 50.0);
        let lut = cyber_lut();
        let plain = [lut[255 * 3], lut[255 * 3 + 1], lut[255 * 3 + 2]];
        let px = |x: usize, y: usize| [img[(y * 8 + x) * 4], img[(y * 8 + x) * 4 + 1], img[(y * 8 + x) * 4 + 2]];
        assert_ne!(px(0, 3), plain, "grid column at e = 0");
        assert_eq!(px(3, 3), plain);
    }

    #[test]
    fn mesh_is_a_grid_with_north_up_uvs() {
        let t = tile([0, 0], vec![0.0, 0.0, 10.0, 10.0], 2, 2);
        let m = terrain_mesh(&t, -1.0).unwrap();
        assert_eq!(m.positions.len(), 4);
        assert_eq!(m.positions[2], [0.0, 400.0, 15.0 - 1.0]);
        let uvs = m.uvs.unwrap();
        assert_eq!(uvs[0], [0.0, 1.0]);
        assert_eq!(uvs[3], [1.0, 0.0]);
        assert_eq!(m.indices.len(), 6);
        assert!(terrain_mesh(&tile([0, 0], vec![1.0], 1, 1), 0.0).is_none());
    }

    #[test]
    fn core_cells_match_the_go_grid() {
        let (lo, hi) = core_cell_range([-1500.0, -1300.0], [1600.0, 1700.0], 400.0);
        assert_eq!(lo, [-4, -4]);
        assert_eq!(hi, [3, 4]);
    }
}
