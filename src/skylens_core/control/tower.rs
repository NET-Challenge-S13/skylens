//! The tower's one 3D view (legacy `controlview/towerViewer.ts`) on pointblitz.
//!
//! One scene, three ways to draw its buildings (점 / 검정 텍스처 건물 / 실사 항공뷰): switching
//! swaps materials and visibility, it never rebuilds what is already on the GPU. Everything the
//! legacy scene sized in its normalised units (fog, camera offsets, drone size, pads, point size)
//! is scaled by [`World::unit`], the metres one legacy unit stood for over the same operating
//! area, so the view frames the city the way the legacy one did.

use crate::buildings::{self, Mulberry32};
use crate::fleet::{Fleet, Placement};
use crate::settings::DisplayMode;
use crate::terrain::{self, Dem, EXAGGERATION};
use skylens_shared::proto::model::v1 as model;
use skylens_shared::proto::payload::v1 as payload;
use skylens_viewer::camera::{ChaseCamera, OverviewOrbit, Pose};
use skylens_viewer::config::{self, color};
use skylens_viewer::glam::DVec3;
use skylens_viewer::pointblitz::core::{Blend, Fog, FrameStyle, LineSet, Material, MeshData, PointSize, Shading, Topology};
use skylens_viewer::pointblitz::io::Point as PbPoint;
use skylens_viewer::pointblitz::io::chunk::encode_with;
use skylens_viewer::view3d::{FrameInfo, FrameSpec, Gpu3d, ViewScreen};
use skylens_viewer::{ChunkStore, SiteFrame};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;

// ---------------------------------------------------------------------------------------------
// Legacy look
// ---------------------------------------------------------------------------------------------

/// Trail points kept per drone.
pub const TRAIL_LENGTH: usize = 40;
/// Legacy normalised footprint: the longest side of the area was 44 units.
pub const TARGET_EXTENT: f64 = 44.0;
/// Legacy `droneViewScale()` on map scenes.
pub const DRONE_VIEW_SCALE: f64 = 0.15;
/// Legacy camera: 55 degree vertical field of view, near plane 0.1 units.
pub const FOV_DEG: f64 = 55.0;
/// Sun position of the legacy scene (three.js (40, 70, -30) = ENU (40, 30, 70)).
pub const SUN_DIR: [f64; 3] = [40.0, 30.0, 70.0];

/// Per-mode look (legacy `STYLES`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeStyle {
    pub background: u32,
    pub fog_near: f64,
    pub fog_far: f64,
    pub graded_drape: bool,
    pub light_sky: u32,
    pub light_ground: u32,
    pub light_intensity: f32,
    pub sun_color: u32,
    pub sun_intensity: f32,
    pub building_emissive: u32,
    pub building_points: bool,
    pub building_prisms: bool,
    pub building_edges: bool,
}

pub fn style(mode: DisplayMode) -> ModeStyle {
    match mode {
        DisplayMode::Points => ModeStyle {
            background: color::CONTROL_BG,
            fog_near: 40.0,
            fog_far: 220.0,
            graded_drape: true,
            light_sky: 0x80b9cc,
            light_ground: 0x071019,
            light_intensity: 1.35,
            sun_color: 0xa4d9e8,
            sun_intensity: 0.85,
            building_emissive: 0x152a33,
            building_points: true,
            building_prisms: false,
            building_edges: false,
        },
        DisplayMode::Black => ModeStyle {
            background: color::CONTROL_BG,
            fog_near: 70.0,
            fog_far: 320.0,
            graded_drape: true,
            light_sky: 0x80b9cc,
            light_ground: 0x071019,
            light_intensity: 1.35,
            sun_color: 0xa4d9e8,
            sun_intensity: 0.85,
            building_emissive: 0x152a33,
            building_points: false,
            building_prisms: true,
            building_edges: true,
        },
        DisplayMode::Aerial => ModeStyle {
            background: 0x4a515c,
            fog_near: 95.0,
            fog_far: 360.0,
            graded_drape: false,
            light_sky: 0xe8eef5,
            light_ground: 0x5f676f,
            light_intensity: 2.4,
            sun_color: 0xffffff,
            sun_intensity: 1.7,
            building_emissive: 0x000000,
            building_points: false,
            building_prisms: true,
            building_edges: false,
        },
    }
}

/// 검정 텍스처 건물 prisms: per-vertex wall/roof colours, Lambert under the mode's lights.
pub fn prism_material(s: &ModeStyle) -> Material {
    let d = DVec3::from(SUN_DIR).normalize();
    Material {
        shading: Shading::Lambert {
            sky: config::rgb(s.light_sky),
            ground: config::rgb(s.light_ground),
            hemi_intensity: s.light_intensity,
            sun_dir: [d.x as f32, d.y as f32, d.z as f32],
            sun_color: config::rgb(s.sun_color),
            sun_intensity: s.sun_intensity,
            emissive: config::rgb(s.building_emissive),
        },
        tint: [1.0, 1.0, 1.0, 1.0],
        blend: Blend::Opaque,
        double_sided: true,
    }
}

/// The satellite drape and the 실사 항공뷰 prisms (`MeshBasicMaterial`, double sided).
pub fn unlit_material() -> Material {
    Material { shading: Shading::Unlit, tint: [1.0, 1.0, 1.0, 1.0], blend: Blend::Opaque, double_sided: true }
}

/// Restrained tactical linework (legacy `buildEdges`).
pub fn edge_lines(points: Vec<[f64; 3]>) -> LineSet {
    LineSet {
        positions: points,
        topology: Topology::List,
        colors: None,
        color: config::rgba(0x7dd8ec, 0.5),
        blend: Blend::Additive,
        depth_test: true,
    }
}

/// The eight-sided wireframe of a drone: 12 edges between the six vertices of an octahedron of
/// `radius`, with its horizontal axes turned to face `forward`.
pub fn octahedron_edges(center: DVec3, forward: DVec3, radius: f64) -> Vec<[f64; 3]> {
    let up = DVec3::Z;
    let f = DVec3::new(forward.x, forward.y, 0.0);
    let f = if f.length_squared() > 1e-12 { f.normalize() } else { DVec3::Y };
    // three.js lookAt: the object's z axis points away from the target.
    let z = -f;
    let x = up.cross(z).normalize_or(DVec3::X);
    let v = [x, -x, up, -up, z, -z].map(|a| center + a * radius);
    let mut out = Vec::with_capacity(24);
    for i in 0..6 {
        for j in (i + 1)..6 {
            // Opposite vertices (pairs 0-1, 2-3, 4-5) share no edge.
            if j == i + 1 && i % 2 == 0 {
                continue;
            }
            out.push(v[i].to_array());
            out.push(v[j].to_array());
        }
    }
    out
}

/// Flat ring in the ground plane (three `RingGeometry(inner, outer, 20)` rotated flat).
pub fn ring_mesh(centers: &[[f64; 3]], inner: f64, outer: f64, segments: usize) -> MeshData {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    for c in centers {
        let base = positions.len() as u32;
        for k in 0..=segments {
            let a = k as f64 / segments as f64 * std::f64::consts::TAU;
            let (s, co) = a.sin_cos();
            positions.push([c[0] + co * inner, c[1] + s * inner, c[2]]);
            positions.push([c[0] + co * outer, c[1] + s * outer, c[2]]);
        }
        for k in 0..segments as u32 {
            let a = base + 2 * k;
            indices.extend_from_slice(&[a, a + 1, a + 3, a, a + 3, a + 2]);
        }
    }
    MeshData { positions, normals: None, colors: None, uvs: None, indices, flat: false }
}

/// The flown path: there and back when looping (legacy `setRoute`).
pub fn route_path(air: &[[f64; 3]], looped: bool) -> Vec<[f64; 3]> {
    let mut path = air.to_vec();
    if looped {
        path.extend(air.iter().rev().skip(1));
    }
    path
}

/// Trail vertex colours: the legacy `droneTrail` colour times the drone tint (linear light),
/// alpha `0.8 * (1 - i / n) * (0.9 active | 0.5)`, newest first.
pub fn trail_colors(n: usize, tint: u32, active: bool) -> Vec<[u8; 4]> {
    let mul = |shift: u32| {
        let a = buildings::srgb_to_linear(((color::DRONE_TRAIL >> shift) & 0xff) as u8);
        let b = buildings::srgb_to_linear(((tint >> shift) & 0xff) as u8);
        buildings::linear_to_srgb(a * b)
    };
    let rgb = [mul(16), mul(8), mul(0)];
    (0..n)
        .map(|i| {
            let alpha = (1.0 - i as f64 / n as f64) * if active { 0.9 } else { 0.5 } * 0.8;
            [rgb[0], rgb[1], rgb[2], (alpha * 255.0).round().clamp(0.0, 255.0) as u8]
        })
        .collect()
}

/// Drone tint by id (legacy `DRONE_TINTS[(id - 1) % 3]`).
pub fn drone_tint(id: u32) -> u32 {
    config::DRONE_TINTS[(id.wrapping_sub(1) % config::DRONE_TINTS.len() as u32) as usize]
}

// ---------------------------------------------------------------------------------------------
// Shared world
// ---------------------------------------------------------------------------------------------

/// What the screen and the 3D view share.
#[derive(Debug, Default)]
pub struct World {
    pub store: ChunkStore,
    pub site: Option<SiteFrame>,
    pub fleet: Fleet,
    /// The route in force (as echoed by the core) and whether it loops.
    pub route: Option<(Vec<model::Gps>, bool)>,
    pub route_rev: u64,
    pub display: DisplayMode,
    pub dem: Dem,
    /// Bumped when terrain tiles change; the DEM follows lazily.
    pub terrain_rev: u64,
    dem_rev: u64,
    /// Bumped when building sets change.
    pub buildings_rev: u64,
}

impl Default for DisplayMode {
    fn default() -> Self {
        crate::settings::DEFAULT_DISPLAY
    }
}

/// Metres per legacy scene unit for an area `[min_lat, min_lon, max_lat, max_lon]`.
pub fn unit_for(min_lat: f64, min_lon: f64, max_lat: f64, max_lon: f64) -> f64 {
    let lat0 = (min_lat + max_lat) / 2.0;
    let w = (max_lon - min_lon) * 111_320.0 * lat0.to_radians().cos();
    let h = (max_lat - min_lat) * 111_320.0;
    let span = w.max(h);
    if span > 0.0 { span / TARGET_EXTENT } else { 3000.0 / TARGET_EXTENT }
}

impl World {
    pub fn new(display: DisplayMode) -> World {
        World { display, ..Default::default() }
    }

    /// Brings the DEM up to date with the stored tiles.
    pub fn refresh_dem(&mut self) {
        if self.dem_rev != self.terrain_rev {
            let tiles: Vec<([i32; 2], Rc<payload::TerrainTile>)> = self.store.terrain_tiles().map(|(c, e)| (c, e.tile.clone())).collect();
            self.dem.rebuild(tiles.iter().map(|(c, t)| (*c, t)));
            self.dem_rev = self.terrain_rev;
        }
    }

    /// Metres per legacy unit over the operating area.
    pub fn unit(&self) -> f64 {
        match &self.site {
            Some(s) if s.has_bounds() => unit_for(s.min_lat, s.min_lon, s.max_lat, s.max_lon),
            _ => 3000.0 / TARGET_EXTENT,
        }
    }

    /// The operating area span in metres (building minimum height rule).
    pub fn span_m(&self) -> f64 {
        self.unit() * TARGET_EXTENT
    }

    /// Absolute ground altitude (GPS metres) under a point; the anchor altitude without a site.
    pub fn ground_alt(&mut self, lat: f64, lon: f64) -> f64 {
        self.refresh_dem();
        let Some(site) = &self.site else { return 0.0 };
        let enu = site.enu(skylens_shared::geo::Gps { lat, lon, alt: site.anchor.alt });
        site.anchor.alt + self.dem.ground(enu[0], enu[1])
    }

    /// Ingests telemetry with the scene placement.
    pub fn ingest(&mut self, t: &model::DroneTelemetry, now_ms: f64) {
        self.refresh_dem();
        let World { fleet, site, dem, .. } = self;
        fleet.ingest(t, now_ms, &Placer { site: site.as_ref(), dem });
    }
}

/// GPS to scene placement with the ground under it.
pub struct Placer<'a> {
    pub site: Option<&'a SiteFrame>,
    pub dem: &'a Dem,
}

impl Placement for Placer<'_> {
    fn place(&self, gps: &model::Gps) -> Option<(DVec3, f64)> {
        let site = self.site?;
        let [e, n, u] = site.enu_of(gps);
        Some((DVec3::new(e, n, u * EXAGGERATION), self.dem.ground(e, n) * EXAGGERATION))
    }
}

// ---------------------------------------------------------------------------------------------
// GPU ids
// ---------------------------------------------------------------------------------------------

const K_TERRAIN: u64 = 1;
const K_TERRAIN_RAW: u64 = 2;
const K_PRISM: u64 = 3;
const K_PRISM_AERIAL: u64 = 4;
const K_EDGES: u64 = 5;
const K_GRID: u64 = 6;
const K_ROUTE: u64 = 7;
const K_DROPS: u64 = 8;
const K_PADS: u64 = 9;
const K_CORE: u64 = 10;
const K_HALO: u64 = 11;
const K_TRAIL: u64 = 12;
const K_POINTS: u64 = 13;

fn id(kind: u64, low: u64) -> u64 {
    (kind << 32) | (low & 0xffff_ffff)
}

fn cell_low(c: [i32; 2]) -> u64 {
    let p = |v: i32| ((i64::from(v) + 32_768) as u64) & 0xffff;
    (p(c[0]) << 16) | p(c[1])
}

// ---------------------------------------------------------------------------------------------
// The view
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct TileGpu {
    stamp: u64,
    textured: bool,
    raw: bool,
}

#[derive(Debug, Clone, Copy)]
struct BuildingGpu {
    stamp: u64,
    /// Terrain stamp whose image the aerial prisms use.
    aerial: Option<u64>,
}

/// Uploads one frame may spend on new tiles before drawing (ms).
const UPLOAD_BUDGET_MS: f64 = 12.0;

pub struct TowerScene {
    world: Rc<RefCell<World>>,
    terrain: HashMap<[i32; 2], TileGpu>,
    buildings: HashMap<[i32; 2], BuildingGpu>,
    points_rev: Option<u64>,
    points_built_ms: f64,
    route_key: Option<(u64, u64)>,
    trails: HashMap<u32, VecDeque<DVec3>>,
    overview: OverviewOrbit,
    overview_rev: Option<u64>,
    chase: ChaseCamera,
    grid: bool,
    point_size_m: Option<f32>,
}

impl TowerScene {
    pub fn new(world: Rc<RefCell<World>>) -> TowerScene {
        TowerScene {
            world,
            terrain: HashMap::new(),
            buildings: HashMap::new(),
            points_rev: None,
            points_built_ms: f64::NEG_INFINITY,
            route_key: None,
            trails: HashMap::new(),
            overview: OverviewOrbit::default(),
            overview_rev: None,
            chase: ChaseCamera::default(),
            grid: false,
            point_size_m: None,
        }
    }

    fn sync_terrain(&mut self, gpu: &mut Gpu3d, w: &World, s: &ModeStyle, unit: f64, deadline: f64) {
        let tiles: Vec<([i32; 2], Rc<payload::TerrainTile>, u64)> = w.store.terrain_tiles().map(|(c, e)| (c, e.tile.clone(), e.stamp)).collect();
        let live: HashSet<[i32; 2]> = tiles.iter().map(|(c, _, _)| *c).collect();
        let gone: Vec<[i32; 2]> = self.terrain.keys().filter(|c| !live.contains(*c)).copied().collect();
        for c in gone {
            gpu.meshes.remove(id(K_TERRAIN, cell_low(c)));
            gpu.meshes.remove(id(K_TERRAIN_RAW, cell_low(c)));
            self.terrain.remove(&c);
        }
        for (cell, tile, stamp) in tiles {
            let known = self.terrain.get(&cell).copied();
            let fresh = known.is_none_or(|k| k.stamp != stamp);
            // 실사 항공뷰 needs this tile's photograph, and its buildings draped with it.
            let buildings_undraped = self.buildings.get(&cell).is_some_and(|b| b.aerial != Some(stamp));
            let want_raw = !s.graded_drape && !fresh && known.is_some_and(|k| k.textured && (!k.raw || buildings_undraped));
            if !fresh && !want_raw {
                continue;
            }
            if skylens_viewer::now_ms() > deadline {
                break;
            }
            let z_off = if tile.ring { -0.08 * unit } else { 0.0 };
            let Some(mesh) = terrain::terrain_mesh(&tile, z_off) else {
                self.terrain.insert(cell, TileGpu { stamp, textured: false, raw: false });
                continue;
            };
            if fresh {
                gpu.meshes.remove(id(K_TERRAIN_RAW, cell_low(cell)));
                // The legacy tower draped only imagery it had; a tile without one is not drawn
                // (its heights still place everything else).
                let textured = match terrain::tile_texture(&tile, true) {
                    Some(tex) => {
                        gpu.meshes.upsert(&gpu.device, &gpu.queue, id(K_TERRAIN, cell_low(cell)), &mesh, Some(&tex), unlit_material());
                        gpu.meshes.set_visible(id(K_TERRAIN, cell_low(cell)), s.graded_drape);
                        true
                    }
                    None => {
                        gpu.meshes.remove(id(K_TERRAIN, cell_low(cell)));
                        false
                    }
                };
                self.terrain.insert(cell, TileGpu { stamp, textured, raw: false });
                // The building drape of this cell follows its image.
                if let Some(b) = self.buildings.get_mut(&cell) {
                    b.aerial = None;
                    gpu.meshes.remove(id(K_PRISM_AERIAL, cell_low(cell)));
                }
                if textured && !s.graded_drape {
                    self.upload_raw(gpu, w, &tile, cell, &mesh, unit);
                }
            } else {
                self.upload_raw(gpu, w, &tile, cell, &mesh, unit);
            }
        }
    }

    /// The photograph drape of a tile, and the 실사 항공뷰 prisms of its buildings.
    fn upload_raw(&mut self, gpu: &mut Gpu3d, w: &World, tile: &payload::TerrainTile, cell: [i32; 2], mesh: &MeshData, unit: f64) {
        let Some(tex) = terrain::tile_texture(tile, false) else { return };
        gpu.meshes.upsert(&gpu.device, &gpu.queue, id(K_TERRAIN_RAW, cell_low(cell)), mesh, Some(&tex), unlit_material());
        if let Some(t) = self.terrain.get_mut(&cell) {
            t.raw = true;
        }
        let stamp = self.terrain.get(&cell).map(|t| t.stamp);
        if let (Some(entry), Some(b)) = (w.store.building_set(cell), self.buildings.get_mut(&cell)) {
            let z_off = if tile.ring { -0.08 * unit } else { 0.0 };
            let rect = [tile.min_e, tile.min_n, tile.max_e, tile.max_n];
            let p = buildings::extrude(&entry.set, Some(rect), z_off, unit, w.span_m());
            if !p.mesh.positions.is_empty() {
                gpu.meshes.upsert(&gpu.device, &gpu.queue, id(K_PRISM_AERIAL, cell_low(cell)), &p.mesh, Some(&tex), unlit_material());
            }
            b.aerial = stamp;
        }
    }

    fn sync_buildings(&mut self, gpu: &mut Gpu3d, w: &World, s: &ModeStyle, unit: f64, deadline: f64) {
        let sets: Vec<([i32; 2], Rc<payload::BuildingSet>, u64)> = w.store.building_sets().map(|(c, e)| (c, e.set.clone(), e.stamp)).collect();
        let live: HashSet<[i32; 2]> = sets.iter().map(|(c, _, _)| *c).collect();
        let gone: Vec<[i32; 2]> = self.buildings.keys().filter(|c| !live.contains(*c)).copied().collect();
        for c in gone {
            for k in [K_PRISM, K_PRISM_AERIAL] {
                gpu.meshes.remove(id(k, cell_low(c)));
            }
            gpu.lines.remove(id(K_EDGES, cell_low(c)));
            self.buildings.remove(&c);
        }
        let material = prism_material(s);
        for (cell, set, stamp) in sets {
            if self.buildings.get(&cell).is_some_and(|b| b.stamp == stamp) {
                continue;
            }
            if skylens_viewer::now_ms() > deadline {
                break;
            }
            let tile = w.store.terrain_tile(cell).map(|e| e.tile.clone());
            let ring = tile.as_ref().is_some_and(|t| t.ring);
            let z_off = if ring { -0.08 * unit } else { 0.0 };
            let p = buildings::extrude(&set, None, z_off, unit, w.span_m());
            gpu.meshes.remove(id(K_PRISM_AERIAL, cell_low(cell)));
            if p.mesh.positions.is_empty() {
                gpu.meshes.remove(id(K_PRISM, cell_low(cell)));
                gpu.lines.remove(id(K_EDGES, cell_low(cell)));
            } else {
                gpu.meshes.upsert(&gpu.device, &gpu.queue, id(K_PRISM, cell_low(cell)), &p.mesh, None, material);
                gpu.lines.upsert(&gpu.device, &gpu.queue, id(K_EDGES, cell_low(cell)), &edge_lines(p.edges));
            }
            // In 실사 항공뷰 the terrain pass drapes the new prisms on the next frame.
            self.buildings.insert(cell, BuildingGpu { stamp, aerial: None });
        }
    }

    /// Visibility and materials for the display mode (legacy `applyMode`).
    fn apply_mode(&mut self, gpu: &mut Gpu3d, mode: DisplayMode, s: &ModeStyle) {
        let material = prism_material(s);
        for (cell, t) in &self.terrain {
            let raw_ready = t.raw && !s.graded_drape;
            gpu.meshes.set_visible(id(K_TERRAIN, cell_low(*cell)), t.textured && !raw_ready);
            gpu.meshes.set_visible(id(K_TERRAIN_RAW, cell_low(*cell)), raw_ready);
        }
        for (cell, b) in &self.buildings {
            // Aerial falls back to the lit prisms where no drape is ready.
            let aerial = mode == DisplayMode::Aerial && b.aerial.is_some();
            gpu.meshes.set_material(id(K_PRISM, cell_low(*cell)), material);
            gpu.meshes.set_visible(id(K_PRISM, cell_low(*cell)), s.building_prisms && !aerial);
            gpu.meshes.set_visible(id(K_PRISM_AERIAL, cell_low(*cell)), s.building_prisms && aerial);
            gpu.lines.set_visible(id(K_EDGES, cell_low(*cell)), s.building_edges);
        }
        gpu.clouds.set_visible(id(K_POINTS, 0), s.building_points);
    }

    /// The 점 point cloud over the area's buildings, rebuilt at most once a second while cells
    /// keep arriving.
    fn sync_points(&mut self, gpu: &mut Gpu3d, w: &World, unit: f64, now: f64) {
        if self.points_rev == Some(w.buildings_rev) || now - self.points_built_ms < 1000.0 {
            return;
        }
        self.points_rev = Some(w.buildings_rev);
        self.points_built_ms = now;
        let core = w.site.as_ref().and_then(|s| s.enu_bounds()).map(|(lo, hi)| terrain::core_cell_range(lo, hi, terrain::CORE_CELL_M));
        let mut sets: Vec<([i32; 2], Rc<payload::BuildingSet>)> = w
            .store
            .building_sets()
            .filter(|(c, _)| core.is_none_or(|(lo, hi)| c[0] >= lo[0] && c[0] <= hi[0] && c[1] >= lo[1] && c[1] <= hi[1]))
            .map(|(c, e)| (c, e.set.clone()))
            .collect();
        sets.sort_by_key(|(c, _)| (c[1], c[0]));
        let list: Vec<&payload::Building> = sets.iter().flat_map(|(_, s)| s.buildings.iter()).collect();
        let total = buildings::total_weight(&list);
        if list.is_empty() || total <= 0.0 {
            gpu.clouds.remove(id(K_POINTS, 0));
            return;
        }
        let mut rng = Mulberry32::new(0xb17d_5eed);
        let pts = buildings::sample_points(&list, &w.dem, buildings::POINT_BUDGET / total, w.span_m(), &mut rng);
        let (lo, hi) = pts.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| (lo.min(p.pos[2]), hi.max(p.pos[2])));
        let span = (hi - lo).max(0.001 * unit);
        let encoded: Vec<PbPoint> = pts
            .iter()
            .map(|p| PbPoint {
                position: [p.pos[0] as f32, p.pos[1] as f32, p.pos[2] as f32],
                color: buildings::ramp_color((p.pos[2] - lo) / span, color::CONTROL_POINT_FAR, color::CONTROL_POINT),
            })
            .collect();
        let chunks: Vec<Vec<u8>> = encoded.chunks(32_768).enumerate().map(|(i, c)| encode_with(0, i as u32, 0, c, 12)).collect();
        let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
        gpu.clouds.replace(&gpu.device, id(K_POINTS, 0), &refs);
        // Legacy PointsMaterial opacity 0.92.
        gpu.clouds.set_alpha(id(K_POINTS, 0), 0.92);
    }

    fn sync_route(&mut self, gpu: &mut Gpu3d, w: &World, unit: f64) {
        let key = (w.route_rev, w.terrain_rev);
        if self.route_key == Some(key) {
            return;
        }
        self.route_key = Some(key);
        let legs: Option<Vec<([f64; 3], [f64; 3])>> = match (&w.route, &w.site) {
            (Some((wps, _)), Some(site)) if wps.len() >= 2 => Some(
                wps.iter()
                    .map(|g| {
                        let [e, n, u] = site.enu_of(g);
                        ([e, n, u * EXAGGERATION], [e, n, w.dem.ground(e, n) * EXAGGERATION])
                    })
                    .collect(),
            ),
            _ => None,
        };
        let Some(legs) = legs else {
            gpu.lines.remove(id(K_ROUTE, 0));
            gpu.lines.remove(id(K_DROPS, 0));
            gpu.meshes.remove(id(K_PADS, 0));
            return;
        };
        let looped = w.route.as_ref().is_some_and(|(_, l)| *l);
        let air: Vec<[f64; 3]> = legs.iter().map(|(a, _)| *a).collect();
        let gold = 0xffd27f;
        gpu.lines.upsert(
            &gpu.device,
            &gpu.queue,
            id(K_ROUTE, 0),
            &LineSet { positions: route_path(&air, looped), topology: Topology::Strip, colors: None, color: config::rgba(gold, 0.9), blend: Blend::Alpha, depth_test: true },
        );
        let drops: Vec<[f64; 3]> = legs.iter().flat_map(|(a, g)| [*a, *g]).collect();
        gpu.lines.upsert(
            &gpu.device,
            &gpu.queue,
            id(K_DROPS, 0),
            &LineSet { positions: drops, topology: Topology::List, colors: None, color: config::rgba(gold, 0.35), blend: Blend::Alpha, depth_test: true },
        );
        let grounds: Vec<[f64; 3]> = legs.iter().map(|(_, g)| *g).collect();
        let pads = ring_mesh(&grounds, 0.25 * unit, 0.42 * unit, 20);
        let pad_material = Material { shading: Shading::Unlit, tint: config::rgba(gold, 0.75), blend: Blend::Alpha, double_sided: true };
        gpu.meshes.upsert(&gpu.device, &gpu.queue, id(K_PADS, 0), &pads, None, pad_material);
    }

    fn sync_grid(&mut self, gpu: &mut Gpu3d, w: &World, unit: f64) {
        // Reference grid, only meaningful when there is no draped terrain under it.
        let want = !self.terrain.values().any(|t| t.textured);
        if want == self.grid {
            return;
        }
        self.grid = want;
        if !want {
            gpu.lines.remove(id(K_GRID, 0));
            return;
        }
        let size = 80.0 * unit;
        let half = size / 2.0;
        let step = size / 32.0;
        let z = w.dem.ground(0.0, 0.0) * EXAGGERATION;
        let mut positions = Vec::new();
        let mut colors = Vec::new();
        for i in 0..=32 {
            let k = -half + f64::from(i) * step;
            let c = config::rgba8(if i == 16 { 0x2a4a8a } else { 0x16294f }, 89);
            positions.extend_from_slice(&[[-half, k, z], [half, k, z], [k, -half, z], [k, half, z]]);
            colors.extend_from_slice(&[c, c, c, c]);
        }
        gpu.lines.upsert(
            &gpu.device,
            &gpu.queue,
            id(K_GRID, 0),
            &LineSet { positions, topology: Topology::List, colors: Some(colors), color: [1.0; 4], blend: Blend::Alpha, depth_test: true },
        );
    }

    fn sync_drones(&mut self, gpu: &mut Gpu3d, fleet: &Fleet, unit: f64) {
        let live: HashSet<u32> = fleet.tracks().iter().filter(|t| t.placed).map(|t| t.info.id).collect();
        let gone: Vec<u32> = self.trails.keys().filter(|i| !live.contains(*i)).copied().collect();
        for d in gone {
            for k in [K_CORE, K_HALO, K_TRAIL] {
                gpu.lines.remove(id(k, u64::from(d)));
            }
            self.trails.remove(&d);
        }
        for t in fleet.tracks().iter().filter(|t| t.placed) {
            let d = t.info.id;
            let active = fleet.active == Some(d);
            let tint = drone_tint(d);
            let scale = if active { 1.6 } else { 1.0 } * DRONE_VIEW_SCALE * unit;
            let core = LineSet {
                positions: octahedron_edges(t.pos, t.forward, 0.6 * scale),
                topology: Topology::List,
                colors: None,
                color: config::rgba(tint, if active { 1.0 } else { 0.65 }),
                blend: if active { Blend::Opaque } else { Blend::Alpha },
                depth_test: true,
            };
            gpu.lines.upsert(&gpu.device, &gpu.queue, id(K_CORE, u64::from(d)), &core);
            if active {
                let halo = LineSet {
                    positions: octahedron_edges(t.pos, t.forward, 0.6 * scale * 1.32),
                    topology: Topology::List,
                    colors: None,
                    color: config::rgba(0xe8fbff, 0.72),
                    blend: Blend::Alpha,
                    depth_test: true,
                };
                gpu.lines.upsert(&gpu.device, &gpu.queue, id(K_HALO, u64::from(d)), &halo);
            } else {
                gpu.lines.remove(id(K_HALO, u64::from(d)));
            }
            let trail = self.trails.entry(d).or_default();
            trail.push_front(t.pos);
            trail.truncate(TRAIL_LENGTH);
            if trail.len() >= 2 {
                let positions: Vec<[f64; 3]> = trail.iter().map(|p| p.to_array()).collect();
                let colors = trail_colors(positions.len(), tint, active);
                gpu.lines.upsert(
                    &gpu.device,
                    &gpu.queue,
                    id(K_TRAIL, u64::from(d)),
                    &LineSet { positions, topology: Topology::Strip, colors: Some(colors), color: [1.0; 4], blend: Blend::Alpha, depth_test: true },
                );
            }
        }
    }

    /// Frames the standby orbit on the area's terrain (legacy `frameOverview`).
    fn frame_overview(&mut self, w: &World, unit: f64) {
        if self.overview_rev == Some(w.terrain_rev) {
            return;
        }
        self.overview_rev = Some(w.terrain_rev);
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for (_, e) in w.store.terrain_tiles() {
            if e.tile.ring {
                continue;
            }
            let (a, b) = terrain::tile_bounds(&e.tile);
            lo = lo.min(DVec3::from(a));
            hi = hi.max(DVec3::from(b));
        }
        if lo.x.is_finite() {
            self.overview.frame_bounds(lo, hi);
        } else {
            self.overview.center = DVec3::ZERO;
            self.overview.radius = 40.0 * unit;
        }
    }
}

impl ViewScreen for TowerScene {
    fn gpu_lost(&mut self) {
        self.terrain.clear();
        self.buildings.clear();
        self.points_rev = None;
        self.points_built_ms = f64::NEG_INFINITY;
        self.route_key = None;
        self.trails.clear();
        self.grid = false;
        self.point_size_m = None;
    }

    fn frame(&mut self, gpu: &mut Gpu3d, info: &FrameInfo) -> FrameSpec {
        let world = self.world.clone();
        let mut w = world.borrow_mut();
        // Legacy loop clamp (MAX_DT = 1/20 s).
        let dt = info.dt.clamp(0.0, 1.0 / 20.0);
        w.refresh_dem();
        w.fleet.update(dt, info.now_ms);
        let unit = w.unit();
        let mode = w.display;
        let s = style(mode);
        let deadline = skylens_viewer::now_ms() + UPLOAD_BUDGET_MS;

        self.sync_terrain(gpu, &w, &s, unit, deadline);
        self.sync_buildings(gpu, &w, &s, unit, deadline);
        if s.building_points {
            self.sync_points(gpu, &w, unit, info.now_ms);
        }
        self.sync_route(gpu, &w, unit);
        self.sync_grid(gpu, &w, unit);
        self.sync_drones(gpu, &w.fleet, unit);
        // Cheap: visibility flags and material constants, every frame so new uploads follow.
        self.apply_mode(gpu, mode, &s);

        self.frame_overview(&w, unit);
        let pose = match w.fleet.focus().filter(|t| t.placed) {
            None => {
                // No telemetry, no fleet: show the operating area.
                self.chase.reset();
                self.overview.step(dt)
            }
            Some(t) => {
                self.chase.scale = DRONE_VIEW_SCALE * unit;
                self.chase.step(t.pos, t.forward, dt)
            }
        };
        // three.js size attenuation is `size * (height / 2) / depth`, without the 1 / tan(fov / 2)
        // of a true metre size, so the legacy 0.16 units become this many metres.
        let point_m = (0.16 * unit * (FOV_DEG.to_radians() / 2.0).tan()) as f32;
        let point_size = (self.point_size_m != Some(point_m)).then(|| {
            self.point_size_m = Some(point_m);
            PointSize::Metres(point_m)
        });
        let bg = config::rgb(s.background);
        FrameSpec {
            camera: Pose::camera(&pose, FOV_DEG, 0.1 * unit),
            style: FrameStyle {
                clear: config::clear(s.background),
                fog: Some(Fog { color: bg, near: (s.fog_near * unit) as f32, far: (s.fog_far * unit) as f32 }),
            },
            point_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn styles_follow_the_legacy_table() {
        let b = style(DisplayMode::Black);
        assert_eq!((b.background, b.fog_near, b.fog_far), (0x0a1128, 70.0, 320.0));
        assert!(b.building_prisms && b.building_edges && !b.building_points && b.graded_drape);
        let a = style(DisplayMode::Aerial);
        assert_eq!((a.background, a.fog_near, a.fog_far), (0x4a515c, 95.0, 360.0));
        assert!(!a.graded_drape && !a.building_edges);
        let p = style(DisplayMode::Points);
        assert_eq!((p.fog_near, p.fog_far), (40.0, 220.0));
        assert!(p.building_points && !p.building_prisms);
    }

    #[test]
    fn unit_matches_the_legacy_normalisation() {
        // Daejeon preset: the longer side is about 3.1 km, so a legacy unit was about 71 m.
        let u = unit_for(36.355, 127.33, 36.382, 127.365);
        assert!((u - 3138.0 / 44.0).abs() < 2.0, "{u}");
    }

    #[test]
    fn octahedron_has_twelve_edges_of_equal_length() {
        let e = octahedron_edges(DVec3::new(1.0, 2.0, 3.0), DVec3::new(1.0, 1.0, 0.0), 2.0);
        assert_eq!(e.len(), 24);
        for pair in e.chunks(2) {
            let d = DVec3::from(pair[0]).distance(DVec3::from(pair[1]));
            assert!((d - 2.0 * std::f64::consts::SQRT_2).abs() < 1e-9);
        }
    }

    #[test]
    fn looping_routes_come_back() {
        let air = [[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        assert_eq!(route_path(&air, false).len(), 3);
        let back = route_path(&air, true);
        assert_eq!(back.len(), 5);
        assert_eq!(back[4], [0.0; 3]);
    }

    #[test]
    fn trail_fades_from_the_newest_point() {
        let c = trail_colors(4, 0x9fe8ff, true);
        assert_eq!(c.len(), 4);
        assert!(c[0][3] > c[3][3]);
        assert_eq!(c[0][3], (0.9f64 * 0.8 * 255.0).round() as u8);
        let inactive = trail_colors(4, 0x9fe8ff, false);
        assert!(inactive[0][3] < c[0][3]);
    }

    #[test]
    fn tints_wrap_like_the_legacy_palette() {
        assert_eq!(drone_tint(1), 0x9fe8ff);
        assert_eq!(drone_tint(2), 0xffd27f);
        assert_eq!(drone_tint(4), 0x9fe8ff);
    }

    #[test]
    fn pads_are_flat_rings() {
        let m = ring_mesh(&[[0.0, 0.0, 5.0]], 1.0, 2.0, 20);
        assert_eq!(m.positions.len(), 42);
        assert_eq!(m.indices.len(), 120);
        assert!(m.positions.iter().all(|p| p[2] == 5.0));
    }
}
