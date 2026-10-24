//! Detection fusion: per-frame UNet output projected onto the reconstructed
//! points and accumulated across frames, zones and drones.
//!
//! Pure math, no cumulus3d types. The engine adapts registered camera poses into
//! [`View`]s and zone clouds into point lists.
//!
//! - One global map. Evidence lives in one map of ENU cells shared by every zone
//!   and drone: [`CELL_M`] squares on the East/North plane split into
//!   [`CELL_H_M`] height bins (2.5D, so a roof and the ground under it are two
//!   cells). A revisit or another drone's frame adds to the same cells and the
//!   same person tracks. Inside a cell the evidence is kept per zone (the zone
//!   whose registered pose fused the frame), only so that a zone rebuild
//!   (preview -> refined) can replace its own share without counting a frame
//!   twice. Every stored frame is fused through exactly one zone (its home).
//! - Occlusion. Per frame, the home zone's cells are projected and splatted into
//!   a coarse depth buffer; only cells on the front-most surface along their
//!   pixel ray (within [`DEPTH_TOL_M`] + [`DEPTH_TOL_FRAC`] x depth) receive
//!   evidence. A coarser copy of the buffer stays with the frame to decide
//!   whether a person track was in view (a hidden track is not a miss).
//! - Hazards. Every visible cell adds the class probabilities at its pixel,
//!   weighted by how decisive that pixel is (its largest class probability). A
//!   cell is a hazard when the weighted mean of `1 - p(normal)` reaches
//!   [`HAZARD_MIN_PROB`]; its class is then the most likely non-normal class (so
//!   fire 0.45 / collapse 0.40 is a fire). Hazard columns of one class that touch
//!   (8-neighbourhood) form one zone with a convex outline.
//! - People. A person point is placed on the cell nearest to it in pixels among
//!   the visible cells on the front surface of its pixel ray, within
//!   [`PERSON_PICK_FRAC`] of the image. Per frame, sightings are assigned one to
//!   one to existing tracks within [`PERSON_MERGE_M`] (greedy by distance), so a
//!   track takes at most one detection per frame and two detections of one frame
//!   are two tracks. Every frame that had the track in view counts as a look, hit
//!   or miss, and the confidence is `score sum / looks`.
//! - Views. A frame carries its drone and capture time ([`FrameSource`]).
//!   Frames of one drone within [`VIEW_WINDOW_MS`] of each other are one view;
//!   a hazard cell or a person needs [`MIN_VIEWS`] distinct views.
//! - Ids are global and stable (`fire-3`, `person-7`): a hazard zone inherits
//!   the id of the previous zone of the same class it overlaps most; a person the
//!   id of the nearest previous person.
//! - Emission. Every detection has one owner segment (a zone that contributed
//!   to it, kept while it still contributes), and [`Fuser::batches`] returns one
//!   batch per owner segment. A detection is in exactly one batch, so the
//!   per-segment replacement contract of the detection chunks holds without
//!   duplicates; when a detection changes owner both batches change.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use skylens_shared::geo::{Enu, GeoAnchor, enu_to_gps};
use skylens_shared::proto::model::v1::Gps;
use skylens_shared::proto::payload::v1::{Detection, DetectionBatch, DetectionCategory};

use crate::detect::{FrameDetections, HAZARD_CLASSES, PersonPoint};

/// Edge of one fusion cell on the East/North plane, metres.
pub const CELL_M: f64 = 1.0;
/// Height of one cell bin, metres (cells are keyed by `floor(z / CELL_H_M)`).
pub const CELL_H_M: f64 = 1.0;
/// Largest edge of the stored probability grid. Frames are box-averaged down to
/// this so hundreds of frames fit in memory (4 bytes per grid pixel). The
/// per-frame depth buffer uses the same size.
pub const GRID_MAX: u32 = 160;
/// Largest edge of the depth buffer kept with each fused frame for person looks.
pub const LOOK_GRID_MAX: u32 = 64;
/// A cell is on the front surface when its depth is at most the nearest depth at
/// its pixel plus this...
pub const DEPTH_TOL_M: f64 = 1.5;
/// ...plus this fraction of its depth.
pub const DEPTH_TOL_FRAC: f64 = 0.02;
/// Largest splat radius of one cell in the depth buffer, buffer pixels.
const SPLAT_MAX_PX: i64 = 8;
/// A cell is a hazard when its fused `1 - p(normal)` reaches this.
pub const HAZARD_MIN_PROB: f64 = 0.5;
/// A hazard cell or a person needs this many distinct views.
pub const MIN_VIEWS: usize = 2;
/// Frames of one drone this close in capture time are one view, milliseconds.
pub const VIEW_WINDOW_MS: i64 = 2_000;
/// View tags kept per cell part or track (only `MIN_VIEWS` of them matter).
const MAX_VIEW_TAGS: usize = 8;
/// Person points below this score are ignored.
pub const PERSON_MIN_SCORE: f32 = 0.3;
/// A person point snaps to a cell projection within this fraction of the
/// image's larger side.
pub const PERSON_PICK_FRAC: f64 = 0.03;
/// Person points closer than this on the ground plane are the same person.
pub const PERSON_MERGE_M: f64 = 2.0;
/// A person is reported once its interpolated confidence (score sum / looks)
/// reaches this (and [`MIN_VIEWS`] views detected it).
pub const PERSON_MIN_CONFIDENCE: f64 = 0.4;

/// Who took a frame and when: frames of one drone within [`VIEW_WINDOW_MS`]
/// count as one view. Without a time every frame is its own view.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrameSource {
    /// Drone (camera) id, e.g. `formation-1/left`.
    pub drone: String,
    /// Capture time, Unix milliseconds.
    pub time_ms: Option<i64>,
}

/// One frame's UNet output, downsampled for storage.
#[derive(Debug, Clone)]
pub struct FrameObservation {
    /// Original frame size; person points are in these pixels.
    pub width: u32,
    pub height: u32,
    grid_w: u32,
    grid_h: u32,
    /// `[class][gy][gx]`, probability * 255.
    prob: Vec<u8>,
    pub people: Vec<PersonPoint>,
    pub source: FrameSource,
}

impl FrameObservation {
    /// Box-averages the hazard probabilities to at most [`GRID_MAX`] per side.
    /// A probability buffer of the wrong length is dropped (no hazard evidence),
    /// the people are kept. The source is empty (see [`Self::with_source`]).
    pub fn from_detections(d: &FrameDetections) -> Self {
        let (w, h) = (d.width.max(1), d.height.max(1));
        let scale = (GRID_MAX as f64 / w.max(h) as f64).min(1.0);
        let gw = ((w as f64 * scale).round() as u32).max(1);
        let gh = ((h as f64 * scale).round() as u32).max(1);
        let plane = (d.width as usize) * (d.height as usize);
        let valid = plane > 0 && d.hazard_prob.len() == HAZARD_CLASSES * plane;
        let mut prob = Vec::new();
        if valid {
            prob = vec![0u8; HAZARD_CLASSES * (gw * gh) as usize];
            for c in 0..HAZARD_CLASSES {
                let src = &d.hazard_prob[c * plane..(c + 1) * plane];
                for gy in 0..gh {
                    let y0 = (gy as u64 * h as u64 / gh as u64) as usize;
                    let y1 = (((gy + 1) as u64 * h as u64 / gh as u64) as usize).max(y0 + 1);
                    for gx in 0..gw {
                        let x0 = (gx as u64 * w as u64 / gw as u64) as usize;
                        let x1 = (((gx + 1) as u64 * w as u64 / gw as u64) as usize).max(x0 + 1);
                        let mut sum = 0.0f64;
                        for y in y0..y1 {
                            let row = &src[y * w as usize..(y + 1) * w as usize];
                            sum += row[x0..x1].iter().map(|v| f64::from(*v)).sum::<f64>();
                        }
                        let mean = sum / ((y1 - y0) * (x1 - x0)) as f64;
                        let i = c * (gw * gh) as usize + (gy * gw + gx) as usize;
                        prob[i] = (mean.clamp(0.0, 1.0) * 255.0).round() as u8;
                    }
                }
            }
        }
        Self { width: w, height: h, grid_w: gw, grid_h: gh, prob, people: d.people.clone(), source: FrameSource::default() }
    }

    /// Sets who took the frame and when.
    pub fn with_source(mut self, source: FrameSource) -> Self {
        self.source = source;
        self
    }

    /// Class probabilities at frame pixel `(x, y)` (pixel centres at +0.5).
    /// `None` outside the frame or when the frame carried no probabilities.
    pub fn hazard_at(&self, x: f64, y: f64) -> Option<[f64; HAZARD_CLASSES]> {
        if self.prob.is_empty() || !(x >= 0.0 && y >= 0.0) {
            return None;
        }
        let (w, h) = (self.width as f64, self.height as f64);
        if x >= w || y >= h {
            return None;
        }
        let gx = ((x / w * self.grid_w as f64) as u32).min(self.grid_w - 1);
        let gy = ((y / h * self.grid_h as f64) as u32).min(self.grid_h - 1);
        let plane = (self.grid_w * self.grid_h) as usize;
        let at = (gy * self.grid_w + gx) as usize;
        Some(std::array::from_fn(|c| f64::from(self.prob[c * plane + at]) / 255.0))
    }
}

/// Camera-frame point (x right, y down, z forward) -> pixel.
pub type Projector = Arc<dyn Fn([f64; 3]) -> Option<[f64; 2]> + Send + Sync>;

/// A registered camera: world (ENU) -> camera pose and its projection.
#[derive(Clone)]
pub struct View {
    /// Size of the image the projector returns pixels in.
    pub width: u32,
    pub height: u32,
    rotation: [[f64; 3]; 3],
    translation: [f64; 3],
    project: Projector,
}

impl std::fmt::Debug for View {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("View").field("width", &self.width).field("height", &self.height).finish()
    }
}

impl View {
    /// `p_cam = rotation * p_world + translation`.
    pub fn new(
        width: u32,
        height: u32,
        rotation: [[f64; 3]; 3],
        translation: [f64; 3],
        project: Projector,
    ) -> Self {
        Self { width, height, rotation, translation, project }
    }

    /// Distortion-free pinhole with focal `f` and the principal point at the
    /// image centre.
    pub fn pinhole(f: f64, width: u32, height: u32, rotation: [[f64; 3]; 3], translation: [f64; 3]) -> Self {
        let (cx, cy) = (width as f64 / 2.0, height as f64 / 2.0);
        let project: Projector = Arc::new(move |p: [f64; 3]| {
            (p[2] > f64::EPSILON).then(|| [f * p[0] / p[2] + cx, f * p[1] / p[2] + cy])
        });
        Self::new(width, height, rotation, translation, project)
    }

    fn to_camera(&self, p: [f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        std::array::from_fn(|i| r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2] + self.translation[i])
    }

    /// Pixel of a world point, if it is in front of the camera and inside the image.
    pub fn project(&self, p: [f64; 3]) -> Option<[f64; 2]> {
        self.project_depth(p).map(|(px, _)| px)
    }

    /// Pixel and camera depth (z) of a world point in front of the camera and inside the image.
    pub fn project_depth(&self, p: [f64; 3]) -> Option<([f64; 2], f64)> {
        let c = self.to_camera(p);
        if c[2] <= f64::EPSILON {
            return None;
        }
        let px = (self.project)(c)?;
        let inside = px[0] >= 0.0 && px[1] >= 0.0 && px[0] < self.width as f64 && px[1] < self.height as f64;
        inside.then_some((px, c[2]))
    }

    /// Approximate focal length in pixels near the optical axis.
    fn focal_px(&self) -> f64 {
        const D: f64 = 1e-3;
        match ((self.project)([0.0, 0.0, 1.0]), (self.project)([D, 0.0, 1.0])) {
            (Some(a), Some(b)) if (b[0] - a[0]).abs() > 0.0 => (b[0] - a[0]).abs() / D,
            _ => self.width.max(self.height) as f64,
        }
    }
}

/// Depth slack of the front-surface test at depth `d`.
fn depth_tol(d: f64) -> f64 {
    DEPTH_TOL_M + DEPTH_TOL_FRAC * d
}

/// Nearest camera depth per buffer pixel of one view (`inf` where nothing projects).
#[derive(Debug, Clone)]
struct DepthGrid {
    w: u32,
    h: u32,
    /// Buffer pixels per view pixel.
    scale: f64,
    depth: Vec<f32>,
}

impl DepthGrid {
    fn for_view(view: &View, max: u32) -> Self {
        let long = view.width.max(view.height).max(1);
        let scale = (max as f64 / long as f64).min(1.0);
        let w = ((view.width as f64 * scale).ceil() as u32).max(1);
        let h = ((view.height as f64 * scale).ceil() as u32).max(1);
        Self { w, h, scale, depth: vec![f32::INFINITY; (w * h) as usize] }
    }

    fn pixel(&self, px: [f64; 2]) -> Option<(i64, i64)> {
        let (x, y) = ((px[0] * self.scale).floor(), (px[1] * self.scale).floor());
        (x >= 0.0 && y >= 0.0 && x < self.w as f64 && y < self.h as f64).then_some((x as i64, y as i64))
    }

    /// Writes `depth` over a square of radius `r` buffer pixels around `px`.
    fn splat(&mut self, px: [f64; 2], depth: f64, r: i64) {
        let Some((cx, cy)) = self.pixel(px) else { return };
        let d = depth as f32;
        for y in (cy - r).max(0)..=(cy + r).min(self.h as i64 - 1) {
            for x in (cx - r).max(0)..=(cx + r).min(self.w as i64 - 1) {
                let i = (y as u32 * self.w + x as u32) as usize;
                if d < self.depth[i] {
                    self.depth[i] = d;
                }
            }
        }
    }

    /// Nearest depth at `px` (`inf` outside or where nothing projected).
    fn front(&self, px: [f64; 2]) -> f64 {
        self.pixel(px).map_or(f64::INFINITY, |(x, y)| f64::from(self.depth[(y as u32 * self.w + x as u32) as usize]))
    }

    /// Whether a point at `depth` seen at `px` lies on the front surface.
    fn visible(&self, px: [f64; 2], depth: f64) -> bool {
        let front = self.front(px);
        !front.is_finite() || depth <= front + depth_tol(front)
    }

    /// Min-pooled copy with at most `max` pixels per side (a pooled pixel keeps its nearest depth,
    /// so the copy errs towards hiding).
    fn pooled(&self, max: u32) -> Self {
        let k = self.w.max(self.h).div_ceil(max.max(1)).max(1);
        if k == 1 {
            return self.clone();
        }
        let (w, h) = (self.w.div_ceil(k), self.h.div_ceil(k));
        let mut depth = vec![f32::INFINITY; (w * h) as usize];
        for y in 0..self.h {
            for x in 0..self.w {
                let i = ((y / k) * w + x / k) as usize;
                depth[i] = depth[i].min(self.depth[(y * self.w + x) as usize]);
            }
        }
        Self { w, h, scale: self.scale / k as f64, depth }
    }
}

/// `(east, north)` cell column.
type ColKey = (i32, i32);
/// `(east, north, height bin)`.
type CellKey = (i32, i32, i32);

fn cell_key(p: [f64; 3]) -> CellKey {
    ((p[0] / CELL_M).floor() as i32, (p[1] / CELL_M).floor() as i32, (p[2] / CELL_H_M).floor() as i32)
}

/// A view: interned drone and capture time.
type ViewTag = (u32, Option<i64>);

/// Adds `tag` unless a view of the same drone within [`VIEW_WINDOW_MS`] is already there.
fn add_view(views: &mut Vec<ViewTag>, tag: ViewTag) {
    let same = views.iter().any(|v| {
        v.0 == tag.0 && matches!((v.1, tag.1), (Some(a), Some(b)) if (a - b).abs() <= VIEW_WINDOW_MS)
    });
    if !same && views.len() < MAX_VIEW_TAGS {
        views.push(tag);
    }
}

/// One zone's share of a cell: its points there and the evidence of the frames it fused.
#[derive(Debug, Clone, Default)]
struct Part {
    sum: [f64; 3],
    points: u32,
    sum_w: f64,
    sum_wp: [f64; HAZARD_CLASSES],
    frames: u32,
    views: Vec<ViewTag>,
}

/// A global ENU cell.
#[derive(Debug, Clone, Default)]
struct Cell {
    /// By segment.
    parts: BTreeMap<u32, Part>,
}

/// A zone's own geometry and registered cameras.
#[derive(Default)]
struct Zone {
    /// Cell and the mean position of the zone's points in it.
    cells: Vec<(CellKey, [f64; 3])>,
    views: HashMap<String, View>,
}

#[derive(Debug, Clone)]
struct Sighting {
    pos: [f64; 3],
    score: f64,
}

/// What fusing one frame left behind for the person tracks.
struct FrameRecord {
    arrival: u64,
    /// Home zone.
    seg: u32,
    view: View,
    look: DepthGrid,
    sightings: Vec<Sighting>,
    tag: ViewTag,
}

impl FrameRecord {
    /// Whether the frame had `p` in view and not hidden behind a nearer surface.
    fn sees(&self, p: [f64; 3]) -> bool {
        self.view.project_depth(p).is_some_and(|(px, d)| self.look.visible(px, d))
    }
}

#[derive(Debug, Clone)]
struct PersonTrack {
    /// Score-weighted mean ENU position.
    pos: [f64; 3],
    score_sum: f64,
    /// Frames that detected it.
    hits: u32,
    looks: u32,
    views: Vec<ViewTag>,
    /// Home zones of the frames that detected it.
    segs: BTreeMap<u32, u32>,
}

impl PersonTrack {
    fn confidence(&self) -> f64 {
        if self.looks == 0 { 0.0 } else { (self.score_sum / self.looks as f64).clamp(0.0, 1.0) }
    }
}

/// Previously reported ids and their owner segments.
#[derive(Default)]
struct Ids {
    hazards: Vec<(String, usize, HashSet<ColKey>)>,
    people: Vec<(String, [f64; 2])>,
    next: HashMap<&'static str, u32>,
    owners: HashMap<String, u32>,
}

impl Ids {
    fn fresh(&mut self, kind: &'static str) -> String {
        let n = self.next.entry(kind).or_insert(0);
        *n += 1;
        format!("{kind}-{n}")
    }

    fn hazard(&mut self, class: usize, cols: &HashSet<ColKey>, used: &mut HashSet<String>) -> String {
        let best = self
            .hazards
            .iter()
            .enumerate()
            .filter(|(_, (id, c, _))| *c == class && !used.contains(id))
            .map(|(i, (_, _, prev))| (i, prev.intersection(cols).count()))
            .filter(|(_, overlap)| *overlap > 0)
            .max_by_key(|(_, overlap)| *overlap);
        let id = match best {
            Some((i, _)) => {
                self.hazards[i].2 = cols.clone();
                self.hazards[i].0.clone()
            }
            None => {
                let id = self.fresh(class_label(class));
                self.hazards.push((id.clone(), class, cols.clone()));
                id
            }
        };
        used.insert(id.clone());
        id
    }

    fn person(&mut self, pos: [f64; 2], used: &mut HashSet<String>) -> String {
        let best = self
            .people
            .iter()
            .enumerate()
            .filter(|(_, (id, _))| !used.contains(id))
            .map(|(i, (_, p))| (i, ((p[0] - pos[0]).powi(2) + (p[1] - pos[1]).powi(2)).sqrt()))
            .filter(|(_, d)| *d <= PERSON_MERGE_M)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        let id = match best {
            Some((i, _)) => {
                self.people[i].1 = pos;
                self.people[i].0.clone()
            }
            None => {
                let id = self.fresh("person");
                self.people.push((id.clone(), pos));
                id
            }
        };
        used.insert(id.clone());
        id
    }

    /// Owner segment of `id`: its previous owner while that still contributes, otherwise the
    /// segment with the largest weight in `candidates` (lowest on a tie).
    fn owner(&mut self, id: &str, candidates: &BTreeMap<u32, u32>) -> u32 {
        let prev = self.owners.get(id).copied();
        let seg = match prev {
            Some(p) if candidates.contains_key(&p) || candidates.is_empty() => p,
            _ => candidates
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                .map_or(0, |(s, _)| *s),
        };
        self.owners.insert(id.to_string(), seg);
        seg
    }
}

/// Class index -> label used in detection ids and `Detection.label`.
pub fn class_label(class: usize) -> &'static str {
    match class {
        1 => "fire",
        2 => "collapse",
        3 => "road_blocked",
        _ => "normal",
    }
}

fn class_category(class: usize) -> DetectionCategory {
    match class {
        1 => DetectionCategory::Fire,
        2 => DetectionCategory::Collapse,
        3 => DetectionCategory::RoadBlocked,
        _ => DetectionCategory::Unspecified,
    }
}

/// Fuses frame observations of every zone and drone into one ENU map.
pub struct Fuser {
    anchor: GeoAnchor,
    /// Observations by stored image name with their arrival number, oldest first for eviction.
    observations: HashMap<String, (u64, Arc<FrameObservation>)>,
    order: VecDeque<String>,
    max_observations: usize,
    arrivals: u64,
    zones: BTreeMap<u32, Zone>,
    cells: BTreeMap<CellKey, Cell>,
    /// Fused frames by name (one home zone each).
    records: HashMap<String, FrameRecord>,
    people: Vec<PersonTrack>,
    drones: Vec<String>,
    ids: Ids,
}

impl Fuser {
    /// `max_observations` bounds the stored frames (oldest dropped first).
    pub fn new(anchor: GeoAnchor, max_observations: usize) -> Self {
        Self {
            anchor,
            observations: HashMap::new(),
            order: VecDeque::new(),
            max_observations: max_observations.max(1),
            arrivals: 0,
            zones: BTreeMap::new(),
            cells: BTreeMap::new(),
            records: HashMap::new(),
            people: Vec::new(),
            drones: Vec::new(),
            ids: Ids::default(),
        }
    }

    fn tag(&mut self, s: &FrameSource) -> ViewTag {
        let i = match self.drones.iter().position(|d| *d == s.drone) {
            Some(i) => i,
            None => {
                self.drones.push(s.drone.clone());
                self.drones.len() - 1
            }
        };
        (i as u32, s.time_ms)
    }

    /// Replaces the geometry of a segment (a preview or a refined cloud). The
    /// segment's share of the evidence is dropped and every stored frame that
    /// has no home zone (its frames included) is fused again, then the person
    /// tracks are rebuilt from every fused frame in arrival order. Ids already
    /// reported are kept where the result overlaps.
    pub fn set_zone(&mut self, segment: u32, points: &[[f64; 3]], views: HashMap<String, View>) {
        if let Some(old) = self.zones.remove(&segment) {
            for (k, _) in old.cells {
                if let Some(c) = self.cells.get_mut(&k) {
                    c.parts.remove(&segment);
                    if c.parts.is_empty() {
                        self.cells.remove(&k);
                    }
                }
            }
        }
        self.records.retain(|_, r| r.seg != segment);

        let mut acc: BTreeMap<CellKey, ([f64; 3], u32)> = BTreeMap::new();
        for p in points {
            if !(p[0].is_finite() && p[1].is_finite() && p[2].is_finite()) {
                continue;
            }
            let e = acc.entry(cell_key(*p)).or_insert(([0.0; 3], 0));
            for i in 0..3 {
                e.0[i] += p[i];
            }
            e.1 += 1;
        }
        let mut cells = Vec::with_capacity(acc.len());
        for (k, (sum, n)) in acc {
            cells.push((k, sum.map(|v| v / f64::from(n))));
            self.cells.entry(k).or_default().parts.insert(segment, Part { sum, points: n, ..Part::default() });
        }
        self.zones.insert(segment, Zone { cells, views });

        // Fuse in arrival order so the tracks evolve the way they would live.
        let mut pending: Vec<(u64, String)> = self
            .observations
            .iter()
            .filter(|(name, _)| !self.records.contains_key(*name))
            .map(|(name, (arrival, _))| (*arrival, name.clone()))
            .collect();
        pending.sort_unstable();
        for (_, name) in pending {
            self.fuse(&name);
        }
        self.rebuild_people();
    }

    /// Stores one frame's observation and fuses it through its home zone (the
    /// lowest segment that has the frame registered). Returns the segment that
    /// fused it (empty when no zone has it yet, or when it was fused before).
    pub fn add_observation(&mut self, name: &str, obs: FrameObservation) -> Vec<u32> {
        let obs = Arc::new(obs);
        match self.observations.get_mut(name) {
            Some(slot) => slot.1 = obs,
            None => {
                self.arrivals += 1;
                self.observations.insert(name.to_string(), (self.arrivals, obs));
                self.order.push_back(name.to_string());
            }
        }
        while self.order.len() > self.max_observations {
            if let Some(old) = self.order.pop_front() {
                // The cells keep its hazard evidence until its zone is rebuilt.
                self.observations.remove(&old);
                self.records.remove(&old);
            }
        }
        if self.records.contains_key(name) {
            return Vec::new();
        }
        match self.fuse(name) {
            Some(seg) => {
                if let Some(rec) = self.records.get(name) {
                    apply_sightings(&mut self.people, rec);
                }
                vec![seg]
            }
            None => Vec::new(),
        }
    }

    /// Fuses a stored observation through its home zone and keeps its record.
    fn fuse(&mut self, name: &str) -> Option<u32> {
        let (arrival, obs) = self.observations.get(name).map(|(a, o)| (*a, Arc::clone(o)))?;
        let (seg, view) = self.zones.iter().find_map(|(s, z)| z.views.get(name).map(|v| (*s, v.clone())))?;
        let tag = self.tag(&obs.source);
        let zone = &self.zones[&seg];
        let rec = fuse_frame(zone, seg, &mut self.cells, &view, &obs, tag, arrival);
        self.records.insert(name.to_string(), rec);
        Some(seg)
    }

    fn rebuild_people(&mut self) {
        let mut recs: Vec<&FrameRecord> = self.records.values().collect();
        recs.sort_by_key(|r| r.arrival);
        let mut people = Vec::new();
        for r in recs {
            apply_sightings(&mut people, r);
        }
        self.people = people;
    }

    /// Segments known to the fuser.
    pub fn segments(&self) -> Vec<u32> {
        self.zones.keys().copied().collect()
    }

    /// The detections owned by `segment` as they stand now (see [`Self::batches`]).
    pub fn batch(&mut self, segment: u32) -> DetectionBatch {
        self.batches().remove(&segment).unwrap_or_default()
    }

    /// Every detection as it stands now, grouped by owner segment. Each detection is in exactly
    /// one batch; segments that own nothing are absent.
    pub fn batches(&mut self) -> BTreeMap<u32, DetectionBatch> {
        let anchor = self.anchor;
        let mut used = HashSet::new();
        let mut out: BTreeMap<u32, DetectionBatch> = BTreeMap::new();

        // Hazard cells, gathered per column and class.
        #[derive(Default)]
        struct Col {
            conf: f64,
            center: [f64; 3],
            bins: u32,
            frames: u32,
            segs: BTreeMap<u32, u32>,
        }
        let mut cols: BTreeMap<(usize, ColKey), Col> = BTreeMap::new();
        for (k, cell) in &self.cells {
            let (mut sum_w, mut sum_wp, mut frames, mut points) = (0.0, [0.0; HAZARD_CLASSES], 0u32, 0u32);
            let mut sum = [0.0; 3];
            let mut views = Vec::new();
            for part in cell.parts.values() {
                sum_w += part.sum_w;
                for i in 0..HAZARD_CLASSES {
                    sum_wp[i] += part.sum_wp[i];
                }
                frames += part.frames;
                points += part.points;
                for i in 0..3 {
                    sum[i] += part.sum[i];
                }
                for v in &part.views {
                    add_view(&mut views, *v);
                }
            }
            if views.len() < MIN_VIEWS || sum_w <= 0.0 || points == 0 {
                continue;
            }
            let mean: [f64; HAZARD_CLASSES] = std::array::from_fn(|i| sum_wp[i] / sum_w);
            let p_hazard = (1.0 - mean[0]).clamp(0.0, 1.0);
            if p_hazard < HAZARD_MIN_PROB {
                continue;
            }
            let class = (1..HAZARD_CLASSES).max_by(|a, b| mean[*a].total_cmp(&mean[*b]).then(b.cmp(a))).unwrap_or(1);
            let col = cols.entry((class, (k.0, k.1))).or_default();
            col.conf = col.conf.max(p_hazard);
            for i in 0..3 {
                col.center[i] += sum[i] / f64::from(points);
            }
            col.bins += 1;
            col.frames = col.frames.max(frames);
            for (s, p) in &cell.parts {
                *col.segs.entry(*s).or_insert(0) += p.points.max(1);
            }
        }
        // Connected components per class, in key order (deterministic).
        let mut seen: BTreeSet<(usize, ColKey)> = BTreeSet::new();
        for &start in cols.keys() {
            if seen.contains(&start) {
                continue;
            }
            let class = start.0;
            let mut comp = Vec::new();
            let mut stack = vec![start];
            seen.insert(start);
            while let Some(k) = stack.pop() {
                comp.push(k.1);
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let nk = (class, (k.1.0 + dx, k.1.1 + dy));
                        if cols.contains_key(&nk) && seen.insert(nk) {
                            stack.push(nk);
                        }
                    }
                }
            }
            comp.sort_unstable();
            let set: HashSet<ColKey> = comp.iter().copied().collect();
            let id = self.ids.hazard(class, &set, &mut used);
            let n = comp.len() as f64;
            let mut centroid = [0.0; 3];
            let mut conf = 0.0;
            let mut frames = 0;
            let mut segs: BTreeMap<u32, u32> = BTreeMap::new();
            let mut corners = Vec::with_capacity(comp.len() * 4);
            for k in &comp {
                let c = &cols[&(class, *k)];
                for i in 0..3 {
                    centroid[i] += c.center[i] / f64::from(c.bins) / n;
                }
                conf += c.conf / n;
                frames = frames.max(c.frames);
                for (s, w) in &c.segs {
                    *segs.entry(*s).or_insert(0) += w;
                }
                let (e0, n0) = (k.0 as f64 * CELL_M, k.1 as f64 * CELL_M);
                corners.extend([[e0, n0], [e0 + CELL_M, n0], [e0 + CELL_M, n0 + CELL_M], [e0, n0 + CELL_M]]);
            }
            let segment = self.ids.owner(&id, &segs);
            let outline = convex_hull(&corners).into_iter().map(|p| to_gps([p[0], p[1], centroid[2]], anchor)).collect();
            out.entry(segment).or_default().detections.push(Detection {
                id,
                category: class_category(class) as i32,
                gps: Some(to_gps(centroid, anchor)),
                confidence: conf.clamp(0.0, 1.0),
                label: class_label(class).to_string(),
                segment,
                outline,
                frames,
            });
        }

        // People.
        for t in &self.people {
            let conf = t.confidence();
            if t.views.len() < MIN_VIEWS || conf < PERSON_MIN_CONFIDENCE {
                continue;
            }
            let id = self.ids.person([t.pos[0], t.pos[1]], &mut used);
            let segment = self.ids.owner(&id, &t.segs);
            out.entry(segment).or_default().detections.push(Detection {
                id,
                category: DetectionCategory::Person as i32,
                gps: Some(to_gps(t.pos, anchor)),
                confidence: conf,
                label: "person".to_string(),
                segment,
                outline: Vec::new(),
                frames: t.hits,
            });
        }
        out
    }
}

fn to_gps(p: [f64; 3], anchor: GeoAnchor) -> Gps {
    enu_to_gps(Enu { e: p[0], n: p[1], u: p[2] }, anchor).into()
}

fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Fuses one frame through its home zone: hazard evidence on every front-surface
/// cell in view, then the frame's person sightings placed in ENU.
fn fuse_frame(
    zone: &Zone,
    seg: u32,
    cells: &mut BTreeMap<CellKey, Cell>,
    view: &View,
    obs: &FrameObservation,
    tag: ViewTag,
    arrival: u64,
) -> FrameRecord {
    // View pixels -> observation pixels (the reconstruction may work on a rescaled image).
    let sx = obs.width as f64 / view.width.max(1) as f64;
    let sy = obs.height as f64 / view.height.max(1) as f64;

    // Depth test: splat every projected cell (about its half width) into the buffer.
    let mut zbuf = DepthGrid::for_view(view, GRID_MAX);
    let focal = view.focal_px();
    let projected: Vec<(usize, [f64; 2], f64)> = zone
        .cells
        .iter()
        .enumerate()
        .filter_map(|(i, (_, c))| view.project_depth(*c).map(|(px, d)| (i, px, d)))
        .collect();
    for (_, px, d) in &projected {
        let r = (focal * 0.5 * CELL_M / d * zbuf.scale).round().clamp(0.0, SPLAT_MAX_PX as f64) as i64;
        zbuf.splat(*px, *d, r);
    }
    let visible: Vec<(usize, [f64; 2], f64)> =
        projected.into_iter().filter(|(_, px, d)| zbuf.visible(*px, *d)).collect();

    for (i, px, _) in &visible {
        let Some(p) = obs.hazard_at(px[0] * sx, px[1] * sy) else { continue };
        let w = p.iter().copied().fold(0.0, f64::max);
        if w <= 0.0 {
            continue;
        }
        let Some(part) = cells.get_mut(&zone.cells[*i].0).and_then(|c| c.parts.get_mut(&seg)) else { continue };
        part.sum_w += w;
        for c in 0..HAZARD_CLASSES {
            part.sum_wp[c] += w * p[c];
        }
        part.frames += 1;
        add_view(&mut part.views, tag);
    }

    // People: the nearest cell in pixels among the visible ones on the front surface of the
    // person's ray (any visible cell when none is), within the pick radius.
    let radius = PERSON_PICK_FRAC * view.width.max(view.height) as f64;
    let mut sightings = Vec::new();
    for person in obs.people.iter().filter(|p| p.score >= PERSON_MIN_SCORE) {
        let target = [person.x as f64 / sx, person.y as f64 / sy];
        let front = zbuf.front(target);
        let near: Vec<(usize, f64, f64)> = visible
            .iter()
            .map(|(i, px, d)| (*i, dist2(*px, target), *d))
            .filter(|(_, dpx, _)| *dpx <= radius)
            .collect();
        let on_ray = if front.is_finite() {
            nearest(near.iter().filter(|(_, _, d)| (d - front).abs() <= depth_tol(front)))
        } else {
            None
        };
        let Some(i) = on_ray.or_else(|| nearest(near.iter())) else { continue };
        sightings.push(Sighting { pos: zone.cells[i].1, score: f64::from(person.score.clamp(0.0, 1.0)) });
    }
    FrameRecord { arrival, seg, view: view.clone(), look: zbuf.pooled(LOOK_GRID_MAX), sightings, tag }
}

/// The candidate `(cell, pixel distance, depth)` nearest in pixels (nearer in depth on a tie).
fn nearest<'a>(it: impl Iterator<Item = &'a (usize, f64, f64)>) -> Option<usize> {
    it.min_by(|a, b| a.1.total_cmp(&b.1).then(a.2.total_cmp(&b.2))).map(|c| c.0)
}

/// Adds one frame's sightings to the tracks: one-to-one assignment to the tracks that existed
/// before the frame (greedy by distance), new tracks for the rest, and a miss for every earlier
/// track the frame had in view but did not detect.
fn apply_sightings(people: &mut Vec<PersonTrack>, rec: &FrameRecord) {
    let before = people.len();
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (si, s) in rec.sightings.iter().enumerate() {
        for (ti, t) in people.iter().enumerate() {
            let d = dist2([s.pos[0], s.pos[1]], [t.pos[0], t.pos[1]]);
            if d <= PERSON_MERGE_M {
                pairs.push((d, si, ti));
            }
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut s_used = vec![false; rec.sightings.len()];
    let mut t_used = vec![false; before];
    for (_, si, ti) in pairs {
        if s_used[si] || t_used[ti] {
            continue;
        }
        s_used[si] = true;
        t_used[ti] = true;
        let (s, t) = (&rec.sightings[si], &mut people[ti]);
        let w = t.score_sum + s.score;
        if w > 0.0 {
            for j in 0..3 {
                t.pos[j] = (t.pos[j] * t.score_sum + s.pos[j] * s.score) / w;
            }
        }
        t.score_sum = w;
        t.hits += 1;
        t.looks += 1;
        add_view(&mut t.views, rec.tag);
        *t.segs.entry(rec.seg).or_insert(0) += 1;
    }
    for (si, s) in rec.sightings.iter().enumerate() {
        if !s_used[si] {
            people.push(PersonTrack {
                pos: s.pos,
                score_sum: s.score,
                hits: 1,
                looks: 1,
                views: vec![rec.tag],
                segs: BTreeMap::from([(rec.seg, 1)]),
            });
        }
    }
    for (ti, t) in people.iter_mut().take(before).enumerate() {
        if !t_used[ti] && rec.sees(t.pos) {
            t.looks += 1;
        }
    }
}

/// Convex hull (monotone chain), counter-clockwise, without the closing point.
pub fn convex_hull(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut p: Vec<[f64; 2]> = points.to_vec();
    p.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let mut lower: Vec<[f64; 2]> = Vec::new();
    for &q in &p {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], q) <= 0.0 {
            lower.pop();
        }
        lower.push(q);
    }
    let mut upper: Vec<[f64; 2]> = Vec::new();
    for &q in p.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], q) <= 0.0 {
            upper.pop();
        }
        upper.push(q);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

#[cfg(test)]
mod tests {
    use super::*;
    use skylens_shared::geo::{Gps as GeoGps, gps_to_enu};

    const ANCHOR: GeoAnchor = GeoGps { lat: 36.3685, lon: 127.3475, alt: 30.0 };
    const W: u32 = 200;
    const H: u32 = 200;
    const F: f64 = 100.0;
    const ALT: f64 = 50.0;

    /// Nadir camera at (e, n, ALT): image x = East, image y = South.
    fn nadir(e: f64, n: f64) -> View {
        let r = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
        // t = -R C, C = (e, n, ALT)
        let t = [-e, n, ALT];
        View::pinhole(F, W, H, r, t)
    }

    /// Flat ground 0..60 m East and North, one point per 0.5 m.
    fn ground() -> Vec<[f64; 3]> {
        let mut v = Vec::new();
        for i in 0..120 {
            for j in 0..120 {
                v.push([i as f64 * 0.5 + 0.25, j as f64 * 0.5 + 0.25, 0.0]);
            }
        }
        v
    }

    /// Per-pixel probabilities: `inside` within the pixel box, normal (0.95) elsewhere.
    fn frame_probs(inside: [f32; HAZARD_CLASSES], x0: usize, x1: usize, y0: usize, y1: usize, people: Vec<PersonPoint>) -> FrameDetections {
        let plane = (W * H) as usize;
        let mut prob = vec![0.0f32; HAZARD_CLASSES * plane];
        for y in 0..H as usize {
            for x in 0..W as usize {
                let i = y * W as usize + x;
                let p = if x >= x0 && x < x1 && y >= y0 && y < y1 { inside } else { [0.95, 0.05, 0.0, 0.0] };
                for c in 0..HAZARD_CLASSES {
                    prob[c * plane + i] = p[c];
                }
            }
        }
        FrameDetections { width: W, height: H, hazard_prob: prob, people }
    }

    /// Frame of size W x H, class `class` with probability 0.9 inside the
    /// pixel box, normal elsewhere.
    fn frame(class: usize, x0: usize, x1: usize, y0: usize, y1: usize, people: Vec<PersonPoint>) -> FrameDetections {
        let mut inside = [0.1, 0.0, 0.0, 0.0];
        if class != 0 {
            inside[class] = 0.9;
        } else {
            inside = [0.95, 0.05, 0.0, 0.0];
        }
        frame_probs(inside, x0, x1, y0, y1, people)
    }

    fn obs(d: FrameDetections, drone: &str, t: i64) -> FrameObservation {
        FrameObservation::from_detections(&d).with_source(FrameSource { drone: drone.into(), time_ms: Some(t) })
    }

    fn fire() -> FrameDetections {
        frame(1, 90, 110, 90, 110, vec![])
    }

    fn person_at(x: f32, y: f32) -> Vec<PersonPoint> {
        vec![PersonPoint { x, y, score: 0.9 }]
    }

    fn empty() -> FrameDetections {
        frame(0, 0, 0, 0, 0, vec![])
    }

    fn views(names: &[(&str, View)]) -> HashMap<String, View> {
        names.iter().map(|(n, v)| (n.to_string(), v.clone())).collect()
    }

    fn enu_of(d: &Detection) -> Enu {
        let g = d.gps.clone().unwrap();
        gps_to_enu(GeoGps { lat: g.lat, lon: g.lon, alt: g.alt }, ANCHOR)
    }

    fn all(f: &mut Fuser) -> Vec<Detection> {
        f.batches().into_values().flat_map(|b| b.detections).collect()
    }

    #[test]
    fn downsample_keeps_box_means() {
        let d = frame(1, 0, 100, 0, 200, vec![]);
        let o = FrameObservation::from_detections(&d);
        assert_eq!((o.width, o.height), (W, H));
        let left = o.hazard_at(10.5, 10.5).unwrap();
        let right = o.hazard_at(150.5, 10.5).unwrap();
        assert!((left[1] - 0.9).abs() < 0.01, "{left:?}");
        assert!((right[0] - 0.95).abs() < 0.01, "{right:?}");
        assert!(o.hazard_at(-1.0, 5.0).is_none());
        assert!(o.hazard_at(200.0, 5.0).is_none());
    }

    #[test]
    fn wrong_length_probabilities_are_dropped() {
        let d = FrameDetections { width: 10, height: 10, hazard_prob: vec![0.5; 7], people: vec![] };
        let o = FrameObservation::from_detections(&d);
        assert!(o.hazard_at(1.0, 1.0).is_none());
    }

    #[test]
    fn nadir_view_projects_ground() {
        let v = nadir(30.0, 30.0);
        let c = v.project([30.0, 30.0, 0.0]).unwrap();
        assert!((c[0] - 100.0).abs() < 1e-9 && (c[1] - 100.0).abs() < 1e-9);
        // 10 m East -> +20 px x; 10 m North -> -20 px y.
        let p = v.project([40.0, 40.0, 0.0]).unwrap();
        assert!((p[0] - 120.0).abs() < 1e-9 && (p[1] - 80.0).abs() < 1e-9);
        // Behind the camera.
        assert!(v.project([30.0, 30.0, 60.0]).is_none());
        assert!((v.focal_px() - F).abs() < 1e-6);
    }

    #[test]
    fn hazard_needs_two_views_then_appears_with_outline() {
        let mut f = Fuser::new(ANCHOR, 64);
        f.set_zone(0, &ground(), views(&[("a", nadir(30.0, 30.0)), ("b", nadir(31.0, 30.0))]));
        // Fire in pixels 90..110 (about 10 m wide around the centre).
        f.add_observation("a", FrameObservation::from_detections(&fire()));
        assert!(all(&mut f).is_empty(), "one frame is not enough");
        let changed = f.add_observation("b", FrameObservation::from_detections(&frame(1, 88, 108, 90, 110, vec![])));
        assert_eq!(changed, vec![0]);
        let b = f.batch(0);
        assert_eq!(b.detections.len(), 1, "{:?}", b.detections);
        let d = &b.detections[0];
        assert_eq!(d.category, DetectionCategory::Fire as i32);
        assert_eq!((d.label.as_str(), d.id.as_str(), d.segment), ("fire", "fire-1", 0));
        assert!(d.frames >= 2);
        assert!(d.confidence >= HAZARD_MIN_PROB && d.confidence <= 1.0);
        assert!(d.outline.len() >= 4);
        let c = enu_of(d);
        assert!((c.e - 30.0).abs() < 2.0 && (c.n - 30.0).abs() < 2.0, "{c:?}");
    }

    #[test]
    fn one_disagreeing_frame_is_outvoted() {
        let mut f = Fuser::new(ANCHOR, 64);
        let names = ["a", "b", "c", "d"];
        let vs: Vec<(&str, View)> = names.iter().map(|n| (*n, nadir(30.0, 30.0))).collect();
        f.set_zone(0, &ground(), views(&vs));
        f.add_observation("a", FrameObservation::from_detections(&fire()));
        for n in &names[1..] {
            f.add_observation(n, FrameObservation::from_detections(&empty()));
        }
        assert!(all(&mut f).is_empty());
    }

    #[test]
    fn mixed_hazard_classes_are_reported_as_the_likeliest_hazard() {
        let mut f = Fuser::new(ANCHOR, 64);
        f.set_zone(0, &ground(), views(&[("a", nadir(30.0, 30.0)), ("b", nadir(30.0, 30.0))]));
        // No class reaches 0.5 alone, but a hazard is 0.85 likely.
        let mixed = || frame_probs([0.15, 0.45, 0.40, 0.0], 90, 110, 90, 110, vec![]);
        f.add_observation("a", obs(mixed(), "d1", 0));
        f.add_observation("b", obs(mixed(), "d2", 0));
        let d = all(&mut f);
        assert_eq!(d.len(), 1, "{d:?}");
        assert_eq!(d[0].category, DetectionCategory::Fire as i32);
        assert!((d[0].confidence - 0.85).abs() < 0.02, "{}", d[0].confidence);
    }

    #[test]
    fn person_needs_repeat_sightings() {
        let mut f = Fuser::new(ANCHOR, 64);
        let names = ["a", "b", "c", "d", "e"];
        let vs: Vec<(&str, View)> = names.iter().map(|n| (*n, nadir(30.0, 30.0))).collect();
        f.set_zone(3, &ground(), views(&vs));
        let seen = || frame(0, 0, 0, 0, 0, person_at(120.0, 80.0));
        f.add_observation("a", FrameObservation::from_detections(&seen()));
        assert!(all(&mut f).is_empty());
        f.add_observation("b", FrameObservation::from_detections(&seen()));
        let b = f.batch(3);
        assert_eq!(b.detections.len(), 1);
        let d = &b.detections[0];
        assert_eq!(d.category, DetectionCategory::Person as i32);
        assert_eq!((d.frames, d.segment), (2, 3));
        assert!(d.id.starts_with("person-"), "{}", d.id);
        assert!(d.outline.is_empty());
        let c = enu_of(d);
        // Pixel (120, 80) is 10 m East and 10 m North of the camera.
        assert!((c.e - 40.0).abs() < 1.5 && (c.n - 40.0).abs() < 1.5, "{c:?}");
        let id = d.id.clone();
        // Three misses lower the interpolated confidence (1.8 / 5) below the threshold.
        for n in ["c", "d", "e"] {
            f.add_observation(n, FrameObservation::from_detections(&empty()));
        }
        assert!(all(&mut f).is_empty());
        // Rebuilding the zone (refined cloud) keeps the id.
        f.set_zone(3, &ground(), views(&vs[..2]));
        let again = all(&mut f);
        assert_eq!(again.len(), 1);
        assert_eq!(again[0].id, id);
    }

    #[test]
    fn single_sighting_among_many_is_not_reported() {
        let mut f = Fuser::new(ANCHOR, 64);
        let names = ["a", "b", "c", "d", "e"];
        let vs: Vec<(&str, View)> = names.iter().map(|n| (*n, nadir(30.0, 30.0))).collect();
        f.set_zone(0, &ground(), views(&vs));
        f.add_observation("a", FrameObservation::from_detections(&frame(0, 0, 0, 0, 0, person_at(100.0, 100.0))));
        for n in &names[1..] {
            f.add_observation(n, FrameObservation::from_detections(&empty()));
        }
        assert!(all(&mut f).is_empty());
    }

    #[test]
    fn one_frame_double_peak_is_not_reported() {
        let mut f = Fuser::new(ANCHOR, 64);
        f.set_zone(0, &ground(), views(&[("a", nadir(30.0, 30.0))]));
        // Two peaks 1 m apart in one frame: two tracks of one hit each, not one track of two.
        let two = vec![PersonPoint { x: 120.0, y: 80.0, score: 0.9 }, PersonPoint { x: 122.0, y: 80.0, score: 0.9 }];
        f.add_observation("a", obs(frame(0, 0, 0, 0, 0, two), "d1", 0));
        assert!(all(&mut f).is_empty());
        assert_eq!(f.people.len(), 2);
        assert!(f.people.iter().all(|t| t.hits == 1));
    }

    #[test]
    fn two_people_in_each_frame_stay_two_tracks() {
        let mut f = Fuser::new(ANCHOR, 64);
        f.set_zone(0, &ground(), views(&[("a", nadir(30.0, 30.0)), ("b", nadir(30.0, 30.0))]));
        let two = || vec![PersonPoint { x: 120.0, y: 80.0, score: 0.9 }, PersonPoint { x: 122.0, y: 80.0, score: 0.9 }];
        f.add_observation("a", obs(frame(0, 0, 0, 0, 0, two()), "d1", 0));
        f.add_observation("b", obs(frame(0, 0, 0, 0, 0, two()), "d2", 0));
        let d = all(&mut f);
        assert_eq!(d.len(), 2, "{d:?}");
        assert!(d.iter().all(|p| p.frames == 2));
        assert_ne!(d[0].id, d[1].id);
    }

    #[test]
    fn same_drone_consecutive_frames_count_once() {
        let mut f = Fuser::new(ANCHOR, 64);
        let vs: Vec<(&str, View)> = ["a", "b", "c"].iter().map(|n| (*n, nadir(30.0, 30.0))).collect();
        f.set_zone(0, &ground(), views(&vs));
        let seen = || frame(1, 90, 110, 90, 110, person_at(120.0, 80.0));
        f.add_observation("a", obs(seen(), "d1", 1_000));
        f.add_observation("b", obs(seen(), "d1", 1_000 + VIEW_WINDOW_MS / 4));
        assert!(all(&mut f).is_empty(), "two frames of one drone 0.5 s apart are one view");
        f.add_observation("c", obs(seen(), "d1", 1_000 + 3 * VIEW_WINDOW_MS));
        let d = all(&mut f);
        assert_eq!(d.len(), 2, "a later look of the same drone is a second view: {d:?}");
    }

    #[test]
    fn two_drones_count_twice() {
        let mut f = Fuser::new(ANCHOR, 64);
        f.set_zone(0, &ground(), views(&[("a", nadir(30.0, 30.0)), ("b", nadir(31.0, 30.0))]));
        let seen = || frame(1, 90, 110, 90, 110, person_at(120.0, 80.0));
        f.add_observation("a", obs(seen(), "formation-1/left", 1_000));
        f.add_observation("b", obs(seen(), "formation-1/right", 1_000));
        let d = all(&mut f);
        let cats: BTreeSet<i32> = d.iter().map(|x| x.category).collect();
        assert_eq!(cats, BTreeSet::from([DetectionCategory::Fire as i32, DetectionCategory::Person as i32]), "{d:?}");
    }

    #[test]
    fn revisit_from_another_zone_merges_into_the_same_ids() {
        let mut f = Fuser::new(ANCHOR, 64);
        let seen = || frame(1, 90, 110, 90, 110, person_at(120.0, 80.0));
        f.set_zone(0, &ground(), views(&[("a", nadir(30.0, 30.0)), ("b", nadir(30.0, 30.0))]));
        f.add_observation("a", obs(seen(), "d1", 0));
        f.add_observation("b", obs(seen(), "d2", 0));
        let first = all(&mut f);
        assert_eq!(first.len(), 2);
        // Another zone (a later pass) covers the same ground and sees the same fire and person.
        f.set_zone(5, &ground(), views(&[("c", nadir(31.0, 30.0)), ("d", nadir(30.0, 31.0))]));
        f.add_observation("c", obs(seen(), "d1", 60_000));
        f.add_observation("d", obs(seen(), "d2", 60_000));
        let batches = f.batches();
        assert_eq!(batches.keys().copied().collect::<Vec<_>>(), vec![0], "owner stays the first zone");
        let again: Vec<Detection> = batches.into_values().flat_map(|b| b.detections).collect();
        assert_eq!(again.len(), 2, "no duplicates: {again:?}");
        let ids = |v: &[Detection]| v.iter().map(|d| d.id.clone()).collect::<BTreeSet<_>>();
        assert_eq!(ids(&again), ids(&first));
        assert!(again.iter().all(|d| d.frames >= 4), "{again:?}");
        // Refining the first zone keeps everything in place.
        f.set_zone(0, &ground(), views(&[("a", nadir(30.0, 30.0)), ("b", nadir(30.0, 30.0))]));
        assert_eq!(ids(&all(&mut f)), ids(&first));
        assert_eq!(all(&mut f).len(), 2);
    }

    #[test]
    fn owner_moves_when_its_zone_no_longer_contributes() {
        let mut f = Fuser::new(ANCHOR, 64);
        f.set_zone(0, &ground(), views(&[("a", nadir(30.0, 30.0)), ("b", nadir(30.0, 30.0))]));
        f.set_zone(2, &ground(), views(&[("a", nadir(30.0, 30.0)), ("b", nadir(30.0, 30.0))]));
        f.add_observation("a", obs(fire(), "d1", 0));
        f.add_observation("b", obs(fire(), "d2", 0));
        let id = f.batch(0).detections[0].id.clone();
        // Zone 0's refined cloud lies elsewhere: the frames are fused through zone 2 now.
        let far: Vec<[f64; 3]> = ground().into_iter().map(|p| [p[0] + 500.0, p[1], p[2]]).collect();
        f.set_zone(0, &far, HashMap::new());
        let b = f.batches();
        assert!(!b.contains_key(&0));
        assert_eq!(b[&2].detections.len(), 1);
        assert_eq!((b[&2].detections[0].id.as_str(), b[&2].detections[0].segment), (id.as_str(), 2));
    }

    /// Ground plus a 4 m x 4 m roof 10 m up over East/North 28..32.
    fn ground_and_roof() -> Vec<[f64; 3]> {
        let mut v = ground();
        for i in 0..8 {
            for j in 0..8 {
                v.push([28.0 + i as f64 * 0.5 + 0.25, 28.0 + j as f64 * 0.5 + 0.25, 10.2]);
            }
        }
        v
    }

    #[test]
    fn occluded_cell_gets_no_evidence() {
        let mut f = Fuser::new(ANCHOR, 64);
        f.set_zone(0, &ground_and_roof(), views(&[("a", nadir(30.0, 30.0))]));
        f.add_observation("a", obs(fire(), "d1", 0));
        let frames = |k: CellKey| f.cells[&k].parts[&0].frames;
        assert_eq!(frames((30, 30, 10)), 1, "the roof is seen");
        assert_eq!(frames((30, 30, 0)), 0, "the ground under the roof is hidden");
        assert_eq!(frames((20, 20, 0)), 1, "open ground is seen");
    }

    #[test]
    fn hidden_track_is_not_a_miss_and_people_land_on_the_front_surface() {
        let mut f = Fuser::new(ANCHOR, 64);
        let names = ["a", "b", "c"];
        let vs: Vec<(&str, View)> = names.iter().map(|n| (*n, nadir(30.0, 30.0))).collect();
        f.set_zone(0, &ground_and_roof(), views(&vs));
        // A person on the roof (pixel of East/North 30.5 at 40 m depth).
        f.add_observation("a", obs(frame(0, 0, 0, 0, 0, person_at(101.25, 98.75)), "d1", 0));
        assert_eq!(f.people.len(), 1);
        assert!(f.people[0].pos[2] > 9.0, "{:?}", f.people[0].pos);
        // A track placed under the roof is not counted as looked at by a nadir frame.
        f.people.push(PersonTrack {
            pos: [30.5, 30.5, 0.0],
            score_sum: 0.9,
            hits: 1,
            looks: 1,
            views: vec![(0, Some(0))],
            segs: BTreeMap::from([(0, 1)]),
        });
        f.add_observation("b", obs(empty(), "d2", 0));
        assert_eq!(f.people[1].looks, 1, "hidden: no miss");
        assert_eq!(f.people[0].looks, 2, "visible: a miss");
    }

    #[test]
    fn hazard_ids_survive_zone_rebuild() {
        let mut f = Fuser::new(ANCHOR, 64);
        let vs = [("a", nadir(30.0, 30.0)), ("b", nadir(30.0, 30.0))];
        f.set_zone(1, &ground(), views(&vs));
        for n in ["a", "b"] {
            f.add_observation(n, FrameObservation::from_detections(&frame(2, 90, 110, 90, 110, vec![])));
        }
        let first = f.batch(1).detections[0].id.clone();
        assert!(first.starts_with("collapse-"), "{first}");
        f.set_zone(1, &ground(), views(&vs));
        assert_eq!(f.batch(1).detections[0].id, first);
        assert_eq!(all(&mut f).len(), 1);
    }

    #[test]
    fn observations_before_the_zone_are_fused_when_it_arrives() {
        let mut f = Fuser::new(ANCHOR, 64);
        for n in ["a", "b"] {
            assert!(f.add_observation(n, FrameObservation::from_detections(&frame(3, 90, 110, 90, 110, vec![]))).is_empty());
        }
        f.set_zone(0, &ground(), views(&[("a", nadir(30.0, 30.0)), ("b", nadir(30.0, 30.0))]));
        let b = f.batch(0);
        assert_eq!(b.detections.len(), 1);
        assert_eq!(b.detections[0].category, DetectionCategory::RoadBlocked as i32);
    }

    #[test]
    fn eviction_bounds_stored_frames() {
        let mut f = Fuser::new(ANCHOR, 2);
        for n in ["a", "b", "c"] {
            f.add_observation(n, FrameObservation::from_detections(&fire()));
        }
        assert_eq!(f.observations.len(), 2);
        assert!(!f.observations.contains_key("a"));
    }

    #[test]
    fn view_window_groups_one_drone_only() {
        let mut v = Vec::new();
        add_view(&mut v, (0, Some(0)));
        add_view(&mut v, (0, Some(VIEW_WINDOW_MS)));
        add_view(&mut v, (1, Some(0)));
        add_view(&mut v, (0, Some(VIEW_WINDOW_MS + 1)));
        add_view(&mut v, (2, None));
        add_view(&mut v, (2, None));
        assert_eq!(v, vec![(0, Some(0)), (1, Some(0)), (0, Some(VIEW_WINDOW_MS + 1)), (2, None), (2, None)]);
    }

    #[test]
    fn hull_of_square_corners() {
        let pts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5], [1.0, 0.0]];
        let h = convex_hull(&pts);
        assert_eq!(h, vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
    }
}
