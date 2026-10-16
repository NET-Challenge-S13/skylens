//! Detection fusion: per-frame UNet output projected onto the reconstructed
//! points and accumulated across frames.
//!
//! Pure math, no cumulus3d types. The engine adapts registered camera poses into
//! [`View`]s and zone clouds into point lists.
//!
//! - Hazards. Zone points are binned into [`CELL_M`] cells on the East/North
//!   plane. Every frame that sees a cell adds its class probabilities at the
//!   projected pixel, weighted by how decisive that pixel is (its largest class
//!   probability). The cell's class is the argmax of the weighted mean, so one
//!   frame that disagrees with several others is outvoted rather than painted on
//!   the map. Hazard cells of one class that touch (8-neighbourhood) form one
//!   zone with a convex outline.
//! - People. A person point is placed on the cell whose projection is nearest
//!   to it, then merged with an existing track within [`PERSON_MERGE_M`]. Every
//!   frame that had the track in view counts as a look, hit or miss, and the
//!   confidence is `score sum / looks`: the inter-frame interpolation.
//! - Ids are stable across updates and zone rebuilds (preview -> refined): a new
//!   hazard zone inherits the id of the previous zone of the same class it
//!   overlaps most; a person inherits the id of the nearest previous person.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use skylens_shared::geo::{Enu, GeoAnchor, enu_to_gps};
use skylens_shared::proto::model::v1::Gps;
use skylens_shared::proto::payload::v1::{Detection, DetectionBatch, DetectionCategory};

use crate::detect::{FrameDetections, HAZARD_CLASSES, PersonPoint};

/// Edge of one fusion cell on the East/North plane, metres.
pub const CELL_M: f64 = 1.0;
/// Largest edge of the stored probability grid. Frames are box-averaged down to
/// this so hundreds of frames fit in memory (4 bytes per grid pixel).
pub const GRID_MAX: u32 = 160;
/// A cell is a hazard when its fused class probability reaches this.
pub const HAZARD_MIN_PROB: f64 = 0.5;
/// ...and at least this many frames saw it.
pub const HAZARD_MIN_FRAMES: u32 = 2;
/// Person points below this score are ignored.
pub const PERSON_MIN_SCORE: f32 = 0.3;
/// A person point snaps to the nearest cell projection within this fraction of
/// the image's larger side.
pub const PERSON_PICK_FRAC: f64 = 0.03;
/// Person points closer than this on the ground plane are the same person.
pub const PERSON_MERGE_M: f64 = 2.0;
/// A person is reported once this many frames detected it...
pub const PERSON_MIN_FRAMES: u32 = 2;
/// ...and its interpolated confidence (score sum / looks) reaches this.
pub const PERSON_MIN_CONFIDENCE: f64 = 0.4;

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
}

impl FrameObservation {
    /// Box-averages the hazard probabilities to at most [`GRID_MAX`] per side.
    /// A probability buffer of the wrong length is dropped (no hazard evidence),
    /// the people are kept.
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
        Self {
            width: w,
            height: h,
            grid_w: gw,
            grid_h: gh,
            prob,
            people: d.people.clone(),
        }
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
        Some(std::array::from_fn(|c| {
            f64::from(self.prob[c * plane + at]) / 255.0
        }))
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
        f.debug_struct("View")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
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
        Self {
            width,
            height,
            rotation,
            translation,
            project,
        }
    }

    /// Distortion-free pinhole with focal `f` and the principal point at the
    /// image centre.
    pub fn pinhole(
        f: f64,
        width: u32,
        height: u32,
        rotation: [[f64; 3]; 3],
        translation: [f64; 3],
    ) -> Self {
        let (cx, cy) = (width as f64 / 2.0, height as f64 / 2.0);
        let project: Projector = Arc::new(move |p: [f64; 3]| {
            (p[2] > f64::EPSILON).then(|| [f * p[0] / p[2] + cx, f * p[1] / p[2] + cy])
        });
        Self::new(width, height, rotation, translation, project)
    }

    fn to_camera(&self, p: [f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        std::array::from_fn(|i| {
            r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2] + self.translation[i]
        })
    }

    /// Pixel of a world point, if it is in front of the camera and inside the image.
    pub fn project(&self, p: [f64; 3]) -> Option<[f64; 2]> {
        let c = self.to_camera(p);
        if c[2] <= f64::EPSILON {
            return None;
        }
        let px = (self.project)(c)?;
        let inside =
            px[0] >= 0.0 && px[1] >= 0.0 && px[0] < self.width as f64 && px[1] < self.height as f64;
        inside.then_some(px)
    }
}

type CellKey = (i32, i32);

fn cell_key(e: f64, n: f64) -> CellKey {
    ((e / CELL_M).floor() as i32, (n / CELL_M).floor() as i32)
}

#[derive(Debug, Clone)]
struct Cell {
    /// Mean ENU position of the points in the cell.
    center: [f64; 3],
    sum_w: f64,
    sum_wp: [f64; HAZARD_CLASSES],
    frames: u32,
}

#[derive(Debug, Clone)]
struct PersonTrack {
    /// Score-weighted mean ENU position.
    pos: [f64; 3],
    score_sum: f64,
    hits: u32,
    looks: u32,
}

impl PersonTrack {
    fn confidence(&self) -> f64 {
        if self.looks == 0 {
            0.0
        } else {
            (self.score_sum / self.looks as f64).min(1.0)
        }
    }
}

#[derive(Default)]
struct Zone {
    cells: BTreeMap<CellKey, Cell>,
    views: HashMap<String, View>,
    people: Vec<PersonTrack>,
    fused: HashSet<String>,
}

/// Previously reported ids of one segment.
#[derive(Default)]
struct Ids {
    hazards: Vec<(String, usize, HashSet<CellKey>)>,
    people: Vec<(String, [f64; 2])>,
    next: u32,
}

impl Ids {
    fn fresh(&mut self, segment: u32, kind: &str) -> String {
        self.next += 1;
        format!("s{segment}-{kind}-{}", self.next)
    }

    fn hazard(
        &mut self,
        segment: u32,
        class: usize,
        cells: &HashSet<CellKey>,
        used: &mut HashSet<String>,
    ) -> String {
        let best = self
            .hazards
            .iter()
            .enumerate()
            .filter(|(_, (id, c, _))| *c == class && !used.contains(id))
            .map(|(i, (_, _, prev))| (i, prev.intersection(cells).count()))
            .filter(|(_, overlap)| *overlap > 0)
            .max_by_key(|(_, overlap)| *overlap);
        let id = match best {
            Some((i, _)) => {
                self.hazards[i].2 = cells.clone();
                self.hazards[i].0.clone()
            }
            None => {
                let id = self.fresh(segment, class_label(class));
                self.hazards.push((id.clone(), class, cells.clone()));
                id
            }
        };
        used.insert(id.clone());
        id
    }

    fn person(&mut self, segment: u32, pos: [f64; 2], used: &mut HashSet<String>) -> String {
        let best = self
            .people
            .iter()
            .enumerate()
            .filter(|(_, (id, _))| !used.contains(id))
            .map(|(i, (_, p))| {
                (
                    i,
                    ((p[0] - pos[0]).powi(2) + (p[1] - pos[1]).powi(2)).sqrt(),
                )
            })
            .filter(|(_, d)| *d <= PERSON_MERGE_M)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        let id = match best {
            Some((i, _)) => {
                self.people[i].1 = pos;
                self.people[i].0.clone()
            }
            None => {
                let id = self.fresh(segment, "person");
                self.people.push((id.clone(), pos));
                id
            }
        };
        used.insert(id.clone());
        id
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

/// Fuses frame observations per segment (cumulus3d zone).
pub struct Fuser {
    anchor: GeoAnchor,
    /// Observations by stored image name, oldest first for eviction.
    observations: HashMap<String, Arc<FrameObservation>>,
    order: VecDeque<String>,
    max_observations: usize,
    zones: BTreeMap<u32, Zone>,
    ids: BTreeMap<u32, Ids>,
}

impl Fuser {
    /// `max_observations` bounds the stored frames (oldest dropped first).
    pub fn new(anchor: GeoAnchor, max_observations: usize) -> Self {
        Self {
            anchor,
            observations: HashMap::new(),
            order: VecDeque::new(),
            max_observations: max_observations.max(1),
            zones: BTreeMap::new(),
            ids: BTreeMap::new(),
        }
    }

    /// Replaces the geometry of a segment (a preview or a refined cloud) and
    /// re-fuses every stored observation of a frame registered in `views`.
    /// Ids already reported for the segment are kept where the result overlaps.
    pub fn set_zone(&mut self, segment: u32, points: &[[f64; 3]], views: HashMap<String, View>) {
        let mut acc: BTreeMap<CellKey, ([f64; 3], usize)> = BTreeMap::new();
        for p in points {
            if !(p[0].is_finite() && p[1].is_finite() && p[2].is_finite()) {
                continue;
            }
            let e = acc.entry(cell_key(p[0], p[1])).or_insert(([0.0; 3], 0));
            for i in 0..3 {
                e.0[i] += p[i];
            }
            e.1 += 1;
        }
        let cells = acc
            .into_iter()
            .map(|(k, (s, n))| {
                let center = s.map(|v| v / n as f64);
                (
                    k,
                    Cell {
                        center,
                        sum_w: 0.0,
                        sum_wp: [0.0; HAZARD_CLASSES],
                        frames: 0,
                    },
                )
            })
            .collect();
        let mut zone = Zone {
            cells,
            views,
            people: Vec::new(),
            fused: HashSet::new(),
        };
        // Fuse in arrival order so person tracks evolve the same way they would live.
        for name in &self.order {
            if let (Some(view), Some(obs)) =
                (zone.views.get(name).cloned(), self.observations.get(name))
            {
                fuse_frame(&mut zone, &view, obs);
                zone.fused.insert(name.clone());
            }
        }
        self.zones.insert(segment, zone);
    }

    /// Stores one frame's observation and fuses it into every segment that has
    /// the frame registered. Returns the segments that changed.
    pub fn add_observation(&mut self, name: &str, obs: FrameObservation) -> Vec<u32> {
        let obs = Arc::new(obs);
        if self
            .observations
            .insert(name.to_string(), obs.clone())
            .is_none()
        {
            self.order.push_back(name.to_string());
        }
        while self.order.len() > self.max_observations {
            if let Some(old) = self.order.pop_front() {
                self.observations.remove(&old);
            }
        }
        let mut changed = Vec::new();
        for (seg, zone) in self.zones.iter_mut() {
            if zone.fused.contains(name) {
                continue;
            }
            if let Some(view) = zone.views.get(name).cloned() {
                fuse_frame(zone, &view, &obs);
                zone.fused.insert(name.to_string());
                changed.push(*seg);
            }
        }
        changed
    }

    /// Segments known to the fuser.
    pub fn segments(&self) -> Vec<u32> {
        self.zones.keys().copied().collect()
    }

    /// Every detection of a segment as it stands now.
    pub fn batch(&mut self, segment: u32) -> DetectionBatch {
        let Some(zone) = self.zones.get(&segment) else {
            return DetectionBatch::default();
        };
        let anchor = self.anchor;
        let ids = self.ids.entry(segment).or_default();
        let mut used = HashSet::new();
        let mut detections = Vec::new();

        // Hazard cells.
        let mut hazard: BTreeMap<CellKey, (usize, f64)> = BTreeMap::new();
        for (k, c) in &zone.cells {
            if c.frames < HAZARD_MIN_FRAMES || c.sum_w <= 0.0 {
                continue;
            }
            let mean: [f64; HAZARD_CLASSES] = std::array::from_fn(|i| c.sum_wp[i] / c.sum_w);
            let (class, p) = mean
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap_or((0, 0.0));
            if class != 0 && p >= HAZARD_MIN_PROB {
                hazard.insert(*k, (class, p));
            }
        }
        // Connected components per class, in key order (deterministic).
        let mut seen: BTreeSet<CellKey> = BTreeSet::new();
        for (&start, &(class, _)) in &hazard {
            if seen.contains(&start) {
                continue;
            }
            let mut comp = Vec::new();
            let mut stack = vec![start];
            seen.insert(start);
            while let Some(k) = stack.pop() {
                comp.push(k);
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let nk = (k.0 + dx, k.1 + dy);
                        if hazard.get(&nk).is_some_and(|(c, _)| *c == class) && seen.insert(nk) {
                            stack.push(nk);
                        }
                    }
                }
            }
            comp.sort_unstable();
            let set: HashSet<CellKey> = comp.iter().copied().collect();
            let id = ids.hazard(segment, class, &set, &mut used);
            let n = comp.len() as f64;
            let mut centroid = [0.0; 3];
            let mut conf = 0.0;
            let mut frames = 0;
            let mut corners = Vec::with_capacity(comp.len() * 4);
            for k in &comp {
                let c = &zone.cells[k];
                for i in 0..3 {
                    centroid[i] += c.center[i] / n;
                }
                conf += hazard[k].1 / n;
                frames = frames.max(c.frames);
                let (e0, n0) = (k.0 as f64 * CELL_M, k.1 as f64 * CELL_M);
                corners.extend([
                    [e0, n0],
                    [e0 + CELL_M, n0],
                    [e0 + CELL_M, n0 + CELL_M],
                    [e0, n0 + CELL_M],
                ]);
            }
            let outline = convex_hull(&corners)
                .into_iter()
                .map(|p| to_gps([p[0], p[1], centroid[2]], anchor))
                .collect();
            detections.push(Detection {
                id,
                category: class_category(class) as i32,
                gps: Some(to_gps(centroid, anchor)),
                confidence: conf,
                label: class_label(class).to_string(),
                segment,
                outline,
                frames,
            });
        }

        // People.
        for t in &zone.people {
            let conf = t.confidence();
            if t.hits < PERSON_MIN_FRAMES || conf < PERSON_MIN_CONFIDENCE {
                continue;
            }
            let id = ids.person(segment, [t.pos[0], t.pos[1]], &mut used);
            detections.push(Detection {
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
        DetectionBatch { detections }
    }
}

fn to_gps(p: [f64; 3], anchor: GeoAnchor) -> Gps {
    enu_to_gps(
        Enu {
            e: p[0],
            n: p[1],
            u: p[2],
        },
        anchor,
    )
    .into()
}

/// Adds one frame to a zone: hazard evidence on every cell in view, then people.
fn fuse_frame(zone: &mut Zone, view: &View, obs: &FrameObservation) {
    // View pixels -> observation pixels (the reconstruction may work on a rescaled image).
    let sx = obs.width as f64 / view.width.max(1) as f64;
    let sy = obs.height as f64 / view.height.max(1) as f64;
    let mut projected: Vec<(CellKey, [f64; 2])> = Vec::new();
    for (k, cell) in zone.cells.iter_mut() {
        let Some(px) = view.project(cell.center) else {
            continue;
        };
        projected.push((*k, px));
        let Some(p) = obs.hazard_at(px[0] * sx, px[1] * sy) else {
            continue;
        };
        let w = p.iter().copied().fold(0.0, f64::max);
        if w <= 0.0 {
            continue;
        }
        cell.sum_w += w;
        for i in 0..HAZARD_CLASSES {
            cell.sum_wp[i] += w * p[i];
        }
        cell.frames += 1;
    }

    let before = zone.people.len();
    let mut matched = vec![false; before];
    let radius = PERSON_PICK_FRAC * view.width.max(view.height) as f64;
    for person in obs.people.iter().filter(|p| p.score >= PERSON_MIN_SCORE) {
        let target = [person.x as f64 / sx, person.y as f64 / sy];
        let pick = projected
            .iter()
            .map(|(k, px)| {
                (
                    k,
                    ((px[0] - target[0]).powi(2) + (px[1] - target[1]).powi(2)).sqrt(),
                )
            })
            .filter(|(_, d)| *d <= radius)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        let Some((k, _)) = pick else { continue };
        let pos = zone.cells[k].center;
        let score = f64::from(person.score.clamp(0.0, 1.0));
        let near = zone
            .people
            .iter()
            .enumerate()
            .map(|(i, t)| {
                (
                    i,
                    ((t.pos[0] - pos[0]).powi(2) + (t.pos[1] - pos[1]).powi(2)).sqrt(),
                )
            })
            .filter(|(i, d)| *d <= PERSON_MERGE_M && !(*i < before && matched[*i]))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        match near {
            Some((i, _)) => {
                let t = &mut zone.people[i];
                let w = t.score_sum + score;
                if w > 0.0 {
                    for j in 0..3 {
                        t.pos[j] = (t.pos[j] * t.score_sum + pos[j] * score) / w;
                    }
                }
                t.score_sum = w;
                t.hits += 1;
                t.looks += 1;
                if i < before {
                    matched[i] = true;
                }
            }
            None => zone.people.push(PersonTrack {
                pos,
                score_sum: score,
                hits: 1,
                looks: 1,
            }),
        }
    }
    // A track this frame could see but did not detect is a miss.
    for (i, t) in zone.people.iter_mut().take(before).enumerate() {
        if !matched[i] && view.project(t.pos).is_some() {
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
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
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

    const ANCHOR: GeoAnchor = GeoGps {
        lat: 36.3685,
        lon: 127.3475,
        alt: 30.0,
    };
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

    /// Frame of size W x H, class `class` with probability 0.9 inside the
    /// pixel box, normal elsewhere.
    fn frame(
        class: usize,
        x0: usize,
        x1: usize,
        y0: usize,
        y1: usize,
        people: Vec<PersonPoint>,
    ) -> FrameDetections {
        let plane = (W * H) as usize;
        let mut prob = vec![0.0f32; HAZARD_CLASSES * plane];
        for y in 0..H as usize {
            for x in 0..W as usize {
                let inside = x >= x0 && x < x1 && y >= y0 && y < y1;
                let i = y * W as usize + x;
                if inside && class != 0 {
                    prob[class * plane + i] = 0.9;
                    prob[i] = 0.1;
                } else {
                    prob[i] = 0.95;
                    prob[plane + i] = 0.05;
                }
            }
        }
        FrameDetections {
            width: W,
            height: H,
            hazard_prob: prob,
            people,
        }
    }

    fn views(names: &[(&str, View)]) -> HashMap<String, View> {
        names
            .iter()
            .map(|(n, v)| (n.to_string(), v.clone()))
            .collect()
    }

    fn enu_of(d: &Detection) -> Enu {
        let g = d.gps.clone().unwrap();
        gps_to_enu(
            GeoGps {
                lat: g.lat,
                lon: g.lon,
                alt: g.alt,
            },
            ANCHOR,
        )
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
        let d = FrameDetections {
            width: 10,
            height: 10,
            hazard_prob: vec![0.5; 7],
            people: vec![],
        };
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
    }

    #[test]
    fn hazard_needs_two_frames_then_appears_with_outline() {
        let mut f = Fuser::new(ANCHOR, 64);
        let v1 = nadir(30.0, 30.0);
        let v2 = nadir(31.0, 30.0);
        f.set_zone(0, &ground(), views(&[("a", v1), ("b", v2)]));
        // Fire in pixels 90..110 (about 10 m wide around the centre).
        f.add_observation(
            "a",
            FrameObservation::from_detections(&frame(1, 90, 110, 90, 110, vec![])),
        );
        assert!(f.batch(0).detections.is_empty(), "one frame is not enough");
        let changed = f.add_observation(
            "b",
            FrameObservation::from_detections(&frame(1, 88, 108, 90, 110, vec![])),
        );
        assert_eq!(changed, vec![0]);
        let b = f.batch(0);
        assert_eq!(b.detections.len(), 1, "{:?}", b.detections);
        let d = &b.detections[0];
        assert_eq!(d.category, DetectionCategory::Fire as i32);
        assert_eq!(d.label, "fire");
        assert_eq!(d.segment, 0);
        assert!(d.frames >= 2);
        assert!(d.confidence >= HAZARD_MIN_PROB);
        assert!(d.outline.len() >= 4);
        let c = enu_of(d);
        assert!(
            (c.e - 30.0).abs() < 2.0 && (c.n - 30.0).abs() < 2.0,
            "{c:?}"
        );
    }

    #[test]
    fn one_disagreeing_frame_is_outvoted() {
        let mut f = Fuser::new(ANCHOR, 64);
        let names = ["a", "b", "c", "d"];
        let vs: Vec<(&str, View)> = names.iter().map(|n| (*n, nadir(30.0, 30.0))).collect();
        f.set_zone(0, &ground(), views(&vs));
        f.add_observation(
            "a",
            FrameObservation::from_detections(&frame(1, 90, 110, 90, 110, vec![])),
        );
        for n in &names[1..] {
            f.add_observation(
                n,
                FrameObservation::from_detections(&frame(0, 0, 0, 0, 0, vec![])),
            );
        }
        assert!(f.batch(0).detections.is_empty());
    }

    #[test]
    fn person_needs_repeat_sightings() {
        let mut f = Fuser::new(ANCHOR, 64);
        let names = ["a", "b", "c", "d", "e"];
        let vs: Vec<(&str, View)> = names.iter().map(|n| (*n, nadir(30.0, 30.0))).collect();
        f.set_zone(3, &ground(), views(&vs));
        let seen = || {
            vec![PersonPoint {
                x: 120.0,
                y: 80.0,
                score: 0.9,
            }]
        };
        f.add_observation(
            "a",
            FrameObservation::from_detections(&frame(0, 0, 0, 0, 0, seen())),
        );
        assert!(f.batch(3).detections.is_empty());
        f.add_observation(
            "b",
            FrameObservation::from_detections(&frame(0, 0, 0, 0, 0, seen())),
        );
        let b = f.batch(3);
        assert_eq!(b.detections.len(), 1);
        let d = &b.detections[0];
        assert_eq!(d.category, DetectionCategory::Person as i32);
        assert_eq!(d.frames, 2);
        assert!(d.outline.is_empty());
        let c = enu_of(d);
        // Pixel (120, 80) is 10 m East and 10 m North of the camera.
        assert!(
            (c.e - 40.0).abs() < 1.5 && (c.n - 40.0).abs() < 1.5,
            "{c:?}"
        );
        let id = d.id.clone();
        // Three misses lower the interpolated confidence (1.8 / 5) below the threshold.
        for n in ["c", "d", "e"] {
            f.add_observation(
                n,
                FrameObservation::from_detections(&frame(0, 0, 0, 0, 0, vec![])),
            );
        }
        assert!(f.batch(3).detections.is_empty());
        // Rebuilding the zone (refined cloud) keeps the id.
        f.set_zone(3, &ground(), views(&vs[..2]));
        let again = f.batch(3);
        assert_eq!(again.detections.len(), 1);
        assert_eq!(again.detections[0].id, id);
    }

    #[test]
    fn single_sighting_among_many_is_not_reported() {
        let mut f = Fuser::new(ANCHOR, 64);
        let names = ["a", "b", "c", "d", "e"];
        let vs: Vec<(&str, View)> = names.iter().map(|n| (*n, nadir(30.0, 30.0))).collect();
        f.set_zone(0, &ground(), views(&vs));
        f.add_observation(
            "a",
            FrameObservation::from_detections(&frame(
                0,
                0,
                0,
                0,
                0,
                vec![PersonPoint {
                    x: 100.0,
                    y: 100.0,
                    score: 0.95,
                }],
            )),
        );
        for n in &names[1..] {
            f.add_observation(
                n,
                FrameObservation::from_detections(&frame(0, 0, 0, 0, 0, vec![])),
            );
        }
        assert!(f.batch(0).detections.is_empty());
    }

    #[test]
    fn hazard_ids_survive_zone_rebuild() {
        let mut f = Fuser::new(ANCHOR, 64);
        let vs = [("a", nadir(30.0, 30.0)), ("b", nadir(30.0, 30.0))];
        f.set_zone(1, &ground(), views(&vs));
        for n in ["a", "b"] {
            f.add_observation(
                n,
                FrameObservation::from_detections(&frame(2, 90, 110, 90, 110, vec![])),
            );
        }
        let first = f.batch(1).detections[0].id.clone();
        assert!(first.starts_with("s1-collapse-"), "{first}");
        f.set_zone(1, &ground(), views(&vs));
        assert_eq!(f.batch(1).detections[0].id, first);
    }

    #[test]
    fn observations_before_the_zone_are_fused_when_it_arrives() {
        let mut f = Fuser::new(ANCHOR, 64);
        for n in ["a", "b"] {
            assert!(
                f.add_observation(
                    n,
                    FrameObservation::from_detections(&frame(3, 90, 110, 90, 110, vec![]))
                )
                .is_empty()
            );
        }
        f.set_zone(
            0,
            &ground(),
            views(&[("a", nadir(30.0, 30.0)), ("b", nadir(30.0, 30.0))]),
        );
        let b = f.batch(0);
        assert_eq!(b.detections.len(), 1);
        assert_eq!(
            b.detections[0].category,
            DetectionCategory::RoadBlocked as i32
        );
    }

    #[test]
    fn eviction_bounds_stored_frames() {
        let mut f = Fuser::new(ANCHOR, 2);
        for n in ["a", "b", "c"] {
            f.add_observation(
                n,
                FrameObservation::from_detections(&frame(1, 90, 110, 90, 110, vec![])),
            );
        }
        assert_eq!(f.observations.len(), 2);
        assert!(!f.observations.contains_key("a"));
    }

    #[test]
    fn hull_of_square_corners() {
        let pts = [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0],
        ];
        let h = convex_hull(&pts);
        assert_eq!(h, vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
    }
}
