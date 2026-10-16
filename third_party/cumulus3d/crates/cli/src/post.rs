/*
 * post.rs
 *
 * Copyright (c) 2026 ParkSangWoo
 *
 * Author(s):
 *
 *      ParkSangWoo <dev.parksangwoo@gmail.com>
 *
 *
 * This file is part of cumulus3d.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//! Post-processing: performs the Python part at the end of the script (preview alignment, per-event snapshots, manifest) and
//! incremental_reanchor.py
//! with the align crate. The model uses the in-memory Reconstruction as is.

use crate::util::{py_float, Json, Logger};
use cumulus3d_align::shared::NameFilter;
use cumulus3d_align::RobustUmeyamaOptions;
use cumulus3d_align::{
    compose_snapshot, decimate, merge_clouds, robust_umeyama, select, shared_point_correspondences, transform_cloud, KdTree,
    SharedPointOptions, SnapshotOptions,
};
use cumulus3d_core::io::{read_ply, write_ply, PlyLayout, PointCloud};
use cumulus3d_core::{Reconstruction, Sim3, Vec3};
use rayon::prelude::*;
use std::collections::HashSet;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Post-processing input (based on PLY paths and event strings).
pub struct PostInput<'a> {
    /// Output folder.
    pub out: &'a Path,
    /// Zone size (number of positions).
    pub span: usize,
    /// Zone overlap (number of positions).
    pub ovl: usize,
    /// (seconds from the first event, event string).
    pub events: Vec<(f64, String)>,
    /// Per-zone preview models.
    pub previews: &'a BTreeMap<usize, Reconstruction>,
    /// Per-zone refined models.
    pub refined: &'a BTreeMap<usize, Reconstruction>,
    /// Per-zone preview PLY paths.
    pub preview_ply: &'a BTreeMap<usize, PathBuf>,
    /// Per-zone refined PLY paths.
    pub refined_ply: &'a BTreeMap<usize, PathBuf>,
    /// Image name -> position.
    pub position: &'a (dyn Fn(&str) -> Option<usize> + Sync),
}

/// Python `tof`: the time of the first event whose first words match or which shares the prefix.
fn tof(events: &[(f64, String)], pat: &str) -> Option<f64> {
    events.iter().find(|(_, s)| s.split(" pos[").next() == Some(pat) || s.starts_with(pat)).map(|(t, _)| *t)
}

/// Robust Umeyama result (s, R, t, residual median, inlier count).
pub(crate) struct Fit {
    pub(crate) sim3: Sim3,
    pub(crate) median: f64,
    pub(crate) inliers: usize,
}

/// Fit a_to_b using the shared 3D points of two models (same-name images, same 2D index).
pub(crate) fn corr_points(a: &Reconstruction, b: &Reconstruction, filter: Option<NameFilter>) -> (Vec<Vec3>, Vec<Vec3>) {
    let opts = SharedPointOptions { position_range: None, name_filter: filter };
    let pairs = shared_point_correspondences(a, b, &opts);
    let mut pa = Vec::with_capacity(pairs.len());
    let mut pb = Vec::with_capacity(pairs.len());
    for (i, j) in pairs {
        if let (Some(x), Some(y)) = (a.point3d(i), b.point3d(j)) {
            pa.push(x.xyz);
            pb.push(y.xyz);
        }
    }
    (pa, pb)
}

pub(crate) fn fit(pa: &[Vec3], pb: &[Vec3]) -> Option<Fit> {
    let r = robust_umeyama(pa, pb, &RobustUmeyamaOptions::default())?;
    Some(Fit { sim3: r.sim3, median: r.median_residual, inliers: r.num_inliers })
}

/// Robust Sim3 taking `a` into the frame of `b` from their shared 3D points, restricted by `filter` when given. Returns the
/// fit and the number of shared pairs, or None with fewer than `min_pairs` pairs (or when the fit fails).
pub(crate) fn fit_shared_points(
    a: &Reconstruction,
    b: &Reconstruction,
    filter: Option<NameFilter>,
    min_pairs: usize,
) -> Option<(Fit, usize)> {
    let (pa, pb) = corr_points(a, b, filter);
    if pa.len() < min_pairs {
        return None;
    }
    fit(&pa, &pb).map(|f| (f, pa.len()))
}

fn transformed(c: &PointCloud, t: &Sim3) -> PointCloud {
    let mut c = c.clone();
    transform_cloud(&mut c, t);
    c
}

fn write(path: &Path, c: &PointCloud) -> Result<(), String> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    write_ply(path, c, PlyLayout::XyzRgbNormal).map_err(|e| format!("{}: {e}", path.display()))
}

fn basename(p: &Path) -> String {
    p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()
}

/// Median of the nearest-neighbor distances < 5 m from the points of the earlier cloud to the later cloud (1/20 subsampled).
fn overlap_median(prev: &PointCloud, cur: &PointCloud) -> Option<(f64, usize)> {
    if prev.is_empty() {
        return None;
    }
    let tree = KdTree::from_f32(&prev.positions);
    let q: Vec<[f64; 3]> = cur.positions.iter().step_by(20).map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect();
    let mut d: Vec<f64> = tree.nearest_many(&q).into_iter().flatten().map(|(_, d2)| d2.sqrt()).filter(|d| *d < 5.0).collect();
    if d.is_empty() {
        return None;
    }
    let n = d.len();
    d.sort_by(f64::total_cmp);
    let m = if n % 2 == 1 { d[n / 2] } else { 0.5 * (d[n / 2 - 1] + d[n / 2]) };
    Some((m, n))
}

/// Per-zone event time (seconds from the first event). None if absent.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ZoneTimes {
    /// `region K arrived`.
    pub arrived: Option<f64>,
    /// `preview K ready`.
    pub preview_ready: Option<f64>,
    /// `refined K pose_ready`.
    pub refined_pose: Option<f64>,
    /// `refined K ready`.
    pub refined_ready: Option<f64>,
}

/// One zone point cloud: the output file name (same name as in full/) and the point cloud.
pub struct ZoneCloud<'a> {
    /// Output file name.
    pub file: String,
    /// Point cloud.
    pub cloud: &'a PointCloud,
}

/// Post-processing with in-memory input (zone results collected from events). Shared by [`run`] and the default hook.
pub struct PostZones<'a> {
    /// Output folder.
    pub out: &'a Path,
    /// Preview zones that have a dense point cloud.
    pub preview: BTreeMap<usize, ZoneCloud<'a>>,
    /// Refined zones that have a dense point cloud.
    pub refined: BTreeMap<usize, ZoneCloud<'a>>,
    /// Preview models (zone 0 is the GPS-aligned one).
    pub preview_models: BTreeMap<usize, &'a Reconstruction>,
    /// Refined models (BA + GPS alignment).
    pub refined_models: BTreeMap<usize, &'a Reconstruction>,
    /// Per-zone event time.
    pub times: BTreeMap<usize, ZoneTimes>,
    /// Position range [lo, hi) used for aligning the preview of zone K (>= 1) = the overlap with the previous zone.
    pub windows: BTreeMap<usize, (usize, usize)>,
    /// Image name -> position.
    pub position: &'a (dyn Fn(&str) -> Option<usize> + Sync),
}

/// Stable-display outputs: every refined zone exactly as shown (`snapshots/event_NN_TTTT.Ts_refined_K.ply`, display
/// frame, in order of arrival) and the same clouds in ENU (`final_frame/refined/`). Each zone gets its own Sim3 from its
/// refined model (`models`) to the GPS-aligned final model (`enu`), fitted on shared 3D points; with too few pairs (or no
/// model for the zone) one global Sim3 from the display model to `enu` (projection centers of shared images) is used.
/// `final_frame/merged.ply` is the thinned (1/[`MERGED_STRIDE`]) and overlap-deduplicated union of the ENU clouds.
pub fn run_stable(
    out: &Path,
    refined: &BTreeMap<usize, ZoneCloud<'_>>,
    models: &BTreeMap<usize, &Reconstruction>,
    times: &BTreeMap<usize, ZoneTimes>,
    display: Option<&Reconstruction>,
    enu: Option<&Reconstruction>,
    log: &Logger,
) -> Result<(), String> {
    let snapdir = out.join("snapshots");
    std::fs::create_dir_all(&snapdir).map_err(|e| e.to_string())?;
    let mut order: Vec<(f64, usize)> = refined.keys().map(|k| (times.get(k).and_then(|t| t.refined_ready).unwrap_or(0.0), *k)).collect();
    order.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (i, (t, k)) in order.iter().enumerate() {
        write(&snapdir.join(format!("event_{:02}_{t:06.1}s_refined_{k}.ply", i + 1)), refined[k].cloud)?;
    }
    log.line(&format!("[post] stable display: 구역 {} 개를 표시한 그대로 snapshots/ 에 기록", order.len()));
    let tf = match (display, enu) {
        (Some(d), Some(e)) => {
            let (mut src, mut dst) = (Vec::new(), Vec::new());
            for id in d.registered_images() {
                let Some(name) = d.image(id).map(|im| im.name.clone()) else { continue };
                let Some(eid) = e.image_by_name(&name).map(|im| im.image_id) else { continue };
                if let (Some(a), Some(b)) = (d.projection_center(id), e.projection_center(eid)) {
                    src.push(a);
                    dst.push(b);
                }
            }
            cumulus3d_align::umeyama(&src, &dst, true)
        }
        _ => None,
    };
    if let Some(t) = &tf {
        log.line(&format!("[post] stable display: ENU 변환 축척 {:.4}", t.scale));
    }
    let Some(enu) = enu else {
        log.line("[post] stable display: ENU 변환 없음(최종 GPS 정렬 실패) — final_frame/ 생략");
        return Ok(());
    };
    let mut placed: BTreeMap<usize, PointCloud> = BTreeMap::new();
    for (k, zc) in refined {
        let own = models.get(k).and_then(|m| {
            let (pa, pb) = corr_points(m, enu, None);
            if pa.len() >= LiveOptions::default().min_pairs {
                fit(&pa, &pb).map(|f| (f, pa.len()))
            } else {
                None
            }
        });
        let t = match (own, tf) {
            (Some((f, n)), _) => {
                log.line(&format!("[post] final_frame 구역 {k}: 공유 점 {n}, 잔차 중앙 {:.3}m, 축척 {:.4}", f.median, f.sim3.scale));
                f.sim3
            }
            (None, Some(t)) => {
                log.line(&format!("[post] final_frame 구역 {k}: 공유 점 부족 — 전체 변환 사용(축척 {:.4})", t.scale));
                t
            }
            (None, None) => {
                log.line(&format!("[post] final_frame 구역 {k}: 변환 없음 — 건너뜀"));
                continue;
            }
        };
        let c = transformed(zc.cloud, &t);
        write(&out.join("final_frame/refined").join(&zc.file), &c)?;
        placed.insert(*k, c);
    }
    let clouds: BTreeMap<usize, &PointCloud> = placed.iter().map(|(k, c)| (*k, c)).collect();
    let merged = LiveDisplay::new(LiveOptions { stride: MERGED_STRIDE, relocate: false, ..LiveOptions::default() })
        .compose(&clouds, &BTreeMap::new());
    write(&out.join("final_frame/merged.ply"), &merged)?;
    log.line(&format!("[post] stable display: final_frame/refined/ {} 개, final_frame/merged.ply 점 {}", placed.len(), merged.len()));
    Ok(())
}

/// Thinning of `final_frame/merged.ply` in stable display (1/2 of each zone's points, then overlap dedupe).
pub const MERGED_STRIDE: usize = 2;

/// Settings for [`LiveDisplay`].
#[derive(Clone, Debug, PartialEq)]
pub struct LiveOptions {
    /// Overlap dedupe voxel edge = this x the median nearest-neighbour spacing of the first thinned zone cloud (so it does
    /// not depend on the scale of the frame). A point of a zone is dropped when an earlier zone already owns its voxel.
    pub voxel_factor: f64,
    /// Every zone is thinned to 1/N before merging.
    pub stride: usize,
    /// Fewer shared 3D points than this: the zone is not moved.
    pub min_pairs: usize,
    /// Place every zone into the newest refined model's frame (fit on shared 3D points). Off in stable display, where all
    /// zone clouds are already in one frame: then [`LiveDisplay::compose`] only thins and drops the overlap duplicates.
    pub relocate: bool,
}

impl Default for LiveOptions {
    fn default() -> Self {
        Self { voxel_factor: 2.0, stride: 6, min_pairs: 20, relocate: true }
    }
}

/// Smallest voxel edge of the overlap dedupe (guards against a zero spacing from duplicate points).
const MIN_VOXEL: f64 = 1e-9;

/// Integer voxel of a point: `floor(p / edge)` per axis.
type VoxelKey = [i64; 3];

fn voxel_key(p: &[f32; 3], inv: f64) -> VoxelKey {
    [(p[0] as f64 * inv).floor() as i64, (p[1] as f64 * inv).floor() as i64, (p[2] as f64 * inv).floor() as i64]
}

/// Multiplicative hasher for [`VoxelKey`] (the default SipHash is too slow for millions of keys).
#[derive(Default)]
struct VoxelHasher(u64);

impl Hasher for VoxelHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.write_u64(*b as u64);
        }
    }

    fn write_i64(&mut self, v: i64) {
        self.write_u64(v as u64);
    }

    fn write_u64(&mut self, v: u64) {
        let h = (self.0.rotate_left(26) ^ v).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 = h ^ (h >> 29);
    }
}

/// Median nearest-neighbour distance of a strided sample (about 4000 points) against the whole cloud. The sample points
/// are left out of the tree so they do not find themselves.
fn median_spacing(pts: &[[f32; 3]]) -> f64 {
    const SAMPLE: usize = 4000;
    let step = (pts.len() / SAMPLE).max(2);
    let q: Vec<[f64; 3]> = pts.iter().step_by(step).map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect();
    let rest: Vec<[f32; 3]> = pts.iter().enumerate().filter(|(i, _)| i % step != 0).map(|(_, p)| *p).collect();
    let mut d: Vec<f64> = KdTree::from_f32(&rest).nearest_many(&q).into_iter().flatten().map(|(_, d2)| d2.sqrt()).collect();
    if d.is_empty() {
        return 0.0;
    }
    d.sort_by(f64::total_cmp);
    d[d.len() / 2]
}

/// splitmix64 finalizer: the fixed integer hash behind the key-based thinning.
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// 1/stride thinning. With point keys a point stays iff `splitmix64(key) % stride == 0`, so the same depth-map pixel
/// survives (or not) in every re-densification of a zone; without keys the index rule of [`decimate`] (index % stride == 0).
fn thin(cloud: &PointCloud, stride: usize) -> PointCloud {
    if stride <= 1 {
        return cloud.clone();
    }
    if !cloud.has_keys() || cloud.keys.len() != cloud.len() {
        return decimate(cloud, stride);
    }
    let keep: Vec<bool> = cloud.keys.par_iter().map(|&k| splitmix64(k).is_multiple_of(stride as u64)).collect();
    select(cloud, &keep)
}

/// Display cloud built the same way as the final output: every refined zone is placed in the frame of the newest
/// refined model with its own Sim3 (robust fit on shared 3D points; skipped with `relocate` off), so zones meet without seams; the
/// overlap of zone k
/// with the earlier zones (voxel ownership, no empty strip at the seam) is dropped so it is not drawn twice. Fits are cached per (zone,
/// reference) pair.
///
/// Ownership is sticky: the first compose in which a voxel holds points assigns it to the lowest zone that has points there,
/// and it keeps that owner for the session, so re-densifying or moving a shown zone does not hand its voxels to another zone
/// (the one exception: when the owner has no points left in the voxel, the lowest zone that has some takes it over, so no
/// hole opens). Thinning is key-based when the clouds carry point keys (`splitmix64(key) % stride == 0`).
#[derive(Default)]
pub struct LiveDisplay {
    opts: LiveOptions,
    fits: HashMap<(usize, usize), Sim3>,
    /// Voxel edge, estimated on the first compose with data and kept for the session so that ownership never flips.
    voxel: Option<f64>,
    /// Voxel → owner zone (sticky, see above).
    owners: HashMap<VoxelKey, usize, BuildHasherDefault<VoxelHasher>>,
}

impl LiveDisplay {
    /// With the given settings.
    pub fn new(opts: LiveOptions) -> Self {
        Self { opts, fits: HashMap::new(), voxel: None, owners: HashMap::default() }
    }

    /// Turns relocation (see [`LiveOptions::relocate`]) on or off.
    pub fn set_relocate(&mut self, on: bool) {
        self.opts.relocate = on;
    }

    /// Forgets the voxel size (only when the whole session restarts); the next compose estimates it again.
    pub fn reset_voxel(&mut self) {
        self.voxel = None;
        self.owners.clear();
    }

    /// Forgets every fit that involves zone `k` (its model changed or was dropped).
    pub fn forget(&mut self, k: usize) {
        self.fits.retain(|(a, b), _| *a != k && *b != k);
    }

    /// Sim3 from zone `j`'s model to zone `cur`'s model (identity when they are the same, a model is missing or too few
    /// points are shared).
    fn fit(&mut self, j: usize, cur: usize, models: &BTreeMap<usize, &Reconstruction>) -> Sim3 {
        if j == cur {
            return Sim3::identity();
        }
        if let Some(t) = self.fits.get(&(j, cur)) {
            return *t;
        }
        let (Some(a), Some(b)) = (models.get(&j), models.get(&cur)) else { return Sim3::identity() };
        let (pa, pb) = corr_points(a, b, None);
        let t = if pa.len() >= self.opts.min_pairs { fit(&pa, &pb).map(|f| f.sim3) } else { None };
        let t = t.unwrap_or_else(Sim3::identity);
        self.fits.insert((j, cur), t);
        t
    }

    /// Merged display cloud of `clouds` (zone → refined cloud) in the frame of the newest zone's model.
    pub fn compose(&mut self, clouds: &BTreeMap<usize, &PointCloud>, models: &BTreeMap<usize, &Reconstruction>) -> PointCloud {
        self.compose_zoned(clouds, models).0
    }

    /// [`compose`](Self::compose) plus the owner zone of every point of the result (same order). Points with equal keys
    /// (the overlap images of two zones densify the same pixel) are kept once, in the lowest zone.
    pub fn compose_zoned(
        &mut self,
        clouds: &BTreeMap<usize, &PointCloud>,
        models: &BTreeMap<usize, &Reconstruction>,
    ) -> (PointCloud, Vec<usize>) {
        let Some(&cur) = clouds.keys().next_back() else { return (PointCloud::default(), Vec::new()) };
        // 1) Thin and place every zone; voxel of every point.
        struct Part {
            zone: usize,
            cloud: PointCloud,
            vox: Vec<VoxelKey>,
        }
        let mut parts: Vec<Part> = Vec::with_capacity(clouds.len());
        for (&j, c) in clouds {
            let t = if self.opts.relocate { self.fit(j, cur, models) } else { Sim3::identity() };
            let mut d = thin(c, self.opts.stride);
            transform_cloud(&mut d, &t);
            if d.is_empty() {
                continue;
            }
            let factor = self.opts.voxel_factor;
            let v = *self.voxel.get_or_insert_with(|| (factor * median_spacing(&d.positions)).max(MIN_VOXEL));
            let inv = 1.0 / v;
            let vox: Vec<VoxelKey> = d.positions.par_iter().map(|p| voxel_key(p, inv)).collect();
            parts.push(Part { zone: j, cloud: d, vox });
        }
        // 2) Ownership: keep an owner that still has points in the voxel, else the lowest zone present takes it.
        type VoxSet = HashSet<VoxelKey, BuildHasherDefault<VoxelHasher>>;
        let sets: Vec<VoxSet> = parts.par_iter().map(|p| p.vox.iter().copied().collect()).collect();
        let idx: HashMap<usize, usize> = parts.iter().enumerate().map(|(i, p)| (p.zone, i)).collect();
        let mut lowest: HashMap<VoxelKey, usize, BuildHasherDefault<VoxelHasher>> = HashMap::default();
        for p in &parts {
            for &v in &p.vox {
                lowest.entry(v).or_insert(p.zone);
            }
        }
        for (v, low) in lowest {
            match self.owners.get(&v) {
                Some(o) if idx.get(o).is_some_and(|&i| sets[i].contains(&v)) => {}
                _ => {
                    self.owners.insert(v, low);
                }
            }
        }
        // 3) Keep each zone's points in the voxels it owns, then drop repeated keys (lowest zone first).
        let owners = &self.owners;
        let mut seen: HashSet<u64, BuildHasherDefault<VoxelHasher>> = HashSet::default();
        let all_keys = parts.iter().all(|p| p.cloud.has_keys());
        let mut out_parts: Vec<PointCloud> = Vec::with_capacity(parts.len());
        let mut zones: Vec<usize> = Vec::new();
        for p in parts {
            let mut keep: Vec<bool> = p.vox.par_iter().map(|v| owners.get(v) == Some(&p.zone)).collect();
            if all_keys {
                for (k, kp) in p.cloud.keys.iter().zip(keep.iter_mut()) {
                    if *kp && !seen.insert(*k) {
                        *kp = false;
                    }
                }
            }
            let c = select(&p.cloud, &keep);
            zones.extend(std::iter::repeat_n(p.zone, c.len()));
            out_parts.push(c);
        }
        (merge_clouds(&out_parts.iter().collect::<Vec<_>>()), zones)
    }
}

/// Whole post-processing (input: PLY paths and event strings). The summary goes to run.log (+ standard output).
pub fn run(inp: &PostInput, log: &Logger) -> Result<(), String> {
    let ev = &inp.events;
    let mut pv: BTreeMap<usize, PointCloud> = BTreeMap::new();
    for (k, p) in inp.preview_ply {
        pv.insert(*k, read_ply(p).map_err(|e| format!("{}: {e}", p.display()))?);
    }
    let mut rf: BTreeMap<usize, PointCloud> = BTreeMap::new();
    for (k, p) in inp.refined_ply {
        rf.insert(*k, read_ply(p).map_err(|e| format!("{}: {e}", p.display()))?);
    }
    let mut times: BTreeMap<usize, ZoneTimes> = BTreeMap::new();
    for k in pv.keys() {
        let t = times.entry(*k).or_default();
        t.preview_ready = tof(ev, &format!("preview {k} ready"));
        t.arrived = tof(ev, &format!("region {k} arrived"));
    }
    for k in rf.keys() {
        let t = times.entry(*k).or_default();
        t.refined_ready = tof(ev, &format!("refined {k} ready"));
        t.refined_pose = tof(ev, &format!("refined {k} pose_ready"));
    }
    let windows = pv
        .keys()
        .map(|k| {
            let start = k * inp.span;
            (*k, (start.saturating_sub(inp.ovl), start + inp.ovl))
        })
        .collect();
    let z = PostZones {
        out: inp.out,
        preview: pv.iter().map(|(k, c)| (*k, ZoneCloud { file: basename(&inp.preview_ply[k]), cloud: c })).collect(),
        refined: rf.iter().map(|(k, c)| (*k, ZoneCloud { file: basename(&inp.refined_ply[k]), cloud: c })).collect(),
        preview_models: inp.previews.iter().map(|(k, m)| (*k, m)).collect(),
        refined_models: inp.refined.iter().map(|(k, m)| (*k, m)).collect(),
        times,
        windows,
        position: inp.position,
    };
    run_zones(&z, log)
}

/// Post-processing body (in-memory input): preview alignment -> aligned/ -> per-event snapshots and manifest -> re-anchoring
/// (final_frame/).
pub fn run_zones(z: &PostZones, log: &Logger) -> Result<(), String> {
    let out = z.out;
    let pv: BTreeMap<usize, &PointCloud> = z.preview.iter().map(|(k, c)| (*k, c.cloud)).collect();
    let rf: BTreeMap<usize, &PointCloud> = z.refined.iter().map(|(k, c)| (*k, c.cloud)).collect();
    let tm = |k: &usize| z.times.get(k).copied().unwrap_or_default();
    let t_p: BTreeMap<usize, Option<f64>> = pv.keys().map(|k| (*k, tm(k).preview_ready)).collect();
    let t_r: BTreeMap<usize, Option<f64>> = rf.keys().map(|k| (*k, tm(k).refined_ready)).collect();
    let t_rp: BTreeMap<usize, Option<f64>> = rf.keys().map(|k| (*k, tm(k).refined_pose)).collect();
    let t_a: BTreeMap<usize, Option<f64>> = pv.keys().map(|k| (*k, tm(k).arrived)).collect();

    // --- Preview alignment: onto the latest refined model whose pose was ready at that time (preview_0, GPS-aligned, if none), using
    // shared points.
    let mut align_rows: Vec<Json> = Vec::new();
    let mut pframe: BTreeMap<usize, String> = BTreeMap::new();
    let mut pv_aligned: BTreeMap<usize, PointCloud> = BTreeMap::new();
    for (&k, cloud) in &pv {
        let (s, med, nk, reference, t): (f64, f64, usize, String, Sim3);
        if k == 0 {
            (s, med, nk, reference, t) = (1.0, 0.0, 0, "GPS(preview_0)".into(), Sim3::identity());
            pframe.insert(k, "preview_0".into());
        } else {
            let tpk = t_p[&k].unwrap_or(f64::INFINITY);
            let cand = rf.keys().copied().filter(|j| t_rp.get(j).copied().flatten().is_some_and(|x| x <= tpk)).max();
            let (b, refname, frame) = match cand.and_then(|j| z.refined_models.get(&j).copied().map(|m| (m, j))) {
                Some((m, j)) => (Some(m), format!("refined_{j}"), format!("refined_{j}")),
                None => (z.preview_models.get(&0).copied(), "preview_0(GPS)".to_string(), "preview_0".to_string()),
            };
            let a = z.preview_models.get(&k).copied();
            let res = match (a, b) {
                (Some(a), Some(b)) => {
                    let (lo, hi) = z.windows.get(&k).copied().unwrap_or((0, 0));
                    let pos = z.position;
                    // The position function is not 'static, so build the set of range names in advance.
                    let names: BTreeSet<String> =
                        a.images().map(|im| im.name.clone()).filter(|n| pos(n).is_some_and(|p| p >= lo && p < hi)).collect();
                    let filt: NameFilter = Arc::new(move |n: &str| names.contains(n));
                    let (mut pa, mut pb) = corr_points(a, b, Some(filt));
                    if pa.len() < 20 {
                        (pa, pb) = corr_points(a, b, None);
                    }
                    fit(&pa, &pb)
                }
                _ => None,
            };
            match res {
                Some(f) => {
                    (s, med, nk, reference, t) = (f.sim3.scale, f.median, f.inliers, refname, f.sim3);
                    pframe.insert(k, frame);
                }
                None => {
                    // Design decision: if there is no correspondence, do not fail; leave it unaligned and mark it.
                    log.line(&format!("[post] 초벌 {k} 정렬 실패(공유 점 부족) — 원래 좌표 유지"));
                    (s, med, nk, reference, t) = (1.0, 0.0, 0, format!("{refname}(실패)"), Sim3::identity());
                    pframe.insert(k, frame);
                }
            }
        }
        align_rows.push(Json::obj(vec![
            ("region", Json::Int(k as i64)),
            ("reference", Json::Str(reference)),
            ("pairs", Json::Int(nk as i64)),
            ("fit_median_m", Json::f(med, 2)),
            ("scale", Json::f(s, 3)),
        ]));
        let c = transformed(cloud, &t);
        write(&out.join("aligned/preview").join(z.preview[&k].file.clone()), &c)?;
        pv_aligned.insert(k, c);
    }
    for (k, c) in &rf {
        write(&out.join("aligned/refined").join(z.refined[k].file.clone()), c)?;
    }

    // --- Snapshots per event order.
    let mut evs: Vec<(f64, &str, usize)> = Vec::new();
    for k in pv.keys() {
        evs.push((t_p[k].unwrap_or(f64::INFINITY), "preview", *k));
    }
    for k in rf.keys() {
        evs.push((t_r[k].unwrap_or(f64::INFINITY), "refined", *k));
    }
    evs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(b.1)).then(a.2.cmp(&b.2)));
    let snap_opts = SnapshotOptions::default();
    let snapdir = out.join("snapshots");
    std::fs::create_dir_all(&snapdir).map_err(|e| e.to_string())?;
    let mut shown_p: BTreeSet<usize> = BTreeSet::new();
    let mut shown_r: BTreeSet<usize> = BTreeSet::new();
    struct Ev {
        i: usize,
        t: f64,
        kind: &'static str,
        k: usize,
        n: usize,
        refined: Vec<usize>,
        preview_only: Vec<usize>,
    }
    let mut man: Vec<Ev> = Vec::new();
    let snap_name = |i: usize, t: f64, kind: &str, k: usize| snapdir.join(format!("event_{i:02}_{t:06.1}s_{kind}_{k}.ply"));
    for (i, &(t, kind, k)) in evs.iter().enumerate() {
        let i = i + 1;
        if kind == "preview" {
            shown_p.insert(k);
        } else {
            shown_r.insert(k);
        }
        let fine: Vec<&PointCloud> = shown_r.iter().map(|j| rf[j]).collect();
        let po: Vec<usize> = shown_p.difference(&shown_r).copied().collect();
        let coarse: Vec<&PointCloud> = po.iter().map(|j| &pv_aligned[j]).collect();
        let a = compose_snapshot(&fine, &coarse, &snap_opts);
        write(&snap_name(i, t, kind, k), &a)?;
        let kind: &'static str = if kind == "preview" { "preview" } else { "refined" };
        man.push(Ev { i, t, kind, k, n: a.len(), refined: shown_r.iter().copied().collect(), preview_only: po });
    }
    let r1 = |x: Option<f64>| Json::of(x, 1);
    let rows: Vec<Json> = pv
        .keys()
        .map(|k| {
            let ta = t_a[k].unwrap_or(0.0);
            let tp = t_p[k].unwrap_or(0.0);
            let trp = t_rp.get(k).copied().flatten();
            let tr = t_r.get(k).copied().flatten();
            Json::obj(vec![
                ("region", Json::Int(*k as i64)),
                ("arrived_s", Json::f(ta, 1)),
                ("preview_ready_s", Json::f(tp, 1)),
                ("preview_latency_s", Json::f(tp - ta, 1)),
                ("refined_pose_s", r1(trp)),
                ("refined_ready_s", r1(tr)),
                ("refined_latency_s", r1(tr.map(|x| x - ta))),
            ])
        })
        .collect();
    let ev_json = |e: &Ev, frame: Option<&str>, n: usize| {
        let mut v = vec![
            ("event", Json::Int(e.i as i64)),
            ("time_s", Json::f(e.t, 1)),
            ("kind", Json::Str(e.kind.into())),
            ("region", Json::Int(e.k as i64)),
            ("points_1of6", Json::Int(n as i64)),
            ("refined", Json::ints(&e.refined)),
            ("preview_only", Json::ints(&e.preview_only)),
        ];
        if let Some(f) = frame {
            v.push(("frame", Json::Str(f.into())));
        }
        Json::obj(v)
    };
    let manifest = |events: Vec<Json>, reanchor: Option<Json>| {
        let mut v = vec![("timeline", Json::Arr(rows.clone())), ("events", Json::Arr(events)), ("align", Json::Arr(align_rows.clone()))];
        if let Some(r) = reanchor {
            v.push(("reanchor", r));
        }
        Json::obj(v)
    };
    let mpath = snapdir.join("manifest.json");
    std::fs::write(&mpath, manifest(man.iter().map(|e| ev_json(e, None, e.n)).collect(), None).dump()).map_err(|e| e.to_string())?;
    log.line("== 구역별 시간(초, 시작 기준)");
    for r in &rows {
        log.line(&r.py_repr());
    }
    log.line("== 정렬");
    for a in &align_rows {
        log.line(&a.py_repr());
    }
    log.line("== 사건 순서");
    for m in &man {
        log.line(&format!("{} {} {} {} {}", m.i, py_float(m.t, 1), m.kind, m.k, m.n));
    }
    let ks: Vec<usize> = rf.keys().copied().collect();
    for w in ks.windows(2) {
        match overlap_median(rf[&w[0]], rf[&w[1]]) {
            Some((m, _)) => log.line(&format!("정밀 {}-{} 겹침 차 중앙 {m:.2}m", w[0], w[1])),
            None => log.line(&format!("정밀 {}-{} 겹침 없음", w[0], w[1])),
        }
    }

    // --- Re-anchoring (incremental_reanchor.py): every time a new refined model appears, bring the already sent zones into its
    // coordinate frame.
    if rf.is_empty() {
        return Ok(());
    }
    let mut models: BTreeMap<String, &Reconstruction> = BTreeMap::new();
    if let Some(m) = z.preview_models.get(&0).copied() {
        models.insert("preview_0".into(), m);
    }
    for (k, m) in &z.refined_models {
        models.insert(format!("refined_{k}"), m);
    }
    let mut cache: HashMap<(String, String), Sim3> = HashMap::new();
    let mut order: Vec<(String, String)> = Vec::new();
    let mut tf = |a: &str, b: &str| -> Sim3 {
        if a == b {
            return Sim3::identity();
        }
        let key = (a.to_string(), b.to_string());
        if let Some(t) = cache.get(&key) {
            return *t;
        }
        let t = match (models.get(a), models.get(b)) {
            (Some(ma), Some(mb)) => {
                let (pa, pb) = corr_points(ma, mb, None);
                match fit(&pa, &pb) {
                    Some(f) => {
                        log.line(&format!(
                            "{a} -> {b}: 점쌍 {}, 잔차 중앙 {:.3}m, 축척 {:.4}, 이동 {:.2}m",
                            f.inliers,
                            f.median,
                            f.sim3.scale,
                            f.sim3.translation.norm()
                        ));
                        f.sim3
                    }
                    None => {
                        log.line(&format!("{a} -> {b}: 점쌍 부족({}), 항등 사용", pa.len()));
                        Sim3::identity()
                    }
                }
            }
            _ => Sim3::identity(),
        };
        cache.insert(key.clone(), t);
        order.push(key);
        t
    };
    for f in std::fs::read_dir(&snapdir).map_err(|e| e.to_string())?.flatten() {
        let n = f.file_name().to_string_lossy().to_string();
        if n.starts_with("event_") && n.ends_with(".ply") {
            let _ = std::fs::remove_file(f.path());
        }
    }
    let mut cur = "preview_0".to_string();
    let mut ev2: Vec<Json> = Vec::new();
    for e in &man {
        if e.kind == "refined" {
            cur = format!("refined_{}", e.k);
        }
        let rs: Vec<PointCloud> = e.refined.iter().map(|j| transformed(rf[j], &tf(&format!("refined_{j}"), &cur))).collect();
        let ps: Vec<PointCloud> = e
            .preview_only
            .iter()
            .map(|j| transformed(&pv_aligned[j], &tf(pframe.get(j).map(|s| s.as_str()).unwrap_or("preview_0"), &cur)))
            .collect();
        let a = compose_snapshot(&rs.iter().collect::<Vec<_>>(), &ps.iter().collect::<Vec<_>>(), &snap_opts);
        write(&snap_name(e.i, e.t, e.kind, e.k), &a)?;
        log.line(&format!("{} {} {} {} 좌표계 {cur} 점 {}", e.i, py_float(e.t, 1), e.kind, e.k, a.len()));
        ev2.push(ev_json(e, Some(&cur), a.len()));
    }
    let last = format!("refined_{}", rf.keys().max().copied().unwrap_or(0));
    let mut final_rf: BTreeMap<usize, PointCloud> = BTreeMap::new();
    for (k, c) in &rf {
        let c2 = transformed(c, &tf(&format!("refined_{k}"), &last));
        write(&out.join("final_frame/refined").join(z.refined[k].file.clone()), &c2)?;
        final_rf.insert(*k, c2);
    }
    for (k, c) in &pv_aligned {
        let fr = pframe.get(k).cloned().unwrap_or_else(|| "preview_0".into());
        write(&out.join("final_frame/preview").join(z.preview[k].file.clone()), &transformed(c, &tf(&fr, &last)))?;
    }
    for w in ks.windows(2) {
        match overlap_median(&final_rf[&w[0]], &final_rf[&w[1]]) {
            Some((m, n)) => log.line(&format!("최종 좌표계 정밀 {}-{} 겹침 차 중앙 {m:.2}m (겹침 점 {n})", w[0], w[1])),
            None => log.line(&format!("최종 좌표계 정밀 {}-{} 겹침 없음", w[0], w[1])),
        }
    }
    let re = Json::Obj(
        order
            .iter()
            .map(|(a, b)| {
                let t = cache[&(a.clone(), b.clone())];
                (format!("{a}->{b}"), Json::obj(vec![("scale", Json::f(t.scale, 4)), ("shift_m", Json::f(t.translation.norm(), 2))]))
            })
            .collect(),
    );
    std::fs::write(&mpath, manifest(ev2, Some(re)).dump()).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud(xs: &[f32]) -> PointCloud {
        PointCloud { positions: xs.iter().map(|x| [*x, 0.0, 0.0]).collect(), ..Default::default() }
    }

    #[test]
    fn live_compose_thins_and_drops_overlap_without_models() {
        // Stride 6 keeps indices 0 and 6 of each zone. Zone 0's kept points are 6 apart -> voxel edge 12 (factor 2).
        let mut xs0 = vec![0.0f32; 12];
        xs0[6] = 6.0;
        let mut xs1 = vec![100.0f32; 12];
        xs1[0] = 1.0; // same voxel as zone 0's point at x = 0 -> dropped
        xs1[6] = 20.0; // free voxel -> kept
        xs1[1] = 50.0; // removed by thinning
        let (c0, c1) = (cloud(&xs0), cloud(&xs1));
        let clouds: BTreeMap<usize, &PointCloud> = [(0, &c0), (1, &c1)].into_iter().collect();
        let models: BTreeMap<usize, &Reconstruction> = BTreeMap::new();
        let mut live = LiveDisplay::default();
        let out = live.compose(&clouds, &models);
        let mut xs: Vec<f32> = out.positions.iter().map(|p| p[0]).collect();
        xs.sort_by(f32::total_cmp);
        assert_eq!(xs, vec![0.0, 6.0, 20.0]);
        assert!(live.compose(&BTreeMap::new(), &models).is_empty());
    }

    #[test]
    fn live_compose_keeps_seam_without_empty_strip() {
        // Two grids of spacing 1 meeting at x = 20; zone 1 starts 4 rows early and is offset by 0.3.
        let grid = |x0: f32, x1: f32, off: f32| PointCloud {
            positions: (x0 as i32..x1 as i32).flat_map(|i| (0..30).map(move |j| [i as f32 + off, j as f32 + off, 0.0])).collect(),
            ..Default::default()
        };
        let (c0, c1) = (grid(0.0, 20.0, 0.0), grid(16.0, 40.0, 0.3));
        let clouds: BTreeMap<usize, &PointCloud> = [(0, &c0), (1, &c1)].into_iter().collect();
        let models: BTreeMap<usize, &Reconstruction> = BTreeMap::new();
        let mut live = LiveDisplay::new(LiveOptions { stride: 1, relocate: false, ..LiveOptions::default() });
        let out = live.compose(&clouds, &models);
        assert!(out.len() < c0.len() + c1.len(), "overlap duplicates are removed");
        assert!(out.len() > 40 * 30 / 2);
        // Every cell of width 2 along x (and every row band) still holds points: no empty strip at the seam.
        for cx in 0..20 {
            for cy in 0..15 {
                let n = out.positions.iter().filter(|p| (p[0] / 2.0).floor() as i32 == cx && (p[1] / 2.0).floor() as i32 == cy).count();
                assert!(n > 0, "empty cell ({cx}, {cy})");
            }
        }
    }

    #[test]
    fn live_compose_voxel_size_and_ownership_are_stable_across_calls() {
        let grid = |x0: i32, x1: i32, off: f32| PointCloud {
            positions: (x0..x1).flat_map(|i| (0..10).map(move |j| [i as f32 + off, j as f32 + off, 0.0])).collect(),
            ..Default::default()
        };
        let (c0, c1, c2) = (grid(0, 12, 0.0), grid(8, 20, 0.3), grid(16, 28, 0.6));
        let models: BTreeMap<usize, &Reconstruction> = BTreeMap::new();
        let mut live = LiveDisplay::new(LiveOptions { stride: 1, relocate: false, ..LiveOptions::default() });
        let a = live.compose(&[(0, &c0), (1, &c1)].into_iter().collect(), &models);
        let voxel = live.voxel;
        assert!(voxel.is_some());
        // A very different first zone would give another estimate if it were redone.
        let b = live.compose(&[(0, &c0), (1, &c1), (2, &c2)].into_iter().collect(), &models);
        assert_eq!(live.voxel, voxel);
        let kept: BTreeSet<[u32; 3]> = b.positions.iter().map(|p| p.map(f32::to_bits)).collect();
        assert!(a.positions.iter().all(|p| kept.contains(&p.map(f32::to_bits))));
        assert!(b.len() > a.len());
    }

    /// Cloud on a lattice (spacing 1) with keys = lattice index + `key0` (so equal lattice cells of two zones get equal keys).
    fn keyed_grid(x0: i32, x1: i32, off: f32, key0: u64) -> PointCloud {
        let mut c = PointCloud::default();
        for i in x0..x1 {
            for j in 0..10 {
                c.positions.push([i as f32 + off, j as f32 + off, 0.0]);
                c.keys.push(key0 + (i as u64) * 100 + j as u64);
            }
        }
        c
    }

    #[test]
    fn key_thinning_keeps_the_same_pixels_after_reordering() {
        let c = keyed_grid(0, 40, 0.0, 7);
        let mut r = c.clone();
        let n = r.len();
        // A re-densification returns other points in another order and with other positions, but the same keys.
        r.positions.reverse();
        r.keys.reverse();
        for p in r.positions.iter_mut() {
            p[2] += 0.5;
        }
        let (a, b) = (thin(&c, 6), thin(&r, 6));
        let ka: BTreeSet<u64> = a.keys.iter().copied().collect();
        let kb: BTreeSet<u64> = b.keys.iter().copied().collect();
        assert_eq!(ka, kb);
        assert!(a.len() > n / 12 && a.len() < n / 3, "{} of {n}", a.len());
        assert!(a.keys.iter().all(|&k| splitmix64(k).is_multiple_of(6)));
        // Without keys the index rule applies.
        let mut nk = c.clone();
        nk.keys.clear();
        assert_eq!(thin(&nk, 6).len(), n.div_ceil(6));
    }

    #[test]
    fn live_compose_ownership_is_sticky_when_a_zone_is_replaced() {
        let models: BTreeMap<usize, &Reconstruction> = BTreeMap::new();
        let mut live = LiveDisplay::new(LiveOptions { stride: 1, relocate: false, ..LiveOptions::default() });
        // Zone 0 covers x 0..12, zone 1 x 8..20; the overlap 8..12 belongs to zone 0.
        let (c0, c1) = (keyed_grid(0, 12, 0.0, 0), keyed_grid(8, 20, 0.2, 5000));
        let (a, za) = live.compose_zoned(&[(0, &c0), (1, &c1)].into_iter().collect(), &models);
        assert_eq!(a.len(), za.len());
        let owner_at = |c: &PointCloud, z: &[usize], x: f32| -> Vec<usize> {
            c.positions.iter().zip(z).filter(|(p, _)| (p[0] - x).abs() < 0.5).map(|(_, z)| *z).collect()
        };
        assert!(owner_at(&a, &za, 10.0).iter().all(|&z| z == 0));
        // Zone 0 is re-densified in a different place (now only x 0..9): zone 1 must not take over the voxels zone 0 still
        // has points in, while the voxels zone 0 lost go to the lowest zone present (no hole).
        let c0b = keyed_grid(0, 9, 0.1, 0);
        let (b, zb) = live.compose_zoned(&[(0, &c0b), (1, &c1)].into_iter().collect(), &models);
        assert!(owner_at(&b, &zb, 8.5).iter().all(|&z| z == 0), "{:?}", owner_at(&b, &zb, 8.5));
        assert!(owner_at(&b, &zb, 10.0).iter().all(|&z| z == 1));
        // Once zone 0 grows back, the voxels that went to zone 1 in between stay with zone 1 (no flicker back), and the
        // ones zone 0 kept stay with zone 0.
        let (c, zc) = live.compose_zoned(&[(0, &c0), (1, &c1)].into_iter().collect(), &models);
        assert!(owner_at(&c, &zc, 11.0).iter().all(|&z| z == 1));
        assert!(owner_at(&c, &zc, 8.5).iter().all(|&z| z == 0));
        assert_eq!(c.len(), zc.len());
        live.reset_voxel();
        assert!(live.owners.is_empty());
    }

    #[test]
    fn live_compose_drops_repeated_keys_keeping_the_lowest_zone() {
        let models: BTreeMap<usize, &Reconstruction> = BTreeMap::new();
        let mut live = LiveDisplay::new(LiveOptions { stride: 1, relocate: false, voxel_factor: 0.01, ..LiveOptions::default() });
        // Same keys in both zones at far apart positions (never the same voxel): only zone 0's survive.
        let (c0, c1) = (keyed_grid(0, 5, 0.0, 0), keyed_grid(100, 105, 0.0, 0));
        let c1 = PointCloud { keys: c0.keys.clone(), ..c1 };
        let (out, z) = live.compose_zoned(&[(0, &c0), (1, &c1)].into_iter().collect(), &models);
        assert_eq!(out.len(), c0.len());
        assert!(z.iter().all(|&z| z == 0));
        let uniq: BTreeSet<u64> = out.keys.iter().copied().collect();
        assert_eq!(uniq.len(), out.len());
    }

    fn tiny_model(shift: f64, scale: f64) -> Reconstruction {
        use cumulus3d_core::{Camera, CameraModelKind, Image, Point3D, Rigid3, TrackEntry, Vec2};
        let mut rec = Reconstruction::new();
        let mut cam = Camera::from_focal(CameraModelKind::Pinhole, 500.0, 640, 480);
        cam.camera_id = 1;
        rec.add_camera_own_rig(cam).unwrap();
        for i in 1..=2u32 {
            let pts: Vec<Vec2> = (0..40).map(|k| Vec2::new(k as f64, 0.0)).collect();
            rec.add_image_own_frame(Image::new(i, format!("camF/camF_{:04}.jpg", i - 1), 1, pts), Some(Rigid3::identity())).unwrap();
            rec.register_image(i).unwrap();
        }
        for k in 0..40u32 {
            let x = Vec3::new(k as f64, (k * k % 7) as f64, (k % 5) as f64 * 2.0);
            let track = vec![TrackEntry::new(1, k), TrackEntry::new(2, k)];
            rec.add_point3d_with_id(
                k as u64 + 1,
                Point3D { xyz: x * scale + Vec3::new(shift, 0.0, -shift), color: [0; 3], error: -1.0, track },
            )
            .unwrap();
        }
        rec
    }

    #[test]
    fn shared_point_fit_recovers_known_sim3() {
        let (a, b) = (tiny_model(0.0, 1.0), tiny_model(3.0, 1.5));
        let (f, n) = fit_shared_points(&a, &b, None, 20).unwrap();
        assert_eq!(n, 40);
        assert!((f.sim3.scale - 1.5).abs() < 1e-9);
        assert!((f.sim3.transform_point(&Vec3::new(1.0, 2.0, 3.0)) - Vec3::new(4.5, 3.0, 4.5 - 3.0)).norm() < 1e-9);
        assert!(f.median < 1e-9);
        // Too few pairs for the requested minimum.
        assert!(fit_shared_points(&a, &b, None, 41).is_none());
        // A name filter that accepts nothing leaves no pairs.
        let none: NameFilter = Arc::new(|_: &str| false);
        assert!(fit_shared_points(&a, &b, Some(none), 1).is_none());
    }
}
