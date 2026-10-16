//! Signed-distance field mesher: per-block distance accumulation and Surface Nets extraction.
//!
//! For every grid point `x` within radius `R` of a bin `p`, the Hoppe distance `d = n·(x − p)` is added with a
//! Gaussian weight. Repeated observations of the same surface become a weighted average, so the surface does not
//! thicken.
//!
//! A cell produces a vertex only when all eight of its corners are observed; unobserved space is never filled. Each
//! block reads local voxels `-1..=dim`, so boundary cells are computed from the same values by both neighbouring
//! blocks and their vertices are bitwise equal.
//!
//! Storage: a block of `dim`³ voxels is split into 8³ bricks and only touched bricks are allocated. Refined segments
//! accumulate into the `base` layer and each preview segment into its own layer. Layers are always summed in the same
//! order (base, then preview layers by segment id), which makes preview replacement bitwise exact.

use crate::bins::{Bin, NormalPolicy, bin_points};
use crate::orient::{Oracle, orient_normals};
use crate::{
    BlockId, BlockMesh, Config, Dirty, ExtractStats, FACE_BITS, Halo, IngestError, IngestStats,
    Level, Mesher, Point, SegmentId, apply_built, drain_dirty, halo_mask, mark_mask, same_mesh,
};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Instant;

/// Per-phase timings of `ingest` and `extract`, in milliseconds.
///
/// Instrumentation only; it does not affect the algorithm. [`take_phase_ms`] returns the totals accumulated since
/// the previous call and resets them. If the environment variable `DELTAMESH_PROF_SPLIT` is set, binning runs once
/// more without normal estimation so that `binning` and `normals` can be reported separately; the time of that extra
/// run is not counted anywhere.
#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseMs {
    /// `bin_points` in total, minus normal estimation when the split is enabled.
    pub binning: f64,
    /// Normal estimation. Only set when the split is enabled.
    pub normals: f64,
    /// Sign of estimated normals (`orient_normals`, including the oracle).
    pub orient: f64,
    /// Dropping the preview layer of the same segment and marking dirty blocks.
    pub drop_layer: f64,
    /// Building per-block bin lists and taking blocks out of the layer (single-threaded).
    pub lists: f64,
    /// Gaussian accumulation (parallel).
    pub splat: f64,
    /// Neighbour masks, putting blocks back into the layer and marking dirty blocks.
    pub dirty: f64,
    /// Splat path: building and packing per-brick bin lists on the CPU.
    pub gpu_prep: f64,
    /// Splat path: buffer uploads.
    pub gpu_upload: f64,
    /// Splat path: waiting for results, from submission to readback.
    pub gpu_wait: f64,
    /// Splat path: adding results to the layer.
    pub gpu_merge: f64,
    /// GPU-resident path: gathering the combined field.
    pub gather: f64,
    /// GPU-resident path: surface extraction on the GPU.
    pub extract_gpu: f64,
    /// GPU-resident path: simplification and change detection on the CPU.
    pub simplify: f64,
    /// Number of `ingest` calls.
    pub calls: u32,
}

static PHASE: Mutex<PhaseMs> = Mutex::new(PhaseMs {
    binning: 0.0,
    normals: 0.0,
    orient: 0.0,
    drop_layer: 0.0,
    lists: 0.0,
    splat: 0.0,
    dirty: 0.0,
    gpu_prep: 0.0,
    gpu_upload: 0.0,
    gpu_wait: 0.0,
    gpu_merge: 0.0,
    gather: 0.0,
    extract_gpu: 0.0,
    simplify: 0.0,
    calls: 0,
});

/// Adds timings from another mesher (the GPU-resident path) to the shared counters.
#[cfg(feature = "gpu")]
pub(crate) fn phase_add(f: impl FnOnce(&mut PhaseMs)) {
    f(&mut PHASE.lock().unwrap());
}

/// Returns the phase timings accumulated since the last call and resets them to zero.
pub fn take_phase_ms() -> PhaseMs {
    std::mem::take(&mut *PHASE.lock().unwrap())
}

#[inline]
fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Voxels per brick edge. A power of two, so the accumulation loop can index with shifts and masks.
const BR: i32 = 8;
const BR_SHIFT: i32 = BR.trailing_zeros() as i32;
const BR_MASK: i32 = BR - 1;
const BR3: usize = (BR * BR * BR) as usize;
const _: () = assert!(BR.count_ones() == 1 && BR <= 64);

/// Weighted sums of one voxel: `Σw·d`, `Σw` and `Σw·rgb`.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct Acc {
    wd: f32,
    w: f32,
    wc: [f32; 3],
}

/// One brick of `BR`³ voxels, x fastest.
type Brick = Box<[Acc; BR3]>;

/// One block of a layer: a dense table of optional bricks, z-major.
#[derive(Clone)]
struct Block {
    bricks: Vec<Option<Brick>>,
    /// Lower corner of the local range ever changed in this layer. Used for dirty marking when the layer is dropped.
    min: [i32; 3],
    /// Upper corner of the local range ever changed in this layer.
    max: [i32; 3],
    /// Mask of blocks (the block itself and neighbours, see `nbit`) that changes in this layer require to be
    /// re-extracted. Used when the layer is dropped.
    nbr: u32,
}

impl Block {
    fn new(dim: i32) -> Self {
        let nb = (dim / BR) as usize;
        Self {
            bricks: vec![None; nb * nb * nb],
            min: [i32::MAX; 3],
            max: [i32::MIN; 3],
            nbr: 0,
        }
    }
    /// Returns the voxel at local coordinate `l`, or `None` if its brick is not allocated.
    #[inline]
    fn get(&self, l: [i32; 3], nb: i32) -> Option<&Acc> {
        let bi = ((l[2] / BR) * nb + l[1] / BR) * nb + l[0] / BR;
        let b = self.bricks[bi as usize].as_ref()?;
        Some(&b[(((l[2] % BR) * BR + l[1] % BR) * BR + l[0] % BR) as usize])
    }
    /// Returns the voxel at local coordinate `l` for writing, allocating its brick. Used only by the per-voxel
    /// reference implementation in tests.
    #[cfg(test)]
    #[inline]
    fn get_mut(&mut self, l: [i32; 3], nb: i32) -> &mut Acc {
        let bi = ((l[2] / BR) * nb + l[1] / BR) * nb + l[0] / BR;
        let b = self.bricks[bi as usize].get_or_insert_with(|| Box::new([Acc::default(); BR3]));
        &mut b[(((l[2] % BR) * BR + l[1] % BR) * BR + l[0] % BR) as usize]
    }
    fn brick_count(&self) -> usize {
        self.bricks.iter().filter(|b| b.is_some()).count()
    }
}

/// One field layer: blocks by id.
type Layer = FxHashMap<BlockId, Block>;

/// Accumulation device of the splat path. Without the `gpu` feature this is an uninhabited type.
#[cfg(feature = "gpu")]
pub type Gpu = crate::gpu::GpuSplat;
#[cfg(not(feature = "gpu"))]
pub enum Gpu {}

/// CPU signed-distance field mesher.
///
/// Refined segments accumulate into the base layer; each preview segment keeps its own layer until its refined
/// version replaces it. Accumulation runs on the CPU, or on the GPU once a device is set with
/// [`SdfMesher::set_gpu`]. Extraction always runs on the CPU.
pub struct SdfMesher {
    cfg: Config,
    base: Layer,
    /// Preview layers by segment. A `BTreeMap` keeps the summation order fixed.
    pending: BTreeMap<SegmentId, Layer>,
    refined: FxHashSet<SegmentId>,
    dirty: Dirty,
    last_extract: ExtractStats,
    meshes: FxHashMap<BlockId, BlockMesh>,
    versions: FxHashMap<BlockId, u32>,
    /// Device used for accumulation (see `set_gpu`). `None` means the CPU.
    gpu: Option<std::sync::Arc<Gpu>>,
}

impl SdfMesher {
    /// Creates an empty mesher.
    ///
    /// # Panics
    ///
    /// Panics if `cfg.block_dim` is not a positive multiple of the brick size (8).
    pub fn new(cfg: Config) -> Self {
        assert!(
            cfg.block_dim % BR == 0 && cfg.block_dim > 0,
            "block_dim must be a positive multiple of the brick size"
        );
        Self {
            cfg,
            base: Layer::default(),
            pending: BTreeMap::new(),
            refined: FxHashSet::default(),
            dirty: Dirty::default(),
            last_extract: ExtractStats::default(),
            meshes: FxHashMap::default(),
            versions: FxHashMap::default(),
            gpu: None,
        }
    }

    /// Sets the device used for accumulation, or `None` for the CPU.
    ///
    /// Several meshers can share one device.
    pub fn set_gpu(&mut self, gpu: Option<std::sync::Arc<Gpu>>) {
        self.gpu = gpu;
    }

    /// Returns whether accumulation runs on the GPU.
    pub fn uses_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// Returns the combined field of one block, for verification.
    ///
    /// Voxels at local coordinates `-1..=dim` are summed over the base layer and then the preview layers in segment
    /// order. Used to compare GPU extraction and the GPU-resident field with the CPU; it is not fast.
    ///
    /// # Returns
    ///
    /// `(dim+2)³` entries indexed by `((z+1)·P + (y+1))·P + (x+1)` with `P = dim+2`, each
    /// `[Σw·d, Σw, Σw·r, Σw·g, Σw·b]`. Unobserved voxels are zero.
    #[doc(hidden)]
    pub fn padded_volume(&self, id: BlockId) -> Vec<[f32; 5]> {
        let dim = self.cfg.block_dim;
        let nb = dim / BR;
        let p = (dim + 2) as usize;
        let mut out = vec![[0.0f32; 5]; p * p * p];
        for layer in std::iter::once(&self.base).chain(self.pending.values()) {
            for lz in -1..=dim {
                for ly in -1..=dim {
                    for lx in -1..=dim {
                        let g = [id[0] * dim + lx, id[1] * dim + ly, id[2] * dim + lz];
                        let bid = [
                            g[0].div_euclid(dim),
                            g[1].div_euclid(dim),
                            g[2].div_euclid(dim),
                        ];
                        let l = [
                            g[0].rem_euclid(dim),
                            g[1].rem_euclid(dim),
                            g[2].rem_euclid(dim),
                        ];
                        let Some(a) = layer.get(&bid).and_then(|b| b.get(l, nb)) else {
                            continue;
                        };
                        if a.w == 0.0 {
                            continue;
                        }
                        let o = &mut out
                            [(((lz + 1) as usize * p) + (ly + 1) as usize) * p + (lx + 1) as usize];
                        o[0] += a.wd;
                        o[1] += a.w;
                        o[2] += a.wc[0];
                        o[3] += a.wc[1];
                        o[4] += a.wc[2];
                    }
                }
            }
        }
        out
    }

    /// Returns the bins that `ingest` would accumulate in the current state, for verification.
    ///
    /// Runs binning and normal orientation through the same code path as `ingest`.
    #[doc(hidden)]
    pub fn prepare_bins(&self, level: Level, pts: &[Point]) -> Vec<Bin> {
        let cfg = self.cfg;
        let policy = if level == Level::Preview {
            NormalPolicy::Estimate
        } else {
            NormalPolicy::Trust
        };
        let (mut bins, bs) = bin_points(
            pts,
            cfg.bin,
            policy,
            cfg.normal_radius_bins,
            cfg.normal_min_neighbors,
        );
        if !bs.fixed.is_empty() {
            let oracle = |p: [f32; 3], n: [f32; 3]| self.base_agreement(p, n);
            let use_oracle = cfg.orient_oracle && !self.base.is_empty();
            orient_normals(
                &mut bins,
                &bs.fixed,
                cfg.bin,
                cfg.orient,
                use_oracle.then_some(&oracle as Oracle),
            );
        }
        bins
    }

    /// Returns the changed local range of every block in a layer, for verification.
    ///
    /// `None` selects the base layer and `Some(seg)` the preview layer of `seg`. Blocks that never changed are
    /// skipped. Sorted by id.
    #[doc(hidden)]
    pub fn layer_ranges(&self, seg: Option<SegmentId>) -> Vec<(BlockId, [i32; 3], [i32; 3])> {
        let layer = match seg {
            None => Some(&self.base),
            Some(s) => self.pending.get(&s),
        };
        let mut v: Vec<_> = layer
            .into_iter()
            .flatten()
            .filter(|(_, b)| b.min[0] <= b.max[0])
            .map(|(id, b)| (*id, b.min, b.max))
            .collect();
        v.sort_unstable_by_key(|x| x.0);
        v
    }

    /// Lists the bins touching each block, for verification and the GPU paths.
    ///
    /// Same as the list built internally by `ingest`.
    #[doc(hidden)]
    pub fn block_lists(cfg: &Config, bins: &[Bin]) -> Vec<(BlockId, Vec<u32>)> {
        block_lists(cfg, bins)
    }

    /// Returns the CPU extraction of one block before simplification, for verification.
    ///
    /// The tuple holds positions, colours, indices and the per-vertex seam flag.
    #[doc(hidden)]
    pub fn build_block_public(&self, id: BlockId) -> (Vec<f32>, Vec<u8>, Vec<u32>, Vec<bool>) {
        self.build_block(id)
    }

    /// Returns the ids of the blocks that currently have a mesh, sorted.
    #[doc(hidden)]
    pub fn block_ids(&self) -> Vec<BlockId> {
        let mut v: Vec<BlockId> = self.meshes.keys().copied().collect();
        v.sort_unstable();
        v
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Returns the segments that currently have a preview layer, in ascending order.
    pub fn pending_segments(&self) -> Vec<SegmentId> {
        self.pending.keys().copied().collect()
    }

    /// Returns the number of blocks the next `extract` will re-extract.
    pub fn dirty_count(&self) -> usize {
        self.dirty.len()
    }

    /// Accumulates `bins` into `layer` and marks the affected blocks in `dirty`.
    ///
    /// Blocks touched by the bins are taken out of the layer and processed in parallel, one block per task with bins
    /// in list order, so the result does not depend on the thread count. If `gpu` is set the splat path is tried
    /// first, and the CPU is used if it fails before changing anything.
    ///
    /// The CPU loop is bitwise equal to a plain per-voxel loop:
    ///
    /// - The squared distance is summed as `(dx² + dy²) + dz²`. Float addition is monotonic, so if `dz²` alone
    ///   exceeds `r²` the whole z plane is outside the radius.
    /// - For fixed `y` and `z` the voxels within the radius form one contiguous run in `x`, because the squared
    ///   distance decreases and then increases along `x`. The run ends are found first and the run is split at brick
    ///   boundaries so that each brick is looked up once. Local coordinates are non-negative, so brick indices use
    ///   shifts and masks.
    ///
    /// Neighbour masks are exact for face neighbours (from the changed range). Diagonal neighbours are narrowed by
    /// re-checking the boundary edges with `diag_mask`.
    fn integrate(
        cfg: &Config,
        layer: &mut Layer,
        bins: &[Bin],
        dirty: &mut Dirty,
        gpu: Option<&Gpu>,
    ) {
        let v = cfg.voxel;
        let dim = cfg.block_dim;
        let r = cfg.splat_radius;
        let r2 = r * r;
        let inv2s2 = 1.0 / (2.0 * (r * 0.5) * (r * 0.5));
        let range = |p: f32| {
            (
                (((p - r) / v).ceil()) as i32,
                (((p + r) / v).floor()) as i32,
            )
        };

        let t_lists = Instant::now();
        let lists = block_lists(cfg, bins);
        let mut work: Vec<Work> = lists
            .into_iter()
            .map(|(id, l)| {
                let blk = layer.remove(&id).unwrap_or_else(|| Block::new(dim));
                (id, blk, l, [i32::MAX; 3], [i32::MIN; 3])
            })
            .collect();
        let lists_ms = ms(t_lists);
        let t_splat = Instant::now();
        let nb = dim / BR;
        let on_gpu = match gpu {
            #[cfg(feature = "gpu")]
            Some(g) => match Self::splat_gpu(cfg, g, bins, &mut work) {
                Ok(()) => true,
                Err(e) => {
                    eprintln!("GPU accumulation failed, using the CPU for this input: {e}");
                    false
                }
            },
            #[cfg(not(feature = "gpu"))]
            Some(never) => match *never {},
            None => false,
        };
        if !on_gpu {
            work.par_iter_mut().for_each(|(id, blk, list, tmin, tmax)| {
                let o = [id[0] * dim, id[1] * dim, id[2] * dim];
                for &i in list.iter() {
                    let b = &bins[i as usize];
                    let rx = range(b.pos[0]);
                    let ry = range(b.pos[1]);
                    let rz = range(b.pos[2]);
                    let (x0, x1) = (rx.0.max(o[0]), rx.1.min(o[0] + dim - 1));
                    for gz in rz.0.max(o[2])..=rz.1.min(o[2] + dim - 1) {
                        let dz = gz as f32 * v - b.pos[2];
                        let dz2 = dz * dz;
                        if dz2 > r2 {
                            continue;
                        }
                        let ndz = b.normal[2] * dz;
                        let lz = gz - o[2];
                        for gy in ry.0.max(o[1])..=ry.1.min(o[1] + dim - 1) {
                            let dy = gy as f32 * v - b.pos[1];
                            let dy2 = dy * dy;
                            let q = |gx: i32| {
                                let dx = gx as f32 * v - b.pos[0];
                                (dx * dx + dy2) + dz2
                            };
                            let mut a0 = x0;
                            while a0 <= x1 && q(a0) > r2 {
                                a0 += 1;
                            }
                            if a0 > x1 {
                                continue;
                            }
                            let mut a1 = x1;
                            while q(a1) > r2 {
                                a1 -= 1;
                            }
                            let ndy = b.normal[1] * dy;
                            let ly = gy - o[1];
                            let brow = ((lz >> BR_SHIFT) * nb + (ly >> BR_SHIFT)) * nb;
                            let orow = (((lz & BR_MASK) << (2 * BR_SHIFT))
                                | ((ly & BR_MASK) << BR_SHIFT))
                                as usize;
                            let (l0, l1) = (a0 - o[0], a1 - o[0]);
                            let mut lx = l0;
                            while lx <= l1 {
                                let end = (lx | BR_MASK).min(l1);
                                let br = blk.bricks[(brow + (lx >> BR_SHIFT)) as usize]
                                    .get_or_insert_with(|| Box::new([Acc::default(); BR3]));
                                for x in lx..=end {
                                    let dx = (x + o[0]) as f32 * v - b.pos[0];
                                    let q = (dx * dx + dy2) + dz2;
                                    let w = (-q * inv2s2).exp();
                                    let d = (b.normal[0] * dx + ndy) + ndz;
                                    let a = &mut br[orow | (x & BR_MASK) as usize];
                                    a.wd += w * d;
                                    a.w += w;
                                    a.wc[0] += w * b.rgb[0];
                                    a.wc[1] += w * b.rgb[1];
                                    a.wc[2] += w * b.rgb[2];
                                }
                                lx = end + 1;
                            }
                            tmin[0] = tmin[0].min(l0);
                            tmax[0] = tmax[0].max(l1);
                            tmin[1] = tmin[1].min(ly);
                            tmax[1] = tmax[1].max(ly);
                            tmin[2] = tmin[2].min(lz);
                            tmax[2] = tmax[2].max(lz);
                        }
                    }
                }
            });
        }
        let splat_ms = ms(t_splat);
        let t_dirty = Instant::now();
        let masks: Vec<u32> = work
            .par_iter()
            .map(|(id, _, list, tmin, tmax)| {
                if tmin[0] > tmax[0] {
                    return 0;
                }
                let m = halo_mask(*tmin, *tmax, dim, Halo::Both);
                if m & !FACE_BITS == 0 {
                    m
                } else {
                    (m & FACE_BITS) | Self::diag_mask(cfg, *id, bins, list, m & !FACE_BITS)
                }
            })
            .collect();
        for ((id, mut blk, _, tmin, tmax), mask) in work.into_iter().zip(masks) {
            if tmin[0] <= tmax[0] {
                mark_mask(dirty, id, mask);
                blk.nbr |= mask;
                for k in 0..3 {
                    blk.min[k] = blk.min[k].min(tmin[k]);
                    blk.max[k] = blk.max[k].max(tmax[k]);
                }
            }
            layer.insert(id, blk);
        }
        let mut ph = PHASE.lock().unwrap();
        ph.lists += lists_ms;
        ph.splat += splat_ms;
        ph.dirty += ms(t_dirty);
    }

    /// Accumulates on the GPU (the splat path).
    ///
    /// For every block, builds per-brick bin lists that keep bin order and packs them into tiles, one tile per brick.
    /// The GPU returns per-voxel contributions in chunks of consecutive tiles; each chunk is added to the blocks it
    /// covers. Every voxel receives exactly one addition, so the result does not depend on the order in which blocks
    /// are processed.
    ///
    /// # Errors
    ///
    /// Returns `Err` without changing anything if a pre-check such as a buffer limit fails; the caller then falls
    /// back to the CPU.
    ///
    /// # Panics
    ///
    /// Panics if the GPU fails after part of the result has already been merged.
    #[cfg(feature = "gpu")]
    fn splat_gpu(cfg: &Config, g: &Gpu, bins: &[Bin], work: &mut [Work]) -> Result<(), String> {
        use crate::gpu::Tile;
        let v = cfg.voxel;
        let dim = cfg.block_dim;
        let r = cfg.splat_radius;
        let r2 = r * r;
        let inv2s2 = 1.0 / (2.0 * (r * 0.5) * (r * 0.5));
        let range = |p: f32| {
            (
                (((p - r) / v).ceil()) as i32,
                (((p + r) / v).floor()) as i32,
            )
        };
        let nb = dim / BR;
        let nbr = (nb * nb * nb) as usize;
        let t_prep = Instant::now();
        let per: Vec<Vec<(u32, Vec<u32>)>> = work
            .par_iter()
            .map(|(id, _, list, _, _)| {
                let o = [id[0] * dim, id[1] * dim, id[2] * dim];
                let mut bl: Vec<Vec<u32>> = vec![Vec::new(); nbr];
                for &i in list.iter() {
                    let b = &bins[i as usize];
                    let rr = [range(b.pos[0]), range(b.pos[1]), range(b.pos[2])];
                    let mut lo = [0i32; 3];
                    let mut hi = [0i32; 3];
                    let mut empty = false;
                    for a in 0..3 {
                        let (l, h) = (rr[a].0.max(o[a]), rr[a].1.min(o[a] + dim - 1));
                        if l > h {
                            empty = true;
                        }
                        lo[a] = (l - o[a]) >> BR_SHIFT;
                        hi[a] = (h - o[a]) >> BR_SHIFT;
                    }
                    if empty {
                        continue;
                    }
                    for bz in lo[2]..=hi[2] {
                        for by in lo[1]..=hi[1] {
                            for bx in lo[0]..=hi[0] {
                                bl[((bz * nb + by) * nb + bx) as usize].push(i);
                            }
                        }
                    }
                }
                bl.into_iter()
                    .enumerate()
                    .filter(|(_, l)| !l.is_empty())
                    .map(|(k, l)| (k as u32, l))
                    .collect()
            })
            .collect();
        let mut tiles: Vec<Tile> = Vec::new();
        let mut owner: Vec<u32> = Vec::new();
        let mut first: Vec<usize> = Vec::with_capacity(work.len() + 1);
        let mut idx: Vec<u32> = Vec::new();
        for (bi, list) in per.iter().enumerate() {
            first.push(tiles.len());
            let id = work[bi].0;
            for (k, l) in list {
                let k = *k as i32;
                let (bx, by, bz) = (k % nb, (k / nb) % nb, k / (nb * nb));
                tiles.push(Tile {
                    ox: id[0] * dim + bx * BR,
                    oy: id[1] * dim + by * BR,
                    oz: id[2] * dim + bz * BR,
                    start: idx.len() as u32,
                    count: l.len() as u32,
                    pad: [0; 3],
                });
                owner.push(k as u32);
                idx.extend_from_slice(l);
            }
        }
        first.push(tiles.len());
        drop(per);
        let packed: Vec<[f32; 4]> = bins
            .iter()
            .flat_map(|b| {
                [
                    [b.pos[0], b.pos[1], b.pos[2], 0.0],
                    [b.normal[0], b.normal[1], b.normal[2], 0.0],
                    [b.rgb[0], b.rgb[1], b.rgb[2], 0.0],
                ]
            })
            .collect();
        let prep_ms = ms(t_prep);
        let mut merged = false;
        let res = g.splat(&packed, &idx, &tiles, v, r2, inv2s2, |t0, out| {
            merged = true;
            let t1 = t0 + out.len() / (512 * 8);
            let b0 = first.partition_point(|&f| f <= t0) - 1;
            let b1 = first.partition_point(|&f| f < t1);
            work[b0..b1]
                .par_iter_mut()
                .enumerate()
                .for_each(|(j, (_, blk, _, tmin, tmax))| {
                    let bi = b0 + j;
                    for t in first[bi].max(t0)..first[bi + 1].min(t1) {
                        let k = owner[t] as i32;
                        let base = [(k % nb) * BR, ((k / nb) % nb) * BR, (k / (nb * nb)) * BR];
                        let src = &out[(t - t0) * 512 * 8..(t - t0 + 1) * 512 * 8];
                        if !(0..512).any(|vi| src[vi * 8 + 1] > 0.0) {
                            continue;
                        }
                        let br = blk.bricks[k as usize]
                            .get_or_insert_with(|| Box::new([Acc::default(); BR3]));
                        for vi in 0..512usize {
                            let c = &src[vi * 8..vi * 8 + 5];
                            if c[1] <= 0.0 {
                                continue;
                            }
                            let a = &mut br[vi];
                            a.wd += c[0];
                            a.w += c[1];
                            a.wc[0] += c[2];
                            a.wc[1] += c[3];
                            a.wc[2] += c[4];
                            let l = [
                                base[0] + (vi & 7) as i32,
                                base[1] + ((vi >> 3) & 7) as i32,
                                base[2] + (vi >> 6) as i32,
                            ];
                            for a in 0..3 {
                                tmin[a] = tmin[a].min(l[a]);
                                tmax[a] = tmax[a].max(l[a]);
                            }
                        }
                    }
                });
        });
        match res {
            Ok(tm) => {
                let mut ph = PHASE.lock().unwrap();
                ph.gpu_prep += prep_ms;
                ph.gpu_upload += tm.upload;
                ph.gpu_wait += tm.wait;
                ph.gpu_merge += tm.sink;
                Ok(())
            }
            Err(e) if !merged => Err(e),
            Err(e) => panic!("GPU accumulation failed after a partial merge: {e}"),
        }
    }

    /// Keeps only the diagonal neighbour candidates in `cand` that are really affected.
    ///
    /// A diagonal neighbour `d` is affected only if a voxel changed on the boundary edge or corner that `d` points to
    /// (`0` or `dim-1` on each non-zero axis of `d`). The in-radius test repeats the accumulation loop with the range
    /// widened by one voxel and a small radius margin, so it can only err towards marking too much.
    pub(crate) fn diag_mask(
        cfg: &Config,
        id: BlockId,
        bins: &[Bin],
        list: &[u32],
        cand: u32,
    ) -> u32 {
        let dc = DiagCtx::new(cfg, id);
        let todo = dc.todo(cand);
        let mut found = 0u32;
        for &i in list {
            found |= dc.hits(bins[i as usize].pos, &todo, found);
            if found == cand {
                break;
            }
        }
        found
    }

    /// Drops the preview layer of `seg` and marks every block it affected as dirty.
    ///
    /// # Returns
    ///
    /// Whether the segment had a preview layer.
    fn drop_layer(&mut self, seg: SegmentId) -> bool {
        let Some(layer) = self.pending.remove(&seg) else {
            return false;
        };
        for (id, blk) in &layer {
            mark_mask(&mut self.dirty, *id, blk.nbr);
        }
        true
    }

    /// Builds a block and simplifies it as configured, with seam vertices locked.
    fn build_simplified(&self, id: BlockId) -> (Vec<f32>, Vec<u8>, Vec<u32>) {
        let (p, c, t, seam) = self.build_block(id);
        crate::simplify::simplify_block(self.cfg.simplify_error, p, c, t, &seam)
    }

    /// Returns the base-layer distance at grid point `g` if it is observed (weight at least `min_weight`).
    fn base_value(&self, g: [i32; 3]) -> Option<f32> {
        let dim = self.cfg.block_dim;
        let id = [
            g[0].div_euclid(dim),
            g[1].div_euclid(dim),
            g[2].div_euclid(dim),
        ];
        let l = [
            g[0].rem_euclid(dim),
            g[1].rem_euclid(dim),
            g[2].rem_euclid(dim),
        ];
        let a = self.base.get(&id)?.get(l, dim / BR)?;
        (a.w >= self.cfg.min_weight).then(|| a.wd / a.w)
    }

    /// Trilinearly interpolates the base-layer distance at `x`. All eight corners must be observed.
    fn base_sample(&self, x: [f32; 3]) -> Option<f32> {
        let inv = 1.0 / self.cfg.voxel;
        let f = [x[0] * inv, x[1] * inv, x[2] * inv];
        let g = [
            f[0].floor() as i32,
            f[1].floor() as i32,
            f[2].floor() as i32,
        ];
        let t = [f[0] - g[0] as f32, f[1] - g[1] as f32, f[2] - g[2] as f32];
        let mut s = 0.0;
        for c in 0..8 {
            let o = [c & 1, (c >> 1) & 1, (c >> 2) & 1];
            let d = self.base_value([g[0] + o[0], g[1] + o[1], g[2] + o[2]])?;
            let mut w = 1.0;
            for a in 0..3 {
                w *= if o[a] == 1 { t[a] } else { 1.0 - t[a] };
            }
            s += w * d;
        }
        Some(s)
    }

    /// Returns how well a preview normal `n` at `p` agrees with the outward direction of the refined surface.
    ///
    /// Computed as the central difference `(D(p+hn) − D(p−hn)) / 2h` of the base field, clamped to `-1..=1`.
    /// Returns `None` where the base field is not observed.
    pub fn base_agreement(&self, p: [f32; 3], n: [f32; 3]) -> Option<f32> {
        let h = 1.5 * self.cfg.voxel;
        let a = self.base_sample([p[0] + h * n[0], p[1] + h * n[1], p[2] + h * n[2]])?;
        let b = self.base_sample([p[0] - h * n[0], p[1] - h * n[1], p[2] - h * n[2]])?;
        Some(((a - b) / (2.0 * h)).clamp(-1.0, 1.0))
    }

    /// Builds the mesh of one block from local voxels `-1..=dim`, summing the base layer and then the preview layers.
    ///
    /// The fourth element is the per-vertex seam flag: vertices in cell layer `-1` or `dim-1` are also produced by
    /// the neighbouring block. Uses the thread-local scratch buffers, which are recreated if the block size differs
    /// or a previous build did not finish cleanly.
    fn build_block(&self, id: BlockId) -> Built {
        let dim = self.cfg.block_dim;
        SCRATCH.with(|cell| {
            let mut slot = cell.borrow_mut();
            if !slot.as_ref().is_some_and(|s| s.dim == dim && s.clean) {
                *slot = Some(Scratch::new(dim));
            }
            let s = slot.as_mut().unwrap();
            s.clean = false;
            let out = self.build_block_in(id, s);
            s.reset();
            out
        })
    }

    /// Builds one block mesh using the scratch buffers `s`.
    ///
    /// 1. Copy. For each layer, the bricks of the 27 surrounding blocks are copied region by region into the padded
    ///    `(dim+2)³` array. Each padding cell comes from exactly one neighbour per layer, so the per-cell summation
    ///    order (base, then previews) is the same as when reading voxel by voxel. Voxels with `w = 0` are not added,
    ///    because `-0.0 + 0.0` would change the sign of zero; this is written as a select instead of a branch. The
    ///    destination offset is taken from the row start `ax.start`, since `ax.start - ax.lo` can be negative and
    ///    must not rely on `usize` wrap-around.
    /// 2. Distances. Only touched regions are evaluated; everything else stays NaN. The division runs without a
    ///    branch (`0/0 = NaN` when `w = 0`, but that value is never selected). For each row `(z, y)` two bit masks
    ///    record the cells with negative and with non-negative distance; NaN fails both comparisons and is in
    ///    neither. The masks are used only when a padded row fits in 64 bits (`dim + 2 <= 64`). Bit `b` is padded
    ///    coordinate `b`, i.e. local `x = b - 1`.
    /// 3. Quads. Cells are visited in `(z, y, x)` order, skipping only cells without a sign change on their `+x`,
    ///    `+y` or `+z` edge, so vertex numbering is the same as for a full scan. Each sign-changing edge emits a quad
    ///    from the vertices of the four cells around it. Vertices are created lazily per cell (cells `-1..dim-1`),
    ///    and cells without a vertex are remembered as well. Corner colours are computed only at the ends of
    ///    sign-changing edges, on first use.
    ///
    /// Winding: the four cells are listed counter-clockwise around `+e_k`. That order is kept when the distance
    /// increases along `+e_k` (inside to outside) and reversed otherwise, so triangles face outward.
    fn build_block_in(&self, id: BlockId, s: &mut Scratch) -> Built {
        let dim = self.cfg.block_dim;
        let nb = dim / BR;
        let nr = nb + 2;
        let p = (dim + 2) as usize;
        let Scratch {
            acc,
            sd,
            cell_vert,
            used_cells,
            region,
            neg,
            pos: posm,
            ..
        } = s;

        let mut any = false;
        let layers = std::iter::once(&self.base).chain(self.pending.values());
        for layer in layers {
            let mut nbr: [Option<&Block>; 27] = [None; 27];
            for (i, slot) in nbr.iter_mut().enumerate() {
                let i = i as i32;
                *slot = layer.get(&[
                    id[0] + i % 3 - 1,
                    id[1] + (i / 3) % 3 - 1,
                    id[2] + i / 9 - 1,
                ]);
            }
            for rz in 0..nr {
                let az = region_axis(rz, nb);
                for ry in 0..nr {
                    let ay = region_axis(ry, nb);
                    for rx in 0..nr {
                        let ax = region_axis(rx, nb);
                        let Some(blk) =
                            nbr[((az.off + 1) * 9 + (ay.off + 1) * 3 + ax.off + 1) as usize]
                        else {
                            continue;
                        };
                        let Some(br) = blk.bricks
                            [((az.brick * nb + ay.brick) * nb + ax.brick) as usize]
                            .as_ref()
                        else {
                            continue;
                        };
                        region[((rz * nr + ry) * nr + rx) as usize] |= R_TOUCHED;
                        for lz in az.lo..az.hi {
                            let pz = (az.start + lz - az.lo) as usize;
                            for ly in ay.lo..ay.hi {
                                let py = (ay.start + ly - ay.lo) as usize;
                                let src = ((lz * BR + ly) * BR) as usize;
                                let dst = (pz * p + py) * p + ax.start as usize;
                                let lo = ax.lo as usize;
                                for lx in lo..ax.hi as usize {
                                    let a = &br[src + lx];
                                    let on = a.w != 0.0;
                                    let t = &mut acc[dst + (lx - lo)];
                                    t.wd = if on { t.wd + a.wd } else { t.wd };
                                    t.w = if on { t.w + a.w } else { t.w };
                                    t.wc[0] = if on { t.wc[0] + a.wc[0] } else { t.wc[0] };
                                    t.wc[1] = if on { t.wc[1] + a.wc[1] } else { t.wc[1] };
                                    t.wc[2] = if on { t.wc[2] + a.wc[2] } else { t.wc[2] };
                                    any |= on;
                                }
                            }
                        }
                    }
                }
            }
        }
        if !any {
            return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        }

        let minw = self.cfg.min_weight;
        let use_mask = p <= 64;
        for (ri, &flags) in region.iter().enumerate() {
            if flags & R_TOUCHED == 0 {
                continue;
            }
            let ri = ri as i32;
            let (az, ay, ax) = (
                region_axis(ri / (nr * nr), nb),
                region_axis((ri / nr) % nr, nb),
                region_axis(ri % nr, nb),
            );
            for z in az.start..az.start + az.hi - az.lo {
                for y in ay.start..ay.start + ay.hi - ay.lo {
                    let row = z as usize * p + y as usize;
                    let (mut nm, mut pm) = (0u64, 0u64);
                    for x in ax.start as usize..(ax.start + ax.hi - ax.lo) as usize {
                        let i = row * p + x;
                        let a = &acc[i];
                        let ok = a.w >= minw;
                        let d = a.wd / a.w;
                        sd[i] = if ok { d } else { f32::NAN };
                        nm |= ((ok && d < 0.0) as u64) << (x & 63);
                        pm |= ((ok && d >= 0.0) as u64) << (x & 63);
                    }
                    if use_mask {
                        neg[row] |= nm;
                        posm[row] |= pm;
                    }
                }
            }
        }

        let idx = |x: i32, y: i32, z: i32| {
            (((z + 1) as usize * p + (y + 1) as usize) * p) + (x + 1) as usize
        };

        let pc = (dim + 1) as usize;
        let cidx = |x: i32, y: i32, z: i32| {
            (((z + 1) as usize * pc + (y + 1) as usize) * pc) + (x + 1) as usize
        };
        let mut pos: Vec<f32> = Vec::new();
        let mut seam: Vec<bool> = Vec::new();
        let mut col: Vec<u8> = Vec::new();
        let mut tri: Vec<u32> = Vec::new();
        let v = self.cfg.voxel;
        let g0 = [id[0] * dim, id[1] * dim, id[2] * dim];
        let sd = &*sd;
        let acc = &*acc;
        let (neg, posm) = (&*neg, &*posm);

        const CORNERS: [[i32; 3]; 8] = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ];
        const EDGES: [(usize, usize); 12] = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];

        let mut vertex = |c: [i32; 3], pos: &mut Vec<f32>, col: &mut Vec<u8>| -> Option<u32> {
            let ci = cidx(c[0], c[1], c[2]);
            match cell_vert[ci] {
                u32::MAX => {}
                NO_VERT => return None,
                vi => return Some(vi),
            }
            used_cells.push(ci as u32);
            let mut d = [0.0f32; 8];
            let mut ii = [0usize; 8];
            for (k, o) in CORNERS.iter().enumerate() {
                let i = idx(c[0] + o[0], c[1] + o[1], c[2] + o[2]);
                d[k] = sd[i];
                if d[k].is_nan() {
                    cell_vert[ci] = NO_VERT;
                    return None;
                }
                ii[k] = i;
            }
            let mut cc = [[0.0f32; 3]; 8];
            let mut have = 0u8;
            let mut s = [0.0f32; 3];
            let mut sc = [0.0f32; 3];
            let mut m = 0.0f32;
            for &(a, b) in &EDGES {
                if (d[a] < 0.0) == (d[b] < 0.0) {
                    continue;
                }
                for k in [a, b] {
                    if have & (1 << k) == 0 {
                        let w = &acc[ii[k]];
                        cc[k] = [w.wc[0] / w.w, w.wc[1] / w.w, w.wc[2] / w.w];
                        have |= 1 << k;
                    }
                }
                let t = d[a] / (d[a] - d[b]);
                for k in 0..3 {
                    s[k] += CORNERS[a][k] as f32 + t * (CORNERS[b][k] - CORNERS[a][k]) as f32;
                    sc[k] += cc[a][k] + t * (cc[b][k] - cc[a][k]);
                }
                m += 1.0;
            }
            if m == 0.0 {
                cell_vert[ci] = NO_VERT;
                return None;
            }
            let vi = (pos.len() / 3) as u32;
            for k in 0..3 {
                pos.push(((g0[k] + c[k]) as f32 + s[k] / m) * v);
                col.push((sc[k] / m).round().clamp(0.0, 255.0) as u8);
            }
            cell_vert[ci] = vi;
            seam.push(c.iter().any(|&x| x == -1 || x == dim - 1));
            Some(vi)
        };

        let mut quads = |x: i32, y: i32, z: i32| {
            let d0 = sd[idx(x, y, z)];
            if d0.is_nan() {
                return;
            }
            for k in 0..3usize {
                let mut e = [0i32; 3];
                e[k] = 1;
                let d1 = sd[idx(x + e[0], y + e[1], z + e[2])];
                if d1.is_nan() || (d0 < 0.0) == (d1 < 0.0) {
                    continue;
                }
                let i = (k + 1) % 3;
                let j = (k + 2) % 3;
                let mut ei = [0i32; 3];
                ei[i] = 1;
                let mut ej = [0i32; 3];
                ej[j] = 1;
                let c = [x, y, z];
                let cells = [
                    c,
                    [c[0] - ei[0], c[1] - ei[1], c[2] - ei[2]],
                    [
                        c[0] - ei[0] - ej[0],
                        c[1] - ei[1] - ej[1],
                        c[2] - ei[2] - ej[2],
                    ],
                    [c[0] - ej[0], c[1] - ej[1], c[2] - ej[2]],
                ];
                let mut q = [0u32; 4];
                let mut ok = true;
                for (t, cell) in cells.iter().enumerate() {
                    match vertex(*cell, &mut pos, &mut col) {
                        Some(vi) => q[t] = vi,
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    continue;
                }
                if d0 >= 0.0 {
                    q.swap(1, 3);
                }
                tri.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
            }
        };

        let inner: u64 = if use_mask {
            ((1u64 << dim) - 1) << 1
        } else {
            0
        };
        for z in 0..dim {
            for y in 0..dim {
                if !use_mask {
                    for x in 0..dim {
                        quads(x, y, z);
                    }
                    continue;
                }
                let r0 = (z + 1) as usize * p + (y + 1) as usize;
                let (n0, p0) = (neg[r0], posm[r0]);
                let (ny, py) = (neg[r0 + 1], posm[r0 + 1]);
                let (nz, pz) = (neg[r0 + p], posm[r0 + p]);
                let cx = (n0 & (p0 >> 1)) | (p0 & (n0 >> 1));
                let cy = (n0 & py) | (p0 & ny);
                let cz = (n0 & pz) | (p0 & nz);
                let mut cand = (cx | cy | cz) & inner;
                while cand != 0 {
                    let b = cand.trailing_zeros() as i32;
                    cand &= cand - 1;
                    quads(b - 1, y, z);
                }
            }
        }
        (pos, col, tri, seam)
    }
}

/// Per-block accumulation job: block id, the block taken out of its layer, indices of the bins touching it, and the
/// lower and upper corner of the local range changed by this call.
type Work = (BlockId, Block, Vec<u32>, [i32; 3], [i32; 3]);

/// Bin lists built from one chunk of bins: an index from block id into the lists, and the per-block lists in order
/// of first appearance.
type ChunkLists = (FxHashMap<BlockId, usize>, Vec<(BlockId, Vec<u32>)>);

/// Lists the bins touching each block, keeping bin order within each block. Shared by the CPU and GPU paths.
///
/// Bins are split into chunks. Each chunk builds its own lists in parallel, and each block then concatenates the
/// chunk lists in chunk order, so the result does not depend on the thread count. Within a chunk the bins are in cell
/// order and consecutive bins usually hit the same block, so the previous block is remembered to skip hash lookups.
/// Blocks are returned in order of first appearance.
pub(crate) fn block_lists(cfg: &Config, bins: &[Bin]) -> Vec<(BlockId, Vec<u32>)> {
    let v = cfg.voxel;
    let dim = cfg.block_dim;
    let r = cfg.splat_radius;
    let range = |p: f32| {
        (
            (((p - r) / v).ceil()) as i32,
            (((p + r) / v).floor()) as i32,
        )
    };
    let nch = rayon::current_num_threads().max(1) * 4;
    let ch = bins.len().div_ceil(nch).max(4096);
    let parts: Vec<ChunkLists> = bins
        .par_chunks(ch)
        .enumerate()
        .map(|(ci, chunk)| {
            let mut index: FxHashMap<BlockId, usize> = FxHashMap::default();
            let mut lists: Vec<(BlockId, Vec<u32>)> = Vec::new();
            let mut last: Option<(BlockId, usize)> = None;
            for (j, b) in chunk.iter().enumerate() {
                let i = (ci * ch + j) as u32;
                let rx = range(b.pos[0]);
                let ry = range(b.pos[1]);
                let rz = range(b.pos[2]);
                for bz in rz.0.div_euclid(dim)..=rz.1.div_euclid(dim) {
                    for by in ry.0.div_euclid(dim)..=ry.1.div_euclid(dim) {
                        for bx in rx.0.div_euclid(dim)..=rx.1.div_euclid(dim) {
                            let id = [bx, by, bz];
                            let k = match last {
                                Some((lid, k)) if lid == id => k,
                                _ => {
                                    let k = *index.entry(id).or_insert_with(|| {
                                        lists.push((id, Vec::new()));
                                        lists.len() - 1
                                    });
                                    last = Some((id, k));
                                    k
                                }
                            };
                            lists[k].1.push(i);
                        }
                    }
                }
            }
            (index, lists)
        })
        .collect();
    let mut order: Vec<BlockId> = Vec::new();
    let mut seen: FxHashMap<BlockId, ()> = FxHashMap::default();
    for (_, l) in &parts {
        for (id, _) in l {
            if seen.insert(*id, ()).is_none() {
                order.push(*id);
            }
        }
    }
    drop(seen);
    let lists: Vec<(BlockId, Vec<u32>)> = order
        .into_par_iter()
        .map(|id| {
            let mut v: Vec<u32> = Vec::new();
            for (index, l) in &parts {
                if let Some(&k) = index.get(&id) {
                    v.extend_from_slice(&l[k].1);
                }
            }
            (id, v)
        })
        .collect();
    drop(parts);
    lists
}

/// Per-block constants for diagonal neighbour tests, shared by `SdfMesher::diag_mask` and `diag_hits`.
pub(crate) struct DiagCtx {
    v: f32,
    dim: i32,
    r: f32,
    r2: f32,
    o: [i32; 3],
    /// Per-axis thresholds for being near the lower boundary face, with one voxel of margin. A bin beyond the
    /// threshold cannot reach any voxel on that face.
    lo_t: [f32; 3],
    /// Per-axis thresholds for being near the upper boundary face.
    hi_t: [f32; 3],
}

/// One diagonal neighbour candidate: its bit and the target global coordinate per axis (`None` for a free axis).
pub(crate) type DiagTodo = (u32, [Option<i32>; 3]);

/// All 20 diagonal neighbour bits, i.e. every bit except the face neighbours and the block itself.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) const DIAG_BITS: u32 = ((1u32 << 27) - 1) & !FACE_BITS;

impl DiagCtx {
    /// Builds the constants for block `id`. The squared radius gets a relative margin of `1e-4`.
    pub(crate) fn new(cfg: &Config, id: BlockId) -> Self {
        let v = cfg.voxel;
        let dim = cfg.block_dim;
        let r = cfg.splat_radius;
        let o = [id[0] * dim, id[1] * dim, id[2] * dim];
        Self {
            v,
            dim,
            r,
            r2: r * r * (1.0 + 1e-4),
            o,
            lo_t: std::array::from_fn(|a| (o[a] + 1) as f32 * v + r),
            hi_t: std::array::from_fn(|a| (o[a] + dim - 2) as f32 * v - r),
        }
    }

    /// Returns the target coordinates of every candidate bit in `cand`.
    pub(crate) fn todo(&self, cand: u32) -> Vec<DiagTodo> {
        let mut todo = Vec::new();
        for b in 0..27i32 {
            let bit = 1u32 << b;
            if cand & bit == 0 {
                continue;
            }
            let d = [b % 3 - 1, (b / 3) % 3 - 1, b / 9 - 1];
            let t = std::array::from_fn(|a| match d[a] {
                -1 => Some(self.o[a]),
                1 => Some(self.o[a] + self.dim - 1),
                _ => None,
            });
            todo.push((bit, t));
        }
        todo
    }

    /// Returns whether `p` is near at least two boundary faces.
    ///
    /// A bin that is not cannot reach any boundary edge or corner; most bins are rejected here.
    #[inline]
    pub(crate) fn near(&self, p: [f32; 3]) -> bool {
        (0..3)
            .filter(|&a| p[a] <= self.lo_t[a] || p[a] >= self.hi_t[a])
            .count()
            >= 2
    }

    /// Returns the `todo` bits whose boundary edge or corner a bin at `p` reaches, skipping bits in `skip`.
    ///
    /// Each bit is tested independently of the others.
    pub(crate) fn hits(&self, p: [f32; 3], todo: &[DiagTodo], skip: u32) -> u32 {
        if !self.near(p) {
            return 0;
        }
        let (v, r, o, dim) = (self.v, self.r, self.o, self.dim);
        let rng: [(i32, i32); 3] = std::array::from_fn(|a| {
            let lo = (((p[a] - r) / v).ceil() as i32 - 1).max(o[a]);
            let hi = (((p[a] + r) / v).floor() as i32 + 1).min(o[a] + dim - 1);
            (lo, hi)
        });
        let mut found = 0u32;
        for (bit, t) in todo {
            if (skip | found) & bit != 0 {
                continue;
            }
            let span: [(i32, i32); 3] = std::array::from_fn(|a| match t[a] {
                Some(g) => (g, g),
                None => rng[a],
            });
            if (0..3).any(|a| span[a].0 < rng[a].0 || span[a].1 > rng[a].1 || span[a].0 > span[a].1)
            {
                continue;
            }
            'hit: for gz in span[2].0..=span[2].1 {
                let dz = gz as f32 * v - p[2];
                for gy in span[1].0..=span[1].1 {
                    let dy = gy as f32 * v - p[1];
                    for gx in span[0].0..=span[0].1 {
                        let dx = gx as f32 * v - p[0];
                        if dx * dx + dy * dy + dz * dz <= self.r2 {
                            found |= bit;
                            break 'hit;
                        }
                    }
                }
            }
        }
        found
    }
}

/// Computes diagonal neighbour hits from the bin side.
///
/// For every block, returns the diagonal bits (within `DIAG_BITS`) whose boundary edge or corner is reached by at
/// least one bin touching the block. For every block `diag_mask(.., list, cand) == hits[id] & cand`, because blocks
/// are enumerated as in `block_lists` and OR does not depend on order. This allows counting per bin chunk in parallel
/// without per-block bin lists. Blocks missing from the result have no hits.
///
/// The cheap near-boundary test runs first; only bins that pass it look up the candidate table of the block.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn diag_hits(cfg: &Config, bins: &[Bin]) -> FxHashMap<BlockId, u32> {
    let v = cfg.voxel;
    let dim = cfg.block_dim;
    let r = cfg.splat_radius;
    let range = |p: f32| {
        (
            (((p - r) / v).ceil()) as i32,
            (((p + r) / v).floor()) as i32,
        )
    };
    let ch = bins
        .len()
        .div_ceil(rayon::current_num_threads().max(1) * 4)
        .max(4096);
    bins.par_chunks(ch)
        .map(|chunk| {
            let mut out: FxHashMap<BlockId, (Vec<DiagTodo>, u32)> = FxHashMap::default();
            for b in chunk {
                let rx = range(b.pos[0]);
                let ry = range(b.pos[1]);
                let rz = range(b.pos[2]);
                for bz in rz.0.div_euclid(dim)..=rz.1.div_euclid(dim) {
                    for by in ry.0.div_euclid(dim)..=ry.1.div_euclid(dim) {
                        for bx in rx.0.div_euclid(dim)..=rx.1.div_euclid(dim) {
                            let id = [bx, by, bz];
                            let dc = DiagCtx::new(cfg, id);
                            if !dc.near(b.pos) {
                                continue;
                            }
                            let e = out.entry(id).or_insert_with(|| (dc.todo(DIAG_BITS), 0));
                            if e.1 != DIAG_BITS {
                                e.1 |= dc.hits(b.pos, &e.0, e.1);
                            }
                        }
                    }
                }
            }
            out.into_iter()
                .filter(|x| x.1.1 != 0)
                .map(|(id, (_, m))| (id, m))
                .collect::<FxHashMap<BlockId, u32>>()
        })
        .reduce(FxHashMap::default, |mut a, b| {
            for (id, m) in b {
                *a.entry(id).or_default() |= m;
            }
            a
        })
}

/// A built block mesh: positions, colours, indices and the per-vertex seam flag.
type Built = (Vec<f32>, Vec<u8>, Vec<u32>, Vec<bool>);

/// `cell_vert` marker for a cell without a vertex (an unobserved corner or no sign change).
const NO_VERT: u32 = u32::MAX - 1;
/// Region flag: bricks were copied into this region.
const R_TOUCHED: u8 = 1;

/// One axis of a region of the padded array, which is split into brick-sized regions.
///
/// Region 0 is local coordinate `-1`, regions `1..=nb` are the bricks and region `nb+1` is local `dim`.
struct RegionAxis {
    /// Offset of the source block (-1, 0 or 1).
    off: i32,
    /// Brick index inside the source block.
    brick: i32,
    /// Start of the voxel range `[lo, hi)` inside the brick.
    lo: i32,
    /// End of the voxel range inside the brick.
    hi: i32,
    /// Start coordinate in the padded array.
    start: i32,
}

/// Returns the axis description of region `r` for blocks of `nb` bricks per edge.
#[inline]
fn region_axis(r: i32, nb: i32) -> RegionAxis {
    if r == 0 {
        RegionAxis {
            off: -1,
            brick: nb - 1,
            lo: BR - 1,
            hi: BR,
            start: 0,
        }
    } else if r == nb + 1 {
        RegionAxis {
            off: 1,
            brick: 0,
            lo: 0,
            hi: 1,
            start: nb * BR + 1,
        }
    } else {
        RegionAxis {
            off: 0,
            brick: r - 1,
            lo: 0,
            hi: BR,
            start: 1 + (r - 1) * BR,
        }
    }
}

/// Calls `f(row, x)` for every padded-array cell in region `ri`, where `row = z·p + y` and the cell index is
/// `row·p + x`.
#[inline]
fn for_region_cells(ri: i32, nb: i32, p: usize, mut f: impl FnMut(usize, usize)) {
    let nr = nb + 2;
    let (az, ay, ax) = (
        region_axis(ri / (nr * nr), nb),
        region_axis((ri / nr) % nr, nb),
        region_axis(ri % nr, nb),
    );
    for z in az.start..az.start + az.hi - az.lo {
        for y in ay.start..ay.start + ay.hi - ay.lo {
            let row = z as usize * p + y as usize;
            for x in ax.start..ax.start + ax.hi - ax.lo {
                f(row, x as usize);
            }
        }
    }
}

/// Per-thread work buffers reused by `build_block`.
///
/// After each build only the cells that were written are restored to their defaults.
struct Scratch {
    dim: i32,
    /// `true` once `reset` has completed.
    clean: bool,
    /// Padded accumulator array, `(dim+2)³`. Default zero.
    acc: Vec<Acc>,
    /// Distance values, same layout as `acc`. Default NaN.
    sd: Vec<f32>,
    /// Vertex index per cell, `(dim+1)³`. Default `u32::MAX`; `NO_VERT` means the cell has no vertex.
    cell_vert: Vec<u32>,
    /// Cells whose `cell_vert` entry was written, for resetting.
    used_cells: Vec<u32>,
    /// Flags per region (`R_TOUCHED`).
    region: Vec<u8>,
    /// Bits of cells with negative distance per row `(z, y)`, `(dim+2)²`. Default zero.
    neg: Vec<u64>,
    /// Bits of cells with non-negative distance per row `(z, y)`. Default zero.
    pos: Vec<u64>,
}

impl Scratch {
    /// Allocates buffers for blocks of `dim`³ voxels.
    fn new(dim: i32) -> Self {
        let p = (dim + 2) as usize;
        let pc = (dim + 1) as usize;
        let nr = (dim / BR + 2) as usize;
        Self {
            dim,
            clean: true,
            acc: vec![Acc::default(); p * p * p],
            sd: vec![f32::NAN; p * p * p],
            cell_vert: vec![u32::MAX; pc * pc * pc],
            used_cells: Vec::new(),
            region: vec![0; nr * nr * nr],
            neg: vec![0; p * p],
            pos: vec![0; p * p],
        }
    }

    /// Restores the defaults of every cell written by the last build.
    fn reset(&mut self) {
        let nb = self.dim / BR;
        let p = (self.dim + 2) as usize;
        for ri in 0..self.region.len() {
            if self.region[ri] & R_TOUCHED != 0 {
                let (acc, sd) = (&mut self.acc, &mut self.sd);
                for_region_cells(ri as i32, nb, p, |row, x| {
                    acc[row * p + x] = Acc::default();
                    sd[row * p + x] = f32::NAN;
                });
            }
            self.region[ri] = 0;
        }
        for &ci in &self.used_cells {
            self.cell_vert[ci as usize] = u32::MAX;
        }
        self.used_cells.clear();
        self.neg.fill(0);
        self.pos.fill(0);
        self.clean = true;
    }
}

thread_local! {
    /// Scratch buffers of `build_block`, one per thread.
    static SCRATCH: std::cell::RefCell<Option<Scratch>> = const { std::cell::RefCell::new(None) };
}

impl Mesher for SdfMesher {
    fn ingest(
        &mut self,
        seg: SegmentId,
        level: Level,
        pts: &[Point],
    ) -> Result<IngestStats, IngestError> {
        if self.refined.contains(&seg) {
            return Err(match level {
                Level::Refined => IngestError::AlreadyRefined(seg),
                Level::Preview => IngestError::PreviewAfterRefined(seg),
            });
        }
        let policy = match level {
            Level::Preview => NormalPolicy::Estimate,
            Level::Refined => NormalPolicy::Trust,
        };
        let cfg = self.cfg;
        let bin_only_ms = if std::env::var_os("DELTAMESH_PROF_SPLIT").is_some() {
            let t = Instant::now();
            let r = bin_points(
                pts,
                cfg.bin,
                NormalPolicy::Ignore,
                cfg.normal_radius_bins,
                cfg.normal_min_neighbors,
            );
            let e = ms(t);
            drop(r);
            Some(e)
        } else {
            None
        };
        let t_bin = Instant::now();
        let (mut bins, bs) = bin_points(
            pts,
            cfg.bin,
            policy,
            cfg.normal_radius_bins,
            cfg.normal_min_neighbors,
        );
        let bin_ms = ms(t_bin);
        let t_orient = Instant::now();
        if !bs.fixed.is_empty() {
            let oracle = |p: [f32; 3], n: [f32; 3]| self.base_agreement(p, n);
            let use_oracle = cfg.orient_oracle && !self.base.is_empty();
            orient_normals(
                &mut bins,
                &bs.fixed,
                cfg.bin,
                cfg.orient,
                use_oracle.then_some(&oracle as Oracle),
            );
        }
        let orient_ms = ms(t_orient);
        let t_drop = Instant::now();
        let removed = self.drop_layer(seg);
        {
            let mut ph = PHASE.lock().unwrap();
            ph.drop_layer += ms(t_drop);
            ph.orient += orient_ms;
            ph.calls += 1;
            match bin_only_ms {
                Some(b) => {
                    let b = b.min(bin_ms);
                    ph.binning += b;
                    ph.normals += bin_ms - b;
                }
                None => ph.binning += bin_ms,
            }
        }
        match level {
            Level::Preview => {
                let mut layer = Layer::default();
                Self::integrate(
                    &cfg,
                    &mut layer,
                    &bins,
                    &mut self.dirty,
                    self.gpu.as_deref(),
                );
                self.pending.insert(seg, layer);
            }
            Level::Refined => {
                Self::integrate(
                    &cfg,
                    &mut self.base,
                    &bins,
                    &mut self.dirty,
                    self.gpu.as_deref(),
                );
                self.refined.insert(seg);
            }
        }
        Ok(IngestStats {
            input_points: pts.len(),
            bins: bins.len(),
            normals_estimated: bs.estimated,
            bins_dropped: bs.dropped,
            removed_preview: removed,
            dirty_blocks: self.dirty.len(),
        })
    }

    fn extract(&mut self) -> Vec<BlockId> {
        let ids = drain_dirty(&mut self.dirty);
        let built: Vec<_> = ids
            .par_iter()
            .map(|&(id, direct)| {
                let m = self.build_simplified(id);
                let same = same_mesh(self.meshes.get(&id), &m.0, &m.1, &m.2);
                (id, direct, same, m)
            })
            .collect();
        apply_built(
            built,
            &mut self.meshes,
            &mut self.versions,
            &mut self.last_extract,
        )
    }

    fn version(&self, id: &BlockId) -> u32 {
        self.versions.get(id).copied().unwrap_or(0)
    }

    fn extract_stats(&self) -> ExtractStats {
        self.last_extract
    }

    fn mesh(&self, id: &BlockId) -> Option<&BlockMesh> {
        self.meshes.get(id)
    }

    fn meshes(&self) -> Box<dyn Iterator<Item = &BlockMesh> + '_> {
        Box::new(self.meshes.values())
    }

    fn field_bytes(&self) -> usize {
        let per_brick = std::mem::size_of::<[Acc; BR3]>();
        let nbr = ((self.cfg.block_dim / BR).pow(3)) as usize;
        let layer = |l: &Layer| {
            l.values()
                .map(|b| b.brick_count() * per_brick + nbr * std::mem::size_of::<Option<Brick>>())
                .sum::<usize>()
        };
        layer(&self.base) + self.pending.values().map(layer).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::testutil::plane;

    fn all_meshes(m: &SdfMesher) -> Vec<BlockMesh> {
        let mut v: Vec<BlockMesh> = m.meshes().cloned().collect();
        v.sort_by_key(|b| b.id);
        v
    }

    #[test]
    fn flat_plane_is_flat_and_faces_up() {
        let mut m = SdfMesher::new(Config::default());
        m.ingest(
            0,
            Level::Refined,
            &plane(-5.0, 5.0, -5.0, 5.0, 0.03, 0.05, [200, 100, 50]),
        )
        .unwrap();
        m.extract();
        assert!(m.total_tris() > 1000);
        for b in m.meshes() {
            for p in b.positions.chunks(3) {
                assert!((p[2] - 0.03).abs() < 0.02, "z {}", p[2]);
            }
            for t in b.indices.chunks(3) {
                let g = |i: u32| &b.positions[i as usize * 3..i as usize * 3 + 3];
                let (a, bb, c) = (g(t[0]), g(t[1]), g(t[2]));
                let u = [bb[0] - a[0], bb[1] - a[1]];
                let w = [c[0] - a[0], c[1] - a[1]];
                assert!(u[0] * w[1] - u[1] * w[0] > 0.0, "triangle faces down");
            }
            for c in b.colors.chunks(3) {
                assert_eq!(c, [200, 100, 50]);
            }
        }
    }

    /// A preview wall standing on a refined floor uses both the oracle (existing field) and propagation to orient
    /// normals. The result must not depend on the thread count.
    #[test]
    fn preview_orientation_is_thread_independent() {
        let floor = plane(-4.0, 4.0, -4.0, 4.0, 0.0, 0.05, [9, 90, 9]);
        let mut wall = plane(-3.0, 3.0, -3.0, 3.0, 0.0, 0.1, [9, 9, 90]);
        for i in 0..60 {
            for k in 0..30 {
                let (y, z) = (-3.0 + i as f32 * 0.1, k as f32 * 0.1);
                wall.push(Point {
                    pos: [0.5, y, z],
                    rgb: [200, 9, 9],
                    normal: [f32::NAN; 3],
                });
                wall.push(Point {
                    pos: [0.5 + 0.03 * ((i * 7 + k) % 3) as f32, y + 0.05, z + 0.05],
                    rgb: [200, 9, 9],
                    normal: [f32::NAN; 3],
                });
            }
        }
        let run = |t: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap();
            pool.install(|| {
                let mut m = SdfMesher::new(Config::default());
                m.ingest(0, Level::Refined, &floor).unwrap();
                m.ingest(1, Level::Preview, &wall).unwrap();
                m.extract();
                all_meshes(&m)
            })
        };
        let a = run(1);
        assert!(a.iter().map(|m| m.tri_count()).sum::<usize>() > 0);
        assert_eq!(a, run(4));
    }

    /// A plane crossing block boundaries (every 6.4 m). After merging vertices by position, open edges may only lie
    /// on the outer border.
    #[test]
    fn seams_share_exact_vertices() {
        let mut m = SdfMesher::new(Config::default());
        m.ingest(
            0,
            Level::Refined,
            &plane(2.0, 11.0, -4.0, 9.0, 0.03, 0.05, [1, 2, 3]),
        )
        .unwrap();
        m.extract();
        assert!(m.block_count() >= 4);
        let mut key: FxHashMap<[u32; 3], u32> = FxHashMap::default();
        let mut verts: Vec<[f32; 3]> = Vec::new();
        let mut edges: FxHashMap<(u32, u32), i32> = FxHashMap::default();
        for b in m.meshes() {
            let map: Vec<u32> = b
                .positions
                .chunks(3)
                .map(|p| {
                    let k = [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
                    *key.entry(k).or_insert_with(|| {
                        verts.push([p[0], p[1], p[2]]);
                        verts.len() as u32 - 1
                    })
                })
                .collect();
            for t in b.indices.chunks(3) {
                for (a, c) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    let (a, c) = (map[a as usize], map[c as usize]);
                    *edges.entry((a.min(c), a.max(c))).or_default() += 1;
                }
            }
        }
        for ((a, c), n) in edges {
            if n == 1 {
                for p in [verts[a as usize], verts[c as usize]] {
                    let border = p[0] < 2.6 || p[0] > 10.4 || p[1] < -3.4 || p[1] > 8.4;
                    assert!(border, "open edge inside the surface: {p:?}");
                }
            }
        }
    }

    #[test]
    fn replace_preview_is_exact() {
        let refined = plane(-3.0, 9.0, -3.0, 3.0, 0.0, 0.05, [100, 100, 100]);
        let preview = plane(-3.0, 4.0, -3.0, 3.0, 0.4, 0.1, [10, 10, 10]);
        let other = plane(5.0, 15.0, -2.0, 2.0, 0.1, 0.05, [50, 60, 70]);

        let mut a = SdfMesher::new(Config::default());
        a.ingest(1, Level::Refined, &other).unwrap();
        a.ingest(0, Level::Preview, &preview).unwrap();
        a.extract();
        assert!(a.total_tris() > 0);
        let st = a.ingest(0, Level::Refined, &refined).unwrap();
        assert!(st.removed_preview);
        a.extract();

        let mut b = SdfMesher::new(Config::default());
        b.ingest(1, Level::Refined, &other).unwrap();
        b.ingest(0, Level::Refined, &refined).unwrap();
        b.extract();

        let (ma, mb) = (all_meshes(&a), all_meshes(&b));
        assert_eq!(ma.len(), mb.len());
        for (x, y) in ma.iter().zip(&mb) {
            assert_eq!(x.id, y.id);
            assert_eq!(x.positions, y.positions);
            assert_eq!(x.colors, y.colors);
            assert_eq!(x.indices, y.indices);
        }
        assert_eq!(a.field_bytes(), b.field_bytes());
    }

    #[test]
    fn no_faces_in_unobserved_gap() {
        let mut pts = plane(0.0, 4.0, 0.0, 4.0, 0.0, 0.05, [9, 9, 9]);
        pts.extend(plane(7.0, 11.0, 0.0, 4.0, 0.0, 0.05, [9, 9, 9]));
        let mut m = SdfMesher::new(Config::default());
        m.ingest(0, Level::Refined, &pts).unwrap();
        m.extract();
        for b in m.meshes() {
            for p in b.positions.chunks(3) {
                assert!(
                    p[0] < 4.0 + 0.4 || p[0] > 7.0 - 0.4,
                    "vertex in the gap at x={}",
                    p[0]
                );
            }
        }
    }

    /// Two observations of the same surface 8 cm apart must merge into one sheet. A separate second sheet would
    /// nearly double the triangle count; a few extra border cells are allowed.
    #[test]
    fn double_coverage_does_not_thicken() {
        let one = plane(0.0, 6.0, 0.0, 6.0, 0.0, 0.05, [9, 9, 9]);
        let two = plane(0.0, 6.0, 0.0, 6.0, 0.08, 0.05, [9, 9, 9]);
        let mut a = SdfMesher::new(Config::default());
        a.ingest(0, Level::Refined, &one).unwrap();
        a.extract();
        let single = a.total_tris();
        a.ingest(1, Level::Refined, &two).unwrap();
        a.extract();
        let both = a.total_tris();
        assert!((both as f64) < single as f64 * 1.02, "{single} -> {both}");
        for b in a.meshes() {
            for p in b.positions.chunks(3) {
                assert!(p[2] > 0.0 && p[2] < 0.08, "z {}", p[2]);
            }
        }
    }

    /// Per-voxel reference accumulation, compared bitwise with `integrate`.
    ///
    /// Builds the dirty set in two ways. `exact` marks, for every changed voxel, only the neighbours that read it:
    /// on each axis the `-1` neighbour if the voxel is at `0` and the `+1` neighbour if it is at `dim-1`, never
    /// `(+1,+1,+1)` since that one does not affect the mesh. This is the exact minimal set. `boxed` applies the range
    /// rule of `halo_mask`, a conservative upper bound.
    fn integrate_ref(
        cfg: &Config,
        layer: &mut Layer,
        bins: &[Bin],
        exact: &mut Dirty,
        boxed: &mut Dirty,
    ) {
        let (v, dim, r) = (cfg.voxel, cfg.block_dim, cfg.splat_radius);
        let r2 = r * r;
        let inv2s2 = 1.0 / (2.0 * (r * 0.5) * (r * 0.5));
        let range = |p: f32| {
            (
                (((p - r) / v).ceil()) as i32,
                (((p + r) / v).floor()) as i32,
            )
        };
        let nb = dim / BR;
        let mut touched: FxHashMap<BlockId, ([i32; 3], [i32; 3], u32)> = FxHashMap::default();
        for b in bins {
            let (rx, ry, rz) = (range(b.pos[0]), range(b.pos[1]), range(b.pos[2]));
            for gz in rz.0..=rz.1 {
                let dz = gz as f32 * v - b.pos[2];
                for gy in ry.0..=ry.1 {
                    let dy = gy as f32 * v - b.pos[1];
                    for gx in rx.0..=rx.1 {
                        let dx = gx as f32 * v - b.pos[0];
                        let q = dx * dx + dy * dy + dz * dz;
                        if q > r2 {
                            continue;
                        }
                        let w = (-q * inv2s2).exp();
                        let d = b.normal[0] * dx + b.normal[1] * dy + b.normal[2] * dz;
                        let g = [gx, gy, gz];
                        let id = [gx.div_euclid(dim), gy.div_euclid(dim), gz.div_euclid(dim)];
                        let l = [g[0] - id[0] * dim, g[1] - id[1] * dim, g[2] - id[2] * dim];
                        let blk = layer.entry(id).or_insert_with(|| Block::new(dim));
                        let a = blk.get_mut(l, nb);
                        a.wd += w * d;
                        a.w += w;
                        a.wc[0] += w * b.rgb[0];
                        a.wc[1] += w * b.rgb[1];
                        a.wc[2] += w * b.rgb[2];
                        let t = touched
                            .entry(id)
                            .or_insert(([i32::MAX; 3], [i32::MIN; 3], 0));
                        let opts = |c: i32| {
                            [
                                0,
                                if c == 0 { -1 } else { 0 },
                                if c == dim - 1 { 1 } else { 0 },
                            ]
                        };
                        for dz in opts(l[2]) {
                            for dy in opts(l[1]) {
                                for dx in opts(l[0]) {
                                    if [dx, dy, dz] != [1, 1, 1] {
                                        t.2 |= crate::nbit(dx, dy, dz);
                                    }
                                }
                            }
                        }
                        for (k, &lk) in l.iter().enumerate() {
                            blk.min[k] = blk.min[k].min(lk);
                            blk.max[k] = blk.max[k].max(lk);
                            t.0[k] = t.0[k].min(lk);
                            t.1[k] = t.1[k].max(lk);
                        }
                    }
                }
            }
        }
        for (id, (tmin, tmax, m)) in touched {
            mark_mask(exact, id, m);
            mark_mask(boxed, id, halo_mask(tmin, tmax, dim, Halo::Both));
        }
    }

    /// `integrate` must match the per-voxel reference bitwise.
    ///
    /// Uses a tilted surface with bins near block boundaries (positions and normals from a simple LCG), several
    /// voxel sizes, and two calls so that adding into existing blocks is covered. The dirty set of `integrate` must
    /// contain the exact set and stay within the box rule, with the same direct flags. `integrate` also creates empty
    /// blocks for blocks that only its candidate box touches; those must stay empty.
    #[test]
    fn integrate_matches_per_voxel_reference() {
        let mut s = 12345u32;
        let mut rnd = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 24) as f32
        };
        let mut bins = Vec::new();
        for _ in 0..3000 {
            let (x, y) = (rnd() * 16.0 - 8.0, rnd() * 16.0 - 8.0);
            let n = [rnd() - 0.5, rnd() - 0.5, 1.0];
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            bins.push(Bin {
                pos: [x, y, 0.3 * x - 0.2 * y + rnd() * 0.3],
                normal: [n[0] / l, n[1] / l, n[2] / l],
                rgb: [rnd() * 255.0, rnd() * 255.0, rnd() * 255.0],
                count: 1,
            });
        }
        for voxel in [0.2f32, 0.1, 0.15] {
            let cfg = Config {
                voxel,
                ..Config::default()
            };
            let (mut la, mut lb) = (Layer::default(), Layer::default());
            let (mut da, mut db, mut dbox) = (Dirty::default(), Dirty::default(), Dirty::default());
            for part in [&bins[..1500], &bins[1500..]] {
                SdfMesher::integrate(&cfg, &mut la, part, &mut da, None);
                integrate_ref(&cfg, &mut lb, part, &mut db, &mut dbox);
            }
            for (id, direct) in &db {
                assert_eq!(da.get(id), Some(direct), "{id:?}");
            }
            for (id, direct) in &da {
                assert_eq!(dbox.get(id), Some(direct), "{id:?}");
            }
            assert!(lb.keys().all(|id| la.contains_key(id)));
            for (id, a) in &la {
                let Some(b) = lb.get(id) else {
                    assert!(a.brick_count() == 0 && a.min[0] > a.max[0], "{id:?}");
                    continue;
                };
                assert_eq!((a.min, a.max), (b.min, b.max), "{id:?}");
                for (x, y) in a.bricks.iter().zip(&b.bricks) {
                    assert_eq!(x.is_some(), y.is_some());
                    if let (Some(x), Some(y)) = (x, y) {
                        for (p, q) in x.iter().zip(y.iter()) {
                            assert_eq!(
                                [p.wd, p.w, p.wc[0], p.wc[1], p.wc[2]].map(f32::to_bits),
                                [q.wd, q.w, q.wc[0], q.wc[1], q.wc[2]].map(f32::to_bits)
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn thread_count_does_not_change_result() {
        let pts = plane(-7.0, 7.0, -7.0, 7.0, 0.0, 0.05, [9, 90, 9]);
        let run = |t: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap();
            pool.install(|| {
                let mut m = SdfMesher::new(Config::default());
                m.ingest(0, Level::Refined, &pts).unwrap();
                m.extract();
                all_meshes(&m)
            })
        };
        assert_eq!(run(1), run(4));
    }

    fn simp() -> Config {
        Config {
            simplify_error: 0.05,
            ..Config::default()
        }
    }

    /// Merges vertices by position and asserts that every endpoint of an open edge satisfies `border`.
    fn assert_no_inner_open_edges(m: &SdfMesher, border: impl Fn([f32; 3]) -> bool) {
        let mut key: FxHashMap<[u32; 3], u32> = FxHashMap::default();
        let mut verts: Vec<[f32; 3]> = Vec::new();
        let mut edges: FxHashMap<(u32, u32), i32> = FxHashMap::default();
        for b in m.meshes() {
            let map: Vec<u32> = b
                .positions
                .chunks(3)
                .map(|p| {
                    let k = [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
                    *key.entry(k).or_insert_with(|| {
                        verts.push([p[0], p[1], p[2]]);
                        verts.len() as u32 - 1
                    })
                })
                .collect();
            for t in b.indices.chunks(3) {
                for (a, c) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    let (a, c) = (map[a as usize], map[c as usize]);
                    *edges.entry((a.min(c), a.max(c))).or_default() += 1;
                }
            }
        }
        for ((a, c), n) in edges {
            if n == 1 {
                for p in [verts[a as usize], verts[c as usize]] {
                    assert!(border(p), "open edge inside the surface: {p:?}");
                }
            }
        }
    }

    /// A tilted wavy surface spanning several blocks. After simplification there must be no gaps at block
    /// boundaries, and the remaining vertices must be a subset of the unsimplified ones with the same colours.
    #[test]
    fn simplified_seams_share_exact_vertices() {
        let mut pts = Vec::new();
        let (mut x, step) = (-9.0f32, 0.05f32);
        while x < 9.0 {
            let mut y = -9.0f32;
            while y < 9.0 {
                let z = 0.3 * (x * 0.7).sin() + 0.1 * y;
                let n = [-0.21 * (x * 0.7).cos(), -0.1, 1.0];
                let l = (n[0] * n[0] + n[1] * n[1] + 1.0f32).sqrt();
                pts.push(Point {
                    pos: [x, y, z],
                    rgb: [9, 9, 9],
                    normal: [n[0] / l, n[1] / l, 1.0 / l],
                });
                y += step;
            }
            x += step;
        }
        let mut full = SdfMesher::new(Config::default());
        full.ingest(0, Level::Refined, &pts).unwrap();
        full.extract();
        let mut m = SdfMesher::new(simp());
        m.ingest(0, Level::Refined, &pts).unwrap();
        m.extract();
        assert!(m.block_count() >= 9);
        assert!(
            m.total_tris() * 2 < full.total_tris(),
            "{} vs {}",
            m.total_tris(),
            full.total_tris()
        );
        assert_no_inner_open_edges(&m, |p| {
            p[0] < -8.4 || p[0] > 8.4 || p[1] < -8.4 || p[1] > 8.4
        });
        let mut orig: FxHashMap<[u32; 3], [u8; 3]> = FxHashMap::default();
        for b in full.meshes() {
            for (p, c) in b.positions.chunks(3).zip(b.colors.chunks(3)) {
                orig.insert(
                    [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()],
                    [c[0], c[1], c[2]],
                );
            }
        }
        for b in m.meshes() {
            assert!(b.indices.iter().all(|&i| (i as usize) < b.vertex_count()));
            for (p, c) in b.positions.chunks(3).zip(b.colors.chunks(3)) {
                assert_eq!(
                    orig.get(&[p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]),
                    Some(&[c[0], c[1], c[2]])
                );
            }
        }
    }

    #[test]
    fn simplified_replace_and_threads_are_exact() {
        let refined = plane(-3.0, 9.0, -3.0, 3.0, 0.0, 0.05, [100, 100, 100]);
        let preview = plane(-3.0, 4.0, -3.0, 3.0, 0.4, 0.1, [10, 10, 10]);
        let other = plane(5.0, 15.0, -2.0, 2.0, 0.1, 0.05, [50, 60, 70]);
        let run = |t: usize, with_preview: bool| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap();
            pool.install(|| {
                let mut a = SdfMesher::new(simp());
                a.ingest(1, Level::Refined, &other).unwrap();
                if with_preview {
                    a.ingest(0, Level::Preview, &preview).unwrap();
                    a.extract();
                }
                a.ingest(0, Level::Refined, &refined).unwrap();
                a.extract();
                all_meshes(&a)
                    .into_iter()
                    .map(|m| (m.id, m.positions, m.colors, m.indices))
                    .collect::<Vec<_>>()
            })
        };
        let base = run(1, false);
        assert_eq!(run(1, true), base);
        assert_eq!(run(4, true), base);
    }

    use crate::testutil::{Batch, check_incremental};

    fn pt(p: [f32; 3], n: [f32; 3]) -> Point {
        Point {
            pos: p,
            rgb: [200, 10, 10],
            normal: n,
        }
    }

    /// Points that change only boundary voxels must refresh the neighbours that read those voxels.
    ///
    /// With a 0.4 m radius each probe reaches a boundary voxel (local 0 or 31) but no voxel of the neighbouring
    /// block, so the neighbour's own field is unchanged although it must be re-extracted. In order, the probes hit:
    /// local x 31 (`+x` neighbour), local x 0 (`-x`), the `(31,31)` edge (diagonal `(+1,+1,0)`), the same edge at
    /// about 95 % of the squared radius (radius boundary test), the `(0,0)` edge (`(-1,-1,0)`), the `(0,0,0)` corner
    /// (`(-1,-1,-1)`) and `(31,0,0)` (`(+1,-1,-1)`). The test is only meaningful if some neighbour that was not
    /// touched directly actually changed.
    #[test]
    fn boundary_voxel_change_refreshes_reader() {
        let base = plane(-4.0, 11.0, -4.0, 11.0, 0.03, 0.05, [1, 2, 3]);
        let up = [0.0, 0.0, 1.0];
        let probes = [
            [5.9, 2.0, 0.03],
            [0.25, 2.0, 0.03],
            [5.95, 5.95, 0.03],
            [5.92, 5.93, 0.03],
            [0.22, 0.22, 0.03],
            [0.22, 0.22, 0.1],
            [5.95, 0.22, 0.1],
        ];
        let mut steps: Vec<Vec<Batch>> = vec![vec![(0, Level::Refined, base)]];
        for (k, p) in probes.iter().enumerate() {
            steps.push(vec![(k as u32 + 1, Level::Refined, vec![pt(*p, up)])]);
        }
        let log = check_incremental(|| SdfMesher::new(Config::default()), &steps);
        let nbr_changed: usize = log[1..]
            .iter()
            .map(|(_, s)| s.reextracted - s.reextracted_direct - s.unchanged_neighbor)
            .sum();
        assert!(nbr_changed > 0, "{log:?}");
    }

    /// Random points near the block corner (6.4, 6.4, 6.4), added step by step, must match a full rebuild; a missed
    /// diagonal neighbour would make them differ. Even steps send a preview that the next step replaces with the
    /// refined segment.
    #[test]
    fn random_corner_points_match_full_rebuild() {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        let mut steps: Vec<Vec<Batch>> = Vec::new();
        for k in 0..12u32 {
            let mut pts = Vec::new();
            for _ in 0..40 {
                let p = [
                    6.4 + (rnd() - 0.5) * 1.6,
                    6.4 + (rnd() - 0.5) * 1.6,
                    6.4 + (rnd() - 0.5) * 1.6,
                ];
                let n = [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5];
                let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-3);
                pts.push(pt(p, [n[0] / l, n[1] / l, n[2] / l]));
            }
            let lv = if k % 2 == 0 {
                Level::Preview
            } else {
                Level::Refined
            };
            steps.push(vec![(k / 2, lv, pts)]);
        }
        check_incremental(|| SdfMesher::new(Config::default()), &steps);
    }

    /// `diag_hits` (counted per bin) must equal `diag_mask` (counted per block list) for every candidate mask.
    ///
    /// Bins cluster around block corners and edges, plus a wide scatter, including negative coordinates. Candidates
    /// are the full set, random subsets and the masks produced by the range rule.
    #[test]
    fn diag_hits_match_diag_mask() {
        let mut s = 0x2545F4914F6CDD1Du64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        for cfg in [
            Config::default(),
            Config {
                voxel: 0.05,
                bin: 0.025,
                splat_radius: 0.1,
                ..Config::default()
            },
        ] {
            let edge = cfg.voxel * cfg.block_dim as f32;
            let mut bins: Vec<Bin> = Vec::new();
            for k in 0..20000 {
                let c = [
                    ((k % 5) as f32 - 2.0) * edge,
                    ((k / 5 % 5) as f32 - 2.0) * edge,
                    ((k / 25 % 3) as f32 - 1.0) * edge,
                ];
                let sp = if k % 3 == 0 {
                    4.0 * edge
                } else {
                    3.0 * cfg.splat_radius
                };
                let pos = [
                    c[0] + (rnd() - 0.5) * sp,
                    c[1] + (rnd() - 0.5) * sp,
                    c[2] + (rnd() - 0.5) * sp,
                ];
                bins.push(Bin {
                    pos,
                    normal: [0.0, 0.0, 1.0],
                    rgb: [0.0; 3],
                    count: 1,
                });
            }
            let hits = diag_hits(&cfg, &bins);
            let lists = block_lists(&cfg, &bins);
            let mut nonzero = 0;
            for (id, list) in &lists {
                let h = hits.get(id).copied().unwrap_or(0);
                assert_eq!(h & !DIAG_BITS, 0);
                let mut cands = vec![DIAG_BITS];
                for _ in 0..4 {
                    cands.push(
                        DIAG_BITS
                            & ((rnd() * (1u32 << 24) as f32) as u32 | ((rnd() * 8.0) as u32) << 24),
                    );
                }
                for (mn, mx) in [
                    ([0, 0, 0], [cfg.block_dim - 1; 3]),
                    ([0, 3, 0], [5, cfg.block_dim - 1, cfg.block_dim - 1]),
                ] {
                    cands.push(halo_mask(mn, mx, cfg.block_dim, Halo::Both) & !FACE_BITS);
                }
                for cand in cands {
                    assert_eq!(
                        SdfMesher::diag_mask(&cfg, *id, &bins, list, cand),
                        h & cand,
                        "{id:?} cand {cand:#x}"
                    );
                }
                nonzero += (h != 0) as usize;
            }
            assert!(nonzero > 20, "only {nonzero} blocks with diagonal hits");
            assert!(hits.keys().all(|id| lists.iter().any(|(b, _)| b == id)));
        }
    }

    /// With simplification enabled, reporting only changed meshes and the narrowed neighbour dirty set must still
    /// match a full rebuild.
    #[test]
    fn incremental_with_simplify_matches_full_rebuild() {
        let base = plane(-4.0, 11.0, -4.0, 11.0, 0.03, 0.05, [1, 2, 3]);
        let bump = plane(5.0, 8.0, 5.0, 8.0, 0.25, 0.05, [9, 9, 9]);
        let probe = vec![pt([5.95, 5.95, 0.03], [0.0, 0.0, 1.0])];
        let steps: Vec<Vec<Batch>> = vec![
            vec![(0, Level::Refined, base)],
            vec![(1, Level::Preview, bump.clone())],
            vec![(2, Level::Refined, probe)],
            vec![(1, Level::Refined, bump)],
        ];
        check_incremental(
            || {
                SdfMesher::new(Config {
                    simplify_error: 0.05,
                    ..Config::default()
                })
            },
            &steps,
        );
    }

    /// Resending an identical preview drops the layer and rebuilds it in the same order, so the meshes are bitwise
    /// equal and no block is reported.
    #[test]
    fn identical_preview_resend_reports_nothing() {
        let base = plane(-4.0, 11.0, -4.0, 4.0, 0.0, 0.05, [9, 9, 9]);
        let pre = plane(3.0, 9.0, -2.0, 2.0, 0.3, 0.1, [50, 50, 50]);
        let steps: Vec<Vec<Batch>> = vec![
            vec![(0, Level::Refined, base), (1, Level::Preview, pre.clone())],
            vec![(1, Level::Preview, pre)],
        ];
        let log = check_incremental(|| SdfMesher::new(Config::default()), &steps);
        let (ids, st) = &log[1];
        assert!(st.reextracted > 0);
        assert!(ids.is_empty(), "{ids:?}");
        assert_eq!(st.updated, 0);
        assert_eq!(st.unchanged_direct + st.unchanged_neighbor, st.reextracted);
    }

    /// Splat path tests. They return early on machines without a suitable GPU.
    #[cfg(feature = "gpu")]
    mod gpu {
        use super::*;
        use std::sync::{Arc, OnceLock};

        fn dev() -> Option<Arc<Gpu>> {
            static G: OnceLock<Option<Arc<Gpu>>> = OnceLock::new();
            G.get_or_init(|| {
                crate::gpu::GpuCtx::shared()
                    .and_then(crate::gpu::GpuSplat::with_ctx)
                    .ok()
                    .map(Arc::new)
            })
            .clone()
        }

        fn mesher(g: Option<Arc<Gpu>>) -> SdfMesher {
            let mut m = SdfMesher::new(Config::default());
            m.set_gpu(g);
            m
        }

        fn scene() -> (Vec<Point>, Vec<Point>, Vec<Point>) {
            let refined = plane(-3.0, 9.0, -3.0, 3.0, 0.0, 0.05, [100, 100, 100]);
            let preview = plane(-3.0, 4.0, -3.0, 3.0, 0.4, 0.1, [10, 10, 10]);
            let other = plane(5.0, 15.0, -2.0, 2.0, 0.1, 0.05, [50, 60, 70]);
            (refined, preview, other)
        }

        /// The splat path must be deterministic and preview replacement must be exact. Versions depend on the
        /// number of extractions, so replacement compares only content (positions, colours, triangles).
        #[test]
        fn gpu_is_deterministic_and_replacement_exact() {
            let Some(g) = dev() else { return };
            let (refined, preview, other) = scene();
            let run = |with_preview: bool| {
                let mut m = mesher(Some(g.clone()));
                m.ingest(1, Level::Refined, &other).unwrap();
                if with_preview {
                    m.ingest(0, Level::Preview, &preview).unwrap();
                    m.extract();
                }
                m.ingest(0, Level::Refined, &refined).unwrap();
                m.extract();
                all_meshes(&m)
            };
            let body = |v: Vec<BlockMesh>| {
                v.into_iter()
                    .map(|m| (m.id, m.positions, m.colors, m.indices))
                    .collect::<Vec<_>>()
            };
            let a = run(true);
            assert!(a.iter().map(|m| m.tri_count()).sum::<usize>() > 1000);
            assert_eq!(a, run(true), "GPU result differs for identical input");
            assert_eq!(
                body(a),
                body(run(false)),
                "preview then refined differs from refined only"
            );
        }

        /// The splat path must stay close to the CPU: triangle counts within 1 %, and fewer than 1 % of GPU
        /// vertices farther than 1 cm from the nearest CPU vertex.
        #[test]
        fn gpu_matches_cpu_closely() {
            let Some(g) = dev() else { return };
            let (refined, _, other) = scene();
            let run = |gpu: Option<Arc<Gpu>>| {
                let mut m = mesher(gpu);
                m.ingest(0, Level::Refined, &refined).unwrap();
                m.ingest(1, Level::Refined, &other).unwrap();
                m.extract();
                m
            };
            let (c, gm) = (run(None), run(Some(g)));
            let (tc, tg) = (c.total_tris() as f64, gm.total_tris() as f64);
            assert!((tc - tg).abs() <= tc * 0.01, "triangle count {tc} vs {tg}");
            let cpu: Vec<[f32; 3]> = c
                .meshes()
                .flat_map(|m| {
                    m.positions
                        .chunks(3)
                        .map(|p| [p[0], p[1], p[2]])
                        .collect::<Vec<_>>()
                })
                .collect();
            let cell = |p: &[f32; 3]| {
                [
                    (p[0] / 0.05).floor() as i32,
                    (p[1] / 0.05).floor() as i32,
                    (p[2] / 0.05).floor() as i32,
                ]
            };
            let mut grid: FxHashMap<[i32; 3], Vec<[f32; 3]>> = FxHashMap::default();
            for p in &cpu {
                grid.entry(cell(p)).or_default().push(*p);
            }
            let mut far = 0usize;
            let mut n = 0usize;
            for m in gm.meshes() {
                for p in m.positions.chunks(3) {
                    let p = [p[0], p[1], p[2]];
                    let k = cell(&p);
                    let mut best = f32::MAX;
                    for dz in -1..=1 {
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                for q in grid
                                    .get(&[k[0] + dx, k[1] + dy, k[2] + dz])
                                    .into_iter()
                                    .flatten()
                                {
                                    best = best
                                        .min((0..3).map(|a| (q[a] - p[a]).powi(2)).sum::<f32>());
                                }
                            }
                        }
                    }
                    n += 1;
                    if best > 0.01 * 0.01 {
                        far += 1;
                    }
                }
            }
            assert!(
                (far as f64) < n as f64 * 0.01,
                "{far}/{n} GPU vertices are more than 1 cm from any CPU vertex"
            );
        }

        #[test]
        fn gpu_seams_stay_closed() {
            let Some(g) = dev() else { return };
            let mut m = mesher(Some(g));
            m.ingest(
                0,
                Level::Refined,
                &plane(2.0, 11.0, -4.0, 9.0, 0.03, 0.05, [1, 2, 3]),
            )
            .unwrap();
            m.extract();
            let mut key: FxHashMap<[u32; 3], u32> = FxHashMap::default();
            let mut verts: Vec<[f32; 3]> = Vec::new();
            let mut edges: FxHashMap<(u32, u32), i32> = FxHashMap::default();
            for b in m.meshes() {
                let map: Vec<u32> = b
                    .positions
                    .chunks(3)
                    .map(|p| {
                        *key.entry([p[0].to_bits(), p[1].to_bits(), p[2].to_bits()])
                            .or_insert_with(|| {
                                verts.push([p[0], p[1], p[2]]);
                                verts.len() as u32 - 1
                            })
                    })
                    .collect();
                for t in b.indices.chunks(3) {
                    for (a, c) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                        let (a, c) = (map[a as usize], map[c as usize]);
                        *edges.entry((a.min(c), a.max(c))).or_default() += 1;
                    }
                }
            }
            for ((a, c), n) in edges {
                if n == 1 {
                    for p in [verts[a as usize], verts[c as usize]] {
                        assert!(
                            p[0] < 2.6 || p[0] > 10.4 || p[1] < -3.4 || p[1] > 8.4,
                            "open edge inside the surface: {p:?}"
                        );
                    }
                }
            }
        }
    }
}
