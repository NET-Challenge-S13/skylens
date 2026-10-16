//! 2.5D height-field mesher, a simple baseline for comparison with the SDF mesher.
//!
//! Binned points are averaged per xy cell into a single height, so walls and overhangs cannot be
//! represented. Cells are grouped into square tiles of `block_dim × block_dim`; each tile is one
//! output block with `z = 0` in its [`BlockId`].
//!
//! A quad (two triangles) is emitted only when all four corner cells have at least
//! `min_weight` samples and the height range across the quad is at most `2 * splat_radius`.
//! Larger steps are skipped because draping a wall like a curtain would create surface far from
//! any real point.
//!
//! Preview replacement uses the same layer scheme as the SDF mesher: each preview segment
//! accumulates into its own layer, and a refined segment drops that layer before adding to the
//! base layer, so the result is bitwise identical to never having seen the preview.

use crate::bins::{Bin, NormalPolicy, bin_points};
use crate::{
    BlockId, BlockMesh, Config, Dirty, ExtractStats, Halo, IngestError, IngestStats, Level, Mesher,
    Point, SegmentId, apply_built, drain_dirty, mark_dirty, same_mesh,
};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BTreeMap;

/// Per-cell accumulator: summed height, sample count and summed colour.
#[derive(Clone, Copy, Default, Debug)]
struct HAcc {
    wz: f32,
    w: f32,
    wc: [f32; 3],
}

/// One tile of cells plus the bounding box (in tile-local cell coordinates) of every cell ever
/// written, used to mark the right blocks dirty when the layer is dropped.
#[derive(Clone)]
struct Tile {
    cells: Vec<HAcc>,
    min: [i32; 2],
    max: [i32; 2],
}

/// Sparse set of tiles keyed by tile coordinate.
type Layer = FxHashMap<[i32; 2], Tile>;

/// Incremental 2.5D height-field mesher.
///
/// Refined segments accumulate into `base`; each preview segment has its own layer in `pending`
/// until its refined version arrives.
pub struct HeightMesher {
    cfg: Config,
    base: Layer,
    pending: BTreeMap<SegmentId, Layer>,
    refined: FxHashSet<SegmentId>,
    dirty: Dirty,
    last_extract: ExtractStats,
    meshes: FxHashMap<BlockId, BlockMesh>,
    versions: FxHashMap<BlockId, u32>,
}

impl HeightMesher {
    /// Creates an empty mesher with the given configuration.
    pub fn new(cfg: Config) -> Self {
        Self {
            cfg,
            base: Layer::default(),
            pending: BTreeMap::new(),
            refined: FxHashSet::default(),
            dirty: Dirty::default(),
            last_extract: ExtractStats::default(),
            meshes: FxHashMap::default(),
            versions: FxHashMap::default(),
        }
    }

    /// Adds binned points to `layer` and marks the affected blocks dirty.
    ///
    /// Only the tile-local range that was actually written is marked. [`Halo::Upper`] applies
    /// because a tile also reads row and column 0 of its `+x`/`+y` neighbours, so a change at
    /// local index 0 dirties the tile on the `-x`/`-y` side as well.
    fn integrate(cfg: &Config, layer: &mut Layer, bins: &[Bin], dirty: &mut Dirty) {
        let dim = cfg.block_dim;
        let inv = 1.0 / cfg.voxel;
        let mut touched: FxHashMap<[i32; 2], ([i32; 2], [i32; 2])> = FxHashMap::default();
        for b in bins {
            let g = [
                (b.pos[0] * inv).floor() as i32,
                (b.pos[1] * inv).floor() as i32,
            ];
            let t = [g[0].div_euclid(dim), g[1].div_euclid(dim)];
            let l = [g[0] - t[0] * dim, g[1] - t[1] * dim];
            let tile = layer.entry(t).or_insert_with(|| Tile {
                cells: vec![HAcc::default(); (dim * dim) as usize],
                min: [i32::MAX; 2],
                max: [i32::MIN; 2],
            });
            let a = &mut tile.cells[(l[1] * dim + l[0]) as usize];
            a.wz += b.pos[2];
            a.w += 1.0;
            for k in 0..3 {
                a.wc[k] += b.rgb[k];
            }
            let e = touched.entry(t).or_insert(([i32::MAX; 2], [i32::MIN; 2]));
            for (k, &lk) in l.iter().enumerate() {
                e.0[k] = e.0[k].min(lk);
                e.1[k] = e.1[k].max(lk);
                tile.min[k] = tile.min[k].min(lk);
                tile.max[k] = tile.max[k].max(lk);
            }
        }
        for (t, (mn, mx)) in touched {
            mark_dirty(dirty, [t[0], t[1], 0], mn, mx, dim, Halo::Upper);
        }
    }

    /// Builds the mesh of tile `id` from the base layer and all pending layers.
    ///
    /// The tile reads a `(dim + 1)²` cell window: its own cells plus row/column 0 of the tiles
    /// at `+x`, `+y` and `+x+y`, so neighbouring tiles share boundary vertices. Vertices sit at
    /// cell centres with the mean height and rounded mean colour.
    ///
    /// # Returns
    ///
    /// `(positions, colors, indices)` in the layout of [`BlockMesh`].
    fn build(&self, id: BlockId) -> (Vec<f32>, Vec<u8>, Vec<u32>) {
        let dim = self.cfg.block_dim;
        let p = (dim + 1) as usize;
        let mut acc = vec![HAcc::default(); p * p];
        for layer in std::iter::once(&self.base).chain(self.pending.values()) {
            for (oy, ox) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let Some(t) = layer.get(&[id[0] + ox, id[1] + oy]) else {
                    continue;
                };
                let (x1, y1) = (if ox == 0 { dim } else { 1 }, if oy == 0 { dim } else { 1 });
                for ly in 0..y1 {
                    for lx in 0..x1 {
                        let a = &t.cells[(ly * dim + lx) as usize];
                        if a.w == 0.0 {
                            continue;
                        }
                        let o = &mut acc[(ly + oy * dim) as usize * p + (lx + ox * dim) as usize];
                        o.wz += a.wz;
                        o.w += a.w;
                        for k in 0..3 {
                            o.wc[k] += a.wc[k];
                        }
                    }
                }
            }
        }
        let minw = self.cfg.min_weight;
        let max_step = 2.0 * self.cfg.splat_radius;
        let v = self.cfg.voxel;
        let mut vid = vec![u32::MAX; p * p];
        let (mut pos, mut col, mut tri) = (Vec::new(), Vec::new(), Vec::new());
        let mut vert = |x: usize, y: usize, pos: &mut Vec<f32>, col: &mut Vec<u8>| -> u32 {
            let i = y * p + x;
            if vid[i] == u32::MAX {
                let a = &acc[i];
                vid[i] = (pos.len() / 3) as u32;
                pos.extend_from_slice(&[
                    ((id[0] * dim + x as i32) as f32 + 0.5) * v,
                    ((id[1] * dim + y as i32) as f32 + 0.5) * v,
                    a.wz / a.w,
                ]);
                for k in 0..3 {
                    col.push((a.wc[k] / a.w).round().clamp(0.0, 255.0) as u8);
                }
            }
            vid[i]
        };
        for y in 0..dim as usize {
            for x in 0..dim as usize {
                let c = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)];
                if c.iter().any(|&(cx, cy)| acc[cy * p + cx].w < minw) {
                    continue;
                }
                let zs = c.map(|(cx, cy)| acc[cy * p + cx].wz / acc[cy * p + cx].w);
                let (lo, hi) = zs
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(a, b), &z| (a.min(z), b.max(z)));
                if hi - lo > max_step {
                    continue;
                }
                let q = c.map(|(cx, cy)| vert(cx, cy, &mut pos, &mut col));
                tri.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
            }
        }
        (pos, col, tri)
    }

    /// Removes the preview layer of `seg`, if any, and marks the blocks it covered dirty.
    ///
    /// # Returns
    ///
    /// `true` if a preview layer existed.
    fn drop_layer(&mut self, seg: SegmentId) -> bool {
        let Some(layer) = self.pending.remove(&seg) else {
            return false;
        };
        for (t, tile) in &layer {
            mark_dirty(
                &mut self.dirty,
                [t[0], t[1], 0],
                tile.min,
                tile.max,
                self.cfg.block_dim,
                Halo::Upper,
            );
        }
        true
    }
}

impl Mesher for HeightMesher {
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
        let cfg = self.cfg;
        let (bins, _) = bin_points(pts, cfg.bin, NormalPolicy::Ignore, 0, 0);
        let removed = self.drop_layer(seg);
        match level {
            Level::Preview => {
                let mut layer = Layer::default();
                Self::integrate(&cfg, &mut layer, &bins, &mut self.dirty);
                self.pending.insert(seg, layer);
            }
            Level::Refined => {
                Self::integrate(&cfg, &mut self.base, &bins, &mut self.dirty);
                self.refined.insert(seg);
            }
        }
        Ok(IngestStats {
            input_points: pts.len(),
            bins: bins.len(),
            removed_preview: removed,
            dirty_blocks: self.dirty.len(),
            ..Default::default()
        })
    }

    /// Rebuilds every dirty tile and reports the blocks whose mesh changed.
    ///
    /// Each rebuilt mesh is compared bitwise with the stored one; if it is identical the version
    /// is not bumped and the block is not reported.
    fn extract(&mut self) -> Vec<BlockId> {
        let ids = drain_dirty(&mut self.dirty);
        let built: Vec<_> = ids
            .par_iter()
            .map(|&(id, direct)| {
                let m = self.build(id);
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
        let per = (self.cfg.block_dim * self.cfg.block_dim) as usize * std::mem::size_of::<HAcc>();
        self.base.len() * per + self.pending.values().map(|l| l.len() * per).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Replacing a preview with its refined segment gives the same meshes as ingesting only the
    /// refined segment.
    #[test]
    fn replace_is_exact() {
        let mk = |z: f32, x1: f32| {
            let mut v = Vec::new();
            let mut x = 0.0;
            while x < x1 {
                let mut y = 0.0;
                while y < 5.0 {
                    v.push(Point {
                        pos: [x, y, z],
                        rgb: [1, 2, 3],
                        normal: [0.0; 3],
                    });
                    y += 0.05;
                }
                x += 0.05;
            }
            v
        };
        let mut a = HeightMesher::new(Config::default());
        a.ingest(0, Level::Preview, &mk(1.0, 4.0)).unwrap();
        a.extract();
        a.ingest(0, Level::Refined, &mk(0.0, 9.0)).unwrap();
        a.extract();
        let mut b = HeightMesher::new(Config::default());
        b.ingest(0, Level::Refined, &mk(0.0, 9.0)).unwrap();
        b.extract();
        let mut ma: Vec<_> = a.meshes().cloned().collect();
        let mut mb: Vec<_> = b.meshes().cloned().collect();
        ma.sort_by_key(|m| m.id);
        mb.sort_by_key(|m| m.id);
        assert_eq!(ma.len(), mb.len());
        for (x, y) in ma.iter().zip(&mb) {
            assert_eq!((&x.positions, &x.indices), (&y.positions, &y.indices));
        }
    }

    /// A tile reads local cells `0..=dim`, so changing local column 0 must also rebuild the tile
    /// at `-x`, while changing column `dim - 1` affects only its own tile.
    ///
    /// `col0` lands in local x column 0 of tile 1, `col31` in local x column 31 of tile 0, and
    /// `corner` in local cell (0, 0) of tile (1, 1).
    #[test]
    fn boundary_cell_change_matches_full_rebuild() {
        use crate::testutil::{Batch, check_incremental, plane};
        let base = plane(-4.0, 11.0, -4.0, 11.0, 0.0, 0.05, [1, 2, 3]);
        let col0 = plane(6.4, 6.5, -2.0, 4.0, 0.3, 0.05, [9, 9, 9]);
        let col31 = plane(6.2, 6.3, -2.0, 4.0, 0.3, 0.05, [9, 9, 9]);
        let corner = plane(6.4, 6.5, 6.4, 6.5, 0.3, 0.05, [9, 9, 9]);
        let steps: Vec<Vec<Batch>> = vec![
            vec![(0, Level::Refined, base)],
            vec![(1, Level::Refined, col0)],
            vec![(2, Level::Refined, col31)],
            vec![(3, Level::Refined, corner)],
        ];
        let log = check_incremental(|| HeightMesher::new(Config::default()), &steps);
        assert!(
            log[1].0.contains(&[0, 0, 0]) && log[1].0.contains(&[1, 0, 0]),
            "{:?}",
            log[1]
        );
        assert_eq!(
            log[2].1.reextracted, log[2].1.reextracted_direct,
            "{:?}",
            log[2]
        );
        assert!(log[3].0.contains(&[0, 0, 0]), "{:?}", log[3]);
    }
}
