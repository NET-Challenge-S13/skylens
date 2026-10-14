//! GPU-resident mesher that keeps the signed-distance field on the GPU.
//!
//! Each ingest runs binning and normal estimation on the GPU, orients normals on the CPU, then
//! accumulates the bins into the resident field. Extraction gathers the dirty blocks into padded
//! volumes on the GPU and runs Surface Nets there. The mesher follows the same [`Mesher`] contract
//! as the CPU mesher ([`crate::sdf::SdfMesher`]):
//!
//! - The field lives only on the GPU. The CPU downloads the bins (for normal orientation and for
//!   building accumulation lists), the changed range of each block, and the meshes of changed
//!   blocks.
//! - Preview segments are kept in per-segment layers on the GPU. When the refined segment arrives,
//!   its preview layer is dropped, so replacement is bitwise exact, as on the CPU.
//! - Dirty marking uses the CPU rules: face neighbours from the changed range, diagonal neighbours
//!   from `diag_mask`. Dropping a layer marks from the range only, which is the conservative choice.
//! - Simplification (meshoptimizer), block versions and change detection run on the CPU, on the
//!   downloaded blocks only.

use crate::bins::{Bin, BinStats};
use crate::bins::{NormalPolicy, bin_points};
use crate::gpu::GpuCtx;
use crate::gpu_bins::GpuBinner;
use crate::gpu_extract::{GpuExtractor, Pending};
use crate::gpu_field::{GpuField, LayerKey};
use crate::orient::{Lists, Orient, orient_normals_par};
use crate::sdf::diag_hits;
use crate::{
    BlockId, BlockMesh, Config, Dirty, ExtractStats, FACE_BITS, Halo, IngestError, IngestStats,
    Level, Mesher, Point, SegmentId, apply_built, drain_dirty, halo_mask, mark_mask, same_mesh,
};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;
use std::time::Instant;

/// Default maximum number of blocks gathered and extracted in one batch.
///
/// The gathered volume buffer holds `blocks * (dim + 2)^3 * 4` bytes, about 157 KB per block for
/// `block_dim = 32`.
const EXTRACT_BATCH: usize = 512;
/// Minimum number of blocks per batch when a small extraction is split for overlap.
const MIN_SPLIT: usize = 64;
/// Maximum number of batches a small extraction is split into.
const MAX_SPLIT: usize = 4;

/// One rebuilt block as consumed by [`apply_built`]: id, directly changed, unchanged, and the
/// simplified mesh (positions, colors, indices).
type Built = (BlockId, bool, bool, (Vec<f32>, Vec<u8>, Vec<u32>));

/// A submitted extraction batch and the `(id, directly changed)` entries it covers.
type InFlight<'a> = (Pending, &'a [(BlockId, bool)]);

/// Mesher that keeps the distance field resident on the GPU.
///
/// Requires the `gpu` feature. Results match the CPU mesher closely (same triangle count within
/// tolerance, same block set) and are bitwise deterministic on a given device.
pub struct GpuMesher {
    cfg: Config,
    binner: GpuBinner,
    field: GpuField,
    extractor: GpuExtractor,
    refined: FxHashSet<SegmentId>,
    /// Two gathered-volume buffers as `(capacity in blocks, buffer)`.
    ///
    /// Consecutive batches alternate between them so that a batch still being read back is not
    /// overwritten by the gather of the next batch.
    vols: [Option<(usize, wgpu::Buffer)>; 2],
    dirty: Dirty,
    last_extract: ExtractStats,
    meshes: FxHashMap<BlockId, BlockMesh>,
    versions: FxHashMap<BlockId, u32>,
}

/// Milliseconds elapsed since `t`.
#[inline]
fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

impl GpuMesher {
    /// Creates a GPU-resident mesher on the given device.
    ///
    /// The extraction buffers (two gathered volumes and two sets of extractor scratch buffers) are
    /// allocated and zero-filled here, so the first extraction does not pay for first-touch
    /// allocation. If that preallocation fails it is retried lazily during extraction.
    ///
    /// # Errors
    ///
    /// Returns an error when the binner, the resident field or the extractor cannot be created
    /// on this device (missing limits or shader compilation failure).
    pub fn new(ctx: Arc<GpuCtx>, cfg: Config) -> Result<Self, String> {
        let mut m = Self {
            binner: GpuBinner::new(ctx.clone())?,
            field: GpuField::new(ctx.clone(), &cfg)?,
            extractor: GpuExtractor::new(ctx.clone())?,
            cfg,
            refined: FxHashSet::default(),
            vols: [None, None],
            dirty: Dirty::default(),
            last_extract: ExtractStats::default(),
            meshes: FxHashMap::default(),
            versions: FxHashMap::default(),
        };
        let maxb = m.batch_max();
        let _ = m.extractor.prewarm(&cfg, maxb, 2);
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vol prewarm"),
            });
        for slot in &mut m.vols {
            if let Ok(v) = m.field.volume_buffer(maxb) {
                enc.clear_buffer(&v, 0, None);
                *slot = Some((maxb, v));
            }
        }
        ctx.queue.submit(Some(enc.finish()));
        Ok(m)
    }

    /// Returns the maximum number of blocks per extraction batch.
    ///
    /// Defaults to [`EXTRACT_BATCH`] and can be overridden with the `DELTAMESH_EXTRACT_BATCH`
    /// environment variable. The result is clamped to the extractor limit and is at least 1.
    fn batch_max(&self) -> usize {
        std::env::var("DELTAMESH_EXTRACT_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(EXTRACT_BATCH)
            .min(self.extractor.max_blocks(&self.cfg))
            .max(1)
    }

    /// Accumulates `bins` into `layer` and marks dirty blocks with the CPU dirty rules.
    ///
    /// No per-block bin lists are built here: [`GpuField`] builds the accumulation lists itself,
    /// and the diagonal-neighbour hits (`diag_hits`, the per-block `diag_mask`) are computed on
    /// the CPU while the GPU accumulates. A block marks its diagonal neighbours only when its
    /// changed range touches them and a bin actually hit that neighbour.
    ///
    /// Timing is recorded in the phase counters: `lists` is CPU list building and upload,
    /// `splat` is the remaining accumulation time (GPU wait, including the diagonal hits), and
    /// `dirty` is mask marking.
    ///
    /// # Errors
    ///
    /// Returns the error from [`GpuField::integrate_bins`] (buffer or pool limits).
    fn integrate(&mut self, layer: LayerKey, bins: &[crate::bins::Bin]) -> Result<(), String> {
        let t = Instant::now();
        let cfg = self.cfg;
        let mut hits = FxHashMap::default();
        let ranges = self
            .field
            .integrate_bins(layer, bins, || hits = diag_hits(&cfg, bins))?;
        let all = ms(t);
        let ft = self.field.last_times();
        crate::sdf::phase_add(|p| {
            p.lists += ft.bricks + ft.upload;
            p.splat += all - ft.bricks - ft.upload;
        });
        let t = Instant::now();
        let dim = self.cfg.block_dim;
        for (id, tmin, tmax) in ranges {
            if tmin[0] > tmax[0] {
                continue;
            }
            let m = halo_mask(tmin, tmax, dim, Halo::Both);
            let mask = if m & !FACE_BITS == 0 {
                m
            } else {
                (m & FACE_BITS) | (hits.get(&id).copied().unwrap_or(0) & m & !FACE_BITS)
            };
            mark_mask(&mut self.dirty, id, mask);
        }
        crate::sdf::phase_add(|p| p.dirty += ms(t));
        Ok(())
    }
}

impl GpuMesher {
    /// Reads back a submitted batch, simplifies its blocks and appends them to `built`.
    ///
    /// Blocks are simplified in parallel and compared bitwise with the stored mesh to decide
    /// whether the block version must change.
    ///
    /// # Panics
    ///
    /// Panics if the GPU readback fails.
    fn finish_batch(&self, (pend, chunk): InFlight<'_>, built: &mut Vec<Built>) {
        let cfg = self.cfg;
        let t = Instant::now();
        let out = self
            .extractor
            .finish(pend)
            .unwrap_or_else(|e| panic!("GPU surface extraction failed: {e}"));
        crate::sdf::phase_add(|p| p.extract_gpu += ms(t));
        let t = Instant::now();
        let meshes = &self.meshes;
        let part: Vec<_> = out
            .into_par_iter()
            .zip(chunk.par_iter())
            .map(|(b, &(id, direct))| {
                debug_assert_eq!(b.id, id);
                let m = crate::simplify::simplify_block(
                    cfg.simplify_error,
                    b.positions,
                    b.colors,
                    b.indices,
                    &b.seam,
                );
                let same = same_mesh(meshes.get(&id), &m.0, &m.1, &m.2);
                (id, direct, same, m)
            })
            .collect();
        crate::sdf::phase_add(|p| p.simplify += ms(t));
        built.extend(part);
    }
}

impl Mesher for GpuMesher {
    /// Bins, orients and accumulates one segment into the resident field.
    ///
    /// Preview segments also request neighbour lists for normal orientation from the GPU binner
    /// (disabled with `DELTAMESH_NO_NBLISTS`); when the GPU cannot build them the CPU searches
    /// neighbours itself. Normal signs are decided on the CPU. When the oracle is enabled and the
    /// base layer is not empty, the oracle values are read from the resident field in one batch
    /// while the CPU builds the neighbour graph. The binner calls the orientation closure with
    /// the GPU results still mapped, so the neighbour lists are not copied.
    ///
    /// If GPU binning fails, this input is binned on the CPU instead. Any preview layer of the
    /// same segment is dropped before accumulation.
    ///
    /// # Panics
    ///
    /// Panics if GPU accumulation fails.
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
        crate::sdf::phase_add(|p| p.calls += 1);
        let policy = match level {
            Level::Preview => NormalPolicy::Estimate,
            Level::Refined => NormalPolicy::Trust,
        };
        let t = Instant::now();
        let want_nb = match (level, cfg.orient) {
            (
                Level::Preview,
                Orient::Propagate {
                    radius, seed_nz, ..
                },
            ) if std::env::var_os("DELTAMESH_NO_NBLISTS").is_none() => Some((radius, seed_nz)),
            _ => None,
        };
        let field = &self.field;
        let orient = |bins: &mut [Bin], bs: &BinStats, lists: Option<Lists>| -> f64 {
            let t = Instant::now();
            if bs.fixed.is_empty() {
                return ms(t);
            }
            let fixed = &bs.fixed;
            let use_oracle = cfg.orient_oracle && !field.base_is_empty();
            let oracle = |b: &[Bin]| -> Option<Vec<Option<f32>>> {
                if !use_oracle {
                    return None;
                }
                let idx: Vec<usize> = (0..b.len())
                    .into_par_iter()
                    .filter(|&i| !fixed[i])
                    .collect();
                let p: Vec<[f32; 3]> = idx.iter().map(|&i| b[i].pos).collect();
                let n: Vec<[f32; 3]> = idx.iter().map(|&i| b[i].normal).collect();
                match field.base_agreement(&p, &n) {
                    Ok(a) => {
                        let mut ans = vec![None; b.len()];
                        for (k, &i) in idx.iter().enumerate() {
                            ans[i] = a[k];
                        }
                        Some(ans)
                    }
                    Err(e) => {
                        eprintln!("GPU oracle failed, orienting normals without it: {e}");
                        None
                    }
                }
            };
            orient_normals_par(bins, fixed, cfg.bin, cfg.orient, oracle, lists);
            ms(t)
        };
        let (bins, bs, orient_ms) = match self.binner.bin_points_nb(
            pts,
            cfg.bin,
            policy,
            cfg.normal_radius_bins,
            cfg.normal_min_neighbors,
            want_nb,
            orient,
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("GPU binning failed, binning this input on the CPU: {e}");
                let (mut b, s) = bin_points(
                    pts,
                    cfg.bin,
                    policy,
                    cfg.normal_radius_bins,
                    cfg.normal_min_neighbors,
                );
                let o = orient(&mut b, &s, None);
                (b, s, o)
            }
        };
        let all = ms(t);
        crate::sdf::phase_add(|p| {
            p.binning += all - orient_ms;
            p.orient += orient_ms;
        });
        let t = Instant::now();
        let dropped = self.field.drop_layer(seg);
        let removed = !dropped.is_empty();
        for (id, mn, mx) in dropped {
            if mn[0] <= mx[0] {
                mark_mask(
                    &mut self.dirty,
                    id,
                    halo_mask(mn, mx, cfg.block_dim, Halo::Both),
                );
            }
        }
        crate::sdf::phase_add(|p| p.dirty += ms(t));
        let layer = match level {
            Level::Preview => LayerKey::Pending(seg),
            Level::Refined => LayerKey::Base,
        };
        if let Err(e) = self.integrate(layer, &bins) {
            panic!("GPU accumulation failed: {e}");
        }
        if level == Level::Refined {
            self.refined.insert(seg);
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

    /// Re-extracts all dirty blocks and returns the ids whose mesh changed.
    ///
    /// Blocks are processed in batches of at most `batch_max()`. As long as each batch keeps at
    /// least `MIN_SPLIT` blocks, the work is split into up to `MAX_SPLIT` batches so that CPU
    /// work (gather setup, unpacking, simplification) overlaps GPU work: batch `k` is gathered
    /// and submitted before batch `k - 1` is read back. The two volume buffers alternate so that
    /// a rerun of batch `k - 1`'s write pass (when its output buffer was too small) does not read
    /// a volume already overwritten by the gather of batch `k`.
    ///
    /// After extraction the field pool reserves its next page if free space is low; the page is
    /// zero-filled while the GPU is idle before the next input.
    ///
    /// # Panics
    ///
    /// Panics if allocating the volume buffer, gathering, or GPU extraction fails.
    fn extract(&mut self) -> Vec<BlockId> {
        let ids = drain_dirty(&mut self.dirty);
        let cfg = self.cfg;
        let maxb = self.batch_max();
        let nbat = ids
            .len()
            .div_ceil(maxb)
            .max((ids.len() / MIN_SPLIT).min(MAX_SPLIT))
            .max(1);
        let batch = ids.len().div_ceil(nbat).max(1);
        let mut built = Vec::with_capacity(ids.len());
        let mut prev: Option<InFlight<'_>> = None;
        for (k, chunk) in ids
            .chunks(batch)
            .map(Some)
            .chain(std::iter::once(None))
            .enumerate()
        {
            if let Some(chunk) = chunk {
                let bid: Vec<BlockId> = chunk.iter().map(|(id, _)| *id).collect();
                let t = Instant::now();
                let slot = &mut self.vols[k % 2];
                if slot.as_ref().is_none_or(|(n, _)| *n < batch) {
                    let v = self
                        .field
                        .volume_buffer(maxb.max(batch))
                        .unwrap_or_else(|e| panic!("failed to allocate GPU volume buffer: {e}"));
                    *slot = Some((maxb.max(batch), v));
                }
                let vol = &slot.as_ref().unwrap().1;
                let vol = self
                    .field
                    .gather_into(&bid, vol)
                    .unwrap_or_else(|e| panic!("GPU field gather failed: {e}"));
                crate::sdf::phase_add(|p| p.gather += ms(t));
                let t = Instant::now();
                let pend = self
                    .extractor
                    .submit(&vol, &bid, &cfg)
                    .unwrap_or_else(|e| panic!("GPU surface extraction failed: {e}"));
                crate::sdf::phase_add(|p| p.extract_gpu += ms(t));
                if let Some(old) = prev.replace((pend, chunk)) {
                    self.finish_batch(old, &mut built);
                }
            } else if let Some(old) = prev.take() {
                self.finish_batch(old, &mut built);
            }
        }
        let out = apply_built(
            built,
            &mut self.meshes,
            &mut self.versions,
            &mut self.last_extract,
        );
        if let Err(e) = self.field.prefetch() {
            eprintln!("failed to prefetch a pool page (will retry on demand): {e}");
        }
        out
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
        self.field.field_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf::SdfMesher;
    use crate::testutil::plane;

    /// Block id and mesh buffers (positions, colors, indices) of one block.
    type BodyMesh = (BlockId, Vec<f32>, Vec<u8>, Vec<u32>);

    fn ctx() -> Option<Arc<GpuCtx>> {
        GpuCtx::shared().ok()
    }

    /// Returns all block meshes sorted by block id.
    fn body(m: &dyn Mesher) -> Vec<BodyMesh> {
        let mut v: Vec<_> = m
            .meshes()
            .map(|b| {
                (
                    b.id,
                    b.positions.clone(),
                    b.colors.clone(),
                    b.indices.clone(),
                )
            })
            .collect();
        v.sort_by_key(|x| x.0);
        v
    }

    /// Builds a refined plane, a preview plane without normals over part of it, and a second
    /// refined plane in another segment.
    fn scene() -> (Vec<Point>, Vec<Point>, Vec<Point>) {
        let refined = plane(-3.0, 9.0, -3.0, 3.0, 0.0, 0.05, [100, 100, 100]);
        let mut preview = plane(-3.0, 4.0, -3.0, 3.0, 0.4, 0.1, [10, 10, 10]);
        for p in &mut preview {
            p.normal = [f32::NAN; 3];
        }
        let other = plane(5.0, 15.0, -2.0, 2.0, 0.1, 0.05, [50, 60, 70]);
        (refined, preview, other)
    }

    #[test]
    fn resident_is_deterministic_and_replacement_exact() {
        let Some(c) = ctx() else { return };
        let (refined, preview, other) = scene();
        let run = |with_preview: bool| {
            let mut m = GpuMesher::new(c.clone(), Config::default()).unwrap();
            m.ingest(1, Level::Refined, &other).unwrap();
            if with_preview {
                m.ingest(0, Level::Preview, &preview).unwrap();
                m.extract();
            }
            m.ingest(0, Level::Refined, &refined).unwrap();
            m.extract();
            body(&m)
        };
        let a = run(true);
        assert!(a.iter().map(|x| x.3.len() / 3).sum::<usize>() > 1000);
        assert_eq!(a, run(true), "same input produced different output");
        assert_eq!(
            a,
            run(false),
            "preview then refined differs from refined only"
        );
    }

    /// Several overlapping batches give the same result when an undersized output buffer forces
    /// a rerun of the write pass while the next batch is already submitted.
    ///
    /// The surface is a wave with a colour gradient so that every block has a distinct volume;
    /// with identical volumes a mix-up between blocks would go unnoticed.
    #[test]
    fn resident_batches_with_rerun_match() {
        let Some(c) = ctx() else { return };
        let cfg = Config {
            voxel: 0.05,
            bin: 0.025,
            splat_radius: 0.1,
            ..Config::default()
        };
        let mut pts = plane(-10.0, 14.0, -8.0, 8.0, 0.0, 0.04, [0, 0, 0]);
        for p in &mut pts {
            let (x, y) = (p.pos[0], p.pos[1]);
            p.pos[2] = 0.3 * (x * 0.9).sin() * (y * 0.7).cos();
            let g = [
                -0.27 * (x * 0.9).cos() * (y * 0.7).cos(),
                0.21 * (x * 0.9).sin() * (y * 0.7).sin(),
                1.0,
            ];
            let l = (g[0] * g[0] + g[1] * g[1] + 1.0f32).sqrt();
            p.normal = [g[0] / l, g[1] / l, 1.0 / l];
            p.rgb = [((x + 10.0) * 10.0) as u8, ((y + 8.0) * 15.0) as u8, 77];
        }
        let run = |hint: bool| {
            let mut m = GpuMesher::new(c.clone(), cfg).unwrap();
            if hint {
                m.extractor.set_out_hint((0.0, 0.0), 1);
            }
            m.ingest(0, Level::Refined, &pts).unwrap();
            let n = m.dirty.len();
            m.extract();
            (n, body(&m))
        };
        let (n, a) = run(false);
        assert!(n >= 2 * MIN_SPLIT, "only one batch ({n} blocks)");
        assert!(a.iter().map(|x| x.3.len() / 3).sum::<usize>() > 10000);
        assert_eq!(a, run(true).1, "batch with a resized output buffer differs");
    }

    #[test]
    fn resident_matches_cpu_closely_and_reports_like_cpu() {
        let Some(c) = ctx() else { return };
        let (refined, preview, other) = scene();
        let mut g = GpuMesher::new(c, Config::default()).unwrap();
        let mut s = SdfMesher::new(Config::default());
        for m in [&mut g as &mut dyn Mesher, &mut s as &mut dyn Mesher] {
            m.ingest(1, Level::Refined, &other).unwrap();
            m.ingest(0, Level::Preview, &preview).unwrap();
            m.extract();
            m.ingest(0, Level::Refined, &refined).unwrap();
            m.extract();
        }
        let (tg, ts) = (g.total_tris() as f64, s.total_tris() as f64);
        assert!((tg - ts).abs() <= ts * 0.01, "triangles {tg} vs {ts}");
        assert_eq!(g.block_count(), s.block_count());
    }

    #[test]
    fn resident_seams_stay_closed() {
        let Some(c) = ctx() else { return };
        let mut m = GpuMesher::new(c, Config::default()).unwrap();
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
        assert!(m.block_count() >= 4);
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
