//! Incremental block meshing of streamed point-cloud segments.
//!
//! `deltamesh` turns a stream of point-cloud segments into a triangle mesh that is split into fixed-size blocks, and
//! after each update re-meshes only the blocks whose geometry actually changed. A client can therefore keep a copy of
//! the mesh in sync by applying per-block updates instead of reloading everything. The crate does no file or network
//! I/O.
//!
//! # Pipeline
//!
//! Each ingested segment is first snapped into small bins (`Config::bin`) so that the cost scales with the number of
//! bins rather than the number of raw points. Bins get a normal, either the one carried by the input points or one
//! estimated from neighbouring bins, and estimated normals are given a consistent sign (see [`orient`]). Every bin
//! then splats a Gaussian-weighted signed distance into a sparse voxel field, stored as blocks of `block_dim`³ voxels
//! (32³ by default) made of 8³-voxel bricks; only touched bricks are allocated. Extraction runs Surface Nets on the
//! blocks marked dirty and can optionally simplify each block with meshoptimizer (`Config::simplify_error`).
//!
//! Two meshers implement the same contract:
//!
//! - [`sdf::SdfMesher`]: signed-distance field and Surface Nets, the main implementation.
//! - [`height::HeightMesher`]: a 2.5D height field, useful as a simple baseline.
//!
//! # Workflow
//!
//! Every segment is sent first as a quick [`Level::Preview`] and later as a [`Level::Refined`] version. Preview data
//! lives in its own per-segment layer, so when the refined segment arrives the preview layer is dropped as a whole
//! instead of being subtracted with floating-point arithmetic. The result after replacement is bitwise identical to
//! ingesting only the refined segment.
//!
//! Using the [`Mesher`] trait:
//!
//! 1. [`Mesher::ingest`] a segment. This updates the field and marks affected blocks dirty; it does not mesh.
//! 2. [`Mesher::extract`] re-meshes the dirty blocks and returns the ids of the blocks whose mesh changed.
//! 3. Fetch each returned block with [`Mesher::mesh`] and send it to the client.
//! 4. When a refined segment arrives, ingest it with the same [`SegmentId`]; the preview layer is replaced and the
//!    next `extract` reports the affected blocks.
//!
//! # Block output contract
//!
//! - Each [`BlockMesh`] is identified by its [`BlockId`], an integer block coordinate.
//! - [`BlockMesh::version`] increases only when the re-extracted mesh differs bitwise from the stored one. Blocks that
//!   were re-extracted but came out identical are not reported.
//! - A reported block whose [`Mesher::mesh`] is `None` (an empty mesh) means the client should delete that block.
//!   [`Mesher::version`] still returns its last version.
//! - Vertices on a block boundary are computed from the same voxels by both neighbouring blocks, so shared boundary
//!   vertices have bitwise equal positions, with or without simplification.
//! - Triangles are wound counter-clockwise when viewed from the outside, i.e. from the side the surface normals point
//!   to.
//!
//! # GPU support
//!
//! The optional `gpu` feature adds wgpu-based paths (Metal, Vulkan, DX12):
//!
//! - The GPU-resident path (`gpu_mesher::GpuMesher`) keeps the field on the GPU and runs binning, normal estimation,
//!   accumulation and extraction there. Only normal orientation, simplification and change detection run on the CPU.
//!   It is deterministic on a given machine but not bitwise identical to the CPU path.
//! - The splat path (`gpu::GpuSplat`, enabled with `SdfMesher::set_gpu`) accumulates on the GPU and merges the result
//!   into the CPU field; extraction stays on the CPU.
//!
//! # Example
//!
//! ```
//! use deltamesh::sdf::SdfMesher;
//! use deltamesh::{Config, Level, Mesher, Point};
//!
//! // A flat 5 m x 5 m patch at z = 0, sampled every 5 cm, with normals pointing up.
//! let mut points = Vec::new();
//! for i in 0..100 {
//!     for j in 0..100 {
//!         points.push(Point {
//!             pos: [i as f32 * 0.05, j as f32 * 0.05, 0.0],
//!             rgb: [180, 180, 180],
//!             normal: [0.0, 0.0, 1.0],
//!         });
//!     }
//! }
//!
//! let mut mesher = SdfMesher::new(Config::default());
//!
//! // The preview ignores the input normals and estimates them.
//! mesher.ingest(7, Level::Preview, &points)?;
//! let changed = mesher.extract();
//! assert!(!changed.is_empty());
//! for id in &changed {
//!     match mesher.mesh(id) {
//!         Some(block) => assert!(block.tri_count() > 0),
//!         None => { /* the client deletes block `id` */ }
//!     }
//! }
//!
//! // The refined segment replaces the preview layer of segment 7.
//! let stats = mesher.ingest(7, Level::Refined, &points)?;
//! assert!(stats.removed_preview);
//! mesher.extract();
//! assert!(mesher.total_tris() > 0);
//!
//! // A segment can be refined only once.
//! assert!(mesher.ingest(7, Level::Refined, &points).is_err());
//! # Ok::<(), deltamesh::IngestError>(())
//! ```

pub mod bins;
mod eig;
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub mod gpu_bins;
#[cfg(feature = "gpu")]
pub mod gpu_extract;
#[cfg(feature = "gpu")]
pub mod gpu_field;
#[cfg(feature = "gpu")]
pub mod gpu_mesher;
pub mod height;
pub mod orient;
pub mod sdf;
mod simplify;

/// One input point.
///
/// Coordinates are world coordinates in metres (for example a local ENU frame). The normal is used only for
/// [`Level::Refined`] input; preview normals are estimated from the points.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub pos: [f32; 3],
    pub rgb: [u8; 3],
    pub normal: [f32; 3],
}

/// Quality level of a segment.
///
/// Each segment arrives first as a preview and later as a refined version that replaces it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Level {
    /// Fast preview. Input normals are ignored and estimated instead. The whole contribution is removed when the
    /// refined version of the same segment arrives.
    Preview,
    /// Final data with trusted normals. Accepted once per segment; a refined segment is never replaced.
    Refined,
}

/// Identifier of an input segment.
pub type SegmentId = u32;
/// Integer block coordinate. The height-field mesher uses only `z = 0`.
pub type BlockId = [i32; 3];

/// Mesh of one block as sent to a client.
///
/// A block with no triangles means "delete this block".
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BlockMesh {
    pub id: BlockId,
    pub version: u32,
    /// Vertex positions as repeated `x y z`, in absolute world coordinates.
    pub positions: Vec<f32>,
    /// Vertex colours as repeated `r g b`.
    pub colors: Vec<u8>,
    pub indices: Vec<u32>,
}

impl BlockMesh {
    pub fn vertex_count(&self) -> usize {
        self.positions.len() / 3
    }
    pub fn tri_count(&self) -> usize {
        self.indices.len() / 3
    }
    /// Heap memory held by the vertex and index buffers, in bytes.
    pub fn heap_bytes(&self) -> usize {
        self.positions.capacity() * 4 + self.colors.capacity() + self.indices.capacity() * 4
    }
}

/// Mesher parameters.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Voxel size of the distance field (cell size of the height field), in metres.
    pub voxel: f32,
    /// Voxels per block edge. Must be a multiple of 8.
    pub block_dim: i32,
    /// Bin size used to merge input points before accumulation, in metres. Accumulation cost is proportional to the
    /// number of bins, not the number of input points.
    pub bin: f32,
    /// Radius within which a bin writes distance values, in metres. No surface is produced beyond this radius.
    pub splat_radius: f32,
    /// Minimum accumulated weight for a voxel to count as observed.
    pub min_weight: f32,
    /// Neighbourhood radius for normal estimation, in bins.
    pub normal_radius_bins: i32,
    /// Minimum neighbour count for normal estimation. Bins with fewer neighbours are dropped.
    pub normal_min_neighbors: usize,
    /// How the sign of estimated normals is chosen.
    pub orient: orient::Orient,
    /// Orient preview normals by the gradient of the existing refined field first, where the two overlap.
    pub orient_oracle: bool,
    /// Absolute simplification error per block mesh, in metres (meshoptimizer). Zero disables simplification. Used
    /// only by the distance-field meshers.
    pub simplify_error: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            voxel: 0.2,
            block_dim: 32,
            bin: 0.1,
            splat_radius: 0.4,
            min_weight: 0.5,
            normal_radius_bins: 2,
            normal_min_neighbors: 6,
            orient: orient::Orient::default(),
            orient_oracle: true,
            simplify_error: 0.0,
        }
    }
}

/// Error returned by [`Mesher::ingest`] for a segment that was already refined.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestError {
    /// The refined version of this segment was already ingested.
    AlreadyRefined(SegmentId),
    /// A preview arrived for a segment that was already refined.
    PreviewAfterRefined(SegmentId),
}

impl std::fmt::Display for IngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyRefined(s) => write!(f, "segment {s} was already refined"),
            Self::PreviewAfterRefined(s) => {
                write!(
                    f,
                    "preview received for segment {s}, which was already refined"
                )
            }
        }
    }
}
impl std::error::Error for IngestError {}

/// Counters reported by [`Mesher::ingest`].
#[derive(Clone, Debug, Default)]
pub struct IngestStats {
    pub input_points: usize,
    /// Number of bins accumulated after merging input points.
    pub bins: usize,
    /// Number of bins whose normal was estimated.
    pub normals_estimated: usize,
    /// Number of bins dropped because no normal could be estimated.
    pub bins_dropped: usize,
    /// Whether a preview layer of the same segment was dropped.
    pub removed_preview: bool,
    /// Number of blocks waiting to be re-extracted, accumulated until the next `extract`.
    pub dirty_blocks: usize,
}

/// Incremental block mesher.
///
/// See the [crate documentation](crate) for the workflow and the block output contract.
pub trait Mesher {
    /// Adds one segment to the field and marks the affected blocks dirty.
    ///
    /// Any preview layer of the same segment is dropped first, whether `pts` is a resent preview or the refined
    /// version.
    ///
    /// # Errors
    ///
    /// Returns [`IngestError`] if the refined version of `seg` was already ingested.
    fn ingest(
        &mut self,
        seg: SegmentId,
        level: Level,
        pts: &[Point],
    ) -> Result<IngestStats, IngestError>;
    /// Re-meshes the dirty blocks.
    ///
    /// # Returns
    ///
    /// The ids of the blocks whose mesh actually changed. Only those blocks get a new version.
    fn extract(&mut self) -> Vec<BlockId>;
    /// Work done and changes found by the last `extract`.
    fn extract_stats(&self) -> ExtractStats {
        ExtractStats::default()
    }
    fn mesh(&self, id: &BlockId) -> Option<&BlockMesh>;
    /// Current version of a block.
    ///
    /// A deleted block (whose `mesh` is `None`) still returns its last version. A block that was never reported
    /// returns 0.
    fn version(&self, id: &BlockId) -> u32 {
        self.mesh(id).map_or(0, |m| m.version)
    }
    fn meshes(&self) -> Box<dyn Iterator<Item = &BlockMesh> + '_>;
    /// Bytes used to store the distance (or height) field.
    fn field_bytes(&self) -> usize;
    fn mesh_bytes(&self) -> usize {
        self.meshes().map(|m| m.heap_bytes()).sum()
    }
    fn total_tris(&self) -> usize {
        self.meshes().map(|m| m.tri_count()).sum()
    }
    fn block_count(&self) -> usize {
        self.meshes().filter(|m| m.tri_count() > 0).count()
    }
}

/// Set of dirty blocks.
///
/// The value is `true` if the block's own field changed and `false` if it is only a neighbour pulled in by the
/// boundary rules.
pub(crate) type Dirty = rustc_hash::FxHashMap<BlockId, bool>;

/// Which boundary voxels of its neighbours a block reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Halo {
    /// Distance field: reads local `-1..=dim`, i.e. voxel `dim-1` of the `-1` neighbour and voxel `0` of the `+1`
    /// neighbour. The voxel that is `-1` on all three axes is never used by any cell vertex, because no edge uses cell
    /// `(-1,-1,-1)`.
    Both,
    /// Height field: reads only local `0..=dim`, i.e. voxel `0` of the `+1` neighbour. Changes at `dim-1` do not
    /// affect neighbours.
    Upper,
}

/// Bit index of the neighbour offset `(dx, dy, dz)` in `{-1,0,1}³`.
///
/// A 27-bit mask built from these bits lists the neighbours that must be re-extracted.
#[inline]
pub(crate) const fn nbit(dx: i32, dy: i32, dz: i32) -> u32 {
    1 << ((dx + 1) + (dy + 1) * 3 + (dz + 1) * 9)
}
/// The block itself.
pub(crate) const SELF_BIT: u32 = nbit(0, 0, 0);
/// The six face neighbours plus the block itself. All other bits are edge or corner (diagonal) neighbours.
pub(crate) const FACE_BITS: u32 = SELF_BIT
    | nbit(-1, 0, 0)
    | nbit(1, 0, 0)
    | nbit(0, -1, 0)
    | nbit(0, 1, 0)
    | nbit(0, 0, -1)
    | nbit(0, 0, 1);

/// Builds the mask of neighbours to re-extract from a changed local range `[min, max]`, including the block itself.
///
/// - If the range reaches `0`, the `-1` neighbour reads that voxel as its local `dim`.
/// - If it reaches `dim-1`, the `+1` neighbour reads it as its local `-1` (only for [`Halo::Both`]).
///
/// The rule is exact for face neighbours, because a voxel on that face really changed. For diagonal neighbours it is
/// a conservative box test. In 3D the `(+1,+1,+1)` neighbour is always cleared: it reads voxel `(dim-1)³` only as its
/// own `(-1,-1,-1)`, which no edge uses, so that voxel has no effect on its mesh.
pub(crate) fn halo_mask<const D: usize>(min: [i32; D], max: [i32; D], dim: i32, halo: Halo) -> u32 {
    let mut ranges = [(0i32, 0i32); 3];
    for a in 0..D {
        let hi = halo == Halo::Both && max[a] == dim - 1;
        ranges[a] = (if min[a] == 0 { -1 } else { 0 }, if hi { 1 } else { 0 });
    }
    let mut m = 0;
    for dz in ranges[2].0..=ranges[2].1 {
        for dy in ranges[1].0..=ranges[1].1 {
            for dx in ranges[0].0..=ranges[0].1 {
                m |= nbit(dx, dy, dz);
            }
        }
    }
    if D == 3 {
        m &= !nbit(1, 1, 1);
    }
    m
}

/// Inserts the blocks of `mask` into `dirty`.
///
/// The block itself is marked as directly changed; neighbours are marked as neighbours unless they are already
/// marked as direct.
pub(crate) fn mark_mask(dirty: &mut Dirty, id: BlockId, mask: u32) {
    for b in 0..27 {
        if mask & (1 << b) == 0 {
            continue;
        }
        let d = [b % 3 - 1, (b / 3) % 3 - 1, b / 9 - 1];
        if d == [0, 0, 0] {
            dirty.insert(id, true);
        } else {
            dirty
                .entry([id[0] + d[0], id[1] + d[1], id[2] + d[2]])
                .or_insert(false);
        }
    }
}

/// Marks dirty blocks directly from the range rule of [`halo_mask`]. Used by the height field.
pub(crate) fn mark_dirty<const D: usize>(
    dirty: &mut Dirty,
    id: BlockId,
    min: [i32; D],
    max: [i32; D],
    dim: i32,
    halo: Halo,
) {
    mark_mask(dirty, id, halo_mask(min, max, dim, halo));
}

/// Work done and changes found by the last `extract`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExtractStats {
    /// Number of blocks re-extracted (the amount of work).
    pub reextracted: usize,
    /// Of those, the blocks whose own field changed. The rest are neighbours pulled in by the boundary rules.
    pub reextracted_direct: usize,
    /// Blocks whose mesh changed, got a new version and were reported.
    pub updated: usize,
    /// Re-extracted directly changed blocks that were bitwise equal to the stored mesh and therefore not reported.
    pub unchanged_direct: usize,
    /// Re-extracted neighbour blocks that were bitwise equal to the stored mesh and therefore not reported.
    pub unchanged_neighbor: usize,
    /// Unchanged blocks that were empty before and after (never reported in the first place).
    pub unchanged_empty: usize,
}

/// Drains the dirty set into a list of `(id, direct)` sorted by id.
pub(crate) fn drain_dirty(dirty: &mut Dirty) -> Vec<(BlockId, bool)> {
    let mut ids: Vec<(BlockId, bool)> = dirty.drain().collect();
    ids.sort_unstable();
    ids
}

/// Returns whether a freshly built mesh is bitwise equal to the stored one.
///
/// A block without a stored mesh compares equal to an empty mesh.
pub(crate) fn same_mesh(old: Option<&BlockMesh>, pos: &[f32], col: &[u8], idx: &[u32]) -> bool {
    match old {
        None => idx.is_empty(),
        Some(m) => {
            m.indices == idx
                && m.colors == col
                && m.positions.len() == pos.len()
                && m.positions
                    .iter()
                    .zip(pos)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        }
    }
}

/// One re-extracted block: `(id, direct, same, (positions, colours, indices))`.
pub(crate) type BuiltBlock = (BlockId, bool, bool, (Vec<f32>, Vec<u8>, Vec<u32>));

/// Applies the result of an `extract`.
///
/// Blocks whose mesh is unchanged keep their version and are not reported. Changed blocks get a new version; an
/// empty mesh removes the stored mesh but keeps the version.
///
/// # Arguments
///
/// * `built` - Re-extracted blocks in id order.
///
/// # Returns
///
/// The ids of the changed blocks, in the order of `built`.
pub(crate) fn apply_built(
    built: Vec<BuiltBlock>,
    meshes: &mut rustc_hash::FxHashMap<BlockId, BlockMesh>,
    versions: &mut rustc_hash::FxHashMap<BlockId, u32>,
    stats: &mut ExtractStats,
) -> Vec<BlockId> {
    *stats = ExtractStats {
        reextracted: built.len(),
        ..Default::default()
    };
    let mut out = Vec::new();
    for (id, direct, same, (positions, colors, indices)) in built {
        stats.reextracted_direct += direct as usize;
        if same {
            stats.unchanged_empty += indices.is_empty() as usize;
            if direct {
                stats.unchanged_direct += 1;
            } else {
                stats.unchanged_neighbor += 1;
            }
            continue;
        }
        let ver = versions.entry(id).or_insert(0);
        *ver += 1;
        if indices.is_empty() {
            meshes.remove(&id);
        } else {
            meshes.insert(
                id,
                BlockMesh {
                    id,
                    version: *ver,
                    positions,
                    colors,
                    indices,
                },
            );
        }
        out.push(id);
    }
    stats.updated = out.len();
    out
}

#[cfg(test)]
pub(crate) mod testutil {
    use super::*;
    use rustc_hash::FxHashMap;

    pub type Batch = (SegmentId, Level, Vec<Point>);

    /// Checks the incremental contract of a mesher over a sequence of steps.
    ///
    /// After every step it extracts and applies only the reported blocks to a client copy, then checks that
    /// (1) the client copy equals the server meshes and (2) after all steps the result is bitwise equal to ingesting
    /// everything and extracting once.
    ///
    /// # Returns
    ///
    /// The reported block ids and extract statistics of each step.
    ///
    /// # Panics
    ///
    /// Panics if any of the checks fail.
    pub fn check_incremental<M: Mesher>(
        mk: impl Fn() -> M,
        steps: &[Vec<Batch>],
    ) -> Vec<(Vec<BlockId>, ExtractStats)> {
        let mut a = mk();
        let mut client: FxHashMap<BlockId, BlockMesh> = FxHashMap::default();
        let mut log = Vec::new();
        for st in steps {
            for (seg, lv, pts) in st {
                a.ingest(*seg, *lv, pts).unwrap();
            }
            let ids = a.extract();
            for id in &ids {
                let v = a.version(id);
                if let Some(old) = client.get(id) {
                    assert!(v > old.version, "version did not increase for {id:?}");
                }
                match a.mesh(id) {
                    Some(m) => {
                        assert_eq!(m.version, v);
                        client.insert(*id, m.clone());
                    }
                    None => {
                        client.remove(id);
                    }
                }
            }
            let mut srv: Vec<BlockMesh> = a.meshes().cloned().collect();
            srv.sort_by_key(|m| m.id);
            let mut cl: Vec<BlockMesh> = client.values().cloned().collect();
            cl.sort_by_key(|m| m.id);
            assert_eq!(srv, cl, "client copy differs from server meshes");
            log.push((ids, a.extract_stats()));
        }
        let mut b = mk();
        for st in steps {
            for (seg, lv, pts) in st {
                b.ingest(*seg, *lv, pts).unwrap();
            }
        }
        b.extract();
        let strip = |m: &M| {
            let mut v: Vec<_> = m
                .meshes()
                .map(|x| {
                    (
                        x.id,
                        x.positions.clone(),
                        x.colors.clone(),
                        x.indices.clone(),
                    )
                })
                .collect();
            v.sort_by_key(|x| x.0);
            v
        };
        assert_eq!(
            strip(&a),
            strip(&b),
            "incremental result differs from a single full extraction"
        );
        log
    }

    /// Samples the axis-aligned rectangle `[x0, x1) x [y0, y1)` at height `z` every `step` metres, with normals
    /// pointing up.
    pub fn plane(
        x0: f32,
        x1: f32,
        y0: f32,
        y1: f32,
        z: f32,
        step: f32,
        rgb: [u8; 3],
    ) -> Vec<Point> {
        let mut v = Vec::new();
        let mut x = x0;
        while x < x1 {
            let mut y = y0;
            while y < y1 {
                v.push(Point {
                    pos: [x, y, z],
                    rgb,
                    normal: [0.0, 0.0, 1.0],
                });
                y += step;
            }
            x += step;
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Checks the neighbour rules of [`halo_mask`].
    ///
    /// A change strictly inside marks only the block itself. Local `0` marks the `-1` neighbour and `dim-1` the `+1`
    /// neighbour. A change touching every face marks all 26 neighbours except `(+1,+1,+1)`. The height field never
    /// marks `+1` neighbours.
    #[test]
    fn halo_mask_rules() {
        let d = 32;
        assert_eq!(halo_mask([3, 3, 3], [20, 20, 20], d, Halo::Both), SELF_BIT);
        assert_eq!(
            halo_mask([0, 3, 3], [20, 20, 20], d, Halo::Both),
            SELF_BIT | nbit(-1, 0, 0)
        );
        assert_eq!(
            halo_mask([3, 3, 3], [31, 20, 20], d, Halo::Both),
            SELF_BIT | nbit(1, 0, 0)
        );
        let all = halo_mask([0, 0, 0], [31, 31, 31], d, Halo::Both);
        assert_eq!(all.count_ones(), 26);
        assert_eq!(all & nbit(1, 1, 1), 0);
        assert_eq!(
            halo_mask([0, 0], [31, 31], d, Halo::Upper),
            SELF_BIT | nbit(-1, 0, 0) | nbit(0, -1, 0) | nbit(-1, -1, 0)
        );
    }
}
