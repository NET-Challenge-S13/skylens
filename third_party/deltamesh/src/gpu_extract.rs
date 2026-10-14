//! GPU Surface Nets extraction for many blocks at once.
//!
//! The rules match `build_block_in` in the CPU mesher (`sdf.rs`). The input is a [`Volumes`]
//! batch: for every block a header of region occupancy bits followed by the `P^3` distances as
//! `f32` bits, where `P = dim + 2` and unobserved samples hold [`MARK`]. Colours are not part of
//! the volume. They are read from the brick pool of the resident field through a per-region slot
//! table, and only for the corners of vertex cells.
//!
//! A batch of blocks costs one submission and one wait ([`GpuExtractor::submit`] followed by
//! [`GpuExtractor::finish`]). Submitting batch `k + 1` before finishing batch `k` lets the CPU
//! unpack results while the GPU works on the next batch.
//!
//! One submission runs these kernels in a single compute pass:
//!
//! - `k_valid`: one thread per 32-bit word of the cell bitmap (32 consecutive cells). A cell has
//!   a vertex when all 8 corners are observed and the signs are mixed. Rows whose regions are not
//!   occupied are skipped without reading distances, and neighbouring cells on an x row share a
//!   corner cross-section. Only cell tiles (`TC` = 2048 cells) that contain a vertex cell are
//!   appended to the tile list and the indirect dispatch arguments.
//! - `k_flags`: per listed tile, the vertex cells are compacted in shared memory (order
//!   preserved). Each vertex cell derives, from its 8 corner signs and the vertex bits of its 27
//!   neighbour cells, three quad bits (edges along +x, +y, +z) and a bit telling whether any quad
//!   uses the cell. Per-tile prefix counts of quads and vertices come from popcounts over three
//!   bit planes (quad count bit 0, bit 1, used), so no 256-wide prefix sum is needed.
//! - scan: exclusive prefix sum of the per-tile vertex counts (integer, deterministic). Sums of
//!   tiles without vertex cells are cleared to zero before the pass.
//! - `k_starts`: vertex start offset of every block.
//! - `k_write`: per listed tile, the vertex cells are compacted again and every used cell writes
//!   its vertex record (offset inside the cell, colour, cell coordinate, quad bits, winding flip)
//!   at its prefix-sum position. The vertex index is the cell order. The output buffer is sized
//!   from the largest per-block count seen in earlier batches; records past its end are not
//!   written.
//!
//! The GPU does not emit quads. The CPU builds the triangle indices from the quad bits of the
//! vertex records and a cell-coordinate to vertex-index table. Quad order is cell z, y, x, then
//! axis, which matches the CPU mesher.
//!
//! The start offsets and the output buffer are copied into one staging buffer and downloaded
//! together. If the output did not fit (rare), only the write pass is rerun with a larger buffer.
//! On unified-memory devices (`GpuCtx::uma`) the output and start buffers are mapped directly,
//! without a staging copy. Pool bindings and accessors come from `gpu_field::pool_wgsl`, which is
//! prepended to the shader. Final positions, `(global cell + offset) * voxel`, are computed on the
//! CPU with the same expression as the CPU mesher.
//!
//! # Determinism
//!
//! Values are never accumulated with atomics. Atomics are used only for bitwise OR and for
//! reserving tile list entries. The tile list order can differ between runs, but every tile has a
//! fixed output position, so the output is bitwise identical on the same device. Shared memory is
//! always written before it is read, so workgroup memory zero-initialisation is disabled to cut
//! workgroup start cost.
//!
//! # Differences from the CPU mesher
//!
//! No orphan vertices are produced when a quad fails part way, and vertex indices follow cell
//! order.

use crate::gpu::GpuCtx;
use crate::gpu_field::{PageLayout, dummy_storage, pool_entries, pool_pages, pool_wgsl};
use crate::{BlockId, Config};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// Extraction result for one block.
///
/// `indices` refer to vertices within this block. `seam[i]` is true when vertex `i` lies in a
/// cell on the block boundary (local cell coordinate `-1` or `dim - 1`); the simplifier keeps
/// those vertices fixed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExtractedBlock {
    pub id: BlockId,
    pub positions: Vec<f32>,
    pub colors: Vec<u8>,
    pub indices: Vec<u32>,
    pub seam: Vec<bool>,
}

/// Workgroup size of all kernels.
const WG: u32 = 256;
/// Cells per tile. Must match `TC` in the shader.
const TC_CELLS: u64 = 2048;
/// Elements per prefix-sum tile (256 threads times 4).
const TILE: u64 = 1024;
/// Scratch memory budget in bytes for the per-cell buffers of one batch.
const SCRATCH_BUDGET: u64 = 256 << 20;

const SHADER: &str = r#"
/// Per-batch parameters.
///
/// - `vbase`: start of the first block's volume in `vol`, in u32 words.
/// - `capv`: number of vertices that fit in `outv`; vertices past it are not written.
/// - `capq`: unused.
/// - `owp`: u32 words of the block header (region occupancy bits).
/// - `stride`: u32 words per block volume.
/// - `obase`: slot table region index of the first block (`first block * nr^3`).
/// - `bsh`, `msh`: brick pool page layout shifts for `pool_page`.
struct Params { dim: u32, nblk: u32, vbase: u32, capv: u32, capq: u32, owp: u32, stride: u32, obase: u32, bsh: u32, msh: u32, pad0: u32, pad1: u32 };

/// Bindings.
///
/// - `vol`: per block `[occupancy bits][P^3 distances]` as f32 bits; unobserved samples hold MARK.
/// - `cells`: per vertex cell, the packed tile-local prefix and flags written by `k_flags`.
/// - `outv`: 5 words per vertex: cell offset x, y, z (f32 bits);
///   `r | g << 8 | b << 16 | quad bits (+x, +y, +z edges) << 24 | flip << 27`;
///   cell coordinate plus one as `x | y << 10 | z << 20`.
/// - `starts`: `[quad starts, nblk + 1][vertex starts, nblk + 1]`.
/// - `sumq`, `sumv`: per-tile quad and vertex counts, turned into exclusive prefix sums.
/// - `vbits`: one bit per cell, set when the cell has a vertex.
/// - `bsl`: tile-local prefix at the first cell of each block (same packing as `cells`).
/// - `targs`: indirect dispatch arguments; x counts tiles with a vertex cell, y = z = 1.
/// - `tlist`: indices of tiles with a vertex cell; the order can differ between runs.
/// - `goff`, `gslots`: per block and region, the range of brick pool slots to read colours
///   from (the same table the gather uses). The pool page bindings and `pool_page` / `pool_wc`
///   are prepended by `gpu_field::pool_wgsl`.
@group(0) @binding(0) var<uniform> U: Params;
@group(0) @binding(1) var<storage, read> vol: array<u32>;
@group(0) @binding(3) var<storage, read_write> cells: array<u32>;
@group(0) @binding(6) var<storage, read_write> outv: array<u32>;
@group(0) @binding(8) var<storage, read_write> starts: array<u32>;
@group(0) @binding(9) var<storage, read_write> sumq: array<u32>;
@group(0) @binding(10) var<storage, read_write> sumv: array<u32>;
@group(0) @binding(11) var<storage, read_write> vbits: array<u32>;
@group(0) @binding(12) var<storage, read_write> bsl: array<u32>;
@group(0) @binding(13) var<storage, read_write> targs: array<atomic<u32>>;
@group(0) @binding(14) var<storage, read_write> tlist: array<u32>;
@group(0) @binding(16) var<storage, read> goff: array<u32>;
@group(0) @binding(17) var<storage, read> gslots: array<u32>;

/// Colour at padded sample `q` of block `blk`: sum(w * c) / sum(w).
///
/// Layers are summed in the same order as the gather, skipping layers with zero weight.
fn corner_color(blk: u32, q: vec3<u32>) -> vec3<f32> {
  let nr = U.dim / 8u + 2u;
  let r = (((q.z + 7u) >> 3u) * nr + ((q.y + 7u) >> 3u)) * nr + ((q.x + 7u) >> 3u);
  let vi = ((((q.z + 7u) & 7u) * 8u) + ((q.y + 7u) & 7u)) * 8u + ((q.x + 7u) & 7u);
  let o = U.obase + blk * nr * nr * nr + r;
  let e1 = goff[o + 1u];
  var w = 0.0;
  var c = vec3<f32>(0.0);
  for (var e = goff[o]; e < e1; e = e + 1u) {
    let pg = pool_page(gslots[e], U.bsh, U.msh);
    let v = pool_wc(pg, vi);
    if (v.x != 0.0) {
      w = w + v.x;
      c = c + v.yzw;
    }
  }
  return c / w;
}

/// Distance of unobserved samples. Must match `MARK` on the Rust side.
const MARK: f32 = 3.0e38;

/// Distance sample at u32 index `i` of `vol`. Unobserved samples are MARK (the gather applies
/// `min_weight`).
fn dv(i: u32) -> f32 {
  return bitcast<f32>(vol[i]);
}

/// Start of block `blk` (its occupancy header) in u32 words. Distances start `owp` words later.
fn vbase_of(blk: u32) -> u32 {
  return U.vbase + blk * U.stride;
}

/// Linear index of padded coordinate `q` within a block's distance array.
fn pidx(q: vec3<u32>) -> u32 {
  let p = U.dim + 2u;
  return (q.z * p + q.y) * p + q.x;
}

/// Offset of cell corner `k = x + 2y + 4z`.
fn corner(k: u32) -> vec3<u32> {
  return vec3<u32>(k & 1u, (k >> 1u) & 1u, k >> 2u);
}

/// Unit vector along axis `a`.
fn unit(a: u32) -> vec3<i32> {
  return vec3<i32>(select(0, 1, a == 0u), select(0, 1, a == 1u), select(0, 1, a == 2u));
}

/// Flags of vertex cell `i`: quad bits in bits 0..3 and the "used by a quad" bit in bit 3.
///
/// Only called for cells with a vertex, so all 8 corners are observed. Uses only the 8 corner
/// signs (`ng`, bit `k` set for a negative corner `k = x + 2y + 4z`) and the vertex bits of the 27
/// neighbour cells from `k_valid`. The neighbour bits are packed as `(x+1) + 3(y+1) + 9(z+1)`;
/// cells outside the block's cell range count as 0. For each of the 9 (y, z) rows, the bits for
/// x-1, x and x+1 are read from three independent words at once.
///
/// For each axis `k` and each of the 4 edges that have this cell on their ring
/// (`v = c + a * ei + b * ej`), the edge is a crossing when its two corners `ov` and `ov + ek`
/// differ in sign. A quad needs all four ring cells `v, v - ei, v - ei - ej, v - ej` to have a
/// vertex. The cell owns the quad of its own edge (`s == 0`).
fn cell_flags(i: u32) -> u32 {
  let pc = U.dim + 1u;
  let pc3 = pc * pc * pc;
  let p = U.dim + 2u;
  let r = i % pc3;
  let cp = vec3<u32>(r % pc, (r / pc) % pc, r / (pc * pc));
  let base = vbase_of(i / pc3) + U.owp;
  var ng = 0u;
  for (var k = 0u; k < 8u; k = k + 1u) {
    ng = ng | (select(0u, 1u, dv(base + pidx(cp + corner(k))) < 0.0) << k);
  }
  let c = vec3<i32>(cp) - vec3<i32>(1);
  let idim = i32(U.dim);
  let ipc = i32(pc);
  var cv = 0u;
  let lastw = arrayLength(&vbits) - 1u;
  let xm = select(7u, 6u, cp.x == 0u) & select(7u, 3u, cp.x == pc - 1u);
  for (var rr = 0; rr < 9; rr = rr + 1) {
    let oy = rr % 3 - 1;
    let oz = rr / 3 - 1;
    let qy = i32(cp.y) + oy;
    let qz = i32(cp.z) + oz;
    if (qy < 0 || qy >= ipc || qz < 0 || qz >= ipc) { continue; }
    let s1 = u32(i32(i) + (oz * ipc + oy) * ipc);
    let w1 = s1 >> 5u;
    let b = s1 & 31u;
    let wc = vbits[w1];
    let wm = vbits[max(w1, 1u) - 1u];
    let wp = vbits[min(w1 + 1u, lastw)];
    let lo = select((wc >> (b - 1u)) & 1u, wm >> 31u, b == 0u);
    let hi = select((wc >> (b + 1u)) & 1u, wp & 1u, b == 31u);
    let three = (lo | (((wc >> b) & 1u) << 1u) | (hi << 2u)) & xm;
    cv = cv | (three << u32(rr * 3));
  }
  var qb = 0u;
  var used = 0u;
  let w = vec3<i32>(1, 3, 9);
  for (var k = 0u; k < 3u; k = k + 1u) {
    let ei = unit((k + 1u) % 3u);
    let ej = unit((k + 2u) % 3u);
    let ek = unit(k);
    for (var s = 0; s < 4; s = s + 1) {
      let ov = (s & 1) * ei + (s >> 1) * ej;
      let v = c + ov;
      if (any(v < vec3<i32>(0)) || any(v >= vec3<i32>(idim))) { continue; }
      let k0 = u32(dot(ov, vec3<i32>(1, 2, 4)));
      let k1 = u32(dot(ov + ek, vec3<i32>(1, 2, 4)));
      if (((ng >> k0) & 1u) == ((ng >> k1) & 1u)) { continue; }
      let b0 = dot(ov + vec3<i32>(1), w);
      let bi = dot(ei, w);
      let bj = dot(ej, w);
      let need = (1u << u32(b0)) | (1u << u32(b0 - bi)) | (1u << u32(b0 - bi - bj)) | (1u << u32(b0 - bj));
      if ((cv & need) != need) { continue; }
      used = 1u;
      if (s == 0) { qb = qb | (1u << k); }
    }
  }
  return qb | (used << 3u);
}

/// Summary of the 4 distances at padded coordinates (x, y..y+1, z..z+1).
///
/// Returns 99 if any of them is unobserved, otherwise the number of negative samples.
fn column(base: u32, x: u32, y: u32, z: u32) -> u32 {
  let p = U.dim + 2u;
  let i0 = base + (z * p + y) * p + x;
  let d0 = dv(i0); let d1 = dv(i0 + p); let d2 = dv(i0 + p * p); let d3 = dv(i0 + p * p + p);
  if (d0 == MARK || d1 == MARK || d2 == MARK || d3 == MARK) { return 99u; }
  return select(0u, 1u, d0 < 0.0) + select(0u, 1u, d1 < 0.0) + select(0u, 1u, d2 < 0.0) + select(0u, 1u, d3 < 0.0);
}

/// Occupancy bits of the `nr` regions along x in region row (ry, rz) of the block header `hb`.
///
/// Regions are numbered `(rz * nr + ry) * nr + rx`, so the bits of a row are contiguous.
/// Assumes `nr <= 32` (`block_dim <= 240`).
fn occ_row(hb: u32, nr: u32, ry: u32, rz: u32) -> u32 {
  let b = (rz * nr + ry) * nr;
  let w = b >> 5u;
  let sft = b & 31u;
  var v = vol[hb + w] >> sft;
  if (sft + nr > 32u) { v = v | (vol[hb + w + 1u] << (32u - sft)); }
  return v & select((1u << nr) - 1u, 0xffffffffu, nr >= 32u);
}

/// Computes the vertex bitmap and the list of tiles that contain vertex cells.
///
/// One thread per bitmap word (32 consecutive cells), see `valid_word`. The bitmap is filled up
/// to whole tiles; bits past `n` are zero. A workgroup covers `256 / TCW` tiles and appends only
/// the tiles with a vertex cell to `tlist`. The list order can differ between runs, but every
/// tile has a fixed output position, so the output does not.
@compute @workgroup_size(256)
fn k_valid(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  if (li < TPW) { atomicStore(&tany[li], 0u); }
  workgroupBarrier();
  let w0 = (wg.y * nwg.x + wg.x) * 256u;
  let w = w0 + li;
  let pc = U.dim + 1u;
  let pc3 = pc * pc * pc;
  let n = U.nblk * pc3;
  let nw = ((n + TC) / TC) * TCW;
  if (w < nw) {
    let bits = valid_word(w, n, pc, pc3);
    vbits[w] = bits;
    if (bits != 0u) { atomicOr(&tany[li / TCW], 1u); }
  }
  workgroupBarrier();
  if (li < TPW && w0 + li * TCW < nw && atomicLoad(&tany[li]) != 0u) {
    let k = atomicAdd(&targs[0], 1u);
    tlist[k] = w0 / TCW + li;
  }
}

/// Tiles per `k_valid` workgroup.
const TPW: u32 = 256u / TCW;
var<workgroup> tany: array<atomic<u32>, TPW>;

/// Vertex bits of bitmap word `w` (cells `32w .. 32w + 31`).
///
/// A cell has a vertex when all 8 corners are observed and the signs are mixed. For each x row
/// the occupancy of the two y and two z regions touched by the cell corners is checked first;
/// rows without occupied regions (most of them) are skipped, as are cells whose corner regions
/// are empty (the gather does not write empty regions). Neighbouring cells on a row share a
/// corner cross-section, so the `column` summary is carried to the next cell.
fn valid_word(w: u32, n: u32, pc: u32, pc3: u32) -> u32 {
  let nr = U.dim / 8u + 2u;
  var bits = 0u;
  var i = w * 32u;
  let iend = min(i + 32u, n);
  while (i < iend) {
    let blk = i / pc3;
    let r = i - blk * pc3;
    let x0 = r % pc;
    let y = (r / pc) % pc;
    let z = r / (pc * pc);
    let seg = min(iend - i, pc - x0);
    let hb = vbase_of(blk);
    let base = hb + U.owp;
    let ry0 = (y + 7u) >> 3u; let ry1 = (y + 8u) >> 3u;
    let rz0 = (z + 7u) >> 3u; let rz1 = (z + 8u) >> 3u;
    let okx = occ_row(hb, nr, ry0, rz0) & occ_row(hb, nr, ry1, rz0) & occ_row(hb, nr, ry0, rz1) & occ_row(hb, nr, ry1, rz1);
    if (okx == 0u) { i = i + seg; continue; }
    var ca = 0u;
    var have = false;
    for (var k = 0u; k < seg; k = k + 1u) {
      let x = x0 + k;
      let need = (1u << ((x + 7u) >> 3u)) | (1u << ((x + 8u) >> 3u));
      if ((okx & need) != need) { have = false; continue; }
      if (!have) { ca = column(base, x, y, z); }
      let cb = column(base, x + 1u, y, z);
      let nn = ca + cb;
      if (ca != 99u && cb != 99u && nn > 0u && nn < 8u) {
        bits = bits | (1u << (i + k - w * 32u));
      }
      ca = cb;
      have = true;
    }
    i = i + seg;
  }
  return bits;
}

/// Cells per tile. Tile-local prefixes fit in 13 bits for quads (at most 3 * TC) and 12 bits
/// for vertices (at most TC).
const TC: u32 = 2048u;
/// Bitmap words per tile.
const TCW: u32 = TC / 32u;
/// Cells per thread when scanning a tile.
const PER: u32 = TC / 256u;

/// Tile compaction state shared by `build_list`, `slot_of`, `k_flags` and `k_write`.
///
/// - `shw`: the tile's bitmap words; `wpre`: popcount of the preceding words.
/// - `list`: compacted vertex cells; the low 16 bits hold the tile-local cell index.
/// - `pl`: three bit planes over list positions (quad count bit 0, quad count bit 1, used);
///   `plw` holds their words and `ppre` the popcount of preceding words within each plane.
var<workgroup> shw: array<u32, TCW>;
var<workgroup> wpre: array<u32, TCW>;
var<workgroup> list: array<u32, TC>;
var<workgroup> shcnt: u32;
var<workgroup> shany: atomic<u32>;
var<workgroup> shn: u32;
var<workgroup> pl: array<atomic<u32>, 3u * TCW>;
var<workgroup> plw: array<u32, 3u * TCW>;
var<workgroup> ppre: array<u32, 3u * TCW>;

/// Collects the vertex cells of tile `t` into `list` in cell order and returns their count.
///
/// The count is uniform across the workgroup. Thread `li` handles cells `li * PER` to
/// `li * PER + PER - 1`. Tiles without vertex cells return 0 before the prefix step; the check
/// is a bitwise OR, so it is deterministic. The exclusive prefix over the `TCW` words is computed
/// by having each thread count the preceding words directly.
fn build_list(t: u32, li: u32) -> u32 {
  if (li == 0u) { atomicStore(&shany, 0u); }
  workgroupBarrier();
  if (li < TCW) {
    let w = vbits[t * TCW + li];
    shw[li] = w;
    if (w != 0u) { atomicOr(&shany, 1u); }
  }
  workgroupBarrier();
  if (li == 0u) { shcnt = atomicLoad(&shany); }
  if (workgroupUniformLoad(&shcnt) == 0u) { return 0u; }
  if (li < TCW) {
    var a = 0u;
    for (var k = 0u; k < li; k = k + 1u) { a = a + countOneBits(shw[k]); }
    wpre[li] = a;
    if (li == TCW - 1u) { shn = a + countOneBits(shw[TCW - 1u]); }
  }
  workgroupBarrier();
  for (var r = 0u; r < PER; r = r + 1u) {
    let c = li * PER + r;
    let w = shw[c >> 5u];
    let b = c & 31u;
    if (((w >> b) & 1u) != 0u) { list[wpre[c >> 5u] + countOneBits(w & ((1u << b) - 1u))] = c; }
  }
  return workgroupUniformLoad(&shn);
}

/// Number of vertex cells before tile-local cell `c`. Valid only after `build_list`.
fn slot_of(c: u32) -> u32 {
  return wpre[c >> 5u] + countOneBits(shw[c >> 5u] & ((1u << (c & 31u)) - 1u));
}

/// Computes cell flags and tile-local prefixes for one listed tile of `TC` cells.
///
/// Vertex cells are compacted first (order preserved) so the heavy per-cell work is evenly
/// spread across threads. For every vertex cell, `cells` receives the tile-local quad prefix
/// (13 bits) | vertex prefix (12 bits) << 13 | quad bits (3) << 25 | used << 28. The tile-local
/// prefix at the first cell of each block that starts in this tile goes to `bsl`. The tile
/// covers `n + 1` elements in total; the last one is the slot for the grand total.
///
/// Each list position `j` is marked in the three bit planes (quad count bit 0, bit 1, used).
/// Bitwise OR makes this independent of thread order. The quad and vertex counts before `j` are
/// the per-plane word prefix plus a popcount inside the word, so no 256-wide prefix sum is
/// needed.
@compute @workgroup_size(256)
fn k_flags(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = tlist[wg.x];
  let pc = U.dim + 1u;
  let pc3 = pc * pc * pc;
  let n = U.nblk * pc3;
  let ntiles = (n + TC) / TC;
  if (t >= ntiles) { return; }
  let cnt = build_list(t, li);
  if (cnt == 0u) {
    if (li == 0u) {
      sumq[t] = 0u; sumv[t] = 0u;
      for (var j = (t * TC + pc3 - 1u) / pc3; j <= U.nblk && j * pc3 < (t + 1u) * TC; j = j + 1u) { bsl[j] = 0u; }
    }
    return;
  }
  if (li < 3u * TCW) { atomicStore(&pl[li], 0u); }
  workgroupBarrier();
  let t0 = t * TC;
  for (var j = li; j < cnt; j = j + 256u) {
    let f = cell_flags(t0 + list[j]);
    list[j] = list[j] | (f << 16u);
    let w = j >> 5u;
    let m = 1u << (j & 31u);
    let qn = countOneBits(f & 7u);
    if ((qn & 1u) != 0u) { atomicOr(&pl[w], m); }
    if ((qn & 2u) != 0u) { atomicOr(&pl[TCW + w], m); }
    if ((f >> 3u) != 0u) { atomicOr(&pl[2u * TCW + w], m); }
  }
  workgroupBarrier();
  if (li < 3u * TCW) {
    plw[li] = atomicLoad(&pl[li]);
    let b0 = (li / TCW) * TCW;
    var a = 0u;
    for (var k = b0; k < li; k = k + 1u) { a = a + countOneBits(atomicLoad(&pl[k])); }
    ppre[li] = a;
  }
  workgroupBarrier();
  for (var j = li; j < cnt; j = j + 256u) {
    let e = list[j];
    let f = e >> 16u;
    cells[t0 + (e & 65535u)] = plane_pre(j) | ((f & 7u) << 25u) | ((f >> 3u) << 28u);
  }
  if (li == 0u) {
    let tot = plane_pre(TC);
    sumq[t] = tot & 8191u;
    sumv[t] = tot >> 13u;
    for (var j = (t0 + pc3 - 1u) / pc3; j <= U.nblk && j * pc3 < t0 + TC; j = j + 1u) {
      bsl[j] = plane_pre(min(slot_of(j * pc3 - t0), cnt));
    }
  }
}

/// Quad count | vertex count << 13 before list position `j`; `j = TC` gives the tile total.
///
/// Valid only after the bit planes of `k_flags` are complete.
fn plane_pre(j: u32) -> u32 {
  let w = min(j >> 5u, TCW - 1u);
  let m = select((1u << (j & 31u)) - 1u, 0xffffffffu, j >= TC);
  let eq = ppre[w] + countOneBits(plw[w] & m) + 2u * (ppre[TCW + w] + countOneBits(plw[TCW + w] & m));
  let ev = ppre[2u * TCW + w] + countOneBits(plw[2u * TCW + w] & m);
  return eq | (ev << 13u);
}

/// Global vertex index of cell `i`: tile-local vertex prefix plus the scanned tile sum.
fn vglob(i: u32) -> u32 { return ((cells[i] >> 13u) & 4095u) + sumv[i / TC]; }

/// Writes the quad and vertex start offset of every block (one thread per block, plus the end).
@compute @workgroup_size(256)
fn k_starts(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let j = (wg.y * nwg.x + wg.x) * 256u + li;
  if (j > U.nblk) { return; }
  let pc = U.dim + 1u;
  let t = (j * pc * pc * pc) / TC;
  starts[j] = (bsl[j] & 8191u) + sumq[t];
  starts[U.nblk + 1u + j] = ((bsl[j] >> 13u) & 4095u) + sumv[t];
}

/// First and second corner of each of the 12 cell edges.
const EA = array<u32, 12>(0u, 2u, 4u, 6u, 0u, 1u, 4u, 5u, 0u, 1u, 2u, 3u);
const EB = array<u32, 12>(1u, 3u, 5u, 7u, 2u, 3u, 6u, 7u, 4u, 5u, 6u, 7u);

/// Writes the vertex records of one listed tile.
///
/// Vertex cells are compacted to the front first so that SIMD lanes are not idle on empty cells.
@compute @workgroup_size(256)
fn k_write(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = tlist[wg.x];
  let pc = U.dim + 1u;
  let n = U.nblk * pc * pc * pc;
  if (t >= (n + TC) / TC) { return; }
  let cnt = build_list(t, li);
  for (var j = li; j < cnt; j = j + 256u) { write_cell(t * TC + list[j]); }
}

/// Writes the vertex record of cell `i` if a quad uses it.
///
/// A cell with a quad always has a vertex (quad bits != 0 implies used = 1). The vertex is the
/// mean of the sign-change points on the 12 cell edges; its colour is the mean of the linearly
/// interpolated corner colours. Corner colours are read only on crossing edges, at most once per
/// corner.
///
/// Only the in-cell offset `s / m` is written. The final position `(global cell + s / m) * voxel`
/// is computed on the CPU with the same f32 expression, because GPU fast math may contract it
/// into an FMA and differ by 1 to 2 ulp. Colours are rounded with `floor(x + 0.5)` to match
/// Rust's `round` (halfway away from zero); negative values are clamped to 0 anyway.
///
/// Winding: counter-clockwise around the +e_k normal, flipped when the first corner distance is
/// non-negative (outside to inside), the same rule as the CPU mesher.
fn write_cell(i: u32) {
  let dim = U.dim;
  let pc = dim + 1u;
  let pc3 = pc * pc * pc;
  let f = cells[i];
  let qb = (f >> 25u) & 7u;
  let used = (f >> 28u) & 1u;
  if (used == 0u) { return; }
  let blk = i / pc3;
  let r = i % pc3;
  let cp = vec3<u32>(r % pc, (r / pc) % pc, r / (pc * pc));
  let p = dim + 2u;
  let base = vbase_of(blk) + U.owp;
  var d: array<f32, 8>;
  var pi: array<u32, 8>;
  for (var k = 0u; k < 8u; k = k + 1u) {
    pi[k] = pidx(cp + corner(k));
    d[k] = dv(base + pi[k]);
  }
  {
    var s = vec3<f32>(0.0);
    var sc = vec3<f32>(0.0);
    var m = 0.0;
    var cc: array<vec3<f32>, 8>;
    var have = 0u;
    for (var e = 0u; e < 12u; e = e + 1u) {
      let a = EA[e]; let b = EB[e];
      if ((d[a] < 0.0) == (d[b] < 0.0)) { continue; }
      if (((have >> a) & 1u) == 0u) { cc[a] = corner_color(blk, cp + corner(a)); have = have | (1u << a); }
      if (((have >> b) & 1u) == 0u) { cc[b] = corner_color(blk, cp + corner(b)); have = have | (1u << b); }
      let ca = cc[a];
      let cbb = cc[b];
      let t = d[a] / (d[a] - d[b]);
      let fa = vec3<f32>(corner(a)); let fb = vec3<f32>(corner(b));
      s = s + (fa + t * (fb - fa));
      sc = sc + (ca + t * (cbb - ca));
      m = m + 1.0;
    }
    let f = s / m;
    let col = vec3<u32>(clamp(floor(sc / m + vec3<f32>(0.5)), vec3<f32>(0.0), vec3<f32>(255.0)));
    let vo = vglob(i);
    if (vo < U.capv) {
      let o = vo * 5u;
      outv[o] = bitcast<u32>(f.x);
      outv[o + 1u] = bitcast<u32>(f.y);
      outv[o + 2u] = bitcast<u32>(f.z);
      outv[o + 3u] = col.x | (col.y << 8u) | (col.z << 16u) | (qb << 24u) | (select(0u, 1u, d[0] >= 0.0) << 27u);
      outv[o + 4u] = cp.x | (cp.y << 10u) | (cp.z << 20u);
    }
  }
}
"#;

const SCAN_SHADER: &str = r#"
struct SP { n: u32, a: u32, b: u32, c: u32 };
@group(0) @binding(0) var<uniform> S: SP;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> sums: array<u32>;
var<workgroup> sh: array<u32, 256>;

/// Exclusive prefix sum within each tile of 1024 elements of `data`.
///
/// Each thread handles 4 elements. The tile total is written to `sums`.
@compute @workgroup_size(256)
fn scan_tile(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.y * nwg.x + wg.x;
  let ntiles = (S.n + 1023u) / 1024u;
  let base = t * 1024u + li * 4u;
  var v = vec4<u32>(0u);
  for (var r = 0u; r < 4u; r = r + 1u) {
    if (base + r < S.n) { v[r] = data[base + r]; }
  }
  let s = v.x + v.y + v.z + v.w;
  sh[li] = s;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off = off * 2u) {
    var add = 0u;
    if (li >= off) { add = sh[li - off]; }
    workgroupBarrier();
    sh[li] = sh[li] + add;
    workgroupBarrier();
  }
  var run = sh[li] - s;
  for (var r = 0u; r < 4u; r = r + 1u) {
    if (base + r < S.n) { data[base + r] = run; }
    run = run + v[r];
  }
  if (li == 255u && t < ntiles) { sums[t] = sh[255]; }
}

/// Adds the sum of all preceding tiles (the scanned `sums`) to every element of a tile.
@compute @workgroup_size(256)
fn scan_add(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.y * nwg.x + wg.x;
  let ntiles = (S.n + 1023u) / 1024u;
  if (t >= ntiles) { return; }
  let add = sums[t];
  let base = t * 1024u + li * 4u;
  for (var r = 0u; r < 4u; r = r + 1u) {
    if (base + r < S.n) { data[base + r] = data[base + r] + add; }
  }
}
"#;

/// Uniform parameters of the extraction shader. Layout matches `Params` in WGSL.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    dim: u32,
    nblk: u32,
    vbase: u32,
    capv: u32,
    capq: u32,
    owp: u32,
    stride: u32,
    obase: u32,
    bsh: u32,
    msh: u32,
    pad: [u32; 2],
}

/// Compute pipelines of the extraction and scan shaders.
struct Pipes {
    valid: wgpu::ComputePipeline,
    flags: wgpu::ComputePipeline,
    starts: wgpu::ComputePipeline,
    write: wgpu::ComputePipeline,
    scan_tile: wgpu::ComputePipeline,
    scan_add: wgpu::ComputePipeline,
}

/// Scratch buffers of one batch.
///
/// Buffers only grow. Each batch in flight owns one `Scratch`; finished batches return it to the
/// extractor's pool for reuse.
#[derive(Default)]
struct Scratch {
    cells: Option<wgpu::Buffer>,
    vbits: Option<wgpu::Buffer>,
    bsl: Option<wgpu::Buffer>,
    sumq: Option<wgpu::Buffer>,
    sumv: Option<wgpu::Buffer>,
    sums: Vec<Option<wgpu::Buffer>>,
    starts: Option<wgpu::Buffer>,
    targs: Option<wgpu::Buffer>,
    tlist: Option<wgpu::Buffer>,
    out: Option<wgpu::Buffer>,
    staging: Option<wgpu::Buffer>,
}

/// A batch that has been submitted but not read back yet.
///
/// Returned by [`GpuExtractor::submit`] and consumed by [`GpuExtractor::finish`].
pub struct Pending {
    sc: Scratch,
    ids: Vec<BlockId>,
    src: Volumes,
    /// Slot table region index of the first block (`block * nr^3`).
    obase: u32,
    /// Volume binding as (offset, size) in bytes.
    vbind: (u64, u64),
    /// Index of the first block's volume within the binding, in u32 words.
    vbase: u32,
    dim: i32,
    voxel: f32,
    ntiles: u64,
    capv: u64,
    sub: Option<wgpu::SubmissionIndex>,
    keep: Vec<wgpu::Buffer>,
}

/// Surface Nets extractor running on the GPU.
///
/// Holds the compiled pipelines and a pool of scratch buffer sets. Methods take `&self`; the
/// scratch pool and size estimates are behind mutexes, and GPU submissions take the device lock
/// (`GpuCtx::lock`).
pub struct GpuExtractor {
    ctx: Arc<GpuCtx>,
    pipes: Pipes,
    /// Number of brick pool page bindings.
    npg: u32,
    /// Buffer bound to unused pool page bindings.
    dummy: wgpu::Buffer,
    /// Idle scratch buffer sets.
    scratch: Mutex<Vec<Scratch>>,
    /// Output size estimate: largest per-block vertex and quad counts seen so far.
    per_block: Mutex<(f64, f64)>,
    /// Maximum blocks per batch; 0 means derived from the budget. For tests.
    max_chunk: usize,
    /// Minimum output buffer capacity in vertices. Tests set it low to exercise the rerun path.
    min_cap: u64,
}

/// Usage of scratch storage buffers.
const SU: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);
/// Usage of readback staging buffers.
const RD: wgpu::BufferUsages = wgpu::BufferUsages::MAP_READ.union(wgpu::BufferUsages::COPY_DST);

/// Returns the buffer in `slot`, reallocating it when it is smaller than `size`.
///
/// New buffers get 25% headroom and are rounded up to 256 bytes; the minimum size is 16 bytes.
fn ensure(
    dev: &wgpu::Device,
    slot: &mut Option<wgpu::Buffer>,
    size: u64,
    usage: wgpu::BufferUsages,
    label: &str,
) -> wgpu::Buffer {
    let size = size.max(16);
    if slot.as_ref().is_none_or(|b| b.size() < size) {
        let sz = (size + size / 4 + 255) & !255;
        *slot = Some(dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: sz,
            usage,
            mapped_at_creation: false,
        }));
    }
    slot.clone().unwrap()
}

/// Binding of `size` bytes (at least 4) of `buf` starting at `off`.
fn bind(buf: &wgpu::Buffer, off: u64, size: u64) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: buf,
        offset: off,
        size: wgpu::BufferSize::new(size.max(4)),
    })
}

/// Splits `g` workgroups into a 2D grid whose dimensions stay within `maxd`.
fn groups(g: u64, maxd: u32) -> (u32, u32) {
    let g = g.max(1);
    if g <= maxd as u64 {
        (g as u32, 1)
    } else {
        let y = g.div_ceil(maxd as u64);
        (g.div_ceil(y) as u32, y as u32)
    }
}

/// Rounds `x` up to a multiple of 256.
const fn align256(x: u64) -> u64 {
    (x + 255) & !255
}

/// Waits for submission `sub`, maps the first `n` bytes of each buffer and passes the mapped views
/// to `f`.
///
/// Later submissions are not waited for, so the CPU can unpack results while the next batch runs
/// on the GPU. If some map callbacks are still pending after `sub` completes, the device is polled
/// once more without a submission index, and each remaining callback is awaited for at most 30
/// seconds so a lost callback fails instead of hanging.
///
/// The device lock is held only while waiting for the mapping and is released before `f` runs:
/// `f` uses rayon, and calling into rayon while holding the lock can deadlock. The caller must not
/// hold the device lock.
///
/// # Errors
///
/// Returns an error if polling fails, a map callback reports an error or does not arrive within
/// 30 seconds, or a mapped range cannot be accessed.
fn map_many<R>(
    ctx: &GpuCtx,
    sub: wgpu::SubmissionIndex,
    bufs: &[(&wgpu::Buffer, u64)],
    f: impl FnOnce(&[&[u8]]) -> R,
) -> Result<R, String> {
    let lk = ctx.lock.lock().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    for &(b, n) in bufs {
        let tx = tx.clone();
        b.slice(0..n.max(4))
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
    }
    drop(tx);
    let waited = ctx
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(sub),
            timeout: None,
        })
        .map_err(|e| format!("GPU wait failed: {e}"));
    let mut err = waited.err();
    let mut got = Vec::with_capacity(bufs.len());
    got.extend(rx.try_iter());
    if got.len() < bufs.len() {
        if let Err(e) = ctx.device.poll(wgpu::PollType::wait_indefinitely()) {
            err.get_or_insert(format!("GPU wait failed: {e}"));
        }
        while got.len() < bufs.len() {
            match rx.recv_timeout(std::time::Duration::from_secs(30)) {
                Ok(r) => got.push(r),
                Err(_) => return Err("GPU map callback did not arrive within 30 s".into()),
            }
        }
    }
    for r in got {
        if let Err(e) = r {
            err.get_or_insert(format!("failed to map GPU result: {e}"));
        }
    }
    drop(lk);
    if let Some(e) = err {
        return Err(e);
    }
    let r = {
        let views = bufs
            .iter()
            .map(|&(b, n)| {
                b.slice(0..n.max(4))
                    .get_mapped_range()
                    .map_err(|e| format!("failed to read mapped GPU result: {e:?}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sl: Vec<&[u8]> = views.iter().map(|v| &v[..]).collect();
        f(&sl)
    };
    for &(b, _) in bufs {
        b.unmap();
    }
    Ok(r)
}

/// Distance value of unobserved samples. Must match `MARK` in the shader.
pub const MARK: f32 = 3.0e38;

/// Number of u32 words in a block header (region occupancy bits).
///
/// Regions are the `(dim / 8 + 2)^3` cells obtained by splitting the padded volume at brick
/// (8-sample) boundaries. The word count is rounded up to a multiple of 4.
pub fn occ_words(block_dim: i32) -> u64 {
    let nr = (block_dim / 8 + 2) as u64;
    nr.pow(3).div_ceil(32).next_multiple_of(4)
}

/// Number of u32 words in one block volume: occupancy header plus `(dim + 2)^3` distances.
pub fn volume_words(block_dim: i32) -> u64 {
    occ_words(block_dim) + ((block_dim + 2) as u64).pow(3)
}

/// Extraction input for a batch of blocks.
///
/// `vol` holds, per block, `[region occupancy bits][P^3 distances]` (see [`occ_words`] and
/// [`volume_words`]). Colours are read from `pool` only for samples that vertex cells need. The
/// slots of block `j`, region `r` are `slots[off[j * nr^3 + r] .. off[j * nr^3 + r + 1]]`, in
/// layer order. Component `c` (`sum(w*d)`, `sum(w)`, `sum(w*r)`, `sum(w*g)`, `sum(w*b)`) of sample
/// `vi` in the brick of slot `s` is at `[l * 2560 + c * 512 + vi]` of page `p`, where
/// `(p, l) = layout.locate(s)` (the brick pool page layout of `gpu_field`).
#[derive(Clone, Debug)]
pub struct Volumes {
    pub vol: wgpu::Buffer,
    pub pool: Vec<wgpu::Buffer>,
    pub layout: PageLayout,
    pub off: wgpu::Buffer,
    pub slots: wgpu::Buffer,
}

/// Converts CPU padded volumes into the [`Volumes`] layout and uploads them. Used for testing and
/// comparison against the CPU mesher.
///
/// Each input sample is `[sum(w*d), sum(w), sum(w*r), sum(w*g), sum(w*b)]`. A sample is observed
/// when `sum(w) >= min_weight` and `sum(w) > 0`; unobserved samples get distance [`MARK`]. The
/// colour pool gets one slot per (block, region) that contains a sample with nonzero weight,
/// holding the CPU sums unchanged, which is equivalent to a single-layer gather. All slots fit in
/// one page.
///
/// # Panics
///
/// Panics if a volume does not have `(block_dim + 2)^3` samples or a buffer cannot be mapped.
pub fn upload_volumes(
    ctx: &GpuCtx,
    vols: &[Vec<[f32; 5]>],
    block_dim: i32,
    min_weight: f32,
) -> Volumes {
    let p = (block_dim + 2) as usize;
    let p3 = p * p * p;
    assert!(
        vols.iter().all(|v| v.len() == p3),
        "block volume size does not match block_dim"
    );
    let (owp, stride) = (
        occ_words(block_dim) as usize,
        volume_words(block_dim) as usize,
    );
    let nr = (block_dim / 8 + 2) as usize;
    let nr3 = nr * nr * nr;
    let reg = |x: usize| (x + 7) >> 3;
    let mut tmp = vec![0u32; vols.len() * stride];
    let used: Vec<Vec<bool>> = tmp
        .par_chunks_mut(stride)
        .zip(vols.par_iter())
        .map(|(dst, src)| {
            let (occ, d) = dst.split_at_mut(owp);
            let mut used = vec![false; nr3];
            for (v, s) in src.iter().enumerate() {
                if s[1] != 0.0 {
                    let ri = (reg(v / (p * p)) * nr + reg((v / p) % p)) * nr + reg(v % p);
                    occ[ri / 32] |= 1 << (ri % 32);
                    used[ri] = true;
                }
                d[v] = if s[1] >= min_weight && s[1] > 0.0 {
                    (s[0] / s[1]).to_bits()
                } else {
                    MARK.to_bits()
                };
            }
            used
        })
        .collect();
    let mut off = Vec::with_capacity(vols.len() * nr3 + 1);
    let mut nslot = 0u32;
    for u in &used {
        for &x in u {
            off.push(nslot);
            nslot += x as u32;
        }
    }
    off.push(nslot);
    let slots: Vec<u32> = (0..nslot).collect();
    let mut pool = vec![0.0f32; (nslot as usize).max(1) * 2560];
    let mut k = 0usize;
    for (j, u) in used.iter().enumerate() {
        for (ri, &x) in u.iter().enumerate() {
            if !x {
                continue;
            }
            let (rz, ry, rx) = (ri / (nr * nr), (ri / nr) % nr, ri % nr);
            for (v, s) in vols[j].iter().enumerate() {
                let (pz, py, px) = (v / (p * p), (v / p) % p, v % p);
                if (reg(pz), reg(py), reg(px)) != (rz, ry, rx) {
                    continue;
                }
                let vi = (((pz + 7) & 7) * 8 + ((py + 7) & 7)) * 8 + ((px + 7) & 7);
                for (c, &val) in s.iter().enumerate() {
                    pool[k * 2560 + c * 512 + vi] = val;
                }
            }
            k += 1;
        }
    }
    let mk = |label: &str, data: &[u8]| {
        let size = (data.len() as u64).max(16);
        let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        if !data.is_empty() {
            buf.slice(0..data.len() as u64)
                .get_mapped_range_mut()
                .expect("failed to map buffer")
                .copy_from_slice(data);
        }
        buf.unmap();
        buf
    };
    let sh = nslot.max(1).next_power_of_two().trailing_zeros();
    let layout = PageLayout {
        bsh: sh,
        msh: sh,
        n: pool_pages(ctx),
    };
    Volumes {
        vol: mk("volumes", bytemuck::cast_slice(&tmp)),
        pool: vec![mk("pool", bytemuck::cast_slice(&pool))],
        layout,
        off: mk("off", bytemuck::cast_slice(&off)),
        slots: mk("slots", bytemuck::cast_slice(&slots)),
    }
}

impl GpuExtractor {
    /// Compiles the extraction pipelines on the given device.
    ///
    /// `k_flags` needs about 19 KB of workgroup memory (the 2048-entry list and prefix arrays plus
    /// the bit planes). Workgroup memory zero-initialisation is disabled because every kernel
    /// writes shared memory before reading it.
    ///
    /// # Errors
    ///
    /// Returns an error if the device offers less workgroup storage than required.
    pub fn new(ctx: Arc<GpuCtx>) -> Result<Self, String> {
        let dev = &ctx.device;
        let need = 19 * 1024;
        let have = dev.limits().max_compute_workgroup_storage_size;
        if have < need {
            return Err(format!(
                "not enough workgroup storage ({have} B < {need} B)"
            ));
        }
        let npg = pool_pages(&ctx);
        let src = format!("{}{SHADER}", pool_wgsl(npg, false));
        let m = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("extract"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let ms = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scan"),
            source: wgpu::ShaderSource::Wgsl(SCAN_SHADER.into()),
        });
        let mk = |module: &wgpu::ShaderModule, e: &str| {
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(e),
                layout: None,
                module,
                entry_point: Some(e),
                compilation_options: wgpu::PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: false,
                    ..Default::default()
                },
                cache: None,
            })
        };
        let pipes = Pipes {
            valid: mk(&m, "k_valid"),
            flags: mk(&m, "k_flags"),
            starts: mk(&m, "k_starts"),
            write: mk(&m, "k_write"),
            scan_tile: mk(&ms, "scan_tile"),
            scan_add: mk(&ms, "scan_add"),
        };
        let dummy = dummy_storage(dev);
        Ok(Self {
            ctx,
            pipes,
            npg,
            dummy,
            scratch: Mutex::new(Vec::new()),
            per_block: Mutex::new((0.0, 0.0)),
            max_chunk: 0,
            min_cap: 4096,
        })
    }

    /// Limits the number of blocks per batch; 0 restores the budget-derived limit. For tests.
    #[doc(hidden)]
    pub fn set_max_chunk(&mut self, n: usize) {
        self.max_chunk = n;
    }

    /// Sets the output size estimate (per-block vertex and quad counts) and the minimum output
    /// capacity. For tests: small values force the write pass to be rerun.
    #[doc(hidden)]
    pub fn set_out_hint(&mut self, per_block: (f64, f64), min_cap: u64) {
        *self.per_block.lock().unwrap() = per_block;
        self.min_cap = min_cap.max(1);
    }

    /// Maximum number of blocks accepted by one [`submit`](Self::submit).
    ///
    /// Bounded by the scratch budget for the per-cell buffers, the storage binding limit for the
    /// volumes, and the per-dimension workgroup limit for the indirect dispatch over tiles.
    pub fn max_blocks(&self, cfg: &Config) -> usize {
        let dim = cfg.block_dim as u64;
        let vol_blk = volume_words(cfg.block_dim) * 4;
        let maxd = self
            .ctx
            .device
            .limits()
            .max_compute_workgroups_per_dimension as u64;
        let k = (SCRATCH_BUDGET / ((dim + 1).pow(3) * 4))
            .min(self.ctx.max_binding / vol_blk - 1)
            .min((maxd * TC_CELLS - 1) / (dim + 1).pow(3))
            .max(1) as usize;
        if self.max_chunk > 0 {
            k.min(self.max_chunk)
        } else {
            k
        }
    }

    /// Extracts all blocks of `vol`, where `ids[j]` is the id of block `j`.
    ///
    /// `voxel` and `block_dim` come from `cfg`; `min_weight` has already been applied by the
    /// gather. Large inputs are split into batches of at most [`max_blocks`](Self::max_blocks);
    /// each batch is read back after the next one is submitted, one GPU round trip per batch.
    ///
    /// # Errors
    ///
    /// Returns an error from [`submit`](Self::submit) or [`finish`](Self::finish).
    pub fn extract(
        &self,
        vol: &Volumes,
        ids: &[BlockId],
        cfg: &Config,
    ) -> Result<Vec<ExtractedBlock>, String> {
        let k = self.max_blocks(cfg);
        let mut out = Vec::with_capacity(ids.len());
        let mut prev: Option<Pending> = None;
        for (ci, chunk) in ids.chunks(k).enumerate() {
            let p = self.submit_at(vol, ci * k, chunk, cfg)?;
            if let Some(q) = prev.replace(p) {
                out.extend(self.finish(q)?);
            }
        }
        if let Some(q) = prev {
            out.extend(self.finish(q)?);
        }
        Ok(out)
    }

    /// Submits one batch of blocks without waiting. Read the result with [`finish`](Self::finish).
    ///
    /// `vol` may be overwritten after this returns: later queue operations run after this batch
    /// has read it.
    ///
    /// # Errors
    ///
    /// Returns an error if `ids.len()` exceeds [`max_blocks`](Self::max_blocks), the volume
    /// buffer is too small, a binding exceeds device limits, or the pool page count does not
    /// match.
    pub fn submit(&self, vol: &Volumes, ids: &[BlockId], cfg: &Config) -> Result<Pending, String> {
        if ids.len() > self.max_blocks(cfg) {
            return Err(format!(
                "too many blocks ({} > {})",
                ids.len(),
                self.max_blocks(cfg)
            ));
        }
        self.submit_at(vol, 0, ids, cfg)
    }

    /// Output buffer capacity in vertices for `nb` blocks.
    ///
    /// Uses the largest per-block count seen so far times a margin. Before any batch has been
    /// seen it assumes one plane crossing each block (`(dim + 1)^2`); a smaller start would make
    /// the first batches rerun their write pass. Unified-memory devices get a larger margin since
    /// they have no staging copy to size.
    fn capv(&self, nb: u64, dim: u64) -> u64 {
        let vpb = self.per_block.lock().unwrap().0;
        let vpb = if vpb == 0.0 {
            ((dim + 1) * (dim + 1)) as f64
        } else {
            vpb
        };
        let margin = if self.ctx.uma { 1.5 } else { 1.25 };
        ((vpb * nb as f64 * margin) as u64).max(self.min_cap)
    }

    /// Sizes the scratch buffers for `nb` blocks and returns
    /// `[cells, vbits, bsl, sumq, sumv, starts, targs, tlist]`.
    fn ensure_scratch(&self, sc: &mut Scratch, nb: u64, dim: u64) -> [wgpu::Buffer; 8] {
        let dev = &self.ctx.device;
        let n_cell = nb * (dim + 1).pow(3);
        let ntiles = (n_cell + 1).div_ceil(TC_CELLS);
        let su_st = if self.ctx.uma {
            SU | wgpu::BufferUsages::MAP_READ
        } else {
            SU
        };
        [
            ensure(dev, &mut sc.cells, (n_cell + 1) * 4, SU, "cells"),
            ensure(dev, &mut sc.vbits, ntiles * TC_CELLS / 8, SU, "vbits"),
            ensure(dev, &mut sc.bsl, (nb + 1) * 4, SU, "bsl"),
            ensure(dev, &mut sc.sumq, ntiles * 4, SU, "sumq"),
            ensure(dev, &mut sc.sumv, ntiles * 4, SU, "sumv"),
            ensure(dev, &mut sc.starts, (nb + 1) * 2 * 4, su_st, "starts"),
            ensure(
                dev,
                &mut sc.targs,
                16,
                SU | wgpu::BufferUsages::INDIRECT,
                "tile args",
            ),
            ensure(dev, &mut sc.tlist, ntiles * 4, SU, "tile list"),
        ]
    }

    /// Preallocates and zero-fills `sets` scratch buffer sets for batches of `nblocks` blocks.
    ///
    /// Moves buffer creation, zero-filling and first-touch memory allocation out of the first
    /// extraction. Call once right after the device is opened. Staging buffers (`MAP_READ`) cannot
    /// be cleared, so they are only created.
    ///
    /// # Errors
    ///
    /// Returns an error if the estimated output exceeds the storage binding limit.
    pub fn prewarm(&self, cfg: &Config, nblocks: usize, sets: usize) -> Result<(), String> {
        let dev = &self.ctx.device;
        let (nb, dim) = (
            nblocks.clamp(1, self.max_blocks(cfg)) as u64,
            cfg.block_dim as u64,
        );
        let capv = self.capv(nb, dim);
        let (total, soff) = Self::out_layout(capv, nb);
        if capv * 20 > self.ctx.max_binding {
            return Err("extraction output exceeds the buffer binding limit".into());
        }
        let _g = self.ctx.lock.lock().unwrap();
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("extract prewarm"),
        });
        let mut pool = self.scratch.lock().unwrap();
        while pool.len() < sets {
            pool.push(Scratch::default());
        }
        for sc in pool.iter_mut() {
            let mut bufs = self.ensure_scratch(sc, nb, dim).to_vec();
            if self.ctx.uma {
                bufs.push(ensure(
                    dev,
                    &mut sc.out,
                    total,
                    SU | wgpu::BufferUsages::MAP_READ,
                    "extract out",
                ));
            } else {
                bufs.push(ensure(dev, &mut sc.out, total, SU, "extract out"));
                ensure(dev, &mut sc.staging, soff + total, RD, "extract staging");
            }
            for b in &bufs {
                enc.clear_buffer(b, 0, None);
            }
        }
        self.ctx.queue.submit(Some(enc.finish()));
        Ok(())
    }

    /// Encodes an in-place exclusive prefix sum of the u32 array `buf[0..n]` into `pass`.
    ///
    /// Tile sums are scanned recursively, one scratch level per recursion depth. Uniform and sum
    /// buffers are pushed to `keep` so they live until the submission completes.
    fn encode_scan(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        sc: &mut Scratch,
        level: usize,
        buf: &wgpu::Buffer,
        n: u64,
        keep: &mut Vec<wgpu::Buffer>,
    ) {
        let dev = &self.ctx.device;
        let maxd = dev.limits().max_compute_workgroups_per_dimension;
        let ntiles = n.div_ceil(TILE);
        if sc.sums.len() <= level {
            sc.sums.resize_with(level + 1, || None);
        }
        let sums = ensure(dev, &mut sc.sums[level], ntiles * 4, SU, "scan sums");
        let ub = uniform(dev, bytemuck::bytes_of(&[n as u32, 0, 0, 0]));
        let mk_bg = |pipe: &wgpu::ComputePipeline| {
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scan"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ub.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bind(buf, 0, n * 4),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bind(&sums, 0, ntiles * 4),
                    },
                ],
            })
        };
        let (gx, gy) = groups(ntiles, maxd);
        pass.set_pipeline(&self.pipes.scan_tile);
        pass.set_bind_group(0, &mk_bg(&self.pipes.scan_tile), &[]);
        pass.dispatch_workgroups(gx, gy, 1);
        if ntiles > 1 {
            self.encode_scan(pass, sc, level + 1, &sums, ntiles, keep);
            pass.set_pipeline(&self.pipes.scan_add);
            pass.set_bind_group(0, &mk_bg(&self.pipes.scan_add), &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        keep.push(ub);
        keep.push(sums);
    }

    /// Output buffer layout for `capv` vertices of 20 bytes and `nb` blocks.
    ///
    /// The staging buffer holds the start offsets followed by the output. Returns
    /// `(output bytes, output offset within staging)`, both 256-byte aligned.
    fn out_layout(capv: u64, nb: u64) -> (u64, u64) {
        let total = align256(capv * 20);
        let soff = align256((nb + 1) * 2 * 4);
        (total, soff)
    }

    /// Sizes the output buffer of batch `p` to `p.capv` and encodes `k_write` into `pass`.
    ///
    /// Returns the (output, staging) buffers. On unified-memory devices the output buffer is
    /// mapped directly and is returned for both.
    ///
    /// # Errors
    ///
    /// Returns an error if the output exceeds the storage binding limit or the pool page count
    /// differs from the one the pipelines were built for.
    fn encode_write(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        p: &mut Pending,
    ) -> Result<(wgpu::Buffer, wgpu::Buffer), String> {
        let dev = &self.ctx.device;
        let nb = p.ids.len() as u64;
        let (total, soff) = Self::out_layout(p.capv, nb);
        if p.capv * 20 > self.ctx.max_binding {
            return Err(format!(
                "extraction output exceeds the buffer binding limit ({} vertices)",
                p.capv
            ));
        }
        let sc = &mut p.sc;
        let (outb, stg) = if self.ctx.uma {
            let o = ensure(
                dev,
                &mut sc.out,
                total,
                SU | wgpu::BufferUsages::MAP_READ,
                "extract out",
            );
            (o.clone(), o)
        } else {
            (
                ensure(dev, &mut sc.out, total, SU, "extract out"),
                ensure(dev, &mut sc.staging, soff + total, RD, "extract staging"),
            )
        };
        let n_cell = nb * ((p.dim + 1) as u64).pow(3);
        let ntiles = p.ntiles;
        let params = Params {
            dim: p.dim as u32,
            nblk: nb as u32,
            vbase: p.vbase,
            capv: p.capv as u32,
            capq: 0,
            owp: occ_words(p.dim) as u32,
            stride: volume_words(p.dim) as u32,
            obase: p.obase,
            bsh: p.src.layout.bsh,
            msh: p.src.layout.msh,
            pad: [0; 2],
        };
        let ub = uniform(dev, bytemuck::bytes_of(&params));
        let get = |b: &Option<wgpu::Buffer>| b.clone().expect("scratch buffer not allocated");
        let (cells, vbits, sumv, targs, tlist) = (
            get(&sc.cells),
            get(&sc.vbits),
            get(&sc.sumv),
            get(&sc.targs),
            get(&sc.tlist),
        );
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ub.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bind(&p.src.vol, p.vbind.0, p.vbind.1),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: bind(&cells, 0, (n_cell + 1) * 4),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: bind(&outb, 0, p.capv * 20),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: bind(&sumv, 0, ntiles * 4),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: bind(&vbits, 0, ntiles * TC_CELLS / 8),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: bind(&tlist, 0, ntiles * 4),
            },
            wgpu::BindGroupEntry {
                binding: 16,
                resource: p.src.off.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 17,
                resource: p.src.slots.as_entire_binding(),
            },
        ];
        if p.src.layout.n != self.npg {
            return Err(format!(
                "pool page binding count mismatch ({} != {})",
                p.src.layout.n, self.npg
            ));
        }
        entries.extend(pool_entries(&p.src.pool, &self.dummy, self.npg));
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipes.write.get_bind_group_layout(0),
            entries: &entries,
        });
        pass.set_pipeline(&self.pipes.write);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups_indirect(&targs, 0);
        p.keep.push(ub);
        Ok((outb, stg))
    }

    /// Copies the start offsets and the whole output buffer into the staging buffer.
    ///
    /// Does nothing on unified-memory devices. Whether the output fit is checked by
    /// [`finish`](Self::finish).
    fn encode_copies(
        &self,
        enc: &mut wgpu::CommandEncoder,
        p: &Pending,
        outb: &wgpu::Buffer,
        stg: &wgpu::Buffer,
    ) {
        if self.ctx.uma {
            return;
        }
        let nb = p.ids.len() as u64;
        let (total, soff) = Self::out_layout(p.capv, nb);
        enc.copy_buffer_to_buffer(p.sc.starts.as_ref().unwrap(), 0, stg, 0, (nb + 1) * 2 * 4);
        enc.copy_buffer_to_buffer(outb, 0, stg, soff, total);
    }

    /// Encodes and submits all kernels for blocks `blk0 .. blk0 + ids.len()` of `src`.
    ///
    /// The volume binding starts at the aligned offset at or before block `blk0`; the shader
    /// starts reading at `vbase`. Each pipeline binds only the bindings it uses (automatic
    /// layout). Tile sums and block prefixes are cleared first, because `k_flags` does not run
    /// for tiles without vertex cells. Cell flags, tile prefixes, the scan of tile sums, block
    /// starts and the vertex write go into one compute pass, and the readback copy is submitted
    /// with it. The quad prefix sum (`sumq`) is not scanned: quads are built on the CPU, and the
    /// quad starts written by `k_starts` are not read.
    ///
    /// # Errors
    ///
    /// Returns an error if the volume buffer is too small or a binding exceeds device limits.
    fn submit_at(
        &self,
        src: &Volumes,
        blk0: usize,
        ids: &[BlockId],
        cfg: &Config,
    ) -> Result<Pending, String> {
        let vol = &src.vol;
        let dev = &self.ctx.device;
        let queue = &self.ctx.queue;
        let maxd = dev.limits().max_compute_workgroups_per_dimension;
        let nb = ids.len() as u64;
        let dim = cfg.block_dim as u64;
        let vol_blk = volume_words(cfg.block_dim) * 4;
        let end = (blk0 as u64 + nb) * vol_blk;
        if vol.size() < end {
            return Err(format!(
                "volume buffer too small ({} B < {end} B)",
                vol.size()
            ));
        }
        let align = dev.limits().min_storage_buffer_offset_alignment as u64;
        let voff = (blk0 as u64 * vol_blk) / align * align;
        let vbase = (blk0 as u64 * vol_blk - voff) / 4;
        if end - voff > self.ctx.max_binding || (end - voff) / 4 > u32::MAX as u64 {
            return Err(format!(
                "volume binding exceeds the limit ({} B)",
                end - voff
            ));
        }
        let n_cell = nb * (dim + 1).pow(3);
        let ntiles = (n_cell + 1).div_ceil(TC_CELLS);
        let capv = self.capv(nb, dim);
        let mut sc = self.scratch.lock().unwrap().pop().unwrap_or_default();
        let [cells, vbits, bsl, sumq, sumv, starts, targs, tlist] =
            self.ensure_scratch(&mut sc, nb, dim);
        let nst = (nb + 1) * 2;
        let params = Params {
            dim: dim as u32,
            nblk: nb as u32,
            vbase: vbase as u32,
            capv: 0,
            capq: 0,
            owp: occ_words(cfg.block_dim) as u32,
            stride: volume_words(cfg.block_dim) as u32,
            obase: (blk0 * ((cfg.block_dim / 8 + 2) as usize).pow(3)) as u32,
            bsh: src.layout.bsh,
            msh: src.layout.msh,
            pad: [0; 2],
        };
        let ub = uniform(dev, bytemuck::bytes_of(&params));
        let mut p = Pending {
            sc: Scratch::default(),
            ids: ids.to_vec(),
            src: src.clone(),
            obase: (blk0 * ((cfg.block_dim / 8 + 2) as usize).pow(3)) as u32,
            vbind: (voff, end - voff),
            vbase: vbase as u32,
            dim: cfg.block_dim,
            voxel: cfg.voxel,
            ntiles,
            capv,
            sub: None,
            keep: Vec::new(),
        };

        let res = |b: u32| -> wgpu::BindingResource<'_> {
            match b {
                0 => ub.as_entire_binding(),
                1 => bind(vol, voff, end - voff),
                3 => bind(&cells, 0, (n_cell + 1) * 4),
                8 => bind(&starts, 0, nst * 4),
                9 => bind(&sumq, 0, ntiles * 4),
                10 => bind(&sumv, 0, ntiles * 4),
                11 => bind(&vbits, 0, ntiles * TC_CELLS / 8),
                12 => bind(&bsl, 0, (nb + 1) * 4),
                13 => bind(&targs, 0, 16),
                14 => bind(&tlist, 0, ntiles * 4),
                _ => unreachable!(),
            }
        };
        let run = |pass: &mut wgpu::ComputePass<'_>,
                   pipe: &wgpu::ComputePipeline,
                   used: &[u32],
                   wgs: u64| {
            let entries: Vec<wgpu::BindGroupEntry> = used
                .iter()
                .map(|&b| wgpu::BindGroupEntry {
                    binding: b,
                    resource: res(b),
                })
                .collect();
            let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipe.get_bind_group_layout(0),
                entries: &entries,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bg, &[]);
            if wgs == 0 {
                pass.dispatch_workgroups_indirect(&targs, 0);
            } else {
                let (gx, gy) = groups(wgs, maxd);
                pass.dispatch_workgroups(gx, gy, 1);
            }
        };

        let _g = self.ctx.lock.lock().unwrap();
        queue.write_buffer(&targs, 0, bytemuck::cast_slice(&[0u32, 1, 1, 0]));
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("extract"),
        });
        enc.clear_buffer(&sumq, 0, Some(ntiles * 4));
        enc.clear_buffer(&sumv, 0, Some(ntiles * 4));
        enc.clear_buffer(&bsl, 0, Some((nb + 1) * 4));
        let (outb, stg) = {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("extract"),
                timestamp_writes: None,
            });
            run(
                &mut pass,
                &self.pipes.valid,
                &[0, 1, 11, 13, 14],
                (ntiles * TC_CELLS / 32).div_ceil(WG as u64),
            );
            run(
                &mut pass,
                &self.pipes.flags,
                &[0, 1, 3, 9, 10, 11, 12, 14],
                0,
            );
            self.encode_scan(&mut pass, &mut sc, 0, &sumv, ntiles, &mut p.keep);
            run(
                &mut pass,
                &self.pipes.starts,
                &[0, 8, 9, 10, 12],
                (nb + 1).div_ceil(WG as u64),
            );
            p.sc = sc;
            self.encode_write(&mut pass, &mut p)?
        };
        self.encode_copies(&mut enc, &p, &outb, &stg);
        p.keep.push(ub);
        p.sub = Some(queue.submit(Some(enc.finish())));
        Ok(p)
    }

    /// Waits for a submitted batch and unpacks its blocks.
    ///
    /// The start offsets are `[quad starts, nb + 1 (unused)][vertex starts, nb + 1]`. Positions
    /// are computed as `((block origin + local cell) as f32 + offset) * voxel`, the same
    /// expression as `build_block_in` in the CPU mesher. Seam vertices are those in cells with a
    /// local coordinate of `-1` or `dim - 1`.
    ///
    /// Triangles are built on the CPU from the quad bits: each block uses a per-thread table from
    /// cell to vertex index. The table is never cleared, because every ring cell of a quad has a
    /// vertex, so only entries written for the current block are read. Quads are emitted in cell
    /// order, then axis order, which matches the CPU mesher. The ring cells of a quad on axis `k`
    /// are `c, c - ei, c - ei - ej, c - ej`; the flip bit swaps the winding.
    ///
    /// The largest per-block vertex count seen is recorded to size later batches. If the output
    /// did not fit, the write pass alone is rerun once with a larger buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if readback fails, or if the output still does not fit after the rerun.
    pub fn finish(&self, mut p: Pending) -> Result<Vec<ExtractedBlock>, String> {
        let nb = p.ids.len();
        let mut blocks: Vec<ExtractedBlock> = p
            .ids
            .iter()
            .map(|&id| ExtractedBlock {
                id,
                ..Default::default()
            })
            .collect();
        let (dim, vx) = (p.dim, p.voxel);
        let mut st: Vec<u32> = Vec::new();
        for attempt in 0..2 {
            let (total, soff) = Self::out_layout(p.capv, nb as u64);
            let nst = (nb as u64 + 1) * 2 * 4;
            let (sb, ob) = if self.ctx.uma {
                (p.sc.starts.clone().unwrap(), p.sc.out.clone().unwrap())
            } else {
                (p.sc.staging.clone().unwrap(), p.sc.staging.clone().unwrap())
            };
            let maps: Vec<(&wgpu::Buffer, u64)> = if self.ctx.uma {
                vec![(&sb, nst), (&ob, total)]
            } else {
                vec![(&sb, soff + total)]
            };
            let fits = {
                map_many(
                    &self.ctx,
                    p.sub.take().expect("batch has no pending submission"),
                    &maps,
                    |views| {
                        st = bytemuck::cast_slice(&views[0][..nst as usize]).to_vec();
                        let vs = &st[nb + 1..];
                        let nv = vs[nb] as u64;
                        if nv > p.capv {
                            return false;
                        }
                        let res = if self.ctx.uma {
                            views[1]
                        } else {
                            &views[0][soff as usize..]
                        };
                        let verts: &[[u32; 5]] = bytemuck::cast_slice(&res[..(nv * 20) as usize]);
                        let pc = (dim + 1) as usize;
                        let step = [1usize, pc, pc * pc];
                        blocks.par_iter_mut().enumerate().for_each_init(
                            Vec::<u32>::new,
                            |map, (j, b)| {
                                let (v0, v1) = (vs[j] as usize, vs[j + 1] as usize);
                                let n = v1 - v0;
                                b.positions.reserve_exact(n * 3);
                                b.colors.reserve_exact(n * 3);
                                b.seam.reserve_exact(n);
                                if map.len() < pc * pc * pc {
                                    map.resize(pc * pc * pc, 0);
                                }
                                let g0 = [b.id[0] * dim, b.id[1] * dim, b.id[2] * dim];
                                let mut nq = 0usize;
                                for (k, v) in verts[v0..v1].iter().enumerate() {
                                    let cp = [
                                        (v[4] & 1023) as usize,
                                        ((v[4] >> 10) & 1023) as usize,
                                        ((v[4] >> 20) & 1023) as usize,
                                    ];
                                    let c = [cp[0] as i32 - 1, cp[1] as i32 - 1, cp[2] as i32 - 1];
                                    for a in 0..3 {
                                        b.positions.push(
                                            ((g0[a] + c[a]) as f32 + f32::from_bits(v[a])) * vx,
                                        );
                                    }
                                    b.colors.extend_from_slice(&[
                                        v[3] as u8,
                                        (v[3] >> 8) as u8,
                                        (v[3] >> 16) as u8,
                                    ]);
                                    b.seam.push(c.iter().any(|&x| x == -1 || x == dim - 1));
                                    map[(cp[2] * pc + cp[1]) * pc + cp[0]] = k as u32;
                                    nq += ((v[3] >> 24) & 7).count_ones() as usize;
                                }
                                let mut idx = Vec::with_capacity(nq * 6);
                                for v in &verts[v0..v1] {
                                    let qb = (v[3] >> 24) & 7;
                                    if qb == 0 {
                                        continue;
                                    }
                                    let r = (((v[4] >> 20) & 1023) as usize * pc
                                        + ((v[4] >> 10) & 1023) as usize)
                                        * pc
                                        + (v[4] & 1023) as usize;
                                    for k in 0..3 {
                                        if (qb >> k) & 1 == 0 {
                                            continue;
                                        }
                                        let (si, sj) = (step[(k + 1) % 3], step[(k + 2) % 3]);
                                        let a = map[r];
                                        let mut bb = map[r - si];
                                        let c2 = map[r - si - sj];
                                        let mut e = map[r - sj];
                                        if (v[3] >> 27) & 1 != 0 {
                                            std::mem::swap(&mut bb, &mut e);
                                        }
                                        idx.extend_from_slice(&[a, bb, c2, a, c2, e]);
                                    }
                                }
                                b.indices = idx;
                            },
                        );
                        true
                    },
                )?
            };
            let nv = st[2 * nb + 1] as u64;
            {
                let mut pb = self.per_block.lock().unwrap();
                pb.0 = pb.0.max(nv as f64 / nb as f64);
            }
            if fits {
                break;
            }
            if attempt == 1 {
                return Err("extraction output did not fit after resizing".into());
            }
            p.capv = p.capv.max(nv + nv / 4);
            let _g = self.ctx.lock.lock().unwrap();
            let mut enc = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("extract write"),
                });
            let (outb, stg) = {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("extract write"),
                    timestamp_writes: None,
                });
                self.encode_write(&mut pass, &mut p)?
            };
            self.encode_copies(&mut enc, &p, &outb, &stg);
            p.sub = Some(self.ctx.queue.submit(Some(enc.finish())));
        }
        self.scratch.lock().unwrap().push(std::mem::take(&mut p.sc));
        Ok(blocks)
    }
}

/// Creates a uniform buffer holding `data`, zero-padded to a multiple of 16 bytes (at least 16).
fn uniform(dev: &wgpu::Device, data: &[u8]) -> wgpu::Buffer {
    let b = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: data.len().max(16).next_multiple_of(16) as u64,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: true,
    });
    let mut d = vec![0u8; data.len().max(16).next_multiple_of(16)];
    d[..data.len()].copy_from_slice(data);
    b.slice(..)
        .get_mapped_range_mut()
        .expect("failed to map uniform buffer")
        .copy_from_slice(&d);
    b.unmap();
    b
}

#[cfg(test)]
#[allow(clippy::type_complexity)]
mod tests {
    use super::*;
    use crate::sdf::SdfMesher;
    use crate::testutil::plane;
    use crate::{Level, Mesher, Point};
    use rustc_hash::FxHashMap;
    use std::sync::OnceLock;

    fn ctx() -> Option<Arc<GpuCtx>> {
        static C: OnceLock<Option<Arc<GpuCtx>>> = OnceLock::new();
        C.get_or_init(|| GpuCtx::shared().ok()).clone()
    }

    /// Xorshift generator for reproducible test data.
    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// Returns the blocks with a mesh, their neighbours (including empty ones) and one distant
    /// empty block, sorted and deduplicated.
    fn test_ids(m: &SdfMesher) -> Vec<BlockId> {
        let mut v: Vec<BlockId> = Vec::new();
        for id in m.block_ids() {
            for d in [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]] {
                v.push([id[0] + d[0], id[1] + d[1], id[2] + d[2]]);
            }
        }
        v.push([1000, -1000, 7]);
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Statistics accumulated while comparing GPU blocks with CPU blocks.
    #[derive(Default, Debug)]
    struct Cmp {
        tris: usize,
        max_err: f32,
        col_diff: usize,
        max_col: i32,
        /// GPU vertices that matched one of several CPU vertices (from different cells) lying
        /// within 1e-5 of each other.
        coincident: usize,
    }

    /// CPU position bits, GPU position bits, GPU colour, seam flag of one matched vertex.
    type Pair = ([u32; 3], [u32; 3], [u8; 3], bool);

    /// Matches one GPU block against the CPU result.
    ///
    /// Vertices are matched by position (within 1e-5 m) and triangles are compared via the three
    /// matched CPU positions. Positions are used instead of indices because different cells can
    /// produce vertices at the same position. Among candidates the match prefers, in order: not
    /// yet taken, same seam flag, smaller colour difference, smaller distance. Triangles are
    /// compared after rotating each so that its smallest position comes first, which keeps the
    /// winding.
    ///
    /// Returns one [`Pair`] per GPU vertex.
    fn compare(
        cpu: &(Vec<f32>, Vec<u8>, Vec<u32>, Vec<bool>),
        g: &ExtractedBlock,
        st: &mut Cmp,
    ) -> Vec<Pair> {
        let (cp, cc, ct, cs) = cpu;
        assert_eq!(
            ct.len(),
            g.indices.len(),
            "block {:?}: triangle count",
            g.id
        );
        if ct.is_empty() {
            assert!(
                g.positions.is_empty(),
                "empty block {:?} has vertices",
                g.id
            );
            return Vec::new();
        }
        const Q: f32 = 1e-4;
        let key = |p: &[f32]| {
            [
                (p[0] / Q).floor() as i64,
                (p[1] / Q).floor() as i64,
                (p[2] / Q).floor() as i64,
            ]
        };
        let mut grid: FxHashMap<[i64; 3], Vec<u32>> = FxHashMap::default();
        let mut referenced = vec![false; cp.len() / 3];
        for &i in ct {
            referenced[i as usize] = true;
        }
        for (i, p) in cp.chunks(3).enumerate() {
            if referenced[i] {
                grid.entry(key(p)).or_default().push(i as u32);
            }
        }
        let ng = g.positions.len() / 3;
        assert_eq!(
            ng,
            referenced.iter().filter(|&&r| r).count(),
            "block {:?}: vertex count",
            g.id
        );
        let bits = |v: &[f32]| [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()];
        let mut map = vec![[0u32; 3]; ng];
        let mut taken = vec![false; cp.len() / 3];
        let mut pairs = Vec::new();
        for (gi, p) in g.positions.chunks(3).enumerate() {
            let k = key(p);
            let gc = &g.colors[gi * 3..gi * 3 + 3];
            let mut best: Option<((bool, bool, i32, f32), u32, f32)> = None;
            let mut ncand = 0;
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let Some(c) = grid.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) else {
                            continue;
                        };
                        for &ci in c {
                            let ci_u = ci as usize;
                            let q = &cp[ci_u * 3..ci_u * 3 + 3];
                            let e = (0..3).map(|a| (q[a] - p[a]).abs()).fold(0.0f32, f32::max);
                            if e > 1e-5 {
                                continue;
                            }
                            ncand += 1;
                            let cd = (0..3)
                                .map(|a| (cc[ci_u * 3 + a] as i32 - gc[a] as i32).abs())
                                .max()
                                .unwrap();
                            let score = (taken[ci_u], cs[ci_u] != g.seam[gi], cd, e);
                            if best.as_ref().is_none_or(|b| {
                                score.partial_cmp(&b.0) == Some(std::cmp::Ordering::Less)
                            }) {
                                best = Some((score, ci, e));
                            }
                        }
                    }
                }
            }
            let Some((_, ci, e)) = best else {
                panic!(
                    "block {:?}: no CPU vertex within 1e-5 m of GPU vertex {gi} {p:?}",
                    g.id
                )
            };
            let ci = ci as usize;
            st.coincident += (ncand > 1) as usize;
            taken[ci] = true;
            map[gi] = bits(&cp[ci * 3..ci * 3 + 3]);
            st.max_err = st.max_err.max(e);
            for a in 0..3 {
                let d = (cc[ci * 3 + a] as i32 - gc[a] as i32).abs();
                assert!(d <= 1, "block {:?}: colour difference {d}", g.id);
                st.max_col = st.max_col.max(d);
                st.col_diff += (d > 0) as usize;
            }
            assert_eq!(cs[ci], g.seam[gi], "block {:?}: seam flag", g.id);
            pairs.push((map[gi], bits(p), [gc[0], gc[1], gc[2]], g.seam[gi]));
        }
        let canon = |t: [[u32; 3]; 3]| {
            let r = (0..3).min_by_key(|&i| t[i]).unwrap();
            [t[r], t[(r + 1) % 3], t[(r + 2) % 3]]
        };
        let cpos = |i: u32| bits(&cp[i as usize * 3..i as usize * 3 + 3]);
        let mut a: Vec<_> = ct
            .chunks(3)
            .map(|t| canon([cpos(t[0]), cpos(t[1]), cpos(t[2])]))
            .collect();
        let mut b: Vec<_> = g
            .indices
            .chunks(3)
            .map(|t| canon([map[t[0] as usize], map[t[1] as usize], map[t[2] as usize]]))
            .collect();
        a.sort_unstable();
        b.sort_unstable();
        assert!(a == b, "block {:?}: triangle sets differ", g.id);
        st.tris += a.len();
        pairs
    }

    /// Extracts the test blocks of `m` on the GPU and compares them with the CPU mesher.
    ///
    /// Checks every block with [`compare`], requires that seam vertices shared by neighbouring
    /// blocks (same CPU position and colour, i.e. the same cell) are bitwise identical on the GPU
    /// regardless of block, that the scene has empty blocks and surfaces crossing block
    /// boundaries, and that a second run is bitwise identical.
    fn check_scene(name: &str, m: &SdfMesher) {
        let Some(c) = ctx() else { return };
        let ids = test_ids(m);
        let vols: Vec<Vec<[f32; 5]>> = ids.iter().map(|&id| m.padded_volume(id)).collect();
        let vol = upload_volumes(&c, &vols, m.config().block_dim, m.config().min_weight);
        let ex = GpuExtractor::new(c.clone()).unwrap();
        let out = ex.extract(&vol, &ids, m.config()).unwrap();
        assert_eq!(out.len(), ids.len());
        let mut st = Cmp::default();
        let mut shared: FxHashMap<([u32; 3], [u8; 3]), (BlockId, [u32; 3], [u8; 3])> =
            FxHashMap::default();
        let mut multi = 0;
        let mut empty = 0;
        for (id, g) in ids.iter().zip(&out) {
            assert_eq!(*id, g.id);
            let cpu = m.build_block_public(*id);
            empty += cpu.2.is_empty() as usize;
            let cmap: FxHashMap<[u32; 3], [u8; 3]> = cpu
                .0
                .chunks(3)
                .zip(cpu.1.chunks(3))
                .map(|(p, c)| {
                    (
                        [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()],
                        [c[0], c[1], c[2]],
                    )
                })
                .collect();
            for (cb, gb, col, seam) in compare(&cpu, g, &mut st) {
                if !seam {
                    continue;
                }
                let e = shared.entry((cb, cmap[&cb])).or_insert((*id, gb, col));
                if e.0 != *id {
                    assert_eq!(
                        (e.1, e.2),
                        (gb, col),
                        "seam vertex differs between neighbouring blocks {:?} {:?}",
                        e.0,
                        id
                    );
                    multi += 1;
                }
            }
        }
        assert!(st.tris > 0);
        assert!(empty > 0, "scene has no empty block");
        assert!(multi > 0, "no surface crosses a block boundary");
        assert_eq!(out, ex.extract(&vol, &ids, m.config()).unwrap());
        eprintln!(
            "{name}: {} blocks ({empty} empty), {} triangles, max position error {:.2e} m, {} differing colour components (max {}), {} coincident vertices, {multi} shared seam checks",
            ids.len(),
            st.tris,
            st.max_err,
            st.col_diff,
            st.max_col,
            st.coincident
        );
    }

    /// A horizontal plane plus a tilted wall near the block boundary at x = 6.4 (normal towards
    /// +x).
    #[test]
    fn planes_across_blocks_match_cpu() {
        let mut m = SdfMesher::new(Config::default());
        m.ingest(
            0,
            Level::Refined,
            &plane(2.0, 11.0, -4.0, 9.0, 0.03, 0.05, [200, 100, 50]),
        )
        .unwrap();
        let mut wall = Vec::new();
        let n = [0.96f32, 0.0, 0.28];
        for i in 0..120 {
            for k in 0..60 {
                let (y, z) = (-3.0 + i as f32 * 0.07, -1.0 + k as f32 * 0.07);
                wall.push(Point {
                    pos: [6.3 - 0.29 * z, y, z],
                    rgb: [(i * 2) as u8, 50, (k * 4) as u8],
                    normal: n,
                });
            }
        }
        m.ingest(1, Level::Refined, &wall).unwrap();
        m.extract();
        check_scene("planes", &m);
    }

    /// A sphere centred on the block corner (6.4, 6.4, 6.4), so it passes through all diagonal
    /// neighbours of that corner.
    #[test]
    fn sphere_at_block_corner_matches_cpu() {
        let mut pts = Vec::new();
        let c = [6.4f32, 6.4, 6.4];
        let r = 2.3f32;
        for i in 0..160 {
            let th = std::f32::consts::PI * (i as f32 + 0.5) / 160.0;
            for j in 0..320 {
                let ph = 2.0 * std::f32::consts::PI * j as f32 / 320.0;
                let n = [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()];
                let p = [c[0] + r * n[0], c[1] + r * n[1], c[2] + r * n[2]];
                pts.push(Point {
                    pos: p,
                    rgb: [(i % 256) as u8, (j % 256) as u8, 128],
                    normal: n,
                });
            }
        }
        let mut m = SdfMesher::new(Config::default());
        m.ingest(0, Level::Refined, &pts).unwrap();
        m.extract();
        check_scene("sphere", &m);
    }

    /// A noisy wave (refined) plus a preview layer without normals.
    ///
    /// Also runs with `block_dim = 16` and voxel 0.1, and with `block_dim = 8`, where a block has
    /// fewer cells (9^3) than a tile (2048), so several block starts fall into one tile.
    #[test]
    fn noisy_points_with_preview_match_cpu() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let mut refined = Vec::new();
        for _ in 0..60000 {
            let (x, y) = (-2.0 + rng.f() * 7.0, -2.0 + rng.f() * 5.0);
            let z = 0.4 * (x * 1.3).sin() * (y * 0.9).cos() + (rng.f() - 0.5) * 0.04;
            let n = [
                -0.52 * (x * 1.3).cos() * (y * 0.9).cos(),
                0.36 * (x * 1.3).sin() * (y * 0.9).sin(),
                1.0,
            ];
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            let rgb = [
                (rng.f() * 255.0) as u8,
                (rng.f() * 255.0) as u8,
                (rng.f() * 255.0) as u8,
            ];
            refined.push(Point {
                pos: [x, y, z],
                rgb,
                normal: [n[0] / l + (rng.f() - 0.5) * 0.1, n[1] / l, n[2] / l],
            });
        }
        let mut preview = Vec::new();
        for _ in 0..20000 {
            let (x, y) = (1.0 + rng.f() * 4.0, -1.0 + rng.f() * 3.0);
            preview.push(Point {
                pos: [x, y, 1.0 + (rng.f() - 0.5) * 0.05],
                rgb: [30, 200, 30],
                normal: [f32::NAN; 3],
            });
        }
        for cfg in [
            Config::default(),
            Config {
                voxel: 0.1,
                block_dim: 16,
                bin: 0.05,
                splat_radius: 0.2,
                ..Config::default()
            },
            Config {
                block_dim: 8,
                ..Config::default()
            },
        ] {
            let mut m = SdfMesher::new(cfg);
            m.ingest(0, Level::Refined, &refined).unwrap();
            m.ingest(1, Level::Preview, &preview).unwrap();
            m.extract();
            check_scene(&format!("noisy dim {}", cfg.block_dim), &m);
        }
    }

    /// The resident field path (gather writes distances only; colours are read from the pool per
    /// vertex) produces the same mesh as uploading the same GPU sums through the CPU.
    ///
    /// Several layers (refined plus two previews) overlap, which covers layer-order summation.
    /// Colours come from the same sums through the same expression and must match exactly. Two
    /// runs on the same device are bitwise identical.
    #[test]
    fn resident_field_path_matches_uploaded_sums() {
        let Some(c) = ctx() else { return };
        use crate::gpu_field::{GpuField, LayerKey};
        let cfg = Config::default();
        let mut m = SdfMesher::new(cfg);
        let mut f = GpuField::new(c.clone(), &cfg).unwrap();
        let mut pre = plane(-1.0, 6.0, -2.0, 5.0, 0.2, 0.08, [10, 220, 30]);
        for p in &mut pre {
            p.normal = [f32::NAN; 3];
        }
        let steps: Vec<(u32, Level, Vec<Point>)> = vec![
            (
                0,
                Level::Refined,
                plane(-3.0, 9.0, -3.0, 7.0, 0.03, 0.05, [200, 100, 50]),
            ),
            (1, Level::Preview, pre),
            (
                2,
                Level::Preview,
                plane(4.0, 12.0, 1.0, 8.0, 0.1, 0.07, [5, 60, 250]),
            ),
        ];
        for (seg, lv, pts) in &steps {
            let bins = m.prepare_bins(*lv, pts);
            let lists = SdfMesher::block_lists(&cfg, &bins);
            let key = if *lv == Level::Refined {
                LayerKey::Base
            } else {
                LayerKey::Pending(*seg)
            };
            f.integrate(key, &bins, &lists).unwrap();
            m.ingest(*seg, *lv, pts).unwrap();
        }
        m.extract();
        let ids = test_ids(&m);
        let p3 = ((cfg.block_dim + 2) as usize).pow(3);
        let raw: Vec<f32> = bytemuck::cast_slice(
            &c.read_buffer(&f.gather_raw(&ids).unwrap(), (ids.len() * p3 * 32) as u64)
                .unwrap(),
        )
        .to_vec();
        let vols: Vec<Vec<[f32; 5]>> = (0..ids.len())
            .map(|j| {
                (0..p3)
                    .map(|v| {
                        let r = &raw[(j * p3 + v) * 8..];
                        [r[0], r[1], r[2], r[3], r[4]]
                    })
                    .collect()
            })
            .collect();
        let ex = GpuExtractor::new(c.clone()).unwrap();
        let up = ex
            .extract(
                &upload_volumes(&c, &vols, cfg.block_dim, cfg.min_weight),
                &ids,
                &cfg,
            )
            .unwrap();
        let fv = f.gather(&ids).unwrap();
        let got = ex.extract(&fv, &ids, &cfg).unwrap();
        assert_eq!(
            got,
            ex.extract(&fv, &ids, &cfg).unwrap(),
            "same input produced different output"
        );
        let mut st = Cmp::default();
        for (a, b) in up.iter().zip(&got) {
            compare(
                &(
                    a.positions.clone(),
                    a.colors.clone(),
                    a.indices.clone(),
                    a.seam.clone(),
                ),
                b,
                &mut st,
            );
            assert_eq!(a.colors.len(), b.colors.len());
        }
        assert!(st.tris > 5000, "triangles {}", st.tris);
        assert_eq!(st.max_col, 0, "colours differ");
        eprintln!(
            "resident path: {} triangles, max position error {:.2e} m",
            st.tris, st.max_err
        );
    }

    /// Enough blocks to need several prefix-sum levels, with each block repeated so the results
    /// can be compared. Also checks small batches (7 blocks: offset volume bindings, overlapped
    /// batches) and undersized output buffers (write pass rerun).
    #[test]
    fn many_blocks_use_several_chunks_and_scan_levels() {
        let Some(c) = ctx() else { return };
        let mut m = SdfMesher::new(Config::default());
        m.ingest(
            0,
            Level::Refined,
            &plane(-7.0, 7.0, -7.0, 7.0, 0.03, 0.05, [9, 90, 9]),
        )
        .unwrap();
        m.extract();
        let base = m.block_ids();
        let ids: Vec<BlockId> = (0..40).flat_map(|_| base.iter().copied()).collect();
        let vols: Vec<Vec<[f32; 5]>> = ids.iter().map(|&id| m.padded_volume(id)).collect();
        let vol = upload_volumes(&c, &vols, m.config().block_dim, m.config().min_weight);
        let out = GpuExtractor::new(c.clone())
            .unwrap()
            .extract(&vol, &ids, m.config())
            .unwrap();
        let mut ex = GpuExtractor::new(c.clone()).unwrap();
        ex.set_max_chunk(7);
        assert_eq!(out, ex.extract(&vol, &ids, m.config()).unwrap());
        let mut ex = GpuExtractor::new(c.clone()).unwrap();
        ex.set_max_chunk(7);
        ex.set_out_hint((0.0, 0.0), 1);
        assert_eq!(out, ex.extract(&vol, &ids, m.config()).unwrap());
        let mut ex = GpuExtractor::new(c.clone()).unwrap();
        ex.set_out_hint((1.0, 1.0), 1);
        assert_eq!(out, ex.extract(&vol, &ids, m.config()).unwrap());
        assert!(ids.len() > 7 * 3, "{}", ids.len());
        for (j, g) in out.iter().enumerate() {
            let first = &out[j % base.len()];
            assert_eq!(g.positions, first.positions);
            assert_eq!(g.indices, first.indices);
            assert_eq!(g.indices.len(), m.build_block_public(g.id).2.len());
        }
    }

    /// Real captured data: a refined segment and a preview segment.
    ///
    /// Reads `r0_refined.ply` and `r1_preview_new.ply` (binary PLY, 27-byte records) from the
    /// directory in `DELTAMESH_DATA` and returns early when the variable is unset. Ignored by
    /// default because it is slow; run with `cargo test --features gpu -- --ignored`.
    #[test]
    #[ignore]
    fn real_data_matches_cpu() {
        let Some(dir) = std::env::var_os("DELTAMESH_DATA").map(std::path::PathBuf::from) else {
            return;
        };
        let read = |name: &str| -> Vec<Point> {
            let b = std::fs::read(dir.join(name)).unwrap();
            let end = b.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
            let f = |r: &[u8], o: usize| f32::from_le_bytes(r[o..o + 4].try_into().unwrap());
            b[end..]
                .as_chunks::<27>()
                .0
                .iter()
                .map(|r| Point {
                    pos: [f(r, 0), f(r, 4), f(r, 8)],
                    rgb: [r[12], r[13], r[14]],
                    normal: [f(r, 15), f(r, 19), f(r, 23)],
                })
                .collect()
        };
        for cfg in [
            Config::default(),
            Config {
                voxel: 0.1,
                bin: 0.05,
                splat_radius: 0.15,
                ..Config::default()
            },
        ] {
            let mut m = SdfMesher::new(cfg);
            m.ingest(0, Level::Refined, &read("r0_refined.ply"))
                .unwrap();
            m.ingest(1, Level::Preview, &read("r1_preview_new.ply"))
                .unwrap();
            m.extract();
            check_scene(&format!("real data voxel {}", cfg.voxel), &m);
        }
    }
}
