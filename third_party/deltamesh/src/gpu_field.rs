//! GPU-resident signed-distance field stored in a paged brick pool.
//!
//! # Storage
//!
//! The field lives in a GPU brick pool. A brick is 8³ voxels with five components
//! (Σw·d, Σw, Σw·r, Σw·g, Σw·b) stored component by component (SoA), 10,240 bytes per brick,
//! the same size as the CPU accumulator with no padding. The pool is split into several storage
//! buffers ("pages", see [`PageLayout`]). Slot `s` maps to (page `p`, local slot `l`), and component
//! `c` of voxel `v` lives at f32 index `(l·5 + c)·512 + v` of page `p`. Shaders bind every page at
//! `POOL_BIND..` and use the accessors from [`pool_wgsl`], which read or write all needed components
//! of one slot behind a single page `switch`.
//!
//! # Accumulation
//!
//! The work unit ("tile") is a 4³ sub-brick. One workgroup of 64 threads handles one tile, one
//! thread per voxel. The CPU builds bin lists per 8³ brick; a tile reads its brick's list 64 entries
//! at a time, rejects bins whose radius does not reach the sub-brick box, and compacts the survivors
//! into workgroup memory in list order. Every voxel therefore adds the same bins in the same order as
//! a per-sub-brick list would, while the CPU only has to build roughly half as many list entries.
//!
//! The lists are built without per-block lists: chunks of bins are processed in parallel, each
//! counting (block, brick) hits and recording the emitted cells; after per-block start offsets are
//! computed, a second pass fills the index list in the recorded order (a two-pass counting sort).
//! Chunk order equals bin order, so each list stays in bin order.
//!
//! Large inputs are split by bin index into a few batches. While the GPU accumulates batch `k`, the
//! CPU builds the lists for batch `k + 1`. Each voxel adds the batches in order, so the result is
//! bitwise identical to submitting everything at once. Upload buffers (a `MAP_WRITE` staging buffer
//! plus a GPU copy) are kept per batch and reused, because wgpu zero-fills every new buffer.
//! Mapped memory is write-combined, so scattered lists are filled in ordinary memory first and then
//! copied sequentially.
//!
//! # Slot management
//!
//! The CPU keeps, per layer (base and each preview segment), a map from block to its `nb³` brick
//! slots and its accumulated changed range. Slots of a dropped layer go back to a free list. Newly
//! assigned slots are not cleared: the accumulation kernel sees the "fresh" flag, starts from zero
//! and writes the whole brick. When the pool is full only a new page is added; existing pages are
//! never moved or copied. The zero-fill and allocation cost of a new page is paid ahead of time while
//! the GPU is idle ([`GpuField::prefetch`], or at the start of an integrate call) or explicitly via
//! [`GpuField::reserve`].
//!
//! Brick allocation matches the CPU mesher: a fresh brick that received no voxel with w > 0 during
//! the call is released as soon as its range mask is read back. Accumulated values are never read
//! back; the only download is one u32 per tile holding eight bits per axis for the changed in-brick
//! coordinates. A bitwise OR (`atomicOr`) inside a workgroup is order independent, so it does not
//! break the determinism rule (no additive atomics). The compaction step computes each thread's
//! output index by OR-ing keep bits and counting the lower set bits, which is deterministic as well.
//! If a call fails before its first batch is submitted (buffer limits, coordinate range), the slots
//! taken during the call are returned and the state is unchanged.
//!
//! # Bitwise agreement with the CPU
//!
//! The "changed voxel" test (q ≤ r²) must match the CPU bit for bit. Grid point coordinates (g·voxel)
//! are read from a table computed on the CPU, and per-bin voxel ranges
//! (`ceil((p − r)/v)..floor((p + r)/v)`) are computed on the CPU and uploaded. Squares and sums are
//! wrapped in `op()` (XOR with a runtime zero) so the compiler cannot contract them into FMAs or
//! reassociate them. Distances and weights are compared with a 1e-4 tolerance and are not wrapped.
//!
//! # Base agreement and gather
//!
//! [`GpuField::base_agreement`] uses a hash table from base brick coordinate to slot (open
//! addressing, linear probing) built on the CPU. Base bricks are only ever added, so after the table
//! is built only new bricks are inserted; it is rebuilt once it would become more than half full.
//!
//! [`GpuField::gather`] produces the extraction input. The CPU builds a slot table per (block,
//! region), where regions split the padded volume along brick boundaries. Only regions that contain
//! bricks get a workgroup, which writes the layer-summed distance (Σw·d / Σw, or `MARK` when
//! unobserved). Each block starts with a header of region occupancy bits. Colour is not written;
//! extraction reads only the cells it needs for vertices from the same slot table and pool
//! ([`Volumes`]). [`GpuField::gather_raw`] writes the raw sums (eight f32 per voxel) for comparison
//! against the CPU padded volume in tests.

use crate::bins::Bin;
use crate::gpu::GpuCtx;
pub use crate::gpu_extract::Volumes;
use crate::{BlockId, Config, SegmentId};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use wgpu::util::DeviceExt;

/// Brick edge length in voxels.
const BR: i32 = 8;
/// Edge length of an accumulation tile (sub-brick); one 64-thread workgroup covers 4³ voxels.
const SB: i32 = 4;
/// Voxels per brick.
const BR3: u32 = 512;
/// Components stored per voxel (Σw·d, Σw, Σw·r, Σw·g, Σw·b).
const COMP: u32 = 5;
/// Size of one brick in bytes (512 voxels × 5 components × 4 bytes).
pub const BRICK_BYTES: u64 = (BR3 * COMP * 4) as u64;
/// Marker for "no slot".
const NONE: u32 = u32::MAX;
/// Default size of the first pool page in bricks (a power of two, about 40 MB).
const INIT_CAP: u32 = 4096;
/// Maximum number of bricks in a fixed-size page (a power of two, about 335 MB).
///
/// The time to allocate and zero-fill a new page grows with this size.
const PAGE_MAX: u32 = 32768;
/// Binding index of pool page 0; page `k` is bound at `POOL_BIND + k`.
pub const POOL_BIND: u32 = 32;
/// Upper bound on the number of pool pages.
const MAX_PAGES: u32 = 16;
/// Largest number of non-pool storage bindings in any shader that reads the pool.
///
/// The extraction write kernel uses 15; one more is kept in reserve.
const OTHER_STORAGE: u32 = 16;
/// Maximum workgroup count along one dispatch axis (wgpu default limit).
const MAX_WG: u32 = 65535;
/// Minimum number of bins per batch when `integrate_bins` splits its input.
const BATCH_MIN: usize = 120_000;
/// Maximum number of batches per integrate call.
const MAX_BATCH: usize = 4;
/// Value written by the agreement kernel when a sample is unobserved.
const AGREE_NONE: f32 = 9.0;

/// Page layout of the brick pool.
///
/// The pool is split into fixed-size storage buffers ("pages"). Growing the pool only allocates new
/// pages; nothing is copied. Page 0 holds `B` slots, page `k` for `1..=g` holds `B·2^(k-1)` slots,
/// and every later page holds `M = B·2^g` slots, where `B = 2^bsh` and `M = 2^msh`. The page of a
/// slot can therefore be found with a shift and a leading-bit count (shader function `pool_page`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageLayout {
    /// log2 of the first page size `B`.
    pub bsh: u32,
    /// log2 of the fixed page size `M`.
    pub msh: u32,
    /// Number of page bindings a shader declares, derived from the device limits.
    pub n: u32,
}

impl PageLayout {
    /// Creates a layout for `n` page bindings.
    ///
    /// # Arguments
    ///
    /// * `first` - Requested size of the first page in slots, rounded up to a power of two.
    /// * `n` - Number of page bindings available to a shader.
    /// * `max_bytes` - Largest allowed size of a single page in bytes.
    ///
    /// The fixed page size `M` is reduced when needed so that the geometric pages before it do not
    /// use up all bindings (small first pages are only used in tests).
    pub fn new(first: u32, n: u32, max_bytes: u64) -> Self {
        let mut m = PAGE_MAX;
        while m > 1 && m as u64 * BRICK_BYTES > max_bytes {
            m /= 2;
        }
        let b = first.max(1).next_power_of_two().min(m);
        let m = ((m as u64).min((b as u64) << n.saturating_sub(2).min(20)) as u32).max(b);
        Self {
            bsh: b.trailing_zeros(),
            msh: m.trailing_zeros(),
            n,
        }
    }

    /// Returns the number of slots in page `k`.
    pub fn page_slots(&self, k: u32) -> u32 {
        let g = self.msh - self.bsh;
        if k == 0 {
            1 << self.bsh
        } else if k <= g {
            1 << (self.bsh + k - 1)
        } else {
            1 << self.msh
        }
    }

    /// Returns `(page, slot within page)` for pool slot `s`.
    ///
    /// Uses the same formula as the shader function `pool_page`.
    pub fn locate(&self, s: u32) -> (u32, u32) {
        let m = 1u32 << self.msh;
        if s >= m {
            return (self.msh - self.bsh + (s >> self.msh), s & (m - 1));
        }
        let q = s >> self.bsh;
        if q == 0 {
            return (0, s);
        }
        let k = 32 - q.leading_zeros();
        (k, s - (1 << (k - 1 + self.bsh)))
    }

    /// Returns the total number of slots once all `n` pages are allocated.
    pub fn max_slots(&self) -> u64 {
        (0..self.n).map(|k| self.page_slots(k) as u64).sum()
    }
}

/// Returns how many pool pages a single shader can bind on this device.
///
/// The count is the per-stage storage buffer limit minus the other storage bindings of the
/// pool-reading shaders, capped at `MAX_PAGES`.
pub fn pool_pages(ctx: &GpuCtx) -> u32 {
    ctx.device
        .limits()
        .max_storage_buffers_per_shader_stage
        .saturating_sub(OTHER_STORAGE)
        .min(MAX_PAGES)
}

/// Generates WGSL declarations for the pool page bindings and their accessor functions.
///
/// Page `k` is declared at binding `POOL_BIND + k`. The generated code provides:
///
/// * `pool_page(s, bsh, msh)` - `(page, slot within page)` for slot `s`.
/// * `pool_dw(pg, o)` - `(Σw·d, Σw)`, or zero when Σw = 0 (the distance is then not read).
/// * `pool_wc(pg, o)` - `(Σw, Σw·r, Σw·g, Σw·b)`, or zero when Σw = 0.
/// * `pool_all(pg, o)` - all five components as `P5 { a: (Σw·d, Σw, Σw·r, Σw·g), b: Σw·b }`,
///   or zero when Σw = 0.
/// * `pool_put(pg, o, a, b)` - writes all five components (only when `rw` is true).
///
/// `o` is the voxel index within the brick; component `c` is at `o + c·512`. Each accessor reads
/// every component it needs behind one `switch` on the page index. Threads of a workgroup usually
/// touch the same page, so the switch does not diverge.
///
/// Each accessor is generated from a template body in which `$P` stands for the page array and
/// `i` for the f32 index within that page.
///
/// # Arguments
///
/// * `n` - Number of page bindings to declare.
/// * `rw` - Declare the pages `read_write` and emit `pool_put`.
pub fn pool_wgsl(n: u32, rw: bool) -> String {
    let acc = if rw { "read_write" } else { "read" };
    let mut s = String::from("struct P5 { a: vec4<f32>, b: f32 };\n");
    for k in 0..n {
        s += &format!(
            "@group(0) @binding({}) var<storage, {acc}> pool{k}: array<f32>;\n",
            POOL_BIND + k
        );
    }
    s += r#"
fn pool_page(s: u32, bsh: u32, msh: u32) -> vec2<u32> {
  if (s >= (1u << msh)) { return vec2<u32>(msh - bsh + (s >> msh), s & ((1u << msh) - 1u)); }
  let q = s >> bsh;
  if (q == 0u) { return vec2<u32>(0u, s); }
  let k = firstLeadingBit(q) + 1u;
  return vec2<u32>(k, s - (1u << (k - 1u + bsh)));
}
"#;
    let func = |head: &str, body: &str| -> String {
        let mut a = format!("fn {head} {{\n  let i = pg.y * 2560u + o;\n  switch pg.x {{\n");
        for k in 0..n {
            let arm = if k + 1 < n {
                format!("case {k}u")
            } else {
                "default".into()
            };
            a += &format!(
                "    {arm}: {{ {} }}\n",
                body.replace("$P", &format!("pool{k}"))
            );
        }
        a + "  }\n}\n"
    };
    s += &func(
        "pool_dw(pg: vec2<u32>, o: u32) -> vec2<f32>",
        "let w = $P[i + 512u]; if (w == 0.0) { return vec2<f32>(0.0); } return vec2<f32>($P[i], w);",
    );
    s += &func(
        "pool_wc(pg: vec2<u32>, o: u32) -> vec4<f32>",
        "let w = $P[i + 512u]; if (w == 0.0) { return vec4<f32>(0.0); } return vec4<f32>(w, $P[i + 1024u], $P[i + 1536u], $P[i + 2048u]);",
    );
    s += &func(
        "pool_all(pg: vec2<u32>, o: u32) -> P5",
        "let w = $P[i + 512u]; if (w == 0.0) { return P5(vec4<f32>(0.0), 0.0); } return P5(vec4<f32>($P[i], w, $P[i + 1024u], $P[i + 1536u]), $P[i + 2048u]);",
    );
    if rw {
        s += &func(
            "pool_put(pg: vec2<u32>, o: u32, a: vec4<f32>, b: f32)",
            "$P[i] = a.x; $P[i + 512u] = a.y; $P[i + 1024u] = a.z; $P[i + 1536u] = a.w; $P[i + 2048u] = b;",
        );
    }
    s
}

/// Builds the `n` pool page bind group entries.
///
/// Bindings for pages that are not allocated yet get `dummy`, which shaders never read.
pub fn pool_entries<'a>(
    pages: &'a [wgpu::Buffer],
    dummy: &'a wgpu::Buffer,
    n: u32,
) -> Vec<wgpu::BindGroupEntry<'a>> {
    (0..n)
        .map(|k| wgpu::BindGroupEntry {
            binding: POOL_BIND + k,
            resource: pages.get(k as usize).unwrap_or(dummy).as_entire_binding(),
        })
        .collect()
}

/// Creates a small storage buffer used to fill unused page bindings.
pub fn dummy_storage(dev: &wgpu::Device) -> wgpu::Buffer {
    dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pool_dummy"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

/// Accumulation kernel (WGSL). Must be prefixed with `pool_wgsl(n, true)`.
///
/// One workgroup processes one 4³ tile and writes one changed-coordinate mask per tile.
const SPLAT: &str = r#"
/// Kernel parameters. `r2` is the splat radius squared, `inv2s2` the Gaussian weight factor,
/// `(gx0, gy0, gz0)` the grid origin of the coordinate table, `offy`/`offz` the offsets of the y and
/// z sections in that table, `r2m` the padded radius used for culling and `zero` a runtime zero
/// used by `op()`.
struct Params { r2: f32, inv2s2: f32, ntiles: u32, nx: u32, gx0: i32, gy0: i32, gz0: i32, zero: u32, offy: u32, offz: u32, r2m: f32, bsh: u32, msh: u32, pad0: u32, pad1: u32, pad2: u32 };
/// One 4³ sub-brick of a brick. `(ox, oy, oz)` is the sub-brick origin in global voxels and `sub`
/// the sub-brick index inside the brick (x + 2y + 4z). `start`/`count` select the bin list of the
/// whole 8³ brick in bin order. `fresh` is non-zero for a newly assigned slot, which starts at zero.
struct Tile { ox: i32, oy: i32, oz: i32, start: u32, count: u32, slot: u32, fresh: u32, sub: u32 };

/// Bindings. `bins` holds 48 bytes per bin: (position, x range), (normal, y range), (colour, z range),
/// where a range packs a signed 24-bit start in the low bits and an 8-bit length in the high bits.
/// `idx` is the per-brick bin index list, `coords` the grid point coordinate table and `masks`
/// receives one changed-coordinate mask per tile.
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> bins: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> idx: array<u32>;
@group(0) @binding(3) var<storage, read> tiles: array<Tile>;
@group(0) @binding(4) var<storage, read> coords: array<f32>;
@group(0) @binding(6) var<storage, read_write> masks: array<u32>;

/// Workgroup state: `wm` is the OR of changed in-brick coordinates, `tl` the current tile, `sb` the
/// bins of the current 64-entry window that reach the sub-brick (in list order), `km` the keep bits
/// of that window and `wcnt` the number of kept bins.
var<workgroup> wm: atomic<u32>;
var<workgroup> tl: Tile;
var<workgroup> sb: array<vec4<u32>, 192>;
var<workgroup> km: array<atomic<u32>, 2>;
var<workgroup> wcnt: u32;

/// Returns `x` unchanged but opaque to the compiler, so it cannot fuse or reassociate it with the
/// surrounding arithmetic (`P.zero` is 0 at run time).
fn op(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) ^ P.zero); }

/// Sign-extends the 24-bit range start packed in the low bits of `u`.
fn lo_of(u: u32) -> i32 { return (bitcast<i32>(u) << 8u) >> 8u; }

/// Accumulates one tile: one workgroup per 4³ sub-brick, one thread per voxel.
///
/// The brick's bin list is read 64 entries at a time. Bins whose radius (padded to `r2m`, so the test
/// can only err towards keeping) does not reach the sub-brick box are dropped. The survivors are
/// compacted into workgroup memory in list order and every thread scans them in that order, so each
/// voxel adds bins in bin order. Compaction uses OR-ed keep bits plus a count of the lower set bits,
/// which is order independent. A fresh tile starts from zero instead of the pool value, and a voxel
/// with Σw = 0 has all components zero because components are only added when w > 0. A voxel is
/// written back when it changed or when the tile is fresh. The keep bits are cleared after the
/// `workgroupUniformLoad` barrier, once every thread has read them; the next window's `atomicOr`
/// only happens after the barrier at the end of the loop.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.y * P.nx + wg.x;
  if (t >= P.ntiles) { return; }
  if (li == 0u) { tl = tiles[t]; atomicStore(&wm, 0u); atomicStore(&km[0], 0u); atomicStore(&km[1], 0u); }
  let tile = workgroupUniformLoad(&tl);
  let l = vec3<u32>(li & 3u, (li >> 2u) & 3u, li >> 4u);
  let g = vec3<i32>(tile.ox, tile.oy, tile.oz) + vec3<i32>(l);
  let bv = l + vec3<u32>(tile.sub & 1u, (tile.sub >> 1u) & 1u, tile.sub >> 2u) * 4u;
  let pg = pool_page(tile.slot, P.bsh, P.msh);
  let i = (bv.z * 8u + bv.y) * 8u + bv.x;
  let x = vec3<f32>(coords[u32(g.x - P.gx0)], coords[P.offy + u32(g.y - P.gy0)], coords[P.offz + u32(g.z - P.gz0)]);
  let b0 = vec3<f32>(coords[u32(tile.ox - P.gx0)], coords[P.offy + u32(tile.oy - P.gy0)], coords[P.offz + u32(tile.oz - P.gz0)]);
  let b1 = vec3<f32>(coords[u32(tile.ox + 3 - P.gx0)], coords[P.offy + u32(tile.oy + 3 - P.gy0)], coords[P.offz + u32(tile.oz + 3 - P.gz0)]);
  var a = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var cb = 0.0;
  if (tile.fresh == 0u) {
    let v = pool_all(pg, i);
    a = v.a;
    cb = v.b;
  }
  var hit = false;
  let end = tile.start + tile.count;
  for (var k0 = tile.start; k0 < end; k0 = k0 + 64u) {
    let k = k0 + li;
    var keep = 0u;
    var e0 = vec4<u32>(0u);
    var e1 = vec4<u32>(0u);
    var e2 = vec4<u32>(0u);
    if (k < end) {
      let bi = idx[k] * 3u;
      e0 = bins[bi];
      e1 = bins[bi + 1u];
      e2 = bins[bi + 2u];
      let p = bitcast<vec3<f32>>(e0.xyz);
      let dd = max(max(b0 - p, p - b1), vec3<f32>(0.0));
      if ((dd.x * dd.x + dd.y * dd.y) + dd.z * dd.z <= P.r2m) { keep = 1u; }
    }
    if (keep == 1u) { atomicOr(&km[li >> 5u], 1u << (li & 31u)); }
    workgroupBarrier();
    let m0 = atomicLoad(&km[0]);
    let m1 = atomicLoad(&km[1]);
    if (keep == 1u) {
      var q = countOneBits(m0 & ((1u << (li & 31u)) - 1u));
      if (li >= 32u) { q = countOneBits(m0) + countOneBits(m1 & ((1u << (li & 31u)) - 1u)); }
      q = q * 3u;
      sb[q] = e0;
      sb[q + 1u] = e1;
      sb[q + 2u] = e2;
    }
    if (li == 0u) { wcnt = countOneBits(m0) + countOneBits(m1); }
    let cnt = workgroupUniformLoad(&wcnt);
    if (li == 0u) { atomicStore(&km[0], 0u); atomicStore(&km[1], 0u); }
    for (var j = 0u; j < cnt; j = j + 1u) {
      let a0 = sb[j * 3u];
      let a1 = sb[j * 3u + 1u];
      let a2 = sb[j * 3u + 2u];
      let lo = vec3<i32>(lo_of(a0.w), lo_of(a1.w), lo_of(a2.w));
      let hi = lo + vec3<i32>(vec3<u32>(a0.w, a1.w, a2.w) >> vec3<u32>(24u));
      if (any(g < lo) || any(g > hi)) { continue; }
      let d = x - bitcast<vec3<f32>>(a0.xyz);
      let q = op(op(op(d.x * d.x) + op(d.y * d.y)) + op(d.z * d.z));
      if (q <= P.r2) {
        let nn = bitcast<vec3<f32>>(a1.xyz);
        let col = bitcast<vec3<f32>>(a2.xyz);
        let ww = exp(-q * P.inv2s2);
        a = a + ww * vec4<f32>((nn.x * d.x + nn.y * d.y) + nn.z * d.z, 1.0, col.x, col.y);
        cb = cb + ww * col.z;
        hit = true;
      }
    }
    workgroupBarrier();
  }
  if (hit) {
    atomicOr(&wm, (1u << bv.x) | (256u << bv.y) | (65536u << bv.z));
  }
  if (hit || tile.fresh != 0u) {
    pool_put(pg, i, a, cb);
  }
  workgroupBarrier();
  if (li == 0u) { masks[t] = atomicLoad(&wm); }
}
"#;

/// Shared prefix of the gather kernels (WGSL). Must be prefixed with `pool_wgsl(n, false)`.
///
/// Sums a voxel over all layers as `a = (Σw·d, Σw, Σw·r, Σw·g)` and `b = Σw·b`. Layer order and the
/// skipping of w = 0 entries match the CPU.
const GATHER_HEAD: &str = r#"
/// Gather parameters. `p` is the padded block edge, `nr` the number of regions per axis, `owp` the
/// number of u32 words in a block header (region occupancy bits), `stride` the number of u32 words
/// per block and `nitems` the number of occupied regions (`owp`, `stride` and `nitems` are only used
/// by the compact layout).
struct GP { nblocks: u32, p: u32, nr: u32, nx: u32, minw: f32, owp: u32, stride: u32, nitems: u32, bsh: u32, msh: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> G: GP;
@group(0) @binding(2) var<storage, read> off: array<u32>;
@group(0) @binding(3) var<storage, read> slots: array<u32>;

/// Distance written for unobserved voxels.
const MARK: f32 = 3.0e38;

/// Layer sum of one voxel. `n` is the number of bricks in the voxel's region (0 for an empty region).
struct Acc { a: vec4<f32>, b: f32, n: u32 };

/// Returns only `(Σw·d, Σw)` of `sum_region` without reading colour. The summation order is the
/// same, so both values are bitwise identical to those of `sum_region`.
fn sum_dist(o: u32, q: vec3<u32>) -> vec2<f32> {
  let vi = ((((q.z + 7u) & 7u) * 8u) + ((q.y + 7u) & 7u)) * 8u + ((q.x + 7u) & 7u);
  let e1 = off[o + 1u];
  var a = vec2<f32>(0.0, 0.0);
  for (var e = off[o]; e < e1; e = e + 1u) {
    let pg = pool_page(slots[e], G.bsh, G.msh);
    let v = pool_dw(pg, vi);
    if (v.y != 0.0) { a = a + v; }
  }
  return a;
}

/// Sums padded voxel `v` of block `j` over all layers.
///
/// Padded coordinate = local + 1. The region is (local + 8) / 8, so region 0 is local -1 and region
/// nb + 1 is local `dim`; the in-brick coordinate is (local + 8) % 8.
fn sum_layers(j: u32, v: u32) -> Acc {
  let px = v % G.p; let py = (v / G.p) % G.p; let pz = v / (G.p * G.p);
  let r = (((pz + 7u) >> 3u) * G.nr + ((py + 7u) >> 3u)) * G.nr + ((px + 7u) >> 3u);
  return sum_region(j * G.nr * G.nr * G.nr + r, vec3<u32>(px, py, pz));
}

/// Sums padded voxel `q` over the slot list of region `o` (block index · nr³ + region).
fn sum_region(o: u32, q: vec3<u32>) -> Acc {
  let vi = ((((q.z + 7u) & 7u) * 8u) + ((q.y + 7u) & 7u)) * 8u + ((q.x + 7u) & 7u);
  let e1 = off[o + 1u];
  var acc: Acc;
  acc.a = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  acc.b = 0.0;
  acc.n = e1 - off[o];
  for (var e = off[o]; e < e1; e = e + 1u) {
    let pg = pool_page(slots[e], G.bsh, G.msh);
    let v = pool_all(pg, vi);
    if (v.a.y != 0.0) {
      acc.a = acc.a + v.a;
      acc.b = acc.b + v.b;
    }
  }
  return acc;
}
"#;

/// Compact gather kernels for extraction (WGSL), appended to `GATHER_HEAD`.
///
/// Layout per block, in u32 words: `[region occupancy bits; owp][distance; P³]`. Only regions that
/// contain bricks are written, one workgroup each; empty regions are left untouched and extraction
/// skips them based on the occupancy bits. Unobserved voxels get distance `MARK`. Colour is not
/// produced here: extraction reads only the cells it needs for vertices from the pool through the
/// same slot table.
const GATHER: &str = r#"
/// Bindings. `items` lists the occupied regions as `o = block · nr³ + region`.
@group(0) @binding(4) var<storage, read_write> outv: array<u32>;
@group(0) @binding(5) var<storage, read> occ: array<u32>;
@group(0) @binding(6) var<storage, read> items: array<u32>;

/// Start and length of region `r` in padded coordinates: region 0 is `[0, 1)`, regions `1..=nb`
/// start at `8(r - 1) + 1` and span 8 voxels, region `nb + 1` is `[p - 1, p)`.
fn rstart(r: u32) -> u32 { return select(select(8u * r - 7u, G.p - 1u, r == G.nr - 1u), 0u, r == 0u); }
fn rlen(r: u32) -> u32 { return select(8u, 1u, r == 0u || r == G.nr - 1u); }

/// Writes the distances of one occupied region.
///
/// A voxel is observed when Σw ≥ min_weight (the same rule as the CPU mesher); its distance is
/// Σw·d / Σw.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let k = wg.y * G.nx + wg.x;
  if (k >= G.nitems) { return; }
  let o = items[k];
  let nr3 = G.nr * G.nr * G.nr;
  let j = o / nr3;
  let r = o - j * nr3;
  let rr = vec3<u32>(r % G.nr, (r / G.nr) % G.nr, r / (G.nr * G.nr));
  let lo = vec3<u32>(rstart(rr.x), rstart(rr.y), rstart(rr.z));
  let ln = vec3<u32>(rlen(rr.x), rlen(rr.y), rlen(rr.z));
  let dbase = j * G.stride + G.owp;
  for (var e = li; e < ln.x * ln.y * ln.z; e = e + 64u) {
    let q = lo + vec3<u32>(e % ln.x, (e / ln.x) % ln.y, e / (ln.x * ln.y));
    let v = (q.z * G.p + q.y) * G.p + q.x;
    let a = sum_dist(o, q);
    outv[dbase + v] = bitcast<u32>(select(MARK, a.x / a.y, a.y >= G.minw && a.y > 0.0));
  }
}

/// Copies the block headers (region occupancy bits), one word per thread.
@compute @workgroup_size(64)
fn head(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = wg.x * 64u + li;
  if (i >= G.nblocks * G.owp) { return; }
  let j = i / G.owp;
  outv[j * G.stride + (i - j * G.owp)] = occ[i];
}
"#;

/// Raw-sum gather kernel (WGSL), appended to `GATHER_HEAD`.
///
/// Writes eight f32 per padded voxel: Σw·d, Σw, Σw·r, Σw·g, Σw·b, 0, 0, 0. Used for accuracy tests
/// and diagnostics.
const GATHER_RAW: &str = r#"
@group(0) @binding(4) var<storage, read_write> outv: array<vec4<f32>>;

/// Writes the layer sums of 64 padded voxels per workgroup.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let L = (wg.y * G.nx + wg.x) * 64u + li;
  let p3 = G.p * G.p * G.p;
  let j = L / p3;
  if (j >= G.nblocks) { return; }
  let acc = sum_layers(j, L - j * p3);
  outv[L * 2u] = acc.a;
  outv[L * 2u + 1u] = vec4<f32>(acc.b, 0.0, 0.0, 0.0);
}
"#;

/// Base agreement kernel (WGSL). Must be prefixed with `pool_wgsl(n, false)`.
///
/// Mirrors the CPU base agreement on the base layer only, looking bricks up through the hash table.
const AGREE: &str = r#"
/// Agreement parameters. `mask` is the hash table mask, `h` the probe offset along the normal,
/// `h2` = 2h, `inv` = 1 / voxel and `zero` a runtime zero used by `op()`.
struct AP { npts: u32, nx: u32, mask: u32, zero: u32, h: f32, h2: f32, inv: f32, minw: f32, bsh: u32, msh: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> A: AP;
@group(0) @binding(2) var<storage, read> table: array<vec4<i32>>;
@group(0) @binding(3) var<storage, read> pts: array<f32>;
@group(0) @binding(4) var<storage, read_write> outv: array<f32>;

/// Returns `x` unchanged but opaque to the compiler, preventing FMA contraction and reassociation.
fn op(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) ^ A.zero); }

/// Looks up the slot of base brick `bk` (linear probing, same hash as `table_insert`); -1 if absent.
fn slot_of(bk: vec3<i32>) -> i32 {
  var h = ((u32(bk.x) * 73856093u) ^ (u32(bk.y) * 19349663u) ^ (u32(bk.z) * 83492791u)) & A.mask;
  loop {
    let e = table[h];
    if (e.w < 0) { return -1; }
    if (all(e.xyz == bk)) { return e.w; }
    h = (h + 1u) & A.mask;
  }
  return -1;
}

/// Distance at grid point `g` as `(1, d)`, or `(0, 0)` when the point is unobserved.
fn value(g: vec3<i32>) -> vec2<f32> {
  let s = slot_of(g >> vec3<u32>(3u, 3u, 3u));
  if (s < 0) { return vec2<f32>(0.0, 0.0); }
  let l = vec3<u32>(g & vec3<i32>(7, 7, 7));
  let pg = pool_page(u32(s), A.bsh, A.msh);
  let i = (l.z * 8u + l.y) * 8u + l.x;
  let v = pool_dw(pg, i);
  if (!(v.y >= A.minw)) { return vec2<f32>(0.0, 0.0); }
  return vec2<f32>(1.0, v.x / v.y);
}

/// Trilinear sample at `x` as `(1, d)`; fails with `(0, 0)` unless all eight corners are observed.
fn sample(x: vec3<f32>) -> vec2<f32> {
  let f = vec3<f32>(op(x.x * A.inv), op(x.y * A.inv), op(x.z * A.inv));
  let fl = floor(f);
  let g = vec3<i32>(fl);
  let t = f - fl;
  var s = 0.0;
  for (var c = 0u; c < 8u; c = c + 1u) {
    let o = vec3<u32>(c & 1u, (c >> 1u) & 1u, (c >> 2u) & 1u);
    let d = value(g + vec3<i32>(o));
    if (d.x == 0.0) { return vec2<f32>(0.0, 0.0); }
    var w = 1.0;
    w = w * select(1.0 - t.x, t.x, o.x == 1u);
    w = w * select(1.0 - t.y, t.y, o.y == 1u);
    w = w * select(1.0 - t.z, t.z, o.z == 1u);
    s = s + w * d.y;
  }
  return vec2<f32>(1.0, s);
}

/// Computes the clamped central difference of the field along each point's normal, or 9.0 when
/// either sample is unobserved.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let k = (wg.y * A.nx + wg.x) * 64u + li;
  if (k >= A.npts) { return; }
  let p = vec3<f32>(pts[k * 6u], pts[k * 6u + 1u], pts[k * 6u + 2u]);
  let n = vec3<f32>(pts[k * 6u + 3u], pts[k * 6u + 4u], pts[k * 6u + 5u]);
  let hn = vec3<f32>(op(A.h * n.x), op(A.h * n.y), op(A.h * n.z));
  let xa = vec3<f32>(op(p.x + hn.x), op(p.y + hn.y), op(p.z + hn.z));
  let xb = vec3<f32>(op(p.x - hn.x), op(p.y - hn.y), op(p.z - hn.z));
  let a = sample(xa);
  var res = 9.0;
  if (a.x != 0.0) {
    let b = sample(xb);
    if (b.x != 0.0) { res = clamp((a.y - b.y) / A.h2, -1.0, 1.0); }
  }
  outv[k] = res;
}
"#;

/// Uniform parameters of the accumulation kernel; mirrors WGSL `Params`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SplatParams {
    r2: f32,
    inv2s2: f32,
    ntiles: u32,
    nx: u32,
    gx0: i32,
    gy0: i32,
    gz0: i32,
    zero: u32,
    offy: u32,
    offz: u32,
    r2m: f32,
    bsh: u32,
    msh: u32,
    pad: [u32; 3],
}

/// One accumulation tile (4³ sub-brick); mirrors WGSL `Tile` (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Tile {
    o: [i32; 3],
    start: u32,
    count: u32,
    slot: u32,
    fresh: u32,
    sub: u32,
}

/// GPU bin record (48 bytes): position, normal and colour, each followed by the packed voxel range
/// of one axis (signed 24-bit start in the low bits, 8-bit length in the high bits).
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct BinG {
    p: [f32; 3],
    rx: u32,
    n: [f32; 3],
    ry: u32,
    c: [f32; 3],
    rz: u32,
}

/// Uniform parameters of the gather kernels; mirrors WGSL `GP`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherParams {
    nblocks: u32,
    p: u32,
    nr: u32,
    nx: u32,
    minw: f32,
    owp: u32,
    stride: u32,
    nitems: u32,
    bsh: u32,
    msh: u32,
    pad: [u32; 2],
}

/// Uniform parameters of the agreement kernel; mirrors WGSL `AP`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AgreeParams {
    npts: u32,
    nx: u32,
    mask: u32,
    zero: u32,
    h: f32,
    h2: f32,
    inv: f32,
    minw: f32,
    bsh: u32,
    msh: u32,
    pad: [u32; 2],
}

/// Layer that an integrate call accumulates into.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LayerKey {
    /// The base layer holding refined segments.
    Base,
    /// The separate layer of one preview segment, dropped when the segment is replaced.
    Pending(SegmentId),
}

/// One block of a layer: its brick slots (`nb³`, `NONE` when absent) and the local range changed in
/// this layer.
struct LBlock {
    slots: Vec<u32>,
    min: [i32; 3],
    max: [i32; 3],
}

/// Blocks of one layer.
type Layer = FxHashMap<BlockId, LBlock>;

/// Inclusive local voxel range `(min, max)`; `min > max` when nothing changed.
type LocalRange = ([i32; 3], [i32; 3]);

/// Blocks with the local range `(min, max)` changed by a call; `min > max` when no voxel changed.
pub type BlockRanges = Vec<(BlockId, [i32; 3], [i32; 3])>;

/// Per-block brick summary: `(brick index, bin count, sub-brick bits)` for each non-empty brick,
/// and the total length of the block's bin lists.
type BrickKeys = (Vec<(u32, u32, u8)>, u32);

/// Hash table used by `base_agreement`.
///
/// Holds the base version it was built for, the GPU buffer, the table mask, a CPU copy of the table
/// and the number of entries.
struct HashCache {
    ver: u64,
    buf: wgpu::Buffer,
    mask: u32,
    table: Vec<[i32; 4]>,
    n: usize,
}

/// Base hash table together with the base bricks added since it was built.
///
/// Base bricks are only ever added (only preview layers are dropped), so the table can be updated
/// incrementally by inserting just the new bricks.
#[derive(Default)]
struct HashState {
    cache: Option<HashCache>,
    /// Bricks added to the base layer since the table was built, as `[bx, by, bz, slot]`.
    added: Vec<[i32; 4]>,
    /// Number of incremental table updates (checked by tests).
    incr: u32,
}

/// Persistent buffers of `base_agreement` (point upload and result readback).
struct AgreeBufs {
    pts: Up,
    out: Up,
}

/// Brick pool usage, for tests and instrumentation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Allocated pool capacity in bricks.
    pub cap: u32,
    /// Number of slots ever handed out (high-water mark).
    pub high: u32,
    /// Length of the free-slot list.
    pub free: usize,
    /// Number of slots in use.
    pub live: usize,
    /// Number of times the pool grew.
    pub grows: u32,
}

/// Time breakdown of the last integrate call in milliseconds, summed over batches.
#[derive(Clone, Copy, Debug, Default)]
pub struct FieldTimes {
    /// Packing bins, counting per brick, assigning slots and building tiles.
    pub bricks: f64,
    /// Building the coordinate table and checking limits.
    pub pack: f64,
    /// Waiting for mappings, filling and submitting the lists (CPU side).
    pub upload: f64,
    /// From the last submission through mask readback and post-processing, including `during`.
    pub gpu: f64,
    /// Computing ranges and releasing bricks that did not change.
    pub post: f64,
    /// CPU work run while waiting for the GPU (the `during` callback of `integrate_bins`).
    pub during: f64,
}

/// Milliseconds elapsed since `t`.
fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Persistent buffer pair: a CPU-mapped staging buffer and a GPU buffer.
///
/// Each buffer is recreated at the next power of two when it is too small.
#[derive(Default)]
struct Up {
    /// Staging buffer mapped by the CPU.
    st: Option<wgpu::Buffer>,
    /// GPU-side storage buffer.
    dst: Option<wgpu::Buffer>,
    /// Capacity of `st` in bytes.
    cap: u64,
    /// Capacity of `dst` in bytes.
    dcap: u64,
    /// Readback pair (`dst`: STORAGE | COPY_SRC, `st`: MAP_READ | COPY_DST) instead of upload.
    read: bool,
}

impl Up {
    /// Returns the `(staging, gpu)` buffer usages for this direction.
    fn usages(&self) -> (wgpu::BufferUsages, wgpu::BufferUsages) {
        if self.read {
            (
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            )
        } else {
            (
                wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            )
        }
    }

    /// Creates a buffer of at least `bytes` (power of two, minimum 64 KiB) and returns its capacity.
    fn mk(dev: &wgpu::Device, bytes: u64, usage: wgpu::BufferUsages) -> (wgpu::Buffer, u64) {
        let cap = bytes.max(16).next_power_of_two().max(1 << 16);
        (
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("field_scratch"),
                size: cap,
                usage,
                mapped_at_creation: false,
            }),
            cap,
        )
    }

    /// Recreates either buffer that is smaller than `bytes`.
    fn ensure(&mut self, dev: &wgpu::Device, bytes: u64) {
        self.ensure_st(dev, bytes);
        self.ensure_dst(dev, bytes);
    }

    /// Recreates the staging buffer if it is smaller than `bytes`.
    fn ensure_st(&mut self, dev: &wgpu::Device, bytes: u64) {
        if self.cap >= bytes.max(16) && self.st.is_some() {
            return;
        }
        let (b, c) = Self::mk(dev, bytes, self.usages().0);
        self.st = Some(b);
        self.cap = c;
    }

    /// Recreates the GPU buffer if it is smaller than `bytes`.
    fn ensure_dst(&mut self, dev: &wgpu::Device, bytes: u64) {
        if self.dcap >= bytes.max(16) && self.dst.is_some() {
            return;
        }
        let (b, c) = Self::mk(dev, bytes, self.usages().1);
        self.dst = Some(b);
        self.dcap = c;
    }
}

/// Buffers reused by every integrate call, one set per batch.
struct Scratch {
    bins: Up,
    idx: Up,
    tiles: Up,
    coords: Up,
    masks: Up,
}

impl Default for Scratch {
    fn default() -> Self {
        Self {
            bins: Up::default(),
            idx: Up::default(),
            tiles: Up::default(),
            coords: Up::default(),
            masks: Up {
                read: true,
                ..Up::default()
            },
        }
    }
}

impl Scratch {
    /// Total bytes held by all buffers of this set.
    fn bytes(&self) -> u64 {
        [
            &self.bins,
            &self.idx,
            &self.tiles,
            &self.coords,
            &self.masks,
        ]
        .iter()
        .map(|u| u.cap + u.dcap)
        .sum()
    }
}

/// A submitted batch whose masks have not been read back yet.
struct Sent {
    /// Blocks in first-seen order.
    order: Vec<BlockId>,
    /// Per brick: `(index into order, brick index, first tile, end tile)`.
    bricks: Vec<(u32, u32, u32, u32)>,
    tiles: Vec<Tile>,
    /// Blocks created in the layer by this batch.
    new_blocks: Vec<BlockId>,
    /// Index of the scratch buffer set used.
    set: usize,
    /// Size of the mask readback in bytes.
    mbytes: u64,
}

/// Raw pointer that lets several threads write disjoint parts of one buffer.
struct SendPtr<T>(*mut T);
// SAFETY: the pointer is only used for writes to disjoint ranges of a buffer that outlives the
// parallel section, so sharing it between threads cannot cause a data race.
unsafe impl<T> Send for SendPtr<T> {}
// SAFETY: see the `Send` impl above.
unsafe impl<T> Sync for SendPtr<T> {}

/// Splits `nwg` workgroups into a 2D dispatch grid that respects the per-axis limit.
fn grid(nwg: u32) -> (u32, u32) {
    let x = nwg.clamp(1, MAX_WG);
    (x, nwg.div_ceil(x).max(1))
}

/// Per-chunk (block, brick) counts for the counting sort.
///
/// After the first pass the counts are turned into write positions.
#[derive(Default)]
struct Part {
    /// Blocks in first-seen order within the chunk.
    ids: Vec<BlockId>,
    /// Block to its index in `ids`.
    index: FxHashMap<BlockId, usize>,
    /// `nk` entries per block: bin count per brick (write position during the second pass).
    cnt: Vec<u32>,
    /// `nk` entries per block: bits of the 4³ sub-bricks reached by a bin's range box.
    sub: Vec<u8>,
    /// (block, brick) cells emitted in the first pass, in bin order. The second pass replays this
    /// instead of recomputing it.
    em: Vec<u32>,
    /// Number of cells emitted per bin.
    per_bin: Vec<u16>,
}

impl Part {
    /// Returns the chunk-local index of block `id`, adding the block if needed.
    fn local(&mut self, id: BlockId, nk: usize) -> usize {
        match self.index.get(&id) {
            Some(&k) => k,
            None => {
                self.ids.push(id);
                self.cnt.resize(self.cnt.len() + nk, 0);
                self.sub.resize(self.sub.len() + nk, 0);
                self.index.insert(id, self.ids.len() - 1);
                self.ids.len() - 1
            }
        }
    }
}

/// Cache of the previous bin's block range and the chunk-local indices of those blocks.
///
/// Bins arrive in cell order, so neighbouring bins usually cover the same block range.
#[derive(Default)]
struct RangeCache {
    key: Option<[i32; 6]>,
    ks: Vec<(BlockId, usize)>,
}

impl RangeCache {
    /// Returns the blocks reached by range `rr` (in `for_blocks` order) with their chunk-local
    /// indices, calling `look` only when the block range changes.
    #[inline]
    fn get(
        &mut self,
        rr: &[(i32, i32); 3],
        dim: i32,
        mut look: impl FnMut(BlockId) -> usize,
    ) -> &[(BlockId, usize)] {
        let key = block_range(rr, dim);
        if self.key != Some(key) {
            self.ks.clear();
            let ks = &mut self.ks;
            for_blocks(&key, |id| ks.push((id, look(id))));
            self.key = Some(key);
        }
        &self.ks
    }
}

/// Floor division of `x` by `dim`, using a shift when `dim` is a power of two.
#[inline(always)]
fn fdiv(x: i32, dim: i32) -> i32 {
    if dim.count_ones() == 1 {
        x >> dim.trailing_zeros()
    } else {
        x.div_euclid(dim)
    }
}

/// Returns the block range `[x0, x1, y0, y1, z0, z1]` covered by a per-axis voxel range.
#[inline(always)]
fn block_range(rr: &[(i32, i32); 3], dim: i32) -> [i32; 6] {
    [
        fdiv(rr[0].0, dim),
        fdiv(rr[0].1, dim),
        fdiv(rr[1].0, dim),
        fdiv(rr[1].1, dim),
        fdiv(rr[2].0, dim),
        fdiv(rr[2].1, dim),
    ]
}

/// Calls `f` for every block in `br`, in z, y, x order (the same order as the CPU block lists).
#[inline(always)]
fn for_blocks(br: &[i32; 6], mut f: impl FnMut(BlockId)) {
    for bz in br[4]..=br[5] {
        for by in br[2]..=br[3] {
            for bx in br[0]..=br[1] {
                f([bx, by, bz]);
            }
        }
    }
}

/// Enumerates the bricks of a block that a bin reaches.
///
/// For each 8³ brick of the block at origin `o` that the bin's range `rr` overlaps and whose box
/// lies within the padded radius `r2m` of `pos`, calls `f(brick index, sub-brick bits)`. The eight
/// bits mark the 4³ sub-bricks that the range box overlaps; bit index = sub-brick index, where the
/// sub-brick coordinate along axis `a` is bit `a`. Sub-bricks outside the range box contain no
/// changed voxel. The padding means the test can only err towards keeping a brick; the per
/// sub-brick distance test is done on the GPU.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn for_bricks(
    o: [i32; 3],
    rr: &[(i32, i32); 3],
    pos: [f32; 3],
    dim: i32,
    nb: i32,
    v: f32,
    r2m: f32,
    mut f: impl FnMut(usize, u8),
) {
    let mut lo = [0i32; 3];
    let mut hi = [0i32; 3];
    let mut ll = [0i32; 3];
    let mut hh = [0i32; 3];
    for a in 0..3 {
        let (l, h) = (rr[a].0.max(o[a]), rr[a].1.min(o[a] + dim - 1));
        if l > h {
            return;
        }
        ll[a] = l - o[a];
        hh[a] = h - o[a];
        lo[a] = ll[a] >> 3;
        hi[a] = hh[a] >> 3;
    }
    let dist2 = |a: usize, k: i32| {
        let g0 = (o[a] + k * BR) as f32 * v;
        let g1 = (o[a] + k * BR + BR - 1) as f32 * v;
        let p = pos[a];
        let d = if p < g0 {
            g0 - p
        } else if p > g1 {
            p - g1
        } else {
            0.0
        };
        d * d
    };
    const PAT: [[u8; 2]; 3] = [[0x55, 0xAA], [0x33, 0xCC], [0x0F, 0xF0]];
    let half = |a: usize, k: i32| {
        let s0 = ((ll[a] - k * BR).max(0)) >> 2;
        let s1 = ((hh[a] - k * BR).min(BR - 1)) >> 2;
        (if s0 == 0 { PAT[a][0] } else { 0 }) | (if s1 == 1 { PAT[a][1] } else { 0 })
    };
    for bz in lo[2]..=hi[2] {
        let dz = dist2(2, bz);
        if dz > r2m {
            continue;
        }
        let mz = half(2, bz);
        for by in lo[1]..=hi[1] {
            let dyz = dz + dist2(1, by);
            if dyz > r2m {
                continue;
            }
            let myz = mz & half(1, by);
            for bx in lo[0]..=hi[0] {
                if dyz + dist2(0, bx) > r2m {
                    continue;
                }
                f(((bz * nb + by) * nb + bx) as usize, myz & half(0, bx));
            }
        }
    }
}

/// Copies `src` into a mapped write-only range in parallel 1 MiB chunks.
///
/// # Panics
///
/// Panics if `out` is shorter than `src`.
fn par_copy(out: &mut wgpu::WriteOnly<'_, [u8]>, src: &[u8]) {
    const CH: usize = 1 << 20;
    let dst = SendPtr(out.as_raw_element_ptr().as_ptr());
    assert!(out.len() >= src.len());
    src.par_chunks(CH).enumerate().for_each(|(i, c)| {
        let dst = &dst;
        // SAFETY: each chunk writes a disjoint range that lies within `out`.
        unsafe { std::ptr::copy_nonoverlapping(c.as_ptr(), dst.0.add(i * CH), c.len()) };
    });
}

/// Inserts entry `[bx, by, bz, slot]` into the open-addressing table `t`.
///
/// Uses linear probing and the same hash as `slot_of` in the agreement shader.
fn table_insert(t: &mut [[i32; 4]], mask: u32, e: [i32; 4]) {
    let mut h = ((e[0] as u32).wrapping_mul(73856093)
        ^ (e[1] as u32).wrapping_mul(19349663)
        ^ (e[2] as u32).wrapping_mul(83492791))
        & mask;
    while t[h as usize][3] >= 0 {
        h = (h + 1) & mask;
    }
    t[h as usize] = e;
}

/// Maps upload staging buffers for writing and waits until all mappings complete.
///
/// Only buffers the GPU is not using are passed in, so the device is polled without blocking instead
/// of waiting for previously submitted work (such as the previous batch's accumulation) to finish.
///
/// # Errors
///
/// Returns an error if polling the device or mapping a buffer fails.
fn map_write(dev: &wgpu::Device, ups: &[(&wgpu::Buffer, u64)]) -> Result<(), String> {
    let (tx, rx) = std::sync::mpsc::channel();
    for &(b, n) in ups {
        let tx = tx.clone();
        b.slice(0..n.max(4))
            .map_async(wgpu::MapMode::Write, move |r| {
                let _ = tx.send(r);
            });
    }
    drop(tx);
    let mut got = 0;
    while got < ups.len() {
        dev.poll(wgpu::PollType::Poll)
            .map_err(|e| format!("GPU poll failed: {e}"))?;
        while let Ok(r) = rx.try_recv() {
            r.map_err(|e| format!("failed to map upload buffer: {e}"))?;
            got += 1;
        }
        if got < ups.len() {
            std::thread::yield_now();
        }
    }
    Ok(())
}

/// GPU-resident signed-distance field with a base layer and per-segment preview layers.
///
/// Field values stay on the GPU in a paged brick pool; the CPU tracks only slot assignments and
/// changed ranges. See the module documentation for the data layout and determinism rules.
///
/// GPU submissions and waits are serialized through `ctx.lock`. Work that uses rayon (list building,
/// point packing, the `during` callback) runs outside that lock: a rayon worker holding the lock
/// could steal a task that wants the same lock and deadlock.
pub struct GpuField {
    ctx: Arc<GpuCtx>,
    cfg: Config,
    /// Bricks per block edge.
    nb: i32,
    base: Layer,
    pending: BTreeMap<SegmentId, Layer>,
    /// Brick pool pages in layout order. Growing only appends pages.
    pages: Vec<wgpu::Buffer>,
    layout: PageLayout,
    /// Buffer bound in place of pages that are not allocated.
    dummy: wgpu::Buffer,
    /// Total slots of the allocated pages.
    cap: u32,
    /// Number of slots ever handed out.
    high: u32,
    /// Released slots available for reuse.
    free: Vec<u32>,
    /// Number of slots in use.
    live: usize,
    /// Number of times the pool grew.
    grows: u32,
    /// Largest increase in live slots during one integrate call; used to decide when to prefetch a
    /// page.
    used_max: u32,
    times: FieldTimes,
    /// Buffer sets, one per batch. `scratch[0].bins.dst` is the bin buffer shared by all batches.
    scratch: Vec<Scratch>,
    /// Version of the base layer, bumped on every base integrate.
    base_ver: u64,
    hash: Mutex<HashState>,
    agree: Mutex<AgreeBufs>,
    p_splat: wgpu::ComputePipeline,
    p_gather: wgpu::ComputePipeline,
    p_gather_head: wgpu::ComputePipeline,
    p_gather_raw: wgpu::ComputePipeline,
    p_agree: wgpu::ComputePipeline,
}

impl GpuField {
    /// Creates a field with the default first page size.
    ///
    /// # Errors
    ///
    /// See [`GpuField::with_capacity`].
    pub fn new(ctx: Arc<GpuCtx>, cfg: &Config) -> Result<Self, String> {
        Self::with_capacity(ctx, cfg, INIT_CAP)
    }

    /// Creates a field whose first pool page holds `cap` bricks; used by pool growth tests.
    ///
    /// Compiles all pipelines and allocates and zero-fills the first page, so that the first
    /// accumulation does not pay for it.
    ///
    /// # Errors
    ///
    /// Returns an error if the device cannot bind at least two pool pages, a shader fails to
    /// compile, or the first page cannot be allocated.
    ///
    /// # Panics
    ///
    /// Panics if `cfg.block_dim` is not a positive multiple of 8.
    #[doc(hidden)]
    pub fn with_capacity(ctx: Arc<GpuCtx>, cfg: &Config, cap: u32) -> Result<Self, String> {
        assert!(
            cfg.block_dim % BR == 0 && cfg.block_dim > 0,
            "block_dim must be a positive multiple of 8"
        );
        let dev = &ctx.device;
        let mk_e = |label, src: &str, entry: &str| {
            let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let mk = |label, src: &str| mk_e(label, src, "main");
        let npg = pool_pages(&ctx);
        if npg < 2 {
            return Err(format!(
                "not enough storage buffer bindings ({npg} pool pages)"
            ));
        }
        let layout = PageLayout::new(cap, npg, ctx.max_binding);
        let (prw, pro) = (pool_wgsl(npg, true), pool_wgsl(npg, false));
        let scope = dev.push_error_scope(wgpu::ErrorFilter::Validation);
        let p_splat = mk("field_splat", &format!("{prw}{SPLAT}"));
        let p_gather = mk("field_gather", &format!("{pro}{GATHER_HEAD}{GATHER}"));
        let p_gather_head = mk_e(
            "field_gather_head",
            &format!("{pro}{GATHER_HEAD}{GATHER}"),
            "head",
        );
        let p_gather_raw = mk(
            "field_gather_raw",
            &format!("{pro}{GATHER_HEAD}{GATHER_RAW}"),
        );
        let p_agree = mk("field_agree", &format!("{pro}{AGREE}"));
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(format!("failed to create shaders: {e:?}"));
        }
        let page0 = Self::make_page(&ctx, &layout, 0)?;
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pool_page"),
        });
        enc.clear_buffer(&page0, 0, None);
        ctx.queue.submit(Some(enc.finish()));
        let cap = layout.page_slots(0);
        let dummy = dummy_storage(dev);
        Ok(Self {
            nb: cfg.block_dim / BR,
            cfg: *cfg,
            ctx,
            base: Layer::default(),
            pending: BTreeMap::new(),
            pages: vec![page0],
            layout,
            dummy,
            cap,
            high: 0,
            free: Vec::new(),
            live: 0,
            grows: 0,
            used_max: 0,
            times: FieldTimes::default(),
            scratch: vec![Scratch::default()],
            base_ver: 0,
            hash: Mutex::new(HashState::default()),
            agree: Mutex::new(AgreeBufs {
                pts: Up::default(),
                out: Up {
                    read: true,
                    ..Up::default()
                },
            }),
            p_splat,
            p_gather,
            p_gather_head,
            p_gather_raw,
            p_agree,
        })
    }

    /// Allocates pool page `k`.
    ///
    /// The contents are left undefined because the accumulation kernel writes every fresh brick in
    /// full. Out-of-memory is caught through an error scope and reported as `Err` instead of a
    /// panic, so the caller can retry later.
    ///
    /// # Errors
    ///
    /// Returns an error if the page exceeds the binding size limit or allocation fails.
    fn make_page(ctx: &GpuCtx, layout: &PageLayout, k: u32) -> Result<wgpu::Buffer, String> {
        let size = layout.page_slots(k) as u64 * BRICK_BYTES;
        if size > ctx.max_binding {
            return Err(format!(
                "brick pool page exceeds the binding limit ({size} B, limit {} B)",
                ctx.max_binding
            ));
        }
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field_pool"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(format!(
                "failed to allocate brick pool page ({size} B): {e}"
            ));
        }
        Ok(buf)
    }

    /// Allocates the next page ahead of time when headroom is low.
    ///
    /// When the remaining slots (free list plus never-used capacity) are fewer than the largest
    /// growth seen in one integrate call, allocates the next page and submits its zero-fill without
    /// waiting. Must be called with the device lock held.
    ///
    /// # Errors
    ///
    /// Returns an error if the page cannot be allocated.
    fn prefetch_page(&mut self) -> Result<(), String> {
        let left = (self.cap - self.high.min(self.cap)) as u64 + self.free.len() as u64;
        let want = self.used_max as u64;
        if want == 0 || left >= want || self.pages.len() as u32 >= self.layout.n {
            return Ok(());
        }
        let k = self.pages.len() as u32;
        let pg = Self::make_page(&self.ctx, &self.layout, k)?;
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pool_page"),
            });
        enc.clear_buffer(&pg, 0, None);
        self.ctx.queue.submit(Some(enc.finish()));
        self.pages.push(pg);
        self.cap += self.layout.page_slots(k);
        self.grows += 1;
        Ok(())
    }

    /// Appends pages until the pool holds at least `need` slots. Existing pages are not copied.
    ///
    /// # Errors
    ///
    /// Returns an error if `need` exceeds the layout's maximum or a page cannot be allocated.
    fn ensure_pages(&mut self, need: u32) -> Result<(), String> {
        if (need as u64) > self.layout.max_slots() {
            return Err(format!(
                "brick pool limit exceeded (need {need}, limit {} bricks)",
                self.layout.max_slots()
            ));
        }
        while self.cap < need {
            let k = self.pages.len() as u32;
            let pg = Self::make_page(&self.ctx, &self.layout, k)?;
            self.pages.push(pg);
            self.cap += self.layout.page_slots(k);
            self.grows += 1;
        }
        Ok(())
    }

    /// Creates an initialized storage buffer, padded to at least 16 bytes.
    fn storage(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        let mut v;
        let contents = if data.len() < 16 {
            v = data.to_vec();
            v.resize(16, 0);
            &v[..]
        } else {
            data
        };
        self.ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// Creates an initialized uniform buffer.
    fn uniform(&self, data: &[u8]) -> wgpu::Buffer {
        self.ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: data,
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Fails if a buffer of `bytes` would exceed the device's maximum binding size.
    fn check(&self, what: &str, bytes: u64) -> Result<(), String> {
        if bytes > self.ctx.max_binding {
            return Err(format!(
                "GPU buffer limit exceeded ({what} {bytes} B, limit {} B)",
                self.ctx.max_binding
            ));
        }
        Ok(())
    }

    /// Returns the layer for `key`, creating an empty preview layer if needed.
    fn layer_mut(&mut self, key: LayerKey) -> &mut Layer {
        match key {
            LayerKey::Base => &mut self.base,
            LayerKey::Pending(s) => self.pending.entry(s).or_default(),
        }
    }

    /// Submits `enc` and reads back the first `bytes` of `staging`.
    ///
    /// Must be called with the device lock held.
    fn submit_read(
        &self,
        enc: wgpu::CommandEncoder,
        staging: &wgpu::Buffer,
        bytes: u64,
    ) -> Result<Vec<u8>, String> {
        self.ctx.queue.submit(Some(enc.finish()));
        if bytes == 0 {
            return Ok(Vec::new());
        }
        let slice = staging.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("GPU wait failed: {e}"))?;
        rx.recv()
            .map_err(|e| format!("{e}"))?
            .map_err(|e| format!("failed to map GPU result: {e}"))?;
        let v = slice
            .get_mapped_range()
            .map_err(|e| format!("{e:?}"))?
            .to_vec();
        staging.unmap();
        Ok(v)
    }

    /// Accumulates `bins` into `layer`, returning ranges in the order of the CPU block lists.
    ///
    /// The lists are rebuilt internally from the bins, so `lists` (the CPU mesher's per-block bin
    /// lists) only determines the order of the result.
    ///
    /// # Returns
    ///
    /// For each entry of `lists`, the local range `(min, max)` changed by this call; `min > max`
    /// when no voxel changed.
    ///
    /// # Errors
    ///
    /// See [`GpuField::integrate_bins`].
    pub fn integrate(
        &mut self,
        layer: LayerKey,
        bins: &[Bin],
        lists: &[(BlockId, Vec<u32>)],
    ) -> Result<Vec<LocalRange>, String> {
        let empty = ([i32::MAX; 3], [i32::MIN; 3]);
        let got = self.integrate_bins(layer, bins, || {})?;
        let map: FxHashMap<BlockId, LocalRange> =
            got.into_iter().map(|(id, mn, mx)| (id, (mn, mx))).collect();
        Ok(lists
            .iter()
            .map(|(id, _)| map.get(id).copied().unwrap_or(empty))
            .collect())
    }

    /// Accumulates `bins` into `layer`.
    ///
    /// Lists are built per chunk of bins in parallel, without per-block lists. Large inputs are split
    /// by bin index into a few batches: while the GPU accumulates batch `k`, the CPU builds the lists
    /// for batch `k + 1`. Each voxel adds batches in bin order, so field values and ranges are
    /// bitwise identical to a single submission.
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer to accumulate into.
    /// * `bins` - Bins with positions, oriented normals and colours.
    /// * `during` - CPU work to run while the GPU accumulates. Not called if the call fails before
    ///   that point.
    ///
    /// # Returns
    ///
    /// The blocks whose bricks were touched, in first-seen input order (deterministic), with the
    /// local range changed by this call; `min > max` when no voxel changed.
    ///
    /// # Errors
    ///
    /// Returns an error when a bin range does not fit the packed format, a buffer or pool limit is
    /// exceeded, or a GPU operation fails. If the first batch fails before submission the state is
    /// unchanged; if a later batch fails, earlier batches remain applied.
    pub fn integrate_bins(
        &mut self,
        layer: LayerKey,
        bins: &[Bin],
        during: impl FnOnce(),
    ) -> Result<BlockRanges, String> {
        let n = (bins.len() / BATCH_MIN).clamp(1, MAX_BATCH);
        self.integrate_split(layer, bins, n, during)
    }

    /// Same as [`GpuField::integrate_bins`] with an explicit batch count; used by tests.
    ///
    /// Before building lists, the next pool page is prefetched if headroom is low. The GPU is idle
    /// during list building, so the zero-fill of the new page does not delay the accumulation
    /// kernel. A prefetch failure is only logged; the page is allocated again when needed.
    ///
    /// # Errors
    ///
    /// See [`GpuField::integrate_bins`].
    #[doc(hidden)]
    pub fn integrate_split(
        &mut self,
        layer: LayerKey,
        bins: &[Bin],
        nbatch: usize,
        during: impl FnOnce(),
    ) -> Result<BlockRanges, String> {
        let mut tm = FieldTimes::default();
        if bins.is_empty() {
            self.times = tm;
            return Ok(Vec::new());
        }
        let bins_bytes = (bins.len() * std::mem::size_of::<BinG>()) as u64;
        self.check("bins", bins_bytes)?;
        let ctx = self.ctx.clone();
        let pre = {
            let _g = ctx.lock.lock().unwrap();
            self.prefetch_page()
        };
        if let Err(e) = pre {
            eprintln!("pool page prefetch failed (retried when needed): {e}");
        }
        let live0 = self.live;
        let nbatch = nbatch.clamp(1, bins.len());
        while self.scratch.len() < nbatch {
            self.scratch.push(Scratch::default());
        }
        self.scratch[0].bins.ensure_dst(&ctx.device, bins_bytes);
        let mut sent: Vec<Sent> = Vec::new();
        for b in 0..nbatch {
            let (s, e) = (bins.len() * b / nbatch, bins.len() * (b + 1) / nbatch);
            match self.send_batch(layer, bins, s..e, b, &mut tm) {
                Ok(Some(x)) => sent.push(x),
                Ok(None) => {}
                Err(e) => {
                    if !sent.is_empty() {
                        let _ = self.finish(layer, sent, &mut tm);
                    }
                    self.cleanup_new(layer, &[]);
                    self.times = tm;
                    return Err(e);
                }
            }
        }
        let td = Instant::now();
        during();
        tm.during = ms(td);
        let t3 = Instant::now();
        let out = self.finish(layer, sent, &mut tm);
        tm.gpu += ms(t3);
        self.cleanup_new(layer, &[]);
        self.times = tm;
        self.used_max = self.used_max.max(self.live.saturating_sub(live0) as u32);
        out
    }

    /// Builds the lists for one batch, assigns slots and submits the batch without waiting.
    ///
    /// All batches share one GPU bin buffer (`scratch[0].bins.dst`); each batch uploads its bins at
    /// its own offset, so bin indices in the lists are global. When an earlier batch has already been
    /// submitted and a later one fails, the caller still finishes the submitted batches, because
    /// their contributions are already in the pool.
    ///
    /// The steps are:
    ///
    /// 1. Map the bin upload buffer, then per chunk pack bins and count (block, brick) hits.
    /// 2. Order blocks by first appearance across chunks, sum the counts per block and turn the
    ///    per-chunk counts into write positions (in parallel per block; blocks write disjoint cells).
    /// 3. Assign slots and append tiles. A fresh brick emits all eight sub-bricks as tiles (those
    ///    outside the range box get an empty list) so the whole brick is written from zero; an
    ///    existing brick only emits sub-bricks its range boxes reach.
    /// 4. Upload the lists and submit. Fallible work (allocating pages) happens before mapping;
    ///    existing pages are never moved. The index list is filled in ordinary memory by replaying the
    ///    cells recorded in step 1 (second pass of the counting sort) and then copied sequentially,
    ///    because mapped memory is write-combined and slow for scattered writes.
    ///
    /// # Returns
    ///
    /// The submitted batch, or `None` when it produced no tiles.
    ///
    /// # Errors
    ///
    /// Returns an error if a bin range cannot be packed, a bin reaches too many bricks, a limit is
    /// exceeded or a GPU operation fails. Slots assigned in this batch are returned and remaining
    /// mappings are released before returning.
    fn send_batch(
        &mut self,
        layer: LayerKey,
        all: &[Bin],
        rng: std::ops::Range<usize>,
        set: usize,
        tm: &mut FieldTimes,
    ) -> Result<Option<Sent>, String> {
        let t0 = Instant::now();
        let bins = &all[rng.clone()];
        let base = rng.start;
        let cfg = self.cfg;
        let v = cfg.voxel;
        let dim = cfg.block_dim;
        let nb = self.nb;
        let nk = (nb * nb * nb) as usize;
        let r = cfg.splat_radius;
        let r2 = r * r;
        let inv2s2 = 1.0 / (2.0 * (r * 0.5) * (r * 0.5));
        let range = |p: f32| {
            (
                (((p - r) / v).ceil()) as i32,
                (((p + r) / v).floor()) as i32,
            )
        };
        let r2m = r2 * (1.0 + 1e-4) + 1e-12;
        let bins_bytes = (bins.len() * std::mem::size_of::<BinG>()) as u64;
        let dev = self.ctx.device.clone();
        self.scratch[set].bins.ensure_st(&dev, bins_bytes);
        let st_bins = self.scratch[set].bins.st.clone().unwrap();
        map_write(&dev, &[(&st_bins, bins_bytes)])?;
        let nch = rayon::current_num_threads().max(1) * 4;
        let ch = bins.len().div_ceil(nch).max(4096);
        let bad = std::sync::atomic::AtomicBool::new(false);
        let many = std::sync::atomic::AtomicBool::new(false);
        let mut parts: Vec<Part> = {
            let mut view = st_bins
                .slice(0..bins_bytes.max(4))
                .get_mapped_range_mut()
                .map_err(|e| format!("{e:?}"))?;
            let mut out = view.slice(..bins_bytes as usize);
            let dst = SendPtr(out.as_raw_element_ptr().as_ptr() as *mut BinG);
            assert!(dst.0.is_aligned());
            let pk = |(l, h): (i32, i32)| {
                let len = (h - l).max(0) as u32;
                if !(-(1 << 23)..(1 << 23)).contains(&l) || len > 255 {
                    return None;
                }
                Some(((l as u32) & 0xff_ffff) | (len << 24))
            };
            bins.par_chunks(ch)
                .enumerate()
                .map(|(ci, chunk)| {
                    let dst = &dst;
                    let mut part = Part::default();
                    let mut rc = RangeCache::default();
                    for (j, b) in chunk.iter().enumerate() {
                        let rr = [range(b.pos[0]), range(b.pos[1]), range(b.pos[2])];
                        let pr = [pk(rr[0]), pk(rr[1]), pk(rr[2])];
                        if pr.iter().any(|x| x.is_none()) {
                            bad.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        let g = BinG {
                            p: b.pos,
                            rx: pr[0].unwrap_or(0),
                            n: b.normal,
                            ry: pr[1].unwrap_or(0),
                            c: b.rgb,
                            rz: pr[2].unwrap_or(0),
                        };
                        // SAFETY: `BinG` is Pod and the pointer is aligned (asserted above). Index
                        // `ci * ch + j` is below `bins.len()` and chunks write disjoint indices.
                        unsafe { dst.0.add(ci * ch + j).write(g) };
                        let e0 = part.em.len();
                        for &(id, k) in rc.get(&rr, dim, |id| part.local(id, nk)) {
                            let cnt = &mut part.cnt[k * nk..(k + 1) * nk];
                            let sub = &mut part.sub[k * nk..(k + 1) * nk];
                            let em = &mut part.em;
                            for_bricks(
                                [id[0] * dim, id[1] * dim, id[2] * dim],
                                &rr,
                                b.pos,
                                dim,
                                nb,
                                v,
                                r2m,
                                |key, m| {
                                    cnt[key] += 1;
                                    sub[key] |= m;
                                    em.push((k * nk + key) as u32);
                                },
                            );
                        }
                        let ne = part.em.len() - e0;
                        if ne > u16::MAX as usize {
                            many.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        part.per_bin.push(ne as u16);
                    }
                    part
                })
                .collect()
        };
        let (bad, many) = (bad.into_inner(), many.into_inner());
        if bad || many {
            st_bins.unmap();
            return Err(if many {
                "a single bin reaches too many bricks".into()
            } else {
                "bin range does not fit a 24-bit start and 8-bit length".into()
            });
        }

        let mut gidx: FxHashMap<BlockId, usize> = FxHashMap::default();
        let mut order: Vec<BlockId> = Vec::new();
        let mut owners: Vec<Vec<(u32, u32)>> = Vec::new();
        for (ci, p) in parts.iter().enumerate() {
            for (li, id) in p.ids.iter().enumerate() {
                let g = *gidx.entry(*id).or_insert_with(|| {
                    order.push(*id);
                    owners.push(Vec::new());
                    order.len() - 1
                });
                owners[g].push((ci as u32, li as u32));
            }
        }
        drop(gidx);
        let per: Vec<BrickKeys> = owners
            .par_iter()
            .map(|ow| {
                let mut tot = vec![0u32; nk];
                let mut sub = vec![0u8; nk];
                for &(ci, li) in ow {
                    let r = li as usize * nk..(li as usize + 1) * nk;
                    let p = &parts[ci as usize];
                    for ((t, m), (&x, &y)) in tot
                        .iter_mut()
                        .zip(sub.iter_mut())
                        .zip(p.cnt[r.clone()].iter().zip(&p.sub[r]))
                    {
                        *t += x;
                        *m |= y;
                    }
                }
                let keys: Vec<(u32, u32, u8)> = tot
                    .iter()
                    .zip(&sub)
                    .enumerate()
                    .filter(|x| *x.1.0 > 0)
                    .map(|(k, (&c, &m))| (k as u32, c, m))
                    .collect();
                let n = keys.iter().map(|k| k.1).sum();
                (keys, n)
            })
            .collect();
        let mut bstart: Vec<u32> = Vec::with_capacity(per.len());
        let mut acc = 0u64;
        for p in &per {
            bstart.push(acc as u32);
            acc += p.1 as u64;
        }
        let total_idx = acc;
        let idx_bytes = total_idx * 4;
        {
            let ptrs: Vec<SendPtr<u32>> = parts
                .iter_mut()
                .map(|p| SendPtr(p.cnt.as_mut_ptr()))
                .collect();
            owners
                .par_iter()
                .zip(per.par_iter())
                .zip(bstart.par_iter())
                .for_each(|((ow, (keys, _)), &s0)| {
                    let ptrs = &ptrs;
                    let mut s = s0;
                    for &(k, _, _) in keys {
                        for &(ci, li) in ow {
                            // SAFETY: the (chunk, block) cells are written only by this block's
                            // task, and the index lies within that chunk's `cnt`.
                            let cell = unsafe {
                                &mut *ptrs[ci as usize].0.add(li as usize * nk + k as usize)
                            };
                            let x = *cell;
                            *cell = s;
                            s += x;
                        }
                    }
                });
        }

        let mut tiles: Vec<Tile> = Vec::new();
        let mut bricks: Vec<(u32, u32, u32, u32)> = Vec::new();
        let mut gmin = [i32::MAX; 3];
        let mut gmax = [i32::MIN; 3];
        let mut new_blocks: Vec<BlockId> = Vec::new();
        {
            let (free, high, live) = (&mut self.free, &mut self.high, &mut self.live);
            let lay: &mut Layer = match layer {
                LayerKey::Base => &mut self.base,
                LayerKey::Pending(s) => self.pending.entry(s).or_default(),
            };
            for (j, (id, (keys, _))) in order.iter().zip(&per).enumerate() {
                if keys.is_empty() {
                    continue;
                }
                let lb = lay.entry(*id).or_insert_with(|| {
                    new_blocks.push(*id);
                    LBlock {
                        slots: vec![NONE; nk],
                        min: [i32::MAX; 3],
                        max: [i32::MIN; 3],
                    }
                });
                let mut off = 0u32;
                for &(k, n, sm) in keys {
                    let kk = k as i32;
                    let bo = [
                        id[0] * dim + (kk % nb) * BR,
                        id[1] * dim + ((kk / nb) % nb) * BR,
                        id[2] * dim + (kk / (nb * nb)) * BR,
                    ];
                    let s = &mut lb.slots[k as usize];
                    let fresh = *s == NONE;
                    if fresh {
                        *s = free.pop().unwrap_or_else(|| {
                            *high += 1;
                            *high - 1
                        });
                        *live += 1;
                    }
                    let t0 = tiles.len() as u32;
                    for sub in 0..8u32 {
                        let on = sm >> sub & 1 != 0;
                        if !on && !fresh {
                            continue;
                        }
                        let so = [
                            bo[0] + (sub & 1) as i32 * SB,
                            bo[1] + ((sub >> 1) & 1) as i32 * SB,
                            bo[2] + (sub >> 2) as i32 * SB,
                        ];
                        tiles.push(Tile {
                            o: so,
                            start: bstart[j] + off,
                            count: if on { n } else { 0 },
                            slot: *s,
                            fresh: fresh as u32,
                            sub,
                        });
                    }
                    off += n;
                    bricks.push((j as u32, k, t0, tiles.len() as u32));
                    for a in 0..3 {
                        gmin[a] = gmin[a].min(bo[a]);
                        gmax[a] = gmax[a].max(bo[a] + BR - 1);
                    }
                }
            }
        }
        tm.bricks += ms(t0);
        if tiles.is_empty() {
            st_bins.unmap();
            self.cleanup_new(layer, &new_blocks);
            return Ok(None);
        }
        let mut ups_mapped = false;
        let mut bins_mapped = true;
        let res = (|| -> Result<u64, String> {
            let t1 = Instant::now();
            let mut coords: Vec<f32> = Vec::new();
            let mut offs = [0u32; 3];
            for a in 0..3 {
                offs[a] = coords.len() as u32;
                coords.extend((gmin[a]..=gmax[a]).map(|g| g as f32 * v));
            }
            let ntiles = tiles.len() as u32;
            self.check("index list", idx_bytes)?;
            self.check("tiles", (tiles.len() * 32) as u64)?;
            let nwg = ntiles as u64;
            if nwg > MAX_WG as u64 * MAX_WG as u64 {
                return Err("workgroup count limit exceeded".into());
            }
            self.ensure_pages(self.high)?;
            let (gx, gy) = grid(nwg as u32);
            let prm = SplatParams {
                r2,
                inv2s2,
                ntiles,
                nx: gx,
                gx0: gmin[0],
                gy0: gmin[1],
                gz0: gmin[2],
                zero: 0,
                offy: offs[1],
                offz: offs[2],
                r2m,
                bsh: self.layout.bsh,
                msh: self.layout.msh,
                pad: [0; 3],
            };
            tm.pack += ms(t1);
            let t2 = Instant::now();
            let b_par = self.uniform(bytemuck::bytes_of(&prm));
            let mbytes = (ntiles as u64 * 4).max(16);
            let tiles_bytes = (tiles.len() * std::mem::size_of::<Tile>()) as u64;
            let coords_bytes = (coords.len() * 4) as u64;
            let sc = &mut self.scratch[set];
            for (u, n) in [
                (&mut sc.idx, idx_bytes),
                (&mut sc.tiles, tiles_bytes),
                (&mut sc.coords, coords_bytes),
                (&mut sc.masks, mbytes),
            ] {
                u.ensure(&dev, n);
            }
            let st_ups = [
                sc.idx.st.clone().unwrap(),
                sc.tiles.st.clone().unwrap(),
                sc.coords.st.clone().unwrap(),
            ];
            let ups = [
                (&st_ups[0], idx_bytes),
                (&st_ups[1], tiles_bytes),
                (&st_ups[2], coords_bytes),
            ];
            map_write(&dev, &ups)?;
            ups_mapped = true;
            let sc = &self.scratch[set];
            {
                let mut idx: Vec<u32> = Vec::with_capacity(total_idx as usize);
                let dst = SendPtr(idx.as_mut_ptr());
                parts.par_iter_mut().enumerate().for_each(|(ci, part)| {
                    let dst = &dst;
                    let cnt = &mut part.cnt;
                    let mut e = part.em.iter();
                    for (j, &ne) in part.per_bin.iter().enumerate() {
                        let i = (base + ci * ch + j) as u32;
                        for &c in e.by_ref().take(ne as usize) {
                            let pos = &mut cnt[c as usize];
                            // SAFETY: the position lies in the range reserved for this
                            // (chunk, block, brick), which no other writer touches, and is below
                            // `total_idx`, the capacity of `idx`.
                            unsafe { dst.0.add(*pos as usize).write(i) };
                            *pos += 1;
                        }
                    }
                });
                // SAFETY: the counting sort above wrote every index in `0..total_idx` exactly once.
                unsafe { idx.set_len(total_idx as usize) };
                let st = sc.idx.st.as_ref().unwrap();
                let mut view = st
                    .slice(0..idx_bytes.max(4))
                    .get_mapped_range_mut()
                    .map_err(|e| format!("{e:?}"))?;
                let mut out = view.slice(..idx_bytes as usize);
                par_copy(&mut out, bytemuck::cast_slice(&idx));
            }
            for (u, data) in [
                (&sc.tiles, bytemuck::cast_slice::<Tile, u8>(&tiles)),
                (&sc.coords, bytemuck::cast_slice::<f32, u8>(&coords)),
            ] {
                let st = u.st.as_ref().unwrap();
                let mut view = st
                    .slice(0..(data.len() as u64).max(4))
                    .get_mapped_range_mut()
                    .map_err(|e| format!("{e:?}"))?;
                view.slice(..data.len()).copy_from_slice(data);
            }
            st_bins.unmap();
            for (u, _) in ups {
                u.unmap();
            }
            ups_mapped = false;
            bins_mapped = false;
            let b_bins = self.scratch[0].bins.dst.as_ref().unwrap();
            let b_idx = sc.idx.dst.as_ref().unwrap();
            let b_tiles = sc.tiles.dst.as_ref().unwrap();
            let b_coords = sc.coords.dst.as_ref().unwrap();
            let b_masks = sc.masks.dst.as_ref().unwrap();
            let staging = sc.masks.st.as_ref().unwrap();
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_par.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_bins.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_idx.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_tiles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: b_coords.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: b_masks.as_entire_binding(),
                },
            ];
            entries.extend(pool_entries(&self.pages, &self.dummy, self.layout.n));
            let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("field_splat"),
                layout: &self.p_splat.get_bind_group_layout(0),
                entries: &entries,
            });
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("field_splat"),
            });
            enc.copy_buffer_to_buffer(
                &st_bins,
                0,
                b_bins,
                base as u64 * std::mem::size_of::<BinG>() as u64,
                bins_bytes,
            );
            for (u, n) in [
                (&sc.idx, idx_bytes),
                (&sc.tiles, tiles_bytes),
                (&sc.coords, coords_bytes),
            ] {
                if n > 0 {
                    enc.copy_buffer_to_buffer(
                        u.st.as_ref().unwrap(),
                        0,
                        u.dst.as_ref().unwrap(),
                        0,
                        n,
                    );
                }
            }
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("field_splat"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.p_splat);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            enc.copy_buffer_to_buffer(b_masks, 0, staging, 0, mbytes);
            {
                let _g = self.ctx.lock.lock().unwrap();
                self.ctx.queue.submit(Some(enc.finish()));
            }
            tm.upload += ms(t2);
            Ok(mbytes)
        })();
        match res {
            Ok(mbytes) => Ok(Some(Sent {
                order,
                bricks,
                tiles,
                new_blocks,
                set,
                mbytes,
            })),
            Err(e) => {
                if bins_mapped {
                    st_bins.unmap();
                }
                if ups_mapped {
                    let sc = &self.scratch[set];
                    for u in [&sc.idx, &sc.tiles, &sc.coords] {
                        u.st.as_ref().unwrap().unmap();
                    }
                }
                self.rollback(layer, &order, &bricks, &tiles, &new_blocks);
                Err(e)
            }
        }
    }

    /// Reads back the masks of the submitted batches, builds block ranges and releases fresh bricks
    /// that never changed.
    ///
    /// Waits for the GPU once. A brick's mask is the OR over all batches, because a brick assigned
    /// in an earlier batch may only change in a later one. Fresh base bricks that did change are
    /// recorded for the incremental hash table update.
    ///
    /// # Errors
    ///
    /// Returns an error if waiting for the GPU or mapping a result buffer fails.
    fn finish(
        &mut self,
        layer: LayerKey,
        sent: Vec<Sent>,
        tm: &mut FieldTimes,
    ) -> Result<BlockRanges, String> {
        let raws: Vec<Vec<u8>> = {
            let _g = self.ctx.lock.lock().unwrap();
            let stagings: Vec<(wgpu::Buffer, u64)> = sent
                .iter()
                .map(|s| (self.scratch[s.set].masks.st.clone().unwrap(), s.mbytes))
                .collect();
            let (tx, rx) = std::sync::mpsc::channel();
            for (i, (b, n)) in stagings.iter().enumerate() {
                let tx = tx.clone();
                b.slice(0..*n).map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send((i, r));
                });
            }
            drop(tx);
            self.ctx
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| format!("GPU wait failed: {e}"))?;
            let mut ok = vec![false; stagings.len()];
            let mut err = None;
            for (i, r) in rx.iter() {
                match r {
                    Ok(()) => ok[i] = true,
                    Err(e) => err = Some(format!("failed to map GPU result: {e}")),
                }
            }
            let mut raws = Vec::with_capacity(stagings.len());
            for (i, (b, n)) in stagings.iter().enumerate() {
                if ok[i] {
                    raws.push(
                        b.slice(0..*n)
                            .get_mapped_range()
                            .map_err(|e| format!("{e:?}"))?
                            .to_vec(),
                    );
                    b.unmap();
                }
            }
            if let Some(e) = err {
                return Err(e);
            }
            raws
        };
        let t4 = Instant::now();
        if layer == LayerKey::Base {
            self.base_ver += 1;
        }
        let nb = self.nb;
        let mut bidx: FxHashMap<BlockId, usize> = FxHashMap::default();
        let mut out: BlockRanges = Vec::new();
        let mut kidx: FxHashMap<(usize, u32), usize> = FxHashMap::default();
        let mut agg: Vec<(usize, u32, u32, u32)> = Vec::new();
        for (s, raw) in sent.iter().zip(&raws) {
            let masks: &[u32] = bytemuck::cast_slice(raw);
            let loc: Vec<usize> = s
                .order
                .iter()
                .map(|id| {
                    *bidx.entry(*id).or_insert_with(|| {
                        out.push((*id, [i32::MAX; 3], [i32::MIN; 3]));
                        out.len() - 1
                    })
                })
                .collect();
            for &(j, k, t0, t1) in &s.bricks {
                let m = masks[t0 as usize..t1 as usize]
                    .iter()
                    .fold(0u32, |a, &b| a | b);
                let tile = &s.tiles[t0 as usize];
                let e = *kidx.entry((loc[j as usize], k)).or_insert_with(|| {
                    agg.push((loc[j as usize], k, 0, NONE));
                    agg.len() - 1
                });
                agg[e].2 |= m;
                if tile.fresh != 0 {
                    agg[e].3 = tile.slot;
                }
            }
        }
        let (free, live) = (&mut self.free, &mut self.live);
        let lay: &mut Layer = match layer {
            LayerKey::Base => &mut self.base,
            LayerKey::Pending(s) => self.pending.entry(s).or_default(),
        };
        let hs = self.hash.get_mut().unwrap();
        let track = layer == LayerKey::Base && hs.cache.is_some();
        for &(b, k, m, fresh_slot) in &agg {
            let id = out[b].0;
            if m == 0 {
                if fresh_slot != NONE {
                    lay.get_mut(&id).unwrap().slots[k as usize] = NONE;
                    free.push(fresh_slot);
                    *live -= 1;
                }
                continue;
            }
            let kk = k as i32;
            if track && fresh_slot != NONE {
                hs.added.push([
                    id[0] * nb + kk % nb,
                    id[1] * nb + (kk / nb) % nb,
                    id[2] * nb + kk / (nb * nb),
                    fresh_slot as i32,
                ]);
            }
            let bo = [(kk % nb) * BR, ((kk / nb) % nb) * BR, (kk / (nb * nb)) * BR];
            let (_, mn, mx) = &mut out[b];
            for a in 0..3 {
                let bits = (m >> (8 * a)) & 0xff;
                mn[a] = mn[a].min(bo[a] + bits.trailing_zeros() as i32);
                mx[a] = mx[a].max(bo[a] + 31 - bits.leading_zeros() as i32);
            }
        }
        for (id, mn, mx) in &out {
            if mn[0] > mx[0] {
                continue;
            }
            if let Some(lb) = lay.get_mut(id) {
                for a in 0..3 {
                    lb.min[a] = lb.min[a].min(mn[a]);
                    lb.max[a] = lb.max[a].max(mx[a]);
                }
            }
        }
        for s in &sent {
            self.cleanup_new(layer, &s.new_blocks);
        }
        tm.post += ms(t4);
        Ok(out)
    }

    /// Returns the slots assigned to fresh bricks of a batch that failed before submission.
    fn rollback(
        &mut self,
        layer: LayerKey,
        order: &[BlockId],
        bricks: &[(u32, u32, u32, u32)],
        tiles: &[Tile],
        new_blocks: &[BlockId],
    ) {
        let (free, live) = (&mut self.free, &mut self.live);
        let lay: &mut Layer = match layer {
            LayerKey::Base => &mut self.base,
            LayerKey::Pending(s) => self.pending.entry(s).or_default(),
        };
        for &(j, k, t0, _) in bricks {
            let t = &tiles[t0 as usize];
            if t.fresh != 0 {
                lay.get_mut(&order[j as usize]).unwrap().slots[k as usize] = NONE;
                free.push(t.slot);
                *live -= 1;
            }
        }
        self.cleanup_new(layer, new_blocks);
    }

    /// Removes blocks created in this call that ended up with no bricks, and drops an empty preview
    /// layer.
    fn cleanup_new(&mut self, layer: LayerKey, new_blocks: &[BlockId]) {
        let lay = self.layer_mut(layer);
        for id in new_blocks {
            if lay
                .get(id)
                .is_some_and(|b| b.slots.iter().all(|&s| s == NONE))
            {
                lay.remove(id);
            }
        }
        if let LayerKey::Pending(s) = layer {
            if self.pending.get(&s).is_some_and(|l| l.is_empty()) {
                self.pending.remove(&s);
            }
        }
    }

    /// Drops the preview layer of segment `seg` and returns its slots to the free list.
    ///
    /// # Returns
    ///
    /// The blocks that changed in that layer with their accumulated local ranges, sorted by id.
    pub fn drop_layer(&mut self, seg: SegmentId) -> Vec<(BlockId, [i32; 3], [i32; 3])> {
        let Some(layer) = self.pending.remove(&seg) else {
            return Vec::new();
        };
        let mut blocks: Vec<(BlockId, LBlock)> = layer.into_iter().collect();
        blocks.sort_unstable_by_key(|b| b.0);
        let mut out = Vec::with_capacity(blocks.len());
        for (id, lb) in blocks {
            for &s in &lb.slots {
                if s != NONE {
                    self.free.push(s);
                    self.live -= 1;
                }
            }
            if lb.min[0] <= lb.max[0] {
                out.push((id, lb.min, lb.max));
            }
        }
        out
    }

    /// Returns the size of the compact gather output for `nblocks` blocks:
    /// `nblocks × (occupancy header + (dim + 2)³) × 4` bytes.
    pub fn gather_bytes(&self, nblocks: usize) -> u64 {
        (nblocks as u64 * crate::gpu_extract::volume_words(self.cfg.block_dim) * 4).max(32)
    }

    /// Returns the size of the raw gather output for `nblocks` blocks: `nblocks × (dim + 2)³ × 32`
    /// bytes.
    pub fn gather_raw_bytes(&self, nblocks: usize) -> u64 {
        let p = (self.cfg.block_dim + 2) as u64;
        (nblocks as u64 * p * p * p * 32).max(32)
    }

    /// Gathers the layer-summed distance volumes of blocks `ids` in the compact layout.
    ///
    /// Layers are summed base first, then preview layers in segment order. The result also carries
    /// the pool pages and per-region slot table, from which extraction reads colour. The work is
    /// submitted without waiting; later work on the same queue sees the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the output exceeds a buffer or dispatch limit.
    pub fn gather(&self, ids: &[BlockId]) -> Result<Volumes, String> {
        let out = self.volume_buffer(ids.len())?;
        self.gather_into(ids, &out)
    }

    /// Gathers raw layer sums of blocks `ids`: eight f32 per padded voxel (Σw·d, Σw, Σw·r, Σw·g,
    /// Σw·b, 0, 0, 0).
    ///
    /// Used to compare against the CPU padded volume in tests and diagnostics.
    ///
    /// # Errors
    ///
    /// Returns an error if the output exceeds a buffer or dispatch limit.
    pub fn gather_raw(&self, ids: &[BlockId]) -> Result<wgpu::Buffer, String> {
        let out_bytes = self.gather_raw_bytes(ids.len());
        let out = self.make_volume(out_bytes)?;
        self.gather_with(ids, &out, true)?;
        Ok(out)
    }

    /// Creates a compact gather output buffer for `nblocks` blocks, for reuse with
    /// [`GpuField::gather_into`].
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer would exceed the binding size limit.
    pub fn volume_buffer(&self, nblocks: usize) -> Result<wgpu::Buffer, String> {
        self.make_volume(self.gather_bytes(nblocks))
    }

    /// Creates a gather output buffer of `out_bytes`.
    fn make_volume(&self, out_bytes: u64) -> Result<wgpu::Buffer, String> {
        self.check("gather output", out_bytes)?;
        Ok(self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field_volume"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }

    /// Same as [`GpuField::gather`], but writes into a caller-provided buffer.
    ///
    /// `out` must be a storage buffer of at least [`GpuField::gather_bytes`]. Reusing it avoids
    /// allocating and zero-filling a large buffer on every call.
    ///
    /// # Errors
    ///
    /// Returns an error if `out` is too small or a limit is exceeded.
    pub fn gather_into(&self, ids: &[BlockId], out: &wgpu::Buffer) -> Result<Volumes, String> {
        let (off, slots) = self.gather_with(ids, out, false)?;
        Ok(Volumes {
            vol: out.clone(),
            pool: self.pages.clone(),
            layout: self.layout,
            off,
            slots,
        })
    }

    /// Builds the per-region slot table and dispatches a gather kernel into `out`.
    ///
    /// For every block and region, the slots of all layers (base first) that cover the region are
    /// listed; `off` holds the start offset of each region (a prefix sum, one entry per region plus
    /// a terminator). The compact layout additionally writes per-block occupancy bits (region index
    /// `(rz·nr + ry)·nr + rx`) and dispatches one workgroup per occupied region
    /// (`o = block · nr³ + region`); the raw layout dispatches one workgroup per 64 voxels.
    ///
    /// # Returns
    ///
    /// The `(off, slots)` GPU buffers of the slot table.
    ///
    /// # Errors
    ///
    /// Returns an error if `out` is too small, the output is not addressable with u32 indices in
    /// the shader, or the dispatch exceeds the workgroup limit.
    fn gather_with(
        &self,
        ids: &[BlockId],
        out: &wgpu::Buffer,
        raw: bool,
    ) -> Result<(wgpu::Buffer, wgpu::Buffer), String> {
        let dim = self.cfg.block_dim;
        let nb = self.nb;
        let nr = nb + 2;
        let nr3 = (nr * nr * nr) as usize;
        let p = (dim + 2) as u64;
        let out_bytes = if raw {
            self.gather_raw_bytes(ids.len())
        } else {
            self.gather_bytes(ids.len())
        };
        if out.size() < out_bytes {
            return Err(format!(
                "gather output buffer too small ({} B < {out_bytes} B)",
                out.size()
            ));
        }
        if out_bytes / 4 > u32::MAX as u64 {
            return Err("gather output exceeds the u32 index range".into());
        }
        if ids.is_empty() {
            return Ok((self.storage("off", &[0; 4]), self.storage("slots", &[0; 4])));
        }
        let layers: Vec<&Layer> = std::iter::once(&self.base)
            .chain(self.pending.values())
            .collect();
        let per: Vec<(Vec<u32>, Vec<u32>)> = ids
            .par_iter()
            .map(|id| {
                let mut cnt = vec![0u32; nr3];
                let mut sl: Vec<u32> = Vec::new();
                let mut nbrs: Vec<[Option<&LBlock>; 27]> = Vec::with_capacity(layers.len());
                for l in &layers {
                    let mut a: [Option<&LBlock>; 27] = [None; 27];
                    let mut any = false;
                    for (i, s) in a.iter_mut().enumerate() {
                        let i = i as i32;
                        *s = l.get(&[
                            id[0] + i % 3 - 1,
                            id[1] + (i / 3) % 3 - 1,
                            id[2] + i / 9 - 1,
                        ]);
                        any |= s.is_some();
                    }
                    if any {
                        nbrs.push(a);
                    }
                }
                let ax = |r: i32| {
                    if r == 0 {
                        (0, nb - 1)
                    } else if r == nb + 1 {
                        (2, 0)
                    } else {
                        (1, r - 1)
                    }
                };
                for rz in 0..nr {
                    let (oz, bz) = ax(rz);
                    for ry in 0..nr {
                        let (oy, by) = ax(ry);
                        for rx in 0..nr {
                            let (ox, bx) = ax(rx);
                            let ri = ((rz * nr + ry) * nr + rx) as usize;
                            let ni = (oz * 9 + oy * 3 + ox) as usize;
                            let k = ((bz * nb + by) * nb + bx) as usize;
                            for a in &nbrs {
                                if let Some(b) = a[ni] {
                                    let s = b.slots[k];
                                    if s != NONE {
                                        sl.push(s);
                                        cnt[ri] += 1;
                                    }
                                }
                            }
                        }
                    }
                }
                (cnt, sl)
            })
            .collect();
        let mut off: Vec<u32> = Vec::with_capacity(ids.len() * nr3 + 1);
        let mut slots: Vec<u32> = Vec::with_capacity(per.iter().map(|x| x.1.len()).sum());
        for (cnt, sl) in &per {
            let mut s = slots.len() as u32;
            for &c in cnt {
                off.push(s);
                s += c;
            }
            slots.extend_from_slice(sl);
        }
        off.push(slots.len() as u32);
        let dev = &self.ctx.device;
        let owp = crate::gpu_extract::occ_words(dim);
        let stride = crate::gpu_extract::volume_words(dim);
        let (occ, items): (Vec<u32>, Vec<u32>) = if raw {
            (vec![0; 4], vec![0; 4])
        } else {
            let mut o = vec![0u32; ids.len() * owp as usize];
            let mut it = Vec::new();
            for (j, (cnt, _)) in per.iter().enumerate() {
                for (r, &c) in cnt.iter().enumerate() {
                    if c > 0 {
                        o[j * owp as usize + r / 32] |= 1 << (r % 32);
                        it.push((j * nr3 + r) as u32);
                    }
                }
            }
            (o, it)
        };
        let nwg = if raw {
            (ids.len() as u64 * p * p * p).div_ceil(64)
        } else {
            items.len() as u64
        };
        if nwg > MAX_WG as u64 * MAX_WG as u64
            || (ids.len() as u64 * owp).div_ceil(64) > MAX_WG as u64
        {
            return Err("workgroup count limit exceeded".into());
        }
        let (gx, gy) = grid(nwg as u32);
        let prm = GatherParams {
            nblocks: ids.len() as u32,
            p: p as u32,
            nr: nr as u32,
            nx: gx,
            minw: self.cfg.min_weight,
            owp: owp as u32,
            stride: stride as u32,
            nitems: items.len() as u32,
            bsh: self.layout.bsh,
            msh: self.layout.msh,
            pad: [0; 2],
        };
        let _g = self.ctx.lock.lock().unwrap();
        let b_par = self.uniform(bytemuck::bytes_of(&prm));
        let b_off = self.storage("off", bytemuck::cast_slice(&off));
        let b_slots = self.storage("slots", bytemuck::cast_slice(&slots));
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("field_gather"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("field_gather"),
                timestamp_writes: None,
            });
            let mut common = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b_par.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_off.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_slots.as_entire_binding(),
                },
            ];
            common.extend(pool_entries(&self.pages, &self.dummy, self.layout.n));
            if raw {
                let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("field_gather_raw"),
                    layout: &self.p_gather_raw.get_bind_group_layout(0),
                    entries: &common,
                });
                pass.set_pipeline(&self.p_gather_raw);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            } else {
                let b_occ = self.storage("occ", bytemuck::cast_slice(&occ));
                let b_items = self.storage("items", bytemuck::cast_slice(&items));
                let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("field_gather_head"),
                    layout: &self.p_gather_head.get_bind_group_layout(0),
                    entries: &[
                        common[0].clone(),
                        common[1].clone(),
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: b_occ.as_entire_binding(),
                        },
                    ],
                });
                pass.set_pipeline(&self.p_gather_head);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups((ids.len() as u64 * owp).div_ceil(64) as u32, 1, 1);
                if !items.is_empty() {
                    let mut e = common.to_vec();
                    e.push(wgpu::BindGroupEntry {
                        binding: 6,
                        resource: b_items.as_entire_binding(),
                    });
                    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("field_gather"),
                        layout: &self.p_gather.get_bind_group_layout(0),
                        entries: &e,
                    });
                    pass.set_pipeline(&self.p_gather);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
            }
        }
        self.ctx.queue.submit(Some(enc.finish()));
        Ok((b_off, b_slots))
    }

    /// Builds the base layer hash table from brick coordinate to slot.
    ///
    /// Open addressing with linear probing; entries are `[bx, by, bz, slot]` and empty cells have
    /// slot -1. Blocks are inserted in sorted order so the table is deterministic.
    ///
    /// # Returns
    ///
    /// `(table, mask, entry count)`. The table size is a power of two of at least twice the entry
    /// count (minimum 16).
    fn base_table(&self) -> (Vec<[i32; 4]>, u32, usize) {
        let n: usize = self
            .base
            .values()
            .map(|b| b.slots.iter().filter(|&&s| s != NONE).count())
            .sum();
        let cap = (n * 2).next_power_of_two().max(16);
        let mask = (cap - 1) as u32;
        let mut t = vec![[0, 0, 0, -1i32]; cap];
        let nb = self.nb;
        let mut ids: Vec<&BlockId> = self.base.keys().collect();
        ids.sort_unstable();
        for id in ids {
            let lb = &self.base[id];
            for (k, &s) in lb.slots.iter().enumerate() {
                if s == NONE {
                    continue;
                }
                let k = k as i32;
                table_insert(
                    &mut t,
                    mask,
                    [
                        id[0] * nb + k % nb,
                        id[1] * nb + (k / nb) % nb,
                        id[2] * nb + k / (nb * nb),
                        s as i32,
                    ],
                );
            }
        }
        (t, mask, n)
    }

    /// Computes the base-layer agreement for points `p` with normals `n` on the GPU.
    ///
    /// Produces the same values as the CPU mesher's base agreement using only the base layer: the
    /// clamped difference of trilinear field samples at `p ± h·n`. The hash table is rebuilt when it
    /// does not exist or would become more than half full; otherwise only bricks added since the
    /// last call are inserted and the whole table is rewritten. Points are packed with rayon before
    /// any lock (hash table, device) is taken.
    ///
    /// # Returns
    ///
    /// One value per point, or `None` when any of the sampled corners is unobserved.
    ///
    /// # Errors
    ///
    /// Returns an error if a buffer limit is exceeded or a GPU operation fails.
    ///
    /// # Panics
    ///
    /// Panics if `p` and `n` differ in length.
    pub fn base_agreement(
        &self,
        p: &[[f32; 3]],
        n: &[[f32; 3]],
    ) -> Result<Vec<Option<f32>>, String> {
        assert_eq!(p.len(), n.len());
        if p.is_empty() {
            return Ok(Vec::new());
        }
        if self.base.is_empty() {
            return Ok(vec![None; p.len()]);
        }
        let pts: Vec<f32> = p
            .par_iter()
            .zip(n.par_iter())
            .flat_map_iter(|(a, b)| [a[0], a[1], a[2], b[0], b[1], b[2]])
            .collect();
        let dev = &self.ctx.device;
        let mut hs = self.hash.lock().unwrap();
        let hs = &mut *hs;
        match hs.cache.as_mut() {
            Some(c) if c.ver == self.base_ver => {}
            Some(c) if (c.n + hs.added.len()) * 2 <= c.table.len() => {
                for e in hs.added.drain(..) {
                    table_insert(&mut c.table, c.mask, e);
                    c.n += 1;
                }
                self.ctx
                    .queue
                    .write_buffer(&c.buf, 0, bytemuck::cast_slice(&c.table));
                c.ver = self.base_ver;
                hs.incr += 1;
            }
            _ => {
                let (t, mask, n) = self.base_table();
                self.check("hash table", (t.len() * 16) as u64)?;
                let buf = dev.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("base_hash"),
                    size: (t.len() * 16) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.ctx
                    .queue
                    .write_buffer(&buf, 0, bytemuck::cast_slice(&t));
                hs.added.clear();
                hs.cache = Some(HashCache {
                    ver: self.base_ver,
                    buf,
                    mask,
                    table: t,
                    n,
                });
            }
        }
        let hc = hs.cache.as_ref().unwrap();
        let pbytes = (p.len() * 24) as u64;
        self.check("points", pbytes)?;
        let np = p.len() as u32;
        let (gx, gy) = grid(np.div_ceil(64));
        let h = 1.5 * self.cfg.voxel;
        let prm = AgreeParams {
            npts: np,
            nx: gx,
            mask: hc.mask,
            zero: 0,
            h,
            h2: 2.0 * h,
            inv: 1.0 / self.cfg.voxel,
            minw: self.cfg.min_weight,
            bsh: self.layout.bsh,
            msh: self.layout.msh,
            pad: [0; 2],
        };
        let obytes = (np as u64 * 4).max(16);
        let _g = self.ctx.lock.lock().unwrap();
        let mut ab = self.agree.lock().unwrap();
        ab.pts.ensure(dev, pbytes);
        ab.out.ensure(dev, obytes);
        let b_par = self.uniform(bytemuck::bytes_of(&prm));
        {
            let st = ab.pts.st.as_ref().unwrap();
            map_write(dev, &[(st, pbytes)])?;
            {
                let mut view = st
                    .slice(0..pbytes.max(4))
                    .get_mapped_range_mut()
                    .map_err(|e| format!("{e:?}"))?;
                view.slice(..pbytes as usize)
                    .copy_from_slice(bytemuck::cast_slice(&pts));
            }
            st.unmap();
        }
        let (b_pts, b_out, staging) = (
            ab.pts.dst.as_ref().unwrap(),
            ab.out.dst.as_ref().unwrap(),
            ab.out.st.as_ref().unwrap(),
        );
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: b_par.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: hc.buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: b_pts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: b_out.as_entire_binding(),
            },
        ];
        entries.extend(pool_entries(&self.pages, &self.dummy, self.layout.n));
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("field_agree"),
            layout: &self.p_agree.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("field_agree"),
        });
        enc.copy_buffer_to_buffer(ab.pts.st.as_ref().unwrap(), 0, b_pts, 0, pbytes);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("field_agree"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.p_agree);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        enc.copy_buffer_to_buffer(b_out, 0, staging, 0, obytes);
        let raw = self.submit_read(enc, staging, obytes)?;
        let vals: &[f32] = bytemuck::cast_slice(&raw);
        Ok(vals[..p.len()]
            .iter()
            .map(|&x| (x != AGREE_NONE).then_some(x))
            .collect())
    }

    /// Returns the bytes allocated for the brick pool.
    pub fn field_bytes(&self) -> usize {
        (self.cap as u64 * BRICK_BYTES) as usize
    }

    /// Returns the GPU memory held by the pool and the integrate scratch buffers (staging buffers,
    /// GPU copies and masks). The hash table and gather outputs are not included.
    pub fn gpu_bytes(&self) -> usize {
        self.field_bytes() + self.scratch.iter().map(|s| s.bytes()).sum::<u64>() as usize
    }

    /// Returns `true` if the base layer has no blocks.
    pub fn base_is_empty(&self) -> bool {
        self.base.is_empty()
    }

    /// Grows the pool to at least `bricks` bricks ahead of time, so later steps do not pay for page
    /// allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the pool limit is exceeded or a page cannot be allocated.
    pub fn reserve(&mut self, bricks: u32) -> Result<(), String> {
        self.ensure_pages(bricks)
    }

    /// Allocates the next pool page ahead of time if headroom is low.
    ///
    /// Call it after GPU work has finished, for example at the end of extraction. The zero-fill of
    /// the new page then runs while waiting for the next input instead of delaying the next
    /// accumulation.
    ///
    /// # Errors
    ///
    /// Returns an error if the page cannot be allocated.
    pub fn prefetch(&mut self) -> Result<(), String> {
        let ctx = self.ctx.clone();
        let _g = ctx.lock.lock().unwrap();
        self.prefetch_page()
    }

    /// Returns the pool page layout.
    pub fn page_layout(&self) -> PageLayout {
        self.layout
    }

    /// Returns the time breakdown of the last integrate call.
    pub fn last_times(&self) -> FieldTimes {
        self.times
    }

    /// Returns the current pool usage.
    pub fn pool_stats(&self) -> PoolStats {
        PoolStats {
            cap: self.cap,
            high: self.high,
            free: self.free.len(),
            live: self.live,
            grows: self.grows,
        }
    }

    /// Returns the GPU context.
    pub fn ctx(&self) -> &Arc<GpuCtx> {
        &self.ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf::SdfMesher;
    use crate::testutil::plane;
    use crate::{Level, Mesher, Point};
    use std::sync::OnceLock;

    fn ctx() -> Option<Arc<GpuCtx>> {
        static C: OnceLock<Option<Arc<GpuCtx>>> = OnceLock::new();
        C.get_or_init(|| GpuCtx::shared().ok()).clone()
    }

    /// Samples the wavy surface z = 0.5 sin(x) cos(y) with normals. It crosses block boundaries and
    /// negative coordinates.
    fn wavy(x0: f32, x1: f32, step: f32, rgb: [u8; 3]) -> Vec<Point> {
        let mut v = Vec::new();
        let n = ((x1 - x0) / step) as i32;
        for i in 0..n {
            for j in 0..n {
                let (x, y) = (x0 + i as f32 * step, x0 + j as f32 * step);
                let z = 0.5 * x.sin() * y.cos();
                let g = [-0.5 * x.cos() * y.cos(), 0.5 * x.sin() * y.sin(), 1.0];
                let l = (g[0] * g[0] + g[1] * g[1] + 1.0f32).sqrt();
                v.push(Point {
                    pos: [x, y, z],
                    rgb,
                    normal: [g[0] / l, g[1] / l, g[2] / l],
                });
            }
        }
        v
    }

    /// Samples a sphere of radius `rad` around `c` with outward normals.
    fn sphere(c: [f32; 3], rad: f32, rgb: [u8; 3]) -> Vec<Point> {
        let mut v = Vec::new();
        let n = (rad * 60.0) as i32;
        for i in 0..n {
            let th = std::f32::consts::PI * (i as f32 + 0.5) / n as f32;
            let m = ((2 * n) as f32 * th.sin()).max(1.0) as i32;
            for j in 0..m {
                let ph = 2.0 * std::f32::consts::PI * j as f32 / m as f32;
                let d = [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()];
                v.push(Point {
                    pos: [c[0] + rad * d[0], c[1] + rad * d[1], c[2] + rad * d[2]],
                    rgb,
                    normal: d,
                });
            }
        }
        v
    }

    type Step = (SegmentId, Level, Vec<Point>);

    /// Mixed refined and preview steps, including a preview replaced by its refined segment.
    fn scenario() -> Vec<Step> {
        vec![
            (1, Level::Refined, wavy(-7.0, 7.0, 0.05, [100, 120, 140])),
            (
                2,
                Level::Preview,
                sphere([3.0, -3.0, 1.0], 2.0, [200, 10, 10]),
            ),
            (
                3,
                Level::Preview,
                plane(-9.0, -2.0, 1.0, 8.0, 0.7, 0.07, [10, 200, 10]),
            ),
            (
                4,
                Level::Preview,
                plane(5.0, 9.0, 5.0, 9.0, -0.3, 0.1, [10, 10, 200]),
            ),
            (
                2,
                Level::Refined,
                sphere([3.0, -3.0, 1.0], 2.0, [210, 20, 20]),
            ),
            (
                5,
                Level::Refined,
                plane(-3.0, 3.0, -9.0, -6.0, 0.25, 0.05, [5, 5, 5]),
            ),
        ]
    }

    /// Feeds the same bins and lists to the CPU mesher and the GPU field, checking changed ranges
    /// and dropped-layer ranges against the CPU after every step.
    struct Pair {
        m: SdfMesher,
        f: GpuField,
        base: FxHashMap<BlockId, ([i32; 3], [i32; 3])>,
        log: Vec<Vec<([i32; 3], [i32; 3])>>,
    }

    impl Pair {
        fn new(c: Arc<GpuCtx>, cfg: Config, cap: u32) -> Self {
            Self {
                m: SdfMesher::new(cfg),
                f: GpuField::with_capacity(c, &cfg, cap).unwrap(),
                base: FxHashMap::default(),
                log: Vec::new(),
            }
        }

        fn step(&mut self, seg: SegmentId, level: Level, pts: &[Point]) {
            let cfg = *self.m.config();
            let bins = self.m.prepare_bins(level, pts);
            let lists = SdfMesher::block_lists(&cfg, &bins);
            let cpu_drop = self.m.layer_ranges(Some(seg));
            let gpu_drop = self.f.drop_layer(seg);
            assert_eq!(
                gpu_drop, cpu_drop,
                "dropped layer ranges differ from CPU (segment {seg})"
            );
            let key = if level == Level::Refined {
                LayerKey::Base
            } else {
                LayerKey::Pending(seg)
            };
            let rs = self.f.integrate(key, &bins, &lists).unwrap();
            assert_eq!(rs.len(), lists.len());
            self.m.ingest(seg, level, pts).unwrap();
            let mut got: Vec<(BlockId, [i32; 3], [i32; 3])> = lists
                .iter()
                .zip(&rs)
                .filter(|(_, r)| r.0[0] <= r.1[0])
                .map(|((id, _), r)| (*id, r.0, r.1))
                .collect();
            got.sort_unstable_by_key(|x| x.0);
            match level {
                Level::Preview => assert_eq!(
                    got,
                    self.m.layer_ranges(Some(seg)),
                    "preview ranges differ from CPU (segment {seg})"
                ),
                Level::Refined => {
                    for (id, mn, mx) in got {
                        let e = self
                            .base
                            .entry(id)
                            .or_insert(([i32::MAX; 3], [i32::MIN; 3]));
                        for a in 0..3 {
                            e.0[a] = e.0[a].min(mn[a]);
                            e.1[a] = e.1[a].max(mx[a]);
                        }
                    }
                    let mut acc: Vec<_> = self.base.iter().map(|(id, r)| (*id, r.0, r.1)).collect();
                    acc.sort_unstable_by_key(|x| x.0);
                    assert_eq!(
                        acc,
                        self.m.layer_ranges(None),
                        "base ranges differ from CPU (segment {seg})"
                    );
                }
            }
            self.log.push(rs);
        }

        /// Returns every block touched by any layer together with its 26 neighbours, sorted.
        fn ids(&self) -> Vec<BlockId> {
            let mut s: FxHashMap<BlockId, ()> = FxHashMap::default();
            let segs = self.m.pending_segments();
            for l in std::iter::once(None).chain(segs.into_iter().map(Some)) {
                for (id, _, _) in self.m.layer_ranges(l) {
                    for d in 0..27 {
                        s.insert(
                            [
                                id[0] + d % 3 - 1,
                                id[1] + (d / 3) % 3 - 1,
                                id[2] + d / 9 - 1,
                            ],
                            (),
                        );
                    }
                }
            }
            let mut v: Vec<BlockId> = s.into_keys().collect();
            v.sort_unstable();
            v
        }
    }

    /// Reads the raw gather output of `ids` back to the CPU.
    fn read_vol(f: &GpuField, ids: &[BlockId]) -> Vec<f32> {
        let p = (f.cfg.block_dim + 2) as u64;
        let bytes = ids.len() as u64 * p * p * p * 32;
        let buf = f.gather_raw(ids).unwrap();
        bytemuck::cast_slice(&f.ctx.read_buffer(&buf, bytes).unwrap()).to_vec()
    }

    /// Compares the raw GPU volume with the CPU padded volume and returns the number of observed
    /// voxels checked.
    fn compare_with_cpu(pair: &Pair) -> usize {
        let ids = pair.ids();
        let cfg = *pair.m.config();
        let p = (cfg.block_dim + 2) as usize;
        let p3 = p * p * p;
        let scale = [cfg.splat_radius, 1.0, 255.0, 255.0, 255.0];
        let mut observed = 0;
        for chunk in ids.chunks(64) {
            let g = read_vol(&pair.f, chunk);
            for (j, id) in chunk.iter().enumerate() {
                let c = pair.m.padded_volume(*id);
                for v in 0..p3 {
                    let gv = &g[(j * p3 + v) * 8..(j * p3 + v) * 8 + 8];
                    let cv = &c[v];
                    assert_eq!(gv[1] == 0.0, cv[1] == 0.0, "Σw = 0 mismatch {id:?} {v}");
                    assert!(gv[5..].iter().all(|&x| x == 0.0));
                    if cv[1] == 0.0 {
                        assert!(
                            gv[..5].iter().all(|&x| x == 0.0),
                            "unobserved voxel is not zero {id:?} {v}"
                        );
                        continue;
                    }
                    observed += 1;
                    for k in 0..5 {
                        let tol = 1e-4 * cv[k].abs().max(cv[1] * scale[k]) + 1e-6;
                        assert!(
                            (gv[k] - cv[k]).abs() <= tol,
                            "{id:?} {v} component {k}: GPU {} CPU {}",
                            gv[k],
                            cv[k]
                        );
                    }
                }
            }
        }
        observed
    }

    #[test]
    fn matches_cpu_ranges_and_volume() {
        let Some(c) = ctx() else { return };
        let mut pair = Pair::new(c, Config::default(), 4);
        for (i, (seg, lv, pts)) in scenario().iter().enumerate() {
            pair.step(*seg, *lv, pts);
            if i == 3 || i == 5 {
                let n = compare_with_cpu(&pair);
                assert!(n > 10000, "observed voxels {n}");
            }
        }
        let st = pair.f.pool_stats();
        assert!(st.grows >= 2, "{st:?}");
        assert_eq!(st.high as usize, st.live + st.free);
    }

    /// The compact gather matches values derived from the raw gather: occupancy bits match the
    /// regions that hold bricks, observed voxels (Σw ≥ min_weight) are identical and the distance is
    /// Σw·d / Σw. The slot table has `blocks · nr³ + 1` offsets. Regions marked empty are never
    /// written, so every voxel inside them must have Σw = 0.
    #[test]
    fn compact_gather_matches_raw() {
        let Some(c) = ctx() else { return };
        let mut pair = Pair::new(c, Config::default(), 16);
        for (seg, lv, pts) in scenario().iter().take(4) {
            pair.step(*seg, *lv, pts);
        }
        let ids = pair.ids();
        let cfg = *pair.m.config();
        let p = cfg.block_dim as usize + 2;
        let p3 = p * p * p;
        let (owp, stride) = (
            crate::gpu_extract::occ_words(cfg.block_dim) as usize,
            crate::gpu_extract::volume_words(cfg.block_dim) as usize,
        );
        let nr = (cfg.block_dim / BR + 2) as usize;
        let raw = read_vol(&pair.f, &ids);
        let g = pair.f.gather(&ids).unwrap();
        let cv: Vec<u32> = bytemuck::cast_slice(
            &pair
                .f
                .ctx
                .read_buffer(&g.vol, pair.f.gather_bytes(ids.len()))
                .unwrap(),
        )
        .to_vec();
        let off: Vec<u32> = bytemuck::cast_slice(
            &pair
                .f
                .ctx
                .read_buffer(&g.off, ((ids.len() * nr * nr * nr + 1) * 4) as u64)
                .unwrap(),
        )
        .to_vec();
        let close = |a: f32, b: f32| (a - b).abs() <= 1e-6 * a.abs().max(1.0);
        let reg = |x: usize| (x + 7) >> 3;
        let (mut observed, mut skipped) = (0, 0);
        for j in 0..ids.len() {
            let occ = &cv[j * stride..j * stride + owp];
            let base = j * stride + owp;
            for v in 0..p3 {
                let r = &raw[(j * p3 + v) * 8..(j * p3 + v) * 8 + 5];
                let ri = (reg(v / (p * p)) * nr + reg((v / p) % p)) * nr + reg(v % p);
                let o = j * nr * nr * nr + ri;
                let has = occ[ri / 32] >> (ri % 32) & 1 == 1;
                assert_eq!(
                    has,
                    off[o + 1] > off[o],
                    "occupancy bits disagree with the slot table {j} {ri}"
                );
                if !has {
                    assert_eq!(r[1], 0.0, "weight in an empty region {j} {v}");
                    skipped += 1;
                    continue;
                }
                let d = f32::from_bits(cv[base + v]);
                if !(r[1] >= cfg.min_weight && r[1] > 0.0) {
                    assert_eq!(d, 3.0e38, "unobserved voxel {j} {v}");
                    continue;
                }
                observed += 1;
                assert!(
                    close(d, r[0] / r[1]),
                    "{j} {v} distance {d} vs {}",
                    r[0] / r[1]
                );
            }
        }
        assert!(observed > 10000, "observed voxels {observed}");
        assert!(skipped > 10000, "empty region voxels {skipped}");
    }

    #[test]
    fn matches_cpu_fine_voxel() {
        let Some(c) = ctx() else { return };
        let cfg = Config {
            voxel: 0.05,
            bin: 0.025,
            splat_radius: 0.1,
            ..Config::default()
        };
        let mut pair = Pair::new(c, cfg, 64);
        pair.step(1, Level::Refined, &wavy(-2.0, 2.0, 0.0125, [1, 2, 3]));
        pair.step(2, Level::Preview, &sphere([0.5, 0.5, 0.3], 0.6, [9, 9, 9]));
        assert!(compare_with_cpu(&pair) > 10000);
    }

    /// Runs `steps` on a fresh pair and returns it with the raw volume of all touched blocks.
    fn run_all(c: &Arc<GpuCtx>, steps: &[Step]) -> (Pair, Vec<f32>) {
        let mut pair = Pair::new(c.clone(), Config::default(), 16);
        for (seg, lv, pts) in steps {
            pair.step(*seg, *lv, pts);
        }
        let ids = pair.ids();
        let v = read_vol(&pair.f, &ids);
        (pair, v)
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn deterministic() {
        let Some(c) = ctx() else { return };
        let s = scenario();
        let (a, va) = run_all(&c, &s);
        let (b, vb) = run_all(&c, &s);
        assert_eq!(a.log, b.log);
        assert!(
            bits(&va) == bits(&vb),
            "same input produced different results"
        );
    }

    /// Preview, drop and refined replacement must equal inserting only the refined segment, and the
    /// preview slots must be reused.
    #[test]
    fn replacement_is_bitwise_exact() {
        let Some(c) = ctx() else { return };
        let s = scenario();
        let with_preview = vec![s[0].clone(), s[1].clone(), s[4].clone()];
        let refined_only = vec![s[0].clone(), s[4].clone()];
        let (a, va) = run_all(&c, &with_preview);
        let (b, vb) = run_all(&c, &refined_only);
        assert_eq!(a.ids(), b.ids());
        assert!(
            bits(&va) == bits(&vb),
            "preview, drop and refined differs from refined only"
        );
        let (sa, sb) = (a.f.pool_stats(), b.f.pool_stats());
        assert_eq!(sa.live, sb.live);
        assert!(sa.high < sb.high + sa.free as u32 + 1);
    }

    /// Splitting bins into batches (overlapping GPU accumulation with list building) gives bitwise
    /// the same result as a single submission.
    #[test]
    fn split_batches_match_single() {
        let Some(c) = ctx() else { return };
        let cfg = Config::default();
        let m = SdfMesher::new(cfg);
        let mut a = GpuField::with_capacity(c.clone(), &cfg, 4).unwrap();
        let mut b = GpuField::with_capacity(c.clone(), &cfg, 4).unwrap();
        let mut ids: FxHashMap<BlockId, ()> = FxHashMap::default();
        let sorted = |mut v: Vec<(BlockId, [i32; 3], [i32; 3])>| {
            v.sort_unstable_by_key(|x| x.0);
            v
        };
        for (k, (seg, lv, pts)) in scenario().into_iter().enumerate() {
            let bins = m.prepare_bins(lv, &pts);
            let key = if lv == Level::Refined {
                LayerKey::Base
            } else {
                LayerKey::Pending(seg)
            };
            assert_eq!(a.drop_layer(seg), b.drop_layer(seg));
            let ra = sorted(a.integrate_split(key, &bins, 1, || {}).unwrap());
            let rb = sorted(b.integrate_split(key, &bins, 2 + k % 3, || {}).unwrap());
            assert_eq!(ra, rb, "step {k}");
            for (id, mn, mx) in ra {
                if mn[0] <= mx[0] {
                    for d in 0..27 {
                        ids.insert(
                            [
                                id[0] + d % 3 - 1,
                                id[1] + (d / 3) % 3 - 1,
                                id[2] + d / 9 - 1,
                            ],
                            (),
                        );
                    }
                }
            }
        }
        let mut ids: Vec<BlockId> = ids.into_keys().collect();
        ids.sort_unstable();
        assert!(ids.len() > 50);
        for ch in ids.chunks(64) {
            assert!(
                bits(&read_vol(&a, ch)) == bits(&read_vol(&b, ch)),
                "split batches produced a different result"
            );
        }
        assert_eq!(a.pool_stats().live, b.pool_stats().live);
    }

    /// Slot to `(page, local slot)` mapping agrees with stacking the page sizes in order, and every
    /// page fits the size limit. The default layout starts at 4096 bricks and settles on fixed pages
    /// of 32768 bricks.
    #[test]
    fn page_layout_locates_every_slot() {
        for (first, n, max) in [
            (4096u32, 14u32, 4u64 << 30),
            (1, 14, 4 << 30),
            (4, 3, 4 << 30),
            (4096, 16, 64 << 20),
            (5, 2, 1 << 30),
        ] {
            let l = PageLayout::new(first, n, max);
            assert!(l.bsh <= l.msh);
            let mut s = 0u64;
            for k in 0..n {
                let ps = l.page_slots(k);
                assert!(
                    ps as u64 * BRICK_BYTES <= max.max(BRICK_BYTES),
                    "{l:?} page {k}"
                );
                for x in [0, 1, ps / 2, ps - 1].into_iter().filter(|&x| x < ps) {
                    assert_eq!(
                        l.locate((s + x as u64) as u32),
                        (k, x),
                        "{l:?} page {k} slot {x}"
                    );
                }
                s += ps as u64;
            }
            assert_eq!(s, l.max_slots());
        }
        let l = PageLayout::new(INIT_CAP, 14, 4 << 30);
        assert_eq!(
            (
                l.page_slots(0),
                l.page_slots(1),
                l.page_slots(3),
                l.page_slots(4),
                l.page_slots(13)
            ),
            (4096, 4096, 16384, 32768, 32768)
        );
    }

    /// Resending a preview reuses its dropped slots, and reused slots carry no stale values into a
    /// new layer.
    #[test]
    fn pool_grows_and_reuses_slots() {
        let Some(c) = ctx() else { return };
        let cfg = Config::default();
        let mut pair = Pair::new(c, cfg, 1);
        let pre = plane(-5.0, 5.0, -5.0, 5.0, 0.3, 0.07, [1, 1, 1]);
        pair.step(1, Level::Preview, &pre);
        let s1 = pair.f.pool_stats();
        assert!(
            s1.grows > 0 && s1.cap >= s1.high && s1.live > 0 && s1.free == 0,
            "{s1:?}"
        );
        pair.step(1, Level::Preview, &pre);
        let s2 = pair.f.pool_stats();
        assert_eq!(
            s2.high, s1.high,
            "dropped slots were not reused {s1:?} {s2:?}"
        );
        assert_eq!(s2.live, s1.live);
        assert!(compare_with_cpu(&pair) > 1000);
        let freed = pair.f.drop_layer(1);
        assert!(!freed.is_empty());
        let s3 = pair.f.pool_stats();
        assert_eq!(s3.live, 0);
        assert_eq!(s3.free as u32, s3.high);
        pair.m = SdfMesher::new(cfg);
        pair.base.clear();
        pair.step(
            7,
            Level::Refined,
            &plane(-4.0, 2.0, -4.0, 2.0, -0.2, 0.05, [3, 3, 3]),
        );
        assert!(compare_with_cpu(&pair) > 1000);
    }

    /// A batch containing a bin beyond the 24-bit coordinate range (2²³ voxels of 0.2 m) fails and
    /// leaves the field and pool state unchanged.
    #[test]
    fn failed_integrate_leaves_state_unchanged() {
        let Some(c) = ctx() else { return };
        let cfg = Config::default();
        let mut pair = Pair::new(c, cfg, 8);
        pair.step(
            1,
            Level::Refined,
            &plane(-3.0, 3.0, -3.0, 3.0, 0.1, 0.05, [7, 7, 7]),
        );
        let before = pair.f.pool_stats();
        let ids = pair.ids();
        let v0 = bits(&read_vol(&pair.f, &ids));
        let mut pts = plane(10.0, 12.0, 10.0, 12.0, 0.0, 0.05, [1, 1, 1]);
        pts.push(Point {
            pos: [2.0e6, 0.0, 0.0],
            rgb: [1, 1, 1],
            normal: [0.0, 0.0, 1.0],
        });
        pts.extend((0..20).map(|i| Point {
            pos: [2.0e6 + 0.02 * i as f32, 0.01 * i as f32, 0.0],
            rgb: [1, 1, 1],
            normal: [0.0, 0.0, 1.0],
        }));
        let bins = pair.m.prepare_bins(Level::Refined, &pts);
        let lists = SdfMesher::block_lists(&cfg, &bins);
        assert!(pair.f.integrate(LayerKey::Base, &bins, &lists).is_err());
        assert!(
            pair.f
                .integrate(LayerKey::Pending(9), &bins, &lists)
                .is_err()
        );
        let after = pair.f.pool_stats();
        assert_eq!(after.live, before.live);
        assert_eq!(after.high as usize, after.live + after.free);
        assert_eq!(pair.f.drop_layer(9), Vec::new());
        assert!(bits(&read_vol(&pair.f, &ids)) == v0);
        assert!(compare_with_cpu(&pair) > 1000);
    }

    /// The incrementally updated hash table still matches the CPU after the base layer grows. The
    /// intermediate preview step checks that preview bricks never enter the table.
    #[test]
    fn base_agreement_after_base_grows() {
        let Some(c) = ctx() else { return };
        let mut pair = Pair::new(c, Config::default(), 16);
        let s = scenario();
        pair.step(s[0].0, s[0].1, &s[0].2);
        let probe = |pair: &Pair| {
            let bins = pair.m.prepare_bins(Level::Preview, &s[1].2);
            let mut p: Vec<[f32; 3]> = bins.iter().map(|b| b.pos).collect();
            let mut n: Vec<[f32; 3]> = bins.iter().map(|b| b.normal).collect();
            p.extend(s[5].2.iter().map(|q| [q.pos[0], q.pos[1], q.pos[2] + 0.03]));
            n.extend(s[5].2.iter().map(|q| q.normal));
            let g = pair.f.base_agreement(&p, &n).unwrap();
            let mut some = 0;
            for i in 0..p.len() {
                let cv = pair.m.base_agreement(p[i], n[i]);
                assert_eq!(g[i].is_some(), cv.is_some(), "None mismatch {i}");
                if let (Some(a), Some(b)) = (g[i], cv) {
                    assert!((a - b).abs() <= 1e-3, "{i}: GPU {a} CPU {b}");
                    some += 1;
                }
            }
            some
        };
        let a = probe(&pair);
        pair.step(s[2].0, s[2].1, &s[2].2);
        pair.step(s[4].0, s[4].1, &s[4].2);
        let b = probe(&pair);
        pair.step(s[5].0, s[5].1, &s[5].2);
        let c2 = probe(&pair);
        assert!(a > 100 && b > a && c2 > b, "{a} {b} {c2}");
        assert!(
            pair.f.hash.lock().unwrap().incr >= 1,
            "incremental table update path was not exercised"
        );
    }

    /// GPU base agreement matches the CPU for preview bin positions with estimated normals and for
    /// random points with random directions.
    #[test]
    fn base_agreement_matches_cpu() {
        let Some(c) = ctx() else { return };
        let mut pair = Pair::new(c, Config::default(), 16);
        let s = scenario();
        pair.step(s[0].0, s[0].1, &s[0].2);
        pair.step(s[4].0, s[4].1, &s[4].2);
        let bins = pair.m.prepare_bins(Level::Preview, &s[1].2);
        let mut p: Vec<[f32; 3]> = bins.iter().map(|b| b.pos).collect();
        let mut n: Vec<[f32; 3]> = bins.iter().map(|b| b.normal).collect();
        let mut seed = 12345u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for _ in 0..50000 {
            p.push([rnd() * 16.0 - 8.0, rnd() * 16.0 - 8.0, rnd() * 2.0 - 1.0]);
            let d = [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5];
            let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-3);
            n.push([d[0] / l, d[1] / l, d[2] / l]);
        }
        let g = pair.f.base_agreement(&p, &n).unwrap();
        let mut some = 0;
        for i in 0..p.len() {
            let cv = pair.m.base_agreement(p[i], n[i]);
            assert_eq!(
                g[i].is_some(),
                cv.is_some(),
                "None mismatch {i} {:?} {:?}",
                g[i],
                cv
            );
            if let (Some(a), Some(b)) = (g[i], cv) {
                assert!((a - b).abs() <= 1e-3, "{i}: GPU {a} CPU {b}");
                some += 1;
            }
        }
        assert!(some > 1000, "points with a value {some}");
        assert!(!pair.f.base_is_empty());
    }
}
