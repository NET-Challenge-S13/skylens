/// GPU point binning and normal estimation, driven by `gpu_bins.rs`.
///
/// Every entry point uses the same bind group 0 layout, so one pipeline layout serves all kernels.
/// Metal compiles WGSL with fast-math, so NaN and infinity checks are done on the bit pattern
/// (see `fin`) instead of with float comparisons.
///
/// Pipeline: `k_pack` builds a 64-bit cell key per point; `k_count`, the scan kernels and `k_scatter`
/// (or `k_scatter_sg`) radix-sort the keys 8 bits per pass; `k_flags`, a scan and `k_segs` find cell
/// boundaries; `k_reduce` sums each cell; `k_slab` and `k_estimate` estimate missing normals.

/// Uniform parameters of one dispatch; layout matches `U` in `gpu_bins.rs`.
///
/// - `n`: number of elements; `nb`: number of 1024-element tiles.
/// - `shift`: bit offset of the current radix digit.
/// - `flag`: where `k_scan_top` also stores the scan total (1: `info[7]`, 2: `info[11]`).
/// - `stride`: u32 words per input point (7 with normal, 4 with position and color only).
/// - `policy`: normal policy (0 Ignore, 1 Trust, 2 Estimate).
/// - `radius`, `min_nb`: normal estimation window radius in cells and minimum neighbor count.
/// - `size`, `inv`: cell size and its reciprocal.
/// - `bx`, `by`, `bz`: key bits per axis; z occupies the low bits, then y, then x.
/// - `vbit`: bit position of the invalid-point flag in the key.
/// - `mnx`, `mny`, `mnz`: minimum cell coordinate per axis; keys store offsets from it.
/// - `rx`, `ry`, `rz`: maximum offset per axis (max - min).
/// - `orad`: neighbor radius in cells for normal orientation lists.
/// - `snz`: orientation seed threshold; a bin with `|nz| >= snz` is not ambiguous.
/// - `stab`: 1 when the per-x slab start table is valid.
/// - `p3`: padding.
struct U {
  n: u32,
  nb: u32,
  shift: u32,
  flag: u32,
  stride: u32,
  policy: u32,
  radius: i32,
  min_nb: u32,
  size: f32,
  inv: f32,
  bx: u32, by: u32, bz: u32,
  vbit: u32,
  mnx: i32, mny: i32, mnz: i32,
  rx: i32, ry: i32, rz: i32,
  orad: i32,
  snz: f32,
  stab: u32,
  p3: u32,
};

/// Bind group 0, shared by all kernels.
///
/// - `pts`: input points, `u.stride` words each (position bits, packed RGB, optional normal bits).
/// - `kin`, `vin` / `kout`, `vout`: sort keys and values, input and output of the current pass.
///   Keys are 64-bit as (low, high) word pairs; values are point indices.
/// - `sdata`: data being scanned (digit histogram or cell-start flags).
/// - `sums`: per-tile sums of the scan.
/// - `info`: counters and error flags. 7: number of cells, 8: key out of range, 9: subgroup layout
///   mismatch, 10: bin position does not map back to its cell key, 11: neighbor list length.
/// - `bins`: 10 words per cell: position (3), normal (3), color (3), count | state << 30.
///   States: 0 fixed normal, 1 needs estimation, 2 estimated, 3 dropped.
/// - `segs`: start of each cell in sorted order, followed by the number of valid points.
/// - `ukeys`: unique cell keys in sorted order.
/// - `slab`: for each x offset the first cell with that x, `rx + 2` entries, the last one is the
///   number of cells.
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> pts: array<u32>;
@group(0) @binding(2) var<storage, read_write> kin: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> vin: array<u32>;
@group(0) @binding(4) var<storage, read_write> kout: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read_write> vout: array<u32>;
@group(0) @binding(6) var<storage, read_write> sdata: array<u32>;
@group(0) @binding(7) var<storage, read_write> sums: array<u32>;
@group(0) @binding(8) var<storage, read_write> info: array<atomic<i32>, 16>;
@group(0) @binding(9) var<storage, read_write> bins: array<u32>;
@group(0) @binding(10) var<storage, read_write> segs: array<u32>;
@group(0) @binding(11) var<storage, read_write> ukeys: array<vec2<u32>>;
@group(0) @binding(12) var<storage, read_write> slab: array<u32>;

const I32MAX: i32 = 2147483647;
const I32MIN: i32 = -2147483647 - 1;

/// Flattens a 2D workgroup grid into a linear workgroup index.
fn gid(w: vec3<u32>, nw: vec3<u32>) -> u32 { return w.x + w.y * nw.x; }

/// True when the f32 with bit pattern `b` is finite.
fn fin(b: u32) -> bool { return (b & 0x7f800000u) != 0x7f800000u; }

/// Position of input point `i`.
fn pos_of(i: u32) -> vec3<f32> {
  let b = i * u.stride;
  return vec3<f32>(bitcast<f32>(pts[b]), bitcast<f32>(pts[b + 1u]), bitcast<f32>(pts[b + 2u]));
}

/// True when all coordinates of input point `i` are finite.
fn ok_of(i: u32) -> bool {
  let b = i * u.stride;
  return fin(pts[b]) && fin(pts[b + 1u]) && fin(pts[b + 2u]);
}

/// Converts to i32, saturating out-of-range values like Rust's `f32 as i32`.
fn sat(f: f32) -> i32 {
  if (f >= 2147483648.0) { return I32MAX; }
  if (f < -2147483648.0) { return I32MIN; }
  return i32(f);
}

/// Cell coordinate of a position, computed exactly as on the CPU.
fn cell(p: vec3<f32>) -> vec3<i32> {
  return vec3<i32>(sat(floor(p.x * u.inv)), sat(floor(p.y * u.inv)), sat(floor(p.z * u.inv)));
}

/// ORs `v` into the 64-bit key `k` starting at bit `at`.
fn put(k: vec2<u32>, v: u32, at: u32) -> vec2<u32> {
  var r = k;
  if (at < 32u) {
    r.x = r.x | (v << at);
    if (at > 0u) { r.y = r.y | (v >> (32u - at)); }
  } else {
    r.y = r.y | (v << (at - 32u));
  }
  return r;
}

/// Extracts `bits` bits of the 64-bit key `k` starting at bit `at`.
fn getb(k: vec2<u32>, at: u32, bits: u32) -> u32 {
  if (bits == 0u) { return 0u; }
  var v: u32;
  if (at < 32u) {
    v = k.x >> at;
    if (at > 0u) { v = v | (k.y << (32u - at)); }
  } else {
    v = k.y >> (at - 32u);
  }
  if (bits >= 32u) { return v; }
  return v & ((1u << bits) - 1u);
}

/// Builds the key of the cell at axis offsets (x, y, z).
fn mk(x: u32, y: u32, z: u32) -> vec2<u32> {
  return put(put(put(vec2<u32>(0u, 0u), z, 0u), y, u.bz), x, u.bz + u.by);
}

/// 64-bit key comparison `a < b`.
fn klt(a: vec2<u32>, b: vec2<u32>) -> bool { return a.y < b.y || (a.y == b.y && a.x < b.x); }

/// Builds the sort key of each point; the value is the point index.
///
/// The per-axis cell range was computed by the CPU with the same rule while uploading. The offset
/// from the minimum uses unsigned subtraction, so anything outside `0..=r` is detected, flagged in
/// `info[8]` (the host returns an error) and clamped. Points with a non-finite position get only the
/// invalid bit set, which sorts them after all valid points.
@compute @workgroup_size(256)
fn k_pack(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let i = gid(w, nw) * 256u + t;
  if (i >= u.n) { return; }
  var k = vec2<u32>(0u, 0u);
  if (ok_of(i)) {
    let c = cell(pos_of(i));
    let ux = bitcast<u32>(c.x) - bitcast<u32>(u.mnx);
    let uy = bitcast<u32>(c.y) - bitcast<u32>(u.mny);
    let uz = bitcast<u32>(c.z) - bitcast<u32>(u.mnz);
    if (ux > u32(u.rx) || uy > u32(u.ry) || uz > u32(u.rz)) {
      atomicMax(&info[8], 1);
    }
    k = mk(min(ux, u32(u.rx)), min(uy, u32(u.ry)), min(uz, u32(u.rz)));
  } else {
    k = put(k, 1u, u.vbit);
  }
  kin[i] = k;
  vin[i] = i;
}

/// Current 8-bit radix digit of a key.
///
/// The sort is an LSD radix sort with 8-bit digits over tiles of 1024 elements (128 threads x 8).
/// Each pass builds a per-tile digit histogram (`k_count`), scans it, and scatters (`k_scatter`):
/// the tile is stably partitioned in shared memory by two 4-bit rounds and then written out so that
/// elements with the same digit are contiguous. Elements with equal digits keep their input order.
fn digit8(k: vec2<u32>) -> u32 {
  if (u.shift < 32u) { return (k.x >> u.shift) & 255u; }
  return (k.y >> (u.shift - 32u)) & 255u;
}

/// 4-bit digit of a key at bit `s`.
fn nib(k: vec2<u32>, s: u32) -> u32 {
  if (s < 32u) { return (k.x >> s) & 15u; }
  return (k.y >> (s - 32u)) & 15u;
}

/// Per-tile 8-bit digit histogram.
var<workgroup> wc: array<atomic<u32>, 256>;

/// Counts digits per tile and stores the histogram digit-major as `sdata[digit * nb + tile]`.
///
/// Integer atomics make the counts exact regardless of order.
@compute @workgroup_size(128)
fn k_count(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let b = gid(w, nw);
  atomicStore(&wc[t], 0u);
  atomicStore(&wc[t + 128u], 0u);
  workgroupBarrier();
  if (b < u.nb) {
    for (var k = 0u; k < 8u; k = k + 1u) {
      let i = b * 1024u + k * 128u + t;
      if (i < u.n) { atomicAdd(&wc[digit8(kin[i])], 1u); }
    }
  }
  workgroupBarrier();
  if (b < u.nb) {
    sdata[t * u.nb + b] = atomicLoad(&wc[t]);
    sdata[(t + 128u) * u.nb + b] = atomicLoad(&wc[t + 128u]);
  }
}

/// Shared memory of `k_scatter`: tile keys and values, the 4-bit histogram laid out as
/// `[digit * 128 + thread]`, a per-thread scan buffer and the 8-bit digit starts within the tile.
var<workgroup> sk: array<vec2<u32>, 1024>;
var<workgroup> sv: array<u32, 1024>;
var<workgroup> ws: array<u32, 2048>;
var<workgroup> wt: array<u32, 128>;
var<workgroup> wl: array<u32, 256>;

/// Stable scatter of one radix pass.
///
/// Each thread loads 8 consecutive elements. Slots past the end of the tile are filled with
/// all-ones keys: every digit is 15, so they stay behind the valid elements through both 4-bit
/// partition rounds. In each round the digit-major histogram `ws[d * 128 + t]` is exclusively
/// scanned, with each thread owning 16 consecutive entries. Finally the elements are written to
/// `histogram scan + position within the tile's digit run`.
@compute @workgroup_size(128)
fn k_scatter(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let b = gid(w, nw);
  let t0 = b * 1024u;
  var cnt = 0u;
  if (b < u.nb) { cnt = min(1024u, u.n - t0); }
  var kk: array<vec2<u32>, 8>;
  var vv: array<u32, 8>;
  atomicStore(&wc[t], 0u);
  atomicStore(&wc[t + 128u], 0u);
  for (var k = 0u; k < 8u; k = k + 1u) {
    let j = t * 8u + k;
    kk[k] = vec2<u32>(0xffffffffu, 0xffffffffu);
    vv[k] = 0u;
    if (j < cnt) {
      kk[k] = kin[t0 + j];
      vv[k] = vin[t0 + j];
    }
  }
  workgroupBarrier();
  for (var k = 0u; k < 8u; k = k + 1u) {
    if (t * 8u + k < cnt) { atomicAdd(&wc[digit8(kk[k])], 1u); }
  }
  for (var rnd = 0u; rnd < 2u; rnd = rnd + 1u) {
    let s = u.shift + rnd * 4u;
    var c: array<u32, 16>;
    for (var k = 0u; k < 8u; k = k + 1u) {
      let d = nib(kk[k], s);
      c[d] = c[d] + 1u;
    }
    for (var d = 0u; d < 16u; d = d + 1u) { ws[d * 128u + t] = c[d]; }
    workgroupBarrier();
    var run = 0u;
    for (var j = 0u; j < 16u; j = j + 1u) {
      let e = t * 16u + j;
      let v = ws[e];
      ws[e] = run;
      run = run + v;
    }
    wt[t] = run;
    workgroupBarrier();
    for (var off = 1u; off < 128u; off = off << 1u) {
      var v = wt[t];
      if (t >= off) { v = v + wt[t - off]; }
      workgroupBarrier();
      wt[t] = v;
      workgroupBarrier();
    }
    let pre = wt[t] - run;
    for (var j = 0u; j < 16u; j = j + 1u) { ws[t * 16u + j] = ws[t * 16u + j] + pre; }
    workgroupBarrier();
    var r: array<u32, 16>;
    for (var k = 0u; k < 8u; k = k + 1u) {
      let d = nib(kk[k], s);
      let p = ws[d * 128u + t] + r[d];
      r[d] = r[d] + 1u;
      sk[p] = kk[k];
      sv[p] = vv[k];
    }
    workgroupBarrier();
    for (var k = 0u; k < 8u; k = k + 1u) {
      kk[k] = sk[t * 8u + k];
      vv[k] = sv[t * 8u + k];
    }
    workgroupBarrier();
  }
  let h0 = atomicLoad(&wc[2u * t]);
  let h1 = atomicLoad(&wc[2u * t + 1u]);
  wt[t] = h0 + h1;
  workgroupBarrier();
  for (var off = 1u; off < 128u; off = off << 1u) {
    var v = wt[t];
    if (t >= off) { v = v + wt[t - off]; }
    workgroupBarrier();
    wt[t] = v;
    workgroupBarrier();
  }
  let e0 = wt[t] - h0 - h1;
  wl[2u * t] = e0;
  wl[2u * t + 1u] = e0 + h0;
  workgroupBarrier();
  for (var k = 0u; k < 8u; k = k + 1u) {
    let j = k * 128u + t;
    if (j < cnt) {
      let key = sk[j];
      let d = digit8(key);
      let g = sdata[d * u.nb + b] + j - wl[d];
      kout[g] = key;
      vout[g] = sv[j];
    }
  }
}

/// Shared scan buffer of the scan kernels.
var<workgroup> wu: array<u32, 256>;

/// First step of the exclusive scan of `sdata`: scans each tile of 1024 elements (256 threads x 4)
/// in place and writes the tile total to `sums`.
@compute @workgroup_size(256)
fn k_scan_tile(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let b = gid(w, nw);
  let base = b * 1024u + t * 4u;
  var v: array<u32, 4>;
  var run = 0u;
  for (var k = 0u; k < 4u; k = k + 1u) {
    let i = base + k;
    var x = 0u;
    if (b < u.nb && i < u.n) { x = sdata[i]; }
    v[k] = run;
    run = run + x;
  }
  wu[t] = run;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off = off << 1u) {
    var y = wu[t];
    if (t >= off) { y = y + wu[t - off]; }
    workgroupBarrier();
    wu[t] = y;
    workgroupBarrier();
  }
  let pre = wu[t] - run;
  for (var k = 0u; k < 4u; k = k + 1u) {
    let i = base + k;
    if (b < u.nb && i < u.n) { sdata[i] = v[k] + pre; }
  }
  if (t == 255u && b < u.nb) { sums[b] = wu[255]; }
}

/// Second step: a single workgroup exclusively scans the `u.nb` tile sums.
///
/// The grand total is written to `sums[u.nb]`, and also to `info[7]` or `info[11]` depending on
/// `u.flag`.
@compute @workgroup_size(256)
fn k_scan_top(@builtin(local_invocation_index) t: u32) {
  let n = u.nb;
  let chunk = (n + 255u) / 256u;
  let s0 = min(t * chunk, n);
  let s1 = min(s0 + chunk, n);
  var run = 0u;
  for (var i = s0; i < s1; i = i + 1u) { run = run + sums[i]; }
  wu[t] = run;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off = off << 1u) {
    var y = wu[t];
    if (t >= off) { y = y + wu[t - off]; }
    workgroupBarrier();
    wu[t] = y;
    workgroupBarrier();
  }
  var acc = wu[t] - run;
  for (var i = s0; i < s1; i = i + 1u) {
    let x = sums[i];
    sums[i] = acc;
    acc = acc + x;
  }
  if (t == 255u) {
    sums[n] = wu[255];
    if (u.flag == 1u) { atomicStore(&info[7], i32(wu[255])); }
    if (u.flag == 2u) { atomicStore(&info[11], i32(wu[255])); }
  }
}

/// Third step: adds each tile's scanned offset to its elements.
@compute @workgroup_size(256)
fn k_scan_add(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let i = gid(w, nw) * 256u + t;
  if (i >= u.n) { return; }
  let b = i / 1024u;
  if (b > 0u) { sdata[i] = sdata[i] + sums[b]; }
}

/// True when sorted element `i` starts a new cell.
fn is_head(i: u32) -> bool {
  if (i == 0u) { return true; }
  let a = kin[i];
  let p = kin[i - 1u];
  return a.x != p.x || a.y != p.y;
}

/// Writes 1 to `sdata` for elements that start a cell, 0 otherwise.
@compute @workgroup_size(256)
fn k_flags(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let i = gid(w, nw) * 256u + t;
  if (i >= u.n) { return; }
  sdata[i] = select(0u, 1u, is_head(i));
}

/// After the flags are scanned, records the start and the unique key of each cell.
///
/// The last thread also writes the end sentinel `segs[cell count] = n`.
@compute @workgroup_size(256)
fn k_segs(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let i = gid(w, nw) * 256u + t;
  if (i >= u.n) { return; }
  if (is_head(i)) {
    let s = sdata[i];
    segs[s] = i;
    ukeys[s] = kin[i];
  }
  if (i == u.n - 1u) { segs[u32(atomicLoad(&info[7]))] = u.n; }
}

/// Cell coordinate encoded in a key.
fn cell_of(k: vec2<u32>) -> vec3<i32> {
  let ux = getb(k, u.bz + u.by, u.bx);
  let uy = getb(k, u.bz, u.by);
  let uz = getb(k, 0u, u.bz);
  return vec3<i32>(bitcast<i32>(ux + bitcast<u32>(u.mnx)), bitcast<i32>(uy + bitcast<u32>(u.mny)), bitcast<i32>(uz + bitcast<u32>(u.mnz)));
}

/// Sums each cell, one thread per cell, in the original input order.
///
/// Positions are accumulated relative to the cell origin in f32 and colors as integer sums. With the
/// Trust policy, finite normals with squared length in [0.25, 2.25] are averaged; if the average is
/// shorter than 0.3 (normals in the cell disagree), or no normal qualifies, the cell is marked for
/// estimation instead.
@compute @workgroup_size(256)
fn k_reduce(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let s = gid(w, nw) * 256u + t;
  let m = u32(atomicLoad(&info[7]));
  if (s >= m) { return; }
  let a = segs[s];
  let e = segs[s + 1u];
  let org = vec3<f32>(cell_of(ukeys[s])) * u.size;
  var sp = vec3<f32>(0.0);
  var sc = vec3<u32>(0u);
  var sn = vec3<f32>(0.0);
  var nv = 0u;
  for (var j = a; j < e; j = j + 1u) {
    let idx = vin[j];
    let b = idx * u.stride;
    let p = vec3<f32>(bitcast<f32>(pts[b]), bitcast<f32>(pts[b + 1u]), bitcast<f32>(pts[b + 2u]));
    sp = sp + (p - org);
    let col = pts[b + 3u];
    sc = sc + vec3<u32>(col & 255u, (col >> 8u) & 255u, (col >> 16u) & 255u);
    if (u.policy == 1u) {
      let bx = pts[b + 4u];
      let by = pts[b + 5u];
      let bz = pts[b + 6u];
      if (fin(bx) && fin(by) && fin(bz)) {
        let n = vec3<f32>(bitcast<f32>(bx), bitcast<f32>(by), bitcast<f32>(bz));
        let l2 = n.x * n.x + n.y * n.y + n.z * n.z;
        if (fin(bitcast<u32>(l2)) && l2 >= 0.25 && l2 <= 2.25) {
          sn = sn + n;
          nv = nv + 1u;
        }
      }
    }
  }
  let cnt = e - a;
  let cf = f32(cnt);
  let pos = org + sp / cf;
  let rgb = vec3<f32>(sc) / cf;
  var nrm = vec3<f32>(0.0);
  var ok = false;
  if (nv > 0u) {
    let l = sqrt(sn.x * sn.x + sn.y * sn.y + sn.z * sn.z);
    if (l > 0.3 * f32(nv)) {
      nrm = sn / l;
      ok = true;
    }
  }
  let st = select(0u, 1u, u.policy != 0u && !ok);
  let o = s * 10u;
  bins[o] = bitcast<u32>(pos.x);
  bins[o + 1u] = bitcast<u32>(pos.y);
  bins[o + 2u] = bitcast<u32>(pos.z);
  bins[o + 3u] = bitcast<u32>(nrm.x);
  bins[o + 4u] = bitcast<u32>(nrm.y);
  bins[o + 5u] = bitcast<u32>(nrm.z);
  bins[o + 6u] = bitcast<u32>(rgb.x);
  bins[o + 7u] = bitcast<u32>(rgb.y);
  bins[o + 8u] = bitcast<u32>(rgb.z);
  bins[o + 9u] = cnt | (st << 30u);
}

/// Position of cell `s`.
fn bpos(s: u32) -> vec3<f32> {
  let o = s * 10u;
  return vec3<f32>(bitcast<f32>(bins[o]), bitcast<f32>(bins[o + 1u]), bitcast<f32>(bins[o + 2u]));
}

/// First index in `ukeys[lo0..hi0]` whose key is `>= x`.
fn lower_bound(lo0: u32, hi0: u32, x: vec2<u32>) -> u32 {
  var lo = lo0;
  var hi = hi0;
  while (lo < hi) {
    let mid = (lo + hi) >> 1u;
    if (klt(ukeys[mid], x)) { lo = mid + 1u; } else { hi = mid; }
  }
  return lo;
}

/// First index in `ukeys[lo0..hi0]` whose key is `> x`.
fn upper_bound(lo0: u32, hi0: u32, x: vec2<u32>) -> u32 {
  var lo = lo0;
  var hi = hi0;
  while (lo < hi) {
    let mid = (lo + hi) >> 1u;
    if (klt(x, ukeys[mid])) { hi = mid; } else { lo = mid + 1u; }
  }
  return lo;
}

/// Eigenvector of the smallest eigenvalue by cyclic Jacobi rotations.
///
/// Fallback for `smallest_eigvec` when the closed-form solution is ill-conditioned.
fn jacobi(a00: f32, a01: f32, a02: f32, a11: f32, a12: f32, a22: f32) -> vec3<f32> {
  var a: array<array<f32, 3>, 3>;
  a[0][0] = a00; a[0][1] = a01; a[0][2] = a02;
  a[1][0] = a01; a[1][1] = a11; a[1][2] = a12;
  a[2][0] = a02; a[2][1] = a12; a[2][2] = a22;
  var v: array<array<f32, 3>, 3>;
  v[0][0] = 1.0; v[1][1] = 1.0; v[2][2] = 1.0;
  for (var it = 0; it < 32; it = it + 1) {
    let off = abs(a[0][1]) + abs(a[0][2]) + abs(a[1][2]);
    let dg = abs(a[0][0]) + abs(a[1][1]) + abs(a[2][2]);
    if (off <= 1e-9 * dg || off < 1e-30) { break; }
    for (var pi = 0; pi < 3; pi = pi + 1) {
      var p = 0;
      var q = 1;
      if (pi == 1) { q = 2; }
      if (pi == 2) { p = 1; q = 2; }
      let apq = a[p][q];
      if (abs(apq) < 1e-30) { continue; }
      let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
      var tt: f32;
      if (theta == 0.0) {
        tt = 1.0;
      } else if (abs(theta) > 1e15) {
        tt = 0.5 / theta;
      } else {
        tt = sign(theta) / (abs(theta) + sqrt(theta * theta + 1.0));
      }
      let c = 1.0 / sqrt(tt * tt + 1.0);
      let sn = tt * c;
      for (var k = 0; k < 3; k = k + 1) {
        let akp = a[k][p];
        let akq = a[k][q];
        a[k][p] = c * akp - sn * akq;
        a[k][q] = sn * akp + c * akq;
      }
      for (var k = 0; k < 3; k = k + 1) {
        let apk = a[p][k];
        let aqk = a[q][k];
        a[p][k] = c * apk - sn * aqk;
        a[q][k] = sn * apk + c * aqk;
      }
      for (var k = 0; k < 3; k = k + 1) {
        let vp = v[k][p];
        let vq = v[k][q];
        v[k][p] = c * vp - sn * vq;
        v[k][q] = sn * vp + c * vq;
      }
    }
  }
  var best = 0;
  if (a[1][1] < a[best][best]) { best = 1; }
  if (a[2][2] < a[best][best]) { best = 2; }
  return vec3<f32>(v[0][best], v[1][best], v[2][best]);
}

/// Eigenvector of the smallest eigenvalue of a symmetric 3x3 matrix, in f32.
///
/// Uses the same method as `eig.rs`: the closed-form trigonometric solution on the scaled matrix,
/// falling back to `jacobi` when the matrix is degenerate or the two smallest eigenvalues nearly
/// coincide. The coincidence tolerance is looser than the CPU's 1e-8 because of f32 precision.
fn smallest_eigvec(m00: f32, m01: f32, m02: f32, m11: f32, m12: f32, m22: f32) -> vec3<f32> {
  let scale = max(max(max(abs(m00), abs(m11)), max(abs(m22), abs(m01))), max(abs(m02), abs(m12)));
  if (!(scale > 1e-30) || !fin(bitcast<u32>(scale))) {
    return jacobi(m00, m01, m02, m11, m12, m22);
  }
  let inv = 1.0 / scale;
  let a00 = m00 * inv; let a11 = m11 * inv; let a22 = m22 * inv;
  let a01 = m01 * inv; let a02 = m02 * inv; let a12 = m12 * inv;
  let q = (a00 + a11 + a22) / 3.0;
  let b00 = a00 - q; let b11 = a11 - q; let b22 = a22 - q;
  let off2 = a01 * a01 + a02 * a02 + a12 * a12;
  let p2 = (b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * off2) / 6.0;
  if (p2 < 1e-12) {
    return jacobi(a00, a01, a02, a11, a12, a22);
  }
  let p = sqrt(p2);
  let det = b00 * (b11 * b22 - a12 * a12) - a01 * (a01 * b22 - a12 * a02) + a02 * (a01 * a12 - b11 * a02);
  let half = clamp(det / (2.0 * p2 * p), -1.0, 1.0);
  let phi = acos(half) / 3.0;
  let lam = q + 2.0 * p * cos(phi + 2.0943951023931953);
  let r0 = vec3<f32>(a00 - lam, a01, a02);
  let r1 = vec3<f32>(a01, a11 - lam, a12);
  let r2 = vec3<f32>(a02, a12, a22 - lam);
  let c0 = cross(r0, r1);
  let c1 = cross(r0, r2);
  let c2 = cross(r1, r2);
  let d0 = dot(c0, c0);
  let d1 = dot(c1, c1);
  let d2 = dot(c2, c2);
  var cb = c0;
  var db = d0;
  if (d1 > db) { cb = c1; db = d1; }
  if (d2 > db) { cb = c2; db = d2; }
  if (db < 1e-6 * p2 * p2) {
    return jacobi(a00, a01, a02, a11, a12, a22);
  }
  return cb / sqrt(db);
}

/// Range `[lo, hi)` of cells with x offset `xu` and y offset in `[ylo, yhi]`.
///
/// With a valid slab table the binary search is limited to the x slab; the result is the same.
fn row_span(xu: u32, ylo: i32, yhi: i32, m: u32) -> vec2<u32> {
  var a = 0u;
  var b = m;
  if (u.stab == 1u) {
    a = slab[xu];
    b = slab[xu + 1u];
  }
  let lo = lower_bound(a, b, mk(xu, u32(ylo), 0u));
  let hi = upper_bound(lo, b, mk(xu, u32(yhi), u32(u.rz)));
  return vec2<u32>(lo, hi);
}

/// Builds the slab table: `slab[x]` is the first cell with key `>= (x, 0, 0)` and `slab[rx + 1]` is
/// the number of cells.
@compute @workgroup_size(256)
fn k_slab(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let x = gid(w, nw) * 256u + t;
  let last = u32(u.rx) + 1u;
  if (x > last) { return; }
  let m = u32(atomicLoad(&info[7]));
  if (x == last) {
    slab[x] = m;
  } else {
    slab[x] = lower_bound(0u, m, mk(x, 0u, 0u));
  }
}

/// Estimates normals of cells marked for estimation, one thread per cell.
@compute @workgroup_size(64)
fn k_estimate(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let s = gid(w, nw) * 64u + t;
  let m = u32(atomicLoad(&info[7]));
  if (s >= m) { return; }
  _ = est_one(s, m);
}

/// Estimates the normal of cell `s` if it is marked for estimation.
///
/// Gathers the covariance of the cell positions in the `(2r+1)^3` window, relative to the center
/// cell; for every (dx, dy) row the z range is found by binary search in the sorted keys. The normal
/// is the smallest eigenvector, flipped to point towards +z. Cells with fewer than `u.min_nb`
/// neighbors or a non-finite result are dropped (state 3), otherwise marked estimated (state 2).
///
/// Returns the number of cells in the window including `s`, or 0 if `s` was not a candidate.
fn est_one(s: u32, m: u32) -> u32 {
  let o = s * 10u;
  let cw = bins[o + 9u];
  if ((cw >> 30u) != 1u) { return 0u; }
  let k = ukeys[s];
  let ux = i32(getb(k, u.bz + u.by, u.bx));
  let uy = i32(getb(k, u.bz, u.by));
  let uz = i32(getb(k, 0u, u.bz));
  let r = u.radius;
  let ylo = max(uy - r, 0);
  let yhi = min(uy + r, u.ry);
  let zlo = u32(max(uz - r, 0));
  let zhi = u32(min(uz + r, u.rz));
  let c0 = bpos(s);
  var n = 0u;
  var sm = vec3<f32>(0.0);
  var sxx = 0.0; var sxy = 0.0; var sxz = 0.0; var syy = 0.0; var syz = 0.0; var szz = 0.0;
  for (var dx = -r; dx <= r; dx = dx + 1) {
    let x = ux + dx;
    if (x < 0 || x > u.rx || ylo > yhi) { continue; }
    let xu = u32(x);
    let sp = row_span(xu, ylo, yhi, m);
    let hi = sp.y;
    var cur = sp.x;
    for (var y = ylo; y <= yhi; y = y + 1) {
      cur = lower_bound(cur, hi, mk(xu, u32(y), zlo));
      let top = mk(xu, u32(y), zhi);
      var c = cur;
      loop {
        if (c >= hi) { break; }
        if (klt(top, ukeys[c])) { break; }
        let d = bpos(c) - c0;
        sm = sm + d;
        sxx = sxx + d.x * d.x;
        sxy = sxy + d.x * d.y;
        sxz = sxz + d.x * d.z;
        syy = syy + d.y * d.y;
        syz = syz + d.y * d.z;
        szz = szz + d.z * d.z;
        n = n + 1u;
        c = c + 1u;
      }
      cur = c;
    }
  }
  var st = 3u;
  if (n >= u.min_nb) {
    let nf = f32(n);
    let mu = sm / nf;
    let v = smallest_eigvec(sxx / nf - mu.x * mu.x, sxy / nf - mu.x * mu.y, sxz / nf - mu.x * mu.z,
                            syy / nf - mu.y * mu.y, syz / nf - mu.y * mu.z, szz / nf - mu.z * mu.z);
    let l = sqrt(dot(v, v));
    var sg = 1.0;
    if (v.z < 0.0) { sg = -1.0; }
    let nn = sg * v / l;
    if (fin(bitcast<u32>(nn.x)) && fin(bitcast<u32>(nn.y)) && fin(bitcast<u32>(nn.z))) {
      bins[o + 3u] = bitcast<u32>(nn.x);
      bins[o + 4u] = bitcast<u32>(nn.y);
      bins[o + 5u] = bitcast<u32>(nn.z);
      st = 2u;
    }
  }
  bins[o + 9u] = (cw & 0x3fffffffu) | (st << 30u);
  return n;
}
