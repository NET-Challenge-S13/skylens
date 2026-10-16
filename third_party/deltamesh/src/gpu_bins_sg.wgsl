/// Subgroup variant of the radix sort scatter. Compiled appended to `gpu_bins.wgsl`; requires the
/// `SUBGROUP` feature.
///
/// A tile holds 1024 elements (128 threads x 8). Subgroup `s` handles the contiguous range
/// `[s * per, (s + 1) * per)` of the tile in 8 steps of `ssz` elements. Lanes with the same digit are
/// found with 8 ballots, one per digit bit, and ranked by popcount, so no shared-memory atomics are
/// needed and the result is deterministic. Per-subgroup digit counts give each element's sorted
/// position within the tile; the resulting permutation is kept in shared memory and read back in
/// sorted order, so neighboring threads write neighboring output slots. The sort is stable.
///
/// Shared memory is kept small because on Metal its size strongly limits occupancy.

/// Shared memory of `k_scatter_sg`.
///
/// - `sgh`: `[subgroup (at most 4)][digit (256)]` counts, then each subgroup's start within the
///   tile's run of that digit.
/// - `tds`: start of each digit within the tile (exclusive scan).
/// - `perm`: sorted position within the tile to element index within the tile.
/// - `wq`: scan buffer.
var<workgroup> sgh: array<u32, 1024>;
var<workgroup> tds: array<u32, 256>;
var<workgroup> perm: array<u32, 1024>;
var<workgroup> wq: array<u32, 128>;

/// Population count of a 128-bit ballot mask.
fn pop4(v: vec4<u32>) -> u32 {
  return countOneBits(v.x) + countOneBits(v.y) + countOneBits(v.z) + countOneBits(v.w);
}

/// Stable scatter of one radix pass using subgroup ballots.
///
/// Requires full subgroups and at most 4 of them (subgroup size of at least 32). Otherwise it sets
/// `info[9]` and does nothing; the host then disables this variant and reruns with `k_scatter`.
/// Within a ballot step, the first lane of each group of equal digits adds the group size to the
/// subgroup's count; each subgroup has its own counters, so these writes never collide.
@compute @workgroup_size(128)
fn k_scatter_sg(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
                @builtin(local_invocation_index) t: u32,
                @builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) ssz: u32,
                @builtin(subgroup_id) sg: u32, @builtin(num_subgroups) nsg: u32) {
  if (nsg * ssz != 128u || nsg > 4u) {
    if (t == 0u) { atomicMax(&info[9], 1); }
    return;
  }
  let b = gid(w, nw);
  let t0 = b * 1024u;
  var cnt = 0u;
  if (b < u.nb) { cnt = min(1024u, u.n - t0); }
  for (var i = t; i < nsg * 256u; i = i + 128u) { sgh[i] = 0u; }
  var lt = vec4<u32>(0u);
  for (var q = 0u; q < 4u; q = q + 1u) {
    let lo = q * 32u;
    if (lane >= lo + 32u) { lt[q] = 0xffffffffu; } else if (lane > lo) { lt[q] = (1u << (lane - lo)) - 1u; }
  }
  workgroupBarrier();
  let per = 8u * ssz;
  var dd: array<u32, 8>;
  var loc: array<u32, 8>;
  for (var k = 0u; k < 8u; k = k + 1u) {
    let j = sg * per + k * ssz + lane;
    let ok = j < cnt;
    var key = vec2<u32>(0u, 0u);
    if (ok) { key = kin[t0 + j]; }
    let d = digit8(key);
    var m = subgroupBallot(ok);
    for (var bit = 0u; bit < 8u; bit = bit + 1u) {
      let on = ((d >> bit) & 1u) == 1u;
      let bb = subgroupBallot(on);
      m = m & select(~bb, bb, on);
    }
    let rank = pop4(m & lt);
    let tot = pop4(m);
    let pre = sgh[sg * 256u + d];
    workgroupBarrier();
    if (ok && rank == 0u) { sgh[sg * 256u + d] = pre + tot; }
    workgroupBarrier();
    dd[k] = d;
    loc[k] = pre + rank;
  }
  var tot2 = vec2<u32>(0u);
  for (var h = 0u; h < 2u; h = h + 1u) {
    let dg = 2u * t + h;
    var run = 0u;
    for (var s = 0u; s < nsg; s = s + 1u) {
      let c = sgh[s * 256u + dg];
      sgh[s * 256u + dg] = run;
      run = run + c;
    }
    tot2[h] = run;
  }
  wq[t] = tot2.x + tot2.y;
  workgroupBarrier();
  for (var off = 1u; off < 128u; off = off << 1u) {
    var v = wq[t];
    if (t >= off) { v = v + wq[t - off]; }
    workgroupBarrier();
    wq[t] = v;
    workgroupBarrier();
  }
  let e0 = wq[t] - tot2.x - tot2.y;
  tds[2u * t] = e0;
  tds[2u * t + 1u] = e0 + tot2.x;
  workgroupBarrier();
  for (var k = 0u; k < 8u; k = k + 1u) {
    let j = sg * per + k * ssz + lane;
    if (j < cnt) {
      let d = dd[k];
      perm[tds[d] + sgh[sg * 256u + d] + loc[k]] = j;
    }
  }
  workgroupBarrier();
  for (var k = 0u; k < 8u; k = k + 1u) {
    let jj = k * 128u + t;
    if (jj < cnt) {
      let src = t0 + perm[jj];
      let key = kin[src];
      let d = digit8(key);
      let g = sdata[d * u.nb + b] + jj - tds[d];
      kout[g] = key;
      vout[g] = vin[src];
    }
  }
}
