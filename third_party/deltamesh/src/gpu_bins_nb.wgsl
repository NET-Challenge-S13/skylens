/// Neighbor lists for normal orientation (`orient.rs`). Compiled appended to `gpu_bins.wgsl`.
///
/// For every estimated bin with `|nz| < snz` (a bin whose orientation is ambiguous), the kernels list
/// the bins within `orad` cells, excluding itself, in the same order as `neighbors` in `orient.rs`
/// (dx, then dy, then z). Entries are final bin indices, i.e. indices after dropped cells are removed;
/// a dropped neighbor is written as 0xffffffff. The CPU computes only the edge weights from these
/// lists, so all floating-point work stays on the CPU and the orientation result is bitwise identical
/// to the CPU neighbor search.

/// Bind group 1.
///
/// - `fno`: 1 for kept cells and 0 for dropped ones; after an exclusive scan, the final index.
/// - `noff`: neighbor count per cell; after an exclusive scan, the start of its list.
/// - `nlist`: concatenated lists of final neighbor indices.
/// - `jo`: list start per final index, followed by the total list length.
@group(1) @binding(0) var<storage, read_write> fno: array<u32>;
@group(1) @binding(1) var<storage, read_write> noff: array<u32>;
@group(1) @binding(2) var<storage, read_write> nlist: array<u32>;
@group(1) @binding(3) var<storage, read_write> jo: array<u32>;

/// State of cell `s` (0 fixed, 1 needs estimation, 2 estimated, 3 dropped).
fn st_at(s: u32) -> u32 { return bins[s * 10u + 9u] >> 30u; }

/// True when cell `s` needs a neighbor list: estimated with `|nz| < u.snz`.
fn need0(s: u32) -> bool {
  return st_at(s) == 2u && abs(bitcast<f32>(bins[s * 10u + 5u])) < u.snz;
}

/// True when the bin position maps back to its own cell key.
///
/// Uses the same f32 multiply and floor as `key(pos)` in `orient.rs`; if any bin fails this check the
/// lists are not used, because the CPU would assign the bin to a different cell.
fn pos_cell_ok(s: u32) -> bool {
  let p = bpos(s);
  let c = cell_of(ukeys[s]);
  return sat(floor(p.x * u.inv)) == c.x && sat(floor(p.y * u.inv)) == c.y && sat(floor(p.z * u.inv)) == c.z;
}

/// Writes the cells in the window around `s`, excluding `s`, to `nlist` starting at `base`.
///
/// Dropped cells are written as 0xffffffff. Writes past the end of `nlist` are skipped so the host
/// can detect overflow and grow the buffer for the next call.
///
/// Returns the number of entries, which equals the window count of `est_one` minus one.
fn nb_walk(s: u32, m: u32, base: u32) -> u32 {
  let k = ukeys[s];
  let ux = i32(getb(k, u.bz + u.by, u.bx));
  let uy = i32(getb(k, u.bz, u.by));
  let uz = i32(getb(k, 0u, u.bz));
  let r = u.orad;
  let ylo = max(uy - r, 0);
  let yhi = min(uy + r, u.ry);
  let zlo = u32(max(uz - r, 0));
  let zhi = u32(min(uz + r, u.rz));
  let cap = arrayLength(&nlist);
  var n = 0u;
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
        if (c != s) {
          if (base + n < cap) { nlist[base + n] = select(0xffffffffu, fno[c], st_at(c) != 3u); }
          n = n + 1u;
        }
        c = c + 1u;
      }
      cur = c;
    }
  }
  return n;
}

/// Estimates normals like `k_estimate`, and also marks kept cells in `fno` and stores neighbor
/// counts in `noff`.
///
/// Reuses the estimation window count, so it is only valid when the estimation radius equals the
/// orientation radius. Flags `info[10]` when a bin position does not map back to its cell.
@compute @workgroup_size(64)
fn k_estimate_nb(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let s = gid(w, nw) * 64u + t;
  if (s >= u.n) { return; }
  let m = u32(atomicLoad(&info[7]));
  if (s >= m) {
    fno[s] = 0u;
    noff[s] = 0u;
    return;
  }
  let n = est_one(s, m);
  let st = st_at(s);
  fno[s] = select(1u, 0u, st == 3u);
  if (st != 3u && !pos_cell_ok(s)) { atomicMax(&info[10], 1); }
  var c = 0u;
  if (need0(s)) { c = n - 1u; }
  noff[s] = c;
}

/// Fills `jo` and `nlist` after `fno` and `noff` have been scanned.
///
/// The thread of the last cell writes the total list length (`info[11]`) after the last entry of
/// `jo`.
@compute @workgroup_size(64)
fn k_nfill(@builtin(workgroup_id) w: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let s = gid(w, nw) * 64u + t;
  let m = u32(atomicLoad(&info[7]));
  if (s >= m) { return; }
  let st = st_at(s);
  if (s == m - 1u) {
    jo[fno[s] + select(1u, 0u, st == 3u)] = u32(atomicLoad(&info[11]));
  }
  if (st == 3u) { return; }
  jo[fno[s]] = noff[s];
  if (need0(s)) { _ = nb_walk(s, m, noff[s]); }
}
