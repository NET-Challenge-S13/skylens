//! Sign orientation of estimated normals.
//!
//! PCA gives the axis of a normal but not its direction. [`Orient::Propagate`] decides signs by
//! propagating along a neighbour graph, most confident edge first, in the manner of
//! Hoppe et al. (1992).
//!
//! Seeds, whose sign is decided up front:
//!
//! - bins with trusted input normals (`fixed`);
//! - optionally, bins where an [`Oracle`] (typically the refined distance field accumulated so
//!   far) answers with magnitude at least [`ORACLE_MIN`]; they take the oracle's sign;
//! - estimated bins with `|n_z| >= seed_nz`, oriented to `+z`. The data is captured from above,
//!   so near-horizontal surfaces face up.
//!
//! Propagation passes the sign from a decided bin `i` to a neighbour `j`, taking edges in order
//! of decreasing confidence `|n_i·n_j|`. At a sharp crease (`|n_i·n_j| < crease`) the test uses
//! `n_i` reflected in the perpendicular bisector plane of the two bin positions instead. This
//! picks the outward-consistent sign at both convex edges (roof edges) and concave edges (wall
//! bases).
//!
//! The processing order depends only on (confidence, bin index), so for the same input the
//! result is identical regardless of thread count. Connected components of undecided bins are
//! solved independently and in parallel; this is bitwise identical to running one global heap
//! (see `propagate`).

use crate::bins::Bin;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Reverse;
use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

/// Strategy for orienting estimated normals.
///
/// The default is `Propagate { radius: 2, seed_nz: 0.7, crease: 0.3 }`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Orient {
    /// Keep the `+z` orientation from binning. Oracle answers, if any, are still applied.
    Up,
    /// Propagate signs over the neighbour graph.
    Propagate {
        /// Neighbour radius in bins (Chebyshev distance).
        radius: i32,
        /// Estimated bins with `|n_z|` at least this value become `+z` seeds. A value above 1
        /// disables these seeds; each component is then rooted at its bin with the largest
        /// `|n_z|`.
        seed_nz: f32,
        /// Edges with `|n_i·n_j|` below this value use the reflected-normal test. With 0 only
        /// `n_i·n_j` is used.
        crease: f32,
    },
}

impl Default for Orient {
    fn default() -> Self {
        Orient::Propagate {
            radius: 2,
            seed_nz: 0.7,
            crease: 0.3,
        }
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Integer cell key of a position, with `inv` the reciprocal of the bin size.
fn key(p: [f32; 3], inv: f32) -> [i32; 3] {
    [
        (p[0] * inv).floor() as i32,
        (p[1] * inv).floor() as i32,
        (p[2] * inv).floor() as i32,
    ]
}

/// Query against an existing surface, such as the accumulated refined distance field.
///
/// Takes `(position, normal)` and returns how well the normal agrees with the outward direction
/// of the existing surface, in `-1.0..=1.0`, or `None` if there is no surface there.
pub type Oracle<'a> = &'a (dyn Fn([f32; 3], [f32; 3]) -> Option<f32> + Sync);

/// Oracle answers with at least this magnitude fix the bin's sign and make it a seed.
pub const ORACLE_MIN: f32 = 0.3;

/// Orients the normals of estimated bins in place.
///
/// Bins with `fixed[i] == true` are never changed and act as seeds. All other bins may have
/// their normal negated; nothing else changes. Estimated bins are expected to arrive oriented
/// towards `+z`, as produced by [`crate::bins::bin_points`].
///
/// If `oracle` is given, it is queried first for every non-fixed bin; bins with a confident
/// answer (see [`ORACLE_MIN`]) take its sign and become seeds as well.
///
/// # Arguments
///
/// * `bins` - Bins sorted by cell, with distinct cell keys.
/// * `fixed` - Per bin, whether its normal is trusted. Must have the same length as `bins`.
/// * `bin_size` - Bin edge length in metres.
/// * `mode` - Orientation strategy.
/// * `oracle` - Optional query against an existing surface.
pub fn orient_normals(
    bins: &mut [Bin],
    fixed: &[bool],
    bin_size: f32,
    mode: Orient,
    oracle: Option<Oracle>,
) {
    debug_assert_eq!(bins.len(), fixed.len());
    if fixed.iter().all(|&f| f) {
        return;
    }
    let ans: Option<Vec<Option<f32>>> = oracle.map(|o| {
        bins.par_iter()
            .zip(fixed)
            .map(|(b, &f)| if f { None } else { o(b.pos, b.normal) })
            .collect()
    });
    orient_normals_with(bins, fixed, bin_size, mode, ans);
}

/// Same as [`orient_normals`], with the oracle answers computed in advance.
///
/// Useful when the answers are obtained in bulk, for example by reading the distance field on
/// the GPU. `ans[i]` is the oracle value for bin `i` with its current normal; entries of fixed
/// bins are ignored.
pub fn orient_normals_with(
    bins: &mut [Bin],
    fixed: &[bool],
    bin_size: f32,
    mode: Orient,
    ans: Option<Vec<Option<f32>>>,
) {
    orient_normals_lists(bins, fixed, bin_size, mode, ans, None)
}

/// Precomputed neighbour lists in CSR form, as produced by the GPU binner
/// (`GpuBinner::bin_points_nb` with the `gpu` feature).
///
/// The neighbours of bin `i` are `list[off[i]..off[i + 1]]` with `u32::MAX` entries skipped.
/// They must be the same set, in the same order, as the CPU neighbour search finds. Every bin
/// that is not fixed and has `|n_z| < seed_nz` must have a list. Bins must be in cell order with
/// distinct cell keys.
#[derive(Clone, Copy)]
pub struct Lists<'a> {
    /// Neighbour radius the lists were built with.
    pub radius: i32,
    /// Seed threshold the lists were built with.
    pub seed_nz: f32,
    /// Row offsets, `bins.len() + 1` entries.
    pub off: &'a [u32],
    /// Neighbour indices; `u32::MAX` marks an empty slot.
    pub list: &'a [u32],
}

/// Same as [`orient_normals_with`], optionally using precomputed neighbour lists.
///
/// The lists are used only if their radius and seed threshold match `mode` and their length
/// matches `bins`; otherwise neighbours are searched on the CPU. The result is the same either
/// way. Oracle answers are applied before solving.
pub fn orient_normals_lists(
    bins: &mut [Bin],
    fixed: &[bool],
    bin_size: f32,
    mode: Orient,
    ans: Option<Vec<Option<f32>>>,
    lists: Option<Lists>,
) {
    debug_assert_eq!(bins.len(), fixed.len());
    if fixed.iter().all(|&f| f) {
        return;
    }
    let (fixed2, _) = apply_oracle(bins, fixed, ans, |_| false);
    let fixed = fixed2.as_deref().unwrap_or(fixed);
    if let Orient::Propagate {
        radius,
        seed_nz,
        crease,
    } = mode
    {
        match lists.filter(|l| {
            l.radius == radius
                && l.seed_nz.to_bits() == seed_nz.to_bits()
                && l.off.len() == bins.len() + 1
        }) {
            Some(l) => propagate_lists(bins, fixed, seed_nz, crease, &l),
            None => propagate(bins, fixed, bin_size, radius, seed_nz, crease),
        }
    }
}

/// Orients normals while computing the oracle answers concurrently with the neighbour graph.
///
/// The result is bitwise identical to
/// `orient_normals_with(bins, fixed, bin_size, mode, oracle(bins))`.
///
/// The graph is built speculatively, assuming the oracle neither decides an undecided bin nor
/// flips any bin. If the answers break that assumption, they are applied and the graph is built
/// again. Speculation is skipped when few bins are undecided (fewer than one in eight and no
/// precomputed lists), since the subset solver in `propagate` is cheaper then.
///
/// # Panics
///
/// Panics if there are `u32::MAX` or more bins.
pub fn orient_normals_par<F>(
    bins: &mut [Bin],
    fixed: &[bool],
    bin_size: f32,
    mode: Orient,
    oracle: F,
    lists: Option<Lists>,
) where
    F: FnOnce(&[Bin]) -> Option<Vec<Option<f32>>> + Send,
{
    debug_assert_eq!(bins.len(), fixed.len());
    if fixed.iter().all(|&f| f) {
        return;
    }
    let Orient::Propagate {
        radius,
        seed_nz,
        crease,
    } = mode
    else {
        let ans = oracle(bins);
        apply_oracle(bins, fixed, ans, |_| false);
        return;
    };
    let n = bins.len();
    assert!(n < u32::MAX as usize, "bin index must fit in u32");
    let lists = lists.filter(|l| {
        l.radius == radius && l.seed_nz.to_bits() == seed_nz.to_bits() && l.off.len() == n + 1
    });
    let need0: Vec<bool> = bins
        .par_iter()
        .zip(fixed.par_iter())
        .map(|(b, &f)| !(f || b.normal[2].abs() >= seed_nz))
        .collect();
    let n_need0 = need0.par_iter().filter(|&&x| x).count();
    let spec = n_need0 > 0 && (lists.is_some() || n_need0 * 8 >= n);
    let ro: &[Bin] = bins;
    let (ans, g0) = rayon::join(
        || oracle(ro),
        || {
            spec.then(|| match &lists {
                Some(l) => graph_from_lists(ro, &need0, crease, l),
                None => neighbors(ro, &need0, 1.0 / bin_size, radius, crease),
            })
        },
    );
    let (fixed2, changed) = apply_oracle(bins, fixed, ans, |i| need0[i]);
    let fixed = fixed2.as_deref().unwrap_or(fixed);
    match (g0, changed) {
        (Some(g), false) => solve(bins, &need0, g),
        _ => match &lists {
            Some(l) => propagate_lists(bins, fixed, seed_nz, crease, l),
            None => propagate(bins, fixed, bin_size, radius, seed_nz, crease),
        },
    }
}

/// Applies oracle answers: bins with `|answer| >= ORACLE_MIN` take its sign and become fixed.
///
/// # Returns
///
/// `(fixed, changed)`. `fixed` is the extended fixed mask, or `None` if there were no answers.
/// `changed` is true if any bin was flipped or any bin with `cand(i)` was fixed, meaning a
/// speculatively built graph is no longer valid.
fn apply_oracle(
    bins: &mut [Bin],
    fixed: &[bool],
    ans: Option<Vec<Option<f32>>>,
    cand: impl Fn(usize) -> bool,
) -> (Option<Vec<bool>>, bool) {
    let Some(ans) = ans else { return (None, false) };
    let mut fixed2 = fixed.to_vec();
    let mut changed = false;
    for (i, a) in ans.into_iter().enumerate() {
        let Some(a) = a else { continue };
        if a.abs() < ORACLE_MIN {
            continue;
        }
        if a < 0.0 {
            let n = bins[i].normal;
            bins[i].normal = [-n[0], -n[1], -n[2]];
            changed = true;
        }
        if cand(i) {
            changed = true;
        }
        fixed2[i] = true;
    }
    (Some(fixed2), changed)
}

/// Signed agreement between two bins; positive means their normals point the same way.
///
/// Normally `n_i·n_j`. If `|n_i·n_j| < crease`, `n_i` is reflected in the perpendicular bisector
/// plane of `p_i` and `p_j` and the result is `n_i'·n_j`. Coincident positions fall back to
/// `n_i·n_j`. The magnitude, clamped to 1, is the edge confidence.
#[inline]
fn agree(pi: [f32; 3], ni: [f32; 3], pj: [f32; 3], nj: [f32; 3], crease: f32) -> f32 {
    let d = dot(ni, nj);
    if d.abs() >= crease {
        return d;
    }
    let e = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
    let l2 = dot(e, e);
    if l2 <= 1e-12 {
        return d;
    }
    d - 2.0 * dot(ni, e) * dot(nj, e) / l2
}

/// Graph edge: target bin and the [`agree`] value computed with the original normals.
///
/// Stored together to keep the propagation loop to one memory stream.
#[derive(Clone, Copy, Default)]
struct Edge {
    to: u32,
    v: f32,
}

/// Neighbour graph restricted to undecided bins.
///
/// Rows are in CSR form, indexed by sorted position. Each undecided bin lists its undecided
/// neighbours in `(dx, dy, z, index)` order, and records the best edge coming in from a decided
/// neighbour.
struct Graph {
    /// Bin index to sorted position. Meaningful only for undecided bins.
    rank: Vec<u32>,
    /// CSR row offsets by sorted position.
    start: Vec<u32>,
    /// Undecided neighbours. After components are split, `to` holds the position within the
    /// component instead of the bin index.
    adj: Vec<Edge>,
    /// Per sorted position, the value of the first edge from a decided neighbour with the
    /// highest confidence; ties keep the earlier edge.
    seed: Vec<Option<f32>>,
}

impl Graph {
    /// Range of `adj` holding the neighbours of bin `i`.
    #[inline]
    fn row(&self, i: usize) -> std::ops::Range<usize> {
        let p = self.rank[i] as usize;
        self.start[p] as usize..self.start[p + 1] as usize
    }
}

/// Per sorted position data for the inner loop of `neighbors`.
///
/// Position, normal, z key and index are packed into one 32-byte line so the inner loop reads a
/// single cache line per candidate.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Cell {
    /// Position followed by normal.
    pn: [f32; 6],
    z: i32,
    /// Bin index, with the [`NEED`] bit set if the bin is undecided.
    to: u32,
}

/// Flag bit in [`Cell::to`] marking an undecided bin.
const NEED: u32 = 1 << 31;

/// Result for one slab or chunk: per position undecided neighbour count, the edges, and per
/// position best seed edge value.
type SlabPart = (Vec<u32>, Vec<Edge>, Vec<Option<f32>>);

/// Builds the neighbour graph of undecided bins on the CPU.
///
/// When fewer than one in eight bins is undecided, only bins in the `(2r+1)²` columns around
/// undecided bins are indexed. The selected bins are stably sorted by cell key; bins with the
/// same x then form a contiguous slab, and within a slab bins with the same `(x, y)` form a
/// contiguous row. Each slab is processed in parallel with monotonic cursors over slabs, rows
/// and z, so no hashing is needed. The work within a slab is sequential, so the graph does not
/// depend on the thread count.
///
/// # Panics
///
/// Panics if there are `2^31` or more bins.
fn neighbors(bins: &[Bin], need: &[bool], inv: f32, r: i32, crease: f32) -> Graph {
    let n = bins.len();
    let keys: Vec<[i32; 3]> = bins.par_iter().map(|b| key(b.pos, inv)).collect();
    let n_need = need.iter().filter(|&&x| x).count();
    let mut order: Vec<u32> = if n_need * 8 < n {
        let mut near: FxHashSet<[i32; 2]> = FxHashSet::default();
        for (k, _) in keys.iter().zip(need).filter(|(_, x)| **x) {
            for dx in -r..=r {
                for dy in -r..=r {
                    near.insert([k[0] + dx, k[1] + dy]);
                }
            }
        }
        (0..n as u32)
            .filter(|&i| near.contains(&[keys[i as usize][0], keys[i as usize][1]]))
            .collect()
    } else {
        (0..n as u32).collect()
    };
    order.par_sort_by_key(|&i| keys[i as usize]);
    let m = order.len();
    let sk: Vec<[i32; 3]> = order.iter().map(|&i| keys[i as usize]).collect();
    drop(keys);
    let mut rows: Vec<u32> = Vec::new();
    let mut slabs: Vec<u32> = Vec::new();
    for p in 0..m {
        if p == 0 || sk[p][0] != sk[p - 1][0] {
            slabs.push(rows.len() as u32);
        }
        if p == 0 || sk[p][0] != sk[p - 1][0] || sk[p][1] != sk[p - 1][1] {
            rows.push(p as u32);
        }
    }
    rows.push(m as u32);
    slabs.push((rows.len() - 1) as u32);
    let row_y: Vec<i32> = rows[..rows.len() - 1]
        .iter()
        .map(|&s| sk[s as usize][1])
        .collect();
    let slab_x: Vec<i32> = slabs[..slabs.len() - 1]
        .iter()
        .map(|&ri| sk[rows[ri as usize] as usize][0])
        .collect();

    assert!(n < NEED as usize, "bin index must fit in 31 bits");
    let cells: Vec<Cell> = order
        .par_iter()
        .zip(sk.par_iter())
        .map(|(&i, k)| {
            let b = &bins[i as usize];
            Cell {
                pn: [
                    b.pos[0],
                    b.pos[1],
                    b.pos[2],
                    b.normal[0],
                    b.normal[1],
                    b.normal[2],
                ],
                z: k[2],
                to: i | if need[i as usize] { NEED } else { 0 },
            }
        })
        .collect();
    let parts: Vec<SlabPart> = (0..slab_x.len())
        .into_par_iter()
        .map(|si| {
            let (p0, p1) = (
                rows[slabs[si] as usize] as usize,
                rows[slabs[si + 1] as usize] as usize,
            );
            let mut cnt = vec![0u32; p1 - p0];
            let mut seed = vec![None; p1 - p0];
            let mut out: Vec<Edge> = Vec::new();
            if !cells[p0..p1].iter().any(|c| c.to & NEED != 0) {
                return (cnt, out, seed);
            }
            let mut nslab: Vec<(usize, usize)> = Vec::new();
            for dx in -r..=r {
                if let Ok(t) = slab_x.binary_search(&(slab_x[si] + dx)) {
                    nslab.push((slabs[t] as usize, slabs[t + 1] as usize));
                }
            }
            let mut lo: Vec<usize> = Vec::new();
            let mut hi: Vec<usize> = Vec::new();
            for ri in slabs[si] as usize..slabs[si + 1] as usize {
                let (s0, e0) = (rows[ri] as usize, rows[ri + 1] as usize);
                let y = row_y[ri];
                for c in nslab.iter_mut() {
                    while c.0 < c.1 && row_y[c.0] < y - r {
                        c.0 += 1;
                    }
                }
                if !cells[s0..e0].iter().any(|c| c.to & NEED != 0) {
                    continue;
                }
                lo.clear();
                hi.clear();
                for &(c, e) in &nslab {
                    let mut t = c;
                    while t < e && row_y[t] <= y + r {
                        lo.push(rows[t] as usize);
                        hi.push(rows[t + 1] as usize);
                        t += 1;
                    }
                }
                for p in s0..e0 {
                    let me = &cells[p];
                    if me.to & NEED == 0 {
                        continue;
                    }
                    let kz = me.z;
                    let (pi, ni) = (
                        [me.pn[0], me.pn[1], me.pn[2]],
                        [me.pn[3], me.pn[4], me.pn[5]],
                    );
                    let before = out.len();
                    let mut sb: Option<f32> = None;
                    let mut best = -1.0f32;
                    for t in 0..lo.len() {
                        let h = hi[t];
                        let mut c = lo[t];
                        while c < h && cells[c].z < kz - r {
                            c += 1;
                        }
                        lo[t] = c;
                        while c < h {
                            let o = &cells[c];
                            if o.z > kz + r {
                                break;
                            }
                            if c != p {
                                let v = agree(
                                    pi,
                                    ni,
                                    [o.pn[0], o.pn[1], o.pn[2]],
                                    [o.pn[3], o.pn[4], o.pn[5]],
                                    crease,
                                );
                                if o.to & NEED != 0 {
                                    out.push(Edge {
                                        to: o.to & !NEED,
                                        v,
                                    });
                                } else {
                                    let s = v.abs().min(1.0);
                                    if s > best {
                                        best = s;
                                        sb = Some(v);
                                    }
                                }
                            }
                            c += 1;
                        }
                    }
                    cnt[p - p0] = (out.len() - before) as u32;
                    seed[p - p0] = sb;
                }
            }
            (cnt, out, seed)
        })
        .collect();
    let mut rank = vec![0u32; n];
    for (p, &i) in order.iter().enumerate() {
        rank[i as usize] = p as u32;
    }
    assemble(parts, rank)
}

/// Concatenates per-slab results, given in sorted position order, into a [`Graph`].
///
/// The edge arrays are copied into place in parallel.
fn assemble(parts: Vec<SlabPart>, rank: Vec<u32>) -> Graph {
    let m: usize = parts.iter().map(|p| p.0.len()).sum();
    let mut start = Vec::with_capacity(m + 1);
    let mut seed = Vec::with_capacity(m);
    let mut at = 0u32;
    for (cnt, _, sd) in &parts {
        for &c in cnt {
            start.push(at);
            at += c;
        }
        seed.extend_from_slice(sd);
    }
    start.push(at);
    let mut adj = vec![Edge::default(); at as usize];
    let mut jobs = Vec::with_capacity(parts.len());
    let mut rest = &mut adj[..];
    for (_, out, _) in &parts {
        let (a, b) = rest.split_at_mut(out.len());
        jobs.push((out, a));
        rest = b;
    }
    jobs.into_par_iter()
        .for_each(|(out, a)| a.copy_from_slice(out));
    drop(parts);
    Graph {
        rank,
        start,
        adj,
        seed,
    }
}

/// Builds the neighbour graph from precomputed [`Lists`].
///
/// Bins are already in cell order with distinct keys, so sorted position equals bin index.
/// The lists have the same order as `neighbors` produces, so the resulting graph is identical.
fn graph_from_lists(bins: &[Bin], need: &[bool], crease: f32, l: &Lists) -> Graph {
    let n = bins.len();
    const CH: usize = 4096;
    let parts: Vec<SlabPart> = (0..n.div_ceil(CH))
        .into_par_iter()
        .map(|ci| {
            let (p0, p1) = (ci * CH, ((ci + 1) * CH).min(n));
            let mut cnt = vec![0u32; p1 - p0];
            let mut seed = vec![None; p1 - p0];
            let mut out: Vec<Edge> = Vec::new();
            for p in p0..p1 {
                if !need[p] {
                    continue;
                }
                let bi = &bins[p];
                let before = out.len();
                let mut sb: Option<f32> = None;
                let mut best = -1.0f32;
                for &c in &l.list[l.off[p] as usize..l.off[p + 1] as usize] {
                    if c == u32::MAX {
                        continue;
                    }
                    let bj = &bins[c as usize];
                    let v = agree(bi.pos, bi.normal, bj.pos, bj.normal, crease);
                    if need[c as usize] {
                        out.push(Edge { to: c, v });
                    } else {
                        let s = v.abs().min(1.0);
                        if s > best {
                            best = s;
                            sb = Some(v);
                        }
                    }
                }
                cnt[p - p0] = (out.len() - before) as u32;
                seed[p - p0] = sb;
            }
            (cnt, out, seed)
        })
        .collect();
    assemble(parts, (0..n as u32).collect())
}

/// Lock-free union-find lookup with path halving.
///
/// Roots are always linked under the smaller root, so a root is the minimum index of its
/// component. Path halving only ever moves a link to an ancestor, so concurrent updates are
/// safe; a failed compare-exchange is simply ignored.
fn find(par: &[AtomicU32], mut x: u32) -> u32 {
    loop {
        let p = par[x as usize].load(Relaxed);
        if p == x {
            return x;
        }
        let g = par[p as usize].load(Relaxed);
        if g != p {
            let _ = par[x as usize].compare_exchange(p, g, Relaxed, Relaxed);
        }
        x = g;
    }
}

/// Lock-free union: links the larger of the two roots under the smaller, retrying on contention.
fn union(par: &[AtomicU32], a: u32, b: u32) {
    let (mut a, mut b) = (a, b);
    loop {
        a = find(par, a);
        b = find(par, b);
        if a == b {
            return;
        }
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        if par[hi as usize]
            .compare_exchange(hi, lo, Relaxed, Relaxed)
            .is_ok()
        {
            return;
        }
    }
}

/// Decides whether the target bin flips, from the edge value `v` measured with the original
/// normals and whether the source bin was flipped.
///
/// Equivalent to evaluating `agree(source normal after flipping, n_j) < 0`. Negation is exact,
/// so zero and NaN are handled identically.
#[inline]
fn flip_of(v: f32, src_flipped: bool) -> bool {
    if src_flipped { v > 0.0 } else { v < 0.0 }
}

/// Propagates signs to undecided bins over the neighbour graph, most confident edge first
/// (Prim-style maximum spanning tree).
///
/// Connected components of undecided bins only meet through decided bins (seeds), and seeds
/// never change. Solving each component separately is therefore bitwise identical to running
/// one heap over everything, so components are solved in parallel. The heap order is
/// (confidence, smaller index first); two entries for the same bin always differ in confidence,
/// so the source bin never affects the order.
///
/// When fewer than one in eight bins is undecided, only the bins in the columns within `r`
/// cells (in x and y) of an undecided bin are extracted and solved. All neighbours of undecided
/// bins lie in those columns, and the subset preserves index order, so edges, components and
/// heap order are unchanged and the result is bitwise identical to solving everything, without
/// allocations or scans proportional to the total bin count.
///
/// # Panics
///
/// Panics if there are `u32::MAX` or more bins.
fn propagate(bins: &mut [Bin], fixed: &[bool], bin_size: f32, r: i32, seed_nz: f32, crease: f32) {
    let n = bins.len();
    assert!(n < u32::MAX as usize, "bin index must fit in u32");
    let need_at = |b: &Bin, f: bool| !(f || b.normal[2].abs() >= seed_nz);
    let n_need = bins
        .par_iter()
        .zip(fixed.par_iter())
        .filter(|(b, f)| need_at(b, **f))
        .count();
    if n_need == 0 {
        return;
    }
    if n_need * 8 < n {
        let inv = 1.0 / bin_size;
        let near: FxHashSet<[i32; 2]> = bins
            .iter()
            .zip(fixed)
            .filter(|(b, f)| need_at(b, **f))
            .flat_map(|(b, _)| {
                let k = key(b.pos, inv);
                (-r..=r).flat_map(move |dx| (-r..=r).map(move |dy| [k[0] + dx, k[1] + dy]))
            })
            .collect();
        let sub: Vec<u32> = (0..n as u32)
            .into_par_iter()
            .filter(|&i| {
                let k = key(bins[i as usize].pos, inv);
                near.contains(&[k[0], k[1]])
            })
            .collect();
        let mut sb: Vec<Bin> = sub.iter().map(|&i| bins[i as usize]).collect();
        let sf: Vec<bool> = sub.iter().map(|&i| fixed[i as usize]).collect();
        propagate_all(&mut sb, &sf, bin_size, r, seed_nz, crease);
        for (b, &i) in sb.iter().zip(&sub) {
            bins[i as usize].normal = b.normal;
        }
        return;
    }
    propagate_all(bins, fixed, bin_size, r, seed_nz, crease);
}

/// Solves all bins without subset extraction; the core of [`propagate`].
///
/// Edge confidence is symmetric, so only the neighbours of undecided bins are needed.
fn propagate_all(
    bins: &mut [Bin],
    fixed: &[bool],
    bin_size: f32,
    r: i32,
    seed_nz: f32,
    crease: f32,
) {
    let n = bins.len();
    let need: Vec<bool> = (0..n)
        .map(|i| !(fixed[i] || bins[i].normal[2].abs() >= seed_nz))
        .collect();
    let g = neighbors(bins, &need, 1.0 / bin_size, r, crease);
    solve(bins, &need, g);
}

/// Propagates using precomputed neighbour [`Lists`]. Bitwise identical to [`propagate`].
fn propagate_lists(bins: &mut [Bin], fixed: &[bool], seed_nz: f32, crease: f32, l: &Lists) {
    let need: Vec<bool> = bins
        .par_iter()
        .zip(fixed.par_iter())
        .map(|(b, &f)| !(f || b.normal[2].abs() >= seed_nz))
        .collect();
    if !need.iter().any(|&x| x) {
        return;
    }
    let g = graph_from_lists(bins, &need, crease, l);
    solve(bins, &need, g);
}

/// Splits the undecided bins of `g` into connected components and orients each in parallel.
///
/// Components are found with a lock-free union-find. They are numbered in order of their root
/// (minimum bin index), members are listed in index order, and `loc` holds each bin's position
/// within its component. Counting is done in parallel per chunk of 4096 bins and the chunks are
/// combined in order, so the numbering does not depend on the thread count. Graph
/// edges are then rewritten to component-local positions. Large components are scheduled first;
/// the result does not depend on scheduling.
fn solve(bins: &mut [Bin], need: &[bool], mut g: Graph) {
    let n = bins.len();

    let par: Vec<AtomicU32> = (0..n as u32).into_par_iter().map(AtomicU32::new).collect();
    (0..n).into_par_iter().filter(|&i| need[i]).for_each(|i| {
        for e in &g.adj[g.row(i)] {
            if e.to as usize > i {
                union(&par, i as u32, e.to);
            }
        }
    });
    const CH: usize = 4096;
    let root: Vec<u32> = (0..n)
        .into_par_iter()
        .map(|i| {
            if need[i] {
                find(&par, i as u32)
            } else {
                u32::MAX
            }
        })
        .collect();
    drop(par);
    let is_root = |i: usize| root[i] == i as u32;
    let rc: Vec<u32> = root
        .par_chunks(CH)
        .enumerate()
        .map(|(c, ch)| (0..ch.len()).filter(|&j| is_root(c * CH + j)).count() as u32)
        .collect();
    let mut rbase = Vec::with_capacity(rc.len());
    let mut ncomp = 0u32;
    for &k in &rc {
        rbase.push(ncomp);
        ncomp += k;
    }
    let mut cid = vec![u32::MAX; n];
    cid.par_chunks_mut(CH).enumerate().for_each(|(c, ch)| {
        let mut k = rbase[c];
        for (j, x) in ch.iter_mut().enumerate() {
            if is_root(c * CH + j) {
                *x = k;
                k += 1;
            }
        }
    });
    let comp: Vec<u32> = (0..n)
        .into_par_iter()
        .map(|i| {
            if need[i] {
                cid[root[i] as usize]
            } else {
                u32::MAX
            }
        })
        .collect();
    drop(cid);
    drop(root);
    let counts: Vec<FxHashMap<u32, u32>> = comp
        .par_chunks(CH)
        .map(|ch| {
            let mut m: FxHashMap<u32, u32> = FxHashMap::default();
            for &c in ch.iter().filter(|&&c| c != u32::MAX) {
                *m.entry(c).or_default() += 1;
            }
            m
        })
        .collect();
    let mut sizes = vec![0u32; ncomp as usize];
    let bases: Vec<FxHashMap<u32, u32>> = counts
        .into_iter()
        .map(|m| {
            m.into_iter()
                .map(|(c, k)| {
                    let b = sizes[c as usize];
                    sizes[c as usize] += k;
                    (c, b)
                })
                .collect()
        })
        .collect();
    let mut loc = vec![0u32; n];
    loc.par_chunks_mut(CH)
        .zip(comp.par_chunks(CH))
        .zip(bases.into_par_iter())
        .for_each(|((lc, cc), mut run)| {
            for (x, &c) in lc.iter_mut().zip(cc) {
                if c != u32::MAX {
                    let r = run.get_mut(&c).unwrap();
                    *x = *r;
                    *r += 1;
                }
            }
        });
    g.adj.par_iter_mut().for_each(|e| e.to = loc[e.to as usize]);
    let mut off = Vec::with_capacity(sizes.len() + 1);
    off.push(0u32);
    for &s in &sizes {
        off.push(off.last().unwrap() + s);
    }
    let members: Vec<u32> = {
        let mut mem = vec![0u32; *off.last().unwrap() as usize];
        let dst = SendPtr(mem.as_mut_ptr());
        (0..n).into_par_iter().for_each(|i| {
            let c = comp[i];
            if c != u32::MAX {
                let d = &dst;
                // SAFETY: (component, position in component) is unique per bin, so no two
                // iterations write the same slot, and every slot is within `mem`.
                unsafe { d.0.add((off[c as usize] + loc[i]) as usize).write(i as u32) };
            }
        });
        mem
    };
    let mut jobs: Vec<u32> = (0..sizes.len() as u32).collect();
    jobs.sort_unstable_by_key(|&c| (Reverse(sizes[c as usize]), c));

    let bins_ro: &[Bin] = bins;
    let flips: Vec<Vec<u32>> = jobs
        .par_iter()
        .with_max_len(1)
        .map(|&c| {
            let mem = &members[off[c as usize] as usize..off[c as usize + 1] as usize];
            orient_component(bins_ro, &g, mem)
        })
        .collect();
    for f in flips {
        for j in f {
            let b = &mut bins[j as usize];
            b.normal = [-b.normal[0], -b.normal[1], -b.normal[2]];
        }
    }
}

/// Raw pointer wrapper for parallel writes to disjoint slots.
struct SendPtr(*mut u32);
// SAFETY: every user writes only slots that no other thread writes.
unsafe impl Sync for SendPtr {}

/// Indexed `D`-ary max-heap with at most one entry per bin.
///
/// Keys are only ever increased, so no stale entries accumulate. A key is
/// `(confidence bits << 32) | !(position in component)`. Positions follow bin index order, and
/// confidence lies in `0..=1`, where the `f32` bit pattern orders like the value. Keys are
/// unique per bin, so the pop order does not depend on insertion order and matches a heap that
/// skips stale entries.
struct KeyHeap {
    h: Vec<u64>,
    /// Position in component to heap slot, `u32::MAX` if absent.
    at: Vec<u32>,
}

impl KeyHeap {
    fn new(k: usize) -> Self {
        KeyHeap {
            h: Vec::new(),
            at: vec![u32::MAX; k],
        }
    }

    #[inline]
    fn key(s: f32, id: usize) -> u64 {
        ((s.to_bits() as u64) << 32) | (!(id as u32)) as u64
    }

    #[inline]
    fn id(key: u64) -> usize {
        !(key as u32) as usize
    }

    /// Inserts a new entry, or raises the key of the existing entry for the same bin.
    #[inline]
    fn raise(&mut self, key: u64) {
        let id = Self::id(key);
        let mut p = self.at[id] as usize;
        if p == u32::MAX as usize {
            p = self.h.len();
            self.h.push(key);
        } else {
            debug_assert!(self.h[p] < key);
        }
        while p > 0 {
            let q = (p - 1) / D;
            let kq = self.h[q];
            if kq >= key {
                break;
            }
            self.h[p] = kq;
            self.at[Self::id(kq)] = p as u32;
            p = q;
        }
        self.h[p] = key;
        self.at[id] = p as u32;
    }

    /// Removes and returns the largest key.
    #[inline]
    fn pop(&mut self) -> Option<u64> {
        let top = *self.h.first()?;
        self.at[Self::id(top)] = u32::MAX;
        let last = self.h.pop().unwrap();
        let n = self.h.len();
        if n > 0 {
            let mut p = 0usize;
            loop {
                let c0 = D * p + 1;
                if c0 >= n {
                    break;
                }
                let mut c = c0;
                let mut kc = self.h[c0];
                for t in c0 + 1..(c0 + D).min(n) {
                    if self.h[t] > kc {
                        c = t;
                        kc = self.h[t];
                    }
                }
                if kc <= last {
                    break;
                }
                self.h[p] = kc;
                self.at[Self::id(kc)] = p as u32;
                p = c;
            }
            self.h[p] = last;
            self.at[Self::id(last)] = p as u32;
        }
        Some(top)
    }
}

/// Heap arity. Key increases are far more frequent than pops, so a wide, shallow heap is
/// faster.
const D: usize = 16;

/// Source marker meaning the best incoming edge comes from a seed.
const SEED: u32 = u32::MAX;

/// Orients one component of undecided bins.
///
/// Bins with an incoming seed edge start in the heap. A component that no seed reaches is
/// rooted at its bin with the largest `|n_z|` (smallest index on ties), which is oriented to
/// `+z`. For every bin, `src` and `srcv` hold the source (position in component, or [`SEED`])
/// and value of its best incoming edge so far.
///
/// # Arguments
///
/// * `mem` - Bin indices of the component in increasing order.
///
/// # Returns
///
/// The bin indices whose normals must be negated.
fn orient_component(bins: &[Bin], g: &Graph, mem: &[u32]) -> Vec<u32> {
    let k = mem.len();
    let mut done = vec![false; k];
    let mut flip = vec![false; k];
    let mut best = vec![-1.0f32; k];
    let mut src = vec![SEED; k];
    let mut srcv = vec![0.0f32; k];
    let mut heap = KeyHeap::new(k);
    for (lj, &j) in mem.iter().enumerate() {
        if let Some(v) = g.seed[g.rank[j as usize] as usize] {
            let s = v.abs().min(1.0);
            best[lj] = s;
            srcv[lj] = v;
            heap.raise(KeyHeap::key(s, lj));
        }
    }
    let mut root = None;
    if heap.h.is_empty() {
        let mut r = 0usize;
        for i in 1..k {
            let (a, b) = (
                bins[mem[i] as usize].normal[2].abs(),
                bins[mem[r] as usize].normal[2].abs(),
            );
            if a.total_cmp(&b).is_gt() {
                r = i;
            }
        }
        flip[r] = bins[mem[r] as usize].normal[2] < 0.0;
        root = Some(r);
    }
    let mut next = root;
    loop {
        let lj = match next.take() {
            Some(r) => {
                done[r] = true;
                r
            }
            None => {
                let Some(e) = heap.pop() else { break };
                let lj = KeyHeap::id(e);
                let fi = src[lj] != SEED && flip[src[lj] as usize];
                flip[lj] = flip_of(srcv[lj], fi);
                done[lj] = true;
                lj
            }
        };
        let row = g.row(mem[lj] as usize);
        for &Edge { to, v } in &g.adj[row] {
            let x = to as usize;
            if done[x] {
                continue;
            }
            let s = v.abs().min(1.0);
            if s > best[x] {
                best[x] = s;
                src[x] = lj as u32;
                srcv[x] = v;
                heap.raise(KeyHeap::key(s, x));
            }
        }
    }
    debug_assert!(done.iter().all(|&d| d));
    mem.iter()
        .zip(&flip)
        .filter(|p| *p.1)
        .map(|p| *p.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straightforward reference: one global heap and hash-based neighbour lookup. The parallel
    /// implementation must match it bitwise.
    mod reference {
        use super::super::{agree, key};
        use crate::bins::Bin;
        use rayon::prelude::*;
        use rustc_hash::{FxHashMap, FxHashSet};
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        /// Neighbour lists in CSR form; bins with `need == false` get an empty list.
        ///
        /// Bins of the same `(x, y)` column are grouped in z order, and only the `(2r+1)²`
        /// surrounding columns are searched.
        fn neighbors(bins: &[Bin], need: &[bool], inv: f32, r: i32) -> (Vec<u32>, Vec<u32>) {
            let n = bins.len();
            let keys: Vec<[i32; 3]> = bins.par_iter().map(|b| key(b.pos, inv)).collect();
            let n_need = need.iter().filter(|&&x| x).count();
            let mut order: Vec<u32> = if n_need * 8 < n {
                let mut near: FxHashSet<[i32; 2]> = FxHashSet::default();
                for (k, _) in keys.iter().zip(need).filter(|(_, x)| **x) {
                    for dx in -r..=r {
                        for dy in -r..=r {
                            near.insert([k[0] + dx, k[1] + dy]);
                        }
                    }
                }
                (0..n as u32)
                    .filter(|&i| near.contains(&[keys[i as usize][0], keys[i as usize][1]]))
                    .collect()
            } else {
                (0..n as u32).collect()
            };
            order.sort_by_key(|&i| keys[i as usize]);
            let mut cols: FxHashMap<[i32; 2], (u32, u32)> = FxHashMap::default();
            for (o, &i) in order.iter().enumerate() {
                let k = keys[i as usize];
                cols.entry([k[0], k[1]])
                    .and_modify(|e| e.1 = o as u32 + 1)
                    .or_insert((o as u32, o as u32 + 1));
            }
            let chunks: Vec<(Vec<u32>, Vec<u32>)> = (0..n)
                .collect::<Vec<_>>()
                .par_chunks(4096)
                .map(|ids| {
                    let mut cnt = Vec::with_capacity(ids.len());
                    let mut out = Vec::new();
                    for &i in ids {
                        if !need[i] {
                            cnt.push(0);
                            continue;
                        }
                        let k = keys[i];
                        let before = out.len();
                        for dx in -r..=r {
                            for dy in -r..=r {
                                let Some(&(s, e)) = cols.get(&[k[0] + dx, k[1] + dy]) else {
                                    continue;
                                };
                                let col = &order[s as usize..e as usize];
                                let lo = col.partition_point(|&j| keys[j as usize][2] < k[2] - r);
                                for &j in &col[lo..] {
                                    if keys[j as usize][2] > k[2] + r {
                                        break;
                                    }
                                    if j as usize != i {
                                        out.push(j);
                                    }
                                }
                            }
                        }
                        cnt.push((out.len() - before) as u32);
                    }
                    (out, cnt)
                })
                .collect();
            let mut start = Vec::with_capacity(n + 1);
            let mut nbr = Vec::with_capacity(chunks.iter().map(|c| c.0.len()).sum());
            let mut at = 0u32;
            for (out, cnt) in chunks {
                for c in cnt {
                    start.push(at);
                    at += c;
                }
                nbr.extend_from_slice(&out);
            }
            start.push(at);
            (start, nbr)
        }

        /// Entries are `(confidence bits, Reverse(target), source)`; confidence lies in `0..=1`,
        /// where the bit pattern orders like the value.
        type Heap = BinaryHeap<(u32, Reverse<u32>, u32)>;

        /// Propagates over all bins with a single heap that skips stale entries.
        ///
        /// Components no seed reaches are rooted, in order of decreasing `|n_z|`, at the first
        /// bin still undecided, which is oriented to `+z`.
        pub(super) fn propagate(
            bins: &mut [Bin],
            fixed: &[bool],
            bin_size: f32,
            r: i32,
            seed_nz: f32,
            crease: f32,
        ) {
            let n = bins.len();
            let mut done = vec![false; n];
            for i in 0..n {
                if fixed[i] || bins[i].normal[2].abs() >= seed_nz {
                    done[i] = true;
                }
            }
            let need: Vec<bool> = done.iter().map(|d| !d).collect();
            let (start, nbr) = neighbors(bins, &need, 1.0 / bin_size, r);

            let mut best = vec![-1.0f32; n];
            let mut heap: Heap = BinaryHeap::new();

            let push_from =
                |i: usize, bins: &[Bin], done: &[bool], best: &mut [f32], heap: &mut Heap| {
                    let bi = bins[i];
                    for &j in &nbr[start[i] as usize..start[i + 1] as usize] {
                        let ju = j as usize;
                        if done[ju] {
                            continue;
                        }
                        let bj = bins[ju];
                        let s = agree(bi.pos, bi.normal, bj.pos, bj.normal, crease)
                            .abs()
                            .min(1.0);
                        if s > best[ju] {
                            best[ju] = s;
                            heap.push((s.to_bits(), Reverse(j), i as u32));
                        }
                    }
                };

            let run = |bins: &mut [Bin], done: &mut [bool], best: &mut [f32], heap: &mut Heap| {
                while let Some((_, Reverse(j), i)) = heap.pop() {
                    let ju = j as usize;
                    if done[ju] {
                        continue;
                    }
                    let bi = bins[i as usize];
                    let v = agree(bi.pos, bi.normal, bins[ju].pos, bins[ju].normal, crease);
                    let bj = &mut bins[ju];
                    if v < 0.0 {
                        bj.normal = [-bj.normal[0], -bj.normal[1], -bj.normal[2]];
                    }
                    done[ju] = true;
                    push_from(ju, bins, done, best, heap);
                }
            };

            for j in 0..n {
                if done[j] {
                    continue;
                }
                let bj = bins[j];
                for &i in &nbr[start[j] as usize..start[j + 1] as usize] {
                    let iu = i as usize;
                    if !done[iu] {
                        continue;
                    }
                    let bi = bins[iu];
                    let s = agree(bi.pos, bi.normal, bj.pos, bj.normal, crease)
                        .abs()
                        .min(1.0);
                    if s > best[j] {
                        best[j] = s;
                        heap.push((s.to_bits(), Reverse(j as u32), i));
                    }
                }
            }
            run(bins, &mut done, &mut best, &mut heap);

            let mut rest: Vec<u32> = (0..n as u32).filter(|&i| !done[i as usize]).collect();
            rest.sort_by(|&a, &b| {
                bins[b as usize].normal[2]
                    .abs()
                    .total_cmp(&bins[a as usize].normal[2].abs())
                    .then(a.cmp(&b))
            });
            for i in rest {
                let iu = i as usize;
                if done[iu] {
                    continue;
                }
                if bins[iu].normal[2] < 0.0 {
                    bins[iu].normal = [
                        -bins[iu].normal[0],
                        -bins[iu].normal[1],
                        -bins[iu].normal[2],
                    ];
                }
                done[iu] = true;
                push_from(iu, bins, &done, &mut best, &mut heap);
                run(bins, &mut done, &mut best, &mut heap);
            }
        }
    }

    fn bin(p: [f32; 3], n: [f32; 3]) -> Bin {
        Bin {
            pos: p,
            normal: n,
            rgb: [0.0; 3],
            count: 1,
        }
    }

    /// A box-shaped building (roof and four walls) on a ground plane.
    ///
    /// Wall normals are inserted with a deliberately wrong sign when `flip_all_walls` is set, as
    /// `+z` alignment would leave them arbitrary. Returns the bins and the true outward normals.
    fn box_scene(flip_all_walls: bool) -> (Vec<Bin>, Vec<[f32; 3]>) {
        let s = 0.1;
        let mut v = Vec::new();
        let mut truth = Vec::new();
        let (x0, x1, y0, y1, h) = (0.0f32, 2.0f32, 0.0f32, 2.0f32, 1.5f32);
        for i in -10..30 {
            for j in -10..30 {
                let (x, y) = (i as f32 * s + 0.05, j as f32 * s + 0.05);
                if x > x0 && x < x1 && y > y0 && y < y1 {
                    continue;
                }
                v.push(bin([x, y, 0.05], [0.0, 0.0, 1.0]));
                truth.push([0.0, 0.0, 1.0]);
            }
        }
        for i in 0..20 {
            for j in 0..20 {
                v.push(bin(
                    [i as f32 * s + 0.05, j as f32 * s + 0.05, h + 0.05],
                    [0.0, 0.0, 1.0],
                ));
                truth.push([0.0, 0.0, 1.0]);
            }
        }
        let sg = if flip_all_walls { -1.0 } else { 1.0 };
        for k in 1..15 {
            let z = k as f32 * s + 0.05;
            for t in 1..19 {
                let u = t as f32 * s + 0.05;
                for (p, n) in [
                    ([x0 + 0.05, u, z], [-1.0, 0.0, 0.0]),
                    ([x1 - 0.05, u, z], [1.0, 0.0, 0.0]),
                    ([u, y0 + 0.05, z], [0.0, -1.0, 0.0]),
                    ([u, y1 - 0.05, z], [0.0, 1.0, 0.0]),
                ] {
                    v.push(bin(p, [sg * n[0], sg * n[1], sg * n[2]]));
                    truth.push(n);
                }
            }
        }
        (v, truth)
    }

    #[test]
    fn propagation_orients_walls_outward() {
        for flip in [false, true] {
            let (mut b, truth) = box_scene(flip);
            let fixed = vec![false; b.len()];
            orient_normals(&mut b, &fixed, 0.1, Orient::default(), None);
            let bad = b
                .iter()
                .zip(&truth)
                .filter(|(x, t)| dot(x.normal, **t) < 0.0)
                .count();
            assert!(bad * 50 < b.len(), "flipped {bad}/{}", b.len());
        }
    }

    /// Matches the reference bitwise on random scenes.
    ///
    /// The scenes contain several components, components without seeds (some clusters are
    /// vertical), crease edges, and ties: equal-confidence edges and equal-`|n_z|` root candidates
    /// (a seedless wall with alternating signs).
    #[test]
    fn matches_single_heap_reference() {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32
        };
        for case in 0..6 {
            let (mut b, _) = box_scene(case % 2 == 1);
            for k in 0..400 {
                let c = [
                    (k % 5) as f32 * 1.7 + 5.0,
                    (k / 80) as f32 * 1.3 - 3.0,
                    rnd() * 2.0,
                ];
                let p = [c[0] + rnd() * 0.3, c[1] + rnd() * 0.3, c[2]];
                let mut n = [
                    rnd() - 0.5,
                    rnd() - 0.5,
                    (rnd() - 0.5) * if case < 3 { 0.4 } else { 2.0 },
                ];
                let l = dot(n, n).sqrt();
                n = [n[0] / l, n[1] / l, n[2] / l];
                if n[2] < 0.0 {
                    n = [-n[0], -n[1], -n[2]];
                }
                b.push(bin(p, n));
            }
            for k in 0..60 {
                let sx = if k % 2 == 0 { 1.0 } else { -1.0 };
                b.push(bin(
                    [-3.0 + (k % 10) as f32 * 0.1, -3.0, (k / 10) as f32 * 0.1],
                    [sx, 0.0, 0.0],
                ));
            }
            let fixed: Vec<bool> = (0..b.len()).map(|i| case >= 4 && i % 7 == 0).collect();
            let seed_nz = if case == 5 { 2.0 } else { 0.7 };
            let mut a = b.clone();
            reference::propagate(&mut a, &fixed, 0.1, 2, seed_nz, 0.3);
            for t in [1, 3] {
                let mut c = b.clone();
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .unwrap();
                pool.install(|| propagate(&mut c, &fixed, 0.1, 2, seed_nz, 0.3));
                let bits = |v: &[Bin]| {
                    v.iter()
                        .map(|x| x.normal.map(f32::to_bits))
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(&a), bits(&c), "case {case} threads {t}");
            }
            assert_ne!(
                a.iter().map(|x| x.normal).collect::<Vec<_>>(),
                b.iter().map(|x| x.normal).collect::<Vec<_>>()
            );
        }
    }

    /// With few undecided bins, solving the extracted subset matches solving everything bitwise.
    ///
    /// Undecided bins are scattered over a wide trusted ground plane, and their normals are
    /// tilted below the seed threshold.
    #[test]
    fn sparse_subset_matches_full() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32
        };
        for case in 0..4 {
            let (mut b, _) = box_scene(case % 2 == 1);
            for i in 0..200 {
                for j in 0..100 {
                    b.push(bin(
                        [
                            i as f32 * 0.1 + 3.05,
                            j as f32 * 0.1 - 4.95,
                            0.05 + 0.02 * rnd(),
                        ],
                        [0.0, 0.0, 1.0],
                    ));
                }
            }
            let fixed: Vec<bool> = (0..b.len())
                .map(|i| i % 97 != (case * 13) % 97 || i % 3 == 0)
                .collect();
            for (x, f) in b.iter_mut().zip(&fixed) {
                if !*f {
                    let mut n = [rnd() - 0.5, rnd() - 0.5, (rnd() - 0.5) * 0.6];
                    let l = dot(n, n).sqrt();
                    n = [n[0] / l, n[1] / l, n[2] / l];
                    x.normal = n;
                }
            }
            let need = b
                .iter()
                .zip(&fixed)
                .filter(|(x, f)| !(**f || x.normal[2].abs() >= 0.7))
                .count();
            assert!(need > 0 && need * 8 < b.len(), "case {case}: need {need}");
            let mut a = b.clone();
            propagate_all(&mut a, &fixed, 0.1, 2, 0.7, 0.3);
            let mut c = b.clone();
            propagate(&mut c, &fixed, 0.1, 2, 0.7, 0.3);
            let bits = |v: &[Bin]| {
                v.iter()
                    .map(|x| x.normal.map(f32::to_bits))
                    .collect::<Vec<_>>()
            };
            assert_eq!(bits(&a), bits(&c), "case {case}");
            assert_ne!(bits(&a), bits(&b), "case {case}: nothing was flipped");
        }
    }

    /// Reads a binary little-endian PLY with 27-byte vertices
    /// (`x y z` f32, `r g b` u8, `nx ny nz` f32). Returns `None` if the file is missing or
    /// malformed.
    fn read_ply(path: &std::path::Path) -> Option<Vec<crate::Point>> {
        let data = std::fs::read(path).ok()?;
        let end = data.windows(11).position(|w| w == b"end_header\n")? + 11;
        let head = std::str::from_utf8(&data[..end]).ok()?;
        let n: usize = head
            .lines()
            .find_map(|l| l.strip_prefix("element vertex "))?
            .trim()
            .parse()
            .ok()?;
        let f = |b: &[u8]| f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        Some(
            data[end..end + n * 27]
                .chunks_exact(27)
                .map(|r| crate::Point {
                    pos: [f(&r[0..]), f(&r[4..]), f(&r[8..])],
                    rgb: [r[12], r[13], r[14]],
                    normal: [f(&r[15..]), f(&r[19..]), f(&r[23..])],
                })
                .collect(),
        )
    }

    /// Directory holding real preview point clouds (`r0_preview_new.ply`, `r1_preview_new.ply`),
    /// taken from the `DELTAMESH_DATA` environment variable.
    fn data_dir() -> Option<std::path::PathBuf> {
        std::env::var_os("DELTAMESH_DATA").map(std::path::PathBuf::from)
    }

    /// Matches the reference bitwise on real preview bins. Skipped when `DELTAMESH_DATA` is unset
    /// or a file is missing.
    #[test]
    fn real_preview_matches_reference() {
        let Some(dir) = data_dir() else {
            return;
        };
        for (file, size) in [
            ("r0_preview_new.ply", 0.1f32),
            ("r1_preview_new.ply", 0.025),
        ] {
            let Some(pts) = read_ply(&dir.join(file)) else {
                eprintln!("{file} not found, skipping");
                continue;
            };
            let (b, bs) =
                crate::bins::bin_points(&pts, size, crate::bins::NormalPolicy::Estimate, 2, 6);
            let mut a = b.clone();
            reference::propagate(&mut a, &bs.fixed, size, 2, 0.7, 0.3);
            let mut c = b.clone();
            orient_normals_with(&mut c, &bs.fixed, size, Orient::default(), None);
            let bits = |v: &[Bin]| {
                v.iter()
                    .map(|x| x.normal.map(f32::to_bits))
                    .collect::<Vec<_>>()
            };
            assert!(bits(&a) == bits(&c), "{file} {size}");
        }
    }

    /// [`orient_normals_par`] matches [`orient_normals_with`] bitwise.
    ///
    /// Cases: no answers, all `None`, undecided bins fixed, flips, flips of seeds only, and
    /// answers below [`ORACLE_MIN`]. Real data from `DELTAMESH_DATA` is added when available.
    #[test]
    fn par_oracle_matches_sequential() {
        let mut sets: Vec<(Vec<Bin>, Vec<bool>, f32)> = Vec::new();
        let (b, _) = box_scene(true);
        let f = vec![false; b.len()];
        sets.push((b, f, 0.1));
        if let Some(pts) = data_dir().and_then(|d| read_ply(&d.join("r0_preview_new.ply"))) {
            for size in [0.1f32, 0.025] {
                let (b, bs) =
                    crate::bins::bin_points(&pts, size, crate::bins::NormalPolicy::Estimate, 2, 6);
                sets.push((b, bs.fixed, size));
            }
        }
        let bits = |v: &[Bin]| {
            v.iter()
                .map(|x| x.normal.map(f32::to_bits))
                .collect::<Vec<_>>()
        };
        for (b, fixed, size) in &sets {
            let n = b.len();
            let cases: Vec<Option<Vec<Option<f32>>>> = vec![
                None,
                Some(vec![None; n]),
                Some((0..n).map(|i| (i % 13 == 0).then_some(0.9)).collect()),
                Some((0..n).map(|i| (i % 17 == 0).then_some(-0.5)).collect()),
                Some(
                    (0..n)
                        .map(|i| (b[i].normal[2].abs() >= 0.7 && i % 5 == 0).then_some(-0.95))
                        .collect(),
                ),
                Some((0..n).map(|i| (i % 7 == 0).then_some(0.2)).collect()),
            ];
            for (k, ans) in cases.into_iter().enumerate() {
                let mut x = b.clone();
                orient_normals_with(&mut x, fixed, *size, Orient::default(), ans.clone());
                let mut y = b.clone();
                orient_normals_par(&mut y, fixed, *size, Orient::default(), move |_| ans, None);
                assert!(bits(&x) == bits(&y), "{size} case {k}");
            }
        }
    }

    /// Times orientation on real preview data.
    ///
    /// Run with
    /// `DELTAMESH_DATA=<dir> cargo test --release -p deltamesh orient::tests::bench_real -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_real() {
        let Some(dir) = data_dir() else {
            return;
        };
        for (file, size) in [
            ("r0_preview_new.ply", 0.1f32),
            ("r0_preview_new.ply", 0.025),
        ] {
            let Some(pts) = read_ply(&dir.join(file)) else {
                continue;
            };
            let (b, bs) =
                crate::bins::bin_points(&pts, size, crate::bins::NormalPolicy::Estimate, 2, 6);
            let mut v: Vec<f64> = (0..15)
                .map(|_| {
                    let mut c = b.clone();
                    let t = std::time::Instant::now();
                    orient_normals_with(&mut c, &bs.fixed, size, Orient::default(), None);
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            v.sort_by(f64::total_cmp);
            eprintln!(
                "{file} {size}: {} bins, median {:.2} ms, min {:.2} ms",
                b.len(),
                v[v.len() / 2],
                v[0]
            );
        }
    }

    /// [`Orient::Up`] without an oracle leaves the normals untouched.
    #[test]
    fn up_keeps_input() {
        let (mut b, _) = box_scene(true);
        let before = b.clone();
        orient_normals(&mut b, &vec![false; before.len()], 0.1, Orient::Up, None);
        assert_eq!(b, before);
    }
}
