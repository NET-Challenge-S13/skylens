//! Groups input points into small cubic cells ("bins") and computes per-bin normals.
//!
//! Binning makes the cost of later stages proportional to the number of occupied cells instead
//! of the raw point count. Each bin stores the mean position, mean colour and point count of the
//! points that fell into it.
//!
//! Normals come from the input when the [`NormalPolicy`] allows it and the averaged input normal
//! is reliable. Otherwise they are estimated by PCA over the neighbouring bins and oriented
//! towards `+z`; [`crate::orient`] later fixes the sign.
//!
//! Output is sorted by cell coordinate and is bitwise independent of the rayon thread count.
//! Points are always summed into a cell in their original input order, so the floating-point
//! sums match a single sequential hash map exactly.

use crate::{Point, eig::smallest_eigvec};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// One occupied cell after binning.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bin {
    /// Mean position of the points in the cell, in metres.
    pub pos: [f32; 3],
    /// Unit normal. Meaningless (all zero) when the policy is [`NormalPolicy::Ignore`].
    pub normal: [f32; 3],
    /// Mean colour, `0.0..=255.0` per channel.
    pub rgb: [f32; 3],
    /// Number of input points in the cell.
    pub count: u32,
}

/// How input normals are used when binning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormalPolicy {
    /// Normals are not needed (height field).
    Ignore,
    /// Use input normals and estimate only bins whose input normals are unusable (refined data).
    Trust,
    /// Discard input normals and estimate every bin (preview data).
    Estimate,
}

/// Per-cell running sums. Positions are summed in `f64` to keep precision at large coordinates.
#[derive(Default)]
struct Acc {
    pos: [f64; 3],
    nrm: [f32; 3],
    n_valid: u32,
    rgb: [u32; 3],
    count: u32,
}

/// Statistics and per-bin flags produced by [`bin_points`].
#[derive(Clone, Debug, Default)]
pub struct BinStats {
    /// Number of bins whose normal was estimated by PCA.
    pub estimated: usize,
    /// Number of bins removed because they had too few neighbours to estimate a normal.
    pub dropped: usize,
    /// Per output bin, whether its normal was taken from the input.
    ///
    /// `false` means the normal was estimated and only oriented towards `+z`; its sign is
    /// decided later by [`crate::orient`]. Empty when no bin needed estimation.
    pub fixed: Vec<bool>,
}

/// Accepts finite normals whose length lies in `0.5..=1.5`.
fn valid_normal(n: [f32; 3]) -> bool {
    let l2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    l2.is_finite() && (0.25..=2.25).contains(&l2)
}

/// Number of input points per parallel chunk. Fixed, not derived from the thread count, so the
/// result does not depend on it.
const CHUNK: usize = 1 << 16;
/// Maximum number of partitions of the x cell range. Each partition is binned in parallel with
/// its own small hash map.
const PARTS: i64 = 256;
/// Stride used when sampling points to estimate the x cell range.
const SAMPLE: usize = 16;

#[inline]
fn finite(p: &Point) -> bool {
    p.pos[0].is_finite() && p.pos[1].is_finite() && p.pos[2].is_finite()
}

#[inline]
fn cell_key(p: &Point, inv: f32) -> [i32; 3] {
    [
        (p.pos[0] * inv).floor() as i32,
        (p.pos[1] * inv).floor() as i32,
        (p.pos[2] * inv).floor() as i32,
    ]
}

/// Adds one point to a cell accumulator. Input normals are summed only under
/// [`NormalPolicy::Trust`] and only when they pass [`valid_normal`].
#[inline]
fn add(a: &mut Acc, p: &Point, policy: NormalPolicy) {
    for i in 0..3 {
        a.pos[i] += p.pos[i] as f64;
        a.rgb[i] += p.rgb[i] as u32;
    }
    if policy == NormalPolicy::Trust && valid_normal(p.normal) {
        for i in 0..3 {
            a.nrm[i] += p.normal[i];
        }
        a.n_valid += 1;
    }
    a.count += 1;
}

/// Bins points with one hash map in input order.
///
/// Used on a single thread, where partitioning costs more than it saves. The result is bitwise
/// identical to [`accumulate`], which is checked by a test.
fn accumulate_serial(pts: &[Point], size: f32, policy: NormalPolicy) -> Vec<([i32; 3], Acc)> {
    let inv = 1.0 / size;
    let mut map: FxHashMap<[i32; 3], Acc> = FxHashMap::default();
    map.reserve(pts.len() / 2);
    for p in pts.iter().filter(|p| finite(p)) {
        add(map.entry(cell_key(p, inv)).or_default(), p, policy);
    }
    let mut v: Vec<([i32; 3], Acc)> = map.into_iter().collect();
    v.sort_unstable_by_key(|c| c.0);
    v
}

/// Returns per-cell sums sorted by cell coordinate, computed in parallel.
///
/// The x cell range is split into up to [`PARTS`] partitions with a monotonic mapping, so
/// concatenating the sorted partitions in order yields a globally sorted list. The x range is
/// estimated from a sample of points only; cells outside it are clamped to the first or last
/// partition, which keeps the mapping monotonic.
///
/// Each chunk of input records, per partition, the indices of its points in input order. Each
/// partition then walks the chunks in order, so the points of any cell are summed in their
/// original input order and the floating-point sums are bitwise identical to
/// [`accumulate_serial`].
///
/// # Panics
///
/// Panics if there are more than `u32::MAX` points.
fn accumulate(pts: &[Point], size: f32, policy: NormalPolicy) -> Vec<([i32; 3], Acc)> {
    assert!(
        pts.len() <= u32::MAX as usize,
        "point index must fit in u32"
    );
    let inv = 1.0 / size;
    let (xmin, xmax) = pts
        .par_chunks(CHUNK)
        .map(|ch| {
            let mut r = (i32::MAX, i32::MIN);
            for p in ch.iter().step_by(SAMPLE).filter(|p| finite(p)) {
                let kx = (p.pos[0] * inv).floor() as i32;
                r = (r.0.min(kx), r.1.max(kx));
            }
            r
        })
        .reduce(|| (i32::MAX, i32::MIN), |a, b| (a.0.min(b.0), a.1.max(b.1)));
    let (xmin, xmax) = if xmin > xmax { (0, 0) } else { (xmin, xmax) };
    let span = xmax as i64 - xmin as i64 + 1;
    let parts = span.min(PARTS);
    let part_of = |kx: i32| ((kx as i64 - xmin as i64).clamp(0, span - 1) * parts / span) as usize;

    let lists: Vec<Vec<Vec<u32>>> = pts
        .par_chunks(CHUNK)
        .enumerate()
        .map(|(ci, ch)| {
            let mut l: Vec<Vec<u32>> = vec![Vec::new(); parts as usize];
            let base = ci * CHUNK;
            for (j, p) in ch.iter().enumerate() {
                if finite(p) {
                    l[part_of((p.pos[0] * inv).floor() as i32)].push((base + j) as u32);
                }
            }
            l
        })
        .collect();

    let per_part: Vec<Vec<([i32; 3], Acc)>> = (0..parts as usize)
        .into_par_iter()
        .map(|b| {
            let mut map: FxHashMap<[i32; 3], Acc> = FxHashMap::default();
            for l in &lists {
                for &i in &l[b] {
                    let p = &pts[i as usize];
                    add(map.entry(cell_key(p, inv)).or_default(), p, policy);
                }
            }
            let mut v: Vec<([i32; 3], Acc)> = map.into_iter().collect();
            v.sort_unstable_by_key(|c| c.0);
            v
        })
        .collect();
    drop(lists);
    let mut out = Vec::with_capacity(per_part.iter().map(Vec::len).sum());
    for v in per_part {
        out.extend(v);
    }
    out
}

/// Bins points into cubic cells of `size` metres and computes per-bin normals.
///
/// Non-finite points are skipped. A bin keeps its averaged input normal only if the policy is
/// [`NormalPolicy::Trust`] and the summed valid normals have length above `0.3` times their
/// count; cells whose input normals disagree average to a short vector and are estimated
/// instead.
///
/// Estimation fits a plane by PCA to the positions of all bins within `normal_radius` cells
/// (Chebyshev distance, so a `(2r+1)³` window) of the bin. Coordinates are taken relative to
/// the centre bin and summed in `f64` to avoid losing precision at large coordinates. The
/// estimated normal points towards `+z`.
///
/// Neighbours are found without hashing. Bins are sorted by `(x, y, z)` cell key, so bins with
/// the same x form a contiguous slab and, within a slab, bins with the same `(x, y)` form a
/// contiguous row sorted by z. Monotonic cursors over slabs, rows and z find the neighbour
/// window; the neighbouring rows form `(2r+1)²` contiguous ranges. Each slab writes its own
/// disjoint output range in parallel, and the work within a slab is sequential, so the result
/// does not depend on the thread count.
///
/// # Arguments
///
/// * `pts` - Input points.
/// * `size` - Bin edge length in metres.
/// * `policy` - How input normals are used.
/// * `normal_radius` - Neighbour radius for PCA, in bins.
/// * `normal_min_neighbors` - Minimum number of bins in the window (including the centre) for
///   an estimate; bins with fewer are dropped.
///
/// # Returns
///
/// The bins sorted by cell coordinate and the [`BinStats`]. For the same input order the output
/// is always bitwise identical.
pub fn bin_points(
    pts: &[Point],
    size: f32,
    policy: NormalPolicy,
    normal_radius: i32,
    normal_min_neighbors: usize,
) -> (Vec<Bin>, BinStats) {
    let cells = if rayon::current_num_threads() > 1 {
        accumulate(pts, size, policy)
    } else {
        accumulate_serial(pts, size, policy)
    };
    let keys: Vec<[i32; 3]> = cells.par_iter().map(|c| c.0).collect();
    let (mut bins, need): (Vec<Bin>, Vec<bool>) = cells
        .par_iter()
        .map(|(_, a)| {
            let c = a.count as f64;
            let pos = [
                (a.pos[0] / c) as f32,
                (a.pos[1] / c) as f32,
                (a.pos[2] / c) as f32,
            ];
            let rgb = [
                a.rgb[0] as f32 / a.count as f32,
                a.rgb[1] as f32 / a.count as f32,
                a.rgb[2] as f32 / a.count as f32,
            ];
            let mut normal = [0.0; 3];
            let mut ok = false;
            if a.n_valid > 0 {
                let l = (a.nrm[0] * a.nrm[0] + a.nrm[1] * a.nrm[1] + a.nrm[2] * a.nrm[2]).sqrt();
                if l > 0.3 * a.n_valid as f32 {
                    normal = [a.nrm[0] / l, a.nrm[1] / l, a.nrm[2] / l];
                    ok = true;
                }
            }
            (
                Bin {
                    pos,
                    normal,
                    rgb,
                    count: a.count,
                },
                policy != NormalPolicy::Ignore && !ok,
            )
        })
        .unzip();
    drop(cells);

    let mut stats = BinStats::default();
    if need.iter().any(|&b| b) {
        let n_bins = keys.len();
        let mut rows: Vec<u32> = Vec::new();
        let mut slabs: Vec<u32> = Vec::new();
        for i in 0..n_bins {
            if i == 0 || keys[i][0] != keys[i - 1][0] {
                slabs.push(rows.len() as u32);
            }
            if i == 0 || keys[i][0] != keys[i - 1][0] || keys[i][1] != keys[i - 1][1] {
                rows.push(i as u32);
            }
        }
        rows.push(n_bins as u32);
        slabs.push((rows.len() - 1) as u32);
        let row_y: Vec<i32> = rows[..rows.len() - 1]
            .iter()
            .map(|&s| keys[s as usize][1])
            .collect();
        let slab_x: Vec<i32> = slabs[..slabs.len() - 1]
            .iter()
            .map(|&ri| keys[rows[ri as usize] as usize][0])
            .collect();
        let zs: Vec<i32> = keys.iter().map(|k| k[2]).collect();
        let pos: Vec<[f32; 3]> = bins.iter().map(|b| b.pos).collect();
        let r = normal_radius;

        let mut est: Vec<Option<[f32; 3]>> = vec![None; n_bins];
        let mut parts: Vec<(usize, &mut [Option<[f32; 3]>])> = Vec::with_capacity(slab_x.len());
        let mut rest = &mut est[..];
        for si in 0..slab_x.len() {
            let len = (rows[slabs[si + 1] as usize] - rows[slabs[si] as usize]) as usize;
            let (a, b) = rest.split_at_mut(len);
            parts.push((si, a));
            rest = b;
        }
        parts.into_par_iter().for_each(|(si, out)| {
            let base = rows[slabs[si] as usize] as usize;
            if !need[base..base + out.len()].iter().any(|&b| b) {
                return;
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
                if !need[s0..e0].iter().any(|&b| b) {
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
                for i in s0..e0 {
                    if !need[i] {
                        continue;
                    }
                    let kz = zs[i];
                    let o = pos[i];
                    let mut n = 0usize;
                    let mut s = [0.0f64; 3];
                    let mut ss = [0.0f64; 6];
                    for j in 0..lo.len() {
                        while lo[j] < hi[j] && zs[lo[j]] < kz - r {
                            lo[j] += 1;
                        }
                        let mut c = lo[j];
                        while c < hi[j] && zs[c] <= kz + r {
                            let p = pos[c];
                            let d = [
                                (p[0] - o[0]) as f64,
                                (p[1] - o[1]) as f64,
                                (p[2] - o[2]) as f64,
                            ];
                            s[0] += d[0];
                            s[1] += d[1];
                            s[2] += d[2];
                            ss[0] += d[0] * d[0];
                            ss[1] += d[0] * d[1];
                            ss[2] += d[0] * d[2];
                            ss[3] += d[1] * d[1];
                            ss[4] += d[1] * d[2];
                            ss[5] += d[2] * d[2];
                            n += 1;
                            c += 1;
                        }
                    }
                    out[i - base] = Some(if n < normal_min_neighbors {
                        [f32::NAN; 3]
                    } else {
                        let nf = n as f64;
                        let m = [s[0] / nf, s[1] / nf, s[2] / nf];
                        let cov = [
                            [
                                ss[0] / nf - m[0] * m[0],
                                ss[1] / nf - m[0] * m[1],
                                ss[2] / nf - m[0] * m[2],
                            ],
                            [
                                ss[1] / nf - m[0] * m[1],
                                ss[3] / nf - m[1] * m[1],
                                ss[4] / nf - m[1] * m[2],
                            ],
                            [
                                ss[2] / nf - m[0] * m[2],
                                ss[4] / nf - m[1] * m[2],
                                ss[5] / nf - m[2] * m[2],
                            ],
                        ];
                        let v = smallest_eigvec(cov);
                        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                        let sgn = if v[2] < 0.0 { -1.0 } else { 1.0 };
                        [
                            (sgn * v[0] / l) as f32,
                            (sgn * v[1] / l) as f32,
                            (sgn * v[2] / l) as f32,
                        ]
                    });
                }
            }
        });
        let mut out = Vec::with_capacity(bins.len());
        for (b, e) in bins.into_iter().zip(est) {
            match e {
                None => {
                    out.push(b);
                    stats.fixed.push(true);
                }
                Some(n) if n[0].is_finite() => {
                    stats.estimated += 1;
                    out.push(Bin { normal: n, ..b });
                    stats.fixed.push(false);
                }
                Some(_) => stats.dropped += 1,
            }
        }
        bins = out;
    }
    (bins, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(n: usize, step: f32) -> Vec<Point> {
        let mut v = Vec::new();
        for i in 0..n {
            for j in 0..n {
                v.push(Point {
                    pos: [i as f32 * step, j as f32 * step, 1.0],
                    rgb: [10, 20, 30],
                    normal: [f32::NAN, 1e15, 0.0],
                });
            }
        }
        v
    }

    #[test]
    fn estimates_up_normal_on_plane() {
        let (b, s) = bin_points(&plane(40, 0.05), 0.1, NormalPolicy::Estimate, 2, 6);
        assert!(!b.is_empty());
        assert_eq!(s.estimated + s.dropped, b.len() + s.dropped);
        for x in &b {
            assert!(x.normal[2] > 0.99, "{:?}", x.normal);
        }
    }

    #[test]
    fn trust_rejects_garbage_normals() {
        let (b, s) = bin_points(&plane(40, 0.05), 0.1, NormalPolicy::Trust, 2, 6);
        assert_eq!(s.estimated, b.len());
    }
    fn acc_bits(a: &Acc) -> ([u64; 3], [u32; 3], u32, [u32; 3], u32) {
        (
            a.pos.map(f64::to_bits),
            a.nrm.map(f32::to_bits),
            a.n_valid,
            a.rgb,
            a.count,
        )
    }

    /// Random points spanning several chunks, with negative coordinates, many points sharing a
    /// cell (every third point), and some NaN and infinite coordinates.
    fn random_points(n: usize, seed: u64) -> Vec<Point> {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut f = move || (next() >> 11) as f64 / (1u64 << 53) as f64;
        (0..n)
            .map(|i| {
                let mut pos = [
                    (f() * 60.0 - 30.0) as f32,
                    (f() * 40.0 - 20.0) as f32,
                    (f() * 8.0 - 2.0) as f32,
                ];
                if i % 3 == 0 {
                    pos = [
                        (f() * 0.5) as f32 + 1.0,
                        (f() * 0.5) as f32,
                        (f() * 0.3) as f32,
                    ];
                }
                if i % 997 == 0 {
                    pos[i % 3] = if i % 2 == 0 { f32::NAN } else { f32::INFINITY };
                }
                let rgb = [
                    (f() * 255.0) as u8,
                    (f() * 255.0) as u8,
                    (f() * 255.0) as u8,
                ];
                let normal = if i % 5 == 0 {
                    [f32::NAN; 3]
                } else {
                    [(f() - 0.5) as f32, (f() - 0.5) as f32, f() as f32 + 0.3]
                };
                Point { pos, rgb, normal }
            })
            .collect()
    }

    #[test]
    fn accumulate_matches_single_map_bitwise() {
        let pts = random_points(300_000, 0x9e37_79b9_7f4a_7c15);
        for &(size, policy) in &[
            (0.1, NormalPolicy::Trust),
            (0.37, NormalPolicy::Estimate),
            (5.0, NormalPolicy::Ignore),
            (1000.0, NormalPolicy::Trust),
        ] {
            let want = accumulate_serial(&pts, size, policy);
            let got = accumulate(&pts, size, policy);
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.0, w.0);
                assert_eq!(acc_bits(&g.1), acc_bits(&w.1));
            }
        }
        assert!(accumulate(&[], 0.1, NormalPolicy::Trust).is_empty());
        let bad = [Point {
            pos: [f32::NAN, 0.0, 0.0],
            ..Default::default()
        }];
        assert!(accumulate(&bad, 0.1, NormalPolicy::Trust).is_empty());
    }

    #[test]
    fn bins_independent_of_threads() {
        let pts = random_points(200_000, 7);
        let run = |t: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap();
            pool.install(|| bin_points(&pts, 0.2, NormalPolicy::Trust, 2, 6).0)
        };
        let bits = |v: Vec<Bin>| -> Vec<[u32; 10]> {
            v.iter()
                .map(|b| {
                    let mut o = [0u32; 10];
                    for i in 0..3 {
                        o[i] = b.pos[i].to_bits();
                        o[3 + i] = b.normal[i].to_bits();
                        o[6 + i] = b.rgb[i].to_bits();
                    }
                    o[9] = b.count;
                    o
                })
                .collect()
        };
        assert_eq!(bits(run(1)), bits(run(4)));
    }
}
