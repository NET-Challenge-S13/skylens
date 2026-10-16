/*
 * rotation_averaging.rs
 *
 * Copyright (c) 2026 ParkSangWoo
 *
 * Author(s):
 *
 *      ParkSangWoo <dev.parksangwoo@gmail.com>
 *
 *
 * This file is part of cumulus3d.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Third-party notices: parts of the algorithms, default parameters and data
 * formats in this file follow other open-source projects. Their copyright
 * notices and licenses are reproduced in THIRD_PARTY_NOTICES.md.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//! View graph and rotation averaging: maximum spanning tree initialization, then L1 (ADMM), then IRLS (Geman-McClure).

use crate::math::{so3_exp, so3_log, CsrMatrix};
use cumulus3d_core::{Error, ImageId, Mat3, Result, Rigid3};
use nalgebra::{Cholesky, DMatrix, DVector, Dyn};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// View graph edge (an image pair with a relative pose). `image_id1 < image_id2`.
#[derive(Clone, Debug)]
pub struct ViewGraphEdge {
    /// Id of the smaller image.
    pub image_id1: ImageId,
    /// Id of the larger image.
    pub image_id2: ImageId,
    /// Relative pose image2 ← image1 (translation is unit length or 0, unused in later stages).
    pub cam1_to_cam2: Rigid3,
    /// Match count in the correspondence graph (after removing duplicates).
    pub num_matches: usize,
    /// Whether it is a valid edge used by rotation averaging and filtering.
    pub valid: bool,
}

/// View graph. Edges are kept in ascending (id1, id2) order (deterministic traversal).
#[derive(Clone, Debug, Default)]
pub struct ViewGraph {
    /// Edge list (ascending (id1, id2)).
    pub edges: Vec<ViewGraphEdge>,
}

impl ViewGraph {
    /// Sorts the edges and builds the view graph.
    pub fn new(mut edges: Vec<ViewGraphEdge>) -> Self {
        edges.sort_by_key(|e| (e.image_id1, e.image_id2));
        Self { edges }
    }
    /// Number of valid edges.
    pub fn num_valid_edges(&self) -> usize {
        self.edges.iter().filter(|e| e.valid).count()
    }
    /// Images at both ends of the valid edges.
    pub fn valid_nodes(&self) -> BTreeSet<ImageId> {
        let mut s = BTreeSet::new();
        for e in self.edges.iter().filter(|e| e.valid) {
            s.insert(e.image_id1);
            s.insert(e.image_id2);
        }
        s
    }
    /// Largest connected component using valid edges between nodes that satisfy `node_ok`.
    /// Design decision: on a tie (same node count), the component with the smaller minimum image id.
    pub fn largest_connected_component(&self, node_ok: impl Fn(ImageId) -> bool) -> BTreeSet<ImageId> {
        let mut adj: BTreeMap<ImageId, Vec<ImageId>> = BTreeMap::new();
        for e in self.edges.iter().filter(|e| e.valid && node_ok(e.image_id1) && node_ok(e.image_id2)) {
            adj.entry(e.image_id1).or_default().push(e.image_id2);
            adj.entry(e.image_id2).or_default().push(e.image_id1);
        }
        let mut seen: BTreeSet<ImageId> = BTreeSet::new();
        let mut best: BTreeSet<ImageId> = BTreeSet::new();
        for &start in adj.keys() {
            if seen.contains(&start) {
                continue;
            }
            let mut comp = BTreeSet::from([start]);
            seen.insert(start);
            let mut q = VecDeque::from([start]);
            while let Some(u) = q.pop_front() {
                for &v in &adj[&u] {
                    if seen.insert(v) {
                        comp.insert(v);
                        q.push_back(v);
                    }
                }
            }
            if comp.len() > best.len() {
                best = comp;
            }
        }
        best
    }
    /// Invalidates the edge if either endpoint is outside `keep`. Returns: number of invalidated edges.
    pub fn invalidate_outside(&mut self, keep: &BTreeSet<ImageId>) -> usize {
        let mut n = 0;
        for e in self.edges.iter_mut().filter(|e| e.valid) {
            if !keep.contains(&e.image_id1) || !keep.contains(&e.image_id2) {
                e.valid = false;
                n += 1;
            }
        }
        n
    }
    /// Invalidates edges whose angular distance between the estimated rotation (world_to_cam)
    /// and the edge relative rotation exceeds `max_deg`.
    pub fn filter_by_relative_rotation(&mut self, rotations: &BTreeMap<ImageId, Mat3>, max_deg: f64) -> usize {
        let max_rad = max_deg.to_radians();
        let mut n = 0;
        for e in self.edges.iter_mut().filter(|e| e.valid) {
            let (Some(r1), Some(r2)) = (rotations.get(&e.image_id1), rotations.get(&e.image_id2)) else { continue };
            let est = r2 * r1.transpose();
            let ang = cumulus3d_core::Quat::from_rotation_matrix(&est).angular_distance(&e.cam1_to_cam2.rotation.normalized());
            if ang > max_rad {
                e.valid = false;
                n += 1;
            }
        }
        n
    }
}

/// IRLS weight function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IrlsWeight {
    /// w = σ² / (e² + σ²)².
    GemanMcClure,
    /// w = (e²)^(−0.75).
    HalfNorm,
}

/// Rotation averaging options (internal fixed values).
#[derive(Clone, Debug)]
pub struct RotationAveragingOptions {
    /// Whether to set the initial rotations from the maximum spanning tree.
    pub use_mst_init: bool,
    /// Maximum outer iterations of the L1 stage.
    pub l1_max_iterations: usize,
    /// Convergence threshold of the L1 stage (radians).
    pub l1_convergence_rad: f64,
    /// Maximum iterations of the IRLS stage.
    pub irls_max_iterations: usize,
    /// Convergence threshold of the IRLS stage (radians).
    pub irls_convergence_rad: f64,
    /// IRLS weight scale σ (degrees).
    pub irls_sigma_deg: f64,
    /// IRLS weight function.
    pub irls_weight: IrlsWeight,
    /// Diagonal regularization value of the normal matrix.
    pub ridge: f64,
    /// L1 ADMM options.
    pub admm: AdmmOptions,
}

impl Default for RotationAveragingOptions {
    fn default() -> Self {
        Self {
            use_mst_init: true,
            l1_max_iterations: 5,
            l1_convergence_rad: 1e-3,
            irls_max_iterations: 100,
            irls_convergence_rad: 1e-3,
            irls_sigma_deg: 5.0,
            irls_weight: IrlsWeight::GemanMcClure,
            ridge: 1e-9,
            admm: AdmmOptions::default(),
        }
    }
}

/// ADMM least absolute deviations options.
#[derive(Clone, Debug)]
pub struct AdmmOptions {
    /// ADMM penalty parameter ρ.
    pub rho: f64,
    /// Over-relaxation parameter α.
    pub alpha: f64,
    /// Absolute convergence tolerance.
    pub abs_tol: f64,
    /// Relative convergence tolerance.
    pub rel_tol: f64,
    /// Maximum inner iterations of the first outer iteration (doubles afterwards, capped at `max_inner_iterations`).
    pub initial_inner_iterations: usize,
    /// Inner iteration limit.
    pub max_inner_iterations: usize,
}

impl Default for AdmmOptions {
    fn default() -> Self {
        Self { rho: 1.0, alpha: 1.0, abs_tol: 1e-4, rel_tol: 1e-2, initial_inner_iterations: 10, max_inner_iterations: 100 }
    }
}

/// min_x ‖A x − b‖₁ via ADMM. `chol` = factorization of (AᵀA + λI).
pub fn admm_l1(a: &CsrMatrix, b: &DVector<f64>, chol: &Cholesky<f64, Dyn>, max_iter: usize, o: &AdmmOptions) -> DVector<f64> {
    let (m, n) = (a.nrows, a.ncols);
    let mut z = DVector::zeros(m);
    let mut u = DVector::zeros(m);
    let mut x = DVector::zeros(n);
    let kappa = 1.0 / o.rho;
    for _ in 0..max_iter {
        x = chol.solve(&a.tr_mul_vec(&(b + &z - &u)));
        let ax = a.mul_vec(&x);
        let yhat = &ax * o.alpha + (&z + b) * (1.0 - o.alpha);
        let z_prev = z.clone();
        let v = &yhat - b + &u;
        z = v.map(|a| a.signum() * (a.abs() - kappa).max(0.0));
        u += &yhat - &z - b;
        let r = (&ax - &z - b).norm();
        let s = (a.tr_mul_vec(&(&z - &z_prev)) * o.rho).norm();
        let eps_pri = (m as f64).sqrt() * o.abs_tol + o.rel_tol * b.norm().max(ax.norm()).max(z.norm());
        let eps_dual = (n as f64).sqrt() * o.abs_tol + o.rel_tol * (a.tr_mul_vec(&u) * o.rho).norm();
        if r < eps_pri && s < eps_dual {
            break;
        }
    }
    x
}

/// General L1 regression (for verification): runs ADMM with the normal matrix plus ridge λ.
pub fn l1_regression(a: &CsrMatrix, b: &DVector<f64>, ridge: f64, max_iter: usize, o: &AdmmOptions) -> Option<DVector<f64>> {
    let chol = a.weighted_normal(None, ridge).cholesky()?;
    let x = admm_l1(a, b, &chol, max_iter, o);
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Maximum spanning tree initialization. Root = smallest id, rotation = identity.
pub fn mst_initialization(nodes: &BTreeSet<ImageId>, edges: &[&ViewGraphEdge]) -> BTreeMap<ImageId, Mat3> {
    let idx: BTreeMap<ImageId, usize> = nodes.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    let mut order: Vec<&&ViewGraphEdge> = edges.iter().collect();
    // Design decision: tie order is ascending (id1, id2).
    order.sort_by(|a, b| b.num_matches.cmp(&a.num_matches).then((a.image_id1, a.image_id2).cmp(&(b.image_id1, b.image_id2))));
    let mut parent: Vec<usize> = (0..nodes.len()).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let mut adj: BTreeMap<ImageId, Vec<(ImageId, Mat3)>> = BTreeMap::new();
    for e in order {
        let (i, j) = (idx[&e.image_id1], idx[&e.image_id2]);
        let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
        if ri == rj {
            continue;
        }
        parent[ri] = rj;
        let r21 = e.cam1_to_cam2.rotation_matrix();
        // c ← relative rotation to p.
        adj.entry(e.image_id1).or_default().push((e.image_id2, r21));
        adj.entry(e.image_id2).or_default().push((e.image_id1, r21.transpose()));
    }
    let mut rot = BTreeMap::new();
    let Some(&root) = nodes.iter().next() else { return rot };
    rot.insert(root, Mat3::identity());
    let mut q = VecDeque::from([root]);
    while let Some(p) = q.pop_front() {
        let rp = rot[&p];
        if let Some(list) = adj.get(&p) {
            let mut list = list.clone();
            list.sort_by_key(|(c, _)| *c);
            for (c, rcp) in list {
                if let std::collections::btree_map::Entry::Vacant(v) = rot.entry(c) {
                    v.insert(rcp * rp);
                    q.push_back(c);
                }
            }
        }
    }
    // Nodes not reached by the tree (possible if the graph is not a single connected component) get identity.
    for n in nodes {
        rot.entry(*n).or_insert_with(Mat3::identity);
    }
    rot
}

/// Rotation averaging result statistics.
#[derive(Clone, Debug, Default)]
pub struct RotationAveragingSummary {
    /// Number of L1 stage iterations.
    pub num_l1_iterations: usize,
    /// Number of IRLS stage iterations.
    pub num_irls_iterations: usize,
}

/// Solves the global rotations (world_to_cam) from the active nodes and the valid edges between them.
/// If `initial` is given and `use_mst_init` is false, starts from that value.
pub fn solve_rotation_averaging(
    nodes: &BTreeSet<ImageId>,
    graph: &ViewGraph,
    initial: Option<&BTreeMap<ImageId, Mat3>>,
    opts: &RotationAveragingOptions,
) -> Result<(BTreeMap<ImageId, Mat3>, RotationAveragingSummary)> {
    if nodes.is_empty() {
        return Err(Error::InvalidArgument("회전 평균: 활성 영상 없음".into()));
    }
    let edges: Vec<&ViewGraphEdge> =
        graph.edges.iter().filter(|e| e.valid && nodes.contains(&e.image_id1) && nodes.contains(&e.image_id2)).collect();
    let mut rot = match (initial, opts.use_mst_init) {
        (Some(init), false) => nodes.iter().map(|n| (*n, init.get(n).copied().unwrap_or_else(Mat3::identity))).collect(),
        _ => mst_initialization(nodes, &edges),
    };
    let ids: Vec<ImageId> = nodes.iter().copied().collect();
    let col: BTreeMap<ImageId, usize> = ids.iter().enumerate().map(|(i, id)| (*id, 3 * i)).collect();
    let n = 3 * ids.len();
    // Coefficient matrix A (fixed).
    let mut a = CsrMatrix::new(n);
    for e in &edges {
        let (c1, c2) = (col[&e.image_id1], col[&e.image_id2]);
        for k in 0..3 {
            a.push_row(&[(c1 + k, -1.0), (c2 + k, 1.0)]);
        }
    }
    let fixed = ids[0];
    let theta_f0 = rot[&fixed];
    for k in 0..3 {
        a.push_row(&[(col[&fixed] + k, 1.0)]);
    }
    let rel: Vec<Mat3> = edges.iter().map(|e| e.cam1_to_cam2.rotation_matrix()).collect();
    let compute_b = |rot: &BTreeMap<ImageId, Mat3>| -> DVector<f64> {
        let mut b = DVector::zeros(a.nrows);
        for (k, e) in edges.iter().enumerate() {
            let (r1, r2) = (rot[&e.image_id1], rot[&e.image_id2]);
            let v = -so3_log(&(r2.transpose() * rel[k] * r1));
            for d in 0..3 {
                b[3 * k + d] = v[d];
            }
        }
        let g = so3_log(&(theta_f0.transpose() * rot[&fixed]));
        let base = 3 * edges.len();
        for d in 0..3 {
            b[base + d] = g[d];
        }
        b
    };
    let apply = |rot: &mut BTreeMap<ImageId, Mat3>, delta: &DVector<f64>| -> f64 {
        let mut s = 0.0;
        for (id, &c) in &col {
            let d = nalgebra::Vector3::new(delta[c], delta[c + 1], delta[c + 2]);
            s += d.norm();
            let r = rot.get_mut(id).expect("노드");
            *r *= so3_exp(&(-d));
        }
        s / ids.len() as f64
    };
    let mut summary = RotationAveragingSummary::default();

    // L1 stage.
    if opts.l1_max_iterations > 0 && !edges.is_empty() {
        let chol = a
            .weighted_normal(None, opts.ridge)
            .cholesky()
            .ok_or_else(|| Error::InvalidArgument("회전 평균: L1 정규 행렬 분해 실패".into()))?;
        let mut inner = opts.admm.initial_inner_iterations;
        let mut prev_norm = f64::INFINITY;
        for _ in 0..opts.l1_max_iterations {
            summary.num_l1_iterations += 1;
            let b = compute_b(&rot);
            let delta = admm_l1(&a, &b, &chol, inner, &opts.admm);
            if delta.iter().any(|v| v.is_nan()) {
                return Err(Error::InvalidArgument("회전 평균: L1 해에 NaN".into()));
            }
            let mean = apply(&mut rot, &delta);
            let dn = delta.norm();
            if mean < opts.l1_convergence_rad || (prev_norm - dn).abs() < 1e-12 {
                break;
            }
            prev_norm = dn;
            inner = (inner * 2).min(opts.admm.max_inner_iterations);
        }
    }

    // IRLS stage.
    let sigma = opts.irls_sigma_deg.to_radians();
    let mut w = vec![1.0; a.nrows];
    for _ in 0..opts.irls_max_iterations {
        if edges.is_empty() {
            break;
        }
        summary.num_irls_iterations += 1;
        let b = compute_b(&rot);
        for k in 0..edges.len() {
            let e2 = b[3 * k].powi(2) + b[3 * k + 1].powi(2) + b[3 * k + 2].powi(2);
            let wk = match opts.irls_weight {
                IrlsWeight::GemanMcClure => sigma * sigma / (e2 + sigma * sigma).powi(2),
                IrlsWeight::HalfNorm => e2.powf(-0.75),
            };
            if wk.is_nan() {
                return Err(Error::InvalidArgument("회전 평균: IRLS 가중 NaN".into()));
            }
            for d in 0..3 {
                w[3 * k + d] = wk;
            }
        }
        let nmat: DMatrix<f64> = a.weighted_normal(Some(&w), opts.ridge);
        let wb = DVector::from_iterator(b.len(), b.iter().zip(&w).map(|(x, y)| x * y));
        let rhs = a.tr_mul_vec(&wb);
        let chol = nmat.cholesky().ok_or_else(|| Error::InvalidArgument("회전 평균: IRLS 분해 실패".into()))?;
        let delta = chol.solve(&rhs);
        if delta.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidArgument("회전 평균: IRLS 해에 NaN".into()));
        }
        let mean = apply(&mut rot, &delta);
        if mean < opts.irls_convergence_rad {
            break;
        }
    }
    // Clean up accumulated numerical error.
    for r in rot.values_mut() {
        *r = crate::math::nearest_rotation(r);
    }
    Ok((rot, summary))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::gaussian;
    use cumulus3d_core::{Quat, Vec3};
    use rand::{RngExt, SeedableRng};

    fn random_rot(rng: &mut rand_pcg::Pcg64, scale: f64) -> Mat3 {
        so3_exp(&(Vec3::new(gaussian(rng), gaussian(rng), gaussian(rng)) * scale))
    }

    /// List of angular errors (degrees) after aligning with a single global rotation.
    fn aligned_errors(est: &BTreeMap<ImageId, Mat3>, truth: &BTreeMap<ImageId, Mat3>) -> Vec<f64> {
        // R_est ≈ R_true · G, then estimate G: instead of using the first image as reference, use the mean
        // (quaternion mean approximation: nearest rotation to the matrix sum).
        let mut sum = Mat3::zeros();
        for (id, r) in est {
            sum += truth[id].transpose() * r;
        }
        let g = crate::math::nearest_rotation(&sum);
        let mut errs: Vec<f64> = est.iter().map(|(id, r)| crate::math::rotation_angle_between(r, &(truth[id] * g)).to_degrees()).collect();
        errs.sort_by(|a, b| a.total_cmp(b));
        errs
    }

    fn make_graph(seed: u64, n: usize, noise_deg: f64, outlier_frac: f64) -> (ViewGraph, BTreeMap<ImageId, Mat3>, Vec<bool>) {
        let mut rng = rand_pcg::Pcg64::seed_from_u64(seed);
        let truth: BTreeMap<ImageId, Mat3> = (1..=n as u32).map(|i| (i, random_rot(&mut rng, 1.0))).collect();
        let mut edges = Vec::new();
        let mut bad = Vec::new();
        for i in 1..=n as u32 {
            for j in i + 1..=n as u32 {
                if j - i > 6 && rng.random_range(0.0..1.0) > 0.1 {
                    continue;
                }
                let mut r = truth[&j] * truth[&i].transpose();
                let is_bad = rng.random_range(0.0..1.0) < outlier_frac;
                if is_bad {
                    let ax = Vec3::new(gaussian(&mut rng), gaussian(&mut rng), gaussian(&mut rng)).normalize();
                    r = so3_exp(&(ax * rng.random_range(30f64..90.0).to_radians())) * r;
                } else if noise_deg > 0.0 {
                    let ax = Vec3::new(gaussian(&mut rng), gaussian(&mut rng), gaussian(&mut rng));
                    r = so3_exp(&(ax * noise_deg.to_radians() / 3f64.sqrt())) * r;
                }
                bad.push(is_bad);
                edges.push(ViewGraphEdge {
                    image_id1: i,
                    image_id2: j,
                    cam1_to_cam2: Rigid3::new(Quat::from_rotation_matrix(&r), Vec3::zeros()),
                    num_matches: rng.random_range(15..500),
                    valid: true,
                });
            }
        }
        (ViewGraph::new(edges), truth, bad)
    }

    #[test]
    fn noise_free_exact() {
        let (g, truth, _) = make_graph(1, 40, 0.0, 0.0);
        let nodes = g.valid_nodes();
        let (est, _) = solve_rotation_averaging(&nodes, &g, None, &RotationAveragingOptions::default()).unwrap();
        let errs = aligned_errors(&est, &truth);
        assert!(errs.last().unwrap().to_radians() < 1e-6, "{:?}", errs.last());
    }

    #[test]
    fn noisy_with_outliers() {
        let (mut g, truth, bad) = make_graph(2, 60, 0.5, 0.1);
        let nodes = g.valid_nodes();
        let (est, _) = solve_rotation_averaging(&nodes, &g, None, &RotationAveragingOptions::default()).unwrap();
        let errs = aligned_errors(&est, &truth);
        let med = errs[errs.len() / 2];
        let max = *errs.last().unwrap();
        assert!(med < 0.3 && max < 1.0, "median {med} max {max}");
        g.filter_by_relative_rotation(&est, 10.0);
        for (e, b) in g.edges.iter().zip(&bad) {
            if *b {
                assert!(!e.valid, "오차 주입 간선 {}-{} 이 남음", e.image_id1, e.image_id2);
            }
        }
    }

    #[test]
    fn admm_matches_lp_solution() {
        // 30×6 overdetermined, 20% outliers. LP solution = exhaustive vertex search
        // (the L1 optimum is a vertex where n residuals are zero).
        let mut rng = rand_pcg::Pcg64::seed_from_u64(3);
        let (m, n) = (30usize, 6usize);
        let a = DMatrix::from_fn(m, n, |_, _| gaussian(&mut rng));
        let xt = DVector::from_fn(n, |_, _| gaussian(&mut rng));
        let mut b = &a * &xt;
        for i in 0..m {
            b[i] += 0.01 * gaussian(&mut rng);
            if i % 5 == 0 {
                b[i] += 5.0 * gaussian(&mut rng);
            }
        }
        let cost = |x: &DVector<f64>| (&a * x - &b).abs().sum();
        let mut best = (f64::INFINITY, DVector::zeros(n));
        fn combos(start: usize, m: usize, k: usize, cur: &mut Vec<usize>, f: &mut dyn FnMut(&[usize])) {
            if cur.len() == k {
                f(cur);
                return;
            }
            for i in start..m {
                cur.push(i);
                combos(i + 1, m, k, cur, f);
                cur.pop();
            }
        }
        combos(0, m, n, &mut Vec::new(), &mut |idx: &[usize]| {
            let sub = DMatrix::from_fn(n, n, |r, c| a[(idx[r], c)]);
            let rhs = DVector::from_fn(n, |r, _| b[idx[r]]);
            if let Some(x) = sub.lu().solve(&rhs) {
                let c = cost(&x);
                if c < best.0 {
                    best = (c, x);
                }
            }
        });
        let csr = CsrMatrix::from_dense(&a);
        let o = AdmmOptions { abs_tol: 1e-10, rel_tol: 1e-10, ..Default::default() };
        let x = l1_regression(&csr, &b, 1e-9, 20000, &o).unwrap();
        assert!((&x - &best.1).amax() < 1e-3, "admm {x} lp {}", best.1);
    }

    #[test]
    fn largest_component_and_filter() {
        let e = |a, b| ViewGraphEdge { image_id1: a, image_id2: b, cam1_to_cam2: Rigid3::identity(), num_matches: 20, valid: true };
        let mut g = ViewGraph::new(vec![e(1, 2), e(2, 3), e(4, 5), e(6, 7), e(7, 8), e(8, 9)]);
        let cc = g.largest_connected_component(|_| true);
        assert_eq!(cc, BTreeSet::from([6, 7, 8, 9]));
        assert_eq!(g.invalidate_outside(&cc), 3);
        assert_eq!(g.num_valid_edges(), 3);
    }
}
