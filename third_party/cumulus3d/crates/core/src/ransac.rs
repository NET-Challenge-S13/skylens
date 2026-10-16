/*
 * ransac.rs
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

//! General-purpose RANSAC / LO-RANSAC.
//!
//! Residuals are "squared errors" and an inlier test is residual <= max_error². Support = (inlier count, sum of inlier residuals).

use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

/// Estimator: minimal/non-minimal solver + residual.
pub trait Estimator {
    /// Input data 1 (e.g. image 1 coordinates).
    type X: Clone;
    /// Input data 2 (e.g. image 2 coordinates).
    type Y: Clone;
    /// Estimated model type.
    type Model: Clone;
    /// Minimum number of samples required by the solver.
    fn min_num_samples(&self) -> usize;
    /// Appends zero or more models from the sample to `models` (empty before the call).
    fn estimate(&self, x: &[Self::X], y: &[Self::Y], models: &mut Vec<Self::Model>);
    /// Fills `residuals` with the squared residuals of all data (length = x.len()).
    fn residuals(&self, x: &[Self::X], y: &[Self::Y], model: &Self::Model, residuals: &mut Vec<f64>);
}

/// Sampler.
pub trait Sampler {
    /// Initializes with n data points.
    fn initialize(&mut self, n: usize);
    /// Maximum number of distinct samples (reflected in the iteration cap).
    fn max_num_samples(&self) -> usize;
    /// Fills `out` with the next sample indices.
    fn sample(&mut self, rng: &mut Pcg64, out: &mut Vec<usize>);
}

/// Random sampler: partial Fisher-Yates, continuing the shuffle from the previous iteration's state each time.
#[derive(Clone, Debug)]
pub struct RandomSubsetSampler {
    k: usize,
    perm: Vec<usize>,
}

impl RandomSubsetSampler {
    /// Creates with sample size k.
    pub fn new(num_samples: usize) -> Self {
        Self { k: num_samples, perm: Vec::new() }
    }
}

impl Sampler for RandomSubsetSampler {
    fn initialize(&mut self, n: usize) {
        self.perm = (0..n).collect();
    }
    fn max_num_samples(&self) -> usize {
        usize::MAX
    }
    fn sample(&mut self, rng: &mut Pcg64, out: &mut Vec<usize>) {
        out.clear();
        let n = self.perm.len();
        for i in 0..self.k.min(n) {
            let j = rng.random_range(i..n);
            self.perm.swap(i, j);
            out.push(self.perm[i]);
        }
    }
}

/// Combination sampler: all k-combinations in lexicographic order (restarting from the beginning when exhausted).
#[derive(Clone, Debug)]
pub struct ExhaustiveSampler {
    k: usize,
    n: usize,
    cur: Vec<usize>,
    first: bool,
}

impl ExhaustiveSampler {
    /// Creates with sample size k.
    pub fn new(num_samples: usize) -> Self {
        Self { k: num_samples, n: 0, cur: Vec::new(), first: true }
    }
}

/// Binomial coefficient C(n, k) (saturating).
pub fn n_choose_k(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut r: u128 = 1;
    for i in 0..k {
        r = r * (n - i) as u128 / (i + 1) as u128;
        if r > usize::MAX as u128 {
            return usize::MAX;
        }
    }
    r as usize
}

impl Sampler for ExhaustiveSampler {
    fn initialize(&mut self, n: usize) {
        self.n = n;
        self.cur = (0..self.k).collect();
        self.first = true;
    }
    fn max_num_samples(&self) -> usize {
        n_choose_k(self.n, self.k)
    }
    fn sample(&mut self, _rng: &mut Pcg64, out: &mut Vec<usize>) {
        if !self.first {
            // next combination
            let (n, k) = (self.n, self.k);
            let mut i = k;
            let mut advanced = false;
            while i > 0 {
                i -= 1;
                if self.cur[i] < n - k + i {
                    self.cur[i] += 1;
                    for j in i + 1..k {
                        self.cur[j] = self.cur[j - 1] + 1;
                    }
                    advanced = true;
                    break;
                }
            }
            if !advanced {
                self.cur = (0..k).collect();
            }
        }
        self.first = false;
        out.clear();
        out.extend_from_slice(&self.cur);
    }
}

/// RANSAC options. Defaults are general-purpose; each stage overrides the values it needs.
#[derive(Clone, Debug, PartialEq)]
pub struct RansacParams {
    /// Inlier threshold (before squaring). Inlier if residual <= max_error².
    pub max_error: f64,
    /// Prior inlier ratio (for computing the static iteration cap).
    pub min_inlier_ratio: f64,
    /// Target success probability (for computing the dynamic iteration cap).
    pub confidence: f64,
    /// Multiplier of the dynamic iteration cap.
    pub dyn_trials_factor: f64,
    /// Minimum number of iterations.
    pub min_trials: usize,
    /// Maximum number of iterations.
    pub max_trials: usize,
    /// None = non-deterministic.
    pub random_seed: Option<u64>,
}

impl Default for RansacParams {
    fn default() -> Self {
        Self {
            max_error: 0.0,
            min_inlier_ratio: 0.1,
            confidence: 0.99,
            dyn_trials_factor: 3.0,
            min_trials: 0,
            max_trials: i32::MAX as usize,
            random_seed: None,
        }
    }
}

/// Support: more inliers is better; with equal counts, a smaller residual sum is better.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Support {
    /// Number of inliers.
    pub num_inliers: usize,
    /// Sum of inlier squared residuals.
    pub residual_sum: f64,
}

impl Default for Support {
    fn default() -> Self {
        Self { num_inliers: 0, residual_sum: f64::INFINITY }
    }
}

impl Support {
    /// Computed from squared residuals and a squared threshold.
    pub fn measure(residuals: &[f64], max_residual: f64) -> Self {
        let mut s = Self { num_inliers: 0, residual_sum: 0.0 };
        for &r in residuals {
            if r <= max_residual {
                s.num_inliers += 1;
                s.residual_sum += r;
            }
        }
        s
    }
    /// Whether `self` is better than `other` (by inlier count, then residual sum).
    pub fn is_better(&self, other: &Support) -> bool {
        if self.num_inliers != other.num_inliers {
            self.num_inliers > other.num_inliers
        } else {
            self.residual_sum < other.residual_sum
        }
    }
}

/// RANSAC result.
#[derive(Clone, Debug)]
pub struct RansacReport<M> {
    /// Whether a model was found.
    pub success: bool,
    /// Number of iterations performed.
    pub num_trials: usize,
    /// Support of the final model.
    pub support: Support,
    /// Per-datum inlier flags.
    pub inlier_mask: Vec<bool>,
    /// Final model (None on failure).
    pub model: Option<M>,
}

impl<M> RansacReport<M> {
    fn failed(num_trials: usize) -> Self {
        Self { success: false, num_trials, support: Support::default(), inlier_mask: Vec::new(), model: None }
    }
}

/// Required iteration count: ceil( ln(1−conf) / ln(1−P_good) × multiplier ),
/// P_good = Π_{i<k} (n_in − i)/(n − i). P_good ≥ 1 → 1, P_good = 0 or conf = 1 → usize::MAX.
pub fn compute_num_trials(num_inliers: usize, num_samples: usize, min_num_samples: usize, confidence: f64, multiplier: f64) -> usize {
    if num_inliers < min_num_samples || num_samples < min_num_samples {
        return usize::MAX;
    }
    let mut p = 1.0f64;
    for i in 0..min_num_samples {
        p *= (num_inliers - i) as f64 / (num_samples - i) as f64;
    }
    if p >= 1.0 {
        return 1;
    }
    if p <= 0.0 {
        return usize::MAX;
    }
    let nom = 1.0 - confidence;
    if nom <= 0.0 {
        return usize::MAX;
    }
    let denom = 1.0 - p;
    if denom <= 0.0 {
        return 1;
    }
    let v = (nom.ln() / denom.ln() * multiplier).ceil();
    if v.is_nan() || v >= usize::MAX as f64 {
        usize::MAX
    } else {
        v as usize
    }
}

/// Static cap at construction: computed with n = 100000, n_in = floor(ratio × 100000), then the minimum with max_trials.
pub fn static_max_num_trials(opts: &RansacParams, min_num_samples: usize) -> usize {
    let n = 100_000usize;
    let n_in = (opts.min_inlier_ratio * n as f64).floor() as usize;
    compute_num_trials(n_in, n, min_num_samples, opts.confidence, opts.dyn_trials_factor).min(opts.max_trials)
}

/// Random number generator: fixed if a seed is given, otherwise seeded from OS/thread randomness.
pub fn make_rng(seed: Option<u64>) -> Pcg64 {
    match seed {
        Some(s) => Pcg64::seed_from_u64(s),
        None => Pcg64::seed_from_u64(rand::rng().random::<u64>()),
    }
}

fn gather<T: Clone>(src: &[T], idx: &[usize], out: &mut Vec<T>) {
    out.clear();
    out.extend(idx.iter().map(|&i| src[i].clone()));
}

/// Shared internal loop. LO-RANSAC if `local` is present.
fn run<E, L, S>(est: &E, local: Option<&L>, mut sampler: S, opts: &RansacParams, x: &[E::X], y: &[E::Y]) -> RansacReport<E::Model>
where
    E: Estimator,
    L: Estimator<X = E::X, Y = E::Y, Model = E::Model>,
    S: Sampler,
{
    let n = x.len();
    let k = est.min_num_samples();
    assert_eq!(n, y.len(), "x, y 길이 불일치");
    if n < k || k == 0 {
        return RansacReport::failed(0);
    }
    let max_residual = opts.max_error * opts.max_error;
    let mut rng = make_rng(opts.random_seed);
    sampler.initialize(n);
    let max_trials = static_max_num_trials(opts, k).min(sampler.max_num_samples());
    let mut dyn_max = max_trials;

    let mut best: Option<(E::Model, Support, bool)> = None;
    let mut best_support = Support::default();
    let mut idx = Vec::with_capacity(k);
    let (mut xs, mut ys) = (Vec::with_capacity(k), Vec::with_capacity(k));
    let mut models = Vec::new();
    let mut residuals = Vec::with_capacity(n);
    let mut local_models = Vec::new();
    let (mut lx, mut ly) = (Vec::new(), Vec::new());
    let mut local_res = Vec::with_capacity(n);
    let mut trials = 0usize;

    'outer: while trials < max_trials {
        let t = trials;
        trials += 1;
        sampler.sample(&mut rng, &mut idx);
        gather(x, &idx, &mut xs);
        gather(y, &idx, &mut ys);
        models.clear();
        est.estimate(&xs, &ys, &mut models);
        for m in models.drain(..) {
            est.residuals(x, y, &m, &mut residuals);
            let support = Support::measure(&residuals, max_residual);
            if support.is_better(&best_support) {
                let mut cand = (m, support, false);
                if let Some(le) = local {
                    // Local optimization.
                    if support.num_inliers > k && support.num_inliers >= le.min_num_samples() {
                        let mut cur_res = residuals.clone();
                        for _ in 0..10 {
                            let prev = cand.1.num_inliers;
                            lx.clear();
                            ly.clear();
                            for (i, r) in cur_res.iter().enumerate() {
                                if *r <= max_residual {
                                    lx.push(x[i].clone());
                                    ly.push(y[i].clone());
                                }
                            }
                            local_models.clear();
                            le.estimate(&lx, &ly, &mut local_models);
                            for lm in local_models.drain(..) {
                                le.residuals(x, y, &lm, &mut local_res);
                                let ls = Support::measure(&local_res, max_residual);
                                if ls.is_better(&cand.1) {
                                    cand = (lm, ls, true);
                                    std::mem::swap(&mut cur_res, &mut local_res);
                                }
                            }
                            if cand.1.num_inliers <= prev {
                                break;
                            }
                        }
                    }
                }
                best_support = cand.1;
                best = Some(cand);
                dyn_max = compute_num_trials(best_support.num_inliers, n, k, opts.confidence, opts.dyn_trials_factor);
            }
            if t >= dyn_max && t >= opts.min_trials {
                break 'outer;
            }
        }
    }

    let Some((model, support, is_local)) = best else { return RansacReport::failed(trials) };
    if support.num_inliers < k {
        return RansacReport { success: false, num_trials: trials, support, inlier_mask: Vec::new(), model: Some(model) };
    }
    match (is_local, local) {
        (true, Some(le)) => le.residuals(x, y, &model, &mut residuals),
        _ => est.residuals(x, y, &model, &mut residuals),
    }
    let inlier_mask = residuals.iter().map(|r| *r <= max_residual).collect();
    RansacReport { success: true, num_trials: trials, support, inlier_mask, model: Some(model) }
}

/// Basic RANSAC (random sampling).
pub fn ransac<E: Estimator>(est: &E, opts: &RansacParams, x: &[E::X], y: &[E::Y]) -> RansacReport<E::Model> {
    run::<E, E, _>(est, None, RandomSubsetSampler::new(est.min_num_samples()), opts, x, y)
}

/// LO-RANSAC (random sampling). `local` is the non-minimal solver.
pub fn lo_ransac<E, L>(est: &E, local: &L, opts: &RansacParams, x: &[E::X], y: &[E::Y]) -> RansacReport<E::Model>
where
    E: Estimator,
    L: Estimator<X = E::X, Y = E::Y, Model = E::Model>,
{
    run(est, Some(local), RandomSubsetSampler::new(est.min_num_samples()), opts, x, y)
}

/// General form with a specified sampler. Basic RANSAC if `local` is None.
pub fn ransac_with_sampler<E, L, S>(
    est: &E,
    local: Option<&L>,
    sampler: S,
    opts: &RansacParams,
    x: &[E::X],
    y: &[E::Y],
) -> RansacReport<E::Model>
where
    E: Estimator,
    L: Estimator<X = E::X, Y = E::Y, Model = E::Model>,
    S: Sampler,
{
    run(est, local, sampler, opts, x, y)
}

/// RANSAC that emits hypotheses in batches and proceeds with their support computed outside (e.g. on the GPU).
///
/// Follows the same sample, model and decision order as [`ransac_with_sampler`]. It emits the hypotheses of several
/// iterations at once ([`propose`](Self::propose)) and replays the iterations in order with the received support
/// ([`consume`](Self::consume)). If the received support equals the CPU computation ([`Support::measure`]), the result is the
/// same. If a stop condition is reached before the end of a batch, the remaining hypotheses are discarded (no effect on the
/// result). Local optimization (LO) and the final inlier mask are rarely invoked, so they are computed on the CPU.
pub struct BatchedRansac<'a, E: Estimator, L, S> {
    est: E,
    local: Option<L>,
    sampler: S,
    x: &'a [E::X],
    y: &'a [E::Y],
    min_trials: usize,
    confidence: f64,
    dyn_trials_factor: f64,
    max_residual: f64,
    rng: Pcg64,
    max_trials: usize,
    dyn_max: usize,
    best: Option<(E::Model, Support, bool)>,
    best_support: Support,
    trials: usize,
    done: bool,
    /// Number of models per iteration in the current batch.
    counts: Vec<usize>,
    /// Models of the current batch (concatenated in iteration order).
    models: Vec<E::Model>,
    /// Upper bound of the next batch size. Starts at min_batch and doubles with each batch.
    ramp: usize,
}

impl<'a, E, L, S> BatchedRansac<'a, E, L, S>
where
    E: Estimator,
    L: Estimator<X = E::X, Y = E::Y, Model = E::Model>,
    S: Sampler,
{
    /// LO-RANSAC if `local` is set. With less data than the minimal sample size, it starts in the finished state.
    pub fn new(est: E, local: Option<L>, mut sampler: S, opts: &RansacParams, x: &'a [E::X], y: &'a [E::Y]) -> Self {
        let n = x.len();
        let k = est.min_num_samples();
        assert_eq!(n, y.len(), "x, y 길이 불일치");
        let degenerate = n < k || k == 0;
        let rng = make_rng(opts.random_seed);
        let max_trials = if degenerate {
            0
        } else {
            sampler.initialize(n);
            static_max_num_trials(opts, k).min(sampler.max_num_samples())
        };
        Self {
            est,
            local,
            sampler,
            x,
            y,
            min_trials: opts.min_trials,
            confidence: opts.confidence,
            dyn_trials_factor: opts.dyn_trials_factor,
            max_residual: opts.max_error * opts.max_error,
            rng,
            max_trials,
            dyn_max: max_trials,
            best: None,
            best_support: Support::default(),
            trials: 0,
            done: degenerate || max_trials == 0,
            counts: Vec::new(),
            models: Vec::new(),
            ramp: 0,
        }
    }

    /// Whether it is finished (no more hypotheses to emit).
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Squared inlier threshold (max_error²).
    pub fn max_residual(&self) -> f64 {
        self.max_residual
    }

    /// Number of iterations performed so far.
    pub fn num_trials(&self) -> usize {
        self.trials
    }

    /// Models made by the last [`propose`](Self::propose) (awaiting their support).
    pub fn models(&self) -> &[E::Model] {
        &self.models
    }

    /// Makes the hypotheses of the next batch of iterations. Batch size = the number of iterations left until stopping,
    /// clamped to [min_batch, cap] (within the iteration limit). The cap is min_batch for the first batch and doubles with each
    /// batch up to max_batch. A good hypothesis found early shrinks the dynamic iteration count, so emitting a large batch from
    /// the start would score iterations the CPU path would never run. Samples are drawn in order (preserving the random
    /// sequence) and the minimal solvers run in parallel. Empty list if finished.
    pub fn propose(&mut self, min_batch: usize, max_batch: usize) -> &[E::Model]
    where
        E: Sync,
        E::X: Sync,
        E::Y: Sync,
        E::Model: Send,
    {
        use rayon::prelude::*;
        self.counts.clear();
        self.models.clear();
        if self.done {
            return &self.models;
        }
        let left = self.max_trials - self.trials;
        let stop = self.dyn_max.max(self.min_trials);
        let need = stop.saturating_sub(self.trials).saturating_add(1);
        let lo = min_batch.max(1);
        let cap = self.ramp.max(lo).min(max_batch.max(lo));
        self.ramp = cap.saturating_mul(2);
        let b = need.max(lo).min(cap).min(left);
        let mut samples: Vec<Vec<usize>> = Vec::with_capacity(b);
        let mut idx = Vec::with_capacity(self.est.min_num_samples());
        for _ in 0..b {
            self.sampler.sample(&mut self.rng, &mut idx);
            samples.push(idx.clone());
        }
        let (est, x, y) = (&self.est, self.x, self.y);
        let per: Vec<Vec<E::Model>> = samples
            .par_iter()
            .with_min_len(4)
            .map(|s| {
                let xs: Vec<E::X> = s.iter().map(|&i| x[i].clone()).collect();
                let ys: Vec<E::Y> = s.iter().map(|&i| y[i].clone()).collect();
                let mut m = Vec::new();
                est.estimate(&xs, &ys, &mut m);
                m
            })
            .collect();
        for m in per {
            self.counts.push(m.len());
            self.models.extend(m);
        }
        &self.models
    }

    /// Replays the iterations with the support of [`models`](Self::models) (same order, same length).
    pub fn consume(&mut self, supports: &[Support]) {
        assert_eq!(supports.len(), self.models.len(), "지지도 수가 모델 수와 다름");
        let models = std::mem::take(&mut self.models);
        let counts = std::mem::take(&mut self.counts);
        if self.done {
            return;
        }
        let k = self.est.min_num_samples();
        let n = self.x.len();
        let mut it = models.into_iter().zip(supports.iter().copied());
        let mut residuals = Vec::with_capacity(n);
        for c in counts {
            let t = self.trials;
            self.trials += 1;
            for _ in 0..c {
                let (m, support) = it.next().expect("모델 수 일치");
                if support.is_better(&self.best_support) {
                    let mut cand = (m, support, false);
                    if let Some(le) = &self.local {
                        if support.num_inliers > k && support.num_inliers >= le.min_num_samples() {
                            self.est.residuals(self.x, self.y, &cand.0, &mut residuals);
                            local_optimize(le, self.x, self.y, self.max_residual, &residuals, &mut cand);
                        }
                    }
                    self.best_support = cand.1;
                    self.best = Some(cand);
                    self.dyn_max = compute_num_trials(self.best_support.num_inliers, n, k, self.confidence, self.dyn_trials_factor);
                }
                if t >= self.dyn_max && t >= self.min_trials {
                    self.done = true;
                    return;
                }
            }
        }
        if self.trials >= self.max_trials {
            self.done = true;
        }
    }

    /// The result (if called before finishing, it wraps up with the best so far).
    pub fn finish(self) -> RansacReport<E::Model> {
        let k = self.est.min_num_samples();
        let trials = self.trials;
        let Some((model, support, is_local)) = self.best else { return RansacReport::failed(trials) };
        if support.num_inliers < k {
            return RansacReport { success: false, num_trials: trials, support, inlier_mask: Vec::new(), model: Some(model) };
        }
        let mut residuals = Vec::with_capacity(self.x.len());
        match (is_local, &self.local) {
            (true, Some(le)) => le.residuals(self.x, self.y, &model, &mut residuals),
            _ => self.est.residuals(self.x, self.y, &model, &mut residuals),
        }
        let max_residual = self.max_residual;
        let inlier_mask = residuals.iter().map(|r| *r <= max_residual).collect();
        RansacReport { success: true, num_trials: trials, support, inlier_mask, model: Some(model) }
    }
}

/// LO step (same rule as the local optimization of [`lo_ransac`]). `residuals` = residuals of the candidate model.
fn local_optimize<L: Estimator>(
    le: &L,
    x: &[L::X],
    y: &[L::Y],
    max_residual: f64,
    residuals: &[f64],
    cand: &mut (L::Model, Support, bool),
) {
    let mut cur_res = residuals.to_vec();
    let (mut lx, mut ly) = (Vec::new(), Vec::new());
    let mut local_models = Vec::new();
    let mut local_res = Vec::with_capacity(x.len());
    for _ in 0..10 {
        let prev = cand.1.num_inliers;
        lx.clear();
        ly.clear();
        for (i, r) in cur_res.iter().enumerate() {
            if *r <= max_residual {
                lx.push(x[i].clone());
                ly.push(y[i].clone());
            }
        }
        local_models.clear();
        le.estimate(&lx, &ly, &mut local_models);
        for lm in local_models.drain(..) {
            le.residuals(x, y, &lm, &mut local_res);
            let ls = Support::measure(&local_res, max_residual);
            if ls.is_better(&cand.1) {
                *cand = (lm, ls, true);
                std::mem::swap(&mut cur_res, &mut local_res);
            }
        }
        if cand.1.num_inliers <= prev {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_trials_formula() {
        let mk = |ratio: f64, conf: f64| RansacParams {
            min_inlier_ratio: ratio,
            confidence: conf,
            max_trials: usize::MAX,
            ..Default::default()
        };
        assert_eq!(static_max_num_trials(&mk(0.25, 0.999), 4), 5296);
        assert_eq!(static_max_num_trials(&mk(0.25, 0.999), 5), 21217);
        assert_eq!(static_max_num_trials(&mk(0.25, 0.999), 7), 339734);
        assert_eq!(static_max_num_trials(&mk(0.7, 0.999), 1), 18);
        assert_eq!(static_max_num_trials(&mk(0.1, 0.99), 3), 13813);
        assert_eq!(compute_num_trials(10, 10, 3, 0.99, 3.0), 1);
        assert_eq!(compute_num_trials(2, 10, 3, 0.99, 3.0), usize::MAX);
        assert_eq!(compute_num_trials(5, 10, 3, 1.0, 3.0), usize::MAX);
    }

    /// 2D line y = a x + b. Minimum 2 points, non-minimal = least squares.
    struct Line;
    impl Estimator for Line {
        type X = f64;
        type Y = f64;
        type Model = (f64, f64);
        fn min_num_samples(&self) -> usize {
            2
        }
        fn estimate(&self, x: &[f64], y: &[f64], models: &mut Vec<(f64, f64)>) {
            let n = x.len() as f64;
            let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
            let sxx: f64 = x.iter().map(|v| (v - mx) * (v - mx)).sum();
            if sxx == 0.0 {
                return;
            }
            let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
            let a = sxy / sxx;
            models.push((a, my - a * mx));
        }
        fn residuals(&self, x: &[f64], y: &[f64], m: &(f64, f64), r: &mut Vec<f64>) {
            r.clear();
            r.extend(x.iter().zip(y).map(|(a, b)| (b - (m.0 * a + m.1)).powi(2)));
        }
    }

    fn data(seed: u64) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
        let mut rng = Pcg64::seed_from_u64(seed);
        let (mut x, mut y, mut truth) = (vec![], vec![], vec![]);
        for i in 0..400 {
            let xv = rng.random_range(-10.0..10.0);
            if i % 2 == 0 {
                x.push(xv);
                y.push(2.0 * xv - 1.0 + rng.random_range(-0.05..0.05));
                truth.push(true);
            } else {
                x.push(xv);
                y.push(rng.random_range(-40.0..40.0));
                truth.push(false);
            }
        }
        (x, y, truth)
    }

    #[test]
    fn line_fit_50pct_outliers() {
        let (x, y, truth) = data(3);
        let opts = RansacParams { max_error: 0.1, random_seed: Some(1), ..Default::default() };
        let r = ransac(&Line, &opts, &x, &y);
        assert!(r.success);
        let (a, b) = r.model.unwrap();
        assert!((a - 2.0).abs() < 0.05 && (b + 1.0).abs() < 0.1, "{a} {b}");
        let lo = lo_ransac(&Line, &Line, &opts, &x, &y);
        assert!(lo.success);
        let (a, b) = lo.model.unwrap();
        assert!((a - 2.0).abs() < 0.005 && (b + 1.0).abs() < 0.01, "{a} {b}");
        // Contains all true inliers, almost no outliers
        let tp = lo.inlier_mask.iter().zip(&truth).filter(|(m, t)| **m && **t).count();
        assert_eq!(tp, 200);
        assert!(lo.support.num_inliers <= 205);
        assert!(lo.num_trials < 200, "동적 상한으로 조기 종료: {}", lo.num_trials);
        // Determinism
        let lo2 = lo_ransac(&Line, &Line, &opts, &x, &y);
        assert_eq!(lo2.num_trials, lo.num_trials);
        assert_eq!(lo2.model, lo.model);
    }

    #[test]
    fn batched_ramps_batch_size() {
        // The first batch is min_batch, then it doubles (while the dynamic iteration count is still large).
        let (x, y, _) = data(5);
        let opts = RansacParams { max_error: 0.1, random_seed: Some(1), min_trials: 1000, max_trials: 1000, ..Default::default() };
        let mut b = BatchedRansac::new(Line, Some(Line), RandomSubsetSampler::new(2), &opts, &x[..], &y[..]);
        let mut sizes = Vec::new();
        let mut res = Vec::new();
        while !b.is_done() && sizes.len() < 4 {
            b.propose(16, 512);
            sizes.push(b.counts.len());
            let s: Vec<Support> = b
                .models()
                .iter()
                .map(|m| {
                    Line.residuals(&x, &y, m, &mut res);
                    Support::measure(&res, b.max_residual())
                })
                .collect();
            b.consume(&s);
        }
        assert_eq!(sizes, vec![16, 32, 64, 128]);
    }

    #[test]
    fn batched_matches_sequential() {
        // With support computed on the CPU, the result equals lo_ransac / ransac_with_sampler regardless of the batch size.
        let (x, y, _) = data(5);
        for seed in 0..5u64 {
            let opts = RansacParams { max_error: 0.1, random_seed: Some(seed), min_trials: 20, ..Default::default() };
            let want = lo_ransac(&Line, &Line, &opts, &x, &y);
            for (lo, hi) in [(1usize, 1usize), (7, 64), (16, 512)] {
                let mut b = BatchedRansac::new(Line, Some(Line), RandomSubsetSampler::new(2), &opts, &x[..], &y[..]);
                let mut res = Vec::new();
                while !b.is_done() {
                    b.propose(lo, hi);
                    let s: Vec<Support> = b
                        .models()
                        .iter()
                        .map(|m| {
                            Line.residuals(&x, &y, m, &mut res);
                            Support::measure(&res, b.max_residual())
                        })
                        .collect();
                    b.consume(&s);
                }
                let got = b.finish();
                assert_eq!(got.num_trials, want.num_trials, "seed {seed} 묶음 {lo}..{hi}");
                assert_eq!(got.model, want.model);
                assert_eq!(got.inlier_mask, want.inlier_mask);
                assert_eq!(got.support, want.support);
            }
        }
        // Exhaustive sampler + too little data.
        let opts = RansacParams { max_error: 0.1, random_seed: Some(1), ..Default::default() };
        let want = ransac_with_sampler(&Line, Some(&Line), ExhaustiveSampler::new(2), &opts, &x[..20], &y[..20]);
        let mut b = BatchedRansac::new(Line, Some(Line), ExhaustiveSampler::new(2), &opts, &x[..20], &y[..20]);
        let mut res = Vec::new();
        while !b.is_done() {
            b.propose(8, 32);
            let s: Vec<Support> = b
                .models()
                .iter()
                .map(|m| {
                    Line.residuals(&x[..20], &y[..20], m, &mut res);
                    Support::measure(&res, b.max_residual())
                })
                .collect();
            b.consume(&s);
        }
        let got = b.finish();
        assert_eq!((got.num_trials, got.model), (want.num_trials, want.model));
        let b = BatchedRansac::new(Line, Some(Line), RandomSubsetSampler::new(2), &opts, &[1.0][..], &[1.0][..]);
        assert!(b.is_done());
        assert!(!b.finish().success);
    }

    #[test]
    fn combination_sampler_and_failure() {
        let mut s = ExhaustiveSampler::new(2);
        s.initialize(4);
        assert_eq!(s.max_num_samples(), 6);
        let mut rng = make_rng(Some(0));
        let mut out = vec![];
        let mut all = vec![];
        for _ in 0..7 {
            s.sample(&mut rng, &mut out);
            all.push(out.clone());
        }
        assert_eq!(all[0], vec![0, 1]);
        assert_eq!(all[5], vec![2, 3]);
        assert_eq!(all[6], vec![0, 1]);
        let opts = RansacParams { max_error: 0.1, random_seed: Some(1), ..Default::default() };
        let r = ransac(&Line, &opts, &[1.0], &[1.0]);
        assert!(!r.success);
        let (x, y, _) = data(4);
        let r = ransac_with_sampler(&Line, Some(&Line), ExhaustiveSampler::new(2), &opts, &x[..20], &y[..20]);
        assert!(r.success);
        assert!(r.num_trials <= 190);
    }
}
