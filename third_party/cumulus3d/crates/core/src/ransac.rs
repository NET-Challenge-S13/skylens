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

//! 범용 RANSAC / LO-RANSAC.
//!
//! 잔차는 "제곱 오차"이고 인라이어 판정은 잔차 ≤ max_error². 지지도 = (인라이어 수, 인라이어 잔차 합).

use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

/// 추정기: 최소/비최소 해법 + 잔차.
pub trait Estimator {
    /// 입력 자료 1(예: 영상1 좌표).
    type X: Clone;
    /// 입력 자료 2(예: 영상2 좌표).
    type Y: Clone;
    /// 추정 모델 타입.
    type Model: Clone;
    /// 해법에 필요한 최소 표본 수.
    fn min_num_samples(&self) -> usize;
    /// 표본으로부터 모델(0개 이상)을 `models` 에 추가한다(호출 전 비워져 있음).
    fn estimate(&self, x: &[Self::X], y: &[Self::Y], models: &mut Vec<Self::Model>);
    /// 모든 자료의 제곱 잔차를 `residuals` 에 채운다(길이 = x.len()).
    fn residuals(&self, x: &[Self::X], y: &[Self::Y], model: &Self::Model, residuals: &mut Vec<f64>);
}

/// 표본 추출기.
pub trait Sampler {
    /// 자료 수 n 으로 초기화.
    fn initialize(&mut self, n: usize);
    /// 서로 다른 표본의 최대 개수(반복 상한에 반영).
    fn max_num_samples(&self) -> usize;
    /// 다음 표본 인덱스를 `out` 에 채운다.
    fn sample(&mut self, rng: &mut Pcg64, out: &mut Vec<usize>);
}

/// 무작위 표본: 부분 피셔-예이츠, 매 반복 이전 셔플 상태에서 이어 섞는다.
#[derive(Clone, Debug)]
pub struct RandomSubsetSampler {
    k: usize,
    perm: Vec<usize>,
}

impl RandomSubsetSampler {
    /// 표본 크기 k 로 생성.
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

/// 조합 표본: 모든 k-조합을 사전식 순서로 차례대로(끝나면 처음부터).
#[derive(Clone, Debug)]
pub struct ExhaustiveSampler {
    k: usize,
    n: usize,
    cur: Vec<usize>,
    first: bool,
}

impl ExhaustiveSampler {
    /// 표본 크기 k 로 생성.
    pub fn new(num_samples: usize) -> Self {
        Self { k: num_samples, n: 0, cur: Vec::new(), first: true }
    }
}

/// 이항계수 C(n, k) (포화).
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
            // 다음 조합
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

/// RANSAC 옵션. 기본값은 범용 설정이며, 각 단계가 필요한 값을 덮어쓴다.
#[derive(Clone, Debug, PartialEq)]
pub struct RansacParams {
    /// 인라이어 임계(제곱 전). 잔차 ≤ max_error² 이면 인라이어.
    pub max_error: f64,
    /// 사전 인라이어 비율(정적 반복 상한 계산용).
    pub min_inlier_ratio: f64,
    /// 성공 확률 목표(동적 반복 상한 계산용).
    pub confidence: f64,
    /// 동적 반복 상한 배수.
    pub dyn_trials_factor: f64,
    /// 최소 반복 수.
    pub min_trials: usize,
    /// 최대 반복 수.
    pub max_trials: usize,
    /// None = 비결정적.
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

/// 지지도: 인라이어 수 많을수록, 같으면 잔차 합 작을수록 우수.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Support {
    /// 인라이어 수.
    pub num_inliers: usize,
    /// 인라이어 제곱 잔차 합.
    pub residual_sum: f64,
}

impl Default for Support {
    fn default() -> Self {
        Self { num_inliers: 0, residual_sum: f64::INFINITY }
    }
}

impl Support {
    /// 제곱 잔차와 제곱 임계로 계산.
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
    /// `self` 가 `other` 보다 나은지(인라이어 수 → 잔차 합 순).
    pub fn is_better(&self, other: &Support) -> bool {
        if self.num_inliers != other.num_inliers {
            self.num_inliers > other.num_inliers
        } else {
            self.residual_sum < other.residual_sum
        }
    }
}

/// RANSAC 결과.
#[derive(Clone, Debug)]
pub struct RansacReport<M> {
    /// 모델을 찾았는지.
    pub success: bool,
    /// 수행한 반복 수.
    pub num_trials: usize,
    /// 최종 모델의 지지도.
    pub support: Support,
    /// 자료별 인라이어 여부.
    pub inlier_mask: Vec<bool>,
    /// 최종 모델(실패 시 None).
    pub model: Option<M>,
}

impl<M> RansacReport<M> {
    fn failed(num_trials: usize) -> Self {
        Self { success: false, num_trials, support: Support::default(), inlier_mask: Vec::new(), model: None }
    }
}

/// 필요 반복 수: ceil( ln(1−conf) / ln(1−P_good) × 배수 ),
/// P_good = Π_{i<k} (n_in − i)/(n − i). P_good ≥ 1 → 1, P_good = 0 또는 conf = 1 → usize::MAX.
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

/// 생성 시 정적 상한: n = 100000, n_in = floor(비율 × 100000) 으로 계산 후 max_trials 와 최솟값.
pub fn static_max_num_trials(opts: &RansacParams, min_num_samples: usize) -> usize {
    let n = 100_000usize;
    let n_in = (opts.min_inlier_ratio * n as f64).floor() as usize;
    compute_num_trials(n_in, n, min_num_samples, opts.confidence, opts.dyn_trials_factor).min(opts.max_trials)
}

/// 난수 생성기: 시드가 있으면 고정, 없으면 OS/스레드 난수로 시드.
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

/// 내부 공통 루프. `local` 이 있으면 LO-RANSAC.
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
                    // 국소 최적화.
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

/// 기본 RANSAC (무작위 표본).
pub fn ransac<E: Estimator>(est: &E, opts: &RansacParams, x: &[E::X], y: &[E::Y]) -> RansacReport<E::Model> {
    run::<E, E, _>(est, None, RandomSubsetSampler::new(est.min_num_samples()), opts, x, y)
}

/// LO-RANSAC (무작위 표본). `local` 은 비최소 해법.
pub fn lo_ransac<E, L>(est: &E, local: &L, opts: &RansacParams, x: &[E::X], y: &[E::Y]) -> RansacReport<E::Model>
where
    E: Estimator,
    L: Estimator<X = E::X, Y = E::Y, Model = E::Model>,
{
    run(est, Some(local), RandomSubsetSampler::new(est.min_num_samples()), opts, x, y)
}

/// 표본기를 지정하는 일반형. `local` 이 None 이면 기본 RANSAC.
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

    /// 2D 직선 y = a x + b. 최소 2점, 비최소 = 최소제곱.
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
        // 모든 진짜 인라이어 포함, 이상치는 거의 없음
        let tp = lo.inlier_mask.iter().zip(&truth).filter(|(m, t)| **m && **t).count();
        assert_eq!(tp, 200);
        assert!(lo.support.num_inliers <= 205);
        assert!(lo.num_trials < 200, "동적 상한으로 조기 종료: {}", lo.num_trials);
        // 결정성
        let lo2 = lo_ransac(&Line, &Line, &opts, &x, &y);
        assert_eq!(lo2.num_trials, lo.num_trials);
        assert_eq!(lo2.model, lo.model);
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
