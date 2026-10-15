/*
 * tr.rs
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
 * SPDX-License-Identifier: Apache-2.0
 */

//! 신뢰 영역 Levenberg–Marquardt 바깥 루프(일반적인 기본 허용치 사용).
//!
//! 문제별 선형대수(Schur 보수, 소형 밀집)는 [`TrProblem`] 구현이 맡는다.

use crate::Termination;

#[derive(Clone, Copy, Debug)]
pub(crate) struct TrOptions {
    pub max_num_iterations: usize,
    pub function_tolerance: f64,
    pub gradient_tolerance: f64,
    pub parameter_tolerance: f64,
    pub initial_radius: f64,
    pub max_radius: f64,
    pub min_radius: f64,
    pub min_relative_decrease: f64,
    pub max_consecutive_invalid_steps: usize,
}

impl TrOptions {
    pub fn new(max_iter: usize, ftol: f64, gtol: f64, ptol: f64) -> Self {
        Self {
            max_num_iterations: max_iter,
            function_tolerance: ftol,
            gradient_tolerance: gtol,
            parameter_tolerance: ptol,
            initial_radius: 1e4,
            max_radius: 1e16,
            min_radius: 1e-32,
            min_relative_decrease: 1e-3,
            max_consecutive_invalid_steps: 10,
        }
    }
}

pub(crate) struct StepInfo {
    /// 선형 모형이 예측한 비용 감소(양수 기대).
    pub model_cost_change: f64,
    /// ‖x_후보 − x‖ (주변 공간).
    pub step_norm: f64,
    /// ‖x‖.
    pub x_norm: f64,
}

pub(crate) trait TrProblem {
    /// 현재 x 에서 잔차·야코비안·기울기를 평가. `first` 면 야코비 열 배율도 정한다.
    /// 반환: 비용(½Σρ). 평가 실패(비유한) 면 None.
    fn linearize(&mut self, first: bool) -> Option<f64>;
    /// ‖x − ⊞(x, −g)‖∞.
    fn grad_inf_norm(&self) -> f64;
    /// 감쇠 μ = radius 로 LM 단계를 풀고 후보 x 를 만든다. 선형 풀이 실패면 None.
    fn compute_step(&mut self, radius: f64) -> Option<StepInfo>;
    /// 후보 x 의 비용.
    fn candidate_cost(&mut self) -> Option<f64>;
    /// 후보를 현재 x 로 채택.
    fn accept_candidate(&mut self);
}

#[derive(Clone, Debug)]
pub(crate) struct TrResult {
    pub num_iterations: usize,
    pub num_successful_steps: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub termination: Termination,
}

pub(crate) fn minimize<P: TrProblem>(p: &mut P, o: &TrOptions) -> TrResult {
    let mut res = TrResult {
        num_iterations: 0,
        num_successful_steps: 0,
        initial_cost: f64::NAN,
        final_cost: f64::NAN,
        termination: Termination::Failure,
    };
    let Some(mut cost) = p.linearize(true) else {
        return res;
    };
    res.initial_cost = cost;
    res.final_cost = cost;
    if p.grad_inf_norm() <= o.gradient_tolerance {
        res.termination = Termination::Convergence;
        return res;
    }
    let mut radius = o.initial_radius;
    let mut decrease_factor = 2.0;
    let mut invalid = 0usize;
    loop {
        if res.num_iterations >= o.max_num_iterations {
            res.termination = Termination::NoConvergence;
            break;
        }
        res.num_iterations += 1;
        let step = p.compute_step(radius).filter(|s| s.model_cost_change.is_finite() && s.model_cost_change >= 0.0);
        let Some(step) = step else {
            invalid += 1;
            if invalid > o.max_consecutive_invalid_steps {
                res.termination = Termination::Failure;
                break;
            }
            radius /= decrease_factor;
            decrease_factor *= 2.0;
            if radius < o.min_radius {
                res.termination = Termination::Convergence;
                break;
            }
            continue;
        };
        invalid = 0;
        if step.step_norm <= o.parameter_tolerance * (step.x_norm + o.parameter_tolerance) {
            res.termination = Termination::Convergence;
            break;
        }
        let new_cost = p.candidate_cost().filter(|c| c.is_finite());
        let (cost_change, rho) = match new_cost {
            Some(nc) => {
                let ch = cost - nc;
                (ch, ch / step.model_cost_change)
            }
            None => (f64::NAN, f64::NAN),
        };
        if cost_change.is_finite() && cost_change.abs() <= o.function_tolerance * cost {
            // 설계 결정: 함수 기준 도달 시 개선된 단계면 채택하고 끝낸다.
            if cost_change > 0.0 {
                p.accept_candidate();
                res.num_successful_steps += 1;
                match p.linearize(false) {
                    Some(c) => cost = c,
                    None => {
                        res.termination = Termination::Failure;
                        break;
                    }
                }
            }
            res.termination = Termination::Convergence;
            break;
        }
        if rho > o.min_relative_decrease {
            p.accept_candidate();
            res.num_successful_steps += 1;
            match p.linearize(false) {
                Some(c) => cost = c,
                None => {
                    res.termination = Termination::Failure;
                    break;
                }
            }
            let f = 1.0 - (2.0 * rho - 1.0).powi(3);
            radius = (radius / f.max(1.0 / 3.0)).min(o.max_radius);
            decrease_factor = 2.0;
            if p.grad_inf_norm() <= o.gradient_tolerance {
                res.termination = Termination::Convergence;
                break;
            }
        } else {
            radius /= decrease_factor;
            decrease_factor *= 2.0;
            if radius < o.min_radius {
                res.termination = Termination::Convergence;
                break;
            }
        }
    }
    res.final_cost = cost;
    res
}
