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

//! Trust-region Levenberg-Marquardt outer loop (using common default tolerances).
//!
//! Problem-specific linear algebra (Schur complement, small dense) is handled by the [`TrProblem`] implementation.

use crate::Termination;

/// Trust-region LM options.
#[derive(Clone, Copy, Debug)]
pub struct TrOptions {
    /// Maximum number of iterations.
    pub max_num_iterations: usize,
    /// Relative cost decrease tolerance (0 = off).
    pub function_tolerance: f64,
    /// Gradient tolerance.
    pub gradient_tolerance: f64,
    /// Parameter change tolerance (0 = off).
    pub parameter_tolerance: f64,
    /// Initial trust-region radius.
    pub initial_radius: f64,
    /// Maximum radius.
    pub max_radius: f64,
    /// Minimum radius (below it the loop ends as converged).
    pub min_radius: f64,
    /// Minimum relative decrease to accept a step.
    pub min_relative_decrease: f64,
    /// Allowed consecutive invalid steps.
    pub max_consecutive_invalid_steps: usize,
}

impl TrOptions {
    /// Builds options from the tolerances, the rest at their defaults.
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

/// Model information of one LM step.
pub struct StepInfo {
    /// Cost decrease predicted by the linear model (expected positive).
    pub model_cost_change: f64,
    /// ‖x_candidate - x‖ (ambient space).
    pub step_norm: f64,
    /// ‖x‖.
    pub x_norm: f64,
}

/// The problem the trust-region loop works on. Public so the CPU and GPU bundle adjusters share the same outer loop.
pub trait TrProblem {
    /// Evaluates residuals, Jacobian, and gradient at the current x. If `first`, also sets the Jacobian column scaling.
    /// Returns: cost (½Σρ). None if evaluation fails (non-finite).
    fn linearize(&mut self, first: bool) -> Option<f64>;
    /// ‖x − ⊞(x, −g)‖∞.
    fn grad_inf_norm(&self) -> f64;
    /// Solves the LM step with damping μ = radius and builds the candidate x. None if the linear solve fails.
    fn compute_step(&mut self, radius: f64) -> Option<StepInfo>;
    /// Cost of the candidate x.
    fn candidate_cost(&mut self) -> Option<f64>;
    /// Accepts the candidate as the current x.
    fn accept_candidate(&mut self);
}

/// Result of the trust-region loop.
#[derive(Clone, Debug)]
pub struct TrResult {
    /// Iterations performed.
    pub num_iterations: usize,
    /// Accepted steps.
    pub num_successful_steps: usize,
    /// Initial cost.
    pub initial_cost: f64,
    /// Final cost.
    pub final_cost: f64,
    /// Termination state.
    pub termination: Termination,
}

/// Trust-region LM outer loop (keeps the scalars only; the heavy work is done by [`TrProblem`]).
pub fn minimize<P: TrProblem>(p: &mut P, o: &TrOptions) -> TrResult {
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
            // Design decision: when the function criterion is reached, accept the step if it improved and finish.
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
