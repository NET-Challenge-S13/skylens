/*
 * abspose.rs
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

//! 단일 자세 정제: 3D 점 고정, 자세 6자유도만, 밀집 6×6 풀이.

use crate::problem::quat_plus;
use crate::tr::{StepInfo, TrProblem};
use crate::Loss;
use cumulus3d_core::geometry::skew;
use cumulus3d_core::{Camera, Rigid3, Vec2, Vec3};
use nalgebra::{Matrix6, Vector6};

pub(crate) struct AbsPoseProblem<'a> {
    pub camera: &'a Camera,
    pub x2: Vec<Vec2>,
    pub x3: Vec<Vec3>,
    pub loss: Loss,
    pub cur: Rigid3,
    cand: Rigid3,
    scale: [f64; 6],
    scaled: bool,
    jtj: Matrix6<f64>,
    g: Vector6<f64>,
    jac: Vec<[f64; 12]>,
    res: Vec<[f64; 2]>,
}

impl<'a> AbsPoseProblem<'a> {
    pub fn new(camera: &'a Camera, x2: Vec<Vec2>, x3: Vec<Vec3>, loss: Loss, pose: Rigid3) -> Self {
        let n = x2.len();
        Self {
            camera,
            x2,
            x3,
            loss,
            cur: pose,
            cand: pose,
            scale: [1.0; 6],
            scaled: false,
            jtj: Matrix6::zeros(),
            g: Vector6::zeros(),
            jac: vec![[0.0; 12]; n],
            res: vec![[0.0; 2]; n],
        }
    }

    fn cost_of(&self, pose: &Rigid3) -> f64 {
        let r = pose.rotation_matrix();
        let mut c = 0.0;
        for (x2, x3) in self.x2.iter().zip(&self.x3) {
            if let Some(p) = self.camera.cam_to_img(&(r * x3 + pose.translation)) {
                c += self.loss.weight((p - x2).norm_squared()).0;
            }
        }
        0.5 * c
    }

    fn evaluate(&mut self) -> f64 {
        let r = self.cur.rotation_matrix();
        let mut cost = 0.0;
        self.jtj = Matrix6::zeros();
        self.g = Vector6::zeros();
        for i in 0..self.x2.len() {
            let xr = r * self.x3[i];
            let xc = xr + self.cur.translation;
            let Some((pix, m)) = self.camera.cam_to_img_with_jacobian(&xc, None) else {
                self.jac[i] = [0.0; 12];
                self.res[i] = [0.0; 2];
                continue;
            };
            let e = pix - self.x2[i];
            let (rho, w) = self.loss.weight(e.norm_squared());
            cost += rho;
            let jrot = m * skew(&xr) * (-2.0);
            let mut j = [0.0; 12];
            for row in 0..2 {
                for q in 0..3 {
                    j[row * 6 + q] = jrot[(row, q)] * w * self.scale[q];
                    j[row * 6 + 3 + q] = m[(row, q)] * w * self.scale[3 + q];
                }
            }
            let rr = [e.x * w, e.y * w];
            for p in 0..6 {
                self.g[p] += j[p] * rr[0] + j[6 + p] * rr[1];
                for q in 0..6 {
                    self.jtj[(p, q)] += j[p] * j[q] + j[6 + p] * j[6 + q];
                }
            }
            self.jac[i] = j;
            self.res[i] = rr;
        }
        0.5 * cost
    }
}

impl TrProblem for AbsPoseProblem<'_> {
    fn linearize(&mut self, first: bool) -> Option<f64> {
        let mut c = self.evaluate();
        if first {
            for q in 0..6 {
                self.scale[q] = 1.0 / (1.0 + self.jtj[(q, q)].sqrt());
            }
            self.scaled = true;
            c = self.evaluate();
        }
        debug_assert!(self.scaled);
        c.is_finite().then_some(c)
    }

    fn grad_inf_norm(&self) -> f64 {
        let g: [f64; 6] = std::array::from_fn(|i| self.g[i] / self.scale[i]);
        let q = self.cur.rotation;
        let qn = quat_plus(&q, &[-g[0], -g[1], -g[2]]);
        let mut m: f64 = (q.w - qn.w).abs().max((q.x - qn.x).abs()).max((q.y - qn.y).abs()).max((q.z - qn.z).abs());
        for &gi in &g[3..] {
            m = m.max(gi.abs());
        }
        m
    }

    fn compute_step(&mut self, radius: f64) -> Option<StepInfo> {
        let mut a = self.jtj;
        for q in 0..6 {
            a[(q, q)] += self.jtj[(q, q)].clamp(1e-6, 1e32) / radius;
        }
        // 설계 결정: 정규방정식 촐레스키로 푼다(정확 산술에서 밀집 QR 과 같은 해).
        let d = a.cholesky()?.solve(&(-self.g));
        if !d.iter().all(|x| x.is_finite()) {
            return None;
        }
        let mut mc = 0.0;
        for (j, r) in self.jac.iter().zip(&self.res) {
            let y0: f64 = (0..6).map(|c| j[c] * d[c]).sum();
            let y1: f64 = (0..6).map(|c| j[6 + c] * d[c]).sum();
            mc -= r[0] * y0 + r[1] * y1 + 0.5 * (y0 * y0 + y1 * y1);
        }
        let du: [f64; 6] = std::array::from_fn(|i| d[i] * self.scale[i]);
        let q = quat_plus(&self.cur.rotation, &du[..3]).normalized();
        let t = self.cur.translation + Vec3::new(du[3], du[4], du[5]);
        self.cand = Rigid3::new(q, t);
        let p0 = self.cur.to_params();
        let p1 = self.cand.to_params();
        let step_norm = p0.iter().zip(&p1).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
        let x_norm = p0.iter().map(|a| a * a).sum::<f64>().sqrt();
        Some(StepInfo { model_cost_change: mc, step_norm, x_norm })
    }

    fn candidate_cost(&mut self) -> Option<f64> {
        let c = self.cost_of(&self.cand);
        c.is_finite().then_some(c)
    }

    fn accept_candidate(&mut self) {
        self.cur = self.cand;
    }
}
