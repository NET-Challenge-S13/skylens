/*
 * loss.rs
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

//! 견고 손실 ρ(s) (s = 잔차 제곱노름)와 잔차·야코비안 보정.

use crate::Loss;

impl Loss {
    /// (ρ(s), ρ'(s), ρ''(s)).
    pub fn evaluate(&self, s: f64) -> [f64; 3] {
        // ρ' 하한: 0 으로 떨어져 야코비안이 사라지는 것을 막는다(최적화 라이브러리 관례).
        const MIN: f64 = f64::MIN_POSITIVE;
        match *self {
            Loss::Trivial => [s, 1.0, 0.0],
            Loss::Huber(a) => {
                let b = a * a;
                if s > b {
                    let r = s.sqrt();
                    let rho1 = (a / r).max(MIN);
                    [2.0 * a * r - b, rho1, -rho1 / (2.0 * s)]
                } else {
                    [s, 1.0, 0.0]
                }
            }
            Loss::SoftL1(a) => {
                let b = a * a;
                let c = 1.0 / b;
                let sum = 1.0 + s * c;
                let tmp = sum.sqrt();
                let rho1 = (1.0 / tmp).max(MIN);
                [2.0 * b * (tmp - 1.0), rho1, -(c * rho1) / (2.0 * sum)]
            }
            Loss::Cauchy(a) => {
                let b = a * a;
                let c = 1.0 / b;
                let sum = 1.0 + s * c;
                let inv = 1.0 / sum;
                [b * sum.ln(), inv.max(MIN), -c * inv * inv]
            }
        }
    }

    /// 비용 기여 ρ(s) 와 잔차·야코비안에 곱할 배율.
    ///
    /// 여기 손실들은 모두 ρ'' ≤ 0 이라 트릭스 보정의 α 항이 0 이 되고(라이브러리 관례),
    /// 보정은 √ρ' 배율 하나로 줄어든다.
    #[inline]
    pub(crate) fn weight(&self, s: f64) -> (f64, f64) {
        match self {
            Loss::Trivial => (s, 1.0),
            _ => {
                let r = self.evaluate(s);
                (r[0], r[1].sqrt())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivatives_match_finite_difference() {
        for loss in [Loss::Trivial, Loss::Huber(1.3), Loss::SoftL1(0.7), Loss::Cauchy(2.0)] {
            for &s in &[0.01f64, 0.5, 1.0, 3.0, 40.0] {
                let h = 1e-6 * s.max(1.0);
                let r = loss.evaluate(s);
                let rp = loss.evaluate(s + h);
                let rm = loss.evaluate(s - h);
                let d1 = (rp[0] - rm[0]) / (2.0 * h);
                let d2 = (rp[1] - rm[1]) / (2.0 * h);
                assert!((d1 - r[1]).abs() < 1e-6, "{loss:?} {s}");
                assert!((d2 - r[2]).abs() < 1e-5, "{loss:?} {s}");
            }
        }
    }
}
