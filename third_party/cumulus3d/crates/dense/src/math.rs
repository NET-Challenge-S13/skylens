/*
 * math.rs
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

//! 작은 수학 도구: 결정적 해시, 오차 함수, 가시 확률 방출, 기하 사전확률, 분위수.
//! 커널 쪽 식과 같은 정의를 두어 손계산 기준값과 단위 시험의 기준으로 쓴다.

/// FNV-1a 64비트(실행·기계와 무관한 결정적 해시).
pub fn hash_bytes(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// 64비트 값 섞기(splitmix64 마무리 단계).
#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// 오차 함수(테일러 급수, |x| ≤ 5 에서 상대 1e−12 수준; 밖은 ±1).
pub fn erf(x: f64) -> f64 {
    if x.abs() > 5.0 {
        return x.signum();
    }
    let mut sum = 0.0;
    let mut term = x; // x^(2n+1) (−1)^n / n!
    let x2 = x * x;
    for n in 0..200 {
        let t = term / (2 * n + 1) as f64;
        sum += t;
        if t.abs() < 1e-17 * sum.abs().max(1e-300) {
            break;
        }
        term *= -x2 / (n + 1) as f64;
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

/// 비용 \[0,2\] 위 절단 반정규 밀도의 정규화 상수 A(σ).
pub fn emission_norm(sigma: f64) -> f64 {
    2.0 / ((2.0 * std::f64::consts::PI).sqrt() * sigma * erf(2.0 / (sigma * std::f64::consts::SQRT_2)))
}

/// "보임" 상태 방출 밀도 E(c) = A·exp(−c²/(2σ²)).
pub fn emission(cost: f64, sigma: f64) -> f64 {
    emission_norm(sigma) * (-cost * cost / (2.0 * sigma * sigma)).exp()
}

/// 방출만 쓴 가시 사후확률 E1/(E1 + 0.5).
pub fn visibility_probability(cost: f64, sigma: f64) -> f64 {
    let e = emission(cost, sigma);
    e / (e + 0.5)
}

/// 삼각측량 사전: cosθ 가 cos(θ_min) 보다 크면(각이 작으면) 감쇠.
pub fn triangulation_prior(cos_theta: f64, min_angle_rad: f64) -> f64 {
    let cmin = min_angle_rad.cos();
    if cos_theta > cmin {
        let s = 1.0 - (1.0 - cos_theta) / (1.0 - cmin);
        (1.0 - s * s).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// 입사각 사전: x = 1 − max(0, cosφ), exp(−x²/(2σ²)).
pub fn incident_prior(cos_phi: f64, sigma: f64) -> f64 {
    let x = 1.0 - cos_phi.max(0.0);
    (-x * x / (2.0 * sigma * sigma)).exp()
}

/// 3×3 호모그래피(행 우선)로 점 하나 옮기기.
pub fn apply_h(h: &[f64; 9], x: f64, y: f64) -> (f64, f64) {
    let z = h[6] * x + h[7] * y + h[8];
    ((h[0] * x + h[1] * y + h[2]) / z, (h[3] * x + h[4] * y + h[5]) / z)
}

/// 해상도 사전: 창 네 모서리를 옮긴 사각형 넓이와 (2R+1)² 의 비 min(a/b, b/a).
pub fn resolution_prior(h: &[f64; 9], x: f64, y: f64, radius: f64) -> f64 {
    let c = [(x - radius, y - radius), (x - radius, y + radius), (x + radius, y + radius), (x + radius, y - radius)];
    let p: Vec<(f64, f64)> = c.iter().map(|&(a, b)| apply_h(h, a, b)).collect();
    let mut area = 0.0;
    for i in 0..4 {
        let (x0, y0) = p[i];
        let (x1, y1) = p[(i + 1) % 4];
        area += x0 * y1 - x1 * y0;
    }
    let a_src = 0.5 * area.abs();
    let a_ref = (2.0 * radius + 1.0).powi(2);
    if a_src <= 0.0 || !a_src.is_finite() {
        return 0.0;
    }
    (a_src / a_ref).min(a_ref / a_src)
}

/// 정렬된 값의 선형 보간 분위수 q∈\[0,1\] (위치 q·(n−1)).
pub fn quantile_sorted(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let pos = q * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    let f = pos - lo as f64;
    v[lo] + (v[hi] - v[lo]) * f
}

/// 중앙값(짝수 개면 가운데 두 값의 평균). 입력을 정렬한다.
pub fn median_in_place(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    quantile_sorted(v, 0.5)
}

/// 이웃 평면 전달: 픽셀 q 의 가설(깊이 dq, 법선 n)이 정하는 평면과 광선 ray_p 의 교점 깊이.
/// 평면이 광선과 거의 평행하거나 교점이 카메라 뒤면 None.
pub fn plane_transfer(dq: f64, n: [f64; 3], ray_q: [f64; 3], ray_p: [f64; 3]) -> Option<f64> {
    let num = n[0] * ray_q[0] + n[1] * ray_q[1] + n[2] * ray_q[2];
    let den = n[0] * ray_p[0] + n[1] * ray_p[1] + n[2] * ray_p[2];
    if den > -1e-6 {
        return None;
    }
    let d = dq * num / den;
    (d > 0.0 && d.is_finite()).then_some(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    #[test]
    fn emission_reference_values() {
        assert!(rel(emission_norm(0.6), 1.330950) < 1e-5);
        assert!(rel(emission(0.0, 0.6), 1.330950) < 1e-5);
        assert!(rel(emission(0.9, 0.6), 0.432096) < 1e-5);
        assert!(rel(emission(1.0, 0.6), 0.331875) < 1e-5);
        assert!(rel(emission(2.0, 0.6), 0.005145) < 1e-4);
    }

    #[test]
    fn erf_values() {
        assert!((erf(0.5) - 0.520_499_877_813_046_5).abs() < 1e-12);
        assert!((erf(2.0) - 0.995_322_265_018_952_7).abs() < 1e-12);
        assert!((erf(-1.0) + 0.842_700_792_949_714_9).abs() < 1e-12);
    }

    #[test]
    fn priors_reference_values() {
        let a1 = 1f64.to_radians();
        assert_eq!(triangulation_prior(1.0, a1), 0.0);
        assert!((triangulation_prior(0.5f64.to_radians().cos(), a1) - 0.437507).abs() < 1e-5);
        assert_eq!(triangulation_prior(2f64.to_radians().cos(), a1), 1.0);
        assert!((incident_prior(1.0, 0.9) - 1.0).abs() < 1e-12);
        assert!((incident_prior(-0.3, 0.9) - 0.539408).abs() < 1e-6);
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!((resolution_prior(&id, 40.0, 30.0, 5.0) - 100.0 / 121.0).abs() < 1e-9);
    }

    #[test]
    fn quantiles() {
        assert!((quantile_sorted(&[1.0, 2.0, 3.0, 4.0], 0.75) - 3.25).abs() < 1e-12);
        let mut v = vec![4.0, 1.0, 3.0, 2.0];
        assert!((median_in_place(&mut v) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn plane_transfer_matches_ray_plane_intersection() {
        // 평면 z = 10 − 0.2 x (카메라 좌표), 법선은 카메라를 향함.
        let n = {
            let v = [0.2f64, 0.0, 1.0];
            let l = (v[0] * v[0] + v[2] * v[2]).sqrt();
            [-v[0] / l, 0.0, -v[2] / l]
        };
        let hit = |r: [f64; 3]| 10.0 / (r[2] + 0.2 * r[0]);
        let rq = [0.1, -0.05, 1.0];
        let rp = [-0.07, 0.12, 1.0];
        let d = plane_transfer(hit(rq), n, rq, rp).unwrap();
        assert!(rel(d, hit(rp)) < 1e-12);
    }
}
