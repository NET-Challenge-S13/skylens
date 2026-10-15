/*
 * geodesy.rs
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

//! WGS84 측지 변환. 각도 입력·출력은 도(degree).

use cumulus3d_core::{Mat3, Vec3};

/// 장반경 a (m).
pub const WGS84_A: f64 = 6378137.0;
/// 편평률 f.
pub const WGS84_F: f64 = 1.0 / 298.257223563;
/// 단반경 b = (1 − f) a (m).
pub const WGS84_B: f64 = (1.0 - WGS84_F) * WGS84_A;
/// 제1이심률 제곱 e² = f (2 − f).
pub const WGS84_E2: f64 = WGS84_F * (2.0 - WGS84_F);

/// 묘유선 곡률반경 N(φ).
fn prime_vertical_radius(sin_lat: f64) -> f64 {
    WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt()
}

/// (위도°, 경도°, 타원체고 m) → ECEF (m).
pub fn lla_to_ecef(lat_deg: f64, lon_deg: f64, alt: f64) -> Vec3 {
    let (sp, cp) = lat_deg.to_radians().sin_cos();
    let (sl, cl) = lon_deg.to_radians().sin_cos();
    let n = prime_vertical_radius(sp);
    Vec3::new((n + alt) * cp * cl, (n + alt) * cp * sl, (n * (1.0 - WGS84_E2) + alt) * sp)
}

/// ECEF → (위도°, 경도°, 고도 m). 반복법(최대 100회, 변화 < 1e−12 에서 종료).
pub fn ecef_to_lla(p: &Vec3) -> (f64, f64, f64) {
    let (x, y, z) = (p.x, p.y, p.z);
    let rho = (x * x + y * y).sqrt();
    let lon = y.atan2(x);
    let mut lat = z.atan2(rho);
    let mut alt = 0.0f64;
    // 설계 결정: 극점(ρ=0)에서 반복식이 0으로 나눈다. 그 경우 닫힌 해(±90°, |Z| − b)를 쓴다.
    if rho < 1e-9 {
        let lat = if z >= 0.0 { 90.0 } else { -90.0 };
        return (lat, lon.to_degrees(), z.abs() - WGS84_B);
    }
    for _ in 0..100 {
        let s = lat.sin();
        let n = prime_vertical_radius(s);
        let new_alt = rho / lat.cos() - n;
        let new_lat = ((z / rho) / (1.0 - WGS84_E2 * n / (n + new_alt))).atan();
        let done = (new_lat - lat).abs() < 1e-12 && (new_alt - alt).abs() < 1e-12;
        lat = new_lat;
        alt = new_alt;
        if done {
            break;
        }
    }
    (lat.to_degrees(), lon.to_degrees(), alt)
}

/// ENU 국소 좌표계(원점 고정). `p_enu = R (p_ecef − r0)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnuFrame {
    /// 원점 (위도°, 경도°, 고도 m) — 입력값.
    pub origin_lla: (f64, f64, f64),
    /// 원점 ECEF.
    pub origin_ecef: Vec3,
    /// ECEF → ENU 회전(행: E, N, U).
    pub rotation: Mat3,
}

impl EnuFrame {
    /// 원점 LLA 로 생성. 원점 위경도를 ECEF 에서 반복법으로 다시 구해 회전을 만든다
    /// (입력과 수치적으로 같으며, 결과의 비트 단위 재현성을 위한 기본 동작).
    pub fn new(lat_deg: f64, lon_deg: f64, alt: f64) -> Self {
        let r0 = lla_to_ecef(lat_deg, lon_deg, alt);
        let (lat0, lon0, _) = ecef_to_lla(&r0);
        let (sp, cp) = lat0.to_radians().sin_cos();
        let (sl, cl) = lon0.to_radians().sin_cos();
        #[rustfmt::skip]
        let rotation = Mat3::new(
            -sl,       cl,      0.0,
            -sp * cl, -sp * sl, cp,
            cp * cl,   cp * sl, sp,
        );
        Self { origin_lla: (lat_deg, lon_deg, alt), origin_ecef: r0, rotation }
    }
    /// ECEF → 이 좌표계의 ENU.
    pub fn ecef_to_enu(&self, p: &Vec3) -> Vec3 {
        self.rotation * (p - self.origin_ecef)
    }
    /// ENU → ECEF.
    pub fn enu_to_ecef(&self, e: &Vec3) -> Vec3 {
        self.rotation.transpose() * e + self.origin_ecef
    }
    /// (위도°, 경도°, 타원체고 m) → ENU.
    pub fn lla_to_enu(&self, lat_deg: f64, lon_deg: f64, alt: f64) -> Vec3 {
        self.ecef_to_enu(&lla_to_ecef(lat_deg, lon_deg, alt))
    }
    /// ENU → (위도°, 경도°, 타원체고 m).
    pub fn enu_to_lla(&self, e: &Vec3) -> (f64, f64, f64) {
        ecef_to_lla(&self.enu_to_ecef(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    fn close(a: &Vec3, b: &Vec3, tol: f64) {
        assert!((a - b).norm() < tol, "{a:?} vs {b:?}");
    }

    #[test]
    fn constants() {
        assert!((WGS84_B - 6356752.314245179).abs() < 1e-8);
        assert!((WGS84_E2 - 0.0066943799901413165).abs() < 1e-18);
    }

    #[test]
    fn lla_to_ecef_known() {
        close(&lla_to_ecef(0.0, 0.0, 0.0), &Vec3::new(6378137.0, 0.0, 0.0), 1e-4);
        close(&lla_to_ecef(90.0, 0.0, 0.0), &Vec3::new(0.0, 0.0, 6356752.314245), 1e-4);
        close(&lla_to_ecef(37.5, 127.0, 50.0), &Vec3::new(-3049062.3651, 4046242.4224, 3861594.5369), 1e-4);
    }

    #[test]
    fn enu_known() {
        let f = EnuFrame::new(37.5, 127.0, 50.0);
        close(&f.lla_to_enu(37.5, 127.0, 50.0), &Vec3::zeros(), 1e-9);
        close(&f.lla_to_enu(37.5001, 127.0001, 60.0), &Vec3::new(8.8426, 11.0988, 10.0000), 1e-3);
    }

    /// 같은 위도·고도에서 경도만 Δ 이동: 정확해 E = (N+h)cosφ sinΔ, N = −sinφ·(N+h)cosφ(cosΔ−1), U = (N+h)cos²φ(cosΔ−1).
    #[test]
    fn enu_one_degree_east() {
        let (lat, lon, h) = (37.5f64, 127.0f64, 50.0f64);
        let f = EnuFrame::new(lat, lon, h);
        let e = f.lla_to_enu(lat, lon + 1.0, h);
        let (sp, cp) = lat.to_radians().sin_cos();
        let nh = prime_vertical_radius(sp) + h;
        let d = 1.0f64.to_radians();
        let expect = Vec3::new(nh * cp * d.sin(), -sp * nh * cp * (d.cos() - 1.0), nh * cp * cp * (d.cos() - 1.0));
        close(&e, &expect, 1e-6);
        // 근사: 동쪽 1° ≈ 88.4 km, 위로 약 −0.6 km (지구 곡률).
        assert!((e.x - 88_400.0).abs() < 200.0, "{e:?}");
        assert!(e.z < 0.0 && e.y > 0.0);
    }

    #[test]
    fn ecef_lla_roundtrip() {
        let mut rng = cumulus3d_core::ransac::make_rng(Some(7));
        for _ in 0..1000 {
            let lat = rng.random_range(-89.9..89.9);
            let lon = rng.random_range(-180.0..180.0);
            let alt = rng.random_range(-500.0..10000.0);
            let (la, lo, al) = ecef_to_lla(&lla_to_ecef(lat, lon, alt));
            assert!((la - lat).abs() < 1e-9 && (lo - lon).abs() < 1e-9 && (al - alt).abs() < 1e-6);
        }
    }

    #[test]
    fn enu_roundtrip() {
        let f = EnuFrame::new(37.5, 127.0, 50.0);
        let e = Vec3::new(120.0, -40.0, 33.0);
        close(&f.lla_to_enu(f.enu_to_lla(&e).0, f.enu_to_lla(&e).1, f.enu_to_lla(&e).2), &e, 1e-6);
    }
}
