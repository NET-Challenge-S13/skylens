/*
 * geometry.rs
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

//! 기하: 쿼터니언, 강체 변환(Rigid3), 상사 변환(Sim3), 각도 유틸.

use nalgebra::{Matrix3, Matrix3x4, UnitQuaternion, Vector2, Vector3};
use std::ops::Mul;

/// 2차원 f64 벡터.
pub type Vec2 = Vector2<f64>;
/// 3차원 f64 벡터.
pub type Vec3 = Vector3<f64>;
/// 3×3 f64 행렬.
pub type Mat3 = Matrix3<f64>;
/// 3×4 f64 행렬.
pub type Mat3x4 = Matrix3x4<f64>;

/// 해밀턴 쿼터니언(i²=j²=k²=ijk=−1, 능동 회전 v' = q v q*). 필드 순서 w, x, y, z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    /// 실수부.
    pub w: f64,
    /// i 성분.
    pub x: f64,
    /// j 성분.
    pub y: f64,
    /// k 성분.
    pub z: f64,
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Quat {
    /// 항등 회전.
    pub const IDENTITY: Quat = Quat { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    /// (w, x, y, z)로 생성(정규화하지 않음).
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }
    /// 파일 순서 [w, x, y, z] 에서.
    pub fn from_wxyz(v: [f64; 4]) -> Self {
        Self { w: v[0], x: v[1], y: v[2], z: v[3] }
    }
    /// `[w, x, y, z]` 배열.
    pub fn to_wxyz(&self) -> [f64; 4] {
        [self.w, self.x, self.y, self.z]
    }
    /// 노름.
    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    /// 단위화. 노름 0 이면 항등.
    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n == 0.0 || !n.is_finite() {
            return Self::IDENTITY;
        }
        Self { w: self.w / n, x: self.x / n, y: self.y / n, z: self.z / n }
    }
    /// 켤레.
    pub fn conjugate(&self) -> Self {
        Self { w: self.w, x: -self.x, y: -self.y, z: -self.z }
    }
    /// 단위 쿼터니언의 역 = 켤레.
    pub fn inverse(&self) -> Self {
        self.conjugate()
    }
    /// 4차원 내적.
    pub fn dot(&self, o: &Quat) -> f64 {
        self.w * o.w + self.x * o.x + self.y * o.y + self.z * o.z
    }
    /// 해밀턴 곱 self ⊗ o (먼저 o, 다음 self 회전).
    pub fn hamilton(&self, o: &Quat) -> Quat {
        Quat {
            w: self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            x: self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            y: self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            z: self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        }
    }
    /// 회전 행렬. 단위가 아니어도 정규화된 회전으로 계산.
    pub fn to_rotation_matrix(&self) -> Mat3 {
        let n2 = self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z;
        let s = if n2 > 0.0 { 2.0 / n2 } else { 0.0 };
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        let (xx, yy, zz) = (x * x * s, y * y * s, z * z * s);
        let (xy, xz, yz) = (x * y * s, x * z * s, y * z * s);
        let (wx, wy, wz) = (w * x * s, w * y * s, w * z * s);
        Mat3::new(1.0 - yy - zz, xy - wz, xz + wy, xy + wz, 1.0 - xx - zz, yz - wx, xz - wy, yz + wx, 1.0 - xx - yy)
    }
    /// 회전 행렬 → 단위 쿼터니언(Shepperd 방식).
    pub fn from_rotation_matrix(m: &Mat3) -> Quat {
        let tr = m[(0, 0)] + m[(1, 1)] + m[(2, 2)];
        let q = if tr > 0.0 {
            let s = (tr + 1.0).sqrt() * 2.0;
            Quat { w: 0.25 * s, x: (m[(2, 1)] - m[(1, 2)]) / s, y: (m[(0, 2)] - m[(2, 0)]) / s, z: (m[(1, 0)] - m[(0, 1)]) / s }
        } else if m[(0, 0)] > m[(1, 1)] && m[(0, 0)] > m[(2, 2)] {
            let s = (1.0 + m[(0, 0)] - m[(1, 1)] - m[(2, 2)]).sqrt() * 2.0;
            Quat { w: (m[(2, 1)] - m[(1, 2)]) / s, x: 0.25 * s, y: (m[(0, 1)] + m[(1, 0)]) / s, z: (m[(0, 2)] + m[(2, 0)]) / s }
        } else if m[(1, 1)] > m[(2, 2)] {
            let s = (1.0 + m[(1, 1)] - m[(0, 0)] - m[(2, 2)]).sqrt() * 2.0;
            Quat { w: (m[(0, 2)] - m[(2, 0)]) / s, x: (m[(0, 1)] + m[(1, 0)]) / s, y: 0.25 * s, z: (m[(1, 2)] + m[(2, 1)]) / s }
        } else {
            let s = (1.0 + m[(2, 2)] - m[(0, 0)] - m[(1, 1)]).sqrt() * 2.0;
            Quat { w: (m[(1, 0)] - m[(0, 1)]) / s, x: (m[(0, 2)] + m[(2, 0)]) / s, y: (m[(1, 2)] + m[(2, 1)]) / s, z: 0.25 * s }
        };
        q.normalized()
    }
    /// 축-각(라디안). 축은 정규화된다.
    pub fn from_axis_angle(axis: &Vec3, angle: f64) -> Quat {
        let n = axis.norm();
        if n == 0.0 {
            return Self::IDENTITY;
        }
        let a = axis / n;
        let (s, c) = (0.5 * angle).sin_cos();
        Quat { w: c, x: a.x * s, y: a.y * s, z: a.z * s }
    }
    /// 회전 벡터(지수 사상) → 쿼터니언.
    pub fn from_rotation_vector(rv: &Vec3) -> Quat {
        let theta = rv.norm();
        if theta < 1e-12 {
            // 1차 근사(작은 각에서 정확도 유지).
            return Quat { w: 1.0, x: 0.5 * rv.x, y: 0.5 * rv.y, z: 0.5 * rv.z }.normalized();
        }
        Self::from_axis_angle(rv, theta)
    }
    /// 쿼터니언 → 회전 벡터(로그 사상), 각도 ∈ [0, π].
    pub fn to_rotation_vector(&self) -> Vec3 {
        let q = self.normalized();
        let q = if q.w < 0.0 { Quat { w: -q.w, x: -q.x, y: -q.y, z: -q.z } } else { q };
        let v = Vec3::new(q.x, q.y, q.z);
        let s = v.norm();
        if s < 1e-12 {
            return 2.0 * v;
        }
        let angle = 2.0 * s.atan2(q.w);
        v * (angle / s)
    }
    /// 벡터 회전.
    pub fn rotate(&self, v: &Vec3) -> Vec3 {
        self.to_rotation_matrix() * v
    }
    /// 두 회전 사이 각거리(라디안) = 2·acos|⟨q1,q2⟩|.
    pub fn angular_distance(&self, other: &Quat) -> f64 {
        let d = self.normalized().dot(&other.normalized()).abs().min(1.0);
        2.0 * d.acos()
    }
    /// nalgebra 단위 쿼터니언으로 변환.
    pub fn to_nalgebra(&self) -> UnitQuaternion<f64> {
        UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(self.w, self.x, self.y, self.z))
    }
    /// nalgebra 단위 쿼터니언에서 변환.
    pub fn from_nalgebra(q: &UnitQuaternion<f64>) -> Quat {
        Quat { w: q.w, x: q.i, y: q.j, z: q.k }
    }
}

impl Mul for Quat {
    type Output = Quat;
    fn mul(self, rhs: Quat) -> Quat {
        self.hamilton(&rhs)
    }
}

/// 반대칭(외적) 행렬 `[v]×`.
pub fn skew(v: &Vec3) -> Mat3 {
    Mat3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// 강체 변환 A_to_B: X_B = R·X_A + t.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Rigid3 {
    /// 회전 R.
    pub rotation: Quat,
    /// 평행이동 t.
    pub translation: Vec3,
}

impl Rigid3 {
    /// 항등 변환.
    pub fn identity() -> Self {
        Self { rotation: Quat::IDENTITY, translation: Vec3::zeros() }
    }
    /// (R, t)로 생성.
    pub fn new(rotation: Quat, translation: Vec3) -> Self {
        Self { rotation, translation }
    }
    /// 회전 행렬과 평행이동으로 생성.
    pub fn from_rotation_matrix(r: &Mat3, t: Vec3) -> Self {
        Self { rotation: Quat::from_rotation_matrix(r), translation: t }
    }
    /// 파일 순서 [qw, qx, qy, qz, tx, ty, tz].
    pub fn from_params(p: &[f64; 7]) -> Self {
        Self { rotation: Quat::new(p[0], p[1], p[2], p[3]), translation: Vec3::new(p[4], p[5], p[6]) }
    }
    /// 파일 순서 `[qw, qx, qy, qz, tx, ty, tz]`.
    pub fn to_params(&self) -> [f64; 7] {
        let q = self.rotation;
        let t = self.translation;
        [q.w, q.x, q.y, q.z, t.x, t.y, t.z]
    }
    /// 회전 행렬 R.
    pub fn rotation_matrix(&self) -> Mat3 {
        self.rotation.to_rotation_matrix()
    }
    /// [R | t].
    pub fn matrix(&self) -> Mat3x4 {
        let r = self.rotation_matrix();
        let mut m = Mat3x4::zeros();
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        m.set_column(3, &self.translation);
        m
    }
    /// `[R | t]` 행렬에서 생성.
    pub fn from_matrix(m: &Mat3x4) -> Self {
        let r: Mat3 = m.fixed_view::<3, 3>(0, 0).into_owned();
        Self::from_rotation_matrix(&r, m.column(3).into_owned())
    }
    /// X_B = R X_A + t.
    pub fn transform_point(&self, p: &Vec3) -> Vec3 {
        self.rotation_matrix() * p + self.translation
    }
    /// 역변환 B_to_A: R' = Rᵀ, t' = −Rᵀ t.
    pub fn inverse(&self) -> Self {
        let rinv = self.rotation.conjugate();
        let rt = rinv.to_rotation_matrix();
        Self { rotation: rinv, translation: -(rt * self.translation) }
    }
    /// 합성 A_to_C = self(B_to_C) ∘ rhs(A_to_B). 결과 회전은 정규화.
    pub fn compose(&self, rhs: &Rigid3) -> Rigid3 {
        Rigid3 {
            rotation: self.rotation.hamilton(&rhs.rotation).normalized(),
            translation: self.translation + self.rotation_matrix() * rhs.translation,
        }
    }
    /// world_to_cam 일 때 투영 중심 C = −Rᵀ t.
    pub fn center(&self) -> Vec3 {
        -(self.rotation_matrix().transpose() * self.translation)
    }
    /// world_to_cam 일 때 시선 방향(카메라 +z 의 세계 좌표) = R 의 셋째 행.
    pub fn viewing_direction(&self) -> Vec3 {
        let r = self.rotation_matrix();
        Vec3::new(r[(2, 0)], r[(2, 1)], r[(2, 2)])
    }
}

impl Mul for Rigid3 {
    type Output = Rigid3;
    fn mul(self, rhs: Rigid3) -> Rigid3 {
        self.compose(&rhs)
    }
}

impl Mul<Vec3> for Rigid3 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Vec3 {
        self.transform_point(&rhs)
    }
}

/// 상사 변환 new_from_old: X_new = s·R·X_old + t.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sim3 {
    /// 스케일 s.
    pub scale: f64,
    /// 회전 R.
    pub rotation: Quat,
    /// 평행이동 t.
    pub translation: Vec3,
}

impl Default for Sim3 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Sim3 {
    /// 항등 변환.
    pub fn identity() -> Self {
        Self { scale: 1.0, rotation: Quat::IDENTITY, translation: Vec3::zeros() }
    }
    /// (s, R, t)로 생성.
    pub fn new(scale: f64, rotation: Quat, translation: Vec3) -> Self {
        Self { scale, rotation, translation }
    }
    /// [sR | t].
    pub fn matrix(&self) -> Mat3x4 {
        let r = self.rotation.to_rotation_matrix() * self.scale;
        let mut m = Mat3x4::zeros();
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        m.set_column(3, &self.translation);
        m
    }
    /// [sR | t] → Sim3. s = 첫 열 노름, R = 좌상 3×3 / s.
    pub fn from_matrix(m: &Mat3x4) -> Self {
        let sr: Mat3 = m.fixed_view::<3, 3>(0, 0).into_owned();
        let s = sr.column(0).norm();
        Self { scale: s, rotation: Quat::from_rotation_matrix(&(sr / s)), translation: m.column(3).into_owned() }
    }
    /// X_new = s·R·X_old + t.
    pub fn transform_point(&self, p: &Vec3) -> Vec3 {
        self.scale * (self.rotation.to_rotation_matrix() * p) + self.translation
    }
    /// 역변환: s' = 1/s, R' = Rᵀ, t' = −Rᵀ t / s.
    pub fn inverse(&self) -> Self {
        let rinv = self.rotation.conjugate();
        Self { scale: 1.0 / self.scale, rotation: rinv, translation: -(rinv.to_rotation_matrix() * self.translation) / self.scale }
    }
    /// 합성 self ∘ rhs.
    pub fn compose(&self, rhs: &Sim3) -> Sim3 {
        Sim3 {
            scale: self.scale * rhs.scale,
            rotation: self.rotation.hamilton(&rhs.rotation).normalized(),
            translation: self.scale * (self.rotation.to_rotation_matrix() * rhs.translation) + self.translation,
        }
    }
    /// 자세(world_to_cam 또는 world_to_rig)에 적용: R' = R Qᵀ, t' = s t − R' T.
    pub fn transform_pose(&self, world_to_cam: &Rigid3) -> Rigid3 {
        let rot = world_to_cam.rotation.hamilton(&self.rotation.conjugate()).normalized();
        let t = self.scale * world_to_cam.translation - rot.to_rotation_matrix() * self.translation;
        Rigid3 { rotation: rot, translation: t }
    }
}

impl Mul for Sim3 {
    type Output = Sim3;
    fn mul(self, rhs: Sim3) -> Sim3 {
        self.compose(&rhs)
    }
}

/// 도 → 라디안.
pub fn deg_to_rad(d: f64) -> f64 {
    d.to_radians()
}
/// 라디안 → 도.
pub fn rad_to_deg(r: f64) -> f64 {
    r.to_degrees()
}

/// 두 벡터 사이 각(라디안), acos(clamp(내적/노름곱)). 노름 0 이면 0.
pub fn angle_between(a: &Vec3, b: &Vec3) -> f64 {
    let n = a.norm() * b.norm();
    if n == 0.0 {
        return 0.0;
    }
    (a.dot(b) / n).clamp(-1.0, 1.0).acos()
}

/// 삼각측량 각: (X−C1), (X−C2) 사이 각 θ 의 min(θ, π−θ).
pub fn triangulation_angle(center1: &Vec3, center2: &Vec3, point: &Vec3) -> f64 {
    let theta = angle_between(&(point - center1), &(point - center2));
    theta.min(std::f64::consts::PI - theta)
}

/// 기준선 길이 ‖C1 − C2‖.
pub fn baseline(center1: &Vec3, center2: &Vec3) -> f64 {
    (center1 - center2).norm()
}

/// 여러 투영 중심 쌍 중 최대 삼각측량 각(라디안). 중심 2개 미만이면 0.
pub fn max_triangulation_angle(centers: &[Vec3], point: &Vec3) -> f64 {
    let mut best = 0.0f64;
    for i in 0..centers.len() {
        for j in 0..i {
            best = best.max(triangulation_angle(&centers[i], &centers[j], point));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_quat(seed: u64) -> Quat {
        let f = |k: u64| (((seed * 7919 + k * 104729) % 1000) as f64 / 500.0) - 1.0;
        Quat::new(f(1), f(2), f(3), f(4) + 0.1).normalized()
    }

    #[test]
    fn quat_matrix_roundtrip() {
        for s in 0..50 {
            let q = rand_quat(s);
            let m = q.to_rotation_matrix();
            assert!((m * m.transpose() - Mat3::identity()).norm() < 1e-12);
            assert!((m.determinant() - 1.0).abs() < 1e-12);
            let q2 = Quat::from_rotation_matrix(&m);
            assert!(q.angular_distance(&q2) < 1e-7);
            assert!((q2.to_rotation_matrix() - m).norm() < 1e-12);
        }
    }

    #[test]
    fn hamilton_convention() {
        // i*j = k
        let i = Quat::new(0.0, 1.0, 0.0, 0.0);
        let j = Quat::new(0.0, 0.0, 1.0, 0.0);
        assert_eq!(i * j, Quat::new(0.0, 0.0, 0.0, 1.0));
        // 90° about z maps x to y (active rotation)
        let q = Quat::from_axis_angle(&Vec3::z(), std::f64::consts::FRAC_PI_2);
        assert!((q.rotate(&Vec3::x()) - Vec3::y()).norm() < 1e-15);
        // matrix product matches quaternion product
        let a = rand_quat(3);
        let b = rand_quat(4);
        let m = (a * b).to_rotation_matrix();
        assert!((m - a.to_rotation_matrix() * b.to_rotation_matrix()).norm() < 1e-12);
        // nalgebra agrees
        let na = a.to_nalgebra();
        assert!((na.to_rotation_matrix().matrix() - a.to_rotation_matrix()).norm() < 1e-12);
    }

    #[test]
    fn rotation_vector_roundtrip() {
        for s in 0..20 {
            let q = rand_quat(s);
            let rv = q.to_rotation_vector();
            let q2 = Quat::from_rotation_vector(&rv);
            assert!(q.angular_distance(&q2) < 1e-7);
        }
    }

    #[test]
    fn rigid3_identities() {
        let a = Rigid3::new(rand_quat(1), Vec3::new(1.0, -2.0, 3.0));
        let b = Rigid3::new(rand_quat(2), Vec3::new(-0.5, 0.2, 4.0));
        let p = Vec3::new(0.3, 0.7, -1.1);
        let ab = a * b;
        assert!(((ab * p) - (a * (b * p))).norm() < 1e-12);
        let ia = a.inverse();
        assert!(((ia * (a * p)) - p).norm() < 1e-12);
        let id = a * ia;
        assert!(id.translation.norm() < 1e-12);
        assert!(id.rotation.angular_distance(&Quat::IDENTITY) < 1e-7);
        // center maps to camera origin
        assert!((a * a.center()).norm() < 1e-12);
        let vd = a.viewing_direction();
        assert!((a.rotation_matrix() * vd - Vec3::z()).norm() < 1e-12);
        let m = Rigid3::from_matrix(&a.matrix());
        assert!(((m * p) - (a * p)).norm() < 1e-12);
    }

    #[test]
    fn sim3_identities() {
        let s = Sim3::new(2.5, rand_quat(5), Vec3::new(1.0, 2.0, 3.0));
        let t = Sim3::new(0.3, rand_quat(6), Vec3::new(-1.0, 0.0, 0.5));
        let p = Vec3::new(0.1, -0.2, 0.3);
        assert!(((s * t).transform_point(&p) - s.transform_point(&t.transform_point(&p))).norm() < 1e-12);
        assert!((s.inverse().transform_point(&s.transform_point(&p)) - p).norm() < 1e-12);
        let m = Sim3::from_matrix(&s.matrix());
        assert!((m.scale - s.scale).abs() < 1e-12);
        assert!((m.transform_point(&p) - s.transform_point(&p)).norm() < 1e-12);
        // pose transform: center moves like a point, projections unchanged up to scale
        let pose = Rigid3::new(rand_quat(7), Vec3::new(0.2, 0.1, 5.0));
        let np = s.transform_pose(&pose);
        assert!((np.center() - s.transform_point(&pose.center())).norm() < 1e-10);
        let x = Vec3::new(0.5, 0.5, 2.0);
        let c_old = pose * x;
        let c_new = np * s.transform_point(&x);
        assert!((c_new - s.scale * c_old).norm() < 1e-10);
    }

    #[test]
    fn tri_angle_symmetric() {
        let c1 = Vec3::new(-1.0, 0.0, 0.0);
        let c2 = Vec3::new(1.0, 0.0, 0.0);
        let x = Vec3::new(0.0, 0.0, 1.0);
        assert!((triangulation_angle(&c1, &c2, &x) - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        let x = Vec3::new(0.0, 0.0, 3.0_f64.sqrt());
        assert!((triangulation_angle(&c1, &c2, &x) - std::f64::consts::FRAC_PI_3).abs() < 1e-12);
        // obtuse folded: angle 120° → 60°
        let x = Vec3::new(0.0, 0.0, 1.0 / 3.0_f64.sqrt());
        assert!((triangulation_angle(&c1, &c2, &x) - std::f64::consts::FRAC_PI_3).abs() < 1e-12);
        assert!(triangulation_angle(&c1, &c1, &x) < 1e-7);
        assert_eq!(triangulation_angle(&c1, &c2, &c1), 0.0);
    }
}
