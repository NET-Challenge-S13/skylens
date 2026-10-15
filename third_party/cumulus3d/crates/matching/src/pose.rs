/*
 * pose.rs
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

//! 상대 자세 분해와 cheirality. sfm 도 사용한다.

use crate::essential::essential_eight_point;
use crate::estimators::{fundamental_eight_point, homography_dlt};
use crate::linalg::svd3;
use cumulus3d_core::geometry::triangulation_angle;
use cumulus3d_core::{Camera, Keypoint, Mat3, Rigid3, TwoViewGeometry, TwoViewGeometryConfig, Vec2, Vec3};

/// E 분해 4후보 (R_a,t), (R_b,t), (R_a,−t), (R_b,−t). t 는 단위 벡터. cam1_to_cam2 규약.
pub fn decompose_essential(e: &Mat3) -> Option<[(Mat3, Vec3); 4]> {
    let (mut u, _, v) = svd3(e)?;
    let mut vt = v.transpose();
    if u.determinant() < 0.0 {
        u = -u;
    }
    if vt.determinant() < 0.0 {
        vt = -vt;
    }
    let w = Mat3::new(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let ra = u * w * vt;
    let rb = u * w.transpose() * vt;
    let t = u.column(2).normalize();
    Some([(ra, t), (rb, t), (ra, -t), (rb, -t)])
}

/// 중점 삼각측량. cam1_to_cam2 = (R, t), r1·r2 는 각 카메라 광선(단위 아니어도 됨).
/// 결과 점은 카메라1 좌표. 어느 쪽이든 깊이(λ) ≤ ε 이면 None.
pub fn triangulate_midpoint(r: &Mat3, t: &Vec3, r1: &Vec3, r2: &Vec3) -> Option<Vec3> {
    let r2p = r.tr_mul(r2);
    let c2 = -r.tr_mul(t);
    let a = Mat3::from_columns(&[*r1, -r2p, -c2]);
    let (_, _, vs) = svd3(&a)?;
    let v = vs.column(2);
    if v[2] == 0.0 {
        return None;
    }
    let l1 = v[0] / v[2];
    let l2 = v[1] / v[2];
    if !(l1 > f64::EPSILON && l2 > f64::EPSILON) {
        return None;
    }
    Some(0.5 * (l1 * r1 + c2 + l2 * r2p))
}

/// E → (cam1_to_cam2, 카메라1 좌표 3D 점들). 성공 점 수가 이전 최선 이상이면 갱신(동점은 나중 후보).
/// 점이 0개면 None.
pub fn pose_from_essential(e: &Mat3, rays1: &[Vec3], rays2: &[Vec3]) -> Option<(Rigid3, Vec<Vec3>)> {
    let cands = decompose_essential(e)?;
    let mut best: Option<(Mat3, Vec3, Vec<Vec3>)> = None;
    for (r, t) in cands {
        let pts: Vec<Vec3> = rays1.iter().zip(rays2).filter_map(|(a, b)| triangulate_midpoint(&r, &t, a, b)).collect();
        if best.as_ref().is_none_or(|b| pts.len() >= b.2.len()) {
            best = Some((r, t, pts));
        }
    }
    let (r, t, pts) = best?;
    if pts.is_empty() {
        return None;
    }
    Some((Rigid3::from_rotation_matrix(&r, t), pts))
}

/// H 분해 후보 (R, t, n) 목록. H 는 픽셀 호모그래피(x2 ~ H x1), K1·K2 는 보정 행렬.
/// 순수 회전이면 후보 1개 (H_n, 0, 0). 규약: H_n ∝ R − t nᵀ (평면 nᵀX = d=1, 카메라1 좌표).
pub fn decompose_homography(h: &Mat3, k1: &Mat3, k2: &Mat3) -> Vec<(Mat3, Vec3, Vec3)> {
    let Some(k2i) = k2.try_inverse() else { return Vec::new() };
    let mut hn = k2i * h * k1;
    let Some((_, s, _)) = svd3(&hn) else { return Vec::new() };
    if s[1] == 0.0 || !s[1].is_finite() {
        return Vec::new();
    }
    hn /= s[1];
    if hn.determinant() < 0.0 {
        hn = -hn;
    }
    let sm = hn.transpose() * hn - Mat3::identity();
    let inf_norm = (0..3).map(|i| (0..3).map(|j| sm[(i, j)].abs()).sum::<f64>()).fold(0.0, f64::max);
    if inf_norm < 1e-3 {
        return vec![(hn, Vec3::zeros(), Vec3::zeros())];
    }
    let s00 = sm[(0, 0)];
    let s11 = sm[(1, 1)];
    let s22 = sm[(2, 2)];
    let s01 = sm[(0, 1)];
    let s02 = sm[(0, 2)];
    let s12 = sm[(1, 2)];
    // 여인자(소행렬식의 반대 부호).
    let m00 = s12 * s12 - s11 * s22;
    let m11 = s02 * s02 - s00 * s22;
    let m22 = s01 * s01 - s00 * s11;
    let m01 = s12 * s02 - s01 * s22;
    let m12 = s01 * s02 - s00 * s12;
    let m02 = s11 * s02 - s01 * s12;
    let sgn = |x: f64| if x >= 0.0 { 1.0 } else { -1.0 };
    let rt = |x: f64| x.max(0.0).sqrt();
    let (n1, n2, s_ii) = {
        let a = [s00.abs(), s11.abs(), s22.abs()];
        let idx = if a[0] >= a[1] && a[0] >= a[2] {
            0
        } else if a[1] >= a[2] {
            1
        } else {
            2
        };
        match idx {
            0 => {
                let e = sgn(m12);
                (Vec3::new(s00, s01 + rt(m22), s02 + e * rt(m11)), Vec3::new(s00, s01 - rt(m22), s02 - e * rt(m11)), s00)
            }
            1 => {
                let e = sgn(m02);
                (Vec3::new(s01 + rt(m22), s11, s12 - e * rt(m00)), Vec3::new(s01 - rt(m22), s11, s12 + e * rt(m00)), s11)
            }
            _ => {
                let e = sgn(m01);
                (Vec3::new(s02 + e * rt(m11), s12 + rt(m00), s22), Vec3::new(s02 - e * rt(m11), s12 - rt(m00), s22), s22)
            }
        }
    };
    let n1 = n1.normalize();
    let n2 = n2.normalize();
    let trs = sm.trace();
    let nu = 2.0 * rt(1.0 + trs - m00 - m11 - m22);
    let rho = rt(2.0 + trs + nu);
    let tau = rt(2.0 + trs - nu);
    if nu == 0.0 {
        return Vec::new();
    }
    let es = sgn(s_ii);
    let ts1 = (tau / 2.0) * (es * rho * n2 - tau * n1);
    let ts2 = (tau / 2.0) * (es * rho * n1 - tau * n2);
    let r1 = hn * (Mat3::identity() - (2.0 / nu) * ts1 * n1.transpose());
    let r2 = hn * (Mat3::identity() - (2.0 / nu) * ts2 * n2.transpose());
    let t1 = r1 * ts1;
    let t2 = r2 * ts2;
    vec![(r1, t1, -n1), (r1, -t1, n1), (r2, t2, -n2), (r2, -t2, n2)]
}

/// H → (cam1_to_cam2, 법선, 3D 점). 성공 점 수가 엄격히 많거나 같으면 오차 합이 작은 후보.
pub fn pose_from_homography(h: &Mat3, k1: &Mat3, k2: &Mat3, rays1: &[Vec3], rays2: &[Vec3]) -> Option<(Rigid3, Vec3, Vec<Vec3>)> {
    let cands = decompose_homography(h, k1, k2);
    let mut best: Option<(Mat3, Vec3, Vec3, Vec<Vec3>, f64)> = None;
    for (r, t, n) in cands {
        let mut pts = Vec::new();
        let mut err = 0.0;
        for (a, b) in rays1.iter().zip(rays2) {
            if let Some(x) = triangulate_midpoint(&r, &t, a, b) {
                let x2 = r * x + t;
                err += (1.0 - cos_between(a, &x)) + (1.0 - cos_between(b, &x2));
                pts.push(x);
            }
        }
        let better = match &best {
            None => true,
            Some(b) => pts.len() > b.3.len() || (pts.len() == b.3.len() && err < b.4),
        };
        if better {
            best = Some((r, t, n, pts, err));
        }
    }
    let (r, t, n, pts, _) = best?;
    Some((Rigid3::from_rotation_matrix(&r, t), n, pts))
}

fn cos_between(a: &Vec3, b: &Vec3) -> f64 {
    let d = a.norm() * b.norm();
    if d == 0.0 {
        0.0
    } else {
        a.dot(b) / d
    }
}

/// 삼각측량 각 중앙값(점 = 카메라1 좌표).
// 설계 결정: 짝수 개일 때의 중앙값 정의. 두 가운데 값의 평균을 쓴다.
pub fn median_triangulation_angle(pose: &Rigid3, pts: &[Vec3]) -> f64 {
    if pts.is_empty() {
        return 0.0;
    }
    let c1 = Vec3::zeros();
    let c2 = pose.center();
    let mut a: Vec<f64> = pts.iter().map(|x| triangulation_angle(&c1, &c2, x)).collect();
    a.sort_by(f64::total_cmp);
    let n = a.len();
    if n % 2 == 1 {
        a[n / 2]
    } else {
        0.5 * (a[n / 2 - 1] + a[n / 2])
    }
}

/// 인라이어 매칭의 단위 광선(역투영 실패 시 영벡터).
pub fn inlier_rays(cam1: &Camera, cam2: &Camera, kps1: &[Keypoint], kps2: &[Keypoint], tvg: &TwoViewGeometry) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut r1 = Vec::with_capacity(tvg.inlier_matches.len());
    let mut r2 = Vec::with_capacity(tvg.inlier_matches.len());
    for m in &tvg.inlier_matches {
        let p1 = kps1[m.idx1 as usize];
        let p2 = kps2[m.idx2 as usize];
        r1.push(cam1.img_to_ray(&Vec2::new(p1.x as f64, p1.y as f64)).unwrap_or_else(Vec3::zeros));
        r2.push(cam2.img_to_ray(&Vec2::new(p2.x as f64, p2.y as f64)).unwrap_or_else(Vec3::zeros));
    }
    (r1, r2)
}

/// 두 뷰 기하에서 상대 자세를 분해해 `cam1_to_cam2`, `tri_angle`, (PoP 이면) 구성을 채운다.
/// 대상: CALIBRATED/UNCALIBRATED/PLANAR/PANORAMIC/PLANAR_OR_PANORAMIC. 실패하면 false.
pub fn recover_two_view_pose(cam1: &Camera, cam2: &Camera, kps1: &[Keypoint], kps2: &[Keypoint], tvg: &mut TwoViewGeometry) -> bool {
    use TwoViewGeometryConfig as C;
    if tvg.inlier_matches.is_empty() {
        return false;
    }
    let (r1, r2) = inlier_rays(cam1, cam2, kps1, kps2, tvg);
    match tvg.config {
        C::Calibrated | C::Uncalibrated => {
            let e = if tvg.config == C::Calibrated {
                match tvg.e {
                    Some(e) => e,
                    None => return false,
                }
            } else {
                match tvg.f {
                    Some(f) => cam2.calibration_matrix().transpose() * f * cam1.calibration_matrix(),
                    None => return false,
                }
            };
            let Some((pose, pts)) = pose_from_essential(&e, &r1, &r2) else { return false };
            tvg.tri_angle = Some(median_triangulation_angle(&pose, &pts));
            tvg.cam1_to_cam2 = Some(pose);
            true
        }
        C::Planar | C::Panoramic | C::PlanarOrRotation => {
            let Some(h) = tvg.h else { return false };
            let Some((pose, _n, pts)) = pose_from_homography(&h, &cam1.calibration_matrix(), &cam2.calibration_matrix(), &r1, &r2) else {
                return false;
            };
            if pose.translation.norm_squared() < 1e-12 {
                if tvg.config == C::PlanarOrRotation {
                    tvg.config = C::Panoramic;
                }
                tvg.tri_angle = Some(0.0);
            } else {
                if tvg.config == C::PlanarOrRotation {
                    tvg.config = C::Planar;
                }
                if pts.is_empty() {
                    return false;
                }
                tvg.tri_angle = Some(median_triangulation_angle(&pose, &pts));
            }
            tvg.cam1_to_cam2 = Some(pose);
            true
        }
        _ => false,
    }
}

/// 전역 매퍼 준비용 일괄 분해: 행렬이 없으면 인라이어로 다시 맞춘 뒤 분해하고 t 를 단위화.
/// CALIBRATED → 8점 E(유효 광선 ≥ 8), UNCALIBRATED → 8점 F, PLANAR 계열 → N점 DLT.
/// UNDEFINED/DEGENERATE/WATERMARK/MULTIPLE 은 false.
pub fn refit_and_estimate_relative_pose(
    cam1: &Camera,
    cam2: &Camera,
    kps1: &[Keypoint],
    kps2: &[Keypoint],
    tvg: &mut TwoViewGeometry,
) -> bool {
    use TwoViewGeometryConfig as C;
    match tvg.config {
        C::Calibrated => {
            if tvg.e.is_none() {
                let (r1, r2) = inlier_rays(cam1, cam2, kps1, kps2, tvg);
                let (a, b): (Vec<Vec3>, Vec<Vec3>) =
                    r1.into_iter().zip(r2).filter(|(a, b)| a.norm_squared() > 0.0 && b.norm_squared() > 0.0).unzip();
                if a.len() < 8 {
                    return false;
                }
                tvg.e = essential_eight_point(&a, &b);
            }
        }
        C::Uncalibrated => {
            if tvg.f.is_none() {
                let (p1, p2) = inlier_points(kps1, kps2, tvg);
                tvg.f = fundamental_eight_point(&p1, &p2);
            }
        }
        C::Planar | C::Panoramic | C::PlanarOrRotation => {
            if tvg.h.is_none() {
                let (p1, p2) = inlier_points(kps1, kps2, tvg);
                tvg.h = homography_dlt(&p1, &p2, false);
            }
        }
        _ => return false,
    }
    if !recover_two_view_pose(cam1, cam2, kps1, kps2, tvg) {
        return false;
    }
    if let Some(p) = tvg.cam1_to_cam2.as_mut() {
        let n = p.translation.norm();
        if n > 1e-12 {
            p.translation /= n;
        }
    }
    true
}

fn inlier_points(kps1: &[Keypoint], kps2: &[Keypoint], tvg: &TwoViewGeometry) -> (Vec<Vec2>, Vec<Vec2>) {
    tvg.inlier_matches
        .iter()
        .map(|m| {
            let a = kps1[m.idx1 as usize];
            let b = kps2[m.idx2 as usize];
            (Vec2::new(a.x as f64, a.y as f64), Vec2::new(b.x as f64, b.y as f64))
        })
        .unzip()
}
