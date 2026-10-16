/*
 * model_aligner.rs
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

//! GPS ENU 정렬: 등록 영상 투영 중심을 GPS 기준점의 ENU 좌표에 강건 Sim3 로 맞춘다.
//!
//! 실패하면 `Err` 를 돌려주고 모델을 건드리지 않는다(빈 출력으로 조용히 진행하지 않음).

use crate::error::{AlignError, Result};
use crate::geodesy::EnuFrame;
use crate::umeyama::{estimate_sim3_ransac, median, RankCheck};
use cumulus3d_core::io::{read_gps_file, GpsRecord};
use cumulus3d_core::ransac::RansacParams;
use cumulus3d_core::{ImageId, Reconstruction, Sim3, Vec3};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// ENU 원점 선택.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum EnuOrigin {
    /// 기본 동작: GPS 목록의 첫 줄.
    #[default]
    FirstRecord,
    /// 세션 전체 고정 원점(위도°, 경도°, 고도 m). 구역마다 원점이 바뀌는 문제를 피한다.
    Explicit {
        /// 위도(°).
        lat: f64,
        /// 경도(°).
        lon: f64,
        /// 타원체고(m).
        alt: f64,
    },
}

/// model_aligner 옵션. 기본값 = 스크립트 호출(`--alignment_max_error 3`, min_common_images 3).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelAlignerOptions {
    /// 인라이어 거리 임계 (m) (변환된 카메라 중심 ↔ GPS ENU 거리 ≤ 3 m).
    pub max_error: f64,
    /// 정렬에 필요한 최소 공통 영상 수.
    pub min_common_images: usize,
    /// ENU 원점 선택.
    pub origin: EnuOrigin,
    /// 신뢰도·반복 수·시드. `max_error` 필드는 무시되고 위 값이 쓰인다.
    pub ransac: RansacParams,
    /// 최소 표본 퇴화 판정 방식.
    pub rank_check: RankCheck,
}

impl Default for ModelAlignerOptions {
    fn default() -> Self {
        Self {
            max_error: 3.0,
            min_common_images: 3,
            origin: EnuOrigin::FirstRecord,
            ransac: RansacParams::default(),
            rank_check: RankCheck::Uncentered,
        }
    }
}

/// 정렬 결과.
#[derive(Clone, Debug)]
pub struct GpsAlignment {
    /// model_to_enu.
    pub sim3: Sim3,
    /// 사용한 ENU 좌표계(원점 포함).
    pub enu: EnuFrame,
    /// 공통 영상(파일 순서): (영상 id, 이름, ENU 위치).
    pub common: Vec<(ImageId, String, Vec3)>,
    /// `common` 과 같은 순서의 인라이어 여부.
    pub inlier_mask: Vec<bool>,
    /// 인라이어 수.
    pub num_inliers: usize,
    /// RANSAC 시행 수.
    pub num_trials: usize,
    /// 정렬 후 ‖C' − g‖ (파일의 각 이름 중 모델에 있고 자세가 있는 영상, 필터링 없음).
    pub errors: Vec<(String, f64)>,
    /// 정렬 오차 평균 (m).
    pub mean_error: f64,
    /// 정렬 오차 중앙값 (m).
    pub median_error: f64,
}

/// GPS 목록 → (이름, ENU). 원점이 `FirstRecord` 인데 목록이 비면 오류.
pub fn gps_to_enu(records: &[GpsRecord], origin: EnuOrigin) -> Result<(EnuFrame, Vec<(String, Vec3)>)> {
    let frame = match origin {
        EnuOrigin::FirstRecord => {
            let r = records.first().ok_or(AlignError::TooFewReferences { found: 0, required: 1 })?;
            EnuFrame::new(r.lat, r.lon, r.alt)
        }
        EnuOrigin::Explicit { lat, lon, alt } => EnuFrame::new(lat, lon, alt),
    };
    let v = records.iter().map(|r| (r.name.clone(), frame.lla_to_enu(r.lat, r.lon, r.alt))).collect();
    Ok((frame, v))
}

/// 정렬 Sim3 만 추정(모델 불변).
pub fn estimate_gps_alignment(rec: &Reconstruction, gps: &[GpsRecord], opts: &ModelAlignerOptions) -> Result<GpsAlignment> {
    if opts.max_error.is_nan() || opts.max_error <= 0.0 {
        return Err(AlignError::BadMaxError(opts.max_error));
    }
    // 2. 기준 위치 < 3 → 실패.
    if gps.len() < 3 {
        return Err(AlignError::TooFewReferences { found: gps.len(), required: 3 });
    }
    let (enu, refs) = gps_to_enu(gps, opts.origin)?;

    // 4. 공통 영상(파일 순서, 이름 일치, 자세 있음, 중복 제외).
    let by_name: HashMap<&str, ImageId> = rec.images().map(|im| (im.name.as_str(), im.image_id)).collect();
    let mut seen = HashSet::new();
    let mut common = Vec::new();
    let (mut src, mut dst) = (Vec::new(), Vec::new());
    for (name, g) in &refs {
        let Some(&id) = by_name.get(name.as_str()) else { continue };
        let Some(c) = rec.projection_center(id) else { continue };
        if !seen.insert(id) {
            continue;
        }
        common.push((id, name.clone(), *g));
        src.push(c);
        dst.push(*g);
    }
    // 5.
    let required = opts.min_common_images.max(3);
    if common.len() < required {
        return Err(AlignError::TooFewCommonImages { found: common.len(), required });
    }
    // 6.
    let ropts = RansacParams { max_error: opts.max_error, ..opts.ransac.clone() };
    let rep = estimate_sim3_ransac(&src, &dst, &ropts, opts.rank_check);
    let num_inliers = rep.support.num_inliers;
    let sim3 = match (rep.success, rep.model) {
        (true, Some(m)) if num_inliers >= 3 => m,
        _ => return Err(AlignError::RansacFailed { inliers: num_inliers }),
    };

    // 8. 정렬 오차(파일의 모든 이름, 필터 없음; 중복 줄도 각각 센다).
    let mut errors = Vec::new();
    for (name, g) in &refs {
        let Some(&id) = by_name.get(name.as_str()) else { continue };
        let Some(c) = rec.projection_center(id) else { continue };
        errors.push((name.clone(), (sim3.transform_point(&c) - g).norm()));
    }
    let mut ev: Vec<f64> = errors.iter().map(|e| e.1).collect();
    let mean_error = if ev.is_empty() { f64::NAN } else { ev.iter().sum::<f64>() / ev.len() as f64 };
    // 설계 결정: 중앙값의 짝수 개 처리. 가운데 둘의 평균을 쓴다.
    let median_error = median(&mut ev);

    Ok(GpsAlignment {
        sim3,
        enu,
        common,
        inlier_mask: rep.inlier_mask,
        num_inliers,
        num_trials: rep.num_trials,
        errors,
        mean_error,
        median_error,
    })
}

/// 추정 후 성공하면 모델(자세·점)에 적용. 실패하면 모델을 바꾸지 않고 `Err`.
pub fn align_to_gps(rec: &mut Reconstruction, gps: &[GpsRecord], opts: &ModelAlignerOptions) -> Result<GpsAlignment> {
    let a = estimate_gps_alignment(rec, gps, opts)?;
    rec.transform(&a.sim3);
    Ok(a)
}

/// GPS 파일 경로판.
pub fn align_to_gps_file(rec: &mut Reconstruction, path: impl AsRef<Path>, opts: &ModelAlignerOptions) -> Result<GpsAlignment> {
    let gps = read_gps_file(path)?;
    align_to_gps(rec, &gps, opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geodesy::EnuFrame;
    use crate::umeyama::tests::{gauss, rand_vec, random_sim3};
    use cumulus3d_core::{Camera, CameraModelKind, Image, Quat, Rigid3, TrackEntry};
    use rand::RngExt;

    /// ENU 궤적(드론 3대 × 위치)을 만들고, 임의 Sim3 로 "모델 좌표"로 옮긴 재구성 + GPS 목록.
    fn scene(num_pos: usize, seed: u64) -> (Reconstruction, Vec<GpsRecord>, Vec<Vec3>, Sim3) {
        let mut rng = cumulus3d_core::ransac::make_rng(Some(seed));
        let enu = EnuFrame::new(37.5, 127.0, 50.0);
        let enu_to_model = random_sim3(&mut rng);
        let mut rec = Reconstruction::new();
        let cam = Camera::new(1, CameraModelKind::Pinhole, 1000, 800, vec![800.0, 800.0, 500.0, 400.0]).unwrap();
        rec.add_camera_own_rig(cam).unwrap();
        let mut gps = Vec::new();
        let mut centers = Vec::new();
        let mut id = 1;
        for p in 0..num_pos {
            for (k, dx) in [-20.0, 0.0, 20.0].iter().enumerate() {
                let c_enu = Vec3::new(p as f64 * 8.0 + dx, p as f64 * 2.0 + dx * 0.5, 30.0 + 3.0 * k as f64);
                let c_model = enu_to_model.transform_point(&c_enu);
                // 아래(−U)를 보는 카메라 + 작은 무작위 회전.
                let down = Quat::from_axis_angle(&Vec3::x(), std::f64::consts::PI);
                let small = Quat::from_axis_angle(&rand_vec(&mut rng, 1.0).normalize(), rng.random_range(-0.2..0.2));
                // 모델 좌표계 회전도 반영해야 아래를 본다: R_cam = small·down·R_mᵀ.
                let rot = small * down * enu_to_model.rotation.conjugate();
                let r = rot.to_rotation_matrix();
                let pose = Rigid3::new(rot, -(r * c_model));
                let pts: Vec<_> = (0..20).map(|i| cumulus3d_core::Vec2::new(10.0 + i as f64, 20.0)).collect();
                let name = format!("cam{k}/img_{p:04}.jpg");
                rec.add_image_own_frame(Image::new(id, &name, 1, pts), Some(pose)).unwrap();
                rec.register_image(id).unwrap();
                let (lat, lon, alt) = enu.enu_to_lla(&c_enu);
                gps.push(GpsRecord { name, lat, lon, alt });
                centers.push(c_enu);
                id += 1;
            }
        }
        // 관측 몇 개(재투영 불변 확인용).
        for i in 0..10u32 {
            let x = enu_to_model.transform_point(&Vec3::new(10.0 + i as f64, 5.0, 0.0));
            let pid = rec.add_point3d(x, vec![], [0, 0, 0]).unwrap();
            for img in 1..=6u32 {
                let pose = rec.world_to_cam(img).unwrap();
                let cam = rec.camera(1).unwrap();
                if let Some(_xy) = cam.cam_to_img(&pose.transform_point(&x)) {
                    rec.add_observation(pid, TrackEntry::new(img, i)).unwrap();
                }
            }
        }
        (rec, gps, centers, enu_to_model)
    }

    /// 모든 관측의 투영 픽셀.
    fn projections(rec: &Reconstruction) -> Vec<cumulus3d_core::Vec2> {
        let mut out = Vec::new();
        for id in rec.point3d_ids() {
            let p = rec.point3d(id).unwrap();
            for e in &p.track {
                let pose = rec.world_to_cam(e.image_id).unwrap();
                let cam = rec.camera(1).unwrap();
                out.push(cam.cam_to_img(&pose.transform_point(&p.xyz)).unwrap());
            }
        }
        out
    }

    #[test]
    fn aligns_synthetic_model() {
        let (mut rec, gps, centers, enu_to_model) = scene(10, 11);
        let before = projections(&rec);
        assert!(before.len() > 10);
        let opts = ModelAlignerOptions { ransac: RansacParams { random_seed: Some(1), ..Default::default() }, ..Default::default() };
        let a = align_to_gps(&mut rec, &gps, &opts).unwrap();
        assert_eq!(a.num_inliers, 30);
        // 원점 = 첫 줄 위치이므로, 첫 영상 중심은 ENU (0,0,0) 이고 나머지는 원래 ENU 에서 c0 만큼 이동.
        let c0 = centers[0];
        for (i, c) in centers.iter().enumerate() {
            let got = rec.projection_center(i as u32 + 1).unwrap();
            assert!((got - (c - c0)).norm() < 1e-3, "{i}: {got:?} vs {:?}", c - c0);
        }
        assert!(a.median_error < 1e-3 && a.mean_error < 1e-3);
        let expect = enu_to_model.inverse();
        assert!((a.sim3.scale - expect.scale).abs() / expect.scale < 1e-6);
        // Sim3 적용 후 재투영 오차 불변.
        let after = projections(&rec);
        assert_eq!(before.len(), after.len());
        for (b, a) in before.iter().zip(&after) {
            assert!((b - a).norm() < 1e-8, "{b:?} vs {a:?}");
        }
        // C' = s R C + t.
        let c_before = enu_to_model.transform_point(&centers[4]);
        assert!((a.sim3.transform_point(&c_before) - rec.projection_center(5).unwrap()).norm() < 1e-9);
    }

    #[test]
    fn explicit_origin_and_order_rule() {
        let (rec, mut gps, centers, _) = scene(6, 12);
        let opts = ModelAlignerOptions { ransac: RansacParams { random_seed: Some(2), ..Default::default() }, ..Default::default() };
        // 줄 순서를 바꾸면 원점이 바뀐다.
        gps.swap(0, 5);
        let mut r1 = rec.clone();
        align_to_gps(&mut r1, &gps, &opts).unwrap();
        assert!(r1.projection_center(6).unwrap().norm() < 1e-3);
        // 명시 원점은 줄 순서와 무관.
        let o = &gps[3];
        let opts2 = ModelAlignerOptions { origin: EnuOrigin::Explicit { lat: o.lat, lon: o.lon, alt: o.alt }, ..opts.clone() };
        let mut r2 = rec.clone();
        align_to_gps(&mut r2, &gps, &opts2).unwrap();
        let id_of_o = r2.images().find(|im| im.name == o.name).unwrap().image_id;
        assert!(r2.projection_center(id_of_o).unwrap().norm() < 1e-3);
        let _ = centers;
    }

    #[test]
    fn robust_to_gps_outliers() {
        // 50 카메라 중 30% 를 20 m 이상 오염, 나머지 σ=1 m 잡음, 임계 3 m.
        let (rec, mut gps, centers, _) = scene(17, 13);
        let mut rng = cumulus3d_core::ransac::make_rng(Some(5));
        let enu = EnuFrame::new(gps[0].lat, gps[0].lon, gps[0].alt);
        let c0 = centers[0];
        let n = 50;
        let gps_trunc: Vec<GpsRecord> = gps.drain(..n).collect();
        let mut out = vec![false; n];
        let mut noisy = Vec::new();
        for (i, g) in gps_trunc.iter().enumerate() {
            let truth = centers[i] - c0;
            let e = if i % 10 < 3 && i != 0 {
                out[i] = true;
                truth + rand_vec(&mut rng, 1.0).normalize() * rng.random_range(20.0..60.0)
            } else if i == 0 {
                truth
            } else {
                truth + Vec3::new(gauss(&mut rng), gauss(&mut rng), gauss(&mut rng))
            };
            let (lat, lon, alt) = enu.enu_to_lla(&e);
            noisy.push(GpsRecord { name: g.name.clone(), lat, lon, alt });
        }
        let opts = ModelAlignerOptions { ransac: RansacParams { random_seed: Some(3), ..Default::default() }, ..Default::default() };
        let mut r = rec.clone();
        let a = align_to_gps(&mut r, &noisy, &opts).unwrap();
        for (i, (id, _, _)) in a.common.iter().enumerate() {
            let k = (*id - 1) as usize;
            if out[k] {
                assert!(!a.inlier_mask[i], "오염 {k} 이 인라이어");
            }
        }
        let mut errs: Vec<f64> =
            (0..n).filter(|k| !out[*k]).map(|k| (r.projection_center(k as u32 + 1).unwrap() - (centers[k] - c0)).norm()).collect();
        let med = median(&mut errs);
        assert!(med < 1.5, "median {med}");
    }

    #[test]
    fn failure_cases_return_err_and_leave_model() {
        let (mut rec, gps, _, _) = scene(4, 14);
        let orig = rec.projection_center(1).unwrap();
        // 공통 영상 2개.
        let mut few = gps[..2].to_vec();
        few.push(GpsRecord { name: "nope.jpg".into(), ..gps[2].clone() });
        assert!(matches!(align_to_gps(&mut rec, &few, &Default::default()), Err(AlignError::TooFewCommonImages { found: 2, .. })));
        assert!(matches!(align_to_gps(&mut rec, &gps[..2], &Default::default()), Err(AlignError::TooFewReferences { .. })));
        let bad = ModelAlignerOptions { max_error: 0.0, ..Default::default() };
        assert!(matches!(align_to_gps(&mut rec, &gps, &bad), Err(AlignError::BadMaxError(_))));
        assert_eq!(rec.projection_center(1).unwrap(), orig);
    }
}
