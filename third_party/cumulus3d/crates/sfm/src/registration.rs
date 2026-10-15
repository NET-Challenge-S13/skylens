/*
 * registration.rs
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

//! 기존 모델에 새 영상 등록(image_registrator).

use crate::absolute_pose::{solve_abs_pose, AbsolutePoseOptions};
use cumulus3d_ba::Loss;
use cumulus3d_core::{
    Camera, FeatureStore, Image, ImageId, MatchGraph, Point2DIdx, Point3DId, Reconstruction, Result, TrackEntry, Vec2, Vec3,
};
use rayon::prelude::*;
use std::collections::BTreeMap;

/// 등록 순서.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegistrationOrder {
    /// 영상 id 오름차순(결정적 순서).
    /// id 는 도착(위치→카메라) 순으로 발급되므로 도착 순서와도 같다.
    ImageId,
    /// 개선: 가시성 피라미드 점수 내림차순으로 매번 다시 골라 등록.
    VisibilityScore,
}

/// image_registrator 옵션.
#[derive(Clone, Debug)]
pub struct RegistrationOptions {
    /// 등록에 필요한 최소 가시 점·대응·인라이어 수.
    pub abs_pose_min_num_inliers: usize,
    /// 절대 자세 RANSAC 옵션.
    pub ransac: AbsolutePoseOptions,
    /// 비정상 카메라 판정: 초점 비율 하한.
    pub min_focal_length_ratio: f64,
    /// 비정상 카메라 판정: 초점 비율 상한.
    pub max_focal_length_ratio: f64,
    /// 비정상 카메라 판정: 추가 파라미터 절대값 상한.
    pub max_extra_param: f64,
    /// RANSAC 뒤 비선형 자세 정제(cumulus3d-ba `refine_abs_pose`) 실행 여부.
    pub refine_pose: bool,
    /// 자세 정제 손실 함수.
    pub refine_loss: Loss,
    /// 자세 정제 최대 반복.
    pub refine_max_num_iterations: usize,
    /// 등록 순서.
    pub order: RegistrationOrder,
    /// 끝나고 등록 실패 영상(프레임·쓰이지 않는 카메라)을 모델에서 삭제.
    pub remove_unregistered: bool,
}

impl Default for RegistrationOptions {
    fn default() -> Self {
        Self {
            abs_pose_min_num_inliers: 30,
            ransac: AbsolutePoseOptions::default(),
            min_focal_length_ratio: 0.1,
            max_focal_length_ratio: 10.0,
            max_extra_param: 1.0,
            refine_pose: true,
            refine_loss: Loss::Cauchy(1.0),
            refine_max_num_iterations: 100,
            order: RegistrationOrder::ImageId,
            remove_unregistered: true,
        }
    }
}

/// 영상 하나 등록 결과.
#[derive(Clone, Debug, PartialEq)]
pub enum RegisterOutcome {
    /// 등록 성공.
    Registered {
        /// 2D–3D 대응 수.
        num_correspondences: usize,
        /// RANSAC 인라이어 수.
        num_inliers: usize,
    },
    /// 가시 3D 점이 부족(가시 점 수).
    TooFewVisiblePoints(usize),
    /// 2D–3D 대응이 부족(대응 수).
    TooFewCorrespondences(usize),
    /// 절대 자세 RANSAC 실패.
    RansacFailed,
    /// 인라이어 부족(인라이어 수).
    TooFewInliers(usize),
    /// 자세 정제 실패.
    RefinementFailed,
    /// 이미 등록된 영상.
    AlreadyRegistered,
}

impl RegisterOutcome {
    /// 등록에 성공했는지.
    pub fn is_registered(&self) -> bool {
        matches!(self, RegisterOutcome::Registered { .. })
    }
}

/// 전체 결과.
#[derive(Clone, Debug, Default)]
pub struct RegistrationReport {
    /// 시도 순서대로 (영상, 결과).
    pub attempts: Vec<(ImageId, RegisterOutcome)>,
    /// 모델에 새로 추가한 영상 수.
    pub num_added_images: usize,
}

impl RegistrationReport {
    /// 등록에 성공한 영상 id(시도 순서).
    pub fn registered(&self) -> Vec<ImageId> {
        self.attempts.iter().filter(|(_, o)| o.is_registered()).map(|(i, _)| *i).collect()
    }
}

/// 저장소의 카메라·영상 중 대응 그래프에 있고 모델에 없는 것을 자세 없이 추가.
pub fn add_missing_images(rec: &mut Reconstruction, store: &FeatureStore, graph: &MatchGraph) -> Result<Vec<ImageId>> {
    let mut ids = graph.image_ids();
    ids.sort();
    let mut added = Vec::new();
    for id in ids {
        if let Some(im) = rec.image(id) {
            // 이름·특징 수 일치 검사.
            let ok = store.image(id).is_some_and(|s| s.name == im.name) && store.num_keypoints(id) == im.num_points2d();
            if !ok {
                return Err(cumulus3d_core::Error::Invariant(format!("영상 {id}: 모델과 저장소의 이름/특징 수 불일치")));
            }
            continue;
        }
        let Some(si) = store.image(id) else { continue };
        if rec.camera(si.camera_id).is_none() {
            let cam = store.camera(si.camera_id).ok_or_else(|| cumulus3d_core::Error::NotFound(format!("카메라 {}", si.camera_id)))?;
            rec.add_camera_own_rig(cam)?;
        }
        let kps = store.keypoints(id).map(|k| k.iter().map(|p| Vec2::new(p.x as f64, p.y as f64)).collect::<Vec<_>>());
        rec.add_image_own_frame(Image::new(id, si.name.clone(), si.camera_id, kps.unwrap_or_default()), None)?;
        added.push(id);
    }
    Ok(added)
}

fn is_bogus(cam: &Camera, o: &RegistrationOptions) -> bool {
    cam.has_implausible_params(o.min_focal_length_ratio, o.max_focal_length_ratio, o.max_extra_param)
}

/// "보이는 3D 점 수": 대응 이웃 중 하나라도 3D 점을 가진 2D 점 수.
pub fn num_visible_points3d(rec: &Reconstruction, graph: &MatchGraph, image_id: ImageId) -> usize {
    let Some(im) = rec.image(image_id) else { return 0 };
    (0..im.num_points2d() as Point2DIdx)
        .filter(|&i| {
            graph
                .find_correspondences(image_id, i)
                .iter()
                .any(|c| rec.image(c.image_id).and_then(|o| o.points2d().get(c.point2d_idx as usize)).is_some_and(|p| p.has_point3d()))
        })
        .count()
}

/// 가시성 피라미드 점수(6단계).
pub fn visibility_score(rec: &Reconstruction, graph: &MatchGraph, image_id: ImageId) -> usize {
    let Some(im) = rec.image(image_id) else { return 0 };
    let Some(cam) = rec.camera(im.camera_id) else { return 0 };
    let (w, h) = (cam.width.max(1) as f64, cam.height.max(1) as f64);
    let mut cells: Vec<std::collections::HashSet<(usize, usize)>> = vec![Default::default(); 6];
    for (i, p) in im.points2d().iter().enumerate() {
        let visible = graph
            .find_correspondences(image_id, i as Point2DIdx)
            .iter()
            .any(|c| rec.image(c.image_id).and_then(|o| o.points2d().get(c.point2d_idx as usize)).is_some_and(|q| q.has_point3d()));
        if !visible {
            continue;
        }
        let mut cx = ((64.0 * p.xy.x / w).floor().max(0.0) as usize).min(63);
        let mut cy = ((64.0 * p.xy.y / h).floor().max(0.0) as usize).min(63);
        for level in (0..6).rev() {
            cells[level].insert((cx, cy));
            cx /= 2;
            cy /= 2;
        }
    }
    cells.iter().enumerate().map(|(l, s)| s.len() * (1usize << (2 * (l + 1)))).sum()
}

/// 2D-3D 대응: (2D 점, 픽셀, 3D 점 id, 좌표).
pub fn collect_2d3d_correspondences(
    rec: &Reconstruction,
    graph: &MatchGraph,
    image_id: ImageId,
    opts: &RegistrationOptions,
) -> Vec<(Point2DIdx, Vec2, Point3DId, Vec3)> {
    let Some(im) = rec.image(image_id) else { return Vec::new() };
    let bogus: BTreeMap<u32, bool> = rec.cameras().iter().map(|(id, c)| (*id, is_bogus(c, opts))).collect();
    let per_point: Vec<Vec<(Point2DIdx, Vec2, Point3DId, Vec3)>> = (0..im.num_points2d() as Point2DIdx)
        .into_par_iter()
        .map(|i| {
            let mut out: Vec<(Point2DIdx, Vec2, Point3DId, Vec3)> = Vec::new();
            for c in graph.find_correspondences(image_id, i) {
                let Some(o) = rec.image(c.image_id) else { continue };
                if !rec.has_pose(c.image_id) {
                    continue;
                }
                let Some(q) = o.points2d().get(c.point2d_idx as usize) else { continue };
                if !q.has_point3d() || out.iter().any(|x| x.2 == q.point3d_id) {
                    continue;
                }
                if bogus.get(&o.camera_id).copied().unwrap_or(true) {
                    continue;
                }
                let Some(p) = rec.point3d(q.point3d_id) else { continue };
                out.push((i, im.point2d(i).xy, q.point3d_id, p.xyz));
            }
            out
        })
        .collect();
    per_point.into_iter().flatten().collect()
}

/// 영상 하나 등록(중심 카메라·초점 고정 경로).
pub fn register_image(
    rec: &mut Reconstruction,
    graph: &MatchGraph,
    image_id: ImageId,
    opts: &RegistrationOptions,
) -> Result<RegisterOutcome> {
    if rec.is_image_registered(image_id) {
        return Ok(RegisterOutcome::AlreadyRegistered);
    }
    let visible = num_visible_points3d(rec, graph, image_id);
    if visible < opts.abs_pose_min_num_inliers {
        return Ok(RegisterOutcome::TooFewVisiblePoints(visible));
    }
    let corrs = collect_2d3d_correspondences(rec, graph, image_id, opts);
    if corrs.len() < opts.abs_pose_min_num_inliers {
        return Ok(RegisterOutcome::TooFewCorrespondences(corrs.len()));
    }
    let im = rec.image(image_id).expect("존재");
    // 설계 결정: 등록 영상 0 인 카메라·엉터리 카메라의 초점 추정(P4Pf)·내부 정제 경로는 구현하지 않고
    // 항상 초점 고정 경로를 쓴다(이 파이프라인의 정상 경우).
    let camera = rec.camera(im.camera_id).expect("존재").clone();
    let p2: Vec<Vec2> = corrs.iter().map(|c| c.1).collect();
    let p3: Vec<Vec3> = corrs.iter().map(|c| c.3).collect();
    let mut ransac = opts.ransac.clone();
    // 영상마다 다른 결정적 시드.
    ransac.random_seed = ransac.random_seed.map(|s| s ^ (image_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let Some(res) = solve_abs_pose(&camera, &p2, &p3, &ransac) else {
        return Ok(RegisterOutcome::RansacFailed);
    };
    if res.num_inliers < opts.abs_pose_min_num_inliers {
        return Ok(RegisterOutcome::TooFewInliers(res.num_inliers));
    }
    let mut pose = res.world_to_cam;
    if opts.refine_pose {
        let s =
            cumulus3d_ba::refine_abs_pose(&camera, &p2, &p3, &res.inlier_mask, &mut pose, opts.refine_loss, opts.refine_max_num_iterations);
        match s {
            Ok(sum) if sum.is_usable() => {}
            _ => return Ok(RegisterOutcome::RefinementFailed),
        }
    }
    rec.set_world_to_cam(image_id, pose)?;
    rec.register_image(image_id)?;
    // 트랙 연장: 첫 번째로 적용된 인라이어만(이미 3D 점이 있으면 무시).
    for (k, c) in corrs.iter().enumerate() {
        if !res.inlier_mask[k] {
            continue;
        }
        let has = rec.image(image_id).expect("존재").point2d(c.0).has_point3d();
        if !has && rec.exists_point3d(c.2) {
            rec.add_observation(c.2, TrackEntry::new(image_id, c.0))?;
        }
    }
    Ok(RegisterOutcome::Registered { num_correspondences: corrs.len(), num_inliers: res.num_inliers })
}

/// image_registrator 전체: 누락 영상 추가 → 자세 없는 영상마다 한 번 시도 → 실패 영상 삭제.
pub fn register_images(
    rec: &mut Reconstruction,
    store: &FeatureStore,
    graph: &MatchGraph,
    opts: &RegistrationOptions,
) -> Result<RegistrationReport> {
    let added = add_missing_images(rec, store, graph)?;
    let mut report = RegistrationReport { num_added_images: added.len(), ..Default::default() };
    let mut pending: Vec<ImageId> = rec.image_ids().filter(|i| !rec.has_pose(*i)).collect();
    pending.sort();
    match opts.order {
        RegistrationOrder::ImageId => {
            for id in pending {
                let o = register_image(rec, graph, id, opts)?;
                report.attempts.push((id, o));
            }
        }
        RegistrationOrder::VisibilityScore => {
            while !pending.is_empty() {
                let rec_ref: &Reconstruction = rec;
                let scores: Vec<usize> = pending.par_iter().map(|i| visibility_score(rec_ref, graph, *i)).collect();
                let mut best = 0;
                for k in 1..pending.len() {
                    if scores[k] > scores[best] {
                        best = k;
                    }
                }
                let id = pending.remove(best);
                let o = register_image(rec, graph, id, opts)?;
                report.attempts.push((id, o));
            }
        }
    }
    if opts.remove_unregistered {
        rec.remove_unregistered();
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cumulus3d_core::{FeatureMatch, Rigid3, TwoViewGeometry, TwoViewGeometryConfig};
    use rand::{RngExt, SeedableRng};

    /// 등록 영상 1, 2 와 3D 점 n 개, 새 영상 3 이 그 점들을 무잡음으로 본다.
    fn setup(n: usize) -> (Reconstruction, MatchGraph, Rigid3) {
        let mut rng = rand_pcg::Pcg64::seed_from_u64(5);
        let cam = crate::synthetic::opencv_camera(1);
        let mut rec = Reconstruction::new();
        rec.add_camera_own_rig(cam.clone()).unwrap();
        let p1 = Rigid3::new(cumulus3d_core::Quat::IDENTITY, Vec3::new(0.0, 0.0, 10.0));
        let p2 = Rigid3::new(cumulus3d_core::Quat::IDENTITY, Vec3::new(-1.0, 0.0, 10.0));
        let p3 = Rigid3::new(cumulus3d_core::Quat::from_axis_angle(&Vec3::y(), 0.05), Vec3::new(1.0, 0.2, 10.0));
        let pts: Vec<Vec3> =
            (0..n).map(|_| Vec3::new(rng.random_range(-3.0..3.0), rng.random_range(-2.0..2.0), rng.random_range(-1.0..1.0))).collect();
        let proj = |p: &Rigid3| -> Vec<Vec2> { pts.iter().map(|x| cam.cam_to_img(&(*p * *x)).unwrap()).collect() };
        for (id, p) in [(1u32, p1), (2, p2)] {
            rec.add_image_own_frame(Image::new(id, format!("{id}"), 1, proj(&p)), Some(p)).unwrap();
            rec.register_image(id).unwrap();
        }
        rec.add_image_own_frame(Image::new(3, "3", 1, proj(&p3)), None).unwrap();
        for k in 0..n as u32 {
            rec.add_point3d(pts[k as usize], vec![TrackEntry::new(1, k), TrackEntry::new(2, k)], [0; 3]).unwrap();
        }
        let mut g = MatchGraph::new();
        for id in 1..=3 {
            g.add_image(id, n);
        }
        let tvg = TwoViewGeometry {
            config: TwoViewGeometryConfig::Calibrated,
            e: None,
            f: None,
            h: None,
            cam1_to_cam2: None,
            inlier_matches: (0..n as u32).map(|k| FeatureMatch::new(k, k)).collect(),
            tri_angle: None,
        };
        g.insert_two_view(1, 3, &tvg);
        (rec, g, p3)
    }

    #[test]
    fn min_visible_boundary() {
        let opts = RegistrationOptions { refine_pose: false, ..Default::default() };
        let (mut rec, g, _) = setup(29);
        assert_eq!(register_image(&mut rec, &g, 3, &opts).unwrap(), RegisterOutcome::TooFewVisiblePoints(29));
        let (mut rec, g, truth) = setup(30);
        let o = register_image(&mut rec, &g, 3, &opts).unwrap();
        assert!(o.is_registered(), "{o:?}");
        let est = rec.world_to_cam(3).unwrap();
        assert!(est.rotation.angular_distance(&truth.rotation) < 1e-6);
        // 트랙 연장: 모든 점이 영상 3 관측을 얻음.
        assert_eq!(rec.image(3).unwrap().num_points3d(), 30);
        rec.check_invariants().unwrap();
    }

    #[test]
    fn min_inlier_boundary() {
        // 30개 대응 중 2개를 크게 틀어 인라이어 28 → 실패.
        let opts = RegistrationOptions { refine_pose: false, ..Default::default() };
        let (rec0, g, _) = setup(30);
        let mut rec = Reconstruction::new();
        rec.add_camera_own_rig(rec0.camera(1).unwrap().clone()).unwrap();
        for id in 1..=3u32 {
            let im = rec0.image(id).unwrap();
            let mut pts: Vec<Vec2> = im.points2d().iter().map(|p| p.xy).collect();
            if id == 3 {
                pts[0].x += 300.0;
                pts[1].y += 300.0;
            }
            rec.add_image_own_frame(Image::new(id, im.name.clone(), 1, pts), rec0.world_to_cam(id)).unwrap();
            if id < 3 {
                rec.register_image(id).unwrap();
            }
        }
        for (_, p) in rec0.points3d() {
            rec.add_point3d(p.xyz, p.track.clone(), [0; 3]).unwrap();
        }
        assert_eq!(register_image(&mut rec, &g, 3, &opts).unwrap(), RegisterOutcome::TooFewInliers(28));
    }
}
