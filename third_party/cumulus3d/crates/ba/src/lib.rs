/*
 * lib.rs
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

//! 번들 조정.
//!
//! 이 파일의 공개 시그니처는 sfm 크레이트와의 계약이다. 구현 중 바꾸지 말 것(필드 추가는 가능).
//!
//! 구성: 신뢰 영역 LM(`tr`), Schur 보수 조립·역대입(`problem`), 축소 계통 풀이기(`linsolve`),
//! 단일 자세 정제(`abspose`), 견고 손실(`loss`).
//!
//! # 사용 예
//!
//! ```
//! use cumulus3d_ba::{refine_abs_pose, Loss};
//! use cumulus3d_core::{Camera, CameraModelKind, Quat, Rigid3, Vec2, Vec3};
//!
//! // 합성 장면: 참 자세로 3D 점을 투영해 2D 관측을 만든다.
//! let cam = Camera::from_focal(CameraModelKind::Pinhole, 1000.0, 1920, 1080);
//! let truth = Rigid3::new(Quat::from_axis_angle(&Vec3::y(), 0.1), Vec3::new(0.2, -0.1, 6.0));
//! let pts3d: Vec<Vec3> = (0..60)
//!     .map(|i| {
//!         let a = i as f64;
//!         Vec3::new((a * 0.37).sin() * 3.0, (a * 0.71).cos() * 2.0, (a * 0.13).sin())
//!     })
//!     .collect();
//! let pts2d: Vec<Vec2> = pts3d.iter().map(|p| cam.cam_to_img(&truth.transform_point(p)).unwrap()).collect();
//! let mask = vec![true; pts3d.len()];
//!
//! // 흐트러진 초기 자세에서 절대 자세 정제.
//! let mut pose = Rigid3::new(Quat::from_axis_angle(&Vec3::y(), 0.12), Vec3::new(0.25, -0.05, 5.9));
//! let summary = refine_abs_pose(&cam, &pts2d, &pts3d, &mask, &mut pose, Loss::Cauchy(1.0), 100).unwrap();
//! assert!(summary.is_usable());
//! assert!((pose.translation - truth.translation).norm() < 1e-6);
//! ```
//!
//! ```no_run
//! use cumulus3d_ba::{bundle_adjust, BaConfig};
//! let mut rec = cumulus3d_core::interop::read_model("model/0").unwrap();
//! let summary = bundle_adjust(&mut rec, &BaConfig::default()).unwrap();
//! println!("RMS {:.3} px, 반복 {}", summary.rms_reprojection_error(), summary.num_iterations);
//! ```
#![warn(missing_docs)]
// 수치 커널은 인덱스 루프가 읽기 쉽다.
#![allow(clippy::needless_range_loop)]

mod abspose;
mod linsolve;
mod loss;
mod problem;
mod tr;

use cumulus3d_core::{CameraId, Error, FrameId, ImageId, Point3DId, Reconstruction, Rigid3, SensorKey, Vec2, Vec3};
use problem::{BaProblem, ProblemInput};
use std::collections::{BTreeMap, HashSet};
use tr::TrOptions;

/// 견고 손실 함수. 스케일은 잔차(픽셀) 단위.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Loss {
    /// 손실 없음(제곱 오차 그대로).
    Trivial,
    /// Soft L1 손실(인자 = 스케일).
    SoftL1(f64),
    /// Cauchy 손실(인자 = 스케일).
    Cauchy(f64),
    /// Huber 손실(인자 = 스케일).
    Huber(f64),
}

/// 축소 카메라 계통 풀이기.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LinearSolverType {
    /// 영상 수로 자동 선택.
    #[default]
    Auto,
    /// 밀집 Schur + 밀집 촐레스키.
    DenseSchur,
    /// 희소 Schur + 희소 촐레스키(faer).
    SparseSchur,
    /// 반복 Schur: PCG + Schur 블록 야코비 전처리.
    IterativeSchur,
}

/// 최적화 종료 상태. `Failure` 외에는 결과를 쓸 수 있다.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Termination {
    /// 허용오차 조건으로 수렴.
    Convergence,
    #[default]
    /// 최대 반복에 도달(결과는 사용 가능).
    NoConvergence,
    /// 수치 실패(결과를 쓰면 안 됨).
    Failure,
}

/// 어떤 변수를 정제·고정할지.
#[derive(Clone, Debug)]
pub struct BaConfig {
    /// 초점 거리 정제 여부.
    pub refine_focal_length: bool,
    /// 주점 정제 여부.
    pub refine_principal_point: bool,
    /// 추가 파라미터(왜곡 등) 정제 여부.
    pub refine_extra_params: bool,
    /// 영상(프레임) 자세 정제 여부.
    pub refine_poses: bool,
    /// 3D 점 위치 정제 여부.
    pub refine_points: bool,
    /// 견고 손실 함수.
    pub loss: Loss,
    /// 최대 LM 반복 수.
    pub max_num_iterations: usize,
    /// 상대 비용 감소 허용오차(0 이면 끔).
    pub function_tolerance: f64,
    /// 기울기 허용오차.
    pub gradient_tolerance: f64,
    /// 파라미터 변화 허용오차(0 이면 끔).
    pub parameter_tolerance: f64,
    /// 대상 영상. 비면 등록된 영상 전부.
    pub images: HashSet<ImageId>,
    /// 자세를 고정할 영상.
    pub constant_poses: HashSet<ImageId>,
    /// 내부 파라미터를 고정할 카메라.
    pub constant_cameras: HashSet<CameraId>,
    /// 위치를 고정할 3D 점.
    pub constant_points: HashSet<Point3DId>,
    /// 게이지 고정을 자동으로 할지(두 영상 고정 규칙). constant_poses 가 비었을 때만 의미 있음.
    pub auto_gauge: bool,
    /// 스레드 수(0 이면 전역 rayon 풀).
    pub num_threads: usize,
    /// 모든 자세의 회전 고정(이동만 정제).
    pub constant_world_to_rig_rotation: bool,
    /// > 0 이면 트랙 길이가 이보다 짧은 점은 문제에서 제외.
    pub min_track_length: usize,
    /// 축소 계통 풀이기 종류.
    pub linear_solver: LinearSolverType,
    /// 반복 풀이기(PCG) 최대 반복.
    pub max_linear_solver_iterations: usize,
    /// `Auto` 에서 밀집 풀이기를 쓰는 최대 영상 수.
    pub dense_solver_image_limit: usize,
    /// `Auto` 에서 희소 풀이기를 쓰는 최대 영상 수.
    pub sparse_solver_image_limit: usize,
    /// BA 전에 깊이 < ε 관측 삭제(사전 처리).
    pub filter_negative_depth: bool,
    /// BA 후 모든 3D 점의 평균 재투영 오차(`Point3D::error`) 재계산.
    pub update_point_errors: bool,
}

impl Default for BaConfig {
    /// 단독 번들 조정 기본값.
    fn default() -> Self {
        Self {
            refine_focal_length: true,
            refine_principal_point: false,
            refine_extra_params: true,
            refine_poses: true,
            refine_points: true,
            loss: Loss::Trivial,
            max_num_iterations: 100,
            function_tolerance: 0.0,
            gradient_tolerance: 1e-4,
            parameter_tolerance: 0.0,
            images: HashSet::new(),
            constant_poses: HashSet::new(),
            constant_cameras: HashSet::new(),
            constant_points: HashSet::new(),
            auto_gauge: true,
            num_threads: 0,
            constant_world_to_rig_rotation: false,
            min_track_length: 0,
            linear_solver: LinearSolverType::Auto,
            max_linear_solver_iterations: 200,
            dense_solver_image_limit: 50,
            sparse_solver_image_limit: 1000,
            filter_negative_depth: true,
            update_point_errors: true,
        }
    }
}

/// 번들 조정 결과 요약.
#[derive(Clone, Debug, Default)]
pub struct BaSummary {
    /// 수행한 반복 수.
    pub num_iterations: usize,
    /// 초기 비용(½·잔차 제곱합).
    pub initial_cost: f64,
    /// 최종 비용(½·잔차 제곱합).
    pub final_cost: f64,
    /// 잔차 수(관측 수 × 2).
    pub num_residuals: usize,
    /// `termination == Convergence` 여부.
    pub converged: bool,
    /// 종료 상태.
    pub termination: Termination,
    /// 채택된(성공한) 단계 수.
    pub num_successful_steps: usize,
    /// 실제로 쓴 선형 풀이기.
    pub linear_solver: LinearSolverType,
    /// 문제에 들어간 영상 수.
    pub num_images: usize,
    /// 문제에 들어간 3D 점 수.
    pub num_points: usize,
    /// 가변(고정 아님) 3D 점 수.
    pub free_point_count: usize,
    /// 사전 필터가 지운 관측 수.
    pub dropped_observations: usize,
}

impl BaSummary {
    /// 결과를 쓸 수 있는지(종료 상태가 실패가 아님).
    pub fn is_usable(&self) -> bool {
        self.termination != Termination::Failure
    }
    /// 최종 RMS 재투영 오차(픽셀, 손실 없을 때 의미 있음): √(2·비용 / 관측 수).
    pub fn rms_reprojection_error(&self) -> f64 {
        if self.num_residuals == 0 {
            0.0
        } else {
            (2.0 * self.final_cost / (self.num_residuals as f64 / 2.0)).sqrt()
        }
    }
}

/// 영상 수로 풀이기 선택(CPU).
pub fn select_linear_solver(num_images: usize, config: &BaConfig) -> LinearSolverType {
    match config.linear_solver {
        LinearSolverType::Auto => {
            if num_images <= config.dense_solver_image_limit {
                LinearSolverType::DenseSchur
            } else if num_images <= config.sparse_solver_image_limit {
                LinearSolverType::SparseSchur
            } else {
                LinearSolverType::IterativeSchur
            }
        }
        s => s,
    }
}

/// 주어진 등록 영상들에서 깊이(카메라 z) < ε 인 관측 삭제. 트랙 ≤ 2 면 점째 삭제(core 규칙).
/// 반환: 삭제한 관측 수.
pub fn filter_negative_depth_observations(rec: &mut Reconstruction, images: &[ImageId]) -> cumulus3d_core::Result<usize> {
    let mut del = Vec::new();
    for &id in images {
        let (Some(pose), Some(im)) = (rec.world_to_cam(id), rec.image(id)) else {
            continue;
        };
        let m = pose.matrix();
        for (idx, p) in im.points2d().iter().enumerate() {
            if !p.has_point3d() {
                continue;
            }
            let Some(pt) = rec.point3d(p.point3d_id) else { continue };
            let x = &pt.xyz;
            let z = m[(2, 0)] * x.x + m[(2, 1)] * x.y + m[(2, 2)] * x.z + m[(2, 3)];
            if z < f64::EPSILON {
                del.push((id, idx as u32));
            }
        }
    }
    let mut n = 0;
    for (id, idx) in del {
        // 앞선 삭제로 점이 사라졌을 수 있다.
        if rec.image(id).is_some_and(|im| im.point2d(idx).has_point3d()) {
            rec.delete_observation(id, idx)?;
            n += 1;
        }
    }
    Ok(n)
}

/// 재구성에 번들 조정을 적용한다. 점·자세·카메라를 제자리에서 갱신.
///
/// 풀이기가 `Failure` 로 끝나도 마지막으로 채택된 해를 써 넣고 `Ok` 를 돌려준다
/// (`summary.termination` 으로 확인). 자명하지 않은 rig 는 프레임 자세를 변수로 두고
/// rig_to_sensor 는 상수로 둔다.
pub fn bundle_adjust(rec: &mut Reconstruction, config: &BaConfig) -> cumulus3d_core::Result<BaSummary> {
    if config.num_threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .build()
            .map_err(|e| Error::InvalidArgument(format!("스레드 풀: {e}")))?;
        pool.install(|| bundle_adjust_impl(rec, config))
    } else {
        bundle_adjust_impl(rec, config)
    }
}

struct Built {
    input: ProblemInput,
    frame_ids: Vec<FrameId>,
    cam_ids: Vec<CameraId>,
    point_ids: Vec<Point3DId>,
    num_images: usize,
}

fn bundle_adjust_impl(rec: &mut Reconstruction, config: &BaConfig) -> cumulus3d_core::Result<BaSummary> {
    let mut images: Vec<ImageId> = if config.images.is_empty() {
        rec.registered_images()
    } else {
        config.images.iter().copied().filter(|&i| rec.is_image_registered(i)).collect()
    };
    images.sort_unstable();
    images.dedup();
    let mut summary = BaSummary { termination: Termination::Convergence, converged: true, ..Default::default() };
    if images.is_empty() {
        return Ok(summary);
    }
    if config.filter_negative_depth {
        summary.dropped_observations = filter_negative_depth_observations(rec, &images)?;
    }
    let Some(built) = build_problem(rec, config, &images) else {
        return Ok(summary);
    };
    let Built { input, frame_ids, cam_ids, point_ids, num_images } = built;
    summary.num_images = num_images;
    summary.num_points = point_ids.len();
    summary.num_residuals = 2 * input.obs_img.len();
    let mut prob = BaProblem::new(input).ok_or_else(|| Error::Invariant("BA 선형 풀이기 구조 생성 실패".into()))?;
    summary.linear_solver = prob.solver_kind;
    summary.free_point_count = prob.num_var_points();
    let opts = TrOptions::new(config.max_num_iterations, config.function_tolerance, config.gradient_tolerance, config.parameter_tolerance);
    let r = tr::minimize(&mut prob, &opts);
    summary.num_iterations = r.num_iterations;
    summary.num_successful_steps = r.num_successful_steps;
    summary.initial_cost = r.initial_cost;
    summary.final_cost = r.final_cost;
    summary.termination = r.termination;
    summary.converged = r.termination == Termination::Convergence;
    // 써 넣기: 자세는 전부(사전 정규화 반영), 카메라·점은 가변인 것만.
    for (p, fid) in frame_ids.iter().enumerate() {
        rec.set_frame_pose(*fid, Some(prob.cur.poses[p]))?;
    }
    for (c, cid) in cam_ids.iter().enumerate() {
        if prob.cam_e[c] != problem::NONE {
            if let Some(cam) = rec.camera_mut(*cid) {
                cam.params.clone_from(&prob.cur.cams[c].params);
            }
        }
    }
    for (j, pid) in point_ids.iter().enumerate() {
        if prob.pt_v[j] != problem::NONE {
            rec.set_point3d_xyz(*pid, prob.cur.pts[j])?;
        }
    }
    if config.update_point_errors {
        rec.update_point3d_errors();
    }
    Ok(summary)
}

fn build_problem(rec: &Reconstruction, config: &BaConfig, images: &[ImageId]) -> Option<Built> {
    let in_set: HashSet<ImageId> = images.iter().copied().collect();
    // 관측 수집 (점 id, 영상 순번, xy)
    let mut raw: Vec<(Point3DId, u32, Vec2)> = Vec::new();
    let mut img_ok: Vec<bool> = vec![false; images.len()];
    for (ii, &id) in images.iter().enumerate() {
        let Some(im) = rec.image(id) else { continue };
        let Some(frame) = rec.frame(im.frame_id) else { continue };
        if frame.world_to_rig.is_none() {
            continue;
        }
        let Some(rig) = rec.rig(frame.rig_id) else { continue };
        let s = SensorKey::camera(im.camera_id);
        if !rig.is_reference_sensor(s) && rig.rig_to_sensor(s).is_none() {
            continue;
        }
        img_ok[ii] = true;
        for p in im.points2d() {
            if p.has_point3d() {
                raw.push((p.point3d_id, ii as u32, p.xy));
            }
        }
    }
    raw.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)).then(a.2.x.total_cmp(&b.2.x)));
    // 점 선택
    let mut obs: Vec<(u32, u32, Vec2)> = Vec::with_capacity(raw.len()); // (점 순번, 영상 순번, xy)
    let mut point_ids: Vec<Point3DId> = Vec::new();
    let mut points: Vec<Vec3> = Vec::new();
    let mut point_const: Vec<bool> = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        let pid = raw[i].0;
        let mut e = i;
        while e < raw.len() && raw[e].0 == pid {
            e += 1;
        }
        let Some(pt) = rec.point3d(pid) else {
            i = e;
            continue;
        };
        if config.min_track_length > 0 && pt.track.len() < config.min_track_length {
            i = e;
            continue;
        }
        let partial = pt.track.iter().any(|t| !in_set.contains(&t.image_id));
        let pj = point_ids.len() as u32;
        point_ids.push(pid);
        points.push(pt.xyz);
        point_const.push(!config.refine_points || partial || config.constant_points.contains(&pid));
        for r in &raw[i..e] {
            obs.push((pj, r.1, r.2));
        }
        i = e;
    }
    if obs.is_empty() {
        return None;
    }
    // 관측 있는 영상만 문제에 넣는다.
    let mut used = vec![false; images.len()];
    for o in &obs {
        used[o.1 as usize] = true;
    }
    let mut img_map = vec![u32::MAX; images.len()];
    let mut frame_idx: BTreeMap<FrameId, u32> = BTreeMap::new();
    let mut cam_idx: BTreeMap<CameraId, u32> = BTreeMap::new();
    let mut frame_ids = Vec::new();
    let mut cam_ids = Vec::new();
    let mut prob_images: Vec<ImageId> = Vec::new();
    let (mut img_pose, mut img_cam, mut img_sensor) = (Vec::new(), Vec::new(), Vec::new());
    for (ii, &id) in images.iter().enumerate() {
        if !used[ii] || !img_ok[ii] {
            continue;
        }
        let im = rec.image(id)?;
        let frame = rec.frame(im.frame_id)?;
        let rig = rec.rig(frame.rig_id)?;
        let s = SensorKey::camera(im.camera_id);
        let p = *frame_idx.entry(im.frame_id).or_insert_with(|| {
            frame_ids.push(im.frame_id);
            (frame_ids.len() - 1) as u32
        });
        let c = *cam_idx.entry(im.camera_id).or_insert_with(|| {
            cam_ids.push(im.camera_id);
            (cam_ids.len() - 1) as u32
        });
        img_map[ii] = prob_images.len() as u32;
        prob_images.push(id);
        img_pose.push(p);
        img_cam.push(c);
        img_sensor.push(if rig.is_reference_sensor(s) { None } else { rig.rig_to_sensor(s) });
    }
    let (obs_img, obs_pt, obs_xy): (Vec<u32>, Vec<u32>, Vec<Vec2>) = {
        let mut a = Vec::with_capacity(obs.len());
        let mut b = Vec::with_capacity(obs.len());
        let mut c = Vec::with_capacity(obs.len());
        for (pj, ii, xy) in obs {
            a.push(img_map[ii as usize]);
            b.push(pj);
            c.push(xy);
        }
        (a, b, c)
    };
    // 자세(정제 전 쿼터니언 정규화)
    let poses: Vec<Rigid3> = frame_ids
        .iter()
        .map(|f| {
            let p = rec.frame(*f).and_then(|fr| fr.world_to_rig).unwrap_or_default();
            Rigid3::new(p.rotation.normalized(), p.translation)
        })
        .collect();
    let mut pose_const = vec![!config.refine_poses; poses.len()];
    for (k, &id) in prob_images.iter().enumerate() {
        if config.constant_poses.contains(&id) {
            pose_const[img_pose[k] as usize] = true;
        }
    }
    let rot_mask = config.constant_world_to_rig_rotation;
    let mut pose_mask = vec![[rot_mask, rot_mask, rot_mask, false, false, false]; poses.len()];
    // 카메라
    let cams: Vec<cumulus3d_core::Camera> = cam_ids.iter().map(|c| rec.camera(*c).cloned()).collect::<Option<_>>()?;
    let cam_var: Vec<Vec<usize>> = cams
        .iter()
        .map(|cam| {
            if config.constant_cameras.contains(&cam.camera_id) {
                return Vec::new();
            }
            let mut v = Vec::new();
            if config.refine_focal_length {
                v.extend_from_slice(cam.model.focal_slots());
            }
            if config.refine_principal_point {
                v.extend_from_slice(cam.model.pp_slots());
            }
            if config.refine_extra_params {
                v.extend_from_slice(cam.model.extra_param_slots());
            }
            v.sort_unstable();
            v.dedup();
            v
        })
        .collect();
    // 게이지 고정
    if config.auto_gauge {
        let n_const_frames = pose_const.iter().filter(|&&c| c).count();
        if n_const_frames < 2 {
            let world_to_cam = |k: usize| -> Rigid3 {
                let p = poses[img_pose[k] as usize];
                match img_sensor[k] {
                    Some(s) => s.compose(&p),
                    None => p,
                }
            };
            let mut image1: Option<usize> = None;
            let mut image2: Option<(usize, usize)> = None;
            for k in 0..prob_images.len() {
                // 설계 결정: 비자명 rig 에서는 기준 센서 영상만 자격(프레임 고정 = 영상 고정이 되도록).
                if img_sensor[k].is_some() {
                    continue;
                }
                match image1 {
                    None => image1 = Some(k),
                    Some(k1) => {
                        if img_pose[k] == img_pose[k1] {
                            continue;
                        }
                        let b = world_to_cam(k1).compose(&world_to_cam(k).inverse());
                        let t = b.translation;
                        let (d, m) = (0..3).map(|i| (i, t[i].abs())).fold((0, -1.0), |a, x| if x.1 > a.1 { x } else { a });
                        if m > 1e-9 {
                            image2 = Some((k, d));
                            break;
                        }
                    }
                }
            }
            match (image1, image2) {
                (Some(k1), Some((k2, d))) => {
                    pose_const[img_pose[k1] as usize] = true;
                    pose_mask[img_pose[k2] as usize][3 + d] = true;
                }
                _ => {
                    // 세 점 고정 대체.
                    let mut fixed: Vec<Vec3> = (0..points.len()).filter(|&j| point_const[j]).map(|j| points[j]).collect();
                    let mut count = fixed.len();
                    let mut j = 0;
                    while count < 3 && j < points.len() {
                        if !point_const[j] {
                            let before = rank3(&fixed);
                            fixed.push(points[j]);
                            if rank3(&fixed) > before {
                                point_const[j] = true;
                                count += 1;
                            } else {
                                fixed.pop();
                            }
                        }
                        j += 1;
                    }
                }
            }
        }
    }
    let num_images = prob_images.len();
    let solver = select_linear_solver(num_images, config);
    Some(Built {
        input: ProblemInput {
            poses,
            pose_const,
            pose_mask,
            cams,
            cam_var,
            img_pose,
            img_cam,
            img_sensor,
            points,
            point_const,
            obs_img,
            obs_pt,
            obs_xy,
            loss: config.loss,
            solver,
            max_linear_solver_iterations: config.max_linear_solver_iterations,
        },
        frame_ids,
        cam_ids,
        point_ids,
        num_images,
    })
}

/// 열벡터(원점 기준, 중심화 안 함) 행렬의 계수.
fn rank3(pts: &[Vec3]) -> usize {
    if pts.is_empty() {
        return 0;
    }
    let m = nalgebra::DMatrix::from_fn(3, pts.len(), |r, c| pts[c][r]);
    let sv = m.svd(false, false).singular_values;
    let mx = sv.iter().cloned().fold(0.0, f64::max);
    sv.iter().filter(|&&s| s > 1e-9 * mx.max(1e-300)).count()
}

/// 자세 하나만 정제(등록 직후 절대 자세 정제용). 2D 점(픽셀)과 3D 점 대응, 인라이어 마스크.
///
/// 3D 점·내부 파라미터 고정, 자세 6자유도. 허용치: 기울기 1.0, 함수 1e−6, 파라미터 1e−8.
/// 풀이기 실패면 `Err` 이고 자세는 그대로다.
pub fn refine_abs_pose(
    camera: &cumulus3d_core::Camera,
    points2d: &[cumulus3d_core::Vec2],
    points3d: &[cumulus3d_core::Vec3],
    inlier_mask: &[bool],
    world_to_cam: &mut cumulus3d_core::Rigid3,
    loss: Loss,
    max_num_iterations: usize,
) -> cumulus3d_core::Result<BaSummary> {
    if points2d.len() != points3d.len() || inlier_mask.len() != points2d.len() {
        return Err(Error::InvalidArgument("refine_abs_pose: 입력 길이 불일치".into()));
    }
    let (mut x2, mut x3) = (Vec::new(), Vec::new());
    for i in 0..points2d.len() {
        if inlier_mask[i] {
            x2.push(points2d[i]);
            x3.push(points3d[i]);
        }
    }
    if x2.is_empty() {
        return Err(Error::InvalidArgument("refine_abs_pose: 인라이어 없음".into()));
    }
    let n = x2.len();
    let pose = Rigid3::new(world_to_cam.rotation.normalized(), world_to_cam.translation);
    let mut prob = abspose::AbsPoseProblem::new(camera, x2, x3, loss, pose);
    // 설계 결정: 함수·파라미터 허용치는 일반적인 기본값(1e−6, 1e−8).
    let opts = TrOptions::new(max_num_iterations, 1e-6, 1.0, 1e-8);
    let r = tr::minimize(&mut prob, &opts);
    if r.termination == Termination::Failure {
        return Err(Error::Invariant("refine_abs_pose: 풀이기 실패".into()));
    }
    *world_to_cam = prob.cur;
    Ok(BaSummary {
        num_iterations: r.num_iterations,
        initial_cost: r.initial_cost,
        final_cost: r.final_cost,
        num_residuals: 2 * n,
        converged: r.termination == Termination::Convergence,
        termination: r.termination,
        num_successful_steps: r.num_successful_steps,
        linear_solver: LinearSolverType::DenseSchur,
        num_images: 1,
        num_points: n,
        free_point_count: 0,
        dropped_observations: 0,
    })
}
