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

//! Bundle adjustment.
//!
//! The public signatures of this file are a contract with the sfm crate. Do not change them during implementation (adding fields is
//! allowed).
//!
//! Components: trust-region LM (`tr`), Schur complement assembly and back-substitution (`problem`), reduced camera system solvers
//! (`linsolve`),
//! single-pose refinement (`abspose`), robust losses (`loss`), solver backends (`backend`: CPU by default, GPU implemented in
//! another crate).
//!
//! # Usage example
//!
//! ```
//! use cumulus3d_ba::{refine_abs_pose, Loss};
//! use cumulus3d_core::{Camera, CameraModelKind, Quat, Rigid3, Vec2, Vec3};
//!
//! // Synthetic scene: project 3D points with the true poses to create 2D observations.
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
//! // Absolute pose refinement from a perturbed initial pose.
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
// Numerical kernels are easier to read with index loops.
#![allow(clippy::needless_range_loop)]

mod abspose;
mod backend;
mod linsolve;
mod loss;
mod problem;
mod tr;

pub use backend::{BaBackend, BaBackendRef, BaSolution, CpuBaBackend};
pub use problem::{quat_plus, ProblemInput};
pub use tr::{minimize, StepInfo, TrOptions, TrProblem, TrResult};

use cumulus3d_core::{CameraId, Error, FrameId, ImageId, Point3DId, Reconstruction, Rigid3, SensorKey, Vec2, Vec3};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Robust loss function. The scale is in residual (pixel) units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Loss {
    /// No loss (squared error as is).
    Trivial,
    /// Soft L1 loss (argument = scale).
    SoftL1(f64),
    /// Cauchy loss (argument = scale).
    Cauchy(f64),
    /// Huber loss (argument = scale).
    Huber(f64),
}

/// Reduced camera system solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LinearSolverType {
    /// Chosen automatically by image count.
    #[default]
    Auto,
    /// Dense Schur + dense Cholesky.
    DenseSchur,
    /// Sparse Schur + sparse Cholesky (faer).
    SparseSchur,
    /// Iterative Schur: PCG + Schur block-Jacobi preconditioner.
    IterativeSchur,
}

/// Optimization termination state. The result is usable except for `Failure`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Termination {
    /// Converged by a tolerance condition.
    Convergence,
    #[default]
    /// Reached the maximum iterations (result is usable).
    NoConvergence,
    /// Numerical failure (the result must not be used).
    Failure,
}

/// Which variables to refine or fix.
#[derive(Clone, Debug)]
pub struct BaConfig {
    /// Whether to refine the focal length.
    pub refine_focal_length: bool,
    /// Whether to refine the principal point.
    pub refine_principal_point: bool,
    /// Whether to refine extra parameters (distortion, etc.).
    pub refine_extra_params: bool,
    /// Whether to refine image (frame) poses.
    pub refine_poses: bool,
    /// Whether to refine 3D point positions.
    pub refine_points: bool,
    /// Robust loss function.
    pub loss: Loss,
    /// Maximum number of LM iterations.
    pub max_num_iterations: usize,
    /// Relative cost-decrease tolerance (0 = off).
    pub function_tolerance: f64,
    /// Gradient tolerance.
    pub gradient_tolerance: f64,
    /// Parameter-change tolerance (0 = off).
    pub parameter_tolerance: f64,
    /// Target images. If empty, all registered images.
    pub images: HashSet<ImageId>,
    /// Images whose poses are held fixed.
    pub constant_poses: HashSet<ImageId>,
    /// Cameras whose intrinsics are held fixed.
    pub constant_cameras: HashSet<CameraId>,
    /// 3D points whose positions are held fixed.
    pub constant_points: HashSet<Point3DId>,
    /// Whether to fix the gauge automatically (two-image fixing rule). Meaningful only when constant_poses is empty.
    pub auto_gauge: bool,
    /// Number of threads (0 = global rayon pool).
    pub num_threads: usize,
    /// Fix the rotation of all poses (refine translation only).
    pub constant_world_to_rig_rotation: bool,
    /// If > 0, points whose track length is shorter than this are excluded from the problem.
    pub min_track_length: usize,
    /// Reduced system solver type.
    pub linear_solver: LinearSolverType,
    /// Maximum iterations of the iterative solver (PCG).
    pub max_linear_solver_iterations: usize,
    /// Maximum image count for which `Auto` uses the dense solver.
    pub dense_solver_image_limit: usize,
    /// Maximum image count for which `Auto` uses the sparse solver.
    pub sparse_solver_image_limit: usize,
    /// Delete observations with depth < ε before BA (preprocessing).
    pub filter_negative_depth: bool,
    /// After BA, recompute the mean reprojection error (`Point3D::error`) of all 3D points.
    pub update_point_errors: bool,
    /// Camera position priors: image → (target projection center in world coordinates, standard deviation in world
    /// units). Adds the residual `(C − target) / σ` per image (reference-sensor images only). When non-empty the
    /// priors fix the gauge and `auto_gauge` is not applied.
    pub position_priors: HashMap<ImageId, (Vec3, f64)>,
    /// Intrinsics priors: camera → (target parameter vector in the camera model's order, per-parameter σ in parameter
    /// units; σ ≤ 0 or NaN skips that parameter). Adds the residual `(p − target)/σ` for every parameter that is refined
    /// in this run (focal / principal point / extra, per the flags above); parameters of constant cameras are skipped.
    /// Vectors shorter than the model's parameter count cover only the leading parameters.
    pub intrinsics_priors: HashMap<CameraId, (Vec<f64>, Vec<f64>)>,
}

impl Default for BaConfig {
    /// Default values for standalone bundle adjustment.
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
            position_priors: HashMap::new(),
            intrinsics_priors: HashMap::new(),
        }
    }
}

/// Bundle adjustment result summary.
#[derive(Clone, Debug, Default)]
pub struct BaSummary {
    /// Number of iterations performed.
    pub num_iterations: usize,
    /// Initial cost (½·sum of squared residuals).
    pub initial_cost: f64,
    /// Final cost (½·sum of squared residuals).
    pub final_cost: f64,
    /// Number of residuals (observations × 2).
    pub num_residuals: usize,
    /// Whether `termination == Convergence`.
    pub converged: bool,
    /// Termination state.
    pub termination: Termination,
    /// Number of accepted (successful) steps.
    pub num_successful_steps: usize,
    /// Linear solver actually used.
    pub linear_solver: LinearSolverType,
    /// Number of images in the problem.
    pub num_images: usize,
    /// Number of 3D points in the problem.
    pub num_points: usize,
    /// Number of variable (non-fixed) 3D points.
    pub free_point_count: usize,
    /// Number of observations removed by the pre-filter.
    pub dropped_observations: usize,
}

impl BaSummary {
    /// Whether the result is usable (the termination state is not a failure).
    pub fn is_usable(&self) -> bool {
        self.termination != Termination::Failure
    }
    /// Final RMS reprojection error (pixels, meaningful when there is no loss): √(2·cost / number of observations).
    pub fn rms_reprojection_error(&self) -> f64 {
        if self.num_residuals == 0 {
            0.0
        } else {
            (2.0 * self.final_cost / (self.num_residuals as f64 / 2.0)).sqrt()
        }
    }
}

/// Chooses the solver by image count (CPU).
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

/// Deletes observations with depth (camera z) < ε in the given registered images. If the track is <= 2, the whole point is deleted (core
/// rule).
/// Returns: the number of observations deleted.
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
        // A point may have disappeared through earlier deletions.
        if rec.image(id).is_some_and(|im| im.point2d(idx).has_point3d()) {
            rec.delete_observation(id, idx)?;
            n += 1;
        }
    }
    Ok(n)
}

/// Applies bundle adjustment to the reconstruction. Updates points, poses, and cameras in place.
///
/// Even if the solver ends in `Failure`, writes back the last accepted solution and returns `Ok`
/// (check via `summary.termination`). A non-trivial rig keeps the frame poses as variables
/// and rig_to_sensor as constants.
pub fn bundle_adjust(rec: &mut Reconstruction, config: &BaConfig) -> cumulus3d_core::Result<BaSummary> {
    bundle_adjust_with(rec, config, &CpuBaBackend)
}

/// [`bundle_adjust`] 와 같되 풀이를 `backend` 에 맡긴다(예: cumulus3d-cuda 의 GPU 백엔드).
///
/// 문제 구성(관측 수집·게이지 고정)과 결과 써 넣기는 백엔드와 무관하게 같다.
pub fn bundle_adjust_with(rec: &mut Reconstruction, config: &BaConfig, backend: &dyn BaBackend) -> cumulus3d_core::Result<BaSummary> {
    if config.num_threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .build()
            .map_err(|e| Error::InvalidArgument(format!("스레드 풀: {e}")))?;
        pool.install(|| bundle_adjust_impl(rec, config, backend))
    } else {
        bundle_adjust_impl(rec, config, backend)
    }
}

struct Built {
    input: ProblemInput,
    frame_ids: Vec<FrameId>,
    cam_ids: Vec<CameraId>,
    point_ids: Vec<Point3DId>,
    num_images: usize,
}

fn bundle_adjust_impl(rec: &mut Reconstruction, config: &BaConfig, backend: &dyn BaBackend) -> cumulus3d_core::Result<BaSummary> {
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
    summary.free_point_count = input.point_const.iter().filter(|&&c| !c).count();
    let opts = TrOptions::new(config.max_num_iterations, config.function_tolerance, config.gradient_tolerance, config.parameter_tolerance);
    let sol = backend.solve(&input, &opts)?;
    summary.linear_solver = sol.linear_solver;
    let r = &sol.result;
    summary.num_iterations = r.num_iterations;
    summary.num_successful_steps = r.num_successful_steps;
    summary.initial_cost = r.initial_cost;
    summary.final_cost = r.final_cost;
    summary.termination = r.termination;
    summary.converged = r.termination == Termination::Convergence;
    // Write back: all poses (reflecting the prior normalization), only the variable cameras and points.
    for (p, fid) in frame_ids.iter().enumerate() {
        rec.set_frame_pose(*fid, Some(sol.poses[p]))?;
    }
    for (c, cid) in cam_ids.iter().enumerate() {
        if !input.cam_var[c].is_empty() {
            if let Some(cam) = rec.camera_mut(*cid) {
                cam.params.clone_from(&sol.cams[c].params);
            }
        }
    }
    for (j, pid) in point_ids.iter().enumerate() {
        if !input.point_const[j] {
            rec.set_point3d_xyz(*pid, sol.points[j])?;
        }
    }
    if config.update_point_errors {
        rec.update_point3d_errors();
    }
    Ok(summary)
}

fn build_problem(rec: &Reconstruction, config: &BaConfig, images: &[ImageId]) -> Option<Built> {
    let in_set: HashSet<ImageId> = images.iter().copied().collect();
    // Collect observations (point id, image index, xy)
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
    // Point selection
    let mut obs: Vec<(u32, u32, Vec2)> = Vec::with_capacity(raw.len()); // (point index, image index, xy)
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
    // Only images with observations go into the problem.
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
    // Poses (quaternion normalization before refinement)
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
    // Cameras
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
    // Position prior (reference-sensor images only): (pose index, target center, σ).
    let mut pose_priors: Vec<(usize, Vec3, f64)> = Vec::new();
    for (k, id) in prob_images.iter().enumerate() {
        if let (Some(&(c, sigma)), None) = (config.position_priors.get(id), img_sensor[k]) {
            if sigma > 0.0 && c.iter().all(|x| x.is_finite()) {
                pose_priors.push((img_pose[k] as usize, c, sigma));
            }
        }
    }
    // Intrinsics prior: (camera index, parameter slot, target, 1/σ). Only slots being refined (filtered on the problem side).
    let mut cam_priors: Vec<(usize, usize, f64, f64)> = Vec::new();
    for (ci, cam) in cams.iter().enumerate() {
        if let Some((target, sigma)) = config.intrinsics_priors.get(&cam.camera_id) {
            for (slot, (&t, &s)) in target.iter().zip(sigma.iter()).enumerate().take(cam.params.len()) {
                if s > 0.0 && s.is_finite() && t.is_finite() {
                    cam_priors.push((ci, slot, t, 1.0 / s));
                }
            }
        }
    }
    // Gauge fixing
    if config.auto_gauge && pose_priors.is_empty() {
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
                // Design decision: for a non-trivial rig only reference-sensor images qualify (so that frame fixing = image fixing).
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
                    // Substitute fixing three points.
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
            pose_priors,
            cam_priors,
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

/// Rank of a matrix of column vectors (relative to the origin, not centered).
fn rank3(pts: &[Vec3]) -> usize {
    if pts.is_empty() {
        return 0;
    }
    let m = nalgebra::DMatrix::from_fn(3, pts.len(), |r, c| pts[c][r]);
    let sv = m.svd(false, false).singular_values;
    let mx = sv.iter().cloned().fold(0.0, f64::max);
    sv.iter().filter(|&&s| s > 1e-9 * mx.max(1e-300)).count()
}

/// Refines a single pose only (for absolute pose refinement right after registration). 2D points (pixels) and 3D point correspondences,
/// inlier mask.
///
/// 3D points and intrinsics fixed, pose 6 DoF. Tolerances: gradient 1.0, function 1e-6, parameter 1e-8.
/// If the solver fails, returns `Err` and the pose is unchanged.
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
    // Design decision: the function and parameter tolerances are common defaults (1e-6, 1e-8).
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
