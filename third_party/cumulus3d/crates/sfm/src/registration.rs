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

//! Registers new images to an existing model (image_registrator).

use crate::absolute_pose::{AbsPoseBackend, AbsolutePoseOptions, CpuAbsPose};
use cumulus3d_ba::Loss;
use cumulus3d_core::{
    Camera, FeatureStore, Image, ImageId, MatchGraph, Point2DIdx, Point3DId, Reconstruction, Result, TrackEntry, Vec2, Vec3,
};
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Registration order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegistrationOrder {
    /// Ascending image id (deterministic order).
    /// Ids are issued in arrival order (position, then camera), so this also equals the arrival order.
    ImageId,
    /// Improved: re-pick every time in descending order of the visibility pyramid score and register.
    VisibilityScore,
}

/// image_registrator options.
#[derive(Clone, Debug)]
pub struct RegistrationOptions {
    /// Minimum number of visible points / correspondences / inliers needed for registration.
    pub abs_pose_min_num_inliers: usize,
    /// Absolute pose RANSAC options.
    pub ransac: AbsolutePoseOptions,
    /// Abnormal camera test: lower bound of the focal ratio.
    pub min_focal_length_ratio: f64,
    /// Abnormal camera test: upper bound of the focal ratio.
    pub max_focal_length_ratio: f64,
    /// Abnormal camera test: upper bound of the absolute value of the extra parameters.
    pub max_extra_param: f64,
    /// Whether to run the nonlinear pose refinement after RANSAC (`refine_abs_pose` of cumulus3d-ba).
    pub refine_pose: bool,
    /// Loss function of the pose refinement.
    pub refine_loss: Loss,
    /// Maximum iterations of the pose refinement.
    pub refine_max_num_iterations: usize,
    /// Registration order.
    pub order: RegistrationOrder,
    /// After finishing, delete images that failed registration (frames and unused cameras) from the model.
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

/// Result of registering one image.
#[derive(Clone, Debug, PartialEq)]
pub enum RegisterOutcome {
    /// Registration succeeded.
    Registered {
        /// Number of 2D–3D correspondences.
        num_correspondences: usize,
        /// Number of RANSAC inliers.
        num_inliers: usize,
    },
    /// Too few visible 3D points (number of visible points).
    TooFewVisiblePoints(usize),
    /// Too few 2D–3D correspondences (number of correspondences).
    TooFewCorrespondences(usize),
    /// Absolute pose RANSAC failed.
    RansacFailed,
    /// Too few inliers (number of inliers).
    TooFewInliers(usize),
    /// Pose refinement failed.
    RefinementFailed,
    /// Image already registered.
    AlreadyRegistered,
}

impl RegisterOutcome {
    /// Whether registration succeeded.
    pub fn is_registered(&self) -> bool {
        matches!(self, RegisterOutcome::Registered { .. })
    }
}

/// Overall result.
#[derive(Clone, Debug, Default)]
pub struct RegistrationReport {
    /// (image, result) in attempt order.
    pub attempts: Vec<(ImageId, RegisterOutcome)>,
    /// Number of images newly added to the model.
    pub num_added_images: usize,
}

impl RegistrationReport {
    /// Ids of successfully registered images (in attempt order).
    pub fn registered(&self) -> Vec<ImageId> {
        self.attempts.iter().filter(|(_, o)| o.is_registered()).map(|(i, _)| *i).collect()
    }
}

/// Adds, without poses, the cameras and images of the store that are in the correspondence graph but not in the model.
pub fn add_missing_images(rec: &mut Reconstruction, store: &FeatureStore, graph: &MatchGraph) -> Result<Vec<ImageId>> {
    let mut ids = graph.image_ids();
    ids.sort();
    let mut added = Vec::new();
    for id in ids {
        if let Some(im) = rec.image(id) {
            // Check that name and feature count match.
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

/// "Number of visible 3D points": number of 2D points where at least one correspondence neighbor has a 3D point.
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

/// Visibility pyramid score (6 levels).
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

/// 2D-3D correspondences: (2D point, pixel, 3D point id, coordinates).
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

/// Registers one image (central camera, fixed-focal path).
pub fn register_image(
    rec: &mut Reconstruction,
    graph: &MatchGraph,
    image_id: ImageId,
    opts: &RegistrationOptions,
) -> Result<RegisterOutcome> {
    register_image_with(rec, graph, image_id, opts, &CpuAbsPose)
}

/// Like [`register_image`], but absolute pose RANSAC is done by `backend`.
pub fn register_image_with(
    rec: &mut Reconstruction,
    graph: &MatchGraph,
    image_id: ImageId,
    opts: &RegistrationOptions,
    backend: &dyn AbsPoseBackend,
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
    // Design decision: the focal estimation (P4Pf) and intrinsics refinement paths for cameras with 0 registered images or bogus cameras
    // are not implemented; the fixed-focal path is always used (the normal case for this pipeline).
    let camera = rec.camera(im.camera_id).expect("존재").clone();
    let p2: Vec<Vec2> = corrs.iter().map(|c| c.1).collect();
    let p3: Vec<Vec3> = corrs.iter().map(|c| c.3).collect();
    let mut ransac = opts.ransac.clone();
    // Deterministic seed that differs per image.
    ransac.random_seed = ransac.random_seed.map(|s| s ^ (image_id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let Some(res) = backend.solve(&camera, &p2, &p3, &ransac) else {
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
    // Track extension: only the first applied inlier (ignored if it already has a 3D point).
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

/// image_registrator as a whole: add missing images, try once per pose-less image, delete failed images.
pub fn register_images(
    rec: &mut Reconstruction,
    store: &FeatureStore,
    graph: &MatchGraph,
    opts: &RegistrationOptions,
) -> Result<RegistrationReport> {
    register_images_with(rec, store, graph, opts, &CpuAbsPose)
}

/// Like [`register_images`], but absolute pose RANSAC is done by `backend`.
pub fn register_images_with(
    rec: &mut Reconstruction,
    store: &FeatureStore,
    graph: &MatchGraph,
    opts: &RegistrationOptions,
    backend: &dyn AbsPoseBackend,
) -> Result<RegistrationReport> {
    let added = add_missing_images(rec, store, graph)?;
    let mut report = RegistrationReport { num_added_images: added.len(), ..Default::default() };
    let mut pending: Vec<ImageId> = rec.image_ids().filter(|i| !rec.has_pose(*i)).collect();
    pending.sort();
    match opts.order {
        RegistrationOrder::ImageId => {
            for id in pending {
                let o = register_image_with(rec, graph, id, opts, backend)?;
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
                let o = register_image_with(rec, graph, id, opts, backend)?;
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

    /// Registered images 1, 2 and n 3D points; new image 3 sees those points without noise.
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
        // Track extension: every point gets an image-3 observation.
        assert_eq!(rec.image(3).unwrap().num_points3d(), 30);
        rec.check_invariants().unwrap();
    }

    #[test]
    fn min_inlier_boundary() {
        // Corrupt 2 of the 30 correspondences heavily: inliers 28, so failure.
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
