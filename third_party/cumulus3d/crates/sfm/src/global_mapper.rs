/*
 * global_mapper.rs
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

//! Global mapper: relative poses, view graph, 2 rounds of rotation averaging, tracks, position estimation, filtering and normalization,
//! then (BA stage), (re-triangulation), and error recomputation.

use crate::positioning::{global_positioning, PositionSolverOptions, PositioningSummary};
use crate::rotation_averaging::{solve_rotation_averaging, RotationAveragingOptions, ViewGraph, ViewGraphEdge};
use crate::tracks::{establish_tracks, TrackOptions, TrackSummary};
use crate::triangulator::{refine_points, PointRefiner, TrackTriangulator, TrackTriangulatorOptions};
use cumulus3d_ba::{BaBackendRef, BaConfig, Loss};
use cumulus3d_core::reconstruction::{FilterErrorUpdate, NormalizeOptions};
use cumulus3d_core::{images_of_pair, Camera, FeatureStore, Image, ImageId, MatchGraph, Reconstruction, Result, Rigid3, Vec2, Vec3};
use rayon::prelude::*;
use std::collections::HashSet;

/// Global mapper options. Defaults are tuned for typical aerial images.
#[derive(Clone, Debug)]
pub struct GlobalSfmOptions {
    /// For pairs with no relative pose, whether to refit E/F/H and decompose the relative pose.
    pub decompose_relative_pose: bool,
    /// Number of BA stage iterations (if 0, only the final filter runs).
    pub ba_num_iterations: usize,
    /// Skip the rotation averaging stage.
    pub skip_rotation_averaging: bool,
    /// Skip the track construction stage.
    pub skip_track_establishment: bool,
    /// Skip the global position estimation stage.
    pub skip_global_positioning: bool,
    /// Skip the BA stage.
    pub skip_bundle_adjustment: bool,
    /// Skip the re-triangulation stage.
    pub skip_retriangulation: bool,
    /// Track construction options.
    pub tracks: TrackOptions,
    /// Maximum angular reprojection error (degrees) of the filter after position estimation.
    pub max_angular_reproj_error_deg: f64,
    /// Threshold of the normalized-coordinate reprojection error filter.
    pub max_normalized_reproj_error: f64,
    /// Minimum triangulation angle (degrees) of the point filter.
    pub min_tri_angle_deg: f64,
    /// Upper bound on the relative rotation error (degrees) beyond which edges are invalidated after rotation averaging.
    pub ra_max_rotation_error_deg: f64,
    /// Rotation averaging options.
    pub rotation_averaging: RotationAveragingOptions,
    /// Global position estimation options.
    pub positioning: PositionSolverOptions,
    /// Common BA settings (what to refine).
    pub ba_refine_focal_length: bool,
    /// Whether BA refines the principal point.
    pub ba_refine_principal_point: bool,
    /// Whether BA refines extra parameters such as distortion.
    pub ba_refine_extra_params: bool,
    /// Huber loss scale of BA.
    pub ba_loss_scale: f64,
    /// Maximum iterations of a single BA run.
    pub ba_max_num_iterations: usize,
    /// Skip the rotation-fixed BA sub-stage.
    pub ba_skip_fixed_rotation_stage: bool,
    /// Skip the all-variables BA sub-stage.
    pub ba_skip_joint_optimization_stage: bool,
    /// Upper bound on the reprojection error (pixels) for track completion in re-triangulation.
    pub tri_complete_max_reproj_error: f64,
    /// Upper bound on the reprojection error (pixels) for track merging in re-triangulation.
    pub tri_merge_max_reproj_error: f64,
    /// Minimum triangulation angle (degrees) in re-triangulation.
    pub tri_min_angle: f64,
    /// Maximum number of point refinements after re-triangulation.
    pub retri_max_refinements: usize,
    /// Fraction of changed observations at which refinement iteration stops.
    pub retri_max_refinement_change: f64,
    /// Maximum iterations of the point-refinement BA in re-triangulation.
    pub retri_ba_max_num_iterations: usize,
    /// 번들 조정 풀이 백엔드(기본 CPU). BA 단계와 재삼각측량 끝의 전체 BA 에 쓴다.
    /// 재삼각측량 중 점만 정제하는 BA(`refine_points`)는 CPU 그대로다.
    pub ba_backend: BaBackendRef,
}

impl Default for GlobalSfmOptions {
    fn default() -> Self {
        Self {
            decompose_relative_pose: true,
            ba_num_iterations: 3,
            skip_rotation_averaging: false,
            skip_track_establishment: false,
            skip_global_positioning: false,
            skip_bundle_adjustment: false,
            skip_retriangulation: false,
            tracks: TrackOptions::default(),
            max_angular_reproj_error_deg: 1.0,
            max_normalized_reproj_error: 0.01,
            min_tri_angle_deg: 1.0,
            ra_max_rotation_error_deg: 10.0,
            rotation_averaging: RotationAveragingOptions::default(),
            positioning: PositionSolverOptions::default(),
            ba_refine_focal_length: true,
            ba_refine_principal_point: false,
            ba_refine_extra_params: true,
            // Design decision: default Huber scale of the global mapper BA is 1.0
            // (may differ if in normalized coordinates rather than pixels).
            ba_loss_scale: 1.0,
            ba_max_num_iterations: 200,
            ba_skip_fixed_rotation_stage: false,
            ba_skip_joint_optimization_stage: false,
            tri_complete_max_reproj_error: 15.0,
            tri_merge_max_reproj_error: 15.0,
            tri_min_angle: 1.0,
            retri_max_refinements: 5,
            retri_max_refinement_change: 0.0005,
            retri_ba_max_num_iterations: 50,
            ba_backend: BaBackendRef::default(),
        }
    }
}

impl GlobalSfmOptions {
    /// User script settings (ba_num_iterations 0, skip_retriangulation, keep_max_num_tracks 100000).
    pub fn script() -> Self {
        let mut o = Self { ba_num_iterations: 0, skip_retriangulation: true, ..Default::default() };
        o.tracks.max_num_tracks = 100_000;
        o
    }
}

/// Per-stage statistics.
#[derive(Clone, Debug, Default)]
pub struct GlobalMapperSummary {
    /// Number of view graph edges.
    pub num_view_graph_edges: usize,
    /// Number of valid edges after rotation averaging.
    pub num_valid_edges_after_ra: usize,
    /// Track construction statistics.
    pub tracks: TrackSummary,
    /// Position estimation statistics.
    pub positioning: PositioningSummary,
    /// Number of observations removed by the filter after position estimation.
    pub num_filtered_after_positioning: usize,
    /// Number of observations removed by the BA stage filter.
    pub num_filtered_after_ba: usize,
    /// Final number of registered images.
    pub registered_image_count: usize,
    /// Final number of 3D points.
    pub num_points3d: usize,
}

/// Result. Returned together with `failure` so the reconstruction up to that point stays usable even on failure.
#[derive(Clone, Debug)]
pub struct GlobalMapperOutput {
    /// Reconstruction built (up to the failure point, on failure).
    pub reconstruction: Reconstruction,
    /// View graph after rotation averaging.
    pub view_graph: ViewGraph,
    /// Per-stage statistics.
    pub summary: GlobalMapperSummary,
    /// Reason an intermediate stage failed. `None` means it succeeded to the end.
    pub failure: Option<String>,
}

/// Builds a pose-less reconstruction from the cameras and images in the store (those in the correspondence graph).
pub fn init_reconstruction(store: &FeatureStore, graph: &MatchGraph) -> Result<Reconstruction> {
    let mut rec = Reconstruction::new();
    let mut ids = graph.image_ids();
    ids.sort();
    for id in ids {
        let Some(si) = store.image(id) else { continue };
        if rec.camera(si.camera_id).is_none() {
            let cam = store.camera(si.camera_id).ok_or_else(|| cumulus3d_core::Error::NotFound(format!("카메라 {}", si.camera_id)))?;
            rec.add_camera_own_rig(cam)?;
        }
        let kps: Vec<Vec2> = store.keypoints(id).map(|k| k.iter().map(|p| Vec2::new(p.x as f64, p.y as f64)).collect()).unwrap_or_default();
        rec.add_image_own_frame(Image::new(id, si.name.clone(), si.camera_id, kps), None)?;
    }
    Ok(rec)
}

/// Builds the view graph by decomposing relative poses. Parallel per pair.
pub fn build_view_graph(rec: &Reconstruction, graph: &MatchGraph, decompose: bool) -> ViewGraph {
    let mut pairs = graph.image_pairs();
    pairs.sort();
    let edges: Vec<ViewGraphEdge> = pairs
        .par_iter()
        .filter_map(|&(pid, n)| {
            let (a, b) = images_of_pair(pid);
            let mut tvg = graph.two_view_geometry(a, b)?;
            if tvg.cam1_to_cam2.is_none() {
                if !decompose {
                    return None;
                }
                let (ia, ib) = (rec.image(a)?, rec.image(b)?);
                let (ca, cb) = (rec.camera(ia.camera_id)?, rec.camera(ib.camera_id)?);
                tvg.inlier_matches = graph.matches_between(a, b);
                if tvg.inlier_matches.is_empty() {
                    return None;
                }
                let kp = |im: &Image| -> Vec<cumulus3d_core::Keypoint> {
                    im.points2d().iter().map(|p| cumulus3d_core::Keypoint::new(p.xy.x as f32, p.xy.y as f32)).collect()
                };
                // Design decision: keypoints are converted back from f64 model coordinates to f32 for passing on
                // (they were f32 originally, so no loss).
                if !cumulus3d_matching::pose::refit_and_estimate_relative_pose(ca, cb, &kp(ia), &kp(ib), &mut tvg) {
                    return None;
                }
            }
            Some(ViewGraphEdge { image_id1: a, image_id2: b, cam1_to_cam2: tvg.cam1_to_cam2?, num_matches: n, valid: true })
        })
        .collect();
    ViewGraph::new(edges)
}

/// One round of rotation averaging. If `posed_only`, only frames that have poses become nodes.
pub fn rotation_averaging_round(rec: &mut Reconstruction, vg: &mut ViewGraph, posed_only: bool, opts: &GlobalSfmOptions) -> Result<()> {
    let nodes = if posed_only {
        let posed: HashSet<ImageId> = rec.image_ids().filter(|i| rec.has_pose(*i)).collect();
        vg.largest_connected_component(|i| posed.contains(&i))
    } else {
        vg.largest_connected_component(|_| true)
    };
    vg.invalidate_outside(&nodes);
    if nodes.is_empty() {
        return Err(cumulus3d_core::Error::InvalidArgument("회전 평균: 연결 성분 없음".into()));
    }
    let (rots, _) = solve_rotation_averaging(&nodes, vg, None, &opts.rotation_averaging)?;
    for (id, r) in &rots {
        let t = Vec3::new(f64::NAN, f64::NAN, f64::NAN);
        rec.set_world_to_cam(*id, Rigid3::from_rotation_matrix(r, t))?;
        rec.register_image(*id)?;
    }
    vg.filter_by_relative_rotation(&rots, opts.ra_max_rotation_error_deg);
    let posed: HashSet<ImageId> = rec.image_ids().filter(|i| rec.has_pose(*i)).collect();
    let cc = vg.largest_connected_component(|i| posed.contains(&i));
    vg.invalidate_outside(&cc);
    for id in rec.registered_images() {
        if !cc.contains(&id) {
            rec.deregister_image(id)?;
        }
    }
    // Clean up images that are not registered but still have poses.
    let stray: Vec<ImageId> = rec.image_ids().filter(|i| rec.has_pose(*i) && !rec.is_image_registered(*i)).collect();
    for id in stray {
        let fid = rec.image(id).expect("존재").frame_id;
        rec.set_frame_pose(fid, None)?;
    }
    Ok(())
}

fn angular_error_deg(cam: &Camera, pose: &Rigid3, xy: &Vec2, x: &Vec3) -> f64 {
    let Some(r) = cam.img_to_ray(xy) else { return f64::INFINITY };
    let xc = pose.transform_point(x);
    let n = xc.norm();
    if n == 0.0 || !n.is_finite() {
        return f64::INFINITY;
    }
    (r.dot(&xc) / n).clamp(-1.0, 1.0).acos().to_degrees()
}

fn normalized_error(cam: &Camera, pose: &Rigid3, xy: &Vec2, x: &Vec3) -> f64 {
    let xc = pose.transform_point(x);
    if xc.z.is_nan() || xc.z < 1e-12 {
        return f64::INFINITY;
    }
    let Some(uv) = cam.img_to_normalized(xy) else { return f64::INFINITY };
    (Vec2::new(xc.x / xc.z, xc.y / xc.z) - uv).norm()
}

/// Angular error filter ("length−1" rule, point error ← mean of remaining observations (degrees)).
/// Returns: number of deleted observations.
pub fn filter_angular_error(rec: &mut Reconstruction, max_deg: f64) -> usize {
    rec.filter_observations(None, max_deg, angular_error_deg, FilterErrorUpdate::MeanOfRemaining)
}

/// Only for cameras with a focal prior, delete observations exceeding the angular error one by one.
pub fn filter_angular_error_prior_focal(rec: &mut Reconstruction, max_deg: f64) -> usize {
    let mut del = Vec::new();
    for (_, p) in rec.points3d() {
        for t in &p.track {
            let Some(im) = rec.image(t.image_id) else { continue };
            let Some(cam) = rec.camera(im.camera_id) else { continue };
            if !cam.focal_from_prior {
                continue;
            }
            let Some(pose) = rec.world_to_cam(t.image_id) else { continue };
            let e = angular_error_deg(cam, &pose, &im.point2d(t.point2d_idx).xy, &p.xyz);
            if e.is_nan() || e > max_deg {
                del.push(*t);
            }
        }
    }
    let mut n = 0;
    for t in del {
        if rec.image(t.image_id).is_some_and(|im| im.point2d(t.point2d_idx).has_point3d()) {
            let _ = rec.delete_observation(t.image_id, t.point2d_idx);
            n += 1;
        }
    }
    n
}

/// Normalized reprojection error filter ("length−1" rule). Returns: number of deleted observations.
pub fn filter_normalized_reproj_error(rec: &mut Reconstruction, max_err: f64) -> usize {
    rec.filter_observations(None, max_err, normalized_error, FilterErrorUpdate::MeanOfRemaining)
}

/// Filter and normalization right after position estimation.
pub fn post_positioning_filters(rec: &mut Reconstruction, opts: &GlobalSfmOptions) -> usize {
    let mut n = filter_angular_error(rec, 2.0 * opts.max_angular_reproj_error_deg);
    n += filter_angular_error_prior_focal(rec, opts.max_angular_reproj_error_deg);
    n += rec.filter_points3d_with_small_triangulation_angle(opts.min_tri_angle_deg, None);
    n += filter_normalized_reproj_error(rec, 10.0 * opts.max_normalized_reproj_error);
    rec.normalize(&NormalizeOptions::default());
    n
}

fn ba_config(rec: &Reconstruction, opts: &GlobalSfmOptions, fix_rotation: bool) -> BaConfig {
    BaConfig {
        refine_focal_length: opts.ba_refine_focal_length && !fix_rotation,
        refine_principal_point: opts.ba_refine_principal_point && !fix_rotation,
        refine_extra_params: opts.ba_refine_extra_params && !fix_rotation,
        refine_poses: true,
        refine_points: true,
        loss: Loss::Huber(opts.ba_loss_scale),
        max_num_iterations: opts.ba_max_num_iterations,
        function_tolerance: 1e-5,
        images: HashSet::new(),
        constant_poses: HashSet::new(),
        constant_cameras: if fix_rotation { rec.cameras().keys().copied().collect() } else { HashSet::new() },
        constant_points: HashSet::new(),
        auto_gauge: true,
        constant_world_to_rig_rotation: fix_rotation,
        min_track_length: opts.tracks.min_num_views_per_track,
        ..Default::default()
    }
}

/// Iterative BA stage. The two final filters run even when `ba_num_iterations` = 0.
pub fn bundle_adjustment_stage(rec: &mut Reconstruction, opts: &GlobalSfmOptions) -> Result<usize> {
    let mut removed = 0;
    for it in 0..opts.ba_num_iterations {
        if !opts.ba_skip_fixed_rotation_stage {
            cumulus3d_ba::bundle_adjust_with(rec, &ba_config(rec, opts, true), opts.ba_backend.get())?;
        }
        if !opts.ba_skip_joint_optimization_stage {
            cumulus3d_ba::bundle_adjust_with(rec, &ba_config(rec, opts, false), opts.ba_backend.get())?;
        }
        rec.normalize(&NormalizeOptions::default());
        // If removal is at most 0.1% of the point count, raise the iteration number and re-filter more strictly;
        // stop early if it stays small to the end.
        let mut k = it;
        let mut stop = false;
        loop {
            let thr = (3usize.saturating_sub(k)).max(1) as f64 * opts.max_normalized_reproj_error;
            let n = filter_normalized_reproj_error(rec, thr);
            removed += n;
            if n as f64 > 0.001 * rec.num_points3d().max(1) as f64 {
                break;
            }
            k += 1;
            if k >= opts.ba_num_iterations {
                stop = true;
                break;
            }
        }
        if stop {
            break;
        }
    }
    removed += filter_normalized_reproj_error(rec, opts.max_normalized_reproj_error);
    removed += rec.filter_points3d_with_small_triangulation_angle(opts.min_tri_angle_deg, None);
    Ok(removed)
}

/// Re-triangulation stage.
pub fn retriangulation_stage(rec: &mut Reconstruction, graph: &MatchGraph, opts: &GlobalSfmOptions) -> Result<()> {
    rec.delete_all_points3d();
    let tri_opts = TrackTriangulatorOptions {
        complete_max_reproj_error: opts.tri_complete_max_reproj_error,
        merge_max_reproj_error: opts.tri_merge_max_reproj_error,
        min_angle: opts.tri_min_angle,
        ..Default::default()
    };
    let mut tri = TrackTriangulator::new(rec, graph, tri_opts);
    for id in rec.registered_images() {
        tri.triangulate_image(rec, id)?;
    }
    tri.complete_tracks(rec, None)?;
    tri.merge_tracks(rec, None)?;
    for _ in 0..opts.retri_max_refinements {
        let num_obs = rec.total_observations().max(1);
        refine_points(rec, None, PointRefiner::BundleAdjuster, opts.retri_ba_max_num_iterations)?;
        let mut changed = tri.complete_tracks(rec, None)?;
        changed += tri.merge_tracks(rec, None)?;
        changed += filter_normalized_reproj_error(rec, opts.max_normalized_reproj_error);
        if (changed as f64) / (num_obs as f64) < opts.retri_max_refinement_change {
            break;
        }
    }
    filter_normalized_reproj_error(rec, opts.max_normalized_reproj_error);
    rec.filter_points3d_with_small_triangulation_angle(opts.min_tri_angle_deg, None);
    cumulus3d_ba::bundle_adjust_with(rec, &ba_config(rec, opts, false), opts.ba_backend.get())?;
    rec.normalize(&NormalizeOptions::default());
    filter_normalized_reproj_error(rec, opts.max_normalized_reproj_error);
    rec.filter_points3d_with_small_triangulation_angle(opts.min_tri_angle_deg, None);
    Ok(())
}

/// Global mapper run. Color extraction is done by the caller with `Reconstruction::extract_colors`.
pub fn global_mapper(store: &FeatureStore, graph: &MatchGraph, opts: &GlobalSfmOptions) -> Result<GlobalMapperOutput> {
    let mut rec = init_reconstruction(store, graph)?;
    let mut vg = build_view_graph(&rec, graph, opts.decompose_relative_pose);
    let mut summary = GlobalMapperSummary { num_view_graph_edges: vg.edges.len(), ..Default::default() };
    let failure = run_stages(&mut rec, graph, &mut vg, opts, &mut summary).err().map(|e| e.to_string());
    rec.update_point3d_errors();
    summary.registered_image_count = rec.registered_image_count();
    summary.num_points3d = rec.num_points3d();
    Ok(GlobalMapperOutput { reconstruction: rec, view_graph: vg, summary, failure })
}

fn run_stages(
    rec: &mut Reconstruction,
    graph: &MatchGraph,
    vg: &mut ViewGraph,
    opts: &GlobalSfmOptions,
    summary: &mut GlobalMapperSummary,
) -> Result<()> {
    if vg.num_valid_edges() == 0 {
        return Err(cumulus3d_core::Error::InvalidArgument("뷰 그래프 간선 없음".into()));
    }
    if !opts.skip_rotation_averaging {
        rotation_averaging_round(rec, vg, false, opts)?;
        rotation_averaging_round(rec, vg, true, opts)?;
    }
    summary.num_valid_edges_after_ra = vg.num_valid_edges();
    if !opts.skip_track_establishment {
        summary.tracks = establish_tracks(rec, graph, vg, &opts.tracks)?;
    }
    if !opts.skip_global_positioning {
        summary.positioning = global_positioning(rec, &opts.positioning)?;
        summary.num_filtered_after_positioning = post_positioning_filters(rec, opts);
    }
    if !opts.skip_bundle_adjustment {
        summary.num_filtered_after_ba = bundle_adjustment_stage(rec, opts)?;
    }
    if !opts.skip_retriangulation {
        retriangulation_stage(rec, graph, opts)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cumulus3d_core::{CameraModelKind, Point3D, TrackEntry};

    fn two_cam_rec(err_deg: f64) -> Reconstruction {
        // Two cameras and one point, with the second image observation rotated by err_deg.
        let mut rec = Reconstruction::new();
        let mut cam = Camera::new(1, CameraModelKind::Pinhole, 1000, 1000, vec![1000.0, 1000.0, 500.0, 500.0]).unwrap();
        cam.focal_from_prior = true;
        rec.add_camera_own_rig(cam.clone()).unwrap();
        let x = Vec3::new(0.0, 0.0, 10.0);
        let poses = [
            Rigid3::new(cumulus3d_core::Quat::IDENTITY, Vec3::new(1.0, 0.0, 0.0)),
            Rigid3::new(cumulus3d_core::Quat::IDENTITY, Vec3::new(-1.0, 0.0, 0.0)),
            Rigid3::new(cumulus3d_core::Quat::IDENTITY, Vec3::new(0.0, 1.0, 0.0)),
        ];
        for (i, p) in poses.iter().enumerate() {
            let mut xc = *p * x;
            if i == 1 {
                xc = crate::math::so3_exp(&(xc.cross(&Vec3::y()).normalize() * err_deg.to_radians())) * xc;
            }
            let xy = cam.cam_to_img(&xc).unwrap();
            let id = i as u32 + 1;
            rec.add_image_own_frame(Image::new(id, format!("{id}"), 1, vec![xy]), Some(*p)).unwrap();
            rec.register_image(id).unwrap();
        }
        let track = (1..=3).map(|i| TrackEntry::new(i, 0)).collect();
        rec.add_point3d_with_id(0, Point3D { xyz: x, color: [0; 3], error: -1.0, track }).unwrap();
        rec
    }

    #[test]
    fn angular_filter_boundary() {
        for (deg, keep) in [(2.0 - 1e-6, true), (2.0 + 1e-6, false)] {
            let mut rec = two_cam_rec(deg);
            filter_angular_error(&mut rec, 2.0);
            let p = rec.point3d(0).unwrap();
            assert_eq!(p.track.len() == 3, keep, "deg {deg}");
        }
        for (deg, keep) in [(1.0 - 1e-6, true), (1.0 + 1e-6, false)] {
            let mut rec = two_cam_rec(deg);
            filter_angular_error_prior_focal(&mut rec, 1.0);
            assert_eq!(rec.point3d(0).unwrap().track.len() == 3, keep);
        }
    }

    #[test]
    fn normalized_filter_boundary_and_point_deletion() {
        // Angular error θ gives normalized coordinate error ≈ tan difference. Two cases near the threshold.
        let rec = two_cam_rec(0.0);
        let x = rec.point3d(0).unwrap().xyz;
        let pose = rec.world_to_cam(2).unwrap();
        let cam = rec.camera(1).unwrap().clone();
        let e = normalized_error(&cam, &pose, &rec.image(2).unwrap().point2d(0).xy, &x);
        assert!(e < 1e-12);
        // If two observations are bad (≥ length−1), the whole point is deleted.
        let mut rec = two_cam_rec(0.0);
        rec.set_world_to_cam(2, Rigid3::new(cumulus3d_core::Quat::IDENTITY, Vec3::new(-1.0, 3.0, 0.0))).unwrap();
        rec.set_world_to_cam(3, Rigid3::new(cumulus3d_core::Quat::IDENTITY, Vec3::new(0.0, 4.0, 0.0))).unwrap();
        filter_normalized_reproj_error(&mut rec, 0.1);
        assert_eq!(rec.num_points3d(), 0);
        // Boundary: 0.1 ± 1e-9 (move the first image observation exactly in normalized coordinates).
        for (d, keep) in [(0.1 - 1e-9, true), (0.1 + 1e-9, false)] {
            let mut rec = two_cam_rec(0.0);
            let shifted = Rigid3::new(cumulus3d_core::Quat::IDENTITY, Vec3::new(1.0 + d * 10.0, 0.0, 0.0));
            rec.set_world_to_cam(1, shifted).unwrap();
            filter_normalized_reproj_error(&mut rec, 0.1);
            assert_eq!(rec.point3d(0).unwrap().track.len() == 3, keep, "d {d}");
        }
    }

    #[test]
    fn normalization_extent() {
        let mut rec = two_cam_rec(0.0);
        for i in 4..20u32 {
            let p = Rigid3::new(cumulus3d_core::Quat::IDENTITY, Vec3::new(i as f64 * 1.7, (i * i % 7) as f64, 0.3 * i as f64));
            rec.add_image_own_frame(Image::new(i, format!("{i}"), 1, vec![]), Some(p)).unwrap();
            rec.register_image(i).unwrap();
        }
        rec.normalize(&NormalizeOptions::default());
        let cs: Vec<Vec3> = rec.registered_images().iter().map(|i| rec.projection_center(*i).unwrap()).collect();
        let n = cs.len();
        let e = n - 1;
        let lo = (0.1 * e as f64).floor() as usize;
        let hi = (0.9 * e as f64).ceil() as usize;
        let mut bmin = Vec3::zeros();
        let mut bmax = Vec3::zeros();
        let mut c = Vec3::zeros();
        for ax in 0..3 {
            let mut v: Vec<f64> = cs.iter().map(|p| p[ax]).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            bmin[ax] = v[lo];
            bmax[ax] = v[hi];
            c[ax] = v[lo..=hi].iter().sum::<f64>() / (hi - lo + 1) as f64;
        }
        assert!(((bmax - bmin).norm() - 10.0).abs() < 1e-9);
        assert!(c.norm() < 1e-9);
    }
}
