/*
 * triangulator.rs
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

//! Triangulation that keeps existing points (point_triangulator): extension, creation, completion, merging, re-triangulation,
//! the pose-fixed point refinement loop, and the filters.

use crate::triangulation::{
    angular_error, estimate_triangulation, CpuTriangulation, ObsQuery, TriObservation, TriProblem, TriangulationBackend,
    TriangulationRansacParams,
};
use cumulus3d_ba::{BaConfig, Loss};
use cumulus3d_core::graph::Correspondence;
use cumulus3d_core::reconstruction::FilterErrorUpdate;
use cumulus3d_core::{
    images_of_pair, Camera, CameraId, Error, ImageId, Mat3x4, MatchGraph, Point2DIdx, Point3DId, Reconstruction, Result, Rigid3,
    TrackEntry, Vec3, INVALID_POINT3D_ID,
};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Incremental triangulator options.
#[derive(Clone, Debug)]
pub struct TrackTriangulatorOptions {
    /// Correspondence-graph transitive search depth when creating new points.
    pub max_transitivity: usize,
    /// Angular error tolerance (degrees) when creating new points.
    pub create_angle_tol: f64,
    /// Angular error tolerance (degrees) when extending existing points.
    pub continue_angle_tol: f64,
    /// Upper bound on the reprojection error (pixels) for merging tracks.
    pub merge_max_reproj_error: f64,
    /// Upper bound on the reprojection error (pixels) for completing tracks.
    pub complete_max_reproj_error: f64,
    /// Transitive search depth when completing tracks.
    pub complete_transitivity: usize,
    /// Angular error tolerance (degrees) for re-triangulation.
    pub retri_angle_tol: f64,
    /// Re-triangulate a pair only when its triangulated ratio is below this.
    pub retri_min_ratio: f64,
    /// Maximum re-triangulation attempts per pair.
    pub retri_max_attempts: usize,
    /// Minimum triangulation angle (degrees).
    pub min_angle: f64,
    /// Whether to skip tracks that have only two images.
    pub skip_two_view_tracks: bool,
    /// Abnormal camera test: lower bound of the focal ratio.
    pub min_focal_length_ratio: f64,
    /// Abnormal camera test: upper bound of the focal ratio.
    pub max_focal_length_ratio: f64,
    /// Abnormal camera test: upper bound of the absolute value of the extra parameters.
    pub max_extra_param: f64,
}

impl Default for TrackTriangulatorOptions {
    fn default() -> Self {
        Self {
            max_transitivity: 1,
            create_angle_tol: 2.0,
            continue_angle_tol: 2.0,
            merge_max_reproj_error: 4.0,
            complete_max_reproj_error: 4.0,
            complete_transitivity: 5,
            retri_angle_tol: 5.0,
            retri_min_ratio: 0.2,
            retri_max_attempts: 1,
            min_angle: 1.5,
            skip_two_view_tracks: true,
            min_focal_length_ratio: 0.1,
            max_focal_length_ratio: 10.0,
            max_extra_param: 1.0,
        }
    }
}

/// Point refinement method (BA with fixed poses and intrinsics).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointRefiner {
    /// `bundle_adjust` of cumulus3d-ba (poses and cameras constant, only points variable). Default behavior.
    BundleAdjuster,
    /// Independent 3-variable LM per point (the same problem as BA when pose and intrinsics are fixed).
    PerPoint,
    /// Skip refinement.
    None,
}

/// Processing scope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TriangulationScope {
    /// Default behavior: re-triangulate all registered images, complete, merge and filter all points.
    AllRegistered,
    /// Incremental: only the given (newly registered) images, their correspondences, and the points changed by them.
    Images(Vec<ImageId>),
}

/// point_triangulator options.
#[derive(Clone, Debug)]
pub struct PointTriangulatorOptions {
    /// Incremental triangulator options.
    pub tri: TrackTriangulatorOptions,
    /// If true, delete all 3D points before starting (false in the pipeline).
    pub clear_points: bool,
    /// Upper bound on the reprojection error (pixels) of the final filter.
    pub filter_max_reproj_error: f64,
    /// Minimum triangulation angle (degrees) of the final filter.
    pub filter_min_tri_angle: f64,
    /// Maximum number of point refinement / re-triangulation iterations.
    pub ba_global_max_refinements: usize,
    /// Fraction of changed observations at which iteration stops.
    pub ba_global_max_refinement_change: f64,
    /// Maximum iterations of the point-refinement BA.
    pub ba_global_max_num_iterations: usize,
    /// Point refinement method.
    pub refiner: PointRefiner,
    /// Processing scope (all, or the new images).
    pub scope: TriangulationScope,
}

impl Default for PointTriangulatorOptions {
    fn default() -> Self {
        Self {
            tri: TrackTriangulatorOptions::default(),
            clear_points: false,
            filter_max_reproj_error: 4.0,
            filter_min_tri_angle: 1.5,
            ba_global_max_refinements: 5,
            ba_global_max_refinement_change: 0.0005,
            ba_global_max_num_iterations: 50,
            refiner: PointRefiner::BundleAdjuster,
            scope: TriangulationScope::AllRegistered,
        }
    }
}

/// Run statistics.
#[derive(Clone, Debug, Default)]
pub struct TriangulationReport {
    /// Number of newly created points.
    pub num_created: usize,
    /// Number of observations appended to existing points.
    pub num_continued: usize,
    /// Number of observations added by track completion.
    pub num_completed: usize,
    /// Sum of the observation counts of merged points.
    pub num_merged: usize,
    /// Number of observations created by re-triangulation.
    pub num_retriangulated: usize,
    /// Number of observations and points removed by the filter.
    pub num_filtered: usize,
    /// Number of refinement iterations performed.
    pub num_refinements: usize,
}

#[derive(Clone)]
struct PoseInfo {
    pose: Rigid3,
    proj: Mat3x4,
    center: Vec3,
    camera: Camera,
    rays: Arc<Vec<Option<Vec3>>>,
}

type PointPlan = (Vec<(Correspondence, Point3DId)>, Vec<Action>);

enum Action {
    Continue(Point3DId, TrackEntry),
    Create(Vec3, Vec<TrackEntry>),
}

/// RANSAC triangulation result (point, inlier mask).
type TriResult = Option<(Vec3, Vec<bool>)>;

/// Read-only preparation of one 2D point (extension applied, right before the first creation RANSAC).
struct Prepared {
    snapshot: Vec<(Correspondence, Point3DId)>,
    state: HashMap<Correspondence, Point3DId>,
    actions: Vec<Action>,
    input: Vec<Correspondence>,
    /// Observations of the first creation RANSAC (None if it does not reach the creation step).
    first: Option<Vec<TriObservation>>,
}

/// Incremental triangulator. At creation it caches the poses and back-projection rays of the registered images (assumes poses are fixed).
pub struct TrackTriangulator<'a> {
    graph: &'a MatchGraph,
    /// Backend that solves creation RANSAC in batches.
    backend: &'a dyn TriangulationBackend,
    /// Triangulation options.
    pub opts: TrackTriangulatorOptions,
    poses: HashMap<ImageId, PoseInfo>,
    merge_trials: HashSet<(Point3DId, Point3DId)>,
    re_trials: HashMap<(ImageId, ImageId), usize>,
    /// Points created or changed in this run.
    pub touched: HashSet<Point3DId>,
    /// Accumulated run statistics.
    pub report: TriangulationReport,
}

impl<'a> TrackTriangulator<'a> {
    /// Builds a triangulator by caching the poses and rays of the registered images of the reconstruction (CPU backend).
    pub fn new(rec: &Reconstruction, graph: &'a MatchGraph, opts: TrackTriangulatorOptions) -> Self {
        Self::with_backend(rec, graph, opts, &CpuTriangulation)
    }

    /// Like [`new`](Self::new), but creation RANSAC is solved in batches by `backend`.
    pub fn with_backend(
        rec: &Reconstruction,
        graph: &'a MatchGraph,
        opts: TrackTriangulatorOptions,
        backend: &'a dyn TriangulationBackend,
    ) -> Self {
        let ids = rec.registered_images();
        let infos: Vec<(ImageId, PoseInfo)> = ids
            .par_iter()
            .filter_map(|&id| {
                let im = rec.image(id)?;
                let cam = rec.camera(im.camera_id)?;
                if cam.has_implausible_params(opts.min_focal_length_ratio, opts.max_focal_length_ratio, opts.max_extra_param) {
                    return None;
                }
                let pose = rec.world_to_cam(id)?;
                let rays = im.points2d().iter().map(|p| cam.img_to_ray(&p.xy)).collect();
                Some((id, PoseInfo { pose, proj: pose.matrix(), center: pose.center(), camera: cam.clone(), rays: Arc::new(rays) }))
            })
            .collect();
        Self {
            graph,
            backend,
            opts,
            poses: infos.into_iter().collect(),
            merge_trials: HashSet::new(),
            re_trials: HashMap::new(),
            touched: HashSet::new(),
            report: TriangulationReport::default(),
        }
    }

    fn ray(&self, c: &Correspondence) -> Option<Vec3> {
        self.poses.get(&c.image_id)?.rays.get(c.point2d_idx as usize).copied().flatten()
    }

    fn corrs(&self, image_id: ImageId, idx: Point2DIdx) -> Vec<Correspondence> {
        let v = if self.opts.max_transitivity <= 1 {
            self.graph.find_correspondences(image_id, idx).to_vec()
        } else {
            self.graph.transitive_matches(image_id, idx, self.opts.max_transitivity)
        };
        v.into_iter().filter(|c| self.poses.contains_key(&c.image_id)).collect()
    }

    fn point_of(rec: &Reconstruction, c: &Correspondence) -> Point3DId {
        rec.image(c.image_id).and_then(|im| im.points2d().get(c.point2d_idx as usize)).map_or(INVALID_POINT3D_ID, |p| p.point3d_id)
    }

    /// Plans one 2D point of one image read-only (extend, then create). Returns: (referenced state, action).
    fn plan_point(&self, rec: &Reconstruction, image_id: ImageId, idx: Point2DIdx) -> Option<PointPlan> {
        let Prepared { snapshot, mut state, mut actions, input, .. } = self.prepare_point(rec, image_id, idx)?;
        self.plan_create(&input, &mut state, &mut actions, self.opts.create_angle_tol);
        Some((snapshot, actions))
    }

    /// First part of [`plan_point`](Self::plan_point): plans up to extension and gathers the observations of the first creation RANSAC.
    fn prepare_point(&self, rec: &Reconstruction, image_id: ImageId, idx: Point2DIdx) -> Option<Prepared> {
        let corrs = self.corrs(image_id, idx);
        if corrs.is_empty() {
            return None;
        }
        let me = Correspondence::new(image_id, idx);
        let mut input = corrs.clone();
        input.push(me);
        let snapshot: Vec<(Correspondence, Point3DId)> = input.iter().map(|c| (*c, Self::point_of(rec, c))).collect();
        let mut state: HashMap<Correspondence, Point3DId> = snapshot.iter().copied().collect();
        let mut actions = Vec::new();
        let n_with = corrs.iter().filter(|c| state[c] != INVALID_POINT3D_ID).count();
        if n_with > 0 && state[&me] == INVALID_POINT3D_ID {
            if let Some(ray) = self.ray(&me) {
                let pi = &self.poses[&image_id];
                let mut best: Option<(f64, Point3DId)> = None;
                for c in &corrs {
                    let pid = state[c];
                    if pid == INVALID_POINT3D_ID {
                        continue;
                    }
                    let Some(p) = rec.point3d(pid) else { continue };
                    let err = angular_error(&pi.proj, &ray, &p.xyz);
                    if best.is_none_or(|(e, _)| err < e) {
                        best = Some((err, pid));
                    }
                }
                if let Some((err, pid)) = best {
                    if err <= self.opts.continue_angle_tol.to_radians() {
                        actions.push(Action::Continue(pid, TrackEntry::new(image_id, idx)));
                        state.insert(me, pid);
                    }
                }
            }
        }
        let first = self.create_round(&input, &state).map(|(_, obs)| obs);
        Some(Prepared { snapshot, state, actions, input, first })
    }

    fn create_params(&self, angle_limit_deg: f64) -> TriangulationRansacParams {
        TriangulationRansacParams { max_angle_error_deg: angle_limit_deg, min_tri_angle_deg: self.opts.min_angle, ..Default::default() }
    }

    /// Input of one creation round: correspondences without a point yet and their observations. None if it does not go to creation.
    fn create_round(
        &self,
        input: &[Correspondence],
        state: &HashMap<Correspondence, Point3DId>,
    ) -> Option<(Vec<Correspondence>, Vec<TriObservation>)> {
        let sel: Vec<Correspondence> =
            input.iter().filter(|c| state.get(c).copied().unwrap_or(INVALID_POINT3D_ID) == INVALID_POINT3D_ID).copied().collect();
        if sel.len() < 2 {
            return None;
        }
        if self.opts.skip_two_view_tracks && sel.len() == 2 && self.graph.in_two_view_track(sel[0].image_id, sel[0].point2d_idx) {
            return None;
        }
        let mut obs = Vec::with_capacity(sel.len());
        let mut elems = Vec::with_capacity(sel.len());
        for c in &sel {
            let (Some(pi), Some(r)) = (self.poses.get(&c.image_id), self.ray(c)) else { continue };
            obs.push(TriObservation { proj: pi.proj, center: pi.center, ray: r });
            elems.push(*c);
        }
        if obs.len() < 2 {
            return None;
        }
        Some((elems, obs))
    }

    /// Dry-run of creation: marks a new point in `state` with a virtual id.
    fn plan_create(
        &self,
        input: &[Correspondence],
        state: &mut HashMap<Correspondence, Point3DId>,
        actions: &mut Vec<Action>,
        angle_limit_deg: f64,
    ) {
        self.plan_create_with(input, state, actions, angle_limit_deg, None);
    }

    /// Like [`plan_create`](Self::plan_create), but the RANSAC result of the first round may be given in advance as `first`
    /// (solved in a batch by the backend). Later rounds (rare) are solved on the CPU.
    fn plan_create_with(
        &self,
        input: &[Correspondence],
        state: &mut HashMap<Correspondence, Point3DId>,
        actions: &mut Vec<Action>,
        angle_limit_deg: f64,
        mut first: Option<TriResult>,
    ) {
        loop {
            let Some((elems, obs)) = self.create_round(input, state) else { return };
            let res = match first.take() {
                Some(r) => r,
                None => estimate_triangulation(&obs, &self.create_params(angle_limit_deg)),
            };
            let Some((x, mask)) = res else { return };
            let track: Vec<TrackEntry> =
                elems.iter().zip(&mask).filter(|(_, m)| **m).map(|(c, _)| TrackEntry::new(c.image_id, c.point2d_idx)).collect();
            let len = track.len();
            for t in &track {
                state.insert(Correspondence::new(t.image_id, t.point2d_idx), INVALID_POINT3D_ID - 1);
            }
            actions.push(Action::Create(x, track));
            // Design decision: "number of input observations" is interpreted as the total input count,
            // regardless of whether a 3D point exists.
            if input.len() < len + 3 {
                return;
            }
        }
    }

    fn apply(&mut self, rec: &mut Reconstruction, actions: Vec<Action>) -> Result<usize> {
        let mut n = 0;
        for a in actions {
            match a {
                Action::Continue(pid, e) => {
                    if rec.exists_point3d(pid) && Self::point_of(rec, &Correspondence::new(e.image_id, e.point2d_idx)) == INVALID_POINT3D_ID
                    {
                        rec.add_observation(pid, e)?;
                        self.touched.insert(pid);
                        self.report.num_continued += 1;
                        n += 1;
                    }
                }
                Action::Create(x, track) => {
                    let id = rec.add_point3d(x, track.clone(), [0, 0, 0])?;
                    self.touched.insert(id);
                    self.report.num_created += 1;
                    n += track.len();
                }
            }
        }
        Ok(n)
    }

    /// Solves the first creation RANSAC of the prepared points in one backend call (in preparation order, None for points that
    /// do not go to creation).
    fn estimate_firsts(&self, preps: &[Option<Prepared>]) -> Vec<Option<TriResult>> {
        let params = self.create_params(self.opts.create_angle_tol);
        let problems: Vec<TriProblem<'_>> =
            preps.iter().filter_map(|p| p.as_ref()?.first.as_deref()).map(|obs| TriProblem { obs, params: params.clone() }).collect();
        if problems.is_empty() {
            return preps.iter().map(|_| None).collect();
        }
        let results = self.backend.estimate_batch(&problems);
        assert_eq!(results.len(), problems.len(), "삼각측량 백엔드 결과 수");
        let mut it = results.into_iter();
        preps.iter().map(|p| p.as_ref().and_then(|p| p.first.as_ref()).map(|_| it.next().expect("결과 수"))).collect()
    }

    /// Triangulates one image. Per-2D-point planning is parallel, application is sequential with state validation
    /// (re-plan if the referenced state changed after planning) so the result equals the sequential run.
    pub fn triangulate_image(&mut self, rec: &mut Reconstruction, image_id: ImageId) -> Result<usize> {
        if !self.poses.contains_key(&image_id) {
            return Ok(0);
        }
        let n = rec.image(image_id).map_or(0, |im| im.num_points2d());
        let plans: Vec<Option<PointPlan>> = {
            let rec_ref: &Reconstruction = rec;
            let this: &Self = self;
            let preps: Vec<Option<Prepared>> =
                (0..n as Point2DIdx).into_par_iter().map(|i| this.prepare_point(rec_ref, image_id, i)).collect();
            // The first creation RANSAC of all 2D points in a single backend call.
            let firsts = this.estimate_firsts(&preps);
            preps
                .into_par_iter()
                .zip(firsts.into_par_iter())
                .map(|(p, f)| {
                    let Prepared { snapshot, mut state, mut actions, input, .. } = p?;
                    this.plan_create_with(&input, &mut state, &mut actions, this.opts.create_angle_tol, f);
                    Some((snapshot, actions))
                })
                .collect()
        };
        let mut total = 0;
        for (i, plan) in plans.into_iter().enumerate() {
            let Some((snap, actions)) = plan else { continue };
            let still = snap.iter().all(|(c, pid)| Self::point_of(rec, c) == *pid);
            let actions = if still {
                actions
            } else {
                match self.plan_point(rec, image_id, i as Point2DIdx) {
                    Some((_, a)) => a,
                    None => continue,
                }
            };
            total += self.apply(rec, actions)?;
        }
        Ok(total)
    }

    fn reproj_ok(&self, image_id: ImageId, xy: &cumulus3d_core::Vec2, x: &Vec3, max_px: f64) -> bool {
        match self.poses.get(&image_id) {
            Some(pi) => Reconstruction::squared_reprojection_error(&pi.pose, &pi.camera, xy, x) <= max_px * max_px,
            None => false,
        }
    }

    /// Track completion. Returns: number of observations added.
    pub fn complete_track(&mut self, rec: &mut Reconstruction, pid: Point3DId) -> Result<usize> {
        let Some(p) = rec.point3d(pid) else { return Ok(0) };
        let x = p.xyz;
        let mut queue: Vec<Correspondence> = p.track.iter().map(|e| Correspondence::new(e.image_id, e.point2d_idx)).collect();
        let mut visited: HashSet<Correspondence> = queue.iter().copied().collect();
        let mut added = 0;
        for _ in 0..self.opts.complete_transitivity {
            let mut next = Vec::new();
            for e in &queue {
                for c in self.graph.find_correspondences(e.image_id, e.point2d_idx) {
                    if !visited.insert(*c) || !self.poses.contains_key(&c.image_id) {
                        continue;
                    }
                    let Some(im) = rec.image(c.image_id) else { continue };
                    let p2 = im.point2d(c.point2d_idx);
                    if p2.has_point3d() {
                        continue;
                    }
                    if self.reproj_ok(c.image_id, &p2.xy, &x, self.opts.complete_max_reproj_error) {
                        rec.add_observation(pid, TrackEntry::new(c.image_id, c.point2d_idx))?;
                        next.push(*c);
                        added += 1;
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            queue = next;
        }
        if added > 0 {
            self.touched.insert(pid);
            self.report.num_completed += added;
        }
        Ok(added)
    }

    /// Track merging. Returns: number of observations involved in the merge (including recursion).
    pub fn merge_track(&mut self, rec: &mut Reconstruction, pid: Point3DId) -> Result<usize> {
        let Some(a) = rec.point3d(pid).cloned() else { return Ok(0) };
        for e in &a.track {
            let corrs: Vec<Correspondence> = self.graph.find_correspondences(e.image_id, e.point2d_idx).to_vec();
            for c in corrs {
                let other = Self::point_of(rec, &c);
                if other == INVALID_POINT3D_ID || other == pid {
                    continue;
                }
                let key = (pid.min(other), pid.max(other));
                if !self.merge_trials.insert(key) {
                    continue;
                }
                let Some(b) = rec.point3d(other).cloned() else { continue };
                let (wa, wb) = (a.track.len() as f64, b.track.len() as f64);
                let x = (a.xyz * wa + b.xyz * wb) / (wa + wb);
                let ok = a.track.iter().chain(b.track.iter()).all(|t| {
                    rec.image(t.image_id)
                        .is_some_and(|im| self.reproj_ok(t.image_id, &im.point2d(t.point2d_idx).xy, &x, self.opts.merge_max_reproj_error))
                });
                if !ok {
                    continue;
                }
                let new_id = rec.merge_points3d(pid, other)?;
                self.touched.insert(new_id);
                let n = a.track.len() + b.track.len();
                self.report.num_merged += n;
                let r = self.merge_track(rec, new_id)?;
                return Ok(if r > 0 { r } else { n });
            }
        }
        Ok(0)
    }

    /// Completes several points. `ids` None = all points.
    pub fn complete_tracks(&mut self, rec: &mut Reconstruction, ids: Option<&[Point3DId]>) -> Result<usize> {
        let ids: Vec<Point3DId> = ids.map_or_else(|| rec.point3d_ids(), |v| v.to_vec());
        let mut n = 0;
        for id in ids {
            n += self.complete_track(rec, id)?;
        }
        Ok(n)
    }

    /// Merges several points (the attempt record is reset for each call batch).
    pub fn merge_tracks(&mut self, rec: &mut Reconstruction, ids: Option<&[Point3DId]>) -> Result<usize> {
        self.merge_trials.clear();
        let ids: Vec<Point3DId> = ids.map_or_else(|| rec.point3d_ids(), |v| v.to_vec());
        let mut n = 0;
        for id in ids {
            if rec.exists_point3d(id) {
                n += self.merge_track(rec, id)?;
            }
        }
        Ok(n)
    }

    /// Re-triangulates insufficiently reconstructed image pairs. If `only` is given, only pairs involving that image.
    pub fn retriangulate(&mut self, rec: &mut Reconstruction, only: Option<&HashSet<ImageId>>) -> Result<usize> {
        let mut pairs = self.graph.image_pairs();
        pairs.sort();
        let mut total = 0;
        for (pair_id, num_corrs) in pairs {
            let (a, b) = images_of_pair(pair_id);
            if !self.poses.contains_key(&a) || !self.poses.contains_key(&b) || num_corrs == 0 {
                continue;
            }
            if let Some(s) = only {
                if !s.contains(&a) && !s.contains(&b) {
                    continue;
                }
            }
            let matches = self.graph.matches_between(a, b);
            let num_tri = matches
                .iter()
                .filter(|m| {
                    let pa = Self::point_of(rec, &Correspondence::new(a, m.idx1));
                    pa != INVALID_POINT3D_ID && pa == Self::point_of(rec, &Correspondence::new(b, m.idx2))
                })
                .count();
            if num_tri as f64 / num_corrs as f64 >= self.opts.retri_min_ratio {
                continue;
            }
            let trials = self.re_trials.entry((a, b)).or_insert(0);
            if *trials >= self.opts.retri_max_attempts {
                continue;
            }
            *trials += 1;
            // Creation RANSAC of the matches with no point on either side, in one batch per pair (the computation does not
            // depend on the model state, so the values solved in advance are used as is in the sequential pass).
            let mut pre: HashMap<usize, TriResult> = HashMap::new();
            {
                let rec_ref: &Reconstruction = rec;
                let empty: HashMap<Correspondence, Point3DId> = HashMap::new();
                let cand: Vec<(usize, Vec<TriObservation>)> = matches
                    .iter()
                    .enumerate()
                    .filter_map(|(k, m)| {
                        let ca = Correspondence::new(a, m.idx1);
                        let cb = Correspondence::new(b, m.idx2);
                        if Self::point_of(rec_ref, &ca) != INVALID_POINT3D_ID || Self::point_of(rec_ref, &cb) != INVALID_POINT3D_ID {
                            return None;
                        }
                        let (_, obs) = self.create_round(&[ca, cb], &empty)?;
                        Some((k, obs))
                    })
                    .collect();
                if !cand.is_empty() {
                    let params = self.create_params(self.opts.create_angle_tol);
                    let problems: Vec<TriProblem<'_>> = cand.iter().map(|(_, o)| TriProblem { obs: o, params: params.clone() }).collect();
                    let res = self.backend.estimate_batch(&problems);
                    assert_eq!(res.len(), problems.len(), "삼각측량 백엔드 결과 수");
                    for ((k, _), r) in cand.iter().zip(res) {
                        pre.insert(*k, r);
                    }
                }
            }
            for (k, m) in matches.iter().enumerate() {
                let ca = Correspondence::new(a, m.idx1);
                let cb = Correspondence::new(b, m.idx2);
                let (pa, pb) = (Self::point_of(rec, &ca), Self::point_of(rec, &cb));
                match (pa != INVALID_POINT3D_ID, pb != INVALID_POINT3D_ID) {
                    (true, true) => {}
                    (true, false) | (false, true) => {
                        let (pid, other) = if pa != INVALID_POINT3D_ID { (pa, cb) } else { (pb, ca) };
                        let (Some(r), Some(p)) = (self.ray(&other), rec.point3d(pid)) else { continue };
                        let err = angular_error(&self.poses[&other.image_id].proj, &r, &p.xyz);
                        if err <= self.opts.retri_angle_tol.to_radians() {
                            rec.add_observation(pid, TrackEntry::new(other.image_id, other.point2d_idx))?;
                            self.touched.insert(pid);
                            total += 1;
                        }
                    }
                    (false, false) => {
                        let input = [ca, cb];
                        let mut state: HashMap<Correspondence, Point3DId> = HashMap::new();
                        let mut actions = Vec::new();
                        self.plan_create_with(&input, &mut state, &mut actions, self.opts.create_angle_tol, pre.remove(&k));
                        total += self.apply(rec, actions)?;
                    }
                }
            }
        }
        self.report.num_retriangulated += total;
        Ok(total)
    }
}

fn depth_and_filter_ids(touched: &HashSet<Point3DId>, rec: &Reconstruction, scoped: bool) -> Option<Vec<Point3DId>> {
    if scoped {
        let mut v: Vec<Point3DId> = touched.iter().copied().filter(|id| rec.exists_point3d(*id)).collect();
        v.sort();
        Some(v)
    } else {
        None
    }
}

/// Table for observation checks: (camera, pose) of the images with a pose, and image id -> (camera index, pose index).
struct PoseTable {
    cameras: Vec<Camera>,
    poses: Vec<Rigid3>,
    index: HashMap<ImageId, (u32, u32)>,
}

fn pose_table(rec: &Reconstruction) -> PoseTable {
    let mut cameras: Vec<Camera> = Vec::new();
    let mut poses = Vec::new();
    let mut index = HashMap::new();
    let mut cam_index: HashMap<CameraId, u32> = HashMap::new();
    for id in rec.image_ids() {
        let Some(pose) = rec.world_to_cam(id) else { continue };
        let Some(im) = rec.image(id) else { continue };
        let Some(cam) = rec.camera(im.camera_id) else { continue };
        let ci = *cam_index.entry(im.camera_id).or_insert_with(|| {
            cameras.push(cam.clone());
            (cameras.len() - 1) as u32
        });
        poses.push(pose);
        index.insert(id, (ci, (poses.len() - 1) as u32));
    }
    PoseTable { cameras, poses, index }
}

/// Removes observations whose depth relative to the registered image is not positive (BA preprocessing).
/// Depths are computed in bulk by `backend`.
fn delete_negative_depth(rec: &mut Reconstruction, ids: Option<&[Point3DId]>, backend: &dyn TriangulationBackend) -> Result<usize> {
    let ids: Vec<Point3DId> = ids.map_or_else(|| rec.point3d_ids(), |v| v.to_vec());
    let table = pose_table(rec);
    let mut queries = Vec::new();
    let mut entries = Vec::new();
    for id in ids {
        let Some(p) = rec.point3d(id) else { continue };
        for t in &p.track {
            let Some(&(camera, pose)) = table.index.get(&t.image_id) else { continue };
            if !rec.is_image_registered(t.image_id) {
                continue;
            }
            let Some(im) = rec.image(t.image_id) else { continue };
            queries.push(ObsQuery { camera, pose, xy: im.point2d(t.point2d_idx).xy, xyz: p.xyz });
            entries.push(*t);
        }
    }
    let checks = if queries.is_empty() { Vec::new() } else { backend.check_observations(&table.cameras, &table.poses, &queries) };
    assert_eq!(checks.len(), queries.len(), "관측 검사 결과 수");
    // has_positive_depth: depth ≥ ε (NaN counts as negative).
    let positive = |d: f64| d >= f64::EPSILON;
    let del: Vec<TrackEntry> = entries.into_iter().zip(&checks).filter(|(_, c)| !positive(c.depth)).map(|(t, _)| t).collect();
    let mut n = 0;
    for t in del {
        if rec.image(t.image_id).is_some_and(|im| im.point2d(t.point2d_idx).has_point3d()) {
            rec.delete_observation(t.image_id, t.point2d_idx)?;
            n += 1;
        }
    }
    Ok(n)
}

/// Pixel reprojection error filter (same rules as `Reconstruction::filter_observations_with_large_reprojection_error`).
/// The errors are computed in bulk by `backend`. Returns: number of removed observations.
fn filter_large_reprojection_error(
    rec: &mut Reconstruction,
    max_reproj_error_px: f64,
    ids: Option<&[Point3DId]>,
    backend: &dyn TriangulationBackend,
) -> usize {
    let ids: Vec<Point3DId> = match ids {
        Some(v) => v.iter().copied().filter(|i| rec.exists_point3d(*i)).collect(),
        None => rec.point3d_ids(),
    };
    let table = pose_table(rec);
    let mut queries = Vec::new();
    // Query index per observation of each point (None for an image without a pose -> error +∞).
    let mut layout: Vec<(Point3DId, Vec<Option<usize>>)> = Vec::with_capacity(ids.len());
    for id in ids {
        let Some(p) = rec.point3d(id) else { continue };
        let mut slots = Vec::with_capacity(p.track.len());
        for t in &p.track {
            match (table.index.get(&t.image_id), rec.image(t.image_id)) {
                (Some(&(camera, pose)), Some(im)) => {
                    queries.push(ObsQuery { camera, pose, xy: im.point2d(t.point2d_idx).xy, xyz: p.xyz });
                    slots.push(Some(queries.len() - 1));
                }
                _ => slots.push(None),
            }
        }
        layout.push((id, slots));
    }
    let checks = if queries.is_empty() { Vec::new() } else { backend.check_observations(&table.cameras, &table.poses, &queries) };
    assert_eq!(checks.len(), queries.len(), "관측 검사 결과 수");
    let evals: Vec<(Point3DId, Vec<f64>)> = layout
        .into_iter()
        .map(|(id, slots)| (id, slots.into_iter().map(|s| s.map_or(f64::INFINITY, |k| checks[k].sq_reproj_error.sqrt())).collect()))
        .collect();
    rec.apply_observation_filter(evals, max_reproj_error_px, FilterErrorUpdate::SumOverOriginalLength)
}

/// Pose- and intrinsics-fixed point refinement. `ids` None = all points.
pub fn refine_points(rec: &mut Reconstruction, ids: Option<&[Point3DId]>, refiner: PointRefiner, max_iterations: usize) -> Result<()> {
    match refiner {
        PointRefiner::None => Ok(()),
        PointRefiner::PerPoint => {
            let ids: Vec<Point3DId> = ids.map_or_else(|| rec.point3d_ids(), |v| v.to_vec());
            let rec_ref: &Reconstruction = rec;
            let updates: Vec<(Point3DId, Vec3)> =
                ids.par_iter().filter_map(|id| refine_one_point(rec_ref, *id, max_iterations).map(|x| (*id, x))).collect();
            for (id, x) in updates {
                rec.set_point3d_xyz(id, x)?;
            }
            Ok(())
        }
        PointRefiner::BundleAdjuster => {
            let reg = rec.registered_images();
            if reg.len() < 2 {
                return Ok(());
            }
            let constant_points = match ids {
                Some(v) => {
                    let keep: HashSet<Point3DId> = v.iter().copied().collect();
                    rec.point3d_ids().into_iter().filter(|i| !keep.contains(i)).collect()
                }
                None => HashSet::new(),
            };
            let cfg = BaConfig {
                refine_focal_length: false,
                refine_principal_point: false,
                refine_extra_params: false,
                refine_poses: false,
                refine_points: true,
                loss: Loss::Trivial,
                max_num_iterations: max_iterations,
                function_tolerance: 0.0,
                gradient_tolerance: 1.0,
                parameter_tolerance: 0.0,
                images: HashSet::new(),
                constant_poses: reg.iter().copied().collect(),
                constant_cameras: rec.cameras().keys().copied().collect(),
                constant_points,
                auto_gauge: false,
                num_threads: 0,
                ..Default::default()
            };
            cumulus3d_ba::bundle_adjust(rec, &cfg)?;
            Ok(())
        }
    }
}

/// LM for one point (minimizes the sum of squared pixel reprojection errors at fixed poses). None on failure.
fn refine_one_point(rec: &Reconstruction, id: Point3DId, max_iterations: usize) -> Option<Vec3> {
    let p = rec.point3d(id)?;
    let obs: Vec<(Rigid3, &Camera, cumulus3d_core::Vec2)> = p
        .track
        .iter()
        .filter_map(|t| {
            let im = rec.image(t.image_id)?;
            Some((rec.world_to_cam(t.image_id)?, rec.camera(im.camera_id)?, im.point2d(t.point2d_idx).xy))
        })
        .collect();
    if obs.len() < 2 {
        return None;
    }
    let cost = |x: &Vec3| -> f64 { obs.iter().map(|(pose, cam, xy)| Reconstruction::squared_reprojection_error(pose, cam, xy, x)).sum() };
    let mut x = p.xyz;
    let mut c = cost(&x);
    let mut mu = 1e-4;
    for _ in 0..max_iterations {
        let mut h = nalgebra::Matrix3::<f64>::zeros();
        let mut g = Vec3::zeros();
        for (pose, cam, xy) in &obs {
            let r = pose.rotation_matrix();
            let xc = r * x + pose.translation;
            let Some((pix, j)) = cam.cam_to_img_with_jacobian(&xc, None) else { continue };
            let jx = j * r;
            let e = pix - xy;
            h += jx.transpose() * jx;
            g += jx.transpose() * e;
        }
        if g.amax() < 1e-10 {
            break;
        }
        let mut improved = false;
        for _ in 0..10 {
            let mut hd = h;
            for k in 0..3 {
                hd[(k, k)] += mu * h[(k, k)].clamp(1e-6, 1e32);
            }
            let Some(dx) = hd.lu().solve(&(-g)) else { break };
            let nx = x + dx;
            let nc = cost(&nx);
            if nc < c {
                let rel = (c - nc) / c.max(1e-300);
                x = nx;
                c = nc;
                mu = (mu / 3.0).max(1e-16);
                improved = true;
                if rel < 1e-12 || dx.norm() < 1e-12 * (x.norm() + 1e-12) {
                    return Some(x);
                }
                break;
            }
            mu *= 4.0;
        }
        if !improved {
            break;
        }
    }
    Some(x)
}

/// point_triangulator. Returns: statistics.
pub fn triangulate_points(rec: &mut Reconstruction, graph: &MatchGraph, opts: &PointTriangulatorOptions) -> Result<TriangulationReport> {
    triangulate_points_with(rec, graph, opts, &CpuTriangulation)
}

/// Like [`triangulate_points`], but creation RANSAC, depth checks and the reprojection error filter are computed in batches by
/// `backend`. Model-changing work (adding, extending, merging and deleting points) is done by the CPU in the same order.
pub fn triangulate_points_with(
    rec: &mut Reconstruction,
    graph: &MatchGraph,
    opts: &PointTriangulatorOptions,
    backend: &dyn TriangulationBackend,
) -> Result<TriangulationReport> {
    if rec.registered_image_count() < 2 {
        return Err(Error::InvalidArgument("point_triangulator: 등록 영상 2장 미만".into()));
    }
    if opts.clear_points {
        rec.delete_all_points3d();
    }
    let mut tri = TrackTriangulator::with_backend(rec, graph, opts.tri.clone(), backend);
    let (images, scoped): (Vec<ImageId>, bool) = match &opts.scope {
        TriangulationScope::AllRegistered => (rec.registered_images(), false),
        TriangulationScope::Images(v) => (v.iter().copied().filter(|i| rec.is_image_registered(*i)).collect(), true),
    };
    let only: HashSet<ImageId> = images.iter().copied().collect();
    for &id in &images {
        tri.triangulate_image(rec, id)?;
    }
    if scoped {
        // Existing points that have observations from new images are also targets of completion and merging.
        for &id in &images {
            if let Some(im) = rec.image(id) {
                tri.touched.extend(im.points2d().iter().filter(|p| p.has_point3d()).map(|p| p.point3d_id));
            }
        }
    }
    let scope_ids = |tri: &TrackTriangulator, rec: &Reconstruction| depth_and_filter_ids(&tri.touched, rec, scoped);

    // Iterative global refinement.
    let ids = scope_ids(&tri, rec);
    tri.complete_tracks(rec, ids.as_deref())?;
    let ids = scope_ids(&tri, rec);
    tri.merge_tracks(rec, ids.as_deref())?;
    tri.retriangulate(rec, if scoped { Some(&only) } else { None })?;
    for _ in 0..opts.ba_global_max_refinements {
        tri.report.num_refinements += 1;
        let num_obs = rec.total_observations().max(1);
        let ids = scope_ids(&tri, rec);
        delete_negative_depth(rec, ids.as_deref(), backend)?;
        let ids = scope_ids(&tri, rec);
        refine_points(rec, ids.as_deref(), opts.refiner, opts.ba_global_max_num_iterations)?;
        let ids = scope_ids(&tri, rec);
        let mut changed = tri.complete_tracks(rec, ids.as_deref())?;
        let ids = scope_ids(&tri, rec);
        changed += tri.merge_tracks(rec, ids.as_deref())?;
        let ids = scope_ids(&tri, rec);
        let f = filter_large_reprojection_error(rec, opts.filter_max_reproj_error, ids.as_deref(), backend);
        let ids = scope_ids(&tri, rec);
        let g = rec.filter_points3d_with_small_triangulation_angle(opts.filter_min_tri_angle, ids.as_deref());
        tri.report.num_filtered += f + g;
        changed += f + g;
        if (changed as f64) / (num_obs as f64) < opts.ba_global_max_refinement_change {
            break;
        }
    }
    rec.update_point3d_errors();
    Ok(tri.report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cumulus3d_core::{CameraModelKind, FeatureMatch, Image, Quat, TwoViewGeometry, TwoViewGeometryConfig, Vec2};

    fn cam() -> Camera {
        Camera::new(1, CameraModelKind::Pinhole, 1000, 1000, vec![1000.0, 1000.0, 500.0, 500.0]).unwrap()
    }

    /// n images (2 m apart horizontally), each with one 2D point = projection of X (+ a per-image perturbation function).
    fn scene(n: u32, x: Vec3, perturb: impl Fn(u32, Vec2) -> Vec2) -> Reconstruction {
        let mut rec = Reconstruction::new();
        let c = cam();
        rec.add_camera_own_rig(c.clone()).unwrap();
        for id in 1..=n {
            let p = Rigid3::new(Quat::IDENTITY, Vec3::new(-2.0 * id as f64, 0.0, 0.0));
            let xy = perturb(id, c.cam_to_img(&(p * x)).unwrap());
            rec.add_image_own_frame(Image::new(id, format!("{id}"), 1, vec![xy]), Some(p)).unwrap();
            rec.register_image(id).unwrap();
        }
        rec
    }

    fn link(g: &mut MatchGraph, a: u32, b: u32) {
        let tvg = TwoViewGeometry {
            config: TwoViewGeometryConfig::Calibrated,
            e: None,
            f: None,
            h: None,
            cam1_to_cam2: None,
            inlier_matches: vec![FeatureMatch::new(0, 0)],
            tri_angle: None,
        };
        g.insert_two_view(a, b, &tvg);
    }

    fn graph(n: u32, edges: &[(u32, u32)]) -> MatchGraph {
        let mut g = MatchGraph::new();
        for id in 1..=n {
            g.add_image(id, 1);
        }
        for (a, b) in edges {
            link(&mut g, *a, *b);
        }
        g
    }

    #[test]
    fn continue_angle_boundary() {
        let x = Vec3::new(5.0, 0.0, 20.0);
        for (deg, ok) in [(1.9f64, true), (2.1, false)] {
            // Turn the image-3 observation into a ray rotated by deg.
            let mut rec = scene(3, x, |id, xy| {
                if id != 3 {
                    return xy;
                }
                let r = cam().img_to_ray(&xy).unwrap();
                let axis = r.cross(&Vec3::y()).normalize();
                let r2 = crate::math::so3_exp(&(axis * deg.to_radians())) * r;
                cam().cam_to_img(&r2).unwrap()
            });
            rec.add_point3d(x, vec![TrackEntry::new(1, 0), TrackEntry::new(2, 0)], [0; 3]).unwrap();
            let g = graph(3, &[(1, 3), (2, 3)]);
            let mut tri = TrackTriangulator::new(&rec, &g, TrackTriangulatorOptions::default());
            tri.triangulate_image(&mut rec, 3).unwrap();
            assert_eq!(rec.image(3).unwrap().point2d(0).has_point3d(), ok, "deg {deg}");
        }
    }

    #[test]
    fn complete_transitivity_depth() {
        let x = Vec3::new(5.0, 0.0, 20.0);
        let mut rec = scene(8, x, |_, xy| xy);
        rec.add_point3d(x, vec![TrackEntry::new(1, 0), TrackEntry::new(2, 0)], [0; 3]).unwrap();
        let g = graph(8, &[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]);
        let mut tri = TrackTriangulator::new(&rec, &g, TrackTriangulatorOptions::default());
        let pid = rec.point3d_ids()[0];
        assert_eq!(tri.complete_track(&mut rec, pid).unwrap(), 5);
        assert!(rec.image(7).unwrap().point2d(0).has_point3d());
        assert!(!rec.image(8).unwrap().point2d(0).has_point3d());
    }

    #[test]
    fn merge_rules() {
        let x = Vec3::new(5.0, 0.0, 20.0);
        for (err_px, merged) in [(1.0, true), (12.0, false)] {
            let mut rec = scene(4, x, |_, xy| xy);
            let shift = Vec3::new(err_px * 20.0 / 1000.0, 0.0, 0.0);
            let a = rec.add_point3d(x, vec![TrackEntry::new(1, 0), TrackEntry::new(2, 0)], [0; 3]).unwrap();
            let _ = a;
            rec.add_point3d(x + shift, vec![TrackEntry::new(3, 0), TrackEntry::new(4, 0)], [0; 3]).unwrap();
            let g = graph(4, &[(2, 3)]);
            let mut tri = TrackTriangulator::new(&rec, &g, TrackTriangulatorOptions::default());
            tri.merge_tracks(&mut rec, None).unwrap();
            assert_eq!(rec.num_points3d() == 1, merged, "err {err_px}");
            if merged {
                let (_, p) = rec.points3d().next().unwrap();
                assert!((p.xyz - (x + shift * 0.5)).norm() < 1e-12);
                assert_eq!(p.track.len(), 4);
            }
        }
    }

    #[test]
    fn reproj_filter_boundary() {
        let x = Vec3::new(5.0, 0.0, 20.0);
        for (e, keep_len) in [(3.99, 3usize), (4.01, 2)] {
            let mut rec = scene(3, x, |id, xy| if id == 3 { xy + Vec2::new(e, 0.0) } else { xy });
            rec.add_point3d(x, (1..=3).map(|i| TrackEntry::new(i, 0)).collect(), [0; 3]).unwrap();
            rec.filter_observations_with_large_reprojection_error(4.0, None);
            assert_eq!(rec.points3d().next().map_or(0, |(_, p)| p.track.len()), keep_len);
        }
        // If 2 of the 3 observations of track 3 are bad, the whole point is deleted.
        let mut rec = scene(3, x, |id, xy| if id >= 2 { xy + Vec2::new(5.0, 0.0) } else { xy });
        rec.add_point3d(x, (1..=3).map(|i| TrackEntry::new(i, 0)).collect(), [0; 3]).unwrap();
        rec.filter_observations_with_large_reprojection_error(4.0, None);
        assert_eq!(rec.num_points3d(), 0);
    }

    #[test]
    fn create_from_correspondences() {
        let x = Vec3::new(5.0, 1.0, 20.0);
        let mut rec = scene(3, x, |_, xy| xy);
        let g = graph(3, &[(1, 2), (2, 3), (1, 3)]);
        let mut tri = TrackTriangulator::new(&rec, &g, TrackTriangulatorOptions::default());
        tri.triangulate_image(&mut rec, 1).unwrap();
        assert_eq!(rec.num_points3d(), 1);
        let (_, p) = rec.points3d().next().unwrap();
        assert!((p.xyz - x).norm() < 1e-6);
        assert_eq!(p.track.len(), 3);
    }
}
