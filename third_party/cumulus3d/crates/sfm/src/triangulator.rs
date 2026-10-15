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

//! 기존 점 유지 삼각측량(point_triangulator): 연장·생성·완성·병합·재삼각측량,
//! 자세 고정 점 정제 반복과 필터.

use crate::triangulation::{angular_error, estimate_triangulation, has_positive_depth, TriObservation, TriangulationRansacParams};
use cumulus3d_ba::{BaConfig, Loss};
use cumulus3d_core::graph::Correspondence;
use cumulus3d_core::{
    images_of_pair, Camera, Error, ImageId, Mat3x4, MatchGraph, Point2DIdx, Point3DId, Reconstruction, Result, Rigid3, TrackEntry, Vec3,
    INVALID_POINT3D_ID,
};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// 증분 삼각측량기 옵션.
#[derive(Clone, Debug)]
pub struct TrackTriangulatorOptions {
    /// 새 점 생성 시 대응 그래프 전이 탐색 깊이.
    pub max_transitivity: usize,
    /// 새 점 생성 시 각도 오차 허용(도).
    pub create_angle_tol: f64,
    /// 기존 점 연장 시 각도 오차 허용(도).
    pub continue_angle_tol: f64,
    /// 트랙 병합 재투영 오차 상한(픽셀).
    pub merge_max_reproj_error: f64,
    /// 트랙 완성 재투영 오차 상한(픽셀).
    pub complete_max_reproj_error: f64,
    /// 트랙 완성 시 전이 탐색 깊이.
    pub complete_transitivity: usize,
    /// 재삼각측량 각도 오차 허용(도).
    pub retri_angle_tol: f64,
    /// 짝의 삼각측량 비율이 이보다 낮을 때만 재삼각측량.
    pub retri_min_ratio: f64,
    /// 짝당 재삼각측량 최대 시도 횟수.
    pub retri_max_attempts: usize,
    /// 최소 삼각측량 각(도).
    pub min_angle: f64,
    /// 두 영상뿐인 트랙은 건너뛸지.
    pub skip_two_view_tracks: bool,
    /// 비정상 카메라 판정: 초점 비율 하한.
    pub min_focal_length_ratio: f64,
    /// 비정상 카메라 판정: 초점 비율 상한.
    pub max_focal_length_ratio: f64,
    /// 비정상 카메라 판정: 추가 파라미터 절대값 상한.
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

/// 점 정제 방식(자세·내부 고정 BA).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointRefiner {
    /// cumulus3d-ba 의 `bundle_adjust` (자세·카메라 상수, 점만 변수). 기본 동작.
    BundleAdjuster,
    /// 점별 독립 3변수 LM(자세·내부가 고정이면 BA 와 같은 문제).
    PerPoint,
    /// 정제 생략.
    None,
}

/// 처리 범위.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TriangulationScope {
    /// 기본 동작: 모든 등록 영상 재삼각측량, 모든 점 완성·병합·필터.
    AllRegistered,
    /// 증분: 주어진(새로 등록된) 영상과 그 대응, 그로 인해 바뀐 점만.
    Images(Vec<ImageId>),
}

/// point_triangulator 옵션.
#[derive(Clone, Debug)]
pub struct PointTriangulatorOptions {
    /// 증분 삼각측량기 옵션.
    pub tri: TrackTriangulatorOptions,
    /// 참이면 시작 전에 모든 3D 점 삭제(파이프라인은 거짓).
    pub clear_points: bool,
    /// 끝 필터의 재투영 오차 상한(픽셀).
    pub filter_max_reproj_error: f64,
    /// 끝 필터의 최소 삼각측량 각(도).
    pub filter_min_tri_angle: f64,
    /// 점 정제·재삼각측량 반복 최대 횟수.
    pub ba_global_max_refinements: usize,
    /// 반복을 멈추는 변경 관측 비율.
    pub ba_global_max_refinement_change: f64,
    /// 점 정제 BA 최대 반복.
    pub ba_global_max_num_iterations: usize,
    /// 점 정제 방식.
    pub refiner: PointRefiner,
    /// 처리 범위(전체 또는 새 영상).
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

/// 실행 통계.
#[derive(Clone, Debug, Default)]
pub struct TriangulationReport {
    /// 새로 만든 점 수.
    pub num_created: usize,
    /// 기존 점에 이어 붙인 관측 수.
    pub num_continued: usize,
    /// 트랙 완성으로 붙인 관측 수.
    pub num_completed: usize,
    /// 병합된 점들의 관측 수 합.
    pub num_merged: usize,
    /// 재삼각측량으로 만든 관측 수.
    pub num_retriangulated: usize,
    /// 필터가 지운 관측·점 수.
    pub num_filtered: usize,
    /// 수행한 정제 반복 수.
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

/// 증분 삼각측량기. 생성 시점 등록 영상의 자세·역투영 광선을 캐시한다(자세 고정 전제).
pub struct TrackTriangulator<'a> {
    graph: &'a MatchGraph,
    /// 삼각측량 옵션.
    pub opts: TrackTriangulatorOptions,
    poses: HashMap<ImageId, PoseInfo>,
    merge_trials: HashSet<(Point3DId, Point3DId)>,
    re_trials: HashMap<(ImageId, ImageId), usize>,
    /// 이번 실행에서 만들어지거나 바뀐 점.
    pub touched: HashSet<Point3DId>,
    /// 누적 실행 통계.
    pub report: TriangulationReport,
}

impl<'a> TrackTriangulator<'a> {
    /// 재구성의 등록 영상 자세·광선을 캐시해 삼각측량기를 만든다.
    pub fn new(rec: &Reconstruction, graph: &'a MatchGraph, opts: TrackTriangulatorOptions) -> Self {
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

    /// 영상 하나의 2D 점 하나를 읽기 전용으로 계획(연장 → 생성). 반환: (참조한 상태, 동작).
    fn plan_point(&self, rec: &Reconstruction, image_id: ImageId, idx: Point2DIdx) -> Option<PointPlan> {
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
        self.plan_create(&input, &mut state, &mut actions, self.opts.create_angle_tol);
        Some((snapshot, actions))
    }

    /// 생성 모의 실행: `state` 에 새 점을 가상 id 로 표시.
    fn plan_create(
        &self,
        input: &[Correspondence],
        state: &mut HashMap<Correspondence, Point3DId>,
        actions: &mut Vec<Action>,
        angle_limit_deg: f64,
    ) {
        loop {
            let sel: Vec<Correspondence> =
                input.iter().filter(|c| state.get(c).copied().unwrap_or(INVALID_POINT3D_ID) == INVALID_POINT3D_ID).copied().collect();
            if sel.len() < 2 {
                return;
            }
            if self.opts.skip_two_view_tracks && sel.len() == 2 && self.graph.in_two_view_track(sel[0].image_id, sel[0].point2d_idx) {
                return;
            }
            let mut obs = Vec::with_capacity(sel.len());
            let mut elems = Vec::with_capacity(sel.len());
            for c in &sel {
                let (Some(pi), Some(r)) = (self.poses.get(&c.image_id), self.ray(c)) else { continue };
                obs.push(TriObservation { proj: pi.proj, center: pi.center, ray: r });
                elems.push(*c);
            }
            if obs.len() < 2 {
                return;
            }
            let ro = TriangulationRansacParams {
                max_angle_error_deg: angle_limit_deg,
                min_tri_angle_deg: self.opts.min_angle,
                ..Default::default()
            };
            let Some((x, mask)) = estimate_triangulation(&obs, &ro) else { return };
            let track: Vec<TrackEntry> =
                elems.iter().zip(&mask).filter(|(_, m)| **m).map(|(c, _)| TrackEntry::new(c.image_id, c.point2d_idx)).collect();
            let len = track.len();
            for t in &track {
                state.insert(Correspondence::new(t.image_id, t.point2d_idx), INVALID_POINT3D_ID - 1);
            }
            actions.push(Action::Create(x, track));
            // 설계 결정: "입력 관측 수" = 3D 점 유무와 무관한 전체 입력 수로 해석.
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

    /// 영상 하나 삼각측량. 2D 점별 계획은 병렬, 적용은 순차 + 상태 검증
    /// (계획 후 참조 상태가 바뀌었으면 다시 계획) → 순차 실행과 같은 결과.
    pub fn triangulate_image(&mut self, rec: &mut Reconstruction, image_id: ImageId) -> Result<usize> {
        if !self.poses.contains_key(&image_id) {
            return Ok(0);
        }
        let n = rec.image(image_id).map_or(0, |im| im.num_points2d());
        let plans: Vec<Option<PointPlan>> = {
            let rec_ref: &Reconstruction = rec;
            (0..n as Point2DIdx).into_par_iter().map(|i| self.plan_point(rec_ref, image_id, i)).collect()
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

    /// 트랙 완성. 반환: 추가된 관측 수.
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

    /// 트랙 병합. 반환: 병합에 관여한 관측 수(재귀 포함).
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

    /// 여러 점 완성. `ids` None = 모든 점.
    pub fn complete_tracks(&mut self, rec: &mut Reconstruction, ids: Option<&[Point3DId]>) -> Result<usize> {
        let ids: Vec<Point3DId> = ids.map_or_else(|| rec.point3d_ids(), |v| v.to_vec());
        let mut n = 0;
        for id in ids {
            n += self.complete_track(rec, id)?;
        }
        Ok(n)
    }

    /// 여러 점 병합(호출 묶음마다 시도 기록 초기화).
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

    /// 덜 재구성된 영상쌍 재삼각측량. `only` 가 있으면 그 영상이 낀 쌍만.
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
            for m in matches {
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
                        self.plan_create(&input, &mut state, &mut actions, self.opts.create_angle_tol);
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

/// 등록 영상 기준 깊이가 양수가 아닌 관측 삭제(BA 사전 처리).
fn delete_negative_depth(rec: &mut Reconstruction, ids: Option<&[Point3DId]>) -> Result<usize> {
    let ids: Vec<Point3DId> = ids.map_or_else(|| rec.point3d_ids(), |v| v.to_vec());
    let mut del = Vec::new();
    for id in ids {
        let Some(p) = rec.point3d(id) else { continue };
        for t in &p.track {
            if let Some(pose) = rec.world_to_cam(t.image_id) {
                if rec.is_image_registered(t.image_id) && !has_positive_depth(&pose.matrix(), &p.xyz) {
                    del.push(*t);
                }
            }
        }
    }
    let mut n = 0;
    for t in del {
        if rec.image(t.image_id).is_some_and(|im| im.point2d(t.point2d_idx).has_point3d()) {
            rec.delete_observation(t.image_id, t.point2d_idx)?;
            n += 1;
        }
    }
    Ok(n)
}

/// 자세·내부 고정 점 정제. `ids` None = 모든 점.
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

/// 점 하나 LM(고정 자세에서 픽셀 재투영 제곱합 최소). 실패 시 None.
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

/// point_triangulator. 반환: 통계.
pub fn triangulate_points(rec: &mut Reconstruction, graph: &MatchGraph, opts: &PointTriangulatorOptions) -> Result<TriangulationReport> {
    if rec.registered_image_count() < 2 {
        return Err(Error::InvalidArgument("point_triangulator: 등록 영상 2장 미만".into()));
    }
    if opts.clear_points {
        rec.delete_all_points3d();
    }
    let mut tri = TrackTriangulator::new(rec, graph, opts.tri.clone());
    let (images, scoped): (Vec<ImageId>, bool) = match &opts.scope {
        TriangulationScope::AllRegistered => (rec.registered_images(), false),
        TriangulationScope::Images(v) => (v.iter().copied().filter(|i| rec.is_image_registered(*i)).collect(), true),
    };
    let only: HashSet<ImageId> = images.iter().copied().collect();
    for &id in &images {
        tri.triangulate_image(rec, id)?;
    }
    if scoped {
        // 새 영상 관측을 가진 기존 점도 완성·병합 대상.
        for &id in &images {
            if let Some(im) = rec.image(id) {
                tri.touched.extend(im.points2d().iter().filter(|p| p.has_point3d()).map(|p| p.point3d_id));
            }
        }
    }
    let scope_ids = |tri: &TrackTriangulator, rec: &Reconstruction| depth_and_filter_ids(&tri.touched, rec, scoped);

    // 반복 전역 정제.
    let ids = scope_ids(&tri, rec);
    tri.complete_tracks(rec, ids.as_deref())?;
    let ids = scope_ids(&tri, rec);
    tri.merge_tracks(rec, ids.as_deref())?;
    tri.retriangulate(rec, if scoped { Some(&only) } else { None })?;
    for _ in 0..opts.ba_global_max_refinements {
        tri.report.num_refinements += 1;
        let num_obs = rec.total_observations().max(1);
        let ids = scope_ids(&tri, rec);
        delete_negative_depth(rec, ids.as_deref())?;
        let ids = scope_ids(&tri, rec);
        refine_points(rec, ids.as_deref(), opts.refiner, opts.ba_global_max_num_iterations)?;
        let ids = scope_ids(&tri, rec);
        let mut changed = tri.complete_tracks(rec, ids.as_deref())?;
        let ids = scope_ids(&tri, rec);
        changed += tri.merge_tracks(rec, ids.as_deref())?;
        let ids = scope_ids(&tri, rec);
        let f = rec.filter_observations_with_large_reprojection_error(opts.filter_max_reproj_error, ids.as_deref());
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

    /// n 개 영상(가로로 2 m 간격), 각 영상 2D 점 1개 = X 의 투영(+ 영상별 섭동 함수).
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
            // 영상 3 관측을 deg 만큼 틀어진 광선으로.
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
        // 트랙 3 에서 2개가 나쁘면 점 전체 삭제.
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
