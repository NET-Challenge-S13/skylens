/*
 * batched.rs
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

//! 두 뷰 기하 검증 백엔드와 짝 묶음 RANSAC.
//!
//! [`TwoViewBackend`] 는 짝 여러 개를 한 번에 검증한다. CPU 기본 구현 [`CpuTwoView`] 는 짝마다
//! [`verify_pair`] 를 병렬로 부르고, GPU 구현(cumulus3d-cuda)은 [`verify_pairs_batched`] 에
//! 가설 채점기([`BatchScorer`])를 넘긴다. [`verify_pairs_batched`] 는 모든 짝의 E/F/H LO-RANSAC 을
//! 나란히 진행하면서, 라운드마다 모든 짝의 가설을 채점기 호출 한 번으로 채점한다. 표본·최소 해법·
//! 판정·LO 는 CPU 경로와 같으므로 채점 결과가 같으면 검증 결과도 같다.

use crate::estimators::{EssentialFivePointEstimator, Fundamental7PtEstimator, FundamentalEightPointEstimator, HomographyEstimator};
use crate::pipeline::verify_pair;
use crate::two_view::{
    assemble_calibrated, assemble_homography_only, assemble_uncalibrated, calibrated_inputs, degenerate, derive_seed, estimate_two_view,
    filter_stationary, finalize_geometry, gather_points, path_of, ransac_opts, EstimationPath, TwoViewOptions, SALT_E, SALT_F, SALT_H,
};
use cumulus3d_core::ransac::{BatchedRansac, Estimator, RandomSubsetSampler, RansacReport, Support};
use cumulus3d_core::{pair_id_of, Camera, FeatureMatch, FeatureStore, ImageId, Keypoint, Mat3, TwoViewGeometry, Vec2, Vec3};
use rayon::prelude::*;
use std::sync::Arc;

/// 검증할 짝 하나.
#[derive(Clone, Copy, Debug)]
pub struct VerifyJob<'a> {
    /// 영상 1.
    pub id1: ImageId,
    /// 영상 2.
    pub id2: ImageId,
    /// 원시 매칭.
    pub matches: &'a [FeatureMatch],
}

/// 두 뷰 기하 검증 백엔드. 결과는 짝마다 [`verify_pair`] 와 같은 규칙(최소 인라이어 적용 후)이고
/// 순서는 입력 순서.
pub trait TwoViewBackend: Send + Sync {
    /// 짝 여러 개 검증.
    fn verify_pairs(&self, store: &FeatureStore, jobs: &[VerifyJob<'_>], opts: &TwoViewOptions) -> Vec<TwoViewGeometry>;
}

/// CPU 기본 구현: 짝마다 [`verify_pair`] 를 병렬로.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuTwoView;

impl TwoViewBackend for CpuTwoView {
    fn verify_pairs(&self, store: &FeatureStore, jobs: &[VerifyJob<'_>], opts: &TwoViewOptions) -> Vec<TwoViewGeometry> {
        jobs.par_iter().map(|j| verify_pair(store, j.id1, j.id2, j.matches, opts)).collect()
    }
}

/// 잔차 종류별 자료 한 벌(채점기 슬롯 하나).
#[derive(Clone, Copy, Debug)]
pub enum ResidualSet<'a> {
    /// 본질 행렬 Sampson 제곱 오차. (영상1 단위 광선, 영상2 단위 광선).
    EssentialRays(&'a [Vec3], &'a [Vec3]),
    /// 기초 행렬 Sampson 제곱 오차. (영상1 픽셀, 영상2 픽셀).
    FundamentalPx(&'a [Vec2], &'a [Vec2]),
    /// 호모그래피 단방향 전이 제곱 오차. (영상1 픽셀, 영상2 픽셀).
    HomographyPx(&'a [Vec2], &'a [Vec2]),
}

impl ResidualSet<'_> {
    /// 자료 수.
    pub fn len(&self) -> usize {
        match self {
            Self::EssentialRays(a, _) => a.len(),
            Self::FundamentalPx(a, _) | Self::HomographyPx(a, _) => a.len(),
        }
    }
    /// 비었는지.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// 채점 요청 하나: 슬롯 `slot` 의 모든 자료에 대해 모델마다 지지도.
#[derive(Clone, Copy, Debug)]
pub struct ScoreRequest<'a> {
    /// [`BatchScorer::load`] 에 넘긴 자료 순서.
    pub slot: usize,
    /// 3×3 모델(E, F 또는 H).
    pub models: &'a [Mat3],
    /// 인라이어 제곱 임계.
    pub max_residual: f64,
}

/// 가설 일괄 채점기. 지지도는 [`Support::measure`] 와 같아야 한다(잔차 합도 자료 순서대로 더한 값).
pub trait BatchScorer {
    /// 자료 슬롯을 등록한다(라운드 전에 한 번, 이전 슬롯은 버린다).
    fn load(&mut self, sets: &[ResidualSet<'_>]) -> Result<(), String>;
    /// 라운드 하나: 요청마다 모델별 지지도.
    fn score(&mut self, reqs: &[ScoreRequest<'_>]) -> Result<Vec<Vec<Support>>, String>;
}

enum OwnedSet {
    E(Vec<Vec3>, Vec<Vec3>),
    F(Vec<Vec2>, Vec<Vec2>),
    H(Vec<Vec2>, Vec<Vec2>),
}

/// CPU 채점기(대조·시험용): 추정기의 잔차 함수와 [`Support::measure`].
#[derive(Default)]
pub struct CpuBatchScorer {
    sets: Vec<OwnedSet>,
}

impl BatchScorer for CpuBatchScorer {
    fn load(&mut self, sets: &[ResidualSet<'_>]) -> Result<(), String> {
        self.sets = sets
            .iter()
            .map(|s| match *s {
                ResidualSet::EssentialRays(a, b) => OwnedSet::E(a.to_vec(), b.to_vec()),
                ResidualSet::FundamentalPx(a, b) => OwnedSet::F(a.to_vec(), b.to_vec()),
                ResidualSet::HomographyPx(a, b) => OwnedSet::H(a.to_vec(), b.to_vec()),
            })
            .collect();
        Ok(())
    }

    fn score(&mut self, reqs: &[ScoreRequest<'_>]) -> Result<Vec<Vec<Support>>, String> {
        let sets = &self.sets;
        Ok(reqs
            .par_iter()
            .map(|r| {
                let mut res = Vec::new();
                r.models
                    .iter()
                    .map(|m| {
                        match &sets[r.slot] {
                            OwnedSet::E(a, b) => EssentialFivePointEstimator.residuals(a, b, m, &mut res),
                            OwnedSet::F(a, b) => Fundamental7PtEstimator.residuals(a, b, m, &mut res),
                            OwnedSet::H(a, b) => HomographyEstimator::default().residuals(a, b, m, &mut res),
                        }
                        Support::measure(&res, r.max_residual)
                    })
                    .collect()
            })
            .collect())
    }
}

/// 한 라운드의 최소 반복 수(동적 상한이 거의 찼을 때 너무 잘게 나누지 않도록).
const MIN_BATCH: usize = 16;
/// 한 라운드의 최대 반복 수.
const MAX_BATCH: usize = 512;

type Rs<'a, E, L> = BatchedRansac<'a, E, L, RandomSubsetSampler>;

enum Stepper<'a> {
    E(Rs<'a, EssentialFivePointEstimator, EssentialFivePointEstimator>),
    F(Rs<'a, Fundamental7PtEstimator, FundamentalEightPointEstimator>),
    H(Rs<'a, HomographyEstimator, HomographyEstimator>),
}

impl Stepper<'_> {
    fn is_done(&self) -> bool {
        match self {
            Self::E(r) => r.is_done(),
            Self::F(r) => r.is_done(),
            Self::H(r) => r.is_done(),
        }
    }
    fn propose(&mut self) {
        match self {
            Self::E(r) => {
                r.propose(MIN_BATCH, MAX_BATCH);
            }
            Self::F(r) => {
                r.propose(MIN_BATCH, MAX_BATCH);
            }
            Self::H(r) => {
                r.propose(MIN_BATCH, MAX_BATCH);
            }
        }
    }
    fn models(&self) -> &[Mat3] {
        match self {
            Self::E(r) => r.models(),
            Self::F(r) => r.models(),
            Self::H(r) => r.models(),
        }
    }
    fn max_residual(&self) -> f64 {
        match self {
            Self::E(r) => r.max_residual(),
            Self::F(r) => r.max_residual(),
            Self::H(r) => r.max_residual(),
        }
    }
    fn consume(&mut self, s: &[Support]) {
        match self {
            Self::E(r) => r.consume(s),
            Self::F(r) => r.consume(s),
            Self::H(r) => r.consume(s),
        }
    }
    fn finish(self) -> RansacReport<Mat3> {
        match self {
            Self::E(r) => r.finish(),
            Self::F(r) => r.finish(),
            Self::H(r) => r.finish(),
        }
    }
}

/// RANSAC 을 돌릴 짝 하나의 입력.
struct PairRun {
    cam1: Camera,
    cam2: Camera,
    kps1: Arc<Vec<Keypoint>>,
    kps2: Arc<Vec<Keypoint>>,
    /// 정지 매칭을 뺀 매칭.
    matches: Vec<FeatureMatch>,
    /// 짝 시드를 넣은 옵션.
    opts: TwoViewOptions,
    path: EstimationPath,
    x1: Vec<Vec2>,
    x2: Vec<Vec2>,
    r1: Vec<Vec3>,
    r2: Vec<Vec3>,
    e_err: f64,
}

enum Prep {
    /// RANSAC 없이 끝난 짝(자료 부족, 다중 모델 경로).
    Done(TwoViewGeometry),
    Run(Box<PairRun>),
}

/// [`verify_pair`] 의 앞부분(자료 조회·시드·정지 매칭)까지.
fn prepare(store: &FeatureStore, job: &VerifyJob<'_>, opts: &TwoViewOptions) -> Prep {
    if job.matches.len() < opts.min_num_inliers {
        return Prep::Done(TwoViewGeometry::default());
    }
    let (Some(i1), Some(i2)) = (store.image(job.id1), store.image(job.id2)) else { return Prep::Done(TwoViewGeometry::default()) };
    let (Some(cam1), Some(cam2)) = (store.camera(i1.camera_id), store.camera(i2.camera_id)) else {
        return Prep::Done(TwoViewGeometry::default());
    };
    let (Some(kps1), Some(kps2)) = (store.keypoints(job.id1), store.keypoints(job.id2)) else {
        return Prep::Done(TwoViewGeometry::default());
    };
    let mut o = opts.clone();
    if let Ok(pid) = pair_id_of(job.id1, job.id2) {
        o.ransac.random_seed = derive_seed(opts.ransac.random_seed, pid);
    }
    let path = path_of(&cam1, &cam2, &o);
    if path == EstimationPath::Multiple {
        let g = estimate_two_view(&cam1, &kps1, &cam2, &kps2, job.matches, &o);
        return Prep::Done(finalize_geometry(g, opts.min_num_inliers));
    }
    let matches = filter_stationary(&kps1, &kps2, job.matches, &o).into_owned();
    if matches.len() < o.min_num_inliers {
        return Prep::Done(finalize_geometry(degenerate(), opts.min_num_inliers));
    }
    let (x1, x2, r1, r2, e_err) = if path == EstimationPath::Calibrated {
        calibrated_inputs(&cam1, &kps1, &cam2, &kps2, &matches, &o)
    } else {
        let (x1, x2) = gather_points(&kps1, &kps2, &matches);
        (x1, x2, Vec::new(), Vec::new(), 0.0)
    };
    Prep::Run(Box::new(PairRun { cam1, cam2, kps1, kps2, matches, opts: o, path, x1, x2, r1, r2, e_err }))
}

/// 짝 묶음 검증. 짝마다 E/F/H LO-RANSAC 을 나란히 진행하고, 라운드마다 모든 짝의 가설을
/// `scorer` 호출 한 번으로 채점한다. 결과 규칙·순서는 [`CpuTwoView`] 와 같다.
/// 다중 모델 경로(`multiple_models`)는 CPU 로 계산한다. 채점기 오류는 그대로 돌려준다.
pub fn verify_pairs_batched(
    store: &FeatureStore,
    jobs: &[VerifyJob<'_>],
    opts: &TwoViewOptions,
    scorer: &mut dyn BatchScorer,
) -> Result<Vec<TwoViewGeometry>, String> {
    let preps: Vec<Prep> = jobs.par_iter().map(|j| prepare(store, j, opts)).collect();

    // 짝 → RANSAC 상태(0 = E, 1 = F, 2 = H).
    let mut steppers: Vec<Stepper<'_>> = Vec::new();
    let mut sets: Vec<ResidualSet<'_>> = Vec::new();
    let mut owner: Vec<(usize, u8)> = Vec::new();
    for (i, p) in preps.iter().enumerate() {
        let Prep::Run(r) = p else { continue };
        let o = &r.opts;
        if r.path == EstimationPath::Calibrated {
            let e = EssentialFivePointEstimator;
            let ro = ransac_opts(o, r.e_err, SALT_E);
            steppers.push(Stepper::E(BatchedRansac::new(
                e,
                Some(e),
                RandomSubsetSampler::new(e.min_num_samples()),
                &ro,
                &r.r1[..],
                &r.r2[..],
            )));
            sets.push(ResidualSet::EssentialRays(&r.r1, &r.r2));
            owner.push((i, 0));
        }
        if r.path != EstimationPath::HomographyOnly {
            let ro = ransac_opts(o, o.ransac.max_error, SALT_F);
            let s = RandomSubsetSampler::new(Fundamental7PtEstimator.min_num_samples());
            steppers.push(Stepper::F(BatchedRansac::new(
                Fundamental7PtEstimator,
                Some(FundamentalEightPointEstimator),
                s,
                &ro,
                &r.x1[..],
                &r.x2[..],
            )));
            sets.push(ResidualSet::FundamentalPx(&r.x1, &r.x2));
            owner.push((i, 1));
        }
        let h = HomographyEstimator { normalize: o.normalize_homography };
        let ro = ransac_opts(o, o.ransac.max_error, SALT_H);
        steppers.push(Stepper::H(BatchedRansac::new(h, Some(h), RandomSubsetSampler::new(h.min_num_samples()), &ro, &r.x1[..], &r.x2[..])));
        sets.push(ResidualSet::HomographyPx(&r.x1, &r.x2));
        owner.push((i, 2));
    }

    if !steppers.is_empty() {
        scorer.load(&sets)?;
    }
    drop(sets);
    loop {
        let active: Vec<usize> = (0..steppers.len()).filter(|&i| !steppers[i].is_done()).collect();
        if active.is_empty() {
            break;
        }
        steppers.par_iter_mut().filter(|s| !s.is_done()).for_each(|s| s.propose());
        let scores = {
            let reqs: Vec<ScoreRequest<'_>> = active
                .iter()
                .map(|&i| ScoreRequest { slot: i, models: steppers[i].models(), max_residual: steppers[i].max_residual() })
                .collect();
            scorer.score(&reqs)?
        };
        if scores.len() != active.len() {
            return Err(format!("채점 결과 수 {} != 요청 수 {}", scores.len(), active.len()));
        }
        let mut per: Vec<Option<Vec<Support>>> = (0..steppers.len()).map(|_| None).collect();
        for (&i, s) in active.iter().zip(scores) {
            if s.len() != steppers[i].models().len() {
                return Err(format!("슬롯 {i}: 지지도 수 {} != 모델 수 {}", s.len(), steppers[i].models().len()));
            }
            per[i] = Some(s);
        }
        steppers.par_iter_mut().zip(per.into_par_iter()).for_each(|(st, s)| {
            if let Some(s) = s {
                st.consume(&s);
            }
        });
    }

    let reports: Vec<RansacReport<Mat3>> = steppers.into_iter().map(Stepper::finish).collect();
    let mut per_pair: Vec<[Option<RansacReport<Mat3>>; 3]> = (0..preps.len()).map(|_| [None, None, None]).collect();
    for ((i, w), rep) in owner.into_iter().zip(reports) {
        per_pair[i][w as usize] = Some(rep);
    }
    let min_inl = opts.min_num_inliers;
    let out = preps
        .into_par_iter()
        .zip(per_pair.into_par_iter())
        .map(|(p, [e, f, h])| match p {
            Prep::Done(g) => g,
            Prep::Run(r) => {
                let k1: &[Keypoint] = &r.kps1[..];
                let k2: &[Keypoint] = &r.kps2[..];
                let g = match r.path {
                    EstimationPath::Calibrated => assemble_calibrated(
                        &r.cam1,
                        k1,
                        &r.cam2,
                        k2,
                        &r.matches,
                        &r.opts,
                        e.expect("E 결과"),
                        f.expect("F 결과"),
                        h.expect("H 결과"),
                    ),
                    EstimationPath::Uncalibrated => {
                        assemble_uncalibrated(&r.cam1, k1, &r.cam2, k2, &r.matches, &r.opts, f.expect("F 결과"), h.expect("H 결과"))
                    }
                    EstimationPath::HomographyOnly => {
                        assemble_homography_only(&r.cam1, k1, &r.cam2, k2, &r.matches, &r.opts, h.expect("H 결과"))
                    }
                    EstimationPath::Multiple => unreachable!("다중 모델은 준비 단계에서 끝남"),
                };
                finalize_geometry(g, min_inl)
            }
        })
        .collect();
    Ok(out)
}

/// 시험용 합성 저장소: 같은 핀홀 카메라(초점 사전값 있음)로 찍은 영상 `n_images` 장과,
/// 이웃 영상 짝마다 참 대응 + 무작위 이상치 매칭. (저장소, [(id1, id2, 매칭)]).
#[doc(hidden)]
pub fn synthetic_pairs(
    seed: u64,
    n_images: usize,
    n_points: usize,
    outlier_ratio: f64,
) -> (FeatureStore, Vec<(ImageId, ImageId, Vec<FeatureMatch>)>) {
    use cumulus3d_core::CameraModelKind;
    use rand::{RngExt, SeedableRng};
    let mut r = rand_pcg::Pcg64::seed_from_u64(seed);
    let store = FeatureStore::new();
    let mut cam = Camera::new(0, CameraModelKind::Pinhole, 2000, 1500, vec![1000.0, 1000.0, 1000.0, 750.0]).expect("카메라");
    cam.focal_from_prior = true;
    cam.camera_id = cumulus3d_core::INVALID_CAMERA_ID;
    let cid = store.add_camera(cam.clone()).expect("카메라 추가");
    let pts: Vec<Vec3> =
        (0..n_points).map(|_| Vec3::new(r.random_range(-6.0..6.0), r.random_range(-5.0..5.0), r.random_range(6.0..15.0))).collect();
    let mut ids = Vec::new();
    // 영상마다 (점 번호 → 키포인트 번호).
    let mut index: Vec<Vec<Option<u32>>> = Vec::new();
    for k in 0..n_images {
        let id = store.add_image(&format!("cam/{k}.jpg"), cid).expect("영상 추가");
        let cx = 0.5 * k as f64;
        let mut kps = Vec::new();
        let mut map = vec![None; n_points];
        for (i, p) in pts.iter().enumerate() {
            let xc = p - Vec3::new(cx, 0.0, 0.0);
            let Some(q) = cam.cam_to_img(&xc) else { continue };
            if q.x < 0.0 || q.y < 0.0 || q.x > 2000.0 || q.y > 1500.0 {
                continue;
            }
            map[i] = Some(kps.len() as u32);
            kps.push(Keypoint::new((q.x + 0.3 * r.random_range(-1.0..1.0)) as f32, (q.y + 0.3 * r.random_range(-1.0..1.0)) as f32));
        }
        for _ in 0..(n_points / 5) {
            kps.push(Keypoint::new(r.random_range(0.0..2000.0f32), r.random_range(0.0..1500.0f32)));
        }
        store.set_keypoints(id, kps);
        ids.push(id);
        index.push(map);
    }
    let mut pairs = Vec::new();
    for a in 0..n_images {
        for b in a + 1..n_images {
            let n2 = store.num_keypoints(ids[b]) as u32;
            let mut m: Vec<FeatureMatch> = Vec::new();
            for i in 0..n_points {
                let (Some(ka), Some(kb)) = (index[a][i], index[b][i]) else { continue };
                if r.random_range(0.0..1.0) < outlier_ratio {
                    m.push(FeatureMatch::new(ka, r.random_range(0..n2)));
                } else {
                    m.push(FeatureMatch::new(ka, kb));
                }
            }
            pairs.push((ids[a], ids[b], m));
        }
    }
    (store, pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_cpu_scorer_equals_cpu_backend() {
        let (store, pairs) = synthetic_pairs(11, 4, 600, 0.4);
        let jobs: Vec<VerifyJob<'_>> = pairs.iter().map(|(a, b, m)| VerifyJob { id1: *a, id2: *b, matches: m }).collect();
        for (seed, calibrated, force_h) in [(Some(3u64), true, false), (Some(5), false, false), (Some(7), true, true)] {
            let mut opts = TwoViewOptions::default();
            opts.ransac.random_seed = seed;
            opts.force_h_use = force_h;
            if !calibrated {
                // 비보정 경로: 초점 사전값 없는 카메라로 바꾼 저장소.
                let s2 = FeatureStore::new();
                let mut ids: Vec<ImageId> = pairs.iter().flat_map(|(a, b, _)| [*a, *b]).collect();
                ids.sort();
                ids.dedup();
                let mut cam = store.camera(store.image(ids[0]).expect("영상").camera_id).expect("카메라");
                cam.focal_from_prior = false;
                cam.camera_id = cumulus3d_core::INVALID_CAMERA_ID;
                let cid = s2.add_camera(cam).expect("카메라");
                for &id in &ids {
                    let im = store.image(id).expect("영상");
                    s2.add_image_with_id(id, &im.name, cid).expect("영상");
                    s2.set_keypoints(id, store.keypoints(id).expect("키포인트").to_vec());
                }
                let want = CpuTwoView.verify_pairs(&s2, &jobs, &opts);
                let got = verify_pairs_batched(&s2, &jobs, &opts, &mut CpuBatchScorer::default()).expect("묶음 검증");
                assert_eq!(got, want);
                continue;
            }
            let want = CpuTwoView.verify_pairs(&store, &jobs, &opts);
            let got = verify_pairs_batched(&store, &jobs, &opts, &mut CpuBatchScorer::default()).expect("묶음 검증");
            assert_eq!(got, want, "seed {seed:?} force_h {force_h}");
            assert!(got.iter().any(|g| g.inlier_matches.len() > 100));
        }
    }
}
