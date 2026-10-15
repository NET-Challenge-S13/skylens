/*
 * two_view.rs
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

//! 두 뷰 기하 검증.

use crate::estimators::{
    EssentialFivePointEstimator, Fundamental7PtEstimator, FundamentalEightPointEstimator, HomographyEstimator, TranslationEstimator,
};
use crate::pose::recover_two_view_pose;
use cumulus3d_core::ransac::{lo_ransac, RansacParams, RansacReport};
use cumulus3d_core::{Camera, FeatureMatch, Keypoint, Mat3, TwoViewGeometry, TwoViewGeometryConfig, Vec2, Vec3};

/// 두 뷰 기하 옵션.
#[derive(Clone, Debug, PartialEq)]
pub struct TwoViewOptions {
    /// 유효 기하로 인정할 최소 인라이어 수.
    pub min_num_inliers: usize,
    /// 판정용 최소 인라이어 비율(0 = 꺼짐). 0 보다 크면 RANSAC 사전 비율도 덮어씀.
    pub min_inlier_ratio: f64,
    /// E 인라이어 / F 인라이어 비율이 이보다 크면 E(보정) 쪽 판정.
    pub min_e_f_inliers_ratio: f64,
    /// H 인라이어 / E(또는 F) 인라이어 비율이 이보다 크면 평면·회전 구성으로 판정.
    pub max_h_inliers_ratio: f64,
    /// 워터마크(영상 가장자리 고정 매칭) 검출 여부.
    pub detect_watermark: bool,
    /// 가장자리 인라이어 비율과 평행이동 인라이어 비율이 이보다 크면 워터마크로 판정.
    pub watermark_inlier_ratio: f64,
    /// 가장자리 띠 폭(영상 대각선 길이 대비 비율).
    pub watermark_band: f64,
    /// 워터마크 평행이동 RANSAC 오차 임계(px).
    pub watermark_detection_max_error: f64,
    /// 인라이어를 제거하며 여러 기하를 반복 추정.
    pub multiple_models: bool,
    /// 다중 모델 모드에서 워터마크 기하를 결과에서 제외.
    pub multiple_ignore_watermark: bool,
    /// 거의 움직이지 않는 매칭을 사전 제거.
    pub filter_stationary_matches: bool,
    /// 정지 매칭 판정 이동량 임계(px).
    pub stationary_matches_max_error: f64,
    /// H 만 추정하는 경로 사용(평면 구성 강제).
    pub force_h_use: bool,
    /// 기하 확정 후 상대 자세·삼각측량 각도도 계산.
    pub compute_relative_pose: bool,
    /// 하틀리 정규화 DLT(개선; 결과가 미세하게 달라지므로 기본 끔).
    pub normalize_homography: bool,
    /// 보정 경로에서 E/F/H 세 RANSAC 을 rayon::join 으로 동시 실행(결과 불변).
    pub parallel_models: bool,
    /// RANSAC: max_error 4 px, 사전 비율 0.25, 신뢰도 0.999, 배수 3, 반복 100..10000.
    /// `random_seed` 가 있으면 모델 종류별로 파생 시드를 쓴다.
    pub ransac: RansacParams,
}

impl Default for TwoViewOptions {
    fn default() -> Self {
        Self {
            min_num_inliers: 15,
            min_inlier_ratio: 0.0,
            min_e_f_inliers_ratio: 0.95,
            max_h_inliers_ratio: 0.8,
            detect_watermark: true,
            watermark_inlier_ratio: 0.7,
            watermark_band: 0.1,
            watermark_detection_max_error: 4.0,
            multiple_models: false,
            multiple_ignore_watermark: true,
            filter_stationary_matches: false,
            stationary_matches_max_error: 4.0,
            force_h_use: false,
            compute_relative_pose: false,
            normalize_homography: false,
            parallel_models: true,
            ransac: RansacParams {
                max_error: 4.0,
                min_inlier_ratio: 0.25,
                confidence: 0.999,
                dyn_trials_factor: 3.0,
                min_trials: 100,
                max_trials: 10000,
                random_seed: None,
            },
        }
    }
}

/// 모델 종류별 시드 파생(splitmix64).
pub fn derive_seed(seed: Option<u64>, salt: u64) -> Option<u64> {
    seed.map(|s| {
        let mut z = s ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    })
}

const SALT_E: u64 = 1;
const SALT_F: u64 = 2;
const SALT_H: u64 = 3;
const SALT_T: u64 = 4;

/// 선택된 인라이어 마스크.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskChoice {
    /// 본질 행렬 인라이어.
    E,
    /// 기초 행렬 인라이어.
    F,
    /// 호모그래피 인라이어.
    H,
}

/// 판정 결과: 구성과 사용할 마스크(DEGENERATE 이면 None).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decision {
    /// 판정된 기하 구성.
    pub config: TwoViewGeometryConfig,
    /// 사용할 인라이어 마스크(DEGENERATE 이면 None).
    pub mask: Option<MaskChoice>,
}

/// 한 RANSAC 의 (성공 여부, 인라이어 수).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ModelOutcome {
    /// RANSAC 성공 여부.
    pub success: bool,
    /// 인라이어 수.
    pub num_inliers: usize,
}

/// 보정 경로 판정.
pub fn decide_calibrated(e: ModelOutcome, f: ModelOutcome, h: ModelOutcome, opts: &TwoViewOptions) -> Decision {
    use TwoViewGeometryConfig as C;
    let min = opts.min_num_inliers;
    let degenerate = Decision { config: C::Degenerate, mask: None };
    if (!e.success && !f.success && !h.success) || (e.num_inliers < min && f.num_inliers < min && h.num_inliers < min) {
        return degenerate;
    }
    let (ne, nf, nh) = (e.num_inliers as f64, f.num_inliers as f64, h.num_inliers as f64);
    let r_ef = ne / nf;
    let r_hf = nh / nf;
    let r_he = nh / ne;
    if e.success && r_ef > opts.min_e_f_inliers_ratio && e.num_inliers >= min {
        let (mut mask, n) = if e.num_inliers >= f.num_inliers { (MaskChoice::E, e.num_inliers) } else { (MaskChoice::F, f.num_inliers) };
        let config = if r_he > opts.max_h_inliers_ratio {
            if h.num_inliers > n {
                mask = MaskChoice::H;
            }
            C::PlanarOrRotation
        } else {
            C::Calibrated
        };
        Decision { config, mask: Some(mask) }
    } else if f.success && f.num_inliers >= min {
        if r_hf > opts.max_h_inliers_ratio {
            let mask = if h.num_inliers > f.num_inliers { MaskChoice::H } else { MaskChoice::F };
            Decision { config: C::PlanarOrRotation, mask: Some(mask) }
        } else {
            Decision { config: C::Uncalibrated, mask: Some(MaskChoice::F) }
        }
    } else if h.success && h.num_inliers >= min {
        Decision { config: C::PlanarOrRotation, mask: Some(MaskChoice::H) }
    } else {
        degenerate
    }
}

/// 비보정 경로 판정. n_H ≥ n_F(같아도) 이면 H 마스크.
pub fn decide_uncalibrated(f: ModelOutcome, h: ModelOutcome, opts: &TwoViewOptions) -> Decision {
    use TwoViewGeometryConfig as C;
    let min = opts.min_num_inliers;
    if (!f.success && !h.success) || (f.num_inliers < min && h.num_inliers < min) {
        return Decision { config: C::Degenerate, mask: None };
    }
    let r_hf = h.num_inliers as f64 / f.num_inliers as f64;
    if r_hf > opts.max_h_inliers_ratio {
        let mask = if h.num_inliers >= f.num_inliers { MaskChoice::H } else { MaskChoice::F };
        Decision { config: C::PlanarOrRotation, mask: Some(mask) }
    } else {
        Decision { config: C::Uncalibrated, mask: Some(MaskChoice::F) }
    }
}

fn outcome<M>(r: &RansacReport<M>) -> ModelOutcome {
    ModelOutcome { success: r.success, num_inliers: r.support.num_inliers }
}

fn kp_xy(k: &Keypoint) -> Vec2 {
    Vec2::new(k.x as f64, k.y as f64)
}

fn select_inliers(matches: &[FeatureMatch], mask: &[bool]) -> Vec<FeatureMatch> {
    matches.iter().zip(mask.iter().chain(std::iter::repeat(&false))).filter(|(_, &m)| m).map(|(m, _)| *m).collect()
}

fn ransac_opts(opts: &TwoViewOptions, max_error: f64, salt: u64) -> RansacParams {
    let mut r = opts.ransac.clone();
    r.max_error = max_error;
    if opts.min_inlier_ratio > 0.0 {
        r.min_inlier_ratio = opts.min_inlier_ratio;
    }
    r.random_seed = derive_seed(opts.ransac.random_seed, salt);
    r
}

fn run_h(opts: &TwoViewOptions, x1: &[Vec2], x2: &[Vec2]) -> RansacReport<Mat3> {
    let h = HomographyEstimator { normalize: opts.normalize_homography };
    lo_ransac(&h, &h, &ransac_opts(opts, opts.ransac.max_error, SALT_H), x1, x2)
}

fn run_f(opts: &TwoViewOptions, x1: &[Vec2], x2: &[Vec2]) -> RansacReport<Mat3> {
    lo_ransac(&Fundamental7PtEstimator, &FundamentalEightPointEstimator, &ransac_opts(opts, opts.ransac.max_error, SALT_F), x1, x2)
}

fn run_e(opts: &TwoViewOptions, max_error: f64, r1: &[Vec3], r2: &[Vec3]) -> RansacReport<Mat3> {
    let e = EssentialFivePointEstimator;
    lo_ransac(&e, &e, &ransac_opts(opts, max_error, SALT_E), r1, r2)
}

/// 두 뷰 기하 추정(원시 결과; 15 미만 → 기본값 변환은 호출자 몫, `finalize_geometry`).
pub fn estimate_two_view(
    cam1: &Camera,
    kps1: &[Keypoint],
    cam2: &Camera,
    kps2: &[Keypoint],
    matches: &[FeatureMatch],
    opts: &TwoViewOptions,
) -> TwoViewGeometry {
    let filtered;
    let matches = if opts.filter_stationary_matches {
        let t2 = opts.stationary_matches_max_error * opts.stationary_matches_max_error;
        filtered = matches
            .iter()
            .filter(|m| (kp_xy(&kps1[m.idx1 as usize]) - kp_xy(&kps2[m.idx2 as usize])).norm_squared() > t2)
            .copied()
            .collect::<Vec<_>>();
        &filtered[..]
    } else {
        matches
    };
    if opts.multiple_models {
        estimate_multiple(cam1, kps1, cam2, kps2, matches, opts)
    } else if opts.force_h_use {
        estimate_homography_only(cam1, kps1, cam2, kps2, matches, opts)
    } else if cam1.focal_from_prior && cam2.focal_from_prior {
        estimate_calibrated(cam1, kps1, cam2, kps2, matches, opts)
    } else {
        estimate_uncalibrated(cam1, kps1, cam2, kps2, matches, opts)
    }
}

/// 인라이어가 `min_num_inliers` 미만이면 기본값(UNDEFINED, 빈 인라이어).
pub fn finalize_geometry(tvg: TwoViewGeometry, min_num_inliers: usize) -> TwoViewGeometry {
    if tvg.inlier_matches.len() < min_num_inliers {
        TwoViewGeometry::default()
    } else {
        tvg
    }
}

fn degenerate() -> TwoViewGeometry {
    TwoViewGeometry { config: TwoViewGeometryConfig::Degenerate, ..Default::default() }
}

fn gather_points(kps1: &[Keypoint], kps2: &[Keypoint], matches: &[FeatureMatch]) -> (Vec<Vec2>, Vec<Vec2>) {
    matches.iter().map(|m| (kp_xy(&kps1[m.idx1 as usize]), kp_xy(&kps2[m.idx2 as usize]))).unzip()
}

fn finish(
    mut tvg: TwoViewGeometry,
    cam1: &Camera,
    kps1: &[Keypoint],
    cam2: &Camera,
    kps2: &[Keypoint],
    opts: &TwoViewOptions,
) -> TwoViewGeometry {
    if opts.detect_watermark && is_watermark(cam1, kps1, cam2, kps2, &tvg.inlier_matches, opts) {
        tvg.config = TwoViewGeometryConfig::Watermark;
    }
    if opts.compute_relative_pose {
        recover_two_view_pose(cam1, cam2, kps1, kps2, &mut tvg);
    }
    tvg
}

fn estimate_calibrated(
    cam1: &Camera,
    kps1: &[Keypoint],
    cam2: &Camera,
    kps2: &[Keypoint],
    matches: &[FeatureMatch],
    opts: &TwoViewOptions,
) -> TwoViewGeometry {
    if matches.len() < opts.min_num_inliers {
        return degenerate();
    }
    let (x1, x2) = gather_points(kps1, kps2, matches);
    let r1: Vec<Vec3> = x1.iter().map(|p| cam1.img_to_ray(p).unwrap_or_else(Vec3::zeros)).collect();
    let r2: Vec<Vec3> = x2.iter().map(|p| cam2.img_to_ray(p).unwrap_or_else(Vec3::zeros)).collect();
    let e_err = (opts.ransac.max_error / cam1.mean_focal() + opts.ransac.max_error / cam2.mean_focal()) / 2.0;

    let (er, (fr, hr)) = if opts.parallel_models {
        rayon::join(|| run_e(opts, e_err, &r1, &r2), || rayon::join(|| run_f(opts, &x1, &x2), || run_h(opts, &x1, &x2)))
    } else {
        (run_e(opts, e_err, &r1, &r2), (run_f(opts, &x1, &x2), run_h(opts, &x1, &x2)))
    };
    let d = decide_calibrated(outcome(&er), outcome(&fr), outcome(&hr), opts);
    let Some(mask) = d.mask else { return degenerate() };
    let m = match mask {
        MaskChoice::E => &er.inlier_mask,
        MaskChoice::F => &fr.inlier_mask,
        MaskChoice::H => &hr.inlier_mask,
    };
    let inliers = select_inliers(matches, m);
    // 판정용 최소 인라이어 비율(기본 꺼짐).
    if opts.min_inlier_ratio > 0.0 && (inliers.len() as f64 / matches.len() as f64) < opts.min_inlier_ratio {
        return degenerate();
    }
    // 설계 결정: 실패한 RANSAC 의 모델은 버리고 성공한 모델만 기록한다.
    let tvg = TwoViewGeometry {
        config: d.config,
        e: er.success.then_some(er.model).flatten(),
        f: fr.success.then_some(fr.model).flatten(),
        h: hr.success.then_some(hr.model).flatten(),
        inlier_matches: inliers,
        ..Default::default()
    };
    finish(tvg, cam1, kps1, cam2, kps2, opts)
}

fn estimate_uncalibrated(
    cam1: &Camera,
    kps1: &[Keypoint],
    cam2: &Camera,
    kps2: &[Keypoint],
    matches: &[FeatureMatch],
    opts: &TwoViewOptions,
) -> TwoViewGeometry {
    if matches.len() < opts.min_num_inliers {
        return degenerate();
    }
    let (x1, x2) = gather_points(kps1, kps2, matches);
    let (fr, hr) = if opts.parallel_models {
        rayon::join(|| run_f(opts, &x1, &x2), || run_h(opts, &x1, &x2))
    } else {
        (run_f(opts, &x1, &x2), run_h(opts, &x1, &x2))
    };
    let d = decide_uncalibrated(outcome(&fr), outcome(&hr), opts);
    let Some(mask) = d.mask else { return degenerate() };
    let m = if mask == MaskChoice::H { &hr.inlier_mask } else { &fr.inlier_mask };
    let tvg = TwoViewGeometry {
        config: d.config,
        f: fr.success.then_some(fr.model).flatten(),
        h: hr.success.then_some(hr.model).flatten(),
        inlier_matches: select_inliers(matches, m),
        ..Default::default()
    };
    finish(tvg, cam1, kps1, cam2, kps2, opts)
}

fn estimate_homography_only(
    cam1: &Camera,
    kps1: &[Keypoint],
    cam2: &Camera,
    kps2: &[Keypoint],
    matches: &[FeatureMatch],
    opts: &TwoViewOptions,
) -> TwoViewGeometry {
    if matches.len() < opts.min_num_inliers {
        return degenerate();
    }
    let (x1, x2) = gather_points(kps1, kps2, matches);
    let hr = run_h(opts, &x1, &x2);
    if !hr.success || hr.support.num_inliers < opts.min_num_inliers {
        return degenerate();
    }
    let tvg = TwoViewGeometry {
        config: TwoViewGeometryConfig::PlanarOrRotation,
        h: hr.model,
        inlier_matches: select_inliers(matches, &hr.inlier_mask),
        ..Default::default()
    };
    finish(tvg, cam1, kps1, cam2, kps2, opts)
}

fn estimate_multiple(
    cam1: &Camera,
    kps1: &[Keypoint],
    cam2: &Camera,
    kps2: &[Keypoint],
    matches: &[FeatureMatch],
    opts: &TwoViewOptions,
) -> TwoViewGeometry {
    let mut single = opts.clone();
    single.multiple_models = false;
    let mut remaining = matches.to_vec();
    let mut found: Vec<TwoViewGeometry> = Vec::new();
    loop {
        let g = estimate_two_view(cam1, kps1, cam2, kps2, &remaining, &single);
        if g.config == TwoViewGeometryConfig::Degenerate || g.inlier_matches.is_empty() {
            break;
        }
        let used: std::collections::HashSet<FeatureMatch> = g.inlier_matches.iter().copied().collect();
        remaining.retain(|m| !used.contains(m));
        // 설계 결정: 워터마크 무시 시에도 그 인라이어는 제거하고 계속 진행.
        if !(opts.multiple_ignore_watermark && g.config == TwoViewGeometryConfig::Watermark) {
            found.push(g);
        }
    }
    match found.len() {
        0 => degenerate(),
        1 => found.pop().expect("1개"),
        _ => TwoViewGeometry {
            config: TwoViewGeometryConfig::Multiple,
            inlier_matches: found.into_iter().flat_map(|g| g.inlier_matches).collect(),
            ..Default::default()
        },
    }
}

/// 워터마크 검출.
pub fn is_watermark(
    cam1: &Camera,
    kps1: &[Keypoint],
    cam2: &Camera,
    kps2: &[Keypoint],
    inliers: &[FeatureMatch],
    opts: &TwoViewOptions,
) -> bool {
    if inliers.is_empty() {
        return false;
    }
    let inside = |cam: &Camera, p: &Vec2| -> bool {
        let (w, h) = (cam.width as f64, cam.height as f64);
        let b = opts.watermark_band * (w * w + h * h).sqrt();
        p.x >= b && p.x <= w - b && p.y >= b && p.y <= h - b
    };
    let (x1, x2) = gather_points(kps1, kps2, inliers);
    let border = x1.iter().zip(&x2).filter(|(a, b)| !inside(cam1, a) && !inside(cam2, b)).count();
    let n = inliers.len() as f64;
    if (border as f64) / n < opts.watermark_inlier_ratio {
        return false;
    }
    let mut r = ransac_opts(opts, opts.watermark_detection_max_error, SALT_T);
    r.min_inlier_ratio = opts.watermark_inlier_ratio;
    let rep = lo_ransac(&TranslationEstimator, &TranslationEstimator, &r, &x1, &x2);
    let nt = if rep.success { rep.support.num_inliers } else { 0 };
    (nt as f64) / n >= opts.watermark_inlier_ratio
}
