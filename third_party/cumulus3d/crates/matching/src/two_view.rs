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

//! Two-view geometry verification.

use crate::estimators::{
    EssentialFivePointEstimator, Fundamental7PtEstimator, FundamentalEightPointEstimator, HomographyEstimator, TranslationEstimator,
};
use crate::pose::recover_two_view_pose;
use cumulus3d_core::ransac::{lo_ransac, RansacParams, RansacReport};
use cumulus3d_core::{Camera, FeatureMatch, Keypoint, Mat3, TwoViewGeometry, TwoViewGeometryConfig, Vec2, Vec3};

/// Two-view geometry options.
#[derive(Clone, Debug, PartialEq)]
pub struct TwoViewOptions {
    /// Minimum number of inliers to accept as a valid geometry.
    pub min_num_inliers: usize,
    /// Minimum inlier ratio for the decision (0 = off). If greater than 0, it also overrides the RANSAC prior ratio.
    pub min_inlier_ratio: f64,
    /// If the E inlier / F inlier ratio exceeds this, decide on the E (calibrated) side.
    pub min_e_f_inliers_ratio: f64,
    /// If the H inlier / E (or F) inlier ratio exceeds this, decide on a planar/rotational configuration.
    pub max_h_inliers_ratio: f64,
    /// Whether to detect watermarks (fixed matches at image edges).
    pub detect_watermark: bool,
    /// If both the edge-inlier ratio and the translation-inlier ratio exceed this, decide it is a watermark.
    pub watermark_inlier_ratio: f64,
    /// Edge strip width (as a fraction of the image diagonal length).
    pub watermark_band: f64,
    /// Watermark translation RANSAC error threshold (px).
    pub watermark_detection_max_error: f64,
    /// Repeatedly estimate several geometries while removing inliers.
    pub multiple_models: bool,
    /// In multi-model mode, exclude the watermark geometry from the result.
    pub multiple_ignore_watermark: bool,
    /// Pre-remove matches that barely move.
    pub filter_stationary_matches: bool,
    /// Movement threshold (px) for deciding a stationary match.
    pub stationary_matches_max_error: f64,
    /// Use the path that estimates only H (forces a planar configuration).
    pub force_h_use: bool,
    /// After the geometry is confirmed, also compute the relative pose and triangulation angle.
    pub compute_relative_pose: bool,
    /// Hartley-normalized DLT (improvement; off by default since results change slightly).
    pub normalize_homography: bool,
    /// On the calibrated path, run the E/F/H RANSACs concurrently with rayon::join (results unchanged).
    pub parallel_models: bool,
    /// RANSAC: max_error 4 px, prior ratio 0.25, confidence 0.999, multiplier 3, iterations 100..10000.
    /// If `random_seed` is set, a derived seed is used per model type.
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

/// Per-model-type seed derivation (splitmix64).
pub fn derive_seed(seed: Option<u64>, salt: u64) -> Option<u64> {
    seed.map(|s| {
        let mut z = s ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    })
}

pub(crate) const SALT_E: u64 = 1;
pub(crate) const SALT_F: u64 = 2;
pub(crate) const SALT_H: u64 = 3;
const SALT_T: u64 = 4;

/// Selected inlier mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskChoice {
    /// Essential matrix inliers.
    E,
    /// Fundamental matrix inliers.
    F,
    /// Homography inliers.
    H,
}

/// Decision result: configuration and the mask to use (None if DEGENERATE).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decision {
    /// Decided geometric configuration.
    pub config: TwoViewGeometryConfig,
    /// Inlier mask to use (None if DEGENERATE).
    pub mask: Option<MaskChoice>,
}

/// (success flag, inlier count) of one RANSAC.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ModelOutcome {
    /// Whether the RANSAC succeeded.
    pub success: bool,
    /// Inlier count.
    pub num_inliers: usize,
}

/// Calibrated-path decision.
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

/// Uncalibrated-path decision. If n_H ≥ n_F (even when equal), use the H mask.
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

pub(crate) fn ransac_opts(opts: &TwoViewOptions, max_error: f64, salt: u64) -> RansacParams {
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

/// Two-view geometry estimation (raw result; the conversion to defaults below 15 is the caller's job, `finalize_geometry`).
pub fn estimate_two_view(
    cam1: &Camera,
    kps1: &[Keypoint],
    cam2: &Camera,
    kps2: &[Keypoint],
    matches: &[FeatureMatch],
    opts: &TwoViewOptions,
) -> TwoViewGeometry {
    let matches = filter_stationary(kps1, kps2, matches, opts);
    let matches = &matches[..];
    match path_of(cam1, cam2, opts) {
        EstimationPath::Multiple => estimate_multiple(cam1, kps1, cam2, kps2, matches, opts),
        EstimationPath::HomographyOnly => estimate_homography_only(cam1, kps1, cam2, kps2, matches, opts),
        EstimationPath::Calibrated => estimate_calibrated(cam1, kps1, cam2, kps2, matches, opts),
        EstimationPath::Uncalibrated => estimate_uncalibrated(cam1, kps1, cam2, kps2, matches, opts),
    }
}

/// 추정 경로.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EstimationPath {
    /// 다중 모델(인라이어를 지우며 반복).
    Multiple,
    /// H 만.
    HomographyOnly,
    /// E/F/H.
    Calibrated,
    /// F/H.
    Uncalibrated,
}

pub(crate) fn path_of(cam1: &Camera, cam2: &Camera, opts: &TwoViewOptions) -> EstimationPath {
    if opts.multiple_models {
        EstimationPath::Multiple
    } else if opts.force_h_use {
        EstimationPath::HomographyOnly
    } else if cam1.focal_from_prior && cam2.focal_from_prior {
        EstimationPath::Calibrated
    } else {
        EstimationPath::Uncalibrated
    }
}

/// 정지 매칭 사전 제거(옵션이 꺼져 있으면 그대로).
pub(crate) fn filter_stationary<'m>(
    kps1: &[Keypoint],
    kps2: &[Keypoint],
    matches: &'m [FeatureMatch],
    opts: &TwoViewOptions,
) -> std::borrow::Cow<'m, [FeatureMatch]> {
    if opts.filter_stationary_matches {
        let t2 = opts.stationary_matches_max_error * opts.stationary_matches_max_error;
        std::borrow::Cow::Owned(
            matches
                .iter()
                .filter(|m| (kp_xy(&kps1[m.idx1 as usize]) - kp_xy(&kps2[m.idx2 as usize])).norm_squared() > t2)
                .copied()
                .collect(),
        )
    } else {
        std::borrow::Cow::Borrowed(matches)
    }
}

/// If inliers are below `min_num_inliers`, defaults (UNDEFINED, empty inliers).
pub fn finalize_geometry(tvg: TwoViewGeometry, min_num_inliers: usize) -> TwoViewGeometry {
    if tvg.inlier_matches.len() < min_num_inliers {
        TwoViewGeometry::default()
    } else {
        tvg
    }
}

pub(crate) fn degenerate() -> TwoViewGeometry {
    TwoViewGeometry { config: TwoViewGeometryConfig::Degenerate, ..Default::default() }
}

pub(crate) fn gather_points(kps1: &[Keypoint], kps2: &[Keypoint], matches: &[FeatureMatch]) -> (Vec<Vec2>, Vec<Vec2>) {
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

/// 보정 경로 입력: (픽셀1, 픽셀2, 광선1, 광선2, E 오차 임계).
pub(crate) type CalibratedInputs = (Vec<Vec2>, Vec<Vec2>, Vec<Vec3>, Vec<Vec3>, f64);

pub(crate) fn calibrated_inputs(
    cam1: &Camera,
    kps1: &[Keypoint],
    cam2: &Camera,
    kps2: &[Keypoint],
    matches: &[FeatureMatch],
    opts: &TwoViewOptions,
) -> CalibratedInputs {
    let (x1, x2) = gather_points(kps1, kps2, matches);
    let r1: Vec<Vec3> = x1.iter().map(|p| cam1.img_to_ray(p).unwrap_or_else(Vec3::zeros)).collect();
    let r2: Vec<Vec3> = x2.iter().map(|p| cam2.img_to_ray(p).unwrap_or_else(Vec3::zeros)).collect();
    let e_err = (opts.ransac.max_error / cam1.mean_focal() + opts.ransac.max_error / cam2.mean_focal()) / 2.0;
    (x1, x2, r1, r2, e_err)
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
    let (x1, x2, r1, r2, e_err) = calibrated_inputs(cam1, kps1, cam2, kps2, matches, opts);
    let (er, (fr, hr)) = if opts.parallel_models {
        rayon::join(|| run_e(opts, e_err, &r1, &r2), || rayon::join(|| run_f(opts, &x1, &x2), || run_h(opts, &x1, &x2)))
    } else {
        (run_e(opts, e_err, &r1, &r2), (run_f(opts, &x1, &x2), run_h(opts, &x1, &x2)))
    };
    assemble_calibrated(cam1, kps1, cam2, kps2, matches, opts, er, fr, hr)
}

/// 보정 경로 마무리: E/F/H RANSAC 결과로 판정·인라이어·워터마크.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_calibrated(
    cam1: &Camera,
    kps1: &[Keypoint],
    cam2: &Camera,
    kps2: &[Keypoint],
    matches: &[FeatureMatch],
    opts: &TwoViewOptions,
    er: RansacReport<Mat3>,
    fr: RansacReport<Mat3>,
    hr: RansacReport<Mat3>,
) -> TwoViewGeometry {
    let d = decide_calibrated(outcome(&er), outcome(&fr), outcome(&hr), opts);
    let Some(mask) = d.mask else { return degenerate() };
    let m = match mask {
        MaskChoice::E => &er.inlier_mask,
        MaskChoice::F => &fr.inlier_mask,
        MaskChoice::H => &hr.inlier_mask,
    };
    let inliers = select_inliers(matches, m);
    // Minimum inlier ratio for the decision (off by default).
    if opts.min_inlier_ratio > 0.0 && (inliers.len() as f64 / matches.len() as f64) < opts.min_inlier_ratio {
        return degenerate();
    }
    // Design decision: discard models from failed RANSACs and record only the successful ones.
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
    assemble_uncalibrated(cam1, kps1, cam2, kps2, matches, opts, fr, hr)
}

/// 비보정 경로 마무리.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_uncalibrated(
    cam1: &Camera,
    kps1: &[Keypoint],
    cam2: &Camera,
    kps2: &[Keypoint],
    matches: &[FeatureMatch],
    opts: &TwoViewOptions,
    fr: RansacReport<Mat3>,
    hr: RansacReport<Mat3>,
) -> TwoViewGeometry {
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
    assemble_homography_only(cam1, kps1, cam2, kps2, matches, opts, hr)
}

/// H 전용 경로 마무리.
pub(crate) fn assemble_homography_only(
    cam1: &Camera,
    kps1: &[Keypoint],
    cam2: &Camera,
    kps2: &[Keypoint],
    matches: &[FeatureMatch],
    opts: &TwoViewOptions,
    hr: RansacReport<Mat3>,
) -> TwoViewGeometry {
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
        // Design decision: even when ignoring watermarks, remove their inliers and continue.
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

/// Watermark detection.
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
