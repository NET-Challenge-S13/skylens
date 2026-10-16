/*
 * model_aligner.rs
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

//! GPS ENU alignment: fits the projection centers of registered images to the ENU coordinates of the GPS reference points
//! with a robust Sim3.
//!
//! On failure it returns `Err` and leaves the model untouched (it does not proceed silently with empty output).

use crate::error::{AlignError, Result};
use crate::geodesy::EnuFrame;
use crate::umeyama::{estimate_sim3_ransac, median, RankCheck};
use cumulus3d_core::io::{read_gps_file, GpsRecord};
use cumulus3d_core::ransac::RansacParams;
use cumulus3d_core::{ImageId, Reconstruction, Sim3, Vec3};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// ENU origin selection.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum EnuOrigin {
    /// Default behavior: the first line of the GPS list.
    #[default]
    FirstRecord,
    /// Origin fixed for the whole session (latitude°, longitude°, altitude m). Avoids the origin changing from zone to zone.
    Explicit {
        /// Latitude (°).
        lat: f64,
        /// Longitude (°).
        lon: f64,
        /// Ellipsoidal height (m).
        alt: f64,
    },
}

/// model_aligner options. Defaults = script invocation (`--alignment_max_error 3`, min_common_images 3).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelAlignerOptions {
    /// Inlier distance threshold (m) (distance between transformed camera center and GPS ENU ≤ 3 m).
    pub max_error: f64,
    /// Minimum number of common images needed for alignment.
    pub min_common_images: usize,
    /// ENU origin selection.
    pub origin: EnuOrigin,
    /// Confidence, iteration count and seed. The `max_error` field is ignored and the value above is used.
    pub ransac: RansacParams,
    /// Minimal-sample degeneracy test method.
    pub rank_check: RankCheck,
}

impl Default for ModelAlignerOptions {
    fn default() -> Self {
        Self {
            max_error: 3.0,
            min_common_images: 3,
            origin: EnuOrigin::FirstRecord,
            ransac: RansacParams::default(),
            rank_check: RankCheck::Uncentered,
        }
    }
}

/// Alignment result.
#[derive(Clone, Debug)]
pub struct GpsAlignment {
    /// model_to_enu.
    pub sim3: Sim3,
    /// ENU frame used (including the origin).
    pub enu: EnuFrame,
    /// Common images (file order): (image id, name, ENU position).
    pub common: Vec<(ImageId, String, Vec3)>,
    /// Inlier flags in the same order as `common`.
    pub inlier_mask: Vec<bool>,
    /// Number of inliers.
    pub num_inliers: usize,
    /// Number of RANSAC trials.
    pub num_trials: usize,
    /// ‖C' − g‖ after alignment (for each name in the file that is in the model and has a pose; no filtering).
    pub errors: Vec<(String, f64)>,
    /// Mean alignment error (m).
    pub mean_error: f64,
    /// Median alignment error (m).
    pub median_error: f64,
}

/// GPS list to (name, ENU). Error if the origin is `FirstRecord` but the list is empty.
pub fn gps_to_enu(records: &[GpsRecord], origin: EnuOrigin) -> Result<(EnuFrame, Vec<(String, Vec3)>)> {
    let frame = match origin {
        EnuOrigin::FirstRecord => {
            let r = records.first().ok_or(AlignError::TooFewReferences { found: 0, required: 1 })?;
            EnuFrame::new(r.lat, r.lon, r.alt)
        }
        EnuOrigin::Explicit { lat, lon, alt } => EnuFrame::new(lat, lon, alt),
    };
    let v = records.iter().map(|r| (r.name.clone(), frame.lla_to_enu(r.lat, r.lon, r.alt))).collect();
    Ok((frame, v))
}

/// Estimates only the alignment Sim3 (model unchanged).
pub fn estimate_gps_alignment(rec: &Reconstruction, gps: &[GpsRecord], opts: &ModelAlignerOptions) -> Result<GpsAlignment> {
    if opts.max_error.is_nan() || opts.max_error <= 0.0 {
        return Err(AlignError::BadMaxError(opts.max_error));
    }
    // 2. Reference positions < 3, so fail.
    if gps.len() < 3 {
        return Err(AlignError::TooFewReferences { found: gps.len(), required: 3 });
    }
    let (enu, refs) = gps_to_enu(gps, opts.origin)?;

    // 4. Common images (file order, names match, has pose, duplicates excluded).
    let by_name: HashMap<&str, ImageId> = rec.images().map(|im| (im.name.as_str(), im.image_id)).collect();
    let mut seen = HashSet::new();
    let mut common = Vec::new();
    let (mut src, mut dst) = (Vec::new(), Vec::new());
    for (name, g) in &refs {
        let Some(&id) = by_name.get(name.as_str()) else { continue };
        let Some(c) = rec.projection_center(id) else { continue };
        if !seen.insert(id) {
            continue;
        }
        common.push((id, name.clone(), *g));
        src.push(c);
        dst.push(*g);
    }
    // 5.
    let required = opts.min_common_images.max(3);
    if common.len() < required {
        return Err(AlignError::TooFewCommonImages { found: common.len(), required });
    }
    // 6.
    let ropts = RansacParams { max_error: opts.max_error, ..opts.ransac.clone() };
    let rep = estimate_sim3_ransac(&src, &dst, &ropts, opts.rank_check);
    let num_inliers = rep.support.num_inliers;
    let sim3 = match (rep.success, rep.model) {
        (true, Some(m)) if num_inliers >= 3 => m,
        _ => return Err(AlignError::RansacFailed { inliers: num_inliers }),
    };

    // 8. Alignment error (all names in the file, no filter; duplicate lines are each counted).
    let mut errors = Vec::new();
    for (name, g) in &refs {
        let Some(&id) = by_name.get(name.as_str()) else { continue };
        let Some(c) = rec.projection_center(id) else { continue };
        errors.push((name.clone(), (sim3.transform_point(&c) - g).norm()));
    }
    let mut ev: Vec<f64> = errors.iter().map(|e| e.1).collect();
    let mean_error = if ev.is_empty() { f64::NAN } else { ev.iter().sum::<f64>() / ev.len() as f64 };
    // Design decision: median for an even count. Use the mean of the middle two.
    let median_error = median(&mut ev);

    Ok(GpsAlignment {
        sim3,
        enu,
        common,
        inlier_mask: rep.inlier_mask,
        num_inliers,
        num_trials: rep.num_trials,
        errors,
        mean_error,
        median_error,
    })
}

/// Applies to the model (poses and points) if estimation succeeds. On failure leaves the model unchanged and returns `Err`.
pub fn align_to_gps(rec: &mut Reconstruction, gps: &[GpsRecord], opts: &ModelAlignerOptions) -> Result<GpsAlignment> {
    let a = estimate_gps_alignment(rec, gps, opts)?;
    rec.transform(&a.sim3);
    Ok(a)
}

/// GPS file path version.
pub fn align_to_gps_file(rec: &mut Reconstruction, path: impl AsRef<Path>, opts: &ModelAlignerOptions) -> Result<GpsAlignment> {
    let gps = read_gps_file(path)?;
    align_to_gps(rec, &gps, opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geodesy::EnuFrame;
    use crate::umeyama::tests::{gauss, rand_vec, random_sim3};
    use cumulus3d_core::{Camera, CameraModelKind, Image, Quat, Rigid3, TrackEntry};
    use rand::RngExt;

    /// Builds an ENU trajectory (3 drones × positions) and moves it to "model coordinates" with an arbitrary Sim3:
    /// the reconstruction + GPS list.
    fn scene(num_pos: usize, seed: u64) -> (Reconstruction, Vec<GpsRecord>, Vec<Vec3>, Sim3) {
        let mut rng = cumulus3d_core::ransac::make_rng(Some(seed));
        let enu = EnuFrame::new(37.5, 127.0, 50.0);
        let enu_to_model = random_sim3(&mut rng);
        let mut rec = Reconstruction::new();
        let cam = Camera::new(1, CameraModelKind::Pinhole, 1000, 800, vec![800.0, 800.0, 500.0, 400.0]).unwrap();
        rec.add_camera_own_rig(cam).unwrap();
        let mut gps = Vec::new();
        let mut centers = Vec::new();
        let mut id = 1;
        for p in 0..num_pos {
            for (k, dx) in [-20.0, 0.0, 20.0].iter().enumerate() {
                let c_enu = Vec3::new(p as f64 * 8.0 + dx, p as f64 * 2.0 + dx * 0.5, 30.0 + 3.0 * k as f64);
                let c_model = enu_to_model.transform_point(&c_enu);
                // Camera looking down (−U) + a small random rotation.
                let down = Quat::from_axis_angle(&Vec3::x(), std::f64::consts::PI);
                let small = Quat::from_axis_angle(&rand_vec(&mut rng, 1.0).normalize(), rng.random_range(-0.2..0.2));
                // The model-frame rotation must also be accounted for to look down: R_cam = small·down·R_mᵀ.
                let rot = small * down * enu_to_model.rotation.conjugate();
                let r = rot.to_rotation_matrix();
                let pose = Rigid3::new(rot, -(r * c_model));
                let pts: Vec<_> = (0..20).map(|i| cumulus3d_core::Vec2::new(10.0 + i as f64, 20.0)).collect();
                let name = format!("cam{k}/img_{p:04}.jpg");
                rec.add_image_own_frame(Image::new(id, &name, 1, pts), Some(pose)).unwrap();
                rec.register_image(id).unwrap();
                let (lat, lon, alt) = enu.enu_to_lla(&c_enu);
                gps.push(GpsRecord { name, lat, lon, alt });
                centers.push(c_enu);
                id += 1;
            }
        }
        // A few observations (for checking reprojection invariance).
        for i in 0..10u32 {
            let x = enu_to_model.transform_point(&Vec3::new(10.0 + i as f64, 5.0, 0.0));
            let pid = rec.add_point3d(x, vec![], [0, 0, 0]).unwrap();
            for img in 1..=6u32 {
                let pose = rec.world_to_cam(img).unwrap();
                let cam = rec.camera(1).unwrap();
                if let Some(_xy) = cam.cam_to_img(&pose.transform_point(&x)) {
                    rec.add_observation(pid, TrackEntry::new(img, i)).unwrap();
                }
            }
        }
        (rec, gps, centers, enu_to_model)
    }

    /// Projected pixels of all observations.
    fn projections(rec: &Reconstruction) -> Vec<cumulus3d_core::Vec2> {
        let mut out = Vec::new();
        for id in rec.point3d_ids() {
            let p = rec.point3d(id).unwrap();
            for e in &p.track {
                let pose = rec.world_to_cam(e.image_id).unwrap();
                let cam = rec.camera(1).unwrap();
                out.push(cam.cam_to_img(&pose.transform_point(&p.xyz)).unwrap());
            }
        }
        out
    }

    #[test]
    fn aligns_synthetic_model() {
        let (mut rec, gps, centers, enu_to_model) = scene(10, 11);
        let before = projections(&rec);
        assert!(before.len() > 10);
        let opts = ModelAlignerOptions { ransac: RansacParams { random_seed: Some(1), ..Default::default() }, ..Default::default() };
        let a = align_to_gps(&mut rec, &gps, &opts).unwrap();
        assert_eq!(a.num_inliers, 30);
        // The origin = first line position, so the first image center is ENU (0,0,0) and the rest are shifted from the original ENU by c0.
        let c0 = centers[0];
        for (i, c) in centers.iter().enumerate() {
            let got = rec.projection_center(i as u32 + 1).unwrap();
            assert!((got - (c - c0)).norm() < 1e-3, "{i}: {got:?} vs {:?}", c - c0);
        }
        assert!(a.median_error < 1e-3 && a.mean_error < 1e-3);
        let expect = enu_to_model.inverse();
        assert!((a.sim3.scale - expect.scale).abs() / expect.scale < 1e-6);
        // Reprojection error is invariant after applying the Sim3.
        let after = projections(&rec);
        assert_eq!(before.len(), after.len());
        for (b, a) in before.iter().zip(&after) {
            assert!((b - a).norm() < 1e-8, "{b:?} vs {a:?}");
        }
        // C' = s R C + t.
        let c_before = enu_to_model.transform_point(&centers[4]);
        assert!((a.sim3.transform_point(&c_before) - rec.projection_center(5).unwrap()).norm() < 1e-9);
    }

    #[test]
    fn explicit_origin_and_order_rule() {
        let (rec, mut gps, centers, _) = scene(6, 12);
        let opts = ModelAlignerOptions { ransac: RansacParams { random_seed: Some(2), ..Default::default() }, ..Default::default() };
        // Changing the line order changes the origin.
        gps.swap(0, 5);
        let mut r1 = rec.clone();
        align_to_gps(&mut r1, &gps, &opts).unwrap();
        assert!(r1.projection_center(6).unwrap().norm() < 1e-3);
        // An explicit origin is independent of the line order.
        let o = &gps[3];
        let opts2 = ModelAlignerOptions { origin: EnuOrigin::Explicit { lat: o.lat, lon: o.lon, alt: o.alt }, ..opts.clone() };
        let mut r2 = rec.clone();
        align_to_gps(&mut r2, &gps, &opts2).unwrap();
        let id_of_o = r2.images().find(|im| im.name == o.name).unwrap().image_id;
        assert!(r2.projection_center(id_of_o).unwrap().norm() < 1e-3);
        let _ = centers;
    }

    #[test]
    fn robust_to_gps_outliers() {
        // Of 50 cameras, 30% are contaminated by 20 m or more, the rest get σ=1 m noise, threshold 3 m.
        let (rec, mut gps, centers, _) = scene(17, 13);
        let mut rng = cumulus3d_core::ransac::make_rng(Some(5));
        let enu = EnuFrame::new(gps[0].lat, gps[0].lon, gps[0].alt);
        let c0 = centers[0];
        let n = 50;
        let gps_trunc: Vec<GpsRecord> = gps.drain(..n).collect();
        let mut out = vec![false; n];
        let mut noisy = Vec::new();
        for (i, g) in gps_trunc.iter().enumerate() {
            let truth = centers[i] - c0;
            let e = if i % 10 < 3 && i != 0 {
                out[i] = true;
                truth + rand_vec(&mut rng, 1.0).normalize() * rng.random_range(20.0..60.0)
            } else if i == 0 {
                truth
            } else {
                truth + Vec3::new(gauss(&mut rng), gauss(&mut rng), gauss(&mut rng))
            };
            let (lat, lon, alt) = enu.enu_to_lla(&e);
            noisy.push(GpsRecord { name: g.name.clone(), lat, lon, alt });
        }
        let opts = ModelAlignerOptions { ransac: RansacParams { random_seed: Some(3), ..Default::default() }, ..Default::default() };
        let mut r = rec.clone();
        let a = align_to_gps(&mut r, &noisy, &opts).unwrap();
        for (i, (id, _, _)) in a.common.iter().enumerate() {
            let k = (*id - 1) as usize;
            if out[k] {
                assert!(!a.inlier_mask[i], "오염 {k} 이 인라이어");
            }
        }
        let mut errs: Vec<f64> =
            (0..n).filter(|k| !out[*k]).map(|k| (r.projection_center(k as u32 + 1).unwrap() - (centers[k] - c0)).norm()).collect();
        let med = median(&mut errs);
        assert!(med < 1.5, "median {med}");
    }

    #[test]
    fn failure_cases_return_err_and_leave_model() {
        let (mut rec, gps, _, _) = scene(4, 14);
        let orig = rec.projection_center(1).unwrap();
        // 2 common images.
        let mut few = gps[..2].to_vec();
        few.push(GpsRecord { name: "nope.jpg".into(), ..gps[2].clone() });
        assert!(matches!(align_to_gps(&mut rec, &few, &Default::default()), Err(AlignError::TooFewCommonImages { found: 2, .. })));
        assert!(matches!(align_to_gps(&mut rec, &gps[..2], &Default::default()), Err(AlignError::TooFewReferences { .. })));
        let bad = ModelAlignerOptions { max_error: 0.0, ..Default::default() };
        assert!(matches!(align_to_gps(&mut rec, &gps, &bad), Err(AlignError::BadMaxError(_))));
        assert_eq!(rec.projection_center(1).unwrap(), orig);
    }
}
