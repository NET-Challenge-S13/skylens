/*
 * intrinsics.rs
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

//! Soft intrinsics prior (`cumulus3d stream --intrinsics-prior`): known camera parameters (from an earlier calibration or
//! run) that start every camera at the right values and pull the bundle adjustment back when the data alone are weak.
//!
//! File format (JSON): `{"cameras": {"<camera_id>": [p0, p1, ...]}}` with the parameters in the camera model's order
//! (OPENCV: `fx fy cx cy k1 k2 p1 p2`). The vector length must equal the model's parameter count. Other keys (e.g. `note`)
//! are ignored. The principal point is not taken from the file (it stays what the session has: the image center) and is
//! not pulled; focal lengths get a relative σ, every other parameter an absolute one.

use crate::util::Json;
use cumulus3d_core::{Camera, CameraId};
use std::collections::BTreeMap;
use std::path::Path;

/// Default relative σ of the focal lengths (residual `(f − f0)/(σ·f0)`).
pub const DEFAULT_FOCAL_SIGMA: f64 = 0.005;
/// Default absolute σ of the distortion (extra) parameters.
pub const DEFAULT_DIST_SIGMA: f64 = 0.01;

/// Known intrinsics per camera id.
#[derive(Clone, Debug, PartialEq)]
pub struct IntrinsicsPrior {
    /// Camera id → parameter vector in the camera model's order.
    pub cameras: BTreeMap<CameraId, Vec<f64>>,
    /// Relative σ of the focal lengths.
    pub focal_sigma: f64,
    /// Absolute σ of the distortion parameters.
    pub dist_sigma: f64,
}

impl IntrinsicsPrior {
    /// Reads the JSON file; σ values as given.
    pub fn load(path: impl AsRef<Path>, focal_sigma: f64, dist_sigma: f64) -> Result<Self, String> {
        let p = path.as_ref();
        let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
        Self::from_json(&text, focal_sigma, dist_sigma).map_err(|e| format!("{}: {e}", p.display()))
    }

    /// Parses the JSON text.
    pub fn from_json(text: &str, focal_sigma: f64, dist_sigma: f64) -> Result<Self, String> {
        if focal_sigma <= 0.0 || !focal_sigma.is_finite() || dist_sigma <= 0.0 || !dist_sigma.is_finite() {
            return Err(format!("내부값 사전 σ 는 0 보다 커야 함 (초점 {focal_sigma}, 왜곡 {dist_sigma})"));
        }
        let j = Json::parse(text)?;
        let Some(Json::Obj(cams)) = j.get("cameras") else { return Err("\"cameras\" 객체가 없음".into()) };
        let mut cameras = BTreeMap::new();
        for (k, v) in cams {
            let id: CameraId = k.trim().parse().map_err(|_| format!("카메라 id {k:?} 는 정수가 아님"))?;
            let Json::Arr(a) = v else { return Err(format!("카메라 {k}: 파라미터 배열이 아님")) };
            let params: Option<Vec<f64>> = a.iter().map(Json::as_f64).collect();
            let params = params.filter(|p| p.iter().all(|x| x.is_finite())).ok_or_else(|| format!("카메라 {k}: 숫자가 아닌 값"))?;
            cameras.insert(id, params);
        }
        if cameras.is_empty() {
            return Err("\"cameras\" 가 비어 있음".into());
        }
        Ok(Self { cameras, focal_sigma, dist_sigma })
    }

    /// The prior parameter vector for `cam` (None when the file has no entry for it); an error when its length differs from
    /// the model's parameter count.
    pub fn params_for(&self, cam: &Camera) -> Result<Option<&[f64]>, String> {
        let Some(p) = self.cameras.get(&cam.camera_id) else { return Ok(None) };
        let n = cam.model.num_params();
        if p.len() != n {
            return Err(format!(
                "카메라 {}: 사전 파라미터 {}개, 모델 {} 은 {}개 필요 ({})",
                cam.camera_id,
                p.len(),
                cam.model.name(),
                n,
                cam.model.params_info()
            ));
        }
        Ok(Some(p))
    }

    /// Sets `cam`'s parameters to the prior (except the principal point). True when the camera has an entry.
    pub fn apply(&self, cam: &mut Camera) -> Result<bool, String> {
        let Some(p) = self.params_for(cam)? else { return Ok(false) };
        let pp = cam.model.pp_slots();
        for (i, v) in p.iter().enumerate() {
            if !pp.contains(&i) {
                cam.params[i] = *v;
            }
        }
        Ok(true)
    }

    /// Target and per-parameter σ for the bundle adjustment (`BaConfig::intrinsics_priors`): focal slots get
    /// `focal_sigma · f0`, the principal point σ 0 (not pulled), the rest `dist_sigma`.
    pub fn ba_prior(&self, cam: &Camera) -> Option<(Vec<f64>, Vec<f64>)> {
        let p = self.params_for(cam).ok()??;
        let mut sigma = vec![self.dist_sigma; p.len()];
        for &s in cam.model.focal_slots() {
            sigma[s] = self.focal_sigma * p[s].abs();
        }
        for &s in cam.model.pp_slots() {
            sigma[s] = 0.0;
        }
        Some((p.to_vec(), sigma))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cumulus3d_core::CameraModelKind;

    const SAMPLE: &str = r#"{"note": "x", "cameras": {
        "1": [1607.39, 1611.24, 1024.0, 576.0, 0.082, -0.084, 0.0013, 0.0014],
        "3": [1601.41, 1599.46, 1024.0, 576.0, 0.077, -0.079, 0.0006, -0.0014]}}"#;

    fn cam(id: u32, model: CameraModelKind, params: Vec<f64>) -> Camera {
        Camera::new(id, model, 2048, 1152, params).unwrap()
    }

    #[test]
    fn loads_and_applies_without_touching_the_principal_point() {
        let pr = IntrinsicsPrior::from_json(SAMPLE, 0.005, 0.01).unwrap();
        assert_eq!(pr.cameras.len(), 2);
        let mut c = cam(1, CameraModelKind::OpenCv, vec![1900.0, 1900.0, 1000.0, 580.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(pr.apply(&mut c).unwrap());
        assert_eq!(c.params[..2], [1607.39, 1611.24]);
        assert_eq!(c.params[2..4], [1000.0, 580.0]);
        assert_eq!(c.params[4], 0.082);
        let mut other = cam(2, CameraModelKind::OpenCv, vec![1900.0; 8]);
        assert!(!pr.apply(&mut other).unwrap());
        assert_eq!(other.params, vec![1900.0; 8]);
        let (t, s) = pr.ba_prior(&c).unwrap();
        assert_eq!(t.len(), 8);
        assert!((s[0] - 0.005 * 1607.39).abs() < 1e-9 && s[2] == 0.0 && s[3] == 0.0 && s[7] == 0.01);
    }

    #[test]
    fn parameter_count_mismatch_is_an_error() {
        let pr = IntrinsicsPrior::from_json(SAMPLE, 0.005, 0.01).unwrap();
        let mut c = cam(1, CameraModelKind::SingleFocalRadial, vec![1900.0, 1024.0, 576.0, 0.0]);
        let e = pr.apply(&mut c).unwrap_err();
        assert!(e.contains("카메라 1") && e.contains("8개"), "{e}");
        // A file whose vector matches a simpler model is accepted for that model.
        let pr = IntrinsicsPrior::from_json(r#"{"cameras": {"1": [1600, 1024, 576, 0.05]}}"#, 0.005, 0.01).unwrap();
        assert!(pr.apply(&mut c).unwrap());
        assert_eq!(c.params, vec![1600.0, 1024.0, 576.0, 0.05]);
    }

    #[test]
    fn rejects_bad_files() {
        assert!(IntrinsicsPrior::from_json("{}", 0.005, 0.01).is_err());
        assert!(IntrinsicsPrior::from_json(r#"{"cameras": {"a": [1]}}"#, 0.005, 0.01).is_err());
        assert!(IntrinsicsPrior::from_json(r#"{"cameras": {"1": [1, "x"]}}"#, 0.005, 0.01).is_err());
        assert!(IntrinsicsPrior::from_json(SAMPLE, 0.0, 0.01).is_err());
    }
}
