/*
 * camera_init.rs
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

//! 카메라 초기 파라미터.

use crate::exif_info::ExifInfo;
use cumulus3d_core::{Camera, CameraModelKind, Error, Result};

/// 초점거리 결정 결과.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FocalEstimate {
    /// 초점거리(화소).
    pub focal: f64,
    /// EXIF 에서 정해졌으면 참.
    pub prior: bool,
}

/// 내장 센서 폭 표 (제조사, 모델, 센서 폭 mm).
// 설계 결정: 공개 제품 사양으로 알려진 드론 카메라 일부만 넣는다.
const SENSOR_WIDTHS: &[(&str, &str, f64)] = &[
    ("dji", "fc6310", 13.2),
    ("dji", "fc6310s", 13.2),
    ("dji", "fc6510", 13.2),
    ("dji", "fc6520", 17.3),
    ("dji", "fc6540", 23.5),
    ("dji", "fc330", 6.17),
    ("dji", "fc300x", 6.17),
    ("dji", "fc300s", 6.17),
    ("dji", "fc220", 6.17),
    ("dji", "fc2103", 6.17),
    ("dji", "fc2204", 6.17),
    ("dji", "fc3170", 6.4),
    ("dji", "fc3411", 6.4),
    ("dji", "fc3582", 9.6),
    ("dji", "l1d-20c", 13.2),
    ("dji", "l2d-20c", 17.3),
    ("dji", "fc7303", 6.3),
    ("dji", "zenmusep1", 35.9),
    ("dji", "m3e", 17.3),
];

/// 제조사·모델로 센서 폭 조회(대소문자·공백 무시, 모델 앞에 제조사가 붙은 경우 허용).
pub fn lookup_sensor_width(make: &str, model: &str) -> Option<f64> {
    let norm = |s: &str| s.chars().filter(|c| !c.is_whitespace() && *c != '_' && *c != '-').collect::<String>().to_lowercase();
    let mk = norm(make);
    let mut md = norm(model);
    if md.starts_with(&mk) {
        md = md[mk.len()..].to_string();
    }
    SENSOR_WIDTHS.iter().find(|(m, n, _)| mk.starts_with(&norm(m)) && norm(n) == md).map(|&(_, _, w)| w)
}

/// 초점거리 규칙. (W, H) 는 입력 영상 크기.
pub fn infer_focal(exif: &ExifInfo, width: u64, height: u64, default_factor: f64) -> FocalEstimate {
    let (w, h) = (width as f64, height as f64);
    let max_size = w.max(h);
    if let Some(f35) = exif.focal_35mm.filter(|&f| f > 0.0) {
        return FocalEstimate { focal: f35 / 43.27 * (w * w + h * h).sqrt(), prior: true };
    }
    if let Some(fmm) = exif.focal_mm.filter(|&f| f > 0.0) {
        if let (Some(r), Some(u)) = (exif.focal_plane_x_resolution, exif.focal_plane_resolution_unit) {
            let px_per_mm = match u {
                2 => Some(r / 25.4),
                3 => Some(r / 10.0),
                4 => Some(r),
                5 => Some(r * 1000.0),
                _ => None,
            };
            if let Some(p) = px_per_mm.filter(|p| *p > 0.0) {
                return FocalEstimate { focal: fmm * p, prior: true };
            }
        }
        if let (Some(mk), Some(md)) = (&exif.make, &exif.model) {
            if let Some(sw) = lookup_sensor_width(mk, md) {
                return FocalEstimate { focal: fmm / sw * max_size, prior: true };
            }
        }
    }
    FocalEstimate { focal: default_factor * max_size, prior: false }
}

/// 새 카메라 초기화: `params` 가 주어지면 그대로(플래그 참), 아니면 EXIF/기본값 초점 + 주점 (W/2, H/2), 왜곡 0.
pub fn init_camera(
    model: CameraModelKind,
    width: u64,
    height: u64,
    exif: &ExifInfo,
    params: Option<&[f64]>,
    default_focal_length_factor: f64,
) -> Result<Camera> {
    if let Some(p) = params {
        let mut cam = Camera::new(cumulus3d_core::INVALID_CAMERA_ID, model, width, height, p.to_vec())?;
        cam.focal_from_prior = true;
        return Ok(cam);
    }
    if width == 0 || height == 0 {
        return Err(Error::InvalidArgument("영상 크기 0".into()));
    }
    let fe = infer_focal(exif, width, height, default_focal_length_factor);
    let mut cam = Camera::from_focal(model, fe.focal, width, height);
    cam.focal_from_prior = fe.prior;
    Ok(cam)
}
