/*
 * exif_info.rs
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

//! EXIF 에서 필요한 값만 읽는다(초점거리, 센서 정보, 방향, GPS).

use exif::{In, Tag, Value};
use std::path::Path;

/// 영상 EXIF 요약. 없는 값은 None.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExifInfo {
    /// 제조사(Make).
    pub make: Option<String>,
    /// 모델(Model).
    pub model: Option<String>,
    /// 초점거리(mm).
    pub focal_mm: Option<f64>,
    /// 35mm 환산 초점거리(mm).
    pub focal_35mm: Option<f64>,
    /// 초점면 x 해상도(FocalPlaneXResolution).
    pub focal_plane_x_resolution: Option<f64>,
    /// 초점면 해상도 단위(2 = 인치, 3 = cm, 4 = mm …).
    pub focal_plane_resolution_unit: Option<u32>,
    /// EXIF Orientation 값(1~8).
    pub orientation: Option<u32>,
    /// (위도, 경도, 고도) [도, 도, m]. 셋 다 있을 때만.
    pub gps: Option<(f64, f64, f64)>,
}

impl ExifInfo {
    /// 파일에서 읽는다. EXIF 가 없거나 깨졌으면 빈 값.
    pub fn read(path: &Path) -> Self {
        let Ok(file) = std::fs::File::open(path) else { return Self::default() };
        let mut r = std::io::BufReader::new(file);
        match exif::Reader::new().read_from_container(&mut r) {
            Ok(ex) => Self::from_exif(&ex),
            Err(_) => Self::default(),
        }
    }

    /// 메모리의 파일 바이트에서 읽는다.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut c = std::io::Cursor::new(bytes);
        match exif::Reader::new().read_from_container(&mut c) {
            Ok(ex) => Self::from_exif(&ex),
            Err(_) => Self::default(),
        }
    }

    fn from_exif(ex: &exif::Exif) -> Self {
        let field = |t: Tag| ex.get_field(t, In::PRIMARY).map(|f| &f.value);
        let ascii = |t: Tag| match field(t) {
            Some(Value::Ascii(v)) => {
                v.first().map(|s| String::from_utf8_lossy(s).trim().trim_end_matches('\0').to_string()).filter(|s| !s.is_empty())
            }
            _ => None,
        };
        let real = |t: Tag| field(t).and_then(|v| value_f64(v, 0));
        let uint = |t: Tag| field(t).and_then(|v| v.get_uint(0));

        let gps = (|| {
            let lat = dms(field(Tag::GPSLatitude)?)?;
            let lon = dms(field(Tag::GPSLongitude)?)?;
            let alt = real(Tag::GPSAltitude)?;
            let lat = if ascii(Tag::GPSLatitudeRef).is_some_and(|s| s.starts_with('S')) { -lat } else { lat };
            let lon = if ascii(Tag::GPSLongitudeRef).is_some_and(|s| s.starts_with('W')) { -lon } else { lon };
            let alt = if uint(Tag::GPSAltitudeRef) == Some(1) { -alt } else { alt };
            Some((lat, lon, alt))
        })();

        Self {
            make: ascii(Tag::Make),
            model: ascii(Tag::Model),
            focal_mm: real(Tag::FocalLength),
            focal_35mm: real(Tag::FocalLengthIn35mmFilm),
            focal_plane_x_resolution: real(Tag::FocalPlaneXResolution),
            focal_plane_resolution_unit: uint(Tag::FocalPlaneResolutionUnit),
            orientation: uint(Tag::Orientation),
            gps,
        }
    }
}

fn value_f64(v: &Value, i: usize) -> Option<f64> {
    match v {
        Value::Rational(r) => r.get(i).filter(|x| x.denom != 0).map(|x| x.to_f64()),
        Value::SRational(r) => r.get(i).filter(|x| x.denom != 0).map(|x| x.to_f64()),
        Value::Float(f) => f.get(i).map(|&x| x as f64),
        Value::Double(f) => f.get(i).copied(),
        _ => v.get_uint(i).map(|x| x as f64),
    }
}

/// 도·분·초 → 도.
fn dms(v: &Value) -> Option<f64> {
    let d = value_f64(v, 0)?;
    let m = value_f64(v, 1).unwrap_or(0.0);
    let s = value_f64(v, 2).unwrap_or(0.0);
    Some(d + m / 60.0 + s / 3600.0)
}
