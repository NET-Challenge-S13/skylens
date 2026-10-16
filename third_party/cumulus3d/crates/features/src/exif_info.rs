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

//! Reads only the needed values from EXIF (focal length, sensor info, orientation, GPS).

use exif::{In, Tag, Value};
use std::path::Path;

/// Image EXIF summary. Missing values are None.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExifInfo {
    /// Manufacturer (Make).
    pub make: Option<String>,
    /// Model (Model).
    pub model: Option<String>,
    /// Focal length (mm).
    pub focal_mm: Option<f64>,
    /// 35mm-equivalent focal length (mm).
    pub focal_35mm: Option<f64>,
    /// Focal-plane x resolution (FocalPlaneXResolution).
    pub focal_plane_x_resolution: Option<f64>,
    /// Focal-plane resolution unit (2 = inch, 3 = cm, 4 = mm ...).
    pub focal_plane_resolution_unit: Option<u32>,
    /// EXIF Orientation value (1~8).
    pub orientation: Option<u32>,
    /// (latitude, longitude, altitude) [deg, deg, m]. Only when all three are present.
    pub gps: Option<(f64, f64, f64)>,
}

impl ExifInfo {
    /// Reads from a file. Empty values if EXIF is missing or corrupt.
    pub fn read(path: &Path) -> Self {
        let Ok(file) = std::fs::File::open(path) else { return Self::default() };
        let mut r = std::io::BufReader::new(file);
        match exif::Reader::new().read_from_container(&mut r) {
            Ok(ex) => Self::from_exif(&ex),
            Err(_) => Self::default(),
        }
    }

    /// Reads from file bytes in memory.
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

/// Degrees/minutes/seconds -> degrees.
fn dms(v: &Value) -> Option<f64> {
    let d = value_f64(v, 0)?;
    let m = value_f64(v, 1).unwrap_or(0.0);
    let s = value_f64(v, 2).unwrap_or(0.0);
    Some(d + m / 60.0 + s / 3600.0)
}
