/*
 * gray.rs
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

//! 8비트 회색 영상, 디코딩·회색 변환·축소·90° 회전.

use cumulus3d_core::{Error, Keypoint, Result};
use std::path::Path;

/// 행 우선 8비트 회색 영상.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrayImage {
    /// 너비(화소).
    pub width: usize,
    /// 높이(화소).
    pub height: usize,
    /// 행 우선 화소 값(길이 width × height).
    pub data: Vec<u8>,
}

impl GrayImage {
    /// 버퍼로 만든다. 길이가 width × height 가 아니면 오류.
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Result<Self> {
        if data.len() != width * height {
            return Err(Error::InvalidArgument(format!("회색 영상 크기 {width}x{height} 와 자료 길이 {} 불일치", data.len())));
        }
        Ok(Self { width, height, data })
    }

    /// 모든 화소가 `v` 인 영상.
    pub fn filled(width: usize, height: usize, v: u8) -> Self {
        Self { width, height, data: vec![v; width * height] }
    }

    /// `[0,1]` 실수 영상에서 만든다(반올림·포화).
    pub fn from_f32(width: usize, height: usize, f: impl Fn(usize, usize) -> f32) -> Self {
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                data.push((f(x, y) * 255.0).round().clamp(0.0, 255.0) as u8);
            }
        }
        Self { width, height, data }
    }

    #[inline]
    /// 화소 (x, y) 값.
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.width + x]
    }

    /// 반시계 90°×k 회전(키포인트 회전 변환과 일치: (x,y) → (y, w−x)).
    pub fn rotate_ccw(&self, k: u32) -> GrayImage {
        let mut img = self.clone();
        for _ in 0..(k % 4) {
            img = img.rotate_ccw_once();
        }
        img
    }

    fn rotate_ccw_once(&self) -> GrayImage {
        let (w, h) = (self.width, self.height);
        // 새 영상 (w'=h, h'=w): new(X, Y) = old(col = w−1−Y, row = X).
        let mut data = vec![0u8; w * h];
        for ny in 0..w {
            let oc = w - 1 - ny;
            let row = &mut data[ny * h..(ny + 1) * h];
            for (nx, v) in row.iter_mut().enumerate() {
                *v = self.data[nx * w + oc];
            }
        }
        GrayImage { width: h, height: w, data }
    }

    /// 이 영상을 `(width, height)` 로 재표본화(Lanczos3, 안티에일리어싱 포함).
    // 설계 결정: 축소 필터는 Lanczos3.
    pub fn resized(&self, width: usize, height: usize) -> GrayImage {
        if width == self.width && height == self.height {
            return self.clone();
        }
        let buf = image::GrayImage::from_raw(self.width as u32, self.height as u32, self.data.clone()).expect("크기 불변식");
        let out = image::imageops::resize(&buf, width as u32, height as u32, image::imageops::FilterType::Lanczos3);
        GrayImage { width, height, data: out.into_raw() }
    }
}

/// RGB → 회색: `round(0.2126 R + 0.7152 G + 0.0722 B)` (감마 선형화 없음).
#[inline]
pub fn rgb_to_gray(r: u8, g: u8, b: u8) -> u8 {
    (0.2126f32 * r as f32 + 0.7152f32 * g as f32 + 0.0722f32 * b as f32).round().clamp(0.0, 255.0) as u8
}

/// 디코딩된 영상을 회색으로(알파 버림, 16비트는 8비트로).
pub fn to_gray(img: &image::DynamicImage) -> Result<GrayImage> {
    use image::DynamicImage as D;
    let (w, h) = (img.width() as usize, img.height() as usize);
    let data = match img {
        D::ImageLuma8(b) => b.as_raw().clone(),
        D::ImageLumaA8(_) | D::ImageLuma16(_) | D::ImageLumaA16(_) => img.to_luma8().into_raw(),
        D::ImageRgb8(b) => b.as_raw().chunks(3).map(|p| rgb_to_gray(p[0], p[1], p[2])).collect(),
        _ => {
            let c = img.color();
            if c.channel_count() == 3 || c.channel_count() == 4 {
                img.to_rgb8().as_raw().chunks(3).map(|p| rgb_to_gray(p[0], p[1], p[2])).collect()
            } else {
                return Err(Error::Unsupported(format!("채널 구성 {c:?}")));
            }
        }
    };
    GrayImage::new(w, h, data)
}

/// 파일을 디코딩해 회색 영상으로 읽는다(EXIF 방향 자동 회전 없음).
pub fn read_gray(path: &Path) -> Result<GrayImage> {
    let reader = image::ImageReader::open(path)?.with_guessed_format().map_err(Error::Io)?;
    let img = reader.decode().map_err(|e| Error::Format(format!("{}: 디코딩 실패: {e}", path.display())))?;
    to_gray(&img)
}

/// 최대 크기 제한: 둘 다 M 이하면 그대로, 아니면 s = M/max(w,h), (round(w s), round(h s)).
pub fn limited_size(width: usize, height: usize, max_size: usize) -> (usize, usize) {
    if width <= max_size && height <= max_size {
        return (width, height);
    }
    let s = max_size as f64 / width.max(height) as f64;
    (((width as f64 * s).round() as usize).max(1), ((height as f64 * s).round() as usize).max(1))
}

/// 키포인트를 크기 (w, h) 영상에서 반시계 90°×k 회전. k 회마다 크기가 바뀌므로 한 번씩 적용.
pub fn rotate_keypoint_ccw(kp: &Keypoint, k: u32, width: f32, height: f32) -> Keypoint {
    let mut p = *kp;
    let (mut w, mut h) = (width, height);
    for _ in 0..(k % 4) {
        p = Keypoint { x: p.y, y: w - p.x, a11: p.a21, a12: p.a22, a21: -p.a11, a22: -p.a12 };
        std::mem::swap(&mut w, &mut h);
    }
    p
}

/// EXIF Orientation → 반시계 회전 횟수 k. 1→0, 3→2, 6→3, 8→1; 그 외(거울 포함·없음) 0.
pub fn orientation_to_rotation(orientation: Option<u32>) -> u32 {
    let g = orientation_gravity(orientation);
    match g {
        Some((gx, gy)) => {
            let k = ((gy.atan2(gx) - std::f64::consts::FRAC_PI_2) / std::f64::consts::FRAC_PI_2).round() as i64;
            k.rem_euclid(4) as u32
        }
        None => 0,
    }
}

/// EXIF Orientation → 영상 좌표 중력 방향 (g_x, g_y). 1:(0,1) 3:(0,−1) 6:(1,0) 8:(−1,0).
pub fn orientation_gravity(orientation: Option<u32>) -> Option<(f64, f64)> {
    match orientation? {
        1 => Some((0.0, 1.0)),
        3 => Some((0.0, -1.0)),
        6 => Some((1.0, 0.0)),
        8 => Some((-1.0, 0.0)),
        _ => None,
    }
}
