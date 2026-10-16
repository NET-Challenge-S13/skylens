/*
 * image.rs
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

//! 영상 버퍼와 재표본 도구(u8 컬러, f32 회색, 적분영상 면적 평균, 쌍선형·3차 보간).

use cumulus3d_core::{Error, Result};
use std::path::Path;

/// 8비트 영상(채널 1 또는 3, 행 우선, 채널 섞어 저장).
#[derive(Clone, Debug, PartialEq)]
pub struct ImageBuffer {
    /// 너비(픽셀).
    pub width: usize,
    /// 높이(픽셀).
    pub height: usize,
    /// 채널 수(1 또는 3).
    pub channels: usize,
    /// 픽셀 값(행 우선, 채널 섞어 저장).
    pub data: Vec<u8>,
}

impl ImageBuffer {
    /// 0 으로 채운 영상.
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Self { width, height, channels, data: vec![0; width * height * channels] }
    }
    /// 파일에서 읽기(회색이면 1채널, 그 밖은 RGB 3채널).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let img = image::open(path.as_ref()).map_err(|e| Error::Format(format!("{}: {e}", path.as_ref().display())))?;
        Ok(match img {
            image::DynamicImage::ImageLuma8(g) => {
                let (w, h) = g.dimensions();
                Self { width: w as usize, height: h as usize, channels: 1, data: g.into_raw() }
            }
            other => {
                let rgb = other.to_rgb8();
                let (w, h) = rgb.dimensions();
                Self { width: w as usize, height: h as usize, channels: 3, data: rgb.into_raw() }
            }
        })
    }
    /// 확장자로 형식을 정해 저장. JPEG 는 `jpeg_quality`(1~100) 로 쓴다.
    pub fn save(&self, path: impl AsRef<Path>, jpeg_quality: u8) -> Result<()> {
        let path = path.as_ref();
        let ct = if self.channels == 1 { image::ExtendedColorType::L8 } else { image::ExtendedColorType::Rgb8 };
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        let map = |e: image::ImageError| Error::Format(format!("{}: {e}", path.display()));
        if ext == "jpg" || ext == "jpeg" {
            let f = std::io::BufWriter::new(std::fs::File::create(path)?);
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(f, jpeg_quality.clamp(1, 100));
            enc.encode(&self.data, self.width as u32, self.height as u32, ct).map_err(map)
        } else {
            image::save_buffer(path, &self.data, self.width as u32, self.height as u32, ct).map_err(map)
        }
    }
    /// 픽셀 값(채널 c).
    #[inline]
    pub fn get(&self, x: usize, y: usize, c: usize) -> u8 {
        self.data[(y * self.width + x) * self.channels + c]
    }
    /// RGB 값(회색이면 복제).
    #[inline]
    pub fn rgb(&self, x: usize, y: usize) -> [u8; 3] {
        let i = (y * self.width + x) * self.channels;
        if self.channels >= 3 {
            [self.data[i], self.data[i + 1], self.data[i + 2]]
        } else {
            [self.data[i]; 3]
        }
    }
    /// 회색 float \[0,1\]: (0.299R + 0.587G + 0.114B)/255.
    pub fn to_gray(&self) -> GrayImage {
        let n = self.width * self.height;
        let mut data = vec![0.0f32; n];
        if self.channels >= 3 {
            for (i, d) in data.iter_mut().enumerate() {
                let p = &self.data[i * self.channels..i * self.channels + 3];
                *d = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0;
            }
        } else {
            for (i, d) in data.iter_mut().enumerate() {
                *d = self.data[i * self.channels] as f32 / 255.0;
            }
        }
        GrayImage { width: self.width, height: self.height, data }
    }
    /// 면적 평균으로 크기 변경(축소용; 확대 시에도 면적 적분으로 동작).
    pub fn resize_area(&self, nw: usize, nh: usize) -> ImageBuffer {
        let mut out = ImageBuffer::new(nw, nh, self.channels);
        for c in 0..self.channels {
            let plane: Vec<f32> = (0..self.width * self.height).map(|i| self.data[i * self.channels + c] as f32).collect();
            let g = GrayImage { width: self.width, height: self.height, data: plane };
            let r = Integral::new(&g).resample(nw, nh);
            for (i, v) in r.data.iter().enumerate() {
                out.data[i * self.channels + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        out
    }
}

/// f32 회색 영상.
#[derive(Clone, Debug, PartialEq)]
pub struct GrayImage {
    /// 너비(픽셀).
    pub width: usize,
    /// 높이(픽셀).
    pub height: usize,
    /// 픽셀 값(행 우선).
    pub data: Vec<f32>,
}

impl GrayImage {
    /// 0 으로 채운 영상.
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![0.0; width * height] }
    }
    #[inline]
    /// 정수 좌표 픽셀 값.
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }
    /// 정수 좌표(가장자리 고정).
    #[inline]
    pub fn at_clamped(&self, x: i32, y: i32) -> f32 {
        let xi = x.clamp(0, self.width as i32 - 1) as usize;
        let yi = y.clamp(0, self.height as i32 - 1) as usize;
        self.data[yi * self.width + xi]
    }
    /// 정수 중심 규약 쌍선형(가장자리 고정, CUDA 텍스처 clamp 와 같은 동작).
    #[inline]
    pub fn bilinear_clamped(&self, x: f32, y: f32) -> f32 {
        bilinear_clamped(&self.data, self.width, self.height, x, y)
    }
    /// 면적 평균 크기 변경(적분영상 사용).
    pub fn resize_area(&self, nw: usize, nh: usize) -> GrayImage {
        Integral::new(self).resample(nw, nh)
    }
    /// 3차(Keys a=−0.5) 보간 크기 변경(확대용, 정수 중심 규약).
    pub fn resize_cubic(&self, nw: usize, nh: usize) -> GrayImage {
        let sx = self.width as f32 / nw as f32;
        let sy = self.height as f32 / nh as f32;
        let mut out = GrayImage::new(nw, nh);
        for y in 0..nh {
            let fy = (y as f32 + 0.5) * sy - 0.5;
            let y0 = fy.floor();
            let ty = fy - y0;
            let wy = cubic_weights(ty);
            for x in 0..nw {
                let fx = (x as f32 + 0.5) * sx - 0.5;
                let x0 = fx.floor();
                let wx = cubic_weights(fx - x0);
                let mut s = 0.0;
                for (j, wyj) in wy.iter().enumerate() {
                    let yy = y0 as i32 - 1 + j as i32;
                    let mut row = 0.0;
                    for (i, wxi) in wx.iter().enumerate() {
                        row += wxi * self.at_clamped(x0 as i32 - 1 + i as i32, yy);
                    }
                    s += wyj * row;
                }
                out.data[y * nw + x] = s;
            }
        }
        out
    }
    /// 비율 s 로 크기 변경: 축소는 면적 평균, 확대는 3차 보간. 크기는 반올림.
    pub fn rescale(&self, s: f64) -> GrayImage {
        let nw = ((self.width as f64 * s).round() as usize).max(1);
        let nh = ((self.height as f64 * s).round() as usize).max(1);
        if s < 1.0 {
            self.resize_area(nw, nh)
        } else {
            self.resize_cubic(nw, nh)
        }
    }
}

fn cubic_weights(t: f32) -> [f32; 4] {
    let a = -0.5f32;
    let f = |x: f32| {
        let x = x.abs();
        if x <= 1.0 {
            (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
        } else if x < 2.0 {
            a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a
        } else {
            0.0
        }
    };
    [f(1.0 + t), f(t), f(1.0 - t), f(2.0 - t)]
}

/// 정수 중심 규약 쌍선형, 가장자리 고정.
#[inline]
pub fn bilinear_clamped(data: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let xm = (w - 1) as f32;
    let ym = (h - 1) as f32;
    let x = x.clamp(0.0, xm);
    let y = y.clamp(0.0, ym);
    let x0 = x as usize; // x ≥ 0 이므로 floor
    let y0 = y as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let r0 = y0 * w;
    let r1 = y1 * w;
    let a = data[r0 + x0] + fx * (data[r0 + x1] - data[r0 + x0]);
    let b = data[r1 + x0] + fx * (data[r1 + x1] - data[r1 + x0]);
    a + fy * (b - a)
}

/// 적분영상(합계표, (w+1)×(h+1), f64). 임의 실수 상자 합을 쌍선형 보간으로 정확히 구한다.
#[derive(Clone, Debug)]
pub struct Integral {
    /// 입력 영상 너비.
    pub width: usize,
    /// 입력 영상 높이.
    pub height: usize,
    sums: Vec<f64>,
}

impl Integral {
    /// 회색 영상의 적분영상을 만든다.
    pub fn new(img: &GrayImage) -> Self {
        let (w, h) = (img.width, img.height);
        let mut sums = vec![0.0f64; (w + 1) * (h + 1)];
        for y in 0..h {
            let mut row = 0.0f64;
            for x in 0..w {
                row += img.data[y * w + x] as f64;
                sums[(y + 1) * (w + 1) + x + 1] = sums[y * (w + 1) + x + 1] + row;
            }
        }
        Self { width: w, height: h, sums }
    }
    /// 모서리 좌표 (x, y) ∈ [0,w]×[0,h] 의 누적합(조각 상수 영상에서 쌍선형이 정확).
    #[inline]
    fn s(&self, x: f64, y: f64) -> f64 {
        let x = x.clamp(0.0, self.width as f64);
        let y = y.clamp(0.0, self.height as f64);
        let x0 = (x.floor() as usize).min(self.width.saturating_sub(1));
        let y0 = (y.floor() as usize).min(self.height.saturating_sub(1));
        let fx = x - x0 as f64;
        let fy = y - y0 as f64;
        let w1 = self.width + 1;
        let a = self.sums[y0 * w1 + x0];
        let b = self.sums[y0 * w1 + x0 + 1];
        let c = self.sums[(y0 + 1) * w1 + x0];
        let d = self.sums[(y0 + 1) * w1 + x0 + 1];
        a * (1.0 - fx) * (1.0 - fy) + b * fx * (1.0 - fy) + c * (1.0 - fx) * fy + d * fx * fy
    }
    /// 모서리 좌표 상자 [x0,x1)×[y0,y1) 평균.
    #[inline]
    pub fn box_mean(&self, x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
        let area = (x1 - x0) * (y1 - y0);
        if area <= 0.0 {
            return 0.0;
        }
        (self.s(x1, y1) - self.s(x0, y1) - self.s(x1, y0) + self.s(x0, y0)) / area
    }
    /// 면적 평균 재표본(출력 픽셀 발자국 = 입력 위 상자).
    pub fn resample(&self, nw: usize, nh: usize) -> GrayImage {
        let sx = self.width as f64 / nw as f64;
        let sy = self.height as f64 / nh as f64;
        let mut out = GrayImage::new(nw, nh);
        for y in 0..nh {
            let (y0, y1) = (y as f64 * sy, (y + 1) as f64 * sy);
            for x in 0..nw {
                out.data[y * nw + x] = self.box_mean(x as f64 * sx, y0, (x + 1) as f64 * sx, y1) as f32;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_resample_exact_halving() {
        let mut g = GrayImage::new(4, 2);
        g.data = vec![1.0, 3.0, 5.0, 7.0, 1.0, 3.0, 5.0, 7.0];
        let r = g.resize_area(2, 1);
        assert!((r.data[0] - 2.0).abs() < 1e-6 && (r.data[1] - 6.0).abs() < 1e-6);
        let r = g.resize_area(3, 1); // 비정수 축척: 첫 칸 = (1 + 3/3)/(4/3)
        assert!((r.data[0] - (1.0 + 3.0 / 3.0) / (4.0 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn bilinear_matches_linear_ramp() {
        let mut g = GrayImage::new(5, 5);
        for y in 0..5 {
            for x in 0..5 {
                g.data[y * 5 + x] = x as f32 + 10.0 * y as f32;
            }
        }
        assert!((g.bilinear_clamped(1.25, 2.5) - 26.25).abs() < 1e-5);
        assert!((g.bilinear_clamped(-3.0, 0.0) - 0.0).abs() < 1e-6);
    }
}
