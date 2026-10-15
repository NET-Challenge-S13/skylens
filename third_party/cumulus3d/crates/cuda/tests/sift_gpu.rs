/*
 * sift_gpu.rs
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

//! CUDA SIFT 가 CPU SIFT 와 같은 결과(특징·기술자 비트 단위)를 내는지. 장치가 없으면 건너뛴다.

use cumulus3d_cuda::{is_available, CudaSift};
use cumulus3d_features::{CpuSift, FeatureSelection, GrayImage, SiftEngine, SiftOptions};

/// 여러 크기의 얼룩 무늬(결정적).
fn image(w: usize, h: usize, seed: u64) -> GrayImage {
    let mut data = vec![0u8; w * h];
    let blobs: Vec<(f64, f64, f64, f64)> = (0..400)
        .map(|i| {
            let mut z = seed.wrapping_add(i).wrapping_mul(0x9e3779b97f4a7c15);
            let mut r = || {
                z ^= z >> 29;
                z = z.wrapping_mul(0xbf58476d1ce4e5b9);
                (z >> 11) as f64 / (1u64 << 53) as f64
            };
            (r() * w as f64, r() * h as f64, 2.0 + 18.0 * r(), r() - 0.5)
        })
        .collect();
    for y in 0..h {
        for x in 0..w {
            let mut v = 0.5;
            for &(cx, cy, s, a) in &blobs {
                let d2 = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)) / (2.0 * s * s);
                if d2 < 9.0 {
                    v += a * (-d2).exp();
                }
            }
            data[y * w + x] = (v.clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
    GrayImage { width: w, height: h, data }
}

#[test]
fn gpu_sift_equals_cpu() {
    if !is_available() {
        eprintln!("CUDA 장치 없음: 건너뜀");
        return;
    }
    let gpu = CudaSift::try_default().expect("CUDA SIFT");
    let cpu = CpuSift::new();
    for (w, h, opts) in [
        (643usize, 417usize, SiftOptions::default()),
        (512, 384, SiftOptions { first_octave: 0, ..SiftOptions::default() }),
        (640, 480, SiftOptions { max_num_features: 300, selection: FeatureSelection::TopK, ..SiftOptions::default() }),
        (
            640,
            480,
            SiftOptions {
                max_num_features: 500,
                selection: FeatureSelection::SpatialGrid { cells: 4 },
                upright: true,
                ..SiftOptions::default()
            },
        ),
    ] {
        let img = image(w, h, (w * h) as u64);
        let c = cpu.extract(&img, &opts).unwrap();
        let g = gpu.extract(&img, &opts).unwrap();
        eprintln!("{w}×{h}: CPU {} GPU {} 특징", c.len(), g.len());
        assert!(!c.is_empty());
        assert_eq!(c.features, g.features);
        assert_eq!(c.descriptors.as_slice(), g.descriptors.as_slice());
    }
}
