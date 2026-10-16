/*
 * undistort_gpu.rs
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

use cumulus3d_core::{Camera, CameraModelKind};
use cumulus3d_cuda::{is_available, CudaUndistort};
use cumulus3d_dense::{CameraUndistortion, ImageBuffer, UndistortBackend, UndistortOptions};
use std::time::Instant;

/// Deterministic textured image: smooth gradients + sinusoids + noise-like hash, per channel.
fn image(w: usize, h: usize, ch: usize) -> ImageBuffer {
    let mut img = ImageBuffer::new(w, h, ch);
    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let (fx, fy) = (x as f64, y as f64);
                let mut z = (y * w + x) as u64 * 31 + c as u64;
                z ^= z >> 13;
                z = z.wrapping_mul(0x9e3779b97f4a7c15);
                let noise = ((z >> 58) as f64) - 32.0; // [-32, 31]
                let v = 128.0 + 60.0 * (fx * 0.05 + c as f64).sin() * (fy * 0.07).cos() + 0.02 * (fx + fy) + 0.3 * noise;
                img.data[(y * w + x) * ch + c] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }
    img
}

fn compare(cpu: &ImageBuffer, gpu: &ImageBuffer, margin: usize) -> (u8, f64, u8) {
    assert_eq!((cpu.width, cpu.height, cpu.channels), (gpu.width, gpu.height, gpu.channels));
    let (mut max_all, mut max_in, mut sum) = (0u8, 0u8, 0u64);
    for y in 0..cpu.height {
        for x in 0..cpu.width {
            for c in 0..cpu.channels {
                let i = (y * cpu.width + x) * cpu.channels + c;
                let d = cpu.data[i].abs_diff(gpu.data[i]);
                sum += d as u64;
                max_all = max_all.max(d);
                if x >= margin && y >= margin && x + margin < cpu.width && y + margin < cpu.height {
                    max_in = max_in.max(d);
                }
            }
        }
    }
    (max_all, sum as f64 / cpu.data.len() as f64, max_in)
}

#[test]
fn gpu_undistort_equals_cpu() {
    if !is_available() {
        eprintln!("CUDA device not available: skipped");
        return;
    }
    let gpu = CudaUndistort::try_default().expect("CUDA undistort");
    let opencv = |w: u64, h: u64| {
        let (fx, fy) = (0.79 * w as f64, 0.79 * w as f64);
        Camera::new(1, CameraModelKind::OpenCv, w, h, vec![fx, fy, w as f64 / 2.0, h as f64 / 2.0, 0.0824, -0.0849, 0.0022, 0.0007])
            .unwrap()
    };
    let cases = [
        // (camera, options, channels): the pipeline case, the same without downscale, other models, gray.
        (opencv(2048, 1152), UndistortOptions::pipeline(), 3),
        (opencv(2048, 1152), UndistortOptions::default(), 3),
        (opencv(640, 360), UndistortOptions { max_image_size: 300, ..UndistortOptions::default() }, 3),
        (opencv(501, 333), UndistortOptions { max_image_size: 200, ..UndistortOptions::default() }, 1),
        (
            Camera::new(2, CameraModelKind::Radial, 640, 480, vec![500.0, 320.0, 240.0, 0.1, -0.03]).unwrap(),
            UndistortOptions { max_image_size: 320, ..UndistortOptions::default() },
            3,
        ),
        (
            Camera::new(3, CameraModelKind::SingleFocalRadial, 640, 480, vec![500.0, 320.0, 240.0, -0.08]).unwrap(),
            UndistortOptions::default(),
            3,
        ),
    ];
    for (cam, opts, ch) in cases {
        let u = CameraUndistortion::new(&cam, &opts).unwrap();
        let img = image(cam.width as usize, cam.height as usize, ch);
        let t = Instant::now();
        let c = u.undistort_image(&img).unwrap();
        let tc = t.elapsed();
        let t = Instant::now();
        let g = gpu.undistort_image(&u, &img).unwrap();
        let tg = t.elapsed();
        // The output camera is the CPU-computed pinhole camera; the GPU image must have exactly its geometry.
        assert_eq!((g.width as u64, g.height as u64), (u.pinhole.width, u.pinhole.height));
        let (max_all, mean, max_in) = compare(&c, &g, 3);
        eprintln!(
            "{}x{} ch{ch} -> {}x{}: CPU {tc:.2?} GPU {tg:.2?}, max |diff| {max_all} (interior {max_in}), mean {mean:.4}",
            cam.width, cam.height, g.width, g.height
        );
        assert!(max_in <= 1, "interior max diff {max_in}");
        assert!(mean <= 0.05, "mean diff {mean}");
    }
}
