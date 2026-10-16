/*
 * matcher_gpu.rs
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

//! Checks that CUDA matching top-2 is bit-identical to the CPU. Skipped if there is no device.

use cumulus3d_cuda::{is_available, CudaMatcher};
use cumulus3d_matching::{CpuMatcher, MatcherBackend};
use std::time::Instant;

fn descriptors(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..n * 128)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Mostly small values + occasional large ones (mimicking the SIFT descriptor distribution), quantized so that ties also occur.
            let v = (s >> 33) as u32 % 64;
            (if v > 56 { v * 3 } else { v / 4 * 4 }) as u8
        })
        .collect()
}

#[test]
fn gpu_top2_equals_cpu() {
    if !is_available() {
        eprintln!("CUDA 장치 없음: 건너뜀");
        return;
    }
    let gpu = CudaMatcher::try_default().expect("CUDA 매칭");
    for (n1, n2) in [(1usize, 1usize), (37, 300), (1000, 777), (8192, 8192)] {
        let d1 = descriptors(n1, 1 + n1 as u64);
        let d2 = descriptors(n2, 99 + n2 as u64);
        let t = Instant::now();
        let c = CpuMatcher::default().top2(&d1, n1, &d2, n2);
        let tc = t.elapsed();
        let t = Instant::now();
        let g = gpu.top2(&d1, n1, &d2, n2);
        let tg = t.elapsed();
        eprintln!("{n1}×{n2}: CPU {tc:.2?}, GPU {tg:.2?}");
        assert_eq!(c.0, g.0, "행 {n1}×{n2}");
        assert_eq!(c.1, g.1, "열 {n1}×{n2}");
    }
}
