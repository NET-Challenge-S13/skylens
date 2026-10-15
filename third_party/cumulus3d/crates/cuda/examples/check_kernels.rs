/*
 * check_kernels.rs
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

//! Compiles every CUDA kernel with NVRTC for the given architectures, without a GPU (CI check).
//!
//! ```bash
//! cargo run --release -p cumulus3d-cuda --example check_kernels -- compute_70 compute_86
//! ```

fn main() {
    let archs: Vec<String> = std::env::args().skip(1).collect();
    let archs = if archs.is_empty() { vec!["compute_70".to_string()] } else { archs };
    let mut failed = false;
    for a in archs {
        let arch: &'static str = Box::leak(a.into_boxed_str());
        match cumulus3d_cuda::check_kernels(arch) {
            Ok(list) => {
                for (name, bytes) in list {
                    println!("{arch} {name}: ptx {bytes} bytes");
                }
            }
            Err(e) => {
                eprintln!("{arch}: {e}");
                failed = true;
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}
