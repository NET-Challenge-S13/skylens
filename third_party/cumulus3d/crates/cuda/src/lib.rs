/*
 * lib.rs
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

//! CUDA backend (cudarc + NVRTC runtime compilation). Builds on machines without CUDA (dynamic loading),
//! and at run time [`is_available`] is false if there is no device.
//!
//! - [`CudaPatchMatch`]: implementation of `cumulus3d_dense::PatchMatchBackend` (per-scale run, evaluation and readout of multi-scale
//!   densification).
//! - [`CudaMatcher`]: implementation of `cumulus3d_matching::MatcherBackend` (integer dot product top-2).
//! - [`CudaSift`]: implementation of `cumulus3d_features::SiftEngine` (scale space, detection, orientation, selection and descriptors
//!   all on the GPU; same quality as CpuSift but not bit-identical).
//! - [`CudaBundleAdjuster`]: implementation of `cumulus3d_ba::BaBackend` (residuals, Jacobians, Schur assembly, reduced system solve
//!   and back-substitution on the GPU, the LM loop on the CPU).
//! - [`CudaSparse`]: implementation of `cumulus3d_matching::TwoViewBackend` (two-view geometry hypothesis scoring),
//!   `cumulus3d_sfm::AbsPoseBackend` (absolute pose hypothesis scoring) and `cumulus3d_sfm::TriangulationBackend` (RANSAC
//!   triangulation, observation checks).
//!
//! `unsafe` is used only where cudarc marks things unsafe (kernel launch, pinned host memory allocation, `DeviceRepr` marking).
//!
//! # Usage example
//!
//! ```
//! use cumulus3d_cuda::{is_available, CudaMatcher};
//! use cumulus3d_matching::MatcherBackend;
//!
//! // On machines without CUDA this is false, so nothing is done.
//! if is_available() {
//!     let matcher = CudaMatcher::try_default().expect("CUDA 장치 0");
//!     let (d1, d2) = (vec![1u8; 128 * 4], vec![1u8; 128 * 3]); // 4 and 3 128-byte SIFT descriptors
//!     let (rows, cols) = matcher.top2(&d1, 4, &d2, 3);
//!     assert_eq!((rows.len(), cols.len()), (4, 3));
//! }
//! ```
//!
//! ```no_run
//! use cumulus3d_cuda::CudaPatchMatch;
//! use cumulus3d_dense::{densify, DenseScene, DensifyOptions};
//!
//! fn run(scene: &DenseScene) -> Result<(), Box<dyn std::error::Error>> {
//!     let pm = CudaPatchMatch::try_default()?; // device 0, default options
//!     let out = densify(scene, &DensifyOptions::default(), &pm, None)?;
//!     println!("조밀 점 {}개", out.cloud.len());
//!     Ok(())
//! }
//! ```
#![warn(missing_docs)]

mod ba;
mod device;
mod matcher;
mod patchmatch;
mod sift;
mod sparse;

pub use ba::{CudaBaError, CudaBundleAdjuster, CPU_CHOLESKY_N, MAX_DENSE_N};
pub use device::{is_available, CudaDevice, GpuError};
pub use matcher::CudaMatcher;
pub use patchmatch::{CudaPatchMatch, CudaPatchMatchOptions};
pub use sift::CudaSift;
pub use sparse::CudaSparse;

/// Tries compiling all kernels with NVRTC without a device (for CI checks). `arch` example: `"compute_70"`.
/// Uses the same options and define combinations as real runs (patchmatch: window step 1 and 2, hardware interpolation on and off, fast
/// math on and off).
/// On success returns the list of (kernel name, PTX byte count); on failure the first error. The NVRTC library must exist.
pub fn check_kernels(arch: &'static str) -> Result<Vec<(String, usize)>, String> {
    use cudarc::nvrtc::compile_ptx_with_opts;
    let compile = |name: String, src: &str, fmad: bool, fast: bool, defs: &[String]| {
        let opts = device::compile_options(Some(arch), fmad, fast, defs);
        compile_ptx_with_opts(src, opts).map(|ptx| (name.clone(), ptx.to_src().len())).map_err(|e| format!("{name}: {e:?}"))
    };
    let mut out = vec![
        compile("sift".into(), sift::SRC, false, false, &[])?,
        compile("matcher".into(), matcher::SRC, true, false, &[])?,
        compile("ba".into(), ba::SRC, true, false, &[])?,
        compile("sparse".into(), sparse::SRC, false, false, &[])?,
    ];
    for step in [1, 2] {
        for hw in [false, true] {
            for fast in [false, true] {
                let defs = patchmatch::defines(5, step, hw, 2);
                out.push(compile(
                    format!("patchmatch(step={step}, hw_interp={hw}, fast_math={fast})"),
                    patchmatch::SRC,
                    true,
                    fast,
                    &defs,
                )?);
            }
        }
    }
    Ok(out)
}
