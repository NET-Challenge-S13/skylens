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

//! CUDA 백엔드(cudarc + NVRTC 런타임 컴파일). CUDA 가 없는 기계에서도 빌드되고(동적 로딩),
//! 실행 시 장치가 없으면 [`is_available`] 이 거짓이다.
//!
//! - [`CudaPatchMatch`]: `cumulus3d_dense::PatchMatchBackend` 구현(다중 스케일 조밀화의 스케일별 실행·평가·판독).
//! - [`CudaMatcher`]: `cumulus3d_matching::MatcherBackend` 구현(정수 내적 top-2).
//! - [`CudaSift`]: `cumulus3d_features::SiftEngine` 구현(GPU 스케일 공간 + CPU 검출·기술자, CpuSift 와 같은 결과).
//!
//! `unsafe` 는 cudarc 가 unsafe 로 둔 곳(커널 발사, 고정 호스트 메모리 할당, `DeviceRepr` 표시)에만 쓴다.
//!
//! # 사용 예
//!
//! ```
//! use cumulus3d_cuda::{is_available, CudaMatcher};
//! use cumulus3d_matching::MatcherBackend;
//!
//! // CUDA 가 없는 기계에서는 거짓이므로 아무것도 하지 않는다.
//! if is_available() {
//!     let matcher = CudaMatcher::try_default().expect("CUDA 장치 0");
//!     let (d1, d2) = (vec![1u8; 128 * 4], vec![1u8; 128 * 3]); // 128바이트 SIFT 기술자 4개, 3개
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
//!     let pm = CudaPatchMatch::try_default()?; // 장치 0, 기본 옵션
//!     let out = densify(scene, &DensifyOptions::default(), &pm, None)?;
//!     println!("조밀 점 {}개", out.cloud.len());
//!     Ok(())
//! }
//! ```
#![warn(missing_docs)]

mod device;
mod matcher;
mod patchmatch;
mod sift;

pub use device::{is_available, CudaDevice, GpuError};
pub use matcher::CudaMatcher;
pub use patchmatch::{CudaPatchMatch, CudaPatchMatchOptions};
pub use sift::CudaSift;

/// 모든 커널을 장치 없이 NVRTC 로 컴파일해 본다(CI 검사용). `arch` 예: `"compute_70"`.
/// 실제 실행 때와 같은 옵션·정의 조합(patchmatch: 창 간격 1·2, 하드웨어 보간 켬·끔, 빠른 수학 켬·끔)을 쓴다.
/// 성공하면 (커널 이름, PTX 바이트 수) 목록, 실패하면 첫 오류. NVRTC 라이브러리는 있어야 한다.
pub fn check_kernels(arch: &'static str) -> Result<Vec<(String, usize)>, String> {
    use cudarc::nvrtc::compile_ptx_with_opts;
    let compile = |name: String, src: &str, fmad: bool, fast: bool, defs: &[String]| {
        let opts = device::compile_options(Some(arch), fmad, fast, defs);
        compile_ptx_with_opts(src, opts).map(|ptx| (name.clone(), ptx.to_src().len())).map_err(|e| format!("{name}: {e:?}"))
    };
    let mut out = vec![compile("sift".into(), sift::SRC, false, false, &[])?, compile("matcher".into(), matcher::SRC, true, false, &[])?];
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
