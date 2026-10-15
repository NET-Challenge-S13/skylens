/*
 * device.rs
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

//! 장치·문맥·NVRTC 컴파일 공용부.

use cudarc::driver::{CudaContext, CudaModule, CudaStream};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

/// CUDA 백엔드 오류.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("CUDA 사용 불가: {0}")]
    /// 드라이버·NVRTC 라이브러리 또는 장치가 없음.
    NotAvailable(String),
    #[error("CUDA 드라이버 오류: {0:?}")]
    /// CUDA 드라이버 호출 오류.
    Driver(#[from] cudarc::driver::DriverError),
    #[error("NVRTC 컴파일 실패: {0}")]
    /// NVRTC 커널 컴파일 실패.
    Compile(String),
}

/// 드라이버·NVRTC 라이브러리가 있고 장치가 하나 이상이면 참.
pub fn is_available() -> bool {
    // 동적 로딩: 라이브러리가 없으면 cudarc 가 panic 하므로 먼저 존재를 확인한다.
    // SAFETY: 라이브러리 열기만 시도한다.
    let libs = unsafe { cudarc::driver::sys::is_culib_present() && cudarc::nvrtc::sys::is_culib_present() };
    if !libs {
        return false;
    }
    std::panic::catch_unwind(|| CudaContext::device_count().map(|n| n > 0).unwrap_or(false)).unwrap_or(false)
}

/// 한 GPU 의 문맥과 전용 스트림.
pub struct CudaDevice {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    arch: Option<&'static str>,
    name: String,
    /// 텍스처 시작 주소·행 간격 정렬(바이트).
    pub(crate) tex_align: usize,
    pub(crate) pitch_align: usize,
}

impl CudaDevice {
    /// 장치 `ordinal` 을 연다.
    pub fn new(ordinal: usize) -> Result<Arc<Self>, GpuError> {
        if !is_available() {
            return Err(GpuError::NotAvailable("드라이버/NVRTC 라이브러리 또는 장치 없음".into()));
        }
        let ctx =
            std::panic::catch_unwind(|| CudaContext::new(ordinal)).map_err(|_| GpuError::NotAvailable("문맥 생성 중 panic".into()))??;
        let stream = ctx.new_stream()?;
        let cc = ctx.compute_capability()?;
        let arch = match cc {
            (7, 0) => Some("compute_70"),
            (7, 2) => Some("compute_72"),
            (7, 5) => Some("compute_75"),
            (8, 0) => Some("compute_80"),
            (8, 6) => Some("compute_86"),
            (8, 7) => Some("compute_87"),
            (8, 9) => Some("compute_89"),
            (9, 0) => Some("compute_90"),
            _ => None,
        };
        let name = ctx.name().unwrap_or_default();
        use cudarc::driver::sys::CUdevice_attribute as A;
        let tex_align = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT).unwrap_or(512).max(4) as usize;
        let pitch_align = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT).unwrap_or(32).max(4) as usize;
        Ok(Arc::new(Self { ctx, stream, arch, name, tex_align, pitch_align }))
    }

    /// 장치 이름(예: "Tesla V100-PCIE-16GB").
    pub fn name(&self) -> &str {
        &self.name
    }

    /// CUDA C 소스를 NVRTC 로 컴파일해 적재한다. `fmad = false` 면 곱셈-덧셈 융합을 막아 CPU 와 반올림을 맞춘다.
    pub(crate) fn compile(&self, src: &str, fmad: bool, defines: &[String]) -> Result<Arc<CudaModule>, GpuError> {
        self.compile_with(src, fmad, false, defines)
    }

    /// `fast = true` 면 근사 나눗셈·제곱근과 비정규수 0 처리(속도 우선 커널용).
    pub(crate) fn compile_with(&self, src: &str, fmad: bool, fast: bool, defines: &[String]) -> Result<Arc<CudaModule>, GpuError> {
        let opts = compile_options(self.arch, fmad, fast, defines);
        let ptx = compile_ptx_with_opts(src, opts).map_err(|e| GpuError::Compile(format!("{e:?}")))?;
        Ok(self.ctx.load_module(ptx)?)
    }
}

/// NVRTC 옵션(장치 컴파일과 [`crate::check_kernels`] 가 같은 값을 쓴다).
pub(crate) fn compile_options(arch: Option<&'static str>, fmad: bool, fast: bool, defines: &[String]) -> CompileOptions {
    let mut options = vec!["--std=c++14".to_string()];
    options.extend(defines.iter().map(|d| format!("-D{d}")));
    CompileOptions { arch, fmad: Some(fmad), ftz: Some(fast), prec_div: Some(!fast), prec_sqrt: Some(!fast), options, ..Default::default() }
}
