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

use cudarc::driver::{CudaContext, CudaModule, CudaStream, DevicePtr, DeviceRepr, DriverError, HostSlice};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

/// 내려 받기 전에 스트림을 먼저 기다리는 복사.
///
/// 일반(페이지) 메모리로 내려 받는 `cuMemcpyDtoHAsync` 는 앞선 커널이 끝날 때까지 드라이버 전역 잠금을
/// 쥐고 기다린다. 그동안 같은 프로세스의 다른 스레드(UNet, 희소 단계, SIFT)의 CUDA 호출이 모두 멈춘다
/// (조밀화가 수백 ms 커널 뒤에서 내려 받으면 다른 단계가 그만큼 선다). `cuStreamSynchronize` 는 잠금 없이
/// 기다리므로, 먼저 동기화하고 나서 복사하면 복사는 곧바로 끝난다.
pub(crate) trait SyncedCopy {
    /// [`CudaStream::memcpy_dtoh`] 와 같되 먼저 스트림을 기다린다.
    fn memcpy_dtoh_synced<T: DeviceRepr, Src: DevicePtr<T>, Dst: HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError>;
    /// [`CudaStream::clone_dtoh`] 와 같되 먼저 스트림을 기다린다.
    fn clone_dtoh_synced<T: DeviceRepr, Src: DevicePtr<T>>(&self, src: &Src) -> Result<Vec<T>, DriverError>;
}

impl SyncedCopy for Arc<CudaStream> {
    fn memcpy_dtoh_synced<T: DeviceRepr, Src: DevicePtr<T>, Dst: HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        self.synchronize()?;
        self.memcpy_dtoh(src, dst)
    }
    fn clone_dtoh_synced<T: DeviceRepr, Src: DevicePtr<T>>(&self, src: &Src) -> Result<Vec<T>, DriverError> {
        self.synchronize()?;
        self.clone_dtoh(src)
    }
}

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
        Self::open(ordinal, false)
    }

    /// 장치 `ordinal` 을 열고 스트림을 가장 높은 우선순위로 만든다. 작은 커널을 자주 내고 결과를 곧바로
    /// 기다리는 희소 단계(`CudaSparse`)만 쓴다. 조밀화(PatchMatch)·번들 조정의 긴 커널 대기열 뒤에 줄 서지
    /// 않게 하기 위해서다. SIFT·매칭까지 올리면 희소 단계와 서로 앞지르려 해 삼각측량이 1.2초에서
    /// 2.3~3.2초로, 번들 조정까지 올려도 2.3초로 느려졌다(290초 데모, 위치당 평균).
    pub fn new_high_priority(ordinal: usize) -> Result<Arc<Self>, GpuError> {
        Self::open(ordinal, true)
    }

    fn open(ordinal: usize, high_priority: bool) -> Result<Arc<Self>, GpuError> {
        if !is_available() {
            return Err(GpuError::NotAvailable("드라이버/NVRTC 라이브러리 또는 장치 없음".into()));
        }
        let ctx =
            std::panic::catch_unwind(|| CudaContext::new(ordinal)).map_err(|_| GpuError::NotAvailable("문맥 생성 중 panic".into()))??;
        let stream = if high_priority {
            ctx.bind_to_thread()?;
            let (_, greatest) = cudarc::driver::result::stream::get_priority_range()?;
            ctx.new_stream_with_priority(greatest)?
        } else {
            ctx.new_stream()?
        };
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
