/*
 * matcher.rs
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

//! `MatcherBackend` 의 CUDA 구현: 정수 내적(__dp4a) + 순차 top-2. 결과는 CPU 백엔드와 비트 단위로 같다.

use crate::device::{CudaDevice, GpuError};
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cumulus3d_matching::{CpuMatcher, MatcherBackend, Top2};
use std::sync::{Arc, Mutex};

pub(crate) const SRC: &str = include_str!("kernels/matcher.cu");
const D: usize = 128;

/// CUDA 기술자 매칭 백엔드.
pub struct CudaMatcher {
    dev: Arc<CudaDevice>,
    f: CudaFunction,
    /// 한 번에 하나만 장치를 쓴다(스트림 공유).
    lock: Mutex<()>,
}

fn pack(d: &[u8], n: usize) -> Vec<u32> {
    (0..n * D / 4).map(|k| u32::from_le_bytes([d[4 * k], d[4 * k + 1], d[4 * k + 2], d[4 * k + 3]])).collect()
}

impl CudaMatcher {
    /// 장치 `dev` 에 매칭 커널을 컴파일해 만든다.
    pub fn new(dev: Arc<CudaDevice>) -> Result<Self, GpuError> {
        let module = dev.compile(SRC, true, &[])?;
        Ok(Self { f: module.load_function("top2_rows")?, dev, lock: Mutex::new(()) })
    }

    /// 장치 0 으로 만든다.
    pub fn try_default() -> Result<Self, GpuError> {
        Self::new(CudaDevice::new(0)?)
    }

    /// GPU top-2(행별, 열별).
    pub fn top2_gpu(&self, d1: &[u8], n1: usize, d2: &[u8], n2: usize) -> Result<(Vec<Top2>, Vec<Top2>), GpuError> {
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let s = &self.dev.stream;
        let a = s.clone_htod(&pack(d1, n1))?;
        let b = s.clone_htod(&pack(d2, n2))?;
        let mut ro = s.alloc_zeros::<u32>(3 * n1.max(1))?;
        let mut co = s.alloc_zeros::<u32>(3 * n2.max(1))?;
        for (q, nq, t, nt, out) in [(&a, n1, &b, n2, &mut ro), (&b, n2, &a, n1, &mut co)] {
            if nq == 0 {
                continue;
            }
            let cfg = LaunchConfig { grid_dim: ((nq as u32).div_ceil(128), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
            let (nq_i, nt_i) = (nq as i32, nt as i32);
            let mut l = s.launch_builder(&self.f);
            l.arg(q).arg(&nq_i).arg(t).arg(&nt_i).arg(out);
            // SAFETY: top2_rows(q[nq·32], nq, t[nt·32], nt, out[3·nq]) 와 같은 인자·크기.
            unsafe { l.launch(cfg) }?;
        }
        let rh = s.clone_dtoh(&ro)?;
        let ch = s.clone_dtoh(&co)?;
        s.synchronize()?;
        let conv = |v: &[u32], n: usize| -> Vec<Top2> {
            (0..n).map(|k| Top2 { best: v[3 * k], idx: v[3 * k + 1], second: v[3 * k + 2] }).collect()
        };
        Ok((conv(&rh, n1), conv(&ch, n2)))
    }
}

impl MatcherBackend for CudaMatcher {
    fn top2(&self, d1: &[u8], n1: usize, d2: &[u8], n2: usize) -> (Vec<Top2>, Vec<Top2>) {
        match self.top2_gpu(d1, n1, d2, n2) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("경고: CUDA 매칭 실패({e}), CPU 로 계산");
                CpuMatcher::default().top2(d1, n1, d2, n2)
            }
        }
    }
}
