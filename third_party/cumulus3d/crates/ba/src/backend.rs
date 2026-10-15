/*
 * backend.rs
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

//! 번들 조정 백엔드: 재구성에서 만든 문제 하나([`ProblemInput`])를 신뢰 영역 LM 으로 푼다.
//!
//! 기본은 CPU([`CpuBaBackend`]). CUDA 구현은 cumulus3d-cuda 의 `CudaBundleAdjuster` 이고,
//! 이 크레이트는 그 크레이트에 의존하지 않는다. 문제 구성(게이지 고정·점 선택)과 결과 써 넣기는
//! 백엔드와 무관하게 [`crate::bundle_adjust_with`] 가 한다.

use crate::problem::{BaProblem, ProblemInput};
use crate::tr::{self, TrOptions, TrResult};
use crate::LinearSolverType;
use cumulus3d_core::{Camera, Error, Result, Rigid3, Vec3};
use std::sync::Arc;

/// 백엔드가 돌려주는 해. 배열 순서는 [`ProblemInput`] 과 같다.
#[derive(Clone, Debug)]
pub struct BaSolution {
    /// 최종 프레임 자세(고정 자세는 입력 그대로).
    pub poses: Vec<Rigid3>,
    /// 최종 카메라(상수 카메라는 입력 그대로).
    pub cams: Vec<Camera>,
    /// 최종 3D 점(고정 점은 입력 그대로).
    pub points: Vec<Vec3>,
    /// 신뢰 영역 루프 결과.
    pub result: TrResult,
    /// 실제로 쓴 축소 계통 풀이기(요약 보고용).
    pub linear_solver: LinearSolverType,
}

/// 번들 조정 문제를 푸는 백엔드.
///
/// 구현은 [`TrProblem`](crate::TrProblem) 을 구현하고 [`minimize`](crate::minimize) 로 바깥 루프를 돌리면
/// CPU 와 같은 종료 규칙을 쓴다.
pub trait BaBackend: Send + Sync {
    /// 백엔드 이름(예: `cpu`, `cuda`).
    fn name(&self) -> &str;
    /// 문제를 푼다. 구조를 만들지 못하면 `Err`, 수치 실패는 `result.termination` 으로 알린다.
    fn solve(&self, input: &ProblemInput, opts: &TrOptions) -> Result<BaSolution>;
}

/// CPU 백엔드(rayon 병렬 Schur 조립 + 밀집·희소·반복 풀이기).
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBaBackend;

impl BaBackend for CpuBaBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn solve(&self, input: &ProblemInput, opts: &TrOptions) -> Result<BaSolution> {
        let mut prob = BaProblem::new(input.clone()).ok_or_else(|| Error::Invariant("BA 선형 풀이기 구조 생성 실패".into()))?;
        let result = tr::minimize(&mut prob, opts);
        let linear_solver = prob.solver_kind;
        let st = prob.cur;
        Ok(BaSolution { poses: st.poses, cams: st.cams, points: st.pts, result, linear_solver })
    }
}

/// 옵션 구조체에 넣어 다니는 공유 백엔드 손잡이(`Clone`·`Debug`·기본값 CPU).
#[derive(Clone)]
pub struct BaBackendRef(pub Arc<dyn BaBackend>);

impl BaBackendRef {
    /// 백엔드를 감싼다.
    pub fn new(backend: Arc<dyn BaBackend>) -> Self {
        Self(backend)
    }
    /// CPU 백엔드.
    pub fn cpu() -> Self {
        Self(Arc::new(CpuBaBackend))
    }
    /// 감싼 백엔드.
    pub fn get(&self) -> &dyn BaBackend {
        &*self.0
    }
}

impl Default for BaBackendRef {
    fn default() -> Self {
        Self::cpu()
    }
}

impl std::fmt::Debug for BaBackendRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("BaBackendRef").field(&self.0.name()).finish()
    }
}

impl From<Arc<dyn BaBackend>> for BaBackendRef {
    fn from(b: Arc<dyn BaBackend>) -> Self {
        Self(b)
    }
}
