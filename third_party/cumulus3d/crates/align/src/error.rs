/*
 * error.rs
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

//! 정렬 오류. 실패를 조용히 넘기지 않도록 모든 실패 조건을 구분한다.

/// align 크레이트 오류.
#[derive(Debug, thiserror::Error)]
pub enum AlignError {
    /// core 크레이트 오류(입출력·자료 구조).
    #[error("core 오류: {0}")]
    Core(#[from] cumulus3d_core::Error),
    /// 인라이어 임계가 양수가 아님.
    #[error("alignment_max_error 는 양수여야 함: {0}")]
    BadMaxError(f64),
    /// 잘못된 인자.
    #[error("잘못된 인자: {0}")]
    InvalidArgument(String),
    /// GPS 기준 위치가 부족함.
    #[error("기준 위치 부족: {found}개 (최소 {required})")]
    TooFewReferences {
        /// 찾은 개수.
        found: usize,
        /// 필요한 최소 개수.
        required: usize,
    },
    /// 모델과 GPS 목록의 공통 영상이 부족함.
    #[error("공통 영상 부족: {found}개 (최소 {required})")]
    TooFewCommonImages {
        /// 찾은 개수.
        found: usize,
        /// 필요한 최소 개수.
        required: usize,
    },
    /// 두 재구성 간 공유 3D 점 대응이 부족함.
    #[error("공유 3D 점 대응 부족: {found}개 (최소 {required})")]
    TooFewCorrespondences {
        /// 찾은 개수.
        found: usize,
        /// 필요한 최소 개수.
        required: usize,
    },
    /// 견고 Sim3 추정이 모델을 찾지 못함.
    #[error("견고 Sim3 추정 실패 (인라이어 {inliers}개)")]
    RansacFailed {
        /// 최선 모델의 인라이어 수.
        inliers: usize,
    },
    /// 점 배치가 퇴화해 Umeyama 해가 없음.
    #[error("Umeyama 해 없음(퇴화 배치)")]
    Degenerate,
}

/// align 크레이트의 `Result`.
pub type Result<T> = std::result::Result<T, AlignError>;
