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

//! Alignment errors. Every failure condition is distinguished so that failures are never silently ignored.

/// align crate error.
#[derive(Debug, thiserror::Error)]
pub enum AlignError {
    /// core crate error (I/O, data structures).
    #[error("core 오류: {0}")]
    Core(#[from] cumulus3d_core::Error),
    /// Inlier threshold is not positive.
    #[error("alignment_max_error 는 양수여야 함: {0}")]
    BadMaxError(f64),
    /// Invalid argument.
    #[error("잘못된 인자: {0}")]
    InvalidArgument(String),
    /// Not enough GPS reference positions.
    #[error("기준 위치 부족: {found}개 (최소 {required})")]
    TooFewReferences {
        /// Number found.
        found: usize,
        /// Minimum number required.
        required: usize,
    },
    /// Not enough images in common between the model and the GPS list.
    #[error("공통 영상 부족: {found}개 (최소 {required})")]
    TooFewCommonImages {
        /// Number found.
        found: usize,
        /// Minimum number required.
        required: usize,
    },
    /// Not enough shared 3D point correspondences between the two reconstructions.
    #[error("공유 3D 점 대응 부족: {found}개 (최소 {required})")]
    TooFewCorrespondences {
        /// Number found.
        found: usize,
        /// Minimum number required.
        required: usize,
    },
    /// Robust Sim3 estimation found no model.
    #[error("견고 Sim3 추정 실패 (인라이어 {inliers}개)")]
    RansacFailed {
        /// Number of inliers of the best model.
        inliers: usize,
    },
    /// Point configuration is degenerate, so the Umeyama solution does not exist.
    #[error("Umeyama 해 없음(퇴화 배치)")]
    Degenerate,
}

/// `Result` of the align crate.
pub type Result<T> = std::result::Result<T, AlignError>;
