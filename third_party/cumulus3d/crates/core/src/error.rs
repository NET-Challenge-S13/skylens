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

//! 오류 타입.

use std::path::PathBuf;

/// core 전반의 오류.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("입출력 오류: {0}")]
    /// 입출력 실패.
    Io(#[from] std::io::Error),
    #[error("구문 오류 ({path}:{line}): {msg}")]
    /// 텍스트 파일 구문 오류(위치 포함).
    Parse {
        /// 파일 경로.
        path: PathBuf,
        /// 줄 번호(1부터).
        line: usize,
        /// 오류 설명.
        msg: String,
    },
    #[error("형식 오류: {0}")]
    /// 파일 형식 위반.
    Format(String),
    #[error("잘못된 인자: {0}")]
    /// 잘못된 인자.
    InvalidArgument(String),
    #[error("찾을 수 없음: {0}")]
    /// 대상(id·이름)을 찾지 못함.
    NotFound(String),
    #[error("이미 존재: {0}")]
    /// 이미 존재하는 대상을 다시 추가함.
    AlreadyExists(String),
    #[error("불변식 위반: {0}")]
    /// 자료 구조 불변식 위반.
    Invariant(String),
    #[error("지원하지 않음: {0}")]
    /// 지원하지 않는 기능·모델.
    Unsupported(String),
}

/// core 전반의 `Result` 별칭.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn parse(path: impl Into<PathBuf>, line: usize, msg: impl Into<String>) -> Self {
        Error::Parse { path: path.into(), line, msg: msg.into() }
    }
}
