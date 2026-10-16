/*
 * mod.rs
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

//! 입출력: PLY, GPS 참조 파일, 숫자 서식·이진 도우미. 모델 파일 형식은 [`crate::interop`].

pub(crate) mod binary;
pub mod fmt;
pub mod gps;
pub mod ply;

pub use gps::{read_gps_file, GpsRecord};
pub use ply::{read_ply, write_ply, PlyLayout, PointCloud};
