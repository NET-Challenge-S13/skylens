/*
 * ids.rs
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

//! Identifier types and pair ids.

use crate::error::{Error, Result};

/// Camera id.
pub type CameraId = u32;
/// rig id.
pub type RigId = u32;
/// Frame id.
pub type FrameId = u32;
/// Image id.
pub type ImageId = u32;
/// Index of a 2D point (feature) within an image.
pub type Point2DIdx = u32;
/// 3D point id.
pub type Point3DId = u64;
/// Pair id (see `pair_id_of`).
pub type PairId = u64;

/// Invalid camera id.
pub const INVALID_CAMERA_ID: CameraId = u32::MAX;
/// Invalid rig id.
pub const INVALID_RIG_ID: RigId = u32::MAX;
/// Invalid frame id.
pub const INVALID_FRAME_ID: FrameId = u32::MAX;
/// Invalid image id.
pub const INVALID_IMAGE_ID: ImageId = u32::MAX;
/// Invalid 2D point index.
pub const INVALID_POINT2D_IDX: Point2DIdx = u32::MAX;
/// Invalid 3D point id (the 2D point is not linked to a 3D point).
pub const INVALID_POINT3D_ID: Point3DId = u64::MAX;
/// Invalid pair id.
pub const INVALID_PAIR_ID: PairId = u64::MAX;

/// Pair id constant M = 2^31 − 1. Image ids must be smaller than this.
pub const MAX_NUM_IMAGES: u64 = 2_147_483_647;

/// Sensor kind. Stored as i32 in files.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SensorKind {
    /// Invalid sensor.
    Invalid = -1,
    /// Camera.
    Camera = 0,
    /// Inertial measurement unit.
    Imu = 1,
}

impl SensorKind {
    /// Integer value -> kind. None if the value is unknown.
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            -1 => Some(SensorKind::Invalid),
            0 => Some(SensorKind::Camera),
            1 => Some(SensorKind::Imu),
            _ => None,
        }
    }
    /// Integer value used for file storage.
    pub fn as_i32(self) -> i32 {
        self as i32
    }
    /// Text format name.
    pub fn name(self) -> &'static str {
        match self {
            SensorKind::Invalid => "INVALID",
            SensorKind::Camera => "CAMERA",
            SensorKind::Imu => "IMU",
        }
    }
    /// Text format name -> kind.
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "INVALID" => Some(SensorKind::Invalid),
            "CAMERA" => Some(SensorKind::Camera),
            "IMU" => Some(SensorKind::Imu),
            _ => None,
        }
    }
}

/// Sensor identifier (kind, id). Lexicographic ordering (kind -> id).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SensorKey {
    /// Sensor kind.
    pub sensor_type: SensorKind,
    /// Sensor id within the kind.
    pub id: u32,
}

impl SensorKey {
    /// Creates from (kind, id).
    pub fn new(sensor_type: SensorKind, id: u32) -> Self {
        Self { sensor_type, id }
    }
    /// Camera sensor (id = camera id).
    pub fn camera(camera_id: CameraId) -> Self {
        Self { sensor_type: SensorKind::Camera, id: camera_id }
    }
}

/// Data identifier (sensor, data id). For camera data the data id is the image id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SensorDataKey {
    /// Sensor that produced the data.
    pub sensor_id: SensorKey,
    /// Data id (the image id for a camera).
    pub id: u64,
}

impl SensorDataKey {
    /// Creates from (sensor, data id).
    pub fn new(sensor_id: SensorKey, id: u64) -> Self {
        Self { sensor_id, id }
    }
    /// Camera image data key.
    pub fn image(camera_id: CameraId, image_id: ImageId) -> Self {
        Self { sensor_id: SensorKey::camera(camera_id), id: image_id as u64 }
    }
}

/// Whether the order of the two image ids is "larger -> smaller", so a swap is needed when storing.
pub fn swap_image_pair(image_id1: ImageId, image_id2: ImageId) -> bool {
    image_id1 > image_id2
}

/// Pair id = M·min + max. Error if an image id is M or greater.
pub fn pair_id_of(image_id1: ImageId, image_id2: ImageId) -> Result<PairId> {
    if image_id1 as u64 >= MAX_NUM_IMAGES || image_id2 as u64 >= MAX_NUM_IMAGES {
        return Err(Error::InvalidArgument(format!("영상 id 는 {MAX_NUM_IMAGES} 미만이어야 함: ({image_id1}, {image_id2})")));
    }
    let (a, b) = if image_id1 <= image_id2 { (image_id1, image_id2) } else { (image_id2, image_id1) };
    Ok(MAX_NUM_IMAGES * a as u64 + b as u64)
}

/// Pair id -> (smaller id, larger id).
pub fn images_of_pair(pair_id: PairId) -> (ImageId, ImageId) {
    let id2 = pair_id % MAX_NUM_IMAGES;
    let id1 = (pair_id - id2) / MAX_NUM_IMAGES;
    (id1 as ImageId, id2 as ImageId)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_id_roundtrip() {
        assert_eq!(pair_id_of(1, 2).unwrap(), 2147483649);
        assert_eq!(pair_id_of(2, 1).unwrap(), 2147483649);
        assert_eq!(images_of_pair(2147483649), (1, 2));
        for (a, b) in [(0u32, 0u32), (5, 3), (2147483646, 7), (100, 2147483646)] {
            let id = pair_id_of(a, b).unwrap();
            assert_eq!(images_of_pair(id), (a.min(b), a.max(b)));
        }
        assert!(pair_id_of(2147483647, 1).is_err());
        assert!(swap_image_pair(2, 1));
        assert!(!swap_image_pair(1, 2));
    }

    #[test]
    fn sensor_ordering() {
        let a = SensorDataKey::image(1, 5);
        let b = SensorDataKey::image(2, 1);
        assert!(a < b);
        assert!(SensorKey::new(SensorKind::Camera, 9) < SensorKey::new(SensorKind::Imu, 0));
    }
}
