/*
 * analyzer.rs
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

//! 모델 통계(model_analyzer 하위 명령 출력).

use crate::io::fmt::format_g;
use crate::reconstruction::Reconstruction;

/// model_analyzer 의 모든 항목.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelStats {
    /// rig 수.
    pub num_rigs: usize,
    /// 카메라 수.
    pub num_cameras: usize,
    /// 프레임 수.
    pub num_frames: usize,
    /// 등록된 프레임 수.
    pub registered_frame_count: usize,
    /// 영상 수.
    pub num_images: usize,
    /// 등록된 영상 수.
    pub registered_image_count: usize,
    /// 3D 점 수.
    pub num_points3d: usize,
    /// 등록 영상들의 "3D 점이 연결된 2D 점 수" 합.
    pub num_observations: usize,
    /// 평균 트랙 길이.
    pub mean_track_length: f64,
    /// 등록 영상당 평균 관측 수.
    pub mean_observations_per_image: f64,
    /// 저장된 점 error(−1 제외) 평균. 재계산하지 않음.
    pub mean_reprojection_error: f64,
}

impl ModelStats {
    /// 재구성에서 통계를 계산한다.
    pub fn compute(rec: &Reconstruction) -> Self {
        Self {
            num_rigs: rec.num_rigs(),
            num_cameras: rec.num_cameras(),
            num_frames: rec.num_frames(),
            registered_frame_count: rec.registered_frame_count(),
            num_images: rec.num_images(),
            registered_image_count: rec.registered_image_count(),
            num_points3d: rec.num_points3d(),
            num_observations: rec.total_observations(),
            mean_track_length: rec.mean_track_len(),
            mean_observations_per_image: rec.mean_obs_per_registered_image(),
            mean_reprojection_error: rec.mean_reproj_error(),
        }
    }

    /// 고정된 항목 순서·문구(로그 접두 제외). 정수 %d, 실수 %f.
    pub fn lines(&self) -> Vec<String> {
        vec![
            format!("Rigs: {}", self.num_rigs),
            format!("Cameras: {}", self.num_cameras),
            format!("Frames: {}", self.num_frames),
            format!("Registered frames: {}", self.registered_frame_count),
            format!("Images: {}", self.num_images),
            format!("Registered images: {}", self.registered_image_count),
            format!("Points: {}", self.num_points3d),
            format!("Observations: {}", self.num_observations),
            format!("Mean track length: {:.6}", self.mean_track_length),
            format!("Mean observations per image: {:.6}", self.mean_observations_per_image),
            format!("Mean reprojection error: {:.6}px", self.mean_reprojection_error),
        ]
    }
}

/// model_analyzer 출력 줄. verbose 이면 카메라·등록 영상 목록을 덧붙인다.
pub fn analyzer_lines(rec: &Reconstruction, verbose: bool) -> Vec<String> {
    let mut out = ModelStats::compute(rec).lines();
    if verbose {
        out.push("--- Cameras ---".into());
        for c in rec.cameras().values() {
            // 설계 결정: 파라미터 숫자 서식은 유효숫자 6자리 %g.
            let params = c.params.iter().map(|p| format_g(*p, 6)).collect::<Vec<_>>().join(", ");
            out.push(format!(" - Camera Id: {}, Model Name: {}, Params: {}", c.camera_id, c.model.name(), params));
        }
        out.push("--- Images ---".into());
        for id in rec.registered_images() {
            let name = rec.image(id).map(|im| im.name.as_str()).unwrap_or("");
            out.push(format!(" - Registered Image Id: {id}, Name: {name}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{Camera, CameraModelKind};
    use crate::geometry::{Rigid3, Vec2, Vec3};
    use crate::reconstruction::{Image, TrackEntry};

    #[test]
    fn small_model_stats() {
        let mut r = Reconstruction::new();
        let mut cam = Camera::from_focal(CameraModelKind::OpenCv, 100.0, 200, 100);
        cam.camera_id = 1;
        r.add_camera_own_rig(cam).unwrap();
        for i in 1..=3u32 {
            let pts = (0..4).map(|k| Vec2::new(k as f64, 0.0));
            r.add_image_own_frame(Image::new(i, format!("i{i}.jpg"), 1, pts), Some(Rigid3::identity())).unwrap();
            r.register_image(i).unwrap();
        }
        let t = |i, k| TrackEntry::new(i, k);
        let tracks = [vec![t(1, 0), t(2, 0)], vec![t(1, 1), t(2, 1), t(3, 1)], vec![t(2, 2), t(3, 2)], vec![t(1, 3), t(3, 3)]];
        for (tr, e) in tracks.into_iter().zip([1.0, 2.0, -1.0, 3.0]) {
            let id = r.add_point3d(Vec3::new(0.0, 0.0, 1.0), tr, [0; 3]).unwrap();
            r.set_point3d_error(id, e).unwrap();
        }
        let s = ModelStats::compute(&r);
        assert_eq!(s.num_observations, 9);
        assert_eq!(s.mean_track_length, 2.25);
        assert_eq!(s.mean_observations_per_image, 3.0);
        assert_eq!(s.mean_reprojection_error, 2.0);
        let l = analyzer_lines(&r, true);
        assert_eq!(
            &l[..11],
            &[
                "Rigs: 1",
                "Cameras: 1",
                "Frames: 3",
                "Registered frames: 3",
                "Images: 3",
                "Registered images: 3",
                "Points: 4",
                "Observations: 9",
                "Mean track length: 2.250000",
                "Mean observations per image: 3.000000",
                "Mean reprojection error: 2.000000px",
            ]
        );
        assert_eq!(l[11], "--- Cameras ---");
        assert_eq!(l[12], " - Camera Id: 1, Model Name: OPENCV, Params: 100, 100, 100, 50, 0, 0, 0, 0");
        assert_eq!(l[14], " - Registered Image Id: 1, Name: i1.jpg");
        let empty = ModelStats::compute(&Reconstruction::new());
        assert_eq!(empty.lines()[10], "Mean reprojection error: 0.000000px");
    }
}
