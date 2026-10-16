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

//! Model statistics (output of the model_analyzer subcommand).

use crate::io::fmt::format_g;
use crate::reconstruction::Reconstruction;

/// All items of model_analyzer.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelStats {
    /// Number of rigs.
    pub num_rigs: usize,
    /// Number of cameras.
    pub num_cameras: usize,
    /// Number of frames.
    pub num_frames: usize,
    /// Number of registered frames.
    pub registered_frame_count: usize,
    /// Number of images.
    pub num_images: usize,
    /// Number of registered images.
    pub registered_image_count: usize,
    /// Number of 3D points.
    pub num_points3d: usize,
    /// Sum over registered images of "number of 2D points linked to a 3D point".
    pub num_observations: usize,
    /// Mean track length.
    pub mean_track_length: f64,
    /// Mean number of observations per registered image.
    pub mean_observations_per_image: f64,
    /// Mean of the stored point errors (excluding -1). Not recomputed.
    pub mean_reprojection_error: f64,
}

impl ModelStats {
    /// Computes the statistics from a reconstruction.
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

    /// Fixed item order and wording (log prefix excluded). Integers use %d, reals use %f.
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

/// model_analyzer output lines. If verbose, the camera and registered-image lists are appended.
pub fn analyzer_lines(rec: &Reconstruction, verbose: bool) -> Vec<String> {
    let mut out = ModelStats::compute(rec).lines();
    if verbose {
        out.push("--- Cameras ---".into());
        for c in rec.cameras().values() {
            // Design decision: parameter numbers are formatted with 6 significant digits (%g).
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
