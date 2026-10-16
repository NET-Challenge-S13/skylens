/*
 * shared.rs
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

//! Shared 3D point correspondences between two reconstructions (post-processing).
//!
//! If the same 2D point index in an image of the same name refers to 3D point i of A and 3D point j of B, then (i, j) is a pair.
//! The preview shares the same features as the refined one (a copy from before BA), so the indices correspond directly.

use crate::error::{AlignError, Result};
use crate::umeyama::{robust_umeyama, RobustUmeyamaOptions, RobustUmeyamaResult};
use cumulus3d_core::{Point3DId, Reconstruction, Vec3};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

/// Image name filter function.
pub type NameFilter = Arc<dyn Fn(&str) -> bool + Send + Sync>;

/// Correspondence collection options.
#[derive(Clone, Default)]
pub struct SharedPointOptions {
    /// Range of position index (the last digit string of the name, [`position_index_from_name`]). Images outside the range are skipped.
    pub position_range: Option<Range<u32>>,
    /// Additional name filter (true = use).
    pub name_filter: Option<NameFilter>,
}

impl std::fmt::Debug for SharedPointOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedPointOptions")
            .field("position_range", &self.position_range)
            .field("name_filter", &self.name_filter.is_some())
            .finish()
    }
}

impl SharedPointOptions {
    fn accepts(&self, name: &str) -> bool {
        if let Some(r) = &self.position_range {
            match position_index_from_name(name) {
                Some(p) if r.contains(&p) => {}
                _ => return false,
            }
        }
        self.name_filter.as_ref().is_none_or(|f| f(name))
    }
}

/// Position index from an image name: the last digit string of the file stem. E.g. `camF/camF_0012.jpg` → 12.
pub fn position_index_from_name(name: &str) -> Option<u32> {
    let file = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let stem = file.rsplit_once('.').map_or(file, |(s, _)| s);
    let end = stem.rfind(|c: char| c.is_ascii_digit())? + 1;
    let start = stem[..end].rfind(|c: char| !c.is_ascii_digit()).map_or(0, |i| i + 1);
    stem[start..end].parse().ok()
}

/// Shared 3D point pairs (A point id, B point id). Duplicates removed, in a sorted deterministic order.
pub fn shared_point_correspondences(a: &Reconstruction, b: &Reconstruction, opts: &SharedPointOptions) -> Vec<(Point3DId, Point3DId)> {
    let b_by_name: HashMap<&str, _> = b.images().map(|im| (im.name.as_str(), im)).collect();
    let mut pairs = Vec::new();
    for ia in a.images() {
        if !opts.accepts(&ia.name) {
            continue;
        }
        let Some(ib) = b_by_name.get(ia.name.as_str()) else { continue };
        // Design decision: if the two images have different 2D point counts (different feature extraction),
        // only the common index range is examined.
        for (pa, pb) in ia.points2d().iter().zip(ib.points2d()) {
            if pa.has_point3d() && pb.has_point3d() {
                pairs.push((pa.point3d_id, pb.point3d_id));
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Estimates the `src_to_dst` Sim3 from shared point correspondences + robust Umeyama
/// (preview → refined, previous refined → new refined).
pub fn align_reconstructions(
    src: &Reconstruction,
    dst: &Reconstruction,
    opts: &SharedPointOptions,
    robust: &RobustUmeyamaOptions,
) -> Result<(RobustUmeyamaResult, Vec<(Point3DId, Point3DId)>)> {
    let pairs = shared_point_correspondences(src, dst, opts);
    if pairs.len() < 3 {
        return Err(AlignError::TooFewCorrespondences { found: pairs.len(), required: 3 });
    }
    let (xs, ys): (Vec<Vec3>, Vec<Vec3>) =
        pairs.iter().map(|(i, j)| (src.point3d(*i).expect("track id").xyz, dst.point3d(*j).expect("track id").xyz)).unzip();
    let r = robust_umeyama(&xs, &ys, robust).ok_or(AlignError::Degenerate)?;
    Ok((r, pairs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::umeyama::tests::{random_sim3, sim_close};
    use cumulus3d_core::{Camera, CameraModelKind, Image, Rigid3, TrackEntry, Vec2};

    fn base(n_img: u32, n_pts: u32) -> Reconstruction {
        let mut rec = Reconstruction::new();
        rec.add_camera_own_rig(camera(1)).unwrap();
        for i in 1..=n_img {
            let pts: Vec<Vec2> = (0..n_pts).map(|k| Vec2::new(k as f64, 0.0)).collect();
            rec.add_image_own_frame(Image::new(i, format!("camF/camF_{:04}.jpg", i - 1), 1, pts), Some(Rigid3::identity())).unwrap();
            rec.register_image(i).unwrap();
        }
        rec
    }

    fn camera(id: u32) -> Camera {
        let mut c = Camera::from_focal(CameraModelKind::Pinhole, 500.0, 640, 480);
        c.camera_id = id;
        c
    }

    #[test]
    fn name_position_index() {
        assert_eq!(position_index_from_name("camF/camF_0012.jpg"), Some(12));
        assert_eq!(position_index_from_name("a7/x_3_0040.png"), Some(40));
        assert_eq!(position_index_from_name("nodigits.jpg"), None);
    }

    #[test]
    fn correspondences_and_alignment() {
        // A: point k is observed at 2D index k of images 1 and 2.
        // B: the same observations with different ids (k+100) and a different coordinate frame.
        let truth = random_sim3(&mut cumulus3d_core::ransac::make_rng(Some(4)));
        let (mut a, mut b) = (base(3, 30), base(3, 30));
        for k in 0..30u32 {
            let x = Vec3::new(k as f64, (k * k % 7) as f64, (k % 5) as f64 * 2.0);
            let ta = vec![TrackEntry::new(1, k), TrackEntry::new(2, k)];
            a.add_point3d_with_id(k as u64 + 1, cumulus3d_core::Point3D { xyz: x, color: [0; 3], error: -1.0, track: ta }).unwrap();
            // B is observed in images 2 and 3, only image 2 is shared. Only the first 25.
            if k < 25 {
                let tb = vec![TrackEntry::new(2, k), TrackEntry::new(3, k)];
                let mut y = truth.transform_point(&x);
                if k == 0 {
                    y += Vec3::new(1000.0, 0.0, 0.0); // outlier
                }
                b.add_point3d_with_id(k as u64 + 100, cumulus3d_core::Point3D { xyz: y, color: [0; 3], error: -1.0, track: tb }).unwrap();
            }
        }
        let pairs = shared_point_correspondences(&a, &b, &Default::default());
        assert_eq!(pairs.len(), 25);
        assert!(pairs.iter().all(|(i, j)| *j == *i + 99));
        // Position range filter: image 2 = position 1. With range [2,3) there is no sharing.
        let none = shared_point_correspondences(&a, &b, &SharedPointOptions { position_range: Some(2..3), ..Default::default() });
        assert!(none.is_empty());
        let one = shared_point_correspondences(&a, &b, &SharedPointOptions { position_range: Some(1..2), ..Default::default() });
        assert_eq!(one.len(), 25);
        let opts = RobustUmeyamaOptions { min_threshold: 1e-6, ..Default::default() };
        let (r, _) = align_reconstructions(&a, &b, &Default::default(), &opts).unwrap();
        sim_close(&r.sim3, &truth, 1e-8);
        assert_eq!(r.num_inliers, 24);
        assert!(!r.inlier_mask[0]);
    }

    #[test]
    fn dedup_pairs() {
        // If the same point pair is observed repeatedly in two images, count it only once.
        let (mut a, mut b) = (base(2, 5), base(2, 5));
        for k in 0..5u32 {
            let t = vec![TrackEntry::new(1, k), TrackEntry::new(2, k)];
            let p = cumulus3d_core::Point3D { xyz: Vec3::new(k as f64, 1.0, 2.0), color: [0; 3], error: -1.0, track: t };
            a.add_point3d_with_id(k as u64 + 1, p.clone()).unwrap();
            b.add_point3d_with_id(k as u64 + 1, p).unwrap();
        }
        assert_eq!(shared_point_correspondences(&a, &b, &Default::default()).len(), 5);
    }
}
