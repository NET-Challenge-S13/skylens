/*
 * synthetic.rs
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

//! 합성 장면 생성기(테스트·벤치용): 드론 3대 직선 비행, 위치마다 3장, OPENCV 카메라,
//! 지면 근처 점, 2D 잡음, 파이프라인 짝 규칙(같은 카메라 간격 1..5,8,16 / 다른 카메라 위치 차 0..4).

use crate::math::{gaussian, so3_exp};
use cumulus3d_core::{
    Camera, CameraModelKind, FeatureMatch, FeatureStore, ImageId, Keypoint, Rigid3, TwoViewGeometry, TwoViewGeometryConfig, Vec2, Vec3,
};
use rand::{RngExt, SeedableRng};
use std::collections::{BTreeMap, HashMap};

/// 장면 설정.
#[derive(Clone, Debug)]
pub struct SceneConfig {
    pub num_positions: usize,
    pub spacing: f64,
    pub altitude: f64,
    pub num_points: usize,
    pub noise_px: f64,
    /// 매칭 중 무작위 오대응 비율.
    pub outlier_ratio: f64,
    pub seed: u64,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self { num_positions: 16, spacing: 2.0, altitude: 30.0, num_points: 4000, noise_px: 0.5, outlier_ratio: 0.01, seed: 0 }
    }
}

/// 생성된 장면.
pub struct Scene {
    pub store: FeatureStore,
    pub truth: BTreeMap<ImageId, Rigid3>,
    pub points: Vec<Vec3>,
    /// (영상, 키포인트) → 점 번호.
    pub kp_point: HashMap<(ImageId, u32), usize>,
    /// 위치별 영상 id (카메라 순).
    pub images_by_position: Vec<Vec<ImageId>>,
    pub image_names: BTreeMap<ImageId, String>,
}

/// 카메라 3대(전방·우·좌) 의 OPENCV 내부값.
pub fn opencv_camera(id: u32) -> Camera {
    let mut c = Camera::new(id, CameraModelKind::OpenCv, 1920, 1080, vec![1500.0, 1500.0, 960.0, 540.0, -0.1, 0.02, 0.001, 0.001])
        .expect("파라미터 수");
    c.focal_from_prior = true;
    c
}

/// 아래를 보는 기본 자세(카메라 z = 세계 −Z, x = 세계 +X)에 롤·피치 기울기.
fn drone_pose(center: Vec3, roll_deg: f64, pitch_deg: f64) -> Rigid3 {
    // cam_to_world: 열 = 카메라 축의 세계 표현.
    let base = cumulus3d_core::Mat3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);
    let tilt = so3_exp(&Vec3::new(pitch_deg.to_radians(), roll_deg.to_radians(), 0.0));
    let cam_to_world = base * tilt;
    let r = cam_to_world.transpose();
    Rigid3::from_rotation_matrix(&r, -(r * center))
}

/// 장면 생성. 영상 id 는 위치 → 카메라 순으로 1 부터.
pub fn generate(cfg: &SceneConfig) -> Scene {
    let mut rng = rand_pcg::Pcg64::seed_from_u64(cfg.seed);
    let store = FeatureStore::new();
    let cams: Vec<Camera> = (1..=3).map(opencv_camera).collect();
    for c in &cams {
        store.add_camera(c.clone()).expect("카메라");
    }
    let len = cfg.num_positions as f64 * cfg.spacing;
    let points: Vec<Vec3> = (0..cfg.num_points)
        .map(|_| Vec3::new(rng.random_range(-25.0..len + 25.0), rng.random_range(-30.0..30.0), rng.random_range(0.0..10.0)))
        .collect();
    // 카메라별 (y 오프셋, 롤, 피치).
    let rigs = [(0.0, 0.0, 20.0), (8.0, 25.0, 5.0), (-8.0, -25.0, 5.0)];
    let mut truth = BTreeMap::new();
    let mut images_by_position = Vec::new();
    let mut image_names = BTreeMap::new();
    let mut kp_point = HashMap::new();
    let mut obs_of: BTreeMap<ImageId, HashMap<usize, u32>> = BTreeMap::new();
    for pos in 0..cfg.num_positions {
        let mut ids = Vec::new();
        for (k, (dy, roll, pitch)) in rigs.iter().enumerate() {
            let jitter = Vec3::new(gaussian(&mut rng), gaussian(&mut rng), gaussian(&mut rng)) * 0.2;
            let c = Vec3::new(pos as f64 * cfg.spacing, *dy, cfg.altitude) + jitter;
            let pose = drone_pose(c, roll + gaussian(&mut rng), pitch + gaussian(&mut rng));
            let name = format!("cam{}/{:04}.jpg", k, pos);
            let id = store.add_image(&name, cams[k].camera_id).expect("영상");
            let mut kps = Vec::new();
            let mut map = HashMap::new();
            for (j, x) in points.iter().enumerate() {
                let Some(xy) = cams[k].cam_to_img(&(pose * *x)) else { continue };
                if xy.x < 1.0 || xy.y < 1.0 || xy.x > 1919.0 || xy.y > 1079.0 {
                    continue;
                }
                let n = Vec2::new(gaussian(&mut rng), gaussian(&mut rng)) * cfg.noise_px;
                let p = xy + n;
                map.insert(j, kps.len() as u32);
                kp_point.insert((id, kps.len() as u32), j);
                kps.push(Keypoint::new(p.x as f32, p.y as f32));
            }
            store.set_keypoints(id, kps);
            truth.insert(id, pose);
            image_names.insert(id, name);
            obs_of.insert(id, map);
            ids.push(id);
        }
        images_by_position.push(ids);
    }
    // 짝 규칙.
    let mut pairs = Vec::new();
    for p in 0..cfg.num_positions {
        for k in 0..3 {
            for g in [1usize, 2, 3, 4, 5, 8, 16] {
                if p + g < cfg.num_positions {
                    pairs.push((images_by_position[p][k], images_by_position[p + g][k]));
                }
            }
            for k2 in 0..3 {
                if k2 == k {
                    continue;
                }
                for g in 0..=4usize {
                    if p + g < cfg.num_positions && (g > 0 || k < k2) {
                        pairs.push((images_by_position[p][k], images_by_position[p + g][k2]));
                    }
                }
            }
        }
    }
    for (a, b) in pairs {
        let (ma, mb) = (&obs_of[&a], &obs_of[&b]);
        let mut matches: Vec<FeatureMatch> = ma.iter().filter_map(|(j, ia)| mb.get(j).map(|ib| FeatureMatch::new(*ia, *ib))).collect();
        matches.sort_by_key(|m| m.idx1);
        let n_out = (matches.len() as f64 * cfg.outlier_ratio) as usize;
        let (na, nb) = (ma.len() as u32, mb.len() as u32);
        for _ in 0..n_out {
            if na == 0 || nb == 0 {
                break;
            }
            matches.push(FeatureMatch::new(rng.random_range(0..na), rng.random_range(0..nb)));
        }
        if matches.len() < 15 {
            continue;
        }
        // 기하 검증이 저장했을 E (참 자세에서).
        let rel = truth[&b] * truth[&a].inverse();
        let e = cumulus3d_core::geometry::skew(&rel.translation.normalize()) * rel.rotation_matrix();
        let tvg = TwoViewGeometry {
            config: TwoViewGeometryConfig::Calibrated,
            e: Some(e),
            f: None,
            h: None,
            cam1_to_cam2: None,
            inlier_matches: matches,
            tri_angle: None,
        };
        store.put_two_view(a, b, &tvg).expect("기하");
    }
    Scene { store, truth, points, kp_point, images_by_position, image_names }
}

impl Scene {
    /// 위치 0..n 영상 이름 집합(대응 그래프 이름 필터용).
    pub fn names_up_to(&self, n: usize) -> std::collections::HashSet<String> {
        self.images_by_position[..n.min(self.images_by_position.len())].iter().flatten().map(|id| self.image_names[id].clone()).collect()
    }
}
