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

//! 합성 장면: 드론 편대(위치마다 OPENCV 카메라 3대, 롤 −25°/0°/+25°)가 지면 점을 내려다본다.
#![allow(dead_code)]

use cumulus3d_core::{
    Camera, CameraModelKind, Image, ImageId, Mat3, Point3DId, Quat, Reconstruction, Rigid3, Sim3, TrackEntry, Vec2, Vec3,
};
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;
use std::collections::BTreeMap;

pub struct SceneOpts {
    pub num_positions: usize,
    pub num_points: usize,
    pub spacing: f64,
    pub altitude: f64,
    pub noise_px: f64,
    pub min_track: usize,
    /// 영상마다 3D 점 없는 여분 2D 점 수.
    pub spare_points: usize,
    pub seed: u64,
}

impl Default for SceneOpts {
    fn default() -> Self {
        Self { num_positions: 12, num_points: 4000, spacing: 8.0, altitude: 50.0, noise_px: 0.5, min_track: 2, spare_points: 4, seed: 1 }
    }
}

pub struct Scene {
    /// 잡음 관측, 정답 자세·점·카메라를 가진 재구성.
    pub truth: Reconstruction,
    pub width: u64,
    pub height: u64,
}

pub const W: u64 = 1000;
pub const H: u64 = 750;

pub fn make_cameras() -> Vec<Camera> {
    let p = [
        [800.0, 805.0, 500.5, 374.0, -0.08, 0.02, 0.001, -0.0005],
        [790.0, 792.0, 498.0, 377.0, -0.05, 0.01, -0.0008, 0.0004],
        [812.0, 810.0, 503.0, 371.5, -0.10, 0.03, 0.0005, 0.0007],
    ];
    (0..3).map(|i| Camera::new(i as u32 + 1, CameraModelKind::OpenCv, W, H, p[i].to_vec()).unwrap()).collect()
}

fn rot_x(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    Mat3::new(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)
}

pub fn make_scene(o: &SceneOpts) -> Scene {
    let mut rng = Pcg64::seed_from_u64(o.seed);
    let cams = make_cameras();
    // 나디르: 카메라 x = 세계 x, y = −세계 y, z = −세계 z.
    let r0 = Mat3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);
    let rolls = [-25f64.to_radians(), 0.0, 25f64.to_radians()];
    let mut poses: Vec<(u32, Rigid3)> = Vec::new();
    for p in 0..o.num_positions {
        for (k, roll) in rolls.iter().enumerate() {
            let c = Vec3::new(
                p as f64 * o.spacing + rng.random_range(-0.5..0.5),
                (k as f64 - 1.0) * 3.0 + rng.random_range(-0.5..0.5),
                o.altitude + rng.random_range(-1.0..1.0),
            );
            let jitter = Quat::from_rotation_vector(&Vec3::new(
                rng.random_range(-0.03..0.03),
                rng.random_range(-0.03..0.03),
                rng.random_range(-0.03..0.03),
            ))
            .to_rotation_matrix();
            let r = jitter * rot_x(*roll) * r0;
            poses.push((k as u32 + 1, Rigid3::from_rotation_matrix(&r, -(r * c))));
        }
    }
    let len = (o.num_positions as f64 - 1.0) * o.spacing;
    let half_w = o.altitude * 0.9;
    let mut points: Vec<Vec3> = Vec::new();
    let mut tracks: Vec<Vec<(usize, Vec2)>> = Vec::new();
    let mut tries = 0;
    while points.len() < o.num_points && tries < o.num_points * 50 {
        tries += 1;
        let x = rng.random_range(-15.0..len + 15.0);
        let y = rng.random_range(-half_w..half_w);
        let z = 2.0 * (0.05 * x).sin() * (0.07 * y).cos() + rng.random_range(-1.5..1.5);
        let xw = Vec3::new(x, y, z);
        let mut tr = Vec::new();
        for (i, (cam, pose)) in poses.iter().enumerate() {
            let xc = pose.transform_point(&xw);
            if xc.z < 1.0 {
                continue;
            }
            let Some(px) = cams[*cam as usize - 1].cam_to_img(&xc) else { continue };
            if px.x < 5.0 || px.y < 5.0 || px.x > W as f64 - 5.0 || px.y > H as f64 - 5.0 {
                continue;
            }
            let n = Vec2::new(gauss(&mut rng) * o.noise_px, gauss(&mut rng) * o.noise_px);
            tr.push((i, px + n));
        }
        if tr.len() >= o.min_track {
            points.push(xw);
            tracks.push(tr);
        }
    }
    // 재구성 조립
    let mut per_image: Vec<Vec<Vec2>> = vec![Vec::new(); poses.len()];
    let mut elems: Vec<Vec<TrackEntry>> = Vec::new();
    for tr in &tracks {
        let mut t = Vec::new();
        for (i, xy) in tr {
            t.push(TrackEntry::new(*i as u32 + 1, per_image[*i].len() as u32));
            per_image[*i].push(*xy);
        }
        elems.push(t);
    }
    for pts in per_image.iter_mut() {
        for _ in 0..o.spare_points {
            pts.push(Vec2::new(rng.random_range(10.0..990.0), rng.random_range(10.0..740.0)));
        }
    }
    let mut rec = Reconstruction::new();
    for c in cams {
        rec.add_camera_own_rig(c).unwrap();
    }
    for (i, (cam, pose)) in poses.iter().enumerate() {
        let id = i as u32 + 1;
        let im = Image::new(id, format!("img_{id:04}.jpg"), *cam, per_image[i].clone());
        rec.add_image_own_frame(im, Some(*pose)).unwrap();
        rec.register_image(id).unwrap();
    }
    for (x, t) in points.iter().zip(elems) {
        rec.add_point3d(*x, t, [128, 128, 128]).unwrap();
    }
    Scene { truth: rec, width: W, height: H }
}

pub fn gauss(rng: &mut Pcg64) -> f64 {
    // 박스–뮬러.
    let u1: f64 = rng.random_range(1e-12..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

pub struct Perturb {
    pub rot_deg: f64,
    pub pos_m: f64,
    pub point_m: f64,
    pub focal_frac: f64,
}

impl Default for Perturb {
    fn default() -> Self {
        Self { rot_deg: 0.5, pos_m: 0.5, point_m: 0.5, focal_frac: 0.02 }
    }
}

fn rand_unit(rng: &mut Pcg64) -> Vec3 {
    Vec3::new(gauss(rng), gauss(rng), gauss(rng)).normalize()
}

/// 초기값 섭동: 자세 회전·중심, 점, 초점.
pub fn perturb(rec: &Reconstruction, p: &Perturb, seed: u64) -> Reconstruction {
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut out = rec.clone();
    for id in rec.registered_images() {
        let pose = rec.world_to_cam(id).unwrap();
        let dr = Quat::from_rotation_vector(&(rand_unit(&mut rng) * p.rot_deg.to_radians()));
        let c = pose.center() + rand_unit(&mut rng) * p.pos_m;
        let q = dr.hamilton(&pose.rotation).normalized();
        let r = q.to_rotation_matrix();
        out.set_world_to_cam(id, Rigid3::new(q, -(r * c))).unwrap();
    }
    for pid in rec.point3d_ids() {
        let x = rec.point3d(pid).unwrap().xyz + rand_unit(&mut rng) * p.point_m;
        out.set_point3d_xyz(pid, x).unwrap();
    }
    let cids: Vec<u32> = rec.cameras().keys().copied().collect();
    for c in cids {
        let cam = out.camera_mut(c).unwrap();
        let s = 1.0 + p.focal_frac * if rng.random_range(0.0..1.0) < 0.5 { -1.0 } else { 1.0 };
        cam.params[0] *= s;
        cam.params[1] *= s;
    }
    out
}

/// 좌표축별 RMS 재투영 오차(픽셀).
pub fn rms_per_coord(rec: &Reconstruction) -> f64 {
    let mut s = 0.0;
    let mut n = 0usize;
    for (_, p) in rec.points3d() {
        for e in &p.track {
            let pose = rec.world_to_cam(e.image_id).unwrap();
            let im = rec.image(e.image_id).unwrap();
            let cam = rec.camera(im.camera_id).unwrap();
            let xy = im.point2d(e.point2d_idx).xy;
            s += Reconstruction::squared_reprojection_error(&pose, cam, &xy, &p.xyz);
            n += 1;
        }
    }
    (s / (2.0 * n as f64)).sqrt()
}

/// 소스 → 타깃 상사 변환(움에야마, 스케일 포함).
pub fn umeyama(src: &[Vec3], dst: &[Vec3]) -> Sim3 {
    let n = src.len() as f64;
    let mx = src.iter().fold(Vec3::zeros(), |a, b| a + b) / n;
    let my = dst.iter().fold(Vec3::zeros(), |a, b| a + b) / n;
    let mut sxy = Mat3::zeros();
    let mut sx2 = 0.0;
    for (x, y) in src.iter().zip(dst) {
        let dx = x - mx;
        sxy += (y - my) * dx.transpose();
        sx2 += dx.norm_squared();
    }
    sxy /= n;
    sx2 /= n;
    let svd = sxy.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut s = Mat3::identity();
    if u.determinant() * vt.determinant() < 0.0 {
        s[(2, 2)] = -1.0;
    }
    let r = u * s * vt;
    let d = svd.singular_values;
    let scale = (d[0] * s[(0, 0)] + d[1] * s[(1, 1)] + d[2] * s[(2, 2)]) / sx2;
    let t = my - scale * r * mx;
    Sim3::new(scale, Quat::from_rotation_matrix(&r), t)
}

/// 추정 → 정답 상사 정렬 후 (최대 회전 오차 도, RMS 중심 오차 m).
pub fn pose_errors(est: &Reconstruction, truth: &Reconstruction) -> (f64, f64) {
    // 중심만으로는 비행 방향 축 회전이 약하게 정해지므로 3D 점도 함께 정렬에 쓴다.
    let ids = truth.registered_images();
    let mut src: Vec<Vec3> = ids.iter().map(|&i| est.projection_center(i).unwrap()).collect();
    let mut dst: Vec<Vec3> = ids.iter().map(|&i| truth.projection_center(i).unwrap()).collect();
    for (pid, p) in truth.points3d() {
        if let Some(q) = est.point3d(pid) {
            src.push(q.xyz);
            dst.push(p.xyz);
        }
    }
    let sim = umeyama(&src, &dst);
    let mut max_r: f64 = 0.0;
    let mut sc = 0.0;
    for (k, &i) in ids.iter().enumerate() {
        let pe = sim.transform_pose(&est.world_to_cam(i).unwrap());
        let pt = truth.world_to_cam(i).unwrap();
        max_r = max_r.max(pe.rotation.angular_distance(&pt.rotation).to_degrees());
        sc += (sim.transform_point(&src[k]) - dst[k]).norm_squared();
    }
    (max_r, (sc / ids.len() as f64).sqrt())
}

pub fn point_ids(rec: &Reconstruction) -> Vec<Point3DId> {
    rec.point3d_ids()
}

pub fn image_poses(rec: &Reconstruction) -> BTreeMap<ImageId, Rigid3> {
    rec.registered_images().into_iter().map(|i| (i, rec.world_to_cam(i).unwrap())).collect()
}
