/*
 * undistort.rs
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

//! 왜곡 보정 검증: 카메라 계산, LUT 기하 정합, 희소 모델 변환, 작업 폴더 출력.

use cumulus3d_core::interop::read_model;
use cumulus3d_core::{Camera, CameraModelKind, Image, Reconstruction, Rigid3, TrackEntry, Vec2, Vec3};
use cumulus3d_dense::{undistort, write_undistorted_workspace, DenseScene, ImageBuffer, SceneOptions, UndistortCache, UndistortOptions};
use std::sync::Arc;

fn cam() -> Camera {
    Camera::new(1, CameraModelKind::OpenCv, 640, 360, vec![503.0, 502.5, 320.0, 180.0, 0.0824, -0.0849, 0.0022, 0.0007]).unwrap()
}

/// 정규화 좌표의 매끄러운 무늬(정합 확인용).
fn pattern(u: f64, v: f64) -> f64 {
    128.0 + 60.0 * (u * 9.0).sin() + 50.0 * (v * 7.0).cos()
}

fn build() -> (Reconstruction, Arc<ImageBuffer>) {
    let c = cam();
    let mut rec = Reconstruction::new();
    rec.add_camera_own_rig(c.clone()).unwrap();
    let poses = [Rigid3::identity(), Rigid3::from_params(&[1.0, 0.0, 0.05, 0.0, -0.5, 0.0, 0.0])];
    let pts: Vec<Vec3> =
        (0..60).map(|k| Vec3::new(((k % 10) as f64 - 4.5) * 0.9, ((k / 10) as f64 - 2.5) * 0.7, 6.0 + (k % 7) as f64 * 0.3)).collect();
    let mut obs = [Vec::new(), Vec::new()];
    for (ii, pose) in poses.iter().enumerate() {
        for p in &pts {
            obs[ii].push(c.cam_to_img(&pose.transform_point(p)).unwrap());
        }
        let im = Image::new(ii as u32 + 1, format!("sub/img{ii}.png"), 1, obs[ii].clone());
        rec.add_image_own_frame(im, Some(*pose)).unwrap();
        rec.register_image(ii as u32 + 1).unwrap();
    }
    for (k, p) in pts.iter().enumerate() {
        rec.add_point3d(*p, vec![TrackEntry::new(1, k as u32), TrackEntry::new(2, k as u32)], [10, 20, 30]).unwrap();
    }
    // 입력 영상: 픽셀 중심을 역왜곡한 정규화 좌표의 무늬.
    let mut img = ImageBuffer::new(640, 360, 1);
    for y in 0..360 {
        for x in 0..640 {
            let uv = c.img_to_normalized(&Vec2::new(x as f64 + 0.5, y as f64 + 0.5)).unwrap();
            img.data[y * 640 + x] = pattern(uv.x, uv.y).round() as u8;
        }
    }
    (rec, Arc::new(img))
}

#[test]
fn undistort_model_images_and_folder() {
    let (rec, img) = build();
    let opts = UndistortOptions { max_image_size: 320, ..UndistortOptions::default() };
    let cache = UndistortCache::new();
    let res = undistort(&rec, &opts, &cache, |_| Some(img.clone())).unwrap();
    assert!(res.failed.is_empty());
    assert_eq!(cache.len(), 1);
    let out = &res.reconstruction;
    let pc = out.camera(1).unwrap();
    assert_eq!(pc.model, CameraModelKind::Pinhole);
    assert_eq!(pc.width.max(pc.height), 320);

    // 2D 점 = 입력 카메라 역투영 → PINHOLE 투영, 자세·3D 점 그대로.
    let src = cam();
    let mut err_sum = 0.0;
    let mut n = 0;
    for iid in [1u32, 2] {
        assert_eq!(out.world_to_cam(iid).unwrap(), rec.world_to_cam(iid).unwrap());
        let (a, b) = (rec.image(iid).unwrap(), out.image(iid).unwrap());
        for (p0, p1) in a.points2d().iter().zip(b.points2d()) {
            let uv = src.img_to_normalized(&p0.xy).unwrap();
            let e = pc.normalized_to_img(&uv);
            assert!((e - p1.xy).norm() < 1e-9);
            assert_eq!(p0.point3d_id, p1.point3d_id);
            // 자기검증: 3D 점을 새 카메라로 투영한 위치와 변환 관측이 맞아야 한다.
            let x = rec.point3d(p0.point3d_id).unwrap().xyz;
            let proj = pc.cam_to_img(&out.world_to_cam(iid).unwrap().transform_point(&x)).unwrap();
            err_sum += (proj - p1.xy).norm();
            n += 1;
        }
    }
    assert!(err_sum / (n as f64) < 1e-6);
    for (id, p) in rec.points3d() {
        let q = out.point3d(id).unwrap();
        assert_eq!(p.xyz, q.xyz);
        assert_eq!(p.color, q.color);
        assert_eq!(p.track, q.track);
    }

    // 영상 기하 정합: 보정 영상 픽셀 (x+0.5, y+0.5) 의 PINHOLE 정규화 좌표 무늬와 비교(가장자리 2px 제외).
    let u = &res.images[&1];
    assert_eq!((u.width as u64, u.height as u64), (pc.width, pc.height));
    let mut diff = Vec::new();
    for y in 2..u.height - 2 {
        for x in 2..u.width - 2 {
            let uv = pc.img_to_normalized(&Vec2::new(x as f64 + 0.5, y as f64 + 0.5)).unwrap();
            diff.push((u.get(x, y, 0) as f64 - pattern(uv.x, uv.y)).abs());
        }
    }
    diff.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = diff[diff.len() / 2];
    eprintln!("보정 영상 무늬 오차 중앙값 {med:.3}, 95% {:.3}", diff[diff.len() * 95 / 100]);
    assert!(med < 1.5, "{med}");

    // 폴더 출력과 다시 읽기 → 조밀화 장면 변환.
    let dir = std::env::temp_dir().join(format!("cumulus3d_undist_{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    write_undistorted_workspace(&res, &dir, &opts).unwrap();
    for f in [
        "images/sub/img0.png",
        "sparse/cameras.bin",
        "sparse/images.bin",
        "sparse/points3D.bin",
        "stereo/fusion.cfg",
        "stereo/patch-match.cfg",
    ] {
        assert!(dir.join(f).exists(), "{f}");
    }
    for d in ["depth_maps/sub", "normal_maps/sub", "consistency_graphs/sub"] {
        assert!(dir.join("stereo").join(d).is_dir());
    }
    let pm = std::fs::read_to_string(dir.join("stereo/patch-match.cfg")).unwrap();
    assert_eq!(pm.lines().count(), 4);
    assert_eq!(pm.lines().nth(1).unwrap(), "__auto__, 20");
    let back = read_model(dir.join("sparse")).unwrap();
    assert_eq!(back.camera(1).unwrap(), pc);
    let scene = DenseScene::from_workspace_dir(&dir, &SceneOptions::default()).unwrap();
    assert_eq!(scene.views.len(), 2);
    // 장면은 정수 중심 규약: 주점 −0.5.
    assert!((scene.views[0].k[2] - (pc.params[2] - 0.5)).abs() < 1e-12);
    assert!((scene.views[0].k[3] - (pc.params[3] - 0.5)).abs() < 1e-12);
    assert!(!scene.points.is_empty() && scene.points.iter().all(|p| p.views.len() == 2));
    std::fs::remove_dir_all(&dir).ok();
}

/// 매개변수가 같은 카메라 여럿(드론 3대 초기값): 캐시를 공유해도 카메라 id 가 유지돼야 한다.
#[test]
fn identical_cameras_keep_ids() {
    let (rec0, img) = build();
    let mut rec = Reconstruction::new();
    for id in 1..=2u32 {
        let mut c = cam();
        c.camera_id = id;
        rec.add_camera_own_rig(c).unwrap();
    }
    for im in rec0.images() {
        let cam_id = im.image_id; // 영상 1 → 카메라 1, 영상 2 → 카메라 2
        let ni = Image::new(im.image_id, im.name.clone(), cam_id, im.points2d().iter().map(|p| p.xy));
        rec.add_image_own_frame(ni, rec0.world_to_cam(im.image_id)).unwrap();
        rec.register_image(im.image_id).unwrap();
    }
    let opts = UndistortOptions { max_image_size: 320, ..UndistortOptions::default() };
    let cache = UndistortCache::new();
    let res = undistort(&rec, &opts, &cache, |_| Some(img.clone())).unwrap();
    assert_eq!(cache.len(), 1);
    for id in 1..=2u32 {
        assert_eq!(res.reconstruction.camera(id).unwrap().camera_id, id);
    }
}
