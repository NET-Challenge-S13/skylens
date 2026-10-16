/*
 * reader_camera.rs
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

//! 이미지 읽기·카메라 초기화 검증.

use cumulus3d_core::{CameraModelKind, FeatureStore, Keypoint};
use cumulus3d_features::*;

#[test]
fn gray_conversion_exact() {
    assert_eq!(rgb_to_gray(255, 0, 0), 54);
    assert_eq!(rgb_to_gray(0, 255, 0), 182);
    assert_eq!(rgb_to_gray(0, 0, 255), 18);
    assert_eq!(rgb_to_gray(10, 20, 30), 19);
    // DynamicImage 경로(RGB, RGBA 알파 버림)
    let rgb = image::RgbImage::from_raw(2, 1, vec![255, 0, 0, 10, 20, 30]).unwrap();
    let g = gray::to_gray(&image::DynamicImage::ImageRgb8(rgb)).unwrap();
    assert_eq!(g.data, vec![54, 19]);
    let rgba = image::RgbaImage::from_raw(1, 1, vec![0, 255, 0, 7]).unwrap();
    let g = gray::to_gray(&image::DynamicImage::ImageRgba8(rgba)).unwrap();
    assert_eq!(g.data, vec![182]);
}

#[test]
fn max_size_limit() {
    assert_eq!(limited_size(4000, 3000, 3200), (3200, 2400));
    assert_eq!(limited_size(5472, 3648, 3200), (3200, 2133));
    assert_eq!(limited_size(3200, 3200, 3200), (3200, 3200));
    assert_eq!(limited_size(1000, 800, 3200), (1000, 800));
}

#[test]
fn focal_rules() {
    let e = ExifInfo { focal_35mm: Some(24.0), ..Default::default() };
    let f = infer_focal(&e, 4000, 3000, 1.2);
    assert!((f.focal - 2773.284).abs() < 1e-3, "{}", f.focal);
    assert!(f.prior);
    let e = ExifInfo {
        focal_mm: Some(4.5),
        focal_plane_x_resolution: Some(1000.0),
        focal_plane_resolution_unit: Some(4),
        ..Default::default()
    };
    let f = infer_focal(&e, 4000, 3000, 1.2);
    assert!((f.focal - 4500.0).abs() < 1e-9);
    assert!(f.prior);
    let f = infer_focal(&ExifInfo::default(), 4000, 3000, 1.2);
    assert_eq!(f.focal, 4800.0);
    assert!(!f.prior);
    // 센서 폭 표
    let e = ExifInfo { focal_mm: Some(8.8), make: Some("DJI".into()), model: Some("FC6310".into()), ..Default::default() };
    let f = infer_focal(&e, 5472, 3648, 1.2);
    assert!((f.focal - 8.8 / 13.2 * 5472.0).abs() < 1e-9);
}

#[test]
fn opencv_init() {
    let cam = init_camera(CameraModelKind::OpenCv, 4000, 3000, &ExifInfo::default(), None, 1.2).unwrap();
    assert_eq!(cam.params, vec![4800.0, 4800.0, 2000.0, 1500.0, 0.0, 0.0, 0.0, 0.0]);
    assert!(!cam.focal_from_prior);
}

#[test]
fn keypoint_rescale() {
    let mut kp = Keypoint::from_scale_orientation(1600.5, 1200.25, 2.0, 0.3);
    kp.rescale(4000.0 / 3200.0, 3000.0 / 2400.0);
    assert!((kp.x - 2000.625).abs() < 1e-5);
    assert!((kp.y - 1500.3125).abs() < 1e-5);
    assert!((kp.scale() - 2.5).abs() < 1e-5);
    assert!((kp.orientation() - 0.3).abs() < 1e-5);
}

#[test]
fn rotation_roundtrip_and_image_consistency() {
    let kp = Keypoint { x: 12.3, y: 45.6, a11: 1.1, a12: -0.4, a21: 0.3, a22: 0.9 };
    let (w, h) = (100.0f32, 70.0f32);
    for k in 0..4u32 {
        let (rw, rh) = if k % 2 == 1 { (h, w) } else { (w, h) };
        let r = rotate_keypoint_ccw(&kp, k, w, h);
        let back = rotate_keypoint_ccw(&r, (4 - k) % 4, rw, rh);
        for (a, b) in kp.to_row().iter().zip(back.to_row().iter()) {
            assert!((a - b).abs() < 1e-5, "k={k}");
        }
    }
    // 영상 회전과 점 회전 규칙 일치: 화소 (x,y) 중심이 같은 값을 가리킴.
    let img = GrayImage::from_f32(7, 5, |x, y| ((x * 31 + y * 17) % 255) as f32 / 255.0);
    for k in 0..4u32 {
        let r = img.rotate_ccw(k);
        for y in 0..5 {
            for x in 0..7 {
                let p = rotate_keypoint_ccw(&Keypoint::new(x as f32 + 0.5, y as f32 + 0.5), k, 7.0, 5.0);
                let (nx, ny) = ((p.x - 0.5).round() as usize, (p.y - 0.5).round() as usize);
                assert_eq!(r.get(nx, ny), img.get(x, y));
            }
        }
    }
    assert_eq!(gray::orientation_to_rotation(Some(1)), 0);
    assert_eq!(gray::orientation_to_rotation(Some(3)), 2);
    assert_eq!(gray::orientation_to_rotation(Some(6)), 3);
    assert_eq!(gray::orientation_to_rotation(Some(8)), 1);
    assert_eq!(gray::orientation_to_rotation(None), 0);
}

fn textured(w: usize, h: usize, seed: u64) -> GrayImage {
    let mut s = seed;
    let mut noise = vec![0f32; w * h];
    for v in noise.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = (s >> 40) as f32 / (1u64 << 24) as f32;
    }
    let mut out = vec![0f32; w * h];
    let mut tmp = vec![0f32; w * h];
    sift::pyramid::gaussian_blur(&noise, &mut out, &mut tmp, w, h, 2.0);
    GrayImage::from_f32(w, h, |x, y| (out[y * w + x] - 0.5) * 3.0 + 0.5)
}

fn input(name: &str, w: usize, h: usize) -> ImageSource {
    ImageSource { name: name.into(), gray: textured(w, h, name.len() as u64 * 7 + 3), exif: ExifInfo::default() }
}

#[test]
fn camera_binding_scenario() {
    let store = FeatureStore::new();
    let ex = FeatureExtractor::new();
    let opts = ExtractionOptions::default();
    let r = ex
        .extract_inputs(&store, vec![input("camF/0.jpg", 96, 64), input("camR/0.jpg", 96, 64), input("camL/0.jpg", 96, 64)], &opts)
        .unwrap();
    assert_eq!(store.num_cameras(), 3);
    let cam_of = |n: &str| store.image_by_name(n).unwrap().camera_id;
    assert_eq!(cam_of("camF/0.jpg"), 1);
    assert_eq!(cam_of("camL/0.jpg"), 2);
    assert_eq!(cam_of("camR/0.jpg"), 3);
    assert_eq!(store.image_id_by_name("camF/0.jpg"), Some(1));
    assert_eq!(store.image_id_by_name("camL/0.jpg"), Some(2));
    for rep in &r {
        assert!(matches!(rep.status, ImageStatus::Extracted { .. }), "{rep:?}");
    }
    let cam = store.camera(1).unwrap();
    assert_eq!(cam.model, CameraModelKind::OpenCv);
    assert_eq!((cam.width, cam.height), (96, 64));

    // 위치 1: 카메라별 existing id
    let mut o2 = ExtractionOptions::default();
    for (n, c) in [("camF/1.jpg", 1), ("camR/1.jpg", 3), ("camL/1.jpg", 2)] {
        o2.reader.camera_mode = CameraMode::Existing(c);
        ex.extract_inputs(&store, vec![input(n, 96, 64)], &o2).unwrap();
        assert_eq!(cam_of(n), c);
    }
    assert_eq!(store.num_cameras(), 3);
    assert_eq!(store.image_id_by_name("camR/1.jpg"), Some(5));

    // 같은 이름 재실행 → 건너뜀
    let r = ex.extract_inputs(&store, vec![input("camR/1.jpg", 96, 64)], &o2).unwrap();
    assert!(matches!(r[0].status, ImageStatus::AlreadyExists { image_id: 5 }));
    assert_eq!(store.num_images(), 6);

    // 없는 카메라 → 즉시 오류
    o2.reader.camera_mode = CameraMode::Existing(42);
    assert!(ex.extract_inputs(&store, vec![input("x/1.jpg", 96, 64)], &o2).is_err());
}

#[test]
fn keypoints_scaled_to_camera_size() {
    // 축소된 영상에서 추출한 좌표가 입력 영상 크기로 환산되는지(최대 크기 64 로 강제 축소).
    let store = FeatureStore::new();
    let ex = FeatureExtractor::new();
    let mut opts = ExtractionOptions::default();
    opts.reader.max_image_size = 100;
    ex.extract_inputs(&store, vec![input("a/x.png", 200, 160)], &opts).unwrap();
    let kps = store.keypoints(1).unwrap();
    assert!(!kps.is_empty());
    let cam = store.camera(1).unwrap();
    assert_eq!((cam.width, cam.height), (200, 160));
    let max_x = kps.iter().map(|k| k.x).fold(0f32, f32::max);
    assert!(max_x > 100.0 && max_x <= 200.0, "{max_x}");
}

#[test]
fn read_files_from_disk() {
    let dir = std::env::temp_dir().join(format!("cumulus3d_features_test_{}", std::process::id()));
    std::fs::create_dir_all(dir.join("camA")).unwrap();
    let g = textured(120, 90, 5);
    let img = image::GrayImage::from_raw(120, 90, g.data.clone()).unwrap();
    img.save(dir.join("camA/p0.png")).unwrap();
    std::fs::write(dir.join("list.txt"), "  camA/p0.png \n\ncamA/missing.png\n").unwrap();
    let names = read_image_list(&dir.join("list.txt")).unwrap();
    assert_eq!(names, vec!["camA/p0.png", "camA/missing.png"]);
    let store = FeatureStore::new();
    let r = FeatureExtractor::new().extract_files(&store, &dir, &names, &ExtractionOptions::default()).unwrap();
    assert_eq!(r.len(), 2);
    // 사전순: missing < p0
    assert!(matches!(r[0].status, ImageStatus::Failed { .. }));
    assert!(matches!(r[1].status, ImageStatus::Extracted { .. }));
    assert_eq!(gray::read_gray(&dir.join("camA/p0.png")).unwrap(), g);
    let _ = std::fs::remove_dir_all(&dir);
}
