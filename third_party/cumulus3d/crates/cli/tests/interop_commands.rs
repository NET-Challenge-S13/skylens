/*
 * interop_commands.rs
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

//! 단계별 호환 하위 명령을 스크립트 순서대로 이어 돌린다(FeatureStore 파일 + 모델 폴더로 상태 전달).

mod common;
use common::make_dataset;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_cumulus3d")).args(args).output().expect("실행");
    let so = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(out.status.success(), "{args:?} 실패:\n{so}\n{}", String::from_utf8_lossy(&out.stderr));
    so
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn interop_chain() {
    let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("interop_chain");
    let _ = std::fs::remove_dir_all(&base);
    let src = base.join("src");
    make_dataset(&src);
    let imgs = src.join("images");
    let db = base.join("db.c3dfs");
    let list = base.join("list.txt");
    // 위치 0–4: 첫 위치는 폴더당 카메라, 이후 기존 카메라 id(스크립트와 같은 호출 방식).
    std::fs::write(&list, "camF/camF_0000.jpg\ncamR/camR_0000.jpg\ncamL/camL_0000.jpg\n").unwrap();
    run(&[
        "feature_extractor",
        "--database_path",
        s(&db),
        "--image_path",
        s(&imgs),
        "--image_list_path",
        s(&list),
        "--ImageReader.single_camera_per_folder",
        "1",
        "--ImageReader.camera_model",
        "OPENCV",
        "--FeatureExtraction.use_gpu",
        "1",
        "--SiftExtraction.max_num_features",
        "8192",
    ]);
    let cam_of = |c: &str| match c {
        "camF" => "1",
        "camL" => "2",
        _ => "3",
    };
    for p in 1..5 {
        for c in ["camF", "camR", "camL"] {
            std::fs::write(&list, format!("{c}/{c}_{:04}.jpg\n", p * 3)).unwrap();
            run(&[
                "feature_extractor",
                "--database_path",
                s(&db),
                "--image_path",
                s(&imgs),
                "--image_list_path",
                s(&list),
                "--ImageReader.existing_camera_id",
                cam_of(c),
                "--ImageReader.camera_model",
                "OPENCV",
            ]);
        }
    }
    let mut pairs = String::new();
    let n = |c: &str, p: usize| format!("{c}/{c}_{:04}.jpg", p * 3);
    for p in 0..5 {
        for a in ["camF", "camR", "camL"] {
            for q in 0..p {
                pairs += &format!("{} {}\n", n(a, q), n(a, p));
            }
            for b in ["camF", "camR", "camL"] {
                if a < b {
                    pairs += &format!("{} {}\n", n(a, p), n(b, p));
                }
            }
        }
    }
    let pl = base.join("pairs.txt");
    std::fs::write(&pl, pairs).unwrap();
    run(&["matches_importer", "--database_path", s(&db), "--match_list_path", s(&pl), "--match_type", "pairs"]);
    let sg = base.join("sg0");
    run(&[
        "global_mapper",
        "--database_path",
        s(&db),
        "--image_path",
        s(&imgs),
        "--output_path",
        s(&sg),
        "--GlobalMapper.ba_num_iterations",
        "0",
        "--GlobalMapper.skip_retriangulation",
        "1",
        "--GlobalMapper.keep_max_num_tracks",
        "100000",
    ]);
    let m0 = sg.join("0");
    let a = run(&["model_analyzer", "--path", s(&m0)]);
    assert!(a.contains("Registered images: 15"), "{a}");
    let tri = base.join("tri");
    run(&[
        "point_triangulator",
        "--database_path",
        s(&db),
        "--image_path",
        s(&imgs),
        "--input_path",
        s(&m0),
        "--output_path",
        s(&tri),
        "--clear_points",
        "0",
    ]);
    let ba = base.join("ba");
    run(&["bundle_adjuster", "--input_path", s(&tri), "--output_path", s(&ba)]);
    let al = base.join("al");
    run(&[
        "model_aligner",
        "--input_path",
        s(&ba),
        "--output_path",
        s(&al),
        "--ref_images_path",
        s(&src.join("gps_ref.txt")),
        "--ref_is_gps",
        "1",
        "--alignment_type",
        "enu",
        "--alignment_max_error",
        "3",
    ]);
    let a = run(&["model_analyzer", "--path", s(&al), "--verbose", "1"]);
    assert!(a.contains("Mean reprojection error: ") && a.contains("Camera Id: 1, Model Name: OPENCV"), "{a}");
    let del = base.join("del.txt");
    std::fs::write(&del, "camF/camF_0000.jpg\ncamR/camR_0000.jpg\n").unwrap();
    let dl = base.join("deleted");
    run(&["image_deleter", "--input_path", s(&al), "--output_path", s(&dl), "--image_names_path", s(&del)]);
    assert!(run(&["model_analyzer", "--path", s(&dl)]).contains("Registered images: 13"));
    let ud = base.join("undist");
    run(&["image_undistorter", "--image_path", s(&imgs), "--input_path", s(&dl), "--output_path", s(&ud), "--max_image_size", "160"]);
    assert!(ud.join("sparse/cameras.bin").exists() && ud.join("images/camL/camL_0006.jpg").exists());
    let ply = base.join("dense.ply");
    if cumulus3d_cuda::is_available() {
        run(&["densify", "-i", s(&ud), "-o", s(&ply), "--number-views", "8"]);
        assert!(cumulus3d_core::io::read_ply(&ply).unwrap().len() > 1000);
    } else {
        // 장치가 없으면 조밀화는 명확한 오류로 끝나야 한다.
        let out = Command::new(env!("CARGO_BIN_EXE_cumulus3d")).args(["densify", "-i", s(&ud), "-o", s(&ply)]).output().unwrap();
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stderr).contains("CUDA"));
    }
    let txt = base.join("txt");
    run(&["model_converter", "--input_path", s(&al), "--output_path", s(&txt), "--output_type", "TXT"]);
    assert!(std::fs::read_to_string(txt.join("images.txt")).unwrap().contains("camR/camR_0012.jpg"));
    // 정렬된 모델에 새 위치 등록: image_registrator 가 저장소의 나머지 영상을 붙인다.
    for p in 5..6 {
        for c in ["camF", "camR", "camL"] {
            std::fs::write(&list, format!("{c}/{c}_{:04}.jpg\n", p * 3)).unwrap();
            run(&[
                "feature_extractor",
                "--database_path",
                s(&db),
                "--image_path",
                s(&imgs),
                "--image_list_path",
                s(&list),
                "--ImageReader.existing_camera_id",
                cam_of(c),
                "--ImageReader.camera_model",
                "OPENCV",
            ]);
        }
    }
    let mut pairs = String::new();
    for a in ["camF", "camR", "camL"] {
        for q in 1..5 {
            pairs += &format!("{} {}\n", n(a, q), n(a, 5));
        }
    }
    std::fs::write(&pl, pairs).unwrap();
    run(&["matches_importer", "--database_path", s(&db), "--match_list_path", s(&pl), "--match_type", "pairs"]);
    let reg = base.join("reg");
    run(&["image_registrator", "--database_path", s(&db), "--input_path", s(&al), "--output_path", s(&reg)]);
    assert!(run(&["model_analyzer", "--path", s(&reg)]).contains("Registered images: 18"));
}
