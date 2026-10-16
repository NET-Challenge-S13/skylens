/*
 * stream_synthetic.rs
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

//! 합성 장면(레이 캐스팅으로 그린 무늬 바닥 + 상자, 드론 3대 × 8위치)으로 `cumulus3d stream` 을 끝까지 돌리고
//! 출력 파일 구조·사건 기록을 확인한다.

mod common;
use common::{make_dataset, NPOS};
use std::path::PathBuf;
use std::process::Command;

#[test]
fn stream_end_to_end_synthetic() {
    if !cumulus3d_cuda::is_available() {
        eprintln!("CUDA 장치 없음: 조밀화가 필요한 스트림 시험을 건너뜀");
        return;
    }
    let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("stream_synth");
    let src = base.join("src");
    let out = base.join("out");
    let _ = std::fs::remove_dir_all(&base);
    make_dataset(&src);
    let status = Command::new(env!("CARGO_BIN_EXE_cumulus3d"))
        .args(["stream", "--src"])
        .arg(&src)
        .arg("--out")
        .arg(&out)
        .args(["--stride", "1", "--span", "4", "--overlap", "1", "--dense-max-image-size", "160", "--quiet", "--save-models"])
        .status()
        .expect("실행");
    assert!(status.success());

    // 사건 기록: 스크립트와 같은 문구·순서.
    let tl = std::fs::read_to_string(out.join("timeline.txt")).unwrap();
    let evs: Vec<String> = tl.lines().map(|l| l.split_once(' ').unwrap().1.to_string()).collect();
    for l in tl.lines() {
        let (t, _) = l.split_once(' ').unwrap();
        let (a, b) = t.split_once('.').unwrap();
        assert!(a.parse::<u64>().is_ok() && b.len() == 9, "시각 형식 {t}");
    }
    let idx = |s: &str| evs.iter().position(|e| e.starts_with(s)).unwrap_or_else(|| panic!("사건 없음: {s}\n{tl}"));
    assert_eq!(evs[0], format!("start NPOS={NPOS}"));
    assert_eq!(idx("init_model Registered images: "), 1);
    let init_n: usize = evs[1].rsplit(' ').next().unwrap().parse().unwrap();
    assert!(init_n >= 13, "초기 모델 등록 {init_n}/15");
    assert!(idx("pos 4 registered ") < idx("region 0 arrived pos[0,5)"));
    assert!(idx("region 0 arrived pos[0,5)") < idx("preview 0 ready"));
    assert!(idx("refined 0 ba_start") < idx("refined 0 pose_ready Registered images: "));
    assert!(idx("refined 0 pose_ready") < idx("refined 0 ready"));
    assert!(evs.iter().any(|e| e.starts_with("refined 0 pose_ready") && e.contains("Mean reprojection error: ") && e.ends_with("px ")));
    assert!(idx("region 1 arrived pos[3,8)") > idx("pos 7 registered "));
    assert!(idx("preview 1 ready") < idx("all positions done"));
    assert!(idx("all positions done") < idx("all refined done"));
    assert!(idx("refined 1 ready") < idx("all refined done"));
    let last_pos = &evs[idx("pos 7 registered ")];
    let reg: usize = last_pos.split(' ').nth(3).unwrap().split('/').next().unwrap().parse().unwrap();
    assert!(reg >= 20, "{last_pos}");

    // 출력 구조.
    for f in [
        "run.log",
        "DONE",
        "full/preview/preview_00_pos0-5.ply",
        "full/preview/preview_01_pos3-8.ply",
        "full/refined/refined_00_pos0-5.ply",
        "full/refined/refined_01_pos3-8.ply",
        "aligned/preview/preview_00_pos0-5.ply",
        "aligned/preview/preview_01_pos3-8.ply",
        "aligned/refined/refined_00_pos0-5.ply",
        "aligned/refined/refined_01_pos3-8.ply",
        "final_frame/refined/refined_00_pos0-5.ply",
        "final_frame/refined/refined_01_pos3-8.ply",
        "final_frame/preview/preview_01_pos3-8.ply",
        "snapshots/manifest.json",
        "snapshots/timeline.txt",
        "work/models/refined_0/images.bin",
        "work/models/preview_1/points3D.bin",
    ] {
        assert!(out.join(f).exists(), "없음: {f}");
    }
    let snaps: Vec<String> = std::fs::read_dir(out.join("snapshots"))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.starts_with("event_") && n.ends_with(".ply"))
        .collect();
    assert_eq!(snaps.len(), 4, "{snaps:?}");
    for n in &snaps {
        // event_NN_TTTT.Ts_kind_K.ply
        let parts: Vec<&str> = n.trim_end_matches(".ply").split('_').collect();
        assert_eq!(parts.len(), 5, "{n}");
        assert_eq!(parts[1].len(), 2);
        assert!(parts[2].ends_with('s') && parts[2].len() == 7, "{n}");
        assert!(parts[3] == "preview" || parts[3] == "refined");
    }
    let man = std::fs::read_to_string(out.join("snapshots/manifest.json")).unwrap();
    for key in [
        "\"timeline\"",
        "\"events\"",
        "\"align\"",
        "\"reanchor\"",
        "\"preview_latency_s\"",
        "\"points_1of6\"",
        "\"frame\"",
        "\"fit_median_m\"",
    ] {
        assert!(man.contains(key), "manifest 에 {key} 없음");
    }
    // 정밀·초벌 모두 GPS ENU 좌표계(첫 GPS 기록 camF_0000 = 원점, 고도 15 m): 바닥(하위 10%)이 z ≈ −15.
    for f in ["full/refined/refined_00_pos0-5.ply", "aligned/preview/preview_01_pos3-8.ply", "final_frame/preview/preview_00_pos0-5.ply"] {
        let cloud = cumulus3d_core::io::read_ply(out.join(f)).unwrap();
        assert!(cloud.len() > 1000, "{f}: 조밀 점 {}", cloud.len());
        let mut z: Vec<f32> = cloud.positions.iter().map(|p| p[2]).collect();
        z.sort_by(f32::total_cmp);
        let floor = z[z.len() / 10];
        assert!((floor + 15.0).abs() < 1.0, "{f}: 바닥 z {floor}");
        let mut y: Vec<f32> = cloud.positions.iter().map(|p| p[1]).collect();
        y.sort_by(f32::total_cmp);
        assert!(y[y.len() / 2].abs() < 2.0, "{f}: 북쪽 중앙 {}", y[y.len() / 2]);
    }
    let align = std::fs::read_to_string(out.join("run.log")).unwrap();
    let refined_err: Vec<f64> = align
        .lines()
        .filter(|l| l.starts_with("[align] refined"))
        .map(|l| l.split("오차 평균 ").nth(1).unwrap().split('m').next().unwrap().parse().unwrap())
        .collect();
    assert_eq!(refined_err.len(), 2);
    assert!(refined_err.iter().all(|e| *e < 0.2), "정밀 GPS 정렬 오차 {refined_err:?}");
    let log = std::fs::read_to_string(out.join("run.log")).unwrap();
    for s in ["== 구역별 시간(초, 시작 기준)", "== 정렬", "== 사건 순서", "정밀 0-1 겹침 차 중앙", "최종 좌표계 정밀 0-1", "== 단계별 시간"]
    {
        assert!(log.contains(s), "run.log 에 {s} 없음");
    }
}
