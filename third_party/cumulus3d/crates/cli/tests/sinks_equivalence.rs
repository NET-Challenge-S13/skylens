/*
 * sinks_equivalence.rs
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

//! 기본 훅(sinks) 동일성: 합성 장면을 `run_stream` 으로 돌린 출력과, 그 timeline 으로 다시 만든 같은 이벤트 열을
//! 기본 훅에 흘려 만든 출력이 같은지 본다. 시각 값은 비교에서 빼고 문구·순서·파일 목록·manifest·점 수를 비교한다.
//!
//! CUDA 가 없으면 조밀화가 없으므로, 구역 점군 대신 저장된 희소 모델의 3D 점을 쓰고
//! 기존 파일 입력 후처리(`post::run`)로 기준 출력을 만든다.

mod common;
mod replay;

use common::{make_dataset, NPOS};
use cumulus3d_cli::events::{Command, Event, EventKind, ZoneRange};
use cumulus3d_cli::pipeline::{Pipeline, Reducer};
use cumulus3d_cli::post::{self, PostInput};
use cumulus3d_cli::sinks::{self, SinkOptions};
use cumulus3d_cli::stream::{regions, run_stream, Layout, StreamConfig, CAMS};
use cumulus3d_cli::util::Logger;
use cumulus3d_core::io::{read_ply, write_ply, PlyLayout, PointCloud};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const SPAN: usize = 4;
const OVL: usize = 1;

/// 받은 이벤트를 그대로 내보내는 리듀서(재생용).
struct Replay;
impl Reducer for Replay {
    type Input = Event;
    fn step(&mut self, e: Event) -> Vec<Event> {
        vec![e]
    }
    fn poll(&mut self) -> Vec<Event> {
        Vec::new()
    }
    fn command(&mut self, _: Command) -> Vec<Event> {
        Vec::new()
    }
}

/// 희소 모델의 3D 점 → 점군(조밀화가 없을 때 구역 점군 대신).
fn sparse_cloud(out: &Path, kind: &str, k: usize) -> PointCloud {
    let m = replay::model(out, &format!("{kind}_{k}"));
    let mut c = PointCloud::default();
    for (_, p) in m.points3d() {
        c.positions.push([p.xyz.x as f32, p.xyz.y as f32, p.xyz.z as f32]);
        c.normals.push([0.0, 0.0, 1.0]);
        c.colors.push(p.color);
    }
    c
}

fn zone_ply(dir: &Path, kind: &str, r: &ZoneRange) -> PathBuf {
    dir.join("full").join(kind).join(format!("{kind}_{:02}_pos{}-{}.ply", r.zone, r.lo, r.hi))
}

fn timeline_texts(dir: &Path) -> Vec<String> {
    std::fs::read_to_string(dir.join("timeline.txt")).unwrap().lines().map(|l| replay::parse_line(l).1).collect()
}

/// run.log 의 `[HH:MM:SS] 사건` 줄 중 사건 문구인 것.
fn stamped_events(dir: &Path, texts: &BTreeSet<String>) -> Vec<String> {
    std::fs::read_to_string(dir.join("run.log"))
        .unwrap()
        .lines()
        .filter_map(|l| {
            let b = l.as_bytes();
            (b.len() > 11 && b[0] == b'[' && b[9] == b']').then(|| l[11..].to_string())
        })
        .filter(|s| texts.contains(s))
        .collect()
}

/// run.log 의 후처리 요약 덩어리("== 구역별 시간" ~ "== 단계별 시간" 앞).
fn post_block(dir: &Path) -> Vec<String> {
    let s = std::fs::read_to_string(dir.join("run.log")).unwrap();
    s.lines().skip_while(|l| !l.starts_with("== 구역별 시간")).take_while(|l| !l.starts_with("== 단계별 시간")).map(String::from).collect()
}

/// 후처리 산출 파일(상대 경로, 스냅샷 이름의 시각 부분 제거, snapshots/timeline.txt 제외) → 점 수.
fn outputs(dir: &Path) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for sub in ["full", "aligned", "final_frame", "snapshots"] {
        let mut stack = vec![dir.join(sub)];
        while let Some(d) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                    continue;
                }
                let rel = p.strip_prefix(dir).unwrap().to_string_lossy().replace('\\', "/");
                if rel == "snapshots/timeline.txt" {
                    continue; // 따로 확인
                }
                let name = if rel.starts_with("snapshots/event_") {
                    let mut parts: Vec<&str> = rel.split('_').collect();
                    parts[2] = "T";
                    parts.join("_")
                } else {
                    rel
                };
                let n = if name.ends_with(".ply") { read_ply(&p).unwrap().len() } else { 0 };
                m.insert(name, n);
            }
        }
    }
    m
}

/// manifest.json 에서 시각 값 줄을 가린다.
fn manifest_masked(dir: &Path) -> String {
    let s = std::fs::read_to_string(dir.join("snapshots/manifest.json")).unwrap();
    s.lines()
        .map(|l| {
            let t = l.trim_start();
            let timed = [
                "\"arrived_s\"",
                "\"preview_ready_s\"",
                "\"preview_latency_s\"",
                "\"refined_pose_s\"",
                "\"refined_ready_s\"",
                "\"refined_latency_s\"",
                "\"time_s\"",
            ];
            if timed.iter().any(|k| t.starts_with(k)) {
                format!("{}: T", t.split(':').next().unwrap())
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn model_dirs(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir.join("work/models"))
        .map(|rd| rd.flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect())
        .unwrap_or_default()
}

#[test]
fn sinks_reproduce_stream_outputs() {
    let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("sinks_equivalence");
    let _ = std::fs::remove_dir_all(&base);
    let (src, old) = (base.join("src"), base.join("old"));
    make_dataset(&src);
    let dense = cumulus3d_cuda::is_available();
    let mut cfg = StreamConfig::new(src.clone(), old.clone());
    (cfg.stride, cfg.span, cfg.overlap) = (1, SPAN, OVL);
    (cfg.dense_max_image_size, cfg.save_models, cfg.echo, cfg.seed, cfg.no_dense) = (160, true, false, Some(7), !dense);
    run_stream(cfg).expect("run_stream");
    let layout = Layout::discover(&src, 1).unwrap();

    // 기준 후처리 출력: 조밀화가 있으면 그 실행 폴더, 없으면 희소 점군 + 파일 입력 후처리.
    let refpost = if dense {
        old.clone()
    } else {
        let d = base.join("ref");
        let mut pv_ply = BTreeMap::new();
        let mut rf_ply = BTreeMap::new();
        for (k, lo, hi) in regions(NPOS, SPAN, OVL) {
            let r = ZoneRange { zone: k, lo, hi };
            for (kind, paths) in [("preview", &mut pv_ply), ("refined", &mut rf_ply)] {
                let p = zone_ply(&d, kind, &r);
                std::fs::create_dir_all(p.parent().unwrap()).unwrap();
                write_ply(&p, &sparse_cloud(&old, kind, k), PlyLayout::XyzNormalRgb).unwrap();
                paths.insert(k, p);
            }
        }
        let load = |kind: &str| -> BTreeMap<usize, _> {
            pv_ply.keys().map(|k| (*k, (*replay::model(&old, &format!("{kind}_{k}"))).clone())).collect()
        };
        let (previews, refined) = (load("preview"), load("refined"));
        let tl = std::fs::read_to_string(old.join("timeline.txt")).unwrap();
        let abs: Vec<(f64, String)> = tl
            .lines()
            .map(|l| {
                let (t, s) = replay::parse_line(l);
                (t.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64(), s)
            })
            .collect();
        let t0 = abs[0].0;
        let log = Logger::new(Some(&d.join("run.log")), false).unwrap();
        let inp = PostInput {
            out: &d,
            span: SPAN,
            ovl: OVL,
            events: abs.into_iter().map(|(t, s)| (t - t0, s)).collect(),
            previews: &previews,
            refined: &refined,
            preview_ply: &pv_ply,
            refined_ply: &rf_ply,
            position: &|n: &str| layout.position(n),
        };
        post::run(&inp, &log).expect("post::run");
        d
    };
    let clouds = |kind: &str, r: &ZoneRange| -> Option<Arc<PointCloud>> {
        let p = zone_ply(&refpost, kind, r);
        p.exists().then(|| Arc::new(read_ply(&p).unwrap()))
    };
    let events = replay::events_from_run(&old, &layout, &CAMS, &clouds);
    assert!(events.iter().any(|e| e.kind() == EventKind::ZoneRefinedPose));

    // 1) 종류별 훅 목록을 동기로 직접 호출. 2) 파이프라인(비동기, on_any)으로.
    let opts = SinkOptions { echo: false, save_models: true };
    let new_sync = base.join("new_sync");
    let mut hooks = sinks::default_sinks(&new_sync, &opts).unwrap();
    for e in &events {
        for (k, h) in hooks.iter_mut() {
            if *k == e.kind() {
                h(e);
            }
        }
    }
    drop(hooks);
    let new_async = base.join("new_async");
    let mut p = sinks::attach(Pipeline::new(Replay), &new_async, &opts).unwrap();
    for e in events.iter().cloned() {
        p.push(e);
    }
    let (_, summary) = p.finish();
    assert_eq!(summary.hook_panics, 0);

    let old_tl = timeline_texts(&old);
    let texts: BTreeSet<String> = old_tl.iter().cloned().collect();
    let old_post = outputs(&refpost);
    assert!(old_post.keys().any(|k| k.starts_with("snapshots/event_")), "{old_post:?}");
    for new in [&new_sync, &new_async] {
        // timeline: 문구·순서.
        assert_eq!(timeline_texts(new), old_tl, "{}", new.display());
        // run.log: 사건 줄 순서, 후처리 요약.
        assert_eq!(stamped_events(new, &texts), stamped_events(&old, &texts));
        let pb = post_block(&refpost);
        assert!(pb.len() > 5, "{pb:?}");
        assert_eq!(post_block(new), pb);
        // 파일 목록 + 점 수.
        assert_eq!(outputs(new), old_post);
        // manifest(시각 값 제외).
        assert_eq!(manifest_masked(new), manifest_masked(&refpost));
        assert_eq!(model_dirs(new), model_dirs(&old));
        assert!(new.join("snapshots/timeline.txt").exists());
    }
}
