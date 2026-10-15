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

//! 기존 `cumulus3d stream` 출력 폴더(timeline.txt + work/models + 구역 점군)에서 이벤트 열을 다시 만든다.
//! timeline 한 줄 = 이벤트 하나(시각 = 그 줄의 epoch 초.나노). 문구가 없는 이벤트(`FrameIngested`, `ZoneAdjusted`)는
//! 입력 배치·저장된 모델로 채운다.
#![allow(dead_code)]

use cumulus3d_cli::events::{Event, Meta, ZoneRange};
use cumulus3d_cli::stream::Layout;
use cumulus3d_core::interop::read_model;
use cumulus3d_core::io::PointCloud;
use cumulus3d_core::Reconstruction;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// `secs.nanos 문구` → (시각, 문구).
pub fn parse_line(l: &str) -> (SystemTime, String) {
    let (t, s) = l.split_once(' ').expect("시각 문구");
    let (a, b) = t.split_once('.').expect("초.나노");
    assert_eq!(b.len(), 9, "나노 자릿수: {l}");
    (UNIX_EPOCH + Duration::new(a.parse().unwrap(), b.parse().unwrap()), s.to_string())
}

fn num(s: &str) -> usize {
    s.parse().unwrap_or_else(|_| panic!("수 아님: {s}"))
}

/// 구역 점군 공급자: (종류 "preview"|"refined", 구역) → 점군(없으면 조밀화 없음).
pub type Clouds<'a> = &'a dyn Fn(&str, &ZoneRange) -> Option<Arc<PointCloud>>;

/// 저장된 모델(없으면 빈 모델).
pub fn model(out: &Path, name: &str) -> Arc<Reconstruction> {
    Arc::new(read_model(out.join("work/models").join(name)).unwrap_or_default())
}

/// timeline 문구 하나를 이벤트로. `ranges` 는 앞서 본 구역 범위(구역 번호로 찾음).
fn to_event(meta: Meta, s: &str, out: &Path, ranges: &mut Vec<ZoneRange>, clouds: Clouds) -> Vec<Event> {
    let w: Vec<&str> = s.split(' ').collect();
    let range = |ranges: &Vec<ZoneRange>, k: usize| *ranges.iter().find(|r| r.zone == k).expect("앞선 region arrived");
    let e = match w.as_slice() {
        ["start", n] => Event::Started { meta, positions: n.strip_prefix("NPOS=").map(num) },
        ["init_model", "Registered", "images:", n] => {
            Event::ModelInitialized { meta, registered: num(n), position: 0, model: Arc::new(Reconstruction::new()) }
        }
        ["adopt", "refined", k, "as", "base"] => Event::BaseAdopted { meta, zone: num(k) },
        ["pos", p, "registered", re] => {
            let (r, e) = re.split_once('/').unwrap();
            Event::PositionDone { meta, position: num(p), registered: num(r), expected: num(e) }
        }
        ["region", k, "arrived", pr] => {
            let (lo, hi) = pr.strip_prefix("pos[").and_then(|x| x.strip_suffix(')')).and_then(|x| x.split_once(',')).unwrap();
            let r = ZoneRange { zone: num(k), lo: num(lo), hi: num(hi) };
            ranges.push(r);
            Event::ZoneArrived { meta, range: r, model: model(out, &format!("snap_{k}")) }
        }
        ["preview", k, "ready"] => {
            let r = range(ranges, num(k));
            let c = clouds("preview", &r);
            Event::ZonePreview {
                meta,
                range: r,
                dense: c.is_some(),
                cloud: c.unwrap_or_default(),
                model: model(out, &format!("preview_{k}")),
                frame: String::new(),
            }
        }
        ["refined", k, "ba_start"] => Event::RefineStarted { meta, zone: num(k) },
        ["refined", k, "pose_ready", "Registered", "images:", n, "Mean", "reprojection", "error:", x, ""] => {
            let k = num(k);
            let mut v = Vec::new();
            if out.join("work/models").join(format!("ba_{k}")).exists() {
                v.push(Event::ZoneAdjusted { meta: meta.clone(), zone: k, model: model(out, &format!("ba_{k}")) });
            }
            v.push(Event::ZoneRefinedPose {
                meta,
                range: range(ranges, k),
                registered: num(n),
                mean_reproj_px: x.strip_suffix("px").unwrap().parse().unwrap(),
                model: model(out, &format!("refined_{k}")),
            });
            return v;
        }
        ["refined", k, "ready"] => {
            let r = range(ranges, num(k));
            let c = clouds("refined", &r);
            Event::ZoneRefined {
                meta,
                range: r,
                dense: c.is_some(),
                cloud: c.unwrap_or_default(),
                model: model(out, &format!("refined_{k}")),
            }
        }
        ["all", "positions", "done"] => Event::AllPositionsDone { meta },
        ["all", "refined", "done"] => {
            let chain = out.join("work/models/chain").exists().then(|| model(out, "chain"));
            Event::AllRefinedDone { meta, chain }
        }
        _ => panic!("모르는 사건 문구: {s:?}"),
    };
    vec![e]
}

/// 출력 폴더의 timeline.txt 로 이벤트 열을 만든다. `Started` 바로 뒤에 위치별 `FrameIngested` 를 넣는다.
pub fn events_from_run(out: &Path, layout: &Layout, cams: &[&str], clouds: Clouds) -> Vec<Event> {
    let tl = std::fs::read_to_string(out.join("timeline.txt")).unwrap();
    let mut ranges = Vec::new();
    let mut evs = Vec::new();
    for (i, l) in tl.lines().enumerate() {
        let (at, s) = parse_line(l);
        let meta = Meta { version: i as u64, at };
        evs.extend(to_event(meta.clone(), &s, out, &mut ranges, clouds));
        if i == 0 {
            for p in 0..layout.frames.len() {
                let images = cams.iter().map(|c| layout.name(c, p)).collect();
                evs.push(Event::FrameIngested { meta: meta.clone(), position: p, images });
            }
        }
    }
    evs
}
