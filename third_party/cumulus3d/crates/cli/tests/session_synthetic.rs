/*
 * session_synthetic.rs
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

//! 상태 값 리듀서(`Session`, `step`/`poll`/`command`/`finish`)를 합성 장면(드론 3대 × 8위치)으로 시험한다.
//! 조밀화는 끈다(장치 없이 돌도록). 이벤트 순서·개수, 되감기(ResetFrom), 구역 무효화, 즉시 수신기, Pipeline 연결.

mod common;
use common::{make_dataset, NPOS};
use cumulus3d_cli::events::{Command, Event, EventKind};
use cumulus3d_cli::pipeline::Pipeline;
use cumulus3d_cli::session::{self, FrameSet, Input, Session, SessionConfig};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

const CAMS: [&str; 3] = ["camF", "camR", "camL"];

/// 시험들이 함께 쓰는 합성 입력 폴더(한 번만 만든다).
fn dataset() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("session_synth_src");
        let _ = std::fs::remove_dir_all(&d);
        make_dataset(&d);
        d
    })
}

fn config(total: Option<usize>) -> SessionConfig {
    let src = dataset();
    let mut c = SessionConfig::new(src.join("images"));
    c.gps = cumulus3d_core::io::read_gps_file(src.join("gps_ref.txt")).unwrap();
    c.span = 4;
    c.overlap = 1;
    c.total_positions = total;
    c.seed = Some(7);
    c.history = 16;
    c
}

fn frames(p: usize) -> FrameSet {
    FrameSet::new(CAMS.iter().map(|c| (c.to_string(), format!("{c}/{c}_{:04}.jpg", p * 3))))
}

fn count(evs: &[Event], k: EventKind) -> usize {
    evs.iter().filter(|e| e.kind() == k).count()
}

fn texts(evs: &[Event]) -> Vec<String> {
    evs.iter().filter_map(|e| e.timeline_text()).collect()
}

fn idx(tl: &[String], s: &str) -> usize {
    tl.iter().position(|e| e.starts_with(s)).unwrap_or_else(|| panic!("사건 없음: {s}\n{tl:#?}"))
}

/// 불변 값 방식으로 끝까지: step 마다 새 Session, 마지막에 finish.
fn run_all(total: Option<usize>) -> (Session, Vec<Event>) {
    let mut s = Session::new(config(total));
    let mut all = Vec::new();
    for p in 0..NPOS {
        let (s2, evs) = session::step(&s, frames(p));
        assert!(s2.failed().is_none(), "{:?}", s2.failed());
        s = s2;
        all.extend(evs);
    }
    let (s2, evs) = session::finish(&s);
    all.extend(evs);
    (s2, all)
}

#[test]
fn events_order_and_counts() {
    let (s, evs) = run_all(Some(NPOS));
    // 버전은 단조 증가, 빠짐 없음(배경 이벤트도 제자리).
    for (i, e) in evs.iter().enumerate() {
        assert_eq!(e.meta().version, i as u64 + 1, "{:?}", e.kind());
    }
    assert_eq!(evs[0].kind(), EventKind::Started);
    assert_eq!(evs.last().unwrap().kind(), EventKind::Finished);
    assert_eq!(count(&evs, EventKind::FrameIngested), NPOS);
    assert_eq!(count(&evs, EventKind::FeaturesExtracted), 3 * NPOS);
    assert_eq!(count(&evs, EventKind::PairsMatched), NPOS);
    assert_eq!(count(&evs, EventKind::ModelInitialized), 1);
    // span 4 + overlap 1: 첫 모델은 위치 4, 그 뒤 위치마다 PositionDone.
    assert_eq!(count(&evs, EventKind::PositionDone), NPOS - 4);
    for k in [
        EventKind::ZoneArrived,
        EventKind::ZonePreview,
        EventKind::RefineStarted,
        EventKind::ZoneAdjusted,
        EventKind::ZoneRefinedPose,
        EventKind::ZoneRefined,
    ] {
        assert_eq!(count(&evs, k), 2, "{k:?}");
    }
    for k in [EventKind::AllPositionsDone, EventKind::AllRefinedDone, EventKind::Finished] {
        assert_eq!(count(&evs, k), 1, "{k:?}");
    }
    assert_eq!(count(&evs, EventKind::Error), 0);
    assert!(count(&evs, EventKind::FrameRegistered) >= 20);

    // timeline 문구·순서.
    let tl = texts(&evs);
    assert_eq!(tl[0], format!("start NPOS={NPOS}"));
    assert_eq!(idx(&tl, "init_model Registered images: "), 1);
    assert!(idx(&tl, "pos 4 registered ") < idx(&tl, "region 0 arrived pos[0,5)"));
    assert!(idx(&tl, "region 0 arrived pos[0,5)") < idx(&tl, "preview 0 ready"));
    assert!(idx(&tl, "refined 0 ba_start") < idx(&tl, "refined 0 pose_ready Registered images: "));
    assert!(idx(&tl, "refined 0 pose_ready") < idx(&tl, "refined 0 ready"));
    assert!(idx(&tl, "region 1 arrived pos[3,8)") > idx(&tl, "pos 7 registered "));
    assert!(idx(&tl, "preview 1 ready") < idx(&tl, "all positions done"));
    assert!(idx(&tl, "all positions done") < idx(&tl, "all refined done"));
    assert!(idx(&tl, "refined 1 ready") < idx(&tl, "all refined done"));
    assert!(tl.iter().any(|e| e.starts_with("refined 0 pose_ready") && e.contains("Mean reprojection error: ") && e.ends_with("px ")));

    // run.log 줄: 단계 시간·통계.
    let logs: Vec<&str> = evs.iter().filter_map(|e| if let Event::Log { line, .. } = e { Some(line.as_str()) } else { None }).collect();
    for pre in [
        "[time] feature pos 0",
        "[match] pos 7",
        "[mapper] ",
        "[register] pos 5",
        "[triangulate] pos 7",
        "[ba] refined 1",
        "[align] refined 0",
    ] {
        assert!(logs.iter().any(|l| l.starts_with(pre)), "로그 없음 {pre}");
    }
    // 정밀 정렬은 GPS 오차가 작다.
    let err: Vec<f64> = logs
        .iter()
        .filter(|l| l.starts_with("[align] refined"))
        .map(|l| l.split("오차 평균 ").nth(1).unwrap().split('m').next().unwrap().parse().unwrap())
        .collect();
    assert_eq!(err.len(), 2);
    assert!(err.iter().all(|e| *e < 0.2), "{err:?}");

    // 점군 이벤트: 조밀화를 껐으므로 자리표시, 모델은 들어 있다.
    for e in &evs {
        match e {
            Event::ZonePreview { dense, model, frame, range, .. } => {
                assert!(!dense);
                assert!(model.registered_image_count() > 0);
                if range.zone == 0 {
                    assert_eq!(frame, "gps");
                }
            }
            Event::ZoneRefined { dense, model, .. } => {
                assert!(!dense);
                assert!(model.registered_image_count() > 0);
            }
            Event::Finished { stage_times, chain_stats, .. } => {
                assert!(stage_times.iter().any(|(k, _, _)| k == "feature"));
                assert!(chain_stats.iter().any(|l| l.starts_with("Registered images: ")));
            }
            _ => {}
        }
    }
    // 상태 조회.
    assert_eq!(s.positions(), NPOS);
    assert!(s.is_finished());
    assert_eq!(s.previews().len(), 2);
    let rf = s.refined();
    assert_eq!(rf.len(), 2);
    assert!(rf.values().all(|z| z.done));
    assert!(s.model().unwrap().registered_image_count() >= 20);
    // 끝난 세션에 더 넣으면 경고만.
    let (_, evs) = session::step(&s, frames(0));
    assert_eq!(evs.iter().map(|e| e.kind()).collect::<Vec<_>>(), vec![EventKind::Warning]);
}

#[test]
fn unknown_total_closes_last_zone_at_finish() {
    let (_, evs) = run_all(None);
    let tl = texts(&evs);
    assert_eq!(tl[0], "start");
    // 구역 1 의 명목 끝(9)은 오지 않으므로 finish 에서 [3,8) 로 닫힌다.
    let a = idx(&tl, "region 1 arrived pos[3,8)");
    assert!(a > idx(&tl, "pos 7 registered "));
    assert!(a < idx(&tl, "all positions done"));
    assert_eq!(count(&evs, EventKind::ZoneRefined), 2);
}

#[test]
fn reset_from_rewinds_state() {
    let mut s = Session::new(config(Some(NPOS)));
    for p in 0..6 {
        s.apply(Input::Frames(frames(p)));
    }
    assert_eq!(s.positions(), 6);
    assert_eq!(s.arrived_zones().len(), 1);
    let before = s.store().num_images();
    assert_eq!(before, 18);

    // 위치 5 부터 다시: 저장소·체인은 위치 4 직후, 구역 0(위치 4 에서 닫힘)은 유지.
    let (mut s, evs) = session::command(&s, Command::ResetFrom(5));
    assert!(evs.iter().any(|e| matches!(e, Event::Reset { position: 5, .. })));
    assert_eq!(count(&evs, EventKind::ZoneInvalidated), 0);
    assert_eq!(s.positions(), 5);
    assert_eq!(s.store().num_images(), 15);
    assert!(s.position_of("camF/camF_0015.jpg").is_none());
    assert_eq!(s.arrived_zones().len(), 1);

    // 다시 넣으면 끝까지 정상.
    let mut all = Vec::new();
    for p in 5..NPOS {
        all.extend(s.apply(Input::Frames(frames(p))));
    }
    let unfinished = s.clone();
    all.extend(s.apply(Input::Finish));
    assert!(s.failed().is_none());
    let tl = texts(&all);
    for p in 5..NPOS {
        idx(&tl, &format!("pos {p} registered "));
    }
    assert!(idx(&tl, "region 1 arrived pos[3,8)") < idx(&tl, "all positions done"));
    assert_eq!(s.store().num_images(), 3 * NPOS);
    assert!(s.model().unwrap().registered_image_count() >= 20);
    assert_eq!(s.refined().len(), 2);

    // 첫 모델 이전으로 되감으면 구역 0 도 무효화되고 모델이 없다.
    // (끝난 세션은 입력을 받지 않으므로 끝내기 전 값에서.)
    let (s2, evs) = session::command(&unfinished, Command::ResetFrom(3));
    assert!(evs.iter().any(|e| matches!(e, Event::ZoneInvalidated { zone: 0, .. })));
    assert!(evs.iter().any(|e| matches!(e, Event::ZoneInvalidated { zone: 1, .. })));
    assert!(s2.model().is_none());
    assert_eq!(s2.positions(), 3);
    assert!(s2.arrived_zones().is_empty());
    assert!(s2.refined().is_empty());
}

#[test]
fn reset_without_history_is_rejected() {
    let mut cfg = config(Some(NPOS));
    cfg.history = 0;
    let mut s = Session::new(cfg);
    for p in 0..3 {
        s.apply(Input::Frames(frames(p)));
    }
    let evs = s.apply(Input::Command(Command::ResetFrom(2)));
    assert_eq!(evs.iter().map(|e| e.kind()).collect::<Vec<_>>(), vec![EventKind::Warning]);
    assert_eq!(s.positions(), 3);
    // 위치 0 으로는 되감기 지점 없이도 된다.
    let evs = s.apply(Input::Command(Command::ResetFrom(0)));
    assert!(evs.iter().any(|e| e.kind() == EventKind::Reset));
    assert_eq!(s.positions(), 0);
    assert_eq!(s.store().num_images(), 0);
    s.apply(Input::Frames(frames(0)));
    assert_eq!(s.store().num_images(), 3);
    assert!(s.failed().is_none());
}

#[test]
fn invalidate_zone_rebuilds_it() {
    let mut s = Session::new(config(Some(NPOS)));
    for p in 0..5 {
        s.apply(Input::Frames(frames(p)));
    }
    let mut after = s.apply(Input::Command(Command::InvalidateZone(0)));
    let evs = after.clone();
    let kinds: Vec<EventKind> = evs.iter().map(|e| e.kind()).filter(|k| *k != EventKind::Log).collect();
    let i = kinds.iter().position(|k| *k == EventKind::ZoneInvalidated).unwrap();
    assert!(kinds[i..].contains(&EventKind::ZoneArrived));
    assert!(kinds[i..].contains(&EventKind::ZonePreview));
    // 아직 오지 않은 구역은 경고만.
    let evs = s.apply(Input::Command(Command::InvalidateZone(1)));
    assert_eq!(evs.iter().map(|e| e.kind()).filter(|k| *k != EventKind::Log).collect::<Vec<_>>(), vec![EventKind::Warning]);
    after.extend(s.apply(Input::Finish));
    // 무효화 뒤에는 (무효화 전 정밀 작업이 아직 돌고 있었어도 그 이벤트는 버려지고) 다시 만든 구역 0 의 정밀본 하나만 나온다.
    let z0 = after.iter().filter(|e| matches!(e, Event::ZoneRefined { range, .. } if range.zone == 0)).count();
    assert_eq!(z0, 1);
    // finish 가 남은 구역 1 을 [3,5) 로 닫는다.
    assert!(after.iter().any(|e| matches!(e, Event::ZoneArrived { range, .. } if range.zone == 1 && range.hi == 5)));
    assert_eq!(s.refined().len(), 2);
    assert!(after.iter().any(|e| e.kind() == EventKind::Finished));
}

#[test]
fn sink_receives_everything_in_version_order() {
    let s = Session::new(config(Some(NPOS)));
    let got: Arc<Mutex<Vec<Event>>> = Arc::default();
    let g = got.clone();
    s.set_sink(Some(Arc::new(move |e| g.lock().unwrap().push(e))));
    let mut s = s;
    let mut returned = Vec::new();
    for p in 0..NPOS {
        returned.extend(s.apply(Input::Frames(frames(p))));
    }
    returned.extend(s.apply(Input::Finish));
    let got = got.lock().unwrap();
    let v: Vec<u64> = got.iter().map(|e| e.meta().version).collect();
    assert_eq!(v, (1..=v.len() as u64).collect::<Vec<_>>());
    assert_eq!(got.last().unwrap().kind(), EventKind::Finished);
    // 반환 벡터는 전경 이벤트(이미 전달된 사본); 배경 정밀 이벤트는 수신기로만 간다.
    assert!(returned.len() < got.len());
    assert!(returned.iter().all(|r| got.iter().any(|g| g.meta().version == r.meta().version && g.kind() == r.kind())));
    assert!(!returned.iter().any(|e| e.kind() == EventKind::ZoneRefined));
    assert_eq!(got.iter().filter(|e| e.kind() == EventKind::ZoneRefined).count(), 2);
}

#[test]
fn drives_pipeline() {
    let s = Session::new(config(Some(NPOS)));
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let sn = seen.clone();
    let mut pl = Pipeline::new(s).sync(true).on_any(move |e: &Event| {
        if let Some(t) = e.timeline_text() {
            sn.lock().unwrap().push(t);
        }
    });
    let rx = pl.subscribe();
    for p in 0..NPOS {
        pl.push(frames(p).into());
    }
    let (s, _) = pl.finish();
    assert!(s.is_finished());
    let tl = seen.lock().unwrap().clone();
    assert_eq!(tl[0], format!("start NPOS={NPOS}"));
    assert_eq!(tl.last().unwrap(), "all refined done");
    let n = rx.iter().count();
    assert_eq!(n as u64, s.version());
}
