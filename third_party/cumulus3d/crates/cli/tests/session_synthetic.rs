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

//! Test the state-value reducer (`Session`, `step`/`poll`/`command`/`finish`) with a synthetic scene (3 drones x 8 positions).
//! Densification is off (so it runs without a device). Event order and count, rewinding (ResetFrom), zone invalidation, immediate
//! receiver, Pipeline hookup.

mod common;
use common::{make_dataset, NPOS};
use cumulus3d_cli::events::{Command, Event, EventKind};
use cumulus3d_cli::pipeline::Pipeline;
use cumulus3d_cli::session::{self, FrameSet, Input, Session, SessionConfig};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

const CAMS: [&str; 3] = ["camF", "camR", "camL"];

/// Synthetic input folder shared by the tests (created only once).
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

/// All the way through in immutable-value style: a new Session at every step, finish at the end.
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
    // Versions increase monotonically with no gaps (background events too, in place).
    for (i, e) in evs.iter().enumerate() {
        assert_eq!(e.meta().version, i as u64 + 1, "{:?}", e.kind());
    }
    assert_eq!(evs[0].kind(), EventKind::Started);
    assert_eq!(evs.last().unwrap().kind(), EventKind::Finished);
    assert_eq!(count(&evs, EventKind::FrameIngested), NPOS);
    assert_eq!(count(&evs, EventKind::FeaturesExtracted), 3 * NPOS);
    assert_eq!(count(&evs, EventKind::PairsMatched), NPOS);
    assert_eq!(count(&evs, EventKind::ModelInitialized), 1);
    // span 4 + overlap 1: the first model is at position 4, then PositionDone at every position.
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

    // Timeline text and order.
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

    // run.log lines: stage times and statistics.
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
    // Refined alignment has a small GPS error.
    let err: Vec<f64> = logs
        .iter()
        .filter(|l| l.starts_with("[align] refined"))
        .map(|l| l.split("오차 평균 ").nth(1).unwrap().split('m').next().unwrap().parse().unwrap())
        .collect();
    assert_eq!(err.len(), 2);
    assert!(err.iter().all(|e| *e < 0.2), "{err:?}");

    // Point-cloud events: densification is off, so placeholders; the models are included.
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
    // State queries.
    assert_eq!(s.positions(), NPOS);
    assert!(s.is_finished());
    assert_eq!(s.previews().len(), 2);
    let rf = s.refined();
    assert_eq!(rf.len(), 2);
    assert!(rf.values().all(|z| z.done));
    assert!(s.model().unwrap().registered_image_count() >= 20);
    // Feeding more into a finished session only gives a warning.
    let (_, evs) = session::step(&s, frames(0));
    assert_eq!(evs.iter().map(|e| e.kind()).collect::<Vec<_>>(), vec![EventKind::Warning]);
}

#[test]
fn unknown_total_closes_last_zone_at_finish() {
    let (_, evs) = run_all(None);
    let tl = texts(&evs);
    assert_eq!(tl[0], "start");
    // The nominal end (9) of zone 1 never arrives, so it is closed as [3,8) in finish.
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

    // Restart from position 5: the store and chain are right after position 4, and zone 0 (closed at position 4) is kept.
    let (mut s, evs) = session::command(&s, Command::ResetFrom(5));
    assert!(evs.iter().any(|e| matches!(e, Event::Reset { position: 5, .. })));
    assert_eq!(count(&evs, EventKind::ZoneInvalidated), 0);
    assert_eq!(s.positions(), 5);
    assert_eq!(s.store().num_images(), 15);
    assert!(s.position_of("camF/camF_0015.jpg").is_none());
    assert_eq!(s.arrived_zones().len(), 1);

    // Feeding again works normally to the end.
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

    // Rewinding to before the first model also invalidates zone 0, and there is no model.
    // (A finished session accepts no input, so use the value from before finishing.)
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
    // Rewinding to position 0 works even without a rewind point.
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
    // A zone that has not arrived yet only gives a warning.
    let evs = s.apply(Input::Command(Command::InvalidateZone(1)));
    assert_eq!(evs.iter().map(|e| e.kind()).filter(|k| *k != EventKind::Log).collect::<Vec<_>>(), vec![EventKind::Warning]);
    after.extend(s.apply(Input::Finish));
    // After invalidation (even if a refinement job from before the invalidation was still running, its events are dropped), only the
    // refined version of the rebuilt zone 0 comes out.
    let z0 = after.iter().filter(|e| matches!(e, Event::ZoneRefined { range, .. } if range.zone == 0)).count();
    assert_eq!(z0, 1);
    // finish closes the remaining zone 1 as [3,5).
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
    // The return vector holds the foreground events (copies of what was already delivered); background refinement events go only to the
    // receiver.
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

/// Runs a stable-display session over the synthetic flight with span 3 / overlap 1 (zones 0, 1, 2).
fn run_stable(tol: f64, max_zone: usize) -> (Session, Vec<Event>) {
    let mut c = config(Some(NPOS));
    c.span = 3;
    c.stable_display = true;
    c.settle_tol = tol;
    c.settle_max_zone = max_zone;
    let mut s = Session::new(c);
    let mut all = Vec::new();
    for p in 0..NPOS {
        let (s2, evs) = session::step(&s, frames(p));
        assert!(s2.failed().is_none(), "{:?}", s2.failed());
        s = s2;
        all.extend(evs);
    }
    let (s, evs) = session::finish(&s);
    all.extend(evs);
    (s, all)
}

fn logs(evs: &[Event]) -> Vec<&str> {
    evs.iter().filter_map(|e| if let Event::Log { line, .. } = e { Some(line.as_str()) } else { None }).collect()
}

#[test]
fn stable_display_freezes_shown_cameras_after_settling() {
    // A huge tolerance settles at zone 1 (zone 0 has nothing to compare with); zone 2 is then anchored.
    let (s, all) = run_stable(1e9, 3);
    assert_eq!(count(&all, EventKind::ZonePreview), 0);
    let refined = s.refined();
    assert!(refined.len() >= 3, "zones {:?}", refined.keys().collect::<Vec<_>>());
    let l = logs(&all);
    assert!(l.iter().any(|x| x.starts_with("[settle] refined 1:") && x.ends_with("→ 고정")), "{l:#?}");
    assert!(!l.iter().any(|x| x.starts_with("[settle] refined 2:")));
    // The model at the settle point (zone 1) is what every later zone keeps: poses and intrinsics never change.
    let at_settle = &refined[&1].model;
    for (k, z) in refined.iter().filter(|(k, _)| **k >= 2) {
        let mut shared = 0;
        for id in at_settle.registered_images() {
            let name = &at_settle.image(id).unwrap().name;
            let Some(cur) = z.model.image_by_name(name) else { continue };
            let (a, b) = (at_settle.world_to_cam(id).unwrap(), z.model.world_to_cam(cur.image_id).unwrap());
            assert!((a.center() - b.center()).norm() < 1e-9, "zone {k}: {name} moved");
            shared += 1;
        }
        assert!(shared > 0, "zone {k} shares no camera with the settled model");
        assert_eq!(cumulus3d_cli::session::intrinsics_change(at_settle.cameras(), z.model.cameras()), 0.0);
    }
    // The final model handed to sinks is aligned to GPS (ENU), not the display frame.
    let chain = all.iter().find_map(|e| if let Event::AllRefinedDone { chain: Some(c), .. } = e { Some(c) } else { None }).expect("chain");
    assert!(logs(&all).iter().any(|x| x.starts_with("[align] final (stable display):")));
    let gps = &config(Some(NPOS)).gps;
    let o = &gps[0];
    let frame = cumulus3d_align::EnuFrame::new(o.lat, o.lon, o.alt);
    let (mut sum, mut n) = (0.0, 0);
    for id in chain.registered_images() {
        let Some(g) = gps.iter().find(|g| g.name == chain.image(id).unwrap().name) else { continue };
        sum += (chain.projection_center(id).unwrap() - frame.lla_to_enu(g.lat, g.lon, g.alt)).norm();
        n += 1;
    }
    assert!(n > 0 && sum / (n as f64) < 1.0, "final chain not in ENU: mean error {} over {n}", sum / (n as f64));
}

#[test]
fn stable_display_forced_settle_replaces_earlier_zones_then_stops() {
    // Tolerance so small it never settles by itself; forced at zone 2. The synthetic run has no dense stage, so the
    // re-placement shows up as repeated ZoneRefinedPose / ZoneRefined events (dense = false) for the earlier zones.
    let (s, all) = run_stable(1e-12, 2);
    let refined = s.refined();
    assert!(refined.len() >= 3, "zones {:?}", refined.keys().collect::<Vec<_>>());
    let l = logs(&all);
    assert!(l.iter().any(|x| x.starts_with("[settle] refined 1:") && x.ends_with("→ 정착 중")), "{l:#?}");
    assert!(l.iter().any(|x| x.starts_with("[settle] refined 2:") && x.ends_with("→ 강제 고정")), "{l:#?}");
    let n = |k: EventKind, z: usize| {
        all.iter()
            .filter(|e| e.kind() == k)
            .filter(|e| matches!(e, Event::ZoneRefined { range, .. } | Event::ZoneRefinedPose { range, .. } if range.zone == z))
            .count()
    };
    // Zone 0: first result + re-placed by zones 1 and 2. Zone 1: first + re-placed by zone 2. Zone 2: only its own.
    assert_eq!(n(EventKind::ZoneRefined, 0), 3);
    assert_eq!(n(EventKind::ZoneRefined, 1), 2);
    assert_eq!(n(EventKind::ZoneRefined, 2), 1);
    assert_eq!(n(EventKind::ZoneRefinedPose, 0), 3);
    // Nothing is re-emitted after the settle point: zones settled at 2 never touch zones 0..1 again.
    for z in refined.keys().filter(|z| **z >= 3) {
        assert_eq!(n(EventKind::ZoneRefined, *z), 1);
    }
    // Zone 0 and 1 now live in the final adjustment's model (their camera poses equal zone 2's).
    let last = &refined[&2].model;
    for j in [0, 1] {
        let m = &refined[&j].model;
        for id in m.registered_images().into_iter().take(5) {
            let name = &m.image(id).unwrap().name;
            let cur = last.image_by_name(name).unwrap();
            assert!((m.world_to_cam(id).unwrap().center() - last.world_to_cam(cur.image_id).unwrap().center()).norm() < 1e-9);
        }
    }
}

#[test]
fn intrinsics_prior_initializes_cameras_and_holds_the_focal_length() {
    // A prior 30 % away from the rendered focal length, very tight: the cameras start there (log) and the stable-display
    // adjustment of every zone stays within 1 % of it (the data alone would pull the focal length back towards the truth).
    let f0 = common::F * 1.3;
    let text = format!(
        r#"{{"cameras": {{"1": [{f0}, {f0}, 160.0, 120.0, 0, 0, 0, 0], "2": [{f0}, {f0}, 160.0, 120.0, 0, 0, 0, 0], "3": [{f0}, {f0}, 160.0, 120.0, 0, 0, 0, 0]}}}}"#
    );
    let mut c = config(Some(NPOS));
    c.span = 3;
    c.stable_display = true;
    c.intrinsics_prior = Some(Arc::new(cumulus3d_cli::intrinsics::IntrinsicsPrior::from_json(&text, 1e-4, 1e-3).unwrap()));
    let mut s = Session::new(c);
    let mut all = Vec::new();
    for p in 0..NPOS {
        let (s2, evs) = session::step(&s, frames(p));
        assert!(s2.failed().is_none(), "{:?}", s2.failed());
        s = s2;
        all.extend(evs);
    }
    let (s, evs) = session::finish(&s);
    all.extend(evs);
    let l = logs(&all);
    assert!(l.iter().any(|x| x.starts_with("[prior] 카메라 1:")), "{l:#?}");
    assert!(l.iter().any(|x| x.starts_with("[prior] refined 0:")), "{l:#?}");
    let refined = s.refined();
    assert!(!refined.is_empty());
    for z in refined.values() {
        for cam in z.model.cameras().values() {
            assert!((cam.params[0] / f0 - 1.0).abs() < 0.01, "fx {} vs prior {f0}", cam.params[0]);
        }
    }
}
