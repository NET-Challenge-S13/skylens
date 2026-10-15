/*
 * compose_synthetic.rs
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

//! Declarative live composition (`compose`) on the synthetic scene (3 cameras x 8 positions), densification off.
//! Append / duplicate / pending / gap, replay of corrections and late arrivals, zone invalidation, strict mode,
//! in-memory payloads, the spawned producer queue, and the host lookup.

mod common;
use common::{make_dataset, NPOS};
use cumulus3d_cli::compose::{
    reconstruct, Backpressure, ComposeError, CompositionHost, EncodedFormat, FrameKey, FrameState, IgnoreReason, LiveFrameSet, Outcome,
    ReconNode, ReconcilePolicy,
};
use cumulus3d_cli::events::{Event, EventKind};
use cumulus3d_core::io::{read_gps_file, GpsRecord};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

const CAMS: [&str; 3] = ["camF", "camR", "camL"];

fn dataset() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("compose_synth_src");
        let _ = std::fs::remove_dir_all(&d);
        make_dataset(&d);
        d
    })
}

/// Own copy of the image folder per test (spooling writes into it).
fn images(tag: &str) -> PathBuf {
    let dst = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("compose_{tag}"));
    let _ = std::fs::remove_dir_all(&dst);
    for cam in CAMS {
        let (s, d) = (dataset().join("images").join(cam), dst.join(cam));
        std::fs::create_dir_all(&d).unwrap();
        for e in std::fs::read_dir(s).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), d.join(e.file_name())).unwrap();
        }
    }
    dst
}

fn gps() -> &'static HashMap<String, GpsRecord> {
    static G: OnceLock<HashMap<String, GpsRecord>> = OnceLock::new();
    G.get_or_init(|| read_gps_file(dataset().join("gps_ref.txt")).unwrap().into_iter().map(|g| (g.name.clone(), g)).collect())
}

fn name(cam: &str, seq: u64) -> String {
    format!("{cam}/{cam}_{:04}.jpg", seq * 3)
}

/// File-backed set for sequence `seq` (position seq of the synthetic flight), GPS fixes named by camera.
fn set(seq: u64, rev: u32) -> LiveFrameSet {
    let mut s = LiveFrameSet::new(FrameKey::new("drone-1", seq, rev));
    for cam in CAMS {
        let g = &gps()[&name(cam, seq)];
        s = s.file(cam, name(cam, seq)).gps_fix(cam, g.lat, g.lon, g.alt);
    }
    s
}

fn node(host: &CompositionHost, tag: &str, policy: ReconcilePolicy) -> (ReconNode, Arc<Mutex<Vec<Event>>>) {
    let log = Arc::new(Mutex::new(Vec::new()));
    let l = Arc::clone(&log);
    let n = ReconNode::declare()
        .id(tag)
        .images(images(tag))
        .cameras(CAMS)
        .zones(4, 1)
        .seed(7)
        .reconcile(policy)
        .sync_hooks(true)
        .on_event(move |e| l.lock().unwrap().push(e.clone()))
        .build(host)
        .unwrap();
    (n, log)
}

fn count(evs: &[Event], k: EventKind) -> usize {
    evs.iter().filter(|e| e.kind() == k).count()
}

#[test]
fn append_duplicate_pending_gap_and_close() {
    let host = CompositionHost::new();
    let (n, log) = node(&host, "append", ReconcilePolicy::AppendOnly);
    assert_eq!(n.compose(FrameState::ready(set(0, 0))).unwrap(), Outcome::Appended { position: 0, skipped: 0 });
    assert_eq!(n.compose(FrameState::ready(set(0, 0))).unwrap(), Outcome::Duplicate { position: 0 });
    // Pending, then an incomplete ready set: both wait, nothing ingested.
    assert_eq!(
        n.compose(FrameState::pending(FrameKey::new("drone-1", 1, 0), ["camL"])).unwrap(),
        Outcome::Waiting { missing: vec!["camL".into()] }
    );
    let partial = LiveFrameSet::new(FrameKey::new("drone-1", 1, 0)).file("camF", name("camF", 1)).file("camR", name("camR", 1));
    assert_eq!(n.compose(FrameState::ready(partial)).unwrap(), Outcome::Waiting { missing: vec!["camL".into()] });
    assert_eq!(n.positions(), 1);
    assert_eq!(n.waiting(), vec![(1, vec!["camL".to_string()])]);
    assert_eq!(n.compose(FrameState::ready(set(1, 0))).unwrap(), Outcome::Appended { position: 1, skipped: 0 });
    assert!(n.waiting().is_empty());
    // Gap: 2 and 3 dropped.
    assert_eq!(n.compose(FrameState::ready(set(4, 0))).unwrap(), Outcome::Appended { position: 2, skipped: 2 });
    for s in 5..NPOS as u64 {
        assert!(matches!(n.compose(FrameState::ready(set(s, 0))).unwrap(), Outcome::Appended { .. }));
    }
    // Late arrival and correction are ignored by AppendOnly; foreign source too.
    assert_eq!(n.compose(FrameState::ready(set(2, 0))).unwrap(), Outcome::Ignored(IgnoreReason::Late));
    assert_eq!(n.compose(FrameState::ready(set(5, 1))).unwrap(), Outcome::Ignored(IgnoreReason::CorrectionNotApplied));
    let mut foreign = set(6, 0);
    foreign.key.source_id = "other".into();
    assert!(matches!(n.compose(FrameState::ready(foreign)).unwrap(), Outcome::Ignored(IgnoreReason::ForeignSource { .. })));
    // Malformed set is an error under any policy.
    let bad = LiveFrameSet::new(FrameKey::new("drone-1", 20, 0)).file("camX", "x.jpg");
    assert!(matches!(n.compose(FrameState::ready(bad)), Err(ComposeError::InvalidFrameSet(_))));

    assert_eq!(n.journal().iter().map(|j| j.0).collect::<Vec<_>>(), vec![0, 1, 4, 5, 6, 7]);
    // The same node through the host.
    assert_eq!(reconstruct(&host, "append", FrameState::ready(set(7, 0))).unwrap(), Outcome::Duplicate { position: 5 });
    assert!(matches!(reconstruct(&host, "nope", FrameState::End), Err(ComposeError::UnknownNode(_))));

    let closed = host.close("append").unwrap();
    assert_eq!(closed.session.positions(), 6);
    let st = closed.stats;
    assert_eq!((st.appended, st.duplicates, st.waiting, st.skipped_sequences, st.ignored), (6, 2, 2, 2, 3));
    let evs = log.lock().unwrap();
    assert_eq!(count(&evs, EventKind::FrameIngested), 6);
    assert_eq!(count(&evs, EventKind::Finished), 1);
    assert!(count(&evs, EventKind::ZonePreview) >= 1);
    // Gap + 3 ignored declarations were reported as warnings.
    let warns: Vec<&str> = evs
        .iter()
        .filter_map(|e| match e {
            Event::Warning { message, .. } if message.starts_with("compose:") => Some(message.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(warns.len(), 4, "{warns:#?}");
    assert!(host.ids().is_empty());
    assert!(matches!(n.close(), Err(ComposeError::Closed)));
    assert_eq!(n.compose(FrameState::ready(set(0, 0))).unwrap(), Outcome::Ignored(IgnoreReason::Closed));
}

#[test]
fn replay_window_applies_corrections_and_late_arrivals() {
    let host = CompositionHost::new();
    let (n, log) = node(&host, "replay", ReconcilePolicy::ReplayWindow { positions: 3 });
    for s in [0u64, 1, 2, 4, 5] {
        n.compose(FrameState::ready(set(s, 0))).unwrap();
    }
    // 3 arrives late: insert at position 3, replay 3 and the two after it.
    assert_eq!(n.compose(FrameState::ready(set(3, 0))).unwrap(), Outcome::Replayed { from: 3, positions: 3 });
    assert_eq!(n.journal(), vec![(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]);
    // Correction of 4 (position 4): replay 4 and 5.
    assert_eq!(n.compose(FrameState::ready(set(4, 1))).unwrap(), Outcome::Replayed { from: 4, positions: 2 });
    assert_eq!(n.journal()[4], (4, 1));
    assert_eq!(n.journal()[5], (5, 0));
    // Older than the window: ignored.
    assert_eq!(n.compose(FrameState::ready(set(1, 1))).unwrap(), Outcome::Ignored(IgnoreReason::OutsideReplayWindow { position: 1 }));
    assert_eq!(n.compose(FrameState::ready(set(4, 0))).unwrap(), Outcome::Ignored(IgnoreReason::StaleRevision { accepted: 1 }));
    for s in 6..NPOS as u64 {
        n.compose(FrameState::ready(set(s, 0))).unwrap();
    }
    assert_eq!(n.compose(FrameState::End).unwrap(), Outcome::Ended);
    let closed = n.close().unwrap();
    assert_eq!(closed.session.positions(), NPOS);
    assert_eq!((closed.stats.replays, closed.stats.replayed_positions), (2, 5));
    let evs = log.lock().unwrap();
    assert_eq!(count(&evs, EventKind::Reset), 2);
    assert_eq!(count(&evs, EventKind::Finished), 1);
    // Every zone of the 8-position flight is present once more after the replays.
    let regs = closed.session.arrived_zones();
    assert_eq!(regs.last().unwrap().hi, NPOS);
}

#[test]
fn invalidate_affected_zones_rebuilds_outputs() {
    let host = CompositionHost::new();
    let (n, log) = node(&host, "invalidate", ReconcilePolicy::InvalidateAffectedZones);
    for s in 0..6u64 {
        n.compose(FrameState::ready(set(s, 0))).unwrap();
    }
    let arrived: Vec<usize> = n.with_session(|s| s.arrived_zones().iter().map(|z| z.zone).collect()).unwrap();
    assert!(arrived.contains(&0), "{arrived:?}");
    // Correction of position 1 carrying in-memory pixels: stored under the original names, zone 0 rebuilt.
    let mut corr = LiveFrameSet::new(FrameKey::new("drone-1", 1, 1));
    for cam in CAMS {
        let bytes = std::fs::read(dataset().join("images").join(name(cam, 1))).unwrap();
        corr = corr.encoded(cam, bytes, EncodedFormat::Jpeg);
    }
    assert_eq!(n.compose(FrameState::ready(corr)).unwrap(), Outcome::Invalidated { position: 1, zones: vec![0] });
    assert_eq!(n.journal()[1], (1, 1));
    // Late arrivals are not inserted by this policy.
    let closed = n.close().unwrap();
    assert_eq!(closed.stats.invalidations, 1);
    let evs = log.lock().unwrap();
    assert_eq!(count(&evs, EventKind::ZoneInvalidated), 1);
}

#[test]
fn strict_rejects_everything_but_the_next_sequence() {
    let host = CompositionHost::new();
    let (n, _) = node(&host, "strict", ReconcilePolicy::Strict);
    n.compose(FrameState::ready(set(0, 0))).unwrap();
    assert!(matches!(n.compose(FrameState::ready(set(2, 0))), Err(ComposeError::Rejected(_))));
    n.compose(FrameState::ready(set(1, 0))).unwrap();
    assert!(matches!(n.compose(FrameState::ready(set(1, 1))), Err(ComposeError::Rejected(_))));
    assert_eq!(n.compose(FrameState::ready(set(1, 0))).unwrap(), Outcome::Duplicate { position: 1 });
    assert_eq!(n.positions(), 2);
    n.close().unwrap();
}

#[test]
fn in_memory_payloads_are_spooled() {
    let host = CompositionHost::new();
    let (n, log) = node(&host, "memory", ReconcilePolicy::AppendOnly);
    let root = images("memory_src");
    for s in 0..5u64 {
        let mut fs = LiveFrameSet::new(FrameKey::new("cam rig/1", s, 0));
        for (i, cam) in CAMS.iter().enumerate() {
            let path = root.join(name(cam, s));
            if i == 0 {
                let img = image::open(&path).unwrap().to_rgb8();
                let (w, h) = img.dimensions();
                fs = fs.rgb8(*cam, w, h, img.into_raw());
            } else {
                fs = fs.encoded(*cam, std::fs::read(&path).unwrap(), EncodedFormat::Jpeg);
            }
            let g = &gps()[&name(cam, s)];
            fs = fs.gps_fix(*cam, g.lat, g.lon, g.alt);
        }
        assert!(matches!(n.compose(FrameState::ready(fs)).unwrap(), Outcome::Appended { .. }));
    }
    let names: Vec<String> = n.with_session(|s| s.frames()[0].iter().map(|f| f.name.clone()).collect()).unwrap();
    assert_eq!(
        names,
        vec![".live/camF/cam_rig_1_0000000000.png", ".live/camR/cam_rig_1_0000000000.jpg", ".live/camL/cam_rig_1_0000000000.jpg"]
    );
    let dir = images_dir(&n);
    for nm in &names {
        assert!(dir.join(nm).is_file(), "{nm}");
    }
    // No partial files left behind.
    let partial = std::fs::read_dir(dir.join(".live/camF"))
        .unwrap()
        .filter(|e| e.as_ref().unwrap().file_name().to_string_lossy().contains(".partial."))
        .count();
    assert_eq!(partial, 0);
    let closed = n.close().unwrap();
    assert!(closed.session.model().is_some_and(|m| m.registered_image_count() >= 12));
    assert_eq!(count(&log.lock().unwrap(), EventKind::Error), 0);
}

fn images_dir(n: &ReconNode) -> PathBuf {
    n.with_session(|s| s.config().image_root.clone()).unwrap()
}

#[test]
fn spawned_node_keep_latest_drops_oldest_ready_sets() {
    let host = CompositionHost::new();
    let n = ReconNode::declare()
        .id("spawned")
        .images(images("spawned"))
        .cameras(CAMS)
        .zones(4, 1)
        .seed(7)
        .backpressure(Backpressure::KeepLatest { max_pending: 2 })
        .build(&host)
        .unwrap();
    let h = n.spawn();
    for s in 0..NPOS as u64 {
        h.send(FrameState::ready(set(s, 0))).unwrap();
    }
    // Drops happen only while enqueueing, so the count is final once every send returned.
    let dropped = h.stats().dropped;
    let (outcomes, closed) = h.close();
    let closed = closed.unwrap();
    assert_eq!(closed.stats.appended + dropped, NPOS as u64);
    assert_eq!(closed.session.positions() as u64, closed.stats.appended);
    assert!(outcomes.iter().all(|o| o.is_ok()), "{outcomes:?}");
    assert!(matches!(outcomes.last(), Some(Ok(Outcome::Ended))));
}

#[test]
fn builder_reports_all_problems_and_duplicate_ids() {
    let host = CompositionHost::new();
    let err = ReconNode::declare()
        .images("/definitely/not/here")
        .cameras(["camF", "camF", "a/b"])
        .zones(2, 2)
        .reconcile(ReconcilePolicy::ReplayWindow { positions: 0 })
        .backpressure(Backpressure::Block { max_pending: 0 })
        .build(&host)
        .unwrap_err();
    let ComposeError::Config(p) = err else { panic!("{err}") };
    assert_eq!(p.len(), 6, "{p:#?}");
    let dir = images("dup");
    ReconNode::declare().id("x").images(&dir).cameras(CAMS).build(&host).unwrap();
    assert!(matches!(ReconNode::declare().id("x").images(&dir).cameras(CAMS).build(&host), Err(ComposeError::DuplicateNode(_))));
    assert_eq!(host.close_all().len(), 1);
}
