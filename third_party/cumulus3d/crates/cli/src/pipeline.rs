/*
 * pipeline.rs
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

//! Closure hook pipeline (`Pipeline`): hook registration, queues, policies, panic isolation, channels.
//!
//! It wraps a reducer (`Reducer`), pushes inputs into it, and for the events the reducer emits:
//! 1) copies all of them to subscription channels (`subscribe`),
//! 2) passes them to the registered hooks (`on`, `on_any`, ...).
//!
//! How hooks run
//! - Asynchronous (default): hooks run on "lane" worker threads. There is one lane per event kind
//!   and one lane for `on_any` hooks. The processing loop (`push`) does not wait for hooks.
//!   Arrival order is preserved within a lane, so order within a kind is always preserved.
//!   (The `on_any` lane preserves the overall order.) The execution order between different kinds is not guaranteed.
//! - Synchronous (`sync(true)`): hooks run immediately on the thread that called `push`, in event order
//!   (for each event, hooks of its kind first, then `on_any` hooks). For tests and file output.
//!
//! The queue policy (`policy`) is set per kind. The default is to keep everything (`QueuePolicy::KeepAll`).
//! For kinds where only the latest matters, such as display updates, earlier unprocessed events can be dropped.
//!
//! If a hook panics, it is caught with `catch_unwind`, an `Event::Error` is sent to subscribers, and the pipeline keeps running.
//! (That `Error` is not delivered to hooks, to prevent an error hook from panicking again and looping forever.)

use crate::events::{Command, Event, EventKind, Frame, Meta, ZoneRange};
use cumulus3d_core::io::ply::PointCloud;
use std::collections::{HashMap, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::SystemTime;

/// The reducer driven by the pipeline.
///
/// The session state of the session module implements this trait. If the session has the immutable-value form
/// (`step(&Session, FrameSet) -> (Session, Vec<Event>)`), implement it as a thin adapter that holds the current session
/// and replaces it with the new session in `step`.
pub trait Reducer {
    /// Input pushed in one go (e.g. the frame bundle of one position).
    type Input;
    /// Process one input and return the resulting events in order.
    fn step(&mut self, input: Self::Input) -> Vec<Event>;
    /// Events to emit as soon as the input is received, before `step` processes it (e.g. decoded frames). Default: none.
    fn prelude(&mut self, _input: &Self::Input) -> Vec<Event> {
        Vec::new()
    }
    /// Collect finished background work (refinement etc.). Does not wait.
    fn poll(&mut self) -> Vec<Event>;
    /// Process a command.
    fn command(&mut self, c: Command) -> Vec<Event>;
    /// Wait until all remaining background work finishes and collect it.
    /// The default implementation is a single `poll`. Reducers with background work must override it.
    fn finish(&mut self) -> Vec<Event> {
        self.poll()
    }
}

/// Per-kind queue policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum QueuePolicy {
    /// Keep everything (default).
    #[default]
    KeepAll,
    /// Drop earlier unprocessed events with the same key (zone index / snapshot index).
    /// Kinds without a key behave like `LatestOnly`.
    LatestPerKey,
    /// Drop all earlier unprocessed events of the same kind.
    LatestOnly,
}

/// Summary returned by `finish`.
#[derive(Clone, Debug, Default)]
pub struct Summary {
    /// Number of events emitted by the reducer.
    pub events: usize,
    /// Number of events per kind.
    pub by_kind: HashMap<EventKind, usize>,
    /// Number of hook calls executed (including panics).
    pub hook_calls: usize,
    /// Number of hook calls that panicked.
    pub hook_panics: usize,
    /// Number of events dropped by the queue policy (counted per lane).
    pub superseded: usize,
}

type Hook = Box<dyn FnMut(&Event) + Send + 'static>;
type HookList = Arc<Mutex<Vec<Hook>>>;

/// Lane key: `Some(kind)` is a per-kind lane, `None` is the `on_any` lane.
type LaneKey = Option<EventKind>;

/// State shared by the worker threads and the processing loop.
struct Shared {
    subscribers: Mutex<Vec<Sender<Event>>>,
    hook_calls: AtomicUsize,
    hook_panics: AtomicUsize,
    superseded: AtomicUsize,
}

impl Shared {
    fn broadcast(&self, ev: &Event) {
        let mut subs = self.subscribers.lock().unwrap_or_else(|p| p.into_inner());
        subs.retain(|tx| tx.send(ev.clone()).is_ok());
    }

    /// Run the hook list for one event and isolate panics.
    fn run_hooks(&self, hooks: &HookList, ev: &Event) {
        let mut hooks = hooks.lock().unwrap_or_else(|p| p.into_inner());
        for hook in hooks.iter_mut() {
            self.hook_calls.fetch_add(1, Ordering::Relaxed);
            let r = catch_unwind(AssertUnwindSafe(|| hook(ev)));
            if let Err(payload) = r {
                self.hook_panics.fetch_add(1, Ordering::Relaxed);
                let what = if let Some(s) = payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "알 수 없는 패닉".to_string()
                };
                let err = Event::Error {
                    meta: Meta { version: ev.meta().version, at: SystemTime::now() },
                    message: format!("{:?} 훅 패닉: {what}", ev.kind()),
                };
                self.broadcast(&err);
            }
        }
    }
}

struct LaneQueue {
    items: VecDeque<Event>,
    busy: bool,
    closed: bool,
}

struct LaneState {
    q: Mutex<LaneQueue>,
    cv: Condvar,
}

struct Lane {
    hooks: HookList,
    state: Arc<LaneState>,
    worker: Option<JoinHandle<()>>,
}

impl Lane {
    fn new() -> Self {
        Lane {
            hooks: Arc::new(Mutex::new(Vec::new())),
            state: Arc::new(LaneState {
                q: Mutex::new(LaneQueue { items: VecDeque::new(), busy: false, closed: false }),
                cv: Condvar::new(),
            }),
            worker: None,
        }
    }

    fn ensure_worker(&mut self, name: String, shared: &Arc<Shared>) {
        if self.worker.is_some() {
            return;
        }
        let state = Arc::clone(&self.state);
        let hooks = Arc::clone(&self.hooks);
        let shared = Arc::clone(shared);
        let h = std::thread::Builder::new()
            .name(name)
            .spawn(move || loop {
                let ev = {
                    let mut q = state.q.lock().unwrap_or_else(|p| p.into_inner());
                    loop {
                        if let Some(ev) = q.items.pop_front() {
                            q.busy = true;
                            break Some(ev);
                        }
                        if q.closed {
                            break None;
                        }
                        q = state.cv.wait(q).unwrap_or_else(|p| p.into_inner());
                    }
                };
                let Some(ev) = ev else { return };
                shared.run_hooks(&hooks, &ev);
                let mut q = state.q.lock().unwrap_or_else(|p| p.into_inner());
                q.busy = false;
                state.cv.notify_all();
            })
            .expect("훅 워커 스레드 생성 실패");
        self.worker = Some(h);
    }

    /// Drop earlier events according to the policy and enqueue. Returns the number dropped.
    fn enqueue(&self, ev: Event, policy: QueuePolicy) -> usize {
        let mut q = self.state.q.lock().unwrap_or_else(|p| p.into_inner());
        let before = q.items.len();
        match policy {
            QueuePolicy::KeepAll => {}
            QueuePolicy::LatestOnly => {
                let k = ev.kind();
                q.items.retain(|e| e.kind() != k);
            }
            QueuePolicy::LatestPerKey => {
                let k = (ev.kind(), supersede_key(&ev));
                q.items.retain(|e| (e.kind(), supersede_key(e)) != k);
            }
        }
        let dropped = before - q.items.len();
        q.items.push_back(ev);
        self.state.cv.notify_all();
        dropped
    }

    /// Wait until the queue is empty and no hook is running.
    fn wait_idle(&self) {
        if self.worker.is_none() {
            return;
        }
        let mut q = self.state.q.lock().unwrap_or_else(|p| p.into_inner());
        while !q.items.is_empty() || q.busy {
            q = self.state.cv.wait(q).unwrap_or_else(|p| p.into_inner());
        }
    }

    fn shutdown(&mut self) {
        {
            let mut q = self.state.q.lock().unwrap_or_else(|p| p.into_inner());
            q.closed = true;
            self.state.cv.notify_all();
        }
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
}

/// Same-key test: zone events use the zone index, snapshots the index. Others have no key.
fn supersede_key(ev: &Event) -> Option<usize> {
    match ev {
        Event::ZoneArrived { range, .. }
        | Event::ZonePreview { range, .. }
        | Event::ZoneRefinedPose { range, .. }
        | Event::ZoneRefined { range, .. } => Some(range.zone),
        Event::BaseAdopted { zone, .. }
        | Event::Reanchored { zone, .. }
        | Event::RefineStarted { zone, .. }
        | Event::ZoneAdjusted { zone, .. }
        | Event::ZoneInvalidated { zone, .. } => Some(*zone),
        Event::Snapshot { index, .. } => Some(*index),
        Event::PositionDone { position, .. } | Event::FrameIngested { position, .. } => Some(*position),
        _ => None,
    }
}

/// Lane set. On drop, it closes the workers, empties the remaining queues and joins.
struct Lanes {
    map: HashMap<LaneKey, Lane>,
}

impl Lanes {
    fn shutdown(&mut self) {
        for lane in self.map.values_mut() {
            lane.shutdown();
        }
    }
}

impl Drop for Lanes {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Event driver that wraps a reducer.
pub struct Pipeline<R: Reducer> {
    reducer: R,
    lanes: Lanes,
    policies: HashMap<EventKind, QueuePolicy>,
    shared: Arc<Shared>,
    sync: bool,
    events: usize,
    by_kind: HashMap<EventKind, usize>,
}

impl<R: Reducer> Pipeline<R> {
    /// Create it with asynchronous hook execution and `KeepAll` for every kind.
    pub fn new(reducer: R) -> Self {
        Pipeline {
            reducer,
            lanes: Lanes { map: HashMap::new() },
            policies: HashMap::new(),
            shared: Arc::new(Shared {
                subscribers: Mutex::new(Vec::new()),
                hook_calls: AtomicUsize::new(0),
                hook_panics: AtomicUsize::new(0),
                superseded: AtomicUsize::new(0),
            }),
            sync: false,
            events: 0,
            by_kind: HashMap::new(),
        }
    }

    /// Set synchronous mode. Decide before the first `push`. Switching to synchronous first drains the asynchronous queues accumulated so
    /// far.
    pub fn sync(mut self, on: bool) -> Self {
        if on && !self.sync {
            self.flush();
        }
        self.sync = on;
        self
    }

    /// Set the per-kind queue policy (only meaningful in asynchronous mode).
    pub fn policy(mut self, kind: EventKind, p: QueuePolicy) -> Self {
        self.policies.insert(kind, p);
        self
    }

    /// Register a hook for a specific kind. Hooks of the same kind run in registration order.
    pub fn on<F>(mut self, kind: EventKind, f: F) -> Self
    where
        F: FnMut(&Event) + Send + 'static,
    {
        self.add_hook(Some(kind), Box::new(f));
        self
    }

    /// Register a hook for all events (preserves the overall order).
    pub fn on_any<F>(mut self, f: F) -> Self
    where
        F: FnMut(&Event) + Send + 'static,
    {
        self.add_hook(None, Box::new(f));
        self
    }

    /// Frame hook: receives each decoded input frame, one camera at a time, before the position is processed.
    /// The session emits frames only if `SessionConfig::decode_frames` is on.
    pub fn on_frame<F>(self, mut f: F) -> Self
    where
        F: FnMut(&Arc<Frame>) + Send + 'static,
    {
        self.on(EventKind::FrameDecoded, move |e| {
            if let Event::FrameDecoded { frame, .. } = e {
                f(frame)
            }
        })
    }

    /// Zone preview point-cloud hook.
    pub fn on_zone_preview<F>(self, mut f: F) -> Self
    where
        F: FnMut(ZoneRange, &Arc<PointCloud>) + Send + 'static,
    {
        self.on(EventKind::ZonePreview, move |e| {
            if let Event::ZonePreview { range, cloud, .. } = e {
                f(*range, cloud)
            }
        })
    }

    /// Zone refined point-cloud hook.
    pub fn on_zone_refined<F>(self, mut f: F) -> Self
    where
        F: FnMut(ZoneRange, &Arc<PointCloud>) + Send + 'static,
    {
        self.on(EventKind::ZoneRefined, move |e| {
            if let Event::ZoneRefined { range, cloud, .. } = e {
                f(*range, cloud)
            }
        })
    }

    /// Snapshot hook (index, composite point cloud).
    pub fn on_snapshot<F>(self, mut f: F) -> Self
    where
        F: FnMut(usize, &Arc<PointCloud>) + Send + 'static,
    {
        self.on(EventKind::Snapshot, move |e| {
            if let Event::Snapshot { index, cloud, .. } = e {
                f(*index, cloud)
            }
        })
    }

    /// Position-done hook (position, registered count, expected count).
    pub fn on_position_done<F>(self, mut f: F) -> Self
    where
        F: FnMut(usize, usize, usize) + Send + 'static,
    {
        self.on(EventKind::PositionDone, move |e| {
            if let Event::PositionDone { position, registered, expected, .. } = e {
                f(*position, *registered, *expected)
            }
        })
    }

    /// Warning/error message hook (those emitted by the reducer; hook panic errors go only to the subscription channel).
    pub fn on_message<F>(self, f: F) -> Self
    where
        F: FnMut(EventKind, &str) + Send + 'static,
    {
        let f = Arc::new(Mutex::new(f));
        let g = Arc::clone(&f);
        self.on(EventKind::Warning, move |e| {
            if let Event::Warning { message, .. } = e {
                (f.lock().unwrap_or_else(|p| p.into_inner()))(EventKind::Warning, message)
            }
        })
        .on(EventKind::Error, move |e| {
            if let Event::Error { message, .. } = e {
                (g.lock().unwrap_or_else(|p| p.into_inner()))(EventKind::Error, message)
            }
        })
    }

    fn add_hook(&mut self, key: LaneKey, hook: Hook) {
        let lane = self.lanes.map.entry(key).or_insert_with(Lane::new);
        lane.hooks.lock().unwrap_or_else(|p| p.into_inner()).push(hook);
    }

    /// Channel that receives all events (including hook panic errors). It is released automatically if the receiver is dropped.
    pub fn subscribe(&self) -> Receiver<Event> {
        let (tx, rx) = channel();
        self.shared.subscribers.lock().unwrap_or_else(|p| p.into_inner()).push(tx);
        rx
    }

    /// Feed one input to the reducer and emit the resulting events. Returns the number of events emitted.
    /// [`Reducer::prelude`] events enter the hook queue before processing, so asynchronous hooks run concurrently with the processing.
    pub fn push(&mut self, input: R::Input) -> usize {
        let pre = self.reducer.prelude(&input);
        let n = self.dispatch(pre);
        let evs = self.reducer.step(input);
        n + self.dispatch(evs)
    }

    /// Collect finished background work and emit it (does not wait).
    pub fn poll(&mut self) -> usize {
        let evs = self.reducer.poll();
        self.dispatch(evs)
    }

    /// Pass a command to the reducer and emit the resulting events.
    pub fn command(&mut self, c: Command) -> usize {
        let evs = self.reducer.command(c);
        self.dispatch(evs)
    }

    /// Wait until all hook executions queued so far have finished.
    pub fn flush(&mut self) {
        for lane in self.lanes.map.values() {
            lane.wait_idle();
        }
    }

    /// Reducer reference.
    pub fn reducer(&self) -> &R {
        &self.reducer
    }

    /// Mutable reducer reference (events are not emitted).
    pub fn reducer_mut(&mut self) -> &mut R {
        &mut self.reducer
    }

    /// Collect and emit the remaining background work, drain the hook queues, close the workers and return the summary.
    /// The subscription channels are closed when the pipeline ends.
    pub fn finish(mut self) -> (R, Summary) {
        let evs = self.reducer.finish();
        self.dispatch(evs);
        self.lanes.shutdown();
        self.shared.subscribers.lock().unwrap_or_else(|p| p.into_inner()).clear();
        let summary = Summary {
            events: self.events,
            by_kind: std::mem::take(&mut self.by_kind),
            hook_calls: self.shared.hook_calls.load(Ordering::Relaxed),
            hook_panics: self.shared.hook_panics.load(Ordering::Relaxed),
            superseded: self.shared.superseded.load(Ordering::Relaxed),
        };
        let Pipeline { reducer, .. } = self;
        (reducer, summary)
    }

    fn dispatch(&mut self, evs: Vec<Event>) -> usize {
        let n = evs.len();
        for ev in evs {
            self.events += 1;
            *self.by_kind.entry(ev.kind()).or_insert(0) += 1;
            self.shared.broadcast(&ev);
            let kind = ev.kind();
            if self.sync {
                for key in [Some(kind), None] {
                    if let Some(lane) = self.lanes.map.get(&key) {
                        self.shared.run_hooks(&lane.hooks, &ev);
                    }
                }
                continue;
            }
            let policy = self.policies.get(&kind).copied().unwrap_or_default();
            let has_kind = self.lanes.map.contains_key(&Some(kind));
            let has_any = self.lanes.map.contains_key(&None);
            let mut pending = Some(ev);
            for (key, present, last) in [(Some(kind), has_kind, !has_any), (None, has_any, true)] {
                if !present {
                    continue;
                }
                let ev = if last { pending.take() } else { pending.clone() };
                let Some(ev) = ev else { break };
                let name = match key {
                    Some(k) => format!("hook-{k:?}"),
                    None => "hook-any".to_string(),
                };
                let lane = self.lanes.map.get_mut(&key).expect("레인 존재 확인됨");
                lane.ensure_worker(name, &self.shared);
                let dropped = lane.enqueue(ev, policy);
                self.shared.superseded.fetch_add(dropped, Ordering::Relaxed);
            }
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cumulus3d_core::Reconstruction;
    use std::sync::mpsc::RecvTimeoutError;
    use std::time::Duration;

    fn meta(v: u64) -> Meta {
        Meta { version: v, at: SystemTime::now() }
    }
    fn range(z: usize) -> ZoneRange {
        ZoneRange { zone: z, lo: z * 10, hi: z * 10 + 10 }
    }
    fn cloud() -> Arc<PointCloud> {
        Arc::new(PointCloud::default())
    }
    fn preview(v: u64, z: usize) -> Event {
        Event::ZonePreview {
            meta: meta(v),
            range: range(z),
            cloud: cloud(),
            model: Arc::new(Reconstruction::default()),
            dense: true,
            frame: "gps".to_string(),
        }
    }
    fn refined(v: u64, z: usize) -> Event {
        Event::ZoneRefined { meta: meta(v), range: range(z), cloud: cloud(), model: Arc::new(Reconstruction::default()), dense: true }
    }

    /// For input n, emit PositionDone(n), and for even n also ZonePreview(zone n/2).
    /// In finish, emit the remaining refined results (ZoneRefined).
    #[derive(Default)]
    struct Fake {
        v: u64,
        zones: Vec<usize>,
        commands: Vec<String>,
    }

    impl Reducer for Fake {
        type Input = usize;
        fn step(&mut self, n: usize) -> Vec<Event> {
            self.v += 1;
            let mut out = vec![Event::PositionDone { meta: meta(self.v), position: n, registered: n, expected: n }];
            if n.is_multiple_of(2) {
                out.push(preview(self.v, n / 2));
                self.zones.push(n / 2);
            }
            out
        }
        fn poll(&mut self) -> Vec<Event> {
            Vec::new()
        }
        fn command(&mut self, c: Command) -> Vec<Event> {
            self.commands.push(format!("{c:?}"));
            match c {
                Command::InvalidateZone(z) => vec![Event::Warning { meta: meta(self.v), message: format!("invalidate {z}") }],
                Command::ResetFrom(p) => vec![Event::Warning { meta: meta(self.v), message: format!("reset {p}") }],
            }
        }
        fn finish(&mut self) -> Vec<Event> {
            self.zones.drain(..).map(|z| refined(self.v, z)).collect()
        }
    }

    fn log<T>() -> Arc<Mutex<Vec<T>>> {
        Arc::new(Mutex::new(Vec::new()))
    }

    #[test]
    fn pipeline_sync_hook_order() {
        let l = log();
        let (a, b, c) = (Arc::clone(&l), Arc::clone(&l), Arc::clone(&l));
        let mut p = Pipeline::new(Fake::default())
            .sync(true)
            .on_any(move |e| a.lock().unwrap().push(format!("any:{:?}", e.kind())))
            .on_position_done(move |pos, _, _| b.lock().unwrap().push(format!("pos:{pos}")))
            .on_zone_preview(move |r, _| c.lock().unwrap().push(format!("prev:{}", r.zone)));
        p.push(1);
        p.push(2);
        let (_, s) = p.finish();
        let got = l.lock().unwrap().clone();
        assert_eq!(got, vec!["pos:1", "any:PositionDone", "pos:2", "any:PositionDone", "prev:1", "any:ZonePreview", "any:ZoneRefined"]);
        assert_eq!(s.events, 4);
        assert_eq!(s.by_kind[&EventKind::PositionDone], 2);
    }

    #[test]
    fn pipeline_async_order_within_kind() {
        let l = log::<String>();
        let any = log::<String>();
        let (a, b) = (Arc::clone(&l), Arc::clone(&any));
        let mut p = Pipeline::new(Fake::default())
            .on_position_done(move |pos, _, _| {
                if pos.is_multiple_of(7) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                a.lock().unwrap().push(pos.to_string())
            })
            .on_any(move |e| b.lock().unwrap().push(format!("{:?}", e.kind())));
        for i in 0..200 {
            p.push(i);
        }
        let (_, s) = p.finish();
        let got: Vec<usize> = l.lock().unwrap().iter().map(|s| s.parse().unwrap()).collect();
        assert_eq!(got, (0..200).collect::<Vec<_>>());
        assert_eq!(any.lock().unwrap().len(), s.events);
        assert_eq!(s.superseded, 0);
    }

    #[test]
    fn pipeline_latest_per_key_policy() {
        // With the hooks blocked, putting in previews of the same zone several times leaves only the last one.
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let seen = log();
        let (g, s2) = (Arc::clone(&gate), Arc::clone(&seen));

        struct Previews(u64);
        impl Reducer for Previews {
            type Input = (usize, u64);
            fn step(&mut self, (z, v): (usize, u64)) -> Vec<Event> {
                self.0 += 1;
                vec![preview(v, z)]
            }
            fn poll(&mut self) -> Vec<Event> {
                Vec::new()
            }
            fn command(&mut self, _: Command) -> Vec<Event> {
                Vec::new()
            }
        }

        let mut p =
            Pipeline::new(Previews(0)).policy(EventKind::ZonePreview, QueuePolicy::LatestPerKey).on(EventKind::ZonePreview, move |e| {
                let (m, cv) = &*g;
                let mut open = m.lock().unwrap();
                while !*open {
                    open = cv.wait(open).unwrap();
                }
                if let Event::ZonePreview { meta, range, .. } = e {
                    s2.lock().unwrap().push(format!("{}@{}", range.zone, meta.version));
                }
            });
        // The worker may have picked up the first event and be blocked, so wait a moment to make sure it is picked up.
        p.push((9, 0));
        std::thread::sleep(Duration::from_millis(50));
        for v in 1..=5 {
            p.push((1, v));
            p.push((2, v));
        }
        {
            let (m, cv) = &*gate;
            *m.lock().unwrap() = true;
            cv.notify_all();
        }
        let (_, s) = p.finish();
        assert_eq!(seen.lock().unwrap().clone(), vec!["9@0", "1@5", "2@5"]);
        assert_eq!(s.superseded, 8);
        assert_eq!(s.events, 11);
    }

    #[test]
    fn pipeline_panic_isolated() {
        let l = log();
        let a = Arc::clone(&l);
        let mut p = Pipeline::new(Fake::default())
            .on_position_done(|pos, _, _| {
                if pos == 1 {
                    panic!("boom");
                }
            })
            .on_position_done(move |pos, _, _| a.lock().unwrap().push(pos));
        let rx = p.subscribe();
        for i in 0..4 {
            p.push(i);
        }
        let (_, s) = p.finish();
        assert_eq!(l.lock().unwrap().clone(), vec![0, 1, 2, 3]);
        assert_eq!(s.hook_panics, 1);
        let errs: Vec<String> = rx
            .iter()
            .filter_map(|e| match e {
                Event::Error { message, meta } => {
                    assert_eq!(meta.version, 2);
                    Some(message)
                }
                _ => None,
            })
            .collect();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("boom"), "{}", errs[0]);
    }

    #[test]
    fn pipeline_channel_receives_all() {
        let mut p = Pipeline::new(Fake::default());
        let rx = p.subscribe();
        p.push(1);
        p.push(2);
        let kinds: Vec<EventKind> = rx.try_iter().map(|e| e.kind()).collect();
        assert_eq!(kinds, vec![EventKind::PositionDone, EventKind::PositionDone, EventKind::ZonePreview]);
        let dropped = p.subscribe();
        drop(dropped);
        p.push(3);
        let (_, s) = p.finish();
        let rest: Vec<EventKind> = rx.iter().map(|e| e.kind()).collect();
        assert_eq!(rest, vec![EventKind::PositionDone, EventKind::ZoneRefined]);
        assert_eq!(s.events, 5);
        assert!(matches!(rx.recv_timeout(Duration::from_millis(1)), Err(RecvTimeoutError::Disconnected)));
    }

    #[test]
    fn pipeline_finish_drains() {
        let n = Arc::new(AtomicUsize::new(0));
        let refined = log();
        let (a, b) = (Arc::clone(&n), Arc::clone(&refined));
        let mut p = Pipeline::new(Fake::default())
            .on(EventKind::PositionDone, move |_| {
                std::thread::sleep(Duration::from_micros(200));
                a.fetch_add(1, Ordering::SeqCst);
            })
            .on_zone_refined(move |r, _| b.lock().unwrap().push(r.zone));
        for i in 0..50 {
            p.push(i);
        }
        let (r, s) = p.finish();
        assert_eq!(n.load(Ordering::SeqCst), 50);
        assert_eq!(refined.lock().unwrap().len(), 25);
        assert!(r.zones.is_empty());
        assert_eq!(s.hook_calls, 75);
    }

    #[test]
    fn pipeline_command_forwarded() {
        let msgs = log();
        let a = Arc::clone(&msgs);
        let mut p = Pipeline::new(Fake::default()).sync(true).on_message(move |k, m| a.lock().unwrap().push(format!("{k:?}:{m}")));
        assert_eq!(p.command(Command::InvalidateZone(3)), 1);
        p.command(Command::ResetFrom(7));
        assert_eq!(p.reducer().commands, vec!["InvalidateZone(3)", "ResetFrom(7)"]);
        let (_, _) = p.finish();
        assert_eq!(msgs.lock().unwrap().clone(), vec!["Warning:invalidate 3", "Warning:reset 7"]);
    }
}
