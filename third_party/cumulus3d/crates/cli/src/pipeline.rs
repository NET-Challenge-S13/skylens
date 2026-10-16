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

//! 람다 훅 파이프라인(`Pipeline`): 훅 등록·큐·정책·패닉 격리·채널.
//!
//! 리듀서(`Reducer`)를 감싸 입력을 밀어 넣고, 리듀서가 내놓은 이벤트를
//! 1) 구독 채널(`subscribe`)로 모두 복제해 보내고,
//! 2) 등록된 훅(`on`, `on_any`, ...)에 넘긴다.
//!
//! 훅 실행 방식
//! - 비동기(기본): 훅은 "레인" 단위 워커 스레드에서 실행된다. 레인은 이벤트 종류마다 하나,
//!   `on_any` 훅을 위한 레인 하나다. 처리 루프(`push`)는 훅을 기다리지 않는다.
//!   같은 레인 안에서는 도착 순서가 보존되므로 종류 안의 순서는 항상 보존된다.
//!   (`on_any` 레인은 전체 순서를 보존한다.) 서로 다른 종류 사이의 실행 순서는 보장하지 않는다.
//! - 동기(`sync(true)`): 훅을 `push` 를 부른 스레드에서 즉시, 이벤트 순서대로 실행한다
//!   (이벤트마다 종류별 훅 → `on_any` 훅 순). 테스트·파일 출력용.
//!
//! 큐 정책(`policy`)은 종류별로 정한다. 기본은 모두 보존(`QueuePolicy::KeepAll`).
//! 화면 갱신처럼 최신 것만 의미 있는 종류는 아직 처리되지 않은 이전 이벤트를 버리게 할 수 있다.
//!
//! 훅이 패닉하면 `catch_unwind` 로 잡아 `Event::Error` 를 구독자에게 보내고 파이프라인은 계속 돈다.
//! (그 `Error` 는 훅에는 전달하지 않는다. 오류 훅이 다시 패닉해 무한 반복되는 것을 막기 위함.)

use crate::events::{Command, Event, EventKind, Frame, Meta, ZoneRange};
use cumulus3d_core::io::ply::PointCloud;
use std::collections::{HashMap, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::SystemTime;

/// 파이프라인이 구동하는 리듀서.
///
/// session 모듈의 세션 상태가 이 trait 를 구현한다. 세션이 불변 값
/// (`step(&Session, FrameSet) -> (Session, Vec<Event>)`) 형태라면 내부에 현재 세션을 들고
/// `step` 에서 새 세션으로 교체하는 얇은 어댑터로 구현하면 된다.
pub trait Reducer {
    /// 한 번에 밀어 넣는 입력(예: 위치 하나의 프레임 묶음).
    type Input;
    /// 입력 하나를 처리하고 그 결과 이벤트를 순서대로 돌려준다.
    fn step(&mut self, input: Self::Input) -> Vec<Event>;
    /// 입력을 받자마자, `step` 의 처리보다 먼저 내보낼 이벤트(예: 디코딩된 프레임). 기본: 없음.
    fn prelude(&mut self, _input: &Self::Input) -> Vec<Event> {
        Vec::new()
    }
    /// 백그라운드 작업(정밀 처리 등) 중 끝난 것을 수거한다. 기다리지 않는다.
    fn poll(&mut self) -> Vec<Event>;
    /// 명령을 처리한다.
    fn command(&mut self, c: Command) -> Vec<Event>;
    /// 남은 백그라운드 작업을 모두 끝날 때까지 기다려 수거한다.
    /// 기본 구현은 `poll` 한 번. 백그라운드 작업이 있는 리듀서는 재정의해야 한다.
    fn finish(&mut self) -> Vec<Event> {
        self.poll()
    }
}

/// 종류별 큐 정책.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum QueuePolicy {
    /// 모두 보존(기본).
    #[default]
    KeepAll,
    /// 같은 키(구역 번호 / 스냅샷 인덱스)의 미처리 이전 이벤트를 버린다.
    /// 키가 없는 종류는 `LatestOnly` 와 같다.
    LatestPerKey,
    /// 같은 종류의 미처리 이전 이벤트를 모두 버린다.
    LatestOnly,
}

/// `finish` 가 돌려주는 요약.
#[derive(Clone, Debug, Default)]
pub struct Summary {
    /// 리듀서가 낸 이벤트 수.
    pub events: usize,
    /// 종류별 이벤트 수.
    pub by_kind: HashMap<EventKind, usize>,
    /// 실행된 훅 호출 수(패닉 포함).
    pub hook_calls: usize,
    /// 패닉한 훅 호출 수.
    pub hook_panics: usize,
    /// 큐 정책으로 버려진 이벤트 수(레인 단위로 셈).
    pub superseded: usize,
}

type Hook = Box<dyn FnMut(&Event) + Send + 'static>;
type HookList = Arc<Mutex<Vec<Hook>>>;

/// 레인 키: `Some(kind)` 는 종류별 레인, `None` 은 `on_any` 레인.
type LaneKey = Option<EventKind>;

/// 워커 스레드와 처리 루프가 함께 쓰는 상태.
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

    /// 훅 목록을 이벤트 하나에 대해 실행하고 패닉을 격리한다.
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

    /// 정책에 따라 이전 이벤트를 버리고 큐에 넣는다. 버린 개수를 돌려준다.
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

    /// 큐가 비고 실행 중인 훅이 없을 때까지 기다린다.
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

/// 같은 키 판별: 구역 이벤트는 구역 번호, 스냅샷은 인덱스. 나머지는 키 없음.
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

/// 레인 묶음. 버려질 때 워커를 닫고 남은 큐를 비운 뒤 합류한다.
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

/// 리듀서를 감싸는 이벤트 구동기.
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
    /// 비동기 훅 실행, 모든 종류 `KeepAll` 로 만든다.
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

    /// 동기 모드 설정. 첫 `push` 전에 정한다. 동기로 바꾸면 그때까지 쌓인 비동기 큐를 먼저 비운다.
    pub fn sync(mut self, on: bool) -> Self {
        if on && !self.sync {
            self.flush();
        }
        self.sync = on;
        self
    }

    /// 종류별 큐 정책 설정(비동기 모드에서만 의미가 있다).
    pub fn policy(mut self, kind: EventKind, p: QueuePolicy) -> Self {
        self.policies.insert(kind, p);
        self
    }

    /// 특정 종류 훅 등록. 같은 종류 훅은 등록 순서대로 실행된다.
    pub fn on<F>(mut self, kind: EventKind, f: F) -> Self
    where
        F: FnMut(&Event) + Send + 'static,
    {
        self.add_hook(Some(kind), Box::new(f));
        self
    }

    /// 모든 이벤트 훅 등록(전체 순서 보존).
    pub fn on_any<F>(mut self, f: F) -> Self
    where
        F: FnMut(&Event) + Send + 'static,
    {
        self.add_hook(None, Box::new(f));
        self
    }

    /// 프레임 훅: 디코딩된 입력 프레임을 카메라 한 장씩, 위치 처리 전에 받는다.
    /// 세션은 `SessionConfig::decode_frames` 가 켜져 있어야 프레임을 낸다.
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

    /// 구역 초벌 점군 훅.
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

    /// 구역 정밀 점군 훅.
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

    /// 스냅샷 훅(인덱스, 합성 점군).
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

    /// 위치 처리 완료 훅(위치, 등록 수, 기대 수).
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

    /// 경고·오류 메시지 훅(리듀서가 낸 것. 훅 패닉 오류는 구독 채널로만 간다).
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

    /// 모든 이벤트(훅 패닉 오류 포함)를 받는 채널. 수신 측이 버려지면 자동 해제된다.
    pub fn subscribe(&self) -> Receiver<Event> {
        let (tx, rx) = channel();
        self.shared.subscribers.lock().unwrap_or_else(|p| p.into_inner()).push(tx);
        rx
    }

    /// 입력 하나를 리듀서에 넣고 결과 이벤트를 내보낸다. 낸 이벤트 수를 돌려준다.
    /// [`Reducer::prelude`] 이벤트는 처리 전에 먼저 훅 큐에 들어가므로, 비동기 훅은 처리와 동시에 실행된다.
    pub fn push(&mut self, input: R::Input) -> usize {
        let pre = self.reducer.prelude(&input);
        let n = self.dispatch(pre);
        let evs = self.reducer.step(input);
        n + self.dispatch(evs)
    }

    /// 끝난 백그라운드 작업을 수거해 내보낸다(기다리지 않음).
    pub fn poll(&mut self) -> usize {
        let evs = self.reducer.poll();
        self.dispatch(evs)
    }

    /// 명령을 리듀서에 전달하고 결과 이벤트를 내보낸다.
    pub fn command(&mut self, c: Command) -> usize {
        let evs = self.reducer.command(c);
        self.dispatch(evs)
    }

    /// 지금까지 큐에 들어간 훅 실행이 모두 끝날 때까지 기다린다.
    pub fn flush(&mut self) {
        for lane in self.lanes.map.values() {
            lane.wait_idle();
        }
    }

    /// 리듀서 참조.
    pub fn reducer(&self) -> &R {
        &self.reducer
    }

    /// 리듀서 가변 참조(이벤트는 내보내지 않음).
    pub fn reducer_mut(&mut self) -> &mut R {
        &mut self.reducer
    }

    /// 남은 백그라운드 작업을 수거해 내보내고, 훅 큐를 모두 비운 뒤 워커를 닫고 요약을 돌려준다.
    /// 구독 채널은 파이프라인이 끝나면 닫힌다.
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

    /// 입력 n 하나에 대해 PositionDone(n), 짝수면 ZonePreview(zone n/2) 를 낸다.
    /// finish 에서 남은 정밀 결과(ZoneRefined)를 낸다.
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
        // 훅을 막아 둔 상태에서 같은 구역 초벌을 여러 번 넣으면 마지막 것만 남는다.
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
        // 첫 이벤트는 워커가 집어 막혀 있을 수 있으므로 잠시 기다려 확실히 집게 한다.
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
