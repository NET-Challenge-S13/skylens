/*
 * blocks.rs
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

//! Live stream (`<out>/live/stream/`, format version 4): after every live event the composed display cloud (point keys and
//! the owner zone of every point) is diffed against what the viewer shows and only the changes are written. Every shown
//! point owns a dense `u32` slot id assigned here, so the viewer writes straight into its GPU buffers at index = id.
//!
//! `packs/e_<seq:04>.bin` (little endian, no header) holds three sections in this order: `n_add` records of 20 B
//! (`u32 id | f32 x y z | u8 r g b | u8 zone`), `n_move` records of 20 B (`u32 id | f32 x y z | u8 r g b | u8 pad=0`, new position and colour), `n_remove` records of
//! 4 B (`u32 id`). `manifest.json` (rewritten atomically after every event) lists the events with their counts and `capacity`
//! (max id + 1 used so far). A key that stays keeps its id (moved when its position changed by more than [`POSITION_EPS`] or its colour changed);
//! a key that leaves frees its id; a new key takes the lowest free id or `capacity++`. An id freed in an event is not handed
//! out in the same event. A zone change is a remove plus an add with a new id. Files are never overwritten.

use crate::util::{py_float, Json};
use cumulus3d_core::io::PointCloud;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::path::{Path, PathBuf};

/// A point whose position moved by more than this (display units, any axis) is sent as a move.
pub const POSITION_EPS: f32 = 1e-4;

/// Manifest format version.
pub const FORMAT_VERSION: i64 = 4;

/// Bytes of an add record.
pub const ADD_BYTES: usize = 20;
/// Bytes of a move record.
pub const MOVE_BYTES: usize = 20;
/// Bytes of a remove record.
pub const REMOVE_BYTES: usize = 4;

/// Live stream settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiveStream {
    /// The display frame is the ENU frame (metres, x east, y north, z up); otherwise `"display"`.
    pub enu: bool,
    /// Geodetic origin `[lat, lon, alt]` of the ENU frame, when known.
    pub origin_gps: Option<[f64; 3]>,
}

/// Cheap hasher for the already well-mixed 64-bit pixel keys.
#[derive(Default)]
pub struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0 ^ (self.0 >> 32)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(8) ^ b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

#[derive(Clone, Copy, Debug)]
struct Shown {
    id: u32,
    pos: [f32; 3],
    color: [u8; 3],
    zone: u8,
    /// Event number that last saw this key (to find the removed ones without a second set).
    seen: u32,
}

/// What one [`DiffStream::write_event`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EventStats {
    /// Points added.
    pub n_add: usize,
    /// Points moved.
    pub n_move: usize,
    /// Points removed.
    pub n_remove: usize,
    /// Points shown after the event.
    pub points: usize,
    /// Pack size in bytes (0 when nothing changed).
    pub bytes: usize,
}

/// Writer of the live diff stream.
pub struct DiffStream {
    dir: PathBuf,
    cfg: LiveStream,
    /// key → what the viewer shows.
    last: HashMap<u64, Shown, BuildHasherDefault<KeyHasher>>,
    /// Free ids (lowest first).
    free: BinaryHeap<Reverse<u32>>,
    /// Max id + 1 used so far.
    capacity: u32,
    /// Manifest events written so far.
    events: Vec<Json>,
    seq: i64,
    /// Counts attempts (also failed ones), stamps [`Shown::seen`].
    epoch: u32,
}

impl DiffStream {
    /// Stream into `dir` (`live/stream`); nothing is written until the first event.
    pub fn new(dir: PathBuf, cfg: LiveStream) -> Self {
        Self { dir, cfg, last: HashMap::default(), free: BinaryHeap::new(), capacity: 0, events: Vec::new(), seq: 0, epoch: 0 }
    }

    /// Number of events written so far (the last `seq`).
    pub fn events_written(&self) -> i64 {
        self.seq
    }

    /// Directory of the stream.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Records one event: `cloud` is the whole composed display (unique keys required), `zones[i]` the owner zone of
    /// point i, `t` seconds since the session start, `zone` the zone that triggered the event, `kind` one of `refined`,
    /// `replace`, `final`. Writes the pack and the manifest; on an error nothing of this event is recorded.
    pub fn write_event(&mut self, t: f64, zone: usize, kind: &str, cloud: &PointCloud, zones: &[usize]) -> Result<EventStats, String> {
        let n = cloud.len();
        if n > 0 && cloud.keys.len() != n {
            return Err("점 키가 없는 점군은 스트림으로 낼 수 없음".into());
        }
        if zones.len() != n {
            return Err("점별 구역 길이가 점 수와 다름".into());
        }
        self.epoch += 1;
        let epoch = self.epoch;
        if self.last.is_empty() {
            self.last.reserve(n);
        }
        // Pass 1: classify. Kept keys are stamped; unseen keys are removals.
        let mut adds: Vec<(u64, [f32; 3], [u8; 3], u8)> = Vec::new();
        let mut moves: Vec<(u64, u32, [f32; 3], [u8; 3])> = Vec::new();
        let mut gone: Vec<u32> = Vec::new();
        let mut replaced = 0usize;
        for (i, &p) in cloud.positions.iter().enumerate() {
            if !p.iter().all(|v| v.is_finite()) {
                continue;
            }
            let key = cloud.keys[i];
            let color = cloud.colors.get(i).copied().unwrap_or([255; 3]);
            let zn = zones[i].min(255) as u8;
            match self.last.get_mut(&key) {
                Some(s) if s.seen == epoch => {}
                Some(s) => {
                    s.seen = epoch;
                    if s.zone != zn {
                        gone.push(s.id);
                        replaced += 1;
                        adds.push((key, p, color, zn));
                    } else if s.color != color || (0..3).any(|a| (s.pos[a] - p[a]).abs() > POSITION_EPS) {
                        moves.push((key, s.id, p, color));
                    }
                }
                None => adds.push((key, p, color, zn)),
            }
        }
        let mut removed_keys: Vec<u64> = Vec::new();
        for (k, s) in &self.last {
            if s.seen != epoch {
                removed_keys.push(*k);
                gone.push(s.id);
            }
        }
        // Ids for the adds: lowest free id of earlier events, else a new one. Undone when the write fails.
        let (cap0, mut taken) = (self.capacity, Vec::new());
        let mut add_ids: Vec<u32> = Vec::with_capacity(adds.len());
        for _ in 0..adds.len() {
            let id = match self.free.pop() {
                Some(Reverse(id)) => {
                    taken.push(id);
                    id
                }
                None => {
                    self.capacity += 1;
                    self.capacity - 1
                }
            };
            add_ids.push(id);
        }
        let n_remove = gone.len();
        let bytes = adds.len() * ADD_BYTES + moves.len() * MOVE_BYTES + n_remove * REMOVE_BYTES;
        let mut pack: Vec<u8> = Vec::with_capacity(bytes);
        for ((_, p, c, z), id) in adds.iter().zip(&add_ids) {
            pack.extend_from_slice(&id.to_le_bytes());
            for v in p {
                pack.extend_from_slice(&v.to_le_bytes());
            }
            pack.extend_from_slice(c);
            pack.push(*z);
        }
        for (_, id, p, c) in &moves {
            pack.extend_from_slice(&id.to_le_bytes());
            for v in p {
                pack.extend_from_slice(&v.to_le_bytes());
            }
            pack.extend_from_slice(c);
            pack.push(0);
        }
        for id in &gone {
            pack.extend_from_slice(&id.to_le_bytes());
        }
        let seq = self.seq + 1;
        let (n_add, n_move) = (adds.len(), moves.len());
        let points = self.last.len() - removed_keys.len() + n_add - replaced;
        let pack_name = (bytes > 0).then(|| format!("packs/e_{seq:04}.bin"));
        let ev = Json::obj(vec![
            ("seq", Json::Int(seq)),
            ("t", Json::f(t, 2)),
            ("zone", Json::Int(zone as i64)),
            ("kind", Json::Str(kind.to_string())),
            ("points", Json::Int(points as i64)),
            ("pack", pack_name.clone().map_or(Json::Null, Json::Str)),
            ("n_add", Json::Int(n_add as i64)),
            ("n_move", Json::Int(n_move as i64)),
            ("n_remove", Json::Int(n_remove as i64)),
            ("bytes", Json::Int(bytes as i64)),
        ]);
        let written = match &pack_name {
            Some(name) => atomic_write(&self.dir.join(name), &pack),
            None => Ok(()),
        };
        self.events.push(ev);
        let written = written.and_then(|_| self.write_manifest());
        if let Err(e) = written {
            self.events.pop();
            self.capacity = cap0;
            for id in taken {
                self.free.push(Reverse(id));
            }
            return Err(e);
        }
        // Commit.
        for k in &removed_keys {
            self.last.remove(k);
        }
        for (k, id, p, c) in &moves {
            if let Some(s) = self.last.get_mut(k) {
                s.pos = *p;
                s.color = *c;
                debug_assert_eq!(s.id, *id);
            }
        }
        self.last.reserve(adds.len().saturating_sub(replaced));
        for ((key, pos, color, zone), id) in adds.into_iter().zip(add_ids) {
            self.last.insert(key, Shown { id, pos, color, zone, seen: epoch });
        }
        self.free.extend(gone.into_iter().map(Reverse));
        self.seq = seq;
        Ok(EventStats { n_add, n_move, n_remove, points: self.last.len(), bytes })
    }

    fn write_manifest(&self) -> Result<(), String> {
        let origin = match self.cfg.origin_gps {
            Some(o) => Json::Arr(o.iter().map(|v| Json::Num(py_float(*v, 9))).collect()),
            None => Json::Null,
        };
        let m = Json::obj(vec![
            ("version", Json::Int(FORMAT_VERSION)),
            ("frame", Json::Str(if self.cfg.enu { "enu" } else { "display" }.into())),
            ("origin_gps", origin),
            ("capacity", Json::Int(self.capacity as i64)),
            ("events", Json::Arr(self.events.clone())),
        ]);
        atomic_write(&self.dir.join("manifest.json"), m.dump().as_bytes())
    }
}

/// Progress heartbeat `<dir>/status.json` of a running session (rewritten atomically; tiny).
#[derive(Clone, Debug, Default)]
pub struct Heartbeat {
    /// Stream folder (`live/stream`).
    pub dir: PathBuf,
    /// `registration` | `triangulation` | `ba` | `dense` | `done`.
    pub stage: &'static str,
    /// Zone being processed (or the latest one).
    pub zone: usize,
    /// Positions processed so far.
    pub positions: usize,
    /// Total positions, if known.
    pub positions_total: Option<usize>,
    /// Registered images so far.
    pub images: usize,
}

impl Heartbeat {
    /// Writes `status.json` for `t` seconds since the session start.
    pub fn write(&self, t: f64) -> Result<(), String> {
        let total = self.positions_total.map_or(Json::Null, |n| Json::Int(n as i64));
        let m = Json::obj(vec![
            ("t", Json::f(t, 1)),
            ("stage", Json::Str(self.stage.into())),
            ("zone", Json::Int(self.zone as i64)),
            ("positions", Json::Int(self.positions as i64)),
            ("positions_total", total),
            ("images", Json::Int(self.images as i64)),
        ]);
        atomic_write(&self.dir.join("status.json"), m.dump().as_bytes())
    }
}

/// Writes `bytes` to `path` through `<path>.tmp` + rename (parent folders are created).
pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, bytes).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("cumulus3d_diff_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn cfg() -> LiveStream {
        LiveStream { enu: true, origin_gps: Some([37.5, 127.0, 30.0]) }
    }

    /// A 12 x 12 x 2 lattice with keys 1..=n and a zone per half.
    fn lattice() -> (PointCloud, Vec<usize>) {
        let mut c = PointCloud::default();
        let mut zones = Vec::new();
        for i in 0..12 {
            for j in 0..12 {
                for k in 0..2 {
                    c.positions.push([i as f32 * 0.7 - 3.0, j as f32 * 0.7 - 3.0, k as f32 * 0.7]);
                    c.colors.push([(i * 20) as u8, (j * 20) as u8, 7]);
                    c.keys.push(c.positions.len() as u64);
                    zones.push((i / 6) as usize);
                }
            }
        }
        (c, zones)
    }

    /// Subset of a cloud by index.
    fn pick(c: &PointCloud, z: &[usize], idx: &[usize]) -> (PointCloud, Vec<usize>) {
        let mut o = PointCloud::default();
        let mut oz = Vec::new();
        for &i in idx {
            o.positions.push(c.positions[i]);
            o.colors.push(c.colors[i]);
            o.keys.push(c.keys[i]);
            oz.push(z[i]);
        }
        (o, oz)
    }

    type Ids = (Vec<u32>, Vec<u32>);
    type Slot = Option<([f32; 3], [u8; 3], u8)>;

    /// One event of the manifest: (seq, pack, n_add, n_move, n_remove, bytes, points).
    struct Ev {
        pack: Option<String>,
        n: [usize; 3],
        bytes: usize,
        points: usize,
    }

    fn manifest(dir: &Path) -> (Json, Vec<Ev>) {
        let m = Json::parse(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
        let Some(Json::Arr(events)) = m.get("events") else { panic!("no events") };
        let num = |e: &Json, k: &str| e.get(k).unwrap().as_f64().unwrap() as usize;
        let evs = events
            .iter()
            .map(|e| Ev {
                pack: match e.get("pack") {
                    Some(Json::Str(p)) => Some(p.clone()),
                    _ => None,
                },
                n: [num(e, "n_add"), num(e, "n_move"), num(e, "n_remove")],
                bytes: num(e, "bytes"),
                points: num(e, "points"),
            })
            .collect();
        (m, evs)
    }

    /// Applies the packs [from, to) to `slots` the way a viewer does; returns the (added, removed) ids of the last one.
    fn apply(dir: &Path, slots: &mut Vec<Slot>, ev: &Ev) -> (Vec<u32>, Vec<u32>) {
        let Some(p) = &ev.pack else { return (vec![], vec![]) };
        let b = std::fs::read(dir.join(p)).unwrap();
        assert_eq!(b.len(), ev.bytes);
        assert_eq!(b.len(), ev.n[0] * ADD_BYTES + ev.n[1] * MOVE_BYTES + ev.n[2] * REMOVE_BYTES);
        let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let f = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let (mut added, mut removed) = (vec![], vec![]);
        let mut o = 0;
        // removes first, as the viewer does
        let rm0 = ev.n[0] * ADD_BYTES + ev.n[1] * MOVE_BYTES;
        for r in 0..ev.n[2] {
            let id = u(rm0 + r * 4);
            removed.push(id);
            slots[id as usize] = None;
        }
        for _ in 0..ev.n[0] {
            let id = u(o) as usize;
            if slots.len() <= id {
                slots.resize(id + 1, None);
            }
            assert!(slots[id].is_none(), "add into a live slot {id}");
            slots[id] = Some(([f(o + 4), f(o + 8), f(o + 12)], [b[o + 16], b[o + 17], b[o + 18]], b[o + 19]));
            added.push(id as u32);
            o += ADD_BYTES;
        }
        for _ in 0..ev.n[1] {
            let id = u(o) as usize;
            let s = slots[id].as_mut().expect("move of an empty slot");
            s.0 = [f(o + 4), f(o + 8), f(o + 12)];
            s.1 = [b[o + 16], b[o + 17], b[o + 18]];
            assert_eq!(b[o + 19], 0);
            o += MOVE_BYTES;
        }
        (added, removed)
    }

    fn replay(dir: &Path) -> (Vec<Slot>, Vec<Ids>) {
        let (_, evs) = manifest(dir);
        let mut slots = Vec::new();
        let ids = evs.iter().map(|e| apply(dir, &mut slots, e)).collect();
        (slots, ids)
    }

    fn shown(slots: &[Slot]) -> Vec<([u32; 3], [u8; 3], u8)> {
        let mut v: Vec<_> = slots.iter().flatten().map(|s| (s.0.map(f32::to_bits), s.1, s.2)).collect();
        v.sort();
        v
    }

    fn expect(c: &PointCloud, z: &[usize]) -> Vec<([u32; 3], [u8; 3], u8)> {
        let mut v: Vec<_> = (0..c.len()).map(|i| (c.positions[i].map(f32::to_bits), c.colors[i], z[i] as u8)).collect();
        v.sort();
        v
    }

    #[test]
    fn packs_in_order_reproduce_the_composed_cloud() {
        let dir = tmpdir("repro");
        let (c, z) = lattice();
        let mut ds = DiffStream::new(dir.clone(), cfg());
        let st = ds.write_event(1.5, 0, "refined", &c, &z).unwrap();
        assert_eq!((st.n_add, st.n_move, st.n_remove, st.points), (c.len(), 0, 0, c.len()));
        // second event: shrink, move some, add new keys
        let keep: Vec<usize> = (0..c.len()).filter(|i| i % 5 != 0).collect();
        let (mut c2, mut z2) = pick(&c, &z, &keep);
        c2.positions[3][1] += 0.5;
        c2.positions[40][2] -= 0.25;
        for j in 0..30u64 {
            c2.positions.push([50.0 + j as f32, 0.0, 1.0]);
            c2.colors.push([1, 2, 3]);
            c2.keys.push(10_000 + j);
            z2.push(2);
        }
        ds.write_event(2.5, 1, "replace", &c2, &z2).unwrap();
        let (slots, _) = replay(&dir);
        assert_eq!(shown(&slots), expect(&c2, &z2));
        let (m, evs) = manifest(&dir);
        assert_eq!(m.get("version").and_then(Json::as_f64), Some(4.0));
        assert!(matches!(m.get("frame"), Some(Json::Str(s)) if s == "enu"));
        assert!(matches!(m.get("origin_gps"), Some(Json::Arr(a)) if a.len() == 3));
        assert_eq!(m.get("capacity").and_then(Json::as_f64), Some(slots.len() as f64));
        assert_eq!(evs[0].pack.as_deref(), Some("packs/e_0001.bin"));
        assert_eq!(evs[0].bytes, c.len() * ADD_BYTES);
        assert_eq!(evs[1].points, c2.len());
        assert_eq!(evs[1].n, [30, 2, c.len() - keep.len()]);
        assert!(!dir.join("manifest.json.tmp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn identical_event_writes_no_pack() {
        let dir = tmpdir("same");
        let (c, z) = lattice();
        let mut ds = DiffStream::new(dir.clone(), cfg());
        ds.write_event(1.0, 0, "refined", &c, &z).unwrap();
        // A sub-threshold wobble is not a change either.
        let mut c2 = c.clone();
        c2.positions[5][0] += 5e-5;
        let st = ds.write_event(2.0, 0, "replace", &c2, &z).unwrap();
        assert_eq!((st.n_add, st.n_move, st.n_remove, st.bytes), (0, 0, 0, 0));
        let (_, evs) = manifest(&dir);
        assert_eq!(evs.len(), 2);
        assert!(evs[1].pack.is_none());
        assert_eq!(evs[1].n, [0, 0, 0]);
        assert_eq!(evs[1].points, c.len());
        assert!(!dir.join("packs/e_0002.bin").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn moving_points_only_moves() {
        let dir = tmpdir("move");
        let (c, z) = lattice();
        let mut ds = DiffStream::new(dir.clone(), cfg());
        ds.write_event(1.0, 0, "refined", &c, &z).unwrap();
        let mut c2 = c.clone();
        for i in [10usize, 200, 101] {
            c2.positions[i][0] += 0.3;
        }
        let st = ds.write_event(2.0, 1, "replace", &c2, &z).unwrap();
        assert_eq!((st.n_add, st.n_move, st.n_remove), (0, 3, 0));
        assert_eq!(st.bytes, 3 * MOVE_BYTES);
        let (slots, _) = replay(&dir);
        assert_eq!(shown(&slots), expect(&c2, &z));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn freed_ids_are_reused_only_in_the_next_event() {
        let dir = tmpdir("reuse");
        let (c, z) = lattice();
        let n = c.len();
        let mut ds = DiffStream::new(dir.clone(), cfg());
        ds.write_event(1.0, 0, "refined", &c, &z).unwrap();
        // event 2: drop the first 10, add 10 new keys
        let (mut c2, mut z2) = pick(&c, &z, &(10..n).collect::<Vec<_>>());
        let new_pt = |c: &mut PointCloud, z: &mut Vec<usize>, key: u64| {
            c.positions.push([40.0 + key as f32, 1.0, 2.0]);
            c.colors.push([9, 9, 9]);
            c.keys.push(key);
            z.push(0);
        };
        for k in 0..10 {
            new_pt(&mut c2, &mut z2, 5000 + k);
        }
        let st = ds.write_event(2.0, 0, "replace", &c2, &z2).unwrap();
        assert_eq!((st.n_add, st.n_remove), (10, 10));
        let (_, ids) = replay(&dir);
        let (added2, removed2) = &ids[1];
        assert!(added2.iter().all(|a| !removed2.contains(a)), "freed id reused in the same event");
        assert!(added2.iter().all(|&a| a as usize >= n), "adds must take new ids");
        // event 3: 10 more keys -> they take the ids freed in event 2 (lowest first)
        let (mut c3, mut z3) = (c2.clone(), z2.clone());
        for k in 0..10 {
            new_pt(&mut c3, &mut z3, 6000 + k);
        }
        let st = ds.write_event(3.0, 0, "replace", &c3, &z3).unwrap();
        assert_eq!((st.n_add, st.n_remove), (10, 0));
        let (slots, ids) = replay(&dir);
        let mut want = removed2.clone();
        want.sort();
        let mut got = ids[2].0.clone();
        got.sort();
        assert_eq!(got, want);
        assert_eq!(ids[2].0, want, "lowest free id first");
        assert_eq!(slots.len(), n + 10, "capacity does not grow when ids are recycled");
        assert_eq!(shown(&slots), expect(&c3, &z3));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn colour_change_is_a_move_with_the_same_id() {
        let dir = tmpdir("colour");
        let (c, z) = lattice();
        let mut ds = DiffStream::new(dir.clone(), cfg());
        ds.write_event(1.0, 0, "refined", &c, &z).unwrap();
        let mut c2 = c.clone();
        c2.colors[7] = [1, 1, 1];
        c2.colors[9] = [2, 2, 2];
        c2.positions[9][0] += 0.2;
        let st = ds.write_event(2.0, 0, "replace", &c2, &z).unwrap();
        assert_eq!((st.n_add, st.n_move, st.n_remove, st.points), (0, 2, 0, c.len()));
        assert_eq!(st.bytes, 2 * MOVE_BYTES);
        let (slots, _) = replay(&dir);
        assert_eq!(slots.len(), c.len());
        assert_eq!(shown(&slots), expect(&c2, &z));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn zone_change_is_remove_plus_add_with_a_new_id() {
        let dir = tmpdir("zone");
        let (c, z) = lattice();
        let mut ds = DiffStream::new(dir.clone(), cfg());
        ds.write_event(1.0, 0, "refined", &c, &z).unwrap();
        let mut z2 = z.clone();
        z2[7] = 5;
        z2[8] = 6;
        let st = ds.write_event(2.0, 0, "replace", &c, &z2).unwrap();
        assert_eq!((st.n_add, st.n_move, st.n_remove, st.points), (2, 0, 2, c.len()));
        let (slots, ids) = replay(&dir);
        assert_eq!(ids[1].1, vec![7, 8]);
        assert!(ids[1].0.iter().all(|&a| a as usize >= c.len()));
        assert_eq!(shown(&slots), expect(&c, &z2));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_clouds_without_keys() {
        let dir = tmpdir("nokeys");
        let mut c = PointCloud::default();
        c.positions.push([0.0; 3]);
        let mut ds = DiffStream::new(dir.clone(), cfg());
        assert!(ds.write_event(0.0, 0, "refined", &c, &[0]).is_err());
        assert!(!dir.join("manifest.json").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
