//! CPU side of the live point cloud: slot buffer, replay timeline and dirty tracking, no GPU.
//!
//! A port of the reference JS viewer (`growTo`, `markDirty`/`flushUpload`, `doRemove`/`doAdd`/
//! `doMove`, `applyParsed`, `applyUndo`, `seek`, `stepBack`, `step`). Points are addressed by
//! dense slot id (`SlotInstance` index). Move animation and fades are done by the GPU from the
//! timestamps in `SlotInstance::tm`; this module only writes them. Every call that changes the
//! slots takes the board clock `now` (seconds) as a parameter; there is no clock in here.
//!
//! Per delta the order is: removes, then adds, then moves. A move of a slot that is not alive is
//! ignored and counted in [`DeltaCloud::ignored_moves`]. An add of an alive slot replaces it.
//!
//! Flow: [`DeltaCloud::ingest`] only records the event (a snapshot resets the timeline and is
//! applied at once); [`DeltaCloud::advance`] applies recorded events while
//! [`DeltaCloud::following`]. The renderer reads [`DeltaCloud::slots`] up to [`DeltaCloud::used`]
//! and uploads [`DeltaCloud::take_dirty`] ranges (or everything when
//! [`DeltaCloud::take_full_upload`] says so).

use crate::slot_cloud::{SlotInstance, T_LONG, T_NEVER};
use skylens_shared::delta::{DeltaError, iter_adds, iter_moves, iter_removes};
use skylens_shared::proto::payload::v1::{DeltaKind, PointsDelta};
use std::collections::HashSet;
use std::ops::Range;
use std::rc::Rc;

/// Seconds a moved point glides from its old to its new position.
pub const MOVE_S: f32 = 0.7;
/// Seconds of the fade in / fade out of an added / removed point.
pub const FADE_S: f32 = 0.15;
/// Points uploaded per frame at most (the budget callers pass to [`DeltaCloud::take_dirty`]).
pub const UPLOAD_BUDGET: usize = 200_000;
/// Ranges closer than this many slots are merged into one upload.
const MERGE_GAP: usize = 64;
/// More queued ranges than this collapse into one covering range.
const MAX_RANGES: usize = 4096;
/// Timestamps beyond this are "never" (alive test).
const T_BIG: f32 = 1.0e8;

fn ease(u: f32) -> f32 {
    if u < 0.5 { 2.0 * u * u } else { 1.0 - (-2.0 * u + 2.0).powi(2) / 2.0 }
}

fn is_alive(s: &SlotInstance) -> bool {
    s.tm[1] < T_BIG && s.tm[2] >= T_BIG
}

/// Previous state of one slot an event touches.
#[derive(Debug, Clone, PartialEq)]
struct Prev {
    id: u32,
    alive: bool,
    pos: [f32; 3],
    rgb: [u8; 3],
    zone: u8,
}

/// What it takes to step an event back: the previous state of every slot it touched.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Undo {
    prev: Vec<Prev>,
}

impl Undo {
    /// Slots the event touched.
    pub fn len(&self) -> usize {
        self.prev.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prev.is_empty()
    }
}

/// One entry of the replay timeline.
#[derive(Debug, Clone)]
pub struct DeltaEvent {
    pub seq: u32,
    pub kind: DeltaKind,
    pub zone: u32,
    /// Model clock, seconds since the first delta.
    pub t_s: f64,
    /// Live points after this delta.
    pub points: u32,
    pub added: u32,
    pub moved: u32,
    pub removed: u32,
    pub snapshot: bool,
    pub delta: Rc<PointsDelta>,
    /// Set once applied; `None` for a snapshot (nothing to step back to) or before applying.
    pub undo: Option<Undo>,
}

/// Outcome of [`DeltaCloud::ingest`].
#[derive(Debug, Clone, PartialEq)]
pub enum Ingest {
    /// Recorded as timeline event `index`, to be applied by `advance`.
    Recorded { index: usize },
    /// A snapshot: the timeline and the slots were reset; it is event 0 and already applied.
    Reset,
    /// `seq` was already ingested; ignored.
    Duplicate,
    /// Malformed (lengths, ids beyond capacity); not recorded.
    Invalid(String),
}

/// Numbers for the HUD.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DeltaStats {
    /// Current event (the last applied one); zeros when none.
    pub moved: u32,
    pub added: u32,
    pub removed: u32,
    /// Live points after the current event.
    pub points: u32,
    /// Highest zone among the applied non-final events.
    pub front_zone: Option<u32>,
    /// Bounding box `(min, max)` of every position seen, for fitting the view.
    pub bbox: Option<([f32; 3], [f32; 3])>,
    /// Board clock at the last apply, `None` before the first.
    pub last_apply_at: Option<f32>,
    /// Model clock of the current event.
    pub t_s: f64,
    /// The current event is the FINAL one.
    pub final_reached: bool,
    /// From the latest ingested delta (progress through the frames); 0 unknown.
    pub positions: u32,
    pub positions_total: u32,
    pub applied: usize,
    pub len: usize,
}

#[derive(Debug)]
pub struct DeltaCloud {
    slots: Vec<SlotInstance>,
    used: usize,
    alive: usize,
    dirty: Vec<Range<usize>>,
    full_upload: bool,
    events: Vec<DeltaEvent>,
    applied: usize,
    /// Apply events as they arrive (default). `step_back`, `step_forward` and `seek` turn it
    /// off; [`DeltaCloud::go_live`] turns it on.
    pub following: bool,
    last_seq: u32,
    /// Sequence gaps seen (applied best-effort).
    pub gaps: u64,
    /// Runs restarted by a delta with seq 1 (a new mission), not counting the first.
    pub runs: u64,
    run: u32,
    /// Duplicate deltas dropped.
    pub duplicates: u64,
    /// Moves of slots that were not alive.
    pub ignored_moves: u64,
    bbox: Option<([f32; 3], [f32; 3])>,
    last_apply_at: Option<f32>,
}

impl Default for DeltaCloud {
    fn default() -> Self {
        DeltaCloud {
            slots: Vec::new(),
            used: 0,
            alive: 0,
            dirty: Vec::new(),
            full_upload: false,
            events: Vec::new(),
            applied: 0,
            following: true,
            last_seq: 0,
            gaps: 0,
            runs: 0,
            run: 0,
            duplicates: 0,
            ignored_moves: 0,
            bbox: None,
            last_apply_at: None,
        }
    }
}

impl DeltaCloud {
    pub fn new() -> Self {
        Self::default()
    }

    // ---- accessors ------------------------------------------------------------------------

    /// All slots, `capacity()` of them. Draw `[..used()]`.
    pub fn slots(&self) -> &[SlotInstance] {
        &self.slots
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// High-water mark: one past the highest slot ever added (the draw range).
    pub fn used(&self) -> usize {
        self.used
    }

    /// Slots currently alive (added and not removed; fading ones count by their final state).
    pub fn alive_count(&self) -> usize {
        self.alive
    }

    pub fn slot(&self, id: u32) -> Option<&SlotInstance> {
        self.slots.get(id as usize)
    }

    pub fn is_alive(&self, id: u32) -> bool {
        self.slot(id).is_some_and(is_alive)
    }

    pub fn events(&self) -> &[DeltaEvent] {
        &self.events
    }

    /// Number of timeline events applied (the index of the next one to apply).
    pub fn applied(&self) -> usize {
        self.applied
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Highest sequence number ingested.
    pub fn last_seq(&self) -> u32 {
        self.last_seq
    }

    /// Whether `advance` has events waiting.
    pub fn pending(&self) -> usize {
        self.events.len() - self.applied
    }

    // ---- storage --------------------------------------------------------------------------

    /// Grows to at least `need` slots (by at least 1.3x). A growth asks for a full upload.
    pub fn ensure_capacity(&mut self, need: usize) {
        let cap = self.slots.len();
        if need <= cap {
            return;
        }
        let n = need.max((cap as f64 * 1.3).ceil() as usize);
        self.slots.resize(n, SlotInstance::EMPTY);
        self.full_upload = true;
        self.dirty.clear();
    }

    fn mark_dirty(&mut self, lo: usize, hi: usize) {
        if self.full_upload {
            return;
        }
        if let Some(l) = self.dirty.last_mut()
            && lo >= l.start
            && lo <= l.end + MERGE_GAP
        {
            l.end = l.end.max(hi);
            return;
        }
        self.dirty.push(lo..hi);
        if self.dirty.len() > MAX_RANGES {
            let a = self.dirty.iter().map(|r| r.start).min().unwrap_or(0);
            let b = self.dirty.iter().map(|r| r.end).max().unwrap_or(0);
            self.dirty.clear();
            self.dirty.push(a..b);
        }
    }

    /// The whole `slots()[..used()]` must be uploaded (capacity grew). Cleared by
    /// [`DeltaCloud::take_full_upload`].
    pub fn needs_full_upload(&self) -> bool {
        self.full_upload
    }

    /// Returns whether a full upload is due and clears the flag.
    pub fn take_full_upload(&mut self) -> bool {
        std::mem::take(&mut self.full_upload)
    }

    /// Slot ranges to upload this frame, at most `budget` slots in total (a range is split to
    /// fit); the rest stays queued. Empty while a full upload is due.
    pub fn take_dirty(&mut self, budget: usize) -> Vec<Range<usize>> {
        let mut out = Vec::new();
        if self.full_upload {
            return out;
        }
        let mut left = budget;
        while left > 0 && !self.dirty.is_empty() {
            let r = self.dirty[0].clone();
            let len = r.end.min(self.slots.len()).saturating_sub(r.start);
            if len <= left {
                if len > 0 {
                    out.push(r.start..r.start + len);
                }
                left -= len;
                self.dirty.remove(0);
            } else {
                out.push(r.start..r.start + left);
                self.dirty[0].start += left;
                left = 0;
            }
        }
        out
    }

    /// Slots still queued for upload.
    pub fn dirty_len(&self) -> usize {
        self.dirty.iter().map(|r| r.len()).sum()
    }

    // ---- slot primitives (JS doRemove / doAdd / doMove) ------------------------------------

    fn remove_slot(&mut self, id: usize, animate: bool, now: f32) {
        let s = &mut self.slots[id];
        if !is_alive(s) {
            return;
        }
        s.tm[2] = if animate { now } else { T_LONG };
        self.alive -= 1;
        self.mark_dirty(id, id + 1);
    }

    fn add_slot(&mut self, id: usize, pos: [f32; 3], rgb: [u8; 3], zone: u8, animate: bool, now: f32) {
        let was = is_alive(&self.slots[id]);
        self.slots[id] = SlotInstance {
            pos,
            from: pos,
            tm: [T_LONG, if animate { now } else { T_LONG }, T_NEVER],
            rgbz: [rgb[0], rgb[1], rgb[2], zone],
        };
        if !was {
            self.alive += 1;
        }
        self.used = self.used.max(id + 1);
        self.grow_bbox(pos);
        self.mark_dirty(id, id + 1);
    }

    /// Returns false when the slot is not alive (nothing done).
    fn move_slot(&mut self, id: usize, pos: [f32; 3], rgb: Option<[u8; 3]>, animate: bool, now: f32) -> bool {
        let s = &mut self.slots[id];
        if !is_alive(s) {
            return false;
        }
        if animate {
            let u = (now - s.tm[0]) / MOVE_S;
            if (0.0..1.0).contains(&u) {
                // a glide still running continues from its in-flight position
                let e = ease(u);
                for (f, p) in s.from.iter_mut().zip(s.pos) {
                    *f += (p - *f) * e;
                }
            } else {
                s.from = s.pos;
            }
            s.tm[0] = now;
        } else {
            s.from = pos;
            s.tm[0] = T_LONG;
        }
        s.pos = pos;
        if let Some(c) = rgb {
            s.rgbz[..3].copy_from_slice(&c); // colour switches instantly
        }
        self.grow_bbox(pos);
        self.mark_dirty(id, id + 1);
        true
    }

    fn grow_bbox(&mut self, p: [f32; 3]) {
        if !p.iter().all(|v| v.is_finite()) {
            return;
        }
        let b = self.bbox.get_or_insert((p, p));
        for (k, v) in p.into_iter().enumerate() {
            b.0[k] = b.0[k].min(v);
            b.1[k] = b.1[k].max(v);
        }
    }

    fn clear_slots(&mut self) {
        for s in &mut self.slots[..self.used] {
            *s = SlotInstance::EMPTY;
        }
        self.alive = 0;
        if self.used > 0 {
            self.mark_dirty(0, self.used);
        }
    }

    // ---- applying events ------------------------------------------------------------------

    /// Applies one delta to the slots; returns the undo log when `want_undo`.
    fn apply_delta(&mut self, d: &PointsDelta, animate: bool, now: f32, want_undo: bool) -> Option<Undo> {
        self.ensure_capacity(d.capacity as usize);
        if d.snapshot {
            self.clear_slots();
        }
        let undo = want_undo.then(|| self.record_prev(d));
        // ids come validated (< capacity); a malformed delta is skipped instead of panicking
        if let Ok(it) = iter_removes(&d.removes) {
            for id in it {
                if (id as usize) < self.slots.len() {
                    self.remove_slot(id as usize, animate, now);
                }
            }
        }
        if let Ok(it) = iter_adds(&d.adds) {
            for a in it {
                if (a.id as usize) < self.slots.len() {
                    self.add_slot(a.id as usize, a.xyz, a.rgb, a.zone, animate, now);
                }
            }
        }
        if let Ok(it) = iter_moves(&d.moves) {
            for m in it {
                if (m.id as usize) >= self.slots.len() || !self.move_slot(m.id as usize, m.xyz, Some(m.rgb), animate, now) {
                    self.ignored_moves += 1;
                }
            }
        }
        undo
    }

    fn record_prev(&self, d: &PointsDelta) -> Undo {
        let mut seen = HashSet::new();
        let mut prev = Vec::new();
        let mut note = |id: u32, slots: &[SlotInstance]| {
            if (id as usize) < slots.len() && seen.insert(id) {
                let s = &slots[id as usize];
                prev.push(Prev { id, alive: is_alive(s), pos: s.pos, rgb: [s.rgbz[0], s.rgbz[1], s.rgbz[2]], zone: s.rgbz[3] });
            }
        };
        if let Ok(it) = iter_removes(&d.removes) {
            for id in it {
                note(id, &self.slots);
            }
        }
        if let Ok(it) = iter_adds(&d.adds) {
            for a in it {
                note(a.id, &self.slots);
            }
        }
        if let Ok(it) = iter_moves(&d.moves) {
            for m in it {
                // a move of a slot that is dead and not added by this delta is ignored: nothing to undo
                if self.is_alive(m.id) {
                    note(m.id, &self.slots);
                }
            }
        }
        Undo { prev }
    }

    /// Inverse of an event: moved points glide back (old colour), added points fade out,
    /// removed points fade back in.
    fn apply_undo(&mut self, u: &Undo, now: f32) {
        for p in &u.prev {
            let id = p.id as usize;
            let cur = is_alive(&self.slots[id]);
            match (p.alive, cur) {
                (true, true) => {
                    self.move_slot(id, p.pos, Some(p.rgb), true, now);
                    self.slots[id].rgbz[3] = p.zone;
                }
                (true, false) => self.add_slot(id, p.pos, p.rgb, p.zone, true, now),
                (false, true) => self.remove_slot(id, true, now),
                (false, false) => {}
            }
        }
    }

    fn apply_event(&mut self, index: usize, animate: bool, now: f32) {
        let d = Rc::clone(&self.events[index].delta);
        let undo = self.apply_delta(&d, animate, now, true);
        self.events[index].undo = if d.snapshot { None } else { undo };
        self.applied = self.applied.max(index + 1);
        self.last_apply_at = Some(now);
    }

    // ---- timeline -------------------------------------------------------------------------

    /// Records a delta. Duplicates (`seq` already ingested) are ignored; a gap is counted and
    /// the delta still recorded; a snapshot resets the timeline and is applied immediately
    /// without animation. Anything else waits for [`DeltaCloud::advance`].
    pub fn ingest(&mut self, d: PointsDelta) -> Ingest {
        if let Err(e) = d.validate() {
            return Ingest::Invalid(DeltaError::to_string(&e));
        }
        // A snapshot only resets a viewer that lacks the state it covers (a late joiner, whose replay
        // starts at the snapshot). A live viewer already holds everything up to its seq and keeps
        // its timeline: the snapshot is a duplicate for it.
        // A different run id: the model started a new run (a new mission).
        // Nothing of the old run may stay on screen, so the timeline restarts like at a snapshot.
        let new_run = d.run != self.run && !self.events.is_empty();
        if new_run {
            self.events.clear();
            self.applied = 0;
            self.clear_slots();
            self.bbox = None;
            self.last_seq = 0;
            self.runs += 1;
        }
        self.run = d.run;
        if d.seq <= self.last_seq && (!d.snapshot || !self.events.is_empty()) {
            self.duplicates += 1;
            return Ingest::Duplicate;
        }
        if !d.snapshot && d.seq > self.last_seq + 1 {
            self.gaps += 1;
        }
        self.last_seq = d.seq;
        let ev = DeltaEvent {
            seq: d.seq,
            kind: d.kind(),
            zone: d.zone,
            t_s: d.t_s,
            points: d.total_points,
            added: (d.adds.len() / skylens_shared::delta::RECORD) as u32,
            moved: (d.moves.len() / skylens_shared::delta::RECORD) as u32,
            removed: (d.removes.len() / skylens_shared::delta::REMOVE_RECORD) as u32,
            snapshot: d.snapshot,
            delta: Rc::new(d),
            undo: None,
        };
        if ev.snapshot {
            self.events.clear();
            self.applied = 0;
            self.clear_slots();
            self.bbox = None;
            self.events.push(ev);
            self.apply_event(0, false, self.last_apply_at.unwrap_or(0.0));
            return Ingest::Reset;
        }
        self.events.push(ev);
        Ingest::Recorded { index: self.events.len() - 1 }
    }

    /// While following, applies every recorded event not yet applied (animated). Returns their
    /// indexes.
    pub fn advance(&mut self, now: f32) -> Vec<usize> {
        if !self.following {
            return Vec::new();
        }
        let todo: Vec<usize> = (self.applied..self.events.len()).collect();
        for &i in &todo {
            self.apply_event(i, true, now);
        }
        todo
    }

    /// Applies the next event with animation; stops following. `None` at the head.
    pub fn step_forward(&mut self, now: f32) -> Option<usize> {
        self.following = false;
        if self.applied >= self.events.len() {
            return None;
        }
        let i = self.applied;
        self.apply_event(i, true, now);
        Some(i)
    }

    /// Un-applies the current event with animation; stops following. False when there is
    /// nothing to step back to (no event, or the current one is the snapshot).
    pub fn step_back(&mut self, now: f32) -> bool {
        self.following = false;
        if self.applied == 0 {
            return false;
        }
        let i = self.applied - 1;
        if self.events[i].snapshot {
            return false;
        }
        match self.events[i].undo.take() {
            Some(u) => {
                self.apply_undo(&u, now);
                self.applied = i;
                self.last_apply_at = Some(now);
                true
            }
            None => {
                self.seek(i, now); // no log (should not happen): instant rebuild
                true
            }
        }
    }

    /// Applies everything still pending (earlier events snap, the last one animates) and
    /// follows the stream again. Returns the applied indexes.
    pub fn go_live(&mut self, now: f32) -> Vec<usize> {
        let todo: Vec<usize> = (self.applied..self.events.len()).collect();
        for (k, &i) in todo.iter().enumerate() {
            self.apply_event(i, k + 1 == todo.len(), now);
        }
        self.following = true;
        todo
    }

    /// Instant rebuild of the state after `n` events (clamped): clears and replays deltas
    /// `0..n` without animation, rebuilding their undo logs. Stops following.
    pub fn seek(&mut self, n: usize, now: f32) {
        let n = n.min(self.events.len());
        let ignored = self.ignored_moves;
        self.clear_slots();
        for e in &mut self.events {
            e.undo = None;
        }
        self.applied = 0;
        for i in 0..n {
            self.apply_event(i, false, now);
        }
        self.applied = n;
        self.ignored_moves = ignored;
        self.following = false;
        self.last_apply_at = Some(now);
        if self.used > 0 {
            self.mark_dirty(0, self.used);
        }
    }

    // ---- stats ----------------------------------------------------------------------------

    pub fn stats(&self) -> DeltaStats {
        let cur = self.applied.checked_sub(1).and_then(|i| self.events.get(i));
        let latest = self.events.last();
        DeltaStats {
            moved: cur.map_or(0, |e| e.moved),
            added: cur.map_or(0, |e| e.added),
            removed: cur.map_or(0, |e| e.removed),
            points: cur.map_or(0, |e| e.points),
            front_zone: self.events[..self.applied]
                .iter()
                .filter(|e| e.kind != DeltaKind::Final)
                .map(|e| e.zone)
                .max(),
            bbox: self.bbox,
            last_apply_at: self.last_apply_at,
            t_s: cur.map_or(0.0, |e| e.t_s),
            final_reached: cur.is_some_and(|e| e.kind == DeltaKind::Final),
            positions: latest.map_or(0, |e| e.delta.positions),
            positions_total: latest.map_or(0, |e| e.delta.positions_total),
            applied: self.applied,
            len: self.events.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skylens_shared::delta::{AddRec, MoveRec, pack_adds, pack_moves, pack_removes};
    use std::collections::HashMap;

    fn add(id: u32, x: f32, zone: u8) -> AddRec {
        AddRec { id, xyz: [x, x + 1.0, x + 2.0], rgb: [id as u8, 2, 3], zone }
    }

    fn delta(seq: u32, adds: &[AddRec], moves: &[MoveRec], removes: &[u32]) -> PointsDelta {
        PointsDelta {
            seq,
            kind: DeltaKind::Refined as i32,
            zone: seq,
            capacity: 64,
            total_points: 0,
            adds: pack_adds(adds),
            moves: pack_moves(moves),
            removes: pack_removes(removes),
            t_s: f64::from(seq),
            ..Default::default()
        }
    }

    fn snap(seq: u32, adds: &[AddRec]) -> PointsDelta {
        PointsDelta { snapshot: true, ..delta(seq, adds, &[], &[]) }
    }

    fn mv(id: u32, x: f32) -> MoveRec {
        MoveRec { id, xyz: [x, 0.0, 0.0], rgb: [9, 9, 9] }
    }

    type State = HashMap<u32, ([f32; 3], [u8; 3], u8)>;

    fn state(c: &DeltaCloud) -> State {
        (0..c.capacity() as u32)
            .filter(|&i| c.is_alive(i))
            .map(|i| {
                let s = c.slot(i).unwrap();
                (i, (s.pos, [s.rgbz[0], s.rgbz[1], s.rgbz[2]], s.rgbz[3]))
            })
            .collect()
    }

    #[test]
    fn apply_order_removes_adds_moves() {
        let mut c = DeltaCloud::new();
        c.ingest(delta(1, &[add(1, 1.0, 0), add(2, 2.0, 0)], &[], &[]));
        c.advance(0.0);
        // id 1 removed then re-added in the same delta ends alive; id 2 moved
        c.ingest(delta(2, &[add(1, 5.0, 3)], &[mv(2, 7.0)], &[1]));
        c.advance(1.0);
        assert!(c.is_alive(1));
        assert_eq!(c.slot(1).unwrap().pos, [5.0, 6.0, 7.0]);
        assert_eq!(c.slot(1).unwrap().rgbz[3], 3);
        assert_eq!(c.slot(2).unwrap().pos, [7.0, 0.0, 0.0]);
        assert_eq!(c.slot(2).unwrap().rgbz[..3], [9, 9, 9]);
        assert_eq!(c.alive_count(), 2);
    }

    #[test]
    fn move_of_dead_id_is_ignored() {
        let mut c = DeltaCloud::new();
        c.ingest(delta(1, &[], &[mv(3, 1.0)], &[]));
        c.advance(0.0);
        assert!(!c.is_alive(3));
        assert_eq!(c.ignored_moves, 1);
    }

    #[test]
    fn seq_one_again_starts_a_new_run_with_an_empty_buffer() {
        let mut c = DeltaCloud::new();
        let run = |mut d: PointsDelta, r: u32| {
            d.run = r;
            d
        };
        c.ingest(run(delta(1, &[add(0, 1.0, 0)], &[], &[]), 7));
        c.ingest(run(delta(2, &[add(1, 2.0, 0)], &[], &[]), 7));
        c.advance(0.0);
        assert_eq!(c.alive_count(), 2);
        assert_eq!(c.ingest(run(delta(1, &[add(0, 1.0, 0)], &[], &[]), 7)), Ingest::Duplicate, "same run: a duplicate");
        c.ingest(run(delta(1, &[add(0, 9.0, 0)], &[], &[]), 8));
        assert_eq!(c.runs, 1);
        c.advance(5.0);
        assert_eq!((c.len(), c.alive_count()), (1, 1), "old run is gone");
        assert!(!c.is_alive(1));
    }

    #[test]
    fn duplicate_gap_snapshot() {
        let mut c = DeltaCloud::new();
        assert!(matches!(c.ingest(delta(1, &[add(0, 0.0, 0)], &[], &[])), Ingest::Recorded { index: 0 }));
        assert_eq!(c.ingest(delta(1, &[], &[], &[])), Ingest::Duplicate);
        assert_eq!(c.ingest(delta(0, &[], &[], &[])), Ingest::Duplicate);
        assert!(matches!(c.ingest(delta(4, &[add(1, 0.0, 0)], &[], &[])), Ingest::Recorded { index: 1 }));
        assert_eq!((c.gaps, c.duplicates), (1, 2));
        c.advance(0.0);
        assert_eq!(c.alive_count(), 2);
        // snapshot resets
        assert_eq!(c.ingest(snap(9, &[add(5, 1.0, 0)])), Ingest::Reset);
        assert_eq!(c.len(), 1);
        assert_eq!(c.applied(), 1);
        assert_eq!(c.alive_count(), 1);
        assert!(c.is_alive(5) && !c.is_alive(0));
        assert_eq!(c.slot(5).unwrap().tm, [T_LONG, T_LONG, T_NEVER]);
        assert_eq!(c.last_seq(), 9);
        assert_eq!(c.ingest(delta(9, &[], &[], &[])), Ingest::Duplicate);
        assert!(matches!(c.ingest(delta(10, &[], &[], &[])), Ingest::Recorded { index: 1 }));
    }

    #[test]
    fn invalid_delta_is_rejected() {
        let mut c = DeltaCloud::new();
        let mut d = delta(1, &[], &[], &[]);
        d.removes = vec![0; 3];
        assert!(matches!(c.ingest(d), Ingest::Invalid(_)));
        assert!(c.is_empty());
    }

    #[test]
    fn ingest_waits_for_advance_and_following() {
        let mut c = DeltaCloud::new();
        c.ingest(delta(1, &[add(0, 0.0, 0)], &[], &[]));
        assert_eq!(c.applied(), 0);
        c.following = false;
        assert!(c.advance(0.0).is_empty());
        c.following = true;
        assert_eq!(c.advance(2.0), vec![0]);
        assert_eq!(c.slot(0).unwrap().tm, [T_LONG, 2.0, T_NEVER]);
        assert_eq!(c.stats().last_apply_at, Some(2.0));
    }

    #[test]
    fn glide_continues_from_in_flight_position() {
        let mut c = DeltaCloud::new();
        c.ingest(delta(1, &[AddRec { id: 0, xyz: [0.0; 3], rgb: [1, 1, 1], zone: 0 }], &[], &[]));
        c.advance(0.0);
        c.ingest(delta(2, &[], &[mv(0, 10.0)], &[]));
        c.advance(10.0);
        let s = *c.slot(0).unwrap();
        assert_eq!((s.from, s.pos, s.tm[0]), ([0.0; 3], [10.0, 0.0, 0.0], 10.0));
        // half way through (u = 0.5 -> eased 0.5)
        c.ingest(delta(3, &[], &[mv(0, 20.0)], &[]));
        c.advance(10.0 + MOVE_S / 2.0);
        let s = *c.slot(0).unwrap();
        assert!((s.from[0] - 5.0).abs() < 1e-4, "{:?}", s.from);
        assert_eq!(s.pos[0], 20.0);
        // a finished glide restarts from the old target
        c.ingest(delta(4, &[], &[mv(0, 30.0)], &[]));
        c.advance(100.0);
        assert_eq!(c.slot(0).unwrap().from[0], 20.0);
    }

    #[test]
    fn colour_switches_instantly_on_move() {
        let mut c = DeltaCloud::new();
        c.ingest(delta(1, &[add(0, 0.0, 4)], &[], &[]));
        c.advance(0.0);
        c.ingest(delta(2, &[], &[mv(0, 1.0)], &[]));
        c.advance(1.0);
        assert_eq!(c.slot(0).unwrap().rgbz, [9, 9, 9, 4]);
    }

    fn rng(seed: &mut u64) -> u32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 16) as u32
    }

    /// Pseudo-random delta stream over `ids` slots plus the reference state after each delta.
    fn random_stream(n: u32, ids: u32, seed: u64) -> (Vec<PointsDelta>, Vec<State>) {
        let mut s = seed;
        let mut r = State::new();
        let (mut out, mut states) = (Vec::new(), Vec::new());
        for seq in 1..=n {
            let (mut rm, mut ad, mut mo) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..rng(&mut s) % 6 {
                rm.push(rng(&mut s) % ids);
            }
            for _ in 0..rng(&mut s) % 8 {
                let id = rng(&mut s) % ids;
                let x = (rng(&mut s) % 1000) as f32;
                ad.push(AddRec { id, xyz: [x, x * 0.5, 1.0], rgb: [(x as u32 % 256) as u8, 1, 2], zone: (seq % 7) as u8 });
            }
            for _ in 0..rng(&mut s) % 8 {
                mo.push(MoveRec { id: rng(&mut s) % ids, xyz: [(rng(&mut s) % 1000) as f32, 5.0, 6.0], rgb: [rng(&mut s) as u8, 7, 8] });
            }
            // reference semantics
            for id in &rm {
                r.remove(id);
            }
            for a in &ad {
                r.insert(a.id, (a.xyz, a.rgb, a.zone));
            }
            for m in &mo {
                if let Some(e) = r.get_mut(&m.id) {
                    e.0 = m.xyz;
                    e.1 = m.rgb;
                }
            }
            out.push(delta(seq, &ad, &mo, &rm));
            states.push(r.clone());
        }
        (out, states)
    }

    #[test]
    fn equals_naive_reference() {
        let (ds, states) = random_stream(200, 40, 0x9e3779b97f4a7c15);
        let mut c = DeltaCloud::new();
        let mut t = 0.0;
        for (i, d) in ds.into_iter().enumerate() {
            c.ingest(d);
            t += 0.01; // faster than the glide: continuations are exercised
            c.advance(t);
            if i % 17 == 0 {
                assert_eq!(state(&c), states[i], "after delta {i}");
                assert_eq!(c.alive_count(), states[i].len());
            }
        }
        assert_eq!(state(&c), *states.last().unwrap());
        assert_eq!(c.alive_count(), states.last().unwrap().len());
    }

    #[test]
    fn seek_equals_sequential_apply() {
        let (ds, states) = random_stream(60, 30, 7);
        let mut a = DeltaCloud::new();
        for d in ds {
            a.ingest(d);
        }
        a.advance(0.0);
        for n in [0usize, 1, 13, 59, 60, 30] {
            a.seek(n, 100.0);
            assert!(!a.following);
            assert_eq!(a.applied(), n);
            let want = if n == 0 { State::new() } else { states[n - 1].clone() };
            assert_eq!(state(&a), want, "seek {n}");
            assert_eq!(a.alive_count(), want.len());
        }
    }

    #[test]
    fn undo_round_trip() {
        let (ds, states) = random_stream(80, 30, 99);
        let mut c = DeltaCloud::new();
        for d in ds {
            c.ingest(d);
        }
        let mut now = 0.0;
        c.advance(now);
        for k in (0..80).rev() {
            now += 10.0; // far beyond any glide/fade
            assert!(c.step_back(now));
            assert_eq!(c.applied(), k);
            let want = if k == 0 { State::new() } else { states[k - 1].clone() };
            assert_eq!(state(&c), want, "back to {k}");
            assert_eq!(c.alive_count(), want.len());
        }
        assert!(!c.step_back(now));
        // and forward again
        for (k, want) in states.iter().enumerate() {
            now += 10.0;
            assert_eq!(c.step_forward(now), Some(k));
            assert_eq!(state(&c), *want);
        }
        assert_eq!(c.step_forward(now), None);
        assert!(!c.following);
        let live = c.go_live(now + 1.0);
        assert!(live.is_empty() && c.following);
    }

    #[test]
    fn step_back_stops_at_snapshot_and_go_live_catches_up() {
        let mut c = DeltaCloud::new();
        c.ingest(snap(5, &[add(0, 1.0, 0)]));
        c.ingest(delta(6, &[add(1, 1.0, 1)], &[], &[]));
        c.ingest(delta(7, &[add(2, 1.0, 2)], &[], &[]));
        c.advance(1.0);
        assert!(c.step_back(2.0));
        assert!(c.step_back(3.0));
        assert!(!c.step_back(4.0)); // snapshot: no-op
        assert_eq!((c.applied(), c.alive_count()), (1, 1));
        c.ingest(delta(8, &[add(3, 1.0, 3)], &[], &[]));
        assert!(c.advance(5.0).is_empty()); // not following
        assert_eq!(c.go_live(6.0), vec![1, 2, 3]);
        assert!(c.following);
        assert_eq!(c.alive_count(), 4);
        // earlier events snapped, the last one animates
        assert_eq!(c.slot(1).unwrap().tm[1], T_LONG);
        assert_eq!(c.slot(3).unwrap().tm[1], 6.0);
    }

    #[test]
    fn step_back_animates_inverse() {
        let mut c = DeltaCloud::new();
        c.ingest(delta(1, &[add(0, 0.0, 0), add(1, 0.0, 0)], &[], &[]));
        c.ingest(delta(2, &[add(2, 0.0, 0)], &[mv(0, 4.0)], &[1]));
        c.advance(0.0);
        assert!(c.step_back(10.0));
        let (s0, s1, s2) = (*c.slot(0).unwrap(), *c.slot(1).unwrap(), *c.slot(2).unwrap());
        assert_eq!(s0.pos, [0.0, 1.0, 2.0]); // glides back
        assert_eq!(s0.tm[0], 10.0);
        assert_eq!(s0.rgbz[..3], [0, 2, 3]); // old colour
        assert_eq!(s2.tm[2], 10.0); // added point fades out
        assert!(!c.is_alive(2));
        assert!(c.is_alive(1) && s1.tm[1] == 10.0); // removed point fades back in
        assert!(!c.following);
    }

    #[test]
    fn capacity_growth_and_full_upload() {
        let mut c = DeltaCloud::new();
        let mut d = delta(1, &[add(3, 0.0, 0)], &[], &[]);
        d.capacity = 10;
        c.ingest(d);
        c.advance(0.0);
        assert_eq!(c.capacity(), 10);
        assert_eq!(c.used(), 4);
        assert!(c.needs_full_upload());
        assert!(c.take_dirty(100).is_empty());
        assert!(c.take_full_upload());
        assert!(!c.needs_full_upload());
        let mut d = delta(2, &[add(10, 0.0, 0)], &[], &[]);
        d.capacity = 11;
        c.ingest(d);
        c.advance(0.0);
        assert_eq!(c.capacity(), 13); // ceil(10 * 1.3)
        assert!(c.take_full_upload());
        assert_eq!(c.used(), 11);
        assert_eq!(c.slots()[10].tm[1], 0.0);
        assert_eq!(c.slots()[5], SlotInstance::EMPTY);
    }

    #[test]
    fn dirty_ranges_merge_and_respect_budget() {
        let mut c = DeltaCloud::new();
        c.ensure_capacity(100_000);
        c.take_full_upload();
        c.mark_dirty(10, 11);
        c.mark_dirty(60, 61); // gap 49 <= 64: merged
        c.mark_dirty(500, 501); // separate
        assert_eq!(c.take_dirty(1000), vec![10..61, 500..501]);
        assert_eq!(c.dirty_len(), 0);
        c.mark_dirty(0, 1000);
        c.mark_dirty(5000, 5100);
        assert_eq!(c.take_dirty(400), vec![0..400]);
        assert_eq!(c.take_dirty(700), vec![400..1000, 5000..5100]);
        assert!(c.take_dirty(10).is_empty());
        // too many ranges collapse
        for i in 0..5000 {
            c.mark_dirty(i * 1000, i * 1000 + 1);
        }
        assert!(c.dirty.len() <= MAX_RANGES);
        assert!(c.take_dirty(usize::MAX).iter().map(|r| r.len()).sum::<usize>() > 0);
    }

    #[test]
    fn stats_track_current_event() {
        let mut c = DeltaCloud::new();
        let mut d1 = delta(1, &[add(0, -1.0, 0), add(1, 3.0, 0)], &[], &[]);
        d1.total_points = 2;
        d1.positions = 3;
        d1.positions_total = 10;
        let mut d2 = delta(2, &[], &[mv(0, 8.0)], &[1]);
        d2.zone = 4;
        d2.total_points = 1;
        let mut d3 = delta(3, &[], &[], &[]);
        d3.kind = DeltaKind::Final as i32;
        d3.zone = 9;
        c.ingest(d1);
        c.ingest(d2);
        c.ingest(d3);
        assert_eq!(c.stats().applied, 0);
        c.advance(2.5);
        let s = c.stats();
        assert_eq!((s.applied, s.len), (3, 3));
        assert!(s.final_reached);
        assert_eq!(s.front_zone, Some(4)); // the final delta does not count
        assert_eq!(s.last_apply_at, Some(2.5));
        assert_eq!(s.bbox, Some(([-1.0, 0.0, 0.0], [8.0, 4.0, 5.0])));
        assert_eq!((s.positions, s.positions_total), (0, 0)); // latest delta has none
        c.step_back(3.0);
        let s = c.stats();
        assert!(!s.final_reached);
        assert_eq!((s.moved, s.removed, s.points, s.t_s), (1, 1, 1, 2.0));
    }
}
