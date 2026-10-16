//! The board's record of the live point stream: every slot message of the current run in seq
//! order, a cursor saying how many are applied to the point layer, and the commands the timeline
//! panel uses (seek, step, follow live). Pure Rust, no GPU: the point layers drain [`LayerOp`]s.
//!
//! The input is the point slot stream of the core (`CHUNK_KIND_POINT_SLOTS`: pointblitz slot
//! messages, one per cumulus3d display event, `generation` = run id, `version` = seq, `segment`
//! = the zone that caused the event). A slot message carries neither a kind nor a clock, so the
//! log derives them: an event that only adds is `REFINED`, one that moves or removes points is
//! `REPLACE`, and an empty event (no adds, moves or removes) is the producer's `FINAL` marker.
//! `t_ms` is the board's own clock: milliseconds from the run's first message to the arrival of
//! this one (a late joiner gets the replayed history within moments, so its early rows are
//! close together).
//!
//! Rules: a message whose run differs from the log's run starts a new run (log and layer are
//! cleared first, [`LayerOp::Reset`]). Messages are stored in seq order; a repeated or older seq
//! is ignored. A late joiner receives the stored messages from seq 1 and they are applied in
//! order. While following live (the default) each new message is applied as it arrives,
//! animated; stepping back or seeking leaves live mode until [`DeltaLog::follow_live`].

use skylens_viewer::live_slots::{AddRec, Batch, MoveRec, RemoveRec};
use skylens_viewer::pointblitz::io::slots::{self, Diff};
use std::rc::Rc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Refined,
    Replace,
    Final,
}

/// One row of the timeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogEntry {
    pub seq: u32,
    pub run_id: u32,
    pub kind: Kind,
    pub zone: u32,
    /// Board clock: milliseconds from the run's first message to this one's arrival.
    pub t_ms: f64,
    pub n_add: u32,
    pub n_move: u32,
    pub n_remove: u32,
    /// Points shown after this event.
    pub total_points: u32,
    /// Frame positions ingested so far / in total (0 unknown), for the progress bar. The slot
    /// stream does not carry them.
    pub positions: u32,
    pub positions_total: u32,
}

/// What the point layer must do, in order (see [`DeltaLog::ops_since`]).
#[derive(Debug, Clone, PartialEq)]
pub enum LayerOp {
    /// Empty every slot (new run, or the start of a seek replay).
    Reset,
    Apply(Rc<Batch>),
}

/// A live point of the slot table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Slot {
    pub xyz: [f32; 3],
    pub rgb: [u8; 3],
    pub zone: u8,
}

/// The decoded records of one message.
#[derive(Debug, Clone, Default, PartialEq)]
struct Delta {
    removes: Vec<u32>,
    adds: Vec<AddRec>,
    moves: Vec<(u32, [f32; 3], [u8; 3])>,
    capacity: u32,
}

impl Delta {
    fn parse(msg: &[u8]) -> Option<Delta> {
        let d = Diff::parse(msg).ok()?;
        Some(Delta {
            removes: d.remove_ids().collect(),
            adds: d.add_records().map(|r| AddRec { id: r.id, xyz: r.position, rgb: r.color, zone: r.zone }).collect(),
            moves: d.move_records().map(|r| (r.id, r.position, r.color)).collect(),
            capacity: d.header.capacity,
        })
    }
}

/// The CPU copy of the point buffer: slot `id` holds a point or is free. Same rules as
/// cumulus3d's `DisplayMirror` (removes, adds, moves); it also makes the inverse of every event.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SlotTable {
    pub slots: Vec<Option<Slot>>,
}

impl SlotTable {
    pub fn len(&self) -> usize {
        self.slots.iter().flatten().count()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(Option::is_none)
    }

    pub fn clear(&mut self) {
        self.slots.clear();
    }

    /// Applies an event; returns the batch that shows it and the batch that undoes it.
    fn apply(&mut self, d: &Delta, animate: bool, undo_animate: bool) -> (Batch, Batch) {
        if self.slots.len() < d.capacity as usize {
            self.slots.resize(d.capacity as usize, None);
        }
        let mut fwd = Batch { capacity: d.capacity, animate, ..Default::default() };
        let mut inv = Batch { capacity: d.capacity, animate: undo_animate, ..Default::default() };
        for &id in &d.removes {
            if let Some(Some(old)) = self.slots.get(id as usize).copied() {
                fwd.removes.push(RemoveRec { id, xyz: old.xyz, rgb: old.rgb, zone: old.zone });
                inv.adds.push(AddRec { id, xyz: old.xyz, rgb: old.rgb, zone: old.zone });
                self.slots[id as usize] = None;
            }
        }
        let mut added = std::collections::HashSet::with_capacity(d.adds.len());
        for a in &d.adds {
            let i = a.id as usize;
            if i >= self.slots.len() {
                self.slots.resize(i + 1, None);
            }
            if let Some(old) = self.slots[i] {
                // An add into a live slot (a replace without a remove): the undo puts the old one back.
                inv.adds.push(AddRec { id: a.id, xyz: old.xyz, rgb: old.rgb, zone: old.zone });
            }
            inv.removes.push(RemoveRec { id: a.id, xyz: a.xyz, rgb: a.rgb, zone: a.zone });
            fwd.adds.push(*a);
            self.slots[i] = Some(Slot { xyz: a.xyz, rgb: a.rgb, zone: a.zone });
            added.insert(a.id);
        }
        for &(id, xyz, rgb) in &d.moves {
            if let Some(Some(p)) = self.slots.get_mut(id as usize) {
                if !added.contains(&id) {
                    fwd.moves.push(MoveRec { id, from: p.xyz, from_rgb: p.rgb, xyz, rgb, zone: p.zone });
                    inv.moves.push(MoveRec { id, from: xyz, from_rgb: rgb, xyz: p.xyz, rgb: p.rgb, zone: p.zone });
                }
                p.xyz = xyz;
                p.rgb = rgb;
            }
        }
        (fwd, inv)
    }

    /// The whole table as one snapping batch.
    fn snapshot(&self) -> Batch {
        let adds = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.map(|s| AddRec { id: i as u32, xyz: s.xyz, rgb: s.rgb, zone: s.zone }))
            .collect();
        Batch { adds, capacity: self.slots.len() as u32, animate: false, ..Default::default() }
    }

    /// Per zone: up to `max` positions spread evenly over its live points, and the live count.
    pub fn zone_samples(&self, max: usize) -> std::collections::BTreeMap<u8, (Vec<[f32; 3]>, u64)> {
        let mut count = std::collections::BTreeMap::<u8, u64>::new();
        for s in self.slots.iter().flatten() {
            *count.entry(s.zone).or_default() += 1;
        }
        let mut out: std::collections::BTreeMap<u8, (Vec<[f32; 3]>, u64)> =
            count.iter().map(|(z, n)| (*z, (Vec::with_capacity(max.min(*n as usize)), *n))).collect();
        let mut seen = std::collections::BTreeMap::<u8, u64>::new();
        for s in self.slots.iter().flatten() {
            let i = seen.entry(s.zone).or_default();
            let (v, n) = out.get_mut(&s.zone).expect("counted");
            // Keep the points whose running index crosses a multiple of n / max.
            if (*i * max as u64) / *n != ((*i + 1) * max as u64) / *n || *n <= max as u64 {
                v.push(s.xyz);
            }
            *i += 1;
        }
        out
    }
}

/// The producer's own words about a slot message (`Chunk.slot_kind`: 1 refined, 2 replace,
/// 3 final, 0 unknown; `slot_t_ms`: ms since the run's start).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SlotMeta {
    pub kind: u32,
    pub t_ms: u32,
}

/// What [`DeltaLog::push`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pushed {
    /// Stored as the next message.
    Stored,
    /// Stored, and it started a new run (everything older was dropped).
    NewRun,
    /// Older, repeated or invalid, ignored.
    Ignored,
}

#[derive(Default)]
pub struct DeltaLog {
    run_id: Option<u32>,
    entries: Vec<LogEntry>,
    msgs: Vec<Rc<[u8]>>,
    /// Inverse of each applied event (index < applied).
    undo: Vec<Batch>,
    applied: usize,
    live: bool,
    table: SlotTable,
    ops: Vec<LayerOp>,
    /// Number of ops trimmed off the front of `ops` (cursors count from the start of the run of ops).
    ops_base: u64,
    version: u64,
    /// Points shown after the last stored message (a running count, whatever is applied).
    stored_total: i64,
    /// Board clock of the run's first message.
    t0: f64,
}

impl DeltaLog {
    pub fn new() -> Self {
        DeltaLog { live: true, ..Default::default() }
    }

    /// Takes one slot message of run `run` (animated when following live). `zone` is the chunk's
    /// segment, `t_ms` the board clock in milliseconds.
    pub fn push(&mut self, run: u32, zone: u32, msg: &[u8], t_ms: f64) -> Pushed {
        self.push_meta(run, zone, msg, t_ms, SlotMeta::default())
    }

    /// Like [`DeltaLog::push`] with what the producer said (`Chunk.slot_kind`, `slot_t_ms`):
    /// a non-zero kind is used as it is and then `slot_t_ms` is the row's clock; zeros fall back
    /// to the content and the arrival clock (old recordings, producers without the fields).
    pub fn push_meta(&mut self, run: u32, zone: u32, msg: &[u8], t_ms: f64, meta: SlotMeta) -> Pushed {
        let Ok(h) = slots::validate(msg) else { return Pushed::Ignored };
        let new_run = self.run_id.is_some_and(|r| r != run);
        if new_run {
            self.clear();
        } else if self.entries.last().is_some_and(|e| h.seq <= e.seq) {
            return Pushed::Ignored;
        }
        let first = self.entries.is_empty();
        if first {
            self.run_id = Some(run);
            self.t0 = t_ms;
        }
        self.stored_total += i64::from(h.n_add) - i64::from(h.n_remove);
        let kind = match meta.kind {
            1 => Kind::Refined,
            2 => Kind::Replace,
            3 => Kind::Final,
            _ if h.n_add + h.n_move + h.n_remove == 0 => Kind::Final,
            _ if h.n_move + h.n_remove > 0 => Kind::Replace,
            _ => Kind::Refined,
        };
        let row_t = if meta.kind != 0 { f64::from(meta.t_ms) } else { (t_ms - self.t0).max(0.0) };
        let entry = LogEntry {
            seq: h.seq,
            run_id: run,
            kind,
            zone,
            t_ms: row_t,
            n_add: h.n_add,
            n_move: h.n_move,
            n_remove: h.n_remove,
            total_points: self.stored_total.max(0) as u32,
            positions: 0,
            positions_total: 0,
        };
        let at_end = self.applied == self.entries.len();
        self.entries.push(entry);
        self.msgs.push(Rc::from(msg));
        if self.live && at_end {
            self.apply_next(true);
        }
        if new_run { Pushed::NewRun } else { Pushed::Stored }
    }

    fn clear(&mut self) {
        self.run_id = None;
        self.entries.clear();
        self.msgs.clear();
        self.undo.clear();
        self.table.clear();
        self.applied = 0;
        self.version += 1;
        self.live = true;
        self.stored_total = 0;
        self.reset_ops();
    }

    /// Forgets the run (the link reconnected: the core replays from the start).
    pub fn reset(&mut self) {
        self.clear();
    }

    /// Applies entry `applied` and moves the cursor.
    fn apply_next(&mut self, animate: bool) {
        let d = Delta::parse(&self.msgs[self.applied]).expect("validated on push");
        let (fwd, inv) = self.table.apply(&d, animate, true);
        self.undo.push(inv);
        if !fwd.is_empty() || fwd.capacity > 0 {
            self.ops.push(LayerOp::Apply(Rc::new(fwd)));
        }
        self.applied += 1;
        self.version += 1;
    }

    /// All stored events' rows, oldest first (index 0 is the run's first message).
    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    /// Number of stored messages.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of events applied to the point layer: entries `0..applied()` are on screen.
    pub fn applied(&self) -> usize {
        self.applied
    }

    /// The entry on screen now (`applied() - 1`), if any.
    pub fn current(&self) -> Option<&LogEntry> {
        self.applied.checked_sub(1).and_then(|i| self.entries.get(i))
    }

    pub fn run_id(&self) -> Option<u32> {
        self.run_id
    }

    /// True once the last stored message is a FINAL one.
    pub fn is_final(&self) -> bool {
        self.entries.last().is_some_and(|e| e.kind == Kind::Final)
    }

    /// Whether new messages are applied as they arrive.
    pub fn is_live(&self) -> bool {
        self.live
    }

    /// Counts every change of what is on screen (cache key for derived data).
    pub fn version(&self) -> u64 {
        self.version
    }

    /// The slot table as it is on screen (tests, picking, bounds).
    pub fn table(&self) -> &SlotTable {
        &self.table
    }

    /// Jumps to the state after message `seq` (no animation: the layer is emptied and the events
    /// replayed). A seq beyond the end goes to the end; reaching the end resumes live following.
    pub fn seek(&mut self, seq: u32) {
        if self.entries.is_empty() {
            return;
        }
        let target = self.entries.iter().rposition(|e| e.seq <= seq).map_or(0, |i| i + 1);
        self.rebuild(target);
        self.live = target == self.entries.len();
    }

    fn rebuild(&mut self, target: usize) {
        self.table.clear();
        self.undo.clear();
        self.applied = 0;
        self.version += 1;
        self.reset_ops();
        while self.applied < target {
            self.apply_next(false);
        }
    }

    /// Stops following new messages and stays on the current event (pause).
    pub fn unfollow(&mut self) {
        self.live = false;
    }

    /// Undoes the last applied event with the inverse animation. Leaves live mode.
    pub fn step_back(&mut self) {
        self.live = false;
        if self.applied == 0 {
            return;
        }
        let inv = self.undo.pop().expect("one undo per applied event");
        self.table_apply_inverse(&inv);
        self.ops.push(LayerOp::Apply(Rc::new(inv)));
        self.applied -= 1;
        self.version += 1;
    }

    /// Applies the inverse batch to the table.
    fn table_apply_inverse(&mut self, inv: &Batch) {
        for r in &inv.removes {
            if let Some(s) = self.table.slots.get_mut(r.id as usize) {
                *s = None;
            }
        }
        for a in &inv.adds {
            let i = a.id as usize;
            if i >= self.table.slots.len() {
                self.table.slots.resize(i + 1, None);
            }
            self.table.slots[i] = Some(Slot { xyz: a.xyz, rgb: a.rgb, zone: a.zone });
        }
        for m in &inv.moves {
            if let Some(Some(p)) = self.table.slots.get_mut(m.id as usize) {
                p.xyz = m.xyz;
                p.rgb = m.rgb;
            }
        }
    }

    /// Applies the next stored event with animation. Reaching the end resumes live following.
    pub fn step_forward(&mut self) {
        if self.applied < self.entries.len() {
            self.apply_next(true);
        }
        self.live = self.applied == self.entries.len();
    }

    /// Applies everything not yet applied (animated) and applies future messages on arrival.
    pub fn follow_live(&mut self) {
        while self.applied < self.entries.len() {
            self.apply_next(true);
        }
        self.live = true;
    }

    /// Everything before a Reset is moot: drop the pending ops and start over with a Reset.
    fn reset_ops(&mut self) {
        self.ops_base += self.ops.len() as u64;
        self.ops.clear();
        self.ops.push(LayerOp::Reset);
    }

    /// Operations for a point layer since its `cursor`. A new layer (`None`) gets the whole
    /// present state in one non-animated batch; after that each layer keeps its cursor, so the
    /// main view and the point panel both follow. A layer that fell behind a Reset starts from it.
    pub fn ops_since(&self, cursor: &mut Option<u64>) -> Vec<LayerOp> {
        let end = self.ops_base + self.ops.len() as u64;
        let out = match *cursor {
            Some(c) => self.ops.get(c.saturating_sub(self.ops_base) as usize..).map(<[_]>::to_vec).unwrap_or_default(),
            None if self.table.is_empty() => Vec::new(),
            None => vec![LayerOp::Reset, LayerOp::Apply(Rc::new(self.table.snapshot()))],
        };
        *cursor = Some(end);
        out
    }

    /// Drops ops every layer has consumed (`upto` = the smallest cursor).
    pub fn trim_ops(&mut self, upto: u64) {
        let n = upto.saturating_sub(self.ops_base).min(self.ops.len() as u64) as usize;
        self.ops.drain(..n);
        self.ops_base += n as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skylens_viewer::pointblitz::io::slots::{Record, SlotHeader};

    fn pt(id: u32, v: f32) -> Record {
        Record { id, position: [v, v + 1.0, v + 2.0], color: [id as u8, 2, 3], zone: (id % 4) as u8 }
    }

    /// One slot message: adds, moves (id, xyz, rgb) and removes.
    fn msg(seq: u32, adds: &[Record], moves: &[(u32, [f32; 3], [u8; 3])], removes: &[u32], capacity: u32) -> Vec<u8> {
        let mut pack = Vec::new();
        for a in adds {
            a.write(&mut pack);
        }
        for &(id, position, color) in moves {
            Record { id, position, color, zone: 0 }.write(&mut pack);
        }
        for id in removes {
            pack.extend(id.to_le_bytes());
        }
        let h = SlotHeader { seq, n_add: adds.len() as u32, n_move: moves.len() as u32, n_remove: removes.len() as u32, capacity };
        slots::encode(&h, &pack).unwrap()
    }

    /// Adds, moves, removes, slot reuse, a REPLACE and a FINAL marker.
    fn sequence() -> Vec<Vec<u8>> {
        let a: Vec<Record> = (0..6).map(|i| pt(i, i as f32)).collect();
        let b: Vec<Record> = (6..9).map(|i| pt(i, i as f32)).collect();
        vec![
            msg(1, &a, &[], &[], 6),
            msg(2, &b, &[], &[], 9),
            // REPLACE: 2 and 3 move, 4 and 5 are removed, ids 4 (reused) and 9 are added.
            msg(3, &[pt(4, 40.0), pt(9, 90.0)], &[(2, [20.0; 3], [9, 9, 9]), (3, [30.0; 3], [8, 8, 8])], &[4, 5], 10),
            msg(4, &[], &[], &[], 10),
        ]
    }

    fn push_all(log: &mut DeltaLog, run: u32, msgs: &[Vec<u8>]) {
        for (i, m) in msgs.iter().enumerate() {
            log.push(run, (i % 3) as u32, m, 1000.0 * i as f64);
        }
    }

    /// Reference slot table with cumulus3d's `DisplayMirror` rules (removes, adds, moves).
    fn mirror(msgs: &[Vec<u8>]) -> Vec<Option<Slot>> {
        let mut t: Vec<Option<Slot>> = Vec::new();
        for m in msgs {
            let d = Diff::parse(m).unwrap();
            t.resize(t.len().max(d.header.capacity as usize), None);
            for id in d.remove_ids() {
                assert!(t[id as usize].take().is_some(), "remove of an empty slot");
            }
            for r in d.add_records() {
                assert!(t[r.id as usize].is_none(), "add into a live slot");
                t[r.id as usize] = Some(Slot { xyz: r.position, rgb: r.color, zone: r.zone });
            }
            for r in d.move_records() {
                let p = t[r.id as usize].as_mut().expect("move of an empty slot");
                p.xyz = r.position;
                p.rgb = r.color;
            }
        }
        t
    }

    fn norm(mut s: Vec<Option<Slot>>) -> Vec<Option<Slot>> {
        while s.last() == Some(&None) {
            s.pop();
        }
        s
    }

    /// Ops since the last call, through one layer cursor.
    fn take(log: &DeltaLog, cur: &mut Option<u64>) -> Vec<LayerOp> {
        log.ops_since(cur)
    }

    #[test]
    fn slot_table_matches_the_reference_mirror() {
        let msgs = sequence();
        let mut log = DeltaLog::new();
        for (i, m) in msgs.iter().enumerate() {
            assert_eq!(log.push(7, 0, m, i as f64), if i == 0 { Pushed::Stored } else { Pushed::Stored });
            assert_eq!(norm(log.table().slots.clone()), norm(mirror(&msgs[..=i])), "after message {}", i + 1);
        }
        assert_eq!((log.len(), log.applied()), (4, 4));
        assert!(log.is_final());
        assert_eq!(log.table().len(), 9);
        let e = log.entries()[2];
        assert_eq!((e.n_add, e.n_move, e.n_remove, e.kind, e.total_points), (2, 2, 2, Kind::Replace, 9));
        assert_eq!(log.entries()[0].kind, Kind::Refined);
        assert_eq!(log.entries()[3].kind, Kind::Final);
    }

    #[test]
    fn the_clock_counts_from_the_first_message() {
        let mut log = DeltaLog::new();
        push_all(&mut log, 1, &sequence());
        let t: Vec<f64> = log.entries().iter().map(|e| e.t_ms).collect();
        assert_eq!(t, vec![0.0, 1000.0, 2000.0, 3000.0]);
    }

    #[test]
    fn step_back_restores_each_earlier_state() {
        let msgs = sequence();
        let mut log = DeltaLog::new();
        push_all(&mut log, 1, &msgs);
        for k in (0..4).rev() {
            log.step_back();
            assert_eq!(log.applied(), k);
            assert!(!log.is_live());
            assert_eq!(norm(log.table().slots.clone()), norm(mirror(&msgs[..k])), "after undoing to {k}");
        }
        log.step_back(); // at the start: nothing
        assert_eq!(log.applied(), 0);
        // Forward again re-applies with animation and reaches live at the end.
        for _ in 0..4 {
            log.step_forward();
        }
        assert!(log.is_live());
        assert_eq!(norm(log.table().slots.clone()), norm(mirror(&msgs)));
    }

    #[test]
    fn undo_batch_is_the_inverse() {
        let mut log = DeltaLog::new();
        push_all(&mut log, 1, &sequence()[..3]);
        let mut cur = Some(0);
        take(&log, &mut cur);
        log.step_back();
        let ops = take(&log, &mut cur);
        let LayerOp::Apply(b) = &ops[0] else { panic!("apply expected") };
        assert!(b.animate);
        assert_eq!(b.removes.iter().map(|r| r.id).collect::<Vec<_>>(), vec![4, 9]); // the adds fade out
        assert_eq!(b.adds.iter().map(|a| a.id).collect::<Vec<_>>(), vec![4, 5]); // the removed come back
        assert_eq!(b.moves.iter().map(|m| m.id).collect::<Vec<_>>(), vec![2, 3]);
        assert_eq!(b.moves[0].xyz, [2.0, 3.0, 4.0]); // glide back to where it was
        assert_eq!(b.moves[0].from, [20.0; 3]);
    }

    #[test]
    fn the_forward_batch_carries_what_the_animation_needs() {
        let mut log = DeltaLog::new();
        push_all(&mut log, 1, &sequence()[..3]);
        let mut cur = None;
        let ops = take(&log, &mut cur);
        // A fresh layer gets the present state in one snap.
        assert!(matches!(&ops[0], LayerOp::Reset));
        let LayerOp::Apply(snap) = &ops[1] else { panic!() };
        assert!(!snap.animate && snap.adds.len() == 9);
        let mut cur = Some(0);
        let ops = take(&log, &mut cur);
        let LayerOp::Apply(b) = ops.last().unwrap() else { panic!() };
        assert!(b.animate);
        assert_eq!(b.removes.iter().map(|r| (r.id, r.xyz)).collect::<Vec<_>>(), vec![(4, [4.0, 5.0, 6.0]), (5, [5.0, 6.0, 7.0])]);
        assert_eq!(b.moves[1].from, [3.0, 4.0, 5.0]);
        assert_eq!(b.moves[1].xyz, [30.0; 3]);
    }

    #[test]
    fn seek_replays_without_animation_and_resumes_live_at_the_end() {
        let msgs = sequence();
        let mut log = DeltaLog::new();
        push_all(&mut log, 1, &msgs);
        let mut cur = None;
        take(&log, &mut cur);
        log.seek(2);
        assert_eq!(log.applied(), 2);
        assert!(!log.is_live());
        let ops = take(&log, &mut cur);
        assert_eq!(ops[0], LayerOp::Reset);
        assert!(ops[1..].iter().all(|o| matches!(o, LayerOp::Apply(b) if !b.animate)));
        assert_eq!(ops.len(), 3);
        assert_eq!(log.table().len(), 9);
        // A message arriving while not live is stored, not applied.
        log.push(1, 0, &msg(5, &[pt(10, 1.0)], &[], &[], 11), 9000.0);
        assert_eq!((log.len(), log.applied()), (5, 2));
        log.follow_live();
        assert_eq!(log.applied(), 5);
        assert!(log.is_live());
        log.seek(99);
        assert!(log.is_live());
    }

    #[test]
    fn unfollow_pauses_on_the_current_event() {
        let msgs = sequence();
        let mut log = DeltaLog::new();
        push_all(&mut log, 1, &msgs[..2]);
        log.unfollow();
        assert!(!log.is_live());
        log.push(1, 0, &msgs[2], 5000.0);
        assert_eq!((log.len(), log.applied()), (3, 2));
    }

    #[test]
    fn a_new_run_id_empties_the_buffer() {
        let msgs = sequence();
        let mut log = DeltaLog::new();
        push_all(&mut log, 1, &msgs);
        let mut cur = None;
        take(&log, &mut cur);
        assert_eq!(log.push(2, 0, &msgs[0], 0.0), Pushed::NewRun);
        let ops = take(&log, &mut cur);
        assert_eq!(ops[0], LayerOp::Reset);
        assert_eq!((log.len(), log.applied(), log.run_id()), (1, 1, Some(2)));
        assert_eq!(log.table().len(), 6);
        // Repeats and older seqs are ignored.
        assert_eq!(log.push(2, 0, &msgs[0], 1.0), Pushed::Ignored);
        // The new run's clock starts at its own first message.
        assert_eq!(log.entries()[0].t_ms, 0.0);
    }

    #[test]
    fn a_reset_forgets_the_run_and_the_replay_starts_over() {
        let msgs = sequence();
        let mut log = DeltaLog::new();
        push_all(&mut log, 1, &msgs);
        log.reset();
        assert_eq!((log.len(), log.applied(), log.run_id()), (0, 0, None));
        assert!(log.table().is_empty());
        push_all(&mut log, 1, &msgs);
        assert_eq!(log.applied(), 4);
    }

    #[test]
    fn two_layers_follow_the_same_log_with_their_own_cursors() {
        let msgs = sequence();
        let mut log = DeltaLog::new();
        let (mut a, mut b) = (Some(0), None);
        push_all(&mut log, 1, &msgs[..2]);
        assert_eq!(take(&log, &mut a).len(), 2); // the two events
        let first = take(&log, &mut b);
        assert_eq!(first.len(), 2); // reset + the whole state at once
        log.push(1, 0, &msgs[2], 3.0);
        assert_eq!(take(&log, &mut a).len(), 1);
        assert_eq!(take(&log, &mut b).len(), 1);
        log.trim_ops(a.unwrap().min(b.unwrap()));
        assert!(take(&log, &mut a).is_empty());
    }

    #[test]
    fn the_producers_kind_and_clock_win_over_the_content_and_arrival() {
        let msgs = sequence();
        let mut log = DeltaLog::new();
        // Arrival times are all the same (a late join); the producer's clock spaces the rows.
        for (i, m) in msgs.iter().enumerate() {
            // The producer calls the first REPLACE-looking event refined on purpose.
            let kind = [1, 1, 1, 3][i];
            log.push_meta(1, 0, m, 5.0, SlotMeta { kind, t_ms: 1000 * (i as u32 + 1) });
        }
        let e = log.entries();
        assert_eq!(e.iter().map(|e| e.kind).collect::<Vec<_>>(), vec![Kind::Refined, Kind::Refined, Kind::Refined, Kind::Final]);
        assert_eq!(e.iter().map(|e| e.t_ms).collect::<Vec<_>>(), vec![1000.0, 2000.0, 3000.0, 4000.0]);
        // Zeros fall back to the content.
        let mut log = DeltaLog::new();
        log.push_meta(1, 0, &msgs[2], 9.0, SlotMeta::default());
        assert_eq!(log.entries()[0].kind, Kind::Replace);
    }

    #[test]
    fn an_invalid_message_is_ignored() {
        let mut log = DeltaLog::new();
        assert_eq!(log.push(1, 0, b"nope", 0.0), Pushed::Ignored);
        assert!(log.is_empty());
    }
}
