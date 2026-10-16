//! The board's record of the live point stream: every `PointsDelta` of the current run in seq
//! order, a cursor saying how many are applied to the point layer, and the commands the timeline
//! panel uses (seek, step, follow live). Pure Rust, no GPU: the point layer drains [`LayerOp`]s.
//!
//! Rules: a delta whose `run` differs from the log's run starts a new run (log and layer are
//! cleared first, [`LayerOp::Reset`]). Deltas are stored in seq order; a repeated or older seq is
//! ignored. A late joiner receives the stored deltas from seq 1 and they are applied in order.
//! While following live (the default) each new delta is applied as it arrives, animated; stepping
//! back or seeking leaves live mode until [`DeltaLog::follow_live`].

use std::rc::Rc;

use skylens_viewer::skylens_shared::delta::{AddRec, MoveRec, iter_adds, iter_moves, iter_removes};
use skylens_viewer::skylens_shared::proto::payload::v1::{DeltaKind as WireKind, PointsDelta};

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
    /// Model clock: milliseconds since the run's first delta (`t_s` of the delta).
    pub t_ms: f64,
    pub n_add: u32,
    pub n_move: u32,
    pub n_remove: u32,
    pub total_points: u32,
    /// Frame positions ingested so far / in total (0 unknown), for the progress bar.
    pub positions: u32,
    pub positions_total: u32,
}

/// One edit of the slot table, ready for the point layer: removes, then adds, then moves.
/// A forward delta, its inverse (step back) and a seek replay all come out in this shape.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Batch {
    pub removes: Vec<u32>,
    pub adds: Vec<AddRec>,
    pub moves: Vec<MoveRec>,
    /// Slot count the layer must hold (every id is below it).
    pub capacity: u32,
    /// Fade (150 ms) and glide (700 ms) instead of a snap.
    pub animate: bool,
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

/// The CPU copy of the point buffer: slot `id` holds a point or is free. Same rules as
/// cumulus3d's `DisplayMirror`; also makes the inverse of every delta.
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

    /// Applies a batch; returns the batch that undoes it (animated as the argument says).
    pub fn apply(&mut self, b: &Batch, undo_animate: bool) -> Batch {
        if self.slots.len() < b.capacity as usize {
            self.slots.resize(b.capacity as usize, None);
        }
        let mut inv = Batch { capacity: b.capacity, animate: undo_animate, ..Default::default() };
        // Undo of removes: put the old point back; of adds: remove (and restore what was there).
        for &id in &b.removes {
            if let Some(Some(old)) = self.slots.get(id as usize).copied() {
                inv.adds.push(AddRec { id, xyz: old.xyz, rgb: old.rgb, zone: old.zone });
                self.slots[id as usize] = None;
            }
        }
        let mut added = Vec::with_capacity(b.adds.len());
        for a in &b.adds {
            let i = a.id as usize;
            if i >= self.slots.len() {
                self.slots.resize(i + 1, None);
            }
            if let Some(old) = self.slots[i] {
                // Add into a live slot (a replace without a remove): the undo puts the old one back.
                inv.adds.push(AddRec { id: a.id, xyz: old.xyz, rgb: old.rgb, zone: old.zone });
            }
            self.slots[i] = Some(Slot { xyz: a.xyz, rgb: a.rgb, zone: a.zone });
            added.push(a.id);
        }
        inv.removes = added.clone();
        let added: std::collections::HashSet<u32> = added.into_iter().collect();
        for m in &b.moves {
            if let Some(Some(p)) = self.slots.get_mut(m.id as usize) {
                if !added.contains(&m.id) {
                    inv.moves.push(MoveRec { id: m.id, xyz: p.xyz, rgb: p.rgb });
                }
                p.xyz = m.xyz;
                p.rgb = m.rgb;
            }
        }
        inv
    }
}

fn batch_of(d: &PointsDelta, animate: bool) -> Batch {
    Batch {
        removes: iter_removes(&d.removes).map(|i| i.collect()).unwrap_or_default(),
        adds: iter_adds(&d.adds).map(|i| i.collect()).unwrap_or_default(),
        moves: iter_moves(&d.moves).map(|i| i.collect()).unwrap_or_default(),
        capacity: d.capacity,
        animate,
    }
}

#[derive(Default)]
pub struct DeltaLog {
    run_id: Option<u32>,
    entries: Vec<LogEntry>,
    deltas: Vec<PointsDelta>,
    /// Inverse of each applied delta (index < applied).
    undo: Vec<Batch>,
    applied: usize,
    live: bool,
    table: SlotTable,
    ops: Vec<LayerOp>,
    /// Number of ops trimmed off the front of `ops` (cursors count from the start of the run of ops).
    ops_base: u64,
    version: u64,
}

/// What [`DeltaLog::push`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pushed {
    /// Stored as the next delta.
    Stored,
    /// Stored, and it started a new run (everything older was dropped).
    NewRun,
    /// Older or repeated seq, ignored.
    Ignored,
}

impl DeltaLog {
    pub fn new() -> Self {
        DeltaLog { live: true, ..Default::default() }
    }

    /// Takes one decoded delta (animated when following live).
    pub fn push(&mut self, delta: PointsDelta) -> Pushed {
        self.push_inner(delta, true)
    }

    /// Takes deltas that arrived together (a late join replays the history from seq 1): only the
    /// last one is animated.
    pub fn push_batch(&mut self, deltas: Vec<PointsDelta>) -> Vec<Pushed> {
        let n = deltas.len();
        deltas.into_iter().enumerate().map(|(i, d)| self.push_inner(d, i + 1 == n)).collect()
    }

    fn push_inner(&mut self, d: PointsDelta, animate: bool) -> Pushed {
        let new_run = self.run_id.is_some_and(|r| r != d.run);
        if new_run {
            self.clear();
        } else if self.entries.last().is_some_and(|e| d.seq <= e.seq) && !(d.snapshot && self.entries.is_empty()) {
            return Pushed::Ignored;
        }
        self.run_id = Some(d.run);
        let entry = LogEntry {
            seq: d.seq,
            run_id: d.run,
            kind: kind_of(d.kind()),
            zone: d.zone,
            t_ms: d.t_s * 1000.0,
            n_add: (d.adds.len() / skylens_viewer::skylens_shared::delta::RECORD) as u32,
            n_move: (d.moves.len() / skylens_viewer::skylens_shared::delta::RECORD) as u32,
            n_remove: (d.removes.len() / skylens_viewer::skylens_shared::delta::REMOVE_RECORD) as u32,
            total_points: d.total_points,
            positions: d.positions,
            positions_total: d.positions_total,
        };
        let at_end = self.applied == self.entries.len();
        self.entries.push(entry);
        self.deltas.push(d);
        if self.live && at_end {
            self.apply_next(animate);
        }
        if new_run { Pushed::NewRun } else { Pushed::Stored }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.deltas.clear();
        self.undo.clear();
        self.table.clear();
        self.applied = 0;
        self.version += 1;
        self.live = true;
        self.reset_ops();
    }

    /// Applies entry `applied` and moves the cursor.
    fn apply_next(&mut self, animate: bool) {
        if self.deltas[self.applied].snapshot {
            // Full state: every live slot goes first, then the adds.
            self.table.clear();
            self.reset_ops();
            self.undo.clear();
            self.undo.resize(self.applied, Batch::default());
        }
        let b = batch_of(&self.deltas[self.applied], animate);
        let inv = self.table.apply(&b, true);
        self.undo.push(inv);
        self.ops.push(LayerOp::Apply(Rc::new(b)));
        self.applied += 1;
        self.version += 1;
    }

    /// All stored deltas' rows, oldest first (index 0 is the run's first delta).
    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    /// Number of stored deltas.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of deltas applied to the point layer: entries `0..applied()` are on screen.
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

    /// True once the last stored delta is a FINAL one.
    pub fn is_final(&self) -> bool {
        self.entries.last().is_some_and(|e| e.kind == Kind::Final)
    }

    /// Whether new deltas are applied as they arrive.
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

    /// Jumps to the state after delta `seq` (no animation: the layer is emptied and the deltas
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

    /// Undoes the last applied delta with the inverse animation. Leaves live mode.
    pub fn step_back(&mut self) {
        self.live = false;
        if self.applied == 0 {
            return;
        }
        let inv = self.undo.pop().expect("one undo per applied delta");
        self.table.apply(&inv, true);
        self.ops.push(LayerOp::Apply(Rc::new(inv)));
        self.applied -= 1;
        self.version += 1;
    }

    /// Applies the next stored delta with animation. Reaching the end resumes live following.
    pub fn step_forward(&mut self) {
        if self.applied < self.entries.len() {
            self.apply_next(true);
        }
        self.live = self.applied == self.entries.len();
    }

    /// Applies everything not yet applied (animated) and applies future deltas on arrival.
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
            None => {
                let adds = self
                    .table
                    .slots
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| s.map(|s| AddRec { id: i as u32, xyz: s.xyz, rgb: s.rgb, zone: s.zone }))
                    .collect();
                let capacity = self.table.slots.len() as u32;
                vec![LayerOp::Reset, LayerOp::Apply(Rc::new(Batch { adds, capacity, animate: false, ..Default::default() }))]
            }
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

pub(crate) fn kind_of(k: WireKind) -> Kind {
    match k {
        WireKind::Replace => Kind::Replace,
        WireKind::Final => Kind::Final,
        _ => Kind::Refined,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skylens_viewer::skylens_shared::delta::{pack_adds, pack_moves, pack_removes};

    /// Reference slot table with cumulus3d's `DisplayMirror` rules (removes, adds, moves).
    #[derive(Default)]
    struct Mirror(Vec<Option<Slot>>);
    impl Mirror {
        fn apply(&mut self, d: &PointsDelta) {
            let b = batch_of(d, false);
            for &id in &b.removes {
                assert!(self.0[id as usize].take().is_some(), "remove of an empty slot");
            }
            if self.0.len() < d.capacity as usize {
                self.0.resize(d.capacity as usize, None);
            }
            for a in &b.adds {
                assert!(self.0[a.id as usize].is_none(), "add into a live slot");
                self.0[a.id as usize] = Some(Slot { xyz: a.xyz, rgb: a.rgb, zone: a.zone });
            }
            for m in &b.moves {
                let p = self.0[m.id as usize].as_mut().expect("move of an empty slot");
                p.xyz = m.xyz;
                p.rgb = m.rgb;
            }
        }
    }

    /// Ops since the last call, through one layer cursor.
    fn take(log: &mut DeltaLog) -> Vec<LayerOp> {
        thread_local!(static CUR: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(Some(0)) });
        let mut c = CUR.get();
        let ops = log.ops_since(&mut c);
        CUR.set(c);
        ops
    }

    fn pt(id: u32, v: f32) -> AddRec {
        AddRec { id, xyz: [v, v + 1.0, v + 2.0], rgb: [id as u8, 2, 3], zone: (id % 4) as u8 }
    }

    /// A deterministic sequence with adds, moves, removes, slot reuse, a REPLACE and a FINAL.
    fn sequence(run: u32) -> Vec<PointsDelta> {
        let mk = |seq: u32, kind: WireKind, adds: Vec<AddRec>, moves: Vec<MoveRec>, removes: Vec<u32>, total: u32, cap: u32| PointsDelta {
            seq,
            kind: kind as i32,
            zone: seq % 3,
            capacity: cap,
            total_points: total,
            t_s: f64::from(seq) * 10.0,
            adds: pack_adds(&adds),
            moves: pack_moves(&moves),
            removes: pack_removes(&removes),
            run,
            ..Default::default()
        };
        vec![
            mk(1, WireKind::Refined, (0..6).map(|i| pt(i, i as f32)).collect(), vec![], vec![], 6, 6),
            mk(2, WireKind::Refined, (6..9).map(|i| pt(i, i as f32)).collect(), vec![], vec![], 9, 9),
            // REPLACE: 2 and 3 move, 4 and 5 are removed, ids 4 and 9 are added (4 reused).
            mk(
                3,
                WireKind::Replace,
                vec![pt(4, 40.0), pt(9, 90.0)],
                vec![
                    MoveRec { id: 2, xyz: [20.0, 20.0, 20.0], rgb: [9, 9, 9] },
                    MoveRec { id: 3, xyz: [30.0, 30.0, 30.0], rgb: [8, 8, 8] },
                ],
                vec![4, 5],
                9,
                10,
            ),
            mk(4, WireKind::Final, vec![], vec![], vec![], 9, 10),
        ]
    }

    fn live(table: &SlotTable) -> Vec<Option<Slot>> {
        table.slots.clone()
    }

    #[test]
    fn slot_table_matches_the_reference_mirror() {
        let mut log = DeltaLog::new();
        let mut mirror = Mirror::default();
        for d in sequence(7) {
            mirror.apply(&d);
            assert_eq!(log.push(d), Pushed::Stored);
            let mut m = mirror.0.clone();
            let mut t = live(log.table());
            let n = m.len().max(t.len());
            m.resize(n, None);
            t.resize(n, None);
            assert_eq!(m, t);
        }
        assert_eq!(log.len(), 4);
        assert_eq!(log.applied(), 4);
        assert!(log.is_final());
        assert_eq!(log.table().len(), 9);
        let e = log.entries()[2];
        assert_eq!((e.n_add, e.n_move, e.n_remove, e.kind), (2, 2, 2, Kind::Replace));
    }

    #[test]
    fn step_back_restores_each_earlier_state() {
        let mut log = DeltaLog::new();
        let mut states = vec![SlotTable::default()];
        for d in sequence(1) {
            log.push(d);
            states.push(log.table().clone());
        }
        let norm = |t: &SlotTable| {
            let mut s = t.slots.clone();
            while s.last() == Some(&None) {
                s.pop();
            }
            s
        };
        for k in (0..4).rev() {
            log.step_back();
            assert_eq!(log.applied(), k);
            assert!(!log.is_live());
            assert_eq!(norm(log.table()), norm(&states[k]), "after undoing to {k}");
        }
        log.step_back(); // at the start: nothing
        assert_eq!(log.applied(), 0);
        // Forward again re-applies with animation and reaches live at the end.
        for _ in 0..4 {
            log.step_forward();
        }
        assert!(log.is_live());
        assert_eq!(norm(log.table()), norm(&states[4]));
    }

    #[test]
    fn undo_batch_is_the_inverse() {
        let mut log = DeltaLog::new();
        for d in sequence(1).into_iter().take(3) {
            log.push(d);
        }
        take(&mut log);
        log.step_back();
        let ops = take(&mut log);
        let LayerOp::Apply(b) = &ops[0] else { panic!("apply expected") };
        assert!(b.animate);
        assert_eq!(b.removes, vec![4, 9]); // the adds of the replace fade out
        let ids = |v: &[AddRec]| v.iter().map(|a| a.id).collect::<Vec<_>>();
        assert_eq!(ids(&b.adds), vec![4, 5]); // the removed points fade back in
        assert_eq!(b.moves.iter().map(|m| m.id).collect::<Vec<_>>(), vec![2, 3]);
        assert_eq!(b.moves[0].xyz, [2.0, 3.0, 4.0]); // glide back to where it was
    }

    #[test]
    fn seek_replays_without_animation_and_resumes_live_at_the_end() {
        let mut log = DeltaLog::new();
        for d in sequence(1) {
            log.push(d);
        }
        take(&mut log);
        log.seek(2);
        assert_eq!(log.applied(), 2);
        assert!(!log.is_live());
        let ops = take(&mut log);
        assert_eq!(ops[0], LayerOp::Reset);
        assert!(ops[1..].iter().all(|o| matches!(o, LayerOp::Apply(b) if !b.animate)));
        assert_eq!(ops.len(), 3);
        assert_eq!(log.table().len(), 9);
        // A delta arriving while not live is stored, not applied.
        let d = PointsDelta { seq: 5, capacity: 10, run: 1, ..Default::default() };
        log.push(d);
        assert_eq!((log.len(), log.applied()), (5, 2));
        log.follow_live();
        assert_eq!(log.applied(), 5);
        assert!(log.is_live());
        log.seek(99);
        assert!(log.is_live());
    }

    #[test]
    fn new_run_id_empties_the_buffer() {
        let mut log = DeltaLog::new();
        for d in sequence(1) {
            log.push(d);
        }
        take(&mut log);
        let first = sequence(2).remove(0);
        assert_eq!(log.push(first), Pushed::NewRun);
        let ops = take(&mut log);
        assert_eq!(ops[0], LayerOp::Reset);
        assert_eq!((log.len(), log.applied(), log.run_id()), (1, 1, Some(2)));
        assert_eq!(log.table().len(), 6);
        // Repeats and older seqs are ignored.
        assert_eq!(log.push(sequence(2).remove(0)), Pushed::Ignored);
    }

    #[test]
    fn late_join_batch_animates_only_the_last() {
        let mut log = DeltaLog::new();
        log.push_batch(sequence(3));
        let ops = take(&mut log);
        let anim: Vec<bool> = ops.iter().filter_map(|o| if let LayerOp::Apply(b) = o { Some(b.animate) } else { None }).collect();
        assert_eq!(anim, vec![false, false, false, true]);
        assert_eq!(log.applied(), 4);
    }

    #[test]
    fn snapshot_resets_before_its_adds() {
        let mut log = DeltaLog::new();
        for d in sequence(1).into_iter().take(2) {
            log.push(d);
        }
        take(&mut log);
        let snap = PointsDelta { seq: 3, snapshot: true, capacity: 4, run: 1, adds: pack_adds(&[pt(0, 0.0), pt(3, 3.0)]), ..Default::default() };
        log.push(snap);
        let ops = take(&mut log);
        assert_eq!(ops[0], LayerOp::Reset);
        assert_eq!(log.table().len(), 2);
    }
}
