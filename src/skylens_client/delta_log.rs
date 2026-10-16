//! The board's record of the live point stream: every `PointsDelta` of the current run in seq
//! order, a cursor saying how many are applied to the point layer, and the commands the timeline
//! panel uses (seek, step, follow live). Pure Rust, no GPU: the point layer drains [`LayerOp`]s.
//!
//! Rules: a delta whose `run` differs from the log's run starts a new run (log and layer are
//! cleared first, [`LayerOp::Reset`]). Deltas are stored in seq order; a repeated or older seq is
//! ignored. A late joiner receives the stored deltas from seq 1 and they are applied in order.
//! While following live (the default) each new delta is applied as it arrives, animated; stepping
//! back or seeking leaves live mode until [`DeltaLog::follow_live`].

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

/// What the point layer must do, in order (see [`DeltaLog::take_ops`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerOp {
    /// Empty every slot and drop undo logs (new run, or a seek that rebuilds).
    Reset,
    /// Apply stored delta `index` (0-based position in [`DeltaLog::entries`]); `animate` = fade/glide.
    Apply { index: usize, animate: bool },
    /// Animate the inverse of delta `index` (moved points glide back, adds fade out, removes fade in).
    Undo { index: usize },
}

#[derive(Default)]
pub struct DeltaLog {
    run_id: Option<u32>,
    entries: Vec<LogEntry>,
    deltas: Vec<PointsDelta>,
    applied: usize,
    live: bool,
    ops: Vec<LayerOp>,
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

    /// Takes a decoded delta. Returns what happened.
    pub fn push(&mut self, delta: PointsDelta) -> Pushed {
        let _ = delta;
        Pushed::Ignored // stub
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

    /// Jumps to the state after delta `seq` (no animation: replays from slot zero).
    pub fn seek(&mut self, seq: u32) {
        let _ = seq;
    }

    /// Undoes the last applied delta with the inverse animation. Leaves live mode.
    pub fn step_back(&mut self) {}

    /// Applies the next stored delta with animation. Leaves live mode unless it reaches the end.
    pub fn step_forward(&mut self) {}

    /// Applies everything not yet applied (animated) and applies future deltas on arrival.
    pub fn follow_live(&mut self) {}

    /// Operations for the point layer since the last call.
    pub fn take_ops(&mut self) -> Vec<LayerOp> {
        std::mem::take(&mut self.ops)
    }

    /// The stored payload of entry `index` (for the point layer).
    pub fn delta(&self, index: usize) -> Option<&PointsDelta> {
        self.deltas.get(index)
    }
}

pub(crate) fn kind_of(k: WireKind) -> Kind {
    match k {
        WireKind::Replace => Kind::Replace,
        WireKind::Final => Kind::Final,
        _ => Kind::Refined,
    }
}
