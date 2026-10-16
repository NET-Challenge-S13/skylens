//! CPU side of `AnimatedPoints`, free of GPU types so it can be tested without an adapter: slot
//! allocation, the per-slot animation record, dirty ranges for partial uploads, capacity growth.
//!
//! Every point owns a slot. A slot holds the target position, the previous position, a colour, an
//! animation mode and the animation start time; the vertex shader interpolates from a `now`
//! uniform, so the CPU writes a slot once per change and not once per frame.

use bytemuck::{Pod, Zeroable};
use glam::DVec3;
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::ops::Range;

/// Fade-in of an added point, seconds.
pub const FADE_IN_S: f32 = 0.15;
/// Fade-out of a removed point, seconds. The slot is recycled when it has run out.
pub const FADE_OUT_S: f32 = 0.15;
/// Glide of a moved point, seconds (ease-out cubic).
pub const GLIDE_S: f32 = 0.7;
/// Points the board may touch per frame; `apply` itself does not enforce it.
pub const FRAME_BUDGET: usize = 200_000;
/// Gaps up to this many slots are bridged when coalescing dirty ranges (one `write_buffer` call
/// instead of two; the bridged slots are re-sent unchanged).
pub const MERGE_GAP: u32 = 16;

pub const MODE_STATIC: u8 = 0;
pub const MODE_FADE_IN: u8 = 1;
pub const MODE_FADE_OUT: u8 = 2;
pub const MODE_GLIDE: u8 = 3;
/// A free slot: the shader draws nothing.
pub const MODE_DEAD: u8 = 4;

/// One slot as the vertex shader reads it (32 bytes, instance stride).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct PointRec {
    /// Target position, metres from the layer origin.
    pub pos: [f32; 3],
    /// Start of a glide, same space.
    pub prev: [f32; 3],
    /// r, g, b, mode.
    pub rgb_mode: [u8; 4],
    /// Animation start, seconds since the book's time base.
    pub start: f32,
}

pub const REC_SIZE: usize = size_of::<PointRec>();

impl PointRec {
    pub fn mode(&self) -> u8 {
        self.rgb_mode[3]
    }
}

/// Ease-out cubic.
pub fn ease_out(k: f32) -> f32 {
    let k = k.clamp(0.0, 1.0);
    1.0 - (1.0 - k).powi(3)
}

/// CPU mirror of the shader: position drawn at book time `t`.
pub fn displayed_pos(r: &PointRec, t: f32) -> [f32; 3] {
    if r.mode() == MODE_GLIDE {
        let e = ease_out((t - r.start) / GLIDE_S);
        [0, 1, 2].map(|i| r.prev[i] + (r.pos[i] - r.prev[i]) * e)
    } else {
        r.pos
    }
}

/// CPU mirror of the shader: opacity at book time `t` (0 for a dead slot).
pub fn displayed_alpha(r: &PointRec, t: f32) -> f32 {
    match r.mode() {
        MODE_FADE_IN => ((t - r.start) / FADE_IN_S).clamp(0.0, 1.0),
        MODE_FADE_OUT => 1.0 - ((t - r.start) / FADE_OUT_S).clamp(0.0, 1.0),
        MODE_DEAD => 0.0,
        _ => 1.0,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Add {
    pub key: u64,
    /// Absolute position (ENU metres).
    pub pos: [f64; 3],
    pub rgb: [u8; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Move {
    pub key: u64,
    pub pos: [f64; 3],
    /// New colour, applied at once; `None` keeps it.
    pub rgb: Option<[u8; 3]>,
}

/// What one `apply` did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ApplyStats {
    pub added: usize,
    pub moved: usize,
    pub removed: usize,
    /// Moves/removes of keys that are not live.
    pub missing: usize,
    /// Adds dropped because the capacity limit was reached.
    pub dropped: usize,
    /// Slots written to the GPU buffer (including bridged gaps).
    pub uploaded: usize,
    /// `write_buffer` calls.
    pub ranges: usize,
    /// The GPU buffer was reallocated (capacity grew).
    pub reallocated: bool,
}

pub struct Book {
    recs: Vec<PointRec>,
    keys: HashMap<u64, u32>,
    free: BTreeSet<u32>,
    /// (book time when the fade-out ends, slot), oldest first.
    fading: VecDeque<(f32, u32)>,
    /// Slots in use or holes below the highest used one: the instance count to draw.
    high: u32,
    capacity: usize,
    max_capacity: usize,
    dirty: Vec<u32>,
    /// All of `recs[..high]` must be re-sent (after a reallocation or a rebase).
    all_dirty: bool,
    origin: DVec3,
    base: Option<f64>,
}

/// Book time is f32 seconds; past this the base moves forward to keep millisecond precision.
const REBASE_AFTER_S: f32 = 4.0 * 3600.0;

impl Book {
    pub fn new(origin: [f64; 3], capacity: usize, max_capacity: usize) -> Self {
        let max_capacity = max_capacity.max(1);
        let capacity = capacity.clamp(1, max_capacity);
        Self {
            recs: vec![PointRec::zeroed(); capacity],
            keys: HashMap::new(),
            free: BTreeSet::new(),
            fading: VecDeque::new(),
            high: 0,
            capacity,
            max_capacity,
            dirty: Vec::new(),
            all_dirty: false,
            origin: DVec3::from(origin),
            base: None,
        }
    }

    pub fn origin(&self) -> DVec3 {
        self.origin
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Live points (not counting fading-out ones).
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
    /// Points still fading out.
    pub fn fading(&self) -> usize {
        self.fading.len()
    }
    /// Instances to draw.
    pub fn draw_count(&self) -> u32 {
        self.high
    }
    pub fn recs(&self) -> &[PointRec] {
        &self.recs[..self.high as usize]
    }
    pub fn slot_of(&self, key: u64) -> Option<u32> {
        self.keys.get(&key).copied()
    }
    pub fn rec(&self, slot: u32) -> &PointRec {
        &self.recs[slot as usize]
    }

    /// Book time of `now` (seconds). The first call fixes the base.
    pub fn time(&mut self, now: f64) -> f32 {
        let base = *self.base.get_or_insert(now);
        let t = (now - base) as f32;
        if t > REBASE_AFTER_S {
            self.rebase(t);
            return (now - self.base.unwrap_or(now)) as f32;
        }
        t
    }

    fn rebase(&mut self, shift: f32) {
        if let Some(b) = &mut self.base {
            *b += f64::from(shift);
        }
        for r in &mut self.recs[..self.high as usize] {
            r.start -= shift;
        }
        for (end, _) in &mut self.fading {
            *end -= shift;
        }
        self.all_dirty = true;
    }

    /// Moves the layer origin; every slot is re-expressed and re-sent.
    pub fn set_origin(&mut self, origin: [f64; 3]) {
        let new = DVec3::from(origin);
        let d = (self.origin - new).as_vec3();
        for r in &mut self.recs[..self.high as usize] {
            for i in 0..3 {
                r.pos[i] += d[i];
                r.prev[i] += d[i];
            }
        }
        self.origin = new;
        self.all_dirty = true;
    }

    fn rel(&self, p: [f64; 3]) -> [f32; 3] {
        (DVec3::from(p) - self.origin).as_vec3().to_array()
    }

    /// Grows (or keeps) the capacity; returns true if it changed. Never below the slots in use.
    pub fn set_capacity(&mut self, cap: usize) -> bool {
        let cap = cap.clamp(self.high as usize, self.max_capacity).max(1);
        if cap == self.capacity {
            return false;
        }
        self.recs.resize(cap, PointRec::zeroed());
        self.capacity = cap;
        self.all_dirty = true;
        true
    }

    fn alloc(&mut self) -> Option<u32> {
        if let Some(s) = self.free.pop_first() {
            return Some(s);
        }
        if self.high as usize >= self.capacity {
            if self.capacity >= self.max_capacity {
                return None;
            }
            self.set_capacity((self.capacity * 2).max(1024));
        }
        let s = self.high;
        self.high += 1;
        Some(s)
    }

    fn put(&mut self, slot: u32, rec: PointRec) {
        self.recs[slot as usize] = rec;
        self.dirty.push(slot);
    }

    /// Frees fade-outs that have run out at `now`; trims dead slots off the end of the draw range.
    pub fn recycle(&mut self, now: f64) {
        let t = self.time(now);
        while let Some(&(end, slot)) = self.fading.front() {
            if end > t {
                break;
            }
            self.fading.pop_front();
            let mut dead = PointRec::zeroed();
            dead.rgb_mode[3] = MODE_DEAD;
            self.put(slot, dead);
            self.free.insert(slot);
        }
        while self.high > 0 && self.free.remove(&(self.high - 1)) {
            self.high -= 1;
        }
    }

    /// Adds, moves and removes at `now` (seconds, any monotonic clock). An add of a live key and a
    /// move of an unknown key are not errors: the first acts as a move, the second is counted in
    /// `missing`. Order: recycle, removes, moves, adds.
    pub fn apply(&mut self, adds: &[Add], moves: &[Move], removes: &[u64], now: f64) -> ApplyStats {
        let mut st = ApplyStats::default();
        self.recycle(now);
        let t = self.time(now);

        for key in removes {
            let Some(slot) = self.keys.remove(key) else {
                st.missing += 1;
                continue;
            };
            let mut r = self.recs[slot as usize];
            let p = displayed_pos(&r, t);
            r.pos = p;
            r.prev = p;
            r.rgb_mode[3] = MODE_FADE_OUT;
            r.start = t;
            self.put(slot, r);
            self.fading.push_back((t + FADE_OUT_S, slot));
            st.removed += 1;
        }
        for m in moves {
            if self.move_one(m.key, m.pos, m.rgb, t) {
                st.moved += 1;
            } else {
                st.missing += 1;
            }
        }
        for a in adds {
            if self.keys.contains_key(&a.key) {
                self.move_one(a.key, a.pos, Some(a.rgb), t);
                st.moved += 1;
                continue;
            }
            let Some(slot) = self.alloc() else {
                st.dropped += 1;
                continue;
            };
            let p = self.rel(a.pos);
            self.keys.insert(a.key, slot);
            self.put(
                slot,
                PointRec {
                    pos: p,
                    prev: p,
                    rgb_mode: [a.rgb[0], a.rgb[1], a.rgb[2], MODE_FADE_IN],
                    start: t,
                },
            );
            st.added += 1;
        }
        st
    }

    fn move_one(&mut self, key: u64, pos: [f64; 3], rgb: Option<[u8; 3]>, t: f32) -> bool {
        let Some(&slot) = self.keys.get(&key) else {
            return false;
        };
        let mut r = self.recs[slot as usize];
        let target = self.rel(pos);
        let shown = displayed_pos(&r, t);
        let far = (0..3).any(|i| (target[i] - r.pos[i]).abs() > 1e-4);
        if far {
            if r.mode() == MODE_FADE_IN && displayed_alpha(&r, t) < 1.0 {
                // Still fading in: appear at the new place, keep fading.
                r.pos = target;
                r.prev = target;
            } else {
                // Glide from where the point is drawn now, so a retarget mid-glide does not jump.
                r.prev = shown;
                r.pos = target;
                r.start = t;
                r.rgb_mode[3] = MODE_GLIDE;
            }
        }
        if let Some(c) = rgb {
            r.rgb_mode[..3].copy_from_slice(&c);
        }
        if r != self.recs[slot as usize] {
            self.put(slot, r);
        }
        true
    }

    /// Drains the touched slots as ranges to upload; `None` slots means the whole draw range.
    pub fn take_dirty(&mut self) -> Vec<Range<usize>> {
        if std::mem::take(&mut self.all_dirty) {
            self.dirty.clear();
            return if self.high == 0 { vec![] } else { vec![0..self.high as usize] };
        }
        let mut d = std::mem::take(&mut self.dirty);
        d.retain(|&s| s < self.high);
        d.sort_unstable();
        d.dedup();
        let mut out: Vec<Range<usize>> = Vec::new();
        for s in d {
            let s = s as usize;
            match out.last_mut() {
                Some(r) if s <= r.end + MERGE_GAP as usize => r.end = s + 1,
                _ => out.push(s..s + 1),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book() -> Book {
        Book::new([100.0, 200.0, 10.0], 4, 1 << 20)
    }
    fn add(key: u64, x: f64) -> Add {
        Add { key, pos: [100.0 + x, 200.0, 10.0], rgb: [1, 2, 3] }
    }

    #[test]
    fn rec_is_32_bytes() {
        assert_eq!(REC_SIZE, 32);
    }

    #[test]
    fn add_fades_in_at_a_slot_relative_to_the_origin() {
        let mut b = book();
        let st = b.apply(&[add(7, 1.5)], &[], &[], 10.0);
        assert_eq!(st.added, 1);
        let r = *b.rec(b.slot_of(7).unwrap());
        assert_eq!(r.pos, [1.5, 0.0, 0.0]);
        assert_eq!(r.mode(), MODE_FADE_IN);
        assert_eq!(r.rgb_mode[..3], [1, 2, 3]);
        assert_eq!(displayed_alpha(&r, 0.0), 0.0);
        assert!((displayed_alpha(&r, 0.075) - 0.5).abs() < 1e-6);
        assert_eq!(displayed_alpha(&r, 0.2), 1.0);
        assert_eq!(b.draw_count(), 1);
    }

    #[test]
    fn move_glides_from_the_displayed_position() {
        let mut b = book();
        b.apply(&[add(1, 0.0)], &[], &[], 0.0);
        b.apply(&[], &[Move { key: 1, pos: [110.0, 200.0, 10.0], rgb: None }], &[], 1.0);
        let r = *b.rec(0);
        assert_eq!(r.mode(), MODE_GLIDE);
        assert_eq!(r.prev, [0.0; 3]);
        assert_eq!(r.pos, [10.0, 0.0, 0.0]);
        // Ease-out: past half way at the half time, exactly there at the end.
        let mid = displayed_pos(&r, 1.0 + GLIDE_S / 2.0)[0];
        assert!(mid > 5.0 && mid < 10.0, "{mid}");
        assert_eq!(displayed_pos(&r, 1.0 + GLIDE_S)[0], 10.0);
        // Retarget in mid glide starts from where the point is drawn.
        let shown = displayed_pos(&r, 1.0 + GLIDE_S / 2.0)[0];
        b.apply(&[], &[Move { key: 1, pos: [100.0, 200.0, 10.0], rgb: None }], &[], 1.0 + f64::from(GLIDE_S / 2.0));
        assert!((b.rec(0).prev[0] - shown).abs() < 1e-4);
    }

    #[test]
    fn unchanged_move_only_recolours() {
        let mut b = book();
        b.apply(&[add(1, 0.0)], &[], &[], 0.0);
        b.take_dirty();
        b.apply(&[], &[Move { key: 1, pos: [100.0, 200.0, 10.0], rgb: Some([9, 9, 9]) }], &[], 5.0);
        assert_eq!(b.rec(0).mode(), MODE_FADE_IN); // nothing re-triggered
        assert_eq!(b.rec(0).rgb_mode[..3], [9, 9, 9]);
        let mut st = b.apply(&[], &[Move { key: 99, pos: [0.0; 3], rgb: None }], &[], 5.0);
        assert_eq!(st.missing, 1);
        st = b.apply(&[], &[], &[99], 5.0);
        assert_eq!(st.missing, 1);
    }

    #[test]
    fn remove_fades_out_then_frees_the_slot_for_reuse() {
        let mut b = book();
        b.apply(&[add(1, 0.0), add(2, 1.0)], &[], &[], 0.0);
        b.apply(&[], &[], &[1], 1.0);
        assert!(b.slot_of(1).is_none());
        assert_eq!(b.len(), 1);
        assert_eq!(b.fading(), 1);
        assert_eq!(b.rec(0).mode(), MODE_FADE_OUT);
        // The slot is still taken while fading: a new point goes to a new slot.
        b.apply(&[add(3, 2.0)], &[], &[], 1.05);
        assert_eq!(b.slot_of(3), Some(2));
        // After the fade the slot is dead and the next add reuses it.
        b.apply(&[], &[], &[], 1.2);
        assert_eq!(b.fading(), 0);
        assert_eq!(b.rec(0).mode(), MODE_DEAD);
        b.apply(&[add(4, 3.0)], &[], &[], 1.2);
        assert_eq!(b.slot_of(4), Some(0));
    }

    #[test]
    fn removing_from_the_end_shrinks_the_draw_range() {
        let mut b = book();
        b.apply(&[add(1, 0.0), add(2, 1.0), add(3, 2.0)], &[], &[], 0.0);
        b.apply(&[], &[], &[2, 3], 1.0);
        assert_eq!(b.draw_count(), 3);
        b.recycle(2.0);
        assert_eq!(b.draw_count(), 1);
    }

    #[test]
    fn capacity_grows_and_resends_everything() {
        let mut b = book();
        let adds: Vec<Add> = (0..10).map(|i| add(i, i as f64)).collect();
        b.apply(&adds, &[], &[], 0.0);
        assert!(b.capacity() >= 10);
        assert_eq!(b.take_dirty(), vec![0..10]);
        assert!(b.take_dirty().is_empty());
        assert!(!b.set_capacity(b.capacity()));
        assert!(b.set_capacity(5000));
        assert_eq!(b.capacity(), 5000);
        assert_eq!(b.take_dirty(), vec![0..10]);
        // Cannot shrink under the slots in use.
        b.set_capacity(2);
        assert_eq!(b.capacity(), 10);
    }

    #[test]
    fn max_capacity_drops_adds() {
        let mut b = Book::new([0.0; 3], 2, 3);
        let adds: Vec<Add> = (0..5).map(|i| add(i, 0.0)).collect();
        let st = b.apply(&adds, &[], &[], 0.0);
        assert_eq!((st.added, st.dropped), (3, 2));
    }

    #[test]
    fn dirty_slots_coalesce_into_ranges() {
        let mut b = Book::new([100.0, 200.0, 10.0], 1000, 1 << 20);
        let adds: Vec<Add> = (0..200).map(|i| add(i, 0.0)).collect();
        b.apply(&adds, &[], &[], 0.0);
        b.take_dirty();
        let mv = |k| Move { key: k, pos: [101.0, 200.0, 10.0], rgb: None };
        // 3, 4, 5 touch; 10 is within the merge gap; 100 is far.
        b.apply(&[], &[mv(3), mv(5), mv(4), mv(10), mv(100)], &[], 1.0);
        assert_eq!(b.take_dirty(), vec![3..11, 100..101]);
        assert!(b.take_dirty().is_empty());
    }

    #[test]
    fn re_adding_a_live_key_moves_it() {
        let mut b = book();
        b.apply(&[add(1, 0.0)], &[], &[], 0.0);
        let st = b.apply(&[add(1, 5.0)], &[], &[], 1.0);
        assert_eq!((st.added, st.moved), (0, 1));
        assert_eq!(b.len(), 1);
        assert_eq!(b.rec(0).pos[0], 5.0);
    }

    #[test]
    fn origin_change_keeps_world_positions() {
        let mut b = book();
        b.apply(&[add(1, 2.0)], &[], &[], 0.0);
        b.take_dirty();
        b.set_origin([101.0, 200.0, 10.0]);
        assert_eq!(b.rec(0).pos[0], 1.0);
        assert_eq!(b.take_dirty(), vec![0..1]);
    }

    #[test]
    fn time_rebases_after_hours_without_changing_animations() {
        let mut b = book();
        b.apply(&[add(1, 0.0)], &[], &[], 0.0);
        let far = f64::from(REBASE_AFTER_S) + 100.0;
        b.recycle(far);
        // The point's fade-in is long over; its start moved back with the base.
        let t = b.time(far);
        assert!(t < 200.0);
        assert_eq!(displayed_alpha(b.rec(0), t), 1.0);
    }
}
