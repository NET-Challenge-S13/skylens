//! The live point layer with per-point animation, on top of pointblitz slot scenes.
//!
//! pointblitz 0.3.0 draws a [`SlotScene`] as it is: 16 bytes per slot, no per-slot time, so an
//! add appears at once and a move jumps. The board wants the cumulus3d viewer's look instead:
//! an add or a remove fades (150 ms) and a move glides (700 ms, ease-out). This module gets that
//! without touching the crate, with two slot scenes:
//!
//! - the **base** scene holds every settled point;
//! - the **overlay** scene holds only the points in flight, in compact slot ids of its own. Each
//!   frame the viewer writes their interpolated position and colour (a fade is the colour moving
//!   to or from the background, which on an opaque point layer over a dark background is the
//!   same thing as opacity).
//!
//! A point in flight is hidden in the base scene and shown in the overlay; when its animation
//! ends it is written to the base scene and leaves the overlay in the same frame, so the
//! surface never doubles up. At most [`ANIM_BUDGET`] points are in flight; the rest of a large
//! event snaps. Every write is a plain slot message with a local seq, so the scenes' own rules
//! (one `write_buffer` of the changed range, doubling capacity) are used as they are.
//!
//! The layer knows nothing of the stream: it takes [`Batch`]es (removes, then adds, then moves,
//! each carrying the values it needs, so no CPU copy of the whole table is kept here).

use pointblitz::io::slots::{self, Record, SlotHeader};
use pointblitz::core::SlotScene;
use std::collections::HashMap;

/// Fade of an add or a remove.
pub const FADE_S: f64 = 0.15;
/// Glide of a move.
pub const GLIDE_S: f64 = 0.7;
/// Points animated at once; a bigger event animates this many and snaps the rest.
pub const ANIM_BUDGET: usize = 200_000;

/// A point that appears.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AddRec {
    pub id: u32,
    pub xyz: [f32; 3],
    pub rgb: [u8; 3],
    pub zone: u8,
}

/// A point that goes away, as it was.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RemoveRec {
    pub id: u32,
    pub xyz: [f32; 3],
    pub rgb: [u8; 3],
    pub zone: u8,
}

/// A point that moves from `from` to `xyz` (and changes colour).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveRec {
    pub id: u32,
    pub from: [f32; 3],
    pub from_rgb: [u8; 3],
    pub xyz: [f32; 3],
    pub rgb: [u8; 3],
    pub zone: u8,
}

/// One edit of the slot table, ready for the layer: removes, then adds, then moves.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Batch {
    pub removes: Vec<RemoveRec>,
    pub adds: Vec<AddRec>,
    pub moves: Vec<MoveRec>,
    /// Slot count the layer must hold (every id is below it).
    pub capacity: u32,
    /// Fade and glide instead of a snap.
    pub animate: bool,
}

impl Batch {
    pub fn len(&self) -> usize {
        self.removes.len() + self.adds.len() + self.moves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Style {
    FadeIn,
    FadeOut,
    Glide,
}

#[derive(Debug, Clone, Copy)]
struct Anim {
    /// Slot id in the base scene.
    id: u32,
    /// Slot id in the overlay scene.
    slot: u32,
    style: Style,
    from: [f32; 3],
    to: [f32; 3],
    from_rgb: [u8; 3],
    to_rgb: [u8; 3],
    zone: u8,
    t0: f64,
    dur: f64,
}

/// What a point looks like at `t` seconds of its animation: position, colour.
fn lerp3(a: [f32; 3], b: [f32; 3], k: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k]
}

fn mix_rgb(a: [u8; 3], b: [u8; 3], k: f32) -> [u8; 3] {
    let m = |x: u8, y: u8| (f32::from(x) + (f32::from(y) - f32::from(x)) * k).round().clamp(0.0, 255.0) as u8;
    [m(a[0], b[0]), m(a[1], b[1]), m(a[2], b[2])]
}

/// Ease-out cubic.
fn ease_out(k: f64) -> f32 {
    let k = k.clamp(0.0, 1.0);
    (1.0 - (1.0 - k).powi(3)) as f32
}

impl Anim {
    fn progress(&self, now: f64) -> f64 {
        if self.dur <= 0.0 { 1.0 } else { ((now - self.t0) / self.dur).clamp(0.0, 1.0) }
    }

    /// Position and colour at `now`; `bg` is what a fade runs to or from.
    fn at(&self, now: f64, bg: [u8; 3]) -> ([f32; 3], [u8; 3]) {
        let k = ease_out(self.progress(now));
        match self.style {
            Style::FadeIn => (self.to, mix_rgb(bg, self.to_rgb, k)),
            Style::FadeOut => (self.from, mix_rgb(self.from_rgb, bg, k)),
            Style::Glide => (lerp3(self.from, self.to, k), mix_rgb(self.from_rgb, self.to_rgb, k)),
        }
    }

    /// Whether the point is on screen once the animation is over.
    fn shown_after(&self) -> bool {
        self.style != Style::FadeOut
    }
}

/// What the base scene must show or hide for an id, written once per frame.
#[derive(Debug, Clone, Copy)]
enum Settle {
    Show { xyz: [f32; 3], rgb: [u8; 3], zone: u8 },
    Hide,
}

/// One slot scene with the next local seq.
struct Lane {
    scene: SlotScene,
    seq: u32,
    capacity: u32,
}

impl Lane {
    fn new() -> Lane {
        Lane { scene: SlotScene::new([0.0; 3]), seq: 0, capacity: 0 }
    }

    fn clear(&mut self) {
        self.scene.clear();
        self.seq = 0;
        self.capacity = 0;
    }

    /// Writes records as adds (an add writes the slot and shows it) and removes as one message.
    fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, adds: &[u8], n_add: u32, removes: &[u32], need: u32) {
        if n_add == 0 && removes.is_empty() {
            return;
        }
        self.capacity = self.capacity.max(need);
        self.seq += 1;
        let mut pack = Vec::with_capacity(adds.len() + removes.len() * 4);
        pack.extend_from_slice(adds);
        for id in removes {
            pack.extend(id.to_le_bytes());
        }
        let header = SlotHeader { seq: self.seq, n_add, n_move: 0, n_remove: removes.len() as u32, capacity: self.capacity };
        let msg = slots::encode(&header, &pack).expect("pack length from the counts");
        if let Err(e) = self.scene.apply(device, queue, &msg) {
            crate::log(&format!("[live-slots] slot message not applied: {e}"));
            // A message that changed nothing leaves its seq to the next one.
            self.seq -= 1;
        }
    }
}

/// The animated live point layer: draw [`LiveSlots::scenes`].
pub struct LiveSlots {
    base: Lane,
    overlay: Lane,
    /// Fade target colour (the view's background).
    bg: [u8; 3],
    anims: Vec<Anim>,
    by_id: HashMap<u32, usize>,
    free: Vec<u32>,
    next_overlay: u32,
    /// Overlay slots to hide at the next tick.
    overlay_gone: Vec<u32>,
    /// Base writes of this frame, last writer wins per id.
    settle: HashMap<u32, Settle>,
    /// Highest base slot id + 1 seen.
    need: u32,
    /// Adds and removes fade; off (the default) they appear and disappear at once and only moves
    /// glide: a fade of hundreds of thousands of points at once reads as a flash.
    fades: bool,
}

impl Default for LiveSlots {
    fn default() -> Self {
        Self::new([13, 15, 20])
    }
}

impl LiveSlots {
    pub fn new(bg: [u8; 3]) -> LiveSlots {
        LiveSlots {
            base: Lane::new(),
            overlay: Lane::new(),
            bg,
            anims: Vec::new(),
            by_id: HashMap::new(),
            free: Vec::new(),
            next_overlay: 0,
            overlay_gone: Vec::new(),
            settle: HashMap::new(),
            need: 0,
            fades: false,
        }
    }

    /// Fade adds and removes (150 ms) instead of showing and hiding them at once.
    pub fn set_fades(&mut self, on: bool) {
        self.fades = on;
    }

    pub fn set_background(&mut self, bg: [u8; 3]) {
        self.bg = bg;
    }

    /// Empties both scenes and forgets every animation.
    pub fn reset(&mut self) {
        self.base.clear();
        self.overlay.clear();
        self.anims.clear();
        self.by_id.clear();
        self.free.clear();
        self.next_overlay = 0;
        self.overlay_gone.clear();
        self.settle.clear();
        self.need = 0;
    }

    /// The scenes to draw: settled points, then the points in flight.
    pub fn scenes(&self) -> [&SlotScene; 2] {
        [&self.base.scene, &self.overlay.scene]
    }

    /// Points on screen (settled and in flight).
    pub fn points(&self) -> u64 {
        self.base.scene.points() + self.overlay.scene.points()
    }

    /// Points in flight.
    pub fn animating(&self) -> usize {
        self.anims.len()
    }

    /// Box of the settled points in ENU metres.
    pub fn bounds(&self) -> Option<([f64; 3], [f64; 3])> {
        self.base.scene.bounds()
    }

    fn start(&mut self, a: Anim) {
        if let Some(&i) = self.by_id.get(&a.id) {
            let slot = self.anims[i].slot;
            self.anims[i] = Anim { slot, ..a };
        } else {
            let slot = self.free.pop().unwrap_or_else(|| {
                self.next_overlay += 1;
                self.next_overlay - 1
            });
            self.by_id.insert(a.id, self.anims.len());
            self.anims.push(Anim { slot, ..a });
        }
    }

    /// The current look of the point animating under `id`.
    fn look(&self, id: u32, now: f64) -> Option<([f32; 3], [u8; 3])> {
        self.by_id.get(&id).map(|&i| self.anims[i].at(now, self.bg))
    }

    /// Takes one batch (not yet on screen until the next [`LiveSlots::tick`]).
    pub fn apply(&mut self, batch: &Batch, now: f64) {
        self.need = self.need.max(batch.capacity);
        let animate = batch.animate;
        let fade = animate && self.fades;
        let mut room = ANIM_BUDGET.saturating_sub(self.anims.len());
        for r in &batch.removes {
            if fade && (room > 0 || self.by_id.contains_key(&r.id)) {
                let (from, from_rgb) = self.look(r.id, now).unwrap_or((r.xyz, r.rgb));
                if !self.by_id.contains_key(&r.id) {
                    room -= 1;
                }
                self.settle.insert(r.id, Settle::Hide);
                self.start(Anim { id: r.id, slot: 0, style: Style::FadeOut, from, to: from, from_rgb, to_rgb: from_rgb, zone: r.zone, t0: now, dur: FADE_S });
            } else {
                self.cancel(r.id);
                self.settle.insert(r.id, Settle::Hide);
            }
        }
        for a in &batch.adds {
            if fade && (room > 0 || self.by_id.contains_key(&a.id)) {
                if !self.by_id.contains_key(&a.id) {
                    room -= 1;
                }
                // Hidden in the base until it lands (an add into a live slot replaces it).
                self.settle.insert(a.id, Settle::Hide);
                self.start(Anim { id: a.id, slot: 0, style: Style::FadeIn, from: a.xyz, to: a.xyz, from_rgb: a.rgb, to_rgb: a.rgb, zone: a.zone, t0: now, dur: FADE_S });
            } else {
                self.cancel(a.id);
                self.settle.insert(a.id, Settle::Show { xyz: a.xyz, rgb: a.rgb, zone: a.zone });
            }
        }
        for m in &batch.moves {
            if animate && (room > 0 || self.by_id.contains_key(&m.id)) {
                let (from, from_rgb) = self.look(m.id, now).unwrap_or((m.from, m.from_rgb));
                if !self.by_id.contains_key(&m.id) {
                    room -= 1;
                }
                self.settle.insert(m.id, Settle::Hide);
                self.start(Anim { id: m.id, slot: 0, style: Style::Glide, from, to: m.xyz, from_rgb, to_rgb: m.rgb, zone: m.zone, t0: now, dur: GLIDE_S });
            } else {
                self.cancel(m.id);
                self.settle.insert(m.id, Settle::Show { xyz: m.xyz, rgb: m.rgb, zone: m.zone });
            }
        }
    }

    /// Drops the animation of `id` (its overlay slot is hidden at the next tick).
    fn cancel(&mut self, id: u32) {
        let Some(i) = self.by_id.remove(&id) else { return };
        let gone = self.anims.swap_remove(i);
        if let Some(moved) = self.anims.get(i) {
            self.by_id.insert(moved.id, i);
        }
        self.overlay_gone.push(gone.slot);
        self.free.push(gone.slot);
    }

    /// Advances the animations to `now` and writes what changed to the GPU: finished points land
    /// in the base scene and leave the overlay in the same frame.
    pub fn tick(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, now: f64) {
        // Finished animations settle in the base scene.
        let mut i = 0;
        while i < self.anims.len() {
            if self.anims[i].progress(now) >= 1.0 {
                let a = self.anims[i];
                if a.shown_after() {
                    self.settle.insert(a.id, Settle::Show { xyz: a.to, rgb: a.to_rgb, zone: a.zone });
                }
                self.cancel(a.id);
            } else {
                i += 1;
            }
        }
        // Base: one message for everything that settled or hid this frame.
        if !self.settle.is_empty() {
            let mut adds = Vec::new();
            let mut n = 0u32;
            let mut removes = Vec::new();
            for (&id, s) in &self.settle {
                match *s {
                    Settle::Show { xyz, rgb, zone } => {
                        Record { id, position: xyz, color: rgb, zone }.write(&mut adds);
                        n += 1;
                    }
                    Settle::Hide => removes.push(id),
                }
            }
            self.settle.clear();
            self.base.write(device, queue, &adds, n, &removes, self.need);
        }
        // Overlay: the points in flight at their look now, and the slots that emptied.
        let mut recs = Vec::with_capacity(self.anims.len() * slots::ADD_BYTES);
        for a in &self.anims {
            let (position, color) = a.at(now, self.bg);
            Record { id: a.slot, position, color, zone: a.zone }.write(&mut recs);
        }
        let n = self.anims.len() as u32;
        let gone = std::mem::take(&mut self.overlay_gone);
        // A slot freed and taken again in the same frame is written, not hidden.
        let live: std::collections::HashSet<u32> = self.anims.iter().map(|a| a.slot).collect();
        let gone: Vec<u32> = gone.into_iter().filter(|s| !live.contains(s)).collect();
        self.overlay.write(device, queue, &recs, n, &gone, self.next_overlay.max(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add(id: u32, v: f32) -> AddRec {
        AddRec { id, xyz: [v, 0.0, 0.0], rgb: [200, 100, 50], zone: 0 }
    }

    #[test]
    fn easing_runs_from_zero_to_one_and_slows_down() {
        assert_eq!(ease_out(0.0), 0.0);
        assert_eq!(ease_out(1.0), 1.0);
        assert!(ease_out(0.5) > 0.8);
    }

    #[test]
    fn a_fade_in_runs_from_the_background_to_the_colour() {
        let a = Anim { id: 1, slot: 0, style: Style::FadeIn, from: [1.0; 3], to: [1.0; 3], from_rgb: [200, 100, 50], to_rgb: [200, 100, 50], zone: 0, t0: 10.0, dur: FADE_S };
        let bg = [0, 0, 0];
        assert_eq!(a.at(10.0, bg), ([1.0; 3], bg));
        assert_eq!(a.at(10.0 + FADE_S, bg), ([1.0; 3], [200, 100, 50]));
        assert!(a.shown_after());
    }

    #[test]
    fn a_glide_interpolates_the_position() {
        let a = Anim { id: 1, slot: 0, style: Style::Glide, from: [0.0; 3], to: [10.0, 0.0, 0.0], from_rgb: [0; 3], to_rgb: [100; 3], zone: 0, t0: 0.0, dur: GLIDE_S };
        let (p0, _) = a.at(0.0, [0; 3]);
        let (pm, cm) = a.at(GLIDE_S / 2.0, [0; 3]);
        let (p1, c1) = a.at(GLIDE_S, [0; 3]);
        assert_eq!(p0, [0.0; 3]);
        assert!(pm[0] > 5.0 && pm[0] < 10.0 && cm[0] > 50);
        assert_eq!((p1, c1), ([10.0, 0.0, 0.0], [100; 3]));
    }

    /// The CPU side of the layer needs no device: batches queue animations and settles.
    #[test]
    fn batches_queue_animations_and_hide_the_base_slot() {
        let mut l = LiveSlots::default();
        l.set_fades(true);
        l.apply(&Batch { adds: vec![add(0, 1.0), add(1, 2.0)], capacity: 2, animate: true, ..Default::default() }, 0.0);
        assert_eq!(l.animating(), 2);
        assert_eq!(l.next_overlay, 2);
        // Moving a point that is still fading in retargets it from where it is, in its slot.
        let m = MoveRec { id: 0, from: [1.0, 0.0, 0.0], from_rgb: [1; 3], xyz: [5.0, 0.0, 0.0], rgb: [9; 3], zone: 0 };
        l.apply(&Batch { moves: vec![m], capacity: 2, animate: true, ..Default::default() }, 0.05);
        assert_eq!(l.animating(), 2);
        let i = l.by_id[&0];
        assert_eq!(l.anims[i].style, Style::Glide);
        assert_eq!(l.anims[i].from, [1.0, 0.0, 0.0]);
        // A remove turns it into a fade-out that ends hidden.
        let r = RemoveRec { id: 0, xyz: [5.0, 0.0, 0.0], rgb: [9; 3], zone: 0 };
        l.apply(&Batch { removes: vec![r], capacity: 2, animate: true, ..Default::default() }, 0.1);
        let i = l.by_id[&0];
        assert_eq!(l.anims[i].style, Style::FadeOut);
        assert!(!l.anims[i].shown_after());
    }

    #[test]
    fn a_snap_batch_animates_nothing_and_cancels_a_running_animation() {
        let mut l = LiveSlots::default();
        l.set_fades(true);
        l.apply(&Batch { adds: vec![add(0, 1.0)], capacity: 1, animate: true, ..Default::default() }, 0.0);
        assert_eq!(l.animating(), 1);
        l.apply(&Batch { adds: vec![add(0, 3.0), add(1, 4.0)], capacity: 2, animate: false, ..Default::default() }, 0.01);
        assert_eq!(l.animating(), 0);
        assert!(matches!(l.settle[&0], Settle::Show { xyz, .. } if xyz[0] == 3.0));
        assert!(l.overlay_gone.contains(&0));
    }

    #[test]
    fn the_animation_budget_snaps_the_rest() {
        let mut l = LiveSlots::default();
        l.set_fades(true);
        let adds: Vec<AddRec> = (0..ANIM_BUDGET as u32 + 10).map(|i| add(i, i as f32)).collect();
        l.apply(&Batch { capacity: adds.len() as u32, adds, animate: true, ..Default::default() }, 0.0);
        assert_eq!(l.animating(), ANIM_BUDGET);
        assert_eq!(l.settle.values().filter(|s| matches!(s, Settle::Show { .. })).count(), 10);
    }

    /// The slot a scene holds as (shown, position, colour), read back from the GPU.
    fn slot(scene: &SlotScene, device: &wgpu::Device, queue: &wgpu::Queue, id: usize) -> (bool, [f32; 3], [u8; 3]) {
        let bytes = scene.read_slots(device, queue);
        let s = &bytes[id * 16..id * 16 + 16];
        let f = |o: usize| f32::from_le_bytes(s[o..o + 4].try_into().unwrap());
        (s[15] & 1 == 1, [f(0), f(4), f(8)], [s[12], s[13], s[14]])
    }

    /// Needs a GPU adapter; skipped (passes) without one, like pointblitz's own GPU tests.
    #[test]
    fn gpu_points_fade_in_glide_and_land_in_the_base_scene() {
        let Some((device, queue, _)) = pointblitz::core::headless::device() else {
            eprintln!("no GPU adapter - skipped");
            return;
        };
        let mut l = LiveSlots::new([0, 0, 0]);
        l.set_fades(true);
        // Two settled points.
        l.apply(&Batch { adds: vec![add(0, 0.0), add(1, 1.0)], capacity: 2, animate: false, ..Default::default() }, 0.0);
        l.tick(&device, &queue, 0.0);
        assert!(slot(&l.base.scene, &device, &queue, 0).0);
        assert_eq!(l.base.scene.points(), 2);
        assert_eq!(l.overlay.scene.points(), 0);

        // Point 1 glides to x = 10 while a new point 2 fades in.
        let m = MoveRec { id: 1, from: [1.0, 0.0, 0.0], from_rgb: [200, 100, 50], xyz: [10.0, 0.0, 0.0], rgb: [200, 100, 50], zone: 0 };
        l.apply(&Batch { adds: vec![add(2, 5.0)], moves: vec![m], capacity: 3, animate: true, ..Default::default() }, 1.0);
        l.tick(&device, &queue, 1.0 + GLIDE_S / 2.0);
        // Hidden in the base, shown in flight, between the two positions.
        assert!(!slot(&l.base.scene, &device, &queue, 1).0);
        assert_eq!(l.base.scene.points(), 2, "point 0 and the faded-in point 2");
        assert_eq!(l.overlay.scene.points(), 1, "the fade-in (150 ms) landed, the glide is in flight");
        let (shown, p, _) = slot(&l.overlay.scene, &device, &queue, l.anims[0].slot as usize);
        assert!(shown && p[0] > 1.0 && p[0] < 10.0, "{p:?}");
        // The faded-in point is in the base at full colour.
        assert_eq!(slot(&l.base.scene, &device, &queue, 2), (true, [5.0, 0.0, 0.0], [200, 100, 50]));

        // Everything lands: one surface, nothing left in flight.
        l.tick(&device, &queue, 1.0 + GLIDE_S + 0.01);
        assert_eq!(l.animating(), 0);
        assert_eq!(l.overlay.scene.points(), 0);
        assert_eq!(slot(&l.base.scene, &device, &queue, 1), (true, [10.0, 0.0, 0.0], [200, 100, 50]));
        assert_eq!(l.base.scene.points(), 3);

        // A remove fades out and leaves the base hidden at once.
        let r = RemoveRec { id: 0, xyz: [0.0; 3], rgb: [200, 100, 50], zone: 0 };
        l.apply(&Batch { removes: vec![r], capacity: 3, animate: true, ..Default::default() }, 3.0);
        l.tick(&device, &queue, 3.05);
        assert_eq!((l.base.scene.points(), l.overlay.scene.points()), (2, 1));
        l.tick(&device, &queue, 3.2);
        assert_eq!((l.base.scene.points(), l.overlay.scene.points()), (2, 0));
    }

    #[test]
    fn without_fades_adds_and_removes_are_instant_and_moves_glide() {
        let mut l = LiveSlots::default();
        let m = MoveRec { id: 1, from: [0.0; 3], from_rgb: [1; 3], xyz: [3.0, 0.0, 0.0], rgb: [2; 3], zone: 0 };
        let r = RemoveRec { id: 2, xyz: [0.0; 3], rgb: [1; 3], zone: 0 };
        l.apply(&Batch { adds: vec![add(0, 1.0)], moves: vec![m], removes: vec![r], capacity: 3, animate: true }, 0.0);
        assert_eq!(l.animating(), 1, "only the move is in flight");
        assert!(matches!(l.settle[&0], Settle::Show { .. }));
        assert!(matches!(l.settle[&2], Settle::Hide));
    }

    #[test]
    fn reset_forgets_everything() {
        let mut l = LiveSlots::default();
        l.set_fades(true);
        l.apply(&Batch { adds: vec![add(0, 1.0)], capacity: 1, animate: true, ..Default::default() }, 0.0);
        l.reset();
        assert_eq!((l.animating(), l.next_overlay, l.settle.len()), (0, 0, 0));
    }
}
