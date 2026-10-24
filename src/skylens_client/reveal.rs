//! Arrival reveal (legacy `splatReveal.ts`): each reconstruction segment fades in when it arrives
//! and firms up as its levels land, and a segment only starts to appear once the segment ahead of
//! it (lower index, earlier in the flight) has faded most of the way in. Arrivals are never
//! delayed; only their appearance is ordered, so the reconstruction reads as advancing along the
//! flight instead of the whole strip flickering in at assorted opacities.
//!
//! The legacy board faded slabs of one final splat scene; here every segment is its own point
//! cloud (and its mesh blocks), so the per-segment opacity is applied to the cloud directly.

use skylens_shared::proto::model::v1 as model;
use std::collections::BTreeMap;

/// Seconds for a freshly revealed segment to fade 0 -> 1 (legacy `CONFIG.reveal.fadeSeconds`).
pub const FADE_SECONDS: f64 = 1.2;
/// How far the segment ahead must have faded in before the next may start (legacy `chainGate`).
pub const CHAIN_GATE: f32 = 0.78;

/// How solid a segment renders at a level (legacy `alphaForLevel`): a first delivery appears
/// translucent and the final level is opaque. The legacy ladder had four levels (0.62, 0.8, 0.95,
/// final 1.0); the pipeline now has preview and refined, which take the first and the final value.
pub fn alpha_for_level(level: model::Level) -> f32 {
    match level {
        model::Level::Refined => 1.0,
        model::Level::Preview | model::Level::Unspecified => 0.62,
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Slab {
    /// What arrivals ask for.
    pending: f32,
    /// What the fade is heading to (released by the chain).
    target: f32,
    /// Current opacity.
    fade: f32,
}

/// Per-segment fade state. Kept across reconnects, so a replay does not fade the board in again.
#[derive(Debug, Clone, Default)]
pub struct RevealFade {
    slabs: BTreeMap<u32, Slab>,
}

impl RevealFade {
    pub fn new() -> Self {
        Self::default()
    }

    /// A segment's chunk landed at `level`: queue its fade (only ever upwards).
    pub fn note_arrival(&mut self, segment: u32, level: model::Level) {
        let target = alpha_for_level(level);
        let slab = self.slabs.entry(segment).or_default();
        if target > slab.pending {
            slab.pending = target;
        }
    }

    /// Eases every segment toward its target; `dt` in seconds.
    pub fn update(&mut self, dt: f64) {
        // Release queued targets along the chain. The gate reads fades, not targets, so the order
        // of this pass does not matter.
        let fades: Vec<(u32, f32)> = self.slabs.iter().map(|(s, v)| (*s, v.fade)).collect();
        for (i, (seg, _)) in fades.iter().enumerate() {
            let slab = self.slabs.get_mut(seg).expect("listed above");
            if slab.pending <= slab.target {
                continue;
            }
            let ahead = i.checked_sub(1).map(|j| fades[j].1);
            if ahead.is_none_or(|f| f >= CHAIN_GATE) {
                slab.target = slab.pending;
            }
        }
        let step = (dt.max(0.0) / FADE_SECONDS.max(0.05)) as f32;
        for slab in self.slabs.values_mut() {
            if slab.target > slab.fade {
                slab.fade = (slab.fade + step).min(slab.target);
            }
        }
    }

    /// Current opacity of a segment (0 when it never arrived).
    pub fn alpha(&self, segment: u32) -> f32 {
        self.slabs.get(&segment).map_or(0.0, |s| s.fade)
    }

    /// Whether any segment is still fading or waiting for the chain.
    pub fn animating(&self) -> bool {
        self.slabs.values().any(|s| s.fade < s.pending)
    }

    /// The first segment's progress toward its first opacity, 0..1, while the board's first
    /// geometry is still coming in; None once it is in (or nothing arrived).
    pub fn first_progress(&self) -> Option<f32> {
        let (_, first) = self.slabs.iter().next()?;
        if first.pending <= 0.0 || first.fade >= first.pending {
            return None;
        }
        Some((first.fade / first.pending).clamp(0.0, 1.0))
    }

    pub fn segments(&self) -> impl Iterator<Item = (u32, f32)> + '_ {
        self.slabs.iter().map(|(s, v)| (*s, v.fade))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model::Level::{Preview, Refined};

    fn run(f: &mut RevealFade, seconds: f64) {
        let steps = (seconds * 60.0).round() as usize;
        for _ in 0..steps {
            f.update(1.0 / 60.0);
        }
    }

    #[test]
    fn levels_map_to_the_legacy_opacities() {
        assert_eq!(alpha_for_level(Preview), 0.62);
        assert_eq!(alpha_for_level(Refined), 1.0);
    }

    #[test]
    fn a_segment_fades_in_over_the_fade_time() {
        let mut f = RevealFade::new();
        assert_eq!(f.alpha(0), 0.0);
        f.note_arrival(0, Refined);
        run(&mut f, 0.6);
        assert!((f.alpha(0) - 0.5).abs() < 0.02, "{}", f.alpha(0));
        run(&mut f, 0.7);
        assert_eq!(f.alpha(0), 1.0);
        assert!(!f.animating());
    }

    #[test]
    fn the_next_segment_waits_for_the_one_ahead() {
        let mut f = RevealFade::new();
        f.note_arrival(0, Refined);
        f.note_arrival(1, Refined);
        run(&mut f, 0.5);
        assert!(f.alpha(0) > 0.3);
        assert_eq!(f.alpha(1), 0.0, "held until segment 0 passes the gate");
        // Segment 0 crosses 0.78 after ~0.94 s; then 1 starts.
        run(&mut f, 0.6);
        assert!(f.alpha(0) >= CHAIN_GATE);
        assert!(f.alpha(1) > 0.0);
    }

    #[test]
    fn a_preview_does_not_hold_the_chain_beyond_its_own_opacity() {
        let mut f = RevealFade::new();
        f.note_arrival(0, Preview);
        f.note_arrival(1, Preview);
        run(&mut f, 3.0);
        // 0.62 never reaches the gate, so 1 stays queued until 0 is refined.
        assert!((f.alpha(0) - 0.62).abs() < 1e-6);
        assert_eq!(f.alpha(1), 0.0);
        f.note_arrival(0, Refined);
        run(&mut f, 3.0);
        assert_eq!(f.alpha(0), 1.0);
        assert!((f.alpha(1) - 0.62).abs() < 1e-6);
    }

    #[test]
    fn levels_only_go_up() {
        let mut f = RevealFade::new();
        f.note_arrival(3, Refined);
        f.note_arrival(3, Preview);
        run(&mut f, 2.0);
        assert_eq!(f.alpha(3), 1.0);
    }

    #[test]
    fn first_progress_reports_the_first_segment_only_while_it_fades() {
        let mut f = RevealFade::new();
        assert_eq!(f.first_progress(), None);
        f.note_arrival(2, Refined);
        assert_eq!(f.first_progress(), Some(0.0));
        run(&mut f, 0.6);
        let p = f.first_progress().unwrap();
        assert!(p > 0.4 && p < 0.6);
        run(&mut f, 1.0);
        assert_eq!(f.first_progress(), None);
    }
}
