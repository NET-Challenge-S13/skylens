//! The point layer's side of the live log: drains a view's [`LayerOp`]s, colours the batches for
//! the view's colour mode, and hands them to the animated slot layer ([`LiveSlots`]).
//!
//! Colour modes are applied here, before the layer sees a record, so changing the mode (or the
//! height range) rebuilds the layer from the present state without animation.

use crate::board::{Board, LiveView};
use crate::delta_log::LayerOp;
use crate::panels_live::ColorMode;
use crate::points;
use skylens_viewer::Gpu3d;
use skylens_viewer::live_slots::{AddRec, Batch, MoveRec, RemoveRec};
use std::rc::Rc;

/// The 0.25 + 0.75 * hsv(fract(zone * 0.618034)) colour of the cumulus3d viewer's zone mode.
pub fn zone_color(zone: u8) -> [u8; 3] {
    let h = (f64::from(zone) * 0.618_034).fract();
    let c = |off: f64| ((((h * 6.0 + off) % 6.0) - 3.0).abs() - 1.0).clamp(0.0, 1.0);
    [c(0.0), c(4.0), c(2.0)].map(|v| ((0.25 + 0.75 * v) * 255.0).round() as u8)
}

/// How a view colours the slots.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Paint {
    Rgb,
    Zone,
    /// Over this range of heights (metres, ENU z).
    Height(f64, f64),
}

impl Paint {
    /// The colour of a point at `z` metres in `zone` whose stream colour is `rgb`.
    pub fn color(self, z: f32, zone: u8, rgb: [u8; 3]) -> [u8; 3] {
        match self {
            Paint::Rgb => rgb,
            Paint::Zone => zone_color(zone),
            Paint::Height(lo, hi) => points::height_color(f64::from(z), lo, hi),
        }
    }

    /// Paint for a mode; the height range is the board's.
    pub fn of(mode: ColorMode, range: Option<(f64, f64)>) -> Paint {
        match (mode, range) {
            (ColorMode::Zone, _) => Paint::Zone,
            (ColorMode::Height, Some((lo, hi))) => Paint::Height(lo, hi),
            _ => Paint::Rgb,
        }
    }

    /// Whether a rebuild is due when going from `self` to `new` (a small change of the height
    /// range does not repaint every point).
    fn differs(self, new: Paint) -> bool {
        match (self, new) {
            (Paint::Height(a, b), Paint::Height(c, d)) => {
                let span = (b - a).abs().max(1.0);
                (a - c).abs() > span * 0.1 || (b - d).abs() > span * 0.1
            }
            (a, b) => std::mem::discriminant(&a) != std::mem::discriminant(&b),
        }
    }
}

/// A copy of `b` painted with `paint`.
pub fn paint_batch(b: &Batch, paint: Paint) -> Batch {
    if paint == Paint::Rgb {
        return b.clone();
    }
    Batch {
        removes: b.removes.iter().map(|r| RemoveRec { rgb: paint.color(r.xyz[2], r.zone, r.rgb), ..*r }).collect(),
        adds: b.adds.iter().map(|a| AddRec { rgb: paint.color(a.xyz[2], a.zone, a.rgb), ..*a }).collect(),
        moves: b
            .moves
            .iter()
            .map(|m| MoveRec {
                from_rgb: paint.color(m.from[2], m.zone, m.from_rgb),
                rgb: paint.color(m.xyz[2], m.zone, m.rgb),
                ..*m
            })
            .collect(),
        capacity: b.capacity,
        animate: b.animate,
    }
}

/// One view's connection to the live log.
pub struct LiveLayer {
    view: LiveView,
    paint: Option<Paint>,
    /// Board clock of the last height-range rebuild (seconds), to not repaint every frame.
    last_rebuild: f64,
}

impl LiveLayer {
    pub fn new(view: LiveView) -> LiveLayer {
        LiveLayer { view, paint: None, last_rebuild: f64::NEG_INFINITY }
    }

    /// The GPU is gone: start over from the present state.
    pub fn lost(&mut self, board: &mut Board) {
        board.live_ops_reset(self.view);
        self.paint = None;
    }

    /// Once per frame: follow the log, then step the animations to `now_s`.
    pub fn pump(&mut self, gpu: &mut Gpu3d, board: &mut Board, paint: Paint, bg: [u8; 3], now_s: f64) {
        let mut rebuild = self.paint.is_none();
        if let Some(have) = self.paint
            && have != paint
            && (std::mem::discriminant(&have) != std::mem::discriminant(&paint) || (have.differs(paint) && now_s - self.last_rebuild > 1.0))
        {
            rebuild = true;
        }
        if rebuild {
            if self.paint.is_some() {
                board.live_ops_reset(self.view);
            }
            self.paint = Some(paint);
            self.last_rebuild = now_s;
        }
        let paint = self.paint.unwrap_or(paint);
        let ops = board.live_ops(self.view);
        let (reset, batches) = collect(&ops, paint);
        let active = gpu.clouds.live_ref().is_some_and(|l| l.animating() > 0);
        if reset || !batches.is_empty() || active {
            gpu.live_step(bg, reset, &batches, now_s);
        }
    }
}

/// The ops as one reset flag and the painted batches after it. Batches that arrive together are
/// a replay (a late join, a burst): only the last one animates.
pub fn collect(ops: &[LayerOp], paint: Paint) -> (bool, Vec<Batch>) {
    let mut reset = false;
    let mut batches: Vec<Rc<Batch>> = Vec::new();
    for op in ops {
        match op {
            LayerOp::Reset => {
                reset = true;
                batches.clear();
            }
            LayerOp::Apply(b) => batches.push(b.clone()),
        }
    }
    let n = batches.len();
    let out = batches
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let mut p = paint_batch(b, paint);
            p.animate &= i + 1 == n;
            p
        })
        .collect();
    (reset, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add(id: u32, z: f32) -> AddRec {
        AddRec { id, xyz: [0.0, 0.0, z], rgb: [10, 20, 30], zone: 3 }
    }

    #[test]
    fn zone_colours_differ_and_are_lit() {
        let c: Vec<[u8; 3]> = (0..6).map(zone_color).collect();
        for (i, a) in c.iter().enumerate() {
            assert!(a.iter().all(|v| *v >= 63), "{a:?}");
            assert!(c.iter().skip(i + 1).all(|b| a != b));
        }
    }

    #[test]
    fn painting_replaces_the_colour_only() {
        let b = Batch { adds: vec![add(1, 0.0), add(2, 10.0)], capacity: 3, animate: true, ..Default::default() };
        let p = paint_batch(&b, Paint::Height(0.0, 10.0));
        assert_eq!(p.adds[0].id, 1);
        assert_ne!(p.adds[0].rgb, p.adds[1].rgb);
        assert_ne!(p.adds[0].rgb, [10, 20, 30]);
        assert_eq!(paint_batch(&b, Paint::Rgb), b);
        assert_eq!(paint_batch(&b, Paint::Zone).adds[0].rgb, zone_color(3));
    }

    #[test]
    fn only_the_last_of_a_burst_animates_and_a_reset_drops_the_earlier() {
        let mk = |animate| Rc::new(Batch { adds: vec![add(0, 0.0)], capacity: 1, animate, ..Default::default() });
        let ops = vec![LayerOp::Apply(mk(true)), LayerOp::Reset, LayerOp::Apply(mk(true)), LayerOp::Apply(mk(true))];
        let (reset, b) = collect(&ops, Paint::Rgb);
        assert!(reset);
        assert_eq!(b.len(), 2);
        assert!(!b[0].animate && b[1].animate);
    }

    #[test]
    fn a_small_height_range_change_does_not_repaint() {
        let a = Paint::Height(0.0, 10.0);
        assert!(!a.differs(Paint::Height(0.2, 10.5)));
        assert!(a.differs(Paint::Height(0.0, 14.0)));
        assert!(a.differs(Paint::Zone));
    }
}
