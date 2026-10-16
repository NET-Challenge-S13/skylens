//! `<input type="range">` for the planner altitude. Blitz has no range control, so a custom
//! widget on the template's input draws a track and thumb in the input's `accent-color`
//! (`--control-ink`) and turns pointer drags into values on the input's min/max/step.

use blitz_dom::node::ComputedStyles;
use blitz_dom::{IntrinsicSizes, Widget};
use blitz_traits::events::UiEvent;
use skylens_viewer::anyrender::{self, PaintScene as _, Scene};
use skylens_viewer::peniko::kurbo::{Affine, Circle, Point, RoundedRect};
use skylens_viewer::peniko::{Color, Fill};
use std::cell::RefCell;
use std::rc::Rc;

/// Value model of a range input.
#[derive(Debug, Clone, PartialEq)]
pub struct Range {
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub value: f64,
    /// Set when the operator moved it; the screen clears it after applying (`input` event).
    pub changed: bool,
    width_css: f64,
    dragging: bool,
    dirty: bool,
}

impl Range {
    pub fn new(min: f64, max: f64, step: f64, value: f64) -> Range {
        Range { min, max, step, value: snap(value, min, max, step), changed: false, width_css: 120.0, dragging: false, dirty: true }
    }

    /// Fraction of the track at the value.
    pub fn fraction(&self) -> f64 {
        if self.max > self.min { (self.value - self.min) / (self.max - self.min) } else { 0.0 }
    }

    /// Sets the value from a position along the track (CSS px from its left edge).
    pub fn set_from_x(&mut self, x: f64) {
        let usable = (self.width_css - THUMB).max(1.0);
        let f = ((x - THUMB / 2.0) / usable).clamp(0.0, 1.0);
        let v = snap(self.min + f * (self.max - self.min), self.min, self.max, self.step);
        if v != self.value {
            self.value = v;
            self.changed = true;
            self.dirty = true;
        }
    }

    fn pointer(&mut self, ev: &UiEvent) {
        match ev {
            UiEvent::PointerDown(e) => {
                self.dragging = true;
                self.set_from_x(f64::from(e.coords.client_x));
            }
            UiEvent::PointerMove(e) if self.dragging => {
                if e.buttons.is_empty() {
                    self.dragging = false;
                } else {
                    self.set_from_x(f64::from(e.coords.client_x));
                }
            }
            UiEvent::PointerUp(_) | UiEvent::PointerCancel(_) => self.dragging = false,
            _ => {}
        }
    }
}

/// Snaps to the step grid from `min`, within bounds (HTML range sanitisation).
pub fn snap(v: f64, min: f64, max: f64, step: f64) -> f64 {
    let v = v.clamp(min, max);
    if step <= 0.0 {
        return v;
    }
    let k = ((v - min) / step).round();
    (min + k * step).min(max)
}

/// Thumb diameter (CSS px).
const THUMB: f64 = 16.0;
const TRACK_H: f64 = 4.0;

pub struct RangeWidget {
    state: Rc<RefCell<Range>>,
}

impl RangeWidget {
    pub fn boxed(state: Rc<RefCell<Range>>) -> Box<dyn Widget> {
        Box::new(RangeWidget { state })
    }
}

impl Widget for RangeWidget {
    fn intrinsic_sizes(&self) -> IntrinsicSizes {
        IntrinsicSizes { width: Some(120.0), height: Some(THUMB as f32), ratio: None }
    }

    fn requires_redraw(&self) -> bool {
        self.state.borrow().dirty
    }

    fn handle_event(&mut self, event: &UiEvent) {
        self.state.borrow_mut().pointer(event);
    }

    fn paint(&mut self, _ctx: &mut dyn anyrender::RenderContext, _styles: &ComputedStyles, width: u32, height: u32, scale: f64) -> Scene {
        let mut scene = Scene::new();
        let mut s = self.state.borrow_mut();
        s.dirty = false;
        if width == 0 || height == 0 {
            return scene;
        }
        let scale = if scale > 0.0 { scale } else { 1.0 };
        let (w, h) = (f64::from(width) / scale, f64::from(height) / scale);
        s.width_css = w;
        let base = Affine::scale(scale);
        let accent = Color::from_rgb8(0x5e, 0xc8, 0xff);
        let rest = Color::from_rgba8(255, 255, 255, 46);
        let cy = h / 2.0;
        let x0 = THUMB / 2.0;
        let x1 = (w - THUMB / 2.0).max(x0);
        let xv = x0 + (x1 - x0) * s.fraction();
        let track = |a: f64, b: f64| RoundedRect::new(a, cy - TRACK_H / 2.0, b, cy + TRACK_H / 2.0, TRACK_H / 2.0);
        scene.fill(Fill::NonZero, base, rest, None, &track(x0, x1));
        scene.fill(Fill::NonZero, base, accent, None, &track(x0, xv));
        scene.fill(Fill::NonZero, base, accent, None, &Circle::new(Point::new(xv, cy), THUMB / 2.0));
        scene
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapping_follows_min_and_step() {
        assert_eq!(snap(61.0, 20.0, 200.0, 5.0), 60.0);
        assert_eq!(snap(63.0, 20.0, 200.0, 5.0), 65.0);
        assert_eq!(snap(500.0, 20.0, 200.0, 5.0), 200.0);
        assert_eq!(snap(0.0, 20.0, 200.0, 5.0), 20.0);
    }

    #[test]
    fn dragging_sets_values_along_the_track() {
        let mut r = Range::new(20.0, 200.0, 5.0, 60.0);
        r.set_from_x(8.0);
        assert_eq!(r.value, 20.0);
        assert!(r.changed);
        r.set_from_x(112.0);
        assert_eq!(r.value, 200.0);
        assert!((r.fraction() - 1.0).abs() < 1e-12);
    }
}
