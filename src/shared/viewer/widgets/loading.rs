//! Full-screen loading overlay (legacy `loadingScreen.ts`), appended to `<body>`:
//!
//! ```html
//! <div class="loading-screen">
//!   <div class="loading-screen__box">
//!     <div class="loading-screen__title">관제탑 · 실지형 장면 로딩</div>
//!     <div class="loading-screen__bar"><span style="width: 40%"></span></div>
//!     <div class="loading-screen__text">40%</div>
//!   </div>
//! </div>
//! ```
//!
//! `done` adds `is-hidden` and removes the element 400 ms later; `fail` shows the message with
//! `is-error` and does the same after 2.5 s.

use crate::dom::{Dom, q};
use blitz_dom::NodeId;

const HIDE_TO_REMOVE_MS: f64 = 400.0;
const FAIL_TO_HIDE_MS: f64 = 2500.0;

#[derive(Debug, Clone)]
pub struct LoadingScreen {
    root: NodeId,
    title: NodeId,
    fill: NodeId,
    text: NodeId,
    hide_at: Option<f64>,
    remove_at: Option<f64>,
    removed: bool,
}

/// The bar width and text for a progress value (legacy `progress`).
pub fn progress_view(percent: f64, label: Option<&str>) -> (String, String) {
    let p = if percent.is_finite() { percent.clamp(0.0, 100.0) } else { 0.0 };
    let width = format!("{}%", trim_float(p));
    let text = label.map_or_else(|| format!("{}%", p.round() as i64), str::to_string);
    (width, text)
}

fn trim_float(v: f64) -> String {
    let s = format!("{v:.3}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    s.to_string()
}

impl LoadingScreen {
    /// Builds the overlay at the end of `<body>`.
    pub fn mount(dom: &mut Dom<'_>, title: &str) -> Option<LoadingScreen> {
        let root = dom.create(q("body"), "div", &[("class", "loading-screen")], None)?;
        let bx = dom.create(root, "div", &[("class", "loading-screen__box")], None)?;
        let title = dom.create(bx, "div", &[("class", "loading-screen__title")], Some(title))?;
        let bar = dom.create(bx, "div", &[("class", "loading-screen__bar")], None)?;
        let fill = dom.create(bar, "span", &[], None)?;
        let text = dom.create(bx, "div", &[("class", "loading-screen__text")], None)?;
        Some(LoadingScreen { root, title, fill, text, hide_at: None, remove_at: None, removed: false })
    }

    pub fn progress(&mut self, dom: &mut Dom<'_>, percent: f64, label: Option<&str>) {
        if self.removed {
            return;
        }
        let (width, text) = progress_view(percent, label);
        dom.set_style(self.fill, "width", &width);
        dom.set_text(self.text, &text);
    }

    /// Fades out and removes the overlay.
    pub fn done(&mut self, dom: &mut Dom<'_>, now_ms: f64) {
        if self.removed || self.remove_at.is_some() {
            return;
        }
        dom.add_class(self.root, "is-hidden");
        self.hide_at = None;
        self.remove_at = Some(now_ms + HIDE_TO_REMOVE_MS);
    }

    pub fn fail(&mut self, dom: &mut Dom<'_>, message: &str, now_ms: f64) {
        if self.removed {
            return;
        }
        dom.set_text(self.title, "장면 로드 실패");
        dom.set_text(self.text, message);
        dom.add_class(self.root, "is-error");
        self.hide_at = Some(now_ms + FAIL_TO_HIDE_MS);
    }

    /// Runs the pending hide/remove steps. Returns true when the DOM changed.
    pub fn tick(&mut self, dom: &mut Dom<'_>, now_ms: f64) -> bool {
        if self.removed {
            return false;
        }
        if self.hide_at.is_some_and(|t| now_ms >= t) {
            self.hide_at = None;
            self.done(dom, now_ms);
            return true;
        }
        if self.remove_at.is_some_and(|t| now_ms >= t) {
            dom.remove(self.root);
            self.removed = true;
            return true;
        }
        false
    }

    pub fn is_removed(&self) -> bool {
        self.removed
    }

    /// Whether `done` or `fail` was called (the overlay is on its way out).
    pub fn is_finishing(&self) -> bool {
        self.removed || self.hide_at.is_some() || self.remove_at.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_is_clamped_and_labelled_like_the_legacy_bar() {
        assert_eq!(progress_view(42.4, None), ("42.4%".to_string(), "42%".to_string()));
        assert_eq!(progress_view(150.0, None), ("100%".to_string(), "100%".to_string()));
        assert_eq!(progress_view(-3.0, Some("지형 타일 3/9")), ("0%".to_string(), "지형 타일 3/9".to_string()));
        assert_eq!(progress_view(f64::NAN, None).0, "0%");
    }
}
