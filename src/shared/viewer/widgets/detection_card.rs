//! Detection card and camera-state chip (legacy board `overlay.ts`), built into
//! `#overlay-status`:
//!
//! ```html
//! <div class="detect-card kind-person is-open">
//!   <div class="detect-card__kicker">생존자 감지</div>
//!   <div class="detect-card__label">생존자</div>
//!   <div class="detect-card__conf">확신도 82%</div>
//!   <div class="detect-card__bar"><span style="width: 82%"></span></div>
//!   <button class="detect-card__btn" type="button">탐지 확인</button>
//! </div>
//! <div class="cam-state">CAM · SYNCED</div>
//! ```
//!
//! A click on the button is recognised with [`DetectionCard::is_confirm`]; the screen then marks
//! the detection confirmed, closes the card and tells the camera to return.

use crate::camera::SyncState;
use crate::dom::{Dom, Sel};
use crate::shell::UiInput;
use blitz_dom::NodeId;
use skylens_shared::proto::payload::v1 as payload;

pub const OVERLAY_ID: &str = "overlay-status";
pub const CONFIRM_CLASS: &str = "detect-card__btn";

/// The two looks of a detection (legacy `det.kind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardKind {
    Person,
    Danger,
}

impl CardKind {
    pub fn of(category: payload::DetectionCategory) -> CardKind {
        match category {
            payload::DetectionCategory::Person => CardKind::Person,
            _ => CardKind::Danger,
        }
    }

    pub fn class(self) -> &'static str {
        match self {
            CardKind::Person => "kind-person",
            CardKind::Danger => "kind-danger",
        }
    }

    /// Card kicker (legacy `KICKER_LABEL`).
    pub fn kicker(self) -> &'static str {
        match self {
            CardKind::Person => "생존자 감지",
            CardKind::Danger => "위험구역 감지",
        }
    }

    pub fn toast(self) -> crate::widgets::ToastKind {
        match self {
            CardKind::Person => crate::widgets::ToastKind::Person,
            CardKind::Danger => crate::widgets::ToastKind::Danger,
        }
    }
}

/// The label shown for a detection: its own label, else a name for its category.
pub fn detection_label(d: &payload::Detection) -> String {
    if !d.label.trim().is_empty() {
        return d.label.clone();
    }
    match d.category() {
        payload::DetectionCategory::Person => "생존자",
        payload::DetectionCategory::Fire => "화재",
        payload::DetectionCategory::Collapse => "붕괴",
        payload::DetectionCategory::RoadBlocked => "도로 차단",
        payload::DetectionCategory::Unspecified => "위험구역",
    }
    .to_string()
}

/// `확신도 N%` and the bar width for a confidence in 0..1.
pub fn confidence_view(confidence: f64) -> (String, String) {
    let pct = if confidence.is_finite() { (confidence * 100.0).round().clamp(0.0, 100.0) as i64 } else { 0 };
    (format!("확신도 {pct}%"), format!("{pct}%"))
}

#[derive(Debug, Clone)]
pub struct DetectionCard {
    card: NodeId,
    kicker: NodeId,
    label: NodeId,
    conf: NodeId,
    fill: NodeId,
    cam_state: NodeId,
    open: Option<String>,
    cam_text: String,
}

impl DetectionCard {
    /// Builds the card and the camera-state chip in the overlay (normally `#overlay-status`).
    pub fn mount<'s>(dom: &mut Dom<'_>, overlay: impl Into<Sel<'s>>) -> Option<DetectionCard> {
        let overlay = dom.find(overlay)?;
        let card = dom.create(overlay, "div", &[("class", "detect-card")], None)?;
        let kicker = dom.create(card, "div", &[("class", "detect-card__kicker")], None)?;
        let label = dom.create(card, "div", &[("class", "detect-card__label")], None)?;
        let conf = dom.create(card, "div", &[("class", "detect-card__conf")], None)?;
        let bar = dom.create(card, "div", &[("class", "detect-card__bar")], None)?;
        let fill = dom.create(bar, "span", &[], None)?;
        dom.create(card, "button", &[("class", CONFIRM_CLASS), ("type", "button")], Some("탐지 확인"))?;
        let cam_state = dom.create(overlay, "div", &[("class", "cam-state")], None)?;
        Some(DetectionCard { card, kicker, label, conf, fill, cam_state, open: None, cam_text: String::new() })
    }

    /// Opens the card on a detection.
    pub fn open(&mut self, dom: &mut Dom<'_>, d: &payload::Detection) {
        let kind = CardKind::of(d.category());
        dom.remove_class(self.card, "kind-person");
        dom.remove_class(self.card, "kind-danger");
        dom.add_class(self.card, kind.class());
        dom.set_text(self.kicker, kind.kicker());
        dom.set_text(self.label, &detection_label(d));
        let (text, width) = confidence_view(d.confidence);
        dom.set_text(self.conf, &text);
        dom.set_style(self.fill, "width", &width);
        dom.add_class(self.card, "is-open");
        self.open = Some(d.id.clone());
    }

    pub fn close(&mut self, dom: &mut Dom<'_>) {
        self.open = None;
        dom.remove_class(self.card, "is-open");
    }

    /// The detection the card shows, if open.
    pub fn open_id(&self) -> Option<&str> {
        self.open.as_deref()
    }

    /// Updates the `.cam-state` chip; closes the card when the camera is back in sync.
    pub fn set_camera_state(&mut self, dom: &mut Dom<'_>, state: SyncState) {
        let text = state.label();
        if self.cam_text != text {
            dom.set_text(self.cam_state, text);
            self.cam_text = text.to_string();
        }
        if state == SyncState::Synced && self.open.is_some() {
            self.close(dom);
        }
    }

    /// Whether a click hit the card's confirm button.
    pub fn is_confirm(input: &UiInput) -> bool {
        input.is_click() && input.closest_class(CONFIRM_CLASS).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_and_labels_follow_the_legacy_card() {
        assert_eq!(CardKind::of(payload::DetectionCategory::Person), CardKind::Person);
        assert_eq!(CardKind::of(payload::DetectionCategory::Fire), CardKind::Danger);
        assert_eq!(CardKind::Person.kicker(), "생존자 감지");
        assert_eq!(CardKind::Danger.class(), "kind-danger");
        let mut d = payload::Detection { label: "2층 창가 생존자".into(), ..Default::default() };
        assert_eq!(detection_label(&d), "2층 창가 생존자");
        d.label.clear();
        d.category = payload::DetectionCategory::Collapse as i32;
        assert_eq!(detection_label(&d), "붕괴");
    }

    #[test]
    fn confidence_is_a_rounded_percentage() {
        assert_eq!(confidence_view(0.824), ("확신도 82%".to_string(), "82%".to_string()));
        assert_eq!(confidence_view(1.7).1, "100%");
        assert_eq!(confidence_view(f64::NAN).0, "확신도 0%");
    }
}
