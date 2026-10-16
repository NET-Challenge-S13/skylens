//! The detection card and the camera-state chip (legacy `ui/overlay.ts`). The template already
//! holds the markup the legacy code built into `#overlay-status` (card, then `.cam-state`, then
//! the waiting banner), so this binds to those nodes instead of building new ones, and writes them
//! the way `openCard` / `closeCard` / `refreshCamState` did.

use skylens_shared::proto::payload::v1 as payload;
use skylens_viewer::camera::SyncState;
use skylens_viewer::widgets::DetectionCard;
use skylens_viewer::widgets::detection_card::{CardKind, confidence_view, detection_label};
use skylens_viewer::{Dom, NodeId, UiInput, q};

pub struct Card {
    card: NodeId,
    kicker: NodeId,
    label: NodeId,
    conf: NodeId,
    fill: NodeId,
    cam_state: NodeId,
    open: Option<String>,
    cam_text: String,
}

impl Card {
    /// Finds the template's card and chip.
    pub fn bind(dom: &Dom<'_>) -> Option<Card> {
        let card = dom.find(q("#overlay-status .detect-card"))?;
        Some(Card {
            card,
            kicker: dom.find_in(card, ".detect-card__kicker")?,
            label: dom.find_in(card, ".detect-card__label")?,
            conf: dom.find_in(card, ".detect-card__conf")?,
            fill: dom.find_in(card, ".detect-card__bar span")?,
            cam_state: dom.find(q("#overlay-status .cam-state"))?,
            open: None,
            cam_text: dom.text(q("#overlay-status .cam-state")).unwrap_or_default(),
        })
    }

    /// Opens the card on a detection (the camera started focusing it).
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

    /// Closes the card; kind and texts stay, like the legacy card.
    pub fn close(&mut self, dom: &mut Dom<'_>) {
        self.open = None;
        dom.remove_class(self.card, "is-open");
    }

    /// Whether a click hit the card's `탐지 확인` button.
    pub fn is_confirm(input: &UiInput) -> bool {
        DetectionCard::is_confirm(input)
    }

    pub fn open_id(&self) -> Option<&str> {
        self.open.as_deref()
    }

    /// `.cam-state` text, written only when it changes.
    pub fn set_camera_state(&mut self, dom: &mut Dom<'_>, state: SyncState) {
        let text = state.label();
        if self.cam_text != text {
            dom.set_text(self.cam_state, text);
            self.cam_text = text.to_string();
        }
    }
}
