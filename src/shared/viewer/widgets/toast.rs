//! Transient toasts in `#toast-root` (legacy `toast.ts`): `<div class="toast kind-…">message</div>`,
//! stacked, each removed 3.2 s after it appeared.

use crate::dom::Dom;
use blitz_dom::NodeId;

pub const TOAST_LIFETIME_MS: f64 = 3200.0;
pub const TOAST_ROOT_ID: &str = "toast-root";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToastKind {
    Person,
    Danger,
    Info,
}

impl ToastKind {
    /// The toast's `class` attribute (info has no kind class).
    pub fn class(self) -> &'static str {
        match self {
            ToastKind::Person => "toast kind-person",
            ToastKind::Danger => "toast kind-danger",
            ToastKind::Info => "toast",
        }
    }
}

/// The toasts on screen.
#[derive(Debug, Default)]
pub struct Toasts {
    live: Vec<(NodeId, f64)>,
}

impl Toasts {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a toast to `#toast-root`; nothing happens when the root is missing.
    pub fn show(&mut self, dom: &mut Dom<'_>, message: &str, kind: ToastKind, now_ms: f64) {
        if let Some(node) = dom.create(TOAST_ROOT_ID, "div", &[("class", kind.class())], Some(message)) {
            self.live.push((node, now_ms + TOAST_LIFETIME_MS));
        }
    }

    /// Removes expired toasts. Returns true when any was removed.
    pub fn tick(&mut self, dom: &mut Dom<'_>, now_ms: f64) -> bool {
        let mut removed = false;
        self.live.retain(|(node, expires)| {
            if *expires <= now_ms {
                dom.remove(*node);
                removed = true;
                false
            } else {
                true
            }
        });
        removed
    }

    /// Removes every toast now.
    pub fn clear(&mut self, dom: &mut Dom<'_>) {
        for (node, _) in self.live.drain(..) {
            dom.remove(node);
        }
    }

    pub fn len(&self) -> usize {
        self.live.len()
    }

    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }

    /// When the next toast expires, if any (to schedule a poll).
    pub fn next_expiry_ms(&self) -> Option<f64> {
        self.live.iter().map(|(_, e)| *e).reduce(f64::min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_match_the_legacy_markup() {
        assert_eq!(ToastKind::Person.class(), "toast kind-person");
        assert_eq!(ToastKind::Danger.class(), "toast kind-danger");
        assert_eq!(ToastKind::Info.class(), "toast");
    }
}
