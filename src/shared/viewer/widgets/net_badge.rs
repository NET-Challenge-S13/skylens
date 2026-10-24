//! Corner badge for the link to the core (legacy `relayBadge.ts` / `statusUi.ts`), mounted into
//! `#net-status`:
//!
//! ```html
//! <div id="net-status" class="net-badge">
//!   <span class="net-badge__dot" style="background: #39d98a"></span><span>코어 스트림 수신 중</span>
//! </div>
//! ```

use crate::dom::{Dom, Sel};
use crate::link::LinkStatus;
use blitz_dom::NodeId;

pub const NET_STATUS_ID: &str = "net-status";

#[derive(Debug, Clone)]
pub struct NetBadge {
    dot: NodeId,
    text: NodeId,
    shown: Option<(String, String)>,
}

impl NetBadge {
    /// Turns the host (normally `#net-status`) into the badge.
    pub fn mount<'s>(dom: &mut Dom<'_>, host: impl Into<Sel<'s>>) -> Option<NetBadge> {
        let host = dom.find(host)?;
        dom.add_class(host, "net-badge");
        dom.replace_children(host, "");
        let dot = dom.create(host, "span", &[("class", "net-badge__dot")], None)?;
        let text = dom.create(host, "span", &[], None)?;
        Some(NetBadge { dot, text, shown: None })
    }

    /// Sets the dot colour (CSS) and the text.
    pub fn set(&mut self, dom: &mut Dom<'_>, dot_css: &str, text: &str) {
        let next = (dot_css.to_string(), text.to_string());
        if self.shown.as_ref() == Some(&next) {
            return;
        }
        dom.set_style(self.dot, "background", dot_css);
        dom.set_text(self.text, text);
        self.shown = Some(next);
    }

    /// Shows the core link the way the board's badge did.
    pub fn show_link(&mut self, dom: &mut Dom<'_>, status: &LinkStatus) {
        self.set(dom, status.severity().dot_css(), status.badge_text());
    }
}
