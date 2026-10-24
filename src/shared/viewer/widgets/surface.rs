//! The one way a floating surface is built (legacy `surface.ts`): a `.sl-surface` with a variant
//! class, an optional leading `.sl-dot` and a title. Position and size stay with the caller.
//!
//! ```html
//! <div class="sl-surface sl-surface--chip pane-label pane-label--status">
//!   <span class="sl-dot"></span><span class="sl-surface__label">현황판 · 실시간 3D 복원</span>
//! </div>
//! ```

use crate::dom::{Dom, Sel};
use blitz_dom::NodeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceVariant {
    Chip,
    Panel,
    Frame,
    Modal,
}

impl SurfaceVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            SurfaceVariant::Chip => "chip",
            SurfaceVariant::Panel => "panel",
            SurfaceVariant::Frame => "frame",
            SurfaceVariant::Modal => "modal",
        }
    }
}

/// The nodes of a built surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Surface {
    pub root: NodeId,
    pub dot: Option<NodeId>,
    pub title: Option<NodeId>,
}

/// The `class` attribute of a surface root.
pub fn surface_class(variant: SurfaceVariant, extra: &str) -> String {
    let mut class = format!("sl-surface sl-surface--{}", variant.as_str());
    for c in extra.split_whitespace() {
        class.push(' ');
        class.push_str(c);
    }
    class
}

/// Builds a surface as the last child of `mount`.
pub fn create_surface<'s>(
    dom: &mut Dom<'_>,
    mount: impl Into<Sel<'s>>,
    variant: SurfaceVariant,
    class_name: &str,
    id: Option<&str>,
    dot: bool,
    title: Option<&str>,
) -> Option<Surface> {
    let class = surface_class(variant, class_name);
    let mut attrs = vec![("class", class.as_str())];
    if let Some(id) = id {
        attrs.push(("id", id));
    }
    let root = dom.create(mount, "div", &attrs, None)?;
    let dot = if dot { dom.create(root, "span", &[("class", "sl-dot")], None) } else { None };
    let title = match title {
        // A chip is its title; a panel gets a heading its contents sit under.
        Some(t) => {
            let class = if variant == SurfaceVariant::Chip { "sl-surface__label" } else { "sl-surface__title" };
            dom.create(root, "span", &[("class", class)], Some(t))
        }
        None => None,
    };
    Some(Surface { root, dot, title })
}

/// The screen's own name, top-left over the view (legacy `mountPaneLabel`). `tone` is
/// `"control"` or `"status"`.
pub fn mount_pane_label<'s>(
    dom: &mut Dom<'_>,
    pane: impl Into<Sel<'s>>,
    text: &str,
    tone: Option<&str>,
) -> Option<Surface> {
    let class = match tone {
        Some(t) => format!("pane-label pane-label--{t}"),
        None => "pane-label".to_string(),
    };
    create_surface(dom, pane, SurfaceVariant::Chip, &class, None, true, Some(text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_match_the_legacy_markup() {
        assert_eq!(
            surface_class(SurfaceVariant::Chip, "pane-label pane-label--status"),
            "sl-surface sl-surface--chip pane-label pane-label--status"
        );
        assert_eq!(surface_class(SurfaceVariant::Panel, ""), "sl-surface sl-surface--panel");
    }
}
