//! The two `.route-modal-overlay` dialogs of the template: the route planner (first, legacy
//! `routeModal.ts`) and the cloud mission list (second, legacy `cloudModal.ts`). DOM writes
//! follow `ui/DOM.md` §7 and §8.

use crate::bridge::CloudMission;
use crate::labels;
use crate::planner::{SPANS, Waypoint};
use skylens_viewer::dom::{Dom, escape};
use skylens_viewer::NodeId;

pub const ROUTE_HINT: &str = "지도를 클릭해 웨이포인트를 추가하세요 · 마커를 클릭하면 삭제됩니다";
pub const ROUTE_HINT_TOO_SHORT: &str = "경로에는 최소 2개의 웨이포인트가 필요합니다";
pub const CLOUD_TITLE_ON: &str = "DJI Fly 계정 클라우드에 올려 원격 RC 2가 받게 한다 (Bridge에 클라우드 설정 필요)";
pub const CLOUD_TITLE_OFF: &str = "Bridge에 DJI 클라우드 자격이 없어 비활성화됨 (SKYLENS_DJI_MC_TOKEN + SKYLENS_DJI_WK_KEY)";
pub const CLOUD_TEXT_ON: &str = "클라우드 전송(원격 RC)";
pub const CLOUD_TEXT_OFF: &str = "클라우드 전송 (설정 필요)";

/// The `<li>` list of the planner (legacy `renderList`, capture form).
pub fn waypoint_list_markup(waypoints: &[Waypoint]) -> String {
    waypoints
        .iter()
        .enumerate()
        .map(|(i, w)| {
            format!(
                "<li class=\"route-modal__item\"><span>{}</span><button type=\"button\" class=\"route-modal__remove\">삭제</button></li>",
                escape(&labels::waypoint_label(i, w.lat, w.lon, w.alt))
            )
        })
        .collect()
}

/// Nodes of the route planner dialog.
pub struct RouteModal {
    pub overlay: Option<NodeId>,
    pub close: Option<NodeId>,
    hint: Option<NodeId>,
    pub spans: Vec<NodeId>,
    alt_text: Option<NodeId>,
    pub range: Option<NodeId>,
    pub loop_input: Option<NodeId>,
    pub formation_input: Option<NodeId>,
    pub spacing_input: Option<NodeId>,
    cloud_label: Option<NodeId>,
    pub cloud_input: Option<NodeId>,
    cloud_text: Option<NodeId>,
    pub canvas: Option<NodeId>,
    pub list: Option<NodeId>,
    pub clear: Option<NodeId>,
    pub assign: Option<NodeId>,
    /// Each checkbox and its label, for [`RouteModal::sync_checks`].
    checks: Vec<(NodeId, NodeId)>,
}

fn nth(v: &[NodeId], i: usize) -> Option<NodeId> {
    v.get(i).copied()
}

/// Sets a checkbox's checked state like the `checked` property (no attribute once the control
/// exists; before layout builds it, the attribute is how Blitz takes the initial state).
pub fn set_checked(dom: &mut Dom<'_>, node: NodeId, on: bool) {
    let done = dom
        .doc()
        .get_node_mut(node)
        .and_then(|n| n.element_data_mut())
        .is_some_and(|el| {
            if el.checkbox_input_checked().is_some() {
                el.set_checkbox_input_checked(on);
                true
            } else {
                false
            }
        });
    if done {
        dom.mark_changed();
    } else if on {
        dom.set_attr(node, "checked", "true");
    } else {
        dom.remove_attr(node, "checked");
    }
}

impl RouteModal {
    pub fn mount(dom: &mut Dom<'_>, overlay: Option<NodeId>) -> RouteModal {
        let find = |dom: &Dom<'_>, sel: &str| overlay.and_then(|o| dom.find_in(o, sel));
        let spans = find(dom, ".route-modal__spans").map(|s| dom.children(s)).unwrap_or_default();
        let alt = find(dom, ".route-modal__alt").map(|a| dom.children(a)).unwrap_or_default();
        let toolbar = find(dom, ".route-modal__toolbar").map(|t| dom.children(t)).unwrap_or_default();
        // Toolbar children: spans, alt label, loop label, formation label, cloud label.
        let labels_of = |dom: &Dom<'_>, i: usize| nth(&toolbar, i).map(|l| dom.children(l)).unwrap_or_default();
        let loop_children = labels_of(dom, 2);
        let form_children = labels_of(dom, 3);
        let cloud_children = labels_of(dom, 4);
        let m = RouteModal {
            overlay,
            close: find(dom, ".route-modal__close"),
            hint: find(dom, ".route-modal__hint"),
            spans,
            alt_text: nth(&alt, 0),
            range: nth(&alt, 1),
            loop_input: nth(&loop_children, 0),
            formation_input: nth(&form_children, 0),
            spacing_input: nth(&form_children, 2),
            cloud_label: nth(&toolbar, 4),
            cloud_input: nth(&cloud_children, 0),
            cloud_text: nth(&cloud_children, 1),
            canvas: find(dom, ".route-modal__canvas"),
            list: find(dom, ".route-modal__list"),
            clear: find(dom, ".route-modal__btn--ghost"),
            assign: find(dom, ".route-modal__btn--primary"),
            checks: [(nth(&loop_children, 0), nth(&toolbar, 2)), (nth(&form_children, 0), nth(&toolbar, 3)), (nth(&cloud_children, 0), nth(&toolbar, 4))]
                .into_iter()
                .filter_map(|(i, l)| Some((i?, l?)))
                .collect(),
        };
        // Each checkbox gets the box that draws it (see `.route-modal__check` in panels.css):
        // Blitz paints native checkboxes white whatever the colour scheme. Appended last so the
        // child indices above stay those of the template; CSS moves it first.
        for &(_, label) in &m.checks {
            dom.append_html(label, "<span class=\"route-modal__check\"></span>");
        }
        // Initial properties (legacy sets them as properties; the attributes are how Blitz
        // takes them).
        if let Some(i) = m.loop_input {
            set_checked(dom, i, true);
        }
        if let Some(i) = m.spacing_input {
            dom.set_attr(i, "value", "10");
            // Blitz builds a text editor for type=number but leaves it out of the intrinsic
            // sizing it gives text inputs, so the box collapses and clips the value. Give it the
            // browser's single-line height (the width comes from the legacy CSS).
            dom.set_style(i, "height", "22px");
            dom.set_style(i, "display", "inline-block");
        }
        m
    }

    pub fn sync_spans(&self, dom: &mut Dom<'_>, span_m: f64) {
        for (i, n) in self.spans.iter().enumerate() {
            dom.toggle_class(*n, "is-active", SPANS.get(i).is_some_and(|s| f64::from(*s) == span_m));
        }
    }

    pub fn span_of(&self, node: NodeId) -> Option<f64> {
        self.spans.iter().position(|n| *n == node).and_then(|i| SPANS.get(i)).map(|s| f64::from(*s))
    }

    pub fn sync_alt_text(&self, dom: &mut Dom<'_>, agl: f64) {
        if let Some(t) = self.alt_text {
            dom.set_text(t, &format!("고도 {}m (지면 기준)", agl as i64));
        }
    }

    pub fn render_list(&self, dom: &mut Dom<'_>, waypoints: &[Waypoint]) {
        if let Some(l) = self.list {
            dom.set_html(l, &waypoint_list_markup(waypoints));
        }
    }

    /// Index of a list row's remove button.
    pub fn remove_index(&self, dom: &Dom<'_>, button: NodeId) -> Option<usize> {
        let list = self.list?;
        let li = dom.doc_ref().get_node(button)?.parent?;
        dom.children(list).iter().position(|n| *n == li)
    }

    pub fn set_spacing_enabled(&self, dom: &mut Dom<'_>, enabled: bool) {
        if let Some(i) = self.spacing_input {
            if enabled {
                dom.remove_attr(i, "disabled");
            } else {
                dom.set_attr(i, "disabled", "");
            }
        }
    }

    /// The cloud toggle offered or not (legacy `setCloudAvailable`). Unchecks when unavailable.
    pub fn set_cloud_available(&self, dom: &mut Dom<'_>, available: bool) {
        if let Some(i) = self.cloud_input {
            if available {
                dom.set_attr(i, "title", CLOUD_TITLE_ON);
                dom.remove_attr(i, "disabled");
            } else {
                set_checked(dom, i, false);
                dom.set_attr(i, "title", CLOUD_TITLE_OFF);
                dom.set_attr(i, "disabled", "");
            }
        }
        if let Some(l) = self.cloud_label {
            dom.toggle_class(l, "is-disabled", !available);
        }
        if let Some(t) = self.cloud_text {
            dom.set_text(t, if available { CLOUD_TEXT_ON } else { CLOUD_TEXT_OFF });
        }
    }

    pub fn open(&self, dom: &mut Dom<'_>) {
        if let Some(h) = self.hint {
            dom.set_text(h, ROUTE_HINT);
            dom.remove_class(h, "is-warn");
        }
        if let Some(o) = self.overlay {
            dom.remove_class(o, "is-hidden");
        }
    }

    pub fn close(&self, dom: &mut Dom<'_>) {
        if let Some(o) = self.overlay {
            dom.add_class(o, "is-hidden");
        }
    }

    pub fn warn_too_short(&self, dom: &mut Dom<'_>) {
        if let Some(h) = self.hint {
            dom.set_text(h, ROUTE_HINT_TOO_SHORT);
            dom.add_class(h, "is-warn");
        }
    }

    pub fn set_panning(&self, dom: &mut Dom<'_>, on: bool) {
        if let Some(c) = self.canvas {
            dom.toggle_class(c, "is-panning", on);
        }
    }

    pub fn is_open(&self, dom: &Dom<'_>) -> bool {
        self.overlay.is_some_and(|o| !dom.has_class(o, "is-hidden"))
    }

    /// Mirrors each checkbox's state onto its label (`is-checked`), which `.route-modal__check`
    /// draws from. A `:checked ~` rule did not follow the state in Blitz.
    pub fn sync_checks(&self, dom: &mut Dom<'_>) {
        for &(input, label) in &self.checks {
            let on = dom
                .doc_ref()
                .get_node(input)
                .and_then(|n| n.element_data())
                .and_then(|el| el.checkbox_input_checked())
                .unwrap_or(false);
            if dom.has_class(label, "is-checked") != on {
                dom.toggle_class(label, "is-checked", on);
            }
        }
    }
}

/// One-line message row of the cloud list (legacy `renderEmpty`).
pub fn cloud_message_markup(message: &str) -> String {
    format!("<div class=\"route-modal__item\">{}</div>", escape(message))
}

/// Mission rows (legacy `renderMissions`).
pub fn cloud_rows_markup(missions: &[CloudMission]) -> String {
    missions
        .iter()
        .map(|m| {
            format!(
                "<div class=\"route-modal__item\"><span>{}</span><button type=\"button\" class=\"route-modal__remove\">삭제</button></div>",
                escape(&m.label())
            )
        })
        .collect()
}

pub const CLOUD_LOADING: &str = "불러오는 중\u{2026}";
pub const CLOUD_EMPTY: &str = "클라우드에 미션이 없습니다.";
pub const CLOUD_FAILED: &str = "불러오지 못했습니다.";

/// The cloud mission dialog.
pub struct CloudModal {
    pub overlay: Option<NodeId>,
    pub close: Option<NodeId>,
    pub list: Option<NodeId>,
    pub refresh: Option<NodeId>,
    pub busy: bool,
    pub missions: Vec<CloudMission>,
}

impl CloudModal {
    pub fn mount(dom: &mut Dom<'_>, overlay: Option<NodeId>) -> CloudModal {
        let find = |sel: &str| overlay.and_then(|o| dom.find_in(o, sel));
        CloudModal { close: find(".route-modal__close"), list: find(".route-modal__list"), refresh: find(".route-modal__btn"), overlay, busy: false, missions: Vec::new() }
    }

    pub fn open(&self, dom: &mut Dom<'_>) {
        if let Some(o) = self.overlay {
            dom.remove_class(o, "is-hidden");
        }
    }

    pub fn close(&self, dom: &mut Dom<'_>) {
        if let Some(o) = self.overlay {
            dom.add_class(o, "is-hidden");
        }
    }

    pub fn show_message(&mut self, dom: &mut Dom<'_>, message: &str) {
        self.missions.clear();
        if let Some(l) = self.list {
            dom.replace_children(l, &cloud_message_markup(message));
        }
    }

    pub fn show_missions(&mut self, dom: &mut Dom<'_>, missions: Vec<CloudMission>) {
        if missions.is_empty() {
            self.show_message(dom, CLOUD_EMPTY);
            return;
        }
        if let Some(l) = self.list {
            dom.replace_children(l, &cloud_rows_markup(&missions));
        }
        self.missions = missions;
    }

    pub fn set_refresh_enabled(&self, dom: &mut Dom<'_>, enabled: bool) {
        if let Some(b) = self.refresh {
            if enabled {
                dom.remove_attr(b, "disabled");
            } else {
                dom.set_attr(b, "disabled", "");
            }
        }
    }

    /// Which mission a row's delete button belongs to.
    pub fn mission_of(&self, dom: &Dom<'_>, button: NodeId) -> Option<(usize, CloudMission)> {
        let list = self.list?;
        let row = dom.doc_ref().get_node(button)?.parent?;
        let i = dom.children(list).iter().position(|n| *n == row)?;
        self.missions.get(i).cloned().map(|m| (i, m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_rows_match_the_capture() {
        let w = [Waypoint { lat: 36.36796, lon: 127.34649, alt: 120.0 }];
        assert_eq!(
            waypoint_list_markup(&w),
            "<li class=\"route-modal__item\"><span>#1  36.36796, 127.34649 · 120m</span><button type=\"button\" class=\"route-modal__remove\">삭제</button></li>"
        );
        assert_eq!(waypoint_list_markup(&[]), "");
    }

    #[test]
    fn cloud_rows() {
        assert_eq!(cloud_message_markup(CLOUD_FAILED), "<div class=\"route-modal__item\">불러오지 못했습니다.</div>");
        let m = CloudMission { uuid: "u".into(), name: "a<b".into(), waypoint_count: Some(3), distance_m: None };
        assert_eq!(
            cloud_rows_markup(&[m]),
            "<div class=\"route-modal__item\"><span>a&lt;b · 3 WP</span><button type=\"button\" class=\"route-modal__remove\">삭제</button></div>"
        );
    }
}
