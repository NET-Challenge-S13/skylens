//! The tower's DOM panels (legacy `ui/missionPanel.ts`, `ui/telemetryPanel.ts`,
//! `ui/settingsPanel.ts`) on the template nodes. Every write follows `ui/DOM.md`.

use crate::fleet::FleetDrone;
use crate::labels::{self, Station};
use crate::settings::DisplayMode;
use skylens_shared::proto::viewer::v1 as viewer;
use skylens_viewer::dom::{Dom, escape};
use skylens_viewer::link::LinkState;
use skylens_viewer::{NodeId, SiteFrame, q};

// ---------------------------------------------------------------------------------------------
// Mission panel
// ---------------------------------------------------------------------------------------------

/// Stepper order (legacy `PHASES`).
pub const PHASES: [viewer::MissionPhase; 4] =
    [viewer::MissionPhase::Idle, viewer::MissionPhase::Assigned, viewer::MissionPhase::AwaitingDrone, viewer::MissionPhase::Active];

/// Fallback message per phase (legacy `FALLBACK_MESSAGE`).
pub fn fallback_message(phase: viewer::MissionPhase) -> &'static str {
    match phase {
        viewer::MissionPhase::Assigned => "태스크 지정 완료",
        viewer::MissionPhase::AwaitingDrone => "드론이 현장에 도착해 연결되면 작업이 시작됩니다",
        viewer::MissionPhase::Active => "드론 연결됨 \u{2014} 임무 수행 중",
        _ => "대기 중 \u{2014} 관제탑에서 경로를 지정하십시오",
    }
}

/// Class of step `i` when phase index `k` is current.
pub fn step_class(i: usize, k: Option<usize>) -> &'static str {
    match k {
        Some(k) if i < k => "mission-panel__step is-done",
        Some(k) if i == k => "mission-panel__step is-active",
        _ => "mission-panel__step",
    }
}

/// `.mission-panel__sub` (legacy `renderSub`).
pub fn sub_text(phase: viewer::MissionPhase, eta_deadline_ms: Option<f64>, now_ms: f64, drones_online: u32) -> String {
    let mut parts = Vec::new();
    if phase == viewer::MissionPhase::AwaitingDrone
        && let Some(deadline) = eta_deadline_ms
    {
        let left = labels::js_round((deadline - now_ms) / 1000.0).max(0.0);
        parts.push(format!("예상 {}초", left as i64));
    }
    parts.push(format!("드론 {drones_online}대 연결됨"));
    parts.join(" · ")
}

pub struct MissionPanel {
    root: Option<NodeId>,
    link: Option<NodeId>,
    link_text: Option<NodeId>,
    steps: Vec<NodeId>,
    message: Option<NodeId>,
    sub: Option<NodeId>,
    phase: viewer::MissionPhase,
    drones_online: u32,
    eta_deadline: Option<f64>,
    next_eta_ms: f64,
    connected: bool,
}

impl MissionPanel {
    /// Finds the template nodes and puts them in the freshly built state (before the core link).
    pub fn mount(dom: &mut Dom<'_>) -> MissionPanel {
        let root = dom.find(q(".mission-panel"));
        let steps = dom.children(q(".mission-panel__steps"));
        let p = MissionPanel {
            root,
            link: dom.find(q(".mission-panel__link")),
            link_text: dom.find(q(".mission-panel__link-text")),
            steps,
            message: dom.find(q(".mission-panel__message")),
            sub: dom.find(q(".mission-panel__sub")),
            phase: viewer::MissionPhase::Idle,
            drones_online: 0,
            eta_deadline: None,
            next_eta_ms: 0.0,
            connected: false,
        };
        if let Some(r) = p.root {
            dom.set_class(r, "sl-surface sl-surface--panel mission-panel");
        }
        if let Some(l) = p.link {
            dom.set_class(l, "mission-panel__link is-disconnected");
        }
        if let Some(t) = p.link_text {
            dom.set_text(t, "코어 연결 대기 중");
        }
        for s in &p.steps {
            dom.set_class(*s, "mission-panel__step");
        }
        if let Some(m) = p.message {
            dom.set_text(m, fallback_message(viewer::MissionPhase::Idle));
        }
        if let Some(s) = p.sub {
            dom.set_text(s, "");
        }
        p
    }

    fn render_sub(&self, dom: &mut Dom<'_>, now_ms: f64) {
        if let Some(s) = self.sub {
            dom.set_text(s, &sub_text(self.phase, self.eta_deadline, now_ms, self.drones_online));
        }
    }

    /// A `mission_status` from the core (legacy `setMission`).
    pub fn set_mission(&mut self, dom: &mut Dom<'_>, m: &viewer::MissionStatus, now_ms: f64) {
        let phase = m.phase();
        self.phase = phase;
        self.drones_online = m.drones_online;
        if let Some(el) = self.message {
            let text = if m.message.is_empty() { fallback_message(phase) } else { m.message.as_str() };
            dom.set_text(el, text);
        }
        let k = PHASES.iter().position(|p| *p == phase);
        for (i, s) in self.steps.iter().enumerate() {
            dom.set_class(*s, step_class(i, k));
        }
        self.eta_deadline = match (phase, m.eta_seconds) {
            (viewer::MissionPhase::AwaitingDrone, Some(eta)) => Some(now_ms + eta * 1000.0),
            _ => None,
        };
        self.next_eta_ms = now_ms + 500.0;
        self.render_sub(dom, now_ms);
    }

    /// Link state change (legacy `setLink`).
    pub fn set_link(&mut self, dom: &mut Dom<'_>, state: LinkState, detail: &str, now_ms: f64) {
        if let Some(l) = self.link {
            dom.toggle_class(l, "is-connected", state == LinkState::Connected);
            dom.toggle_class(l, "is-connecting", state == LinkState::Connecting);
            dom.toggle_class(l, "is-disconnected", state == LinkState::Disconnected);
        }
        if let Some(t) = self.link_text {
            dom.set_text(t, detail);
        }
        if let Some(r) = self.root {
            dom.toggle_class(r, "is-stale", state != LinkState::Connected);
        }
        self.connected = state == LinkState::Connected;
        if state == LinkState::Connected {
            self.render_sub(dom, now_ms);
            return;
        }
        // The countdown stops (`tick` only runs while connected); the deadline is kept, as the
        // legacy timer kept it.
        let (message, sub) = if state == LinkState::Connecting {
            ("코어에 연결하는 중입니다", "연결되면 임무 상태와 드론 위치가 표시됩니다")
        } else {
            (
                "코어와 연결되어 있지 않습니다 \u{2014} 표시된 임무 상태는 최신이 아닐 수 있습니다",
                "연결이 복구될 때까지 드론 위치와 임무 상태를 갱신할 수 없습니다",
            )
        };
        if let Some(m) = self.message {
            dom.set_text(m, message);
        }
        if let Some(s) = self.sub {
            dom.set_text(s, sub);
        }
    }

    /// The 500 ms countdown redraw while awaiting the drone.
    pub fn tick(&mut self, dom: &mut Dom<'_>, now_ms: f64) {
        if self.connected && self.eta_deadline.is_some() && now_ms >= self.next_eta_ms {
            self.next_eta_ms = now_ms + 500.0;
            self.render_sub(dom, now_ms);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Telemetry panel
// ---------------------------------------------------------------------------------------------

/// A fleet row's markup (legacy `ensureRow`, capture form).
pub fn row_markup(id: u32, station: Station) -> String {
    let [r, g, b] = {
        let t = crate::tower::drone_tint(id);
        [(t >> 16) & 0xff, (t >> 8) & 0xff, t & 0xff]
    };
    format!(
        "<li class=\"telemetry-row\" tabindex=\"0\"><div class=\"telemetry-row__head\"><span class=\"telemetry-row__dot\" style=\"background: rgb({r}, {g}, {b});\"></span><span class=\"telemetry-row__name\">{}</span><span class=\"telemetry-row__stale\">수신 끊김</span></div><div class=\"telemetry-row__vals\"></div></li>",
        escape(station.label())
    )
}

struct Row {
    id: u32,
    li: NodeId,
    name: Option<NodeId>,
    vals: Option<NodeId>,
}

pub struct TelemetryPanel {
    empty: Option<NodeId>,
    list: Option<NodeId>,
    rows: Vec<Row>,
}

impl TelemetryPanel {
    pub fn mount(dom: &mut Dom<'_>) -> TelemetryPanel {
        TelemetryPanel { empty: dom.find(q(".telemetry-panel__empty")), list: dom.find(q(".telemetry-panel__list")), rows: Vec::new() }
    }

    /// The drone a row element belongs to.
    pub fn row_drone(&self, node: NodeId) -> Option<u32> {
        self.rows.iter().find(|r| r.li == node).map(|r| r.id)
    }

    /// Re-renders from the fleet snapshot (legacy `render`).
    pub fn render(&mut self, dom: &mut Dom<'_>, drones: &[FleetDrone], active: Option<u32>) {
        if let Some(e) = self.empty {
            dom.toggle_class(e, "is-hidden", !drones.is_empty());
        }
        let Some(list) = self.list else { return };
        self.rows.retain(|r| {
            let keep = drones.iter().any(|d| d.id == r.id);
            if !keep {
                dom.remove(r.li);
            }
            keep
        });
        for d in drones {
            if !self.rows.iter().any(|r| r.id == d.id) {
                let added = dom.append_html(list, &row_markup(d.id, d.station));
                if let Some(li) = added.first().copied() {
                    let name = dom.find_in(li, ".telemetry-row__name");
                    let vals = dom.find_in(li, ".telemetry-row__vals");
                    self.rows.push(Row { id: d.id, li, name, vals });
                }
            }
            let Some(row) = self.rows.iter().find(|r| r.id == d.id) else { continue };
            let (li, name, vals) = (row.li, row.name, row.vals);
            if let Some(n) = name {
                dom.set_text(n, d.station.label());
            }
            dom.toggle_class(li, "is-active", active == Some(d.id));
            dom.toggle_class(li, "is-stale", d.stale);
            if let Some(v) = vals {
                dom.set_text(v, &labels::telemetry_vals(d.lat, d.lon, d.alt, d.heading_deg, d.speed, d.battery_pct));
            }
        }
        // Row order follows the fleet order; touched only when it differs.
        let desired: Vec<NodeId> = drones.iter().filter_map(|d| self.rows.iter().find(|r| r.id == d.id).map(|r| r.li)).collect();
        let current = dom.children(list);
        if desired != current {
            let doc = dom.doc();
            let mut m = doc.mutate();
            for li in &desired {
                m.remove_node(*li);
            }
            m.append_children(list, &desired);
            drop(m);
            dom.mark_changed();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Settings panel
// ---------------------------------------------------------------------------------------------

pub const AERIAL_UNAVAILABLE_HINT: &str = "위성 영상을 불러올 수 없어 사용할 수 없습니다 (VWorld 키 확인)";
pub const OVERRIDE_NOTE: &str = "URL(?display=)로 지정된 표시 방식입니다 \u{2014} 저장된 설정은 변경되지 않습니다";

/// 건물 출처 line (legacy `SOURCE_LABEL` + count).
pub fn building_source_text(site: &SiteFrame) -> String {
    if site.buildings && site.building_count > 0 {
        format!("VWorld 실측 건물 데이터 · {}동", labels::grouped(u64::from(site.building_count)))
    } else if site.buildings {
        "VWorld 실측 건물 데이터".to_string()
    } else {
        "건물 데이터 없음".to_string()
    }
}

pub struct SettingsPanel {
    pub overlay: Option<NodeId>,
    buttons: Vec<(DisplayMode, NodeId)>,
    aerial_hint: Option<NodeId>,
    meta: Vec<NodeId>,
    aerial_available: bool,
}

impl SettingsPanel {
    pub fn mount(dom: &mut Dom<'_>, overridden: bool) -> SettingsPanel {
        let overlay = dom.find(q(".settings-overlay"));
        let mut buttons = Vec::new();
        for n in dom.children(q(".settings-panel__options")) {
            if let Some(m) = dom.attr(n, "data-mode").and_then(|v| DisplayMode::parse(&v)) {
                buttons.push((m, n));
            }
        }
        let aerial_hint = buttons.iter().find(|(m, _)| *m == DisplayMode::Aerial).and_then(|(_, n)| dom.find_in(*n, ".settings-option__hint"));
        let meta: Vec<NodeId> = dom.children(q(".settings-panel__meta")).into_iter().filter(|n| dom.doc_ref().get_node(*n).and_then(|x| x.element_data()).is_some_and(|e| &*e.name.local == "dd")).collect();
        if overridden && let Some(note) = dom.find(q(".settings-panel__note")) {
            dom.set_text(note, OVERRIDE_NOTE);
        }
        SettingsPanel { overlay, buttons, aerial_hint, meta, aerial_available: true }
    }

    pub fn mode_of(&self, node: NodeId) -> Option<DisplayMode> {
        self.buttons.iter().find(|(_, n)| *n == node).map(|(m, _)| *m)
    }

    pub fn aerial_available(&self) -> bool {
        self.aerial_available
    }

    /// The selected option (legacy settings subscription).
    pub fn show_mode(&self, dom: &mut Dom<'_>, mode: DisplayMode) {
        for (m, n) in &self.buttons {
            let on = *m == mode;
            dom.toggle_class(*n, "is-active", on);
            dom.set_attr(*n, "aria-checked", if on { "true" } else { "false" });
        }
    }

    /// Values that come from the loaded scene (legacy: fixed when the panel was built; here they
    /// follow the core's `Site`).
    pub fn show_site(&mut self, dom: &mut Dom<'_>, site: &SiteFrame) {
        self.aerial_available = site.imagery;
        if let Some((_, aerial)) = self.buttons.iter().find(|(m, _)| *m == DisplayMode::Aerial) {
            if site.imagery {
                dom.remove_attr(*aerial, "disabled");
            } else {
                dom.set_attr(*aerial, "disabled", "");
            }
        }
        if let Some(h) = self.aerial_hint {
            dom.set_text(h, if site.imagery { "위성/항공 영상을 그대로 적용" } else { AERIAL_UNAVAILABLE_HINT });
        }
        let values = [
            labels::area_label(&site.name, site.min_lat, site.min_lon),
            building_source_text(site),
            if site.imagery { "사용 가능".to_string() } else { "사용 불가".to_string() },
        ];
        for (dd, v) in self.meta.iter().zip(values.iter()) {
            dom.set_text(*dd, v);
        }
    }

    pub fn toggle(&self, dom: &mut Dom<'_>) {
        if let Some(o) = self.overlay {
            let hidden = dom.has_class(o, "is-hidden");
            dom.toggle_class(o, "is-hidden", !hidden);
        }
    }

    pub fn close(&self, dom: &mut Dom<'_>) {
        if let Some(o) = self.overlay {
            dom.add_class(o, "is-hidden");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_and_messages() {
        let k = PHASES.iter().position(|p| *p == viewer::MissionPhase::AwaitingDrone);
        assert_eq!(step_class(0, k), "mission-panel__step is-done");
        assert_eq!(step_class(2, k), "mission-panel__step is-active");
        assert_eq!(step_class(3, k), "mission-panel__step");
        assert_eq!(fallback_message(viewer::MissionPhase::Unspecified), "대기 중 \u{2014} 관제탑에서 경로를 지정하십시오");
    }

    #[test]
    fn sub_line_counts_down() {
        assert_eq!(sub_text(viewer::MissionPhase::AwaitingDrone, Some(8_400.0), 0.0, 0), "예상 8초 · 드론 0대 연결됨");
        assert_eq!(sub_text(viewer::MissionPhase::AwaitingDrone, Some(0.0), 5_000.0, 1), "예상 0초 · 드론 1대 연결됨");
        assert_eq!(sub_text(viewer::MissionPhase::Active, Some(1.0), 0.0, 3), "드론 3대 연결됨");
    }

    #[test]
    fn row_markup_matches_the_capture() {
        assert_eq!(
            row_markup(1, Station::Center),
            "<li class=\"telemetry-row\" tabindex=\"0\"><div class=\"telemetry-row__head\"><span class=\"telemetry-row__dot\" style=\"background: rgb(159, 232, 255);\"></span><span class=\"telemetry-row__name\">중앙 드론</span><span class=\"telemetry-row__stale\">수신 끊김</span></div><div class=\"telemetry-row__vals\"></div></li>"
        );
    }

    #[test]
    fn building_source_wording() {
        let mut s = SiteFrame {
            anchor: skylens_shared::geo::Gps { lat: 0.0, lon: 0.0, alt: 0.0 },
            name: "daejeon".into(),
            min_lat: 0.0,
            min_lon: 0.0,
            max_lat: 1.0,
            max_lon: 1.0,
            imagery: true,
            buildings: true,
            building_count: 6194,
        };
        assert_eq!(building_source_text(&s), "VWorld 실측 건물 데이터 · 6,194동");
        s.buildings = false;
        assert_eq!(building_source_text(&s), "건물 데이터 없음");
    }
}
