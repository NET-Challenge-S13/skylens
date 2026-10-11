//! The Blitz document of a screen and everything it does: parse the screen's
//! HTML/CSS, mount the 3D widget, feed viewer events into the [`Store`], write
//! the panels back into the DOM, and turn clicks and keys into control messages.
//!
//! Web and native share this file. Only `native.rs` / `web.rs` differ, and only in
//! how they open the window and load fonts.

use crate::chunks::SceneOp;
use crate::display::{DisplayMode, Layer};
use crate::link::{Link, LinkEvent};
use crate::protocol::{self, Outbox};
use crate::state::{Changed, LinkState, Store};
use crate::view3d::{
    shared_view, CameraRequest, MarkerKind, Overlay, OverlayDetection, OverlayDrone, SharedView, View3d,
};
use crate::views::{self, Toast, ToastKind};
use blitz_dom::{
    BaseDocument, DocGuard, DocGuardMut, Document, DocumentConfig, EventDriver, EventHandler, FontContext,
    LocalName, Namespace, NodeId, QualName,
};
use blitz_html::{HtmlDocument, HtmlProvider};
use blitz_traits::events::{DomEvent, DomEventData, EventState, UiEvent};
use keyboard_types::Code;
use skylens_shared::geo::{gps_to_enu, GeoAnchor, Gps};
use skylens_shared::proto::model::v1 as model;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::task::Context;

/// Default ENU origin: the model server's default anchor (SKYLENS_ANCHOR).
pub const DEFAULT_ANCHOR: GeoAnchor = Gps {
    lat: 36.3685,
    lon: 127.3475,
    alt: 30.0,
};

/// Native default; the web build derives the URL from the page's host instead.
pub const DEFAULT_CORE_URL: &str = "ws://127.0.0.1:8080/viewer";

/// Manual control is streamed at this period while a stick key is held (10 Hz).
const MANUAL_PERIOD_MS: f64 = 100.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    /// 관제탑: route planning, mission status, drones, display options, manual keys.
    Control,
    /// 현황판: arrival-based reconstruction, server ladder, detections, camera.
    Status,
}

impl Screen {
    pub fn parse(s: &str) -> Option<Screen> {
        match s.trim() {
            "control" => Some(Screen::Control),
            "status" => Some(Screen::Status),
            _ => None,
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Screen::Control => "SkyLens · 관제탑",
            Screen::Status => "SkyLens · 현황판",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub screen: Screen,
    pub core_url: String,
    pub anchor: GeoAnchor,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            screen: Screen::Status,
            core_url: DEFAULT_CORE_URL.to_string(),
            anchor: DEFAULT_ANCHOR,
        }
    }
}

pub const USAGE: &str = "usage: skylens-client [--screen control|status] [--core ws://HOST:8080/viewer] [--anchor LAT,LON[,ALT]]";

impl Config {
    /// Parses `--screen`, `--core` and `--anchor` (also `--key=value`).
    pub fn from_args<I: IntoIterator<Item = String>>(args: I) -> Result<Config, String> {
        let mut cfg = Config::default();
        let mut it = args.into_iter();
        while let Some(arg) = it.next() {
            let (key, inline) = match arg.split_once('=') {
                Some((k, v)) => (k.to_string(), Some(v.to_string())),
                None => (arg.clone(), None),
            };
            let mut value = |name: &str| -> Result<String, String> {
                inline.clone().or_else(|| it.next()).ok_or_else(|| format!("{name} needs a value\n{USAGE}"))
            };
            match key.as_str() {
                "--screen" => {
                    let v = value("--screen")?;
                    cfg.screen = Screen::parse(&v).ok_or_else(|| format!("unknown screen '{v}'\n{USAGE}"))?;
                }
                "--core" => cfg.core_url = normalize_core_url(&value("--core")?),
                "--anchor" => {
                    let v = value("--anchor")?;
                    cfg.anchor = parse_anchor(&v).ok_or_else(|| format!("bad anchor '{v}'\n{USAGE}"))?;
                }
                "-h" | "--help" => return Err(USAGE.to_string()),
                other => return Err(format!("unknown argument '{other}'\n{USAGE}")),
            }
        }
        Ok(cfg)
    }
}

/// `LAT,LON` or `LAT,LON,ALT`.
pub fn parse_anchor(s: &str) -> Option<GeoAnchor> {
    let parts: Vec<f64> = s.split(',').map(|p| p.trim().parse::<f64>()).collect::<Result<_, _>>().ok()?;
    let (lat, lon, alt) = match parts[..] {
        [lat, lon] => (lat, lon, DEFAULT_ANCHOR.alt),
        [lat, lon, alt] => (lat, lon, alt),
        _ => return None,
    };
    ((-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon) && alt.is_finite())
        .then_some(Gps { lat, lon, alt })
}

/// Accepts `host:port`, `http(s)://…` or `ws(s)://…`; adds `/viewer` when no path is given.
pub fn normalize_core_url(s: &str) -> String {
    let s = s.trim();
    let with_scheme = if let Some(rest) = s.strip_prefix("http://") {
        format!("ws://{rest}")
    } else if let Some(rest) = s.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if s.contains("://") {
        s.to_string()
    } else {
        format!("ws://{s}")
    };
    let after_scheme = with_scheme.split_once("://").map_or("", |(_, r)| r);
    match after_scheme.find('/') {
        None => format!("{with_scheme}/viewer"),
        Some(i) if &after_scheme[i..] == "/" => format!("{with_scheme}viewer"),
        Some(_) => with_scheme,
    }
}

/// The screen's HTML with its stylesheets inlined and the anchor filled in.
pub fn page_html(screen: Screen, anchor: &GeoAnchor) -> String {
    let theme = ("theme.css", include_str!("ui/theme.css"));
    let (html, sheets) = match screen {
        Screen::Control => (
            include_str!("ui/control.html"),
            [theme, ("control.css", include_str!("ui/control.css"))],
        ),
        Screen::Status => (
            include_str!("ui/status.html"),
            [theme, ("status.css", include_str!("ui/status.css"))],
        ),
    };
    let mut page = html.to_string();
    for (name, css) in sheets {
        page = page.replace(
            &format!("<link rel=\"stylesheet\" href=\"{name}\" />"),
            &format!("<style>\n{css}\n</style>"),
        );
    }
    page.replace("{{ANCHOR_LAT}}", &format!("{:.6}", anchor.lat))
        .replace("{{ANCHOR_LON}}", &format!("{:.6}", anchor.lon))
        .replace("{{DEFAULT_ALT}}", &format!("{:.0}", anchor.alt + 60.0))
        .replace(
            "{{ANCHOR_LABEL}}",
            &format!("{:.5}, {:.5} · {:.0}m", anchor.lat, anchor.lon, anchor.alt),
        )
}

/// What a DOM event asked for. Collected during event dispatch, applied afterwards
/// (the dispatcher holds the document).
#[derive(Debug, Clone)]
enum UiAction {
    Click { action: String, arg: Option<String> },
    Key { code: Code, down: bool, in_input: bool },
}

struct Collector<'a> {
    actions: &'a mut Vec<UiAction>,
}

impl EventHandler for Collector<'_> {
    fn handle_event(
        &mut self,
        chain: &[NodeId],
        event: &mut DomEvent,
        doc: &mut dyn Document,
        event_state: &mut EventState,
    ) {
        match &event.data {
            DomEventData::Click(_) => {
                let doc = doc.inner();
                let action_attr = LocalName::from("data-action");
                let arg_attr = LocalName::from("data-arg");
                // The innermost element with a data-action wins; "none" stops the
                // walk so a click inside a modal does not reach its backdrop.
                for id in chain {
                    let Some(node) = doc.get_node(*id) else { continue };
                    if let Some(action) = node.attr(action_attr.clone()) {
                        if action != "none" {
                            self.actions.push(UiAction::Click {
                                action: action.to_string(),
                                arg: node.attr(arg_attr.clone()).map(str::to_string),
                            });
                        }
                        break;
                    }
                }
            }
            DomEventData::KeyDown(k) | DomEventData::KeyUp(k) => {
                let in_input = doc
                    .inner()
                    .get_node(event.target)
                    .and_then(|n| n.element_data())
                    .is_some_and(|e| e.name.local == LocalName::from("input"));
                let down = matches!(event.data, DomEventData::KeyDown(_));
                if down && k.is_auto_repeating && !in_input {
                    return;
                }
                if !in_input && k.code == Code::Tab {
                    event_state.prevent_default();
                }
                self.actions.push(UiAction::Key { code: k.code, down, in_input });
            }
            _ => {}
        }
    }
}

/// Writes into the document by element id, skipping writes that would not change
/// anything (each write restyles and relayouts the subtree).
struct Dom<'a> {
    doc: &'a mut BaseDocument,
    cache: &'a mut HashMap<String, String>,
    changed: bool,
}

impl Dom<'_> {
    fn write(&mut self, key: String, value: &str, f: impl FnOnce(&mut BaseDocument, NodeId)) {
        if self.cache.get(&key).is_some_and(|v| v == value) {
            return;
        }
        let id = key.split(['#', '@']).next().unwrap_or_default();
        let Some(node) = self.doc.get_element_by_id(id) else {
            return;
        };
        f(&mut *self.doc, node);
        self.cache.insert(key, value.to_string());
        self.changed = true;
    }

    fn html(&mut self, id: &str, html: &str) {
        self.write(format!("{id}#html"), html, |doc, node| doc.mutate().set_inner_html(node, html));
    }

    fn text(&mut self, id: &str, text: &str) {
        self.html(id, &views::escape(text));
    }

    fn attr(&mut self, id: &str, name: &str, value: &str) {
        self.write(format!("{id}@{name}"), value, |doc, node| {
            let qual = QualName::new(None, Namespace::from(""), LocalName::from(name));
            doc.mutate().set_attribute(node, qual, value);
        });
    }

    fn class(&mut self, id: &str, class: &str) {
        self.attr(id, "class", class);
    }
}

fn input_value(doc: &BaseDocument, id: &str) -> Option<String> {
    let node = doc.get_node(doc.get_element_by_id(id)?)?;
    let el = node.element_data()?;
    if let Some(t) = el.text_input_data() {
        return Some(t.editor.raw_text().to_string());
    }
    el.attr(LocalName::from("value")).map(str::to_string)
}

fn checkbox_checked(doc: &BaseDocument, id: &str) -> bool {
    doc.get_element_by_id(id)
        .and_then(|n| doc.get_node(n))
        .and_then(|n| n.element_data())
        .and_then(|e| e.checkbox_input_checked())
        .unwrap_or(false)
}

/// Centre of a pointblitz chunk in ENU, for framing the camera on first data.
fn chunk_center(bytes: &[u8]) -> Option<[f64; 3]> {
    let h = pointblitz::io::chunk::decode_header(bytes).ok()?;
    Some([0, 1, 2].map(|i| h.origin[i] + f64::from(h.bbox_min[i] + h.bbox_max[i]) / 2.0))
}

pub struct App {
    cfg: Config,
    store: Store,
    link: Link,
    outbox: Outbox,
    view: SharedView,
    cache: HashMap<String, String>,
    actions: Vec<UiAction>,
    held: HashSet<Code>,
    manual_active: bool,
    last_manual_ms: f64,
    toasts: Vec<Toast>,
    draft: Vec<model::Gps>,
    route_hint: Option<String>,
    route_open: bool,
    settings_open: bool,
    display: DisplayMode,
    focused_detection: Option<String>,
    framed_points: bool,
    framed_drone: bool,
}

impl App {
    fn new(cfg: Config, view: SharedView) -> Self {
        let link = Link::start(cfg.core_url.clone());
        Self {
            cfg,
            store: Store::new(),
            link,
            outbox: Outbox::new(),
            view,
            cache: HashMap::new(),
            actions: Vec::new(),
            held: HashSet::new(),
            manual_active: false,
            last_manual_ms: 0.0,
            toasts: Vec::new(),
            draft: Vec::new(),
            route_hint: None,
            route_open: false,
            settings_open: false,
            display: DisplayMode::default(),
            focused_detection: None,
            framed_points: false,
            framed_drone: false,
        }
    }

    fn enu(&self, g: &model::Gps) -> [f64; 3] {
        let e = gps_to_enu(Gps::from(g.clone()), self.cfg.anchor);
        [e.e, e.n, e.u]
    }

    fn toast(&mut self, text: impl Into<String>, kind: ToastKind) {
        self.toasts.push(Toast {
            text: text.into(),
            kind,
            expires_ms: crate::now_ms() + views::TOAST_LIFETIME_MS,
        });
        if self.toasts.len() > 4 {
            self.toasts.remove(0);
        }
    }

    fn send(&mut self, body: protocol::Body) -> bool {
        let bytes = self.outbox.encode(body, crate::now_ms() as i64);
        self.link.send(bytes)
    }

    /// Link events and frames since the last poll.
    fn pump(&mut self, now: f64) -> Changed {
        let mut changed = Changed::default();
        let url = self.cfg.core_url.clone();
        for ev in self.link.drain() {
            let c = match ev {
                LinkEvent::Connecting => {
                    // While retrying, keep saying why the last attempt failed.
                    let detail = match self.store.link.state {
                        LinkState::Disconnected => format!("재연결 중 · {}", self.store.link.detail),
                        _ => format!("코어 연결 중 · {url}"),
                    };
                    self.store.set_link(LinkState::Connecting, detail)
                }
                LinkEvent::Connected => {
                    self.framed_points = false;
                    self.framed_drone = false;
                    self.focused_detection = None;
                    self.store.set_link(LinkState::Connected, format!("코어 연결됨 · {url}"))
                }
                LinkEvent::Disconnected(reason) => {
                    self.release_manual();
                    self.store.set_link(LinkState::Disconnected, format!("코어 연결 끊김 · {reason}"))
                }
                LinkEvent::Frame(bytes) => self.store.apply_frame(&bytes, now),
            };
            changed.merge(c);
        }
        changed
    }

    fn poll(&mut self, doc: &mut BaseDocument, cx: Option<Context<'_>>) -> bool {
        if let Some(cx) = cx {
            self.link.set_waker(cx.waker());
        }
        let now = crate::now_ms();
        let changed = self.pump(now);

        // Scene work for the widget.
        {
            let mut v = self.view.borrow_mut();
            if v.needs_resync {
                v.needs_resync = false;
                v.push_ops(self.store.chunks.full_resync());
            }
            if !changed.scene.is_empty() {
                if !self.framed_points {
                    let first = changed.scene.iter().find_map(|op| match op {
                        SceneOp::InsertPoints(layer, p) if *layer == Layer::Recon || self.cfg.screen == Screen::Control => {
                            chunk_center(p)
                        }
                        _ => None,
                    });
                    if let Some(c) = first {
                        v.camera_request = Some(CameraRequest::FrameIfIdle(c));
                        self.framed_points = true;
                    }
                }
                v.push_ops(changed.scene.clone());
            }
        }

        for id in &changed.revealed {
            if let Some(d) = self.store.detections.get(id) {
                let card = views::detection_card(d);
                let kind = if views::is_person(d.category()) { ToastKind::Person } else { ToastKind::Danger };
                let text = format!("{} · {}", card.kicker, card.label);
                self.toasts.push(Toast {
                    text,
                    kind,
                    expires_ms: now + views::TOAST_LIFETIME_MS,
                });
                self.focused_detection = Some(id.clone());
            }
        }
        if changed.route && self.cfg.screen == Screen::Control
            && let Some(r) = &self.store.route
        {
            let n = r.waypoints.len();
            self.toast(format!("코어가 경로를 확인했습니다 · 웨이포인트 {n}개"), ToastKind::Info);
        }

        self.tick_manual(now);
        self.toasts.retain(|t| t.expires_ms > now);

        let overlay_changed = self.update_overlay(now);
        let dom_changed = self.render(doc, now);
        overlay_changed || dom_changed || changed.any()
    }

    fn update_overlay(&mut self, now: f64) -> bool {
        let anchor_enu = |g: &model::Gps| self.enu(g);
        let drones: Vec<OverlayDrone> = self
            .store
            .drones_in_order()
            .into_iter()
            .filter_map(|d| {
                let t = d.telemetry.as_ref()?;
                let g = t.gps.clone()?;
                Some(OverlayDrone {
                    pos: anchor_enu(&g),
                    heading_deg: t.heading_deg,
                    tint: views::drone_tint(d.id),
                    active: self.store.active_drone == Some(d.id),
                    stale: d.stale(now),
                })
            })
            .collect();
        let route: Vec<[f64; 3]> = self
            .store
            .route
            .as_ref()
            .map(|r| r.waypoints.iter().map(&anchor_enu).collect())
            .unwrap_or_default();
        let draft: Vec<[f64; 3]> = if self.route_open { self.draft.iter().map(&anchor_enu).collect() } else { Vec::new() };
        let detections: Vec<OverlayDetection> = self
            .store
            .detections
            .visible(self.store.chunks.arrived())
            .filter_map(|d| {
                let g = d.gps.clone()?;
                Some(OverlayDetection {
                    id: d.id.clone(),
                    kind: if views::is_person(d.category()) { MarkerKind::Person } else { MarkerKind::Hazard },
                    pos: anchor_enu(&g),
                    outline: d.outline.iter().map(&anchor_enu).collect(),
                    focused: self.focused_detection.as_deref() == Some(d.id.as_str()),
                })
            })
            .collect();
        let overlay = Overlay {
            drones,
            route,
            draft,
            detections,
        };

        if self.cfg.screen == Screen::Control && !self.framed_drone
            && let Some(d) = overlay.drones.iter().find(|d| d.active)
        {
            self.view.borrow_mut().camera_request = Some(CameraRequest::FrameIfIdle(d.pos));
            self.framed_drone = true;
        }

        let mut v = self.view.borrow_mut();
        if !overlay_eq(&v.overlay, &overlay) {
            v.overlay = overlay;
            v.dirty = true;
            return true;
        }
        v.dirty
    }

    fn render(&mut self, doc: &mut BaseDocument, now: f64) -> bool {
        let screen = self.cfg.screen;
        let (stats_points, backend, gpu_active) = {
            let v = self.view.borrow();
            (v.stats.points, v.stats.backend.clone(), v.stats.gpu_active)
        };
        let store = &self.store;
        let mut dom = Dom {
            doc,
            cache: &mut self.cache,
            changed: false,
        };

        // Camera panel (both screens).
        let cam = views::camera_view(store);
        dom.text("video-label", cam.label);
        dom.text("video-state", &cam.state);
        dom.text("video-url", &cam.url);
        dom.text("video-overlay", &cam.overlay);
        dom.html("toast-root", &views::toasts_html(&self.toasts));

        match screen {
            Screen::Control => {
                let m = views::mission_view(store, now);
                dom.class("mission-panel", &m.panel_class);
                dom.class("mission-link", &m.link_class);
                dom.text("mission-link-text", &m.link_text);
                for (id, class) in &m.steps {
                    dom.class(id, class);
                }
                dom.text("mission-message", &m.message);
                dom.text("mission-sub", &m.sub);

                dom.html("telemetry-list", &views::telemetry_rows_html(store, now));
                dom.class(
                    "telemetry-empty",
                    if store.drones.is_empty() { "telemetry-panel__empty" } else { "telemetry-panel__empty is-hidden" },
                );
                dom.class("manual-badge", if self.manual_active { "manual-badge" } else { "manual-badge is-hidden" });

                dom.class(
                    "route-overlay",
                    if self.route_open { "route-modal-overlay" } else { "route-modal-overlay is-hidden" },
                );
                dom.html("route-list", &views::route_list_html(&self.draft));
                match &self.route_hint {
                    Some(h) => {
                        dom.class("route-hint", "route-modal__hint is-warn");
                        dom.text("route-hint", h);
                    }
                    None => {
                        dom.class("route-hint", "route-modal__hint");
                        dom.text("route-hint", "위도·경도·고도(m)를 입력하고 추가를 누르세요 · 목록에서 삭제할 수 있습니다");
                    }
                }
                let target = match store.route_leader().and_then(|id| store.drones.get(&id)) {
                    Some(d) => format!("배정 대상: {}", views::station_label(d.station)),
                    None => "배정 대상: 편대 (드론 연결 전)".to_string(),
                };
                dom.text("route-target", &target);

                dom.class(
                    "settings-overlay",
                    if self.settings_open { "settings-overlay" } else { "settings-overlay is-hidden" },
                );
                for mode in DisplayMode::ALL {
                    let class = if mode == self.display { "settings-option is-active" } else { "settings-option" };
                    dom.class(&format!("display-{}", mode.id()), class);
                }
                let (terrain, buildings) = store.chunks.visible_points().iter().fold((0, 0), |(t, b), (l, _)| match l {
                    Layer::Terrain => (t + 1, b),
                    Layer::Buildings => (t, b + 1),
                    Layer::Recon => (t, b),
                });
                let terrain_text = if terrain + buildings == 0 {
                    "코어 지형 피드 대기 중 (VWorld 키가 없으면 지형·건물 없이 동작)".to_string()
                } else {
                    format!("지형 청크 {terrain} · 건물 청크 {buildings}")
                };
                dom.text("settings-terrain", &terrain_text);
                let renderer = if gpu_active {
                    format!("pointblitz · {backend} · 점 {stats_points} · 메시 블록 {}", store.chunks.mesh_blocks())
                } else {
                    "GPU 준비 중".to_string()
                };
                dom.text("settings-renderer", &renderer);
            }
            Screen::Status => {
                let (state, text) = views::net_view(store);
                dom.attr("net-dot", "data-state", state);
                dom.text("net-text", &text);

                let s = views::server_view(store);
                dom.attr("server-dot", "data-state", s.state);
                dom.text("server-text", s.text);
                dom.text("server-detail", &s.detail);
                dom.text("server-link", &s.link);
                let levels = views::segment_levels(store.server.as_ref(), &store.chunks);
                dom.class(
                    "server-ladder",
                    if levels.is_empty() { "server-status__ladder is-empty" } else { "server-status__ladder" },
                );
                dom.html("server-ladder", &views::ladder_html(&levels));

                let arrived = store.chunks.arrived();
                let (people, hazards) = store.detections.visible(arrived).fold((0, 0), |(p, h), d| {
                    if views::is_person(d.category()) { (p + 1, h) } else { (p, h + 1) }
                });
                dom.text("legend-person", &format!("생존자 {people}"));
                dom.text("legend-danger", &format!("위험구역 {hazards}"));

                match self.focused_detection.as_deref().and_then(|id| store.detections.get(id)) {
                    Some(d) => {
                        let card = views::detection_card(d);
                        dom.class("detect-card", &card.class);
                        dom.text("detect-kicker", card.kicker);
                        dom.text("detect-label", &card.label);
                        dom.text("detect-conf", &card.conf);
                        dom.attr("detect-fill", "style", &card.fill_style);
                    }
                    None => dom.class("detect-card", "detect-card"),
                }
            }
        }
        dom.changed
    }

    /// Applies clicks and keys collected during event dispatch.
    fn process_actions(&mut self, doc: &mut BaseDocument) -> bool {
        let actions = std::mem::take(&mut self.actions);
        if actions.is_empty() {
            return false;
        }
        for action in actions {
            match action {
                UiAction::Click { action, arg } => self.click(doc, &action, arg.as_deref()),
                UiAction::Key { code, down, in_input } => self.key(code, down, in_input),
            }
        }
        let now = crate::now_ms();
        self.tick_manual(now);
        self.update_overlay(now);
        self.render(doc, now);
        true
    }

    fn click(&mut self, doc: &BaseDocument, action: &str, arg: Option<&str>) {
        match action {
            "open-route" => {
                self.route_open = true;
                self.settings_open = false;
                self.route_hint = None;
            }
            "close-route" => self.route_open = false,
            "add-waypoint" => {
                let lat = input_value(doc, "wp-lat").unwrap_or_default();
                let lon = input_value(doc, "wp-lon").unwrap_or_default();
                let alt = input_value(doc, "wp-alt").unwrap_or_default();
                match views::parse_waypoint(&lat, &lon, &alt) {
                    Ok(wp) => {
                        self.draft.push(wp);
                        self.route_hint = None;
                    }
                    Err(e) => self.route_hint = Some(e),
                }
            }
            "remove-waypoint" => {
                if let Some(i) = arg.and_then(|a| a.parse::<usize>().ok())
                    && i < self.draft.len()
                {
                    self.draft.remove(i);
                }
            }
            "clear-route" => {
                self.draft.clear();
                self.route_hint = None;
            }
            "assign-route" => {
                if self.draft.len() < 2 {
                    self.route_hint = Some("경로에는 최소 2개의 웨이포인트가 필요합니다".into());
                    return;
                }
                let looped = checkbox_checked(doc, "route-loop");
                // A route belongs to the formation and the centre drone carries it.
                // Before any drone is connected the legacy default id 1 is used.
                let leader = self.store.route_leader().unwrap_or(1);
                let n = self.draft.len();
                let body = protocol::assign_route(leader, &self.draft, looped);
                if self.send(body) {
                    self.toast(
                        format!("경로 전송됨 · 웨이포인트 {n}개{}", if looped { " · 왕복" } else { "" }),
                        ToastKind::Info,
                    );
                    self.route_open = false;
                } else {
                    self.toast("코어에 연결되어 있지 않아 경로를 전송하지 못했습니다", ToastKind::Danger);
                }
            }
            "open-settings" => {
                self.settings_open = true;
                self.route_open = false;
            }
            "close-settings" => self.settings_open = false,
            "set-display" => {
                if let Some(mode) = arg.and_then(DisplayMode::parse)
                    && mode != self.display
                {
                    self.display = mode;
                    let mut v = self.view.borrow_mut();
                    v.display = mode;
                    v.push_ops(vec![SceneOp::RebuildPoints(self.store.chunks.visible_points())]);
                }
            }
            "select-drone" => {
                if let Some(id) = arg.and_then(|a| a.parse::<u32>().ok()) {
                    self.store.select_drone(id);
                }
            }
            "close-detection" => self.focused_detection = None,
            other => crate::log(&format!("[ui] unhandled action {other}")),
        }
    }

    fn key(&mut self, code: Code, down: bool, in_input: bool) {
        if in_input {
            return;
        }
        if down && code == Code::Escape {
            self.route_open = false;
            self.settings_open = false;
            self.focused_detection = None;
            return;
        }
        if down && code == Code::KeyF {
            let target = self
                .store
                .active()
                .and_then(|d| d.gps())
                .map(|g| self.enu(&g))
                .or_else(|| self.view.borrow().overlay.detections.last().map(|d| d.pos));
            if let Some(p) = target {
                let mut v = self.view.borrow_mut();
                v.camera_request = Some(CameraRequest::Focus(p));
                v.dirty = true;
            }
            return;
        }
        if self.cfg.screen != Screen::Control || self.route_open || self.settings_open {
            return;
        }
        match code {
            Code::ArrowUp
            | Code::ArrowDown
            | Code::ArrowLeft
            | Code::ArrowRight
            | Code::KeyW
            | Code::KeyA
            | Code::KeyS
            | Code::KeyD
            | Code::KeyQ
            | Code::KeyE => {
                if down {
                    self.held.insert(code);
                } else {
                    self.held.remove(&code);
                }
            }
            Code::Digit1 | Code::Digit2 | Code::Digit3 if down => {
                let n = match code {
                    Code::Digit1 => 0,
                    Code::Digit2 => 1,
                    _ => 2,
                };
                let id = self.store.drones_in_order().get(n).map(|d| d.id);
                if let Some(id) = id {
                    self.store.select_drone(id);
                }
            }
            Code::Tab if down => {
                self.store.cycle_drone();
            }
            _ => {}
        }
    }

    fn axes(&self) -> (f64, f64, f64) {
        let on = |a: Code, b: Code| if self.held.contains(&a) || self.held.contains(&b) { 1.0 } else { 0.0 };
        let forward = on(Code::ArrowUp, Code::KeyW) - on(Code::ArrowDown, Code::KeyS);
        let yaw = on(Code::ArrowRight, Code::KeyD) - on(Code::ArrowLeft, Code::KeyA);
        let climb = on(Code::KeyE, Code::KeyE) - on(Code::KeyQ, Code::KeyQ);
        (forward, yaw, climb)
    }

    /// Streams stick values at 10 Hz while keys are held, then one neutral frame on
    /// release so a lost key-up cannot leave a drone on a stale command.
    fn tick_manual(&mut self, now: f64) {
        if self.cfg.screen != Screen::Control {
            return;
        }
        let active = !self.held.is_empty();
        let Some(drone) = self.store.active_drone else {
            self.manual_active = false;
            return;
        };
        if active {
            if now - self.last_manual_ms >= MANUAL_PERIOD_MS {
                let (f, y, c) = self.axes();
                self.send(protocol::manual_control(drone, f, y, c));
                self.last_manual_ms = now;
            }
        } else if self.manual_active {
            self.send(protocol::manual_control(drone, 0.0, 0.0, 0.0));
        }
        self.manual_active = active;
    }

    fn release_manual(&mut self) {
        self.held.clear();
        self.manual_active = false;
    }
}

fn overlay_eq(a: &Overlay, b: &Overlay) -> bool {
    a.route == b.route
        && a.draft == b.draft
        && a.drones.len() == b.drones.len()
        && a.drones.iter().zip(&b.drones).all(|(x, y)| {
            x.pos == y.pos && x.heading_deg == y.heading_deg && x.active == y.active && x.stale == y.stale && x.tint == y.tint
        })
        && a.detections.len() == b.detections.len()
        && a.detections.iter().zip(&b.detections).all(|(x, y)| {
            x.id == y.id && x.pos == y.pos && x.outline == y.outline && x.focused == y.focused && x.kind == y.kind
        })
}

/// A screen as a Blitz document.
pub struct SkylensDocument {
    inner: BaseDocument,
    app: App,
}

impl SkylensDocument {
    /// Parses the screen's HTML, mounts the 3D widget and starts the core link.
    /// `font_ctx` is None on native (system fonts) and the bundled font on web.
    pub fn new(cfg: Config, font_ctx: Option<FontContext>) -> Self {
        let html = page_html(cfg.screen, &cfg.anchor);
        let config = DocumentConfig {
            html_parser_provider: Some(Arc::new(HtmlProvider)),
            font_ctx,
            ..Default::default()
        };
        let mut inner = HtmlDocument::from_html(&html, config).into_inner();
        let view = shared_view();
        if let Some(node) = inner.get_element_by_id("view3d") {
            inner.mutate().set_custom_widget(node, Box::new(View3d::new(view.clone())));
        } else {
            crate::log("[app] #view3d missing; 3D view disabled");
        }
        let mut app = App::new(cfg, view);
        app.render(&mut inner, crate::now_ms());
        Self { inner, app }
    }
}

impl Document for SkylensDocument {
    fn inner(&self) -> DocGuard<'_> {
        DocGuard::Ref(&self.inner)
    }

    fn inner_mut(&mut self) -> DocGuardMut<'_> {
        DocGuardMut::Ref(&mut self.inner)
    }

    fn handle_ui_event(&mut self, event: UiEvent) {
        {
            let mut driver = EventDriver::new(
                &mut self.inner,
                Collector {
                    actions: &mut self.app.actions,
                },
            );
            driver.handle_ui_event(event);
        }
        let changed = self.app.process_actions(&mut self.inner);
        // Camera drags reach the widget without a DOM change; ask for a frame.
        if changed || self.app.view.borrow().dirty {
            self.inner.shell_provider.request_redraw();
        }
    }

    fn poll(&mut self, task_context: Option<Context<'_>>) -> bool {
        self.app.poll(&mut self.inner, task_context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn parses_cli_arguments() {
        let cfg = Config::from_args(args("--screen control --core 10.0.0.5:8080 --anchor 36.37,127.35,40")).unwrap();
        assert_eq!(cfg.screen, Screen::Control);
        assert_eq!(cfg.core_url, "ws://10.0.0.5:8080/viewer");
        assert_eq!(cfg.anchor, Gps { lat: 36.37, lon: 127.35, alt: 40.0 });

        let cfg = Config::from_args(args("--screen=status --core=ws://core:8080/viewer")).unwrap();
        assert_eq!(cfg.screen, Screen::Status);
        assert_eq!(cfg.core_url, "ws://core:8080/viewer");

        assert_eq!(Config::from_args(Vec::new()).unwrap(), Config::default());
        assert!(Config::from_args(args("--screen tower")).is_err());
        assert!(Config::from_args(args("--core")).is_err());
        assert!(Config::from_args(args("--bogus")).is_err());
    }

    #[test]
    fn normalizes_core_urls() {
        assert_eq!(normalize_core_url("core:8080"), "ws://core:8080/viewer");
        assert_eq!(normalize_core_url("http://core:8080/"), "ws://core:8080/viewer");
        assert_eq!(normalize_core_url("https://bridge"), "wss://bridge/viewer");
        assert_eq!(normalize_core_url("ws://core:8080/custom"), "ws://core:8080/custom");
    }

    #[test]
    fn parses_anchors() {
        assert_eq!(parse_anchor("36.1,127.2"), Some(Gps { lat: 36.1, lon: 127.2, alt: DEFAULT_ANCHOR.alt }));
        assert_eq!(parse_anchor("36.1, 127.2, 12"), Some(Gps { lat: 36.1, lon: 127.2, alt: 12.0 }));
        assert_eq!(parse_anchor("100,0"), None);
        assert_eq!(parse_anchor("1"), None);
        assert_eq!(parse_anchor("a,b"), None);
    }

    #[test]
    fn page_html_inlines_styles_and_fills_placeholders() {
        for screen in [Screen::Control, Screen::Status] {
            let page = page_html(screen, &DEFAULT_ANCHOR);
            assert!(!page.contains("<link rel=\"stylesheet\""), "{screen:?}");
            assert!(!page.contains("{{"), "{screen:?}");
            assert!(page.contains("--control-ink"), "{screen:?}");
            assert!(page.contains("id=\"view3d\""), "{screen:?}");
            assert!(page.contains("id=\"video-url\""), "{screen:?}");
        }
        let control = page_html(Screen::Control, &DEFAULT_ANCHOR);
        assert!(control.contains("value=\"36.368500\""));
        assert!(control.contains("data-action=\"assign-route\""));
        assert!(control.contains(".route-modal"));
        let status = page_html(Screen::Status, &DEFAULT_ANCHOR);
        assert!(status.contains("id=\"server-ladder\""));
        assert!(status.contains(".segment-row"));
    }

    #[test]
    fn chunk_center_is_origin_plus_box_middle() {
        use pointblitz::io::chunk::encode_with;
        use pointblitz::io::Point;
        let pts = vec![
            Point { position: [100.0, 200.0, 10.0], color: [0; 3] },
            Point { position: [110.0, 220.0, 30.0], color: [0; 3] },
        ];
        let c = chunk_center(&encode_with(0, 0, 0, &pts, 12)).unwrap();
        assert!((c[0] - 105.0).abs() < 1e-3 && (c[1] - 210.0).abs() < 1e-3 && (c[2] - 20.0).abs() < 1e-3);
    }
}
