//! The status board as a Blitz screen: the legacy `status.html` DOM and `status-panels.css`,
//! one read-only socket to the core (`ws /viewer`), and the DOM writes the legacy TypeScript made
//! (see `ui/DOM.md`).
//!
//! The board never sends anything to the core and never shows the base map: it shows the site's
//! reconstruction as it arrives, the drones from telemetry, the route, the detections once their
//! segment is final, and the health of the link.

use crate::board::Board;
use crate::card::Card;
use crate::inset::InsetScene;
use crate::minimap::{MapData, MinimapWidget};
use crate::panels::{self, BannerInput, PanelState};
use crate::scene::StatusScene;
use skylens_shared::proto::model::v1 as model;
use skylens_shared::proto::payload::v1 as payload;
use skylens_viewer::camera::{SyncEvent, SyncState};
use skylens_viewer::link::VIEWER_PATH;
use skylens_viewer::protocol::Body;
use skylens_viewer::widgets::{CardKind, LoadingScreen, NetBadge, Toasts};
use skylens_viewer::widgets::detection_card::detection_label;
use skylens_viewer::{CoreLink, Dom, LinkEvent, NodeId, Page, PollCx, ScreenApp, ScreenDocument, SiteFrame, UiInput, View3d, q};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

/// The legacy template and the board's own stylesheet, both unmodified.
pub const STATUS_HTML: &str = include_str!("ui/status.html");
pub const STATUS_PANELS_CSS: &str = include_str!("ui/status-panels.css");

pub const LOADING_TITLE: &str = "현황판 · 3D 복원 데이터 로딩";
pub const WINDOW_TITLE: &str = "SkyLens · 현황판 - 실시간 3D 복원";
pub const DEFAULT_CORE: &str = "ws://127.0.0.1:8080/viewer";

/// The page: legacy template plus the shared CSS and `status-panels.css`.
pub fn page() -> Page {
    Page::legacy(STATUS_HTML, &[STATUS_PANELS_CSS])
}

/// How the board starts.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// `ws://host:port/viewer`.
    pub core_url: String,
    /// The point panel starts open (legacy `?points=off` starts it closed).
    pub points_on: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { core_url: DEFAULT_CORE.to_string(), points_on: true }
    }
}

/// Parses `--core URL` (host:port, http or ws URL; `/viewer` added when no path) and
/// `--points on|off`.
pub fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Options, String> {
    let mut o = Options::default();
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        let (key, inline) = match a.split_once('=') {
            Some((k, v)) => (k.to_string(), Some(v.to_string())),
            None => (a.clone(), None),
        };
        let mut value = |name: &str| inline.clone().or_else(|| it.next()).ok_or(format!("{name} needs a value"));
        match key.as_str() {
            "--core" => o.core_url = skylens_viewer::link::normalize_core_url(&value("--core")?, VIEWER_PATH),
            "--points" => {
                o.points_on = match value("--points")?.as_str() {
                    "on" => true,
                    "off" => false,
                    v => return Err(format!("--points on|off, not {v}")),
                }
            }
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    Ok(o)
}

pub const USAGE: &str = "usage: skylens-client [--core ws://host:8080/viewer] [--points on|off]";

/// Ladder row nodes, by segment index, in first-seen order.
struct LadderRow {
    pips: Vec<NodeId>,
    note: NodeId,
}

pub struct StatusApp {
    link: CoreLink,
    board: Rc<RefCell<Board>>,
    scene: Rc<RefCell<StatusScene>>,
    inset: Option<Rc<RefCell<InsetScene>>>,
    map: Rc<RefCell<MapData>>,
    loading: Option<LoadingScreen>,
    toasts: Toasts,
    badge: Option<NetBadge>,
    card: Option<Card>,
    ladder: BTreeMap<u32, LadderRow>,
    minimap_size: Option<(i64, i64, u32, u32)>,
    site_seen: bool,
    last_poll_ms: Option<f64>,
}

impl StatusApp {
    pub fn new(opts: &Options) -> StatusApp {
        let board = Rc::new(RefCell::new(Board::new(opts.points_on)));
        StatusApp {
            link: CoreLink::start(opts.core_url.clone()),
            scene: Rc::new(RefCell::new(StatusScene::new(board.clone()))),
            board,
            inset: None,
            map: Rc::new(RefCell::new(MapData::default())),
            loading: None,
            toasts: Toasts::new(),
            badge: None,
            card: None,
            ladder: BTreeMap::new(),
            minimap_size: None,
            site_seen: false,
            last_poll_ms: None,
        }
    }

    /// The board as a Blitz document.
    pub fn document(opts: &Options) -> ScreenDocument<StatusApp> {
        ScreenDocument::new(&page(), StatusApp::new(opts))
    }

    // ---- link ---------------------------------------------------------------------------------

    fn on_event(&mut self, ev: LinkEvent, dom: &mut Dom<'_>, now: f64) {
        match ev {
            LinkEvent::Connecting | LinkEvent::Invalid(_) => {}
            LinkEvent::Connected => {
                self.board.borrow_mut().reconnected();
                if let Some(l) = &mut self.loading {
                    l.progress(dom, 60.0, None);
                }
            }
            LinkEvent::Disconnected(_) => {
                if !self.site_seen
                    && let Some(l) = &mut self.loading
                    && !l.is_finishing()
                {
                    l.fail(dom, self.link.status.detail(), now);
                }
            }
            LinkEvent::Message(env) => {
                let Some(body) = env.body else { return };
                let mut b = self.board.borrow_mut();
                match body {
                    Body::Site(site) => {
                        if let Some(frame) = SiteFrame::from_site(&site) {
                            b.site = Some(frame);
                            self.site_seen = true;
                            if let Some(l) = &mut self.loading {
                                l.progress(dom, 100.0, None);
                                l.done(dom, now);
                            }
                        }
                    }
                    Body::Chunk(chunk) => match chunk.kind() {
                        // The tower's base map; the board shows only the reconstruction.
                        model::ChunkKind::Terrain | model::ChunkKind::Buildings => {}
                        _ => {
                            b.chunk(&chunk);
                        }
                    },
                    Body::Telemetry(t) => b.telemetry(&t),
                    Body::AssignRoute(r) => b.assign_route(&r),
                    Body::MissionStatus(m) => b.mission = Some(m),
                    Body::ServerStatus(s) => b.core_status = Some(s),
                    _ => {}
                }
            }
        }
    }

    // ---- detections and camera ------------------------------------------------------------------

    fn step_detections(&mut self, dom: &mut Dom<'_>, now: f64, dt: f64) {
        let mut b = self.board.borrow_mut();
        // Reveal: a toast per detection whose segment's final level just arrived.
        for d in b.reveal_new() {
            let kind = CardKind::of(d.category());
            self.toasts.show(dom, &detection_label(&d), kind.toast(), now);
        }
        // Camera state machine; the card opens on focus and closes back in sync.
        for ev in b.cam.step(dt) {
            if let SyncEvent::StateChanged(SyncState::Synced) = ev
                && let Some(card) = &mut self.card
            {
                card.close(dom);
            }
        }
        if b.cam.state == SyncState::Synced
            && b.cam.focused.is_none()
            && let Some((id, at)) = b.focus_candidate()
        {
            b.cam.focus(id.clone(), at);
            let det = b.store.detection(&id).cloned();
            if let (Some(card), Some(det)) = (&mut self.card, det) {
                card.open(dom, &det);
            }
        }
        if let Some(card) = &mut self.card {
            card.set_camera_state(dom, b.cam.state);
        }
    }

    /// `탐지 확인`: the detection is acknowledged, the card closes and the camera returns.
    fn confirm(&mut self, dom: &mut Dom<'_>) {
        let Some(card) = &mut self.card else { return };
        let Some(id) = card.open_id().map(str::to_string) else { return };
        card.close(dom);
        let mut b = self.board.borrow_mut();
        b.confirmed.insert(id.clone());
        if b.cam.focused.as_deref() == Some(id.as_str()) {
            b.cam.confirm();
        }
    }

    // ---- panels ---------------------------------------------------------------------------------

    fn render_server_panel(&mut self, dom: &mut Dom<'_>) {
        let b = self.board.borrow();
        let status = &self.link.status;
        let state = PanelState::of(status.connected(), status.receiving);
        dom.set_attr(q(".server-status__dot"), "data-state", state.data_state());
        dom.set_text(q(".server-status__text"), state.label());
        let core = b.core_status.as_ref();
        let chunks = core.map_or(0, |c| c.chunks).max(b.local_chunks);
        let detections = core.map_or(0, |c| c.detections).max(b.local_detections);
        let details = dom.find_all(".server-status__detail");
        if let [counts, line, ..] = details.as_slice() {
            dom.set_text(*counts, &panels::counts_line(chunks, detections, core.and_then(|c| c.latency_ms)));
            let mission = b.mission.as_ref().map(|m| (m.message.as_str(), m.drones_online));
            dom.set_text(*line, &panels::link_line(state, mission, status.badge_text()));
        }
        // Ladder: one row per segment the core reports, appended in first-seen order.
        let segments: Vec<(u32, u32)> =
            core.map(|c| c.segments.iter().map(|s| (s.index, panels::level_number(s.level()))).collect()).unwrap_or_default();
        drop(b);
        for (index, level) in segments {
            if !self.ladder.contains_key(&index) {
                let added = dom.append_html(q(".server-status__ladder"), &panels::ladder_row_markup(index, level));
                let Some(row) = added.first().copied() else { continue };
                let pips = dom.find_in(row, ".segment-row__bar").map(|bar| dom.children(bar)).unwrap_or_default();
                let Some(note) = dom.find_in(row, ".segment-row__note") else { continue };
                self.ladder.insert(index, LadderRow { pips, note });
            }
            let row = &self.ladder[&index];
            for (i, pip) in row.pips.iter().enumerate() {
                dom.set_attr(*pip, "data-on", if (i as u32) < level { "1" } else { "0" });
            }
            dom.set_text(row.note, &panels::ladder_note(level));
            dom.set_attr(row.note, "data-refining", panels::ladder_refining(level));
        }
    }

    fn render_banner(&mut self, dom: &mut Dom<'_>) {
        let b = self.board.borrow();
        let input = BannerInput {
            link_up: self.link.status.connected(),
            detail: self.link.status.badge_text(),
            has_geometry: b.has_geometry(),
            preparing: b.first_progress.map(|p| f64::from(p) * 100.0),
            failed: b.geometry_failed,
        };
        let text = panels::banner_text(&input);
        dom.set_text("board-waiting", &text);
        dom.toggle_class("board-waiting", "is-visible", !text.is_empty());
    }

    fn render_points(&mut self, dom: &mut Dom<'_>) {
        let (on, delivered, refined, total, has_segments, range) = {
            let b = self.board.borrow();
            let delivered = b.segments.len();
            let refined = b.segments.values().filter(|s| s.refined).count();
            (b.point_layer_on, delivered, refined, b.total_points(), b.store.segments().next().is_some(), b.height_range())
        };
        dom.set_text(q(".point-toggle__meta"), &panels::point_meta(on, delivered, refined, total));
        dom.toggle_class("point-toggle", "is-on", on);
        dom.set_attr("point-toggle", "aria-pressed", if on { "true" } else { "false" });

        // The panel appears with the first point chunk and stays.
        if self.inset.is_none() && has_segments {
            let added = dom.append_html(q("section.pane"), &panels::inset_markup(on));
            if let Some(view) = added.first().and_then(|root| dom.find_in(*root, ".point-inset__view")) {
                let inset = Rc::new(RefCell::new(InsetScene::new(self.board.clone())));
                dom.mount_widget(view, View3d::boxed(inset.clone()));
                self.inset = Some(inset);
            }
        }
        if self.inset.is_some() {
            dom.toggle_class("point-inset", "is-open", on);
            dom.toggle_class("point-inset", "has-data", has_segments);
            if total > 0 {
                dom.set_style(q(".point-inset__empty"), "display", "none");
            } else if dom.attr(q(".point-inset__empty"), "style").is_some_and(|s| s.contains("display")) {
                dom.remove_style(q(".point-inset__empty"), "display");
            }
            let (high, low) = panels::legend_ticks(range);
            dom.set_text(q(".point-inset__tick--high"), &high);
            dom.set_text(q(".point-inset__tick--low"), &low);
        }
    }

    /// The minimap canvas takes the host's inner size, like the legacy `resize`.
    fn size_minimap(&mut self, dom: &mut Dom<'_>) {
        let Some(host) = dom.find("minimap") else { return };
        let Some(canvas) = dom.find(q("#minimap .minimap__canvas")) else { return };
        let (w, h) = {
            let doc = dom.doc_ref();
            let Some(node) = doc.get_node(host) else { return };
            let l = node.final_layout();
            let w = (l.size.width - l.border.left - l.border.right).round() as i64;
            let h = (l.size.height - l.border.top - l.border.bottom).round() as i64;
            (if w > 0 { w } else { 160 }, if h > 0 { h } else { 160 })
        };
        let scale = dom.doc_ref().viewport().scale_f64();
        let dpr = if scale > 0.0 { scale.min(2.0) } else { 1.0 };
        let (pw, ph) = (((w as f64) * dpr).round().max(1.0) as u32, ((h as f64) * dpr).round().max(1.0) as u32);
        if self.minimap_size == Some((w, h, pw, ph)) {
            return;
        }
        self.minimap_size = Some((w, h, pw, ph));
        dom.set_attr(canvas, "width", &pw.to_string());
        dom.set_attr(canvas, "height", &ph.to_string());
        dom.set_style(canvas, "width", &format!("{w}px"));
        dom.set_style(canvas, "height", &format!("{h}px"));
    }

    fn update_map(&mut self) {
        let b = self.board.borrow();
        let active = b.active_drone;
        let mut m = self.map.borrow_mut();
        m.drones = b.drones.iter().map(|d| ([d.pos.x, d.pos.y], Some(d.id) == active)).collect();
        m.detections = b
            .store
            .visible_detections()
            .filter_map(|d| {
                let p = b.detection_pos(d)?;
                Some(([p.x, p.y], d.category() == payload::DetectionCategory::Person))
            })
            .collect();
        m.route = b.route.as_ref().map(|r| r.iter().map(|p| [p.x, p.y]).collect()).unwrap_or_default();
        m.chunks = b.segment_centers().into_iter().map(|(_, c)| [c.x, c.y]).collect();
    }
}

impl ScreenApp for StatusApp {
    fn mount(&mut self, dom: &mut Dom<'_>) {
        dom.mount_widget("status-view", View3d::boxed(self.scene.clone()));
        if let Some(canvas) = dom.find(q("#minimap .minimap__canvas")) {
            dom.mount_widget(canvas, MinimapWidget::boxed(self.map.clone()));
        }
        self.badge = NetBadge::mount(dom, "net-status");
        if dom.find("minimap").is_some() {
            // The legacy CSS puts the link badge and the minimap at the same top-right corner
            // (both `right: 16px`), so the 200 px minimap covered the badge. Seat the badge to
            // the minimap's left instead: 16 px margin + 200 px minimap + 8 px gap.
            dom.set_style("net-status", "right", "224px");
        }
        self.card = Card::bind(dom);
        self.loading = LoadingScreen::mount(dom, LOADING_TITLE);
        if let Some(l) = &mut self.loading {
            l.progress(dom, 0.0, None);
        }
    }

    fn poll(&mut self, dom: &mut Dom<'_>, cx: &mut PollCx<'_>) -> bool {
        if let Some(w) = cx.waker {
            self.link.set_waker(w);
        }
        let now = cx.now_ms;
        let dt = self.last_poll_ms.map_or(0.0, |l| ((now - l) / 1000.0).clamp(0.0, 0.25));
        self.last_poll_ms = Some(now);

        for ev in self.link.drain() {
            self.on_event(ev, dom, now);
        }
        self.board.borrow_mut().refresh();
        self.step_detections(dom, now, dt);

        if let Some(badge) = &mut self.badge {
            badge.show_link(dom, &self.link.status);
        }
        self.render_server_panel(dom);
        self.render_banner(dom);
        self.render_points(dom);
        self.size_minimap(dom);
        self.update_map();

        if let Some(l) = &mut self.loading {
            l.tick(dom, now);
            if l.is_removed() {
                self.loading = None;
            }
        }
        self.toasts.tick(dom, now);
        true
    }

    fn input(&mut self, input: &UiInput, dom: &mut Dom<'_>) -> bool {
        if Card::is_confirm(input) {
            self.confirm(dom);
            return true;
        }
        if input.is_click() && input.closest_id("point-toggle").is_some() {
            {
                let mut b = self.board.borrow_mut();
                b.point_layer_on = !b.point_layer_on;
            }
            self.render_points(dom);
            return true;
        }
        if input.kind == skylens_viewer::shell::InputKind::DoubleClick
            && input.closest_class("point-inset__view").is_some()
            && let Some(inset) = &self.inset
        {
            inset.borrow_mut().reframe();
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_default_to_the_local_core() {
        let o = parse_args(Vec::<String>::new()).unwrap();
        assert_eq!(o, Options::default());
        assert_eq!(o.core_url, "ws://127.0.0.1:8080/viewer");
        let o = parse_args(["--core".to_string(), "core.local:8080".to_string()]).unwrap();
        assert_eq!(o.core_url, "ws://core.local:8080/viewer");
        let o = parse_args(["--core=http://10.0.0.2:8080".to_string(), "--points".to_string(), "off".to_string()]).unwrap();
        assert_eq!(o.core_url, "ws://10.0.0.2:8080/viewer");
        assert!(!o.points_on);
        assert!(parse_args(["--points".to_string(), "maybe".to_string()]).is_err());
        assert!(parse_args(["--screen".to_string()]).is_err());
        assert!(parse_args(["--core".to_string()]).is_err());
    }

    #[test]
    fn the_page_is_the_legacy_template_with_the_legacy_css() {
        let p = page();
        assert_eq!(p.template, STATUS_HTML);
        assert_eq!(p.stylesheets.last().map(String::as_str), Some(STATUS_PANELS_CSS));
        let html = p.html();
        for id in ["status-view", "overlay-status", "net-status", "server-status", "minimap", "board-waiting", "point-toggle", "toast-root"] {
            assert!(html.contains(&format!("id=\"{id}\"")), "{id}");
        }
    }
}
