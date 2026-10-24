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
use crate::live_ui::LivePanels;
use crate::panels::{self, BannerInput, PanelState};
use crate::scene::StatusScene;
use skylens_shared::proto::model::v1 as model;
use skylens_viewer::camera::{SyncEvent, SyncState};
use skylens_viewer::link::VIEWER_PATH;
use skylens_viewer::protocol::{Body, GpuLossAlert};
use skylens_viewer::widgets::{CardKind, LoadingScreen, ToastKind, Toasts};
use skylens_viewer::widgets::detection_card::detection_label;
use skylens_viewer::{CoreLink, Dom, LinkEvent, Page, PollCx, ScreenApp, ScreenDocument, SiteFrame, UiInput, View3d, q};
use std::cell::RefCell;
use std::rc::Rc;

/// The legacy template and the board's own stylesheet, both unmodified.
pub const STATUS_HTML: &str = include_str!("ui/status.html");
pub const STATUS_PANELS_CSS: &str = include_str!("ui/status-panels.css");
/// Toolbar, monitor, settings panel, delay pattern and minimap of the live reconstruction.
pub const LIVE_PANELS_CSS: &str = include_str!("ui/live-panels.css");

pub const LOADING_TITLE: &str = "현황판 · 3D 복원 데이터 로딩";
pub const WINDOW_TITLE: &str = "SkyLens · 현황판 - 실시간 3D 복원";
/// Where a board with no `--core` goes first: the bridge (Raspberry Pi) on its TLS board port,
/// which relays `/viewer` to the Daejeon core. Its self-signed certificate is pinned in
/// `skylens_viewer::link::PINNED_CERTS`.
pub const DEFAULT_CORE: &str = "wss://175.126.98.44:8443/viewer";
/// Where it goes when the bridge does not answer: a core on this machine (the demo, or
/// `adb reverse tcp:8080 tcp:8080` on an emulator).
pub const LOCAL_CORE: &str = "ws://127.0.0.1:8080/viewer";
/// The loading overlay's title while the core is not reachable and the link keeps retrying.
pub const WAITING_TITLE: &str = "코어 연결 대기 중";

/// The loading overlay's detail line while waiting for the core: the address and the attempt
/// the link is on (`attempts` as in [`skylens_viewer::LinkStatus::attempts`]).
pub fn waiting_text(core_url: &str, attempts: u32) -> String {
    if attempts == 0 { core_url.to_string() } else { format!("{core_url} · 시도 {attempts}회") }
}

/// The page: legacy template plus the shared CSS and `status-panels.css`.
pub fn page() -> Page {
    Page::legacy(STATUS_HTML, &[STATUS_PANELS_CSS, LIVE_PANELS_CSS])
}

/// How the board starts.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// `ws://host:port/viewer`.
    pub core_url: String,
    /// Tried when `core_url` does not answer. Only the default has one: an explicit core is the
    /// only one tried.
    pub fallback_core: Option<String>,
    /// The point panel starts open (legacy `?points=off` starts it closed).
    pub points_on: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { core_url: DEFAULT_CORE.to_string(), fallback_core: Some(LOCAL_CORE.to_string()), points_on: true }
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
            "--core" => o.set_core(&value("--core")?),
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

impl Options {
    /// One explicit core (`--core` or a setting), normalised like `--core`, with no fallback.
    pub fn set_core(&mut self, core: &str) {
        self.core_url = skylens_viewer::link::normalize_core_url(core, VIEWER_PATH);
        self.fallback_core = None;
    }

    /// The addresses the link tries, in order.
    pub fn core_urls(&self) -> Vec<String> {
        std::iter::once(self.core_url.clone()).chain(self.fallback_core.clone()).collect()
    }

    /// For the log line: `a` or `a -> b`.
    pub fn core_label(&self) -> String {
        self.core_urls().join(" -> ")
    }
}

pub const USAGE: &str = "usage: skylens-client [--core ws://host:8080/viewer] [--points on|off]";

/// Options from loose settings that may be missing (Android has no command line and reads them
/// from system properties or files, see README). `core` is normalised like `--core`; `points`
/// starts the point panel closed only when it is `off`. Blank values count as missing.
pub fn options_from_settings(core: Option<&str>, points: Option<&str>) -> Options {
    let mut o = Options::default();
    if let Some(c) = core.map(str::trim).filter(|c| !c.is_empty()) {
        o.set_core(c);
    }
    if let Some(p) = points.map(str::trim).filter(|p| !p.is_empty()) {
        o.points_on = !p.eq_ignore_ascii_case("off");
    }
    o
}

pub struct StatusApp {
    link: CoreLink,
    board: Rc<RefCell<Board>>,
    scene: Rc<RefCell<StatusScene>>,
    inset: Option<Rc<RefCell<InsetScene>>>,
    loading: Option<LoadingScreen>,
    toasts: Toasts,
    card: Option<Card>,
    /// The delay pattern's markup as last written.
    delay: String,
    /// The point panel's inline placement last written (outer None: not written yet).
    inset_place: Option<Option<[f64; 4]>>,
    site_seen: bool,
    last_poll_ms: Option<f64>,
    gpu_alert: GpuLossAlert,
    /// The panels of the live point stream (the log itself is `Board::live`).
    pub live: LivePanels,
}

impl StatusApp {
    pub fn new(opts: &Options) -> StatusApp {
        let board = Rc::new(RefCell::new(Board::new(opts.points_on)));
        let settings = board.borrow().settings.clone();
        StatusApp {
            link: CoreLink::start_any(opts.core_urls()),
            scene: Rc::new(RefCell::new(StatusScene::new(board.clone()))),
            board,
            inset: None,
            loading: None,
            toasts: Toasts::new(),
            card: None,
            delay: String::new(),
            inset_place: None,
            gpu_alert: GpuLossAlert::default(),
            site_seen: false,
            last_poll_ms: None,
            live: LivePanels::new(settings),
        }
    }

    /// The board as a Blitz document.
    pub fn document(opts: &Options) -> ScreenDocument<StatusApp> {
        ScreenDocument::new(&page(), StatusApp::new(opts))
    }

    // ---- link ---------------------------------------------------------------------------------

    fn on_event(&mut self, ev: LinkEvent, dom: &mut Dom<'_>, now: f64) {
        match ev {
            LinkEvent::Invalid(_) => {}
            LinkEvent::Connecting => self.show_waiting(dom, false),
            LinkEvent::Connected => {
                self.board.borrow_mut().reconnected();
                if let Some(l) = &mut self.loading {
                    l.progress(dom, 60.0, None);
                }
            }
            // Not a failure: the link keeps retrying (the core may simply not be up yet), and the
            // overlay turns back into the loading bar on the next connect.
            LinkEvent::Disconnected(_) => self.show_waiting(dom, true),
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
                    Body::ServerStatus(s) => {
                        if let Some(msg) = self.gpu_alert.update(&s) {
                            self.toasts.show(dom, &msg, ToastKind::Danger, now);
                        }
                        b.core_status = Some(s);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Puts the loading overlay into its waiting state (`start`), or refreshes it when it is
    /// already waiting. Once the site has arrived the overlay is gone and the banner and the
    /// server panel report the link instead.
    fn show_waiting(&mut self, dom: &mut Dom<'_>, start: bool) {
        if self.site_seen {
            return;
        }
        let Some(l) = &mut self.loading else { return };
        if start || l.is_waiting() {
            l.wait(dom, WAITING_TITLE, &waiting_text(&self.link.current_url(), self.link.status.attempts));
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
        // The panel is the board's one link indicator: its title is the wording the top-right
        // badge used to show (`코어 스트림 수신 중`, `코어 연결됨 · 데이터 대기`, ...).
        dom.set_text(q(".server-status__text"), status.badge_text());
        let core = b.core_status.as_ref();
        let chunks = core.map_or(0, |c| c.chunks).max(b.local_chunks);
        let detections = core.map_or(0, |c| c.detections).max(b.local_detections);
        let details = dom.find_all(".server-status__detail");
        if let [counts, line, ..] = details.as_slice() {
            dom.set_text(*counts, &panels::counts_line(chunks, detections, core.and_then(|c| c.latency_ms)));
            let mission = b.mission.as_ref().map(|m| (m.message.as_str(), m.drones_online));
            dom.set_text(*line, &panels::link_line(state, mission));
        }
        // Delay pattern: the live stream's zones when there is one, else the core's segments.
        let grid = if b.live.entries().is_empty() {
            let segments: Vec<(u32, u32)> =
                core.map(|c| c.segments.iter().map(|s| (s.index, panels::level_number(s.level()))).collect()).unwrap_or_default();
            crate::panels_live::delay_grid_segments(&segments)
        } else {
            crate::panels_live::delay_grid_live(b.live.entries())
        };
        drop(b);
        let markup = crate::panels_live::delay_markup(&grid);
        if markup != self.delay {
            dom.replace_children(q(".server-status__ladder"), &markup);
            self.delay = markup;
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
            (b.point_layer_on, delivered, refined, b.total_points(), !b.point_segments().is_empty(), b.height_range())
        };
        // The panel is there from the start (empty until the first point chunk) and only folds.
        if self.inset.is_none() {
            let added = dom.append_html(q("section.pane"), &panels::inset_markup(on));
            if let Some(view) = added.first().and_then(|root| dom.find_in(*root, ".point-inset__view")) {
                let inset = Rc::new(RefCell::new(InsetScene::new(self.board.clone())));
                dom.mount_widget(view, View3d::boxed(inset.clone()));
                self.inset = Some(inset);
            }
        }
        if self.inset.is_some() {
            dom.set_text(q(".point-inset__meta"), &panels::point_meta(on, delivered, refined, total));
            dom.toggle_class("point-inset", "is-open", on);
            dom.set_text("point-inset-fold", panels::fold_label(on));
            dom.set_attr("point-inset-fold", "aria-expanded", if on { "true" } else { "false" });
            if total > 0 && has_segments {
                dom.set_style(q(".point-inset__empty"), "display", "none");
            } else if dom.attr(q(".point-inset__empty"), "style").is_some_and(|s| s.contains("display")) {
                dom.remove_style(q(".point-inset__empty"), "display");
            }
            let (high, low) = panels::legend_ticks(range);
            dom.set_text(q(".point-inset__tick--high"), &high);
            dom.set_text(q(".point-inset__tick--low"), &low);
            self.place_inset(dom);
        }
    }

    /// On a short viewport (a phone on its side) the point panel moves beside the server panel
    /// instead of under it; see [`panels::inset_placement`]. Inline styles, so the stylesheet keeps
    /// one placement.
    fn place_inset(&mut self, dom: &mut Dom<'_>) {
        let place = {
            let vp = dom.doc_ref().viewport();
            let scale = vp.scale_f64();
            let scale = if scale > 0.0 { scale } else { 1.0 };
            panels::inset_placement(f64::from(vp.window_size.0) / scale, f64::from(vp.window_size.1) / scale)
        };
        if self.inset_place == Some(place) {
            return;
        }
        self.inset_place = Some(place);
        match place {
            Some([left, top, width, height]) => {
                dom.set_style("point-inset", "left", &format!("{left}px"));
                dom.set_style("point-inset", "top", &format!("{top}px"));
                dom.set_style("point-inset", "bottom", "auto");
                dom.set_style("point-inset", "width", &format!("{width}px"));
                dom.set_style("point-inset", "height", &format!("{height}px"));
            }
            None => {
                for prop in ["left", "top", "bottom", "width", "height"] {
                    dom.remove_style("point-inset", prop);
                }
            }
        }
    }
}

impl ScreenApp for StatusApp {
    fn mount(&mut self, dom: &mut Dom<'_>) {
        dom.mount_widget("status-view", View3d::boxed(self.scene.clone()));
        self.card = Card::bind(dom);
        self.live.mount(dom);
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

        self.render_server_panel(dom);
        self.render_banner(dom);
        self.render_points(dom);
        self.live.poll(dom, &self.board.borrow().live, now, dt);

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
        if self.live.input(input, dom) {
            return true;
        }
        if Card::is_confirm(input) {
            self.confirm(dom);
            return true;
        }
        if input.is_click() && input.closest_id("point-inset-fold").is_some() {
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
    fn arguments_default_to_the_bridge_then_the_local_core() {
        let o = parse_args(Vec::<String>::new()).unwrap();
        assert_eq!(o, Options::default());
        assert_eq!(o.core_urls(), ["wss://175.126.98.44:8443/viewer", "ws://127.0.0.1:8080/viewer"]);
        assert_eq!(o.core_label(), "wss://175.126.98.44:8443/viewer -> ws://127.0.0.1:8080/viewer");
        let o = parse_args(["--core".to_string(), "core.local:8080".to_string()]).unwrap();
        assert_eq!(o.core_url, "ws://core.local:8080/viewer");
        assert_eq!(o.core_urls(), ["ws://core.local:8080/viewer"]);
        let o = parse_args(["--core=http://10.0.0.2:8080".to_string(), "--points".to_string(), "off".to_string()]).unwrap();
        assert_eq!(o.core_url, "ws://10.0.0.2:8080/viewer");
        assert!(!o.points_on);
        assert!(parse_args(["--points".to_string(), "maybe".to_string()]).is_err());
        assert!(parse_args(["--screen".to_string()]).is_err());
        assert!(parse_args(["--core".to_string()]).is_err());
    }

    #[test]
    fn waiting_text_names_the_core_and_the_attempt() {
        assert_eq!(waiting_text(LOCAL_CORE, 0), "ws://127.0.0.1:8080/viewer");
        assert_eq!(waiting_text("ws://10.0.0.2:8080/viewer", 3), "ws://10.0.0.2:8080/viewer · 시도 3회");
    }

    #[test]
    fn settings_fall_back_to_the_defaults() {
        assert_eq!(options_from_settings(None, None), Options::default());
        assert_eq!(options_from_settings(Some("  "), Some("")), Options::default());
        let o = options_from_settings(Some(" 192.168.0.10:8080\n"), Some("OFF"));
        assert_eq!(o.core_url, "ws://192.168.0.10:8080/viewer");
        assert_eq!(o.fallback_core, None);
        assert!(!o.points_on);
        let o = options_from_settings(Some("wss://bridge.example/viewer"), Some("on"));
        assert_eq!(o.core_url, "wss://bridge.example/viewer");
        assert!(o.points_on);
    }

    #[test]
    fn the_page_is_the_legacy_template_with_the_legacy_css() {
        let p = page();
        assert_eq!(p.template, STATUS_HTML);
        assert_eq!(p.stylesheets.last().map(String::as_str), Some(LIVE_PANELS_CSS));
        assert!(p.stylesheets.iter().any(|c| c == STATUS_PANELS_CSS));
        let html = p.html();
        for id in ["status-view", "overlay-status", "board-toolbar", "board-settings-btn", "server-status", "board-waiting", "toast-root"] {
            assert!(html.contains(&format!("id=\"{id}\"")), "{id}");
        }
    }
}
