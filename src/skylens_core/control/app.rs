//! The tower screen (legacy `control.ts`): the core link, the panels, the planner, the bridge
//! side effects and the keys, wired onto the legacy template.

use crate::bridge::{self, Download, Member, MissionResult};
use crate::fleet::FleetDrone;
use crate::http::{self, Http, Outcome, Request};
use crate::labels::{self, Station};
use crate::manual::{ManualInput, Tick};
use crate::modals::{self, CloudModal, RouteModal};
use crate::panels::{MissionPanel, SettingsPanel, TelemetryPanel};
use crate::planner::{self, Planner, PlannerEvent, PlannerWidget, Waypoint};
use crate::settings::{DisplayMode, Settings};
use crate::slider::{Range, RangeWidget};
use crate::storage;
use crate::tower::{TowerScene, World};
use crate::video::{Feed, Fix, VideoConfig, VideoPanel};
use skylens_viewer::link::{LinkEvent, LinkState, LinkStatus};
use skylens_viewer::protocol::{self, Body};
use skylens_viewer::shell::{InputKind, UiInput};
use skylens_viewer::widgets::{LoadingScreen, ToastKind, Toasts};
use skylens_viewer::{Code, CoreLink, Dom, NodeId, PollCx, ScreenApp, SiteFrame, View3d, q};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// The legacy template and the screen's own stylesheets (unchanged files).
pub const CONTROL_HTML: &str = include_str!("ui/control.html");
pub const CONTROL_CSS: &str = include_str!("ui/control.css");
pub const PANELS_CSS: &str = include_str!("ui/panels.css");

pub const LOADING_TITLE: &str = "관제탑 · 실지형 장면 로딩";
/// Telemetry rows re-render at 5 Hz.
const PANEL_INTERVAL_MS: f64 = 200.0;
/// Safety net: if the core stops sending base map progress altogether, stop waiting after this
/// long without a new tile so the operator is never locked out.
const LOADING_IDLE_MS: f64 = 60_000.0;
/// Planner tile request timeout.
const TILE_TIMEOUT_MS: u64 = 15_000;

/// Run options.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// `ws(s)://host:port/control`.
    pub core_url: String,
    /// `?display=` / `--display`.
    pub display_override: Option<DisplayMode>,
    /// Mission bridge base URL.
    pub bridge: String,
    /// Live video player (native); None shows the standby pattern only.
    pub video: Option<VideoConfig>,
}

/// The legacy template with the legacy stylesheets in the legacy import order
/// (`control.ts`: style.css, control/control.css, ui/panels.css).
pub fn page() -> skylens_viewer::Page {
    skylens_viewer::Page::legacy(CONTROL_HTML, &[CONTROL_CSS, PANELS_CSS])
}

/// Background request tags.
#[derive(Debug, Clone)]
pub enum HttpTag {
    Site,
    Tile { token: u64, z: u32, tx: u32, ty: u32 },
    Health,
    Mission,
    MissionDownload(MissionResult),
    Formation,
    FormationDownload { members: Vec<Member>, index: usize },
    CloudList,
    CloudDelete { name: String, button: Option<NodeId> },
}

/// The legacy tower kept "관제탑 · 실지형 장면 로딩" up until the whole scene had loaded and
/// then showed it at once. The core streams the base map cell by cell, so the loading screen
/// follows the core's progress in `Site` and lifts only when every base map chunk is here.
#[derive(Debug, Default)]
struct Loading {
    site_ms: Option<f64>,
    last_tile_ms: f64,
    finished: bool,
    /// From the latest `Site`.
    ready: bool,
    cells_total: u32,
    chunks_total: u32,
}

pub struct ControlApp {
    opts: Options,
    link: CoreLink,
    origin: String,
    world: Rc<RefCell<World>>,
    scene: Rc<RefCell<TowerScene>>,
    planner: Rc<RefCell<Planner>>,
    range: Rc<RefCell<Range>>,
    http: Http<HttpTag>,
    settings: Settings,
    toasts: Toasts,
    loading_screen: Option<LoadingScreen>,
    loading: Loading,
    mission: Option<MissionPanel>,
    telemetry: Option<TelemetryPanel>,
    settings_panel: Option<SettingsPanel>,
    route: Option<RouteModal>,
    cloud: Option<CloudModal>,
    video: Option<VideoPanel>,
    toolbar: Vec<NodeId>,
    manual: ManualInput,
    last_state: Option<LinkState>,
    shadow: LinkStatus,
    next_panel_ms: f64,
    cam_drone: Option<u32>,
    link_seen: HashMap<String, String>,
    site_requested: bool,
    planner_centred: bool,
}

impl ControlApp {
    pub fn new(opts: Options) -> ControlApp {
        let settings = Settings::load(opts.display_override);
        let world = Rc::new(RefCell::new(World::new(settings.display)));
        let scene = Rc::new(RefCell::new(TowerScene::new(world.clone())));
        let stored = planner::parse_center(storage::get(storage::MAP_CENTER_KEY).as_deref());
        let planner = Rc::new(RefCell::new(Planner::new(stored.unwrap_or(Waypoint { lat: 0.0, lon: 0.0, alt: 0.0 }), stored.is_some())));
        let origin = http::http_origin(&opts.core_url);
        ControlApp {
            link: CoreLink::start(opts.core_url.clone()),
            origin,
            world,
            scene,
            range: Rc::new(RefCell::new(Range::new(20.0, 200.0, 5.0, 60.0))),
            planner_centred: stored.is_some(),
            planner,
            http: Http::new(),
            settings,
            toasts: Toasts::new(),
            loading_screen: None,
            loading: Loading::default(),
            mission: None,
            telemetry: None,
            settings_panel: None,
            route: None,
            cloud: None,
            video: None,
            toolbar: Vec::new(),
            manual: ManualInput::new(),
            last_state: None,
            shadow: LinkStatus::default(),
            next_panel_ms: 0.0,
            cam_drone: None,
            link_seen: HashMap::new(),
            site_requested: false,
            opts,
        }
    }

    fn toast(&mut self, dom: &mut Dom<'_>, message: &str, kind: ToastKind) {
        self.toasts.show(dom, message, kind, skylens_viewer::now_ms());
    }

    // ---- core link ------------------------------------------------------------------------

    fn handle_link(&mut self, dom: &mut Dom<'_>, now: f64) {
        for ev in self.link.drain() {
            // The link's own status already folded the whole batch in; this one follows event by
            // event so no transition is skipped.
            self.shadow.observe(&ev, now);
            match ev {
                LinkEvent::Connected => {
                    // The core replays what it holds from the start.
                    let mut w = self.world.borrow_mut();
                    w.store.clear();
                    w.terrain_rev += 1;
                    w.buildings_rev += 1;
                }
                LinkEvent::Message(env) => {
                    if let Some(body) = env.body {
                        self.handle_body(dom, body, now);
                    }
                }
                LinkEvent::Invalid(e) => skylens_viewer::log(&format!("[control] dropped frame: {e}")),
                LinkEvent::Connecting | LinkEvent::Disconnected(_) => {}
            }
            let state = self.shadow.state;
            if self.last_state != Some(state) {
                self.last_state = Some(state);
                let detail = self.shadow.detail();
                if let Some(m) = self.mission.as_mut() {
                    m.set_link(dom, state, detail, now);
                }
                if state == LinkState::Disconnected {
                    // Telemetry stops meaning anything once the link drops: no ghosts.
                    self.world.borrow_mut().fleet.clear();
                    if let Some(t) = self.telemetry.as_mut() {
                        t.render(dom, &[], None);
                    }
                    if let Some(v) = self.video.as_mut() {
                        v.set_feed(dom, None);
                        v.set_telemetry(None);
                    }
                    if self.loading.site_ms.is_none() && !self.loading.finished {
                        self.loading.finished = true;
                        if let Some(l) = self.loading_screen.as_mut() {
                            l.fail(dom, detail, now);
                        }
                    }
                }
            }
        }
    }

    fn handle_body(&mut self, dom: &mut Dom<'_>, body: Body, now: f64) {
        match body {
            Body::Site(site) => self.on_site(dom, &site, now),
            Body::Telemetry(t) => {
                self.world.borrow_mut().ingest(&t, now);
                if let (Some(v), Some(g)) = (self.video.as_mut(), t.gps.as_ref()) {
                    v.set_telemetry(Some(Fix { drone_id: t.drone_id, lat: g.lat, lon: g.lon, alt: g.alt }));
                }
            }
            Body::CameraFeed(f) => {
                let feed = Feed { drone_id: f.drone_id, station: Station::from_proto(f.station).unwrap_or(Station::Center), whep_url: f.whep_url };
                if let Some(v) = self.video.as_mut() {
                    v.set_feed(dom, Some(feed));
                }
            }
            Body::AssignRoute(r) => {
                // The core echoes the route in force, also to a tower that joins later.
                let mut w = self.world.borrow_mut();
                w.route = (r.waypoints.len() >= 2).then(|| (r.waypoints.clone(), r.r#loop));
                w.route_rev += 1;
            }
            Body::PlannedRoute(r) => {
                // A route prepared before the session (the demo's flight line): it goes into an
                // empty planner and the operator only presses assign. Nothing is sent.
                let before = self.planner.borrow().center;
                if self.planner.borrow_mut().load_planned(&r.waypoints, r.r#loop) {
                    skylens_viewer::log(&format!("[control] planned route loaded into the planner ({} waypoints)", r.waypoints.len()));
                    if let Some(i) = self.route.as_ref().and_then(|m| m.loop_input) {
                        modals::set_checked(dom, i, r.r#loop);
                    }
                    self.render_waypoints(dom);
                    let moved = self.planner.borrow().center != before;
                    if moved && self.route.as_ref().is_some_and(|m| m.is_open(dom)) {
                        self.load_tiles();
                    }
                }
            }
            Body::MissionStatus(m) => {
                if let Some(p) = self.mission.as_mut() {
                    p.set_mission(dom, &m, now);
                }
            }
            Body::Chunk(c) => {
                let applied = self.world.borrow_mut().store.apply(&c);
                let mut w = self.world.borrow_mut();
                match applied.change {
                    skylens_viewer::Change::Terrain { .. } => {
                        w.terrain_rev += 1;
                        self.loading.last_tile_ms = now;
                    }
                    skylens_viewer::Change::Buildings { .. } => w.buildings_rev += 1,
                    skylens_viewer::Change::Invalid(e) => skylens_viewer::log(&format!("[control] dropped chunk: {e}")),
                    _ => {}
                }
            }
            Body::LinkStatus(l) => {
                // Only transitions are worth a line.
                let line = format!("{}", if l.connected { "up" } else { "down" });
                if self.link_seen.get(&l.hop) != Some(&line) {
                    skylens_viewer::log(&format!("[control] link {}: {line}", l.hop));
                    self.link_seen.insert(l.hop.clone(), line);
                }
            }
            _ => {}
        }
    }

    fn on_site(&mut self, dom: &mut Dom<'_>, site: &skylens_shared::proto::viewer::v1::Site, now: f64) {
        let Some(frame) = SiteFrame::from_site(site) else { return };
        {
            let mut w = self.world.borrow_mut();
            if w.site.as_ref().is_some_and(|s| s.anchor != frame.anchor) {
                // A different origin: nothing stored is placed right any more.
                w.store.clear();
                w.fleet.clear();
                w.terrain_rev += 1;
                w.buildings_rev += 1;
                w.route_rev += 1;
            }
            w.site = Some(frame.clone());
        }
        if self.loading.site_ms.is_none() {
            self.loading.site_ms = Some(now);
            self.loading.last_tile_ms = now;
        }
        self.loading.ready = site.base_map_ready;
        self.loading.cells_total = site.base_map_cells_total;
        self.loading.chunks_total = site.base_map_chunks;
        if let Some(p) = self.settings_panel.as_mut() {
            p.show_site(dom, &frame);
        }
        {
            let mut p = self.planner.borrow_mut();
            let anchor = Waypoint { lat: frame.anchor.lat, lon: frame.anchor.lon, alt: frame.anchor.alt };
            p.anchor = Some(anchor);
            if !self.planner_centred && !p.pinned {
                p.center = anchor;
                self.planner_centred = true;
            }
            p.mark_dirty();
        }
        if !self.site_requested {
            // The core knows where the operations centre is; the planner may open there.
            self.site_requested = true;
            self.http.send(HttpTag::Site, Request::get(format!("{}/site", self.origin), 4_000));
        }
    }

    fn update_loading(&mut self, dom: &mut Dom<'_>, now: f64) {
        if let Some(l) = self.loading_screen.as_mut() {
            l.tick(dom, now);
        }
        if self.loading.finished {
            return;
        }
        let Some(site_ms) = self.loading.site_ms else { return };
        let (tiles, chunks) = {
            let w = self.world.borrow();
            let tiles = w.store.terrain_tiles().count();
            (tiles, tiles + w.store.building_sets().count())
        };
        // Progress over the terrain cells the core has sent or still owes; the last step waits
        // for the buildings too.
        let total = (self.loading.cells_total as usize).max(1);
        let pct = (tiles as f64 / total as f64 * 100.0).min(if self.loading.ready { 100.0 } else { 99.0 });
        if let Some(l) = self.loading_screen.as_mut() {
            l.progress(dom, pct, None);
        }
        let complete = self.loading.ready && chunks >= self.loading.chunks_total as usize;
        let stalled = now - self.loading.last_tile_ms.max(site_ms) > LOADING_IDLE_MS;
        if complete || stalled {
            self.loading.finished = true;
            if let Some(l) = self.loading_screen.as_mut() {
                l.done(dom, now);
            }
        }
    }

    // ---- panels ---------------------------------------------------------------------------

    fn render_fleet(&mut self, dom: &mut Dom<'_>, now: f64) {
        let (drones, active): (Vec<FleetDrone>, Option<u32>) = {
            let mut w = self.world.borrow_mut();
            w.fleet.refresh_stale(now);
            (w.fleet.drones(), w.fleet.active)
        };
        if let Some(t) = self.telemetry.as_mut() {
            t.render(dom, &drones, active);
        }
        self.sync_camera(dom);
    }

    /// Points the camera panel at the selected aircraft (legacy `syncCamera`).
    fn sync_camera(&mut self, dom: &mut Dom<'_>) {
        let target = {
            let w = self.world.borrow();
            w.fleet.active.and_then(|a| w.fleet.track(a)).map(|t| (t.info.id, t.info.station))
        };
        let Some((id, station)) = target else { return };
        if self.cam_drone == Some(id) {
            return;
        }
        self.cam_drone = Some(id);
        if let Some(v) = self.video.as_mut() {
            v.select(dom, id, station);
        }
    }

    fn selection_changed(&mut self, dom: &mut Dom<'_>, now: f64) {
        self.render_fleet(dom, now);
    }

    fn set_display(&mut self, dom: &mut Dom<'_>, mode: DisplayMode) {
        self.settings.set_display(mode);
        self.world.borrow_mut().display = mode;
        if let Some(p) = self.settings_panel.as_ref() {
            p.show_mode(dom, mode);
        }
        dom.mark_changed();
    }

    // ---- planner --------------------------------------------------------------------------

    fn load_tiles(&mut self) {
        let (token, z, tiles) = {
            let mut p = self.planner.borrow_mut();
            let token = p.new_view();
            let (z, tiles) = p.tile_plan();
            (token, z, tiles)
        };
        for (tx, ty) in tiles {
            let url = format!("{}{}", self.origin, planner::tile_path(z, tx, ty));
            self.http.send(HttpTag::Tile { token, z, tx, ty }, Request::get(url, TILE_TIMEOUT_MS));
        }
    }

    fn render_waypoints(&mut self, dom: &mut Dom<'_>) {
        let wps = self.planner.borrow().waypoints.clone();
        if let Some(r) = self.route.as_ref() {
            r.render_list(dom, &wps);
        }
        self.planner.borrow_mut().mark_dirty();
    }

    /// Absolute altitude for a point at the planner's clearance (legacy `pxToGps`).
    fn waypoint_at(&mut self, lat: f64, lon: f64, agl: f64) -> Waypoint {
        let ground = self.world.borrow_mut().ground_alt(lat, lon);
        Waypoint { lat, lon, alt: labels::js_round(ground + agl) }
    }

    fn drain_planner(&mut self, dom: &mut Dom<'_>) {
        let events: Vec<PlannerEvent> = std::mem::take(&mut self.planner.borrow_mut().events);
        for ev in events {
            match ev {
                PlannerEvent::Click { x, y } => {
                    let hit = self.planner.borrow().hit(x, y);
                    match hit {
                        Some(i) => {
                            self.planner.borrow_mut().waypoints.remove(i);
                        }
                        None => {
                            let ((lat, lon), agl) = {
                                let p = self.planner.borrow();
                                (p.px_to_latlon(x, y), p.agl)
                            };
                            let w = self.waypoint_at(lat, lon, agl);
                            self.planner.borrow_mut().waypoints.push(w);
                        }
                    }
                    self.render_waypoints(dom);
                }
                PlannerEvent::PanStarted => {
                    if let Some(r) = self.route.as_ref() {
                        r.set_panning(dom, true);
                    }
                }
                PlannerEvent::PanEnded => {
                    if let Some(r) = self.route.as_ref() {
                        r.set_panning(dom, false);
                    }
                    let c = self.planner.borrow().center;
                    storage::set(storage::MAP_CENTER_KEY, &planner::stored_center(&c));
                    self.load_tiles();
                }
            }
        }
        // The altitude slider (legacy `input` handler).
        let moved = {
            let mut r = self.range.borrow_mut();
            let changed = r.changed;
            r.changed = false;
            changed.then_some(r.value)
        };
        if let Some(agl) = moved {
            self.planner.borrow_mut().agl = agl;
            if let Some(r) = self.route.as_ref() {
                r.sync_alt_text(dom, agl);
            }
            let points: Vec<(f64, f64)> = self.planner.borrow().waypoints.iter().map(|w| (w.lat, w.lon)).collect();
            let updated: Vec<Waypoint> = points.into_iter().map(|(lat, lon)| self.waypoint_at(lat, lon, agl)).collect();
            self.planner.borrow_mut().waypoints = updated;
            self.render_waypoints(dom);
        }
    }

    fn open_route(&mut self, dom: &mut Dom<'_>) {
        // Reflect the bridge's cloud capability before the planner is used.
        self.http.send(HttpTag::Health, Request::get(format!("{}/health", self.opts.bridge), bridge::HEALTH_TIMEOUT_MS));
        if let Some(r) = self.route.as_ref() {
            r.open(dom);
        }
        self.load_tiles();
    }

    fn close_route(&mut self, dom: &mut Dom<'_>) {
        // Cancels tile loads in flight.
        self.planner.borrow_mut().token += 1;
        if let Some(r) = self.route.as_ref() {
            r.close(dom);
        }
    }

    fn assign(&mut self, dom: &mut Dom<'_>) {
        let (wps, looped, agl, formation, spacing, cloud) = {
            let p = self.planner.borrow();
            (p.waypoints.iter().map(Waypoint::gps).collect::<Vec<_>>(), p.looped, p.agl, p.formation, p.spacing_m, p.cloud)
        };
        if wps.len() < 2 {
            if let Some(r) = self.route.as_ref() {
                r.warn_too_short(dom);
            }
            return;
        }
        let drone = self.world.borrow().fleet.leader_id();
        let sent = self.link.send(protocol::assign_route(drone, &wps, looped));
        let kind = if sent { ToastKind::Info } else { ToastKind::Danger };
        self.toast(dom, &labels::route_sent_toast(sent, wps.len(), looped), kind);
        let base = bridge::normalize_base(&self.opts.bridge);
        if formation {
            let body = bridge::formation_body(&wps, looped, agl, spacing, cloud);
            self.http.send(HttpTag::Formation, Request::post_json(format!("{base}/formation"), body, bridge::formation_timeout_ms(cloud)));
        } else {
            let body = bridge::mission_body(drone, &wps, looped, agl, cloud);
            self.http.send(HttpTag::Mission, Request::post_json(format!("{base}/missions"), body, bridge::mission_timeout_ms(cloud)));
        }
        self.close_route(dom);
    }

    // ---- cloud missions -------------------------------------------------------------------

    fn refresh_cloud(&mut self, dom: &mut Dom<'_>) {
        let Some(c) = self.cloud.as_mut() else { return };
        if c.busy {
            return;
        }
        c.busy = true;
        c.set_refresh_enabled(dom, false);
        c.show_message(dom, modals::CLOUD_LOADING);
        let base = bridge::normalize_base(&self.opts.bridge);
        self.http.send(HttpTag::CloudList, Request::get(format!("{base}/cloud/missions"), bridge::LIST_TIMEOUT_MS));
    }

    fn delete_cloud(&mut self, dom: &mut Dom<'_>, button: NodeId) {
        let Some(c) = self.cloud.as_mut() else { return };
        if c.busy || dom.attr(button, "disabled").is_some() {
            return;
        }
        let Some((_, mission)) = c.mission_of(dom, button) else { return };
        c.busy = true;
        dom.set_attr(button, "disabled", "");
        let base = bridge::normalize_base(&self.opts.bridge);
        self.http.send(
            HttpTag::CloudDelete { name: mission.name.clone(), button: Some(button) },
            Request::post_json(format!("{base}/cloud/delete"), bridge::delete_body(&mission.uuid), bridge::LIST_TIMEOUT_MS),
        );
    }

    // ---- HTTP results ---------------------------------------------------------------------

    fn download(&mut self, tag: HttpTag, d: &Download) {
        self.http.send(tag, Request::get(d.url.clone(), bridge::DOWNLOAD_TIMEOUT_MS));
    }

    /// The bridge answer as JSON, or the stringified legacy error.
    fn bridge_json(&self, outcome: &Outcome) -> Result<serde_json::Value, String> {
        match outcome {
            Err(t) => Err(bridge::unreachable_error(&bridge::normalize_base(&self.opts.bridge), t)),
            Ok(r) if !r.ok() => Err(bridge::js_error(&bridge::error_message(r.status, &r.body))),
            Ok(r) => r.json().ok_or_else(|| "SyntaxError: Unexpected end of JSON input".to_string()),
        }
    }

    /// A downloaded KMZ (legacy `download()`), or the stringified error.
    fn downloaded(outcome: &Outcome, file_name: &str) -> Result<(), String> {
        match outcome {
            Err(t) => Err(t.clone()),
            Ok(r) if !r.ok() => Err(bridge::js_error(&bridge::error_message(r.status, &r.body))),
            Ok(r) => crate::download::save(file_name, &r.body).map_err(|e| bridge::js_error(&e)),
        }
    }

    fn handle_http(&mut self, dom: &mut Dom<'_>) {
        for (tag, outcome) in self.http.drain() {
            match tag {
                HttpTag::Site => {
                    let gps = outcome.ok().filter(|r| r.ok()).and_then(|r| r.json()).and_then(|v| {
                        let g = v.get("gps")?;
                        Some((g.get("lat")?.as_f64()?, g.get("lon")?.as_f64()?, g.get("alt").and_then(|a| a.as_f64()).unwrap_or(0.0)))
                    });
                    if let Some((lat, lon, alt)) = gps {
                        let pinned = self.planner.borrow().pinned;
                        if !pinned {
                            let mut p = self.planner.borrow_mut();
                            p.center = Waypoint { lat, lon, alt };
                            p.mark_dirty();
                            self.planner_centred = true;
                        }
                        let open = self.route.as_ref().is_some_and(|r| r.is_open(dom));
                        if !pinned && open {
                            self.load_tiles();
                        }
                    }
                }
                HttpTag::Tile { token, z, tx, ty } => {
                    let mut p = self.planner.borrow_mut();
                    if token != p.token {
                        continue;
                    }
                    if let Ok(r) = outcome
                        && r.ok()
                        && let Some(image) = planner::decode_tile(&r.content_type, &r.body)
                    {
                        p.tiles.push(planner::MapTile { z, tx, ty, image });
                        p.mark_dirty();
                    }
                    // Otherwise (no VWorld key: 503) the map keeps its grid.
                }
                HttpTag::Health => {
                    let available = bridge::cloud_configured(outcome.ok().filter(|r| r.ok()).and_then(|r| r.json()).as_ref());
                    if !available {
                        self.planner.borrow_mut().cloud = false;
                    }
                    if let Some(r) = self.route.as_ref() {
                        r.set_cloud_available(dom, available);
                    }
                }
                HttpTag::Mission => match self.bridge_json(&outcome).and_then(|v| bridge::parse_mission(&v)) {
                    Err(e) => self.toast(dom, &format!("DJI 임무 생성 실패 · {e}"), ToastKind::Danger),
                    Ok(m) => {
                        for w in &m.warnings {
                            skylens_viewer::log(&format!("[mission] {w}"));
                        }
                        if m.installed {
                            self.toast(dom, &bridge::mission_toast(&m), ToastKind::Info);
                        } else {
                            let d = m.download.clone();
                            self.download(HttpTag::MissionDownload(m), &d);
                        }
                    }
                },
                HttpTag::MissionDownload(m) => match Self::downloaded(&outcome, &m.download.file_name) {
                    Err(e) => self.toast(dom, &format!("DJI 임무 생성 실패 · {e}"), ToastKind::Danger),
                    Ok(()) => self.toast(dom, &bridge::mission_toast(&m), ToastKind::Info),
                },
                HttpTag::Formation => match self.bridge_json(&outcome).and_then(|v| bridge::parse_formation(&v)) {
                    Err(e) => self.toast(dom, &format!("편대 KMZ 생성 실패 · {e}"), ToastKind::Danger),
                    Ok(members) => {
                        // Sequential downloads, in station order as the bridge returned them.
                        let d = members[0].download.clone();
                        self.download(HttpTag::FormationDownload { members, index: 0 }, &d);
                    }
                },
                HttpTag::FormationDownload { members, index } => {
                    let name = members[index].download.file_name.clone();
                    match Self::downloaded(&outcome, &name) {
                        Err(e) => self.toast(dom, &format!("편대 KMZ 생성 실패 · {e}"), ToastKind::Danger),
                        Ok(()) if index + 1 < members.len() => {
                            let d = members[index + 1].download.clone();
                            self.download(HttpTag::FormationDownload { members, index: index + 1 }, &d);
                        }
                        Ok(()) => {
                            for m in &members {
                                for w in &m.warnings {
                                    skylens_viewer::log(&format!("[mission {}] {w}", m.station));
                                }
                            }
                            self.toast(dom, &bridge::formation_toast(&members), ToastKind::Info);
                        }
                    }
                }
                HttpTag::CloudList => {
                    let result = self.bridge_json(&outcome).map(|v| bridge::parse_cloud_missions(&v));
                    let Some(c) = self.cloud.as_mut() else { continue };
                    match result {
                        Ok(list) => c.show_missions(dom, list),
                        Err(e) => {
                            c.show_message(dom, modals::CLOUD_FAILED);
                            self.toasts.show(dom, &format!("클라우드 미션 조회 실패 · {e}"), ToastKind::Danger, skylens_viewer::now_ms());
                        }
                    }
                    if let Some(c) = self.cloud.as_mut() {
                        c.busy = false;
                        c.set_refresh_enabled(dom, true);
                    }
                }
                HttpTag::CloudDelete { name, button } => {
                    let result = self.bridge_json(&outcome).map(|v| bridge::parse_removed(&v));
                    if let Some(c) = self.cloud.as_mut() {
                        c.busy = false;
                    }
                    match result {
                        Ok(true) => {
                            self.toast(dom, &format!("삭제됨 · {name}"), ToastKind::Info);
                            self.refresh_cloud(dom);
                        }
                        Ok(false) => self.toast(dom, &format!("삭제 요청됨(미확인) · {name}"), ToastKind::Danger),
                        Err(e) => {
                            self.toast(dom, &format!("삭제 실패 · {e}"), ToastKind::Danger);
                            if let Some(b) = button {
                                dom.remove_attr(b, "disabled");
                            }
                        }
                    }
                }
            }
        }
    }

    // ---- input ----------------------------------------------------------------------------

    fn on_click(&mut self, input: &UiInput, dom: &mut Dom<'_>, now: f64) {
        let target = input.target().map(|t| t.node);
        let on_path = |node: Option<NodeId>| node.is_some_and(|n| input.path.iter().any(|e| e.node == n));

        // Toolbar.
        if let Some(btn) = input.closest_class("control-toolbar__btn") {
            match self.toolbar.iter().position(|n| *n == btn.node) {
                Some(0) => self.open_route(dom),
                Some(1) => {
                    if let Some(c) = self.cloud.as_ref() {
                        c.open(dom);
                    }
                    self.refresh_cloud(dom);
                }
                Some(2) => {
                    if let Some(s) = self.settings_panel.as_ref() {
                        s.toggle(dom);
                    }
                }
                _ => {}
            }
            return;
        }

        // Settings.
        let settings_overlay = self.settings_panel.as_ref().and_then(|s| s.overlay);
        if target.is_some() && target == settings_overlay || input.closest_class("settings-panel__close").is_some() {
            if let Some(s) = self.settings_panel.as_ref() {
                s.close(dom);
            }
            return;
        }
        if let Some(opt) = input.closest_class("settings-option") {
            let mode = self.settings_panel.as_ref().and_then(|s| s.mode_of(opt.node));
            if let Some(mode) = mode
                && dom.attr(opt.node, "disabled").is_none()
            {
                self.set_display(dom, mode);
            }
            return;
        }

        // Route planner.
        let (route_overlay, route_close, clear, assign, list) = match self.route.as_ref() {
            Some(r) => (r.overlay, r.close, r.clear, r.assign, r.list),
            None => (None, None, None, None, None),
        };
        if route_overlay.is_some() && target == route_overlay || on_path(route_close) {
            self.close_route(dom);
            return;
        }
        if on_path(route_overlay) {
            if let Some(span_btn) = input.closest_class("route-modal__span") {
                if let Some(span) = self.route.as_ref().and_then(|r| r.span_of(span_btn.node)) {
                    self.planner.borrow_mut().span_m = span;
                    if let Some(r) = self.route.as_ref() {
                        r.sync_spans(dom, span);
                    }
                    self.load_tiles();
                }
            } else if on_path(clear) {
                self.planner.borrow_mut().waypoints.clear();
                self.render_waypoints(dom);
            } else if on_path(assign) {
                self.assign(dom);
            } else if let Some(rm) = input.closest_class("route-modal__remove")
                && on_path(list)
            {
                let idx = self.route.as_ref().and_then(|r| r.remove_index(dom, rm.node));
                if let Some(i) = idx {
                    let mut p = self.planner.borrow_mut();
                    if i < p.waypoints.len() {
                        p.waypoints.remove(i);
                    }
                }
                self.render_waypoints(dom);
            }
            return;
        }

        // Cloud missions.
        let (cloud_overlay, cloud_close, refresh) = match self.cloud.as_ref() {
            Some(c) => (c.overlay, c.close, c.refresh),
            None => (None, None, None),
        };
        if cloud_overlay.is_some() && target == cloud_overlay || on_path(cloud_close) {
            if let Some(c) = self.cloud.as_ref() {
                c.close(dom);
            }
            return;
        }
        if on_path(cloud_overlay) {
            if on_path(refresh) {
                if refresh.is_some_and(|b| dom.attr(b, "disabled").is_none()) {
                    self.refresh_cloud(dom);
                }
            } else if let Some(rm) = input.closest_class("route-modal__remove") {
                self.delete_cloud(dom, rm.node);
            }
            return;
        }

        // Fleet rows.
        if let Some(row) = input.closest_class("telemetry-row") {
            self.select_row(dom, row.node, now);
        }
    }

    fn select_row(&mut self, dom: &mut Dom<'_>, row: NodeId, now: f64) {
        let id = self.telemetry.as_ref().and_then(|t| t.row_drone(row));
        if let Some(id) = id {
            self.world.borrow_mut().fleet.select_by_operator(id);
            self.selection_changed(dom, now);
        }
    }

    fn on_key_down(&mut self, input: &UiInput, dom: &mut Dom<'_>, now: f64) {
        let Some(code) = input.code() else { return };
        self.manual.key_down(code);
        let switched = {
            let mut w = self.world.borrow_mut();
            match code {
                Code::Digit1 => w.fleet.switch_to(1),
                Code::Digit2 => w.fleet.switch_to(2),
                Code::Digit3 => w.fleet.switch_to(3),
                Code::Tab => w.fleet.cycle(),
                _ => false,
            }
        };
        if switched {
            self.selection_changed(dom, now);
        }
        if code == Code::Escape
            && let Some(s) = self.settings_panel.as_ref()
        {
            s.close(dom);
        }
        if matches!(code, Code::Enter | Code::Space)
            && let Some(row) = input.closest_class("telemetry-row")
        {
            self.select_row(dom, row.node, now);
        }
    }

    fn on_input(&mut self, input: &UiInput, dom: &mut Dom<'_>) {
        let Some(target) = input.target().map(|t| t.node) else { return };
        let Some(r) = self.route.as_ref() else { return };
        let on = input.value.as_deref() == Some("true");
        if Some(target) == r.loop_input {
            self.planner.borrow_mut().looped = on;
        } else if Some(target) == r.formation_input {
            self.planner.borrow_mut().formation = on;
            r.set_spacing_enabled(dom, on);
        } else if Some(target) == r.cloud_input {
            self.planner.borrow_mut().cloud = on;
        } else if Some(target) == r.spacing_input {
            let v = input.value.as_deref().map(str::trim).and_then(|s| s.parse::<f64>().ok());
            if let Some(v) = v.filter(|v| v.is_finite() && *v > 0.0) {
                self.planner.borrow_mut().spacing_m = v;
            }
        }
    }
}

impl ScreenApp for ControlApp {
    fn mount(&mut self, dom: &mut Dom<'_>) {
        dom.mount_widget("control-view", View3d::boxed(self.scene.clone()));
        self.loading_screen = LoadingScreen::mount(dom, LOADING_TITLE);
        if let Some(l) = self.loading_screen.as_mut() {
            l.progress(dom, 0.0, None);
        }
        self.mission = Some(MissionPanel::mount(dom));
        self.telemetry = Some(TelemetryPanel::mount(dom));
        self.video = Some(VideoPanel::mount(dom, self.opts.video.clone()));

        let overlays = dom.find_all(".route-modal-overlay");
        let route = RouteModal::mount(dom, overlays.first().copied());
        if let Some(c) = route.canvas {
            dom.mount_widget(c, PlannerWidget::boxed(self.planner.clone()));
        }
        if let Some(r) = route.range {
            // Blitz lays a range input out as an empty box; give the widget the browser's
            // slider height (the width comes from control.css).
            dom.set_style(r, "display", "inline-block");
            dom.set_style(r, "height", "16px");
            dom.mount_widget(r, RangeWidget::boxed(self.range.clone()));
        }
        {
            let p = self.planner.borrow();
            route.sync_spans(dom, p.span_m);
            route.sync_alt_text(dom, p.agl);
            route.render_list(dom, &p.waypoints);
        }
        self.route = Some(route);
        self.cloud = Some(CloudModal::mount(dom, overlays.get(1).copied()));

        let panel = SettingsPanel::mount(dom, self.settings.overridden);
        panel.show_mode(dom, self.settings.display);
        self.settings_panel = Some(panel);
        self.toolbar = dom.children("control-toolbar");
        dom.remove_class(q("body"), "is-manual");
    }

    fn poll(&mut self, dom: &mut Dom<'_>, cx: &mut PollCx<'_>) -> bool {
        if let Some(w) = cx.waker {
            self.link.set_waker(w);
        }
        let now = cx.now_ms;
        self.handle_link(dom, now);
        if let Some(m) = self.mission.as_mut() {
            m.tick(dom, now);
        }
        if now >= self.next_panel_ms {
            self.next_panel_ms = now + PANEL_INTERVAL_MS;
            self.render_fleet(dom, now);
        }
        // Manual control: 10 Hz while a stick key is held, one neutral frame on release.
        let (send, changed) = self.manual.tick(now);
        if let Tick::Send(axes) = send {
            let drone = self.world.borrow().fleet.active.unwrap_or(1);
            self.link.send(protocol::manual_control(drone, axes.forward, axes.yaw, axes.climb));
        }
        if let Some(active) = changed {
            dom.toggle_class(q("body"), "is-manual", active);
        }
        if let Some(v) = self.video.as_mut() {
            v.tick(dom, now);
        }
        self.handle_http(dom);
        self.drain_planner(dom);
        self.update_loading(dom, now);
        self.toasts.tick(dom, now);
        false
    }

    fn input(&mut self, input: &UiInput, dom: &mut Dom<'_>) -> bool {
        let now = skylens_viewer::now_ms();
        match input.kind {
            InputKind::Click => self.on_click(input, dom, now),
            InputKind::KeyDown => self.on_key_down(input, dom, now),
            InputKind::KeyUp => {
                if let Some(code) = input.code() {
                    self.manual.key_up(code);
                }
            }
            InputKind::Input => self.on_input(input, dom),
            _ => {}
        }
        self.drain_planner(dom);
        true
    }

    fn prevent_default(&self, input: &UiInput) -> bool {
        if input.kind != InputKind::KeyDown {
            return false;
        }
        match input.code() {
            Some(Code::Tab) => true,
            Some(Code::Enter | Code::Space) => input.closest_class("telemetry-row").is_some(),
            _ => false,
        }
    }
}
