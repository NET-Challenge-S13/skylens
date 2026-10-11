//! Pure view functions: state in, text and HTML fragments out.
//!
//! The DOM binding (`app.rs`) writes these into the Blitz document by element id.
//! Keeping them free of Blitz makes the wording and the markup testable.

use crate::chunks::ChunkBook;
use crate::state::{Drone, LinkState, Store};
use skylens_shared::proto::model::v1 as model;
use skylens_shared::proto::payload::v1 as payload;
use skylens_shared::proto::viewer::v1 as viewer;
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// Drone tints, the same palette the three.js rigs used (legacy DRONE_TINTS).
pub const DRONE_TINTS: [[u8; 3]; 3] = [[0x9f, 0xe8, 0xff], [0xff, 0xd2, 0x7f], [0xc7, 0x9f, 0xff]];

pub fn drone_tint(id: u32) -> [u8; 3] {
    DRONE_TINTS[(id.max(1) as usize - 1) % DRONE_TINTS.len()]
}

pub fn hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

/// Escapes text for HTML element content and attribute values.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

pub fn station_label(station: model::DroneStation) -> &'static str {
    match station {
        model::DroneStation::Left => "좌측 드론",
        model::DroneStation::Right => "우측 드론",
        _ => "중앙 드론",
    }
}

pub fn station_cam(station: model::DroneStation) -> &'static str {
    match station {
        model::DroneStation::Left => "LEFT CAM",
        model::DroneStation::Right => "RIGHT CAM",
        _ => "CENTER CAM",
    }
}

/// The four phases of the mission stepper, with their element ids.
pub const PHASES: [(viewer::MissionPhase, &str, &str); 4] = [
    (viewer::MissionPhase::Idle, "step-idle", "대기"),
    (viewer::MissionPhase::Assigned, "step-assigned", "태스크 지정"),
    (viewer::MissionPhase::AwaitingDrone, "step-awaiting-drone", "드론 연결 대기"),
    (viewer::MissionPhase::Active, "step-active", "임무 수행"),
];

/// Shown only before the core has sent any mission status; afterwards the core's
/// own (already localized) message wins.
pub fn phase_fallback(phase: viewer::MissionPhase) -> &'static str {
    match phase {
        viewer::MissionPhase::Assigned => "태스크 지정 완료",
        viewer::MissionPhase::AwaitingDrone => "드론이 현장에 도착해 연결되면 작업이 시작됩니다",
        viewer::MissionPhase::Active => "드론 연결됨. 임무 수행 중",
        _ => "대기 중. 관제탑에서 경로를 지정하십시오",
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissionView {
    pub panel_class: String,
    pub link_class: String,
    pub link_text: String,
    /// (element id, class) per stepper step.
    pub steps: Vec<(&'static str, String)>,
    pub message: String,
    pub sub: String,
}

pub fn mission_view(store: &Store, now_ms: f64) -> MissionView {
    let link_state = match store.link.state {
        LinkState::Connected => "is-connected",
        LinkState::Connecting => "is-connecting",
        LinkState::Disconnected => "is-disconnected",
    };
    let phase = match store.phase() {
        viewer::MissionPhase::Unspecified => viewer::MissionPhase::Idle,
        p => p,
    };
    let active_idx = PHASES.iter().position(|(p, _, _)| *p == phase).unwrap_or(0);
    let steps = PHASES
        .iter()
        .enumerate()
        .map(|(i, (_, id, _))| {
            let class = match i.cmp(&active_idx) {
                std::cmp::Ordering::Less => "mission-panel__step is-done",
                std::cmp::Ordering::Equal => "mission-panel__step is-active",
                std::cmp::Ordering::Greater => "mission-panel__step",
            };
            (*id, class.to_string())
        })
        .collect();

    let (message, sub) = match store.link.state {
        LinkState::Connected => {
            let message = store
                .mission
                .as_ref()
                .map(|m| m.status.message.clone())
                .filter(|m| !m.is_empty())
                .unwrap_or_else(|| phase_fallback(phase).to_string());
            let mut parts = Vec::new();
            if phase == viewer::MissionPhase::AwaitingDrone
                && let Some(left) = store.eta_left_s(now_ms)
            {
                parts.push(format!("예상 {}초", left.round() as i64));
            }
            let online = store
                .mission
                .as_ref()
                .map(|m| m.status.drones_online)
                .unwrap_or_else(|| store.drones.values().filter(|d| d.hello).count() as u32);
            parts.push(format!("드론 {online}대 연결됨"));
            (message, parts.join(" · "))
        }
        LinkState::Connecting => (
            "코어에 연결하는 중입니다".to_string(),
            "연결되면 임무 상태와 드론 위치가 표시됩니다".to_string(),
        ),
        LinkState::Disconnected => (
            "코어와 연결되어 있지 않습니다. 표시된 임무 상태는 최신이 아닐 수 있습니다".to_string(),
            "연결이 복구될 때까지 드론 위치와 임무 상태를 갱신할 수 없습니다".to_string(),
        ),
    };

    let stale = store.link.state != LinkState::Connected;
    MissionView {
        panel_class: format!(
            "sl-surface sl-surface--panel mission-panel{}",
            if stale { " is-stale" } else { "" }
        ),
        link_class: format!("mission-panel__link {link_state}"),
        link_text: store.link.detail.clone(),
        steps,
        message,
        sub,
    }
}

fn gps_line(t: &model::DroneTelemetry) -> String {
    let g = t.gps.clone().unwrap_or_default();
    format!(
        "{:.5}, {:.5} · {:.0}m · {:.0}° · {:.1}m/s · {:.0}%",
        g.lat, g.lon, g.alt, t.heading_deg, t.speed, t.battery_pct
    )
}

/// Rows of the control tower's drone list, in formation order.
pub fn telemetry_rows_html(store: &Store, now_ms: f64) -> String {
    let mut html = String::new();
    for d in store.drones_in_order() {
        let mut class = String::from("telemetry-row");
        if store.active_drone == Some(d.id) {
            class.push_str(" is-active");
        }
        if d.stale(now_ms) {
            class.push_str(" is-stale");
        }
        let badge = if d.hello { "" } else { "<span class=\"telemetry-row__badge\">연결 대기</span>" };
        let vals = match &d.telemetry {
            Some(t) => gps_line(t),
            None if d.model.is_empty() => "위치 수신 대기".to_string(),
            None => format!("위치 수신 대기 · {}", d.model),
        };
        let _ = write!(
            html,
            "<li class=\"{class}\" data-action=\"select-drone\" data-arg=\"{id}\">\
<div class=\"telemetry-row__head\"><span class=\"telemetry-row__dot\" style=\"background: {tint}\"></span>\
<span class=\"telemetry-row__name\">{name}</span>{badge}<span class=\"telemetry-row__stale\">수신 끊김</span></div>\
<div class=\"telemetry-row__vals\">{vals}</div></li>",
            id = d.id,
            tint = hex(drone_tint(d.id)),
            name = station_label(d.station),
            vals = escape(&vals),
        );
    }
    html
}

/// Waypoints being planned, one removable row each.
pub fn route_list_html(draft: &[model::Gps]) -> String {
    if draft.is_empty() {
        return "<li class=\"route-modal__empty\">웨이포인트가 없습니다</li>".to_string();
    }
    let mut html = String::new();
    for (i, wp) in draft.iter().enumerate() {
        let _ = write!(
            html,
            "<li class=\"route-modal__item\"><span>#{n}  {lat:.5}, {lon:.5} · {alt:.0}m</span>\
<button type=\"button\" class=\"route-modal__remove\" data-action=\"remove-waypoint\" data-arg=\"{i}\">삭제</button></li>",
            n = i + 1,
            lat = wp.lat,
            lon = wp.lon,
            alt = wp.alt,
        );
    }
    html
}

/// Parses the route form. Latitude and longitude in degrees, altitude in metres.
pub fn parse_waypoint(lat: &str, lon: &str, alt: &str) -> Result<model::Gps, String> {
    let num = |s: &str, what: &str| -> Result<f64, String> {
        s.trim()
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .ok_or_else(|| format!("{what} 값을 숫자로 입력하세요"))
    };
    let lat = num(lat, "위도")?;
    let lon = num(lon, "경도")?;
    let alt = num(alt, "고도")?;
    if !(-90.0..=90.0).contains(&lat) {
        return Err("위도는 -90 ~ 90 사이여야 합니다".into());
    }
    if !(-180.0..=180.0).contains(&lon) {
        return Err("경도는 -180 ~ 180 사이여야 합니다".into());
    }
    Ok(model::Gps { lat, lon, alt })
}

fn level_rank(level: model::Level) -> u8 {
    match level {
        model::Level::Unspecified => 0,
        model::Level::Preview => 1,
        model::Level::Refined => 2,
    }
}

/// Per-segment level: what the core reports, raised by what actually arrived here.
pub fn segment_levels(server: Option<&viewer::ServerStatus>, book: &ChunkBook) -> BTreeMap<u32, u8> {
    let mut levels: BTreeMap<u32, u8> = BTreeMap::new();
    if let Some(s) = server {
        for seg in &s.segments {
            let e = levels.entry(seg.index).or_default();
            *e = (*e).max(level_rank(seg.level()));
        }
    }
    for (seg, level) in book.levels() {
        let e = levels.entry(*seg).or_default();
        *e = (*e).max(level_rank(*level));
    }
    levels
}

/// The ladder rows of the status board's server panel: two pips per segment
/// (preview, refined).
pub fn ladder_html(levels: &BTreeMap<u32, u8>) -> String {
    let mut html = String::new();
    for (seg, level) in levels {
        let pip = |on: bool| format!("<span class=\"segment-row__pip\" data-on=\"{}\"></span>", u8::from(on));
        let note = match level {
            0 => "복원 중",
            1 => "수준 1 · 미리보기",
            _ => "수준 2 · 정제",
        };
        let _ = write!(
            html,
            "<div class=\"segment-row\"><span class=\"segment-row__name\">구간 {n}</span>\
<span class=\"segment-row__bar\">{p1}{p2}</span>\
<span class=\"segment-row__note\" data-refining=\"{refining}\">{note}{tail}</span></div>",
            n = seg + 1,
            p1 = pip(*level >= 1),
            p2 = pip(*level >= 2),
            refining = u8::from(*level == 1),
            tail = if *level == 1 { " · 정제 중" } else { "" },
        );
    }
    html
}

#[derive(Debug, Clone, PartialEq)]
pub struct ServerView {
    /// waiting | connected | receiving
    pub state: &'static str,
    pub text: &'static str,
    pub detail: String,
    pub link: String,
}

pub fn server_view(store: &Store) -> ServerView {
    let server = store.server.as_ref().filter(|_| store.link.state == LinkState::Connected);
    let state = match server {
        Some(s) if s.receiving => "receiving",
        Some(s) if s.connected => "connected",
        _ => "waiting",
    };
    let text = match state {
        "receiving" => "서버 · 수신 중",
        "connected" => "서버 · 연결됨",
        _ => "서버 · 대기 중",
    };
    let chunks = server.map_or(store.chunks.chunks_received, |s| s.chunks.max(store.chunks.chunks_received));
    let detections = server.map_or(store.detections.len() as u64, |s| s.detections.max(store.detections.len() as u64));
    let mut detail = format!("청크 {chunks} · 탐지 {detections}");
    if let Some(ms) = server.and_then(|s| s.latency_ms) {
        let _ = write!(detail, " · {}ms", ms.round() as i64);
    }
    let link = match (store.link.state, store.mission.as_ref()) {
        (LinkState::Connected, Some(m)) if state == "receiving" => {
            format!("{} · 드론 {}대", m.status.message, m.status.drones_online)
        }
        (LinkState::Connected, Some(m)) if !m.status.message.is_empty() => m.status.message.clone(),
        (LinkState::Connected, _) if server.is_none() => "코어 연결됨 · 서버 상태 대기 중".to_string(),
        (LinkState::Connected, _) => "코어 연결됨".to_string(),
        (_, _) => store.link.detail.clone(),
    };
    ServerView { state, text, detail, link }
}

/// Corner badge: this screen's link to the core, then the worst hop the core reports.
pub fn net_view(store: &Store) -> (&'static str, String) {
    match store.link.state {
        LinkState::Connecting => return ("warn", "코어 연결 중".to_string()),
        LinkState::Disconnected => return ("down", format!("코어 끊김 · {}", store.link.detail)),
        LinkState::Connected => {}
    }
    if let Some(down) = store.hops.values().find(|h| !h.connected) {
        return ("warn", format!("{} 끊김", down.hop));
    }
    let worst = store
        .hops
        .values()
        .filter_map(|h| h.latency_ms)
        .fold(None::<f64>, |acc, v| Some(acc.map_or(v, |a| a.max(v))));
    match worst {
        Some(ms) => ("ok", format!("코어 연결됨 · 경로 {}개 · 최대 {}ms", store.hops.len(), ms.round() as i64)),
        None => ("ok", "코어 연결됨".to_string()),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CameraView {
    pub label: &'static str,
    pub state: String,
    pub url: String,
    pub overlay: String,
}

/// The camera panel follows the active drone; with no drone, any announced feed.
pub fn camera_view(store: &Store) -> CameraView {
    let drone: Option<&Drone> = store.active();
    let feed = drone
        .and_then(|d| store.feed_for(d.id))
        .or_else(|| store.camera_feeds.values().next());
    let station = drone
        .map(|d| d.station)
        .or_else(|| feed.map(|f| f.station()))
        .unwrap_or(model::DroneStation::Center);
    let state = match feed {
        Some(_) => "WHEP 라이브 영상 · 이 화면 안에서는 재생하지 않습니다".to_string(),
        None if drone.is_some() => "영상 주소 대기 중".to_string(),
        None => "영상 대기".to_string(),
    };
    let overlay = match drone.and_then(|d| d.telemetry.as_ref()) {
        Some(t) => {
            let g = t.gps.clone().unwrap_or_default();
            format!("{} · {:.5}, {:.5} · {:.0}m · {:.0}°", station_cam(station), g.lat, g.lon, g.alt, t.heading_deg)
        }
        None => format!("{} · NO SIGNAL", station_cam(station)),
    };
    CameraView {
        label: station_cam(station),
        state,
        url: feed.map(|f| f.whep_url.clone()).unwrap_or_default(),
        overlay,
    }
}

pub fn category_name(c: payload::DetectionCategory) -> &'static str {
    match c {
        payload::DetectionCategory::Person => "생존자",
        payload::DetectionCategory::Fire => "화재",
        payload::DetectionCategory::Collapse => "붕괴",
        payload::DetectionCategory::RoadBlocked => "도로 차단",
        payload::DetectionCategory::Unspecified => "탐지",
    }
}

pub fn is_person(c: payload::DetectionCategory) -> bool {
    c == payload::DetectionCategory::Person
}

#[derive(Debug, Clone, PartialEq)]
pub struct DetectionCardView {
    pub class: String,
    pub kicker: &'static str,
    pub label: String,
    pub conf: String,
    pub fill_style: String,
}

pub fn detection_card(d: &payload::Detection) -> DetectionCardView {
    let person = is_person(d.category());
    let pct = (d.confidence * 100.0).round().clamp(0.0, 100.0) as i64;
    let label = if d.label.is_empty() { category_name(d.category()).to_string() } else { d.label.clone() };
    DetectionCardView {
        class: format!("detect-card is-open {}", if person { "kind-person" } else { "kind-danger" }),
        kicker: if person { "생존자 감지" } else { "위험구역 감지" },
        label,
        conf: format!("확신도 {pct}% · 프레임 {} · 구간 {}", d.frames, d.segment + 1),
        fill_style: format!("width: {pct}%"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToastKind {
    Info,
    Person,
    Danger,
}

#[derive(Debug, Clone)]
pub struct Toast {
    pub text: String,
    pub kind: ToastKind,
    pub expires_ms: f64,
}

pub const TOAST_LIFETIME_MS: f64 = 3_200.0;

pub fn toasts_html(toasts: &[Toast]) -> String {
    let mut html = String::new();
    for t in toasts {
        let kind = match t.kind {
            ToastKind::Info => "",
            ToastKind::Person => " kind-person",
            ToastKind::Danger => " kind-danger",
        };
        let _ = write!(html, "<div class=\"toast{kind}\">{}</div>", escape(&t.text));
    }
    html
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Body;

    fn env(body: Body) -> viewer::Envelope {
        viewer::Envelope { seq: 1, origin_ts_ms: 0, from: 2, body: Some(body) }
    }

    fn connected() -> Store {
        let mut s = Store::new();
        s.set_link(LinkState::Connected, "코어 연결됨");
        s
    }

    #[test]
    fn escape_covers_markup() {
        assert_eq!(escape("<a href=\"x\">&'"), "&lt;a href=&quot;x&quot;&gt;&amp;&#39;");
    }

    #[test]
    fn tints_cycle_by_id() {
        assert_eq!(hex(drone_tint(1)), "#9fe8ff");
        assert_eq!(drone_tint(4), drone_tint(1));
        assert_eq!(drone_tint(0), drone_tint(1));
    }

    #[test]
    fn mission_view_follows_phase_and_link() {
        let mut s = Store::new();
        let v = mission_view(&s, 0.0);
        assert!(v.panel_class.contains("is-stale"));
        assert_eq!(v.message, "코어에 연결하는 중입니다");

        s.set_link(LinkState::Connected, "코어 연결됨");
        s.apply(
            env(Body::MissionStatus(viewer::MissionStatus {
                phase: viewer::MissionPhase::AwaitingDrone as i32,
                message: "드론 도착 대기".into(),
                drones_online: 0,
                eta_seconds: Some(12.0),
            })),
            1_000.0,
        );
        let v = mission_view(&s, 3_000.0);
        assert!(!v.panel_class.contains("is-stale"));
        assert_eq!(v.link_class, "mission-panel__link is-connected");
        assert_eq!(v.message, "드론 도착 대기");
        assert_eq!(v.sub, "예상 10초 · 드론 0대 연결됨");
        let classes: Vec<&str> = v.steps.iter().map(|(_, c)| c.as_str()).collect();
        assert_eq!(
            classes,
            vec![
                "mission-panel__step is-done",
                "mission-panel__step is-done",
                "mission-panel__step is-active",
                "mission-panel__step"
            ]
        );
    }

    #[test]
    fn telemetry_rows_escape_and_mark_active() {
        let mut s = connected();
        s.apply(
            env(Body::DroneHello(viewer::DroneHello {
                drone_id: 2,
                station: model::DroneStation::Center as i32,
                model: "<x>".into(),
            })),
            0.0,
        );
        let html = telemetry_rows_html(&s, 0.0);
        assert!(html.contains("is-active"));
        assert!(html.contains("중앙 드론"));
        assert!(html.contains("&lt;x&gt;"));
        assert!(html.contains("data-arg=\"2\""));
        assert!(!html.contains("연결 대기"));
    }

    #[test]
    fn waypoint_parsing() {
        let g = parse_waypoint(" 36.3685 ", "127.3475", "60").unwrap();
        assert_eq!((g.lat, g.lon, g.alt), (36.3685, 127.3475, 60.0));
        assert!(parse_waypoint("91", "0", "0").is_err());
        assert!(parse_waypoint("0", "181", "0").is_err());
        assert!(parse_waypoint("abc", "0", "0").is_err());
        assert!(parse_waypoint("0", "0", "NaN").is_err());
    }

    #[test]
    fn route_list_numbers_rows() {
        assert!(route_list_html(&[]).contains("웨이포인트가 없습니다"));
        let html = route_list_html(&vec![model::Gps { lat: 1.0, lon: 2.0, alt: 3.0 }; 2]);
        assert!(html.contains("#2"));
        assert!(html.contains("data-arg=\"1\""));
    }

    #[test]
    fn ladder_merges_core_and_local_levels() {
        let server = viewer::ServerStatus {
            segments: vec![
                viewer::SegmentStatus { index: 0, level: model::Level::Refined as i32 },
                viewer::SegmentStatus { index: 1, level: model::Level::Unspecified as i32 },
            ],
            ..Default::default()
        };
        let book = ChunkBook::new();
        let levels = segment_levels(Some(&server), &book);
        assert_eq!(levels.get(&0), Some(&2));
        assert_eq!(levels.get(&1), Some(&0));
        let html = ladder_html(&levels);
        assert!(html.contains("구간 1"));
        assert!(html.contains("수준 2 · 정제"));
        assert!(html.contains("복원 중"));
    }

    #[test]
    fn server_and_net_views() {
        let mut s = Store::new();
        assert_eq!(server_view(&s).state, "waiting");
        assert_eq!(net_view(&s).0, "warn");
        s.set_link(LinkState::Connected, "코어 연결됨");
        s.apply(
            env(Body::ServerStatus(viewer::ServerStatus {
                connected: true,
                receiving: true,
                chunks: 12,
                detections: 3,
                latency_ms: Some(41.6),
                ..Default::default()
            })),
            0.0,
        );
        let v = server_view(&s);
        assert_eq!(v.state, "receiving");
        assert_eq!(v.detail, "청크 12 · 탐지 3 · 42ms");
        s.apply(
            env(Body::LinkStatus(viewer::LinkStatus {
                hop: "proxy->core".into(),
                connected: false,
                latency_ms: None,
                mbps: None,
            })),
            0.0,
        );
        assert_eq!(net_view(&s), ("warn", "proxy->core 끊김".to_string()));
    }

    #[test]
    fn camera_view_shows_the_whep_url_of_the_active_drone() {
        let mut s = connected();
        assert_eq!(camera_view(&s).state, "영상 대기");
        s.apply(
            env(Body::CameraFeed(viewer::CameraFeed {
                drone_id: 3,
                station: model::DroneStation::Left as i32,
                whep_url: "http://core:10889/left/whep".into(),
            })),
            0.0,
        );
        let v = camera_view(&s);
        assert_eq!(v.url, "http://core:10889/left/whep");
        assert_eq!(v.label, "LEFT CAM");
    }

    #[test]
    fn detection_card_wording() {
        let d = payload::Detection {
            id: "z".into(),
            category: payload::DetectionCategory::Fire as i32,
            confidence: 0.736,
            segment: 0,
            frames: 5,
            ..Default::default()
        };
        let v = detection_card(&d);
        assert!(v.class.contains("kind-danger"));
        assert_eq!(v.label, "화재");
        assert_eq!(v.fill_style, "width: 74%");
        assert_eq!(v.conf, "확신도 74% · 프레임 5 · 구간 1");
    }
}
