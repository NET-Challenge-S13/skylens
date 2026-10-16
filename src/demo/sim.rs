//! The formation's side of the core: three drones on `ws /uplink`, and a
//! control-tower watcher on `ws /control` that logs mission phases (and, with
//! `--auto`, assigns the pre-planned route itself).
//!
//! Scenario (COMPONENTS.md §4): the drones connect but stay silent until a
//! route is assigned (the core hands it to every uplink, or replays it to one
//! that attaches later). They then fly in for `ETA` seconds sending telemetry,
//! say `drone_hello` on arrival (the only arrival signal the core counts), and
//! fly with telemetry every 0.5 s. Every socket keeps reading so the core's
//! pings get answered.
//!
//! What "fly" means depends on the [`Flight`]: with the recorded footage each
//! drone replays its view's flight record on its clip's timeline (drone 1 =
//! left, 2 = center, 3 = right), jumping where the clip jumps; with
//! `--synthetic` the formation flies the fixed track of `route.rs`.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures_util::stream::SplitSink;
use futures_util::{SinkExt, StreamExt};
use prost::Message as _;
use skylens_shared::geo::Gps;
use skylens_shared::proto::model::v1 as model;
use skylens_shared::proto::viewer::v1::{self as viewer, ComponentId, Envelope, envelope::Body};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, watch};
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use crate::board::ws_config;
use crate::footage::{self, Track, View};
use crate::paths::{Drone, FORMATION};
use crate::procs::log;
use crate::route::{self, APPROACH_M, ALTITUDE_M, Path, SITE_ANCHOR, SPEED_MPS};

/// Telemetry period.
pub const PERIOD: Duration = Duration::from_millis(500);
/// Seconds between the route and the hello; the core's countdown is set to the same.
pub const ETA_SECS: f64 = 10.0;

type Sink = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, WsMessage>;

/// What the drones fly once they have a route (unless `--follow-route`).
#[derive(Debug, Clone)]
pub enum Flight {
    /// `--synthetic`: the fixed track around the default anchor, at `ALTITUDE_M`.
    Synthetic,
    /// The recorded flight: drone i replays view i, `starts[i]` being when its publisher
    /// started (clip time = elapsed since then, modulo the clip length). `ground_alt` turns
    /// heights above takeoff into absolute altitudes.
    Recorded { track: Track, starts: [Instant; 3], ground_alt: f64 },
}

impl Flight {
    fn view(&self, index: usize) -> Option<(&View, Instant, f64)> {
        match self {
            Flight::Synthetic => None,
            Flight::Recorded { track, starts, ground_alt } => {
                Some((track.views()[index], starts[index], *ground_alt))
            }
        }
    }
}

/// One telemetry fix: absolute GPS, compass heading, ground speed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fix {
    pub gps: Gps,
    pub heading_deg: f64,
    pub speed: f64,
}

/// A recorded view at wall time `now`, in sync with a publisher that started at `start`.
pub fn recorded_fix(view: &View, start: Instant, now: Instant, ground_alt: f64) -> Fix {
    let t = footage::clip_time(now.saturating_duration_since(start).as_secs_f64(), view.duration);
    let s = view.sample(t);
    Fix { gps: s.gps(ground_alt), heading_deg: s.heading_deg, speed: s.speed }
}

/// One station of the synthetic formation.
pub fn formation_fix(pose: &route::Pose, drone: &Drone, alt: f64) -> Fix {
    Fix { gps: route::to_gps(pose, drone.station, SITE_ANCHOR, alt), heading_deg: pose.heading_deg, speed: pose.speed }
}

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

fn envelope(seq: u64, body: Body) -> Envelope {
    Envelope { seq, origin_ts_ms: now_ms(), from: ComponentId::Demo as i32, body: Some(body) }
}

pub fn hello(seq: u64, drone: &Drone) -> Envelope {
    envelope(
        seq,
        Body::DroneHello(viewer::DroneHello {
            drone_id: drone.id,
            station: drone.station as i32,
            model: format!("SkyLens D-{}", drone.name.to_uppercase()),
        }),
    )
}

/// Telemetry fills every slot `DroneTelemetry` has: position (absolute altitude), heading,
/// ground speed, battery and the drone's clock. The flight record's gimbal angles have no slot.
pub fn telemetry(seq: u64, drone: &Drone, fix: &Fix, flight_secs: f64, t_ms: i64) -> Envelope {
    envelope(
        seq,
        Body::Telemetry(model::DroneTelemetry {
            drone_id: drone.id,
            station: drone.station as i32,
            gps: Some(fix.gps.into()),
            heading_deg: fix.heading_deg,
            speed: fix.speed,
            battery_pct: route::battery_pct(flight_secs),
            t_ms,
        }),
    )
}

/// The route an envelope carries, if it is an `assign_route`.
pub fn route_of(bytes: &[u8]) -> Option<viewer::AssignRoute> {
    match Envelope::decode(bytes).ok()?.body? {
        Body::AssignRoute(r) => Some(r),
        _ => None,
    }
}

/// Short Korean line for a mission phase.
pub fn phase_name(phase: i32) -> &'static str {
    match viewer::MissionPhase::try_from(phase).unwrap_or(viewer::MissionPhase::Unspecified) {
        viewer::MissionPhase::Idle => "대기",
        viewer::MissionPhase::Assigned => "태스크 지정",
        viewer::MissionPhase::AwaitingDrone => "드론 연결 대기",
        viewer::MissionPhase::Active => "임무 수행",
        viewer::MissionPhase::Unspecified => "알 수 없음",
    }
}

async fn connect(url: &str) -> anyhow::Result<WebSocketStream<MaybeTlsStream<TcpStream>>> {
    let mut last = None;
    for _ in 0..20 {
        match tokio_tungstenite::connect_async_with_config(url, Some(ws_config()), true).await {
            Ok((ws, _)) => return Ok(ws),
            Err(e) => last = Some(e),
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    Err(anyhow::anyhow!("{url}: {}", last.map(|e| e.to_string()).unwrap_or_default()))
}

struct Uplink {
    drone: Drone,
    sink: Sink,
    seq: u64,
}

impl Uplink {
    async fn send(&mut self, env: impl FnOnce(u64) -> Envelope) -> anyhow::Result<()> {
        self.seq += 1;
        let bytes = env(self.seq).encode_to_vec();
        self.sink
            .send(WsMessage::Binary(bytes.into()))
            .await
            .map_err(|e| anyhow::anyhow!("drone {} uplink: {e}", self.drone.id))
    }
}

/// How the drones fly the current route.
#[derive(Debug, Clone, PartialEq)]
pub enum Mode {
    /// The synthetic formation along a path, at an absolute altitude.
    Formation { path: Path, alt: f64 },
    /// Each drone replays its recorded view.
    Recorded,
}

/// Mean altitude of a route's waypoints (`ALTITUDE_M` without any).
pub fn route_alt(r: &viewer::AssignRoute) -> f64 {
    if r.waypoints.is_empty() {
        return ALTITUDE_M;
    }
    r.waypoints.iter().map(|g| g.alt).sum::<f64>() / r.waypoints.len() as f64
}

/// The operator's waypoints with `--follow-route`, else the footage (recorded or synthetic).
pub fn choose_mode(r: &viewer::AssignRoute, follow_route: bool, flight: &Flight) -> Mode {
    if follow_route {
        let waypoints: Vec<Gps> = r.waypoints.iter().map(|g| Gps { lat: g.lat, lon: g.lon, alt: g.alt }).collect();
        if let Some(path) = Path::from_waypoints(&waypoints, SITE_ANCHOR, r.r#loop) {
            return Mode::Formation { path, alt: route_alt(r) };
        }
        log("drone", "경로 지점이 둘 미만이라 촬영 경로로 비행합니다");
    }
    match flight {
        Flight::Synthetic => Mode::Formation { path: Path::footage_track(), alt: ALTITUDE_M },
        Flight::Recorded { .. } => Mode::Recorded,
    }
}

/// Fixes of the three drones while flying in, `t` seconds into the approach. Recorded drones
/// end where their clip will be at arrival (`targets`), so the hello lands on the footage.
fn approach_fixes(mode: &Mode, targets: &[(Gps, f64); 3], t: f64) -> [Fix; 3] {
    std::array::from_fn(|i| match mode {
        Mode::Formation { path, alt } => formation_fix(&route::approach_pose(path, t, ETA_SECS), &FORMATION[i], *alt),
        Mode::Recorded => {
            let (target, course) = targets[i];
            let (gps, speed) = footage::approach_gps(target, course, APPROACH_M, t, ETA_SECS);
            Fix { gps, heading_deg: course, speed }
        }
    })
}

/// Fixes of the three drones on the route: `leg` seconds into the formation's path, or the
/// recorded views at `now`.
fn flight_fixes(mode: &Mode, flight: &Flight, leg: f64, now: Instant) -> [Fix; 3] {
    std::array::from_fn(|i| match (mode, flight.view(i)) {
        (Mode::Recorded, Some((view, start, ground))) => recorded_fix(view, start, now, ground),
        (Mode::Formation { path, alt }, _) => formation_fix(&path.pose_at(leg, SPEED_MPS), &FORMATION[i], *alt),
        // Recorded mode without a record cannot happen (choose_mode); fly the synthetic track.
        (Mode::Recorded, None) => formation_fix(&Path::footage_track().pose_at(leg, SPEED_MPS), &FORMATION[i], ALTITUDE_M),
    })
}

/// Where each recorded drone will be when it arrives, and its direction of flight.
fn arrival_targets(flight: &Flight, arrival: Instant) -> [(Gps, f64); 3] {
    std::array::from_fn(|i| match flight.view(i) {
        Some((view, start, ground)) => (recorded_fix(view, start, arrival, ground).gps, view.course_deg()),
        None => (SITE_ANCHOR, 0.0),
    })
}

/// Run the three drones until an uplink fails (returned as an error).
pub async fn run_drones(core: String, follow_route: bool, flight: Flight) -> anyhow::Result<()> {
    let url = format!("ws://{core}/uplink");
    let (route_tx, mut route_rx) = watch::channel::<Option<viewer::AssignRoute>>(None);
    let (dead_tx, mut dead_rx) = mpsc::unbounded_channel::<String>();
    let mut uplinks = Vec::new();
    for drone in FORMATION {
        let ws = connect(&url).await?;
        let (sink, mut stream) = ws.split();
        let (route_tx, dead_tx) = (route_tx.clone(), dead_tx.clone());
        // Reader: keeps the socket answering pings and picks up routes.
        tokio::spawn(async move {
            let reason = loop {
                match stream.next().await {
                    Some(Ok(WsMessage::Binary(b))) => {
                        if let Some(r) = route_of(&b) {
                            route_tx.send_replace(Some(r));
                        }
                    }
                    Some(Ok(WsMessage::Close(_))) | None => break "closed by the core".to_string(),
                    Some(Ok(_)) => {}
                    Some(Err(e)) => break e.to_string(),
                }
            };
            let _ = dead_tx.send(format!("drone {} uplink: {reason}", drone.id));
        });
        uplinks.push(Uplink { drone, sink, seq: 0 });
    }
    log("drone", "드론 3대 업링크 연결, 대기 중 (관제탑에서 경로를 배정하면 출발)");

    // ---- idle: wait for a route ----
    let first = loop {
        tokio::select! {
            changed = route_rx.changed() => {
                changed.map_err(|_| anyhow::anyhow!("route channel closed"))?;
                if let Some(r) = route_rx.borrow_and_update().clone() { break r; }
            }
            Some(reason) = dead_rx.recv() => anyhow::bail!(reason),
        }
    };
    let mut mode = choose_mode(&first, follow_route, &flight);
    log(
        "drone",
        &format!("경로 수신 ({}개 지점). {ETA_SECS:.0}초 뒤 현장 도착 예정", first.waypoints.len()),
    );

    // ---- approach: telemetry, no hello ----
    let mut tick = tokio::time::interval(PERIOD);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let started = tokio::time::Instant::now();
    let targets = arrival_targets(&flight, Instant::now() + Duration::from_secs_f64(ETA_SECS));
    loop {
        tokio::select! {
            _ = tick.tick() => {}
            Some(reason) = dead_rx.recv() => anyhow::bail!(reason),
        }
        let t = started.elapsed().as_secs_f64();
        let fixes = approach_fixes(&mode, &targets, t);
        for (u, fix) in uplinks.iter_mut().zip(fixes) {
            let drone = u.drone;
            u.send(|seq| telemetry(seq, &drone, &fix, t, now_ms())).await?;
        }
        if t >= ETA_SECS {
            break;
        }
    }

    // ---- arrival ----
    for u in &mut uplinks {
        let drone = u.drone;
        u.send(|seq| hello(seq, &drone)).await?;
    }
    match mode {
        Mode::Recorded => log("drone", "현장 도착: drone_hello 전송, 드론마다 자기 영상의 비행 기록을 재생"),
        Mode::Formation { .. } => log("drone", "현장 도착: drone_hello 전송, 경로 비행 시작"),
    }

    // ---- flight ----
    let mut leg_start = tokio::time::Instant::now();
    loop {
        tokio::select! {
            _ = tick.tick() => {}
            changed = route_rx.changed() => {
                changed.map_err(|_| anyhow::anyhow!("route channel closed"))?;
                if let Some(r) = route_rx.borrow_and_update().clone() {
                    if follow_route {
                        mode = choose_mode(&r, true, &flight);
                        leg_start = tokio::time::Instant::now();
                        log("drone", &format!("새 경로 ({}개 지점) 비행", r.waypoints.len()));
                    } else {
                        log("drone", "새 경로 수신. 촬영 경로를 계속 비행합니다 (--follow-route 로 바꿀 수 있음)");
                    }
                }
                continue;
            }
            Some(reason) = dead_rx.recv() => anyhow::bail!(reason),
        }
        let flight_secs = started.elapsed().as_secs_f64();
        let fixes = flight_fixes(&mode, &flight, leg_start.elapsed().as_secs_f64(), Instant::now());
        for (u, fix) in uplinks.iter_mut().zip(fixes) {
            let drone = u.drone;
            u.send(|seq| telemetry(seq, &drone, &fix, flight_secs, now_ms())).await?;
        }
    }
}

/// Watch `/control`: log mission phases, and with `auto` assign that route once the tower
/// socket has been up for a few seconds. Returns when the socket fails.
pub async fn run_control_watch(core: String, auto: Option<viewer::AssignRoute>) -> anyhow::Result<()> {
    let url = format!("ws://{core}/control");
    let ws = connect(&url).await?;
    let (mut sink, mut stream) = ws.split();
    let mut assign_at = auto.is_some().then(|| tokio::time::Instant::now() + Duration::from_secs(3));
    let mut last_phase = None;
    loop {
        let at = assign_at;
        let wait = async move {
            match at {
                Some(at) => tokio::time::sleep_until(at).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            _ = wait => {
                assign_at = None;
                if let Some(r) = &auto {
                    let env = envelope(1, Body::AssignRoute(r.clone()));
                    sink.send(WsMessage::Binary(env.encode_to_vec().into())).await?;
                    log("auto", &format!("--auto: 미리 계획된 경로({}개 지점)를 배정했습니다", r.waypoints.len()));
                }
            }
            frame = stream.next() => match frame {
                Some(Ok(WsMessage::Binary(b))) => {
                    match Envelope::decode(&b[..]) {
                        Ok(Envelope { body: Some(Body::MissionStatus(s)), .. }) if last_phase != Some(s.phase) => {
                            last_phase = Some(s.phase);
                            log("tower", &format!("미션: {} ({}) 드론 {}대", phase_name(s.phase), s.message, s.drones_online));
                        }
                        Ok(Envelope { body: Some(Body::PlannedRoute(p)), .. }) => {
                            log("tower", &format!("관제탑에 미리 계획된 경로 {}개 지점이 들어갔습니다", p.waypoints.len()));
                        }
                        _ => {}
                    }
                }
                Some(Ok(WsMessage::Close(_))) | None => anyhow::bail!("control socket closed by the core"),
                Some(Ok(_)) => {}
                Some(Err(e)) => anyhow::bail!("control socket: {e}"),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECORD: &str = r#"{"views": {
        "left":   {"video": "left.mp4", "duration": 10.0,
                   "segments": [{"start": 0.0, "duration": 10.0, "rel_alt_m": 40.0}],
                   "track": [{"t": 0.0, "lat": 36.3660, "lon": 127.3420, "rel_alt_m": 40.0, "aircraft_yaw": 45.0},
                             {"t": 10.0, "lat": 36.3650, "lon": 127.3425, "rel_alt_m": 40.0, "aircraft_yaw": 45.0}]},
        "center": {"video": "center.mp4", "duration": 10.0,
                   "track": [{"t": 0.0, "lat": 36.3660, "lon": 127.3420, "rel_alt_m": 20.0, "aircraft_yaw": 165.0},
                             {"t": 10.0, "lat": 36.3650, "lon": 127.3425, "rel_alt_m": 20.0, "aircraft_yaw": 165.0}]},
        "right":  {"video": "right.mp4", "duration": 10.0,
                   "track": [{"t": 0.0, "lat": 36.3660, "lon": 127.3420, "rel_alt_m": 30.0, "aircraft_yaw": 285.0},
                             {"t": 10.0, "lat": 36.3650, "lon": 127.3425, "rel_alt_m": 30.0, "aircraft_yaw": 285.0}]}}}"#;

    fn recorded(starts: [Instant; 3]) -> Flight {
        Flight::Recorded { track: Track::parse(RECORD).unwrap(), starts, ground_alt: 50.0 }
    }

    #[test]
    fn hello_names_the_station() {
        let env = hello(7, &FORMATION[0]);
        assert_eq!(env.seq, 7);
        assert_eq!(env.from, ComponentId::Demo as i32);
        let Some(Body::DroneHello(h)) = env.body else { panic!() };
        assert_eq!(h.drone_id, 1);
        assert_eq!(h.station, model::DroneStation::Left as i32);
        assert_eq!(h.model, "SkyLens D-LEFT");
    }

    #[test]
    fn telemetry_carries_every_field() {
        let fix = Fix { gps: Gps { lat: 36.36, lon: 127.34, alt: 92.5 }, heading_deg: 165.0, speed: 3.5 };
        let env = telemetry(1, &FORMATION[2], &fix, 100.0, 123);
        let Some(Body::Telemetry(t)) = env.body else { panic!() };
        assert_eq!(t.drone_id, 3);
        assert_eq!(t.station, model::DroneStation::Right as i32);
        assert_eq!(t.t_ms, 123);
        let gps = t.gps.unwrap();
        assert_eq!((gps.lat, gps.lon, gps.alt), (36.36, 127.34, 92.5));
        assert_eq!((t.heading_deg, t.speed), (165.0, 3.5));
        assert!((t.battery_pct - route::battery_pct(100.0)).abs() < 1e-12);
    }

    #[test]
    fn synthetic_stations_fly_the_old_track() {
        let pose = Path::footage_track().pose_at(0.0, SPEED_MPS);
        let fix = formation_fix(&pose, &FORMATION[2], ALTITUDE_M);
        assert_eq!(fix.gps.alt, ALTITUDE_M);
        assert!(fix.gps.lat > SITE_ANCHOR.lat - 0.01 && fix.gps.lat < SITE_ANCHOR.lat + 0.01);
        assert!((fix.heading_deg - route::TRACK_BEARING_DEG).abs() < 1e-9);
    }

    #[test]
    fn recorded_drones_follow_their_clip() {
        let t0 = Instant::now();
        let flight = recorded([t0, t0 + Duration::from_secs(2), t0]);
        let now = t0 + Duration::from_secs(5);
        let fixes = flight_fixes(&Mode::Recorded, &flight, 0.0, now);
        // Drone 1: 5 s into its clip, half way, 50 + 40 m.
        assert!((fixes[0].gps.lat - 36.3655).abs() < 1e-9);
        assert!((fixes[0].gps.alt - 90.0).abs() < 1e-9);
        assert!((fixes[0].heading_deg - 45.0).abs() < 1e-9);
        assert!(fixes[0].speed > 0.0);
        // Drone 2 started 2 s later: 3 s in, its own height and heading.
        assert!((fixes[1].gps.lat - 36.3657).abs() < 1e-9);
        assert!((fixes[1].gps.alt - 70.0).abs() < 1e-9);
        assert!((fixes[1].heading_deg - 165.0).abs() < 1e-9);
        // The clip loops: 12 s after the start is 2 s in.
        let later = recorded_fix(&Track::parse(RECORD).unwrap().left, t0, t0 + Duration::from_secs(12), 50.0);
        assert!((later.gps.lat - 36.3658).abs() < 1e-9);
        // A publisher that has not started yet reads as clip time 0.
        let early = recorded_fix(&Track::parse(RECORD).unwrap().left, now, t0, 50.0);
        assert!((early.gps.lat - 36.3660).abs() < 1e-12);
    }

    #[test]
    fn recorded_approach_lands_on_the_clip() {
        let t0 = Instant::now();
        let flight = recorded([t0; 3]);
        let arrival = t0 + Duration::from_secs(4);
        let targets = arrival_targets(&flight, arrival);
        let at_arrival = flight_fixes(&Mode::Recorded, &flight, 0.0, arrival);
        let end = approach_fixes(&Mode::Recorded, &targets, ETA_SECS);
        for i in 0..3 {
            assert!((end[i].gps.lat - at_arrival[i].gps.lat).abs() < 1e-12);
            assert!((end[i].gps.alt - at_arrival[i].gps.alt).abs() < 1e-9);
        }
        let start = approach_fixes(&Mode::Recorded, &targets, 0.0);
        // Behind the line: north of where it arrives (the line runs south-south-east).
        assert!(start[1].gps.lat > end[1].gps.lat);
        assert!((start[1].speed - APPROACH_M / ETA_SECS).abs() < 1e-9);
    }

    #[test]
    fn route_round_trips_through_the_wire() {
        let wps = route::footage_waypoints(SITE_ANCHOR);
        let r = viewer::AssignRoute { drone_id: 2, waypoints: wps.into_iter().map(Into::into).collect(), r#loop: true };
        let env = envelope(1, Body::AssignRoute(r));
        let r = route_of(&env.encode_to_vec()).unwrap();
        assert_eq!((r.drone_id, r.waypoints.len()), (2, 2));
        assert!(r.r#loop);
        assert!(route_of(&hello(1, &FORMATION[0]).encode_to_vec()).is_none());
        assert!(route_of(b"\xff\xff").is_none());
    }

    #[test]
    fn modes() {
        let t0 = Instant::now();
        let mut r = viewer::AssignRoute { drone_id: 2, waypoints: vec![], r#loop: true };
        // Too few points: back to the footage.
        assert_eq!(choose_mode(&r, true, &recorded([t0; 3])), Mode::Recorded);
        assert_eq!(
            choose_mode(&r, true, &Flight::Synthetic),
            Mode::Formation { path: Path::footage_track(), alt: ALTITUDE_M }
        );
        r.waypoints = route::footage_waypoints(SITE_ANCHOR)
            .into_iter()
            .map(|g| model::Gps { lat: g.lat, lon: g.lon, alt: 70.0 })
            .collect();
        match choose_mode(&r, true, &recorded([t0; 3])) {
            Mode::Formation { path, alt } => {
                assert_eq!(path.motion, route::Motion::PingPong);
                assert_eq!(alt, 70.0);
            }
            m => panic!("{m:?}"),
        }
        assert_eq!(choose_mode(&r, false, &recorded([t0; 3])), Mode::Recorded);
        assert_eq!(route_alt(&viewer::AssignRoute::default()), ALTITUDE_M);
    }

    #[test]
    fn phases_have_names() {
        assert_eq!(phase_name(viewer::MissionPhase::AwaitingDrone as i32), "드론 연결 대기");
        assert_eq!(phase_name(99), "알 수 없음");
    }
}
