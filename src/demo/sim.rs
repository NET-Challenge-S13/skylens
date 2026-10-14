//! The formation's side of the core: three drones on `ws /uplink`, and a
//! control-tower watcher on `ws /control` that logs mission phases (and, with
//! `--auto`, assigns the route itself).
//!
//! Scenario (COMPONENTS.md §4): the drones connect but stay silent until a
//! route is assigned (the core hands it to every uplink, or replays it to one
//! that attaches later). They then fly in for `ETA` seconds sending telemetry,
//! say `drone_hello` on arrival (the only arrival signal the core counts), and
//! fly the track with telemetry every 0.5 s. Every socket keeps reading so the
//! core's pings get answered.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
use crate::paths::{Drone, FORMATION};
use crate::procs::log;
use crate::route::{self, Path, SITE_ANCHOR, SPEED_MPS};

/// Telemetry period.
pub const PERIOD: Duration = Duration::from_millis(500);
/// Seconds between the route and the hello; the core's countdown is set to the same.
pub const ETA_SECS: f64 = 10.0;

type Sink = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, WsMessage>;

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

pub fn telemetry(seq: u64, drone: &Drone, pose: &route::Pose, flight_secs: f64, t_ms: i64) -> Envelope {
    let gps = route::to_gps(pose, drone.station, SITE_ANCHOR);
    envelope(
        seq,
        Body::Telemetry(model::DroneTelemetry {
            drone_id: drone.id,
            station: drone.station as i32,
            gps: Some(gps.into()),
            heading_deg: pose.heading_deg,
            speed: pose.speed,
            battery_pct: route::battery_pct(flight_secs),
            t_ms,
        }),
    )
}

pub fn assign_route(seq: u64, waypoints: Vec<Gps>) -> Envelope {
    envelope(
        seq,
        Body::AssignRoute(viewer::AssignRoute {
            drone_id: FORMATION[1].id,
            waypoints: waypoints.into_iter().map(Into::into).collect(),
            r#loop: true,
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

/// Run the three drones until an uplink fails (returned as an error).
pub async fn run_drones(core: String, follow_route: bool) -> anyhow::Result<()> {
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
    log("drone", "드론 3대 업링크 연결, 대기 중 (관제탑에서 경로를 지정하면 출발)");

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
    let mut path = choose_path(&first, follow_route);
    log(
        "drone",
        &format!("경로 수신 ({}개 지점). {ETA_SECS:.0}초 뒤 현장 도착 예정", first.waypoints.len()),
    );

    // ---- approach: telemetry, no hello ----
    let mut tick = tokio::time::interval(PERIOD);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let started = tokio::time::Instant::now();
    loop {
        tokio::select! {
            _ = tick.tick() => {}
            Some(reason) = dead_rx.recv() => anyhow::bail!(reason),
        }
        let t = started.elapsed().as_secs_f64();
        let pose = route::approach_pose(&path, t, ETA_SECS);
        for u in &mut uplinks {
            let drone = u.drone;
            u.send(|seq| telemetry(seq, &drone, &pose, t, now_ms())).await?;
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
    log("drone", "현장 도착: drone_hello 전송, 경로 비행 시작");

    // ---- flight ----
    let mut leg_start = tokio::time::Instant::now();
    loop {
        tokio::select! {
            _ = tick.tick() => {}
            changed = route_rx.changed() => {
                changed.map_err(|_| anyhow::anyhow!("route channel closed"))?;
                if let Some(r) = route_rx.borrow_and_update().clone() {
                    if follow_route {
                        path = choose_path(&r, true);
                        leg_start = tokio::time::Instant::now();
                        log("drone", &format!("새 경로 ({}개 지점) 비행", r.waypoints.len()));
                    } else {
                        log("drone", "새 경로 수신. 데모 촬영 경로를 계속 비행합니다 (--follow-route 로 바꿀 수 있음)");
                    }
                }
                continue;
            }
            Some(reason) = dead_rx.recv() => anyhow::bail!(reason),
        }
        let flight = started.elapsed().as_secs_f64();
        let pose = path.pose_at(leg_start.elapsed().as_secs_f64(), SPEED_MPS);
        for u in &mut uplinks {
            let drone = u.drone;
            u.send(|seq| telemetry(seq, &drone, &pose, flight, now_ms())).await?;
        }
    }
}

/// The footage track, or the operator's waypoints with `--follow-route`.
pub fn choose_path(r: &viewer::AssignRoute, follow_route: bool) -> Path {
    if follow_route {
        let waypoints: Vec<Gps> = r.waypoints.iter().map(|g| Gps { lat: g.lat, lon: g.lon, alt: g.alt }).collect();
        if let Some(p) = Path::from_waypoints(&waypoints, SITE_ANCHOR, r.r#loop) {
            return p;
        }
        log("drone", "경로 지점이 둘 미만이라 데모 촬영 경로로 비행합니다");
    }
    Path::footage_track()
}

/// Watch `/control`: log mission phases, and with `auto` assign the footage
/// track once the tower socket is up. Returns when the socket fails.
pub async fn run_control_watch(core: String, auto: bool) -> anyhow::Result<()> {
    let url = format!("ws://{core}/control");
    let ws = connect(&url).await?;
    let (mut sink, mut stream) = ws.split();
    let mut assign_at = auto.then(|| tokio::time::Instant::now() + Duration::from_secs(3));
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
                let env = assign_route(1, route::footage_waypoints(SITE_ANCHOR));
                sink.send(WsMessage::Binary(env.encode_to_vec().into())).await?;
                log("auto", "--auto: 데모 촬영 경로를 배정했습니다");
            }
            frame = stream.next() => match frame {
                Some(Ok(WsMessage::Binary(b))) => {
                    if let Ok(Envelope { body: Some(Body::MissionStatus(s)), .. }) = Envelope::decode(&b[..])
                        && last_phase != Some(s.phase)
                    {
                        last_phase = Some(s.phase);
                        log("tower", &format!("미션: {} ({}) 드론 {}대", phase_name(s.phase), s.message, s.drones_online));
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
    fn telemetry_carries_station_gps() {
        let pose = Path::footage_track().pose_at(0.0, SPEED_MPS);
        let env = telemetry(1, &FORMATION[2], &pose, 0.0, 123);
        let Some(Body::Telemetry(t)) = env.body else { panic!() };
        assert_eq!(t.drone_id, 3);
        assert_eq!(t.station, model::DroneStation::Right as i32);
        assert_eq!(t.t_ms, 123);
        let gps = t.gps.unwrap();
        assert_eq!(gps.alt, route::ALTITUDE_M);
        assert!(gps.lat > SITE_ANCHOR.lat - 0.01 && gps.lat < SITE_ANCHOR.lat + 0.01);
        assert!((t.heading_deg - route::TRACK_BEARING_DEG).abs() < 1e-9);
    }

    #[test]
    fn route_round_trips_through_the_wire() {
        let env = assign_route(1, route::footage_waypoints(SITE_ANCHOR));
        let r = route_of(&env.encode_to_vec()).unwrap();
        assert_eq!(r.waypoints.len(), 2);
        assert!(r.r#loop);
        assert!(route_of(&hello(1, &FORMATION[0]).encode_to_vec()).is_none());
        assert!(route_of(b"\xff\xff").is_none());
    }

    #[test]
    fn follow_route_uses_the_waypoints() {
        let mut r = viewer::AssignRoute { drone_id: 2, waypoints: vec![], r#loop: true };
        assert_eq!(choose_path(&r, true), Path::footage_track());
        r.waypoints = route::footage_waypoints(SITE_ANCHOR).into_iter().map(Into::into).collect();
        assert_eq!(choose_path(&r, true).motion, route::Motion::PingPong);
        assert_eq!(choose_path(&r, false), Path::footage_track());
    }

    #[test]
    fn phases_have_names() {
        assert_eq!(phase_name(viewer::MissionPhase::AwaitingDrone as i32), "드론 연결 대기");
        assert_eq!(phase_name(99), "알 수 없음");
    }
}
