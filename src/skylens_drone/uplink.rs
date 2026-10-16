//! Optional uplink to the core (`ws /uplink`).
//!
//! When `SKYLENS_CORE_UPLINK` is set the bridge connects to the core as a
//! WebSocket client. Every frame is one binary `skylens.viewer.v1.Envelope`.
//!
//! - On connect it sends one `drone_hello` per configured formation station:
//!   that is the only signal the core counts as "drone arrived".
//! - On `assign_route` it builds the three-aircraft formation missions through
//!   the same code path as `POST /formation`, and uploads them to the DJI cloud
//!   when cloud delivery is configured.
//! - When the socket drops it reconnects with exponential backoff. The proxy in
//!   front of the core does not migrate open sockets on failover, so a dropped
//!   socket is the normal failover signal.
//!
//! The Lito X1 has no SDK, so the bridge never sends live telemetry.
//!
//! The core replays the current route to every new uplink connection. The
//! bridge remembers the last route it turned into missions and skips an
//! identical one, so a reconnect does not re-upload the same three missions.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, bail};
use futures_util::{SinkExt, StreamExt};
use prost::Message as _;
use serde_json::{Value, json};
use skylens_shared::proto::model::v1::DroneStation;
use skylens_shared::proto::viewer::v1::{AssignRoute, ComponentId, DroneHello, Envelope, envelope};
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message as WsMessage;

use crate::server::App;

pub const DEFAULT_STATIONS: &str = "left=1,center=2,right=3";
pub const DEFAULT_MODEL: &str = "DJI Lito X1 (DJI Fly waypoint mission)";
pub const DEFAULT_SPACING_M: f64 = 10.0;

/// One formation member announced to the core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Station {
    pub station: DroneStation,
    pub drone_id: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UplinkConfig {
    pub url: String,
    pub stations: Vec<Station>,
    pub spacing_m: f64,
    pub model: String,
    pub backoff_initial: Duration,
    pub backoff_max: Duration,
}

impl UplinkConfig {
    /// `None` when `SKYLENS_CORE_UPLINK` is unset or empty.
    ///
    /// - `SKYLENS_UPLINK_STATIONS`: `station[=droneId]` list, default
    ///   `left=1,center=2,right=3`. A station without an id gets its 1-based position.
    /// - `SKYLENS_FORMATION_SPACING_M`: triangle side for routes from the core, default 10.
    /// - `SKYLENS_UPLINK_MODEL`: model string shown on the operator panel.
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> anyhow::Result<Option<Self>> {
        let var = |k: &str| get(k).map(|v| v.trim().to_owned()).filter(|v| !v.is_empty());
        let Some(url) = var("SKYLENS_CORE_UPLINK") else {
            return Ok(None);
        };
        let stations =
            parse_stations(&var("SKYLENS_UPLINK_STATIONS").unwrap_or_else(|| DEFAULT_STATIONS.into()))?;
        let spacing_m = match var("SKYLENS_FORMATION_SPACING_M") {
            Some(v) => {
                let s: f64 = v.parse().with_context(|| format!("SKYLENS_FORMATION_SPACING_M={v}"))?;
                if !s.is_finite() || s <= 0.0 {
                    bail!("SKYLENS_FORMATION_SPACING_M must be a positive number");
                }
                s
            }
            None => DEFAULT_SPACING_M,
        };
        Ok(Some(Self {
            url,
            stations,
            spacing_m,
            model: var("SKYLENS_UPLINK_MODEL").unwrap_or_else(|| DEFAULT_MODEL.into()),
            backoff_initial: Duration::from_millis(500),
            backoff_max: Duration::from_secs(10),
        }))
    }
}

fn station_of(name: &str) -> Option<DroneStation> {
    match name.trim().to_ascii_lowercase().as_str() {
        "left" => Some(DroneStation::Left),
        "center" | "centre" => Some(DroneStation::Center),
        "right" => Some(DroneStation::Right),
        _ => None,
    }
}

/// Parse `left=1,center=2,right=3` (ids optional).
pub fn parse_stations(spec: &str) -> anyhow::Result<Vec<Station>> {
    let mut out: Vec<Station> = Vec::new();
    for (index, item) in spec.split(',').map(str::trim).filter(|s| !s.is_empty()).enumerate() {
        let (name, id) = match item.split_once('=') {
            Some((name, id)) => {
                let id: u32 = id.trim().parse().with_context(|| format!("bad drone id in '{item}'"))?;
                (name, id)
            }
            None => (item, index as u32 + 1),
        };
        let station = station_of(name)
            .with_context(|| format!("unknown station '{name}' (left, center, right)"))?;
        if out.iter().any(|s| s.station == station || s.drone_id == id) {
            bail!("station list repeats a station or drone id: '{spec}'");
        }
        out.push(Station { station, drone_id: id });
    }
    if out.is_empty() {
        bail!("SKYLENS_UPLINK_STATIONS names no station");
    }
    Ok(out)
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

fn make_envelope(seq: &mut u64, body: envelope::Body) -> Envelope {
    *seq += 1;
    Envelope {
        seq: *seq,
        origin_ts_ms: now_ms(),
        from: ComponentId::Drone as i32,
        body: Some(body),
    }
}

/// One `drone_hello` per station, in configuration order.
pub fn hello_envelopes(cfg: &UplinkConfig, seq: &mut u64) -> Vec<Envelope> {
    cfg.stations
        .iter()
        .map(|s| {
            make_envelope(
                seq,
                envelope::Body::DroneHello(DroneHello {
                    drone_id: s.drone_id,
                    station: s.station as i32,
                    model: cfg.model.clone(),
                }),
            )
        })
        .collect()
}

/// The `POST /formation` body equivalent to an `assign_route` from the core.
///
/// The core sends only waypoints and the loop flag, so the flight parameters
/// are the bridge defaults (30 m AGL, 2 m/s, take-off altitude inferred from
/// the first waypoint).
pub fn route_payload(route: &AssignRoute, spacing_m: f64, deliver_cloud: bool) -> Value {
    let waypoints: Vec<Value> = route
        .waypoints
        .iter()
        .map(|g| json!({"lat": g.lat, "lon": g.lon, "alt": g.alt}))
        .collect();
    let mut payload = json!({
        "name": format!("core-drone{}", route.drone_id),
        "droneId": route.drone_id,
        "loop": route.r#loop,
        "formation": {"spacingM": spacing_m},
        "waypoints": waypoints,
    });
    if deliver_cloud {
        payload["deliver"] = json!("cloud");
    }
    payload
}

/// Exponential backoff between reconnect attempts.
#[derive(Debug, Clone)]
pub struct Backoff {
    initial: Duration,
    max: Duration,
    next: Duration,
}

impl Backoff {
    pub fn new(initial: Duration, max: Duration) -> Self {
        Self { initial, max, next: initial }
    }

    pub fn reset(&mut self) {
        self.next = self.initial;
    }

    pub fn next_delay(&mut self) -> Duration {
        let delay = self.next;
        self.next = (self.next * 2).min(self.max);
        delay
    }
}

/// The URL without credentials or query string, for logs.
fn redact_url(url: &str) -> String {
    let no_query = url.split(['?', '#']).next().unwrap_or(url);
    let Some((scheme, rest)) = no_query.split_once("://") else {
        return no_query.to_owned();
    };
    let (authority, path) = match rest.find('/') {
        Some(i) => rest.split_at(i),
        None => (rest, ""),
    };
    let host = authority.rsplit('@').next().unwrap_or(authority);
    format!("{scheme}://{host}{path}")
}

/// Remembers the last route turned into missions, and serialises processing.
#[derive(Default)]
pub struct RouteState {
    last: Mutex<Option<Vec<u8>>>,
}

/// Build (and maybe upload) the formation for one route, unless it is the
/// route already handled. Returns whether missions were built.
pub async fn process_route(app: &App, state: &RouteState, route: AssignRoute, spacing_m: f64) -> bool {
    let fingerprint = route.encode_to_vec();
    let mut last = state.last.lock().await;
    if last.as_deref() == Some(&fingerprint[..]) {
        tracing::info!("[uplink] route for drone {} unchanged; missions already built", route.drone_id);
        return false;
    }
    let payload = route_payload(&route, spacing_m, app.cloud.is_some());
    match app.formation(&payload).await {
        Ok(resp) => {
            *last = Some(fingerprint);
            for m in &resp.members {
                let cloud = m
                    .cloud
                    .as_ref()
                    .map(|c| format!(", cloud {} verified={}", c.mission_uuid, c.verified))
                    .unwrap_or_default();
                tracing::info!("[uplink] {} -> {}{}", m.station, m.file_name, cloud);
            }
            true
        }
        Err(e) => {
            tracing::warn!("[uplink] could not build missions for assign_route: {e}");
            false
        }
    }
}

/// Connect, announce, serve one socket until it closes or fails.
async fn session(
    app: &Arc<App>,
    cfg: &UplinkConfig,
    state: &Arc<RouteState>,
    seq: &mut u64,
) -> anyhow::Result<()> {
    let (ws, _) = tokio_tungstenite::connect_async(cfg.url.as_str())
        .await
        .context("connect failed")?;
    tracing::info!("[uplink] connected to {}", redact_url(&cfg.url));
    let (mut sink, mut stream) = ws.split();
    for hello in hello_envelopes(cfg, seq) {
        sink.send(WsMessage::Binary(hello.encode_to_vec().into())).await.context("send drone_hello")?;
    }
    while let Some(frame) = stream.next().await {
        match frame.context("read")? {
            WsMessage::Binary(bytes) => match Envelope::decode(&bytes[..]) {
                Ok(Envelope { body: Some(envelope::Body::AssignRoute(route)), .. }) => {
                    tracing::info!(
                        "[uplink] assign_route for drone {}: {} waypoint(s), loop={}",
                        route.drone_id,
                        route.waypoints.len(),
                        route.r#loop
                    );
                    let (app, state, spacing) = (app.clone(), state.clone(), cfg.spacing_m);
                    tokio::spawn(async move {
                        process_route(&app, &state, route, spacing).await;
                    });
                }
                Ok(Envelope { body: Some(envelope::Body::ManualControl(_)), .. }) => {
                    tracing::debug!("[uplink] manual_control ignored: the Lito X1 has no stick API");
                }
                Ok(_) => {}
                Err(e) => tracing::warn!("[uplink] undecodable frame: {e}"),
            },
            WsMessage::Text(_) => tracing::warn!("[uplink] ignoring a text frame"),
            WsMessage::Close(_) => break,
            _ => {}
        }
    }
    Ok(())
}

/// Run the uplink forever, reconnecting with backoff.
pub async fn run(app: Arc<App>, cfg: UplinkConfig) {
    let state = Arc::new(RouteState::default());
    let mut backoff = Backoff::new(cfg.backoff_initial, cfg.backoff_max);
    let mut seq = 0u64;
    loop {
        let started = tokio::time::Instant::now();
        match session(&app, &cfg, &state, &mut seq).await {
            Ok(()) => tracing::warn!("[uplink] closed by {}", redact_url(&cfg.url)),
            Err(e) => tracing::warn!("[uplink] {}: {e:#}", redact_url(&cfg.url)),
        }
        // A session that lived a while was healthy; start over from the short delay.
        if started.elapsed() > cfg.backoff_max {
            backoff.reset();
        }
        tokio::time::sleep(backoff.next_delay()).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mission::tests::write_lito_template;
    use crate::server::Settings;
    use skylens_shared::proto::model::v1::Gps;
    use std::collections::HashMap;
    use std::path::Path;

    fn route() -> AssignRoute {
        AssignRoute {
            drone_id: 1,
            waypoints: vec![
                Gps { lat: 36.3685, lon: 127.3475, alt: 80.0 },
                Gps { lat: 36.3687, lon: 127.3477, alt: 85.0 },
            ],
            r#loop: false,
        }
    }

    fn app_in(dir: &Path) -> Arc<App> {
        let template = dir.join("lito-template.kmz");
        write_lito_template(&template);
        let settings = Settings {
            host: "127.0.0.1".into(),
            port: 8091,
            template,
            output: dir.join("missions"),
            install_target: None,
            allowed_origins: Default::default(),
        };
        Arc::new(App::new(settings, None))
    }

    fn kmz_count(dir: &Path) -> usize {
        std::fs::read_dir(dir.join("missions"))
            .map(|rd| rd.filter(|e| e.as_ref().unwrap().path().extension().is_some_and(|x| x == "kmz")).count())
            .unwrap_or(0)
    }

    #[test]
    fn config_is_off_without_url_and_parses_overrides() {
        assert_eq!(UplinkConfig::from_lookup(|_| None).unwrap(), None);
        let vars: HashMap<&str, &str> = [
            ("SKYLENS_CORE_UPLINK", "ws://127.0.0.1:8082/uplink"),
            ("SKYLENS_UPLINK_STATIONS", "right=7, left"),
            ("SKYLENS_FORMATION_SPACING_M", "12.5"),
        ]
        .into_iter()
        .collect();
        let cfg = UplinkConfig::from_lookup(|k| vars.get(k).map(|v| v.to_string())).unwrap().unwrap();
        assert_eq!(cfg.url, "ws://127.0.0.1:8082/uplink");
        assert_eq!(
            cfg.stations,
            vec![
                Station { station: DroneStation::Right, drone_id: 7 },
                Station { station: DroneStation::Left, drone_id: 2 },
            ]
        );
        assert_eq!(cfg.spacing_m, 12.5);
        assert_eq!(cfg.model, DEFAULT_MODEL);
    }

    #[test]
    fn default_stations_are_the_three_formation_members() {
        let s = parse_stations(DEFAULT_STATIONS).unwrap();
        assert_eq!(
            s.iter().map(|s| (s.station, s.drone_id)).collect::<Vec<_>>(),
            vec![(DroneStation::Left, 1), (DroneStation::Center, 2), (DroneStation::Right, 3)]
        );
        assert!(parse_stations("left=1,left=2").is_err());
        assert!(parse_stations("left=1,right=1").is_err());
        assert!(parse_stations("top=1").is_err());
        assert!(parse_stations(" , ").is_err());
    }

    #[test]
    fn hellos_are_binary_envelopes_from_the_drone() {
        let cfg = UplinkConfig::from_lookup(|k| (k == "SKYLENS_CORE_UPLINK").then(|| "ws://x/uplink".into()))
            .unwrap()
            .unwrap();
        let mut seq = 0;
        let hellos = hello_envelopes(&cfg, &mut seq);
        assert_eq!(hellos.len(), 3);
        assert_eq!(seq, 3);
        for (i, env) in hellos.iter().enumerate() {
            let decoded = Envelope::decode(env.encode_to_vec().as_slice()).unwrap();
            assert_eq!(decoded.seq, i as u64 + 1);
            assert_eq!(decoded.from, ComponentId::Drone as i32);
            assert!(decoded.origin_ts_ms > 0);
            let Some(envelope::Body::DroneHello(hello)) = decoded.body else { panic!("not a hello") };
            assert_eq!(hello.drone_id, cfg.stations[i].drone_id);
            assert_eq!(hello.station, cfg.stations[i].station as i32);
            assert_eq!(hello.model, DEFAULT_MODEL);
        }
    }

    #[test]
    fn route_payload_matches_the_formation_request() {
        let mut r = route();
        r.r#loop = true;
        let p = route_payload(&r, 10.0, true);
        assert_eq!(p["name"], "core-drone1");
        assert_eq!(p["loop"], true);
        assert_eq!(p["deliver"], "cloud");
        assert_eq!(p["formation"]["spacingM"], 10.0);
        assert_eq!(p["waypoints"][1], json!({"lat": 36.3687, "lon": 127.3477, "alt": 85.0}));
        assert!(route_payload(&route(), 10.0, false).get("deliver").is_none());
    }

    #[test]
    fn backoff_doubles_up_to_the_cap_and_resets() {
        let mut b = Backoff::new(Duration::from_millis(500), Duration::from_secs(2));
        let seq: Vec<u128> = (0..5).map(|_| b.next_delay().as_millis()).collect();
        assert_eq!(seq, vec![500, 1000, 2000, 2000, 2000]);
        b.reset();
        assert_eq!(b.next_delay(), Duration::from_millis(500));
    }

    #[test]
    fn redacts_credentials_and_query() {
        assert_eq!(redact_url("ws://u:p@core:8082/uplink?token=x"), "ws://core:8082/uplink");
        assert_eq!(redact_url("ws://127.0.0.1:8082/uplink"), "ws://127.0.0.1:8082/uplink");
    }

    #[tokio::test]
    async fn same_route_is_built_once() {
        let dir = tempfile::tempdir().unwrap();
        let app = app_in(dir.path());
        let state = RouteState::default();
        assert!(process_route(&app, &state, route(), 10.0).await);
        assert_eq!(kmz_count(dir.path()), 3);
        assert!(!process_route(&app, &state, route(), 10.0).await);
        let mut changed = route();
        changed.r#loop = true;
        // Same second, same file names: the count stays 3 but the build runs.
        assert!(process_route(&app, &state, changed, 10.0).await);
    }

    #[tokio::test]
    async fn failed_build_is_retried_on_replay() {
        let dir = tempfile::tempdir().unwrap();
        let app = app_in(dir.path());
        std::fs::remove_file(dir.path().join("lito-template.kmz")).unwrap();
        let state = RouteState::default();
        assert!(!process_route(&app, &state, route(), 10.0).await);
        write_lito_template(&dir.path().join("lito-template.kmz"));
        assert!(process_route(&app, &state, route(), 10.0).await);
    }

    /// A fake core: drops the first socket after the hellos, then sends a route
    /// on the second. The bridge must reconnect, re-announce, and build missions.
    #[tokio::test]
    async fn reconnects_announces_and_builds_routes_from_the_core() {
        let dir = tempfile::tempdir().unwrap();
        let app = app_in(dir.path());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let mut cfg = UplinkConfig::from_lookup(|k| {
            (k == "SKYLENS_CORE_UPLINK").then(|| format!("ws://{addr}/uplink"))
        })
        .unwrap()
        .unwrap();
        cfg.backoff_initial = Duration::from_millis(20);
        cfg.backoff_max = Duration::from_millis(100);
        tokio::spawn(run(app, cfg));

        async fn read_hellos(
            ws: &mut tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
        ) -> Vec<DroneHello> {
            let mut out = Vec::new();
            while out.len() < 3 {
                let WsMessage::Binary(bytes) = ws.next().await.unwrap().unwrap() else { continue };
                let env = Envelope::decode(&bytes[..]).unwrap();
                let Some(envelope::Body::DroneHello(h)) = env.body else { panic!("expected hello") };
                out.push(h);
            }
            out
        }

        let run_core = async {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
            assert_eq!(read_hellos(&mut ws).await.len(), 3);
            drop(ws); // failover: the socket just goes away

            let (tcp, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(tcp).await.unwrap();
            let hellos = read_hellos(&mut ws).await;
            let stations: Vec<i32> = hellos.iter().map(|h| h.station).collect();
            assert_eq!(
                stations,
                vec![DroneStation::Left as i32, DroneStation::Center as i32, DroneStation::Right as i32]
            );
            let assign = Envelope {
                seq: 1,
                origin_ts_ms: now_ms(),
                from: ComponentId::Core as i32,
                body: Some(envelope::Body::AssignRoute(route())),
            };
            ws.send(WsMessage::Binary(assign.encode_to_vec().into())).await.unwrap();
            ws
        };
        let _ws = tokio::time::timeout(Duration::from_secs(10), run_core).await.expect("core timed out");

        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while kmz_count(dir.path()) < 3 {
            assert!(tokio::time::Instant::now() < deadline, "missions were not built");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}
