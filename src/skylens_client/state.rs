//! Screen state: everything the core told this screen, reduced from envelopes.
//!
//! `Store::apply` is the only way viewer events change state. It returns a
//! [`Changed`] that says which panels to re-render and what the 3D view must do,
//! so the UI binding never diffs state itself.

use crate::chunks::{ChunkBook, SceneOp};
use crate::protocol::{Body, SeqTracker};
use skylens_shared::proto::model::v1 as model;
use skylens_shared::proto::payload::v1 as payload;
use skylens_shared::proto::viewer::v1 as viewer;
use std::collections::{BTreeMap, BTreeSet};

/// Telemetry older than this marks a drone row as stale (legacy STALE_AFTER).
pub const STALE_AFTER_MS: f64 = 3_000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkState {
    Connecting,
    Connected,
    Disconnected,
}

#[derive(Debug, Clone)]
pub struct Link {
    pub state: LinkState,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct Drone {
    pub id: u32,
    pub station: model::DroneStation,
    pub model: String,
    /// A `drone_hello` was received: the only signal that the drone has arrived.
    pub hello: bool,
    pub telemetry: Option<model::DroneTelemetry>,
    pub received_at_ms: f64,
}

impl Drone {
    pub fn stale(&self, now_ms: f64) -> bool {
        self.telemetry.is_some() && now_ms - self.received_at_ms > STALE_AFTER_MS
    }

    pub fn gps(&self) -> Option<model::Gps> {
        self.telemetry.as_ref().and_then(|t| t.gps.clone())
    }

    /// Formation order: left, centre, right, then unknown.
    fn order(&self) -> (u8, u32) {
        let rank = match self.station {
            model::DroneStation::Left => 0,
            model::DroneStation::Center => 1,
            model::DroneStation::Right => 2,
            model::DroneStation::Unspecified => 3,
        };
        (rank, self.id)
    }
}

#[derive(Debug, Clone)]
pub struct Mission {
    pub status: viewer::MissionStatus,
    pub received_at_ms: f64,
}

/// Which parts of the screen must be redrawn after an update.
#[derive(Debug, Default)]
pub struct Changed {
    pub link: bool,
    pub mission: bool,
    pub drones: bool,
    pub route: bool,
    pub camera: bool,
    pub server: bool,
    pub hops: bool,
    pub detections: bool,
    /// Detection ids that became visible with this update (for toasts).
    pub revealed: Vec<String>,
    pub scene: Vec<SceneOp>,
    /// The session restarted: the 3D view must drop everything it holds.
    pub reset: bool,
}

impl Changed {
    pub fn any(&self) -> bool {
        self.link
            || self.mission
            || self.drones
            || self.route
            || self.camera
            || self.server
            || self.hops
            || self.detections
            || !self.scene.is_empty()
            || self.reset
    }

    pub fn merge(&mut self, other: Changed) {
        self.link |= other.link;
        self.mission |= other.mission;
        self.drones |= other.drones;
        self.route |= other.route;
        self.camera |= other.camera;
        self.server |= other.server;
        self.hops |= other.hops;
        self.detections |= other.detections;
        self.revealed.extend(other.revealed);
        if other.reset {
            self.scene.clear();
        }
        self.scene.extend(other.scene);
        self.reset |= other.reset;
    }
}

/// Detections keyed by their stable id; an update replaces the previous one.
#[derive(Debug, Default)]
pub struct Detections {
    by_id: BTreeMap<String, payload::Detection>,
    revealed: BTreeSet<String>,
}

impl Detections {
    pub fn merge(&mut self, batch: Vec<payload::Detection>) {
        for d in batch {
            if !d.id.is_empty() {
                self.by_id.insert(d.id.clone(), d);
            }
        }
    }

    /// Detections whose segment has arrived, in id order.
    pub fn visible<'a>(&'a self, arrived: &'a BTreeSet<u32>) -> impl Iterator<Item = &'a payload::Detection> {
        self.by_id.values().filter(move |d| arrived.contains(&d.segment))
    }

    /// Marks visible detections as revealed and returns those revealed just now.
    pub fn reveal(&mut self, arrived: &BTreeSet<u32>) -> Vec<String> {
        let fresh: Vec<String> = self
            .by_id
            .values()
            .filter(|d| arrived.contains(&d.segment) && !self.revealed.contains(&d.id))
            .map(|d| d.id.clone())
            .collect();
        self.revealed.extend(fresh.iter().cloned());
        fresh
    }

    pub fn get(&self, id: &str) -> Option<&payload::Detection> {
        self.by_id.get(id)
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

pub struct Store {
    pub link: Link,
    pub mission: Option<Mission>,
    pub drones: BTreeMap<u32, Drone>,
    pub active_drone: Option<u32>,
    /// The operator picked the active drone; stop steering it to the centre.
    pub operator_selected: bool,
    pub route: Option<viewer::AssignRoute>,
    pub camera_feeds: BTreeMap<u32, viewer::CameraFeed>,
    pub hops: BTreeMap<String, viewer::LinkStatus>,
    pub server: Option<viewer::ServerStatus>,
    pub chunks: ChunkBook,
    pub detections: Detections,
    pub seq: SeqTracker,
    pub frames: u64,
    pub bad_frames: u64,
}

impl Default for Store {
    fn default() -> Self {
        Self::new()
    }
}

impl Store {
    pub fn new() -> Self {
        Self {
            link: Link {
                state: LinkState::Connecting,
                detail: "코어 연결 대기 중".into(),
            },
            mission: None,
            drones: BTreeMap::new(),
            active_drone: None,
            operator_selected: false,
            route: None,
            camera_feeds: BTreeMap::new(),
            hops: BTreeMap::new(),
            server: None,
            chunks: ChunkBook::new(),
            detections: Detections::default(),
            seq: SeqTracker::default(),
            frames: 0,
            bad_frames: 0,
        }
    }

    /// Link state from the transport. A new connection resets the session: the
    /// core replays what it holds to every screen that joins, so keeping the old
    /// chunks would show them twice.
    pub fn set_link(&mut self, state: LinkState, detail: impl Into<String>) -> Changed {
        let mut changed = Changed {
            link: true,
            ..Default::default()
        };
        if state == LinkState::Connected && self.link.state != LinkState::Connected {
            self.reset_session();
            changed = Changed {
                link: true,
                mission: true,
                drones: true,
                route: true,
                camera: true,
                server: true,
                hops: true,
                detections: true,
                reset: true,
                scene: vec![SceneOp::RebuildPoints(vec![]), SceneOp::ResetMesh(vec![])],
                revealed: vec![],
            };
        }
        if state == LinkState::Disconnected && self.link.state == LinkState::Connected {
            // Telemetry means nothing once the link is gone: drop the drones rather
            // than leave them frozen over the map. Reconstruction stays on screen.
            self.drones.clear();
            self.active_drone = None;
            self.camera_feeds.clear();
            changed.drones = true;
            changed.camera = true;
            changed.mission = true;
        }
        self.link = Link {
            state,
            detail: detail.into(),
        };
        changed
    }

    fn reset_session(&mut self) {
        self.mission = None;
        self.drones.clear();
        self.active_drone = None;
        self.operator_selected = false;
        self.route = None;
        self.camera_feeds.clear();
        self.hops.clear();
        self.server = None;
        self.chunks = ChunkBook::new();
        self.detections = Detections::default();
        self.seq.reset();
    }

    /// Applies one raw WebSocket frame.
    pub fn apply_frame(&mut self, bytes: &[u8], now_ms: f64) -> Changed {
        match crate::protocol::decode_envelope(bytes) {
            Ok(env) => self.apply(env, now_ms),
            Err(_) => {
                self.bad_frames += 1;
                Changed::default()
            }
        }
    }

    pub fn apply(&mut self, env: viewer::Envelope, now_ms: f64) -> Changed {
        self.frames += 1;
        self.seq.observe(env.seq);
        let mut changed = Changed::default();
        let Some(body) = env.body else {
            return changed;
        };
        match body {
            Body::Site(_) => {}
            Body::DroneHello(h) => {
                let d = self.drone_entry(h.drone_id, now_ms);
                d.hello = true;
                d.station = h.station();
                d.model = h.model;
                self.pick_active();
                changed.drones = true;
                changed.camera = true;
            }
            Body::Telemetry(t) => {
                let d = self.drone_entry(t.drone_id, now_ms);
                if t.station() != model::DroneStation::Unspecified {
                    d.station = t.station();
                }
                d.telemetry = Some(t);
                d.received_at_ms = now_ms;
                self.pick_active();
                changed.drones = true;
                changed.camera = true;
            }
            Body::AssignRoute(r) => {
                self.route = (!r.waypoints.is_empty()).then_some(r);
                changed.route = true;
            }
            Body::ManualControl(_) => {}
            Body::MissionStatus(m) => {
                self.mission = Some(Mission {
                    status: m,
                    received_at_ms: now_ms,
                });
                changed.mission = true;
                changed.server = true;
            }
            Body::Chunk(c) => {
                let applied = self.chunks.apply(&c);
                changed.scene = applied.ops;
                if let Some(dets) = applied.detections {
                    self.detections.merge(dets);
                    changed.detections = true;
                }
                if applied.newly_arrived.is_some() {
                    changed.detections = true;
                }
                if changed.detections {
                    changed.revealed = self.detections.reveal(self.chunks.arrived());
                }
                changed.server = true;
            }
            Body::CameraFeed(f) => {
                self.camera_feeds.insert(f.drone_id, f);
                changed.camera = true;
            }
            Body::LinkStatus(l) => {
                let prev = self.hops.insert(l.hop.clone(), l.clone());
                changed.hops = prev.as_ref() != Some(&l);
            }
            Body::ServerStatus(s) => {
                self.server = Some(s);
                changed.server = true;
            }
        }
        changed
    }

    fn drone_entry(&mut self, id: u32, now_ms: f64) -> &mut Drone {
        self.drones.entry(id).or_insert_with(|| Drone {
            id,
            station: model::DroneStation::Unspecified,
            model: String::new(),
            hello: false,
            telemetry: None,
            received_at_ms: now_ms,
        })
    }

    /// The centre drone is active until the operator picks one.
    fn pick_active(&mut self) {
        if self.operator_selected && self.active_drone.is_some_and(|id| self.drones.contains_key(&id)) {
            return;
        }
        let centre = self
            .drones
            .values()
            .find(|d| d.station == model::DroneStation::Center)
            .map(|d| d.id);
        let keep = self.active_drone.filter(|id| self.drones.contains_key(id));
        self.active_drone = centre.or(keep).or_else(|| self.drones.keys().next().copied());
    }

    /// Operator selection (telemetry row click, digit keys, Tab).
    pub fn select_drone(&mut self, id: u32) -> bool {
        if !self.drones.contains_key(&id) {
            return false;
        }
        self.operator_selected = true;
        let changed = self.active_drone != Some(id);
        self.active_drone = Some(id);
        changed
    }

    /// Next drone in formation order (Tab).
    pub fn cycle_drone(&mut self) -> bool {
        let ids: Vec<u32> = self.drones_in_order().iter().map(|d| d.id).collect();
        if ids.is_empty() {
            return false;
        }
        let next = match self.active_drone.and_then(|a| ids.iter().position(|i| *i == a)) {
            Some(i) => ids[(i + 1) % ids.len()],
            None => ids[0],
        };
        self.select_drone(next)
    }

    pub fn drones_in_order(&self) -> Vec<&Drone> {
        let mut v: Vec<&Drone> = self.drones.values().collect();
        v.sort_by_key(|d| d.order());
        v
    }

    pub fn active(&self) -> Option<&Drone> {
        self.active_drone.and_then(|id| self.drones.get(&id))
    }

    /// The drone that carries a route: the formation centre, else the active one.
    pub fn route_leader(&self) -> Option<u32> {
        self.drones
            .values()
            .find(|d| d.station == model::DroneStation::Center)
            .map(|d| d.id)
            .or(self.active_drone)
    }

    /// Seconds left while the mission is time-bound, counted down locally from the
    /// last snapshot the core sent.
    pub fn eta_left_s(&self, now_ms: f64) -> Option<f64> {
        let m = self.mission.as_ref()?;
        let eta = m.status.eta_seconds?;
        Some((eta - (now_ms - m.received_at_ms) / 1000.0).max(0.0))
    }

    pub fn phase(&self) -> viewer::MissionPhase {
        self.mission
            .as_ref()
            .map(|m| m.status.phase())
            .unwrap_or(viewer::MissionPhase::Idle)
    }

    /// Camera feed for a drone, if the core announced one.
    pub fn feed_for(&self, id: u32) -> Option<&viewer::CameraFeed> {
        self.camera_feeds.get(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pointblitz::io::chunk::{encode_with, FLAG_LAST_IN_GENERATION};
    use pointblitz::io::Point;
    use prost::Message as _;

    fn env(body: Body, seq: u64) -> viewer::Envelope {
        viewer::Envelope {
            seq,
            origin_ts_ms: 0,
            from: viewer::ComponentId::Core as i32,
            body: Some(body),
        }
    }

    fn gps() -> model::Gps {
        model::Gps { lat: 36.37, lon: 127.35, alt: 60.0 }
    }

    fn telemetry(id: u32, station: model::DroneStation) -> Body {
        Body::Telemetry(model::DroneTelemetry {
            drone_id: id,
            station: station as i32,
            gps: Some(gps()),
            heading_deg: 90.0,
            speed: 5.0,
            battery_pct: 80.0,
            t_ms: 0,
        })
    }

    fn points_chunk(segment: u32) -> Body {
        let pts = vec![Point { position: [0.0; 3], color: [1, 2, 3] }];
        Body::Chunk(model::Chunk {
            kind: model::ChunkKind::Points as i32,
            segment,
            level: model::Level::Preview as i32,
            payload: encode_with(0, 0, FLAG_LAST_IN_GENERATION, &pts, 12),
            ..Default::default()
        })
    }

    fn detections_chunk(ids: &[(&str, u32)]) -> Body {
        let batch = payload::DetectionBatch {
            detections: ids
                .iter()
                .map(|(id, seg)| payload::Detection {
                    id: (*id).into(),
                    segment: *seg,
                    category: payload::DetectionCategory::Person as i32,
                    gps: Some(gps()),
                    confidence: 0.9,
                    ..Default::default()
                })
                .collect(),
        };
        Body::Chunk(model::Chunk {
            kind: model::ChunkKind::Detections as i32,
            payload: batch.encode_to_vec(),
            ..Default::default()
        })
    }

    fn connected() -> Store {
        let mut s = Store::new();
        s.set_link(LinkState::Connected, "연결됨");
        s
    }

    #[test]
    fn detections_wait_for_their_segment() {
        let mut s = connected();
        let c = s.apply(env(detections_chunk(&[("a", 1), ("b", 2)]), 1), 0.0);
        assert!(c.detections);
        assert!(c.revealed.is_empty());
        assert_eq!(s.detections.visible(s.chunks.arrived()).count(), 0);

        let c = s.apply(env(points_chunk(1), 2), 0.0);
        assert_eq!(c.revealed, vec!["a".to_string()]);
        let visible: Vec<&str> = s.detections.visible(s.chunks.arrived()).map(|d| d.id.as_str()).collect();
        assert_eq!(visible, vec!["a"]);

        // An update of a revealed detection is not revealed again.
        let c = s.apply(env(detections_chunk(&[("a", 1)]), 3), 0.0);
        assert!(c.revealed.is_empty());
    }

    #[test]
    fn hello_marks_arrival_and_centre_becomes_active() {
        let mut s = connected();
        s.apply(env(telemetry(1, model::DroneStation::Left), 1), 0.0);
        assert_eq!(s.active_drone, Some(1));
        assert!(!s.drones[&1].hello);
        s.apply(
            env(
                Body::DroneHello(viewer::DroneHello {
                    drone_id: 2,
                    station: model::DroneStation::Center as i32,
                    model: "Lito X1".into(),
                }),
                2,
            ),
            0.0,
        );
        assert!(s.drones[&2].hello);
        assert_eq!(s.active_drone, Some(2));
        assert_eq!(s.route_leader(), Some(2));

        // Operator choice sticks.
        assert!(s.select_drone(1));
        s.apply(env(telemetry(2, model::DroneStation::Center), 3), 0.0);
        assert_eq!(s.active_drone, Some(1));
        assert_eq!(s.route_leader(), Some(2));
    }

    #[test]
    fn drones_list_in_formation_order_and_tab_cycles() {
        let mut s = connected();
        s.apply(env(telemetry(7, model::DroneStation::Right), 1), 0.0);
        s.apply(env(telemetry(5, model::DroneStation::Center), 2), 0.0);
        s.apply(env(telemetry(9, model::DroneStation::Left), 3), 0.0);
        let order: Vec<u32> = s.drones_in_order().iter().map(|d| d.id).collect();
        assert_eq!(order, vec![9, 5, 7]);
        assert_eq!(s.active_drone, Some(5));
        s.cycle_drone();
        assert_eq!(s.active_drone, Some(7));
        s.cycle_drone();
        assert_eq!(s.active_drone, Some(9));
    }

    #[test]
    fn telemetry_goes_stale() {
        let mut s = connected();
        s.apply(env(telemetry(1, model::DroneStation::Center), 1), 1_000.0);
        assert!(!s.drones[&1].stale(2_000.0));
        assert!(s.drones[&1].stale(4_500.0));
    }

    #[test]
    fn eta_counts_down_from_the_snapshot() {
        let mut s = connected();
        s.apply(
            env(
                Body::MissionStatus(viewer::MissionStatus {
                    phase: viewer::MissionPhase::AwaitingDrone as i32,
                    message: "대기".into(),
                    drones_online: 0,
                    eta_seconds: Some(30.0),
                }),
                1,
            ),
            10_000.0,
        );
        assert_eq!(s.phase(), viewer::MissionPhase::AwaitingDrone);
        assert_eq!(s.eta_left_s(20_000.0), Some(20.0));
        assert_eq!(s.eta_left_s(60_000.0), Some(0.0));
    }

    #[test]
    fn reconnect_resets_the_session() {
        let mut s = connected();
        s.apply(env(points_chunk(0), 1), 0.0);
        s.apply(env(telemetry(1, model::DroneStation::Center), 2), 0.0);
        let c = s.set_link(LinkState::Disconnected, "끊김");
        assert!(c.drones);
        assert!(s.drones.is_empty());
        // Reconstruction stays visible while disconnected.
        assert_eq!(s.chunks.visible_points().len(), 1);

        let c = s.set_link(LinkState::Connected, "연결됨");
        assert!(c.reset);
        assert!(s.chunks.visible_points().is_empty());
        assert!(matches!(c.scene[..], [SceneOp::RebuildPoints(_), SceneOp::ResetMesh(_)]));
    }

    #[test]
    fn route_echo_is_stored_and_empty_route_clears() {
        let mut s = connected();
        let r = viewer::AssignRoute { drone_id: 1, waypoints: vec![gps(), gps()], r#loop: false };
        s.apply(env(Body::AssignRoute(r), 1), 0.0);
        assert_eq!(s.route.as_ref().unwrap().waypoints.len(), 2);
        s.apply(env(Body::AssignRoute(viewer::AssignRoute::default()), 2), 0.0);
        assert!(s.route.is_none());
    }

    #[test]
    fn link_status_changes_are_detected() {
        let mut s = connected();
        let l = viewer::LinkStatus { hop: "proxy->core".into(), connected: true, latency_ms: None, mbps: None };
        assert!(s.apply(env(Body::LinkStatus(l.clone()), 1), 0.0).hops);
        assert!(!s.apply(env(Body::LinkStatus(l), 2), 0.0).hops);
    }

    #[test]
    fn bad_frames_are_counted() {
        let mut s = connected();
        let c = s.apply_frame(&[0xff, 0x00, 0x13], 0.0);
        assert!(!c.any());
        assert_eq!(s.bad_frames, 1);
    }
}
