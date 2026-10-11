//! Telemetry in, drone tracks out (legacy `drones/telemetryFleet.ts`). No local motion model:
//! positions come only from `DroneTelemetry`, eased on screen.
//!
//! The selection rule is the legacy one: until the operator picks an aircraft in the fleet list,
//! the centre drone is selected as soon as it reports (the wingmen usually announce first).

use crate::labels::Station;
use skylens_viewer::glam::DVec3;
use skylens_shared::proto::model::v1 as model;
use skylens_viewer::camera::{damp_factor, heading_forward};

/// Seconds without telemetry before a drone is shown as stale.
pub const STALE_AFTER_S: f64 = 3.0;
const POSITION_DAMP: f64 = 5.0;
const HEADING_DAMP: f64 = 6.0;

/// What the panels show of one drone.
#[derive(Debug, Clone, PartialEq)]
pub struct FleetDrone {
    pub id: u32,
    pub station: Station,
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
    pub heading_deg: f64,
    pub speed: f64,
    pub battery_pct: f64,
    /// Unix ms the tower received the last reading.
    pub received_at: f64,
    pub stale: bool,
}

/// A drone on screen: the reading plus the eased scene pose.
#[derive(Debug, Clone)]
pub struct Track {
    pub info: FleetDrone,
    /// Scene position (ENU metres, exaggerated vertical) and level forward vector.
    pub pos: DVec3,
    pub forward: DVec3,
    pub target: Option<DVec3>,
    pub target_forward: DVec3,
    pub placed: bool,
}

/// Turns a GPS fix into a scene position and the ground height under it (scene z); None until
/// the site frame is known.
pub trait Placement {
    fn place(&self, gps: &model::Gps) -> Option<(DVec3, f64)>;
}

#[derive(Debug, Default)]
pub struct Fleet {
    /// In insertion order sorted by station (legacy `state.drones`).
    tracks: Vec<Track>,
    /// The selected aircraft (legacy `state.activeDroneId`).
    pub active: Option<u32>,
    /// The operator chose an aircraft in the fleet list.
    pub operator_picked: bool,
    /// Bumped whenever `active` changes, so the camera panel can follow.
    pub active_rev: u64,
}

impl Fleet {
    pub fn new() -> Fleet {
        Fleet::default()
    }

    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    pub fn track(&self, id: u32) -> Option<&Track> {
        self.tracks.iter().find(|t| t.info.id == id)
    }

    fn set_active(&mut self, id: u32) {
        if self.active != Some(id) {
            self.active = Some(id);
            self.active_rev += 1;
        }
    }

    /// One telemetry frame off the core link.
    pub fn ingest(&mut self, t: &model::DroneTelemetry, now_ms: f64, placement: &dyn Placement) {
        let gps = t.gps.clone().unwrap_or_default();
        let station = Station::from_proto(t.station);
        if self.track(t.drone_id).is_none() {
            self.tracks.push(Track {
                info: FleetDrone {
                    id: t.drone_id,
                    station: station.unwrap_or(Station::Center),
                    lat: gps.lat,
                    lon: gps.lon,
                    alt: gps.alt,
                    heading_deg: t.heading_deg,
                    speed: t.speed,
                    battery_pct: t.battery_pct,
                    received_at: now_ms,
                    stale: false,
                },
                pos: DVec3::ZERO,
                forward: DVec3::Y,
                target: None,
                target_forward: DVec3::Y,
                placed: false,
            });
            // Stable sort keeps arrival order within a station.
            self.tracks.sort_by_key(|t| t.info.station.rank());
            let centre = self.tracks.iter().find(|t| t.info.station == Station::Center).map(|t| t.info.id);
            let missing = !self.active.is_some_and(|a| self.tracks.iter().any(|t| t.info.id == a));
            let take_centre = !self.operator_picked && centre.is_some() && self.active != centre;
            if missing || take_centre {
                let id = centre.or_else(|| self.tracks.first().map(|t| t.info.id)).unwrap_or(1);
                self.set_active(id);
            }
        }
        let track = self.tracks.iter_mut().find(|x| x.info.id == t.drone_id).expect("inserted above");
        if let Some(s) = station {
            track.info.station = s;
        }
        track.info.lat = gps.lat;
        track.info.lon = gps.lon;
        track.info.alt = gps.alt;
        track.info.heading_deg = t.heading_deg;
        track.info.speed = t.speed;
        track.info.battery_pct = t.battery_pct;
        track.info.received_at = now_ms;
        track.info.stale = false;
        track.target_forward = heading_forward(t.heading_deg);
        if let Some((mut target, ground_z)) = placement.place(&gps) {
            if target.z < ground_z {
                target.z = ground_z;
            }
            track.target = Some(target);
            if !track.placed {
                track.placed = true;
                track.pos = target;
                track.forward = track.target_forward;
            }
        }
    }

    /// Marks drones stale after 3 s of silence. Returns true when any flag changed.
    pub fn refresh_stale(&mut self, now_ms: f64) -> bool {
        let mut changed = false;
        for t in &mut self.tracks {
            let stale = (now_ms - t.info.received_at) / 1000.0 > STALE_AFTER_S;
            changed |= stale != t.info.stale;
            t.info.stale = stale;
        }
        changed
    }

    /// Eases every placed, live drone toward its last fix (call once per frame).
    pub fn update(&mut self, dt: f64, now_ms: f64) {
        self.refresh_stale(now_ms);
        let kp = damp_factor(POSITION_DAMP, dt);
        let kh = damp_factor(HEADING_DAMP, dt);
        for t in &mut self.tracks {
            if !t.placed || t.info.stale {
                continue;
            }
            if let Some(target) = t.target {
                t.pos = t.pos.lerp(target, kp);
            }
            let f = t.forward.lerp(t.target_forward, kh);
            t.forward = if f.length_squared() > 1e-12 { f.normalize() } else { t.target_forward };
        }
    }

    /// Snapshot for the panels, ordered by station (legacy `drones()`).
    pub fn drones(&self) -> Vec<FleetDrone> {
        let mut v: Vec<FleetDrone> = self.tracks.iter().map(|t| t.info.clone()).collect();
        v.sort_by_key(|d| d.station.rank());
        v
    }

    /// The operator picked a row in the fleet list.
    pub fn select_by_operator(&mut self, id: u32) {
        self.operator_picked = true;
        if self.track(id).is_some() {
            self.set_active(id);
        }
    }

    /// Keys 1/2/3: select a drone id if it exists (legacy `switchActive`; not an operator pick).
    pub fn switch_to(&mut self, id: u32) -> bool {
        if self.track(id).is_none() || self.active == Some(id) {
            return false;
        }
        self.set_active(id);
        true
    }

    /// Tab: the next drone in fleet order.
    pub fn cycle(&mut self) -> bool {
        if self.tracks.is_empty() {
            return false;
        }
        let ids: Vec<u32> = self.tracks.iter().map(|t| t.info.id).collect();
        let idx = self.active.and_then(|a| ids.iter().position(|i| *i == a));
        let next = match idx {
            Some(i) => ids[(i + 1) % ids.len()],
            // indexOf -1 + 1 = 0
            None => ids[0],
        };
        self.switch_to(next)
    }

    /// The drone the camera follows: the selected one, else the first.
    pub fn focus(&self) -> Option<&Track> {
        self.active.and_then(|a| self.track(a)).or_else(|| self.tracks.first())
    }

    /// The route belongs to the formation centre (legacy `getLeaderId`).
    pub fn leader_id(&self) -> u32 {
        self.tracks
            .iter()
            .find(|t| t.info.station == Station::Center)
            .map(|t| t.info.id)
            .or(self.active)
            .unwrap_or(1)
    }

    /// The link dropped: forget every drone (the selection id is kept, like the legacy state).
    pub fn clear(&mut self) {
        self.tracks.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Flat;
    impl Placement for Flat {
        fn place(&self, gps: &model::Gps) -> Option<(DVec3, f64)> {
            Some((DVec3::new(gps.lon, gps.lat, gps.alt), 5.0))
        }
    }

    fn tele(id: u32, station: model::DroneStation, alt: f64) -> model::DroneTelemetry {
        model::DroneTelemetry {
            drone_id: id,
            station: station as i32,
            gps: Some(model::Gps { lat: 1.0, lon: 2.0, alt }),
            heading_deg: 90.0,
            speed: 3.0,
            battery_pct: 80.0,
            t_ms: 0,
        }
    }

    #[test]
    fn centre_takes_the_selection_until_the_operator_picks() {
        let mut f = Fleet::new();
        f.ingest(&tele(2, model::DroneStation::Left, 50.0), 0.0, &Flat);
        assert_eq!(f.active, Some(2));
        f.ingest(&tele(1, model::DroneStation::Center, 50.0), 0.0, &Flat);
        assert_eq!(f.active, Some(1));
        f.select_by_operator(2);
        f.ingest(&tele(3, model::DroneStation::Right, 50.0), 0.0, &Flat);
        assert_eq!(f.active, Some(2));
        let order: Vec<u32> = f.drones().iter().map(|d| d.id).collect();
        assert_eq!(order, vec![2, 1, 3]);
    }

    #[test]
    fn keys_switch_and_cycle() {
        let mut f = Fleet::new();
        for (id, s) in [(2, model::DroneStation::Left), (1, model::DroneStation::Center), (3, model::DroneStation::Right)] {
            f.ingest(&tele(id, s, 50.0), 0.0, &Flat);
        }
        assert!(!f.switch_to(9));
        assert!(f.switch_to(3));
        assert!(f.cycle());
        // Fleet order is left (2), centre (1), right (3): after 3 comes 2.
        assert_eq!(f.active, Some(2));
        assert!(!f.operator_picked);
        assert_eq!(f.leader_id(), 1);
    }

    #[test]
    fn ground_clamp_snap_and_stale() {
        let mut f = Fleet::new();
        f.ingest(&tele(1, model::DroneStation::Center, 1.0), 0.0, &Flat);
        let t = f.track(1).unwrap();
        assert!(t.placed);
        assert_eq!(t.pos.z, 5.0, "below ground is lifted onto it");
        assert!(!f.refresh_stale(2000.0));
        assert!(f.refresh_stale(3500.0));
        assert!(f.drones()[0].stale);
    }

    #[test]
    fn easing_moves_toward_the_fix() {
        let mut f = Fleet::new();
        f.ingest(&tele(1, model::DroneStation::Center, 10.0), 0.0, &Flat);
        let mut moved = tele(1, model::DroneStation::Center, 20.0);
        moved.gps.as_mut().unwrap().lat = 11.0;
        f.ingest(&moved, 10.0, &Flat);
        f.update(0.05, 20.0);
        let y = f.track(1).unwrap().pos.y;
        assert!(y > 1.0 && y < 11.0);
    }
}
