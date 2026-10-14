//! Where the synthetic formation is at a given moment (`--synthetic`, and
//! `--follow-route`). The recorded flight is replayed by `footage.rs` instead.
//!
//! The old clips carry no GPS, so with `--synthetic` the formation flies a
//! fixed track east-north-east of the site anchor and restarts it from the top
//! when it reaches the end, the way the footage loops. With `--follow-route` it
//! flies the operator's waypoints back and forth.
//!
//! All maths is in local ENU metres around the site anchor; [`to_gps`] converts
//! with the shared GPS<->ENU formula.

use skylens_shared::geo::{self, Enu, Gps};
use skylens_shared::proto::model::v1::DroneStation;

/// The core's default ENU origin (Daejeon preset: 36.3685, 127.3475, 30 m).
pub const SITE_ANCHOR: Gps = Gps { lat: 36.3685, lon: 127.3475, alt: 30.0 };
/// Flight altitude sent as `Gps.alt`, metres.
pub const ALTITUDE_M: f64 = 90.0;
/// Lateral spacing between neighbouring aircraft, metres.
pub const SPACING_M: f64 = 10.0;
/// Ground speed along the track, m/s.
pub const SPEED_MPS: f64 = 4.0;
/// Footage track: start, compass bearing and length.
pub const TRACK_START: (f64, f64) = (30.0, 10.0);
pub const TRACK_BEARING_DEG: f64 = 67.5; // east-north-east
pub const TRACK_LENGTH_M: f64 = 240.0;
/// How far behind the route start the formation appears before it arrives.
pub const APPROACH_M: f64 = 150.0;

/// A horizontal position (east, north) in metres.
pub type En = (f64, f64);

/// Compass bearing of a direction vector, degrees in `[0, 360)` (0 = north, 90 = east).
pub fn bearing_deg(de: f64, dn: f64) -> f64 {
    let b = de.atan2(dn).to_degrees();
    if b < 0.0 { b + 360.0 } else { b }
}

/// Unit vector (east, north) of a compass bearing.
pub fn unit(bearing: f64) -> En {
    let r = bearing.to_radians();
    (r.sin(), r.cos())
}

/// Offset of a station from the formation centre, given the formation's heading.
/// Left is to the left of the direction of travel.
pub fn station_offset(station: DroneStation, heading_deg: f64, spacing: f64) -> En {
    let (e, n) = unit(heading_deg);
    let left = (-n, e);
    let k = match station {
        DroneStation::Left => 1.0,
        DroneStation::Right => -1.0,
        _ => 0.0,
    };
    (left.0 * spacing * k, left.1 * spacing * k)
}

/// What happens at the end of the path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Motion {
    /// Jump back to the start (the footage restarts the same way).
    Restart,
    /// Turn around and fly back.
    PingPong,
    /// Stop at the last point.
    Hold,
}

/// A polyline flown at constant speed.
#[derive(Debug, Clone, PartialEq)]
pub struct Path {
    pub points: Vec<En>,
    pub motion: Motion,
}

/// Position and heading of the formation centre.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub e: f64,
    pub n: f64,
    pub heading_deg: f64,
    pub speed: f64,
}

impl Path {
    /// The fixed track the footage is placed on.
    pub fn footage_track() -> Path {
        let (de, dn) = unit(TRACK_BEARING_DEG);
        let start = TRACK_START;
        let end = (start.0 + de * TRACK_LENGTH_M, start.1 + dn * TRACK_LENGTH_M);
        Path { points: vec![start, end], motion: Motion::Restart }
    }

    /// The operator's waypoints, in ENU around `anchor`. Repeated points are
    /// dropped. `None` when fewer than two distinct points remain.
    pub fn from_waypoints(waypoints: &[Gps], anchor: Gps, back_and_forth: bool) -> Option<Path> {
        let mut points: Vec<En> = Vec::new();
        for w in waypoints {
            let p = geo::gps_to_enu(*w, anchor);
            let p = (p.e, p.n);
            if points.last().is_none_or(|q| (q.0 - p.0).hypot(q.1 - p.1) > 0.01) {
                points.push(p);
            }
        }
        if points.len() < 2 {
            return None;
        }
        let motion = if back_and_forth { Motion::PingPong } else { Motion::Hold };
        Some(Path { points, motion })
    }

    pub fn length(&self) -> f64 {
        self.points.windows(2).map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1)).sum()
    }

    /// Point and segment bearing at arc length `s` (clamped to the path).
    fn at_distance(&self, s: f64) -> (En, f64) {
        let mut left = s.max(0.0);
        let segments = self.points.len().saturating_sub(1);
        for (i, w) in self.points.windows(2).enumerate() {
            let (a, b) = (w[0], w[1]);
            let len = (b.0 - a.0).hypot(b.1 - a.1);
            if left <= len || i + 1 == segments {
                let f = if len > 0.0 { (left / len).min(1.0) } else { 0.0 };
                return ((a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f), bearing_deg(b.0 - a.0, b.1 - a.1));
            }
            left -= len;
        }
        // Fewer than two points: stand still facing north.
        (self.points.first().copied().unwrap_or((0.0, 0.0)), 0.0)
    }

    /// The formation centre `t` seconds after it started the path at `speed`.
    pub fn pose_at(&self, t: f64, speed: f64) -> Pose {
        let len = self.length();
        let travelled = (t.max(0.0)) * speed;
        if len <= 0.0 {
            let (p, h) = self.at_distance(0.0);
            return Pose { e: p.0, n: p.1, heading_deg: h, speed: 0.0 };
        }
        match self.motion {
            Motion::Restart => {
                let (p, h) = self.at_distance(travelled % len);
                Pose { e: p.0, n: p.1, heading_deg: h, speed }
            }
            Motion::Hold => {
                let (p, h) = self.at_distance(travelled.min(len));
                let moving = if travelled < len { speed } else { 0.0 };
                Pose { e: p.0, n: p.1, heading_deg: h, speed: moving }
            }
            Motion::PingPong => {
                let cycle = travelled % (2.0 * len);
                if cycle <= len {
                    let (p, h) = self.at_distance(cycle);
                    Pose { e: p.0, n: p.1, heading_deg: h, speed }
                } else {
                    let (p, h) = self.at_distance(2.0 * len - cycle);
                    Pose { e: p.0, n: p.1, heading_deg: (h + 180.0) % 360.0, speed }
                }
            }
        }
    }

    /// The first point and the bearing of the first segment.
    pub fn start(&self) -> (En, f64) {
        self.at_distance(0.0)
    }
}

/// Flying in: `t` seconds into an approach of `duration` seconds that ends at
/// the path start, coming from `APPROACH_M` behind it along its first bearing.
pub fn approach_pose(path: &Path, t: f64, duration: f64) -> Pose {
    let ((e0, n0), heading) = path.start();
    let (ue, un) = unit(heading);
    let f = if duration > 0.0 { (t / duration).clamp(0.0, 1.0) } else { 1.0 };
    let back = APPROACH_M * (1.0 - f);
    let speed = if duration > 0.0 && f < 1.0 { APPROACH_M / duration } else { 0.0 };
    Pose { e: e0 - ue * back, n: n0 - un * back, heading_deg: heading, speed }
}

/// GPS of one station at absolute altitude `alt`, given the formation centre.
pub fn to_gps(pose: &Pose, station: DroneStation, anchor: Gps, alt: f64) -> Gps {
    let (oe, on) = station_offset(station, pose.heading_deg, SPACING_M);
    let g = geo::enu_to_gps(Enu { e: pose.e + oe, n: pose.n + on, u: 0.0 }, anchor);
    Gps { lat: g.lat, lon: g.lon, alt }
}

/// The footage track as waypoints, for `--auto`.
pub fn footage_waypoints(anchor: Gps) -> Vec<Gps> {
    Path::footage_track()
        .points
        .iter()
        .map(|&(e, n)| {
            let g = geo::enu_to_gps(Enu { e, n, u: 0.0 }, anchor);
            Gps { lat: g.lat, lon: g.lon, alt: ALTITUDE_M }
        })
        .collect()
}

/// Battery after `t` seconds of flight: a slow, visible drain that never runs out.
pub fn battery_pct(t: f64) -> f64 {
    (100.0 - t.max(0.0) * 0.05).max(20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    fn line(len: f64, motion: Motion) -> Path {
        Path { points: vec![(0.0, 0.0), (len, 0.0)], motion }
    }

    #[test]
    fn bearings_are_compass_degrees() {
        assert!(close(bearing_deg(0.0, 1.0), 0.0));
        assert!(close(bearing_deg(1.0, 0.0), 90.0));
        assert!(close(bearing_deg(0.0, -1.0), 180.0));
        assert!(close(bearing_deg(-1.0, 0.0), 270.0));
        let (e, n) = unit(90.0);
        assert!(close(e, 1.0) && close(n, 0.0));
    }

    #[test]
    fn left_is_left_of_travel() {
        // heading north: left is west
        let l = station_offset(DroneStation::Left, 0.0, 10.0);
        assert!(close(l.0, -10.0) && close(l.1, 0.0));
        let r = station_offset(DroneStation::Right, 0.0, 10.0);
        assert!(close(r.0, 10.0) && close(r.1, 0.0));
        // heading east: left is north
        let l = station_offset(DroneStation::Left, 90.0, 10.0);
        assert!(close(l.0, 0.0) && close(l.1, 10.0));
        let c = station_offset(DroneStation::Center, 37.0, 10.0);
        assert!(close(c.0, 0.0) && close(c.1, 0.0));
    }

    #[test]
    fn restart_wraps_to_the_start() {
        let p = line(100.0, Motion::Restart);
        let a = p.pose_at(5.0, 10.0);
        assert!(close(a.e, 50.0) && close(a.heading_deg, 90.0) && close(a.speed, 10.0));
        let b = p.pose_at(12.0, 10.0);
        assert!(close(b.e, 20.0), "{b:?}");
    }

    #[test]
    fn ping_pong_turns_around() {
        let p = line(100.0, Motion::PingPong);
        let a = p.pose_at(12.0, 10.0);
        assert!(close(a.e, 80.0) && close(a.heading_deg, 270.0), "{a:?}");
        let b = p.pose_at(21.0, 10.0);
        assert!(close(b.e, 10.0) && close(b.heading_deg, 90.0), "{b:?}");
    }

    #[test]
    fn hold_stops_at_the_end() {
        let p = line(100.0, Motion::Hold);
        let a = p.pose_at(50.0, 10.0);
        assert!(close(a.e, 100.0) && close(a.speed, 0.0));
        assert!(close(p.pose_at(5.0, 10.0).speed, 10.0));
    }

    #[test]
    fn corners_follow_the_polyline() {
        let p = Path { points: vec![(0.0, 0.0), (30.0, 0.0), (30.0, 40.0)], motion: Motion::Hold };
        assert!(close(p.length(), 70.0));
        let a = p.pose_at(4.0, 10.0); // 40 m: 10 m up the second leg
        assert!(close(a.e, 30.0) && close(a.n, 10.0) && close(a.heading_deg, 0.0), "{a:?}");
        let end = p.pose_at(100.0, 10.0);
        assert!(close(end.e, 30.0) && close(end.n, 40.0));
    }

    #[test]
    fn footage_track_heads_east_north_east() {
        let p = Path::footage_track();
        assert!(close(p.length(), TRACK_LENGTH_M));
        let ((e, n), h) = p.start();
        assert!(close(e, TRACK_START.0) && close(n, TRACK_START.1));
        assert!(close(h, TRACK_BEARING_DEG));
        let end = p.points[1];
        assert!(end.0 > end.1 && end.1 > n, "east more than north, and north of the start");
    }

    #[test]
    fn approach_ends_at_the_path_start() {
        let p = line(100.0, Motion::Restart);
        let first = approach_pose(&p, 0.0, 10.0);
        assert!(close(first.e, -APPROACH_M) && close(first.n, 0.0));
        assert!(close(first.speed, APPROACH_M / 10.0));
        let half = approach_pose(&p, 5.0, 10.0);
        assert!(close(half.e, -APPROACH_M / 2.0));
        let last = approach_pose(&p, 10.0, 10.0);
        assert!(close(last.e, 0.0) && close(last.speed, 0.0));
    }

    #[test]
    fn waypoints_round_trip_through_gps() {
        let wps = footage_waypoints(SITE_ANCHOR);
        assert_eq!(wps.len(), 2);
        assert!(wps.iter().all(|g| close(g.alt, ALTITUDE_M)));
        // north-east of the anchor
        assert!(wps[1].lat > SITE_ANCHOR.lat && wps[1].lon > SITE_ANCHOR.lon);
        let p = Path::from_waypoints(&wps, SITE_ANCHOR, true).unwrap();
        assert_eq!(p.motion, Motion::PingPong);
        assert!((p.length() - TRACK_LENGTH_M).abs() < 1e-3);
        assert!(Path::from_waypoints(&wps[..1], SITE_ANCHOR, true).is_none());
        assert!(Path::from_waypoints(&[wps[0], wps[0]], SITE_ANCHOR, false).is_none());
    }

    #[test]
    fn stations_spread_across_the_track_at_altitude() {
        let pose = Path::footage_track().pose_at(0.0, SPEED_MPS);
        let l = to_gps(&pose, DroneStation::Left, SITE_ANCHOR, ALTITUDE_M);
        let r = to_gps(&pose, DroneStation::Right, SITE_ANCHOR, ALTITUDE_M);
        let le = geo::gps_to_enu(l, SITE_ANCHOR);
        let re = geo::gps_to_enu(r, SITE_ANCHOR);
        assert!(((le.e - re.e).hypot(le.n - re.n) - 2.0 * SPACING_M).abs() < 1e-3);
        assert!(close(l.alt, ALTITUDE_M));
    }

    #[test]
    fn battery_drains_slowly_and_stops_at_twenty() {
        assert!(close(battery_pct(0.0), 100.0));
        assert!(close(battery_pct(100.0), 95.0));
        assert!(close(battery_pct(1e6), 20.0));
    }
}
