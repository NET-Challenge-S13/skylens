//! The board's automatic camera (legacy `statusViewer.ts` `followPose` / `focusPose`), in ENU
//! metres (x east, y north, z up). The legacy three.js scene was y-up with z = -north; every rule
//! here is the same rule turned into that frame.
//!
//! - **Follow**: look at the ground the active drone is scanning, from behind and above its
//!   heading, pulled back far enough to keep the built strip (which trails the aircraft) in frame,
//!   with the fog and the far side re-derived from that distance.
//! - **Interior**: when the reconstruction turns out to be a roofed walkway, the camera walks the
//!   corridor's own centreline at eye height instead of hovering over the roof.
//! - **Focus**: approach a detection at eye level along the corridor, never crosswise.

use crate::glam::{DVec2, DVec3};
use crate::points::quantile;
use skylens_viewer::Pose;

/// Distance (m) the status camera holds from a focused detection (legacy `focusDistance`).
pub const FOCUS_DISTANCE: f64 = skylens_viewer::config::camera::FOCUS_DISTANCE;

/// The corridor's centreline, measured from the reconstruction: median position per 6 m bin along
/// its main axis, smoothed. `dir` is the along axis, `mean` its origin, `pts[i]` the centre at
/// `u0 + i * du` (None where the strip has a gap).
#[derive(Debug, Clone, PartialEq)]
pub struct CorridorPath {
    pub dir: DVec2,
    pub mean: DVec2,
    pub u0: f64,
    pub du: f64,
    pub pts: Vec<Option<DVec2>>,
}

impl CorridorPath {
    /// Along-corridor position of a ground point.
    pub fn along(&self, p: DVec2) -> f64 {
        (p - self.mean).dot(self.dir)
    }

    /// Centreline point at along-position `u`, lerped between the nearest bins.
    pub fn point_at(&self, u: f64) -> DVec2 {
        let last = self.pts.len().saturating_sub(1);
        let f = (u - self.u0) / self.du;
        let mut lo = (f.floor().max(0.0) as usize).min(last);
        let mut hi = (lo + 1).min(last);
        while lo > 0 && self.pts[lo].is_none() {
            lo -= 1;
        }
        while hi < last && self.pts[hi].is_none() {
            hi += 1;
        }
        let a = self.pts[lo].or(self.pts[hi]);
        let b = self.pts[hi].or(self.pts[lo]);
        let (Some(a), Some(b)) = (a, b) else { return self.mean };
        if lo == hi {
            return a;
        }
        let t = ((f - lo as f64) / (hi - lo) as f64).clamp(0.0, 1.0);
        a + (b - a) * t
    }
}

/// Median ground position per 6 m along-axis bin of the samples (legacy `buildCorridorPath`).
pub fn build_corridor_path(samples: &[DVec3]) -> Option<CorridorPath> {
    if samples.len() < 500 {
        return None;
    }
    let n_f = samples.len() as f64;
    let mean = samples.iter().fold(DVec2::ZERO, |a, p| a + p.truncate()) / n_f;
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in samples {
        let d = p.truncate() - mean;
        sxx += d.x * d.x;
        sxy += d.x * d.y;
        syy += d.y * d.y;
    }
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let dir = DVec2::new(theta.cos(), theta.sin());

    let mut us: Vec<f64> = samples.iter().map(|p| (p.truncate() - mean).dot(dir)).collect();
    us.sort_by(f64::total_cmp);
    let u0 = quantile(&us, 0.01)?;
    let u1 = quantile(&us, 0.99)?;
    let du = 6.0;
    let n = (((u1 - u0) / du).ceil() as usize + 1).max(2);
    let mut bins: Vec<(Vec<f64>, Vec<f64>)> = vec![(Vec::new(), Vec::new()); n];
    for p in samples {
        let u = (p.truncate() - mean).dot(dir);
        let i = ((u - u0) / du).round();
        if i >= 0.0 && (i as usize) < n {
            bins[i as usize].0.push(p.x);
            bins[i as usize].1.push(p.y);
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let raw: Vec<Option<DVec2>> = bins
        .into_iter()
        .map(|(mut xs, mut ys)| (xs.len() >= 40).then(|| DVec2::new(median(&mut xs), median(&mut ys))))
        .collect();
    if raw.iter().filter(|p| p.is_some()).count() < 2 {
        return None;
    }
    // Moving average over +-4 bins: per-bin medians zigzag with whatever stands in each slice, and
    // a camera riding the raw polyline sways wall to wall.
    let pts = raw
        .iter()
        .enumerate()
        .map(|(i, p)| {
            p.map(|_| {
                let lo = i.saturating_sub(4);
                let hi = (i + 4).min(raw.len() - 1);
                let near: Vec<DVec2> = raw[lo..=hi].iter().flatten().copied().collect();
                near.iter().fold(DVec2::ZERO, |a, q| a + *q) / near.len() as f64
            })
        })
        .collect();
    Some(CorridorPath { dir, mean, u0, du, pts })
}

/// A reconstruction recognised as an interior (roof over a walkway).
#[derive(Debug, Clone, PartialEq)]
pub struct Interior {
    /// Camera eye height.
    pub eye_z: f64,
    pub floor_z: f64,
}

/// Floor and roof are the two densest horizontal sheets (not the low and high percentiles: a
/// glossy floor reconstructs its mirror image below itself). A gap of 2.5 to 40 m between them
/// makes the scene an interior, with the eye at `min(floor + 2.2, roof - 1.2)`.
pub fn detect_interior(samples: &[DVec3]) -> Option<Interior> {
    let mut zs: Vec<f64> = samples.iter().map(|p| p.z).filter(|z| z.is_finite()).collect();
    if zs.len() <= 200 {
        return None;
    }
    zs.sort_by(f64::total_cmp);
    let lo = quantile(&zs, 0.02)?;
    let hi = quantile(&zs, 0.98)?;
    let bin = 0.4;
    let n = (((hi - lo) / bin).ceil() as usize).max(4);
    let mut counts = vec![0usize; n];
    for z in &zs {
        let i = ((z - lo) / bin).floor();
        if i >= 0.0 && (i as usize) < n {
            counts[i as usize] += 1;
        }
    }
    let mid = zs[zs.len() / 2];
    let (mut floor_bin, mut roof_bin) = (0, n - 1);
    for (i, c) in counts.iter().enumerate() {
        let centre = lo + (i as f64 + 0.5) * bin;
        if centre < mid && *c > counts[floor_bin] {
            floor_bin = i;
        }
        if centre >= mid && *c > counts[roof_bin] {
            roof_bin = i;
        }
    }
    let floor = lo + (floor_bin as f64 + 0.5) * bin;
    let roof = lo + (roof_bin as f64 + 0.5) * bin;
    let gap = roof - floor;
    (gap < 40.0 && gap > 2.5).then(|| Interior { eye_z: (floor + 2.2).min(roof - 1.2), floor_z: floor })
}

/// Framing the board derives from its first geometry (legacy: from the placeholder cloud).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Framing {
    /// Dense-core radius: 60th percentile distance from the centre.
    pub radius: f64,
    pub center: DVec3,
}

impl Framing {
    /// Until geometry arrives.
    pub const DEFAULT: Framing = Framing { radius: 20.0, center: DVec3::ZERO };

    /// Frames on the dense core, not the full bounds: a wide spread-out ground would inflate the
    /// bounding sphere and leave the subject tiny.
    pub fn from_samples(samples: &[DVec3]) -> Option<Framing> {
        if samples.is_empty() {
            return None;
        }
        let (mut lo, mut hi) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
        for p in samples {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let center = (lo + hi) / 2.0;
        let mut d: Vec<f64> = samples.iter().map(|p| p.distance(center)).collect();
        d.sort_by(f64::total_cmp);
        let core = quantile(&d, 0.6).unwrap_or(10.0);
        Some(Framing { radius: core.max(1.0), center })
    }

    pub fn follow_dist(&self) -> f64 {
        self.radius * 1.6
    }

    pub fn follow_height(&self) -> f64 {
        self.radius * 1.3
    }

    /// Fog before the follow takes over: `radius * 3 .. radius * 9`.
    pub fn base_fog(&self) -> (f64, f64) {
        (self.radius * 3.0, self.radius * 9.0)
    }

    /// Operator zoom limits (`OrbitControls` min / max distance).
    pub fn zoom_limits(&self) -> (f64, f64) {
        (self.radius * 0.2, self.radius * 12.0)
    }

    /// The opening view: from the south-east, above, looking at the centre.
    pub fn opening_pose(&self) -> Pose {
        let d = self.radius * 2.6;
        Pose::new(self.center + DVec3::new(0.75 * d, -0.75 * d, 0.4 * d), self.center)
    }
}

/// The active drone as the follow sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DronePose {
    pub pos: DVec3,
    /// Look direction, tilted down (see [`drone_forward`]).
    pub forward: DVec3,
}

/// Look direction for a compass heading: level heading plus the slight downward tilt that makes
/// the follow look at the ground being captured rather than the horizon (legacy `-0.35`).
pub fn drone_forward(heading_deg: f64) -> DVec3 {
    let h = heading_deg.to_radians();
    DVec3::new(h.sin(), h.cos(), -0.35).normalize()
}

/// Level heading of a look direction; south when it is vertical (legacy fallback `+z`).
fn heading_xy(forward: DVec3) -> DVec3 {
    let h = DVec3::new(forward.x, forward.y, 0.0);
    if h.length_squared() < 1e-6 { DVec3::NEG_Y } else { h.normalize() }
}

/// A desired camera pose and the fog that goes with it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FollowOut {
    pub pose: Pose,
    pub fog: (f64, f64),
}

/// Smoothed along-corridor position and latched travel direction: the raw scan point jitters with
/// the drone's gaze and the travel sign wobbles through zero at turns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Follower {
    follow_u: Option<f64>,
    travel_sign: f64,
}

impl Default for Follower {
    fn default() -> Self {
        Follower { follow_u: None, travel_sign: 1.0 }
    }
}

/// What the follow reads besides the drone.
pub struct FollowInput<'a> {
    pub framing: Framing,
    pub floor_z: f64,
    pub interior: Option<&'a Interior>,
    pub path: Option<&'a CorridorPath>,
    /// Centres of the delivered segments.
    pub chunk_centers: &'a [DVec3],
}

impl Follower {
    pub fn pose(&mut self, drone: &DronePose, input: &FollowInput<'_>, dt: f64) -> FollowOut {
        let p = drone.pos;
        let f = drone.forward;
        let heading = heading_xy(f);
        let follow_dist = input.framing.follow_dist();

        // Where the drone's look ray meets the ground (the scanned spot).
        let t = if f.z < -1e-3 { (p.z - input.floor_z) / -f.z } else { follow_dist };
        let t = t.clamp(0.0, follow_dist * 3.0);
        let ground_hit = p + f * t;

        if let Some(interior) = input.interior {
            let dist = 14.0;
            let (pos, target) = match input.path {
                Some(path) => {
                    // Along-position from the aircraft, not its gaze ray (which jitters).
                    let u_hit = path.along(p.truncate());
                    let prev = self.follow_u.unwrap_or(u_hit);
                    let u = prev + (u_hit - prev) * (1.0 - (-dt.max(0.0)).exp());
                    self.follow_u = Some(u);
                    let dot = heading.truncate().dot(path.dir);
                    if dot.abs() > 0.35 {
                        self.travel_sign = dot.signum();
                    }
                    let t2 = path.point_at(u);
                    let p2 = path.point_at(u - self.travel_sign * dist);
                    (p2.extend(interior.eye_z), t2.extend(interior.eye_z - 0.5))
                }
                None => {
                    let target = ground_hit.truncate().extend(interior.eye_z - 0.5);
                    let pos = (target - heading * dist).truncate().extend(interior.eye_z);
                    (pos, target)
                }
            };
            return FollowOut { pose: Pose::new(pos, target), fog: (40.0, 260.0) };
        }

        // The reconstruction trails the aircraft: pull back far enough to hold the built strip in
        // frame and look at the middle of it rather than at the aircraft's own footprint.
        let tail = input
            .chunk_centers
            .iter()
            .copied()
            .filter(|c| c.distance(ground_hit) > 0.0)
            .max_by(|a, b| a.distance(ground_hit).total_cmp(&b.distance(ground_hit)));
        let lag = tail.map_or(0.0, |c| c.distance(ground_hit));
        let dist = (lag * 1.2).max(follow_dist).min(follow_dist * 10.0);
        let height = input.framing.follow_height().max(dist * 0.55);
        let target = tail.map_or(ground_hit, |c| ground_hit.lerp(c, 0.45));
        let pos = target - heading * dist + DVec3::Z * height;
        FollowOut { pose: Pose::new(pos, target), fog: (dist * 1.4, dist * 7.0) }
    }
}

/// The corridor direction at `p` in flight order: the vector between the delivered segments
/// either side of the nearest one; else the drone's level heading; else east.
pub fn corridor_axis_at(p: DVec3, segment_centers: &[(u32, DVec3)], drone: Option<&DronePose>) -> DVec3 {
    let mut by_seg = segment_centers.to_vec();
    by_seg.sort_by_key(|(s, _)| *s);
    if by_seg.len() >= 2 {
        let k = by_seg
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let da = (a.1 - p).truncate().length_squared();
                let db = (b.1 - p).truncate().length_squared();
                da.total_cmp(&db)
            })
            .map_or(0, |(i, _)| i);
        let a = by_seg[k.saturating_sub(1)].1;
        let b = by_seg[(k + 1).min(by_seg.len() - 1)].1;
        let dir = DVec3::new(b.x - a.x, b.y - a.y, 0.0);
        if dir.length_squared() > 1e-6 {
            return dir.normalize();
        }
    }
    let dir = drone.map_or(DVec3::X, |d| DVec3::new(d.forward.x, d.forward.y, 0.0));
    if dir.length_squared() > 1e-6 { dir.normalize() } else { DVec3::X }
}

/// Approach a detection at eye level along the corridor (legacy `focusPose`): stand on the
/// centreline behind it in flight order, look at the body (1.1 m above the feet), not the pin.
pub fn focus_pose(
    feet: DVec3,
    interior: Option<&Interior>,
    path: Option<&CorridorPath>,
    segment_centers: &[(u32, DVec3)],
    drone: Option<&DronePose>,
) -> Pose {
    let eye_z = interior.map_or(feet.z + 1.7, |i| i.eye_z);
    let target = feet + DVec3::Z * 1.1;
    let axis = corridor_axis_at(feet, segment_centers, drone);
    let back = FOCUS_DISTANCE * 0.7;
    let pos = match path {
        Some(path) => {
            let u_feet = path.along(feet.truncate());
            let s = axis.truncate().dot(path.dir).signum();
            let sign = if s == 0.0 { 1.0 } else { s };
            path.point_at(u_feet - sign * back).extend(eye_z)
        }
        None => (feet - axis * back).truncate().extend(eye_z),
    };
    Pose::new(pos, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A corridor 5 m wide, 3 m high, running east for 120 m: floor and roof sheets plus walls.
    fn corridor() -> Vec<DVec3> {
        let mut v = Vec::new();
        for i in 0..1200 {
            let x = f64::from(i) * 0.1;
            for k in 0..5 {
                let y = f64::from(k) - 2.0;
                v.push(DVec3::new(x, y, 0.0));
                v.push(DVec3::new(x, y, 3.0));
            }
            v.push(DVec3::new(x, -2.5, 1.5));
            v.push(DVec3::new(x, 2.5, 1.5));
        }
        v
    }

    #[test]
    fn a_corridor_is_an_interior_with_its_centreline_along_x() {
        let s = corridor();
        let i = detect_interior(&s).unwrap();
        assert!((i.floor_z - 0.2).abs() < 0.3, "{i:?}");
        assert!((i.eye_z - 1.8).abs() < 0.4, "{i:?}");
        let path = build_corridor_path(&s).unwrap();
        assert!(path.dir.x.abs() > 0.99);
        let mid = path.point_at(path.along(DVec2::new(60.0, 0.0)));
        assert!(mid.y.abs() < 0.3 && (mid.x - 60.0).abs() < 4.0, "{mid:?}");
    }

    #[test]
    fn open_ground_is_not_an_interior() {
        let flat: Vec<DVec3> = (0..1000).map(|i| DVec3::new(f64::from(i), 0.0, f64::from(i % 3) * 0.1)).collect();
        assert!(detect_interior(&flat).is_none());
        assert!(detect_interior(&flat[..100]).is_none());
        assert!(build_corridor_path(&flat[..100]).is_none());
    }

    #[test]
    fn framing_uses_the_dense_core() {
        let mut s: Vec<DVec3> = (0..100).map(|i| DVec3::new(f64::from(i % 10), f64::from(i / 10), 0.0)).collect();
        s.push(DVec3::new(500.0, 500.0, 0.0));
        let f = Framing::from_samples(&s).unwrap();
        assert!(f.radius < 400.0);
        assert!(Framing::from_samples(&[]).is_none());
        let o = Framing { radius: 10.0, center: DVec3::ZERO }.opening_pose();
        assert!(o.eye.x > 0.0 && o.eye.y < 0.0 && o.eye.z > 0.0);
    }

    #[test]
    fn follow_trails_behind_and_above_and_widens_with_the_built_strip() {
        let framing = Framing { radius: 10.0, center: DVec3::ZERO };
        let drone = DronePose { pos: DVec3::new(0.0, 0.0, 10.0), forward: drone_forward(0.0) };
        let mut f = Follower::default();
        let input = FollowInput { framing, floor_z: 0.0, interior: None, path: None, chunk_centers: &[] };
        let out = f.pose(&drone, &input, 1.0 / 60.0);
        // Heading north: the camera sits south of the target and above it.
        assert!(out.pose.eye.y < out.pose.target.y);
        assert!(out.pose.eye.z > out.pose.target.z);
        assert!((out.pose.target.z).abs() < 1e-9, "looks at the ground");
        assert_eq!(out.fog, (16.0 * 1.4, 16.0 * 7.0));

        let centers = [DVec3::new(0.0, -200.0, 0.0)];
        let input = FollowInput { chunk_centers: &centers, ..input };
        let wide = f.pose(&drone, &input, 1.0 / 60.0);
        assert!(wide.pose.distance() > out.pose.distance());
        assert!(wide.pose.target.y < out.pose.target.y, "looks back toward the built strip");
        // Capped at ten times the base distance.
        assert!(wide.fog.0 <= 160.0 * 1.4 + 1e-9);
    }

    #[test]
    fn interior_follow_rides_the_centreline_at_eye_height() {
        let s = corridor();
        let interior = detect_interior(&s).unwrap();
        let path = build_corridor_path(&s).unwrap();
        let drone = DronePose { pos: DVec3::new(60.0, 1.8, 2.5), forward: drone_forward(90.0) };
        let mut f = Follower::default();
        let input = FollowInput {
            framing: Framing::DEFAULT,
            floor_z: interior.floor_z,
            interior: Some(&interior),
            path: Some(&path),
            chunk_centers: &[],
        };
        let out = f.pose(&drone, &input, 1.0 / 60.0);
        assert_eq!(out.fog, (40.0, 260.0));
        assert!((out.pose.eye.z - interior.eye_z).abs() < 1e-9);
        assert!(out.pose.eye.y.abs() < 0.5, "on the centreline, not on the drone's wall");
        // Flying east: the camera trails 14 m west of the target, whichever way the axis points.
        assert!((out.pose.target.x - out.pose.eye.x - 14.0).abs() < 1.0, "{out:?}");
    }

    #[test]
    fn focus_stands_behind_the_detection_along_the_flight() {
        let centers = [(0, DVec3::new(0.0, 0.0, 0.0)), (1, DVec3::new(20.0, 0.0, 0.0)), (2, DVec3::new(40.0, 0.0, 0.0))];
        let feet = DVec3::new(21.0, 3.0, 0.0);
        let p = focus_pose(feet, None, None, &centers, None);
        assert!((p.target - DVec3::new(21.0, 3.0, 1.1)).length() < 1e-9);
        assert!((p.eye.x - (21.0 - FOCUS_DISTANCE * 0.7)).abs() < 1e-9);
        assert!((p.eye.y - 3.0).abs() < 1e-9);
        assert!((p.eye.z - 1.7).abs() < 1e-9);
        // No segments and no drone: the axis falls back to east.
        assert_eq!(corridor_axis_at(feet, &[], None), DVec3::X);
    }
}
