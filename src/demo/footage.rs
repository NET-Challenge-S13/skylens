//! The recorded flight (`data/demo/track.json`): one aircraft flew the same line three times
//! with the gimbal turned to three headings (left, center, right), once high and once low; the
//! two passes of each view are joined into one clip. This module reads that record, replays it
//! on the clip's timeline, and derives what the launcher hands the core: the site centre and the
//! planned route.
//!
//! `rel_alt_m` is height above the takeoff point. `Gps.alt` everywhere in SkyLens is absolute
//! (metres above sea level: the terrain is the DEM minus the anchor altitude, and screens place a
//! drone at `alt - anchor.alt`), so a fix is `ground_alt + rel_alt_m`.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use skylens_shared::geo::{self, Enu, Gps};

use crate::route::{bearing_deg, unit};

/// Douglas-Peucker tolerance for turn points of the planned route, metres. Consumer GPS wanders
/// a few metres; a real turn moves the line more than this.
pub const TURN_TOLERANCE_M: f64 = 8.0;

#[derive(Debug, Clone, Deserialize)]
struct TrackFile {
    views: HashMap<String, ViewFile>,
}

#[derive(Debug, Clone, Deserialize)]
struct ViewFile {
    #[serde(default)]
    video: String,
    duration: f64,
    #[serde(default)]
    segments: Vec<Segment>,
    track: Vec<Row>,
}

/// One pass of a view, on the joined clip's timeline.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Segment {
    /// The original DJI clip (kept for the record; nothing reads it).
    #[serde(default)]
    #[allow(dead_code)]
    pub source: String,
    pub start: f64,
    pub duration: f64,
    pub rel_alt_m: f64,
}

/// One flight record row (about 10 Hz). Angles in degrees.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Row {
    pub t: f64,
    pub lat: f64,
    pub lon: f64,
    pub rel_alt_m: f64,
    #[serde(default)]
    pub aircraft_yaw: Option<f64>,
    #[serde(default)]
    pub gimbal_yaw: Option<f64>,
    #[serde(default)]
    pub gimbal_pitch: Option<f64>,
}

impl Row {
    fn finite(&self) -> bool {
        self.t.is_finite() && self.lat.is_finite() && self.lon.is_finite() && self.rel_alt_m.is_finite()
    }
}

/// Where a view's aircraft is at a clip time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub lat: f64,
    pub lon: f64,
    /// Above the takeoff point.
    pub rel_alt_m: f64,
    /// Aircraft yaw (compass degrees, `[0, 360)`).
    pub heading_deg: f64,
    /// Ground speed between the surrounding rows, m/s.
    pub speed: f64,
}

impl Sample {
    /// Absolute GPS, given the takeoff point's ground elevation.
    pub fn gps(&self, ground_alt: f64) -> Gps {
        Gps { lat: self.lat, lon: self.lon, alt: ground_alt + self.rel_alt_m }
    }
}

/// One view's record, validated: rows sorted by time, at least two of them.
#[derive(Debug, Clone, PartialEq)]
pub struct View {
    pub name: String,
    pub video: String,
    /// Length of the joined clip, seconds (the loop period of its publisher).
    pub duration: f64,
    /// Passes, sorted by start. Never empty (a record without segments is one pass).
    pub segments: Vec<Segment>,
    pub rows: Vec<Row>,
    /// Ground speed from each row to the next within its pass, m/s.
    speeds: Vec<f64>,
}

/// The three views of `track.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    pub left: View,
    pub center: View,
    pub right: View,
}

impl Track {
    pub fn load(path: &Path) -> Result<Track, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Track::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Track, String> {
        let file: TrackFile = serde_json::from_str(text).map_err(|e| format!("not a flight record: {e}"))?;
        let mut views = file.views;
        let mut take = |name: &str| -> Result<View, String> {
            let v = views.remove(name).ok_or_else(|| format!("view {name:?} missing"))?;
            View::new(name, v)
        };
        Ok(Track { left: take("left")?, center: take("center")?, right: take("right")? })
    }

    /// Views in formation order (drone 1, 2, 3).
    pub fn views(&self) -> [&View; 3] {
        [&self.left, &self.center, &self.right]
    }

    /// Centre of the box around every row of every view (lat, lon).
    pub fn centre(&self) -> (f64, f64) {
        let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
        for r in self.views().iter().flat_map(|v| v.rows.iter()) {
            lo = (lo.0.min(r.lat), lo.1.min(r.lon));
            hi = (hi.0.max(r.lat), hi.1.max(r.lon));
        }
        ((lo.0 + hi.0) / 2.0, (lo.1 + hi.1) / 2.0)
    }
}

impl View {
    fn new(name: &str, v: ViewFile) -> Result<View, String> {
        if !(v.duration.is_finite() && v.duration > 0.0) {
            return Err(format!("view {name:?}: duration {} is not positive", v.duration));
        }
        let mut rows: Vec<Row> = v.track.into_iter().filter(Row::finite).collect();
        rows.sort_by(|a, b| a.t.total_cmp(&b.t));
        if rows.len() < 2 {
            return Err(format!("view {name:?}: {} usable track row(s), want at least 2", rows.len()));
        }
        let mut segments: Vec<Segment> =
            v.segments.into_iter().filter(|s| s.start.is_finite() && s.rel_alt_m.is_finite()).collect();
        segments.sort_by(|a, b| a.start.total_cmp(&b.start));
        if segments.is_empty() {
            let rel = rows.iter().map(|r| r.rel_alt_m).sum::<f64>() / rows.len() as f64;
            segments.push(Segment { source: String::new(), start: 0.0, duration: v.duration, rel_alt_m: rel });
        }
        let mut view = View { name: name.to_string(), video: v.video, duration: v.duration, segments, rows, speeds: Vec::new() };
        view.speeds = view.pair_speeds();
        Ok(view)
    }

    /// Index of the pass a clip time falls in.
    pub fn segment_of(&self, t: f64) -> usize {
        self.segments.iter().rposition(|s| s.start <= t).unwrap_or(0)
    }

    fn pair_speeds(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.rows.len());
        let mut last = 0.0;
        for (i, a) in self.rows.iter().enumerate() {
            if let Some(b) = self.rows.get(i + 1)
                && self.segment_of(a.t) == self.segment_of(b.t)
                && b.t > a.t
            {
                last = horizontal_m(a, b) / (b.t - a.t);
            }
            out.push(last);
        }
        out
    }

    fn heading_of(&self, i: usize) -> f64 {
        let r = &self.rows[i];
        r.aircraft_yaw.or(r.gimbal_yaw).map_or_else(|| self.course_deg(), |h| h.rem_euclid(360.0))
    }

    /// The aircraft at clip time `t` (seconds into the joined clip). Rows are interpolated
    /// linearly, the heading along the shorter way round. Between the passes there is no
    /// interpolation: the aircraft jumps exactly where the clip cuts.
    pub fn sample(&self, t: f64) -> Sample {
        let rows = &self.rows;
        let at = |i: usize| Sample {
            lat: rows[i].lat,
            lon: rows[i].lon,
            rel_alt_m: rows[i].rel_alt_m,
            heading_deg: self.heading_of(i),
            speed: self.speeds[i],
        };
        let next = rows.partition_point(|r| r.t <= t);
        if next == 0 {
            return at(0);
        }
        let i = next - 1;
        if i + 1 >= rows.len() {
            return at(i);
        }
        let (a, b) = (&rows[i], &rows[i + 1]);
        if self.segment_of(a.t) != self.segment_of(b.t) || b.t <= a.t {
            return at(i);
        }
        let f = ((t - a.t) / (b.t - a.t)).clamp(0.0, 1.0);
        Sample {
            lat: lerp(a.lat, b.lat, f),
            lon: lerp(a.lon, b.lon, f),
            rel_alt_m: lerp(a.rel_alt_m, b.rel_alt_m, f),
            heading_deg: lerp_angle(self.heading_of(i), self.heading_of(i + 1), f),
            speed: self.speeds[i],
        }
    }

    /// Rows of the first pass.
    pub fn first_pass(&self) -> Vec<&Row> {
        self.rows.iter().filter(|r| self.segment_of(r.t) == 0).collect()
    }

    /// Direction of flight on the first pass (its first row to its last), compass degrees.
    /// Not the aircraft yaw: the side views flew the line yawed towards their side.
    pub fn course_deg(&self) -> f64 {
        let pass = self.first_pass();
        let (a, b) = match (pass.first(), pass.last()) {
            (Some(a), Some(b)) if a != b => (*a, *b),
            _ => (&self.rows[0], &self.rows[self.rows.len() - 1]),
        };
        let d = enu_of(b, a);
        bearing_deg(d.e, d.n)
    }

    /// Height of the first pass above takeoff.
    pub fn first_pass_rel_alt(&self) -> f64 {
        self.segments[0].rel_alt_m
    }

    /// One log line: clip, passes, course and gimbal.
    pub fn summary(&self) -> String {
        let passes: Vec<String> =
            self.segments.iter().map(|s| format!("{:.0}초 {:.0} m", s.duration, s.rel_alt_m)).collect();
        let pitches: Vec<f64> = self.rows.iter().filter_map(|r| r.gimbal_pitch).collect();
        let yaws: Vec<f64> = self.rows.iter().filter_map(|r| r.gimbal_yaw).collect();
        let mean = |v: &[f64]| if v.is_empty() { None } else { Some(v.iter().sum::<f64>() / v.len() as f64) };
        let gimbal = match (mean(&yaws), mean(&pitches)) {
            (Some(y), Some(p)) => format!(", 짐벌 방위 {y:.0}° 피치 {p:.0}°"),
            _ => String::new(),
        };
        format!(
            "{} {} {:.1}초 (구간 {}), 진행 방위 {:.0}°{gimbal}",
            self.name,
            self.video,
            self.duration,
            passes.join(" + "),
            self.course_deg()
        )
    }
}

fn lerp(a: f64, b: f64, f: f64) -> f64 {
    a + (b - a) * f
}

/// Angle interpolation the shorter way round, result in `[0, 360)`.
pub fn lerp_angle(a: f64, b: f64, f: f64) -> f64 {
    let d = (b - a + 540.0).rem_euclid(360.0) - 180.0;
    (a + d * f).rem_euclid(360.0)
}

/// Seconds into a looping clip of `duration`, `elapsed` seconds after its publisher started.
pub fn clip_time(elapsed: f64, duration: f64) -> f64 {
    if duration > 0.0 { elapsed.max(0.0).rem_euclid(duration) } else { 0.0 }
}

/// Seconds into a clip that plays once: past the end it stays on the last instant.
pub fn clip_time_once(elapsed: f64, duration: f64) -> f64 {
    if duration > 0.0 { elapsed.max(0.0).min(duration) } else { 0.0 }
}

fn gps_of(r: &Row) -> Gps {
    Gps { lat: r.lat, lon: r.lon, alt: 0.0 }
}

fn enu_of(r: &Row, origin: &Row) -> Enu {
    geo::gps_to_enu(gps_of(r), gps_of(origin))
}

fn horizontal_m(a: &Row, b: &Row) -> f64 {
    let d = enu_of(b, a);
    d.e.hypot(d.n)
}

/// Indices of the points Douglas-Peucker keeps at `tolerance` metres: always both ends, plus
/// every point further than that from the simplified line.
pub fn simplify(points: &[(f64, f64)], tolerance: f64) -> Vec<usize> {
    fn distance(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len2 = dx * dx + dy * dy;
        if len2 == 0.0 {
            return (p.0 - a.0).hypot(p.1 - a.1);
        }
        let s = (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0);
        (p.0 - (a.0 + s * dx)).hypot(p.1 - (a.1 + s * dy))
    }
    fn keep(points: &[(f64, f64)], lo: usize, hi: usize, tol: f64, out: &mut Vec<usize>) {
        if hi <= lo + 1 {
            return;
        }
        let (mut far, mut at) = (0.0, lo);
        for i in lo + 1..hi {
            let d = distance(points[i], points[lo], points[hi]);
            if d > far {
                far = d;
                at = i;
            }
        }
        if far > tol {
            keep(points, lo, at, tol, out);
            out.push(at);
            keep(points, at, hi, tol, out);
        }
    }
    match points.len() {
        0 => vec![],
        1 => vec![0],
        n => {
            let mut out = vec![0];
            keep(points, 0, n - 1, tolerance, &mut out);
            out.push(n - 1);
            out
        }
    }
}

/// The route the tower starts with: the center view's first pass, from its start to its end
/// plus any turn the line makes, at that pass's absolute altitude.
pub fn planned_waypoints(center: &View, ground_alt: f64) -> Vec<Gps> {
    let pass = center.first_pass();
    let Some(origin) = pass.first() else { return Vec::new() };
    let points: Vec<(f64, f64)> = pass.iter().map(|r| {
        let d = enu_of(r, origin);
        (d.e, d.n)
    }).collect();
    let alt = ground_alt + center.first_pass_rel_alt();
    let mut out: Vec<Gps> = Vec::new();
    for i in simplify(&points, TURN_TOLERANCE_M) {
        let g = Gps { lat: pass[i].lat, lon: pass[i].lon, alt };
        if out.last().is_none_or(|p| (p.lat, p.lon) != (g.lat, g.lon)) {
            out.push(g);
        }
    }
    out
}

/// `planned-route.json` for the core (`SKYLENS_CORE_PLANNED_ROUTE`).
pub fn planned_route_json(drone_id: u32, waypoints: &[Gps], looped: bool) -> String {
    let wps: Vec<serde_json::Value> =
        waypoints.iter().map(|g| serde_json::json!({ "lat": g.lat, "lon": g.lon, "alt": g.alt })).collect();
    serde_json::to_string_pretty(&serde_json::json!({ "drone_id": drone_id, "loop": looped, "waypoints": wps }))
        .unwrap_or_default()
}

/// Flying in before the hello: `t` seconds into an approach of `eta` seconds that ends at
/// `target`, coming from `approach_m` behind it along `course_deg`, at the target's height.
/// Returns the fix and the approach speed.
pub fn approach_gps(target: Gps, course_deg: f64, approach_m: f64, t: f64, eta: f64) -> (Gps, f64) {
    let (ue, un) = unit(course_deg);
    let f = if eta > 0.0 { (t / eta).clamp(0.0, 1.0) } else { 1.0 };
    let back = approach_m * (1.0 - f);
    let speed = if eta > 0.0 && f < 1.0 { approach_m / eta } else { 0.0 };
    let g = geo::enu_to_gps(Enu { e: -ue * back, n: -un * back, u: 0.0 }, target);
    (g, speed)
}

/// `SKYLENS_CORE_SITE` in the point form: the footage centre at the ground elevation.
pub fn site_value(centre: (f64, f64), ground_alt: f64) -> String {
    format!("{:.6},{:.6},{:.1}", centre.0, centre.1, ground_alt)
}

/// The anchor a `SKYLENS_CORE_SITE` value gives the core, when the launcher can tell: the point
/// form, or the default preset. `None` for other presets and boxes.
pub fn site_anchor(site: &str) -> Option<Gps> {
    let s = site.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("daejeon") {
        return Some(crate::route::SITE_ANCHOR);
    }
    let nums: Vec<f64> = s.split(',').map(|p| p.trim().parse::<f64>()).collect::<Result<_, _>>().ok()?;
    match nums[..] {
        [lat, lon] => Some(Gps { lat, lon, alt: 0.0 }),
        [lat, lon, alt] => Some(Gps { lat, lon, alt }),
        _ => None,
    }
}

/// `SKYLENS_ANCHOR` for the model server (`lat,lon,alt`), the same origin as the core's.
pub fn anchor_value(a: Gps) -> String {
    format!("{},{},{}", a.lat, a.lon, a.alt)
}

/// The camera estimate (`camera.json`). Only logged: the model server takes no intrinsics.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Camera {
    #[serde(default)]
    pub model: String,
    pub video_width: u32,
    pub video_height: u32,
    pub fx_estimate_px: f64,
    pub fy_estimate_px: f64,
    pub cx: f64,
    pub cy: f64,
}

impl Camera {
    pub fn load(path: &Path) -> Option<Camera> {
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
    }

    /// Intrinsics at a frame `width` (the core scales frames keeping the aspect; 0 = source).
    pub fn scaled(&self, width: u32) -> (f64, f64, f64, f64) {
        let s = if width == 0 || self.video_width == 0 { 1.0 } else { f64::from(width) / f64::from(self.video_width) };
        (self.fx_estimate_px * s, self.fy_estimate_px * s, self.cx * s, self.cy * s)
    }

    /// One log line, intrinsics at the extracted frame width.
    pub fn describe(&self, frame_width: u32) -> String {
        let (fx, fy, cx, cy) = self.scaled(frame_width);
        format!(
            "{} {}x{}, 추출 프레임 기준 추정 fx {fx:.1} fy {fy:.1} cx {cx:.1} cy {cy:.1} px",
            self.model, self.video_width, self.video_height
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    /// A straight line south-south-east: pass 1 high (0..4 s), pass 2 low (4..6 s).
    const RECORD: &str = r#"{
      "views": {
        "left":   {"video": "left.mp4", "duration": 6.0,
                   "segments": [{"source": "a.mp4", "start": 0.0, "duration": 4.0, "rel_alt_m": 44.0},
                                {"source": "b.mp4", "start": 4.0, "duration": 2.0, "rel_alt_m": 22.0}],
                   "track": [{"t": 0.0, "lat": 36.3660, "lon": 127.3420, "rel_alt_m": 44.0, "aircraft_yaw": 350.0},
                             {"t": 2.0, "lat": 36.3656, "lon": 127.3422, "rel_alt_m": 44.0, "aircraft_yaw": 10.0},
                             {"t": 4.0, "lat": 36.3660, "lon": 127.3420, "rel_alt_m": 22.0, "aircraft_yaw": 45.0},
                             {"t": 6.0, "lat": 36.3656, "lon": 127.3422, "rel_alt_m": 22.0, "aircraft_yaw": 45.0}]},
        "center": {"video": "center.mp4", "duration": 6.0,
                   "segments": [{"start": 0.0, "duration": 4.0, "rel_alt_m": 44.0},
                                {"start": 4.0, "duration": 2.0, "rel_alt_m": 22.0}],
                   "track": [{"t": 0.0, "lat": 36.3660, "lon": 127.3420, "rel_alt_m": 43.5},
                             {"t": 1.0, "lat": 36.3658, "lon": 127.3421, "rel_alt_m": 44.0},
                             {"t": 2.0, "lat": 36.3656, "lon": 127.3422, "rel_alt_m": 44.5},
                             {"t": 5.0, "lat": 36.3660, "lon": 127.3420, "rel_alt_m": 22.0}]},
        "right":  {"video": "right.mp4", "duration": 6.0,
                   "track": [{"t": 1.0, "lat": 36.3650, "lon": 127.3430, "rel_alt_m": 30.0, "aircraft_yaw": 285.0},
                             {"t": 0.0, "lat": 36.3652, "lon": 127.3428, "rel_alt_m": 30.0, "aircraft_yaw": 285.0}]}
      },
      "notes": {"anything": "ignored"}
    }"#;

    fn track() -> Track {
        Track::parse(RECORD).unwrap()
    }

    #[test]
    fn parses_three_views() {
        let t = track();
        assert_eq!(t.left.video, "left.mp4");
        assert_eq!(t.left.segments.len(), 2);
        assert_eq!(t.left.segments[0].source, "a.mp4");
        // No segments: one pass over the whole clip at the mean height.
        assert_eq!(t.right.segments.len(), 1);
        assert!(close(t.right.segments[0].rel_alt_m, 30.0, 1e-9));
        // Rows come sorted.
        assert!(t.right.rows[0].t < t.right.rows[1].t);
        assert_eq!(t.views().map(|v| v.name.as_str()), ["left", "center", "right"]);
    }

    #[test]
    fn rejects_broken_records() {
        assert!(Track::parse("nope").is_err());
        assert!(Track::parse(r#"{"views": {}}"#).unwrap_err().contains("left"));
        let one_row = RECORD.replace(
            r#"{"t": 1.0, "lat": 36.3650, "lon": 127.3430, "rel_alt_m": 30.0, "aircraft_yaw": 285.0},"#,
            "",
        );
        assert!(Track::parse(&one_row).unwrap_err().contains("right"));
        let no_duration = RECORD.replace(r#""right.mp4", "duration": 6.0"#, r#""right.mp4", "duration": 0.0"#);
        assert_ne!(no_duration, RECORD);
        assert!(Track::parse(&no_duration).unwrap_err().contains("duration"));
    }

    #[test]
    fn interpolates_within_a_pass() {
        let v = track().left;
        let s = v.sample(1.0);
        assert!(close(s.lat, 36.3658, 1e-12) && close(s.lon, 127.3421, 1e-12));
        assert!(close(s.rel_alt_m, 44.0, 1e-12));
        // 350 -> 10 goes through north, not round the south.
        assert!(close(s.heading_deg, 0.0, 1e-9), "{}", s.heading_deg);
        // ~49 m in 2 s
        let d = horizontal_m(&v.rows[0], &v.rows[1]);
        assert!(close(s.speed, d / 2.0, 1e-9) && d > 40.0 && d < 55.0, "{d}");
    }

    #[test]
    fn jumps_between_passes_without_smoothing() {
        let v = track().left;
        // 3 s: after the last row of pass 1 (2 s), before the first row of pass 2 (4 s): hold.
        let held = v.sample(3.0);
        assert!(close(held.lat, 36.3656, 1e-12) && close(held.rel_alt_m, 44.0, 1e-12));
        // 4 s: pass 2 starts, back at the top of the line and low.
        let low = v.sample(4.0);
        assert!(close(low.lat, 36.3660, 1e-12) && close(low.rel_alt_m, 22.0, 1e-12));
        assert_eq!((v.segment_of(3.9), v.segment_of(4.0)), (0, 1));
        // Before the first row and after the last: the end rows.
        assert_eq!(v.sample(-1.0).lat, v.rows[0].lat);
        assert_eq!(v.sample(99.0).lat, v.rows[3].lat);
        // The center view's rows straddle the cut (2 s in pass 1, 5 s in pass 2).
        let c = track().center;
        assert!(close(c.sample(3.5).rel_alt_m, 44.5, 1e-12));
    }

    #[test]
    fn heading_wraps_and_falls_back_to_the_course() {
        assert!(close(lerp_angle(350.0, 10.0, 0.25), 355.0, 1e-9));
        assert!(close(lerp_angle(10.0, 350.0, 0.5), 0.0, 1e-9));
        assert!(close(lerp_angle(90.0, 180.0, 0.5), 135.0, 1e-9));
        // No yaw in the record: the direction of flight (south-south-east here).
        let c = track().center;
        let h = c.sample(0.5).heading_deg;
        assert!(close(h, c.course_deg(), 1e-9));
        assert!(h > 150.0 && h < 175.0, "{h}");
    }

    #[test]
    fn clip_time_loops_with_the_publisher() {
        assert!(close(clip_time(0.0, 82.1), 0.0, 1e-12));
        assert!(close(clip_time(10.0, 82.1), 10.0, 1e-12));
        assert!(close(clip_time(82.1 + 5.0, 82.1), 5.0, 1e-9));
        assert!(close(clip_time(-3.0, 82.1), 0.0, 1e-12));
        assert_eq!(clip_time(5.0, 0.0), 0.0);
    }

    #[test]
    fn clip_time_once_stops_at_the_end() {
        assert!(close(clip_time_once(10.0, 82.1), 10.0, 1e-12));
        assert!(close(clip_time_once(82.1 + 5.0, 82.1), 82.1, 1e-12));
        assert!(close(clip_time_once(-3.0, 82.1), 0.0, 1e-12));
        assert_eq!(clip_time_once(5.0, 0.0), 0.0);
    }

    #[test]
    fn centre_covers_every_view() {
        let (lat, lon) = track().centre();
        assert!(close(lat, (36.3650 + 36.3660) / 2.0, 1e-12));
        assert!(close(lon, (127.3420 + 127.3430) / 2.0, 1e-12));
        assert_eq!(site_value((lat, lon), 48.04), "36.365500,127.342500,48.0");
    }

    #[test]
    fn simplify_keeps_ends_and_turns() {
        // Straight with GPS jitter: just the ends.
        let line: Vec<(f64, f64)> = (0..20).map(|i| (f64::from(i) * 4.0, if i % 2 == 0 { 1.5 } else { -1.5 })).collect();
        assert_eq!(simplify(&line, TURN_TOLERANCE_M), vec![0, 19]);
        // An L: the corner stays.
        let mut l: Vec<(f64, f64)> = (0..=10).map(|i| (f64::from(i) * 5.0, 0.0)).collect();
        l.extend((1..=10).map(|i| (50.0, f64::from(i) * 5.0)));
        assert_eq!(simplify(&l, TURN_TOLERANCE_M), vec![0, 10, 20]);
        assert_eq!(simplify(&[], 1.0), Vec::<usize>::new());
        assert_eq!(simplify(&[(1.0, 1.0)], 1.0), vec![0]);
    }

    #[test]
    fn planned_route_is_the_first_center_pass() {
        let c = track().center;
        let wps = planned_waypoints(&c, 48.0);
        // Pass 1 is three collinear rows: start and end only, at 48 + 44 m.
        assert_eq!(wps.len(), 2);
        assert_eq!((wps[0].lat, wps[0].lon), (36.3660, 127.3420));
        assert_eq!((wps[1].lat, wps[1].lon), (36.3656, 127.3422));
        assert!(wps.iter().all(|g| close(g.alt, 92.0, 1e-12)));

        let json: serde_json::Value = serde_json::from_str(&planned_route_json(2, &wps, true)).unwrap();
        assert_eq!(json["drone_id"], 2);
        assert_eq!(json["loop"], true);
        assert_eq!(json["waypoints"].as_array().unwrap().len(), 2);
        assert_eq!(json["waypoints"][1]["alt"], 92.0);
    }

    #[test]
    fn approach_ends_on_the_target() {
        let target = Gps { lat: 36.3656, lon: 127.3424, alt: 92.0 };
        let (start, speed) = approach_gps(target, 165.0, 150.0, 0.0, 10.0);
        let d = geo::gps_to_enu(start, target);
        assert!(close(d.e.hypot(d.n), 150.0, 1e-6));
        // behind: north-north-west of the target when flying south-south-east
        assert!(d.n > 0.0 && d.e < 0.0);
        assert!(close(start.alt, 92.0, 1e-9) && close(speed, 15.0, 1e-12));
        let (end, speed) = approach_gps(target, 165.0, 150.0, 10.0, 10.0);
        assert!(close(end.lat, target.lat, 1e-12) && close(end.lon, target.lon, 1e-12) && speed == 0.0);
    }

    #[test]
    fn absolute_altitude_is_ground_plus_relative() {
        let s = track().left.sample(0.0);
        let g = s.gps(48.0);
        assert!(close(g.alt, 92.0, 1e-12));
    }

    #[test]
    fn site_anchors_the_launcher_can_tell() {
        assert_eq!(site_anchor("36.1, 127.2, 50"), Some(Gps { lat: 36.1, lon: 127.2, alt: 50.0 }));
        assert_eq!(site_anchor("36.1,127.2"), Some(Gps { lat: 36.1, lon: 127.2, alt: 0.0 }));
        assert_eq!(site_anchor("daejeon"), Some(crate::route::SITE_ANCHOR));
        assert_eq!(site_anchor(""), Some(crate::route::SITE_ANCHOR));
        assert_eq!(site_anchor("seoul"), None);
        assert_eq!(site_anchor("127.0,36.0,127.02,36.01"), None);
        assert_eq!(anchor_value(Gps { lat: 36.5, lon: 127.25, alt: 48.0 }), "36.5,127.25,48");
    }

    #[test]
    fn camera_scales_with_the_frame_width() {
        let c: Camera = serde_json::from_str(
            r#"{"model":"FC9589","sensor":"x","lens":"y","video_width":1920,"video_height":1080,
                "fx_estimate_px":1329,"fy_estimate_px":1329,"cx":960,"cy":540,"note":"estimate"}"#,
        )
        .unwrap();
        assert_eq!(c.scaled(1920), (1329.0, 1329.0, 960.0, 540.0));
        assert_eq!(c.scaled(0), (1329.0, 1329.0, 960.0, 540.0));
        assert_eq!(c.scaled(960), (664.5, 664.5, 480.0, 270.0));
    }
}
