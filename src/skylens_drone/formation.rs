//! Fan one planned route into a three-aircraft formation of DJI Fly missions.
//!
//! SkyLens flies three aircraft one after another to stand in for a simultaneous
//! formation. Doing that by hand meant computing three offset waypoint sets in a
//! spreadsheet and typing each into DJI Fly. This module does the same
//! arithmetic and writes the three KMZ missions directly.
//!
//! Geometry. The planned route is the formation CENTROID track. The three
//! members sit at the vertices of an equilateral triangle of side `spacing_m`
//! in the horizontal plane, translated rigidly with the route:
//!
//! ```text
//! front-left  (station 'left')   forward +R/2, lateral -s/2
//! front-right (station 'right')  forward +R/2, lateral +s/2
//! rear-centre (station 'center') forward -R,   lateral  0        R = s/sqrt(3)
//! ```
//!
//! Forward is the route bearing (first waypoint -> last); lateral is positive
//! to the right of travel. Altitude is left untouched: every member keeps the
//! planned per-waypoint GPS altitude, so the mission builder derives the same
//! relative heights against the shared take-off point.
//!
//! Port of `legacy/mission_bridge/formation.py`.

use std::path::Path;

use serde_json::{Value, json};

use crate::mission::{Limits, MissionError, MissionResult, build_mission, flight_of, number, truthy};

/// Metres per degree of latitude; longitude is scaled by cos(lat) at use. Flat
/// approximation: sub-centimetre error over a formation offset of tens of metres.
pub const M_PER_DEG_LAT: f64 = 111_320.0;

/// Formation member stations in the order the missions are written.
#[cfg(test)]
pub const STATIONS: [&str; 3] = ["left", "right", "center"];

/// `(station, forward_m, lateral_m)` for an equilateral triangle of side `spacing_m`.
pub fn member_offsets(spacing_m: f64) -> Result<[(&'static str, f64, f64); 3], MissionError> {
    if !spacing_m.is_finite() || spacing_m <= 0.0 {
        return Err(MissionError::new("formation.spacingM must be a positive number"));
    }
    let r = spacing_m / 3.0_f64.sqrt();
    Ok([
        ("left", r / 2.0, -spacing_m / 2.0),  // front-left
        ("right", r / 2.0, spacing_m / 2.0),  // front-right
        ("center", -r, 0.0),                   // rear-centre
    ])
}

/// A waypoint parsed loosely, as the formation code did before validation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gps {
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
}

/// Bearing (degrees from north) of the straight line first -> last waypoint.
pub fn route_bearing_deg(waypoints: &[Gps]) -> Result<f64, MissionError> {
    let (first, last) = (waypoints[0], waypoints[waypoints.len() - 1]);
    let lat0 = first.lat.to_radians();
    let east = (last.lon - first.lon) * M_PER_DEG_LAT * lat0.cos();
    let north = (last.lat - first.lat) * M_PER_DEG_LAT;
    if east == 0.0 && north == 0.0 {
        return Err(MissionError::new("route start and end coincide; cannot derive a heading"));
    }
    Ok(east.atan2(north).to_degrees())
}

fn shift(lat: f64, lon: f64, east_m: f64, north_m: f64) -> (f64, f64) {
    let dlat = north_m / M_PER_DEG_LAT;
    let dlon = east_m / (M_PER_DEG_LAT * lat.to_radians().cos());
    (lat + dlat, lon + dlon)
}

/// Rigidly translate every waypoint by a `(forward, lateral)` offset.
pub fn offset_waypoints(waypoints: &[Gps], heading_deg: f64, forward_m: f64, lateral_m: f64) -> Vec<Gps> {
    let h = heading_deg.to_radians();
    let east = forward_m * h.sin() + lateral_m * h.cos();
    let north = forward_m * h.cos() - lateral_m * h.sin();
    waypoints
        .iter()
        .map(|wp| {
            let (lat, lon) = shift(wp.lat, wp.lon, east, north);
            Gps { lat, lon, alt: wp.alt }
        })
        .collect()
}

/// Parse `payload.waypoints` for the formation geometry.
///
/// Python called `float()` on these directly, so a malformed waypoint surfaced
/// as an unhandled exception (HTTP 500). Here it is a [`MissionError`] (400)
/// with the same message the single-mission validator would give.
fn parse_waypoints(payload: &Value) -> Result<Vec<Gps>, MissionError> {
    let raw = match payload.get("waypoints") {
        Some(Value::Array(items)) if items.len() >= 2 => items,
        _ => return Err(MissionError::new("waypoints must contain at least two points")),
    };
    raw.iter()
        .enumerate()
        .map(|(index, item)| {
            if !item.is_object() {
                return Err(MissionError(format!("waypoint {index} must be an object")));
            }
            Ok(Gps {
                lat: number(item.get("lat"), &format!("waypoint {index}.lat"))?,
                lon: number(item.get("lon"), &format!("waypoint {index}.lon"))?,
                alt: number(item.get("alt"), &format!("waypoint {index}.alt"))?,
            })
        })
        .collect()
}

fn spacing_of(payload: &Value) -> Result<f64, MissionError> {
    match payload.get("formation") {
        Some(Value::Object(f)) => match f.get("spacingM") {
            None | Some(Value::Null) => Ok(10.0),
            Some(v) => number(Some(v), "formation.spacingM")
                .map_err(|_| MissionError::new("formation.spacingM must be a positive number")),
        },
        _ => Ok(10.0),
    }
}

/// Write one KMZ per formation member. Returns `[(station, result), ...]`.
pub fn build_formation(
    payload: &Value,
    template_path: &Path,
    output_dir: &Path,
    base_name: &str,
    stamp: &str,
    limits: Option<Limits>,
) -> anyhow::Result<Vec<(&'static str, MissionResult)>> {
    let raw = parse_waypoints(payload)?;
    let offsets = member_offsets(spacing_of(payload)?)?;
    let heading = route_bearing_deg(&raw)?;
    let flight = flight_of(payload);
    let do_loop = truthy(payload.get("loop"));

    let mut results = Vec::with_capacity(offsets.len());
    for (station, forward_m, lateral_m) in offsets {
        let waypoints: Vec<Value> = offset_waypoints(&raw, heading, forward_m, lateral_m)
            .into_iter()
            .map(|g| json!({"lat": g.lat, "lon": g.lon, "alt": g.alt}))
            .collect();
        let member_payload = json!({
            "name": format!("{base_name}-{station}"),
            "loop": do_loop,
            "flight": flight,
            "waypoints": waypoints,
        });
        let output_path = output_dir.join(format!("{stamp}-{base_name}-{station}.kmz"));
        results.push((station, build_mission(&member_payload, template_path, &output_path, limits)?));
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mission::tests::{payload, write_lito_template};

    fn route() -> Vec<Gps> {
        vec![
            Gps { lat: 36.3685, lon: 127.3475, alt: 80.0 },
            Gps { lat: 36.3687, lon: 127.3477, alt: 85.0 },
        ]
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-11, "{a} != {b}");
    }

    /// Local east/north metres between two points, same flat model.
    fn en(a: Gps, b: Gps) -> (f64, f64) {
        let east = (b.lon - a.lon) * M_PER_DEG_LAT * a.lat.to_radians().cos();
        let north = (b.lat - a.lat) * M_PER_DEG_LAT;
        (east, north)
    }

    #[test]
    fn offsets_form_an_equilateral_triangle() {
        let offsets = member_offsets(10.0).unwrap();
        let pts: Vec<(f64, f64)> = offsets.iter().map(|(_, f, l)| (*f, *l)).collect();
        let side = |a: (f64, f64), b: (f64, f64)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
        close(side(pts[0], pts[1]), 10.0);
        close(side(pts[1], pts[2]), 10.0);
        close(side(pts[2], pts[0]), 10.0);
        // centroid on the planned track
        close(pts.iter().map(|p| p.0).sum::<f64>(), 0.0);
        close(pts.iter().map(|p| p.1).sum::<f64>(), 0.0);
        assert_eq!(offsets.map(|o| o.0), STATIONS);
    }

    #[test]
    fn rejects_non_positive_spacing() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                member_offsets(bad).unwrap_err().0,
                "formation.spacingM must be a positive number"
            );
        }
    }

    #[test]
    fn bearing_and_offsets_match_python_vectors() {
        // Values printed by legacy formation.py for the same route and spacing 10.
        let heading = route_bearing_deg(&route()).unwrap();
        close(heading, 38.841710110358875);
        let expect = [
            ("left", [(36.3685483677008, 127.34747675157107), (36.368748367700796, 127.34767675151132)]),
            ("right", [(36.36849202823019, 127.34756364435992), (36.36869202823019, 127.34776364452354)]),
            ("center", [(36.368459604069, 127.347459604069), (36.368659604069, 127.34765960396517)]),
        ];
        for ((station, f, l), (want_station, want)) in member_offsets(10.0).unwrap().into_iter().zip(expect) {
            assert_eq!(station, want_station);
            let out = offset_waypoints(&route(), heading, f, l);
            for (got, (lat, lon)) in out.iter().zip(want) {
                close(got.lat, lat);
                close(got.lon, lon);
            }
            assert_eq!(out[0].alt, 80.0);
            assert_eq!(out[1].alt, 85.0);
        }
    }

    #[test]
    fn front_pair_is_spacing_apart_across_track() {
        let heading = route_bearing_deg(&route()).unwrap();
        let o = member_offsets(10.0).unwrap();
        let left = offset_waypoints(&route(), heading, o[0].1, o[0].2)[0];
        let right = offset_waypoints(&route(), heading, o[1].1, o[1].2)[0];
        let (e, n) = en(left, right);
        assert!(((e * e + n * n).sqrt() - 10.0).abs() < 1e-3);
    }

    #[test]
    fn coincident_start_and_end_have_no_heading() {
        let mut r = route();
        r[1] = r[0];
        assert_eq!(
            route_bearing_deg(&r).unwrap_err().0,
            "route start and end coincide; cannot derive a heading"
        );
    }

    #[test]
    fn writes_three_member_missions() {
        let dir = tempfile::tempdir().unwrap();
        let template = dir.path().join("lito-template.kmz");
        write_lito_template(&template);
        let mut p = payload(false);
        p["formation"] = json!({"spacingM": 12});

        let members =
            build_formation(&p, &template, dir.path(), "route", "20260101-000000", None).unwrap();

        assert_eq!(members.iter().map(|m| m.0).collect::<Vec<_>>(), STATIONS);
        for (station, result) in &members {
            assert_eq!(result.waypoint_count, 2);
            assert_eq!(result.takeoff_alt_msl, 50.0);
            assert_eq!(
                result.path.file_name().unwrap().to_str().unwrap(),
                format!("20260101-000000-route-{station}.kmz")
            );
            assert!(result.path.is_file());
        }
    }

    #[test]
    fn formation_requires_a_template() {
        let dir = tempfile::tempdir().unwrap();
        let err = build_formation(
            &payload(false),
            &dir.path().join("missing.kmz"),
            dir.path(),
            "route",
            "stamp",
            None,
        )
        .unwrap_err();
        assert!(err.downcast_ref::<MissionError>().unwrap().0.contains("template KMZ not found"));
    }
}
