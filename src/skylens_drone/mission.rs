//! Build a DJI Fly KMZ mission from a SkyLens GPS route.
//!
//! DJI does not publish the consumer-aircraft enum values used by DJI Fly. A KMZ
//! created by the actual Lito X1 is therefore mandatory: this module reads the
//! aircraft and payload identifiers from that file and refuses to fall back to an
//! enterprise model. The generated dialect follows the KMZ/WPML shape used by
//! DJI Fly/Litchi imports; ground-test it after DJI Fly or firmware updates.
//!
//! Port of `legacy/mission_bridge/mission.py`. The request body stays a loose
//! JSON value on purpose: the Python bridge accepted numbers given as strings
//! and reported every bad field with a specific message, and so does this one.

use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use regex::Regex;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::kmz;

pub const WPML_NS: &str = "http://www.dji.com/wpmz/1.0.2";
pub const KML_NS: &str = "http://www.opengis.net/kml/2.2";

/// Input or template cannot safely produce a mission. Maps to HTTP 400.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionError(pub String);

impl MissionError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for MissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for MissionError {}

/// Safety limits applied to every mission.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    pub min_height_m: f64,
    pub max_height_m: f64,
    pub max_route_m: f64,
    pub max_waypoints: f64,
}

impl Default for Limits {
    fn default() -> Self {
        Self { min_height_m: 5.0, max_height_m: 120.0, max_route_m: 5000.0, max_waypoints: 200.0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Waypoint {
    pub lat: f64,
    pub lon: f64,
    pub absolute_alt_msl: f64,
    pub relative_height: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissionResult {
    pub path: PathBuf,
    pub sha256: String,
    pub waypoint_count: usize,
    pub route_distance_m: f64,
    pub takeoff_alt_msl: f64,
    pub warnings: Vec<String>,
}

/// Aircraft identifiers copied from the Lito X1 template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemplateIds {
    pub drone_enum: i64,
    pub drone_sub_enum: i64,
    pub payload_enum: Option<i64>,
}

/// Python `float(value)` with the bridge's messages: bools, null, arrays and
/// objects are rejected, numeric strings are accepted, non-finite values fail.
pub fn number(value: Option<&Value>, label: &str) -> Result<f64, MissionError> {
    let not_a_number = || MissionError(format!("{label} must be a number"));
    let out = match value {
        Some(Value::Number(n)) => n.as_f64().ok_or_else(not_a_number)?,
        Some(Value::String(s)) => s.trim().parse::<f64>().map_err(|_| not_a_number())?,
        _ => return Err(not_a_number()),
    };
    if !out.is_finite() {
        return Err(MissionError(format!("{label} must be finite")));
    }
    Ok(out)
}

/// `_number(obj.get(key, default), label)`: the default applies only when the
/// key is absent; an explicit `null` is an error, as in Python.
pub fn number_or(obj: &Value, key: &str, default: f64, label: &str) -> Result<f64, MissionError> {
    match obj.get(key) {
        None => Ok(default),
        Some(v) => number(Some(v), label),
    }
}

/// Python truthiness of a JSON value (`bool(payload.get("loop"))`).
pub fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// `payload["flight"]` when it is an object, otherwise an empty object.
pub fn flight_of(payload: &Value) -> Value {
    match payload.get("flight") {
        Some(v @ Value::Object(_)) => v.clone(),
        _ => Value::Object(Default::default()),
    }
}

/// Great-circle distance in metres (haversine, R = 6 371 km).
pub fn distance_m(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (lat1, lon1) = (a.0.to_radians(), a.1.to_radians());
    let (lat2, lon2) = (b.0.to_radians(), b.1.to_radians());
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let h = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    6_371_000.0 * 2.0 * h.sqrt().min(1.0).asin()
}

pub fn read_template_ids(template_path: &Path) -> anyhow::Result<TemplateIds> {
    if !template_path.is_file() {
        return Err(MissionError(format!(
            "Lito X1 template KMZ not found: {}. Create and save a dummy waypoint mission in \
             DJI Fly first.",
            template_path.display()
        ))
        .into());
    }
    let bytes = std::fs::read(template_path)
        .with_context(|| format!("reading template {}", template_path.display()))?;
    let invalid = || MissionError::new("template is not a valid KMZ/ZIP file");
    let names = kmz::names(&bytes).map_err(|_| invalid())?;
    let source = names
        .iter()
        .find(|n| n.ends_with("waylines.wpml"))
        .or_else(|| names.iter().find(|n| n.ends_with("template.kml")))
        .ok_or_else(|| MissionError::new("template KMZ has no template.kml or waylines.wpml"))?;
    let raw = kmz::read(&bytes, source).map_err(|_| invalid())?;
    let text = String::from_utf8_lossy(&raw);
    Ok(template_ids_from_text(&text)?)
}

/// Pull the aircraft identifiers out of template XML text.
///
/// The consumer DJI Fly / Litchi KMZ for an integrated-camera aircraft (Lito
/// X1) carries no `<wpml:payloadInfo>`: there is no removable payload to
/// identify. Enterprise templates do. Payload is optional and mirrors the
/// template; emitting a value the template never had would be a guess.
pub fn template_ids_from_text(text: &str) -> Result<TemplateIds, MissionError> {
    let read = |tag: &str| -> Option<i64> {
        let pattern = format!(r"<wpml:{tag}>(-?\d+)</wpml:{tag}>");
        let re = Regex::new(&pattern).expect("static tag pattern");
        re.captures(text).and_then(|c| c[1].parse::<i64>().ok())
    };
    let required = |tag: &str| -> Result<i64, MissionError> {
        read(tag).ok_or_else(|| MissionError(format!("template KMZ has no wpml:{tag}")))
    };
    Ok(TemplateIds {
        drone_enum: required("droneEnumValue")?,
        drone_sub_enum: required("droneSubEnumValue")?,
        payload_enum: read("payloadEnumValue"),
    })
}

/// Validated waypoints, take-off altitude, route length and warnings.
pub struct Prepared {
    pub points: Vec<Waypoint>,
    pub takeoff_alt: f64,
    pub distance: f64,
    pub warnings: Vec<String>,
}

pub fn prepare_waypoints(payload: &Value, limits: &Limits) -> Result<Prepared, MissionError> {
    let raw = match payload.get("waypoints") {
        Some(Value::Array(items)) if items.len() >= 2 => items,
        _ => return Err(MissionError::new("waypoints must contain at least two points")),
    };
    let max_waypoints = limits.max_waypoints as i64;
    if raw.len() as i64 > max_waypoints {
        return Err(MissionError(format!("waypoint count exceeds {max_waypoints}")));
    }

    let flight = flight_of(payload);
    let agl = number_or(&flight, "aglM", 30.0, "flight.aglM")?;
    if !(limits.min_height_m <= agl && agl <= limits.max_height_m) {
        return Err(MissionError(format!(
            "flight.aglM must be within {:.0}-{:.0} m",
            limits.min_height_m, limits.max_height_m
        )));
    }

    let mut parsed = Vec::with_capacity(raw.len());
    for (index, item) in raw.iter().enumerate() {
        if !item.is_object() {
            return Err(MissionError(format!("waypoint {index} must be an object")));
        }
        let lat = number(item.get("lat"), &format!("waypoint {index}.lat"))?;
        let lon = number(item.get("lon"), &format!("waypoint {index}.lon"))?;
        let alt = number(item.get("alt"), &format!("waypoint {index}.alt"))?;
        if !(-85.0..=85.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
            return Err(MissionError(format!("waypoint {index} is outside valid WGS84 bounds")));
        }
        parsed.push((lat, lon, alt));
    }

    let mut warnings = Vec::new();
    let takeoff_alt = match flight.get("takeoffAltMsl") {
        None | Some(Value::Null) => {
            warnings.push(
                "takeoffAltMsl was inferred from waypoint 1 and AGL; confirm the actual takeoff \
                 elevation in DJI Fly"
                    .to_owned(),
            );
            parsed[0].2 - agl
        }
        Some(v) => number(Some(v), "flight.takeoffAltMsl")?,
    };

    let mut points: Vec<Waypoint> = parsed
        .iter()
        .map(|&(lat, lon, alt)| Waypoint {
            lat,
            lon,
            absolute_alt_msl: alt,
            relative_height: alt - takeoff_alt,
        })
        .collect();
    for (index, point) in points.iter().enumerate() {
        let h = point.relative_height;
        if !(limits.min_height_m <= h && h <= limits.max_height_m) {
            return Err(MissionError(format!(
                "waypoint {index} relative height {h:.1} m is outside {:.0}-{:.0} m",
                limits.min_height_m, limits.max_height_m
            )));
        }
    }

    let mut distance: f64 =
        points.windows(2).map(|w| distance_m((w[0].lat, w[0].lon), (w[1].lat, w[1].lon))).sum();
    if distance > limits.max_route_m {
        return Err(MissionError(format!(
            "route distance {distance:.0} m exceeds configured limit {:.0} m",
            limits.max_route_m
        )));
    }

    if truthy(payload.get("loop")) {
        // DJI Fly has no infinite-loop contract. Make one inspectable out-and-back
        // mission and let the operator explicitly start another sortie if needed.
        let back: Vec<Waypoint> = points[..points.len() - 1].iter().rev().copied().collect();
        points.extend(back);
        distance *= 2.0;
        warnings.push(
            "loop was converted to one out-and-back pass; DJI Fly will not repeat indefinitely"
                .to_owned(),
        );
        if points.len() as i64 > max_waypoints {
            return Err(MissionError::new("out-and-back conversion exceeds the waypoint limit"));
        }
    }

    Ok(Prepared { points, takeoff_alt, distance, warnings })
}

fn mission_config(ids: &TemplateIds, speed: f64, rth: f64) -> String {
    let payload = match ids.payload_enum {
        Some(value) => format!(
            "
      <wpml:payloadInfo>
        <wpml:payloadEnumValue>{value}</wpml:payloadEnumValue>
        <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
      </wpml:payloadInfo>"
        ),
        None => String::new(),
    };
    format!(
        "    <wpml:missionConfig>
      <wpml:flyToWaylineMode>safely</wpml:flyToWaylineMode>
      <wpml:finishAction>goHome</wpml:finishAction>
      <wpml:exitOnRCLost>executeLostAction</wpml:exitOnRCLost>
      <wpml:executeRCLostAction>hover</wpml:executeRCLostAction>
      <wpml:takeOffSecurityHeight>{rth:.2}</wpml:takeOffSecurityHeight>
      <wpml:globalTransitionalSpeed>{speed:.2}</wpml:globalTransitionalSpeed>
      <wpml:globalRTHHeight>{rth:.2}</wpml:globalRTHHeight>
      <wpml:droneInfo>
        <wpml:droneEnumValue>{}</wpml:droneEnumValue>
        <wpml:droneSubEnumValue>{}</wpml:droneSubEnumValue>
      </wpml:droneInfo>{payload}
    </wpml:missionConfig>",
        ids.drone_enum, ids.drone_sub_enum
    )
}

fn actions(index: usize, last: usize, gimbal_pitch: f64) -> String {
    let mut rows = vec![format!(
        "          <wpml:action>
            <wpml:actionId>0</wpml:actionId>
            <wpml:actionActuatorFunc>gimbalRotate</wpml:actionActuatorFunc>
            <wpml:actionActuatorFuncParam>
              <wpml:gimbalRotateMode>absoluteAngle</wpml:gimbalRotateMode>
              <wpml:gimbalPitchRotateEnable>1</wpml:gimbalPitchRotateEnable>
              <wpml:gimbalPitchRotateAngle>{gimbal_pitch:.2}</wpml:gimbalPitchRotateAngle>
              <wpml:gimbalRollRotateEnable>0</wpml:gimbalRollRotateEnable>
              <wpml:gimbalYawRotateEnable>0</wpml:gimbalYawRotateEnable>
              <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
            </wpml:actionActuatorFuncParam>
          </wpml:action>"
    )];
    if index == 0 || index == last {
        let func = if index == 0 { "startRecord" } else { "stopRecord" };
        rows.push(format!(
            "          <wpml:action>
            <wpml:actionId>1</wpml:actionId>
            <wpml:actionActuatorFunc>{func}</wpml:actionActuatorFunc>
            <wpml:actionActuatorFuncParam>
              <wpml:payloadPositionIndex>0</wpml:payloadPositionIndex>
            </wpml:actionActuatorFuncParam>
          </wpml:action>"
        ));
    }
    format!(
        "        <wpml:actionGroup>
          <wpml:actionGroupId>{index}</wpml:actionGroupId>
          <wpml:actionGroupStartIndex>{index}</wpml:actionGroupStartIndex>
          <wpml:actionGroupEndIndex>{index}</wpml:actionGroupEndIndex>
          <wpml:actionGroupMode>sequence</wpml:actionGroupMode>
          <wpml:actionTrigger><wpml:actionTriggerType>reachPoint</wpml:actionTriggerType></wpml:actionTrigger>
{}
        </wpml:actionGroup>",
        rows.join("\n")
    )
}

fn template_kml(
    points: &[Waypoint],
    ids: &TemplateIds,
    speed: f64,
    rth: f64,
    pitch: f64,
    timestamp: i64,
) -> String {
    let marks: Vec<String> = points
        .iter()
        .enumerate()
        .map(|(index, p)| {
            format!(
                "      <Placemark>
        <Point><coordinates>{:.8},{:.8}</coordinates></Point>
        <wpml:index>{index}</wpml:index>
        <wpml:height>{:.2}</wpml:height>
        <wpml:ellipsoidHeight>{:.2}</wpml:ellipsoidHeight>
        <wpml:useGlobalHeight>0</wpml:useGlobalHeight>
        <wpml:useGlobalSpeed>1</wpml:useGlobalSpeed>
        <wpml:useGlobalHeadingParam>1</wpml:useGlobalHeadingParam>
        <wpml:useGlobalTurnParam>1</wpml:useGlobalTurnParam>
        <wpml:gimbalPitchAngle>{pitch:.2}</wpml:gimbalPitchAngle>
      </Placemark>",
                p.lon, p.lat, p.relative_height, p.absolute_alt_msl
            )
        })
        .collect();
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="{KML_NS}" xmlns:wpml="{WPML_NS}">
  <Document>
    <wpml:author>SkyLens</wpml:author>
    <wpml:createTime>{timestamp}</wpml:createTime>
    <wpml:updateTime>{timestamp}</wpml:updateTime>
{config}
    <Folder>
      <wpml:templateType>waypoint</wpml:templateType>
      <wpml:templateId>0</wpml:templateId>
      <wpml:waylineCoordinateSysParam>
        <wpml:coordinateMode>WGS84</wpml:coordinateMode>
        <wpml:heightMode>relativeToStartPoint</wpml:heightMode>
        <wpml:positioningType>GPS</wpml:positioningType>
      </wpml:waylineCoordinateSysParam>
      <wpml:autoFlightSpeed>{speed:.2}</wpml:autoFlightSpeed>
      <wpml:gimbalPitchMode>usePointSetting</wpml:gimbalPitchMode>
      <wpml:globalWaypointHeadingParam><wpml:waypointHeadingMode>followWayline</wpml:waypointHeadingMode></wpml:globalWaypointHeadingParam>
      <wpml:globalWaypointTurnMode>toPointAndStopWithContinuityCurvature</wpml:globalWaypointTurnMode>
      <wpml:globalUseStraightLine>1</wpml:globalUseStraightLine>
{marks}
    </Folder>
  </Document>
</kml>
"#,
        config = mission_config(ids, speed, rth),
        marks = marks.join("\n"),
    )
}

fn waylines_wpml(points: &[Waypoint], ids: &TemplateIds, speed: f64, rth: f64, pitch: f64) -> String {
    let last = points.len() - 1;
    let marks: Vec<String> = points
        .iter()
        .enumerate()
        .map(|(index, p)| {
            format!(
                "      <Placemark>
        <Point><coordinates>{:.8},{:.8}</coordinates></Point>
        <wpml:index>{index}</wpml:index>
        <wpml:executeHeight>{:.2}</wpml:executeHeight>
        <wpml:waypointSpeed>{speed:.2}</wpml:waypointSpeed>
        <wpml:waypointHeadingParam><wpml:waypointHeadingMode>followWayline</wpml:waypointHeadingMode></wpml:waypointHeadingParam>
        <wpml:waypointTurnParam>
          <wpml:waypointTurnMode>toPointAndStopWithContinuityCurvature</wpml:waypointTurnMode>
          <wpml:waypointTurnDampingDist>0.20</wpml:waypointTurnDampingDist>
        </wpml:waypointTurnParam>
        <wpml:useStraightLine>1</wpml:useStraightLine>
{}
      </Placemark>",
                p.lon,
                p.lat,
                p.relative_height,
                actions(index, last, pitch)
            )
        })
        .collect();
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="{KML_NS}" xmlns:wpml="{WPML_NS}">
  <Document>
{config}
    <Folder>
      <wpml:templateId>0</wpml:templateId>
      <wpml:waylineId>0</wpml:waylineId>
      <wpml:autoFlightSpeed>{speed:.2}</wpml:autoFlightSpeed>
      <wpml:executeHeightMode>relativeToStartPoint</wpml:executeHeightMode>
{marks}
    </Folder>
  </Document>
</kml>
"#,
        config = mission_config(ids, speed, rth),
        marks = marks.join("\n"),
    )
}

/// Validate a SkyLens route and write a Lito-identified DJI Fly KMZ.
///
/// Errors are `anyhow` so callers can tell a [`MissionError`] (bad input, 400)
/// from an I/O failure (500) with `downcast_ref`.
pub fn build_mission(
    payload: &Value,
    template_path: &Path,
    output_path: &Path,
    limits: Option<Limits>,
) -> anyhow::Result<MissionResult> {
    let limits = limits.unwrap_or_default();
    let ids = read_template_ids(template_path)?;
    let prepared = prepare_waypoints(payload, &limits)?;
    let flight = flight_of(payload);
    let speed = number_or(&flight, "speedMps", 2.0, "flight.speedMps")?;
    let pitch = number_or(&flight, "gimbalPitchDeg", -30.0, "flight.gimbalPitchDeg")?;
    let rth = number_or(&flight, "rthHeightM", 30.0, "flight.rthHeightM")?;
    if !(0.5..=10.0).contains(&speed) {
        return Err(MissionError::new("flight.speedMps must be within 0.5-10 m/s").into());
    }
    if !(-90.0..=30.0).contains(&pitch) {
        return Err(
            MissionError::new("flight.gimbalPitchDeg must be within -90 to 30 degrees").into()
        );
    }
    if !(limits.min_height_m <= rth && rth <= limits.max_height_m) {
        return Err(
            MissionError::new("flight.rthHeightM is outside the configured height limits").into()
        );
    }

    let timestamp = chrono::Utc::now().timestamp_millis();
    let points = &prepared.points;
    let template = template_kml(points, &ids, speed, rth, pitch, timestamp);
    let waylines = waylines_wpml(points, &ids, speed, rth, pitch);
    let bytes = kmz::write(&[
        ("wpmz/template.kml", template.as_bytes()),
        ("wpmz/waylines.wpml", waylines.as_bytes()),
    ])
    .context("writing KMZ archive")?;

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(output_path, &bytes)
        .with_context(|| format!("writing {}", output_path.display()))?;
    let sha256 = hex::encode(Sha256::digest(&bytes));
    Ok(MissionResult {
        path: output_path.to_path_buf(),
        sha256,
        waypoint_count: points.len(),
        route_distance_m: prepared.distance,
        takeoff_alt_msl: prepared.takeoff_alt,
        warnings: prepared.warnings,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    /// A stand-in for the KMZ DJI Fly saves on a real Lito X1.
    pub(crate) fn write_lito_template(path: &Path) {
        let xml = r#"<?xml version="1.0"?>
<kml xmlns:wpml="http://www.dji.com/wpmz/1.0.2">
  <wpml:droneEnumValue>123</wpml:droneEnumValue>
  <wpml:droneSubEnumValue>7</wpml:droneSubEnumValue>
  <wpml:payloadEnumValue>456</wpml:payloadEnumValue>
</kml>
"#;
        let bytes = kmz::write(&[("wpmz/template.kml", xml.as_bytes())]).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    pub(crate) fn payload(do_loop: bool) -> Value {
        json!({
            "name": "test-route",
            "droneId": 1,
            "loop": do_loop,
            "flight": {
                "aglM": 30,
                "takeoffAltMsl": 50,
                "speedMps": 2,
                "gimbalPitchDeg": -35,
                "rthHeightM": 40
            },
            "waypoints": [
                {"lat": 36.3685, "lon": 127.3475, "alt": 80},
                {"lat": 36.3687, "lon": 127.3477, "alt": 85}
            ]
        })
    }

    fn mission_error(err: &anyhow::Error) -> String {
        err.downcast_ref::<MissionError>().expect("expected a MissionError").to_string()
    }

    fn read_entry(path: &Path, name: &str) -> String {
        let bytes = std::fs::read(path).unwrap();
        String::from_utf8(kmz::read(&bytes, name).unwrap()).unwrap()
    }

    #[test]
    fn builds_dji_kmz_with_template_aircraft_ids() {
        let dir = tempfile::tempdir().unwrap();
        let template = dir.path().join("lito-template.kmz");
        let output = dir.path().join("route.kmz");
        write_lito_template(&template);

        let result = build_mission(&payload(false), &template, &output, None).unwrap();

        assert_eq!(result.waypoint_count, 2);
        assert_eq!(result.takeoff_alt_msl, 50.0);
        assert_eq!(result.sha256.len(), 64);
        let bytes = std::fs::read(&output).unwrap();
        let mut names = kmz::names(&bytes).unwrap();
        names.sort();
        assert_eq!(names, vec!["wpmz/template.kml", "wpmz/waylines.wpml"]);
        assert_eq!(result.sha256, hex::encode(Sha256::digest(&bytes)));
        let waylines = read_entry(&output, "wpmz/waylines.wpml");
        assert!(waylines.contains("<wpml:droneEnumValue>123</wpml:droneEnumValue>"));
        assert!(waylines.contains("<wpml:payloadEnumValue>456</wpml:payloadEnumValue>"));
        assert!(waylines.contains("<wpml:executeHeight>30.00</wpml:executeHeight>"));
        assert!(waylines.contains("<wpml:executeHeight>35.00</wpml:executeHeight>"));
        assert!(waylines.contains("<wpml:actionActuatorFunc>startRecord</wpml:actionActuatorFunc>"));
        assert!(waylines.contains("<wpml:actionActuatorFunc>stopRecord</wpml:actionActuatorFunc>"));
        assert!(waylines.contains("<coordinates>127.34750000,36.36850000</coordinates>"));
        assert!(waylines.contains("<wpml:gimbalPitchRotateAngle>-35.00</wpml:gimbalPitchRotateAngle>"));
        let template_kml = read_entry(&output, "wpmz/template.kml");
        assert!(template_kml.contains("<wpml:ellipsoidHeight>85.00</wpml:ellipsoidHeight>"));
        assert!(template_kml.contains("<wpml:globalRTHHeight>40.00</wpml:globalRTHHeight>"));
    }

    #[test]
    fn loop_becomes_one_out_and_back_pass() {
        let dir = tempfile::tempdir().unwrap();
        let template = dir.path().join("lito-template.kmz");
        let output = dir.path().join("route.kmz");
        write_lito_template(&template);

        let result = build_mission(&payload(true), &template, &output, None).unwrap();

        assert_eq!(result.waypoint_count, 3);
        assert!(result.warnings.iter().any(|w| w.contains("out-and-back")));
        let waylines = read_entry(&output, "wpmz/waylines.wpml");
        assert_eq!(waylines.matches("<wpml:index>").count(), 3);
    }

    #[test]
    fn refuses_generation_without_real_template() {
        let dir = tempfile::tempdir().unwrap();
        let err = build_mission(
            &payload(false),
            &dir.path().join("missing.kmz"),
            &dir.path().join("route.kmz"),
            None,
        )
        .unwrap_err();
        assert!(mission_error(&err).contains("template KMZ not found"));
        assert!(!dir.path().join("route.kmz").exists());
    }

    #[test]
    fn rejects_height_outside_safety_limits() {
        let dir = tempfile::tempdir().unwrap();
        let template = dir.path().join("lito-template.kmz");
        write_lito_template(&template);
        let mut bad = payload(false);
        bad["waypoints"][1]["alt"] = json!(250);

        let err = build_mission(&bad, &template, &dir.path().join("route.kmz"), None).unwrap_err();
        assert!(mission_error(&err).contains("relative height"));
    }

    #[test]
    fn rejects_a_file_that_is_not_a_zip() {
        let dir = tempfile::tempdir().unwrap();
        let template = dir.path().join("lito-template.kmz");
        std::fs::write(&template, b"not a zip").unwrap();
        let err = build_mission(&payload(false), &template, &dir.path().join("r.kmz"), None)
            .unwrap_err();
        assert_eq!(mission_error(&err), "template is not a valid KMZ/ZIP file");
    }

    #[test]
    fn template_without_payload_omits_payload_block() {
        let ids = template_ids_from_text(
            "<wpml:droneEnumValue>68</wpml:droneEnumValue>\
             <wpml:droneSubEnumValue>0</wpml:droneSubEnumValue>",
        )
        .unwrap();
        assert_eq!(ids, TemplateIds { drone_enum: 68, drone_sub_enum: 0, payload_enum: None });
        assert!(!mission_config(&ids, 2.0, 30.0).contains("payloadInfo"));
        let missing = template_ids_from_text("<wpml:droneEnumValue>68</wpml:droneEnumValue>");
        assert_eq!(missing.unwrap_err().0, "template KMZ has no wpml:droneSubEnumValue");
    }

    #[test]
    fn missing_takeoff_altitude_is_inferred_with_a_warning() {
        let mut p = payload(false);
        p["flight"].as_object_mut().unwrap().remove("takeoffAltMsl");
        let prepared = prepare_waypoints(&p, &Limits::default()).unwrap();
        assert_eq!(prepared.takeoff_alt, 50.0); // 80 m first waypoint - 30 m AGL
        assert!(prepared.warnings[0].contains("takeoffAltMsl was inferred"));
    }

    #[test]
    fn input_errors_carry_the_python_messages() {
        let limits = Limits::default();
        let mut p = payload(false);
        p["waypoints"] = json!([{"lat": 1, "lon": 2, "alt": 3}]);
        assert_eq!(
            prepare_waypoints(&p, &limits).err().unwrap().0,
            "waypoints must contain at least two points"
        );

        let mut p = payload(false);
        p["flight"]["aglM"] = json!(true);
        assert_eq!(prepare_waypoints(&p, &limits).err().unwrap().0, "flight.aglM must be a number");

        let mut p = payload(false);
        p["flight"]["aglM"] = json!(200);
        assert_eq!(
            prepare_waypoints(&p, &limits).err().unwrap().0,
            "flight.aglM must be within 5-120 m"
        );

        let mut p = payload(false);
        p["waypoints"][0]["lat"] = json!(89.0);
        assert_eq!(
            prepare_waypoints(&p, &limits).err().unwrap().0,
            "waypoint 0 is outside valid WGS84 bounds"
        );

        let mut p = payload(false);
        p["waypoints"][1] = json!(5);
        assert_eq!(prepare_waypoints(&p, &limits).err().unwrap().0, "waypoint 1 must be an object");

        let mut p = payload(false);
        p["waypoints"][1]["lat"] = json!("36.3687"); // numeric strings pass, as in Python
        assert!(prepare_waypoints(&p, &limits).is_ok());

        let mut p = payload(false);
        p["waypoints"][1]["lon"] = json!("NaN");
        assert_eq!(prepare_waypoints(&p, &limits).err().unwrap().0, "waypoint 1.lon must be finite");

        let mut p = payload(false);
        p["waypoints"][1] = json!({"lat": 36.5, "lon": 127.3477, "alt": 85});
        let err = prepare_waypoints(&p, &limits).err().unwrap().0;
        assert!(err.starts_with("route distance ") && err.ends_with(" m exceeds configured limit 5000 m"));
    }

    #[test]
    fn flight_parameters_are_range_checked() {
        let dir = tempfile::tempdir().unwrap();
        let template = dir.path().join("lito-template.kmz");
        write_lito_template(&template);
        let out = dir.path().join("r.kmz");
        for (key, value, message) in [
            ("speedMps", json!(11), "flight.speedMps must be within 0.5-10 m/s"),
            ("gimbalPitchDeg", json!(45), "flight.gimbalPitchDeg must be within -90 to 30 degrees"),
            ("rthHeightM", json!(2), "flight.rthHeightM is outside the configured height limits"),
        ] {
            let mut p = payload(false);
            p["flight"][key] = value;
            let err = build_mission(&p, &template, &out, None).unwrap_err();
            assert_eq!(mission_error(&err), message);
        }
    }

    #[test]
    fn haversine_matches_python() {
        // math.asin based value from mission._distance_m for the test route.
        let d = distance_m((36.3685, 127.3475), (36.3687, 127.3477));
        assert!((d - 28.552444896991414).abs() < 1e-9, "{d}");
    }
}
