//! Client side of the DJI Fly mission bridge (`skylens_drone`, default `http://127.0.0.1:8091`),
//! ported from the legacy `missionBridge.ts`: request bodies, response parsing and the error
//! wording the toasts show. The transport is [`crate::http`].

use serde_json::{Value, json};
use skylens_shared::proto::model::v1 as model;

pub const DEFAULT_BRIDGE: &str = "http://127.0.0.1:8091";

/// Timeouts of the legacy client (ms).
pub const HEALTH_TIMEOUT_MS: u64 = 4_000;
pub const DOWNLOAD_TIMEOUT_MS: u64 = 10_000;
pub const LIST_TIMEOUT_MS: u64 = 30_000;

pub fn mission_timeout_ms(cloud: bool) -> u64 {
    if cloud { 90_000 } else { 10_000 }
}

pub fn formation_timeout_ms(cloud: bool) -> u64 {
    if cloud { 90_000 } else { 15_000 }
}

/// The bridge base URL without trailing slashes.
pub fn normalize_base(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
}

/// A bridge error as the toasts show it. The legacy client printed `String(new Error(message))`,
/// so every toast started with `Error: `; the toast's red dot already says that.
pub fn js_error(message: &str) -> String {
    message.to_string()
}

/// "Bridge unreachable" as the toasts show it. The legacy text told the operator to run
/// `npm run mission-bridge` and appended the raw transport error (`TypeError: ... (os error
/// 10061)`); the bridge is the Rust `skylens-drone` now, and the transport detail goes to the log
/// (see the caller), not the operator.
pub fn unreachable_error(endpoint: &str) -> String {
    js_error(&format!("Mission Bridge에 연결할 수 없습니다 ({endpoint}). 브리지(skylens-drone)가 실행 중인지 확인하세요."))
}

/// `errorMessage(response)`: the JSON `error` string, else `Mission Bridge HTTP <status>`.
pub fn error_message(status: u16, body: &[u8]) -> String {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string))
        .unwrap_or_else(|| format!("Mission Bridge HTTP {status}"))
}

/// `GET /health` -> whether the cloud toggle may be offered. Any failure means "not available".
pub fn cloud_configured(body: Option<&Value>) -> bool {
    body.and_then(|v| v.get("cloudConfigured")).and_then(Value::as_bool).unwrap_or(false)
}

fn waypoints_json(waypoints: &[model::Gps]) -> Value {
    Value::Array(waypoints.iter().map(|w| json!({ "lat": w.lat, "lon": w.lon, "alt": w.alt })).collect())
}

fn flight_json(agl_m: f64) -> Value {
    json!({
        "aglM": agl_m,
        "speedMps": 2,
        "gimbalPitchDeg": -30,
        "rthHeightM": agl_m.max(30.0),
    })
}

/// `POST /missions` body (one aircraft).
pub fn mission_body(drone_id: u32, waypoints: &[model::Gps], looped: bool, agl_m: f64, cloud: bool) -> String {
    let mut body = json!({
        "name": format!("skylens-drone-{drone_id}"),
        "droneId": drone_id,
        "waypoints": waypoints_json(waypoints),
        "loop": looped,
        "flight": flight_json(agl_m),
    });
    if cloud {
        body["deliver"] = json!("cloud");
    }
    body.to_string()
}

/// `POST /formation` body (three aircraft around the planned centroid track).
pub fn formation_body(waypoints: &[model::Gps], looped: bool, agl_m: f64, spacing_m: f64, cloud: bool) -> String {
    let mut body = json!({
        "name": "skylens-formation",
        "waypoints": waypoints_json(waypoints),
        "loop": looped,
        "flight": flight_json(agl_m),
        "formation": { "spacingM": spacing_m },
    });
    if cloud {
        body["deliver"] = json!("cloud");
    }
    body.to_string()
}

/// A built KMZ to fetch.
#[derive(Debug, Clone, PartialEq)]
pub struct Download {
    pub url: String,
    pub file_name: String,
}

/// `POST /missions` answer, as far as the tower uses it.
#[derive(Debug, Clone, PartialEq)]
pub struct MissionResult {
    pub download: Download,
    pub waypoint_count: u64,
    pub installed: bool,
    pub warnings: Vec<String>,
    pub cloud: bool,
}

fn warnings(v: &Value) -> Vec<String> {
    v.get("warnings")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|w| w.as_str().map(str::to_string)).collect())
        .unwrap_or_default()
}

/// Parses `POST /missions`; the error is the stringified legacy error.
pub fn parse_mission(v: &Value) -> Result<MissionResult, String> {
    let file_name = v.get("fileName").and_then(Value::as_str).unwrap_or("");
    let url = v.get("downloadUrl").and_then(Value::as_str).unwrap_or("");
    if file_name.is_empty() || url.is_empty() {
        return Err(js_error("Mission Bridge 응답에 KMZ 정보가 없습니다"));
    }
    Ok(MissionResult {
        download: Download { url: url.to_string(), file_name: file_name.to_string() },
        waypoint_count: v.get("waypointCount").and_then(Value::as_u64).unwrap_or(0),
        installed: v.get("installed").and_then(Value::as_bool).unwrap_or(false),
        warnings: warnings(v),
        cloud: v.get("cloud").is_some_and(|c| !c.is_null()),
    })
}

/// One formation member.
#[derive(Debug, Clone, PartialEq)]
pub struct Member {
    pub station: String,
    pub download: Download,
    pub warnings: Vec<String>,
    pub cloud: bool,
}

/// Parses `POST /formation`.
pub fn parse_formation(v: &Value) -> Result<Vec<Member>, String> {
    let members: Vec<Member> = v
        .get("members")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|m| Member {
                    station: m.get("station").and_then(Value::as_str).unwrap_or("").to_string(),
                    download: Download {
                        url: m.get("downloadUrl").and_then(Value::as_str).unwrap_or("").to_string(),
                        file_name: m.get("fileName").and_then(Value::as_str).unwrap_or("").to_string(),
                    },
                    warnings: warnings(m),
                    cloud: m.get("cloud").is_some_and(|c| !c.is_null()),
                })
                .collect()
        })
        .unwrap_or_default();
    if members.is_empty() {
        return Err(js_error("Mission Bridge 응답에 편대 미션이 없습니다"));
    }
    Ok(members)
}

/// The formation toast (legacy `control.ts`).
pub fn formation_toast(members: &[Member]) -> String {
    let warned: usize = members.iter().map(|m| m.warnings.len()).sum();
    let list = members.iter().map(|m| crate::labels::formation_station_name(&m.station)).collect::<Vec<_>>().join(" · ");
    let uploaded = members.iter().filter(|m| m.cloud).count();
    let cloud_note = if uploaded > 0 { format!(" · 클라우드 {uploaded}기 업로드") } else { String::new() };
    let mut s = format!("편대 KMZ {}개 다운로드됨 · {list}{cloud_note}", members.len());
    if warned > 0 {
        s.push_str(&format!(" · 확인 {warned}건"));
    }
    s
}

/// The single-mission toast (legacy `control.ts`).
pub fn mission_toast(m: &MissionResult) -> String {
    let delivered = if m.installed { "DJI Fly 임무에 설치됨" } else { "KMZ 다운로드됨" };
    let cloud_note = if m.cloud { " · 클라우드 업로드됨(원격 RC)" } else { "" };
    let warning = if m.warnings.is_empty() { String::new() } else { format!(" · 확인 {}건", m.warnings.len()) };
    format!("{delivered}{cloud_note} · {} WP{warning}", m.waypoint_count)
}

/// A mission in the DJI Fly account cloud.
#[derive(Debug, Clone, PartialEq)]
pub struct CloudMission {
    pub uuid: String,
    pub name: String,
    pub waypoint_count: Option<u64>,
    pub distance_m: Option<f64>,
}

impl CloudMission {
    /// The row label (legacy `renderMissions`).
    pub fn label(&self) -> String {
        let points = self.waypoint_count.map_or_else(|| "?".to_string(), |n| n.to_string());
        let dist = self.distance_m.map_or_else(String::new, |d| format!(" · {}m", crate::labels::js_round(d) as i64));
        let name = if self.name.is_empty() { "(이름 없음)" } else { self.name.as_str() };
        format!("{name} · {points} WP{dist}")
    }
}

/// Parses `GET /cloud/missions`.
pub fn parse_cloud_missions(v: &Value) -> Vec<CloudMission> {
    v.get("missions")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|m| CloudMission {
                    uuid: m.get("uuid").and_then(Value::as_str).unwrap_or("").to_string(),
                    name: m.get("name").and_then(Value::as_str).unwrap_or("").to_string(),
                    waypoint_count: m.get("waypointCount").and_then(Value::as_u64),
                    distance_m: m.get("distanceM").and_then(Value::as_f64),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// `POST /cloud/delete` body.
pub fn delete_body(uuid: &str) -> String {
    json!({ "missionUuid": uuid }).to_string()
}

/// Parses `POST /cloud/delete`: whether the mission was removed.
pub fn parse_removed(v: &Value) -> bool {
    v.get("removed").and_then(Value::as_bool).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wp(lat: f64, lon: f64, alt: f64) -> model::Gps {
        model::Gps { lat, lon, alt }
    }

    #[test]
    fn bodies_match_the_legacy_requests() {
        let w = [wp(36.3685, 127.3475, 82.0), wp(36.3687, 127.3477, 84.0)];
        let m: Value = serde_json::from_str(&mission_body(1, &w, false, 20.0, true)).unwrap();
        assert_eq!(m["name"], "skylens-drone-1");
        assert_eq!(m["droneId"], 1);
        assert_eq!(m["deliver"], "cloud");
        assert_eq!(m["flight"]["rthHeightM"], 30.0);
        assert_eq!(m["flight"]["speedMps"], 2);
        assert_eq!(m["waypoints"][1]["alt"], 84.0);
        let f: Value = serde_json::from_str(&formation_body(&w, true, 60.0, 10.0, false)).unwrap();
        assert_eq!(f["name"], "skylens-formation");
        assert_eq!(f["formation"]["spacingM"], 10.0);
        assert_eq!(f["flight"]["rthHeightM"], 60.0);
        assert!(f.get("deliver").is_none());
    }

    #[test]
    fn errors_are_worded_for_the_operator() {
        assert_eq!(error_message(400, br#"{"error":"bad route"}"#), "bad route");
        assert_eq!(error_message(502, b"oops"), "Mission Bridge HTTP 502");
        assert_eq!(
            unreachable_error("http://127.0.0.1:8091"),
            "Mission Bridge에 연결할 수 없습니다 (http://127.0.0.1:8091). 브리지(skylens-drone)가 실행 중인지 확인하세요."
        );
    }

    #[test]
    fn mission_and_formation_toasts() {
        let v = json!({"fileName":"a.kmz","downloadUrl":"http://x/missions/a.kmz","waypointCount":3,"installed":false,"warnings":["w"]});
        let m = parse_mission(&v).unwrap();
        assert_eq!(mission_toast(&m), "KMZ 다운로드됨 · 3 WP · 확인 1건");
        assert!(parse_mission(&json!({})).is_err());
        let f = json!({"members":[
            {"station":"left","fileName":"l.kmz","downloadUrl":"u1","warnings":[],"cloud":{"missionUuid":"x","verified":true}},
            {"station":"right","fileName":"r.kmz","downloadUrl":"u2","warnings":["a","b"]},
            {"station":"center","fileName":"c.kmz","downloadUrl":"u3","warnings":[]}
        ]});
        let members = parse_formation(&f).unwrap();
        assert_eq!(formation_toast(&members), "편대 KMZ 3개 다운로드됨 · 앞·왼쪽 · 앞·오른쪽 · 뒤·중앙 · 클라우드 1기 업로드 · 확인 2건");
        assert_eq!(parse_formation(&json!({"members":[]})).unwrap_err(), "Mission Bridge 응답에 편대 미션이 없습니다");
    }

    #[test]
    fn cloud_rows() {
        let v = json!({"missions":[{"uuid":"u","name":"","waypointCount":null,"distanceM":1234.5},{"uuid":"v","name":"route","waypointCount":4}]});
        let list = parse_cloud_missions(&v);
        assert_eq!(list[0].label(), "(이름 없음) · ? WP · 1235m");
        assert_eq!(list[1].label(), "route · 4 WP");
        assert!(parse_removed(&json!({"removed":true})));
        assert!(!cloud_configured(None));
        assert!(cloud_configured(Some(&json!({"cloudConfigured":true}))));
    }
}
