//! Words and number formats on the tower screen, exactly as the legacy TypeScript wrote them
//! (`stationLabel.ts`, `telemetryPanel.ts`, `videoPanel.ts`, `missionBridge.ts`), plus the
//! JavaScript number formatting they relied on (`toFixed`, `Math.round`, `toLocaleString`).

use skylens_shared::proto::model::v1 as model;

/// Formation station of a drone. The operator names aircraft by where they fly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Station {
    Left,
    Center,
    Right,
}

impl Station {
    /// The wire value; unspecified is None (the legacy fell back to centre where it mattered).
    pub fn from_proto(v: i32) -> Option<Station> {
        match model::DroneStation::try_from(v).ok()? {
            model::DroneStation::Left => Some(Station::Left),
            model::DroneStation::Center => Some(Station::Center),
            model::DroneStation::Right => Some(Station::Right),
            model::DroneStation::Unspecified => None,
        }
    }

    /// Bridge JSON spelling (`left` / `center` / `right`).
    pub fn parse(s: &str) -> Option<Station> {
        match s {
            "left" => Some(Station::Left),
            "center" => Some(Station::Center),
            "right" => Some(Station::Right),
            _ => None,
        }
    }

    /// Fleet list and readout name (legacy `stationLabel`).
    pub fn label(self) -> &'static str {
        match self {
            Station::Left => "좌측 드론",
            Station::Center => "중앙 드론",
            Station::Right => "우측 드론",
        }
    }

    /// Camera panel header (legacy `stationCamLabel`).
    pub fn cam_label(self) -> &'static str {
        match self {
            Station::Left => "LEFT CAM",
            Station::Center => "CENTER CAM",
            Station::Right => "RIGHT CAM",
        }
    }

    /// Screen order of the formation, left to right (legacy `STATION_RANK`).
    pub fn rank(self) -> u8 {
        match self {
            Station::Left => 0,
            Station::Center => 1,
            Station::Right => 2,
        }
    }
}

/// Formation member name in the KMZ toast (legacy `stationName`, `missionBridge.ts`); unknown
/// stations are shown as sent.
pub fn formation_station_name(station: &str) -> String {
    match station {
        "left" => "앞·왼쪽".to_string(),
        "right" => "앞·오른쪽".to_string(),
        "center" => "뒤·중앙".to_string(),
        other => other.to_string(),
    }
}

/// JavaScript `Number.prototype.toFixed(digits)`: the decimal closest to the value, and on an
/// exact tie the larger magnitude (Rust's formatter would pick the even digit).
pub fn to_fixed(v: f64, digits: usize) -> String {
    if !v.is_finite() {
        return if v.is_nan() { "NaN".into() } else if v > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let p = 10f64.powi(digits as i32);
    let scaled = v.abs() * p;
    if scaled < 9.0e15 && scaled.fract() == 0.5 {
        let rounded = (scaled.trunc() + 1.0) / p;
        let s = format!("{rounded:.digits$}");
        return if v < 0.0 { format!("-{s}") } else { s };
    }
    format!("{v:.digits$}")
}

/// JavaScript `Math.round`: halves go toward +infinity.
pub fn js_round(v: f64) -> f64 {
    (v + 0.5).floor()
}

/// `n.toLocaleString('ko-KR')` for whole numbers: thousands separated by commas.
pub fn grouped(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `.telemetry-row__vals` (legacy `telemetryPanel.ts`).
pub fn telemetry_vals(lat: f64, lon: f64, alt: f64, heading_deg: f64, speed: f64, battery_pct: f64) -> String {
    format!(
        "{}, {} · {}m · {}° · {}m/s · {}%",
        to_fixed(lat, 5),
        to_fixed(lon, 5),
        to_fixed(alt, 0),
        to_fixed(heading_deg, 0),
        to_fixed(speed, 1),
        to_fixed(battery_pct, 0)
    )
}

/// The position part of the camera overlay (legacy `updateOverlay`).
pub fn overlay_where(lat: f64, lon: f64, alt: f64) -> String {
    let ns = if lat >= 0.0 { 'N' } else { 'S' };
    let ew = if lon >= 0.0 { 'E' } else { 'W' };
    format!("{}°{ns} {}°{ew} · {}m", to_fixed(lat.abs(), 5), to_fixed(lon.abs(), 5), to_fixed(alt, 0))
}

/// The camera overlay line (legacy `updateOverlay`), playback errors aside.
pub fn overlay_text(who: &str, fix: Option<(f64, f64, f64)>, has_feed: bool) -> String {
    match (fix, has_feed) {
        (None, false) => format!("{who} · NO SIGNAL"),
        (None, true) => format!("{who} · 수신 대기"),
        (Some((lat, lon, alt)), false) => format!("{who} · {} · 영상 대기", overlay_where(lat, lon, alt)),
        (Some((lat, lon, alt)), true) => format!("{who} · {}", overlay_where(lat, lon, alt)),
    }
}

/// A route planner list row label (legacy `renderList`). Two spaces after the number.
pub fn waypoint_label(index: usize, lat: f64, lon: f64, alt: f64) -> String {
    format!("#{}  {}, {} · {}m", index + 1, to_fixed(lat, 5), to_fixed(lon, 5), to_fixed(alt, 0))
}

/// Span button label (legacy: `${s/1000}km` from 1000 m up, else `${s}m`).
pub fn span_label(span_m: u32) -> String {
    if span_m >= 1000 {
        format!("{}km", js_number(f64::from(span_m) / 1000.0))
    } else {
        format!("{span_m}m")
    }
}

/// JavaScript's default number to string for the short values used here (no exponent forms).
pub fn js_number(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        let s = format!("{v}");
        s
    }
}

/// The planner scale bar text: a quarter of the span (legacy `drawScaleAndCompass`).
pub fn scale_label(span_m: f64) -> String {
    let meters = span_m / 4.0;
    if meters >= 1000.0 { format!("{}km", js_number(meters / 1000.0)) } else { format!("{}m", js_number(js_round(meters))) }
}

/// The route-sent toast (legacy `control.ts` onAssign).
pub fn route_sent_toast(sent: bool, waypoints: usize, looped: bool) -> String {
    if sent {
        format!("경로 전송됨 · 웨이포인트 {waypoints}개{}", if looped { " · 왕복" } else { "" })
    } else {
        "코어에 연결되어 있지 않아 경로를 전송하지 못했습니다".to_string()
    }
}

/// The settings panel area label: `${name} (${south.toFixed(3)}, ${west.toFixed(3)})`.
pub fn area_label(name: &str, min_lat: f64, min_lon: f64) -> String {
    format!("{name} ({}, {})", to_fixed(min_lat, 3), to_fixed(min_lon, 3))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_fixed_rounds_ties_away_like_javascript() {
        assert_eq!(to_fixed(0.5, 0), "1");
        assert_eq!(to_fixed(2.5, 0), "3");
        assert_eq!(to_fixed(-2.5, 0), "-3");
        assert_eq!(to_fixed(149.4, 0), "149");
        assert_eq!(to_fixed(36.368721, 5), "36.36872");
        assert_eq!(to_fixed(12.0, 1), "12.0");
        assert_eq!(to_fixed(36.355, 3), "36.355");
    }

    #[test]
    fn js_round_goes_up_on_halves() {
        assert_eq!(js_round(7.5), 8.0);
        assert_eq!(js_round(-7.5), -7.0);
        assert_eq!(js_round(7.49), 7.0);
    }

    #[test]
    fn grouped_matches_ko_kr_locale() {
        assert_eq!(grouped(6194), "6,194");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(1_234_567), "1,234,567");
        assert_eq!(grouped(0), "0");
    }

    #[test]
    fn telemetry_row_matches_the_capture() {
        assert_eq!(
            telemetry_vals(36.368721, 127.348243, 149.2, 135.0, 12.0, 98.0),
            "36.36872, 127.34824 · 149m · 135° · 12.0m/s · 98%"
        );
    }

    #[test]
    fn overlay_variants() {
        assert_eq!(overlay_text("중앙 드론", None, false), "중앙 드론 · NO SIGNAL");
        assert_eq!(overlay_text("중앙 드론", None, true), "중앙 드론 · 수신 대기");
        assert_eq!(
            overlay_text("중앙 드론", Some((36.36699, 127.3453, 105.0)), false),
            "중앙 드론 · 36.36699°N 127.34530°E · 105m · 영상 대기"
        );
        assert_eq!(
            overlay_text("중앙 드론", Some((36.36872, 127.34824, 149.0)), true),
            "중앙 드론 · 36.36872°N 127.34824°E · 149m"
        );
        assert_eq!(overlay_where(-1.0, -2.0, 3.0), "1.00000°S 2.00000°W · 3m");
    }

    #[test]
    fn planner_labels() {
        assert_eq!(waypoint_label(0, 36.36796, 127.34649, 120.0), "#1  36.36796, 127.34649 · 120m");
        assert_eq!(span_label(300), "300m");
        assert_eq!(span_label(1200), "1.2km");
        assert_eq!(span_label(2500), "2.5km");
        assert_eq!(scale_label(600.0), "150m");
        assert_eq!(scale_label(300.0), "75m");
        assert_eq!(scale_label(1200.0), "300m");
        assert_eq!(scale_label(2500.0), "625m");
    }

    #[test]
    fn toasts_and_area() {
        assert_eq!(route_sent_toast(true, 3, true), "경로 전송됨 · 웨이포인트 3개 · 왕복");
        assert_eq!(route_sent_toast(true, 2, false), "경로 전송됨 · 웨이포인트 2개");
        assert_eq!(route_sent_toast(false, 2, true), "코어에 연결되어 있지 않아 경로를 전송하지 못했습니다");
        assert_eq!(area_label("daejeon", 36.355, 127.33), "daejeon (36.355, 127.330)");
    }

    #[test]
    fn stations() {
        assert_eq!(Station::from_proto(2), Some(Station::Center));
        assert_eq!(Station::from_proto(0), None);
        assert_eq!(Station::Left.label(), "좌측 드론");
        assert_eq!(Station::Right.cam_label(), "RIGHT CAM");
        assert_eq!(formation_station_name("left"), "앞·왼쪽");
        assert_eq!(formation_station_name("center"), "뒤·중앙");
        assert_eq!(formation_station_name("odd"), "odd");
    }
}
