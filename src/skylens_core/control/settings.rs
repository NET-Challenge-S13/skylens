//! Operator display option (legacy `settings.ts`): how the buildings of the one scene are drawn.
//! Persisted as `{"display":"black"}` under the legacy key; `?display=` (web) or `--display`
//! (native) wins for one run and is never written back.

use crate::storage;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DisplayMode {
    /// 점: buildings as a sampled point cloud.
    Points,
    /// 검정 텍스처 건물: lit prisms with cyan linework (default).
    Black,
    /// 실사 항공뷰: the satellite image over terrain and buildings.
    Aerial,
}

impl DisplayMode {
    pub const ALL: [DisplayMode; 3] = [DisplayMode::Points, DisplayMode::Black, DisplayMode::Aerial];

    /// The `data-mode` value and stored spelling.
    pub fn id(self) -> &'static str {
        match self {
            DisplayMode::Points => "points",
            DisplayMode::Black => "black",
            DisplayMode::Aerial => "aerial",
        }
    }

    pub fn parse(s: &str) -> Option<DisplayMode> {
        match s.trim() {
            "points" => Some(DisplayMode::Points),
            "black" => Some(DisplayMode::Black),
            "aerial" => Some(DisplayMode::Aerial),
            _ => None,
        }
    }
}

pub const DEFAULT_DISPLAY: DisplayMode = DisplayMode::Black;

/// Reads the stored value; anything unreadable is the default.
pub fn parse_stored(raw: Option<&str>) -> DisplayMode {
    raw.and_then(|r| serde_json::from_str::<serde_json::Value>(r).ok())
        .and_then(|v| v.get("display").and_then(|d| d.as_str()).and_then(DisplayMode::parse))
        .unwrap_or(DEFAULT_DISPLAY)
}

/// The stored form of a choice.
pub fn stored_form(mode: DisplayMode) -> String {
    serde_json::json!({ "display": mode.id() }).to_string()
}

/// The current choice and whether it came from the URL / command line.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    pub display: DisplayMode,
    pub overridden: bool,
}

impl Settings {
    /// `override_mode` is the `?display=` / `--display` value, if valid.
    pub fn load(override_mode: Option<DisplayMode>) -> Settings {
        match override_mode {
            Some(display) => Settings { display, overridden: true },
            None => Settings { display: parse_stored(storage::get(storage::SETTINGS_KEY).as_deref()), overridden: false },
        }
    }

    /// Changes the choice and persists it. Returns false when it was already the choice.
    pub fn set_display(&mut self, mode: DisplayMode) -> bool {
        if self.display == mode {
            return false;
        }
        self.display = mode;
        storage::set(storage::SETTINGS_KEY, &stored_form(mode));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_values_round_trip_and_fall_back() {
        for m in DisplayMode::ALL {
            assert_eq!(parse_stored(Some(&stored_form(m))), m);
            assert_eq!(DisplayMode::parse(m.id()), Some(m));
        }
        assert_eq!(parse_stored(None), DisplayMode::Black);
        assert_eq!(parse_stored(Some("{not json")), DisplayMode::Black);
        assert_eq!(parse_stored(Some("{\"display\":\"neon\"}")), DisplayMode::Black);
        assert_eq!(stored_form(DisplayMode::Points), "{\"display\":\"points\"}");
    }

    #[test]
    fn override_is_marked() {
        let s = Settings::load(Some(DisplayMode::Aerial));
        assert!(s.overridden);
        assert_eq!(s.display, DisplayMode::Aerial);
    }
}
