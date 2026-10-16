//! Small key/value persistence: `localStorage` on the web (the legacy keys), a JSON file on
//! native (`<config dir>/skylens/control.json`, one entry per legacy key).
//!
//! Failures are swallowed like the legacy code did: a blocked or corrupt store must never stop
//! the tower from starting, it only forgets.

/// Display option (legacy `settings.ts`).
pub const SETTINGS_KEY: &str = "skylens.control.settings.v1";
/// Where the operator left the planner map (legacy `routeModal.ts`).
pub const MAP_CENTER_KEY: &str = "skylens.control.mapCenter.v1";

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use std::path::PathBuf;

    fn file() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("SKYLENS_CONTROL_SETTINGS") {
            if !p.trim().is_empty() {
                return Some(PathBuf::from(p));
            }
        }
        let base = if cfg!(windows) {
            std::env::var_os("APPDATA").map(PathBuf::from)
        } else if let Some(x) = std::env::var_os("XDG_CONFIG_HOME") {
            Some(PathBuf::from(x))
        } else {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config"))
        }?;
        Some(base.join("skylens").join("control.json"))
    }

    fn read_all() -> serde_json::Map<String, serde_json::Value> {
        let Some(path) = file() else { return Default::default() };
        std::fs::read(&path)
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default()
    }

    pub fn get(key: &str) -> Option<String> {
        read_all().get(key).and_then(|v| v.as_str()).map(str::to_string)
    }

    pub fn set(key: &str, value: &str) {
        let Some(path) = file() else { return };
        let mut all = read_all();
        all.insert(key.to_string(), serde_json::Value::String(value.to_string()));
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(text) = serde_json::to_string_pretty(&serde_json::Value::Object(all)) {
            let _ = std::fs::write(&path, text);
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    fn storage() -> Option<web_sys::Storage> {
        web_sys::window()?.local_storage().ok().flatten()
    }

    pub fn get(key: &str) -> Option<String> {
        storage()?.get_item(key).ok().flatten()
    }

    pub fn set(key: &str, value: &str) {
        if let Some(s) = storage() {
            let _ = s.set_item(key, value);
        }
    }
}

/// The stored string for a key, if any.
pub fn get(key: &str) -> Option<String> {
    imp::get(key)
}

/// Stores a string under a key (best effort).
pub fn set(key: &str, value: &str) {
    imp::set(key, value)
}
