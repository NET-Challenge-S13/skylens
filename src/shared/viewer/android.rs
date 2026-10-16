//! Android plumbing for the screens: settings without a command line, and logcat.
//!
//! A screen's `android_main` hands the activity to [`crate::shell::run_android`]; the settings it
//! needs (e.g. the core address) are read here, so no rebuild is needed to point an installed APK
//! at another core:
//!
//! 1. system property `debug.skylens.<key>` (`adb shell setprop debug.skylens.core ws://…`;
//!    `debug.*` is writable from adb without root, cleared on reboot)
//! 2. `<key>.txt` in the app's external files directory
//!    (`/sdcard/Android/data/<package>/files/`, `adb push` works on release builds)
//! 3. `<key>.txt` in the app's internal files directory (`adb shell run-as`, debug builds only)
//!
//! The file format is [`setting_value`]: the first line that is neither blank nor a `#` comment.

/// The value in a settings file: the first line that is neither blank nor a `#` comment, trimmed.
pub fn setting_value(text: &str) -> Option<String> {
    text.lines().map(str::trim).find(|l| !l.is_empty() && !l.starts_with('#')).map(str::to_string)
}

/// The system property that overrides a setting.
pub fn property_name(key: &str) -> String {
    format!("debug.skylens.{key}")
}

/// The file in a files directory that holds a setting.
pub fn file_name(key: &str) -> String {
    format!("{key}.txt")
}

#[cfg(target_os = "android")]
pub use self::platform::*;

#[cfg(target_os = "android")]
mod platform {
    use super::{file_name, property_name, setting_value};
    pub use android_activity::AndroidApp;
    use std::ffi::{CStr, CString, c_char, c_int};

    const TAG: &CStr = c"skylens";
    const ANDROID_LOG_INFO: c_int = 4;
    const ANDROID_LOG_ERROR: c_int = 6;

    #[link(name = "log")]
    unsafe extern "C" {
        fn __android_log_write(prio: c_int, tag: *const c_char, text: *const c_char) -> c_int;
    }

    fn write(prio: c_int, msg: &str) {
        // Interior NULs would make CString fail; there are none after this.
        let Ok(text) = CString::new(msg.replace('\0', " ")) else { return };
        // SAFETY: both pointers are valid NUL-terminated strings for the duration of the call.
        unsafe {
            __android_log_write(prio, TAG.as_ptr(), text.as_ptr());
        }
    }

    /// A line in logcat under the tag `skylens` (`adb logcat -s skylens`).
    pub fn log(msg: &str) {
        write(ANDROID_LOG_INFO, msg);
    }

    /// An error line in logcat under the tag `skylens`.
    pub fn log_error(msg: &str) {
        write(ANDROID_LOG_ERROR, msg);
    }

    /// Sends panics to logcat; stderr goes nowhere on Android.
    pub fn install_panic_hook() {
        std::panic::set_hook(Box::new(|info| write(ANDROID_LOG_ERROR, &format!("panic: {info}"))));
    }

    /// A system property, if set and not blank.
    pub fn system_property(name: &str) -> Option<String> {
        let name = CString::new(name).ok()?;
        let mut buf = [0 as c_char; libc::PROP_VALUE_MAX as usize];
        // SAFETY: `name` is NUL-terminated and `buf` has the PROP_VALUE_MAX bytes the call may write.
        let len = unsafe { libc::__system_property_get(name.as_ptr(), buf.as_mut_ptr()) };
        if len <= 0 {
            return None;
        }
        // SAFETY: the call wrote a NUL-terminated string into `buf`.
        let value = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().trim().to_string();
        (!value.is_empty()).then_some(value)
    }

    /// A setting by key (`core`, `points`, …), from the system property, then the external files
    /// directory, then the internal one (see the module docs).
    pub fn setting(app: &AndroidApp, key: &str) -> Option<String> {
        if let Some(v) = system_property(&property_name(key)) {
            return Some(v);
        }
        [app.external_data_path(), app.internal_data_path()]
            .into_iter()
            .flatten()
            .find_map(|dir| std::fs::read_to_string(dir.join(file_name(key))).ok().as_deref().and_then(setting_value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_settings_file_gives_its_first_real_line() {
        assert_eq!(setting_value("ws://10.0.0.2:8080/viewer\n"), Some("ws://10.0.0.2:8080/viewer".into()));
        assert_eq!(setting_value("# core\n\n  gateway.local:8080  \nignored"), Some("gateway.local:8080".into()));
        assert_eq!(setting_value("\r\n# only a comment\r\n"), None);
        assert_eq!(setting_value(""), None);
    }

    #[test]
    fn keys_map_to_a_property_and_a_file() {
        assert_eq!(property_name("core"), "debug.skylens.core");
        assert_eq!(file_name("points"), "points.txt");
    }
}
