//! Screen layer shared by the control tower (skylens_core/control) and the status board
//! (skylens_client): legacy shared/viewer CSS and widgets, core link, chunk store, pointblitz 3D view.
//!
//! - [`shell`]: a Blitz document from an HTML template plus the legacy CSS and the bundled fonts,
//!   native window, Android and wasm entry points, DOM event dispatch to a [`shell::ScreenApp`].
//! - [`android`]: settings without a command line (system properties, files) and logcat.
//! - [`dom`]: small DOM helpers (text, classes, attributes, inline styles, child markup).
//! - [`widgets`]: loading screen, toast, pane label, net badge, detection card, with the exact DOM
//!   the legacy TypeScript produced.
//! - [`link`]: the one WebSocket to the core, binary `viewer.v1.Envelope` frames, reconnect.
//! - [`protocol`]: envelope encoding, control messages, payload decoding.
//! - [`site`]: the operating area from the core; `Site.anchor` is the ENU origin.
//! - [`chunks`]: chunk replacement rules and per-kind access.
//! - [`camera`]: camera controllers ported from the legacy three.js viewers, and screen projection.
//! - [`view3d`]: the Blitz custom widget that owns the GPU and draws a frame with pointblitz.
//! - [`config`]: legacy tuning constants and colours.

pub mod android;
pub mod camera;
pub mod chunks;
pub mod config;
pub mod dom;
pub mod link;
pub mod protocol;
pub mod shell;
pub mod site;
pub mod view3d;
pub mod widgets;

pub use blitz_dom::NodeId;
pub use keyboard_types::Code;
// The versions this crate builds against, so the screens need not pin them again.
pub use {anyrender, glam, peniko, pointblitz, skylens_shared, wgpu};

pub use camera::{Pose, Projector};
pub use chunks::{Applied, Change, ChunkStore, SlotPoints, ZoneSlots};
pub use dom::{Dom, Sel, q};
pub use link::{CoreLink, LinkEvent, LinkStatus};
pub use shell::{Page, PollCx, ScreenApp, ScreenDocument, UiInput, WindowOptions};
pub use site::SiteFrame;
pub use view3d::{FrameInfo, FrameSpec, Gpu3d, SlotSync, View3d, ViewInput, ViewScreen};

/// Wall-clock milliseconds since the Unix epoch.
#[cfg(not(target_arch = "wasm32"))]
pub fn now_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}

/// Wall-clock milliseconds since the Unix epoch.
#[cfg(target_arch = "wasm32")]
pub fn now_ms() -> f64 {
    js_sys::Date::now()
}

/// Diagnostics: stderr on native, the browser console on web.
#[cfg(not(any(target_arch = "wasm32", target_os = "android")))]
pub fn log(msg: &str) {
    eprintln!("{msg}");
}

/// Diagnostics: logcat on Android (tag `skylens`), where stderr goes nowhere.
#[cfg(target_os = "android")]
pub fn log(msg: &str) {
    android::log(msg);
}

/// Diagnostics: stderr on native, the browser console on web.
#[cfg(target_arch = "wasm32")]
pub fn log(msg: &str) {
    web_sys::console::log_1(&msg.into());
}
