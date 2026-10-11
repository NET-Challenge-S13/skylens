//! UI logic shared by the web (wasm) and native (blitz) clients.
//!
//! Both screens (관제탑 control tower, 현황판 status board) are one HTML/CSS file
//! each under `ui/`, parsed and drawn by Blitz on web and native alike. The 3D
//! view is a Blitz custom widget that renders pointblitz chunks and the mesh pass
//! into a wgpu texture. Everything above the entry points is one code base:
//!
//! - [`protocol`]: `ws /viewer` envelopes and chunk payload decoding
//! - [`state`]: the reducer from viewer events to screen state
//! - [`chunks`]: chunk replacement rules (preview, refined, generation, mesh version)
//! - [`views`]: state to text/HTML fragments
//! - [`app`]: the Blitz document, DOM binding, clicks and keys
//! - [`view3d`]: the 3D widget (pointblitz + [`mesh::MeshPass`] + markers)
//! - [`link`]: WebSocket with reconnect (tokio-tungstenite native, browser WebSocket on web)

pub mod app;
pub mod camera;
pub mod chunks;
pub mod display;
pub mod link;
pub mod mesh;
pub mod protocol;
pub mod state;
pub mod view3d;
pub mod views;

#[cfg(not(target_arch = "wasm32"))]
pub mod native;
#[cfg(target_arch = "wasm32")]
pub mod web;

pub use app::{Config, Screen, SkylensDocument};

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
#[cfg(not(target_arch = "wasm32"))]
pub fn log(msg: &str) {
    eprintln!("{msg}");
}

/// Diagnostics: stderr on native, the browser console on web.
#[cfg(target_arch = "wasm32")]
pub fn log(msg: &str) {
    web_sys::console::log_1(&msg.into());
}
