//! Control tower screen (관제탑). Served by the Go core as a web build and runnable as a native
//! window. The legacy TypeScript tower ported to Rust: the same HTML template and CSS on Blitz,
//! the 3D view on pointblitz.
//!
//! - [`app`]: the screen (core link, panels, planner, bridge side effects, keys).
//! - [`tower`]: the 3D view (terrain drape, buildings in three display modes, drones, route,
//!   overview and chase cameras).
//! - [`terrain`], [`buildings`]: terrain tiles and building footprints into the legacy looks.
//! - [`planner`], [`slider`], [`modals`]: the route planner map, its altitude slider, the dialogs.
//! - [`panels`], [`video`], [`whep`]: mission, fleet and settings panels; the camera panel and its
//!   live feed (ffmpeg on native, the browser's WebRTC on the web).
//! - [`fleet`], [`manual`]: telemetry tracks and selection; keyboard manual control.
//! - [`bridge`], [`http`], [`download`], [`storage`], [`settings`], [`labels`], [`glyphs`]:
//!   mission bridge client, background HTTP, KMZ saving, persistence, wording.

pub mod app;
pub mod bridge;
pub mod buildings;
pub mod download;
pub mod fleet;
pub mod glyphs;
pub mod http;
pub mod labels;
pub mod manual;
pub mod modals;
pub mod panels;
pub mod planner;
pub mod settings;
pub mod slider;
pub mod storage;
pub mod terrain;
pub mod tower;
pub mod video;
pub mod whep;

#[cfg(not(target_arch = "wasm32"))]
pub mod native;
#[cfg(target_arch = "wasm32")]
pub mod web;

pub use app::{ControlApp, Options};
