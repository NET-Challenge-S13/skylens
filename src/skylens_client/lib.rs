//! Status board (현황판) for field commanders outside KOREN: the legacy TypeScript board's HTML and
//! CSS, unchanged, drawn by Blitz, with the reconstruction view on pointblitz. Web (wasm) and
//! native share all of it; only the entry points differ.
//!
//! The board connects read-only to the core's `ws /viewer` (through the gateway), takes the
//! `Site` anchor as its ENU origin, and shows only what has been reconstructed and has arrived.
//!
//! - [`app`]: the Blitz screen (link, DOM writes as in `ui/DOM.md`)
//! - [`board`]: what the board knows, shared by the DOM side and the two 3D views
//! - [`scene`] / [`inset`]: the main reconstruction view and the point panel's view
//! - [`follow`], [`reveal`], [`markers`], [`points`]: camera rules, arrival fade, detection
//!   markers, point chunk reading (pure, tested)
//! - [`minimap`], [`card`], [`panels`]: the minimap widget, the detection card, panel text

pub mod app;
pub mod board;
pub mod card;
pub mod follow;
pub mod inset;
pub mod markers;
pub mod minimap;
pub mod panels;
pub mod points;
pub mod reveal;
pub mod scene;

// The versions skylens-viewer builds against.
pub(crate) use skylens_viewer::{anyrender, glam, peniko, pointblitz};

pub use app::{Options, StatusApp, parse_args};

/// Opens the board in a native window and runs until it closes.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_native(opts: &Options) -> anyhow::Result<()> {
    skylens_viewer::log(&format!("[skylens-client] 현황판 · core {}", opts.core_url));
    let doc = StatusApp::document(opts);
    skylens_viewer::shell::run_native(doc, skylens_viewer::WindowOptions::new(app::WINDOW_TITLE).with_icon(include_bytes!("../../res/icon/status/skylens-status-256.png")))
}

/// Web entry: draws the board into `<canvas id="skylens">`. The core is `?core=` or the page's own
/// origin (the gateway proxies `/viewer`); `?points=off` starts the point panel closed.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() -> Result<(), wasm_bindgen::JsValue> {
    let opts = Options {
        core_url: skylens_viewer::shell::page_core_url(skylens_viewer::link::VIEWER_PATH),
        points_on: skylens_viewer::shell::query_param("points").as_deref() != Some("off"),
    };
    skylens_viewer::log(&format!("[skylens] 현황판 · core {}", opts.core_url));
    skylens_viewer::shell::run_web(StatusApp::document(&opts), "skylens")
}
