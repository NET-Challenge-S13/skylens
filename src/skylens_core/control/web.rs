//! Web entry (`wasm32-unknown-unknown`): the tower document drawn into `<canvas id="skylens">` of
//! `web/index.html` (see `build-web.sh`). The core serves this build, so the socket is the page's
//! own origin with `/control` unless `?core=` says otherwise. `?display=` picks the building
//! display for one load, `?missionBridge=` the bridge. Live video is played by the browser's own
//! WebRTC (see [`crate::whep`]); `?whepBase=` replaces the page origin as the base of the feed's
//! WHEP URL and `?ice=` adds STUN/TURN servers (comma separated, none by default).

use crate::app::{ControlApp, Options, page};
use crate::bridge::{DEFAULT_BRIDGE, normalize_base};
use crate::settings::DisplayMode;
use skylens_viewer::ScreenDocument;
use skylens_viewer::link::CONTROL_PATH;
use skylens_viewer::shell::{page_core_url, query_param, run_web};
use wasm_bindgen::prelude::*;

/// The host page's canvas.
pub const CANVAS_ID: &str = "skylens";

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    let opts = Options {
        core_url: page_core_url(CONTROL_PATH),
        display_override: query_param("display").and_then(|d| DisplayMode::parse(&d)),
        bridge: normalize_base(&query_param("missionBridge").unwrap_or_else(|| DEFAULT_BRIDGE.to_string())),
        video: None,
    };
    let doc = ScreenDocument::new(&page(), ControlApp::new(opts));
    run_web(doc, CANVAS_ID)
}
