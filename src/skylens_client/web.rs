//! Web entry (`wasm32-unknown-unknown`): the same Blitz document drawn into a
//! `<canvas id="skylens">` by `blitz-shell` with the vello_hybrid renderer, which
//! runs on WebGPU and falls back to WebGL2. Follows Blitz's examples/wasm_hello.
//!
//! Configuration comes from the page:
//! - `?screen=control|status`, else `<meta name="skylens-screen" content="…">`, else status
//! - `?core=ws://…`, else `ws(s)://<page host>/viewer` (core and gateway both proxy `/viewer`)
//! - `?anchor=LAT,LON[,ALT]`, else the default anchor
//! - `fonts/NotoSansKR.ttf` next to the page, registered as every generic family,
//!   because browsers do not expose system fonts to wasm

use crate::app::{normalize_core_url, parse_anchor, Config, Screen, SkylensDocument, DEFAULT_ANCHOR};
use anyrender_vello_hybrid::VelloHybridWindowRenderer;
use blitz_shell::{BlitzApplication, BlitzShellProxy, WindowConfig};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast as _;
use wasm_bindgen_futures::JsFuture;
use winit::event_loop::EventLoop;
use winit::platform::web::WindowAttributesWeb;
use winit::window::WindowAttributes;

const FONT_URL: &str = "fonts/NotoSansKR.ttf";

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&format!("[skylens] panic: {info}").into());
    }));
    wasm_bindgen_futures::spawn_local(async {
        if let Err(e) = boot().await {
            web_sys::console::error_1(&e);
        }
    });
    Ok(())
}

fn config_from_page(window: &web_sys::Window) -> Config {
    let location = window.location();
    let params = location
        .search()
        .ok()
        .and_then(|s| web_sys::UrlSearchParams::new_with_str(&s).ok());
    let param = |name: &str| params.as_ref().and_then(|p| p.get(name)).filter(|v| !v.is_empty());

    let meta_screen = window
        .document()
        .and_then(|d| d.query_selector("meta[name=skylens-screen]").ok().flatten())
        .and_then(|m| m.get_attribute("content"));
    let screen = param("screen")
        .or(meta_screen)
        .and_then(|s| Screen::parse(&s))
        .unwrap_or(Screen::Status);

    let core_url = match param("core") {
        Some(c) => normalize_core_url(&c),
        None => {
            let secure = location.protocol().map(|p| p == "https:").unwrap_or(false);
            let host = location.host().unwrap_or_else(|_| "127.0.0.1:8080".into());
            format!("{}://{host}/viewer", if secure { "wss" } else { "ws" })
        }
    };
    let anchor = param("anchor").and_then(|a| parse_anchor(&a)).unwrap_or(DEFAULT_ANCHOR);
    Config { screen, core_url, anchor }
}

async fn fetch_bytes(window: &web_sys::Window, url: &str) -> Result<Vec<u8>, JsValue> {
    let resp: web_sys::Response = JsFuture::from(window.fetch_with_str(url)).await?.dyn_into()?;
    if !resp.ok() {
        return Err(JsValue::from_str(&format!("{url}: HTTP {}", resp.status())));
    }
    let buf = JsFuture::from(resp.array_buffer()?).await?;
    Ok(js_sys::Uint8Array::new(&buf).to_vec())
}

async fn boot() -> Result<(), JsValue> {
    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let cfg = config_from_page(&window);
    document.set_title(cfg.screen.title());
    crate::log(&format!("[skylens] {} · core {}", cfg.screen.title(), cfg.core_url));

    let font_ctx = match fetch_bytes(&window, FONT_URL).await {
        Ok(bytes) => Some(blitz_dom::build_single_font_ctx(&bytes)),
        Err(e) => {
            crate::log(&format!("[skylens] Korean font not loaded ({e:?}); text may not render"));
            None
        }
    };

    let canvas = document
        .get_element_by_id("skylens")
        .ok_or("missing <canvas id=\"skylens\">")?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;
    // Focusable, so key events (manual control, Tab, Esc) reach the app.
    canvas.set_tab_index(0);
    canvas.style().set_property("outline", "none")?;
    let _ = canvas.focus();

    let doc = SkylensDocument::new(cfg, font_ctx);
    let event_loop = EventLoop::new().map_err(|e| JsValue::from_str(&format!("{e}")))?;
    let (proxy, receiver) = BlitzShellProxy::new(event_loop.create_proxy());
    // No with_surface_size on wasm: the page's CSS sizes the canvas (see wasm_hello).
    let attributes = WindowAttributes::default()
        .with_platform_attributes(Box::new(WindowAttributesWeb::default().with_canvas(Some(canvas))));
    let mut app = BlitzApplication::<VelloHybridWindowRenderer>::new(proxy, receiver);
    app.add_window(WindowConfig::with_attributes(
        Box::new(doc),
        VelloHybridWindowRenderer::new(),
        attributes,
    ));
    event_loop
        .run_app(app)
        .map_err(|e| JsValue::from_str(&format!("{e}")))?;
    Ok(())
}
