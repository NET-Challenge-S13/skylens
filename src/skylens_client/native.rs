//! Native entry: one `blitz-shell` window per screen, rendered with vello on
//! Vulkan / DX12 / Metal. System fonts come from Blitz's `system-fonts` feature.

use crate::app::{Config, SkylensDocument};
use anyrender_vello::VelloWindowRenderer;
use blitz_shell::{create_default_event_loop, BlitzApplication, BlitzShellProxy, WindowConfig};
use winit::dpi::LogicalSize;
use winit::window::WindowAttributes;

pub fn run(cfg: Config) -> anyhow::Result<()> {
    crate::log(&format!(
        "[skylens-client] {} · core {} · anchor {:.5},{:.5},{:.1}",
        cfg.screen.title(),
        cfg.core_url,
        cfg.anchor.lat,
        cfg.anchor.lon,
        cfg.anchor.alt
    ));
    let title = cfg.screen.title();
    let doc = SkylensDocument::new(cfg, None);

    let event_loop = create_default_event_loop();
    let (proxy, receiver) = BlitzShellProxy::new(event_loop.create_proxy());
    let mut application = BlitzApplication::new(proxy, receiver);
    let attributes = WindowAttributes::default()
        .with_title(title)
        .with_window_icon(window_icon())
        .with_surface_size(LogicalSize::new(1440.0, 900.0));
    application.add_window(WindowConfig::with_attributes(
        Box::new(doc),
        VelloWindowRenderer::new(),
        attributes,
    ));
    event_loop
        .run_app(application)
        .map_err(|e| anyhow::anyhow!("event loop: {e}"))
}

/// The SkyLens icon for the title bar and task bar (the exe icon is embedded by build.rs).
fn window_icon() -> Option<winit::icon::Icon> {
    let png = include_bytes!("../../res/icon/skylens-256.png");
    let rgba = image::load_from_memory(png).ok()?.into_rgba8();
    let (w, h) = rgba.dimensions();
    winit::icon::RgbaIcon::new(rgba.into_raw(), w, h).ok().map(Into::into)
}
