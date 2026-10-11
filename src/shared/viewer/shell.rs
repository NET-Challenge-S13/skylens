//! Document shell: a Blitz document built from an HTML template, the legacy CSS (unmodified) and
//! the bundled fonts; DOM event dispatch to the screen; native window and wasm entry points.
//!
//! A screen implements [`ScreenApp`] and is wrapped in a [`ScreenDocument`]:
//!
//! ```ignore
//! let page = Page::legacy(include_str!("ui/status.html"), &[include_str!("ui/status-panels.css")]);
//! let doc = ScreenDocument::new(&page, StatusApp::new(cfg));
//! skylens_viewer::shell::run_native(doc, WindowOptions::new("SkyLens · 현황판"))?;
//! ```
//!
//! The legacy pages were empty shells filled by TypeScript at runtime; the template given here is
//! the DOM those scripts produced (captured in `.worktrees/legacy-dom/`), and the screen changes
//! only values and classes through [`crate::dom::Dom`].

use crate::dom::{Dom, MarkupCache};
use blitz_dom::{BaseDocument, DocGuard, DocGuardMut, Document, DocumentConfig, EventDriver, EventHandler, FontContext, LocalName, NodeId};
use blitz_html::{HtmlDocument, HtmlProvider};
use blitz_traits::events::{DomEvent, DomEventData, EventState, MouseEventButton, UiEvent};
use keyboard_types::{Code, Key, Modifiers};
use std::sync::Arc;
use std::task::{Context, Waker};

// ---------------------------------------------------------------------------------------------
// Bundled assets
// ---------------------------------------------------------------------------------------------

/// The legacy shared stylesheet (`shared/viewer/style.css`), unmodified.
pub const STYLE_CSS: &str = include_str!("ui/style.css");
/// `@font-face` rules for the bundled fonts.
pub const FONTS_CSS: &str = include_str!("ui/fonts.css");

/// The fonts the legacy CSS asks for, bundled so web and native draw the same glyphs.
pub const BUNDLED_FONTS: &[(&str, &[u8])] = &[
    ("Oswald", include_bytes!("fonts/Oswald-VariableFont_wght.ttf")),
    ("Barlow", include_bytes!("fonts/Barlow-Regular.ttf")),
    ("Barlow", include_bytes!("fonts/Barlow-Medium.ttf")),
    ("Barlow", include_bytes!("fonts/Barlow-SemiBold.ttf")),
    ("IBM Plex Mono", include_bytes!("fonts/IBMPlexMono-Regular.ttf")),
    ("IBM Plex Mono", include_bytes!("fonts/IBMPlexMono-Medium.ttf")),
    ("IBM Plex Mono", include_bytes!("fonts/IBMPlexMono-SemiBold.ttf")),
    ("Noto Sans KR", include_bytes!("fonts/NotoSansKR-VariableFont_wght.ttf")),
];

/// The SkyLens window icon (the exe icon is embedded by each screen's build script).
/// Fallback window icon when a screen gives none (each screen ships its own, see `WindowOptions`).
pub const WINDOW_ICON_PNG: &[u8] = include_bytes!("../../../res/icon/skylens-256.png");

/// A font context with only the bundled fonts (no system fonts, so both targets match). Generic
/// families fall back to Noto Sans KR, which covers Korean and Latin.
pub fn font_context() -> FontContext {
    use parley::fontique::{Blob, Collection, CollectionOptions, FamilyId, GenericFamily, SourceCache};

    let mut ctx = FontContext {
        source_cache: SourceCache::new_shared(),
        collection: Collection::new(CollectionOptions { shared: false, system_fonts: false }),
    };
    let mut register = |bytes: &'static [u8]| -> Vec<FamilyId> {
        let blob: Blob<u8> = Blob::new(Arc::new(bytes) as Arc<dyn AsRef<[u8]> + Send + Sync>);
        ctx.collection.register_fonts(blob, None).into_iter().map(|(id, _)| id).collect()
    };
    let _ = register(blitz_dom::BULLET_FONT);
    let mut family = |name: &str| -> Vec<FamilyId> {
        BUNDLED_FONTS.iter().filter(|(n, _)| *n == name).flat_map(|(_, b)| register(*b)).collect()
    };
    // Oswald is only ever named explicitly (display font), so it joins no generic family.
    family("Oswald");
    let barlow = family("Barlow");
    let plex = family("IBM Plex Mono");
    let noto = family("Noto Sans KR");

    let dedup = |mut v: Vec<FamilyId>| {
        let mut seen = Vec::new();
        v.retain(|id| {
            let fresh = !seen.contains(id);
            seen.push(*id);
            fresh
        });
        v
    };
    let sans = dedup([barlow.clone(), noto.clone()].concat());
    let mono = dedup([plex, noto.clone()].concat());
    let korean = dedup(noto);
    ctx.collection.set_generic_families(GenericFamily::SansSerif, sans.into_iter());
    ctx.collection.set_generic_families(GenericFamily::Monospace, mono.into_iter());
    for generic in [GenericFamily::Serif, GenericFamily::SystemUi] {
        ctx.collection.set_generic_families(generic, korean.iter().copied());
    }
    ctx
}

// ---------------------------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------------------------

/// An HTML template plus the stylesheets to inline into it.
#[derive(Debug, Clone, PartialEq)]
pub struct Page {
    pub template: String,
    pub stylesheets: Vec<String>,
}

impl Page {
    pub fn new(template: impl Into<String>) -> Page {
        Page { template: template.into(), stylesheets: Vec::new() }
    }

    /// The template with the shared legacy CSS (`fonts.css`, `style.css`) followed by the
    /// screen's own sheets, in the legacy cascade order.
    pub fn legacy(template: impl Into<String>, screen_css: &[&str]) -> Page {
        let mut p = Page::new(template).with_css(FONTS_CSS).with_css(STYLE_CSS);
        for css in screen_css {
            p = p.with_css(*css);
        }
        p
    }

    pub fn with_css(mut self, css: impl Into<String>) -> Page {
        self.stylesheets.push(css.into());
        self
    }

    /// The final markup.
    pub fn html(&self) -> String {
        let sheets: Vec<&str> = self.stylesheets.iter().map(String::as_str).collect();
        inline_styles(&self.template, &sheets)
    }
}

/// Inlines stylesheets as `<style>` blocks at the end of `<head>` (author origin, in order), and
/// drops `<script>` elements and stylesheet links, which nothing would load.
pub fn inline_styles(template: &str, sheets: &[&str]) -> String {
    let mut html = strip_elements(template, "script");
    html = strip_stylesheet_links(&html);
    let mut block = String::new();
    for css in sheets {
        block.push_str("<style>\n");
        block.push_str(css);
        block.push_str("\n</style>\n");
    }
    match find_ci(&html, "</head>") {
        Some(i) => html.insert_str(i, &block),
        None => match find_ci(&html, "<body") {
            Some(i) => html.insert_str(i, &format!("<head>{block}</head>")),
            None => html.insert_str(0, &block),
        },
    }
    html
}

fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    haystack.to_ascii_lowercase().find(&needle.to_ascii_lowercase())
}

/// Removes every `<tag ...>...</tag>` (case-insensitive).
fn strip_elements(html: &str, tag: &str) -> String {
    let lower = html.to_ascii_lowercase();
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut out = String::with_capacity(html.len());
    let mut pos = 0;
    while let Some(rel) = lower[pos..].find(&open) {
        let start = pos + rel;
        // `<scripts` or `<script-x` are other tags.
        let next = lower[start + open.len()..].chars().next();
        if !matches!(next, Some(c) if c.is_whitespace() || c == '>' || c == '/') {
            out.push_str(&html[pos..start + open.len()]);
            pos = start + open.len();
            continue;
        }
        out.push_str(&html[pos..start]);
        pos = match lower[start..].find(&close) {
            Some(end) => start + end + close.len(),
            None => html.len(),
        };
    }
    out.push_str(&html[pos..]);
    out
}

/// Removes `<link ... rel="stylesheet" ...>` tags; the sheets are inlined instead.
fn strip_stylesheet_links(html: &str) -> String {
    let lower = html.to_ascii_lowercase();
    let mut out = String::with_capacity(html.len());
    let mut pos = 0;
    while let Some(rel) = lower[pos..].find("<link") {
        let start = pos + rel;
        let end = lower[start..].find('>').map_or(html.len(), |e| start + e + 1);
        let tag = &lower[start..end];
        out.push_str(&html[pos..start]);
        if !tag.contains("stylesheet") {
            out.push_str(&html[start..end]);
        }
        pos = end;
    }
    out.push_str(&html[pos..]);
    out
}

// ---------------------------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    Click,
    DoubleClick,
    ContextMenu,
    PointerDown,
    PointerUp,
    KeyDown,
    KeyUp,
    /// A text field's value changed (`value` holds it).
    Input,
    Focus,
    Blur,
}

/// One element on the event path.
#[derive(Debug, Clone, PartialEq)]
pub struct ElementInfo {
    pub node: NodeId,
    pub tag: String,
    pub id: Option<String>,
    pub classes: Vec<String>,
    /// `data-*` attributes without the prefix, e.g. `("action", "open-route")`.
    pub data: Vec<(String, String)>,
}

impl ElementInfo {
    pub fn has_class(&self, class: &str) -> bool {
        self.classes.iter().any(|c| c == class)
    }

    pub fn data(&self, name: &str) -> Option<&str> {
        self.data.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyInput {
    pub code: Code,
    pub repeat: bool,
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub meta: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointerInput {
    pub page_x: f32,
    pub page_y: f32,
    pub client_x: f32,
    pub client_y: f32,
    /// 0 main, 1 auxiliary (middle), 2 secondary (right).
    pub button: u8,
    pub shift: bool,
}

/// A DOM event as the screen sees it: what happened, and the element path from the target up.
#[derive(Debug, Clone, PartialEq)]
pub struct UiInput {
    pub kind: InputKind,
    /// Target first, then its element ancestors up to `<html>`.
    pub path: Vec<ElementInfo>,
    pub key: Option<KeyInput>,
    /// The printable key text, if any (e.g. `"a"`).
    pub key_text: Option<String>,
    pub pointer: Option<PointerInput>,
    pub value: Option<String>,
}

impl UiInput {
    pub fn is_click(&self) -> bool {
        self.kind == InputKind::Click
    }

    pub fn target(&self) -> Option<&ElementInfo> {
        self.path.first()
    }

    /// Innermost element on the path with this id.
    pub fn closest_id(&self, id: &str) -> Option<&ElementInfo> {
        self.path.iter().find(|e| e.id.as_deref() == Some(id))
    }

    /// Innermost element on the path with this class.
    pub fn closest_class(&self, class: &str) -> Option<&ElementInfo> {
        self.path.iter().find(|e| e.has_class(class))
    }

    /// Innermost element on the path with `data-<name>`, and the value.
    pub fn closest_data(&self, name: &str) -> Option<(&ElementInfo, &str)> {
        self.path.iter().find_map(|e| e.data(name).map(|v| (e, v)))
    }

    pub fn closest_tag(&self, tag: &str) -> Option<&ElementInfo> {
        self.path.iter().find(|e| e.tag.eq_ignore_ascii_case(tag))
    }

    /// The event targets a text field (keys typed there are not shortcuts).
    pub fn in_text_field(&self) -> bool {
        self.target()
            .is_some_and(|e| matches!(e.tag.as_str(), "input" | "textarea" | "select"))
    }

    pub fn code(&self) -> Option<Code> {
        self.key.map(|k| k.code)
    }

    pub fn key_down(&self, code: Code) -> bool {
        self.kind == InputKind::KeyDown && self.code() == Some(code)
    }
}

fn element_info(doc: &BaseDocument, node: NodeId) -> Option<ElementInfo> {
    let el = doc.get_node(node)?.element_data()?;
    let mut info = ElementInfo { node, tag: el.name.local.to_string(), id: None, classes: Vec::new(), data: Vec::new() };
    for a in el.attrs() {
        let name: &str = &a.name.local;
        if name == "id" {
            info.id = Some(a.value.clone());
        } else if name == "class" {
            info.classes = a.value.split_whitespace().map(str::to_string).collect();
        } else if let Some(key) = name.strip_prefix("data-") {
            info.data.push((key.to_string(), a.value.clone()));
        }
    }
    Some(info)
}

fn button_index(b: MouseEventButton) -> u8 {
    match b {
        MouseEventButton::Main => 0,
        MouseEventButton::Auxiliary => 1,
        MouseEventButton::Secondary => 2,
        _ => 3,
    }
}

fn key_input(k: &blitz_traits::events::BlitzKeyEvent) -> KeyInput {
    KeyInput {
        code: k.code,
        repeat: k.is_auto_repeating,
        shift: k.modifiers.contains(Modifiers::SHIFT),
        ctrl: k.modifiers.contains(Modifiers::CONTROL),
        alt: k.modifiers.contains(Modifiers::ALT),
        meta: k.modifiers.contains(Modifiers::META),
    }
}

/// Turns a dispatched DOM event into a [`UiInput`].
pub fn ui_input(doc: &BaseDocument, chain: &[NodeId], data: &DomEventData) -> Option<UiInput> {
    let pointer = |e: &blitz_traits::events::BlitzPointerEvent| PointerInput {
        page_x: e.coords.page_x,
        page_y: e.coords.page_y,
        client_x: e.coords.client_x,
        client_y: e.coords.client_y,
        button: button_index(e.button),
        shift: e.mods.contains(Modifiers::SHIFT),
    };
    let (kind, key, key_text, ptr, value) = match data {
        DomEventData::Click(e) => (InputKind::Click, None, None, Some(pointer(e)), None),
        DomEventData::DoubleClick(e) => (InputKind::DoubleClick, None, None, Some(pointer(e)), None),
        DomEventData::ContextMenu(e) => (InputKind::ContextMenu, None, None, Some(pointer(e)), None),
        DomEventData::PointerDown(e) => (InputKind::PointerDown, None, None, Some(pointer(e)), None),
        DomEventData::PointerUp(e) => (InputKind::PointerUp, None, None, Some(pointer(e)), None),
        DomEventData::KeyDown(k) => (InputKind::KeyDown, Some(key_input(k)), key_text(k), None, None),
        DomEventData::KeyUp(k) => (InputKind::KeyUp, Some(key_input(k)), key_text(k), None, None),
        DomEventData::Input(i) => (InputKind::Input, None, None, None, Some(i.value.clone())),
        DomEventData::Focus(_) => (InputKind::Focus, None, None, None, None),
        DomEventData::Blur(_) => (InputKind::Blur, None, None, None, None),
        _ => return None,
    };
    let path = chain.iter().filter_map(|n| element_info(doc, *n)).collect();
    Some(UiInput { kind, path, key, key_text, pointer: ptr, value })
}

fn key_text(k: &blitz_traits::events::BlitzKeyEvent) -> Option<String> {
    match &k.key {
        Key::Character(s) => Some(s.to_string()),
        _ => k.text.as_ref().map(|t| t.to_string()),
    }
}

// ---------------------------------------------------------------------------------------------
// Screen app and document
// ---------------------------------------------------------------------------------------------

/// What [`ScreenApp::poll`] gets besides the DOM.
pub struct PollCx<'a> {
    pub now_ms: f64,
    /// Hand this to [`crate::link::CoreLink::set_waker`] so socket events wake the window.
    pub waker: Option<&'a Waker>,
}

/// A screen: owns its state, the core link and its widgets, and writes into the DOM.
pub trait ScreenApp: 'static {
    /// Once, right after the document is parsed: mount the 3D widget, build widgets, first values.
    fn mount(&mut self, dom: &mut Dom<'_>) {
        let _ = dom;
    }

    /// On every wake (socket event, ticker, redraw). Return true when the frame must be redrawn
    /// for a reason the DOM writes do not cover (they are tracked already).
    fn poll(&mut self, dom: &mut Dom<'_>, cx: &mut PollCx<'_>) -> bool;

    /// A DOM event (click, key, input), after Blitz dispatched it.
    fn input(&mut self, input: &UiInput, dom: &mut Dom<'_>) -> bool;

    /// Asked during dispatch: cancel the browser default for this event (e.g. Tab outside text
    /// fields, so focus does not move).
    fn prevent_default(&self, input: &UiInput) -> bool {
        let _ = input;
        false
    }
}

struct Collector<'a, A: ScreenApp> {
    inputs: &'a mut Vec<UiInput>,
    app: &'a A,
}

impl<A: ScreenApp> EventHandler for Collector<'_, A> {
    fn handle_event(&mut self, chain: &[NodeId], event: &mut DomEvent, doc: &mut dyn Document, event_state: &mut EventState) {
        let doc = doc.inner();
        let Some(input) = ui_input(&doc, chain, &event.data) else { return };
        if self.app.prevent_default(&input) {
            event_state.prevent_default();
        }
        self.inputs.push(input);
    }
}

/// A screen as a Blitz document.
pub struct ScreenDocument<A: ScreenApp> {
    inner: BaseDocument,
    app: A,
    cache: MarkupCache,
    inputs: Vec<UiInput>,
}

impl<A: ScreenApp> ScreenDocument<A> {
    /// Parses the page with the bundled fonts and mounts the app.
    pub fn new(page: &Page, app: A) -> Self {
        let config = DocumentConfig {
            html_parser_provider: Some(Arc::new(HtmlProvider)),
            font_ctx: Some(font_context()),
            ..Default::default()
        };
        let inner = HtmlDocument::from_html(&page.html(), config).into_inner();
        let mut doc = ScreenDocument { inner, app, cache: MarkupCache::default(), inputs: Vec::new() };
        {
            let mut dom = Dom::new(&mut doc.inner, &mut doc.cache);
            doc.app.mount(&mut dom);
        }
        doc
    }

    pub fn app(&self) -> &A {
        &self.app
    }

    pub fn app_mut(&mut self) -> &mut A {
        &mut self.app
    }

    /// Runs `f` with the app and a DOM view (e.g. from a test or an entry point).
    pub fn with_dom<R>(&mut self, f: impl FnOnce(&mut A, &mut Dom<'_>) -> R) -> R {
        let mut dom = Dom::new(&mut self.inner, &mut self.cache);
        f(&mut self.app, &mut dom)
    }
}

impl<A: ScreenApp> Document for ScreenDocument<A> {
    fn inner(&self) -> DocGuard<'_> {
        DocGuard::Ref(&self.inner)
    }

    fn inner_mut(&mut self) -> DocGuardMut<'_> {
        DocGuardMut::Ref(&mut self.inner)
    }

    fn handle_ui_event(&mut self, event: UiEvent) {
        {
            let collector = Collector { inputs: &mut self.inputs, app: &self.app };
            let mut driver = EventDriver::new(&mut self.inner, collector);
            driver.handle_ui_event(event);
        }
        let inputs = std::mem::take(&mut self.inputs);
        let changed = {
            let mut dom = Dom::new(&mut self.inner, &mut self.cache);
            for input in &inputs {
                self.app.input(input, &mut dom);
            }
            dom.changed()
        };
        if changed {
            // Blitz hit-tests against `paint_children`, which only a resolve rebuilds. Without
            // this, a pointer event that arrives after we removed nodes walks a stale child list
            // and panics on the missing node.
            self.inner.resolve(crate::now_ms() as f64 / 1000.0);
        }
        // Pointer and wheel events may have moved the 3D camera without touching the DOM.
        self.inner.shell_provider.request_redraw();
    }

    fn poll(&mut self, task_context: Option<Context<'_>>) -> bool {
        let waker = task_context.as_ref().map(|c| c.waker());
        let mut cx = PollCx { now_ms: crate::now_ms(), waker };
        let (wants, changed) = {
            let mut dom = Dom::new(&mut self.inner, &mut self.cache);
            let wants = self.app.poll(&mut dom, &mut cx);
            (wants, dom.changed())
        };
        if changed {
            // Same reason as in `handle_ui_event`: keep the hit-test tree in step with the DOM.
            self.inner.resolve(cx.now_ms as f64 / 1000.0);
        }
        wants || changed
    }
}

/// Reads an attribute of a node (for apps that keep node ids).
pub fn node_attr(doc: &BaseDocument, node: NodeId, name: &str) -> Option<String> {
    doc.get_node(node)?.element_data()?.attr(LocalName::from(name)).map(str::to_string)
}

// ---------------------------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------------------------

/// Native window options.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowOptions {
    pub title: String,
    /// The screen's icon (PNG, used off Windows; on Windows the exe's embedded icon is used).
    pub icon_png: Option<&'static [u8]>,
    /// Logical size.
    pub width: f64,
    pub height: f64,
}

impl WindowOptions {
    pub fn new(title: impl Into<String>) -> WindowOptions {
        WindowOptions { title: title.into(), icon_png: None, width: 1440.0, height: 900.0 }
    }

    /// The screen's own icon (PNG) for platforms other than Windows.
    pub fn with_icon(mut self, png: &'static [u8]) -> WindowOptions {
        self.icon_png = Some(png);
        self
    }
}

/// Opens one `blitz-shell` window rendered with vello (Vulkan / DX12 / Metal) and runs the event
/// loop until it closes.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_native(doc: impl Document, opts: WindowOptions) -> anyhow::Result<()> {
    use anyrender_vello::VelloWindowRenderer;
    use blitz_shell::{BlitzApplication, BlitzShellProxy, WindowConfig, create_default_event_loop};
    use winit::dpi::LogicalSize;
    use winit::window::WindowAttributes;

    let event_loop = create_default_event_loop();
    let (proxy, receiver) = BlitzShellProxy::new(event_loop.create_proxy());
    let mut application = BlitzApplication::new(proxy, receiver);
    let attributes = WindowAttributes::default()
        .with_title(opts.title)
        .with_surface_size(LogicalSize::new(opts.width, opts.height));
    // Windows: winit turns an RGBA icon into one 256x256 HICON, which the task bar cannot scale
    // to its own size (64 px at 200 %) and shows as a grey placeholder. Load the exe's embedded
    // icon (resource 1, written by each screen's build.rs) at the exact system sizes instead:
    // ICON_SMALL for the title bar, ICON_BIG (`with_taskbar_icon`) for the task bar.
    #[cfg(windows)]
    let attributes = {
        let (small, big) = windows_icons();
        attributes.with_window_icon(small).with_platform_attributes(Box::new(
            winit::platform::windows::WindowAttributesWindows::default().with_taskbar_icon(big),
        ))
    };
    #[cfg(not(windows))]
    let attributes = attributes.with_window_icon(window_icon_from(opts.icon_png.unwrap_or(WINDOW_ICON_PNG)));
    application.add_window(WindowConfig::with_attributes(Box::new(doc), VelloWindowRenderer::new(), attributes));
    event_loop.run_app(application).map_err(|e| anyhow::anyhow!("event loop: {e}"))
}

/// The default SkyLens icon for the title bar and task bar.
#[cfg(not(target_arch = "wasm32"))]
pub fn window_icon() -> Option<winit::icon::Icon> {
    window_icon_from(WINDOW_ICON_PNG)
}

/// A window icon from PNG bytes.
#[cfg(not(target_arch = "wasm32"))]
pub fn window_icon_from(png: &[u8]) -> Option<winit::icon::Icon> {
    let rgba = image::load_from_memory(png).ok()?.into_rgba8();
    let (w, h) = rgba.dimensions();
    winit::icon::RgbaIcon::new(rgba.into_raw(), w, h).ok().map(Into::into)
}

/// The exe's embedded icon (resource 1) at the system's small and large icon sizes.
#[cfg(windows)]
fn windows_icons() -> (Option<winit::icon::Icon>, Option<winit::icon::Icon>) {
    use windows_sys::Win32::UI::WindowsAndMessaging::{GetSystemMetrics, SM_CXICON, SM_CXSMICON};
    use winit::dpi::PhysicalSize;
    use winit::platform::windows::WinIcon;
    let load = |metric| {
        // SAFETY: GetSystemMetrics has no preconditions.
        let px = unsafe { GetSystemMetrics(metric) }.max(16) as u32;
        WinIcon::from_resource(1, Some(PhysicalSize::new(px, px))).ok().map(Into::into)
    };
    let fallback = || window_icon();
    (load(SM_CXSMICON).or_else(fallback), load(SM_CXICON).or_else(fallback))
}

/// Draws the document into `<canvas id="{canvas_id}">` with vello_hybrid (WebGPU, else WebGL2),
/// following Blitz's examples/wasm_hello. The page's CSS sizes the canvas.
#[cfg(target_arch = "wasm32")]
pub fn run_web(doc: impl Document, canvas_id: &str) -> Result<(), wasm_bindgen::JsValue> {
    use anyrender_vello_hybrid::VelloHybridWindowRenderer;
    use blitz_shell::{BlitzApplication, BlitzShellProxy, WindowConfig};
    use wasm_bindgen::JsCast as _;
    use winit::event_loop::EventLoop;
    use winit::platform::web::WindowAttributesWeb;
    use winit::window::WindowAttributes;

    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&format!("[skylens] panic: {info}").into());
    }));
    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let canvas = document
        .get_element_by_id(canvas_id)
        .ok_or_else(|| format!("missing <canvas id=\"{canvas_id}\">"))?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;
    // Focusable, so key events (manual control, Tab, Esc) reach the app.
    canvas.set_tab_index(0);
    canvas.style().set_property("outline", "none")?;
    let _ = canvas.focus();

    let event_loop = EventLoop::new().map_err(|e| wasm_bindgen::JsValue::from_str(&format!("{e}")))?;
    let (proxy, receiver) = BlitzShellProxy::new(event_loop.create_proxy());
    let attributes = WindowAttributes::default()
        .with_platform_attributes(Box::new(WindowAttributesWeb::default().with_canvas(Some(canvas))));
    let mut app = BlitzApplication::<VelloHybridWindowRenderer>::new(proxy, receiver);
    app.add_window(WindowConfig::with_attributes(Box::new(doc), VelloHybridWindowRenderer::new(), attributes));
    event_loop.run_app(app).map_err(|e| wasm_bindgen::JsValue::from_str(&format!("{e}")))?;
    Ok(())
}

/// A query parameter of the page URL (`?core=…`), empty values ignored.
#[cfg(target_arch = "wasm32")]
pub fn query_param(name: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    let params = web_sys::UrlSearchParams::new_with_str(&search).ok()?;
    params.get(name).filter(|v| !v.trim().is_empty())
}

/// The core socket for this page: `?core=` (host, http or ws URL) if given, else the page's own
/// origin with `path`.
#[cfg(target_arch = "wasm32")]
pub fn page_core_url(path: &str) -> String {
    if let Some(c) = query_param("core") {
        return crate::link::normalize_core_url(&c, path);
    }
    let location = web_sys::window().map(|w| w.location());
    let secure = location.as_ref().and_then(|l| l.protocol().ok()).is_some_and(|p| p == "https:");
    let host = location.and_then(|l| l.host().ok()).unwrap_or_else(|| "127.0.0.1:8080".into());
    crate::link::page_core_url(secure, &host, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn styles_go_at_the_end_of_head_in_order() {
        let html = "<!doctype html><html><head><title>x</title></head><body><p>a</p></body></html>";
        let out = inline_styles(html, &[".a{}", ".b{}"]);
        let a = out.find(".a{}").unwrap();
        let b = out.find(".b{}").unwrap();
        let head_end = out.find("</head>").unwrap();
        assert!(a < b && b < head_end);
        assert!(out.find("<title>").unwrap() < a);
    }

    #[test]
    fn a_template_without_head_still_gets_styles() {
        let out = inline_styles("<body><div id=\"x\"></div></body>", &[".x{}"]);
        assert!(out.starts_with("<head><style>"));
        let out = inline_styles("<div></div>", &[".x{}"]);
        assert!(out.starts_with("<style>"));
    }

    #[test]
    fn scripts_and_stylesheet_links_are_dropped() {
        let html = concat!(
            "<head><SCRIPT type=\"module\" src=\"/@vite/client\"></script>",
            "<link rel=\"icon\" href=\"/favicon.svg\">",
            "<link rel=\"stylesheet\" href=\"style.css\" />",
            "</head><body><div class=\"scripts\">keep</div>",
            "<script type=\"module\">let x = '</div>';</script></body>"
        );
        let out = inline_styles(html, &[]);
        assert!(!out.to_ascii_lowercase().contains("<script"));
        assert!(!out.contains("style.css"));
        assert!(out.contains("favicon.svg"));
        assert!(out.contains("<div class=\"scripts\">keep</div>"));
    }

    #[test]
    fn legacy_page_orders_shared_css_first() {
        let p = Page::new("<head></head>").with_css("a").with_css("b");
        assert_eq!(p.stylesheets, vec!["a".to_string(), "b".to_string()]);
        let l = Page::legacy("<head></head>", &[".screen{}"]);
        assert_eq!(l.stylesheets.len(), 3);
        assert_eq!(l.stylesheets[1], STYLE_CSS);
        assert_eq!(l.stylesheets[2], ".screen{}");
    }

    fn info(tag: &str, id: Option<&str>, classes: &[&str], data: &[(&str, &str)]) -> ElementInfo {
        ElementInfo {
            node: NodeId::from_u64(1),
            tag: tag.into(),
            id: id.map(str::to_string),
            classes: classes.iter().map(|c| c.to_string()).collect(),
            data: data.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
        }
    }

    #[test]
    fn input_paths_answer_closest_queries() {
        let input = UiInput {
            kind: InputKind::Click,
            path: vec![
                info("span", None, &["route-modal__span-label"], &[]),
                info("button", None, &["route-modal__span"], &[("span", "300")]),
                info("div", Some("route-overlay"), &["route-modal-overlay"], &[]),
            ],
            key: None,
            key_text: None,
            pointer: None,
            value: None,
        };
        assert!(input.is_click());
        assert_eq!(input.closest_class("route-modal__span").map(|e| e.tag.as_str()), Some("button"));
        assert_eq!(input.closest_data("span").map(|(_, v)| v), Some("300"));
        assert!(input.closest_id("route-overlay").is_some());
        assert!(input.closest_id("nope").is_none());
        assert_eq!(input.closest_tag("BUTTON").map(|e| e.has_class("route-modal__span")), Some(true));
        assert!(!input.in_text_field());
    }
}
