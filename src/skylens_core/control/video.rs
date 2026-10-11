//! Camera panel (legacy `control/videoPanel.ts`): one aircraft's feed, titled after its station
//! (LEFT / CENTER / RIGHT CAM), with a readout line over it.
//!
//! - Standby: the legacy synthetic pattern (slow hue gradient, scanlines, crosshair) drawn by
//!   [`StandbyWidget`] on `canvas.video-panel__canvas` (320 x 200 buffer, `object-fit: cover`).
//! - Live (native only): the feed the core announces in `camera_feed` is read by an `ffmpeg`
//!   subprocess (input URL from a template, by default SRT from the MediaMTX next to the core)
//!   and its frames are drawn by [`FrameWidget`] on the `<video>` element, toggled with the same
//!   `is-hidden` classes as the legacy `<video>`. The web build has no decoder for the WHEP feed
//!   and stays on standby.

use crate::labels::{self, Station};
use blitz_dom::Widget;
use blitz_dom::node::ComputedStyles;
use skylens_viewer::NodeId;
use skylens_viewer::anyrender::{self, PaintScene as _, Scene};
use skylens_viewer::dom::Dom;
use skylens_viewer::peniko::kurbo::{Affine, BezPath, Circle, Point, Rect, Stroke};
use skylens_viewer::peniko::{Color, Fill, Gradient};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Legacy canvas buffer.
pub const STANDBY_W: f64 = 320.0;
pub const STANDBY_H: f64 = 200.0;

/// Decoded frame size requested from ffmpeg (16:10 like the panel, cropped to fill).
pub const FRAME_W: u32 = 640;
pub const FRAME_H: u32 = 400;

/// Playback errors, worded as the legacy panel showed them.
pub const ERR_GENERIC: &str = "영상 재생 불가";
pub const ERR_SOURCE: &str = "영상 재생 불가 \u{2014} 소스 오류";

/// `hsl(h, s%, l%)` to RGB bytes.
pub fn hsl(h: f64, s: f64, l: f64) -> [u8; 3] {
    let h = h.rem_euclid(360.0) / 360.0;
    let s = s / 100.0;
    let l = l / 100.0;
    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;
    let hue = |mut t: f64| {
        if t < 0.0 {
            t += 1.0;
        }
        if t > 1.0 {
            t -= 1.0;
        }
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };
    [hue(h + 1.0 / 3.0), hue(h), hue(h - 1.0 / 3.0)].map(|c| (c * 255.0).round().clamp(0.0, 255.0) as u8)
}

/// Transform that draws a `bw x bh` buffer into `w x h` like `object-fit: cover`.
pub fn cover(bw: f64, bh: f64, w: f64, h: f64) -> Affine {
    let s = (w / bw).max(h / bh);
    Affine::translate(((w - bw * s) / 2.0, (h - bh * s) / 2.0)) * Affine::scale(s)
}

/// Draws the legacy standby pattern at time `t` seconds into the 320 x 200 buffer space.
pub fn draw_standby(scene: &mut Scene, base: Affine, t: f64) {
    let (w, h) = (STANDBY_W, STANDBY_H);
    let hue = (t * 8.0) % 360.0;
    let [r0, g0, b0] = hsl(hue, 40.0, 12.0);
    let [r1, g1, b1] = hsl((hue + 60.0) % 360.0, 35.0, 6.0);
    let grad = Gradient::new_linear((0.0, 0.0), (w, h)).with_stops([Color::from_rgb8(r0, g0, b0), Color::from_rgb8(r1, g1, b1)]);
    let full = Rect::new(0.0, 0.0, w, h);
    scene.fill(Fill::NonZero, base, &grad, None, &full);

    let mut lines = BezPath::new();
    let mut y = 0.0;
    while y < h {
        lines.move_to((0.0, y));
        lines.line_to((w, y));
        y += 3.0;
    }
    scene.stroke(&Stroke::new(1.0), base, Color::from_rgba8(160, 220, 255, 15), None, &lines);

    let (cx, cy) = (w / 2.0, h / 2.0);
    let mut cross = BezPath::new();
    for (a, b) in [((cx - 14.0, cy), (cx - 4.0, cy)), ((cx + 4.0, cy), (cx + 14.0, cy)), ((cx, cy - 14.0), (cx, cy - 4.0)), ((cx, cy + 4.0), (cx, cy + 14.0))] {
        cross.move_to(a);
        cross.line_to(b);
    }
    let ink = Color::from_rgba8(159, 232, 255, 140);
    scene.stroke(&Stroke::new(1.2), base, ink, None, &cross);
    scene.stroke(&Stroke::new(1.2), base, ink, None, &Circle::new(Point::new(cx, cy), 22.0));
}

/// Standby pattern widget.
pub struct StandbyWidget {
    start_ms: Option<f64>,
}

impl StandbyWidget {
    pub fn boxed() -> Box<dyn Widget> {
        Box::new(StandbyWidget { start_ms: None })
    }
}

impl Widget for StandbyWidget {
    fn requires_redraw(&self) -> bool {
        true
    }

    fn paint(&mut self, _ctx: &mut dyn anyrender::RenderContext, _styles: &ComputedStyles, width: u32, height: u32, _scale: f64) -> Scene {
        let mut scene = Scene::new();
        if width == 0 || height == 0 {
            return scene;
        }
        let now = skylens_viewer::now_ms();
        let t = (now - *self.start_ms.get_or_insert(now)) / 1000.0;
        let (w, h) = (f64::from(width), f64::from(height));
        scene.push_clip_layer(Fill::NonZero, Affine::IDENTITY, &Rect::new(0.0, 0.0, w, h));
        draw_standby(&mut scene, cover(STANDBY_W, STANDBY_H, w, h), t);
        scene.pop_layer();
        scene
    }
}

/// One decoded RGBA frame.
#[derive(Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgba: Arc<Vec<u8>>,
}

/// The latest frame of the live player, shared with [`FrameWidget`].
pub type FrameSlot = Arc<Mutex<Option<Frame>>>;

/// Draws the latest live frame like `object-fit: cover`.
pub struct FrameWidget {
    slot: FrameSlot,
}

impl FrameWidget {
    pub fn boxed(slot: FrameSlot) -> Box<dyn Widget> {
        Box::new(FrameWidget { slot })
    }
}

impl Widget for FrameWidget {
    fn requires_redraw(&self) -> bool {
        self.slot.lock().map(|s| s.is_some()).unwrap_or(false)
    }

    fn paint(&mut self, _ctx: &mut dyn anyrender::RenderContext, _styles: &ComputedStyles, width: u32, height: u32, _scale: f64) -> Scene {
        use skylens_viewer::peniko::{Blob, ImageAlphaType, ImageBrush, ImageData, ImageFormat};
        let mut scene = Scene::new();
        let Some(frame) = self.slot.lock().ok().and_then(|s| s.clone()) else { return scene };
        if width == 0 || height == 0 || frame.width == 0 || frame.height == 0 {
            return scene;
        }
        let image = ImageData {
            data: Blob::new(frame.rgba.clone() as Arc<dyn AsRef<[u8]> + Send + Sync>),
            format: ImageFormat::Rgba8,
            alpha_type: ImageAlphaType::Alpha,
            width: frame.width,
            height: frame.height,
        };
        let (w, h) = (f64::from(width), f64::from(height));
        scene.push_clip_layer(Fill::NonZero, Affine::IDENTITY, &Rect::new(0.0, 0.0, w, h));
        let brush = ImageBrush { image: &image, sampler: Default::default() };
        scene.draw_image(brush, cover(f64::from(frame.width), f64::from(frame.height), w, h));
        scene.pop_layer();
        scene
    }
}

// ---------------------------------------------------------------------------------------------
// Feed URL template
// ---------------------------------------------------------------------------------------------

/// Where the native player reads a drone's video.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoConfig {
    /// `{host}` (core host), `{id}`, `{station}` (left/center/right), `{path}` (MediaMTX path
    /// from the WHEP URL, else `drone{id}`), `{whep}` (the announced WHEP URL).
    pub template: String,
    pub ffmpeg: String,
    pub host: String,
}

pub const DEFAULT_TEMPLATE: &str = "srt://{host}:10890?streamid=read:{path}";

/// The MediaMTX path inside a WHEP URL (`/live/drone1/whep` -> `drone1`).
pub fn whep_path(whep: &str) -> Option<String> {
    let start = whep.find("/live/")? + "/live/".len();
    let rest = &whep[start..];
    let path = rest.strip_suffix("/whep").or_else(|| rest.split_once("/whep").map(|(p, _)| p))?;
    (!path.is_empty()).then(|| path.to_string())
}

pub fn station_word(s: Station) -> &'static str {
    match s {
        Station::Left => "left",
        Station::Center => "center",
        Station::Right => "right",
    }
}

/// The player input for a feed.
pub fn feed_url(cfg: &VideoConfig, feed: &Feed) -> String {
    let path = whep_path(&feed.whep_url).unwrap_or_else(|| format!("drone{}", feed.drone_id));
    cfg.template
        .replace("{host}", &cfg.host)
        .replace("{id}", &feed.drone_id.to_string())
        .replace("{station}", station_word(feed.station))
        .replace("{path}", &path)
        .replace("{whep}", &feed.whep_url)
}

/// A `camera_feed` as the panel files it.
#[derive(Debug, Clone, PartialEq)]
pub struct Feed {
    pub drone_id: u32,
    pub station: Station,
    pub whep_url: String,
}

// ---------------------------------------------------------------------------------------------
// Native player
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum PlayerState {
    Starting,
    Playing,
    Failed(String),
}

#[cfg(not(target_arch = "wasm32"))]
pub struct Player {
    child: Option<std::process::Child>,
    state: Arc<Mutex<PlayerState>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Player {
    /// Starts `ffmpeg` reading `url` into raw RGBA frames for `slot`.
    pub fn start(ffmpeg: &str, url: &str, slot: FrameSlot) -> Player {
        use std::io::Read as _;
        use std::process::{Command, Stdio};
        let state = Arc::new(Mutex::new(PlayerState::Starting));
        let filter = format!("scale={FRAME_W}:{FRAME_H}:force_original_aspect_ratio=increase,crop={FRAME_W}:{FRAME_H}");
        let spawned = Command::new(ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-fflags", "nobuffer", "-flags", "low_delay", "-i", url, "-an", "-vf", filter.as_str(), "-pix_fmt", "rgba", "-f", "rawvideo", "-"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn();
        let mut child = match spawned {
            Ok(c) => c,
            Err(e) => {
                skylens_viewer::log(&format!("[video] cannot start {ffmpeg}: {e}"));
                *state.lock().expect("fresh mutex") = PlayerState::Failed(ERR_GENERIC.to_string());
                return Player { child: None, state };
            }
        };
        if let Some(mut out) = child.stdout.take() {
            let st = state.clone();
            let _ = std::thread::Builder::new().name("skylens-video".into()).spawn(move || {
                let size = (FRAME_W * FRAME_H * 4) as usize;
                loop {
                    let mut buf = vec![0u8; size];
                    if out.read_exact(&mut buf).is_err() {
                        break;
                    }
                    if let Ok(mut s) = slot.lock() {
                        *s = Some(Frame { width: FRAME_W, height: FRAME_H, rgba: Arc::new(buf) });
                    }
                    if let Ok(mut s) = st.lock() {
                        if *s == PlayerState::Starting {
                            *s = PlayerState::Playing;
                        }
                    }
                }
                if let Ok(mut s) = slot.lock() {
                    *s = None;
                }
                if let Ok(mut s) = st.lock() {
                    *s = PlayerState::Failed(ERR_SOURCE.to_string());
                }
            });
        }
        Player { child: Some(child), state }
    }

    pub fn state(&self) -> PlayerState {
        self.state.lock().map(|s| s.clone()).unwrap_or(PlayerState::Failed(ERR_GENERIC.to_string()))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for Player {
    fn drop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------------------------

/// Telemetry the readout uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fix {
    pub drone_id: u32,
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
}

/// What the panel shows; DOM changes follow `DOM.md` §5.
pub struct VideoPanel {
    canvas: Option<NodeId>,
    video: Option<NodeId>,
    label: Option<NodeId>,
    overlay: Option<NodeId>,
    feeds: HashMap<u32, Feed>,
    selected: Option<u32>,
    station: Station,
    feed: Option<Feed>,
    telemetry: Option<Fix>,
    playback_error: String,
    /// The URL the `<video>` element carries (`src`), when showing live video.
    src: Option<String>,
    pub config: Option<VideoConfig>,
    pub slot: FrameSlot,
    #[cfg(not(target_arch = "wasm32"))]
    player: Option<Player>,
    playing_seen: bool,
    /// A failed live source is tried again from here (a drone says hello before its stream is up).
    retry_at: Option<f64>,
}

/// Wait before reading a failed live source again (ms).
pub const RETRY_MS: f64 = 3000.0;

impl VideoPanel {
    /// Finds the template nodes and mounts the widgets. `config` is None where live video cannot
    /// play (web).
    pub fn mount(dom: &mut Dom<'_>, config: Option<VideoConfig>) -> VideoPanel {
        let canvas = dom.find(skylens_viewer::q(".video-panel__canvas"));
        let video = dom.find(skylens_viewer::q(".video-panel__video"));
        let label = dom.find(skylens_viewer::q(".video-panel__label"));
        let overlay = dom.find(skylens_viewer::q(".video-panel__overlay"));
        let slot: FrameSlot = Arc::new(Mutex::new(None));
        if let Some(c) = canvas {
            dom.mount_widget(c, StandbyWidget::boxed());
        }
        if let (Some(v), Some(_)) = (video, config.as_ref()) {
            dom.mount_widget(v, FrameWidget::boxed(slot.clone()));
        }
        // Freshly built, the legacy overlay was empty until the first frame filled it.
        if let Some(o) = overlay {
            dom.set_text(o, "");
        }
        VideoPanel {
            canvas,
            video,
            label,
            overlay,
            feeds: HashMap::new(),
            selected: None,
            station: Station::Center,
            feed: None,
            telemetry: None,
            playback_error: String::new(),
            src: None,
            config,
            slot,
            #[cfg(not(target_arch = "wasm32"))]
            player: None,
            playing_seen: false,
            retry_at: None,
        }
    }

    pub fn selected(&self) -> Option<u32> {
        self.selected
    }

    /// Points the panel at a drone: retitle, play its feed if one has landed.
    pub fn select(&mut self, dom: &mut Dom<'_>, drone_id: u32, station: Station) {
        self.selected = Some(drone_id);
        self.station = station;
        if let Some(l) = self.label {
            dom.set_text(l, station.cam_label());
        }
        self.telemetry = None;
        match self.feeds.get(&drone_id).cloned() {
            Some(f) => self.show(dom, f),
            None => self.standby(dom),
        }
    }

    /// Files a feed; shows it if its drone is the one selected. None: the link dropped.
    pub fn set_feed(&mut self, dom: &mut Dom<'_>, feed: Option<Feed>) {
        let Some(f) = feed else {
            self.feeds.clear();
            self.standby(dom);
            return;
        };
        self.feeds.insert(f.drone_id, f.clone());
        match self.selected {
            None => self.select(dom, f.drone_id, f.station),
            Some(s) if s == f.drone_id => self.show(dom, f),
            Some(_) => {}
        }
    }

    pub fn set_telemetry(&mut self, fix: Option<Fix>) {
        if let (Some(t), Some(s)) = (fix, self.selected) {
            if t.drone_id != s {
                return;
            }
        }
        self.telemetry = fix;
    }

    fn show(&mut self, dom: &mut Dom<'_>, next: Feed) {
        self.feed = Some(next.clone());
        let Some(cfg) = self.config.clone() else { return };
        let url = feed_url(&cfg, &next);
        if self.src.as_deref() != Some(url.as_str()) {
            self.show_video(dom, &cfg, &url);
        }
    }

    fn standby(&mut self, dom: &mut Dom<'_>) {
        self.feed = None;
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.player = None;
        }
        if let Ok(mut s) = self.slot.lock() {
            *s = None;
        }
        self.src = None;
        self.playing_seen = false;
        self.retry_at = None;
        self.playback_error.clear();
        if let Some(v) = self.video {
            dom.set_class(v, "video-panel__video is-hidden");
            dom.remove_attr(v, "src");
        }
        if let Some(c) = self.canvas {
            dom.set_class(c, "video-panel__canvas");
        }
    }

    fn show_video(&mut self, dom: &mut Dom<'_>, cfg: &VideoConfig, url: &str) {
        self.src = Some(url.to_string());
        self.playback_error.clear();
        self.playing_seen = false;
        self.retry_at = None;
        if let Some(v) = self.video {
            dom.set_class(v, "video-panel__video");
            dom.set_attr(v, "src", url);
            dom.set_attr(v, "loop", "");
        }
        if let Some(c) = self.canvas {
            dom.set_class(c, "video-panel__canvas is-hidden");
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Ok(mut s) = self.slot.lock() {
                *s = None;
            }
            self.player = Some(Player::start(&cfg.ffmpeg, url, self.slot.clone()));
        }
        #[cfg(target_arch = "wasm32")]
        let _ = cfg;
    }

    fn fail(&mut self, dom: &mut Dom<'_>, reason: &str) {
        self.playback_error = reason.to_string();
        if let Some(v) = self.video {
            dom.add_class(v, "is-hidden");
        }
        if let Some(c) = self.canvas {
            dom.remove_class(c, "is-hidden");
        }
    }

    /// Player state and the overlay line, once per poll (the legacy did it every frame).
    pub fn tick(&mut self, dom: &mut Dom<'_>, now_ms: f64) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.retry_at.is_some_and(|t| now_ms >= t) && self.player.is_none() {
                self.retry_at = None;
                if let (Some(cfg), Some(url)) = (self.config.clone(), self.src.clone()) {
                    self.playing_seen = false;
                    self.player = Some(Player::start(&cfg.ffmpeg, &url, self.slot.clone()));
                }
            }
            let state = self.player.as_ref().map(Player::state);
            match state {
                Some(PlayerState::Playing) if !self.playing_seen => {
                    self.playing_seen = true;
                    self.playback_error.clear();
                    if let Some(v) = self.video {
                        dom.remove_class(v, "is-hidden");
                    }
                    if let Some(c) = self.canvas {
                        dom.add_class(c, "is-hidden");
                    }
                }
                Some(PlayerState::Failed(reason)) => {
                    self.player = None;
                    self.retry_at = Some(now_ms + RETRY_MS);
                    if self.playback_error != reason || self.playing_seen {
                        self.playing_seen = false;
                        self.fail(dom, &reason);
                    }
                }
                _ => {}
            }
        }
        let text = if !self.playback_error.is_empty() {
            self.playback_error.clone()
        } else {
            let who = self.station.label();
            let fix = self.telemetry.filter(|t| self.selected.is_none_or(|s| s == t.drone_id)).map(|t| (t.lat, t.lon, t.alt));
            // Without a decoder (web) a filed feed is not on screen, so it does not count.
            let has_feed = self.feed.is_some() && self.config.is_some();
            let _ = now_ms;
            labels::overlay_text(who, fix, has_feed)
        };
        if let Some(o) = self.overlay {
            dom.set_text(o, &text);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsl_matches_css() {
        assert_eq!(hsl(0.0, 100.0, 50.0), [255, 0, 0]);
        assert_eq!(hsl(120.0, 100.0, 50.0), [0, 255, 0]);
        assert_eq!(hsl(240.0, 100.0, 25.0), [0, 0, 128]);
        assert_eq!(hsl(0.0, 0.0, 12.0), [31, 31, 31]);
    }

    #[test]
    fn cover_fills_and_centres() {
        let t = cover(320.0, 200.0, 640.0, 200.0);
        let p = t * Point::new(0.0, 0.0);
        let q = t * Point::new(320.0, 200.0);
        assert!((q.x - p.x - 640.0).abs() < 1e-9);
        assert!(p.y <= 0.0 && q.y >= 200.0);
    }

    #[test]
    fn feed_urls_from_the_template() {
        let cfg = VideoConfig { template: DEFAULT_TEMPLATE.into(), ffmpeg: "ffmpeg".into(), host: "10.0.0.2".into() };
        let f = Feed { drone_id: 2, station: Station::Left, whep_url: "/live/drone2/whep".into() };
        assert_eq!(feed_url(&cfg, &f), "srt://10.0.0.2:10890?streamid=read:drone2");
        let g = Feed { whep_url: String::new(), ..f.clone() };
        assert_eq!(feed_url(&cfg, &g), "srt://10.0.0.2:10890?streamid=read:drone2");
        let cfg2 = VideoConfig { template: "rtsp://{host}/{station}/{id}".into(), ..cfg };
        assert_eq!(feed_url(&cfg2, &f), "rtsp://10.0.0.2/left/2");
        assert_eq!(whep_path("http://x:8889/live/drone/whep"), Some("drone".into()));
        assert_eq!(whep_path("/whep"), None);
    }
}
