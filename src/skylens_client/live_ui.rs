//! DOM glue of the live-reconstruction panels (`panels_live.rs` holds every rule): mounts the
//! waiting screen, monitor, timeline and controls bar on the board, keeps them in step with the
//! [`DeltaLog`], and turns clicks into log commands.
//!
//! The waiting wave and the monitor's dots are Blitz canvas widgets ([`WaveWidget`]).

use crate::anyrender::{self, PaintScene as _};
use crate::delta_log::DeltaLog;
use crate::panels_live::{
    self as pl, Cmd, ColorMode, Control, LiveSettings, Phase, Replay, Snapshot,
};
use crate::peniko::kurbo::{Affine, Rect};
use crate::peniko::{Color, Fill};
use blitz_dom::node::ComputedStyles;
use blitz_dom::Widget;
use skylens_viewer::shell::UiInput;
use skylens_viewer::{Dom, q};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

// ---------------------------------------------------------------------------------------------
// Wave widget
// ---------------------------------------------------------------------------------------------

/// What the glue tells a wave widget each frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaveCfg {
    pub opacity: f32,
    /// A flat sheet (the run is done).
    pub calm: bool,
}

/// A grid of dots on a travelling wave, seen from a slowly orbiting camera: the SF-blue
/// "3D 공간 생성 중" field. `cols`/`rows` pick the density.
pub struct WaveWidget {
    cfg: Rc<Cell<WaveCfg>>,
    cols: usize,
    rows: usize,
    /// Dot size in CSS px at unit perspective.
    dot: f64,
    t0: f64,
}

impl WaveWidget {
    pub fn boxed(cfg: Rc<Cell<WaveCfg>>, cols: usize, rows: usize, dot: f64) -> Box<dyn Widget> {
        Box::new(WaveWidget { cfg, cols, rows, dot, t0: skylens_viewer::now_ms() })
    }
}

/// Wave height at a grid position (the viewer's vertex shader).
pub fn wave_height(x: f64, y: f64, t: f64) -> f64 {
    let r = x.hypot(y);
    0.11 * (r * 15.0 - t * 2.4).sin() / (1.0 + r * 2.2)
        + 0.05 * (r * 6.5 - t * 1.2 + 1.3).sin()
        + 0.035 * (x * 6.0 + t * 0.9).sin() * (y * 5.0 - t * 0.7).sin()
}

/// Height to 0..1 colour position.
pub fn wave_k(h: f64) -> f64 {
    (h * 4.5 + 0.45).clamp(0.0, 1.0)
}

/// Deep blue to SkyLens cyan to near white.
pub fn wave_rgb(k: f64) -> [f32; 3] {
    let lo = [0.05, 0.28, 0.85];
    let mid = [0.37, 0.78, 1.0];
    let hi = [0.85, 1.0, 1.0];
    let mix = |a: [f64; 3], b: [f64; 3], f: f64| [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
    let c = if k < 0.6 { mix(lo, mid, k / 0.6) } else { mix(mid, hi, (k - 0.6) / 0.4) };
    [c[0] as f32, c[1] as f32, c[2] as f32]
}

impl WaveWidget {
    fn draw(&self, scene: &mut anyrender::Scene, w: f64, h: f64, scale: f64) {
        let cfg = self.cfg.get();
        if cfg.opacity <= 0.0 {
            return;
        }
        let t = (skylens_viewer::now_ms() - self.t0) / 1000.0;
        let (ca, sa) = ((t * 0.18).cos(), (t * 0.18).sin());
        let (cx, cy) = (w / 2.0, h * 0.64);
        let tf = Affine::scale(scale);
        for i in 0..self.cols {
            for j in 0..self.rows {
                let x = (i as f64 / (self.cols - 1) as f64 - 0.5) * 2.0;
                let y = (j as f64 / (self.rows - 1) as f64 - 0.5) * 2.0;
                let hh = if cfg.calm { 0.0 } else { wave_height(x, y, t) * 1.45 };
                let k = wave_k(hh);
                let (xr, yr) = (x * ca - y * sa, x * sa + y * ca);
                let sc = 1.0 / (1.0 + yr * 0.22);
                let sx = cx + xr * sc * w * 0.47;
                let sy = cy + yr * sc * h * 0.2 - hh * sc * h * 0.62;
                let size = self.dot * sc + 0.3;
                let [r, g, b] = wave_rgb(k);
                let a = (0.35 + 0.65 * k as f32) * cfg.opacity;
                scene.fill(Fill::NonZero, tf, Color::new([r, g, b, a]), None, &Rect::new(sx - size / 2.0, sy - size / 2.0, sx + size / 2.0, sy + size / 2.0));
            }
        }
    }
}

impl Widget for WaveWidget {
    fn requires_redraw(&self) -> bool {
        self.cfg.get().opacity > 0.0
    }

    fn paint(&mut self, _ctx: &mut dyn anyrender::RenderContext, _styles: &ComputedStyles, width: u32, height: u32, scale: f64) -> anyrender::Scene {
        let mut scene = anyrender::Scene::new();
        let scale = if scale > 0.0 { scale } else { 1.0 };
        let (w, h) = (f64::from(width) / scale, f64::from(height) / scale);
        if w >= 1.0 && h >= 1.0 {
            self.draw(&mut scene, w, h, scale);
        }
        scene
    }
}

// ---------------------------------------------------------------------------------------------
// Glue
// ---------------------------------------------------------------------------------------------

const FLASH_MS: f64 = 1200.0;

/// Remembered DOM text, so a frame without change writes nothing.
#[derive(Default)]
struct Shown {
    wait_sub: String,
    mon: Option<pl::MonitorText>,
    play: String,
    size: String,
    speed: u32,
    mode: Option<ColorMode>,
    wait_opacity: Option<i64>,
    timeline_open: Option<bool>,
}

pub struct LivePanels {
    pub settings: Rc<RefCell<LiveSettings>>,
    replay: Replay,
    wait_cfg: Rc<Cell<WaveCfg>>,
    mon_cfg: Rc<Cell<WaveCfg>>,
    wait_fade: f64,
    wait_started_ms: f64,
    last_sig: (usize, usize, Option<u32>),
    last_change_ms: f64,
    flash_until: f64,
    shown: Shown,
    mounted: bool,
}

impl LivePanels {
    /// `settings` is shared with the board, whose point layers read it.
    pub fn new(settings: Rc<RefCell<LiveSettings>>) -> LivePanels {
        LivePanels {
            settings,
            replay: Replay::default(),
            wait_cfg: Rc::new(Cell::new(WaveCfg { opacity: 1.0, calm: false })),
            mon_cfg: Rc::new(Cell::new(WaveCfg { opacity: 1.0, calm: false })),
            wait_fade: 1.0,
            wait_started_ms: skylens_viewer::now_ms(),
            last_sig: (0, 0, None),
            last_change_ms: skylens_viewer::now_ms(),
            flash_until: 0.0,
            shown: Shown::default(),
            mounted: false,
        }
    }

    /// Appends the panels to the board's pane and mounts the wave widgets.
    pub fn mount(&mut self, dom: &mut Dom<'_>) {
        let markup = pl::live_markup(&self.settings.borrow());
        dom.append_html(q("section.pane"), &markup);
        if let Some(c) = dom.find("live-wait-wave") {
            dom.mount_widget(c, WaveWidget::boxed(self.wait_cfg.clone(), 84, 52, 2.4));
        }
        if let Some(c) = dom.find("live-monitor-wave") {
            dom.mount_widget(c, WaveWidget::boxed(self.mon_cfg.clone(), 50, 30, 1.7));
        }
        self.mounted = true;
    }

    /// Applies panel commands to the log.
    pub fn apply(log: &mut DeltaLog, cmds: &[Cmd]) {
        for c in cmds {
            match *c {
                Cmd::Seek(seq) => log.seek(seq),
                Cmd::StepBack => log.step_back(),
                Cmd::StepForward => log.step_forward(),
                Cmd::FollowLive => log.follow_live(),
                Cmd::Unfollow => log.unfollow(),
            }
        }
    }

    fn snapshot(log: &DeltaLog) -> Snapshot<'_> {
        Snapshot { entries: log.entries(), applied: log.applied(), live: log.is_live() }
    }

    /// A click or key on the board. True when it was one of ours.
    pub fn input(&mut self, input: &UiInput, dom: &mut Dom<'_>, log: &mut DeltaLog) -> bool {
        if !input.is_click() || !self.mounted {
            return false;
        }
        let Some((_, what)) = input.closest_data("live") else { return false };
        let control = match what {
            "prev" => Some(Control::Prev),
            "next" => Some(Control::Next),
            "play" => Some(Control::PlayPause),
            "golive" => Some(Control::GoLive),
            _ => None,
        };
        if let Some(c) = control {
            let cmds = self.replay.control(c, &Self::snapshot(log));
            Self::apply(log, &cmds);
            return true;
        }
        let mut s = self.settings.borrow_mut();
        match what {
            "speed" => s.speed = pl::next_speed(s.speed),
            "size-" => s.point_size = pl::step_size(s.point_size, -1),
            "size+" => s.point_size = pl::step_size(s.point_size, 1),
            "fit" => s.fit_requested = true,
            "mode" => {
                if let Some((_, key)) = input.closest_data("mode")
                    && let Some(m) = ColorMode::from_key(key)
                {
                    s.color_mode = m;
                }
            }
            _ => return false,
        }
        drop(s);
        self.render_settings(dom);
        true
    }

    fn render_settings(&mut self, dom: &mut Dom<'_>) {
        let s = self.settings.borrow().clone();
        let size = pl::size_text(s.point_size);
        if self.shown.size != size {
            dom.set_text("live-size", &size);
            self.shown.size = size;
        }
        if self.shown.speed != s.speed {
            dom.set_text("live-speed", &format!("{}x", s.speed));
            self.shown.speed = s.speed;
        }
        if self.shown.mode != Some(s.color_mode) {
            for b in dom.find_all("[data-live=mode]") {
                let on = dom.attr(b, "data-mode").as_deref() == Some(s.color_mode.key());
                dom.toggle_class(b, "is-on", on);
            }
            self.shown.mode = Some(s.color_mode);
        }
    }

    /// Once per frame. `dt` seconds since the last call, `now` in ms.
    pub fn poll(&mut self, dom: &mut Dom<'_>, log: &mut DeltaLog, now: f64, dt: f64) {
        if !self.mounted {
            return;
        }
        // 1. The log moved by itself: a new run, a growing list.
        let sig = (log.len(), log.applied(), log.run_id());
        if sig != self.last_sig {
            if sig.2 != self.last_sig.2 || sig.0 < self.last_sig.0 {
                self.replay.reset();
            }
            if sig.1 > self.last_sig.1 && sig.2 == self.last_sig.2 {
                self.flash_until = now + FLASH_MS;
            }
            self.last_change_ms = now;
            self.last_sig = sig;
        }
        // 2. A running replay.
        let speed = self.settings.borrow().speed;
        let cmds = self.replay.tick(dt, speed, &Self::snapshot(log));
        Self::apply(log, &cmds);
        let snap = Self::snapshot(log);

        self.render_wait(dom, &snap, now, dt);
        self.render_monitor(dom, &snap, now);
        self.render_timeline(dom, &snap);
        self.render_settings(dom);
    }

    fn render_wait(&mut self, dom: &mut Dom<'_>, snap: &Snapshot<'_>, now: f64, dt: f64) {
        let waiting = pl::phase(snap) == Phase::Waiting;
        if waiting && self.wait_fade <= 0.0 {
            self.wait_started_ms = now;
        }
        self.wait_fade = pl::fade_step(self.wait_fade, waiting, dt);
        self.wait_cfg.set(WaveCfg { opacity: self.wait_fade as f32, calm: false });
        let op = (self.wait_fade * 100.0).round() as i64;
        if self.shown.wait_opacity != Some(op) {
            self.shown.wait_opacity = Some(op);
            if op <= 0 {
                dom.set_style("live-wait", "display", "none");
            } else {
                dom.remove_style("live-wait", "display");
                dom.set_style("live-wait", "opacity", &format!("{:.2}", self.wait_fade));
            }
        }
        if op > 0 {
            let sub = pl::wait_sub("", (now - self.wait_started_ms) / 1000.0);
            if self.shown.wait_sub != sub {
                dom.set_text("live-wait-sub", &sub);
                self.shown.wait_sub = sub;
            }
        }
    }

    fn render_monitor(&mut self, dom: &mut Dom<'_>, snap: &Snapshot<'_>, now: f64) {
        let text = pl::monitor(snap, (now - self.last_change_ms) / 1000.0);
        if self.shown.mon != text {
            match &text {
                Some(m) => {
                    dom.set_style("live-monitor", "display", "block");
                    dom.set_text("live-mon-label", &m.label);
                    dom.set_text("live-mon-sub", &m.sub);
                    dom.set_text("live-mon-age", &m.age);
                    dom.toggle_class("live-monitor", "is-done", m.done);
                    self.mon_cfg.set(WaveCfg { opacity: 1.0, calm: m.done });
                }
                None => {
                    dom.set_style("live-monitor", "display", "none");
                }
            }
            self.shown.mon = text;
        }
        dom.toggle_class("live-monitor", "is-flash", now < self.flash_until);
    }

    /// The controls bar opens with the first event; there is no event list on the board.
    fn render_timeline(&mut self, dom: &mut Dom<'_>, snap: &Snapshot<'_>) {
        let open = !snap.entries.is_empty();
        if self.shown.timeline_open != Some(open) {
            dom.toggle_class("live-controls", "is-open", open);
            self.shown.timeline_open = Some(open);
        }
        let glyph = pl::play_glyph(self.replay.shows_pause(snap));
        if self.shown.play != glyph {
            dom.set_text("live-play", glyph);
            self.shown.play = glyph.to_string();
        }
    }
}

