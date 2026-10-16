//! The live reconstruction display (points-delta stream): the state the board keeps for it and
//! the pure text, markup and mapping functions of its panels. DOM writes are in `livepanel.rs`,
//! the animated waves in `waves.rs`, the GPU drive in `scene.rs`.
//!
//! Reference: the cumulus3d three.js viewer (waiting screen, monitor, event timeline, controls).
//! Everything here is a pure function of numbers so it can be tested without a window.
//!
//! # Segment ladder mapping
//!
//! The server panel's ladder has two pips per zone (preview, refined). With deltas the core
//! reports every zone at the refined level (the chunk's level is always REFINED), which says
//! nothing, so the board derives the ladder from the applied delta timeline instead
//! ([`zone_levels`]):
//!
//! - `REFINED`: the zone is shown. The first pip (preview) is lit, the row says "정제 중".
//! - `REPLACE`: the zone was re-densified. Both pips lit (refined), with a brief pulse
//!   ([`PULSE_S`]) on the row.
//! - `FINAL`: every zone is refined.
//!
//! Detections still reveal through the chunk store, which marks a delta's zone as arrived at the
//! refined level, so a detection shows once a delta of its zone has arrived.

use skylens_shared::proto::payload::v1::DeltaKind;
use skylens_viewer::delta_cloud::{DeltaEvent, DeltaStats};
use skylens_viewer::slot_cloud::{MODE_HEIGHT, MODE_RGB, MODE_ZONE};
use std::collections::BTreeMap;

/// Border flash of the monitor after an applied event.
pub const FLASH_S: f32 = 1.2;
/// Ladder row pulse after a REPLACE.
pub const PULSE_S: f32 = 1.2;
/// The waiting screen fades out this long once the first delta is applied.
pub const WAIT_FADE_S: f32 = 0.8;
/// Timeline rows kept in the DOM at most.
pub const MAX_ROWS: usize = 200;
/// Height of one timeline row in CSS pixels (`.live-ev` in `status-panels.css`).
pub const ROW_H: f64 = 22.0;
/// Replay speeds of the speed buttons.
pub const SPEEDS: [u32; 4] = [1, 5, 20, 60];
/// Point size control range and step (reference `size` slider: 0.5..10).
pub const SIZE_MIN: f32 = 0.5;
pub const SIZE_MAX: f32 = 10.0;
pub const SIZE_STEP: f32 = 0.5;
pub const SIZE_DEFAULT: f32 = 2.0;

/// Colour mode of the live cloud.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMode {
    #[default]
    Rgb,
    Zone,
    Height,
}

impl ColorMode {
    pub const ALL: [ColorMode; 3] = [ColorMode::Rgb, ColorMode::Zone, ColorMode::Height];

    /// `SlotParams::mode`.
    pub fn slot_mode(self) -> u32 {
        match self {
            ColorMode::Rgb => MODE_RGB,
            ColorMode::Zone => MODE_ZONE,
            ColorMode::Height => MODE_HEIGHT,
        }
    }

    /// `data-mode` of the button.
    pub fn key(self) -> &'static str {
        match self {
            ColorMode::Rgb => "rgb",
            ColorMode::Zone => "zone",
            ColorMode::Height => "height",
        }
    }

    pub fn from_key(key: &str) -> Option<ColorMode> {
        ColorMode::ALL.into_iter().find(|m| m.key() == key)
    }

    pub fn label(self) -> &'static str {
        match self {
            ColorMode::Rgb => "RGB",
            ColorMode::Zone => "구역",
            ColorMode::Height => "고도",
        }
    }
}

/// What the live display keeps between frames. Lives in the `Board`; the scene advances it per
/// frame, the panels read it and the buttons write it.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveState {
    pub mode: ColorMode,
    /// Point size control value (reference units, see [`size_metres`]).
    pub size: f32,
    /// Replay speed multiplier.
    pub speed: u32,
    /// Replaying the received events at `speed` (the cloud is not following).
    pub replay: bool,
    /// Replay clock: model seconds (event `t_s`).
    pub replay_t: f64,
    /// Waiting screen opacity: 1 until something is shown, then fades out.
    pub wait_fade: f32,
    /// Some delta was applied at least once.
    pub ever_applied: bool,
    /// Board clock until which the monitor border flashes.
    pub flash_until: f32,
    /// Applied count seen last frame (flash only on forward steps).
    pub prev_applied: usize,
    /// Ladder pulse deadlines per zone (board clock).
    pub pulse_until: BTreeMap<u32, f32>,
    /// "화면 맞춤" was pressed; the scene consumes it.
    pub fit_requested: bool,
    /// The event list is open.
    pub events_open: bool,
}

impl Default for LiveState {
    fn default() -> Self {
        LiveState {
            mode: ColorMode::Rgb,
            size: SIZE_DEFAULT,
            speed: 20,
            replay: false,
            replay_t: 0.0,
            wait_fade: 1.0,
            ever_applied: false,
            flash_until: 0.0,
            prev_applied: 0,
            pulse_until: BTreeMap::new(),
            fit_requested: false,
            events_open: true,
        }
    }
}

impl LiveState {
    /// The waiting screen still has something to show (opacity above zero).
    pub fn waiting(&self) -> bool {
        self.wait_fade > 0.0
    }

    /// One frame of the fade: out over [`WAIT_FADE_S`] once `shown` (a delta applied or other
    /// geometry in), otherwise stays at full.
    pub fn tick_wait(&mut self, shown: bool, dt: f32) {
        if shown {
            self.wait_fade = (self.wait_fade - dt / WAIT_FADE_S).max(0.0);
        }
    }

    pub fn size_up(&mut self) {
        self.size = (self.size + SIZE_STEP).min(SIZE_MAX);
    }

    pub fn size_down(&mut self) {
        self.size = (self.size - SIZE_STEP).max(SIZE_MIN);
    }

    pub fn set_speed(&mut self, speed: u32) {
        if SPEEDS.contains(&speed) {
            self.speed = speed;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------------------------

/// `m:ss`; `round` rounds the seconds instead of flooring them (reference `mmss`).
pub fn mmss(sec: f64, round: bool) -> String {
    let s = if sec.is_finite() { sec.max(0.0) } else { 0.0 };
    let s = if round { s.round() } else { s.floor() } as u64;
    format!("{}:{:02}", s / 60, s % 60)
}

/// `1,234,567`.
pub fn group(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `위치 a/b` (or `위치 a` when the total is unknown); empty while nothing was ingested.
pub fn positions_text(positions: u32, total: u32) -> String {
    match (positions, total) {
        (0, 0) => String::new(),
        (p, 0) => format!("위치 {p}"),
        (p, t) => format!("위치 {p}/{t}"),
    }
}

/// Sub line of the waiting screen: progress when known, then `경과 m:ss`.
pub fn wait_sub(positions: u32, total: u32, elapsed_s: f64) -> String {
    let p = positions_text(positions, total);
    if p.is_empty() {
        format!("경과 {}", mmss(elapsed_s, false))
    } else {
        format!("{p}  ·  경과 {}", mmss(elapsed_s, false))
    }
}

/// The monitor panel's three lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonitorText {
    pub label: String,
    pub sub: String,
    pub age: String,
    pub done: bool,
}

/// Monitor text. `age_s` is the time since the last applied event (board clock).
pub fn monitor_text(s: &DeltaStats, age_s: f32) -> MonitorText {
    if s.final_reached {
        let zones = s.front_zone.map_or(0, |z| z + 1);
        let progress = if s.positions_total > 0 { positions_text(s.positions, s.positions_total) } else { format!("구역 {zones}개") };
        return MonitorText {
            label: format!("생성 완료 · 총 {}점 · {}", group(u64::from(s.points)), mmss(s.t_s, true)),
            sub: progress,
            age: String::new(),
            done: true,
        };
    }
    let label = match s.front_zone {
        Some(z) => format!("생성 중 · 구역 {}", z + 1),
        None => "생성 중".to_string(),
    };
    let sub = if s.positions_total > 0 { positions_text(s.positions, s.positions_total) } else { format!("이벤트 {}/{}", s.applied, s.len) };
    MonitorText { label, sub, age: format!("마지막 갱신 {}초 전", age_s.max(0.0).floor() as u64), done: false }
}

/// Progress bar: fraction 0..1 and its label (positions when the stream reports them).
pub fn progress(s: &DeltaStats) -> (f32, String) {
    if s.positions_total > 0 {
        let f = (s.positions as f32 / s.positions_total as f32).clamp(0.0, 1.0);
        (f, format!("POSITIONS {}/{}", s.positions, s.positions_total))
    } else {
        let f = if s.len == 0 { 0.0 } else { (s.applied as f32 / s.len as f32).clamp(0.0, 1.0) };
        (f, format!("EVENTS {}/{}", s.applied, s.len))
    }
}

/// MOVED / NEW / GONE of the current event, formatted.
pub fn counters(s: &DeltaStats) -> [String; 3] {
    [group(u64::from(s.moved)), group(u64::from(s.added)), group(u64::from(s.removed))]
}

/// Point size in metres for the control value `size`, given the extent of the cloud (diagonal of
/// its box, metres). The reference derives pixels from `size * ref / distance` with
/// `ref = 0.75 * diagonal`; for a 900 px tall view that is about `size * 0.00078 * diagonal`.
pub fn size_metres(size: f32, diagonal: f32) -> f32 {
    let d = if diagonal.is_finite() && diagonal > 1.0 { diagonal } else { 60.0 };
    (size * 0.00078 * d).max(0.02)
}

// ---------------------------------------------------------------------------------------------
// Timeline rows
// ---------------------------------------------------------------------------------------------

/// Kind badge text of a row (`SNAP` for the snapshot).
pub fn kind_badge(kind: DeltaKind, snapshot: bool) -> &'static str {
    if snapshot {
        return "SNAP";
    }
    match kind {
        DeltaKind::Refined => "REFINED",
        DeltaKind::Replace => "REPLACE",
        DeltaKind::Final => "FINAL",
        DeltaKind::Unspecified => "DELTA",
    }
}

fn kind_class(kind: DeltaKind, snapshot: bool) -> &'static str {
    if snapshot {
        return "snap";
    }
    match kind {
        DeltaKind::Refined => "refined",
        DeltaKind::Replace => "replace",
        DeltaKind::Final => "final",
        DeltaKind::Unspecified => "other",
    }
}

/// Row state against the applied count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowState {
    /// Applied, not the current one.
    Done,
    /// The last applied event.
    Current,
    /// Not applied yet.
    Future,
}

impl RowState {
    pub fn of(index: usize, applied: usize) -> RowState {
        if index + 1 == applied {
            RowState::Current
        } else if index < applied {
            RowState::Done
        } else {
            RowState::Future
        }
    }

    pub fn class(self) -> &'static str {
        match self {
            RowState::Done => "live-ev is-done",
            RowState::Current => "live-ev is-cur",
            RowState::Future => "live-ev",
        }
    }
}

/// `+Δs` since the previous event (the first row counts from zero).
pub fn delta_text(t_s: f64, prev_t_s: Option<f64>) -> String {
    let gap = t_s - prev_t_s.unwrap_or(0.0);
    format!("+{}s", gap.max(0.0).round() as u64)
}

/// One timeline row as markup. `index` is the event's index in the whole timeline (`data-ev`).
pub fn row_markup(index: usize, e: &DeltaEvent, prev_t_s: Option<f64>, state: RowState) -> String {
    let kind = e.kind;
    format!(
        "<div class=\"{}\" data-ev=\"{index}\"><span class=\"live-ev__dot\"></span><span class=\"live-ev__seq\">{}</span><span class=\"live-ev__z\">Z{}</span><span class=\"live-ev__k live-ev__k--{}\">{}</span><span class=\"live-ev__t\">T+{}</span><span class=\"live-ev__d\">{}</span><span class=\"live-ev__p\">{}</span></div>",
        state.class(),
        e.seq,
        e.zone,
        kind_class(kind, e.snapshot),
        kind_badge(kind, e.snapshot),
        mmss(e.t_s, false),
        delta_text(e.t_s, prev_t_s),
        group(u64::from(e.points)),
    )
}

/// Which events the list shows: at most [`MAX_ROWS`], the current one inside, centred when the
/// timeline is longer. Returns the first event index.
pub fn window_start(len: usize, applied: usize) -> usize {
    if len <= MAX_ROWS {
        return 0;
    }
    let cur = applied.saturating_sub(1);
    cur.saturating_sub(MAX_ROWS / 2).min(len - MAX_ROWS)
}

/// Scroll offset that puts row `row` (index inside the window) at the middle of a list
/// `view_h` tall.
pub fn scroll_for_row(row: usize, view_h: f64) -> f64 {
    (row as f64 * ROW_H - (view_h - ROW_H) / 2.0).max(0.0)
}

// ---------------------------------------------------------------------------------------------
// Segment ladder (see the module comment)
// ---------------------------------------------------------------------------------------------

/// Ladder level (1 shown, 2 refined) of every zone from the applied events, in order.
pub fn zone_levels(applied: impl IntoIterator<Item = (u32, DeltaKind)>) -> BTreeMap<u32, u32> {
    let mut out: BTreeMap<u32, u32> = BTreeMap::new();
    let mut fin = false;
    for (zone, kind) in applied {
        match kind {
            DeltaKind::Refined | DeltaKind::Unspecified => {
                let e = out.entry(zone).or_insert(1);
                *e = (*e).max(1);
            }
            DeltaKind::Replace => {
                out.insert(zone, 2);
            }
            DeltaKind::Final => fin = true,
        }
    }
    if fin {
        for v in out.values_mut() {
            *v = 2;
        }
    }
    out
}

/// Zones whose most recent applied event is a REPLACE at index `i` (for the pulse): the zone of
/// the last applied event when it is a REPLACE.
pub fn pulse_zone(last: Option<(u32, DeltaKind)>) -> Option<u32> {
    last.and_then(|(z, k)| (k == DeltaKind::Replace).then_some(z))
}

/// Calm wave when the stream is final.
pub fn wave_calm(s: &DeltaStats) -> bool {
    s.final_reached
}

#[cfg(test)]
mod tests {
    use super::*;
    use skylens_shared::proto::payload::v1::PointsDelta;
    use std::rc::Rc;

    fn ev(seq: u32, kind: DeltaKind, zone: u32, t_s: f64, points: u32) -> DeltaEvent {
        DeltaEvent {
            seq,
            kind,
            zone,
            t_s,
            points,
            added: 0,
            moved: 0,
            removed: 0,
            snapshot: false,
            delta: Rc::new(PointsDelta::default()),
            undo: None,
        }
    }

    fn stats(applied: usize, len: usize) -> DeltaStats {
        DeltaStats { applied, len, points: 1_234_567, t_s: 83.4, front_zone: Some(2), ..Default::default() }
    }

    #[test]
    fn minutes_and_seconds() {
        assert_eq!(mmss(0.0, false), "0:00");
        assert_eq!(mmss(59.9, false), "0:59");
        assert_eq!(mmss(59.9, true), "1:00");
        assert_eq!(mmss(83.4, true), "1:23");
        assert_eq!(mmss(3725.0, false), "62:05");
        assert_eq!(mmss(-4.0, false), "0:00");
        assert_eq!(mmss(f64::NAN, false), "0:00");
    }

    #[test]
    fn thousands_are_grouped() {
        assert_eq!(group(0), "0");
        assert_eq!(group(999), "999");
        assert_eq!(group(1000), "1,000");
        assert_eq!(group(1_234_567), "1,234,567");
    }

    #[test]
    fn the_waiting_sub_line_adds_progress_when_known() {
        assert_eq!(wait_sub(0, 0, 65.7), "경과 1:05");
        assert_eq!(wait_sub(12, 0, 3.0), "위치 12  ·  경과 0:03");
        assert_eq!(wait_sub(12, 480, 3.0), "위치 12/480  ·  경과 0:03");
    }

    #[test]
    fn monitor_while_running() {
        let s = stats(3, 7);
        let m = monitor_text(&s, 4.9);
        assert_eq!(m.label, "생성 중 · 구역 3");
        assert_eq!(m.sub, "이벤트 3/7");
        assert_eq!(m.age, "마지막 갱신 4초 전");
        assert!(!m.done);
        let m = monitor_text(&DeltaStats { positions: 40, positions_total: 120, ..s }, 0.0);
        assert_eq!(m.sub, "위치 40/120");
        let m = monitor_text(&DeltaStats { front_zone: None, ..stats(1, 1) }, 0.0);
        assert_eq!(m.label, "생성 중");
    }

    #[test]
    fn monitor_when_final() {
        let m = monitor_text(&DeltaStats { final_reached: true, ..stats(7, 7) }, 9.0);
        assert_eq!(m.label, "생성 완료 · 총 1,234,567점 · 1:23");
        assert_eq!(m.sub, "구역 3개");
        assert_eq!(m.age, "");
        assert!(m.done);
        let m = monitor_text(&DeltaStats { final_reached: true, positions: 480, positions_total: 480, ..stats(7, 7) }, 0.0);
        assert_eq!(m.sub, "위치 480/480");
    }

    #[test]
    fn progress_prefers_positions() {
        assert_eq!(progress(&stats(3, 6)), (0.5, "EVENTS 3/6".to_string()));
        assert_eq!(progress(&stats(0, 0)), (0.0, "EVENTS 0/0".to_string()));
        let (f, l) = progress(&DeltaStats { positions: 30, positions_total: 120, ..stats(3, 6) });
        assert_eq!((f, l.as_str()), (0.25, "POSITIONS 30/120"));
    }

    #[test]
    fn counters_are_grouped() {
        let s = DeltaStats { moved: 1200, added: 5, removed: 0, ..Default::default() };
        assert_eq!(counters(&s), ["1,200".to_string(), "5".to_string(), "0".to_string()]);
    }

    #[test]
    fn rows_carry_the_reference_columns() {
        let e = ev(12, DeltaKind::Replace, 3, 83.4, 1_234_567);
        let m = row_markup(11, &e, Some(70.2), RowState::Current);
        assert_eq!(
            m,
            "<div class=\"live-ev is-cur\" data-ev=\"11\"><span class=\"live-ev__dot\"></span><span class=\"live-ev__seq\">12</span><span class=\"live-ev__z\">Z3</span><span class=\"live-ev__k live-ev__k--replace\">REPLACE</span><span class=\"live-ev__t\">T+1:23</span><span class=\"live-ev__d\">+13s</span><span class=\"live-ev__p\">1,234,567</span></div>"
        );
        let mut snap = ev(5, DeltaKind::Refined, 0, 0.0, 10);
        snap.snapshot = true;
        let m = row_markup(0, &snap, None, RowState::Future);
        assert!(m.starts_with("<div class=\"live-ev\" data-ev=\"0\">"));
        assert!(m.contains("live-ev__k--snap\">SNAP<"));
        assert_eq!(kind_badge(DeltaKind::Final, false), "FINAL");
        assert_eq!(kind_badge(DeltaKind::Refined, false), "REFINED");
        assert_eq!(delta_text(5.0, None), "+5s");
        assert_eq!(delta_text(5.0, Some(9.0)), "+0s");
    }

    #[test]
    fn row_states_follow_the_applied_count() {
        assert_eq!(RowState::of(0, 0), RowState::Future);
        assert_eq!(RowState::of(0, 1), RowState::Current);
        assert_eq!(RowState::of(0, 3), RowState::Done);
        assert_eq!(RowState::of(2, 3), RowState::Current);
        assert_eq!(RowState::of(3, 3), RowState::Future);
    }

    #[test]
    fn the_row_window_is_capped_and_holds_the_current_row() {
        assert_eq!(window_start(50, 50), 0);
        assert_eq!(window_start(200, 200), 0);
        // Following the head of a long timeline: the last 200.
        assert_eq!(window_start(1000, 1000), 800);
        // Stepped back into the middle: centred.
        assert_eq!(window_start(1000, 500), 499 - 100);
        // Near the start.
        assert_eq!(window_start(1000, 20), 0);
        for applied in [0usize, 1, 99, 100, 101, 500, 999, 1000] {
            let s = window_start(1000, applied);
            assert!(s + MAX_ROWS <= 1000);
            if applied > 0 {
                assert!((s..s + MAX_ROWS).contains(&(applied - 1)), "{applied}");
            }
        }
        assert_eq!(scroll_for_row(0, 200.0), 0.0);
        assert_eq!(scroll_for_row(50, 220.0), 50.0 * ROW_H - (220.0 - ROW_H) / 2.0);
    }

    #[test]
    fn the_ladder_maps_kinds_to_levels() {
        use DeltaKind::*;
        let l = zone_levels([(0, Refined), (1, Refined)]);
        assert_eq!(l.get(&0), Some(&1));
        assert_eq!(l.get(&1), Some(&1));
        // A REPLACE re-densifies the zone; a later REFINED does not lower it.
        let l = zone_levels([(0, Refined), (0, Replace), (0, Refined), (1, Refined)]);
        assert_eq!(l.get(&0), Some(&2));
        assert_eq!(l.get(&1), Some(&1));
        // FINAL lifts every zone.
        let l = zone_levels([(0, Refined), (1, Refined), (1, Final)]);
        assert_eq!(l.values().copied().collect::<Vec<_>>(), vec![2, 2]);
        assert!(zone_levels([]).is_empty());
        assert_eq!(pulse_zone(Some((4, Replace))), Some(4));
        assert_eq!(pulse_zone(Some((4, Refined))), None);
        assert_eq!(pulse_zone(None), None);
    }

    #[test]
    fn colour_modes_map_to_the_gpu_modes_and_back() {
        assert_eq!(ColorMode::Rgb.slot_mode(), MODE_RGB);
        assert_eq!(ColorMode::Zone.slot_mode(), MODE_ZONE);
        assert_eq!(ColorMode::Height.slot_mode(), MODE_HEIGHT);
        for m in ColorMode::ALL {
            assert_eq!(ColorMode::from_key(m.key()), Some(m));
        }
        assert_eq!(ColorMode::from_key("x"), None);
        assert_eq!(ColorMode::Zone.label(), "구역");
    }

    #[test]
    fn controls_stay_in_range() {
        let mut s = LiveState::default();
        for _ in 0..40 {
            s.size_up();
        }
        assert_eq!(s.size, SIZE_MAX);
        for _ in 0..40 {
            s.size_down();
        }
        assert_eq!(s.size, SIZE_MIN);
        s.set_speed(60);
        assert_eq!(s.speed, 60);
        s.set_speed(7);
        assert_eq!(s.speed, 60);
        assert!(size_metres(2.0, 150.0) > size_metres(1.0, 150.0));
        assert!(size_metres(0.5, f32::NAN) >= 0.02);
    }

    #[test]
    fn the_waiting_screen_fades_out_only_after_something_is_shown() {
        let mut s = LiveState::default();
        s.tick_wait(false, 1.0);
        assert_eq!(s.wait_fade, 1.0);
        assert!(s.waiting());
        s.tick_wait(true, 0.4);
        assert!((s.wait_fade - 0.5).abs() < 1e-6);
        s.tick_wait(true, 1.0);
        assert_eq!(s.wait_fade, 0.0);
        assert!(!s.waiting());
    }
}
