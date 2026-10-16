//! Text, markup and state rules of the board's live-reconstruction panels: the waiting screen, the
//! top-right monitor, the event timeline, the controls bar, and the mapping of the server panel's
//! segment ladder onto the timeline's zones.
//!
//! Everything here is pure: it reads a [`Snapshot`] of the [`DeltaLog`](crate::delta_log::DeltaLog)
//! and answers with strings, markup and [`Cmd`]s. The DOM writes live in `live_ui.rs`.

use crate::delta_log::{Kind, LogEntry};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------------------------
// What the panels read
// ---------------------------------------------------------------------------------------------

/// The log as the panels see it.
#[derive(Debug, Clone, Copy)]
pub struct Snapshot<'a> {
    pub entries: &'a [LogEntry],
    /// Entries `0..applied` are on screen.
    pub applied: usize,
    /// New deltas are applied as they arrive.
    pub live: bool,
}

impl<'a> Snapshot<'a> {
    pub fn current(&self) -> Option<&'a LogEntry> {
        self.applied.checked_sub(1).and_then(|i| self.entries.get(i))
    }

    pub fn is_final(&self) -> bool {
        self.entries.last().is_some_and(|e| e.kind == Kind::Final)
    }
}

/// Where the reconstruction is: nothing yet, growing, or done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// No delta has arrived (or a new run just started): the waiting screen shows.
    Waiting,
    Live,
    /// The entry on screen is the FINAL one.
    Final,
}

pub fn phase(s: &Snapshot<'_>) -> Phase {
    if s.entries.is_empty() {
        Phase::Waiting
    } else if s.current().is_some_and(|e| e.kind == Kind::Final) {
        Phase::Final
    } else {
        Phase::Live
    }
}

// ---------------------------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------------------------

/// `m:ss`. Seconds are floored, or rounded when `round` (the total time of a finished run).
pub fn mmss(sec: f64, round: bool) -> String {
    let s = if sec.is_finite() { sec.max(0.0) } else { 0.0 };
    let s = if round { s.round() } else { s.floor() } as u64;
    format!("{}:{:02}", s / 60, s % 60)
}

/// `T+m:ss` of an entry's model clock (milliseconds).
pub fn t_plus(t_ms: f64) -> String {
    format!("T+{}", mmss(t_ms / 1000.0, false))
}

/// `+Ns`: seconds since the previous entry (since the run start for the first one).
pub fn delta_secs(prev_ms: Option<f64>, t_ms: f64) -> String {
    let gap = (t_ms - prev_ms.unwrap_or(0.0)) / 1000.0;
    format!("+{}s", if gap.is_finite() { gap.max(0.0).round() as u64 } else { 0 })
}

/// `1,234,567`.
pub fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `Z0`, `Z1`...
pub fn zone_label(zone: u32) -> String {
    format!("Z{zone}")
}

/// The badge text of a kind (upper case, as the legend and the rows print it).
pub fn kind_label(kind: Kind) -> &'static str {
    match kind {
        Kind::Preview => "PREVIEW",
        Kind::Refined => "REFINED",
        Kind::Replace => "REPLACE",
        Kind::Final => "FINAL",
    }
}

/// The badge's CSS class suffix: `k-refined`, `k-replace`, `k-final`.
pub fn kind_class(kind: Kind) -> &'static str {
    match kind {
        Kind::Preview => "k-preview",
        Kind::Refined => "k-refined",
        Kind::Replace => "k-replace",
        Kind::Final => "k-final",
    }
}


/// The change counts of one entry: points that moved, were added, were removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counters {
    pub moved: u32,
    pub new: u32,
    pub gone: u32,
}

pub fn counters(e: &LogEntry) -> Counters {
    Counters { moved: e.n_move, new: e.n_add, gone: e.n_remove }
}

/// The three counter chips of the current entry (`MOVED / NEW / GONE`).
pub fn counters_markup(c: &Counters) -> String {
    format!(
        "<span class=\"live-cnt__chip live-cnt__mv\">MOVED<b>{}</b></span><span class=\"live-cnt__chip live-cnt__ap\">NEW<b>{}</b></span><span class=\"live-cnt__chip live-cnt__ds\">GONE<b>{}</b></span>",
        thousands(u64::from(c.moved)),
        thousands(u64::from(c.new)),
        thousands(u64::from(c.gone)),
    )
}

// ---------------------------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------------------------

/// `위치 12/40`, empty without a total.
pub fn positions_text(e: &LogEntry) -> String {
    if e.positions_total > 0 {
        format!("위치 {}/{}", e.positions, e.positions_total)
    } else if e.positions > 0 {
        format!("위치 {}", e.positions)
    } else {
        String::new()
    }
}

/// The progress bar: label and fraction 0..=1. Frame positions when the core reports a total,
/// else applied entries over stored entries.
pub fn progress(s: &Snapshot<'_>) -> (String, f64) {
    if let Some(e) = s.entries.last().filter(|e| e.positions_total > 0) {
        let frac = (f64::from(e.positions) / f64::from(e.positions_total)).clamp(0.0, 1.0);
        return (format!("위치 {}/{}", e.positions, e.positions_total), frac);
    }
    let frac = if s.entries.is_empty() { 0.0 } else { s.applied as f64 / s.entries.len() as f64 };
    (format!("이벤트 {}/{}", s.applied, s.entries.len()), frac.clamp(0.0, 1.0))
}

/// The newest zone among the applied entries (None before the first).
pub fn front_zone(s: &Snapshot<'_>) -> Option<u32> {
    s.entries[..s.applied.min(s.entries.len())].iter().filter(|e| e.kind != Kind::Final).map(|e| e.zone).max()
}

// ---------------------------------------------------------------------------------------------
// Top-right monitor
// ---------------------------------------------------------------------------------------------

/// The monitor's three lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonitorText {
    pub label: String,
    pub sub: String,
    /// Empty once the run is done.
    pub age: String,
    pub done: bool,
}

/// `None` while nothing is applied (the monitor is hidden). `since_update_s`: seconds since the
/// panel last saw the log change.
pub fn monitor(s: &Snapshot<'_>, since_update_s: f64) -> Option<MonitorText> {
    let cur = s.current()?;
    let zones = front_zone(s).map_or(0, |z| z + 1);
    let prog = positions_text(cur);
    if cur.kind == Kind::Final {
        let sub = if prog.is_empty() { format!("구역 {zones}개") } else { prog };
        return Some(MonitorText {
            label: format!("생성 완료 · 총 {}점 · {}", thousands(u64::from(cur.total_points)), mmss(cur.t_ms / 1000.0, true)),
            sub,
            age: String::new(),
            done: true,
        });
    }
    let sub = if prog.is_empty() { format!("이벤트 {}/{}", s.applied, s.entries.len()) } else { prog };
    Some(MonitorText {
        label: format!("생성 중 · 구역 {zones}"),
        sub,
        age: format!("마지막 갱신 {}초 전", since_update_s.max(0.0).floor() as u64),
        done: false,
    })
}

// ---------------------------------------------------------------------------------------------
// Waiting screen
// ---------------------------------------------------------------------------------------------

pub const WAIT_TITLE: &str = "3D 공간 생성 중";
/// Seconds the screen takes to fade out after the first delta.
pub const WAIT_FADE_OUT_S: f64 = 0.8;
/// Seconds it takes to fade back in (a new run).
pub const WAIT_FADE_IN_S: f64 = 0.4;

/// The waiting screen's sub line: the progress the core reported (if any) and the time waited.
pub fn wait_sub(progress: &str, waited_s: f64) -> String {
    if progress.is_empty() { format!("경과 {}", mmss(waited_s, false)) } else { format!("{progress}  ·  경과 {}", mmss(waited_s, false)) }
}

/// One step of the waiting screen's opacity toward 1 (no delta yet) or 0 (a delta arrived).
pub fn fade_step(current: f64, waiting: bool, dt: f64) -> f64 {
    if waiting {
        (current + dt / WAIT_FADE_IN_S).min(1.0)
    } else {
        (current - dt / WAIT_FADE_OUT_S).max(0.0)
    }
}

// ---------------------------------------------------------------------------------------------
// Timeline rows
// ---------------------------------------------------------------------------------------------

/// How a row is drawn relative to the cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowState {
    /// Not applied yet.
    Pending,
    /// Applied, an earlier one.
    Done,
    /// The entry on screen.
    Current,
}

pub fn row_state(index: usize, applied: usize) -> RowState {
    if applied > 0 && index + 1 == applied {
        RowState::Current
    } else if index < applied {
        RowState::Done
    } else {
        RowState::Pending
    }
}

pub fn row_class(state: RowState) -> &'static str {
    match state {
        RowState::Pending => "live-ev",
        RowState::Done => "live-ev is-done",
        RowState::Current => "live-ev is-done is-cur",
    }
}

/// One timeline row; `data-seq` is what a click seeks to.
pub fn row_markup(entries: &[LogEntry], index: usize, applied: usize) -> String {
    let e = &entries[index];
    let prev = index.checked_sub(1).map(|i| entries[i].t_ms);
    format!(
        "<div class=\"{}\" data-seq=\"{}\"><span class=\"live-ev__dot\"></span><span class=\"live-ev__seq\">{}</span><span class=\"live-ev__z\">{}</span><span class=\"live-ev__k {}\">{}</span><span class=\"live-ev__tt\">{}</span><span class=\"live-ev__dl\">{}</span><span class=\"live-ev__p\">{}</span></div>",
        row_class(row_state(index, applied)),
        e.seq,
        e.seq,
        zone_label(e.zone),
        kind_class(e.kind),
        kind_label(e.kind),
        t_plus(e.t_ms),
        delta_secs(prev, e.t_ms),
        thousands(u64::from(e.total_points)),
    )
}

// ---------------------------------------------------------------------------------------------
// Controls and the commands they give the log
// ---------------------------------------------------------------------------------------------

/// What the log must do (the app applies these to the `DeltaLog`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmd {
    Seek(u32),
    StepBack,
    StepForward,
    FollowLive,
    /// Stop following new events and stay on the current one.
    Unfollow,
}

/// Buttons of the timeline header and the controls bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control {
    Prev,
    Next,
    PlayPause,
    /// The `LIVE` button of the follow indicator.
    GoLive,
    Row(u32),
}

/// Replay speeds the speed button cycles through.
pub const SPEEDS: [u32; 4] = [1, 5, 20, 60];

pub fn next_speed(current: u32) -> u32 {
    let i = SPEEDS.iter().position(|s| *s == current).map_or(0, |i| i + 1);
    SPEEDS[i % SPEEDS.len()]
}

/// Point size range of the `-`/`+` buttons (pixels), and the step.
pub const SIZE_MIN: f32 = 0.5;
pub const SIZE_MAX: f32 = 10.0;
pub const SIZE_STEP: f32 = 0.5;

pub fn step_size(current: f32, dir: i32) -> f32 {
    (current + SIZE_STEP * dir as f32).clamp(SIZE_MIN, SIZE_MAX)
}

/// How the point layer is coloured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMode {
    #[default]
    Rgb,
    Zone,
    Height,
}

impl ColorMode {
    pub const ALL: [ColorMode; 3] = [ColorMode::Rgb, ColorMode::Zone, ColorMode::Height];

    pub fn label(self) -> &'static str {
        match self {
            ColorMode::Rgb => "RGB",
            ColorMode::Zone => "구역색",
            ColorMode::Height => "높이색",
        }
    }

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
}

/// What the controls bar sets for the point layer. The point layer reads this each frame
/// (`fit_requested` is taken and cleared by whoever frames the cloud).
#[derive(Debug, Clone, PartialEq)]
pub struct LiveSettings {
    pub point_size: f32,
    pub color_mode: ColorMode,
    /// Replay speed multiplier (model seconds per wall second).
    pub speed: u32,
    pub fit_requested: bool,
    /// Draw the reconstruction mesh under the points (off: points only, like the reference viewer).
    pub show_mesh: bool,
}

impl Default for LiveSettings {
    fn default() -> Self {
        LiveSettings { point_size: 2.0, color_mode: ColorMode::Rgb, speed: 20, fit_requested: false, show_mesh: false }
    }
}

/// The panels' own state: replay play/pause, and the clock that paces a replay.
#[derive(Debug, Clone, Default)]
pub struct Replay {
    /// A replay is running (only meaningful while not following live).
    pub playing: bool,
    /// Model seconds reached by the replay clock.
    clock_s: f64,
}

impl Replay {
    /// The play button shows pause while live (live is playing) or while a replay runs.
    pub fn shows_pause(&self, s: &Snapshot<'_>) -> bool {
        s.live || self.playing
    }

    /// A button or row was clicked. Returns what the log must do.
    pub fn control(&mut self, c: Control, s: &Snapshot<'_>) -> Vec<Cmd> {
        match c {
            Control::Row(seq) => {
                self.playing = false;
                vec![Cmd::Seek(seq)]
            }
            Control::Prev => {
                self.playing = false;
                if s.applied == 0 { vec![] } else { vec![Cmd::StepBack] }
            }
            Control::Next => {
                self.playing = false;
                if s.applied >= s.entries.len() { vec![] } else { vec![Cmd::StepForward] }
            }
            Control::GoLive => {
                self.playing = false;
                vec![Cmd::FollowLive]
            }
            Control::PlayPause => {
                if self.shows_pause(s) {
                    // Pause: stop the replay; a live follow is left by staying on the current entry.
                    self.playing = false;
                    if s.live { vec![Cmd::Unfollow] } else { vec![] }
                } else if s.applied >= s.entries.len() {
                    self.playing = false;
                    vec![Cmd::FollowLive]
                } else {
                    self.playing = true;
                    self.clock_s = s.current().map_or(0.0, |e| e.t_ms / 1000.0);
                    vec![]
                }
            }
        }
    }

    /// Advances the replay clock by `dt` wall seconds at `speed`x and steps the log when the next
    /// entry's time has come. At most one step per call.
    pub fn tick(&mut self, dt: f64, speed: u32, s: &Snapshot<'_>) -> Vec<Cmd> {
        if !self.playing {
            return vec![];
        }
        if s.live || s.applied >= s.entries.len() {
            // The replay caught up (the log went live again): nothing left to pace.
            self.playing = false;
            return vec![];
        }
        self.clock_s += dt.max(0.0) * f64::from(speed);
        if s.entries[s.applied].t_ms / 1000.0 <= self.clock_s { vec![Cmd::StepForward] } else { vec![] }
    }

    /// The log moved by itself (new run, seek elsewhere): drop a stale replay.
    pub fn reset(&mut self) {
        self.playing = false;
        self.clock_s = 0.0;
    }
}

/// The follow indicator: `(class, text)`.
pub fn follow_indicator(s: &Snapshot<'_>) -> (&'static str, &'static str) {
    if s.live { ("live-follow is-live", "LIVE") } else { ("live-follow", "되감기 중 · LIVE로") }
}

/// The play button's glyph.
pub fn play_glyph(shows_pause: bool) -> &'static str {
    if shows_pause { "||" } else { "\u{25B6}" }
}

// ---------------------------------------------------------------------------------------------
// Segment ladder <-> zones
// ---------------------------------------------------------------------------------------------

/// What the timeline knows about one zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZoneStat {
    pub zone: u32,
    /// Stored entries of the zone.
    pub entries: u32,
    /// Kind of the zone's newest entry.
    pub last: Kind,
}

/// Per zone, from every stored entry (the ladder shows what has arrived, not what is applied).
pub fn zone_stats(entries: &[LogEntry]) -> BTreeMap<u32, ZoneStat> {
    let mut m: BTreeMap<u32, ZoneStat> = BTreeMap::new();
    for e in entries {
        let s = m.entry(e.zone).or_insert(ZoneStat { zone: e.zone, entries: 0, last: e.kind });
        s.entries += 1;
        s.last = e.kind;
    }
    m
}

/// Ladder level of a segment: the timeline's deltas are refined point sets, so a segment with a
/// delta is at the top level; otherwise the core's own level stands.
pub fn ladder_level(core_level: u32, zone: Option<&ZoneStat>) -> u32 {
    match zone {
        Some(z) if z.last == Kind::Preview => 1,
        Some(_) => crate::panels::LADDER_LEVELS,
        None => core_level,
    }
}

/// `수준 2 · Z0 REFINED ×3`: the ladder note with the zone and its latest kind.
pub fn ladder_note_zone(level: u32, zone: Option<&ZoneStat>) -> String {
    let base = if level == 0 { crate::panels::ladder_note(0) } else { format!("수준 {level}") };
    match zone {
        None => crate::panels::ladder_note(level),
        Some(z) if z.entries > 1 => format!("{base} · {} {} ×{}", zone_label(z.zone), kind_label(z.last), z.entries),
        Some(z) => format!("{base} · {} {}", zone_label(z.zone), kind_label(z.last)),
    }
}

// ---------------------------------------------------------------------------------------------
// Markup of the whole live UI (appended once to the pane)
// ---------------------------------------------------------------------------------------------

pub fn color_buttons_markup(active: ColorMode) -> String {
    ColorMode::ALL
        .iter()
        .map(|m| {
            format!(
                "<button type=\"button\" class=\"live-btn live-seg{}\" data-live=\"mode\" data-mode=\"{}\">{}</button>",
                if *m == active { " is-on" } else { "" },
                m.key(),
                m.label(),
            )
        })
        .collect()
}

/// Waiting screen, monitor, timeline and controls bar, in this order.
pub fn live_markup(settings: &LiveSettings) -> String {
    format!(
        concat!(
            "<div id=\"live-wait\" class=\"live-wait\"><canvas id=\"live-wait-wave\" class=\"live-wait__wave\"></canvas><div class=\"live-wait__cap\"><div class=\"live-wait__big\">{title}</div><div class=\"live-wait__sub\" id=\"live-wait-sub\"></div></div></div>",
            "<div id=\"live-monitor\" class=\"live-monitor\"><canvas id=\"live-monitor-wave\" class=\"live-monitor__wave\"></canvas><div class=\"live-monitor__label\" id=\"live-mon-label\"></div><div class=\"live-monitor__sub\" id=\"live-mon-sub\"></div><div class=\"live-monitor__age\" id=\"live-mon-age\"></div></div>",
            "<div id=\"live-controls\" class=\"sl-surface sl-surface--panel live-controls\">",
            "<button type=\"button\" class=\"live-btn live-ic\" id=\"live-prev\" data-live=\"prev\" title=\"한 단계 뒤로\">|\u{25C0}</button>",
            "<button type=\"button\" class=\"live-btn live-ic\" id=\"live-play\" data-live=\"play\" title=\"재생 / 일시정지\">||</button>",
            "<button type=\"button\" class=\"live-btn live-ic\" id=\"live-next\" data-live=\"next\" title=\"한 단계 앞으로\">\u{25B6}|</button>",
            "<button type=\"button\" class=\"live-btn\" id=\"live-speed\" data-live=\"speed\" title=\"재생 속도\">{speed}x</button>",
            "<span class=\"live-sep\"></span>",
            "<span class=\"live-lbl\">점 크기</span><button type=\"button\" class=\"live-btn live-ic\" data-live=\"size-\">\u{2212}</button><span class=\"live-val\" id=\"live-size\">{size}</span><button type=\"button\" class=\"live-btn live-ic\" data-live=\"size+\">+</button>",
            "<span class=\"live-sep\"></span>",
            "<span class=\"live-lbl\">색</span>{modes}",
            "<span class=\"live-sep\"></span>",
            "<button type=\"button\" class=\"live-btn\" id=\"live-fit\" data-live=\"fit\">전체 보기</button>",
            "<button type=\"button\" class=\"live-btn live-mesh{mesh_on}\" id=\"live-mesh\" data-live=\"mesh\" title=\"복원 메시 표시\">메시</button>",
            "</div>",
        ),
        title = WAIT_TITLE,
        speed = settings.speed,
        size = size_text(settings.point_size),
        mesh_on = if settings.show_mesh { " is-on" } else { "" },
        modes = color_buttons_markup(settings.color_mode),
    )
}

pub fn size_text(size: f32) -> String {
    format!("{size:.1}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(seq: u32, kind: Kind, zone: u32, t_ms: f64, total: u32) -> LogEntry {
        LogEntry {
            seq,
            run_id: 1,
            kind,
            zone,
            t_ms,
            n_add: 10,
            n_move: 2_000,
            n_remove: 3,
            total_points: total,
            positions: 0,
            positions_total: 0,
        }
    }

    fn log() -> Vec<LogEntry> {
        vec![
            entry(1, Kind::Refined, 0, 12_400.0, 50_000),
            entry(2, Kind::Replace, 0, 37_900.0, 51_000),
            entry(3, Kind::Refined, 1, 95_000.0, 120_000),
            entry(4, Kind::Final, 1, 128_600.0, 1_234_567),
        ]
    }

    fn snap(entries: &[LogEntry], applied: usize, live: bool) -> Snapshot<'_> {
        Snapshot { entries, applied, live }
    }

    #[test]
    fn times_are_floored_but_the_total_rounds() {
        assert_eq!(mmss(0.0, false), "0:00");
        assert_eq!(mmss(59.9, false), "0:59");
        assert_eq!(mmss(59.9, true), "1:00");
        assert_eq!(mmss(125.0, false), "2:05");
        assert_eq!(mmss(-3.0, false), "0:00");
        assert_eq!(mmss(f64::NAN, true), "0:00");
        assert_eq!(t_plus(12_400.0), "T+0:12");
        assert_eq!(t_plus(3_725_000.0), "T+62:05");
    }

    #[test]
    fn gaps_count_from_the_previous_entry() {
        assert_eq!(delta_secs(None, 12_400.0), "+12s");
        assert_eq!(delta_secs(Some(12_400.0), 37_900.0), "+26s");
        assert_eq!(delta_secs(Some(50_000.0), 40_000.0), "+0s");
    }

    #[test]
    fn counts_get_thousands_separators() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_000), "1,000");
        assert_eq!(thousands(1_234_567), "1,234,567");
        let c = counters(&log()[0]);
        assert_eq!(c, Counters { moved: 2_000, new: 10, gone: 3 });
        let m = counters_markup(&c);
        assert!(m.contains("MOVED<b>2,000</b>") && m.contains("NEW<b>10</b>") && m.contains("GONE<b>3</b>"));
    }

    #[test]
    fn rows_carry_seq_zone_kind_time_gap_and_points() {
        let l = log();
        let r = row_markup(&l, 1, 2);
        assert!(r.starts_with("<div class=\"live-ev is-done is-cur\" data-seq=\"2\">"));
        assert!(r.contains(">Z0<") && r.contains("k-replace\">REPLACE<"));
        assert!(r.contains(">T+0:37<") && r.contains(">+26s<") && r.contains(">51,000<"));
        assert!(row_markup(&l, 3, 2).starts_with("<div class=\"live-ev\" data-seq=\"4\">"));
        assert!(row_markup(&l, 3, 4).contains("k-final\">FINAL<"));
        assert_eq!(row_state(0, 0), RowState::Pending);
        assert_eq!(row_state(0, 2), RowState::Done);
        assert_eq!(row_state(1, 2), RowState::Current);
    }

    #[test]
    fn phases_go_waiting_live_final() {
        let l = log();
        assert_eq!(phase(&snap(&[], 0, true)), Phase::Waiting);
        assert_eq!(phase(&snap(&l[..1], 1, true)), Phase::Live);
        assert_eq!(phase(&snap(&l, 3, true)), Phase::Live);
        assert_eq!(phase(&snap(&l, 4, true)), Phase::Final);
        // Stepping back off the final entry makes it live again.
        assert_eq!(phase(&snap(&l, 3, false)), Phase::Live);
        assert!(snap(&l, 1, false).is_final());
    }

    #[test]
    fn the_wait_screen_fades_out_on_the_first_delta_and_back_in_for_a_new_run() {
        let mut v = 1.0;
        for _ in 0..4 {
            v = fade_step(v, false, 0.1);
        }
        assert!((v - 0.5).abs() < 1e-9);
        for _ in 0..20 {
            v = fade_step(v, false, 0.1);
        }
        assert_eq!(v, 0.0);
        v = fade_step(v, true, 0.2);
        assert!((v - 0.5).abs() < 1e-9);
        assert_eq!(fade_step(0.9, true, 1.0), 1.0);
        assert_eq!(wait_sub("", 75.2), "경과 1:15");
        assert_eq!(wait_sub("위치 3/40", 5.0), "위치 3/40  ·  경과 0:05");
    }

    #[test]
    fn the_monitor_reports_zone_progress_and_age_then_the_total() {
        let l = log();
        assert_eq!(monitor(&snap(&l, 0, true), 0.0), None);
        let m = monitor(&snap(&l, 3, true), 7.9).unwrap();
        assert_eq!(m.label, "생성 중 · 구역 2");
        assert_eq!(m.sub, "이벤트 3/4");
        assert_eq!(m.age, "마지막 갱신 7초 전");
        assert!(!m.done);
        let m = monitor(&snap(&l, 4, true), 1.0).unwrap();
        assert_eq!(m.label, "생성 완료 · 총 1,234,567점 · 2:09");
        assert_eq!(m.sub, "구역 2개");
        assert_eq!(m.age, "");
        assert!(m.done);
        // The core's frame progress replaces the event count.
        let mut p = l.clone();
        p[2].positions = 12;
        p[2].positions_total = 40;
        assert_eq!(monitor(&snap(&p, 3, true), 0.0).unwrap().sub, "위치 12/40");
    }

    #[test]
    fn the_progress_bar_prefers_positions_over_events() {
        let l = log();
        assert_eq!(progress(&snap(&l, 2, true)), ("이벤트 2/4".to_string(), 0.5));
        assert_eq!(progress(&snap(&[], 0, true)), ("이벤트 0/0".to_string(), 0.0));
        let mut p = l.clone();
        p[3].positions = 10;
        p[3].positions_total = 40;
        assert_eq!(progress(&snap(&p, 2, true)), ("위치 10/40".to_string(), 0.25));
    }

    #[test]
    fn the_follow_indicator_and_play_glyph_follow_the_log() {
        let l = log();
        assert_eq!(follow_indicator(&snap(&l, 4, true)), ("live-follow is-live", "LIVE"));
        assert_eq!(follow_indicator(&snap(&l, 2, false)).0, "live-follow");
        assert_eq!(play_glyph(true), "||");
        assert_eq!(play_glyph(false), "\u{25B6}");
    }

    #[test]
    fn clicking_a_row_seeks_and_leaves_replay() {
        let l = log();
        let mut r = Replay { playing: true, clock_s: 5.0 };
        assert_eq!(r.control(Control::Row(2), &snap(&l, 4, true)), vec![Cmd::Seek(2)]);
        assert!(!r.playing);
    }

    #[test]
    fn stepping_stops_a_replay_and_respects_the_ends() {
        let l = log();
        let mut r = Replay::default();
        assert_eq!(r.control(Control::Prev, &snap(&l, 0, false)), vec![]);
        assert_eq!(r.control(Control::Prev, &snap(&l, 2, true)), vec![Cmd::StepBack]);
        assert_eq!(r.control(Control::Next, &snap(&l, 4, true)), vec![]);
        assert_eq!(r.control(Control::Next, &snap(&l, 2, false)), vec![Cmd::StepForward]);
        assert_eq!(r.control(Control::GoLive, &snap(&l, 2, false)), vec![Cmd::FollowLive]);
    }

    #[test]
    fn pausing_while_live_unfollows_and_play_resumes_a_replay_then_live() {
        let l = log();
        let mut r = Replay::default();
        // Live and "playing": the button shows pause; pressing it parks the log on the current entry.
        assert!(r.shows_pause(&snap(&l, 3, true)));
        assert_eq!(r.control(Control::PlayPause, &snap(&l, 3, true)), vec![Cmd::Unfollow]);
        // Not live now: the button shows play and starts a replay from the current entry.
        assert!(!r.shows_pause(&snap(&l, 3, false)));
        assert_eq!(r.control(Control::PlayPause, &snap(&l, 3, false)), vec![]);
        assert!(r.playing && r.shows_pause(&snap(&l, 3, false)));
        // Pausing the replay is a no-op for the log.
        assert_eq!(r.control(Control::PlayPause, &snap(&l, 3, false)), vec![]);
        assert!(!r.playing);
        // Play at the end of the log goes live instead.
        assert_eq!(r.control(Control::PlayPause, &snap(&l, 4, false)), vec![Cmd::FollowLive]);
    }

    #[test]
    fn a_replay_steps_when_the_clock_reaches_the_next_entry() {
        let l = log();
        let mut r = Replay::default();
        r.control(Control::PlayPause, &snap(&l, 1, false)); // clock starts at 12.4 s; next is 37.9 s
        assert!(r.playing);
        assert_eq!(r.tick(0.5, 20, &snap(&l, 1, false)), vec![]); // +10 s -> 22.4
        assert_eq!(r.tick(0.5, 20, &snap(&l, 1, false)), vec![]); // 32.4
        assert_eq!(r.tick(0.5, 20, &snap(&l, 1, false)), vec![Cmd::StepForward]); // 42.4 >= 37.9
        // The log reaches the end and is live again: the replay ends by itself.
        assert_eq!(r.tick(0.5, 20, &snap(&l, 4, true)), vec![]);
        assert!(!r.playing);
        // Not playing: never steps.
        assert_eq!(r.tick(10.0, 60, &snap(&l, 1, false)), vec![]);
    }

    #[test]
    fn speed_and_size_buttons_cycle_and_clamp() {
        assert_eq!(next_speed(1), 5);
        assert_eq!(next_speed(20), 60);
        assert_eq!(next_speed(60), 1);
        assert_eq!(next_speed(7), 1);
        assert_eq!(step_size(2.0, 1), 2.5);
        assert_eq!(step_size(0.5, -1), 0.5);
        assert_eq!(step_size(10.0, 1), 10.0);
        assert_eq!(size_text(2.0), "2.0");
        assert_eq!(ColorMode::from_key("zone"), Some(ColorMode::Zone));
        assert_eq!(ColorMode::from_key("x"), None);
        assert_eq!(LiveSettings::default().color_mode, ColorMode::Rgb);
    }

    #[test]
    fn the_ladder_agrees_with_the_timeline_zones() {
        let l = log();
        let z = zone_stats(&l);
        assert_eq!(z[&0], ZoneStat { zone: 0, entries: 2, last: Kind::Replace });
        assert_eq!(z[&1], ZoneStat { zone: 1, entries: 2, last: Kind::Final });
        // A zone with deltas is at the top level whatever the core said.
        assert_eq!(ladder_level(1, z.get(&0)), 2);
        assert_eq!(ladder_level(1, None), 1);
        assert_eq!(ladder_note_zone(2, z.get(&0)), "수준 2 · Z0 REPLACE ×2");
        assert_eq!(ladder_note_zone(1, None), "수준 1 · 미리보기");
        let one = zone_stats(&l[..1]);
        assert_eq!(ladder_note_zone(2, one.get(&0)), "수준 2 · Z0 REFINED");
    }

    #[test]
    fn the_live_markup_has_every_element_the_glue_looks_up() {
        let m = live_markup(&LiveSettings::default());
        for id in [
            "live-wait", "live-wait-wave", "live-wait-sub", "live-monitor", "live-monitor-wave", "live-mon-label", "live-mon-sub",
            "live-mon-age",
            "live-controls", "live-prev", "live-play", "live-next", "live-speed", "live-size", "live-fit",
        ] {
            assert!(m.contains(&format!("id=\"{id}\"")), "{id}");
        }
        assert!(m.contains("3D 공간 생성 중") && m.contains(">20x<") && m.contains("data-mode=\"height\""));
        assert!(m.contains("live-seg is-on\" data-live=\"mode\" data-mode=\"rgb\""));
    }
}
