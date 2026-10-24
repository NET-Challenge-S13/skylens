//! Text, markup and state rules of the board's live-reconstruction panels: the waiting screen, the
//! top-right monitor, the settings panel, and the delay pattern of the server panel.
//!
//! Everything here is pure: it reads a [`Snapshot`] of the [`DeltaLog`](crate::delta_log::DeltaLog)
//! and answers with strings and markup. The DOM writes live in `live_ui.rs` and `app.rs`.

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

/// `Z0`, `Z1`...
pub fn zone_label(zone: u32) -> String {
    format!("Z{zone}")
}

/// The kind as the panels print it (upper case).
pub fn kind_label(kind: Kind) -> &'static str {
    match kind {
        Kind::Preview => "PREVIEW",
        Kind::Refined => "REFINED",
        Kind::Replace => "REPLACE",
        Kind::Final => "FINAL",
    }
}

/// `위치 12/40`, empty without a count.
pub fn positions_text(e: &LogEntry) -> String {
    if e.positions_total > 0 {
        format!("위치 {}/{}", e.positions, e.positions_total)
    } else if e.positions > 0 {
        format!("위치 {}", e.positions)
    } else {
        String::new()
    }
}

/// The newest zone among the applied entries (None before the first).
pub fn front_zone(s: &Snapshot<'_>) -> Option<u32> {
    s.entries[..s.applied.min(s.entries.len())].iter().filter(|e| e.kind != Kind::Final).map(|e| e.zone).max()
}

// ---------------------------------------------------------------------------------------------
// Top-right monitor
// ---------------------------------------------------------------------------------------------

/// The monitor's lines: a heading like the server panel's, then two readouts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonitorText {
    pub label: String,
    pub detail: String,
    /// Empty once the run is done.
    pub age: String,
    pub done: bool,
}

/// `None` while nothing is applied (the monitor is hidden). `since_update_s`: seconds since the
/// panel last saw the log change. No point count: the minimap's header already shows it.
pub fn monitor(s: &Snapshot<'_>, since_update_s: f64) -> Option<MonitorText> {
    let cur = s.current()?;
    let zones = front_zone(s).map_or(0, |z| z + 1);
    let prog = positions_text(cur);
    if cur.kind == Kind::Final {
        let detail = if prog.is_empty() { format!("구역 {zones}개") } else { prog };
        return Some(MonitorText {
            label: "3D 복원 · 생성 완료".to_string(),
            detail,
            age: format!("소요 {}", mmss(cur.t_ms / 1000.0, true)),
            done: true,
        });
    }
    let events = if prog.is_empty() { format!("이벤트 {}/{}", s.applied, s.entries.len()) } else { prog };
    Some(MonitorText {
        label: "3D 복원 · 생성 중".to_string(),
        detail: format!("구역 {zones} · {events}"),
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

/// The waiting screen's sub line: the progress the core reported (if any) and the time waited
/// since a drone connected; before that, that the board waits for one (no clock runs).
pub fn wait_sub(progress: &str, waited_s: Option<f64>) -> String {
    let time = match waited_s {
        Some(s) => format!("경과 {}", mmss(s, false)),
        None => "드론 연결 대기".to_string(),
    };
    if progress.is_empty() { time } else { format!("{progress}  ·  {time}") }
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
// Settings
// ---------------------------------------------------------------------------------------------

/// Point size range of the `-`/`+` buttons (pixels), and the step.
pub const SIZE_MIN: f32 = 1.0;
pub const SIZE_MAX: f32 = 6.0;
pub const SIZE_STEP: f32 = 0.5;

pub fn step_size(current: f32, dir: i32) -> f32 {
    (current + SIZE_STEP * dir as f32).clamp(SIZE_MIN, SIZE_MAX)
}

pub fn size_text(size: f32) -> String {
    format!("{size:.1}")
}

/// What the main view draws of the reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Display {
    /// The deltamesh surface, lit; the live points are kept but not drawn.
    #[default]
    Mesh,
    /// The live points over the mesh, as the stream delivers them.
    Points,
}

impl Display {
    pub const ALL: [Display; 2] = [Display::Mesh, Display::Points];

    pub fn key(self) -> &'static str {
        match self {
            Display::Mesh => "mesh",
            Display::Points => "points",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Display::Mesh => "메시",
            Display::Points => "점",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Display::Mesh => "복원된 표면 위에 점을 함께 표시",
            Display::Points => "수신한 점을 그대로 표시 (점 크기·색 조절)",
        }
    }

    pub fn from_key(key: &str) -> Option<Display> {
        Display::ALL.into_iter().find(|d| d.key() == key)
    }
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

/// What the settings panel sets for the main view. The view reads this each frame
/// (`fit_requested` is taken and cleared by whoever frames the cloud).
#[derive(Debug, Clone, PartialEq)]
pub struct LiveSettings {
    pub display: Display,
    pub point_size: f32,
    pub color_mode: ColorMode,
    pub fit_requested: bool,
    /// Per detection class ([`CONF_CLASSES`] order), the confidence (0..1) below which a
    /// detection is not shown (markers, pins, alerts).
    pub min_conf: [f32; 4],
}

/// The detection classes the settings give a confidence threshold: key (`data-conf-class`),
/// label, default (percent). Persons default high (a false alert sends people); road blocks
/// next, since open ground reads as one easily.
pub const CONF_CLASSES: [(&str, &str, u32); 4] =
    [("person", "사람", 80), ("fire", "화재", 60), ("collapse", "붕괴", 60), ("road_blocked", "도로 막힘", 70)];

/// The thresholds the settings offer (percent).
pub const CONF_STEPS: [u32; 5] = [50, 60, 70, 80, 90];

/// Index in [`CONF_CLASSES`] of a detection category (`None` for an unknown one).
pub fn conf_class(category: skylens_shared::proto::payload::v1::DetectionCategory) -> Option<usize> {
    use skylens_shared::proto::payload::v1::DetectionCategory as C;
    match category {
        C::Person => Some(0),
        C::Fire => Some(1),
        C::Collapse => Some(2),
        C::RoadBlocked => Some(3),
        C::Unspecified => None,
    }
}

impl Default for LiveSettings {
    fn default() -> Self {
        LiveSettings {
            display: Display::Mesh,
            point_size: 2.0,
            color_mode: ColorMode::Rgb,
            fit_requested: false,
            min_conf: CONF_CLASSES.map(|(_, _, d)| d as f32 / 100.0),
        }
    }
}

/// The settings overlay (closed). Point size and colour apply to the point display only, so their
/// group is disabled while the mesh is shown.
pub fn settings_markup(s: &LiveSettings) -> String {
    let displays: String = Display::ALL
        .iter()
        .map(|d| {
            let on = *d == s.display;
            let badge = if *d == Display::default() { "<span class=\"board-settings__badge\">기본값</span>" } else { "" };
            format!(
                "<button type=\"button\" class=\"board-option{}\" data-live=\"display\" data-display=\"{}\" role=\"radio\" aria-checked=\"{}\"><span class=\"board-option__name\">{}{badge}</span><span class=\"board-option__hint\">{}</span></button>",
                if on { " is-active" } else { "" },
                d.key(),
                on,
                d.label(),
                d.hint(),
            )
        })
        .collect();
    let modes: String = ColorMode::ALL
        .iter()
        .map(|m| {
            format!(
                "<button type=\"button\" class=\"board-seg{}\" data-live=\"mode\" data-mode=\"{}\">{}</button>",
                if *m == s.color_mode { " is-on" } else { "" },
                m.key(),
                m.label(),
            )
        })
        .collect();
    format!(
        concat!(
            "<div id=\"board-settings\" class=\"board-settings-overlay is-hidden\" data-live=\"settings-backdrop\">",
            "<div class=\"sl-surface sl-surface--modal board-settings\">",
            "<button type=\"button\" class=\"board-settings__close\" data-live=\"settings-close\" aria-label=\"설정 닫기\">×</button>",
            "<div class=\"board-settings__title\">설정 · Display</div>",
            "<div class=\"board-settings__section\">메인 화면 표시</div>",
            "<div class=\"board-settings__options\" role=\"radiogroup\">{displays}</div>",
            "<div class=\"board-settings__group{points_off}\" id=\"board-settings-points\">",
            "<div class=\"board-settings__section\">점 표시</div>",
            "<div class=\"board-settings__row\"><span class=\"board-settings__lbl\">점 크기</span>",
            "<button type=\"button\" class=\"board-seg board-seg--ic\" data-live=\"size-\">\u{2212}</button>",
            "<span class=\"board-settings__val\" id=\"board-size\">{size}</span>",
            "<button type=\"button\" class=\"board-seg board-seg--ic\" data-live=\"size+\">+</button></div>",
            "<div class=\"board-settings__row\"><span class=\"board-settings__lbl\">색</span>{modes}</div>",
            "</div>",
            "<div class=\"board-settings__section\">탐지</div>",
            "<div class=\"board-settings__row\"><button type=\"button\" class=\"board-seg\" data-live=\"conf-menu\" id=\"board-conf-toggle\">클래스별 신뢰도 · 펼치기</button>",
            "<span class=\"board-settings__note\">이 값보다 확신도가 낮은 탐지는 표시하지 않습니다</span></div>",
            "<div class=\"board-settings__sub is-off\" id=\"board-conf-menu\">{confs}</div>",
            "<div class=\"board-settings__section\">카메라</div>",
            "<div class=\"board-settings__row\"><button type=\"button\" class=\"board-seg\" data-live=\"fit\">전체 보기</button>",
            "<span class=\"board-settings__note\">복원된 범위 전체가 보이게 맞춥니다</span></div>",
            "</div></div>",
        ),
        displays = displays,
        points_off = if s.display == Display::Points { "" } else { " is-off" },
        size = size_text(s.point_size),
        modes = modes,
        confs = conf_rows(&s.min_conf),
    )
}

/// One row per class of threshold buttons (`data-live="conf"`, `data-conf-class`,
/// `data-conf`), the current one on.
pub fn conf_rows(min_conf: &[f32; 4]) -> String {
    CONF_CLASSES
        .iter()
        .zip(min_conf)
        .map(|((key, label, _), v)| {
            let buttons: String = CONF_STEPS
                .iter()
                .map(|p| {
                    format!(
                        "<button type=\"button\" class=\"board-seg{}\" data-live=\"conf\" data-conf-class=\"{key}\" data-conf=\"{p}\">{p}%</button>",
                        if conf_step(*v) == *p { " is-on" } else { "" },
                    )
                })
                .collect();
            format!("<div class=\"board-settings__row\"><span class=\"board-settings__lbl\">{label}</span>{buttons}</div>")
        })
        .collect()
}

/// The step (percent) a threshold shows as.
pub fn conf_step(min_conf: f32) -> u32 {
    (min_conf * 100.0).round() as u32
}

/// A mouse with one part lit, for the controls guide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MousePart {
    Left,
    Right,
    Wheel,
}

/// Inline SVG of a mouse (14 x 20 CSS px) with `part` filled in the board ink.
fn mouse_svg(part: MousePart) -> String {
    let lit = match part {
        MousePart::Left => "<path d=\"M7 1.6 A5.4 5.4 0 0 0 1.6 7 L1.6 8 L7 8 Z\" fill=\"#d9cba8\"/>",
        MousePart::Right => "<path d=\"M7 1.6 A5.4 5.4 0 0 1 12.4 7 L12.4 8 L7 8 Z\" fill=\"#d9cba8\"/>",
        MousePart::Wheel => "<rect x=\"5.9\" y=\"3\" width=\"2.2\" height=\"4\" rx=\"1.1\" fill=\"#d9cba8\"/>",
    };
    format!(
        "<svg class=\"board-help__mouse\" width=\"14\" height=\"20\" viewBox=\"0 0 14 20\">{lit}         <rect x=\"1\" y=\"1\" width=\"12\" height=\"18\" rx=\"6\" fill=\"none\" stroke=\"#93a5c2\" stroke-width=\"1.2\"/>         <line x1=\"1\" y1=\"8\" x2=\"13\" y2=\"8\" stroke=\"#93a5c2\" stroke-width=\"1.2\"/>         <line x1=\"7\" y1=\"1\" x2=\"7\" y2=\"8\" stroke=\"#93a5c2\" stroke-width=\"1.2\"/></svg>"
    )
}

/// A key cap.
fn key(label: &str) -> String {
    format!("<kbd class=\"board-help__key\">{label}</kbd>")
}

/// The guide's sections: title, then (how, what it does) rows. `how` is markup (key caps, the
/// mouse picture, words).
fn help_sections() -> Vec<(&'static str, Vec<(String, &'static str)>)> {
    let with = |icon: String, words: &str| format!("{icon}<span>{words}</span>");
    vec![
        (
            "메인 화면 · 마우스",
            vec![
                (with(mouse_svg(MousePart::Left), "왼쪽 드래그"), "회전"),
                (with(mouse_svg(MousePart::Right), "오른쪽 드래그"), "이동"),
                (format!("{}<span>또는</span>{}<span>+</span>{}<span>드래그</span>", key("Shift"), key("Ctrl"), mouse_svg(MousePart::Left)), "이동"),
                (with(mouse_svg(MousePart::Wheel), "휠 · 가운데 버튼 드래그"), "확대 · 축소"),
            ],
        ),
        (
            "메인 화면 · 터치",
            vec![("<span>한 손가락 드래그</span>".to_string(), "회전"), ("<span>두 손가락 벌리기 · 오므리기</span>".to_string(), "확대 · 축소")],
        ),
        (
            "미니맵",
            vec![
                (with(mouse_svg(MousePart::Left), "드래그"), "회전"),
                (with(mouse_svg(MousePart::Wheel), "휠"), "확대 · 축소"),
                (with(mouse_svg(MousePart::Left), "더블클릭"), "처음 시점으로"),
            ],
        ),
    ]
}

/// The controls guide: a dialog like the settings one, opened from the toolbar's guide button.
pub fn help_markup() -> String {
    let mut sections = String::new();
    for (title, rows) in help_sections() {
        sections.push_str(&format!("<div class=\"board-settings__section\">{title}</div><dl class=\"board-help__list\">"));
        for (how, what) in rows {
            sections.push_str(&format!("<dt>{how}</dt><dd>{what}</dd>"));
        }
        sections.push_str("</dl>");
    }
    format!(
        concat!(
            "<div id=\"board-help\" class=\"board-settings-overlay is-hidden\" data-live=\"help-backdrop\">",
            "<div class=\"sl-surface sl-surface--modal board-settings\">",
            "<button type=\"button\" class=\"board-settings__close\" data-live=\"help-close\" aria-label=\"조작 안내 닫기\">×</button>",
            "<div class=\"board-settings__title\">조작 · Guide</div>",
            "{sections}",
            "<div class=\"board-settings__note board-help__note\">화면을 직접 움직이면 자동 카메라(CAM)가 잠시 멈춥니다. ",
            "설정의 전체 보기를 누르면 복원된 범위 전체로 돌아갑니다.</div>",
            "</div></div>",
        ),
        sections = sections,
    )
}

/// Waiting screen and monitor, in this order (the two parts the glue mounts separately).
pub fn live_markup() -> String {
    format!("{}{}", wait_markup(), monitor_markup())
}

/// The waiting screen, over the whole pane.
pub fn wait_markup() -> String {
    format!(
        "<div id=\"live-wait\" class=\"live-wait\"><canvas id=\"live-wait-wave\" class=\"live-wait__wave\"></canvas><div class=\"live-wait__cap\"><div class=\"live-wait__big\">{title}</div><div class=\"live-wait__sub\" id=\"live-wait-sub\"></div></div></div>",
        title = WAIT_TITLE,
    )
}

/// The monitor panel, which goes at the end of the status row (`#board-row-status`).
pub fn monitor_markup() -> String {
    String::from(
        concat!(
            "<div id=\"live-monitor\" class=\"sl-surface sl-surface--panel live-monitor\">",
            "<div class=\"live-monitor__head\"><div class=\"live-monitor__body\">",
            "<span class=\"live-monitor__label\" id=\"live-mon-label\"></span>",
            "<span class=\"live-monitor__detail\" id=\"live-mon-sub\"></span>",
            "<span class=\"live-monitor__detail\" id=\"live-mon-age\"></span>",
            "</div></div>",
            "<canvas id=\"live-monitor-wave\" class=\"live-monitor__wave\" width=\"270\" height=\"64\"></canvas>",
            "</div>",
        ),
    )
}

// ---------------------------------------------------------------------------------------------
// Delay pattern (server panel)
// ---------------------------------------------------------------------------------------------

/// Refinement rows the pattern shows; later refinements of a zone stay in the top row.
pub const DELAY_ROWS: u32 = 3;
/// Column slots the pattern shows at most. Past that the steps scroll left one per new step and
/// the first slot becomes an ellipsis, so the newest `DELAY_COLS - 1` steps stay in view.
pub const DELAY_COLS: u32 = 6;

/// One cell of the pattern: what zone `zone` got at refinement `row` (1 = first delivery) while
/// the reconstruction front was at step `col`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DelayCell {
    pub row: u32,
    pub col: u32,
    pub zone: u32,
    /// The kind of the delivery (live stream); None for the core's segment levels.
    pub kind: Option<Kind>,
}

/// The whole pattern: every cell, and whether the run is final.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DelayGrid {
    pub cells: Vec<DelayCell>,
    pub done: bool,
}

impl DelayGrid {
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// The step the front is at (the right-most column).
    pub fn last_col(&self) -> u32 {
        self.cells.iter().map(|c| c.col).max().unwrap_or(0)
    }
}

/// The pattern of the live stream: a zone's n-th delivery (preview, then refined, then any
/// replace) sits in row n at step zone + n, so zone k is first seen at step k + 1 and its next
/// stage one step and one row further, above zone k + 1's first delivery. That is the staircase
/// of the delay pattern. (Placing a stage at the step the newest zone had reached stacked a
/// zone's preview and refined copies in one column, since with previews on the refined copy
/// comes before the next zone.)
pub fn delay_grid_live(entries: &[LogEntry]) -> DelayGrid {
    let mut grid = DelayGrid::default();
    let mut deliveries: BTreeMap<u32, u32> = BTreeMap::new();
    for e in entries {
        if e.kind == Kind::Final {
            grid.done = true;
            continue;
        }
        let n = deliveries.entry(e.zone).or_insert(0);
        *n += 1;
        let row = (*n).min(DELAY_ROWS);
        let cell = DelayCell { row, col: e.zone + row, zone: e.zone, kind: Some(e.kind) };
        match grid.cells.iter_mut().find(|c| c.row == cell.row && c.col == cell.col && c.zone == cell.zone) {
            Some(c) => *c = cell,
            None => grid.cells.push(cell),
        }
    }
    grid
}

/// The pattern of the core's segment levels (`(segment index, level number)`, level 0 while
/// processing): segment s reached level l at step s + l, the legacy delay pattern.
pub fn delay_grid_segments(segments: &[(u32, u32)]) -> DelayGrid {
    let mut grid = DelayGrid::default();
    for (s, level) in segments {
        for row in 1..=(*level).min(DELAY_ROWS) {
            grid.cells.push(DelayCell { row, col: s + row, zone: *s, kind: None });
        }
    }
    grid
}

/// Colour slot of a zone in the pattern's palette (CSS `data-hue` 0..7).
pub fn zone_hue(zone: u32) -> u32 {
    zone % 8
}

/// The pattern as markup: a grid of `rows x columns` cells, row 1 at the bottom, the step numbers
/// under it, the newest delivery marked. At most [`DELAY_COLS`] slots: steps 1..=6, or once past
/// that an ellipsis and the last five steps.
pub fn delay_markup(grid: &DelayGrid) -> String {
    if grid.is_empty() {
        return String::new();
    }
    let last = grid.last_col();
    let clipped = last > DELAY_COLS;
    let first = if clipped { last - (DELAY_COLS - 2) } else { 1 };
    let rows = grid.cells.iter().map(|c| c.row).max().unwrap_or(1).max(2);
    let newest = grid.cells.last().copied();
    let mut out = String::from("<div class=\"delay\">");
    out.push_str("<div class=\"delay__axis\">정제</div><div class=\"delay__grid\">");
    for row in (1..=rows).rev() {
        out.push_str(&format!("<div class=\"delay__row\"><span class=\"delay__lv\">{row}</span>"));
        if clipped {
            out.push_str("<span class=\"delay__cell delay__cell--more\">\u{2026}</span>");
        }
        for col in first..=last {
            match grid.cells.iter().rev().find(|c| c.row == row && c.col == col) {
                Some(c) => {
                    let is_new = newest == Some(*c) && !grid.done;
                    let title = match c.kind {
                        Some(k) => format!("{} · {}단계 · {}", zone_label(c.zone), c.row, kind_label(k)),
                        None => format!("구간 {} · {}단계", c.zone + 1, c.row),
                    };
                    out.push_str(&format!(
                        "<span class=\"delay__cell{}\" data-hue=\"{}\" title=\"{}\">{}</span>",
                        if is_new { " is-new" } else { "" },
                        zone_hue(c.zone),
                        title,
                        match c.kind {
                            Some(_) => zone_label(c.zone),
                            None => format!("S{}", c.zone + 1),
                        },
                    ));
                }
                None => out.push_str("<span class=\"delay__cell\"></span>"),
            }
        }
        out.push_str("</div>");
    }
    out.push_str("<div class=\"delay__row delay__steps\"><span class=\"delay__lv\"></span>");
    if clipped {
        out.push_str("<span class=\"delay__step\">\u{2026}</span>");
    }
    for col in first..=last {
        out.push_str(&format!("<span class=\"delay__step\">{col}</span>"));
    }
    out.push_str("</div></div>");
    out.push_str(&format!(
        "<div class=\"delay__caption\">{}</div></div>",
        if grid.done { "진행 단계 · 생성 완료" } else { "진행 단계" }
    ));
    out
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
        assert_eq!(wait_sub("", Some(75.2)), "경과 1:15");
        assert_eq!(wait_sub("위치 3/40", Some(5.0)), "위치 3/40  ·  경과 0:05");
        assert_eq!(wait_sub("", None), "드론 연결 대기");
    }

    #[test]
    fn the_monitor_reports_zone_progress_and_age_then_the_total() {
        let l = log();
        assert_eq!(monitor(&snap(&l, 0, true), 0.0), None);
        let m = monitor(&snap(&l, 3, true), 7.9).unwrap();
        assert_eq!(m.label, "3D 복원 · 생성 중");
        assert_eq!(m.detail, "구역 2 · 이벤트 3/4");
        assert_eq!(m.age, "마지막 갱신 7초 전");
        assert!(!m.done);
        let m = monitor(&snap(&l, 4, true), 1.0).unwrap();
        assert_eq!(m.label, "3D 복원 · 생성 완료");
        assert_eq!(m.detail, "구역 2개");
        assert_eq!(m.age, "소요 2:09");
        assert!(m.done);
        // The core's frame progress replaces the event count.
        let mut p = l.clone();
        p[2].positions = 12;
        p[2].positions_total = 40;
        assert_eq!(monitor(&snap(&p, 3, true), 0.0).unwrap().detail, "구역 2 · 위치 12/40");
    }

    #[test]
    fn size_and_modes_clamp_and_parse() {
        assert_eq!(step_size(2.0, 1), 2.5);
        assert_eq!(step_size(1.0, -1), 1.0);
        assert_eq!(step_size(6.0, 1), 6.0);
        assert_eq!(size_text(2.0), "2.0");
        assert_eq!(ColorMode::from_key("zone"), Some(ColorMode::Zone));
        assert_eq!(ColorMode::from_key("x"), None);
        assert_eq!(Display::from_key("points"), Some(Display::Points));
        assert_eq!(LiveSettings::default().display, Display::Mesh);
        assert_eq!(LiveSettings::default().color_mode, ColorMode::Rgb);
    }

    #[test]
    fn the_settings_panel_disables_point_options_under_the_mesh() {
        let m = settings_markup(&LiveSettings::default());
        assert!(m.contains("board-settings__group is-off"));
        assert!(m.contains("board-option is-active\" data-live=\"display\" data-display=\"mesh\""));
        assert!(m.contains("id=\"board-size\">2.0<") && m.contains("data-mode=\"height\""));
        let p = settings_markup(&LiveSettings { display: Display::Points, ..LiveSettings::default() });
        assert!(p.contains("class=\"board-settings__group\" id=\"board-settings-points\""));
    }

    #[test]
    fn the_live_markup_has_every_element_the_glue_looks_up() {
        let m = live_markup();
        for id in ["live-wait", "live-wait-wave", "live-wait-sub", "live-monitor", "live-monitor-wave", "live-mon-label", "live-mon-sub", "live-mon-age"] {
            assert!(m.contains(&format!("id=\"{id}\"")), "{id}");
        }
        assert!(m.contains("3D 공간 생성 중") && m.contains("sl-surface--panel live-monitor"));
    }

    #[test]
    fn each_class_has_its_threshold_row_with_its_default_on() {
        use skylens_shared::proto::payload::v1::DetectionCategory as C;
        let s = LiveSettings::default();
        assert_eq!(s.min_conf.map(conf_step), [80, 60, 60, 70]);
        let m = settings_markup(&s);
        for (key, label, d) in CONF_CLASSES {
            assert!(m.contains(&format!("is-on\" data-live=\"conf\" data-conf-class=\"{key}\" data-conf=\"{d}\"")), "{key}");
            assert!(m.contains(label));
        }
        assert!(m.contains("id=\"board-conf-menu\""));
        assert_eq!(conf_class(C::RoadBlocked), Some(3));
        assert_eq!(conf_class(C::Unspecified), None);
    }

    #[test]
    fn the_guide_lists_mouse_touch_and_minimap_controls() {
        let m = help_markup();
        assert!(m.contains("id=\"board-help\"") && m.contains("data-live=\"help-close\"") && m.contains("data-live=\"help-backdrop\""));
        for text in ["메인 화면 · 마우스", "메인 화면 · 터치", "미니맵", ">Shift</kbd>", ">Ctrl</kbd>", "<dd>이동</dd>", "더블클릭", "<svg class=\"board-help__mouse\""] {
            assert!(m.contains(text), "{text}");
        }
    }

    #[test]
    fn live_deliveries_climb_one_row_per_refinement_on_a_staircase() {
        // Previews on: Z0's preview, then its refined copy before Z1 arrives. The refined copy
        // still sits at step 2, above Z1's preview, not on Z0's own column.
        let l = vec![
            entry(1, Kind::Preview, 0, 0.0, 1),
            entry(2, Kind::Refined, 0, 0.0, 1),
            entry(3, Kind::Preview, 1, 0.0, 2),
            entry(4, Kind::Refined, 1, 0.0, 2),
            entry(5, Kind::Preview, 2, 0.0, 3),
        ];
        let g = delay_grid_live(&l);
        let at = |row, col| g.cells.iter().find(|c| c.row == row && c.col == col).map(|c| c.zone);
        assert_eq!(at(1, 1), Some(0));
        assert_eq!(at(1, 2), Some(1));
        assert_eq!(at(2, 2), Some(0));
        assert_eq!(at(1, 3), Some(2));
        assert_eq!(at(2, 3), Some(1));
        assert_eq!(g.last_col(), 3);
        assert!(!g.done);
        let mut f = l.clone();
        f.push(entry(6, Kind::Final, 2, 0.0, 3));
        assert!(delay_grid_live(&f).done);
    }

    #[test]
    fn refinements_past_the_top_row_stay_in_it() {
        let l: Vec<LogEntry> = (1..=5).map(|i| entry(i, Kind::Replace, 0, 0.0, 1)).collect();
        let g = delay_grid_live(&l);
        assert!(g.cells.iter().all(|c| c.row <= DELAY_ROWS && c.col == c.row));
        assert_eq!(g.cells.len(), DELAY_ROWS as usize);
    }

    #[test]
    fn core_segments_follow_the_legacy_delay_pattern() {
        let g = delay_grid_segments(&[(0, 2), (1, 1), (2, 0)]);
        let cells: Vec<(u32, u32, u32)> = g.cells.iter().map(|c| (c.row, c.col, c.zone)).collect();
        assert_eq!(cells, [(1, 1, 0), (2, 2, 0), (1, 2, 1)]);
    }

    #[test]
    fn the_markup_shows_the_last_steps_bottom_row_first_in_reading_order() {
        assert_eq!(delay_markup(&DelayGrid::default()), "");
        let l: Vec<LogEntry> = (0..12).map(|z| entry(z + 1, Kind::Refined, z, 0.0, 1)).collect();
        let m = delay_markup(&delay_grid_live(&l));
        // Six slots: an ellipsis in the first, then steps 8..12.
        assert!(m.contains("<span class=\"delay__step\">\u{2026}</span><span class=\"delay__step\">8</span>"));
        assert!(m.contains("<span class=\"delay__step\">12</span>"));
        assert!(!m.contains("<span class=\"delay__step\">7</span>"));
        assert_eq!(m.matches("class=\"delay__step\"").count(), DELAY_COLS as usize);
        // Up to six steps there is no ellipsis.
        let six = delay_markup(&delay_grid_live(&l[..6]));
        assert!(!six.contains('\u{2026}'));
        assert_eq!(six.matches("class=\"delay__step\"").count(), 6);
        // The newest delivery is marked, its zone keeps its hue.
        assert!(m.contains("delay__cell is-new\" data-hue=\"3\" title=\"Z11 · 1단계 · REFINED\">Z11<"));
        // Two rows at least, the top one first.
        assert!(m.find("delay__lv\">2<").unwrap() < m.find("delay__lv\">1<").unwrap());
    }
}
