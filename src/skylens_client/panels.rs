//! Text and markup of the board's panels, exactly as the legacy TypeScript wrote them
//! (`ui/serverStatus.ts`, `ui/pointToggle.ts`, `status.ts` waiting banner, `pointInset.ts`).
//! Pure functions; the DOM writes live in `app.rs`. See `ui/DOM.md` for every variant.

use skylens_shared::proto::model::v1 as model;

// ---------------------------------------------------------------------------------------------
// Server panel (#server-status)
// ---------------------------------------------------------------------------------------------

/// The panel's state (`data-state` of `.server-status__dot`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelState {
    Waiting,
    Connected,
    Receiving,
}

impl PanelState {
    /// Receiving wins over connected; neither is waiting.
    pub fn of(connected: bool, receiving: bool) -> PanelState {
        if receiving {
            PanelState::Receiving
        } else if connected {
            PanelState::Connected
        } else {
            PanelState::Waiting
        }
    }

    pub fn data_state(self) -> &'static str {
        match self {
            PanelState::Waiting => "waiting",
            PanelState::Connected => "connected",
            PanelState::Receiving => "receiving",
        }
    }

}

/// First `.server-status__detail`: `청크 N · 탐지 N`, plus `· Nms` when the core reports latency.
pub fn counts_line(chunks: u64, detections: u64, latency_ms: Option<f64>) -> String {
    let mut s = format!("청크 {chunks} · 탐지 {detections}");
    if let Some(ms) = latency_ms.filter(|v| v.is_finite()) {
        s.push_str(&format!(" · {}ms", ms.round() as i64));
    }
    s
}

/// Second `.server-status__detail`: the mission line once data flows, else empty (hidden). The
/// link wording itself is the panel's title now, so it is not repeated here.
pub fn link_line(state: PanelState, mission: Option<(&str, u32)>) -> String {
    match (state, mission) {
        (PanelState::Receiving, Some((message, drones))) => format!("{message} · 드론 {drones}대"),
        _ => String::new(),
    }
}

/// Levels a segment can reach (preview, refined): the ladder's pip count.
pub const LADDER_LEVELS: u32 = 2;

/// Ladder level number of a protocol level (0 while the segment is processing).
pub fn level_number(level: model::Level) -> u32 {
    match level {
        model::Level::Unspecified => 0,
        model::Level::Preview => 1,
        model::Level::Refined => 2,
    }
}

/// `.segment-row__note`. The legacy note read `수준 N · S스텝 · 라벨` from the core's step ladder;
/// the pipeline now reports only preview and refined, so the label names the level.
pub fn ladder_note(level: u32) -> String {
    match level {
        0 => "복원 중".to_string(),
        1 => "수준 1 · 미리보기".to_string(),
        n => format!("수준 {n} · 정제본"),
    }
}

/// `data-refining`: a level is in but the segment is still being refined (CSS appends
/// ` · 정제 중`).
pub fn ladder_refining(level: u32) -> &'static str {
    if level > 0 && level < LADDER_LEVELS { "1" } else { "0" }
}

/// One ladder row, as the legacy `segmentRow` built it.
pub fn ladder_row_markup(index: u32, level: u32) -> String {
    let pips: String = (0..LADDER_LEVELS)
        .map(|i| format!("<span class=\"segment-row__pip\" data-on=\"{}\"></span>", if i < level { "1" } else { "0" }))
        .collect();
    format!(
        "<div class=\"segment-row\"><span class=\"segment-row__name\">구간 {}</span><span class=\"segment-row__bar\">{pips}</span><span class=\"segment-row__note\" data-refining=\"{}\">{}</span></div>",
        index + 1,
        ladder_refining(level),
        ladder_note(level),
    )
}

// ---------------------------------------------------------------------------------------------
// Waiting banner (#board-waiting)
// ---------------------------------------------------------------------------------------------

/// What the banner knows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BannerInput<'a> {
    pub link_up: bool,
    /// The badge wording of the link (`LinkStatus::badge_text`).
    pub detail: &'a str,
    pub has_geometry: bool,
    /// The first geometry is still coming in: 0..100.
    pub preparing: Option<f64>,
    /// The last reconstruction chunk could not be used.
    pub failed: bool,
}

/// The banner text; empty hides it (legacy `mountWaitingBanner`).
pub fn banner_text(b: &BannerInput<'_>) -> String {
    if !b.link_up {
        return if b.has_geometry {
            format!("{} \u{2014} 마지막 수신 상태 표시 중", b.detail)
        } else {
            format!("{} \u{2014} 복원 데이터 없음", b.detail)
        };
    }
    if !b.has_geometry {
        return "코어 연결됨 · 첫 구간 복원 대기 중".to_string();
    }
    if let Some(p) = b.preparing {
        return format!("구간 수신 · 렌더 준비 {}%", p.round() as i64);
    }
    if b.failed {
        return "구간 로드 실패 \u{2014} 다음 수준 대기".to_string();
    }
    String::new()
}

// ---------------------------------------------------------------------------------------------
// Point panel (#point-inset): the minimap, folded and unfolded from its own header
// ---------------------------------------------------------------------------------------------

/// `.point-inset__meta`, the counts beside the panel's title.
pub fn point_meta(on: bool, delivered: usize, refined: usize, points: u64) -> String {
    if delivered == 0 {
        "수신 대기".to_string()
    } else if !on {
        format!("{delivered}구간 수신")
    } else {
        format!("{delivered}구간 · 정밀 {refined} · {}만 점", (points as f64 / 10_000.0).round() as u64)
    }
}

/// The fold button's label for an open or folded panel.
pub fn fold_label(open: bool) -> &'static str {
    if open { "접기" } else { "열기" }
}

/// The point panel's markup (legacy `PointInset` constructor), open or folded. The header holds
/// the title, the counts and the fold button; folded, only the header is left.
pub fn inset_markup(open: bool) -> String {
    format!(
        "<div class=\"{}\" id=\"point-inset\"><div class=\"point-inset__head\"><span class=\"point-inset__title\">미니맵 · 고도</span><span class=\"point-inset__meta\"></span><button type=\"button\" class=\"point-inset__fold\" id=\"point-inset-fold\" aria-expanded=\"{}\">{}</button></div><div class=\"point-inset__view\" title=\"드래그: 회전 · 휠: 확대 · 더블클릭: 처음 시점\"><div class=\"point-inset__empty\">수신 대기</div></div><div class=\"point-inset__legend\"><span class=\"point-inset__tick point-inset__tick--high\"></span><div class=\"point-inset__bar\" style=\"background: {};\"></div><span class=\"point-inset__tick point-inset__tick--low\"></span><span class=\"point-inset__caption\">하위 1%<br>높이 기준</span></div></div>",
        if open { "point-inset is-open" } else { "point-inset" },
        if open { "true" } else { "false" },
        fold_label(open),
        crate::points::legend_gradient(),
    )
}

/// Viewports at most this tall (CSS px) get the side placement of [`inset_placement`].
pub const SHORT_VIEWPORT_PX: f64 = 600.0;

/// Where the point panel goes on a short viewport, as inline `left`/`top`/`width`/`height` (px).
///
/// `status-panels.css` stacks the server panel down from the top-left (`top: 54px`) and the
/// point panel up from the bottom-left (`bottom: 16px`, 250 px tall). On a phone on its side
/// (about 410 px tall) the server panel's ladder ran into the point panel. There the panel moves
/// beside the server panel's column (16 px + 340 px max-width + 16 px gap) and leaves a 232 px
/// column on the right for the detection card in the bottom-right corner. None keeps the
/// stylesheet's placement.
pub fn inset_placement(viewport_w: f64, viewport_h: f64) -> Option<[f64; 4]> {
    if !(viewport_h > 0.0 && viewport_h <= SHORT_VIEWPORT_PX) {
        return None;
    }
    const LEFT: f64 = 372.0;
    const TOP: f64 = 54.0;
    let width = (viewport_w - LEFT - 232.0).clamp(180.0, 380.0);
    let height = (viewport_h - TOP - 16.0).clamp(120.0, 250.0);
    Some([LEFT, TOP, width.round(), height.round()])
}

/// The legend's high and low ticks for a height range (empty without one).
pub fn legend_ticks(range: Option<(f64, f64)>) -> (String, String) {
    match range {
        Some((low, high)) => (format!("+{:.1} m", high - low), "0 m".to_string()),
        None => (String::new(), String::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_panel_matches_the_captures() {
        assert_eq!(PanelState::of(true, false).data_state(), "connected");
        assert_eq!(PanelState::of(true, true).data_state(), "receiving");
        assert_eq!(PanelState::of(false, false).data_state(), "waiting");
        assert_eq!(counts_line(0, 0, Some(0.0)), "청크 0 · 탐지 0 · 0ms");
        assert_eq!(counts_line(17, 1, Some(1.6)), "청크 17 · 탐지 1 · 2ms");
        assert_eq!(counts_line(3, 0, None), "청크 3 · 탐지 0");
        assert_eq!(
            link_line(PanelState::Receiving, Some(("드론 연결됨 \u{2014} 임무 수행 중", 3))),
            "드론 연결됨 \u{2014} 임무 수행 중 · 드론 3대"
        );
        assert_eq!(link_line(PanelState::Connected, Some(("m", 3))), "");
    }

    #[test]
    fn ladder_rows_use_the_legacy_markup() {
        assert_eq!(
            ladder_row_markup(0, 1),
            "<div class=\"segment-row\"><span class=\"segment-row__name\">구간 1</span><span class=\"segment-row__bar\"><span class=\"segment-row__pip\" data-on=\"1\"></span><span class=\"segment-row__pip\" data-on=\"0\"></span></span><span class=\"segment-row__note\" data-refining=\"1\">수준 1 · 미리보기</span></div>"
        );
        assert!(ladder_row_markup(2, 0).contains("data-refining=\"0\">복원 중<"));
        assert!(ladder_row_markup(3, 2).contains("data-refining=\"0\">수준 2 · 정제본<"));
        assert_eq!(level_number(model::Level::Refined), 2);
    }

    #[test]
    fn banner_follows_the_legacy_order() {
        let base = BannerInput { link_up: true, detail: "코어 연결됨 · 데이터 대기", has_geometry: false, preparing: None, failed: false };
        assert_eq!(banner_text(&base), "코어 연결됨 · 첫 구간 복원 대기 중");
        let down = BannerInput { link_up: false, detail: "코어 응답 없음 · 재연결 대기", ..base };
        assert_eq!(banner_text(&down), "코어 응답 없음 · 재연결 대기 \u{2014} 복원 데이터 없음");
        assert_eq!(
            banner_text(&BannerInput { has_geometry: true, ..down }),
            "코어 응답 없음 · 재연결 대기 \u{2014} 마지막 수신 상태 표시 중"
        );
        let live = BannerInput { has_geometry: true, ..base };
        assert_eq!(banner_text(&live), "");
        assert_eq!(banner_text(&BannerInput { preparing: Some(41.6), ..live }), "구간 수신 · 렌더 준비 42%");
        assert_eq!(banner_text(&BannerInput { failed: true, ..live }), "구간 로드 실패 \u{2014} 다음 수준 대기");
    }

    #[test]
    fn point_panel_text() {
        assert_eq!(point_meta(true, 0, 0, 0), "수신 대기");
        assert_eq!(point_meta(false, 3, 1, 50_000), "3구간 수신");
        assert_eq!(point_meta(true, 3, 1, 1_234_567), "3구간 · 정밀 1 · 123만 점");
        assert_eq!(legend_ticks(Some((2.0, 14.46))), ("+12.5 m".to_string(), "0 m".to_string()));
        assert_eq!(legend_ticks(None), (String::new(), String::new()));
        let m = inset_markup(true);
        assert!(m.starts_with("<div class=\"point-inset is-open\" id=\"point-inset\">"));
        assert!(m.contains("하위 1%<br>높이 기준"));
        assert!(m.contains("미니맵 · 고도") && m.contains("더블클릭: 처음 시점"));
        assert!(m.contains("aria-expanded=\"true\">접기</button>"));
        let folded = inset_markup(false);
        assert!(folded.starts_with("<div class=\"point-inset\" id=\"point-inset\">"));
        assert!(folded.contains("aria-expanded=\"false\">열기</button>"));
    }

    #[test]
    fn point_panel_moves_beside_the_server_panel_only_on_short_viewports() {
        // Desktop and portrait keep the stylesheet's bottom-left placement.
        assert_eq!(inset_placement(1920.0, 1080.0), None);
        assert_eq!(inset_placement(411.0, 914.0), None);
        assert_eq!(inset_placement(0.0, 0.0), None);
        // Pixel 7 landscape (2400 x 1080 at 2.625): beside the server column, clear of the right edge.
        assert_eq!(inset_placement(914.0, 411.0), Some([372.0, 54.0, 310.0, 250.0]));
        // Shorter still: the panel shrinks to fit under the top row.
        assert_eq!(inset_placement(1200.0, 300.0), Some([372.0, 54.0, 380.0, 230.0]));
        assert_eq!(inset_placement(600.0, 160.0), Some([372.0, 54.0, 180.0, 120.0]));
    }
}
