//! DOM side of the live reconstruction display: waiting caption, monitor, controls and the event
//! timeline (`ui/status.html` ids `live-*`, documented in `ui/DOM.md` §9). Text and markup come
//! from the pure functions in `live.rs`; this file only writes them and routes the clicks.
//!
//! Writes are skipped when nothing changed (the [`Dom`] helpers dedupe), and the timeline rows
//! are rebuilt only when the event window changes, otherwise just their state classes.

use crate::board::Board;
use crate::live::{self, ColorMode, RowState};
use crate::waves::MonitorWave;
use skylens_viewer::delta_cloud::DeltaCloud;
use skylens_viewer::shell::UiInput;
use skylens_viewer::{Dom, NodeId};
use std::cell::Cell;
use std::rc::Rc;

/// What the timeline list was last built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Window {
    start: usize,
    count: usize,
    /// Timeline length and the first event's sequence at build time (a snapshot replaces the
    /// timeline without changing its length).
    len: usize,
    first_seq: u32,
}

pub struct LivePanel {
    calm: Rc<Cell<bool>>,
    window: Option<Window>,
    rows: Vec<NodeId>,
    states: Vec<RowState>,
    applied_shown: usize,
    /// Scroll offset to apply once the new rows are laid out (next poll).
    pending_scroll: Option<f64>,
    canvas_size: Option<(i64, i64, u32, u32)>,
}

impl LivePanel {
    pub fn mount(dom: &mut Dom<'_>) -> LivePanel {
        let calm = Rc::new(Cell::new(false));
        if let Some(canvas) = dom.find("live-monitor-canvas") {
            dom.mount_widget(canvas, MonitorWave::boxed(calm.clone()));
        }
        LivePanel { calm, window: None, rows: Vec::new(), states: Vec::new(), applied_shown: usize::MAX, pending_scroll: None, canvas_size: None }
    }

    // ---- per poll ---------------------------------------------------------------------------

    pub fn render(&mut self, dom: &mut Dom<'_>, board: &Board) {
        if let Some(y) = self.pending_scroll.take() {
            dom.scroll_to("live-events", 0.0, y);
        }
        let cloud = board.store.delta_cloud_ref();
        let stats = cloud.stats();
        let live = &board.live;
        let now = board.time as f32;

        // Waiting caption (the wave itself is painted by the scene).
        dom.toggle_class("live-wait", "is-visible", live.waiting());
        if live.waiting() {
            dom.set_style("live-wait", "opacity", &format!("{:.2}", live.wait_fade));
            dom.set_text("live-wait-sub", &live::wait_sub(stats.positions, stats.positions_total, board.time));
        }

        // Monitor.
        let show = stats.applied > 0;
        dom.toggle_class("live-monitor", "is-visible", show);
        if show {
            let age = stats.last_apply_at.map_or(0.0, |t| now - t);
            let m = live::monitor_text(&stats, age);
            dom.toggle_class("live-monitor", "is-done", m.done);
            dom.toggle_class("live-monitor", "is-flash", now < live.flash_until);
            dom.set_text("live-monitor-label", &m.label);
            dom.set_text("live-monitor-sub", &m.sub);
            dom.set_text("live-monitor-age", &m.age);
            self.calm.set(live::wave_calm(&stats));
            self.size_canvas(dom);
        }

        // Panel.
        let has = !cloud.is_empty();
        dom.toggle_class("live-panel", "is-visible", has);
        if !has {
            self.window = None;
            return;
        }
        dom.toggle_class("live-indicator", "is-on", cloud.following);
        dom.toggle_class("live-go-live", "is-on", !cloud.following);
        dom.set_text("live-play", if cloud.following || live.replay { "일시정지" } else { "재생" });
        for b in dom.find_all(".live-seg__btn[data-speed]") {
            let on = dom.attr(b, "data-speed").and_then(|v| v.parse::<u32>().ok()) == Some(live.speed);
            dom.toggle_class(b, "is-on", on);
        }
        for b in dom.find_all(".live-seg__btn[data-mode]") {
            let on = dom.attr(b, "data-mode").as_deref().and_then(ColorMode::from_key) == Some(live.mode);
            dom.toggle_class(b, "is-on", on);
        }
        dom.set_text("live-size-val", &format!("{:.1}", live.size));
        dom.set_text("live-t", &live::mmss(stats.t_s, stats.final_reached));
        dom.set_text("live-pts", &live::group(u64::from(stats.points)));
        dom.set_text("live-evt", &format!("{}/{}", stats.applied, stats.len));
        let (frac, label) = live::progress(&stats);
        dom.set_text("live-bar-label", &label);
        dom.set_text("live-bar-pct", &format!("{}%", (frac * 100.0).round() as u32));
        dom.set_style("live-bar-fill", "width", &format!("{:.1}%", frac * 100.0));
        let [moved, new, gone] = live::counters(&stats);
        dom.set_text("live-cnt-moved", &moved);
        dom.set_text("live-cnt-new", &new);
        dom.set_text("live-cnt-gone", &gone);

        dom.set_text("live-evhead", if live.events_open { "- EVENTS" } else { "+ EVENTS" });
        dom.toggle_class("live-events", "is-collapsed", !live.events_open);
        if live.events_open {
            self.render_rows(dom, cloud);
        }
    }

    /// Rows of the timeline: rebuilt when the window moves or the timeline changes, else only the
    /// state classes of rows whose state changed.
    fn render_rows(&mut self, dom: &mut Dom<'_>, cloud: &DeltaCloud) {
        let events = cloud.events();
        let applied = cloud.applied();
        let start = live::window_start(events.len(), applied);
        let count = (events.len() - start).min(live::MAX_ROWS);
        let want = Window { start, count, len: events.len(), first_seq: events.first().map_or(0, |e| e.seq) };
        if self.window != Some(want) {
            let mut markup = String::new();
            for i in start..start + count {
                let prev = i.checked_sub(1).map(|p| events[p].t_s);
                markup.push_str(&live::row_markup(i, &events[i], prev, RowState::of(i, applied)));
            }
            dom.set_html("live-events", &markup);
            self.rows = dom.children("live-events");
            self.states = (start..start + count).map(|i| RowState::of(i, applied)).collect();
            self.window = Some(want);
            self.applied_shown = usize::MAX;
        } else {
            for (k, row) in self.rows.iter().enumerate() {
                let st = RowState::of(start + k, applied);
                if self.states.get(k) != Some(&st) {
                    dom.set_class(*row, st.class());
                    if let Some(s) = self.states.get_mut(k) {
                        *s = st;
                    }
                }
            }
        }
        if self.applied_shown != applied {
            self.applied_shown = applied;
            if applied > start {
                let view_h = dom.size_of("live-events").map_or(200.0, |(_, h)| h);
                self.pending_scroll = Some(live::scroll_for_row(applied - 1 - start, view_h));
            }
        }
    }

    /// The monitor canvas takes its host's size times the device pixel ratio, like the minimap.
    fn size_canvas(&mut self, dom: &mut Dom<'_>) {
        let Some(canvas) = dom.find("live-monitor-canvas") else { return };
        let Some((w, h)) = dom.size_of("live-monitor") else { return };
        let (w, h) = (w.round() as i64 - 2, h.round() as i64 - 2);
        let (w, h) = (if w > 0 { w } else { 220 }, if h > 0 { h } else { 140 });
        let scale = dom.doc_ref().viewport().scale_f64();
        let dpr = if scale > 0.0 { scale.min(2.0) } else { 1.0 };
        let (pw, ph) = (((w as f64) * dpr).round().max(1.0) as u32, ((h as f64) * dpr).round().max(1.0) as u32);
        if self.canvas_size == Some((w, h, pw, ph)) {
            return;
        }
        self.canvas_size = Some((w, h, pw, ph));
        dom.set_attr(canvas, "width", &pw.to_string());
        dom.set_attr(canvas, "height", &ph.to_string());
    }

    // ---- clicks -----------------------------------------------------------------------------

    /// Routes a click on one of the live controls. True when it was one of them.
    pub fn input(&mut self, input: &UiInput, board: &mut Board) -> bool {
        if !input.is_click() {
            return false;
        }
        let now = board.time as f32;
        let Board { store, live: state, .. } = board;
        if let Some((_, idx)) = input.closest_data("ev") {
            if let Ok(i) = idx.parse::<usize>() {
                state.replay = false;
                store.delta_cloud().seek(i + 1, now);
            }
            return true;
        }
        if let Some((_, speed)) = input.closest_data("speed") {
            if let Ok(s) = speed.parse::<u32>() {
                state.set_speed(s);
            }
            return true;
        }
        if let Some((_, mode)) = input.closest_data("mode") {
            if let Some(m) = ColorMode::from_key(mode) {
                state.mode = m;
            }
            return true;
        }
        let Some((_, act)) = input.closest_data("act") else { return false };
        let cloud = store.delta_cloud();
        match act {
            "prev" => {
                state.replay = false;
                cloud.step_back(now);
            }
            "next" => {
                state.replay = false;
                cloud.step_forward(now);
            }
            "play" => play_pause(cloud, state, now),
            "live" => {
                state.replay = false;
                cloud.go_live(now);
            }
            "size-up" => state.size_up(),
            "size-down" => state.size_down(),
            "fit" => state.fit_requested = true,
            "events" => state.events_open = !state.events_open,
            _ => return false,
        }
        true
    }
}

/// Pause when following or replaying; otherwise replay from here at the chosen speed, or go
/// live when already at the head.
pub fn play_pause(cloud: &mut DeltaCloud, state: &mut live::LiveState, now: f32) {
    if cloud.following || state.replay {
        cloud.following = false;
        state.replay = false;
        return;
    }
    if cloud.applied() >= cloud.len() {
        cloud.go_live(now);
        return;
    }
    state.replay_t = replay_start(cloud);
    state.replay = true;
}

/// Model time to resume the replay from: the current event, or just before the first.
pub fn replay_start(cloud: &DeltaCloud) -> f64 {
    match cloud.applied().checked_sub(1) {
        Some(i) => cloud.events()[i].t_s,
        None => cloud.events().first().map_or(0.0, |e| e.t_s - 0.01),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skylens_shared::delta::{AddRec, pack_adds};
    use skylens_shared::proto::payload::v1::{DeltaKind, PointsDelta};

    fn delta(seq: u32, t_s: f64) -> PointsDelta {
        PointsDelta {
            seq,
            kind: DeltaKind::Refined as i32,
            zone: seq,
            capacity: 16,
            total_points: seq,
            adds: pack_adds(&[AddRec { id: seq, xyz: [1.0, 2.0, 3.0], rgb: [1, 2, 3], zone: 0 }]),
            t_s,
            ..Default::default()
        }
    }

    fn cloud(n: u32) -> DeltaCloud {
        let mut c = DeltaCloud::new();
        for i in 1..=n {
            c.ingest(delta(i, f64::from(i) * 10.0));
        }
        c.advance(0.0);
        c
    }

    #[test]
    fn play_pause_pauses_replays_and_goes_live() {
        let mut c = cloud(4);
        let mut s = live::LiveState::default();
        // Following: pause.
        play_pause(&mut c, &mut s, 1.0);
        assert!(!c.following && !s.replay);
        // At the head and paused: play goes live.
        play_pause(&mut c, &mut s, 1.0);
        assert!(c.following);
        // Stepped back: play replays from the current event.
        c.step_back(2.0);
        c.step_back(2.0);
        assert_eq!(c.applied(), 2);
        play_pause(&mut c, &mut s, 3.0);
        assert!(s.replay && !c.following);
        assert_eq!(s.replay_t, 20.0);
        // Replaying: pause.
        play_pause(&mut c, &mut s, 3.0);
        assert!(!s.replay);
    }

    #[test]
    fn replay_starts_just_before_the_first_event() {
        let mut c = cloud(3);
        c.seek(0, 1.0);
        assert!((replay_start(&c) - 9.99).abs() < 1e-9);
        assert_eq!(replay_start(&DeltaCloud::new()), 0.0);
    }

    // ---- DOM wiring on the real template ------------------------------------------------------

    use skylens_viewer::shell::{ElementInfo, InputKind, ScreenApp, ScreenDocument};
    use skylens_viewer::{PollCx, UiInput as Input};

    struct Blank;
    impl ScreenApp for Blank {
        fn poll(&mut self, _: &mut Dom<'_>, _: &mut PollCx<'_>) -> bool {
            false
        }
        fn input(&mut self, _: &Input, _: &mut Dom<'_>) -> bool {
            false
        }
    }

    fn click(data: (&str, &str)) -> Input {
        let info = ElementInfo {
            node: NodeId::from_u64(1),
            tag: "button".into(),
            id: None,
            classes: vec![],
            data: vec![(data.0.to_string(), data.1.to_string())],
        };
        Input { kind: InputKind::Click, path: vec![info], key: None, key_text: None, pointer: None, value: None }
    }

    fn board_with(n: u32) -> Board {
        let mut b = Board::new(true);
        for i in 1..=n {
            b.store.delta_cloud().ingest(delta(i, f64::from(i) * 10.0));
        }
        b.store.delta_cloud().advance(0.0);
        b
    }

    fn doc() -> ScreenDocument<Blank> {
        ScreenDocument::new(&crate::app::page(), Blank)
    }

    #[test]
    fn nothing_shows_before_the_first_delta_except_the_waiting_caption() {
        let mut d = doc();
        let board = Board::new(true);
        d.with_dom(|_, dom| {
            let mut p = LivePanel::mount(dom);
            p.render(dom, &board);
            assert!(dom.has_class("live-wait", "is-visible"));
            assert_eq!(dom.text("live-wait-sub").as_deref(), Some("경과 0:00"));
            assert!(!dom.has_class("live-monitor", "is-visible"));
            assert!(!dom.has_class("live-panel", "is-visible"));
        });
    }

    #[test]
    fn deltas_show_the_monitor_the_counters_and_the_timeline() {
        let mut d = doc();
        let mut board = board_with(3);
        d.with_dom(|_, dom| {
            let mut p = LivePanel::mount(dom);
            p.render(dom, &board);
            assert!(dom.has_class("live-monitor", "is-visible"));
            assert!(dom.has_class("live-panel", "is-visible"));
            assert_eq!(dom.text("live-monitor-label").as_deref(), Some("생성 중 · 구역 4"));
            assert_eq!(dom.text("live-monitor-sub").as_deref(), Some("이벤트 3/3"));
            assert_eq!(dom.text("live-evt").as_deref(), Some("3/3"));
            assert_eq!(dom.text("live-bar-label").as_deref(), Some("EVENTS 3/3"));
            assert!(dom.has_class("live-indicator", "is-on"));
            assert!(!dom.has_class("live-go-live", "is-on"));
            assert_eq!(dom.text("live-play").as_deref(), Some("일시정지"));
            let rows = dom.find_all(".live-ev");
            assert_eq!(rows.len(), 3);
            assert!(dom.has_class(rows[2], "is-cur") && dom.has_class(rows[0], "is-done"));

            // A click on a row seeks; the cloud stops following and the rows follow the state.
            assert!(p.input(&click(("ev", "0")), &mut board));
            p.render(dom, &board);
            assert_eq!(board.store.delta_cloud_ref().applied(), 1);
            assert!(!board.store.delta_cloud_ref().following);
            let rows = dom.find_all(".live-ev");
            assert!(dom.has_class(rows[0], "is-cur") && !dom.has_class(rows[1], "is-done"));
            assert!(!dom.has_class("live-indicator", "is-on") && dom.has_class("live-go-live", "is-on"));
            assert_eq!(dom.text("live-play").as_deref(), Some("재생"));

            // Step forward, back, and the LIVE button.
            assert!(p.input(&click(("act", "next")), &mut board));
            assert_eq!(board.store.delta_cloud_ref().applied(), 2);
            assert!(p.input(&click(("act", "prev")), &mut board));
            assert_eq!(board.store.delta_cloud_ref().applied(), 1);
            assert!(p.input(&click(("act", "live")), &mut board));
            assert!(board.store.delta_cloud_ref().following);
            assert_eq!(board.store.delta_cloud_ref().applied(), 3);
        });
    }

    #[test]
    fn controls_write_the_live_state() {
        let mut d = doc();
        let mut board = board_with(1);
        d.with_dom(|_, dom| {
            let mut p = LivePanel::mount(dom);
            assert!(p.input(&click(("mode", "zone")), &mut board));
            assert_eq!(board.live.mode, ColorMode::Zone);
            assert!(p.input(&click(("speed", "60")), &mut board));
            assert_eq!(board.live.speed, 60);
            assert!(p.input(&click(("act", "size-up")), &mut board));
            assert_eq!(board.live.size, 2.5);
            assert!(p.input(&click(("act", "fit")), &mut board));
            assert!(board.live.fit_requested);
            assert!(p.input(&click(("act", "events")), &mut board));
            assert!(!board.live.events_open);
            assert!(!p.input(&click(("act", "nonsense")), &mut board));
            let mut other = click(("x", "y"));
            other.kind = InputKind::PointerDown;
            assert!(!p.input(&other, &mut board));
            p.render(dom, &board);
            assert!(dom.has_class("live-events", "is-collapsed"));
            assert_eq!(dom.text("live-size-val").as_deref(), Some("2.5"));
            let on: Vec<String> = dom.find_all(".live-seg__btn.is-on").into_iter().filter_map(|b| dom.text(b)).collect();
            assert_eq!(on, vec!["60x".to_string(), "구역".to_string()]);
        });
    }

    #[test]
    fn the_final_event_turns_the_monitor_into_the_summary() {
        let mut d = doc();
        let mut board = Board::new(true);
        let c = board.store.delta_cloud();
        c.ingest(delta(1, 10.0));
        c.ingest(PointsDelta { kind: DeltaKind::Final as i32, ..delta(2, 83.4) });
        c.advance(0.0);
        d.with_dom(|_, dom| {
            let mut p = LivePanel::mount(dom);
            p.render(dom, &board);
            assert!(dom.has_class("live-monitor", "is-done"));
            assert_eq!(dom.text("live-monitor-label").as_deref(), Some("생성 완료 · 총 2점 · 1:23"));
            assert_eq!(dom.text("live-monitor-age").unwrap_or_default(), "");
        });
    }
}
