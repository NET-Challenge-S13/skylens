//! Keyboard manual control (legacy `drones/manualControl.ts` + `manualLink.ts`): arrows fly the
//! selected drone forward/back and yaw it, Q/E climb. While any stick key is held the tower sends
//! `manual_control` at 10 Hz; on release it sends one neutral frame. Nothing moves locally: the
//! result comes back as telemetry.

use skylens_viewer::Code;
use std::collections::BTreeSet;

/// Stick keys.
pub fn is_move_key(code: Code) -> bool {
    matches!(code, Code::ArrowUp | Code::ArrowDown | Code::ArrowLeft | Code::ArrowRight | Code::KeyQ | Code::KeyE)
}

fn key_index(code: Code) -> u8 {
    match code {
        Code::ArrowUp => 0,
        Code::ArrowDown => 1,
        Code::ArrowLeft => 2,
        Code::ArrowRight => 3,
        Code::KeyQ => 4,
        _ => 5,
    }
}

/// Axes in -1..1.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Axes {
    pub forward: f64,
    pub yaw: f64,
    pub climb: f64,
}

/// What the 10 Hz tick decided.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tick {
    /// Nothing to send.
    Idle,
    /// Send these axes.
    Send(Axes),
}

#[derive(Debug, Default)]
pub struct ManualInput {
    down: BTreeSet<u8>,
    was_active: bool,
    next_send_ms: f64,
}

/// Send period (legacy `1000 / manualSendHz`).
pub const PERIOD_MS: f64 = 1000.0 / skylens_viewer::config::link::MANUAL_SEND_HZ;

impl ManualInput {
    pub fn new() -> ManualInput {
        ManualInput::default()
    }

    pub fn key_down(&mut self, code: Code) {
        if is_move_key(code) {
            self.down.insert(key_index(code));
        }
    }

    pub fn key_up(&mut self, code: Code) {
        if is_move_key(code) {
            self.down.remove(&key_index(code));
        }
    }

    pub fn active(&self) -> bool {
        !self.down.is_empty()
    }

    pub fn axes(&self) -> Axes {
        let on = |c: Code| if self.down.contains(&key_index(c)) { 1.0 } else { 0.0 };
        Axes {
            forward: on(Code::ArrowUp) - on(Code::ArrowDown),
            yaw: on(Code::ArrowRight) - on(Code::ArrowLeft),
            climb: on(Code::KeyE) - on(Code::KeyQ),
        }
    }

    /// Runs the interval timer: returns what to send now (if a period elapsed) and whether the
    /// active flag changed (for `body.is-manual`).
    pub fn tick(&mut self, now_ms: f64) -> (Tick, Option<bool>) {
        if now_ms < self.next_send_ms {
            return (Tick::Idle, None);
        }
        // Like setInterval: the next tick a period after this one (catching up if late).
        self.next_send_ms = if self.next_send_ms == 0.0 || now_ms - self.next_send_ms > PERIOD_MS {
            now_ms + PERIOD_MS
        } else {
            self.next_send_ms + PERIOD_MS
        };
        let active = self.active();
        let send = if active {
            Tick::Send(self.axes())
        } else if self.was_active {
            Tick::Send(Axes::default())
        } else {
            Tick::Idle
        };
        let changed = (active != self.was_active).then_some(active);
        self.was_active = active;
        (send, changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_follow_the_legacy_mapping() {
        let mut m = ManualInput::new();
        m.key_down(Code::ArrowUp);
        m.key_down(Code::ArrowLeft);
        m.key_down(Code::KeyE);
        assert_eq!(m.axes(), Axes { forward: 1.0, yaw: -1.0, climb: 1.0 });
        m.key_down(Code::ArrowDown);
        assert_eq!(m.axes().forward, 0.0);
        m.key_down(Code::KeyA);
        assert!(m.active());
    }

    #[test]
    fn sends_at_ten_hertz_then_one_neutral_frame() {
        let mut m = ManualInput::new();
        assert_eq!(m.tick(1000.0), (Tick::Idle, None));
        m.key_down(Code::ArrowUp);
        assert_eq!(m.tick(1050.0), (Tick::Idle, None));
        let (t, changed) = m.tick(1100.0);
        assert_eq!(t, Tick::Send(Axes { forward: 1.0, yaw: 0.0, climb: 0.0 }));
        assert_eq!(changed, Some(true));
        m.key_up(Code::ArrowUp);
        assert_eq!(m.tick(1200.0), (Tick::Send(Axes::default()), Some(false)));
        assert_eq!(m.tick(1300.0), (Tick::Idle, None));
    }
}
