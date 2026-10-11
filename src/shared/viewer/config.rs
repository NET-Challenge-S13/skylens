//! Tuning constants of the legacy `shared/viewer/config.ts` that both screens still use, and
//! colour helpers for the pointblitz frame (whose colour target stores sRGB-encoded bytes, so a
//! legacy `0xRRGGBB` maps to `byte / 255` with no linearisation).

/// Camera sync (legacy `CONFIG.camera`).
pub mod camera {
    /// FOCUSING / RETURNING tween duration (s).
    pub const TWEEN_SECONDS: f64 = 1.8;
    /// Distance (m) the status camera holds from a focused detection.
    pub const FOCUS_DISTANCE: f64 = 12.0;
    /// Smoothing factor for the SYNCED follow (0..1 per frame at 60 fps).
    pub const SYNC_LERP: f64 = 0.08;
}

/// Core link (legacy `CONFIG.control`).
pub mod link {
    /// Reconnect backoff bounds (ms).
    pub const RECONNECT_MIN_MS: u64 = 800;
    pub const RECONNECT_MAX_MS: u64 = 8_000;
    /// Manual-control send rate (Hz) while keys are held.
    pub const MANUAL_SEND_HZ: f64 = 10.0;
}

/// Palette (legacy `CONFIG.color`).
pub mod color {
    pub const CONTROL_BG: u32 = 0x0a1128;
    pub const CONTROL_POINT: u32 = 0x3fd0e0;
    pub const CONTROL_POINT_FAR: u32 = 0x2a4a8a;
    pub const DRONE_CORE: u32 = 0x9fe8ff;
    pub const DRONE_TRAIL: u32 = 0x4a90d9;
    pub const STATUS_BG: u32 = 0x0d0f14;
    pub const STATUS_POINT: u32 = 0xc8c2b8;
    pub const MARKER_DANGER: u32 = 0xff4d4d;
    pub const MARKER_PERSON: u32 = 0x39d98a;
}

/// Drone id palette so the tower can distinguish the swarm (legacy `DRONE_TINTS`), indexed by
/// `drone_id - 1`.
pub const DRONE_TINTS: [u32; 3] = [0x9fe8ff, 0xffd27f, 0xc79fff];

/// Tint of a drone id (1-based); white outside the palette, like the legacy fallback.
pub fn drone_tint(drone_id: u32) -> u32 {
    drone_id
        .checked_sub(1)
        .and_then(|i| DRONE_TINTS.get(i as usize).copied())
        .unwrap_or(0xffffff)
}

/// `0xRRGGBB` as `[r, g, b]` in 0..1 (fog colour, tints).
pub fn rgb(hex: u32) -> [f32; 3] {
    [(hex >> 16) & 0xff, (hex >> 8) & 0xff, hex & 0xff].map(|c| c as f32 / 255.0)
}

/// `0xRRGGBB` plus alpha as `[r, g, b, a]` in 0..1 (material tints, line colours).
pub fn rgba(hex: u32, alpha: f32) -> [f32; 4] {
    let [r, g, b] = rgb(hex);
    [r, g, b, alpha]
}

/// `0xRRGGBB` as an opaque clear colour for `FrameStyle::clear`.
pub fn clear(hex: u32) -> [f64; 4] {
    let [r, g, b] = rgb(hex);
    [f64::from(r), f64::from(g), f64::from(b), 1.0]
}

/// `0xRRGGBB` as RGBA8 bytes.
pub fn rgba8(hex: u32, alpha: u8) -> [u8; 4] {
    [((hex >> 16) & 0xff) as u8, ((hex >> 8) & 0xff) as u8, (hex & 0xff) as u8, alpha]
}

/// `0xRRGGBB` as a CSS colour, e.g. `#9fe8ff`.
pub fn css_hex(hex: u32) -> String {
    format!("#{:06x}", hex & 0xffffff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colours_are_byte_over_255() {
        assert_eq!(rgb(0xff0000), [1.0, 0.0, 0.0]);
        let c = clear(color::CONTROL_BG);
        assert!((c[0] - 10.0 / 255.0).abs() < 1e-6);
        assert!((c[1] - 17.0 / 255.0).abs() < 1e-6);
        assert!((c[2] - 40.0 / 255.0).abs() < 1e-6);
        assert_eq!(c[3], 1.0);
        assert_eq!(rgba8(0x39d98a, 200), [0x39, 0xd9, 0x8a, 200]);
        assert_eq!(css_hex(0x9fe8ff), "#9fe8ff");
        assert_eq!(css_hex(0x0000ff), "#0000ff");
    }

    #[test]
    fn drone_tints_follow_the_legacy_palette() {
        assert_eq!(drone_tint(1), 0x9fe8ff);
        assert_eq!(drone_tint(3), 0xc79fff);
        assert_eq!(drone_tint(0), 0xffffff);
        assert_eq!(drone_tint(9), 0xffffff);
    }
}
