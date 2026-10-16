//! Contract between the UNet (SkyLensNet) and the reconstruction frame hook.
//!
//! The reconstruction engine (`models::skylens`) calls a [`Detector`] from its
//! cumulus3d `on_frame` hook, once per decoded camera frame, then projects the
//! result onto the point cloud and fuses it across frames by confidence.
//! `models::skylensnet` provides the libtorch implementation.

/// Hazard classes of the segmentation head: 0 normal, 1 fire, 2 collapse,
/// 3 road_blocked (255 ignore is a training-only label).
pub const HAZARD_CLASSES: usize = 4;

/// One person found by the point-detection head, in frame pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersonPoint {
    pub x: f32,
    pub y: f32,
    pub score: f32,
}

/// What the UNet saw in one frame.
#[derive(Debug, Clone)]
pub struct FrameDetections {
    pub width: u32,
    pub height: u32,
    /// Per-pixel class probabilities after softmax, laid out
    /// `[class][y][x]` (HAZARD_CLASSES * height * width).
    pub hazard_prob: Vec<f32>,
    pub people: Vec<PersonPoint>,
}

pub trait Detector: Send + Sync {
    /// `rgb` is RGB8 row-major (width * height * 3), as cumulus3d hands it to the
    /// frame hook. `thermal` is one byte per pixel when a thermal camera exists;
    /// without it the model runs with the thermal channel dropped.
    fn detect(
        &self,
        rgb: &[u8],
        thermal: Option<&[u8]>,
        width: u32,
        height: u32,
    ) -> anyhow::Result<FrameDetections>;
}
