//! Live reconstruction engine (replaces the 3DGS pipeline that lived here).
//!
//! cumulus3d `CompositionHost` grows the point cloud from live frame sets, the
//! UNet runs as the `on_frame` hook, deltamesh turns zone clouds into mesh
//! blocks, and everything leaves as `Chunk`s.
//!
//! The signatures below are the contract `app.rs` codes against. The bodies are
//! filled in by the reconstruction work.

use std::sync::Arc;

use skylens_shared::proto::model::v1::{
    Chunk, DetectJobRequest, DetectJobResult, FrameSet, ReconJobRequest, ReconJobResult,
};

use crate::detect::Detector;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// ENU origin; every zone and chunk is expressed about it.
    pub anchor: skylens_shared::geo::GeoAnchor,
    /// Camera names in a frame set, e.g. ["left", "center", "right"].
    pub cameras: Vec<String>,
    /// cumulus3d zone span and overlap, in frame sets.
    pub zone_span: usize,
    pub zone_overlap: usize,
    /// Working directory for cumulus3d (models, snapshots).
    pub work_dir: std::path::PathBuf,
    /// Use the GPU backends of cumulus3d and deltamesh.
    pub gpu: bool,
}

/// Result of declaring one frame set to the reconstruction node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushOutcome {
    /// New set, ingested once.
    Ingested,
    /// Already accepted or superseded; nothing happened.
    Ignored,
    /// Waiting for cameras that have not arrived yet.
    Pending,
}

pub struct Engine {
    _private: (),
}

impl Engine {
    pub fn new(cfg: EngineConfig, detector: Option<Arc<dyn Detector>>) -> anyhow::Result<Self> {
        let _ = (cfg, detector);
        todo!("reconstruction engine")
    }

    /// Declare one frame set (gRPC PushFrames).
    pub fn push(&self, set: FrameSet) -> anyhow::Result<PushOutcome> {
        let _ = set;
        todo!("reconstruction engine")
    }

    /// Chunks with a sequence above `after_sequence`, then every new one as it is
    /// produced (gRPC SubscribeChunks). Already-produced chunks are replayed first.
    pub fn subscribe(&self, after_sequence: u64) -> tokio::sync::mpsc::Receiver<Chunk> {
        let _ = after_sequence;
        todo!("reconstruction engine")
    }

    /// REST/gRPC recon job. `steps` selects the level (preview or refined).
    /// `progress` receives 0.0..=1.0.
    pub fn run_recon_job(
        &self,
        req: &ReconJobRequest,
        progress: &(dyn Fn(f64) + Send + Sync),
    ) -> anyhow::Result<ReconJobResult> {
        let _ = (req, progress);
        todo!("reconstruction engine")
    }

    /// REST/gRPC detect job over the segment's sources.
    pub fn run_detect_job(
        &self,
        req: &DetectJobRequest,
        progress: &(dyn Fn(f64) + Send + Sync),
    ) -> anyhow::Result<DetectJobResult> {
        let _ = (req, progress);
        todo!("reconstruction engine")
    }
}
