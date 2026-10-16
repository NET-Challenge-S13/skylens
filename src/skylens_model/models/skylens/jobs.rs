//! The REST/gRPC job API on top of the same reconstruction code.
//!
//! A job builds its own short-lived `ReconNode` (separate host and folder, so it
//! never disturbs the live node), declares every position of its sources, and
//! closes the node. The whole job is **one zone** (`zones(positions, 0)`): zone 0
//! is the only zone cumulus3d aligns to GPS in its preview, so a preview-level
//! job still lands in ENU metres about the anchor.
//!
//! `steps` selects the level. It was the gsplat training step count; the old
//! core ladder was 1,000 / 7,000 / 30,000 steps. Below [`REFINED_MIN_STEPS`] the
//! job returns the zone preview (no bundle adjustment); at or above it, the job
//! waits for the refined zone (bundle adjustment, GPS re-alignment, refined
//! densification).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, anyhow, bail};
use cumulus3d_cli::compose::{CompositionHost, FrameKey, FrameState, LiveFrameSet, ReconNode, ReconcilePolicy};
use cumulus3d_cli::declare::{AlignFrame, Dense};
use cumulus3d_cli::events::{Event, ZoneRange};
use cumulus3d_core::io::ply::PointCloud;
use cumulus3d_core::Reconstruction;
use skylens_shared::proto::model::v1::{Gps as ProtoGps, Level, Source};

use super::chunks::ZonePoint;
use super::frames::ANCHOR_RECORD;
use super::fusion::{FrameObservation, View};
use super::recon::{zone_points, zone_views};
use super::sources;
use super::EngineConfig;
use crate::detect::Detector;

/// `steps` at or above this produce the refined level; below, the preview.
pub const REFINED_MIN_STEPS: u32 = 7_000;

/// Level selected by a job's `steps`.
pub fn level_for_steps(steps: u32) -> Level {
    if steps >= REFINED_MIN_STEPS { Level::Refined } else { Level::Preview }
}

/// What a job reconstructed.
pub struct JobOutput {
    pub points: Vec<ZonePoint>,
    pub views: HashMap<String, View>,
    pub observations: Vec<(String, FrameObservation)>,
    /// The cumulus3d cloud as delivered (empty without densification).
    pub cloud: Arc<PointCloud>,
}

/// Dense settings shared by the live node and jobs. Densification needs CUDA.
pub fn dense_for(gpu: bool) -> Option<Dense> {
    gpu.then(|| Dense::profile("fast").fusion_min_views(2))
}

/// Runs one job node over `sources` in `dir` (created, left for the caller to remove).
#[allow(clippy::too_many_arguments)]
pub fn run(
    cfg: &EngineConfig,
    dir: &Path,
    sources: &[Source],
    track: &[ProtoGps],
    level: Level,
    detector: Option<Arc<dyn Detector>>,
    progress: &(dyn Fn(f64) + Send + Sync),
) -> anyhow::Result<JobOutput> {
    let images = dir.join("images");
    std::fs::create_dir_all(&images).with_context(|| format!("create {}", images.display()))?;
    progress(0.0);
    let input = sources::prepare(&images, sources, track, &cfg.cameras)?;
    let n = input.names.len();
    if n == 0 {
        bail!("sources produced no frames");
    }
    if !input.gps.iter().flatten().any(Option::is_some) {
        bail!("sources carry no GPS (poses or track); the cloud cannot be placed in ENU");
    }
    progress(0.15);

    let host = CompositionHost::new();
    let mut b = ReconNode::declare()
        .id("job")
        .images(&images)
        .cameras(input.cameras.clone())
        .zones(n, 0)
        .align(AlignFrame::Enu)
        .fixed_enu_origin(true)
        .reconcile(ReconcilePolicy::AppendOnly);
    if cfg.gpu {
        b = b.gpu();
    }
    b = match dense_for(cfg.gpu) {
        Some(d) => b.dense(d),
        None => b.no_dense(),
    };
    let observations: Arc<Mutex<Vec<(String, FrameObservation)>>> = Arc::default();
    if let Some(det) = detector {
        let sink = Arc::clone(&observations);
        // Runs on the frame hook's own lane; the node closes only after the lane drains.
        b = b.on_frame(move |f| match det.detect(&f.rgb, None, f.width, f.height) {
            Ok(d) => sink
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push((f.name.clone(), FrameObservation::from_detections(&d))),
            Err(e) => tracing::warn!("detector failed on {}: {e:#}", f.name),
        });
    }
    let node = b.build(&host).map_err(|e| anyhow!("job node: {e}"))?;
    let events = node.subscribe().context("job node closed before start")?;
    // Drain events while composing: decoded frames are large and must not pile up in the channel.
    let collector = std::thread::Builder::new()
        .name("job-events".into())
        .spawn(move || collect(events))
        .context("spawn job event collector")?;

    let anchor = cfg.anchor;
    for p in 0..n {
        let mut set = LiveFrameSet::new(FrameKey::new("job", p as u64, 0)).gps_fix(ANCHOR_RECORD, anchor.lat, anchor.lon, anchor.alt);
        for (i, cam) in input.cameras.iter().enumerate() {
            set = set.file(cam.clone(), input.names[p][i].clone());
            if let Some(g) = input.gps[p][i] {
                set = set.gps_fix(cam.clone(), g.lat, g.lon, g.alt);
            }
        }
        if let Err(e) = node.compose(FrameState::ready(set)) {
            let _ = host.close_all();
            let _ = collector.join();
            bail!("position {p}: {e}");
        }
        progress(0.15 + 0.65 * (p + 1) as f64 / n as f64);
    }
    // Closing waits for the background refinement and drains the hook lanes;
    // the subscription channel then closes and the collector returns.
    let closed = host.close("job").map_err(|e| anyhow!("closing job node: {e}"));
    let seen = collector.join().map_err(|_| anyhow!("job event collector panicked"))?;
    closed?;
    progress(0.9);

    let (range, cloud, model) = match level {
        Level::Refined => seen.refined.ok_or_else(|| anyhow!("no refined result (too few registered frames?)"))?,
        _ => {
            let (range, cloud, model, frame) =
                seen.preview.ok_or_else(|| anyhow!("no preview result (too few registered frames?)"))?;
            if frame == "unaligned" {
                bail!("GPS alignment failed; the preview is not in ENU");
            }
            (range, cloud, model)
        }
    };
    let points = zone_points(&cloud, &model, range, &seen.positions);
    let views = zone_views(&model, range, &seen.positions);
    let observations = std::mem::take(&mut *observations.lock().unwrap_or_else(|p| p.into_inner()));
    Ok(JobOutput { points, views, observations, cloud })
}

type PreviewZone = (ZoneRange, Arc<PointCloud>, Arc<Reconstruction>, String);
type RefinedZone = (ZoneRange, Arc<PointCloud>, Arc<Reconstruction>);

#[derive(Default)]
struct Seen {
    positions: HashMap<String, usize>,
    preview: Option<PreviewZone>,
    refined: Option<RefinedZone>,
}

/// Keeps what a job needs from its node's events (zone 0) until the channel closes.
fn collect(events: std::sync::mpsc::Receiver<Event>) -> Seen {
    let mut seen = Seen::default();
    for ev in events {
        match ev {
            Event::FrameIngested { position, images, .. } => {
                for im in images {
                    seen.positions.insert(im, position);
                }
            }
            Event::ZonePreview { range, cloud, model, frame, .. } if range.zone == 0 => {
                seen.preview = Some((range, cloud, model, frame));
            }
            Event::ZoneRefined { range, cloud, model, .. } if range.zone == 0 => {
                seen.refined = Some((range, cloud, model));
            }
            Event::Warning { message, .. } => tracing::debug!("job: {message}"),
            Event::Error { message, .. } => tracing::warn!("job: {message}"),
            _ => {}
        }
    }
    seen
}

/// Fresh job folder under `<work_dir>/jobs/`.
pub fn job_dir(work_dir: &Path, kind: &str, segment: u32) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default();
    work_dir.join("jobs").join(format!("{kind}-s{segment}-{stamp}-{}", std::process::id()))
}

/// Writes the job's cloud as PLY (`x y z red green blue [nx ny nz]`).
pub fn write_cloud(path: &Path, out: &JobOutput) -> anyhow::Result<u64> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let cloud = if out.cloud.is_empty() {
        PointCloud {
            positions: out.points.iter().map(|p| p.pos).collect(),
            normals: Vec::new(),
            colors: out.points.iter().map(|p| p.rgb).collect(),
            keys: Vec::new(),
        }
    } else {
        (*out.cloud).clone()
    };
    cumulus3d_core::io::ply::write_ply(path, &cloud, cumulus3d_core::io::ply::PlyLayout::XyzRgbNormal)
        .map_err(|e| anyhow!("write {}: {e}", path.display()))?;
    Ok(std::fs::metadata(path)?.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_map_to_levels_at_the_threshold() {
        assert_eq!(level_for_steps(0), Level::Preview);
        assert_eq!(level_for_steps(1_000), Level::Preview);
        assert_eq!(level_for_steps(REFINED_MIN_STEPS - 1), Level::Preview);
        assert_eq!(level_for_steps(REFINED_MIN_STEPS), Level::Refined);
        assert_eq!(level_for_steps(30_000), Level::Refined);
    }

    #[test]
    fn dense_only_with_gpu() {
        assert!(dense_for(false).is_none());
        assert!(dense_for(true).is_some_and(|d| d.enabled));
    }

    #[test]
    fn job_dirs_are_distinct_per_kind_and_segment() {
        let w = Path::new("work");
        assert_ne!(job_dir(w, "recon", 1), job_dir(w, "detect", 1));
        assert!(job_dir(w, "recon", 3).starts_with("work/jobs"));
    }
}
