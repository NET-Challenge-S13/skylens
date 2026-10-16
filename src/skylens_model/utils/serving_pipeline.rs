//! Rust twin of `utils/serving_pipeline.py`: the compute seam, one entry
//! function per job type.
//!
//! [`Pipeline::run_recon`] and [`Pipeline::run_detect`] are the ONLY places the
//! REST and gRPC layers touch compute.
//!
//! * demo mode (`SKYLENS_DEMO=1`) behaves exactly like the Python demo path: it
//!   resolves a prebuilt `segment x level` PLY through the manifest
//!   `split_segments.py` writes, checks the file exists, and returns its real
//!   byte size after a delay proportional to the requested step count. Nothing
//!   is computed and the live engine is never built.
//! * live mode hands the job to [`models::skylens::Engine`] on a blocking
//!   thread. When the engine could not be built, jobs fail with
//!   [`PipelineUnavailable`] naming the reason instead of inventing results.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;

use super::geo;
use super::serving_config::Settings;
use super::serving_registry::Report;
use super::serving_schemas::{
    DetectJobRequest, DetectJobResult, DetectResultKind, DetectionCategory, DetectionKind,
    DetectionResult, Gps, JobSource, ReconJobRequest, ReconJobResult, ReconResultKind, SplatAlign,
};
use crate::models::skylens::Engine;

/// The requested compute cannot run in this process, and why.
#[derive(Debug, Clone, thiserror::Error)]
#[error("PipelineUnavailable: {0}")]
pub struct PipelineUnavailable(pub String);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn short_hex() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..12].to_string()
}

// ---------------------------------------------------------------------------
// Anchor frames
// ---------------------------------------------------------------------------
//
// The FIRST recon job of a flight establishes a frame and gets its id back;
// every later job must send that id back so all segments land in one space
// (see serving_pipeline.py and API.md §3 for the measured reason).

#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub id: String,
    pub origin_segment: i64,
    pub created_at: f64,
    /// Placement of chunks reconstructed in this frame (identity for demo assets).
    pub align: SplatAlign,
}

impl Frame {
    fn new(id: String, origin_segment: i64) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        Frame {
            id,
            origin_segment,
            created_at,
            align: SplatAlign::default(),
        }
    }
}

/// In-memory frame table. Lost on restart, like everything else here.
#[derive(Debug, Default)]
pub struct FrameStore {
    frames: Mutex<HashMap<String, Frame>>,
}

impl FrameStore {
    pub fn resolve(&self, requested: Option<&str>, segment: i64) -> Frame {
        let mut frames = lock(&self.frames);
        let Some(requested) = requested else {
            let frame = Frame::new(format!("frame-{}", short_hex()), segment);
            frames.insert(frame.id.clone(), frame.clone());
            tracing::info!("frame {} established by segment {segment}", frame.id);
            return frame;
        };
        if let Some(known) = frames.get(requested) {
            return known.clone();
        }
        // Unknown id: the core outlived this process. Adopting it beats failing
        // the job; the core still uses one id for the whole flight.
        let adopted = Frame::new(requested.to_string(), segment);
        frames.insert(adopted.id.clone(), adopted.clone());
        tracing::warn!("adopting unknown anchorFrame {requested} (server restart?)");
        adopted
    }

    pub fn count(&self) -> usize {
        lock(&self.frames).len()
    }
}

// ---------------------------------------------------------------------------
// Python truthiness helpers over the manifest JSON
// ---------------------------------------------------------------------------

/// `value or None`: missing, null, false, 0, "", [] and {} are all falsy.
fn truthy(v: Option<&Value>) -> Option<&Value> {
    match v? {
        Value::Null => None,
        Value::Bool(false) => None,
        Value::Number(n) if n.as_f64() == Some(0.0) => None,
        Value::String(s) if s.is_empty() => None,
        Value::Array(a) if a.is_empty() => None,
        Value::Object(o) if o.is_empty() => None,
        other => Some(other),
    }
}

/// Python `float(x)` for the shapes a manifest holds.
fn float_of(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// `float(obj.get(key) or default)`.
fn float_or(obj: &Value, key: &str, default: f64) -> f64 {
    truthy(obj.get(key)).and_then(float_of).unwrap_or(default)
}

/// Python `str(x)`.
fn str_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// `[float(v) for v in (obj.get(key) or default)]`.
fn floats_or(obj: &Value, key: &str, default: &[f64]) -> Vec<f64> {
    match truthy(obj.get(key)).and_then(Value::as_array) {
        Some(items) => items.iter().map(|v| float_of(v).unwrap_or(0.0)).collect(),
        None => default.to_vec(),
    }
}

fn segments_of(manifest: &Value) -> Vec<Value> {
    truthy(manifest.get("segments"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Demo assets
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct DemoAsset {
    pub level: i64,
    pub steps: i64,
    pub label: String,
    pub url: String,
    pub path: PathBuf,
    pub bytes: u64,
}

/// Read the segment x level manifest written by split_segments.py.
pub fn load_manifest(settings: &Settings) -> anyhow::Result<Value> {
    let path = &settings.manifest;
    if !path.exists() {
        return Err(PipelineUnavailable(format!(
            "demo manifest missing: {}. Build it with `uv run python -m \
             skylens_model.models.skylens.split_segments res/static/demo/step*_light.ply \
             --segments 4`.",
            path.display()
        ))
        .into());
    }
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

/// Pick the prebuilt asset for this segment at the best level <= steps.
pub fn resolve_asset(settings: &Settings, segment: i64, steps: i64) -> anyhow::Result<DemoAsset> {
    let manifest = load_manifest(settings)?;
    let segments = segments_of(&manifest);
    if segments.is_empty() {
        return Err(PipelineUnavailable(format!(
            "demo manifest has no segments: {}",
            settings.manifest.display()
        ))
        .into());
    }

    // The core may cut a flight into more segments than the demo has assets
    // for; wrap so the stream keeps flowing instead of erroring out mid-demo.
    let index = segment.rem_euclid(segments.len() as i64);
    if index != segment {
        tracing::warn!("segment {segment} wrapped to demo segment {index}");
    }
    let mut levels: Vec<Value> = truthy(segments[index as usize].get("levels"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if levels.is_empty() {
        return Err(PipelineUnavailable(format!("demo segment {index} has no levels")).into());
    }
    let level_steps = |l: &Value| l.get("steps").and_then(float_of).unwrap_or(0.0);
    levels.sort_by(|a, b| level_steps(a).total_cmp(&level_steps(b)));

    // Highest level the requested step count has paid for; if the request asks
    // for fewer steps than the cheapest asset, give the cheapest.
    let mut chosen = &levels[0];
    for level in &levels {
        if level_steps(level) <= steps as f64 {
            chosen = level;
        }
    }

    let rel = chosen
        .get("url")
        .map(str_of)
        .ok_or_else(|| anyhow::anyhow!("demo level without url in segment {index}"))?;
    let joined = settings.demo_root().join(&rel);
    let path = std::fs::canonicalize(&joined).unwrap_or(joined);
    if !path.exists() {
        return Err(PipelineUnavailable(format!(
            "demo asset listed in manifest but missing on disk: {}",
            path.display()
        ))
        .into());
    }
    let bytes = std::fs::metadata(&path)?.len();
    let int = |key: &str| chosen.get(key).and_then(float_of).unwrap_or(0.0) as i64;
    Ok(DemoAsset {
        level: int("level"),
        steps: int("steps"),
        label: chosen.get("label").map(str_of).unwrap_or_default(),
        url: format!("{}/{}", settings.url_base, rel),
        path,
        bytes,
    })
}

/// Simulated processing time for a request of `steps`, clamped to the limits.
pub fn demo_seconds(settings: &Settings, steps: i64) -> f64 {
    settings.min_seconds.max(
        settings
            .max_seconds
            .min(steps as f64 * settings.step_seconds),
    )
}

/// Burn the simulated processing time, reporting progress as it goes.
pub async fn simulate(seconds: f64, report: &Report, ticks: u32) {
    let per_tick = if seconds.is_finite() && seconds > 0.0 {
        seconds / ticks as f64
    } else {
        0.0
    };
    for i in 0..ticks {
        tokio::time::sleep(Duration::from_secs_f64(per_tick)).await;
        report((i + 1) as f64 / ticks as f64);
    }
}

// ---------------------------------------------------------------------------
// Placement of demo chunks (port of _flight / _demo_align)
// ---------------------------------------------------------------------------

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

/// Ground distance flown, heading in radians east-of-north, and the middle fix.
pub fn flight(fixes: &[Gps]) -> Option<(f64, f64, Gps)> {
    if fixes.len() < 2 {
        return None;
    }
    let m_per_lat = 111_320.0;
    let m_per_lon = 111_320.0 * fixes[0].lat.to_radians().cos();
    let flown: f64 = fixes
        .windows(2)
        .map(|w| ((w[1].lon - w[0].lon) * m_per_lon).hypot((w[1].lat - w[0].lat) * m_per_lat))
        .sum();
    let last = fixes[fixes.len() - 1];
    let east = (last.lon - fixes[0].lon) * m_per_lon;
    let north = (last.lat - fixes[0].lat) * m_per_lat;
    if flown <= 0.0 || (east == 0.0 && north == 0.0) {
        return None;
    }
    Some((flown, east.atan2(north), fixes[fixes.len() / 2]))
}

/// Every pose the segment was filmed from, in order.
pub fn segment_fixes(sources: &[JobSource]) -> Vec<Gps> {
    sources
        .iter()
        .flat_map(|s| s.poses.iter().map(|p| p.gps))
        .collect()
}

/// Put the prebuilt chunk on the ground it stands for: scale it to what was
/// flown, yaw its principal axis onto the flown heading, centre it on the
/// middle fix. See `_demo_align` in serving_pipeline.py for the full rationale.
pub fn demo_align(
    settings: &Settings,
    track: &[Gps],
    fixes: &[Gps],
    manifest: &Value,
    index: usize,
) -> SplatAlign {
    let measured = if track.is_empty() { fixes } else { track };
    let segments = segments_of(manifest);
    let Some((flown, heading, middle)) = flight(measured) else {
        return SplatAlign::default();
    };
    if segments.is_empty() {
        return SplatAlign::default();
    }

    let scale = if float_or(manifest, "metersPerUnit", 0.0) > 0.0 {
        1.0
    } else {
        let mut extents: Vec<f64> = segments
            .iter()
            .map(|s| float_or(s, "extent", 0.0))
            .filter(|e| *e > 0.0)
            .collect();
        let typical = median(&mut extents);
        if typical <= 0.0 {
            tracing::warn!("demo manifest has no segment extents; placing chunks unscaled");
            return SplatAlign::default();
        }
        flown / typical
    };

    // Scene axes are x=East, y=Up, z=-North; yaw = asset heading MINUS flown heading.
    let axis = floats_or(manifest, "axis", &[1.0, 0.0, 0.0]);
    let ax0 = axis.first().copied().unwrap_or(0.0);
    let asset_yaw = ax0.atan2(if axis.len() > 2 { -axis[2] } else { 0.0 });
    let yaw = asset_yaw - heading;
    let half = yaw / 2.0;
    let rotation = [0.0, half.sin(), 0.0, half.cos()];

    let centroid = segments
        .get(index)
        .map(|s| floats_or(s, "centroid", &[0.0, 0.0, 0.0]))
        .unwrap_or_else(|| vec![0.0; 3]);
    let c = |i: usize| centroid.get(i).copied().unwrap_or(0.0);
    let (sx, sy, sz) = (c(0) * scale, c(1) * scale, c(2) * scale);
    let (cos_y, sin_y) = (yaw.cos(), yaw.sin());
    let rx = cos_y * sx + sin_y * sz;
    let rz = -sin_y * sx + cos_y * sz;

    // Horizontal from the flight, height from the site datum.
    SplatAlign {
        anchor: Some(Gps {
            lat: middle.lat,
            lon: middle.lon,
            alt: settings.anchor.alt,
        }),
        position: [-rx, -sy, -rz],
        rotation,
        scale: [scale, scale, scale],
    }
}

/// The people that are REALLY in the capture (`<demo root>/people.json`).
pub fn load_people(settings: &Settings) -> anyhow::Result<Vec<Value>> {
    let path = settings.demo_root().join("people.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let data: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    Ok(truthy(data.get("people"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Whether the live engine exists in this process.
pub enum LiveEngine {
    /// Demo mode: the engine is never built.
    Disabled,
    Ready(Arc<Engine>),
    /// Live mode, but building the engine failed; jobs report this reason.
    Failed(String),
}

pub struct Pipeline {
    pub settings: Settings,
    pub frames: FrameStore,
    /// Placement of the CURRENT flight's scene, kept from the first recon job so
    /// detections land on the geometry the board actually draws.
    flight_align: Mutex<Option<SplatAlign>>,
    live: LiveEngine,
}

impl Pipeline {
    pub fn new(settings: Settings, live: LiveEngine) -> Self {
        Pipeline {
            settings,
            frames: FrameStore::default(),
            flight_align: Mutex::new(None),
            live,
        }
    }

    /// The live engine, or why there is none.
    pub fn engine(&self) -> Result<Arc<Engine>, PipelineUnavailable> {
        match &self.live {
            LiveEngine::Ready(engine) => Ok(engine.clone()),
            LiveEngine::Disabled => Err(PipelineUnavailable(
                "the live engine is not built in demo mode (SKYLENS_DEMO=1)".into(),
            )),
            LiveEngine::Failed(reason) => Err(PipelineUnavailable(format!(
                "the live reconstruction engine failed to start: {reason}"
            ))),
        }
    }

    // -- reconstruction ----------------------------------------------------

    /// Reconstruct one capture segment at one refinement level.
    pub async fn run_recon(
        &self,
        req: ReconJobRequest,
        report: Report,
    ) -> anyhow::Result<ReconJobResult> {
        let frame = self
            .frames
            .resolve(req.anchor_frame.as_deref(), req.segment);
        if self.settings.demo {
            self.recon_demo(&req, &frame, &report).await
        } else {
            self.recon_live(req, frame, report).await
        }
    }

    async fn recon_demo(
        &self,
        req: &ReconJobRequest,
        frame: &Frame,
        report: &Report,
    ) -> anyhow::Result<ReconJobResult> {
        let asset = resolve_asset(&self.settings, req.segment, req.steps)?;
        tracing::info!(
            "recon(demo) segment {} steps {} -> level {} ({} steps) {}",
            req.segment,
            req.steps,
            asset.level,
            asset.steps,
            asset.url
        );
        // Proportional to the REQUESTED steps: the delay pattern is about how
        // long a level makes the operator wait.
        simulate(demo_seconds(&self.settings, req.steps), report, 20).await;
        let manifest = load_manifest(&self.settings)?;
        let n = segments_of(&manifest).len().max(1) as i64;
        let index = req.segment.rem_euclid(n) as usize;
        let align = demo_align(
            &self.settings,
            &req.track,
            &segment_fixes(&req.sources),
            &manifest,
            index,
        );
        // The board places the WHOLE scene by the origin segment's align;
        // detections reuse it so a marker lands on the drawn person.
        if req.segment == frame.origin_segment && align.anchor.is_some() {
            *lock(&self.flight_align) = Some(align);
        }
        Ok(ReconJobResult {
            kind: ReconResultKind::Tag,
            segment: req.segment,
            steps: asset.steps,
            url: asset.url,
            bytes: asset.bytes,
            align,
            anchor_frame: frame.id.clone(),
        })
    }

    async fn recon_live(
        &self,
        req: ReconJobRequest,
        frame: Frame,
        report: Report,
    ) -> anyhow::Result<ReconJobResult> {
        let engine = self.engine().map_err(|e| {
            PipelineUnavailable(format!(
                "{}; requested segment={} steps={} frame={} sources={}. Run with \
                 SKYLENS_DEMO=1 for the prebuilt assets.",
                e.0,
                req.segment,
                req.steps,
                frame.id,
                req.sources.len()
            ))
        })?;
        let request = skylens_shared::proto::model::v1::ReconJobRequest::from(&req);
        let out =
            tokio::task::spawn_blocking(move || engine.run_recon_job(&request, &*report)).await??;
        let mut result = ReconJobResult::from(out);
        if result.anchor_frame.is_empty() {
            result.anchor_frame = frame.id;
        }
        Ok(result)
    }

    // -- detection ---------------------------------------------------------

    /// Detect people and danger zones in one capture segment.
    pub async fn run_detect(
        &self,
        req: DetectJobRequest,
        report: Report,
    ) -> anyhow::Result<DetectJobResult> {
        if self.settings.demo {
            self.detect_demo(&req, &report).await
        } else {
            self.detect_live(req, report).await
        }
    }

    /// Markers for the people that really stand in this segment's slab. No
    /// danger markers: the capture has no collapsed structure, and an invented
    /// one on the board is indistinguishable from a real one.
    async fn detect_demo(
        &self,
        req: &DetectJobRequest,
        report: &Report,
    ) -> anyhow::Result<DetectJobResult> {
        let manifest = load_manifest(&self.settings)?;
        let nseg = segments_of(&manifest).len().max(1) as i64;
        let index = req.segment.rem_euclid(nseg);

        let flight_align = *lock(&self.flight_align);
        let align = flight_align.unwrap_or_else(|| {
            demo_align(
                &self.settings,
                &[],
                &segment_fixes(&req.sources),
                &manifest,
                index as usize,
            )
        });
        let mut detections = Vec::new();
        if let Some(anchor) = align.anchor {
            let axis = floats_or(&manifest, "axis", &[1.0, 0.0, 0.0]);
            let origin = floats_or(&manifest, "origin", &[0.0, 0.0, 0.0]);
            if axis.len() != 3 || origin.len() != 3 {
                anyhow::bail!("demo manifest axis/origin must have three components");
            }
            let boundaries = floats_or(&manifest, "boundaries", &[]);
            let yaw = 2.0 * align.rotation[1].atan2(align.rotation[3]);
            let (cos_y, sin_y) = (yaw.cos(), yaw.sin());
            let scale = align.scale[0];
            let geo_anchor = geo::Gps {
                lat: anchor.lat,
                lon: anchor.lon,
                alt: anchor.alt,
            };
            for person in load_people(&self.settings)? {
                let pos = person
                    .get("pos")
                    .and_then(Value::as_array)
                    .filter(|p| p.len() == 3)
                    .ok_or_else(|| anyhow::anyhow!("people.json entry without a 3-vector pos"))?;
                let p: Vec<f64> = pos.iter().map(|v| float_of(v).unwrap_or(0.0)).collect();
                let proj: f64 = (0..3).map(|i| (p[i] - origin[i]) * axis[i]).sum();
                let slab = boundaries.iter().filter(|b| proj > **b).count() as i64;
                if slab != index {
                    continue;
                }
                let (sx, sy, sz) = (p[0] * scale, p[1] * scale, p[2] * scale);
                // world = R_yaw * (s * p) + align.position, in scene axes.
                let wx = cos_y * sx + sin_y * sz + align.position[0];
                let wy = sy + align.position[1];
                let wz = -sin_y * sx + cos_y * sz + align.position[2];
                let gps = geo::enu_to_gps(
                    geo::Enu {
                        e: wx,
                        n: -wz,
                        u: wy,
                    },
                    geo_anchor,
                );
                detections.push(DetectionResult {
                    kind: DetectionKind::Tag,
                    id: truthy(person.get("id"))
                        .map(str_of)
                        .unwrap_or_else(|| format!("person-{slab}")),
                    category: DetectionCategory::Person,
                    gps: gps.into(),
                    confidence: float_or(&person, "confidence", 0.9),
                    label: truthy(person.get("label"))
                        .map(str_of)
                        .unwrap_or_else(|| "구조 대상자 추정".to_string()),
                    segment: req.segment,
                });
            }
        }
        simulate(self.settings.min_seconds.max(0.2), report, 4).await;
        Ok(DetectJobResult {
            kind: DetectResultKind::Tag,
            segment: req.segment,
            detections,
        })
    }

    async fn detect_live(
        &self,
        req: DetectJobRequest,
        report: Report,
    ) -> anyhow::Result<DetectJobResult> {
        let engine = self.engine().map_err(|e| {
            PipelineUnavailable(format!(
                "{}; requested segment={} sources={}. Run with SKYLENS_DEMO=1 for demo markers.",
                e.0,
                req.segment,
                req.sources.len()
            ))
        })?;
        let request = skylens_shared::proto::model::v1::DetectJobRequest::from(&req);
        let out = tokio::task::spawn_blocking(move || engine.run_detect_job(&request, &*report))
            .await??;
        Ok(DetectJobResult::try_from(out)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::serving_schemas::{DroneTelemetry, TelemetryKind};
    use serde_json::json;
    use std::collections::HashMap;
    use std::time::Instant;

    struct Demo {
        dir: PathBuf,
        settings: Settings,
    }

    impl Drop for Demo {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn demo(manifest: Value, files: &[&str], extra: &[(&str, &str)]) -> Demo {
        let dir = std::env::temp_dir().join(format!("skylens-pipeline-{}", short_hex()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("segments.json"), manifest.to_string()).unwrap();
        for f in files {
            std::fs::write(dir.join(f), b"ply-bytes").unwrap();
        }
        let mut env: HashMap<String, String> = HashMap::from([
            ("SKYLENS_DEMO".into(), "1".into()),
            (
                "SKYLENS_DEMO_MANIFEST".into(),
                dir.join("segments.json").display().to_string(),
            ),
            ("SKYLENS_DEMO_MIN_SECONDS".into(), "0.05".into()),
            ("SKYLENS_DEMO_MAX_SECONDS".into(), "0.4".into()),
            ("SKYLENS_DEMO_STEP_SECONDS".into(), "0.0001".into()),
        ]);
        for (k, v) in extra {
            env.insert(k.to_string(), v.to_string());
        }
        let settings = Settings::from_lookup(|k| env.get(k).cloned());
        Demo { dir, settings }
    }

    fn two_levels() -> Value {
        json!({
            "segments": [
                {"levels": [
                    {"level": 1, "steps": 7000, "url": "seg0_hi.ply", "label": "hi"},
                    {"level": 0, "steps": 250, "url": "seg0_lo.ply", "label": "lo"}
                ], "extent": 10.0, "centroid": [0.0, 0.0, 0.0]},
                {"levels": [{"level": 0, "steps": 250, "url": "seg1_lo.ply"}],
                 "extent": 10.0, "centroid": [1.0, 0.0, 0.0]}
            ],
            "axis": [1.0, 0.0, 0.0],
            "origin": [0.0, 0.0, 0.0],
            "boundaries": [5.0]
        })
    }

    fn report_sink() -> (Report, Arc<Mutex<Vec<f64>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        (Arc::new(move |v: f64| s.lock().unwrap().push(v)), seen)
    }

    #[test]
    fn asset_level_selection_matches_python() {
        let d = demo(
            two_levels(),
            &["seg0_hi.ply", "seg0_lo.ply", "seg1_lo.ply"],
            &[],
        );
        let s = &d.settings;
        assert_eq!(resolve_asset(s, 0, 7000).unwrap().steps, 7000);
        assert_eq!(resolve_asset(s, 0, 300).unwrap().steps, 250);
        // Fewer steps than the cheapest level: the cheapest.
        assert_eq!(resolve_asset(s, 0, 10).unwrap().steps, 250);
        let a = resolve_asset(s, 0, 300).unwrap();
        assert_eq!(a.url, "/res/static/demo/seg0_lo.ply");
        assert_eq!(a.bytes, 9);
        assert_eq!(a.label, "lo");
        // Segments beyond the manifest wrap.
        assert_eq!(
            resolve_asset(s, 3, 9999).unwrap().url,
            "/res/static/demo/seg1_lo.ply"
        );
    }

    #[test]
    fn missing_asset_or_manifest_is_pipeline_unavailable() {
        let d = demo(two_levels(), &["seg0_lo.ply"], &[]);
        let err = resolve_asset(&d.settings, 0, 7000).unwrap_err();
        assert!(
            err.downcast_ref::<PipelineUnavailable>().is_some(),
            "{err:#}"
        );

        let mut s = d.settings.clone();
        s.manifest = d.dir.join("nope.json");
        let err = resolve_asset(&s, 0, 1).unwrap_err();
        assert!(format!("{err:#}").starts_with("PipelineUnavailable: demo manifest missing"));
    }

    #[test]
    fn demo_delay_is_clamped() {
        let d = demo(two_levels(), &[], &[]);
        assert_eq!(demo_seconds(&d.settings, 1), 0.05);
        assert!((demo_seconds(&d.settings, 1000) - 0.1).abs() < 1e-12);
        assert_eq!(demo_seconds(&d.settings, 10_000_000), 0.4);
    }

    #[tokio::test]
    async fn demo_recon_waits_reports_and_returns_asset() {
        let d = demo(
            two_levels(),
            &["seg0_hi.ply", "seg0_lo.ply", "seg1_lo.ply"],
            &[],
        );
        let p = Pipeline::new(d.settings.clone(), LiveEngine::Disabled);
        let (report, seen) = report_sink();
        let req = ReconJobRequest {
            segment: 0,
            sources: vec![],
            steps: 1000, // 1000 * 0.0001 = 0.1 s
            track: vec![],
            anchor_frame: None,
        };
        let t0 = Instant::now();
        let out = p.run_recon(req, report).await.unwrap();
        let elapsed = t0.elapsed().as_secs_f64();
        assert!(elapsed >= 0.095, "too fast: {elapsed}");
        assert!(elapsed < 2.0, "too slow: {elapsed}");
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 20);
        assert_eq!(*seen.last().unwrap(), 1.0);
        assert_eq!(out.steps, 250);
        assert!(out.anchor_frame.starts_with("frame-"));
        assert_eq!(out.align, SplatAlign::default()); // no track, no poses
        assert_eq!(p.frames.count(), 1);
    }

    #[tokio::test]
    async fn demo_recon_reuses_and_adopts_frames() {
        let d = demo(two_levels(), &["seg0_lo.ply", "seg1_lo.ply"], &[]);
        let p = Pipeline::new(d.settings.clone(), LiveEngine::Disabled);
        let (report, _) = report_sink();
        let mk = |segment, anchor: Option<String>| ReconJobRequest {
            segment,
            sources: vec![],
            steps: 1,
            track: vec![],
            anchor_frame: anchor,
        };
        let first = p.run_recon(mk(0, None), report.clone()).await.unwrap();
        let second = p
            .run_recon(mk(1, Some(first.anchor_frame.clone())), report.clone())
            .await
            .unwrap();
        assert_eq!(second.anchor_frame, first.anchor_frame);
        let adopted = p
            .run_recon(mk(1, Some("frame-old".into())), report)
            .await
            .unwrap();
        assert_eq!(adopted.anchor_frame, "frame-old");
        assert_eq!(p.frames.count(), 2);
    }

    #[tokio::test]
    async fn demo_align_scales_to_flown_track() {
        let d = demo(
            two_levels(),
            &["seg0_lo.ply", "seg1_lo.ply"],
            &[("SKYLENS_ANCHOR", "36,127,30")],
        );
        let p = Pipeline::new(d.settings.clone(), LiveEngine::Disabled);
        let (report, _) = report_sink();
        // Due north, about 40 m.
        let track = vec![
            Gps {
                lat: 36.0,
                lon: 127.0,
                alt: 60.0,
            },
            Gps {
                lat: 36.0 + 40.0 / 111_320.0,
                lon: 127.0,
                alt: 60.0,
            },
        ];
        let req = ReconJobRequest {
            segment: 0,
            sources: vec![],
            steps: 1,
            track,
            anchor_frame: None,
        };
        let out = p.run_recon(req, report).await.unwrap();
        let anchor = out.align.anchor.expect("anchored");
        assert_eq!(
            anchor.alt, 30.0,
            "height from the site datum, not the aircraft"
        );
        assert!(
            (out.align.scale[0] - 4.0).abs() < 1e-6,
            "{:?}",
            out.align.scale
        );
        // Asset axis is east (yaw pi/2), flight is north (0): yaw = pi/2.
        let yaw = 2.0 * out.align.rotation[1].atan2(out.align.rotation[3]);
        assert!((yaw - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    }

    #[tokio::test]
    async fn demo_detect_emits_only_people_in_the_slab() {
        let d = demo(
            two_levels(),
            &["seg0_lo.ply", "seg1_lo.ply"],
            &[("SKYLENS_ANCHOR", "36,127,30")],
        );
        std::fs::write(
            d.dir.join("people.json"),
            json!({"people": [
                {"id": "p-a", "pos": [1.0, 0.0, 0.0], "confidence": 0.8},
                {"pos": [9.0, 0.0, 0.0]}
            ]})
            .to_string(),
        )
        .unwrap();
        let p = Pipeline::new(d.settings.clone(), LiveEngine::Disabled);
        let (report, seen) = report_sink();
        let pose = |lat: f64| DroneTelemetry {
            kind: TelemetryKind::Tag,
            drone_id: 1,
            station: None,
            gps: Gps {
                lat,
                lon: 127.0,
                alt: 60.0,
            },
            heading_deg: 0.0,
            speed: 5.0,
            battery_pct: 90.0,
            t: 0,
        };
        let req = DetectJobRequest {
            segment: 1,
            sources: vec![JobSource {
                uri: "x".into(),
                poses: vec![pose(36.0), pose(36.0004)],
            }],
        };
        let t0 = Instant::now();
        let out = p.run_detect(req, report).await.unwrap();
        assert!(t0.elapsed().as_secs_f64() >= 0.19, "detect floor is 0.2 s");
        assert_eq!(seen.lock().unwrap().len(), 4);
        assert_eq!(out.detections.len(), 1);
        let det = &out.detections[0];
        assert_eq!(det.id, "person-1");
        assert_eq!(det.confidence, 0.9);
        assert_eq!(det.label, "구조 대상자 추정");
        assert_eq!(det.segment, 1);
    }

    #[tokio::test]
    async fn live_mode_without_engine_fails_loudly() {
        let d = demo(two_levels(), &[], &[("SKYLENS_DEMO", "0")]);
        let p = Pipeline::new(d.settings.clone(), LiveEngine::Failed("no gpu".into()));
        let (report, _) = report_sink();
        let req = DetectJobRequest {
            segment: 0,
            sources: vec![],
        };
        let err = p.run_detect(req, report).await.unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.starts_with("PipelineUnavailable: "), "{msg}");
        assert!(msg.contains("no gpu"), "{msg}");
    }

    #[test]
    fn flight_needs_two_distinct_fixes() {
        let a = Gps {
            lat: 36.0,
            lon: 127.0,
            alt: 0.0,
        };
        assert!(flight(&[a]).is_none());
        assert!(flight(&[a, a]).is_none());
        let b = Gps {
            lat: 36.0,
            lon: 127.001,
            alt: 0.0,
        };
        let (flown, heading, _) = flight(&[a, b]).unwrap();
        assert!(flown > 80.0 && flown < 100.0);
        assert!((heading - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    }
}
