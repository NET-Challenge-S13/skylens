//! Rust twin of `utils/serving_config.py`: server settings, all from the
//! environment so nothing is hardcoded per host.
//!
//! | env | default | meaning |
//! |---|---|---|
//! | `SKYLENS_DEMO`              | `0`                  | 1 = demo mode: serve prebuilt assets, compute nothing |
//! | `SKYLENS_DEMO_MANIFEST`     | `<repo>/res/static/demo/segments.json` | segment x level manifest |
//! | `SKYLENS_DEMO_URL_BASE`     | `/res/static/demo`   | url prefix the clients fetch assets under |
//! | `SKYLENS_DEMO_STEP_SECONDS` | `0.0006`             | simulated seconds per training step |
//! | `SKYLENS_DEMO_MIN_SECONDS`  | `0.3`                | floor on the simulated delay |
//! | `SKYLENS_DEMO_MAX_SECONDS`  | `12.0`               | cap on the simulated delay |
//! | `SKYLENS_ANCHOR`            | `36.3685,127.3475,30` | ENU origin, matches shared/viewer/config.ts |
//!
//! Rust-only (live engine and listeners; `app.py` takes its port from uvicorn):
//!
//! | env | default | meaning |
//! |---|---|---|
//! | `SKYLENS_REST_ADDR`     | `0.0.0.0:8100` | REST listener (same endpoints as app.py) |
//! | `SKYLENS_GRPC_ADDR`     | `0.0.0.0:8101` | gRPC `skylens.model.v1.ModelService` listener |
//! | `SKYLENS_CHECKPOINT`    | unset          | SkyLensNet checkpoint; unset = live mode runs without a detector |
//! | `SKYLENS_CAMERAS`       | `left,center,right` | camera names in a frame set |
//! | `SKYLENS_ZONE_SPAN`     | `6`            | cumulus3d zone span, in frame sets |
//! | `SKYLENS_ZONE_OVERLAP`  | `2`            | cumulus3d zone overlap, in frame sets |
//! | `SKYLENS_SETTLE_TOL`    | `0.02`         | stable display: intrinsics count as settled when two refined zones' focal lengths differ by less than this (relative) |
//! | `SKYLENS_SETTLE_MAX_ZONE` | `1`          | stable display: intrinsics are frozen at this zone at the latest (earlier zones may be re-densified once) |
//! | `SKYLENS_INTRINSICS_SIGMA` | `0.02`      | intrinsics prior: relative focal sigma (the demo focal is an estimate); distortion sigma `SKYLENS_INTRINSICS_DIST_SIGMA` = `0.01` |
//! | `SKYLENS_WORK_DIR`      | `<repo>/runs/model` | cumulus3d working directory |
//! | `SKYLENS_GPU`           | unset (= CUDA available) | force the GPU backends on or off |
//! | `SKYLENS_STABLE_DISPLAY` | `1`           | cumulus3d stable display: points leave as the live diff stream (point slot chunks); `0` = zone previews and refined zones as point chunks |
//! | `SKYLENS_LIVE_STRIDE`   | `6`            | stable display: keep 1 of this many points in the live display (1 = every point) |
//! | `SKYLENS_ZONE_PREVIEW`  | `1`            | stable display: sparse zone previews as `Preview` deltas (slot_kind 4) before the refined zone |
//! | `SKYLENS_BACKLOG_BATCH` | unset          | backlog batching threshold (cumulus3d default 3); positions waiting at least this many are processed as one batch |
//! | `SKYLENS_CAMERA_JSON`  | `<repo>/data/demo/camera.json` | camera intrinsics estimate used as the live node's intrinsics prior; a missing file means no prior |
//! | `SKYLENS_FRAME_WIDTH`   | unset (= the file's `video_width`) | width of the pushed frames; the prior's fx, fy, cx, cy are scaled by this / `video_width` |
//! | `SKYLENS_PRECISION`     | `auto`         | SkyLensNet arithmetic on CUDA: `auto` (by compute capability: >= 8.0 bf16, 7.x fp16 autocast, else fp32), `fp32`, `fp16`, `bf16` |
//! | `DELTAMESH_EXTRACT_BATCH` | `128` (set by `app::main`) | deltamesh GPU mesher blocks per extraction batch; deltamesh's own default 512 holds about 0.7 GB more GPU memory |

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::geo::Gps;
use crate::models::skylensnet::modeling_skylensnet::Precision;

/// `src/skylens_model` -> `src` -> repo.
pub fn repo_root() -> PathBuf {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    here.parent().and_then(Path::parent).unwrap_or(here).to_path_buf()
}

pub const DEFAULT_ANCHOR: Gps = Gps { lat: 36.3685, lon: 127.3475, alt: 30.0 };

#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    pub demo: bool,
    pub manifest: PathBuf,
    pub url_base: String,
    pub step_seconds: f64,
    pub min_seconds: f64,
    pub max_seconds: f64,
    pub anchor: Gps,
    // -- Rust-only ---------------------------------------------------------
    pub rest_addr: SocketAddr,
    pub grpc_addr: SocketAddr,
    pub checkpoint: Option<PathBuf>,
    pub cameras: Vec<String>,
    pub zone_span: usize,
    pub zone_overlap: usize,
    pub work_dir: PathBuf,
    /// `None` = decide from CUDA availability at engine build time.
    pub gpu: Option<bool>,
    /// cumulus3d stable display with the live diff stream.
    pub stable_display: bool,
    /// Stable display: live display thinning (1 of this many points, at least 1).
    pub live_stride: usize,
    /// Stable display: sparse zone previews as `Preview` deltas (`SKYLENS_ZONE_PREVIEW`, default on).
    pub zone_preview: bool,
    /// Backlog batch threshold (`SKYLENS_BACKLOG_BATCH`; unset = cumulus3d default, 0/1 = off).
    pub backlog_batch: Option<usize>,
    /// Camera intrinsics estimate for the intrinsics prior (`None` = no prior).
    pub camera_json: Option<PathBuf>,
    /// Width of the pushed frames (`None` = the file's `video_width`).
    pub frame_width: Option<u32>,
    /// Stable display: intrinsics settle tolerance (relative focal change between refined zones).
    pub settle_tol: f64,
    /// Stable display: the intrinsics are frozen at this zone at the latest.
    pub settle_max_zone: usize,
    /// Intrinsics prior: relative focal sigma and absolute distortion sigma.
    pub focal_sigma: f64,
    pub dist_sigma: f64,
    /// `None` = `auto`; an unparsable value is also `auto`.
    pub precision: Option<Precision>,
}

impl Settings {
    /// Directory the manifest's relative urls resolve against.
    pub fn demo_root(&self) -> PathBuf {
        self.manifest.parent().map(Path::to_path_buf).unwrap_or_default()
    }

    /// Read settings through `get` (the process environment in production, a
    /// map in tests).
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Self {
        let flag = |name: &str| -> Option<bool> {
            get(name).map(|raw| matches!(raw.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        };
        let num = |name: &str, default: f64| -> f64 {
            get(name).and_then(|raw| raw.trim().parse::<f64>().ok()).unwrap_or(default)
        };
        let int = |name: &str, default: usize| -> usize {
            get(name).and_then(|raw| raw.trim().parse::<usize>().ok()).unwrap_or(default)
        };
        let addr = |name: &str, default: &str| -> SocketAddr {
            get(name)
                .and_then(|raw| raw.trim().parse().ok())
                .unwrap_or_else(|| default.parse().expect("valid default address"))
        };

        let manifest = get("SKYLENS_DEMO_MANIFEST")
            .map(PathBuf::from)
            .unwrap_or_else(|| repo_root().join("res").join("static").join("demo").join("segments.json"));
        let manifest = std::path::absolute(&manifest).unwrap_or(manifest);

        let cameras: Vec<String> = get("SKYLENS_CAMERAS")
            .unwrap_or_else(|| "left,center,right".into())
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        Settings {
            demo: flag("SKYLENS_DEMO").unwrap_or(false),
            manifest,
            url_base: get("SKYLENS_DEMO_URL_BASE")
                .unwrap_or_else(|| "/res/static/demo".into())
                .trim_end_matches('/')
                .to_string(),
            step_seconds: num("SKYLENS_DEMO_STEP_SECONDS", 0.0006),
            min_seconds: num("SKYLENS_DEMO_MIN_SECONDS", 0.3),
            max_seconds: num("SKYLENS_DEMO_MAX_SECONDS", 12.0),
            anchor: parse_anchor(get("SKYLENS_ANCHOR").as_deref()),
            rest_addr: addr("SKYLENS_REST_ADDR", "0.0.0.0:8100"),
            grpc_addr: addr("SKYLENS_GRPC_ADDR", "0.0.0.0:8101"),
            checkpoint: get("SKYLENS_CHECKPOINT")
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .map(PathBuf::from),
            cameras,
            zone_span: int("SKYLENS_ZONE_SPAN", 6),
            zone_overlap: int("SKYLENS_ZONE_OVERLAP", 2),
            work_dir: get("SKYLENS_WORK_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| repo_root().join("runs").join("model")),
            gpu: flag("SKYLENS_GPU"),
            stable_display: flag("SKYLENS_STABLE_DISPLAY").unwrap_or(true),
            zone_preview: flag("SKYLENS_ZONE_PREVIEW").unwrap_or(true),
            backlog_batch: get("SKYLENS_BACKLOG_BATCH").and_then(|v| v.trim().parse().ok()),
            live_stride: int("SKYLENS_LIVE_STRIDE", crate::models::skylens::DEFAULT_LIVE_STRIDE).max(1),
            camera_json: Some(
                get("SKYLENS_CAMERA_JSON")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| repo_root().join("data").join("demo").join("camera.json")),
            ),
            frame_width: get("SKYLENS_FRAME_WIDTH").and_then(|v| v.trim().parse().ok()).filter(|w| *w > 0),
            settle_tol: num("SKYLENS_SETTLE_TOL", 0.02),
            settle_max_zone: int("SKYLENS_SETTLE_MAX_ZONE", 1),
            focal_sigma: num("SKYLENS_INTRINSICS_SIGMA", 0.02),
            dist_sigma: num("SKYLENS_INTRINSICS_DIST_SIGMA", 0.01),
            precision: get("SKYLENS_PRECISION").and_then(|raw| match raw.parse::<Precision>() {
                Ok(Precision::Auto) => None,
                Ok(p) => Some(p),
                Err(err) => {
                    tracing::warn!("SKYLENS_PRECISION ignored: {err}");
                    None
                }
            }),
        }
    }

    pub fn from_env() -> Self {
        Self::from_lookup(|name| std::env::var(name).ok())
    }
}

/// `"lat,lon,alt"`; anything unparsable falls back to the default anchor, like
/// `_anchor()` in serving_config.py (extra fields past the third are ignored).
pub fn parse_anchor(raw: Option<&str>) -> Gps {
    let raw = raw.unwrap_or("36.3685,127.3475,30");
    let parts: Vec<&str> = raw.split(',').take(3).collect();
    if parts.len() != 3 {
        return DEFAULT_ANCHOR;
    }
    let parsed: Result<Vec<f64>, _> = parts.iter().map(|p| p.trim().parse::<f64>()).collect();
    match parsed {
        Ok(v) => Gps { lat: v[0], lon: v[1], alt: v[2] },
        Err(_) => DEFAULT_ANCHOR,
    }
}

/// Read once per process. Restart the server to change these.
pub fn settings() -> &'static Settings {
    static SETTINGS: OnceLock<Settings> = OnceLock::new();
    SETTINGS.get_or_init(Settings::from_env)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn with(pairs: &[(&str, &str)]) -> Settings {
        let map: HashMap<String, String> =
            pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        Settings::from_lookup(|k| map.get(k).cloned())
    }

    #[test]
    fn defaults_match_python() {
        let s = with(&[]);
        assert!(!s.demo);
        assert!(s.manifest.ends_with("res/static/demo/segments.json"));
        assert_eq!(s.url_base, "/res/static/demo");
        assert_eq!(s.step_seconds, 0.0006);
        assert_eq!(s.min_seconds, 0.3);
        assert_eq!(s.max_seconds, 12.0);
        assert_eq!(s.anchor, DEFAULT_ANCHOR);
        assert_eq!(s.rest_addr.port(), 8100);
        assert_eq!(s.grpc_addr.port(), 8101);
        assert_eq!(s.checkpoint, None);
        assert_eq!(s.cameras, vec!["left", "center", "right"]);
        assert_eq!(s.gpu, None);
        assert_eq!(s.precision, None);
        assert!(s.stable_display);
        assert_eq!(s.live_stride, 6);
        assert!(s.zone_preview);
        assert_eq!(s.backlog_batch, None);
        assert_eq!((s.zone_span, s.zone_overlap), (6, 2));
        assert_eq!((s.settle_tol, s.settle_max_zone), (0.02, 1));
        assert_eq!((s.focal_sigma, s.dist_sigma), (0.02, 0.01));
    }

    #[test]
    fn reconstruction_knobs_override() {
        let s = with(&[
            ("SKYLENS_ZONE_SPAN", "9"),
            ("SKYLENS_SETTLE_MAX_ZONE", "3"),
            ("SKYLENS_SETTLE_TOL", "0.01"),
            ("SKYLENS_INTRINSICS_SIGMA", "0.005"),
        ]);
        assert_eq!((s.zone_span, s.settle_max_zone, s.settle_tol, s.focal_sigma), (9, 3, 0.01, 0.005));
    }

    #[test]
    fn stable_display_and_live_stride_overrides() {
        let s = with(&[("SKYLENS_STABLE_DISPLAY", "0"), ("SKYLENS_LIVE_STRIDE", "1")]);
        assert!(!s.stable_display);
        assert_eq!(s.live_stride, 1);
        assert_eq!(with(&[("SKYLENS_LIVE_STRIDE", "0")]).live_stride, 1, "at least 1");
    }

    #[test]
    fn precision_override_parses() {
        assert_eq!(with(&[("SKYLENS_PRECISION", "FP16")]).precision, Some(Precision::Fp16Autocast));
        assert_eq!(with(&[("SKYLENS_PRECISION", "bf16")]).precision, Some(Precision::Bf16));
        assert_eq!(with(&[("SKYLENS_PRECISION", "fp32")]).precision, Some(Precision::Fp32));
        assert_eq!(with(&[("SKYLENS_PRECISION", "auto")]).precision, None);
        assert_eq!(with(&[("SKYLENS_PRECISION", "int8")]).precision, None);
    }

    #[test]
    fn flags_and_numbers_parse_like_python() {
        for raw in ["1", "true", " YES ", "on"] {
            assert!(with(&[("SKYLENS_DEMO", raw)]).demo, "{raw}");
        }
        for raw in ["0", "no", "", "2"] {
            assert!(!with(&[("SKYLENS_DEMO", raw)]).demo, "{raw}");
        }
        let s = with(&[("SKYLENS_DEMO_STEP_SECONDS", "abc"), ("SKYLENS_DEMO_MIN_SECONDS", "0.5")]);
        assert_eq!(s.step_seconds, 0.0006);
        assert_eq!(s.min_seconds, 0.5);
        assert_eq!(with(&[("SKYLENS_DEMO_URL_BASE", "/x/y//")]).url_base, "/x/y");
    }

    #[test]
    fn anchor_falls_back_on_garbage() {
        assert_eq!(parse_anchor(Some("1,2,3,4")), Gps { lat: 1.0, lon: 2.0, alt: 3.0 });
        assert_eq!(parse_anchor(Some("1,2")), DEFAULT_ANCHOR);
        assert_eq!(parse_anchor(Some("a,b,c")), DEFAULT_ANCHOR);
    }

    #[test]
    fn demo_root_is_manifest_dir() {
        let s = with(&[("SKYLENS_DEMO_MANIFEST", "/tmp/demo/segments.json")]);
        assert!(s.demo_root().ends_with("demo"));
    }
}
