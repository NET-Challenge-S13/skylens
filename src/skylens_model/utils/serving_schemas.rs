//! Rust twin of `utils/serving_schemas.py`: serde mirrors of the model API wire
//! contract, plus the conversions to and from the gRPC messages in
//! `skylens/model/v1/model.proto`.
//!
//! Field names on the wire are IDENTICAL to the Python models (camelCase, no
//! aliasing), and every struct rejects unknown keys like pydantic
//! `extra="forbid"`: a field the core renamed should fail loudly here.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use skylens_shared::proto::model::v1 as pb;
use skylens_shared::proto::payload::v1 as payload;

// ---------------------------------------------------------------------------
// Literal tags (pydantic `Literal["..."] = "..."`)
// ---------------------------------------------------------------------------

macro_rules! literal_tag {
    ($name:ident, $value:literal) => {
        #[doc = concat!("The literal `\"", $value, "\"`; any other value is rejected.")]
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
        pub enum $name {
            #[default]
            #[serde(rename = $value)]
            Tag,
        }
    };
}

literal_tag!(TelemetryKind, "telemetry");
literal_tag!(DetectionKind, "detection");
literal_tag!(ReconResultKind, "recon-result");
literal_tag!(DetectResultKind, "detect-result");
literal_tag!(HealthStatus, "ok");

// ---------------------------------------------------------------------------
// Borrowed shapes (geo.ts / protocol.ts §2 §4 §5)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gps {
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct DroneTelemetry {
    #[serde(default)]
    pub kind: TelemetryKind,
    pub drone_id: i64,
    /// Formation station ("left" | "center" | "right"). Optional so a fix from an
    /// older drone build still validates; the compute side does not use it.
    #[serde(default)]
    pub station: Option<String>,
    pub gps: Gps,
    pub heading_deg: f64,
    pub speed: f64,
    pub battery_pct: f64,
    pub t: i64,
}

/// Where a reconstructed chunk lands in the shared frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SplatAlign {
    pub anchor: Option<Gps>,
    pub position: [f64; 3],
    pub rotation: [f64; 4],
    pub scale: [f64; 3],
}

impl Default for SplatAlign {
    fn default() -> Self {
        IDENTITY_ALIGN
    }
}

pub const IDENTITY_ALIGN: SplatAlign = SplatAlign {
    anchor: None,
    position: [0.0, 0.0, 0.0],
    rotation: [0.0, 0.0, 0.0, 1.0],
    scale: [1.0, 1.0, 1.0],
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DetectionCategory {
    Person,
    Danger,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetectionResult {
    #[serde(default)]
    pub kind: DetectionKind,
    pub id: String,
    pub category: DetectionCategory,
    pub gps: Gps,
    pub confidence: f64,
    pub label: String,
    pub segment: i64,
}

/// One video slice plus the poses covering it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobSource {
    pub uri: String,
    #[serde(default)]
    pub poses: Vec<DroneTelemetry>,
}

// ---------------------------------------------------------------------------
// §7 Model API
// ---------------------------------------------------------------------------

/// POST /recon/jobs
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ReconJobRequest {
    pub segment: i64,
    #[serde(default)]
    pub sources: Vec<JobSource>,
    pub steps: i64,
    /// The stretch of route this segment covers, sampled start to end; the
    /// metric scale a reconstruction from images does not have on its own.
    #[serde(default)]
    pub track: Vec<Gps>,
    /// Frame established by the FIRST segment, forced onto every later one.
    #[serde(default)]
    pub anchor_frame: Option<String>,
}

/// POST /detect/jobs
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetectJobRequest {
    pub segment: i64,
    #[serde(default)]
    pub sources: Vec<JobSource>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct JobAccepted {
    pub job_id: String,
    /// Server-side queue position at accept time (0 = starts immediately).
    pub queued: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobState {
    Queued,
    Running,
    Done,
    Failed,
}

impl JobState {
    pub const ALL: [JobState; 4] = [Self::Queued, Self::Running, Self::Done, Self::Failed];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Done => "done",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ReconJobResult {
    #[serde(default)]
    pub kind: ReconResultKind,
    pub segment: i64,
    pub steps: i64,
    pub url: String,
    pub bytes: u64,
    #[serde(default)]
    pub align: SplatAlign,
    /// Frame this job established; later jobs must be given it as anchorFrame.
    pub anchor_frame: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetectJobResult {
    #[serde(default)]
    pub kind: DetectResultKind,
    pub segment: i64,
    #[serde(default)]
    pub detections: Vec<DetectionResult>,
}

/// Discriminated by the `kind` tag each variant carries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JobResult {
    Recon(ReconJobResult),
    Detect(DetectJobResult),
}

/// GET /jobs/{job_id}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct JobStatus {
    pub job_id: String,
    pub state: JobState,
    pub progress: f64,
    #[serde(default)]
    pub result: Option<JobResult>,
    #[serde(default)]
    pub error: Option<String>,
}

/// GET /health: liveness plus what this process can actually compute.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Health {
    #[serde(default)]
    pub status: HealthStatus,
    pub version: String,
    pub demo: bool,
    pub torch: bool,
    pub cuda: bool,
    #[serde(default)]
    pub device: Option<String>,
    /// First CUDA failure after start; `cuda` is false from then on.
    #[serde(default)]
    pub gpu_error: Option<String>,
    #[serde(default)]
    pub jobs: BTreeMap<String, u64>,
}

// ---------------------------------------------------------------------------
// gRPC conversions (model.proto)
// ---------------------------------------------------------------------------

fn u32_of(v: i64) -> u32 {
    v.clamp(0, u32::MAX as i64) as u32
}

impl From<Gps> for pb::Gps {
    fn from(g: Gps) -> Self {
        pb::Gps {
            lat: g.lat,
            lon: g.lon,
            alt: g.alt,
        }
    }
}

impl From<pb::Gps> for Gps {
    fn from(g: pb::Gps) -> Self {
        Gps {
            lat: g.lat,
            lon: g.lon,
            alt: g.alt,
        }
    }
}

impl From<skylens_shared::geo::Gps> for Gps {
    fn from(g: skylens_shared::geo::Gps) -> Self {
        Gps {
            lat: g.lat,
            lon: g.lon,
            alt: g.alt,
        }
    }
}

impl From<Gps> for skylens_shared::geo::Gps {
    fn from(g: Gps) -> Self {
        skylens_shared::geo::Gps {
            lat: g.lat,
            lon: g.lon,
            alt: g.alt,
        }
    }
}

fn station_to_pb(station: Option<&str>) -> pb::DroneStation {
    match station {
        Some("left") => pb::DroneStation::Left,
        Some("center") => pb::DroneStation::Center,
        Some("right") => pb::DroneStation::Right,
        _ => pb::DroneStation::Unspecified,
    }
}

fn station_from_pb(raw: i32) -> Option<String> {
    match pb::DroneStation::try_from(raw).unwrap_or(pb::DroneStation::Unspecified) {
        pb::DroneStation::Left => Some("left".into()),
        pb::DroneStation::Center => Some("center".into()),
        pb::DroneStation::Right => Some("right".into()),
        pb::DroneStation::Unspecified => None,
    }
}

impl From<&DroneTelemetry> for pb::DroneTelemetry {
    fn from(t: &DroneTelemetry) -> Self {
        pb::DroneTelemetry {
            drone_id: u32_of(t.drone_id),
            station: station_to_pb(t.station.as_deref()) as i32,
            gps: Some(t.gps.into()),
            heading_deg: t.heading_deg,
            speed: t.speed,
            battery_pct: t.battery_pct,
            t_ms: t.t,
        }
    }
}

impl From<pb::DroneTelemetry> for DroneTelemetry {
    fn from(t: pb::DroneTelemetry) -> Self {
        DroneTelemetry {
            kind: TelemetryKind::Tag,
            drone_id: t.drone_id as i64,
            station: station_from_pb(t.station),
            gps: t.gps.map(Gps::from).unwrap_or(Gps {
                lat: 0.0,
                lon: 0.0,
                alt: 0.0,
            }),
            heading_deg: t.heading_deg,
            speed: t.speed,
            battery_pct: t.battery_pct,
            t: t.t_ms,
        }
    }
}

impl From<&JobSource> for pb::Source {
    fn from(s: &JobSource) -> Self {
        pb::Source {
            uri: s.uri.clone(),
            poses: s.poses.iter().map(Into::into).collect(),
        }
    }
}

impl From<pb::Source> for JobSource {
    fn from(s: pb::Source) -> Self {
        JobSource {
            uri: s.uri,
            poses: s.poses.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<&ReconJobRequest> for pb::ReconJobRequest {
    fn from(r: &ReconJobRequest) -> Self {
        pb::ReconJobRequest {
            segment: u32_of(r.segment),
            sources: r.sources.iter().map(Into::into).collect(),
            track: r.track.iter().copied().map(Into::into).collect(),
            steps: u32_of(r.steps),
            anchor_frame: r.anchor_frame.clone(),
        }
    }
}

impl From<pb::ReconJobRequest> for ReconJobRequest {
    fn from(r: pb::ReconJobRequest) -> Self {
        ReconJobRequest {
            segment: r.segment as i64,
            sources: r.sources.into_iter().map(Into::into).collect(),
            steps: r.steps as i64,
            track: r.track.into_iter().map(Into::into).collect(),
            anchor_frame: r.anchor_frame,
        }
    }
}

impl From<&DetectJobRequest> for pb::DetectJobRequest {
    fn from(r: &DetectJobRequest) -> Self {
        pb::DetectJobRequest {
            segment: u32_of(r.segment),
            sources: r.sources.iter().map(Into::into).collect(),
        }
    }
}

impl From<pb::DetectJobRequest> for DetectJobRequest {
    fn from(r: pb::DetectJobRequest) -> Self {
        DetectJobRequest {
            segment: r.segment as i64,
            sources: r.sources.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<&JobAccepted> for pb::JobAccepted {
    fn from(a: &JobAccepted) -> Self {
        pb::JobAccepted {
            job_id: a.job_id.clone(),
            queued: a.queued.min(u32::MAX as u64) as u32,
        }
    }
}

impl From<SplatAlign> for pb::Align {
    fn from(a: SplatAlign) -> Self {
        pb::Align {
            anchor: a.anchor.map(Into::into),
            position: a.position.to_vec(),
            rotation: a.rotation.to_vec(),
            scale: a.scale.to_vec(),
        }
    }
}

impl From<pb::Align> for SplatAlign {
    fn from(a: pb::Align) -> Self {
        let mut out = IDENTITY_ALIGN;
        out.anchor = a.anchor.map(Into::into);
        if let Ok(p) = <[f64; 3]>::try_from(a.position.as_slice()) {
            out.position = p;
        }
        if let Ok(r) = <[f64; 4]>::try_from(a.rotation.as_slice()) {
            out.rotation = r;
        }
        if let Ok(s) = <[f64; 3]>::try_from(a.scale.as_slice()) {
            out.scale = s;
        }
        out
    }
}

impl From<&ReconJobResult> for pb::ReconJobResult {
    fn from(r: &ReconJobResult) -> Self {
        pb::ReconJobResult {
            segment: u32_of(r.segment),
            steps: u32_of(r.steps),
            url: r.url.clone(),
            bytes: r.bytes,
            align: Some(r.align.into()),
            anchor_frame: r.anchor_frame.clone(),
        }
    }
}

impl From<pb::ReconJobResult> for ReconJobResult {
    fn from(r: pb::ReconJobResult) -> Self {
        ReconJobResult {
            kind: ReconResultKind::Tag,
            segment: r.segment as i64,
            steps: r.steps as i64,
            url: r.url,
            bytes: r.bytes,
            align: r.align.map(Into::into).unwrap_or_default(),
            anchor_frame: r.anchor_frame,
        }
    }
}

/// JSON `DetectionResult` -> `skylens.payload.v1.Detection`.
///
/// The JSON contract only knows `person | danger`; the payload enum splits
/// danger into fire / collapse / road_blocked. A JSON `danger` cannot say which,
/// so it goes out as UNSPECIFIED rather than as an invented hazard class.
impl From<&DetectionResult> for payload::Detection {
    fn from(d: &DetectionResult) -> Self {
        let category = match d.category {
            DetectionCategory::Person => payload::DetectionCategory::Person,
            DetectionCategory::Danger => payload::DetectionCategory::Unspecified,
        };
        payload::Detection {
            id: d.id.clone(),
            category: category as i32,
            gps: Some(d.gps.into()),
            confidence: d.confidence,
            label: d.label.clone(),
            segment: u32_of(d.segment),
            outline: Vec::new(),
            frames: 0,
        }
    }
}

impl From<payload::Detection> for DetectionResult {
    fn from(d: payload::Detection) -> Self {
        let category = match payload::DetectionCategory::try_from(d.category) {
            Ok(payload::DetectionCategory::Person) => DetectionCategory::Person,
            _ => DetectionCategory::Danger,
        };
        DetectionResult {
            kind: DetectionKind::Tag,
            id: d.id,
            category,
            gps: d.gps.map(Gps::from).unwrap_or(Gps {
                lat: 0.0,
                lon: 0.0,
                alt: 0.0,
            }),
            confidence: d.confidence,
            label: d.label,
            segment: d.segment as i64,
        }
    }
}

impl From<&DetectJobResult> for pb::DetectJobResult {
    fn from(r: &DetectJobResult) -> Self {
        use prost::Message;
        let batch = payload::DetectionBatch {
            detections: r.detections.iter().map(Into::into).collect(),
        };
        pb::DetectJobResult {
            segment: u32_of(r.segment),
            detections: batch.encode_to_vec(),
        }
    }
}

impl TryFrom<pb::DetectJobResult> for DetectJobResult {
    type Error = prost::DecodeError;

    fn try_from(r: pb::DetectJobResult) -> Result<Self, Self::Error> {
        use prost::Message;
        let batch = payload::DetectionBatch::decode(r.detections.as_slice())?;
        Ok(DetectJobResult {
            kind: DetectResultKind::Tag,
            segment: r.segment as i64,
            detections: batch.detections.into_iter().map(Into::into).collect(),
        })
    }
}

impl From<JobState> for pb::JobState {
    fn from(s: JobState) -> Self {
        match s {
            JobState::Queued => pb::JobState::Queued,
            JobState::Running => pb::JobState::Running,
            JobState::Done => pb::JobState::Done,
            JobState::Failed => pb::JobState::Failed,
        }
    }
}

impl From<&JobStatus> for pb::JobStatus {
    fn from(s: &JobStatus) -> Self {
        let result = s.result.as_ref().map(|r| match r {
            JobResult::Recon(r) => pb::job_status::Result::Recon(r.into()),
            JobResult::Detect(d) => pb::job_status::Result::Detect(d.into()),
        });
        pb::JobStatus {
            job_id: s.job_id.clone(),
            state: pb::JobState::from(s.state) as i32,
            progress: s.progress,
            result,
            error: s.error.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn recon_request_uses_camel_case_and_defaults() {
        let req: ReconJobRequest =
            serde_json::from_value(json!({"segment": 0, "steps": 1000, "anchorFrame": null}))
                .unwrap();
        assert_eq!(req.anchor_frame, None);
        assert!(req.sources.is_empty() && req.track.is_empty());

        let back = serde_json::to_value(&req).unwrap();
        let keys: Vec<_> = back.as_object().unwrap().keys().cloned().collect();
        for k in ["segment", "sources", "steps", "track", "anchorFrame"] {
            assert!(keys.contains(&k.to_string()), "missing {k} in {keys:?}");
        }
    }

    #[test]
    fn unknown_fields_are_rejected_at_every_level() {
        let top = json!({"segment": 0, "steps": 1, "extra": 1});
        assert!(serde_json::from_value::<ReconJobRequest>(top).is_err());

        let nested = json!({"segment": 0, "sources": [{"uri": "a", "bogus": true}]});
        assert!(serde_json::from_value::<DetectJobRequest>(nested).is_err());

        let snake = json!({"segment": 0, "steps": 1, "anchor_frame": "x"});
        assert!(serde_json::from_value::<ReconJobRequest>(snake).is_err());
    }

    #[test]
    fn telemetry_kind_literal_is_enforced() {
        let ok = json!({"droneId": 1, "gps": {"lat": 1.0, "lon": 2.0, "alt": 3.0},
            "headingDeg": 0.0, "speed": 1.0, "batteryPct": 90.0, "t": 5});
        let t: DroneTelemetry = serde_json::from_value(ok.clone()).unwrap();
        assert_eq!(t.kind, TelemetryKind::Tag);
        assert_eq!(serde_json::to_value(&t).unwrap()["kind"], "telemetry");

        let mut bad = ok;
        bad["kind"] = json!("detection");
        assert!(serde_json::from_value::<DroneTelemetry>(bad).is_err());
    }

    #[test]
    fn missing_required_field_is_rejected() {
        assert!(serde_json::from_value::<ReconJobRequest>(json!({"segment": 0})).is_err());
    }

    #[test]
    fn job_status_serializes_like_fastapi() {
        let status = JobStatus {
            job_id: "recon-abc".into(),
            state: JobState::Done,
            progress: 1.0,
            result: Some(JobResult::Recon(ReconJobResult {
                kind: ReconResultKind::Tag,
                segment: 0,
                steps: 1000,
                url: "/res/static/demo/x.ply".into(),
                bytes: 12,
                align: SplatAlign::default(),
                anchor_frame: "frame-1".into(),
            })),
            error: None,
        };
        let v = serde_json::to_value(&status).unwrap();
        assert_eq!(v["jobId"], "recon-abc");
        assert_eq!(v["state"], "done");
        assert_eq!(v["error"], serde_json::Value::Null);
        assert_eq!(v["result"]["kind"], "recon-result");
        assert_eq!(v["result"]["anchorFrame"], "frame-1");
        assert_eq!(
            v["result"]["align"]["rotation"],
            json!([0.0, 0.0, 0.0, 1.0])
        );
        assert_eq!(v["result"]["align"]["anchor"], serde_json::Value::Null);

        let round: JobStatus = serde_json::from_value(v).unwrap();
        assert_eq!(round, status);
    }

    #[test]
    fn detect_result_round_trips_through_untagged_union() {
        let r = JobResult::Detect(DetectJobResult {
            kind: DetectResultKind::Tag,
            segment: 2,
            detections: vec![],
        });
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["kind"], "detect-result");
        assert_eq!(serde_json::from_value::<JobResult>(v).unwrap(), r);
    }

    #[test]
    fn health_and_accepted_field_names() {
        let a = serde_json::to_value(JobAccepted {
            job_id: "j".into(),
            queued: 2,
        })
        .unwrap();
        assert_eq!(a, json!({"jobId": "j", "queued": 2}));

        let h = Health {
            status: HealthStatus::Tag,
            version: "0.1.0".into(),
            demo: true,
            torch: true,
            cuda: false,
            device: None,
            gpu_error: None,
            jobs: BTreeMap::new(),
        };
        let v = serde_json::to_value(h).unwrap();
        assert_eq!(v["status"], "ok");
        assert!(v.get("device").is_some());
        assert!(v.get("gpu_error").is_some());
    }

    #[test]
    fn proto_round_trips() {
        let req = ReconJobRequest {
            segment: 3,
            sources: vec![JobSource {
                uri: "srt://x".into(),
                poses: vec![DroneTelemetry {
                    kind: TelemetryKind::Tag,
                    drone_id: 2,
                    station: Some("center".into()),
                    gps: Gps {
                        lat: 1.0,
                        lon: 2.0,
                        alt: 3.0,
                    },
                    heading_deg: 90.0,
                    speed: 4.0,
                    battery_pct: 50.0,
                    t: 123,
                }],
            }],
            steps: 7000,
            track: vec![Gps {
                lat: 1.0,
                lon: 2.0,
                alt: 3.0,
            }],
            anchor_frame: Some("frame-x".into()),
        };
        let back = ReconJobRequest::from(pb::ReconJobRequest::from(&req));
        assert_eq!(back, req);

        let det = DetectJobResult {
            kind: DetectResultKind::Tag,
            segment: 1,
            detections: vec![DetectionResult {
                kind: DetectionKind::Tag,
                id: "p1".into(),
                category: DetectionCategory::Person,
                gps: Gps {
                    lat: 1.0,
                    lon: 2.0,
                    alt: 3.0,
                },
                confidence: 0.9,
                label: "x".into(),
                segment: 1,
            }],
        };
        let back = DetectJobResult::try_from(pb::DetectJobResult::from(&det)).unwrap();
        assert_eq!(back, det);
    }
}
