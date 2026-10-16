//! Rust twin of `app.py`: REST on 8100 (same endpoints, same JSON) and the gRPC
//! `skylens.model.v1.ModelService` on 8101. Both transports share one job
//! registry and one pipeline, so a job submitted over gRPC can be polled over
//! REST and the other way round.
//!
//! ```text
//! POST /recon/jobs   ReconJobRequest  -> JobAccepted
//! POST /detect/jobs  DetectJobRequest -> JobAccepted
//! GET  /jobs/{id}                     -> JobStatus (404 when unknown)
//! GET  /health                        -> liveness + libtorch/CUDA availability
//! ```
//!
//! Jobs are held in memory only. A restart loses every job; that is the
//! decision in COMPONENTS.md §7, not an oversight.

use std::sync::{Arc, OnceLock};

use anyhow::Context as _;
use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::{FromRequest, Path, Request, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use skylens_shared::proto::model::v1 as pb;
use skylens_shared::proto::model::v1::model_service_server::{ModelService, ModelServiceServer};
use tokio::sync::watch;
use tokio_stream::StreamExt as _;
use tokio_stream::wrappers::ReceiverStream;
use tonic::codegen::BoxStream;
use tonic::{Status, Streaming};

use crate::detect::Detector;
use crate::models::skylens::{Engine, EngineConfig, PushOutcome};
use crate::utils::serving_config::{Settings, settings};
use crate::utils::serving_pipeline::{LiveEngine, Pipeline};
use crate::utils::serving_registry::{JobKind, JobRegistry, Report, Runner, RunnerFuture};
use crate::utils::serving_schemas::{
    DetectJobRequest, Health, HealthStatus, JobAccepted, JobResult, ReconJobRequest,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// gRPC frame sets carry encoded camera images; the tonic default (4 MiB) is
/// smaller than one set from three cameras.
const MAX_GRPC_MESSAGE: usize = 64 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Shared state and job submission (used by REST and gRPC alike)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<JobRegistry>,
    pub pipeline: Arc<Pipeline>,
}

impl AppState {
    pub fn new(pipeline: Pipeline) -> Self {
        AppState { registry: Arc::new(JobRegistry::default()), pipeline: Arc::new(pipeline) }
    }
}

/// Queue a reconstruction of one capture segment at one refinement level.
pub fn submit_recon(state: &AppState, req: ReconJobRequest) -> JobAccepted {
    let pipeline = state.pipeline.clone();
    let runner: Runner = Box::new(move |report: Report| -> RunnerFuture {
        Box::pin(async move { pipeline.run_recon(req, report).await.map(JobResult::Recon) })
    });
    let (job_id, queued) = state.registry.submit(JobKind::Recon, runner);
    JobAccepted { job_id, queued }
}

/// Queue people / danger-zone detection over one capture segment.
pub fn submit_detect(state: &AppState, req: DetectJobRequest) -> JobAccepted {
    let pipeline = state.pipeline.clone();
    let runner: Runner = Box::new(move |report: Report| -> RunnerFuture {
        Box::pin(async move { pipeline.run_detect(req, report).await.map(JobResult::Detect) })
    });
    let (job_id, queued) = state.registry.submit(JobKind::Detect, runner);
    JobAccepted { job_id, queued }
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuProbe {
    /// libtorch is linked and answers.
    pub torch: bool,
    pub cuda: bool,
    pub device: Option<String>,
}

/// Cached: the answer cannot change while the process lives. Blocking on the
/// first call (libtorch initialises CUDA), so call it off the async threads.
pub fn gpu_probe() -> GpuProbe {
    static PROBE: OnceLock<GpuProbe> = OnceLock::new();
    PROBE
        .get_or_init(|| match std::panic::catch_unwind(tch::Cuda::is_available) {
            Ok(true) => GpuProbe { torch: true, cuda: true, device: Some("cuda:0".into()) },
            Ok(false) => GpuProbe { torch: true, cuda: false, device: None },
            Err(_) => {
                tracing::error!("libtorch cuda probe panicked");
                GpuProbe::default()
            }
        })
        .clone()
}

/// First CUDA failure after start. The device does not come back without a
/// restart (driver reset, GPU dropped off the bus), and cumulus3d and the UNet
/// fall back to the CPU on their own, so without this the server would report
/// CUDA while running many times slower.
static GPU_FAULT: OnceLock<String> = OnceLock::new();

/// How often the GPU watch runs its check.
pub const GPU_WATCH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

pub fn gpu_fault() -> Option<String> {
    GPU_FAULT.get().cloned()
}

fn record_gpu_fault(err: &str) {
    // libtorch errors carry a long hint after the first line.
    let first = err.lines().next().unwrap_or(err).trim().to_string();
    if GPU_FAULT.set(first.clone()).is_ok() {
        tracing::error!("GPU lost: {first}. Reconstruction and the UNet continue on the CPU, much slower");
    }
}

/// One tiny libtorch op on the GPU. Once the device is lost every call fails.
fn cuda_check() -> Result<(), String> {
    std::panic::catch_unwind(|| {
        tch::Tensor::f_ones([4], (tch::Kind::Float, tch::Device::Cuda(0)))
            .and_then(|t| t.f_sum(tch::Kind::Float))
            .and_then(|t| t.f_double_value(&[]))
            .map(|_| ())
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|_| Err("CUDA check panicked".into()))
}

/// Checks the GPU every [`GPU_WATCH_INTERVAL`] until the first failure.
/// Nothing to watch when CUDA was not available at start.
pub fn spawn_gpu_watch() {
    if !gpu_probe().cuda {
        return;
    }
    let spawned = std::thread::Builder::new().name("skylens-gpu-watch".into()).spawn(|| loop {
        std::thread::sleep(GPU_WATCH_INTERVAL);
        if let Err(e) = cuda_check() {
            record_gpu_fault(&e);
            return;
        }
    });
    if let Err(e) = spawned {
        tracing::warn!("GPU watch not started: {e}");
    }
}

/// What `/health` reports about the GPU: CUDA only while no failure was seen,
/// and the failure once there was one.
pub fn gpu_state(probe: &GpuProbe, fault: Option<String>) -> (bool, Option<String>) {
    match fault {
        Some(e) if probe.cuda => (false, Some(e)),
        _ => (probe.cuda, None),
    }
}

/// Liveness plus what this process can actually compute.
pub async fn health(state: &AppState) -> Health {
    let probe = tokio::task::spawn_blocking(gpu_probe).await.unwrap_or_default();
    let (cuda, gpu_error) = gpu_state(&probe, gpu_fault());
    let mut jobs = state.registry.counts();
    jobs.insert("frames".into(), state.pipeline.frames.count() as u64);
    Health {
        status: HealthStatus::Tag,
        version: VERSION.into(),
        demo: state.pipeline.settings.demo,
        torch: probe.torch,
        cuda,
        device: probe.device,
        gpu_error,
        jobs,
    }
}

// ---------------------------------------------------------------------------
// REST
// ---------------------------------------------------------------------------

/// JSON body extractor that answers every bad body with 422, like FastAPI does
/// for a pydantic model with `extra="forbid"` (axum's own `Json` uses 400 for
/// syntax errors and 415 for a missing content type).
pub struct StrictJson<T>(pub T);

fn unprocessable(kind: &str, loc: Vec<Value>, msg: String) -> Response {
    let mut path = vec![json!("body")];
    path.extend(loc);
    let body = json!({"detail": [{"type": kind, "loc": path, "msg": msg}]});
    (StatusCode::UNPROCESSABLE_ENTITY, Json(body)).into_response()
}

/// The name between the first pair of backticks in a serde message.
fn quoted_field(msg: &str) -> Option<String> {
    let start = msg.find('`')? + 1;
    let len = msg[start..].find('`')?;
    Some(msg[start..start + len].to_string())
}

fn json_rejection(err: &serde_json::Error) -> Response {
    use serde_json::error::Category;
    let msg = err.to_string();
    match err.classify() {
        Category::Syntax | Category::Eof | Category::Io => {
            unprocessable("json_invalid", vec![], format!("JSON decode error: {msg}"))
        }
        Category::Data => {
            if msg.starts_with("unknown field") {
                let loc = quoted_field(&msg).map(Value::from).into_iter().collect();
                unprocessable("extra_forbidden", loc, format!("Extra inputs are not permitted ({msg})"))
            } else if msg.starts_with("missing field") {
                let loc = quoted_field(&msg).map(Value::from).into_iter().collect();
                unprocessable("missing", loc, format!("Field required ({msg})"))
            } else {
                unprocessable("value_error", vec![], msg)
            }
        }
    }
}

fn is_json_content_type(raw: &str) -> bool {
    let mime = raw.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
    match mime.split_once('/') {
        Some((_, sub)) => sub == "json" || sub.ends_with("+json"),
        None => false,
    }
}

impl<T, S> FromRequest<S> for StrictJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        // FastAPI parses a body with no content type as JSON and refuses a body
        // with a non-JSON one; both refusals are 422.
        if let Some(ct) = req.headers().get(header::CONTENT_TYPE) {
            if !is_json_content_type(ct.to_str().unwrap_or("")) {
                return Err(unprocessable(
                    "model_attributes_type",
                    vec![],
                    "Input should be a valid dictionary or object to extract fields from".into(),
                ));
            }
        }
        let bytes = Bytes::from_request(req, state).await.map_err(IntoResponse::into_response)?;
        serde_json::from_slice::<T>(&bytes).map(StrictJson).map_err(|e| json_rejection(&e))
    }
}

async fn post_recon_job(
    State(state): State<AppState>,
    StrictJson(req): StrictJson<ReconJobRequest>,
) -> Json<JobAccepted> {
    Json(submit_recon(&state, req))
}

async fn post_detect_job(
    State(state): State<AppState>,
    StrictJson(req): StrictJson<DetectJobRequest>,
) -> Json<JobAccepted> {
    Json(submit_detect(&state, req))
}

async fn get_job(State(state): State<AppState>, Path(job_id): Path<String>) -> Response {
    match state.registry.status(&job_id) {
        Some(status) => Json(status).into_response(),
        None => (StatusCode::NOT_FOUND, Json(json!({"detail": format!("unknown job {job_id}")})))
            .into_response(),
    }
}

async fn get_health(State(state): State<AppState>) -> Json<Health> {
    Json(health(&state).await)
}

async fn not_found() -> Response {
    (StatusCode::NOT_FOUND, Json(json!({"detail": "Not Found"}))).into_response()
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/recon/jobs", post(post_recon_job))
        .route("/detect/jobs", post(post_detect_job))
        .route("/jobs/{job_id}", get(get_job))
        .route("/health", get(get_health))
        .fallback(not_found)
        .with_state(state)
}

// ---------------------------------------------------------------------------
// gRPC
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ModelGrpc {
    pub state: AppState,
}

fn engine_status(state: &AppState) -> Result<Arc<Engine>, Status> {
    state.pipeline.engine().map_err(|e| Status::unavailable(e.to_string()))
}

#[tonic::async_trait]
impl ModelService for ModelGrpc {
    async fn submit_recon(
        &self,
        request: tonic::Request<pb::ReconJobRequest>,
    ) -> Result<tonic::Response<pb::JobAccepted>, Status> {
        let accepted = submit_recon(&self.state, request.into_inner().into());
        Ok(tonic::Response::new((&accepted).into()))
    }

    async fn submit_detect(
        &self,
        request: tonic::Request<pb::DetectJobRequest>,
    ) -> Result<tonic::Response<pb::JobAccepted>, Status> {
        let accepted = submit_detect(&self.state, request.into_inner().into());
        Ok(tonic::Response::new((&accepted).into()))
    }

    async fn get_job(
        &self,
        request: tonic::Request<pb::GetJobRequest>,
    ) -> Result<tonic::Response<pb::JobStatus>, Status> {
        let job_id = request.into_inner().job_id;
        match self.state.registry.status(&job_id) {
            Some(status) => Ok(tonic::Response::new((&status).into())),
            None => Err(Status::not_found(format!("unknown job {job_id}"))),
        }
    }

    async fn health(
        &self,
        _request: tonic::Request<pb::HealthRequest>,
    ) -> Result<tonic::Response<pb::HealthResponse>, Status> {
        let h = health(&self.state).await;
        Ok(tonic::Response::new(pb::HealthResponse {
            ok: true,
            gpu: h.cuda,
            version: h.version,
            gpu_error: h.gpu_error.unwrap_or_default(),
        }))
    }

    async fn push_frames(
        &self,
        request: tonic::Request<Streaming<pb::FrameSet>>,
    ) -> Result<tonic::Response<pb::PushFramesSummary>, Status> {
        let engine = engine_status(&self.state)?;
        let mut stream = request.into_inner();
        let mut summary = pb::PushFramesSummary::default();
        while let Some(set) = stream.message().await? {
            let engine = engine.clone();
            let started = std::time::Instant::now();
            let outcome = tokio::task::spawn_blocking(move || engine.push(set))
                .await
                .map_err(|e| Status::internal(format!("frame push panicked: {e}")))?
                .map_err(|e| {
                    Status::internal(format!(
                        "{e:#} (after ingested={} ignored={} pending={})",
                        summary.ingested, summary.ignored, summary.pending
                    ))
                })?;
            tracing::debug!("frame set {outcome:?} in {:.2}s", started.elapsed().as_secs_f64());
            match outcome {
                PushOutcome::Ingested => summary.ingested += 1,
                PushOutcome::Ignored => summary.ignored += 1,
                PushOutcome::Pending => summary.pending += 1,
            }
        }
        Ok(tonic::Response::new(summary))
    }

    type SubscribeChunksStream = BoxStream<pb::Chunk>;

    async fn subscribe_chunks(
        &self,
        request: tonic::Request<pb::SubscribeRequest>,
    ) -> Result<tonic::Response<Self::SubscribeChunksStream>, Status> {
        let engine = engine_status(&self.state)?;
        let after = request.into_inner().after_sequence;
        let rx = tokio::task::spawn_blocking(move || engine.subscribe(after))
            .await
            .map_err(|e| Status::internal(format!("subscribe panicked: {e}")))?;
        let stream = ReceiverStream::new(rx).map(Ok::<pb::Chunk, Status>);
        Ok(tonic::Response::new(Box::pin(stream) as Self::SubscribeChunksStream))
    }
}

// ---------------------------------------------------------------------------
// Live engine
// ---------------------------------------------------------------------------

/// Build the reconstruction engine with the SkyLensNet detector when a
/// checkpoint is configured. Blocking (loads libtorch weights).
pub fn build_engine(settings: &Settings) -> anyhow::Result<Engine> {
    use crate::models::skylensnet::modeling_skylensnet::{DetectorOptions, Precision, SkyLensNetDetector};

    let cuda = gpu_probe().cuda;
    let gpu = settings.gpu.unwrap_or(cuda);
    let detector: Option<Arc<dyn Detector>> = match &settings.checkpoint {
        None => {
            tracing::warn!("SKYLENS_CHECKPOINT unset: live engine runs without a detector");
            None
        }
        Some(path) => {
            let device = if cuda { tch::Device::Cuda(0) } else { tch::Device::Cpu };
            let options = DetectorOptions {
                precision: settings.precision.unwrap_or(Precision::Auto),
                ..Default::default()
            };
            match SkyLensNetDetector::load_with_options(path, device, options) {
                Ok(det) => {
                    tracing::info!("SkyLensNet loaded from {} on {device:?}", path.display());
                    Some(Arc::new(det) as Arc<dyn Detector>)
                }
                Err(err) => {
                    // Reconstruction is still real without detection; say loudly
                    // that detection is off rather than refusing to start.
                    tracing::error!(
                        "SkyLensNet checkpoint {} failed to load, running without a detector: {err:#}",
                        path.display()
                    );
                    None
                }
            }
        }
    };
    std::fs::create_dir_all(&settings.work_dir)
        .with_context(|| format!("creating work dir {}", settings.work_dir.display()))?;
    Engine::new(
        EngineConfig {
            anchor: settings.anchor,
            cameras: settings.cameras.clone(),
            zone_span: settings.zone_span,
            zone_overlap: settings.zone_overlap,
            work_dir: settings.work_dir.clone(),
            gpu,
        },
        detector,
    )
}

/// The engine for live mode; demo mode never builds it.
pub async fn build_live(settings: &Settings) -> LiveEngine {
    if settings.demo {
        return LiveEngine::Disabled;
    }
    let owned = settings.clone();
    match tokio::task::spawn_blocking(move || build_engine(&owned)).await {
        Ok(Ok(engine)) => LiveEngine::Ready(Arc::new(engine)),
        Ok(Err(err)) => {
            tracing::error!("live engine failed to start: {err:#}");
            LiveEngine::Failed(format!("{err:#}"))
        }
        Err(join) => {
            tracing::error!("live engine construction panicked: {join}");
            LiveEngine::Failed(format!("engine construction panicked: {join}"))
        }
    }
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

async fn wait_stop(mut rx: watch::Receiver<bool>) {
    let _ = rx.wait_for(|stop| *stop).await;
}

/// Run both servers until Ctrl-C.
pub async fn serve(settings: Settings) -> anyhow::Result<()> {
    tracing::info!(
        "skylens_model api up (demo={}, manifest={})",
        settings.demo,
        if settings.demo { settings.manifest.display().to_string() } else { "-".into() }
    );
    let live = build_live(&settings).await;
    if !settings.demo {
        spawn_gpu_watch();
    }
    let state = AppState::new(Pipeline::new(settings.clone(), live));
    state.registry.start();

    let (stop_tx, stop_rx) = watch::channel(false);
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutting down");
        let _ = stop_tx.send(true);
    });

    let listener = tokio::net::TcpListener::bind(settings.rest_addr)
        .await
        .with_context(|| format!("binding REST on {}", settings.rest_addr))?;
    tracing::info!("REST on {}", settings.rest_addr);
    tracing::info!("gRPC skylens.model.v1.ModelService on {}", settings.grpc_addr);

    let rest = axum::serve(listener, router(state.clone()))
        .with_graceful_shutdown(wait_stop(stop_rx.clone()));
    let grpc_service = ModelServiceServer::new(ModelGrpc { state: state.clone() })
        .max_decoding_message_size(MAX_GRPC_MESSAGE)
        .max_encoding_message_size(MAX_GRPC_MESSAGE);
    let grpc = tonic::transport::Server::builder()
        .add_service(grpc_service)
        .serve_with_shutdown(settings.grpc_addr, wait_stop(stop_rx));

    let outcome = tokio::try_join!(
        async { rest.await.context("REST server") },
        async { grpc.await.context("gRPC server") },
    );
    state.registry.stop();
    outcome.map(|_| ())
}

/// On Windows the linker drops libtorch's CUDA library because nothing references it
/// directly, so `tch::Cuda::is_available()` reports false even with a CUDA build of libtorch.
/// Loading it before the first tch call registers the CUDA backend.
#[cfg(windows)]
fn load_torch_cuda() {
    use windows_sys::Win32::System::LibraryLoader::LoadLibraryA;
    // SAFETY: a NUL-terminated library name; a failed load just leaves us on CPU.
    let handle = unsafe { LoadLibraryA(c"torch_cuda.dll".as_ptr().cast()) };
    if handle.is_null() {
        tracing::warn!("torch_cuda.dll not found; UNet inference stays on CPU");
    }
}

pub fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    #[cfg(windows)]
    load_torch_cuda();
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    runtime.block_on(serve(settings().clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use std::collections::HashMap;
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn extract<T: DeserializeOwned>(body: &str, ct: Option<&str>) -> Result<T, (u16, Value)> {
        let mut builder = axum::http::Request::builder().method("POST").uri("/x");
        if let Some(ct) = ct {
            builder = builder.header(header::CONTENT_TYPE, ct);
        }
        let req = builder.body(Body::from(body.to_string())).unwrap();
        match StrictJson::<T>::from_request(req, &()).await {
            Ok(StrictJson(v)) => Ok(v),
            Err(resp) => {
                let status = resp.status().as_u16();
                let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
                Err((status, serde_json::from_slice(&bytes).unwrap()))
            }
        }
    }

    #[test]
    fn gpu_fault_turns_cuda_off_and_is_reported() {
        let cuda = GpuProbe { torch: true, cuda: true, device: Some("cuda:0".into()) };
        assert_eq!(gpu_state(&cuda, None), (true, None));
        assert_eq!(gpu_state(&cuda, Some("lost".into())), (false, Some("lost".into())));
        // Without CUDA at start there is nothing to lose.
        let cpu = GpuProbe { torch: true, cuda: false, device: None };
        assert_eq!(gpu_state(&cpu, Some("lost".into())), (false, None));
    }

    #[tokio::test]
    async fn extra_fields_are_422_with_fastapi_detail() {
        let ct = Some("application/json");
        let err = extract::<ReconJobRequest>(r#"{"segment":0,"steps":1,"bogus":1}"#, ct)
            .await
            .unwrap_err();
        assert_eq!(err.0, 422);
        assert_eq!(err.1["detail"][0]["type"], "extra_forbidden");
        assert_eq!(err.1["detail"][0]["loc"], json!(["body", "bogus"]));
    }

    #[tokio::test]
    async fn malformed_missing_and_wrong_type_are_422() {
        let ct = Some("application/json");
        assert_eq!(extract::<ReconJobRequest>("{not json", ct).await.unwrap_err().0, 422);
        assert_eq!(extract::<ReconJobRequest>("", ct).await.unwrap_err().0, 422);
        let missing = extract::<ReconJobRequest>(r#"{"segment":0}"#, ct).await.unwrap_err();
        assert_eq!(missing.0, 422);
        assert_eq!(missing.1["detail"][0]["type"], "missing");
        let wrong = extract::<DetectJobRequest>(r#"{"segment":"x"}"#, ct).await.unwrap_err();
        assert_eq!(wrong.0, 422);
        let text = extract::<DetectJobRequest>(r#"{"segment":0}"#, Some("text/plain")).await;
        assert_eq!(text.unwrap_err().0, 422);
    }

    #[tokio::test]
    async fn valid_body_parses_with_or_without_content_type() {
        let ok: DetectJobRequest = extract(r#"{"segment":2}"#, None).await.unwrap();
        assert_eq!(ok.segment, 2);
        let ok: ReconJobRequest = extract(
            r#"{"segment":0,"steps":1000,"anchorFrame":null,"sources":[]}"#,
            Some("application/json; charset=utf-8"),
        )
        .await
        .unwrap();
        assert_eq!(ok.steps, 1000);
    }

    // -- end to end over a real socket ---------------------------------------

    struct TempDemo(PathBuf);
    impl Drop for TempDemo {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn demo_settings() -> (TempDemo, Settings) {
        let dir = std::env::temp_dir()
            .join(format!("skylens-app-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = json!({"segments": [
            {"levels": [{"level": 0, "steps": 250, "url": "seg0.ply"}], "extent": 1.0}
        ]});
        std::fs::write(dir.join("segments.json"), manifest.to_string()).unwrap();
        std::fs::write(dir.join("seg0.ply"), b"0123456789").unwrap();
        let env: HashMap<&str, String> = HashMap::from([
            ("SKYLENS_DEMO", "1".to_string()),
            ("SKYLENS_DEMO_MANIFEST", dir.join("segments.json").display().to_string()),
            ("SKYLENS_DEMO_MIN_SECONDS", "0.02".to_string()),
            ("SKYLENS_DEMO_MAX_SECONDS", "0.1".to_string()),
        ]);
        let settings = Settings::from_lookup(|k| env.get(k).cloned());
        (TempDemo(dir), settings)
    }

    async fn http(addr: SocketAddr, method: &str, path: &str, body: &str) -> (u16, Value) {
        let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
        let req = format!(
            "{method} {path} HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        sock.write_all(req.as_bytes()).await.unwrap();
        let mut buf = Vec::new();
        sock.read_to_end(&mut buf).await.unwrap();
        let text = String::from_utf8_lossy(&buf).to_string();
        let status: u16 = text[9..12].parse().unwrap();
        let payload = text.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or("");
        (status, serde_json::from_str(payload).unwrap_or(Value::Null))
    }

    #[tokio::test]
    async fn rest_end_to_end_in_demo_mode() {
        let (_tmp, settings) = demo_settings();
        let state = AppState::new(Pipeline::new(settings, LiveEngine::Disabled));
        state.registry.start();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(axum::serve(listener, router(state.clone())).into_future());

        let (code, body) = http(addr, "POST", "/recon/jobs", r#"{"segment":0,"steps":1000,"anchorFrame":null,"sources":[]}"#).await;
        assert_eq!(code, 200, "{body}");
        assert_eq!(body["queued"], 0);
        let job_id = body["jobId"].as_str().unwrap().to_string();
        assert!(job_id.starts_with("recon-"));

        let mut last = Value::Null;
        for _ in 0..200 {
            let (code, status) = http(addr, "GET", &format!("/jobs/{job_id}"), "").await;
            assert_eq!(code, 200);
            last = status;
            if last["state"] == "done" || last["state"] == "failed" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(last["state"], "done", "{last}");
        assert_eq!(last["progress"], 1.0);
        assert_eq!(last["error"], Value::Null);
        assert_eq!(last["result"]["kind"], "recon-result");
        assert_eq!(last["result"]["url"], "/res/static/demo/seg0.ply");
        assert_eq!(last["result"]["bytes"], 10);
        assert!(last["result"]["anchorFrame"].as_str().unwrap().starts_with("frame-"));

        let (code, body) = http(addr, "POST", "/detect/jobs", r#"{"segment":2,"extra":true}"#).await;
        assert_eq!(code, 422, "{body}");

        let (code, body) = http(addr, "GET", "/jobs/recon-unknown", "").await;
        assert_eq!(code, 404);
        assert_eq!(body["detail"], "unknown job recon-unknown");

        let (code, body) = http(addr, "GET", "/health", "").await;
        assert_eq!(code, 200);
        assert_eq!(body["status"], "ok");
        assert_eq!(body["demo"], true);
        assert_eq!(body["jobs"]["done"], 1);
        assert_eq!(body["jobs"]["frames"], 1);
        for key in ["version", "torch", "cuda", "device"] {
            assert!(body.get(key).is_some(), "missing {key}");
        }

        server.abort();
        state.registry.stop();
    }

    #[tokio::test]
    async fn grpc_shares_the_registry_with_rest() {
        let (_tmp, settings) = demo_settings();
        let state = AppState::new(Pipeline::new(settings, LiveEngine::Disabled));
        state.registry.start();
        let svc = ModelGrpc { state: state.clone() };
        let accepted = svc
            .submit_detect(tonic::Request::new(pb::DetectJobRequest { segment: 0, sources: vec![] }))
            .await
            .unwrap()
            .into_inner();
        assert!(accepted.job_id.starts_with("detect-"));
        // Visible through the REST-side registry.
        assert!(state.registry.status(&accepted.job_id).is_some());

        let missing = svc
            .get_job(tonic::Request::new(pb::GetJobRequest { job_id: "nope".into() }))
            .await
            .unwrap_err();
        assert_eq!(missing.code(), tonic::Code::NotFound);

        // Demo mode has no live engine: the live RPCs say so.
        let sub = svc
            .subscribe_chunks(tonic::Request::new(pb::SubscribeRequest { after_sequence: 0 }))
            .await;
        assert_eq!(sub.err().map(|s| s.code()), Some(tonic::Code::Unavailable));
        state.registry.stop();
    }
}
