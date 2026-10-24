//! Localhost HTTP bridge for SkyLens -> DJI Fly missions.
//!
//! Port of `legacy/mission_bridge/server.py`: same routes, JSON bodies, status
//! codes, CORS rule and environment variables. Routing is done by hand in one
//! handler, as the Python `BaseHTTPRequestHandler` did, so the order of checks
//! (unknown path -> origin -> body) and every error shape stay identical.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context as _;
use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::{Request, State};
use axum::http::header::{
    CACHE_CONTROL, CONTENT_DISPOSITION, CONTENT_LENGTH, CONTENT_TYPE, ORIGIN, VARY,
};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
use axum::response::Response;
use percent_encoding::percent_decode_str;
use regex::Regex;
use serde::Serialize;
use serde_json::{Value, json};

use crate::cloud::{CloudClient, CloudError};
use crate::formation::build_formation;
use crate::mission::{MissionError, build_mission};
use crate::paths::expand_user;

pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 8091;
pub const DEFAULT_ALLOWED_ORIGINS: &str =
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080";
pub const MAX_BODY: usize = 1024 * 1024;

/// Everything the bridge reads from the environment at start-up.
#[derive(Debug, Clone)]
pub struct Settings {
    pub host: String,
    pub port: u16,
    pub template: PathBuf,
    pub output: PathBuf,
    pub install_target: Option<String>,
    pub allowed_origins: HashSet<String>,
}

impl Settings {
    /// Read the `SKYLENS_MISSION_*` / `SKYLENS_DJI_*` variables. Relative
    /// defaults resolve against `root`; explicit values are used as given
    /// (relative to the working directory), like the Python bridge.
    pub fn from_lookup(root: &Path, get: impl Fn(&str) -> Option<String>) -> anyhow::Result<Self> {
        let var = |k: &str| get(k).filter(|v| !v.trim().is_empty());
        let host = var("SKYLENS_MISSION_HOST").unwrap_or_else(|| DEFAULT_HOST.to_owned());
        let port = match var("SKYLENS_MISSION_PORT") {
            Some(p) => p.trim().parse::<u16>().with_context(|| format!("SKYLENS_MISSION_PORT={p}"))?,
            None => DEFAULT_PORT,
        };
        let template = var("SKYLENS_DJI_TEMPLATE_KMZ")
            .map(|v| expand_user(&v))
            .unwrap_or_else(|| root.join("config").join("lito-x1-template.kmz"));
        let output = var("SKYLENS_MISSION_OUTPUT")
            .map(|v| expand_user(&v))
            .unwrap_or_else(|| root.join("output").join("missions"));
        let install_target = var("SKYLENS_DJI_INSTALL_TARGET").map(|v| v.trim().to_owned());
        let origins = get("SKYLENS_MISSION_ALLOWED_ORIGINS")
            .unwrap_or_else(|| DEFAULT_ALLOWED_ORIGINS.to_owned());
        let allowed_origins = origins
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();
        Ok(Self { host, port, template, output, install_target, allowed_origins })
    }
}

/// A request failure, mapped to the Python status codes.
#[derive(Debug)]
pub enum BridgeError {
    /// Bad input, missing template, cloud requested but not configured: 400.
    Mission(String),
    /// DJI cloud rejected or unreachable: 502.
    Cloud(String),
    /// Anything else: 500 with a generic body, details only in the log.
    Internal(anyhow::Error),
}

impl From<anyhow::Error> for BridgeError {
    fn from(err: anyhow::Error) -> Self {
        if let Some(m) = err.downcast_ref::<MissionError>() {
            return BridgeError::Mission(m.to_string());
        }
        if let Some(c) = err.downcast_ref::<CloudError>() {
            return BridgeError::Cloud(c.to_string());
        }
        BridgeError::Internal(err)
    }
}

impl From<MissionError> for BridgeError {
    fn from(err: MissionError) -> Self {
        BridgeError::Mission(err.0)
    }
}

impl From<CloudError> for BridgeError {
    fn from(err: CloudError) -> Self {
        BridgeError::Cloud(err.to_string())
    }
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::Mission(m) | BridgeError::Cloud(m) => f.write_str(m),
            BridgeError::Internal(e) => write!(f, "{e:#}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CloudRef {
    pub mission_uuid: String,
    pub verified: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SingleResponse {
    pub ok: bool,
    pub file_name: String,
    pub download_url: String,
    pub sha256: String,
    pub waypoint_count: usize,
    pub route_distance_m: f64,
    pub takeoff_alt_msl: f64,
    pub installed: bool,
    pub install_target: Option<String>,
    pub warnings: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cloud: Option<CloudRef>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct MemberRow {
    pub station: String,
    pub file_name: String,
    pub download_url: String,
    pub sha256: String,
    pub waypoint_count: usize,
    pub route_distance_m: f64,
    pub takeoff_alt_msl: f64,
    pub warnings: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cloud: Option<CloudRef>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FormationResponse {
    pub ok: bool,
    pub installed: bool,
    pub members: Vec<MemberRow>,
}

/// Python `round(x, 1)` for the response numbers.
fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// `_safe_name`: a file-name-safe slug of the request's `name`, at most 60 chars.
pub fn safe_name(value: Option<&Value>) -> String {
    // str(value or "route") with Python truthiness
    let raw = match value {
        None | Some(Value::Null) | Some(Value::Bool(false)) => "route".to_owned(),
        Some(Value::Bool(true)) => "True".to_owned(),
        Some(Value::Number(n)) if n.as_f64() == Some(0.0) => "route".to_owned(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::String(s)) if s.is_empty() => "route".to_owned(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) if a.is_empty() => "route".to_owned(),
        Some(Value::Object(o)) if o.is_empty() => "route".to_owned(),
        Some(other) => other.to_string(),
    };
    let re = Regex::new(r"[^A-Za-z0-9._-]+").expect("static pattern");
    let clean = re.replace_all(&raw, "-");
    let clean = clean.trim_matches(['-', '.', '_']);
    // Every remaining char is ASCII, so a byte cut is a char cut.
    let cut = &clean[..clean.len().min(60)];
    if cut.is_empty() { "route".to_owned() } else { cut.to_owned() }
}

fn wants_cloud(payload: &Value) -> bool {
    let deliver = match payload.get("deliver") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    };
    deliver.trim().eq_ignore_ascii_case("cloud")
}

/// Python `PurePath.suffix`.
fn suffix(name: &str) -> &str {
    match name.rfind('.') {
        Some(i) if i > 0 && i < name.len() - 1 => &name[i..],
        _ => "",
    }
}

/// The mission bridge: settings plus the optional DJI cloud client.
pub struct App {
    pub settings: Settings,
    pub cloud: Option<Arc<CloudClient>>,
}

impl App {
    pub fn new(settings: Settings, cloud: Option<CloudClient>) -> Self {
        Self { settings, cloud: cloud.map(Arc::new) }
    }

    fn require_cloud(&self) -> Result<&CloudClient, BridgeError> {
        self.cloud.as_deref().ok_or_else(|| {
            BridgeError::Mission(
                "cloud delivery requested but not configured: set SKYLENS_DJI_MC_TOKEN and \
                 SKYLENS_DJI_WK_KEY"
                    .to_owned(),
            )
        })
    }

    fn download_url(&self, file_name: &str) -> String {
        format!("http://{}:{}/missions/{file_name}", self.settings.host, self.settings.port)
    }

    pub fn health(&self) -> Value {
        json!({
            "ok": true,
            "templateReady": self.settings.template.is_file(),
            "template": self.settings.template.display().to_string(),
            "output": self.settings.output.display().to_string(),
            "installConfigured": self.settings.install_target.is_some(),
            "cloudConfigured": self.cloud.is_some(),
        })
    }

    /// Copy the KMZ over one explicit DJI Fly placeholder, backing it up first.
    fn install(&self, source: &Path) -> Result<(bool, Option<String>), BridgeError> {
        let Some(target_raw) = &self.settings.install_target else {
            return Ok((false, None));
        };
        let target = expand_user(target_raw);
        let is_kmz = target
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("kmz"));
        if !is_kmz {
            return Err(BridgeError::Mission(
                "SKYLENS_DJI_INSTALL_TARGET must name one explicit .kmz file".to_owned(),
            ));
        }
        let parent = target.parent().map(Path::to_path_buf).unwrap_or_default();
        let parent_is_dir = if parent.as_os_str().is_empty() { true } else { parent.is_dir() };
        if !parent_is_dir {
            return Err(BridgeError::Mission(format!(
                "install target directory is not available: {}",
                parent.display()
            )));
        }
        if target.exists() {
            let backup_dir = self.settings.output.join("backups");
            std::fs::create_dir_all(&backup_dir).context("creating backup directory")?;
            let stem = target.file_stem().and_then(|s| s.to_str()).unwrap_or("target");
            let unix = chrono::Utc::now().timestamp();
            let backup = backup_dir.join(format!("{stem}-{unix}.kmz"));
            std::fs::copy(&target, &backup).context("backing up install target")?;
        }
        std::fs::copy(source, &target).context("installing mission")?;
        Ok((true, Some(target.display().to_string())))
    }

    async fn upload(&self, path: &Path, distance_m: f64) -> Result<CloudRef, BridgeError> {
        let cloud = self.require_cloud()?;
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("mission");
        let result = cloud.upload_mission_file(path, stem, distance_m).await?;
        Ok(CloudRef { mission_uuid: result.mission_uuid, verified: result.verified })
    }

    /// `POST /missions`: one KMZ, optional install, optional cloud delivery.
    pub async fn single(&self, payload: &Value) -> Result<SingleResponse, BridgeError> {
        let name = safe_name(payload.get("name"));
        let stamp = local_stamp();
        let output_path = self.settings.output.join(format!("{stamp}-{name}.kmz"));
        let result = build_mission(payload, &self.settings.template, &output_path, None)?;
        let (installed, install_target) = self.install(&result.path)?;
        let file_name = file_name_of(&result.path);
        let cloud = if wants_cloud(payload) {
            Some(self.upload(&result.path, result.route_distance_m).await?)
        } else {
            None
        };
        Ok(SingleResponse {
            ok: true,
            download_url: self.download_url(&file_name),
            file_name,
            sha256: result.sha256,
            waypoint_count: result.waypoint_count,
            route_distance_m: round1(result.route_distance_m),
            takeoff_alt_msl: round1(result.takeoff_alt_msl),
            installed,
            install_target,
            warnings: result.warnings,
            cloud,
        })
    }

    /// `POST /formation`: three member KMZs, never auto-installed (one target
    /// file cannot mean three aircraft), optionally delivered to the cloud.
    /// The core uplink calls this too.
    pub async fn formation(&self, payload: &Value) -> Result<FormationResponse, BridgeError> {
        let name = safe_name(payload.get("name"));
        let stamp = local_stamp();
        let members =
            build_formation(payload, &self.settings.template, &self.settings.output, &name, &stamp, None)?;
        let cloud = if wants_cloud(payload) { Some(self.require_cloud()?) } else { None };
        let mut rows = Vec::with_capacity(members.len());
        for (station, result) in members {
            let file_name = file_name_of(&result.path);
            let cloud_ref = match cloud {
                Some(_) => Some(self.upload(&result.path, result.route_distance_m).await?),
                None => None,
            };
            rows.push(MemberRow {
                station: station.to_owned(),
                download_url: self.download_url(&file_name),
                file_name,
                sha256: result.sha256,
                waypoint_count: result.waypoint_count,
                route_distance_m: round1(result.route_distance_m),
                takeoff_alt_msl: round1(result.takeoff_alt_msl),
                warnings: result.warnings,
                cloud: cloud_ref,
            });
        }
        Ok(FormationResponse { ok: true, installed: false, members: rows })
    }

    /// `POST /cloud/delete`.
    pub async fn cloud_delete(&self, payload: &Value) -> Result<Value, BridgeError> {
        let pick = |key: &str| match payload.get(key) {
            None | Some(Value::Null) | Some(Value::Bool(false)) => String::new(),
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
        };
        let mut mission_uuid = pick("missionUuid");
        if mission_uuid.is_empty() {
            mission_uuid = pick("uuid");
        }
        let mission_uuid = mission_uuid.trim().to_owned();
        if mission_uuid.is_empty() {
            return Err(BridgeError::Mission("missionUuid is required".to_owned()));
        }
        let removed = self.require_cloud()?.delete_mission(&mission_uuid).await?;
        Ok(json!({"ok": true, "missionUuid": mission_uuid, "removed": removed}))
    }
}

fn file_name_of(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// `time.strftime("%Y%m%d-%H%M%S")` in local time.
fn local_stamp() -> String {
    chrono::Local::now().format("%Y%m%d-%H%M%S").to_string()
}

// ── HTTP layer ──────────────────────────────────────────────────────────

pub fn router(app: Arc<App>) -> Router {
    Router::new().fallback(dispatch).with_state(app)
}

/// Bind and serve until the process ends.
pub async fn serve(app: Arc<App>, listener: tokio::net::TcpListener) -> anyhow::Result<()> {
    axum::serve(listener, router(app)).await.context("HTTP server stopped")
}

struct Ctx<'a> {
    origin: Option<String>,
    allowed: &'a HashSet<String>,
}

impl Ctx<'_> {
    fn origin_allowed(&self) -> bool {
        self.origin.as_ref().is_none_or(|o| self.allowed.contains(o))
    }

    fn cors(&self, headers: &mut HeaderMap) {
        if let Some(origin) = &self.origin
            && self.allowed.contains(origin)
            && let Ok(value) = HeaderValue::from_str(origin)
        {
            headers.insert("access-control-allow-origin", value);
            headers.insert(VARY, HeaderValue::from_static("Origin"));
        }
    }

    fn json(&self, status: StatusCode, value: &impl Serialize) -> Response {
        let body = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
        let mut resp = Response::new(Body::from(body));
        *resp.status_mut() = status;
        let headers = resp.headers_mut();
        self.cors(headers);
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json; charset=utf-8"));
        headers.insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
        resp
    }

    fn error(&self, status: StatusCode, message: &str) -> Response {
        self.json(status, &json!({"ok": false, "error": message}))
    }

    fn bridge_error(&self, err: BridgeError) -> Response {
        match err {
            BridgeError::Mission(m) => self.error(StatusCode::BAD_REQUEST, &m),
            BridgeError::Cloud(m) => self.error(StatusCode::BAD_GATEWAY, &m),
            BridgeError::Internal(e) => {
                tracing::error!("[mission-bridge] unexpected error: {e:#}");
                self.error(StatusCode::INTERNAL_SERVER_ERROR, "internal error")
            }
        }
    }
}

async fn dispatch(State(app): State<Arc<App>>, req: Request) -> Response {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let origin = req.headers().get(ORIGIN).map(|v| v.to_str().unwrap_or("").to_owned());
    let ctx = Ctx { origin, allowed: &app.settings.allowed_origins };
    let resp = if method == Method::GET {
        get(&app, &ctx, uri.path()).await
    } else if method == Method::POST {
        let full = uri.path_and_query().map(|p| p.as_str()).unwrap_or(uri.path()).to_owned();
        post(&app, &ctx, &full, req).await
    } else if method == Method::OPTIONS {
        options(&ctx)
    } else {
        let mut resp = Response::new(Body::from(format!("Unsupported method ('{method}')")));
        *resp.status_mut() = StatusCode::NOT_IMPLEMENTED;
        resp
    };
    // Paths and status only: bodies may carry tokens or coordinates.
    tracing::info!("[mission-bridge] \"{} {}\" {}", method, uri.path(), resp.status().as_u16());
    resp
}

fn options(ctx: &Ctx<'_>) -> Response {
    if !ctx.origin_allowed() {
        return ctx.error(StatusCode::FORBIDDEN, "origin not allowed");
    }
    let mut resp = Response::new(Body::empty());
    *resp.status_mut() = StatusCode::NO_CONTENT;
    let headers = resp.headers_mut();
    ctx.cors(headers);
    headers.insert("access-control-allow-methods", HeaderValue::from_static("GET, POST, OPTIONS"));
    headers.insert("access-control-allow-headers", HeaderValue::from_static("Content-Type"));
    headers.insert("access-control-max-age", HeaderValue::from_static("600"));
    resp
}

async fn get(app: &App, ctx: &Ctx<'_>, path: &str) -> Response {
    if path == "/health" {
        return ctx.json(StatusCode::OK, &app.health());
    }
    if path == "/cloud/missions" {
        let Some(cloud) = app.cloud.as_deref() else {
            return ctx.error(StatusCode::SERVICE_UNAVAILABLE, "cloud not configured");
        };
        return match cloud.list_missions().await {
            Ok(items) => {
                let missions: Vec<Value> = items
                    .iter()
                    .map(|it| {
                        json!({
                            "uuid": it.get("uuid").cloned().unwrap_or(Value::Null),
                            "name": it.get("name").cloned().unwrap_or(Value::Null),
                            "waypointCount": it.get("point_num").cloned().unwrap_or(Value::Null),
                            "distanceM": it.get("distance").cloned().unwrap_or(Value::Null),
                        })
                    })
                    .collect();
                ctx.json(StatusCode::OK, &json!({"ok": true, "missions": missions}))
            }
            Err(e) => ctx.error(StatusCode::BAD_GATEWAY, &e.to_string()),
        };
    }
    if let Some(rest) = path.strip_prefix("/missions/") {
        return download(app, ctx, rest);
    }
    ctx.error(StatusCode::NOT_FOUND, "not found")
}

fn download(app: &App, ctx: &Ctx<'_>, rest: &str) -> Response {
    let not_found = || ctx.error(StatusCode::NOT_FOUND, "mission not found");
    let decoded = percent_decode_str(rest).decode_utf8_lossy();
    // Final path component only (both separators, whatever the host OS).
    let name = decoded.rsplit(['/', '\\']).next().unwrap_or("");
    if name.is_empty() || name == "." || name == ".." || name.contains(':') {
        return not_found();
    }
    if !suffix(name).eq_ignore_ascii_case(".kmz") {
        return not_found();
    }
    let path = app.settings.output.join(name);
    if !path.is_file() {
        return not_found();
    }
    let body = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => return not_found(),
    };
    let len = body.len();
    let mut resp = Response::new(Body::from(body));
    let headers = resp.headers_mut();
    ctx.cors(headers);
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/vnd.google-earth.kmz"));
    if let Ok(v) = HeaderValue::from_str(&format!("attachment; filename=\"{name}\"")) {
        headers.insert(CONTENT_DISPOSITION, v);
    }
    headers.insert(CONTENT_LENGTH, HeaderValue::from(len));
    headers.insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    resp
}

async fn post(app: &App, ctx: &Ctx<'_>, full_path: &str, req: Request) -> Response {
    if !matches!(full_path, "/missions" | "/formation" | "/cloud/delete") {
        return ctx.error(StatusCode::NOT_FOUND, "not found");
    }
    if !ctx.origin_allowed() {
        return ctx.error(StatusCode::FORBIDDEN, "origin not allowed");
    }
    let payload = match read_payload(req).await {
        Ok(p) => p,
        Err(message) => return ctx.error(StatusCode::BAD_REQUEST, &message),
    };
    match full_path {
        "/cloud/delete" => match app.cloud_delete(&payload).await {
            Ok(body) => ctx.json(StatusCode::OK, &body),
            Err(e) => ctx.bridge_error(e),
        },
        "/formation" => match app.formation(&payload).await {
            Ok(body) => ctx.json(StatusCode::CREATED, &body),
            Err(e) => ctx.bridge_error(e),
        },
        _ => match app.single(&payload).await {
            Ok(body) => ctx.json(StatusCode::CREATED, &body),
            Err(e) => ctx.bridge_error(e),
        },
    }
}

/// Content-Length gated read of a JSON object body (at most [`MAX_BODY`]).
async fn read_payload(req: Request) -> Result<Value, String> {
    const EMPTY_OR_LARGE: &str = "request body is empty or too large";
    let length: usize = req
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    if length == 0 || length > MAX_BODY {
        return Err(EMPTY_OR_LARGE.to_owned());
    }
    let bytes = to_bytes(req.into_body(), MAX_BODY).await.map_err(|_| EMPTY_OR_LARGE.to_owned())?;
    let payload: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if !payload.is_object() {
        return Err("request body must be a JSON object".to_owned());
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud::CloudConfig;
    use crate::mission::tests::{payload, write_lito_template};
    use std::collections::HashMap;

    struct Bridge {
        base: String,
        dir: tempfile::TempDir,
        http: reqwest::Client,
    }

    async fn start(with_template: bool, cloud: Option<CloudClient>, install: Option<String>) -> Bridge {
        let dir = tempfile::tempdir().unwrap();
        let template = dir.path().join("lito-template.kmz");
        if with_template {
            write_lito_template(&template);
        }
        let output = dir.path().join("missions");
        std::fs::create_dir_all(&output).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let settings = Settings {
            host: "127.0.0.1".into(),
            port,
            template,
            output,
            install_target: install,
            allowed_origins: ["http://localhost:5173".to_owned()].into_iter().collect(),
        };
        let app = Arc::new(App::new(settings, cloud));
        tokio::spawn(serve(app, listener));
        Bridge { base: format!("http://127.0.0.1:{port}"), dir, http: reqwest::Client::new() }
    }

    impl Bridge {
        async fn post(&self, path: &str, body: &Value) -> (u16, Value) {
            let resp = self
                .http
                .post(format!("{}{path}", self.base))
                .header("content-type", "application/json")
                .body(serde_json::to_vec(body).unwrap())
                .send()
                .await
                .unwrap();
            let status = resp.status().as_u16();
            (status, serde_json::from_slice(&resp.bytes().await.unwrap()).unwrap())
        }

        async fn get(&self, path: &str) -> reqwest::Response {
            self.http.get(format!("{}{path}", self.base)).send().await.unwrap()
        }
    }

    #[test]
    fn settings_defaults_resolve_against_the_root() {
        let root = Path::new("/repo");
        let s = Settings::from_lookup(root, |_| None).unwrap();
        assert_eq!(s.host, "127.0.0.1");
        assert_eq!(s.port, 8091);
        assert_eq!(s.template, root.join("config").join("lito-x1-template.kmz"));
        assert_eq!(s.output, root.join("output").join("missions"));
        assert_eq!(s.install_target, None);
        assert_eq!(s.allowed_origins.len(), 4);
        assert!(s.allowed_origins.contains("http://127.0.0.1:8080"));

        let vars: HashMap<&str, &str> = [
            ("SKYLENS_MISSION_PORT", "9000"),
            ("SKYLENS_DJI_TEMPLATE_KMZ", "/t/lito.kmz"),
            ("SKYLENS_MISSION_ALLOWED_ORIGINS", " http://a:1 ,,http://b:2"),
            ("SKYLENS_DJI_INSTALL_TARGET", " /mnt/x.kmz "),
        ]
        .into_iter()
        .collect();
        let s = Settings::from_lookup(root, |k| vars.get(k).map(|v| v.to_string())).unwrap();
        assert_eq!(s.port, 9000);
        assert_eq!(s.template, PathBuf::from("/t/lito.kmz"));
        assert_eq!(s.install_target.as_deref(), Some("/mnt/x.kmz"));
        assert_eq!(s.allowed_origins, HashSet::from(["http://a:1", "http://b:2"].map(String::from)));
    }

    #[test]
    fn safe_name_matches_python() {
        assert_eq!(safe_name(Some(&json!("skylens drone/1"))), "skylens-drone-1");
        assert_eq!(safe_name(Some(&json!("..한글.."))), "route");
        assert_eq!(safe_name(Some(&json!("_a_"))), "a");
        assert_eq!(safe_name(None), "route");
        assert_eq!(safe_name(Some(&json!(""))), "route");
        assert_eq!(safe_name(Some(&json!(7))), "7");
        assert_eq!(safe_name(Some(&json!("x".repeat(80)))).len(), 60);
    }

    #[test]
    fn deliver_cloud_is_case_and_space_insensitive() {
        assert!(wants_cloud(&json!({"deliver": " Cloud "})));
        assert!(!wants_cloud(&json!({"deliver": "download"})));
        assert!(!wants_cloud(&json!({})));
    }

    #[tokio::test]
    async fn health_reports_template_and_cloud_state() {
        let bridge = start(false, None, None).await;
        let resp = bridge.get("/health").await;
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.headers()["cache-control"], "no-store");
        let body: Value = resp.json().await.unwrap();
        assert_eq!(body["ok"], true);
        assert_eq!(body["templateReady"], false);
        assert_eq!(body["installConfigured"], false);
        assert_eq!(body["cloudConfigured"], false);
    }

    #[tokio::test]
    async fn post_missions_refuses_without_template() {
        let bridge = start(false, None, None).await;
        let (status, body) = bridge.post("/missions", &payload(false)).await;
        assert_eq!(status, 400);
        assert_eq!(body["ok"], false);
        assert!(body["error"].as_str().unwrap().contains("template KMZ not found"));
    }

    #[tokio::test]
    async fn post_missions_builds_and_serves_the_kmz() {
        let bridge = start(true, None, None).await;
        let (status, body) = bridge.post("/missions", &payload(false)).await;
        assert_eq!(status, 201, "{body}");
        assert_eq!(body["ok"], true);
        assert_eq!(body["waypointCount"], 2);
        assert_eq!(body["takeoffAltMsl"], 50.0);
        assert_eq!(body["routeDistanceM"], 28.6);
        assert_eq!(body["installed"], false);
        assert_eq!(body["installTarget"], Value::Null);
        assert!(body.get("cloud").is_none());
        let file_name = body["fileName"].as_str().unwrap();
        assert!(file_name.ends_with("-test-route.kmz"));
        assert_eq!(body["downloadUrl"], format!("{}/missions/{file_name}", bridge.base));

        let resp = bridge.get(&format!("/missions/{file_name}")).await;
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.headers()["content-type"], "application/vnd.google-earth.kmz");
        assert_eq!(
            resp.headers()["content-disposition"],
            format!("attachment; filename=\"{file_name}\"").as_str()
        );
        let bytes = resp.bytes().await.unwrap();
        assert_eq!(bytes.as_ref(), std::fs::read(bridge.dir.path().join("missions").join(file_name)).unwrap());
    }

    #[tokio::test]
    async fn post_formation_returns_three_members() {
        let bridge = start(true, None, None).await;
        let mut p = payload(false);
        p["formation"] = json!({"spacingM": 10});
        let (status, body) = bridge.post("/formation", &p).await;
        assert_eq!(status, 201, "{body}");
        assert_eq!(body["installed"], false);
        let stations: Vec<&str> =
            body["members"].as_array().unwrap().iter().map(|m| m["station"].as_str().unwrap()).collect();
        assert_eq!(stations, ["left", "right", "center"]);
        for m in body["members"].as_array().unwrap() {
            assert!(m["fileName"].as_str().unwrap().ends_with(&format!("-test-route-{}.kmz", m["station"].as_str().unwrap())));
        }
    }

    #[tokio::test]
    async fn cloud_requests_without_config_fail_like_python() {
        let bridge = start(true, None, None).await;
        let mut p = payload(false);
        p["deliver"] = json!("cloud");
        let (status, body) = bridge.post("/missions", &p).await;
        assert_eq!(status, 400);
        assert!(body["error"].as_str().unwrap().starts_with("cloud delivery requested but not configured"));

        let resp = bridge.get("/cloud/missions").await;
        assert_eq!(resp.status(), 503);
        let body: Value = resp.json().await.unwrap();
        assert_eq!(body, json!({"ok": false, "error": "cloud not configured"}));

        let (status, body) = bridge.post("/cloud/delete", &json!({})).await;
        assert_eq!(status, 400);
        assert_eq!(body["error"], "missionUuid is required");
        let (status, _) = bridge.post("/cloud/delete", &json!({"uuid": "u1"})).await;
        assert_eq!(status, 400);
    }

    #[tokio::test]
    async fn cloud_errors_map_to_bad_gateway() {
        // Nothing listens on port 9 of localhost: the DJI call is unreachable.
        let cloud = CloudClient::with_base_url(CloudConfig::new("t", "k"), "http://127.0.0.1:9");
        let bridge = start(true, Some(cloud), None).await;
        let resp = bridge.get("/cloud/missions").await;
        assert_eq!(resp.status(), 502);
        let body: Value = resp.json().await.unwrap();
        assert!(body["error"].as_str().unwrap().contains("unreachable"));
        let health: Value = bridge.get("/health").await.json().await.unwrap();
        assert_eq!(health["cloudConfigured"], true);
    }

    #[tokio::test]
    async fn rejects_bad_bodies_paths_and_origins() {
        let bridge = start(true, None, None).await;
        let (status, body) = bridge.post("/unknown", &json!({})).await;
        assert_eq!((status, body["error"].as_str().unwrap()), (404, "not found"));

        let (status, body) = bridge.post("/missions", &json!([1, 2])).await;
        assert_eq!((status, body["error"].as_str().unwrap()), (400, "request body must be a JSON object"));

        let resp = bridge
            .http
            .post(format!("{}/missions", bridge.base))
            .body("{not json")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400);

        let resp = bridge.http.post(format!("{}/missions", bridge.base)).send().await.unwrap();
        assert_eq!(resp.status(), 400);
        let body: Value = resp.json().await.unwrap();
        assert_eq!(body["error"], "request body is empty or too large");

        let resp = bridge
            .http
            .post(format!("{}/missions", bridge.base))
            .header("origin", "http://evil.example")
            .body("{}")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 403);

        let resp = bridge.get("/missions/..%2F..%2FCargo.toml").await;
        assert_eq!(resp.status(), 404);
        let resp = bridge.get("/missions/missing.kmz").await;
        assert_eq!(resp.status(), 404);
        let resp = bridge.get("/nope").await;
        assert_eq!(resp.status(), 404);
    }

    #[tokio::test]
    async fn cors_preflight_and_headers_follow_the_allow_list() {
        let bridge = start(true, None, None).await;
        let resp = bridge
            .http
            .request(reqwest::Method::OPTIONS, format!("{}/missions", bridge.base))
            .header("origin", "http://localhost:5173")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 204);
        assert_eq!(resp.headers()["access-control-allow-origin"], "http://localhost:5173");
        assert_eq!(resp.headers()["access-control-allow-methods"], "GET, POST, OPTIONS");

        let resp = bridge
            .http
            .request(reqwest::Method::OPTIONS, format!("{}/missions", bridge.base))
            .header("origin", "http://evil.example")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 403);

        let resp = bridge
            .http
            .get(format!("{}/health", bridge.base))
            .header("origin", "http://localhost:5173")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.headers()["access-control-allow-origin"], "http://localhost:5173");
        assert_eq!(resp.headers()["vary"], "Origin");
    }

    #[tokio::test]
    async fn install_target_is_backed_up_then_replaced() {
        let target_dir = tempfile::tempdir().unwrap();
        let target = target_dir.path().join("SKYLENS_TARGET.kmz");
        std::fs::write(&target, b"old mission").unwrap();
        let bridge = start(true, None, Some(target.to_string_lossy().into_owned())).await;

        let (status, body) = bridge.post("/missions", &payload(false)).await;
        assert_eq!(status, 201, "{body}");
        assert_eq!(body["installed"], true);
        assert_eq!(body["installTarget"], target.display().to_string());
        let installed = std::fs::read(&target).unwrap();
        assert_ne!(installed, b"old mission");
        let backups: Vec<_> = std::fs::read_dir(bridge.dir.path().join("missions").join("backups"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert_eq!(backups.len(), 1);
        assert_eq!(std::fs::read(&backups[0]).unwrap(), b"old mission");
    }

    #[tokio::test]
    async fn install_target_must_be_a_kmz() {
        let bridge = start(true, None, Some("/somewhere/target.txt".into())).await;
        let (status, body) = bridge.post("/missions", &payload(false)).await;
        assert_eq!(status, 400);
        assert_eq!(body["error"], "SKYLENS_DJI_INSTALL_TARGET must name one explicit .kmz file");
    }
}
