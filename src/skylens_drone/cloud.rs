//! Deliver a SkyLens KMZ mission to the DJI Fly consumer cloud.
//!
//! DJI Fly syncs waypoint missions through an undocumented account-scoped cloud
//! so an RC 2 can pull them over the air. This module speaks that protocol:
//! the DJI request signature (`x-wk-sign`, HMAC-SHA256) and a manual AWS SigV4
//! S3 upload, no AWS SDK.
//!
//! Secrets never live in source and never reach the logs. The account token and
//! the signing key come from the environment (or the files the Windows launcher
//! used); without them the cloud path stays disabled and the bridge keeps
//! working in download / local-install mode.
//!
//! Flow (verified against the live service by the Python bridge):
//!   1. POST /api/waypoint/mission/compare        announce the mission uuid
//!   2. POST /api/waypoint/file/upload/sts/init   get AWS STS creds per file
//!   3. PUT  s3://<bucket>/<object_key>           upload mission_file + mission_cover
//!
//! The cloud "mission file" is NOT the raw wpmz KMZ. It is a wrapper zip holding
//! `<mission_uuid>.kmz` (the raw wpmz KMZ) plus `image/ShotSnap.json`. Uploading
//! a raw KMZ makes DJI Fly refuse to open it.
//!
//! Port of `legacy/mission_bridge/cloud.py`.

use std::fmt;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use base64::Engine as _;
use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};
use regex::Regex;
use reqwest::Method;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::kmz;
use crate::paths::expand_user;

pub const BASE_URL: &str = "https://flyapi.djigate.com";
pub const DEFAULT_SECRET_ID: &str = "uav";
pub const DEFAULT_APP_INFO: &str = "Android-SkyLens-13-KR";

/// 1x1 PNG placeholder cover. A mission needs a `mission_cover` object to bind.
const PLACEHOLDER_PNG_B64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC";

/// Cache lifetime for a URL-sourced token, so each request does not re-fetch.
const TOKEN_URL_TTL: Duration = Duration::from_secs(15);

const AUTH_HINTS: [&str; 7] = ["token", "auth", "login", "expire", "unauthor", "登录", "鉴权"];

type HmacSha256 = Hmac<Sha256>;

/// The DJI cloud rejected a request or is unreachable. Maps to HTTP 502.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloudError {
    /// Rejected for an auth/token reason (retryable after a token refresh).
    Auth(String),
    Other(String),
}

impl fmt::Display for CloudError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CloudError::Auth(m) | CloudError::Other(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for CloudError {}

fn other(message: impl Into<String>) -> CloudError {
    CloudError::Other(message.into())
}

/// DJI cloud credentials and signing parameters.
///
/// The account token expires. `token_file` lets an external refresher keep a
/// fresh token on disk; it is re-read on every request. `token_url` points at
/// a central token server. `token` is the static fallback.
#[derive(Clone, PartialEq, Eq)]
pub struct CloudConfig {
    pub token: String,
    pub wk_key: String,
    pub secret_id: String,
    pub app_info: String,
    pub token_file: Option<String>,
    pub token_url: Option<String>,
}

impl fmt::Debug for CloudConfig {
    // Secrets stay out of any `{:?}` that reaches a log.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CloudConfig")
            .field("token", &"<redacted>")
            .field("wk_key", &"<redacted>")
            .field("secret_id", &self.secret_id)
            .field("app_info", &self.app_info)
            .field("token_file", &self.token_file)
            .field("token_url", &self.token_url.as_ref().map(|_| "<set>"))
            .finish()
    }
}

fn trimmed(value: Option<String>) -> String {
    value.map(|v| v.trim().to_owned()).unwrap_or_default()
}

impl CloudConfig {
    #[cfg(test)]
    pub fn new(token: &str, wk_key: &str) -> Self {
        Self {
            token: token.to_owned(),
            wk_key: wk_key.to_owned(),
            secret_id: DEFAULT_SECRET_ID.to_owned(),
            app_info: DEFAULT_APP_INFO.to_owned(),
            token_file: None,
            token_url: None,
        }
    }

    /// `CloudConfig.from_env()` over an arbitrary variable lookup.
    ///
    /// Needs the signing key and at least one token source. A token file or URL
    /// counts even when it is empty now; a refresher may fill it before the
    /// first upload.
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Option<Self> {
        let token = trimmed(get("SKYLENS_DJI_MC_TOKEN"));
        let wk_key = trimmed(get("SKYLENS_DJI_WK_KEY"));
        let token_file = Some(trimmed(get("SKYLENS_DJI_TOKEN_FILE"))).filter(|s| !s.is_empty());
        let token_url = Some(trimmed(get("SKYLENS_DJI_TOKEN_URL"))).filter(|s| !s.is_empty());
        if wk_key.is_empty() {
            return None;
        }
        if token.is_empty() && token_file.is_none() && token_url.is_none() {
            return None;
        }
        let secret_id = Some(trimmed(get("SKYLENS_DJI_WK_SECRETID")))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_SECRET_ID.to_owned());
        let app_info = Some(trimmed(get("SKYLENS_DJI_APP_INFO")))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_APP_INFO.to_owned());
        Some(Self { token, wk_key, secret_id, app_info, token_file, token_url })
    }

    /// [`from_lookup`](Self::from_lookup) plus the defaults
    /// `scripts/run-mission-bridge.ps1` applied before starting the Python bridge:
    ///
    /// - no `SKYLENS_DJI_WK_KEY` -> read it from `<root>/config/.dji_wk_key`
    /// - no `SKYLENS_DJI_TOKEN_FILE` -> `<root>/output/.dji_token`
    ///
    /// Only paths are logged, never the key or token.
    pub fn from_lookup_with_launcher_defaults(
        root: &Path,
        get: impl Fn(&str) -> Option<String>,
    ) -> Option<Self> {
        let key_from_env = trimmed(get("SKYLENS_DJI_WK_KEY"));
        let wk_key = if key_from_env.is_empty() {
            let key_file = root.join("config").join(".dji_wk_key");
            match std::fs::read_to_string(&key_file) {
                Ok(text) => strip_bom(&text).trim().to_owned(),
                Err(_) => {
                    tracing::warn!(
                        "no signing key: set SKYLENS_DJI_WK_KEY or write it to {}. Cloud \
                         delivery stays disabled.",
                        key_file.display()
                    );
                    String::new()
                }
            }
        } else {
            key_from_env
        };

        let file_from_env = get("SKYLENS_DJI_TOKEN_FILE").filter(|s| !s.is_empty());
        let token_file = file_from_env.unwrap_or_else(|| {
            root.join("output").join(".dji_token").to_string_lossy().into_owned()
        });
        if !expand_user(token_file.trim()).exists() {
            tracing::warn!(
                "no token at {}. Cloud delivery stays disabled until one is written.",
                token_file
            );
        }

        Self::from_lookup(|k| match k {
            "SKYLENS_DJI_WK_KEY" => Some(wk_key.clone()),
            "SKYLENS_DJI_TOKEN_FILE" => Some(token_file.clone()),
            _ => get(k),
        })
    }
}

fn strip_bom(text: &str) -> &str {
    text.strip_prefix('\u{feff}').unwrap_or(text)
}

// ── DJI request signing (x-wk-sign) ─────────────────────────────────────

/// base64(HMAC-SHA256) of the DJI canonical string. Pure and deterministic.
pub fn sign_request(path_with_query: &str, nonce: i64, timestamp_ms: i64, cfg: &CloudConfig) -> String {
    let signed = format!(
        "{path_with_query}&X-Wk-Nonce={nonce}&X-Wk-SecretId={}\
         &X-Wk-Signature-Method=HmacSHA256&X-Wk-Timestamp={timestamp_ms}",
        cfg.secret_id
    );
    let mut mac =
        <HmacSha256 as Mac>::new_from_slice(cfg.wk_key.as_bytes()).expect("HMAC takes any key size");
    mac.update(signed.as_bytes());
    base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes())
}

fn looks_like_auth_error(message: &str) -> bool {
    let low = message.to_lowercase();
    AUTH_HINTS.iter().any(|hint| low.contains(hint))
}

// ── KMZ helpers ─────────────────────────────────────────────────────────

/// Waypoint GPS list and centre read back from a KMZ, for the cloud metadata.
pub fn kmz_points(kmz_bytes: &[u8]) -> (Vec<Value>, f64, f64) {
    let mut lats: Vec<f64> = Vec::new();
    let mut lons: Vec<f64> = Vec::new();
    if let Ok(names) = kmz::names(kmz_bytes) {
        let name = names
            .iter()
            .find(|n| n.ends_with("waylines.wpml"))
            .or_else(|| names.iter().find(|n| n.ends_with(".kml")));
        if let Some(xml) = name.and_then(|n| kmz::read(kmz_bytes, n).ok()) {
            let xml = String::from_utf8_lossy(&xml);
            let re = Regex::new(r"<coordinates>\s*([-\d.]+),([-\d.]+)").expect("static pattern");
            for caps in re.captures_iter(&xml) {
                // Python stopped at the first unparsable number and kept what it had.
                let Ok(lon) = caps[1].parse::<f64>() else { break };
                lons.push(lon);
                let Ok(lat) = caps[2].parse::<f64>() else { break };
                lats.push(lat);
            }
        }
    }
    let gps = lats
        .iter()
        .zip(&lons)
        .map(|(la, lo)| json!({"latitude": la, "longitude": lo}))
        .collect();
    let mean = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
    (gps, mean(&lats), mean(&lons))
}

/// Wrap a raw wpmz KMZ as the cloud mission-file zip (`<uuid>.kmz` + ShotSnap).
///
/// Bytes that are not a raw wpmz KMZ (already wrapped, or not a zip at all)
/// come back unchanged.
pub fn wrap_mission_package(kmz_bytes: &[u8], mission_uuid: &str) -> Result<Vec<u8>, CloudError> {
    let Ok(names) = kmz::names(kmz_bytes) else {
        return Ok(kmz_bytes.to_vec());
    };
    let is_raw = names.iter().any(|n| n.ends_with("template.kml") || n.ends_with("waylines.wpml"));
    if !is_raw {
        return Ok(kmz_bytes.to_vec());
    }
    let inner = format!("{mission_uuid}.kmz");
    kmz::write(&[
        (inner.as_str(), kmz_bytes),
        ("image/ShotSnap.json", &br#"{"WAY_POINT": {}, "POI_POINT": {}}"#[..]),
    ])
    .map_err(|e| other(format!("could not wrap mission package: {e}")))
}

// ── AWS S3 upload (manual SigV4) ────────────────────────────────────────

/// `urllib.parse.quote(seg, safe="~")`: everything but unreserved characters.
const S3_KEY_SEGMENT: &AsciiSet =
    &NON_ALPHANUMERIC.remove(b'-').remove(b'.').remove(b'_').remove(b'~');

/// STS credentials and destination for one file, from `sts/init`.
#[derive(Clone, PartialEq, Eq)]
pub struct S3Target {
    pub ak: String,
    pub sk: String,
    pub session_token: String,
    pub bucket: String,
    pub region: String,
    pub object_key: String,
}

impl S3Target {
    fn from_meta(meta: &Value) -> Result<Self, CloudError> {
        let field = |v: &Value, key: &str| -> Result<String, CloudError> {
            v.get(key)
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| other(format!("DJI sts/init file entry has no {key}")))
        };
        let auth = meta
            .get("authorization")
            .ok_or_else(|| other("DJI sts/init file entry has no authorization"))?;
        Ok(Self {
            ak: field(auth, "ak")?,
            sk: field(auth, "sk")?,
            session_token: field(auth, "session_token")?,
            bucket: field(meta, "cloud_bucket_name")?,
            region: field(meta, "region")?,
            object_key: field(meta, "object_key")?,
        })
    }
}

/// Everything a signed S3 PUT needs. Pure output of [`sign_s3_put`].
#[cfg_attr(not(test), allow(dead_code))] // host and canonical_uri are pinned by tests
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedPut {
    pub url: String,
    pub host: String,
    pub canonical_uri: String,
    pub amz_date: String,
    pub payload_hash: String,
    pub authorization: String,
}

fn hmac_bytes(key: &[u8], msg: &str) -> Vec<u8> {
    let mut mac = <HmacSha256 as Mac>::new_from_slice(key).expect("HMAC takes any key size");
    mac.update(msg.as_bytes());
    mac.finalize().into_bytes().to_vec()
}

/// AWS SigV4 for `PUT` of `body` to `target`, signed at `now`.
pub fn sign_s3_put(target: &S3Target, body: &[u8], now: DateTime<Utc>) -> SignedPut {
    let host = format!("{}.s3.{}.amazonaws.com", target.bucket, target.region);
    let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
    let datestamp = now.format("%Y%m%d").to_string();
    let canonical_uri = format!(
        "/{}",
        target
            .object_key
            .split('/')
            .map(|seg| utf8_percent_encode(seg, S3_KEY_SEGMENT).to_string())
            .collect::<Vec<_>>()
            .join("/")
    );
    let payload_hash = hex::encode(Sha256::digest(body));
    let token = &target.session_token;
    let canonical_headers = format!(
        "host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n\
         x-amz-security-token:{token}\n"
    );
    let signed_headers = "host;x-amz-content-sha256;x-amz-date;x-amz-security-token";
    let canonical_request = [
        "PUT",
        &canonical_uri,
        "",
        &canonical_headers,
        signed_headers,
        &payload_hash,
    ]
    .join("\n");
    let scope = format!("{datestamp}/{}/s3/aws4_request", target.region);
    let string_to_sign = [
        "AWS4-HMAC-SHA256",
        &amz_date,
        &scope,
        &hex::encode(Sha256::digest(canonical_request.as_bytes())),
    ]
    .join("\n");
    let k_date = hmac_bytes(format!("AWS4{}", target.sk).as_bytes(), &datestamp);
    let k_region = hmac_bytes(&k_date, &target.region);
    let k_service = hmac_bytes(&k_region, "s3");
    let k_signing = hmac_bytes(&k_service, "aws4_request");
    let signature = hex::encode(hmac_bytes(&k_signing, &string_to_sign));
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, \
         Signature={signature}",
        target.ak
    );
    SignedPut {
        url: format!("https://{host}{canonical_uri}"),
        host,
        canonical_uri,
        amz_date,
        payload_hash,
        authorization,
    }
}

// ── client ──────────────────────────────────────────────────────────────

/// Result of pushing one mission to the account cloud.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloudResult {
    pub mission_uuid: String,
    pub verified: bool,
}

/// DJI cloud client: one per configuration, shared across requests.
pub struct CloudClient {
    cfg: CloudConfig,
    http: reqwest::Client,
    base_url: String,
    token_url_cache: Mutex<Option<(Instant, String)>>,
}

impl CloudClient {
    pub fn new(cfg: CloudConfig) -> Self {
        Self::with_base_url(cfg, BASE_URL)
    }

    pub fn with_base_url(cfg: CloudConfig, base_url: &str) -> Self {
        Self {
            cfg,
            http: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_owned(),
            token_url_cache: Mutex::new(None),
        }
    }

    async fn fetch_token_url(&self, url: &str, force: bool) -> String {
        let cached = self.token_url_cache.lock().expect("token cache poisoned").clone();
        if let Some((at, value)) = &cached
            && !force
            && at.elapsed() < TOKEN_URL_TTL
        {
            return value.clone();
        }
        let fetched = async {
            let resp = self
                .http
                .get(url)
                .timeout(Duration::from_secs(10))
                .send()
                .await?
                .error_for_status()?;
            resp.text().await
        }
        .await;
        match fetched {
            Ok(text) => {
                let value = strip_bom(&text).trim().to_owned();
                *self.token_url_cache.lock().expect("token cache poisoned") =
                    Some((Instant::now(), value.clone()));
                value
            }
            Err(_) => cached.map(|(_, v)| v).unwrap_or_default(),
        }
    }

    /// Current token. Priority: URL (cached briefly) -> file -> static env token.
    /// `force` bypasses the URL cache, used by the one auth-error retry.
    pub async fn resolve_token(&self, force: bool) -> String {
        if let Some(url) = &self.cfg.token_url {
            let value = self.fetch_token_url(url, force).await;
            if !value.is_empty() {
                return value;
            }
        }
        if let Some(file) = &self.cfg.token_file
            && let Ok(text) = tokio::fs::read_to_string(expand_user(file)).await
        {
            let value = strip_bom(&text).trim();
            if !value.is_empty() {
                return value.to_owned();
            }
        }
        self.cfg.token.clone()
    }

    async fn wk_headers(&self, path_with_query: &str, force_token: bool) -> Vec<(&'static str, String)> {
        let nonce = rand::random::<i32>() as i64;
        let ts = Utc::now().timestamp_millis();
        let sign = sign_request(path_with_query, nonce, ts, &self.cfg);
        vec![
            ("x-wk-nonce", nonce.to_string()),
            ("x-wk-secretid", self.cfg.secret_id.clone()),
            ("x-wk-timestamp", ts.to_string()),
            ("x-wk-sign", sign),
            ("x-mc-token", self.resolve_token(force_token).await),
            ("x-request-id", uuid::Uuid::new_v4().to_string()),
            ("x-app-info", self.cfg.app_info.clone()),
            ("content-type", "application/json; charset=UTF-8".to_owned()),
            ("user-agent", "okhttp/4.12.0".to_owned()),
        ]
    }

    async fn api_once(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
        force_token: bool,
    ) -> Result<Value, CloudError> {
        let mut req = self
            .http
            .request(method.clone(), format!("{}{path}", self.base_url))
            .timeout(Duration::from_secs(30));
        for (name, value) in self.wk_headers(path, force_token).await {
            req = req.header(name, value);
        }
        if let Some(body) = body {
            req = req.body(serde_json::to_vec(body).expect("JSON value serializes"));
        }
        let resp = req.send().await.map_err(|e| {
            // reqwest errors carry the URL, which holds no secret (path only).
            other(format!("DJI {method} {path} unreachable: {e}"))
        })?;
        let status = resp.status();
        if status == 401 || status == 403 {
            return Err(CloudError::Auth(format!("DJI {method} {path} -> HTTP {}", status.as_u16())));
        }
        if status.is_client_error() || status.is_server_error() {
            return Err(other(format!("DJI {method} {path} -> HTTP {}", status.as_u16())));
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| other(format!("DJI {method} {path} unreachable: {e}")))?;
        let parsed: Value = serde_json::from_slice(&bytes)
            .map_err(|_| other(format!("DJI {method} {path} returned invalid JSON")))?;
        let code = parsed.get("result").and_then(|r| r.get("code")).and_then(Value::as_f64);
        if code == Some(0.0) {
            return Ok(parsed);
        }
        let msg = match parsed.get("result") {
            Some(Value::Object(r)) => match r.get("msg") {
                None | Some(Value::Null) => "None".to_owned(),
                Some(Value::String(s)) => s.clone(),
                Some(v) => v.to_string(),
            },
            _ => "unknown error".to_owned(),
        };
        let message = format!("DJI {method} {path} rejected: {msg}");
        if looks_like_auth_error(&msg) { Err(CloudError::Auth(message)) } else { Err(other(message)) }
    }

    /// Call the DJI API; retry once on an auth error after re-reading the token.
    /// The retry matters only when a token file or URL is configured.
    async fn api(&self, method: Method, path: &str, body: Option<&Value>) -> Result<Value, CloudError> {
        match self.api_once(method.clone(), path, body, false).await {
            Err(CloudError::Auth(message)) => {
                if self.cfg.token_file.is_none() && self.cfg.token_url.is_none() {
                    return Err(CloudError::Auth(message));
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
                self.api_once(method, path, body, true).await
            }
            result => result,
        }
    }

    pub async fn compare(&self, missions: Vec<Value>, start_time: i64) -> Result<Value, CloudError> {
        let body = json!({"missions": missions, "start_time": start_time});
        self.api(Method::POST, "/api/waypoint/mission/compare", Some(&body)).await
    }

    pub async fn check(&self, uuids: &[String]) -> Result<Vec<Value>, CloudError> {
        let body = json!({"mission_uuid_list": uuids});
        let data = self.api(Method::POST, "/api/waypoint/mission/check", Some(&body)).await?;
        Ok(data
            .get("data")
            .and_then(|d| d.get("items"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default())
    }

    pub async fn list_missions(&self) -> Result<Vec<Value>, CloudError> {
        let compared = self.compare(Vec::new(), 0).await?;
        let uuids: Vec<String> = compared
            .get("data")
            .and_then(|d| d.get("mission_download_list"))
            .and_then(Value::as_array)
            .map(|list| {
                list.iter()
                    .map(|v| v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        if uuids.is_empty() { Ok(Vec::new()) } else { self.check(&uuids).await }
    }

    pub async fn delete_mission(&self, mission_uuid: &str) -> Result<bool, CloudError> {
        let now = Utc::now().timestamp_millis();
        self.compare(
            vec![json!({
                "uuid": mission_uuid,
                "name": "",
                "md5": "",
                "point_gps_list": [],
                "app_create_time": now - 1000,
                "app_update_time": now,
                "app_delete_time": now,
            })],
            0,
        )
        .await?;
        let remaining = self.list_missions().await?;
        Ok(!remaining.iter().any(|it| it.get("uuid").and_then(Value::as_str) == Some(mission_uuid)))
    }

    async fn s3_put(&self, target: &S3Target, body: &[u8]) -> Result<(), CloudError> {
        let signed = sign_s3_put(target, body, Utc::now());
        let resp = self
            .http
            .put(&signed.url)
            .timeout(Duration::from_secs(120))
            .header("Authorization", &signed.authorization)
            .header("x-amz-content-sha256", &signed.payload_hash)
            .header("x-amz-date", &signed.amz_date)
            .header("x-amz-security-token", &target.session_token)
            .body(body.to_vec())
            .send()
            .await
            .map_err(|e| other(format!("S3 upload unreachable: {}", e.without_url())))?;
        let status = resp.status().as_u16();
        if status != 200 && status != 204 {
            return Err(other(format!("S3 upload failed: HTTP {status}")));
        }
        Ok(())
    }

    /// Push one KMZ to the account cloud so an RC 2 can download it over the air.
    pub async fn upload_mission(
        &self,
        kmz_bytes: &[u8],
        name: &str,
        distance_m: f64,
        cover_png: Option<&[u8]>,
    ) -> Result<CloudResult, CloudError> {
        let mission_uuid = uuid::Uuid::new_v4().to_string();
        let now = Utc::now().timestamp_millis();
        let (gps, lat, lon) = kmz_points(kmz_bytes);
        let package = wrap_mission_package(kmz_bytes, &mission_uuid)?;
        let placeholder;
        let cover = match cover_png {
            Some(c) => c,
            None => {
                placeholder = base64::engine::general_purpose::STANDARD
                    .decode(PLACEHOLDER_PNG_B64)
                    .expect("static PNG");
                &placeholder[..]
            }
        };
        let package_md5 = hex::encode(md5::Md5::digest(&package));
        let cover_md5 = hex::encode(md5::Md5::digest(cover));

        let base = json!({
            "app_create_time": now,
            "app_update_time": now,
            "biz_version": "1",
            "distance": distance_m,
            "drone_list": [],
            "duration": 0,
            "latitude": lat,
            "longitude": lon,
            "mission_name": name,
            "mission_uuid": mission_uuid,
            "point_gps_list": gps,
            "point_num": gps.len(),
        });
        let with = |extra: Value| {
            let mut entry = base.clone();
            if let (Value::Object(e), Value::Object(x)) = (&mut entry, extra) {
                e.extend(x);
            }
            entry
        };
        let files = vec![
            with(json!({
                "biz_type": "mission_file",
                "md5": package_md5,
                "name": format!("{mission_uuid}.kmz"),
                "uuid": uuid::Uuid::new_v4().to_string(),
            })),
            with(json!({
                "biz_type": "mission_cover",
                "md5": cover_md5,
                "name": format!("{name}.png"),
                "uuid": uuid::Uuid::new_v4().to_string(),
            })),
        ];
        self.compare(
            vec![json!({
                "uuid": mission_uuid,
                "name": name,
                "md5": package_md5,
                "point_gps_list": gps,
                "app_create_time": now,
                "app_update_time": now,
                "app_delete_time": 0,
            })],
            0,
        )
        .await?;
        let sts = self
            .api(Method::POST, "/api/waypoint/file/upload/sts/init", Some(&json!({"files": files})))
            .await?;
        let metas = sts
            .get("data")
            .and_then(|d| d.get("files"))
            .and_then(Value::as_array)
            .ok_or_else(|| other("DJI sts/init returned no files"))?;
        for meta in metas {
            let target = S3Target::from_meta(meta)?;
            let is_mission = meta.get("biz_type").and_then(Value::as_str) == Some("mission_file");
            self.s3_put(&target, if is_mission { &package[..] } else { cover }).await?;
        }

        let mut verified = false;
        let wanted = [mission_uuid.clone()];
        for _ in 0..6 {
            tokio::time::sleep(Duration::from_secs(2)).await;
            let items = self.check(&wanted).await?;
            if items.iter().any(|it| it.get("uuid").and_then(Value::as_str) == Some(mission_uuid.as_str())) {
                verified = true;
                break;
            }
        }
        Ok(CloudResult { mission_uuid, verified })
    }

    pub async fn upload_mission_file(
        &self,
        kmz_path: &Path,
        name: &str,
        distance_m: f64,
    ) -> anyhow::Result<CloudResult> {
        let bytes = tokio::fs::read(kmz_path).await?;
        Ok(self.upload_mission(&bytes, name, distance_m, None).await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn raw_kmz() -> Vec<u8> {
        kmz::write(&[
            ("wpmz/template.kml", &b"<kml/>"[..]),
            (
                "wpmz/waylines.wpml",
                &b"<kml><coordinates>127.1,36.1</coordinates><coordinates>127.2,36.2</coordinates></kml>"[..],
            ),
        ])
        .unwrap()
    }

    fn lookup(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + use<> {
        let map: HashMap<String, String> =
            vars.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |k| map.get(k).cloned()
    }

    #[test]
    fn sign_request_matches_captured_vector() {
        // Captured from the live service: this exact canonical string + uav key
        // produced this signature. Pins the algorithm so a refactor cannot drift.
        let cfg = CloudConfig::new("t", "d62db06b88a3efe3eb62a50cf5a25569");
        let sign = sign_request("/api/waypoint/mission/compare", 1309760617, 1790707096180, &cfg);
        assert_eq!(sign, "HgkYpTnkrCcsWg8INZ3pe2N+0aVfisOXtOSZSl23Ex8=");
    }

    #[test]
    fn s3_sigv4_matches_python_vector() {
        // Expected values printed by the legacy cloud._s3_put arithmetic for the
        // same inputs (a key with a space and a tilde exercises the quoting).
        let target = S3Target {
            ak: "AKIDEXAMPLE".into(),
            sk: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".into(),
            session_token: "session/token+abc=".into(),
            bucket: "dji-fly-bucket".into(),
            region: "us-west-2".into(),
            object_key: "waypoint/user 1/mission~a.kmz".into(),
        };
        let now = DateTime::parse_from_rfc3339("2026-09-17T01:34:56Z").unwrap().with_timezone(&Utc);
        let signed = sign_s3_put(&target, b"hello kmz", now);
        assert_eq!(signed.host, "dji-fly-bucket.s3.us-west-2.amazonaws.com");
        assert_eq!(signed.canonical_uri, "/waypoint/user%201/mission~a.kmz");
        assert_eq!(
            signed.url,
            "https://dji-fly-bucket.s3.us-west-2.amazonaws.com/waypoint/user%201/mission~a.kmz"
        );
        assert_eq!(signed.amz_date, "20260917T013456Z");
        assert_eq!(
            signed.payload_hash,
            "6ffe2beed6e5462fe0f1ee5fc044065c46c5f7c863334b7b42489480f4f40322"
        );
        assert_eq!(
            signed.authorization,
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260917/us-west-2/s3/aws4_request, \
             SignedHeaders=host;x-amz-content-sha256;x-amz-date;x-amz-security-token, \
             Signature=c0a77c60e6f0972aa82a0849a5b3afe8b090a445d022222613ca55deda220db1"
        );
    }

    #[test]
    fn wrap_raw_kmz_into_cloud_package() {
        let raw = raw_kmz();
        let package = wrap_mission_package(&raw, "mission-uuid-1").unwrap();
        let names = kmz::names(&package).unwrap();
        assert!(names.contains(&"mission-uuid-1.kmz".to_owned()));
        assert!(names.contains(&"image/ShotSnap.json".to_owned()));
        // inner kmz must be the untouched raw mission
        assert_eq!(kmz::read(&package, "mission-uuid-1.kmz").unwrap(), raw);
        let shot: Value =
            serde_json::from_slice(&kmz::read(&package, "image/ShotSnap.json").unwrap()).unwrap();
        assert_eq!(shot, json!({"WAY_POINT": {}, "POI_POINT": {}}));
    }

    #[test]
    fn wrap_is_idempotent_for_already_wrapped() {
        let raw = raw_kmz();
        let package = wrap_mission_package(&raw, "mission-uuid-1").unwrap();
        let again = wrap_mission_package(&package, "mission-uuid-2").unwrap();
        assert_eq!(again, package); // already a wrapper, returned unchanged
    }

    #[test]
    fn wrap_leaves_non_zip_bytes_alone() {
        assert_eq!(wrap_mission_package(b"plain", "u").unwrap(), b"plain");
    }

    #[test]
    fn kmz_points_reads_waylines_coordinates() {
        let (gps, lat, lon) = kmz_points(&raw_kmz());
        assert_eq!(
            gps,
            vec![
                json!({"latitude": 36.1, "longitude": 127.1}),
                json!({"latitude": 36.2, "longitude": 127.2}),
            ]
        );
        assert!((lat - 36.15).abs() < 1e-12);
        assert!((lon - 127.15).abs() < 1e-12);
        assert_eq!(kmz_points(b"nope"), (vec![], 0.0, 0.0));
    }

    #[test]
    fn config_from_env_requires_both_secrets() {
        assert_eq!(CloudConfig::from_lookup(lookup(&[])), None);
        // key still missing
        assert_eq!(CloudConfig::from_lookup(lookup(&[("SKYLENS_DJI_MC_TOKEN", "token")])), None);
        let cfg = CloudConfig::from_lookup(lookup(&[
            ("SKYLENS_DJI_MC_TOKEN", "token"),
            ("SKYLENS_DJI_WK_KEY", "key"),
        ]))
        .unwrap();
        assert_eq!(cfg.token, "token");
        assert_eq!(cfg.secret_id, "uav");
        assert_eq!(cfg.app_info, "Android-SkyLens-13-KR");
    }

    #[test]
    fn token_file_or_url_counts_as_a_token_source() {
        let cfg = CloudConfig::from_lookup(lookup(&[
            ("SKYLENS_DJI_WK_KEY", " key "),
            ("SKYLENS_DJI_TOKEN_FILE", "output/.dji_token"),
            ("SKYLENS_DJI_WK_SECRETID", "  "),
        ]))
        .unwrap();
        assert_eq!(cfg.wk_key, "key");
        assert_eq!(cfg.token, "");
        assert_eq!(cfg.token_file.as_deref(), Some("output/.dji_token"));
        assert_eq!(cfg.secret_id, "uav");
        assert!(
            CloudConfig::from_lookup(lookup(&[
                ("SKYLENS_DJI_WK_KEY", "key"),
                ("SKYLENS_DJI_TOKEN_URL", "http://rig:8077/"),
            ]))
            .is_some()
        );
    }

    #[test]
    fn launcher_defaults_read_the_key_file_and_default_token_path() {
        let root = tempfile::tempdir().unwrap();
        // Without the key file nothing is configured.
        assert_eq!(CloudConfig::from_lookup_with_launcher_defaults(root.path(), lookup(&[])), None);

        std::fs::create_dir_all(root.path().join("config")).unwrap();
        std::fs::write(root.path().join("config").join(".dji_wk_key"), "\u{feff}filekey\r\n").unwrap();
        let cfg = CloudConfig::from_lookup_with_launcher_defaults(root.path(), lookup(&[])).unwrap();
        assert_eq!(cfg.wk_key, "filekey");
        let default_token = root.path().join("output").join(".dji_token");
        assert_eq!(cfg.token_file, Some(default_token.to_string_lossy().into_owned()));

        // Explicit environment wins over both defaults.
        let cfg = CloudConfig::from_lookup_with_launcher_defaults(
            root.path(),
            lookup(&[("SKYLENS_DJI_WK_KEY", "envkey"), ("SKYLENS_DJI_TOKEN_FILE", "/t")]),
        )
        .unwrap();
        assert_eq!(cfg.wk_key, "envkey");
        assert_eq!(cfg.token_file.as_deref(), Some("/t"));
    }

    #[test]
    fn debug_output_redacts_secrets() {
        let cfg = CloudConfig::new("secret-token", "secret-key");
        let shown = format!("{cfg:?}");
        assert!(!shown.contains("secret-token") && !shown.contains("secret-key"));
    }

    #[test]
    fn auth_hints_match_python() {
        assert!(looks_like_auth_error("Token Expired"));
        assert!(looks_like_auth_error("请先登录"));
        assert!(!looks_like_auth_error("mission not found"));
    }

    #[tokio::test]
    async fn token_file_wins_over_static_token_and_is_reread() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join(".dji_token");
        let mut cfg = CloudConfig::new("static", "key");
        cfg.token_file = Some(file.to_string_lossy().into_owned());
        let client = CloudClient::new(cfg);
        assert_eq!(client.resolve_token(false).await, "static"); // file missing
        std::fs::write(&file, "fresh\n").unwrap();
        assert_eq!(client.resolve_token(false).await, "fresh");
        std::fs::write(&file, "fresher").unwrap();
        assert_eq!(client.resolve_token(false).await, "fresher");
    }

    /// A local stand-in for flyapi.djigate.com that answers compare/check.
    async fn mock_dji(first_auth_fails: bool) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        use axum::{Json, Router, http::HeaderMap, routing::post};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let app = Router::new()
            .route(
                "/api/waypoint/mission/compare",
                post(move |headers: HeaderMap| {
                    let counter = counter.clone();
                    async move {
                        let n = counter.fetch_add(1, Ordering::SeqCst);
                        assert!(headers.contains_key("x-wk-sign"));
                        if first_auth_fails && n == 0 {
                            return Json(json!({"result": {"code": 1, "msg": "token expired"}}));
                        }
                        Json(json!({"result": {"code": 0}, "data": {"mission_download_list": ["u1"]}}))
                    }
                }),
            )
            .route(
                "/api/waypoint/mission/check",
                post(|| async {
                    Json(json!({"result": {"code": 0}, "data": {"items": [
                        {"uuid": "u1", "name": "route", "point_num": 2, "distance": 28.5}
                    ]}}))
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), calls)
    }

    #[tokio::test]
    async fn list_missions_compares_then_checks() {
        let (base, _) = mock_dji(false).await;
        let client = CloudClient::with_base_url(CloudConfig::new("t", "k"), &base);
        let items = client.list_missions().await.unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["uuid"], "u1");
    }

    #[tokio::test]
    async fn auth_rejection_retries_once_only_with_a_token_source() {
        // Static token: no retry, the auth error surfaces.
        let (base, calls) = mock_dji(true).await;
        let client = CloudClient::with_base_url(CloudConfig::new("t", "k"), &base);
        let err = client.list_missions().await.unwrap_err();
        assert!(matches!(err, CloudError::Auth(ref m) if m.contains("rejected: token expired")));
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Token file configured: one retry, which succeeds.
        let (base, calls) = mock_dji(true).await;
        let mut cfg = CloudConfig::new("t", "k");
        cfg.token_file = Some("does-not-exist".into());
        let client = CloudClient::with_base_url(cfg, &base);
        assert_eq!(client.list_missions().await.unwrap().len(), 1);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }
}
