//! The board server: what the bridge nginx (`src/skylens_gateway/nginx-skylens.conf`)
//! does in production, in-process so the demo needs no nginx.
//!
//! | path        | production                          | here                              |
//! |-------------|-------------------------------------|-----------------------------------|
//! | `/`         | status board web build              | `src/skylens_client/dist`         |
//! | `/viewer`   | WebSocket -> proxy -> core `/viewer`| WebSocket -> core `/viewer`       |
//! | `/uplink`   | WebSocket -> proxy -> core `/uplink`| WebSocket -> core `/uplink`       |
//! | `/live/`    | WHEP -> Daejeon MediaMTX :10889     | WHEP -> local MediaMTX :10889     |
//! | `/health`   | (none)                              | `ok`, for the launcher            |

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::extract::ws::{Message as AxMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{Request, State};
use axum::http::{HeaderName, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get};
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message as TMessage;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use crate::http1;

/// Largest WebSocket message relayed (point and mesh chunks can be large).
pub const MAX_WS_MESSAGE: usize = 512 << 20;
/// nginx `client_max_body_size 1m`.
const MAX_LIVE_BODY: usize = 1 << 20;

#[derive(Debug, Clone)]
pub struct BoardConfig {
    /// Status board web build (`status.html`, `pkg/`).
    pub dist: PathBuf,
    /// Core `host:port`.
    pub core: String,
    /// MediaMTX WHEP `host:port`.
    pub whep: String,
}

pub fn router(cfg: BoardConfig) -> Router {
    let cfg = Arc::new(cfg);
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/viewer", get(ws_viewer))
        .route("/uplink", get(ws_uplink))
        .route("/live/{*rest}", any(live))
        .fallback(static_file)
        .with_state(cfg)
}

// ---- static files ---------------------------------------------------------------

/// The file under `dist` a request path names, or `None` when the path tries
/// to leave `dist`. Directories (and `/`) resolve to their `status.html`, like
/// nginx `index status.html`.
pub fn static_rel(uri_path: &str) -> Option<PathBuf> {
    let mut rel = PathBuf::new();
    for seg in uri_path.split('/') {
        match seg {
            "" | "." => continue,
            ".." => return None,
            s if s.contains('\\') || s.contains(':') || s.contains('\0') => return None,
            s => rel.push(s),
        }
    }
    if uri_path.ends_with('/') || rel.as_os_str().is_empty() {
        rel.push("status.html");
    }
    Some(rel)
}

pub fn mime_for(path: &Path) -> &'static str {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "html" | "htm" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "wasm" => "application/wasm",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "svg" => "image/svg+xml",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "ttf" => "font/ttf",
        "txt" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

async fn static_file(State(cfg): State<Arc<BoardConfig>>, req: Request) -> Response {
    if !cfg.dist.join("status.html").is_file() {
        let msg = format!(
            "현황판 웹 빌드가 없습니다: {}\n먼저 bash src/skylens_client/build-web.sh 를 실행하십시오.\n",
            cfg.dist.display()
        );
        return (StatusCode::SERVICE_UNAVAILABLE, [(header::CONTENT_TYPE, "text/plain; charset=utf-8")], msg).into_response();
    }
    let Some(rel) = static_rel(req.uri().path()) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let mut path = cfg.dist.join(&rel);
    if path.is_dir() {
        path.push("status.html");
    }
    match tokio::fs::read(&path).await {
        Ok(bytes) => (
            [(header::CONTENT_TYPE, mime_for(&path)), (header::CACHE_CONTROL, "no-cache")],
            bytes,
        )
            .into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

// ---- WebSocket relay --------------------------------------------------------------

pub fn ws_config() -> WebSocketConfig {
    WebSocketConfig::default()
        .max_message_size(Some(MAX_WS_MESSAGE))
        .max_frame_size(Some(MAX_WS_MESSAGE))
}

async fn ws_viewer(State(cfg): State<Arc<BoardConfig>>, ws: WebSocketUpgrade) -> Response {
    relay(&cfg, ws, "/viewer").await
}

async fn ws_uplink(State(cfg): State<Arc<BoardConfig>>, ws: WebSocketUpgrade) -> Response {
    relay(&cfg, ws, "/uplink").await
}

/// Connect to the core first, so a missing core is a 502 rather than a socket
/// that opens and immediately closes.
async fn relay(cfg: &BoardConfig, ws: WebSocketUpgrade, path: &'static str) -> Response {
    let url = format!("ws://{}{path}", cfg.core);
    match tokio_tungstenite::connect_async_with_config(url.as_str(), Some(ws_config()), true).await {
        Ok((upstream, _)) => ws
            .max_message_size(MAX_WS_MESSAGE)
            .max_frame_size(MAX_WS_MESSAGE)
            .on_upgrade(move |client| pump(client, upstream)),
        Err(e) => (StatusCode::BAD_GATEWAY, format!("core {url}: {e}")).into_response(),
    }
}

/// Copy data frames both ways until either side closes. Pings are answered by
/// each side's own WebSocket stack, so they are not relayed.
async fn pump(client: WebSocket, upstream: WebSocketStream<MaybeTlsStream<TcpStream>>) {
    let (mut c_tx, mut c_rx) = client.split();
    let (mut u_tx, mut u_rx) = upstream.split();
    let down = async {
        while let Some(Ok(msg)) = u_rx.next().await {
            let out = match msg {
                TMessage::Binary(b) => AxMessage::Binary(b),
                TMessage::Text(t) => AxMessage::Text(t.as_str().into()),
                TMessage::Close(_) => break,
                _ => continue,
            };
            if c_tx.send(out).await.is_err() {
                break;
            }
        }
        let _ = c_tx.send(AxMessage::Close(None)).await;
    };
    let up = async {
        while let Some(Ok(msg)) = c_rx.next().await {
            let out = match msg {
                AxMessage::Binary(b) => TMessage::Binary(b),
                AxMessage::Text(t) => TMessage::Text(t.as_str().into()),
                AxMessage::Close(_) => break,
                _ => continue,
            };
            if u_tx.send(out).await.is_err() {
                break;
            }
        }
        let _ = u_tx.send(TMessage::Close(None)).await;
    };
    tokio::select! {
        _ = down => {}
        _ = up => {}
    }
}

// ---- WHEP signalling -----------------------------------------------------------------

/// `/live/<rest>?q` -> `/<rest>?q` on MediaMTX (nginx `proxy_pass http://…/;`).
pub fn live_upstream_path(path_and_query: &str) -> String {
    let rest = path_and_query.strip_prefix("/live").unwrap_or(path_and_query);
    if rest.starts_with('/') { rest.to_string() } else { format!("/{rest}") }
}

/// nginx `proxy_redirect / /live/`: a session URL MediaMTX returns as `/x`
/// (or absolute on the upstream) comes back under `/live/x`.
pub fn rewrite_location(location: &str, upstream: &str) -> String {
    let local = location.strip_prefix(&format!("http://{upstream}")).unwrap_or(location);
    match local.strip_prefix('/') {
        Some(rest) if !local.starts_with("//") => format!("/live/{rest}"),
        _ => location.to_string(),
    }
}

async fn live(State(cfg): State<Arc<BoardConfig>>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let path = live_upstream_path(parts.uri.path_and_query().map(|p| p.as_str()).unwrap_or("/"));
    let body = match axum::body::to_bytes(body, MAX_LIVE_BODY).await {
        Ok(b) => b,
        Err(_) => return StatusCode::PAYLOAD_TOO_LARGE.into_response(),
    };
    let mut headers: Vec<(String, String)> = Vec::new();
    for (k, v) in &parts.headers {
        if http1::is_hop_by_hop(k.as_str()) {
            continue;
        }
        if let Ok(v) = v.to_str() {
            headers.push((k.as_str().to_string(), v.to_string()));
        }
    }
    headers.push(("X-Forwarded-Proto".into(), "http".into()));
    let res = match http1::request(&cfg.whep, parts.method.as_str(), &path, &headers, &body, Duration::from_secs(60)).await {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_GATEWAY, format!("MediaMTX {}: {e}", cfg.whep)).into_response(),
    };
    let mut out = axum::http::Response::builder().status(StatusCode::from_u16(res.status).unwrap_or(StatusCode::BAD_GATEWAY));
    for (k, v) in &res.headers {
        if http1::is_hop_by_hop(k) {
            continue;
        }
        let v = if k.eq_ignore_ascii_case("location") { rewrite_location(v, &cfg.whep) } else { v.clone() };
        if let (Ok(name), Ok(value)) = (HeaderName::try_from(k.as_str()), HeaderValue::try_from(v)) {
            out = out.header(name, value);
        }
    }
    out.body(Body::from(res.body)).unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_paths_stay_inside_dist() {
        assert_eq!(static_rel("/"), Some(PathBuf::from("status.html")));
        assert_eq!(static_rel(""), Some(PathBuf::from("status.html")));
        assert_eq!(static_rel("/status.html"), Some(PathBuf::from("status.html")));
        assert_eq!(static_rel("/pkg/status_screen_bg.wasm"), Some(Path::new("pkg").join("status_screen_bg.wasm")));
        assert_eq!(static_rel("/pkg/"), Some(Path::new("pkg").join("status.html")));
        assert_eq!(static_rel("/./favicon.png"), Some(PathBuf::from("favicon.png")));
        assert_eq!(static_rel("/../secret"), None);
        assert_eq!(static_rel("/pkg/../../x"), None);
        assert_eq!(static_rel("/C:/Windows"), None);
        assert_eq!(static_rel("/a\\..\\b"), None);
    }

    #[test]
    fn mime_types_for_the_web_build() {
        assert_eq!(mime_for(Path::new("status.html")), "text/html; charset=utf-8");
        assert_eq!(mime_for(Path::new("pkg/x.wasm")), "application/wasm");
        assert_eq!(mime_for(Path::new("pkg/x.JS")), "text/javascript; charset=utf-8");
        assert_eq!(mime_for(Path::new("favicon.png")), "image/png");
        assert_eq!(mime_for(Path::new("blob")), "application/octet-stream");
    }

    #[test]
    fn live_paths_drop_the_prefix() {
        assert_eq!(live_upstream_path("/live/drone1/whep"), "/drone1/whep");
        assert_eq!(live_upstream_path("/live/drone1/whep?x=1"), "/drone1/whep?x=1");
        assert_eq!(live_upstream_path("/live/"), "/");
    }

    #[test]
    fn session_location_comes_back_under_live() {
        assert_eq!(rewrite_location("/drone1/whep/abc", "127.0.0.1:10889"), "/live/drone1/whep/abc");
        assert_eq!(
            rewrite_location("http://127.0.0.1:10889/drone1/whep/abc", "127.0.0.1:10889"),
            "/live/drone1/whep/abc"
        );
        assert_eq!(rewrite_location("https://elsewhere/x", "127.0.0.1:10889"), "https://elsewhere/x");
        assert_eq!(rewrite_location("whep/abc", "127.0.0.1:10889"), "whep/abc");
    }

    // ---- served end to end on loopback ----

    async fn serve(app: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        addr
    }

    fn temp_dist(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("skylens-demo-board-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("pkg")).unwrap();
        std::fs::write(dir.join("status.html"), "<html>board</html>").unwrap();
        std::fs::write(dir.join("pkg").join("a.wasm"), [0u8, 97, 115, 109]).unwrap();
        dir
    }

    fn cfg(dist: PathBuf, core: &str, whep: &str) -> BoardConfig {
        BoardConfig { dist, core: core.to_string(), whep: whep.to_string() }
    }

    #[tokio::test]
    async fn serves_the_board_and_refuses_traversal() {
        let dist = temp_dist("static");
        let addr = serve(router(cfg(dist.clone(), "127.0.0.1:1", "127.0.0.1:1"))).await;
        let t = Duration::from_secs(5);
        let root = http1::request(&addr, "GET", "/", &[], &[], t).await.unwrap();
        assert_eq!(root.status, 200);
        assert_eq!(root.header("content-type"), Some("text/html; charset=utf-8"));
        assert_eq!(root.body, b"<html>board</html>");
        let wasm = http1::request(&addr, "GET", "/pkg/a.wasm", &[], &[], t).await.unwrap();
        assert_eq!(wasm.header("content-type"), Some("application/wasm"));
        assert_eq!(http1::request(&addr, "GET", "/missing.js", &[], &[], t).await.unwrap().status, 404);
        assert_eq!(http1::request(&addr, "GET", "/health", &[], &[], t).await.unwrap().body, b"ok");
        let _ = std::fs::remove_dir_all(&dist);
    }

    #[tokio::test]
    async fn missing_build_is_a_clear_503() {
        let dist = std::env::temp_dir().join("skylens-demo-board-no-such-dist");
        let addr = serve(router(cfg(dist, "127.0.0.1:1", "127.0.0.1:1"))).await;
        let r = http1::request(&addr, "GET", "/", &[], &[], Duration::from_secs(5)).await.unwrap();
        assert_eq!(r.status, 503);
        assert!(String::from_utf8_lossy(&r.body).contains("build-web.sh"));
    }

    #[tokio::test]
    async fn proxies_whep_and_rewrites_the_session_location() {
        use axum::http::Method;
        // A stand-in MediaMTX: POST /drone1/whep answers 201 + Location.
        let fake = Router::new().route(
            "/drone1/whep",
            any(|method: Method, headers: axum::http::HeaderMap, body: String| async move {
                assert_eq!(method, Method::POST);
                assert_eq!(headers.get("content-type").unwrap(), "application/sdp");
                (
                    StatusCode::CREATED,
                    [(header::LOCATION, "/drone1/whep/session-1"), (header::CONTENT_TYPE, "application/sdp")],
                    format!("answer to {body}"),
                )
            }),
        );
        let whep = serve(fake).await;
        let addr = serve(router(cfg(temp_dist("live"), "127.0.0.1:1", &whep))).await;
        let r = http1::request(
            &addr,
            "POST",
            "/live/drone1/whep",
            &[("Content-Type".into(), "application/sdp".into())],
            b"offer",
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        assert_eq!(r.status, 201);
        assert_eq!(r.header("location"), Some("/live/drone1/whep/session-1"));
        assert_eq!(r.body, b"answer to offer");
    }

    #[tokio::test]
    async fn relays_websocket_frames_to_the_core() {
        // A stand-in core: echoes binary frames on /viewer.
        async fn echo(ws: WebSocketUpgrade) -> Response {
            ws.on_upgrade(|mut s| async move {
                while let Some(Ok(m)) = s.next().await {
                    if let AxMessage::Binary(b) = m {
                        let mut v = b.to_vec();
                        v.reverse();
                        if s.send(AxMessage::Binary(v.into())).await.is_err() {
                            break;
                        }
                    }
                }
            })
        }
        let core = serve(Router::new().route("/viewer", get(echo))).await;
        let addr = serve(router(cfg(temp_dist("ws"), &core, "127.0.0.1:1"))).await;
        let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/viewer")).await.unwrap();
        ws.send(TMessage::Binary(vec![1u8, 2, 3].into())).await.unwrap();
        let reply = loop {
            match ws.next().await.unwrap().unwrap() {
                TMessage::Binary(b) => break b,
                _ => continue,
            }
        };
        assert_eq!(&reply[..], &[3, 2, 1]);
    }

    #[tokio::test]
    async fn no_core_is_a_bad_gateway() {
        let addr = serve(router(cfg(temp_dist("nocore"), "127.0.0.1:1", "127.0.0.1:1"))).await;
        let err = tokio_tungstenite::connect_async(format!("ws://{addr}/viewer")).await.unwrap_err();
        match err {
            tokio_tungstenite::tungstenite::Error::Http(resp) => assert_eq!(resp.status(), 502),
            other => panic!("expected an HTTP error, got {other}"),
        }
    }
}
