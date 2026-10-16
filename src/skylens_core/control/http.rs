//! Fire-and-collect HTTP for the tower: planner map tiles, the core's `/site`, the mission bridge.
//! Requests run in the background (a thread each on native, `fetch` on the web) and their results
//! are collected in [`Http::drain`] from the screen's poll, which the core link wakes every 100 ms.

use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
pub enum Method {
    Get,
    /// POST with a JSON body.
    PostJson(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub method: Method,
    pub url: String,
    pub timeout_ms: u64,
}

impl Request {
    pub fn get(url: impl Into<String>, timeout_ms: u64) -> Request {
        Request { method: Method::Get, url: url.into(), timeout_ms }
    }

    pub fn post_json(url: impl Into<String>, body: String, timeout_ms: u64) -> Request {
        Request { method: Method::PostJson(body), url: url.into(), timeout_ms }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    pub status: u16,
    pub content_type: String,
    pub body: Vec<u8>,
}

impl Response {
    pub fn ok(&self) -> bool {
        (200..300).contains(&self.status)
    }

    pub fn json(&self) -> Option<serde_json::Value> {
        serde_json::from_slice(&self.body).ok()
    }
}

/// A transport failure, worded like the JavaScript error the legacy code stringified (e.g.
/// `TypeError: Failed to fetch`).
pub type Outcome = Result<Response, String>;

/// Background requests tagged with `T`, collected later.
pub struct Http<T: 'static> {
    done: Rc<RefCell<Vec<(T, Outcome)>>>,
    #[cfg(not(target_arch = "wasm32"))]
    rx: std::sync::mpsc::Receiver<(T, Outcome)>,
    #[cfg(not(target_arch = "wasm32"))]
    tx: std::sync::mpsc::Sender<(T, Outcome)>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + 'static> Http<T> {
    pub fn new() -> Http<T> {
        let (tx, rx) = std::sync::mpsc::channel();
        Http { done: Rc::default(), rx, tx }
    }

    pub fn send(&self, tag: T, req: Request) {
        let tx = self.tx.clone();
        let _ = std::thread::Builder::new().name("skylens-http".into()).spawn(move || {
            let _ = tx.send((tag, native_fetch(&req)));
        });
    }

    pub fn drain(&self) -> Vec<(T, Outcome)> {
        let mut out: Vec<(T, Outcome)> = self.done.borrow_mut().drain(..).collect();
        out.extend(self.rx.try_iter());
        out
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn native_fetch(req: &Request) -> Outcome {
    let agent = ureq::AgentBuilder::new().timeout(std::time::Duration::from_millis(req.timeout_ms.max(1))).build();
    let result = match &req.method {
        Method::Get => agent.get(&req.url).call(),
        Method::PostJson(body) => agent.post(&req.url).set("Content-Type", "application/json").send_string(body),
    };
    let resp = match result {
        Ok(r) => r,
        // A non-2xx answer is still an answer (the legacy code read its JSON error).
        Err(ureq::Error::Status(_, r)) => r,
        Err(ureq::Error::Transport(t)) => return Err(format!("TypeError: {t}")),
    };
    let status = resp.status();
    let content_type = resp.header("content-type").unwrap_or("").to_string();
    let mut body = Vec::new();
    use std::io::Read as _;
    if let Err(e) = resp.into_reader().take(64 * 1024 * 1024).read_to_end(&mut body) {
        return Err(format!("TypeError: {e}"));
    }
    Ok(Response { status, content_type, body })
}

#[cfg(target_arch = "wasm32")]
impl<T: 'static> Http<T> {
    pub fn new() -> Http<T> {
        Http { done: Rc::default() }
    }

    pub fn send(&self, tag: T, req: Request) {
        let done = self.done.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let outcome = web_fetch(&req).await;
            done.borrow_mut().push((tag, outcome));
        });
    }

    pub fn drain(&self) -> Vec<(T, Outcome)> {
        self.done.borrow_mut().drain(..).collect()
    }
}

#[cfg(target_arch = "wasm32")]
async fn web_fetch(req: &Request) -> Outcome {
    use wasm_bindgen::JsCast as _;
    use wasm_bindgen_futures::JsFuture;

    let js_err = |e: wasm_bindgen::JsValue| -> String {
        if let Some(err) = e.dyn_ref::<js_sys::Error>() {
            String::from(err.to_string())
        } else {
            e.as_string().unwrap_or_else(|| format!("{e:?}"))
        }
    };
    let window = web_sys::window().ok_or_else(|| "TypeError: no window".to_string())?;
    let init = web_sys::RequestInit::new();
    init.set_mode(web_sys::RequestMode::Cors);
    let signal = web_sys::AbortSignal::timeout_with_u32(req.timeout_ms.min(u64::from(u32::MAX)) as u32);
    init.set_signal(Some(&signal));
    match &req.method {
        Method::Get => init.set_method("GET"),
        Method::PostJson(body) => {
            init.set_method("POST");
            let headers = web_sys::Headers::new().map_err(js_err)?;
            headers.set("Content-Type", "application/json").map_err(js_err)?;
            init.set_headers(&headers);
            init.set_body(&wasm_bindgen::JsValue::from_str(body));
        }
    }
    let request = web_sys::Request::new_with_str_and_init(&req.url, &init).map_err(js_err)?;
    let resp = JsFuture::from(window.fetch_with_request(&request)).await.map_err(js_err)?;
    let resp: web_sys::Response = resp.dyn_into().map_err(js_err)?;
    let status = resp.status();
    let content_type = resp.headers().get("content-type").ok().flatten().unwrap_or_default();
    let buf = JsFuture::from(resp.array_buffer().map_err(js_err)?).await.map_err(js_err)?;
    let body = js_sys::Uint8Array::new(&buf).to_vec();
    Ok(Response { status, content_type, body })
}

/// The core's HTTP origin for a WebSocket URL (`ws://host:8080/control` -> `http://host:8080`).
pub fn http_origin(ws_url: &str) -> String {
    let (scheme, rest) = match ws_url.split_once("://") {
        Some(("wss", r)) => ("https", r),
        Some(("https", r)) => ("https", r),
        Some((_, r)) => ("http", r),
        None => ("http", ws_url),
    };
    let host = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    format!("{scheme}://{host}")
}

/// The host (without port) of a URL.
pub fn host_of(url: &str) -> String {
    let rest = url.split_once("://").map_or(url, |(_, r)| r);
    let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    let authority = authority.rsplit_once('@').map_or(authority, |(_, a)| a);
    if let Some(stripped) = authority.strip_prefix('[') {
        return stripped.split(']').next().unwrap_or(stripped).to_string();
    }
    authority.split(':').next().unwrap_or(authority).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origins_follow_the_socket() {
        assert_eq!(http_origin("ws://127.0.0.1:8080/control"), "http://127.0.0.1:8080");
        assert_eq!(http_origin("wss://core.example/control?x=1"), "https://core.example");
        assert_eq!(http_origin("core.local:8080"), "http://core.local:8080");
        assert_eq!(host_of("ws://10.0.0.5:8080/control"), "10.0.0.5");
        assert_eq!(host_of("ws://[::1]:8080/control"), "::1");
        assert_eq!(host_of("core.local"), "core.local");
    }

    #[test]
    fn responses_report_success_and_json() {
        let r = Response { status: 200, content_type: "application/json".into(), body: b"{\"ok\":true}".to_vec() };
        assert!(r.ok());
        assert_eq!(r.json().unwrap()["ok"], true);
        let e = Response { status: 400, content_type: String::new(), body: vec![] };
        assert!(!e.ok() && e.json().is_none());
    }
}
