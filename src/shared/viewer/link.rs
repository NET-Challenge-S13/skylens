//! The one WebSocket to the core (`ws /control` for the tower, `ws /viewer` for boards).
//!
//! Every frame is one binary `skylens.viewer.v1.Envelope`. The proxy does not migrate sockets, so
//! when the link drops, or the core is not up yet, it keeps retrying with a backoff (0.5 s doubling
//! to 5 s) and the core replays what it holds; screens clear their stores on
//! [`LinkEvent::Connected`]. Native also gives up on a handshake after
//! [`crate::config::link::CONNECT_TIMEOUT_MS`] and on a silent connection after
//! [`crate::config::link::IDLE_TIMEOUT_MS`], so a half-open socket cannot stall the screen.
//!
//! A native link can be given several addresses ([`CoreLink::start_any`]): every round tries them
//! in order and stays on the first that answers, so a screen can prefer the bridge and fall back
//! to a core on this machine. `wss` runs on rustls with the web roots plus [`PINNED_CERTS`], the
//! bridge's self-signed board certificate as the build fetched it, trusted as exactly that
//! certificate.
//!
//! Native runs tokio-tungstenite on a background thread; web uses the browser WebSocket. Both
//! queue events for [`CoreLink::drain`] and wake the Blitz window with the waker given to
//! [`CoreLink::set_waker`] (from `Document::poll`). A ticker wakes it every [`TICK_MS`] too, for
//! countdowns, toasts and the 10 Hz manual-control stream.
//!
//! [`LinkStatus`] turns the events into the legacy wording of `coreLink.ts` (tower) and
//! `relayBadge.ts` (board badge).

use crate::protocol::{self, Body, Outbox};
use skylens_shared::proto::viewer::v1 as viewer;
use std::task::Waker;

pub const TICK_MS: u64 = 100;

/// Default socket paths on the core.
pub const CONTROL_PATH: &str = "/control";
pub const VIEWER_PATH: &str = "/viewer";

/// Server certificates (DER) a native `wss` link accepts as they are, whatever name it dialled:
/// the bridge's self-signed board certificate, which `build.rs` asks the bridge for at build time
/// (empty when the bridge did not answer). If the bridge makes a new one, rebuild, or that address
/// stops answering and the link falls through to the next one.
pub const PINNED_CERTS: &[&[u8]] = include!(concat!(env!("OUT_DIR"), "/pinned_certs.rs"));

#[derive(Debug, Clone, PartialEq)]
pub enum LinkEvent {
    Connecting,
    Connected,
    /// The socket closed or could not open; the text says why.
    Disconnected(String),
    Message(viewer::Envelope),
    /// A frame that is not an Envelope (dropped, reported for diagnostics).
    Invalid(String),
}

/// Raw events from the transport, before decoding.
#[derive(Debug)]
enum RawEvent {
    Connecting,
    Connected,
    Disconnected(String),
    Frame(Vec<u8>),
}

fn decode(ev: RawEvent) -> LinkEvent {
    match ev {
        RawEvent::Connecting => LinkEvent::Connecting,
        RawEvent::Connected => LinkEvent::Connected,
        RawEvent::Disconnected(r) => LinkEvent::Disconnected(r),
        RawEvent::Frame(bytes) => match protocol::decode_envelope(&bytes) {
            Ok(env) => LinkEvent::Message(env),
            Err(e) => LinkEvent::Invalid(e.to_string()),
        },
    }
}

/// Accepts `host:port`, `http(s)://…` or `ws(s)://…`; adds `default_path` when no path is given.
pub fn normalize_core_url(s: &str, default_path: &str) -> String {
    let s = s.trim();
    let with_scheme = if let Some(rest) = s.strip_prefix("http://") {
        format!("ws://{rest}")
    } else if let Some(rest) = s.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if s.contains("://") {
        s.to_string()
    } else {
        format!("ws://{s}")
    };
    let path = default_path.trim_start_matches('/');
    let after_scheme = with_scheme.split_once("://").map_or("", |(_, r)| r);
    match after_scheme.find('/') {
        None => format!("{with_scheme}/{path}"),
        Some(i) if &after_scheme[i..] == "/" => format!("{with_scheme}{path}"),
        Some(_) => with_scheme,
    }
}

/// The socket URL for a page served from `host` (`host[:port]`), over TLS when `secure`.
/// The core serves the tower build and the gateway proxies `/viewer`, so the socket lives on the
/// page's own origin.
pub fn page_core_url(secure: bool, host: &str, path: &str) -> String {
    format!("{}://{host}/{}", if secure { "wss" } else { "ws" }, path.trim_start_matches('/'))
}

/// Link state as the screens show it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkState {
    Connecting,
    Connected,
    Disconnected,
}

/// Colour class of the badge dot (legacy `relayBadge.ts`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Ok,
    Warn,
    Down,
}

impl Severity {
    /// Dot colour, exactly the legacy values.
    pub fn dot_css(self) -> &'static str {
        match self {
            Severity::Ok => "#39d98a",
            Severity::Warn => "#ffd27f",
            Severity::Down => "#ff4d4d",
        }
    }
}

/// What the screens know about the link.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkStatus {
    pub state: LinkState,
    pub ever_connected: bool,
    /// Data (chunks, detections, telemetry) is flowing on this connection.
    pub receiving: bool,
    /// Why the last attempt failed.
    pub reason: Option<String>,
    /// Connection attempts started since the last successful connect (the current one included).
    pub attempts: u32,
    /// Frames received on this connection.
    pub frames: u64,
    /// Unix ms of the last frame; 0 before the first one.
    pub last_frame_ms: f64,
    /// Frames that could not be decoded.
    pub invalid: u64,
    /// Sends refused because the core was not connected.
    pub dropped_sends: u64,
}

impl Default for LinkStatus {
    fn default() -> Self {
        Self {
            state: LinkState::Connecting,
            ever_connected: false,
            receiving: false,
            reason: None,
            attempts: 0,
            frames: 0,
            last_frame_ms: 0.0,
            invalid: 0,
            dropped_sends: 0,
        }
    }
}

impl LinkStatus {
    /// Folds one event in. Returns true when anything the screens show changed.
    pub fn observe(&mut self, ev: &LinkEvent, now_ms: f64) -> bool {
        let before = (self.state, self.receiving, self.ever_connected);
        match ev {
            LinkEvent::Connecting => {
                self.state = LinkState::Connecting;
                self.attempts = self.attempts.saturating_add(1);
            }
            LinkEvent::Connected => {
                self.state = LinkState::Connected;
                self.ever_connected = true;
                self.reason = None;
                self.attempts = 0;
                self.frames = 0;
                self.receiving = false;
            }
            LinkEvent::Disconnected(reason) => {
                self.state = LinkState::Disconnected;
                self.receiving = false;
                self.reason = Some(reason.clone());
            }
            LinkEvent::Message(env) => {
                self.frames += 1;
                self.last_frame_ms = now_ms;
                use protocol::Body as B;
                if matches!(env.body, Some(B::Chunk(_) | B::Telemetry(_))) {
                    self.receiving = true;
                }
            }
            LinkEvent::Invalid(_) => self.invalid += 1,
        }
        before != (self.state, self.receiving, self.ever_connected)
    }

    /// The tower's wording (legacy `coreLink.ts` state details).
    pub fn detail(&self) -> &'static str {
        match self.state {
            LinkState::Connecting if self.ever_connected => "코어 재연결 시도 중",
            LinkState::Connecting => "코어 연결 시도 중",
            LinkState::Connected => "코어 연결됨",
            LinkState::Disconnected if self.ever_connected => "코어 연결 끊김",
            LinkState::Disconnected => "코어에 연결할 수 없습니다",
        }
    }

    /// The board badge's wording (legacy `relayClient.ts` describe(), upstream half).
    pub fn badge_text(&self) -> &'static str {
        match self.state {
            LinkState::Connected if self.receiving => "코어 스트림 수신 중",
            LinkState::Connected => "코어 연결됨 · 데이터 대기",
            LinkState::Connecting => "코어 연결 시도 중",
            LinkState::Disconnected => "코어 응답 없음 · 재연결 대기",
        }
    }

    pub fn severity(&self) -> Severity {
        match self.state {
            LinkState::Connected => Severity::Ok,
            LinkState::Connecting => Severity::Warn,
            LinkState::Disconnected => Severity::Down,
        }
    }

    pub fn connected(&self) -> bool {
        self.state == LinkState::Connected
    }
}

/// The link plus its outbox: what a screen holds.
pub struct CoreLink {
    transport: Transport,
    outbox: Outbox,
    url: String,
    pub status: LinkStatus,
}

impl CoreLink {
    /// Opens the socket in the background and keeps it open until dropped.
    pub fn start(url: impl Into<String>) -> CoreLink {
        CoreLink::start_any(vec![url.into()])
    }

    /// Like [`Self::start`], trying `urls` in order on every round (the web build, which always
    /// talks to its own origin, uses only the first).
    pub fn start_any(urls: Vec<String>) -> CoreLink {
        assert!(!urls.is_empty(), "CoreLink needs an address");
        CoreLink {
            url: urls[0].clone(),
            transport: Transport::start(urls),
            outbox: Outbox::new(),
            status: LinkStatus::default(),
        }
    }

    /// The first (preferred) address.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// The address the link is on or trying now.
    pub fn current_url(&self) -> String {
        self.transport.current_url()
    }

    /// Wakes this waker when an event arrives (pass the one from `Document::poll`).
    pub fn set_waker(&self, waker: &Waker) {
        self.transport.set_waker(waker);
    }

    /// Events since the last call, decoded, with [`Self::status`] already updated.
    pub fn drain(&mut self) -> Vec<LinkEvent> {
        let now = crate::now_ms();
        let events: Vec<LinkEvent> = self.transport.drain().into_iter().map(decode).collect();
        for ev in &events {
            self.status.observe(ev, now);
        }
        events
    }

    /// Sends one control message. False when the core is not connected: nothing is buffered
    /// across reconnects, because stale control must not fly a drone later.
    pub fn send(&mut self, body: Body) -> bool {
        let bytes = self.outbox.encode(body, crate::now_ms() as i64);
        let ok = self.transport.send(bytes);
        if !ok {
            self.status.dropped_sends += 1;
        }
        ok
    }

    pub fn connected(&self) -> bool {
        self.transport.connected()
    }
}

#[cfg(not(target_arch = "wasm32"))]
use native::Transport;
#[cfg(target_arch = "wasm32")]
use web::Transport;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::{RawEvent, TICK_MS, Waker};
    use crate::config::link::{CONNECT_TIMEOUT_MS, IDLE_TIMEOUT_MS, MAX_MESSAGE_BYTES};
    use crate::protocol::Backoff;
    use futures_util::{SinkExt as _, StreamExt as _};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tokio_tungstenite::tungstenite::Message as WsMessage;

    /// Message and frame limits large enough for point slot messages.
    fn ws_config() -> tokio_tungstenite::tungstenite::protocol::WebSocketConfig {
        tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default()
            .max_message_size(Some(MAX_MESSAGE_BYTES))
            .max_frame_size(Some(MAX_MESSAGE_BYTES))
    }

    #[derive(Default)]
    struct Shared {
        waker: Mutex<Option<Waker>>,
        connected: AtomicBool,
        /// The address being tried or held.
        current: Mutex<String>,
    }

    impl Shared {
        fn wake(&self) {
            if let Ok(guard) = self.waker.lock()
                && let Some(w) = guard.as_ref()
            {
                w.wake_by_ref();
            }
        }
    }

    pub struct Transport {
        events: mpsc::Receiver<RawEvent>,
        outgoing: tokio::sync::mpsc::UnboundedSender<Vec<u8>>,
        shared: Arc<Shared>,
    }

    impl Transport {
        pub fn start(urls: Vec<String>) -> Transport {
            let (ev_tx, ev_rx) = mpsc::channel();
            let (out_tx, out_rx) = tokio::sync::mpsc::unbounded_channel();
            let shared = Arc::new(Shared { current: Mutex::new(urls[0].clone()), ..Shared::default() });

            let net_shared = shared.clone();
            std::thread::Builder::new()
                .name("skylens-link".into())
                .spawn(move || {
                    let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                        Ok(rt) => rt,
                        Err(e) => {
                            let _ = ev_tx.send(RawEvent::Disconnected(format!("tokio: {e}")));
                            return;
                        }
                    };
                    rt.block_on(run(urls, ev_tx, out_rx, net_shared));
                })
                .expect("spawn link thread");

            let tick_shared = Arc::downgrade(&shared);
            std::thread::Builder::new()
                .name("skylens-tick".into())
                .spawn(move || {
                    loop {
                        std::thread::sleep(Duration::from_millis(TICK_MS));
                        match tick_shared.upgrade() {
                            Some(s) => s.wake(),
                            None => return,
                        }
                    }
                })
                .expect("spawn tick thread");

            Transport { events: ev_rx, outgoing: out_tx, shared }
        }

        pub fn set_waker(&self, waker: &Waker) {
            if let Ok(mut w) = self.shared.waker.lock()
                && !w.as_ref().is_some_and(|old| old.will_wake(waker))
            {
                *w = Some(waker.clone());
            }
        }

        pub fn drain(&self) -> Vec<RawEvent> {
            self.events.try_iter().collect()
        }

        pub fn send(&self, bytes: Vec<u8>) -> bool {
            self.shared.connected.load(Ordering::SeqCst) && self.outgoing.send(bytes).is_ok()
        }

        pub fn connected(&self) -> bool {
            self.shared.connected.load(Ordering::SeqCst)
        }

        pub fn current_url(&self) -> String {
            self.shared.current.lock().map(|u| u.clone()).unwrap_or_default()
        }
    }

    /// One client config for every `wss` address: the web roots, plus [`super::PINNED_CERTS`]
    /// accepted byte for byte. None (plain `ws` only) if the config cannot be built.
    fn tls_connector() -> Option<tokio_tungstenite::Connector> {
        match tls::client_config(super::PINNED_CERTS) {
            Ok(config) => Some(tokio_tungstenite::Connector::Rustls(Arc::new(config))),
            Err(e) => {
                crate::log(&format!("[link] TLS 설정 실패, wss 주소는 연결할 수 없습니다: {e}"));
                None
            }
        }
    }

    async fn run(
        urls: Vec<String>,
        events: mpsc::Sender<RawEvent>,
        mut outgoing: tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>,
        shared: Arc<Shared>,
    ) {
        let emit = |ev: RawEvent| -> bool {
            let ok = events.send(ev).is_ok();
            shared.wake();
            ok
        };
        let mut backoff = Backoff::default();
        let tls = tls_connector();
        loop {
            // One round: the addresses in order, until one answers and its connection ends.
            for url in &urls {
                if let Ok(mut current) = shared.current.lock() {
                    current.clone_from(url);
                }
                match session(url, tls.clone(), &emit, &mut outgoing, &shared).await {
                    Session::Stop => return,
                    Session::Unreachable => continue,
                    Session::Ended => {
                        backoff.reset();
                        break;
                    }
                }
            }
            // Back off, then start over from the preferred address.
            tokio::time::sleep(Duration::from_millis(backoff.next_delay_ms())).await;
        }
    }

    enum Session {
        /// The screen is gone.
        Stop,
        /// The handshake failed or timed out.
        Unreachable,
        /// Connected, then the connection ended.
        Ended,
    }

    async fn session(
        url: &str,
        tls: Option<tokio_tungstenite::Connector>,
        emit: &impl Fn(RawEvent) -> bool,
        outgoing: &mut tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>,
        shared: &Shared,
    ) -> Session {
        if !emit(RawEvent::Connecting) {
            return Session::Stop;
        }
        let attempt = tokio::time::timeout(
            Duration::from_millis(CONNECT_TIMEOUT_MS),
            tokio_tungstenite::connect_async_tls_with_config(url, Some(ws_config()), false, tls),
        );
        match attempt.await {
            Ok(Ok((ws, _))) => {
                // Drop anything queued while disconnected.
                while outgoing.try_recv().is_ok() {}
                shared.connected.store(true, Ordering::SeqCst);
                if !emit(RawEvent::Connected) {
                    return Session::Stop;
                }
                let (mut sink, mut stream) = ws.split();
                let idle_for = Duration::from_millis(IDLE_TIMEOUT_MS);
                let idle = tokio::time::sleep(idle_for);
                tokio::pin!(idle);
                let reason = loop {
                    tokio::select! {
                        // Any frame, pings included, proves the connection is alive.
                        () = &mut idle => break "코어 응답 없음 (수신 끊김)".to_string(),
                        msg = stream.next() => match msg {
                            Some(Ok(WsMessage::Binary(b))) => {
                                idle.as_mut().reset(tokio::time::Instant::now() + idle_for);
                                if !emit(RawEvent::Frame(b.to_vec())) {
                                    return Session::Stop;
                                }
                            }
                            Some(Ok(WsMessage::Close(frame))) => {
                                break frame.map_or_else(
                                    || "코어가 연결을 닫았습니다".to_string(),
                                    |f| format!("코어가 연결을 닫았습니다 ({})", u16::from(f.code)),
                                );
                            }
                            Some(Ok(_)) => idle.as_mut().reset(tokio::time::Instant::now() + idle_for),
                            Some(Err(e)) => break e.to_string(),
                            None => break "연결이 끊어졌습니다".to_string(),
                        },
                        out = outgoing.recv() => match out {
                            Some(bytes) => {
                                if let Err(e) = sink.send(WsMessage::binary(bytes)).await {
                                    break e.to_string();
                                }
                            }
                            // The CoreLink was dropped.
                            None => return Session::Stop,
                        },
                    }
                };
                shared.connected.store(false, Ordering::SeqCst);
                if !emit(RawEvent::Disconnected(reason)) {
                    return Session::Stop;
                }
                Session::Ended
            }
            Ok(Err(e)) => {
                if !emit(RawEvent::Disconnected(format!("{url}: {e}"))) {
                    return Session::Stop;
                }
                Session::Unreachable
            }
            Err(_) => {
                if !emit(RawEvent::Disconnected(format!("{url}: 코어 응답 없음 (연결 시간 초과)"))) {
                    return Session::Stop;
                }
                Session::Unreachable
            }
        }
    }

    mod tls {
        use rustls::client::WebPkiServerVerifier;
        use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
        use rustls::crypto::{CryptoProvider, verify_tls12_signature, verify_tls13_signature};
        use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
        use rustls::{ClientConfig, DigitallySignedStruct, Error, RootCertStore, SignatureScheme};
        use std::sync::Arc;

        /// rustls client config: web PKI, except that a server presenting one of `pinned` (DER)
        /// is accepted without a chain or a name check.
        pub fn client_config(pinned: &[&[u8]]) -> Result<ClientConfig, String> {
            let provider = Arc::new(rustls::crypto::ring::default_provider());
            let pinned = pinned.iter().map(|der| CertificateDer::from(der.to_vec())).collect();
            let roots = RootCertStore { roots: webpki_roots::TLS_SERVER_ROOTS.to_vec() };
            let webpki = WebPkiServerVerifier::builder_with_provider(Arc::new(roots), provider.clone())
                .build()
                .map_err(|e| e.to_string())?;
            let verifier = Pinned { pinned, webpki, provider: provider.clone() };
            Ok(ClientConfig::builder_with_provider(provider)
                .with_safe_default_protocol_versions()
                .map_err(|e| e.to_string())?
                .dangerous()
                .with_custom_certificate_verifier(Arc::new(verifier))
                .with_no_client_auth())
        }

        #[derive(Debug)]
        struct Pinned {
            pinned: Vec<CertificateDer<'static>>,
            webpki: Arc<WebPkiServerVerifier>,
            provider: Arc<CryptoProvider>,
        }

        impl ServerCertVerifier for Pinned {
            fn verify_server_cert(
                &self,
                end_entity: &CertificateDer<'_>,
                intermediates: &[CertificateDer<'_>],
                server_name: &ServerName<'_>,
                ocsp_response: &[u8],
                now: UnixTime,
            ) -> Result<ServerCertVerified, Error> {
                if self.pinned.iter().any(|p| p.as_ref() == end_entity.as_ref()) {
                    return Ok(ServerCertVerified::assertion());
                }
                self.webpki.verify_server_cert(end_entity, intermediates, server_name, ocsp_response, now)
            }

            // The handshake signatures are still checked against the presented key, so a copy
            // of the certificate is no use without its private key.
            fn verify_tls12_signature(
                &self,
                message: &[u8],
                cert: &CertificateDer<'_>,
                dss: &DigitallySignedStruct,
            ) -> Result<HandshakeSignatureValid, Error> {
                verify_tls12_signature(message, cert, dss, &self.provider.signature_verification_algorithms)
            }

            fn verify_tls13_signature(
                &self,
                message: &[u8],
                cert: &CertificateDer<'_>,
                dss: &DigitallySignedStruct,
            ) -> Result<HandshakeSignatureValid, Error> {
                verify_tls13_signature(message, cert, dss, &self.provider.signature_verification_algorithms)
            }

            fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
                self.provider.signature_verification_algorithms.supported_schemes()
            }
        }

        #[cfg(test)]
        mod tests {
            #[test]
            fn the_client_config_takes_the_pinned_certificates() {
                super::client_config(crate::link::PINNED_CERTS).expect("client config");
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use super::{RawEvent, TICK_MS, Waker};
    use crate::protocol::Backoff;
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::rc::Rc;
    use wasm_bindgen::JsCast as _;
    use wasm_bindgen::prelude::*;
    use web_sys::{BinaryType, CloseEvent, MessageEvent, WebSocket};

    struct Inner {
        url: String,
        ws: Option<WebSocket>,
        queue: VecDeque<RawEvent>,
        waker: Option<Waker>,
        connected: bool,
        stopped: bool,
        backoff: Backoff,
        on_open: Option<Closure<dyn FnMut()>>,
        on_message: Option<Closure<dyn FnMut(MessageEvent)>>,
        on_close: Option<Closure<dyn FnMut(CloseEvent)>>,
        tick: Option<(i32, Closure<dyn FnMut()>)>,
    }

    pub struct Transport {
        inner: Rc<RefCell<Inner>>,
    }

    fn push(inner: &Rc<RefCell<Inner>>, ev: RawEvent) {
        let waker = {
            let mut i = inner.borrow_mut();
            i.queue.push_back(ev);
            i.waker.clone()
        };
        if let Some(w) = waker {
            w.wake();
        }
    }

    fn connect(inner: &Rc<RefCell<Inner>>) {
        if inner.borrow().stopped {
            return;
        }
        push(inner, RawEvent::Connecting);
        let url = inner.borrow().url.clone();
        let ws = match WebSocket::new(&url) {
            Ok(ws) => ws,
            Err(e) => {
                push(inner, RawEvent::Disconnected(format!("코어 주소가 올바르지 않습니다 ({e:?})")));
                schedule_reconnect(inner);
                return;
            }
        };
        ws.set_binary_type(BinaryType::Arraybuffer);

        let i = inner.clone();
        let on_open = Closure::<dyn FnMut()>::new(move || {
            {
                let mut b = i.borrow_mut();
                b.connected = true;
                b.backoff.reset();
            }
            push(&i, RawEvent::Connected);
        });
        let i = inner.clone();
        let on_message = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
            if let Ok(buf) = e.data().dyn_into::<js_sys::ArrayBuffer>() {
                push(&i, RawEvent::Frame(js_sys::Uint8Array::new(&buf).to_vec()));
            }
        });
        let i = inner.clone();
        let on_close = Closure::<dyn FnMut(CloseEvent)>::new(move |e: CloseEvent| {
            i.borrow_mut().connected = false;
            push(&i, RawEvent::Disconnected(format!("연결이 끊어졌습니다 (code {})", e.code())));
            schedule_reconnect(&i);
        });
        ws.set_onopen(Some(on_open.as_ref().unchecked_ref()));
        ws.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        ws.set_onclose(Some(on_close.as_ref().unchecked_ref()));

        let mut b = inner.borrow_mut();
        if let Some(old) = b.ws.take() {
            old.set_onopen(None);
            old.set_onmessage(None);
            old.set_onclose(None);
        }
        b.ws = Some(ws);
        b.on_open = Some(on_open);
        b.on_message = Some(on_message);
        b.on_close = Some(on_close);
    }

    fn schedule_reconnect(inner: &Rc<RefCell<Inner>>) {
        let delay = inner.borrow_mut().backoff.next_delay_ms();
        let i = inner.clone();
        let cb = Closure::once_into_js(move || connect(&i));
        if let Some(window) = web_sys::window() {
            let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(cb.unchecked_ref(), delay as i32);
        }
    }

    impl Transport {
        pub fn current_url(&self) -> String {
            self.inner.borrow().url.clone()
        }

        pub fn start(urls: Vec<String>) -> Transport {
            let inner = Rc::new(RefCell::new(Inner {
                url: urls.into_iter().next().unwrap_or_default(),
                ws: None,
                queue: VecDeque::new(),
                waker: None,
                connected: false,
                stopped: false,
                backoff: Backoff::default(),
                on_open: None,
                on_message: None,
                on_close: None,
                tick: None,
            }));
            let i = Rc::downgrade(&inner);
            let tick = Closure::<dyn FnMut()>::new(move || {
                let waker = i.upgrade().and_then(|i| i.borrow().waker.clone());
                if let Some(w) = waker {
                    w.wake();
                }
            });
            if let Some(window) = web_sys::window()
                && let Ok(handle) = window
                    .set_interval_with_callback_and_timeout_and_arguments_0(tick.as_ref().unchecked_ref(), TICK_MS as i32)
            {
                inner.borrow_mut().tick = Some((handle, tick));
            }
            connect(&inner);
            Transport { inner }
        }

        pub fn set_waker(&self, waker: &Waker) {
            let mut i = self.inner.borrow_mut();
            if !i.waker.as_ref().is_some_and(|old| old.will_wake(waker)) {
                i.waker = Some(waker.clone());
            }
        }

        pub fn drain(&self) -> Vec<RawEvent> {
            self.inner.borrow_mut().queue.drain(..).collect()
        }

        pub fn send(&self, bytes: Vec<u8>) -> bool {
            let i = self.inner.borrow();
            match (&i.ws, i.connected) {
                (Some(ws), true) => ws.send_with_u8_array(&bytes).is_ok(),
                _ => false,
            }
        }

        pub fn connected(&self) -> bool {
            self.inner.borrow().connected
        }
    }

    impl Drop for Transport {
        fn drop(&mut self) {
            let mut i = self.inner.borrow_mut();
            i.stopped = true;
            if let Some((handle, _)) = i.tick.take()
                && let Some(window) = web_sys::window()
            {
                window.clear_interval_with_handle(handle);
            }
            if let Some(ws) = i.ws.take() {
                ws.set_onclose(None);
                let _ = ws.close();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skylens_shared::proto::model::v1 as model;

    #[test]
    fn normalizes_core_urls() {
        assert_eq!(normalize_core_url("core:8080", VIEWER_PATH), "ws://core:8080/viewer");
        assert_eq!(normalize_core_url("http://core:8080/", CONTROL_PATH), "ws://core:8080/control");
        assert_eq!(normalize_core_url("https://bridge", VIEWER_PATH), "wss://bridge/viewer");
        assert_eq!(normalize_core_url("ws://core:8080/custom", VIEWER_PATH), "ws://core:8080/custom");
        assert_eq!(page_core_url(true, "gw.example:443", "/viewer"), "wss://gw.example:443/viewer");
        assert_eq!(page_core_url(false, "10.0.0.2:8080", "control"), "ws://10.0.0.2:8080/control");
    }

    fn chunk_env() -> LinkEvent {
        LinkEvent::Message(viewer::Envelope {
            body: Some(Body::Chunk(model::Chunk::default())),
            ..Default::default()
        })
    }

    #[test]
    fn status_wording_follows_the_legacy_screens() {
        let mut s = LinkStatus::default();
        assert_eq!(s.detail(), "코어 연결 시도 중");
        assert_eq!(s.badge_text(), "코어 연결 시도 중");
        assert_eq!(s.severity(), Severity::Warn);

        assert!(s.observe(&LinkEvent::Disconnected("refused".into()), 0.0));
        assert_eq!(s.detail(), "코어에 연결할 수 없습니다");
        assert_eq!(s.badge_text(), "코어 응답 없음 · 재연결 대기");
        assert_eq!(s.severity().dot_css(), "#ff4d4d");

        s.observe(&LinkEvent::Connected, 0.0);
        assert_eq!(s.detail(), "코어 연결됨");
        assert_eq!(s.badge_text(), "코어 연결됨 · 데이터 대기");

        // A site message is not data; a chunk is.
        let site = LinkEvent::Message(viewer::Envelope { body: Some(Body::Site(viewer::Site::default())), ..Default::default() });
        assert!(!s.observe(&site, 1.0));
        assert!(s.observe(&chunk_env(), 2.0));
        assert_eq!(s.badge_text(), "코어 스트림 수신 중");
        assert_eq!(s.frames, 2);
        assert_eq!(s.last_frame_ms, 2.0);

        s.observe(&LinkEvent::Disconnected("closed".into()), 3.0);
        assert_eq!(s.detail(), "코어 연결 끊김");
        assert!(!s.receiving);
        s.observe(&LinkEvent::Connecting, 4.0);
        assert_eq!(s.detail(), "코어 재연결 시도 중");
        assert_eq!(s.reason.as_deref(), Some("closed"));
    }

    #[test]
    fn attempts_count_until_a_connect_succeeds() {
        let mut s = LinkStatus::default();
        assert_eq!(s.attempts, 0);
        for n in 1..=3 {
            s.observe(&LinkEvent::Connecting, 0.0);
            assert_eq!(s.attempts, n);
            s.observe(&LinkEvent::Disconnected("refused".into()), 0.0);
            assert_eq!(s.attempts, n, "a failure does not start an attempt");
        }
        s.observe(&LinkEvent::Connecting, 0.0);
        s.observe(&LinkEvent::Connected, 0.0);
        assert_eq!(s.attempts, 0);
        assert!(s.connected());
        // A later drop counts from one again.
        s.observe(&LinkEvent::Disconnected("closed".into()), 0.0);
        s.observe(&LinkEvent::Connecting, 0.0);
        assert_eq!(s.attempts, 1);
    }

    #[test]
    fn decodes_frames_into_envelopes() {
        use prost::Message as _;
        let env = viewer::Envelope { seq: 7, ..Default::default() };
        match decode(RawEvent::Frame(env.encode_to_vec())) {
            LinkEvent::Message(e) => assert_eq!(e.seq, 7),
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(decode(RawEvent::Frame(vec![0xff, 0xff, 0xff])), LinkEvent::Invalid(_)));
    }
}
