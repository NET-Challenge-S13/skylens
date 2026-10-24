//! Live video for the web build: the announced WHEP feed played by the browser's own WebRTC.
//!
//! Blitz draws the whole tower into one canvas, so decoded video cannot be a widget of it. The
//! browser decodes the feed into a real `<video>` element that sits over the canvas, exactly on
//! the camera panel's layout rect (re-placed every poll, so a resize follows), together with a
//! copy of the panel's label and readout line so the video does not cover them.
//!
//! - [`resolve_whep_url`], [`resolve_location`], [`backoff_ms`]: pure URL and timing rules.
//! - [`Supervisor`]: the connect / retry state machine, free of the browser (unit tested).
//! - `web::WebPlayer` (wasm only): `RTCPeerConnection`, the WHEP exchange, the overlay elements.

/// How long one connection attempt may take before it counts as failed (ms).
pub const CONNECT_TIMEOUT_MS: f64 = 15_000.0;
/// First and largest wait before a failed feed is tried again (ms).
pub const BACKOFF_FIRST_MS: f64 = 1_000.0;
pub const BACKOFF_MAX_MS: f64 = 10_000.0;

/// Absolute WHEP URL for the feed's announced `whep_url`: an absolute URL is kept, a rooted path
/// (`/live/drone1/whep`) goes under `base` (the page origin, or `?whepBase=`), anything else is
/// taken relative to `base`. None for an empty URL.
pub fn resolve_whep_url(base: &str, whep: &str) -> Option<String> {
    let whep = whep.trim();
    if whep.is_empty() {
        return None;
    }
    if whep.contains("://") {
        return Some(whep.to_string());
    }
    let base = base.trim_end_matches('/');
    if base.is_empty() {
        return None;
    }
    Some(match whep.strip_prefix('/') {
        Some(rest) => format!("{}/{rest}", origin_of(base)),
        None => format!("{base}/{whep}"),
    })
}

/// `scheme://host:port` of a URL (the URL itself when it has no path).
fn origin_of(url: &str) -> &str {
    match url.find("://") {
        Some(i) => {
            let rest = &url[i + 3..];
            match rest.find(['/', '?', '#']) {
                Some(j) => &url[..i + 3 + j],
                None => url,
            }
        }
        None => url,
    }
}

/// The session URL a WHEP server answered with (`Location`), made absolute against the URL the
/// offer was posted to. The Pi's nginx rewrites it to `/live/...`; MediaMTX itself answers
/// relative to the POST path.
pub fn resolve_location(request_url: &str, location: &str) -> String {
    let location = location.trim();
    if location.contains("://") {
        return location.to_string();
    }
    if let Some(rest) = location.strip_prefix('/') {
        return format!("{}/{rest}", origin_of(request_url));
    }
    let path_end = request_url.find(['?', '#']).unwrap_or(request_url.len());
    let dir = match request_url[..path_end].rfind('/') {
        Some(i) if i > request_url.find("://").map_or(0, |p| p + 2) => &request_url[..i],
        _ => &request_url[..path_end],
    };
    format!("{dir}/{location}")
}

/// Wait before retry number `attempt` (0 based): 1 s, 2 s, 4 s, 8 s, then 10 s.
pub fn backoff_ms(attempt: u32) -> f64 {
    (BACKOFF_FIRST_MS * f64::from(1u32 << attempt.min(10))).min(BACKOFF_MAX_MS)
}

/// What the connection did, reported back by the browser callbacks.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// The `<video>` is playing frames.
    Live,
    /// The attempt or the established connection is gone.
    Failed(String),
}

/// What the player must do next.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    None,
    /// Close any current connection and open one for `url`; its events carry `epoch`.
    Start { epoch: u64, url: String },
    /// Close the current connection.
    Stop,
}

/// Where the feed is, as the overlay line words it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Nothing to play.
    Idle,
    Connecting,
    Live,
    /// Failed or lost; trying again.
    NoSignal,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Phase {
    Idle,
    Connecting { since: f64 },
    Live,
    Backoff { retry_at: f64 },
}

/// Keeps one feed connected: starts it, notices failure, retries with backoff, switches target.
#[derive(Debug)]
pub struct Supervisor {
    target: Option<String>,
    phase: Phase,
    epoch: u64,
    /// Failures since the last time the feed was live.
    attempt: u32,
}

impl Default for Supervisor {
    fn default() -> Self {
        Supervisor { target: None, phase: Phase::Idle, epoch: 0, attempt: 0 }
    }
}

impl Supervisor {
    /// The feed to play (None: nothing). A changed target starts over at once.
    pub fn set_target(&mut self, target: Option<String>, now: f64) -> Action {
        if target == self.target {
            return Action::None;
        }
        self.target = target;
        self.attempt = 0;
        self.epoch += 1;
        match self.target.clone() {
            Some(url) => {
                self.phase = Phase::Connecting { since: now };
                Action::Start { epoch: self.epoch, url }
            }
            None => {
                self.phase = Phase::Idle;
                Action::Stop
            }
        }
    }

    /// An event from the connection started with `epoch`; events of older connections are dropped.
    pub fn on_event(&mut self, epoch: u64, event: &Event, now: f64) -> Action {
        if epoch != self.epoch || self.target.is_none() {
            return Action::None;
        }
        match event {
            Event::Live => {
                if matches!(self.phase, Phase::Connecting { .. }) {
                    self.phase = Phase::Live;
                    self.attempt = 0;
                }
                Action::None
            }
            Event::Failed(_) if matches!(self.phase, Phase::Connecting { .. } | Phase::Live) => self.fail(now),
            Event::Failed(_) => Action::None,
        }
    }

    /// Time passing: a stuck attempt times out, a backoff ends.
    pub fn poll(&mut self, now: f64) -> Action {
        match self.phase {
            Phase::Connecting { since } if now - since > CONNECT_TIMEOUT_MS => self.fail(now),
            Phase::Backoff { retry_at } if now >= retry_at => {
                self.epoch += 1;
                self.phase = Phase::Connecting { since: now };
                match self.target.clone() {
                    Some(url) => Action::Start { epoch: self.epoch, url },
                    None => Action::None,
                }
            }
            _ => Action::None,
        }
    }

    fn fail(&mut self, now: f64) -> Action {
        let wait = backoff_ms(self.attempt);
        self.attempt += 1;
        self.phase = Phase::Backoff { retry_at: now + wait };
        Action::Stop
    }

    pub fn status(&self) -> Status {
        match self.phase {
            Phase::Idle => Status::Idle,
            Phase::Live => Status::Live,
            Phase::Connecting { .. } if self.attempt == 0 => Status::Connecting,
            Phase::Connecting { .. } | Phase::Backoff { .. } => Status::NoSignal,
        }
    }

    pub fn target(&self) -> Option<&str> {
        self.target.as_deref()
    }
}

/// Where the camera panel sits in the page, in CSS pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Rect {
    /// Shrunk by `d` on every side (the panel's 1 px border stays visible).
    pub fn inset(self, d: f64) -> Rect {
        Rect { x: self.x + d, y: self.y + d, w: (self.w - 2.0 * d).max(0.0), h: (self.h - 2.0 * d).max(0.0) }
    }

    /// Whether anything of it is on screen.
    pub fn shown(&self) -> bool {
        self.w >= 1.0 && self.h >= 1.0
    }
}

#[cfg(target_arch = "wasm32")]
pub use web::WebPlayer;

#[cfg(target_arch = "wasm32")]
mod web {
    use super::{Action, Event, Rect, Status, Supervisor};
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    use wasm_bindgen::JsCast as _;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{
        HtmlElement, HtmlVideoElement, MediaStream, RtcConfiguration, RtcIceGatheringState, RtcPeerConnection, RtcPeerConnectionState, RtcRtpTransceiverDirection,
        RtcRtpTransceiverInit, RtcSdpType, RtcSessionDescriptionInit, RtcTrackEvent,
    };

    /// Longest wait for ICE gathering before the offer goes out with what was gathered (ms).
    const GATHER_MAX_MS: f64 = 3_000.0;
    /// Timeout of the WHEP POST (ms).
    const POST_TIMEOUT_MS: u32 = 10_000;

    type Queue = Rc<RefCell<Vec<(u64, Event)>>>;

    fn js_err(e: JsValue) -> String {
        if let Some(err) = e.dyn_ref::<js_sys::Error>() {
            String::from(err.to_string())
        } else {
            e.as_string().unwrap_or_else(|| format!("{e:?}"))
        }
    }

    async fn sleep(ms: i32) {
        let promise = js_sys::Promise::new(&mut |resolve, _| {
            if let Some(w) = web_sys::window() {
                let _ = w.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms);
            }
        });
        let _ = JsFuture::from(promise).await;
    }

    /// One WebRTC connection to a WHEP endpoint.
    struct Conn {
        pc: RtcPeerConnection,
        video: HtmlVideoElement,
        /// The session URL to DELETE on teardown, once the server has answered.
        session: RefCell<Option<String>>,
        closed: Cell<bool>,
        /// Callbacks the peer connection and the video hold.
        keep: RefCell<Vec<Box<dyn std::any::Any>>>,
    }

    impl Conn {
        fn close(&self) {
            if self.closed.replace(true) {
                return;
            }
            self.pc.set_ontrack(None);
            self.pc.set_onconnectionstatechange(None);
            self.pc.close();
            self.video.set_onplaying(None);
            self.video.set_src_object(None);
            if let Some(url) = self.session.borrow_mut().take() {
                delete_session(&url);
            }
        }
    }

    /// WHEP teardown: DELETE on the session URL, fire and forget.
    fn delete_session(url: &str) {
        let Some(window) = web_sys::window() else { return };
        let init = web_sys::RequestInit::new();
        init.set_method("DELETE");
        let _ = js_sys::Reflect::set(&init, &"keepalive".into(), &JsValue::TRUE);
        if let Ok(req) = web_sys::Request::new_with_str_and_init(url, &init) {
            let _ = window.fetch_with_request(&req);
        }
    }

    pub struct WebPlayer {
        sup: Supervisor,
        events: Queue,
        conn: Option<Rc<Conn>>,
        ice_servers: Vec<String>,
        // Overlay elements.
        wrap: HtmlElement,
        video: HtmlVideoElement,
        label: HtmlElement,
        readout: HtmlElement,
        placed: Option<(String, bool)>,
        texts: (String, String),
    }

    fn element(doc: &web_sys::Document, tag: &str, css: &str) -> Result<HtmlElement, JsValue> {
        let el = doc.create_element(tag)?.dyn_into::<HtmlElement>()?;
        el.set_attribute("style", css)?;
        Ok(el)
    }

    impl WebPlayer {
        /// Builds the overlay (hidden) in the page; `ice_servers` are STUN/TURN URLs (none needed
        /// on a LAN).
        pub fn new(ice_servers: Vec<String>) -> Option<WebPlayer> {
            Self::build(ice_servers).map_err(|e| skylens_viewer::log(&format!("[video] no overlay: {}", js_err(e)))).ok()
        }

        fn build(ice_servers: Vec<String>) -> Result<WebPlayer, JsValue> {
            let doc = web_sys::window().and_then(|w| w.document()).ok_or("no document")?;
            let body = doc.body().ok_or("no body")?;
            let wrap = element(
                &doc,
                "div",
                "position:fixed;left:0;top:0;width:0;height:0;display:none;pointer-events:none;overflow:hidden;z-index:5;background:#000;border-radius:5px;",
            )?;
            wrap.set_id("skylens-cam");
            let video = doc.create_element("video")?.dyn_into::<HtmlVideoElement>()?;
            video.set_attribute("style", "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;background:#000;")?;
            video.set_muted(true);
            video.set_autoplay(true);
            video.set_attribute("muted", "")?;
            video.set_attribute("playsinline", "")?;
            let frame = element(&doc, "div", "position:absolute;inset:5px;border:1px solid rgba(159,232,255,.14);border-radius:3px;pointer-events:none;")?;
            let font = "'Noto Sans KR','Malgun Gothic',sans-serif";
            let label = element(
                &doc,
                "div",
                &format!(
                    "position:absolute;top:7px;left:9px;display:flex;align-items:center;gap:6px;font:600 10.5px {font};letter-spacing:.12em;color:#5ec8ff;text-shadow:0 1px 3px rgba(0,0,0,.85);"
                ),
            )?;
            let readout = element(
                &doc,
                "div",
                &format!(
                    "position:absolute;bottom:6px;left:9px;right:9px;font:10.5px ui-monospace,'Cascadia Code',{font};letter-spacing:.02em;color:#c8e8ff;text-shadow:0 1px 3px rgba(0,0,0,.85);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                ),
            )?;
            for child in [video.unchecked_ref::<web_sys::Node>(), frame.as_ref(), label.as_ref(), readout.as_ref()] {
                wrap.append_child(child)?;
            }
            body.append_child(&wrap)?;
            Ok(WebPlayer {
                sup: Supervisor::default(),
                events: Rc::default(),
                conn: None,
                ice_servers,
                wrap,
                video,
                label,
                readout,
                placed: None,
                texts: (String::new(), String::new()),
            })
        }

        pub fn status(&self) -> Status {
            self.sup.status()
        }

        /// Once per poll: connection events and timers, the effective target (None while the
        /// panel is not on screen or the page is hidden), and the overlay's place and texts.
        /// `rect` is the camera panel's rect; `covered` hides the overlay behind a dialog.
        pub fn tick(&mut self, now: f64, target: Option<String>, rect: Option<Rect>, covered: bool) {
            let queued: Vec<(u64, Event)> = self.events.borrow_mut().drain(..).collect();
            for (epoch, ev) in queued {
                if let Event::Failed(why) = &ev {
                    skylens_viewer::log(&format!("[video] feed lost: {why}"));
                }
                let a = self.sup.on_event(epoch, &ev, now);
                self.apply(a);
            }
            let a = self.sup.set_target(target, now);
            self.apply(a);
            let a = self.sup.poll(now);
            self.apply(a);
            let show = self.sup.status() == Status::Live && !covered && rect.is_some_and(|r| r.shown());
            self.place(rect, show);
        }

        /// The panel's title and readout line, copied onto the overlay.
        pub fn set_texts(&mut self, label: &str, readout: &str) {
            if self.texts.0 != label {
                self.label.set_inner_html("");
                let dot = element(&web_sys::window().and_then(|w| w.document()).expect("document"), "span", "width:6px;height:6px;border-radius:50%;background:#ff5252;box-shadow:0 0 6px #ff5252;");
                if let Ok(dot) = dot {
                    let _ = self.label.append_child(&dot);
                }
                let _ = self.label.append_child(&web_sys::window().and_then(|w| w.document()).expect("document").create_text_node(label));
                self.texts.0 = label.to_string();
            }
            if self.texts.1 != readout {
                self.readout.set_text_content(Some(readout));
                self.texts.1 = readout.to_string();
            }
        }

        fn place(&mut self, rect: Option<Rect>, show: bool) {
            let r = rect.unwrap_or(Rect { x: 0.0, y: 0.0, w: 0.0, h: 0.0 }).inset(1.0);
            let css = if show {
                format!("{:.2}px;{:.2}px;{:.2}px;{:.2}px", r.x, r.y, r.w, r.h)
            } else {
                String::new()
            };
            if self.placed.as_ref().is_some_and(|(c, s)| *s == show && *c == css) {
                return;
            }
            let style = self.wrap.style();
            if show {
                let _ = style.set_property("left", &format!("{:.2}px", r.x));
                let _ = style.set_property("top", &format!("{:.2}px", r.y));
                let _ = style.set_property("width", &format!("{:.2}px", r.w));
                let _ = style.set_property("height", &format!("{:.2}px", r.h));
                let _ = style.set_property("display", "block");
            } else {
                let _ = style.set_property("display", "none");
            }
            self.placed = Some((css, show));
        }

        fn apply(&mut self, action: Action) {
            match action {
                Action::None => {}
                Action::Stop => self.stop(),
                Action::Start { epoch, url } => {
                    self.stop();
                    self.start(epoch, url);
                }
            }
        }

        fn stop(&mut self) {
            if let Some(c) = self.conn.take() {
                c.close();
            }
        }

        fn start(&mut self, epoch: u64, url: String) {
            let config = RtcConfiguration::new();
            if !self.ice_servers.is_empty() {
                let list = js_sys::Array::new();
                for s in &self.ice_servers {
                    let server = js_sys::Object::new();
                    let _ = js_sys::Reflect::set(&server, &"urls".into(), &JsValue::from_str(s));
                    list.push(&server);
                }
                config.set_ice_servers(&list);
            }
            let pc = match RtcPeerConnection::new_with_configuration(&config) {
                Ok(pc) => pc,
                Err(e) => {
                    self.events.borrow_mut().push((epoch, Event::Failed(js_err(e))));
                    return;
                }
            };
            let conn = Rc::new(Conn { pc, video: self.video.clone(), session: RefCell::new(None), closed: Cell::new(false), keep: RefCell::new(Vec::new()) });
            self.conn = Some(conn.clone());
            let events = self.events.clone();

            // Remote video track -> the <video>.
            let video = self.video.clone();
            let on_track = Closure::<dyn FnMut(RtcTrackEvent)>::new(move |ev: RtcTrackEvent| {
                let stream = ev.streams().get(0).dyn_into::<MediaStream>().ok().or_else(|| {
                    let s = MediaStream::new().ok()?;
                    s.add_track(&ev.track());
                    Some(s)
                });
                video.set_src_object(stream.as_ref());
                let _ = video.play();
            });
            conn.pc.set_ontrack(Some(on_track.as_ref().unchecked_ref()));
            // Frames on screen -> live.
            let q = events.clone();
            let on_playing = Closure::<dyn FnMut()>::new(move || q.borrow_mut().push((epoch, Event::Live)));
            self.video.set_onplaying(Some(on_playing.as_ref().unchecked_ref()));
            // The connection dropped.
            let q = events.clone();
            let pc = conn.pc.clone();
            let on_state = Closure::<dyn FnMut()>::new(move || {
                let state = pc.connection_state();
                if matches!(state, RtcPeerConnectionState::Failed | RtcPeerConnectionState::Closed | RtcPeerConnectionState::Disconnected) {
                    q.borrow_mut().push((epoch, Event::Failed(format!("connection {state:?}"))));
                }
            });
            conn.pc.set_onconnectionstatechange(Some(on_state.as_ref().unchecked_ref()));
            conn.keep.borrow_mut().extend([Box::new(on_track) as Box<dyn std::any::Any>, Box::new(on_playing), Box::new(on_state)]);

            wasm_bindgen_futures::spawn_local(async move {
                if let Err(e) = exchange(&conn, &url).await {
                    if !conn.closed.get() {
                        events.borrow_mut().push((epoch, Event::Failed(e)));
                    }
                }
            });
        }
    }

    /// The WHEP exchange: recvonly offer, wait for the candidates, POST, apply the answer.
    async fn exchange(conn: &Rc<Conn>, url: &str) -> Result<(), String> {
        let pc = &conn.pc;
        let init = RtcRtpTransceiverInit::new();
        init.set_direction(RtcRtpTransceiverDirection::Recvonly);
        pc.add_transceiver_with_str_and_init("video", &init);
        let offer = JsFuture::from(pc.create_offer()).await.map_err(js_err)?;
        let offer: RtcSessionDescriptionInit = offer.unchecked_into();
        JsFuture::from(pc.set_local_description(&offer)).await.map_err(js_err)?;
        // No trickle: the offer goes out with every candidate it will have.
        let start = js_sys::Date::now();
        while pc.ice_gathering_state() != RtcIceGatheringState::Complete && js_sys::Date::now() - start < GATHER_MAX_MS {
            if conn.closed.get() {
                return Ok(());
            }
            sleep(50).await;
        }
        let sdp = pc.local_description().map(|d| d.sdp()).ok_or("no local description")?;

        let window = web_sys::window().ok_or("no window")?;
        let headers = web_sys::Headers::new().map_err(js_err)?;
        headers.set("Content-Type", "application/sdp").map_err(js_err)?;
        let req_init = web_sys::RequestInit::new();
        req_init.set_method("POST");
        req_init.set_headers(&headers);
        req_init.set_body(&JsValue::from_str(&sdp));
        req_init.set_signal(Some(&web_sys::AbortSignal::timeout_with_u32(POST_TIMEOUT_MS)));
        let request = web_sys::Request::new_with_str_and_init(url, &req_init).map_err(js_err)?;
        let resp = JsFuture::from(window.fetch_with_request(&request)).await.map_err(js_err)?;
        let resp: web_sys::Response = resp.dyn_into().map_err(js_err)?;
        let status = resp.status();
        if !(status == 200 || status == 201) {
            return Err(format!("WHEP {status}"));
        }
        if let Some(loc) = resp.headers().get("location").ok().flatten() {
            let session = super::resolve_location(&absolute(url), &loc);
            if conn.closed.get() {
                delete_session(&session);
                return Ok(());
            }
            *conn.session.borrow_mut() = Some(session);
        }
        let answer = JsFuture::from(resp.text().map_err(js_err)?).await.map_err(js_err)?.as_string().ok_or("no answer")?;
        if conn.closed.get() {
            return Ok(());
        }
        let desc = RtcSessionDescriptionInit::new(RtcSdpType::Answer);
        desc.set_sdp(&answer);
        JsFuture::from(pc.set_remote_description(&desc)).await.map_err(js_err)?;
        Ok(())
    }

    /// `url` as the browser resolves it (so a relative one gets the page's origin).
    fn absolute(url: &str) -> String {
        web_sys::Url::new(url).map(|u| u.href()).unwrap_or_else(|_| url.to_string())
    }

    /// The page's origin, the base of relative WHEP URLs.
    pub fn page_origin() -> String {
        web_sys::window().and_then(|w| w.location().origin().ok()).unwrap_or_default()
    }
}

#[cfg(target_arch = "wasm32")]
pub use web::page_origin;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whep_urls_resolve_against_the_base() {
        let b = "https://10.9.8.37:8091";
        assert_eq!(resolve_whep_url(b, "/live/drone1/whep").as_deref(), Some("https://10.9.8.37:8091/live/drone1/whep"));
        assert_eq!(resolve_whep_url("https://h:8091/tower/", "/live/drone1/whep").as_deref(), Some("https://h:8091/live/drone1/whep"));
        assert_eq!(resolve_whep_url(b, "http://mtx:8889/drone2/whep").as_deref(), Some("http://mtx:8889/drone2/whep"));
        assert_eq!(resolve_whep_url(b, "live/drone3/whep").as_deref(), Some("https://10.9.8.37:8091/live/drone3/whep"));
        assert_eq!(resolve_whep_url(b, "  "), None);
        assert_eq!(resolve_whep_url("", "/live/x/whep"), None);
    }

    #[test]
    fn session_locations_resolve_against_the_post() {
        let post = "https://10.9.8.37:8091/live/drone1/whep";
        assert_eq!(resolve_location(post, "/live/drone1/whep/abc"), "https://10.9.8.37:8091/live/drone1/whep/abc");
        assert_eq!(resolve_location(post, "http://x/y"), "http://x/y");
        assert_eq!(resolve_location(post, "abc"), "https://10.9.8.37:8091/live/drone1/abc");
        assert_eq!(resolve_location("https://h:1", "abc"), "https://h:1/abc");
        assert_eq!(resolve_location("https://h:1/w?x=1/2", "abc"), "https://h:1/abc");
    }

    #[test]
    fn backoff_doubles_to_a_cap() {
        let w: Vec<f64> = (0..6).map(backoff_ms).collect();
        assert_eq!(w, vec![1000.0, 2000.0, 4000.0, 8000.0, 10_000.0, 10_000.0]);
        assert_eq!(backoff_ms(u32::MAX), BACKOFF_MAX_MS);
    }

    fn url(s: &str) -> Option<String> {
        Some(s.to_string())
    }

    #[test]
    fn target_starts_switches_and_stops() {
        let mut s = Supervisor::default();
        assert_eq!(s.status(), Status::Idle);
        assert_eq!(s.set_target(None, 0.0), Action::None);
        assert_eq!(s.set_target(url("a"), 0.0), Action::Start { epoch: 1, url: "a".into() });
        assert_eq!(s.status(), Status::Connecting);
        assert_eq!(s.set_target(url("a"), 1.0), Action::None);
        assert_eq!(s.on_event(1, &Event::Live, 50.0), Action::None);
        assert_eq!(s.status(), Status::Live);
        // Switching drones starts the new feed and drops the old events.
        assert_eq!(s.set_target(url("b"), 100.0), Action::Start { epoch: 2, url: "b".into() });
        assert_eq!(s.status(), Status::Connecting);
        assert_eq!(s.on_event(1, &Event::Failed("late".into()), 101.0), Action::None);
        assert_eq!(s.status(), Status::Connecting);
        assert_eq!(s.set_target(None, 200.0), Action::Stop);
        assert_eq!(s.status(), Status::Idle);
        assert_eq!(s.on_event(2, &Event::Live, 201.0), Action::None);
        assert_eq!(s.status(), Status::Idle);
    }

    #[test]
    fn failures_retry_with_growing_backoff() {
        let mut s = Supervisor::default();
        s.set_target(url("a"), 0.0);
        assert_eq!(s.on_event(1, &Event::Failed("x".into()), 10.0), Action::Stop);
        assert_eq!(s.status(), Status::NoSignal);
        assert_eq!(s.poll(500.0), Action::None);
        assert_eq!(s.poll(1010.0), Action::Start { epoch: 2, url: "a".into() });
        assert_eq!(s.status(), Status::NoSignal);
        // Second failure waits twice as long.
        assert_eq!(s.on_event(2, &Event::Failed("x".into()), 1100.0), Action::Stop);
        assert_eq!(s.poll(3000.0), Action::None);
        assert_eq!(s.poll(3100.0), Action::Start { epoch: 3, url: "a".into() });
        // Live resets the count: a later drop retries after the first wait again.
        assert_eq!(s.on_event(3, &Event::Live, 3200.0), Action::None);
        assert_eq!(s.status(), Status::Live);
        assert_eq!(s.on_event(3, &Event::Failed("gone".into()), 9000.0), Action::Stop);
        assert_eq!(s.poll(9999.0), Action::None);
        assert!(matches!(s.poll(10_000.0), Action::Start { epoch: 4, .. }));
    }

    #[test]
    fn a_stuck_attempt_times_out() {
        let mut s = Supervisor::default();
        s.set_target(url("a"), 0.0);
        assert_eq!(s.poll(CONNECT_TIMEOUT_MS), Action::None);
        assert_eq!(s.poll(CONNECT_TIMEOUT_MS + 1.0), Action::Stop);
        assert_eq!(s.status(), Status::NoSignal);
        // Retargeting during backoff starts at once with a fresh count.
        assert_eq!(s.set_target(url("b"), 16_000.0), Action::Start { epoch: 2, url: "b".into() });
        assert_eq!(s.status(), Status::Connecting);
    }

    #[test]
    fn rects_inset_and_show() {
        let r = Rect { x: 10.0, y: 20.0, w: 320.0, h: 200.0 }.inset(1.0);
        assert_eq!(r, Rect { x: 11.0, y: 21.0, w: 318.0, h: 198.0 });
        assert!(r.shown());
        assert!(!Rect { x: 0.0, y: 0.0, w: 0.0, h: 50.0 }.shown());
        assert_eq!(Rect { x: 0.0, y: 0.0, w: 1.0, h: 1.0 }.inset(1.0).w, 0.0);
    }
}
