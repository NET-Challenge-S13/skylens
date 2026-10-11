//! The one WebSocket to the core (`ws /viewer`), with reconnect and backoff.
//!
//! Native runs tokio-tungstenite on a background thread; web uses the browser
//! WebSocket through wasm-bindgen. Both hand frames to the document through a
//! queue and wake the Blitz window with the waker from `Document::poll`. A ticker
//! wakes it every [`TICK_MS`] too, for countdowns, stale telemetry and the 10 Hz
//! manual-control stream.

use std::task::Waker;

pub const TICK_MS: u64 = 100;

#[derive(Debug, Clone)]
pub enum LinkEvent {
    Connecting,
    Connected,
    Disconnected(String),
    Frame(Vec<u8>),
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::Link;
#[cfg(target_arch = "wasm32")]
pub use web::Link;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::{LinkEvent, TICK_MS, Waker};
    use crate::protocol::Backoff;
    use futures_util::{SinkExt as _, StreamExt as _};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tokio_tungstenite::tungstenite::Message as WsMessage;

    #[derive(Default)]
    struct Shared {
        waker: Mutex<Option<Waker>>,
        connected: AtomicBool,
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

    pub struct Link {
        events: mpsc::Receiver<LinkEvent>,
        outgoing: tokio::sync::mpsc::UnboundedSender<Vec<u8>>,
        shared: Arc<Shared>,
    }

    impl Link {
        pub fn start(url: String) -> Link {
            let (ev_tx, ev_rx) = mpsc::channel();
            let (out_tx, out_rx) = tokio::sync::mpsc::unbounded_channel();
            let shared = Arc::new(Shared::default());

            let net_shared = shared.clone();
            std::thread::Builder::new()
                .name("skylens-link".into())
                .spawn(move || {
                    let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                        Ok(rt) => rt,
                        Err(e) => {
                            let _ = ev_tx.send(LinkEvent::Disconnected(format!("tokio: {e}")));
                            return;
                        }
                    };
                    rt.block_on(run(url, ev_tx, out_rx, net_shared));
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

            Link {
                events: ev_rx,
                outgoing: out_tx,
                shared,
            }
        }

        pub fn set_waker(&self, waker: &Waker) {
            if let Ok(mut w) = self.shared.waker.lock()
                && !w.as_ref().is_some_and(|old| old.will_wake(waker))
            {
                *w = Some(waker.clone());
            }
        }

        pub fn drain(&self) -> Vec<LinkEvent> {
            self.events.try_iter().collect()
        }

        /// Queues a frame; false when the core is not connected (nothing is buffered
        /// across reconnects: stale control must not fly a drone later).
        pub fn send(&self, bytes: Vec<u8>) -> bool {
            self.shared.connected.load(Ordering::SeqCst) && self.outgoing.send(bytes).is_ok()
        }

        pub fn connected(&self) -> bool {
            self.shared.connected.load(Ordering::SeqCst)
        }
    }

    async fn run(
        url: String,
        events: mpsc::Sender<LinkEvent>,
        mut outgoing: tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>,
        shared: Arc<Shared>,
    ) {
        let emit = |ev: LinkEvent| -> bool {
            let ok = events.send(ev).is_ok();
            shared.wake();
            ok
        };
        let mut backoff = Backoff::default();
        loop {
            if !emit(LinkEvent::Connecting) {
                return;
            }
            match tokio_tungstenite::connect_async(url.as_str()).await {
                Ok((ws, _)) => {
                    backoff.reset();
                    // Drop anything queued while disconnected.
                    while outgoing.try_recv().is_ok() {}
                    shared.connected.store(true, Ordering::SeqCst);
                    if !emit(LinkEvent::Connected) {
                        return;
                    }
                    let (mut sink, mut stream) = ws.split();
                    let reason = loop {
                        tokio::select! {
                            msg = stream.next() => match msg {
                                Some(Ok(WsMessage::Binary(b))) => {
                                    if !emit(LinkEvent::Frame(b.to_vec())) {
                                        return;
                                    }
                                }
                                Some(Ok(WsMessage::Close(frame))) => {
                                    break frame.map_or_else(|| "코어가 연결을 닫았습니다".to_string(), |f| format!("코어가 연결을 닫았습니다 ({})", u16::from(f.code)));
                                }
                                Some(Ok(_)) => {}
                                Some(Err(e)) => break e.to_string(),
                                None => break "연결이 끊어졌습니다".to_string(),
                            },
                            out = outgoing.recv() => match out {
                                Some(bytes) => {
                                    if let Err(e) = sink.send(WsMessage::binary(bytes)).await {
                                        break e.to_string();
                                    }
                                }
                                None => return,
                            },
                        }
                    };
                    shared.connected.store(false, Ordering::SeqCst);
                    if !emit(LinkEvent::Disconnected(reason)) {
                        return;
                    }
                }
                Err(e) => {
                    if !emit(LinkEvent::Disconnected(e.to_string())) {
                        return;
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(backoff.next_delay_ms())).await;
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use super::{LinkEvent, TICK_MS, Waker};
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
        queue: VecDeque<LinkEvent>,
        waker: Option<Waker>,
        connected: bool,
        backoff: Backoff,
        on_open: Option<Closure<dyn FnMut()>>,
        on_message: Option<Closure<dyn FnMut(MessageEvent)>>,
        on_close: Option<Closure<dyn FnMut(CloseEvent)>>,
        _tick: Option<Closure<dyn FnMut()>>,
    }

    pub struct Link {
        inner: Rc<RefCell<Inner>>,
    }

    fn push(inner: &Rc<RefCell<Inner>>, ev: LinkEvent) {
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
        push(inner, LinkEvent::Connecting);
        let url = inner.borrow().url.clone();
        let ws = match WebSocket::new(&url) {
            Ok(ws) => ws,
            Err(e) => {
                push(inner, LinkEvent::Disconnected(format!("{e:?}")));
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
            push(&i, LinkEvent::Connected);
        });
        let i = inner.clone();
        let on_message = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
            if let Ok(buf) = e.data().dyn_into::<js_sys::ArrayBuffer>() {
                push(&i, LinkEvent::Frame(js_sys::Uint8Array::new(&buf).to_vec()));
            }
        });
        let i = inner.clone();
        let on_close = Closure::<dyn FnMut(CloseEvent)>::new(move |e: CloseEvent| {
            i.borrow_mut().connected = false;
            push(&i, LinkEvent::Disconnected(format!("연결이 끊어졌습니다 (code {})", e.code())));
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
            let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
                cb.unchecked_ref(),
                delay as i32,
            );
        }
    }

    impl Link {
        pub fn start(url: String) -> Link {
            let inner = Rc::new(RefCell::new(Inner {
                url,
                ws: None,
                queue: VecDeque::new(),
                waker: None,
                connected: false,
                backoff: Backoff::default(),
                on_open: None,
                on_message: None,
                on_close: None,
                _tick: None,
            }));
            let i = inner.clone();
            let tick = Closure::<dyn FnMut()>::new(move || {
                let waker = i.borrow().waker.clone();
                if let Some(w) = waker {
                    w.wake();
                }
            });
            if let Some(window) = web_sys::window() {
                let _ = window.set_interval_with_callback_and_timeout_and_arguments_0(
                    tick.as_ref().unchecked_ref(),
                    TICK_MS as i32,
                );
            }
            inner.borrow_mut()._tick = Some(tick);
            connect(&inner);
            Link { inner }
        }

        pub fn set_waker(&self, waker: &Waker) {
            let mut i = self.inner.borrow_mut();
            if !i.waker.as_ref().is_some_and(|old| old.will_wake(waker)) {
                i.waker = Some(waker.clone());
            }
        }

        pub fn drain(&self) -> Vec<LinkEvent> {
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
}
