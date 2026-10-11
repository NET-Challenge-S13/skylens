//! Wire helpers for `ws /viewer`.
//!
//! Every WebSocket frame is one binary `skylens.viewer.v1.Envelope` (viewer.proto).
//! Screens receive viewer events and the control tower sends `assign_route` and
//! `manual_control`. Chunk payloads are decoded here too, because the core never
//! does: mesh blocks and detections are `skylens.payload.v1` messages, points,
//! terrain and buildings are pointblitz chunk v2 bytes.

use prost::Message as _;
use skylens_shared::proto::model::v1 as model;
use skylens_shared::proto::payload::v1 as payload;
use skylens_shared::proto::viewer::v1 as viewer;

pub use viewer::envelope::Body;

/// Decodes one WebSocket frame.
pub fn decode_envelope(bytes: &[u8]) -> Result<viewer::Envelope, prost::DecodeError> {
    viewer::Envelope::decode(bytes)
}

/// Decodes the payload of a `CHUNK_KIND_DETECTIONS` chunk.
pub fn decode_detections(bytes: &[u8]) -> Result<payload::DetectionBatch, prost::DecodeError> {
    payload::DetectionBatch::decode(bytes)
}

/// Decodes the payload of a `CHUNK_KIND_MESH_BLOCK` chunk.
pub fn decode_mesh_block(bytes: &[u8]) -> Result<payload::MeshBlock, prost::DecodeError> {
    payload::MeshBlock::decode(bytes)
}

/// Builds outgoing envelopes with a monotonic sequence number, so the core can
/// spot gaps in what this screen sent.
#[derive(Debug, Default)]
pub struct Outbox {
    seq: u64,
}

impl Outbox {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn envelope(&mut self, body: Body, now_ms: i64) -> viewer::Envelope {
        self.seq += 1;
        viewer::Envelope {
            seq: self.seq,
            origin_ts_ms: now_ms,
            from: viewer::ComponentId::Client as i32,
            body: Some(body),
        }
    }

    pub fn encode(&mut self, body: Body, now_ms: i64) -> Vec<u8> {
        self.envelope(body, now_ms).encode_to_vec()
    }
}

/// `assign_route` for the drone that carries the route (the formation centre).
pub fn assign_route(drone_id: u32, waypoints: &[model::Gps], looped: bool) -> Body {
    Body::AssignRoute(viewer::AssignRoute {
        drone_id,
        waypoints: waypoints.to_vec(),
        r#loop: looped,
    })
}

/// `manual_control` with every axis clamped to -1..1.
pub fn manual_control(drone_id: u32, forward: f64, yaw: f64, climb: f64) -> Body {
    let axis = |v: f64| if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 };
    Body::ManualControl(viewer::ManualControl {
        drone_id,
        forward: axis(forward),
        yaw: axis(yaw),
        climb: axis(climb),
    })
}

/// Reconnect delays: 500 ms, doubling, capped at 8 s; reset after a successful connect.
#[derive(Debug, Clone)]
pub struct Backoff {
    attempt: u32,
    base_ms: u64,
    max_ms: u64,
}

impl Default for Backoff {
    fn default() -> Self {
        Self {
            attempt: 0,
            base_ms: 500,
            max_ms: 8_000,
        }
    }
}

impl Backoff {
    pub fn next_delay_ms(&mut self) -> u64 {
        let shift = self.attempt.min(16);
        self.attempt = self.attempt.saturating_add(1);
        self.base_ms.saturating_mul(1u64 << shift).min(self.max_ms)
    }

    pub fn reset(&mut self) {
        self.attempt = 0;
    }
}

/// Counts gaps in the sequence numbers the core sends.
#[derive(Debug, Default, Clone)]
pub struct SeqTracker {
    last: Option<u64>,
    pub gaps: u64,
}

impl SeqTracker {
    pub fn observe(&mut self, seq: u64) {
        if let Some(last) = self.last
            && seq > last + 1
        {
            self.gaps += seq - last - 1;
        }
        if self.last.is_none_or(|last| seq > last) {
            self.last = Some(seq);
        }
    }

    /// A new connection starts a new sequence.
    pub fn reset(&mut self) {
        self.last = None;
    }

    pub fn last(&self) -> Option<u64> {
        self.last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gps(lat: f64, lon: f64, alt: f64) -> model::Gps {
        model::Gps { lat, lon, alt }
    }

    #[test]
    fn assign_route_round_trips_through_an_envelope() {
        let mut out = Outbox::new();
        let wps = [gps(36.37, 127.34, 60.0), gps(36.38, 127.35, 65.0)];
        let bytes = out.encode(assign_route(2, &wps, true), 1_700_000_000_000);
        let env = decode_envelope(&bytes).unwrap();
        assert_eq!(env.seq, 1);
        assert_eq!(env.origin_ts_ms, 1_700_000_000_000);
        assert_eq!(env.from(), viewer::ComponentId::Client);
        match env.body {
            Some(Body::AssignRoute(r)) => {
                assert_eq!(r.drone_id, 2);
                assert!(r.r#loop);
                assert_eq!(r.waypoints, wps.to_vec());
            }
            other => panic!("unexpected body {other:?}"),
        }
    }

    #[test]
    fn outbox_sequence_is_monotonic() {
        let mut out = Outbox::new();
        let a = out.envelope(manual_control(1, 0.0, 0.0, 0.0), 0);
        let b = out.envelope(manual_control(1, 0.0, 0.0, 0.0), 0);
        assert_eq!((a.seq, b.seq), (1, 2));
    }

    #[test]
    fn manual_control_clamps_axes() {
        match manual_control(3, 2.0, -5.0, f64::NAN) {
            Body::ManualControl(m) => {
                assert_eq!(m.drone_id, 3);
                assert_eq!((m.forward, m.yaw, m.climb), (1.0, -1.0, 0.0));
            }
            other => panic!("unexpected body {other:?}"),
        }
    }

    #[test]
    fn backoff_doubles_up_to_the_cap_and_resets() {
        let mut b = Backoff::default();
        let delays: Vec<u64> = (0..7).map(|_| b.next_delay_ms()).collect();
        assert_eq!(delays, vec![500, 1000, 2000, 4000, 8000, 8000, 8000]);
        b.reset();
        assert_eq!(b.next_delay_ms(), 500);
    }

    #[test]
    fn seq_tracker_counts_missing_frames() {
        let mut t = SeqTracker::default();
        for s in [1, 2, 5, 6, 6, 9] {
            t.observe(s);
        }
        assert_eq!(t.gaps, 4); // 3, 4, 7, 8
        assert_eq!(t.last(), Some(9));
        t.reset();
        t.observe(1);
        assert_eq!(t.gaps, 4);
    }

    #[test]
    fn decodes_detection_batches() {
        let batch = payload::DetectionBatch {
            detections: vec![payload::Detection {
                id: "p-1".into(),
                category: payload::DetectionCategory::Person as i32,
                gps: Some(gps(36.37, 127.35, 31.0)),
                confidence: 0.82,
                label: "생존자".into(),
                segment: 3,
                outline: vec![],
                frames: 4,
            }],
        };
        let back = decode_detections(&batch.encode_to_vec()).unwrap();
        assert_eq!(back, batch);
        assert_eq!(back.detections[0].category(), payload::DetectionCategory::Person);
    }

    #[test]
    fn rejects_garbage_frames() {
        assert!(decode_envelope(&[0xff, 0xff, 0xff]).is_err());
    }
}
