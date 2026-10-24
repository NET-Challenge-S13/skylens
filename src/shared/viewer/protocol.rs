//! Wire helpers for `ws /control` and `ws /viewer`.
//!
//! Every WebSocket frame is one binary `skylens.viewer.v1.Envelope`. Screens receive viewer
//! events; the control tower also sends `assign_route` and `manual_control`. Chunk payloads are
//! decoded here because the core never does: mesh blocks, detections, terrain tiles and building
//! sets are `skylens.payload.v1` messages, points are pointblitz chunk v2 bytes.

use prost::Message as _;
use skylens_shared::proto::model::v1 as model;
use skylens_shared::proto::payload::v1 as payload;
use skylens_shared::proto::viewer::v1 as viewer;

pub use viewer::envelope::Body;

/// Decodes one WebSocket frame.
pub fn decode_envelope(bytes: &[u8]) -> Result<viewer::Envelope, prost::DecodeError> {
    viewer::Envelope::decode(bytes)
}

/// Payload of a `CHUNK_KIND_DETECTIONS` chunk.
pub fn decode_detections(bytes: &[u8]) -> Result<payload::DetectionBatch, prost::DecodeError> {
    payload::DetectionBatch::decode(bytes)
}

/// Payload of a `CHUNK_KIND_MESH_BLOCK` chunk.
pub fn decode_mesh_block(bytes: &[u8]) -> Result<payload::MeshBlock, prost::DecodeError> {
    payload::MeshBlock::decode(bytes)
}

/// Payload of a `CHUNK_KIND_TERRAIN` chunk.
pub fn decode_terrain_tile(bytes: &[u8]) -> Result<payload::TerrainTile, prost::DecodeError> {
    payload::TerrainTile::decode(bytes)
}

/// Payload of a `CHUNK_KIND_BUILDINGS` chunk.
pub fn decode_building_set(bytes: &[u8]) -> Result<payload::BuildingSet, prost::DecodeError> {
    payload::BuildingSet::decode(bytes)
}

/// Builds outgoing envelopes with a monotonic sequence number, so the core can spot gaps in what
/// this screen sent.
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

/// `manual_control` with every axis clamped to -1..1 (non-finite values become 0).
pub fn manual_control(drone_id: u32, forward: f64, yaw: f64, climb: f64) -> Body {
    let axis = |v: f64| if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 };
    Body::ManualControl(viewer::ManualControl {
        drone_id,
        forward: axis(forward),
        yaw: axis(yaw),
        climb: axis(climb),
    })
}

/// Reconnect delays like the legacy `coreLink.ts`: start at the minimum, double after every
/// failed attempt up to the maximum, back to the minimum after a successful connect. The bounds
/// are [`crate::config::link`]'s (0.5 s to 5 s).
#[derive(Debug, Clone)]
pub struct Backoff {
    next_ms: u64,
    min_ms: u64,
    max_ms: u64,
}

impl Default for Backoff {
    fn default() -> Self {
        Self::new(crate::config::link::RECONNECT_MIN_MS, crate::config::link::RECONNECT_MAX_MS)
    }
}

impl Backoff {
    pub fn new(min_ms: u64, max_ms: u64) -> Self {
        let min_ms = min_ms.max(1);
        Self { next_ms: min_ms, min_ms, max_ms: max_ms.max(min_ms) }
    }

    pub fn next_delay_ms(&mut self) -> u64 {
        let wait = self.next_ms;
        self.next_ms = self.next_ms.saturating_mul(2).min(self.max_ms);
        wait
    }

    pub fn reset(&mut self) {
        self.next_ms = self.min_ms;
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

/// Turns the `ServerStatus` heartbeat into one warning when the model server loses its GPU.
/// The status repeats every tick, so only the change is reported; a model restart that clears
/// the error arms it again.
#[derive(Debug, Default)]
pub struct GpuLossAlert {
    shown: bool,
}

impl GpuLossAlert {
    /// The toast text to show now, if any.
    pub fn update(&mut self, status: &viewer::ServerStatus) -> Option<String> {
        let lost = !status.model_gpu_error.is_empty();
        let show = lost && !self.shown;
        self.shown = lost;
        show.then(|| "연산 서버 GPU 끊김 · CPU로 계속 처리해 복원이 느려집니다".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_loss_alerts_once_and_rearms_after_recovery() {
        let mut a = GpuLossAlert::default();
        let ok = viewer::ServerStatus::default();
        let lost = viewer::ServerStatus { model_gpu_error: "CUDA error".into(), ..Default::default() };
        assert!(a.update(&ok).is_none());
        assert!(a.update(&lost).is_some());
        assert!(a.update(&lost).is_none());
        assert!(a.update(&ok).is_none());
        assert!(a.update(&lost).is_some());
    }

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
    fn backoff_doubles_up_to_the_bound_and_resets() {
        let mut b = Backoff::default();
        let delays: Vec<u64> = (0..6).map(|_| b.next_delay_ms()).collect();
        assert_eq!(delays, vec![500, 1000, 2000, 4000, 5000, 5000]);
        b.reset();
        assert_eq!(b.next_delay_ms(), 500);
        // Degenerate bounds still wait and never shrink below the minimum.
        let mut z = Backoff::new(0, 0);
        assert_eq!((z.next_delay_ms(), z.next_delay_ms()), (1, 1));
    }

    #[test]
    fn seq_tracker_counts_missing_frames() {
        let mut t = SeqTracker::default();
        for s in [1, 2, 5, 6, 6, 9] {
            t.observe(s);
        }
        assert_eq!(t.gaps, 4);
        assert_eq!(t.last(), Some(9));
        t.reset();
        t.observe(1);
        assert_eq!(t.gaps, 4);
    }

    #[test]
    fn decodes_payloads() {
        let tile = payload::TerrainTile { cell: vec![1, 2], cols: 2, rows: 2, heights: vec![0.0; 4], ..Default::default() };
        assert_eq!(decode_terrain_tile(&tile.encode_to_vec()).unwrap(), tile);
        let set = payload::BuildingSet {
            cell: vec![1, 2],
            buildings: vec![payload::Building { id: "b".into(), outline: vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0], base: 1.0, height: 9.0 }],
        };
        assert_eq!(decode_building_set(&set.encode_to_vec()).unwrap(), set);
        assert!(decode_envelope(&[0xff, 0xff, 0xff]).is_err());
    }
}
