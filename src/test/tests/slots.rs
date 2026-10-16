//! The point slot path of stable display, across components (decision 0052 of pointblitz):
//!
//! ```text
//! cumulus3d display diff (KeyedDiff) ─> model slot encoder (slotstream.rs) ─> CHUNK_KIND_POINT_SLOTS
//!     ─> core (keeps every message in order) ─> screen ChunkStore (slot state per zone)
//! ```
//!
//! The encoder is compiled here from the model server's source: the model crate links libtorch,
//! which this test does not need. Its own unit tests run as part of this file too.

#[path = "../../skylens_model/models/skylens/slotstream.rs"]
#[allow(dead_code)] // the model uses all of it; this test does not
mod slotstream;

use cumulus3d_cli::live::{DisplayDelta, DisplayMirror, KeyedDiff};
use cumulus3d_core::io::PointCloud;
use pointblitz_io::slots::{self, Fold, HEADER_LEN};
use skylens_shared::proto::model::v1 as model;
use skylens_viewer::{Change, ChunkStore};

/// A display cloud of keyed points: key k at x = k (shifted by `dx`), owned by `zone_of(k)`.
fn cloud(keys: &[u64], dx: f32, zone_of: impl Fn(u64) -> usize) -> (PointCloud, Vec<usize>) {
    let mut c = PointCloud::default();
    let mut zones = Vec::new();
    for &k in keys {
        c.positions.push([k as f32 + dx, 0.5 * k as f32, 3.0]);
        c.colors.push([200, 100, k as u8]);
        c.keys.push(k);
        zones.push(zone_of(k));
    }
    (c, zones)
}

/// The deltas of a session: two zones appear, then zone 0 is placed again (moved) and loses a point.
fn deltas() -> Vec<(usize, DisplayDelta)> {
    let mut diff = KeyedDiff::new();
    let mut out = Vec::new();
    let z = |k: u64| (k / 100) as usize;
    let (c, zs) = cloud(&[0, 1, 2], 0.0, z);
    out.push((0, diff.apply(&c, &zs).unwrap()));
    let (c, zs) = cloud(&[0, 1, 2, 100, 101, 102], 0.0, z);
    out.push((1, diff.apply(&c, &zs).unwrap()));
    // Settling: zone 0 placed again 0.5 m east, point 2 gone.
    let mut keys: Vec<u64> = vec![0, 1];
    keys.extend([100, 101, 102]);
    let mut c = PointCloud::default();
    let mut zs = Vec::new();
    for &k in &keys {
        let dx = if z(k) == 0 { 0.5 } else { 0.0 };
        c.positions.push([k as f32 + dx, 0.5 * k as f32, 3.0]);
        c.colors.push([200, 100, k as u8]);
        c.keys.push(k);
        zs.push(z(k));
    }
    out.push((1, diff.apply(&c, &zs).unwrap()));
    out
}

#[test]
fn display_deltas_rebuild_the_same_points_on_a_screen_and_after_a_reconnect() {
    let mut enc = slotstream::SlotEncoder::new(11);
    let mut mirror = DisplayMirror::new();
    let mut fold = Fold::new();
    let mut screen = ChunkStore::new();
    let mut chunks = Vec::new();
    for (i, (zone, d)) in deltas().into_iter().enumerate() {
        mirror.apply(&d).unwrap();
        let mut c = enc.encode(&d, zone).unwrap();
        fold.apply(&c.payload).unwrap();
        c.sequence = i as u64 + 1;
        let a = screen.apply(&c);
        assert!(matches!(a.change, Change::SlotsApplied { .. }), "{:?}", a.change);
        chunks.push(c);
    }
    let sp = screen.slots();
    assert_eq!((sp.seq(), sp.points(), sp.stream()), (3, mirror.len() as u64, 11));
    assert_eq!(sp.zone(0).unwrap().points, 2);
    assert_eq!(sp.zone(1).unwrap().points, 3);
    assert_eq!(screen.levels()[&1], model::Level::Refined);
    // The screen's folded state is the producer's folded state.
    assert_eq!(sp.state_message(), fold.message());

    // A reconnect: the core replays every message in order; the state is the same, the epoch new.
    let epoch = sp.epoch();
    screen.clear();
    for c in &chunks {
        screen.apply(c);
    }
    assert_eq!(screen.slots().state_message(), fold.message());
    assert_ne!(screen.slots().epoch(), epoch);
    for (c, i) in chunks.iter().zip(1..) {
        assert_eq!(slots::decode_header(&c.payload).unwrap().seq, i);
        assert!(c.payload.len() >= HEADER_LEN);
    }
}

#[test]
fn a_new_model_run_starts_the_screen_over() {
    let mut screen = ChunkStore::new();
    for (stream, n) in [(11u32, 3usize), (12, 1)] {
        let mut enc = slotstream::SlotEncoder::new(stream);
        for (zone, d) in deltas().into_iter().take(n) {
            screen.apply(&enc.encode(&d, zone).unwrap());
        }
    }
    let sp = screen.slots();
    assert_eq!((sp.stream(), sp.seq(), sp.points()), (12, 1, 3));
}
