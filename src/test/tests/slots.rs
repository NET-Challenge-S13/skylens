//! The point slot path of stable display, across components (decision 0052 of pointblitz):
//!
//! ```text
//! cumulus3d live/stream/ ─> model follower (slotstream.rs) ─> CHUNK_KIND_POINT_SLOTS
//!     ─> core (keeps every message in order) ─> screen ChunkStore (slot state per zone)
//! ```
//!
//! The follower is compiled here from the model server's source: the model crate links libtorch,
//! which this test does not need. Its own unit tests run as part of this file too.

#[path = "../../skylens_model/models/skylens/slotstream.rs"]
#[allow(dead_code)] // the model uses all of it; this test does not
mod slotstream;

use pointblitz_io::slots::{self, Fold, HEADER_LEN};
use skylens_shared::proto::model::v1 as model;
use skylens_viewer::{Change, ChunkStore};
use std::path::{Path, PathBuf};

const ANCHOR: skylens_shared::geo::GeoAnchor = skylens_shared::geo::Gps { lat: 36.3685, lon: 127.3475, alt: 30.0 };

/// A v4 pack of adds (id, x, zone), moves (id, x) and removes.
fn pack(adds: &[(u32, f32, u8)], moves: &[(u32, f32)], removes: &[u32]) -> (Vec<u8>, [u32; 3]) {
    let mut p = Vec::new();
    for &(id, x, zone) in adds {
        slots::Record { id, position: [x, 0.5 * x, 3.0 + f32::from(zone)], color: [200, 100, 50], zone }.write(&mut p);
    }
    for &(id, x) in moves {
        slots::Record { id, position: [x, x, 9.0], color: [1, 2, 3], zone: 0 }.write(&mut p);
    }
    for id in removes {
        p.extend(id.to_le_bytes());
    }
    (p, [adds.len() as u32, moves.len() as u32, removes.len() as u32])
}

/// One event: triggering zone, kind, (pack, [n_add, n_move, n_remove]).
type Event<'a> = (u32, &'a str, (Vec<u8>, [u32; 3]));

/// Writes a v4 stream directory the way cumulus3d does (packs, then the manifest).
fn write_stream(dir: &Path, events: &[Event<'_>]) {
    std::fs::create_dir_all(dir.join("packs")).unwrap();
    let mut items = Vec::new();
    for (i, (zone, kind, (p, n))) in events.iter().enumerate() {
        let seq = i + 1;
        let name = format!("packs/e_{seq:04}.bin");
        std::fs::write(dir.join(&name), p).unwrap();
        items.push(format!(
            r#"{{"seq":{seq},"t":{seq}.0,"zone":{zone},"kind":"{kind}","points":0,"pack":"{name}","n_add":{},"n_move":{},"n_remove":{},"bytes":{}}}"#,
            n[0],
            n[1],
            n[2],
            p.len()
        ));
    }
    let m = format!(
        r#"{{"version":4,"frame":"enu","origin_gps":[{},{},{}],"capacity":0,"events":[{}]}}"#,
        ANCHOR.lat,
        ANCHOR.lon,
        ANCHOR.alt,
        items.join(",")
    );
    std::fs::write(dir.join("manifest.json"), m).unwrap();
}

fn temp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("skylens-slots-path-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

#[test]
fn follower_chunks_rebuild_the_same_points_on_a_screen_and_after_a_reconnect() {
    let dir = temp("screen");
    write_stream(
        &dir,
        &[
            (0, "refined", pack(&[(0, 1.0, 0), (1, 2.0, 0), (2, 3.0, 0)], &[], &[])),
            (1, "refined", pack(&[(3, 4.0, 1), (4, 5.0, 1), (5, 6.0, 1)], &[], &[])),
            // Settling: zone 0 placed again, one point leaves for zone 1 (remove + new id).
            (1, "replace", pack(&[(6, 7.0, 1)], &[(0, 1.5), (1, 2.5)], &[2])),
        ],
    );
    let mut f = slotstream::Follower::new(&dir, 11, ANCHOR);
    let chunks = f.poll().unwrap();
    assert_eq!(chunks.len(), 3);

    let mut fold = Fold::new();
    let mut screen = ChunkStore::new();
    for (i, c) in chunks.iter().enumerate() {
        fold.apply(&c.payload).unwrap();
        let mut c = c.clone();
        c.sequence = i as u64 + 1;
        let a = screen.apply(&c);
        assert!(matches!(a.change, Change::SlotsApplied { .. }), "{:?}", a.change);
    }
    let sp = screen.slots();
    assert_eq!((sp.seq(), sp.points(), sp.stream()), (3, fold.points() as u64, 11));
    assert_eq!(sp.zone(0).unwrap().points, 2);
    assert_eq!(sp.zone(1).unwrap().points, 4);
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
    let _ = std::fs::remove_dir_all(&dir);
}

/// The real theater30 stable run through the follower into a screen chunk store.
/// `SKYLENS_THEATER30_STREAM=<...>/theater30-stable/live/stream cargo test -p skylens-tests -- --ignored`
#[test]
#[ignore = "needs SKYLENS_THEATER30_STREAM (a cumulus3d v4 live/stream directory)"]
fn theater30_stream_reaches_the_screen_whole() {
    let Some(dir) = std::env::var_os("SKYLENS_THEATER30_STREAM") else { return };
    let dir = PathBuf::from(dir);
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    let events = manifest["events"].as_array().unwrap();
    let mut f = slotstream::Follower::new(&dir, 1, ANCHOR);
    let chunks = f.poll().unwrap();
    assert_eq!(chunks.len(), events.len());
    let mut screen = ChunkStore::new();
    for (c, e) in chunks.iter().zip(events) {
        screen.apply(c);
        assert_eq!(screen.slots().points(), e["points"].as_u64().unwrap(), "after event {}", e["seq"]);
    }
    let sp = screen.slots();
    assert_eq!(u64::from(sp.capacity()), manifest["capacity"].as_u64().unwrap());
    let per_zone: Vec<(u32, u64)> = sp.zones().map(|(z, s)| (z, s.points)).collect();
    assert_eq!(per_zone.iter().map(|(_, n)| n).sum::<u64>(), sp.points());
    eprintln!("theater30: {} events, {} points, zones {per_zone:?}", chunks.len(), sp.points());
}
