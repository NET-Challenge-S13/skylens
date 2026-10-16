//! Replay of a recorded cumulus3d live stream (manifest v4 + `packs/e_NNNN.bin`)
//! through the engine's normal publishing path, for tools and tests that run
//! without a GPU. See `examples/delta_replay.rs`.
//!
//! Stream layout: `manifest.json` (`version` 4, `frame` "enu", `origin_gps`,
//! `capacity`, `events[{seq,t,zone,kind,points,pack,n_add,n_move,n_remove,bytes}]`)
//! and one little-endian pack per event with the sections, in order,
//! `n_add` x 20 B (u32 id, f32 x,y,z, u8 r,g,b, u8 zone), `n_move` x 20 B
//! (u32 id, f32 x,y,z, u8 r,g,b, u8 pad) and `n_remove` x 4 B (u32 id).

use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::{Context, Result, bail, ensure};
use cumulus3d_cli::events::{Event, Meta};
use cumulus3d_cli::live::{DeltaKind, DisplayDelta, DisplayFrame, PointAdd, PointMove};
use serde::Deserialize;

/// Record size of an add and of a move.
const REC: usize = 20;

#[derive(Debug, Clone, Deserialize)]
pub struct ManifestEvent {
    pub seq: u64,
    pub t: f64,
    pub zone: usize,
    pub kind: String,
    pub points: usize,
    /// `null` for the empty final event.
    pub pack: Option<String>,
    pub n_add: usize,
    pub n_move: usize,
    pub n_remove: usize,
    pub bytes: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub frame: String,
    pub origin_gps: [f64; 3],
    pub capacity: u32,
    pub events: Vec<ManifestEvent>,
}

/// One recorded display delta, ready to be injected.
#[derive(Debug, Clone)]
pub struct ReplayEvent {
    pub seq: u64,
    /// Seconds since the session start (manifest `t`).
    pub t: f64,
    pub zone: usize,
    pub kind: DeltaKind,
    pub delta: Arc<DisplayDelta>,
}

/// Reads and checks `manifest.json`.
pub fn read_manifest(dir: &Path) -> Result<Manifest> {
    let path = dir.join("manifest.json");
    let raw = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let m: Manifest =
        serde_json::from_slice(&raw).with_context(|| format!("parse {}", path.display()))?;
    ensure!(
        m.version == 4,
        "unsupported stream version {} (want 4)",
        m.version
    );
    ensure!(
        m.frame == "enu",
        "unsupported stream frame {:?} (want \"enu\")",
        m.frame
    );
    Ok(m)
}

fn kind_of(s: &str) -> Result<DeltaKind> {
    Ok(match s {
        "refined" => DeltaKind::Refined,
        "replace" => DeltaKind::Replace,
        "final" => DeltaKind::Final,
        other => bail!("unknown event kind {other:?}"),
    })
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

fn f32_at(b: &[u8], i: usize) -> f32 {
    f32::from_bits(u32_at(b, i))
}

#[allow(clippy::chunks_exact_to_as_chunks)]
/// Decodes one pack into a delta. `capacity` is the largest slot id + 1 seen
/// so far (cumulus3d's meaning); `total_points` the manifest's `points`.
pub fn parse_pack(
    bytes: &[u8],
    ev: &ManifestEvent,
    seq: u64,
    capacity: u32,
) -> Result<(DisplayDelta, u32)> {
    let want = ev.n_add * REC + ev.n_move * REC + ev.n_remove * 4;
    ensure!(
        bytes.len() == want,
        "pack {}: {} bytes, sections need {want}",
        ev.pack.as_deref().unwrap_or("-"),
        bytes.len()
    );
    let (add_b, rest) = bytes.split_at(ev.n_add * REC);
    let (move_b, remove_b) = rest.split_at(ev.n_move * REC);
    let mut capacity = capacity;
    let adds: Vec<PointAdd> = add_b
        .chunks_exact(REC)
        .map(|r| PointAdd {
            id: u32_at(r, 0),
            xyz: [f32_at(r, 4), f32_at(r, 8), f32_at(r, 12)],
            rgb: [r[16], r[17], r[18]],
            zone: r[19],
        })
        .collect();
    for a in &adds {
        capacity = capacity.max(a.id + 1);
    }
    let moves = move_b
        .chunks_exact(REC)
        .map(|r| PointMove {
            id: u32_at(r, 0),
            xyz: [f32_at(r, 4), f32_at(r, 8), f32_at(r, 12)],
            rgb: [r[16], r[17], r[18]],
        })
        .collect();
    let removes = remove_b.chunks_exact(4).map(|r| u32_at(r, 0)).collect();
    Ok((
        DisplayDelta {
            seq,
            adds,
            moves,
            removes,
            total_points: ev.points,
            capacity,
        },
        capacity,
    ))
}

/// Loads every event of a stream directory, checking pack sizes against the
/// manifest and that the running point count follows `points`.
pub fn load_stream(dir: &Path) -> Result<Vec<ReplayEvent>> {
    let m = read_manifest(dir)?;
    load_events(dir, &m)
}

/// [`load_stream`] with an already read manifest.
pub fn load_events(dir: &Path, m: &Manifest) -> Result<Vec<ReplayEvent>> {
    let mut out = Vec::with_capacity(m.events.len());
    let (mut capacity, mut shown) = (0u32, 0usize);
    for ev in &m.events {
        let bytes = match &ev.pack {
            Some(pack) => {
                let path = dir.join(pack);
                std::fs::read(&path).with_context(|| format!("read {}", path.display()))?
            }
            None => Vec::new(),
        };
        ensure!(
            bytes.len() as u64 == ev.bytes,
            "pack {}: {} bytes, manifest says {}",
            ev.pack.as_deref().unwrap_or("-"),
            bytes.len(),
            ev.bytes
        );
        let (delta, cap) = parse_pack(&bytes, ev, ev.seq, capacity)?;
        capacity = cap;
        shown = (shown + ev.n_add)
            .checked_sub(ev.n_remove)
            .with_context(|| format!("event {}: removes exceed the shown points", ev.seq))?;
        ensure!(
            shown == ev.points,
            "event {}: {shown} points after the delta, manifest says {}",
            ev.seq,
            ev.points
        );
        out.push(ReplayEvent {
            seq: ev.seq,
            t: ev.t,
            zone: ev.zone,
            kind: kind_of(&ev.kind)?,
            delta: Arc::new(delta),
        });
    }
    ensure!(
        capacity <= m.capacity,
        "slot ids reach {capacity}, manifest capacity is {}",
        m.capacity
    );
    Ok(out)
}

/// The cumulus3d event for a recorded delta, in the ENU frame about `origin_gps`.
/// The engine anchor must equal `origin_gps` so the consumer does not translate.
pub fn to_event(ev: &ReplayEvent, origin_gps: [f64; 3]) -> Event {
    Event::DisplayDelta {
        meta: Meta {
            version: 1,
            at: SystemTime::now(),
        },
        delta: Arc::clone(&ev.delta),
        zone: ev.zone,
        kind: ev.kind,
        t: ev.t,
        frame: DisplayFrame {
            enu: true,
            origin_gps: Some(origin_gps),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::skylens::{DEFAULT_DISPLAY_STRIDE, Engine, EngineConfig, PointsMode};
    use skylens_shared::delta::{from_chunk, iter_adds, iter_moves, iter_removes};
    use skylens_shared::proto::model::v1::ChunkKind;

    const ORIGIN: [f64; 3] = [36.3685, 127.3475, 30.0];

    type Slot = Option<([f32; 3], [u8; 3], u8)>;

    fn add(id: u32, x: f32, c: u8, zone: u8) -> Vec<u8> {
        let mut b = id.to_le_bytes().to_vec();
        for v in [x, 2.0, 3.0] {
            b.extend(v.to_le_bytes());
        }
        b.extend([c, c + 1, c + 2, zone]);
        b
    }

    fn mv(id: u32, x: f32, c: u8) -> Vec<u8> {
        let mut b = add(id, x, c, 0);
        b[19] = 0;
        b
    }

    /// Writes a 3-event stream (refined adds / replace with moves, removes and
    /// adds / final) and returns the manifest json events' packs.
    fn write_stream(dir: &Path) {
        std::fs::create_dir_all(dir.join("packs")).unwrap();
        let p1: Vec<u8> = (0..6)
            .flat_map(|i| add(i, i as f32, 10 + i as u8, 0))
            .collect();
        let mut p2: Vec<u8> = [add(6, 6.0, 40, 1), add(7, 7.0, 50, 1)].concat();
        p2.extend([mv(1, 1.5, 90), mv(2, 2.0, 91)].concat());
        p2.extend([3u32, 4u32].iter().flat_map(|i| i.to_le_bytes()));
        for (n, p) in [(1, &p1), (2, &p2), (3, &Vec::new())] {
            std::fs::write(dir.join(format!("packs/e_{n:04}.bin")), p).unwrap();
        }
        let ev = |seq, t, zone, kind, points, n_add, n_move, n_remove, bytes| {
            serde_json::json!({"seq": seq, "t": t, "zone": zone, "kind": kind, "points": points,
                "pack": if bytes == 0 { serde_json::Value::Null } else { format!("packs/e_{seq:04}.bin").into() }, "n_add": n_add, "n_move": n_move, "n_remove": n_remove, "bytes": bytes})
        };
        let m = serde_json::json!({"version": 4, "frame": "enu", "origin_gps": ORIGIN, "capacity": 8, "events": [
            ev(1, 1.0, 0, "refined", 6, 6, 0, 0, 120),
            ev(2, 2.0, 1, "replace", 6, 2, 2, 2, 88),
            ev(3, 3.0, 1, "final", 6, 0, 0, 0, 0)]});
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec_pretty(&m).unwrap(),
        )
        .unwrap();
    }

    /// Applies the pack records directly (removes, adds, moves).
    fn reference(events: &[ReplayEvent]) -> Vec<Slot> {
        let mut s: Vec<Slot> = Vec::new();
        for e in events {
            let d = &e.delta;
            s.resize(s.len().max(d.capacity as usize), None);
            for &id in &d.removes {
                s[id as usize] = None;
            }
            for a in &d.adds {
                s[a.id as usize] = Some((a.xyz, a.rgb, a.zone));
            }
            for m in &d.moves {
                let slot = s[m.id as usize].as_mut().unwrap();
                slot.0 = m.xyz;
                slot.1 = m.rgb;
            }
        }
        s
    }

    #[test]
    fn replayed_stream_reproduces_the_final_state_from_the_latest_snapshot() {
        let dir = std::env::temp_dir().join(format!("skylens-replay-{}", std::process::id()));
        write_stream(&dir);
        let events = load_stream(&dir).unwrap();
        assert_eq!(
            events
                .iter()
                .map(|e| (
                    e.delta.adds.len(),
                    e.delta.moves.len(),
                    e.delta.removes.len()
                ))
                .collect::<Vec<_>>(),
            [(6, 0, 0), (2, 2, 2), (0, 0, 0)]
        );
        assert_eq!(events[1].delta.capacity, 8);
        let want = reference(&events);

        let cfg = EngineConfig {
            anchor: skylens_shared::geo::Gps {
                lat: ORIGIN[0],
                lon: ORIGIN[1],
                alt: ORIGIN[2],
            },
            cameras: ["left", "center", "right"].map(String::from).to_vec(),
            zone_span: 12,
            zone_overlap: 2,
            work_dir: dir.join("work"),
            gpu: false,
            points_mode: PointsMode::Delta,
            display_stride: DEFAULT_DISPLAY_STRIDE,
            snapshot_every: 2,
        };
        let engine = Engine::new(cfg, None).unwrap();
        for e in &events {
            engine.inject_event(to_event(e, ORIGIN)).unwrap();
        }
        // Wait until the final delta is published, then subscribe: the replay
        // starts from the latest snapshot the log retained.
        let t0 = std::time::Instant::now();
        let mut chunks = Vec::new();
        loop {
            let mut rx = engine.subscribe(0);
            chunks.clear();
            while let Ok(c) = rx.try_recv() {
                assert_eq!(c.kind, ChunkKind::PointsDelta as i32);
                chunks.push(from_chunk(&c).unwrap());
            }
            let fin = skylens_shared::proto::payload::v1::DeltaKind::Final as i32;
            if chunks.last().is_some_and(|d| d.kind == fin && !d.snapshot) {
                break;
            }
            assert!(
                t0.elapsed().as_secs() < 20,
                "timed out; got {} chunks",
                chunks.len()
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        // Snapshot after delta 2 bumps the generation; the log keeps it and delta 3.
        assert!(chunks[0].snapshot, "replay starts at the latest snapshot");
        assert_eq!(chunks.len(), 2);
        let mut got: Vec<Slot> = Vec::new();
        for d in &chunks {
            if d.snapshot {
                got.clear();
            }
            got.resize(got.len().max(d.capacity as usize), None);
            for id in iter_removes(&d.removes).unwrap() {
                got[id as usize] = None;
            }
            for a in iter_adds(&d.adds).unwrap() {
                got[a.id as usize] = Some((a.xyz, a.rgb, a.zone));
            }
            for m in iter_moves(&d.moves).unwrap() {
                let s = got[m.id as usize].as_mut().unwrap();
                s.0 = m.xyz;
                s.1 = m.rgb;
            }
        }
        assert_eq!(got, want);
        assert_eq!(got.iter().flatten().count(), 6);
        drop(engine);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_streams_are_rejected() {
        let dir = std::env::temp_dir().join(format!("skylens-replay-bad-{}", std::process::id()));
        write_stream(&dir);
        std::fs::write(dir.join("packs/e_0001.bin"), [0u8; 7]).unwrap();
        assert!(load_stream(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Real recording; run with `cargo test -p skylens-model --release -- --ignored replay_real_stream --nocapture`
    /// (stream dir from `SKYLENS_REPLAY_STREAM`).
    #[test]
    #[ignore]
    fn replay_real_stream() {
        let dir = std::env::var("SKYLENS_REPLAY_STREAM").expect("set SKYLENS_REPLAY_STREAM");
        let t = std::time::Instant::now();
        let events = load_stream(Path::new(&dir)).unwrap();
        let m = read_manifest(Path::new(&dir)).unwrap();
        assert_eq!(events.len(), m.events.len());
        for (e, me) in events.iter().zip(&m.events) {
            assert_eq!(
                (
                    e.delta.adds.len(),
                    e.delta.moves.len(),
                    e.delta.removes.len()
                ),
                (me.n_add, me.n_move, me.n_remove)
            );
        }
        println!("parsed {} events in {:.2?}", events.len(), t.elapsed());
    }
}
