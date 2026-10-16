//! Follower of the cumulus3d live diff stream (v4) the live node writes in stable display.
//!
//! cumulus3d 0.4.0 produces the point slot diff only through its file sinks:
//! `<out>/live/stream/manifest.json` plus `packs/e_NNNN.bin` (decision 0052 of pointblitz). The
//! follower re-reads the manifest every [`POLL_INTERVAL`] with `pointblitz_io::slots::dir`, wraps
//! each new event's pack in the 32-byte slot message header (the pack itself is not parsed or
//! changed) and publishes it as one `CHUNK_KIND_POINT_SLOTS` chunk:
//!
//! | chunk field | value |
//! |---|---|
//! | `segment` | the zone that triggered the event |
//! | `level` | refined (stable display has no preview) |
//! | `version` | the message seq (1, 2, ... in manifest order) |
//! | `generation` | the stream id of this model run ([`stream_id`]) |
//!
//! The manifest is rewritten atomically after each pack, so every event it lists has its pack on
//! disk. An event that cannot be read yet (a pack missing or short) is retried on the next poll;
//! later events wait behind it, so messages always leave in seq order.
//!
//! The module needs nothing from the rest of the model server (no libtorch), so the cross-component
//! tests in `src/test` compile it on its own and run its tests on machines without CUDA.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use pointblitz_io::slots::dir::StreamDir;
use skylens_shared::geo::GeoAnchor;
use skylens_shared::proto::model::v1::{Chunk, ChunkKind, Level};

/// How often the manifest is re-read.
pub const POLL_INTERVAL: Duration = Duration::from_millis(200);
/// Largest difference (degrees, metres) between the manifest's origin and the engine anchor that
/// still counts as the same ENU origin.
const ORIGIN_TOLERANCE_DEG: f64 = 1e-7;
const ORIGIN_TOLERANCE_M: f64 = 0.01;

/// The stream directory under a node's output folder.
pub fn stream_dir(out: &Path) -> PathBuf {
    out.join("live").join("stream")
}

/// Whole seconds since the Unix epoch at `at` (wrapping in 2106), at least 1.
pub fn stream_id(at: SystemTime) -> u32 {
    at.duration_since(UNIX_EPOCH).map(|d| d.as_secs() as u32).unwrap_or(1).max(1)
}

/// Prefix of a live run folder under `<work_dir>/runs/`: `live-<stream id>-<pid>`.
const RUN_PREFIX: &str = "live-";

/// Folder name of a live run.
pub fn run_folder_name(stream: u32, pid: u32) -> String {
    format!("{RUN_PREFIX}{stream}-{pid}")
}

/// Stream id of a live run folder name (`live-<id>-<pid>`).
pub fn run_folder_stream(name: &str) -> Option<u32> {
    let rest = name.strip_prefix(RUN_PREFIX)?;
    let (id, pid) = rest.split_once('-')?;
    pid.parse::<u32>().ok()?;
    id.parse().ok()
}

/// Live run folders under `runs`, oldest stream first.
fn run_folders(runs: &Path) -> Vec<(u32, PathBuf)> {
    let Ok(rd) = std::fs::read_dir(runs) else { return Vec::new() };
    let mut out: Vec<(u32, PathBuf)> = rd
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| Some((run_folder_stream(e.file_name().to_str()?)?, e.path())))
        .collect();
    out.sort();
    out
}

/// Stream id of a new model run: higher than every run folder under `runs` and at least the wall
/// clock second, so a restart within the same second or after the clock went back still gets a
/// higher id (the core and the screens drop a stream whose id is not higher than the current one).
pub fn next_stream_id(runs: &Path, now: SystemTime) -> u32 {
    let after = run_folders(runs).last().map_or(0, |(id, _)| id.saturating_add(1));
    stream_id(now).max(after)
}

/// Removes the oldest live run folders under `runs` so that at most `keep` remain. Returns how
/// many were removed; a folder that cannot be removed is logged and skipped.
pub fn prune_runs(runs: &Path, keep: usize) -> usize {
    let folders = run_folders(runs);
    let excess = folders.len().saturating_sub(keep);
    let mut removed = 0;
    for (_, dir) in folders.into_iter().take(excess) {
        match std::fs::remove_dir_all(&dir) {
            Ok(()) => removed += 1,
            Err(e) => tracing::warn!("old live run {}: {e}", dir.display()),
        }
    }
    removed
}

/// First chunk sequence of a model run minus one: the stream id in the upper 32 bits. The core
/// resumes `SubscribeChunks` after the last sequence it stored; with sequences unique per run a
/// restarted model's chunks are all above that, so none of them is skipped.
pub fn sequence_base(stream: u32) -> u64 {
    u64::from(stream) << 32
}

/// One slot message as a chunk.
pub fn slots_chunk(stream: u32, zone: u32, seq: u32, message: Vec<u8>) -> Chunk {
    Chunk {
        kind: ChunkKind::PointSlots as i32,
        segment: zone,
        level: Level::Refined as i32,
        version: seq,
        generation: stream,
        payload: message,
        ..Default::default()
    }
}

/// Reader state of one stream directory.
pub struct Follower {
    dir: PathBuf,
    stream: u32,
    anchor: GeoAnchor,
    reader: Option<StreamDir>,
    /// Seq of the next event to publish.
    next: u32,
    origin_checked: bool,
}

impl Follower {
    pub fn new(dir: impl Into<PathBuf>, stream: u32, anchor: GeoAnchor) -> Self {
        Self { dir: dir.into(), stream, anchor, reader: None, next: 1, origin_checked: false }
    }

    /// Seq of the next event to publish (1 before any).
    pub fn next_seq(&self) -> u32 {
        self.next
    }

    /// Events of the manifest not published yet, as chunks in seq order. Nothing (and no error)
    /// while the manifest does not exist yet. On an error the events from the failing one on are
    /// returned again by a later call.
    pub fn poll(&mut self) -> anyhow::Result<Vec<Chunk>> {
        if self.reader.is_none() {
            if !self.dir.join("manifest.json").is_file() {
                return Ok(Vec::new());
            }
            self.reader = Some(StreamDir::open(&self.dir)?);
        } else if let Some(r) = self.reader.as_mut() {
            r.refresh()?;
        }
        let reader = self.reader.as_mut().expect("opened above");
        if !self.origin_checked {
            self.origin_checked = true;
            check_origin(reader.manifest().origin_gps, self.anchor);
        }
        let pending: Vec<_> = reader.events().iter().filter(|e| e.seq >= self.next).cloned().collect();
        let mut out = Vec::with_capacity(pending.len());
        for e in pending {
            if e.seq != self.next {
                tracing::warn!("live stream: event {} follows {}; seqs are missing", e.seq, self.next.saturating_sub(1));
            }
            let message = match reader.message(e.seq) {
                Ok(m) => m,
                Err(err) if out.is_empty() => return Err(err.into()),
                Err(err) => {
                    // Publish what is in order now; the failing event is retried next time.
                    tracing::debug!("live stream event {}: {err}; retrying", e.seq);
                    break;
                }
            };
            tracing::info!(
                "live stream event {} ({} zone {}): +{} ~{} -{}, {} points, {:.1} MB",
                e.seq,
                e.kind,
                e.zone,
                e.n_add,
                e.n_move,
                e.n_remove,
                e.points,
                message.len() as f64 / 1e6
            );
            out.push(slots_chunk(self.stream, e.zone, e.seq, message));
            self.next = e.seq + 1;
        }
        Ok(out)
    }
}

fn check_origin(origin: Option<[f64; 3]>, anchor: GeoAnchor) {
    match origin {
        Some([lat, lon, alt]) => {
            if (lat - anchor.lat).abs() > ORIGIN_TOLERANCE_DEG
                || (lon - anchor.lon).abs() > ORIGIN_TOLERANCE_DEG
                || (alt - anchor.alt).abs() > ORIGIN_TOLERANCE_M
            {
                tracing::warn!(
                    "live stream origin {lat:.7},{lon:.7},{alt:.2} is not the engine anchor {:.7},{:.7},{:.2}; \
                     screens place the slots about the anchor",
                    anchor.lat,
                    anchor.lon,
                    anchor.alt
                );
            }
        }
        None => tracing::warn!("live stream manifest has no origin; slots are assumed to be about the anchor"),
    }
}

/// Follows `dir` until `stop`, handing every new event's chunk to `publish` in seq order.
pub fn spawn(
    dir: PathBuf,
    stream: u32,
    anchor: GeoAnchor,
    publish: impl Fn(Vec<Chunk>) + Send + 'static,
    stop: Arc<AtomicBool>,
) -> anyhow::Result<JoinHandle<()>> {
    let handle = std::thread::Builder::new().name("skylens-slots".into()).spawn(move || {
        let mut f = Follower::new(dir, stream, anchor);
        let mut failing = false;
        while !stop.load(Ordering::Relaxed) {
            match f.poll() {
                Ok(chunks) => {
                    failing = false;
                    if !chunks.is_empty() {
                        publish(chunks);
                    }
                }
                Err(e) if !failing => {
                    failing = true;
                    tracing::warn!("live stream: {e:#} (retrying)");
                }
                Err(e) => tracing::debug!("live stream: {e:#}"),
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    })?;
    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pointblitz_io::slots::{self, Diff, Fold, HEADER_LEN};

    const ANCHOR: GeoAnchor = skylens_shared::geo::Gps { lat: 36.3685, lon: 127.3475, alt: 30.0 };

    /// One v4 pack: adds (id, x, zone), moves (id, x), removes.
    fn pack(adds: &[(u32, f32, u8)], moves: &[(u32, f32)], removes: &[u32]) -> Vec<u8> {
        let mut p = Vec::new();
        for &(id, x, zone) in adds {
            slots::Record { id, position: [x, -x, 1.0], color: [10, 20, 30], zone }.write(&mut p);
        }
        for &(id, x) in moves {
            slots::Record { id, position: [x, x, 2.0], color: [40, 50, 60], zone: 0 }.write(&mut p);
        }
        for id in removes {
            p.extend(id.to_le_bytes());
        }
        p
    }

    struct Ev {
        zone: u32,
        kind: &'static str,
        counts: [u32; 3],
        pack: Vec<u8>,
    }

    /// Writes `events` the way cumulus3d does: packs first, then the manifest (all events so far).
    fn write_stream(dir: &Path, events: &[Ev], capacity: u32) {
        std::fs::create_dir_all(dir.join("packs")).unwrap();
        let mut items = Vec::new();
        for (i, e) in events.iter().enumerate() {
            let seq = i + 1;
            let name = (!e.pack.is_empty()).then(|| format!("packs/e_{seq:04}.bin"));
            if let Some(n) = &name {
                std::fs::write(dir.join(n), &e.pack).unwrap();
            }
            let pack = name.map_or("null".to_string(), |n| format!("\"{n}\""));
            items.push(format!(
                r#"{{"seq":{seq},"t":{seq}.5,"zone":{},"kind":"{}","points":0,"pack":{pack},"n_add":{},"n_move":{},"n_remove":{},"bytes":{}}}"#,
                e.zone,
                e.kind,
                e.counts[0],
                e.counts[1],
                e.counts[2],
                e.pack.len()
            ));
        }
        let manifest = format!(
            r#"{{"version":4,"frame":"enu","origin_gps":[36.3685,127.3475,30.0],"capacity":{capacity},"events":[{}]}}"#,
            items.join(",")
        );
        let tmp = dir.join("manifest.json.tmp");
        std::fs::write(&tmp, manifest).unwrap();
        std::fs::rename(tmp, dir.join("manifest.json")).unwrap();
    }

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("skylens-slots-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn events() -> Vec<Ev> {
        vec![
            Ev { zone: 0, kind: "refined", counts: [3, 0, 0], pack: pack(&[(0, 1.0, 0), (1, 2.0, 0), (2, 3.0, 0)], &[], &[]) },
            Ev { zone: 1, kind: "refined", counts: [2, 0, 0], pack: pack(&[(3, 4.0, 1), (4, 5.0, 1)], &[], &[]) },
            Ev { zone: 1, kind: "replace", counts: [1, 1, 1], pack: pack(&[(0, 6.0, 1)], &[(2, 7.0)], &[1]) },
            Ev { zone: 1, kind: "final", counts: [0, 0, 0], pack: Vec::new() },
        ]
    }

    #[test]
    fn nothing_until_the_manifest_exists_then_every_event_in_order() {
        let dir = temp("order");
        let mut f = Follower::new(&dir, 42, ANCHOR);
        assert!(f.poll().unwrap().is_empty());

        let all = events();
        write_stream(&dir, &all[..2], 5);
        let first = f.poll().unwrap();
        assert_eq!(first.iter().map(|c| (c.version, c.segment)).collect::<Vec<_>>(), vec![(1, 0), (2, 1)]);
        assert!(f.poll().unwrap().is_empty(), "nothing new");

        write_stream(&dir, &all, 5);
        let rest = f.poll().unwrap();
        assert_eq!(rest.iter().map(|c| c.version).collect::<Vec<_>>(), vec![3, 4]);
        assert_eq!(f.next_seq(), 5);

        let mut fold = Fold::new();
        for c in first.iter().chain(&rest) {
            assert_eq!(c.kind, ChunkKind::PointSlots as i32);
            assert_eq!((c.level, c.generation), (Level::Refined as i32, 42));
            let h = slots::validate(&c.payload).unwrap();
            assert_eq!(h.seq, c.version);
            fold.apply(&c.payload).unwrap();
        }
        // The packs travel unchanged behind the header.
        for (c, e) in first.iter().chain(&rest).zip(&all) {
            assert_eq!(&c.payload[HEADER_LEN..], e.pack.as_slice());
        }
        // Capacity per event: the highest add id + 1 so far.
        let caps: Vec<u32> = first.iter().chain(&rest).map(|c| slots::decode_header(&c.payload).unwrap().capacity).collect();
        assert_eq!(caps, vec![3, 5, 5, 5]);
        assert_eq!((fold.seq(), fold.points()), (4, 4));
        let state = fold.message();
        let d = Diff::parse(&state).unwrap();
        assert_eq!(d.add_records().map(|r| (r.id, r.zone)).collect::<Vec<_>>(), vec![(0, 1), (2, 0), (3, 1), (4, 1)]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_pack_not_on_disk_yet_is_retried_and_later_events_wait() {
        let dir = temp("retry");
        let all = events();
        write_stream(&dir, &all[..3], 5);
        std::fs::remove_file(dir.join("packs/e_0002.bin")).unwrap();
        let mut f = Follower::new(&dir, 7, ANCHOR);
        let got = f.poll().unwrap();
        assert_eq!(got.iter().map(|c| c.version).collect::<Vec<_>>(), vec![1], "2 is missing, 3 waits behind it");
        assert!(f.poll().is_err(), "the missing pack is reported");
        write_stream(&dir, &all[..3], 5);
        let got = f.poll().unwrap();
        assert_eq!(got.iter().map(|c| c.version).collect::<Vec<_>>(), vec![2, 3]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_follower_thread_publishes_every_event_once() {
        let dir = temp("thread");
        write_stream(&dir, &events(), 5);
        let got: Arc<std::sync::Mutex<Vec<Chunk>>> = Arc::default();
        let sink = Arc::clone(&got);
        let stop = Arc::new(AtomicBool::new(false));
        let publish = move |c: Vec<Chunk>| sink.lock().unwrap().extend(c);
        let h = spawn(dir.clone(), 9, ANCHOR, publish, Arc::clone(&stop)).unwrap();
        let t0 = std::time::Instant::now();
        while got.lock().unwrap().len() < 4 && t0.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(20));
        }
        // A few more polls change nothing.
        std::thread::sleep(3 * POLL_INTERVAL);
        stop.store(true, Ordering::Relaxed);
        h.join().unwrap();
        let versions: Vec<u32> = got.lock().unwrap().iter().map(|c| c.version).collect();
        assert_eq!(versions, vec![1, 2, 3, 4]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stream_ids_grow_with_time() {
        let a = stream_id(UNIX_EPOCH + Duration::from_secs(1_800_000_000));
        let b = stream_id(UNIX_EPOCH + Duration::from_secs(1_800_000_001));
        assert!(b > a);
        assert_eq!(stream_id(UNIX_EPOCH), 1);
    }

    #[test]
    fn next_stream_ids_pass_earlier_runs_even_with_the_clock_behind() {
        let runs = temp("ids");
        let now = UNIX_EPOCH + Duration::from_secs(1_800_000_000);
        assert_eq!(next_stream_id(&runs, now), 1_800_000_000, "no runs yet: the clock");
        std::fs::create_dir_all(runs.join(run_folder_name(1_800_000_000, 41))).unwrap();
        // A restart within the same second, then one with the clock an hour behind.
        assert_eq!(next_stream_id(&runs, now), 1_800_000_001);
        std::fs::create_dir_all(runs.join(run_folder_name(1_800_000_001, 42))).unwrap();
        assert_eq!(next_stream_id(&runs, now - Duration::from_secs(3600)), 1_800_000_002);
        // Other folders do not count.
        std::fs::create_dir_all(runs.join("live-notes")).unwrap();
        std::fs::create_dir_all(runs.join("live-9999999999-1")).unwrap();
        assert_eq!(next_stream_id(&runs, now), 1_800_000_002);
        assert_eq!(run_folder_stream("live-7-12"), Some(7));
        assert_eq!(run_folder_stream("live-7"), None);
        let _ = std::fs::remove_dir_all(&runs);
    }

    #[test]
    fn pruning_keeps_the_newest_runs_only() {
        let runs = temp("prune");
        for id in [5u32, 3, 9, 7] {
            std::fs::create_dir_all(runs.join(run_folder_name(id, 1)).join("live")).unwrap();
        }
        std::fs::create_dir_all(runs.join("keep-me")).unwrap();
        assert_eq!(prune_runs(&runs, 2), 2);
        let mut left: Vec<String> =
            std::fs::read_dir(&runs).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        left.sort();
        assert_eq!(left, vec!["keep-me", "live-7-1", "live-9-1"]);
        assert_eq!(prune_runs(&runs, 2), 0);
        assert_eq!(prune_runs(&runs.join("missing"), 0), 0);
        let _ = std::fs::remove_dir_all(&runs);
    }

    #[test]
    fn sequences_of_a_later_run_are_above_every_sequence_of_an_earlier_one() {
        let (a, b) = (1_800_000_000u32, 1_800_000_001u32);
        assert!(sequence_base(b) > sequence_base(a) + u64::from(u32::MAX));
        assert_eq!(sequence_base(a) >> 32, u64::from(a));
    }

    /// The real theater30 stable run (decision 0052: 14 events, 2,881,929 points at the end).
    /// `SKYLENS_THEATER30_STREAM=<...>/theater30-stable/live/stream cargo test -p skylens-model -- --ignored`
    #[test]
    #[ignore = "needs SKYLENS_THEATER30_STREAM (a cumulus3d v4 live/stream directory)"]
    fn follows_the_real_theater30_stream() {
        let Some(dir) = std::env::var_os("SKYLENS_THEATER30_STREAM") else { return };
        let mut f = Follower::new(PathBuf::from(dir), 1, ANCHOR);
        let chunks = f.poll().unwrap();
        assert!(!chunks.is_empty());
        let mut fold = Fold::new();
        let mut bytes = 0usize;
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.version as usize, i + 1);
            fold.apply(&c.payload).unwrap();
            bytes += c.payload.len() - HEADER_LEN;
        }
        eprintln!("{} events, {} points, {:.1} MB of packs", chunks.len(), fold.points(), bytes as f64 / 1e6);
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(stream_dir_of(&f).join("manifest.json")).unwrap()).unwrap();
        let last = manifest["events"].as_array().unwrap().last().unwrap();
        assert_eq!(fold.points() as u64, last["points"].as_u64().unwrap());
        assert_eq!(fold.capacity() as u64, manifest["capacity"].as_u64().unwrap());
    }

    fn stream_dir_of(f: &Follower) -> &Path {
        &f.dir
    }
}
