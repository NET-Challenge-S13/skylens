//! Proto `FrameSet` -> cumulus3d `LiveFrameSet`, and assembly of sets whose
//! cameras arrive in separate pushes.
//!
//! The ENU origin. cumulus3d fixes the origin at the *first* GPS record of the
//! session (`fixed_enu_origin`). Every set therefore starts with a record named
//! [`ANCHOR_RECORD`] at the engine anchor. No image has that name, so it never
//! takes part in the alignment itself; it only pins the origin, and every zone
//! and chunk comes out in ENU metres about the anchor.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use anyhow::{Context, bail};
use cumulus3d_cli::compose::{EncodedFormat, FrameKey as LiveKey, LiveFrameSet};
use skylens_shared::geo::GeoAnchor;
use skylens_shared::proto::model::v1::{FrameKey, FrameSet, ImageFormat, LiveImage};

use super::fusion::FrameSource;

/// Name of the GPS record that pins the ENU origin at the anchor.
pub const ANCHOR_RECORD: &str = ".skylens-anchor";
/// Incomplete sets kept while waiting for their other cameras.
pub const MAX_PARTIAL_SETS: usize = 64;
/// Completed keys remembered so a late partial re-send is recognised.
const MAX_DONE_KEYS: usize = 4096;

type KeyId = (String, u64, u32);

fn key_id(k: &FrameKey) -> KeyId {
    (k.source_id.clone(), k.sequence, k.revision)
}

/// Sniffs JPEG/PNG when the producer left the format unspecified.
fn sniff(data: &[u8]) -> Option<EncodedFormat> {
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some(EncodedFormat::Jpeg)
    } else if data.starts_with(&[0x89, b'P', b'N', b'G']) {
        Some(EncodedFormat::Png)
    } else {
        None
    }
}

/// Converts a complete proto set. Cameras must be among `cameras`.
pub fn to_live_set(set: &FrameSet, cameras: &[String], anchor: GeoAnchor) -> anyhow::Result<LiveFrameSet> {
    let key = set.key.as_ref().context("frame set without key")?;
    let mut live = LiveFrameSet::new(LiveKey::new(key.source_id.clone(), key.sequence, key.revision));
    if set.captured_at_ms > 0 {
        live = live.at(SystemTime::UNIX_EPOCH + Duration::from_millis(set.captured_at_ms as u64));
    }
    live = live.gps_fix(ANCHOR_RECORD, anchor.lat, anchor.lon, anchor.alt);
    for im in &set.images {
        if !cameras.iter().any(|c| *c == im.camera) {
            bail!("frame set {}#{}: unknown camera {:?} (expected one of {cameras:?})", key.source_id, key.sequence, im.camera);
        }
        let data = im.data.clone();
        live = match im.format() {
            ImageFormat::Jpeg => live.encoded(im.camera.clone(), data, EncodedFormat::Jpeg),
            ImageFormat::Png => live.encoded(im.camera.clone(), data, EncodedFormat::Png),
            ImageFormat::Rgb8 => {
                let need = im.width as usize * im.height as usize * 3;
                if im.width == 0 || im.height == 0 || data.len() != need {
                    bail!("camera {}: {} bytes for {}x{} RGB8", im.camera, data.len(), im.width, im.height);
                }
                live.rgb8(im.camera.clone(), im.width, im.height, data)
            }
            ImageFormat::Unspecified => match sniff(&data) {
                Some(f) => live.encoded(im.camera.clone(), data, f),
                None => bail!("camera {}: image format unspecified and not JPEG/PNG", im.camera),
            },
        };
        if let Some(g) = &im.gps {
            live = live.gps_fix(im.camera.clone(), g.lat, g.lon, g.alt);
        }
    }
    Ok(live)
}

/// Capture times remembered for detection fusion.
pub const MAX_CLOCK_ENTRIES: usize = 8192;

/// `source_id` as cumulus3d writes it into a spooled image name: ASCII letters,
/// digits, `-` and `_` kept, anything else `_` (cumulus3d `spool_name`).
pub fn spool_source(source_id: &str) -> String {
    source_id.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

/// `(source, sequence)` of a spooled image name `<spool>/<camera>/<source>_<sequence:010>.<ext>`.
fn parse_spool_name(name: &str) -> Option<(String, u64)> {
    let file = name.rsplit(['/', '\\']).next()?;
    let stem = file.rsplit_once('.').map_or(file, |(s, _)| s);
    let (src, seq) = stem.rsplit_once('_')?;
    Some((src.to_string(), seq.parse().ok()?))
}

#[derive(Default)]
struct ClockState {
    times: HashMap<(String, u64), i64>,
    order: VecDeque<(String, u64)>,
}

/// Capture time of every declared set, so a decoded frame (which only carries its stored image
/// name and camera) can tell fusion which drone took it and when. Recorded before the set is
/// composed, so the lookup never races the frame hook.
#[derive(Default)]
pub struct FrameClock {
    st: Mutex<ClockState>,
}

impl FrameClock {
    /// Remembers the capture time of `key`'s set (`captured_at_ms <= 0`: now).
    pub fn record(&self, key: &FrameKey, captured_at_ms: i64) {
        let t = if captured_at_ms > 0 {
            captured_at_ms
        } else {
            SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
        };
        let id = (spool_source(&key.source_id), key.sequence);
        let mut st = self.st.lock().unwrap_or_else(|p| p.into_inner());
        if st.times.insert(id.clone(), t).is_none() {
            st.order.push_back(id);
            while st.order.len() > MAX_CLOCK_ENTRIES {
                if let Some(old) = st.order.pop_front() {
                    st.times.remove(&old);
                }
            }
        }
    }

    /// Drone (`<source>/<camera>`, each camera is one drone of the formation) and capture time of
    /// a decoded frame. A name that is not a spooled one gives the camera alone and no time.
    pub fn source_of(&self, name: &str, camera: &str) -> FrameSource {
        let Some((src, seq)) = parse_spool_name(name) else {
            return FrameSource { drone: camera.to_string(), time_ms: None };
        };
        let st = self.st.lock().unwrap_or_else(|p| p.into_inner());
        let time_ms = st.times.get(&(src.clone(), seq)).copied();
        FrameSource { drone: format!("{src}/{camera}"), time_ms }
    }
}

/// What [`Assembler::offer`] made of a push.
#[derive(Debug)]
pub enum Assembled {
    /// Every camera is present: declare it.
    Complete(FrameSet),
    /// Still waiting for these cameras.
    Pending { key: FrameKey, missing: Vec<String> },
    /// A partial re-send of a set that was already completed.
    Repeat,
}

struct Partial {
    images: BTreeMap<String, LiveImage>,
    captured_at_ms: i64,
}

/// Merges per-camera pushes of the same key into complete sets.
pub struct Assembler {
    cameras: Vec<String>,
    partial: BTreeMap<KeyId, Partial>,
    arrival: VecDeque<KeyId>,
    done: BTreeSet<KeyId>,
    done_order: VecDeque<KeyId>,
}

impl Assembler {
    pub fn new(cameras: Vec<String>) -> Self {
        Self {
            cameras,
            partial: BTreeMap::new(),
            arrival: VecDeque::new(),
            done: BTreeSet::new(),
            done_order: VecDeque::new(),
        }
    }

    fn missing(&self, have: impl Fn(&str) -> bool) -> Vec<String> {
        self.cameras.iter().filter(|c| !have(c)).cloned().collect()
    }

    fn mark_done(&mut self, id: KeyId) {
        if self.done.insert(id.clone()) {
            self.done_order.push_back(id);
            while self.done_order.len() > MAX_DONE_KEYS {
                if let Some(old) = self.done_order.pop_front() {
                    self.done.remove(&old);
                }
            }
        }
    }

    pub fn offer(&mut self, set: FrameSet) -> anyhow::Result<Assembled> {
        let key = set.key.clone().context("frame set without key")?;
        for im in &set.images {
            if !self.cameras.contains(&im.camera) {
                bail!("frame set {}#{}: unknown camera {:?}", key.source_id, key.sequence, im.camera);
            }
        }
        let id = key_id(&key);
        let complete_alone = self.missing(|c| set.images.iter().any(|i| i.camera == c)).is_empty();
        if complete_alone {
            if self.partial.remove(&id).is_some() {
                self.arrival.retain(|k| *k != id);
            }
            self.mark_done(id);
            return Ok(Assembled::Complete(set));
        }
        if self.done.contains(&id) {
            return Ok(Assembled::Repeat);
        }
        if !self.partial.contains_key(&id) {
            self.arrival.push_back(id.clone());
        }
        let entry = self
            .partial
            .entry(id.clone())
            .or_insert_with(|| Partial { images: BTreeMap::new(), captured_at_ms: set.captured_at_ms });
        if entry.captured_at_ms <= 0 {
            entry.captured_at_ms = set.captured_at_ms;
        }
        for im in set.images {
            entry.images.insert(im.camera.clone(), im);
        }
        let missing = {
            let have = &entry.images;
            self.cameras.iter().filter(|c| !have.contains_key(*c)).cloned().collect::<Vec<_>>()
        };
        if missing.is_empty() {
            let p = self.partial.remove(&id).expect("present");
            self.arrival.retain(|k| *k != id);
            self.mark_done(id);
            let images = self.cameras.iter().filter_map(|c| p.images.get(c).cloned()).collect();
            return Ok(Assembled::Complete(FrameSet { key: Some(key), images, captured_at_ms: p.captured_at_ms }));
        }
        while self.arrival.len() > MAX_PARTIAL_SETS {
            if let Some(old) = self.arrival.pop_front() {
                self.partial.remove(&old);
            }
        }
        Ok(Assembled::Pending { key, missing })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cumulus3d_cli::compose::FramePayload;
    use skylens_shared::proto::model::v1::Gps;

    const ANCHOR: GeoAnchor = skylens_shared::geo::Gps { lat: 36.3685, lon: 127.3475, alt: 30.0 };

    fn cams() -> Vec<String> {
        ["left", "center", "right"].map(String::from).to_vec()
    }

    fn jpeg(camera: &str) -> LiveImage {
        LiveImage {
            camera: camera.into(),
            format: ImageFormat::Jpeg as i32,
            data: vec![0xFF, 0xD8, 0xFF, 0xE0],
            gps: Some(Gps { lat: 36.37, lon: 127.35, alt: 80.0 }),
            ..Default::default()
        }
    }

    fn set(seq: u64, images: Vec<LiveImage>) -> FrameSet {
        FrameSet {
            key: Some(FrameKey { source_id: "formation-1".into(), sequence: seq, revision: 0 }),
            images,
            captured_at_ms: 1_700_000_000_000,
        }
    }

    #[test]
    fn converts_key_payloads_and_gps_with_the_anchor_first() {
        let mut s = set(7, vec![jpeg("left"), jpeg("center")]);
        s.images.push(LiveImage {
            camera: "right".into(),
            format: ImageFormat::Rgb8 as i32,
            data: vec![0; 2 * 2 * 3],
            width: 2,
            height: 2,
            gps: None,
        });
        let live = to_live_set(&s, &cams(), ANCHOR).unwrap();
        assert_eq!(live.key, LiveKey::new("formation-1", 7, 0));
        assert_eq!(live.images.len(), 3);
        assert!(matches!(live.images[0].payload, FramePayload::Encoded { format: EncodedFormat::Jpeg, .. }));
        assert!(matches!(live.images[2].payload, FramePayload::Rgb8 { width: 2, height: 2, .. }));
        assert_eq!(live.gps[0].name, ANCHOR_RECORD);
        assert_eq!((live.gps[0].lat, live.gps[0].lon, live.gps[0].alt), (ANCHOR.lat, ANCHOR.lon, ANCHOR.alt));
        assert_eq!(live.gps.iter().filter(|g| g.name == "left").count(), 1);
        assert_eq!(live.gps.len(), 3);
        assert_eq!(live.timestamp, SystemTime::UNIX_EPOCH + Duration::from_millis(1_700_000_000_000));
    }

    #[test]
    fn rejects_unknown_cameras_and_bad_rgb() {
        assert!(to_live_set(&set(0, vec![jpeg("rear")]), &cams(), ANCHOR).is_err());
        let bad = LiveImage { camera: "left".into(), format: ImageFormat::Rgb8 as i32, data: vec![0; 5], width: 2, height: 2, gps: None };
        assert!(to_live_set(&set(0, vec![bad]), &cams(), ANCHOR).is_err());
        assert!(to_live_set(&FrameSet::default(), &cams(), ANCHOR).is_err());
    }

    #[test]
    fn unspecified_format_is_sniffed() {
        let mut im = jpeg("left");
        im.format = ImageFormat::Unspecified as i32;
        let live = to_live_set(&set(0, vec![im]), &cams(), ANCHOR).unwrap();
        assert!(matches!(live.images[0].payload, FramePayload::Encoded { format: EncodedFormat::Jpeg, .. }));
        let mut junk = jpeg("left");
        junk.format = ImageFormat::Unspecified as i32;
        junk.data = vec![1, 2, 3];
        assert!(to_live_set(&set(0, vec![junk]), &cams(), ANCHOR).is_err());
    }

    #[test]
    fn assembles_cameras_across_pushes() {
        let mut a = Assembler::new(cams());
        match a.offer(set(1, vec![jpeg("left")])).unwrap() {
            Assembled::Pending { missing, .. } => assert_eq!(missing, vec!["center", "right"]),
            other => panic!("{other:?}"),
        }
        assert!(matches!(a.offer(set(1, vec![jpeg("right")])).unwrap(), Assembled::Pending { .. }));
        match a.offer(set(1, vec![jpeg("center")])).unwrap() {
            Assembled::Complete(s) => {
                let order: Vec<_> = s.images.iter().map(|i| i.camera.as_str()).collect();
                assert_eq!(order, ["left", "center", "right"]);
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(a.offer(set(1, vec![jpeg("left")])).unwrap(), Assembled::Repeat));
        // A complete re-send still goes through, so the node can report it as a duplicate.
        let full = set(1, vec![jpeg("left"), jpeg("center"), jpeg("right")]);
        assert!(matches!(a.offer(full).unwrap(), Assembled::Complete(_)));
    }

    #[test]
    fn clock_resolves_spooled_names() {
        let c = FrameClock::default();
        c.record(&FrameKey { source_id: "formation 1".into(), sequence: 42, revision: 0 }, 1_700_000_000_500);
        let s = c.source_of("C:/work/live/spool/left/formation_1_0000000042.jpg", "left");
        assert_eq!(s, FrameSource { drone: "formation_1/left".into(), time_ms: Some(1_700_000_000_500) });
        let unknown = c.source_of("spool/right/formation_1_0000000043.jpg", "right");
        assert_eq!(unknown, FrameSource { drone: "formation_1/right".into(), time_ms: None });
        assert_eq!(c.source_of("frames/img.png", "left"), FrameSource { drone: "left".into(), time_ms: None });
    }

    #[test]
    fn partial_sets_are_bounded() {
        let mut a = Assembler::new(cams());
        for seq in 0..(MAX_PARTIAL_SETS as u64 + 10) {
            a.offer(set(seq, vec![jpeg("left")])).unwrap();
        }
        assert_eq!(a.partial.len(), MAX_PARTIAL_SETS);
        assert!(!a.partial.contains_key(&("formation-1".to_string(), 0, 0)));
    }
}
