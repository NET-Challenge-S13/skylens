//! Job inputs: `Source.uri`s decoded into per-position frame sets.
//!
//! - A `uri` is a local path (an optional `file://` prefix is stripped).
//!   - A directory: its `.jpg`/`.jpeg`/`.png` files in name order are the frames.
//!   - An image file: one frame.
//!   - Any other file is treated as a video and sampled at [`JOB_FPS`] with an
//!     `ffmpeg` subprocess (`SKYLENS_FFMPEG` overrides the binary).
//! - Source `i` is camera `i` of the engine's camera list (`left`, `center`,
//!   `right`), so one source per drone of the formation.
//! - Positions are the frame index; every camera contributes its frame at the
//!   same index, up to the shortest source, evenly thinned to
//!   [`MAX_JOB_POSITIONS`].
//! - GPS of a frame: the source's poses sampled at the same relative position
//!   (`index / (count - 1)`); without poses, the segment `track` interpolated the
//!   same way; otherwise none.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, bail, ensure};
use skylens_shared::geo::Gps;
use skylens_shared::proto::model::v1::{Gps as ProtoGps, Source};

/// Video sampling rate for jobs, frames per second.
pub const JOB_FPS: f64 = 2.0;
/// Upper bound of positions per job (the whole segment becomes one zone).
pub const MAX_JOB_POSITIONS: usize = 120;

const IMAGE_EXT: [&str; 3] = ["jpg", "jpeg", "png"];

fn is_image(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| IMAGE_EXT.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

fn local_path(uri: &str) -> PathBuf {
    PathBuf::from(uri.strip_prefix("file://").unwrap_or(uri))
}

fn ffmpeg_bin() -> String {
    std::env::var("SKYLENS_FFMPEG").unwrap_or_else(|_| "ffmpeg".to_string())
}

fn list_images(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && is_image(p))
        .collect();
    v.sort();
    Ok(v)
}

/// Frames of one source as image files; videos are extracted into `scratch`.
fn source_frames(uri: &str, scratch: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let path = local_path(uri);
    ensure!(path.exists(), "source not found: {}", path.display());
    if path.is_dir() {
        return list_images(&path);
    }
    if is_image(&path) {
        return Ok(vec![path]);
    }
    std::fs::create_dir_all(scratch)?;
    let status = Command::new(ffmpeg_bin())
        .arg("-nostdin")
        .args(["-loglevel", "error", "-i"])
        .arg(&path)
        .args(["-vf", &format!("fps={JOB_FPS}"), "-q:v", "2"])
        .arg(scratch.join("%06d.jpg"))
        .status()
        .with_context(|| format!("run {} (set SKYLENS_FFMPEG)", ffmpeg_bin()))?;
    ensure!(status.success(), "ffmpeg failed on {}: {status}", path.display());
    list_images(scratch)
}

/// `count` indices spread evenly over `0..n` (all of them when `n <= count`).
pub fn pick_evenly(n: usize, count: usize) -> Vec<usize> {
    if n <= count {
        return (0..n).collect();
    }
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![0];
    }
    (0..count).map(|i| ((i as f64) * (n - 1) as f64 / (count - 1) as f64).round() as usize).collect()
}

/// GPS at relative position `frac` (0..=1): a pose sample, else the track
/// interpolated, else none.
pub fn gps_at(frac: f64, poses: &[Gps], track: &[Gps]) -> Option<Gps> {
    let frac = frac.clamp(0.0, 1.0);
    if !poses.is_empty() {
        return Some(poses[(frac * (poses.len() - 1) as f64).round() as usize]);
    }
    match track.len() {
        0 => None,
        1 => Some(track[0]),
        n => {
            let x = frac * (n - 1) as f64;
            let i = (x.floor() as usize).min(n - 2);
            let t = x - i as f64;
            let (a, b) = (track[i], track[i + 1]);
            Some(Gps { lat: a.lat + (b.lat - a.lat) * t, lon: a.lon + (b.lon - a.lon) * t, alt: a.alt + (b.alt - a.alt) * t })
        }
    }
}

/// One job's input, staged under the job's image root.
#[derive(Debug)]
pub struct JobFrames {
    pub cameras: Vec<String>,
    /// `[position][camera]` image name relative to the image root.
    pub names: Vec<Vec<String>>,
    /// `[position][camera]` GPS fix.
    pub gps: Vec<Vec<Option<Gps>>>,
}

/// Decodes `sources` and stages their frames under `root/<camera>/`.
pub fn prepare(root: &Path, sources: &[Source], track: &[ProtoGps], cameras: &[String]) -> anyhow::Result<JobFrames> {
    ensure!(!sources.is_empty(), "job has no sources");
    let cams: Vec<String> =
        (0..sources.len()).map(|i| cameras.get(i).cloned().unwrap_or_else(|| format!("cam{i}"))).collect();
    let track: Vec<Gps> = track.iter().cloned().map(Gps::from).collect();
    let mut frames = Vec::with_capacity(sources.len());
    for (src, cam) in sources.iter().zip(&cams) {
        let scratch = root.join(".extract").join(cam);
        let f = source_frames(&src.uri, &scratch)?;
        if f.is_empty() {
            bail!("source {} has no frames", src.uri);
        }
        frames.push(f);
    }
    let n = frames.iter().map(Vec::len).min().unwrap_or(0);
    let picks = pick_evenly(n, MAX_JOB_POSITIONS);
    let mut names = Vec::with_capacity(picks.len());
    let mut gps = Vec::with_capacity(picks.len());
    for (p, &idx) in picks.iter().enumerate() {
        let mut row_names = Vec::with_capacity(cams.len());
        let mut row_gps = Vec::with_capacity(cams.len());
        for (i, cam) in cams.iter().enumerate() {
            let src = &frames[i][idx];
            let ext = src.extension().and_then(|e| e.to_str()).unwrap_or("jpg").to_ascii_lowercase();
            let rel = format!("{cam}/{p:06}.{ext}");
            let dst = root.join(&rel);
            std::fs::create_dir_all(dst.parent().expect("has parent"))?;
            if std::fs::hard_link(src, &dst).is_err() {
                std::fs::copy(src, &dst).with_context(|| format!("copy {} -> {}", src.display(), dst.display()))?;
            }
            row_names.push(rel);
            let count = frames[i].len();
            let frac = if count > 1 { idx as f64 / (count - 1) as f64 } else { 0.0 };
            let poses: Vec<Gps> = sources[i].poses.iter().filter_map(|t| t.gps.clone().map(Gps::from)).collect();
            row_gps.push(gps_at(frac, &poses, &track));
        }
        names.push(row_names);
        gps.push(row_gps);
    }
    Ok(JobFrames { cameras: cams, names, gps })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(lat: f64) -> Gps {
        Gps { lat, lon: 127.0, alt: 50.0 }
    }

    #[test]
    fn picks_spread_over_the_range() {
        assert_eq!(pick_evenly(3, 10), vec![0, 1, 2]);
        assert_eq!(pick_evenly(10, 4), vec![0, 3, 6, 9]);
        assert_eq!(pick_evenly(10, 1), vec![0]);
        assert!(pick_evenly(5, 0).is_empty());
    }

    #[test]
    fn gps_prefers_poses_then_track() {
        let poses = [g(1.0), g(2.0), g(3.0)];
        assert_eq!(gps_at(0.5, &poses, &[]).unwrap().lat, 2.0);
        assert_eq!(gps_at(1.0, &poses, &[g(9.0)]).unwrap().lat, 3.0);
        let track = [g(0.0), g(10.0)];
        assert!((gps_at(0.25, &[], &track).unwrap().lat - 2.5).abs() < 1e-12);
        assert_eq!(gps_at(0.7, &[], &[g(4.0)]).unwrap().lat, 4.0);
        assert!(gps_at(0.3, &[], &[]).is_none());
    }

    #[test]
    fn stages_image_folders_per_camera() {
        let base = std::env::temp_dir().join(format!("skylens-sources-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let mk = |name: &str, n: usize| {
            let d = base.join("in").join(name);
            std::fs::create_dir_all(&d).unwrap();
            for i in 0..n {
                std::fs::write(d.join(format!("f{i:03}.jpg")), [0xFF, 0xD8, 0xFF]).unwrap();
            }
            std::fs::write(d.join("notes.txt"), b"x").unwrap();
            d.to_string_lossy().into_owned()
        };
        let sources = vec![
            Source { uri: mk("a", 4), poses: vec![] },
            Source { uri: format!("file://{}", mk("b", 3)), poses: vec![] },
        ];
        let track = [ProtoGps { lat: 36.0, lon: 127.0, alt: 40.0 }, ProtoGps { lat: 36.1, lon: 127.0, alt: 40.0 }];
        let cams = ["left", "center", "right"].map(String::from);
        let root = base.join("job");
        let jf = prepare(&root, &sources, &track, &cams).unwrap();
        assert_eq!(jf.cameras, vec!["left", "center"]);
        assert_eq!(jf.names.len(), 3);
        assert_eq!(jf.names[2], vec!["left/000002.jpg", "center/000002.jpg"]);
        assert!(root.join("center/000002.jpg").is_file());
        // Source b: frame 2 of 3 is the end of the track.
        assert!((jf.gps[2][1].unwrap().lat - 36.1).abs() < 1e-12);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn missing_source_is_an_error() {
        let src = vec![Source { uri: "/definitely/not/here.mp4".into(), poses: vec![] }];
        assert!(prepare(&std::env::temp_dir().join("skylens-none"), &src, &[], &[]).is_err());
    }
}
