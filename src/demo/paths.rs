//! Where things live on disk, and the small templates built from them: the
//! formation table, footage files, SRT publish URLs, the local MediaMTX config,
//! executable discovery.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use skylens_shared::proto::model::v1::DroneStation;

/// One aircraft of the demo formation and the footage it "films".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Drone {
    pub id: u32,
    pub station: DroneStation,
    /// Lower-case station name, also the footage suffix and the model's camera name.
    pub name: &'static str,
}

/// Drone 1 = left, 2 = center, 3 = right (the setup the lead verified by hand).
pub const FORMATION: [Drone; 3] = [
    Drone { id: 1, station: DroneStation::Left, name: "left" },
    Drone { id: 2, station: DroneStation::Center, name: "center" },
    Drone { id: 3, station: DroneStation::Right, name: "right" },
];

/// `src/demo` -> repo root.
pub fn repo_root() -> PathBuf {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    here.parent().and_then(Path::parent).unwrap_or(here).to_path_buf()
}

/// Cargo's target directory: `CARGO_TARGET_DIR` when set, else `<repo>/target`.
pub fn target_dir(repo: &Path, cargo_target_dir: Option<&str>) -> PathBuf {
    match cargo_target_dir.map(str::trim).filter(|s| !s.is_empty()) {
        Some(dir) => {
            let p = PathBuf::from(dir);
            if p.is_absolute() { p } else { repo.join(p) }
        }
        None => repo.join("target"),
    }
}

/// `name` plus the platform's executable suffix (`.exe` on Windows).
pub fn exe(name: &str) -> String {
    format!("{name}{}", std::env::consts::EXE_SUFFIX)
}

/// `<repo>/data`: where the footage zip is extracted (git-ignored).
pub fn data_dir(repo: &Path) -> PathBuf {
    repo.join("data")
}

/// `<repo>/data/demo`: the recorded footage, its flight record and camera estimate.
pub fn demo_dir(repo: &Path) -> PathBuf {
    data_dir(repo).join("demo")
}

/// `data/demo/track.json`: the flight record of every view.
pub fn track_file(repo: &Path) -> PathBuf {
    demo_dir(repo).join("track.json")
}

/// `data/demo/camera.json`: the camera estimate (optional).
pub fn camera_file(repo: &Path) -> PathBuf {
    demo_dir(repo).join("camera.json")
}

/// The clip a drone publishes: `data/demo/<station>.mp4` (the recorded flight), or with
/// `--synthetic` the old `res/static/video/demo_<station>.mp4`.
pub fn footage(repo: &Path, drone: &Drone, synthetic: bool) -> PathBuf {
    if synthetic {
        repo.join("res").join("static").join("video").join(format!("demo_{}.mp4", drone.name))
    } else {
        demo_dir(repo).join(format!("{}.mp4", drone.name))
    }
}

/// What the recorded-flight demo needs on disk; missing any of it triggers the download.
pub fn recorded_files(repo: &Path) -> Vec<PathBuf> {
    FORMATION.iter().map(|d| footage(repo, d, false)).chain([track_file(repo)]).collect()
}

/// `<target>/demo/planned-route.json`, written for the core.
pub fn planned_route_file(run_dir: &Path) -> PathBuf {
    run_dir.join("planned-route.json")
}

/// The MediaMTX path a drone publishes to and the core reads (`drone{id}`).
pub fn stream_path(drone: &Drone) -> String {
    format!("drone{}", drone.id)
}

/// SRT publish URL for one drone's footage.
/// Default SRT payload, six MPEG-TS packets (6 x 188).
///
/// The obvious value is 1316 (7 x 188), which is what fills a 1500-byte frame
/// once SRT, UDP and IP headers are on. It fails the moment the footage crosses
/// a VPN: the tunnel adds its own header, the data packets exceed the path MTU
/// and are dropped, while the small handshake packets still get through. The
/// stream then looks connected and never publishes, and MediaMTX answers every
/// reader with "no stream is available". One packet less leaves room for the
/// tunnel and costs about 15% more packets, which is nothing for three feeds.
pub const SRT_PKT_SIZE: u32 = 1128;

pub fn srt_publish_url(srt_addr: &str, drone: &Drone, pkt_size: u32) -> String {
    format!("srt://{srt_addr}?streamid=publish:{}&pkt_size={pkt_size}", stream_path(drone))
}

/// TCP publish URL for one drone's footage (`--feed-tcp`): the base port plus the drone index.
pub fn tcp_publish_url(base: &str, drone: &Drone) -> String {
    match base.rsplit_once(':') {
        Some((host, port)) => format!("tcp://{host}:{}", port.parse::<u16>().unwrap_or(0) + (drone.id as u16 - 1)),
        None => format!("tcp://{base}"),
    }
}

/// ffmpeg arguments that publish a file to MediaMTX at real time, without re-encoding unless `kbps` is set;
/// in a loop unless `looped` is false (then it plays once).
/// `limit` stops the feed after that many seconds of the clip (`-t`, before the input).
pub fn publisher_args(file: &Path, url: &str, looped: bool, limit: Option<f64>, kbps: u32) -> Vec<String> {
    let codec: Vec<String> = if kbps == 0 {
        vec!["-c:v".into(), "copy".into()]
    } else {
        // Closed GOP of one second so a reader can join anywhere; CBR-ish so a feed never bursts.
        ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-g", "30", "-pix_fmt", "yuv420p"]
            .iter()
            .map(|s| s.to_string())
            .chain(["-b:v".to_string(), format!("{kbps}k"), "-maxrate".to_string(), format!("{kbps}k"), "-bufsize".to_string(), format!("{}k", kbps * 2)])
            .collect()
    };
    let loop_args: &[&str] = if looped { &["-stream_loop", "-1"] } else { &[] };
    let limit_args: Vec<String> = limit.map(|d| vec!["-t".to_string(), format!("{d:.3}")]).unwrap_or_default();
    ["-hide_banner", "-nostats", "-loglevel", "warning", "-re"]
    .iter()
    .chain(loop_args)
    .map(|s| s.to_string())
    .chain(limit_args)
    .chain(["-i".to_string()])
    .chain([file.display().to_string()])
    .chain(["-an".to_string()])
    .chain(codec)
    .chain(["-f", "mpegts"].iter().map(|s| s.to_string()))
    .chain([url.to_string()])
    .collect()
}

/// Turn the Daejeon deploy config into one that works on a single dev machine:
///
/// - `paths:` accepts any name (`all_others`). The deploy config only defines
///   `drone`; the demo publishes `drone1..3`, which is what the core reads.
/// - ICE candidates from every interface instead of the Daejeon VM's `ens3`,
///   which does not exist here.
///
/// Everything else (ports, disabled protocols) stays as deployed.
pub fn local_mediamtx_config(deploy: &str) -> String {
    let mut out = String::with_capacity(deploy.len());
    for line in deploy.lines() {
        if line.starts_with("paths:") {
            break;
        }
        if line.trim_start().starts_with("webrtcIPsFromInterfacesList:") {
            out.push_str("webrtcIPsFromInterfacesList: []   # demo: every local interface\n");
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("paths:\n  all_others:\n");
    out
}

/// Of the libtorch `lib` directories cargo left under `build/torch-sys-*`, the
/// most recently modified one (the one the last build linked against).
pub fn newest(candidates: Vec<(PathBuf, SystemTime)>) -> Option<PathBuf> {
    candidates.into_iter().max_by_key(|(_, t)| *t).map(|(p, _)| p)
}

/// `target/<profile>/build/torch-sys-*/out/libtorch/libtorch/lib`, newest first.
pub fn libtorch_lib_dir(profile_dir: &Path) -> Option<PathBuf> {
    let build = profile_dir.join("build");
    let entries = std::fs::read_dir(&build).ok()?;
    let mut found = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("torch-sys-") {
            continue;
        }
        let lib = entry.path().join("out").join("libtorch").join("libtorch").join("lib");
        if let Ok(meta) = std::fs::metadata(&lib)
            && meta.is_dir()
        {
            let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            found.push((lib, modified));
        }
    }
    newest(found)
}

/// Find an executable on `PATH` (with the platform suffix).
pub fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    which_in(name, std::env::split_paths(&path))
}

pub fn which_in(name: &str, dirs: impl IntoIterator<Item = PathBuf>) -> Option<PathBuf> {
    let file = exe(name);
    dirs.into_iter().map(|d| d.join(&file)).find(|p| p.is_file())
}

/// The user's home directory (`USERPROFILE` on Windows, `HOME` elsewhere).
pub fn home_dir() -> Option<PathBuf> {
    let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    std::env::var_os(var).map(PathBuf::from).filter(|p| !p.as_os_str().is_empty())
}

/// Where MediaMTX is looked for, in order: `--mediamtx`, `~/tools/mediamtx/`, `PATH`.
pub fn mediamtx_candidates(explicit: Option<&Path>, home: Option<&Path>) -> Vec<PathBuf> {
    if let Some(p) = explicit {
        return vec![p.to_path_buf()];
    }
    let mut out = Vec::new();
    if let Some(h) = home {
        out.push(h.join("tools").join("mediamtx").join(exe("mediamtx")));
    }
    out
}

/// Where the Go core binary is looked for, in order.
pub fn core_candidates(explicit: Option<&Path>, repo: &Path, target: &Path) -> Vec<PathBuf> {
    if let Some(p) = explicit {
        return vec![p.to_path_buf()];
    }
    let core = repo.join("src").join("skylens_core");
    vec![
        target.join(exe("skylens-core")),
        core.join("target").join(exe("skylens-core")),
        core.join(exe("skylens-core")),
        core.join(exe("skylens_core")),
    ]
}

pub fn first_file(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|p| p.is_file()).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const DEPLOY: &str = "\
logLevel: info
rtsp: no
srt: yes
srtAddress: :10890
webrtcAddress: :10889
webrtcIPsFromInterfaces: yes
webrtcIPsFromInterfacesList: [ens3]
webrtcAdditionalHosts: []

pathDefaults:
  source: publisher

paths:
  drone:                        # comment
";

    #[test]
    fn formation_is_left_center_right_one_to_three() {
        let ids: Vec<_> = FORMATION.iter().map(|d| (d.id, d.name)).collect();
        assert_eq!(ids, vec![(1, "left"), (2, "center"), (3, "right")]);
        assert_eq!(FORMATION[1].station, DroneStation::Center);
    }

    #[test]
    fn footage_and_stream_templates() {
        let repo = Path::new("/repo");
        assert_eq!(footage(repo, &FORMATION[0], false), Path::new("/repo/data/demo/left.mp4"));
        assert_eq!(footage(repo, &FORMATION[1], false), Path::new("/repo/data/demo/center.mp4"));
        assert_eq!(footage(repo, &FORMATION[0], true), Path::new("/repo/res/static/video/demo_left.mp4"));
        assert_eq!(footage(repo, &FORMATION[2], true), Path::new("/repo/res/static/video/demo_right.mp4"));
        assert_eq!(track_file(repo), Path::new("/repo/data/demo/track.json"));
        assert_eq!(camera_file(repo), Path::new("/repo/data/demo/camera.json"));
        assert_eq!(
            recorded_files(repo),
            vec![
                PathBuf::from("/repo/data/demo/left.mp4"),
                PathBuf::from("/repo/data/demo/center.mp4"),
                PathBuf::from("/repo/data/demo/right.mp4"),
                PathBuf::from("/repo/data/demo/track.json"),
            ]
        );
        assert_eq!(planned_route_file(Path::new("/repo/target/demo")), Path::new("/repo/target/demo/planned-route.json"));
        assert_eq!(stream_path(&FORMATION[1]), "drone2");
        assert_eq!(
            srt_publish_url("127.0.0.1:10890", &FORMATION[2], SRT_PKT_SIZE),
            "srt://127.0.0.1:10890?streamid=publish:drone3&pkt_size=1128"
        );
    }

    #[test]
    fn publisher_copies_video_in_a_real_time_loop() {
        let args = publisher_args(Path::new("demo_left.mp4"), &srt_publish_url("127.0.0.1:10890", &FORMATION[0], SRT_PKT_SIZE), true, None, 0);
        let joined = args.join(" ");
        assert!(joined.contains("-re -stream_loop -1 -i demo_left.mp4"));
        assert!(joined.contains("-an -c:v copy -f mpegts"));
        assert_eq!(args.last().unwrap(), "srt://127.0.0.1:10890?streamid=publish:drone1&pkt_size=1128");
    }

    #[test]
    fn publisher_can_cap_the_bitrate() {
        let joined = publisher_args(Path::new("a.mp4"), "srt://h:1", true, None, 2500).join(" ");
        assert!(joined.contains("-c:v libx264") && joined.contains("-b:v 2500k -maxrate 2500k -bufsize 5000k"));
        assert!(!joined.contains("copy"));
    }

    #[test]
    fn publisher_plays_once_without_looping() {
        let args = publisher_args(Path::new("demo_left.mp4"), "tcp://127.0.0.1:19101", false, None, 0);
        assert!(!args.contains(&"-t".to_string()));
        let joined = args.join(" ");
        assert!(joined.contains("-re -i demo_left.mp4"));
        assert!(!joined.contains("-stream_loop"));
    }

    #[test]
    fn publisher_limit_stops_the_feed() {
        let args = publisher_args(Path::new("left.mp4"), "tcp://127.0.0.1:19101", false, Some(52.3), 0);
        assert!(args.join(" ").contains("-re -t 52.300 -i left.mp4"));
    }

    #[test]
    fn tcp_feed_uses_one_port_per_drone() {
        assert_eq!(tcp_publish_url("127.0.0.1:19101", &FORMATION[2]), "tcp://127.0.0.1:19103");
    }

    #[test]
    fn mediamtx_config_accepts_any_path_and_every_interface() {
        let cfg = local_mediamtx_config(DEPLOY);
        assert!(cfg.ends_with("paths:\n  all_others:\n"));
        assert!(!cfg.contains("drone:"));
        assert!(!cfg.contains("ens3"));
        assert!(cfg.contains("webrtcIPsFromInterfacesList: []"));
        // untouched settings survive
        for kept in ["srtAddress: :10890", "webrtcAddress: :10889", "rtsp: no", "pathDefaults:", "  source: publisher"] {
            assert!(cfg.contains(kept), "{kept} missing");
        }
        assert_eq!(cfg.matches("paths:").count(), 1);
    }

    #[test]
    fn target_dir_honours_cargo_target_dir() {
        let repo = Path::new("/repo");
        assert_eq!(target_dir(repo, None), Path::new("/repo/target"));
        assert_eq!(target_dir(repo, Some("  ")), Path::new("/repo/target"));
        assert_eq!(target_dir(repo, Some("out")), Path::new("/repo/out"));
        let abs = std::env::temp_dir().join("tgt");
        assert_eq!(target_dir(repo, Some(abs.to_str().unwrap())), abs);
    }

    #[test]
    fn newest_libtorch_wins() {
        let t0 = SystemTime::UNIX_EPOCH;
        let picked = newest(vec![
            (PathBuf::from("a"), t0 + Duration::from_secs(10)),
            (PathBuf::from("b"), t0 + Duration::from_secs(30)),
            (PathBuf::from("c"), t0 + Duration::from_secs(20)),
        ]);
        assert_eq!(picked, Some(PathBuf::from("b")));
        assert_eq!(newest(vec![]), None);
    }

    #[test]
    fn libtorch_dir_is_found_under_build() {
        let base = std::env::temp_dir().join(format!("skylens-demo-libtorch-{}", std::process::id()));
        let lib = base.join("build").join("torch-sys-abc").join("out").join("libtorch").join("libtorch").join("lib");
        std::fs::create_dir_all(&lib).unwrap();
        std::fs::create_dir_all(base.join("build").join("other-crate")).unwrap();
        assert_eq!(libtorch_lib_dir(&base), Some(lib));
        assert_eq!(libtorch_lib_dir(&base.join("missing")), None);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn which_in_finds_executables_with_suffix() {
        let dir = std::env::temp_dir().join(format!("skylens-demo-which-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(exe("fakeffmpeg")), b"").unwrap();
        assert_eq!(which_in("fakeffmpeg", [PathBuf::from("/nope"), dir.clone()]), Some(dir.join(exe("fakeffmpeg"))));
        assert_eq!(which_in("absent", [dir.clone()]), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn candidate_orders() {
        let repo = Path::new("/repo");
        let target = Path::new("/repo/target");
        assert_eq!(core_candidates(Some(Path::new("x")), repo, target), vec![PathBuf::from("x")]);
        let c = core_candidates(None, repo, target);
        assert_eq!(c[0], target.join(exe("skylens-core")));
        assert_eq!(c.len(), 4);
        let m = mediamtx_candidates(None, Some(Path::new("/home/u")));
        assert_eq!(m, vec![Path::new("/home/u/tools/mediamtx").join(exe("mediamtx"))]);
        assert!(mediamtx_candidates(None, None).is_empty());
    }
}
