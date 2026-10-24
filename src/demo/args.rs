//! Command line of `skylens-demo`.

use std::path::PathBuf;

/// Ports the launcher does not let you move: the model server's REST and gRPC
/// listeners and MediaMTX's WHEP signalling (the screens and the core assume them).
pub const MODEL_REST_PORT: u16 = 8100;
pub const MODEL_GRPC_PORT: u16 = 8101;
pub const WHEP_PORT: u16 = 10889;
pub const SRT_PORT: u16 = 10890;
pub const WEBRTC_UDP_PORT: u16 = 10189;

// ---- the deployment, for --online ------------------------------------------------------------
// These are the KOREN setup as it stands, not something the demo can discover: the bridge is a
// fixed address on the VPN and the ports are the ones skylens_gateway installs. docs/DEPLOYMENT.md
// is where they are written down and reasoned about.
/// The bridge (the Raspberry Pi) on the VPN.
pub const ONLINE_HOST: &str = "10.9.8.37";
/// The core, reached through the bridge's control tower relay. Not port 80: that one serves the
/// status board and does not carry `/control`, so the demo would have nowhere to assign a route.
pub const ONLINE_CORE_PORT: u16 = 8090;
/// The control tower over TLS on the bridge. Reachable from the VPN and the lab network only.
pub const ONLINE_TOWER_TLS_PORT: u16 = 8091;
/// The status board as everyone else reaches it: the router forwards 8443 to the bridge's 443.
pub const ONLINE_BOARD_URL: &str = "https://175.126.98.44:8443/";

#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// Core HTTP/WebSocket port (production uses 8080).
    pub core_port: u16,
    /// Board server port, playing the gateway's part (the legacy board port).
    pub board_port: u16,
    /// Run without the model server (core only, to look at the screens).
    pub no_model: bool,
    /// gRPC address of a model server running elsewhere (e.g. behind an SSH tunnel). The
    /// launcher then starts no model server of its own; it waits for the REST `/health`
    /// of the same host on the model REST port.
    pub model_addr: Option<String>,
    /// Run against the deployment instead of this machine: the bridge host to reach it by
    /// (`--online` for the usual one, `--online=HOST` for another). It fills in `remote_core` and
    /// `srt_addr` and turns on `auto` and `full_footage`, so one flag is the whole live setup.
    pub online: Option<String>,
    /// A core that already runs elsewhere (host:port, e.g. an SSH tunnel to the Daejeon VM). The
    /// launcher then starts no MediaMTX, core, model server or board; it only publishes the
    /// footage and flies the drones against that core.
    pub remote_core: Option<String>,
    /// SRT ingest (host:port) the footage is published to; default the local MediaMTX.
    pub srt_addr: Option<String>,
    pub srt_pkt_size: u32,
    /// Re-encode each feed to this many kbit/s (0 copies the clip as it is). The clips are
    /// ~10 Mbit/s each; over the VPN that saturates the uplink and starves the drones' sockets,
    /// so `--online` defaults to 2500.
    pub video_kbps: Option<u32>,
    /// Publish the footage as MPEG-TS over plain TCP instead (host:port, drone N goes to
    /// port+N-1), for links that block SRT's UDP; something on the far side feeds MediaMTX.
    pub feed_tcp: Option<String>,
    /// Frames per second the core extracts per drone.
    pub fps: f64,
    /// Width of extracted frames in px (0 keeps the source size).
    pub frame_width: u32,
    /// MediaMTX executable; searched for when absent.
    pub mediamtx: Option<PathBuf>,
    /// Also launch the native control tower and status board.
    pub native: bool,
    /// Assign the route ourselves so the scenario runs unattended.
    pub auto: bool,
    /// Fly the operator's waypoints instead of the footage track.
    pub follow_route: bool,
    /// Play the footage once instead of looping it; the drones hold their last fix afterwards.
    pub no_loop: bool,
    /// Play every pass of the footage (default: only the first pass).
    pub full_footage: bool,
    /// Seconds to wait before the footage starts playing (the drones follow the clip, so they
    /// wait too); lets the drones come online before the first frame is published.
    pub publish_delay: f64,
    /// SkyLensNet checkpoint for the model server.
    pub checkpoint: Option<PathBuf>,
    /// Go core executable; searched for when absent.
    pub core_bin: Option<PathBuf>,
    /// Cargo profile directory of the native screens (`release` or `debug`).
    pub profile: Option<String>,
    /// The old clips (`res/static/video/demo_*.mp4`) and the synthetic track
    /// instead of the recorded flight.
    pub synthetic: bool,
    /// Where the recorded footage zip is downloaded from.
    pub footage_url: String,
    /// Never download the footage; fail with the manual steps instead.
    pub no_fetch: bool,
    /// Ground elevation at the takeoff point, metres above sea level (skips the DEM lookup).
    pub ground_alt: Option<f64>,
    /// `SKYLENS_CORE_SITE`, passed to the core verbatim (default: centred on the footage).
    pub site: Option<String>,
    pub help: bool,
}

/// The recorded footage, published as a GitHub release asset.
pub const FOOTAGE_URL: &str =
    "https://github.com/NET-Challenge-S13/skylens/releases/download/demo-footage-v1/skylens-demo-footage.zip";

impl Default for Options {
    fn default() -> Self {
        Options {
            online: None,
            core_port: 8080,
            board_port: 8090,
            no_model: false,
            model_addr: None,
            remote_core: None,
            srt_addr: None,
            srt_pkt_size: crate::paths::SRT_PKT_SIZE,
            video_kbps: None,
            feed_tcp: None,
            fps: 2.0,
            frame_width: 1920,
            mediamtx: None,
            native: false,
            auto: false,
            follow_route: false,
            no_loop: false,
            full_footage: false,
            publish_delay: 0.0,
            checkpoint: None,
            core_bin: None,
            profile: None,
            synthetic: false,
            footage_url: FOOTAGE_URL.to_string(),
            no_fetch: false,
            ground_alt: None,
            site: None,
            help: false,
        }
    }
}

/// Per-feed video bitrate for `--online` (kbit/s).
pub const ONLINE_VIDEO_KBPS: u32 = 2500;

pub const USAGE: &str = "usage: cargo run -p skylens-demo -- [options]

  --core-port N        core port (default 8080; the control tower is served here)
  --board-port N       status board server port (default 8090)
  --no-model           do not start the model server (look at the screens only)
  --model-addr H:P     use a model server already running at gRPC host:port (e.g. an
                       SSH tunnel to a GPU node) instead of starting one; its REST
                       /health is expected on the same host, port 8100
  --online[=HOST]      run against the deployment, not this machine: the footage goes to
                       the bridge, the drones fly against the Daejeon core, and the
                       screens people watch are the deployed ones. Nothing is started
                       here or there: the servers run as services, and the launcher
                       waits for the core to answer and gives up if it does not.
                       HOST is the bridge, default the usual one. Includes --auto and
                       --full-footage. Needs the VPN
  --remote-core H:P    use a core already running at host:port (e.g. an SSH tunnel to
                       Daejeon): start only the footage feeds and the drones
  --srt-addr H:P       publish the footage over SRT to this host:port (default: the
                       local MediaMTX, 127.0.0.1:10890)
  --video-kbps N       re-encode each footage feed to N kbit/s (0 = send the clip as it is;
                       --online defaults to 2500, the clips' own ~10 Mbit/s each would
                       saturate the VPN uplink and drop the drones' sockets)
  --srt-pkt-size N     SRT payload in bytes (default 1128, six MPEG-TS packets).
                       Raise it to 1316 on a plain LAN; anything crossing a
                       VPN needs the smaller one or the data packets exceed the
                       path MTU and the stream never publishes
  --feed-tcp H:P       publish the footage as MPEG-TS over TCP to host:port (drone N to
                       port+N-1) instead of SRT, for links that block UDP
  --fps X              frames per second extracted per drone (default 2)
  --frame-width N      extracted frame width in px, 0 = source size (default 1920)
  --mediamtx PATH      MediaMTX executable (default ~/tools/mediamtx/mediamtx, then PATH)
  --native             also open the native control tower and status board
  --auto               assign the pre-planned route automatically (unattended run)
  --follow-route       drones fly the assigned waypoints instead of the footage track
  --no-loop            play the footage once instead of looping it (drones then hold
                       their last position)
  --full-footage       play every pass of the footage (default: only the first, high
                       pass, about 53 s; the second low pass adds no points)
  --publish-delay S    wait S seconds before the footage starts (e.g. until the drones
                       are online)
  --synthetic          old clips (res/static/video/demo_*.mp4) on a synthetic track
                       instead of the recorded flight in data/demo
  --footage-url URL    where to download the recorded footage zip from
                       (default: the demo-footage-v1 GitHub release)
  --no-fetch           never download the footage; print the manual steps instead
  --ground-alt M       ground elevation at takeoff in metres above sea level
                       (default: the core's DEM source, else a built-in estimate)
  --site SITE          SKYLENS_CORE_SITE for the core, verbatim
                       (default: lat,lon,alt centred on the recorded flight)
  --checkpoint PATH    SkyLensNet checkpoint (default runs/skylens/final)
  --core-bin PATH      Go core executable (default target/skylens-core)
  --profile NAME       cargo profile of the native screens: release or debug
                       (default: release when built, else debug)
  -h, --help           this text";

/// A bare host or address, with no port and nothing that would turn into one.
fn host(name: &str, v: &str) -> Result<String, String> {
    let v = v.trim();
    match v {
        "" => Err(format!("{name} needs a host")),
        _ if v.contains(':') || v.contains('/') => Err(format!("{name}: a host without a port, not {v}")),
        _ => Ok(v.to_string()),
    }
}

fn host_port(name: &str, v: String) -> Result<String, String> {
    match v.rsplit_once(':') {
        Some((h, p)) if !h.is_empty() && p.parse::<u16>().is_ok_and(|p| p > 0) => Ok(v),
        _ => Err(format!("{name}: not host:port: {v}")),
    }
}

fn port(name: &str, v: &str) -> Result<u16, String> {
    match v.parse::<u16>() {
        Ok(p) if p > 0 => Ok(p),
        _ => Err(format!("{name}: not a port: {v}")),
    }
}

pub fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<Options, String> {
    let mut o = Options::default();
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut value = |name: &str| -> Result<String, String> {
            match &inline {
                Some(v) => Ok(v.clone()),
                None => it.next().ok_or_else(|| format!("{name} needs a value")),
            }
        };
        match flag.as_str() {
            "--core-port" => o.core_port = port("--core-port", &value("--core-port")?)?,
            "--board-port" => o.board_port = port("--board-port", &value("--board-port")?)?,
            "--model-addr" => o.model_addr = Some(host_port("--model-addr", value("--model-addr")?)?),
            "--online" => {
                // No space-separated value: `--online --auto` must not swallow the next
                // flag as a host.
                o.online = Some(match &inline {
                    Some(v) if !v.trim().is_empty() => host("--online", v)?,
                    Some(_) => return Err("--online=HOST needs a host".into()),
                    None => ONLINE_HOST.to_string(),
                });
            }
            "--remote-core" => o.remote_core = Some(host_port("--remote-core", value("--remote-core")?)?),
            "--feed-tcp" => o.feed_tcp = Some(host_port("--feed-tcp", value("--feed-tcp")?)?),
            "--srt-addr" => o.srt_addr = Some(host_port("--srt-addr", value("--srt-addr")?)?),
            "--srt-pkt-size" => {
                let v = value("--srt-pkt-size")?;
                o.srt_pkt_size = v.parse().map_err(|_| format!("--srt-pkt-size 는 숫자여야 합니다: {v}"))?;
            }
            "--video-kbps" => {
                let v = value("--video-kbps")?;
                o.video_kbps = Some(v.parse().map_err(|_| format!("--video-kbps 는 숫자여야 합니다: {v}"))?);
            }
            "--publish-delay" => {
                let v = value("--publish-delay")?;
                o.publish_delay = match v.parse::<f64>() {
                    Ok(x) if x.is_finite() && x >= 0.0 => x,
                    _ => return Err(format!("--publish-delay needs a number of seconds, got {v}")),
                };
            }
            "--fps" => {
                let v = value("--fps")?;
                o.fps = match v.parse::<f64>() {
                    Ok(x) if x.is_finite() && x > 0.0 => x,
                    _ => return Err(format!("--fps: not a positive number: {v}")),
                };
            }
            "--frame-width" => {
                let v = value("--frame-width")?;
                o.frame_width = v.parse().map_err(|_| format!("--frame-width: not a width: {v}"))?;
            }
            "--mediamtx" => o.mediamtx = Some(PathBuf::from(value("--mediamtx")?)),
            "--checkpoint" => o.checkpoint = Some(PathBuf::from(value("--checkpoint")?)),
            "--core-bin" => o.core_bin = Some(PathBuf::from(value("--core-bin")?)),
            "--profile" => {
                let v = value("--profile")?;
                o.profile = Some(match v.as_str() {
                    "release" => "release".to_string(),
                    "debug" | "dev" => "debug".to_string(),
                    _ => return Err(format!("--profile: release or debug, not {v}")),
                });
            }
            "--footage-url" => {
                let v = value("--footage-url")?;
                if !(v.starts_with("https://") || v.starts_with("http://")) {
                    return Err(format!("--footage-url: not an http(s) URL: {v}"));
                }
                o.footage_url = v;
            }
            "--ground-alt" => {
                let v = value("--ground-alt")?;
                o.ground_alt = match v.parse::<f64>() {
                    Ok(x) if x.is_finite() && (-500.0..=9000.0).contains(&x) => Some(x),
                    _ => return Err(format!("--ground-alt: not an elevation in metres: {v}")),
                };
            }
            "--site" => {
                let v = value("--site")?;
                if v.trim().is_empty() {
                    return Err("--site needs a value".into());
                }
                o.site = Some(v);
            }
            "--no-model" | "--native" | "--auto" | "--follow-route" | "--no-loop" | "--full-footage" | "--synthetic" | "--no-fetch" | "-h" | "--help"
                if inline.is_some() =>
            {
                return Err(format!("{flag} takes no value"));
            }
            "--no-model" => o.no_model = true,
            "--native" => o.native = true,
            "--auto" => o.auto = true,
            "--follow-route" => o.follow_route = true,
            "--no-loop" => o.no_loop = true,
            "--full-footage" => o.full_footage = true,
            "--synthetic" => o.synthetic = true,
            "--no-fetch" => o.no_fetch = true,
            "-h" | "--help" => o.help = true,
            other => return Err(format!("unknown option: {other}")),
        }
    }
    if let Some(host) = o.online.clone() {
        // A preset, not a mode: anything given explicitly stays as it is.
        o.remote_core.get_or_insert_with(|| format!("{host}:{ONLINE_CORE_PORT}"));
        o.srt_addr.get_or_insert_with(|| format!("{host}:{SRT_PORT}"));
        // The live run is the whole flight, and it starts by itself: without these two
        // the launcher takes everything down after the first zone and waits for someone
        // to assign a route in the control tower.
        o.auto = true;
        o.full_footage = true;
        o.video_kbps.get_or_insert(ONLINE_VIDEO_KBPS);
    }
    let fixed = [MODEL_REST_PORT, MODEL_GRPC_PORT, WHEP_PORT, SRT_PORT];
    if o.core_port == o.board_port {
        return Err(format!("--core-port and --board-port are both {}", o.core_port));
    }
    for p in [o.core_port, o.board_port] {
        if fixed.contains(&p) {
            return Err(format!("port {p} is taken by the model server or MediaMTX"));
        }
    }
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Options, String> {
        parse_args(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn defaults_match_production_ports() {
        let o = parse(&[]).unwrap();
        assert_eq!(o, Options::default());
        assert_eq!((o.core_port, o.board_port), (8080, 8090));
        assert_eq!(o.fps, 2.0);
        assert_eq!(o.frame_width, 1920);
        assert!(!o.no_model && !o.native && !o.auto && !o.follow_route && !o.no_loop && !o.synthetic);
    }

    #[test]
    fn parses_values_in_both_spellings() {
        let o = parse(&[
            "--core-port", "9080", "--board-port=9090", "--fps", "1.5", "--frame-width=0",
            "--mediamtx", "C:/tools/mediamtx.exe", "--checkpoint=runs/x", "--core-bin", "core.exe",
            "--profile", "dev",
        ])
        .unwrap();
        assert_eq!((o.core_port, o.board_port), (9080, 9090));
        assert_eq!(o.fps, 1.5);
        assert_eq!(o.frame_width, 0);
        assert_eq!(o.mediamtx, Some(PathBuf::from("C:/tools/mediamtx.exe")));
        assert_eq!(o.checkpoint, Some(PathBuf::from("runs/x")));
        assert_eq!(o.core_bin, Some(PathBuf::from("core.exe")));
        assert_eq!(o.profile.as_deref(), Some("debug"));
    }

    #[test]
    fn parses_flags() {
        let o = parse(&["--no-model", "--native", "--auto", "--follow-route"]).unwrap();
        assert!(o.no_model && o.native && o.auto && o.follow_route);
        assert!(parse(&["-h"]).unwrap().help);
        assert!(parse(&["--help"]).unwrap().help);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(parse(&["--core-port"]).is_err());
        assert!(parse(&["--core-port", "0"]).is_err());
        assert!(parse(&["--core-port", "70000"]).is_err());
        assert!(parse(&["--fps", "0"]).is_err());
        assert!(parse(&["--fps", "-1"]).is_err());
        assert!(parse(&["--fps", "abc"]).is_err());
        assert!(parse(&["--frame-width", "-5"]).is_err());
        assert!(parse(&["--profile", "fast"]).is_err());
        assert!(parse(&["--auto=1"]).is_err());
        assert!(parse(&["--bogus"]).is_err());
    }

    #[test]
    fn footage_options() {
        let o = parse(&[]).unwrap();
        assert!(!o.synthetic && !o.no_fetch);
        assert_eq!(o.footage_url, FOOTAGE_URL);
        assert!(FOOTAGE_URL.ends_with("/demo-footage-v1/skylens-demo-footage.zip"));
        assert_eq!((o.ground_alt, o.site.as_deref()), (None, None));

        let o = parse(&[
            "--synthetic", "--no-fetch", "--footage-url", "http://mirror/f.zip?a=b", "--ground-alt=48.5",
            "--site", "36.3655,127.3423,48.5",
        ])
        .unwrap();
        assert!(o.synthetic && o.no_fetch);
        assert_eq!(o.footage_url, "http://mirror/f.zip?a=b");
        assert_eq!(o.ground_alt, Some(48.5));
        assert_eq!(o.site.as_deref(), Some("36.3655,127.3423,48.5"));
        assert_eq!(parse(&["--site=daejeon"]).unwrap().site.as_deref(), Some("daejeon"));

        assert!(parse(&["--footage-url", "ftp://x/f.zip"]).is_err());
        assert!(parse(&["--ground-alt", "high"]).is_err());
        assert!(parse(&["--ground-alt", "99999"]).is_err());
        assert!(parse(&["--site", " "]).is_err());
        assert!(parse(&["--synthetic=1"]).is_err());
        assert!(parse(&["--no-fetch=yes"]).is_err());
    }

    #[test]
    fn model_addr() {
        assert_eq!(parse(&["--model-addr", "127.0.0.1:8101"]).unwrap().model_addr.as_deref(), Some("127.0.0.1:8101"));
        assert!(parse(&["--model-addr", "8101"]).is_err());
        assert!(parse(&["--model-addr=host:x"]).is_err());
        assert!(parse(&["--model-addr"]).is_err());
        let o = parse(&["--remote-core", "127.0.0.1:18080", "--srt-addr=host:10890"]).unwrap();
        assert_eq!((o.remote_core.as_deref(), o.srt_addr.as_deref()), (Some("127.0.0.1:18080"), Some("host:10890")));
        assert!(parse(&["--remote-core", "x"]).is_err());
        assert_eq!(parse(&["--feed-tcp", "127.0.0.1:19101"]).unwrap().feed_tcp.as_deref(), Some("127.0.0.1:19101"));
    }

    #[test]
    fn rejects_port_clashes() {
        assert!(parse(&["--core-port", "8090"]).is_err());
        assert!(parse(&["--board-port", "8100"]).is_err());
        assert!(parse(&["--core-port", "10889"]).is_err());
        assert!(parse(&["--core-port", "8081", "--board-port", "8082"]).is_ok());
    }

    #[test]
    fn online_is_the_deployment_in_one_flag() {
        let o = parse(&["--online"]).unwrap();
        assert_eq!(o.online.as_deref(), Some(ONLINE_HOST));
        assert_eq!(o.video_kbps, Some(ONLINE_VIDEO_KBPS));
        assert_eq!(o.remote_core.as_deref(), Some(format!("{ONLINE_HOST}:{ONLINE_CORE_PORT}").as_str()));
        assert_eq!(o.srt_addr.as_deref(), Some(format!("{ONLINE_HOST}:{SRT_PORT}").as_str()));
        // The live run flies the whole flight and starts by itself.
        assert!(o.auto && o.full_footage);

        // Another bridge, same shape.
        let o = parse(&["--online=10.0.0.9"]).unwrap();
        assert_eq!(o.remote_core.as_deref(), Some("10.0.0.9:8090"));
        assert_eq!(o.srt_addr.as_deref(), Some("10.0.0.9:10890"));

        // What is given explicitly wins, in either order.
        let o = parse(&["--online", "--remote-core", "127.0.0.1:9999"]).unwrap();
        assert_eq!(o.remote_core.as_deref(), Some("127.0.0.1:9999"));
        let o = parse(&["--srt-addr", "127.0.0.1:1", "--online"]).unwrap();
        assert_eq!(o.srt_addr.as_deref(), Some("127.0.0.1:1"));

        // A host, not an address: the ports are ours to pick.
        assert!(parse(&["--online=10.0.0.9:8090"]).is_err());
        assert!(parse(&["--online="]).is_err());
    }
}
