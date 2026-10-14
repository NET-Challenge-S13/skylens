//! Command line of `skylens-demo`.

use std::path::PathBuf;

/// Ports the launcher does not let you move: the model server's REST and gRPC
/// listeners and MediaMTX's WHEP signalling (the screens and the core assume them).
pub const MODEL_REST_PORT: u16 = 8100;
pub const MODEL_GRPC_PORT: u16 = 8101;
pub const WHEP_PORT: u16 = 10889;
pub const SRT_PORT: u16 = 10890;
pub const WEBRTC_UDP_PORT: u16 = 10189;

#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// Core HTTP/WebSocket port (production uses 8080).
    pub core_port: u16,
    /// Board server port, playing the gateway's part (the legacy board port).
    pub board_port: u16,
    /// Run without the model server (core only, to look at the screens).
    pub no_model: bool,
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
    /// SkyLensNet checkpoint for the model server.
    pub checkpoint: Option<PathBuf>,
    /// Go core executable; searched for when absent.
    pub core_bin: Option<PathBuf>,
    /// Cargo profile directory of the native screens (`release` or `debug`).
    pub profile: Option<String>,
    pub help: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            core_port: 8080,
            board_port: 8090,
            no_model: false,
            fps: 2.0,
            frame_width: 1920,
            mediamtx: None,
            native: false,
            auto: false,
            follow_route: false,
            checkpoint: None,
            core_bin: None,
            profile: None,
            help: false,
        }
    }
}

pub const USAGE: &str = "usage: cargo run -p skylens-demo -- [options]

  --core-port N        core port (default 8080; the control tower is served here)
  --board-port N       status board server port (default 8090)
  --no-model           do not start the model server (look at the screens only)
  --fps X              frames per second extracted per drone (default 2)
  --frame-width N      extracted frame width in px, 0 = source size (default 1920)
  --mediamtx PATH      MediaMTX executable (default ~/tools/mediamtx/mediamtx, then PATH)
  --native             also open the native control tower and status board
  --auto               assign the route automatically (unattended run)
  --follow-route       drones fly the assigned waypoints instead of the footage track
  --checkpoint PATH    SkyLensNet checkpoint (default runs/skylens/final)
  --core-bin PATH      Go core executable (default target/skylens-core)
  --profile NAME       cargo profile of the native screens: release or debug
                       (default: release when built, else debug)
  -h, --help           this text";

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
            "--no-model" | "--native" | "--auto" | "--follow-route" | "-h" | "--help" if inline.is_some() => {
                return Err(format!("{flag} takes no value"));
            }
            "--no-model" => o.no_model = true,
            "--native" => o.native = true,
            "--auto" => o.auto = true,
            "--follow-route" => o.follow_route = true,
            "-h" | "--help" => o.help = true,
            other => return Err(format!("unknown option: {other}")),
        }
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
        assert!(!o.no_model && !o.native && !o.auto && !o.follow_route);
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
    fn rejects_port_clashes() {
        assert!(parse(&["--core-port", "8090"]).is_err());
        assert!(parse(&["--board-port", "8100"]).is_err());
        assert!(parse(&["--core-port", "10889"]).is_err());
        assert!(parse(&["--core-port", "8081", "--board-port", "8082"]).is_ok());
    }
}
