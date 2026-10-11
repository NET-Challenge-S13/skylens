//! Native window entry: `skylens-control [--core URL] [--display MODE] [--bridge URL]
//! [--video TEMPLATE|off] [--ffmpeg PATH]`.

use crate::app::{ControlApp, Options, page};
use crate::bridge::DEFAULT_BRIDGE;
use crate::settings::DisplayMode;
use crate::video::{DEFAULT_TEMPLATE, VideoConfig};
use skylens_viewer::link::{CONTROL_PATH, normalize_core_url};
use skylens_viewer::{ScreenDocument, WindowOptions};

pub const DEFAULT_CORE: &str = "ws://127.0.0.1:8080/control";

pub const USAGE: &str = "usage: skylens-control [--core ws://host:8080/control] [--display points|black|aerial] \
[--bridge http://127.0.0.1:8091] [--video TEMPLATE|off] [--ffmpeg PATH]

  --core     core socket (host, http(s) or ws(s) URL; /control is added when no path is given)
  --display  building display for this run only (not saved), like ?display= on the web
  --bridge   DJI Fly mission bridge (KMZ export, formation, cloud missions)
  --video    ffmpeg input for the camera panel; {host} {id} {station} {path} {whep}
             (default srt://{host}:10890?streamid=read:{path}, env SKYLENS_CONTROL_VIDEO_URL)
  --ffmpeg   ffmpeg executable (default ffmpeg, env SKYLENS_FFMPEG)";

/// Parses the command line (environment variables give the defaults of `--video` and `--ffmpeg`).
pub fn parse_args<I: IntoIterator<Item = String>>(args: I, env: impl Fn(&str) -> Option<String>) -> Result<Options, String> {
    let mut core = DEFAULT_CORE.to_string();
    let mut display = None;
    let mut bridge = DEFAULT_BRIDGE.to_string();
    let mut video = env("SKYLENS_CONTROL_VIDEO_URL").filter(|v| !v.trim().is_empty()).unwrap_or_else(|| DEFAULT_TEMPLATE.to_string());
    let mut ffmpeg = env("SKYLENS_FFMPEG").filter(|v| !v.trim().is_empty()).unwrap_or_else(|| "ffmpeg".to_string());
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) if k.starts_with("--") => (k.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut value = |name: &str| inline.clone().or_else(|| it.next()).ok_or_else(|| format!("{name} needs a value\n\n{USAGE}"));
        match key.as_str() {
            "--core" => core = normalize_core_url(&value("--core")?, CONTROL_PATH),
            "--display" => {
                let v = value("--display")?;
                display = Some(DisplayMode::parse(&v).ok_or_else(|| format!("unknown display mode {v:?}\n\n{USAGE}"))?);
            }
            "--bridge" => bridge = crate::bridge::normalize_base(&value("--bridge")?),
            "--video" => video = value("--video")?,
            "--ffmpeg" => ffmpeg = value("--ffmpeg")?,
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown argument {other:?}\n\n{USAGE}")),
        }
    }
    let video = (!video.eq_ignore_ascii_case("off")).then(|| VideoConfig { template: video, ffmpeg, host: crate::http::host_of(&core) });
    Ok(Options { core_url: core, display_override: display, bridge, video })
}

/// Opens the tower window and runs until it closes.
pub fn run(opts: Options) -> anyhow::Result<()> {
    let doc = ScreenDocument::new(&page(), ControlApp::new(opts));
    skylens_viewer::shell::run_native(doc, WindowOptions::new("SkyLens · 관제탑"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn defaults() {
        let o = parse_args(Vec::<String>::new(), no_env).unwrap();
        assert_eq!(o.core_url, DEFAULT_CORE);
        assert_eq!(o.bridge, DEFAULT_BRIDGE);
        assert_eq!(o.display_override, None);
        let v = o.video.unwrap();
        assert_eq!(v.template, DEFAULT_TEMPLATE);
        assert_eq!(v.host, "127.0.0.1");
    }

    #[test]
    fn flags_and_inline_values() {
        let args = ["--core", "10.0.0.5:8080", "--display=points", "--bridge", "http://x:1/", "--video", "off"].map(String::from);
        let o = parse_args(args, no_env).unwrap();
        assert_eq!(o.core_url, "ws://10.0.0.5:8080/control");
        assert_eq!(o.display_override, Some(DisplayMode::Points));
        assert_eq!(o.bridge, "http://x:1");
        assert!(o.video.is_none());
        assert!(parse_args(["--display", "neon"].map(String::from), no_env).is_err());
        assert!(parse_args(["--wat"].map(String::from), no_env).is_err());
        assert!(parse_args(["--core"].map(String::from), no_env).is_err());
    }

    #[test]
    fn environment_gives_the_video_defaults() {
        let env = |k: &str| match k {
            "SKYLENS_CONTROL_VIDEO_URL" => Some("rtsp://{host}/{path}".to_string()),
            "SKYLENS_FFMPEG" => Some("C:/ffmpeg/bin/ffmpeg.exe".to_string()),
            _ => None,
        };
        let v = parse_args(Vec::<String>::new(), env).unwrap().video.unwrap();
        assert_eq!(v.template, "rtsp://{host}/{path}");
        assert_eq!(v.ffmpeg, "C:/ffmpeg/bin/ffmpeg.exe");
    }
}
