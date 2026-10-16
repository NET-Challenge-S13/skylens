//! Everything the demo needs, found (or explained) before anything starts.

use std::path::{Path, PathBuf};

use crate::args::{self, Options};
use crate::http1;
use crate::paths::{self, FORMATION, exe};

/// Resolved locations of every program and folder the demo uses.
#[derive(Debug, Clone)]
pub struct Plan {
    pub repo: PathBuf,
    /// Generated files of this run (MediaMTX config, planned route): `<target>/demo`.
    pub run_dir: PathBuf,
    pub footage: Vec<PathBuf>,
    pub mediamtx: PathBuf,
    pub mediamtx_deploy_config: PathBuf,
    pub ffmpeg: PathBuf,
    pub core_bin: PathBuf,
    /// Model server, its libtorch `lib` folder and the checkpoint (absent with `--no-model`).
    pub model: Option<ModelPlan>,
    pub control_dist: PathBuf,
    pub client_dist: PathBuf,
    /// Native control tower and status board (with `--native`).
    pub native: Option<(PathBuf, PathBuf)>,
    /// `.env.vworld` at the repo root, when present.
    pub vworld_env: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct ModelPlan {
    pub bin: PathBuf,
    pub libtorch: Option<PathBuf>,
    pub checkpoint: PathBuf,
}

/// Problems that stop the demo, and ones it can run with.
#[derive(Debug, Default)]
pub struct Findings {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

fn rel(repo: &Path, p: &Path) -> String {
    p.strip_prefix(repo).unwrap_or(p).display().to_string().replace('\\', "/")
}

pub fn resolve(o: &Options, repo: &Path) -> (Option<Plan>, Findings) {
    let mut f = Findings::default();
    let target = paths::target_dir(repo, std::env::var("CARGO_TARGET_DIR").ok().as_deref());

    // footage (the recorded flight was fetched before this, when it could be)
    let footage: Vec<PathBuf> = FORMATION.iter().map(|d| paths::footage(repo, d, o.synthetic)).collect();
    let mut wanted = footage.clone();
    if !o.synthetic {
        wanted.push(paths::track_file(repo));
    }
    let missing: Vec<String> = wanted.iter().filter(|p| !p.is_file()).map(|p| rel(repo, p)).collect();
    if !missing.is_empty() {
        let pairing = if o.synthetic {
            "드론 1·2·3 이 각각 demo_left · demo_center · demo_right 를 송출합니다 (--synthetic)."
        } else {
            "드론 1·2·3 이 각각 data/demo 의 left · center · right 를 송출하고 track.json 의 비행 기록을 재생합니다."
        };
        f.errors.push(format!("촬영 영상이 없습니다: {}\n    {pairing}", missing.join(", ")));
    }

    // ffmpeg
    let ffmpeg = paths::which("ffmpeg");
    if ffmpeg.is_none() {
        f.errors.push(
            "ffmpeg 를 PATH 에서 찾지 못했습니다.\n    영상 송출과 코어의 프레임 추출에 필요합니다. 설치 후 PATH 에 넣으십시오 (https://ffmpeg.org/download.html)."
                .into(),
        );
    }

    // MediaMTX
    let home = paths::home_dir();
    let mut mtx_candidates = paths::mediamtx_candidates(o.mediamtx.as_deref(), home.as_deref());
    let mediamtx = paths::first_file(&mtx_candidates).or_else(|| if o.mediamtx.is_none() { paths::which("mediamtx") } else { None });
    if mediamtx.is_none() {
        if o.mediamtx.is_none() {
            mtx_candidates.push(PathBuf::from("PATH 의 mediamtx"));
        }
        let suggested = home
            .as_deref()
            .map(|h| h.join("tools").join("mediamtx").join(exe("mediamtx")).display().to_string())
            .unwrap_or_else(|| format!("~/tools/mediamtx/{}", exe("mediamtx")));
        f.errors.push(format!(
            "MediaMTX 를 찾지 못했습니다 (찾아본 곳: {}).\n    https://github.com/bluenviron/mediamtx/releases 에서 받아 {suggested} 에 두거나 --mediamtx <경로> 로 지정하십시오.",
            mtx_candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
        ));
    }
    let mediamtx_deploy_config = repo.join("src").join("skylens_core").join("deploy").join("mediamtx.yml");
    if !mediamtx_deploy_config.is_file() {
        f.errors.push(format!("MediaMTX 설정 원본이 없습니다: {}", rel(repo, &mediamtx_deploy_config)));
    }

    // Go core
    let core_candidates = paths::core_candidates(o.core_bin.as_deref(), repo, &target);
    let core_bin = paths::first_file(&core_candidates);
    if core_bin.is_none() {
        f.errors.push(format!(
            "Go 코어 실행 파일이 없습니다 (찾아본 곳: {}).\n    빌드: cd src/skylens_core && go build -o ../../target/{} .",
            core_candidates.iter().map(|p| rel(repo, p)).collect::<Vec<_>>().join(", "),
            exe("skylens-core")
        ));
    }

    // model server
    let model = if o.no_model || o.model_addr.is_some() {
        None
    } else {
        let release = target.join("release");
        let bin = release.join(exe("skylens-model"));
        if !bin.is_file() {
            f.errors.push(format!(
                "모델 서버 release 빌드가 없습니다: {}\n    빌드: cargo build --release -p skylens-model   (debug 빌드는 너무 느려 쓰지 않습니다)\n    모델 없이 화면만 보려면 --no-model",
                rel(repo, &bin)
            ));
        }
        let libtorch = paths::libtorch_lib_dir(&release);
        if libtorch.is_none() && cfg!(windows) && bin.is_file() {
            f.errors.push(format!(
                "libtorch DLL 폴더를 찾지 못했습니다: {}/build/torch-sys-*/out/libtorch/libtorch/lib\n    cargo build --release -p skylens-model 을 다시 실행하십시오.",
                rel(repo, &release)
            ));
        }
        let checkpoint = o.checkpoint.clone().unwrap_or_else(|| repo.join("runs").join("skylens").join("final"));
        let checkpoint = if checkpoint.is_absolute() { checkpoint } else { repo.join(checkpoint) };
        if !checkpoint.exists() {
            f.errors.push(format!(
                "SkyLensNet 체크포인트가 없습니다: {}\n    학습 결과를 그 자리에 두거나 --checkpoint <경로> 로 지정하십시오. 모델 없이 화면만 보려면 --no-model",
                rel(repo, &checkpoint)
            ));
        }
        Some(ModelPlan { bin, libtorch, checkpoint })
    };

    // web builds: warn only, the native screens still work
    let control_dist = repo.join("src").join("skylens_core").join("control").join("dist");
    if !control_dist.join("index.html").is_file() {
        f.warnings.push(format!(
            "관제탑 웹 빌드가 없습니다 ({}). 브라우저 관제탑은 비어 있습니다.\n    빌드: bash src/skylens_core/control/build-web.sh   (또는 --native)",
            rel(repo, &control_dist)
        ));
    }
    let client_dist = repo.join("src").join("skylens_client").join("dist");
    if !client_dist.join("status.html").is_file() {
        f.warnings.push(format!(
            "현황판 웹 빌드가 없습니다 ({}). 브라우저 현황판은 안내 문구만 보입니다.\n    빌드: bash src/skylens_client/build-web.sh   (또는 --native)",
            rel(repo, &client_dist)
        ));
    }

    // native screens
    let native = if o.native {
        let profile = o.profile.clone().unwrap_or_else(|| {
            if target.join("release").join(exe("skylens-control")).is_file() { "release".into() } else { "debug".into() }
        });
        let dir = target.join(&profile);
        let control = dir.join(exe("skylens-control"));
        let client = dir.join(exe("skylens-client"));
        let flag = if profile == "release" { " --release" } else { "" };
        for (bin, pkg) in [(&control, "skylens-control"), (&client, "skylens-client")] {
            if !bin.is_file() {
                f.errors.push(format!("{} 가 없습니다.\n    빌드: cargo build{flag} -p {pkg}", rel(repo, bin)));
            }
        }
        Some((control, client))
    } else {
        None
    };

    let vworld = repo.join(".env.vworld");
    let vworld_env = vworld.is_file().then_some(vworld);

    if !f.errors.is_empty() {
        return (None, f);
    }
    let plan = Plan {
        repo: repo.to_path_buf(),
        run_dir: target.join("demo"),
        footage,
        mediamtx: mediamtx.expect("checked"),
        mediamtx_deploy_config,
        ffmpeg: ffmpeg.expect("checked"),
        core_bin: core_bin.expect("checked"),
        model,
        control_dist,
        client_dist,
        native,
        vworld_env,
    };
    (Some(plan), f)
}

/// Ports another process already holds. A leftover from an earlier run would
/// answer the readiness checks and the demo would quietly run against old code.
pub async fn ports_in_use(o: &Options) -> Vec<String> {
    let mut tcp = vec![(o.core_port, "코어"), (o.board_port, "현황판 서버"), (args::WHEP_PORT, "MediaMTX WHEP")];
    if !o.no_model && o.model_addr.is_none() {
        tcp.push((args::MODEL_REST_PORT, "모델 REST"));
        tcp.push((args::MODEL_GRPC_PORT, "모델 gRPC"));
    }
    let mut taken = Vec::new();
    for (port, what) in tcp {
        if http1::listening(&format!("127.0.0.1:{port}")).await {
            taken.push(format!("{port} ({what})"));
        }
    }
    for (port, what) in [(args::SRT_PORT, "MediaMTX SRT/udp"), (args::WEBRTC_UDP_PORT, "MediaMTX WebRTC/udp")] {
        if std::net::UdpSocket::bind(("0.0.0.0", port)).is_err() {
            taken.push(format!("{port} ({what})"));
        }
    }
    taken
}

pub fn ports_message(taken: &[String]) -> String {
    let ports: Vec<&str> = taken.iter().map(|t| t.split(' ').next().unwrap_or("")).collect();
    let windows = format!(
        "Get-NetTCPConnection -LocalPort {} | Select-Object LocalPort, OwningProcess",
        ports.join(",")
    );
    format!(
        "포트를 이미 다른 프로세스가 쓰고 있습니다: {}\n    이전 데모가 완전히 끝나지 않았을 가능성이 큽니다. 그대로 두면 런처가 그 프로세스를\n    자기 컴포넌트로 오인해 옛 코드로 데모가 돕니다. 정리한 뒤 다시 실행하십시오.\n    Windows (PowerShell)  {windows}\n    그 외                 lsof -i :{}",
        taken.join(", "),
        ports.join(" -i :"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_message_names_every_port() {
        let msg = ports_message(&["8080 (코어)".into(), "10890 (MediaMTX SRT/udp)".into()]);
        assert!(msg.contains("8080 (코어), 10890 (MediaMTX SRT/udp)"));
        assert!(msg.contains("-LocalPort 8080,10890"));
        assert!(msg.contains("lsof -i :8080 -i :10890"));
    }

    #[test]
    fn missing_everything_is_reported_in_korean() {
        let repo = std::env::temp_dir().join(format!("skylens-demo-empty-repo-{}", std::process::id()));
        std::fs::create_dir_all(&repo).unwrap();
        let o = Options { mediamtx: Some(repo.join("nope.exe")), core_bin: Some(repo.join("nope-core")), ..Options::default() };
        let (plan, f) = resolve(&o, &repo);
        assert!(plan.is_none());
        let all = f.errors.join("\n");
        for needle in ["촬영 영상", "MediaMTX", "Go 코어", "go build", "cargo build --release -p skylens-model", "체크포인트"] {
            assert!(all.contains(needle), "{needle} not in:\n{all}");
        }
        assert_eq!(f.warnings.len(), 2, "both web builds are warnings");
        assert!(all.contains("data/demo/left.mp4") && all.contains("data/demo/track.json"), "{all}");
        let (_, f) = resolve(&Options { synthetic: true, ..o.clone() }, &repo);
        let all = f.errors.join("\n");
        assert!(all.contains("res/static/video/demo_left.mp4") && !all.contains("track.json"), "{all}");
        let (_, f) = resolve(&Options { no_model: true, ..o }, &repo);
        assert!(!f.errors.join("\n").contains("체크포인트"));
        let _ = std::fs::remove_dir_all(&repo);
    }
}
