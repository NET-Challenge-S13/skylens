//! End-to-end demo launcher: every component, for real, on one machine.
//!
//!     cargo run -p skylens-demo -- [--native] [--auto] [--no-model] ...
//!
//! Before anything starts: the recorded footage is downloaded into `data/demo` when missing,
//! the ground elevation at takeoff is looked up, and the planned route is derived from the
//! center view (written to `<target>/demo/planned-route.json` for the core).
//!
//! Start order, each gated on readiness before the next:
//!
//!   MediaMTX (:10889 WHEP, :10890 SRT)
//!   -> footage publishers (ffmpeg, drone1..3 = left/center/right)
//!   -> model server (:8100 REST, :8101 gRPC)
//!   -> core (:8080, serves the control tower)
//!   -> board server (:8090, the gateway's part: status board, /viewer, /uplink, /live/)
//!   -> drone uplink simulator (idle until a route is assigned)
//!   -> native screens (--native)
//!
//! Ctrl-C, or any component dying, stops everything. See README.md.

mod args;
mod board;
mod elevation;
mod fetch;
mod footage;
mod http1;
mod paths;
mod preflight;
mod procs;
mod route;
mod sim;

use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use skylens_shared::geo::Gps;
use skylens_shared::proto::viewer::v1 as viewer;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use args::Options;
use footage::Track;
use preflight::Plan;
use procs::{Event, Events, Running, Spec, log};

/// What this run flies and tells the core, settled before anything starts.
struct Scenario {
    /// The recorded flight; `None` with `--synthetic`.
    track: Option<Track>,
    /// Ground elevation at takeoff, metres above sea level (recorded flight only).
    ground_alt: f64,
    /// `SKYLENS_CORE_SITE`; `None` leaves the core's default (daejeon).
    site: Option<String>,
    /// The core's ENU anchor when the launcher can tell, for the model server's `SKYLENS_ANCHOR`.
    anchor: Option<Gps>,
    /// The route the tower starts with, and what `--auto` assigns.
    planned: viewer::AssignRoute,
}

/// Ground elevation at takeoff: `--ground-alt`, else the core's DEM source, else the built-in
/// estimate. Returns the value and where it came from.
async fn ground_elevation(o: &Options) -> (f64, String) {
    if let Some(g) = o.ground_alt {
        return (g, "--ground-alt".into());
    }
    let (lat, lon) = elevation::TAKEOFF;
    let looked_up = tokio::task::spawn_blocking(move || elevation::ground_alt(lat, lon)).await;
    match looked_up {
        Ok(Ok(g)) => (g, "DEM (terrarium z12, 코어 지형과 같은 출처)".into()),
        Ok(Err(e)) => {
            eprintln!(
                "  경고: 이륙 지점 지면 고도를 DEM 에서 얻지 못했습니다 ({e}).\n    추정값 {:.0} m 를 씁니다. 정확한 값을 알면 --ground-alt <m> 로 주십시오.",
                elevation::FALLBACK_GROUND_ALT_M
            );
            (elevation::FALLBACK_GROUND_ALT_M, "추정값 (DEM 조회 실패)".into())
        }
        Err(e) => (elevation::FALLBACK_GROUND_ALT_M, format!("추정값 ({e})")),
    }
}

/// Settles the flight, site, anchor and planned route. `Err` is a Korean message.
async fn scenario(o: &Options, repo: &Path) -> Result<Scenario, String> {
    let center = paths::FORMATION[1].id;
    if o.synthetic {
        let site = o.site.clone();
        let anchor = footage::site_anchor(site.as_deref().unwrap_or(""));
        let waypoints = route::footage_waypoints(route::SITE_ANCHOR);
        let planned = viewer::AssignRoute { drone_id: center, waypoints: waypoints.into_iter().map(Into::into).collect(), r#loop: true };
        return Ok(Scenario { track: None, ground_alt: 0.0, site, anchor, planned });
    }
    let track = Track::load(&paths::track_file(repo)).map_err(|e| {
        format!("비행 기록을 읽지 못했습니다: {e}\n    data/demo 를 지우고 다시 실행하면 새로 내려받습니다.")
    })?;
    for v in track.views() {
        log("footage", &v.summary());
    }
    let track = if o.full_footage {
        track
    } else {
        let t = track.first_pass_only();
        log("footage", &format!("첫 비행 구간만 재생: {:.1}초 (전체는 --full-footage)", t.center.duration));
        t
    };
    if let Some(c) = footage::Camera::load(&paths::camera_file(repo)) {
        log("footage", &format!("카메라 {} (모델 서버는 내부 파라미터를 받지 않아 넘기지 않습니다)", c.describe(o.frame_width)));
    }
    let (ground_alt, from) = ground_elevation(o).await;
    log("footage", &format!("이륙 지점 지면 고도 {ground_alt:.1} m, {from}. 텔레메트리 고도 = 지면 + 이륙 기준 고도"));
    let site = Some(o.site.clone().unwrap_or_else(|| footage::site_value(track.centre(), ground_alt)));
    let anchor = footage::site_anchor(site.as_deref().unwrap_or(""));
    let waypoints = footage::planned_waypoints(&track.center, ground_alt);
    if waypoints.len() < 2 {
        return Err("track.json 의 center 첫 구간에서 경로를 만들지 못했습니다 (지점이 둘 미만)".into());
    }
    let planned = viewer::AssignRoute { drone_id: center, waypoints: waypoints.into_iter().map(Into::into).collect(), r#loop: true };
    Ok(Scenario { track: Some(track), ground_alt, site, anchor, planned })
}

/// Why the demo is stopping.
enum Stop {
    Interrupted,
    Failed(String),
}

/// React to an event: `Some` when the demo has to stop.
fn judge(ev: Event) -> Option<Stop> {
    match ev {
        Event::Interrupt => Some(Stop::Interrupted),
        Event::Exited { name, critical: true, code } => {
            Some(Stop::Failed(format!("{name} 가 종료되었습니다 (code={})", code.map_or("없음".into(), |c| c.to_string()))))
        }
        Event::Exited { name, critical: false, code } => {
            log("launch", &format!("{name} 창이 닫혔습니다 (code={code:?}). 데모는 계속 돕니다"));
            None
        }
        Event::Failed { name, error } => Some(Stop::Failed(format!("{name}: {error}"))),
    }
}

/// Poll `probe` until it answers, while still reacting to events.
async fn wait_ready<F, Fut>(
    what: &str,
    timeout: Duration,
    events: &mut mpsc::UnboundedReceiver<Event>,
    mut probe: F,
) -> Result<(), Stop>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if probe().await {
            log("launch", &format!("{what} 준비됨"));
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(Stop::Failed(format!("{what} 가 {}초 안에 준비되지 않았습니다", timeout.as_secs())));
        }
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(400)) => {}
            Some(ev) = events.recv() => if let Some(stop) = judge(ev) { return Err(stop) },
        }
    }
}

/// Wait a fixed time, still reacting to events (a child that dies right away).
async fn settle(d: Duration, events: &mut mpsc::UnboundedReceiver<Event>) -> Result<(), Stop> {
    let until = tokio::time::Instant::now() + d;
    loop {
        tokio::select! {
            _ = tokio::time::sleep_until(until) => return Ok(()),
            Some(ev) = events.recv() => if let Some(stop) = judge(ev) { return Err(stop) },
        }
    }
}

/// Run an in-process part; its error stops the demo.
fn task<F>(name: &'static str, events: &Events, fut: F) -> JoinHandle<()>
where
    F: Future<Output = anyhow::Result<()>> + Send + 'static,
{
    let events = events.clone();
    tokio::spawn(async move {
        if let Err(e) = fut.await {
            let _ = events.send(Event::Failed { name: name.to_string(), error: format!("{e:#}") });
        }
    })
}

fn spec(plan: &Plan, name: &str, tag: &str, program: &Path, argv: Vec<String>, env: Vec<(String, String)>) -> Spec {
    Spec {
        name: name.to_string(),
        tag: tag.to_string(),
        program: program.to_path_buf(),
        args: argv,
        env,
        cwd: plan.repo.clone(),
        critical: true,
        delay: Duration::ZERO,
    }
}

fn kv(k: &str, v: impl ToString) -> (String, String) {
    (k.to_string(), v.to_string())
}

/// Environment of the Go core.
fn core_env(plan: &Plan, o: &Options, s: &Scenario, planned_file: &Path) -> Vec<(String, String)> {
    let mut env = vec![
        kv("SKYLENS_CORE_ADDR", format!(":{}", o.core_port)),
        kv("SKYLENS_CORE_WEB_DIR", plan.control_dist.display()),
        kv("SKYLENS_MODEL_GRPC", o.model_addr.clone().unwrap_or_else(|| format!("127.0.0.1:{}", args::MODEL_GRPC_PORT))),
        kv("SKYLENS_CORE_FRAME_SOURCE", if o.no_model { "off" } else { "ffmpeg" }),
        kv("SKYLENS_CORE_FFMPEG", plan.ffmpeg.display()),
        kv("SKYLENS_CORE_STREAM_PATH", "drone{id}"),
        kv("SKYLENS_CORE_FRAME_URL", format!("srt://127.0.0.1:{}?streamid=read:{{path}}", args::SRT_PORT)),
        kv("SKYLENS_CORE_FRAME_FPS", o.fps),
        kv("SKYLENS_CORE_FRAME_WIDTH", o.frame_width),
        kv("SKYLENS_CORE_DRONE_ETA_SEC", sim::ETA_SECS),
        kv("SKYLENS_CORE_PLANNED_ROUTE", planned_file.display()),
    ];
    if let Some(site) = &s.site {
        env.push(kv("SKYLENS_CORE_SITE", site));
    }
    if let Some(v) = &plan.vworld_env {
        env.push(kv("SKYLENS_VWORLD_ENV", v.display()));
    }
    env
}

/// Environment of the model server: listeners, checkpoint, the core's ENU anchor (its chunks
/// are in ENU about it), libtorch DLLs on the search path.
fn model_env(m: &preflight::ModelPlan, anchor: Option<Gps>) -> Vec<(String, String)> {
    let mut env = vec![
        kv("SKYLENS_CHECKPOINT", m.checkpoint.display()),
        kv("SKYLENS_REST_ADDR", format!("127.0.0.1:{}", args::MODEL_REST_PORT)),
        kv("SKYLENS_GRPC_ADDR", format!("127.0.0.1:{}", args::MODEL_GRPC_PORT)),
    ];
    if let Some(a) = anchor {
        env.push(kv("SKYLENS_ANCHOR", footage::anchor_value(a)));
    }
    if let Some(lib) = &m.libtorch {
        let var = if cfg!(windows) { "PATH" } else { "LD_LIBRARY_PATH" };
        let old = std::env::var_os(var).unwrap_or_default();
        let dirs = std::iter::once(lib.clone()).chain(std::env::split_paths(&old));
        if let Ok(joined) = std::env::join_paths(dirs) {
            env.push((var.to_string(), joined.to_string_lossy().into_owned()));
        }
    }
    env
}

async fn start_all(
    plan: &Plan,
    o: &Options,
    s: &Scenario,
    events: &Events,
    rx: &mut mpsc::UnboundedReceiver<Event>,
    children: &mut Vec<Running>,
    tasks: &mut Vec<JoinHandle<()>>,
) -> Result<(), Stop> {
    let fail = |e: anyhow::Error| Stop::Failed(format!("{e:#}"));

    // ---- generated files: the planned route for the core ----
    std::fs::create_dir_all(&plan.run_dir).map_err(|e| Stop::Failed(format!("{}: {e}", plan.run_dir.display())))?;
    let planned_file: PathBuf = paths::planned_route_file(&plan.run_dir);
    let waypoints: Vec<Gps> = s.planned.waypoints.iter().map(|g| Gps { lat: g.lat, lon: g.lon, alt: g.alt }).collect();
    std::fs::write(&planned_file, footage::planned_route_json(s.planned.drone_id, &waypoints, s.planned.r#loop))
        .map_err(|e| Stop::Failed(format!("{}: {e}", planned_file.display())))?;
    log(
        "launch",
        &format!(
            "미리 계획된 경로: {}개 지점, 고도 {:.1} m ({})",
            waypoints.len(),
            waypoints.first().map_or(0.0, |g| g.alt),
            planned_file.display()
        ),
    );

    // ---- MediaMTX ----
    let remote = o.remote_core.is_some();
    let whep = format!("127.0.0.1:{}", args::WHEP_PORT);
    if !remote {
        let deploy = std::fs::read_to_string(&plan.mediamtx_deploy_config)
            .map_err(|e| Stop::Failed(format!("{}: {e}", plan.mediamtx_deploy_config.display())))?;
        let config = plan.run_dir.join("mediamtx.yml");
        std::fs::write(&config, paths::local_mediamtx_config(&deploy)).map_err(|e| Stop::Failed(format!("{}: {e}", config.display())))?;
        log("launch", &format!("MediaMTX 시작 ({})", config.display()));
        children.push(
            procs::spawn(spec(plan, "mediamtx", "mtx", &plan.mediamtx, vec![config.display().to_string()], vec![]), events.clone())
                .map_err(fail)?,
        );
            wait_ready("MediaMTX", Duration::from_secs(20), rx, || http1::listening(&whep)).await?;
    }

    // ---- footage publishers ----
    // Each publisher's start is the zero of its clip's timeline; the drone replaying that
    // view's flight record counts from it.
    let srt_addr = o.srt_addr.clone().unwrap_or_else(|| format!("127.0.0.1:{}", args::SRT_PORT));
    let mut starts = [Instant::now(); 3];
    for (i, (drone, file)) in paths::FORMATION.iter().zip(&plan.footage).enumerate() {
        let tag = format!("pub-{}", &drone.name[..1].to_uppercase());
        let name = format!("송출 drone{} ({})", drone.id, drone.name);
        let url = match &o.feed_tcp {
            Some(base) => paths::tcp_publish_url(base, drone),
            None => paths::srt_publish_url(&srt_addr, drone, o.srt_pkt_size),
        };
        let argv = paths::publisher_args(file, &url, !o.no_loop, if o.full_footage { None } else { s.track.as_ref().map(|t| t.views()[i].duration) }, o.video_kbps.unwrap_or(0));
        let mut publisher = spec(plan, &name, &tag, &plan.ffmpeg, argv, vec![]);
        // A clip that plays once ends on its own; that is not a failure.
        publisher.critical = !o.no_loop;
        publisher.delay = Duration::from_secs_f64(o.publish_delay);
        children.push(procs::spawn(publisher, events.clone()).map_err(fail)?);
        starts[i] = Instant::now() + Duration::from_secs_f64(o.publish_delay);
    }
    settle(Duration::from_millis(1500), rx).await?;
    log("launch", &format!("영상 송출 3개 시작 (drone1=left, drone2=center, drone3=right, {})", if o.no_loop { "한 번만 재생" } else { "반복 재생" }));

    // ---- model server ----
    if remote {
        log("launch", &format!("--remote-core: 코어 {} 를 쓰고 모델 서버·MediaMTX·현황판 서버는 띄우지 않습니다", o.remote_core.as_deref().unwrap_or("")));
    } else if let Some(m) = &plan.model {
        log("launch", "모델 서버 시작 (libtorch·CUDA 로딩에 시간이 걸릴 수 있습니다)");
        if s.anchor.is_none() {
            log("launch", "경고: --site 에서 기준점을 알 수 없어 모델 서버는 기본 기준점(대전)으로 청크를 냅니다");
        }
        children.push(
            procs::spawn(spec(plan, "skylens-model", "model", &m.bin, vec![], model_env(m, s.anchor)), events.clone())
                .map_err(fail)?,
        );
        let rest = format!("127.0.0.1:{}", args::MODEL_REST_PORT);
        wait_ready("모델 서버", Duration::from_secs(300), rx, || http1::ok(&rest, "/health")).await?;
    } else if let Some(addr) = &o.model_addr {
        let host = addr.rsplit_once(':').map_or(addr.as_str(), |(h, _)| h);
        let rest = format!("{host}:{}", args::MODEL_REST_PORT);
        log("launch", &format!("원격 모델 서버 사용: gRPC {addr}, REST {rest}"));
        wait_ready("원격 모델 서버", Duration::from_secs(60), rx, || http1::ok(&rest, "/health")).await?;
    } else {
        log("launch", "--no-model: 모델 서버 없이 진행 (프레임 추출 끔)");
    }

    // ---- core and board server ----
    let core = o.remote_core.clone().unwrap_or_else(|| format!("127.0.0.1:{}", o.core_port));
    if remote {
        wait_ready("원격 코어", Duration::from_secs(60), rx, || http1::ok(&core, "/health")).await?;
    } else {
        match &s.site {
            Some(site) => log("launch", &format!("코어 시작 (작전 구역 SKYLENS_CORE_SITE={site})")),
            None => log("launch", "코어 시작 (작전 구역: 코어 기본값 daejeon)"),
        }
        let env = core_env(plan, o, s, &planned_file);
        children.push(procs::spawn(spec(plan, "skylens-core", "core", &plan.core_bin, vec![], env), events.clone()).map_err(fail)?);
            wait_ready("코어", Duration::from_secs(60), rx, || http1::ok(&core, "/health")).await?;

        // ---- board server ----
        let listener = tokio::net::TcpListener::bind(("0.0.0.0", o.board_port))
            .await
            .map_err(|e| Stop::Failed(format!("현황판 서버 :{}: {e}", o.board_port)))?;
        let app = board::router(board::BoardConfig { dist: plan.client_dist.clone(), core: core.clone(), whep: whep.clone() });
        tasks.push(task("board", events, async move { axum::serve(listener, app).await.map_err(anyhow::Error::from) }));
        let board_addr = format!("127.0.0.1:{}", o.board_port);
        wait_ready("현황판 서버", Duration::from_secs(10), rx, || http1::ok(&board_addr, "/health")).await?;
    }

    // ---- drones and the control watcher ----
    let flight = match &s.track {
        Some(track) => sim::Flight::Recorded { track: track.clone(), starts, ground_alt: s.ground_alt, looped: !o.no_loop },
        None => sim::Flight::Synthetic,
    };
    tasks.push(task("drone", events, sim::run_drones(core.clone(), o.follow_route, flight)));
    let auto = o.auto.then(|| s.planned.clone());
    // On a slow tunnel the core drops this socket (its send queue overflows on the base map). The
    // drones are what the demo needs, so with a remote core a lost watcher is only a warning.
    let watch = sim::run_control_watch(core.clone(), auto);
    tasks.push(task("tower-watch", events, async move {
        match watch.await {
            Err(e) if remote => {
                log("tower", &format!("관제탑 감시가 끊겼습니다 (원격 코어라 계속 진행): {e:#}"));
                Ok(())
            }
            other => other,
        }
    }));

    // ---- native screens ----
    if let Some((control, client)) = &plan.native {
        let mut tower = spec(
            plan,
            "skylens-control",
            "tower",
            control,
            vec!["--core".into(), format!("ws://127.0.0.1:{}/control", o.core_port)],
            vec![],
        );
        tower.critical = false;
        let mut board = spec(
            plan,
            "skylens-client",
            "client",
            client,
            vec!["--core".into(), format!("ws://127.0.0.1:{}/viewer", o.board_port)],
            vec![],
        );
        board.critical = false;
        children.push(procs::spawn(tower, events.clone()).map_err(fail)?);
        children.push(procs::spawn(board, events.clone()).map_err(fail)?);
    }
    Ok(())
}

fn banner(plan: &Plan, o: &Options, s: &Scenario) {
    let line = "  ─────────────────────────────────────────────────────────────";
    println!();
    println!("  SkyLens 데모");
    println!("{line}");
    if let Some(host) = &o.online {
        // The deployed screens. The launcher serves neither of them in this mode, so
        // pointing at localhost would send the operator to a port with nothing on it.
        println!("  관제탑   https://{host}:{}/        (대전 코어가 만들고 브리지가 중계 · VPN·랜만)", args::ONLINE_TOWER_TLS_PORT);
        println!("  현황판   {}   (어느 망에서나)", args::ONLINE_BOARD_URL);
        println!("           https://{host}/              (VPN 안에서 같은 화면)");
    } else {
        println!("  관제탑   http://localhost:{}/          (코어가 서빙)", o.core_port);
        println!("  현황판   http://localhost:{}/          (게이트웨이 역할, 런처 내장 서버)", o.board_port);
    }
    if plan.native.is_some() {
        println!("  네이티브 관제탑 -> ws://127.0.0.1:{}/control, 현황판 -> ws://127.0.0.1:{}/viewer", o.core_port, o.board_port);
    }
    if o.online.is_some() {
        // The launcher starts no model here, but one is running: the Daejeon core talks
        // to the Pangyo GPU node over its own tunnel. Saying "off" would read as "no
        // reconstruction today", which is not what this mode does.
        println!("  모델     대전 코어가 판교 모델 서버를 씁니다 (이 런처가 띄우지 않습니다)");
    } else if let Some(addr) = &o.model_addr {
        println!("  모델     원격 gRPC {addr}  (REST {}번 포트 /health)", args::MODEL_REST_PORT);
    } else if plan.model.is_some() {
        println!("  모델     http://127.0.0.1:{}/health  (gRPC :{})", args::MODEL_REST_PORT, args::MODEL_GRPC_PORT);
    } else {
        println!("  모델     끔 (--no-model)");
    }
    if o.online.is_some() {
        println!("  영상     브리지로 SRT 송출 :{} · 화면은 현황판과 같은 출처의 WHEP 으로 받습니다", args::SRT_PORT);
        println!("  서버     대전 코어·모델·MediaMTX·터널은 서비스로 항상 떠 있습니다 (이 런처가 띄우지 않습니다)");
    } else {
        println!("  영상     WHEP http://localhost:{}/live/drone{{1,2,3}}/whep, SRT :{}", o.board_port, args::SRT_PORT);
    }
    if s.track.is_some() {
        println!("  촬영본   data/demo (실제 비행 기록 재생, 이륙 지점 지면 {:.1} m)", s.ground_alt);
    } else {
        println!("  촬영본   res/static/video/demo_*.mp4 (--synthetic, 합성 경로)");
    }
    println!();
    println!("  시나리오");
    println!("   1. 지금은 정지 상태. 드론 3대는 업링크에 붙어 대기, 복원 없음");
    if o.auto {
        println!("   2. --auto: 3초 뒤 런처가 미리 계획된 경로({}개 지점)를 배정", s.planned.waypoints.len());
    } else {
        println!(
            "   2. 관제탑 경로 계획에 촬영 경로({}개 지점)가 미리 들어가 있습니다. 경로 계획을 열고 배정을 누르십시오",
            s.planned.waypoints.len()
        );
    }
    println!("   3. \"태스크 지정 완료\" 후 드론 연결 대기 (약 {:.0}초, 그동안 드론이 현장으로 이동)", sim::ETA_SECS);
    println!("   4. 드론이 도착해 hello 를 보내면 \"드론 연결됨\", 프레임 추출 시작");
    println!("   5. 구간마다 미리보기 점군·메시가 먼저 뜨고 정제본이 교체");
    println!("   6. 위험구역·사람 탐지 결과가 지도 위에 표시");
    if !o.follow_route {
        if s.track.is_some() {
            println!("   (드론마다 자기 영상의 비행 기록을 영상과 맞춰 재생합니다. 배정 경로를 따르게 하려면 --follow-route)");
        } else {
            println!("   (드론은 촬영본에 맞춘 고정 경로를 날며 텔레메트리를 보냅니다. 배정 경로를 따르게 하려면 --follow-route)");
        }
    }
    println!();
    println!("  Ctrl+C 로 전체 종료");
    println!();
}

async fn run() -> i32 {
    procs::start_clock();
    let o = match args::parse_args(std::env::args().skip(1)) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("skylens-demo: {e}\n\n{}", args::USAGE);
            return 2;
        }
    };
    if o.help {
        println!("{}", args::USAGE);
        return 0;
    }

    let repo = paths::repo_root();
    if !o.synthetic {
        let (repo2, url, no_fetch) = (repo.clone(), o.footage_url.clone(), o.no_fetch);
        let fetched = tokio::task::spawn_blocking(move || fetch::ensure(&repo2, &url, no_fetch))
            .await
            .unwrap_or_else(|e| Err(format!("내려받기 작업이 멈췄습니다: {e}")));
        if let Err(e) = fetched {
            eprintln!("\n  오류: {e}\n");
            return 1;
        }
    }
    let (plan, findings) = preflight::resolve(&o, &repo);
    for w in &findings.warnings {
        eprintln!("  경고: {w}");
    }
    let Some(plan) = plan else {
        eprintln!();
        for e in &findings.errors {
            eprintln!("  오류: {e}");
        }
        eprintln!();
        eprintln!("  위 항목을 해결한 뒤 다시 실행하십시오.");
        return 1;
    };
    let taken = preflight::ports_in_use(&o).await;
    if !taken.is_empty() {
        eprintln!("\n  오류: {}\n", preflight::ports_message(&taken));
        return 1;
    }
    let scenario = match scenario(&o, &repo).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("\n  오류: {e}\n");
            return 1;
        }
    };

    #[cfg(windows)]
    log("launch", &procs::job::init());

    let (events, mut rx) = mpsc::unbounded_channel::<Event>();
    {
        let events = events.clone();
        tokio::spawn(async move {
            let mut pressed = false;
            while tokio::signal::ctrl_c().await.is_ok() {
                if pressed {
                    log("launch", "강제 종료");
                    std::process::exit(130);
                }
                pressed = true;
                let _ = events.send(Event::Interrupt);
            }
        });
    }

    let mut children = Vec::new();
    let mut tasks = Vec::new();
    let stop = match start_all(&plan, &o, &scenario, &events, &mut rx, &mut children, &mut tasks).await {
        Ok(()) => {
            banner(&plan, &o, &scenario);
            loop {
                match rx.recv().await {
                    Some(ev) => {
                        if let Some(stop) = judge(ev) {
                            break stop;
                        }
                    }
                    None => break Stop::Failed("event channel closed".into()),
                }
            }
        }
        Err(stop) => stop,
    };

    let code = match &stop {
        Stop::Interrupted => {
            log("launch", "Ctrl+C: 전체 종료");
            0
        }
        Stop::Failed(why) => {
            log("launch", &format!("중단: {why}"));
            1
        }
    };
    for t in &tasks {
        t.abort();
    }
    procs::stop_all(children).await;
    log("launch", "종료 완료");
    code
}

#[tokio::main]
async fn main() {
    let code = run().await;
    // Exiting closes the Job Object handle; Windows then ends anything left
    // (ffmpeg readers the core started, for one).
    std::process::exit(code);
}
