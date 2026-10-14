//! End-to-end demo launcher: every component, for real, on one machine.
//!
//!     cargo run -p skylens-demo -- [--native] [--auto] [--no-model] ...
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
mod http1;
mod paths;
mod preflight;
mod procs;
mod route;
mod sim;

use std::future::Future;
use std::path::Path;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use args::Options;
use preflight::Plan;
use procs::{Event, Events, Running, Spec, log};

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
    }
}

fn kv(k: &str, v: impl ToString) -> (String, String) {
    (k.to_string(), v.to_string())
}

/// Environment of the Go core.
fn core_env(plan: &Plan, o: &Options) -> Vec<(String, String)> {
    let mut env = vec![
        kv("SKYLENS_CORE_ADDR", format!(":{}", o.core_port)),
        kv("SKYLENS_CORE_WEB_DIR", plan.control_dist.display()),
        kv("SKYLENS_MODEL_GRPC", format!("127.0.0.1:{}", args::MODEL_GRPC_PORT)),
        kv("SKYLENS_CORE_FRAME_SOURCE", if o.no_model { "off" } else { "ffmpeg" }),
        kv("SKYLENS_CORE_FFMPEG", plan.ffmpeg.display()),
        kv("SKYLENS_CORE_STREAM_PATH", "drone{id}"),
        kv("SKYLENS_CORE_FRAME_URL", format!("srt://127.0.0.1:{}?streamid=read:{{path}}", args::SRT_PORT)),
        kv("SKYLENS_CORE_FRAME_FPS", o.fps),
        kv("SKYLENS_CORE_FRAME_WIDTH", o.frame_width),
        kv("SKYLENS_CORE_DRONE_ETA_SEC", sim::ETA_SECS),
    ];
    if let Some(v) = &plan.vworld_env {
        env.push(kv("SKYLENS_VWORLD_ENV", v.display()));
    }
    env
}

/// Environment of the model server: listeners, checkpoint, libtorch DLLs on the search path.
fn model_env(m: &preflight::ModelPlan) -> Vec<(String, String)> {
    let mut env = vec![
        kv("SKYLENS_CHECKPOINT", m.checkpoint.display()),
        kv("SKYLENS_REST_ADDR", format!("127.0.0.1:{}", args::MODEL_REST_PORT)),
        kv("SKYLENS_GRPC_ADDR", format!("127.0.0.1:{}", args::MODEL_GRPC_PORT)),
    ];
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
    events: &Events,
    rx: &mut mpsc::UnboundedReceiver<Event>,
    children: &mut Vec<Running>,
    tasks: &mut Vec<JoinHandle<()>>,
) -> Result<(), Stop> {
    let fail = |e: anyhow::Error| Stop::Failed(format!("{e:#}"));

    // ---- MediaMTX ----
    std::fs::create_dir_all(&plan.run_dir).map_err(|e| Stop::Failed(format!("{}: {e}", plan.run_dir.display())))?;
    let deploy = std::fs::read_to_string(&plan.mediamtx_deploy_config)
        .map_err(|e| Stop::Failed(format!("{}: {e}", plan.mediamtx_deploy_config.display())))?;
    let config = plan.run_dir.join("mediamtx.yml");
    std::fs::write(&config, paths::local_mediamtx_config(&deploy)).map_err(|e| Stop::Failed(format!("{}: {e}", config.display())))?;
    log("launch", &format!("MediaMTX 시작 ({})", config.display()));
    children.push(
        procs::spawn(spec(plan, "mediamtx", "mtx", &plan.mediamtx, vec![config.display().to_string()], vec![]), events.clone())
            .map_err(fail)?,
    );
    let whep = format!("127.0.0.1:{}", args::WHEP_PORT);
    wait_ready("MediaMTX", Duration::from_secs(20), rx, || http1::listening(&whep)).await?;

    // ---- footage publishers ----
    for (drone, file) in paths::FORMATION.iter().zip(&plan.footage) {
        let tag = format!("pub-{}", &drone.name[..1].to_uppercase());
        let name = format!("송출 drone{} ({})", drone.id, drone.name);
        let argv = paths::publisher_args(file, args::SRT_PORT, drone);
        children.push(procs::spawn(spec(plan, &name, &tag, &plan.ffmpeg, argv, vec![]), events.clone()).map_err(fail)?);
    }
    settle(Duration::from_millis(1500), rx).await?;
    log("launch", "영상 송출 3개 시작 (drone1=left, drone2=center, drone3=right, 반복 재생)");

    // ---- model server ----
    if let Some(m) = &plan.model {
        log("launch", "모델 서버 시작 (libtorch·CUDA 로딩에 시간이 걸릴 수 있습니다)");
        children.push(procs::spawn(spec(plan, "skylens-model", "model", &m.bin, vec![], model_env(m)), events.clone()).map_err(fail)?);
        let rest = format!("127.0.0.1:{}", args::MODEL_REST_PORT);
        wait_ready("모델 서버", Duration::from_secs(300), rx, || http1::ok(&rest, "/health")).await?;
    } else {
        log("launch", "--no-model: 모델 서버 없이 진행 (프레임 추출 끔)");
    }

    // ---- core ----
    log("launch", "코어 시작");
    children.push(procs::spawn(spec(plan, "skylens-core", "core", &plan.core_bin, vec![], core_env(plan, o)), events.clone()).map_err(fail)?);
    let core = format!("127.0.0.1:{}", o.core_port);
    wait_ready("코어", Duration::from_secs(60), rx, || http1::ok(&core, "/health")).await?;

    // ---- board server ----
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", o.board_port))
        .await
        .map_err(|e| Stop::Failed(format!("현황판 서버 :{}: {e}", o.board_port)))?;
    let app = board::router(board::BoardConfig { dist: plan.client_dist.clone(), core: core.clone(), whep: whep.clone() });
    tasks.push(task("board", events, async move { axum::serve(listener, app).await.map_err(anyhow::Error::from) }));
    let board_addr = format!("127.0.0.1:{}", o.board_port);
    wait_ready("현황판 서버", Duration::from_secs(10), rx, || http1::ok(&board_addr, "/health")).await?;

    // ---- drones and the control watcher ----
    tasks.push(task("drone", events, sim::run_drones(core.clone(), o.follow_route)));
    tasks.push(task("tower-watch", events, sim::run_control_watch(core.clone(), o.auto)));

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

fn banner(plan: &Plan, o: &Options) {
    let line = "  ─────────────────────────────────────────────────────────────";
    println!();
    println!("  SkyLens 데모");
    println!("{line}");
    println!("  관제탑   http://localhost:{}/          (코어가 서빙)", o.core_port);
    println!("  현황판   http://localhost:{}/          (게이트웨이 역할, 런처 내장 서버)", o.board_port);
    if plan.native.is_some() {
        println!("  네이티브 관제탑 -> ws://127.0.0.1:{}/control, 현황판 -> ws://127.0.0.1:{}/viewer", o.core_port, o.board_port);
    }
    if plan.model.is_some() {
        println!("  모델     http://127.0.0.1:{}/health  (gRPC :{})", args::MODEL_REST_PORT, args::MODEL_GRPC_PORT);
    } else {
        println!("  모델     끔 (--no-model)");
    }
    println!("  영상     WHEP http://localhost:{}/live/drone{{1,2,3}}/whep, SRT :{}", o.board_port, args::SRT_PORT);
    println!();
    println!("  시나리오");
    println!("   1. 지금은 정지 상태. 드론 3대는 업링크에 붙어 대기, 복원 없음");
    if o.auto {
        println!("   2. --auto: 3초 뒤 런처가 데모 촬영 경로를 배정");
    } else {
        println!("   2. 관제탑에서 경로 계획으로 경유 지점을 지정하고 배정");
    }
    println!("   3. \"태스크 지정 완료\" 후 드론 연결 대기 (약 {:.0}초, 그동안 드론이 현장으로 이동)", sim::ETA_SECS);
    println!("   4. 드론이 도착해 hello 를 보내면 \"드론 연결됨\", 프레임 추출 시작");
    println!("   5. 구간마다 미리보기 점군·메시가 먼저 뜨고 정제본이 교체");
    println!("   6. 위험구역·사람 탐지 결과가 지도 위에 표시");
    if !o.follow_route {
        println!("   (드론은 촬영본에 맞춘 고정 경로를 날며 텔레메트리를 보냅니다. 배정 경로를 따르게 하려면 --follow-route)");
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
    let stop = match start_all(&plan, &o, &events, &mut rx, &mut children, &mut tasks).await {
        Ok(()) => {
            banner(&plan, &o);
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
