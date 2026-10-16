//! Replays a recorded cumulus3d live stream through the model server's real
//! publishing path (engine consumer -> `ChunkLog` -> gRPC `SubscribeChunks`), so
//! skylens_core and the board can be tested end to end without a GPU.
//!
//! ```text
//! SKYLENS_REST_ADDR=0.0.0.0:8100 SKYLENS_GRPC_ADDR=0.0.0.0:8101 \
//! SKYLENS_WORK_DIR=/tmp/skylens-replay \
//! cargo run --release -p skylens-model --example delta_replay -- \
//!     --stream <dir with manifest.json + packs/> [--speed 10] [--start-after 0] [--hold]
//! ```
//!
//! Ports and engine knobs come from the same `SKYLENS_*` variables as the real
//! server (`utils::serving_config`): REST `SKYLENS_REST_ADDR` (default
//! 0.0.0.0:8100, `/health`), gRPC `SKYLENS_GRPC_ADDR` (default 0.0.0.0:8101,
//! `SubscribeChunks`), `SKYLENS_WORK_DIR`. Demo
//! mode, the UNet and the GPU are forced off; the engine anchor is the stream's
//! `origin_gps`.
//!
//! Options: `--speed N` paces events by manifest `t` / N (default 10, 0 = no
//! pacing), `--start-after S` waits S seconds before the first event (for the
//! waiting-screen), `--hold` keeps serving after the last event (otherwise the
//! process exits shortly after it), `--check` only parses the stream, pushes it
//! through an engine as fast as possible and prints timings.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use skylens_model::app::{build_engine, serve_with};
use skylens_model::models::skylens::Engine;
use skylens_model::models::skylens::replay::{ReplayEvent, load_events, read_manifest, to_event};
use skylens_model::utils::serving_config::Settings;
use skylens_model::utils::serving_pipeline::LiveEngine;
use pointblitz_io::slots;
use skylens_shared::geo::Gps;
use skylens_shared::proto::model::v1::ChunkKind;

struct Args {
    stream: PathBuf,
    speed: f64,
    start_after: f64,
    hold: bool,
    check: bool,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        stream: PathBuf::new(),
        speed: 10.0,
        start_after: 0.0,
        hold: false,
        check: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().with_context(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--stream" => a.stream = PathBuf::from(val("--stream")?),
            "--speed" => a.speed = val("--speed")?.parse().context("--speed")?,
            "--start-after" => {
                a.start_after = val("--start-after")?.parse().context("--start-after")?
            }
            "--hold" => a.hold = true,
            "--check" => a.check = true,
            other => bail!("unknown argument {other:?}"),
        }
    }
    if a.stream.as_os_str().is_empty() {
        bail!("--stream DIR is required");
    }
    Ok(a)
}

fn settings_for(origin: [f64; 3]) -> Settings {
    let mut s = Settings::from_env();
    s.demo = false;
    s.checkpoint = None;
    s.gpu = Some(false);
    s.anchor = Gps {
        lat: origin[0],
        lon: origin[1],
        alt: origin[2],
    };
    s
}

fn feed(engine: &Engine, events: &[ReplayEvent], origin: [f64; 3], speed: f64, start_after: f64) {
    std::thread::sleep(Duration::from_secs_f64(start_after.max(0.0)));
    let t0 = Instant::now();
    for ev in events {
        if speed > 0.0 {
            let due = Duration::from_secs_f64(ev.t / speed);
            if let Some(wait) = due.checked_sub(t0.elapsed()) {
                std::thread::sleep(wait);
            }
        }
        match engine.inject_event(to_event(ev, origin)) {
            Ok(()) => tracing::info!(
                "replay: event {} ({:?}, zone {}): +{} ~{} -{} -> {} points",
                ev.seq,
                ev.kind,
                ev.zone,
                ev.delta.adds.len(),
                ev.delta.moves.len(),
                ev.delta.removes.len(),
                ev.delta.total_points
            ),
            Err(e) => {
                tracing::error!("replay stopped: {e:#}");
                return;
            }
        }
    }
    tracing::info!("replay: all {} events injected", events.len());
}

fn run_check(args: &Args) -> Result<()> {
    let t = Instant::now();
    let m = read_manifest(&args.stream)?;
    let events = load_events(&args.stream, &m)?;
    let parse = t.elapsed();
    let adds: usize = events.iter().map(|e| e.delta.adds.len()).sum();
    println!(
        "parsed {} events ({adds} adds, final {} points) in {parse:.2?}",
        events.len(),
        events.last().map_or(0, |e| e.delta.total_points)
    );
    let settings = settings_for(m.origin_gps);
    let work = std::env::temp_dir().join(format!("skylens-replay-check-{}", std::process::id()));
    let mut settings = settings;
    settings.work_dir = work.clone();
    let engine = build_engine(&settings)?;
    let mut rx = engine.subscribe(0);
    let t = Instant::now();
    for e in &events {
        engine.inject_event(to_event(e, m.origin_gps))?;
    }
    let (mut n, mut bytes) = (0usize, 0usize);
    loop {
        match rx.try_recv() {
            Ok(c) => {
                n += 1;
                bytes += c.payload.len();
                if c.kind == ChunkKind::PointSlots as i32 && c.version as usize == events.len() {
                    slots::validate(&c.payload)?;
                    break;
                }
            }
            Err(_) if t.elapsed() > Duration::from_secs(600) => {
                bail!("timed out waiting for the final chunk")
            }
            Err(_) => std::thread::sleep(Duration::from_millis(5)),
        }
    }
    println!(
        "published {n} chunks ({:.1} MB) in {:.2?}",
        bytes as f64 / 1e6,
        t.elapsed()
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(&work);
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = parse_args()?;
    if args.check {
        return run_check(&args);
    }
    let m = read_manifest(&args.stream)?;
    let t = Instant::now();
    let events = load_events(&args.stream, &m)?;
    tracing::info!(
        "replay: {} events loaded in {:.2?} from {}",
        events.len(),
        t.elapsed(),
        args.stream.display()
    );
    let settings = settings_for(m.origin_gps);
    let engine = Arc::new(build_engine(&settings)?);
    let feeder = {
        let engine = Arc::clone(&engine);
        let (origin, speed, start_after, hold) =
            (m.origin_gps, args.speed, args.start_after, args.hold);
        std::thread::Builder::new()
            .name("delta-replay".into())
            .spawn(move || {
                feed(&engine, &events, origin, speed, start_after);
                if !hold {
                    // Let subscribers drain the last chunks, then end the process.
                    std::thread::sleep(Duration::from_secs(5));
                    std::process::exit(0);
                }
            })?
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let out = runtime.block_on(serve_with(settings, LiveEngine::Ready(engine)));
    drop(feeder);
    out
}
