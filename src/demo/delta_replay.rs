//! Replays a recorded cumulus3d live stream (format v4) through the real core path.
//!
//!     cargo run -p skylens-demo --bin delta-replay -- --stream <dir> [--listen 127.0.0.1:8101]
//!         [--core http://127.0.0.1:8080] [--anchor lat,lon[,alt]] [--speed 1]
//!
//! The tool impersonates the model server: it serves `skylens.model.v1.ModelService` and the
//! core, started with `SKYLENS_MODEL_GRPC=<listen>`, subscribes to it like to the real one.
//! The first `SubscribeChunks` starts the clock; each manifest event is then sent as a
//! `CHUNK_KIND_POINTS_DELTA` chunk at `t / speed` seconds. Every replay carries a new run id
//! (unix seconds), so boards drop the previous mission and start from an empty buffer.
//!
//! `stream/` holds `manifest.json` and `packs/e_NNNN.bin` (adds, then moves, then removes;
//! 20 B / 20 B / 4 B little-endian records). The manifest frame is ENU about `origin_gps`;
//! the core's frame is ENU about its site anchor. When the two differ (`--anchor`, or the
//! anchor read from `--core`'s `/site`) every point is shifted by the offset between them,
//! as the model server does by pinning the engine origin at the core anchor. Without either,
//! the anchor is taken to be `origin_gps` (start the core with `SKYLENS_CORE_ANCHOR=lat,lon,alt`).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use skylens_shared::delta;
use skylens_shared::geo::{Gps, gps_to_enu};
use skylens_shared::proto::model::v1 as pb;
use skylens_shared::proto::model::v1::model_service_server::{ModelService, ModelServiceServer};
use skylens_shared::proto::payload::v1::{DeltaKind, PointsDelta};
use tokio::sync::Mutex;
use tonic::codegen::BoxStream;
use tonic::{Status, Streaming};

const MAX_MSG: usize = 256 << 20;

#[derive(Deserialize)]
struct Manifest {
    version: u32,
    #[serde(default)]
    frame: String,
    origin_gps: [f64; 3],
    capacity: u32,
    events: Vec<Event>,
}

#[derive(Deserialize, Clone)]
struct Event {
    seq: u32,
    t: f64,
    zone: u32,
    kind: String,
    points: u32,
    pack: Option<String>,
    n_add: usize,
    n_move: usize,
    n_remove: usize,
}

struct Replay {
    dir: PathBuf,
    manifest: Manifest,
    /// ENU offset (e, n, u) added to every point: origin_gps expressed in the core anchor frame.
    shift: [f32; 3],
    speed: f64,
    run: u32,
    start: Mutex<Option<Instant>>,
}

impl Replay {
    fn build_delta(&self, ev: &Event) -> Result<PointsDelta> {
        let (mut adds, mut moves, removes) = match &ev.pack {
            Some(p) => {
                let buf = std::fs::read(self.dir.join(p)).with_context(|| format!("reading {p}"))?;
                let (a, m, r) = (ev.n_add * delta::RECORD, ev.n_move * delta::RECORD, ev.n_remove * delta::REMOVE_RECORD);
                if buf.len() != a + m + r {
                    bail!("{p}: {} bytes, manifest says {}", buf.len(), a + m + r);
                }
                (buf[..a].to_vec(), buf[a..a + m].to_vec(), buf[a + m..].to_vec())
            }
            None => (Vec::new(), Vec::new(), Vec::new()),
        };
        shift_records(&mut adds, self.shift);
        shift_records(&mut moves, self.shift);
        let kind = match ev.kind.as_str() {
            "refined" => DeltaKind::Refined,
            "replace" => DeltaKind::Replace,
            "final" => DeltaKind::Final,
            other => bail!("unknown event kind {other:?}"),
        };
        let d = PointsDelta {
            seq: ev.seq,
            kind: kind as i32,
            zone: ev.zone,
            capacity: self.manifest.capacity,
            total_points: ev.points,
            snapshot: false,
            t_s: ev.t,
            adds,
            moves,
            removes,
            run: self.run,
            ..Default::default()
        };
        d.validate().map_err(|e| anyhow::anyhow!("event {}: {e}", ev.seq))?;
        Ok(d)
    }
}

/// Adds `shift` to the xyz of every 20 B record (f32 LE at bytes 4..16).
fn shift_records(buf: &mut [u8], shift: [f32; 3]) {
    if shift == [0.0; 3] {
        return;
    }
    for rec in buf.chunks_exact_mut(delta::RECORD) {
        for (i, s) in shift.iter().enumerate() {
            let o = 4 + i * 4;
            let v = f32::from_le_bytes([rec[o], rec[o + 1], rec[o + 2], rec[o + 3]]) + s;
            rec[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
}

struct Server(Arc<Replay>);

#[tonic::async_trait]
impl ModelService for Server {
    async fn submit_recon(&self, _: tonic::Request<pb::ReconJobRequest>) -> Result<tonic::Response<pb::JobAccepted>, Status> {
        Err(Status::unimplemented("delta-replay serves the live chunk stream only"))
    }
    async fn submit_detect(&self, _: tonic::Request<pb::DetectJobRequest>) -> Result<tonic::Response<pb::JobAccepted>, Status> {
        Err(Status::unimplemented("delta-replay serves the live chunk stream only"))
    }
    async fn get_job(&self, _: tonic::Request<pb::GetJobRequest>) -> Result<tonic::Response<pb::JobStatus>, Status> {
        Err(Status::not_found("no jobs"))
    }
    async fn health(&self, _: tonic::Request<pb::HealthRequest>) -> Result<tonic::Response<pb::HealthResponse>, Status> {
        Ok(tonic::Response::new(pb::HealthResponse { ok: true, gpu: false, version: "delta-replay".into(), gpu_error: String::new() }))
    }
    async fn push_frames(&self, request: tonic::Request<Streaming<pb::FrameSet>>) -> Result<tonic::Response<pb::PushFramesSummary>, Status> {
        let mut s = request.into_inner();
        let mut ignored = 0;
        while s.message().await?.is_some() {
            ignored += 1;
        }
        Ok(tonic::Response::new(pb::PushFramesSummary { ingested: 0, ignored, pending: 0 }))
    }

    type SubscribeChunksStream = BoxStream<pb::Chunk>;

    async fn subscribe_chunks(&self, request: tonic::Request<pb::SubscribeRequest>) -> Result<tonic::Response<Self::SubscribeChunksStream>, Status> {
        let after = request.into_inner().after_sequence;
        let rp = self.0.clone();
        let start = *rp.start.lock().await.get_or_insert_with(|| {
            eprintln!("[replay] core subscribed: clock started (run {}, speed x{})", rp.run, rp.speed);
            Instant::now()
        });
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        tokio::spawn(async move {
            for ev in rp.manifest.events.iter().filter(|e| u64::from(e.seq) > after) {
                tokio::time::sleep_until((start + Duration::from_secs_f64(ev.t / rp.speed)).into()).await;
                let rp2 = rp.clone();
                let ev2 = ev.clone();
                let chunk = match tokio::task::spawn_blocking(move || rp2.build_delta(&ev2).map(|d| delta::to_chunk(u64::from(ev2.seq), &d, 0))).await {
                    Ok(Ok(c)) => c,
                    Ok(Err(e)) => {
                        let _ = tx.send(Err(Status::internal(format!("{e:#}")))).await;
                        return;
                    }
                    Err(e) => {
                        let _ = tx.send(Err(Status::internal(e.to_string()))).await;
                        return;
                    }
                };
                eprintln!(
                    "[replay] t={:>7.2}s seq {:>2} zone {} {:<7} +{} ~{} -{} -> {} points ({} MB)",
                    ev.t, ev.seq, ev.zone, ev.kind, ev.n_add, ev.n_move, ev.n_remove, ev.points, chunk.payload.len() >> 20
                );
                if tx.send(Ok(chunk)).await.is_err() {
                    return; // core went away; a reconnect resumes from after_sequence
                }
            }
            eprintln!("[replay] done");
            // Keep the stream open: ending it makes the core resubscribe.
            tx.closed().await;
        });
        Ok(tonic::Response::new(Box::pin(tokio_stream_from(rx))))
    }
}

fn tokio_stream_from<T: Send + 'static>(mut rx: tokio::sync::mpsc::Receiver<T>) -> impl futures_util::Stream<Item = T> {
    futures_util::stream::poll_fn(move |cx| rx.poll_recv(cx))
}

struct Args {
    stream: PathBuf,
    listen: String,
    core: Option<String>,
    anchor: Option<Gps>,
    speed: f64,
}

fn parse_gps(s: &str) -> Result<Gps> {
    let v: Vec<f64> = s.split(',').map(|p| p.trim().parse::<f64>()).collect::<Result<_, _>>().with_context(|| format!("{s:?}: want lat,lon[,alt]"))?;
    match v[..] {
        [lat, lon] => Ok(Gps { lat, lon, alt: 0.0 }),
        [lat, lon, alt] => Ok(Gps { lat, lon, alt }),
        _ => bail!("{s:?}: want lat,lon[,alt]"),
    }
}

fn parse_args() -> Result<Args> {
    let mut a = Args { stream: PathBuf::new(), listen: "127.0.0.1:8101".into(), core: None, anchor: None, speed: 1.0 };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--stream" => a.stream = val()?.into(),
            "--listen" => a.listen = val()?,
            "--core" => a.core = Some(val()?),
            "--anchor" => a.anchor = Some(parse_gps(&val()?)?),
            "--speed" => a.speed = val()?.parse().context("--speed")?,
            "-h" | "--help" => {
                println!("delta-replay --stream <dir> [--listen 127.0.0.1:8101] [--core http://127.0.0.1:8080] [--anchor lat,lon[,alt]] [--speed 1]");
                std::process::exit(0);
            }
            other => bail!("unknown flag {other}"),
        }
    }
    if a.stream.as_os_str().is_empty() {
        bail!("--stream <dir> is required");
    }
    if a.speed <= 0.0 {
        bail!("--speed must be positive");
    }
    Ok(a)
}

fn http_json(url: &str) -> Result<serde_json::Value> {
    Ok(serde_json::from_str(&ureq::get(url).timeout(Duration::from_secs(5)).call()?.into_string()?)?)
}

fn core_anchor(core: &str) -> Result<Gps> {
    let v = http_json(&format!("{}/site", core.trim_end_matches('/')))?;
    let a = &v["anchor"];
    Ok(Gps { lat: a["lat"].as_f64().context("anchor.lat")?, lon: a["lon"].as_f64().context("anchor.lon")?, alt: a["alt"].as_f64().unwrap_or(0.0) })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    let text = std::fs::read_to_string(args.stream.join("manifest.json")).with_context(|| format!("{}/manifest.json", args.stream.display()))?;
    let manifest: Manifest = serde_json::from_str(&text).context("manifest.json")?;
    if manifest.version != 4 {
        bail!("manifest version {} (want 4)", manifest.version);
    }
    if !manifest.frame.is_empty() && manifest.frame != "enu" {
        bail!("manifest frame {:?} (want enu)", manifest.frame);
    }
    let origin = Gps { lat: manifest.origin_gps[0], lon: manifest.origin_gps[1], alt: manifest.origin_gps[2] };

    let anchor = match (args.anchor, &args.core) {
        (Some(a), _) => Some(a),
        (None, Some(core)) => {
            let c = core.clone();
            Some(tokio::task::spawn_blocking(move || core_anchor(&c)).await??)
        }
        _ => None,
    };
    let shift = match anchor {
        Some(a) => {
            let e = gps_to_enu(origin, a);
            [e.e as f32, e.n as f32, e.u as f32]
        }
        None => [0.0; 3],
    };
    let run = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as u32;
    eprintln!(
        "[replay] {} events, capacity {}, origin {:.6},{:.6},{:.1}; core anchor {}; shift ENU ({:.2}, {:.2}, {:.2}) m; run {run}",
        manifest.events.len(), manifest.capacity, origin.lat, origin.lon, origin.alt,
        anchor.map_or("= origin (start the core with SKYLENS_CORE_ANCHOR)".to_string(), |a| format!("{:.6},{:.6},{:.1}", a.lat, a.lon, a.alt)),
        shift[0], shift[1], shift[2],
    );

    let replay = Arc::new(Replay { dir: args.stream.clone(), manifest, shift, speed: args.speed, run, start: Mutex::new(None) });
    let addr = args.listen.parse().with_context(|| format!("--listen {}", args.listen))?;
    eprintln!("[replay] model-server stand-in on {addr}; start the core with SKYLENS_MODEL_GRPC={}", args.listen);
    let svc = ModelServiceServer::new(Server(replay)).max_decoding_message_size(MAX_MSG).max_encoding_message_size(MAX_MSG);
    tonic::transport::Server::builder().add_service(svc).serve_with_shutdown(addr, async { let _ = tokio::signal::ctrl_c().await; }).await?;
    Ok(())
}
