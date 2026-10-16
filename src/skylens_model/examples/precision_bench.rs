//! SkyLensNet precision benchmark: fp32 vs fp16 autocast vs bf16 on one CUDA GPU.
//!
//! ```text
//! cargo run --release -p skylens-model --example precision_bench -- \
//!     --checkpoint runs/skylens/final --size 1920x1080 --warmup 5 --iters 30
//! ```
//!
//! Each precision runs in its own child process (this binary re-executed with
//! `--child`), so peak GPU memory is not polluted by the previous precision's
//! caching allocator and a hard failure (an unsupported bf16 kernel on Volta)
//! only loses that row. The parent compares every precision with fp32 and
//! prints one table. Without `--checkpoint` (and without `SKYLENS_CHECKPOINT`)
//! a full-size, randomly initialised model with the deployment architecture
//! (resnet-50 encoder, UNet decoder 256/128/64/32, offset head) is built;
//! latency and memory are then real, agreement numbers are only indicative.
//!
//! Options: `--checkpoint DIR`, `--size WxH` (default 1920x1080, the demo
//! footage), `--image FILE` (use a real frame instead of the synthetic one,
//! its size wins), `--thermal` (synthetic thermal plane), `--warmup N` (5),
//! `--iters N` (30), `--device N` (0), `--precisions fp32,fp16,bf16`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use skylens_model::detect::{Detector, FrameDetections, HAZARD_CLASSES, PersonPoint};
use skylens_model::models::skylensnet::configuration_skylensnet::SkyLensConfig;
use skylens_model::models::skylensnet::modeling_skylensnet::{
    ArchSpec, CONFIG_NAME, DetectorOptions, Precision, ResNetSpec, SkyLensForDisasterPerception,
    SkyLensNetDetector, WEIGHTS_NAME, cuda_compute_capability, precision_for_capability,
};
use tch::{Device, nn};

/// Pixels within this distance count as the same person across precisions.
const MATCH_RADIUS: f32 = 8.0;

struct Args {
    checkpoint: Option<PathBuf>,
    size: (u32, u32),
    image: Option<PathBuf>,
    thermal: bool,
    warmup: usize,
    iters: usize,
    device: usize,
    precisions: Vec<Precision>,
    /// Internal: run one precision and write the result under `out`.
    child: Option<Precision>,
    out: Option<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        checkpoint: std::env::var_os("SKYLENS_CHECKPOINT")
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty()),
        size: (1920, 1080),
        image: None,
        thermal: false,
        warmup: 5,
        iters: 30,
        device: 0,
        precisions: vec![Precision::Fp32, Precision::Fp16Autocast, Precision::Bf16],
        child: None,
        out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| anyhow!("{name} needs a value"));
        match flag.as_str() {
            "--checkpoint" => a.checkpoint = Some(PathBuf::from(value("--checkpoint")?)),
            "--size" => {
                let v = value("--size")?;
                let (w, h) = v.split_once('x').ok_or_else(|| anyhow!("--size is WxH"))?;
                a.size = (w.parse()?, h.parse()?);
            }
            "--image" => a.image = Some(PathBuf::from(value("--image")?)),
            "--thermal" => a.thermal = true,
            "--warmup" => a.warmup = value("--warmup")?.parse()?,
            "--iters" => a.iters = value("--iters")?.parse()?,
            "--device" => a.device = value("--device")?.parse()?,
            "--precisions" => {
                a.precisions = value("--precisions")?
                    .split(',')
                    .map(|p| p.parse::<Precision>().map_err(|e| anyhow!(e)))
                    .collect::<Result<_>>()?;
            }
            "--child" => {
                a.child = Some(
                    value("--child")?
                        .parse::<Precision>()
                        .map_err(|e| anyhow!(e))?,
                )
            }
            "--out" => a.out = Some(PathBuf::from(value("--out")?)),
            "-h" | "--help" => {
                println!("see the module docs of examples/precision_bench.rs");
                std::process::exit(0);
            }
            other => bail!("unknown argument {other}"),
        }
    }
    if a.iters == 0 {
        bail!("--iters must be positive");
    }
    Ok(a)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    match args.child {
        Some(p) => child(&args, p),
        None => parent(&args),
    }
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

struct Frame {
    width: u32,
    height: u32,
    rgb: Vec<u8>,
    thermal: Option<Vec<u8>>,
}

fn load_frame(args: &Args) -> Result<Frame> {
    if let Some(path) = &args.image {
        let img = image::open(path)
            .with_context(|| format!("reading {}", path.display()))?
            .to_rgb8();
        let (w, h) = img.dimensions();
        let thermal = args
            .thermal
            .then(|| synthetic_thermal(w as usize, h as usize));
        return Ok(Frame {
            width: w,
            height: h,
            rgb: img.into_raw(),
            thermal,
        });
    }
    let (w, h) = (args.size.0 as usize, args.size.1 as usize);
    // Smooth gradients, stripes and a few bright blobs: enough structure for the
    // heads to produce non-trivial maps. Deterministic, so every precision sees the same frame.
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let blob =
                |cx: f32, cy: f32| (-(((fx - cx).powi(2) + (fy - cy).powi(2)) * 400.0)).exp();
            let b = blob(0.3, 0.4) + blob(0.65, 0.55) + blob(0.8, 0.2);
            let base = [
                0.35 + 0.25 * (fx * 9.0).sin() * (fy * 7.0).cos(),
                0.4 + 0.3 * fy,
                0.3 + 0.3 * fx,
            ];
            for (c, v) in base.iter().enumerate() {
                rgb[(y * w + x) * 3 + c] = ((v + 0.5 * b).clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
    }
    let thermal = args.thermal.then(|| synthetic_thermal(w, h));
    Ok(Frame {
        width: w as u32,
        height: h as u32,
        rgb,
        thermal,
    })
}

fn synthetic_thermal(w: usize, h: usize) -> Vec<u8> {
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
            (255.0 * (0.5 + 0.4 * (x * 5.0 + y * 3.0).sin())) as u8
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

/// Full-size random model with the deployment architecture, saved as a `save_pretrained` directory.
fn random_checkpoint(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let config = SkyLensConfig {
        backbone: "microsoft/resnet-50".into(),
        use_pretrained_backbone: false,
        use_offset_head: true,
        ..Default::default()
    };
    let vs = nn::VarStore::new(Device::Cpu);
    let spec = ResNetSpec::for_name(&config.backbone).ok_or_else(|| anyhow!("unknown backbone"))?;
    SkyLensForDisasterPerception::new(
        &vs.root(),
        &config,
        &ArchSpec {
            backbone: spec,
            wh_stem: true,
        },
    )?;
    vs.save(dir.join(WEIGHTS_NAME))?;
    std::fs::write(dir.join(CONFIG_NAME), config.to_json_string()?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Child: one precision
// ---------------------------------------------------------------------------

/// Samples this process' GPU memory (MiB) through `nvidia-smi` while timing runs.
struct MemoryProbe {
    peak: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

fn process_gpu_mib() -> Option<u64> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    let me = std::process::id().to_string();
    String::from_utf8_lossy(&out.stdout).lines().find_map(|l| {
        let (pid, mib) = l.split_once(',')?;
        (pid.trim() == me)
            .then(|| mib.trim().parse().ok())
            .flatten()
    })
}

impl MemoryProbe {
    fn start() -> Self {
        let (peak, stop) = (
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicBool::new(false)),
        );
        let (p, s) = (peak.clone(), stop.clone());
        let handle = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                if let Some(m) = process_gpu_mib() {
                    p.fetch_max(m, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        });
        Self {
            peak,
            stop,
            handle: Some(handle),
        }
    }

    fn finish(mut self) -> Option<u64> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.join().ok();
        }
        // The last sample after the timed loop, in case the thread never ran.
        let last = process_gpu_mib().unwrap_or(0);
        let peak = self.peak.load(Ordering::Relaxed).max(last);
        (peak > 0).then_some(peak)
    }
}

fn child(args: &Args, precision: Precision) -> Result<()> {
    let out = args
        .out
        .clone()
        .ok_or_else(|| anyhow!("--child needs --out"))?;
    let name = precision.to_string();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_precision(args, precision, &out)
    }))
    .unwrap_or_else(|p| {
        let msg = p
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "panic".into());
        Err(anyhow!("panic: {msg}"))
    });
    let summary = match result {
        Ok(v) => v,
        Err(e) => json!({ "error": format!("{e:#}") }),
    };
    std::fs::write(out.join(format!("{name}.json")), summary.to_string())?;
    Ok(())
}

fn run_precision(args: &Args, precision: Precision, out: &Path) -> Result<Value> {
    let checkpoint = args
        .checkpoint
        .as_ref()
        .ok_or_else(|| anyhow!("--checkpoint missing"))?;
    let device = Device::Cuda(args.device);
    let frame = load_frame(args)?;
    let opts = DetectorOptions {
        precision,
        ..Default::default()
    };
    let probe = MemoryProbe::start();
    let t_load = Instant::now();
    let det = SkyLensNetDetector::load_with_options(checkpoint, device, opts)?;
    let load_s = t_load.elapsed().as_secs_f64();
    let thermal = frame.thermal.as_deref();
    let run = || det.detect(&frame.rgb, thermal, frame.width, frame.height);
    for _ in 0..args.warmup {
        run()?;
    }
    let mut times = Vec::with_capacity(args.iters);
    let mut last = None;
    for _ in 0..args.iters {
        let t = Instant::now();
        let d = run()?;
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        last = Some(d);
    }
    let peak_mib = probe.finish();
    let d = last.ok_or_else(|| anyhow!("no iterations"))?;
    let bytes: Vec<u8> = d.hazard_prob.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out.join(format!("{precision}.prob")), bytes)?;
    let people: Vec<[f32; 3]> = d.people.iter().map(|p| [p.x, p.y, p.score]).collect();
    Ok(json!({
        "resolved": det.options().precision.to_string(),
        "load_s": load_s,
        "times_ms": times,
        "peak_mib": peak_mib,
        "width": d.width,
        "height": d.height,
        "people": people,
    }))
}

// ---------------------------------------------------------------------------
// Parent: orchestrate and compare
// ---------------------------------------------------------------------------

struct Row {
    precision: Precision,
    outcome: Result<Run, String>,
}

struct Run {
    times_ms: Vec<f64>,
    load_s: f64,
    peak_mib: Option<u64>,
    detections: FrameDetections,
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * q).ceil() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn read_run(out: &Path, precision: Precision) -> Result<Run, String> {
    let json_path = out.join(format!("{precision}.json"));
    let text = std::fs::read_to_string(&json_path)
        .map_err(|_| "child produced no result (crashed)".to_string())?;
    let v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    if let Some(e) = v.get("error").and_then(Value::as_str) {
        return Err(e.to_string());
    }
    let raw = std::fs::read(out.join(format!("{precision}.prob"))).map_err(|e| e.to_string())?;
    let hazard_prob: Vec<f32> = raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    let people = v["people"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|p| PersonPoint {
                    x: p[0].as_f64().unwrap_or(0.0) as f32,
                    y: p[1].as_f64().unwrap_or(0.0) as f32,
                    score: p[2].as_f64().unwrap_or(0.0) as f32,
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(Run {
        times_ms: v["times_ms"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default(),
        load_s: v["load_s"].as_f64().unwrap_or(0.0),
        peak_mib: v["peak_mib"].as_u64(),
        detections: FrameDetections {
            width: v["width"].as_u64().unwrap_or(0) as u32,
            height: v["height"].as_u64().unwrap_or(0) as u32,
            hazard_prob,
            people,
        },
    })
}

struct Agreement {
    max_abs: f32,
    mean_abs: f64,
    argmax_pct: f64,
    people: usize,
    matched: usize,
    mean_pos: f64,
}

fn compare(reference: &FrameDetections, other: &FrameDetections) -> Agreement {
    let (mut max_abs, mut sum) = (0.0f32, 0.0f64);
    for (a, b) in reference.hazard_prob.iter().zip(&other.hazard_prob) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum += d as f64;
    }
    let n = reference
        .hazard_prob
        .len()
        .min(other.hazard_prob.len())
        .max(1);
    let px = n / HAZARD_CLASSES;
    let argmax = |p: &[f32], i: usize| {
        (0..HAZARD_CLASSES)
            .max_by(|&a, &b| p[a * px + i].total_cmp(&p[b * px + i]))
            .unwrap_or(0)
    };
    let same = (0..px)
        .filter(|&i| argmax(&reference.hazard_prob, i) == argmax(&other.hazard_prob, i))
        .count();

    // Greedy one-to-one match of the other run's people to the reference, by score.
    let mut order: Vec<&PersonPoint> = reference.people.iter().collect();
    order.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut used = vec![false; other.people.len()];
    let (mut matched, mut dist) = (0usize, 0.0f64);
    for r in order {
        let best = other
            .people
            .iter()
            .enumerate()
            .filter(|(i, _)| !used[*i])
            .map(|(i, o)| (i, ((o.x - r.x).powi(2) + (o.y - r.y).powi(2)).sqrt()))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((i, d)) = best
            && d <= MATCH_RADIUS
        {
            used[i] = true;
            matched += 1;
            dist += d as f64;
        }
    }
    Agreement {
        max_abs,
        mean_abs: sum / n as f64,
        argmax_pct: 100.0 * same as f64 / px as f64,
        people: other.people.len(),
        matched,
        mean_pos: if matched > 0 {
            dist / matched as f64
        } else {
            f64::NAN
        },
    }
}

fn gpu_name(device: usize) -> String {
    Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
            "-i",
            &device.to_string(),
        ])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown (nvidia-smi not found)".into())
}

fn parent(args: &Args) -> Result<()> {
    if !tch::Cuda::is_available() {
        bail!(
            "CUDA is not available to libtorch (no GPU, or the CPU libtorch was linked); this benchmark needs a GPU"
        );
    }
    let cc = cuda_compute_capability(args.device as i32);
    let (auto, reason) = precision_for_capability(cc);
    let out =
        std::env::temp_dir().join(format!("skylensnet-precision-bench-{}", std::process::id()));
    std::fs::create_dir_all(&out)?;

    let (checkpoint, random) = match &args.checkpoint {
        Some(p) => (p.clone(), false),
        None => {
            let dir = out.join("random-checkpoint");
            println!(
                "no checkpoint given: building a randomly initialised resnet-50 SkyLensNet at {}",
                dir.display()
            );
            random_checkpoint(&dir)?;
            (dir, true)
        }
    };
    let frame = load_frame(args)?;
    let exe = std::env::current_exe()?;

    println!("gpu:        {}", gpu_name(args.device));
    println!(
        "capability: {}",
        cc.map_or("unknown".into(), |(a, b)| format!("{a}.{b}"))
    );
    println!("auto picks: {auto} ({reason})");
    println!(
        "checkpoint: {}{}",
        checkpoint.display(),
        if random {
            " (random weights: agreement is indicative only)"
        } else {
            ""
        }
    );
    println!(
        "frame:      {}x{} {}, warmup {}, timed {}",
        frame.width,
        frame.height,
        if frame.thermal.is_some() {
            "RGB+thermal"
        } else {
            "RGB"
        },
        args.warmup,
        args.iters
    );
    drop(frame);

    let mut rows = Vec::new();
    for &precision in &args.precisions {
        println!("-- running {precision} ...");
        let mut cmd = Command::new(&exe);
        cmd.arg("--child")
            .arg(precision.to_string())
            .arg("--out")
            .arg(&out)
            .arg("--checkpoint")
            .arg(&checkpoint)
            .args(["--size", &format!("{}x{}", args.size.0, args.size.1)])
            .args([
                "--warmup",
                &args.warmup.to_string(),
                "--iters",
                &args.iters.to_string(),
            ])
            .args(["--device", &args.device.to_string()]);
        if let Some(img) = &args.image {
            cmd.arg("--image").arg(img);
        }
        if args.thermal {
            cmd.arg("--thermal");
        }
        let status = cmd.status()?;
        let mut outcome = read_run(&out, precision);
        if let (Err(e), false) = (&mut outcome, status.success()) {
            *e = format!("{e}; exit status {status}");
        }
        rows.push(Row { precision, outcome });
    }

    println!();
    println!(
        "{:<14} {:>9} {:>9} {:>9} {:>10} {:>8}",
        "precision", "median ms", "p90 ms", "fps", "peak MiB", "load s"
    );
    for r in &rows {
        match &r.outcome {
            Ok(run) => {
                let mut t = run.times_ms.clone();
                t.sort_by(f64::total_cmp);
                let med = percentile(&t, 0.5);
                println!(
                    "{:<14} {:>9.1} {:>9.1} {:>9.1} {:>10} {:>8.1}",
                    r.precision.to_string(),
                    med,
                    percentile(&t, 0.9),
                    1000.0 / med,
                    run.peak_mib.map_or("n/a".into(), |m| m.to_string()),
                    run.load_s
                );
            }
            Err(e) => println!("{:<14} FAILED: {e}", r.precision.to_string()),
        }
    }

    let reference = rows
        .iter()
        .find(|r| r.precision == Precision::Fp32)
        .and_then(|r| r.outcome.as_ref().ok());
    println!();
    match reference {
        None => println!("no fp32 result: agreement not computed"),
        Some(base) => {
            println!(
                "agreement with fp32 (fp32 found {} people; match radius {MATCH_RADIUS} px)",
                base.detections.people.len()
            );
            println!(
                "{:<14} {:>11} {:>12} {:>10} {:>8} {:>8} {:>11}",
                "precision",
                "max |dp|",
                "mean |dp|",
                "argmax %",
                "people",
                "matched",
                "mean pos px"
            );
            for r in rows.iter().filter(|r| r.precision != Precision::Fp32) {
                if let Ok(run) = &r.outcome {
                    let a = compare(&base.detections, &run.detections);
                    println!(
                        "{:<14} {:>11.5} {:>12.7} {:>10.3} {:>8} {:>8} {:>11.3}",
                        r.precision.to_string(),
                        a.max_abs,
                        a.mean_abs,
                        a.argmax_pct,
                        a.people,
                        a.matched,
                        a.mean_pos
                    );
                }
            }
        }
    }
    std::fs::remove_dir_all(&out).ok();
    Ok(())
}
