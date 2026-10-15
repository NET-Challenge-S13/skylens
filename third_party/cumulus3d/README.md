English | [한국어](README.ko.md)

# cumulus3d

**cumulus3d** — progressive multi-view 3D reconstruction in Rust: point clouds accumulate like cumulus clouds as video frames arrive.
It is built and validated for **drone-formation aerial video**: three cameras flying in formation (about 30 m altitude, oblique downward views) with GPS.
All default parameters were tuned and measured on this setting; other kinds of video are not validated yet and may need re-tuning.

A Rust workspace that builds 3D point clouds from drone-formation video **progressively**.
While video arrives position by position, it chains feature extraction → matching → SfM → GPS alignment → densification inside a single process,
produces a fast preview (no BA) and a refined result (with BA) for every zone, and stacks per-step snapshots on the refined map (ENU).
Stages hand data to each other through in-memory data structures, with no intermediate database or on-disk model folders.

API doc comments are currently in Korean; English translation is planned.

## Crates

| Crate | Role |
|---|---|
| `crates/core` (`cumulus3d-core`) | Camera models, geometry (Rigid3/Sim3), reconstruction data structures, correspondence graph, feature store, RANSAC, PLY/GPS I/O, model file format (`interop`) |
| `crates/features` (`cumulus3d-features`) | Image reading, EXIF, camera initialization, SIFT detection and descriptors |
| `crates/matching` (`cumulus3d-matching`) | Descriptor matching, pair lists, two-view geometry (E/F/H) estimation and verification |
| `crates/ba` (`cumulus3d-ba`) | Bundle adjustment (trust-region LM, Schur-complement linear solver), absolute pose refinement |
| `crates/sfm` (`cumulus3d-sfm`) | Global SfM (rotation averaging + position estimation), image registration, triangulation |
| `crates/align` (`cumulus3d-align`) | GPS ↔ ENU conversion, model alignment (Umeyama, robust estimation), point cloud re-referencing |
| `crates/dense` (`cumulus3d-dense`) | Undistortion, scene conversion, PatchStereo depth maps, filtering and fusion → dense point cloud |
| `crates/cuda` (`cumulus3d-cuda`) | GPU backends (PatchStereo, descriptor matching, SIFT scale space). Builds on machines without CUDA |
| `crates/cli` (`cumulus3d-cli`) | The `cumulus3d` executable: `cumulus3d stream` and per-stage subcommands |

The full public API map (which function to use for what, and how the crates depend on each other) is in [docs/API.md](docs/API.md).
Each crate's public items are listed in `crates/*/README.md`; executable usage is in `crates/cli/README.md`.

## Build and test

```bash
cargo build --release
cargo test --release --workspace
cargo clippy --workspace --all-targets
```

## Usage

```bash
# Progressive pipeline (input: images/camF|camR|camL/*.jpg + gps_ref.txt)
target/release/cumulus3d stream --src <input folder> --out <output folder>
# Use the GPU backend (CUDA 12.x)
target/release/cumulus3d stream --src <input> --out <output> --gpu
# Same run from a declarative plan file (see below)
target/release/cumulus3d run examples/plans/aerial-formation.toml
# Per-stage subcommands (for existing scripts): feature_extractor, matches_importer, global_mapper, image_registrator,
# point_triangulator, bundle_adjuster, model_aligner, model_analyzer, model_converter, image_deleter,
# image_undistorter, densify
target/release/cumulus3d model_analyzer --path <model folder>
```

## Declarative usage (plan → build → run)

Full reference of every declarative function (plan fields, builder methods, checks, run steps, TOML mapping): [docs/DECLARATIVE.md](docs/DECLARATIVE.md).

The `declare` module provides a declarative API for composing the reconstruction pipeline. A `Plan` assembled through the
builder is a side-effect-free value: it is validated as a whole in `build()` and executed lazily in `run()`, which covers
position-wise ingestion, waiting for background refinement and shutdown.

```rust
use cumulus3d_cli::declare::{Dense, Recon, Sinks};
use cumulus3d_cli::events::{Event, EventKind};

let recon = Recon::declare()
    .input("data")                       // data/images/<camera>/, data/gps_ref.txt
    .preset("aerial-formation")          // camF/camR/camL, formation pairing, stride 3
    .zones(12, 2)
    .dense(Dense::profile("fast").fusion_min_views(3))
    .sinks(Sinks::default_files("out"))
    .seed(7)
    .on(EventKind::PositionDone, |e: &Event| println!("{}", e.timeline_text().unwrap()))
    .on_zone_preview(|zone, cloud| println!("preview {}: {} points", zone.zone, cloud.len()))
    .build()?;                           // checks only; every problem is reported at once (PlanError)
println!("{}", recon.plan().to_toml());  // the plan is a value: compare, clone, save as TOML
let summary = recon.run()?;              // executes here
```

`build()` checks: input and camera folders exist, at least one position is selected, a GPS file exists when ENU alignment is
requested, backend names are known and a CUDA device is present when a CUDA backend (including densification) is requested,
the output folder is writable (and does not contain the input when it is cleaned), and option values are in range.

The plan can be stored as TOML and run from the command line. `cumulus3d stream` itself runs through this layer.

```bash
target/release/cumulus3d plan --print-default > plan.toml   # default plan (aerial-formation preset)
target/release/cumulus3d plan --check plan.toml             # check only
target/release/cumulus3d run plan.toml                      # check, then run
```

An annotated example is in [`examples/plans/aerial-formation.toml`](examples/plans/aerial-formation.toml). Relative paths in a plan
file are resolved against the current working directory.

## Live input (declare state → reconcile)

For input that keeps arriving, `compose` keeps a reconstruction node alive and reconciles each declared frame set against what it
already accepted: new sets are ingested once, re-declarations are no-ops, incomplete sets wait, and late or corrected frames follow
an explicit policy (append-only, replay window, zone invalidation, strict). In-memory frames, a producer thread with backpressure
and a host that addresses nodes by id are included. Reference: [docs/COMPOSE.md](docs/COMPOSE.md).

```rust
use cumulus3d_cli::compose::{CompositionHost, FrameKey, FrameState, LiveFrameSet, ReconNode};

let host = CompositionHost::new();
let recon = ReconNode::declare().images("live/images").cameras(["camF", "camR", "camL"]).zones(12, 2).build(&host)?;
let set = LiveFrameSet::new(FrameKey::new("drone-1", 0, 0))
    .file("camF", "camF/0000.jpg").file("camR", "camR/0000.jpg").file("camL", "camL/0000.jpg");
recon.compose(FrameState::ready(set.clone()))?;  // ingested
recon.compose(FrameState::ready(set))?;          // no-op
let closed = recon.close()?;
```

## Event-driven architecture

The pipeline is split into four layers: **state value + reducer + event + hook**. Internal computation is owned by the state object,
and the outside world only sees a functional endpoint: "feed an input, get a new state and events back".

```
new frame set ──▶ step(&Session, FrameSet) ──▶ (new Session, Vec<Event>)
                                                  │
    Pipeline (queues · policies · panic isolation)┤
          ┌──────────────┬──────────────┬─────────┴────┬──────────────┐
   on(FrameRegistered) on_zone_preview on_zone_refined   on_any        subscribe()
    (closure hook)     (closure hook)  (closure hook)  (default sinks) (channel receiver)
```

### 1. State and reducer (`session`)

| Function | What it does |
|---|---|
| `step(&Session, FrameSet) -> (Session, Vec<Event>)` | Processes the frame set of one position: feature extraction → pair matching → registration and triangulation → preview densification when a zone closes; refinement starts in the background |
| `poll(&Session)` | Collects finished background refinement results as events (non-blocking) |
| `command(&Session, Command)` | `ResetFrom(position)`: rewind to the state just before that position; `InvalidateZone(k)`: rebuild zone k |
| `finish(&Session)` | Closes the remaining zones, waits for all background work, and collects the results |

- `Session` is a value. Models and the feature store are shared via `Arc` and copied only on change, so passing state around is cheap.
- The same input yields the same events (with a fixed seed). Every event carries a monotonically increasing session version.

### 2. Events (`events`)

Each frame set you feed produces results split by stage, as events. Point clouds carry only the changed zone pieces, not the whole cloud every time.

| When | Events |
|---|---|
| Start | `Started` |
| Per position | `FrameDecoded` (per camera image, before processing; with a frame hook) → `FrameIngested` → `FeaturesExtracted` (per image) → `PairsMatched` → `ModelInitialized` (for the first model) → `FrameRegistered` (per image) → `PositionDone` |
| When a zone closes | `ZoneArrived` → `ZonePreview` (preview point cloud, no BA) |
| Background refinement | `RefineStarted` → `ZoneAdjusted` → `ZoneRefinedPose` (pose statistics after BA and GPS alignment) → `ZoneRefined` (refined point cloud, replaces the preview of the same zone) |
| Refined result adopted | `BaseAdopted` (registration continues on top of the refined model from then on) |
| Coordinate frame update | `Reanchored` (only the Sim3 transform of zones already sent), `Snapshot` |
| Command results | `Reset`, `ZoneInvalidated` |
| End | `AllPositionsDone` → `AllRefinedDone` → `Finished` (per-stage timings and final model statistics) |
| Other | `Log`, `Warning`, `Error` |

An edge client can keep its view up to date by handling just three cases: "add (`ZonePreview`) · replace (`ZoneRefined`) · transform (`Reanchored`)".

### 3. Pipeline and hooks (`pipeline`)

- `Pipeline::new(reducer)` wraps a reducer and provides `push(input)` / `poll()` / `command(c)` / `finish()`.
- Hooks: `.on(EventKind, |e| …)`, `.on_any(…)`, and the convenience methods `.on_frame` (decoded input frames, delivered before the position is processed), `.on_zone_preview`, `.on_zone_refined`, `.on_snapshot`,
  `.on_position_done`, `.on_message`. To receive events over a channel, use `.subscribe() -> Receiver<Event>`.
- Execution: asynchronous by default (a worker-thread queue per kind, order preserved within a kind), so slow hooks never block computation. With `.sync(true)` hooks run immediately on the same thread.
- Queue policy `.policy(kind, …)`: `KeepAll` (default), `LatestPerKey` (drops older unprocessed events for the same zone — for refreshing a preview display), `LatestOnly`.
- If a hook panics, the pipeline keeps running and an `Error` event is sent to subscriber channels.
- `finish()` returns `(reducer, Summary)` (event count, counts per kind, hook calls and panics, events dropped by policy).
- Reducers are abstracted by the `pipeline::Reducer` trait, so the same driver can run other state machines instead of `Session`.

### 4. Default output hooks (`sinks`)

A single call, `sinks::attach(pipeline, out, &SinkOptions)`, attaches a set of hooks that produce the same files as `cumulus3d stream`.

| Hook | Output |
|---|---|
| `TimelineSink` | `timeline.txt` (event time + message) |
| `RunLogSink` | `run.log`, `DONE` |
| `ZonePlySink` | `full/{preview,refined}/*.ply` |
| `ModelSink` (`save_models`) | `work/models/…` |
| `SnapshotSink` | `aligned/`, `snapshots/event_NN_*.ply`, `snapshots/manifest.json`, `final_frame/` |

`cumulus3d stream` itself runs on this architecture: it assembles a session plus the default output hooks, calls `push` for every position, then calls `finish`.

### Frame hook: per-frame input before processing

`on_frame` hands each decoded input frame to a closure as soon as a position arrives, **before** feature extraction, matching and
registration run, one camera image at a time. Hooks run on their own worker queue, so downstream consumers (for example an object
detector) work on the frame while reconstruction proceeds. The pixel buffer is shared via `Arc` and is never copied between hooks.

```rust
use cumulus3d_cli::declare::{Recon, Sinks};

Recon::declare()
    .input("data")
    .preset("aerial-formation")
    .sinks(Sinks::none())
    .on_frame(|f| {
        // f.position, f.camera, f.name, f.path, f.width, f.height,
        // f.rgb: Arc<[u8]> (RGB8, row-major), f.gps: Option<GpsRecord>
        consume(f.position, &f.camera, f.width, f.height, &f.rgb);
    })
    .build()?
    .run()?;
```

Frame decoding is off unless a frame hook is registered (`SessionConfig::decode_frames` when using `Pipeline` directly).
Registration results for the same image follow later as `FrameRegistered` (camera pose `cam_from_world`).

### Library: pipeline + closure hooks

Every time you feed the frame set of one position, the session emits per-stage events (`ZoneArrived`, `ZonePreview`, `ZoneRefined`, …),
and the hooks you registered receive them. `sinks::attach` attaches the default hooks that produce the same files as `cumulus3d stream`
(timeline.txt, run.log, full/, aligned/, snapshots/, manifest.json, final_frame/).

```rust
use cumulus3d_cli::events::{Event, EventKind};
use cumulus3d_cli::pipeline::Pipeline;
use cumulus3d_cli::session::{FrameSet, Input, Session, SessionConfig};
use cumulus3d_cli::sinks::{self, SinkOptions};

let mut cfg = SessionConfig::new("data/images");
cfg.gps = cumulus3d_core::io::read_gps_file("data/gps_ref.txt")?;
cfg.total_positions = Some(80);

let p = Pipeline::new(Session::new(cfg))
    .on_zone_preview(|zone, cloud| println!("preview {}: {} points", zone.zone, cloud.len()))
    .on_zone_refined(|zone, cloud| println!("refined {}: {} points", zone.zone, cloud.len()))
    .on(EventKind::PositionDone, |e: &Event| println!("{}", e.timeline_text().unwrap()));
let mut p = sinks::attach(p, "out".as_ref(), &SinkOptions { echo: false, save_models: false })?;

for pos in 0..80 {
    let frames = ["camF", "camR", "camL"].map(|c| (c, format!("{c}/{c}_{:04}.jpg", pos * 3)));
    p.push(Input::Frames(FrameSet::new(frames)));
}
let (_session, summary) = p.finish(); // waits for remaining refinement and drains the hook queues
```

To run it right away on a synthetic scene: `cargo run --release -p cumulus3d-cli --example hooks -- <output folder>`.

## Third-party notices

Some algorithms, default parameters and data formats follow permissively licensed open-source projects.
Their copyright notices and license terms are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Copyright (c) 2026 ParkSangWoo. Licensed under the Apache License, Version 2.0 (`LICENSE`, `NOTICE`). Third-party notices: `THIRD_PARTY_NOTICES.md`.
