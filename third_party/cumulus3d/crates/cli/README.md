English | [한국어](https://github.com/AH100-1/cumulus3d/blob/main/crates/cli/README.ko.md)

# cumulus3d-cli

The `cumulus3d` executable, which progressively builds 3D point clouds from drone-formation video (three cameras, GPS), and the library (`cumulus3d_cli`) that holds its assembly.

API doc comments are currently in Korean; English translation is planned.

**Pipeline stage**: the whole assembly. It chains the other crates (feature extraction → matching → SfM → BA → GPS alignment → densification)
inside a single process and handles **progressive streaming**: while video arrives position by position, it emits per-zone previews (no BA) and
refined models (with BA) as events. Data passes between stages as in-memory structures (no intermediate database or model folders).

Build: `cargo build --release -p cumulus3d-cli` → `target/release/cumulus3d`.

## Executable usage

### `cumulus3d stream`

```text
cumulus3d stream --src <input dir> --out <output dir> [--span 12 --overlap 2 --stride 3] [options]
```

Input folder: `images/camF|camR|camL/<cam>_NNNN.jpg`, `gps_ref.txt` (lines: `camF/camF_0000.jpg latitude longitude altitude`).
Only frames `camF_NNNN.jpg` with `NNNN % stride == 0` are used, and **position = index in the list of selected frames**.
The output folder is deleted and recreated.

| Option | Default | Description |
|---|---|---|
| `--span` / `--overlap` / `--stride` | 12 / 2 / 3 | Zone size, overlap (in positions), frame interval |
| `--incremental-triangulation` | off | Triangulate only newly registered images (faster; results may differ) |
| `--depth-cache` | off | Reuse depth maps of overlapping images |
| `--fixed-enu-origin` | off | Fix the origin of every ENU alignment to the first line of the GPS file |
| `--pm-backend` | cuda | PatchMatch backend. Without a device, the densification stage fails |
| `--mvs-profile` | fast | Densification profile (`fast`\|`quality`) |
| `--sift-backend` / `--match-backend` | cpu | SIFT and descriptor-matching backends (`cpu`\|`cuda`, identical results) |
| `--gpu` | off | Set all three backends above to `cuda` |
| `--serialize-dense` | off | Serialize preview and refined densification backend calls one at a time |
| `--threads N` | 0 | Number of rayon threads (0 = number of cores) |
| `--max-positions N` | all | Only the first N positions |
| `--save-models` | off | Save per-zone sparse models (`work/models/`) |
| `--dense-max-image-size` | 960 | Maximum image size for undistortion |
| `--number-views` | 10 | Number of source views for densification (≤ 32) |
| `--max-num-features` / `--sift-max-image-size` | 8192 / 3200 | Maximum SIFT feature count and input size |
| `--seed N` | none | Fix the RANSAC seed for two-view geometry and GPS alignment |
| `--no-dense` | off | Skip densification (for development/testing; no post-processing output) |
| `--quiet` | off | Write progress only to run.log |

Output: `timeline.txt`, `run.log`, `full/{preview,refined}/*.ply` (per-zone dense point clouds), `aligned/`, `snapshots/` (per-event snapshots,
`manifest.json`), `final_frame/` (zone point clouds brought into the last refined frame), `work/models/` (`--save-models`), `DONE`.
The table of outputs per hook is in the `sinks` module documentation.

```bash
cumulus3d stream --src <input-dir> --out <output-dir> --stride 1
cumulus3d stream --src <input-dir> --out <output-dir-quick> --stride 1 --max-positions 15 --dense-max-image-size 320
```

### `cumulus3d run` / `cumulus3d plan` (declarative plan)

```text
cumulus3d run <plan.toml> [--check]        # check the plan (every problem at once), then run
cumulus3d plan --print-default             # default plan (aerial-formation preset) as TOML
cumulus3d plan --check <plan.toml>         # check only
```

A plan file holds the same settings as the `stream` options (`[input]`, `[zones]`, `[align]`, `[features]`, `[matching]`,
`[sparse]`, `[dense]` with `[dense.fusion]`/`[dense.filter]`, `[sinks]`, and top-level `seed`, `threads`, `gps`, `pairing`).
Missing keys take their defaults; unknown keys are errors. Example: `examples/plans/aerial-formation.toml` at the repository root.
With the same settings, `run` produces the same outputs as `stream`.

### Stage subcommands

`feature_extractor`, `matches_importer`, `global_mapper`, `image_registrator`, `point_triangulator`, `bundle_adjuster`,
`model_aligner`, `model_analyzer`, `model_converter`, `image_deleter`, `image_undistorter`, `densify`.
They accept the command names and option strings used by existing scripts (`--database_path`, `--ImageReader.camera_model`, etc.).
State between stages is carried by `--database_path` (a cumulus3d feature-store binary file in `C3DFS` format; files with the old marker from before 0.3.0 are also read) and model folders (`cumulus3d_core::interop` format).
Boolean options accept `1/0/true/false`; GPU-related options (`--FeatureExtraction.use_gpu`, etc.) are accepted but ignored.

```bash
cumulus3d feature_extractor --database_path db.c3dfs --image_path images --ImageReader.single_camera_per_folder 1 --ImageReader.camera_model OPENCV
cumulus3d matches_importer --database_path db.c3dfs --match_list_path pairs.txt --match_type pairs
cumulus3d global_mapper --database_path db.c3dfs --image_path images --output_path sg0   # model goes to sg0/0
cumulus3d model_aligner --input_path in --output_path out --ref_images_path gps.txt --ref_is_gps 1 --alignment_type enu --alignment_max_error 3
cumulus3d model_converter --input_path model --output_path out --output_type BIN|TXT|PLY
cumulus3d image_undistorter --image_path images --input_path in --output_path dense --max_image_size 960
cumulus3d densify -i dense -o dense.ply [--mvs-profile fast|quality --number-views 10 --fusion-mode consistency|traversal|score --stats]
cumulus3d densify -i model --image_path images -o dense.ply      # undistortion done in memory too
cumulus3d densify -i dense -o out/x.ply --fusion-variants variants.txt   # depth maps once, several fusion settings
```

`model_aligner` supports only `--ref_is_gps 1 --alignment_type enu`. The feature store is cumulus3d's own binary format, not SQLite.

## Main entry points

| Function/type | Role | Input → output |
|---|---|---|
| `session::SessionConfig::new` | Default session settings (zone 12 / overlap 2, CPU SIFT and matching, no densification) | image folder → `SessionConfig` |
| `session::Session::new` | Empty progressive-reconstruction state value | `SessionConfig` → `Session` |
| `session::step` | Processes one position: features → pair matching → registration and triangulation → preview when a zone closes, refinement in the background | `(&Session, FrameSet)` → `(Session, Vec<Event>)` |
| `session::poll` | Collects events from finished background refinement jobs (non-blocking) | `&Session` → `(Session, Vec<Event>)` |
| `session::command` | `ResetFrom(position)` rewind / `InvalidateZone(k)` zone rebuild | `(&Session, Command)` → `(Session, Vec<Event>)` |
| `session::finish` | Close remaining zones → wait for background jobs → summary | `&Session` → `(Session, Vec<Event>)` |
| `pipeline::Pipeline::new` | Wraps a reducer to provide hooks, queues, and panic isolation | `R: Reducer` (e.g. `Session`) → `Pipeline<R>` |
| `Pipeline::on` / `on_any` / `on_frame` / `on_zone_preview` / `on_zone_refined` | Register lambda hooks | event kind + closure → `Pipeline` |
| `Pipeline::push` / `finish` | Feed input / finish and get the summary | `Input` → number of emitted events / → `(R, Summary)` |
| `Pipeline::subscribe` | Channel receiving every event | → `Receiver<Event>` |
| `events::Event` | Per-stage results (`ZonePreview`, `ZoneRefined`, `FrameRegistered` …) | — |
| `sinks::attach` | Attaches the file-output hooks that produce the same files as `cumulus3d stream` | `(Pipeline, output folder, &SinkOptions)` → `Pipeline` |
| `declare::Recon::declare` → `.build()` → `.run()` | Declarative builder: record the plan, check it, run it in one call | builder methods → `Recon` → `Summary` |
| `declare::Plan::from_toml` / `to_toml` | Plan ↔ TOML (`cumulus3d run plan.toml`) | `&str` ↔ `Plan` |
| `stream::run_stream` | Full `cumulus3d stream` run (through `declare`) | `StreamConfig` → `Result<(), String>` |
| `stream::Layout::discover` / `stream::frame_set` | Input folder → per-position frame sets | `(folder, stride)` → `Layout`; `(&Layout, position)` → `FrameSet` |
| `densewrap::dense_model` | Densifies one model (exclude images → undistort → densify) | `(&Reconstruction, keep, image folder, &DenseConfig)` → `DenseRun` |
| `interop::run` | Runs a stage subcommand | `InteropCmd` → `Result<(), String>` |

## Public items

### `session` — state value and reducer

| Item | Kind | Role |
|---|---|---|
| `FrameImage` | struct | One image of a frame set (`camera`, `name`) |
| `FrameSet` | struct | Frame set of one position (`images`, `gps`); `new(list of (camera, name))` |
| `SessionConfig` | struct | Session settings (image folder, GPS, zone size and overlap, seed, feature-extraction options, SIFT and matching backends, densification, number of rewind points); `new` |
| `Input` | enum | Reducer input: `Frames`, `Command`, `Poll`, `Log`, `Finish` |
| `EventSink` | type | Immediate-delivery receiver `Arc<dyn Fn(Event) + Send + Sync>` |
| `PreviewZone` | struct | Zone preview result (range, model, point cloud, frame) |
| `RefinedZone` | struct | Zone refined result (range, BA + GPS-aligned model, point cloud, `done`) |
| `Session` | struct | Progressive-reconstruction state value (`Clone` is cheap). Implements `Reducer` |
| `Session::new` / `advance` / `apply` | fn | Create / owned state transition / in-place state transition |
| `Session::set_sink` / `emit_external` | fn | Set the immediate-delivery receiver / emit an external event in session-version order |
| `Session::positions` / `frames` / `position_of` / `model` / `adopted` / `store` / `graph` / `config` | fn | Queries: number of processed positions, frames per position, position of an image, chain model, adopted refined zones, feature store, correspondence graph, settings |
| `Session::arrived_zones` / `previews` / `refined` / `pending_jobs` / `version` / `failed` / `is_finished` | fn | Queries: arrived zones, preview and refined results, remaining background jobs, event version, fatal error, whether finished |
| `step` / `poll` / `command` / `finish` | fn | Functional reducer (see the entry-point table above) |
| `pairs_for` | fn | Matching-pair rule for position p (same camera at intervals 1..5, 8, 16 / different cameras at position offsets 0..4) |
| `zones_upto` | fn | Total positions, zone size, overlap → list of zone ranges |

### `pipeline` — hook pipeline

| Item | Kind | Role |
|---|---|---|
| `Reducer` | trait | Reducer driven by the pipeline (`step`, `poll`, `command`, `finish`) |
| `QueuePolicy` | enum | Per-kind queue policy: `KeepAll` (default), `LatestPerKey`, `LatestOnly` |
| `Summary` | struct | Run summary (event count, count per kind, hook calls and panics, dropped events) |
| `Pipeline` | struct | Reducer + hook executor (asynchronous lanes or synchronous) |
| `Pipeline::new` / `sync` / `policy` | fn | Create / synchronous mode / per-kind queue policy |
| `Pipeline::on` / `on_any` / `on_frame` / `on_zone_preview` / `on_zone_refined` / `on_snapshot` / `on_position_done` / `on_message` | fn | Register hooks |
| `Pipeline::subscribe` | fn | Channel receiving every event (including hook-panic errors) |
| `Pipeline::push` / `poll` / `command` / `flush` | fn | Input / collect background results / command / drain hook queues |
| `Pipeline::reducer` / `reducer_mut` / `finish` | fn | Reducer reference / mutable reference / `(R, Summary)` after finishing |

### `events` — events and commands

| Item | Kind | Role |
|---|---|---|
| `Meta` | struct | Header of every event (session version, time of occurrence) |
| `ZoneRange` | struct | Zone number and position range `[lo, hi)` |
| `Event` | enum | Per-stage results: `Started`, `FrameIngested`, `FeaturesExtracted`, `PairsMatched`, `ModelInitialized`, `FrameRegistered`, `PositionDone`, `ZoneArrived`, `ZonePreview`, `RefineStarted`, `ZoneAdjusted`, `ZoneRefinedPose`, `ZoneRefined`, `BaseAdopted`, `Reanchored`, `Snapshot`, `AllPositionsDone`, `AllRefinedDone`, `Finished`, `Reset`, `ZoneInvalidated`, `Log`, `Warning`, `Error` |
| `Event::kind` / `meta` / `timeline_text` | fn | Kind / header / timeline.txt text |
| `EventKind` | enum | Event kind (for hook registration and filtering) |
| `Command` | enum | `ResetFrom(position)`, `InvalidateZone(zone)` |

### `sinks` — default output hooks

| Item | Kind | Role |
|---|---|---|
| `Hook` | type | Hook `Box<dyn FnMut(&Event) + Send>` |
| `TIMELINE_TABLE` | const | Event kind → timeline text mapping table |
| `Sink` | trait | Hook that receives events and produces output (`kinds`, `handle`) |
| `hooks` / `any_hook` | fn | `Sink` → per-kind hook list / a single hook for `on_any` |
| `RunLog` | struct | Shared run.log handle (`open`, `logger`, `line`, `stamped_at`, `add_time`) |
| `TimelineSink` | struct | Writes `timeline.txt` (`create`, `line`) |
| `RunLogSink` | struct | Writes `run.log` and `DONE` |
| `ZonePlySink` | struct | Writes `full/{preview,refined}/*.ply` |
| `ModelSink` | struct | Writes per-zone sparse models to `work/models/` |
| `SnapshotSink` | struct | `aligned/`, `snapshots/`, `final_frame/`, run.log summary (`finish`) |
| `SinkOptions` | struct | Default hook settings (`echo`, `save_models`) |
| `DefaultSinks` | struct | Bundle of the hooks above (`new`, `with_set`, `run_log`, `into_hook`) |
| `SinkSet` | struct | Which default hooks to enable (`all()`) |
| `default_sinks` | fn | Default hooks as a per-kind list |
| `attach` | fn | Attaches the default hooks to a pipeline as a single `on_any` hook |

### `stream` — `cumulus3d stream` assembly

| Item | Kind | Role |
|---|---|---|
| `CAMS` | const | Camera folders `["camF", "camR", "camL"]` |
| `StreamConfig` | struct | Stream settings (1:1 with the command-line options); `new(src, out)` |
| `Layout` | struct | Position ↔ image name (`discover`, `name`, `position`) |
| `frame_set` | fn | Frame set of position p |
| `pairs_for_position` | fn | Matching pairs of position p (as name pairs) |
| `regions` | fn | Zone list `(k, lo, hi)` |
| `session_config` | fn | `StreamConfig` → `SessionConfig` |
| `Layout::discover_in` / `frame_set_of` | fn | Same as `discover` / `frame_set` for any image folder and camera list |
| `run_stream` | fn | Runs the stream (`Plan::from_stream` → `Recon`) |

### `declare` — declarative builder (plan → build → run)

Full reference: [docs/DECLARATIVE.md](https://github.com/AH100-1/cumulus3d/blob/main/docs/DECLARATIVE.md).

| Item | Kind | Role |
|---|---|---|
| `Recon` | struct | Checked, runnable reconstruction: `declare()`, `from_plan(plan)`, `plan()`, `positions()`, `layout()`, `run() -> Result<Summary, String>` |
| `ReconBuilder` | struct | Records the plan only. `input`, `images`, `cameras`, `preset`, `stride`, `positions`, `pairing`, `zones`, `gps`, `no_gps`, `align`, `fixed_enu_origin`, `features`, `match_backend`, `gpu`, `incremental_triangulation`, `dense`, `no_dense`, `sinks`, `seed`, `threads`; hooks `on`, `on_any`, `on_frame`, `on_zone_preview`, `on_zone_refined`, `on_snapshot`, `on_position_done`, `on_message`, `policy`; `plan()`, `build() -> Result<Recon, PlanError>` |
| `Plan` | struct | Plan value (Clone/PartialEq/Debug, serde): `seed`, `threads`, `gps`, `pairing`, `input`, `zones`, `align`, `features`, `matching`, `sparse`, `dense`, `sinks`; `to_toml`, `from_toml`, `load`, `from_stream` |
| `Source`, `Zones`, `Align`/`AlignFrame`, `Features`, `Matching`, `Sparse`, `Pairing` | struct/enum | Plan sections |
| `Dense`, `Fusion`, `Filter` | struct | Densification plan; `Dense::profile("fast").fusion_min_views(3)…`, `Dense::off()`, `apply`, `config` |
| `Sinks` | struct | Output hooks: `default_files(out)`, `none()`, `timeline`, `run_log`, `zone_ply`, `snapshots`, `models`, `echo`, `clean` |
| `PlanError`, `Problem` | struct | All check problems (`field`, `message`); `has(field)` |
| `DEFAULT_PRESET`, `PRESETS` | const | `"aerial-formation"` |

### `densewrap` — shared densification path

| Item | Kind | Role |
|---|---|---|
| `make_pm_backend` | fn | Selects the PatchMatch backend (`cuda`) |
| `parse_profile` | fn | Parses `--mvs-profile` → `MvsProfile` |
| `make_sift_backend` / `make_match_backend` | fn | Selects the SIFT and descriptor-matching backends (`cpu`\|`cuda`) |
| `DenseConfig` | struct | Densification settings (undistortion, scene and densify options, score fusion, backend, serialization lock, cache); `new` |
| `DenseRun` | struct | Densification result summary (image count, scene, output, stage times) |
| `dense_model` | fn | Densifies a model (exclude images → undistort → scene conversion → densify) |

### `post` — post-processing (alignment, snapshots, re-anchoring)

| Item | Kind | Role |
|---|---|---|
| `PostInput` | struct | Post-processing input based on PLY paths and event strings |
| `ZoneTimes` | struct | Per-zone event times (arrival, preview, refined pose, refined done) |
| `ZoneCloud` | struct | One zone point cloud (file name, point cloud) |
| `PostZones` | struct | In-memory post-processing input |
| `run` | fn | Full post-processing (read PLY → `run_zones`) |
| `run_zones` | fn | Preview alignment → `aligned/` → per-event snapshots and manifest → `final_frame/` |

### `fusion_variants` — batch run of fusion variants

| Item | Kind | Role |
|---|---|---|
| `FusionVariant` | struct | One fusion variant (name, flags, options, score-fusion settings) |
| `read_variants` | fn | Reads a variants file (`name\|fusion flags`) |
| `run_variants` | fn | Fuses each variant from a single set of depth maps and writes PLY files and `fusion_variants.tsv` |

### `interop` — stage subcommands

| Item | Kind | Role |
|---|---|---|
| `parse_flag` | fn | `1/0/true/false` boolean parser |
| `InteropCmd` | enum | The 12 subcommands |
| `FeatureExtractorArgs`, `MatchesImporterArgs`, `GlobalMapperArgs`, `ImageRegistratorArgs`, `PointTriangulatorArgs`, `BundleAdjusterArgs`, `ModelAlignerArgs`, `ModelAnalyzerArgs`, `ModelConverterArgs`, `ImageDeleterArgs`, `ImageUndistorterArgs`, `DensifyArgs` | struct | Per-subcommand arguments |
| `extract_colors` | fn | Extracts 3D point colors from an image folder |
| `run` | fn | Runs a subcommand |

### `util` — logging, timing, and serialization helpers

| Item | Kind | Role |
|---|---|---|
| `Logger` | struct | run.log (+ stdout) logger (`new`, `line`, `stamped`, `stamped_at`, `clock`, `clock_at`) |
| `Timeline` | struct | Timeline file writer (`new`, `ev`, `relative`) |
| `StageTimes` | struct | Accumulated time per stage (`add`, `snapshot`) |
| `timed` | fn | Times one stage and records it in run.log |
| `py_float` | fn | Rounded number as a string with a trailing `.0` |
| `Json` | enum | Small JSON value (`obj`, `f`, `of`, `ints`, `dump`, `py_repr`, `get`) |

## Examples

Feed an image folder position by position for progressive reconstruction, producing the same files as `cumulus3d stream` (requires real images).

```rust,no_run
use cumulus3d_cli::events::{Event, EventKind};
use cumulus3d_cli::pipeline::Pipeline;
use cumulus3d_cli::session::{Input, Session, SessionConfig};
use cumulus3d_cli::sinks::{self, SinkOptions};
use cumulus3d_cli::stream::{frame_set, Layout};
use cumulus3d_core::io::read_gps_file;
use std::path::Path;

fn main() -> Result<(), String> {
    let src = Path::new("data");
    let layout = Layout::discover(src, 3)?;
    let mut cfg = SessionConfig::new(src.join("images"));
    cfg.gps = read_gps_file(src.join("gps_ref.txt")).map_err(|e| e.to_string())?;
    cfg.total_positions = Some(layout.frames.len());

    let pipeline = Pipeline::new(Session::new(cfg)).on(EventKind::ZonePreview, |e: &Event| {
        if let Event::ZonePreview { range, cloud, .. } = e {
            println!("preview zone {}: {} points", range.zone, cloud.len());
        }
    });
    let mut pipeline = sinks::attach(pipeline, Path::new("out"), &SinkOptions::default()).map_err(|e| e.to_string())?;
    for p in 0..layout.frames.len() {
        pipeline.push(Input::Frames(frame_set(&layout, p)));
    }
    let (_session, summary) = pipeline.finish();
    println!("{} events", summary.events);
    Ok(())
}
```

Functional use, passing only state values without hooks:

```rust,no_run
use cumulus3d_cli::session::{finish, step, FrameSet, Session, SessionConfig};

let s0 = Session::new(SessionConfig::new("data/images"));
let frames = FrameSet::new([("camF", "camF/camF_0000.jpg"), ("camR", "camR/camR_0000.jpg")]);
let (s1, events) = step(&s0, frames);
for e in &events {
    if let Some(t) = e.timeline_text() {
        println!("{t}");
    }
}
let (_done, _summary_events) = finish(&s1);
```

Both examples are also included as doc-tests in `src/lib.rs`. For an example that runs end to end on a synthetic scene: `cargo run --release -p cumulus3d-cli --example hooks`.

## Feature flags and hardware

- There are no cargo feature flags. `cumulus3d-cuda` is always linked, but it **loads** the CUDA libraries **dynamically**, so it builds and runs on machines without CUDA.
- Densification (`stream` by default, the `densify` subcommand) requires the GPU PatchMatch backend (CUDA 12.x driver). Without a device,
  `stream` logs an error at the densification stage; use `--no-dense` to run without densification. The `SessionConfig::new` default is no densification.
- `--gpu` or `--sift-backend cuda` / `--match-backend cuda` require a CUDA 12.x driver. Results are identical to the CPU backends.
- Tests: `cargo test --release -p cumulus3d-cli` (stream end to end on a synthetic scene, subcommand chains, equivalence of default hook output).
