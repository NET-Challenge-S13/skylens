English | [한국어](CHANGELOG.ko.md)

# Changelog

Developed under the name skyrecon up to and including 0.3.0.

This project follows [Semantic Versioning](https://semver.org/). During 0.x, minor versions (0.1 → 0.2) may contain breaking changes.
The date of each version is the actual work/commit date.

## [0.4.0] — planned (in progress)
### Changed
- CI: CUDA kernels compiled with NVRTC in a CUDA container on every push (no GPU needed; `cumulus3d_cuda::check_kernels`,
  example `check_kernels`), required `rustfmt` check (`rustfmt.toml`: width 140), content-policy scan, runners pinned
  (`ubuntu-24.04`, `macos-15`), `actions/checkout@v5`, faster test builds. Release workflow on `v*` tags: all checks, tag/version
  match, `cumulus3d` binaries (Linux x86_64, macOS arm64) on a GitHub Release, semver check and `cargo publish --workspace`
  (only when `CARGO_REGISTRY_TOKEN` is set). Dependabot for Cargo and GitHub Actions.
- License: Apache-2.0 only (was MIT OR Apache-2.0). Copyright holder ParkSangWoo; `LICENSE`, `NOTICE` (root and every crate),
  `authors` in Cargo metadata, and a license header in every source file (`SPDX-License-Identifier: Apache-2.0`; files whose algorithms
  follow other projects point to `THIRD_PARTY_NOTICES.md`). CI checks that every source file carries the header.

### Added
- Live composition layer `compose` (declare the current input state, the retained node reconciles it): `CompositionHost`,
  `ReconNode::declare()…build(&host)`, `compose(FrameState)`, `reconstruct(&host, id, state)`, `close()`. Frame identity
  `FrameKey { source_id, sequence, revision }`; duplicate declarations are no-ops, `Pending`/incomplete sets wait, gaps are reported.
  `ReconcilePolicy::{AppendOnly, ReplayWindow, InvalidateAffectedZones, Strict}` for late arrivals and corrections (built on
  `ResetFrom`/`InvalidateZone`). In-memory payloads (`Rgb8`, `Encoded`) spooled atomically. `ReconNode::spawn` producer queue with
  `Backpressure::{Block, KeepLatest, Unbounded}`, `try_send`, queue statistics. Reference: `docs/COMPOSE.md`.
- Declarative builder layer `declare` in front of the event-driven pipeline. Builder methods only record a plan; nothing runs until
  `.build()` (checks) and `.run()` (position loop, background refinement wait and finish in one call).
  - `Recon::declare() -> ReconBuilder`: `.input`, `.images`, `.cameras`, `.preset("aerial-formation")`, `.stride`, `.positions`,
    `.pairing`, `.zones(span, overlap)`, `.gps`, `.no_gps`, `.align`, `.fixed_enu_origin`, `.features`, `.match_backend`, `.gpu`,
    `.incremental_triangulation`, `.dense(Dense::profile("fast").fusion_min_views(3)…)`, `.no_dense`, `.sinks(Sinks::default_files(out))`,
    `.seed`, `.threads`, hooks `.on(EventKind, closure)`, `.on_any`, `.on_zone_preview`, `.on_zone_refined`, `.on_snapshot`,
    `.on_position_done`, `.on_message`, `.policy` (hooks are kept in the builder, not in the plan).
- Frame hook: `on_frame(|frame: &Arc<Frame>| …)` on `Pipeline` and `ReconBuilder` delivers each decoded input frame (RGB8, size,
  camera, path, GPS record) **before** the position is processed, one camera image at a time. New event `Event::FrameDecoded`,
  `Reducer::prelude` (events dispatched before `step`), `SessionConfig::decode_frames` (off by default; the builder turns it on
  when a frame hook is registered).
  - `ReconBuilder::build() -> Result<Recon, PlanError>`: checks input and camera folders, selected position count, GPS file when ENU
    alignment is requested, backend names and CUDA device availability (including densification), output folder writability, and option
    ranges. All problems are collected and reported at once.
  - `Recon::run() -> Result<Summary, String>`, `Recon::plan()`.
  - Plan value `Plan` (Clone/PartialEq/Debug, serde): input (image folder, camera folders, stride, position count), pairing strategy
    (`formation`), zones, GPS file, alignment (ENU), densification (backend, profile, fusion and filter overrides), output sinks, seed, threads.
    `Plan::to_toml`/`from_toml`/`load`/`from_stream`.
- Commands `cumulus3d run <plan.toml> [--check]`, `cumulus3d plan --print-default`, `cumulus3d plan --check <plan.toml>`.
  Example plan `examples/plans/aerial-formation.toml`.
- `sinks::SinkSet` and `DefaultSinks::with_set` (choose which default hooks to enable), `stream::Layout::discover_in`,
  `Layout::truncate`, `stream::frame_set_of` (any image folder and camera list).
- Dependencies `serde` and `toml` (both MIT OR Apache-2.0).
- Tests `tests/declare_synthetic.rs` (no side effects before run, same plan gives the same result, `run` matches `stream`).
### Changed
- `cumulus3d stream` (`stream::run_stream`) now runs through the declarative layer (`Plan::from_stream` → `Recon`). Outputs are unchanged.
  Plan problems are now reported before anything runs; in particular, densification without a CUDA device fails at the start with a
  clear message instead of at the first zone.

## [0.3.1] — 2026-10-09
### Changed
- Renamed the project to cumulus3d. Crates `cumulus3d-core`/`-features`/`-matching`/`-ba`/`-sfm`/`-align`/`-dense`/`-cuda`/`-cli`,
  executable `cumulus3d`, repository https://github.com/AH100-1/cumulus3d, environment variable prefix `CUMULUS3D_`.
- Changed the feature store file magic to `C3DFS`. Files saved with the old magic are still read as before. Behavior is unchanged.

## [0.3.0] — 2026-10-09
### Added
- Event contract `events`: `Event`/`EventKind`/`Meta` (session version, emission time)/`ZoneRange`/`Command`.
  Per-stage events `Started`, `FrameIngested`, `FeaturesExtracted`, `PairsMatched`, `ModelInitialized`, `FrameRegistered`,
  `PositionDone`, `ZoneArrived`, `ZonePreview`, `RefineStarted`, `ZoneAdjusted`, `ZoneRefinedPose`, `ZoneRefined`,
  `BaseAdopted`, `Reanchored`, `Snapshot`, `AllPositionsDone`, `AllRefinedDone`, `Finished`, `Reset`, `ZoneInvalidated`,
  `Log`, `Warning`, `Error`. Zone events carry the model and point cloud (`Arc`). `Event::timeline_text` gives the timeline.txt message.
- State-value session `session`: `Session` + `step(&Session, FrameSet) -> (Session, Vec<Event>)`, `poll`, `command`, `finish`.
  For each frame set of one position (`FrameSet`) it emits features → matching → registration and triangulation → zone preview/refined (background) as events, and writes no files.
  `SessionConfig` (zone size and overlap, total position count, GPS, densification settings, number of rewind points), `Command::ResetFrom`/`InvalidateZone`.
- Closure-hook pipeline `pipeline::Pipeline`: `.on(EventKind, hook)`, `.on_any`, `.on_zone_preview`/`.on_zone_refined`/
  `.on_snapshot`/`.on_position_done`/`.on_message`, channel receiver `subscribe`. Hooks run on per-kind worker lanes and never block processing
  (with `sync(true)` they run immediately and synchronously), per-kind queue policies (`KeepAll`/`LatestPerKey`/`LatestOnly`), hook panic isolation (`Error` event), and a `finish` summary.
- Default output hooks `sinks`: `TimelineSink` (timeline.txt), `RunLogSink` (run.log, DONE), `ZonePlySink` (full/*.ply), `ModelSink` (work/models),
  `SnapshotSink` (preview alignment, masking, per-event snapshots, manifest.json, re-anchored final_frame). Attach them all at once with `sinks::attach(pipeline, out, opts)`;
  a per-kind list `default_sinks` is also available. Output messages, file names, and the manifest format are the same as `skyrecon stream` in 0.2.
- In-memory input for post-processing `post::run_zones` (`PostZones`): takes zone point clouds and models collected from events directly. The existing `post::run` wraps it.
- Example `examples/hooks.rs` (closure hooks + default output hooks on a synthetic scene), test `tests/sinks_equivalence.rs`
  (compares the timeline, run.log, file list, manifest, and point counts of `run_stream` output against the output of feeding the same event sequence to the default hooks).
### Changed
- `util::Logger::clock_at`/`stamped_at`: prefix an `[HH:MM:SS]` header using a given time (for recording event emission times).

## [0.2.0] — in progress (2026-10-09)
### Added
- Densification fusion runner `skyrecon densify --fusion-variants <file>`: builds depth maps only once and produces several results by varying only the fusion settings.
- Fusion option `--fusion-residual none|release|second-pass` (second-pass fusion of remaining pixels) with `--residual-*`, and `--residual-out` to write only the second-pass points.
- Score-based fusion `--fusion-mode score` (`--score-tau`, `--score-sigma-e`, `--score-sigma-theta`, `--score-lambda`).
- Densification tuning records `docs/tuning/`: density and depth-relief sweep (2026-10-09), residual pixel fusion comparison (2026-10-09).
### Planned changes
- Update densification defaults (minimum consistent views for fusion, normal tolerance angle, depth map filter, median filter, hole filling) based on the sweep results.

## [0.1.0] — 2026-10-08
### Added
- Single-process progressive reconstruction pipeline `skyrecon stream`: per-position arrival → feature extraction → pair matching → global SfM → image registration and triangulation →
  per-zone preview (no BA) and refined (BA) → GPS (ENU) alignment → densification → per-step snapshots and re-anchoring.
- Crates: core, features (SIFT), matching (two-view geometry), ba (bundle adjustment), sfm (global and incremental), align (GPS, Sim3), dense (undistortion, GPU multi-view stereo),
  cuda (GPU backend), cli.
- Model file format (cameras/images/points3D) read/write and per-stage compatible subcommands (`interop`).
### Measurements (Tesla V100, 3 drones × 80 positions = 240 images)
- Full streaming: last refined result at 648 s, densification of 240 images in 61 s, 13.47 million points.
