English | [한국어](DECLARATIVE.ko.md)

# Declarative functions (`cumulus3d_cli::declare`)

The `declare` module provides a declarative API for composing the reconstruction pipeline.
A `Plan` assembled through the builder is a side-effect-free value: it is validated as a whole in `build()` and executed lazily in `run()`.
No I/O or device initialisation happens during building; execution runs on the event-driven runtime (`session`, `pipeline`, `sinks`).

```
Recon::declare()                    create a builder with an empty plan
  .input(..).stride(..).zones(..)   record values in the Plan only
  .dense(..).sinks(..)              〃
  .on_zone_preview(|..| ..)         hooks are kept apart from the plan
  .build()?                         check the plan → Recon (still nothing runs)
  .run()?                           run: assemble session + pipeline + output hooks → to the end
```

Code: `crates/cli/src/declare.rs` · tests: `crates/cli/tests/declare_synthetic.rs` · plan file example: `examples/plans/aerial-formation.toml`

## 1. Three stages

| Stage | Function | What it does | Side effects |
|---|---|---|---|
| Record | `Recon::declare()` / `Recon::from_plan(plan)` → `ReconBuilder` methods | Write values into the `Plan`, add hooks to the hook list | None |
| Check | `ReconBuilder::build() -> Result<Recon, PlanError>` | Check the whole plan; return every problem at once | Reads the input folder listing and GPS file only |
| Run | `Recon::run() -> Result<Summary, String>` | Assemble the engine from the plan and run the position loop, wait for background refinement, finish | Writes outputs, uses the GPU |

## 2. The plan value `Plan`

A value type with no execution logic: it can be cloned, compared and serialised (TOML), and with a fixed seed the same plan yields the same result.

| Group | Fields | Defaults (`aerial-formation` preset) |
|---|---|---|
| Top level | `seed`, `threads`, `gps`, `pairing` | none, 0 (all cores), `<input>/gps_ref.txt`, `formation` |
| `input` | `images`, `cameras`, `stride`, `positions` | `<input>/images`, `["camF","camR","camL"]`, 3, all |
| `zones` | `span`, `overlap` | 12, 2 |
| `align` | `frame` (`enu`/`none`), `fixed_origin` | `enu`, false |
| `features` | `backend` (`cpu`/`cuda`), `max_num_features`, `max_image_size` | `cpu`, 8192, 3200 |
| `matching` | `backend` | `cpu` |
| `sparse` | `incremental_triangulation` | false |
| `dense` | `enabled`, `backend`, `profile` (`fast`/`quality`), `max_image_size`, `views`, `serialize`, `depth_cache` | true, `cuda`, `fast`, 960, 10, false, false |
| `dense.fusion` | `mode`, `min_views`, `depth_error`, `normal_error`, `reproj_error`, `residual` | profile defaults |
| `dense.filter` | `min_views`, `geom_error`, `min_ncc`, `median` | profile defaults |
| `sinks` | `out`, `clean`, `echo`, `timeline`, `run_log`, `zone_ply`, `snapshots`, `models` | `out`, true, false, true, true, true, true, false |

Plan functions:

| Function | Role |
|---|---|
| `Plan::to_toml(&self) -> String` | Plan → TOML string |
| `Plan::from_toml(&str) -> Result<Plan, String>` | TOML → plan. Missing keys take defaults, unknown keys are errors |
| `Plan::load(path) -> Result<Plan, String>` | Read a TOML file |
| `Plan::from_stream(&StreamConfig) -> Plan` | `cumulus3d stream` options → plan |

## 3. The builder `ReconBuilder` — recording functions

Each method consumes `self`, updates only the plan and returns it, so the pipeline is composed by method chaining.

| Method | Plan field it records |
|---|---|
| `input(src)` | `input.images = src/images`, `gps = src/gps_ref.txt` |
| `images(path)`, `cameras([..])` | `input.images`, `input.cameras` |
| `preset("aerial-formation")` | The three-camera formation defaults (cameras, pairing, stride). Unknown names fail in `build()` |
| `stride(n)`, `positions(n)` | `input.stride`, `input.positions` |
| `pairing(..)` | `pairing` |
| `zones(span, overlap)` | `zones` |
| `gps(path)`, `no_gps()` | `gps` |
| `align(AlignFrame::Enu \| None)`, `fixed_enu_origin(bool)` | `align` |
| `features(..)`, `match_backend(..)`, `gpu()` | `features`, `matching` (`gpu()` sets the feature, matching and densification backends to `cuda`) |
| `incremental_triangulation(bool)` | `sparse` |
| `dense(Dense)`, `no_dense()` | `dense` |
| `sinks(Sinks)` | `sinks` |
| `seed(n)`, `threads(n)` | `seed`, `threads` |
| `plan()` | Read the plan recorded so far |

**Hook functions** — closures are not serialisable, so they are kept in the builder's hook list rather than in the `Plan`:
`on(EventKind, f)`, `on_any(f)`, `on_frame(f)`, `on_zone_preview(f)`, `on_zone_refined(f)`, `on_snapshot(f)`, `on_position_done(f)`, `on_message(f)`, `policy(kind, QueuePolicy)`.

### Densification settings `Dense` (chainable)

Start with `Dense::profile("fast")` or `Dense::off()` and chain:
`backend`, `max_image_size`, `views`, `serialize`, `depth_cache`, `fusion_mode`, `fusion_min_views`, `fusion_depth_error`,
`fusion_normal_error`, `fusion_reproj_error`, `fusion_residual`, `filter_min_views`, `filter_geom_error`, `filter_min_ncc`, `median_filter`.

### Output settings `Sinks` (chainable)

Start with `Sinks::default_files(out)` or `Sinks::none()` and switch `clean`, `echo`, `timeline`, `run_log`, `zone_ply`, `snapshots`, `models` on or off.

## 4. Checking with `build()`

`build()` never modifies the file system; it collects every problem, tagged with its plan field path, into a single `PlanError`.

| Check | What |
|---|---|
| Preset | Known name |
| Input | Image and camera folders exist, camera list non-empty without duplicates, `stride ≥ 1`, at least one position selected |
| Zones | `span ≥ 1`, `overlap < span` |
| Alignment | With ENU alignment, the GPS file exists, parses and has records for the selected images |
| Backends | Names are `cpu`/`cuda`, and a device exists when `cuda` is chosen |
| Densification | When enabled: backend `cuda` plus a device, a valid profile, `views` in 1..=32, sizes and fusion/filter values in range |
| Output | The output folder (or its nearest existing parent) is writable, and a folder that will be cleaned does not contain the input |

## 5. Running with `Recon`

| Function | Role |
|---|---|
| `Recon::declare() -> ReconBuilder` | Start from an empty (default) plan |
| `Recon::from_plan(Plan) -> ReconBuilder` | Start from an existing plan (e.g. TOML) |
| `plan()`, `positions()`, `layout()` | The checked plan, number of selected positions, image layout |
| `run(self) -> Result<Summary, String>` | Run |

What `run()` assembles:
1. `Plan.dense` → densification settings, `Plan.sinks` → default output hooks (cleans the previous output folder when `clean`)
2. The rest of the plan → session settings (GPS, zones, seed, feature/matching backends)
3. Connect output hooks and user hooks to `Pipeline::new(Session::new(..))`
4. `push` positions in order, then `finish()` waits for background refinement → `Summary`

## 6. How to use it

### In code

```rust
use cumulus3d_cli::declare::{Dense, Recon, Sinks};

let summary = Recon::declare()
    .input("data")
    .preset("aerial-formation")
    .zones(12, 2)
    .dense(Dense::profile("fast").fusion_min_views(3).fusion_normal_error(25.0))
    .sinks(Sinks::default_files("out"))
    .seed(7)
    .on_zone_preview(|zone, cloud| println!("preview zone {}: {} points", zone.zone, cloud.len()))
    .build()?        // check
    .run()?;         // run
```

### As a plan file (TOML)

Builder methods and TOML keys correspond one to one (e.g. `.zones(12, 2)` ↔ `[zones] span = 12, overlap = 2`).

```toml
seed = 7
gps = "data/gps_ref.txt"
[input]
images = "data/images"
cameras = ["camF", "camR", "camL"]
stride = 3
[zones]
span = 12
overlap = 2
[dense]
profile = "fast"
[dense.fusion]
min_views = 3
normal_error = 25.0
[sinks]
out = "out"
```

### From the command line

```bash
cumulus3d plan --print-default > plan.toml   # write the default plan
cumulus3d plan --check plan.toml             # check only
cumulus3d run plan.toml                      # check, then run
cumulus3d stream --src data --out out        # the existing command also runs through this layer
```

## 7. Guarantees (covered by tests)

- Until `build()`, no files or folders are created, no hooks run and the input folder is untouched.
- Running the same plan twice, or saving it as TOML and loading it back, produces the same output files (with a fixed seed).
- `cumulus3d run plan.toml` and `cumulus3d stream` produce the same outputs.
- An invalid plan fails before running and reports every problem at once.
