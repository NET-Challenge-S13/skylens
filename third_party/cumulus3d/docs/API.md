English | [한국어](API.ko.md)

# cumulus3d public API map

An overview of the whole workspace. The complete list of each crate's public items is in that crate's README.

API doc comments are currently in Korean; English translation is planned.

## Crate dependencies

```
                         cumulus3d-core
        ┌──────────┬──────────┼──────────┬──────────┐
        ▼          ▼          ▼          ▼          ▼
    features   matching       ba       align      dense
        │          │ └────┐   │                     │
        │          │      ▼   ▼                     │
        │          │      sfm                       │
        │          ▼                                │
        └──────▶ cuda ◀─────────────────────────────┘
                   │
                   ▼
                  cli  (uses all of core·features·matching·ba·sfm·align·dense·cuda)
```

| Crate | Direct dependencies (within the workspace) |
|---|---|
| `cumulus3d-core` | none |
| `cumulus3d-features` | core |
| `cumulus3d-matching` | core |
| `cumulus3d-ba` | core |
| `cumulus3d-align` | core |
| `cumulus3d-dense` | core |
| `cumulus3d-sfm` | core, ba, matching |
| `cumulus3d-cuda` | core, features, matching, dense |
| `cumulus3d-cli` | all eight above |

Pipeline order: images → **features** (SIFT) → **matching** (descriptor matching, two-view geometry) → **sfm** (global SfM, registration, triangulation; uses **ba** internally)
→ **align** (GPS ENU alignment, re-referencing) → **dense** (dense point cloud). **cuda** is the GPU backend for features, matching, and dense; **cli** assembles all of them into the progressive pipeline.

## Which function to use for what

| To do this | Use this function / type | Crate |
|---|---|---|
| **Progressive reconstruction from an image folder (library)** | `Pipeline::new(Session::new(SessionConfig::new(images)))` → per position `.push(Input::Frames(stream::frame_set(&layout, p)))` → `.finish()` (`Layout::discover(src, stride)`) | cli |
| **Declarative reconstruction: record a plan, check it, run it** | `cumulus3d_cli::declare::Recon::declare().input(src).preset("aerial-formation").zones(12, 2).dense(Dense::profile("fast")).sinks(Sinks::default_files(out)).build()?.run()` | cli |
| Plan as a value / TOML file (`cumulus3d run plan.toml`) | `cumulus3d_cli::declare::Plan` (`to_toml`, `from_toml`, `load`, `from_stream`), `Recon::from_plan` | cli |
| Full run equivalent to the command-line `cumulus3d stream` | `cumulus3d_cli::stream::run_stream(StreamConfig::new(src, out))` | cli |
| State transitions as values, without hooks | `cumulus3d_cli::session::{step, poll, command, finish}` | cli |
| Attach the standard output files (timeline, PLY, snapshots) | `cumulus3d_cli::sinks::attach(pipeline, out, &SinkOptions)` | cli |
| Receive zone preview/refined point clouds | `Pipeline::on_zone_preview` / `on_zone_refined` / `on(EventKind, …)` / `subscribe()` | cli |
| Run per-stage subcommands from code | `cumulus3d_cli::interop::run(InteropCmd::…)` | cli |
| Densify one model (including undistortion) | `cumulus3d_cli::densewrap::dense_model` | cli |
| Read/write model files | `cumulus3d_core::interop::read_model` / `write_model_binary` / `write_model_text` | core |
| Feature and match store | `cumulus3d_core::FeatureStore` (`save` / `load`) | core |
| Build and incrementally update the correspondence graph | `cumulus3d_core::MatchGraph::from_store` / `update_from_store` | core |
| Generic robust estimation | `cumulus3d_core::ransac::lo_ransac` (implement `Estimator`) | core |
| PLY and GPS file I/O | `cumulus3d_core::io::{read_ply, write_ply, read_gps_file}` | core |
| Feature extraction for a batch of images | `cumulus3d_features::FeatureExtractor::extract_inputs` / `extract_files` | features |
| SIFT on a single image | `cumulus3d_features::CpuSift` + `SiftEngine::extract` | features |
| EXIF and initial camera values | `cumulus3d_features::ExifInfo::read`, `init_camera` | features |
| Pair matching and geometric verification over the store | `cumulus3d_matching::match_pairs` (from a file list: `match_pair_list_file`) | matching |
| Two-image geometry (E/F/H) | `cumulus3d_matching::estimate_two_view` | matching |
| Descriptor matching only | `cumulus3d_matching::CpuMatcher` + `MatcherBackend::match_descriptors` | matching |
| Two-view relative pose | `cumulus3d_matching::recover_two_view_pose` / `pose_from_essential` | matching |
| Build the first model (global SfM) | `cumulus3d_sfm::global_mapper` (`GlobalSfmOptions`) | sfm |
| Register new images into an existing model | `cumulus3d_sfm::register_images` (single image: `registration::register_image`) | sfm |
| Triangulate only the new images | `cumulus3d_sfm::triangulate_points` + `TriangulationScope::Images(..)` | sfm |
| Pose estimation from 2D–3D | `cumulus3d_sfm::absolute_pose::solve_abs_pose` | sfm |
| Global rotation averaging | `cumulus3d_sfm::rotation_averaging::solve_rotation_averaging` | sfm |
| Bundle adjustment | `cumulus3d_ba::bundle_adjust` + `BaConfig` | ba |
| Refine a single pose | `cumulus3d_ba::refine_abs_pose` | ba |
| Align a model to GPS ENU | `cumulus3d_align::align_to_gps` / `align_to_gps_file` (Sim3 only: `estimate_gps_alignment`) | align |
| Align two reconstructions (preview ↔ refined) | `cumulus3d_align::align_reconstructions` | align |
| Chained re-referencing of zone coordinate frames | `cumulus3d_align::AnchorChain::push_model` | align |
| Compose a snapshot point cloud | `cumulus3d_align::compose_snapshot` | align |
| GPS ↔ ENU conversion | `cumulus3d_align::EnuFrame` | align |
| Undistortion | `cumulus3d_dense::undistort` / `undistort_from_dir` | dense |
| Build a densification scene | `cumulus3d_dense::DenseScene::from_reconstruction` / `from_workspace_dir` | dense |
| Full densification (depth maps → filtering → fusion) | `cumulus3d_dense::densify` (backend: `cumulus3d_cuda::CudaPatchMatch`) | dense |
| Compute depth maps only and fuse later | `cumulus3d_dense::compute_depth_maps` → `fuse_depth_maps` | dense |
| Point cloud quality statistics / save PLY | `cumulus3d_dense::cloud_stats`, `DenseOutput::write_ply` | dense |
| Check GPU availability | `cumulus3d_cuda::is_available` | cuda |
| GPU densification / matching / SIFT | `CudaPatchMatch::try_default`, `CudaMatcher`, `CudaSift` (pass each in the dense, matching, or features backend slot) | cuda |

The GPU requires a CUDA 12.x driver. `cumulus3d-cuda` loads CUDA dynamically, so it builds on machines without CUDA. There is no CPU backend for densification.

## Per-crate documentation

| Crate | Documentation |
|---|---|
| `cumulus3d-core` | [crates/core/README.md](../crates/core/README.md) |
| `cumulus3d-features` | [crates/features/README.md](../crates/features/README.md) |
| `cumulus3d-matching` | [crates/matching/README.md](../crates/matching/README.md) |
| `cumulus3d-ba` | [crates/ba/README.md](../crates/ba/README.md) |
| `cumulus3d-sfm` | [crates/sfm/README.md](../crates/sfm/README.md) |
| `cumulus3d-align` | [crates/align/README.md](../crates/align/README.md) |
| `cumulus3d-dense` | [crates/dense/README.md](../crates/dense/README.md) |
| `cumulus3d-cuda` | [crates/cuda/README.md](../crates/cuda/README.md) |
| `cumulus3d-cli` | [crates/cli/README.md](../crates/cli/README.md) |
