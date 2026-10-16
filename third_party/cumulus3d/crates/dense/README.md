English | [한국어](https://github.com/AH100-1/cumulus3d/blob/main/crates/dense/README.ko.md)

# cumulus3d-dense

Undistortion and multi-view densification: estimates, filters, and fuses depth maps from a registered sparse model and its images to produce a dense point cloud (PLY).

> API doc comments are currently in Korean; English translation is planned.

**Pipeline stage**: the last stage, after SfM (`cumulus3d-sfm`) and GPS alignment (`cumulus3d-align`).
Sparse model → undistortion (PINHOLE) → densification scene → neighbor views and depth ranges → multi-scale PatchMatch depth maps →
filtering and post-processing → fusion → dense point cloud.

PatchMatch execution itself (red–black propagation at one scale, view selection, refinement, readout) is handled by a
backend implementing [`PatchMatchBackend`](https://github.com/AH100-1/cumulus3d/blob/main/crates/dense/src/kernel.rs); this crate handles everything around it (scene, neighbor selection, depth ranges, multi-scale progression, upsampling, detail-restoration decisions,
filter thresholds, post-processing, fusion, statistics).

## Main entry points

| Function/type | Role | Input → output |
|---|---|---|
| `undistort` | Undistort all registered images (image loading is a caller closure) | `&Reconstruction`, `&UndistortOptions`, `&UndistortCache`, `Fn(&Image) -> Option<Arc<ImageBuffer>>` → `Result<UndistortResult>` |
| `undistort_from_dir` | Read from an image folder and undistort | `&Reconstruction`, image folder, options, cache → `Result<UndistortResult>` |
| `write_undistorted_workspace` | Write the undistorted result to a workspace folder (`images/`, `sparse/`, `stereo/`) | `&UndistortResult`, output folder, options → `Result<()>` |
| `DenseScene::from_reconstruction` | Build a scene from an in-memory undistorted model and images | `&Reconstruction`, `&BTreeMap<ImageId, Arc<ImageBuffer>>`, `&SceneOptions` → `Result<DenseScene>` |
| `DenseScene::from_workspace_dir` | Build a scene from an undistorted workspace folder | folder, `&SceneOptions` → `Result<DenseScene>` |
| `DensifyOptions::with_profile` | Default settings for a profile (`Fast`/`Quality`) | `MvsProfile` → `DensifyOptions` |
| `densify` | Full densification (depth maps → filtering → fusion) | `&DenseScene`, `&DensifyOptions`, `&dyn PatchMatchBackend`, `Option<&DepthMapCache>` → `Result<DenseOutput>` |
| `densify::densify_with` | `densify` + optional score fusion | the above + `Option<&ScoreFusionOptions>` → `Result<DenseOutput>` |
| `compute_depth_maps` | Compute depth maps only (no fusion) | `&DenseScene`, `&DensifyOptions`, backend, cache → `Result<DepthMapSet>` |
| `fuse_depth_maps` | Fuse computed depth maps | `&DenseScene`, `&DepthMapSet`, `&DensifyOptions`, thread count → `FusionOutput` |
| `densify::fuse_output` | Re-run only fusion from stored depth maps | `&DenseScene`, `&DepthMapSet`, `&DensifyOptions`, `Option<&ScoreFusionOptions>` → `DenseOutput` |
| `fuse` | Fuse an array of depth maps (low level) | `&DenseScene`, `&[Option<FusionInput>]`, overlap lists, `&FusionParams`, thread count → `FusionOutput` |
| `DenseOutput::write_ply` | Write the point cloud as PLY (x y z nx ny nz red green blue) | path → `Result<()>` |
| `cloud_stats` | Point cloud quality statistics (spacing, GSD, outlier and duplicate rates) | scene, depth maps, point cloud, visibility, sample count, tolerance → `CloudStats` |
| `PatchMatchBackend` | Backend boundary trait (GPU implementation is `cumulus3d-cuda::CudaPatchMatch`) | `&KernelInput` → `Box<dyn PatchMatchSession>` |

## Public items

"Root" marks items re-exported at the crate root as `cumulus3d_dense::name`.

### `undistort` — undistortion

| Item | Kind | Role |
|---|---|---|
| `UndistortOptions` (root) | struct | Blank-pixel ratio, scale range, `max_image_size`, region of interest, number of source images, JPEG quality. `pipeline()`: `max_image_size = 960` |
| `undistorted_camera` (root) | fn | Compute the distortion-free PINHOLE camera (keeps the camera id) |
| `CameraUndistortion` (root) | struct | Undistortion result for one camera (`source`, `pinhole`, resampling map). `new`, `undistort_image` |
| `UndistortCache` (root) | struct | Cache of camera-parameter hash → `CameraUndistortion` (shared across zones). `new`, `get`, `len`, `is_empty` |
| `undistort_reconstruction` (root) | fn | Transform the sparse model: cameras to PINHOLE, 2D observations to new coordinates |
| `UndistortResult` (root) | struct | `reconstruction`, `images` (image id → undistorted image), `failed` |
| `undistort` (root) | fn | Undistort all registered images (image loader closure) |
| `undistort_from_dir` (root) | fn | Read from an image folder and undistort |
| `write_undistorted_workspace` (root) | fn | Write the undistorted result to a dense reconstruction workspace folder |

### `image` — image buffers

| Item | Kind | Role |
|---|---|---|
| `ImageBuffer` (root) | struct | 8-bit 1/3-channel image. `new`, `load`, `save`, `get`, `rgb`, `to_gray`, `resize_area` |
| `GrayImage` (root) | struct | f32 grayscale image. `new`, `at`, `at_clamped`, `bilinear_clamped`, `resize_area`, `resize_cubic`, `rescale` |
| `Integral` | struct | Integral image (f64). `new`, `box_mean`, `resample` (area-average resampling) |
| `bilinear_clamped` | fn | Bilinear interpolation with integer-center convention (edge-clamped) |

### `scene` — densification scene

| Item | Kind | Role |
|---|---|---|
| `SceneOptions` (root) | struct | `max_image_size` (upper bound on the long side, downscale only) |
| `DenseView` (root) | struct | One view (`image_id`, `name`, size, `k`, `r`, `t`, `gray`, `color`). `center`, `to_cam`, `to_world`, `dir_to_world`, `dir_to_cam`, `project_cam`, `ray`, `geometry_hash` |
| `ScenePoint` (root) | struct | Sparse point (`xyz`, indices of observing views) |
| `DenseScene` (root) | struct | `views`, `points`. `from_reconstruction`, `from_workspace_dir` |
| `gray_of` | fn | RGB → 8-bit gray |

### `params` — settings

| Item | Kind | Role |
|---|---|---|
| `DensifyOptions` (root) | struct | Full densification settings. `with_profile`, `schedule`, `fingerprint` (hash for cache keys) |
| `MvsProfile` (root) | enum | `Fast` (default) / `Quality`. `parse`, `name` |
| `LevelSchedule` (root) | struct | Per-scale iteration schedule (photometric iterations, detail-restoration initialization, geometric rounds and iterations) |
| `PmParams` (root) | struct | PatchMatch constants (window, bilateral σ, voting τ0/α/τ1/β/n1/n2, priors, geometric λ/δ, refinement perturbation). `sigma_s`, `window_offsets` |
| `FilterParams` (root) | struct | Depth map filter tolerances (min_ncc, minimum triangulation angle, minimum consistent views, reprojection tolerance, median filter) |
| `FusionMode` (root) | enum | `Traversal` (expansion median) / `Consistency` (consistent average, default). `parse` |
| `FusionResidual` (root) | enum | Handling of leftover pixels: `None` / `Release` / `SecondPass`. `parse`, `name` |
| `ResidualParams` (root) | struct | Second-pass fusion tolerances |
| `FusionParams` (root) | struct | Fusion tolerances (mode, minimum consistent views, inverse-variance weighting, depth/normal/reprojection tolerances, etc.) |
| `NeighborParams` (root) | struct | Number of neighbor views (≤ 32), minimum triangulation angle, directional diversity |

### `kernel` — backend boundary

| Item | Kind | Role |
|---|---|---|
| `PatchMatchBackend` (root) | trait | Backend: `name`, `begin(&KernelInput) -> Box<dyn PatchMatchSession>` |
| `PatchMatchSession` (root) | trait | Backend state for the duration of one call: `run`, `evaluate`, `filter`, `upsample`/`median_filter` (default implementation = host computation) |
| `KernelInput` (root) | struct | Full kernel input (views, number of scales, `PmParams`, `FilterParams`, seed) |
| `KernelView` (root) | struct | One view (random key, per-scale images, source views, relative geometry, depth range) |
| `LevelImage` (root) | struct | Grayscale image and K at one scale |
| `PairGeometry` (root) | struct | Reference → source relative geometry (`r`, `t`, `center`) |
| `ViewState` (root) | struct | Pixel state (depth, normal, cost, coarse-scale hypothesis). `new` |
| `DepthSnapshot` (root) | struct | Depth snapshot read by geometric runs |
| `RunParams` (root) | struct | Parameters for one run (scale, geometric or not, initialization, iteration count, run number, prior usage) |
| `rng_state`, `rng_next` | fn | Per-pixel random number definition identical to the kernel |

### `densify` — progression

| Item | Kind | Role |
|---|---|---|
| `densify` (root) | fn | Full densification: depth maps → filtering → fusion |
| `densify_with` | fn | `densify` + optional score fusion |
| `compute_depth_maps` (root) | fn | Compute final depth maps for all views |
| `fuse_depth_maps` (root) | fn | Fuse depth maps (`opts.fusion.mode`) |
| `fuse_depth_maps_with` | fn | Fuse depth maps + optional score fusion |
| `fuse_output` | fn | Re-run only fusion from stored depth maps and build a `DenseOutput` |
| `pair_geometry` (root) | fn | Reference → source relative geometry |
| `DenseOutput` (root) | struct | `cloud`, `visibility`, `timings`, `cache_hits`, `depth_views`, `depth_maps`, `residual`. `write_ply`, `num_residual`, `write_residual_ply` |
| `DepthMapSet` (root) | struct | Per-view depth maps, source views, baselines, cache hits, number of scales, timings |
| `DepthMapResult` (root) | struct | One depth map (pre-/post-filter depth, normal, cost, whether cached) |
| `DenseTimings` (root) | struct | Per-stage timings. `depth`, `total` |

### `fusion`, `fusion_score` — fusion

| Item | Kind | Role |
|---|---|---|
| `fuse` (root) | fn | Consistency or traversal fusion depending on `FusionParams::mode` |
| `fuse_consistency` | fn | Consistency fusion (average of the reference pixel and consistent points in overlapping views; deterministic) |
| `fuse_traversal` | fn | Neighbor-of-neighbor expansion + per-component median fusion |
| `FusionInput` (root) | struct | Depth, normal, cost, and baseline of one view. `plain` |
| `FusionOutput` (root) | struct | Point cloud, visibility, second-pass fusion flags, timings. `num_residual`, `residual_cloud` |
| `fusion_score::ScoreFusionOptions` | struct | Score fusion options. `from_fusion` |
| `fusion_score::fuse_scored` | fn | Score fusion (consistency contribution − free-space violation penalty) |

### `neighbors` — neighbor views and depth ranges

| Item | Kind | Role |
|---|---|---|
| `PairStats` | struct | Shared point count and triangulation angle per view pair. `new`, `select`, `select_diverse`, `select_all` |
| `depth_ranges` | fn | Per-view depth range (sparse point depth 1%/99% × 0.75/1.25) |
| `depth_range_of` | fn | Range of a single depth list |

### `upsample` — pyramid and upsampling

| Item | Kind | Role |
|---|---|---|
| `num_levels` | fn | Compute the number of scales |
| `build_pyramid` | fn | Grayscale pyramid (0 = coarsest) and per-scale K |
| `joint_bilateral_upsample` | fn | Joint bilateral upsampling |
| `median_plane_filter` | fn | 5×5 median plane filter |
| `downsample_depth` | fn | Downsample a depth map by 2× |

### `postproc` — post-processing

| Item | Kind | Role |
|---|---|---|
| `PostParams` (root) | struct | Speckle removal and hole filling settings |
| `remove_speckles` | fn | Remove small speckles (number of pixels removed) |
| `fill_holes` | fn | Edge-aware hole filling (number of pixels filled) |

### `stats` — quality statistics

| Item | Kind | Role |
|---|---|---|
| `cloud_stats` (root) | fn | Median point spacing, GSD, outlier ratio, duplicate rate, local plane deviation |
| `CloudStats` (root) | struct | The statistics above |

### `cache` — depth map cache

| Item | Kind | Role |
|---|---|---|
| `DepthMapCache` (root) | struct | (image geometry hash, settings hash) → depth map; thread-safe, capacity-limited. `new`, `with_capacity`, `get`, `insert`, `len`, `is_empty` |
| `CachedDepth` (root) | struct | One cached view (pre-/post-filter depth, normal, cost) |

### `math` — numerical tools

| Item | Kind | Role |
|---|---|---|
| `hash_bytes`, `mix64` | fn | Deterministic hash (FNV-1a), splitmix64 mixing |
| `erf` | fn | Error function |
| `emission_norm`, `emission`, `visibility_probability` | fn | Emission density of a cost and visibility probability |
| `triangulation_prior`, `incident_prior`, `resolution_prior` | fn | View selection priors |
| `apply_h` | fn | Transfer a point with a 3×3 homography |
| `quantile_sorted`, `median_in_place` | fn | Quantile, median |
| `plane_transfer` | fn | Depth of a neighbor plane transferred to another ray |

### `synthetic` — synthetic scenes for testing

| Item | Kind | Role |
|---|---|---|
| `make_scene` | fn | Textured floor + box scene, grid of cameras, ground-truth depth and normals, sparse points |
| `SynthConfig` | struct | Image size, focal length, camera grid, boxes, seed, number of sparse points |
| `SynthScene` | struct | `scene`, ground-truth `depth`/`normal`, `config`. `surface_distance` |
| `SynthBox` | struct | Axis-aligned box |
| `texture` | fn | Deterministic texture function |

## Examples

Fuse the ground-truth depth maps of a synthetic scene on the CPU (no backend needed; same as the doc-test in `src/lib.rs`):

```rust
use cumulus3d_dense::neighbors::PairStats;
use cumulus3d_dense::synthetic::{make_scene, SynthConfig};
use cumulus3d_dense::{fuse, FusionInput, FusionParams};

let cfg = SynthConfig { width: 96, height: 72, focal: 75.0, num_points: 500, ..SynthConfig::default() };
let s = make_scene(&cfg);
let stats = PairStats::new(&s.scene);
let overlap: Vec<Vec<usize>> = (0..s.scene.views.len()).map(|v| stats.select(v, 50, 0.0)).collect();
let inputs: Vec<Option<FusionInput>> =
    s.depth.iter().zip(&s.normal).map(|(d, n)| Some(FusionInput::plain(d, n))).collect();
let out = fuse(&s.scene, &inputs, &overlap, &FusionParams::default(), 1);
assert!(out.cloud.len() > 0);
```

Real densification (requires a GPU backend and real data; the doc-test is `no_run`):

```rust,no_run
use cumulus3d_dense::{densify, undistort_from_dir, DenseScene, DensifyOptions, PatchMatchBackend, SceneOptions, UndistortCache, UndistortOptions};

fn run(backend: &dyn PatchMatchBackend) -> cumulus3d_core::Result<()> {
    let rec = cumulus3d_core::interop::read_model("sparse/0")?;
    let und = undistort_from_dir(&rec, "images", &UndistortOptions::pipeline(), &UndistortCache::new())?;
    let scene = DenseScene::from_reconstruction(&und.reconstruction, &und.images, &SceneOptions::default())?;
    let out = densify(&scene, &DensifyOptions::default(), backend, None)?;
    out.write_ply("dense/fused.ply")?;
    Ok(())
}
```

Pass `cumulus3d_cuda::CudaPatchMatch` as `backend` (see the `cumulus3d-cuda` crate).

## Feature flags and hardware

- No feature flags. Pure Rust; builds on any machine.
- **Densification (`densify`, `compute_depth_maps`) requires a `PatchMatchBackend` implementation.** This crate has no CPU backend;
  the only current implementation is `CudaPatchMatch` in `cumulus3d-cuda` (NVIDIA GPU, CUDA 12.x driver).
- What works on the CPU without a backend: undistortion, scene construction, neighbor selection and depth ranges, upsampling and median filtering, post-processing,
  fusion (`fuse`, `fuse_depth_maps`, `fuse_output`), statistics, synthetic scenes. Parallelism via rayon.

## Processing flow

1. **Scene**: undistorted model (pinhole) + images → views (K, pose, gray/color). Pixel coordinates inside the scene are integer = pixel center (principal point −0.5).
2. **Neighbor views**: from sparse point tracks, compute the shared point count and the 75th-percentile triangulation angle for each view pair, and pick up to 10 views with an angle of at least 1°, ordered by shared count.
   Optionally, per-bin attenuation over baseline azimuth spreads the chosen directions evenly (`NeighborParams::diversity_decay` < 1).
3. **Depth range**: the 1%/99% values of observed sparse point depths times 0.75/1.25. Used only for random initialization.
4. **Multi-scale** (downscale factor 0.5, up to 3 levels): the coarsest scale runs a photometric pass from random initialization; higher scales do joint bilateral upsampling →
   detail restoration (pixels where upsampled-hypothesis cost − photometric-pass cost > 0.1 are replaced with the photometric result) → 2 geometric-consistency runs
   (double-buffered: every reference view reads the previous round's depth snapshot, so the result is independent of execution order).
5. **Readout and filtering**: after a 5×5 median plane filter, keep only pixels for which at least 2 source views (or the number of source views, if fewer) satisfy all of: triangulation angle ≥ 3°, incidence cos > 0, emission visibility probability ≥ E (0.9),
   forward–backward reprojection ≤ 1 px.
6. **Post-processing**: remove small, high-cost components among relative-depth-connected regions; optionally fill holes, stopping at intensity edges.
7. **Fusion**: the default is consistency fusion — project the reference pixel's point into overlapping views; if at least 5 views agree in depth (1% relative), reprojection (2 px), and normal (10°),
   output the inverse-variance-weighted average of the reference and the consistent points (`σ_d = d²·σ_px/(f·b)`, `σ_px = 0.25 + cost`).
   Traversal median fusion (`FusionMode::Traversal`) and score fusion (`ScoreFusionOptions`) are also available.
8. **Output**: PLY (x y z nx ny nz red green blue, 27 bytes/point).

## Profiles

| | `Fast` (default) | `Quality` |
|---|---|---|
| Window | radius 5, step 2 (36 samples) | radius 5, step 1 (121 samples) |
| Coarsest scale | photometric 6 + geometric 2×2 | photometric 7 + geometric 2×6 |
| Higher scales | detail restoration 3 (starting from upsampled hypothesis) + geometric 2×2 | detail restoration 6 (random start) + geometric 2×6 |

## Determinism

Per-pixel random numbers come from a counter-based sequence determined by (seed, image name hash, scale, run, iteration, half-step, pixel), so they are independent
of thread scheduling. Geometric runs use a double-buffered snapshot, so they are independent of view processing order. Consistency fusion and score fusion are also deterministic (traversal fusion is not when run in parallel).

## Tests

- `cargo test -p cumulus3d-dense`: numerical reference values, neighbor selection, relative geometry, upsampling and median filtering, fusion of synthetic-scene ground-truth depth maps,
  second-pass fusion, post-processing, undistortion.
- GPU accuracy: `tests/patchmatch_gpu.rs` in `cumulus3d-cuda` (compared against synthetic-scene ground truth).

## References

- M. Bleyer, C. Rhemann, C. Rother. PatchMatch Stereo – Stereo Matching with Slanted Support Windows. BMVC 2011.
- S. Galliani, K. Lasinger, K. Schindler. Massively Parallel Multiview Stereopsis by Surface Normal Diffusion. ICCV 2015.
- J. L. Schönberger, E. Zheng, J.-M. Frahm, M. Pollefeys. Pixelwise View Selection for Unstructured Multi-View Stereo. ECCV 2016.
- Q. Xu, W. Tao. Multi-Scale Geometric Consistency Guided Multi-View Stereo. CVPR 2019.
- J. Kopf, M. F. Cohen, D. Lischinski, M. Uyttendaele. Joint Bilateral Upsampling. SIGGRAPH 2007.

## License

Apache-2.0 (`LICENSE`, `NOTICE`).
