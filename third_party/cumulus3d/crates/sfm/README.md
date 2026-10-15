English | [한국어](https://github.com/AH100-1/cumulus3d/blob/main/crates/sfm/README.ko.md)

# cumulus3d-sfm

SfM crate that builds camera poses and sparse 3D points from features and matches (global SfM, new-image registration, triangulation).

API doc comments are currently in Korean; English translation is planned.

**Pipeline stage**: takes the `FeatureStore` filled by `cumulus3d-matching` (features and two-view geometry) and the `MatchGraph` built from it,
and produces a sparse reconstruction (`Reconstruction`). The first model is built by global SfM (relative poses → rotation averaging → tracks →
positioning → BA → retriangulation); after that, the model grows by registering new images and triangulating each time a position arrives.
The result is passed on to `cumulus3d-align` (GPS alignment) and `cumulus3d-dense` (densification). Bundle adjustment uses `cumulus3d-ba`.

## Main entry points

| Function/type | Role | Input → output |
|---|---|---|
| `global_mapper` | Full global SfM (one model for the largest connected component) | `&FeatureStore, &MatchGraph, &GlobalSfmOptions` → `Result<GlobalMapperOutput>` |
| `GlobalSfmOptions` | Global SfM options. `default()`: 3 BA rounds + retriangulation; `script()`: 0 BA rounds, no retriangulation, track cap 100000 | — |
| `GlobalMapperOutput` | Result `{ reconstruction, view_graph, summary, failure }`. On an intermediate failure, the model so far plus the `failure` reason | — |
| `register_images` | Add missing images → try registering each unposed image once → delete failed images | `&mut Reconstruction, &FeatureStore, &MatchGraph, &RegistrationOptions` → `Result<RegistrationReport>` |
| `registration::register_image` | Register one image (P3P/EPnP LO-RANSAC + pose refinement) | `&mut Reconstruction, &MatchGraph, ImageId, &RegistrationOptions` → `Result<RegisterOutcome>` |
| `RegistrationOptions` / `RegistrationOrder` | Registration options / order (`ImageId` default, `VisibilityScore`) | — |
| `RegistrationReport` | Per-attempt results (`attempts`); `registered()` gives the successfully registered image ids | — |
| `triangulate_points` | Extend existing points → create new points → complete and merge tracks → retriangulate → refine → filter | `&mut Reconstruction, &MatchGraph, &PointTriangulatorOptions` → `Result<TriangulationReport>` |
| `PointTriangulatorOptions` | Triangulation options (`scope`, `refiner`, filter thresholds, etc.) | — |
| `TriangulationScope` | `AllRegistered` (default) or `Images(new images)`: process only new images and changed points | — |
| `PointRefiner` | Point refinement method: `BundleAdjuster` (default) / `PerPoint` (per-point 3-variable LM) / `None` | — |
| `absolute_pose::solve_abs_pose` | Absolute pose estimation from 2D–3D correspondences (P3P + EPnP LO-RANSAC) | `&Camera, &[Vec2], &[Vec3], &AbsolutePoseOptions` → `Option<AbsolutePoseResult>` |
| `positioning::global_positioning` | With rotations fixed, estimate camera centers and point positions | `&mut Reconstruction, &PositionSolverOptions` → `Result<PositioningSummary>` |
| `rotation_averaging::solve_rotation_averaging` | Global rotation estimation (maximum spanning tree → L1 ADMM → IRLS) | `&BTreeSet<ImageId>, &ViewGraph, Option<&BTreeMap<ImageId, Mat3>>, &RotationAveragingOptions` → `Result<(BTreeMap<ImageId, Mat3>, RotationAveragingSummary)>` |

Re-exported at the crate root: `global_mapper`, `GlobalSfmOptions`, `GlobalMapperOutput`, `register_images`, `RegistrationOptions`,
`RegistrationOrder`, `RegistrationReport`, `triangulate_points`, `PointRefiner`, `PointTriangulatorOptions`, `TriangulationScope`.
Everything else is used through its module path (`cumulus3d_sfm::<module>::<item>`).

## Public items

### `global_mapper` — global SfM

| Item | Kind | Role |
|---|---|---|
| `global_mapper` | fn | Run the global mapper (re-exported at root). Color extraction is done by the caller via `Reconstruction::extract_colors` |
| `GlobalSfmOptions` | struct | Stage skip flags, track / rotation averaging / positioning options, filter thresholds, BA and retriangulation settings (re-exported at root). `script()` preset |
| `GlobalMapperSummary` | struct | Per-stage statistics (edge count, track and positioning statistics, filter removal counts, registered image and point counts) |
| `GlobalMapperOutput` | struct | `reconstruction`, `view_graph`, `summary`, `failure: Option<String>` (re-exported at root) |
| `init_reconstruction` | fn | Create an unposed reconstruction from the store's cameras and images (those in the match graph) |
| `build_view_graph` | fn | Build the view graph from per-pair relative poses (parallel per pair, re-estimating relative poses when needed) |
| `rotation_averaging_round` | fn | Apply one round of rotation averaging to the reconstruction (`posed_only`: only posed frames) |
| `filter_angular_error` | fn | Angular error filter. Returns the number of removed observations |
| `filter_angular_error_prior_focal` | fn | Remove observations exceeding the angular error for cameras with a prior focal length |
| `filter_normalized_reproj_error` | fn | Reprojection error filter in normalized coordinates |
| `post_positioning_filters` | fn | Filters and normalization right after positioning |
| `bundle_adjustment_stage` | fn | Iterative BA stage (rotation-fixed BA → full BA → filter) |
| `retriangulation_stage` | fn | Retriangulation + repeated point refinement |

### `registration` — new-image registration

| Item | Kind | Role |
|---|---|---|
| `register_images` | fn | Register multiple images (re-exported at root) |
| `register_image` | fn | Register one image (fixed focal length path) |
| `add_missing_images` | fn | Add images that are in the match graph but not in the model, without poses |
| `collect_2d3d_correspondences` | fn | Collect an image's 2D–3D correspondences `(2D point, pixel, 3D point id, coordinates)` |
| `num_visible_points3d` | fn | Number of 2D points whose correspondence neighbors have a 3D point |
| `visibility_score` | fn | Visibility pyramid score (6 levels) |
| `RegistrationOptions` | struct | Minimum inliers, RANSAC, degenerate camera checks, pose refinement, order, whether to delete failed images (re-exported at root) |
| `RegistrationOrder` | enum | `ImageId` (ascending image id = arrival order) / `VisibilityScore` (re-exported at root) |
| `RegisterOutcome` | enum | `Registered{..}`, `TooFewVisiblePoints`, `TooFewCorrespondences`, `RansacFailed`, `TooFewInliers`, `RefinementFailed`, `AlreadyRegistered`. `is_registered()` |
| `RegistrationReport` | struct | `attempts`, `num_added_images`, `registered()` (re-exported at root) |

### `triangulator` — incremental triangulation

| Item | Kind | Role |
|---|---|---|
| `triangulate_points` | fn | Full triangulation process (re-exported at root) |
| `refine_points` | fn | Point refinement with poses and intrinsics fixed (`ids` = `None` means all points) |
| `TrackTriangulator` | struct | Incremental triangulator (caches registered image poses and rays). Methods `new`, `triangulate_image`, `complete_track(s)`, `merge_track(s)`, `retriangulate`; fields `opts`, `touched`, `report` |
| `TrackTriangulatorOptions` | struct | Creation/extension angle tolerances, merge/completion errors, retriangulation conditions, minimum angle, degenerate camera checks |
| `PointTriangulatorOptions` | struct | `tri`, `clear_points`, final filter thresholds, refinement iterations, `refiner`, `scope` (re-exported at root) |
| `PointRefiner` | enum | `BundleAdjuster` / `PerPoint` / `None` (re-exported at root) |
| `TriangulationScope` | enum | `AllRegistered` / `Images(Vec<ImageId>)` (re-exported at root) |
| `TriangulationReport` | struct | Counts of created, extended, completed, merged, retriangulated, filtered and refined |

### `triangulation` — triangulation primitives

| Item | Kind | Role |
|---|---|---|
| `triangulate_dlt` | fn | Two-view DLT (normalized image plane coordinates) |
| `triangulate_multi_view` | fn | Multi-view triangulation (unit rays in camera coordinates) |
| `triangulate_midpoint` | fn | Two-ray midpoint method (camera-1 coordinates, `None` if behind the camera) |
| `has_positive_depth` | fn | Positive depth check |
| `angular_error` | fn | Angular error between observed ray and point direction (radians) |
| `estimate_triangulation` | fn | RANSAC multi-view triangulation → `(point, inlier mask)` |
| `TriObservation` | struct | RANSAC input observation (`proj`, `center`, `ray`). `new(pose, ray)` |
| `TriangulationRansacParams` | struct | Angular error, minimum angle, RANSAC settings, cap on observation count for exhaustive search |

### `absolute_pose` — absolute pose

| Item | Kind | Role |
|---|---|---|
| `solve_abs_pose` | fn | P3P + EPnP LO-RANSAC (fixed focal length). `None` if inliers < 3 |
| `p3p` | fn | Three-point minimal solver (up to 4 solutions) |
| `epnp` | fn | EPnP (4 or more points) |
| `Corr2D` | struct | Pixel coordinates and back-projected ray of a 2D–3D correspondence |
| `AbsolutePoseOptions` | struct | RANSAC options (max error 12 px, seed `Some(0)` by default) |
| `AbsolutePoseResult` | struct | `world_to_cam`, `inlier_mask`, `num_inliers`, `num_trials` |

### `rotation_averaging` — view graph and rotation averaging

| Item | Kind | Role |
|---|---|---|
| `ViewGraph` | struct | View graph (edges (id1, id2) in ascending order). `new`, `num_valid_edges`, `valid_nodes`, `largest_connected_component`, `invalidate_outside`, `filter_by_relative_rotation` |
| `ViewGraphEdge` | struct | Edge: two image ids, `cam1_to_cam2`, `num_matches`, `valid` |
| `solve_rotation_averaging` | fn | Global rotation (world_to_cam) estimation |
| `mst_initialization` | fn | Maximum spanning tree initialization (root = smallest id) |
| `admm_l1` | fn | Minimize ‖Ax − b‖₁ (ADMM, reusing the factorization) |
| `l1_regression` | fn | General L1 regression (for validation) |
| `RotationAveragingOptions` | struct | MST initialization, L1/IRLS iterations and convergence, IRLS weighting, regularization, ADMM options |
| `AdmmOptions` | struct | ADMM coefficients, tolerance, inner iterations |
| `IrlsWeight` | enum | `GemanMcClure` / `HalfNorm` |
| `RotationAveragingSummary` | struct | L1 and IRLS iteration counts |

### `tracks` — track construction

| Item | Kind | Role |
|---|---|---|
| `establish_tracks` | fn | Build tracks from valid edges and add them to the reconstruction as 3D points |
| `filter_candidate_tracks` | fn | In-image consistency and minimum view checks |
| `select_tracks` | fn | Selection in descending (length, id) order |
| `TrackOptions` | struct | Consistency distance, required tracks per image, minimum view count, maximum track count |
| `TrackSummary` | struct | Counts of candidate, inconsistent, too-few-views, passed and selected tracks |

### `positioning` — global positioning

| Item | Kind | Role |
|---|---|---|
| `global_positioning` | fn | Estimate centers and points from point–camera direction constraints (rotations fixed, Huber loss, built-in LM) |
| `PositionSolverOptions` | struct | Variable selection, loss scale, LM iterations and tolerance, seed, weighting for uncalibrated cameras |
| `PositioningSummary` | struct | Frame, point and observation counts, iteration count, cost, convergence |

### `math` — numerical helpers

| Item | Kind | Role |
|---|---|---|
| `umeyama` | fn | Similarity alignment dst ≈ s·R·src + t (for evaluation and tests) |
| `gaussian` | fn | Standard normal random numbers (for synthetic data and tests) |
| `CsrMatrix` | struct | Compressed sparse row matrix. `new`, `push_row`, `from_dense`, `mul_vec`, `tr_mul_vec`, `weighted_normal` |

### `synthetic` — synthetic scenes (hidden from docs, for tests)

It is `#[doc(hidden)]` but public, so examples and tests can use it. It is not guaranteed as a stable API.

| Item | Kind | Role |
|---|---|---|
| `generate` | fn | Generate a synthetic scene of three drones flying in a straight line (image ids ordered position → camera, starting at 1) |
| `opencv_camera` | fn | OPENCV-model intrinsics of the three cameras |
| `SceneConfig` | struct | Number of positions, spacing, altitude, point count, noise, outlier ratio, seed |
| `Scene` | struct | `store`, ground-truth poses `truth`, `points`, image ids per position, etc. `names_up_to(n)` |

## Example

Builds the first model from the first 5 positions of a synthetic scene, registers the 6th position, then triangulates only the new images
(the same code runs as a doc-test in `lib.rs`).

```rust
use cumulus3d_core::{MatchGraph, MatchGraphOptions};
use cumulus3d_sfm::synthetic::{generate, SceneConfig};
use cumulus3d_sfm::{
    global_mapper, register_images, triangulate_points, GlobalSfmOptions, PointTriangulatorOptions,
    RegistrationOptions, TriangulationScope,
};

let scene = generate(&SceneConfig { num_positions: 6, num_points: 1500, ..Default::default() });
let graph_up_to = |n: usize| {
    let opts = MatchGraphOptions { image_names: scene.names_up_to(n), ..Default::default() };
    MatchGraph::from_store(&scene.store, &opts)
};

// 1) First model via global SfM.
let out = global_mapper(&scene.store, &graph_up_to(5), &GlobalSfmOptions::script())?;
assert!(out.failure.is_none());
let mut rec = out.reconstruction;

// 2) A new position arrives: register → triangulate only the new images.
let graph = graph_up_to(6);
let report = register_images(&mut rec, &scene.store, &graph, &RegistrationOptions::default())?;
let new_images = report.registered();
let opts = PointTriangulatorOptions { scope: TriangulationScope::Images(new_images), ..Default::default() };
let tri = triangulate_points(&mut rec, &graph, &opts)?;
println!("{} registered images, {} new points", rec.registered_image_count(), tri.num_created);
```

With real images, fill the `FeatureStore` using `cumulus3d-features` (extraction) and `cumulus3d-matching` (`match_pairs`),
and build the match graph with `MatchGraph::from_store`.

## Determinism

- All traversals are in ascending id order: rotation averaging root/gauge = smallest active image id, track ids = sorted component order, registration order = ascending image id.
- RANSAC seed defaults to `Some(0)`. Triangulation RANSAC uses combinatorial sampling (no randomness).
- Image triangulation computes per-2D-point plans in parallel and, when applying them sequentially, re-plans only points whose reference state changed → same result as sequential execution.
- Parallel summation in positioning uses fixed chunks, so results are bit-identical regardless of thread count.

## Feature flags and hardware

- No feature flags. CPU only (rayon parallelism); no GPU required.
- Dependencies: `cumulus3d-core`, `cumulus3d-ba`, `cumulus3d-matching`, `nalgebra`, `rayon`, `rand`, `rand_pcg`, `thiserror`.

## Tests

`cargo test --release -p cumulus3d-sfm`: minimal solver accuracy (P3P, EPnP), LO-RANSAC robustness, rotation averaging and positioning accuracy,
track / filter / registration edge cases, and the full global mapper → sequential registration → triangulation process on a three-drone synthetic scene (48 images).
