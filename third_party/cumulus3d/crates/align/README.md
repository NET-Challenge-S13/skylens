English | [한국어](https://github.com/AH100-1/cumulus3d/blob/main/crates/align/README.ko.md)

# cumulus3d-align

Coordinate alignment and point cloud post-processing: GPS (WGS84) ↔ ENU conversion, robust Sim3 alignment of a reconstruction to GPS ENU,
alignment of two reconstructions via shared 3D points, zone coordinate-frame chaining (re-anchoring), and point cloud masking, decimation, and snapshot composition.
Every Sim3 follows the core convention `new_from_old` (`X' = sRX + t`).

API doc comments are currently in Korean; English translation is planned.

**Pipeline stage**: after SfM. Aligns each zone's refined model to the GPS-referenced ENU frame (the refined map) (`align_to_gps`),
moves the preview into the refined model's frame (`align_reconstructions`), chains the frames of successive zones (`AnchorChain`),
and builds the step-by-step snapshot point clouds (`compose_snapshot`).

## Main entry points

| Function/type | Role | Input → output |
|---|---|---|
| `align_to_gps` | Fits the projection centers of registered images to GPS ENU with a robust Sim3 and applies it to the model (model unchanged on failure) | `&mut Reconstruction`, `&[GpsRecord]`, `&ModelAlignerOptions` → `Result<GpsAlignment>` |
| `align_to_gps_file` | Same as above, reading the GPS list from a file | `&mut Reconstruction`, path, `&ModelAlignerOptions` → `Result<GpsAlignment>` |
| `estimate_gps_alignment` | Estimates the alignment Sim3 only (model unchanged) | `&Reconstruction`, `&[GpsRecord]`, `&ModelAlignerOptions` → `Result<GpsAlignment>` |
| `align_reconstructions` | Estimates the Sim3 between two reconstructions from shared 3D points + robust Umeyama | `&Reconstruction`×2, `&SharedPointOptions`, `&RobustUmeyamaOptions` → `Result<(RobustUmeyamaResult, correspondence pairs)>` |
| `AnchorChain` | Per-zone chain of "latest frame ← zone frame" transforms | `push_model(prev, new, ..)` → `zone_to_latest(k)` |
| `compose_snapshot` | Combines refined zones + (non-overlapping) preview zones, then decimates to 1/N | `&[&PointCloud]`, `&[&PointCloud]`, `&SnapshotOptions` → `PointCloud` |
| `EnuFrame` | ENU frame with a fixed origin (LLA/ECEF/ENU conversions) | `new(lat, lon, alt)` → `lla_to_enu`, `enu_to_lla`, … |
| `umeyama` | Closed-form Sim3 (or rigid) solution from point correspondences | `&[Vec3]`, `&[Vec3]`, `bool` → `Option<Sim3>` |
| `robust_umeyama` | Median-based iteratively refitted Umeyama | `&[Vec3]`, `&[Vec3]`, `&RobustUmeyamaOptions` → `Option<RobustUmeyamaResult>` |
| `estimate_sim3_ransac` | Robust Sim3 via LO-RANSAC | `&[Vec3]`, `&[Vec3]`, `&RansacParams`, `RankCheck` → `RansacReport<Sim3>` |
| `transform_cloud` | Applies a Sim3 to a point cloud (points and normals) | `&mut PointCloud`, `&Sim3` → `()` |
| `KdTree` | Static 3D KD-tree (nearest-neighbor and radius queries, including parallel variants) | point list → query results |

## Public items

Items marked ✓ in the "Root" column are also re-exported at the crate root (`cumulus3d_align::`).

### `cloud` — point cloud post-processing (core `PointCloud`)

| Item | Kind | Root | Role |
|---|---|---|---|
| `transform_cloud` | fn | ✓ | Applies sRx+t to points and Rn to normals (computed in f64) |
| `select` | fn | | New point cloud keeping only points whose mask is true |
| `remove_near` | fn | ✓ | Removes points within distance ≤ r of a reference KD-tree |
| `decimate` | fn | ✓ | Takes every stride-th point (indices 0, N, 2N, …) |
| `merge_clouds` | fn | ✓ | Concatenates clouds (missing normals filled with 0, missing colors with white) |
| `SnapshotOptions` | struct | ✓ | `mask_radius` (default 1.5 m), `stride` (default 6) |
| `compose_snapshot` | fn | ✓ | All refined points + preview points not overlapping the refined ones, then 1/N decimation (inputs share one frame) |

### `error`

| Item | Kind | Root | Role |
|---|---|---|---|
| `AlignError` | enum | ✓ | `Core`, `BadMaxError`, `InvalidArgument`, `TooFewReferences`, `TooFewCommonImages`, `TooFewCorrespondences`, `RansacFailed`, `Degenerate` |
| `Result<T>` | type | ✓ | `std::result::Result<T, AlignError>` |

### `geodesy` — WGS84 LLA ↔ ECEF ↔ ENU

| Item | Kind | Root | Role |
|---|---|---|---|
| `WGS84_A`, `WGS84_F`, `WGS84_B`, `WGS84_E2` | const | ✓ | Semi-major axis, flattening, semi-minor axis, eccentricity² |
| `lla_to_ecef` | fn | ✓ | (latitude°, longitude°, ellipsoidal height m) → ECEF |
| `ecef_to_lla` | fn | ✓ | ECEF → (latitude°, longitude°, ellipsoidal height m) (iterative; closed form at the poles) |
| `EnuFrame` | struct | ✓ | ENU frame with a fixed origin. Fields `origin_lla`, `origin_ecef`, `rotation` |
| `EnuFrame::new` | fn | | Creates from an origin LLA |
| `EnuFrame::ecef_to_enu` / `enu_to_ecef` | fn | | ECEF ↔ ENU |
| `EnuFrame::lla_to_enu` / `enu_to_lla` | fn | | LLA ↔ ENU |

### `kdtree`

| Item | Kind | Root | Role |
|---|---|---|---|
| `KdTree` | struct | ✓ | Static KD-tree (keeps original indices) |
| `KdTree::new` / `from_f32` | fn | | Builds from f64 / f32 (PLY) points |
| `KdTree::len` / `is_empty` | fn | | Point count / whether empty |
| `KdTree::nearest` | fn | | Nearest neighbor (original index, distance²) |
| `KdTree::any_within` / `within_radius` | fn | | Whether any point lies within radius r / all such points |
| `KdTree::nearest_many` / `any_within_many_f32` | fn | | Parallel query variants |

### `model_aligner` — GPS ENU alignment

| Item | Kind | Root | Role |
|---|---|---|---|
| `EnuOrigin` | enum | ✓ | `FirstRecord` (default: first line of the GPS list), `Explicit { lat, lon, alt }` (fixed per-session origin) |
| `ModelAlignerOptions` | struct | ✓ | `max_error` (3 m), `min_common_images` (3), `origin`, `ransac`, `rank_check` (`Uncentered`) |
| `GpsAlignment` | struct | ✓ | `sim3` (model_to_enu), `enu`, `common`, `inlier_mask`, `num_inliers`, `num_trials`, `errors`, `mean_error`, `median_error` |
| `gps_to_enu` | fn | ✓ | GPS list → (ENU frame, list of (name, ENU)) |
| `estimate_gps_alignment` | fn | ✓ | Estimates the alignment Sim3 (model unchanged). Fails on: `max_error ≤ 0`, fewer than 3 references, too few common images, RANSAC failure |
| `align_to_gps` | fn | ✓ | Estimates, then on success calls `rec.transform(sim3)` |
| `align_to_gps_file` | fn | ✓ | Variant taking a GPS file path |

### `reanchor`

| Item | Kind | Root | Role |
|---|---|---|---|
| `AnchorChain` | struct | ✓ | Set of chained Sim3 transforms between zone frames |
| `AnchorChain::new` / `len` / `is_empty` | fn | | Create / number of zones / whether empty |
| `AnchorChain::push` | fn | | Adds a new zone with `Option<new_from_prev>` (if Some, composed into every earlier zone) |
| `AnchorChain::push_model` | fn | | Computes the linking Sim3 from points shared by the previous and new refined models and adds it (the zone is added even on failure; the error is returned) |
| `AnchorChain::zone_to_latest` / `is_linked` | fn | | Zone → latest-frame transform / whether linked to the previous zone |

### `shared` — shared 3D points of two reconstructions

| Item | Kind | Root | Role |
|---|---|---|---|
| `NameFilter` | type | | Image name filter `Arc<dyn Fn(&str) -> bool + Send + Sync>` |
| `SharedPointOptions` | struct | ✓ | `position_range` (position index range), `name_filter` |
| `position_index_from_name` | fn | ✓ | Last run of digits in the file stem (`camF/camF_0012.jpg` → 12) |
| `shared_point_correspondences` | fn | ✓ | (A point id, B point id) pairs from images with the same name and the same 2D index (sorted, deduplicated) |
| `align_reconstructions` | fn | ✓ | Shared points + robust Umeyama → `src_to_dst` |

### `umeyama` — Sim3 estimation

| Item | Kind | Root | Role |
|---|---|---|---|
| `RankCheck` | enum | ✓ | Degeneracy check: `Uncentered` (default), `Centered` (rejects only collinear configurations), `None` |
| `umeyama` | fn | ✓ | Closed-form Umeyama solution (src_to_dst; None for fewer than 3 points or degenerate input) |
| `Sim3Estimator` | struct | ✓ | Implements core `ransac::Estimator` (minimum 3 pairs, residual ‖y − Tx‖²) |
| `estimate_sim3_ransac` | fn | ✓ | Robust Sim3 via LO-RANSAC (`max_error` = distance threshold) |
| `RobustUmeyamaOptions` | struct | ✓ | `iterations` (5), `factor` (3), `min_threshold` (0.3), `estimate_scale` (true) |
| `RobustUmeyamaResult` | struct | ✓ | `sim3`, `median_residual`, `num_inliers`, `inlier_mask`, `residuals` |
| `robust_umeyama` | fn | ✓ | Iterative refitting: threshold = max(factor · median residual, min_threshold) |

## Examples

Same code as the crate-level doc-test (synthetic data).

```rust
use cumulus3d_align::{compose_snapshot, umeyama, EnuFrame, SnapshotOptions};
use cumulus3d_core::io::PointCloud;
use cumulus3d_core::{Quat, Sim3, Vec3};

// GPS (WGS84) ↔ ENU: east/north/up coordinates in meters from the origin.
let enu = EnuFrame::new(37.5, 127.0, 10.0);
let p = enu.lla_to_enu(37.5001, 127.0001, 10.0);
let (lat, lon, _alt) = enu.enu_to_lla(&p);
assert!((lat - 37.5001).abs() < 1e-9 && (lon - 127.0001).abs() < 1e-9);

// Estimate the Sim3 (src → dst) between two sets of corresponding points.
let truth = Sim3::new(2.0, Quat::from_axis_angle(&Vec3::z(), 0.3), Vec3::new(1.0, 2.0, 3.0));
let src: Vec<Vec3> = (0..10).map(|i| Vec3::new(i as f64, (i * i) as f64 * 0.1, (i as f64).sin())).collect();
let dst: Vec<Vec3> = src.iter().map(|x| truth.transform_point(x)).collect();
let est = umeyama(&src, &dst, true).unwrap();
assert!((est.scale - 2.0).abs() < 1e-9);

// Snapshot: drop preview points within 1.5 m of refined points, merge, then take 1/stride.
let fine = PointCloud { positions: vec![[0.0, 0.0, 0.0]], ..Default::default() };
let coarse = PointCloud { positions: vec![[0.5, 0.0, 0.0], [10.0, 0.0, 0.0]], ..Default::default() };
let snap = compose_snapshot(&[&fine], &[&coarse], &SnapshotOptions { mask_radius: 1.5, stride: 1 });
assert_eq!(snap.len(), 2);
```

Aligning a reconstruction to the GPS-referenced ENU frame:

```rust,no_run
use cumulus3d_align::{align_to_gps_file, ModelAlignerOptions};
let mut rec = cumulus3d_core::interop::read_model("model/0").unwrap();
let a = align_to_gps_file(&mut rec, "gps_ref.txt", &ModelAlignerOptions::default()).unwrap();
println!("inliers {}/{}, median error {:.2} m", a.num_inliers, a.common.len(), a.median_error);
```

## Feature flags and hardware

- No feature flags. CPU only (point cloud processing and KD-tree queries run in parallel with rayon).
- The GPS file format (lines of `name latitude longitude altitude`) is read by `cumulus3d_core::io::read_gps_file`.
