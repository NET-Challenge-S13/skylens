English | [한국어](https://github.com/AH100-1/cumulus3d/blob/main/crates/matching/README.ko.md)

# cumulus3d-matching

SIFT descriptor matching, two-view geometry (E/F/H) estimation and verification, and relative pose decomposition.

API doc comments are currently in Korean; English translation is planned.

**Pipeline stage**: the step after feature extraction (`cumulus3d-features`). For each image pair in a pair list it matches
descriptors, verifies the geometry with E/F/H LO-RANSAC, and writes the raw matches and the `TwoViewGeometry` to the `FeatureStore`.
SfM (`cumulus3d-sfm`) consumes these results and the decomposition functions of the `pose` module. The matching kernel sits behind the
`MatcherBackend` trait, so it can be swapped for a GPU backend (`CudaMatcher` in `cumulus3d-cuda`).

## Main entry points

| Function/type | Role | Input → output |
|---|---|---|
| `match_pairs` | Match and verify a pair list and write it to the store (already processed pairs are skipped) | `&FeatureStore`, `&[(ImageId, ImageId)]`, `&PairMatchingOptions`, `&dyn MatcherBackend` → `Result<MatchingStats>` |
| `match_pair_list_file` | Read a pair list file and run `match_pairs` | `&FeatureStore`, path, options, backend → `Result<(PairList, MatchingStats)>` |
| `estimate_two_view` | Estimate the geometry of two images (E/F/H LO-RANSAC + configuration decision) | camera·keypoints ×2, `&[FeatureMatch]`, `&TwoViewOptions` → `TwoViewGeometry` |
| `verify_pair` | Verify geometry from two images in the store and their matches (applies the minimum-inlier rule) | `&FeatureStore`, id1, id2, matches, `&TwoViewOptions` → `TwoViewGeometry` |
| `CpuMatcher` + `MatcherBackend::match_descriptors` | Brute-force descriptor matching (ratio, distance, cross check) | `&Descriptors` ×2, `&DescriptorMatchOptions`, max count → `Vec<FeatureMatch>` |
| `PairMatchingOptions` / `TwoViewOptions` / `DescriptorMatchOptions` | Matching, geometry and descriptor options | Start from `Default` and modify fields |
| `read_pair_list` / `parse_pair_list` | Parse a pair list file/text | path·text, name → id lookup function → `PairList` |
| `recover_two_view_pose` | Fill in relative pose and triangulation angle from two-view geometry | camera·keypoints, `&mut TwoViewGeometry` → `bool` |
| `pose_from_essential` | E → relative pose + 3D points (cheirality) | `&Mat3`, rays ×2 → `Option<(Rigid3, Vec<Vec3>)>` |
| `essential_five_point` / `fundamental_seven_point` / `homography_dlt` | Minimal solvers | correspondences → matrix candidates |

## Public items

"root" marks items re-exported at the crate root as `cumulus3d_matching::name`.

### `pipeline`

| Item | Kind | Role |
|---|---|---|
| `match_pairs` (root) | fn | Pair matching → verification → write. Parallel per block and per pair, written in input order. If only matches exist, verifies only; if only geometry exists, re-matches |
| `verify_pair` (root) | fn | Verify geometry of one pair (reads camera and keypoints from the store) |
| `match_pair_list_file` (root) | fn | Read a pair list file + `match_pairs` |
| `PairMatchingOptions` (root) | struct | `sift`, `geometry`, `max_num_matches` (32768; the effective value is capped by the store's max keypoint count), `block_size` (1225), `skip_geometric_verification` |
| `MatchingStats` (root) | struct | Counts of skipped, matched, verify-only, and valid-geometry pairs |

### `pairs`

| Item | Kind | Role |
|---|---|---|
| `PairList` (root) | struct | `pairs` (file order, unordered duplicates and self pairs removed), `missing_names` |
| `parse_pair_list` (root) | fn | Text parsing: trims whitespace, ignores empty lines and `#`, separator is a single space (not a tab) |
| `read_pair_list` (root) | fn | Read file + parse |

### `descriptor`

| Item | Kind | Role |
|---|---|---|
| `MatcherBackend` (root) | trait | Matching kernel: `top2` (required, row/column top-2), `match_descriptors` (default implementation) |
| `CpuMatcher` (root) | struct | CPU backend: u8×u8→u32 integer dot product, rayon-parallel row blocks + column tiles (`row_block`, `col_tile`) |
| `DescriptorMatchOptions` (root) | struct | `max_ratio` (0.8), `max_distance` (0.7 rad), `cross_check` (true), `rule` |
| `AcceptRule` (root) | enum | Boundary handling: `Gpu` (default, strict <), `CpuBruteForce` (≤) |
| `Top2` (root) | struct | Per-row (column) maximum, index and second-best value. `push`, `merge` |
| `DOT_NORM` | const | Squared norm of a normalized descriptor (512²) |
| `dot_to_angle` | fn | Dot product → angle (f32 `acos`) |
| `apply_tests` | fn | Apply ratio, distance and cross checks to top-2 results |
| `top2_naive` | fn | Sequential reference implementation (for tests, bit-identical to `CpuMatcher`) |

### `two_view`

| Item | Kind | Role |
|---|---|---|
| `estimate_two_view` (root) | fn | Two-view geometry estimation. Calibrated path (E/F/H) when both cameras have prior focal lengths, otherwise uncalibrated (F/H). Supports multiple models, forced H, and stationary-match filtering |
| `finalize_geometry` (root) | fn | Default (UNDEFINED) if inliers are below the minimum count |
| `TwoViewOptions` (root) | struct | Minimum inliers (15), ratio thresholds, watermark, multiple models, relative pose computation, RANSAC parameters, etc. |
| `decide_calibrated` (root) | fn | Configuration decision for the calibrated path (E/F/H results → `Decision`) |
| `decide_uncalibrated` (root) | fn | Configuration decision for the uncalibrated path (F/H) |
| `Decision` (root) | struct | Decided configuration + mask to use |
| `MaskChoice` (root) | enum | `E`, `F`, `H` inlier mask |
| `ModelOutcome` (root) | struct | Success and inlier count of one RANSAC run |
| `is_watermark` (root) | fn | Detect matches fixed at the image border (watermark) |
| `derive_seed` | fn | Derive a RANSAC seed per model kind (for reproducibility) |

### `pose`

| Item | Kind | Role |
|---|---|---|
| `decompose_essential` (root) | fn | E → 4 (R, t) candidates |
| `pose_from_essential` (root) | fn | E → relative pose + 3D points in camera-1 coordinates |
| `decompose_homography` (root) | fn | H → (R, t, n) candidates |
| `pose_from_homography` (root) | fn | H → relative pose, plane normal, 3D points |
| `triangulate_midpoint` (root) | fn | Midpoint triangulation (None if depth ≤ ε) |
| `recover_two_view_pose` (root) | fn | Geometry → finalize `cam1_to_cam2`, `tri_angle`, planar/rotation configuration |
| `refit_and_estimate_relative_pose` (root) | fn | Preparation for global SfM: if no matrix exists, refit from inliers, then decompose and normalize t |
| `median_triangulation_angle` | fn | Median triangulation angle |
| `inlier_rays` | fn | Unit rays of inlier matches |

### `essential`

| Item | Kind | Role |
|---|---|---|
| `essential_five_point` (root) | fn | Five-point (N ≥ 5) E solutions (up to 10, norm 1) |
| `essential_eight_point` (root) | fn | Eight-or-more-point E (unit rays, rank 2 enforced) |
| `epipolar_row` | fn | One row of the epipolar constraint |

### `estimators` — implementations of `cumulus3d_core::ransac::Estimator`

| Item | Kind | Role |
|---|---|---|
| `fundamental_seven_point` (root) | fn | Seven-point F (up to 3) |
| `fundamental_eight_point` (root) | fn | Normalized eight-point F (rank 2) |
| `homography_dlt` (root) | fn | DLT H (4-point LU / N-point SVD, optional Hartley normalization) |
| `sampson_error_sq` (root) | fn | Squared Sampson error (zero denominator → ∞) |
| `homography_transfer_error_sq` (root) | fn | Squared one-way H transfer error |
| `EssentialFivePointEstimator` (root) | struct | Five-point E estimator (unit-ray input) |
| `Fundamental7PtEstimator` (root) | struct | Seven-point F estimator |
| `FundamentalEightPointEstimator` (root) | struct | Normalized eight-point F local estimator |
| `HomographyEstimator` (root) | struct | DLT H estimator (`normalize` option) |
| `TranslationEstimator` (root) | struct | 2D translation estimator (for watermark detection) |

### `linalg`

| Item | Kind | Role |
|---|---|---|
| `null_space_9` | fn | Null-space basis of an N×9 constraint matrix |
| `singular_values_9` | fn | Singular values and smallest right singular vector of a 9-column matrix |
| `mat3_from_row_major` | fn | Row-major 9-vector → 3×3 |
| `svd3` | fn | Re-export of `cumulus3d_core::linalg::svd3` |
| `null_vector3` | fn | Approximate null vector of a 3×3 |
| `hartley_normalize` | fn | Hartley normalization (points, transform T) |
| `solve8` | fn | 8×8 LU with partial pivoting |

### `poly`

| Item | Kind | Role |
|---|---|---|
| `roots_companion` | fn | Complex polynomial roots via companion-matrix eigenvalues |
| `solve_cubic_monic` | fn | Real roots of a monic cubic (analytic) |
| `poly_mul` / `poly_add` / `poly_eval` | fn | Polynomial product / sum·difference / evaluation |

## Example

Runs descriptor matching and two-view geometry on synthetic data (same code as the doc-test in the crate docs).

```rust
use cumulus3d_core::{Camera, CameraModelKind, Descriptors, FeatureMatch, Keypoint, TwoViewGeometryConfig};
use cumulus3d_matching::{estimate_two_view, CpuMatcher, DescriptorMatchOptions, MatcherBackend, TwoViewOptions};

// 1) Descriptor matching: descriptor i is an orthogonal vector whose only nonzero components 4i..4i+4 are 255.
let make = |order: &[usize]| {
    let mut d = Descriptors::new();
    for &i in order {
        let mut row = [0u8; 128];
        row[4 * i..4 * i + 4].fill(255);
        d.push(&row);
    }
    d
};
let d1 = make(&(0..32).collect::<Vec<_>>());
let d2 = make(&(0..32).rev().collect::<Vec<_>>());
let m = CpuMatcher::default().match_descriptors(&d1, &d2, &DescriptorMatchOptions::default(), 1000);
assert_eq!(m.len(), 32);
assert!(m.iter().all(|x| x.idx2 == 31 - x.idx1));

// 2) Two-view geometry: two images from the same pinhole camera translated by 1 along the x axis.
let mut cam = Camera::from_focal(CameraModelKind::Pinhole, 500.0, 640, 480);
cam.focal_from_prior = true; // A known focal length selects the calibrated (E) path.
let (mut kps1, mut kps2, mut matches) = (Vec::new(), Vec::new(), Vec::new());
for i in 0..60u32 {
    let (x, y, z) = (((i * 37) % 23) as f64 * 0.2 - 2.2, ((i * 17) % 19) as f64 * 0.2 - 1.8, 4.0 + ((i * 7) % 11) as f64 * 0.3);
    let px = |cx: f64| Keypoint::new((500.0 * cx / z + 320.0) as f32, (500.0 * y / z + 240.0) as f32);
    kps1.push(px(x));
    kps2.push(px(x - 1.0));
    matches.push(FeatureMatch::new(i, i));
}
let tvg = estimate_two_view(&cam, &kps1, &cam, &kps2, &matches, &TwoViewOptions::default());
assert_eq!(tvg.config, TwoViewGeometryConfig::Calibrated);
assert!(tvg.inlier_matches.len() >= 50);
```

At the store level, a single call to `match_pairs(&store, &pairs, &PairMatchingOptions::default(), &CpuMatcher::default())`
does matching, verification and writing. Benchmark: `cargo run --release -p cumulus3d-matching --example bench_match` (8192² matching).

## Feature flags and hardware

- No feature flags. Pure CPU (rayon) implementation.
- Swapping `MatcherBackend` for `CudaMatcher` from `cumulus3d-cuda` runs descriptor matching (integer dot-product GEMM + top-2) on the GPU
  (requires a CUDA 12.x driver). The GPU backend implements only `top2`; the check rules share this crate's default implementation.

## Behavior notes

- Reproducibility: with `TwoViewOptions::ransac.random_seed = Some(s)`, a seed derived from (s, pair id, model kind) is used.
- If raw matches or inliers are below `min_num_inliers` (default 15), empty matches and default geometry (UNDEFINED) are written. A processed pair always writes two rows (matches and geometry).
- Models from failed RANSAC runs are not written (only successful E/F/H).
- Hartley-normalized H (`normalize_homography`) is off by default because it changes results slightly.
- With `skip_geometric_verification`, only raw matches are written (no geometry row).
- Missing image data (camera, keypoints, descriptors) yields `Error::NotFound` before processing.
