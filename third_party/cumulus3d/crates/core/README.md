English | [한국어](https://github.com/AH100-1/cumulus3d/blob/main/crates/core/README.ko.md)

# cumulus3d-core

The common foundation crate of cumulus3d: camera models, geometry (quaternions, rigid and similarity transforms),
feature and match types, an in-memory feature store, a correspondence graph, sparse reconstruction data structures,
generic RANSAC, and file I/O.

API doc comments are currently in Korean; English translation is planned.

**Role in the pipeline**: it does not run a stage of its own; it defines the types and stores exchanged by every stage
(features → matching → SfM → alignment → densification). The feature stage writes to a `FeatureStore`, the matching stage
writes two-view geometries into the same store, and SfM reads correspondences through a `MatchGraph` to build a `Reconstruction`.

## Conventions

- Numbers are f64 (keypoints only are f32). Vectors and matrices use the nalgebra aliases `Vec2, Vec3, Mat3, Mat3x4`.
- IDs: `CameraId/RigId/FrameId/ImageId/Point2DIdx = u32`, `Point3DId/PairId = u64`. Invalid value = the maximum value
  (`INVALID_*`). Image IDs must be < 2^31−1 (because of the pair ID computation).
- Poses: `Rigid3` = A_to_B, `X_B = R X_A + t`. Image poses are **world_to_cam**, center `C = −Rᵀt`.
  `a * b` = a ∘ b (A_to_C = B_to_C * A_to_B).
- Quaternion `Quat {w,x,y,z}`: Hamilton convention, file order w,x,y,z.
- Camera coordinates: x right, y down, z forward. The pixel origin is the top-left corner; **the center of the top-left pixel = (0.5, 0.5)**.
- `Sim3` is new_from_old: `X' = s R X + t`. Use `Sim3::transform_pose` for poses.
- RANSAC residuals are **squared** errors; inlier = residual ≤ max_error².
- Pair data is stored internally in the "smaller id → larger id" direction; lookups in the opposite direction return the inverted data.

## Main entry points

| Function/type | Role | Input → output |
|---|---|---|
| `Reconstruction` | Sparse reconstruction (cameras, rigs, frames, images, 3D points). `clone()` is cheap (copy-on-write) | `new()` → empty model |
| `interop::read_model` | Read a model folder (binary first, text otherwise) | folder path → `Result<Reconstruction>` |
| `interop::write_model_binary` / `write_model_text` | Write a model folder (registered images only) | `&Reconstruction`, folder, `ImageOrder` → `Result<()>` |
| `FeatureStore` | Thread-safe in-memory feature store (images, cameras, keypoints, descriptors, matches, two-view geometries) | `new()` / `load(path)` → store |
| `MatchGraph::from_store` / `update_from_store` | Build / incrementally update a correspondence graph from the store's verified pairs | `&FeatureStore`, `&MatchGraphOptions` → graph / number of new pairs |
| `Camera` | Camera intrinsics with projection and unprojection | `from_focal(model, f, w, h)` → camera; `cam_to_img(&Xc)` → pixel |
| `Rigid3`, `Sim3`, `Quat` | Rigid and similarity transforms, rotations | parameters → transform; `transform_point(&p)` → point |
| `ransac::lo_ransac` / `ransac::ransac` | Generic (LO-)RANSAC | estimator, `RansacParams`, data → `RansacReport` |
| `io::read_ply` / `io::write_ply` | Read/write PLY point clouds | path ↔ `PointCloud` |
| `io::read_gps_file` | Read a GPS reference file (first line = ENU origin) | path → `Vec<GpsRecord>` |
| `analyzer::ModelStats::compute` | Model statistics summary | `&Reconstruction` → `ModelStats` |
| `linalg::svd3` | 3×3 SVD that stays accurate with repeated singular values | `&Mat3` → `Option<(U, σ, V)>` |
| `pair_id_of` / `images_of_pair` | Two images ↔ pair ID | `(ImageId, ImageId)` ↔ `PairId` |

## Public items

"root" marks items that are also re-exported from the crate root (`cumulus3d_core::…`).

### `analyzer`

| Item | Kind | Role |
|---|---|---|
| `ModelStats` | struct | Model statistics (counts of rigs, cameras, frames, images, points; observation count, mean track length, mean reprojection error, etc.) |
| `ModelStats::compute` | fn | Compute statistics from a reconstruction (averages the stored point errors as-is, no recomputation) |
| `ModelStats::lines` | fn | Summary text lines in a fixed order |
| `analyzer_lines` | fn | Model analysis output lines. With `verbose`, appends the camera and registered-image lists |

### `camera`

| Item | Kind | Role |
|---|---|---|
| `CameraModelKind` (root) | enum | Supported models `SingleFocalPinhole=0, Pinhole=1, SingleFocalRadial=2, Radial=3, OpenCv=4`. `from_id, id, name, from_name, num_params, params_info, focal_slots, pp_slots, extra_param_slots` |
| `Camera` (root) | struct | `{camera_id, model, width, height, params, focal_from_prior}`. `focal_from_prior` is in-memory only |
| `Camera::new` / `from_focal` | fn | Construct with parameter-count validation / construct with focal f, principal point (W/2, H/2), zero distortion |
| `Camera::mean_focal`, `focal_length_x/y`, `set_focal_length`, `principal_point_x/y`, `set_principal_point`, `extra_params`, `verify_params` | fn | Parameter access and setting |
| `Camera::is_undistorted`, `calibration_matrix`, `has_implausible_params` | fn | No-distortion check, K matrix, implausible-parameter check |
| `Camera::rescale` / `rescale_to` | fn | Adjust focal and principal point by a uniform factor / to a new size |
| `Camera::cam_to_img`, `normalized_to_img` | fn | Camera coordinates → pixel (None if Z < ε), normalized plane → pixel |
| `Camera::img_to_normalized`, `img_to_ray` | fn | Pixel → normalized plane (undistortion by Newton iteration), pixel → unit ray |
| `Camera::img_to_cam_threshold` | fn | Pixel threshold → normalized-plane threshold (÷ mean focal) |
| `Camera::cam_to_img_with_jacobian` | fn | Projection + ∂pixel/∂camera coords, optionally ∂pixel/∂params (row-major 2×num_params) |
| `UNDISTORT_MAX_ITERS`, `UNDISTORT_STEP_SQ_TOL` | const | Undistortion iteration cap (100), convergence test (‖step‖² < 1e−10) |

### `error`

| Item | Kind | Role |
|---|---|---|
| `Error` (root) | enum | `Io, Parse{path,line,msg}, Format, InvalidArgument, NotFound, AlreadyExists, Invariant, Unsupported` |
| `Result<T>` (root) | type | `std::result::Result<T, Error>` |

### `features`

| Item | Kind | Role |
|---|---|---|
| `DESCRIPTOR_DIM` (root) | const | SIFT descriptor dimension 128 |
| `Keypoint` (root) | struct | Pixel coordinates + affine shape `{x, y, a11, a12, a21, a22}` (f32). `new, from_scale_orientation, from_row(2/4/6 columns), to_row, scale, scale_x, scale_y, orientation, shear, rescale` |
| `Descriptors` (root) | struct | One image's u8 descriptors N×128 (row-major). `new, with_capacity, from_vec, len, is_empty, row, row_mut, push, as_slice, into_vec, truncate` |
| `FeatureMatch` (root) | struct | Keypoint index pair `{idx1, idx2}`. `new, swapped` |
| `TwoViewGeometryConfig` (root) | enum | Two-view configuration kind `Undefined=0 … CalibratedRig=9`. `from_i32, as_i32` |
| `TwoViewGeometry` (root) | struct | `{config, e, f, h, cam1_to_cam2, inlier_matches, tri_angle}`. `inverted/invert` (Fᵀ, Eᵀ, H⁻¹, inverse pose, swapped matches) |

### `geometry`

| Item | Kind | Role |
|---|---|---|
| `Vec2, Vec3, Mat3, Mat3x4` (root) | type | nalgebra f64 aliases |
| `Quat` (root) | struct | Hamilton quaternion. `IDENTITY, new, from_wxyz, to_wxyz, norm, normalized, conjugate, inverse, dot, hamilton, to_rotation_matrix, from_rotation_matrix, from_axis_angle, from_rotation_vector, to_rotation_vector, rotate, angular_distance, to_nalgebra, from_nalgebra`; `Quat * Quat` |
| `Rigid3` (root) | struct | Rigid transform `{rotation, translation}`. `identity, new, from_rotation_matrix, from_params, to_params, rotation_matrix, matrix, from_matrix, transform_point, inverse, compose, center, viewing_direction`; `Rigid3 * Rigid3`, `Rigid3 * Vec3` |
| `Sim3` (root) | struct | Similarity transform `{scale, rotation, translation}`. `identity, new, matrix, from_matrix, transform_point, inverse, compose, transform_pose`; `Sim3 * Sim3` |
| `skew` | fn | Skew-symmetric (cross-product) matrix `[v]×` |
| `deg_to_rad`, `rad_to_deg` | fn | Angle unit conversion |
| `angle_between` | fn | Angle between two vectors (radians) |
| `triangulation_angle`, `max_triangulation_angle` | fn | Triangulation angle between two (or several) projection centers and a point |
| `baseline` | fn | Distance between two projection centers |

### `graph`

| Item | Kind | Role |
|---|---|---|
| `MatchGraph` (root) | struct | Correspondence graph at image and 2D-point level. Images and pairs can be added incrementally |
| `MatchGraph::new`, `from_store`, `update_from_store` | fn | Empty graph / build from a store / apply only newly recorded pairs |
| `MatchGraph::add_image`, `insert_two_view`, `replace_two_view` | fn | Add a node, add a pair (returns `AddPairStats`), update geometry |
| `MatchGraph::find_correspondences`, `has_correspondences`, `in_two_view_track`, `transitive_matches` | fn | Direct and transitive correspondence lookup for a 2D point |
| `MatchGraph::matches_between`, `two_view_geometry`, `image_pairs`, `image_ids` | fn | Matches between two images, direction-aligned geometry, pair list, image list |
| `MatchGraph::exists_image`, `exists_image_pair`, `num_images`, `pair_count`, `num_points2d`, `observation_count_of`, `match_count_of`, `match_count_between` | fn | Count and existence queries |
| `MatchGraph::union_find_tracks` | fn | Connected components (tracks) of correspondences that pass the pair filter, in deterministic order |
| `MatchGraphOptions` (root) | struct | `{min_num_matches, ignore_watermarks, image_names, keep_all_images}` |
| `Correspondence` | struct | One correspondence `{image_id, point2d_idx}` |
| `AddPairStats` | struct | Pair insertion statistics `{num_added, num_out_of_range, num_duplicates, self_pair}` |

### `ids` (all root)

| Item | Kind | Role |
|---|---|---|
| `CameraId, RigId, FrameId, ImageId, Point2DIdx, Point3DId, PairId` | type | Identifier aliases |
| `INVALID_CAMERA_ID, INVALID_RIG_ID, INVALID_FRAME_ID, INVALID_IMAGE_ID, INVALID_POINT2D_IDX, INVALID_POINT3D_ID, INVALID_PAIR_ID` | const | Invalid values (maximum of each type) |
| `MAX_NUM_IMAGES` | const | Pair ID constant M = 2^31 − 1 |
| `SensorKind` | enum | `Invalid=-1, Camera=0, Imu=1`. `from_i32, as_i32, name, from_name` |
| `SensorKey` | struct | Sensor identifier `{sensor_type, id}`. `new, camera` |
| `SensorDataKey` | struct | Data identifier `{sensor_id, id}`. `new, image` |
| `pair_id_of` | fn | Pair ID = M·min + max (error if an image ID ≥ M) |
| `images_of_pair` | fn | Pair ID → (smaller id, larger id) |
| `swap_image_pair` | fn | Whether the order must be swapped when storing (id1 > id2) |

### `interop`

Handles the model file format (`cameras`/`images`/`points3D`, optionally `rigs`/`frames`; `.bin` and `.txt`) and the dense-reconstruction workspace layout.

| Item | Kind | Role |
|---|---|---|
| `read_model` | fn | Read a model folder: binary first, text otherwise |
| `read_model_binary`, `read_model_text` | fn | Read a binary / text model (models without rigs and frames are supported) |
| `write_model_binary`, `write_model_text` | fn | Write a model (all cameras and rigs, posed frames, registered images only, all points; text numbers use %.17g) |
| `ImageOrder` | enum | Image order in the images file: `Registration` (default, registration order) / `ById` |
| `STEREO_SUBDIRS` | const | Subfolder names under `stereo/` (`depth_maps, normal_maps, consistency_graphs`) |
| `create_stereo_dirs` | fn | Create the `stereo/` subfolders of a dense-reconstruction workspace |
| `write_stereo_configs` | fn | Write `stereo/patch-match.cfg` and `stereo/fusion.cfg` |

### `io`

| Item | Kind | Role |
|---|---|---|
| `io::PointCloud` | struct | Point cloud `{positions, normals, colors}`. `len, is_empty, has_normals, has_colors` |
| `io::PlyLayout` | enum | Property layout for writing: `XyzRgbNormal` (default) / `XyzNormalRgb` |
| `io::read_ply` | fn | Read PLY (ascii / binary LE / binary BE, by property name) |
| `io::write_ply` | fn | Write binary LE PLY (normals omitted if absent, white if no colors) |
| `io::GpsRecord` | struct | One GPS reference line `{name, lat, lon, alt}` |
| `io::read_gps_file` | fn | Read `name latitude longitude altitude` lines (order preserved, first line = ENU origin) |
| `io::fmt::format_g`, `io::fmt::g17` | fn | Locale-independent `%.{p}g` / `%.17g` number formatting |

### `linalg`

| Item | Kind | Role |
|---|---|---|
| `svd3` | fn | One-sided Jacobi 3×3 SVD (σ descending, det U = +1, accurate with repeated singular values). Shared by matching, sfm and align |

### `ransac`

| Item | Kind | Role |
|---|---|---|
| `Estimator` | trait | Minimal/non-minimal solvers + squared residuals (`X, Y, Model`, `min_num_samples, estimate, residuals`) |
| `Sampler` | trait | Sampler (`initialize, max_num_samples, sample`) |
| `RandomSubsetSampler` | struct | Random samples (partial Fisher–Yates continuous shuffle) |
| `ExhaustiveSampler` | struct | All k-combinations in lexicographic order |
| `RansacParams` | struct | `{max_error, min_inlier_ratio, confidence, dyn_trials_factor, min_trials, max_trials, random_seed}` |
| `Support` | struct | Support `{num_inliers, residual_sum}`. `measure, is_better` |
| `RansacReport<M>` | struct | `{success, num_trials, support, inlier_mask, model}` |
| `ransac` | fn | Basic RANSAC |
| `lo_ransac` | fn | LO-RANSAC (adds a non-minimal estimator for local optimization) |
| `ransac_with_sampler` | fn | General form with a chosen sampler |
| `compute_num_trials`, `static_max_num_trials` | fn | Required number of trials / static cap at construction |
| `make_rng` | fn | Create a seeded (or random) Pcg64 |
| `n_choose_k` | fn | Binomial coefficient (saturating) |

### `reconstruction`

| Item | Kind | Role |
|---|---|---|
| `Reconstruction` (root) | struct | Sparse reconstruction. Every public mutating operation preserves the invariants (bidirectional links, etc.) |
| `Reconstruction::new`, `cameras/camera/camera_mut`, `rigs/rig/rig_mut`, `frames/frame`, `images/image/image_ids/image_by_name/exists_image`, `points3d/point3d/point3d_ids/exists_point3d` | fn | Construction and lookup |
| `Reconstruction::num_cameras/num_rigs/num_frames/num_images/num_points3d`, `registered_frame_count`, `registered_image_count`, `max_point3d_id` | fn | Counts |
| `Reconstruction::add_camera`, `add_camera_own_rig`, `add_rig`, `add_frame`, `add_image`, `add_image_own_frame` | fn | Building (including variants that auto-create trivial rigs and frames) |
| `Reconstruction::world_to_cam`, `projection_center`, `has_pose`, `set_world_to_cam`, `set_frame_pose` | fn | Pose lookup and setting |
| `Reconstruction::register_frame/register_image`, `deregister_frame/deregister_image`, `is_frame_registered/is_image_registered`, `registered_frames`, `registered_images` | fn | Registration and deregistration (deregistration deletes observations) |
| `Reconstruction::add_point3d`, `add_point3d_with_id`, `add_observation`, `delete_observation`, `delete_point3d`, `merge_points3d`, `delete_all_points3d`, `set_point3d_xyz/error/color` | fn | Editing 3D points and observations |
| `Reconstruction::squared_reprojection_error`, `update_point3d_errors`, `point3d_reprojection_error` | fn | Reprojection error computation and update |
| `Reconstruction::filter_observations`, `filter_observations_with_large_reprojection_error`, `filter_points3d_with_small_triangulation_angle` | fn | Observation and point filters |
| `Reconstruction::total_observations`, `mean_track_len`, `mean_obs_per_registered_image`, `mean_reproj_error` | fn | Statistics |
| `Reconstruction::transform`, `normalize` | fn | Apply a Sim3 / normalize extent (returns the applied Sim3) |
| `Reconstruction::deregister_images_by_id/by_name`, `remove_unregistered` | fn | Deregister images (returns a list of warnings), delete unregistered elements |
| `Reconstruction::extract_colors`, `check_invariants` | fn | Extract point colors with an image sampler, check invariants |
| `Rig` (root) | struct | `{rig_id, ref_sensor_id, sensors}`. `new, trivial, num_sensors, is_reference_sensor, has_sensor, rig_to_sensor` |
| `Frame` (root) | struct | `{frame_id, rig_id, world_to_rig}`. `new, attach_data, data_ids, has_pose, image_ids` |
| `Image` (root) | struct | `{image_id, name, camera_id, frame_id}`. `new, points2d, point2d, num_points2d, num_points3d` |
| `Point2D` (root) | struct | `{xy, point3d_id}`. `new, has_point3d` |
| `Point3D` (root) | struct | `{xyz, color, error, track}`. `new, has_error, track_len` |
| `TrackEntry` (root) | struct | Track element `{image_id, point2d_idx}`. `new` |
| `FilterErrorUpdate` | enum | How point errors are updated after filtering: `None, SumOverOriginalLength, MeanOfRemaining` |
| `NormalizeOptions` | struct | `{fixed_scale, extent, p0, p1, use_images}` |
| `bilinear_rgb` | fn | Bilinear interpolation of a u8 RGB image (None outside bounds) |

### `store`

| Item | Kind | Role |
|---|---|---|
| `FeatureStore` (root) | struct | Thread-safe (RwLock) in-memory feature store; large data is shared via `Arc`. All methods take `&self` |
| `FeatureStore::new`, `save`, `load` | fn | Create; save/load in its own binary format |
| `FeatureStore::add_camera`, `update_camera`, `camera`, `cameras`, `num_cameras` | fn | Cameras (an invalid id is assigned max+1) |
| `FeatureStore::add_image`, `add_image_with_id`, `image`, `image_by_name`, `image_id_by_name`, `images`, `image_ids`, `num_images`, `exists_image` | fn | Images (names are unique) |
| `FeatureStore::set_pose_prior`, `pose_prior` | fn | Position priors |
| `FeatureStore::set_keypoints`, `set_descriptors`, `keypoints`, `descriptors`, `exists_keypoints`, `exists_descriptors`, `num_keypoints`, `largest_keypoint_count` | fn | Features |
| `FeatureStore::write_matches`, `read_matches`, `exists_matches`, `delete_matches`, `num_matched_pairs` | fn | Raw matches (callable in either direction) |
| `FeatureStore::put_two_view`, `get_two_view`, `contains_two_view`, `remove_two_view`, `two_view_geometries`, `two_view_geometries_since`, `num_two_view_geometries` | fn | Two-view geometries (with a log position for incremental consumption) |
| `FeatureStore::retain_copy` | fn | A copy keeping only images that satisfy a predicate (for rewinding progressive processing; large data shared) |
| `StoreImage` | struct | Store image row `{image_id, name, camera_id}` |
| `PosePrior` | struct | Position prior `{position, coordinate_system, gravity}` (0 = WGS84) |

## Example

Build a reconstruction from two images and one 3D point, compute the reprojection error, then save and load it as model files
(same as the doc-test in `src/lib.rs`).

```rust
use cumulus3d_core::analyzer::ModelStats;
use cumulus3d_core::interop::{read_model, write_model_binary, ImageOrder};
use cumulus3d_core::{Camera, CameraModelKind, Image, Quat, Reconstruction, Rigid3, TrackEntry, Vec3};

fn main() -> cumulus3d_core::Result<()> {
    let mut cam = Camera::from_focal(CameraModelKind::Pinhole, 1000.0, 1920, 1080);
    cam.camera_id = 1;
    let mut rec = Reconstruction::new();
    rec.add_camera_own_rig(cam.clone())?;

    // Project one world point into two cameras (1 m baseline) to create 2D observations.
    let x = Vec3::new(0.2, -0.1, 5.0);
    for (id, tx) in [(1u32, 0.0), (2, -1.0)] {
        let world_to_cam = Rigid3::new(Quat::IDENTITY, Vec3::new(tx, 0.0, 0.0));
        let xy = cam.cam_to_img(&world_to_cam.transform_point(&x)).unwrap();
        rec.add_image_own_frame(Image::new(id, format!("img{id}.jpg"), 1, [xy]), Some(world_to_cam))?;
        rec.register_image(id)?;
    }
    let pid = rec.add_point3d(x, vec![TrackEntry::new(1, 0), TrackEntry::new(2, 0)], [255, 0, 0])?;
    rec.update_point3d_errors();
    assert!(rec.point3d(pid).unwrap().error < 1e-9);
    rec.check_invariants()?;

    let dir = std::env::temp_dir().join("cumulus3d_core_doc_example");
    std::fs::create_dir_all(&dir)?;
    write_model_binary(&rec, &dir, ImageOrder::default())?;
    let back = read_model(&dir)?;
    let stats = ModelStats::compute(&back);
    assert_eq!((stats.registered_image_count, stats.num_points3d), (2, 1));
    Ok(())
}
```

## Feature flags and hardware

None. This is a pure-CPU crate with no feature flags. Dependencies: nalgebra, rayon (parallel observation filtering), rand/rand_pcg, thiserror.
