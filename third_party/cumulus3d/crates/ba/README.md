English | [한국어](https://github.com/AH100-1/cumulus3d/blob/main/crates/ba/README.ko.md)

# cumulus3d-ba

Bundle adjustment (BA) and single-pose refinement. A trust-region Levenberg–Marquardt solver with point Schur elimination
and dense/sparse Cholesky and PCG solvers for the reduced system. Results are deterministic regardless of the thread count.

API doc comments are currently in Korean; English translation is planned.

**Pipeline stage**: SfM (`cumulus3d-sfm`) uses it for absolute pose refinement right after image registration (`refine_abs_pose`)
and for local and global bundle adjustment of refined zones (`bundle_adjust`). The standalone command `cumulus3d bundle_adjuster` also calls these functions.

## Main entry points

| Function/type | Role | Input → output |
|---|---|---|
| `bundle_adjust` | Apply bundle adjustment to the whole reconstruction (or a subset of images), updating poses, points and cameras in place | `&mut Reconstruction`, `&BaConfig` → `Result<BaSummary>` |
| `BaConfig` | Variables to refine or fix, loss, tolerances, solver choice (`Default` = standalone bundle adjustment defaults) | struct |
| `BaSummary` | Summary of iterations, cost and termination state. `is_usable()`, `rms_reprojection_error()` | struct |
| `refine_abs_pose` | Refine a single camera pose (6 DoF) from 2D–3D correspondences | `&Camera`, `&[Vec2]`, `&[Vec3]`, `&[bool]`, `&mut Rigid3`, `Loss`, iteration count → `Result<BaSummary>` |
| `Loss` | Robust loss (`Trivial`, `SoftL1`, `Cauchy`, `Huber`; scale = pixels) | enum |
| `LinearSolverType` | Reduced-system solver (`Auto`, `DenseSchur`, `SparseSchur`, `IterativeSchur`) | enum |
| `filter_negative_depth_observations` | Delete observations behind the camera (depth < ε) (BA preprocessing) | `&mut Reconstruction`, `&[ImageId]` → `Result<usize>` |
| `select_linear_solver` | Choose the solver by image count when `Auto` | image count, `&BaConfig` → `LinearSolverType` |

## Public items

All items live at the crate root (`cumulus3d_ba::`) (submodules are private).

| Item | Kind | Role |
|---|---|---|
| `bundle_adjust` | fn | Bundle adjustment. Even if the solver ends in `Failure`, the last accepted solution is written back and `Ok` is returned (check with `summary.is_usable()`) |
| `refine_abs_pose` | fn | Absolute pose refinement (points and intrinsics fixed; tolerances gradient 1.0 / function 1e−6 / parameter 1e−8). On failure returns `Err` and leaves the pose unchanged |
| `filter_negative_depth_observations` | fn | Delete observations with depth < ε (the whole point if its track ≤ 2), returns the number deleted |
| `select_linear_solver` | fn | `Auto`: dense if ≤ `dense_solver_image_limit`(50), sparse if ≤ `sparse_solver_image_limit`(1000), otherwise PCG |
| `BaConfig` | struct | BA configuration (table below) |
| `BaSummary` | struct | Result summary: `num_iterations, initial_cost, final_cost(½Σρ), num_residuals, converged, termination, num_successful_steps, linear_solver, num_images, num_points, free_point_count, dropped_observations` |
| `BaSummary::is_usable` | fn | True unless the termination state is `Failure` |
| `BaSummary::rms_reprojection_error` | fn | √(2·cost / number of observations) (pixels, meaningful without a loss) |
| `Loss` | enum | `Trivial`, `SoftL1(a)`, `Cauchy(a)`, `Huber(a)` |
| `LinearSolverType` | enum | `Auto` (default), `DenseSchur`, `SparseSchur` (faer), `IterativeSchur` (PCG + block Jacobi) |
| `Termination` | enum | `Convergence`, `NoConvergence` (default), `Failure` |

### `BaConfig` defaults

| Field | Default |
|---|---|
| `refine_focal_length` / `refine_principal_point` / `refine_extra_params` | true / false / true |
| `refine_poses` / `refine_points` | true / true |
| `loss` | `Loss::Trivial` |
| `max_num_iterations` | 100 |
| `function_tolerance` / `gradient_tolerance` / `parameter_tolerance` | 0 / 1e−4 / 0 |
| `images`, `constant_poses`, `constant_cameras`, `constant_points` | empty sets (if `images` is empty, all registered images) |
| `auto_gauge` | true (when there are no constant poses, fixes the gauge by holding two images constant) |
| `num_threads` | 0 (global rayon pool; > 0 uses a dedicated pool) |
| `constant_world_to_rig_rotation` | false |
| `min_track_length` | 0 (off) |
| `linear_solver` / `max_linear_solver_iterations` | `Auto` / 200 |
| `dense_solver_image_limit` / `sparse_solver_image_limit` | 50 / 1000 |
| `filter_negative_depth` / `update_point_errors` | true / true |

Common combinations: refine points only after triangulation → `refine_poses=false, refine_focal_length=false, refine_extra_params=false`;
robust loss for local BA → `loss: Loss::SoftL1(1.0)`.

## Example

Refine an absolute pose from synthetic correspondences (same code as the doc-test in the crate documentation).

```rust
use cumulus3d_ba::{refine_abs_pose, Loss};
use cumulus3d_core::{Camera, CameraModelKind, Quat, Rigid3, Vec2, Vec3};

// Synthetic scene: project 3D points with the true pose to create 2D observations.
let cam = Camera::from_focal(CameraModelKind::Pinhole, 1000.0, 1920, 1080);
let truth = Rigid3::new(Quat::from_axis_angle(&Vec3::y(), 0.1), Vec3::new(0.2, -0.1, 6.0));
let pts3d: Vec<Vec3> = (0..60)
    .map(|i| {
        let a = i as f64;
        Vec3::new((a * 0.37).sin() * 3.0, (a * 0.71).cos() * 2.0, (a * 0.13).sin())
    })
    .collect();
let pts2d: Vec<Vec2> = pts3d.iter().map(|p| cam.cam_to_img(&truth.transform_point(p)).unwrap()).collect();
let mask = vec![true; pts3d.len()];

// Refine the absolute pose from a perturbed initial pose.
let mut pose = Rigid3::new(Quat::from_axis_angle(&Vec3::y(), 0.12), Vec3::new(0.25, -0.05, 5.9));
let summary = refine_abs_pose(&cam, &pts2d, &pts3d, &mask, &mut pose, Loss::Cauchy(1.0), 100).unwrap();
assert!(summary.is_usable());
assert!((pose.translation - truth.translation).norm() < 1e-6);
```

Bundle adjustment of a whole reconstruction:

```rust,no_run
use cumulus3d_ba::{bundle_adjust, BaConfig};
let mut rec = cumulus3d_core::interop::read_model("model/0").unwrap();
let summary = bundle_adjust(&mut rec, &BaConfig::default()).unwrap();
println!("RMS {:.3} px, {} iterations", summary.rms_reprojection_error(), summary.num_iterations);
```

## Feature flags and hardware

- No feature flags. CPU only (rayon parallelism, sparse Cholesky via faer).
- Non-trivial rigs treat frame poses as variables and rig_to_sensor as constant (relative sensor poses are not refined).
- Benchmark: `cargo run --release -p cumulus3d-ba --example ba_bench [positions points solver]`
  (240 images, 200k points, 1.77M observations: one BA run takes about 5.6 s with the sparse solver on a 10-core M-series machine).
