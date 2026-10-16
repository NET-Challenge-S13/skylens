[English](https://github.com/AH100-1/cumulus3d/blob/main/crates/ba/README.md) | 한국어

# cumulus3d-ba

번들 조정(BA)과 단일 자세 정제. 신뢰 영역 Levenberg–Marquardt 위에 점 Schur 소거와 밀집/희소 촐레스키·PCG
축소 계통 풀이기를 얹었다. 결과는 스레드 수와 무관하게 결정적이다.

**파이프라인 단계**: SfM(`cumulus3d-sfm`)이 영상 등록 직후 절대 자세 정제(`refine_abs_pose`)와
구역 정밀본의 국소·전역 번들 조정(`bundle_adjust`)에 쓴다. 단독 명령 `cumulus3d bundle_adjuster` 도 이 함수를 부른다.

## 주요 진입점

| 함수/타입 | 역할 | 입력 → 출력 |
|---|---|---|
| `bundle_adjust` | 재구성 전체(또는 일부 영상)에 번들 조정을 적용해 자세·점·카메라를 제자리 갱신 | `&mut Reconstruction`, `&BaConfig` → `Result<BaSummary>` |
| `BaConfig` | 정제·고정할 변수, 손실, 허용오차, 풀이기 선택 (`Default` = 단독 번들 조정 기본값) | 구조체 |
| `BaSummary` | 반복 수·비용·종료 상태 요약. `is_usable()`, `rms_reprojection_error()` | 구조체 |
| `refine_abs_pose` | 2D–3D 대응으로 카메라 자세 하나(6자유도)만 정제 | `&Camera`, `&[Vec2]`, `&[Vec3]`, `&[bool]`, `&mut Rigid3`, `Loss`, 반복 수 → `Result<BaSummary>` |
| `Loss` | 견고 손실(`Trivial`, `SoftL1`, `Cauchy`, `Huber`; 스케일 = 픽셀) | 열거형 |
| `LinearSolverType` | 축소 계통 풀이기(`Auto`, `DenseSchur`, `SparseSchur`, `IterativeSchur`) | 열거형 |
| `filter_negative_depth_observations` | 카메라 뒤(깊이 < ε) 관측 삭제(BA 사전 처리) | `&mut Reconstruction`, `&[ImageId]` → `Result<usize>` |
| `select_linear_solver` | `Auto` 일 때 영상 수로 풀이기 결정 | 영상 수, `&BaConfig` → `LinearSolverType` |

## 공개 항목

모든 항목은 크레이트 루트(`cumulus3d_ba::`)에 있다(하위 모듈은 비공개).

| 항목 | 종류 | 역할 |
|---|---|---|
| `bundle_adjust` | fn | 번들 조정. 풀이기가 `Failure` 로 끝나도 마지막 채택 해를 써 넣고 `Ok` (`summary.is_usable()` 로 확인) |
| `refine_abs_pose` | fn | 절대 자세 정제(점·내부 고정, 허용치 기울기 1.0 / 함수 1e−6 / 파라미터 1e−8). 실패 시 `Err`, 자세 불변 |
| `filter_negative_depth_observations` | fn | 깊이 < ε 관측 삭제(트랙 ≤ 2 면 점째 삭제), 삭제 수 반환 |
| `select_linear_solver` | fn | `Auto`: ≤ `dense_solver_image_limit`(50) 밀집, ≤ `sparse_solver_image_limit`(1000) 희소, 그 외 PCG |
| `BaConfig` | struct | BA 설정(아래 표) |
| `BaSummary` | struct | 결과 요약: `num_iterations, initial_cost, final_cost(½Σρ), num_residuals, converged, termination, num_successful_steps, linear_solver, num_images, num_points, free_point_count, dropped_observations` |
| `BaSummary::is_usable` | fn | 종료 상태가 `Failure` 가 아니면 참 |
| `BaSummary::rms_reprojection_error` | fn | √(2·비용 / 관측 수) (픽셀, 손실 없을 때 의미) |
| `Loss` | enum | `Trivial`, `SoftL1(a)`, `Cauchy(a)`, `Huber(a)` |
| `LinearSolverType` | enum | `Auto`(기본), `DenseSchur`, `SparseSchur`(faer), `IterativeSchur`(PCG + 블록 야코비) |
| `Termination` | enum | `Convergence`, `NoConvergence`(기본), `Failure` |

### `BaConfig` 기본값

| 필드 | 기본 |
|---|---|
| `refine_focal_length` / `refine_principal_point` / `refine_extra_params` | true / false / true |
| `refine_poses` / `refine_points` | true / true |
| `loss` | `Loss::Trivial` |
| `max_num_iterations` | 100 |
| `function_tolerance` / `gradient_tolerance` / `parameter_tolerance` | 0 / 1e−4 / 0 |
| `images`, `constant_poses`, `constant_cameras`, `constant_points` | 빈 집합(`images` 가 비면 등록 영상 전부) |
| `auto_gauge` | true (상수 자세가 없을 때 두 영상 고정으로 게이지 고정) |
| `num_threads` | 0 (전역 rayon 풀; > 0 이면 전용 풀) |
| `constant_world_to_rig_rotation` | false |
| `min_track_length` | 0 (끔) |
| `linear_solver` / `max_linear_solver_iterations` | `Auto` / 200 |
| `dense_solver_image_limit` / `sparse_solver_image_limit` | 50 / 1000 |
| `filter_negative_depth` / `update_point_errors` | true / true |

자주 쓰는 조합: 삼각측량 뒤 점만 정제 → `refine_poses=false, refine_focal_length=false, refine_extra_params=false`;
국소 BA 견고 손실 → `loss: Loss::SoftL1(1.0)`.

## 사용 예

합성 대응으로 절대 자세를 정제한다(크레이트 문서의 doc-test 와 같은 코드).

```rust
use cumulus3d_ba::{refine_abs_pose, Loss};
use cumulus3d_core::{Camera, CameraModelKind, Quat, Rigid3, Vec2, Vec3};

// 합성 장면: 참 자세로 3D 점을 투영해 2D 관측을 만든다.
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

// 흐트러진 초기 자세에서 절대 자세 정제.
let mut pose = Rigid3::new(Quat::from_axis_angle(&Vec3::y(), 0.12), Vec3::new(0.25, -0.05, 5.9));
let summary = refine_abs_pose(&cam, &pts2d, &pts3d, &mask, &mut pose, Loss::Cauchy(1.0), 100).unwrap();
assert!(summary.is_usable());
assert!((pose.translation - truth.translation).norm() < 1e-6);
```

재구성 전체 번들 조정:

```rust,no_run
use cumulus3d_ba::{bundle_adjust, BaConfig};
let mut rec = cumulus3d_core::interop::read_model("model/0").unwrap();
let summary = bundle_adjust(&mut rec, &BaConfig::default()).unwrap();
println!("RMS {:.3} px, 반복 {}", summary.rms_reprojection_error(), summary.num_iterations);
```

## 기능 플래그·하드웨어

- 기능 플래그 없음. CPU 전용(rayon 병렬, 희소 촐레스키는 faer).
- 비자명 rig 는 프레임 자세를 변수로, rig_to_sensor 를 상수로 둔다(센서 상대 자세 정제는 하지 않음).
- 성능 측정: `cargo run --release -p cumulus3d-ba --example ba_bench [위치수 점수 풀이기]`
  (240장·점 20만·관측 177만, 희소 풀이기에서 BA 1회 약 5.6 s, M 계열 10코어).
