[English](https://github.com/AH100-1/cumulus3d/blob/main/crates/core/README.md) | 한국어

# cumulus3d-core

cumulus3d 의 공통 기반 크레이트: 카메라 모델, 기하(쿼터니언·강체·상사 변환), 특징·매칭 자료형,
메모리 특징 저장소, 대응 그래프, 희소 재구성 자료 구조, 범용 RANSAC, 파일 입출력을 제공한다.

**파이프라인에서 맡는 단계**: 특정 단계를 수행하지 않고, 모든 단계(특징 → 매칭 → SfM → 정렬 → 조밀화)가
주고받는 자료형과 저장소를 정의한다. 특징 단계는 `FeatureStore` 에 쓰고, 매칭 단계는 같은 저장소에
두 뷰 기하를 쓰며, SfM 은 `MatchGraph` 로 대응을 읽어 `Reconstruction` 을 만든다.

## 규약

- 수치는 f64(키포인트만 f32). 벡터·행렬은 nalgebra 별칭 `Vec2, Vec3, Mat3, Mat3x4`.
- id: `CameraId/RigId/FrameId/ImageId/Point2DIdx = u32`, `Point3DId/PairId = u64`. 무효값 = 최댓값
  (`INVALID_*`). 영상 id < 2^31−1(짝 id 계산 때문).
- 자세: `Rigid3` = A_to_B, `X_B = R X_A + t`. 영상 자세는 **world_to_cam**, 중심 `C = −Rᵀt`.
  `a * b` = a ∘ b (A_to_C = B_to_C * A_to_B).
- 쿼터니언 `Quat {w,x,y,z}`: 해밀턴 규약, 파일 순서 w,x,y,z.
- 카메라 좌표: x 오른쪽, y 아래, z 앞. 픽셀 원점은 좌상단 모서리, **좌상단 화소 중심 = (0.5, 0.5)**.
- `Sim3` 는 new_from_old: `X' = s R X + t`. 자세에는 `Sim3::transform_pose` 를 쓴다.
- RANSAC 잔차는 **제곱** 오차, 인라이어 = 잔차 ≤ max_error².
- 짝 자료는 내부적으로 "작은 id → 큰 id" 방향으로 저장하며, 반대 방향 조회 시 역변환해 돌려준다.

## 주요 진입점

| 함수/타입 | 역할 | 입력 → 출력 |
|---|---|---|
| `Reconstruction` | 희소 재구성(카메라·rig·프레임·영상·3D 점). `clone()` 은 copy-on-write 라 저렴 | `new()` → 빈 모델 |
| `interop::read_model` | 모델 폴더 읽기(이진 우선, 없으면 텍스트) | 폴더 경로 → `Result<Reconstruction>` |
| `interop::write_model_binary` / `write_model_text` | 모델 폴더 쓰기(등록 영상만) | `&Reconstruction`, 폴더, `ImageOrder` → `Result<()>` |
| `FeatureStore` | 스레드 안전 메모리 특징 저장소(영상·카메라·키포인트·기술자·매칭·두 뷰 기하) | `new()` / `load(path)` → 저장소 |
| `MatchGraph::from_store` / `update_from_store` | 저장소의 검증된 짝으로 대응 그래프 구성·증분 갱신 | `&FeatureStore`, `&MatchGraphOptions` → 그래프 / 새 짝 수 |
| `Camera` | 카메라 내부 파라미터와 투영·역투영 | `from_focal(model, f, w, h)` → 카메라; `cam_to_img(&Xc)` → 픽셀 |
| `Rigid3`, `Sim3`, `Quat` | 강체·상사 변환, 회전 | 파라미터 → 변환; `transform_point(&p)` → 점 |
| `ransac::lo_ransac` / `ransac::ransac` | 범용 (LO-)RANSAC | 추정기, `RansacParams`, 자료 → `RansacReport` |
| `io::read_ply` / `io::write_ply` | PLY 점군 읽기·쓰기 | 경로 ↔ `PointCloud` |
| `io::read_gps_file` | GPS 참조 파일 읽기(첫 줄 = ENU 원점) | 경로 → `Vec<GpsRecord>` |
| `analyzer::ModelStats::compute` | 모델 통계 요약 | `&Reconstruction` → `ModelStats` |
| `linalg::svd3` | 겹친 특이값에도 정확한 3×3 SVD | `&Mat3` → `Option<(U, σ, V)>` |
| `pair_id_of` / `images_of_pair` | 영상 두 개 ↔ 짝 id | `(ImageId, ImageId)` ↔ `PairId` |

## 공개 항목

"루트" 표시는 크레이트 루트(`cumulus3d_core::…`)에서도 재노출되는 항목이다.

### `analyzer`

| 항목 | 종류 | 역할 |
|---|---|---|
| `ModelStats` | struct | 모델 통계(rig·카메라·프레임·영상·점 수, 관측 수, 평균 트랙 길이, 평균 재투영 오차 등) |
| `ModelStats::compute` | fn | 재구성에서 통계 계산(저장된 점 오차를 그대로 평균, 재계산 안 함) |
| `ModelStats::lines` | fn | 고정 순서의 요약 문자열 줄 |
| `analyzer_lines` | fn | 모델 분석 출력 줄. `verbose` 면 카메라·등록 영상 목록을 덧붙임 |

### `camera`

| 항목 | 종류 | 역할 |
|---|---|---|
| `CameraModelKind` (루트) | enum | 지원 모델 `SingleFocalPinhole=0, Pinhole=1, SingleFocalRadial=2, Radial=3, OpenCv=4`. `from_id, id, name, from_name, num_params, params_info, focal_slots, pp_slots, extra_param_slots` |
| `Camera` (루트) | struct | `{camera_id, model, width, height, params, focal_from_prior}`. `focal_from_prior` 는 메모리 전용 |
| `Camera::new` / `from_focal` | fn | 파라미터 개수 검증 생성 / 초점 f·주점 (W/2, H/2)·왜곡 0 으로 생성 |
| `Camera::mean_focal`, `focal_length_x/y`, `set_focal_length`, `principal_point_x/y`, `set_principal_point`, `extra_params`, `verify_params` | fn | 파라미터 접근·설정 |
| `Camera::is_undistorted`, `calibration_matrix`, `has_implausible_params` | fn | 왜곡 없음 판정, K 행렬, 비정상 파라미터 판정 |
| `Camera::rescale` / `rescale_to` | fn | 균일 배율 / 새 크기로 초점·주점 조정 |
| `Camera::cam_to_img`, `normalized_to_img` | fn | 카메라 좌표 → 픽셀(Z < ε 이면 None), 정규화 평면 → 픽셀 |
| `Camera::img_to_normalized`, `img_to_ray` | fn | 픽셀 → 정규화 평면(뉴턴 반복 왜곡 제거), 픽셀 → 단위 광선 |
| `Camera::img_to_cam_threshold` | fn | 픽셀 임계 → 정규화 평면 임계(÷ 평균 초점) |
| `Camera::cam_to_img_with_jacobian` | fn | 투영 + ∂픽셀/∂카메라좌표, 선택적으로 ∂픽셀/∂파라미터(행 우선 2×num_params) |
| `UNDISTORT_MAX_ITERS`, `UNDISTORT_STEP_SQ_TOL` | const | 왜곡 제거 반복 상한(100), 수렴 판정(‖step‖² < 1e−10) |

### `error`

| 항목 | 종류 | 역할 |
|---|---|---|
| `Error` (루트) | enum | `Io, Parse{path,line,msg}, Format, InvalidArgument, NotFound, AlreadyExists, Invariant, Unsupported` |
| `Result<T>` (루트) | type | `std::result::Result<T, Error>` |

### `features`

| 항목 | 종류 | 역할 |
|---|---|---|
| `DESCRIPTOR_DIM` (루트) | const | SIFT 기술자 차원 128 |
| `Keypoint` (루트) | struct | 픽셀 좌표 + 아핀 형상 `{x, y, a11, a12, a21, a22}`(f32). `new, from_scale_orientation, from_row(2/4/6열), to_row, scale, scale_x, scale_y, orientation, shear, rescale` |
| `Descriptors` (루트) | struct | 영상 하나의 u8 기술자 N×128(행 우선). `new, with_capacity, from_vec, len, is_empty, row, row_mut, push, as_slice, into_vec, truncate` |
| `FeatureMatch` (루트) | struct | 키포인트 인덱스 쌍 `{idx1, idx2}`. `new, swapped` |
| `TwoViewGeometryConfig` (루트) | enum | 두 뷰 구성 종류 `Undefined=0 … CalibratedRig=9`. `from_i32, as_i32` |
| `TwoViewGeometry` (루트) | struct | `{config, e, f, h, cam1_to_cam2, inlier_matches, tri_angle}`. `inverted/invert`(Fᵀ, Eᵀ, H⁻¹, 자세 역, 매칭 교환) |

### `geometry`

| 항목 | 종류 | 역할 |
|---|---|---|
| `Vec2, Vec3, Mat3, Mat3x4` (루트) | type | nalgebra f64 별칭 |
| `Quat` (루트) | struct | 해밀턴 쿼터니언. `IDENTITY, new, from_wxyz, to_wxyz, norm, normalized, conjugate, inverse, dot, hamilton, to_rotation_matrix, from_rotation_matrix, from_axis_angle, from_rotation_vector, to_rotation_vector, rotate, angular_distance, to_nalgebra, from_nalgebra`; `Quat * Quat` |
| `Rigid3` (루트) | struct | 강체 변환 `{rotation, translation}`. `identity, new, from_rotation_matrix, from_params, to_params, rotation_matrix, matrix, from_matrix, transform_point, inverse, compose, center, viewing_direction`; `Rigid3 * Rigid3`, `Rigid3 * Vec3` |
| `Sim3` (루트) | struct | 상사 변환 `{scale, rotation, translation}`. `identity, new, matrix, from_matrix, transform_point, inverse, compose, transform_pose`; `Sim3 * Sim3` |
| `skew` | fn | 반대칭(외적) 행렬 `[v]×` |
| `deg_to_rad`, `rad_to_deg` | fn | 각도 단위 변환 |
| `angle_between` | fn | 두 벡터 사이 각(라디안) |
| `triangulation_angle`, `max_triangulation_angle` | fn | 두(여러) 투영 중심과 점의 삼각측량 각 |
| `baseline` | fn | 두 투영 중심 사이 거리 |

### `graph`

| 항목 | 종류 | 역할 |
|---|---|---|
| `MatchGraph` (루트) | struct | 영상·2D 점 단위 대응 그래프. 영상·짝을 증분으로 추가 가능 |
| `MatchGraph::new`, `from_store`, `update_from_store` | fn | 빈 그래프 / 저장소에서 구성 / 새로 기록된 짝만 반영 |
| `MatchGraph::add_image`, `insert_two_view`, `replace_two_view` | fn | 노드 추가, 짝 추가(`AddPairStats` 반환), 기하 갱신 |
| `MatchGraph::find_correspondences`, `has_correspondences`, `in_two_view_track`, `transitive_matches` | fn | 2D 점의 직접·추이적 대응 조회 |
| `MatchGraph::matches_between`, `two_view_geometry`, `image_pairs`, `image_ids` | fn | 두 영상 매칭, 방향 맞춘 기하, 짝 목록, 영상 목록 |
| `MatchGraph::exists_image`, `exists_image_pair`, `num_images`, `pair_count`, `num_points2d`, `observation_count_of`, `match_count_of`, `match_count_between` | fn | 개수·존재 조회 |
| `MatchGraph::union_find_tracks` | fn | 짝 필터를 통과한 대응의 연결 성분(트랙), 결정적 순서 |
| `MatchGraphOptions` (루트) | struct | `{min_num_matches, ignore_watermarks, image_names, keep_all_images}` |
| `Correspondence` | struct | 대응 하나 `{image_id, point2d_idx}` |
| `AddPairStats` | struct | 짝 추가 통계 `{num_added, num_out_of_range, num_duplicates, self_pair}` |

### `ids` (모두 루트)

| 항목 | 종류 | 역할 |
|---|---|---|
| `CameraId, RigId, FrameId, ImageId, Point2DIdx, Point3DId, PairId` | type | 식별자 별칭 |
| `INVALID_CAMERA_ID, INVALID_RIG_ID, INVALID_FRAME_ID, INVALID_IMAGE_ID, INVALID_POINT2D_IDX, INVALID_POINT3D_ID, INVALID_PAIR_ID` | const | 무효값(각 자료형 최댓값) |
| `MAX_NUM_IMAGES` | const | 짝 id 상수 M = 2^31 − 1 |
| `SensorKind` | enum | `Invalid=-1, Camera=0, Imu=1`. `from_i32, as_i32, name, from_name` |
| `SensorKey` | struct | 센서 식별자 `{sensor_type, id}`. `new, camera` |
| `SensorDataKey` | struct | 데이터 식별자 `{sensor_id, id}`. `new, image` |
| `pair_id_of` | fn | 짝 id = M·min + max (영상 id ≥ M 이면 오류) |
| `images_of_pair` | fn | 짝 id → (작은 id, 큰 id) |
| `swap_image_pair` | fn | 저장 시 순서 교환이 필요한지(id1 > id2) |

### `interop`

모델 파일 형식(`cameras`/`images`/`points3D`, 선택적으로 `rigs`/`frames`; `.bin`·`.txt`)과 조밀 복원 작업 폴더 구성을 다룬다.

| 항목 | 종류 | 역할 |
|---|---|---|
| `read_model` | fn | 모델 폴더 읽기: 이진 우선, 없으면 텍스트 |
| `read_model_binary`, `read_model_text` | fn | 이진 / 텍스트 모델 읽기(rigs·frames 없는 모델도 지원) |
| `write_model_binary`, `write_model_text` | fn | 모델 쓰기(카메라·rig 전부, 자세 있는 프레임, 등록 영상만, 점 전부; 텍스트 숫자는 %.17g) |
| `ImageOrder` | enum | images 파일 영상 순서 `Registration`(기본, 등록 순서) / `ById` |
| `STEREO_SUBDIRS` | const | `stereo/` 아래 하위 폴더 이름(`depth_maps, normal_maps, consistency_graphs`) |
| `create_stereo_dirs` | fn | 조밀 복원 작업 폴더의 `stereo/` 하위 폴더 생성 |
| `write_stereo_configs` | fn | `stereo/patch-match.cfg`, `stereo/fusion.cfg` 쓰기 |

### `io`

| 항목 | 종류 | 역할 |
|---|---|---|
| `io::PointCloud` | struct | 점군 `{positions, normals, colors}`. `len, is_empty, has_normals, has_colors` |
| `io::PlyLayout` | enum | 쓰기 속성 배치 `XyzRgbNormal`(기본) / `XyzNormalRgb` |
| `io::read_ply` | fn | PLY 읽기(ascii / 이진 LE / 이진 BE, 속성 이름 기반) |
| `io::write_ply` | fn | 이진 LE PLY 쓰기(법선 없으면 생략, 색 없으면 흰색) |
| `io::GpsRecord` | struct | GPS 참조 한 줄 `{name, lat, lon, alt}` |
| `io::read_gps_file` | fn | `이름 위도 경도 고도` 줄 읽기(순서 유지, 첫 줄 = ENU 원점) |
| `io::fmt::format_g`, `io::fmt::g17` | fn | 로캘 무관 `%.{p}g` / `%.17g` 숫자 서식 |

### `linalg`

| 항목 | 종류 | 역할 |
|---|---|---|
| `svd3` | fn | 단측 야코비 3×3 SVD(σ 내림차순, det U = +1, 겹친 특이값에서도 정확). matching·sfm·align 공용 |

### `ransac`

| 항목 | 종류 | 역할 |
|---|---|---|
| `Estimator` | trait | 최소/비최소 해법 + 제곱 잔차(`X, Y, Model`, `min_num_samples, estimate, residuals`) |
| `Sampler` | trait | 표본 추출기(`initialize, max_num_samples, sample`) |
| `RandomSubsetSampler` | struct | 무작위 표본(부분 피셔-예이츠 연속 셔플) |
| `ExhaustiveSampler` | struct | 모든 k-조합을 사전식 순서로 |
| `RansacParams` | struct | `{max_error, min_inlier_ratio, confidence, dyn_trials_factor, min_trials, max_trials, random_seed}` |
| `Support` | struct | 지지도 `{num_inliers, residual_sum}`. `measure, is_better` |
| `RansacReport<M>` | struct | `{success, num_trials, support, inlier_mask, model}` |
| `ransac` | fn | 기본 RANSAC |
| `lo_ransac` | fn | LO-RANSAC(국소 최적화용 비최소 추정기 추가) |
| `ransac_with_sampler` | fn | 표본기를 지정하는 일반형 |
| `compute_num_trials`, `static_max_num_trials` | fn | 필요 반복 수 / 생성 시 정적 상한 |
| `make_rng` | fn | 시드 고정(또는 무작위) Pcg64 생성 |
| `n_choose_k` | fn | 이항계수(포화) |

### `reconstruction`

| 항목 | 종류 | 역할 |
|---|---|---|
| `Reconstruction` (루트) | struct | 희소 재구성. 모든 공개 수정 연산이 불변식(양방향 연결 등)을 유지 |
| `Reconstruction::new`, `cameras/camera/camera_mut`, `rigs/rig/rig_mut`, `frames/frame`, `images/image/image_ids/image_by_name/exists_image`, `points3d/point3d/point3d_ids/exists_point3d` | fn | 생성·조회 |
| `Reconstruction::num_cameras/num_rigs/num_frames/num_images/num_points3d`, `registered_frame_count`, `registered_image_count`, `max_point3d_id` | fn | 개수 |
| `Reconstruction::add_camera`, `add_camera_own_rig`, `add_rig`, `add_frame`, `add_image`, `add_image_own_frame` | fn | 구성(자명 rig·프레임 자동 생성 변형 포함) |
| `Reconstruction::world_to_cam`, `projection_center`, `has_pose`, `set_world_to_cam`, `set_frame_pose` | fn | 자세 조회·설정 |
| `Reconstruction::register_frame/register_image`, `deregister_frame/deregister_image`, `is_frame_registered/is_image_registered`, `registered_frames`, `registered_images` | fn | 등록·해제(해제 시 관측 삭제) |
| `Reconstruction::add_point3d`, `add_point3d_with_id`, `add_observation`, `delete_observation`, `delete_point3d`, `merge_points3d`, `delete_all_points3d`, `set_point3d_xyz/error/color` | fn | 3D 점·관측 편집 |
| `Reconstruction::squared_reprojection_error`, `update_point3d_errors`, `point3d_reprojection_error` | fn | 재투영 오차 계산·갱신 |
| `Reconstruction::filter_observations`, `filter_observations_with_large_reprojection_error`, `filter_points3d_with_small_triangulation_angle` | fn | 관측·점 필터 |
| `Reconstruction::total_observations`, `mean_track_len`, `mean_obs_per_registered_image`, `mean_reproj_error` | fn | 통계 |
| `Reconstruction::transform`, `normalize` | fn | Sim3 적용 / 범위 정규화(적용한 Sim3 반환) |
| `Reconstruction::deregister_images_by_id/by_name`, `remove_unregistered` | fn | 영상 등록 해제(경고 목록 반환), 미등록 요소 삭제 |
| `Reconstruction::extract_colors`, `check_invariants` | fn | 영상 표본기로 점 색 추출, 불변식 검사 |
| `Rig` (루트) | struct | `{rig_id, ref_sensor_id, sensors}`. `new, trivial, num_sensors, is_reference_sensor, has_sensor, rig_to_sensor` |
| `Frame` (루트) | struct | `{frame_id, rig_id, world_to_rig}`. `new, attach_data, data_ids, has_pose, image_ids` |
| `Image` (루트) | struct | `{image_id, name, camera_id, frame_id}`. `new, points2d, point2d, num_points2d, num_points3d` |
| `Point2D` (루트) | struct | `{xy, point3d_id}`. `new, has_point3d` |
| `Point3D` (루트) | struct | `{xyz, color, error, track}`. `new, has_error, track_len` |
| `TrackEntry` (루트) | struct | 트랙 원소 `{image_id, point2d_idx}`. `new` |
| `FilterErrorUpdate` | enum | 필터 후 점 오차 갱신 방식 `None, SumOverOriginalLength, MeanOfRemaining` |
| `NormalizeOptions` | struct | `{fixed_scale, extent, p0, p1, use_images}` |
| `bilinear_rgb` | fn | u8 RGB 영상 양선형 보간(범위 밖 None) |

### `store`

| 항목 | 종류 | 역할 |
|---|---|---|
| `FeatureStore` (루트) | struct | 스레드 안전(RwLock) 메모리 특징 저장소, 큰 자료는 `Arc` 공유. 모든 메서드가 `&self` |
| `FeatureStore::new`, `save`, `load` | fn | 생성, 자체 이진 형식 저장·적재 |
| `FeatureStore::add_camera`, `update_camera`, `camera`, `cameras`, `num_cameras` | fn | 카메라(무효 id 면 최대+1 발급) |
| `FeatureStore::add_image`, `add_image_with_id`, `image`, `image_by_name`, `image_id_by_name`, `images`, `image_ids`, `num_images`, `exists_image` | fn | 영상(이름 고유) |
| `FeatureStore::set_pose_prior`, `pose_prior` | fn | 사전 위치 |
| `FeatureStore::set_keypoints`, `set_descriptors`, `keypoints`, `descriptors`, `exists_keypoints`, `exists_descriptors`, `num_keypoints`, `largest_keypoint_count` | fn | 특징 |
| `FeatureStore::write_matches`, `read_matches`, `exists_matches`, `delete_matches`, `num_matched_pairs` | fn | 원시 매칭(어느 방향으로든 호출 가능) |
| `FeatureStore::put_two_view`, `get_two_view`, `contains_two_view`, `remove_two_view`, `two_view_geometries`, `two_view_geometries_since`, `num_two_view_geometries` | fn | 두 뷰 기하(증분 소비용 로그 위치 포함) |
| `FeatureStore::retain_copy` | fn | 조건을 만족하는 영상만 남긴 사본(점진 처리 되감기용, 큰 자료 공유) |
| `StoreImage` | struct | 저장소 영상 행 `{image_id, name, camera_id}` |
| `PosePrior` | struct | 사전 위치 `{position, coordinate_system, gravity}` (0 = WGS84) |

## 사용 예

두 영상과 3D 점 하나로 재구성을 만들고, 재투영 오차를 계산한 뒤 모델 파일로 저장·적재한다
(`src/lib.rs` 의 doc-test 와 같다).

```rust
use cumulus3d_core::analyzer::ModelStats;
use cumulus3d_core::interop::{read_model, write_model_binary, ImageOrder};
use cumulus3d_core::{Camera, CameraModelKind, Image, Quat, Reconstruction, Rigid3, TrackEntry, Vec3};

fn main() -> cumulus3d_core::Result<()> {
    let mut cam = Camera::from_focal(CameraModelKind::Pinhole, 1000.0, 1920, 1080);
    cam.camera_id = 1;
    let mut rec = Reconstruction::new();
    rec.add_camera_own_rig(cam.clone())?;

    // 세계 점 하나를 두 카메라(기선 1 m)에 투영해 2D 관측을 만든다.
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

## 기능 플래그·하드웨어

없음. 순수 CPU 크레이트이며 기능 플래그가 없다. 의존: nalgebra, rayon(관측 필터 병렬), rand/rand_pcg, thiserror.
