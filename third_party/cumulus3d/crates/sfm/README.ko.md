[English](https://github.com/AH100-1/cumulus3d/blob/main/crates/sfm/README.md) | 한국어

# cumulus3d-sfm

특징·매칭 결과로 카메라 자세와 희소 3D 점을 만드는 SfM 크레이트(전역 SfM, 새 영상 등록, 삼각측량).

**파이프라인 단계**: `cumulus3d-matching` 이 채운 `FeatureStore`(특징·두 뷰 기하)와 그로부터 만든 `MatchGraph` 를 받아
희소 재구성(`Reconstruction`)을 만든다. 첫 모델은 전역 SfM(상대 자세 → 회전 평균 → 트랙 → 위치 추정 → BA →
재삼각측량)으로, 이후 위치가 도착할 때마다 새 영상 등록과 삼각측량으로 모델을 키운다. 결과는 `cumulus3d-align`
(GPS 정렬)과 `cumulus3d-dense`(조밀화)로 넘어간다. 번들 조정은 `cumulus3d-ba` 를 쓴다.

## 주요 진입점

| 함수/타입 | 역할 | 입력 → 출력 |
|---|---|---|
| `global_mapper` | 전역 SfM 전 과정(최대 연결 성분 하나의 모델) | `&FeatureStore, &MatchGraph, &GlobalSfmOptions` → `Result<GlobalMapperOutput>` |
| `GlobalSfmOptions` | 전역 SfM 옵션. `default()`: BA 3회 + 재삼각측량, `script()`: BA 0회·재삼각측량 생략·트랙 상한 100000 | — |
| `GlobalMapperOutput` | 결과 `{ reconstruction, view_graph, summary, failure }`. 중간 실패 시 그때까지의 모델과 `failure` 사유 | — |
| `register_images` | 누락 영상 추가 → 자세 없는 영상마다 한 번 등록 시도 → 실패 영상 삭제 | `&mut Reconstruction, &FeatureStore, &MatchGraph, &RegistrationOptions` → `Result<RegistrationReport>` |
| `registration::register_image` | 영상 하나 등록(P3P/EPnP LO-RANSAC + 자세 정제) | `&mut Reconstruction, &MatchGraph, ImageId, &RegistrationOptions` → `Result<RegisterOutcome>` |
| `RegistrationOptions` / `RegistrationOrder` | 등록 옵션 / 순서(`ImageId` 기본, `VisibilityScore`) | — |
| `RegistrationReport` | 시도별 결과(`attempts`), `registered()` 로 성공 영상 id | — |
| `triangulate_points` | 기존 점 연장 → 새 점 생성 → 트랙 완성·병합 → 재삼각측량 → 정제 → 필터 | `&mut Reconstruction, &MatchGraph, &PointTriangulatorOptions` → `Result<TriangulationReport>` |
| `PointTriangulatorOptions` | 삼각측량 옵션(`scope`, `refiner`, 필터 임계 등) | — |
| `TriangulationScope` | `AllRegistered`(기본) 또는 `Images(새 영상)`: 새 영상과 바뀐 점만 처리 | — |
| `PointRefiner` | 점 정제 방식: `BundleAdjuster`(기본) / `PerPoint`(점별 3변수 LM) / `None` | — |
| `absolute_pose::solve_abs_pose` | 2D–3D 대응으로 절대 자세 추정(P3P + EPnP LO-RANSAC) | `&Camera, &[Vec2], &[Vec3], &AbsolutePoseOptions` → `Option<AbsolutePoseResult>` |
| `positioning::global_positioning` | 회전 고정, 카메라 중심·점 위치 추정 | `&mut Reconstruction, &PositionSolverOptions` → `Result<PositioningSummary>` |
| `rotation_averaging::solve_rotation_averaging` | 전역 회전 추정(최대 신장 트리 → L1 ADMM → IRLS) | `&BTreeSet<ImageId>, &ViewGraph, Option<&BTreeMap<ImageId, Mat3>>, &RotationAveragingOptions` → `Result<(BTreeMap<ImageId, Mat3>, RotationAveragingSummary)>` |

크레이트 루트 재노출: `global_mapper`, `GlobalSfmOptions`, `GlobalMapperOutput`, `register_images`, `RegistrationOptions`,
`RegistrationOrder`, `RegistrationReport`, `triangulate_points`, `PointRefiner`, `PointTriangulatorOptions`, `TriangulationScope`.
나머지는 모듈 경로(`cumulus3d_sfm::<모듈>::<항목>`)로 쓴다.

## 공개 항목

### `global_mapper` — 전역 SfM

| 항목 | 종류 | 역할 |
|---|---|---|
| `global_mapper` | fn | 전역 매퍼 실행(루트 재노출). 색 추출은 호출자가 `Reconstruction::extract_colors` 로 한다 |
| `GlobalSfmOptions` | struct | 단계 생략 플래그, 트랙·회전 평균·위치 추정 옵션, 필터 임계, BA·재삼각측량 설정(루트 재노출). `script()` 사전 설정 |
| `GlobalMapperSummary` | struct | 단계별 통계(간선 수, 트랙·위치 추정 통계, 필터 삭제 수, 등록 영상·점 수) |
| `GlobalMapperOutput` | struct | `reconstruction`, `view_graph`, `summary`, `failure: Option<String>`(루트 재노출) |
| `init_reconstruction` | fn | 저장소의 카메라·영상(대응 그래프에 있는 것)으로 자세 없는 재구성 생성 |
| `build_view_graph` | fn | 짝별 상대 자세로 뷰 그래프 구성(짝 단위 병렬, 필요하면 상대 자세 재추정) |
| `rotation_averaging_round` | fn | 회전 평균 한 회차를 재구성에 반영(`posed_only` 면 자세 있는 프레임만) |
| `filter_angular_error` | fn | 각도 오차 필터. 삭제 관측 수 반환 |
| `filter_angular_error_prior_focal` | fn | 초점 사전값이 있는 카메라의 각도 오차 초과 관측 삭제 |
| `filter_normalized_reproj_error` | fn | 정규 좌표 재투영 오차 필터 |
| `post_positioning_filters` | fn | 위치 추정 직후 필터·정규화 |
| `bundle_adjustment_stage` | fn | 반복 BA 단계(회전 고정 BA → 전체 BA → 필터) |
| `retriangulation_stage` | fn | 재삼각측량 + 점 정제 반복 |

### `registration` — 새 영상 등록

| 항목 | 종류 | 역할 |
|---|---|---|
| `register_images` | fn | 여러 영상 등록(루트 재노출) |
| `register_image` | fn | 영상 하나 등록(초점 고정 경로) |
| `add_missing_images` | fn | 대응 그래프에 있고 모델에 없는 영상을 자세 없이 추가 |
| `collect_2d3d_correspondences` | fn | 영상의 2D–3D 대응 `(2D 점, 픽셀, 3D 점 id, 좌표)` 수집 |
| `num_visible_points3d` | fn | 대응 이웃 중 3D 점을 가진 2D 점 수 |
| `visibility_score` | fn | 가시성 피라미드 점수(6단계) |
| `RegistrationOptions` | struct | 최소 인라이어, RANSAC, 비정상 카메라 판정, 자세 정제, 순서, 실패 영상 삭제 여부(루트 재노출) |
| `RegistrationOrder` | enum | `ImageId`(영상 id 오름차순 = 도착 순) / `VisibilityScore`(루트 재노출) |
| `RegisterOutcome` | enum | `Registered{..}`, `TooFewVisiblePoints`, `TooFewCorrespondences`, `RansacFailed`, `TooFewInliers`, `RefinementFailed`, `AlreadyRegistered`. `is_registered()` |
| `RegistrationReport` | struct | `attempts`, `num_added_images`, `registered()`(루트 재노출) |

### `triangulator` — 증분 삼각측량

| 항목 | 종류 | 역할 |
|---|---|---|
| `triangulate_points` | fn | 삼각측량 전 과정(루트 재노출) |
| `refine_points` | fn | 자세·내부 고정 점 정제(`ids` 가 `None` 이면 모든 점) |
| `TrackTriangulator` | struct | 증분 삼각측량기(등록 영상 자세·광선 캐시). 메서드 `new`, `triangulate_image`, `complete_track(s)`, `merge_track(s)`, `retriangulate`; 필드 `opts`, `touched`, `report` |
| `TrackTriangulatorOptions` | struct | 생성·연장 각도 허용, 병합·완성 오차, 재삼각측량 조건, 최소 각, 비정상 카메라 판정 |
| `PointTriangulatorOptions` | struct | `tri`, `clear_points`, 끝 필터 임계, 정제 반복, `refiner`, `scope`(루트 재노출) |
| `PointRefiner` | enum | `BundleAdjuster` / `PerPoint` / `None`(루트 재노출) |
| `TriangulationScope` | enum | `AllRegistered` / `Images(Vec<ImageId>)`(루트 재노출) |
| `TriangulationReport` | struct | 생성·연장·완성·병합·재삼각측량·필터·정제 횟수 |

### `triangulation` — 삼각측량 기본 연산

| 항목 | 종류 | 역할 |
|---|---|---|
| `triangulate_dlt` | fn | 두 뷰 DLT(정규 평면 좌표) |
| `triangulate_multi_view` | fn | 다중 뷰 삼각측량(카메라 좌표 단위 광선) |
| `triangulate_midpoint` | fn | 두 광선 중점법(카메라1 좌표, 카메라 뒤면 `None`) |
| `has_positive_depth` | fn | 깊이 양수 검사 |
| `angular_error` | fn | 관측 광선과 점 방향 사이 각도 오차(라디안) |
| `estimate_triangulation` | fn | RANSAC 다중 뷰 삼각측량 → `(점, 인라이어 마스크)` |
| `TriObservation` | struct | RANSAC 입력 관측(`proj`, `center`, `ray`). `new(pose, ray)` |
| `TriangulationRansacParams` | struct | 각도 오차·최소 각·RANSAC 설정, 전수 탐색 관측 수 상한 |

### `absolute_pose` — 절대 자세

| 항목 | 종류 | 역할 |
|---|---|---|
| `solve_abs_pose` | fn | P3P + EPnP LO-RANSAC(초점 고정). 인라이어 < 3 이면 `None` |
| `p3p` | fn | 3점 최소해(최대 4해) |
| `epnp` | fn | EPnP(4점 이상) |
| `Corr2D` | struct | 2D–3D 대응의 픽셀 좌표·역투영 광선 |
| `AbsolutePoseOptions` | struct | RANSAC 옵션(최대 오차 12px, 시드 `Some(0)` 기본) |
| `AbsolutePoseResult` | struct | `world_to_cam`, `inlier_mask`, `num_inliers`, `num_trials` |

### `rotation_averaging` — 뷰 그래프와 회전 평균

| 항목 | 종류 | 역할 |
|---|---|---|
| `ViewGraph` | struct | 뷰 그래프(간선 (id1, id2) 오름차순). `new`, `num_valid_edges`, `valid_nodes`, `largest_connected_component`, `invalidate_outside`, `filter_by_relative_rotation` |
| `ViewGraphEdge` | struct | 간선: 두 영상 id, `cam1_to_cam2`, `num_matches`, `valid` |
| `solve_rotation_averaging` | fn | 전역 회전(world_to_cam) 추정 |
| `mst_initialization` | fn | 최대 신장 트리 초기화(루트 = 최소 id) |
| `admm_l1` | fn | ‖Ax − b‖₁ 최소화(ADMM, 분해 재사용) |
| `l1_regression` | fn | 일반 L1 회귀(검증용) |
| `RotationAveragingOptions` | struct | MST 초기화, L1·IRLS 반복·수렴, IRLS 가중, 정칙화, ADMM 옵션 |
| `AdmmOptions` | struct | ADMM 계수·허용오차·내부 반복 |
| `IrlsWeight` | enum | `GemanMcClure` / `HalfNorm` |
| `RotationAveragingSummary` | struct | L1·IRLS 반복 횟수 |

### `tracks` — 트랙 구성

| 항목 | 종류 | 역할 |
|---|---|---|
| `establish_tracks` | fn | 유효 간선으로 트랙을 만들어 재구성에 3D 점으로 추가 |
| `filter_candidate_tracks` | fn | 영상 내 일관성·최소 뷰 검사 |
| `select_tracks` | fn | (길이, id) 내림차순 선별 |
| `TrackOptions` | struct | 일관성 거리, 영상별 필요 트랙 수, 최소 뷰 수, 최대 트랙 수 |
| `TrackSummary` | struct | 후보·불일치·뷰 부족·통과·선별 트랙 수 |

### `positioning` — 전역 위치 추정

| 항목 | 종류 | 역할 |
|---|---|---|
| `global_positioning` | fn | 점–카메라 방향 제약(회전 고정, Huber 손실, 자체 LM)으로 중심·점 추정 |
| `PositionSolverOptions` | struct | 변수 선택, 손실 척도, LM 반복·허용오차, 시드, 비보정 카메라 가중 |
| `PositioningSummary` | struct | 프레임·점·관측 수, 반복 수, 비용, 수렴 여부 |

### `math` — 수치 보조

| 항목 | 종류 | 역할 |
|---|---|---|
| `umeyama` | fn | dst ≈ s·R·src + t 상사 정렬(평가·테스트용) |
| `gaussian` | fn | 표준 정규 난수(합성 자료·테스트용) |
| `CsrMatrix` | struct | 행 압축 희소 행렬. `new`, `push_row`, `from_dense`, `mul_vec`, `tr_mul_vec`, `weighted_normal` |

### `synthetic` — 합성 장면(문서 숨김, 시험용)

`#[doc(hidden)]` 이지만 공개돼 있어 예제·시험에서 쓸 수 있다. 안정 API 로 보장하지 않는다.

| 항목 | 종류 | 역할 |
|---|---|---|
| `generate` | fn | 드론 3대 직선 비행 합성 장면 생성(영상 id 는 위치 → 카메라 순, 1부터) |
| `opencv_camera` | fn | 카메라 3대의 OPENCV 내부값 |
| `SceneConfig` | struct | 위치 수, 간격, 고도, 점 수, 잡음, 오대응 비율, 시드 |
| `Scene` | struct | `store`, 참 자세 `truth`, `points`, 위치별 영상 id 등. `names_up_to(n)` |

## 사용 예

합성 장면의 앞 5개 위치로 첫 모델을 만들고, 6번째 위치를 등록한 뒤 새 영상만 삼각측량한다
(같은 코드가 `lib.rs` 의 doc-test 로 실행된다).

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

// 1) 전역 SfM 으로 첫 모델.
let out = global_mapper(&scene.store, &graph_up_to(5), &GlobalSfmOptions::script())?;
assert!(out.failure.is_none());
let mut rec = out.reconstruction;

// 2) 새 위치 도착: 등록 → 새 영상만 삼각측량.
let graph = graph_up_to(6);
let report = register_images(&mut rec, &scene.store, &graph, &RegistrationOptions::default())?;
let new_images = report.registered();
let opts = PointTriangulatorOptions { scope: TriangulationScope::Images(new_images), ..Default::default() };
let tri = triangulate_points(&mut rec, &graph, &opts)?;
println!("등록 {} 장, 새 점 {}", rec.registered_image_count(), tri.num_created);
```

실제 영상에서는 `FeatureStore` 를 `cumulus3d-features`(추출)와 `cumulus3d-matching`(`match_pairs`)으로 채우고
`MatchGraph::from_store` 로 대응 그래프를 만든다.

## 결정성

- 모든 순회는 id 오름차순: 회전 평균 루트·게이지 = 최소 활성 영상 id, 트랙 id = 정렬된 성분 순, 등록 순서 = 영상 id 오름차순.
- RANSAC 시드 기본 `Some(0)`. 삼각측량 RANSAC 은 조합 표본(난수 없음).
- 영상 삼각측량은 2D 점별 계획을 병렬로 계산하고, 순차 적용 때 참조 상태가 바뀐 점만 다시 계획한다 → 순차 실행과 같은 결과.
- 위치 추정의 병렬 합산은 고정 덩어리 단위라 스레드 수와 무관하게 비트 단위로 같다.

## 기능 플래그·하드웨어

- 기능 플래그 없음. CPU 전용(rayon 병렬), GPU 불필요.
- 의존: `cumulus3d-core`, `cumulus3d-ba`, `cumulus3d-matching`, `nalgebra`, `rayon`, `rand`, `rand_pcg`, `thiserror`.

## 시험

`cargo test --release -p cumulus3d-sfm`: 최소해법 정확도(P3P·EPnP), LO-RANSAC 견고성, 회전 평균·위치 추정 정확도,
트랙·필터·등록 경계값, 드론 3대 합성 장면(48장)의 전역 매퍼·순차 등록·삼각측량 전 과정.
