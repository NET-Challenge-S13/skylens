[English](https://github.com/AH100-1/cumulus3d/blob/main/crates/dense/README.md) | 한국어

# cumulus3d-dense

왜곡 보정과 다시점 조밀화: 등록된 희소 모델과 영상에서 깊이맵을 추정·필터·융합해 조밀 점군(PLY)을 만든다.

**파이프라인 단계**: SfM(`cumulus3d-sfm`)·GPS 정렬(`cumulus3d-align`) 뒤의 마지막 단계.
희소 모델 → 왜곡 보정(PINHOLE) → 조밀화 장면 → 이웃 뷰·깊이 범위 → 다중 스케일 PatchMatch 깊이맵 →
필터·후처리 → 융합 → 조밀 점군.

PatchMatch 실행 자체(한 스케일의 적·흑 전파, 뷰 선택, 정제, 판독)는 [`PatchMatchBackend`](https://github.com/AH100-1/cumulus3d/blob/main/crates/dense/src/kernel.rs) 를 구현한
백엔드가 맡고, 이 크레이트는 그 바깥(장면, 이웃 선택, 깊이 범위, 다중 스케일 진행, 상향 표본, 세부 복원 판정,
필터 문턱, 후처리, 융합, 통계)을 맡는다.

## 주요 진입점

| 함수/타입 | 역할 | 입력 → 출력 |
|---|---|---|
| `undistort` | 등록 영상 전부 왜곡 보정(영상 읽기는 호출자 클로저) | `&Reconstruction`, `&UndistortOptions`, `&UndistortCache`, `Fn(&Image) -> Option<Arc<ImageBuffer>>` → `Result<UndistortResult>` |
| `undistort_from_dir` | 영상 폴더에서 읽어 왜곡 보정 | `&Reconstruction`, 영상 폴더, 옵션, 캐시 → `Result<UndistortResult>` |
| `write_undistorted_workspace` | 보정 결과를 작업 폴더(`images/`, `sparse/`, `stereo/`)로 쓰기 | `&UndistortResult`, 출력 폴더, 옵션 → `Result<()>` |
| `DenseScene::from_reconstruction` | 메모리 내 보정 모델·영상으로 장면 구성 | `&Reconstruction`, `&BTreeMap<ImageId, Arc<ImageBuffer>>`, `&SceneOptions` → `Result<DenseScene>` |
| `DenseScene::from_workspace_dir` | 보정 작업 폴더에서 장면 구성 | 폴더, `&SceneOptions` → `Result<DenseScene>` |
| `DensifyOptions::with_profile` | 프로파일(`Fast`/`Quality`) 기본 설정 | `MvsProfile` → `DensifyOptions` |
| `densify` | 조밀화 전체(깊이맵 → 필터 → 융합) | `&DenseScene`, `&DensifyOptions`, `&dyn PatchMatchBackend`, `Option<&DepthMapCache>` → `Result<DenseOutput>` |
| `densify::densify_with` | `densify` + 선택적 점수 융합 | 위 + `Option<&ScoreFusionOptions>` → `Result<DenseOutput>` |
| `compute_depth_maps` | 깊이맵만 계산(융합 없음) | `&DenseScene`, `&DensifyOptions`, 백엔드, 캐시 → `Result<DepthMapSet>` |
| `fuse_depth_maps` | 계산된 깊이맵 융합 | `&DenseScene`, `&DepthMapSet`, `&DensifyOptions`, 스레드 수 → `FusionOutput` |
| `densify::fuse_output` | 보관한 깊이맵으로 융합만 다시 실행 | `&DenseScene`, `&DepthMapSet`, `&DensifyOptions`, `Option<&ScoreFusionOptions>` → `DenseOutput` |
| `fuse` | 깊이맵 배열 융합(저수준) | `&DenseScene`, `&[Option<FusionInput>]`, 겹침 목록, `&FusionParams`, 스레드 수 → `FusionOutput` |
| `DenseOutput::write_ply` | 점군 PLY 쓰기(x y z nx ny nz red green blue) | 경로 → `Result<()>` |
| `cloud_stats` | 점군 품질 통계(간격, GSD, 이상점·중복률) | 장면, 깊이맵, 점군, 가시성, 표본 수, 허용치 → `CloudStats` |
| `PatchMatchBackend` | 백엔드 경계 trait(GPU 구현은 `cumulus3d-cuda::CudaPatchMatch`) | `&KernelInput` → `Box<dyn PatchMatchSession>` |

## 공개 항목

"루트"는 크레이트 루트에서 `cumulus3d_dense::이름` 으로 재노출된 항목이다.

### `undistort` — 왜곡 보정

| 항목 | 종류 | 역할 |
|---|---|---|
| `UndistortOptions` (루트) | struct | 빈 픽셀 비율, 배율 범위, `max_image_size`, 관심 영역, 원천 영상 수, JPEG 품질. `pipeline()`: `max_image_size = 960` |
| `undistorted_camera` (루트) | fn | 왜곡 없는 PINHOLE 카메라 계산(카메라 id 유지) |
| `CameraUndistortion` (루트) | struct | 카메라 하나의 보정 결과(`source`, `pinhole`, 재표본 맵). `new`, `undistort_image` |
| `UndistortCache` (루트) | struct | 카메라 매개변수 해시 → `CameraUndistortion` 캐시(구역 사이 공유). `new`, `get`, `len`, `is_empty` |
| `undistort_reconstruction` (루트) | fn | 희소 모델 변환: 카메라 PINHOLE 로, 2D 관측 새 좌표로 |
| `UndistortResult` (루트) | struct | `reconstruction`, `images`(영상 id → 보정 영상), `failed` |
| `undistort` (루트) | fn | 등록 영상 전부 보정(영상 로더 클로저) |
| `undistort_from_dir` (루트) | fn | 영상 폴더에서 읽어 보정 |
| `write_undistorted_workspace` (루트) | fn | 보정 결과를 조밀 복원 작업 폴더로 쓰기 |

### `image` — 영상 버퍼

| 항목 | 종류 | 역할 |
|---|---|---|
| `ImageBuffer` (루트) | struct | 8비트 1/3채널 영상. `new`, `load`, `save`, `get`, `rgb`, `to_gray`, `resize_area` |
| `GrayImage` (루트) | struct | f32 회색 영상. `new`, `at`, `at_clamped`, `bilinear_clamped`, `resize_area`, `resize_cubic`, `rescale` |
| `Integral` | struct | 적분영상(f64). `new`, `box_mean`, `resample`(면적 평균 재표본) |
| `bilinear_clamped` | fn | 정수 중심 규약 쌍선형 보간(가장자리 고정) |

### `scene` — 조밀화 장면

| 항목 | 종류 | 역할 |
|---|---|---|
| `SceneOptions` (루트) | struct | `max_image_size`(긴 변 상한, 줄이기만) |
| `DenseView` (루트) | struct | 뷰 하나(`image_id`, `name`, 크기, `k`, `r`, `t`, `gray`, `color`). `center`, `to_cam`, `to_world`, `dir_to_world`, `dir_to_cam`, `project_cam`, `ray`, `geometry_hash` |
| `ScenePoint` (루트) | struct | 희소점(`xyz`, 관측 뷰 색인) |
| `DenseScene` (루트) | struct | `views`, `points`. `from_reconstruction`, `from_workspace_dir` |
| `gray_of` | fn | RGB → 회색 8비트 |

### `params` — 설정

| 항목 | 종류 | 역할 |
|---|---|---|
| `DensifyOptions` (루트) | struct | 조밀화 전체 설정. `with_profile`, `schedule`, `fingerprint`(캐시 열쇠용 해시) |
| `MvsProfile` (루트) | enum | `Fast`(기본) / `Quality`. `parse`, `name` |
| `LevelSchedule` (루트) | struct | 스케일별 반복 일정(광도 반복, 세부 복원 초기화, 기하 회차·반복) |
| `PmParams` (루트) | struct | PatchMatch 상수(창, 양방향 σ, 투표 τ0/α/τ1/β/n1/n2, 사전, 기하 λ/δ, 정제 섭동). `sigma_s`, `window_offsets` |
| `FilterParams` (루트) | struct | 깊이맵 필터 허용치(min_ncc, 최소 삼각측량각, 최소 일치 뷰, 재투영 허용, 중앙값 필터) |
| `FusionMode` (루트) | enum | `Traversal`(확장 중앙값) / `Consistency`(일치 평균, 기본). `parse` |
| `FusionResidual` (루트) | enum | 남은 픽셀 처리 `None` / `Release` / `SecondPass`. `parse`, `name` |
| `ResidualParams` (루트) | struct | 2차 융합 허용치 |
| `FusionParams` (루트) | struct | 융합 허용치(방식, 최소 일치 뷰, 역분산 가중, 깊이·법선·재투영 허용치 등) |
| `NeighborParams` (루트) | struct | 이웃 뷰 수(≤ 32), 최소 삼각측량각, 방향 다양성 |

### `kernel` — 백엔드 경계

| 항목 | 종류 | 역할 |
|---|---|---|
| `PatchMatchBackend` (루트) | trait | 백엔드: `name`, `begin(&KernelInput) -> Box<dyn PatchMatchSession>` |
| `PatchMatchSession` (루트) | trait | 호출 하나 동안의 백엔드 상태: `run`, `evaluate`, `filter`, `upsample`·`median_filter`(기본 구현 = 호스트 계산) |
| `KernelInput` (루트) | struct | 커널 입력 전체(뷰, 스케일 수, `PmParams`, `FilterParams`, 시드) |
| `KernelView` (루트) | struct | 뷰 하나(난수 열쇠, 스케일별 영상, 원천 뷰, 상대 기하, 깊이 범위) |
| `LevelImage` (루트) | struct | 한 스케일의 회색 영상과 K |
| `PairGeometry` (루트) | struct | 기준 → 원천 상대 기하(`r`, `t`, `center`) |
| `ViewState` (루트) | struct | 픽셀 상태(깊이, 법선, 비용, 거친 스케일 가설). `new` |
| `DepthSnapshot` (루트) | struct | 기하 실행이 읽는 깊이 스냅숏 |
| `RunParams` (루트) | struct | 실행 하나의 매개변수(스케일, 기하 여부, 초기화, 반복 수, 실행 번호, 사전 사용) |
| `rng_state`, `rng_next` | fn | 커널과 같은 픽셀 난수 정의 |

### `densify` — 진행

| 항목 | 종류 | 역할 |
|---|---|---|
| `densify` (루트) | fn | 조밀화 전체: 깊이맵 → 필터 → 융합 |
| `densify_with` | fn | `densify` + 선택적 점수 융합 |
| `compute_depth_maps` (루트) | fn | 모든 뷰의 최종 깊이맵 계산 |
| `fuse_depth_maps` (루트) | fn | 깊이맵 융합(`opts.fusion.mode`) |
| `fuse_depth_maps_with` | fn | 깊이맵 융합 + 선택적 점수 융합 |
| `fuse_output` | fn | 보관한 깊이맵으로 융합만 다시 실행해 `DenseOutput` 구성 |
| `pair_geometry` (루트) | fn | 기준 → 원천 상대 기하 |
| `DenseOutput` (루트) | struct | `cloud`, `visibility`, `timings`, `cache_hits`, `depth_views`, `depth_maps`, `residual`. `write_ply`, `num_residual`, `write_residual_ply` |
| `DepthMapSet` (루트) | struct | 뷰별 깊이맵, 원천 뷰, 기준선, 캐시 적중, 스케일 수, 시간 |
| `DepthMapResult` (루트) | struct | 깊이맵 한 장(필터 전·후 깊이, 법선, 비용, 캐시 여부) |
| `DenseTimings` (루트) | struct | 단계별 시간. `depth`, `total` |

### `fusion`, `fusion_score` — 융합

| 항목 | 종류 | 역할 |
|---|---|---|
| `fuse` (루트) | fn | `FusionParams::mode` 에 따라 일치 또는 확장 융합 |
| `fuse_consistency` | fn | 일치 융합(기준 픽셀 + 겹침 뷰 일치 점 평균, 결정적) |
| `fuse_traversal` | fn | 이웃의 이웃 확장 + 성분별 중앙값 융합 |
| `FusionInput` (루트) | struct | 뷰 하나의 깊이·법선·비용·기준선. `plain` |
| `FusionOutput` (루트) | struct | 점군, 가시성, 2차 융합 표시, 시간. `num_residual`, `residual_cloud` |
| `fusion_score::ScoreFusionOptions` | struct | 점수 융합 옵션. `from_fusion` |
| `fusion_score::fuse_scored` | fn | 점수 융합(일치 기여 − 자유공간 위반 벌점) |

### `neighbors` — 이웃 뷰·깊이 범위

| 항목 | 종류 | 역할 |
|---|---|---|
| `PairStats` | struct | 뷰 쌍 공유 점 수·삼각측량각. `new`, `select`, `select_diverse`, `select_all` |
| `depth_ranges` | fn | 뷰별 깊이 범위(희소점 깊이 1%/99% × 0.75/1.25) |
| `depth_range_of` | fn | 깊이 목록 하나의 범위 |

### `upsample` — 피라미드·상향 표본

| 항목 | 종류 | 역할 |
|---|---|---|
| `num_levels` | fn | 스케일 수 계산 |
| `build_pyramid` | fn | 회색 피라미드(0 = 최저)와 스케일별 K |
| `joint_bilateral_upsample` | fn | 결합 양방향 상향 표본 |
| `median_plane_filter` | fn | 5×5 중앙값 평면 필터 |
| `downsample_depth` | fn | 깊이맵 2배 축소 |

### `postproc` — 후처리

| 항목 | 종류 | 역할 |
|---|---|---|
| `PostParams` (루트) | struct | 작은 조각 제거·틈 메우기 설정 |
| `remove_speckles` | fn | 작은 조각 제거(지운 픽셀 수) |
| `fill_holes` | fn | 경계 인식 틈 메우기(채운 픽셀 수) |

### `stats` — 품질 통계

| 항목 | 종류 | 역할 |
|---|---|---|
| `cloud_stats` (루트) | fn | 점 간격 중앙값, GSD, 이상점 비율, 중복률, 국소 평면 이탈 |
| `CloudStats` (루트) | struct | 위 통계 값 |

### `cache` — 깊이맵 캐시

| 항목 | 종류 | 역할 |
|---|---|---|
| `DepthMapCache` (루트) | struct | (영상 기하 해시, 설정 해시) → 깊이맵, 스레드 안전·용량 제한. `new`, `with_capacity`, `get`, `insert`, `len`, `is_empty` |
| `CachedDepth` (루트) | struct | 캐시된 뷰 하나(필터 전·후 깊이, 법선, 비용) |

### `math` — 수치 도구

| 항목 | 종류 | 역할 |
|---|---|---|
| `hash_bytes`, `mix64` | fn | 결정적 해시(FNV-1a), splitmix64 섞기 |
| `erf` | fn | 오차 함수 |
| `emission_norm`, `emission`, `visibility_probability` | fn | 비용의 방출 밀도와 가시 확률 |
| `triangulation_prior`, `incident_prior`, `resolution_prior` | fn | 뷰 선택 사전확률 |
| `apply_h` | fn | 3×3 호모그래피로 점 옮기기 |
| `quantile_sorted`, `median_in_place` | fn | 분위수, 중앙값 |
| `plane_transfer` | fn | 이웃 평면을 다른 광선으로 옮긴 깊이 |

### `synthetic` — 시험용 합성 장면

| 항목 | 종류 | 역할 |
|---|---|---|
| `make_scene` | fn | 무늬 바닥 + 상자 장면, 격자 카메라, 참 깊이·법선, 희소점 |
| `SynthConfig` | struct | 영상 크기, 초점, 카메라 격자, 상자, 시드, 희소점 수 |
| `SynthScene` | struct | `scene`, 참 `depth`·`normal`, `config`. `surface_distance` |
| `SynthBox` | struct | 축 정렬 상자 |
| `texture` | fn | 결정적 무늬 함수 |

## 사용 예

합성 장면의 참 깊이맵을 CPU 에서 융합(백엔드 불필요, `src/lib.rs` 의 doc-test 와 같다):

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

실제 조밀화(GPU 백엔드와 실데이터 필요, doc-test 는 `no_run`):

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

`backend` 에는 `cumulus3d_cuda::CudaPatchMatch` 를 넘긴다(`cumulus3d-cuda` 크레이트 참조).

## 기능 플래그·하드웨어

- 기능 플래그 없음. 순수 Rust 이고 어느 기계에서나 빌드된다.
- **조밀화(`densify`, `compute_depth_maps`)는 `PatchMatchBackend` 구현이 필요하다.** 이 크레이트에는 CPU 백엔드가 없고,
  현재 구현은 `cumulus3d-cuda` 의 `CudaPatchMatch`(NVIDIA GPU, CUDA 12.x 드라이버) 하나뿐이다.
- 백엔드 없이 CPU 에서 되는 것: 왜곡 보정, 장면 구성, 이웃 선택·깊이 범위, 상향 표본·중앙값 필터, 후처리,
  융합(`fuse`, `fuse_depth_maps`, `fuse_output`), 통계, 합성 장면. 병렬화는 rayon.

## 처리 흐름

1. **장면**: 왜곡 보정된 모델(핀홀) + 영상 → 뷰(K·자세·회색/컬러). 장면 안 픽셀 좌표는 정수 = 픽셀 중심(주점 −0.5).
2. **이웃 뷰**: 희소점 트랙으로 뷰 쌍의 공유 점 수와 삼각측량각 75 분위를 구해, 각이 1° 이상인 뷰를 공유 수 순으로 최대 10개.
   선택적으로 기준선 방위각 구간별 감쇠로 방향이 고르게 퍼지게 고른다(`NeighborParams::diversity_decay` < 1).
3. **깊이 범위**: 관측 희소점 깊이의 1%/99% 값에 0.75/1.25 배. 무작위 초기화에만 쓴다.
4. **다중 스케일**(축소율 0.5, 최대 3단): 최저 스케일은 무작위 초기화 광도 실행, 위 스케일은 결합 양방향 상향 표본 →
   세부 복원(상향 가설 비용 − 광도 실행 비용 > 0.1 인 픽셀은 광도 실행 결과로 교체) → 기하 일관성 실행 2회
   (모든 기준 뷰가 직전 회차 깊이 스냅숏을 읽는 이중 버퍼라 실행 순서와 무관).
5. **판독·필터**: 5×5 중앙값 평면 필터 뒤, 원천 뷰마다 삼각측량각 ≥ 3°, 입사 cos > 0, 방출 가시 확률 ≥ E(0.9),
   순·역 재투영 ≤ 1 px 를 모두 만족한 뷰가 2개(원천 수가 적으면 그 수) 이상인 픽셀만 남긴다.
6. **후처리**: 상대 깊이 연속 성분 중 작고 비용이 큰 조각 제거, 선택적으로 밝기 경계에서 멈추는 틈 메우기.
7. **융합**: 기본은 일치 융합 — 기준 픽셀의 점을 겹침 뷰에 투영해 깊이(상대 1%)·재투영(2 px)·법선(10°)이 맞는 뷰가
   5개 이상이면 기준과 일치 점들의 역분산 가중 평균(`σ_d = d²·σ_px/(f·b)`, `σ_px = 0.25 + 비용`).
   그 밖에 확장 중앙값 융합(`FusionMode::Traversal`)과 점수 융합(`ScoreFusionOptions`)이 있다.
8. **출력**: PLY(x y z nx ny nz red green blue, 27 바이트/점).

## 프로파일

| | `Fast`(기본) | `Quality` |
|---|---|---|
| 창 | 반경 5, 간격 2 (36 표본) | 반경 5, 간격 1 (121 표본) |
| 최저 스케일 | 광도 6 + 기하 2×2 | 광도 7 + 기하 2×6 |
| 위 스케일 | 세부 복원 3(상향 가설에서 시작) + 기하 2×2 | 세부 복원 6(무작위 시작) + 기하 2×6 |

## 결정성

픽셀 난수는 (시드, 영상 이름 해시, 스케일, 실행, 반복, 반쪽 단계, 픽셀) 로 정해지는 카운터 기반 수열이라 스레드 배치와
무관하다. 기하 실행은 스냅숏 이중 버퍼라 뷰 처리 순서와 무관하다. 일치 융합·점수 융합도 결정적이다(확장 융합은 병렬일 때 아님).

## 시험

- `cargo test -p cumulus3d-dense`: 수치 기준값, 이웃 선택, 상대 기하, 상향 표본·중앙값 필터, 합성 장면 참 깊이맵 융합,
  2차 융합, 후처리, 왜곡 보정.
- GPU 정확도는 `cumulus3d-cuda` 의 `tests/patchmatch_gpu.rs`(합성 장면 참값과 비교).

## 참고문헌

- M. Bleyer, C. Rhemann, C. Rother. PatchMatch Stereo – Stereo Matching with Slanted Support Windows. BMVC 2011.
- S. Galliani, K. Lasinger, K. Schindler. Massively Parallel Multiview Stereopsis by Surface Normal Diffusion. ICCV 2015.
- J. L. Schönberger, E. Zheng, J.-M. Frahm, M. Pollefeys. Pixelwise View Selection for Unstructured Multi-View Stereo. ECCV 2016.
- Q. Xu, W. Tao. Multi-Scale Geometric Consistency Guided Multi-View Stereo. CVPR 2019.
- J. Kopf, M. F. Cohen, D. Lischinski, M. Uyttendaele. Joint Bilateral Upsampling. SIGGRAPH 2007.

## 라이선스

Apache-2.0(`LICENSE`, `NOTICE`).
