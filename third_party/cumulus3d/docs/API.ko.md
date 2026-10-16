[English](API.md) | 한국어

# cumulus3d 공개 API 지도

워크스페이스 전체를 한눈에 보는 문서다. 각 크레이트의 공개 항목 전체 목록은 크레이트별 README 에 있다.

## 크레이트 의존 관계

```
                         cumulus3d-core
        ┌──────────┬──────────┼──────────┬──────────┐
        ▼          ▼          ▼          ▼          ▼
    features   matching       ba       align      dense
        │          │ └────┐   │                     │
        │          │      ▼   ▼                     │
        │          │      sfm                       │
        │          ▼                                │
        └──────▶ cuda ◀─────────────────────────────┘
                   │
                   ▼
                  cli  (core·features·matching·ba·sfm·align·dense·cuda 전부 사용)
```

| 크레이트 | 직접 의존(워크스페이스 안) |
|---|---|
| `cumulus3d-core` | 없음 |
| `cumulus3d-features` | core |
| `cumulus3d-matching` | core |
| `cumulus3d-ba` | core |
| `cumulus3d-align` | core |
| `cumulus3d-dense` | core |
| `cumulus3d-sfm` | core, ba, matching |
| `cumulus3d-cuda` | core, features, matching, dense |
| `cumulus3d-cli` | 위 여덟 개 전부 |

파이프라인 순서: 영상 → **features**(SIFT) → **matching**(기술자 매칭·두 뷰 기하) → **sfm**(전역 SfM·등록·삼각측량, 내부에서 **ba**)
→ **align**(GPS ENU 정렬·재기준) → **dense**(조밀 점군). **cuda** 는 features·matching·dense 의 GPU 백엔드, **cli** 는 이 모두를 점진 파이프라인으로 조립한다.

## 무엇을 하려면 어떤 함수를 쓰나

| 하려면 | 이 함수 / 타입 | 크레이트 |
|---|---|---|
| **영상 폴더로 점진 재구성(라이브러리)** | `Pipeline::new(Session::new(SessionConfig::new(images)))` → 위치마다 `.push(Input::Frames(stream::frame_set(&layout, p)))` → `.finish()` (`Layout::discover(src, stride)`) | cli |
| **선언형 재구성: 계획 기록 → 검사 → 실행** | `cumulus3d_cli::declare::Recon::declare().input(src).preset("aerial-formation").zones(12, 2).dense(Dense::profile("fast")).sinks(Sinks::default_files(out)).build()?.run()` | cli |
| 계획 값 / TOML 파일(`cumulus3d run plan.toml`) | `cumulus3d_cli::declare::Plan`(`to_toml`, `from_toml`, `load`, `from_stream`), `Recon::from_plan` | cli |
| 명령줄 `cumulus3d stream` 과 같은 전체 실행 | `cumulus3d_cli::stream::run_stream(StreamConfig::new(src, out))` | cli |
| 훅 없이 값으로 상태 전이 | `cumulus3d_cli::session::{step, poll, command, finish}` | cli |
| 표준 출력 파일(timeline, PLY, 스냅샷) 붙이기 | `cumulus3d_cli::sinks::attach(pipeline, out, &SinkOptions)` | cli |
| 구역 초벌·정밀 점군 받기 | `Pipeline::on_zone_preview` / `on_zone_refined` / `on(EventKind, …)` / `subscribe()` | cli |
| 단계별 하위 명령을 코드에서 실행 | `cumulus3d_cli::interop::run(InteropCmd::…)` | cli |
| 모델 하나 조밀화(왜곡 보정 포함) | `cumulus3d_cli::densewrap::dense_model` | cli |
| 모델 파일 읽기·쓰기 | `cumulus3d_core::interop::read_model` / `write_model_binary` / `write_model_text` | core |
| 특징·매칭 저장소 | `cumulus3d_core::FeatureStore` (`save` / `load`) | core |
| 대응 그래프 만들기·증분 갱신 | `cumulus3d_core::MatchGraph::from_store` / `update_from_store` | core |
| 범용 견고 추정 | `cumulus3d_core::ransac::lo_ransac` (`Estimator` 구현) | core |
| PLY·GPS 파일 입출력 | `cumulus3d_core::io::{read_ply, write_ply, read_gps_file}` | core |
| 영상 묶음 특징 추출 | `cumulus3d_features::FeatureExtractor::extract_inputs` / `extract_files` | features |
| 영상 한 장 SIFT | `cumulus3d_features::CpuSift` + `SiftEngine::extract` | features |
| EXIF·카메라 초기값 | `cumulus3d_features::ExifInfo::read`, `init_camera` | features |
| 저장소의 짝 매칭·기하 검증 | `cumulus3d_matching::match_pairs` (파일 목록: `match_pair_list_file`) | matching |
| 두 영상 기하(E/F/H) | `cumulus3d_matching::estimate_two_view` | matching |
| 기술자만 매칭 | `cumulus3d_matching::CpuMatcher` + `MatcherBackend::match_descriptors` | matching |
| 두 뷰 상대 자세 | `cumulus3d_matching::recover_two_view_pose` / `pose_from_essential` | matching |
| 첫 모델 만들기(전역 SfM) | `cumulus3d_sfm::global_mapper` (`GlobalSfmOptions`) | sfm |
| 기존 모델에 새 영상 등록 | `cumulus3d_sfm::register_images` (한 장: `registration::register_image`) | sfm |
| 새 영상만 삼각측량 | `cumulus3d_sfm::triangulate_points` + `TriangulationScope::Images(..)` | sfm |
| 2D–3D 로 자세 추정 | `cumulus3d_sfm::absolute_pose::solve_abs_pose` | sfm |
| 전역 회전 평균 | `cumulus3d_sfm::rotation_averaging::solve_rotation_averaging` | sfm |
| 번들 조정 | `cumulus3d_ba::bundle_adjust` + `BaConfig` | ba |
| 자세 하나 정제 | `cumulus3d_ba::refine_abs_pose` | ba |
| 모델을 GPS ENU 에 정렬 | `cumulus3d_align::align_to_gps` / `align_to_gps_file` (Sim3 만: `estimate_gps_alignment`) | align |
| 두 재구성(초벌↔정밀) 정렬 | `cumulus3d_align::align_reconstructions` | align |
| 구역 좌표계 연쇄 재기준 | `cumulus3d_align::AnchorChain::push_model` | align |
| 스냅샷 점군 합성 | `cumulus3d_align::compose_snapshot` | align |
| GPS ↔ ENU 변환 | `cumulus3d_align::EnuFrame` | align |
| 왜곡 보정 | `cumulus3d_dense::undistort` / `undistort_from_dir` | dense |
| 조밀화 장면 만들기 | `cumulus3d_dense::DenseScene::from_reconstruction` / `from_workspace_dir` | dense |
| 조밀화 전체(깊이맵 → 필터 → 융합) | `cumulus3d_dense::densify` (백엔드: `cumulus3d_cuda::CudaPatchMatch`) | dense |
| 깊이맵만 계산 후 나중에 융합 | `cumulus3d_dense::compute_depth_maps` → `fuse_depth_maps` | dense |
| 점군 품질 통계 / PLY 저장 | `cumulus3d_dense::cloud_stats`, `DenseOutput::write_ply` | dense |
| GPU 사용 가능 확인 | `cumulus3d_cuda::is_available` | cuda |
| GPU 조밀화 / 매칭 / SIFT | `CudaPatchMatch::try_default`, `CudaMatcher`, `CudaSift` (각각 dense·matching·features 백엔드 자리에 넘김) | cuda |

GPU 는 CUDA 12.x 드라이버가 필요하다. `cumulus3d-cuda` 는 동적 로딩이라 CUDA 없는 기계에서도 빌드된다. 조밀화에는 CPU 백엔드가 없다.

## 크레이트별 문서

| 크레이트 | 문서 |
|---|---|
| `cumulus3d-core` | [crates/core/README.ko.md](../crates/core/README.ko.md) |
| `cumulus3d-features` | [crates/features/README.ko.md](../crates/features/README.ko.md) |
| `cumulus3d-matching` | [crates/matching/README.ko.md](../crates/matching/README.ko.md) |
| `cumulus3d-ba` | [crates/ba/README.ko.md](../crates/ba/README.ko.md) |
| `cumulus3d-sfm` | [crates/sfm/README.ko.md](../crates/sfm/README.ko.md) |
| `cumulus3d-align` | [crates/align/README.ko.md](../crates/align/README.ko.md) |
| `cumulus3d-dense` | [crates/dense/README.ko.md](../crates/dense/README.ko.md) |
| `cumulus3d-cuda` | [crates/cuda/README.ko.md](../crates/cuda/README.ko.md) |
| `cumulus3d-cli` | [crates/cli/README.ko.md](../crates/cli/README.ko.md) |
