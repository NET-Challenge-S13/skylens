[English](CHANGELOG.md) | 한국어

# Changelog

0.3.0 까지는 skyrecon 이라는 이름으로 개발됨.

이 프로젝트는 [유의적 버전(SemVer)](https://semver.org/lang/ko/)을 따른다. 0.x 동안에는 부 버전(0.1 → 0.2)에서 호환이 깨질 수 있다.
각 버전의 날짜는 실제 작업·커밋 날짜다.

## [0.4.0] — 계획(작업 중)
### 변경
- CI: 매 push 마다 CUDA 컨테이너에서 NVRTC 로 커널 컴파일 검사(GPU 불필요, `cumulus3d_cuda::check_kernels`, 예제 `check_kernels`),
  `rustfmt` 검사 필수화(`rustfmt.toml`: 폭 140), 내용 정책 검사, 러너 고정(`ubuntu-24.04`, `macos-15`), `actions/checkout@v5`,
  시험 빌드 단축. `v*` 태그 릴리스 워크플로: 전체 검사, 태그·버전 일치, `cumulus3d` 실행 파일(Linux x86_64, macOS arm64)을
  GitHub Release 에 첨부, semver 검사와 `cargo publish --workspace`(`CARGO_REGISTRY_TOKEN` 이 있을 때만). Cargo·Actions 용 Dependabot.
- 라이선스: Apache-2.0 단독(이전 MIT OR Apache-2.0). 저작권자 ParkSangWoo. `LICENSE`, `NOTICE`(루트와 모든 크레이트),
  Cargo `authors`, 모든 소스 파일 머리 주석(`SPDX-License-Identifier: Apache-2.0`, 다른 프로젝트를 따른 파일은 `THIRD_PARTY_NOTICES.md` 안내).
  CI 가 모든 소스 파일의 머리 주석을 검사.

### 추가
- 실시간 합성 층 `compose`(지금의 입력 상태를 선언하면 유지되는 노드가 조정): `CompositionHost`,
  `ReconNode::declare()…build(&host)`, `compose(FrameState)`, `reconstruct(&host, id, state)`, `close()`. 프레임 식별자
  `FrameKey { source_id, sequence, revision }`. 재선언은 아무것도 하지 않고, `Pending`·불완전 묶음은 기다리며, 순번 건너뜀을 알린다.
  늦은 도착·정정용 `ReconcilePolicy::{AppendOnly, ReplayWindow, InvalidateAffectedZones, Strict}`(`ResetFrom`/`InvalidateZone` 기반).
  메모리 입력(`Rgb8`, `Encoded`)은 원자적으로 저장. `ReconNode::spawn` 생산자 큐와 `Backpressure::{Block, KeepLatest, Unbounded}`,
  `try_send`, 큐 통계. 설명: `docs/COMPOSE.ko.md`.
- 이벤트 기반 파이프라인 앞의 선언형 빌더 층 `declare`. 빌더 메서드는 계획만 기록하고, `.build()`(검사)와
  `.run()`(위치 반복·배경 정밀 작업 대기·종료를 한 번에) 전에는 아무것도 실행하지 않는다.
  - `Recon::declare() -> ReconBuilder`: `.input`, `.images`, `.cameras`, `.preset("aerial-formation")`, `.stride`, `.positions`,
    `.pairing`, `.zones(span, overlap)`, `.gps`, `.no_gps`, `.align`, `.fixed_enu_origin`, `.features`, `.match_backend`, `.gpu`,
    `.incremental_triangulation`, `.dense(Dense::profile("fast").fusion_min_views(3)…)`, `.no_dense`, `.sinks(Sinks::default_files(out))`,
    `.seed`, `.threads`, 훅 `.on(EventKind, 클로저)`, `.on_any`, `.on_zone_preview`, `.on_zone_refined`, `.on_snapshot`,
    `.on_position_done`, `.on_message`, `.policy`(훅은 계획이 아니라 빌더에 보관).
- 프레임 훅: `Pipeline`·`ReconBuilder` 의 `on_frame(|frame: &Arc<Frame>| …)` 이 디코딩된 입력 프레임(RGB8, 크기, 카메라, 경로,
  GPS 기록)을 위치 처리 **전에** 카메라 한 장씩 전달한다. 새 이벤트 `Event::FrameDecoded`, `Reducer::prelude`(`step` 전에 내보내는
  이벤트), `SessionConfig::decode_frames`(기본 꺼짐; 빌더는 프레임 훅을 등록하면 켠다).
  - `ReconBuilder::build() -> Result<Recon, PlanError>`: 입력·카메라 폴더, 선택된 위치 수, ENU 정렬 시 GPS 파일,
    백엔드 이름과 CUDA 장치(조밀화 포함), 출력 폴더 쓰기 가능, 옵션 값 범위를 검사. 문제를 모두 모아 한 번에 보고.
  - `Recon::run() -> Result<Summary, String>`, `Recon::plan()`.
  - 계획 값 `Plan`(Clone/PartialEq/Debug, serde): 입력(영상 폴더, 카메라 폴더, stride, 위치 수), 짝 전략(`formation`),
    구역, GPS 파일, 정렬(ENU), 조밀화(백엔드, 프로파일, 융합·필터 덮어쓰기), 출력 훅, 시드, 스레드.
    `Plan::to_toml`/`from_toml`/`load`/`from_stream`.
- 명령 `cumulus3d run <plan.toml> [--check]`, `cumulus3d plan --print-default`, `cumulus3d plan --check <plan.toml>`.
  예제 계획 `examples/plans/aerial-formation.toml`.
- `sinks::SinkSet`·`DefaultSinks::with_set`(켤 기본 훅 선택), `stream::Layout::discover_in`, `Layout::truncate`,
  `stream::frame_set_of`(임의 영상 폴더·카메라 목록).
- 의존성 `serde`, `toml`(모두 MIT OR Apache-2.0).
- 시험 `tests/declare_synthetic.rs`(실행 전 부수효과 없음, 같은 계획 = 같은 결과, `run` 과 `stream` 출력 동일).
### 변경
- `cumulus3d stream`(`stream::run_stream`)이 선언형 층(`Plan::from_stream` → `Recon`)으로 실행된다. 출력은 같다.
  계획 문제는 실행 전에 보고된다. 특히 CUDA 장치 없이 조밀화를 켜면 첫 구역이 아니라 시작 때 분명한 메시지로 실패한다.

## [0.3.1] — 2026-10-09
### 변경
- 프로젝트 이름을 cumulus3d 로 변경. 크레이트 `cumulus3d-core`/`-features`/`-matching`/`-ba`/`-sfm`/`-align`/`-dense`/`-cuda`/`-cli`,
  실행 파일 `cumulus3d`, 저장소 https://github.com/AH100-1/cumulus3d, 환경 변수 접두어 `CUMULUS3D_`.
- 특징 저장소 파일 표지를 `C3DFS` 로 변경. 옛 표지로 저장된 파일도 그대로 읽는다. 동작은 바뀌지 않았다.

## [0.3.0] — 2026-10-09
### 추가
- 이벤트 계약 `events`: `Event`/`EventKind`/`Meta`(세션 버전·발생 시각)/`ZoneRange`/`Command`.
  단계별 이벤트 `Started`, `FrameIngested`, `FeaturesExtracted`, `PairsMatched`, `ModelInitialized`, `FrameRegistered`,
  `PositionDone`, `ZoneArrived`, `ZonePreview`, `RefineStarted`, `ZoneAdjusted`, `ZoneRefinedPose`, `ZoneRefined`,
  `BaseAdopted`, `Reanchored`, `Snapshot`, `AllPositionsDone`, `AllRefinedDone`, `Finished`, `Reset`, `ZoneInvalidated`,
  `Log`, `Warning`, `Error`. 구역 이벤트는 모델·점군(`Arc`)을 함께 싣는다. `Event::timeline_text` 가 timeline.txt 문구를 준다.
- 상태 값 세션 `session`: `Session` + `step(&Session, FrameSet) -> (Session, Vec<Event>)`, `poll`, `command`, `finish`.
  위치 하나의 프레임 묶음(`FrameSet`)마다 특징 → 매칭 → 등록·삼각측량 → 구역 초벌/정밀(배경)을 이벤트로 내보내며 파일은 쓰지 않는다.
  `SessionConfig`(구역·겹침, 전체 위치 수, GPS, 조밀화 설정, 되감기 지점 수), `Command::ResetFrom`·`InvalidateZone`.
- 람다 훅 파이프라인 `pipeline::Pipeline`: `.on(EventKind, 훅)`, `.on_any`, `.on_zone_preview`/`.on_zone_refined`/
  `.on_snapshot`/`.on_position_done`/`.on_message`, 채널 수신 `subscribe`. 훅은 종류별 워커 레인에서 돌아 처리를 막지 않고
  (`sync(true)` 면 즉시 동기 실행), 종류별 큐 정책(`KeepAll`/`LatestPerKey`/`LatestOnly`), 훅 패닉 격리(`Error` 이벤트), `finish` 요약.
- 기본 출력 훅 `sinks`: `TimelineSink`(timeline.txt), `RunLogSink`(run.log·DONE), `ZonePlySink`(full/*.ply), `ModelSink`(work/models),
  `SnapshotSink`(초벌 정렬·마스킹·사건별 스냅샷·manifest.json·재고정 final_frame). `sinks::attach(pipeline, out, opts)` 로 한 번에 붙이고,
  종류별 목록 `default_sinks` 도 있다. 출력 문구·파일 이름·manifest 형식은 0.2 의 `skyrecon stream` 과 같다.
- 후처리 메모리 입력 `post::run_zones`(`PostZones`): 이벤트로 모은 구역 점군·모델을 바로 받는다. 기존 `post::run` 은 이를 감싼다.
- 예제 `examples/hooks.rs`(합성 장면에 람다 훅 + 기본 출력 훅), 시험 `tests/sinks_equivalence.rs`
  (`run_stream` 출력과, 같은 이벤트 열을 기본 훅에 흘린 출력의 timeline·run.log·파일 목록·manifest·점 수 비교).
### 변경
- `util::Logger::clock_at`/`stamped_at`: 주어진 시각으로 `[HH:MM:SS]` 머리를 붙인다(이벤트 발생 시각 기록용).

## [0.2.0] — 작업 중 (2026-10-09)
### 추가
- 조밀화 융합 실행기 `skyrecon densify --fusion-variants <파일>`: 깊이맵은 한 번만 만들고 융합 설정만 바꿔 여러 결과를 냄.
- 융합 옵션 `--fusion-residual none|release|second-pass`(남은 픽셀 2차 융합)와 `--residual-*`, 2차 점만 쓰는 `--residual-out`.
- 점수 기반 융합 `--fusion-mode score`(`--score-tau`, `--score-sigma-e`, `--score-sigma-theta`, `--score-lambda`).
- 조밀화 튜닝 기록 `docs/tuning/`: 밀도·입체감 스윕(2026-10-09), 잔여 픽셀 융합 비교(2026-10-09).
### 변경 예정
- 조밀화 기본값(융합 최소 일치 뷰, 법선 허용 각도, 깊이맵 필터, 중앙값 필터, 틈 메우기)을 스윕 결과로 갱신.

## [0.1.0] — 2026-10-08
### 추가
- 한 프로세스 점진 재구성 파이프라인 `skyrecon stream`: 위치 단위 도착 → 특징 추출 → 짝 매칭 → 전역 SfM → 영상 등록·삼각측량 →
  구역별 초벌(BA 없음)·정밀(BA) → GPS(ENU) 정렬 → 조밀화 → 단계별 스냅샷·재고정.
- 크레이트: core, features(SIFT), matching(두 뷰 기하), ba(번들 조정), sfm(전역·증분), align(GPS·Sim3), dense(왜곡 보정·GPU 다시점 스테레오),
  cuda(GPU 백엔드), cli.
- 모델 파일 형식(cameras/images/points3D) 읽기·쓰기와 단계별 호환 하위 명령(`interop`).
### 측정 (Tesla V100, 드론 3대 × 80위치 = 240장)
- 전체 스트리밍 마지막 정밀본 648초, 조밀화 240장 61초, 1,347만 점.
