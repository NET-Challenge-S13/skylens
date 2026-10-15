[English](https://github.com/AH100-1/cumulus3d/blob/main/crates/cli/README.md) | 한국어

# cumulus3d-cli

드론 편대 영상으로 3D 점군을 점진적으로 만드는 `cumulus3d` 실행 파일과, 그 조립 부분을 담은 라이브러리(`cumulus3d_cli`).

**파이프라인에서 맡는 단계**: 전체 조립. 다른 크레이트(특징 추출 → 매칭 → SfM → BA → GPS 정렬 → 조밀화)를
한 프로세스 안에서 잇고, 영상이 위치 단위로 도착하는 동안 구역별 초벌(BA 없음)·정밀본(BA 있음)을 이벤트로 내보내는
**점진 스트리밍**을 맡는다. 단계 사이 데이터는 메모리 자료 구조로 넘긴다(중간 데이터베이스·모델 폴더 없음).

빌드: `cargo build --release -p cumulus3d-cli` → `target/release/cumulus3d`.

## 실행 파일 사용법

### `cumulus3d stream`

```text
cumulus3d stream --src <입력 폴더> --out <출력 폴더> [--span 12 --overlap 2 --stride 3] [옵션]
```

입력 폴더: `images/camF|camR|camL/<cam>_NNNN.jpg`, `gps_ref.txt`(줄: `camF/camF_0000.jpg 위도 경도 고도`).
`camF_NNNN.jpg` 중 `NNNN % stride == 0` 인 프레임만 쓰며, **위치 = 선택된 프레임 목록의 순번**이다.
출력 폴더는 지우고 새로 만든다.

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--span` / `--overlap` / `--stride` | 12 / 2 / 3 | 구역 크기·겹침(위치 수)·프레임 간격 |
| `--incremental-triangulation` | 끔 | 새로 등록된 영상만 삼각측량(빠름, 결과가 다를 수 있음) |
| `--depth-cache` | 끔 | 겹침 영상의 깊이맵 재사용 |
| `--fixed-enu-origin` | 끔 | 모든 ENU 정렬의 원점을 GPS 파일 첫 줄로 고정 |
| `--pm-backend` | cuda | PatchMatch 백엔드. 장치가 없으면 조밀화 단계에서 오류 |
| `--mvs-profile` | fast | 조밀화 프로파일(`fast`\|`quality`) |
| `--sift-backend` / `--match-backend` | cpu | SIFT·기술자 매칭 백엔드(`cpu`\|`cuda`, 결과 동일) |
| `--gpu` | 끔 | 위 세 백엔드를 모두 `cuda` 로 |
| `--serialize-dense` | 끔 | 초벌·정밀 조밀화 백엔드 호출을 하나씩 직렬화 |
| `--threads N` | 0 | rayon 스레드 수(0 = 코어 수) |
| `--max-positions N` | 전체 | 앞 N 위치만 |
| `--save-models` | 끔 | 구역별 희소 모델(`work/models/`) 저장 |
| `--dense-max-image-size` | 960 | 왜곡 보정 최대 영상 크기 |
| `--number-views` | 10 | 조밀화 원천 뷰 수(≤ 32) |
| `--max-num-features` / `--sift-max-image-size` | 8192 / 3200 | SIFT 최대 특징 수·입력 크기 |
| `--seed N` | 없음 | 두 뷰 기하·GPS 정렬 RANSAC 시드 고정 |
| `--no-dense` | 끔 | 조밀화 생략(개발·시험용; 후처리 출력 없음) |
| `--quiet` | 끔 | 진행 기록을 run.log 에만 |

출력: `timeline.txt`, `run.log`, `full/{preview,refined}/*.ply`(구역별 조밀 점군), `aligned/`, `snapshots/`(사건별 스냅샷,
`manifest.json`), `final_frame/`(마지막 정밀 좌표계로 맞춘 구역 점군), `work/models/`(`--save-models`), `DONE`.
훅별 출력 표는 `sinks` 모듈 문서에 있다.

```bash
cumulus3d stream --src <input-dir> --out <output-dir> --stride 1
cumulus3d stream --src <input-dir> --out <output-dir-quick> --stride 1 --max-positions 15 --dense-max-image-size 320
```

### `cumulus3d run` / `cumulus3d plan` (선언형 계획)

```text
cumulus3d run <plan.toml> [--check]        # 계획 검사(문제를 모두 한 번에) 후 실행
cumulus3d plan --print-default             # 기본 계획(aerial-formation 프리셋) TOML
cumulus3d plan --check <plan.toml>         # 검사만
```

계획 파일은 `stream` 옵션과 같은 설정을 담는다(`[input]`, `[zones]`, `[align]`, `[features]`, `[matching]`, `[sparse]`,
`[dense]` 와 `[dense.fusion]`/`[dense.filter]`, `[sinks]`, 최상위 `seed`, `threads`, `gps`, `pairing`).
빠진 항목은 기본값, 모르는 항목은 오류. 예: 저장소 루트의 `examples/plans/aerial-formation.toml`.
같은 설정이면 `run` 은 `stream` 과 같은 출력을 만든다.

### 단계별 하위 명령

`feature_extractor`, `matches_importer`, `global_mapper`, `image_registrator`, `point_triangulator`, `bundle_adjuster`,
`model_aligner`, `model_analyzer`, `model_converter`, `image_deleter`, `image_undistorter`, `densify`.
기존 스크립트가 쓰던 명령 이름·옵션 문자열(`--database_path`, `--ImageReader.camera_model` 등)을 받는다.
단계 사이 상태는 `--database_path`(cumulus3d 특징 저장소 이진 파일, `C3DFS` 형식, 0.3.0 이전의 옛 표지 파일도 읽음)와 모델 폴더(`cumulus3d_core::interop` 형식)로 잇는다.
불린 옵션은 `1/0/true/false` 를 받고, GPU 관련 옵션(`--FeatureExtraction.use_gpu` 등)은 받기만 하고 무시한다.

```bash
cumulus3d feature_extractor --database_path db.c3dfs --image_path images --ImageReader.single_camera_per_folder 1 --ImageReader.camera_model OPENCV
cumulus3d matches_importer --database_path db.c3dfs --match_list_path pairs.txt --match_type pairs
cumulus3d global_mapper --database_path db.c3dfs --image_path images --output_path sg0   # 모델은 sg0/0
cumulus3d model_aligner --input_path in --output_path out --ref_images_path gps.txt --ref_is_gps 1 --alignment_type enu --alignment_max_error 3
cumulus3d model_converter --input_path model --output_path out --output_type BIN|TXT|PLY
cumulus3d image_undistorter --image_path images --input_path in --output_path dense --max_image_size 960
cumulus3d densify -i dense -o dense.ply [--mvs-profile fast|quality --number-views 10 --fusion-mode consistency|traversal|score --stats]
cumulus3d densify -i model --image_path images -o dense.ply      # 왜곡 보정까지 메모리에서
cumulus3d densify -i dense -o out/x.ply --fusion-variants variants.txt   # 깊이맵 한 번, 융합 설정 여러 개
```

`model_aligner` 는 `--ref_is_gps 1 --alignment_type enu` 만 지원한다. 특징 저장소는 SQLite 가 아니라 cumulus3d 자체 이진 형식이다.

## 주요 진입점

| 함수/타입 | 역할 | 입력 → 출력 |
|---|---|---|
| `session::SessionConfig::new` | 세션 설정 기본값(구역 12/겹침 2, CPU SIFT·매칭, 조밀화 없음) | 영상 폴더 → `SessionConfig` |
| `session::Session::new` | 빈 점진 재구성 상태 값 | `SessionConfig` → `Session` |
| `session::step` | 위치 하나 처리: 특징 → 짝 매칭 → 등록·삼각측량 → 구역 닫힘 시 초벌, 정밀은 배경 | `(&Session, FrameSet)` → `(Session, Vec<Event>)` |
| `session::poll` | 끝난 배경 정밀 작업 이벤트 수거(기다리지 않음) | `&Session` → `(Session, Vec<Event>)` |
| `session::command` | `ResetFrom(위치)` 되감기 / `InvalidateZone(k)` 구역 재생성 | `(&Session, Command)` → `(Session, Vec<Event>)` |
| `session::finish` | 남은 구역 닫기 → 배경 작업 대기 → 요약 | `&Session` → `(Session, Vec<Event>)` |
| `pipeline::Pipeline::new` | 리듀서를 감싸 훅·큐·패닉 격리 제공 | `R: Reducer`(예: `Session`) → `Pipeline<R>` |
| `Pipeline::on` / `on_any` / `on_frame` / `on_zone_preview` / `on_zone_refined` | 람다 훅 등록 | 이벤트 종류 + 클로저 → `Pipeline` |
| `Pipeline::push` / `finish` | 입력 넣기 / 끝내고 요약 받기 | `Input` → 낸 이벤트 수 / → `(R, Summary)` |
| `Pipeline::subscribe` | 모든 이벤트를 받는 채널 | → `Receiver<Event>` |
| `events::Event` | 단계별 결과(`ZonePreview`, `ZoneRefined`, `FrameRegistered` …) | — |
| `sinks::attach` | `cumulus3d stream` 과 같은 파일 출력 훅을 붙임 | `(Pipeline, 출력 폴더, &SinkOptions)` → `Pipeline` |
| `declare::Recon::declare` → `.build()` → `.run()` | 선언형 빌더: 계획 기록 → 검사 → 한 번에 실행 | 빌더 메서드 → `Recon` → `Summary` |
| `declare::Plan::from_toml` / `to_toml` | 계획 ↔ TOML(`cumulus3d run plan.toml`) | `&str` ↔ `Plan` |
| `stream::run_stream` | `cumulus3d stream` 전체 실행(`declare` 경유) | `StreamConfig` → `Result<(), String>` |
| `stream::Layout::discover` / `stream::frame_set` | 입력 폴더 → 위치별 프레임 묶음 | `(폴더, stride)` → `Layout`; `(&Layout, 위치)` → `FrameSet` |
| `densewrap::dense_model` | 모델 하나를 조밀화(영상 제외 → 왜곡 보정 → densify) | `(&Reconstruction, keep, 영상 폴더, &DenseConfig)` → `DenseRun` |
| `interop::run` | 단계별 하위 명령 실행 | `InteropCmd` → `Result<(), String>` |

## 공개 항목

### `session` — 상태 값과 리듀서

| 항목 | 종류 | 역할 |
|---|---|---|
| `FrameImage` | struct | 프레임 묶음의 영상 하나(`camera`, `name`) |
| `FrameSet` | struct | 위치 하나의 프레임 묶음(`images`, `gps`); `new((카메라, 이름) 목록)` |
| `SessionConfig` | struct | 세션 설정(영상 폴더, GPS, 구역 크기·겹침, 시드, 특징 추출 옵션, SIFT·매칭 백엔드, 조밀화, 되감기 수); `new` |
| `Input` | enum | 리듀서 입력: `Frames`, `Command`, `Poll`, `Log`, `Finish` |
| `EventSink` | type | 즉시 전달 수신기 `Arc<dyn Fn(Event) + Send + Sync>` |
| `PreviewZone` | struct | 구역 초벌 결과(범위, 모델, 점군, 좌표계) |
| `RefinedZone` | struct | 구역 정밀 결과(범위, BA + GPS 정렬 모델, 점군, `done`) |
| `Session` | struct | 점진 재구성 상태 값(`Clone` 이 싸다). `Reducer` 구현 |
| `Session::new` / `advance` / `apply` | fn | 생성 / 소유 상태 전이 / 제자리 상태 전이 |
| `Session::set_sink` / `emit_external` | fn | 즉시 전달 수신기 설정 / 외부 이벤트를 세션 버전 순서로 발행 |
| `Session::positions` / `frames` / `position_of` / `model` / `adopted` / `store` / `graph` / `config` | fn | 조회: 처리한 위치 수, 위치별 프레임, 영상 위치, 체인 모델, 채택된 정밀 구역, 특징 저장소, 대응 그래프, 설정 |
| `Session::arrived_zones` / `previews` / `refined` / `pending_jobs` / `version` / `failed` / `is_finished` | fn | 조회: 도착 구역, 초벌·정밀 결과, 남은 배경 작업, 이벤트 버전, 치명 오류, 종료 여부 |
| `step` / `poll` / `command` / `finish` | fn | 함수형 리듀서(위 진입점 표) |
| `pairs_for` | fn | 위치 p 의 매칭 짝 규칙(같은 카메라 간격 1..5, 8, 16 / 다른 카메라 위치 차 0..4) |
| `zones_upto` | fn | 전체 위치 수·구역 크기·겹침 → 구역 범위 목록 |

### `pipeline` — 훅 파이프라인

| 항목 | 종류 | 역할 |
|---|---|---|
| `Reducer` | trait | 파이프라인이 구동하는 리듀서(`step`, `poll`, `command`, `finish`) |
| `QueuePolicy` | enum | 종류별 큐 정책: `KeepAll`(기본), `LatestPerKey`, `LatestOnly` |
| `Summary` | struct | 실행 요약(이벤트 수, 종류별 수, 훅 호출·패닉 수, 버린 이벤트 수) |
| `Pipeline` | struct | 리듀서 + 훅 실행기(비동기 레인 또는 동기) |
| `Pipeline::new` / `sync` / `policy` | fn | 생성 / 동기 모드 / 종류별 큐 정책 |
| `Pipeline::on` / `on_any` / `on_frame` / `on_zone_preview` / `on_zone_refined` / `on_snapshot` / `on_position_done` / `on_message` | fn | 훅 등록 |
| `Pipeline::subscribe` | fn | 모든 이벤트(훅 패닉 오류 포함) 수신 채널 |
| `Pipeline::push` / `poll` / `command` / `flush` | fn | 입력 / 배경 결과 수거 / 명령 / 훅 큐 비우기 |
| `Pipeline::reducer` / `reducer_mut` / `finish` | fn | 리듀서 참조 / 가변 참조 / 종료 후 `(R, Summary)` |

### `events` — 이벤트와 명령

| 항목 | 종류 | 역할 |
|---|---|---|
| `Meta` | struct | 모든 이벤트의 머리말(세션 버전, 발생 시각) |
| `ZoneRange` | struct | 구역 번호와 위치 범위 `[lo, hi)` |
| `Event` | enum | 단계별 결과: `Started`, `FrameIngested`, `FeaturesExtracted`, `PairsMatched`, `ModelInitialized`, `FrameRegistered`, `PositionDone`, `ZoneArrived`, `ZonePreview`, `RefineStarted`, `ZoneAdjusted`, `ZoneRefinedPose`, `ZoneRefined`, `BaseAdopted`, `Reanchored`, `Snapshot`, `AllPositionsDone`, `AllRefinedDone`, `Finished`, `Reset`, `ZoneInvalidated`, `Log`, `Warning`, `Error` |
| `Event::kind` / `meta` / `timeline_text` | fn | 종류 / 머리말 / timeline.txt 문구 |
| `EventKind` | enum | 이벤트 종류(훅 등록·필터링용) |
| `Command` | enum | `ResetFrom(위치)`, `InvalidateZone(구역)` |

### `sinks` — 기본 출력 훅

| 항목 | 종류 | 역할 |
|---|---|---|
| `Hook` | type | 훅 `Box<dyn FnMut(&Event) + Send>` |
| `TIMELINE_TABLE` | const | 이벤트 종류 → timeline 문구 변환표 |
| `Sink` | trait | 이벤트를 받아 출력을 만드는 훅(`kinds`, `handle`) |
| `hooks` / `any_hook` | fn | `Sink` → 종류별 훅 목록 / `on_any` 용 훅 하나 |
| `RunLog` | struct | run.log 공유 손잡이(`open`, `logger`, `line`, `stamped_at`, `add_time`) |
| `TimelineSink` | struct | `timeline.txt` 출력(`create`, `line`) |
| `RunLogSink` | struct | `run.log`·`DONE` 출력 |
| `ZonePlySink` | struct | `full/{preview,refined}/*.ply` 출력 |
| `ModelSink` | struct | `work/models/` 구역별 희소 모델 출력 |
| `SnapshotSink` | struct | `aligned/`, `snapshots/`, `final_frame/`, run.log 요약(`finish`) |
| `SinkOptions` | struct | 기본 훅 설정(`echo`, `save_models`) |
| `DefaultSinks` | struct | 위 훅 묶음(`new`, `with_set`, `run_log`, `into_hook`) |
| `SinkSet` | struct | 켤 기본 훅(`all()`) |
| `default_sinks` | fn | 기본 훅을 종류별 목록으로 |
| `attach` | fn | 파이프라인에 기본 훅을 `on_any` 하나로 붙임 |

### `stream` — `cumulus3d stream` 조립

| 항목 | 종류 | 역할 |
|---|---|---|
| `CAMS` | const | 카메라 폴더 `["camF", "camR", "camL"]` |
| `StreamConfig` | struct | 스트림 설정(명령줄 옵션과 1:1); `new(src, out)` |
| `Layout` | struct | 위치 ↔ 영상 이름(`discover`, `name`, `position`) |
| `frame_set` | fn | 위치 p 의 프레임 묶음 |
| `pairs_for_position` | fn | 위치 p 의 매칭 짝(이름 짝) |
| `regions` | fn | 구역 목록 `(k, lo, hi)` |
| `session_config` | fn | `StreamConfig` → `SessionConfig` |
| `Layout::discover_in` / `frame_set_of` | fn | 임의 영상 폴더·카메라 목록용 `discover` / `frame_set` |
| `run_stream` | fn | 스트림 실행(`Plan::from_stream` → `Recon`) |

### `declare` — 선언형 빌더(계획 → 검사 → 실행)

전체 안내: [docs/DECLARATIVE.ko.md](https://github.com/AH100-1/cumulus3d/blob/main/docs/DECLARATIVE.ko.md).

| 항목 | 종류 | 역할 |
|---|---|---|
| `Recon` | struct | 검사를 마친 실행 가능한 재구성: `declare()`, `from_plan(plan)`, `plan()`, `positions()`, `layout()`, `run() -> Result<Summary, String>` |
| `ReconBuilder` | struct | 계획만 기록. `input`, `images`, `cameras`, `preset`, `stride`, `positions`, `pairing`, `zones`, `gps`, `no_gps`, `align`, `fixed_enu_origin`, `features`, `match_backend`, `gpu`, `incremental_triangulation`, `dense`, `no_dense`, `sinks`, `seed`, `threads`; 훅 `on`, `on_any`, `on_frame`, `on_zone_preview`, `on_zone_refined`, `on_snapshot`, `on_position_done`, `on_message`, `policy`; `plan()`, `build() -> Result<Recon, PlanError>` |
| `Plan` | struct | 계획 값(Clone/PartialEq/Debug, serde): `seed`, `threads`, `gps`, `pairing`, `input`, `zones`, `align`, `features`, `matching`, `sparse`, `dense`, `sinks`; `to_toml`, `from_toml`, `load`, `from_stream` |
| `Source`, `Zones`, `Align`/`AlignFrame`, `Features`, `Matching`, `Sparse`, `Pairing` | struct/enum | 계획 항목 |
| `Dense`, `Fusion`, `Filter` | struct | 조밀화 계획; `Dense::profile("fast").fusion_min_views(3)…`, `Dense::off()`, `apply`, `config` |
| `Sinks` | struct | 출력 훅: `default_files(out)`, `none()`, `timeline`, `run_log`, `zone_ply`, `snapshots`, `models`, `echo`, `clean` |
| `PlanError`, `Problem` | struct | 검사 문제 전부(`field`, `message`); `has(field)` |
| `DEFAULT_PRESET`, `PRESETS` | const | `"aerial-formation"` |

### `densewrap` — 조밀화 공용 경로

| 항목 | 종류 | 역할 |
|---|---|---|
| `make_pm_backend` | fn | PatchMatch 백엔드 선택(`cuda`) |
| `parse_profile` | fn | `--mvs-profile` 해석 → `MvsProfile` |
| `make_sift_backend` / `make_match_backend` | fn | SIFT·기술자 매칭 백엔드 선택(`cpu`\|`cuda`) |
| `DenseConfig` | struct | 조밀화 설정(왜곡 보정·장면·densify 옵션, 점수 융합, 백엔드, 직렬화 잠금, 캐시); `new` |
| `DenseRun` | struct | 조밀화 결과 요약(영상 수, 장면, 출력, 단계 시간) |
| `dense_model` | fn | 모델 조밀화(영상 제외 → 왜곡 보정 → 장면 변환 → densify) |

### `post` — 후처리(정렬·스냅샷·재기준)

| 항목 | 종류 | 역할 |
|---|---|---|
| `PostInput` | struct | PLY 경로·사건 문자열 기반 후처리 입력 |
| `ZoneTimes` | struct | 구역별 사건 시각(도착, 초벌, 정밀 자세, 정밀 완료) |
| `ZoneCloud` | struct | 구역 점군 하나(파일 이름, 점군) |
| `PostZones` | struct | 메모리 입력형 후처리 입력 |
| `run` | fn | 후처리 전체(PLY 읽기 → `run_zones`) |
| `run_zones` | fn | 초벌 정렬 → `aligned/` → 사건별 스냅샷·manifest → `final_frame/` |

### `fusion_variants` — 융합 변형 일괄 실행

| 항목 | 종류 | 역할 |
|---|---|---|
| `FusionVariant` | struct | 융합 변형 하나(이름, 플래그, 옵션, 점수 융합 설정) |
| `read_variants` | fn | 변형 파일(`이름\|융합 플래그들`) 읽기 |
| `run_variants` | fn | 깊이맵 한 벌로 변형마다 융합해 PLY·`fusion_variants.tsv` 출력 |

### `interop` — 단계별 하위 명령

| 항목 | 종류 | 역할 |
|---|---|---|
| `parse_flag` | fn | `1/0/true/false` 불린 파서 |
| `InteropCmd` | enum | 하위 명령 12개 |
| `FeatureExtractorArgs`, `MatchesImporterArgs`, `GlobalMapperArgs`, `ImageRegistratorArgs`, `PointTriangulatorArgs`, `BundleAdjusterArgs`, `ModelAlignerArgs`, `ModelAnalyzerArgs`, `ModelConverterArgs`, `ImageDeleterArgs`, `ImageUndistorterArgs`, `DensifyArgs` | struct | 하위 명령별 인자 |
| `extract_colors` | fn | 영상 폴더에서 3D 점 색 추출 |
| `run` | fn | 하위 명령 실행 |

### `util` — 기록·시간·직렬화 도우미

| 항목 | 종류 | 역할 |
|---|---|---|
| `Logger` | struct | run.log(+표준 출력) 기록기(`new`, `line`, `stamped`, `stamped_at`, `clock`, `clock_at`) |
| `Timeline` | struct | timeline 파일 기록(`new`, `ev`, `relative`) |
| `StageTimes` | struct | 단계별 누적 시간(`add`, `snapshot`) |
| `timed` | fn | 단계 하나를 재고 run.log 에 기록 |
| `py_float` | fn | 반올림한 수를 `.0` 붙은 문자열로 |
| `Json` | enum | 작은 JSON 값(`obj`, `f`, `of`, `ints`, `dump`, `py_repr`, `get`) |

## 사용 예

영상 폴더를 위치 단위로 넣어 점진 재구성하고, `cumulus3d stream` 과 같은 파일을 출력한다(실제 영상이 필요하다).

```rust,no_run
use cumulus3d_cli::events::{Event, EventKind};
use cumulus3d_cli::pipeline::Pipeline;
use cumulus3d_cli::session::{Input, Session, SessionConfig};
use cumulus3d_cli::sinks::{self, SinkOptions};
use cumulus3d_cli::stream::{frame_set, Layout};
use cumulus3d_core::io::read_gps_file;
use std::path::Path;

fn main() -> Result<(), String> {
    let src = Path::new("data");
    let layout = Layout::discover(src, 3)?;
    let mut cfg = SessionConfig::new(src.join("images"));
    cfg.gps = read_gps_file(src.join("gps_ref.txt")).map_err(|e| e.to_string())?;
    cfg.total_positions = Some(layout.frames.len());

    let pipeline = Pipeline::new(Session::new(cfg)).on(EventKind::ZonePreview, |e: &Event| {
        if let Event::ZonePreview { range, cloud, .. } = e {
            println!("초벌 구역 {}: 점 {}", range.zone, cloud.len());
        }
    });
    let mut pipeline = sinks::attach(pipeline, Path::new("out"), &SinkOptions::default()).map_err(|e| e.to_string())?;
    for p in 0..layout.frames.len() {
        pipeline.push(Input::Frames(frame_set(&layout, p)));
    }
    let (_session, summary) = pipeline.finish();
    println!("이벤트 {} 개", summary.events);
    Ok(())
}
```

훅 없이 상태 값만 주고받는 함수형 사용:

```rust,no_run
use cumulus3d_cli::session::{finish, step, FrameSet, Session, SessionConfig};

let s0 = Session::new(SessionConfig::new("data/images"));
let frames = FrameSet::new([("camF", "camF/camF_0000.jpg"), ("camR", "camR/camR_0000.jpg")]);
let (s1, events) = step(&s0, frames);
for e in &events {
    if let Some(t) = e.timeline_text() {
        println!("{t}");
    }
}
let (_done, _summary_events) = finish(&s1);
```

두 예는 `src/lib.rs` 의 doc-test 로도 들어 있다. 합성 장면으로 끝까지 도는 예제는 `cargo run --release -p cumulus3d-cli --example hooks`.

## 기능 플래그·하드웨어

- cargo 기능 플래그는 없다. `cumulus3d-cuda` 는 항상 링크되지만 CUDA 라이브러리를 **동적 로딩**하므로 CUDA 가 없는 기계에서도 빌드·실행된다.
- 조밀화(`stream` 기본, `densify` 하위 명령)는 GPU PatchMatch 백엔드가 필요하다(CUDA 12.x 드라이버). 장치가 없으면
  `stream` 은 조밀화 단계에서 오류를 기록하고, `--no-dense` 로 조밀화 없이 돌릴 수 있다. `SessionConfig::new` 기본값은 조밀화 없음.
- `--gpu` 또는 `--sift-backend cuda` / `--match-backend cuda` 는 CUDA 12.x 드라이버가 필요하다. 결과는 CPU 백엔드와 같다.
- 시험: `cargo test --release -p cumulus3d-cli`(합성 장면으로 stream 끝까지, 하위 명령 연쇄, 기본 훅 출력 동등성).
