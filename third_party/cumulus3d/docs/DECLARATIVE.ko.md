[English](DECLARATIVE.md) | 한국어

# 선언형 함수 안내 (`cumulus3d_cli::declare`)

`declare` 모듈은 재구성 파이프라인을 선언적으로 구성하는 API 를 제공한다.
빌더로 구성한 `Plan` 은 부수효과가 없는 값(value)이며, `build()` 에서 일괄 검증된 뒤 `run()` 에서 지연 실행(lazy execution)된다.
빌더 단계에서는 어떤 I/O 나 장치 초기화도 일어나지 않고, 실행은 이벤트 기반 런타임(`session`·`pipeline`·`sinks`) 위에서 이루어진다.

```
Recon::declare()                    빈 계획으로 빌더 생성
  .input(..).stride(..).zones(..)   계획(Plan)에 값만 기록
  .dense(..).sinks(..)              〃
  .on_zone_preview(|..| ..)         훅은 계획과 따로 보관
  .build()?                         계획 검사 → Recon (아직 실행 안 함)
  .run()?                           실행: 세션 + 파이프라인 + 출력 훅 조립 → 끝까지
```

코드: `crates/cli/src/declare.rs` · 시험: `crates/cli/tests/declare_synthetic.rs` · 계획 파일 예: `examples/plans/aerial-formation.toml`

## 1. 세 단계

| 단계 | 함수 | 하는 일 | 부수효과 |
|---|---|---|---|
| 기록 | `Recon::declare()` / `Recon::from_plan(plan)` → `ReconBuilder` 메서드들 | 계획(`Plan`)에 값 기록, 훅 목록에 훅 추가 | 없음 |
| 검사 | `ReconBuilder::build() -> Result<Recon, PlanError>` | 계획 전체 검사. 문제를 모두 모아 한 번에 반환 | 입력 폴더 목록·GPS 파일 읽기만 |
| 실행 | `Recon::run() -> Result<Summary, String>` | 계획대로 엔진을 조립해 위치 반복·배경 정밀 대기·종료까지 실행 | 출력 파일 작성, GPU 사용 |

## 2. 계획 값 `Plan`

실행 로직을 포함하지 않는 값 타입으로, 복제·비교·직렬화(TOML)가 가능하며 시드가 고정되면 같은 계획은 같은 결과를 낸다.

| 묶음 | 필드 | 기본값 (`aerial-formation` 프리셋) |
|---|---|---|
| 최상위 | `seed`, `threads`, `gps`, `pairing` | 없음, 0(코어 수), `<입력>/gps_ref.txt`, `formation` |
| `input` | `images`, `cameras`, `stride`, `positions` | `<입력>/images`, `["camF","camR","camL"]`, 3, 전부 |
| `zones` | `span`, `overlap` | 12, 2 |
| `align` | `frame`(`enu`/`none`), `fixed_origin` | `enu`, false |
| `features` | `backend`(`cpu`/`cuda`), `max_num_features`, `max_image_size` | `cpu`, 8192, 3200 |
| `matching` | `backend` | `cpu` |
| `sparse` | `incremental_triangulation` | false |
| `dense` | `enabled`, `backend`, `profile`(`fast`/`quality`), `max_image_size`, `views`, `serialize`, `depth_cache` | true, `cuda`, `fast`, 960, 10, false, false |
| `dense.fusion` | `mode`, `min_views`, `depth_error`, `normal_error`, `reproj_error`, `residual` | 프로필 기본값 |
| `dense.filter` | `min_views`, `geom_error`, `min_ncc`, `median` | 프로필 기본값 |
| `sinks` | `out`, `clean`, `echo`, `timeline`, `run_log`, `zone_ply`, `snapshots`, `models` | `out`, true, false, true, true, true, true, false |

계획 함수:

| 함수 | 역할 |
|---|---|
| `Plan::to_toml(&self) -> String` | 계획 → TOML 문자열 |
| `Plan::from_toml(&str) -> Result<Plan, String>` | TOML → 계획. 빠진 항목은 기본값, 모르는 항목은 오류 |
| `Plan::load(path) -> Result<Plan, String>` | TOML 파일 읽기 |
| `Plan::from_stream(&StreamConfig) -> Plan` | `cumulus3d stream` 옵션 → 계획 |

## 3. 빌더 `ReconBuilder` — 기록 함수

모든 메서드는 `self` 를 소비해 계획만 갱신한 뒤 반환하므로 메서드 체이닝으로 구성한다.

| 메서드 | 기록하는 계획 항목 |
|---|---|
| `input(src)` | `input.images = src/images`, `gps = src/gps_ref.txt` |
| `images(path)`, `cameras([..])` | `input.images`, `input.cameras` |
| `preset("aerial-formation")` | 카메라 3대 편대 기본 묶음(카메라, 짝 전략, 간격). 모르는 이름은 `build()` 에서 오류 |
| `stride(n)`, `positions(n)` | `input.stride`, `input.positions` |
| `pairing(..)` | `pairing` |
| `zones(span, overlap)` | `zones` |
| `gps(path)`, `no_gps()` | `gps` |
| `align(AlignFrame::Enu \| None)`, `fixed_enu_origin(bool)` | `align` |
| `features(..)`, `match_backend(..)`, `gpu()` | `features`, `matching` (`gpu()` 는 특징·매칭·조밀화 백엔드를 모두 `cuda` 로) |
| `incremental_triangulation(bool)` | `sparse` |
| `dense(Dense)`, `no_dense()` | `dense` |
| `sinks(Sinks)` | `sinks` |
| `seed(n)`, `threads(n)` | `seed`, `threads` |
| `plan()` | 지금까지 기록한 계획 조회 |

**훅 함수** — 클로저는 직렬화할 수 없으므로 `Plan` 이 아니라 빌더의 훅 목록에 보관한다:
`on(EventKind, f)`, `on_any(f)`, `on_frame(f)`, `on_zone_preview(f)`, `on_zone_refined(f)`, `on_snapshot(f)`, `on_position_done(f)`, `on_message(f)`, `policy(kind, QueuePolicy)`.

### 조밀화 설정 `Dense` (체이닝)

`Dense::profile("fast")` 또는 `Dense::off()` 로 시작해 이어 쓴다:
`backend`, `max_image_size`, `views`, `serialize`, `depth_cache`, `fusion_mode`, `fusion_min_views`, `fusion_depth_error`,
`fusion_normal_error`, `fusion_reproj_error`, `fusion_residual`, `filter_min_views`, `filter_geom_error`, `filter_min_ncc`, `median_filter`.

### 출력 설정 `Sinks` (체이닝)

`Sinks::default_files(out)` 또는 `Sinks::none()` 로 시작해 `clean`, `echo`, `timeline`, `run_log`, `zone_ply`, `snapshots`, `models` 로 켜고 끈다.

## 4. 검사 `build()`

`build()` 는 파일 시스템을 변경하지 않으며, 발견한 문제를 계획 항목 경로와 함께 `PlanError` 하나로 모아 반환한다.

| 검사 | 내용 |
|---|---|
| 프리셋 | 아는 이름인지 |
| 입력 | 영상 폴더·카메라 폴더 존재, 카메라 목록이 비어 있지 않고 중복 없음, `stride ≥ 1`, 선택된 위치 ≥ 1 |
| 구역 | `span ≥ 1`, `overlap < span` |
| 정렬 | ENU 정렬이면 GPS 파일이 있고 읽히며 선택된 영상의 기록이 있음 |
| 백엔드 | 이름이 `cpu`/`cuda` 이고, `cuda` 면 장치가 있음 |
| 조밀화 | 켜져 있으면 백엔드 `cuda` + 장치, 프로필 이름, `views` 1..=32, 크기·융합·필터 값 범위 |
| 출력 | 출력 폴더(또는 가장 가까운 상위 폴더)에 쓸 수 있음, 지울 출력 폴더 안에 입력이 들어 있지 않음 |

## 5. 실행 `Recon`

| 함수 | 역할 |
|---|---|
| `Recon::declare() -> ReconBuilder` | 빈 계획(기본값)부터 |
| `Recon::from_plan(Plan) -> ReconBuilder` | 기존 계획(TOML 등)부터 |
| `plan()`, `positions()`, `layout()` | 검사를 통과한 계획, 선택된 위치 수, 영상 배치 조회 |
| `run(self) -> Result<Summary, String>` | 실행 |

`run()` 이 하는 조립:
1. `Plan.dense` → 조밀화 설정, `Plan.sinks` → 기본 출력 훅(`clean` 이면 이전 출력 폴더 정리)
2. 나머지 계획 → 세션 설정(GPS, 구역, 시드, 특징·매칭 백엔드)
3. `Pipeline::new(Session::new(..))` 에 출력 훅과 사용자 훅 연결
4. 위치를 차례로 `push`, 마지막에 `finish()` 로 배경 정밀 작업까지 기다림 → `Summary`

## 6. 쓰는 방법

### 코드

```rust
use cumulus3d_cli::declare::{Dense, Recon, Sinks};

let summary = Recon::declare()
    .input("data")
    .preset("aerial-formation")
    .zones(12, 2)
    .dense(Dense::profile("fast").fusion_min_views(3).fusion_normal_error(25.0))
    .sinks(Sinks::default_files("out"))
    .seed(7)
    .on_zone_preview(|zone, cloud| println!("초벌 구역 {}: 점 {}", zone.zone, cloud.len()))
    .build()?        // 검사
    .run()?;         // 실행
```

### 계획 파일 (TOML)

빌더 메서드와 TOML 키는 1:1 로 대응한다(예: `.zones(12, 2)` ↔ `[zones] span = 12, overlap = 2`).

```toml
seed = 7
gps = "data/gps_ref.txt"
[input]
images = "data/images"
cameras = ["camF", "camR", "camL"]
stride = 3
[zones]
span = 12
overlap = 2
[dense]
profile = "fast"
[dense.fusion]
min_views = 3
normal_error = 25.0
[sinks]
out = "out"
```

### 명령줄

```bash
cumulus3d plan --print-default > plan.toml   # 기본 계획 만들기
cumulus3d plan --check plan.toml             # 검사만
cumulus3d run plan.toml                      # 검사 후 실행
cumulus3d stream --src data --out out        # 기존 명령도 내부에서 이 층으로 실행됨
```

## 7. 보장되는 성질 (시험으로 확인)

- `build()` 전까지 파일·폴더 생성 없음, 훅 실행 없음, 입력 폴더 변경 없음.
- 같은 계획을 두 번 실행하거나 TOML 로 저장했다 다시 읽어 실행해도 결과 파일이 같다(시드 고정 시).
- `cumulus3d run plan.toml` 과 `cumulus3d stream` 의 출력이 같다.
- 잘못된 계획은 실행 전에 실패하고, 모든 문제를 한 번에 보고한다.
