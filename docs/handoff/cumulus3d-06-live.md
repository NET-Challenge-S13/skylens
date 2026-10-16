# 인수인계: cumulus3d 0.6 라이브 표시 델타 (feat/cumulus3d-06-live)

## 목표
모델 서버가 cumulus3d `ReconNode`(안정 표시, 고정 ENU 원점, `display_deltas(stride 6)`)를 V100 참조 실행과 같은 설정으로 만들고, 합성된 점 델타(`POINTS_DELTA` 청크)만 내보낸다. 메시는 합성 표시에서 만들어 점 표면과 겹치지 않게 한다.

## 현재 상태 (이 커밋 기준)
- 이미 커밋됨(b3737c6 이전): points-delta 청크·코어 저장 규칙·run id, 현황판 표시, `delta-replay` 도구.
- 이 커밋에 담긴 미커밋 작업:
  - `jobs.rs`: `live_plan()`(settle_tol 0.01, settle_max_zone 3), `dense_for()`(fast, max_levels 1, visibility 8, 융합 평활 2회/반경 1.5/법선 15도).
  - `mod.rs`: `LIVE_SEED = 1`, `DEFAULT_DISPLAY_STRIDE = 6`, 노드에 `.plan(live_plan())`, `.seed()`. 증분 삼각측량은 이미 켜져 있음.
  - `live.rs`: delta/both 모드에서 구간 점군 메시를 끄고 합성 표시(거울)에서 메시 생성(`mesh_display`). 테스트 추가.
  - `serving_config.rs`, `docs/COMPONENTS.md`, `src/skylens_model/README.md`: 기본값과 설정표 갱신.
  - `examples/delta_replay.rs`: cargo fmt만 적용.
- 검증 완료: `cargo test -p skylens-model -p skylens-shared` 통과(196+9), `cargo test -p skylens-demo` 80 통과, cargo fmt 적용. clippy는 기존 경고만 있고 새 오류 없음(전부 확인하지는 않음).
- crates.io 확인 시점에 cumulus3d-cli 최신은 0.6.0. 0.7.0이 올라왔으면 올리고 `.gpu()` 희소/BA 백엔드를 문서대로 켠다.

## 남은 일
1. crates.io에서 0.7.0 확인, 있으면 bump.
2. 내장 `intrinsics_prior`(fx=fy=1329 @1920x1080, cx 960, cy 540, 왜곡 0, sigma 0.005/0.01): 0.6.0 에서는 `NodeBuilder::build` 가 `Plan.sparse.intrinsics_prior` 를 세션에 넘기지 않아 적용 불가(jobs.rs `live_plan` 주석). 0.7.0 에 노출되면 적용.
3. zones 12/overlap 2 와 데모 프레임율 대응을 문서에 명시(COMPONENTS.md 표에 12/2 있음, 프레임율 환산 미작성).
4. `go test ./...` (코어 변경 없으면 생략 가능), `delta-replay` 빌드 확인(`cargo clippy -p skylens-model --all-targets` 로 예제 포함 빌드됨).
5. PR 을 develop 으로 연다. 본문: 변경 내용, 프로토콜, 설정표, 테스트, V100 현장 검증 항목. 커밋·PR 에 AI 표기 금지(CLAUDE.md).

## 테스트 방법 (macOS)
`.cargo/config.toml` 의 `TORCH_CUDA_VERSION` 은 macOS 에서 실패한다. 임시로 주석 처리하고(커밋 금지), pip 로 torch==2.11.0 을 받아
`LIBTORCH=<site-packages>/torch DYLD_LIBRARY_PATH=$LIBTORCH/lib LIBTORCH_BYPASS_VERSION_CHECK=1 cargo test -p skylens-model -p skylens-shared -p skylens-demo`.
Linux CI 는 `.github/workflows/cumulus3d-sync.yml` 의 CPU libtorch 방식을 따른다.
