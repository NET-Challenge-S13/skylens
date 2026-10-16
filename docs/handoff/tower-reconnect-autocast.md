# 인계: 관제탑 재접속 대기 + autocast 경고 제거

브랜치 `fix/tower-reconnect-autocast` (base: origin/develop). 목표 PR 대상은 develop.

## 목표
1. 관제탑(`src/skylens_core/control/app.rs`)이 코어보다 먼저 떠도 `fail()`로 끝나지 않고, 현황판(b7847b4)처럼 "코어 연결 대기 중" 상태를 보이며 무한 재시도하다 연결되면 로딩 바로 돌아간다.
2. `src/skylens_model/models/skylensnet/modeling_skylensnet.rs` (~899-902행)에서 `tch::autocast`가 정밀도와 무관하게 매 프레임 호출되어 libtorch 경고가 나온다. 해결된 정밀도가 `Fp16Autocast`일 때만 호출하고 그 외 동작은 동일해야 한다.

## 한 일
- 관제탑: `WAITING_TITLE`, `waiting_text`, `overlay_waits`, `update_waiting` 추가. Connecting/Disconnected 때 첫 Site 전이면 `LoadingScreen::wait`, Connected 때 `progress(0)`로 복귀. 기존 `fail()` 호출 제거. 단위 테스트 3개 추가. `cargo test -p skylens-control` 75개 통과. shared 코드는 수정하지 않음(b7847b4의 backoff/attempts/타임아웃을 그대로 사용).

## 현재 상태 (중단 시점)
- 관제탑 재접속 대기: 완료, 커밋됨(40c494e). `cargo test -p skylens-control` 75개 통과.
- autocast: `modeling_skylensnet.rs`의 `run`을 수정함. 클로저 `forward`를 만들고 `autocast`(CUDA이고 정밀도가 Fp16Autocast)일 때만 `tch::autocast(true, forward)`, 아니면 `forward()`를 직접 호출. 컴파일과 테스트는 아직 검증하지 못함. 별도 테스트는 추가하지 않음(기존 정밀도 선택 테스트가 있음).
- `cargo test -p skylens-model`: macOS에서 `.cargo/config.toml`의 `TORCH_CUDA_VERSION = "cu126"`이 torch-sys 빌드를 실패시킴(CUDA 판은 Linux/Windows 전용). 환경변수를 비워도 "설정됨"으로 취급되므로, 로컬 확인 때는 그 줄을 잠시 주석 처리하거나(커밋하지 말 것) Linux에서 CPU libtorch를 받아 `LIBTORCH`를 지정해야 한다. 이 환경에서는 컴파일이 끝나기 전에 중단함.
- 웹 번들 빌드(`PYTHON3=$(which python3) bash src/skylens_core/control/build-web.sh`): 시작했으나 완료 확인 못 함(cargo 경쟁으로 오래 걸림). 다시 실행해서 dist가 생기는지 확인해야 한다.
- PR은 열지 않았고 머지도 하지 않음.

## 남은 일
- `cargo test -p skylens-model`을 CPU libtorch로 실행해 autocast 변경이 컴파일되고 통과하는지 확인.
- 웹 번들 빌드 확인.
- PR을 develop으로 올리고 CI 통과 후 머지(충돌 없으면). 커밋/PR에 AI 표기 금지, 커밋 메시지는 영어. PR 본문에 대전 호스트 dist 배포는 별도로 한다고 적을 것.
- 대전 배포(로컬에서만): 현재 `netchallenge:skylens_online/web/dist`를 `dist.bak-<시각>`으로 백업한 뒤 `rsync -a --delete src/skylens_core/control/dist/ netchallenge:skylens_online/web/dist/` (재시작 불필요).
