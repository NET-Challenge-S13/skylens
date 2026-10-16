# 인계: 관제탑 재접속 대기 + autocast 경고 제거

브랜치 `fix/tower-reconnect-autocast` (base: origin/develop). 목표 PR 대상은 develop.

## 목표
1. 관제탑(`src/skylens_core/control/app.rs`)이 코어보다 먼저 떠도 `fail()`로 끝나지 않고, 현황판(b7847b4)처럼 "코어 연결 대기 중" 상태를 보이며 무한 재시도하다 연결되면 로딩 바로 돌아간다.
2. `src/skylens_model/models/skylensnet/modeling_skylensnet.rs` (~899-902행)에서 `tch::autocast`가 정밀도와 무관하게 매 프레임 호출되어 libtorch 경고가 나온다. 해결된 정밀도가 `Fp16Autocast`일 때만 호출하고 그 외 동작은 동일해야 한다.

## 한 일
- 관제탑: `WAITING_TITLE`, `waiting_text`, `overlay_waits`, `update_waiting` 추가. Connecting/Disconnected 때 첫 Site 전이면 `LoadingScreen::wait`, Connected 때 `progress(0)`로 복귀. 기존 `fail()` 호출 제거. 단위 테스트 3개 추가. `cargo test -p skylens-control` 75개 통과. shared 코드는 수정하지 않음(b7847b4의 backoff/attempts/타임아웃을 그대로 사용).

## 남은 일
- autocast: `tch::autocast(autocast, || ...)`를 `if autocast { tch::autocast(true, f) } else { f() }` 형태로 변경. `cargo test -p skylens-model` (CPU libtorch 필요, `LIBTORCH` 설정; 스크래치패드나 torch 2.11 venv에서 찾을 것).
- 수동 확인: `PYTHON3=$(which python3) bash src/skylens_core/control/build-web.sh` 후 페이지를 먼저 열고 Go 코어(`target/skylens-core`)를 나중에 띄워 대기 화면 -> 연결 확인.
- PR을 develop으로 올리고 CI 통과 후 머지(충돌 없으면). 커밋/PR에 AI 표기 금지, 커밋 메시지는 영어.
- 대전 배포(이 단계는 로컬에서만 수행): 현재 `netchallenge:skylens_online/web/dist`를 `dist.bak-<시각>`으로 백업한 뒤 `rsync -a --delete src/skylens_core/control/dist/ netchallenge:skylens_online/web/dist/` (재시작 불필요).
