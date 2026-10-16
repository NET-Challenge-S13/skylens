# 현황판 라이브 패널 인수인계 (feat/board-live-panels)

## 목표
cumulus3d 로컬 뷰어의 표시부를 SkyLens 현황판 DOM 패널로 옮긴다: 대기 화면, 우상단 모니터, 이벤트 타임라인, 컨트롤 바, 구간 사다리와 구역 일치.
기존 SkyLens 패널(서버 패널, 미니맵, 탐지 카드 등)과 b7847b4, 8b13803 수정은 그대로 둔다.

## 완료
- `src/skylens_client/panels_live.rs`: 순수 모델. 포맷(T+m:ss, +Δs, 천 단위), 단계(Waiting/Live/Final), 모니터 문구, 진행률, 행 마크업, 카운터(MOVED/NEW/GONE), Replay 상태기계(재생, 일시정지, 팔로우), 설정(LiveSettings, ColorMode), 사다리-구역 매핑. 단위 테스트 포함.
- `src/skylens_client/live_ui.rs`: DOM 연결(LivePanels)과 WaveWidget(캔버스 점 물결). 클릭을 DeltaLog 명령으로 변환한다.
- `src/skylens_client/ui/live-panels.css`: 스타일. `app.rs`의 `page()`가 status-panels.css 뒤에 붙인다.
- `app.rs`: mount, poll, input에 훅을 걸었고 서버 패널 사다리에 타임라인 구역을 합쳤다.
- 네이티브 테스트 63개 통과, 웹(dev) 빌드 성공.

## 빌드하는 DeltaLog API
`skylens_client::delta_log` (feat/board-keyed-live 병합본). 로그는 `Board.live`에 있다.
`entries()`, `applied()`, `is_live()`, `run_id()`, `len()`, `seek(seq)`, `step_back()`, `step_forward()`, `follow_live()`.
`LogEntry{seq,run_id,kind,zone,t_ms,n_add,n_move,n_remove,total_points,positions,positions_total}`.

## 남은 일
1. 시각 확인이 아직 없다. 대기 화면, 모니터, 타임라인, 컨트롤 바 배치와 겹침(탐지 카드, 미니맵, 짧은 뷰포트). `cargo run -p skylens-demo --bin delta-replay` (브랜치 feat/delta-replay)와 Go 코어로 이벤트 14개를 흘려 확인한다. 웹은 `PYTHON3=$(which python3) bash src/skylens_client/build-web.sh`.
2. 설정 훅: `LivePanels.settings`(`Rc<RefCell<LiveSettings>>`: point_size, color_mode, speed, fit_requested)를 점 레이어가 읽게 연결한다. 점 레이어 쪽 설정 struct가 나오면 합친다. 색상 모드 Zone은 점 레이어에 구역색이 필요하다.
3. 일시정지(재생 중 LIVE 해제)는 `seek(현재 seq)`로 대체했다. DeltaLog에 `unfollow()`가 생기면 `panels_live::Replay::control`의 PlayPause 분기를 바꾼다.
4. 타임라인 행 자동 스크롤(`scroll_current_into_view`)과 Blitz 렌더(grid/flex, opacity, display 토글)가 실제로 동작하는지 확인한다. 안 되면 보이는 행 창만 렌더하는 방식으로 바꾼다.
5. 대기 화면 sub에 코어 진행(positions)을 넣으려면 `wait_sub`에 문자열을 넘긴다(지금은 경과 시간만).
6. README.md(`src/skylens_client`)에 패널 설명 추가, 커밋 메시지는 영어, AI 표기 금지.

## 테스트
`cargo test -p skylens-client --lib` (panels_live 테스트 17개 포함).
