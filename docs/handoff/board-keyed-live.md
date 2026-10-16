# 현황판 점 키 기반 라이브 레이어: 인수인계

브랜치 `feat/board-keyed-live` (feat/cumulus3d-06-live 위에 쌓임, PR 대상 develop). 커밋 메시지 규칙은 CLAUDE.md(영어, AI 표기 금지).

## 목표
구역별 점 청크를 쌓아 생기는 유령(겹침) 표면을 없애고, cumulus3d 0.6 포인트 키 델타(`PointsDelta`)를 슬롯 버퍼에 적용해 로컬 cumulus3d 뷰어와 같은 모습으로 보여 준다. 레퍼런스: `docs/reference/cumulus3d-viewer/`.
요구: removes→adds→moves 순서와 seq 순서 적용, run id 바뀌면 빈 버퍼로 시작, 늦게 붙으면 seq 1부터 재생, 구 Points 청크 렌더 무시, add/remove 페이드 150 ms, move 글라이드 700 ms(GPU), 첫 델타 전 대기 화면("3D 공간 생성 중"), 우상단 모니터(생성 중 · 구역 N · 진행 · 마지막 갱신 n초 전 / 생성 완료 · 총 N점 · m:ss), 이벤트 타임라인(seq, 구역, REFINED/REPLACE/FINAL, T+, +Δs, 점 수, MOVED/NEW/GONE, 진행 바, 행 클릭=seek, 이전/다음 단계=undo 역애니메이션, 라이브 추종), 컨트롤(재생/일시정지, 속도, 점 크기, 색상 RGB/구역/높이, 화면 맞춤), 프레임당 20만 점 업로드 예산.

## 완료 (이 브랜치)
- `src/skylens_client/delta_log.rs`: DeltaLog (아래 API), SlotTable(CPU 슬롯 미러), 역연산 Batch, 단위 테스트 7개 (cumulus3d DisplayMirror 규칙과 동일한 참조 미러와 비교, undo 역연산, seek, run 리셋, 늦은 합류 배치, 스냅샷).
- `src/shared/viewer/chunks.rs`: `ChunkKind::PointsDelta` 처리 (`Change::PointsDelta`, `Applied.delta`, run 리셋, 중복 seq는 Stale, 구역의 첫 refined/replace 델타 = 구역 도착 -> 탐지 공개, 델타 모드가 되면 구 Points 청크는 Stale).
- `src/skylens_client/board.rs`: `Board.live: DeltaLog`, 델타 일괄 투입(`refresh`), 슬롯 테이블에서 구역별 `segments`(샘플, 높이 범위, 코어 박스, 중심, 점 크기) 유도 -> 프레이밍, 미니맵, 점 패널 크기 계산이 델타 모드에서도 동작. `Board::live_ops(who)`로 뷰마다 ops 커서(LIVE_MAIN, LIVE_INSET).
- 테스트 통과: `cargo test -p skylens-client -p skylens-viewer`.

## 남은 일 (순서대로)
1. `feat/pointblitz-anim`(origin, 0f9fe75 이후) 머지: pointblitz-core 0.2.1을 `third_party/pointblitz/pointblitz-core`에 벤더링하고 `[patch.crates-io]`에 추가한 브랜치. API: `AnimatedPoints::new(device, color_format, origin:[f64;3], capacity)`, `apply(device, queue, adds:&[Add{key,pos,rgb}], moves:&[Move{key,pos,rgb:Option}], removes:&[u64], now_s)`, `prepare(...)`, `draw(&mut pass)`, `set_point_size`, `set_color_mode(Rgb|Height{low,high}|Flat)`. 구역 색상 모드(Zone)와 Add에 zone이 필요하면 pblitz 쪽에 요청(`scratchpad/status/pblitz.md`는 이 Mac 로컬).
2. `src/shared/viewer/view3d.rs`: `Gpu3d`에 `live: Option<AnimatedPoints>` 추가, `render()`의 pass 안에서 `prepare`(카메라, fog, now) 후 `points.draw` 다음에 `live.draw`. origin은 첫 배치의 중심 등 장면 중심.
3. `src/skylens_client/scene.rs`, `inset.rs`: `frame()`에서 `board.live_ops(LIVE_MAIN / LIVE_INSET)`를 받아 Reset(레이어 비움) / Apply(Batch)를 `AnimatedPoints.apply`로 전달 (Batch의 animate=false면 now를 과거로 줘서 스냅, 프레임당 20만 점 예산 분할). 기존 `gpu.sync_segments`/RevealFade 점 구름 경로는 델타 모드에서 건너뜀 (`board.store.delta_mode()`); 메시 블록 경로는 유지. 점 패널(inset)은 같은 델타를 높이 색상 모드로 그림. `gpu_lost`에서 `board.live_ops_reset(who)`.
4. 점 크기/색상 모드/재생 속도 훅: Board에 설정 struct(점 크기 배율, 색상 모드 RGB/구역/높이)를 두고 패널이 쓰고 레이어가 읽음. 기존 점/고도 색 토글(`point_layer_on`)은 유지.
5. 합류: `feat/board-live-panels`(패널 에이전트: panels_live.rs, CSS 대기 화면/모니터/타임라인/컨트롤, 구간 사다리 매핑)를 이 브랜치에 머지. 패널은 `DeltaLog`만 사용한다.
6. 검증: 재생 도구 `feat/delta-replay`(origin): `cargo run -p skylens-demo --bin delta-replay -- --stream <v4 스트림 디렉터리> --listen 127.0.0.1:18711 --anchor <lat,lon[,alt]> --speed 5`, Go 코어를 `SKYLENS_MODEL_GRPC=127.0.0.1:18711`로 실행(18101 포트는 다른 프로세스가 씀). 웹 빌드 `PROFILE=dev bash src/skylens_client/build-web.sh`(wasm32 타깃과 wasm-bindgen 필요), dist를 localhost로 서빙해 실제 Chrome(`--remote-debugging-port`, 임시 `--user-data-dir`, WebGPU는 보안 컨텍스트 필요)으로 스크린샷: 대기, 진행 중(타임라인+모니터), REPLACE 글라이드 중, 최종. 무산(유령) 없이 무대/객석이 하나의 표면인지 확인. 스트림 예: `/Users/swoo/Desktop/study/NETChallenge/results/theater_30m/v040_20261016/v040/out/live/stream`(로컬 경로, 클라우드에서는 없음).
7. PR: develop 대상, "feat/cumulus3d-06-live 위에 쌓임" 명시, 병합하지 않음. 네이티브 `cargo build -p skylens-client`와 웹 빌드 둘 다 확인. 

## DeltaLog API (`skylens_client::delta_log`)
`DeltaLog::new()`, `push(PointsDelta) -> Pushed{Stored,NewRun,Ignored}`, `push_batch(Vec<_>)`(마지막만 애니메이션), `entries() -> &[LogEntry]`, `len()`, `applied()`, `current()`, `run_id()`, `is_final()`, `is_live()`, `table()`, `version()`.
`LogEntry{seq, run_id, kind: Kind{Refined,Replace,Final}, zone, t_ms, n_add, n_move, n_remove, total_points, positions, positions_total}`.
명령: `seek(seq)`(Reset + 무애니메이션 재생, 끝이면 라이브), `step_back()`(역연산 Batch, 라이브 해제), `step_forward()`, `follow_live()`.
레이어 소비: `ops_since(&mut Option<u64>) -> Vec<LayerOp{Reset, Apply(Rc<Batch{removes,adds,moves,capacity,animate}>)}>`, 새 소비자(None)는 현재 상태 전체를 한 배치로 받음. 보드는 `Board::live_ops(LIVE_MAIN|LIVE_INSET)`로 감싸 트림까지 처리.

## 테스트 방법
단위: `cargo test -p skylens-client -p skylens-viewer`. 시각 확인은 위 6번. 레퍼런스 뷰어는 `docs/reference/cumulus3d-viewer/serve.sh` 없이 `index.html?data=<스트림>` 로 정적 서빙해 비교 가능.
