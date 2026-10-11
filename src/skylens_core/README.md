# skylens_core (Go)

대전에서 도는 코어 서버. 연산 없이 **보관·스케줄·배포**만 하는 I/O 서버이고, 청크 `payload`는 풀지 않는다. 컴포넌트 경계는 `docs/COMPONENTS.md` §3.4, 설계는 `docs/SPEC.md`가 단일 출처다.

## 무엇이 도는가

| 경로 | 하는 일 |
|---|---|
| `GET /health` | 미션·모델 서버 연결·보관 카운터·프레임·지형·소켓 상태(JSON). VWorld 키는 절대 싣지 않는다 |
| `ws /uplink` | 드론(또는 데모)이 `drone_hello`·`telemetry`를 올린다. 코어는 `assign_route`·`manual_control`을 내려보낸다 |
| `ws /viewer` | 화면(관제탑·현황판). 코어가 밀고, 화면은 `assign_route`·`manual_control`만 보낸다 |
| `/` | 관제탑 웹 빌드 정적 서빙(`SKYLENS_CORE_WEB_DIR`) |

모든 WebSocket 프레임은 바이너리 `skylens.viewer.v1.Envelope` 하나다. `seq`는 소켓마다 1부터 센다.

- **미션:** `idle → assigned → awaiting-drone → active`. 경로 지정 뒤 `SKYLENS_CORE_ASSIGNED_HOLD_MS` 동안 "태스크 지정 완료"를 보여 주고 드론 대기로 넘어간다. **드론 도착은 `drone_hello`로만 판정**하고 텔레메트리는 세지 않는다. 마지막 드론의 업링크가 끊기면 다시 대기로 돌아간다.
- **보관(인메모리):** 정제본은 같은 구간의 미리보기와 이전 세대를 대체하고, 메시 블록은 더 높은 `version`만 받고, 탐지 결과는 도착 순서대로 쌓는다(구간당 상한). 버려진 청크는 화면에도 보내지 않는다. 새 화면은 현재 상태(미션 → 링크 → 경로 → 텔레메트리 → 영상 위치 → 지형·건물 → 점 → 메시 → 탐지 → 서버 상태)를 먼저 받고 이어서 실시간을 받는다.
- **모델 서버(gRPC :8101):** `SubscribeChunks`를 늘 열어 두고, 다시 붙을 때 `after_sequence`를 마지막으로 보관한 시퀀스로 준다. `PushFrames`는 모델이 없는 동안 묶음을 큐에 쌓아 두고, 다시 붙으면 최근에 보낸 묶음을 다시 선언한다(이미 받은 묶음의 재선언은 모델 쪽에서 무시된다).
- **프레임:** 드론이 hello하면 대전 MediaMTX의 그 드론 경로를 `ffmpeg`로 읽어 JPEG를 뽑고, 같은 순간의 편대 카메라 프레임을 `FrameSet (source_id, sequence, revision)`으로 묶는다. 늦게 온 카메라가 있으면 같은 시퀀스를 다음 revision으로 다시 보낸다. 프레임 소스는 `frames.Source` 인터페이스라 데모나 테스트가 직접 넣을 수도 있다.
- **영상 위치:** hello마다 `camera_feed`로 WHEP 주소(기본 `/live/drone<id>/whep`)를 알린다. 영상 자체는 MediaMTX가 보낸다.
- **지형·건물:** 경로가 지정되면 그 영역(여백 포함)의 지형·건물 청크를 `terrain` 패키지에서 받아 다른 청크처럼 보관하고 내보낸다. VWorld 키가 없으면 위성·건물 없이 지형만 낸다.

## 환경변수

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `SKYLENS_CORE_ADDR` | `:8080` | 리슨 주소 |
| `SKYLENS_CORE_WEB_DIR` | `src/skylens_client/dist` | 관제탑 웹 빌드. 상대 경로면 작업 폴더, 그다음 저장소 루트 기준 |
| `SKYLENS_MODEL_GRPC` | `127.0.0.1:8101` | 모델 서버 gRPC 주소 |
| `SKYLENS_CORE_FRAME_SOURCE` | `ffmpeg` | `ffmpeg` 또는 `off` |
| `SKYLENS_CORE_FFMPEG` | `ffmpeg` | ffmpeg 실행 파일 |
| `SKYLENS_CORE_STREAM_PATH` | `drone{id}` | 드론의 MediaMTX 경로 이름. `{id}`·`{station}`(left/center/right) |
| `SKYLENS_CORE_FRAME_URL` | `srt://127.0.0.1:10890?streamid=read:{path}` | ffmpeg 입력. 대전 MediaMTX는 RTSP를 끄고 SRT만 열어 두므로 SRT로 읽는다 |
| `SKYLENS_CORE_FRAME_FPS` | `2` | 드론당 초당 추출 프레임 |
| `SKYLENS_CORE_FRAME_QUALITY` | `5` | JPEG 품질(ffmpeg `-q:v`, 2가 최고) |
| `SKYLENS_CORE_FRAME_MAX_WAIT_MS` | `1500` | 빠진 카메라를 기다리는 최대 시간 |
| `SKYLENS_CORE_FRAME_SOURCE_ID` | `formation` | `FrameKey.source_id` |
| `SKYLENS_CORE_PUSH_QUEUE` | `256` | 모델이 없을 때 쌓아 두는 묶음 수 |
| `SKYLENS_CORE_PUSH_REDECLARE` | `32` | 다시 붙을 때 재선언하는 최근 묶음 수 |
| `SKYLENS_CORE_WHEP_BASE` | `/live/{path}/whep` | 화면에 알리는 WHEP 주소. `{path}`·`{id}`·`{station}` |
| `SKYLENS_CORE_ASSIGNED_HOLD_MS` | `2000` | "태스크 지정 완료"를 보여 주는 시간 |
| `SKYLENS_CORE_DRONE_ETA_SEC` | `10` | 드론 대기 카운트다운. 0이면 표시하지 않는다 |
| `SKYLENS_CORE_STATUS_MS` | `1000` | `server_status` 주기 |
| `SKYLENS_CORE_DETECTIONS_PER_SEGMENT` | `256` | 구간당 보관하는 탐지 청크 상한 |
| `SKYLENS_CORE_TERRAIN` | 켜짐 | `off`면 지형·건물을 내지 않는다 |
| `SKYLENS_CORE_TERRAIN_STYLE` | `black` | `black`·`aerial`·`points` |
| `SKYLENS_CORE_TERRAIN_MARGIN_M` | `500` | 경로 경계 상자에 더하는 여백(m) |
| `SKYLENS_CORE_TERRAIN_CACHE` | 사용자 캐시 폴더 | 지형 타일 캐시 |
| `SKYLENS_CORE_ANCHOR` | 첫 경로의 첫 웨이포인트 | ENU 기준점 `lat,lon[,alt]` |
| `VWORLD_KEY` | 없음 | 직접 주면 키 파일보다 우선한다 |
| `SKYLENS_VWORLD_ENV` | 저장소 부모 폴더의 `.env.vworld` | 키 파일 위치 |

VWorld 키는 저장소 **부모 폴더**의 `.env.vworld`(`VWORLD_KEY=...`, 예시는 루트의 `.env.vworld.example`)에서 읽는다. 저장소 루트는 `go.work`가 있는 폴더로 찾는다. 키는 로그·`/health`·화면 어디에도 나가지 않는다.

## 실행

생성 코드(`gen/`)가 먼저 있어야 한다. `src/shared/proto`에서 `protoc-gen-go`·`protoc-gen-go-grpc`로 `src/skylens_core/gen/`에 만든다.

```bash
cd src/skylens_core
go test ./...
go run .                                   # :8080

# 모델 서버 없이, 프레임 추출 없이
SKYLENS_CORE_FRAME_SOURCE=off go run .

# 대전 MediaMTX의 경로가 drone 하나뿐일 때 (deploy/mediamtx.yml)
SKYLENS_CORE_STREAM_PATH=drone go run .
```

## 구조

| 패키지 | 하는 일 |
|---|---|
| `main.go` | 설정을 읽고 HTTP·gRPC·지형을 배선한다 |
| `internal/config` | 환경변수와 `.env.vworld` |
| `internal/core` | 업링크·뷰어 메시지 처리, 미션·보관·프레임·지형 배선, `/health` |
| `internal/mission` | 미션 상태 기계 |
| `internal/store` | 인메모리 보관과 교체 규칙 |
| `internal/ws` | Envelope WebSocket, 팬아웃·재생 |
| `internal/modelclient` | `SubscribeChunks`·`PushFrames` 재접속 루프 |
| `internal/frames` | ffmpeg 프레임 소스, JPEG 분리, FrameSet 묶기 |
| `internal/backoff` | 재시도 간격 |
| `terrain/` | 지형·건물 청크 (별도 작업) |
| `deploy/` | 대전 MediaMTX·설치 스크립트 |
| `legacy/` | 예전 TypeScript 코어. 참고용, 빌드하지 않는다 |
