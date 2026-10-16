# skylens_core (Go)

대전에서 도는 코어 서버. 연산 없이 **보관·스케줄·배포**만 하는 I/O 서버이고, 청크 `payload`는 풀지 않는다. 컴포넌트 경계는 `docs/COMPONENTS.md` §3.4, 설계는 `docs/SPEC.md`가 단일 출처다.

## 무엇이 도는가

| 경로 | 하는 일 |
|---|---|
| `GET /health` | 미션·모델 서버 연결·보관 카운터·프레임·지형·소켓·작전 구역 상태(JSON). VWorld 키는 절대 싣지 않는다 |
| `GET /site` | 작전 구역(JSON). 이전 코어와 같은 `gps`·`source`·`label`에 `viewer.v1.Site`와 같은 값(`anchor`·`bbox`·`min_lat`…·`imagery`·`buildings`·`building_count`)을 더한다 |
| `GET /vworld/wmts/{z}/{y}/{x}.jpeg` | 경로 계획 지도의 위성 타일. 키는 서버에서 붙이고 디스크 캐시를 거친다 |
| `GET /vworld/wfs?...` | VWorld WFS 그대로(키·`domain`은 서버에서 붙이고, 클라이언트가 보낸 키는 버린다). 이전 Vite 프록시와 같은 경로 |
| `ws /uplink` | 드론(또는 데모)이 `drone_hello`·`telemetry`를 올린다. 코어는 `assign_route`·`manual_control`을 내려보낸다 |
| `ws /control` | 관제탑. 모든 것과 바탕 지도(지형·건물)를 받고, 경로가 배정되기 전이면 미리 계획된 경로(`planned_route`)도 받는다. `assign_route`·`manual_control`은 이 소켓으로만 받는다 |
| `ws /viewer` | 현황판(게이트웨이 nginx 경유). 읽기 전용이고 바탕 지도는 보내지 않는다. 이 소켓으로 들어온 메시지는 버리고 로그만 남긴다 |
| `/` | 관제탑 웹 빌드 정적 서빙(`SKYLENS_CORE_WEB_DIR`) |

모든 WebSocket 프레임은 바이너리 `skylens.viewer.v1.Envelope` 하나다. `seq`는 소켓마다 1부터 센다. 두 화면 소켓은 붙자마자 `Site`를 먼저 받고, 이어서 보관분 재생, 그다음 실시간을 받는다.

- **작전 구역(Site):** `SKYLENS_CORE_SITE`가 정하고, 기본은 이전 관제탑의 대전 프리셋(서 127.33, 남 36.355, 동 127.365, 북 36.382)이다. 기준점(ENU 원점)은 프리셋의 기준점(대전은 이전 `CONFIG.geo.anchor`와 같은 36.3685, 127.3475, 30 m)이고 `SKYLENS_CORE_ANCHOR`로 바꿀 수 있다. 기준점은 세션 동안 고정이고 지형·건물·화면이 그리는 모든 것의 원점이다. 경로가 기준점을 옮기지 않는다. `imagery`·`buildings`·`building_count`는 지금 낼 수 있는 것과 보낸 양을 따르고, 값이 바뀌면 상태 주기(`SKYLENS_CORE_STATUS_MS`)마다 두 화면에 `Site`를 다시 보낸다.
- **미리 계획된 경로:** `SKYLENS_CORE_PLANNED_ROUTE`가 JSON 파일을 가리키면 시작할 때 읽어 둔다. 아직 아무 경로도 배정되지 않았으면 관제탑이 붙을 때마다 `Site` 바로 다음에 `planned_route`로 보낸다. 배정이 아니라서 미션 단계도 드론도 그대로이고, 관제탑이 비어 있는 경로 계획에 지점을 채워 운용자는 배정만 누르면 된다. 한 번 배정되면 더는 보내지 않고 배정된 경로를 재생한다. 현황판(`/viewer`)과 드론(`/uplink`)에는 가지 않는다. 데모 런처가 촬영 경로로 이 파일을 만든다.
  ```json
  {"drone_id": 2, "loop": true, "waypoints": [{"lat": 36.3657, "lon": 127.3422, "alt": 95.5}, {"lat": 36.3651, "lon": 127.3424, "alt": 95.5}]}
  ```
  `alt`는 다른 모든 `Gps.alt`처럼 절대 고도(m)다. 지점이 둘 미만이거나 위경도가 범위 밖이면 설정 오류로 시작하지 않는다.
- **미션:** `idle → assigned → awaiting-drone → active`. 경로 지정 뒤 `SKYLENS_CORE_ASSIGNED_HOLD_MS` 동안 "태스크 지정 완료"를 보여 주고 드론 대기로 넘어간다. **드론 도착은 `drone_hello`로만 판정**하고 텔레메트리는 세지 않는다. 마지막 드론의 업링크가 끊기면 다시 대기로 돌아간다.
- **보관(인메모리):** 정제본은 같은 구간의 미리보기와 이전 세대를 대체하고, 메시 블록은 더 높은 `version`만 받고, 탐지 결과는 도착 순서대로 쌓는다(구간당 상한). 지형·건물은 셀마다 하나를 두고, 정밀 셀이 같은 자리의 외곽 링 셀을 대체한다(반대로는 대체하지 않는다). 점 슬롯(`CHUNK_KIND_POINT_SLOTS`)은 합치지 않고 현재 스트림(`generation`)의 메시지를 순번(`version`)대로 모두 둔다. 빠진 순번 뒤의 것은 들고 있다가 빈 곳이 채워지면 순서대로 함께 보내고, 더 새 스트림 id가 오면 이전 스트림을 버린다. 버려진 청크는 화면에도 보내지 않는다. 새 화면은 현재 상태(작전 구역 → 미리 계획된 경로(관제탑만, 배정 전) → 미션 → 링크 → 경로 → 텔레메트리 → 영상 위치 → 지형·건물(관제탑만) → 점 → 점 슬롯 → 메시 → 탐지 → 서버 상태)를 먼저 받고 이어서 실시간을 받는다.
- **모델 서버(gRPC :8101):** `SubscribeChunks`를 늘 열어 두고, 다시 붙을 때 `after_sequence`를 마지막으로 보관한 시퀀스로 준다. `PushFrames`는 모델이 없는 동안 묶음을 큐에 쌓아 두고, 다시 붙으면 최근에 보낸 묶음을 다시 선언한다(이미 받은 묶음의 재선언은 모델 쪽에서 무시된다).
- **프레임:** 드론이 hello하면 대전 MediaMTX의 그 드론 경로를 `ffmpeg`로 읽어 JPEG를 뽑고, 같은 순간의 편대 카메라 프레임을 `FrameSet (source_id, sequence, revision)`으로 묶는다. 늦게 온 카메라가 있으면 같은 시퀀스를 다음 revision으로 다시 보낸다. 프레임 소스는 `frames.Source` 인터페이스라 데모나 테스트가 직접 넣을 수도 있다.
- **영상 위치:** hello마다 `camera_feed`로 WHEP 주소(기본 `/live/drone<id>/whep`)를 알린다. 영상 자체는 MediaMTX가 보낸다.
- **바탕 지도(지형·건물):** 코어가 뜨자마자 작전 구역의 지형·건물을 기준점에서 가까운 셀부터 내보내고 보관한다. 그래서 나중에 붙은 관제탑도 바로 지도를 받는다. 경로가 지정되면 그 영역(여백 `SKYLENS_CORE_TERRAIN_MARGIN_M` 포함)만큼 영역을 넓히고, 이미 보낸 셀은 건너뛰고 새 셀만 첫 웨이포인트에서 가까운 순으로 보낸다. 영역 밖에는 3배 넓이의 성긴 외곽 링이 붙는다. 바탕 지도는 관제탑(`ws /control`)에만 간다.
- **원본 데이터만 보낸다:** 지형 청크(`CHUNK_KIND_TERRAIN`)의 `payload`는 `payload.v1.TerrainTile`이다. 400 m 셀 위의 DEM 높이 격자(정밀 10 m, 링 40 m 간격. 셀 경계마다 꼭짓점이 있어 이웃 셀과 경계 높이를 공유한다. 0행이 남쪽 `min_n`)를 기준점 고도 대비 미터로 싣고, 셀의 위성 영상을 북쪽이 위인 JPEG(정밀 512 px, 링 64 px)로 싣는다. 키가 없거나 위성이 안 되면 영상은 비운다. 건물 청크(`CHUNK_KIND_BUILDINGS`)는 `payload.v1.BuildingSet`이다. 외곽선은 ENU `e,n` 쌍의 반시계 방향 닫힌 고리(첫 꼭짓점을 끝에 한 번 더 둔다), `base`는 외곽선 아래 가장 낮은 DEM 지면 높이(기준점 고도 대비), `height`는 이전 규칙(`height`가 2 m보다 크면 그 값, 아니면 층수 × 3.3 m, 둘 다 없으면 6 m)이다. 색 보정·표시 옵션·수직 과장은 코어가 하지 않고 관제탑이 한다. VWorld 키가 없으면 위성·건물 없이 지형만 낸다.

## 환경변수

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `SKYLENS_CORE_ADDR` | `:8080` | 리슨 주소 |
| `SKYLENS_CORE_WEB_DIR` | `src/skylens_core/control/dist` | 관제탑 웹 빌드. 상대 경로면 작업 폴더, 그다음 저장소 루트 기준 |
| `SKYLENS_MODEL_GRPC` | `127.0.0.1:8101` | 모델 서버 gRPC 주소 |
| `SKYLENS_CORE_FRAME_SOURCE` | `ffmpeg` | `ffmpeg` 또는 `off` |
| `SKYLENS_CORE_FFMPEG` | `ffmpeg` | ffmpeg 실행 파일 |
| `SKYLENS_CORE_STREAM_PATH` | `drone{id}` | 드론의 MediaMTX 경로 이름. `{id}`·`{station}`(left/center/right) |
| `SKYLENS_CORE_FRAME_URL` | `srt://127.0.0.1:10890?streamid=read:{path}` | ffmpeg 입력. 대전 MediaMTX는 RTSP를 끄고 SRT만 열어 두므로 SRT로 읽는다 |
| `SKYLENS_CORE_FRAME_FPS` | `2` | 드론당 초당 추출 프레임 |
| `SKYLENS_CORE_FRAME_QUALITY` | `5` | JPEG 품질(ffmpeg `-q:v`, 2가 최고) |
| `SKYLENS_CORE_FRAME_WIDTH` | `1920` | 추출 프레임 가로 폭(px, 비율 유지). `0`이면 원본 크기 |
| `SKYLENS_CORE_FRAME_MAX_WAIT_MS` | `1500` | 빠진 카메라를 기다리는 최대 시간 |
| `SKYLENS_CORE_FRAME_SOURCE_ID` | `formation` | `FrameKey.source_id` |
| `SKYLENS_CORE_PUSH_QUEUE` | `256` | 모델이 없을 때 쌓아 두는 묶음 수 |
| `SKYLENS_CORE_PUSH_REDECLARE` | `32` | 다시 붙을 때 재선언하는 최근 묶음 수 |
| `SKYLENS_CORE_WHEP_BASE` | `/live/{path}/whep` | 화면에 알리는 WHEP 주소. `{path}`·`{id}`·`{station}` |
| `SKYLENS_CORE_ASSIGNED_HOLD_MS` | `2000` | "태스크 지정 완료"를 보여 주는 시간 |
| `SKYLENS_CORE_DRONE_ETA_SEC` | `10` | 드론 대기 카운트다운. 0이면 표시하지 않는다 |
| `SKYLENS_CORE_STATUS_MS` | `1000` | `server_status` 주기 |
| `SKYLENS_CORE_DETECTIONS_PER_SEGMENT` | `256` | 구간당 보관하는 탐지 청크 상한 |
| `SKYLENS_CORE_SITE` | `daejeon` | 작전 구역. 프리셋 이름(`daejeon`·`seoul`·`uljin`·`uljinup`·`gangneung`), `서,남,동,북` 상자, 또는 이전 형식 `lat,lon[,alt]`(그 점이 기준점이고 대전 프리셋 크기의 상자를 둘러친다) |
| `SKYLENS_CORE_ANCHOR` | 프리셋의 기준점 | ENU 기준점 `lat,lon[,alt]`. 대전은 36.3685, 127.3475, 30 m. 기준점이 따로 없는 프리셋·상자는 상자 중심, 고도 0 |
| `SKYLENS_CORE_PLANNED_ROUTE` | 없음 | 미리 계획된 경로 JSON 파일. 경로가 배정되기 전까지 관제탑에 `planned_route`로 보낸다 |
| `SKYLENS_CORE_TERRAIN` | 켜짐 | `off`면 바탕 지도(지형·건물)를 내지 않는다. `/vworld` 프록시는 그대로 돈다 |
| `SKYLENS_CORE_TERRAIN_MARGIN_M` | `500` | 경로 경계 상자에 더하는 여백(m) |
| `SKYLENS_CORE_TERRAIN_CACHE` | 사용자 캐시 폴더 | 지형 타일·위성 타일·건물 응답 캐시(`/vworld` 프록시도 같이 쓴다) |
| `VWORLD_KEY` | 없음 | 직접 주면 키 파일보다 우선한다 |
| `SKYLENS_VWORLD_ENV` | 저장소 부모 폴더의 `.env.vworld` | 키 파일 위치 |

VWorld 키는 저장소 **부모 폴더**의 `.env.vworld`(`VWORLD_KEY=...`, 예시는 루트의 `.env.vworld.example`)에서 읽는다. 저장소 루트는 `go.work`가 있는 폴더로 찾는다. 키는 로그·`/health`·`/site`·`/vworld` 응답·화면 어디에도 나가지 않고 캐시 경로에도 들어가지 않는다. 키가 없으면 `/vworld`는 503을 준다.

## 실행

생성 코드(`gen/`)가 먼저 있어야 한다. `src/shared/proto`에서 `protoc-gen-go`·`protoc-gen-go-grpc`로 `src/skylens_core/gen/`에 만든다. 지금 생성본은 protoc 36.2, protoc-gen-go v1.36.12, protoc-gen-go-grpc v1.6.2로 만들었다(저장소 루트에서):

```bash
protoc -I src/shared/proto \
  --go_out=src/skylens_core --go_opt=module=github.com/NET-Challenge-S13/skylens/src/skylens_core \
  --go-grpc_out=src/skylens_core --go-grpc_opt=module=github.com/NET-Challenge-S13/skylens/src/skylens_core \
  src/shared/proto/skylens/model/v1/model.proto src/shared/proto/skylens/payload/v1/payload.proto \
  src/shared/proto/skylens/viewer/v1/viewer.proto
```

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
| `internal/config` | 환경변수와 `.env.vworld`, 작전 구역 프리셋 |
| `internal/core` | 업링크·관제탑·현황판 메시지 처리, 미션·보관·프레임·바탕 지도 배선, `/site`·`/health` |
| `internal/mission` | 미션 상태 기계 |
| `internal/store` | 인메모리 보관과 교체 규칙 |
| `internal/ws` | Envelope WebSocket, 팬아웃·재생 |
| `internal/modelclient` | `SubscribeChunks`·`PushFrames` 재접속 루프 |
| `internal/frames` | ffmpeg 프레임 소스, JPEG 분리, FrameSet 묶기 |
| `internal/backoff` | 재시도 간격 |
| `terrain/` | 바탕 지도 청크(`TerrainTile`·`BuildingSet`)와 `/vworld` 프록시 |
| `deploy/` | 대전 MediaMTX·설치 스크립트 |
| `legacy/` | 예전 TypeScript 코어. 참고용, 빌드하지 않는다 |
