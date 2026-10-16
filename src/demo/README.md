# demo: 엔드 투 엔드 데모 런처

`skylens-demo`는 SkyLens 전 컴포넌트를 **실제 코드 그대로** 한 대의 머신에서 띄운다. 런처가 하는 일은 의존 순서대로 띄우고, 데모 설정을 환경변수로 넣고, 로그를 읽을 수 있게 묶는 것뿐이다. 한 대의 머신에서 재현할 수 없는 것만 바꾼다(`docs/COMPONENTS.md` §3.9, `docs/SPEC.md` §11).

| 실제 | 데모 |
|---|---|
| 기체 카메라 → SRT | 실제 촬영본 `data/demo/{left,center,right}.mp4`를 ffmpeg가 실시간 속도로 반복 송출 |
| 실제 비행 · GPS | 런처 안의 드론 시뮬레이터가 촬영본의 비행 기록(`track.json`)을 영상과 맞춰 재생하며 `ws /uplink`로 `drone_hello`·텔레메트리를 보낸다 |
| 운용자의 경로 지정 | 촬영 경로가 관제탑 경로 계획에 미리 들어가 있다. 운용자는 배정만 누른다 |
| 브리지 nginx (게이트웨이) | 런처 안의 현황판 서버(:8090) |
| 대전 프록시 nginx | 없음. 현황판 서버가 코어에 바로 붙는다 |

복원·탐지·메시·배포는 데모에서도 실제 경로다. 영상 → 코어 프레임 추출 → 모델 서버(gRPC) → 점·메시·탐지 청크 → 코어 → 관제탑과 현황판.

## 촬영본

DJI Lito X1(카메라 FC9589) **한 대**가 같은 직선(약 76 m, 방위 약 165°, 남남동)을 짐벌 방향만 바꿔 세 번 날았다. 짐벌 방위는 45°(왼쪽), 165°(앞, 가운데), -75°(오른쪽)이고 피치는 약 -60°다. 방향마다 이륙 지점 기준 약 43~45 m 높이로 한 번, 약 19~22 m로 한 번 찍었다. 그래서 방향마다 클립이 둘이고, 둘을 이어 붙인 것이 한 파일이다.

| 파일 | 내용 |
|---|---|
| `left.mp4` · `center.mp4` · `right.mp4` | 1920x1080 H.264 29.97 fps, B 프레임 없음, 각 약 82초(1차 약 53초 + 2차 약 30초) |
| `track.json` | 방향마다 이어 붙인 영상 시간축(`t`, 초) 위의 비행 기록(약 10 Hz): 위경도, `rel_alt_m`(이륙 지점 기준 높이), 기체·짐벌 자세. 구간(`segments`)마다 원본 클립과 높이 |
| `camera.json` | 카메라 내부 파라미터 추정값(fx·fy 1329 px, cx 960, cy 540 @ 1920x1080). 파일에 기록된 값이 아니라 추정이다 |

이륙 지점은 북위 36.3656, 동경 127.3424 부근(대전 유성, 코어 기본 작전 구역 안)이다.

**어디서 오는가.** 공개 저장소의 GitHub 릴리스 `demo-footage-v1`의 `skylens-demo-footage.zip`이다. `data/demo`에 영상 셋과 `track.json` 중 하나라도 없으면 런처가 시작 전에 내려받는다. 진행률을 MB로 찍고, `data/skylens-demo-footage.zip.part`에 받은 뒤 이름을 바꾸고, zip인지 확인하고, `data/`에 푼 다음(밖을 가리키는 항목이 있으면 아무것도 쓰지 않고 멈춘다) zip을 지운다. 프록시는 `HTTPS_PROXY`·`HTTP_PROXY`를 따른다. 받지 못하면 직접 받는 방법을 한국어로 알려 준다.

```bash
# 직접 받을 때
curl -L -o skylens-demo-footage.zip https://github.com/NET-Challenge-S13/skylens/releases/download/demo-footage-v1/skylens-demo-footage.zip
unzip skylens-demo-footage.zip -d data      # data/demo/... 가 생긴다
```

`data/`는 git에 올리지 않는다.

## 실행

```bash
cargo run -p skylens-demo                  # 브라우저로 볼 때
cargo run -p skylens-demo -- --native      # 네이티브 관제탑·현황판 창도 함께
cargo run -p skylens-demo -- --auto        # 미리 계획된 경로를 런처가 배정 (사람 없이 끝까지)
cargo run -p skylens-demo -- --no-model    # 모델 서버 없이 화면만
cargo run -p skylens-demo -- --synthetic   # 옛 합성 영상과 합성 경로
```

`Ctrl+C` 한 번이면 전체가 순서대로 내려간다(두 번 누르면 즉시 종료). 어느 컴포넌트든 죽으면 런처가 나머지도 정리한다. 한쪽이 사라진 파이프라인을 멀쩡한 것처럼 두면 진단이 더 어렵기 때문이다. 네이티브 창은 예외라서, 창을 닫아도 데모는 계속 돈다.

Windows에서는 런처가 시작하자마자 자기 자신을 **kill-on-close Job Object**에 넣는다. 모든 자식과 손자(코어가 띄우는 ffmpeg 포함)가 그 잡을 물려받으므로, 런처가 어떤 식으로 끝나든 남는 프로세스가 없다. 자식은 각자 프로세스 그룹을 가지므로 콘솔의 `Ctrl+C`는 런처만 받고, 런처가 역순으로 내린다.

### 옵션

| 옵션 | 기본값 | 뜻 |
|---|---|---|
| `--core-port N` | `8080` | 코어 포트(운영과 같다). 관제탑 웹도 여기서 나온다 |
| `--board-port N` | `8090` | 현황판 서버 포트(이전 현황판 포트) |
| `--no-model` | 끔 | 모델 서버를 띄우지 않는다. 코어의 프레임 추출도 끈다 |
| `--model-addr H:P` | 없음 | 이미 떠 있는 모델 서버(예: GPU 노드로 가는 SSH 터널)의 gRPC 주소. 모델 서버를 띄우지 않고, 같은 호스트의 REST(8100) `/health`만 기다린다 |
| `--remote-core H:P` | 없음 | 이미 떠 있는 코어(예: 대전으로 가는 SSH 터널)를 쓴다. MediaMTX·코어·모델·현황판 서버를 띄우지 않고, 영상 송출과 드론 시뮬레이터만 돈다 |
| `--srt-addr H:P` | `127.0.0.1:10890` | 영상을 SRT로 보낼 곳(브리지나 대전 MediaMTX) |
| `--fps X` | `2` | 코어가 드론마다 뽑는 초당 프레임(`SKYLENS_CORE_FRAME_FPS`) |
| `--frame-width N` | `1920` | 뽑는 프레임 가로 폭, `0`이면 원본(`SKYLENS_CORE_FRAME_WIDTH`) |
| `--mediamtx PATH` | `~/tools/mediamtx/mediamtx`, 다음은 `PATH` | MediaMTX 실행 파일 |
| `--native` | 끔 | `target/<profile>/skylens-control`·`skylens-client`도 띄운다 |
| `--auto` | 끔 | 관제탑 대신 런처가 3초 뒤 미리 계획된 경로를 그대로 배정한다 |
| `--follow-route` | 끔 | 드론이 촬영 기록 대신 배정된 경유 지점을 편대로 왕복 비행한다 |
| `--synthetic` | 끔 | 실제 촬영본 대신 옛 `res/static/video/demo_*.mp4`와 합성 경로를 쓴다 |
| `--footage-url URL` | 릴리스 `demo-footage-v1` | 촬영본 zip을 받을 주소 |
| `--no-fetch` | 끔 | 촬영본이 없어도 내려받지 않고, 직접 받는 방법을 알려 주고 멈춘다 |
| `--ground-alt M` | DEM, 안 되면 50 | 이륙 지점 지면 고도(해발 m). 아래 "고도" 참고 |
| `--site SITE` | 촬영본 중심 | 코어의 `SKYLENS_CORE_SITE`에 그대로 넘긴다 |
| `--checkpoint PATH` | `runs/skylens/final` | SkyLensNet 체크포인트 |
| `--core-bin PATH` | `target/skylens-core` | Go 코어 실행 파일 |
| `--profile release\|debug` | 빌드돼 있으면 release | 네이티브 화면의 cargo 프로필 |

## 준비물

런처는 시작 전에 아래를 모두 확인하고, 빠진 것을 한꺼번에 한국어로 알려 준다. 웹 빌드만은 경고로 끝나고 계속 돈다(네이티브 화면은 웹 빌드 없이도 된다).

| 항목 | 위치 | 없을 때 |
|---|---|---|
| 촬영본 | `data/demo/left.mp4` · `center.mp4` · `right.mp4` · `track.json` (`--synthetic`이면 `res/static/video/demo_*.mp4`) | 릴리스에서 자동으로 받는다. 받지 못하거나 `--no-fetch`면 오류 |
| ffmpeg | `PATH` | 오류 |
| MediaMTX | `~/tools/mediamtx/mediamtx(.exe)` 또는 `PATH` 또는 `--mediamtx` | 오류. [릴리스](https://github.com/bluenviron/mediamtx/releases)에서 받아 그 자리에 둔다 |
| Go 코어 | `target/skylens-core(.exe)` | 오류. `cd src/skylens_core && go build -o ../../target/skylens-core.exe .` |
| 모델 서버 | `target/release/skylens-model(.exe)` | 오류. `cargo build --release -p skylens-model` (debug 빌드는 너무 느리다) |
| libtorch DLL | `target/release/build/torch-sys-*/out/libtorch/libtorch/lib` 중 가장 최근 것 | Windows에서 오류. 모델 서버의 `PATH` 앞에 붙인다 |
| 체크포인트 | `runs/skylens/final` | 오류 |
| 관제탑 웹 빌드 | `src/skylens_core/control/dist` | 경고. `bash src/skylens_core/control/build-web.sh` |
| 현황판 웹 빌드 | `src/skylens_client/dist` | 경고. `bash src/skylens_client/build-web.sh` |
| 네이티브 화면 | `target/<profile>/skylens-control` · `skylens-client` | `--native`일 때 오류. `cargo build --release -p skylens-control -p skylens-client` |
| VWorld 키 | 저장소 루트의 `.env.vworld` | 있으면 코어에 `SKYLENS_VWORLD_ENV`로 넘기고, 없으면 코어 기본 위치(저장소 부모 폴더)를 쓴다 |

포트를 이미 누가 쓰고 있으면 시작하지 않는다. 이전 실행의 잔여 프로세스가 준비 확인에 대답해 버리면 옛 코드로 데모가 돌기 때문이다.

## 기동 순서

각 단계는 준비 확인을 통과해야 다음으로 넘어간다.

시작 전에 촬영본을 확인(없으면 내려받기)하고, 이륙 지점 지면 고도를 정하고, 미리 계획된 경로를 `target/demo/planned-route.json`에 쓴다.

```
MediaMTX(:10889 WHEP, :10890 SRT) → 영상 송출 3개 → 모델 서버(:8100, :8101)
→ 코어(:8080) → 현황판 서버(:8090) → 드론 시뮬레이터 → 네이티브 화면(--native)
```

| 단계 | 하는 일 | 준비 확인 |
|---|---|---|
| MediaMTX | `src/skylens_core/deploy/mediamtx.yml`을 복사해 `target/demo/mediamtx.yml`을 만든다. 바꾸는 것은 둘뿐이다: `paths:`를 `all_others`로(배포 설정은 `drone` 하나만 두지만 코어는 `drone{id}`를 읽는다), ICE 후보 인터페이스를 대전 VM의 `ens3` 대신 전부로 | :10889 TCP |
| 영상 송출 | `ffmpeg -re -stream_loop -1 -i data/demo/<station>.mp4 -an -c:v copy -f mpegts srt://127.0.0.1:10890?streamid=publish:drone<id>`. 송출을 띄운 순간이 그 영상 시간축의 0초다 | 1.5초 안에 죽지 않음 |
| 모델 서버 | `SKYLENS_CHECKPOINT`, `SKYLENS_REST_ADDR=127.0.0.1:8100`, `SKYLENS_GRPC_ADDR=127.0.0.1:8101`, `SKYLENS_ANCHOR`(코어와 같은 기준점. 모델은 청크를 이 기준점의 ENU로 낸다). CUDA는 모델 서버가 스스로 올린다 | `GET /health` (최대 300초) |
| 코어 | `SKYLENS_CORE_ADDR=:8080`, `SKYLENS_CORE_WEB_DIR=src/skylens_core/control/dist`, `SKYLENS_CORE_FRAME_SOURCE=ffmpeg`, `SKYLENS_CORE_STREAM_PATH=drone{id}`, `SKYLENS_CORE_DRONE_ETA_SEC=10`, `SKYLENS_CORE_SITE=<위도>,<경도>,<지면 고도>`(촬영본 중심), `SKYLENS_CORE_PLANNED_ROUTE=target/demo/planned-route.json` | `GET /health` |
| 현황판 서버 | 아래 표 | `GET /health` |
| 드론 시뮬레이터 | 드론마다 `ws /uplink` 하나. 관제탑 소켓 하나로 미션 단계를 로그에 남긴다 | 연결 |

드론과 촬영본의 짝: **드론 1 = left, 2 = center, 3 = right.**

## 현황판 서버 (:8090)

운영에서 브리지 nginx(`src/skylens_gateway/nginx-skylens.conf`)가 하는 일을 런처 안에서 axum으로 한다. 데모에 nginx는 필요 없다.

| 경로 | 운영 | 데모 |
|---|---|---|
| `/` | 현황판 웹 빌드(`index status.html`) | `src/skylens_client/dist`, `/`는 `status.html` |
| `/viewer` | WebSocket → 대전 프록시 → 코어 `/viewer` | WebSocket → 코어 `/viewer` |
| `/uplink` | WebSocket → 대전 프록시 → 코어 `/uplink` | WebSocket → 코어 `/uplink` |
| `/live/` | WHEP → 대전 MediaMTX :10889, 세션 `Location`도 `/live/` 아래로 | WHEP → 로컬 MediaMTX :10889, 같은 `Location` 재작성 |
| `/health` | 없음 | `ok` (런처 준비 확인용) |

관제탑은 운영과 같이 코어가 직접 서빙한다(`http://localhost:8080/`).

## 접속 주소

| 화면 | 주소 |
|---|---|
| 관제탑 (웹) | http://localhost:8080/ |
| 현황판 (웹) | http://localhost:8090/ |
| 관제탑 (네이티브) | `skylens-control --core ws://127.0.0.1:8080/control` |
| 현황판 (네이티브) | `skylens-client --core ws://127.0.0.1:8090/viewer` (실제 현황판처럼 현황판 서버를 거친다) |

## 시나리오 (`docs/COMPONENTS.md` §4)

| 단계 | 일어나는 일 |
|---|---|
| 1 | 정지 상태. 드론 3대는 업링크에 붙어 있지만 아무것도 보내지 않는다. 복원 없음. 관제탑 경로 계획에는 촬영 경로가 이미 그려져 있다 |
| 2 | 관제탑에서 경로 계획을 열고 배정을 누른다(`--auto`면 런처가 3초 뒤 같은 경로를 배정). 코어가 경로를 업링크로 내려보낸다 |
| 3 | "태스크 지정 완료" 후 드론 연결 대기. 드론은 10초 동안 현장으로 날아오며 텔레메트리만 보낸다. 도착 지점은 도착 순간 자기 영상이 찍고 있는 자리다 |
| 4 | 도착하면 드론마다 `drone_hello`. 코어가 "드론 연결됨"으로 바꾸고 각 드론의 MediaMTX 경로에서 프레임을 뽑기 시작한다 |
| 5 | 구간마다 미리보기 점군·메시가 먼저 뜨고 정제본이 교체한다 |
| 6 | 위험구역·사람 탐지 결과가 지도 위에 표시된다 |

드론의 "도착"은 `drone_hello`로만 판정한다. 이동 중 텔레메트리를 접속으로 세면 연결 대기 단계가 사라진다. 런처 로그의 `tower` 줄이 미션 단계가 바뀔 때마다 찍힌다.

### 미리 계획된 경로

런처는 center 영상의 1차 구간(높은 비행)에서 경로를 만든다. 시작점과 끝점, 그리고 직선이 아니면 꺾이는 지점(8 m보다 크게 벗어나는 점, Douglas-Peucker)을 남기고, 고도는 1차 구간의 절대 고도(지면 + 1차 구간 높이)다. `loop`는 켠다. 이것을 `target/demo/planned-route.json`에 써서 코어에 `SKYLENS_CORE_PLANNED_ROUTE`로 넘기면, 코어가 관제탑에 `planned_route`로 보내고 관제탑은 비어 있는 경로 계획에 그 지점을 채운다(운용자가 이미 찍은 지점이 있으면 건드리지 않는다). 운용자는 배정만 누르면 되고, `--auto`도 정확히 이 경로를 배정한다. 경로가 한 번 배정되면 코어는 더 이상 보내지 않는다.

### 비행 기록 재생

드론마다 자기 영상의 비행 기록을 그 영상과 맞춰 재생한다(드론 1 = left, 2 = center, 3 = right). 영상 시각은 (지금 - 그 영상 송출 시작) mod 영상 길이이고, 기록의 앞뒤 행 사이를 위경도·고도·방위(360°를 넘는 쪽은 짧은 쪽으로) 모두 선형 보간한다. 0.5초마다 텔레메트리를 보낸다.

| `DroneTelemetry` | 값 |
|---|---|
| `gps.lat` · `gps.lon` | 기록 그대로(보간) |
| `gps.alt` | 이륙 지점 지면 고도 + `rel_alt_m` |
| `heading_deg` | 기체 방위(`aircraft_yaw`). 없으면 1차 구간의 진행 방위 |
| `speed` | 앞뒤 기록 사이 수평 거리 / 시간 |
| `battery_pct` | 비행 시간에 따라 천천히 줄어드는 값(기록에 없다) |
| `t_ms` | 보낸 시각 |

짐벌 자세(`gimbal_pitch` 등)는 담을 자리가 프로토콜에 없어 보내지 않는다. 1차에서 2차로 넘어가는 곳과 영상이 처음으로 돌아가는 곳에서는 **드론도 그대로 건너뛴다**. 영상이 끊기는 곳을 보간으로 메우지 않는다. 세 드론은 사실 같은 기체가 다른 시각에 같은 선을 난 것이라, 화면에서 서로 가깝게 겹쳐 보인다.

도착 전 10초 접근 비행은 1차 구간의 진행 방향 뒤쪽 150 m에서 출발해, 도착 순간 자기 영상이 있는 자리로 들어온다.

### 고도

SkyLens의 `Gps.alt`는 해발 절대 고도다. 코어는 지형을 DEM 높이 - 기준점 고도로 내고, 화면은 드론을 `alt - 기준점 고도`에 놓는다. 그래서 드론 고도는 이륙 지점 지면 고도 + `rel_alt_m`이고, 지면 고도는 다음 순서로 정한다.

1. `--ground-alt M`
2. 코어 지형과 같은 DEM(AWS terrarium 타일, SRTM 약 30 m, zoom 12)을 이륙 지점에서 쌍선형으로 읽은 값. 코어의 지형 캐시(`SKYLENS_CORE_TERRAIN_CACHE`, 없으면 사용자 캐시 폴더의 `skylens/terrain`)에 타일이 있으면 그것을 쓰고, 없으면 받아서 같은 자리에 넣는다. 같은 출처라서 드론이 화면의 지형 위로 정확히 `rel_alt_m`만큼 떠 있다
3. 둘 다 안 되면 추정값 50 m(경고를 찍는다)

### 작전 구역

`--site`가 없으면 코어를 `SKYLENS_CORE_SITE=<위도>,<경도>,<지면 고도>`로 띄운다. 위경도는 세 영상의 모든 기록을 둘러싼 상자의 중심이다. 이 점 형식은 대전 프리셋 크기(약 3 km x 3 km)의 상자를 둘러치므로, 76 m짜리 촬영 경로와 경로 계획 지도(최대 2.5 km)에 충분하다. 기준점 고도가 지면이라 ENU 높이 0이 이륙 지점 지면과 맞는다. 모델 서버에도 같은 기준점을 `SKYLENS_ANCHOR`로 준다. `--site`로 프리셋 이름이나 상자를 주면 런처는 기준점을 알 수 없어 모델 서버가 기본 기준점(대전)을 쓴다(경고를 찍는다).

### 카메라 내부 파라미터

모델 서버는 내부 파라미터를 받지 않는다(`SKYLENS_CAMERAS`는 카메라 이름 목록일 뿐이고, 설정 어디에도 fx·fy·cx·cy를 넣을 자리가 없다). 그래서 `camera.json`은 로그에 추출 프레임 폭 기준으로 환산해 찍기만 한다.

### `--follow-route`와 `--synthetic`

`--follow-route`를 주면 배정된 경유 지점을 따라 편대(좌우 10 m)로 4 m/s로 날고, 배정의 `loop`가 켜져 있으면 왕복한다. 고도는 경유 지점 고도의 평균이다. 화면의 경로 선과 드론 위치가 맞지만, 촬영본의 움직임과 텔레메트리가 어긋난다.

`--synthetic`은 옛 동작이다. `res/static/video/demo_*.mp4`를 송출하고, 편대가 기준점(36.3685, 127.3475)에서 동쪽 30 m, 북쪽 10 m 지점부터 방위 67.5°로 240 m를 고도 90 m, 4 m/s로 날고 끝에서 처음으로 돌아간다. 작전 구역은 코어 기본값(daejeon)이고, 미리 계획된 경로는 이 합성 경로다.

## 로그

모든 줄은 `경과 초 태그 | 내용` 형식이다.

| 태그 | 출처 |
|---|---|
| `launch` | 런처 |
| `fetch` | 촬영본 내려받기·압축 해제 |
| `footage` | 촬영본 요약, 카메라 추정값, 지면 고도 |
| `mtx` | MediaMTX |
| `pub-L` · `pub-C` · `pub-R` | 영상 송출 ffmpeg |
| `model` | 모델 서버 |
| `core` | 코어 |
| `drone` | 드론 시뮬레이터 |
| `tower` | 관제탑 소켓으로 본 미션 단계 |
| `auto` | `--auto` 배정 |
| `client` | 네이티브 현황판 |

## 구조

| 파일 | 하는 일 |
|---|---|
| `main.rs` | 기동 순서, 준비 확인, 종료 |
| `args.rs` | 명령행 |
| `preflight.rs` | 준비물·포트 확인과 안내 문구 |
| `paths.rs` | 편대 표, 영상·송출 주소, 로컬 MediaMTX 설정, 실행 파일 찾기 |
| `procs.rs` | 자식 프로세스, 로그 접두사, Windows Job Object |
| `board.rs` | 현황판 서버(정적 파일, WebSocket 중계, WHEP 프록시) |
| `http1.rs` | 준비 확인과 WHEP 프록시에 쓰는 작은 HTTP/1.0 클라이언트 |
| `fetch.rs` | 촬영본 zip 내려받기, 확인, 안전한 압축 해제 |
| `footage.rs` | 비행 기록(`track.json`) 읽기·보간, 작전 구역 중심, 미리 계획된 경로 |
| `elevation.rs` | 이륙 지점 지면 고도(코어와 같은 DEM, 코어 캐시 공유) |
| `route.rs` | 합성 편대 위치 계산(경로, 기체 간격, 접근 비행). `--synthetic`·`--follow-route` |
| `sim.rs` | 드론 업링크 시뮬레이터(비행 기록 재생 또는 합성 편대), 관제탑 소켓 감시와 `--auto` |
| `legacy/` | 이전 TypeScript 런처. 참고용, 빌드하지 않는다 |
