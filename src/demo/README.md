# demo: 엔드 투 엔드 데모 런처

`skylens-demo`는 SkyLens 전 컴포넌트를 **실제 코드 그대로** 한 대의 머신에서 띄운다. 런처가 하는 일은 의존 순서대로 띄우고, 데모 설정을 환경변수로 넣고, 로그를 읽을 수 있게 묶는 것뿐이다. 한 대의 머신에서 재현할 수 없는 것만 바꾼다(`docs/COMPONENTS.md` §3.9, `docs/SPEC.md` §11).

| 실제 | 데모 |
|---|---|
| 기체 카메라 → SRT | `res/static/video/demo_{left,center,right}.mp4`를 ffmpeg가 실시간 속도로 반복 송출 |
| 실제 비행 · GPS | 런처 안의 드론 시뮬레이터가 `ws /uplink`로 `drone_hello`·텔레메트리를 보낸다 |
| 브리지 nginx (게이트웨이) | 런처 안의 현황판 서버(:8090) |
| 대전 프록시 nginx | 없음. 현황판 서버가 코어에 바로 붙는다 |

복원·탐지·메시·배포는 데모에서도 실제 경로다. 영상 → 코어 프레임 추출 → 모델 서버(gRPC) → 점·메시·탐지 청크 → 코어 → 관제탑과 현황판.

## 실행

```bash
cargo run -p skylens-demo                  # 브라우저로 볼 때
cargo run -p skylens-demo -- --native      # 네이티브 관제탑·현황판 창도 함께
cargo run -p skylens-demo -- --auto        # 경로를 런처가 배정 (사람 없이 끝까지)
cargo run -p skylens-demo -- --no-model    # 모델 서버 없이 화면만
```

`Ctrl+C` 한 번이면 전체가 순서대로 내려간다(두 번 누르면 즉시 종료). 어느 컴포넌트든 죽으면 런처가 나머지도 정리한다. 한쪽이 사라진 파이프라인을 멀쩡한 것처럼 두면 진단이 더 어렵기 때문이다. 네이티브 창은 예외라서, 창을 닫아도 데모는 계속 돈다.

Windows에서는 런처가 시작하자마자 자기 자신을 **kill-on-close Job Object**에 넣는다. 모든 자식과 손자(코어가 띄우는 ffmpeg 포함)가 그 잡을 물려받으므로, 런처가 어떤 식으로 끝나든 남는 프로세스가 없다. 자식은 각자 프로세스 그룹을 가지므로 콘솔의 `Ctrl+C`는 런처만 받고, 런처가 역순으로 내린다.

### 옵션

| 옵션 | 기본값 | 뜻 |
|---|---|---|
| `--core-port N` | `8080` | 코어 포트(운영과 같다). 관제탑 웹도 여기서 나온다 |
| `--board-port N` | `8090` | 현황판 서버 포트(이전 현황판 포트) |
| `--no-model` | 끔 | 모델 서버를 띄우지 않는다. 코어의 프레임 추출도 끈다 |
| `--fps X` | `2` | 코어가 드론마다 뽑는 초당 프레임(`SKYLENS_CORE_FRAME_FPS`) |
| `--frame-width N` | `1920` | 뽑는 프레임 가로 폭, `0`이면 원본(`SKYLENS_CORE_FRAME_WIDTH`) |
| `--mediamtx PATH` | `~/tools/mediamtx/mediamtx`, 다음은 `PATH` | MediaMTX 실행 파일 |
| `--native` | 끔 | `target/<profile>/skylens-control`·`skylens-client`도 띄운다 |
| `--auto` | 끔 | 관제탑 대신 런처가 3초 뒤 경로를 배정한다 |
| `--follow-route` | 끔 | 드론이 촬영 경로 대신 배정된 경유 지점을 왕복 비행한다 |
| `--checkpoint PATH` | `runs/skylens/final` | SkyLensNet 체크포인트 |
| `--core-bin PATH` | `target/skylens-core` | Go 코어 실행 파일 |
| `--profile release\|debug` | 빌드돼 있으면 release | 네이티브 화면의 cargo 프로필 |

## 준비물

런처는 시작 전에 아래를 모두 확인하고, 빠진 것을 한꺼번에 한국어로 알려 준다. 웹 빌드만은 경고로 끝나고 계속 돈다(네이티브 화면은 웹 빌드 없이도 된다).

| 항목 | 위치 | 없을 때 |
|---|---|---|
| 촬영 영상 | `res/static/video/demo_left.mp4` · `demo_center.mp4` · `demo_right.mp4` | 오류 |
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

```
MediaMTX(:10889 WHEP, :10890 SRT) → 영상 송출 3개 → 모델 서버(:8100, :8101)
→ 코어(:8080) → 현황판 서버(:8090) → 드론 시뮬레이터 → 네이티브 화면(--native)
```

| 단계 | 하는 일 | 준비 확인 |
|---|---|---|
| MediaMTX | `src/skylens_core/deploy/mediamtx.yml`을 복사해 `target/demo/mediamtx.yml`을 만든다. 바꾸는 것은 둘뿐이다: `paths:`를 `all_others`로(배포 설정은 `drone` 하나만 두지만 코어는 `drone{id}`를 읽는다), ICE 후보 인터페이스를 대전 VM의 `ens3` 대신 전부로 | :10889 TCP |
| 영상 송출 | `ffmpeg -re -stream_loop -1 -i demo_<station>.mp4 -an -c:v copy -f mpegts srt://127.0.0.1:10890?streamid=publish:drone<id>` | 1.5초 안에 죽지 않음 |
| 모델 서버 | `SKYLENS_CHECKPOINT`, `SKYLENS_REST_ADDR=127.0.0.1:8100`, `SKYLENS_GRPC_ADDR=127.0.0.1:8101`. CUDA는 모델 서버가 스스로 올린다 | `GET /health` (최대 300초) |
| 코어 | `SKYLENS_CORE_ADDR=:8080`, `SKYLENS_CORE_WEB_DIR=src/skylens_core/control/dist`, `SKYLENS_CORE_FRAME_SOURCE=ffmpeg`, `SKYLENS_CORE_STREAM_PATH=drone{id}`, `SKYLENS_CORE_DRONE_ETA_SEC=10` | `GET /health` |
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
| 1 | 정지 상태. 드론 3대는 업링크에 붙어 있지만 아무것도 보내지 않는다. 복원 없음 |
| 2 | 관제탑에서 경로 계획으로 경유 지점을 지정하고 배정한다(`--auto`면 런처가 3초 뒤 배정). 코어가 경로를 업링크로 내려보낸다 |
| 3 | "태스크 지정 완료" 후 드론 연결 대기. 드론은 10초 동안 현장으로 날아오며 텔레메트리만 보낸다 |
| 4 | 도착하면 드론마다 `drone_hello`. 코어가 "드론 연결됨"으로 바꾸고 각 드론의 MediaMTX 경로에서 프레임을 뽑기 시작한다 |
| 5 | 구간마다 미리보기 점군·메시가 먼저 뜨고 정제본이 교체한다 |
| 6 | 위험구역·사람 탐지 결과가 지도 위에 표시된다 |

드론의 "도착"은 `drone_hello`로만 판정한다. 이동 중 텔레메트리를 접속으로 세면 연결 대기 단계가 사라진다. 런처 로그의 `tower` 줄이 미션 단계가 바뀔 때마다 찍힌다.

### 비행 경로

촬영본에는 GPS가 없으므로 시뮬레이터의 텔레메트리가 파이프라인 전체의 GPS다. 기본으로 편대는 관제탑이 배정한 경로와 무관하게 **촬영본에 맞춘 고정 경로**를 난다.

- 기준점(36.3685, 127.3475, 코어 기본 작전 구역의 ENU 원점)에서 동쪽 30 m, 북쪽 10 m 지점에서 출발해 방위 67.5°(동북동)로 240 m
- 고도 90 m, 4 m/s, 0.5초마다 텔레메트리
- 좌·우 기체는 진행 방향 기준 좌우 10 m
- 끝에 닿으면 처음으로 돌아간다(촬영본이 반복 재생되는 것과 같다)

`--follow-route`를 주면 배정된 경유 지점을 따라 날고, 배정의 `loop`가 켜져 있으면 왕복한다. 화면의 경로 선과 드론 위치가 맞지만, 촬영본의 움직임과 텔레메트리가 어긋날 수 있다.

## 로그

모든 줄은 `경과 초 태그 | 내용` 형식이다.

| 태그 | 출처 |
|---|---|
| `launch` | 런처 |
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
| `route.rs` | 편대 위치 계산(경로, 기체 간격, 접근 비행) |
| `sim.rs` | 드론 업링크 시뮬레이터, 관제탑 소켓 감시와 `--auto` |
| `legacy/` | 이전 TypeScript 런처. 참고용, 빌드하지 않는다 |
