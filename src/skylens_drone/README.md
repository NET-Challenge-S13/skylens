# skylens_drone — 드론 클라이언트

현장의 드론에서 도는 컴포넌트. 촬영 영상을 **일정 구간마다 잘라 H.265로 게이트웨이에 올리고**, 자기 위치·자세(텔레메트리)를 계속 흘려보내며, 관제탑이 내려보낸 **경로 지정(`AssignRoute`)과 수동 조종(`ManualControl`)을 받는다**.

컴포넌트 경계와 책임의 단일 출처는 [`res/docs/COMPONENTS.md`](../../res/docs/COMPONENTS.md) §3.1이고, 메시지 계약은 [`src/shared/protocol.ts`](../shared/protocol.ts)다. 이 드론은 `DroneHello` · `DroneTelemetry` · `VideoSegment`의 **발신처(origin)** 이며 `AssignRoute` · `ManualControl`의 **수신처**다.

---

## 1. 무엇을 하는가

```
1. 게이트웨이 접속 → DroneHello 로 자기를 알린다
2. 경로가 올 때까지 대기 (idle)
3. AssignRoute 수신 → 현장으로 이동 (transit, 기본 10초)
4. 현장 도착 → 경로 비행 시작, 텔레메트리 연속 송신
5. 경로의 한 구간(slice)을 지날 때마다 VideoSegment 송신
6. loop=true 면 마지막 경유점에서 되돌아 왕복 반복
```

**세그먼트를 자르는 기준은 시계가 아니라 주행거리다.** `odometerM`이 `구간길이 × (n+1)`을 넘을 때 n번째 슬라이스를 낸다. COMPONENTS.md §5.2의 "딜레이 패턴의 트리거는 시간이 아니라 드론의 이동량"이 여기서 만들어진다. 코어는 도착한 `VideoSegment`를 그대로 구간으로 쓰면 된다.

각 `VideoSegment`에는 **그 슬라이스를 덮는 포즈 배열**(`poses`)이 실린다. 복원 단계에서 카메라 위치를 처음부터 역산하지 않아도 되게 하는 값이다.

---

## 2. 두 가지 접속 모드

`SKYLENS_LINK_MODE`로 고른다. 두 모드가 **같은 메시지**를 나르고 반송로만 다르다.

| 모드 | 접속 경로 | 동작 |
|---|---|---|
| `relay` (기본) | `ws://게이트웨이:8081/uplink` | 모든 프레임이 게이트웨이 소켓을 타고 프록시로 릴레이된다. 협상 없음 |
| `webrtc` | `ws://게이트웨이:8081/signal` | 게이트웨이는 **홀펀칭 시그널링만** 중계한다. `signal-hello` → `signal-offer` → `signal-answer`/`signal-ice` → `signal-ready{direct}` 를 거치고 나면 업링크가 `direct` 종단(프록시)으로 옮겨가고, 이후 게이트웨이는 미디어를 한 바이트도 보지 않는다 |

경로(`/uplink`·`/signal`)는 게이트웨이의 `dronePath()`와 맞춰져 있다. 게이트웨이는 다른 경로의 업그레이드를 거절하므로, `--gateway`에 **경로 없이 오리진만** 주면(`ws://10.0.0.4:8081`) 모드에 맞는 경로가 자동으로 붙는다.

**펀치가 실패하면** 드론은 시그널링 소켓으로 미디어를 흘려보내지 않는다(게이트웨이가 어차피 릴레이하지 않는다). 프레임을 버리고 `link.dropped`를 올리며 로그를 남긴다 — 조용히 사라지는 것보다 세어지는 편이 낫다.

### webrtc 모드에서 실제로 협상되는 것

시그널링 교환은 **진짜**다(게이트웨이·프록시와 실제로 주고받는다). 다만 프록시가 Node에서 DataChannel을 종단하지 못해 `signal-ready`로 돌려주는 `direct`는 **협상된 채널을 대신하는 WebSocket**이다(프록시 쪽 문서화된 한계). 그래서 드론이 보내는 offer SDP도 ICE 에이전트가 없는, 형식상 올바른 자리표시자다. 진짜 `RTCPeerConnection`을 넣으면 `link.ts`의 `offerSdp()`와 `openDirect()` 두 메서드만 바뀐다.

---

## 3. 인코딩 — 진짜와 대역(代役)의 경계

| | 무엇이 진짜인가 |
|---|---|
| `LiveCapture` (카메라) | **진짜 H.265.** WebCodecs `VideoEncoder`를 `hev1.1.6.L123.B0`로 설정하고 `isConfigSupported()`로 확인한 뒤 카메라 트랙을 인코딩한다. 2초마다 키프레임을 넣어 슬라이스가 앞 슬라이스 없이도 디코딩되게 한다. HEVC 인코더가 없는 런타임에서는 **시작을 거부한다** — 흉내내지 않는다 |
| `DemoCapture` (데모) | 카메라가 없으므로 미리 촬영한 클립으로 대체한다(COMPONENTS.md §5.1). 파일은 **진짜 HEVC**다 — 아래 참조 |

### 데모 영상이 정말 H.265인 이유

촬영 원본 `res/static/video/*.mp4`는 **H.264 High 10**이다. 계약상 `VideoSegment.codec`은 `'h265'`뿐인데, H.264 파일에 h265 딱지를 붙이면 그 뒤로 측정되는 크기·지연 수치가 전부 **일어나지 않은 일에 대한 측정**이 된다. 그래서 라벨을 고치는 대신 **실제로 한 번 인코딩한다**:

```
res/static/video/*.mp4        H.264 High 10   (촬영 원본, 커밋됨)
res/static/video/h265/*.mp4   HEVC Main 10    (생성물, 실제로 전송되는 것)
```

```bash
npx tsx src/skylens_drone/tools/transcodeDemoFootage.ts
```

`hevc_nvenc`가 있으면 그것을, 없으면 `libx265`를 쓴다. 3840×2160 / 59.94fps / 약 12 Mbps VBR / 2초 GOP — 4K 5G 업링크로 그럴듯한 예산이고, GOP 구조는 `LiveCapture`가 WebCodecs에 요구하는 것과 같다. 5개 클립에 NVENC 기준 약 45초.

`DemoCapture`는 슬라이스마다 `wireCodecOf()`로 클립의 **측정된** 코덱을 확인하고, h265가 아니면 던진다. `VideoSegment.codec`은 `CaptureSource`가 미리 선언해 둔 값이 아니라 **그 바이트를 실제로 만든 쪽이 슬라이스 단위로 돌려준 값**(`SliceResult.codec`)을 복사한다.

**여전히 시뮬레이션인 것**: 인코딩이 촬영과 동시에 일어나지 않고 미리 끝나 있다. 슬라이스가 실제로 몇 초였든 클립 하나가 통째로 대응된다 — 그래서 `durationMs`는 진짜 비행 시간, `bytes`는 진짜 파일 크기지만 **둘의 비율은 실제 순간 비트레이트가 아니다**. 비행 역학·배터리 소모·10초 전개 시간도 모두 모형이다.

> 결과적으로 `src/shared/protocol.ts`는 **바꿀 필요가 없다.** `codec: 'h265'`는 참이다.

---

## 4. 데모 동작 (COMPONENTS.md §5.2)

`SKYLENS_DEMO=1`(또는 `--demo`, 브라우저에서는 `?demo`)일 때:

- 경로가 지정되기 전에는 **정지**. 게이트웨이 소켓은 붙어 있고 `DroneHello`도 보내지만 움직이지 않는다.
- `AssignRoute`를 받으면 기지에서 **현장으로 이동**(`transit`). 기지 위치는 경유점 0에서 `homeOffsetM`(기본 250 m)만큼 진입 반대 방향·25 m 아래로 잡아, 지도에서 실제 접근처럼 보이게 한다.
- **약 10초 뒤 현장 도착** → "드론 연결됨" 로그와 함께 경로 비행 시작.
- `loop: true`면 마지막 경유점에서 되돌아 **왕복 반복**(핑퐁). `foldOdometer()`가 누적 주행거리를 접어서 방향과 랩을 낸다.
- 텔레메트리는 `telemetryHz`(기본 5 Hz)로 계속.
- 구간을 지날 때마다 `VideoSegment`. 클립은 **경로상 위치(좌/중앙/우)와 진행 방향**으로 고른다 — 역방향 구간에는 역방향 촬영본이 나간다(중앙 패스는 역방향 촬영본이 없어 순방향을 재사용).

수동 조종은 `manualIdleReturn`초(기본 2초) 동안 입력이 없으면 자동으로 경로로 복귀한다. 수동 중 이동도 주행거리에 누적되므로 슬라이싱은 "어떤 모드로 날았는지"와 무관하게 "얼마나 날았는지"에만 걸린다.

---

## 5. 실행

### 브라우저 (cargo 불필요)

```bash
npm run dev
# http://localhost:5173/src/skylens_drone/index.html?demo&drone=1
# http://localhost:5173/src/skylens_drone/index.html?demo&mode=webrtc&gateway=ws://10.0.0.4:8081
```

쿼리 플래그는 Node 러너가 읽는 `SKYLENS_*` 환경변수와 **같은 키로 매핑**된다(`core/config.ts`). 운용 패널은 연결 상태·모드·지정 경로·텔레메트리·촬영/전송 상태를 보여주고, `W/S`(전후) `A/D`(회전) `R/F`(상승)로 수동 조종을 넣을 수 있다.

> ⚠️ 프로덕션 빌드에는 `vite.config.ts`의 `rollupOptions.input`에 `drone: 'src/skylens_drone/index.html'` 항목이 필요하다. 개발 서버(`npm run dev`)에서는 그대로 동작한다.

### 헤드리스 (데모 런처가 쓰는 경로)

```bash
npx tsx src/skylens_drone/node/run.ts --demo --gateway=ws://127.0.0.1:8081
```

브라우저와 **같은 `DroneApp`** 이고 `WebSocket`만 `ws`로, 패널만 stdout으로 바뀐다. 데모 영상이 없으면 전사 명령을 알려주고 즉시 종료한다.

### Tauri

```bash
cd src/skylens_drone/src-tauri && cargo tauri dev
```

셸은 **의도적으로 비어 있다**: `#[tauri::command]`도, 상태도, IPC도 없이 브라우저와 같은 페이지를 웹뷰로 연다. 모든 로직이 TypeScript에 있어서 데모 런처는 Rust 툴체인 없이 돈다.

이 저장소에서 `cargo check` 통과를 확인했다(tauri 2.11.5, `x86_64-pc-windows-msvc`). 처음에는 실패했는데 원인은 **`icons/icon.ico` 누락**이었다 — `tauri-build`는 번들링을 꺼도(`bundle.active: false`) Windows 리소스 파일 생성을 위해 이 파일을 요구한다. `icons/`의 아이콘은 자리표시자로 생성한 것이니 실제 배포 전에 교체할 것.

---

## 6. 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `SKYLENS_DRONE_ID` | `1` | 드론 번호 |
| `SKYLENS_DRONE_MODEL` | `SkyLens D1 (H.265 / 5G)` | 패널에 뜨는 기체 문자열 |
| `SKYLENS_LINK_MODE` | `relay` | `relay` \| `webrtc` |
| `SKYLENS_GATEWAY_URL` | `ws://127.0.0.1:8081` | 오리진만 주면 모드별 경로가 붙는다 |
| `SKYLENS_DEMO` | (없음) | 카메라를 데모 영상으로 대체하고 비행을 모의 |
| `SKYLENS_DRONE_TELEMETRY_HZ` | `5` | 텔레메트리 송신 주기 |
| `SKYLENS_DRONE_SPEED` | `12` | 경로 순항 속도 m/s |
| `SKYLENS_DRONE_TRANSIT_SPEED` | `25` | 기지→현장 이동 속도 m/s |
| `SKYLENS_DRONE_ARRIVAL_MS` | `10000` | 경로 지정 후 현장 도착까지 |
| `SKYLENS_DRONE_SLICES` | `4` | 편도 1회당 슬라이스 수 = `VideoSegment` 수 |
| `SKYLENS_DRONE_BATTERY_DRAIN` | `3.5` | 분당 배터리 소모 % |
| `SKYLENS_DRONE_MANUAL_IDLE` | `2.0` | 스틱을 놓고 경로로 복귀하기까지 초 |
| `SKYLENS_DRONE_HOME_OFFSET_M` | `250` | 경유점 0에서 기지까지 거리 |
| `SKYLENS_DRONE_RECONNECT_MIN_MS` / `_MAX_MS` | `500` / `8000` | 재접속 백오프 |
| `SKYLENS_DRONE_AUTOROUTE` | (없음) | 코어 없이 내장 경로를 스스로 지정 |
| `SKYLENS_DRONE_HELLO_ON_ARRIVAL` | (없음) | `DroneHello`를 접속 시점이 아니라 **현장 도착 시점**에 보낸다 |
| `SKYLENS_DRONE_UPLOAD_URL` | (없음) | 브라우저 실촬영 모드에서 슬라이스를 올릴 엔드포인트 |

CLI 별칭도 같다: `--demo --gateway=… --drone=2 --mode=webrtc --hz=10 --slices=6 --arrival=10000 --autoroute`.

> `SKYLENS_DRONE_UPLOAD_URL`이 없는 브라우저 실촬영 모드에서는 슬라이스 uri가 `blob:` 핸들이 되어 **그 페이지 안에서만** 유효하다. 코어는 가져갈 수 없다. 이 경우 로그로 경고한다.

---

## 7. 검증

```bash
npx tsx src/test/drone/scenario.ts --seconds=50            # relay
npx tsx src/test/drone/scenario.ts --seconds=32 --mode=webrtc
```

`src/test/drone/fakeGateway.ts`는 게이트웨이(+뒤의 프록시) 역할을 대신하는 스크립트다. 실제와 같은 종단을 연다 — relay는 `/uplink`, webrtc는 `/signal`과 `/direct` — 그래서 드론 쪽에 특별 분기가 필요 없다. 코어 역할도 겸해서 hello를 받으면 `AssignRoute(loop: true)`를 내리고, 중간에 `ManualControl`을 넣어 인계 경로를 찔러 본다. webrtc 모드에서는 **시그널링 소켓에 미디어가 흘렀는지**를 세어 0인지 확인한다.

`npm test`(Playwright)가 집어가지 않도록 `*.spec.ts`가 아닌 이름을 쓴다. 수십 초 동안 실시간 타이머를 도는 시나리오라 단위 테스트로 두기에 맞지 않는다.

---

## 8. 파일

```
src/skylens_drone/
├─ index.html                  진입 셸 (Tauri 웹뷰도 이 페이지를 연다)
├─ main.ts                     브라우저 진입점 — 설정·캡처 선택·UI 배선
├─ liveCamera.ts               getUserMedia → LiveCapture, 슬라이스 업로드 정책
├─ core/                       런타임 비의존 (DOM 없음, node: 없음)
│  ├─ config.ts                설정 해석 (env · 쿼리 · argv 한 형태로)
│  ├─ drone.ts                 DroneApp — 상태기계·슬라이싱·텔레메트리
│  ├─ flight.ts                경로 기하·왕복 접기·수동 적분 (순수 수학)
│  ├─ link.ts                  GatewayLink — relay/webrtc 두 반송로
│  ├─ capture.ts               CaptureSource — DemoCapture · LiveCapture
│  └─ demoAssets.ts            촬영 대체 클립 목록 (ffprobe 실측값)
├─ ui/                         패널 위젯
│  ├─ panel.ts                 연결·모드·경로·텔레메트리·촬영 5블록
│  ├─ log.ts                   이벤트 로그
│  ├─ sticks.ts                수동 조종 입력
│  ├─ preview.ts               현재 대체 클립 미리보기 (기본 비재생)
│  └─ drone.css
├─ node/
│  └─ run.ts                   헤드리스 러너 (데모 런처 진입점)
├─ tools/
│  └─ transcodeDemoFootage.ts  H.264 원본 → H.265 전사 (1회)
└─ src-tauri/                  얇은 셸 (Cargo.toml · build.rs · src/main.rs · icons/)

src/test/drone/
├─ fakeGateway.ts              게이트웨이+프록시 대역
└─ scenario.ts                 두 모드 전체 시퀀스 검증
```
