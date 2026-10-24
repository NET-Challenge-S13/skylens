# skylens_client

KOREN 외부망의 지휘관이 보는 **현황판**이다. 이전 TypeScript 현황판의 HTML(`ui/status.html`)과 CSS(`ui/status-panels.css`, 공용 `style.css`)를 고치지 않고 Blitz로 그리며, three.js 대신 pointblitz가 복원 결과를 그린다. 웹(wasm), 데스크톱 네이티브, Android가 같은 코드를 쓴다. iOS는 나중이다.

화면 뼈대, 코어 연결, 청크 저장소, 3D 위젯, 카메라 부품, 공용 위젯은 `src/shared/viewer`(crate `skylens-viewer`)에 있고, 이 crate는 현황판 고유의 것만 만든다. 관제탑은 `src/skylens_core/control`이다.

## 동작

- 코어의 `ws /viewer` 하나에 읽기 전용으로 붙는다(게이트웨이를 거친다). 코어로는 아무것도 보내지 않는다.
- 처음 오는 `Site`의 앵커를 ENU 원점으로 쓴다. 지형과 건물(관제탑의 바탕 지도)은 받아도 그리지 않는다.
- 복원되어 도착한 것만 보인다. 점은 stable display의 점 슬롯(pointblitz 슬롯 차분 메시지, `SlotScene` 하나)으로 받거나, 구간마다 점 청크(pointblitz chunk v2)를 미리보기, 정제본 순서로 받는다(조밀화가 없을 때, 데모 PLY). deltamesh 메시 블록과 융합된 탐지도 받는다.
- 점 슬롯 스트림은 `live_layer`가 애니메이션 점 레이어(`skylens_viewer::LiveSlots`: 정착한 점의 `SlotScene`과 비행 중인 점의 오버레이 `SlotScene` 둘)로 그린다. 추가·삭제는 150 ms 페이드(배경색으로 가는 색 보간), 이동은 700 ms ease-out 글라이드이고 한 번에 20만 점까지 움직인다(넘치면 나머지는 즉시). 구간별 서서히 나타남은 없다(메시만 구간별). 구간 정보(표본 2,000개, 높이 범위, 상자, 점 수, 미니맵 중심)는 추가 레코드의 구간 바이트로 슬롯 상태에서 읽는다. 점 크기는 구간 상자 면적 / 점 수로 잰 간격에서 정한다.
- **이벤트 로그** (`delta_log.rs`, `Board.live`): 현재 실행의 슬롯 메시지를 순번대로 모두 두고 타임라인 행(순번, 구간, REFINED/REPLACE/FINAL, T+, +Δs, 점 수, MOVED/NEW/GONE), `seek`, 이전·다음 단계(역연산 배치로 역애니메이션), 라이브 따라가기/멈춤을 준다. 슬롯 메시지에는 종류와 시계가 없어서 로그가 정한다: 추가만 있으면 REFINED, 이동·삭제가 있으면 REPLACE, 비어 있는 메시지(추가·이동·삭제 0)는 FINAL 표지, T+는 실행의 첫 메시지 도착부터 잰 보드 시계다(늦게 붙으면 재생 이력이 한꺼번에 와서 앞 행들의 간격이 좁다). 스트림 id가 커지면 새 실행(버퍼 비움), 재연결하면 코어가 처음부터 다시 보낸다.
- **라이브 패널** (`panels_live.rs` 규칙, `live_ui.rs` DOM, `ui/live-panels.css`): 대기 화면(SF 파랑 물결 "3D 공간 생성 중"), 우상단 모니터(서버 패널과 같은 머리글 `3D 복원 · 생성 중`, 구역·이벤트·점 수, 마지막 갱신, 아래쪽 물결 띠. 새 이벤트마다 테두리가 잠깐 빛난다), 우상단 툴바의 `설정 · Display` 버튼과 설정 창.
- **설정 창**: 메인 화면 표시(메시가 기본값, 점), 점 표시(점 크기, 색 RGB/구역색/높이색, 점 표시일 때만 켜진다), 전체 보기. 메시 표시는 메시 블록이 하나라도 있을 때만 점을 숨기고, 그때 메시는 불투명하게 그린다.
- **딜레이 패턴** (서버 패널): 가로는 진행 단계(가장 앞선 구역), 세로는 정제 단계(아래가 첫 전달)인 격자다. 구역 k는 k단계에 1행으로 처음 들어오고, 다음 구역이 들어와 정착시키면(REPLACE) 한 칸 위로 올라가 계단 모양이 된다. 같은 구역은 같은 색이고 최근 8단계만 보인다. 슬롯 스트림이 없으면 코어의 구간 수준으로 같은 격자를 그린다.
- 다시 연결되면 받은 청크를 비운다. 코어가 보관분을 처음부터 다시 보낸다.

## 이전 현황판과 같은 것

| 부분 | 내용 | 코드 |
|---|---|---|
| 페이지 | `status.html` 템플릿, `status-panels.css`, 로딩 화면 `현황판 · 3D 복원 데이터 로딩`, 화면 이름 칩 | `app.rs` |
| 연결 배지 | `#net-status`, 이제 코어까지 한 구간이라 문구는 `코어 스트림 수신 중` 등 코어 쪽만 | `app.rs` |
| 서버 패널 | 점 상태, 청크·탐지·지연, 임무 문구, 딜레이 패턴(사다리 행 대신) | `app.rs`, `panels.rs`, `panels_live.rs` |
| 대기 배너 | `#board-waiting`의 대기·끊김·수신·실패 문구 | `panels.rs` |
| 3D 복원 | 구간별 점 구름이 도착하면 1.2초 동안 나타나고, 앞 구간이 0.78까지 나타난 뒤 다음 구간이 시작한다. 메시는 이전 장면의 따뜻한 환경광과 키 라이트로 비춘다. 배경 `#0d0f14`, 거리 안개 | `scene.rs`, `reveal.rs` |
| 탐지 표시 | 원뿔 핀과 깜빡이는 바닥 고리(사람 `#39d98a`, 위험 `#ff4d4d`). 구간의 마지막 수준(정제본)이 와야 보인다 | `markers.rs`, `board.rs` |
| 카메라 | 활성 드론이 찍는 곳을 따라가고, 복원된 띠가 화면에 들어오게 물러나고, 실내 복도면 눈높이로 중심선을 걷는다. 드래그 중과 놓은 뒤 3.5초는 자동 카메라가 멈춘다. SYNCED, FOCUSING, LOCKED, RETURNING(1.8초) | `follow.rs`, `scene.rs` |
| 탐지 카드 | 확신도, `탐지 확인`을 누르면 카메라가 돌아온다. 이전 미니맵 자리(우하단)에 있다 | `card.rs` |
| 미니맵 | 좌하단 `미니맵 · 고도` 토글과 높이색 패널(`하위 1% 높이 기준` 범례). 복원 범위를 중심으로 계속 도는 카메라, 그 위에 탐지 핀과 움직이는 드론. 드래그로 돌리고 휠로 확대하며 더블클릭이면 처음 시점이다. 이전 2D 미니맵(`MINIMAP · TOP-DOWN`)과 그 자동 범위 이동은 없앴다 | `inset.rs`, `points.rs`, `markers.rs` |

## 네이티브 실행

```bash
cargo run -p skylens-client --release -- --core ws://127.0.0.1:8080/viewer
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--core URL` | `wss://175.126.98.44:8443/viewer`, 닿지 않으면 `ws://127.0.0.1:8080/viewer` | `host:port`나 `http://`도 받는다. 경로가 없으면 `/viewer`를 붙인다. 주면 그 주소 하나만 쓴다 |
| `--points on\|off` | `on` | 점 패널을 연 채로 시작할지 |

폰트는 `skylens-viewer`에 들어 있어 웹과 네이티브가 같은 글꼴을 쓴다.

## 웹 빌드

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version <Cargo.lock의 wasm-bindgen 버전> --locked
bash src/skylens_client/build-web.sh
```

결과는 `src/skylens_client/dist/`에 생긴다(커밋하지 않는다). 게이트웨이 nginx가 이 폴더를 그대로 서빙하고 진입 파일은 `status.html`이다.

| 파일 | 내용 |
|---|---|
| `status.html` · `index.html` | 현황판 페이지 |
| `pkg/` | wasm과 JS 글루 |
| `favicon.png` | 아이콘 |

페이지 주소 옵션:

- `?core=ws://…/viewer`: 없으면 페이지를 연 호스트의 `/viewer`로 붙는다
- `?points=off`: 점 패널을 닫은 채로 시작한다

wasm은 `file://`로 열리지 않는다. 확인만 할 때는 `python -m http.server -d src/skylens_client/dist 8000`처럼 HTTP로 띄운다.

## Android

같은 라이브러리(`libstatus_screen.so`)를 NativeActivity APK로 묶는다. 진입점은 `lib.rs`의 `android_main`이고, 창과 이벤트 루프는 데스크톱과 같은 blitz-shell + vello(wgpu, Vulkan)다. HTML·CSS·폰트·아이콘은 라이브러리 안에 들어 있어 APK에 따로 넣는 파일이 없다. 패키지 이름은 `kr.skylens.board`, 앱 이름은 `SkyLens 현황판`, 가로 화면 고정, 최소 API 26(Android 8.0), 대상 API 35다. 설정은 `Cargo.toml`의 `[package.metadata.android]`에 있다.

### 준비

```bash
cargo install cargo-apk
rustup target add aarch64-linux-android x86_64-linux-android
```

- Android SDK(`platforms;android-35`, build-tools, platform-tools)와 NDK(r25 이상, 이 PC는 `28.0.12433566`)
- JDK. cargo-apk가 서명에 `keytool`과 `apksigner`를 쓴다. Android Studio의 `jbr`이나 JetBrains IDE의 `jbr`이면 된다

스크립트가 다음을 스스로 찾는다. 못 찾으면 환경변수로 준다.

| 환경변수 | 기본값 |
|---|---|
| `ANDROID_HOME` | `%LOCALAPPDATA%\Android\Sdk` 또는 `~/Android/Sdk` |
| `ANDROID_NDK_ROOT` | `$ANDROID_HOME/ndk/` 아래 가장 새 버전 |
| `JAVA_HOME` | Android Studio의 `jbr`, 없으면 `%LOCALAPPDATA%\Programs\*\jbr` |
| `PROFILE` | `release`. `dev`는 폰에서 너무 느리다 |
| `CARGO_APK_RELEASE_KEYSTORE` · `CARGO_APK_RELEASE_KEYSTORE_PASSWORD` | SDK 디버그 키(`~/.android/debug.keystore`, 없으면 만든다). adb 설치용이고 스토어 배포용이 아니다 |

### 빌드 · 설치 · 실행

```bash
bash src/skylens_client/build-android.sh              # aarch64 + x86_64를 APK 하나에
bash src/skylens_client/build-android.sh device       # 실기기용(aarch64)만
bash src/skylens_client/build-android.sh emulator --run   # 에뮬레이터용(x86_64), 설치하고 실행하고 로그를 본다
```

결과는 `target/release/apk/skylens-board.apk`다. 손으로 할 때는 다음과 같다.

```bash
adb install -r target/release/apk/skylens-board.apk
adb shell am start -n kr.skylens.board/android.app.NativeActivity
adb logcat -s skylens RustStdoutStderr AndroidRuntime
```

### 코어 주소

명령줄이 없으므로 다시 빌드하지 않고 바꿀 수 있게 아래 순서로 읽는다. 앞의 것이 이긴다. 값은 `--core`와 같이 `host:port`, `http://`, `ws://`를 받고 경로가 없으면 `/viewer`를 붙인다.

| 순서 | 어디 | 바꾸는 법 | 비고 |
|---|---|---|---|
| 1 | 시스템 속성 `debug.skylens.core` · `debug.skylens.points` | `adb shell setprop debug.skylens.core ws://192.168.0.10:8080/viewer` | 재부팅하면 지워진다. 값은 91자까지 |
| 2 | 외부 파일 폴더의 `core.txt` · `points.txt` | `adb push core.txt /sdcard/Android/data/kr.skylens.board/files/core.txt` | 릴리스 빌드에도 된다. 첫 줄(빈 줄과 `#` 주석 제외)만 읽는다 |
| 3 | 내부 파일 폴더의 같은 파일 | `adb shell run-as kr.skylens.board sh -c 'echo ws://… > files/core.txt'` | 디버그(`PROFILE=dev`) 빌드만 |
| 4 | 기본값 | 없음 | 브리지 `wss://175.126.98.44:8443/viewer`를 먼저, 닿지 않으면 `ws://127.0.0.1:8080/viewer`. 점 패널 열림 |

기본값은 매 연결마다 브리지를 먼저 시도하고(연결 제한 5초), 응답이 없을 때만 이 기기의 8080으로 간다. 브리지는 자체 서명 인증서라서, 빌드할 때 `skylens-viewer`의 `build.rs`가 브리지에 TLS로 붙어 그 인증서를 받아 앱에 고정한다(`SKYLENS_BRIDGE_TLS`로 다른 `host:port`, `off`면 고정하지 않음, `SKYLENS_BRIDGE_CERT`로 PEM 파일을 대신 지정). 빌드 때 브리지가 응답하지 않으면 경고만 남기고 고정 없이 빌드되며, 그 앱은 로컬 코어로만 붙는다. 브리지가 인증서를 새로 만들면 다시 빌드한다(`cargo clean -p skylens-viewer`). 로컬 코어를 쓰려면 `adb reverse tcp:8080 tcp:8080`으로 기기의 8080을 개발 PC의 코어(또는 게이트웨이)로 돌린다. 실기기와 에뮬레이터 모두 된다. 에뮬레이터에서는 `ws://10.0.2.2:8080/viewer`도 개발 PC다. 설정을 바꾼 뒤에는 앱을 다시 시작한다(`adb shell am force-stop kr.skylens.board` 후 실행).

### 터치

- 한 손가락은 마우스 왼쪽 버튼처럼 동작한다. 3D 뷰에서 끌면 회전하고, 짧게 누르면 클릭이다(탐지 카드, 점 토글, 패널 버튼).
- 두 손가락을 벌리고 오므리면 확대·축소다. 손가락 거리 16 CSS px마다 휠 한 칸으로 바꾼다(`view3d.rs`의 `Touches`). 두 손가락 이동(팬)은 없다.
- 점 패널의 더블클릭 맞춤은 두 번 탭도 더블클릭으로 들어올 것으로 보지만(Blitz는 탭에서도 클릭을 만든다) 기기에서 확인하지 않았다.

### 알아둘 것

- 화면 배치는 이전 현황판 CSS 그대로라 데스크톱 화면을 기준으로 한다. 태블릿이 맞고, 폰 가로 화면에서는 패널이 좁을 수 있다.
- 앱이 백그라운드로 가면 GPU 표면을 버리고 돌아오면 다시 만든다. 3D 뷰는 받은 청크를 다시 올린다. 코어 연결은 백그라운드에서도 유지된다.
- winit은 한 프로세스에 이벤트 루프를 하나만 허용한다. 그래서 액티비티가 끝나면(뒤로 가기) 프로세스를 끝내고, 회전·화면 크기 변경은 액티비티를 다시 만들지 않고 크기 변경으로 받는다(`config_changes`).
- 에뮬레이터는 Vulkan이 켜진 x86_64 이미지(API 35 google_apis_playstore)와 호스트 GPU(`-gpu host`)가 필요하다. 소프트웨어 렌더러에서는 vello가 돌지 않을 수 있다.
- 로그는 `adb logcat -s skylens`. 패닉도 같은 태그로 남는다.

## 테스트

```bash
cargo test -p skylens-client
```

구간 노출, 카메라 규칙, 패널 문구, 미니맵 범위, 점 청크 읽기, 탐지 공개 규칙은 GPU 없이 도는 단위 테스트가 있다.

## 이전과 다른 것

- 복원 결과가 하나의 최종 스플랫 장면이 아니라 구간별 점 구름과 메시 블록이다. 노출 규칙(도착 순서, 페이드, 앞 구간 기다리기)은 같다.
- 수준이 미리보기와 정제본 둘이다. 사다리는 점 두 개이고 문구는 `수준 1 · 미리보기`, `수준 2 · 정제본`이다. 정제 중이면 CSS가 ` · 정제 중`을 붙인다.
- 처음 시점과 안개 거리는 자리표시 점 구름 대신 처음 도착한 복원 결과의 크기로 정한다.
- `?pick=on` 좌표 측정과 `?render=points`, `?reveal=off` 같은 디버그 옵션은 옮기지 않았다.
