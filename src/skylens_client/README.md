# skylens_client

관제탑과 현황판 화면이다. 화면마다 HTML/CSS 한 벌(`ui/`)을 두고, 웹과 데스크톱 네이티브 모두 Blitz가 같은 파일을 읽어 그린다. 3D 뷰는 Blitz 커스텀 위젯(`<object id="view3d">`) 안에서 pointblitz가 점(복원 결과·지형·건물)을 그리고, 메시 블록은 `mesh::MeshPass`가, 드론·경로·탐지 마커는 같은 위젯이 벡터 도형으로 덧그린다.

코어와는 `ws /viewer` 하나로만 통신한다. 모든 프레임은 바이너리 `skylens.viewer.v1.Envelope` 하나다. 연결이 끊기면 0.5초에서 시작해 최대 8초까지 간격을 늘리며 다시 붙고, 다시 붙으면 코어가 보관분을 재생하므로 화면은 그동안 받은 청크를 비우고 새로 받는다.

## 구성

| 파일 | 하는 일 |
|---|---|
| `ui/control.html` · `ui/status.html` | 두 화면의 마크업. 예전 TS가 실행 중에 만들던 패널도 고정 id를 가진 정적 마크업으로 옮겼다 |
| `ui/theme.css` · `ui/control.css` · `ui/status.css` | 디자인 토큰과 패널 스타일 (three.js 시절 스타일을 그대로 옮김) |
| `protocol.rs` | Envelope 인코딩·디코딩, `assign_route` · `manual_control`, 재접속 간격 |
| `state.rs` | 뷰어 이벤트를 화면 상태로 줄이는 리듀서 (미션·드론·경로·카메라·서버 상태·탐지) |
| `chunks.rs` | 청크 교체 규칙. 미리보기는 덧붙이고, 정제본은 구간별로 미리보기를 대체하고, 새 세대가 옛 세대를 대체하고, 메시는 `version`이 더 클 때만 받는다 |
| `views.rs` | 상태를 패널 문구·HTML 조각으로 바꾸는 순수 함수 |
| `app.rs` | Blitz 문서, DOM 갱신, 클릭·키 처리 |
| `view3d.rs` · `camera.rs` · `display.rs` | 3D 위젯, 궤도 카메라, 표시 방식(점 / 검정 텍스처 / 실사 항공뷰) |
| `link.rs` | WebSocket. 네이티브는 tokio-tungstenite, 웹은 브라우저 WebSocket |
| `native.rs` · `web.rs` · `main.rs` | 진입점. 웹과 네이티브가 갈리는 곳은 여기뿐이다 |

## 네이티브 실행

```bash
cargo run -p skylens-client --release -- --screen control --core ws://127.0.0.1:8080/viewer
cargo run -p skylens-client --release -- --screen status  --core 127.0.0.1:8080
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--screen control\|status` | `status` | 관제탑 또는 현황판 |
| `--core URL` | `ws://127.0.0.1:8080/viewer` | `host:port`나 `http://`도 받는다. 경로가 없으면 `/viewer`를 붙인다 |
| `--anchor LAT,LON[,ALT]` | `36.3685,127.3475,30` | ENU 원점. 모델 서버·코어 지형 피드와 같아야 한다 |

폰트는 시스템 폰트를 쓴다(Windows는 맑은 고딕). 렌더링은 vello + wgpu(Vulkan / DX12 / Metal)다.

## 웹 빌드

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version <Cargo.lock의 wasm-bindgen 버전> --locked
bash src/skylens_client/build-web.sh
```

결과는 `src/skylens_client/dist/`에 생긴다(커밋하지 않는다).

| 파일 | 서빙하는 곳 |
|---|---|
| `control.html` | 코어(관제탑) |
| `status.html` · `index.html` | 브리지 게이트웨이(현황판) |
| `pkg/` | wasm과 JS 글루 |
| `fonts/NotoSansKR.ttf` | 한글 폰트. 브라우저는 wasm에 시스템 폰트를 주지 않으므로 앱이 직접 등록한다. 처음 빌드할 때 한 번 내려받아 `web/fonts/`에 둔다(`SKYLENS_FONT_FILE`로 바꿀 수 있다) |

페이지 주소 옵션:

- `?screen=control|status`: 없으면 페이지의 `<meta name="skylens-screen">`를 따른다
- `?core=ws://…/viewer`: 없으면 페이지를 연 호스트의 `/viewer`로 붙는다
- `?anchor=LAT,LON[,ALT]`

wasm은 `file://`로 열리지 않는다. 확인만 할 때는 `python -m http.server -d src/skylens_client/dist 8000`처럼 HTTP로 띄운다. 렌더러는 vello_hybrid이고 WebGPU를 먼저 쓰고 없으면 WebGL2로 내려간다.

## 조작

| 입력 | 동작 |
|---|---|
| 드래그 | 3D 뷰 회전 |
| 오른쪽 드래그 · Shift+드래그 | 이동 |
| 휠 | 확대·축소 |
| F | 선택한 드론(없으면 마지막 탐지)으로 시점 이동 |
| Esc | 열린 창 닫기 |
| 관제탑: 1·2·3 · Tab | 드론 선택 (편대 순서 좌·중앙·우) |
| 관제탑: 화살표 · WASD | 전진·선회 (`manual_control`, 10 Hz) |
| 관제탑: Q / E | 하강 / 상승. 키를 떼면 정지 명령을 한 번 보낸다 |

## 테스트

```bash
cargo test -p skylens-client
```

리듀서·청크 교체 규칙·프로토콜·문구·인자 파싱은 GPU 없이 도는 단위 테스트가 있다.

## 지금 하지 않는 것

- **라이브 영상 재생.** Blitz 안에서는 WHEP 영상을 재생할 수 없다. 카메라 패널은 `camera_feed`가 알려 준 WHEP 주소와 드론 텔레메트리만 보여 준다.
- **지도 클릭 경로 계획.** 예전 경로 계획 창의 타일 지도는 옮기지 않았다. 웨이포인트는 위도·경도·고도를 입력해 추가한다.
- **DJI KMZ 내보내기·클라우드 미션.** `skylens_drone`(Mission Bridge)이 맡는다. 관제탑은 `assign_route`만 보낸다.
- **미니맵.** 현황판의 위에서 본 미니맵은 옮기지 않았다.
