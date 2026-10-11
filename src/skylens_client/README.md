# skylens_client

KOREN 외부망의 지휘관이 보는 **현황판**이다. 이전 TypeScript 현황판의 HTML(`ui/status.html`)과 CSS(`ui/status-panels.css`, 공용 `style.css`)를 고치지 않고 Blitz로 그리며, three.js 대신 pointblitz가 복원 결과를 그린다. 웹(wasm)과 데스크톱 네이티브가 같은 코드를 쓴다. Android/iOS는 나중이다.

화면 뼈대, 코어 연결, 청크 저장소, 3D 위젯, 카메라 부품, 공용 위젯은 `src/shared/viewer`(crate `skylens-viewer`)에 있고, 이 crate는 현황판 고유의 것만 만든다. 관제탑은 `src/skylens_core/control`이다.

## 동작

- 코어의 `ws /viewer` 하나에 읽기 전용으로 붙는다(게이트웨이를 거친다). 코어로는 아무것도 보내지 않는다.
- 처음 오는 `Site`의 앵커를 ENU 원점으로 쓴다. 지형과 건물(관제탑의 바탕 지도)은 받아도 그리지 않는다.
- 복원되어 도착한 것만 보인다. 구간마다 점 청크(cumulus3d, pointblitz chunk v2)를 미리보기, 정제본 순서로 받고, deltamesh 메시 블록과 융합된 탐지를 받는다.
- 다시 연결되면 받은 청크를 비운다. 코어가 보관분을 처음부터 다시 보낸다.

## 이전 현황판과 같은 것

| 부분 | 내용 | 코드 |
|---|---|---|
| 페이지 | `status.html` 템플릿, `status-panels.css`, 로딩 화면 `현황판 · 3D 복원 데이터 로딩`, 화면 이름 칩 | `app.rs` |
| 연결 배지 | `#net-status`, 이제 코어까지 한 구간이라 문구는 `코어 스트림 수신 중` 등 코어 쪽만 | `app.rs` |
| 서버 패널 | 점 상태, 청크·탐지·지연, 임무 문구, 구간별 사다리 행(`ui/DOM.md` §4) | `app.rs`, `panels.rs` |
| 대기 배너 | `#board-waiting`의 대기·끊김·수신·실패 문구 | `panels.rs` |
| 3D 복원 | 구간별 점 구름이 도착하면 1.2초 동안 나타나고, 앞 구간이 0.78까지 나타난 뒤 다음 구간이 시작한다. 메시는 이전 장면의 따뜻한 환경광과 키 라이트로 비춘다. 배경 `#0d0f14`, 거리 안개 | `scene.rs`, `reveal.rs` |
| 탐지 표시 | 원뿔 핀과 깜빡이는 바닥 고리(사람 `#39d98a`, 위험 `#ff4d4d`). 구간의 마지막 수준(정제본)이 와야 보인다 | `markers.rs`, `board.rs` |
| 카메라 | 활성 드론이 찍는 곳을 따라가고, 복원된 띠가 화면에 들어오게 물러나고, 실내 복도면 눈높이로 중심선을 걷는다. 드래그 중과 놓은 뒤 3.5초는 자동 카메라가 멈춘다. SYNCED, FOCUSING, LOCKED, RETURNING(1.8초) | `follow.rs`, `scene.rs` |
| 탐지 카드 | 확신도, `탐지 확인`을 누르면 카메라가 돌아온다 | `card.rs` |
| 미니맵 | `MINIMAP · TOP-DOWN`, 내용에 맞춘 범위, 축척 막대, 구간 원, 금색 경로, 탐지 핀, 드론 | `minimap.rs` |
| 점 패널 | `점 · 고도색` 토글, 높이색 패널(따로 도는 카메라, 더블클릭으로 다시 맞춤, `하위 1% 높이 기준` 범례) | `inset.rs`, `points.rs` |

## 네이티브 실행

```bash
cargo run -p skylens-client --release -- --core ws://127.0.0.1:8080/viewer
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--core URL` | `ws://127.0.0.1:8080/viewer` | `host:port`나 `http://`도 받는다. 경로가 없으면 `/viewer`를 붙인다 |
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
