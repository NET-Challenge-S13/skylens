# skylens-viewer

관제탑(`src/skylens_core/control`, crate `skylens-control`)과 현황판(`src/skylens_client`, crate `skylens-client`)이 같이 쓰는 화면 층이다. 이전 TypeScript의 `shared/viewer/`와 `shared/net/`에 해당한다.

두 화면 crate는 이 crate 위에서 **화면 고유의 것만** 만든다. 패널 문구와 상태, 지형·건물 모습(관제탑), 구간 노출과 추적 카메라 규칙(현황판)이 화면 쪽이고, 문서 뼈대·코어 연결·청크 저장소·3D 위젯·카메라 부품·공용 위젯은 여기 있다.

## 모듈

| 모듈 | 하는 일 |
|---|---|
| `shell` | HTML 템플릿 + 이전 CSS(수정 없이) + 번들 폰트로 Blitz 문서를 만든다. `ScreenApp` 트레이트, `ScreenDocument`, 네이티브 창(`run_native`, 창 아이콘 `res/icon/skylens-256.png`), Android 진입(`run_android`, NativeActivity 위에서 같은 창), wasm 진입(`run_web`) |
| `android` | Android 전용 도우미. 명령줄 대신 설정을 시스템 속성 `debug.skylens.<키>`, 외부·내부 파일 폴더의 `<키>.txt` 순서로 읽고(`setting`), 로그를 logcat 태그 `skylens`로 보낸다. 파일 형식 해석은 모든 플랫폼에서 테스트한다 |
| `dom` | `Dom`: id·class·선택자로 요소를 찾고 텍스트·class·속성·인라인 스타일·자식 마크업을 바꾼다. 바뀌지 않는 쓰기는 건너뛴다 |
| `widgets` | 로딩 화면, 토스트, 패널 라벨(`sl-surface`), 연결 배지, 탐지 카드. 이전 TS가 만들던 DOM 그대로 |
| `link` | 코어와의 WebSocket 하나(`/control` 또는 `/viewer`). 바이너리 `viewer.v1.Envelope`, 재접속(0.5초에서 두 배씩 최대 5초, 네이티브는 연결 5초·무수신 45초 제한), 이전 문구의 연결 상태 |
| `protocol` | Envelope 인코딩, `assign_route` · `manual_control`, payload 디코딩(메시 블록·탐지·지형 타일·건물 묶음) |
| `site` | 코어가 보내는 `Site`. `Site.anchor`가 ENU 원점이고 화면은 원점을 스스로 정하지 않는다 |
| `chunks` | 청크 저장소. 교체 규칙과 종류별 조회, 구간 도착 추적, 탐지 노출 |
| `camera` | 이전 three.js 카메라 동작을 부품으로 옮긴 것과 화면 투영 |
| `view3d` | Blitz 커스텀 위젯. GPU를 갖고 pointblitz로 한 프레임을 그린다 |
| `config` | 이전 `config.ts`의 값(카메라 시간, 재접속 간격, 색, 드론 색) |

## 화면 하나 만들기

```rust
use skylens_viewer::{Dom, Page, PollCx, ScreenApp, ScreenDocument, UiInput, View3d, WindowOptions};
use skylens_viewer::{ChunkStore, CoreLink, SiteFrame, link};
use skylens_viewer::widgets::{NetBadge, Toasts, mount_pane_label};
use std::{cell::RefCell, rc::Rc};

struct StatusApp {
    link: CoreLink,
    store: ChunkStore,
    site: Option<SiteFrame>,
    view: Rc<RefCell<StatusScene>>, // impl ViewScreen
    toasts: Toasts,
    badge: Option<NetBadge>,
}

impl ScreenApp for StatusApp {
    fn mount(&mut self, dom: &mut Dom<'_>) {
        dom.mount_widget("status-view", View3d::boxed(self.view.clone()));
        mount_pane_label(dom, skylens_viewer::q(".pane--status"), "현황판 · 실시간 3D 복원", Some("status"));
        self.badge = NetBadge::mount(dom, "net-status");
    }

    fn poll(&mut self, dom: &mut Dom<'_>, cx: &mut PollCx<'_>) -> bool {
        if let Some(w) = cx.waker { self.link.set_waker(w); }
        for ev in self.link.drain() {
            // Connected: self.store.clear() (코어가 보관분을 다시 보낸다)
            // Message(env): Site -> SiteFrame::from_site, Chunk -> self.store.apply(..) ...
        }
        if let Some(b) = &mut self.badge { b.show_link(dom, &self.link.status); }
        self.toasts.tick(dom, cx.now_ms);
        true // 3D가 움직이는 동안 다시 그린다
    }

    fn input(&mut self, input: &UiInput, dom: &mut Dom<'_>) -> bool { false }
}

let page = Page::legacy(include_str!("ui/status.html"), &[include_str!("ui/status-panels.css")]);
let doc = ScreenDocument::new(&page, StatusApp::new(url));
skylens_viewer::shell::run_native(doc, WindowOptions::new("SkyLens · 현황판"))?;
```

- 템플릿은 이전 앱이 실제로 만든 DOM(`.worktrees/legacy-dom/`)이다. `Page::legacy`가 `fonts.css` · `style.css` 다음에 화면 CSS를 `<head>` 끝에 `<style>`로 넣고, `<script>`와 스타일시트 `<link>`는 뺀다.
- 폰트(Oswald · Barlow · IBM Plex Mono · Noto Sans KR)는 바이너리에 들어 있어 웹과 네이티브가 같은 글꼴을 쓴다. 시스템 폰트는 쓰지 않는다. 일반 글꼴(sans-serif 등)은 Noto Sans KR로 떨어진다.
- 웹은 `shell::run_web(doc, "<canvas id>")`, 코어 주소는 `shell::page_core_url(link::VIEWER_PATH)`(`?core=`가 있으면 그것, 없으면 페이지와 같은 호스트).

## 이벤트

`ScreenApp::input`은 Blitz가 디스패치한 DOM 이벤트(클릭·키·입력·포커스)를 `UiInput`으로 받는다. `path`는 대상부터 조상까지의 요소이고 각 요소의 `id` · `classes` · `data-*`를 담는다.

```rust
if input.closest_class("route-modal__close").is_some() { /* 닫기 */ }
if let Some((_, span)) = input.closest_data("span") { /* data-span="300" */ }
if input.key_down(skylens_viewer::Code::Tab) && !input.in_text_field() { /* 드론 전환 */ }
```

디스패치 중에 기본 동작을 막을지는 `ScreenApp::prevent_default`가 정한다(예: 텍스트 입력 밖의 Tab).

3D 뷰 위의 포인터는 DOM 이벤트가 아니라 `ViewScreen::input`으로 `ViewInput`(뷰 기준 CSS 픽셀)이 간다. 4px 안에서 누르고 뗀 것은 `ViewInput::Click`이 따로 온다(관제탑 지도 클릭 경로 계획).

## DOM 쓰기

```rust
dom.toggle_class("route-overlay", "is-hidden", !open);
dom.set_text("mission-message", &m.message);
dom.set_attr(q(".server-status__dot"), "data-state", "receiving");
dom.set_style(fill, "width", "42%");
dom.set_html("server-ladder", &ladder_markup);   // 같은 마크업이면 건너뜀
let row = dom.append_html("telemetry-list", "<div class=\"hud__drone\">…</div>");
```

문자열 하나는 id, `q("...")`는 CSS 선택자, `NodeId`는 찾아 둔 노드다. 마크업에 넣는 값은 `dom::escape`로 감싼다.

## 공용 위젯

| 위젯 | 이전 코드 | 붙는 곳 |
|---|---|---|
| `LoadingScreen` | `loadingScreen.ts` | `<body>` 끝. `progress` · `done`(0.4초 뒤 제거) · `fail`(2.5초 뒤 사라짐), 매 poll에 `tick` |
| `Toasts` | `toast.ts` | `#toast-root`. `toast kind-person` / `kind-danger`, 3.2초 뒤 제거 |
| `mount_pane_label` · `create_surface` | `surface.ts` | 화면 이름 칩 `sl-surface sl-surface--chip pane-label pane-label--{tone}` |
| `NetBadge` | `relayBadge.ts` | `#net-status`. 점 색(`#39d98a` / `#ffd27f` / `#ff4d4d`)과 `LinkStatus::badge_text` |
| `DetectionCard` | 현황판 `overlay.ts` | `#overlay-status`. 카드와 `.cam-state` 칩, `is_confirm`으로 `탐지 확인` 클릭 판별 |

## 코어 연결

`CoreLink::start(url)`이 백그라운드에서 소켓을 연다(네이티브는 tokio-tungstenite 스레드, 웹은 브라우저 WebSocket). `drain()`이 디코딩된 `LinkEvent`를 돌려주고 그때 `status`도 갱신된다.

- `LinkEvent::Connected`를 받으면 `ChunkStore::clear()`를 부른다. 프록시는 소켓을 옮겨 주지 않으므로 다시 붙으면 코어가 보관분을 처음부터 다시 보낸다.
- `send(body)`는 연결이 없으면 `false`이고 버퍼에 쌓지 않는다. 늦게 도착한 조종 명령이 드론을 움직이면 안 되기 때문이다.
- 문구: 관제탑은 `LinkStatus::detail()`(`코어 연결 시도 중` · `코어 재연결 시도 중` · `코어 연결됨` · `코어 연결 끊김` · `코어에 연결할 수 없습니다`), 현황판 배지는 `badge_text()`(`코어 스트림 수신 중` · `코어 연결됨 · 데이터 대기` · `코어 연결 시도 중` · `코어 응답 없음 · 재연결 대기`).
- 100ms마다 창을 깨우므로 토스트 만료나 10Hz 수동 조종 같은 타이머는 `poll`에서 시각으로 처리한다.

## 좌표

pointblitz 좌표는 ENU 미터 `[동, 북, 상]`(z가 위)다. `SiteFrame::enu(gps)` · `enu_of(&model::Gps)` · `gps(enu)` · `gps_msg(enu)`로 바꾼다. 이전 three.js 장면은 `[동, 상, -북]`이었으므로 이전 오프셋을 옮길 때는 `site::legacy_scene_to_enu`를 쓴다. `Site`가 오기 전에는 원점이 없으니 GPS를 장면에 놓지 않는다.

## 청크 저장소

`ChunkStore::apply(&chunk)`가 `Applied { change, newly_arrived, revealed }`를 돌려준다.

| 종류 | 규칙 | 조회 |
|---|---|---|
| 점 (`POINTS`) | 미리보기는 덧붙임. 정제본은 같은 구간의 미리보기를, 새 세대는 옛 세대를 대체한다. 대체본은 첫 패스 완료(또는 마지막 청크) 플래그가 올 때 한 번에 바뀐다. 바뀔 때마다 구간의 `epoch`가 바뀐다 | `segments()`, `segment(n)` |
| 메시 블록 | 블록 id별로 `version`이 더 클 때만. 삼각형 0개면 지운다(지운 버전보다 옛 것은 되살아나지 않는다) | `mesh_blocks()` |
| 탐지 | id별로 갱신. 구간이 도착해야 보이고, 새로 보이게 된 id가 `revealed`로 한 번씩 온다(토스트·카메라 초점) | `visible_detections()` |
| 지형 타일 · 건물 묶음 | 셀별로 최신. 0이 아닌 더 옛 `version`은 버린다 | `terrain_tiles()`, `building_sets()` |
| 점 슬롯 (`POINT_SLOTS`) | 스트림 하나(`generation` = 스트림 id). 메시지(`version` = 순번)를 순번대로 한 슬롯 상태에 적용하고, 빠진 순번 뒤의 것은 들고 있다가 이어서 적용한다(64개까지. 넘으면 버린다). 같은 순번은 버린다. 더 새 스트림 id는 처음부터(새 `epoch`). 추가 레코드의 구간 바이트로 구간별 점 수·상자·표본을 센다(255는 구간 255 이상: 사건 구간으로 읽는다). 최근 메시지 8개를 두어 3D 뷰가 바뀐 것만 올리고, 더 뒤처지면 접은 상태(`state_message`)로 다시 만든다 | `slots()` |

구간은 점이나 메시 청크가 처음 오거나, 점 슬롯에서 그 구간의 첫 점이 보이면(정제 수준) 도착한 것이다(`arrived()`, `levels()`는 구간별 최고 수준).

## 애니메이션 점 레이어 (`live_slots`)

pointblitz 0.3.0 `SlotScene`은 슬롯당 16바이트에 시간이 없어서 추가는 즉시 나타나고 이동은 점프한다. `LiveSlots`는 크레이트를 고치지 않고 슬롯 장면 둘로 보완한다. 정착한 점은 base 장면, 비행 중인 점(페이드·글라이드)은 컴팩트 id의 overlay 장면에 있고 프레임마다 보간한 위치·색을 슬롯 메시지로 쓴다(페이드 = 배경색으로 가는 색 보간, 불투명 레이어에서 불투명도와 같다). 끝나는 프레임에 base로 옮겨 한 표면만 남는다. `Batch`(removes → adds → moves, 각 레코드가 옛 값을 지님)를 먹고, 항목이 이미 비행 중이면 지금 모습에서 다시 시작한다. 한 번에 `ANIM_BUDGET`(20만) 점까지 움직이고 넘치면 즉시 반영한다.

## 3D 뷰

화면은 `ViewScreen`을 구현해 `Rc<RefCell<_>>`로 갖고, 같은 것을 `View3d::boxed`로 이전 캔버스 id(`control-view` · `status-view`)에 붙인다. 위젯이 장치와 색·깊이 타깃을 갖고, 매 프레임 화면의 `frame`을 부른다.

```rust
impl ViewScreen for StatusScene {
    fn frame(&mut self, gpu: &mut Gpu3d, info: &FrameInfo) -> FrameSpec {
        let store = self.store.borrow();
        let sync = gpu.sync_segments(&store, 0.0);           // 구간마다 pointblitz Scene 하나
        gpu.sync_slots(&store);                             // 점 슬롯 스트림: SlotScene 하나 (바뀐 메시지만, 못 따라가면 접은 상태로)
        // 또는 애니메이션이 필요한 화면(현황판): gpu.live_step(bg, reset, &batches, now_s) 가 LiveSlots 를 먹이고 한 걸음 나아간다
        for s in sync.added { self.fade.start(s, info.now_ms); }
        for (seg, a) in self.fade.alphas(info.now_ms) {
            gpu.clouds.set_alpha(view3d::segment_cloud(seg), a); // 노출 페이드
        }
        gpu.sync_mesh_blocks(&store, |_| self.mesh_material);
        gpu.lines.upsert(&gpu.device, &gpu.queue, ROUTE, &self.route_lines());
        let camera = self.camera_pose(info).camera(50.0, 0.05);
        self.projector = Some(info.projector(&camera));      // DOM 오버레이 위치용
        FrameSpec {
            camera,
            style: FrameStyle { clear: config::clear(config::color::STATUS_BG), fog: Some(self.fog) },
            point_size: None,
        }
    }
}
```

- 한 프레임은 패스 하나다. 불투명 메시, 점(구간별 Scene, Scene별 알파), 반투명 메시, 선 순서로 깊이 버퍼를 같이 쓴다.
- id 공간: `segment_cloud(n)`(점 구름)과 `mesh_block_key(id)`(복원 메시)는 공용 동기화가 쓴다. 화면이 직접 올리는 메시·선·점 구름은 `SEGMENT_CLOUDS`(2^40)와 `MESH_BLOCKS`(2^62) 아래의 id를 쓴다.
- 색: 색 타깃이 `Rgba8Unorm`이라 이전 `0xRRGGBB`는 바이트/255 그대로다(`config::rgb` · `rgba` · `clear`).
- 오버레이를 벡터로 그리려면 `paint_overlay`에서 `info.projector_px(..)`(물리 픽셀)를 쓴다. DOM 요소를 3D 위치에 놓으려면 `info.projector(..)`(CSS 픽셀)로 투영해 `dom.set_style(el, "transform", ..)`.

## 카메라 부품

| 부품 | 이전 동작 |
|---|---|
| `OverviewOrbit` | 관제탑, 드론이 없을 때 작전 구역을 초당 4도로 도는 개요 카메라. `frame_bounds`는 반경 `max(가로, 세로) * 0.62 + 8` |
| `ChaseCamera` | 관제탑, 선택한 드론 뒤 9m·위 4m(× 0.15), 60fps 기준 프레임당 0.06 lerp. 개요로 돌아가면 `reset` |
| `CameraSync` · `Tween` | 현황판 `cameraSync.ts`. SYNCED(궤도) → FOCUSING(1.8초 이징) → LOCKED → RETURNING. `SyncState::label`이 `CAM · SYNCED` 등 |
| `DampedFollow` | 현황판 추적, `1 - exp(-2.5 dt)`로 원하는 자세에 다가감 |
| `UserOrbit` | 현황판 `OrbitControls`(감쇠 0.08, 왼쪽 회전·오른쪽 이동·휠 줌). 끄는 동안과 놓은 뒤 3.5초 동안 `auto_paused`가 참이라 자동 카메라를 멈춘다 |
| `Projector` | ENU를 뷰 픽셀로, 픽셀을 광선으로(`pick_plane`으로 지면 선택) |

이전 lerp는 60fps를 가정했으므로 `camera::per_frame`으로 프레임 시간에 맞춰 같은 움직임을 낸다. 현황판의 추적 자세(지면 교차점, 복원된 꼬리까지 물러나기, 실내 복도 모드)는 현황판 crate가 `statusViewer.ts`를 따라 만든다.
