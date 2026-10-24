# 현황판 화면 DOM 변경 명세

이 문서는 `status.html`(정적 템플릿) 위에서 **레거시 TypeScript가 런타임에 바꾸던 DOM**을 빠짐없이 적은 것이다. Rust 화면 코드는 여기에 적힌 대로만 DOM을 바꾼다. 새 클래스·새 요소·새 문구를 만들지 않는다.

## 0. 읽는 법

- **출처**: 레거시 소스는 커밋 `17c4d92`(읽기 전용 체크아웃 `.worktrees/legacy-ui/`). `파일:줄`은 그 체크아웃의 `src/` 기준 경로다.
- **캡처**: `.worktrees/legacy-dom/status-idle.html`, `status-active.html`, `status-active-late.html`. 템플릿은 `status-idle.html`의 `<body>`에서 만들었다.
- **문자열 표기**: 화면에 나가는 문구는 Rust 문자열 리터럴로 적는다. 레거시 문구 안의 긴 대시(U+2014)는 `\u{2014}`로 적는다. Rust 코드에 그대로 붙여 넣으면 같은 글자가 된다.
- **`${...}`**: 레거시 템플릿 문자열의 자리표시자. `toLocaleString('ko-KR')`은 천 단위 쉼표, `toFixed(n)`은 소수 n자리 고정.
- **인라인 스타일 직렬화**: TS가 `'#39d98a'`로 넣은 색은 캡처에 `rgb(57, 217, 138)`로 나온다. 같은 값이다. 캡처와 바이트 단위로 맞추려면 `rgb(...)` 형태로 쓴다.
- **템플릿 초기 상태**: 템플릿은 "중계 서버와 코어 모두 연결됨, 데이터 대기" 상태다(`status-idle`).

## 1. 템플릿에 없는 것, 템플릿과 다른 것

| 항목 | 내용 |
|---|---|
| 3D 캔버스 | `<canvas id="status-view">`. three.js가 붙이던 `data-engine`, `width`, `height`, OrbitControls가 붙이던 `style="touch-action: none;"`을 뺐다. pointblitz 3D 뷰의 마운트 지점이다 |
| 스플랫 라이브러리 DOM | `status-active*` 캡처의 `#status-view` 안쪽에 있는 `.progressBarOuterContainer`, `.spinnerOuterContainer0`, `.infoPanel`과 그 `<style>` 세 개는 gaussian-splats-3d 라이브러리가 넣은 것이다. 레거시 TS가 만든 것이 아니고 전부 `display: none`이다. 재현하지 않는다 |
| 점 패널 `#point-inset` | 점 청크가 처음 올 때 만들어진다(§8). 캡처 세 개 모두 점 청크가 없어 들어 있지 않다 |
| 로딩 화면 | `.loading-screen`은 장면 로드 동안만 있다(§11) |
| 토스트 | `#toast-root`는 비어 있다(§10). 관제탑과 달리 이 화면의 `#toast-root`에는 클래스가 없다(레거시 셸 그대로) |
| `?pick=on` 좌표 표시 | 디버그 전용 고정 박스(§12). 기본 화면에는 없다 |
| HUD | `overlay.ts:29-89`는 `#hud-control`에 드론 목록을 그리지만 현황판 셸에 그 요소가 없어 아무것도 하지 않는다. 재현하지 않는다 |

## 2. 페이지 골격과 고정 요소

레거시 셸(`res/static/status.html`)은 `section.pane.pane--status.pane--full` 안에 `canvas#status-view`, `div#overlay-status.overlay`, `div#net-status.net-badge`, `div#server-status`, `div#minimap`을, `section` 뒤에 `div#toast-root`를 두었다. 나머지는 TS가 만들었다.

| 순서 | 위치 | 만드는 곳 | 결과 |
|---|---|---|---|
| 1 | `body` 끝 | `shared/viewer/ui/loadingScreen.ts:9-30` (`status.ts:63-68`) | `.loading-screen` (로드 후 제거) |
| 2 | `#overlay-status` 안 | `skylens_client/ui/overlay.ts:111-143` (`status.ts:79`) | `.detect-card`, `.cam-state` |
| 3 | `section.pane` 끝 | `shared/viewer/ui/surface.ts:79-87` (`status.ts:274-275`) | 화면 이름 칩 |
| 4 | `#server-status` | `skylens_client/ui/serverStatus.ts:31-60` (`status.ts:277`) | 클래스 추가, 내용 |
| 5 | `#net-status` | `skylens_client/ui/relayBadge.ts:29-45` (`status.ts:278`) | 점과 문구 |
| 6 | `#minimap` | `skylens_client/ui/minimap.ts:36-45` (`status.ts:281-283`) | 클래스 추가, 캔버스 |
| 7 | `#overlay-status` 끝 | `status.ts:387-393` (`status.ts:284`) | `#board-waiting` |
| 8 | `section.pane` 끝 | `skylens_client/ui/pointToggle.ts:8-22` (`status.ts:285`) | `#point-toggle` |
| 9 | `section.pane` 끝 (지연) | `skylens_client/statusview/pointInset.ts:56-93` (`statusview/statusViewer.ts:527-535`) | `#point-inset` |

### 2.1 화면 이름 칩 (변하지 않음)

```html
<div class="sl-surface sl-surface--chip pane-label pane-label--status"><span class="sl-dot"></span><span class="sl-surface__label">현황판 · 실시간 3D 복원</span></div>
```

## 3. 중계 배지 `#net-status`

만드는 곳: `skylens_client/ui/relayBadge.ts:29-45`. 셸의 내용을 `replaceChildren`으로 바꾸고 `net-badge` 클래스를 보장한다.

```html
<div id="net-status" class="net-badge"><span class="net-badge__dot" style="background: rgb(57, 217, 138);"></span><span>코어 연결됨 · 데이터 대기</span></div>
```

중계 상태가 바뀔 때마다(`onStatus`, `relayBadge.ts:40-44`):

- 점의 `background`: 두 구간 중 나쁜 쪽(`severity`, `relayBadge.ts:17-21`)으로 정한다(`relayBadge.ts:23-27`).

  | 상태 | 조건 | 색 (TS 값 / 직렬화) |
  |---|---|---|
  | ok | 중계 서버와 코어 모두 online | `#39d98a` / `rgb(57, 217, 138)` |
  | warn | 끊긴 쪽이 connecting | `#ffd27f` / `rgb(255, 210, 127)` |
  | down | 끊긴 쪽이 offline | `#ff4d4d` / `rgb(255, 77, 77)` |

- 문구 `span`: `"${detail}${boards}"`. `boards` = 같은 중계에 붙은 현황판이 2대 이상이면 `" · 현황판 ${n}대"`, 아니면 빈 문자열.
- `detail`(`skylens_client/sources/relayClient.ts:141-151`, 처음 값은 `relayClient.ts:132`):

  | 중계 서버 | 코어 | 수신 중 | detail |
  |---|---|---|---|
  | connecting | 상관없음 | 상관없음 | `"중계 서버 연결 시도 중"` |
  | offline | 상관없음 | 상관없음 | `"중계 서버 연결 끊김 · 재연결 대기"` |
  | online | online | 예 | `"코어 스트림 수신 중"` |
  | online | online | 아니오 | `"코어 연결됨 · 데이터 대기"` |
  | online | connecting | 상관없음 | `"코어 연결 시도 중"` |
  | online | offline | 상관없음 | `"코어 응답 없음 · 재연결 대기"` |

  "수신 중"은 스플랫 청크, 점 청크, 탐지, 텔레메트리 중 하나라도 받은 뒤부터다. 어느 구간이든 online이 아니게 되면 다시 아니오가 된다(`relayClient.ts:173-243`).

캡처: idle `"코어 연결됨 · 데이터 대기"`, active `"코어 스트림 수신 중"`. 점은 둘 다 `rgb(57, 217, 138)`.

## 4. 서버 패널 `#server-status`

만드는 곳: `skylens_client/ui/serverStatus.ts:31-60`. 셸 요소에 `sl-surface sl-surface--panel server-status` 클래스를 더하고 자식을 붙인다.

### 4.1 구조

```html
<div id="server-status" class="sl-surface sl-surface--panel server-status"><span class="server-status__dot" data-state="connected"></span><div class="server-status__body"><span class="server-status__text">서버 · 연결됨</span><span class="server-status__detail">청크 0 · 탐지 0 · 0ms</span><span class="server-status__detail">코어 연결됨 · 데이터 대기</span></div><div class="server-status__ladder"></div></div>
```

### 4.2 `render(s)` `serverStatus.ts:98-133`

중계 상태가 바뀔 때마다.

1. 패널 상태: 서버가 `receiving`이면 `receiving`, 아니고 `connected`면 `connected`, 둘 다 아니면 `waiting`.
   - `.server-status__dot`의 `data-state` = 패널 상태.
   - `.server-status__text` = `"서버 · 대기 중"` / `"서버 · 연결됨"` / `"서버 · 수신 중"`(`serverStatus.ts:24-28`).
2. 첫 번째 `.server-status__detail` = `"청크 ${chunks} · 탐지 ${detections}"`, 지연값이 있으면 `" · ${round(latencyMs)}ms"`를 덧붙인다.
3. 두 번째 `.server-status__detail` = 패널 상태가 `receiving`이고 임무 상태가 있으면 `"${mission.message} · 드론 ${mission.dronesOnline}대"`, 아니면 §3의 `detail`. 임무 문구는 관제탑과 같은 코어 문구다(`skylens_core/server/mission.ts:16-21`):
   `"대기 중 \u{2014} 관제탑에서 경로를 지정하십시오"`, `"태스크 지정 완료"`, `"드론이 현장에 도착해 연결되면 작업이 시작됩니다"`, `"드론 연결됨 \u{2014} 임무 수행 중"`.
4. 단계 사다리(§4.3).

캡처된 변형:

| 캡처 | data-state | text | detail 1 | detail 2 |
|---|---|---|---|---|
| `status-idle` | `connected` | `"서버 · 연결됨"` | `"청크 0 · 탐지 0 · 0ms"` | `"코어 연결됨 · 데이터 대기"` |
| `status-active` | `receiving` | `"서버 · 수신 중"` | `"청크 3 · 탐지 0 · 1ms"` | `"드론 연결됨 \u{2014} 임무 수행 중 · 드론 3대"` |
| `status-active-late` | `receiving` | `"서버 · 수신 중"` | `"청크 17 · 탐지 1 · 2ms"` | 위와 같음 |

### 4.3 단계 사다리 `.server-status__ladder` `segmentRow`, `serverStatus.ts:65-96,120-132`

코어가 보낸 `segments` 배열을 받은 그대로 그린다. 구간마다 행 하나를 `index` 순으로 처음 볼 때 끝에 붙인다. 구간의 `levels`(점 개수)가 바뀌면 그 행의 내용을 다시 만든다(자리는 그대로).

행 마크업(캡처 `status-active` 그대로):

```html
<div class="segment-row"><span class="segment-row__name">구간 1</span><span class="segment-row__bar"><span class="segment-row__pip" data-on="1"></span><span class="segment-row__pip" data-on="1"></span><span class="segment-row__pip" data-on="0"></span></span><span class="segment-row__note" data-refining="1">수준 2 · 7,000스텝 · 실용 품질</span></div>
```

- `.segment-row__name` = `"구간 ${index + 1}"`.
- `.segment-row__pip`는 `levels`개. `i < level`이면 `data-on="1"`, 아니면 `"0"`.
- `.segment-row__note`:
  - `level == 0`: `"복원 중"`
  - 그 밖: `"수준 ${level} · ${steps.toLocaleString('ko-KR')}스텝"` + 라벨이 있으면 `" · ${label}"`.
  - `data-refining` = `0 < level < levels`면 `"1"`, 아니면 `"0"`.
- 라벨은 코어가 보낸다(`skylens_core/server/ladder.ts:22-38`): 250 = `"형상 윤곽 식별"`, 1000 = `"골격 배치"`, 3500 = `"표면 형성"`, 7000 = `"실용 품질"`, 15000 = `"형상 개수 고정"`, 30000 = `"최종 품질"`. 그 사이 값은 아래쪽 가장 가까운 라벨, 250 미만은 `"개략 형상"`.

캡처된 변형(구간, 켜진 점, note, data-refining):

| 캡처 | 행 |
|---|---|
| `status-idle` | 없음 (`<div class="server-status__ladder"></div>`) |
| `status-active` | 구간 1: 1,1,0 `"수준 2 · 7,000스텝 · 실용 품질"` `1` / 구간 2: 1,0,0 `"수준 1 · 1,000스텝 · 골격 배치"` `1` / 구간 3~6: 0,0,0 `"복원 중"` `0` |
| `status-active-late` | 구간 1: 1,1,1 `"수준 3 · 30,000스텝 · 최종 품질"` `0` / 구간 2~8: 1,1,0 `"수준 2 · 7,000스텝 · 실용 품질"` `1` |

## 5. 미니맵 `#minimap`

만드는 곳: `skylens_client/ui/minimap.ts:36-45`. 셸 요소에 `sl-surface sl-surface--frame` 클래스를 더하고 캔버스를 붙인다.

```html
<div id="minimap" class="sl-surface sl-surface--frame"><canvas class="minimap__canvas" width="198" height="198" style="width: 198px; height: 198px;"></canvas></div>
```

- 캔버스 크기는 TS가 정한다(`resize`, `minimap.ts:130-140`). 처음과 창 크기가 바뀔 때마다: `cssW = host.clientWidth || 160`, `cssH = host.clientHeight || 160`, `dpr = min(2, devicePixelRatio || 1)`, `width = round(cssW*dpr)`, `height = round(cssH*dpr)`, 인라인 `width: ${cssW}px; height: ${cssH}px;`. 템플릿 값 198은 캡처 당시 계산 결과(dpr 1)이며 Rust가 실제 크기로 다시 써야 한다.
- 그 밖의 내용은 전부 캔버스 2D 그림이다(매 프레임, `minimap.ts:144-219`): 테두리, 축척 막대와 `"${n} m"`/`"${n} km"` 글자, 복원된 구간 원, 경로선, 탐지 핀, 드론 점. DOM 변화는 없다.

## 6. 오버레이 `#overlay-status`

레거시에서는 세 자식이 이 순서로 들어 있었다: `.detect-card`, `.cam-state`, `#board-waiting`. 지금은 `.cam-state`를 `#board-toolbar` 맨 앞(설정 버튼과 연결 배지 왼쪽)으로 옮겼다. 레거시 자리는 연결 배지와 같은 우상단이라 배지 밑에 깔려 글자가 겹쳐 보였다.

### 6.1 탐지 카드 `.detect-card` `overlay.ts:111-163`

```html
<div class="detect-card"><div class="detect-card__kicker"></div><div class="detect-card__label"></div><div class="detect-card__conf"></div><div class="detect-card__bar"><span></span></div><button class="detect-card__btn" type="button">탐지 확인</button></div>
```

버튼은 `class` 다음 `type` 순서다(캡처 그대로, 다른 버튼들과 다르다).

- `openCard(id)` (`overlay.ts:145-158`, 카메라가 탐지에 초점을 맞출 때 `detection-focused` 이벤트):
  - 카드 클래스에서 `kind-person`, `kind-danger`를 빼고 `kind-${kind}`를 더한 뒤 `is-open`을 더한다. 결과 `detect-card kind-person is-open` 또는 `detect-card kind-danger is-open`.
  - `.detect-card__kicker` = `person` → `"생존자 감지"`, `danger` → `"위험구역 감지"`, 그 밖 → 빈 문자열(`overlay.ts:23-26`).
  - `.detect-card__label` = 탐지 라벨(코어가 보낸 `label`).
  - `.detect-card__conf` = `"확신도 ${pct}%"`, `pct = round(confidence*100)`.
  - 막대 `span`의 인라인 `width: ${pct}%`.
- `closeCard()` (`overlay.ts:160-163`, 확인 버튼, 카메라 `SYNCED` 복귀, reset): `is-open`만 뺀다. `kind-*`와 문구는 남는다.

캡처 세 개 모두 카드는 닫혀 있고 비어 있다.

### 6.2 카메라 상태 `.cam-state` `refreshCamState`, `overlay.ts:165-172`

매 프레임, 그리고 카메라 상태 이벤트 때. 텍스트가 다를 때만 쓴다.

| 상태 | 텍스트 |
|---|---|
| `SYNCED` | `"CAM · SYNCED"` |
| `FOCUSING` | `"CAM · FOCUSING"` |
| `LOCKED` | `"CAM · LOCKED"` |
| `RETURNING` | `"CAM · RETURNING"` |
| 그 밖 | `"CAM · ${state}"` |

상태 전이는 `statusview/cameraSync.ts:76,83,130,140`. 캡처는 모두 `"CAM · SYNCED"`.

### 6.3 대기 배너 `#board-waiting` `mountWaitingBanner`, `status.ts:387-423`

```html
<div class="splat-loading is-visible" id="board-waiting">코어 연결됨 · 첫 구간 복원 대기 중</div>
```

`class` 다음 `id` 순서다(캡처 그대로). 매 프레임 문구를 계산해 바뀔 때만 쓰고, 문구가 비어 있지 않을 때만 `is-visible`을 켠다.

| 조건 (위에서부터 첫 번째) | 텍스트 |
|---|---|
| 중계 서버나 코어가 online이 아님, 화면에 지오메트리 있음 | `"${detail} \u{2014} 마지막 수신 상태 표시 중"` |
| 같은 조건, 지오메트리 없음 | `"${detail} \u{2014} 복원 데이터 없음"` |
| 지오메트리 없음 | `"코어 연결됨 · 첫 구간 복원 대기 중"` |
| 스플랫 로딩 중 | `"구간 수신 · 렌더 준비 ${round(progress)}%"` |
| 스플랫 로드 오류 | `"구간 로드 실패 \u{2014} 다음 수준 대기"` |
| 그 밖 | 빈 문자열 |

`detail`은 §3의 표. 캡처: idle은 위 마크업, active와 active-late는 `<div class="splat-loading" id="board-waiting"></div>`(빈 텍스트, `is-visible` 없음).

## 7. 점 토글 `#point-toggle`

> 지금은 없다. 토글 버튼은 미니맵 패널(`#point-inset`) 머리의 `#point-inset-fold`(`접기`/`열기`)로 합쳤고, 아래 `meta` 문구는 머리의 `.point-inset__meta`에 쓴다. 패널은 처음부터 있고 접히기만 한다. 아래는 레거시 기록이다.

만드는 곳: `skylens_client/ui/pointToggle.ts:8-22`. `section.pane` 끝, 화면 이름 칩 뒤.

```html
<button type="button" id="point-toggle" class="point-toggle is-on" aria-pressed="true"><span class="point-toggle__label">점 · 고도색</span><span class="point-toggle__meta">수신 대기</span></button>
```

`update()` (`pointToggle.ts:30-50`, 생성 직후와 매 프레임, 클릭 직후):

- 켜짐 여부 `on`(기본 켜짐, URL `?points=off`면 꺼짐, `status.ts:183-185`). 버튼에 `is-on` 토글, `aria-pressed="${on}"`.
- `.point-toggle__meta`:

  | 조건 | 텍스트 |
  |---|---|
  | 받은 점 구간 없음 | `"수신 대기"` |
  | 꺼짐 | `"${delivered}구간 수신"` |
  | 켜짐 | `"${delivered}구간 · 정밀 ${refined} · ${pts}${loading}"` |

  `delivered` = 표시 중이거나 요청됐거나 로딩 중인 구간 수, `refined` = 수준 2 이상이 표시된 구간 수, `pts` = `"${(points/10000).toFixed(0)}만 점"`, `loading` = 로딩 중인 구간이 있으면 `" · 로딩"`.

클릭하면 점 패널을 켜고 끈다(§8).

캡처 세 개 모두 `is-on`, `aria-pressed="true"`, `"수신 대기"`.

## 8. 점 패널 `#point-inset` (지연 생성)

만드는 곳: `skylens_client/statusview/pointInset.ts:56-93`. 첫 점 청크가 올 때 `statusViewer.ts:527-535`가 `#status-view`의 부모(`section.pane`) 끝에 붙인다. 코드에서 옮긴 마크업:

```html
<div class="point-inset is-open" id="point-inset"><div class="point-inset__title">점 · 고도색</div><div class="point-inset__view" title="드래그: 회전 · 휠: 확대 · 더블클릭: 전체 보기"><div class="point-inset__empty">불러오는 중</div></div><div class="point-inset__legend"><span class="point-inset__tick point-inset__tick--high"></span><div class="point-inset__bar" style="background: linear-gradient(to top, rgb(0,0,128) 0%, rgb(0,0,255) 12.5%, rgb(0,128,255) 25%, rgb(0,255,255) 37.5%, rgb(128,255,128) 50%, rgb(255,255,0) 62.5%, rgb(255,128,0) 75%, rgb(255,0,0) 87.5%, rgb(128,0,0) 100%);"></div><span class="point-inset__tick point-inset__tick--low"></span><span class="point-inset__caption">하위 1% 높이 기준</span></div></div>
```

- 막대 그라데이션 값은 `legendGradient()`(`pointInset.ts:20-32`)의 계산 결과다. 브라우저 직렬화 형태(쉼표 뒤 공백 등)는 캡처가 없어 확인하지 못했다.
- `setEnabled(on)` (`pointInset.ts:111-114`, 점 토글): `is-open` 토글.
- 매 프레임 `render` (`pointInset.ts:218-231`):
  - 점 구간이 하나라도 제안되었으면 `has-data` 추가. CSS는 `.point-inset.is-open.has-data`일 때만 보인다.
  - `.point-inset__empty`: 점이 1개 이상이면 인라인 `display: none`, 아니면 인라인 `display` 제거.
  - 높이 범위가 있으면 `--high` = `"+${span.toFixed(1)} m"`, `--low` = `"0 m"`. 없으면 둘 다 빈 문자열.
- 패널 속 점 렌더링은 `.point-inset__view` 사각형 위치에 3D로 그린다. DOM 변화는 없다.

## 9. 숨김과 표시 요약

| 요소 | 클래스 | 처음 |
|---|---|---|
| `.detect-card` | `is-open`일 때 보임 | 닫힘 |
| `#board-waiting` | `is-visible`일 때 보임 | 템플릿은 보임 |
| `#point-inset` | 처음부터 보임. `is-open`이 없으면 머리만 남는다(접힘) | 열림 |

## 10. 토스트 `#toast-root`

만드는 곳: `shared/viewer/ui/toast.ts:10-25`. 끝에 붙이고 3200 ms 뒤 제거한다.

```html
<div class="toast kind-person">${label}</div>
```

현황판의 유일한 토스트는 탐지가 화면에 드러날 때다(`overlay.ts:179-182`, 이벤트는 `statusViewer.ts:477`). 문구 = 탐지 라벨, 종류 = 탐지 종류(`person` → `toast kind-person`, `danger` → `toast kind-danger`).

## 11. 로딩 화면 `.loading-screen`

`shared/viewer/ui/loadingScreen.ts:9-50`. 관제탑과 같은 마크업이고 제목만 `"현황판 · 3D 복원 데이터 로딩"`(`status.ts:63`).

```html
<div class="loading-screen"><div class="loading-screen__box"><div class="loading-screen__title">현황판 · 3D 복원 데이터 로딩</div><div class="loading-screen__bar"><span style="width: 42%;"></span></div><div class="loading-screen__text">42%</div></div></div>
```

`progress(p)`: 막대 `span` 인라인 `width: ${p}%`, 텍스트 `"${round(p)}%"`. `done()`: `is-hidden` 추가, 400 ms 뒤 제거. `fail(msg)`: 제목 `"장면 로드 실패"`, 텍스트 = `msg`, `is-error` 추가, 2500 ms 뒤 제거.

현황판은 `fail`을 쓰지 않는다. 코어가 아직 떠 있지 않거나 연결이 끊겨도 링크가 계속 재시도하므로, 사이트를 받기 전에는 `wait`로 제목 `"코어 연결 대기 중"`, 텍스트 `"<코어 주소> · 시도 N회"`를 보이고 화면을 그대로 둔다. 다음 연결에서 `progress`가 원래 제목을 되돌린다(이전 화면에는 없던 상태다).

## 12. 디버그 좌표 표시 (`?pick=on`)

`status.ts:239-272`. `body` 끝에 클래스 없는 `div`를 붙이고 인라인 스타일을 직접 쓴다:

```
position:fixed;left:50%;bottom:18px;transform:translateX(-50%);z-index:50;background:rgba(10,14,20,0.9);color:#9fe8ff;padding:8px 14px;border-radius:6px;font:12px/1.5 monospace;pointer-events:none;
```

문구: 처음 `"pick=on · 복원 지오메트리를 클릭하면 좌표가 표시됩니다"`, 빗나가면 `"히트 없음 \u{2014} 지오메트리 위를 클릭하세요"`, 맞으면 `"asset [${x, y, z}]  world [${x, y, z}]"`(각 값 `toFixed(2)`, `", "`로 이음, asset이 없으면 `?`, `]` 뒤 공백 두 칸).
