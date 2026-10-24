# 관제탑 화면 DOM 변경 명세

이 문서는 `control.html`(정적 템플릿) 위에서 **레거시 TypeScript가 런타임에 바꾸던 DOM**을 빠짐없이 적은 것이다. Rust 화면 코드는 여기에 적힌 대로만 DOM을 바꾼다. 새 클래스·새 요소·새 문구를 만들지 않는다.

## 0. 읽는 법

- **출처**: 레거시 소스는 커밋 `17c4d92`(읽기 전용 체크아웃 `.worktrees/legacy-ui/`). `파일:줄`은 그 체크아웃의 `src/` 기준 경로다.
- **캡처**: `.worktrees/legacy-dom/control-*.html`. Playwright가 상태별로 직렬화한 런타임 DOM이다. 템플릿은 `control-idle.html`의 `<body>`에서 만들었다.
- **문자열 표기**: 화면에 나가는 문구는 Rust 문자열 리터럴로 적는다. 레거시 문구 안의 긴 대시(U+2014)는 이 문서에서 `\u{2014}`로 적는다. Rust 코드에 그대로 붙여 넣으면 같은 글자가 된다. 이 문서의 HTML 조각에서는 같은 글자를 `&#x2014;`로 적었다. 템플릿 HTML에는 실제 글자가 들어 있다.
- **`${...}`**: 레거시 템플릿 문자열의 자리표시자. 숫자 서식은 함께 적었다(`toFixed(5)` = 소수 5자리 고정, `toLocaleString('ko-KR')` = 천 단위 쉼표).
- **인라인 스타일 직렬화**: TS가 `el.style.background = '#9fe8ff'`로 넣은 값은 캡처에 `style="background: rgb(159, 232, 255);"`로 나온다. 둘은 같은 값이다. Rust는 캡처에 나온 형태(`rgb(...)`)로 쓰면 캡처와 바이트 단위로 같아진다.
- **템플릿 초기 상태**: 템플릿은 "코어 연결됨, 임무 대기, 드론 0대" 상태다(`control-idle`). TS가 막 DOM을 만든 직후의 상태(코어 연결 전)는 각 절의 "생성 직후"에 따로 적었다.

## 1. 템플릿에 없는 것, 템플릿과 다른 것

| 항목 | 내용 |
|---|---|
| 3D 캔버스 | `<canvas id="control-view">`. three.js가 붙이던 `data-engine`, `width`, `height` 속성은 뺐다. 이 요소가 pointblitz 3D 뷰의 마운트 지점이다 |
| 로딩 화면 | `.loading-screen`은 장면을 받는 동안만 `body` 끝에 붙었다가 사라진다(§12). idle 캡처 시점에는 이미 제거되어 템플릿에 없다 |
| 토스트 | `#toast-root`는 비어 있다. 토스트는 3.2초 뒤 스스로 지워진다(§11) |
| `body.is-manual` | 수동 조종 중에만 붙는 클래스(§13) |
| `video` 의 `muted` | `videoPanel.ts:44`는 `video.muted = true`를 **프로퍼티**로 넣는다. 속성이 아니라 캡처에 없다 |
| 모달 입력의 `style=""` | settings/route/cloud 상태 캡처에서 `input`에 붙은 빈 `style=""`은 Playwright 조작의 부산물이다. TS는 넣지 않는다. 재현하지 않는다 |

## 2. 페이지 골격과 고정 요소

`res/static/control.html` 셸(레거시)은 `section.pane.pane--control.pane--full` 안에 빈 마운트 6개만 가졌다. 나머지는 모두 TS가 만들었다. 만든 순서와 위치:

| 순서 | 위치 | 만드는 곳 | 결과 |
|---|---|---|---|
| 1 | `body` 끝 | `shared/viewer/ui/loadingScreen.ts:9-30` (`control.ts:67`) | `.loading-screen` (장면 로드 후 제거) |
| 2 | `#mission-panel-mount` 안 | `skylens_core/ui/missionPanel.ts:37-72` (`control.ts:86-88`) | `.mission-panel` |
| 3 | `#telemetry-panel-mount` 안 | `skylens_core/ui/telemetryPanel.ts:24-36` (`control.ts:89-91`) | `.telemetry-panel` |
| 4 | `section.pane` 끝 | `shared/viewer/ui/surface.ts:79-87` (`control.ts:94`) | 화면 이름 칩 |
| 5 | `#video-panel-mount` 안 | `skylens_core/control/videoPanel.ts:32-55` (`control.ts:96`) | `.video-panel` |
| 6 | `body` 끝 | `skylens_core/control/routeModal.ts:203-356` (`control.ts:171`) | 첫 번째 `.route-modal-overlay` (경로 계획) |
| 7 | `body` 끝 | `skylens_core/ui/settingsPanel.ts:42-137` (`control.ts:252`) | `.settings-overlay` |
| 8 | `body` 끝 | `skylens_core/control/cloudModal.ts:14-48` (`control.ts:261`) | 두 번째 `.route-modal-overlay` (클라우드 미션) |
| 9 | `#control-toolbar` 안 | `control.ts:55-62`, `control.ts:263-275` | 툴바 버튼 3개 |

### 2.1 화면 이름 칩 (변하지 않음)

```html
<div class="sl-surface sl-surface--chip pane-label pane-label--control"><span class="sl-dot"></span><span class="sl-surface__label">관제탑 · 실지형 상황도</span></div>
```

`createSurface`(`surface.ts:39-72`)가 `variant: 'chip'`, `dot: true`로 만든다. 런타임에 다시 바뀌지 않는다.

### 2.2 툴바 (변하지 않음)

```html
<div id="control-toolbar" class="control-toolbar"><button type="button" class="control-toolbar__btn">경로 계획 · Route</button><button type="button" class="control-toolbar__btn">클라우드 미션 · Cloud</button><button type="button" class="control-toolbar__btn">설정 · Display</button></div>
```

버튼 동작(`control.ts:266-273`):

| 버튼 | 동작 |
|---|---|
| 1번 `경로 계획 · Route` | 브리지 상태를 물어 `routeModal.setCloudAvailable(h.cloudConfigured)`(§7.6), 곧바로 `routeModal.open()`(§7.7) |
| 2번 `클라우드 미션 · Cloud` | `cloudModal.open()`(§8.4) |
| 3번 `설정 · Display` | `settingsPanel.toggle()`(§9.4) |

DOM 자체는 바뀌지 않는다.

## 3. 임무 패널 `#mission-panel-mount > .mission-panel`

만드는 곳: `skylens_core/ui/missionPanel.ts:37-72`.

### 3.1 구조

```html
<div class="sl-surface sl-surface--panel mission-panel">
  <div class="mission-panel__link is-connected"><span class="mission-panel__dot"></span><span class="mission-panel__link-text">코어 연결됨</span></div>
  <ol class="mission-panel__steps">
    <li class="mission-panel__step is-active">대기</li>
    <li class="mission-panel__step">태스크 지정</li>
    <li class="mission-panel__step">드론 연결 대기</li>
    <li class="mission-panel__step">임무 수행</li>
  </ol>
  <div class="mission-panel__message">대기 중 &#x2014; 관제탑에서 경로를 지정하십시오</div>
  <div class="mission-panel__sub">드론 0대 연결됨</div>
</div>
```

(보기 좋게 줄을 나눴다. 실제 템플릿은 태그 사이 공백 없이 한 줄이다.)

**생성 직후**(코어 연결 전, `missionPanel.ts:43,48,66`): 링크 행 클래스는 `mission-panel__link is-disconnected`, 링크 문구 `"코어 연결 대기 중"`, 단계 `li`에 상태 클래스 없음, 메시지는 `FALLBACK_MESSAGE.idle`, `.mission-panel__sub`는 빈 문자열. 곧바로 `setLink('connecting', ...)`가 와서 §3.3대로 바뀐다.

### 3.2 `setMission(status)` `missionPanel.ts:95-119`

코어의 `mission-status` 메시지가 올 때마다 호출된다(`control.ts:139`).

1. `.mission-panel__message` 텍스트 = `status.message`, 비어 있으면 `FALLBACK_MESSAGE[phase]`(`missionPanel.ts:24-29`). 코어가 보내는 문구(`skylens_core/server/mission.ts:16-21`)와 대체 문구는 같다.

   | phase | 문구 |
   |---|---|
   | `idle` | `"대기 중 \u{2014} 관제탑에서 경로를 지정하십시오"` |
   | `assigned` | `"태스크 지정 완료"` |
   | `awaiting-drone` | `"드론이 현장에 도착해 연결되면 작업이 시작됩니다"` |
   | `active` | `"드론 연결됨 \u{2014} 임무 수행 중"` |

2. 단계 `li` 네 개(`PHASES`, `missionPanel.ts:17-22`: `idle`, `assigned`, `awaiting-drone`, `active` 순서): 현재 phase의 인덱스를 `k`라 하면 `i < k`인 `li`에 `is-done`, `i == k`에 `is-active`, 나머지는 둘 다 없음. `classList.toggle`이라 클래스 순서는 `mission-panel__step is-done` / `mission-panel__step is-active`.

3. `.mission-panel__sub` 텍스트(`renderSub`, `missionPanel.ts:79-87`): 조각을 `" · "`로 잇는다.
   - phase가 `awaiting-drone`이고 `etaSeconds`가 있으면 첫 조각 `"예상 ${left}초"`. `left = max(0, round((deadline - now)/1000))`, `deadline = 수신 시각 + etaSeconds*1000`. 이 동안 500 ms마다 다시 그린다(`missionPanel.ts:112-114`).
   - 항상 마지막 조각 `"드론 ${dronesOnline}대 연결됨"`.

캡처된 변형:

| 캡처 | 단계 클래스 (대기, 태스크 지정, 드론 연결 대기, 임무 수행) | message | sub |
|---|---|---|---|
| `control-idle` | `is-active`, 없음, 없음, 없음 | `"대기 중 \u{2014} 관제탑에서 경로를 지정하십시오"` | `"드론 0대 연결됨"` |
| `control-assigned` | `is-done`, `is-done`, `is-active`, 없음 | `"드론이 현장에 도착해 연결되면 작업이 시작됩니다"` | `"예상 8초 · 드론 0대 연결됨"` |
| `control-active`, `control-active-late` | `is-done`, `is-done`, `is-done`, `is-active` | `"드론 연결됨 \u{2014} 임무 수행 중"` | `"드론 3대 연결됨"` |

### 3.3 `setLink(state, detail)` `missionPanel.ts:121-145`

코어 링크 상태가 바뀔 때마다(`control.ts:119`).

- `.mission-panel__link`: `is-connected`, `is-connecting`, `is-disconnected` 중 state에 맞는 하나만 켠다(`toggle` 세 번, 이 순서). 생성 때 클래스가 `is-disconnected`였으므로 한 번 연결된 뒤의 클래스 문자열은 `mission-panel__link is-connected`(캡처와 같음), 그 뒤 끊기면 `mission-panel__link is-disconnected`.
- `.mission-panel__link-text` 텍스트 = `detail`. 값은 `skylens_core/coreLink.ts`가 정한다:

  | state | detail | 출처 |
  |---|---|---|
  | connecting | `"코어 연결 시도 중"` (처음), `"코어 재연결 시도 중"` (한 번이라도 연결된 뒤) | `coreLink.ts:145` |
  | connected | `"코어 연결됨"` | `coreLink.ts:162` |
  | disconnected | `"코어 연결 끊김"` | `coreLink.ts:184` |
  | disconnected | `"코어에 연결할 수 없습니다"` (첫 연결 실패) | `coreLink.ts:188` |
  | disconnected | `"코어 주소가 올바르지 않습니다 (${err})"` | `coreLink.ts:152` |

- 루트 `.mission-panel`: state가 `connected`가 아니면 `is-stale` 추가, 맞으면 제거. 클래스 문자열은 `sl-surface sl-surface--panel mission-panel is-stale`.
- state가 `connected`가 아니면 ETA 타이머를 멈추고 문구를 덮어쓴다:

  | state | `.mission-panel__message` | `.mission-panel__sub` |
  |---|---|---|
  | connecting | `"코어에 연결하는 중입니다"` | `"연결되면 임무 상태와 드론 위치가 표시됩니다"` |
  | disconnected | `"코어와 연결되어 있지 않습니다 \u{2014} 표시된 임무 상태는 최신이 아닐 수 있습니다"` | `"연결이 복구될 때까지 드론 위치와 임무 상태를 갱신할 수 없습니다"` |

  단계 `li`의 클래스는 건드리지 않는다(마지막 상태 유지).
- state가 `connected`면 `.mission-panel__sub`만 §3.2의 `renderSub` 규칙으로 다시 쓴다. 메시지는 다음 `setMission`이 올 때까지 바꾸지 않는다.

## 4. 텔레메트리 패널 `#telemetry-panel-mount > .telemetry-panel`

만드는 곳: `skylens_core/ui/telemetryPanel.ts:24-36`.

### 4.1 구조

```html
<div class="sl-surface sl-surface--panel telemetry-panel"><div class="telemetry-panel__empty">연결된 드론이 없습니다</div><ul class="telemetry-panel__list"></ul></div>
```

### 4.2 `render(drones)` `telemetryPanel.ts:96-132`

200 ms마다(`control.ts:291,306-309`), 코어 연결이 끊기면 빈 목록으로 한 번(`control.ts:124`) 호출된다.

1. `.telemetry-panel__empty`: 드론이 1대 이상이면 `is-hidden` 추가, 0대면 제거. 클래스 문자열은 `telemetry-panel__empty is-hidden`.
2. 목록에서 사라진 드론의 `li`를 제거한다(`telemetryPanel.ts:99-104`).
3. 처음 보는 드론은 `li`를 만들어 `ul.telemetry-panel__list` 끝에 붙인다(`ensureRow`, `telemetryPanel.ts:47-93`). 행 마크업(캡처 그대로):

   ```html
   <li class="telemetry-row" tabindex="0"><div class="telemetry-row__head"><span class="telemetry-row__dot" style="background: rgb(159, 232, 255);"></span><span class="telemetry-row__name">중앙 드론</span><span class="telemetry-row__stale">수신 끊김</span></div><div class="telemetry-row__vals">36.36872, 127.34824 · 149m · 135° · 12.0m/s · 98%</div></li>
   ```

   - `tabindex="0"`은 `li.tabIndex = 0`(`telemetryPanel.ts:53`).
   - `.telemetry-row__dot`의 `background`: 드론 **id** 기준 색 `DRONE_TINTS[(id - 1) % 3]`(`skylens_core/drones/telemetryFleet.ts:241-243`, `shared/viewer/config.ts:153`). id 1 = `#9fe8ff` = `rgb(159, 232, 255)`, id 2 = `#ffd27f` = `rgb(255, 210, 127)`, id 3 = `#c79fff` = `rgb(199, 159, 255)`. 행을 만들 때 한 번만 정한다.
   - `.telemetry-row__stale` 문구 `"수신 끊김"`은 고정이다. 보이고 숨는 것은 CSS가 행의 `is-stale`로 정한다.
4. 모든 드론에 대해 매번(`telemetryPanel.ts:106-116`):
   - `.telemetry-row__name` 텍스트 = 스테이션 이름(`skylens_core/ui/stationLabel.ts:9-13`): `left` = `"좌측 드론"`, `center` = `"중앙 드론"`, `right` = `"우측 드론"`, 모르는 값이면 `"중앙 드론"`.
   - `li`의 `is-active`: 이 드론이 선택된 드론(`state.activeDroneId`)일 때만.
   - `li`의 `is-stale`: 텔레메트리가 3초 넘게 없을 때(`telemetryFleet.ts:25`).
   - 클래스 순서는 `telemetry-row is-active is-stale`(토글 순서).
   - `.telemetry-row__vals` 텍스트:
     `"${lat.toFixed(5)}, ${lon.toFixed(5)} · ${alt.toFixed(0)}m · ${headingDeg.toFixed(0)}° · ${speed.toFixed(1)}m/s · ${batteryPct.toFixed(0)}%"`
5. `li` 순서를 함대 순서(좌, 중앙, 우)에 맞춘다. 이미 맞으면 손대지 않는다(`telemetryPanel.ts:127-131`).

행 클릭 또는 Enter/Space(`telemetryPanel.ts:73-87`)는 그 드론을 선택한다. 그 결과 다음 렌더에서 `is-active`가 옮겨 가고, 영상 패널이 그 드론으로 바뀐다(§5.3, `control.ts:104-112`).

캡처된 변형:

| 캡처 | empty | 행 |
|---|---|---|
| `control-idle` | 클래스 `telemetry-panel__empty` | 없음 |
| `control-assigned` | `telemetry-panel__empty is-hidden` | 좌측(id 2, `rgb(255, 210, 127)`), 중앙(id 1, `rgb(159, 232, 255)`, `is-active`), 우측(id 3, `rgb(199, 159, 255)`). vals 예: `36.36699, 127.34530 · 105m · 45° · 25.1m/s · 100%` |
| `control-active` | 같음 | 같은 세 행, vals만 바뀜: `36.36889, 127.34830 · 152m · 135° · 12.0m/s · 98%` 등 |
| `control-active-late` | 같음 | vals 예: `36.36878, 127.34721 · 142m · 45° · 12.0m/s · 96%` |

## 5. 영상 패널 `#video-panel-mount > .video-panel`

만드는 곳: `skylens_core/control/videoPanel.ts:32-55`.

### 5.1 구조

```html
<div class="sl-surface sl-surface--frame video-panel"><canvas class="video-panel__canvas" width="320" height="200"></canvas><video class="video-panel__video is-hidden" autoplay="" playsinline=""></video><div class="video-panel__label">CENTER CAM</div><div class="video-panel__overlay">중앙 드론 · NO SIGNAL</div></div>
```

- 캔버스 `width="320" height="200"`은 TS가 정한 값이다(`videoPanel.ts:38-39`). three.js가 아니므로 템플릿에 남겼다. 영상이 없을 때 이 캔버스에 대기 화면(그라데이션, 주사선, 조준선)을 매 프레임 그린다(`videoPanel.ts:70-118`).
- **생성 직후** `.video-panel__overlay`는 비어 있고, 첫 프레임에서 `updateOverlay`가 채운다.

### 5.2 `.video-panel__overlay` 텍스트 `updateOverlay`, `videoPanel.ts:120-139`

매 프레임 다시 쓴다. `who` = 선택된 드론의 스테이션 이름(§4.2의 표, 기본 `"중앙 드론"`).

| 조건 | 텍스트 |
|---|---|
| 재생 오류가 있음 | 오류 문구(§5.4) |
| 선택 드론의 텔레메트리 없음, 영상 없음 | `"${who} · NO SIGNAL"` |
| 선택 드론의 텔레메트리 없음, 영상 있음 | `"${who} · 수신 대기"` |
| 텔레메트리 있음, 영상 없음 | `"${who} · ${where} · 영상 대기"` |
| 텔레메트리 있음, 영상 있음 | `"${who} · ${where}"` |

`where` = `"${abs(lat).toFixed(5)}°${N|S} ${abs(lon).toFixed(5)}°${E|W} · ${alt.toFixed(0)}m"` (위도 ≥ 0이면 `N`, 경도 ≥ 0이면 `E`).

캡처: idle `"중앙 드론 · NO SIGNAL"`, assigned `"중앙 드론 · 36.36699°N 127.34530°E · 105m · 영상 대기"`, active `"중앙 드론 · 36.36872°N 127.34824°E · 149m"`.

### 5.3 `.video-panel__label` 텍스트 `select`, `videoPanel.ts:144-152`

선택 드론이 바뀔 때 `stationCamLabel`(`stationLabel.ts:15-19,27-29`): `left` = `"LEFT CAM"`, `center` = `"CENTER CAM"`, `right` = `"RIGHT CAM"`.

### 5.4 캔버스와 비디오 전환

두 요소는 항상 하나만 보인다. `is-hidden` 클래스로 바꾼다.

| 함수 | 캔버스 클래스 | 비디오 클래스 | 비디오 속성 |
|---|---|---|---|
| `standby` `videoPanel.ts:240-248` (영상 없음, 드론 바꿈) | `video-panel__canvas` | `video-panel__video is-hidden` | `src` 속성 제거 |
| `showVideo(url)` `videoPanel.ts:250-270` | `video-panel__canvas is-hidden` | `video-panel__video` | `src="${url}"`, `loop=""` |
| `playing` 이벤트 `videoPanel.ts:295-300` | `is-hidden` 추가 | `is-hidden` 제거 | 없음 |
| `fail(reason)` `videoPanel.ts:272-277` | `is-hidden` 제거 | `is-hidden` 추가 | 없음 |

- `url` = `feed.previewUri ?? feed.uri`. 캡처(active): `<video class="video-panel__video" autoplay="" playsinline="" src="/res/static/video/demo_center.mp4" loop="">`, 캔버스는 `video-panel__canvas is-hidden`.
- 역방향 재생(`feed.reverse`)일 때는 `loop`를 끈다(`video.loop = false`, `videoPanel.ts:204`). 속성 `loop`가 사라진다.
- 오류 문구(`.video-panel__overlay`에 그대로 나감):
  - `"영상 재생 불가 \u{2014} 자동재생 차단"` (`videoPanel.ts:267`, NotAllowedError)
  - `"영상 재생 불가"` (`videoPanel.ts:267`, 그 밖의 거부)
  - `"영상 재생 불가 \u{2014} 코덱 미지원"` (`videoPanel.ts:282`, MediaError 코드 4)
  - `"영상 재생 불가 \u{2014} 소스 오류"` (`videoPanel.ts:282`, 그 밖의 MediaError)
- 코어 연결이 끊기면 `setFeed(null)` + `setTelemetry(null)`(`control.ts:126-127`): 모든 피드를 잊고 `standby`.

## 6. 장면 위 3D 요소

경로선, 드론 리그, 지형, 건물은 three.js 장면 안에 그렸다(`controlview/towerViewer.ts`). DOM 요소가 아니므로 이 문서 범위 밖이다. `#control-view`가 그 마운트다.

## 7. 경로 계획 모달 (첫 번째 `.route-modal-overlay`)

만드는 곳: `skylens_core/control/routeModal.ts:203-356`. `body` 끝에 붙는다. 같은 클래스의 오버레이가 하나 더 있으므로(클라우드 미션, §8) Rust는 순서(첫 번째) 또는 내부 구조로 구분한다.

### 7.1 구조 (템플릿 그대로)

```html
<div class="route-modal-overlay is-hidden"><div class="route-modal"><button type="button" class="route-modal__close">×</button><div class="route-modal__title">경로 계획 · Route Planner</div><div class="route-modal__hint">지도를 클릭해 웨이포인트를 추가하세요 · 마커를 클릭하면 삭제됩니다</div><div class="route-modal__toolbar"><div class="route-modal__spans"><button type="button" class="route-modal__span">300m</button><button type="button" class="route-modal__span is-active">600m</button><button type="button" class="route-modal__span">1.2km</button><button type="button" class="route-modal__span">2.5km</button></div><label class="route-modal__alt"><span>고도 60m (지면 기준)</span><input type="range" min="20" max="200" step="5"></label><label class="route-modal__loop"><input type="checkbox"><span>왕복 반복</span></label><label class="route-modal__loop"><input type="checkbox"><span>편대 3기 KMZ</span><input type="number" min="1" max="100" step="0.5" class="route-modal__spacing" title="정삼각형 한 변 (m) · 앞왼·앞오·뒤중" disabled=""><span>m</span></label><label class="route-modal__loop"><input type="checkbox" title="DJI Fly 계정 클라우드에 올려 원격 RC 2가 받게 한다 (Bridge에 클라우드 설정 필요)"><span>클라우드 전송(원격 RC)</span></label></div><div class="route-modal__map"><canvas class="route-modal__canvas" width="460" height="460"></canvas></div><ol class="route-modal__list"></ol><div class="route-modal__actions"><button type="button" class="route-modal__btn route-modal__btn--ghost">Clear</button><button type="button" class="route-modal__btn route-modal__btn--primary">Assign</button></div></div></div>
```

입력값은 **프로퍼티**로 들어가 캡처에 속성으로 나오지 않는다. 초기값:

| 요소 | 프로퍼티 | 초기값 | 출처 |
|---|---|---|---|
| `input[type=range]` | `value` | `"60"` | `routeModal.ts:250` |
| 왕복 반복 체크박스 | `checked` | `true` | `routeModal.ts:270` |
| 편대 체크박스 | `checked` | `false` | `routeModal.ts:283` |
| `.route-modal__spacing` | `value` | `"10"` | `routeModal.ts:291` |
| 클라우드 체크박스 | `checked` | `false` | `routeModal.ts:313` |

캔버스 `width="460" height="460"`은 TS 값(`routeModal.ts:329-330`, `MAP_PX`)이라 남겼다. 지도(위성 타일 또는 격자, 축척 막대, 북쪽 표시, 기준점 십자, 경로선, 번호 마커)는 이 캔버스에 2D로 그린다(`routeModal.ts:391-510`). DOM 변화는 없다.

### 7.2 축척 버튼 `.route-modal__span` `routeModal.ts:224-240`

네 버튼 `300m`, `600m`, `1.2km`, `2.5km`(`SPANS = [300, 600, 1200, 2500]`, 라벨은 1000 이상이면 `${s/1000}km`). 선택된 하나에만 `is-active`. 초기 600 m.

### 7.3 고도 라벨 `label.route-modal__alt > span` `routeModal.ts:251-253`

슬라이더 `input` 때마다 `"고도 ${agl}m (지면 기준)"`. 이미 찍은 웨이포인트의 고도도 다시 계산하고 목록을 다시 그린다(`routeModal.ts:254-263`).

### 7.4 편대 간격 입력 `routeModal.ts:297-300`

편대 체크박스 `change`: `.route-modal__spacing`의 `disabled` = 체크 해제 여부. 체크하면 `disabled` 속성이 사라진다.

### 7.5 웨이포인트 목록 `ol.route-modal__list` `renderList`, `routeModal.ts:512-531`

지도 클릭(추가/마커 클릭 삭제), 행의 삭제 버튼, Clear, 고도 변경 때마다 **목록 전체를 비우고 다시 만든다**(`list.innerHTML = ''`). 행 마크업(캡처 `control-route-planner-filled` 그대로):

```html
<li class="route-modal__item"><span>#1  36.36796, 127.34649 · 120m</span><button type="button" class="route-modal__remove">삭제</button></li>
```

- 라벨: `"#${i + 1}  ${lat.toFixed(5)}, ${lon.toFixed(5)} · ${alt.toFixed(0)}m"`. **`#번호` 뒤 공백이 두 칸**이다.
- `alt`는 지면 고도 + 슬라이더 값을 반올림한 절대 고도다(`routeModal.ts:200`).
- 모달을 닫거나 Assign 해도 목록은 비우지 않는다. `control-assigned` 캡처에도 세 행이 남아 있다.

### 7.6 클라우드 전송 토글 `setCloudAvailable(available)` `routeModal.ts:670-686`

툴바 1번 버튼을 누를 때마다 브리지 상태로 호출된다.

| available | 체크박스 | label 클래스 | 문구 `span` | 체크박스 `title` |
|---|---|---|---|---|
| `false` | `disabled`, `checked=false` | `route-modal__loop is-disabled` | `"클라우드 전송 (설정 필요)"` | `"Bridge에 DJI 클라우드 자격이 없어 비활성화됨 (SKYLENS_DJI_MC_TOKEN + SKYLENS_DJI_WK_KEY)"` |
| `true` | `disabled` 해제 | `route-modal__loop` | `"클라우드 전송(원격 RC)"` | `"DJI Fly 계정 클라우드에 올려 원격 RC 2가 받게 한다 (Bridge에 클라우드 설정 필요)"` |

캡처(브리지 없음): `<label class="route-modal__loop is-disabled"><input type="checkbox" title="Bridge에 DJI 클라우드 자격이 없어 비활성화됨 (SKYLENS_DJI_MC_TOKEN + SKYLENS_DJI_WK_KEY)" disabled=""><span>클라우드 전송 (설정 필요)</span></label>`. 속성 순서는 `title` 다음 `disabled`다(캡처에 따라 `style=""`이 끼지만 재현하지 않는다).

### 7.7 열기, 닫기, 안내 문구

| 동작 | 변화 | 출처 |
|---|---|---|
| `open()` | `.route-modal__hint` 텍스트를 `"지도를 클릭해 웨이포인트를 추가하세요 · 마커를 클릭하면 삭제됩니다"`로 되돌리고 `is-warn` 제거, 오버레이 `is-hidden` 제거 | `routeModal.ts:688-694` |
| `close()` (× 버튼, 오버레이 바깥 클릭, Assign 성공) | 오버레이에 `is-hidden` 추가 | `routeModal.ts:628-631,653-656` |
| Assign, 웨이포인트 2개 미만 | `.route-modal__hint` 텍스트 `"경로에는 최소 2개의 웨이포인트가 필요합니다"`, 클래스 `route-modal__hint is-warn`. 모달은 열린 채로 | `routeModal.ts:637-640` |
| 지도 드래그 중 | 캔버스에 `is-panning` 추가, 놓으면 제거 | `routeModal.ts:577,585` |

Assign 성공 시 결과 토스트는 §11 참조.

## 8. 클라우드 미션 모달 (두 번째 `.route-modal-overlay`)

> 지금 템플릿은 레거시와 다르다. 닫기 버튼(`×`)을 제목 밖 첫 자식으로 옮겨 경로 계획·설정 창과 같은 모양으로 했고, 제목은 `div.route-modal__title` `클라우드 미션 · Cloud`, 설명은 `DJI Fly 계정 클라우드에 있는 웨이포인트 미션입니다. 원격 RC 2가 동기화하면 받아갑니다.`이다. 오버레이에 `route-modal-overlay--center`를 더해 설정 창처럼 화면 가운데에 띄운다(경로 계획은 지도가 밀리지 않게 위에 붙인 그대로). 아래는 레거시 기록이다.

만드는 곳: `skylens_core/control/cloudModal.ts:14-48`.

### 8.1 구조 (템플릿 그대로)

```html
<div class="route-modal-overlay is-hidden"><div class="route-modal"><h2 class="route-modal__title">클라우드 미션<button type="button" class="route-modal__close" aria-label="닫기">✕</button></h2><p class="route-modal__hint">DJI Fly 계정 클라우드의 웨이포인트 미션. 원격 RC 2가 동기화하면 받아간다.</p><div class="route-modal__list"></div><div class="route-modal__actions"><button type="button" class="route-modal__btn">새로고침</button></div></div></div>
```

경로 계획 모달과 달리 목록이 `div.route-modal__list`이고, 닫기 버튼이 제목 `h2` 안에 있으며 글자가 `✕`(U+2715)다.

### 8.2 목록 `div.route-modal__list`

항상 `replaceChildren`으로 통째로 바꾼다.

- 메시지 한 줄(`renderEmpty`, `cloudModal.ts:52-58`):

  ```html
  <div class="route-modal__item">불러오지 못했습니다.</div>
  ```

  문구: `"불러오는 중…"`(새로고침 시작, `cloudModal.ts:90`, 끝 글자는 U+2026), `"클라우드에 미션이 없습니다."`(`cloudModal.ts:62`), `"불러오지 못했습니다."`(`cloudModal.ts:94`).
- 미션 행(`renderMissions`, `cloudModal.ts:60-84`):

  ```html
  <div class="route-modal__item"><span>${name} · ${points} WP${dist}</span><button type="button" class="route-modal__remove">삭제</button></div>
  ```

  `name`이 비면 `"(이름 없음)"`, `points` = `waypointCount`, 없으면 `"?"`, `dist` = `distanceM`이 있으면 `" · ${round(distanceM)}m"`, 없으면 빈 문자열. 캡처에는 브리지가 없어 이 행이 없다. 마크업은 코드에서 옮겼다.

### 8.3 버튼 비활성

- 새로고침 중 `.route-modal__btn`(새로고침)에 `disabled`, 끝나면 해제(`cloudModal.ts:89,98`).
- 행의 삭제 버튼은 누르면 `disabled`, 실패하면 해제(`cloudModal.ts:105,115`). 성공하면 목록을 새로 불러온다(`cloudModal.ts:119`).

### 8.4 열기, 닫기

- `open()`: 오버레이 `is-hidden` 제거 후 새로고침(`cloudModal.ts:122-125`).
- `close()`(✕, 바깥 클릭): `is-hidden` 추가(`cloudModal.ts:126-128,134-137`).

캡처 `control-open-클라우드_미션_Cloud`: 오버레이 `route-modal-overlay`(숨김 없음), 목록에 `<div class="route-modal__item">불러오지 못했습니다.</div>`, 토스트 하나(§11).

## 9. 설정 패널 `.settings-overlay`

만드는 곳: `skylens_core/ui/settingsPanel.ts:42-137`.

### 9.1 구조 (템플릿 그대로)

```html
<div class="settings-overlay is-hidden"><div class="sl-surface sl-surface--modal settings-panel"><button type="button" class="settings-panel__close" aria-label="설정 닫기">×</button><div class="settings-panel__title">설정 · Display</div><div class="settings-panel__section">건물 표시 방식</div><div class="settings-panel__options" role="radiogroup" aria-label="건물 표시 방식"><button type="button" class="settings-option" data-mode="points" role="radio" aria-checked="false"><span class="settings-option__name">점</span><span class="settings-option__hint">VWorld 건물을 포인트로 표시</span></button><button type="button" class="settings-option is-active" data-mode="black" role="radio" aria-checked="true"><span class="settings-option__name">검정 텍스처 건물<span class="settings-option__badge">기본값</span></span><span class="settings-option__hint">건물에 검정 텍스처를 적용해 표시</span></button><button type="button" class="settings-option" data-mode="aerial" role="radio" aria-checked="false"><span class="settings-option__name">실사 항공뷰</span><span class="settings-option__hint">위성/항공 영상을 그대로 적용</span></button></div><dl class="settings-panel__meta"><dt>작전 구역</dt><dd>daejeon (36.355, 127.330)</dd><dt>건물 출처</dt><dd>VWorld 실측 건물 데이터 · 6,194동</dd><dt>위성 영상</dt><dd>사용 가능</dd></dl><div class="settings-panel__note">선택은 이 브라우저에 저장되어 다음 접속 때 그대로 복원됩니다</div></div></div>
```

### 9.2 표시 방식 버튼 `settingsPanel.ts:141-148`

설정이 바뀔 때마다(초기 한 번 포함) 세 버튼 각각: 선택된 것에 `is-active` + `aria-checked="true"`, 나머지는 `is-active` 없음 + `aria-checked="false"`. 모드 목록은 `skylens_core/settings.ts:16-24`.

### 9.3 장면 로드 결과로 한 번 정해지는 값 `settingsPanel.ts:95-133`

이 값들은 패널을 만들 때 한 번 정해지고 그 뒤 바뀌지 않는다. 템플릿은 대전 장면, 위성 영상 있음, VWorld 건물 6,194동의 결과다. Rust는 실제 장면 값으로 다시 써야 한다.

| 요소 | 규칙 |
|---|---|
| `data-mode="aerial"` 버튼 | 위성 영상이 없으면 `disabled` 속성, 그 버튼의 `.settings-option__hint` = `"위성 영상을 불러올 수 없어 사용할 수 없습니다 (VWorld 키 확인)"` |
| 1번 `dd` (작전 구역) | `"${CONFIG.control.defaultMap} (${bbox[1].toFixed(3)}, ${bbox[0].toFixed(3)})"` (`control.ts:258`) |
| 2번 `dd` (건물 출처) | 출처 라벨 + 건물 수가 0보다 크면 `" · ${n.toLocaleString('ko-KR')}동"`. 출처 라벨: `vworld` = `"VWorld 실측 건물 데이터"`, `synthetic` = `"대체 블록 (데모 대체 데이터 · 실측 아님)"`, `none` = `"건물 데이터 없음"`(`settingsPanel.ts:36-40`). `synthetic`이면 `dd`에 `class="is-warn"` |
| 3번 `dd` (위성 영상) | `"사용 가능"` 또는 `"사용 불가"` |
| `.settings-panel__note` | URL `?display=`로 지정되었으면 `"URL(?display=)로 지정된 표시 방식입니다 \u{2014} 저장된 설정은 변경되지 않습니다"`, 아니면 템플릿 문구 |

### 9.4 열기, 닫기

- 툴바 3번: `.settings-overlay`의 `is-hidden` 토글(`settingsPanel.ts:165-167`).
- ×, 바깥 클릭, Escape: `is-hidden` 추가(`settingsPanel.ts:150,153-160`).

캡처 `control-open-settings`: 클래스가 `settings-overlay`(숨김 없음)로 바뀐 것 말고는 idle과 같다.

## 10. 숨김 상태 요약

| 요소 | 숨김 클래스 | 처음 |
|---|---|---|
| 경로 계획 오버레이 | `is-hidden` | 숨김 |
| 클라우드 미션 오버레이 | `is-hidden` | 숨김 |
| 설정 오버레이 | `is-hidden` | 숨김 |
| `.telemetry-panel__empty` | `is-hidden` | 보임 |
| `.video-panel__canvas` / `.video-panel__video` | `is-hidden` | 캔버스 보임, 비디오 숨김 |

## 11. 토스트 `#toast-root`

만드는 곳: `shared/viewer/ui/toast.ts:10-25`. `#toast-root`(클래스 `toast-root`) 끝에 붙이고 3200 ms 뒤 제거한다. 여러 개가 쌓일 수 있다.

```html
<div class="toast">경로 전송됨 · 웨이포인트 3개 · 왕복</div>
<div class="toast kind-danger">클라우드 미션 조회 실패 · Error: Mission Bridge에 연결할 수 없습니다 (http://127.0.0.1:8091). npm run mission-bridge를 실행하세요. TypeError: Failed to fetch</div>
```

종류가 `info`면 `toast`만, `danger`/`person`이면 `toast kind-danger`/`toast kind-person`. 두 번째 줄은 `control-open-클라우드_미션_Cloud` 캡처 그대로다.

관제탑이 띄우는 토스트 전부:

| 언제 | 문구 | 종류 | 출처 |
|---|---|---|---|
| Assign, 전송 성공 | `"경로 전송됨 · 웨이포인트 ${n}개${loop ? " · 왕복" : ""}"` | info | `control.ts:186-191` |
| Assign, 코어 끊김 | `"코어에 연결되어 있지 않아 경로를 전송하지 못했습니다"` | danger | `control.ts:189` |
| 편대 KMZ 성공 | `"편대 KMZ ${members.length}개 다운로드됨 · ${list}${cloudNote}"` + 경고가 있으면 `" · 확인 ${warned}건"`. `list` = 스테이션 이름을 `" · "`로 이음(`left` = `"앞·왼쪽"`, `right` = `"앞·오른쪽"`, `center` = `"뒤·중앙"`, `missionBridge.ts:112-116`). `cloudNote` = 업로드가 있으면 `" · 클라우드 ${uploaded}기 업로드"` | info | `control.ts:194-203` |
| 편대 KMZ 실패 | `"편대 KMZ 생성 실패 · ${error}"` | danger | `control.ts:208` |
| 단일 임무 성공 | `"${delivered}${cloudNote} · ${waypointCount} WP${warning}"`. `delivered` = `"DJI Fly 임무에 설치됨"` 또는 `"KMZ 다운로드됨"`, `cloudNote` = `" · 클라우드 업로드됨(원격 RC)"` 또는 빈 문자열, `warning` = `" · 확인 ${n}건"` 또는 빈 문자열 | info | `control.ts:213-217` |
| 단일 임무 실패 | `"DJI 임무 생성 실패 · ${error}"` | danger | `control.ts:221` |
| 클라우드 목록 실패 | `"클라우드 미션 조회 실패 · ${error}"` | danger | `cloudModal.ts:95` |
| 클라우드 삭제 | `"삭제됨 · ${name}"` (info) 또는 `"삭제 요청됨(미확인) · ${name}"` (danger) | 표시 | `cloudModal.ts:109-112` |
| 클라우드 삭제 실패 | `"삭제 실패 · ${error}"` | danger | `cloudModal.ts:114` |

`${error}`는 JS `String(error)` 결과다(예: `"Error: ..."`).

## 12. 로딩 화면 `.loading-screen`

만드는 곳: `shared/viewer/ui/loadingScreen.ts:9-50`. 장면을 받는 동안 `body` 끝에 붙는다(`control.ts:67-69`).

```html
<div class="loading-screen"><div class="loading-screen__box"><div class="loading-screen__title">관제탑 · 실지형 장면 로딩</div><div class="loading-screen__bar"><span style="width: 42%;"></span></div><div class="loading-screen__text">42%</div></div></div>
```

- `progress(p)`: 막대 `span`의 인라인 `width: ${p}%`(0..100으로 자름), `.loading-screen__text` = `"${round(p)}%"`(라벨이 주어지면 라벨).
- `done()`: `is-hidden` 추가, 400 ms 뒤 요소 제거.
- `fail(msg)`: 제목 `"장면 로드 실패"`, 텍스트 = `msg`, `is-error` 추가, 2500 ms 뒤 `done`과 같은 제거.

## 13. `body` 클래스

- `is-manual`: 키보드 수동 조종이 켜져 있는 동안 `document.body`에 붙는다(`control.ts:244-249`, `drones/manualLink.ts`의 `onActiveChange`).
