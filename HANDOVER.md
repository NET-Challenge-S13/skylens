# 인계 문서 — 현황판 3D 복원이 화면에 안 그려지는 문제

작성: 2026-08-19 · 브랜치 `feat/console` · 마지막 커밋 `168552a`

이 문서는 **아직 안 풀린 문제 하나**와, 거기까지 오면서 확인·수정한 것들을 넘기기 위한 것이다.
읽는 순서: §1(미해결 문제) → §2(재현) → §3(배제한 것) → §4(다음에 볼 곳) → §5(이번에 고친 것) → §6(도구).

---

## 1. 미해결: 현황판 3D 뷰에 스플랫이 하나도 안 그려진다

**증상.** 현황판(`http://localhost:8090/res/static/status.html`)에서 좌측 패널은 "구간 1~8 · 수준 3 · 최종 품질"로 정상 표시되고 청크 카운터도 올라가는데(46만 개 스플랫), 3D 화면에는 탐지 마커와 UI만 보이고 **복원 지오메트리가 한 픽셀도 안 나온다**.

**중요 — 이건 배치 문제가 아니다.** 좌표는 전부 맞다:

| 측정 | 결과 | 검사 |
|---|---|---|
| 청크가 배정 경로 위에 있는가 | 최대 이탈 **1 m** | `boardSpaceCheck.mjs` |
| 복원이 경로를 따라 연속인가 | 걸치는 10 m 구간의 **100%** | `splatScaleCheck.mjs` |
| 카메라 시야 안에 있는가 | 샘플 3063/3063 | `boardViewCheck.mjs` |
| 안개·원거리 평면 안인가 | 206–251 m, 안개 끝 2078 m | `boardViewCheck.mjs` |
| 실제 그려진 픽셀 | 배경 아닌 픽셀 ≈ 0 (UI 제외) | `boardPixelCheck.mjs` |

즉 **로드·배치·프레이밍은 다 맞는데 래스터화만 안 된다.**

---

## 2. 재현 절차

```bash
npm run demo:clean          # 포트 정리 (이전 실행이 남아 있으면 런처가 거부한다)
npm run demo                # 5개 컴포넌트 기동
node src/test/control/assignRoute.mjs        # 경로 배정 (안 하면 아무 일도 안 일어난다)
# 약 60~90초 뒤
node src/test/client/boardPixelCheck.mjs     # 청크 위에 카메라를 세우고 픽셀을 센다
```

`C:/tmp/skylens-shots/board-close.png` 를 보면 마커만 있고 지오메트리가 없다.

---

## 3. 배제한 것 (다시 파지 말 것)

1. **내가 이번에 넣은 배치 변경이 원인이 아니다.**
   모델의 정렬을 예전처럼 항등(`SplatAlign()`)으로 되돌려 코어의 `placeOnRoute`만 쓰게 해도 똑같이 안 그려진다. 실험용 환경변수는 제거했으니 다시 넣으려면 `serving_pipeline._demo_align` 첫 줄에서 조기 반환하면 된다.

2. **스케일 변환이 원인이 아니다.**
   자산을 `split_segments --length 300`으로 **미터 단위로 재생성**해서 배치가 강체(scale 1)가 된 뒤에도 동일하다.

3. **부유물 클립은 버그가 있었고 고쳤지만, 그것만으로는 안 된다.**
   `splatReveal.ts`의 클립이 `modelMatrix * splatCenter`로 월드 좌표를 구했는데, 이 라이브러리는 `dynamicScene: true`에서 씬마다 `transforms[sceneIndex]`로 배치하고 **`modelMatrix`는 항등**이다. 로컬 좌표를 월드 상자와 비교하고 있었다. `transform * splatCenter`로 고쳤고, 추가로 **클립 자체를 기본 해제**했다(`?clip=on`으로만 켜짐). 그래도 안 그려진다.

4. **좌표계 오해 하나.** `loadedChunks()[].position`(옛 이름)은 **스플랫 씬의 원점**이지 지오메트리 위치가 아니다. 모델이 조각 중심을 앵커에 맞추려고 보정 오프셋을 넣기 때문이다. 지금은 `center` 필드가 실제 위치이며 `align.anchor`에서 온다. 미니맵·카메라가 이걸 쓴다.

---

## 4. 다음에 볼 곳 (우선순위 순)

### (a) 셰이더 패치가 링크에 실패하는지
`src/skylens_client/statusview/splatReveal.ts`의 `attachTo()`가 `onBeforeCompile`에서 문자열 치환으로 셰이더를 고친다. 치환 앵커가 라이브러리 버전과 안 맞으면 `String.replace`가 **조용히 아무 것도 안 하고**, 그 결과 정점 셰이더는 `varying float vReveal;`를 선언하는데 프래그먼트는 선언하지 않아 **프로그램 링크가 깨진다** → 아무 것도 안 그려진다.

확인할 앵커 (`node_modules/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js`):
- 정점: `attribute uint splatIndex;`, `vColor = uintToRGBAVec(sampledCenterColor.r);` (7674행 부근에 존재 확인함)
- 프래그먼트: `varying vec2 vPosition;`, `gl_FragColor = vec4(color.rgb, opacity);` ← **이 둘은 미확인**

가장 먼저 할 일: 브라우저 콘솔에서 `THREE.WebGLProgram` 컴파일 에러를 잡는다. 내가 만든 `boardPixelCheck.mjs`에 콘솔 수집을 넣어뒀지만, 페이지가 바빠 `page.screenshot()`이 타임아웃 나면서 출력을 못 봤다. **패치를 통째로 비활성화(`attachTo`를 즉시 return)하고 렌더링이 살아나는지부터 보는 게 가장 빠른 이분법이다.**

### (b) 라이브러리의 씬 개수 / 정렬 워커
`MAX_SCENES = 64`(우리 쪽)와 라이브러리의 `Constants.MaxScenes`가 다를 수 있다. 구간 8개 × 레벨 교체까지 겹치면 씬 인덱스가 커진다. `sceneIndex`가 배열 범위를 넘으면 셰이더가 이상 동작한다.

### (c) `removeSplatScene` 경합
콘솔에 간헐적으로 `Cannot read properties of null (reading 'visitLeaves')`가 뜬다. 라이브러리가 비동기로 만들던 splat 트리를 제거가 앞질러서 나는 경합이다(`gaussian-splats-3d.module.js` 9642행 부근, `this.splatTree = this.baseSplatTree` 직후 널 참조). 딜레이 패턴이 레벨을 계속 교체하므로 자주 발생한다. 이게 메시를 망가뜨려 렌더가 죽는지 확인할 것.

---

## 5. 이번 세션에서 고친 것 (전부 검증됨, 되돌리지 말 것)

### 관제탑
- **경로 고도가 해발 기준이었다.** 계획 화면의 "고도 60m"가 GPS 절대고도로 나가서, 지면이 해발 57~60 m인 대전에서는 경로선과 편대가 지면 아래에 그려졌다. 이제 관제탑이 각 경유점 아래 지형 표고를 더한다. 검사: `clearanceCheck.mjs` (모든 경유점이 자기 지형 위 동일 간격).
- **건물 최소 높이 규칙이 월드 단위 고정값**이라, 3 km 도심 씬에서 캠퍼스 전체가 11.9 m 슬래브로 눌렸다(1층 창고 ×3.6). 씬이 덮는 실제 지면 크기(약 1/1200)에 묶었다. 검사: `buildingHeightCheck.mjs` (중앙값 ×1.00).
- **건물 발자국 확대 제거** — 낮은 건물이 중심 기준 최대 3배로 넓어져 실제 외곽선이 수십 m 이동했다.
- **드론 목록 DOM이 편대 순서를 안 따라갔다.** 행이 최초 보고 순서로 붙고 이후 재정렬이 없었다. 검사: `fleetOrderCheck.mjs`.
- **카메라 정지 플래그가 오버뷰 회전만 막고 추적 카메라는 못 막았다** — top-down 디버그가 매 프레임 덮여서, 내가 두 번 오진했다.

### 현황판
- **`assign-route`가 뷰어 종류 허용목록 3곳에서 조용히 버려졌다**(브라우저 스트림 / 릴레이 업스트림 / 릴레이 재생 캐시). 이제 `shared/protocol.ts`의 `VIEWER_MESSAGE_KINDS` 하나이고, 빠뜨리면 컴파일이 깨지는 완전성 검사가 붙어 있다.
- **데모 탐지가 비행과 무관한 고정 오프셋**에 찍혔다 → 세그먼트를 촬영한 포즈 기준으로 배치.
- **미니맵에 경로선과 복원 구간을 그린다.** 복원은 원래 비행보다 뒤처지는데(그게 딜레이 패턴이다), 아무 것도 안 그리면 그 지연과 오배치를 구분할 수 없었다.

### 데모/인프라
- **런처가 이전 실행의 서버를 자기 컴포넌트로 오인**했다(헬스 URL에 뭐든 응답하면 준비 완료로 판단). 좀비가 살아 있으면 **옛 코드로 데모가 돈다** — "서버 캐시" 증상의 정체. 이제 포트 선점을 거부하고, 종료 시 프로세스 트리를 죽인다. `npm run demo:clean` 추가.
- **HMR 프록시가 Vite를 죽였다** — 핸드셰이크 잔여 바이트를 `unshift()`로 잘못된 소켓에 넣어 Vite의 프레임이 Vite로 되돌아갔다(`WS_ERR_EXPECTED_MASK`).
- **모델 API가 모든 잡을 HTTP 422로 거부**했다 — 드론이 `station` 필드를 붙이기 시작했는데 파이썬 스키마 미러가 `extra="forbid"`였다.
- **데모 자산을 미터 단위로 재생성**(`split_segments --length 300`, 30.873 m/unit). 매니페스트에 `metersPerUnit`, 세그먼트별 `extent`/`centroid`가 들어간다.
- **`ReconJobRequest.track` 추가** — 세그먼트가 덮는 경로 구간. 세그먼트는 기체가 **떠날 때** 닫히므로 첫 레벨은 영상 슬라이스가 도착하기 전에 나간다. 포즈가 없어도 배치할 수 있어야 해서 코어가 이걸 보낸다.

---

## 6. 알아둘 것

**작업 규칙** (`CLAUDE.md`):
- 커밋 메시지에 **Claude 공동저자·AI 생성 표기 금지**. 메시지는 **영어**, 문서 본문은 한국어.
- 요청받지 않으면 커밋하지 않는다.

**환경**:
- Windows. **Bash 툴에서는 `node`/`npx`/`uv`가 PATH에 없다** — 전부 PowerShell로 실행할 것.
- `npm test`는 **데모를 끈 상태**에서 돌려야 한다(켜져 있으면 현황판이 진짜 서버에 붙어 오탐).

**검사 도구** (`src/test/`, 전부 `npm run demo` 필요):

| 파일 | 무엇을 재는가 |
|---|---|
| `control/assignRoute.mjs` | 경로 배정(다른 검사들의 전제) |
| `control/routeFidelityCheck.mjs` | 계획 → 코어 → 실제 비행 좌표 |
| `control/clearanceCheck.mjs` | 경로·기체가 지면 위에 있는가 |
| `control/registrationCheck.mjs` | 계획 지도 ↔ 3D 드레이프 (상관계수로 배율·이동 측정) |
| `control/footprintOverlayCheck.mjs` | 건물 발자국을 계획 지도에 겹쳐 그림 |
| `control/buildingHeightCheck.mjs` | 그려진 높이 vs VWorld 속성 |
| `control/phantomBuildingCheck.mjs` | 없는 건물이 그려지는가 / 있는 건물이 빠지는가 |
| `control/fleetOrderCheck.mjs` | 드론 목록 DOM 순서 |
| `client/boardSpaceCheck.mjs` | 현황판의 기체·복원·탐지가 경로 위에 있는가 |
| `client/splatScaleCheck.mjs` | 복원이 경로를 따라 연속인가 |
| `client/boardViewCheck.mjs` | **왜 화면이 비었는가** (시야·안개·클립·원거리 평면을 각각) |
| `client/boardPixelCheck.mjs` | 실제로 픽셀이 그려지는가 |
| `client/routeArrivalCheck.mjs` | 경로가 현황판에 도달하는가(실시간/늦은 접속) |

측정 없이 스크린샷만 보고 판단하지 말 것. 이번 세션에서 그렇게 두 번 오진했고, 두 번 다 원인은 "카메라가 내가 생각한 곳을 안 보고 있었다"였다.
