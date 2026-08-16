<div align="center">

# 🛸 SkyLens

**멀티드론 영상을 실시간 3D로 복원하고, 그 위에 AI가 위험구역·사람을 표시하는 재난 인텔리전스 플랫폼**

*NET 챌린지 캠프 시즌13*

<br/>

![status](https://img.shields.io/badge/status-prototype-9fe8ff)
![stack](https://img.shields.io/badge/Three.js%20+%20TS-c8c2b8)
![sync](https://img.shields.io/badge/WebRTC-PeerJS%20%2B%20STUN-ff9a4d)
![splat](https://img.shields.io/badge/Gaussian%20Splatting-real-c084fc)
![coords](https://img.shields.io/badge/coords-GPS%20%2F%20ENU-4ade80)
![tests](https://img.shields.io/badge/e2e-Playwright%209%2F9-39d98a)

</div>

---

## 개요

SkyLens는 여러 대의 드론이 재난 현장을 분할 탐색하며 보낸 영상을 고속망으로 모아 **현장을 실시간 3D(Gaussian Splatting)로 복원**하고, 같은 영상에 **AI를 돌려 위험구역·사람을 감지**해 3D 현장 위에 마커로 얹는 재난 대응 시스템입니다.

이 저장소는 **두 대의 분리된 컴퓨터**에서 각각 한 화면씩 띄우는 운영 프로토타입입니다:

- **SIM (관제탑)** — 오퍼레이터가 **실제 GPS로 드론 경로를 지정**하는 컨트롤타워. 지정된 경로를 리더 드론이 비행하고 군집 드론이 동행하며, 메인 드론의 촬영 영상을 확인합니다.
- **RECON (3D 복원 상황판)** — 드론이 수집한 사진을 **서버가 Gaussian 스플랫 청크로 점진 전송**하면 3D가 계속 확장되고, **서버의 인간 탐지 모델 결과(GPS)**가 도착하면 3D 위에 마커로 표시됩니다.

> 📄 기획·설계: [IDEA.md](res/docs/IDEA.md) · [ARCHITECTURE.md](res/docs/ARCHITECTURE.md) · [PROJECT.md](PROJECT.md)

---

## 데모 모드 vs 실서버 모드

기본값은 **실제 서버 데이터로 동작**합니다. 자체 완결형 자동 데모는 **명시적 옵션**으로만 켜집니다.

| | 기본 (실모드) | 데모 (`?demo`) |
|---|---|---|
| 드론 | 경로 지정 전까지 **대기** | 자동 스윕 경로 비행 |
| 스플랫 | 서버 청크 수신 시 표시 (없으면 **대기**) | mock이 청크 스트리밍 → 즉시 복원 |
| 탐지 | 서버 탐지 결과 도착 시 | mock이 순차 전송 |
| 용도 | 실제 파이프라인 연결용 | 전체 흐름 시연 |

서버는 아직 실 백엔드가 없어 **인터페이스 + mock provider**로 구현돼 있습니다([serverSource.ts](src/skylens_core/server/serverSource.ts)). 실 백엔드는 `connect(url)` 지점에 연결하면 됩니다.

---

## 좌표계 (GPS ↔ ENU ↔ 씬)

실세계 GPS를 1급 좌표계로 씁니다([geo.ts](src/skylens_core/geo.ts)):

- **GeoAnchor**(기준 GPS, `CONFIG.geo.anchor`)를 원점으로 하는 **로컬 ENU(동/북/상) 미터** 프레임 — 1 unit = 1 m.
- 드론 경로는 **GPS로 지정** → ENU → 씬으로 변환. 탐지 결과도 **GPS로 수신** → 씬 좌표로 변환해 마커 배치.
- 각 스플랫 청크는 **명시적 align transform**(pos/rot/scale, 선택적 GPS anchor)을 가져 공통 ENU 프레임에 정렬됩니다.

---

## 빠른 시작

```bash
npm install
npm run dev          # LAN 노출 (다른 컴퓨터에서도 접속)
```

| 컴퓨터 | 접속 주소 |
|---|---|
| A (SIM · 관제탑) | `http://<서버IP>:5173/res/static/sim.html?room=demo` |
| B (RECON · 지휘관) | `http://<서버IP>:5173/res/static/recon.html?room=demo` |

> 랜딩 페이지 `http://<서버IP>:5173/res/static/` 에서 두 화면 링크를 눌러 들어가도 됩니다.

같은 `?room=` 값이면 WebRTC(PeerJS 공개 브로커 + 구글 STUN)로 자동 연결. 전체 흐름을 바로 보려면 뒤에 **`&demo`** 를 붙이세요.

| 명령어 | 설명 |
|---|---|
| `npm run dev` | 개발 서버 (HMR, LAN) |
| `npm run build` | 타입체크 + 멀티페이지 빌드 |
| `npm test` | Playwright E2E |

### 주요 쿼리 옵션
`?demo` 자동 데모 · `?room=<이름>` P2P 방 · `?splat=off|light|<url>` 스플랫 자산 · `?reveal=on/off` 스플랫 reveal 마스크 · `?spin=off` 카메라 자동회전 끔 · `?up=<preset|euler>` 스플랫 방향 · `?level=on` PCA 자동 레벨링.

**실지형 지도 옵션** (`?map` 사용 시)
- `?map` — 실지형 지도 씬 활성화; `?map=uljin|gangneung|서,남,동,북` 으로 씬/영역 선택 가능
- `?tex=sat|off` — 위성 드레이프 (sat: VWorld 위성영상, off: 없음)
- `?drone=<n>` — 드론 뷰 스케일 (기본값 0.15)
- `?ring=off` — 배경 지형 링 비활성화
- `?bld=off` — 3D 건물 렌더링 비활성화

---

## 조작법 (SIM)
- **경로 계획 모달** — 툴바 `경로 계획 · ROUTE` → GPS 웨이포인트 추가 → **배정**하면 리더가 그 경로를 비행.
- **방향키 ↑↓←→** 수동 조향(전/후진 + 좌/우 점진 회전), **Q/E** 고도, **1/2/3·Tab** 드론 전환, **Space** 일시정지.

### 실지형 지도 씬 (`?map`)

지형 메시와 건물 3D 모델을 실제 고도·영상·위치 데이터로 렌더하여 현장을 사실적으로 재구성합니다.

- **지형 메시** — AWS Terrain Tiles DEM(한국 약 30m급) 기반 고도 데이터
- **위성 드레이프** — VWorld WMTS 위성 이미지를 지형 위에 매핑
- **3D 건물** — VWorld 건물 폴리곤 데이터(WFS lt_c_bldginfo)를 지붕까지 입체 프리즘으로 렌더
- **월드 스트리밍** — 드론 반경 내 미로드 셀을 가까운 순으로 실시간 로드 · 씬 주변 3배 저해상 배경 지형 링

**기준 씬**: 대전(충남대~카이스트 일대, ~3km, 약 6,191동)

#### VWorld 키 설정

VWorld 위성/건물 데이터는 API 인증 키가 필요합니다. 저장소 **부모 폴더**에 다음 파일을 생성하세요:

```bash
# 저장소 바로 바깥 디렉터리에서:
cat > .env.vworld << EOF
VWORLD_KEY=<vworld.kr에서 발급한 인증키>
VWORLD_DOMAIN=http://localhost:5173
EOF
```

키가 없으면 지형·건물 렌더링 없이 기존 씬만 동작합니다(graceful degradation). Vite dev 프록시에서만 서버측에서 키를 주입하며, 프론트엔드 번들에는 절대 노출되지 않습니다.

---

## 프로젝트 구조

```
res/static/             # 정적 html 셸 (진입점) — /src 모듈을 로드
│  └─ index.html · sim.html · recon.html
src/
├─ skylens_core/        # 관제탑(SIM) + 공유 토대 — client가 단방향 의존
│  ├─ config · types · store · math
│  ├─ geo.ts            # GPS↔ENU↔씬
│  ├─ mode.ts           # isDemo() (?demo)
│  ├─ protocol.ts       # p2p 스냅샷 + 서버 메시지 스키마
│  ├─ sim.ts  style.css                   # SIM 진입점 + 공유 베이스 스타일
│  ├─ server/serverSource.ts   # 서버 인터페이스 + mock
│  ├─ net/     peer.ts (WebRTC) · statusUi.ts
│  ├─ data/    sceneSource.ts · routes.ts(GPS경로) · paths.ts(데모) · sceneData.ts
│  ├─ drones/  pathFollower.ts (리더 경로 + 군집) · manualControl.ts
│  ├─ sim/     routeModal.ts · videoPanel.ts · sim.css
│  ├─ simview/ lowfiViewer.ts             # SIM 관제 3D 뷰
│  └─ ui/      loadingScreen · toast      # 공유 UI
├─ skylens_client/      # 지휘관(RECON) — 서버 구동 3D 복원 상황판
│  ├─ recon.ts
│  ├─ reconview/ reconViewer · splatScene(다중 청크) · splatReveal · reveal · cameraSync
│  ├─ ui/      overlay · minimap · serverStatus · recon-panels.css
│  └─ data/    detections.ts
└─ skylens_model/       # AI 모델 (Python 스캐폴드)
   ├─ models/   skylens/   # transformers 표준 모델 (config·modeling)
   ├─ datasets/                            # 데이터셋 자리
   └─ utils/geo.py                         # ENU 수식 미러 (TS와 동기)

src/test/smoke.spec.ts     # Playwright E2E
```

---

## 현재 상태 & 로드맵

**구현됨**
- ✅ 두 컴퓨터 분리 구성 + WebRTC P2P 동기화, 연결/서버 수신 상태 표기
- ✅ **관제탑 SIM**: GPS 경로 계획 모달 · 리더 경로 비행 · 군집 동행 · 드론 카메라 패널
- ✅ **서버 구동 RECON**: 스플랫 청크 점진 복원 · GPS 탐지 마커 · 미니맵 · 대기 상태
- ✅ GPS/ENU 좌표계 + 명시적 스플랫 정렬
- ✅ 실사 Gaussian Splatting(정원 씬, 자동 레벨·프레이밍·floater clip) + 드론 스캔 progressive reveal
- ✅ 데모 옵트인(`?demo`) — 기본은 실서버 데이터
- ✅ 전문 컨트롤룸 UI 디자인 시스템
- ✅ Python 모델 스캐폴드(detection/splat 인터페이스 + geo 미러)
- ✅ E2E 9종 통과

**다음 단계**
- ⏳ 실 백엔드 연결(`serverSource.connect`) — 실드론 텔레메트리 · 탐지 모델 · 스플랫 재구성
- ⏳ `skylens_model` 실제 추론(UNet 4채널 탐지 + gsplat) 통합
- ⏳ 자체 촬영 스플랫으로 교체 · KOREN/Core HPC 파이프라인

---

## 기술 스택
**Three.js** · **@mkkellogg/gaussian-splats-3d** · **TypeScript** · **Vite** · **PeerJS/WebRTC** · **Playwright** · **Python**(모델) · **AWS Terrain Tiles** · **VWorld**(위성/건물)

<div align="center">
<sub>SkyLens — 재난 현장을 실시간 3D로, 그 위에 AI를 얹다.</sub>
</div>
