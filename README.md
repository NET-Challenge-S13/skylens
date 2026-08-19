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

이 저장소는 파이프라인 전체를 컴포넌트로 나눠 담고 있습니다 — 드론 → 게이트웨이 → 프록시 → 코어 → (모델 API · 관제탑 · 현황판). 구성과 각 컴포넌트의 책임은 **[COMPONENTS.md](res/docs/COMPONENTS.md)가 단일 출처**입니다.

운영자가 보는 화면은 둘이고, **서로 직접 연결되지 않습니다.** 둘 사이에는 파이프라인 전체가 놓입니다:

- **관제탑** — 오퍼레이터가 **실제 GPS로 드론 경로를 지정**하는 화면. VWorld 실지형 위에 드론의 실제 텔레메트리를 그리며, 건물 표시를 점·검정 텍스처·실사 항공뷰 중에서 고를 수 있습니다. 코어가 서빙합니다.
- **현황판** — 드론이 지나간 **구간부터** 서버가 복원 결과를 보내옵니다. 한 구간을 최종 품질까지 한 번에 처리하지 않고 **낮은 수준을 먼저 확정해 띄운 뒤 정제**하며, 앞 구간이 정제되는 동안 다음 구간의 낮은 수준이 도착합니다(**딜레이 패턴**). **서버의 인간 탐지 모델 결과(GPS)**가 도착하면 3D 위에 마커로 표시됩니다.

> 📄 기획·설계: [IDEA.md](res/docs/IDEA.md) · [ARCHITECTURE.md](res/docs/ARCHITECTURE.md) · [PROJECT.md](PROJECT.md)

---

## 데모 모드 vs 실운영

데모에서도 **컴포넌트는 전부 실제 코드**입니다. 한 대의 머신에서 재현할 수 없는 두 가지만 바뀝니다.

| | 실운영 | 데모 |
|---|---|---|
| 드론 촬영 | 기체 카메라 → H.265 실시간 인코딩 | `res/static/video/h265`의 사전 인코딩 영상 |
| 3D 복원 | Core HPC에서 gsplat 학습 | 미리 만들어 둔 구간×수준 자산 |

나머지는 전부 실제 경로입니다 — 드론이 게이트웨이에 붙고, 프록시가 코어로 중계하고, 코어가 모델 API에 잡을 발행하고, 현황판이 릴레이를 통해 받습니다. 화면은 도착한 것만 그리며, 파이프라인이 없으면 **없다고 표시**합니다(시뮬레이션으로 메우지 않습니다).

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
npm run demo         # 파이프라인 전체를 데모 모드로 기동
```

| 화면 | 접속 주소 |
|---|---|
| 관제탑 | `http://localhost:8080/res/static/control.html` |
| 현황판 | `http://localhost:8090/res/static/status.html` |
| 드론 패널 | `http://localhost:5173/src/skylens_drone/index.html` |

관제탑에서 경로를 지정하면 **태스크 지정 완료 → 드론 연결 대기(약 10초) → 주행 시작**으로 이어지고, 드론이 구간을 지날 때마다 현황판이 딜레이 패턴으로 갱신됩니다. 자세한 시나리오는 [demo/README.md](src/demo/README.md).

| 명령어 | 설명 |
|---|---|
| `npm run demo` | 데모 런처 (전 컴포넌트 조립) |
| `npm run dev` | Vite 개발 서버만 |
| `npm run core` / `client` / `gateway` / `proxy` / `drone` / `model` | 컴포넌트 개별 기동 |
| `npm run build` | 타입체크 + 멀티페이지 빌드 |
| `npm test` | Playwright E2E |

### 주요 쿼리 옵션
`?demo` 자동 데모 · `?room=<이름>` P2P 방 · `?splat=off|demo|cdn|light|<url>` 스플랫 자산 · `?delay=off` 딜레이 패턴 끄고 단일 장면으로 · `?reveal=on/off` 스플랫 reveal 마스크 · `?spin=off` 카메라 자동회전 끔 · `?up=<preset|euler>` 스플랫 방향 · `?level=on` PCA 자동 레벨링.

**실지형 지도 옵션** (`?map` 사용 시)
- `?map` — 실지형 지도 씬 활성화; `?map=uljin|gangneung|서,남,동,북` 으로 씬/영역 선택 가능
- `?tex=sat|off` — 위성 드레이프 (sat: VWorld 위성영상, off: 없음)
- `?drone=<n>` — 드론 뷰 스케일 (기본값 0.15)
- `?ring=off` — 배경 지형 링 비활성화
- `?bld=off` — 3D 건물 렌더링 비활성화

---

## 딜레이 패턴 3D 복원 (현황판)

복원은 촬영이 끝난 뒤 한 번에 처리하는 방식이 아니라, 드론이 지나간 **구간**마다 낮은 학습 스텝의 결과를 먼저 확정해 내보내고 뒤이어 정제합니다. 앞 구간이 정제되는 동안 다음 구간의 첫 수준이 도착하므로 화면에는 여러 구간이 서로 다른 수준으로 동시에 올라옵니다 (중간보고서 Ⅱ-3-다).

| 수준 | 학습 스텝 | 지휘관이 확인 가능한 것 |
|---|---:|---|
| 1 | 250 | 형상 윤곽만 식별 |
| 2 | 1,000 | 공간 구조 식별 가능 |
| 3 | 3,500 | 표면 형성 |
| 4 | 7,000 | 진입 동선 판단이 가능한 실용 품질 |

새 수준이 도착하면 같은 구간의 낮은 수준을 **교체**하고(누적되지 않음), 처리가 밀려 이미 추월당한 수준은 아예 건너뜁니다. 좌측 상단 서버 패널에 구간별 현재 수준이 표시됩니다.

**데모 자산 만들기** — `res/static/demo/`에 학습 스텝별 경량 PLY(`step00250_light.ply` …)를 두고 구간으로 자릅니다. 자산은 커밋하지 않으며, 없으면 현황판은 단일 장면 스트림으로 자동 폴백합니다.

```bash
uv run python -m skylens_model.models.skylens.split_segments res/static/demo/step*_light.ply --segments 4
```

구간 경계는 장면의 주축(복도 촬영이면 드론 진행 방향)을 기준 파일의 분위수로 잘라 모든 수준에 동일하게 적용하므로, 같은 구간 번호는 항상 같은 공간 조각을 가리킵니다. 타이밍은 `CONFIG.delayPattern`(구간 주기·수준별 지연)에서 조정합니다.

---

## 조작법 (관제탑)
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

컴포넌트 경계와 책임은 [COMPONENTS.md](res/docs/COMPONENTS.md)가 단일 출처입니다.

```
res/static/           # 정적 html 셸 — index · control · status
src/
├─ shared/            # 컴포넌트 공통 계약: protocol · geo · types (DOM·Three 없음)
│  ├─ viewer/         # 두 화면이 함께 쓰는 브라우저 층 (씬 소스 · 스토어 · 설정)
│  └─ net/            # WebRTC 트랜스포트
├─ skylens_drone/     # Tauri 드론 클라이언트 — 비행 · H.265 슬라이스 전송
├─ skylens_gateway/   # KOREN 진입점 (relay | webrtc 홀펀칭)
├─ skylens_proxy/     # KOREN 내부 다중 경로 + 페일오버
├─ skylens_core/      # 관제탑 UI + server/(오케스트레이터 · 인메모리 스토어 · 배포)
├─ skylens_model/     # FastAPI 연산 API + 모델 · 3DGS 복원 파이프라인
├─ skylens_client/    # 현황판 UI + server/(웹서버 · WebRTC 중계)
└─ demo/             # 데모 런처
src/test/             # 모든 테스트와 검증 하네스가 여기 모입니다
```

| 포트 | 컴포넌트 |
|---:|---|
| 8080 | 코어 (`ws /uplink`, `ws /viewer`, 관제탑 서빙) |
| 8081 · 8082 | 게이트웨이 · 프록시 |
| 8090 | 현황판 웹서버 + WebRTC 중계 |
| 8100 | 모델 API |
| 5173 | Vite (두 화면의 개발 모드 원본) |

---

## 현재 상태 & 로드맵

**동작 확인됨** (전부 실제 실행으로 검증)
- ✅ **파이프라인 전 구간**: 드론 → 게이트웨이 → 프록시 → 코어 → 모델 API → 현황판
- ✅ **딜레이 패턴**: 구간이 드론의 **이동량**으로 닫히고, 앞 구간이 정제되는 동안 다음 구간의 수준 1이 도착
- ✅ **미션 단계**: 대기 → 태스크 지정 완료 → 드론 연결 대기(10초) → 주행
- ✅ **경로 이중화**: 프록시가 코어 장애 시 대기 경로로 전환, 복구 시 복귀
- ✅ **관제탑**: VWorld 실지형 + 점·검정 텍스처·실사 항공뷰 3옵션, GPS 좌표계
- ✅ **현황판**: 구간이 도착해야 보이고, 마커도 해당 구간이 복원돼야 표시
- ✅ 모델 API 부재 시 잡 재제출로 복구, 코어 부재 시 마지막 상태 유지

**다음 단계**
- ⏳ 실제 복원·추론 연결 (현재 데모 자산으로 대체, 연결 지점은 `PipelineUnavailable`로 명시)
- ⏳ 2D 탐지 → 3D 좌표 변환(Depth 레이캐스팅) 모듈
- ⏳ 증분 복원의 좌표계 고정
- ⏳ `.spz` 전환 (전송량 약 10배 감소)
- ⏳ KOREN 회선 위 실배치 · 클라이언트 간 P2P 재분배

---

## 기술 스택
**Three.js** · **@mkkellogg/gaussian-splats-3d** · **TypeScript** · **Vite** · **PeerJS/WebRTC** · **Playwright** · **Python**(모델) · **AWS Terrain Tiles** · **VWorld**(위성/건물)

<div align="center">
<sub>SkyLens — 재난 현장을 실시간 3D로, 그 위에 AI를 얹다.</sub>
</div>
