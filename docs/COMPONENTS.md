---
tags: [skylens, architecture]
links: "[[ARCHITECTURE]] · [[NETWORK_ARCHITECTURE]] · [[SPEC]]"
---

# SkyLens 컴포넌트 구성

이 문서는 **저장소를 어떤 컴포넌트로 쪼개고 각 컴포넌트가 무엇을 책임지는지**의 단일 출처다.
`ARCHITECTURE.md`가 4계층(수집·전송·연산·배포)을 설명한다면, 이 문서는 그 계층을 **실제 코드 경계**로 옮긴 결과다.

> **스택 전환 중.** TypeScript를 걷어내고 Rust(드론·모델 서버·클라이언트·데모)와 Go(코어)로 다시 짓는다. 전달만 하는 서버(게이트웨이·프록시)는 코드 없이 nginx 설정만 남긴다. 아래는 **목표 구성**이고, 각 컴포넌트가 지금 어디까지 옮겨졌는지는 §10에 있다. 옮겨지기 전 코드는 각 컴포넌트의 `legacy/`에 참고용으로만 남아 있고 빌드되지 않는다.

---

## 1. 컴포넌트 목록

| 컴포넌트 | 망 위치 | 역할 | 스택 |
|---|---|---|---|
| `skylens_drone` | 현장 | Mission Bridge. 관제탑 GPS 경로를 DJI Fly 웨이포인트 임무(KMZ)로 바꾸고, 편대 3기로 펼치고, DJI 클라우드로 올린다 | Rust |
| `skylens_gateway` | KOREN **외부망** (브리지) | 외부 진입 창구. 현황판 웹·시그널링·드론 업링크를 받아 터널로 대전에 넘긴다. MediaMTX가 드론 SRT 영상을 받아 둔다 | nginx 설정 + MediaMTX |
| `skylens_proxy` | KOREN **내부망** (대전) | 코어로 가는 다중 경로. 선호 경로가 죽으면 대기 경로로 넘긴다 | nginx 설정 |
| `skylens_core` | KOREN **내부망** (대전) | 코어 서버(데이터 전체 보관 + 작업 오케스트레이션 + 배포, Go)와 **관제탑 화면**(코어가 웹으로 제공, Rust + 이전 HTML/CSS). 라이브 영상 원점(MediaMTX WHEP)도 여기 | Go 서버 + Rust 관제탑 + MediaMTX |
| `skylens_model` | KOREN **내부망** (판교) | 연산 서버. 라이브 프레임 입력으로 점군을 키우고(cumulus3d), UNet을 프레임 훅으로 붙이고, 메시로 보정해(deltamesh) 증분으로 내보낸다 | Rust 서버 + Python 학습 코드 |
| `skylens_client` | KOREN **외부망** (지휘관 단말) | **현황판 화면**. 이전 HTML/CSS를 Blitz로 그대로 그리고 렌더링은 pointblitz. 웹·데스크톱 네이티브·Android(이후 iOS)가 같은 코드를 쓴다. 제어 경로가 없다 | Rust + 이전 HTML/CSS |
| `shared` | 해당 없음 | 공통 계약(proto · 좌표계, `skylens-shared`)과 두 화면 공용 층(`viewer/`, `skylens-viewer`: 공용 CSS · 위젯 · 코어 연결 · 청크 저장소 · pointblitz 3D 뷰). 배포 단위가 아닌 의존 라이브러리 | Rust 라이브러리 |
| `test` | 해당 없음 | 컴포넌트를 가로지르는 테스트. 테스트를 먼저 쓰고 구현한다(TDD) | Rust (+ Python 학습 테스트) |
| `demo` | 해당 없음 | 전 컴포넌트를 실제로 띄워 엔드 투 엔드로 돌리는 데모 런처 | Rust |

Python은 **UNet 학습 코드와 데이터셋**에만 남는다. 서빙·복원·드론 쪽 Python은 전부 Rust로 옮긴다.

---

## 2. 데이터 흐름

### 2.1 세 갈래

```
제어       관제탑 ─▶ core ─▶ drone(Mission Bridge) ─▶ DJI 클라우드 ─▶ 기체 온보드 비행

영상·프레임 기체 ─SRT─▶ gateway(브리지 MediaMTX) ◀─당김─ core(대전 MediaMTX)
                                                         ├─WHEP 직결─▶ 현황판 라이브 영상
                                                         └─gRPC 프레임─▶ model(판교) CompositionHost

3D·마커    model ─gRPC 증분 청크─▶ core(보관) ─▶ proxy ─▶ gateway ─▶ 현황판
                                         └──────────────────▶ 관제탑 (core 직결)
```

- **제어**는 관제탑 → 코어 한 방향이다. 현황판에서 코어로 가는 제어 경로는 의도적으로 없다.
- **라이브 영상**은 캐시가 안 되고 접속자 수만큼 대역이 필요해 대전이 현황판에 WebRTC(WHEP)로 직접 쏜다. 브리지는 시그널링만 넘긴다.
- **3D·마커**는 모든 화면이 같은 청크를 받으므로 캐시가 되고, 코어가 보관했다가 내보낸다.

### 2.2 KOREN 진입

KOREN은 등록 IP 외 인바운드를 막고, 브리지 → 대전은 SSH 하나만 열려 있다. 그래서 밖에서 들어오는 HTTP·WebSocket은 전부 **브리지 nginx → SSH 터널 → 대전 프록시 nginx → 코어** 순서를 탄다. 자세한 근거는 `src/skylens_core/deploy/README.md`.

---

## 3. 컴포넌트별 책임

### 3.1 `skylens_drone` (Rust)

예전 `skylens_mission_bridge`를 흡수했다. Lito X1은 공개 DJI SDK가 없어 실시간 스틱 명령을 보낼 수 없으므로, 기체가 공식 지원하는 **DJI Fly 웨이포인트 비행**을 쓴다.

- 경로 → KMZ 임무 생성 (Lito X1 템플릿 KMZ 기반, 템플릿 없으면 거부)
- 경로 하나를 정삼각형 편대 3기의 임무로 펼치기
- DJI Fly 계정 클라우드로 임무 업로드·조회·삭제
- HTTP 표면은 Python 구현과 같다(`:8091`, `/health` · `/missions` · `/formation` · `/cloud/*`)


### 3.2 `skylens_gateway` (nginx · 브리지)

코드가 없다. 이 폴더는 **브리지 호스트에 통째로 올리는 설치 묶음**이다.

| 파일 | 하는 일 |
|---|---|
| `nginx-skylens.conf` | `/` 현황판 웹(정적 파일, wasm MIME 지정), `/viewer`·`/uplink` WebSocket(터널 20882 → 대전 프록시), `/live/{path}/whep` WHEP(터널 20889), `/healthz`(터널 생존) · `/health`(자신) |
| `mediamtx-bridge.yml` | 드론 SRT 수신 :10890. 경로는 코어가 정한 `drone1`~`drone3` |
| `setup-bridge.sh` · `check-bridge.sh` · `bridge.env.example` | 설치·점검·설정. 호스트는 라즈베리파이 5(Debian 13 · arm64)이고, MediaMTX 바이너리는 `uname -m` 으로 고른다. 맥용 `*-macos.sh`(launchd·Homebrew)도 같이 있다 |

### 3.3 `skylens_proxy` (nginx · 대전)

코어 경로를 **이중화**한다. 선호 경로가 연결 거부·무응답·502/503이면 대기 경로로 넘기고, 일정 시간 뒤 선호 경로로 돌아온다. 오픈소스 nginx는 수동 헬스체크만 하므로 열린 WebSocket은 페일오버 때 끊기고, 재접속은 드론·클라이언트가 한다(`src/skylens_proxy/README.md`).

### 3.4 `skylens_core` (Go · 대전)

시스템의 중심. 다섯 가지를 한다.

1. **관제탑 화면.** 관제탑 화면 코드는 코어에 있다(`src/skylens_core/control/`, Rust crate `skylens-control`). 코어 서버가 그 웹 빌드를 서빙하고, 같은 코드는 데스크톱 네이티브 창으로도 뜬다. 관제탑은 코어의 `ws /control`에 직결하고, 제어(경로 배정·수동 조종)는 이 소켓으로만 들어온다. 관제탑에만 바탕 지도(지형·건물)가 가고, 경로가 배정되기 전이면 미리 계획된 경로(`planned_route`, `SKYLENS_CORE_PLANNED_ROUTE`)도 간다. 관제탑은 그것을 빈 경로 계획에 채울 뿐이고 배정은 운용자가 한다.
2. **데이터 전체 보관.** 프레임 메타·텔레메트리·증분 청크·탐지 마커. 이번 단계는 DB 없이 인메모리이고 세션이 끝나면 사라진다.
3. **작업 오케스트레이션.** 대전 MediaMTX에서 프레임을 꺼내 모델 서버에 gRPC(`PushFrames`)로 넘기고, 모델이 낸 증분 청크를 gRPC(`SubscribeChunks`)로 받아 보관한다. 미리보기(preview)·정제(refined) 순서의 스케줄 결정권은 **코어에만** 있다.

코어는 **청크 내용을 풀지 않는다.** 점·메시 블록·탐지 결과는 `payload` 바이트 그대로 보관하고 화면에 넘긴다. 코어가 읽는 것은 옆에 붙은 라우팅 메타데이터(종류·구간·수준·블록 id·version·세대)뿐이다.
4. **배포.** 청크·마커·미션 상태를 화면들로 민다. 새 화면이 붙으면 보관분을 먼저 재생한다. 현황판은 `ws /viewer`(읽기 전용)로 받고, 이 소켓으로 들어온 제어 메시지는 버린다.
5. **라이브 영상 원점.** `deploy/`의 MediaMTX가 브리지 SRT를 당겨 WHEP로 쏜다. 영상 서버 코드는 0줄이다.

### 3.5 `skylens_model` (Rust 서버 · 판교)

연산 서버. **REST 엔드포인트는 `app.py`와 같다**(§7). 같은 잡 API를 gRPC `ModelService`로도 열고, 코어는 gRPC로 붙는다. 라이브 입력(`PushFrames`)과 증분 출력(`SubscribeChunks`)은 gRPC 스트림에만 있다. 계약은 `src/shared/proto/skylens/model/v1/model.proto`. 내부는 세 층이다.

1. **라이브 점군(cumulus3d).** `CompositionHost`에 `ReconNode`를 하나 띄워 두고, 코어가 넘긴 프레임 묶음을 `LiveFrameSet`으로 선언해 `compose`한다. 이미 받은 묶음의 재선언은 무시되고, 빠진 카메라가 있는 묶음은 기다린다. 노드는 cumulus3d **stable display**로 돈다(기본, `SKYLENS_STABLE_DISPLAY`): 미리보기 없이 구간마다 정제본이 나오고, 한 번 보인 구간은 제자리에 있다(내부값이 안정되기 전 구간 3까지는 다시 놓일 수 있다). cumulus3d가 표시 점군을 메모리에서 합성하고(`display_deltas`: 키 해시 1/6 솎기, 겹침 voxel 소유, 낮은 구간 우선) 구간 결과마다 `Event::DisplayDelta`(점 추가·이동·삭제, 조밀한 슬롯 id)를 주면, `models/skylens/slotstream.rs`의 `SlotEncoder`가 이를 pointblitz 슬롯 메시지(PBSD 머리 + v4 팩)로 써서 `CHUNK_KIND_POINT_SLOTS` 청크로 낸다. 파일 싱크도 실행 폴더도 없다. 점은 이 경로로만 나가고, 조밀화가 없을 때(CUDA 없음)와 조밀화가 실패한 구간만 정제 구간의 희소 점을 점 청크로 보낸다. 메시는 구간 점군이 아니라 합성된 표시(거울)에서 만들어 겹친 구간이 이중 면을 만들지 않는다. 스트림 id는 앞선 실행보다 늘 크고, 청크 시퀀스도 실행마다 그 id에서 시작해 모델 서버가 다시 떠도 코어가 새 실행의 청크를 빠짐없이 받는다.
2. **UNet 훅.** 같은 노드에 `on_frame` 훅으로 UNet을 붙인다. 훅은 위치 계산 전에 디코딩된 RGB 프레임을 받아 별도 큐에서 돌므로 복원을 막지 않는다. 프레임마다 나온 위험구역 확률과 사람 점은 등록된 카메라 포즈로 점군에 투영하고, **여러 프레임의 결과를 confidence 가중으로 보간**해 최종 지도 좌표(GPS 기준 ENU)에 정렬한다.
3. **메시 보정(deltamesh).** 구간 점군을 `ingest(segment, Preview|Refined, points)`로 넣고 `extract()`가 돌려준 **바뀐 블록만** 내보낸다. 정제본은 같은 구간의 미리보기를 정확히 대체한다. 블록마다 `version`이 있어 클라이언트는 더 새 버전만 받으면 된다. stable display에서는 아직 다시 놓일 수 있는 구간(구간 3까지)을 미리보기 수준으로 넣어 두고, 구간 4가 오면 그 구간들의 마지막 점을 정제 수준으로 넣는다(정제 수준은 구간마다 한 번만 넣을 수 있다). 구간이 4개 이하인 실행은 구간 0~3의 메시가 미리보기 수준으로 남는다.

UNet 쪽 규칙:

- **학습은 Python 그대로.** `datasets/`·`models/skylensnet/`·`utils/`의 학습 코드는 전부 보존한다.
- **파일마다 짝이 되는 `.rs`를 같은 폴더에 둔다.** `modeling_skylensnet.py` 옆에 `modeling_skylensnet.rs`가 있는 식이다. Rust 쪽은 추론에 필요한 만큼을 옮기고, **PyTorch와 같은 libtorch를 `tch`로 링크**해 학습과 같은 연산을 그대로 쓴다.
- 3DGS 구현(`models/skylens/`의 COLMAP·gsplat 파이프라인)은 쓰지 않는다. 같은 폴더에서 cumulus3d 기반 Rust 구현으로 대체한다. 측정 기록(`RESULTS.md`)은 근거 자료로 남긴다.

### 3.6 `skylens_client` (Rust + 이전 HTML/CSS)

외부망 지휘관이 보는 **현황판**이다. 이전 `skylens_client`의 현황판 화면(`status.ts` · `statusview/` · `ui/`)을 옮긴 것이고, 이전의 웹서버·WebRTC 중계는 게이트웨이 nginx로 갔다.

- 복원되어 도착한 것만 보인다. 바탕 지도(지형·건물)는 없다.
- 코어의 `ws /viewer`(게이트웨이 → 프록시 경유)로 받기만 하고 제어 경로는 없다.
- 이전 패널을 그대로 갖는다: 서버 패널과 구간별 수준 사다리, 미니맵, 점·고도색 토글과 인셋, 대기 배너, 카메라 상태 칩, 탐지 카드.
- 웹(wasm), 데스크톱 네이티브, Android가 같은 코드이고 iOS는 그 뒤다. 웹 빌드는 게이트웨이 nginx가 서빙한다. Android는 cargo-apk로 만든 NativeActivity APK이고(`build-android.sh`), 진입점과 설정 읽기는 `skylens-viewer`의 `shell::run_android` · `android` 모듈에 있어 관제탑도 같은 길을 쓸 수 있다.

### 3.7 `shared` (Rust)

두 crate로 나뉜다.

- `skylens-shared`(`src/shared/`): 프로토콜과 GPS↔ENU 좌표 변환. 드론·모델·화면·데모·테스트가 같이 쓴다. 화면 의존성이 없다.
- `skylens-viewer`(`src/shared/viewer/`): 관제탑과 현황판이 같이 쓰는 화면 층. 이전 `shared/viewer/`에 해당한다. 공용 CSS(`style.css` 원본), 공용 위젯(패널 라벨 · 토스트 · 로딩 화면 · 탐지 카드), 코어 연결과 재접속, 청크 저장소, Blitz 문서 뼈대, pointblitz 3D 뷰와 카메라.

- **코어↔모델 계약은 `proto/`의 protobuf다.** Rust는 tonic/prost(protox로 컴파일해 protoc가 필요 없다), Go는 같은 `.proto`에서 생성한 코드를 쓴다. 필드를 바꾸면 `.proto` 하나만 고친다.
- 청크 `payload`(점·메시 블록·탐지 결과)의 인코딩은 `shared`가 정하고 화면이 푼다. 코어는 이 인코딩을 몰라도 된다.

좌표 변환은 `shared`(Rust)와 `skylens_model/utils/geo.py`(Python) 두 벌이고 수치적으로 동기 유지한다.

### 3.8 `test`

컴포넌트를 가로지르는 테스트를 모은다. 기능은 **테스트를 먼저 쓰고** 구현한다.

- 단위 테스트는 각 언어 관례대로 코드 옆에 둔다(Rust `#[cfg(test)]`, Go `_test.go`).
- `src/test/`에는 컴포넌트 경계 테스트(Rust crate `skylens-tests`)와 UNet 학습 코드의 pytest가 있다.

### 3.9 `demo`

전 컴포넌트를 **실제로** 띄워 엔드 투 엔드로 돌리는 런처. 한 대의 머신에서 재현할 수 없는 것만 바꾼다.

| 실제 | 데모 |
|---|---|
| 기체 카메라 → SRT | 실제 촬영본(`data/demo`, 없으면 릴리스 `demo-footage-v1`에서 받는다)을 SRT로 송출 |
| 실제 비행 | 촬영본에 기록된 GPS·고도·방위를 영상과 맞춰 재생 |
| 운용자의 경로 지정 | 촬영 경로를 관제탑 경로 계획에 미리 넣어 둔다(`planned_route`). 배정은 운용자가 누른다 |

복원·탐지·메시·배포는 데모에서도 실제 경로를 탄다. 파이프라인이 없으면 없다고 표시하고 시뮬레이션으로 메우지 않는다.

### 3.10 화면 이식 원칙 (관제탑 · 현황판 공통)

- **화면은 이전 TypeScript 화면과 같아야 한다.** 패널 배치, 문구, 색, 폰트, 애니메이션, 키 조작, 3D 모습까지 같다. 바뀌는 것은 렌더러(three.js → pointblitz)와 언어(TS → Rust)뿐이다.
- **HTML/CSS는 이전 것을 그대로 쓴다.** CSS 파일(`style.css` · `panels.css` · `control.css` · `status-panels.css`)은 수정하지 않고 가져온다. 이전 화면은 HTML 셸이 비어 있고 TS가 런타임에 DOM을 만들었으므로, 이전 앱이 실제로 만든 DOM을 상태별로 떠 와서(`.worktrees/legacy-dom/`) 같은 태그·id·class 구조를 Blitz에 넘긴다. Rust는 마크업을 새로 짓지 않고 값과 class만 바꾼다.
- **3D 뷰는 three.js 캔버스가 있던 자리에 pointblitz를 끼운다.** 이전 three.js 장면의 모습(지형 면과 위성 드레이프, 음영 프리즘 건물과 모서리선, 와이어프레임 드론과 궤적, 경로와 지상 패드, 안개와 배경색, 카메라 동작)을 pointblitz 0.2.1(crates.io)의 레이어(패스 안 그리기 · 메시 · 텍스처 · 선 · 안개)로 재현한다.
- 폰트(Oswald · Barlow · IBM Plex Mono · Noto Sans KR)는 앱에 포함해 웹과 네이티브가 같은 글꼴을 쓴다.

---

## 4. 시나리오

```
1. 데모 시작        → 시스템은 정지 상태
2. 관제탑 경로 지정  → "태스크 지정 완료"
                     "드론이 현장에 도착해 연결되면 작업이 시작됩니다" 안내
3. 드론 연결 대기 화면
4. 드론 hello 수신   → "드론 연결됨", 프레임 수신 시작
5. 구간마다 미리보기 점군·메시가 먼저 뜨고 정제본이 교체
6. UNet 결과가 프레임 사이에서 보간되어 지도 위 위험구역·사람으로 표시
```

드론의 "도착"은 hello로만 판정한다. 이동 중 텔레메트리를 접속으로 세면 연결 대기 단계가 사라진다.

---

## 5. 디렉터리 구조

```
Cargo.toml              Rust 워크스페이스 (shared · shared/viewer · drone · model · core/control · client · demo · test)
go.work                 Go 워크스페이스 (skylens_core)
pyproject.toml          Python (UNet 학습 · 데이터셋)
src/
├─ shared/              공통 계약 crate (lib.rs) + proto/ (gRPC · WebSocket 계약)
│  └─ viewer/           두 화면 공용 층 crate (공용 CSS · 위젯 · 연결 · 3D 뷰)
├─ skylens_drone/       Rust Mission Bridge (main.rs)
├─ skylens_gateway/     브리지 nginx · MediaMTX · 설치 스크립트
├─ skylens_proxy/       대전 nginx 다중 경로
├─ skylens_core/        Go 코어 서버 (go.mod · main.go · internal/ · terrain/)
│  ├─ control/          관제탑 화면 crate (이전 HTML/CSS + Rust, 웹 · 네이티브)
│  └─ deploy/           대전 MediaMTX · 설치 스크립트 · KOREN 배치 문서
├─ skylens_model/       Rust 서버(lib.rs · main.rs) + Python 학습 코드, 같은 폴더에 .py/.rs 짝
├─ skylens_client/      현황판 화면 crate (이전 HTML/CSS + Rust, 웹 · 네이티브)
├─ test/                경계 테스트 crate + pytest
└─ demo/                Rust 데모 런처
res/static/             데모 자산 (영상 · 이미지 · 구간 자산)
```

옮겨지기 전 코드는 각 컴포넌트의 `legacy/`에 있다. 빌드·테스트 대상이 아니고, 포팅이 끝난 컴포넌트부터 지운다.

---

## 6. 렌더링과 UI

| 층 | 웹 | 데스크톱 네이티브 |
|---|---|---|
| 패널·위젯 | Blitz가 HTML/CSS를 그대로 그림 (wasm) | Blitz가 같은 HTML/CSS를 그림 |
| 3D 뷰 | pointblitz-web (WebGPU / WebGL2) | pointblitz (Vulkan / DX12 / Metal) |
| 상태·이벤트 | Blitz DOM 위 Rust → wasm | Blitz DOM 위 Rust 네이티브 |
| 받는 것 | 코어가 미는 증분 청크. 관제탑은 지형·건물 포함, 현황판은 복원 결과만 | 같은 청크 |

pointblitz는 서버에서 헤드리스로 그려 영상으로 보내는 모드도 있다. GPU가 약한 단말을 위한 대안으로 남겨 둔다.

---

## 7. 포트 맵

| 컴포넌트 | 호스트 | 포트 | 엔드포인트 |
|---|---|---:|---|
| `skylens_core` | 대전 | 8080 | `ws /uplink`(드론) · `ws /control`(관제탑, 제어 가능) · `ws /viewer`(현황판, 읽기 전용) · `GET /site` · `GET /vworld/*`(키를 서버에서 붙이는 VWorld 프록시) · `GET /health` · 관제탑 웹 |
| `skylens_proxy` | 대전 | 8082 | 코어로 넘김 · `GET /health` · `GET /core/health` |
| `skylens_gateway` | 브리지 | 80 | `/` 현황판 웹 · `ws /viewer`·`ws /uplink`(터널 → 프록시) · `/live/` WHEP · `/healthz` · `/health` |
| `skylens_gateway` MediaMTX | 브리지 | 10890/udp | 드론 SRT 수신 |
| `skylens_core` MediaMTX | 대전 | 10889 · 10189/udp | WHEP 시그널링 · WebRTC 미디어 |
| `skylens_drone` | 현장 PC | 8091 | `/health` · `/missions` · `/formation` · `/cloud/*` |
| `skylens_model` | 판교 | 8100 | REST: `POST /recon/jobs` · `POST /detect/jobs` · `GET /jobs/{id}` · `GET /health` |
| `skylens_model` | 판교 | 8101 | gRPC `skylens.model.v1.ModelService` (위 네 개 + `PushFrames` · `SubscribeChunks`) |

`skylens_model`의 REST 경로와 요청·응답 스키마는 `app.py`와 같게 유지한다(`src/skylens_model/API.md`). 코어는 8101 gRPC만 쓴다.

---

## 8. 빌드

| 명령 | 하는 일 |
|---|---|
| `cargo build --workspace` | Rust 전체 |
| `cargo test --workspace` | Rust 테스트 |
| `cargo clippy --workspace --all-targets` | Rust 린트 |
| `go build ./...` (`src/skylens_core`) | Go 코어 |
| `go test ./...` (`src/skylens_core`) | Go 테스트 |
| `uv sync` · `uv run pytest` | Python 학습 코드 |
| `nginx -t` | 게이트웨이·프록시 설정 검사 (배치 호스트에서) |

- Rust는 1.99 이상(stable)이 필요하다. pointblitz 0.3이 1.99를 요구하고, 루트 `Cargo.toml`의 `rust-version`이 이를 강제한다. 웹 빌드에는 `rustup target add wasm32-unknown-unknown`.
- UNet 추론은 `tch` 0.24(libtorch 2.11)이다. libtorch는 빌드할 때 `download-libtorch`로 받아 오며 Python venv와 무관하다. CUDA 판은 `.cargo/config.toml`의 `TORCH_CUDA_VERSION`(cu126)이 정한다.
- UNet 추론 정밀도는 CUDA compute capability로 정한다(8.0 이상 bf16, 7.x 즉 V100은 fp16 autocast, 그 미만 fp32, CPU fp32). `SKYLENS_PRECISION`으로 강제하고, 비교는 `cargo run --release -p skylens-model --example precision_bench`. 자세한 내용은 `src/skylens_model/models/skylensnet/README.md` §5.
- 공용 의존성 버전은 루트 `Cargo.toml`의 `[workspace.dependencies]` 한 곳에서 정한다.
- cumulus3d는 crates.io 릴리스(0.11)를 그대로 쓴다. SkyLens의 GPU 패치(전체 GPU SIFT, CUDA BA·희소 단계)는 릴리스에 들어 있고 `.gpu()`가 고른다. 저장소에 벤더 복사본은 없다.
- cumulus3d 새 릴리스는 `.github/workflows/cumulus3d-sync.yml`이 따라간다. 버전 요구를 올리고 `skylens-model` 검사를 돌려 `deps/cumulus3d-v<버전>` PR을 연다. 흐름과 시크릿은 `docs/CUMULUS3D_SYNC.md`.
- gRPC 코드 생성: Rust는 빌드 스크립트에서 protox로 컴파일하므로 protoc가 없어도 된다. Go는 `protoc-gen-go`·`protoc-gen-go-grpc`로 `src/skylens_core/gen/`에 생성한다.

---

## 9. 구현상의 결정

**전달 서버에 코드를 두지 않는다.** 게이트웨이·프록시가 하던 일은 HTTP·WebSocket을 다른 호스트로 넘기는 것뿐이라 nginx 설정으로 충분하다. TS 프록시가 하던 능동 프로브·소켓 이전·버퍼링은 잃고, 대신 재접속·재시도를 양 끝(드론·클라이언트)이 맡는다.

**코어만 Go다.** 코어는 연산 없이 보관·스케줄·배포만 하는 I/O 서버다. Rust `shared`를 쓸 수 없으므로 모델과는 protobuf를 계약으로 삼고 gRPC로 통신한다.

**코어는 청크를 풀지 않는다.** 점·메시·탐지 결과는 모델이 만들고 화면이 푼다. 코어가 그 사이에서 디코딩하면 인코딩이 바뀔 때마다 Go 쪽도 같이 바뀌어야 한다. 메타데이터만 읽고 `payload`는 바이트로 나르면 인코딩은 Rust 두 끝(`shared`)만 안다.

**메시는 pointblitz에 렌더 경로를 더해 그린다.** 렌더러를 둘로 나누지 않고, 점과 메시가 같은 장면·카메라·청크 스트림을 쓴다.

**잡 요청의 `steps`는 스키마를 유지하고 수준 선택으로 의미를 바꾼다.** gsplat 학습 스텝이던 필드지만 `app.py`와 표면을 같게 두기 위해 남기고, 값으로 미리보기/정제 수준을 고른다. 값과 수준의 대응은 `skylens_model`이 정한다.

**모델 서버의 표면은 바꾸지 않는다.** 코어가 보는 모델은 `app.py`와 같은 네 엔드포인트다. 내부가 gsplat 학습에서 라이브 점군 + 메시로 바뀌어도 코어 쪽 계약은 그대로다.

**UNet은 복원과 같은 프레임을 본다.** 탐지를 별도 잡으로 돌리면 프레임을 두 번 디코딩하고 포즈를 따로 맞춰야 한다. `on_frame` 훅으로 붙이면 복원이 등록한 포즈로 바로 투영할 수 있다.

**프레임 하나의 탐지를 믿지 않는다.** 드론 영상은 같은 지점을 여러 프레임이 본다. 각 프레임의 결과를 confidence로 가중해 보간하므로 한 프레임의 오검출이 지도에 바로 찍히지 않는다.

**학습은 Python에 두고, 추론은 libtorch를 링크한다.** 학습 생태계(torch·데이터 로더·실험 기록)를 Rust로 옮길 이유가 없다. 추론은 `tch`로 학습과 같은 libtorch를 쓰므로 레이어를 다시 구현하거나 수치 일치를 따로 검증할 필요가 없다. 서빙 호스트(판교)에 이미 CUDA와 PyTorch가 있어 네이티브 의존성 부담도 새로 생기지 않는다.

**화면과의 WebSocket도 protobuf다.** 모든 프레임이 `skylens.viewer.v1.Envelope` 하나를 바이너리로 싣는다(`src/shared/proto/skylens/viewer/v1/viewer.proto`). 청크는 `skylens.model.v1.Chunk`를 그대로 실으므로 코어가 모델에게 받은 것을 다시 감싸지 않고 넘긴다. `payload` 인코딩은 `skylens/payload/v1/payload.proto`(메시 블록·탐지 결과)와 pointblitz 청크 포맷 v2(점·지형·건물)다.

**브리지 터널에 프록시 :8082를 더한다.** 브리지가 `-L 127.0.0.1:20882:127.0.0.1:8082`를 함께 걸고, 대전의 브리지 키 `permitopen`에 8082를 더한다. 게이트웨이 nginx는 `/viewer`·`/uplink`를 이 터널로 넘긴다.

**HTML/CSS를 원본으로 삼고 Blitz로 그린다.** Blitz가 HTML/CSS를 그대로 그리므로 웹과 네이티브가 UI를 두 벌 만들지 않는다. 웹에서도 브라우저 DOM이 아니라 Blitz를 쓰므로 이벤트 처리 코드도 한 벌이다.

**관제탑 지형·건물도 pointblitz로 그리고 서버가 증분으로 준다.** three.js를 버리므로 지형(AWS Terrain Tiles DEM)·위성 드레이프(VWorld WMTS)·건물(VWorld WFS)도 같은 렌더러를 쓴다. 코어는 셀마다 지형 타일(DEM 격자 + 위성 영상)과 건물 외곽선·높이를 원본 그대로 보내고, 모습(색 보정·압출·모서리선·표시 옵션)은 관제탑이 이전 three.js와 같은 규칙으로 만든다. 접속 직후 작전 구역부터 보내므로 관제탑은 처음부터 지도가 보인다.

### 열린 결정

지금 남은 열린 결정은 없다.


---

## 10. 이행 단계

| 단계 | 내용 | 상태 |
|---|---|---|
| 0 | 루트 JS/TS 설정 제거, Rust 워크스페이스·Go 모듈·빈 crate 세팅 | 완료 |
| 0' | 코어↔모델 gRPC 계약(`model.proto`) 작성 | 완료 (protox 컴파일 확인) |
| 1 | 폴더 재배치: `skylens_infra` 해체, `skylens_mission_bridge` → `skylens_drone`, TS 코드 → `legacy/`, Tauri 드론 앱 삭제 | 완료 |
| 2 | `shared`: proto 계약(model · viewer · payload) + 좌표계 Rust 포팅 | 완료 |
| 3 | `skylens_drone`: Mission Bridge Rust 포팅 + 코어 업링크 | 코드 완료 (테스트 57개 통과, 실기체 미검증) |
| 4 | `skylens_core`: Go 서버 + gRPC 클라이언트 + 지형·건물 pointblitz 청크 | 코드 완료 (go test 통과) |
| 5 | `skylens_model`: REST·gRPC + cumulus3d 라이브 복원 + deltamesh + UNet(tch) 훅·보간 | 코드 완료 (테스트 179개 통과, 실데이터 미검증) |
| 6 | 화면: 1차 이식은 이전 화면과 달라 다시 한다. 관제탑은 `skylens_core/control`, 현황판은 `skylens_client`, 공용 층은 `shared/viewer`로 나누고 이전 HTML/CSS·DOM을 그대로 쓴다(§3.10). 현황판 Android APK(cargo-apk, 한 손가락 회전·두 손가락 확대) 코드 추가 | 재작업 중 (Android는 기기 미검증) |
| 6' | pointblitz 레이어(패스 안 그리기 · 메시 · 텍스처 · 선 · 안개 · 배경색): pointblitz 0.2.1 에 들어가 crates.io 에서 바로 받는다(`third_party/pointblitz` 패치 제거) | 완료 |
| 7 | 게이트웨이 nginx 경로·터널 8082 | 완료 |
| 8 | `demo`: 엔드 투 엔드 런처 | 대기 |
| 9 | `legacy/` 전부 삭제 | 대기 |

---

## 11. 이번 단계에서 하지 않는 것

- DB 연결 (인메모리로 두고 세션 종료 시 소멸)
- 클라이언트 간 P2P 재분배 (코어 직접 배포로 시작)
- iOS 클라이언트, Android 관제탑 (현황판 Android 이후)
