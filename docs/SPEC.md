---
tags: [skylens, spec]
related: "[[INTENT.md]], [[COMPONENTS.md]], [[ARCHITECTURE.md]], [[NETWORK_ARCHITECTURE.md]]"
---

# SkyLens — 설계 스펙 (Spec)

> 지금 이 시스템이 **무엇으로 되어 있고 어떻게 동작하는지**만 적는다.

왜 이걸 만드는지와 무엇을 주장하는지는 [INTENT.md](INTENT.md)에 있다. 각 결정의 측정 근거와 실험 기록은 실험 PR에 있고, 여기서는 링크만 건다. "예전에는 X였다", "Y를 시도했다 버렸다"는 적지 않는다 — 그건 Git과 PR이 갖는다.

각 절은 한 줄 요약(`>`)으로 시작한다. 제목과 요약만 이어 읽으면 전체 설계가 한 장으로 보인다.

---

## 1. 파이프라인

> 드론이 보낸 영상이 게이트웨이·프록시를 거쳐 코어에 모이고, 코어가 모델 API에 잡을 발행해 받은 결과를 두 화면으로 내보낸다.

```
드론 ─▶ 게이트웨이 ─▶ 프록시 ─▶ 코어 ─▶ 모델 API
                                 │       └─▶ 복원 결과 · 탐지 마커
                                 ├─▶ 관제탑 (코어가 직접 서빙, ws /viewer 직결)
                                 └─▶ 클라이언트 릴레이 ─▶ 현황판
```

제어는 반대 방향으로만 흐른다: 관제탑 → 코어(`assign-route`, `manual-control`). 배포는 코어 → 릴레이 → 현황판 단방향이고, 현황판에서 코어로 가는 경로는 의도적으로 없다.

## 2. 컴포넌트 경계

> 저장소는 망 위치를 따라 8개 컴포넌트로 갈라지고, 각 컴포넌트는 `src/` 아래 한 디렉터리를 차지한다.

| 컴포넌트 | 망 위치 | 책임 | 스택 |
|---|---|---|---|
| `skylens_drone` | 현장 | 촬영 영상을 H.265 세그먼트로 잘라 전송 + 텔레메트리 | Tauri (Rust 셸 + TS UI) |
| `skylens_gateway` | KOREN 외부망 | 내부망 진입점. `relay` / `webrtc` 2모드 | Node + TS |
| `skylens_proxy` | KOREN 내부망 | 이중화·다중화된 접속 경로. 코어로 전달 | Node + TS |
| `skylens_core` | KOREN 내부망 | 관제탑 서빙 + 전체 데이터 보관 + 잡 오케스트레이션 + 배포 | Node + TS / Vite |
| `skylens_model` | KOREN 내부망 | 연산 API. 3DGS 복원 + 탐지 추론 | Python · FastAPI |
| `skylens_client` | KOREN 외부망 | 현황판 웹서버 + WebRTC 시그널링 중계 | Node + TS / Vite |
| `shared` | — | 컴포넌트 공통 계약. 배포 단위가 아닌 의존 라이브러리 | TS 라이브러리 |
| `demo` | — | 컴포넌트를 조립해 모킹 모드로 기동하는 런처 | Node + TS |

경계와 각 책임의 자세한 내용은 [COMPONENTS.md](COMPONENTS.md)가 갖는다.

### 2.1 shared의 2층 구조

> 순수층은 DOM·Three 의존이 없어 Node·브라우저·Tauri 어디서나 import 되고, 브라우저 전용 코드는 그 아래 하위 디렉터리로 내린다.

- 순수층: `geo.ts` · `protocol.ts` · `types.ts`
- `viewer/` — 두 웹 UI가 함께 쓰는 브라우저 층(씬 소스·스토어·설정·공용 위젯)
- `net/` — WebRTC 트랜스포트

## 3. 통신 프로토콜

> 모든 메시지는 `Envelope`로 감싼 태그드 유니온이고, 정의는 `src/shared/protocol.ts` 한 곳에만 있다.

| 방향 | 타입 | 메시지 |
|---|---|---|
| 업링크 (드론→코어) | `UplinkMessage` | `drone-hello` · `telemetry` · `video-segment` |
| 제어 (관제탑→코어) | `ControlMessage` | `assign-route` · `manual-control` |
| 뷰어 (코어→화면) | `ViewerMessage` | `mission-status` · `splat-chunk` · `camera-feed` · `detection` · `link-status` · `server-status` |
| 잡 (코어↔모델) | — | `ReconJobRequest` · `DetectJobRequest` → `JobStatus` → `recon-result` · `detect-result` |

- 미션 단계는 `idle → assigned → awaiting-drone → active` 네 값이다.
- 코어→클라이언트 배포는 현재 **WebSocket**이다. 코어는 `Distributor` 인터페이스 뒤에서 밀고, 브라우저를 향한 WebRTC 중계와 시그널링은 `skylens_client`가 세운다. 클라이언트 간 P2P 재분배가 붙을 자리도 이 시그널링이다.

## 4. 좌표계

> 실세계 GPS가 1급 좌표계이고, 씬 좌표로의 변환은 렌더 경계에서만 일어난다.

- `GeoAnchor`(기준 GPS, `CONFIG.geo.anchor`)를 원점으로 하는 **로컬 ENU(동/북/상) 미터** 프레임. 1 unit = 1 m.
- 드론 경로는 GPS로 지정 → ENU → 씬. 탐지 결과도 GPS로 수신 → 씬 좌표로 변환해 마커 배치.
- 각 스플랫 청크는 명시적 align transform(pos/rot/scale, 선택적 GPS anchor)을 갖고 공통 ENU 프레임에 정렬된다.
- 구현은 `src/skylens_core/geo.ts`와 `src/skylens_model/utils/geo.py` **두 벌**이다. 같은 수식의 두 구현이므로 한쪽만 고치지 않는다.

## 5. 딜레이 패턴 복원

> 구간마다 낮은 학습 스텝의 결과를 먼저 확정해 내보내고 뒤이어 정제한다. 스케줄 결정권은 코어에만 있다.

| 수준 | 학습 스텝 | 지휘관이 확인 가능한 것 |
|---|---:|---|
| 1 | 250 | 형상 윤곽만 식별 |
| 2 | 1,000 | 공간 구조 식별 가능 |
| 3 | 3,500 | 표면 형성 |
| 4 | 7,000 | 진입 동선 판단이 가능한 실용 품질 |

- **트리거는 시간이 아니라 드론의 이동량이다.** 드론이 한 구간을 통과하면 그 구간의 수준 1이 확정되고 이후 수준이 뒤따른다.
- 새 수준이 도착하면 같은 구간의 낮은 수준을 **교체**한다(누적하지 않음). 이미 추월당한 수준은 건너뛴다.
- 클라이언트는 도착한 것을 받을 뿐이다. 클라이언트가 타이머로 스스로 단계를 진행시키지 않는다.
- 타이밍 값은 `CONFIG.delayPattern`(구간 주기·수준별 지연)에 있다.

## 6. 관제탑 화면

> VWorld 실지형 한 갈래로 통합했고, 좌표는 GPS를 직접 쓴다.

- 건물 표시는 별도 씬이 아니라 표시 옵션이다: **점 / 검정 텍스처(기본값) / 실사 항공뷰**.
- 지형 메시는 AWS Terrain Tiles DEM(한국 약 30m급), 위성 드레이프는 VWorld WMTS, 3D 건물은 VWorld WFS `lt_c_bldginfo` 폴리곤을 지붕까지 프리즘으로 렌더한다.
- 월드 스트리밍: 드론 반경 내 미로드 셀을 가까운 순으로 로드하고, 씬 주변에 3배 저해상 배경 지형 링을 둔다.
- 기준 씬은 대전(충남대~카이스트 일대, 약 3km, 약 6,191동).
- 조작: 경로 계획 모달에서 GPS 웨이포인트를 추가해 배정. 방향키 수동 조향, Q/E 고도, 1/2/3·Tab 드론 전환, Space 일시정지.
- VWorld 키는 저장소 **부모 폴더**의 `.env.vworld`에서 읽고 Vite dev 프록시가 서버측에서만 주입한다. 프론트엔드 번들에 노출되지 않는다. 키가 없으면 지형·건물 없이 기존 씬만 동작한다.

## 7. 현황판 화면

> 구간 청크가 도착해야 그 구간이 보인다. "보인다 = 복원되어 도착했다"가 하나의 진실이다.

- 노출(reveal)은 드론 궤적이 아니라 **도착 기준**이다.
- 탐지 마커도 해당 구간이 복원돼야 표시된다.
- 좌측 상단 서버 패널에 구간별 현재 수준이 뜬다.
- 데모 자산이 없으면 단일 장면 스트림으로 자동 폴백한다.

## 8. AI 모델

> RGB와 열화상을 입력단에서 합친 4채널 단일 백본에 헤드 둘을 달아, 위험구역과 사람을 함께 낸다.

- 백본: UNet, 입력 4채널(RGB 3 + 열 1).
- 세그 헤드 — 위험구역(stuff). 통합 클래스 스키마는 `0 normal / 1 fire / 2 collapse / 3 road_blocked / 255 ignore`.
- 점 검출 헤드 — 사람.
- modality dropout으로 열화상이 없는 입력도 견딘다.
- 헤드별로 분리 학습한다.
- **`road_blocked` 픽셀이 있는 학습 이미지는 오버샘플링한다.** 배수는 4다. 이 클래스는 학습 표본의 약 2%에만 나타나 대부분의 배치에 아예 들어오지 않는다. 근거: experiment/road-oversample.
- **오버샘플링할 때 fire_seg 도 같은 배수로 불려 세그 데이터셋 구성비를 보존한다.** RescueNet만 불리면 fire의 유일한 출처인 fire_seg의 비중이 24.2%에서 17.8%로 희석되어 `fire`가 학습되지 않는다. 근거: experiment/balanced-oversample.
- 모달리티 융합은 latent 융합이 아니라 **Hybrid Fusion**이다. 합쳐지는 단계가 모달리티마다 다르다: 영상+열화상은 입력단, 포즈는 투영단, 소리는 결정단.
- 설계 결정과 그 근거의 단일 출처는 `src/skylens_model/README.md`다. 상위 문서와 어긋나면 모델 문서가 우선한다.

## 9. 2D→3D 투영

> Depth Map 레이캐스팅(핀홀 역투영)으로 2D 탐지 결과를 세계좌표 마커로 바꾼다.

드론 포즈(extrinsics)와 깊이만 있으면 되고, 별도의 3D 탐지 모델을 두지 않는다. 산출은 Core HPC(`skylens_model`)에서 하고 코어는 GPS 좌표로 받는다.

## 10. 3DGS 복원 파이프라인

> 프레임 추출 → COLMAP SfM → gsplat 학습 3단계이고, 특징점은 ALIKED를 쓴다.

- COLMAP 4.1(ALIKED + LightGlue 내장). 실내 저텍스처 장면에서 기본 SIFT는 촬영본을 여러 모델로 쪼갠다.
- 전수 매칭을 쓴다.
- 내부 파라미터를 명시하고 자기보정을 함께 건다.
- 점진적 출력으로 첫 화면을 약 30초에 낸다.
- 전송용 경량 PLY를 따로 낸다.
- 구간 분할은 장면의 주축(복도 촬영이면 드론 진행 방향)을 기준 파일의 분위수로 잘라 **모든 수준에 동일하게** 적용한다. 같은 구간 번호는 항상 같은 공간 조각을 가리킨다.
- 설계 근거는 `src/skylens_model/models/skylens/README.md`, 측정값은 `RESULTS.md`, 설치는 `INSTALL.md`가 갖는다.

## 11. 데모 모드

> 컴포넌트는 전부 실제 코드이고, 한 대의 머신에서 재현할 수 없는 두 가지만 바뀐다.

| | 실운영 | 데모 |
|---|---|---|
| 드론 촬영 | 기체 카메라 → H.265 실시간 인코딩 | `res/static/video/h265`의 사전 인코딩 영상 |
| 3D 복원 | Core HPC에서 gsplat 학습 | `res/static/demo`의 사전 제작 구간×수준 자산 |

나머지는 전부 실제 경로다. 화면은 도착한 것만 그리고, 파이프라인이 없으면 **없다고 표시**한다(`PipelineUnavailable`). 시뮬레이션으로 메우지 않는다.

시나리오: 정지 상태 → 관제탑 경로 지정("태스크 지정 완료") → 드론 연결 대기(10초) → "드론 연결됨", 주행 시작 → 지정 경로 왕복 → 이동량에 따라 현황판 갱신.

드론의 "도착"은 `drone-hello`로만 판정한다. 드론은 이동 중에도 텔레메트리를 보내므로, 텔레메트리를 접속으로 세면 연결 대기 단계가 사라진다.

## 12. 배포 (KOREN 실회선)

> 브리지·대전·판교 세 호스트에 컴포넌트를 얹고, 스플랫 경로와 라이브 영상 경로를 분리한다.

| 호스트 | 망 | 올라가는 컴포넌트 | 추가로 도는 것 |
|---|---|---|---|
| 브리지 `175.126.98.44` | KOREN 외부망 · 등록 IP | `skylens_gateway` :8081 · `skylens_client` :8090 | nginx :80 · MediaMTX SRT :10890 · 대전행 SSH 터널(autossh) |
| 대전 `116.89.187.181` | KOREN 내부망 | `skylens_proxy` :8082 · `skylens_core` :8080 | MediaMTX WHEP :10889 / ICE :10189 |
| 판교 `10.246.246.9` | KOREN 내부망 · V100 | `skylens_model` :8100 | 대전과 터널로만 통신 |

- 스플랫·마커는 §1의 경로를 그대로 탄다. **라이브 영상은 대전이 현황판에 WebRTC(WHEP)로 직접 쏜다** — 영상은 캐시가 안 되고 접속자 수만큼 대역이 필요해 브리지가 나르면 브리지 회선이 상한이 된다. 대전 출구는 4스트림 2,618 Mbps 실측.
- 진입은 SRT, 배포는 WebRTC. MediaMTX가 SRT 수신과 WHEP 송출을 모두 내장이라 영상 서버 코드가 0줄이다.
- 브리지가 대전으로 SSH 터널(`-L 127.0.0.1:20889:127.0.0.1:10889`)을 건다. 대전에는 브리지 키를 `restrict,permitopen="127.0.0.1:10889"`로 등록해 키가 새도 그 포트 외에는 못 쓰게 한다.
- 서버 측 STUN·TURN을 두지 않는다. 대전·판교는 공인 IP 직결이라 자기 주소를 알고, 단말은 구글 공개 STUN을 쓴다.
- 자세한 설정과 근거는 `src/skylens_infra/README.md`.

## 13. 포트 맵

> 컴포넌트 포트는 8080~8100 대역이고, 밖으로 노출되는 건 브리지의 80 · 8081 · 10890뿐이다.

| 컴포넌트 | 포트 | 엔드포인트 |
|---|---:|---|
| `skylens_core` | 8080 | `ws /uplink` · `ws /viewer` · `GET /health` · 관제탑 서빙 |
| `skylens_gateway` | 8081 | 드론 접속 · `GET /health` |
| `skylens_proxy` | 8082 | 게이트웨이/드론 접속 · `GET /health` |
| `skylens_client` | 8090 | 현황판 서빙 · WebRTC 시그널링 · `GET /health` |
| `skylens_model` | 8100 | `POST /recon/jobs` · `POST /detect/jobs` · `GET /jobs/{id}` · `GET /health` |
| Vite 개발 서버 | 5173 | 두 웹 UI의 개발 모드 원본 |

## 14. 저장과 내구성

> 이번 단계에는 DB가 없다. 코어가 세그먼트·복원 결과·마커·텔레메트리를 인메모리로 들고 있고, 세션이 끝나면 사라진다.

- 프록시는 코어 장애 시 대기 경로로 전환하고 복구되면 돌아온다.
- 모델 API가 없으면 코어가 잡을 재제출한다.
- 코어가 없으면 현황판은 마지막 상태를 유지한다.

## 15. 빌드와 테스트

> TypeScript는 npm, Python은 uv로 관리한다.

| 명령 | 하는 일 |
|---|---|
| `npm run demo` | 전 컴포넌트를 데모 모드로 조립 기동 |
| `npm run dev` | Vite 개발 서버만 |
| `npm run core` / `client` / `gateway` / `proxy` / `drone` / `model` | 컴포넌트 개별 기동 |
| `npm run build` | `tsc` 타입체크 + 멀티페이지 빌드 |
| `npm test` | Playwright E2E |
| `uv sync` / `uv run pytest` | Python 의존성 설치 / 테스트 |
| `uv run ruff check src/` | Python 린트 (line-length 100, target py311) |

- Python 3.11 이상 3.14 미만. `.python-version`은 3.13.
- torch·torchvision은 PyPI가 아니라 CUDA 12.8 인덱스에서 받는다(`[tool.uv.sources]`).
- `pyproject.toml`은 자동 탐색 대신 `packages = ["src/skylens_model"]`로 파이썬 패키지를 명시한다.
- 정적 html 셸(`index` · `control` · `status`)은 `res/static/`에 모여 있고, 접속 URL은 `/res/static/{control,status}.html`이다. `/`는 랜딩을 자동 서빙하지 않는다.

## 16. 브랜치와 연구 기록

> `develop`이 ResearchTree 루트 브랜치이고, 실험은 `experiment/*` 브랜치와 그 PR이 기록한다.

- 루트 브랜치는 `develop`이다. ResearchTree의 기본값은 `research`이므로 **`RESEARCHTREE_ROOT=develop`을 환경변수로 준다** — 이 값은 `.researchtree.yml`로 지정할 수 없다.
- 실험 브랜치 접두사는 기본값 `experiment/`를 그대로 쓴다.
- `.researchtree.yml`이 `spec: docs/SPEC.md` · `intent: docs/INTENT.md`를 가리킨다.
- 버전 태그는 `develop/v1`, `develop/v2` … 형태다(루트 브랜치 이름이 접두사).
- `main`은 배포용 브랜치다. `scripts/sync-main.sh`가 `develop`을 `main`에 머지하면서 설계 문서(`docs/*.md`와 루트 `.md` 중 `README.md` 제외)를 걷어낸다. `docs/` 하위 폴더(`figures/` · `materials/`)는 그대로 둔다.
