---
tags: [skylens, spec]
related: "[[INTENT.md]], [[COMPONENTS.md]], [[ARCHITECTURE.md]], [[NETWORK_ARCHITECTURE.md]]"
---

# SkyLens 설계 스펙 (Spec)

> 지금 이 시스템이 **무엇으로 되어 있고 어떻게 동작하는지**만 적는다.

왜 이걸 만드는지와 무엇을 주장하는지는 [INTENT.md](INTENT.md)에 있다. 각 결정의 측정 근거와 실험 기록은 실험 PR에 있고, 여기서는 링크만 건다. "예전에는 X였다", "Y를 시도했다 버렸다"는 적지 않는다. 그건 Git과 PR이 갖는다.

각 절은 한 줄 요약(`>`)으로 시작한다. 제목과 요약만 이어 읽으면 전체 설계가 한 장으로 보인다.

---

## 1. 파이프라인

> 기체 영상이 브리지를 거쳐 대전 코어에 모이고, 코어가 판교 모델 서버에 프레임을 넘겨 받은 증분 청크를 두 화면으로 내보낸다.

```
관제탑 ─제어─▶ 코어 ─▶ 드론(Mission Bridge) ─▶ DJI 클라우드 ─▶ 기체
기체 ─SRT─▶ 게이트웨이(브리지) ◀─당김─ 코어(대전 MediaMTX) ─WHEP─▶ 현황판 라이브 영상
                                     └─gRPC 프레임─▶ 모델(판교) ─gRPC 증분 청크─▶ 코어
코어 ─▶ 관제탑 (코어가 직접 서빙, ws /control 직결, 제어는 이 소켓으로만)
코어 ─▶ 프록시 ─▶ 게이트웨이 ─▶ 현황판
```

제어는 관제탑 → 코어 한 방향이다. 현황판에서 코어로 가는 제어 경로는 의도적으로 없다.

## 2. 컴포넌트 경계

> 저장소는 망 위치를 따라 컴포넌트로 갈라지고, 각 컴포넌트는 `src/` 아래 한 디렉터리를 차지한다. 연산과 화면은 Rust, 코어는 Go, 전달 서버는 nginx 설정이다.

| 컴포넌트 | 망 위치 | 책임 | 스택 |
|---|---|---|---|
| `skylens_drone` | 현장 | GPS 경로 → DJI Fly KMZ 임무, 편대 3기 전개, DJI 클라우드 업로드 | Rust |
| `skylens_gateway` | KOREN 외부망 (브리지) | 외부 진입. 현황판 웹·WHEP 시그널링·드론 SRT 수신 | nginx + MediaMTX |
| `skylens_proxy` | KOREN 내부망 (대전) | 코어 경로 이중화 | nginx |
| `skylens_core` | KOREN 내부망 (대전) | 코어 서버(보관 + 오케스트레이션 + 배포 + 라이브 영상 원점)와 관제탑 화면(코어가 웹으로 제공) | Go 서버 + Rust 관제탑 + MediaMTX |
| `skylens_model` | KOREN 내부망 (판교) | 라이브 점군(cumulus3d) + UNet 프레임 훅 + 메시 보정(deltamesh) | Rust 서버 + Python 학습 |
| `skylens_client` | KOREN 외부망 (지휘관 단말) | 현황판 화면. 복원된 것만 보이고 제어 경로가 없다 | Rust + 이전 HTML/CSS |
| `shared` | 해당 없음 | 공통 계약(proto · 좌표계)과 두 화면 공용 층(`viewer/`) | Rust 라이브러리 |
| `test` | 해당 없음 | 컴포넌트 경계 테스트 (TDD) | Rust + pytest |
| `demo` | 해당 없음 | 전 컴포넌트를 띄우는 엔드 투 엔드 런처 | Rust |

경계와 각 책임의 자세한 내용, 그리고 컴포넌트별 이행 상태는 [COMPONENTS.md](COMPONENTS.md)가 갖는다. 옮겨지기 전 코드는 각 컴포넌트의 `legacy/`에 있고 빌드되지 않는다.

## 3. 통신 프로토콜

> 코어↔모델은 gRPC, 코어↔드론·화면은 protobuf를 싣는 WebSocket이다. 계약은 `src/shared/proto` 하나이고, 코어는 청크 내용을 풀지 않고 메타데이터만 읽는다.

| 방향 | 타입 | 메시지 |
|---|---|---|
| 업링크 (드론→코어) | `UplinkMessage` | `drone-hello` · `telemetry` |
| 제어 (관제탑→코어) | `ControlMessage` | `assign-route` · `manual-control` |
| 뷰어 (코어→화면) | `ViewerMessage` | `assign-route` · `mission-status` · `point-chunk` · `mesh-block` · `detection` · `telemetry` · `camera-feed` · `link-status` · `server-status` |
| 관제탑 전용 (코어→관제탑) | `ViewerMessage` | `planned-route`(미리 계획된 경로, 배정 전까지) · 바탕 지도(지형·건물 청크) |
| 잡 (코어↔모델, gRPC) | `ModelService` | `SubmitRecon` · `SubmitDetect` · `GetJob` · `Health` (REST와 같은 잡 API) |
| 라이브 (코어↔모델, gRPC) | `ModelService` | `PushFrames`(프레임 묶음 스트림) · `SubscribeChunks`(증분 청크 스트림) |

- 미션 단계는 `idle → assigned → awaiting-drone → active` 네 값이다. 마지막 드론이 끊기면 `awaiting-drone` 으로 돌아간다(`idle` 이 아니다).
- **경로 배정이 새 미션의 시작이다.** 드론이 떠 있지 않은 동안(`active` 가 아닌 동안) 경로가 배정되면 코어는 지난 미션이 복원한 것을 먼저 버린다. 점·메시·탐지·점 슬롯 스트림이 대상이고, 바탕 지도(지형·건물)와 텔레메트리·영상·경로·링크 상태는 남는다. 모델 구독의 이어받을 지점(`after_sequence`)도 그대로 둔다. 되돌리면 모델이 방금 버린 실행을 처음부터 다시 보내 도로 채운다.
- **새 미션은 모델에게도 알린다.** 코어가 `StartMission` 을 부르면 모델은 엔진을 새로 만들고, 새 스트림 id 로 1번 메시지부터 다시 쓴다. 라이브 점은 차분의 연속이라 받은 쪽이 처음부터 적용해야 그림이 되고 모델은 지난 메시지를 다시 보내지 않으므로, **화면을 비울 수 있는 것은 모델뿐이다.** 코어는 응답을 기다리지 않는다(엔진 생성은 GPU 를 쓰고, 부르는 쪽은 화면이 붙어 있는 소켓이다). 새 스트림은 청크 구독으로 들어온다.
- 모델이 죽어 있거나 이 호출을 모르면 미션은 그대로 진행하고 로그만 남긴다. 영상과 드론은 보이고, 라이브 점은 모델이 돌아올 때까지 비어 있다.
- 그래도 앞부분이 사라진 스트림을 받는 경우(옛 모델, 또는 코어만 먼저 올라간 경우)를 대비해, **그 스트림의 처음 오는 메시지를 새 시작으로 삼는다**(코어와 화면 같은 규칙). 1번을 기다리면 그 모델 실행이 끝날 때까지 라이브 점이 멈춘다. 실제로 2026-10-17 에 그렇게 멈췄다. 다만 이렇게 이어받으면 이미 지나간 add 를 되살리지는 못하므로 화면은 거의 비어 있다. 제대로 비우고 다시 채우는 길은 위의 `StartMission` 뿐이다.
- 화면은 자기가 받은 것을 스스로 들고 있으므로, 비우는 신호는 **화면 소켓을 끊는 것**이다. 양쪽 화면은 재접속 때 자기 사본을 비우고 리플레이를 다시 받는다(`src/shared/viewer/chunks.rs`). 버릴 것이 없으면 소켓을 끊지 않는다. 드론이 떠 있는 동안의 재배정은 비행 중 재계획이라 화면을 그대로 둔다.
- `planned-route`는 배정이 아니다. 코어가 `SKYLENS_CORE_PLANNED_ROUTE` 파일에서 읽어, 아직 경로가 배정되지 않았으면 관제탑이 붙을 때 작전 구역 바로 다음에 보낸다. 관제탑은 경로 계획이 비어 있을 때만 그 지점을 채우고, 운용자가 배정을 눌러야 `assign-route`가 나간다.
- 코어↔모델 계약은 `src/shared/proto/skylens/model/v1/model.proto` 하나다. Rust는 tonic/prost, Go는 같은 파일에서 생성한 코드를 쓴다.
- 청크(`Chunk`)는 라우팅 메타데이터(종류·구간·수준·블록 id·version·세대)와 불투명한 `payload`로 되어 있다. 코어는 메타데이터만 읽고 `payload`는 그대로 보관·전달한다. `payload` 인코딩은 `shared`가 정하고 화면이 푼다.
- 모델 서버는 `app.py`와 같은 REST 네 엔드포인트도 그대로 연다. 코어는 gRPC만 쓴다.
- `ws /uplink`·`ws /viewer`의 모든 프레임은 바이너리 `skylens.viewer.v1.Envelope` 하나다. 청크는 모델이 준 `Chunk`를 그대로 싣는다.
- `payload` 인코딩: 메시 블록·탐지 결과는 `skylens/payload/v1/payload.proto`, 점·지형·건물은 pointblitz 청크 포맷 v2, 점 슬롯(`CHUNK_KIND_POINT_SLOTS`)은 pointblitz 슬롯 차분 메시지(32바이트 `PBSD` 머리 + cumulus3d live diff stream v4 팩 그대로). 점 슬롯 청크는 `segment` = 사건을 낸 구간, `level` = 정제, `version` = 메시지 순번(1부터), `generation` = 스트림 id(모델 실행마다 하나, 앞선 실행보다 늘 크다. 새 id면 처음부터)다. 모델 청크의 `sequence`는 실행마다 스트림 id를 위 32비트에 둔 값에서 시작하므로, 모델 서버가 다시 뜨면 코어가 이어받는 시점(`after_sequence`)보다 새 실행의 청크가 모두 크다.
- 영상 자체는 이 프로토콜을 타지 않는다. SRT(진입)와 WHEP(배포)로 따로 간다.

## 4. 좌표계

> 실세계 GPS가 1급 좌표계이고, 씬 좌표로의 변환은 렌더 경계에서만 일어난다.

- `GeoAnchor`(기준 GPS)를 원점으로 하는 **로컬 ENU(동/북/상) 미터** 프레임. 1 unit = 1 m.
- 드론 경로는 GPS로 지정 → ENU → 씬. 탐지 결과도 GPS로 받아 씬 좌표로 바꿔 마커를 둔다.
- 점군은 cumulus3d가 GPS로 정렬한 ENU 좌표로 나오고, 모든 구간이 같은 원점을 쓴다(고정 ENU 원점).
- 구현은 `src/shared`(Rust)와 `src/skylens_model/utils/geo.py`(Python) **두 벌**이다. 같은 수식의 두 구현이므로 한쪽만 고치지 않는다.

## 5. 점진적 복원

> 라이브 복원은 cumulus3d stable display로 돈다. 구간은 정제본으로 한 번 보이면 제자리에 있고, 점은 바뀐 것만(추가·이동·제거) 점 슬롯 차분으로 나간다.

| 경로 | 출처 | 화면에서 하는 일 |
|---|---|---|
| 점 슬롯 (기본, stable display) | cumulus3d `Event::DisplayDelta`(메모리 표시 델타), 모델 서버가 슬롯 메시지로 씀 | 슬롯 장면 하나에 사건 순서대로 적용한다(추가·이동은 칸을 쓰고 보이게, 제거는 숨김). 빠진 순번이 있으면 기다린다 |
| 정제 구간 (stable display) | cumulus3d `ZoneRefined` | 메시(deltamesh)와 탐지 융합의 입력. 점은 슬롯으로 나가므로 점 청크는 없다 |
| 미리보기·정제 점 청크 (`SKYLENS_STABLE_DISPLAY=0`, 조밀화 없음, 잡) | `ZonePreview` / `ZoneRefined` | 미리보기는 점을 덧붙이고, 정제본이 같은 구간을 세대째 교체한다 |

- **트리거는 시간이 아니라 들어온 프레임이다.** 구간(zone)에 프레임이 차면 그 구간의 정제가 시작되고, 구간 K의 정제는 K−1을 기다린다. stable display의 미리보기는 선택이다(`SKYLENS_ZONE_PREVIEW`, 기본 끔): 켜면 구간이 닫히는 즉시 희소 점이 `slot_kind` 4로 먼저 나가고, 그 구간의 정제 델타가 대체한다.
- 구간은 GPS 위치 제약(σ 1 m)으로 ENU에 놓여 한 번 보이면 움직이지 않는다. 예외로 내부값이 안정되기 전(초점 변화 1 % 미만, 늦어도 `SKYLENS_SETTLE_MAX_ZONE` 구간까지, 기본 1)에는 앞 구간을 다시 놓거나 다시 조밀화하고, 그 결과는 `replace` 사건(대부분 이동)으로 나간다.
- 점 슬롯은 화면용으로 솎은 점(`SKYLENS_LIVE_STRIDE`, 기본 2: 둘에 하나)이고 구간 겹침은 지운 것이다.
- 점 슬롯 청크의 `slot_kind`는 메시지가 왜 나왔는지를 말한다: 1 정제(refined) · 2 재배치(replace) · 3 최종(final, 빈 메시지, 실행의 마지막) · 4 미리보기(preview). `slot_t_ms`는 cumulus3d 세션 시각(실행 시작 기준 ms)이다. 코어는 둘 다 읽지 않고, 현황판의 라이브 패널(델타 로그 타임라인)이 PREVIEW · REFINED · REPLACE · FINAL 배지로 보여 준다. 메시는 합성된 표시(구간 겹침이 지워진 점)에서 만든다.
- 메시는 바뀐 블록만 나가고, 블록마다 `version`이 있어 화면은 더 새 버전만 받는다. 삼각형이 0개인 블록은 지우라는 뜻이다. 표시 델타가 바뀔 때마다 바뀐 구간의 표시 점을 미리보기로 다시 넣으므로 블록 청크의 `level`은 미리보기다.
- 조밀화가 없으면(CUDA 없음) 차분 스트림이 비므로 정제 구간의 희소 점을 점 청크로 보낸다. 조밀화가 실행 중 실패한 구간(`ZoneRefined`의 `dense`가 거짓)도 차분 스트림에 실리지 않으므로 그 구간만 점 청크로 보낸다.
- 화면은 도착한 것을 받을 뿐이다. 화면이 타이머로 스스로 단계를 진행시키지 않는다.

## 6. 관제탑 화면

> 관제탑은 코어의 화면이다. 코어가 웹으로 서빙하고 `ws /control`에 직결한다. 화면은 이전 TS 관제탑과 같다.

- 코드는 `src/skylens_core/control/`이다. 패널·위젯은 이전 HTML/CSS를 그대로 Blitz가 그리고, 3D 뷰는 pointblitz다.
- 지형은 위성 사진을 드레이프한 면, 건물은 음영 프리즘과 모서리선, 드론은 와이어프레임 팔면체와 궤적, 경로는 금색 선과 지상 패드다. 드론이 없으면 개요 회전, 있으면 선택한 드론을 쫓는 카메라다.
- 경로 계획은 위성 지도 클릭이고, 고도는 지면 기준(AGL)이며 DEM으로 절대 고도로 바꿔 보낸다.
- 바탕 지도는 접속 직후 작전 구역(`GET /site`)부터 받는다. 기준점(ENU 원점)은 코어가 정하고 화면은 그것을 따른다.
- 건물 표시는 별도 씬이 아니라 표시 옵션이다: **점 / 검정 텍스처(기본값) / 실사 항공뷰**.
- 지형은 AWS Terrain Tiles DEM(한국 약 30m급), 위성 드레이프는 VWorld WMTS, 3D 건물은 VWorld WFS `lt_c_bldginfo` 폴리곤이다. 코어가 이것을 pointblitz 청크로 바꿔 화면이 보는 영역부터 증분으로 내보내고, 화면은 복원 결과와 같은 pointblitz로 그린다.
- 기준 씬은 대전(충남대~카이스트 일대, 약 3km, 약 6,191동).
- 조작: 경로 계획 모달에서 GPS 웨이포인트를 추가해 배정한다. 배정된 경로는 `skylens_drone`이 KMZ 임무로 바꾼다.
- VWorld 키는 저장소 **부모 폴더**의 `.env.vworld`에서 코어가 읽어 서버측에서만 주입한다. 화면 번들에 노출되지 않는다. 키가 없으면 지형·건물 없이 동작한다.

## 7. 현황판 화면

> 구간 청크가 도착해야 그 구간이 보인다. "보인다 = 복원되어 도착했다"가 하나의 진실이다.

- 코드는 `src/skylens_client/`이다. 이전 HTML/CSS를 Blitz로 그대로 그리고, 웹(wasm), 데스크톱 네이티브, Android가 같은 코드다. iOS는 후순위다.
- 코어의 `ws /viewer`(읽기 전용)로 받는다. 바탕 지도는 받지 않는다.
- 노출(reveal)은 드론 궤적이 아니라 **도착 기준**이다. 탐지 마커도 해당 구간이 복원돼야 표시된다. 점 슬롯에서는 추가 레코드의 구간 바이트가 점의 구간이고, 그 구간의 첫 점이 보이면 그 구간이 정제 수준으로 도착한 것이다.
- 점 청크 구간은 도착하면 서서히 나타나지만, 점 슬롯은 슬롯 장면 하나라 사건이 오면 바로 불투명하게 보인다. 메시 블록의 구간별 서서히 나타나기는 그대로다.
- 라이브 영상은 대전 MediaMTX에서 WHEP로 직접 받는다.
- 좌측 상단 서버 패널에 구간별 현재 수준이 뜬다.
- 파이프라인이 없으면 없다고 표시한다(`PipelineUnavailable`).

## 8. AI 모델

> RGB와 열화상을 입력단에서 합친 4채널 단일 백본에 헤드 둘을 달아, 위험구역과 사람을 함께 낸다.

- 백본: UNet, 입력 4채널(RGB 3 + 열 1).
- 세그 헤드는 위험구역(stuff). 통합 클래스 스키마는 `0 normal / 1 fire / 2 collapse / 3 road_blocked / 255 ignore`.
- 점 검출 헤드는 사람을 맡는다.
- modality dropout으로 열화상이 없는 입력도 견딘다.
- 헤드별로 분리 학습한다.
- **`road_blocked` 픽셀이 있는 학습 이미지는 오버샘플링한다.** 배수는 4다. 이 클래스는 학습 표본의 약 2%에만 나타나 대부분의 배치에 아예 들어오지 않는다. 근거: experiment/road-oversample.
- **오버샘플링할 때 fire_seg 도 같은 배수로 불려 세그 데이터셋 구성비를 보존한다.** RescueNet만 불리면 fire의 유일한 출처인 fire_seg의 비중이 24.2%에서 17.8%로 희석되어 `fire`가 학습되지 않는다. 근거: experiment/balanced-oversample.
- 모달리티 융합은 latent 융합이 아니라 **Hybrid Fusion**이다. 합쳐지는 단계가 모달리티마다 다르다: 영상+열화상은 입력단, 포즈는 투영단, 소리는 결정단.
- 세그 손실은 cross-entropy 와 soft Dice 를 1:1 로 더한다. Dice 는 ignore(255) 픽셀을 분자·분모에서 뺀다. 근거: experiment/seg-dice-loss (#33).
- 사람 헤드는 히트맵과 상자 크기(wh)가 3×3 stem 을 공유하지 않는다. 히트맵은 `person_stem`, wh 는 `wh_stem` 을 따로 거친다. 근거: experiment/combo-dice-wh (#38).
- **상자 크기 가지는 공유 특징을 읽기만 하고 기울기를 되돌려보내지 않는다**(`wh_detach`). `wh_stem` 은 전용이지만 그 입력은 공유 디코더 출력이라, 상자 손실 가중치를 올리면 기울기가 백본까지 흘러 사람 히트맵과 위험구역을 깎았다. 입력을 detach 하면 `wh_stem` 과 `wh_head` 는 정상적으로 학습되면서 백본은 상자 손실을 보지 않는다. 근거: experiment/wh-detach (#72).
- 사람 학습 표본 중 VisDrone 은 이미지를 짧은 변 765 로 맞춘 원본 스케일에서 512×512 무작위 크롭 3 장으로 만든다. 추론도 같은 스케일의 512 타일(64px 이상 겹침)로 하고, 타일 경계의 중복 검출은 병합한다. LLVIP 는 전체 프레임을 512 로 줄여 쓴다. 근거: experiment/combo-visdrone-tiles (#39).
- 재난 수색 드론 데이터셋 SARD(누운 사람 · 탈진 · 부상 자세 포함)를 사람 학습에 넣는다. VisDrone 과 같은 원본 스케일 타일 규칙을 따른다(`--sard tiles`). 분할은 영상 프레임 번호 50 장 단위 구간으로 나누고 경계 10 장을 뺀다(`data/sard/SPLIT.json`). 근거: experiment/sard-person-tiles (#48).
- 사람 헤드는 히트맵 stem 위에 서브픽셀 오프셋 헤드를 둔다. 검출 중심은 `(셀 + 예측 오프셋) × stride` 다. 근거: experiment/tiles-offset-head (#43).
- 사람 검출은 실제 추론 방식으로 평가한다. LLVIP 전체 프레임과 VisDrone · SARD 원본 스케일 타일의 검출을 한 precision-recall 로 합산하고(`pooled_target`), 지표는 점수 0.05 이상 검출로, F1 은 임계값 0.3 으로 잰다. **판정하는 네 지표는 전부 점 기준이다**(정밀도 0.5 이상에서 재현율 · 최저 임계값 재현율 · F1@0.3 · 점 AP). 박스 mAP@50 과 mAP@50:95 는 계속 계산하고 보고하지만 통과 여부를 정하지 않는다(`REFERENCE_TARGETS`). 근거: experiment/point-ap-primary (#73). 점 매칭 반경은 `max(8px, 0.15 × 박스 높이)` 이고 고정 8px 결과도 함께 적는다. 오프셋이 없는 모델의 중심은 `(셀 + 0.5) × stride` 다. 구현은 `scripts/eval_deploy.py`. 근거: experiment/eval-decode-fix (#44), experiment/eval-size-relative-match (#52).
- 학습은 5 에폭 이상으로 판정한다. road_blocked 는 cross-entropy 단독에서 3 에폭 이후에야 나타난다. 근거: experiment/balanced-5ep (#29).
- **학습은 Python, 추론은 Rust다.** 학습 코드의 각 `.py` 옆에 같은 이름의 `.rs`를 둔다. Rust 추론은 PyTorch와 같은 libtorch를 `tch`로 링크한다.
- 설계 결정과 그 근거의 단일 출처는 `src/skylens_model/README.md`다. 상위 문서와 어긋나면 모델 문서가 우선한다.

## 9. 2D→3D 투영과 프레임 간 보간

> UNet은 복원 노드의 프레임 훅으로 붙고, 프레임마다의 결과를 등록된 포즈로 점군에 투영해 confidence 가중으로 합친다.

- 훅은 cumulus3d `on_frame`이다. 위치 계산 전에 디코딩된 RGB를 받고 별도 큐에서 돌아 복원을 막지 않는다.
- 투영은 복원이 등록한 카메라 포즈와 점군 깊이로 한다. 별도의 3D 탐지 모델을 두지 않는다.
- 한 지점을 본 여러 프레임의 위험구역 확률·사람 점을 confidence로 가중해 보간한 뒤 최종 지도(ENU)에 정렬한다. 한 프레임의 오검출이 바로 찍히지 않는다.
- 산출은 `skylens_model`이 하고 코어는 GPS 좌표로 받는다.

## 10. 3D 복원 파이프라인

> cumulus3d `CompositionHost`의 라이브 입력으로 점군을 키우고, deltamesh가 구간 점군을 블록 메시로 보정해 바뀐 블록만 내보낸다.

- 잡 요청의 `steps`는 `app.py`와 스키마를 같게 두려고 남긴 필드이고, 값으로 미리보기/정제 수준을 고른다. 대응은 `skylens_model`이 정한다.
- 노드는 `ReconNode::declare()`로 선언하고 `CompositionHost`가 id로 들고 있는다. 프레임 묶음(`LiveFrameSet`, 키는 `FrameKey(드론, 순번, 개정)`)을 `compose`하면 새 묶음만 한 번 들어가고, 재선언은 무시되며, 카메라가 빠진 묶음은 기다린다.
- 늦게 오거나 고쳐진 프레임은 노드의 reconcile 정책(append-only / replay window / zone invalidation / strict)을 따른다.
- 구간은 `zones(span, overlap)`으로 나눈다. 같은 구간 번호는 항상 같은 공간 조각을 가리킨다.
- 메시는 deltamesh `SdfMesher`(GPU는 `GpuMesher`, 프로세스당 장치 하나)다. 0.1 m 복셀(deltamesh 기본값 0.2 m의 절반, 점 묶음 0.05 m·영향 반경 0.2 m도 함께 절반. 0.05 m는 현황판에서 오히려 품질이 떨어져 되돌렸다), 32³ 블록. 경계 정점이 이웃 블록과 비트 단위로 같아 균열 없이 이어진다.
- deltamesh는 crates.io 릴리스(0.1.1)를 쓴다. 유니폼 간격을 GPU 정렬값의 배수로 맞추는 수정(정렬값 64바이트 GPU에서 첫 구간이 멈추던 문제)이 릴리스에 들어 있다. 벤더 복사본은 없다.
- 렌더링은 pointblitz(0.3)다. 점 청크는 16비트 위치(포맷 v2)이고 파싱은 메인 스레드 밖에서 한다. 점 슬롯은 `SlotScene` 하나(칸 = 슬롯 id, 칸당 16바이트)에 쌓고, 같은 패스에서 점 청크 장면들과 함께 그린다. 메시 블록은 pointblitz에 더한 메시 렌더 경로가 정점 색에 흰 빛(고른 빛 0.85 + 위에서 오는 빛 0.15)만 얹어 그린다. 현황판의 메시 표시는 메시를 바닥면으로 깔고 그 위에 점을 함께 그린다(세부와 색은 점, 점 사이 빈틈은 메시). 점 표시는 메시를 그리지 않는다.
- cumulus3d는 crates.io 릴리스(0.11)를 쓴다. 전체 GPU SIFT, CUDA BA·희소 단계는 릴리스에 들어 있고 `.gpu()`가 고른다. 벤더 복사본은 없다. 새 릴리스는 `cumulus3d-sync.yml`이 `deps/cumulus3d-v<버전>` PR로 따라간다(`docs/CUMULUS3D_SYNC.md`).
- 라이브 노드 기본값: 특징 선행(lookahead) 4, 백로그 묶음(`SKYLENS_BACKLOG_BATCH`) 1 = 끔, 구간 미리보기(`SKYLENS_ZONE_PREVIEW`) 끔, 표시 솎기(`SKYLENS_LIVE_STRIDE`) 2. 묶음을 끈 이유: cumulus3d 0.11.0에서 batch > 1이면 V100 라이브의 표시 축척이 약 30배 줄었다(원인 미확인. cumulus3d 0.11.1이 노드 기본값을 1로 바꾸고 축척 회귀 테스트를 더했지만, 이 저장소의 락은 0.11.0이라 `SKYLENS_BACKLOG_BATCH` 기본값으로 직접 1을 준다). 카메라 내부값은 `SKYLENS_CAMERA_JSON`이 없으면 내장 DJI FC9589 1920×1080(fx 1505 · fy 1510 · cx 960 · cy 540)이다.
- 점 슬롯 스트림 id는 모델 실행마다 늘어난다(시작 시각 초, 같은 프로세스에서는 앞선 id + 1 이상). 코어와 화면은 더 큰 id의 첫 메시지가 오면 옛 실행의 메시지를 버리고 빈 점 버퍼에서 시작한다. 라이브 노드는 파일을 쓰지 않는다.
- 이전 복원 방식(COLMAP·gsplat)의 측정 기록은 `src/skylens_model/models/skylens/RESULTS.md`에 근거로 남는다.

## 11. 데모 모드

> 컴포넌트는 전부 실제 코드이고, 한 대의 머신에서 재현할 수 없는 기체만 바뀐다.

| | 실운영 | 데모 |
|---|---|---|
| 촬영 | 기체 카메라 → SRT | 실제 촬영본(`data/demo`, 릴리스 `demo-footage-v1`에서 자동으로 받는다)을 SRT로 송출. 기본은 첫 비행(높은 1차, 약 53초)만 재생하고, `--full-footage`면 낮은 2차까지 재생한다 |
| 비행 | DJI Fly 웨이포인트 | 촬영본에 기록된 GPS·고도·방위를 영상과 맞춰 재생 |
| 경로 계획 | 운용자가 지도에서 지정 | 촬영 경로가 관제탑 경로 계획에 미리 들어가 있다(`planned-route`). 운용자는 배정만 누른다 |

복원·탐지·메시·배포는 데모에서도 실제 경로다. 파이프라인이 없으면 **없다고 표시**한다(`PipelineUnavailable`). 시뮬레이션으로 메우지 않는다.

시나리오: 정지 상태 → 관제탑 경로 배정("태스크 지정 완료") → 드론 연결 대기 → "드론 연결됨", 프레임 수신 시작 → 구간마다 정제본이 자리 잡음(미리보기는 `SKYLENS_ZONE_PREVIEW`를 켰을 때만 먼저 보인다) → 위험구역·사람 표시.

- `--online[=HOST]`는 같은 런처를 실배치 위에서 돌린다. 영상은 브리지(파이) MediaMTX로, 드론 시뮬레이터는 대전 코어로 붙고 이 머신에서는 아무것도 띄우지 않는다. `--auto`·`--full-footage`를 포함한다.
- `--online`은 영상을 피드마다 2500 kbps로 재인코딩한다(`--video-kbps`). 원본 약 10 Mbps×3이 VPN 상행을 채워 드론 소켓이 끊겼다. 드론 업링크가 끊기면 백오프로 다시 붙는다.
- 2026-10-18 18:50 `--online` 실측: 2분에 드론 3대 ACTIVE, 프레임 431, 구간 5, 탐지 6. 첫 구간은 약 38초에 보이고 다음 구간은 17~24초 간격, 53초 비행은 약 5분에 완성된다. 처리량은 아직 영상보다 느리다(`DEPLOYMENT.md` §12.9).

드론의 "도착"은 `drone-hello`로만 판정한다. 드론은 이동 중에도 텔레메트리를 보내므로, 텔레메트리를 접속으로 세면 연결 대기 단계가 사라진다.

## 12. 배포 (KOREN 실회선)

> 브리지·대전·판교 세 호스트에 컴포넌트를 얹고, 3D·마커 경로와 라이브 영상 경로를 분리한다.

| 호스트 | 망 | 올라가는 것 |
|---|---|---|
| 브리지 `175.126.98.44` | KOREN 외부망 · 등록 IP | `skylens_gateway`: nginx :80 · MediaMTX SRT :10890 · 대전행 SSH 터널(autossh) |
| 대전 `116.89.187.181` | KOREN 내부망 | `skylens_proxy` nginx :8082 · `skylens_core` :8080 · MediaMTX WHEP :10889 / ICE :10189 |
| 판교 `10.246.246.9` | KOREN 내부망 · V100 | `skylens_model` :8100 (대전과 터널로만 통신) |

- **라이브 영상은 대전이 현황판에 WebRTC(WHEP)로 직접 쏜다.** 영상은 캐시가 안 되고 접속자 수만큼 대역이 필요해 브리지가 나르면 브리지 회선이 상한이 된다. 대전 출구는 4스트림 2,618 Mbps 실측.
- 진입은 SRT, 배포는 WebRTC. MediaMTX가 둘 다 내장이라 영상 서버 코드가 0줄이다.
- 브리지가 대전으로 SSH 터널 둘(`-L 127.0.0.1:20889:127.0.0.1:10889` WHEP, `-L 127.0.0.1:20882:127.0.0.1:8082` 프록시)을 건다. 대전에는 브리지 키를 `restrict,permitopen="127.0.0.1:10889",permitopen="127.0.0.1:8082"`로 등록해 키가 새도 그 두 포트 외에는 못 쓰게 한다.
- 서버 측 STUN·TURN을 두지 않는다. 대전·판교는 공인 IP 직결이라 자기 주소를 알고, 단말은 구글 공개 STUN을 쓴다.
- 자세한 설정과 근거는 `src/skylens_core/deploy/README.md`, 브리지 설치는 `src/skylens_gateway/README.md`, 다중 경로는 `src/skylens_proxy/README.md`.

## 13. 포트 맵

> 밖으로 노출되는 건 브리지의 80 · 10890뿐이다.

| 컴포넌트 | 포트 | 엔드포인트 |
|---|---:|---|
| `skylens_core` | 8080 | `ws /uplink` · `ws /control` · `ws /viewer` · `GET /site` · `GET /vworld/*` · `GET /health` · 관제탑 웹 |
| `skylens_proxy` | 8082 | 코어로 넘김 · `GET /health` · `GET /core/health` |
| `skylens_gateway` | 80 | `/` 현황판 웹 · `ws /viewer`·`ws /uplink`(터널 20882 → 프록시) · `/live/{path}/whep` WHEP · `/healthz` · `/health` |
| `skylens_drone` | 8091 | `/health` · `/missions` · `/formation` · `/cloud/*` |
| `skylens_model` | 8100 | REST `POST /recon/jobs` · `POST /detect/jobs` · `GET /jobs/{id}` · `GET /health` (`app.py`와 같음) |
| `skylens_model` | 8101 | gRPC `skylens.model.v1.ModelService` (코어가 쓰는 쪽) |

## 14. 저장과 내구성

> 이번 단계에는 DB가 없다. 코어가 프레임 메타·청크·마커·텔레메트리를 인메모리로 들고 있고, 세션이 끝나면 사라진다.

- 점 슬롯 메시지는 합치지 않고 현재 스트림의 것을 순번대로 모두 들고 있다가, 화면이 붙으면 처음부터 다시 보낸다(theater30 한 번에 약 125 MB). 더 새 스트림 id가 오면 이전 스트림을 버린다. 빠진 순번 뒤에 들고 있는 메시지는 64개, 256 MiB까지다. 넘으면 들고 있던 것을 버리고 로그를 남긴다(스트림의 처음을 놓친 경우로, 빈 곳이 채워지지 않는다).
- 점 슬롯 메시지 하나는 사건 하나라 크다(theater30 최대 14.6 MB, 화면용 점을 모두 쓰면 약 90 MB). 코어의 화면 소켓은 프레임마다 15초에 1 MiB/s로 보내는 시간을 더해 기다리고, 실시간 적체는 512 MiB까지 둔다. 네이티브 화면은 메시지·프레임을 512 MiB까지 받고, 모델 gRPC는 256 MiB까지다.
- 프록시는 코어 선호 경로가 죽으면 대기 경로로 넘기고, 일정 시간 뒤 선호 경로를 다시 시도한다. 열린 소켓은 옮기지 않으므로 드론·화면이 재접속한다.
- 모델 서버가 없으면 코어가 잡을 재제출한다.
- 코어가 없으면 화면은 마지막 상태를 유지한다.
- 코어를 다시 띄우면 들고 있던 것이 사라진다. 다만 모델 구독이 처음부터 다시 받으므로 지난 실행의 청크가 다시 쌓인다. 깨끗한 화면으로 시작하려면 코어 재시작이 아니라 **경로를 새로 배정**한다(§3).

## 15. 빌드와 테스트

> Rust는 cargo 워크스페이스, Go는 go.work, Python은 uv로 관리한다.

| 명령 | 하는 일 |
|---|---|
| `cargo build --workspace` / `cargo test --workspace` | Rust 빌드 / 테스트 |
| `cargo clippy --workspace --all-targets` | Rust 린트 |
| `go build ./...` / `go test ./...` (`src/skylens_core`) | Go 코어 빌드 / 테스트 |
| `uv sync` / `uv run pytest` | Python 의존성 설치 / 학습 코드 테스트 |
| `uv run ruff check src/` | Python 린트 (line-length 100, target py311) |

- gRPC 코드 생성: Rust는 protox(순수 Rust)로 컴파일해 protoc가 필요 없고, Go는 `protoc-gen-go`·`protoc-gen-go-grpc`로 `src/skylens_core/gen/`에 생성한다.
- Rust는 1.99 이상 stable이 필요하다(루트 `Cargo.toml`의 `rust-version`). 웹 빌드에는 `wasm32-unknown-unknown` 타깃. 공용 의존성 버전은 `[workspace.dependencies]`에 있다.
- UNet 추론은 `tch` 0.24(libtorch 2.11). libtorch는 빌드 때 내려받고(`download-libtorch`, CUDA 판은 `.cargo/config.toml`의 `TORCH_CUDA_VERSION`) Python venv와 무관하다.
- 테스트를 먼저 쓴다. 단위 테스트는 코드 옆(Rust `#[cfg(test)]`, Go `_test.go`), 컴포넌트 경계 테스트는 `src/test/`.
- Python 3.11 이상 3.14 미만. `.python-version`은 3.13. torch·torchvision은 CUDA 12.8 인덱스에서 받는다(`[tool.uv.sources]`).
- `pyproject.toml`은 `packages = ["src/skylens_model"]`로 파이썬 패키지를 명시한다. 같은 폴더의 `.rs` 파일은 Rust crate `skylens-model`에 속한다.
- nginx 설정은 배치 호스트에서 `nginx -t`로 검사한다. 개발 머신에는 nginx가 없어도 된다.

## 16. 브랜치와 연구 기록

> `develop`이 ResearchTree 루트 브랜치이고, 실험은 `experiment/*` 브랜치와 그 PR이 기록한다.

- 실험 브랜치 접두사는 기본값 `experiment/`를 그대로 쓴다.
- 설정은 저장소 루트의 `.researchtree` 파일 하나에 있다. `root: develop` · `spec: docs/SPEC.md` · `intent: docs/INTENT.md`.
  뷰어·CLI·Python API가 같은 파일을 읽으므로 환경변수를 쓰지 않는다. 뷰어가 읽으려면 기본 브랜치 `main`에도 있어야 하고, `scripts/sync-main.sh`가 같이 넘긴다.
- 실험 워크트리는 저장소 안 `.worktrees/<이름>`에 만든다. 이 경로는 `.gitignore` 대상이다.
- 버전 태그는 `develop/v1`, `develop/v2` … 형태다(루트 브랜치 이름이 접두사).
- `main`은 배포용 브랜치다. `scripts/sync-main.sh`가 `develop`을 `main`에 머지하면서 설계 문서(`docs/*.md`와 루트 `.md` 중 `README.md` 제외)를 걷어낸다. `docs/` 하위 폴더(`figures/` · `materials/` · `site/`)는 그대로 둔다. 지운 문서를 가리키던 상대 링크는 `develop` 절대 URL로 바뀐다.
