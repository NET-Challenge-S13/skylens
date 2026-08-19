---
tags: [skylens, model, api]
links: "[[COMPONENTS]] · [[ARCHITECTURE]]"
---

# skylens_model 연산 API

`skylens_model` 컴포넌트의 REST 표면. **코어가 발행한 잡을 받아 처리하고 결과를 돌려주는 것**이
전부다([COMPONENTS.md §3.5](../../res/docs/COMPONENTS.md)). 시스템에서 유일한 요청/응답 창구이며,
나머지 경로는 전부 푸시다.

- 진입점: `src/skylens_model/app.py` (`skylens_model.app:app`)
- 와이어 계약의 단일 출처: `src/shared/protocol.ts` **§7 Model API**
  — `src/skylens_model/serving/schemas.py`가 그 인터페이스를 **필드명 그대로**(camelCase) 미러링한다.
  protocol.ts §7이 바뀌면 이 파일도 같이 바뀌어야 한다.
- 잡은 **인메모리**로만 산다. 프로세스가 죽으면 잡 기록도 사라진다 —
  COMPONENTS.md §7의 결정이지 누락이 아니다.

---

## 1. 실행

```bash
uv sync --group serve                                   # fastapi + uvicorn[standard]

# 데모 모드 (학습하지 않고 미리 만들어 둔 자산을 돌려준다)
SKYLENS_DEMO=1 uv run uvicorn skylens_model.app:app --port 8100

# 실모드
uv run uvicorn skylens_model.app:app --port 8100
```

OpenAPI 문서는 `http://127.0.0.1:8100/docs`.

### 환경변수

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `SKYLENS_DEMO` | `0` | `1`이면 데모 모드 |
| `SKYLENS_DEMO_MANIFEST` | `<repo>/res/static/demo/segments.json` | 구간×수준 매니페스트 경로 |
| `SKYLENS_DEMO_URL_BASE` | `/res/static/demo` | 결과 `url`에 붙는 접두사 (클라이언트가 받아가는 경로) |
| `SKYLENS_DEMO_STEP_SECONDS` | `0.0006` | 시뮬레이션 지연 = 요청 steps × 이 값 |
| `SKYLENS_DEMO_MIN_SECONDS` / `_MAX_SECONDS` | `0.3` / `12.0` | 그 지연의 하한·상한 |
| `SKYLENS_ANCHOR` | `36.3685,127.3475,30` | ENU 원점 GPS. `shared/viewer/config.ts`의 anchor와 맞춘다 |

---

## 2. 엔드포인트

| 메서드 | 경로 | 요청 | 응답 |
|---|---|---|---|
| `POST` | `/recon/jobs` | `ReconJobRequest` | `JobAccepted` |
| `POST` | `/detect/jobs` | `DetectJobRequest` | `JobAccepted` |
| `GET` | `/jobs/{job_id}` | — | `JobStatus` (없으면 404) |
| `GET` | `/health` | — | 생존 확인 + torch/CUDA 가용성 |

요청 본문은 **추가 필드를 거부**한다(`extra=forbid`). 코어가 필드명을 바꾸면 조용히 무시되는 대신
422로 즉시 드러난다.

### 2.1 잡 실행 모델

- 잡은 **한 번에 하나씩, 접수 순서대로** 돈다. 실제 워크로드(gsplat 학습, UNet 추론)는 둘 다 GPU를
  통째로 원하므로 겹쳐 돌릴 이유가 없다. 덕분에 `JobAccepted.queued`는 "내 앞에 몇 개 있는가"라는
  정직한 숫자가 된다(0이면 바로 시작).
- 상태 전이: `queued → running → done | failed`. `progress`는 0..1, `result`는 `done`일 때만 채워진다.
- 잡 하나가 실패해도 워커는 죽지 않는다. 실패 사유는 `error`에 문자열로 담긴다.
- 완료된 잡은 512건까지 보관하고 오래된 것부터 버린다(인메모리).

### 2.2 `POST /recon/jobs`

```jsonc
{
  "segment": 0,                    // 복원할 촬영 구간
  "steps": 1000,                   // 이 수준에서 돌릴 학습 스텝
  "anchorFrame": null,             // 첫 잡은 null, 이후는 첫 잡이 돌려준 id
  "sources": [ { "uri": "...", "poses": [ /* DroneTelemetry */ ] } ]
}
```

응답 `JobAccepted` → `{ "jobId": "recon-…", "queued": 0 }`.
완료 시 `JobStatus.result`는 `ReconJobResult`(`kind: "recon-result"`):
`segment · steps · url · bytes · align · anchorFrame`.

### 2.3 `POST /detect/jobs`

`{ "segment": 2, "sources": [...] }` → 완료 시 `DetectJobResult`(`kind: "detect-result"`),
`detections`는 `DetectionResult` 배열(`category`는 `person | danger`, 좌표는 GPS).

### 2.4 `GET /health`

```json
{"status":"ok","version":"0.1.0","demo":true,"torch":true,"cuda":true,
 "device":"NVIDIA GeForce RTX 4050 Laptop GPU",
 "jobs":{"queued":0,"running":0,"done":3,"failed":0,"frames":1}}
```

torch/CUDA 조회는 프로세스당 **한 번만** 하고 캐시한다(torch import가 수 초 걸린다).

---

## 3. anchorFrame — 좌표계 고정

**첫 복원 잡이 좌표계를 정하고, 이후 모든 잡이 그 좌표계를 강제로 물려받는다.**

- 첫 잡: `anchorFrame: null` → 결과에 `anchorFrame: "frame-…"`가 담겨 온다.
- 이후 잡: 그 id를 `anchorFrame`에 그대로 실어 보낸다.

왜 필요한가 — gsplat 데이터로더는 `normalize=True`일 때 **카메라 집합으로부터** 정규화 변환을
계산한다. 다음 구간의 카메라가 늘면 변환이 새로 계산되고, 실측에서 75장 기준과 150장 기준이
**87.8° 틀어졌다**. 앞 구간에서 맞춰 둔 가우시안이 통째로 회전된 공간에 떨어진다
(중간보고서 Ⅲ-1-바, [models/skylens/README.md §5](models/skylens/README.md)).
그래서 첫 잡의 변환을 저장해 두고 이후 잡에 강제한다. 그 저장된 변환에 붙은 이름이 frame id다.

데모 모드는 이미 한 장면에서 잘라낸 자산을 쓰므로 변환을 다시 유도하지 않는다.
**배관과 계약만 그대로 지킨다** — id를 발급하고, 받은 id를 재사용하고, 결과에 실어 돌려준다.

> 서버가 재시작되면 frame 테이블이 비어 있다. 이때 모르는 id가 들어오면 **거부하지 않고 그 id를
> 그대로 채택**하고 경고 로그를 남긴다. 코어는 한 비행 내내 같은 id를 쓰고 있고, 그 일관성이
> frame의 존재 이유이기 때문이다.

---

## 4. 데모 모드 (`SKYLENS_DEMO=1`)

### 4.1 복원 잡 — 아무것도 학습하지 않는다

1. `res/static/demo/segments.json`을 읽는다. 이 매니페스트는
   `python -m skylens_model.models.skylens.split_segments`가 만든다.
2. 요청한 `segment`의 수준 목록에서 **`steps` 이하 중 가장 높은 수준**을 고른다.
   요청이 가장 싼 수준보다도 낮으면 가장 싼 것을 준다.
   (예: `steps=7000` → step07000 자산, `steps=300` → step00250 자산)
3. 요청 `steps`에 비례한 시간만큼 **지연**시키며 `progress`를 올린다.
   지연의 근거는 자산의 스텝 수가 아니라 요청한 스텝 수다 — 딜레이 패턴에서 중요한 것은
   "이 수준이 지휘관을 얼마나 기다리게 하는가"이기 때문이다.
4. 파일이 **디스크에 실제로 있는지 확인**하고, 실제 바이트 수와 함께 `ReconJobResult`를 돌려준다.
   매니페스트에는 있는데 파일이 없으면 잡을 실패시킨다(가짜 크기를 지어내지 않는다).

매니페스트의 구간 수(현재 4)보다 큰 `segment`가 오면 **나머지 연산으로 순환**시킨다.
데모 도중 스트림이 끊기는 것보다 낫다고 판단했고, 순환할 때마다 경고 로그를 남긴다.

### 4.2 탐지 잡 — 고정된 마커

구간당 2건(`person` 1 · `danger` 1)을 고정으로 돌려준다. 좌표는 `SKYLENS_ANCHOR` 기준
ENU 오프셋을 `utils/geo.py`로 GPS 변환한 값이고, 구간마다 동쪽으로 12 m씩 밀어 마커가
한 점에 겹치지 않게 한다. id는 `det-s{segment}-{i}`로 결정적이다.

---

## 5. 실모드 — 지금 무엇이 연결돼 있고 무엇이 아닌가

연산 진입점은 `serving/pipeline.py`의 **함수 두 개뿐**이다: `run_recon()` · `run_detect()`.
실제 파이프라인을 붙이는 작업은 이 두 함수 안쪽 한 곳을 고치는 일이다.

| 잡 | 데모 모드 | 실모드 |
|---|---|---|
| recon | ✅ 실제 자산 해석 + 지연 시뮬레이션 | ❌ `PipelineUnavailable` — 이음매(seam)만 있음 |
| detect | ✅ 고정 마커 | ❌ `PipelineUnavailable` |

실모드가 아직 비어 있는 이유를 **각 함수의 독스트링에 붙일 작업 순서와 함께** 적어 두었다.

- **복원**: COLMAP(ALIKED+LightGlue, 전수매칭) → gsplat 학습 → 경량 PLY 추출. 분 단위로 걸리고
  `recon` 의존성 그룹과 CUDA COLMAP 빌드를 요구한다([INSTALL.md](models/skylens/INSTALL.md)).
  **요청 핸들러 안에서 돌 수 있는 작업이 아니다** — 별도 워커 프로세스가 필요하고, 이 코루틴은
  그것을 await 하는 형태가 된다. 그 워커가 생기기 전까지는 지어내지 않고 실패시킨다.
- **탐지**: 학습된 SkyLensNet 체크포인트가 저장소에 없고, 2D→3D 투영 레이어(depth 레이캐스팅,
  [README.md](README.md) 레이어 ②)가 아직 구현되지 않았다. 가짜 마커를 돌려주면 현황판에서
  진짜와 구분되지 않으므로 실패시킨다.

실패한 잡의 `error`에는 무엇이 없어서 실패했는지가 그대로 담긴다.

```
PipelineUnavailable: live 3DGS reconstruction is not wired in this process
(missing deps: gsplat, pycolmap; `uv sync --group recon`); requested segment=0
steps=1000 frame=frame-… sources=0. Run with SKYLENS_DEMO=1 for the prebuilt assets.
```

---

## 6. 확인해 본 호출

```bash
SKYLENS_DEMO=1 uv run uvicorn skylens_model.app:app --port 8100 &

curl -s localhost:8100/health

curl -s -X POST localhost:8100/recon/jobs -H 'content-type: application/json' \
  -d '{"segment":0,"steps":1000,"anchorFrame":null,"sources":[]}'
# {"jobId":"recon-946c21d2cc34","queued":0}

curl -s localhost:8100/jobs/recon-946c21d2cc34
# {"state":"done","progress":1.0,"result":{"kind":"recon-result","segment":0,"steps":1000,
#  "url":"/res/static/demo/segments/seg0_step01000.ply","bytes":216072,
#  "align":{…},"anchorFrame":"frame-572ed139ed58"},"error":null}

curl -s -X POST localhost:8100/detect/jobs -H 'content-type: application/json' \
  -d '{"segment":2,"sources":[]}'
```
