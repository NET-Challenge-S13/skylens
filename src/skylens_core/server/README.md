---
tags: [skylens, core, server]
links: "[[COMPONENTS]] · [[ARCHITECTURE]] · [[PROJECT]]"
---

# skylens_core / server — 오케스트레이터

`skylens_core` 컴포넌트의 **서버 절반**. 관제탑 UI(`control.ts` · `controlview/` · `control/` · `ui/`)는
컴포넌트 루트에 그대로 있고, 이 디렉터리는 [COMPONENTS.md §3.4](../../../res/docs/COMPONENTS.md)의
네 가지 책임만 맡는다.

| 책임 | 사는 곳 |
|---|---|
| 1. 관제탑 화면 서버 | `web.ts` |
| 2. 데이터 전체 보관 (**인메모리**) | `store.ts` |
| 3. 작업 오케스트레이터 (**딜레이 패턴**) | `segmenter.ts` → `ingest.ts` → `orchestrator.ts` |
| 4. 배포 | `distributor.ts` |

```
프록시 ─ws /uplink─→ Ingest ─→ Store ─→ Orchestrator ─REST─→ 모델 API(8100)
                        │                    │
                        │                    └─→ Distributor ─ws /viewer─→ 뷰어
                        └─ 구간 닫힘(이동량) ──┘
```

---

## 1. 실행

```bash
npx tsx src/skylens_core/server/index.ts

# 관제탑 UI 없이 헤드리스로 (Vite를 안 띄웠을 때)
SKYLENS_CORE_WEB_MODE=off npx tsx src/skylens_core/server/index.ts
```

모델 API가 필요하다(없어도 죽지는 않는다, §5):

```bash
SKYLENS_DEMO=1 uv run uvicorn skylens_model.app:app --port 8100
```

### 파일

| 파일 | 역할 |
|---|---|
| `index.ts` | 진입점. HTTP + 두 개의 WebSocket 경로를 세우고 아래 객체들을 배선한다 |
| `config.ts` | 환경변수 → `CoreConfig`. 코드를 고치지 않고 데모 런처가 포트를 바꿀 수 있게 |
| `ingest.ts` | 업링크 수신. **구간이 언제 닫히는지 결정한다** |
| `segmenter.ts` | GPS → 경로 위 호 길이(arc length) → 구간 인덱스. 시계가 없다 |
| `store.ts` | 드론·구간·청크·마커·텔레메트리. 전부 메모리 |
| `ladder.ts` | 수준 사다리(중간보고서 표 8·표 9). 수준 번호 ↔ 학습 스텝 ↔ 문구 |
| `orchestrator.ts` | **딜레이 패턴**. 큐·우선순위·드롭·anchorFrame |
| `modelClient.ts` | 모델 API REST 클라이언트(protocol §7) |
| `distributor.ts` | 뷰어 팬아웃 + 역방향 제어. **WebRTC 이음매**(§6) |
| `mission.ts` | `idle → assigned → awaiting-drone → active` |
| `web.ts` | 관제탑 화면 서빙(dev 역프록시 / prod 정적) |

검증용 스크립트는 저장소 규칙에 따라 `src/test/core/`에 있다(§7).

---

## 2. 엔드포인트 · 포트

포트 **8080** ([COMPONENTS.md §7](../../../res/docs/COMPONENTS.md) 포트 맵).

| 경로 | 방향 | 내용 |
|---|---|---|
| `ws /uplink` | 프록시 → 코어 | `Envelope<UplinkMessage>` — `drone-hello` · `telemetry` · `video-segment`. 프록시가 올리는 `link-status`도 받는다 |
| `ws /uplink` | 코어 → 프록시 | `Envelope<ControlMessage>` — `assign-route` · `manual-control`. 프록시가 드론으로 내려보낸다 |
| `ws /viewer` | 코어 → 뷰어 | `Envelope<ViewerMessage>` — `mission-status` · `splat-chunk` · `detection` · `telemetry` · `server-status` · `link-status` |
| `ws /viewer` | 뷰어 → 코어 | `ControlMessage`(맨몸 또는 Envelope, 둘 다 받는다) |
| `GET /health` | — | 미션·구간·큐·모델 API·전송 카운터 전체 |
| 그 외 | — | 관제탑 화면 (`SKYLENS_CORE_WEB_MODE`) |

**뷰어가 붙으면 현재 상태를 전부 리플레이한다** — 미션 상태 → 드론별 마지막 텔레메트리 →
구간별 최신 청크 → 마커 → 서버 상태. 늦게 들어온 현황판이 빈 화면을 보지 않게 하기 위해서다.

**드론이 늦게 붙어도 경로를 받는다** — 업링크가 새로 붙으면 지정돼 있던 `assign-route`를 그
소켓에 다시 내려보낸다. 데모 시나리오가 "경로 지정 → 드론 도착" 순서이기 때문이다.

> **콜드 스타트 주의.** 코어는 `npx tsx`로 뜨는 데 2~3초가 걸리고, 이는 프록시의 기동 유예
> (`SKYLENS_PROXY_STARTUP_GRACE_MS`, 기본 500 ms)보다 길다. 그래서 둘을 동시에 켜면 프록시가
> 잠깐 대기 경로를 승격했다가 코어가 뜨면 **failback** 한다. 프록시 쪽에서 의도된 동작으로
> 확인·문서화됐다. 이 왕복이 싫으면 코어를 먼저 띄우거나 유예를 늘린다.

---

## 3. 환경변수

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `SKYLENS_CORE_HOST` / `SKYLENS_CORE_PORT` | `0.0.0.0` / `8080` | 리슨 주소 |
| `SKYLENS_CORE_UPLINK_PATH` / `_VIEWER_PATH` | `/uplink` / `/viewer` | ws 경로 |
| `SKYLENS_MODEL_URL` | `http://localhost:8100` | 모델 API 베이스 URL |
| `SKYLENS_CORE_MODEL_POLL_MS` | `500` | `GET /jobs/{id}` 폴링 간격 |
| `SKYLENS_CORE_MODEL_JOB_TIMEOUT_MS` | `900000` | 잡 하나의 상한 |
| `SKYLENS_CORE_MODEL_RETRY_MS` | `3000` | 전송 실패 후 재시도 백오프 |
| `SKYLENS_CORE_MODEL_MAX_ATTEMPTS` | `20` | 이 횟수를 넘으면 잡을 포기한다 |
| **`SKYLENS_CORE_SEGMENT_METERS`** | `40` | **구간 하나의 경로 길이(m). 딜레이 패턴의 눈금** |
| **`SKYLENS_CORE_LEVEL_STEPS`** | `1000,7000,30000` | **수준 사다리(학습 스텝, 오름차순)** |
| **`SKYLENS_CORE_RECON_CONCURRENCY`** | `2` | **동시에 띄우는 복원 잡 수. 2 이상이어야 겹침이 생긴다** |
| `SKYLENS_CORE_DETECT_CONCURRENCY` | `1` | 탐지 레인 폭 |
| `SKYLENS_CORE_DETECT` | `1` | 탐지 잡 발행 여부 |
| `SKYLENS_CORE_ASSIGNED_HOLD_MS` | `2000` | "태스크 지정 완료"를 보여 주는 시간 |
| `SKYLENS_CORE_DRONE_ETA_SEC` | `10` | 드론 대기 카운트다운(데모 시나리오 4) |
| `SKYLENS_CORE_STATUS_MS` | `1000` | `ServerStatus` 푸시 주기 |
| `SKYLENS_CORE_TELEMETRY_HISTORY` | `600` | 드론별 보관 텔레메트리 개수 |
| `SKYLENS_CORE_WEB_MODE` | `dev` | `dev`(Vite 역프록시) · `prod`(정적) · `off` |
| `SKYLENS_CORE_WEB_TARGET` | `http://127.0.0.1:5173` | dev 모드 Vite 주소 |
| `SKYLENS_CORE_WEB_DIST` | `dist` | prod 모드 정적 루트 |
| `SKYLENS_DEMO` | `0` | 로그 표기용. **스케줄은 데모/실모드가 완전히 같다** |

> `SKYLENS_CORE_ENDPOINTS`는 코어가 아니라 **프록시**의 변수다(프록시가 다이얼할 코어 주소 목록).

---

## 4. 딜레이 패턴 — 구현된 규칙

> 딜레이 패턴의 스케줄 결정권은 **코어에만** 있다. 클라이언트는 도착한 것을 받을 뿐이며,
> 클라이언트가 타이머로 스스로 단계를 진행시키는 구조는 폐기한다. (COMPONENTS.md §3.4)

### 4.1 트리거는 시간이 아니라 이동량이다

`ingest.ts`와 `segmenter.ts`에는 **시계가 없다.**

1. 텔레메트리가 오면 드론 GPS를 지정 경로 폴리라인에 투영해 **출발점부터의 호 길이**를 얻는다
   (`RouteTracker.project`). 경로가 없으면 같은 계산을 단순 주행거리계로 한다.
2. `구간 인덱스 = floor(호 길이 / SEGMENT_METERS)`.
3. 인덱스가 바뀌면 **드론이 떠난 구간이 닫힌다.** 닫힘이 유일한 스케줄 트리거다.

구간은 **기간이 아니라 장소**다. 왕복 경로에서 되돌아오면 호 길이가 되감기므로 드론은
새 구간을 만드는 대신 **이미 지난 구간에 다시 들어간다**(`passes` 증가).

영상 슬라이스는 **찍힌 자리**에 귀속된다 — 슬라이스 중간 포즈의 호 길이로 구간을 정한다.
포즈가 없으면 그때 드론이 있는 구간으로 넣는다.

### 4.2 사다리와 겹침

`orchestrator.ts`의 네 규칙이 전부다.

- **R1 — 구간이 닫히면 그 구간의 다음 수준이 즉시 나간다.**
  요청하는 칸은 `배달된 수준 + 1`이다. 첫 닫힘이면 수준 1, 다시 지나가면 아직 못 받은 칸,
  이미 꼭대기면 아무것도 요청하지 않는다(왕복 반복이 그냥 맞아떨어진다).
  **수준 1 잡은 동시성 상한을 무시하고 나간다** — 방금 지나온 자리가 정제 잡 뒤에서 기다리는 일은 없다.
- **R2 — 정제는 뒤따르고, 다음 구간의 첫 수준과 겹친다.**
  수준 L이 배달되는 즉시 L+1을 큐에 넣는다. `RECON_CONCURRENCY ≥ 2`이므로 그 정제 잡이 아직
  날고 있는 동안 다음 구간이 닫혀 자기 수준 1을 쏜다. 이 겹침이 보고서가 말하는 지연 배치다.
- **R3 — 추월당한 수준은 실행하지 않고 버린다.**
  큐에서 기다리는 사이 그 구간에 더 높은 수준이 이미 배달됐다면 그 잡은 실행해 봐야 화면을
  퇴보시킬 뿐이다. 디스패치 직전에 걸러 내고 `reconJobsDropped`로 센다.
- **R4 — 첫 복원 잡이 좌표계를 정하고, 이후 모든 잡이 그것을 물려받는다.**
  `anchorFrame`이 아직 없으면 복원 레인은 **한 개만** 돈다. "첫 잡" 둘이 동시에 나가면 좌표계가
  둘 생기기 때문이다. 첫 결과의 `anchorFrame`을 저장하고 이후 모든 요청에 실어 보낸다
  (근거: [API.md §3](../../skylens_model/API.md), 75장↔150장 정규화가 87.8° 틀어진 실측).

**우선순위**(높은 것부터): ① 구간의 첫 수준 → ② 낮은 칸 → ③ 앞선 구간 → ④ 먼저 큐에 들어온 것.

탐지는 **별도 레인**이다. 느린 탐지 잡이 복원 사다리를 막지 못한다.

### 4.3 실제로 관측된 모습

`SEGMENT_METERS=40`, 사다리 `250,1000,3500`, 240 m 경로를 12 m/s로 비행하며 뽑은 코어 로그다
(슬라이스 수신 줄은 생략). 구간 2의 **정제(수준 2)가 날고 있는 동안** 구간 3의 **수준 1**이 나간다
— 이것이 겹침이다.

```
[core] segment 2 level 1/3 (250 steps, 125 KiB, 2568 ms) → viewers
[core] queue    recon-s2-l2-#11 steps=1000 "공간의 골격과 구조물 배치" (1 waiting)
[core] dispatch recon-s2-l2-#11 after 0 ms (recon 2/2 in flight, 0 waiting)   ← 구간 2 정제 시작
[core] segment 3 closed at 162.0 m (pass 1, gen 1, 3 slice(s), level 0)
[core] queue    recon-s3-l1-#13 steps=250 "형상 윤곽만 식별" (1 waiting)
[core] dispatch recon-s3-l1-#13 after 1 ms (recon 2/2 in flight, 0 waiting)   ← 겹침: 구간 3 수준 1
[core] segment 2 level 2/3 (1000 steps, 339 KiB, 1028 ms) → viewers
[core] queue    recon-s2-l3-#14 steps=3500 "표면 형성" (1 waiting)
[core] dispatch recon-s2-l3-#14 after 0 ms (recon 2/2 in flight, 0 waiting)
[core] segment 3 level 1/3 (250 steps, 170 KiB, 1028 ms) → viewers
[core] segment 2 level 3/3 (3500 steps, 3362 KiB, 3067 ms) → viewers [final]
```

뷰어가 받은 것도 같은 순서다 — 구간 2가 수준 2에 올라 있는 동안 구간 3의 수준 1이 도착한다.

```
[viewer] t+34.5s SPLAT segment 2 level 2 (1000 steps, 339 KiB, "공간의 골격과 구조물 배치")
[viewer] t+35.3s SPLAT segment 3 level 1 (250 steps, 170 KiB, "형상 윤곽만 식별")
[viewer] t+37.6s SPLAT segment 2 level 3 (3500 steps, 3362 KiB, "표면 형성") FINAL
```

**왕복 2회차**에서는 이미 꼭대기에 오른 구간이 다시 닫혀도 복원 잡이 생기지 않는다(R1).
탐지 잡만 나가고, 그마저 새 마커가 없으면 조용히 끝난다.

```
[core] segment 3 closed at 120.0 m (pass 2, gen 2, 6 slice(s), level 3)
[core] queue    detect-s3-l0-#26 (1 waiting)
[core] segment 3 detection: 0 new marker(s)
```

R1의 **상한 면제**는 큐가 밀렸을 때 드러난다. 모델 API가 죽어 12건이 대기하던 상황에서
구간 14가 닫히자 그 수준 1이 상한(2)을 넘겨 나갔다.

```
[core] queue    recon-s14-l1-#45 steps=250 "형상 윤곽만 식별" (13 waiting)
[core] dispatch recon-s14-l1-#45 after 0 ms (recon 3/2 in flight, 12 waiting)
```

`recon 3/2 in flight`는 오타가 아니라 R1이다 — 정제가 상한을 채우고 있어도 방금 지나온 자리는
기다리지 않는다. 밀린 구간 12개를 소화하는 동안 이 면제가 57회 발동했다.

> **R3(드롭)은 어느 실행에서도 발동하지 않았다.** 지금 정책에서는 한 구간에 복원 잡이 항상
> 하나뿐이기 때문이다 — 수준 L+1은 L이 배달된 뒤에야 큐에 들어가고, 재통과는 `busy()`에 막힌다.
> 그래서 "큐에서 기다리는 사이 추월당하는" 상황 자체가 생기지 않는다. R3은 동시성 정책이
> 바뀌었을 때를 위한 안전망으로 남아 있고, 현재로서는 **도달하지 않는 경로**다.

---

## 5. 모델 API가 없을 때

**정상 상태로 취급한다.** `modelClient.ts`는 두 가지 실패를 구분한다.

| 상황 | 예외 | 코어의 반응 |
|---|---|---|
| 접속 불가 · HTTP 오류 · 깨진 JSON | `ModelUnreachable` | 잡을 **큐에 되돌리고** `RETRY_MS` 뒤 재시도. `MAX_ATTEMPTS`를 넘으면 포기 |
| 폴링 중 `404` (모델이 재시작해 잡 id가 사라짐) | `ModelUnreachable` | 같다 — 잡을 되돌려 **새로 제출**한다 |
| 잡이 돌았는데 실패(예: 실모드 `PipelineUnavailable`) | `JobFailed` | 그 칸은 재시도하지 않는다. 나중에 그 자리를 다시 지나가면 다시 요청된다 |

어느 쪽이든 텔레메트리·미션·뷰어 팬아웃은 계속 돈다. 비행 도중 모델 API를 죽였을 때의 실측:

```
[core] model API unreachable at http://localhost:8100 (TypeError: fetch failed) —
       jobs stay queued, everything else keeps running
[core] detect-s8-l0-#32 deferred (attempt 1/20): ModelUnreachable: POST /detect/jobs: TypeError: fetch failed
[core] recon-s8-l1-#33  deferred (attempt 1/20): ModelUnreachable: POST /recon/jobs: TypeError: fetch failed
[core] segment 14 closed at 603.0 m (pass 1, gen 1, 4 slice(s), level 0)   ← 수집·구간 나누기는 계속된다
```

같은 동안 뷰어는 멀쩡히 돌아간다 — 구간이 `L0`으로 쌓이기만 할 뿐 화면이 얼지 않는다.

```
[viewer] t+26.5s STATUS receiving=true chunks=0 lastSeq=57 latency=1ms | s0:L0/3 s1:L0/3 s2:L0/3 s3:L0/3
[viewer] t+34.6s STATUS receiving=true chunks=0 lastSeq=96 latency=1ms | s0:L0/3 … s5:L0/3
```

모델 API를 다시 띄우면 밀려 있던 잡이 그대로 나간다(대기 시간이 `after …` 에 그대로 찍힌다).
재시작 전에 제출돼 있던 잡 두 건은 id가 사라져 `404`를 받았고, **폴링을 계속하는 대신 재제출**됐다.

```
[core] model API reachable at http://localhost:8100
[core] recon-s7-l2-#31 deferred (attempt 1/20): ModelUnreachable: GET /jobs/recon-becfb5633ff4:
       job is gone (HTTP 404) — the model API restarted
[core] dispatch recon-s9-l1-#35 after 18251 ms (recon 2/2 in flight, 13 waiting)
[core] segment 9 level 1/3 (250 steps, 105 KiB, 525 ms) → viewers
```

> 이 `404` 처리는 실측에서 발견해 고친 것이다. 그전에는 사라진 id를 잡 타임아웃(15분)까지
> 계속 폴링해서, 모델을 재시작하면 그 잡이 15분간 좀비로 남았다.

`/health`의 `model.reachable`과 `jobs.queued`를 보면 무엇이 밀려 있는지 보인다.
코어는 어느 경우에도 죽지 않는다 — 위 실행 내내 `GET /health`가 정상 응답했다.

---

## 6. WebRTC 이음매

지시된 형태는 코어 → 현황판 **WebRTC**다. Node에는 WebRTC 구현이 없고 네이티브 의존을 이번
단계에 끌어들이지 않기로 했으므로(COMPONENTS.md §8), 코어는 `Distributor` **인터페이스 뒤에서**
WebSocket으로 `skylens_client`에 밀고, 브라우저를 향한 WebRTC 중계와 시그널링은
`skylens_client`가 세운다.

```
코어 ──ws /viewer──→ skylens_client(8090) ──WebRTC──→ 브라우저 현황판
      (WsDistributor)        (중계 · 시그널링)
```

오케스트레이터는 자기가 무엇에 대고 밀고 있는지 **모른다**. `broadcast(msg)` 하나만 안다.
나중에 §2.3(클라이언트 간 P2P 재분배)이 붙으면 `WebRtcDistributor`를 끼우기만 하면 되고,
스케줄 코드는 한 줄도 바뀌지 않는다.

---

## 7. 검증 스크립트 (`src/test/core/`)

저장소의 모든 테스트·검증 하네스는 `src/test/` 아래로 모은다.

| 스크립트 | 하는 일 |
|---|---|
| `fakeUplink.ts` | 프록시+드론 대역. 경로를 **실제로 비행**하며 텔레메트리와 영상 슬라이스를 올린다. 코어가 내려보낸 `assign-route`를 받아 그 경로로 갈아탄다 |
| `fakeViewer.ts` | 뷰어 대역. 받은 `ViewerMessage`를 전부 찍는다. `--assign`을 주면 관제탑처럼 경로를 지정한다 |

```bash
# 1) 모델 API
SKYLENS_DEMO=1 uv run uvicorn skylens_model.app:app --port 8100
# 2) 코어
SKYLENS_CORE_WEB_MODE=off SKYLENS_CORE_LEVEL_STEPS=250,1000,3500 \
  npx tsx src/skylens_core/server/index.ts
# 3) 뷰어(관제탑 역할로 경로 지정)
npx tsx src/test/core/fakeViewer.ts --assign
# 4) 드론
npx tsx src/test/core/fakeUplink.ts
```

---

## 8. 이번 단계에서 하지 않는 것

- **DB를 붙이지 않는다.** 전부 인메모리이고 프로세스가 죽으면 데이터도 죽는다
  (COMPONENTS.md §3.4 · §9). `store.ts`는 디스크에 손대지 않는다.
- 클라이언트 간 P2P 스플랫 재분배(§2.3) — 코어가 모든 뷰어에 직접 배포한다.
- 인증·권한. 코렌 내부망 전제이고, 뷰어 소켓은 누구에게나 열려 있다.
- 경로를 **다시** 지정하면 호 길이의 0점이 새 경로로 바뀐다. 이미 배달된 구간을 지우지는
  않으므로, 비행 중 경로 교체는 구간 인덱스의 의미를 바꾼다. 지금은 그 상태로 둔다.
