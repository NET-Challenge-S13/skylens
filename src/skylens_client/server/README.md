---
tags: [skylens, component, client]
links: "[[COMPONENTS]] · [[ARCHITECTURE]]"
---

# `skylens_client` 서버 — 현황판 웹서버 + WebRTC 연결 중계

KOREN **외부망**에 놓이는 컴포넌트다(`res/docs/COMPONENTS.md` §3.6). 포트 **8090** 하나에
현황판이 필요로 하는 모든 것이 올라간다. 코어(내부망)로 들어가는 소켓은 **정확히 하나**이고,
브라우저는 그 하나를 N개의 현황판이 나눠 쓴다.

```
                       KOREN 내부망            KOREN 외부망
skylens_core :8080  ──ws /viewer──▶  skylens_client :8090  ──ws /stream──▶  현황판(브라우저) × N
   (배포자)              업스트림 1개              (중계)                        다운스트림 N개
```

---

## 1. 엔드포인트

| 경로 | 종류 | 설명 |
|---|---|---|
| `/health` | GET | 업스트림 상태 · 접속 현황판 수 · 중계 카운터 (§5) |
| `/stream` | WebSocket | 현황판 피드. 코어의 `ViewerMessage`를 **그대로** 흘려보낸다 |
| `/peerjs/**` | HTTP + WebSocket | PeerJS 시그널링. 브라우저를 향한 WebRTC 연결 중계 (§4) |
| 그 외 전부 | HTTP | 현황판 웹앱. dev는 Vite(5173) 리버스 프록시, prod는 `dist/` 정적 서빙 |

현황판 접속 주소는 **`http://<호스트>:8090/res/static/status.html`** 하나다.
같은 오리진에서 웹앱과 스트림이 함께 나오므로 브라우저 쪽에 설정할 것이 없다.
Vite(5173)에서 직접 열었다면 같은 호스트의 8090으로 붙고, `?relay=<포트|ws URL>`로 덮어쓸 수 있다.

## 2. 실행

```bash
npx tsx src/skylens_client/server/index.ts          # dev (Vite 5173 프록시)
SKYLENS_CLIENT_MODE=prod npx tsx src/skylens_client/server/index.ts   # prod (dist/ 서빙)
```

환경변수(`config.ts`): `SKYLENS_CLIENT_PORT`(8090) · `SKYLENS_CLIENT_HOST` ·
`SKYLENS_CORE_WS`(`ws://localhost:8080/viewer`) · `SKYLENS_CLIENT_MODE`(dev|prod) ·
`SKYLENS_VITE_URL` · `SKYLENS_CLIENT_DIST` · `SKYLENS_CLIENT_BACKOFF_MIN/MAX`.

> `package.json`에 스크립트는 아직 없다. `"client": "tsx src/skylens_client/server/index.ts"`
> 항목이 있으면 편하다 — 이 컴포넌트는 `package.json`을 건드리지 않는다.

## 3. 중계 규칙

**번역하지 않는다.** `/stream`으로 나가는 프레임은 코어가 보낸 `ViewerMessage`
그 자체다(`src/shared/protocol.ts`). 코어가 `Envelope`로 감싸 보내면 벗겨서 내보낸다.
현황판이 보내는 것은 무시한다 — 제어는 관제탑→코어 경로에만 있다.

**중계만 하는 프레임 2종**은 코어가 모르는, 중계기 자신에 대한 정보다
(`src/skylens_client/relayProtocol.ts`).

| 프레임 | 시점 | 용도 |
|---|---|---|
| `relay-hello` | 접속 직후 1회 | 현황판 id, 서버 시각, 방 이름, 시그널링 경로, 재생한 프레임 수 |
| `relay-status` | 업스트림 상태가 바뀔 때마다 | `connecting`/`online`/`offline` + 한국어 사유 |

이 둘이 있어야 현황판이 **"코어가 조용한 것"** 과 **"코어에 닿지 못하는 것"** 을 구분해
표시할 수 있다. `kind` 값이 `ViewerMessage`와 겹치지 않으므로 수신 측은 하나의 union을 스위치한다.

**재생 캐시.** 코어는 상태를 한 번만 흘려보낸다. 살아있는 프레임만 중계하면 새로고침한 현황판이나
늦게 붙은 두 번째 현황판은 다음 청크가 올 때까지 빈 화면을 본다. 그래서 중계기는
**키별 최신 프레임 하나씩**을 들고 있다가 접속 직후 재생한다.

```
mission-status(1) → link-status(hop별) → telemetry(드론별)
                 → splat-chunk(구간별 최고 수준) → detection(id별) → server-status(1)
```

구간별로 **최고 수준만** 남긴다 — 낮은 수준이 캐시에 살아남으면 새로고침한 현황판이
정제된 결과 뒤에 거친 결과를 받게 된다. 이 규칙은 코어의 딜레이 패턴 규칙과 같은 것이며,
중계기가 스케줄을 흉내내는 것이 아니라 **같은 대체 규칙을 캐시에 적용**할 뿐이다.

**업스트림 재연결**은 지터를 넣은 지수 백오프(기본 0.5s→8s)다. 연결 실패는 `error`와 `close`를
모두 발생시키므로 소켓당 한 번만 재시도로 계산한다. 상태가 바뀔 때마다 모든 현황판에
`relay-status`가 즉시 나가므로, 코어가 죽어도 화면은 멈추지 않고 **대기 상태**를 말한다.
이미 도착한 기하는 지운다 — 가 아니라 **유지한다**. 복원되어 도착한 것은 시뮬레이션이 아니다.

## 4. WebRTC 연결 중계 — 지금 무엇이 살아있고, 무엇을 위한 준비인가

이 절이 이 컴포넌트에서 가장 오해하기 쉬운 부분이라 명시한다.

**지금 살아있는 것 (LIVE).** 현황판은 데이터를 **`/stream` WebSocket**으로 받는다.
`res/docs/COMPONENTS.md` §8이 적어 둔 대로 Node에는 WebRTC 구현이 없고 이번 단계에
네이티브 의존을 끌어들이지 않기로 했으므로, 코어→클라이언트도 클라이언트→브라우저도
WebSocket이다. 현황판에 보이는 모든 것은 이 경로로 온다.

**준비해 둔 것 (PREPARED).** `/peerjs`에 **PeerJS 시그널링 서버가 실제로 떠 있다.**
등록된 피어와 방을 집계해 `/health`의 `peer` 항목으로 노출한다. 다만 **지금은 아무도 등록하지 않는다.**
이것이 겨냥하는 것은 COMPONENTS.md **§2.3의 클라이언트 간 P2P 재분배**다 — 현황판이 여럿일 때
코어가 모든 현황판에 스플랫을 직접 올리면 업링크가 병목이므로, 이미 구간을 가진 현황판이
아직 없는 현황판에 전달하게 만들어 코어의 전송 부담을 줄인다. 그 핸드셰이크에는 브로커가 필요하고,
**그 브로커가 여기다.**

id는 한 곳에서만 발급한다: 중계기가 `relay-hello`로 `skylens-<room>-board-<n>` 형태의
`peerId`를 건네고, 등록되는 id에서 방 이름을 역산해 `/health`에 방별 메시가 보이게 해 둔다.
P2P를 켜는 시점에 브라우저 쪽에서 그 id로 `Peer`를 만들기만 하면 되고,
코어 쪽에서는 `Distributor` 구현만 바꾸면 된다.

## 5. `/health`

```jsonc
{
  "component": "skylens_client", "port": 8090,
  "web":      { "mode": "dev (vite proxy)", "target": "http://localhost:5173", "ok": true },
  "upstream": { "url": "...", "state": "online|connecting|offline",
                "detail": "코어 응답 없음 · ECONNREFUSED",
                "since": 0, "retries": 0, "received": 0, "malformed": 0 },
  "boards":   { "connected": 1, "seen": 3, "rooms": { "default": 1 } },
  "relayed":  { "total": 628, "writes": 644, "replayed": 26, "dropped": 0,
                "byKind": { "telemetry": 593, "splat-chunk": 12, "detection": 3, ... },
                "cached": { "segments": 4, "detections": 3, "drones": 1, "mission": true } },
  "peer":     { "path": "/peerjs", "wsPath": "/peerjs/peerjs", "live": true,
                "peers": 0, "rooms": {}, "purpose": "..." }
}
```

`relayed.total`은 코어에서 받아 내보낸 `ViewerMessage` 수, `writes`는 실제 소켓 쓰기 수
(= 총합 × 현황판 수), `replayed`는 접속하는 현황판에 캐시에서 재생한 프레임 수다.

## 6. 검증

`src/test/client/`에 두 개가 있다.

| 파일 | 역할 |
|---|---|
| `fakeCore.ts` | 코어 대역. 실제 계약(`ws://localhost:8080/viewer`)으로 `res/static/demo`의 구간×수준 자산을 **엇갈리게** 흘려보낸다. 사다리 높이·라벨·스텝수는 `segments.json`에서 읽는다 |
| `boardCheck.ts` | 브라우저(Playwright)로 현황판을 열어 타임라인을 **출력**한다. 코어 정지 → 기동 → 정지 → 새로고침 4단계 |

```bash
npm run dev                                     # Vite 5173
npx tsx src/skylens_client/server/index.ts       # 중계 8090
npx tsx src/test/client/boardCheck.ts            # fakeCore를 직접 띄우고 끈다
```
