# skylens_gateway — KOREN 외부망 진입점

드론이 KOREN 내부망으로 들어오는 **첫 홉**이다. 두 가지 모드로 동작하며
(`COMPONENTS.md` §3.2), 어느 모드에서도 **페이로드를 해석하거나 바꾸지 않는다**.
게이트웨이는 번역기가 아니라 전송로다.

```
[relay ]  드론 ──ws──> 게이트웨이 ──ws──> 프록시 ──> 코어
[webrtc]  드론 ──시그널링만── 게이트웨이
             └──────── 직결 ────────> 프록시 ──> 코어
```

---

## 1. 모드

| 모드 | `SKYLENS_GATEWAY_MODE` | 드론 접속 경로 | 하는 일 |
|---|---|---|---|
| 릴레이 (기본) | `relay` | `ws://<host>:8081/uplink` | `Envelope<UplinkMessage>`를 프록시 `/ingress`로 그대로 넘긴다 |
| 홀펀칭 | `webrtc` | `ws://<host>:8081/signal` | SDP/ICE만 중계한다. **미디어는 한 바이트도 지나가지 않는다** |

값이 둘 중 하나가 아니면 경고를 찍고 `relay`로 떨어진다.
한 프로세스는 한 모드만 연다 — 모드에 맞지 않는 경로로 오는 업그레이드 요청은 거절한다.

### relay 모드에서 프레임에 하는 유일한 일

`Envelope`에 **홉 스탬프**(`path[]`, `src/skylens_gateway/types.ts`)를 덧붙인다.
`seq` · `originTs` · `from` · `payload`는 건드리지 않으므로, 코어는 이걸 평범한
`Envelope`로 읽고 `path`는 홉별 지연을 계산할 때만 보면 된다.

`path`와 시그널링 프레임을 `src/shared/protocol.ts`에 넣지 않은 것은 의도다.
둘 다 **두 컴포넌트 사이의 전송 관심사**이지 컴포넌트 공통 계약이 아니다.

### webrtc 모드에서 미디어를 나르지 않는다는 것의 의미

게이트웨이는 드론에게 세션 id를 주고(`signal-hello`), 드론의 offer를 프록시로 넘기고,
프록시의 answer/ICE를 돌려주고, `signal-ready`가 오면 **빠진다**. 이후 드론은
프록시의 `direct` 주소(`ws://…:8082/direct`)로 직접 보낸다. 이때 코어에 도착하는
프레임의 `path`에는 `proxy` 홉만 있고 `gateway` 홉이 없다 — 이것이 미디어가
게이트웨이를 거치지 않았다는 **증거**이며 드릴이 실제로 확인하는 항목이다.

---

## 2. 백프레셔 — 드론은 절대 막지 않는다

프록시가 죽어 있어도 드론의 비행은 계속된다. 그래서 상류 링크(`upstream.ts`)는:

1. 잠깐 버퍼링한다 (`SKYLENS_GATEWAY_QUEUE`, 기본 256프레임)
2. 큐가 차면 **가장 오래된 것부터 버리고** 개수를 센다 (`droppedOverflow`)
3. 큐에 있어도 `SKYLENS_GATEWAY_QUEUE_AGE_MS`(기본 5초)보다 오래된 프레임은 버린다
   (`droppedStale`) — 낡은 텔레메트리는 없느니만 못하다
4. 버린 사실을 로그와 `/health` 양쪽에 보고한다
5. 상류가 돌아오면 **버퍼를 먼저 비운 뒤** 새 프레임을 보낸다 (순서 보존)

비디오는 `uri`로 참조만 하고 바이트를 싣지 않으므로 프레임 하나가 작게 유지되고,
이 드롭 정책이 의미 있는 밸브로 동작한다.

실제 관측(프록시를 내린 채 21프레임 전송, 큐 5로 축소):

```
[gateway] upstream backpressure: queue full (5), dropped oldest — 1 total
/health → "queued":5, "sent":0, "droppedOverflow":23, "lastError":"connect ECONNREFUSED …"
```

드론 쪽은 21프레임을 모두 보내고 정상 종료했다 — **한 번도 막히지 않았다**.

---

## 3. 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `SKYLENS_GATEWAY_MODE` | `relay` | `relay` \| `webrtc` |
| `SKYLENS_GATEWAY_PORT` | `8081` | HTTP + WebSocket 포트 |
| `SKYLENS_GATEWAY_HOST` | `0.0.0.0` | 바인드 주소 |
| `SKYLENS_PROXY_URL` | `ws://127.0.0.1:8082` | 프록시 기준 주소 (아래 둘의 기본값을 여기서 유도) |
| `SKYLENS_PROXY_INGRESS_URL` | `<PROXY_URL>/ingress` | relay 모드 상류 |
| `SKYLENS_PROXY_SIGNAL_URL` | `<PROXY_URL>/signal` | webrtc 모드 제어 채널 |
| `SKYLENS_GATEWAY_QUEUE` | `256` | 상류가 죽었을 때 버틸 프레임 수 |
| `SKYLENS_GATEWAY_QUEUE_AGE_MS` | `5000` | 이보다 오래된 대기 프레임은 폐기 |
| `SKYLENS_GATEWAY_RECONNECT_MS` | `1000` | 상류 재접속 간격 |
| `SKYLENS_GATEWAY_PING_MS` | `2000` | 상류 ping 주기(지연 측정) |
| `SKYLENS_GATEWAY_STATUS_MS` | `1000` | `LinkStatus` 발행 주기 |

---

## 4. 엔드포인트

| 경로 | 설명 |
|---|---|
| `ws /uplink` | 드론 접속 (relay 모드에서만 열린다) |
| `ws /signal` | 드론 시그널링 (webrtc 모드에서만 열린다) |
| `GET /health` | 모드 · 업타임 · 상류 상태 · 카운터 |

`/health`의 `counters.framesAccepted`는 **드롭되지 않고 상류 링크에 넘어간** 프레임 수다
(큐에 들어간 것도 포함). 실제로 소켓에 써진 수는 `upstream.sent`를 본다.

## 5. LinkStatus

게이트웨이는 자기가 소유한 홉만 보고한다.

- relay: `hop: "drone→gateway"`, `mode: "relay"`, `mbps` 측정값 포함
- webrtc: `hop: "drone→gateway (signalling only)"`, `mbps: null`
  — 미디어가 이 홉을 지나지 않으므로 보고할 비트레이트가 **없다**

---

## 6. 실행

```bash
# 릴레이 모드 (기본)
npx tsx src/skylens_gateway/index.ts

# 홀펀칭 모드
SKYLENS_GATEWAY_MODE=webrtc npx tsx src/skylens_gateway/index.ts
```

Windows PowerShell:

```powershell
$env:SKYLENS_GATEWAY_MODE='webrtc'; npx tsx src/skylens_gateway/index.ts
```

### 검증용 하네스

하네스는 `src/test/` 아래에 있다 (컴포넌트 디렉터리에는 런타임 코드만 둔다).

```bash
npx tsx src/test/gateway/fakeDrone.ts                        # relay 경유
npx tsx src/test/gateway/fakeDrone.ts ws://127.0.0.1:8082/direct   # 프록시 직결
npx tsx src/test/gateway/fakeDroneWebrtc.ts                  # 시그널링 후 직결

npx tsx src/test/proxy/pipelineDrill.ts    # 게이트웨이+프록시 5단계 드릴
npx tsx src/test/proxy/liveCoreCheck.ts    # 실제 코어 상대 통합 드릴
```

---

## 7. 한계 (이번 단계)

- 실제 WebRTC 피어 연결은 만들지 않는다. Node에 내장 구현이 없고 네이티브 의존
  (node-datachannel 등)을 끌어들이지 않기로 했다(`COMPONENTS.md` §8과 같은 판단).
  시그널링 계약은 진짜 형태 그대로 두고, `signal-ready`가 건네는 `direct` 주소만
  프록시의 WebSocket이 대신한다. 나중에 갈아끼울 때 바뀌는 곳은 프록시의
  응답 함수 하나뿐이다.
- 인증·TLS 없음. KOREN 등록 IP 안이라는 전제.
