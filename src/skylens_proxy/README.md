# skylens_proxy — KOREN 내부망 다중 경로

프록시가 존재하는 이유는 하나다. **재난 상황에서 코어로 가는 한 경로가 죽어도
서비스가 유지되어야 한다** (`COMPONENTS.md` §3.3). 그래서 이 컴포넌트의 본체는
전달 코드가 아니라 `corePaths.ts`의 **헬스체크 · 페일오버 · 복귀** 로직이다.

```
게이트웨이 [relay] ──> /ingress ┐
                                ├─> 프록시 ─┬─> 코어 경로 #0 (활성)
드론 [webrtc 직결]  ──> /direct  ┘          └─> 코어 경로 #1 (대기, 연결 유지)
```

---

## 1. 두 종류의 입력을 모두 받는다

| 경로 | 누가 붙나 |
|---|---|
| `ws /ingress` | 게이트웨이가 릴레이한 연결 |
| `ws /direct` | 홀펀칭을 마친 드론의 **직결** 연결 |
| `ws /signal` | 게이트웨이와의 홀펀칭 시그널링 제어 채널 |
| `GET /health` | 경로별 상태 · 페일오버 이력 · 카운터 |

`/signal`에서 프록시는 **응답하는 쪽(answering peer)**이다. offer가 오면
answer·ICE·`signal-ready`를 돌려주고, `signal-ready`에 드론이 직접 붙을
주소(`SKYLENS_PROXY_PUBLIC_URL`)를 실어 보낸다.

---

## 2. 다중 경로 · 헬스체크 · 페일오버

`SKYLENS_CORE_ENDPOINTS`는 **우선순위 순서**의 목록이다. 0번이 선호 경로,
나머지는 대기 경로다.

- 대기 경로도 **연결을 유지하고 계속 프로브**한다. 그래서 페일오버가 재접속이
  아니라 **포인터 이동**이고, 전환에 걸리는 시간이 재접속 시간에 묶이지 않는다.
- 건강 판정은 **두 조건 모두**를 본다.
  1. 소켓이 열려 있다
  2. `healthTimeout` 안에 ping에 대한 pong이 왔다
- (2)가 있는 이유: **회선은 살아 있는데 반대편이 죽은** 경우는 close 이벤트로는
  잡히지 않는다. 재난 상황에서 더 흔한 쪽이 이 경우다.
- `failback=true`(기본)이면 더 나은 우선순위 경로가 회복되는 즉시 되돌아간다.
  `false`면 현재 경로가 건강한 동안 그대로 머문다(sticky).
- 건강한 경로가 하나도 없으면 프레임을 버퍼링하고, 큐가 차면 오래된 것부터 버리며,
  경로가 살아나면 **버퍼를 먼저 비운 뒤** 새 프레임을 보낸다.

### 부팅 순서에 대한 주의

기동 유예(`SKYLENS_PROXY_STARTUP_GRACE_MS`, 기본 500ms) 동안에는 더 나은 경로가
아직 접속 중일 수 있으므로 대기 경로를 성급히 승격하지 않는다. 다만 **실제 코어는
기동에 수 초가 걸리므로** 유예가 지나면 대기 경로가 먼저 활성이 되고, 코어가 뜨는
순간 `FAILBACK`으로 되돌아온다. 이건 정상 동작이다 — 그 몇 초 동안에도 드론의
프레임은 갈 곳이 있었다는 뜻이다.

### 코어가 없을 때

코어가 재배포 중이라 아예 없어도 프록시는 죽지 않는다. 경로를 계속 재시도하면서
대기 경로로 넘기고, 그것마저 없으면 버퍼링한다. 드론과 게이트웨이는 이 사실을
`LinkStatus`로만 통보받고 계속 동작한다.

---

## 3. 전달할 때 더하는 것

페이로드·`seq`·`originTs`·`from`은 **그대로** 둔다. 더하는 것은 자기 홉 스탬프
하나뿐이며(`types.ts`), 거기에 `via`(그 프레임이 실제로 나간 KOREN 경로)를 적는다.
코어는 이걸 보고 지연을 **경로별로** 귀속시킬 수 있다.

```
path=gateway->proxy(ws://127.0.0.1:8080/uplink)   # 릴레이 경유
path=proxy(ws://127.0.0.1:8080/uplink)            # webrtc 직결 (게이트웨이 홉 없음)
```

`LinkStatus`의 `hop`에는 활성 경로 주소를 그대로 적는다
(`proxy→core (ws://…/uplink)`). 배지가 "연결됨"이 아니라 **어느 회선이
나르고 있는지**를 보여줄 수 있게 하기 위해서다.

---

## 4. 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `SKYLENS_PROXY_PORT` | `8082` | HTTP + WebSocket 포트 |
| `SKYLENS_PROXY_HOST` | `0.0.0.0` | 바인드 주소 |
| `SKYLENS_CORE_ENDPOINTS` | `ws://127.0.0.1:8080/uplink,ws://127.0.0.1:8180/uplink` | **우선순위 순** 코어 경로 목록 (쉼표 구분) |
| `SKYLENS_PROXY_PUBLIC_URL` | `ws://127.0.0.1:8082/direct` | 홀펀칭 후 드론에게 알려 줄 직결 주소 |
| `SKYLENS_PROXY_FAILBACK` | `true` | 더 나은 경로가 살아나면 되돌아갈지 |
| `SKYLENS_PROXY_HEALTH_INTERVAL_MS` | `1000` | 경로별 프로브 주기 |
| `SKYLENS_PROXY_HEALTH_TIMEOUT_MS` | `3000` | 이 시간 안에 pong이 없으면 불건강 |
| `SKYLENS_PROXY_RECONNECT_MS` | `1000` | 죽은 경로 재접속 간격 |
| `SKYLENS_PROXY_STARTUP_GRACE_MS` | `500` | 기동 시 우선순위 역전 방지 창 |
| `SKYLENS_PROXY_QUEUE` | `512` | 모든 경로가 죽었을 때 버틸 프레임 수 |
| `SKYLENS_PROXY_QUEUE_AGE_MS` | `5000` | 이보다 오래된 대기 프레임은 폐기 |
| `SKYLENS_PROXY_STATUS_MS` | `1000` | `LinkStatus` 발행 주기 |

`/health`의 `ingress`에는 현재 열린 수(`open`·`gateway`·`droneDirect`)와 함께
누적치(`seenDroneDirect`·`framesFromDroneDirect`)가 있다. 이미 떠난 직결 드론은
`open`만 봐서는 보이지 않는데, 홀펀칭을 디버깅할 때 정작 보고 싶은 게 그 드론이다.

---

## 5. 실행

```bash
npx tsx src/skylens_proxy/index.ts

# 경로를 3중화하고 sticky 로 두기
SKYLENS_CORE_ENDPOINTS=ws://a/uplink,ws://b/uplink,ws://c/uplink \
SKYLENS_PROXY_FAILBACK=false npx tsx src/skylens_proxy/index.ts
```

### 검증

```bash
npx tsx src/test/proxy/pipelineDrill.ts    # 5단계: 릴레이 → 페일오버 → 복귀 → 무응답 → webrtc
npx tsx src/test/proxy/liveCoreCheck.ts    # 실제 skylens_core 를 띄워 놓고 하는 통합 드릴
npx tsx src/test/proxy/fakeCore.ts 8180 standby   # 대기 경로 대역
```

`fakeCore.ts`는 `FAKE_CORE_HANG_AFTER_MS` / `FAKE_CORE_HANG_FOR_MS`로 **이벤트 루프를
막을** 수 있다. 소켓은 열린 채 pong만 끊기므로, 프로세스를 죽여서는 재현할 수 없는
"회선은 살아 있고 반대편만 죽은" 상황을 그대로 만든다.

관측된 전환(드릴 phase 4):

```
[proxy] core path #0 UNHEALTHY: ws://127.0.0.1:8080/uplink (no response)
[proxy] FAILOVER ws://127.0.0.1:8080/uplink -> ws://127.0.0.1:8180/uplink (path #0 went unhealthy)
[proxy] FAILBACK ws://127.0.0.1:8180/uplink -> ws://127.0.0.1:8080/uplink (path #0 became healthy)
```

---

## 6. 알려진 사항

- **하네스는 타입체크에서 빠져 있다.** `tsconfig.json`이 `src/test`를 `exclude`
  하므로 `src/test/gateway/**`·`src/test/proxy/**`의 하네스는 `npx tsc --noEmit`
  대상이 아니다. 컴포넌트 코드(`src/skylens_proxy/**`, `src/skylens_gateway/**`)는
  그대로 타입체크된다. 하네스까지 검사하려면 `tsconfig.json`에서 `src/test` 제외를
  풀어야 하는데, 그 파일은 이 컴포넌트의 소유가 아니라 그대로 두었다.
- 실제 WebRTC DataChannel은 종단하지 않는다. `signal-ready`가 주는 직결 주소는
  협상된 채널을 대신하는 WebSocket이다. 교체 지점은 `index.ts`의 응답 함수 하나다.
- DB 없음. 프록시는 상태를 보관하지 않는다 — 전송 중인 프레임만 잠깐 들고 있는다.
