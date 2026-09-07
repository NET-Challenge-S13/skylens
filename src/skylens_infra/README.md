# skylens_infra — 실제 KOREN 회선 위 배치

`res/docs/COMPONENTS.md` §9가 "이번 단계에서 하지 않는 것"으로 남긴 **실제 KOREN 회선 위 배치**를 다룬다.
새 컴포넌트가 아니라 **배포 계층**이다 — 어느 컴포넌트가 어느 호스트에 올라가고, KOREN 방화벽을 어떻게 지나며, 라이브 영상은 어떤 경로로 나가는지. 코드보다 **설정과 그 설정을 고른 이유**가 본체다.

컴포넌트 경계·포트의 단일 출처는 `COMPONENTS.md`이고, 이 문서는 그 컴포넌트를 **호스트에 얹는 방법**만 말한다.

## 호스트 ↔ 컴포넌트

| 호스트 | 망 | COMPONENTS 컴포넌트 | 이 디렉터리가 추가로 올리는 것 |
|---|---|---|---|
| **브리지** `175.126.98.44` | KOREN 외부망 · 등록 IP | `skylens_gateway` :8081 · `skylens_client` :8090 | nginx :80 · MediaMTX SRT 수신 :10890 · 대전 터널 수신 |
| **대전** `116.89.187.181` | KOREN 내부망 | `skylens_proxy` :8082 · `skylens_core` :8080 | MediaMTX WHEP :10889 / ICE :10189 · 브리지로 가는 리버스 터널 |
| **판교** `10.246.246.9` | KOREN 내부망 · V100 | `skylens_model` :8100 | — (대전과 터널로만 통신) |

컴포넌트 포트(8080~8100)는 호스트 안 또는 KOREN 내부망 사이에서만 쓰이고, 밖으로 노출되는 건 브리지의 80·8081·10890 뿐이다. KOREN 안에서 인바운드로 열리는 포트는 정책(10000~15000)에 맞춘 MediaMTX 셋이다.

## 두 갈래 흐름

COMPONENTS §2의 데이터 흐름(드론 → 게이트웨이 → 프록시 → 코어 → 모델 → 코어 → 클라이언트)은 **스플랫·마커** 경로다. 이 디렉터리는 거기에 **라이브 영상** 경로 하나를 옆에 붙인다.

```
스플랫·마커  드론 ─▶ gateway(브리지) ─▶ proxy(대전) ─▶ core(대전) ─▶ model(판교) ─▶ core ─ws─▶ client(브리지) ─▶ 현황판
라이브 영상  게이트웨이 ─SRT─▶ [브리지 MediaMTX] ◀─당김─ 대전 MediaMTX ═══ WebRTC(WHEP) 직결 ═══▶ 현황판
                                                        └── 시그널링 HTTP: 현황판 → 브리지 nginx /live/ → 터널 → 대전
```

라이브 영상은 캐시가 안 되고 접속자 수만큼 대역이 필요해 브리지가 나르면 브리지 회선이 상한이 된다. 그래서 **대전이 직접 쏜다**(출구 2,618 Mbps, 4스트림 실측). 스플랫은 모두 같은 파일이라 캐시가 되므로 COMPONENTS 대로 client가 중계해도 된다.

## 왜 이 구성인가

**브리지가 필요한 이유는 WebRTC가 아니라 KOREN 정책이다.** WebRTC는 영상 중계를 없애지만 주소 교환(시그널링)은 원래 누군가 해야 하고, 보통은 영상 서버 자신이 HTTP로 받는다. KOREN은 등록 IP 외 인바운드를 막으므로 그 HTTP를 밖에서 받아줄 곳이 필요하다. 드론 게이트웨이(5G, 유동 IP)도 같은 이유로 못 들어가므로 SRT를 브리지가 받아두고 대전이 당겨간다. KOREN이 대전 10889/TCP·10890/UDP를 출발지 제한 없이 열어주면 이 우회는 걷어낼 수 있다 — 신청은 병행한다.

**대전이 브리지로 리버스 터널을 건다.** 브리지→대전 인바운드는 SSH 26022 외에 막혀 있어(11000 실측 차단) nginx가 대전 10889에 직접 못 붙는다. 대전→브리지 아웃바운드는 열려 있으므로 대전이 `-R 20889:127.0.0.1:10889`로 통로를 열고 nginx는 브리지 로컬 20889로 보낸다. 브리지의 `tunnel` 계정은 셸 없이 이 포트 하나만 열 수 있다. 같은 이유로 core(대전)→client(브리지)의 `ws /viewer` 업스트림도 대전이 나가는 방향이라 그대로 된다.

**MediaMTX를 쓴다.** SRT로 받아 WHEP로 내보내는 게 내장 기능이라 영상 서버 코드가 0줄이다. Node에는 WebRTC 미디어 구현이 없고(COMPONENTS §8), aiortc로 영상을 하면 재인코딩을 직접 해야 한다.

**진입은 SRT, 배포는 WebRTC.** 진입을 WHIP로 해도 시그널링 때문에 브리지가 그대로 필요하고 버는 건 0.5초 안팎이다. SRT는 불안정한 회선으로 서버에 밀어넣는 용도로 만들어졌고 ffmpeg 한 줄이다.

**서버 측 STUN·TURN을 두지 않는다.** 대전·판교는 공인 IP가 랜카드에 직결(NAT 없음)이라 자기 주소를 안다. 단말은 구글 공개 STUN. TURN은 판교↔브라우저 실측에서 불필요했고, UDP를 막는 망을 만나면 그때 coturn을 브리지에 올린다.

## 포트

| 서비스 | 호스트 | 포트 | 방향 |
|---|---|---:|---|
| 현황판 웹·시그널링 (nginx → client) | 브리지 | 80 | 브라우저 → 브리지 |
| 라이브 WHEP (nginx `/live/` → 터널) | 브리지 | 80 | 브라우저 → 브리지 → 대전 |
| 드론 접속 (gateway) | 브리지 | 8081 | 드론 → 브리지 |
| SRT 수신 (MediaMTX) | 브리지 | 10890/udp | 게이트웨이 → 브리지 |
| 터널 착지 | 브리지 | 20889 | 로컬만 |
| WHEP 시그널링 (MediaMTX) | 대전 | 10889 | 터널 경유만 |
| WebRTC 미디어 (MediaMTX) | 대전 | 10189/udp | 대전 → 브라우저, 대전이 개시 |
| SRT (대전, 로컬 테스트용) | 대전 | 10890/udp | 1단계만 |

`9092`(Kafka)·`9000~9001`(MinIO)·ICMP는 개방 불가 통보 → 큐·저장소는 `10092`·`10900/10901`. ICMP가 없어 상태 점검은 TCP로 한다.

## 구조

```
skylens_infra/
├─ bridge/      175 에 올릴 것 — 폴더째 복사해 setup-bridge.sh 하나로 끝 (bridge/README.md)
├─ daejeon/     대전 — MediaMTX 설정 2종(로컬 테스트 / 브리지 pull) + 터널 서비스 + setup
├─ scripts/     공용 — MediaMTX 설치, ffmpeg SRT 테스트 소스, aiortc WHEP 수신 프로브
└─ worklog/     로컬 작업 로그 (gitignore)
```

## 배포

**브리지** — `bridge/README.md`. 폴더 복사 → `bridge.env` → `sudo ./setup-bridge.sh` → `./check-bridge.sh`.

**대전**
```bash
sudo ./daejeon/setup-daejeon.sh local                 # 1단계: 로컬 ffmpeg 소스로 WHEP 검증
sudo ./daejeon/setup-daejeon.sh bridge <ssh_port>     # 2단계: 브리지에서 당겨오기 + 리버스 터널
```

**검증** — 브라우저 없이 WHEP 종단을 확인한다.
```bash
./scripts/test-source.sh 127.0.0.1 10890 drone &                          # 대전 로컬
~/rtc/bin/python scripts/whep_probe.py http://127.0.0.1:10889/drone 5     # 대전 안에서
~/rtc/bin/python scripts/whep_probe.py http://175.126.98.44/live/drone 5  # 밖에서, 브리지 경유
# [결과] video 110 frames … 첫 프레임까지 1427 ms   ← 2026-09-07 대전 로컬 실측
```

## 진행 상태

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 대전 MediaMTX, SRT→WHEP 종단 | ✅ 실측 검증 |
| 1' | 브리지 nginx·터널 → **브라우저 실재생** | 브리지 설치 대기 |
| 2 | 브리지 SRT 수신 + 대전 pull | 설정 준비됨 |
| 3 | `skylens_gateway`·`skylens_client`를 브리지에, `proxy`·`core`를 대전에, `model`을 판교에 실배치 | 미착수 — COMPONENTS §9 |
| — | TLS (도메인 + certbot) | 미착수. HTTP에서도 RTCPeerConnection은 동작 |

## COMPONENTS.md 에 제안하는 것

§2에 **라이브 영상 흐름**(SRT → 대전 MediaMTX → WHEP 직결)이 없다. 스플랫·마커와는 캐시 가능 여부가 달라 경로가 갈리므로 §2.4로 한 단락 추가를 제안한다. 이 PR에서는 단일 출처를 건드리지 않고 여기에만 적어 둔다.
