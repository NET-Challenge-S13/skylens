# 브리지 서버 설치 (175.126.98.44)

이 폴더를 통째로 서버에 올리고 스크립트 하나 실행하면 끝난다. 기존 서비스(WireGuard 등)는 건드리지 않는다.

## 이 서버가 하는 일

KOREN은 등록된 IP만 들어갈 수 있는 폐쇄망이라 지휘관 브라우저도, 현장 드론 게이트웨이도 직접 못 들어간다. 이 서버가 **밖에서 대신 받아주는 창구**다.

| 역할 | 무엇을 | 양 |
|---|---|---|
| 시그널링 프록시 | 브라우저의 WebRTC 주소 교환을 대전으로 전달 (SSH 터널 경유) | 몇 KB |
| 현황판·업링크 중계 | `/viewer`(현황판 WebSocket)와 `/uplink`를 터널 20882로 대전 프록시에 넘긴다 | 실측 없음 |
| 드론 영상 수신 | 게이트웨이의 SRT 영상을 받아두면 대전이 당겨감 | 15~33 Mbps, 1부 |
| 현황판 호스팅 | 지휘관이 여는 현황판은 **정적 파일**이다. 빌드 산출물을 `WEB_DIR`에 두면 nginx가 그대로 서빙한다 (서버 프로세스 없음) | 작음 |

**영상 배포는 하지 않는다.** 지휘관 30명에게 나가는 영상은 KOREN 대전이 직접 쏜다. 그래서 이 서버 회선이 빨라야 할 필요가 없다.

## 설치 순서: 대전 키 등록이 먼저다

브리지 터널은 두 포워드(`20889→10889`, `20882→8082`)를 한 SSH 연결에 싣는다. 대전에 키가 아직 `permitopen 8082`를 허용하지 않은 채 브리지 터널이 올라가면, `ExitOnForwardFailure=yes` 때문에 **SSH 연결 자체가 끊기고 WHEP(20889)까지 같이 죽는다.** 그래서 순서는 이렇다.

1. 브리지에서 `sudo ./setup-bridge.sh` 를 한 번 돌려 공개키를 받는다 (이 단계에서 터널은 대전 키 등록 전이라 어차피 붙지 않는다).
2. 공개키를 정준모에게 보내고, 대전에서 `register-bridge-key.sh`로 `permitopen 10889·8082`가 등록됐다는 답을 받는다.
3. 브리지에서 `./check-bridge.sh` 로 `20889`·`20882` 두 포트가 열렸는지 본다.

이미 옛 권한(10889만)으로 등록된 키는 `register-bridge-key.sh`를 다시 돌리면 권한 줄이 갱신된다. 그 뒤에 브리지를 재설치한다.

## 실행 순서

```bash
# 1) 이 폴더를 서버로 복사
git clone -b develop https://github.com/NET-Challenge-S13/skylens.git
cd skylens/src/skylens_gateway

# 2) 설정: 80 이 이미 쓰이면 HTTP_PORT 를 고친다. 현황판 파일 위치는 WEB_DIR. 그 외는 그대로.
cp bridge.env.example bridge.env
nano bridge.env

# 3) 설치 (WEB_DIR 생성, nginx·터널·MediaMTX 설정)
chmod +x *.sh
sudo ./setup-bridge.sh

# 4) 현황판 파일 복사: 빌드 산출물 전체를 WEB_DIR에 둔다. 진입 파일은 status.html
sudo cp -r <현황판 빌드 산출물>/. /opt/skylens/web/

# 5) 점검
./check-bridge.sh
```

설치가 끝나면 화면에 **공개키 한 줄**과 **SRT publish 비밀번호**가 출력된다. **공개키를 정준모에게 보내면** 대전에 등록되고 터널이 자동으로 붙는다. 위 순서의 1·2단계가 이것이다.

## 경로와 포트

nginx(:80) 한 곳에서 네 갈래로 나뉜다.

| 경로 | 어디로 간다 | 비고 |
|---|---|---|
| `/` | `WEB_DIR`(기본 `/opt/skylens/web`)의 정적 파일 | 현황판. `status.html`이 진입. 프록시 없음 |
| `/viewer` | `127.0.0.1:20882` → 대전 프록시 `:8082` → 코어 `/viewer` | 현황판 WebSocket. 읽기 1시간 타임아웃 |
| `/uplink` | `127.0.0.1:20882` → 대전 프록시 `:8082` → 코어 `/uplink` | 업링크 WebSocket. 읽기 1시간 타임아웃 |
| `/live/` | `127.0.0.1:20889` → 대전 MediaMTX `:10889` | WHEP 시그널링 |
| `/healthz` | `127.0.0.1:20889` | 터널 생존 확인 |

터널 입구는 브리지 로컬에만 열린다(`127.0.0.1:20889`, `127.0.0.1:20882`). 대전 프록시 `:8082`는 브리지 밖에서 직접 닿지 않는다.

설치가 끝나면 열어야 하는 것은 아래 두 포트뿐이다. 나머지 `20889`·`20882`는 로컬 전용이다.

## 열어야 하는 포트 (두 개)

밖에서 이 서버로 들어오는 포트. 공유기 뒤면 둘 다 포트포워딩.

| 포트 | 프로토콜 | 누가 들어오나 |
|---|---|---|
| 80 (HTTP_PORT) | TCP | 지휘관 브라우저 |
| 10890 (SRT_PORT) | UDP | 드론 게이트웨이, 그리고 대전(영상을 당겨감) |

SSH 는 열 필요 없다 — 대전 쪽 터널은 **이 서버가 대전으로** 건다(175 는 KOREN 등록 IP 라 대전 SSH 에 들어갈 수 있다).

## 무엇이 설치되나

| 항목 | 위치 |
|---|---|
| 터널 키 | `/etc/skylens/tunnel_key(.pub)` — `skytunnel` 계정(셸 없음) 소유 |
| 터널 서비스 | systemd `daejeon-tunnel`: `autossh -L 127.0.0.1:20889:127.0.0.1:10889 -L 127.0.0.1:20882:127.0.0.1:8082 ubuntu@116.89.187.181 -p 26022`. 끊기면 자동 재연결 |
| nginx 사이트 | `/etc/nginx/sites-available/skylens`: `/` → `WEB_DIR` 정적 파일, `/viewer`·`/uplink` → `:20882`(터널), `/live/` → `:20889`(터널) |
| 현황판 정적 파일 | `WEB_DIR` (기본 `/opt/skylens/web`). 설치 시 디렉터리만 만들고, 파일은 직접 복사한다 |
| MediaMTX | `/opt/mediamtx/`, systemd `mediamtx`, `mediamtx` 계정 |
| ufw 규칙 | ufw가 **켜져 있을 때만** 80/tcp, 10890/udp 추가 |

대전 쪽에서는 이 키를 `restrict,permitopen="127.0.0.1:10889",permitopen="127.0.0.1:8082"` 로 등록하므로, 키가 유출돼도 대전의 10889(WHEP)와 8082(프록시) 외에는 아무것도 못 한다.

## 현황판 갱신

현황판은 정적 파일이라 서버를 재시작할 필요가 없다. 새 빌드 산출물을 `WEB_DIR`에 덮어쓰면 바로 반영된다.

```bash
sudo cp -r <새 빌드 산출물>/. /opt/skylens/web/
```

## 되돌리기

```bash
sudo systemctl disable --now mediamtx daejeon-tunnel
sudo rm -f /etc/systemd/system/mediamtx.service /etc/systemd/system/daejeon-tunnel.service \
           /etc/nginx/sites-enabled/skylens /etc/nginx/sites-available/skylens
sudo systemctl daemon-reload
sudo nginx -t && sudo systemctl reload nginx
sudo userdel skytunnel; sudo userdel mediamtx; sudo rm -rf /opt/mediamtx /etc/skylens /opt/skylens
```

## 막히면

- `check-bridge.sh`에서 `127.0.0.1:20889 없음` + `Permission denied` — 정상. 공개키가 아직 대전에 등록되지 않은 것. 보내고 기다리면 된다.
- `127.0.0.1:20882 없음` 인데 `20889`는 열림. 대전 키의 `permitopen`에 8082가 없다. 대전에서 `register-bridge-key.sh`를 다시 돌린다.
- `20889`·`20882` 둘 다 없음, 로그에 `administratively prohibited` 또는 `open failed`. 대전 키가 옛 권한이라 8082 포워드가 거부된 것. 위 "설치 순서"를 따른다.
- `대전 SSH 도달 불가` — 이 서버 IP 가 KOREN 에 등록돼 있는지 확인(`175.126.98.44` 여야 함).
- `nginx -t` 실패 — 포트 충돌. `bridge.env` 의 `HTTP_PORT` 를 바꾸고 재실행.
- `/` 가 404. `WEB_DIR`에 현황판 파일이 없거나 `status.html`이 빠진 것. 4단계 복사를 확인한다.
- `/viewer` 연결 실패. `check-bridge.sh`의 "대전 프록시 터널" 항목을 본다. 프록시(`:8082`)가 대전에서 떠 있는지도 확인한다.
