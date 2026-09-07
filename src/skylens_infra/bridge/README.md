# 브리지 서버 설치 (175.126.98.44 — 맥미니)

이 폴더를 통째로 서버에 올리고 스크립트 하나 실행하면 끝난다. 기존 서비스(WireGuard 등)는 건드리지 않는다.

**호스트가 macOS 이므로 `*-macos.sh` 를 쓴다.** Ubuntu 판(`setup-bridge.sh` / `check-bridge.sh`)은 리눅스 서버로 옮길 때를 위해 남겨둔 것이다.

## 이 서버가 하는 일

KOREN은 등록된 IP만 들어갈 수 있는 폐쇄망이라 지휘관 브라우저도, 현장 드론 게이트웨이도 직접 못 들어간다. 이 서버가 **밖에서 대신 받아주는 창구**다.

| 역할 | 무엇을 | 양 |
|---|---|---|
| 시그널링 프록시 | 브라우저의 WebRTC 주소 교환을 대전으로 전달 (SSH 터널 경유) | 몇 KB |
| 드론 영상 수신 | 게이트웨이의 SRT 영상을 받아두면 대전이 당겨감 | 15~33 Mbps, 1부 |
| 페이지 호스팅 | 지휘관이 여는 웹페이지 (`skylens_client`) | 작음 |

**영상 배포는 하지 않는다.** 지휘관 30명에게 나가는 영상은 KOREN 대전이 직접 쏜다. 그래서 이 서버 회선이 빨라야 할 필요가 없다.

## 준비물

- **Homebrew** — 없으면 https://brew.sh 안내대로 먼저 설치
- 관리자 계정 (sudo 가능)

## 실행 순서

```bash
# 1) 받기
git clone -b develop https://github.com/NET-Challenge-S13/skylens.git
cd skylens/src/skylens_infra/bridge

# 2) 설정 — 80 이 이미 쓰이면 HTTP_PORT 를 고친다 (8080 은 Homebrew nginx 기본과 겹치니 피할 것). 그 외는 그대로.
cp bridge.env.example bridge.env
nano bridge.env

# 3) 설치  (일반 계정에서 sudo 로)
chmod +x *.sh
sudo ./setup-bridge-macos.sh

# 4) 점검
./check-bridge-macos.sh
```

설치가 끝나면 화면에 **공개키 한 줄**과 **SRT publish 비밀번호**가 출력된다. **공개키를 정준모에게 보내면** 대전에 등록되고 터널이 자동으로 붙는다.

## 열어야 하는 포트 (두 개)

밖에서 이 맥으로 들어오는 포트. 공유기 뒤면 둘 다 포트포워딩.

| 포트 | 프로토콜 | 누가 들어오나 |
|---|---|---|
| 80 (HTTP_PORT) | TCP | 지휘관 브라우저 |
| 10890 (SRT_PORT) | UDP | 드론 게이트웨이, 그리고 대전(영상을 당겨감) |

SSH 는 열 필요 없다 — 대전 쪽 터널은 **이 맥이 대전으로** 건다(175 는 KOREN 등록 IP 라 대전 SSH 에 들어갈 수 있다).

## 무엇이 설치되나 (macOS)

| 항목 | 위치 |
|---|---|
| Homebrew 패키지 | `nginx`, `autossh` |
| 터널 키 | `/opt/skylens/tunnel/tunnel_key(.pub)` — 실행한 계정 소유, 0600 |
| 터널 서비스 | LaunchDaemon `com.skylens.daejeon-tunnel` — `autossh -L 127.0.0.1:20889:127.0.0.1:10889 ubuntu@116.89.187.181 -p 26022`. 끊기면 자동 재연결, 재부팅 후 자동 시작 |
| nginx | `$(brew --prefix)/etc/nginx/servers/skylens.conf` — `/` → `:8090`(client), `/live/` → `:20889`(터널). nginx 가 이미 떠 있으면 reload 만, 없으면 LaunchDaemon `com.skylens.nginx`(root, 80 포트용) |
| MediaMTX | `/opt/skylens/mediamtx/`, LaunchDaemon `com.skylens.mediamtx` |
| 로그 | `/opt/skylens/logs/*.log` |
| macOS 방화벽 | 켜져 있을 때만 mediamtx·nginx 허용 추가 |

대전 쪽에서는 이 키를 `command="/bin/false",restrict,permitopen="127.0.0.1:10889"` 로 등록하므로, 키가 유출돼도 대전 10889 포트 외에는 아무것도 못 한다.

## 되돌리기 (macOS)

```bash
for s in daejeon-tunnel mediamtx nginx; do sudo launchctl bootout system/com.skylens.$s 2>/dev/null; done
sudo rm -f /Library/LaunchDaemons/com.skylens.*.plist
sudo rm -f "$(brew --prefix)/etc/nginx/servers/skylens.conf"
sudo rm -rf /opt/skylens
```

## 막히면

- `check` 에서 `127.0.0.1:20889 없음` + `Permission denied` — 정상. 공개키가 아직 대전에 등록되지 않은 것. 보내고 기다리면 된다.
- `대전 SSH 도달 불가` — 이 맥의 공인 IP 가 `175.126.98.44` 인지 확인. 다른 회선이면 KOREN 이 거부한다.
- `nginx -t` 실패 — 포트 충돌. `bridge.env` 의 `HTTP_PORT` 를 바꾸고 재실행.
- `/` 가 502 — `skylens_client`(8090)가 아직 안 떠 있어서. `/live/` 만 쓰는 단계에서는 정상.
- 맥이 잠자기에 들어가면 서비스가 멈춘다 — 시스템 설정 → 에너지 절약에서 **잠자기 방지** (또는 `sudo pmset -a sleep 0`).
