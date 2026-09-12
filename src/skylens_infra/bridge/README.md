# 브리지 서버 설치 (175.126.98.44)

이 폴더를 통째로 서버에 올리고 스크립트 하나 실행하면 끝난다. 기존 서비스(WireGuard 등)는 건드리지 않는다.

## 이 서버가 하는 일

KOREN은 등록된 IP만 들어갈 수 있는 폐쇄망이라 지휘관 브라우저도, 현장 드론 게이트웨이도 직접 못 들어간다. 이 서버가 **밖에서 대신 받아주는 창구**다.

| 역할 | 무엇을 | 양 |
|---|---|---|
| 시그널링 프록시 | 브라우저의 WebRTC 주소 교환을 대전으로 전달 (SSH 터널 경유) | 몇 KB |
| 드론 영상 수신 | 게이트웨이의 SRT 영상을 받아두면 대전이 당겨감 | 15~33 Mbps, 1부 |
| 페이지 호스팅 | 지휘관이 여는 웹페이지 (`skylens_client`) | 작음 |

**영상 배포는 하지 않는다.** 지휘관 30명에게 나가는 영상은 KOREN 대전이 직접 쏜다. 그래서 이 서버 회선이 빨라야 할 필요가 없다.

## 실행 순서

```bash
# 1) 이 폴더를 서버로 복사
git clone -b develop https://github.com/NET-Challenge-S13/skylens.git
cd skylens/src/skylens_infra/bridge

# 2) 설정 — 80 이 이미 쓰이면 HTTP_PORT 를 고친다. 그 외는 그대로.
cp bridge.env.example bridge.env
nano bridge.env

# 3) 설치
chmod +x *.sh
sudo ./setup-bridge.sh

# 4) 점검
./check-bridge.sh
```

설치가 끝나면 화면에 **공개키 한 줄**과 **SRT publish 비밀번호**가 출력된다. **공개키를 정준모에게 보내면** 대전에 등록되고 터널이 자동으로 붙는다.

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
| 터널 서비스 | systemd `daejeon-tunnel` — `autossh -L 127.0.0.1:20889:127.0.0.1:10889 ubuntu@116.89.187.181 -p 26022`. 끊기면 자동 재연결 |
| nginx 사이트 | `/etc/nginx/sites-available/skylens` — `/` → `:8090`(client), `/live/` → `:20889`(터널) |
| MediaMTX | `/opt/mediamtx/`, systemd `mediamtx`, `mediamtx` 계정 |
| ufw 규칙 | ufw가 **켜져 있을 때만** 80/tcp, 10890/udp 추가 |

대전 쪽에서는 이 키를 `restrict,permitopen="127.0.0.1:10889"` 로 등록하므로, 키가 유출돼도 대전 10889 포트 외에는 아무것도 못 한다.

## 되돌리기

```bash
sudo systemctl disable --now mediamtx daejeon-tunnel
sudo rm -f /etc/systemd/system/mediamtx.service /etc/systemd/system/daejeon-tunnel.service \
           /etc/nginx/sites-enabled/skylens /etc/nginx/sites-available/skylens
sudo systemctl daemon-reload
sudo nginx -t && sudo systemctl reload nginx
sudo userdel skytunnel; sudo userdel mediamtx; sudo rm -rf /opt/mediamtx /etc/skylens
```

## 막히면

- `check-bridge.sh`에서 `127.0.0.1:20889 없음` + `Permission denied` — 정상. 공개키가 아직 대전에 등록되지 않은 것. 보내고 기다리면 된다.
- `대전 SSH 도달 불가` — 이 서버 IP 가 KOREN 에 등록돼 있는지 확인(`175.126.98.44` 여야 함).
- `nginx -t` 실패 — 포트 충돌. `bridge.env` 의 `HTTP_PORT` 를 바꾸고 재실행.
- `/` 가 502 — `skylens_client`(8090)가 아직 안 떠 있어서. `/live/` 만 쓰는 단계에서는 정상.
