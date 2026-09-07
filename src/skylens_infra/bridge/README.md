# 브리지 서버 설치 (175.126.98.44)

이 폴더를 통째로 서버에 올리고 스크립트 하나 실행하면 끝난다. 기존 서비스(WireGuard 등)는 건드리지 않는다.

## 이 서버가 하는 일

KOREN은 등록된 IP만 들어갈 수 있는 폐쇄망이라 지휘관 브라우저도, 현장 드론 게이트웨이도 직접 못 들어간다. 이 서버가 **밖에서 대신 받아주는 창구**다.

| 역할 | 무엇을 | 양 |
|---|---|---|
| 시그널링 프록시 | 브라우저의 WebRTC 주소 교환을 대전으로 전달 | 몇 KB |
| 드론 영상 수신 | 게이트웨이의 SRT 영상을 받아두면 대전이 당겨감 | 15~33 Mbps, 1부 |
| 페이지 호스팅 | 지휘관이 여는 웹페이지 | 작음 |

**영상 배포는 하지 않는다.** 지휘관 30명에게 나가는 영상은 KOREN 대전이 직접 쏜다. 그래서 이 서버 회선이 빨라야 할 필요가 없다.

## 실행 순서

```bash
# 1) 이 폴더를 서버로 복사 (예: scp -r bridge/ user@175.126.98.44:~/)
# 2) 설정 — SSH 포트가 22가 아니거나 80이 이미 쓰이면 고친다
cp bridge.env.example bridge.env
nano bridge.env

# 3) 설치
chmod +x *.sh
sudo ./setup-bridge.sh

# 4) 점검
./check-bridge.sh
```

설치가 끝나면 화면에 **SSH 포트 · SRT publish 비밀번호 · 브라우저 주소**가 출력된다. 이 값을 정준모에게 전달하면 된다.

## 공유기 뒤에 있다면

아래 세 포트를 이 서버로 포트포워딩해야 한다. SSH가 이미 밖에서 되고 있으면 SSH 포트는 되어 있는 것이다.

| 포트 | 프로토콜 | 용도 |
|---|---|---|
| 80 (HTTP_PORT) | TCP | 브라우저 |
| 10890 (SRT_PORT) | UDP | 드론 게이트웨이 |
| 22 (SSH_PORT) | TCP | 대전 터널 |

## 무엇이 설치되나

| 항목 | 위치 |
|---|---|
| `tunnel` 계정 | 셸 없음. 대전 공개키(`tunnel.pub`)만 등록, 포트포워딩 외 전부 차단(`restrict`) |
| sshd 드롭인 | `/etc/ssh/sshd_config.d/skylens-tunnel.conf` — tunnel 계정은 `127.0.0.1:20889`만 열 수 있음 |
| nginx 사이트 | `/etc/nginx/sites-available/skylens` — `:80 → 127.0.0.1:20889` |
| MediaMTX | `/opt/mediamtx/`, systemd `mediamtx.service`, `mediamtx` 계정으로 실행 |
| ufw 규칙 | ufw가 **켜져 있을 때만** 80/tcp, 10890/udp 추가 |

## 되돌리기

```bash
sudo systemctl disable --now mediamtx
sudo rm -f /etc/systemd/system/mediamtx.service /etc/nginx/sites-enabled/skylens \
           /etc/nginx/sites-available/skylens /etc/ssh/sshd_config.d/skylens-tunnel.conf
sudo nginx -t && sudo systemctl reload nginx
sudo sshd -t && sudo systemctl reload ssh
sudo userdel -r tunnel; sudo userdel mediamtx; sudo rm -rf /opt/mediamtx
```

## 막히면

- `check-bridge.sh`에서 `127.0.0.1:20889 없음` — 정상. 대전이 터널을 걸어야 생긴다. 대전 쪽 작업.
- `nginx -t` 실패 — 포트 충돌. `bridge.env`의 `HTTP_PORT`를 바꾸고 재실행.
- `sshd -t` 실패 — 스크립트가 reload를 **하지 않는다**(잠기지 않음). 출력된 오류를 전달.
