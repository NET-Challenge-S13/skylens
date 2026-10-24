# 브리지 서버 설치 (175.126.98.44)

이 폴더를 통째로 서버에 올리고 스크립트 하나 실행하면 끝난다. 기존 서비스(WireGuard 등)는 건드리지 않는다.

**지금 이 호스트는 라즈베리파이 5(Debian 13 trixie · arm64)다. 그래서 `setup-bridge.sh` · `check-bridge.sh`(리눅스 판)를 쓴다.** `*-macos.sh` 는 맥으로 옮길 때를 위해 남겨둔 것이다. 두 판은 같은 일을 하고 같은 `bridge.env` 와 같은 `nginx-skylens.conf` 를 쓴다. 다른 것은 셋뿐이다 — 패키지(apt / Homebrew), 서비스(systemd / launchd), 터널 키 위치(`/etc/skylens` / `/opt/skylens/tunnel`). MediaMTX 바이너리는 설치 스크립트가 `uname -m` 으로 골라 받는다(파이는 `linux_arm64`).

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
bash src/skylens_client/build-web.sh                       # 빌드하는 쪽에서 한 번
sudo cp -R src/skylens_client/dist/. /opt/skylens/web/      # 브리지에서

# 5) 점검
./check-bridge.sh
```

맥에 올릴 때는 3·5단계만 다르다. 준비물은 **Homebrew**(없으면 https://brew.sh )와 sudo 가능한 일반 계정이다.

```bash
cd skylens/src/skylens_gateway
cp bridge.env.example bridge.env && nano bridge.env   # HTTP_PORT=8080 은 피한다 (Homebrew nginx 기본과 겹침)
chmod +x *.sh
sudo ./setup-bridge-macos.sh                          # root 로그인 상태가 아니라 일반 계정에서 sudo
sudo cp -R <현황판 빌드 산출물>/. /opt/skylens/web/
./check-bridge-macos.sh
```

설치가 끝나면 화면에 **공개키 한 줄**과 **SRT publish 비밀번호**가 출력된다. **공개키를 정준모에게 보내면** 대전에 등록되고 터널이 자동으로 붙는다. 위 순서의 1·2단계가 이것이다.

## 경로와 포트

nginx(:80) 한 곳에서 갈라진다.

| 경로 | 어디로 간다 | 비고 |
|---|---|---|
| `/` | `WEB_DIR`(기본 `/opt/skylens/web`)의 정적 파일 | 현황판. `status.html`이 진입. 프록시 없음 |
| `/viewer` | `127.0.0.1:20882` → 대전 프록시 `:8082` → 코어 `/viewer` | 현황판 WebSocket. 읽기 1시간 타임아웃 |
| `/uplink` | `127.0.0.1:20882` → 대전 프록시 `:8082` → 코어 `/uplink` | 업링크 WebSocket. 읽기 1시간 타임아웃 |
| `/live/` | `127.0.0.1:20889` → 대전 MediaMTX `:10889` | WHEP 시그널링. 세션 `Location`도 `/live/` 아래로 되돌린다 |
| `/healthz` | `127.0.0.1:20889` | 터널 생존 확인 (대전이 답해야 200) |
| `/health` | nginx 자신 | 이 서버 생존 확인. 터널이 죽어도 답한다 |
| `:8091/` (TLS) · `:8090/` | `127.0.0.1:20882` → 대전 프록시 → **코어 전체** | **관제탑**. 랜·VPN 전용 server 블록이고 공유기 포워딩이 없다 |

경로 이름은 코어가 정한다. `/live/{path}/whep` 의 `{path}` 는 `drone1`·`drone2`·`drone3`(`SKYLENS_CORE_STREAM_PATH` 기본값 `drone{id}`)이고, 대전·브리지 MediaMTX 설정의 `paths:` 가 그 세 이름이어야 한다. 코어의 `/control`(관제탑)·`/site`·`/vworld`는 KOREN 안에서만 쓰므로 밖으로 내보내지 않는다.

**중개되는 것은 시그널링과 제어뿐이다.** 영상 미디어는 대전 `10189/udp` 에서 브라우저로 직결되고(WebRTC), 드론 SRT 는 이 호스트의 MediaMTX `:10890/udp` 가 받는다. nginx 를 지나는 양은 SDP 몇 KB 와 WebSocket 뿐이다.

### TLS — 선택이 아니라 필수다

**현황판과 관제탑은 HTTPS 로 띄운다.** 브라우저가 WebGPU 를 **보안 컨텍스트**에서만 열어 주고, 화면 렌더러가 그게 없으면 `Error creating DeviceHandle: RequestAdapter` 로 패닉하면서 빈 화면이 된다. 그래서 같은 server 블록이 `HTTPS_PORT`(기본 443)도 듣고, 평문으로 들어온 **페이지 요청**은 거기로 넘긴다. `/health`·`/healthz`·`/viewer`·`/uplink`·`/live/` 는 두 스킴 모두에서 그대로 답하므로 점검 스크립트와 드론 쪽은 영향이 없다. **WebSocket 핸드셰이크는 넘기지 않는다** — 301 을 받으면 소켓이 깨지므로, 리다이렉트는 스킴과 `Upgrade` 헤더를 함께 보는 map 으로 건다.

인증서는 설치 스크립트가 자체 서명으로 만든다(`/etc/skylens/tls/board.crt`·`board.key`, SAN 에 호스트 이름과 이 호스트의 주소들). **자체 서명이어도 경고를 한 번 넘기면 보안 컨텍스트는 성립한다.** 경고 자체를 없애려면 셋 중 하나다.

1. `mkcert` 등으로 만든 인증서를 **같은 두 경로에 덮어쓰고** 설치 스크립트를 다시 돌린다(기존 인증서는 절대 덮어쓰지 않는다). 루트 CA 를 보는 쪽 PC 에 설치해야 한다.
2. DNS-01 로 받은 진짜 인증서를 같은 자리에 둔다. 팀이 이미 duckdns 를 쓰고 있어 인바운드 없이도 발급된다.
3. `REDIRECT_TO_HTTPS=no` 로 두고 평문으로 쓰되, 보는 쪽 Chrome 을 `--unsafely-treat-insecure-origin-as-secure=http://<주소>` 로 띄운다. 시연 당일 임시방편이다.

`443`·`8090`·`8091` 은 **공유기에 포워딩하지 않는다.** 랜과 VPN 에서만 닿는다.

### 관제탑

관제탑은 본래 KOREN 안 화면이다. 코어가 `:8080` 에서 서빙하고 대전 프록시가 앞단이다. 운영 랜에서도 보려고 **별도 포트**에 블록을 하나 더 둔다(`TOWER_PORT` 평문 · `TOWER_TLS_PORT` TLS). 현황판 블록과 합치지 않는 이유는, 현황판 쪽은 `/viewer`·`/uplink`·`/live/` 외에는 아무것도 내보내지 않아야 하는데 관제탑은 **코어 전체**(페이지 · `ws /control` · `ws /uplink` · `GET /site` · `/vworld` 타일 프록시)가 필요해서다.

접근 제한은 `TOWER_ALLOW` 로 적고, 설치 스크립트가 `allow` 줄들 뒤에 `deny all` 을 붙여 ACL 파일로 쓴다(nginx 는 순서대로 읽는다). 기본값은 `127.0.0.1 10.9.8.0/24 10.9.9.0/24` 다.

정적 파일 쪽에 두 가지가 더 들어 있다.

- **wasm MIME.** 현황판은 wasm 이라 `application/wasm` 이 아니면 브라우저가 모듈을 거부한다. nginx 의 `mime.types` 에 wasm 이 들어간 것은 1.21.1 부터인데 Ubuntu 22.04 는 1.18 이다. 그래서 `.wasm` location 에서 `types { }` 로 비우고 `default_type` 으로 못박았다. `check-bridge.sh` 가 실제 응답의 Content-Type 을 확인한다.
- **gzip 과 캐시.** wasm·JS 는 gzip 으로 보내고, 모든 정적 응답에 `Cache-Control: no-cache` 를 붙인다. 빌드를 덮어써도 파일 이름이 그대로여서 캐시가 남으면 옛 화면이 뜬다. `no-cache` 는 "쓰지 말라"가 아니라 "쓸 때마다 물어보라"라서, 안 바뀌면 304 로 끝난다.
- **wasm 은 압축 수준 1 이다.** 현황판 wasm 이 22.7 MB 다. 파이 5 에서 재 보니 요청당 level 1 은 CPU 0.50초·전송 9.9 MB, level 5 는 0.86초·9.1 MB, level 9 는 1.98초·9.0 MB 였다. 지휘관 30명이 한꺼번에 들어오는 상황에서는 바이트 8% 를 더 보내는 쪽이 CPU 를 3분의 1 덜 쓰는 것보다 비싸지 않다. 이 nginx 빌드에는 미리 압축해 두는 `gzip_static` 모듈이 없다.

터널 입구는 브리지 로컬에만 열린다(`127.0.0.1:20889`, `127.0.0.1:20882`). 대전 프록시 `:8082`는 브리지 밖에서 직접 닿지 않는다.

같은 네 경로를 nginx 없이 흉내내는 참조 구현이 `src/demo/board.rs` 에 있다. 설정을 고치면 그쪽 주석·테스트와 어긋나지 않는지 같이 본다.

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

## 무엇이 설치되나 (macOS)

| 항목 | 위치 |
|---|---|
| Homebrew 패키지 | `nginx`, `autossh` |
| 터널 키 | `/opt/skylens/tunnel/tunnel_key(.pub)` — 실행한 계정 소유, 0600 |
| 터널 서비스 | LaunchDaemon `com.skylens.daejeon-tunnel` — `autossh` 에 포워드 둘(`20889→10889`, `20882→8082`). 끊기면 자동 재연결, 재부팅 후 자동 시작 |
| nginx | `$(brew --prefix)/etc/nginx/servers/skylens.conf`. 이미 떠 있으면 reload 만, 없으면 LaunchDaemon `com.skylens.nginx`(root, 80 포트용) |
| 현황판 정적 파일 | `WEB_DIR`(기본 `/opt/skylens/web`). 설치 시 디렉터리만 만든다 |
| MediaMTX | `/opt/skylens/mediamtx/`, LaunchDaemon `com.skylens.mediamtx` |
| 로그 | `/opt/skylens/logs/*.log` |
| macOS 방화벽 | 켜져 있을 때만 mediamtx·nginx 허용 추가 |

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

macOS:

```bash
for s in daejeon-tunnel mediamtx nginx; do sudo launchctl bootout system/com.skylens.$s 2>/dev/null; done
sudo rm -f /Library/LaunchDaemons/com.skylens.*.plist
sudo rm -f "$(brew --prefix)/etc/nginx/servers/skylens.conf"
sudo rm -rf /opt/skylens
```

## 막히면

- `check-bridge.sh`에서 `127.0.0.1:20889 없음` + `Permission denied` — 정상. 공개키가 아직 대전에 등록되지 않은 것. 보내고 기다리면 된다.
- `127.0.0.1:20882 없음` 인데 `20889`는 열림. 대전 키의 `permitopen`에 8082가 없다. 대전에서 `register-bridge-key.sh`를 다시 돌린다.
- `20889`·`20882` 둘 다 없음, 로그에 `administratively prohibited` 또는 `open failed`. 대전 키가 옛 권한이라 8082 포워드가 거부된 것. 위 "설치 순서"를 따른다.
- `대전 SSH 도달 불가` — 이 서버 IP 가 KOREN 에 등록돼 있는지 확인(`175.126.98.44` 여야 함).
- `nginx -t` 실패 — 포트 충돌. `bridge.env` 의 `HTTP_PORT` 를 바꾸고 재실행.
- `/` 가 404. `WEB_DIR`에 현황판 파일이 없거나 `status.html`이 빠진 것. 4단계 복사를 확인한다.
- `/viewer` 연결 실패. `check-bridge.sh`의 "대전 프록시 터널" 항목을 본다. 프록시(`:8082`)가 대전에서 떠 있는지도 확인한다.
- 화면이 하얗고 콘솔에 `Incorrect response MIME type` / `expected application/wasm`. wasm 이 `application/octet-stream` 으로 나간 것. `check-bridge.sh` 의 "wasm MIME" 줄을 보고, nginx 설정이 이 저장소 판인지 확인한다(옛 설정에는 `.wasm` location 이 없다).
- `/live/droneN/whep` 가 404. 대전 MediaMTX 의 `paths:` 에 그 이름이 없는 것. `mediamtx.yml`·`mediamtx-bridge.yml` 둘 다 `drone1`~`drone3` 인지 본다. 시그널링은 되는데 화면이 검으면 미디어(대전 `10189/udp`) 쪽이므로 nginx 와 무관하다.

macOS 에서만 겪는 것:

- 맥이 잠자기에 들면 서비스가 멈춘다 — 시스템 설정 → 에너지 절약에서 **잠자기 방지**, 또는 `sudo pmset -a sleep 0`.
- 설치 중 `nginx.conf 에 'include servers/*;' 가 없습니다` 경고 — Homebrew nginx 의 `nginx.conf` `http { }` 안에 그 줄을 넣어야 이 설정이 읽힌다.
- `HTTP_PORT=8080` 은 Homebrew nginx 기본 서버와 겹쳐 설치가 멈춘다. 80 이나 다른 포트를 쓴다.
- 이미 일반 계정으로 nginx 가 떠 있으면 80 포트를 못 연다. 그 nginx 를 내리고 다시 설치하거나 `HTTP_PORT` 를 1024 이상으로 둔다.
