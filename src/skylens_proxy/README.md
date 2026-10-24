# skylens_proxy: KOREN 내부망 다중 경로

프록시가 존재하는 이유는 하나다. **재난 상황에서 코어로 가는 한 경로가 죽어도 서비스가 유지되어야 한다** ([COMPONENTS.md](https://github.com/NET-Challenge-S13/skylens/blob/develop/docs/COMPONENTS.md) §3.3). 이 컴포넌트는 코드가 없고 **nginx 설정 파일**이 전부다.

```
게이트웨이(브리지) ──SSH 터널──> 프록시 :8082 ─┬─> 코어 경로 #0 :8080 (선호)
                                               └─> 코어 경로 #1 :8180 (대기, backup)
```

| 파일 | 설치 위치 | 내용 |
|---|---|---|
| `nginx-skylens-proxy.conf` | `/etc/nginx/conf.d/` | upstream 두 경로 + 페일오버 규칙 |
| `skylens-proxy-headers.conf` | `/etc/nginx/snippets/` | 모든 location 공용 헤더·타임아웃 |

## 동작

- 모든 요청(HTTP·WebSocket)을 `skylens_core` upstream으로 넘긴다. `/core/health`는 코어의 `/health`로 간다.
- 0번 경로가 연결 거부·2초 무응답·502/503이면 대기 경로로 넘긴다(`proxy_next_upstream`). `fail_timeout`(3초)이 지나면 0번을 다시 시도하므로 복귀(failback)도 자동이다.
- `GET /health`는 프록시 자신의 생존만 답한다.

## TS 구현과 달라진 점

nginx 오픈소스판은 **수동 헬스체크**만 한다. 그래서 TS 프록시가 하던 것 중 셋이 빠진다.

| TS 프록시 | nginx |
|---|---|
| 대기 경로도 연결을 유지하며 ping/pong으로 능동 프로브 | 실제 요청이 실패해야 경로를 내린다 |
| 페일오버 때 열린 WebSocket을 다른 경로로 옮김 | 열린 소켓은 끊기고, 클라이언트가 재접속해서 살아 있는 경로로 간다 |
| 모든 경로가 죽으면 프레임을 버퍼링 | 버퍼링 없음. 502를 받은 드론·화면이 재시도한다 |

재접속과 재시도는 드론(`skylens_drone`)과 클라이언트(`skylens_client`) 쪽 책임으로 옮긴다.

## 설치 (대전)

```bash
sudo cp skylens-proxy-headers.conf /etc/nginx/snippets/
sudo cp nginx-skylens-proxy.conf   /etc/nginx/conf.d/
sudo nginx -t && sudo systemctl reload nginx
curl -s http://127.0.0.1:8082/health        # 프록시
curl -s http://127.0.0.1:8082/core/health   # 프록시 → 코어
```
