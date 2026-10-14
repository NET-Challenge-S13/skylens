# skylens-control (관제탑)

`skylens_core`의 관제탑 화면이다. 이전 TypeScript 관제탑(`control.ts`, `controlview/towerViewer.ts`, `control/`, `ui/`, `drones/`)을 Rust로 옮겼다. HTML과 CSS는 이전 것을 그대로 쓰고(`ui/control.html`, `ui/control.css`, `ui/panels.css`, 손대지 않는다), 화면은 Blitz가, 3D는 pointblitz가 그린다. 런타임 DOM 변경은 `ui/DOM.md`를 따른다.

Go 코어가 `dist/`를 `/`에서 서빙하고, 같은 코드가 네이티브 창으로도 돈다.

## 실행

```bash
# 네이티브 창
cargo run -p skylens-control --release -- --core ws://127.0.0.1:8080/control

# 웹 빌드 (dist/에 index.html, pkg/, favicon.png)
bash src/skylens_core/control/build-web.sh
```

| 옵션 (네이티브) | 웹 쿼리 | 기본값 | 뜻 |
|---|---|---|---|
| `--core` | `?core=` | `ws://127.0.0.1:8080/control` (웹은 페이지와 같은 호스트의 `/control`) | 코어 소켓. 호스트만 주면 `/control`을 붙인다 |
| `--display` | `?display=` | 저장된 값, 없으면 `black` | 건물 표시 방식(`points`, `black`, `aerial`). 이번 실행에만 쓰고 저장하지 않는다 |
| `--bridge` | `?missionBridge=` | `http://127.0.0.1:8091` | DJI Fly 미션 브리지(`skylens_drone`) |
| `--video` | 없음 | `srt://{host}:10890?streamid=read:{path}` (`SKYLENS_CONTROL_VIDEO_URL`) | 영상 패널의 ffmpeg 입력. `{host}` 코어 호스트, `{id}`, `{station}`, `{path}`(WHEP 주소의 MediaMTX 경로, 없으면 `drone{id}`), `{whep}`. `off`면 대기 화면만 |
| `--ffmpeg` | 없음 | `ffmpeg` (`SKYLENS_FFMPEG`) | ffmpeg 실행 파일 |

설정 저장: 웹은 이전과 같은 `localStorage` 키(`skylens.control.settings.v1`, `skylens.control.mapCenter.v1`), 네이티브는 `%APPDATA%/skylens/control.json`(리눅스 `~/.config/skylens/control.json`, `SKYLENS_CONTROL_SETTINGS`로 바꿀 수 있다). KMZ는 네이티브에서 사용자 `Downloads` 폴더(`SKYLENS_DOWNLOAD_DIR`)에 저장한다.

## 흐름

1. `ws /control`에 붙으면 코어가 `Site`를 먼저 보낸다. `Site.anchor`가 ENU 원점이다. 이어서 보관분(미션, 경로, 텔레메트리, 영상 위치, 지형 `TerrainTile`, 건물 `BuildingSet`)이 온다.
2. 로딩 화면(`관제탑 · 실지형 장면 로딩`)은 작전 구역을 덮는 정밀 지형 셀이 다 오면 사라진다. 5초 동안 새 셀이 없으면(코어가 지형을 끈 경우) 그대로 닫고, `Site` 전에 연결이 실패하면 실패 문구를 보여 준다.
3. `assign_route`와 `manual_control`은 이 소켓으로만 보낸다.
4. 경로가 배정되기 전이면 코어가 `Site` 바로 다음에 `planned_route`(미리 계획된 경로)를 보낸다. 경로 계획에 운용자가 찍은 지점이 없을 때만 그 지점과 왕복 여부를 채우고(경로가 지도 밖이면 지도를 경로 쪽으로 옮긴다), 아무것도 보내지 않는다. 운용자는 경로 계획을 열어 배정만 누르면 된다. 화면의 HTML·CSS는 바뀌지 않는다.

## 모듈

| 파일 | 하는 일 |
|---|---|
| `app.rs` | 화면 전체 배선(`ScreenApp`): 코어 연결, 패널, 경로 계획, 브리지 호출, 키 |
| `tower.rs` | 3D 뷰(`ViewScreen`): 표시 방식별 STYLES, 지형 드레이프, 건물 프리즘과 윤곽선, 점 구름, 드론 와이어프레임과 궤적, 경로, 개요 회전과 추적 카메라 |
| `terrain.rs` | DEM 샘플링, 이전 차가운 색보정(LUT, 격자선), 지형 메시 |
| `buildings.rs` | 프리즘 돌출, 지붕 삼각분할, 윤곽선, 점 샘플링(`mulberry32`), 높이 램프 |
| `planner.rs`, `slider.rs`, `glyphs.rs` | 경로 계획 지도 위젯, 고도 슬라이더 위젯, 지도 위 글자 |
| `modals.rs`, `panels.rs`, `video.rs` | 경로 계획과 클라우드 미션 대화상자, 임무와 함대와 설정 패널, 영상 패널 |
| `fleet.rs`, `manual.rs` | 텔레메트리 트랙과 선택 규칙, 키보드 수동 조종 |
| `bridge.rs`, `http.rs`, `download.rs`, `storage.rs`, `settings.rs`, `labels.rs` | 브리지 요청과 응답, 백그라운드 HTTP, KMZ 저장, 설정 저장, 문구와 숫자 형식 |

## 이전 화면과 맞춘 것

- 장면 단위: 이전 장면은 작전 구역의 긴 변을 44단위로 정규화했다. 여기서는 ENU 미터를 그대로 쓰고, 이전 코드가 장면 단위로 정한 값(안개 거리, 카메라 오프셋, 드론 크기, 패드 크기, 점 크기)에 같은 구역에서의 단위당 미터를 곱한다. 수직은 1.5배 과장한다.
- 키: 화살표(전진/후진, 좌우 선회, 10Hz 전송, 놓으면 정지 프레임 한 번), Q/E 상승·하강, 1/2/3 드론 id 선택, Tab 순환, Esc는 설정만 닫는다. 수동 조종 중에는 `body.is-manual`.

## 이전 화면과 다른 것

- 웹 빌드는 영상을 재생하지 않는다(WHEP 디코더가 없다). 영상 패널은 대기 화면과 위치 문구만 보여 준다.
- 점 표시의 가산 혼합(AdditiveBlending)은 pointblitz 점 파이프라인에 없어 불투명도 0.92의 일반 혼합으로 그린다.
- 경로 계획 지도의 글자는 시스템 글꼴 대신 번들 Barlow로 그린다.
