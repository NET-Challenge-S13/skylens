# SkyLens Lito X1 Mission Bridge

SkyLens 웹에서 만든 GPS 경로를 DJI Fly가 읽는 KMZ 임무로 바꾸는 로컬 서비스다.
Lito X1은 공개 DJI SDK를 지원하지 않으므로 실시간 스틱 명령을 보내지 않는다.
대신 기체가 공식 지원하는 **DJI Fly 웨이포인트 비행**을 사용한다.

```text
관제 웹 경로 → localhost Mission Bridge → KMZ → DJI Fly → Lito X1 온보드 비행
```

참고 자료:

- DJI Lito X1 공식 매뉴얼, 웨이포인트 비행 §4.5:
  https://dl.djicdn.com/downloads/DJI_Lito_X1/UM/20260417/DJI_Lito_X1_User_Manual_ko.pdf
- DJI SDK 호환성 표(Lito X1은 SDK 미지원):
  https://repair.dji.com/help/content?customId=01700000763&lang=en&paperDocType=ARTICLE&re=US&spaceId=17
- Litchi Hub의 Lito X1 → DJI Fly KMZ 전송 가이드(독립 검증용):
  https://forum.flylitchi.com/t/guide-how-to-use-litchi-hub-with-dji-fly/24332

## 1. 준비물

필수:

- DJI Lito X1
- DJI RC 2 또는 DJI RC-N3 + Android/iPhone
- 최신 DJI Fly와 기체·조종기 펌웨어
- SkyLens를 실행할 PC와 USB 데이터 케이블
- 비행 가능한 개방 장소, 현장 조종사, 관찰자
- 이 저장소의 Node.js/npm과 Python 3.11 이상

선택:

- Litchi Hub 및 Litchi Hub Bridge: KMZ 경로가 실제 DJI Fly에 들어가는지 먼저 검증할 때 사용
- OpenMTP(오픈소스): macOS에서 Android/RC의 MTP 파일을 수동으로 다룰 때 사용
  https://github.com/ganeshrvel/openmtp
- Android Platform Tools: RC-N3에 연결할 Android 기기가 ADB 파일 전송을 허용할 때만 사용
  https://developer.android.com/tools/releases/platform-tools

Litchi는 오픈소스가 아니며 SkyLens 실행에 필수도 아니다. 이 디렉터리의 Bridge는 Python
표준 라이브러리만 사용하고, 저장소의 MIT 라이선스를 따른다.

## 2. 최초 한 번: Lito X1 템플릿 KMZ 확보

소비자용 DJI Fly KMZ의 기체·페이로드 enum은 DJI 공개 문서에 없다. 따라서 다른 기체의
값을 추측하면 안 된다. Lito X1과 DJI Fly가 직접 만든 정상 임무를 템플릿으로 사용한다.

1. 기체와 조종기를 켜고 DJI Fly에 연결한다.
2. DJI Fly 카메라 화면에서 웨이포인트 모드로 들어간다.
3. 실제 시험 장소 근처에 웨이포인트 2개를 둔다.
4. 고도 30m, 속도 2m/s, 종료 동작 RTH, 신호 유실 시 호버로 설정한다.
5. 이름을 `SKYLENS_TEMPLATE_DO_NOT_FLY`로 저장하고 DJI Fly를 완전히 종료한다.
6. DJI Fly 장치를 PC에 파일 전송 모드로 연결한다.
7. Android/RC 계열에서는 아래 경로의 가장 최근 임무 폴더를 찾는다.

   ```text
   Android/data/dji.go.v5/files/waypoint/<mission-id>/<mission-file>.kmz
   ```

8. 그 KMZ를 저장소의 아래 위치로 **복사**한다. 원본을 이동하거나 삭제하지 않는다.

   ```text
   config/lito-x1-template.kmz
   ```

9. 템플릿은 실제 위치와 기체 식별정보를 포함할 수 있으므로 `.gitignore`에 등록되어 있다.

RC 2는 USB-C로 PC에 직접 연결한다. RC-N3는 휴대폰의 USB-C가 조종기 연결에 필요하므로,
임무 파일을 복사할 때는 휴대폰을 RC-N3에서 잠시 분리한다. iPhone/iPad는 DJI Fly의 파일
접근 방식이 다르므로, 첫 실증에는 Litchi Hub Bridge의 Smart Export를 권장한다.

## 3. Bridge 실행

저장소 루트에서:

```bash
npm run mission-bridge
```

정상이면 다음과 비슷하게 표시된다.

```text
[mission-bridge] listening on http://127.0.0.1:8091
[mission-bridge] Lito template: .../config/lito-x1-template.kmz (ready)
[mission-bridge] output: .../output/missions
[mission-bridge] install target: not configured (download-only mode)
```

상태 확인:

```bash
curl http://127.0.0.1:8091/health
```

`templateReady`가 `true`여야 한다. 템플릿이 없으면 Bridge는 살아 있지만 KMZ 생성을
거부한다. M30 같은 임의의 기본값으로 대신 생성하지 않는다.

## 4. SkyLens 실행과 임무 생성

터미널을 각각 열어 실행한다.

```bash
npm run dev
npm run core
npm run mission-bridge
```

관제 화면을 열고 `경로 계획 · Route`에서:

1. 실제 시험 장소로 지도를 이동한다.
2. 처음에는 웨이포인트 2개만 둔다.
3. 지면 기준 고도를 30m로 둔다.
4. 왕복은 첫 시험에서 끈다.
5. `경로 전송`을 누른다.

Bridge가 실행 중이면 KMZ가 `output/missions/`에 저장되고 브라우저에서도 다운로드된다.
왕복을 켜면 무한 반복하지 않고 `출발 → 끝 → 출발` 1회 왕복으로 변환된다.

Bridge 기본 안전 제한:

| 항목 | 제한 |
|---|---:|
| 상대고도 | 5-120m |
| 속도 | 0.5-10m/s |
| 총 경로 | 5km 이하 |
| 웨이포인트 | 200개 이하 |
| 짐벌 피치 | -90-30° |
| 종료 동작 | RTH 고정 |
| 신호 유실 | 호버 고정 |

SkyLens 경로의 고도는 해발고도이고 DJI Fly는 이륙점 기준 상대고도를 사용한다. 현재
Bridge는 별도 이륙점 고도가 없으면 첫 웨이포인트의 지면고도를 이륙점으로 추정하고 경고를
남긴다. DJI Fly에서 실제 이륙점과 모든 웨이포인트 상대고도를 반드시 다시 확인한다.

## 5. DJI Fly로 전송

### 안전한 첫 단계: 다운로드 모드

1. DJI Fly에서 `SKYLENS_TARGET`이라는 임시 임무를 만들고 저장한다.
2. 앱을 완전히 종료한다.
3. PC에 장치를 연결하고 그 임무의 KMZ를 별도 폴더에 백업한다.
4. SkyLens가 내려받은 KMZ의 이름을 기존 파일과 같게 바꾼다.
5. 기존 `SKYLENS_TARGET` KMZ를 새 파일로 교체한다.
6. 장치를 안전하게 분리하고 DJI Fly를 다시 연다.
7. 임무를 열어 지도, 고도, 속도, 짐벌, 촬영, RTH를 하나씩 확인한다.

### 선택: 명시적 자동 설치

장치의 임무 폴더가 PC에 안정적으로 마운트되어 있고 정확한 KMZ 경로를 확인한 뒤에만 쓴다.

```bash
export SKYLENS_DJI_INSTALL_TARGET='/exact/path/to/SKYLENS_TARGET.kmz'
npm run mission-bridge
```

Bridge는 지정된 **그 한 파일만** 교체하며, 교체 전 파일은
`output/missions/backups/`에 복사한다. 디렉터리나 자동 탐색 결과를 삭제하지 않는다.
macOS의 MTP 장치는 일반 파일시스템처럼 마운트되지 않는 경우가 많아 자동 설치가 어려울 수
있다. 그런 경우 다운로드 모드 또는 Litchi Hub Bridge를 사용한다.

### 선택: DJI 클라우드 원격 전달

PC와 조종기가 USB로 붙어 있지 않아도, DJI Fly 계정 클라우드에 임무를 올려 원격의 RC 2가
Wi-Fi로 받아가게 할 수 있다. 조종기에 임무가 내려오면 조종사가 열어 확인하고 비행한다.

설정은 `config/mission-bridge.env.example`의 DJI 클라우드 항목 두 개를 로컬 환경에만 넣는다.
둘 다 계정/앱 비밀값이라 **저장소에 커밋하지 않는다**. 둘 중 하나라도 없으면 이 경로는
자동으로 꺼지고 다운로드/설치 모드만 동작한다.

- `SKYLENS_DJI_MC_TOKEN`: 로그인된 DJI Fly 세션의 `x-mc-token` (만료되므로 업로드가 인증
  오류를 내면 다시 추출한다)
- `SKYLENS_DJI_WK_KEY`: `X-Wk-SecretId=uav` 요청 서명에 쓰는 HMAC 키

설정한 뒤, 경로 전송 요청 본문에 `"deliver": "cloud"`를 넣으면 KMZ를 만든 다음 클라우드에
올린다. 응답의 `cloud.missionUuid`로 조종기에 뜰 임무를 식별한다. 편대 3기도 각각 올라간다.

관제탑 웹에서는 경로 계획 창의 **"클라우드 전송(원격 RC)"** 체크로 켜고, 툴바의
**"클라우드 미션 · Cloud"** 버튼으로 현재 클라우드 임무를 **목록 조회·삭제**할 수 있다.

클라우드 임무 파일은 raw KMZ가 아니라 **래퍼 zip**(`<uuid>.kmz` + `image/ShotSnap.json`)이어야
DJI Fly가 연다. Bridge가 자동으로 감싸므로 여기서 신경 쓸 것은 없다.

### 토큰 자동 갱신

`x-mc-token`은 만료되므로 Bridge를 재시작하지 않고 갱신하는 길을 둔다.
`SKYLENS_DJI_TOKEN_FILE`에 토큰 파일 경로를 지정하면 Bridge는 **매 요청마다 그 파일을
다시 읽고**, 인증 오류가 나면 파일을 다시 읽어 **한 번 재시도**한다. 그 파일을 최신으로
유지하는 것이 갱신기다.

```bash
# 로그인된 DJI Fly가 떠 있는 WSA(또는 루팅 안드로이드) + frida-server + pip install frida 필요
python scripts/dji_token_refresh.py            # 실행 중인 앱에서 토큰을 읽어 파일에 기록
```

`scripts/dji_token_refresh.py`는 bridge의 일부가 아니라 추출 리그 도구다(frida 사용). 앱이
스스로 갱신하는 살아있는 토큰을 그대로 읽어 파일에 쓴다. 한 번 실행하거나, 작업
스케줄러/cron으로 주기 실행하면 Bridge가 자동으로 새 토큰을 집어쓴다. 완전 무인화는
WSA 리그가 떠 있어야 가능하다.

> 이 경로는 공개 SDK가 아니라 DJI Fly 클라우드 동기화를 관찰해 맞춘 것이다. DJI Fly/펌웨어
> 업데이트 후에는 2점 지상 검증을 다시 한다.

## 6. 첫 실비행 체크리스트

첫 실비행은 웹에서 이륙시키는 시험이 아니다. 조종사가 DJI Fly에서 최종 확인하고 실행한다.

- 프로펠러·배터리·짐벌·GNSS·나침반 상태 정상
- 지도상의 장소가 현재 시험장과 일치
- 홈포인트가 실제 이륙점에 기록됨
- 모든 고도가 **이륙점 기준**으로 안전함
- 직선 구간 아래와 옆에 사람·차량·전선·수목·건물 없음
- 속도 2m/s, 웨이포인트 2개, 왕복 끔
- RTH 고도가 주변 장애물보다 높음
- 신호 유실 시 호버 설정 확인
- 조종사가 RC를 잡고 비행 일시 정지/RTH 버튼을 즉시 누를 수 있음
- 관찰자가 기체를 계속 육안으로 확인
- 첫 비행은 수동 이륙 → 안전고도 호버 → 임무 시작 → 중간 일시 정지 → 수동 착륙

DJI Fly가 임무를 읽지 못하거나 값이 다르면 비행하지 않는다. DJI Fly/기체 펌웨어 업데이트
후에도 같은 2점 지상 검증을 반복한다.

## 7. 현재 가능한 범위와 다음 단계

가능:

- SkyLens 웹에서 실제 GPS 경로 생성
- Lito X1 식별값이 들어간 KMZ 생성
- 속도·상대고도·짐벌·녹화 시작/종료·RTH 정의
- 다운로드 또는 명시적 단일 파일 교체
- **편대 3기 KMZ 자동 생성**: 관제 경로 계획에서 `편대 3기 KMZ`를 켜고 간격(m)을
  넣으면, 계획한 경로를 편대 무게중심 트랙으로 보고 앞왼·앞오·뒤중 3개 미션을
  한 번에 만든다(정삼각형, 기본 변 10m). 예전에 엑셀로 6좌표를 손계산해 3개 미션에
  직접 입력하던 과정을 대체한다. 자세한 기하는 `formation.py` 참조
- **DJI 클라우드 원격 전달**: `deliver:"cloud"`로 KMZ를 계정 클라우드에 올려 원격 RC 2가
  Wi-Fi로 받게 한다. 목록(`GET /cloud/missions`)과 삭제(`POST /cloud/delete`)도 된다.
  자세한 설정은 §5의 "DJI 클라우드 원격 전달" 참조

아직 불가능:

- 웹 W/A/S/D 실시간 스틱 제어
- 웹에서 DJI Fly 임무 시작 버튼 누르기
- 공식 API를 통한 실시간 Lito X1 텔레메트리·영상 수신
- 실제 Lito 템플릿 없이 완전한 호환성 검증

> **payload 블록 처리**: Lito X1처럼 통합 카메라 기체의 DJI Fly/Litchi KMZ에는
> `<wpml:payloadInfo>`가 없다. Bridge는 템플릿에 payload enum이 있으면 그대로 싣고,
> 없으면 블록을 생략한다(엔터프라이즈 기체 값을 짐작해 넣지 않는다).

실기체 템플릿을 확보한 다음 해야 할 일은 `생성 KMZ를 DJI Fly에서 열기 → 값 비교 →
프로펠러 없는 상태의 지상 검증 → 개방 장소 2점 저속 시험` 순서다.

## 8. API

`POST /formation` — 계획 경로 하나를 편대 3기 미션으로 팬아웃. 요청은 아래
`POST /missions` 본문에 `"formation": {"spacingM": 10}`을 더한 형태다(단, `droneId`는
무시). 응답은 `members` 배열로, 각 항목이 `station`(`left`|`right`|`center`)과
그 기체의 KMZ `downloadUrl`을 담는다. 자동 설치는 하지 않는다(대상 파일 하나에
3개를 설치할 수 없으므로 항상 다운로드 전용).

### 클라우드 전달 API

- `POST /missions` 또는 `POST /formation` 본문에 `"deliver": "cloud"`를 더하면 KMZ 생성
  후 DJI 클라우드에 올리고, 응답에 `"cloud": {"missionUuid": ..., "verified": true}`를 넣는다.
  클라우드 미설정 시 이 요청은 502로 거부되며 다운로드/설치 요청은 영향받지 않는다.
- `GET /cloud/missions` 계정 클라우드의 임무 목록(`uuid`, `name`, `waypointCount`, `distanceM`).
  관제탑 툴바 "클라우드 미션 · Cloud" 버튼이 이 결과를 보여주고 각 항목을 삭제한다.
- `POST /cloud/delete` 본문 `{"missionUuid": "..."}` 로 해당 임무를 삭제.
- `GET /health`의 `cloudConfigured`로 클라우드 설정 여부를 확인한다.
- 토큰 만료 대비: `SKYLENS_DJI_TOKEN_FILE`에 토큰 파일을 두면 매 요청 재로딩하고 인증
  오류 시 1회 재시도한다. 파일은 `scripts/dji_token_refresh.py`가 갱신한다(위 §5 참조).

`POST /missions`

```json
{
  "name": "skylens-drone-1",
  "droneId": 1,
  "deliver": "cloud",
  "loop": false,
  "flight": {
    "aglM": 30,
    "takeoffAltMsl": 52,
    "speedMps": 2,
    "gimbalPitchDeg": -30,
    "rthHeightM": 40
  },
  "waypoints": [
    { "lat": 36.3685, "lon": 127.3475, "alt": 82 },
    { "lat": 36.3687, "lon": 127.3477, "alt": 84 }
  ]
}
```

기본 바인딩은 `127.0.0.1`이며 CORS도 localhost의 5173/8080 포트만 허용한다. 다른
관제 PC에서 접근할 때는 `SKYLENS_MISSION_HOST`와 `SKYLENS_MISSION_ALLOWED_ORIGINS`를
명시적으로 설정하고, 신뢰할 수 있는 현장망 밖으로 노출하지 않는다.
