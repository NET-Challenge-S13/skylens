# CLAUDE.md — SkyLens 작업 가이드

SkyLens는 **멀티드론 영상을 실시간 3D(Gaussian Splatting)로 복원하고 그 위에 AI가 위험구역·사람을 표시하는 재난 인텔리전스 플랫폼**이다 (NET 챌린지 캠프 시즌13).
이 저장소에는 **TypeScript 운영 프로토타입**과 **Python AI 모델 패키지**가 함께 들어 있다.

---

## 🚫 커밋 규정 (최우선)

> **커밋 메시지에 Claude를 공동저자로 넣지 말 것.**
>
> - `Co-Authored-By: Claude ...` 줄 **금지**
> - `🤖 Generated with Claude Code` 등 **어떤 형태의 AI 생성 표시도 금지**
> - PR 본문에도 AI 생성 표기를 넣지 않는다.
>
> 커밋 메시지는 사람이 쓴 것처럼 변경 내용만 담는다. 기존 히스토리 스타일(`feat:`, `fix:`, `docs:`, `test:` 접두사)을 따른다.

> **커밋 메시지는 영어로 쓴다.**
>
> - 제목·본문 모두 **한글 금지**. `feat: 경로계획 클릭 지도` (X) → `feat: click-to-plan route on map` (O)
> - 명령형 현재시제(`add`, `fix`, `move`), 제목은 소문자로 시작하고 마침표 없이.
> - 문서(`.md`) 본문은 계속 한국어로 쓴다 — 이 규칙은 **커밋 메시지에만** 적용된다.

또한 **요청받지 않았으면 커밋하지 않는다.**

---

## 1. 문서 지도

| 문서 | 역할 |
|---|---|
| `README.md` (루트) | 저장소 소개. 데모/실서버 모드, GPS↔ENU 좌표계, 빠른 시작(`npm run dev`), 관제탑/현황판 접속 주소와 쿼리 옵션, 조작법, 프로젝트 구조 트리, 현재 구현 상태·로드맵. **TypeScript 프로토타입(뷰어) 중심** |
| `PROJECT.md` (루트) | **중간평가 프로토타입 구현 계획**. 무엇을 증명하고 무엇을 증명하지 않는지(§0), 데모 컨셉(§1), 사전 촬영·gsplat 준비(§2), 기술 스택(§3), **관제탑·GPS 경로 계획(§4)**, **현황판·서버 스트림 3D 축적(§5)**, 위험구역 오버레이(§6), 연출과 실제의 경계 원칙(§7), 관제탑↔현황판 동기화·카메라 협력(§8), 구현 로드맵(§9)·리스크(§10)·최종평가 확장(§11) |
| `docs/COMPONENTS.md` | **컴포넌트 구성의 단일 출처**. 8개 컴포넌트(드론·게이트웨이·프록시·코어·모델·클라이언트·shared·데모)의 역할과 망 위치, Gateway/WebRTC 두 데이터 흐름, 관제탑 화면 통합 규칙(§4), 데모 시나리오(§5), 디렉터리 구조(§6), 포트 맵(§7), 구현상의 결정(§8) |
| `docs/INTENT.md` | **의도(왜)** — ResearchTree 인텐트 문서. 문제 정의(소방드론 현황·홍제동 사례·각주 출처), 목표, **주장 N1~N7**(실험이 검증하는 단위), 제약, 하지 않는 것, 열린 결정, KOREN 활용 논거 |
| `docs/SPEC.md` | **현재 설계의 단일 출처** — ResearchTree 스펙 문서. 파이프라인·컴포넌트 경계·프로토콜·좌표계·딜레이 패턴·두 화면·AI 모델·3DGS 복원·데모·KOREN 배포·포트·빌드·브랜치까지 16절. 각 절은 한 줄 요약(`>`)으로 시작한다. **지금 상태만 적고 역사는 적지 않는다** |
| `docs/ARCHITECTURE.md` | **통합 아키텍처(무엇을)**. 3대 설계 원칙, 4-Tier 구성(캡처/전송/Core HPC/Edge·클라이언트), 데이터 플로우, **§3-A AI 모델·융합 파이프라인**(UNet 4채널, Depth Map 레이캐스팅, Hybrid Fusion, 기술 선택 배제 근거) |
| `docs/DATASETS.md` | **학습 데이터셋 조사**. "RGB+열 페어 + 재난 + 사람 + 위험구역"을 모두 가진 공개 데이터는 없다는 결론과, A(4채널 정합)/B(위험구역 세그)/C(사람 인스턴스) 3축 조합 권장. FLAME 3, RescueNet, SARD, AIResQ, LLVIP, VisDrone 등 |
| `src/skylens_model/README.md` | **AI 모델 설계 철학의 단일 출처(어떻게)**. 레이어 분리 원칙(탐지/투영/랜드마크 융합/소리 보정), UNet 채택·TransUNet 보류 근거, 단일 백본+이중 헤드, modality dropout, 점 검출 헤드, 헤드별 분리 학습, 배제한 대안 표. **결정과 그 근거**를 기록 |
| `src/skylens_model/models/skylens/README.md` | **3DGS 복원 설계의 단일 출처**. 파이프라인 3단계(프레임 추출 → COLMAP SfM → gsplat 학습), 특징점을 ALIKED로 정한 근거, 전수매칭이 필요한 이유, 내부 파라미터 명시 + 자기보정 조합, 점진적 출력(30초 첫 화면), PLY 경량판. **촬영 가이드**(포즈 정확도 요구·편대 배치·드론 자동비행 가능 범위)와 증분 복원 가능 여부 결론 |
| `src/skylens_model/models/skylens/RESULTS.md` | 위 결정들의 **측정 근거**. 실험 5건 · 학습 조건 28개 수치 전체. 판정이 보류된 실험은 그 이유까지 명시 |
| `src/skylens_model/models/skylens/INSTALL.md` | COLMAP(CUDA 빌드)·gsplat 설치 절차, `recon` 의존성 그룹에서 뺀 항목과 이유, **밟았던 함정 10건** |
| `src/skylens_model/datasets/README.md` | 데이터셋 **API 계약**(`__getitem__` 반환 dict, `None`은 정상값), 통합 클래스 스키마(0 normal / 1 fire / 2 collapse / 3 road_blocked / 255 ignore), RescueNet·VisDrone 매핑, 자동 다운로드 가능 여부 판정 |
| `src/skylens_model/utils/README.md` | `geo.py`가 `src/skylens_core/geo.ts`의 순수 파이썬 미러라는 사실 — **두 파일은 수치적으로 동기 유지** |
| `train.ipynb` (루트) | 데이터셋 → 학습 → 추론 결과를 마커 좌표로 흘려보내는 학습 노트북 |

### 문서 간 관계

```
docs/INTENT.md          왜 이걸 하는가 · 무엇을 주장하는가 (문제·목표·주장 N1~N7·열린 결정)
      ↓
docs/SPEC.md            지금 무엇으로 되어 있는가 (현재 설계의 단일 출처)
      ↓
docs/ARCHITECTURE.md    4-Tier 상세, §3-A 모델·융합 파이프라인
      ↓
PROJECT.md              중간평가에서 어디까지 만드는가 (범위 축소 + 연출 계획)
      ↓
README.md               실제로 만들어진 것을 어떻게 돌리는가

실험 PR                 무엇을 시도했고 어떤 수치가 나왔는가 (근거는 전부 여기)

docs/DATASETS.md   ──→  src/skylens_model/README.md  ──→  datasets/README.md
(학습 데이터 근거)       (모델을 어떻게 설계할 것인가)      (그 설계의 코드 계약)
```

- `docs/DATASETS.md`는 `ARCHITECTURE.md §3-A`를 학습 가능한 형태로 뒷받침하고, 모델 README의 여러 결정(데이터 부족 → UNet, modality dropout, 헤드별 분리 학습)의 **직접 근거**다.
- ⚠️ `src/skylens_model/README.md`는 상위 문서보다 **최신**이다. `ARCHITECTURE.md`/`INTENT.md`에 남은 "UNet / TransUNet 병기", "인스턴스 헤드"(→ 점 검출 헤드) 같은 표현은 아직 정정 대기 상태다. **모델 문서가 우선한다.**

### 작업별 필독 문서

| 하려는 작업 | 먼저 읽을 것 |
|---|---|
| AI 모델 수정·추가 (`src/skylens_model/models`) | **`src/skylens_model/README.md` 필독** → `docs/ARCHITECTURE.md §3-A` |
| 데이터셋 클래스 추가·수정 | `src/skylens_model/datasets/README.md` → `docs/DATASETS.md` |
| 학습 노트북 / 학습 루프 | `src/skylens_model/README.md` §6(학습 전략) → `train.ipynb` |
| 컴포넌트 경계·서버·프로토콜 관련 작업 | **`docs/COMPONENTS.md` 필독** → `src/shared/protocol.ts` |
| 데모·뷰어(관제탑/현황판) UI·연출 작업 | **`docs/COMPONENTS.md`** → `PROJECT.md` → `README.md` |
| 좌표계·GPS 관련 | `README.md` 좌표계 절 → `src/skylens_core/geo.ts` + `src/skylens_model/utils/geo.py` (둘 다 고칠 것) |
| 3DGS 복원·촬영 계획 (`src/skylens_model/models/skylens`) | **`src/skylens_model/models/skylens/README.md` 필독** → `RESULTS.md`(수치) → `INSTALL.md`(환경) |
| 기획 문구·발표 자료 | `docs/INTENT.md` → `docs/ARCHITECTURE.md` |
| 실험 시작·기록·결론 | **§5 ResearchTree** → `researchtree spec --intent` · `--summary` · `--claims` |

---

## 2. 저장소 구조

```
res/static/          # 정적 html 셸(진입점): index / control / status .html → /src 모듈을 절대경로로 로드
                     # demo/ — 딜레이 패턴용 구간×수준 PLY (생성물, 커밋하지 않음)
src/
├─ shared/           # 컴포넌트 공통 계약. 순수층(geo·protocol·types: DOM·Three 없음)
│                    #  + viewer/(두 웹 UI 공용 브라우저 층) + net/(WebRTC 트랜스포트)
├─ skylens_drone/    # Tauri 드론 클라이언트 (현장)
├─ skylens_gateway/  # KOREN 외부망 진입점 (relay | webrtc 2모드)
├─ skylens_proxy/    # KOREN 내부망 다중 경로
├─ skylens_core/     # KOREN 내부망: 관제탑 UI(control.ts·controlview/·ui/) + server/(오케스트레이터·스토어·배포)
├─ skylens_model/    # KOREN 내부망: FastAPI 연산 서버(app.py) + 모델·3DGS 복원 파이프라인
├─ skylens_client/   # KOREN 외부망: 현황판 UI(status.ts·statusview/·ui/) + server/(웹서버·WebRTC 중계)
└─ demo/            # 데모 런처 (컴포넌트 조립 + 모킹)
src/test/smoke.spec.ts  # Playwright E2E
docs/            # INTENT · SPEC · COMPONENTS · ARCHITECTURE · DATASETS · NETWORK_ARCHITECTURE
```

컴포넌트 경계와 각 컴포넌트의 책임은 **`docs/COMPONENTS.md`가 단일 출처**다.

두 스택은 **`src/` 아래에 공존**하고, 세 진입 html은 **`res/static/`** 에 모여 있다(루트는 설정 파일만). 접속 URL은 `/res/static/{control,status}.html` — `/`는 랜딩을 자동 서빙하지 않는다. `pyproject.toml`은 자동 탐색 대신 `packages = ["src/skylens_model"]`로 파이썬 패키지를 명시한다.

---

## 3. 빌드 · 테스트

**TypeScript** (`package.json`)

| 명령 | 설명 |
|---|---|
| `npm run dev` | Vite 개발 서버 (HMR, LAN 노출) |
| `npm run build` | `tsc` 타입체크 + 멀티페이지 빌드 |
| `npm test` | Playwright E2E |
| `npm run test:headed` | 브라우저 표시 E2E |
| `npm run preview` | 빌드 결과 미리보기 |

접속: `http://<IP>:5173/res/static/control.html?room=demo` / `/res/static/status.html?room=demo` (같은 `room`이면 WebRTC 연결, `&demo`로 자동 데모).

**Python** (`pyproject.toml`, requires-python >=3.11,<3.14 · `.python-version` = 3.13)

- 의존성은 **uv** 로 관리한다. `uv sync` 하면 `.venv/` 가 만들어지고 패키지가 editable 로 설치된다.
- 실행은 `uv run <cmd>` (예: `uv run pytest`, `uv run jupyter lab`). `.venv` 를 직접 activate 해도 된다.
- `uv.lock` 은 **커밋한다**(재현성). 의존성을 바꾸면 `uv lock` 후 락 파일도 함께 커밋.
- 그룹: `train` / `notebook` / `dev`(train+notebook 포함). 기본은 `dev` 가 설치된다.
- torch·torchvision 은 PyPI 가 아니라 **CUDA 12.8 인덱스**(`download.pytorch.org/whl/cu128`)에서 받는다 — `[tool.uv.sources]` 참조.
- 린트: `uv run ruff check src/` (line-length 100, target py311)
- 테스트: `uv run pytest` (`testpaths = src/skylens_model/tests`)

---

## 4. 기타 규약

- **학습 산출물·데이터는 커밋하지 않는다.** `.gitignore`가 `data/`, `runs/`, `outputs/`, `checkpoints/`, `wandb/`, `*.ckpt`, `*.pt`, `*.pth`, `*.safetensors`를 제외한다. 데이터셋은 저장소에 포함하지 않고 각 `Dataset` 클래스가 `root` 아래에서 찾는다.
- `.claude/`, `.omc/`도 git 제외 대상이다.
- `src/skylens_core/geo.ts` ↔ `src/skylens_model/utils/geo.py`는 **같은 수식의 두 구현**이다. 한쪽만 고치지 말 것.
- 문서는 한국어로 작성한다. 코드 주석은 기존 파일의 언어(TS는 영어, Python 독스트링은 영어)를 따른다.
- `docs/` 안 문서의 이미지는 `docs/figures/`에 있다.
- 문서 상단 YAML frontmatter의 `[[...]]` 링크는 Obsidian 위키링크로, 파일명 기반이라 경로 이동과 무관하다.

---

## 5. ResearchTree (브랜치 = 실험, PR = 실험 노트)

이 저장소는 **ResearchTree**로 실험 이력을 관리한다. 하나의 실험은 하나의 브랜치이고, 그 브랜치의 PR 본문이 그 실험의 기록이다. 자세한 규약은 `.claude/skills/researchtree/SKILL.md`에 있다.

### 설정

| 항목 | 값 | 어디서 지정 |
|---|---|---|
| 루트 브랜치 | **`develop`** | 환경변수 `RESEARCHTREE_ROOT=develop` (파일로는 지정 불가) |
| 실험 브랜치 접두사 | `experiment/` (기본값) | — |
| 스펙 문서 | `docs/SPEC.md` | `.researchtree.yml` |
| 인텐트 문서 | `docs/INTENT.md` | `.researchtree.yml` |
| 버전 태그 | `develop/v1`, `develop/v2` … | `researchtree release` |

```bash
export RESEARCHTREE_ROOT=develop      # 셸 프로필이나 .env 에 넣어 두면 편하다
researchtree memory                   # 지금까지의 실험 트리
researchtree spec --intent            # 의도와 주장
researchtree spec --summary           # 현재 설계 한 장 요약
researchtree spec --claims            # 주장별로 어떤 실험이 붙었는지
```

Windows PowerShell 이면 `$env:RESEARCHTREE_ROOT = "develop"`.

### 작업 순서

1. 실험을 제안하기 전에 `researchtree memory` 로 **같은 가설이 이미 기각됐는지** 확인한다.
2. 브랜치는 워크트리로 딴다. `git worktree add -b experiment/<이름> ../skylens-<이름> develop/v1`
3. 설계를 바꾸는 실험이면 **`docs/SPEC.md` 를 그 브랜치에서 먼저 고치고** 구현한다. 값만 조정하는 실험은 스펙을 건드리지 않고 `spec: none` 이라 적는다.
4. 푸시하고 **draft PR** 을 연다. base 는 `develop`(버전에서 출발) 또는 부모 실험 브랜치. 본문 맨 위에 YAML 블록을 둔다.
5. 결론을 쓰고 지표를 채운 뒤, **채택이면 머지 / 기각이면 머지 없이 close** 한다.

### 지켜야 할 것

- **PR·결론·스펙·커밋 메시지는 사람이 읽는 글이다.** 오늘 합류한 팀원이 읽고 이해할 수 있게 쓴다. 결론을 먼저, 숫자에는 단위와 비교 대상을. 축약된 은어(`R1-e`, `plan C`)를 그냥 쓰지 않는다.
- 브랜치 하나에 가설 하나, PR 하나, YAML 블록 하나.
- `main`·`develop` 외 브랜치를 base 로 하는 실험 PR 은 트리에 안 나온다. 실험 PR 의 base 는 `develop` 또는 부모 실험 브랜치다.
- **머지·close·태그·`research`(=`develop`) 푸시는 요청받았을 때만 한다.** 결과만 보고하고 채택 여부는 사람이 정한다.
- 실험 브랜치를 손으로 지우지 않는다. `researchtree release` 가 안전하게 처리한다.

### `main` 과 `develop`

`develop` 은 연구 루트 브랜치라 설계 문서가 함께 산다. `main` 은 배포용이라 그 문서를 들고 가지 않는다.

```bash
bash scripts/sync-main.sh --dry-run   # 무엇을 지울지만 확인
bash scripts/sync-main.sh             # develop → main 머지 + 설계 문서 제거 (push 안 함)
bash scripts/sync-main.sh --push      # 머지 후 origin/main 으로 push
```

지우는 대상은 `docs/` **바로 아래**의 `.md` 전부와 프로젝트 루트의 `.md` 중 `README.md` 를 뺀 나머지다. `docs/figures/` · `docs/materials/` 같은 하위 폴더는 건드리지 않는다.
