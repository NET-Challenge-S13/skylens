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
| `README.md` (루트) | 저장소 소개. 데모/실서버 모드, GPS↔ENU 좌표계, 빠른 시작(`npm run dev`), SIM/RECON 접속 주소와 쿼리 옵션, 조작법, 프로젝트 구조 트리, 현재 구현 상태·로드맵. **TypeScript 프로토타입(뷰어) 중심** |
| `PROJECT.md` (루트) | **중간평가 프로토타입 구현 계획**. 무엇을 증명하고 무엇을 증명하지 않는지(§0), 데모 컨셉(§1), 사전 촬영·gsplat 준비(§2), 기술 스택(§3), **SIM 제어탑·GPS 경로 계획(§4)**, **RECON 지휘관 상황판·서버 스트림 3D 축적(§5)**, 위험구역 오버레이(§6), 연출과 실제의 경계 원칙(§7), SIM↔RECON 동기화·카메라 협력(§8), 구현 로드맵(§9)·리스크(§10)·최종평가 확장(§11) |
| `res/docs/IDEA.md` | **기획서(왜)**. 과제명(안), 문제 정의(소방드론 현황·홍제동 사례·각주 출처), 해결 방법 3단계(분할탐색 → KOREN/Core HPC 3DGS+UNet → Edge VM 3D 상황판), YAMNet 소리 확장, KOREN 활용 논거 |
| `res/docs/ARCHITECTURE.md` | **통합 아키텍처(무엇을)**. 3대 설계 원칙, 4-Tier 구성(캡처/전송/Core HPC/Edge·클라이언트), 데이터 플로우, **§3-A AI 모델·융합 파이프라인**(UNet 4채널, Depth Map 레이캐스팅, Hybrid Fusion, 기술 선택 배제 근거) |
| `res/docs/DATASETS.md` | **학습 데이터셋 조사**. "RGB+열 페어 + 재난 + 사람 + 위험구역"을 모두 가진 공개 데이터는 없다는 결론과, A(4채널 정합)/B(위험구역 세그)/C(사람 인스턴스) 3축 조합 권장. FLAME 3, RescueNet, SARD, AIResQ, LLVIP, VisDrone 등 |
| `src/skylens_model/README.md` | **AI 모델 설계 철학의 단일 출처(어떻게)**. 레이어 분리 원칙(탐지/투영/랜드마크 융합/소리 보정), UNet 채택·TransUNet 보류 근거, 단일 백본+이중 헤드, modality dropout, 점 검출 헤드, 헤드별 분리 학습, 배제한 대안 표. **결정과 그 근거**를 기록 |
| `src/skylens_model/datasets/README.md` | 데이터셋 **API 계약**(`__getitem__` 반환 dict, `None`은 정상값), 통합 클래스 스키마(0 normal / 1 fire / 2 collapse / 3 road_blocked / 255 ignore), RescueNet·VisDrone 매핑, 자동 다운로드 가능 여부 판정 |
| `src/skylens_model/utils/README.md` | `geo.py`가 `src/skylens_core/geo.ts`의 순수 파이썬 미러라는 사실 — **두 파일은 수치적으로 동기 유지** |
| `train.ipynb` (루트) | 데이터셋 → 학습 → 추론 결과를 마커 좌표로 흘려보내는 학습 노트북 |

### 문서 간 관계

```
res/docs/IDEA.md            왜 이걸 하는가 (문제·해결 구상·KOREN 필연성)
      ↓
res/docs/ARCHITECTURE.md    무엇을 만드는가 (4-Tier, §3-A 모델·융합 파이프라인)
      ↓
PROJECT.md              중간평가에서 어디까지 만드는가 (범위 축소 + 연출 계획)
      ↓
README.md               실제로 만들어진 것을 어떻게 돌리는가

res/docs/DATASETS.md   ──→  src/skylens_model/README.md  ──→  datasets/README.md
(학습 데이터 근거)       (모델을 어떻게 설계할 것인가)      (그 설계의 코드 계약)
```

- `res/docs/DATASETS.md`는 `ARCHITECTURE.md §3-A`를 학습 가능한 형태로 뒷받침하고, 모델 README의 여러 결정(데이터 부족 → UNet, modality dropout, 헤드별 분리 학습)의 **직접 근거**다.
- ⚠️ `src/skylens_model/README.md`는 상위 문서보다 **최신**이다. `ARCHITECTURE.md`/`IDEA.md`에 남은 "UNet / TransUNet 병기", "인스턴스 헤드"(→ 점 검출 헤드) 같은 표현은 아직 정정 대기 상태다. **모델 문서가 우선한다.**

### 작업별 필독 문서

| 하려는 작업 | 먼저 읽을 것 |
|---|---|
| AI 모델 수정·추가 (`src/skylens_model/models`) | **`src/skylens_model/README.md` 필독** → `res/docs/ARCHITECTURE.md §3-A` |
| 데이터셋 클래스 추가·수정 | `src/skylens_model/datasets/README.md` → `res/docs/DATASETS.md` |
| 학습 노트북 / 학습 루프 | `src/skylens_model/README.md` §6(학습 전략) → `train.ipynb` |
| 데모·뷰어(SIM/RECON) UI·연출 작업 | **`PROJECT.md`** → `README.md` |
| 좌표계·GPS 관련 | `README.md` 좌표계 절 → `src/skylens_core/geo.ts` + `src/skylens_model/utils/geo.py` (둘 다 고칠 것) |
| 기획 문구·발표 자료 | `res/docs/IDEA.md` → `res/docs/ARCHITECTURE.md` |

---

## 2. 저장소 구조

```
res/static/          # 정적 html 셸(진입점): index / sim / recon .html → /src 모듈을 절대경로로 로드
src/
├─ skylens_core/     # TS 관제탑(SIM) + 공유 토대. 순수 core(DOM·Three 없음: config·types·store·geo·mode·protocol) + SIM·공유 브라우저 코드(sim.ts, net, server, data, drones, simview, sim/, 공유 ui, style.css)
├─ skylens_client/   # TS 지휘관(RECON) 앱: recon.ts, reconview, recon ui, data/detections — **core에 단방향 의존**
└─ skylens_model/    # Python AI 모델 패키지: models/, datasets/, utils/
tests/smoke.spec.ts  # Playwright E2E
res/docs/            # IDEA · ARCHITECTURE · DATASETS
```

두 스택은 **`src/` 아래에 공존**하고, 세 진입 html은 **`res/static/`** 에 모여 있다(루트는 설정 파일만). 접속 URL은 `/res/static/{sim,recon}.html` — `/`는 랜딩을 자동 서빙하지 않는다. `pyproject.toml`은 자동 탐색 대신 `packages = ["src/skylens_model"]`로 파이썬 패키지를 명시한다.

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

접속: `http://<IP>:5173/res/static/sim.html?room=demo` / `/res/static/recon.html?room=demo` (같은 `room`이면 WebRTC 연결, `&demo`로 자동 데모).

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
- `res/docs/` 안 문서의 이미지 링크는 `../figures/...` 로 `res/figures/` 디렉터리를 가리킨다 (해당 디렉터리는 아직 저장소에 없음 — 이미지 추가 시 `res/figures/` 생성).
- 문서 상단 YAML frontmatter의 `[[...]]` 링크는 Obsidian 위키링크로, 파일명 기반이라 경로 이동과 무관하다.
