# CLAUDE.md: SkyLens 작업 가이드

SkyLens는 **멀티드론 영상을 실시간 3D(점군·메시)로 복원하고 그 위에 AI가 위험구역·사람을 표시하는 재난 인텔리전스 플랫폼**이다 (NET 챌린지 캠프 시즌13).
이 저장소는 **Rust**(드론·모델 서버·클라이언트·데모), **Go**(코어), **nginx 설정**(게이트웨이·프록시), **Python**(UNet 학습·데이터셋)으로 되어 있다. TypeScript 프로토타입에서 옮기는 중이며, 옮겨지기 전 코드는 각 컴포넌트의 `legacy/`에 참고용으로만 있다(빌드 대상 아님). 진행 상태는 `docs/COMPONENTS.md` §10.

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
> - 문서(`.md`) 본문은 계속 한국어로 쓴다. 이 규칙은 **커밋 메시지에만** 적용된다.

또한 **요청받지 않았으면 커밋하지 않는다.**

---

## 1. 문서 지도

| 문서 | 역할 |
|---|---|
| `README.md` (루트) | 저장소 소개. 데모/실서버 모드, GPS↔ENU 좌표계, 빠른 시작(cargo·go·uv 빌드), 프로젝트 구조 트리, 포트, 현재 구현 상태·로드맵. 쿼리 옵션·조작법 절은 아직 TS 프로토타입 기준 |
| `PROJECT.md` (루트) | **중간평가 프로토타입 구현 계획**. 무엇을 증명하고 무엇을 증명하지 않는지(§0), 데모 컨셉(§1), 사전 촬영·gsplat 준비(§2), 기술 스택(§3), **관제탑·GPS 경로 계획(§4)**, **현황판·서버 스트림 3D 축적(§5)**, 위험구역 오버레이(§6), 연출과 실제의 경계 원칙(§7), 관제탑↔현황판 동기화·카메라 협력(§8), 구현 로드맵(§9)·리스크(§10)·최종평가 확장(§11) |
| `docs/COMPONENTS.md` | **컴포넌트 구성의 단일 출처**. 컴포넌트(드론·게이트웨이·프록시·코어·모델·클라이언트·shared·test·데모)의 역할·망 위치·스택, 세 갈래 데이터 흐름, 렌더링과 UI(pointblitz·Blitz), 포트 맵, 빌드, 구현상의 결정과 열린 결정(§9), **스택 전환 이행 단계(§10)** |
| `docs/INTENT.md` | **의도(왜)**. ResearchTree 인텐트 문서. 문제 정의(소방드론 현황·홍제동 사례·각주 출처), 목표, **주장 N1~N7**(실험이 검증하는 단위), 제약, 하지 않는 것, 열린 결정, KOREN 활용 논거 |
| `docs/SPEC.md` | **현재 설계의 단일 출처**. ResearchTree 스펙 문서. 파이프라인·컴포넌트 경계·프로토콜·좌표계·점진적 복원·두 화면·AI 모델·투영과 프레임 간 보간·3D 복원(cumulus3d·deltamesh)·데모·KOREN 배포·포트·빌드·브랜치까지 16절. 각 절은 한 줄 요약(`>`)으로 시작한다. **지금 상태만 적고 역사는 적지 않는다** |
| `docs/ARCHITECTURE.md` | **통합 아키텍처(무엇을)**. 3대 설계 원칙, 4-Tier 구성(캡처/전송/Core HPC/Edge·클라이언트), 데이터 플로우, **§3-A AI 모델·융합 파이프라인**(UNet 4채널, Depth Map 레이캐스팅, Hybrid Fusion, 기술 선택 배제 근거) |
| `docs/DATASETS.md` | **학습 데이터셋 조사**. "RGB+열 페어 + 재난 + 사람 + 위험구역"을 모두 가진 공개 데이터는 없다는 결론과, A(4채널 정합)/B(위험구역 세그)/C(사람 인스턴스) 3축 조합 권장. FLAME 3, RescueNet, SARD, AIResQ, LLVIP, VisDrone 등 |
| `src/skylens_model/README.md` | **AI 모델 설계 철학의 단일 출처(어떻게)**. 레이어 분리 원칙(탐지/투영/랜드마크 융합/소리 보정), UNet 채택·TransUNet 보류 근거, 단일 백본+이중 헤드, modality dropout, 점 검출 헤드, 헤드별 분리 학습, 배제한 대안 표. **결정과 그 근거**를 기록 |
| `src/skylens_model/models/skylens/README.md` | **이전 3DGS 복원 설계 기록**(cumulus3d 기반 Rust 구현으로 대체 중, 새 설계는 `docs/SPEC.md` §10). 촬영 가이드는 계속 유효. 파이프라인 3단계(프레임 추출 → COLMAP SfM → gsplat 학습), 특징점을 ALIKED로 정한 근거, 전수매칭이 필요한 이유, 내부 파라미터 명시 + 자기보정 조합, 점진적 출력(30초 첫 화면), PLY 경량판. **촬영 가이드**(포즈 정확도 요구·편대 배치·드론 자동비행 가능 범위)와 증분 복원 가능 여부 결론 |
| `src/skylens_model/models/skylens/RESULTS.md` | 위 결정들의 **측정 근거**. 실험 5건 · 학습 조건 28개 수치 전체. 판정이 보류된 실험은 그 이유까지 명시 |
| `src/skylens_model/models/skylens/INSTALL.md` | COLMAP(CUDA 빌드)·gsplat 설치 절차, `recon` 의존성 그룹에서 뺀 항목과 이유, **밟았던 함정 10건** |
| `src/skylens_model/datasets/README.md` | 데이터셋 **API 계약**(`__getitem__` 반환 dict, `None`은 정상값), 통합 클래스 스키마(0 normal / 1 fire / 2 collapse / 3 road_blocked / 255 ignore), RescueNet·VisDrone 매핑, 자동 다운로드 가능 여부 판정 |
| `src/skylens_model/utils/README.md` | `geo.py`가 좌표 변환의 파이썬 구현이라는 사실(짝은 `src/shared`, 예전엔 `geo.ts`). **두 파일은 수치적으로 동기 유지** |
| `docs/site/` | **GitHub Pages 가이드북**. 프로젝트를 처음 보는 사람에게 설명하는 정적 사이트. 6페이지(홈·동작 방식·두 화면·시스템 구조·실행·AI 모델). 빌드 스텝 없이 `.github/workflows/pages.yml` 이 `main` push 때 그대로 올린다. 말투는 토스 톤, 디자인은 관제탑 UI 토큰을 따른다 |
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
| 컴포넌트 경계·서버·프로토콜 관련 작업 | **`docs/COMPONENTS.md` 필독** → `src/shared/proto/skylens/model/v1/model.proto`(코어↔모델) · `src/shared` (예전 정의는 `src/shared/legacy/protocol.ts`) |
| 데모·뷰어(관제탑/현황판) UI·연출 작업 | **`docs/COMPONENTS.md` §3.10(화면 이식 원칙)** → 이전 화면 DOM·캡처(`.worktrees/legacy-dom/`, 커밋 `17c4d92`) |
| 좌표계·GPS 관련 | `README.md` 좌표계 절 → `src/shared` + `src/skylens_model/utils/geo.py` (둘 다 고칠 것) |
| 3D 복원·촬영 계획 (`src/skylens_model/models/skylens`) | **`docs/SPEC.md` §9~10** → `docs/COMPONENTS.md` §3.5 → `models/skylens/README.md`(촬영 가이드·이전 3DGS 근거) |
| 기획 문구·발표 자료 | `docs/INTENT.md` → `docs/ARCHITECTURE.md` |
| 실험 시작·기록·결론 | **§5 ResearchTree** → `researchtree spec --intent` · `--summary` · `--claims` |

---

## 2. 저장소 구조

```
Cargo.toml           # Rust 워크스페이스. 공용 의존성 버전은 [workspace.dependencies] 한 곳
go.work              # Go 워크스페이스 (src/skylens_core)
pyproject.toml       # Python: UNet 학습 · 데이터셋
res/static/          # 데모 자산 (영상 · 이미지 · demo/ 구간 자산은 생성물, 커밋하지 않음)
src/
├─ shared/           # 공통 계약 crate(proto · 좌표계) + viewer/(두 화면 공용 층 crate)
├─ skylens_drone/    # Rust Mission Bridge (예전 skylens_mission_bridge 흡수)
├─ skylens_gateway/  # 브리지 호스트: nginx 설정 · MediaMTX · 설치 스크립트 (코드 없음)
├─ skylens_proxy/    # 대전: nginx 다중 경로 설정 (코드 없음)
├─ skylens_core/     # Go 코어 서버 + control/(관제탑 화면 crate, 코어가 웹으로 제공) + deploy/(대전 MediaMTX · KOREN 배치 문서)
├─ skylens_model/    # Rust 연산 서버 + Python 학습 코드. 같은 폴더에 .py 와 짝이 되는 .rs
├─ skylens_client/   # 현황판 화면 crate: 이전 HTML/CSS를 Blitz로 그대로 + pointblitz (웹 · 네이티브 같은 코드)
├─ test/             # 컴포넌트 경계 테스트(Rust crate) + pytest
└─ demo/             # Rust 엔드 투 엔드 데모 런처
docs/            # INTENT · SPEC · COMPONENTS · ARCHITECTURE · DATASETS · NETWORK_ARCHITECTURE
  └─ site/       # GitHub Pages 가이드북 (정적 HTML/CSS/JS, 빌드 없음)
scripts/         # sync-main.sh (develop → main 동기화) · 학습 평가 스크립트
```

컴포넌트 경계와 각 컴포넌트의 책임은 **`docs/COMPONENTS.md`가 단일 출처**다. 각 컴포넌트의 `legacy/`는 포팅 참고용이고, 포팅이 끝나면 지운다. `legacy/` 코드를 고치지 않는다.

`pyproject.toml`은 자동 탐색 대신 `packages = ["src/skylens_model"]`로 파이썬 패키지를 명시한다.

---

## 3. 빌드 · 테스트

**테스트를 먼저 쓴다(TDD).** 단위 테스트는 각 언어 관례대로 코드 옆(Rust `#[cfg(test)]`, Go `_test.go`), 컴포넌트를 가로지르는 테스트는 `src/test/`.

**Rust** (루트 `Cargo.toml` 워크스페이스)

| 명령 | 설명 |
|---|---|
| `cargo build --workspace` | 전체 빌드 |
| `cargo test --workspace` | 테스트 |
| `cargo clippy --workspace --all-targets` | 린트 |
| `cargo run -p skylens-demo` | 데모 런처 |

- Rust 1.99 이상 stable(pointblitz 0.3 요구, `rust-version`이 강제). 웹 빌드는 `wasm32-unknown-unknown` 타깃. 버전을 올려도 된다.
- UNet 추론은 `tch` 0.24(libtorch 2.11). libtorch는 빌드 때 내려받는다(`download-libtorch`, CUDA 판은 `.cargo/config.toml`의 `TORCH_CUDA_VERSION`). Python venv와 무관하다.
- 프로토콜 계약은 `src/shared/proto/`(model · viewer · payload). `skylens-shared`가 protox로 컴파일하므로 protoc가 없어도 된다.
- cumulus3d는 crates.io 릴리스(0.9)를 쓴다(벤더 복사본 없음). 라이브 점은 stable display의 메모리 표시 델타(`Event::DisplayDelta`)를 점 슬롯 청크로 내보낸다.

**Go** (`src/skylens_core`, 루트 `go.work`)

| 명령 | 설명 |
|---|---|
| `go build ./...` | 코어 빌드 |
| `go test ./...` | 테스트 |

**nginx** (`skylens_gateway`, `skylens_proxy`): 설정 파일만 관리한다. 개발 머신에 nginx가 없어도 되고, 검사는 배치 호스트에서 `nginx -t`로 한다.

**Python** (`pyproject.toml`, requires-python >=3.11,<3.14 · `.python-version` = 3.13). UNet 학습 코드와 데이터셋에만 쓴다.

- 의존성은 **uv** 로 관리한다. `uv sync` 하면 `.venv/` 가 만들어지고 패키지가 editable 로 설치된다.
- 실행은 `uv run <cmd>` (예: `uv run pytest`, `uv run jupyter lab`). `.venv` 를 직접 activate 해도 된다.
- `uv.lock` 은 **커밋한다**(재현성). 의존성을 바꾸면 `uv lock` 후 락 파일도 함께 커밋.
- 그룹: `train` / `notebook` / `dev`(train+notebook 포함). 기본은 `dev` 가 설치된다.
- torch·torchvision 은 PyPI 가 아니라 **CUDA 12.8 인덱스**(`download.pytorch.org/whl/cu128`)에서 받는다. `[tool.uv.sources]` 참조.
- 린트: `uv run ruff check src/` (line-length 100, target py311)
- 테스트: `uv run pytest` (`testpaths = src/test`)

---

## 4. 기타 규약

- **학습 산출물·데이터는 커밋하지 않는다.** `.gitignore`가 `data/`, `runs/`, `outputs/`, `checkpoints/`, `wandb/`, `*.ckpt`, `*.pt`, `*.pth`, `*.safetensors`를 제외한다. 데이터셋은 저장소에 포함하지 않고 각 `Dataset` 클래스가 `root` 아래에서 찾는다.
- `.claude/`, `.omc/`도 git 제외 대상이다.
- `src/shared`(Rust 좌표계) ↔ `src/skylens_model/utils/geo.py`는 **같은 수식의 두 구현**이다. 한쪽만 고치지 말 것.
- 문서는 한국어로 작성한다. 코드 주석은 영어(Rust·Go 주석, Python 독스트링)로 쓴다.
- **슬로건·캐치프레이즈를 지어내지 않는다.** 프로젝트를 소개하는 문구는 저장소에 이미 있는 말(`README.md` 한 줄 소개, `docs/INTENT.md`의 과제명)을 쓴다. 그럴듯한 카피를 새로 만들어 넣지 말 것. 문서·가이드북·발표자료 전부 해당된다.
- **문서에 음대시(`—`)를 쓰지 않는다.** 마침표로 문장을 끊거나, 쉼표·콜론·괄호를 쓴다. 표에서 "해당 없음"을 나타낼 때도 `—` 대신 글자로 적는다. 가이드북(`docs/site/`)의 본문도 같다.
- `docs/` 안 문서의 이미지는 `docs/figures/`에 있다.
- 문서 상단 YAML frontmatter의 `[[...]]` 링크는 Obsidian 위키링크로, 파일명 기반이라 경로 이동과 무관하다.

---

## 5. ResearchTree (브랜치 = 실험, PR = 실험 노트)

이 저장소는 **ResearchTree**로 실험 이력을 관리한다. 하나의 실험은 하나의 브랜치이고, 그 브랜치의 PR 본문이 그 실험의 기록이다. 자세한 규약은 `.claude/skills/researchtree/SKILL.md`에 있다.

### 설정

설정은 전부 저장소 루트의 **`.researchtree`** 파일에 있다(확장자 없음). 뷰어·CLI·Python API가 같은 파일을 읽으므로 머신마다 따로 맞출 것이 없다. 환경변수는 쓰지 않는다.

| 항목 | 값 | 어디서 지정 |
|---|---|---|
| 루트 브랜치 | **`develop`** | `.researchtree` 의 `root:` |
| 실험 브랜치 접두사 | `experiment/` (기본값) | 지정하지 않음 |
| 스펙 문서 | `docs/SPEC.md` | `.researchtree` 의 `spec:` |
| 인텐트 문서 | `docs/INTENT.md` | `.researchtree` 의 `intent:` |
| 버전 태그 | `develop/v1`, `develop/v2` … | `researchtree release` |

> `.researchtree` 는 **기본 브랜치(`main`)에 있어야** 뷰어가 읽는다. `scripts/sync-main.sh` 가 `main` 으로 같이 넘긴다.
>
> 뷰어 주소는 계정을 먼저 적는다: `https://darkpyonix.github.io/researchtree/?user=NET-Challenge-S13&repo=skylens`. `researchtree open` 이 그 주소를 출력한다.

```bash
researchtree memory                   # 지금까지의 실험 트리
researchtree spec --intent            # 의도와 주장
researchtree spec --summary           # 현재 설계 한 장 요약
researchtree spec --claims            # 주장별로 어떤 실험이 붙었는지
```

### 작업 순서

> ⚠️ **순서를 틀리지 마라. 구현보다 draft PR 이 먼저다.**
>
> 실험은 트리에 보이는 순간부터 실험이다. 구현을 다 끝내고 마지막에 푸시하면, 그동안 트리에는 실험이 0개로 보이고 사람이 진행 상황을 볼 수 없다. 실제로 한 번 이렇게 틀렸다.

1. **기존 기록을 읽는다.** `researchtree memory` 로 같은 가설이 이미 기각됐는지, `researchtree spec --claims` 로 어떤 주장에 붙일지 확인한다.
2. **워크트리로 브랜치를 딴다.** `git worktree add -b experiment/<이름> .worktrees/<이름> develop/v1`
   저장소 밖에 디렉터리를 만들지 않는다. `.worktrees/` 는 `.gitignore` 대상이다.
3. **설계를 바꾸는 실험이면 `docs/SPEC.md` 를 먼저 고친다.** 값만 조정하는 실험은 스펙을 건드리지 않고 `spec: none` 이라 적는다.
4. **곧바로 푸시하고 draft PR 을 연다.** 여기까지가 구현 전에 끝나야 한다. base 는 `develop`(버전에서 출발) 또는 부모 실험 브랜치. 본문 맨 위에 YAML 블록(`hypothesis` · `parent` · `change` · `claims`)을 둔다. 가설과 판정 기준을 이 시점에 써 둬야 결과를 보고 기준을 바꾸는 일이 없다.
   ```bash
   git push -u origin experiment/<이름>
   gh pr create --draft --base develop --title "<가설 한 줄>" --body-file <본문>
   researchtree memory            # running 으로 보이는지 확인한다
   ```
5. **그다음 구현한다.** 커밋마다 푸시해서 진행 상황이 PR에 계속 반영되게 한다. 마지막에 몰아서 푸시하지 않는다.
6. 학습하고 측정한다. 지표는 `rt.log()` 로 PR에 적는다.
7. 결론을 쓰고 지표를 채운 뒤, **채택이면 머지 / 기각이면 머지 없이 close** 한다. 이 판단은 사람이 한다.

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
bash scripts/sync-main.sh             # 임시 release 브랜치까지만 만들고 멈춘다
bash scripts/sync-main.sh --pr        # push 하고 main 으로 가는 PR 을 연다
```

지우는 대상은 `docs/` **바로 아래**의 `.md` 전부와 프로젝트 루트의 `.md` 중 `README.md` 를 뺀 나머지다. `docs/figures/` · `docs/materials/` · `docs/site/` 같은 하위 폴더는 건드리지 않는다. 지운 문서를 가리키던 상대 링크는 `develop` 절대 URL 로 바뀐다.

`main` 에는 PR 을 거쳐야 하는 보호 규칙이 걸려 있다. 그래서 이 스크립트는 `main` 을 직접 건드리지 않고 임시 `release` 브랜치에 결과를 만들어 PR 을 연다. 머지는 사람이 한다.
