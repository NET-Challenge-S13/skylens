# cumulus3d 자동 동기화

> upstream `AH100-1/cumulus3d`가 새 릴리스를 내면 `third_party/cumulus3d`를 그 릴리스로 다시 가져오고, SkyLens 패치를 파일 단위 3-way 병합으로 옮기고, 빌드를 확인한 뒤 `develop`으로 가는 PR을 연다. 충돌이나 빌드 실패는 Claude Code가 고쳐 보고, 못 고치면 draft PR로 사람에게 넘긴다.

관련 파일은 네 개다.

| 파일 | 역할 |
|---|---|
| `.github/workflows/cumulus3d-sync.yml` | 트리거, 러너 준비, 보정 단계, PR 생성 |
| `scripts/cumulus3d-sync.sh` | 실제 작업. 로컬에서도 같은 명령으로 돈다. 커밋도 푸시도 하지 않는다 |
| `third_party/cumulus3d/UPSTREAM` | 지금 사본이 기반한 upstream 리비전과 버전(`KEY=VALUE`). 스크립트가 읽고 갱신한다 |
| `third_party/cumulus3d/PATCHES.md` | SkyLens 패치 목록과 근거. 병합 결과를 판단하는 기준 |

---

## 1. 트리거

| 트리거 | 설명 |
|---|---|
| `repository_dispatch` `cumulus3d-released` | upstream이 릴리스 때 보낸다. `client_payload`는 `version`, `tag`, `previous_version`, `breaking`, `changelog`, `release_url` |
| `workflow_dispatch` | 손으로 실행. 입력 `version`(비우면 crates.io 최신), 선택 `tag` |
| 매일 06:17 KST | 디스패치를 놓쳤을 때를 위한 보험. crates.io의 `cumulus3d-cli` 최신 안정 버전이 `UPSTREAM`의 버전보다 높으면 진행하고, 같으면 아무것도 하지 않는다. 같은 버전의 `sync/cumulus3d-v<버전>` 브랜치나 PR(닫힌 것 포함)이 있으면 다시 열지 않는다 |

> `repository_dispatch`와 `schedule`은 **기본 브랜치(`main`)에 있는 워크플로만** 실행된다. 이 워크플로는 `develop`에 들어간 뒤 `develop`이 `main`으로 릴리스(`scripts/sync-main.sh`)되어야 동작한다. 그 전에는 `scripts/cumulus3d-sync.sh`를 로컬에서 돌려 볼 수 있다(§4).

---

## 2. 흐름

1. **대상 결정** (`resolve`). 페이로드나 입력의 버전, 없으면 crates.io 최신. 태그 `v<버전>`의 커밋을 `git ls-remote`로 찾는다. 이미 그 리비전이면 종료.
2. **재벤더링** (`apply`). 임시 디렉터리에 upstream의 **기반 리비전**(`UPSTREAM`의 `rev`)과 **새 태그**를 받아 `git archive`로 푼다. `third_party/cumulus3d`(SkyLens 사본)와 함께 세 벌로 파일마다 비교한다.
   - SkyLens가 안 건드린 파일은 새 upstream 것을 쓴다.
   - upstream이 안 바꾼 파일은 SkyLens 것을 그대로 둔다.
   - 둘 다 바꾼 파일은 `git merge-file --diff3`로 병합한다. 겹치면 충돌 마커(`skylens (vendored)` / `base` / `upstream <태그>`)를 남긴다.
   - upstream에 이미 같은 내용이 들어갔으면 `SAME_AS_UPSTREAM`으로 표시한다(패치 삭제 후보).
   - 한쪽이 지우고 다른 쪽이 고친 파일, 바이너리는 충돌로 센다.
   - `Cargo.lock`은 벤더링 대상이 아니다(루트 워크스페이스가 잠근다).
3. **핀 갱신.** 루트 `Cargo.toml`과 `src/skylens_model/Cargo.toml`의 git `rev`, `PATCHES.md` 첫 문단의 리비전, `UPSTREAM`을 새 값으로 바꾸고 `cargo update -p cumulus3d-cli`로 락을 맞춘다.
4. **검사** (충돌이 없을 때만). 아래 표.
5. **보정** (충돌이나 실패가 있고 시크릿이 있을 때). `anthropics/claude-code-action@v1`이 프롬프트를 받아 고치고 커밋한다. 끝나면 검사를 다시 돌린다.
6. **PR.** `sync/cumulus3d-v<버전>` 브랜치를 푸시하고 `develop`으로 PR을 연다(이미 열려 있으면 본문 갱신). 충돌이나 실패가 남았으면 draft.

의존성 방식은 **git `rev` 고정 + 벤더링 사본 `[patch]`**를 그대로 유지한다. crates.io 버전과 `[patch.crates-io]`로 바꾸면 핀이 더 단순해지지만, `CompositionHost`처럼 릴리스에 없던 API를 쓰던 이력이 있고 패치 사본이 어차피 경로 의존이어서, 이번 자동화는 구조를 바꾸지 않고 `rev`만 올리는 쪽을 택했다. 새 릴리스가 crates.io에 있으면 나중에 한 번에 바꿀 수 있다.

### 검사 항목

| 검사 | 명령 |
|---|---|
| 벤더링 사본 전체 빌드 | `third_party/cumulus3d`에서 `cargo check --workspace --all-targets` |
| 벤더링 사본 단위 테스트 | 같은 곳에서 `cargo test --workspace --exclude cumulus3d-cuda --lib` |
| 모델 서버 빌드 | 루트에서 `cargo check -p skylens-model --all-targets` |
| 모델 서버 단위 테스트 | 루트에서 `cargo test -p skylens-model --lib` |

cumulus3d를 쓰는 SkyLens 크레이트는 `skylens-model` 하나라 나머지 멤버(client, control, drone, demo)는 검사하지 않는다. `cumulus3d-cuda`는 NVRTC를 런타임에 불러오므로 CUDA 툴킷 없이 컴파일은 되지만, 테스트는 GPU가 필요해 뺀다.

러너에는 libtorch 때문에 따로 손을 쓴다. `.cargo/config.toml`이 `TORCH_CUDA_VERSION=cu126`을 정하고, `torch-sys`는 Linux에서 이 변수를 `cu` + 값으로 읽어 `cpu`라는 값을 받지 못한다. 그래서 워크플로가 CPU libtorch(2.11.0)를 직접 내려받아 `LIBTORCH`로 가리킨다(`LIBTORCH`가 다운로드보다 우선한다). `tch`의 버전을 올리면 `LIBTORCH_VERSION`도 같이 올린다.

---

## 3. 시크릿과 저장소 설정

| 설정 | 용도 |
|---|---|
| `ANTHROPIC_API_KEY` 또는 `CLAUDE_CODE_OAUTH_TOKEN` (Actions 시크릿, 둘 중 하나) | 보정 단계. 둘 다 없으면 보정 단계는 건너뛰고, 충돌 마커와 실패 요약이 든 draft PR만 연다 |
| Settings > Actions > General > "Allow GitHub Actions to create and approve pull requests" | `GITHUB_TOKEN`으로 PR을 열기 위해 켜 둔다 |
| (upstream 쪽) `SKYLENS_DISPATCH_TOKEN` 같은 토큰 | upstream 저장소가 이 저장소로 `repository_dispatch`를 보내려면 이 저장소의 `contents: write`(또는 `repo` 범위) 권한이 있는 토큰이 필요하다. upstream 저장소의 시크릿으로 둔다 |

워크플로 권한은 `contents: write`, `pull-requests: write`뿐이다. `GITHUB_TOKEN`으로 만든 PR은 다른 워크플로를 깨우지 않는다(지금 `develop`에는 PR 검사 워크플로가 없다).

보정 단계는 허용 도구를 `Read`, `Edit`, `Write`, `Glob`, `Grep`과 `cargo`, 읽기·커밋용 `git`, `grep`으로 제한한다. 푸시는 모델이 아니라 워크플로가 한다. 이 단계는 `continue-on-error`라서 실패해도 PR은 열린다.

---

## 4. 로컬에서 쓰기

```bash
# 무엇으로 올릴지만 확인 (아무것도 바꾸지 않는다)
scripts/cumulus3d-sync.sh resolve --version 0.4.0

# 재벤더링 + 핀 갱신 (커밋·푸시 없음). 충돌이 있으면 종료 코드 2
scripts/cumulus3d-sync.sh apply --version 0.4.0

# 재벤더링 후 검사까지 (실패하면 종료 코드 3)
scripts/cumulus3d-sync.sh run --version 0.4.0

# 검사만
scripts/cumulus3d-sync.sh check
```

리포트는 `./.cumulus3d-sync/`(`SYNC_OUT`으로 바꿈)에 쌓인다: `sync-report.md`(파일별 결과), `merge-status.tsv`, `conflicts.txt`, `skylens-patch.diff`(기반 리비전 대비 SkyLens 패치 전체), `check-report.md`, `checks/*.log`. 이 디렉터리는 커밋하지 않는다. 로컬에서 `skylens-model`을 검사하려면 `LIBTORCH`(와 맞는 `LIBTORCH_BYPASS_VERSION_CHECK=1`)를 직접 지정해야 한다. macOS에서는 `.cargo/config.toml`의 `cu126`이 맞지 않는다.

---

## 5. PR을 받은 뒤

PR 본문의 **V100 검증 체크리스트**는 CI가 못 하는 일이다. GPU 테스트, `CUMULUS3D_CUDA_COMPARE=1` 비교, 라이브 복원 한 번을 확인한 뒤 머지한다. 머지 전에 `PATCHES.md`에서 upstream에 들어간 항목이 있는지 보고 있으면 지운다. 병합은 줄 단위라서 충돌 없이 합쳐졌어도 의미가 어긋날 수 있다. `MERGED_CLEAN`으로 표시된 파일도 훑어본다.
