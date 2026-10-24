# cumulus3d 자동 업데이트

> cumulus3d가 crates.io에 새 릴리스를 내면 `cumulus3d-cli`·`cumulus3d-core`의 버전 요구를 올리고, 락을 갱신하고, 빌드·테스트를 확인한 뒤 `develop`으로 가는 PR `deps/cumulus3d-v<버전>`을 연다. SkyLens는 cumulus3d를 벤더링하지 않고 crates.io 릴리스를 그대로 쓴다.

관련 파일은 `.github/workflows/cumulus3d-sync.yml` 하나다.

## 1. 트리거

| 트리거 | 설명 |
|---|---|
| `repository_dispatch` `cumulus3d-released` | upstream이 릴리스 때 보낸다. `client_payload`는 `version`, `previous_version`, `breaking`, `changelog`, `release_url` |
| `workflow_dispatch` | 손으로 실행. 입력 `version`(비우면 crates.io 최신 안정 버전) |
| 매일 06:17 KST | 디스패치를 놓쳤을 때를 위한 보험. `Cargo.lock`의 `cumulus3d-cli`보다 최신이 있을 때만 진행한다. 같은 버전의 `deps/cumulus3d-v<버전>` 브랜치나 PR(닫힌 것 포함)이 있으면 다시 열지 않는다 |

> `repository_dispatch`와 `schedule`은 **기본 브랜치(`main`)에 있는 워크플로만** 실행된다. `develop`이 `main`으로 릴리스(`scripts/sync-main.sh`)된 뒤부터 동작한다.

## 2. 흐름

1. **대상 결정.** 페이로드·입력의 버전, 없으면 crates.io 최신. `Cargo.lock`의 현재 버전과 같으면 종료.
2. **요구 올리기.** 루트 `Cargo.toml`과 `src/*/Cargo.toml`에서 `cumulus3d-cli = "x.y"`, `cumulus3d-core = "x.y"`로 선언한 줄을 새 버전의 `major.minor`로 바꾸고 `cargo update -p cumulus3d-cli -p cumulus3d-core`로 락을 옮긴다. 락이 새 버전으로 가지 않으면 실패한다.
3. **검사.** CPU libtorch(2.11.0)를 `LIBTORCH`로 가리키고 `cargo check -p skylens-model --all-targets`, `cargo test -p skylens-model --lib`. `.cargo/config.toml`의 `TORCH_CUDA_VERSION=cu126`은 Linux `torch-sys`가 `cpu`로 받지 못해 CPU 판을 직접 내려받는다. `tch`를 올리면 `LIBTORCH_VERSION`도 같이 올린다.
4. **보정(선택).** 실패하고 시크릿이 있으면 `anthropics/claude-code-action@v1`이 API 변경에 맞춰 `src/`를 고치고 커밋한다. 끝나면 검사를 다시 돌린다. 실패해도 PR은 열린다.
5. **PR.** `deps/cumulus3d-v<버전>` 브랜치를 푸시하고 `develop`으로 PR을 연다(이미 열려 있으면 본문 갱신). PR 본문에 changelog 발췌(페이로드, 없으면 크레이트에 든 `CHANGELOG.md`)와 V100 검증 체크리스트가 든다. 검사가 실패로 남으면 draft.

## 3. 시크릿과 저장소 설정

| 설정 | 용도 |
|---|---|
| `ANTHROPIC_API_KEY` 또는 `CLAUDE_CODE_OAUTH_TOKEN` (둘 중 하나) | 보정 단계. 둘 다 없으면 건너뛴다 |
| Settings > Actions > General > "Allow GitHub Actions to create and approve pull requests" | `GITHUB_TOKEN`으로 PR을 열기 위해 켠다 |
| upstream 쪽 dispatch 토큰 | upstream이 이 저장소로 `repository_dispatch`를 보내려면 이 저장소의 `contents: write` 권한이 있는 토큰이 필요하다 |

## 4. PR을 받은 뒤

CI는 GPU가 없어 CUDA 경로와 실제 복원을 못 돌린다. PR 본문 체크리스트대로 V100에서 라이브 복원을 한 번 돌려 점군 수·시간을 이전 버전과 비교한 뒤 머지한다.
