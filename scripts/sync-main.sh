#!/usr/bin/env bash
#
# sync-main.sh — develop 을 main 에 머지하면서 설계 문서를 걷어낸다.
#
# main 은 배포용 브랜치다. develop 에 쌓이는 설계·기획 문서(INTENT / SPEC / ARCHITECTURE …)는
# 연구 기록이지 배포물이 아니므로 main 에서는 지운다. 남기는 것은 README.md 하나다.
#
# 지우는 것
#   - docs/*.md            (docs 바로 아래 .md 만. figures/ · materials/ 등 하위 폴더는 그대로 둔다)
#   - 프로젝트 루트의 *.md  (README.md 는 예외)
#
# 사용법
#   bash scripts/sync-main.sh            # 머지하고 결과를 보여준다 (push 는 안 한다)
#   bash scripts/sync-main.sh --push     # 머지 후 origin/main 으로 push
#   bash scripts/sync-main.sh --dry-run  # 무엇을 지울지만 보여주고 아무것도 바꾸지 않는다
#
set -euo pipefail

SOURCE_BRANCH="${SOURCE_BRANCH:-develop}"
TARGET_BRANCH="${TARGET_BRANCH:-main}"
KEEP_ROOT_DOC="README.md"

PUSH=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --push)    PUSH=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

cd "$(git rev-parse --show-toplevel)"

# --- 사전 점검 -------------------------------------------------------------

# 추적 중인 파일에 변경이 남아 있으면 멈춘다. 머지가 그 변경을 덮어쓸 수 있기 때문이다.
# untracked 파일은 머지에 영향이 없으므로 막지 않는다 — 다만 대상 브랜치에 같은 이름의
# 파일이 있으면 git 이 브랜치 전환 단계에서 알아서 거부한다.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "error: tracked files have uncommitted changes. commit or stash first." >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

UNTRACKED="$(git ls-files --others --exclude-standard)"
if [ -n "$UNTRACKED" ]; then
  echo "note: untracked files present (left alone):"
  printf '    %s\n' $UNTRACKED
fi

START_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
restore_branch() { git switch "$START_BRANCH" >/dev/null 2>&1 || true; }

echo "==> fetching origin"
git fetch origin --prune

for b in "$SOURCE_BRANCH" "$TARGET_BRANCH"; do
  if ! git show-ref --verify --quiet "refs/heads/$b"; then
    echo "==> creating local $b from origin/$b"
    git branch "$b" "origin/$b"
  fi
done

# --- 무엇을 지울지 계산 -----------------------------------------------------

# docs 바로 아래의 .md 만. -- 뒤의 pathspec 에서 하위 디렉터리는 제외된다.
mapfile -t DOC_MDS < <(git ls-tree -r --name-only "$SOURCE_BRANCH" -- docs/ \
  | grep -E '^docs/[^/]+\.md$' || true)

mapfile -t ROOT_MDS < <(git ls-tree --name-only "$SOURCE_BRANCH" \
  | grep -E '^[^/]+\.md$' | grep -v -x "$KEEP_ROOT_DOC" || true)

TO_REMOVE=("${DOC_MDS[@]}" "${ROOT_MDS[@]}")

echo "==> docs that will be removed on $TARGET_BRANCH:"
if [ "${#TO_REMOVE[@]}" -eq 0 ]; then
  echo "    (none)"
else
  printf '    %s\n' "${TO_REMOVE[@]}"
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "==> dry run, nothing changed"
  exit 0
fi

# --- 머지 ------------------------------------------------------------------

trap restore_branch EXIT

echo "==> switching to $TARGET_BRANCH"
git switch "$TARGET_BRANCH"
git merge --ff-only "origin/$TARGET_BRANCH" 2>/dev/null || true

echo "==> merging $SOURCE_BRANCH into $TARGET_BRANCH"
# -X theirs: main 에서 문서를 지운 이력과 develop 의 같은 파일이 충돌하므로,
# 내용 충돌은 항상 develop 쪽을 채택한다. 지우는 일은 머지 후에 다시 한다.
if ! git merge --no-ff -X theirs "$SOURCE_BRANCH" -m "merge $SOURCE_BRANCH into $TARGET_BRANCH"; then
  echo "error: merge conflict. resolve it, then re-run this script." >&2
  exit 1
fi

# --- 문서 제거 --------------------------------------------------------------

REMOVED=0
for f in "${TO_REMOVE[@]}"; do
  if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    git rm -q "$f"
    REMOVED=$((REMOVED + 1))
  fi
done

if [ "$REMOVED" -gt 0 ]; then
  git commit -q -m "chore: drop design docs from $TARGET_BRANCH"
  echo "==> removed $REMOVED design doc(s)"
else
  echo "==> no design docs to remove"
fi

# --- 결과 ------------------------------------------------------------------

echo "==> $TARGET_BRANCH is now:"
git --no-pager log --oneline -3

if [ "$PUSH" -eq 1 ]; then
  echo "==> pushing to origin/$TARGET_BRANCH"
  git push origin "$TARGET_BRANCH"
else
  echo "==> not pushed. run: git push origin $TARGET_BRANCH"
fi
