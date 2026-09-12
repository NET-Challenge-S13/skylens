#!/usr/bin/env bash
#
# sync-main.sh: develop 을 main 에 머지하면서 설계 문서를 걷어낸다.
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

TMP_REMOVED="$(mktemp)"

# --- 사전 점검 -------------------------------------------------------------

# 추적 중인 파일에 변경이 남아 있으면 멈춘다. 머지가 그 변경을 덮어쓸 수 있기 때문이다.
# untracked 파일은 머지에 영향이 없으므로 막지 않는다: 다만 대상 브랜치에 같은 이름의
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
cleanup() {
  rm -f "$TMP_REMOVED"
  [ "${SWITCHED:-0}" = 1 ] && git switch "$START_BRANCH" >/dev/null 2>&1 || true
}
trap cleanup EXIT

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

SWITCHED=1

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

# --- 남은 문서의 링크 고치기 -------------------------------------------------
#
# README 와 각 컴포넌트 README 는 방금 지운 문서들을 상대경로로 가리킨다.
# 그대로 두면 main 의 첫 화면에서 깨진 링크가 된다. 상대경로를 develop 의
# 절대 URL 로 바꿔서, 문서는 main 에 없더라도 읽고 싶은 사람은 따라갈 수 있게 한다.

if [ "$REMOVED" -gt 0 ]; then
  printf '%s\n' "${TO_REMOVE[@]}" > "$TMP_REMOVED"

  SLUG="$(git config --get remote.origin.url \
    | sed -E 's#^.*github\.com[:/]##; s#\.git$##')"

  if [ -z "$SLUG" ]; then
    echo "warn: cannot read owner/name from origin; leaving links alone" >&2
  elif ! command -v node >/dev/null 2>&1; then
    echo "warn: node not found; leaving links alone" >&2
  else
    node - "$SLUG" "$SOURCE_BRANCH" "$TMP_REMOVED" <<'NODE'
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const [slug, branch, removedFile] = process.argv.slice(2);
const base = `https://github.com/${slug}/blob/${branch}/`;
const removed = new Set(
  fs.readFileSync(removedFile, 'utf8').split('\n').filter(Boolean),
);

const tracked = execFileSync('git', ['ls-files', '-z', '*.md'], { encoding: 'utf8' })
  .split('\0')
  .filter(Boolean);

let files = 0;
let links = 0;

for (const file of tracked) {
  const before = fs.readFileSync(file, 'utf8');
  // ](target) where target is a relative path, optionally with #anchor.
  const after = before.replace(/\]\(([^)\s]+?)(#[^)\s]*)?\)/g, (whole, target, anchor = '') => {
    if (/^([a-z][a-z0-9+.-]*:|\/|#)/i.test(target)) return whole; // absolute, root or bare anchor
    const resolved = path.posix.normalize(
      path.posix.join(path.posix.dirname(file.split(path.sep).join('/')), target),
    );
    if (!removed.has(resolved)) return whole;
    links += 1;
    return `](${base}${resolved}${anchor})`;
  });
  if (after !== before) {
    fs.writeFileSync(file, after);
    files += 1;
  }
}

console.log(`==> rewrote ${links} link(s) in ${files} file(s) to point at ${branch}`);
NODE
    git add -A -- '*.md'
  fi
fi

if [ "$REMOVED" -gt 0 ]; then
  git commit -q -m "chore: drop design docs from $TARGET_BRANCH

The design documents live on $SOURCE_BRANCH. Links that pointed at them
are rewritten to absolute $SOURCE_BRANCH URLs so they still resolve."
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
