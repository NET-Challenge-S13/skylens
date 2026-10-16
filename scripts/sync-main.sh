#!/usr/bin/env bash
#
# sync-main.sh: develop 을 main 에 올리는 PR 을 만든다.
#
# main 은 배포용 브랜치다. develop 에 쌓이는 설계·기획 문서(INTENT / SPEC / ARCHITECTURE 등)는
# 연구 기록이지 배포물이 아니므로 main 에서는 지운다. 남기는 것은 README.md 하나다.
#
# main 에는 "PR 을 거쳐야 한다"는 보호 규칙이 걸려 있다. 그래서 이 스크립트는 main 을
# 직접 건드리지 않고, 임시 release 브랜치에 결과를 만들어 push 한 뒤 PR 을 연다.
# PR 을 머지하는 것은 사람이 한다.
#
# 지우는 것
#   - docs/*.md            (docs 바로 아래 .md 만. figures/ · materials/ · site/ 는 그대로 둔다)
#   - 프로젝트 루트의 *.md  (README.md 는 예외)
#
# 지운 문서를 가리키던 상대 링크는 develop 의 절대 URL 로 바꾼다.
#
# 사용법
#   bash scripts/sync-main.sh            # 임시 브랜치까지만 만들고 멈춘다 (push 안 함)
#   bash scripts/sync-main.sh --pr       # push 하고 main 으로 가는 PR 을 연다
#   bash scripts/sync-main.sh --dry-run  # 무엇을 지울지만 보여주고 아무것도 바꾸지 않는다
#
# 환경변수: SOURCE_BRANCH(develop) · TARGET_BRANCH(main) · RELEASE_BRANCH(release)
#
set -euo pipefail

SOURCE_BRANCH="${SOURCE_BRANCH:-develop}"
TARGET_BRANCH="${TARGET_BRANCH:-main}"
RELEASE_BRANCH="${RELEASE_BRANCH:-release}"
KEEP_ROOT_DOC="README.md"

OPEN_PR=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --pr)      OPEN_PR=1 ;;
    --push)    OPEN_PR=1 ;;   # 예전 이름
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

cd "$(git rev-parse --show-toplevel)"

TMP_REMOVED="$(mktemp)"
START_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
SWITCHED=0
cleanup() {
  rm -f "$TMP_REMOVED"
  if [ "$SWITCHED" = 1 ]; then
    git merge --abort >/dev/null 2>&1 || true
    git switch "$START_BRANCH" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# --- 사전 점검 -------------------------------------------------------------

# 추적 중인 파일에 변경이 남아 있으면 멈춘다. 머지가 그 변경을 덮어쓸 수 있기 때문이다.
# untracked 파일은 머지에 영향이 없으므로 막지 않는다.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "error: tracked files have uncommitted changes. commit or stash first." >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

echo "==> fetching origin"
git fetch origin --prune

for b in "$SOURCE_BRANCH" "$TARGET_BRANCH"; do
  git show-ref --verify --quiet "refs/remotes/origin/$b" || {
    echo "error: origin/$b not found" >&2; exit 1; }
done

# --- 무엇을 지울지 계산 -----------------------------------------------------

# docs 바로 아래의 .md 만. 하위 디렉터리(figures · materials · site)는 건드리지 않는다.
mapfile -t DOC_MDS < <(git ls-tree -r --name-only "origin/$SOURCE_BRANCH" -- docs/ \
  | grep -E '^docs/[^/]+\.md$' || true)

mapfile -t ROOT_MDS < <(git ls-tree --name-only "origin/$SOURCE_BRANCH" \
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

# --- 임시 release 브랜치 ----------------------------------------------------

SWITCHED=1
echo "==> building $RELEASE_BRANCH from origin/$TARGET_BRANCH"
git switch -q -C "$RELEASE_BRANCH" "origin/$TARGET_BRANCH"

echo "==> merging origin/$SOURCE_BRANCH"
# -X theirs 는 내용 충돌만 해결한다. main 이 지운 문서를 develop 이 고치면
# delete/modify 충돌(DU)이 되는데 여기에는 -X 가 적용되지 않아 머지가 멈춘다.
# 매번 생기는 정상 상황이므로, 충돌한 경로는 전부 develop 쪽으로 되살린 뒤
# 아래 "문서 제거" 단계에서 다시 지운다.
if ! git merge --no-ff -X theirs "origin/$SOURCE_BRANCH" \
      -m "merge $SOURCE_BRANCH into $TARGET_BRANCH"; then
  UNMERGED="$(git diff --name-only --diff-filter=U)"
  if [ -z "$UNMERGED" ]; then
    echo "error: merge failed with no conflicted paths. resolve it by hand." >&2
    exit 1
  fi
  echo "==> restoring $(printf '%s\n' "$UNMERGED" | grep -c .) conflicted path(s) from $SOURCE_BRANCH"
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    if git cat-file -e "origin/$SOURCE_BRANCH:$f" 2>/dev/null; then
      git checkout "origin/$SOURCE_BRANCH" -- "$f"
      git add -- "$f"
    else
      git rm -q -- "$f"
    fi
  done <<< "$UNMERGED"
  if [ -n "$(git diff --name-only --diff-filter=U)" ]; then
    echo "error: conflicts remain after auto-resolve. resolve them by hand." >&2
    exit 1
  fi
  git commit -q --no-edit
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

SLUG="$(git config --get remote.origin.url | sed -E 's#^.*github\.com[:/]##; s#\.git$##')"

if [ "$REMOVED" -gt 0 ]; then
  printf '%s\n' "${TO_REMOVE[@]}" > "$TMP_REMOVED"
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

  git commit -q -m "chore: drop design docs from $TARGET_BRANCH

The design documents live on $SOURCE_BRANCH. Links that pointed at them
are rewritten to absolute $SOURCE_BRANCH URLs so they still resolve."
  echo "==> removed $REMOVED design doc(s)"
else
  echo "==> no design docs to remove"
fi

# --- 결과 ------------------------------------------------------------------

if git diff --quiet "origin/$TARGET_BRANCH" "$RELEASE_BRANCH"; then
  echo "==> $TARGET_BRANCH is already up to date; nothing to propose"
  exit 0
fi

echo "==> $RELEASE_BRANCH is now:"
git --no-pager log --oneline -3

if [ "$OPEN_PR" -eq 0 ]; then
  SWITCHED=0   # 브랜치를 남겨 두고 그 위에 머문다
  echo "==> not pushed. review it, then run: bash scripts/sync-main.sh --pr"
  exit 0
fi

echo "==> pushing $RELEASE_BRANCH"
git push -q --force-with-lease origin "$RELEASE_BRANCH"

if ! command -v gh >/dev/null 2>&1; then
  echo "==> gh not found. open the PR by hand:"
  echo "    https://github.com/$SLUG/compare/$TARGET_BRANCH...$RELEASE_BRANCH"
  exit 0
fi

EXISTING="$(gh pr list --head "$RELEASE_BRANCH" --base "$TARGET_BRANCH" \
  --state open --json url -q '.[0].url' 2>/dev/null || true)"

if [ -n "$EXISTING" ]; then
  echo "==> updated the open pull request: $EXISTING"
else
  gh pr create \
    --base "$TARGET_BRANCH" \
    --head "$RELEASE_BRANCH" \
    --title "release: sync $SOURCE_BRANCH into $TARGET_BRANCH" \
    --body "$(cat <<BODY
\`scripts/sync-main.sh\` 가 만든 PR 이다.

- \`$SOURCE_BRANCH\` 를 \`$TARGET_BRANCH\` 에 머지한다.
- 설계 문서 ${REMOVED}개를 지운다. \`docs/\` 바로 아래의 \`.md\` 와 루트의 \`.md\`(\`README.md\` 제외)가 대상이고, \`docs/figures/\` · \`docs/materials/\` · \`docs/site/\` 는 그대로 둔다.
- 지운 문서를 가리키던 상대 링크는 \`$SOURCE_BRANCH\` 절대 URL 로 바꾼다.

머지하면 \`.github/workflows/pages.yml\` 이 \`docs/site/\` 를 GitHub Pages 에 배포한다.
BODY
)"
fi
