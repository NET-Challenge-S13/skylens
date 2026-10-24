#!/bin/bash
# The daily roll-up on Daejeon: summarise everything up to yesterday into a report file.
cd ~/lab || exit 1
mkdir -p reports
DAY=$(date +%Y-%m-%d)
OUT=reports/$DAY.md
{
  echo "# KOREN 대조 측정 집계 — $DAY"
  echo
  echo "생성: $(date -Is) · 노드: 대전(KOREN) · 파이(상용 회선)"
  echo
  python3 aggregate.py --md results/*.jsonl 2>/dev/null
  echo
  echo "## 수집 상태"
  echo
  echo '```'
  for f in results/*.jsonl; do
    [ -f "$f" ] || continue
    printf "%-28s %6d 표본  마지막 %s\n" "$(basename "$f")" "$(wc -l < "$f")" "$(tail -1 "$f" | grep -o '"ts": "[^"]*"' | cut -d'"' -f4)"
  done
  echo '```'
} > "$OUT"
echo "$OUT"
