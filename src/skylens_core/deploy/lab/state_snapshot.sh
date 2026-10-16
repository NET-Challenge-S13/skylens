#!/bin/bash
# 노드 상태를 한 시간마다 기록한다. 측정값만 남기면 나중에 "그때 무슨 설정이었나"를
# 알 수 없다. 설정과 상태도 같이 남겨야 결과를 재현할 수 있다.
OUT=~/lab/results/state-$(date +%Y%m%d).log
{
  echo "===== $(date -Is) ====="
  echo "[uptime] $(uptime -p 2>/dev/null) | load $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "[disk] $(df -h / | awk 'NR==2{print $4" free ("$5" used)"}')"
  echo "[mem] $(free -m | awk '/Mem:/{print $3"MB used / "$2"MB"}')"
  echo "[crontab]"; crontab -l 2>/dev/null | grep -v '^#' | sed 's/^/  /'
  if command -v systemctl >/dev/null && systemctl list-units --all 2>/dev/null | grep -q pangyo-tunnel; then
    echo "[tunnel] $(systemctl is-active pangyo-tunnel.service) · 재시작 $(systemctl show pangyo-tunnel.service -p NRestarts --value)회"
    echo "[tunnel-port] $(ss -ltn | grep -c 19022) 개 리스닝"
  fi
  echo "[samples]"
  for f in ~/lab/results/*.jsonl; do
    [ -f "$f" ] || continue
    printf "  %-30s %5d\n" "$(basename "$f")" "$(wc -l < "$f")"
  done
  echo "[errors]"
  for e in ~/lab/results/*error.log; do
    [ -s "$e" ] && printf "  %s: %d 줄\n" "$(basename "$e")" "$(wc -l < "$e")"
  done
} >> "$OUT" 2>&1
