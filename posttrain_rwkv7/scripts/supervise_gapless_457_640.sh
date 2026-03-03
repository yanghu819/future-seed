#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/future-seed-posttrain

LOG="runs/_supervise_gapless_457_640.log"
START=457
END=568
POLICY="results/_codex53_team_policy.json"

echo "[$(date +%F_%T)] supervisor started" >> "$LOG"

while true; do
  if ps -eo cmd | grep -q "run_round77_82_fastdiscover.py --queue"; then
    sleep 20
    continue
  fi

  LAST=$(ls runs/_summary_round*_fastdiscover.txt 2>/dev/null | sed -E 's@.*_summary_round([0-9]+)_fastdiscover.txt@\1@' | sort -n | tail -n 1)
  if [ -z "${LAST:-}" ]; then
    LAST=$((START - 1))
  fi

  if [ "$LAST" -ge "$END" ]; then
    echo "[$(date +%F_%T)] completed through round${LAST}, supervisor exit" >> "$LOG"
    exit 0
  fi

  NEXT=$((LAST + 1))
  if [ "$NEXT" -lt "$START" ]; then
    NEXT=$START
  fi

  PICK=$(python3 - "$NEXT" <<'PY'
import glob
import re
import sys

start = int(sys.argv[1])
for f in sorted(glob.glob("results/_search_queue_round*_fastloop.json")):
    m = re.search(r"_round(\d+)_(\d+)_fastloop\.json$", f)
    if not m:
        continue
    a, b = int(m.group(1)), int(m.group(2))
    if a <= start <= b:
        print(f"{f} {a} {b}")
        break
PY
)

  if [ -z "${PICK:-}" ]; then
    echo "[$(date +%F_%T)] no queue covers round${NEXT}, sleeping" >> "$LOG"
    sleep 30
    continue
  fi

  Q=$(echo "$PICK" | awk '{print $1}')
  QEND=$(echo "$PICK" | awk '{print $3}')
  echo "[$(date +%F_%T)] launching from round${NEXT} via ${Q} (to ${QEND})" >> "$LOG"

  nohup ./.venv/bin/python scripts/run_round77_82_fastdiscover.py \
    --queue "$Q" \
    --round_from "$NEXT" \
    --round_to "$QEND" \
    --policy "$POLICY" \
    > "runs/_launcher_round${NEXT}_${QEND}_autoheal.log" 2>&1 &

  sleep 10
done
