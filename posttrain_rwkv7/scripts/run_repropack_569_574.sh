#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PY_BIN="${PY_BIN:-./.venv/bin/python}"
QUEUE="${QUEUE:-results/_search_queue_round569_576_fastloop.json}"
POLICY="${POLICY:-results/_codex53_team_policy.json}"
ROUND_FROM="${ROUND_FROM:-569}"
ROUND_TO="${ROUND_TO:-574}"
LOG_PATH="${LOG_PATH:-runs/_launcher_round569_574_repropack.log}"

if [[ ! -x "$PY_BIN" ]]; then
  echo "python not found or not executable: $PY_BIN"
  echo "set PY_BIN=<python_path> and retry"
  exit 1
fi

if [[ ! -f "$QUEUE" ]]; then
  echo "queue file not found: $QUEUE"
  exit 1
fi

if [[ ! -f "$POLICY" ]]; then
  echo "policy file not found: $POLICY"
  exit 1
fi

mkdir -p runs

echo "[repropack] starting rounds ${ROUND_FROM}-${ROUND_TO}"
echo "[repropack] queue=${QUEUE}"
echo "[repropack] policy=${POLICY}"
echo "[repropack] log=${LOG_PATH}"

nohup "$PY_BIN" scripts/run_round77_82_fastdiscover.py \
  --queue "$QUEUE" \
  --round_from "$ROUND_FROM" \
  --round_to "$ROUND_TO" \
  --policy "$POLICY" \
  > "$LOG_PATH" 2>&1 &

echo "[repropack] started pid=$!"
echo "[repropack] tail -f $LOG_PATH"
