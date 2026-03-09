#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

LOG_DIR=artifacts/down
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/down.log"

MODEL_ID=${MODEL_ID:-RWKV/RWKV7-Goose-World3-2.9B-HF}
DATASETS=${DATASETS:-mbpp,arc,race,humaneval}
OUT_DIR=${OUT_DIR:-data}
CACHE_DIR=${CACHE_DIR:-cache/modelscope}
MODEL_DIR=${MODEL_DIR:-models}
VENV_DIR=${VENV_DIR:-.venv}

printf '[down] model=%s\n' "$MODEL_ID"
printf '[down] datasets=%s\n' "$DATASETS"
printf '[down] log=%s\n' "$LOG_FILE"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python download_assets.py \
  --model_id "$MODEL_ID" \
  --datasets "$DATASETS" \
  --out_dir "$OUT_DIR" \
  --cache_dir "$CACHE_DIR" \
  --model_dir "$MODEL_DIR" \
  >"$LOG_FILE" 2>&1

tail -n 8 "$LOG_FILE"
printf '[down] done\n'
