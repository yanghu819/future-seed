#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=${VENV_DIR:-.venv}
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MODEL_ID=${MODEL_ID:-RWKV/RWKV7-Goose-World3-2.9B-HF}
MODEL_DIR=${MODEL_DIR:-models/${MODEL_ID//\//__}}
TASK=${TASK:-arc_mask}
DATA_ROOT=${DATA_ROOT:-data}
RUN_ROOT=${RUN_ROOT:-runs/run1}
mkdir -p "$RUN_ROOT"

case "$TASK" in
  arc_mask)
    DEFAULT_TRAIN_JSONL="$DATA_ROOT/arc/train.jsonl"
    DEFAULT_EVAL_JSONL="$DATA_ROOT/arc/eval.jsonl"
    ;;
  mbpp_mask)
    DEFAULT_TRAIN_JSONL="$DATA_ROOT/mbpp/train.jsonl"
    DEFAULT_EVAL_JSONL="$DATA_ROOT/mbpp/eval.jsonl"
    ;;
  race_mask)
    DEFAULT_TRAIN_JSONL="$DATA_ROOT/race/train.jsonl"
    DEFAULT_EVAL_JSONL="$DATA_ROOT/race/eval.jsonl"
    ;;
  *)
    echo "[run1] unsupported task=$TASK" >&2
    exit 2
    ;;
esac
TRAIN_JSONL=${TRAIN_JSONL:-$DEFAULT_TRAIN_JSONL}
EVAL_JSONL=${EVAL_JSONL:-$DEFAULT_EVAL_JSONL}

COMMON_ARGS=(
  --task "$TASK"
  --model_id "$MODEL_ID"
  --model_local_dir "$MODEL_DIR"
  --train_jsonl "$TRAIN_JSONL"
  --eval_jsonl "$EVAL_JSONL"
  --train_limit "256"
  --eval_limit "128"
  --max_steps "8"
  --eval_every "4"
  --batch_size "1"
  --grad_accum "4"
  --lr "2e-4"
  --max_length "768"
  --load_in_4bit
)

printf '[run1] task=%s model=%s\n' "$TASK" "$MODEL_ID"
printf '[run1] baseline smoke\n'
python train_mask_probe.py "${COMMON_ARGS[@]}" --output_dir "$RUN_ROOT/baseline"
printf '[run1] fs smoke\n'
python train_mask_probe.py "${COMMON_ARGS[@]}" --future_seed --future_seed_layer_start 1 --future_seed_alpha_init -2.0 --output_dir "$RUN_ROOT/fs"

python - <<'PY'
import json
from pathlib import Path
root = Path('runs/run1')
base = json.loads((root / 'baseline/summary.json').read_text())
fs = json.loads((root / 'fs/summary.json').read_text())
print(json.dumps({
  'event': 'run1_summary',
  'task': fs['task'],
  'baseline_mask_exact': base['best_mask_exact'],
  'fs_mask_exact': fs['best_mask_exact'],
  'baseline_mask_token_acc': base['best_mask_token_acc'],
  'fs_mask_token_acc': fs['best_mask_token_acc'],
}))
PY
