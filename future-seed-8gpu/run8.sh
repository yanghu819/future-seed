#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=${VENV_DIR:-.venv}
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

MODEL_ID=${MODEL_ID:-RWKV/RWKV7-Goose-World3-2.9B-HF}
MODEL_DIR=${MODEL_DIR:-models/${MODEL_ID//\//__}}
DATA_ROOT=${DATA_ROOT:-data}
RUN_ROOT=${RUN_ROOT:-runs/run8}
TASKS=${TASKS:-arc_mask,mbpp_mask}
NPROC=${NPROC:-8}
mkdir -p "$RUN_ROOT"

run_task() {
  local task="$1"
  local fs_flag="$2"
  local tag="$3"
  local train_jsonl eval_jsonl max_steps eval_every max_length grad_accum train_limit eval_limit

  case "$task" in
    arc_mask)
      train_jsonl="$DATA_ROOT/arc/train.jsonl"
      eval_jsonl="$DATA_ROOT/arc/eval.jsonl"
      max_steps=${ARC_MAX_STEPS:-120}
      eval_every=${ARC_EVAL_EVERY:-60}
      max_length=${ARC_MAX_LENGTH:-768}
      grad_accum=${ARC_GRAD_ACCUM:-4}
      train_limit=${ARC_TRAIN_LIMIT:-4000}
      eval_limit=${ARC_EVAL_LIMIT:-512}
      ;;
    mbpp_mask)
      train_jsonl="$DATA_ROOT/mbpp/train.jsonl"
      eval_jsonl="$DATA_ROOT/mbpp/eval.jsonl"
      max_steps=${MBPP_MAX_STEPS:-120}
      eval_every=${MBPP_EVAL_EVERY:-60}
      max_length=${MBPP_MAX_LENGTH:-1024}
      grad_accum=${MBPP_GRAD_ACCUM:-8}
      train_limit=${MBPP_TRAIN_LIMIT:-1200}
      eval_limit=${MBPP_EVAL_LIMIT:-256}
      ;;
    race_mask)
      train_jsonl="$DATA_ROOT/race/train.jsonl"
      eval_jsonl="$DATA_ROOT/race/eval.jsonl"
      max_steps=${RACE_MAX_STEPS:-120}
      eval_every=${RACE_EVAL_EVERY:-60}
      max_length=${RACE_MAX_LENGTH:-1024}
      grad_accum=${RACE_GRAD_ACCUM:-4}
      train_limit=${RACE_TRAIN_LIMIT:-4000}
      eval_limit=${RACE_EVAL_LIMIT:-512}
      ;;
    *)
      echo "[run8] unsupported task=$task" >&2
      exit 2
      ;;
  esac

  local out_dir="$RUN_ROOT/$tag"
  mkdir -p "$out_dir"
  printf '[run8] task=%s tag=%s future_seed=%s\n' "$task" "$tag" "$fs_flag"
  torchrun --standalone --nproc_per_node "$NPROC" train_mask_probe.py \
    --task "$task" \
    --model_id "$MODEL_ID" \
    --model_local_dir "$MODEL_DIR" \
    --train_jsonl "$train_jsonl" \
    --eval_jsonl "$eval_jsonl" \
    --output_dir "$out_dir" \
    --train_limit "$train_limit" \
    --eval_limit "$eval_limit" \
    --max_steps "$max_steps" \
    --eval_every "$eval_every" \
    --batch_size 1 \
    --grad_accum "$grad_accum" \
    --lr 2e-4 \
    --max_length "$max_length" \
    --bf16 \
    $fs_flag
  python - <<PY
import json
from pathlib import Path
p = Path('$out_dir/summary.json')
obj = json.loads(p.read_text())
print(json.dumps({'event':'task_summary','task':obj['task'],'future_seed':obj['future_seed'],'best_mask_exact':obj['best_mask_exact'],'best_mask_token_acc':obj['best_mask_token_acc'],'output_dir':str(p.parent)}))
PY
}

IFS=',' read -r -a task_list <<< "$TASKS"
for task in "${task_list[@]}"; do
  run_task "$task" "" "${task}_baseline"
  run_task "$task" "--future_seed --future_seed_layer_start 1 --future_seed_alpha_init -2.0" "${task}_fs"
done

python - <<'PY'
import json
from pathlib import Path
root = Path('runs/run8')
rows = []
for p in sorted(root.glob('*/summary.json')):
    obj = json.loads(p.read_text())
    rows.append({'tag': p.parent.name, 'task': obj['task'], 'future_seed': obj['future_seed'], 'mask_exact': obj['best_mask_exact'], 'mask_token_acc': obj['best_mask_token_acc']})
print(json.dumps({'event':'run8_done','rows':rows}, ensure_ascii=False))
PY
