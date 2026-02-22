#!/usr/bin/env bash
set -euo pipefail

# Round36: real-task punctuation-restore scene discovery with aggressive pruning.
# - Serial, single GPU.
# - Quick stage on multiple real datasets.
# - Promote only TOP-1 FS config per task to medium stage.
# - Enforce positive quick gain threshold.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
  cd "$SCRIPT_DIR/.."
else
  cd "$SCRIPT_DIR"
fi

if [[ -f scripts/train_punc_restore_sft.py ]]; then
  PY_PREFIX="scripts/"
else
  PY_PREFIX=""
fi

export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/torch_extensions
export HF_HOME=/root/autodl-tmp/hf
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_transformers
export HF_ENDPOINT=https://huggingface.co
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=8.9

SEED=0
BSZ=4
Q_BUDGET=90
Q_STEPS=130
M_BUDGET=220
M_STEPS=320
PRUNE=0.005

RUN_REC=runs/_round36_punc_scene_discovery_s0_records.jsonl
SUMMARY=runs/_summary_round36_punc_scene_discovery_s0.txt
LOG=runs/_log_round36_punc_scene_discovery_s0.$(date +%Y%m%d_%H%M%S).log
: > "$RUN_REC"

metric_best() {
  local run_dir="$1"
  ./.venv/bin/python - <<PY
import json
from pathlib import Path
p=Path("$run_dir")/"metrics.jsonl"
best_loss=None
best_acc=None
for line in p.read_text().splitlines():
    if not line.strip():
        continue
    r=json.loads(line)
    if "val_loss" in r:
        v=float(r["val_loss"])
        best_loss=v if best_loss is None or v<best_loss else best_loss
    if "val_tok_acc" in r:
        v=float(r["val_tok_acc"])
        best_acc=v if best_acc is None or v>best_acc else best_acc
print(f"{best_loss}\t{best_acc}")
PY
}

record() { printf "%s\n" "$1" >> "$RUN_REC"; }

run_train() {
  local task="$1"; shift
  local stage="$1"; shift
  local cfg="$1"; shift
  local mode="$1"; shift
  local budget="$1"; shift
  local steps="$1"; shift
  local log="runs/_round36_${task}_${stage}_${cfg}.log"
  rm -f "$log"
  echo "RUN task=$task stage=$stage cfg=$cfg mode=$mode" >&2
  set +e
  ./.venv/bin/python "${PY_PREFIX}train_punc_restore_sft.py" \
    --train_data_seed "$SEED" --val_data_seed 1234 \
    --model_lr 3e-5 --seed_scale 1.0 \
    --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 \
    "$@" --mode "$mode" --seed "$SEED" --bsz "$BSZ" \
    --time_budget_sec "$budget" --max_steps "$steps" > "$log" 2>&1
  local code=$?
  set -e
  if [[ $code -ne 0 ]]; then
    local err
    err=$(tail -n 3 "$log" | tr '\n' ' ' | sed 's/"/\\"/g')
    record "{\"task\":\"$task\",\"stage\":\"$stage\",\"config\":\"$cfg\",\"status\":\"fail\",\"bsz\":$BSZ,\"best_val_loss\":null,\"best_val_tok_acc\":null,\"run_dir\":\"\",\"error\":\"$err\"}"
    echo -e "nan\tnan\t"
    return
  fi
  local run_dir
  run_dir=$(tail -n 1 "$log")
  read -r best_loss best_acc < <(metric_best "$run_dir")
  record "{\"task\":\"$task\",\"stage\":\"$stage\",\"config\":\"$cfg\",\"status\":\"ok\",\"bsz\":$BSZ,\"best_val_loss\":$best_loss,\"best_val_tok_acc\":$best_acc,\"run_dir\":\"$run_dir\"}"
  echo -e "${best_loss}\t${best_acc}\t${run_dir}"
}

select_and_promote() {
  local task="$1"; shift
  local b_acc="$1"; shift
  local q_head="$1"; shift
  local q_scalar="$1"; shift
  local common_args=("$@")

  local d_head d_scalar
  d_head=$(python3 - <<PY
print(float("$q_head")-float("$b_acc"))
PY
)
  d_scalar=$(python3 - <<PY
print(float("$q_scalar")-float("$b_acc"))
PY
)

  local best_cfg best_delta
  if python3 - <<PY | grep -q 1; then
print(1 if float("$d_head")>=float("$d_scalar") else 0)
PY
    best_cfg="head_l8"
    best_delta="$d_head"
  else
    best_cfg="scalar_l8_sched_cos"
    best_delta="$d_scalar"
  fi

  local keep
  keep=$(python3 - <<PY
print(1 if float("$best_delta") >= float("$PRUNE") else 0)
PY
)

  if [[ "$keep" == "1" ]]; then
    if [[ "$best_cfg" == "head_l8" ]]; then
      run_train "$task" med head_l8 prompt_fs "$M_BUDGET" "$M_STEPS" "${common_args[@]}" \
        --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
    else
      run_train "$task" med scalar_l8_sched_cos prompt_fs "$M_BUDGET" "$M_STEPS" "${common_args[@]}" \
        --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0 >/dev/null
    fi
  fi
}

run_task() {
  local task="$1"; shift
  local common_args=("$@")

  read -r _ b_acc _ < <(run_train "$task" quick baseline no_fs "$Q_BUDGET" "$Q_STEPS" "${common_args[@]}")
  [[ "$b_acc" == "nan" ]] && return

  read -r _ q_head _ < <(run_train "$task" quick head_l8 prompt_fs "$Q_BUDGET" "$Q_STEPS" "${common_args[@]}" \
    --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0)

  read -r _ q_scalar _ < <(run_train "$task" quick scalar_l8_sched_cos prompt_fs "$Q_BUDGET" "$Q_STEPS" "${common_args[@]}" \
    --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0)

  if [[ "$q_head" == "nan" || "$q_scalar" == "nan" ]]; then
    return
  fi

  select_and_promote "$task" "$b_acc" "$q_head" "$q_scalar" "${common_args[@]}"
}

{
  echo "Round36 started: $(date)"

  # Fix known broken/incomplete cache shard that previously blocked wikitext runs.
  python3 - <<'PY'
from pathlib import Path
import shutil
root=Path("/root/autodl-tmp/hf_datasets/wikitext/wikitext-2-raw-v1/0.0.0")
if root.exists():
    for p in root.glob("*.incomplete"):
        shutil.rmtree(p, ignore_errors=True)
print("wikitext incomplete cache cleaned")
PY

  # Control positives.
  run_task punc_hotpot \
    --ds hotpot_qa --ds_cfg distractor --train_split train --val_split validation \
    --n_train 800 --n_val 160 --min_chars 48 --max_chars 220 \
    --fill_notes_to_max --note_pool_size 1024 \
    --max_prompt_tokens 1536 --min_prompt_tokens 512 --max_answer_tokens 128 \
    --eval_every 20 --val_batches 4

  run_task punc_mbpp \
    --ds mbpp --ds_cfg '' --train_split train --val_split test \
    --n_train 320 --n_val 80 --min_chars 32 --max_chars 360 \
    --fill_notes_to_max --note_pool_size 512 \
    --max_prompt_tokens 1536 --min_prompt_tokens 512 --max_answer_tokens 160 \
    --eval_every 20 --val_batches 4

  # New real-text scenes.
  run_task punc_wikitext \
    --ds wikitext --ds_cfg wikitext-2-raw-v1 --train_split train --val_split validation \
    --n_train 1200 --n_val 240 --min_chars 64 --max_chars 220 \
    --fill_notes_to_max --note_pool_size 1024 \
    --max_prompt_tokens 1536 --min_prompt_tokens 512 --max_answer_tokens 128 \
    --eval_every 20 --val_batches 4

  run_task punc_agnews \
    --ds ag_news --ds_cfg '' --train_split train --val_split test \
    --n_train 1200 --n_val 240 --min_chars 48 --max_chars 220 \
    --fill_notes_to_max --note_pool_size 1024 \
    --max_prompt_tokens 1536 --min_prompt_tokens 512 --max_answer_tokens 128 \
    --eval_every 20 --val_batches 4

  run_task punc_squad \
    --ds squad --ds_cfg '' --train_split train --val_split validation \
    --n_train 900 --n_val 180 --min_chars 64 --max_chars 260 \
    --fill_notes_to_max --note_pool_size 1024 \
    --max_prompt_tokens 1536 --min_prompt_tokens 512 --max_answer_tokens 128 \
    --eval_every 20 --val_batches 4

  ./.venv/bin/python - <<'PY' | tee "$SUMMARY"
import json
from pathlib import Path
rows=[json.loads(x) for x in Path("runs/_round36_punc_scene_discovery_s0_records.jsonl").read_text().splitlines() if x.strip()]
print("="*116)
print("Round36 punc scene discovery summary")
print("="*116)
for task in sorted({r["task"] for r in rows}):
    print(f"[{task}]")
    b=[r for r in rows if r["task"]==task and r["stage"]=="quick" and r["config"]=="baseline" and r["status"]=="ok"]
    if not b:
        print("  baseline failed")
        print("-"*116)
        continue
    bacc=float(b[0]["best_val_tok_acc"])
    print(f"  baseline quick: acc={bacc*100:.2f}%")
    q=[r for r in rows if r["task"]==task and r["stage"]=="quick" and r["config"]!="baseline" and r["status"]=="ok"]
    for r in sorted(q,key=lambda x:float(x["best_val_tok_acc"])-bacc, reverse=True):
        d=float(r["best_val_tok_acc"])-bacc
        print(f"    {r['config']:22s} d_acc={d*100:+.2f}pp acc={float(r['best_val_tok_acc'])*100:.2f}%")
    m=[r for r in rows if r["task"]==task and r["stage"]=="med" and r["status"]=="ok"]
    if m:
        print("  med (top1 only):")
        for r in sorted(m,key=lambda x:float(x["best_val_tok_acc"]), reverse=True):
            d=float(r["best_val_tok_acc"])-bacc
            print(f"    {r['config']:22s} d_acc={d*100:+.2f}pp acc={float(r['best_val_tok_acc'])*100:.2f}%")
    else:
        print("  med skipped (no config passed prune)")
    print("-"*116)
PY

  echo "Round36 finished: $(date)"
} 2>&1 | tee "$LOG"

