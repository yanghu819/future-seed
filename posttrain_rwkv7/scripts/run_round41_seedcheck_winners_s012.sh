#!/usr/bin/env bash
set -euo pipefail

# Round41: seed-check (s0/s1/s2) for Round40 winners.
# Quick-only to validate robustness before further expansion.

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

SEEDS="0 1 2"

RUN_REC=runs/_round41_seedcheck_winners_s012_records.jsonl
SUMMARY=runs/_summary_round41_seedcheck_winners_s012.txt
LOG=runs/_log_round41_seedcheck_winners_s012.$(date +%Y%m%d_%H%M%S).log
: > "$RUN_REC"

metric_best() {
  local run_dir="$1"
  ./.venv/bin/python - <<PY
import json
from pathlib import Path
p=Path("$run_dir")/"metrics.jsonl"
best_acc=None
for line in p.read_text().splitlines():
    if not line.strip():
        continue
    r=json.loads(line)
    if "val_tok_acc" in r:
        v=float(r["val_tok_acc"])
        best_acc=v if best_acc is None or v>best_acc else best_acc
print(best_acc if best_acc is not None else "nan")
PY
}

record() { printf "%s\n" "$1" >> "$RUN_REC"; }

run_one() {
  local task="$1"; shift
  local seed="$1"; shift
  local stage="$1"; shift
  local cfg="$1"; shift
  local mode="$1"; shift
  local script="$1"; shift
  local budget="$1"; shift
  local steps="$1"; shift
  local log="runs/_round41_${task}_s${seed}_${cfg}.log"
  rm -f "$log"
  set +e
  ./.venv/bin/python "${PY_PREFIX}${script}" \
    --train_data_seed "$seed" --val_data_seed 1234 \
    --mode "$mode" --seed "$seed" \
    --time_budget_sec "$budget" --max_steps "$steps" \
    "$@" > "$log" 2>&1
  local code=$?
  set -e
  if [[ $code -ne 0 ]]; then
    local err
    err=$(tail -n 3 "$log" | tr '\n' ' ' | sed 's/"/\\"/g')
    record "{\"task\":\"$task\",\"seed\":$seed,\"stage\":\"$stage\",\"config\":\"$cfg\",\"status\":\"fail\",\"best_val_tok_acc\":null,\"run_dir\":\"\",\"error\":\"$err\"}"
    echo "nan"
    return
  fi
  local run_dir
  run_dir=$(tail -n 1 "$log")
  local best_acc
  best_acc=$(metric_best "$run_dir")
  record "{\"task\":\"$task\",\"seed\":$seed,\"stage\":\"$stage\",\"config\":\"$cfg\",\"status\":\"ok\",\"best_val_tok_acc\":$best_acc,\"run_dir\":\"$run_dir\"}"
  echo "$best_acc"
}

run_pair() {
  local task="$1"; shift
  local script="$1"; shift
  local budget="$1"; shift
  local steps="$1"; shift
  local fs_cfg="$1"; shift
  local fs_mode="$1"; shift
  local fs_extra=("$@")

  for s in $SEEDS; do
    local bacc facc
    bacc=$(run_one "$task" "$s" quick baseline no_fs "$script" "$budget" "$steps" "${fs_extra[@]}")
    facc=$(run_one "$task" "$s" quick "$fs_cfg" "$fs_mode" "$script" "$budget" "$steps" "${fs_extra[@]}")
    echo "PAIR task=$task seed=$s baseline=$bacc fs=$facc"
  done
}

{
  echo "Round41 started: $(date)"

  # Winner 1: punc_hotpot + head_l8
  run_pair punc_hotpot train_punc_restore_sft.py 80 120 head_l8 prompt_fs \
    --model_lr 3e-5 --seed_scale 1.0 \
    --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 \
    --ds hotpot_qa --ds_cfg distractor --train_split train --val_split validation \
    --n_train 800 --n_val 160 --min_chars 48 --max_chars 220 \
    --fill_notes_to_max --note_pool_size 1024 \
    --max_prompt_tokens 1536 --min_prompt_tokens 512 --max_answer_tokens 128 \
    --eval_every 20 --val_batches 4 --bsz 4

  # Winner 2: punc_mbpp + scalar_l8_trainable
  run_pair punc_mbpp train_punc_restore_sft.py 80 120 scalar_l8_trainable prompt_fs \
    --model_lr 3e-5 --seed_scale 1.0 \
    --alpha_lr 2e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 \
    --ds mbpp --ds_cfg '' --train_split train --val_split test \
    --n_train 320 --n_val 80 --min_chars 32 --max_chars 360 \
    --fill_notes_to_max --note_pool_size 512 \
    --max_prompt_tokens 1536 --min_prompt_tokens 512 --max_answer_tokens 160 \
    --eval_every 20 --val_batches 4 --bsz 4

  # Winner 3: punc_squad + scalar_l8_sched_cos
  run_pair punc_squad train_punc_restore_sft.py 80 120 scalar_l8_sched_cos prompt_fs \
    --model_lr 3e-5 --seed_scale 1.0 \
    --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0 \
    --ds squad --ds_cfg '' --train_split train --val_split validation \
    --n_train 900 --n_val 180 --min_chars 64 --max_chars 260 \
    --fill_notes_to_max --note_pool_size 1024 \
    --max_prompt_tokens 1536 --min_prompt_tokens 512 --max_answer_tokens 128 \
    --eval_every 20 --val_batches 4 --bsz 4

  # Winner 4: mbpp_longctx_qafter + head_l8
  run_pair mbpp_longctx_qafter train_mbpp_longctx_sft.py 90 120 head_l8 prompt_fs \
    --model_lr 3e-5 --seed_scale 1.0 \
    --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 \
    --ds mbpp --ds_cfg '' --train_split train --val_split test \
    --n_train 320 --n_val 120 \
    --fill_notes_to_max --note_pool_size 1024 \
    --max_prompt_tokens 3072 --min_prompt_tokens 1024 --max_answer_tokens 160 \
    --eval_every 25 --val_batches 8 --bsz 1

  ./.venv/bin/python - <<'PY' | tee "$SUMMARY"
import json
from pathlib import Path
rows=[json.loads(x) for x in Path("runs/_round41_seedcheck_winners_s012_records.jsonl").read_text().splitlines() if x.strip()]
print("="*116)
print("Round41 seedcheck winners s012 summary")
print("="*116)
for task in sorted({r["task"] for r in rows}):
    print(f"[{task}]")
    ds=[]
    for seed in [0,1,2]:
        b=[r for r in rows if r["task"]==task and r["seed"]==seed and r["config"]=="baseline" and r["status"]=="ok"]
        f=[r for r in rows if r["task"]==task and r["seed"]==seed and r["config"]!="baseline" and r["status"]=="ok"]
        if not b or not f:
            continue
        bacc=float(b[0]["best_val_tok_acc"]); facc=float(f[0]["best_val_tok_acc"]); d=facc-bacc
        ds.append(d)
        print(f"  seed={seed}: baseline={bacc*100:.2f}% fs={facc*100:.2f}% d_acc={d*100:+.2f}pp")
    if ds:
        print(f"  mean d_acc={(sum(ds)/len(ds))*100:+.2f}pp, positive_seeds={sum(1 for x in ds if x>0)}/{len(ds)}")
    else:
        print("  no complete pairs")
    print("-"*116)
PY
  echo "Round41 finished: $(date)"
} 2>&1 | tee "$LOG"

