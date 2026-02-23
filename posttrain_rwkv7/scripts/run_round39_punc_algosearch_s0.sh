#!/usr/bin/env bash
set -euo pipefail

# Round39: aggressive algorithm search on known-working real scenes (cached datasets).
# Goal: maximize FS gain quickly with strict pruning.

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
Q_BUDGET=80
Q_STEPS=120
M_BUDGET=220
M_STEPS=320
PRUNE=0.005
TOPK=2

RUN_REC=runs/_round39_punc_algosearch_s0_records.jsonl
SUMMARY=runs/_summary_round39_punc_algosearch_s0.txt
LOG=runs/_log_round39_punc_algosearch_s0.$(date +%Y%m%d_%H%M%S).log
: > "$RUN_REC"

metric_best() {
  local run_dir="$1"
  ./.venv/bin/python - <<PY
import json
from pathlib import Path
p=Path("$run_dir")/"metrics.jsonl"
best_loss=None; best_acc=None
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
  local log="runs/_round39_${task}_${stage}_${cfg}.log"
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

run_task() {
  local task="$1"; shift
  local common_args=("$@")

  read -r _ b_acc _ < <(run_train "$task" quick baseline no_fs "$Q_BUDGET" "$Q_STEPS" "${common_args[@]}")
  [[ "$b_acc" == "nan" ]] && return

  # cfg_name|extra args
  local CANDS=(
    "head_l8|--alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0"
    "head_l10|--alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0"
    "scalar_l8_sched_cos|--alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0"
    "scalar_l10_sched_cos|--alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 10 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0"
    "scalar_l8_trainable|--alpha_lr 2e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0"
    "scalar_l8_node|--alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_clip 1.0"
  )

  local keep_cfgs=()
  local keep_deltas=()
  for item in "${CANDS[@]}"; do
    local cfg="${item%%|*}"
    local argstr="${item#*|}"
    # shellcheck disable=SC2206
    local args=($argstr)
    read -r _ f_acc _ < <(run_train "$task" quick "$cfg" prompt_fs "$Q_BUDGET" "$Q_STEPS" "${common_args[@]}" "${args[@]}")
    [[ "$f_acc" == "nan" ]] && continue
    local d
    d=$(python3 - <<PY
print(float("$f_acc")-float("$b_acc"))
PY
)
    local keep
    keep=$(python3 - <<PY
print(1 if float("$d") >= float("$PRUNE") else 0)
PY
)
    if [[ "$keep" == "1" ]]; then
      keep_cfgs+=("$cfg")
      keep_deltas+=("$d")
    fi
  done

  # Promote top-K by quick delta.
  if [[ "${#keep_cfgs[@]}" -gt 0 ]]; then
    ./.venv/bin/python - <<PY > /tmp/round39_rank_${task}.txt
cfgs=${keep_cfgs[@]+"${keep_cfgs[*]}"}
deltas=${keep_deltas[@]+"${keep_deltas[*]}"}
cfg_list = cfgs.split() if isinstance(cfgs,str) else []
d_list = [float(x) for x in (deltas.split() if isinstance(deltas,str) else [])]
pairs=sorted(zip(cfg_list,d_list), key=lambda x:x[1], reverse=True)
for c,d in pairs[:$TOPK]:
    print(c, d)
PY
    while read -r cfg _; do
      [[ -z "${cfg:-}" ]] && continue
      local argstr=""
      for item in "${CANDS[@]}"; do
        local c="${item%%|*}"
        if [[ "$c" == "$cfg" ]]; then
          argstr="${item#*|}"
          break
        fi
      done
      # shellcheck disable=SC2206
      local args=($argstr)
      run_train "$task" med "$cfg" prompt_fs "$M_BUDGET" "$M_STEPS" "${common_args[@]}" "${args[@]}" >/dev/null
    done < /tmp/round39_rank_${task}.txt
  fi
}

{
  echo "Round39 started: $(date)"

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

  run_task punc_squad \
    --ds squad --ds_cfg '' --train_split train --val_split validation \
    --n_train 900 --n_val 180 --min_chars 64 --max_chars 260 \
    --fill_notes_to_max --note_pool_size 1024 \
    --max_prompt_tokens 1536 --min_prompt_tokens 512 --max_answer_tokens 128 \
    --eval_every 20 --val_batches 4

  ./.venv/bin/python - <<'PY' | tee "$SUMMARY"
import json
from pathlib import Path
rows=[json.loads(x) for x in Path("runs/_round39_punc_algosearch_s0_records.jsonl").read_text().splitlines() if x.strip()]
print("="*116)
print("Round39 punc algorithm search summary")
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
        print("  med (top2):")
        for r in sorted(m,key=lambda x:float(x["best_val_tok_acc"]), reverse=True):
            d=float(r["best_val_tok_acc"])-bacc
            print(f"    {r['config']:22s} d_acc={d*100:+.2f}pp acc={float(r['best_val_tok_acc'])*100:.2f}%")
    else:
        print("  med skipped (no config passed prune)")
    print("-"*116)
PY
  echo "Round39 finished: $(date)"
} 2>&1 | tee "$LOG"

