#!/usr/bin/env bash
set -euo pipefail

# Round27: robustness check for newly-found positive regimes.
# - mbpp_low: scalar_l8_trainable
# - punc_restore: head_l8
# seeds: 0,1,2 (serial)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
  cd "$SCRIPT_DIR/.."
else
  cd "$SCRIPT_DIR"
fi

if [[ -f scripts/train_mbpp_longctx_sft.py ]]; then
  PY_PREFIX="scripts/"
else
  PY_PREFIX=""
fi

export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/torch_extensions
export HF_HOME=/root/autodl-tmp/hf
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_transformers
export HF_ENDPOINT=https://huggingface.co
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=8.9

SEEDS="0 1 2"
BSZ=2
Q_BUDGET=110
Q_STEPS=170
M_BUDGET=300
M_STEPS=460
PRUNE_MBPP=0.004
PRUNE_PUNC=0.003

RUN_REC=runs/_round27_seedcheck_positive_s012_records.jsonl
SUMMARY=runs/_summary_round27_seedcheck_positive_s012.txt
LOG=runs/_log_round27_seedcheck_positive_s012.$(date +%Y%m%d_%H%M%S).log
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
        v=float(r["val_loss"]); best_loss=v if best_loss is None or v<best_loss else best_loss
    if "val_tok_acc" in r:
        v=float(r["val_tok_acc"]); best_acc=v if best_acc is None or v>best_acc else best_acc
print(f"{best_loss}\t{best_acc}")
PY
}

record() { printf "%s\n" "$1" >> "$RUN_REC"; }

run_train() {
  local task="$1"; shift
  local seed="$1"; shift
  local stage="$1"; shift
  local cfg="$1"; shift
  local trainer="$1"; shift
  local mode="$1"; shift
  local budget="$1"; shift
  local steps="$1"; shift
  local log="runs/_round27_${task}_s${seed}_${stage}_${cfg}.log"
  rm -f "$log"
  echo "RUN task=$task seed=$seed stage=$stage cfg=$cfg mode=$mode budget=$budget" >&2
  set +e
  ./.venv/bin/python "${PY_PREFIX}${trainer}" "$@" --mode "$mode" --seed "$seed" --bsz "$BSZ" \
    --time_budget_sec "$budget" --max_steps "$steps" > "$log" 2>&1
  local code=$?
  set -e
  if [[ $code -ne 0 ]]; then
    local err
    err=$(tail -n 2 "$log" | tr '\n' ' ' | sed 's/"/\\"/g')
    record "{\"task\":\"$task\",\"seed\":$seed,\"stage\":\"$stage\",\"config\":\"$cfg\",\"status\":\"fail\",\"bsz\":$BSZ,\"best_val_loss\":null,\"best_val_tok_acc\":null,\"run_dir\":\"\",\"error\":\"$err\"}"
    echo -e "nan\tnan\t"
    return
  fi
  local run_dir
  run_dir=$(tail -n 1 "$log")
  read -r best_loss best_acc < <(metric_best "$run_dir")
  record "{\"task\":\"$task\",\"seed\":$seed,\"stage\":\"$stage\",\"config\":\"$cfg\",\"status\":\"ok\",\"bsz\":$BSZ,\"best_val_loss\":$best_loss,\"best_val_tok_acc\":$best_acc,\"run_dir\":\"$run_dir\"}"
  echo -e "${best_loss}\t${best_acc}\t${run_dir}"
}

run_mbpp_seed() {
  local s="$1"
  local -a base=(
    --train_data_seed "$s" --val_data_seed 1234
    --ds mbpp --train_split train --val_split test
    --fill_notes_to_max --note_pool_size 1024
    --n_train 320 --n_val 80
    --max_prompt_tokens 3072 --min_prompt_tokens 768 --max_answer_tokens 160
    --eval_every 20 --val_batches 4
    --model_lr 3e-5 --seed_scale 1.0
    --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0
  )
  read -r _ b_acc _ < <(run_train mbpp_low "$s" quick baseline train_mbpp_longctx_sft.py no_fs "$Q_BUDGET" "$Q_STEPS" "${base[@]}")
  [[ "$b_acc" == "nan" ]] && return
  read -r _ q_acc _ < <(run_train mbpp_low "$s" quick scalar_l8_trainable train_mbpp_longctx_sft.py prompt_fs "$Q_BUDGET" "$Q_STEPS" "${base[@]}" --alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0)
  [[ "$q_acc" == "nan" ]] && return
  d_acc=$(python3 - <<PY
print(float("$q_acc")-float("$b_acc"))
PY
)
  keep=$(python3 - <<PY
print(1 if float("$d_acc") >= float("$PRUNE_MBPP") else 0)
PY
)
  if [[ "$keep" == "1" ]]; then
    run_train mbpp_low "$s" med scalar_l8_trainable train_mbpp_longctx_sft.py prompt_fs "$M_BUDGET" "$M_STEPS" "${base[@]}" --alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
  fi
}

run_punc_seed() {
  local s="$1"
  local -a base=(
    --train_data_seed "$s" --val_data_seed 1234
    --ds hotpot_qa --ds_cfg distractor --train_split train --val_split validation
    --n_train 800 --n_val 160
    --min_chars 48 --max_chars 220
    --fill_notes_to_max --note_pool_size 1024
    --max_prompt_tokens 1536 --min_prompt_tokens 512 --max_answer_tokens 128
    --eval_every 20 --val_batches 4
    --model_lr 3e-5 --seed_scale 1.0
    --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0
  )
  read -r _ b_acc _ < <(run_train punc_restore "$s" quick baseline train_punc_restore_sft.py no_fs "$Q_BUDGET" "$Q_STEPS" "${base[@]}")
  [[ "$b_acc" == "nan" ]] && return
  read -r _ q_acc _ < <(run_train punc_restore "$s" quick head_l8 train_punc_restore_sft.py prompt_fs "$Q_BUDGET" "$Q_STEPS" "${base[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0)
  [[ "$q_acc" == "nan" ]] && return
  d_acc=$(python3 - <<PY
print(float("$q_acc")-float("$b_acc"))
PY
)
  keep=$(python3 - <<PY
print(1 if float("$d_acc") >= float("$PRUNE_PUNC") else 0)
PY
)
  if [[ "$keep" == "1" ]]; then
    run_train punc_restore "$s" med head_l8 train_punc_restore_sft.py prompt_fs "$M_BUDGET" "$M_STEPS" "${base[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
  fi
}

{
  echo "Round27 started: $(date)"
  for s in $SEEDS; do
    run_mbpp_seed "$s"
    run_punc_seed "$s"
  done

  ./.venv/bin/python - <<'PY' | tee "$SUMMARY"
import json
from pathlib import Path

rows=[json.loads(x) for x in Path("runs/_round27_seedcheck_positive_s012_records.jsonl").read_text().splitlines() if x.strip()]

def summarize(task, fs_cfg):
    out=[]
    for seed in [0,1,2]:
        b=[r for r in rows if r.get("task")==task and r.get("seed")==seed and r.get("stage")=="quick" and r.get("config")=="baseline" and r.get("status")=="ok"]
        f=[r for r in rows if r.get("task")==task and r.get("seed")==seed and r.get("stage")=="quick" and r.get("config")==fs_cfg and r.get("status")=="ok"]
        if not b or not f:
            continue
        bacc=float(b[0]["best_val_tok_acc"]); facc=float(f[0]["best_val_tok_acc"])
        out.append((seed, bacc, facc, facc-bacc))
    return out

mb=summarize("mbpp_low","scalar_l8_trainable")
pc=summarize("punc_restore","head_l8")

print("="*116)
print("Round27 seedcheck summary (s0/s1/s2)")
print("="*116)
for name,data in [("mbpp_low + scalar_l8_trainable",mb),("punc_restore + head_l8",pc)]:
    print(f"[{name}]")
    if not data:
        print("  no complete quick pairs")
        print("-"*116)
        continue
    ds=[x[3] for x in data]
    print("  quick deltas:")
    for seed,b,f,d in data:
        print(f"    seed={seed}: baseline={b*100:.2f}% fs={f*100:.2f}% d_acc={d*100:+.2f}pp")
    mean=sum(ds)/len(ds)
    pos=sum(1 for x in ds if x>0)
    print(f"  mean d_acc={mean*100:+.2f}pp, positive_seeds={pos}/{len(ds)}")
    print("-"*116)
PY
  echo "Round27 finished: $(date)"
} 2>&1 | tee "$LOG"

