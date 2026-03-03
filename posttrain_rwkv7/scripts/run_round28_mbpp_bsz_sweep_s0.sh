#!/usr/bin/env bash
set -euo pipefail

# Round28: isolate MBPP throughput sensitivity by sweeping batch size.

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

SEED=0
BSZ_LIST="2 4 6 8"
Q_BUDGET=90
Q_STEPS=140
M_BUDGET=280
M_STEPS=420
PRUNE=0.002

RUN_REC=runs/_round28_mbpp_bsz_sweep_s0_records.jsonl
SUMMARY=runs/_summary_round28_mbpp_bsz_sweep_s0.txt
LOG=runs/_log_round28_mbpp_bsz_sweep_s0.$(date +%Y%m%d_%H%M%S).log
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
  local bsz="$1"; shift
  local stage="$1"; shift
  local cfg="$1"; shift
  local mode="$1"; shift
  local budget="$1"; shift
  local steps="$1"; shift
  local log="runs/_round28_mbpp_bsz${bsz}_${stage}_${cfg}.log"
  rm -f "$log"
  echo "RUN bsz=$bsz stage=$stage cfg=$cfg mode=$mode budget=$budget" >&2
  set +e
  ./.venv/bin/python "${PY_PREFIX}train_mbpp_longctx_sft.py" \
    --train_data_seed "$SEED" --val_data_seed 1234 \
    --ds mbpp --train_split train --val_split test \
    --fill_notes_to_max --note_pool_size 1024 \
    --n_train 320 --n_val 80 \
    --max_prompt_tokens 3072 --min_prompt_tokens 768 --max_answer_tokens 160 \
    --eval_every 20 --val_batches 4 \
    --model_lr 3e-5 --seed_scale 1.0 \
    --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 \
    "$@" --mode "$mode" --seed "$SEED" --bsz "$bsz" \
    --time_budget_sec "$budget" --max_steps "$steps" > "$log" 2>&1
  local code=$?
  set -e
  if [[ $code -ne 0 ]]; then
    local err
    err=$(tail -n 2 "$log" | tr '\n' ' ' | sed 's/"/\\"/g')
    record "{\"task\":\"mbpp\",\"bsz\":$bsz,\"stage\":\"$stage\",\"config\":\"$cfg\",\"status\":\"fail\",\"best_val_loss\":null,\"best_val_tok_acc\":null,\"run_dir\":\"\",\"error\":\"$err\"}"
    echo -e "nan\tnan\t"
    return
  fi
  local run_dir
  run_dir=$(tail -n 1 "$log")
  read -r best_loss best_acc < <(metric_best "$run_dir")
  record "{\"task\":\"mbpp\",\"bsz\":$bsz,\"stage\":\"$stage\",\"config\":\"$cfg\",\"status\":\"ok\",\"best_val_loss\":$best_loss,\"best_val_tok_acc\":$best_acc,\"run_dir\":\"$run_dir\"}"
  echo -e "${best_loss}\t${best_acc}\t${run_dir}"
}

{
  echo "Round28 started: $(date)"
  for b in $BSZ_LIST; do
    read -r _ b_acc _ < <(run_train "$b" quick baseline no_fs "$Q_BUDGET" "$Q_STEPS")
    if [[ "$b_acc" == "nan" ]]; then
      continue
    fi
    read -r _ q_acc _ < <(run_train "$b" quick scalar_l8_trainable prompt_fs "$Q_BUDGET" "$Q_STEPS" --alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0)
    if [[ "$q_acc" == "nan" ]]; then
      continue
    fi
    d_acc=$(python3 - <<PY
print(float("$q_acc")-float("$b_acc"))
PY
)
    keep=$(python3 - <<PY
print(1 if float("$d_acc") >= float("$PRUNE") else 0)
PY
)
    if [[ "$keep" == "1" ]]; then
      run_train "$b" med scalar_l8_trainable prompt_fs "$M_BUDGET" "$M_STEPS" --alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 >/dev/null
    fi
  done

  ./.venv/bin/python - <<'PY' | tee "$SUMMARY"
import json
from pathlib import Path
rows=[json.loads(x) for x in Path("runs/_round28_mbpp_bsz_sweep_s0_records.jsonl").read_text().splitlines() if x.strip()]
print("="*116)
print("Round28 MBPP throughput sweep summary")
print("="*116)
for b in sorted({r.get("bsz") for r in rows if "bsz" in r}):
    print(f"[bsz={b}]")
    base=[r for r in rows if r.get("bsz")==b and r.get("stage")=="quick" and r.get("config")=="baseline" and r.get("status")=="ok"]
    fs=[r for r in rows if r.get("bsz")==b and r.get("stage")=="quick" and r.get("config")=="scalar_l8_trainable" and r.get("status")=="ok"]
    if not base:
      print("  baseline failed")
      print("-"*116)
      continue
    bacc=float(base[0]["best_val_tok_acc"])
    bloss=float(base[0]["best_val_loss"])
    print(f"  baseline quick: acc={bacc*100:.2f}% loss={bloss:.4f}")
    if fs:
      facc=float(fs[0]["best_val_tok_acc"])
      print(f"  fs quick: acc={facc*100:.2f}% d_acc={(facc-bacc)*100:+.2f}pp")
    else:
      print("  fs quick: failed")
    med=[r for r in rows if r.get("bsz")==b and r.get("stage")=="med" and r.get("config")=="scalar_l8_trainable" and r.get("status")=="ok"]
    if med:
      macc=float(med[0]["best_val_tok_acc"])
      mloss=float(med[0]["best_val_loss"])
      print(f"  fs med: acc={macc*100:.2f}% d_acc={(macc-bacc)*100:+.2f}pp loss={mloss:.4f}")
    print("-"*116)
PY
  echo "Round28 finished: $(date)"
} 2>&1 | tee "$LOG"
