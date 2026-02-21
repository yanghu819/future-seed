#!/usr/bin/env bash
set -euo pipefail

# Round25: salvage punc_restore with memory-safe config.

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
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=8.9

SEED=0
BSZ=2
Q_BUDGET=120
Q_STEPS=180
M_BUDGET=360
M_STEPS=560
PRUNE=0.003

RUN_REC=runs/_round25_punc_salvage_records.jsonl
SUMMARY=runs/_summary_round25_punc_salvage_s0.txt
LOG=runs/_log_round25_punc_salvage_s0.$(date +%Y%m%d_%H%M%S).log
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
  local stage="$1"; shift
  local cfg="$1"; shift
  local mode="$1"; shift
  local budget="$1"; shift
  local steps="$1"; shift
  local log="runs/_round25_punc_${stage}_${cfg}.log"
  rm -f "$log"
  echo "RUN stage=$stage cfg=$cfg mode=$mode budget=$budget" >&2
  set +e
  ./.venv/bin/python "${PY_PREFIX}train_punc_restore_sft.py" "$@" --mode "$mode" --seed "$SEED" --bsz "$BSZ" \
    --time_budget_sec "$budget" --max_steps "$steps" > "$log" 2>&1
  local code=$?
  set -e
  if [[ $code -ne 0 ]]; then
    local err
    err=$(tail -n 2 "$log" | tr '\n' ' ' | sed 's/"/\\"/g')
    record "{\"stage\":\"$stage\",\"config\":\"$cfg\",\"status\":\"fail\",\"bsz\":$BSZ,\"best_val_loss\":null,\"best_val_tok_acc\":null,\"run_dir\":\"\",\"error\":\"$err\"}"
    echo -e "nan\tnan\t"
    return
  fi
  local run_dir
  run_dir=$(tail -n 1 "$log")
  read -r best_loss best_acc < <(metric_best "$run_dir")
  record "{\"stage\":\"$stage\",\"config\":\"$cfg\",\"status\":\"ok\",\"bsz\":$BSZ,\"best_val_loss\":$best_loss,\"best_val_tok_acc\":$best_acc,\"run_dir\":\"$run_dir\"}"
  echo -e "${best_loss}\t${best_acc}\t${run_dir}"
}

BASE_ARGS=(
  --train_data_seed 0 --val_data_seed 1234
  --ds hotpot_qa --ds_cfg distractor --train_split train --val_split validation
  --n_train 800 --n_val 160
  --min_chars 48 --max_chars 220
  --fill_notes_to_max --note_pool_size 1024
  --max_prompt_tokens 1536 --min_prompt_tokens 512 --max_answer_tokens 128
  --eval_every 20 --val_batches 4
  --model_lr 3e-5 --seed_scale 1.0
  --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0
)

{
  echo "Round25 started: $(date)"
  read -r _ b_acc _ < <(run_train quick baseline no_fs "$Q_BUDGET" "$Q_STEPS" "${BASE_ARGS[@]}")
  if [[ "$b_acc" != "nan" ]]; then
    while read -r cfg args; do
      [[ -z "${cfg:-}" ]] && continue
      read -r _ q_acc _ < <(run_train quick "$cfg" prompt_fs "$Q_BUDGET" "$Q_STEPS" "${BASE_ARGS[@]}" $args)
      [[ "$q_acc" == "nan" ]] && continue
      d_acc=$(python3 - <<PY
print(float("$q_acc")-float("$b_acc"))
PY
)
      keep=$(python3 - <<PY
print(1 if float("$d_acc") >= float("$PRUNE") else 0)
PY
)
      if [[ "$keep" != "1" ]]; then
        echo "PRUNE cfg=$cfg d_acc_pp=$(python3 - <<PY
print(float("$d_acc")*100.0)
PY
)%" >&2
        continue
      fi
      run_train med "$cfg" prompt_fs "$M_BUDGET" "$M_STEPS" "${BASE_ARGS[@]}" $args >/dev/null
    done <<'EOF'
scalar_l8_trainable --alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0
scalar_l8_sched_cos --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0
head_l8 --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0
EOF
  fi

  ./.venv/bin/python - <<'PY' | tee "$SUMMARY"
import json
from pathlib import Path
rows=[json.loads(x) for x in Path("runs/_round25_punc_salvage_records.jsonl").read_text().splitlines() if x.strip()]
print("="*106)
print("Round25 punc salvage summary")
print("="*106)
b=[r for r in rows if r.get("stage")=="quick" and r.get("config")=="baseline" and r.get("status")=="ok"]
if not b:
    print("baseline failed")
else:
    b=b[0]; bacc=float(b["best_val_tok_acc"])
    print(f"baseline quick: acc={bacc*100:.2f}% loss={float(b['best_val_loss']):.4f} bsz={b['bsz']}")
    q=[r for r in rows if r.get("stage")=="quick" and r.get("config")!="baseline" and r.get("status")=="ok"]
    for r in sorted(q,key=lambda x:float(x["best_val_tok_acc"])-bacc, reverse=True):
        da=float(r["best_val_tok_acc"])-bacc
        print(f"  {r['config']:24s} d_acc={da*100:+.2f}pp acc={float(r['best_val_tok_acc'])*100:.2f}%")
    m=[r for r in rows if r.get("stage")=="med" and r.get("status")=="ok"]
    if m:
        print("med confirmed:")
        for r in sorted(m,key=lambda x:float(x["best_val_tok_acc"]), reverse=True):
            da=float(r["best_val_tok_acc"])-bacc
            print(f"  {r['config']:24s} d_acc={da*100:+.2f}pp acc={float(r['best_val_tok_acc'])*100:.2f}% loss={float(r['best_val_loss']):.4f}")
PY
  echo "Round25 finished: $(date)"
} 2>&1 | tee "$LOG"

