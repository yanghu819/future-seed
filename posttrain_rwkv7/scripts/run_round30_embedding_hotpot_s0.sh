#!/usr/bin/env bash
set -euo pipefail

# Round30: embedding smoke test (baseline vs FS) on Hotpot retrieval pairs.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
  cd "$SCRIPT_DIR/.."
else
  cd "$SCRIPT_DIR"
fi

if [[ -f scripts/train_embedding_hotpot_fs.py ]]; then
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
BSZ=4
MAX_STEPS=220
TIME_BUDGET=1200

RUN_REC=runs/_round30_embedding_hotpot_s0_records.jsonl
SUMMARY=runs/_summary_round30_embedding_hotpot_s0.txt
LOG=runs/_log_round30_embedding_hotpot_s0.$(date +%Y%m%d_%H%M%S).log
: > "$RUN_REC"

record() { printf "%s\n" "$1" >> "$RUN_REC"; }

run_one() {
  local mode="$1"; shift
  local extra="$*"
  local log="runs/_round30_embed_${mode}.log"
  rm -f "$log"
  echo "RUN mode=$mode" >&2
  set +e
  ./.venv/bin/python "${PY_PREFIX}train_embedding_hotpot_fs.py" \
    --mode "$mode" --seed "$SEED" \
    --train_data_seed 0 --val_data_seed 1234 \
    --ds hotpot_qa --ds_cfg distractor --train_split train --val_split validation \
    --n_train 1024 --n_val 256 \
    --max_q_tokens 128 --max_doc_tokens 1024 --min_doc_tokens 256 \
    --bsz "$BSZ" --max_steps "$MAX_STEPS" --time_budget_sec "$TIME_BUDGET" \
    --eval_every 20 --emb_dim 256 --lr 1e-3 --weight_decay 0.01 --temp 0.07 \
    --alpha_lr 5e-4 --alpha_init -2 --fs_layer_start 8 --fs_norm --fs_detach --fs_clip 1.0 \
    $extra > "$log" 2>&1
  local code=$?
  set -e

  if [[ $code -ne 0 ]]; then
    local err
    err=$(tail -n 4 "$log" | tr '\n' ' ' | sed 's/"/\\"/g')
    record "{\"mode\":\"$mode\",\"status\":\"fail\",\"run_dir\":\"\",\"val_r1\":null,\"val_r5\":null,\"val_mrr10\":null,\"error\":\"$err\"}"
    return
  fi

  local run_dir
  run_dir=$(tail -n 1 "$log")
  read -r r1 r5 mrr < <(python3 - <<PY
import json
from pathlib import Path
p=Path("$run_dir")/"summary.json"
obj=json.loads(p.read_text())
b=obj.get("best",{})
print(b.get("val_r1",0.0), b.get("val_r5",0.0), b.get("val_mrr10",0.0))
PY
)
  record "{\"mode\":\"$mode\",\"status\":\"ok\",\"run_dir\":\"$run_dir\",\"val_r1\":$r1,\"val_r5\":$r5,\"val_mrr10\":$mrr}"
}

{
  echo "Round30 embedding started: $(date)"
  run_one baseline
  run_one fs

  python3 - <<'PY' | tee "$SUMMARY"
import json
from pathlib import Path
rows=[json.loads(x) for x in Path("runs/_round30_embedding_hotpot_s0_records.jsonl").read_text().splitlines() if x.strip()]
print("="*116)
print("Round30 embedding summary (Hotpot retrieval)")
print("="*116)
by={r["mode"]:r for r in rows}
for m in ["baseline","fs"]:
    r=by.get(m)
    if not r:
        print(f"[{m}] missing")
        continue
    if r["status"]!="ok":
        print(f"[{m}] failed")
        continue
    print(f"[{m}] R@1={float(r['val_r1'])*100:.2f}% R@5={float(r['val_r5'])*100:.2f}% MRR@10={float(r['val_mrr10'])*100:.2f}%")
if by.get("baseline",{}).get("status")=="ok" and by.get("fs",{}).get("status")=="ok":
    db=float(by["fs"]["val_r1"]) - float(by["baseline"]["val_r1"])
    dm=float(by["fs"]["val_mrr10"]) - float(by["baseline"]["val_mrr10"])
    print("-"*116)
    print(f"delta FS-baseline: d_R@1={db*100:+.2f}pp d_MRR@10={dm*100:+.2f}pp")
PY
  echo "Round30 embedding finished: $(date)"
} 2>&1 | tee "$LOG"
