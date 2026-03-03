#!/usr/bin/env bash
set -euo pipefail

# Quick prune-first runner:
# 1) run no_fs baseline on seeds {0,1}
# 2) run seed=0 for each FS config, prune immediately if d_acc is negative enough
# 3) run seed=1 only for survivors
# 4) rank configs by mean delta and emit top candidates for full 5-seed run

cd /root/autodl-tmp/future-seed-posttrain

export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/torch_extensions
export HF_HOME=/root/autodl-tmp/hf
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_transformers
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=8.9

SEEDS=(0 1)
RUN_REC=runs/_round17_prune_auto_records.jsonl
SUMMARY=runs/_summary_hotpot_round17_prune_auto_s01.txt
TOPK=2
PRUNE_SEED0_DACC=-0.002
PROMOTE_MEAN_DACC=0.003

: > "$RUN_REC"

COMMON=(
  --ds hotpot_qa --ds_cfg distractor --train_split train --val_split validation
  --n_train 600 --n_val 120
  --max_prompt_tokens 4096 --min_prompt_tokens 1536 --max_answer_tokens 24
  --bsz 2 --time_budget_sec 300 --max_steps 300 --eval_every 25 --val_batches 6
  --model_lr 3e-5 --seed_scale 1.0
  --fs_norm --fs_detach --fs_clip 1.0
)

metric_best () {
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
    if 'val_loss' in r:
        v=float(r['val_loss']); best_loss=v if best_loss is None or v<best_loss else best_loss
    if 'val_tok_acc' in r:
        v=float(r['val_tok_acc']); best_acc=v if best_acc is None or v>best_acc else best_acc
print(f"{best_loss}\t{best_acc}")
PY
}

run_no_fs_seed () {
  local seed="$1"
  local run_dir
  run_dir=$(./.venv/bin/python train_hotpot_longctx_sft.py \
    --mode no_fs --seed "$seed" --train_data_seed 0 --val_data_seed 1234 \
    --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 \
    "${COMMON[@]}" | tail -n 1)
  read -r best_loss best_acc < <(metric_best "$run_dir")
  ./.venv/bin/python - <<PY >> "$RUN_REC"
import json
print(json.dumps({
  "config":"baseline_no_fs",
  "seed":$seed,
  "mode":"no_fs",
  "run_dir":"$run_dir",
  "best_val_loss":float("$best_loss"),
  "best_val_tok_acc":float("$best_acc")
}))
PY
}

echo "RUN round17: baselines"
for seed in "${SEEDS[@]}"; do
  echo "  baseline no_fs seed=$seed"
  run_no_fs_seed "$seed"
done

baseline_acc_seed0=$(
  ./.venv/bin/python - <<PY
import json
from pathlib import Path
rows=[json.loads(x) for x in Path("$RUN_REC").read_text().splitlines() if x.strip()]
for r in rows:
    if r["config"]=="baseline_no_fs" and r["seed"]==0:
        print(r["best_val_tok_acc"]); break
PY
)
baseline_acc_seed1=$(
  ./.venv/bin/python - <<PY
import json
from pathlib import Path
rows=[json.loads(x) for x in Path("$RUN_REC").read_text().splitlines() if x.strip()]
for r in rows:
    if r["config"]=="baseline_no_fs" and r["seed"]==1:
        print(r["best_val_tok_acc"]); break
PY
)

run_fs_seed () {
  local cfg="$1"; shift
  local seed="$1"; shift
  local run_dir
  run_dir=$(./.venv/bin/python train_hotpot_longctx_sft.py \
    --mode prompt_fs --seed "$seed" --train_data_seed 0 --val_data_seed 1234 \
    "${COMMON[@]}" "$@" | tail -n 1)
  read -r best_loss best_acc < <(metric_best "$run_dir")
  local base_acc
  if [[ "$seed" == "0" ]]; then
    base_acc="$baseline_acc_seed0"
  else
    base_acc="$baseline_acc_seed1"
  fi
  local d_acc
  d_acc=$(./.venv/bin/python - <<PY
print(float("$best_acc") - float("$base_acc"))
PY
)
  ./.venv/bin/python - <<PY >> "$RUN_REC"
import json
print(json.dumps({
  "config":"$cfg",
  "seed":$seed,
  "mode":"prompt_fs",
  "run_dir":"$run_dir",
  "best_val_loss":float("$best_loss"),
  "best_val_tok_acc":float("$best_acc"),
  "baseline_acc":float("$base_acc"),
  "d_acc":float("$d_acc")
}))
PY
  echo "$d_acc"
}

run_cfg_pruned () {
  local cfg="$1"; shift
  echo "RUN round17 cfg=$cfg seed=0"
  d0=$(run_fs_seed "$cfg" 0 "$@")
  echo "  seed0 d_acc=$d0"
  keep=$(
    ./.venv/bin/python - <<PY
d=float("$d0")
print(1 if d >= float("$PRUNE_SEED0_DACC") else 0)
PY
  )
  if [[ "$keep" != "1" ]]; then
    echo "  PRUNED after seed0 (d_acc < $PRUNE_SEED0_DACC)"
    return
  fi
  echo "RUN round17 cfg=$cfg seed=1"
  d1=$(run_fs_seed "$cfg" 1 "$@")
  echo "  seed1 d_acc=$d1"
}

# New + old candidates for quick screen.
run_cfg_pruned scalar_l10_a3 \
  --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10

run_cfg_pruned scalar_l12_a3 \
  --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 12

run_cfg_pruned scalar_l10_a2 \
  --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 10

run_cfg_pruned scalar_l10_sched_cos \
  --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 \
  --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0

run_cfg_pruned head_l10 \
  --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 10

run_cfg_pruned inputgate_l10 \
  --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 --fs_in_gate --fs_w_lr 5e-4 --fs_b_lr 5e-4

./.venv/bin/python - <<'PY' | tee "$SUMMARY"
import json
from pathlib import Path
import statistics as st

RUN_REC=Path("runs/_round17_prune_auto_records.jsonl")
TOPK=2
PROMOTE_MEAN_DACC=0.003

rows=[json.loads(x) for x in RUN_REC.read_text().splitlines() if x.strip()]
base={r["seed"]:r for r in rows if r["config"]=="baseline_no_fs"}
cfg_rows={}
for r in rows:
    if r["config"]=="baseline_no_fs":
        continue
    cfg_rows.setdefault(r["config"], []).append(r)

print("="*100)
print("Round17 quick-prune summary (Hotpot QA, seeds 0/1)")
print("="*100)
promote=[]
for cfg, rs in sorted(cfg_rows.items()):
    dacc=[]
    dloss=[]
    for r in rs:
        b=base[r["seed"]]
        da=r.get("d_acc", float(r["best_val_tok_acc"])-float(b["best_val_tok_acc"]))
        dl=float(r["best_val_loss"])-float(b["best_val_loss"])
        dacc.append(float(da))
        dloss.append(float(dl))
    mean_da=st.mean(dacc)
    std_da=st.pstdev(dacc) if len(dacc)>1 else 0.0
    mean_dl=st.mean(dloss)
    std_dl=st.pstdev(dloss) if len(dloss)>1 else 0.0
    print(f"{cfg:20s} seeds={len(rs)} mean d_acc={mean_da:+.4f} std={std_da:.4f} | mean d_loss={mean_dl:+.4f} std={std_dl:.4f}")
    if mean_da >= PROMOTE_MEAN_DACC and len(rs) >= 2:
        promote.append((cfg, mean_da))

promote=sorted(promote, key=lambda x: x[1], reverse=True)[:TOPK]
print("-"*100)
if not promote:
    print(f"PROMOTE: none (threshold mean d_acc >= {PROMOTE_MEAN_DACC:+.4f})")
else:
    print("PROMOTE TOP:")
    for i,(cfg,score) in enumerate(promote,1):
        print(f"  {i}. {cfg}  mean d_acc={score:+.4f}")
print("-"*100)
print("Next: run promoted configs on seeds 0..4 with longer budget.")
PY

