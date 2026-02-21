#!/usr/bin/env bash
set -euo pipefail

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
RUN_REC=runs/_round16_fastscreen_records.jsonl
: > "$RUN_REC"

COMMON=(
  --ds hotpot_qa --ds_cfg distractor --train_split train --val_split validation
  --n_train 600 --n_val 120
  --max_prompt_tokens 4096 --min_prompt_tokens 1536 --max_answer_tokens 24
  --bsz 2 --time_budget_sec 300 --max_steps 300 --eval_every 25 --val_batches 6
  --model_lr 3e-5 --seed_scale 1.0
  --fs_norm --fs_detach --fs_clip 1.0
)

# Baseline no_fs once per seed.
for seed in "${SEEDS[@]}"; do
  echo "RUN round16 baseline no_fs seed=$seed"
  RUN_DIR=$(./.venv/bin/python train_hotpot_longctx_sft.py \
    --mode no_fs --seed "$seed" --train_data_seed 0 --val_data_seed 1234 \
    --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 \
    "${COMMON[@]}" | tail -n 1)

  ./.venv/bin/python - <<PY >> "$RUN_REC"
import json
from pathlib import Path
p=Path("$RUN_DIR")/"metrics.jsonl"
best_loss=None; best_acc=None
for line in p.read_text().splitlines():
    r=json.loads(line)
    if 'val_loss' in r:
        v=float(r['val_loss']); best_loss=v if best_loss is None or v<best_loss else best_loss
    if 'val_tok_acc' in r:
        v=float(r['val_tok_acc']); best_acc=v if best_acc is None or v>best_acc else best_acc
print(json.dumps({"config":"baseline_no_fs","seed":$seed,"mode":"no_fs","run_dir":"$RUN_DIR","best_val_loss":best_loss,"best_val_tok_acc":best_acc}))
PY
done

run_cfg () {
  local cfg_name="$1"; shift
  for seed in "${SEEDS[@]}"; do
    echo "RUN round16 cfg=${cfg_name} seed=${seed}"
    RUN_DIR=$(./.venv/bin/python train_hotpot_longctx_sft.py \
      --mode prompt_fs --seed "$seed" --train_data_seed 0 --val_data_seed 1234 \
      "${COMMON[@]}" "$@" | tail -n 1)

    ./.venv/bin/python - <<PY >> "$RUN_REC"
import json
from pathlib import Path
p=Path("$RUN_DIR")/"metrics.jsonl"
best_loss=None; best_acc=None
for line in p.read_text().splitlines():
    r=json.loads(line)
    if 'val_loss' in r:
        v=float(r['val_loss']); best_loss=v if best_loss is None or v<best_loss else best_loss
    if 'val_tok_acc' in r:
        v=float(r['val_tok_acc']); best_acc=v if best_acc is None or v>best_acc else best_acc
print(json.dumps({"config":"$cfg_name","seed":$seed,"mode":"prompt_fs","run_dir":"$RUN_DIR","best_val_loss":best_loss,"best_val_tok_acc":best_acc}))
PY
  done
}

run_cfg scalar_l10_a3 --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10
run_cfg scalar_l8_a3 --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 8
run_cfg scalar_l10_sched --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 --fs_alpha_schedule linear --fs_alpha_min 0.5 --fs_alpha_max 1.0
run_cfg head_l10 --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 10
run_cfg inputgate_l10 --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 --fs_in_gate --fs_w_lr 5e-4 --fs_b_lr 5e-4

./.venv/bin/python - <<'PY' | tee runs/_summary_hotpot_round16_fastscreen_s01.txt
import json
from pathlib import Path
import statistics as st

p = Path('runs/_round16_fastscreen_records.jsonl')
rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
base = {r['seed']: r for r in rows if r['config'] == 'baseline_no_fs'}
by = {}
for r in rows:
    if r['config'] == 'baseline_no_fs':
        continue
    b = base[r['seed']]
    d_acc = r['best_val_tok_acc'] - b['best_val_tok_acc']
    d_loss = r['best_val_loss'] - b['best_val_loss']
    by.setdefault(r['config'], []).append((r['seed'], d_acc, d_loss, b['best_val_tok_acc'], r['best_val_tok_acc']))

for cfg, vals in by.items():
    dacc = [x[1] for x in vals]
    dloss = [x[2] for x in vals]
    pos = sum(1 for x in dacc if x > 0)
    neg = sum(1 for x in dacc if x < 0)
    print("=" * 100)
    print(
        f"{cfg}: pairs={len(vals)} mean d_acc={st.mean(dacc):+.4f} std={st.pstdev(dacc):.4f} | "
        f"mean d_loss={st.mean(dloss):+.4f} std={st.pstdev(dloss):.4f} | sign(+/0/-)={pos}/{len(vals)-pos-neg}/{neg}"
    )
    for seed, da, dl, bacc, facc in vals:
        print(f"  seed={seed} d_acc={da:+.4f} (no={bacc:.4f} fs={facc:.4f}) d_loss={dl:+.4f}")
PY
