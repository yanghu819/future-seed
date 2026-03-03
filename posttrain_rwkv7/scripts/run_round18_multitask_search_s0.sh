#!/usr/bin/env bash
set -euo pipefail

# Breadth-first algorithm/task search with aggressive pruning.
# Single seed, short budget, many tasks/configs.

cd /root/autodl-tmp/future-seed-posttrain

export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/torch_extensions
export HF_HOME=/root/autodl-tmp/hf
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_transformers
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=8.9

SEED=0
RUN_REC=runs/_round18_multitask_search_s0_records.jsonl
SUMMARY=runs/_summary_round18_multitask_search_s0.txt
: > "$RUN_REC"

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

run_one () {
  local task="$1"; shift
  local cfg="$1"; shift
  local mode="$1"; shift
  local trainer="$1"; shift

  echo "RUN task=$task cfg=$cfg mode=$mode"
  local run_dir
  run_dir=$(./.venv/bin/python "$trainer" "$@" | tail -n 1)
  read -r best_loss best_acc < <(metric_best "$run_dir")

  ./.venv/bin/python - <<PY >> "$RUN_REC"
import json
print(json.dumps({
  "task":"$task",
  "config":"$cfg",
  "seed":$SEED,
  "mode":"$mode",
  "run_dir":"$run_dir",
  "best_val_loss":float("$best_loss"),
  "best_val_tok_acc":float("$best_acc")
}))
PY
}

task_hotpot () {
  local task=hotpot
  local BASE=(
    --seed "$SEED" --train_data_seed 0 --val_data_seed 1234
    --ds hotpot_qa --ds_cfg distractor --train_split train --val_split validation
    --n_train 500 --n_val 120
    --max_prompt_tokens 4096 --min_prompt_tokens 1536 --max_answer_tokens 24
    --bsz 2 --time_budget_sec 150 --max_steps 240 --eval_every 25 --val_batches 6
    --model_lr 3e-5 --seed_scale 1.0
    --fs_norm --fs_detach --fs_clip 1.0
  )
  run_one "$task" baseline no_fs train_hotpot_longctx_sft.py \
    --mode no_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10
  run_one "$task" scalar_l10_a3 prompt_fs train_hotpot_longctx_sft.py \
    --mode prompt_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10
  run_one "$task" scalar_l12_a3 prompt_fs train_hotpot_longctx_sft.py \
    --mode prompt_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 12
  run_one "$task" head_l10 prompt_fs train_hotpot_longctx_sft.py \
    --mode prompt_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 10
  run_one "$task" scalar_l10_sched_cos prompt_fs train_hotpot_longctx_sft.py \
    --mode prompt_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0
  run_one "$task" inputgate_l10 prompt_fs train_hotpot_longctx_sft.py \
    --mode prompt_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 10 --fs_in_gate --fs_w_lr 5e-4 --fs_b_lr 5e-4
}

task_mbpp () {
  local task=mbpp
  local BASE=(
    --seed "$SEED" --train_data_seed 0 --val_data_seed 1234
    --ds mbpp --train_split train --val_split test
    --fill_notes_to_max --note_pool_size 2048
    --n_train 700 --n_val 160
    --max_prompt_tokens 4096 --min_prompt_tokens 1536 --max_answer_tokens 160
    --bsz 1 --time_budget_sec 150 --max_steps 220 --eval_every 25 --val_batches 6
    --model_lr 3e-5 --seed_scale 1.0
    --fs_norm --fs_detach --fs_clip 1.0
  )
  run_one "$task" baseline no_fs train_mbpp_longctx_sft.py \
    --mode no_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6
  run_one "$task" scalar_l6_a2 prompt_fs train_mbpp_longctx_sft.py \
    --mode prompt_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6
  run_one "$task" scalar_l8_a3 prompt_fs train_mbpp_longctx_sft.py \
    --mode prompt_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 8
  run_one "$task" head_l6 prompt_fs train_mbpp_longctx_sft.py \
    --mode prompt_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 6
  run_one "$task" scalar_l6_sched_cos prompt_fs train_mbpp_longctx_sft.py \
    --mode prompt_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0
  run_one "$task" scalar_l6_trainable prompt_fs train_mbpp_longctx_sft.py \
    --mode prompt_fs "${BASE[@]}" --alpha_lr 5e-4 --alpha_init -2 --fs_variant scalar --fs_layer_start 6
}

task_protein_ss () {
  local task=protein_ss
  local BASE=(
    --seed "$SEED" --train_data_seed 0 --val_data_seed 1234
    --ds lamm-mit/protein_secondary_structure_from_PDB --split train
    --n_train 1200 --n_val 240
    --max_seq_len 512 --min_seq_len 96 --num_queries 48 --query_region random
    --fill_notes_to_max --note_pool_size 2048 --max_note_seq_len 256
    --max_prompt_tokens 2048 --min_prompt_tokens 1024 --max_answer_tokens 128
    --bsz 2 --time_budget_sec 140 --max_steps 220 --eval_every 25 --val_batches 8
    --model_lr 3e-5 --seed_scale 1.0
    --fs_norm --fs_detach --fs_clip 1.0
  )
  run_one "$task" baseline no_fs train_protein_ss_spot_sft.py \
    --mode no_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6
  run_one "$task" scalar_l6_a2 prompt_fs train_protein_ss_spot_sft.py \
    --mode prompt_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6
  run_one "$task" scalar_l8_a3 prompt_fs train_protein_ss_spot_sft.py \
    --mode prompt_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 8
  run_one "$task" head_l6 prompt_fs train_protein_ss_spot_sft.py \
    --mode prompt_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 6
  run_one "$task" scalar_l6_sched_cos prompt_fs train_protein_ss_spot_sft.py \
    --mode prompt_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0
}

task_protein_contact () {
  local task=protein_contact
  local BASE=(
    --seed "$SEED" --train_data_seed 0 --val_data_seed 1234
    --ds heya5/protein_contact_map --train_split train --val_split valid
    --n_train 1400 --n_val 180
    --max_seq_len 384 --min_seq_len 96 --num_pairs 32 --min_sep 8 --contact_cutoff 8.0
    --fill_notes_to_max --note_pool_size 2048 --max_note_seq_len 256
    --max_prompt_tokens 2048 --min_prompt_tokens 1024 --max_answer_tokens 128
    --bsz 2 --time_budget_sec 140 --max_steps 220 --eval_every 25 --val_batches 8
    --model_lr 3e-5 --seed_scale 1.0
    --fs_norm --fs_detach --fs_clip 1.0
  )
  run_one "$task" baseline no_fs train_protein_contact_pair_sft.py \
    --mode no_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6
  run_one "$task" scalar_l6_a2 prompt_fs train_protein_contact_pair_sft.py \
    --mode prompt_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6
  run_one "$task" scalar_l8_a3 prompt_fs train_protein_contact_pair_sft.py \
    --mode prompt_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant scalar --fs_layer_start 8
  run_one "$task" head_l6 prompt_fs train_protein_contact_pair_sft.py \
    --mode prompt_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -3 --fs_variant head --alpha_head_init -3 --alpha_head_lr 5e-4 --fs_layer_start 6
  run_one "$task" scalar_l6_sched_cos prompt_fs train_protein_contact_pair_sft.py \
    --mode prompt_fs "${BASE[@]}" --alpha_lr 0 --alpha_init -2 --fs_variant scalar --fs_layer_start 6 --fs_alpha_schedule cosine --fs_alpha_min 0.4 --fs_alpha_max 1.0
}

task_hotpot
task_mbpp
task_protein_ss
task_protein_contact

./.venv/bin/python - <<'PY' | tee "$SUMMARY"
import json
from pathlib import Path
from collections import defaultdict

p = Path("runs/_round18_multitask_search_s0_records.jsonl")
rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

base = {}
for r in rows:
    if r["config"] == "baseline" and r["mode"] == "no_fs":
        base[r["task"]] = r

by = defaultdict(list)
for r in rows:
    if r["config"] == "baseline":
        continue
    b = base[r["task"]]
    d_acc = float(r["best_val_tok_acc"]) - float(b["best_val_tok_acc"])
    d_loss = float(r["best_val_loss"]) - float(b["best_val_loss"])
    by[r["task"]].append((r["config"], d_acc, d_loss, r["best_val_tok_acc"], b["best_val_tok_acc"]))

print("=" * 100)
print("Round18 multitask search (seed=0, short budget)")
print("=" * 100)
for task, vals in sorted(by.items()):
    print(f"[{task}] baseline acc={base[task]['best_val_tok_acc']:.4f} loss={base[task]['best_val_loss']:.4f}")
    vals = sorted(vals, key=lambda x: x[1], reverse=True)
    for cfg, da, dl, acc, bacc in vals:
        mark = "KEEP" if da >= 0.003 else ("MAYBE" if da >= 0.0 else "PRUNE")
        print(f"  {cfg:24s} d_acc={da:+.4f} d_loss={dl:+.4f} fs_acc={acc:.4f} ({mark})")
    print("-" * 100)
print("Next: promote KEEP configs to seeds 0..4 with longer budget.")
PY

