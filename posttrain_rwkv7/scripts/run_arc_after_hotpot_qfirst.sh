#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/future-seed-posttrain

WAIT_PID_FILE=runs/run_hotpot_qfirst_after_current.pid
if [[ ! -f "$WAIT_PID_FILE" ]]; then
  echo "Missing $WAIT_PID_FILE" >&2
  exit 1
fi
WAIT_PID=$(cat "$WAIT_PID_FILE")

echo "Waiting for hotpot q_first pipeline pid=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done

echo "=== HOTPOT q_first pipeline DONE ==="

export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/torch_extensions
export HF_HOME=/root/autodl-tmp/hf
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_transformers
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=8.9

L=512
SEEDS=(0 1 2)

echo "=== ARC options-first sweep ==="
for seed in "${SEEDS[@]}"; do
  for mode in no_fs prompt_fs; do
    echo "RUN arc options-first L=$L seed=$seed mode=$mode"
    ./.venv/bin/python train_arc_mc_sft.py \
      --mode "$mode" \
      --seed "$seed" \
      --train_data_seed 0 \
      --val_data_seed 1234 \
      --ds ai2_arc \
      --ds_cfg ARC-Challenge \
      --train_split train \
      --val_split validation \
      --n_train 1000 \
      --n_val 200 \
      --max_prompt_tokens "$L" \
      --max_answer_tokens 8 \
      --bsz 8 \
      --time_budget_sec 240 \
      --eval_every 50 \
      --val_batches 16 \
      --model_lr 3e-5 \
      --alpha_lr 0 \
      --alpha_init -2 \
      --seed_scale 1.0 \
      --fs_variant scalar
  done
done

./summarize_arc_by_order.py \
  --seeds "0,1,2" \
  --bsz 8 \
  --time_budget_sec 240 \
  --max_steps 0 \
  --eval_every 50 \
  --val_batches 16 \
  --ds ai2_arc \
  --ds_cfg ARC-Challenge \
  --train_split train \
  --val_split validation \
  --alpha_init -2 \
  --alpha_lr 0 \
  --fs_variant scalar \
  --model_lr 3e-5 \
  --seed_scale 1.0 \
  --n_train 1000 \
  --n_val 200 \
  --max_prompt_tokens "$L" \
  --max_answer_tokens 8 | tee runs/_summary_arc_optionsfirst.txt

echo "=== ARC question-first sweep ==="
for seed in "${SEEDS[@]}"; do
  for mode in no_fs prompt_fs; do
    echo "RUN arc q_first L=$L seed=$seed mode=$mode"
    ./.venv/bin/python train_arc_mc_sft.py \
      --q_first \
      --mode "$mode" \
      --seed "$seed" \
      --train_data_seed 0 \
      --val_data_seed 1234 \
      --ds ai2_arc \
      --ds_cfg ARC-Challenge \
      --train_split train \
      --val_split validation \
      --n_train 1000 \
      --n_val 200 \
      --max_prompt_tokens "$L" \
      --max_answer_tokens 8 \
      --bsz 8 \
      --time_budget_sec 240 \
      --eval_every 50 \
      --val_batches 16 \
      --model_lr 3e-5 \
      --alpha_lr 0 \
      --alpha_init -2 \
      --seed_scale 1.0 \
      --fs_variant scalar
  done
done

./summarize_arc_by_order.py \
  --q_first \
  --seeds "0,1,2" \
  --bsz 8 \
  --time_budget_sec 240 \
  --max_steps 0 \
  --eval_every 50 \
  --val_batches 16 \
  --ds ai2_arc \
  --ds_cfg ARC-Challenge \
  --train_split train \
  --val_split validation \
  --alpha_init -2 \
  --alpha_lr 0 \
  --fs_variant scalar \
  --model_lr 3e-5 \
  --seed_scale 1.0 \
  --n_train 1000 \
  --n_val 200 \
  --max_prompt_tokens "$L" \
  --max_answer_tokens 8 | tee runs/_summary_arc_qfirst.txt

echo "DONE"

