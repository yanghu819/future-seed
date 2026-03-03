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

L=4096
SEEDS=(0 1 2)

ALPHA_INIT=-2
ALPHA_LR=0
FS_LAYER_START=6
FS_CLIP=1.0

TIME_BUDGET=360
N_TRAIN=800
N_VAL=200
MIN_PROMPT=1536

for seed in "${SEEDS[@]}"; do
  for mode in no_fs prompt_fs; do
    echo "RUN mbpp q_after L=$L seed=$seed mode=$mode"
    ./.venv/bin/python train_mbpp_longctx_sft.py \
      --mode "$mode" \
      --seed "$seed" \
      --train_data_seed 0 \
      --val_data_seed 1234 \
      --ds mbpp \
      --train_split train \
      --val_split test \
      --fill_notes_to_max \
      --note_pool_size 2048 \
      --n_train "$N_TRAIN" \
      --n_val "$N_VAL" \
      --max_prompt_tokens "$L" \
      --min_prompt_tokens "$MIN_PROMPT" \
      --max_answer_tokens 160 \
      --bsz 1 \
      --time_budget_sec "$TIME_BUDGET" \
      --eval_every 25 \
      --val_batches 16 \
      --model_lr 3e-5 \
      --alpha_lr "$ALPHA_LR" \
      --alpha_init "$ALPHA_INIT" \
      --seed_scale 1.0 \
      --fs_variant scalar \
      --fs_layer_start "$FS_LAYER_START" \
      --fs_norm \
      --fs_detach \
      --fs_clip "$FS_CLIP"
  done
done

./.venv/bin/python summarize_mbpp_by_len.py \
  --seeds "0,1,2" \
  --bsz 1 \
  --time_budget_sec "$TIME_BUDGET" \
  --max_steps 0 \
  --eval_every 25 \
  --val_batches 16 \
  --ds mbpp \
  --train_split train \
  --val_split test \
  --fill_notes_to_max \
  --note_pool_size 2048 \
  --alpha_init "$ALPHA_INIT" \
  --alpha_lr "$ALPHA_LR" \
  --model_lr 3e-5 \
  --fs_variant scalar \
  --seed_scale 1.0 \
  --n_train "$N_TRAIN" \
  --n_val "$N_VAL" \
  --max_prompt_tokens "$L" \
  --min_prompt_tokens "$MIN_PROMPT" \
  --max_answer_tokens 160 \
  --fs_layer_start "$FS_LAYER_START" \
  --fs_norm \
  --fs_detach \
  --fs_clip "$FS_CLIP" | tee runs/_summary_mbpp_qafter_stabilized_len4096_r1_s012.txt

