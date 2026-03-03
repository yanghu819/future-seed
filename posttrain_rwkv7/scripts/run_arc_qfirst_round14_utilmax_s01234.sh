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

L=512
SEEDS=(0 1 2 3 4)

ALPHA_INIT=-2
ALPHA_LR=0
FS_LAYER_START=6
FS_CLIP=1.0

for seed in "${SEEDS[@]}"; do
  for mode in no_fs prompt_fs; do
    echo "RUN arc q-first r14 utilmax L=$L seed=$seed mode=$mode"
    ./.venv/bin/python train_arc_mc_sft.py \
      --mode "$mode" \
      --seed "$seed" \
      --train_data_seed 0 \
      --val_data_seed 1234 \
      --q_first \
      --ds ai2_arc \
      --ds_cfg ARC-Challenge \
      --train_split train \
      --val_split validation \
      --n_train 1000 \
      --n_val 200 \
      --max_prompt_tokens "$L" \
      --max_answer_tokens 8 \
      --bsz 64 \
      --time_budget_sec 240 \
      --max_steps 2000 \
      --eval_every 50 \
      --val_batches 8 \
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

./.venv/bin/python summarize_arc_by_order.py \
  --q_first \
  --seeds "0,1,2,3,4" \
  --bsz 64 \
  --time_budget_sec 240 \
  --max_steps 2000 \
  --eval_every 50 \
  --val_batches 8 \
  --ds ai2_arc \
  --ds_cfg ARC-Challenge \
  --train_split train \
  --val_split validation \
  --alpha_init "$ALPHA_INIT" \
  --alpha_lr "$ALPHA_LR" \
  --model_lr 3e-5 \
  --fs_variant scalar \
  --seed_scale 1.0 \
  --n_train 1000 \
  --n_val 200 \
  --max_prompt_tokens "$L" \
  --max_answer_tokens 8 \
  --fs_layer_start "$FS_LAYER_START" \
  --fs_norm \
  --fs_detach \
  --fs_clip "$FS_CLIP" | tee runs/_summary_arc_qfirst_r14_utilmax_s01234.txt
