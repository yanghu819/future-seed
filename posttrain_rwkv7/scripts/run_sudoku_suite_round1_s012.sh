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

SEEDS=(0 1 2)
TIME_BUDGET=180

ALPHA_INIT=-2
ALPHA_LR=0
FS_LAYER_START=6
FS_CLIP=1.0

# (base, mask_count, region, tag)
SETTINGS=(
  "2 8 prefix sudoku4_prefix"
  "2 8 suffix sudoku4_suffix"
  "3 40 prefix sudoku9_prefix"
  "3 40 suffix sudoku9_suffix"
)

for st in "${SETTINGS[@]}"; do
  read -r BASE MASK_COUNT REGION TAG <<< "$st"

  for seed in "${SEEDS[@]}"; do
    for mode in no_fs prompt_fs; do
      echo "RUN $TAG seed=$seed mode=$mode"
      ./.venv/bin/python train_sudoku_sft.py \
        --mode "$mode" \
        --seed "$seed" \
        --val_seed 1234 \
        --base "$BASE" \
        --mask_count "$MASK_COUNT" \
        --mask_region "$REGION" \
        --bsz 8 \
        --time_budget_sec "$TIME_BUDGET" \
        --eval_every 50 \
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

  ./.venv/bin/python summarize_sudoku_by_setting.py \
    --seeds "0,1,2" \
    --bsz 8 \
    --time_budget_sec "$TIME_BUDGET" \
    --max_steps 0 \
    --eval_every 50 \
    --val_batches 16 \
    --base "$BASE" \
    --mask_count "$MASK_COUNT" \
    --mask_region "$REGION" \
    --alpha_init "$ALPHA_INIT" \
    --alpha_lr "$ALPHA_LR" \
    --model_lr 3e-5 \
    --fs_variant scalar \
    --seed_scale 1.0 \
    --fs_layer_start "$FS_LAYER_START" \
    --fs_norm \
    --fs_detach \
    --fs_clip "$FS_CLIP" | tee "runs/_summary_${TAG}_r1_s012.txt"
done

