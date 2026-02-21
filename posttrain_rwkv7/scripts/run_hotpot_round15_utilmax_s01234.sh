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
SEEDS=(0 1 2 3 4)
TIME_BUDGET=900
N_TRAIN=1000
N_VAL=200
MIN_PROMPT=1536
ALPHA_INIT=-3
FS_LAYER_START=10

for qfirst in 0 1; do
  for seed in "${SEEDS[@]}"; do
    for mode in no_fs prompt_fs; do
      echo "RUN hotpot r15 utilmax qfirst=$qfirst L=$L lstart=$FS_LAYER_START alpha=$ALPHA_INIT seed=$seed mode=$mode"
      EXTRA=()
      if [ "$qfirst" = "1" ]; then EXTRA+=(--q_first); fi
      ./.venv/bin/python train_hotpot_longctx_sft.py \
        --mode "$mode" \
        --seed "$seed" \
        --train_data_seed 0 \
        --val_data_seed 1234 \
        "${EXTRA[@]}" \
        --ds hotpot_qa \
        --ds_cfg distractor \
        --train_split train \
        --val_split validation \
        --n_train "$N_TRAIN" \
        --n_val "$N_VAL" \
        --max_prompt_tokens "$L" \
        --min_prompt_tokens "$MIN_PROMPT" \
        --max_answer_tokens 24 \
        --bsz 2 \
        --time_budget_sec "$TIME_BUDGET" \
        --max_steps 900 \
        --eval_every 25 \
        --val_batches 8 \
        --model_lr 3e-5 \
        --alpha_lr 0 \
        --alpha_init "$ALPHA_INIT" \
        --seed_scale 1.0 \
        --fs_variant scalar \
        --fs_layer_start "$FS_LAYER_START" \
        --fs_norm \
        --fs_detach \
        --fs_clip 1.0
    done
  done

  EXTRA_SUM=()
  OUT_NAME="_summary_hotpot_qafter_len4096_r15_utilmax_s01234.txt"
  if [ "$qfirst" = "1" ]; then
    EXTRA_SUM+=(--q_first)
    OUT_NAME="_summary_hotpot_qfirst_len4096_r15_utilmax_s01234.txt"
  fi

  ./.venv/bin/python summarize_hotpot_by_len.py \
    "${EXTRA_SUM[@]}" \
    --seeds "0,1,2,3,4" \
    --bsz 2 \
    --time_budget_sec "$TIME_BUDGET" \
    --max_steps 900 \
    --eval_every 25 \
    --val_batches 8 \
    --ds hotpot_qa \
    --ds_cfg distractor \
    --train_split train \
    --val_split validation \
    --alpha_init "$ALPHA_INIT" \
    --alpha_lr 0 \
    --model_lr 3e-5 \
    --fs_variant scalar \
    --seed_scale 1.0 \
    --n_train "$N_TRAIN" \
    --n_val "$N_VAL" \
    --max_prompt_tokens "$L" \
    --min_prompt_tokens "$MIN_PROMPT" \
    --max_answer_tokens 24 \
    --fs_layer_start "$FS_LAYER_START" \
    --fs_norm \
    --fs_detach \
    --fs_clip 1.0 | tee "runs/${OUT_NAME}"
done
