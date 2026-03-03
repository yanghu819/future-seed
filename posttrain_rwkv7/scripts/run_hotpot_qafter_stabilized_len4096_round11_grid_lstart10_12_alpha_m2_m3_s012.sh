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

TIME_BUDGET=480
N_TRAIN=1000
N_VAL=200
MIN_PROMPT=1536

LSTARTS=(10 12)
ALPHAS=(-2 -3)

for lstart in "${LSTARTS[@]}"; do
  for alpha_init in "${ALPHAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for mode in no_fs prompt_fs; do
        echo "RUN hotpot r11 q_after L=$L lstart=$lstart alpha=$alpha_init seed=$seed mode=$mode"
        ./.venv/bin/python train_hotpot_longctx_sft.py \
          --mode "$mode" \
          --seed "$seed" \
          --train_data_seed 0 \
          --val_data_seed 1234 \
          --ds hotpot_qa \
          --ds_cfg distractor \
          --train_split train \
          --val_split validation \
          --n_train "$N_TRAIN" \
          --n_val "$N_VAL" \
          --max_prompt_tokens "$L" \
          --min_prompt_tokens "$MIN_PROMPT" \
          --max_answer_tokens 24 \
          --bsz 1 \
          --time_budget_sec "$TIME_BUDGET" \
          --eval_every 25 \
          --val_batches 16 \
          --model_lr 3e-5 \
          --alpha_lr 0 \
          --alpha_init "$alpha_init" \
          --seed_scale 1.0 \
          --fs_variant scalar \
          --fs_layer_start "$lstart" \
          --fs_norm \
          --fs_detach \
          --fs_clip 1.0
      done
    done

    ./.venv/bin/python summarize_hotpot_by_len.py \
      --seeds "0,1,2" \
      --bsz 1 \
      --time_budget_sec "$TIME_BUDGET" \
      --max_steps 0 \
      --eval_every 25 \
      --val_batches 16 \
      --ds hotpot_qa \
      --ds_cfg distractor \
      --train_split train \
      --val_split validation \
      --alpha_init "$alpha_init" \
      --alpha_lr 0 \
      --model_lr 3e-5 \
      --fs_variant scalar \
      --seed_scale 1.0 \
      --n_train "$N_TRAIN" \
      --n_val "$N_VAL" \
      --max_prompt_tokens "$L" \
      --min_prompt_tokens "$MIN_PROMPT" \
      --max_answer_tokens 24 \
      --fs_layer_start "$lstart" \
      --fs_norm \
      --fs_detach \
      --fs_clip 1.0 | tee "runs/_summary_hotpot_qafter_stabilized_len4096_r11_lstart${lstart}_alpha${alpha_init}_s012.txt"
  done
done

