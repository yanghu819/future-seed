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

L=2048
SEEDS=(0 1 2)

ALPHA_INIT=-2
ALPHA_LR=5e-4
FS_LAYER_START=8
FS_CLIP=1.0
SEED_SCALE=0.5

TIME_BUDGET=360
N_TRAIN=2000
N_VAL=160
MAX_SEQ=384
MIN_SEQ=96
NUM_PAIRS=64
MIN_PROMPT=1024

for seed in "${SEEDS[@]}"; do
  for mode in no_fs prompt_fs; do
    echo "RUN protein_contact_pair r3_balanced q_after L=$L seed=$seed mode=$mode"
    ./.venv/bin/python train_protein_contact_pair_sft.py \
      --mode "$mode" \
      --seed "$seed" \
      --train_data_seed 0 \
      --val_data_seed 1234 \
      --ds heya5/protein_contact_map \
      --train_split train \
      --val_split valid \
      --n_train "$N_TRAIN" \
      --n_val "$N_VAL" \
      --max_seq_len "$MAX_SEQ" \
      --min_seq_len "$MIN_SEQ" \
      --num_pairs "$NUM_PAIRS" \
      --min_sep 8 \
      --contact_cutoff 8.0 \
      --pair_sampling balanced \
      --pos_ratio 0.5 \
      --fill_notes_to_max \
      --note_pool_size 2048 \
      --max_note_seq_len 256 \
      --max_prompt_tokens "$L" \
      --min_prompt_tokens "$MIN_PROMPT" \
      --max_answer_tokens 256 \
      --bsz 2 \
      --time_budget_sec "$TIME_BUDGET" \
      --eval_every 25 \
      --val_batches 24 \
      --model_lr 3e-5 \
      --alpha_lr "$ALPHA_LR" \
      --alpha_init "$ALPHA_INIT" \
      --seed_scale "$SEED_SCALE" \
      --fs_variant scalar \
      --fs_layer_start "$FS_LAYER_START" \
      --fs_norm \
      --fs_detach \
      --fs_clip "$FS_CLIP"
  done
done

./.venv/bin/python summarize_protein_contact_pair.py \
  --seeds "0,1,2" \
  --bsz 2 \
  --time_budget_sec "$TIME_BUDGET" \
  --max_steps 0 \
  --eval_every 25 \
  --val_batches 24 \
  --ds heya5/protein_contact_map \
  --train_split train \
  --val_split valid \
  --n_train "$N_TRAIN" \
  --n_val "$N_VAL" \
  --max_seq_len "$MAX_SEQ" \
  --min_seq_len "$MIN_SEQ" \
  --num_pairs "$NUM_PAIRS" \
  --min_sep 8 \
  --contact_cutoff 8.0 \
  --pair_sampling balanced \
  --pos_ratio 0.5 \
  --fill_notes_to_max \
  --note_pool_size 2048 \
  --max_note_seq_len 256 \
  --max_prompt_tokens "$L" \
  --min_prompt_tokens "$MIN_PROMPT" \
  --max_answer_tokens 256 \
  --alpha_init "$ALPHA_INIT" \
  --alpha_lr "$ALPHA_LR" \
  --model_lr 3e-5 \
  --fs_variant scalar \
  --seed_scale "$SEED_SCALE" \
  --fs_layer_start "$FS_LAYER_START" \
  --fs_norm \
  --fs_detach \
  --fs_clip "$FS_CLIP" | tee runs/_summary_protein_contact_pair_qafter_len2048_r3_balanced_s012.txt

