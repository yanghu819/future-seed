#!/usr/bin/env bash
set -e

cd rwkv-diff-future-seed
mkdir -p logs weights

# full finetune (baseline)
PYTHONUNBUFFERED=1 TRAIN=1 FUTURE_SEED=1 FS_MASK_ONLY=0 QA_TASK=1 QA_EVAL=1 QA_MODE=qa \
N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 SEQ_LEN=96 DEVICE_BSZ=8 BATCH_SIZE=16 \
MAX_ITERS=200 EVAL_INTERVAL=100 EVAL_ITERS=4 WEIGHTS_PATH=weights/qa_full.pt \
LOG_SAMPLE=0 LOG_OUTPUT=0 \
python rwkv_diff_future_seed.py | tee logs/qa_full.log

# state+mask only
PYTHONUNBUFFERED=1 TRAIN=1 FUTURE_SEED=1 FS_MASK_ONLY=1 QA_TASK=1 QA_EVAL=1 QA_MODE=qa \
N_LAYER=2 N_EMBD=128 HEAD_SIZE=32 SEQ_LEN=96 DEVICE_BSZ=8 BATCH_SIZE=16 \
MAX_ITERS=200 EVAL_INTERVAL=100 EVAL_ITERS=4 WEIGHTS_PATH=weights/qa_fs_mask_only.pt \
LOG_SAMPLE=0 LOG_OUTPUT=0 \
python rwkv_diff_future_seed.py | tee logs/qa_fs_mask_only.log
