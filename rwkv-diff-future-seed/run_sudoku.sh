#!/usr/bin/env bash
set -euo pipefail

mkdir -p weights exp

# NOTE:
# - Requires RWKV7 cuda_wind build (on AutoDL):
#   RWKV7_KERNEL=cuda_wind RWKV7_CUDA_SRC=/path/to/rwkv_cuda_wind
# - Eval-only requires WEIGHTS_PATH to exist and TRAIN=0.

COMMON="PYTHONUNBUFFERED=1 SUDOKU_TASK=1 SUDOKU_EVAL=1 SUDOKU_MASK_MODE=random \
HEAD_SIZE=32 SEQ_LEN=128 DEVICE_BSZ=64 BATCH_SIZE=256 EVAL_ITERS=4"

echo "== Train FS=0 curriculum [4,12] eval holes=12"
env $COMMON TRAIN=1 FUTURE_SEED=0 N_LAYER=4 N_EMBD=128 \
SUDOKU_HOLES_MIN=4 SUDOKU_HOLES_MAX=12 SUDOKU_HOLES_TEST=12 SUDOKU_TRIALS=200 \
MAX_ITERS=2000 EVAL_INTERVAL=400 \
WEIGHTS_PATH=weights/sudoku_fs0_curr4_12_L4_e128.pt \
python -u rwkv_diff_future_seed.py | tee exp/sudoku_fs0_curr4_12_L4_e128.log

echo "== Train FS=1 curriculum [4,12] eval holes=12"
env $COMMON TRAIN=1 FUTURE_SEED=1 N_LAYER=12 N_EMBD=128 \
SUDOKU_HOLES_MIN=4 SUDOKU_HOLES_MAX=12 SUDOKU_HOLES_TEST=12 SUDOKU_TRIALS=200 \
MAX_ITERS=2500 EVAL_INTERVAL=500 \
WEIGHTS_PATH=weights/sudoku_fs1_curr4_12_L12_e128.pt \
python -u rwkv_diff_future_seed.py | tee exp/sudoku_fs1_curr4_12_L12_e128.log

echo "== Phase curve sweep (trials=2000)"
for h in 4 6 8 10 12 14; do
  echo "--- FS0 holes=$h"
  env $COMMON TRAIN=0 FUTURE_SEED=0 N_LAYER=4 N_EMBD=128 \
  SUDOKU_HOLES_TEST=$h SUDOKU_TRIALS=2000 \
  WEIGHTS_PATH=weights/sudoku_fs0_curr4_12_L4_e128.pt \
  python -u rwkv_diff_future_seed.py | tail -n 1

  echo "--- FS1 holes=$h"
  env $COMMON TRAIN=0 FUTURE_SEED=1 N_LAYER=12 N_EMBD=128 \
  SUDOKU_HOLES_TEST=$h SUDOKU_TRIALS=2000 \
  WEIGHTS_PATH=weights/sudoku_fs1_curr4_12_L12_e128.pt \
  python -u rwkv_diff_future_seed.py | tail -n 1
done

echo "Done."

