#!/usr/bin/env bash
set -euo pipefail

# WikiText-2 prefix infill on byte-level tokens (VOCAB_SIZE=256).
# Requires bin files built locally, then copied to the machine:
#   python tools/build_hf_bins.py --dataset wikitext --config wikitext-2-raw-v1 \
#     --train_split train --val_split validation --fields text --out_dir data/wikitext2_bytes
#
# Run (CUDA):
#   bash rwkv-diff-future-seed/run_wikitext_prefix.sh /path/to/data/wikitext2_bytes

cd "$(dirname "$0")"
DATA_DIR="${1:-}"
if [[ -z "${DATA_DIR}" ]]; then
  echo "usage: $0 /path/to/data/wikitext2_bytes"
  exit 2
fi

mkdir -p exp weights

PY="${PYTHON:-}"
if [[ -z "${PY}" ]]; then
  if [[ -x "../.venv/bin/python" ]]; then
    PY="../.venv/bin/python"
  elif [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
  else
    PY="python3"
  fi
fi

COMMON=(
  PYTHONUNBUFFERED=1
  RWKV7_KERNEL=cuda_wind
  VOCAB_SIZE=256
  SEQ_LEN=512
  N_EMBD=512
  N_LAYER=12
  HEAD_SIZE=64
  BATCH_SIZE=256
  DEVICE_BSZ=8
  MAX_ITERS=20000
  EVAL_INTERVAL=1000
  EVAL_ITERS=50
  DATA_BIN="${DATA_DIR}/train.bin"
  DATA_VAL_BIN="${DATA_DIR}/val.bin"
  BIN_MASK_MODE=prefix
  BIN_PREFIX_RATIO=0.50
  MASKACC_EVAL=1
  MASKACC_SPLIT=val
  MASKACC_ITERS=50
  LOG_SAMPLE=0
)

echo "[1/3] RWKV FS=0"
env "${COMMON[@]}" MODEL=rwkv FUTURE_SEED=0 TRAIN=1 \
  WEIGHTS_PATH=weights/wikitext2_prefix_fs0_rwkv.pt \
  "${PY}" rwkv_diff_future_seed.py | tee exp/wikitext2_prefix_fs0_rwkv.log

echo "[2/3] RWKV FS=1"
env "${COMMON[@]}" MODEL=rwkv FUTURE_SEED=1 FUTURE_SEED_ALPHA_INIT=-2 TRAIN=1 \
  WEIGHTS_PATH=weights/wikitext2_prefix_fs1_rwkv.pt \
  "${PY}" rwkv_diff_future_seed.py | tee exp/wikitext2_prefix_fs1_rwkv.log

echo "[3/3] Transformer MLM"
env "${COMMON[@]}" MODEL=transformer TRAIN=1 TRANS_N_HEAD=8 \
  WEIGHTS_PATH=weights/wikitext2_prefix_tfmlm.pt \
  "${PY}" rwkv_diff_future_seed.py | tee exp/wikitext2_prefix_tfmlm.log

echo "Done."
