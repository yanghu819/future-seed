#!/usr/bin/env bash
set -euo pipefail

# KVSORT baselines: RWKV FS=0 vs FS=1 vs Transformer-MLM.
# Notes:
# - keys-only + no-sep is the clean permutation setting (Hungarian decoding optional via DECODE=hungarian).
# - Logs go to ./exp (gitignored).

cd "$(dirname "$0")"
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
  HEAD_SIZE=64
  SEQ_LEN=256
  N_EMBD=256
  N_LAYER=8
  BATCH_SIZE=256
  DEVICE_BSZ=16
  MAX_ITERS=2400
  EVAL_INTERVAL=400
  EVAL_ITERS=200
  LOG_SAMPLE=1
  LOG_WIN=40
  KVSORT_TASK=1
  KVSORT_EVAL=1
  KVSORT_NOISE=0
  KVSORT_USE_ORDER=1
  KVSORT_KEYS_ONLY=1
  KVSORT_KEYS_SEP=0
  KVSORT_MASK_SEP=1
  KVSORT_N_MIN=20
  KVSORT_N_MAX=20
  KVSORT_N_TEST=20
)

echo "[1/3] RWKV FS=0"
env "${COMMON[@]}" \
  MODEL=rwkv FUTURE_SEED=0 TRAIN=1 \
  WEIGHTS_PATH=weights/kvsort_keys36_n20_fs0_rwkv.pt \
  "${PY}" rwkv_diff_future_seed.py | tee exp/kvsort_keys36_n20_fs0_rwkv.log

echo "[2/3] RWKV FS=1"
env "${COMMON[@]}" \
  MODEL=rwkv FUTURE_SEED=1 FUTURE_SEED_ALPHA_INIT=-2 TRAIN=1 \
  WEIGHTS_PATH=weights/kvsort_keys36_n20_fs1_rwkv.pt \
  "${PY}" rwkv_diff_future_seed.py | tee exp/kvsort_keys36_n20_fs1_rwkv.log

echo "[3/3] Transformer MLM"
env "${COMMON[@]}" \
  MODEL=transformer TRAIN=1 TRANS_N_HEAD=8 \
  WEIGHTS_PATH=weights/kvsort_keys36_n20_tfmlm.pt \
  "${PY}" rwkv_diff_future_seed.py | tee exp/kvsort_keys36_n20_tfmlm.log

echo "Done. For Hungarian decoding, run eval-only like:"
echo "  DECODE=hungarian TRAIN=0 KVSORT_EVAL=1 WEIGHTS_PATH=... python rwkv_diff_future_seed.py"
