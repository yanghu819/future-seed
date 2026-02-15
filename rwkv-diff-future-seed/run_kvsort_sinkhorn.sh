#!/usr/bin/env bash
set -euo pipefail

# KVSORT: Transformer-MLM with structured permutation loss (Sinkhorn assignment).
# This is a stronger baseline than plain token-wise CE.

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
export PATH="$(cd "$(dirname "${PY}")" && pwd):$PATH"

COMMON=(
  PYTHONUNBUFFERED=1
  SEQ_LEN=256
  N_EMBD=256
  N_LAYER=8
  TRANS_N_HEAD=8
  TRANS_DROPOUT=0.0
  TRANS_FF_MULT=4
  BATCH_SIZE=256
  DEVICE_BSZ=16
  MAX_ITERS=800
  EVAL_INTERVAL=200
  EVAL_ITERS=50
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

TAG=kvsort_keys36_n20_tfmlm_sinkhorn_it800

echo "[train] ${TAG}"
env "${COMMON[@]}" \
  MODEL=transformer TRAIN=1 \
  PERM_SINKHORN=1 PERM_SINKHORN_LAMBDA=0.1 PERM_SINKHORN_TAU=1.0 PERM_SINKHORN_ITERS=20 \
  WEIGHTS_PATH="weights/${TAG}.pt" \
  LOG_JSONL="exp/${TAG}.jsonl" \
  "${PY}" rwkv_diff_future_seed.py | tee "exp/${TAG}.log"

echo "[eval] argmax"
env "${COMMON[@]}" \
  MODEL=transformer TRAIN=0 KVSORT_EVAL=1 DECODE=argmax \
  WEIGHTS_PATH="weights/${TAG}.pt" \
  "${PY}" rwkv_diff_future_seed.py | tee "exp/${TAG}_eval_argmax.log"

echo "[eval] hungarian"
env "${COMMON[@]}" \
  MODEL=transformer TRAIN=0 KVSORT_EVAL=1 DECODE=hungarian \
  WEIGHTS_PATH="weights/${TAG}.pt" \
  "${PY}" rwkv_diff_future_seed.py | tee "exp/${TAG}_eval_hungarian.log"

echo "Done."
