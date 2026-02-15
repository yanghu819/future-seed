#!/usr/bin/env bash
set -euo pipefail

# KVSORT: compare Transformer-Causal (no future) vs Transformer-Causal + ATTN_FS (cross-layer future-seed).
# Uses the clean permutation setting (keys-only + no-sep). Optional Hungarian decode for eval.

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

train_one () {
  local tag="$1"; shift
  echo "[train] ${tag}"
  env "${COMMON[@]}" \
    MODEL=transformer_causal TRAIN=1 \
    WEIGHTS_PATH="weights/${tag}.pt" \
    LOG_JSONL="exp/${tag}.jsonl" \
    "$@" \
    "${PY}" rwkv_diff_future_seed.py | tee "exp/${tag}.log"
}

eval_one () {
  local tag="$1"; shift
  echo "[eval] ${tag}"
  env "${COMMON[@]}" \
    MODEL=transformer_causal TRAIN=0 KVSORT_EVAL=1 \
    WEIGHTS_PATH="weights/${tag}.pt" \
    "$@" \
    "${PY}" rwkv_diff_future_seed.py | tee "exp/${tag}_eval.log"
}

eval_hungarian () {
  local tag="$1"; shift
  echo "[eval hungarian] ${tag}"
  env "${COMMON[@]}" \
    MODEL=transformer_causal TRAIN=0 KVSORT_EVAL=1 DECODE=hungarian \
    WEIGHTS_PATH="weights/${tag}.pt" \
    "$@" \
    "${PY}" rwkv_diff_future_seed.py | tee "exp/${tag}_eval_hungarian.log"
}

# (A) causal transformer baseline (no future)
train_one kvsort_keys36_n20_tfc_fs0 \
  ATTN_FS=0 MAX_ITERS=200 EVAL_INTERVAL=200 EVAL_ITERS=50
eval_one kvsort_keys36_n20_tfc_fs0 \
  ATTN_FS=0
eval_hungarian kvsort_keys36_n20_tfc_fs0 \
  ATTN_FS=0

# (B) causal transformer + attention future-seed (global token)
train_one kvsort_keys36_n20_tfc_attnfs \
  ATTN_FS=1 ATTN_FS_COLLECTOR=learned
eval_one kvsort_keys36_n20_tfc_attnfs \
  ATTN_FS=1 ATTN_FS_COLLECTOR=learned
eval_hungarian kvsort_keys36_n20_tfc_attnfs \
  ATTN_FS=1 ATTN_FS_COLLECTOR=learned

echo "Done."
