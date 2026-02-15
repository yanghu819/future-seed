#!/usr/bin/env bash
set -euo pipefail

# PERMFILL: compare Transformer-Causal (no future) vs Transformer-Causal + ATTN_FS (cross-layer future-seed).
# Includes an anchor setting (k=2) to avoid the fully-masked degenerate case.

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
  N_LAYER=12
  TRANS_N_HEAD=8
  TRANS_DROPOUT=0.0
  TRANS_FF_MULT=4
  BATCH_SIZE=256
  DEVICE_BSZ=16
  MAX_ITERS=1200
  EVAL_INTERVAL=200
  EVAL_ITERS=200
  LOG_SAMPLE=1
  LOG_WIN=40
  PERMFILL_TASK=1
  PERMFILL_EVAL=1
  PERMFILL_USE_ORDER=1
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

eval_sweep () {
  local tag="$1"; shift
  local decode="${1:-argmax}"; shift || true
  echo "[eval sweep] ${tag} (DECODE=${decode})"
  for n in 24 28 32 36; do
    echo "  n_test=${n}"
    env "${COMMON[@]}" \
      MODEL=transformer_causal TRAIN=0 PERMFILL_EVAL=1 DECODE="${decode}" \
      PERMFILL_N_TEST="${n}" \
      WEIGHTS_PATH="weights/${tag}.pt" \
      "$@" \
      "${PY}" rwkv_diff_future_seed.py | tee "exp/${tag}_eval_${decode}_n${n}.log"
  done
}

# (A) causal transformer baseline (no future)
train_one permfill_anchor2_n24_tfc_fs0 \
  ATTN_FS=0 \
  PERMFILL_N_MIN=24 PERMFILL_N_MAX=24 \
  PERMFILL_ANCHOR=1 PERMFILL_ANCHOR_K=2
eval_sweep permfill_anchor2_n24_tfc_fs0 argmax \
  ATTN_FS=0 PERMFILL_ANCHOR=1 PERMFILL_ANCHOR_K=2
eval_sweep permfill_anchor2_n24_tfc_fs0 hungarian \
  ATTN_FS=0 PERMFILL_ANCHOR=1 PERMFILL_ANCHOR_K=2

# (B) causal transformer + attention future-seed (global token)
train_one permfill_anchor2_n24_tfc_attnfs \
  ATTN_FS=1 ATTN_FS_COLLECTOR=learned \
  PERMFILL_N_MIN=24 PERMFILL_N_MAX=24 \
  PERMFILL_ANCHOR=1 PERMFILL_ANCHOR_K=2
eval_sweep permfill_anchor2_n24_tfc_attnfs argmax \
  ATTN_FS=1 ATTN_FS_COLLECTOR=learned PERMFILL_ANCHOR=1 PERMFILL_ANCHOR_K=2
eval_sweep permfill_anchor2_n24_tfc_attnfs hungarian \
  ATTN_FS=1 ATTN_FS_COLLECTOR=learned PERMFILL_ANCHOR=1 PERMFILL_ANCHOR_K=2

echo "Done."

