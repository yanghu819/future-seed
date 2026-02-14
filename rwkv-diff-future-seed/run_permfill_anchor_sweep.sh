#!/usr/bin/env bash
set -euo pipefail

# PERMFILL: show length extrapolation + anchor effect.
# Produces weights + eval sweep logs under ./exp (gitignored).

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
  RWKV7_KERNEL=cuda_wind
  HEAD_SIZE=64
  SEQ_LEN=256
  N_EMBD=256
  N_LAYER=12
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
    MODEL=rwkv FUTURE_SEED=1 FUTURE_SEED_ALPHA_INIT=-2 TRAIN=1 \
    WEIGHTS_PATH="weights/${tag}.pt" \
    "$@" \
    "${PY}" rwkv_diff_future_seed.py | tee "exp/${tag}.log"
}

eval_sweep () {
  local tag="$1"; shift
  echo "[eval sweep] ${tag}"
  for n in 24 28 32 36; do
    echo "  n_test=${n}"
    env "${COMMON[@]}" \
      MODEL=rwkv FUTURE_SEED=1 TRAIN=0 PERMFILL_EVAL=1 \
      PERMFILL_N_TEST="${n}" \
      WEIGHTS_PATH="weights/${tag}.pt" \
      "$@" \
      "${PY}" rwkv_diff_future_seed.py | tee "exp/${tag}_eval_n${n}.log"
  done
}

# (A) train n_max=24, no anchor
train_one permfill_n24_fs1_L12_seq256 \
  PERMFILL_N_MIN=24 PERMFILL_N_MAX=24 PERMFILL_ANCHOR=0
eval_sweep permfill_n24_fs1_L12_seq256 \
  PERMFILL_ANCHOR=0

# (B) train n_max=24, anchor k=2
train_one permfill_anchor2_n24_fs1_L12_seq256 \
  PERMFILL_N_MIN=24 PERMFILL_N_MAX=24 PERMFILL_ANCHOR=1 PERMFILL_ANCHOR_K=2
eval_sweep permfill_anchor2_n24_fs1_L12_seq256 \
  PERMFILL_ANCHOR=1 PERMFILL_ANCHOR_K=2

# (C) train n_max=32, anchor k=2
train_one permfill_anchor2_n32_fs1_L12_seq256 \
  PERMFILL_N_MIN=32 PERMFILL_N_MAX=32 PERMFILL_ANCHOR=1 PERMFILL_ANCHOR_K=2
eval_sweep permfill_anchor2_n32_fs1_L12_seq256 \
  PERMFILL_ANCHOR=1 PERMFILL_ANCHOR_K=2

echo "Done."
