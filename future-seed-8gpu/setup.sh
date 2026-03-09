#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

LOG_DIR=artifacts/setup
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/install.log"

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-.venv}

printf '[setup] cwd=%s\n' "$PWD"
printf '[setup] venv=%s\n' "$VENV_DIR"
printf '[setup] log=%s\n' "$LOG_FILE"

if ! command -v uv >/dev/null 2>&1; then
  echo '[setup] uv not found; installing via official script' 
  curl -LsSf https://astral.sh/uv/install.sh | sh >>"$LOG_FILE" 2>&1
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if [ ! -d "$VENV_DIR" ]; then
  uv venv --python "$PYTHON_BIN" --system-site-packages "$VENV_DIR" >>"$LOG_FILE" 2>&1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install -U pip >>"$LOG_FILE" 2>&1
uv pip install -q \
  transformers>=4.49.0 \
  datasets>=3.3.0 \
  modelscope>=1.25.0 \
  huggingface_hub>=0.29.0 \
  peft>=0.15.0 \
  accelerate>=1.4.0 \
  bitsandbytes>=0.45.0 \
  flash-linear-attention==0.4.1 \
  >>"$LOG_FILE" 2>&1

python - <<'PY' >>"$LOG_FILE" 2>&1
import torch
print({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'gpus': torch.cuda.device_count()})
PY

printf '[setup] done\n'
printf '[setup] detailed_log=%s\n' "$LOG_FILE"
