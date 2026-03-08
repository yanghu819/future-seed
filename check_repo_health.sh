#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PY_BIN="${PY_BIN:-python3}"
QUEUE="${QUEUE:-posttrain_rwkv7/results/_search_queue_round805_808_breadth_roi.json}"
RUN_LINKS=1
RUN_SELF_TEST=1
RUN_DRY_RUN=1
RUN_PAPER=1

usage() {
  cat <<'EOF'
usage: bash check_repo_health.sh [options]

options:
  --skip-links       skip markdown link validation
  --skip-self-test   skip fastdiscover self-test
  --skip-dry-run     skip fastdiscover dry-run
  --skip-paper       skip paper submission build
  --queue PATH       override dry-run queue path
  --python PATH      override python executable
  -h, --help         show this help

environment:
  PY_BIN             default python executable
  QUEUE              default dry-run queue path
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-links)
      RUN_LINKS=0
      shift
      ;;
    --skip-self-test)
      RUN_SELF_TEST=0
      shift
      ;;
    --skip-dry-run)
      RUN_DRY_RUN=0
      shift
      ;;
    --skip-paper)
      RUN_PAPER=0
      shift
      ;;
    --queue)
      QUEUE="$2"
      shift 2
      ;;
    --python)
      PY_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

echo "[health] root=$ROOT_DIR"
echo "[health] python=$PY_BIN"
echo "[health] queue=$QUEUE"

if [[ "$RUN_LINKS" -eq 1 ]]; then
  echo "[health] markdown link check"
  "$PY_BIN" - <<'PY'
import re
from pathlib import Path

files = [
    Path("README.md"),
    Path("PAPER.md"),
    Path("RESULTS.md"),
    Path("GETTING_STARTED.md"),
    Path("TASK_INDEX.md"),
    Path("posttrain_rwkv7/README.md"),
    Path("posttrain_rwkv7/LEGACY_AUTODL.md"),
    Path("posttrain_rwkv7/ARCHIVE_ROUNDS.md"),
    Path("posttrain_rwkv7/results/README_RESULTS.md"),
    Path("paper/neurips2025/README.md"),
    Path("paper/neurips2025/ARTIFACT_GUIDE.md"),
]
pat = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
errors = []
for path in files:
    text = path.read_text(encoding="utf-8")
    for match in pat.finditer(text):
        target = match.group(1).split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            errors.append(f"{path}: {target}")
if errors:
    print("broken links:")
    for err in errors:
        print(f"  {err}")
    raise SystemExit(1)
print("all_links_ok")
PY
else
  echo "[health] markdown link check: skipped"
fi

if [[ "$RUN_SELF_TEST" -eq 1 ]]; then
  echo "[health] fastdiscover self-test"
  "$PY_BIN" posttrain_rwkv7/scripts/run_round77_82_fastdiscover.py --self_test
else
  echo "[health] fastdiscover self-test: skipped"
fi

if [[ "$RUN_DRY_RUN" -eq 1 ]]; then
  echo "[health] fastdiscover dry-run"
  "$PY_BIN" posttrain_rwkv7/scripts/run_round77_82_fastdiscover.py \
    --queue "$QUEUE" \
    --round_from 805 \
    --round_to 808 \
    --dry_run >/tmp/future_seed_repo_health_dry_run.log
  echo "[health] dry-run output saved to /tmp/future_seed_repo_health_dry_run.log"
else
  echo "[health] fastdiscover dry-run: skipped"
fi

if [[ "$RUN_PAPER" -eq 1 ]]; then
  echo "[health] paper build"
  (cd paper/neurips2025 && ./build.sh submission >/tmp/future_seed_repo_health_paper.log)
  echo "[health] paper build log saved to /tmp/future_seed_repo_health_paper.log"
else
  echo "[health] paper build: skipped"
fi

echo "[health] done"
