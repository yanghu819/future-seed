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
RUN_METRICS=1
RUN_LAYOUT=1
RUN_ANON=1
RUN_PACKAGE=1
PACKAGE_MODE="auto"
CHECKPOINT_PATH="${FUTURE_SEED_CHECKPOINT_PATH:-}"
CHECKPOINT_URL="${FUTURE_SEED_CHECKPOINT_URL:-}"
CHECKPOINT_SHA256="${FUTURE_SEED_CHECKPOINT_SHA256:-}"
ALLOW_OVERSIZE=0

usage() {
  cat <<'EOF'
usage: bash check_repo_health.sh [options]

options:
  --skip-links          skip markdown link validation
  --skip-self-test      skip fastdiscover self-test
  --skip-dry-run        skip fastdiscover dry-run
  --skip-metrics        skip paper metrics snapshot verification
  --skip-layout         skip NeurIPS paper layout/page-budget verification
  --skip-paper          skip paper submission/preprint builds
  --skip-anonymity      skip curated source anonymity verification
  --skip-package        skip supplementary ZIP packaging
  --queue PATH          override dry-run queue path
  --python PATH         override python executable
  --package-mode MODE   one of: auto, omit, bundle, link
  --checkpoint-path P   checkpoint file to bundle into the supplementary ZIP
  --checkpoint-url URL  anonymous external checkpoint URL to record in package metadata
  --checkpoint-sha256 H checkpoint SHA256 to record for bundle/link mode
  --allow-oversize      allow supplementary ZIP above the NeurIPS 100MB threshold
  -h, --help            show this help

environment:
  PY_BIN                        default python executable
  QUEUE                         default dry-run queue path
  FUTURE_SEED_CHECKPOINT_PATH   default bundled checkpoint path
  FUTURE_SEED_CHECKPOINT_URL    default anonymous checkpoint URL
  FUTURE_SEED_CHECKPOINT_SHA256 default checkpoint SHA256 for bundle/link mode
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
    --skip-metrics)
      RUN_METRICS=0
      shift
      ;;
    --skip-layout)
      RUN_LAYOUT=0
      shift
      ;;
    --skip-paper)
      RUN_PAPER=0
      shift
      ;;
    --skip-anonymity)
      RUN_ANON=0
      shift
      ;;
    --skip-package)
      RUN_PACKAGE=0
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
    --package-mode)
      PACKAGE_MODE="$2"
      shift 2
      ;;
    --checkpoint-path)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --checkpoint-url)
      CHECKPOINT_URL="$2"
      shift 2
      ;;
    --checkpoint-sha256)
      CHECKPOINT_SHA256="$2"
      shift 2
      ;;
    --allow-oversize)
      ALLOW_OVERSIZE=1
      shift
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
echo "[health] package_mode=$PACKAGE_MODE"

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
    Path("paper/neurips2025/LOCAL_REPRO.md"),
    Path("paper/neurips2025/SUPPLEMENTARY_MANIFEST.md"),
    Path("paper/neurips2025/COMPUTE_ACCOUNTING.md"),
    Path("paper/neurips2025/ASSET_LICENSE_MATRIX.md"),
    Path("paper/neurips2025/REPRO_MATRIX.md"),
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

if [[ "$RUN_METRICS" -eq 1 ]]; then
  echo "[health] paper metrics snapshot verification"
  "$PY_BIN" paper/neurips2025/verify_metrics_snapshot.py >/tmp/future_seed_repo_health_metrics.log 2>&1
  echo "[health] metrics verification log saved to /tmp/future_seed_repo_health_metrics.log"
else
  echo "[health] paper metrics snapshot verification: skipped"
fi

if [[ "$RUN_PAPER" -eq 1 ]]; then
  echo "[health] paper build"
  (cd paper/neurips2025 && ./build.sh submission >/tmp/future_seed_repo_health_paper_submission.log 2>&1)
  (cd paper/neurips2025 && ./build.sh preprint >/tmp/future_seed_repo_health_paper_preprint.log 2>&1)
  echo "[health] paper submission log saved to /tmp/future_seed_repo_health_paper_submission.log"
  echo "[health] paper preprint log saved to /tmp/future_seed_repo_health_paper_preprint.log"
else
  echo "[health] paper build: skipped"
fi

if [[ "$RUN_LAYOUT" -eq 1 ]]; then
  echo "[health] paper layout verification"
  "$PY_BIN" paper/neurips2025/verify_submission_layout.py >/tmp/future_seed_repo_health_layout.log 2>&1
  echo "[health] layout verification log saved to /tmp/future_seed_repo_health_layout.log"
else
  echo "[health] paper layout verification: skipped"
fi

if [[ "$RUN_ANON" -eq 1 ]]; then
  echo "[health] source anonymity verification"
  "$PY_BIN" paper/neurips2025/verify_anonymity_snapshot.py >/tmp/future_seed_repo_health_anonymity.log 2>&1
  echo "[health] anonymity verification log saved to /tmp/future_seed_repo_health_anonymity.log"
else
  echo "[health] source anonymity verification: skipped"
fi

if [[ "$RUN_PACKAGE" -eq 1 ]]; then
  echo "[health] supplementary packaging"
  PACKAGE_CMD=(
    "$PY_BIN" paper/neurips2025/package_submission_bundle.py
    --mode "$PACKAGE_MODE"
    --python "$PY_BIN"
  )
  if [[ -n "$CHECKPOINT_PATH" ]]; then
    PACKAGE_CMD+=(--checkpoint-path "$CHECKPOINT_PATH")
  fi
  if [[ -n "$CHECKPOINT_URL" ]]; then
    PACKAGE_CMD+=(--checkpoint-url "$CHECKPOINT_URL")
  fi
  if [[ -n "$CHECKPOINT_SHA256" ]]; then
    PACKAGE_CMD+=(--checkpoint-sha256 "$CHECKPOINT_SHA256")
  fi
  if [[ "$RUN_ANON" -eq 0 ]]; then
    PACKAGE_CMD+=(--skip-anonymity-check)
  fi
  if [[ "$ALLOW_OVERSIZE" -eq 1 ]]; then
    PACKAGE_CMD+=(--allow-oversize)
  fi
  "${PACKAGE_CMD[@]}" >/tmp/future_seed_repo_health_package.log 2>&1
  echo "[health] package log saved to /tmp/future_seed_repo_health_package.log"
else
  echo "[health] supplementary packaging: skipped"
fi

echo "[health] done"
