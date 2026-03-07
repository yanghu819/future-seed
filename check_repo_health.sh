#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PY_BIN="${PY_BIN:-python3}"
QUEUE="${QUEUE:-posttrain_rwkv7/results/_search_queue_round805_808_breadth_roi.json}"

echo "[health] root=$ROOT_DIR"
echo "[health] python=$PY_BIN"

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

echo "[health] fastdiscover self-test"
"$PY_BIN" posttrain_rwkv7/scripts/run_round77_82_fastdiscover.py --self_test

echo "[health] fastdiscover dry-run"
"$PY_BIN" posttrain_rwkv7/scripts/run_round77_82_fastdiscover.py \
  --queue "$QUEUE" \
  --round_from 805 \
  --round_to 808 \
  --dry_run >/tmp/future_seed_repo_health_dry_run.log
echo "[health] dry-run output saved to /tmp/future_seed_repo_health_dry_run.log"

echo "[health] paper build"
(cd paper/neurips2025 && ./build.sh submission >/tmp/future_seed_repo_health_paper.log)
echo "[health] paper build log saved to /tmp/future_seed_repo_health_paper.log"

echo "[health] done"
