#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PY_BIN="${PY_BIN:-./.venv/bin/python}"
QUEUE="${QUEUE:-results/_search_queue_round569_576_fastloop.json}"
POLICY="${POLICY:-results/_codex53_team_policy.json}"

FAIL=0

ok() {
  echo "[ok] $*"
}

warn() {
  echo "[warn] $*"
}

err() {
  echo "[err] $*"
  FAIL=1
}

check_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    ok "file exists: $path"
  else
    err "missing file: $path"
  fi
}

check_dir() {
  local path="$1"
  if [[ -d "$path" ]]; then
    ok "dir exists: $path"
  else
    err "missing dir: $path"
  fi
}

check_exec() {
  local path="$1"
  if [[ -x "$path" ]]; then
    ok "executable: $path"
  else
    err "not executable: $path"
  fi
}

echo "[doctor] root=$ROOT_DIR"

check_dir "scripts"
check_dir "results"
check_exec "scripts/run_repropack_569_574.sh"
check_file "scripts/run_round77_82_fastdiscover.py"
check_file "scripts/generate_fastdiscover_audit.py"
check_file "$QUEUE"
check_file "$POLICY"

if [[ -x "$PY_BIN" ]]; then
  ok "python ready: $PY_BIN"
  if "$PY_BIN" scripts/run_round77_82_fastdiscover.py --self_test >/tmp/repro_doctor_selftest.log 2>&1; then
    ok "orchestrator self-test passed"
  else
    err "orchestrator self-test failed (see /tmp/repro_doctor_selftest.log)"
  fi
else
  warn "python not found at $PY_BIN"
  if command -v python3 >/dev/null 2>&1; then
    FB_PY="$(command -v python3)"
    warn "fallback available: $FB_PY"
    if "$FB_PY" scripts/run_round77_82_fastdiscover.py --self_test >/tmp/repro_doctor_selftest.log 2>&1; then
      ok "orchestrator self-test passed with fallback python3"
    else
      err "orchestrator self-test failed with fallback python3 (see /tmp/repro_doctor_selftest.log)"
    fi
  else
    err "no python3 in PATH"
  fi
fi

COUNT_RECORDS=$(find results -maxdepth 1 -type f -name '_round*_fastdiscover_records.jsonl' | wc -l | tr -d ' ')
COUNT_SUMMARY=$(find results -maxdepth 1 -type f -name '_summary_round*_fastdiscover.txt' | wc -l | tr -d ' ')
ok "results snapshot: records=$COUNT_RECORDS summaries=$COUNT_SUMMARY"

if [[ "$FAIL" -ne 0 ]]; then
  echo "[doctor] FAILED"
  exit 1
fi

echo "[doctor] PASSED"
echo "[doctor] next commands:"
echo "  bash scripts/run_repropack_569_574.sh"
echo "  bash scripts/sync_runs_to_results.sh --round-from 569 --round-to 574"
echo "  python3 scripts/generate_fastdiscover_audit.py --results_dir results --round_from 77 --round_to 569 --out_prefix results/_audit_round77_569"
