#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

SRC_DIR="runs"
DST_DIR="results"
DRY_RUN=0
ROUND_FROM=-1
ROUND_TO=-1

usage() {
  echo "Usage: $0 [--src <runs_dir>] [--dst <results_dir>] [--round-from <int>] [--round-to <int>] [--dry-run]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)
      SRC_DIR="$2"
      shift 2
      ;;
    --dst)
      DST_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --round-from)
      ROUND_FROM="$2"
      shift 2
      ;;
    --round-to)
      ROUND_TO="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown arg: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$SRC_DIR" ]]; then
  echo "source dir not found: $SRC_DIR"
  exit 1
fi

mkdir -p "$DST_DIR"

within_round_filter() {
  local name="$1"
  local round
  if [[ "$ROUND_FROM" -lt 0 && "$ROUND_TO" -lt 0 ]]; then
    return 0
  fi
  if [[ "$name" =~ _round([0-9]+)_ ]]; then
    round="${BASH_REMATCH[1]}"
    if [[ "$ROUND_FROM" -ge 0 && "$round" -lt "$ROUND_FROM" ]]; then
      return 1
    fi
    if [[ "$ROUND_TO" -ge 0 && "$round" -gt "$ROUND_TO" ]]; then
      return 1
    fi
    return 0
  fi
  return 1
}

PATTERNS=(
  "_round*_fastdiscover_records.jsonl"
  "_summary_round*_fastdiscover.txt"
  "_launcher_round*.log"
  "_useful_task_pool_fastdiscover.json"
)

COPIED=0
for PAT in "${PATTERNS[@]}"; do
  while IFS= read -r -d '' SRC; do
    BASE="$(basename "$SRC")"
    if ! within_round_filter "$BASE"; then
      continue
    fi
    DST="$DST_DIR/$BASE"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[dry-run] cp $SRC -> $DST"
    else
      cp "$SRC" "$DST"
      echo "[copied] $SRC -> $DST"
    fi
    COPIED=$((COPIED + 1))
  done < <(find "$SRC_DIR" -maxdepth 1 -type f -name "$PAT" -print0)
done

echo "[sync] done, files=$COPIED src=$SRC_DIR dst=$DST_DIR round_from=$ROUND_FROM round_to=$ROUND_TO"
