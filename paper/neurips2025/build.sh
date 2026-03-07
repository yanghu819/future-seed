#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mode="${1:-submission}"
mkdir -p build
python3 render_tables.py
case "$mode" in
  submission) src="main.tex" ;;
  preprint) src="main_preprint.tex" ;;
  *) echo "usage: ./build.sh [submission|preprint]" >&2; exit 1 ;;
esac

tectonic -X compile --keep-logs --keep-intermediates --outdir build "$src"
cp -f "build/${src%.tex}.pdf" "build/neurips2025-${mode}.pdf"

echo "Wrote: $(pwd)/build/neurips2025-${mode}.pdf"
