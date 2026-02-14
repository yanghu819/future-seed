#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

tectonic -X compile --outdir build main.tex
cp -f build/main.pdf future-seed-sudoku.pdf
echo "Wrote: $(pwd)/future-seed-sudoku.pdf"

