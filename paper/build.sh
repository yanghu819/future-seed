#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

tectonic -X compile --outdir build main.tex
cp -f build/main.pdf future-seed-report.pdf
cp -f build/main.pdf future-seed-sudoku.pdf
tectonic -X compile --outdir build main_en.tex
cp -f build/main_en.pdf future-seed-report-en.pdf

echo "Wrote: $(pwd)/future-seed-report.pdf"
echo "Wrote: $(pwd)/future-seed-report-en.pdf"
