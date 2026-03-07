#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mode="${1:-submission}"
mkdir -p build
python3 render_tables.py

python3 - "$mode" <<'PY'
from pathlib import Path
import sys
mode = sys.argv[1]
root = Path('.')
text = (root / 'main.tex').read_text()
if mode == 'submission':
    pkg = r'\usepackage{neurips_2025}'
    author = """\\author{
  Anonymous Authors \\\
  Paper under double-blind review
}"""
    tmp = root / '.main_submission.tex'
elif mode == 'preprint':
    pkg = r'\usepackage[preprint]{neurips_2025}'
    author = """\\author{
  Future-Seed Project Team \\\
  Author information omitted in repository snapshot
}"""
    tmp = root / '.main_preprint.tex'
else:
    raise SystemExit('usage: ./build.sh [submission|preprint]')
text = text.replace('__NEURIPS_PACKAGE__', pkg)
text = text.replace('__AUTHOR_BLOCK__', author)
tmp.write_text(text)
print(tmp.name)
PY

src=".main_${mode}.tex"
tectonic -X compile --keep-logs --keep-intermediates --outdir build "$src"
cp -f "build/${src%.tex}.pdf" "build/neurips2025-${mode}.pdf"
rm -f .main_submission.tex .main_preprint.tex

echo "Wrote: $(pwd)/build/neurips2025-${mode}.pdf"
