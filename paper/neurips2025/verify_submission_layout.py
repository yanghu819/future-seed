#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / 'build'

CASES = {
    'submission': {
        'pdf_path': BUILD / 'neurips2025-submission.pdf',
        'log_path': BUILD / 'main.log',
        'max_content_pages': 9,
    },
    'preprint': {
        'pdf_path': BUILD / 'neurips2025-preprint.pdf',
        'log_path': BUILD / 'main_preprint.log',
        'max_content_pages': 9,
    },
}

MARKERS = ('BIBSTART_PAGE', 'APPENDIX_PAGE', 'CHECKLIST_PAGE')


def parse_markers(log_path: Path) -> dict[str, int]:
    text = log_path.read_text(encoding='utf-8', errors='replace')
    found: dict[str, int] = {}
    for marker in MARKERS:
        m = re.search(rf'{marker}=(\d+)', text)
        if not m:
            raise AssertionError(f'missing {marker} in {log_path}')
        found[marker] = int(m.group(1))
    return found


def inspect_case(name: str, *, pdf_path: Path, log_path: Path, max_content_pages: int) -> list[str]:
    if not pdf_path.exists():
        raise AssertionError(f'missing PDF: {pdf_path}')
    if not log_path.exists():
        raise AssertionError(f'missing log: {log_path}')
    markers = parse_markers(log_path)
    pdf = PdfReader(str(pdf_path))
    total_pages = len(pdf.pages)
    bib_page = markers['BIBSTART_PAGE']
    appendix_page = markers['APPENDIX_PAGE']
    checklist_page = markers['CHECKLIST_PAGE']
    content_pages = bib_page - 1
    if content_pages > max_content_pages:
        raise AssertionError(
            f'{name}: content pages {content_pages} exceed NeurIPS limit {max_content_pages}'
        )
    if not (1 <= bib_page <= appendix_page <= checklist_page <= total_pages + 1):
        raise AssertionError(
            f'{name}: invalid section page order '
            f'(bib={bib_page}, appendix={appendix_page}, checklist={checklist_page}, total={total_pages})'
        )
    return [
        f'{name}: total_pages={total_pages}',
        f'{name}: content_pages={content_pages}',
        f'{name}: bibliography starts on page {bib_page}',
        f'{name}: appendix starts on page {appendix_page}',
        f'{name}: checklist starts on page {checklist_page}',
    ]


def main() -> None:
    out: list[str] = []
    for name, cfg in CASES.items():
        out.extend(inspect_case(name, **cfg))
    print('submission_layout_ok')
    for line in out:
        print(f'- {line}')


if __name__ == '__main__':
    main()
