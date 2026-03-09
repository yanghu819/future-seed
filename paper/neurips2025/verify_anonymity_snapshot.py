#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from artifact_manifest import ROOT, is_text_file, iter_manifest_paths

BANNED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ('local user name', re.compile(r'\btorusmini\b', re.IGNORECASE)),
    ('repository owner', re.compile(r'\byanghu819\b', re.IGNORECASE)),
    ('explicit local absolute path', re.compile(r'/Users/[^\s\"\']+')),
    ('legacy remote path', re.compile(r'/root/autodl-tmp\b', re.IGNORECASE)),
    ('legacy ssh key path', re.compile(r'~/.ssh/autodl_ed25519', re.IGNORECASE)),
    ('legacy SSH command', re.compile(r'ssh -o StrictHostKeyChecking=no', re.IGNORECASE)),
    ('legacy remote host', re.compile(r'connect\.bjb1\.seetacloud\.com', re.IGNORECASE)),
    ('legacy root login', re.compile(r'root@connect', re.IGNORECASE)),
    ('owner GitHub URL', re.compile(r'github\.com/yanghu819', re.IGNORECASE)),
]

TEXT_SUFFIXES = {'.bib', '.json', '.jsonl', '.log', '.md', '.py', '.sh', '.sty', '.tex', '.txt', '.yaml', '.yml'}


@dataclass
class Finding:
    where: str
    rule: str
    snippet: str


def iter_text_lines(path: Path) -> Iterable[tuple[int, str]]:
    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        text = path.read_text(encoding='utf-8', errors='ignore')
    for idx, line in enumerate(text.splitlines(), start=1):
        yield idx, line


def scan_text_blob(label: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        for rule_name, pattern in BANNED_PATTERNS:
            match = pattern.search(line)
            if match:
                findings.append(Finding(f'{label}:{idx}', rule_name, line.strip()[:240]))
    return findings


def scan_path(path: Path) -> list[Finding]:
    rel = str(path.relative_to(ROOT))
    findings: list[Finding] = []
    for rule_name, pattern in BANNED_PATTERNS:
        if pattern.search(rel):
            findings.append(Finding(rel, rule_name, rel))
    if path.name == 'verify_anonymity_snapshot.py':
        return findings
    if path.suffix.lower() in TEXT_SUFFIXES:
        for idx, line in iter_text_lines(path):
            for rule_name, pattern in BANNED_PATTERNS:
                if pattern.search(line):
                    findings.append(Finding(f'{rel}:{idx}', rule_name, line.strip()[:240]))
    return findings


def scan_zip(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            for rule_name, pattern in BANNED_PATTERNS:
                if pattern.search(info.filename):
                    findings.append(Finding(f'{path.name}:{info.filename}', rule_name, info.filename))
            if Path(info.filename).name == 'verify_anonymity_snapshot.py':
                continue
            suffix = Path(info.filename).suffix.lower()
            if suffix not in TEXT_SUFFIXES:
                continue
            try:
                data = zf.read(info).decode('utf-8')
            except UnicodeDecodeError:
                data = zf.read(info).decode('utf-8', errors='ignore')
            findings.extend(scan_text_blob(f'{path.name}:{info.filename}', data))
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description='Verify that the curated anonymous snapshot does not leak local identity or legacy remote state.')
    ap.add_argument('--zip', dest='zip_path', type=Path, default=None, help='optional supplementary zip to inspect in addition to source files')
    ap.add_argument('--json', action='store_true', help='emit JSON report')
    ap.add_argument('--strict', action='store_true', help='exit non-zero on findings (default behavior)')
    args = ap.parse_args()

    findings: list[Finding] = []
    for path in iter_manifest_paths():
        findings.extend(scan_path(path))

    sub_ready = ROOT / 'paper' / 'neurips2025' / 'SUBMISSION_READY.md'
    if sub_ready.exists():
        findings.extend(scan_path(sub_ready))

    if args.zip_path is not None:
        findings.extend(scan_zip(args.zip_path))

    if args.json:
        print(json.dumps([f.__dict__ for f in findings], indent=2, ensure_ascii=False))
    else:
        if findings:
            print('anonymity findings:')
            for finding in findings:
                print(f'- {finding.where} [{finding.rule}] {finding.snippet}')
        else:
            print('anonymity_ok')

    if findings:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
