#!/usr/bin/env python3
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / 'paper' / 'neurips2025'
BUILD_DIR = PAPER_DIR / 'build'
DIST_DIR = PAPER_DIR / 'dist'

ZIP_BASENAME = 'future-seed-neurips2025-supplementary.zip'
SHA256_BASENAME = 'SHA256SUMS.txt'
MAX_SUPPLEMENTARY_BYTES = 100 * 1024 * 1024
OFFICIAL_SUPPLEMENTARY_POLICY_URL = (
    'https://neurips.cc/Conferences/2025/PaperInformation/CodeSubmissionPolicy'
)
OFFICIAL_SUBMISSION_INSTRUCTIONS_URL = (
    'https://neurips.cc/Conferences/2025/PaperInformation/AuthorInstructions'
)

PAPER_TOP_LEVEL_FILES = [
    'paper/neurips2025/main.tex',
    'paper/neurips2025/main_preprint.tex',
    'paper/neurips2025/appendix.tex',
    'paper/neurips2025/checklist.tex',
    'paper/neurips2025/references.bib',
    'paper/neurips2025/neurips_2025.sty',
    'paper/neurips2025/neurips_2025.tex',
    'paper/neurips2025/neurips_2025.pdf',
    'paper/neurips2025/README.md',
    'paper/neurips2025/ARTIFACT_GUIDE.md',
    'paper/neurips2025/LOCAL_REPRO.md',
    'paper/neurips2025/TASK_MATRIX.md',
    'paper/neurips2025/METRICS_PROVENANCE.md',
    'paper/neurips2025/REFERENCE_AUDIT.md',
    'paper/neurips2025/SUPPLEMENTARY_MANIFEST.md',
    'paper/neurips2025/COMPUTE_ACCOUNTING.md',
    'paper/neurips2025/ASSET_LICENSE_MATRIX.md',
    'paper/neurips2025/REPRO_MATRIX.md',
    'paper/neurips2025/render_tables.py',
    'paper/neurips2025/verify_metrics_snapshot.py',
    'paper/neurips2025/verify_submission_layout.py',
    'paper/neurips2025/verify_anonymity_snapshot.py',
    'paper/neurips2025/package_submission_bundle.py',
    'paper/neurips2025/artifact_manifest.py',
]

PAPER_GLOBS = [
    'paper/neurips2025/data/*.json',
    'paper/neurips2025/tables/*.tex',
]

ROOT_FILES = [
    'check_repo_health.sh',
]

POSTTRAIN_SCRIPT_FILES = [
    'posttrain_rwkv7/scripts/README_SCRIPTS.md',
    'posttrain_rwkv7/scripts/cache_defaults.py',
    'posttrain_rwkv7/scripts/run_round77_82_fastdiscover.py',
    'posttrain_rwkv7/scripts/rwkv7_g1d.py',
    'posttrain_rwkv7/scripts/train_protein_ss_spot_sft.py',
    'posttrain_rwkv7/scripts/train_punc_restore_sft.py',
    'posttrain_rwkv7/scripts/train_mbpp_longctx_sft.py',
    'posttrain_rwkv7/scripts/train_arc_mc_sft.py',
    'posttrain_rwkv7/scripts/train_hotpot_longctx_sft.py',
]

POSTTRAIN_QUEUE_FILES = [
    'posttrain_rwkv7/results/_codex53_team_policy.json',
    'posttrain_rwkv7/results/_search_queue_round783_790_realtask_exploit_v3.json',
    'posttrain_rwkv7/results/_search_queue_round799_802_final_confirm.json',
    'posttrain_rwkv7/results/_search_queue_round803_804_mbpp_altconfirm.json',
    'posttrain_rwkv7/results/_search_queue_round805_808_breadth_roi.json',
]

POSTTRAIN_RUN_GLOBS = [
    'posttrain_rwkv7/runs/_summary_round78[3-8]_fastdiscover.txt',
    'posttrain_rwkv7/runs/_summary_round80[0-8]_fastdiscover.txt',
    'posttrain_rwkv7/runs/_summary_round799_fastdiscover.txt',
    'posttrain_rwkv7/runs/_round78[3-8]_fastdiscover_records.jsonl',
    'posttrain_rwkv7/runs/_round80[0-8]_fastdiscover_records.jsonl',
    'posttrain_rwkv7/runs/_round799_fastdiscover_records.jsonl',
]

RWKV_DIFF_GLOBS = [
    'rwkv-diff-future-seed/*',
    'rwkv-diff-future-seed/logs/*',
]

TEXT_SUFFIXES = {
    '.bib',
    '.json',
    '.jsonl',
    '.log',
    '.md',
    '.py',
    '.sh',
    '.sty',
    '.tex',
    '.txt',
    '.yaml',
    '.yml',
}


def _expand_globs(patterns: Iterable[str]) -> Iterator[Path]:
    for pattern in patterns:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                yield path


def iter_manifest_paths() -> Iterator[Path]:
    seen: set[Path] = set()
    for rel in PAPER_TOP_LEVEL_FILES + ROOT_FILES + POSTTRAIN_SCRIPT_FILES + POSTTRAIN_QUEUE_FILES:
        path = ROOT / rel
        if path.is_file() and path not in seen:
            seen.add(path)
            yield path
    for group in (PAPER_GLOBS, RWKV_DIFF_GLOBS, POSTTRAIN_RUN_GLOBS):
        for path in _expand_globs(group):
            if path not in seen:
                seen.add(path)
                yield path


def manifest_relpaths() -> list[str]:
    return sorted(str(path.relative_to(ROOT)) for path in iter_manifest_paths())


def sha256_file(path: Path) -> str:
    h = sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES
