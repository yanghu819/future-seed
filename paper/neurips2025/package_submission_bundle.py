#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from artifact_manifest import (
    BUILD_DIR,
    DIST_DIR,
    MAX_SUPPLEMENTARY_BYTES,
    OFFICIAL_SUBMISSION_INSTRUCTIONS_URL,
    OFFICIAL_SUPPLEMENTARY_POLICY_URL,
    PAPER_DIR,
    ROOT,
    SHA256_BASENAME,
    ZIP_BASENAME,
    iter_manifest_paths,
    sha256_file,
)


def default_checkpoint_spec() -> dict[str, str | None]:
    return {
        'mode': 'omit',
        'path': os.environ.get('FUTURE_SEED_CHECKPOINT_PATH'),
        'url': os.environ.get('FUTURE_SEED_CHECKPOINT_URL'),
        'sha256': os.environ.get('FUTURE_SEED_CHECKPOINT_SHA256'),
    }


def resolve_checkpoint_spec(args: argparse.Namespace) -> dict[str, str | None]:
    spec = default_checkpoint_spec()
    if args.mode != 'auto':
        spec['mode'] = args.mode
    if args.checkpoint_path is not None:
        spec['path'] = str(args.checkpoint_path)
    if args.checkpoint_url is not None:
        spec['url'] = args.checkpoint_url
    if args.checkpoint_sha256 is not None:
        spec['sha256'] = args.checkpoint_sha256

    if spec['mode'] == 'auto':
        if spec['path']:
            spec['mode'] = 'bundle'
        elif spec['url']:
            spec['mode'] = 'link'
        else:
            spec['mode'] = 'omit'

    if spec['mode'] == 'bundle':
        if not spec['path']:
            raise SystemExit('bundle mode requires --checkpoint-path or FUTURE_SEED_CHECKPOINT_PATH')
        path = Path(spec['path']).expanduser()
        if not path.is_file():
            raise SystemExit(f'checkpoint file not found: {path}')
        spec['path'] = str(path.resolve())
        if not spec['sha256']:
            spec['sha256'] = sha256_file(path)
    elif spec['mode'] == 'link':
        if not spec['url']:
            raise SystemExit('link mode requires --checkpoint-url or FUTURE_SEED_CHECKPOINT_URL')
    elif spec['mode'] == 'omit':
        spec['path'] = None
        spec['url'] = None
    else:
        raise SystemExit(f'unsupported checkpoint mode: {spec["mode"]}')

    return spec


def run_anonymity_check(py_bin: str, zip_path: Path | None = None) -> None:
    cmd = [py_bin, str(PAPER_DIR / 'verify_anonymity_snapshot.py')]
    if zip_path is not None:
        cmd += ['--zip', str(zip_path)]
    subprocess.run(cmd, cwd=ROOT, check=True)


def stage_manifest_files(staging_root: Path) -> list[str]:
    staged: list[str] = []
    for src in iter_manifest_paths():
        rel = src.relative_to(ROOT)
        dst = staging_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        staged.append(str(rel))
    return sorted(staged)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def human_bytes(num: int) -> str:
    units = ['B', 'KB', 'MB', 'GB']
    value = float(num)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f'{value:.1f}{unit}'
        value /= 1024.0
    return f'{num}B'


def build_submission_ready(
    submission_pdf: Path,
    preprint_pdf: Path,
    zip_path: Path,
    sha_path: Path,
    checkpoint_spec: dict[str, str | None],
) -> str:
    lines = [
        '# Submission Ready',
        '',
        f'- Generated at (UTC): `{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")}`',
        f'- Submission PDF: `{submission_pdf.relative_to(ROOT)}`',
        f'- Preprint PDF: `{preprint_pdf.relative_to(ROOT)}`',
        f'- Supplementary ZIP: `{zip_path.relative_to(ROOT)}`',
        f'- SHA256 sums: `{sha_path.relative_to(ROOT)}`',
        f'- Supplementary ZIP size: `{human_bytes(zip_path.stat().st_size)}`',
        '',
        '## Official Submission References',
        '',
        f'- Author instructions: `{OFFICIAL_SUBMISSION_INSTRUCTIONS_URL}`',
        f'- Code/data supplementary policy: `{OFFICIAL_SUPPLEMENTARY_POLICY_URL}`',
        f'- Current package target: `< {human_bytes(MAX_SUPPLEMENTARY_BYTES)}` for the anonymous ZIP unless large files move to an anonymous external URL.',
        '',
        '## Checkpoint Handling',
        '',
        f'- Mode: `{checkpoint_spec["mode"]}`',
    ]
    if checkpoint_spec['mode'] == 'bundle':
        ckpt = Path(str(checkpoint_spec['path']))
        lines += [
            f'- Bundled file: `assets/checkpoints/{ckpt.name}` inside the ZIP',
            f'- SHA256: `{checkpoint_spec["sha256"]}`',
        ]
    elif checkpoint_spec['mode'] == 'link':
        lines += [
            '- The ZIP does not contain the checkpoint blob.',
            f'- Anonymous retrieval URL: `{checkpoint_spec["url"]}`',
            f'- SHA256: `{checkpoint_spec.get("sha256") or "not provided"}`',
        ]
    else:
        lines += [
            '- No checkpoint blob is bundled in the current anonymous snapshot.',
            '- Checklist answers for reproducibility and open access therefore remain conservative.',
        ]
    lines += [
        '',
        '## Verification Path',
        '',
        'Run the repository-level submission check before upload:',
        '',
        '```bash',
        'bash check_repo_health.sh',
        '```',
        '',
        'This validates links, the fastdiscover self-test, a non-mutating dry run, paper metric consistency, page-budget boundaries, source anonymity, ZIP anonymity, and supplementary packaging.',
    ]
    return '\n'.join(lines) + '\n'


def main() -> int:
    ap = argparse.ArgumentParser(description='Create the anonymous NeurIPS supplementary ZIP and submission-ready metadata.')
    ap.add_argument('--mode', choices=['auto', 'omit', 'bundle', 'link'], default='auto')
    ap.add_argument('--checkpoint-path', type=Path, default=None)
    ap.add_argument('--checkpoint-url', default=None)
    ap.add_argument('--checkpoint-sha256', default=None)
    ap.add_argument('--python', default=os.environ.get('PY_BIN', 'python3'))
    ap.add_argument('--skip-anonymity-check', action='store_true')
    ap.add_argument('--allow-oversize', action='store_true', help='allow a ZIP above the NeurIPS 100MB supplementary threshold')
    args = ap.parse_args()

    checkpoint_spec = resolve_checkpoint_spec(args)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    submission_pdf = BUILD_DIR / 'neurips2025-submission.pdf'
    preprint_pdf = BUILD_DIR / 'neurips2025-preprint.pdf'
    if not submission_pdf.exists() or not preprint_pdf.exists():
        raise SystemExit('build PDFs first with ./build.sh submission and ./build.sh preprint')

    if not args.skip_anonymity_check:
        run_anonymity_check(args.python)

    root_name = 'future-seed-neurips2025-supplementary'
    zip_path = DIST_DIR / ZIP_BASENAME
    sha_path = DIST_DIR / SHA256_BASENAME

    with tempfile.TemporaryDirectory(prefix='future_seed_supp_') as tmpdir:
        staging_root = Path(tmpdir) / root_name
        staged_files = stage_manifest_files(staging_root)

        checkpoint_note = {
            'mode': checkpoint_spec['mode'],
            'path': None,
            'url': checkpoint_spec['url'],
            'sha256': checkpoint_spec['sha256'],
        }
        if checkpoint_spec['mode'] == 'bundle':
            ckpt_src = Path(str(checkpoint_spec['path']))
            ckpt_rel = Path('assets/checkpoints') / ckpt_src.name
            ckpt_dst = staging_root / ckpt_rel
            ckpt_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ckpt_src, ckpt_dst)
            checkpoint_note['path'] = str(ckpt_rel)
            staged_files.append(str(ckpt_rel))

        manifest_payload = {
            'generated_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'root': root_name,
            'files': sorted(staged_files),
            'checkpoint': checkpoint_note,
        }
        write_text(staging_root / 'MANIFEST.json', json.dumps(manifest_payload, indent=2, ensure_ascii=False) + '\n')

        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for path in sorted(staging_root.rglob('*')):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(Path(tmpdir))))

    if zip_path.stat().st_size > MAX_SUPPLEMENTARY_BYTES and not args.allow_oversize:
        raise SystemExit(
            f'supplementary ZIP is {human_bytes(zip_path.stat().st_size)}; this exceeds the NeurIPS 100MB limit. '
            'Use checkpoint link mode or rerun with --allow-oversize only if you are intentionally preparing an external-link workflow.'
        )

    if not args.skip_anonymity_check:
        run_anonymity_check(args.python, zip_path=zip_path)

    sha_lines = []
    for artifact in [submission_pdf, preprint_pdf, zip_path]:
        sha_lines.append(f'{sha256_file(artifact)}  {artifact.relative_to(ROOT)}')
    write_text(sha_path, '\n'.join(sha_lines) + '\n')

    submission_ready = ROOT / 'paper' / 'neurips2025' / 'SUBMISSION_READY.md'
    write_text(submission_ready, build_submission_ready(submission_pdf, preprint_pdf, zip_path, sha_path, checkpoint_spec))

    print(f'wrote_zip={zip_path}')
    print(f'wrote_sha256={sha_path}')
    print(f'wrote_submission_ready={submission_ready}')
    print(f'zip_size={human_bytes(zip_path.stat().st_size)}')
    print(f'checkpoint_mode={checkpoint_spec["mode"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
