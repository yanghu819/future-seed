#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
PYTHON = ROOT / '.venv' / 'bin' / 'python'
RECORDS = RUNS / '_round61_strict_seed1_frontier_records.jsonl'
SUMMARY = RUNS / '_summary_round61_strict_seed1_frontier.txt'


@dataclass
class Candidate:
    name: str
    args: List[str]


@dataclass
class Task:
    name: str
    base_args: List[str]
    cands: List[Candidate]
    prune_pp: float = 0.30


def env() -> dict:
    e = os.environ.copy()
    e['TORCH_EXTENSIONS_DIR'] = '/root/autodl-tmp/torch_extensions'
    e['HF_HOME'] = '/root/autodl-tmp/hf'
    e['HF_DATASETS_CACHE'] = '/root/autodl-tmp/hf_datasets'
    e['TRANSFORMERS_CACHE'] = '/root/autodl-tmp/hf_transformers'
    e['HF_ENDPOINT'] = 'https://huggingface.co'
    e['HF_DATASETS_OFFLINE'] = '1'
    e['HF_HUB_OFFLINE'] = '1'
    e['PYTHONUNBUFFERED'] = '1'
    e['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    e['TORCH_CUDA_ARCH_LIST'] = '8.9'
    return e


def train_script() -> Path:
    p = ROOT / 'scripts' / 'train_punc_restore_sft.py'
    return p if p.exists() else ROOT / 'train_punc_restore_sft.py'


def append(rec: dict) -> None:
    with RECORDS.open('a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def best_metrics(run_dir: Path) -> tuple[Optional[float], Optional[float]]:
    p = run_dir / 'metrics.jsonl'
    if not p.exists():
        return None, None
    best_loss = None
    best_acc = None
    for line in p.read_text(encoding='utf-8', errors='ignore').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if 'val_loss' in r:
            v = float(r['val_loss'])
            best_loss = v if best_loss is None or v < best_loss else best_loss
        if 'val_tok_acc' in r:
            v = float(r['val_tok_acc'])
            best_acc = v if best_acc is None or v > best_acc else best_acc
    return best_loss, best_acc


def run(task: Task, stage: str, cfg: str, mode: str, budget: int, steps: int, eval_every: int, val_batches: int, extra: List[str]) -> Optional[float]:
    log = RUNS / f'_round61_{task.name}_{stage}_{cfg}.log'
    cmd = [
        str(PYTHON),
        str(train_script()),
        '--train_data_seed', '1',
        '--val_data_seed', '1234',
        '--mode', mode,
        '--seed', '1',
        '--time_budget_sec', str(budget),
        '--max_steps', str(steps),
        '--eval_every', str(eval_every),
        '--val_batches', str(val_batches),
        '--model_lr', '3e-5',
        '--seed_scale', '1.0',
        *task.base_args,
        *extra,
    ]
    with log.open('w', encoding='utf-8') as lf:
        p = subprocess.run(cmd, cwd=ROOT, env=env(), stdout=lf, stderr=subprocess.STDOUT)

    lines = log.read_text(encoding='utf-8', errors='ignore').splitlines()
    if p.returncode != 0:
        append({
            'task': task.name,
            'stage': stage,
            'config': cfg,
            'status': 'fail',
            'best_val_loss': None,
            'best_val_tok_acc': None,
            'run_dir': '',
            'error': 'rc_nonzero',
        })
        return None

    run_dir = lines[-1].strip() if lines else ''
    best_loss, best_acc = best_metrics(Path(run_dir)) if run_dir else (None, None)
    if best_acc is None:
        append({
            'task': task.name,
            'stage': stage,
            'config': cfg,
            'status': 'fail',
            'best_val_loss': best_loss,
            'best_val_tok_acc': None,
            'run_dir': run_dir,
            'error': 'no_val_metric',
        })
        return None

    append({
        'task': task.name,
        'stage': stage,
        'config': cfg,
        'status': 'ok',
        'best_val_loss': best_loss,
        'best_val_tok_acc': best_acc,
        'run_dir': run_dir,
    })
    return best_acc


def run_task(task: Task) -> None:
    b = run(task, 'quick', 'baseline', 'no_fs', 150, 150, 20, 2, [])
    if b is None:
        return

    best = None
    for c in task.cands:
        a = run(task, 'quick', c.name, 'prompt_fs', 150, 150, 20, 2, c.args)
        if a is None:
            continue
        if best is None or a > best[1]:
            best = (c, a)

    if best is None:
        return

    dpp = (best[1] - b) * 100.0
    if dpp < task.prune_pp:
        append({
            'task': task.name,
            'stage': 'decision',
            'config': best[0].name,
            'status': 'pruned',
            'best_val_loss': None,
            'best_val_tok_acc': None,
            'run_dir': '',
            'reason': f'quick_d_acc={dpp:+.2f}pp < prune_pp={task.prune_pp:.2f}pp',
        })
        return

    run(task, 'med', 'baseline', 'no_fs', 280, 300, 20, 3, [])
    run(task, 'med', best[0].name, 'prompt_fs', 280, 300, 20, 3, best[0].args)


def summarize() -> None:
    rows = [json.loads(x) for x in RECORDS.read_text(encoding='utf-8', errors='ignore').splitlines() if x.strip()]
    out = ['=' * 112, 'Round61 strict seed1 frontier summary', '=' * 112]
    for t in sorted({r['task'] for r in rows if r.get('task')}):
        out.append(f'[{t}]')
        bq = [
            r for r in rows
            if r.get('task') == t and r.get('stage') == 'quick' and r.get('config') == 'baseline' and r.get('status') == 'ok'
        ]
        if not bq:
            out.append('  quick baseline failed')
            out.append('-' * 112)
            continue

        bq_acc = float(bq[0]['best_val_tok_acc'])
        out.append(f'  quick baseline: {bq_acc * 100:.2f}%')
        qfs = [
            r for r in rows
            if r.get('task') == t and r.get('stage') == 'quick' and r.get('config') != 'baseline' and r.get('status') == 'ok'
        ]
        for r in sorted(qfs, key=lambda x: float(x['best_val_tok_acc']), reverse=True):
            d = (float(r['best_val_tok_acc']) - bq_acc) * 100.0
            out.append(f"    {r['config']:28s} d_acc={d:+.2f}pp acc={float(r['best_val_tok_acc']) * 100:.2f}%")

        mb = [
            r for r in rows
            if r.get('task') == t and r.get('stage') == 'med' and r.get('config') == 'baseline' and r.get('status') == 'ok'
        ]
        mf = [
            r for r in rows
            if r.get('task') == t and r.get('stage') == 'med' and r.get('config') != 'baseline' and r.get('status') == 'ok'
        ]
        if mb and mf:
            mb_acc = float(mb[0]['best_val_tok_acc'])
            best_med = max(mf, key=lambda x: float(x['best_val_tok_acc']))
            best_med_acc = float(best_med['best_val_tok_acc'])
            d_vs_med = (best_med_acc - mb_acc) * 100.0
            d_vs_qb = (best_med_acc - bq_acc) * 100.0
            out.append('  med:')
            out.append(f"    baseline                     acc={mb_acc * 100:.2f}%")
            out.append(
                f"    {best_med['config']:28s} d_acc_vs_med_base={d_vs_med:+.2f}pp "
                f"d_acc_vs_quick_base={d_vs_qb:+.2f}pp acc={best_med_acc * 100:.2f}%"
            )
        else:
            pruned = [
                r for r in rows
                if r.get('task') == t and r.get('stage') == 'decision' and r.get('status') == 'pruned'
            ]
            if pruned:
                out.append(f"  med skipped: {pruned[-1].get('reason', 'pruned')}")
            else:
                out.append('  med skipped')

        out.append('-' * 112)

    SUMMARY.write_text('\n'.join(out) + '\n', encoding='utf-8')
    print('\n'.join(out))


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    RECORDS.write_text('', encoding='utf-8')

    squad = Task(
        name='squad_strict_seed1',
        base_args=[
            '--ds', 'squad', '--ds_cfg', '', '--train_split', 'train', '--val_split', 'validation',
            '--n_train', '1200', '--n_val', '160', '--min_chars', '64', '--max_chars', '260',
            '--fill_notes_to_max', '--note_pool_size', '1024', '--max_prompt_tokens', '1536',
            '--min_prompt_tokens', '512', '--max_answer_tokens', '128', '--bsz', '12',
        ],
        cands=[
            Candidate('scalar_l8_sched_cos', [
                '--alpha_lr', '0', '--alpha_init', '-2', '--fs_variant', 'scalar', '--fs_layer_start', '8',
                '--fs_norm', '--fs_detach', '--fs_clip', '1.0',
                '--fs_alpha_schedule', 'cosine', '--fs_alpha_min', '0.4', '--fs_alpha_max', '1.0',
            ]),
            Candidate('scalar_l8_sched_cos_highfloor', [
                '--alpha_lr', '0', '--alpha_init', '-2', '--fs_variant', 'scalar', '--fs_layer_start', '8',
                '--fs_norm', '--fs_detach', '--fs_clip', '1.0',
                '--fs_alpha_schedule', 'cosine', '--fs_alpha_min', '0.6', '--fs_alpha_max', '1.0',
            ]),
            Candidate('scalar_l8_train1e4', [
                '--alpha_lr', '1e-4', '--alpha_init', '-2', '--fs_variant', 'scalar', '--fs_layer_start', '8',
                '--fs_norm', '--fs_detach', '--fs_clip', '1.0',
            ]),
        ],
        prune_pp=0.30,
    )

    mbpp = Task(
        name='mbpp_strict_seed1',
        base_args=[
            '--ds', 'mbpp', '--ds_cfg', '', '--train_split', 'train', '--val_split', 'test',
            '--n_train', '340', '--n_val', '100', '--min_chars', '24', '--max_chars', '360',
            '--fill_notes_to_max', '--note_pool_size', '512', '--max_prompt_tokens', '1536',
            '--min_prompt_tokens', '384', '--max_answer_tokens', '160', '--bsz', '12',
        ],
        cands=[
            Candidate('head_l10_strong', [
                '--alpha_lr', '0', '--alpha_init', '-2', '--fs_variant', 'head',
                '--alpha_head_init', '-2', '--alpha_head_lr', '1e-3', '--fs_layer_start', '10',
                '--fs_norm', '--fs_detach', '--fs_clip', '1.0',
            ]),
            Candidate('head_l10_midlr', [
                '--alpha_lr', '0', '--alpha_init', '-2', '--fs_variant', 'head',
                '--alpha_head_init', '-2', '--alpha_head_lr', '7e-4', '--fs_layer_start', '10',
                '--fs_norm', '--fs_detach', '--fs_clip', '1.0',
            ]),
            Candidate('head_l10_sched_cos', [
                '--alpha_lr', '0', '--alpha_init', '-2', '--fs_variant', 'head',
                '--alpha_head_init', '-2', '--alpha_head_lr', '1e-3', '--fs_layer_start', '10',
                '--fs_norm', '--fs_detach', '--fs_clip', '1.0',
                '--fs_alpha_schedule', 'cosine', '--fs_alpha_min', '0.5', '--fs_alpha_max', '1.0',
            ]),
        ],
        prune_pp=0.30,
    )

    run_task(squad)
    run_task(mbpp)
    summarize()


if __name__ == '__main__':
    main()
