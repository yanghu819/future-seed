#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
PYTHON = ROOT / '.venv' / 'bin' / 'python'
RECORDS = RUNS / '_round71_mbpp3_squad2_records.jsonl'
SUMMARY = RUNS / '_summary_round71_mbpp3_squad2.txt'


@dataclass
class Candidate:
    name: str
    args: List[str]


@dataclass
class Task:
    name: str
    seed: int
    base_args: List[str]
    cands: List[Candidate]
    quick_budget: int
    quick_steps: int
    quick_eval_every: int
    quick_val_batches: int
    med_budget: int
    med_steps: int
    med_eval_every: int
    med_val_batches: int
    prune_drop_pp: float = -0.5
    med_gate_pp: float = 0.0
    promote_topk: int = 1


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


def run_one(task: Task, stage: str, cfg: str, mode: str, budget: int, steps: int, eval_every: int, val_batches: int, extra: List[str]) -> Optional[float]:
    log = RUNS / f'_round71_{task.name}_{stage}_{cfg}.log'
    cmd = [
        str(PYTHON), str(train_script()),
        '--train_data_seed', str(task.seed), '--val_data_seed', '1234', '--mode', mode, '--seed', str(task.seed),
        '--time_budget_sec', str(budget), '--max_steps', str(steps), '--eval_every', str(eval_every), '--val_batches', str(val_batches),
        '--model_lr', '3e-5', '--seed_scale', '1.0',
        *task.base_args, *extra,
    ]

    with log.open('w', encoding='utf-8') as lf:
        p = subprocess.run(cmd, cwd=ROOT, env=env(), stdout=lf, stderr=subprocess.STDOUT)

    lines = log.read_text(encoding='utf-8', errors='ignore').splitlines()
    if p.returncode != 0:
        append({'task': task.name, 'stage': stage, 'config': cfg, 'status': 'fail', 'best_val_loss': None, 'best_val_tok_acc': None, 'run_dir': '', 'error': 'rc_nonzero'})
        return None

    rd = lines[-1].strip() if lines else ''
    bl, ba = best_metrics(Path(rd)) if rd else (None, None)
    if ba is None:
        append({'task': task.name, 'stage': stage, 'config': cfg, 'status': 'fail', 'best_val_loss': bl, 'best_val_tok_acc': None, 'run_dir': rd, 'error': 'no_val_metric'})
        return None

    append({'task': task.name, 'stage': stage, 'config': cfg, 'status': 'ok', 'best_val_loss': bl, 'best_val_tok_acc': ba, 'run_dir': rd})
    return ba


def run_task(task: Task) -> None:
    b = run_one(task, 'quick', 'baseline', 'no_fs', task.quick_budget, task.quick_steps, task.quick_eval_every, task.quick_val_batches, [])
    if b is None:
        return

    scored: List[Tuple[Candidate, float, float]] = []
    for c in task.cands:
        a = run_one(task, 'quick', c.name, 'prompt_fs', task.quick_budget, task.quick_steps, task.quick_eval_every, task.quick_val_batches, c.args)
        if a is None:
            continue
        dpp = (a - b) * 100.0
        scored.append((c, a, dpp))
        if dpp < task.prune_drop_pp:
            append({'task': task.name, 'stage': 'quick_decision', 'config': c.name, 'status': 'pruned', 'reason': f'quick_drop {dpp:+.2f}pp < {task.prune_drop_pp:+.2f}pp'})

    if not scored:
        return

    promoted = [x for x in sorted(scored, key=lambda z: z[1], reverse=True) if x[2] >= task.med_gate_pp]
    promoted = promoted[: task.promote_topk]

    if not promoted:
        best = max(scored, key=lambda z: z[1])
        append({'task': task.name, 'stage': 'med_decision', 'config': best[0].name, 'status': 'pruned', 'reason': f'best_quick {best[2]:+.2f}pp < med_gate {task.med_gate_pp:+.2f}pp'})
        return

    run_one(task, 'med', 'baseline', 'no_fs', task.med_budget, task.med_steps, task.med_eval_every, task.med_val_batches, [])
    for c, _a, _dpp in promoted:
        run_one(task, 'med', c.name, 'prompt_fs', task.med_budget, task.med_steps, task.med_eval_every, task.med_val_batches, c.args)


def summarize() -> None:
    rows = [json.loads(x) for x in RECORDS.read_text(encoding='utf-8', errors='ignore').splitlines() if x.strip()]
    out = ['=' * 112, 'Round71 mbpp3+squad2 summary', '=' * 112]
    rank = []

    for task in sorted({r['task'] for r in rows if 'task' in r}):
        out.append(f'[{task}]')
        bq = [r for r in rows if r.get('task') == task and r.get('stage') == 'quick' and r.get('config') == 'baseline' and r.get('status') == 'ok']
        if not bq:
            out.append('  quick baseline failed')
            out.append('-' * 112)
            continue
        bq_acc = float(bq[0]['best_val_tok_acc'])
        out.append(f'  quick baseline: {bq_acc * 100:.2f}%')

        qfs = [r for r in rows if r.get('task') == task and r.get('stage') == 'quick' and r.get('config') != 'baseline' and r.get('status') == 'ok']
        for r in sorted(qfs, key=lambda x: float(x['best_val_tok_acc']), reverse=True):
            dpp = (float(r['best_val_tok_acc']) - bq_acc) * 100.0
            out.append(f"    {r['config']:28s} d_acc={dpp:+.2f}pp acc={float(r['best_val_tok_acc']) * 100:.2f}%")

        mb = [r for r in rows if r.get('task') == task and r.get('stage') == 'med' and r.get('config') == 'baseline' and r.get('status') == 'ok']
        mf = [r for r in rows if r.get('task') == task and r.get('stage') == 'med' and r.get('config') != 'baseline' and r.get('status') == 'ok']
        if mb and mf:
            bmed = float(mb[0]['best_val_tok_acc'])
            out.append('  med:')
            out.append(f'    baseline                     acc={bmed * 100:.2f}%')
            for r in sorted(mf, key=lambda x: float(x['best_val_tok_acc']), reverse=True):
                fmed = float(r['best_val_tok_acc'])
                dmed = (fmed - bmed) * 100.0
                out.append(f"    {r['config']:28s} d_acc_vs_med={dmed:+.2f}pp acc={fmed * 100:.2f}%")
                rank.append((task, 'med', r['config'], dmed))
        else:
            md = [r for r in rows if r.get('task') == task and r.get('stage') == 'med_decision' and r.get('status') == 'pruned']
            if md:
                out.append(f"  med skipped: {md[-1].get('reason', 'pruned')}")
            else:
                out.append('  med skipped')
            if qfs:
                best_q = max(qfs, key=lambda x: float(x['best_val_tok_acc']))
                dq = (float(best_q['best_val_tok_acc']) - bq_acc) * 100.0
                rank.append((task, 'quick', best_q['config'], dq))

        pruned = [r for r in rows if r.get('task') == task and r.get('status') == 'pruned']
        if pruned:
            out.append('  pruned:')
            for p in pruned:
                out.append(f"    {p.get('config', '?'):28s} {p.get('reason', '')}")
        out.append('-' * 112)

    out.append('[useful_task_ranking_this_round]')
    for t, st, cfg, d in sorted(rank, key=lambda x: x[3], reverse=True):
        out.append(f'  {t:32s} stage={st:5s} cfg={cfg:24s} d_acc={d:+.2f}pp')

    SUMMARY.write_text('\n'.join(out) + '\n', encoding='utf-8')
    print('\n'.join(out))


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    RECORDS.write_text('', encoding='utf-8')

    mbpp_seed3 = Task(
        name='mbpp_seed3_rescue384',
        seed=3,
        base_args=['--ds','mbpp','--ds_cfg','','--train_split','train','--val_split','test','--n_train','340','--n_val','100','--min_chars','24','--max_chars','360','--fill_notes_to_max','--note_pool_size','384','--max_prompt_tokens','1408','--min_prompt_tokens','384','--max_answer_tokens','160','--bsz','12'],
        cands=[
            Candidate('head_l10_clip07',['--alpha_lr','0','--alpha_init','-2','--fs_variant','head','--alpha_head_init','-2','--alpha_head_lr','1e-3','--fs_layer_start','10','--fs_norm','--fs_detach','--fs_clip','0.7']),
            Candidate('head_l10_strong',['--alpha_lr','0','--alpha_init','-2','--fs_variant','head','--alpha_head_init','-2','--alpha_head_lr','1e-3','--fs_layer_start','10','--fs_norm','--fs_detach','--fs_clip','1.0']),
        ],
        quick_budget=160, quick_steps=160, quick_eval_every=20, quick_val_batches=2,
        med_budget=300, med_steps=320, med_eval_every=20, med_val_batches=3,
        prune_drop_pp=-0.5, med_gate_pp=0.2, promote_topk=2,
    )

    squad_seed2 = Task(
        name='squad_seed2_scalar_reconfirm',
        seed=2,
        base_args=['--ds','squad','--ds_cfg','','--train_split','train','--val_split','validation','--n_train','1200','--n_val','160','--min_chars','64','--max_chars','260','--fill_notes_to_max','--note_pool_size','1024','--max_prompt_tokens','1536','--min_prompt_tokens','512','--max_answer_tokens','128','--bsz','12'],
        cands=[
            Candidate('scalar_l8_train1e4',['--alpha_lr','1e-4','--alpha_init','-2','--fs_variant','scalar','--fs_layer_start','8','--fs_norm','--fs_detach','--fs_clip','1.0']),
            Candidate('scalar_l8_train8e5',['--alpha_lr','8e-5','--alpha_init','-2','--fs_variant','scalar','--fs_layer_start','8','--fs_norm','--fs_detach','--fs_clip','1.0']),
            Candidate('scalar_l8_train1e4_clip07',['--alpha_lr','1e-4','--alpha_init','-2','--fs_variant','scalar','--fs_layer_start','8','--fs_norm','--fs_detach','--fs_clip','0.7']),
        ],
        quick_budget=150, quick_steps=160, quick_eval_every=20, quick_val_batches=2,
        med_budget=280, med_steps=320, med_eval_every=20, med_val_batches=3,
        prune_drop_pp=-0.5, med_gate_pp=-0.1, promote_topk=1,
    )

    run_task(mbpp_seed3)
    run_task(squad_seed2)
    summarize()


if __name__ == '__main__':
    main()
