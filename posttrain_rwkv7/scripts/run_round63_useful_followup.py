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
RECORDS = RUNS / '_round63_useful_followup_records.jsonl'
SUMMARY = RUNS / '_summary_round63_useful_followup.txt'


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
    med_gate_pp: float = 0.2
    run_med: bool = True


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
    log = RUNS / f'_round63_{task.name}_{stage}_{cfg}.log'
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

    best = None
    for c in task.cands:
        a = run_one(task, 'quick', c.name, 'prompt_fs', task.quick_budget, task.quick_steps, task.quick_eval_every, task.quick_val_batches, c.args)
        if a is None:
            continue
        dpp = (a - b) * 100.0
        if dpp < task.prune_drop_pp:
            append({'task': task.name, 'stage': 'quick_decision', 'config': c.name, 'status': 'pruned', 'reason': f'quick_drop {dpp:+.2f}pp < {task.prune_drop_pp:+.2f}pp'})
        if best is None or a > best[1]:
            best = (c, a)

    if best is None or not task.run_med:
        return

    best_dpp = (best[1] - b) * 100.0
    if best_dpp < task.med_gate_pp:
        append({'task': task.name, 'stage': 'med_decision', 'config': best[0].name, 'status': 'pruned', 'reason': f'best_quick {best_dpp:+.2f}pp < med_gate {task.med_gate_pp:+.2f}pp'})
        return

    run_one(task, 'med', 'baseline', 'no_fs', task.med_budget, task.med_steps, task.med_eval_every, task.med_val_batches, [])
    run_one(task, 'med', best[0].name, 'prompt_fs', task.med_budget, task.med_steps, task.med_eval_every, task.med_val_batches, best[0].args)


def summarize() -> None:
    rows = [json.loads(x) for x in RECORDS.read_text(encoding='utf-8', errors='ignore').splitlines() if x.strip()]
    out = ['=' * 112, 'Round63 useful-followup summary', '=' * 112]

    ranking = []

    for task in sorted({r['task'] for r in rows if 'task' in r}):
        out.append(f'[{task}]')

        bq = [r for r in rows if r.get('task') == task and r.get('stage') == 'quick' and r.get('config') == 'baseline' and r.get('status') == 'ok']
        if not bq:
            out.append('  quick baseline failed')
            out.append('-' * 112)
            continue
        bq_acc = float(bq[0]['best_val_tok_acc'])
        out.append(f'  quick baseline: {bq_acc*100:.2f}%')

        qfs = [r for r in rows if r.get('task') == task and r.get('stage') == 'quick' and r.get('config') != 'baseline' and r.get('status') == 'ok']
        for r in sorted(qfs, key=lambda x: float(x['best_val_tok_acc']), reverse=True):
            dpp = (float(r['best_val_tok_acc']) - bq_acc) * 100.0
            out.append(f"    {r['config']:28s} d_acc={dpp:+.2f}pp acc={float(r['best_val_tok_acc'])*100:.2f}%")

        mb = [r for r in rows if r.get('task') == task and r.get('stage') == 'med' and r.get('config') == 'baseline' and r.get('status') == 'ok']
        mf = [r for r in rows if r.get('task') == task and r.get('stage') == 'med' and r.get('config') != 'baseline' and r.get('status') == 'ok']
        if mb and mf:
            bmed = float(mb[0]['best_val_tok_acc'])
            best_med = max(mf, key=lambda x: float(x['best_val_tok_acc']))
            fmed = float(best_med['best_val_tok_acc'])
            dmed = (fmed - bmed) * 100.0
            out.append('  med:')
            out.append(f'    baseline                     acc={bmed*100:.2f}%')
            out.append(f"    {best_med['config']:28s} d_acc_vs_med={dmed:+.2f}pp acc={fmed*100:.2f}%")
            ranking.append((task, 'med', best_med['config'], dmed))
        else:
            md = [r for r in rows if r.get('task') == task and r.get('stage') == 'med_decision' and r.get('status') == 'pruned']
            if md:
                out.append(f"  med skipped: {md[-1].get('reason', 'pruned')}")
            else:
                out.append('  med skipped')
            if qfs:
                best_q = max(qfs, key=lambda x: float(x['best_val_tok_acc']))
                dq = (float(best_q['best_val_tok_acc']) - bq_acc) * 100.0
                ranking.append((task, 'quick', best_q['config'], dq))

        pruned = [r for r in rows if r.get('task') == task and r.get('status') == 'pruned']
        if pruned:
            out.append('  pruned:')
            for p in pruned:
                out.append(f"    {p.get('config','?'):28s} {p.get('reason','')}")

        out.append('-' * 112)

    out.append('[useful_task_ranking_this_round]')
    for t, st, cfg, d in sorted(ranking, key=lambda x: x[3], reverse=True):
        out.append(f'  {t:30s} stage={st:5s} cfg={cfg:24s} d_acc={d:+.2f}pp')

    SUMMARY.write_text('\n'.join(out) + '\n', encoding='utf-8')
    print('\n'.join(out))


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    RECORDS.write_text('', encoding='utf-8')

    punc_seed1 = Task(
        name='punc_restore_seed1_confirm',
        seed=1,
        base_args=['--ds','hotpot_qa','--ds_cfg','distractor','--train_split','train','--val_split','validation','--n_train','800','--n_val','160','--min_chars','48','--max_chars','220','--fill_notes_to_max','--note_pool_size','1024','--max_prompt_tokens','1536','--min_prompt_tokens','512','--max_answer_tokens','128','--bsz','2'],
        cands=[
            Candidate('head_l8',['--alpha_lr','0','--alpha_init','-3','--fs_variant','head','--alpha_head_init','-3','--alpha_head_lr','5e-4','--fs_layer_start','8','--fs_norm','--fs_detach','--fs_clip','1.0']),
            Candidate('scalar_l8_sched_cos',['--alpha_lr','0','--alpha_init','-2','--fs_variant','scalar','--fs_layer_start','8','--fs_norm','--fs_detach','--fs_clip','1.0','--fs_alpha_schedule','cosine','--fs_alpha_min','0.4','--fs_alpha_max','1.0']),
        ],
        quick_budget=120, quick_steps=160, quick_eval_every=20, quick_val_batches=3,
        med_budget=260, med_steps=400, med_eval_every=20, med_val_batches=3,
        prune_drop_pp=-0.5, med_gate_pp=0.2,
    )

    punc_seed2 = Task(
        name='punc_restore_seed2_confirm',
        seed=2,
        base_args=['--ds','hotpot_qa','--ds_cfg','distractor','--train_split','train','--val_split','validation','--n_train','800','--n_val','160','--min_chars','48','--max_chars','220','--fill_notes_to_max','--note_pool_size','1024','--max_prompt_tokens','1536','--min_prompt_tokens','512','--max_answer_tokens','128','--bsz','2'],
        cands=[
            Candidate('head_l8',['--alpha_lr','0','--alpha_init','-3','--fs_variant','head','--alpha_head_init','-3','--alpha_head_lr','5e-4','--fs_layer_start','8','--fs_norm','--fs_detach','--fs_clip','1.0']),
            Candidate('scalar_l8_sched_cos',['--alpha_lr','0','--alpha_init','-2','--fs_variant','scalar','--fs_layer_start','8','--fs_norm','--fs_detach','--fs_clip','1.0','--fs_alpha_schedule','cosine','--fs_alpha_min','0.4','--fs_alpha_max','1.0']),
        ],
        quick_budget=120, quick_steps=160, quick_eval_every=20, quick_val_batches=3,
        med_budget=260, med_steps=400, med_eval_every=20, med_val_batches=3,
        prune_drop_pp=-0.5, med_gate_pp=0.2,
    )

    mbpp_seed2_rescue = Task(
        name='mbpp_seed2_regrescue',
        seed=2,
        base_args=['--ds','mbpp','--ds_cfg','','--train_split','train','--val_split','test','--n_train','340','--n_val','100','--min_chars','24','--max_chars','360','--fill_notes_to_max','--note_pool_size','384','--max_prompt_tokens','1408','--min_prompt_tokens','384','--max_answer_tokens','160','--bsz','12'],
        cands=[
            Candidate('head_l10_sched_soft',['--alpha_lr','0','--alpha_init','-2','--fs_variant','head','--alpha_head_init','-2','--alpha_head_lr','1e-3','--fs_layer_start','10','--fs_norm','--fs_detach','--fs_clip','1.0','--fs_alpha_schedule','cosine','--fs_alpha_min','0.8','--fs_alpha_max','1.0']),
            Candidate('head_l10_clip07',['--alpha_lr','0','--alpha_init','-2','--fs_variant','head','--alpha_head_init','-2','--alpha_head_lr','1e-3','--fs_layer_start','10','--fs_norm','--fs_detach','--fs_clip','0.7']),
            Candidate('head_l11_strong',['--alpha_lr','0','--alpha_init','-2','--fs_variant','head','--alpha_head_init','-2','--alpha_head_lr','1e-3','--fs_layer_start','11','--fs_norm','--fs_detach','--fs_clip','1.0']),
        ],
        quick_budget=160, quick_steps=160, quick_eval_every=20, quick_val_batches=2,
        med_budget=300, med_steps=320, med_eval_every=20, med_val_batches=3,
        prune_drop_pp=-0.5, med_gate_pp=0.2,
    )

    # Prioritize fastest usefulness confirmation first.
    run_task(punc_seed1)
    run_task(punc_seed2)
    run_task(mbpp_seed2_rescue)
    summarize()


if __name__ == '__main__':
    main()
