#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
PY = ROOT / ".venv" / "bin" / "python"
RECORDS = RUNS / "_round40_highvalue_queue_s0_records.jsonl"
SUMMARY = RUNS / "_summary_round40_highvalue_queue_s0.txt"
LOG = RUNS / f"_log_round40_highvalue_queue_s0.{time.strftime('%Y%m%d_%H%M%S')}.log"

SEED = 0


@dataclass
class Candidate:
    name: str
    args: List[str]


@dataclass
class TaskSpec:
    task: str
    train_script: str
    prune: float
    topk: int
    quick_budget: int
    quick_steps: int
    med_budget: int
    med_steps: int
    base_args: List[str]
    candidates: List[Candidate]


def _set_env() -> dict:
    env = os.environ.copy()
    env["TORCH_EXTENSIONS_DIR"] = "/root/autodl-tmp/torch_extensions"
    env["HF_HOME"] = "/root/autodl-tmp/hf"
    env["HF_DATASETS_CACHE"] = "/root/autodl-tmp/hf_datasets"
    env["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/hf_transformers"
    env["HF_ENDPOINT"] = "https://huggingface.co"
    env["HF_DATASETS_OFFLINE"] = "0"
    env["HF_HUB_OFFLINE"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["TORCH_CUDA_ARCH_LIST"] = "8.9"
    return env


def _resolve_train_script(script_name: str) -> Path:
    p1 = ROOT / "scripts" / script_name
    if p1.exists():
        return p1
    p2 = ROOT / script_name
    if p2.exists():
        return p2
    # default to scripts/ path for clearer error log
    return p1


def _parse_best(run_dir: Path) -> tuple[Optional[float], Optional[float]]:
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return None, None
    best_loss = None
    best_acc = None
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "val_loss" in r:
            v = float(r["val_loss"])
            best_loss = v if best_loss is None or v < best_loss else best_loss
        if "val_tok_acc" in r:
            v = float(r["val_tok_acc"])
            best_acc = v if best_acc is None or v > best_acc else best_acc
    return best_loss, best_acc


def _append_record(rec: dict) -> None:
    with RECORDS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_train(
    *,
    env: dict,
    task: str,
    script_name: str,
    stage: str,
    cfg: str,
    mode: str,
    budget: int,
    steps: int,
    base_args: List[str],
    extra_args: List[str],
) -> tuple[Optional[float], Optional[float], str]:
    log = RUNS / f"_round40_{task}_{stage}_{cfg}.log"
    cmd = [
        str(PY),
        str(_resolve_train_script(script_name)),
        "--train_data_seed",
        str(SEED),
        "--val_data_seed",
        "1234",
        "--mode",
        mode,
        "--seed",
        str(SEED),
        "--time_budget_sec",
        str(budget),
        "--max_steps",
        str(steps),
        *base_args,
        *extra_args,
    ]
    with log.open("w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        err = ""
        try:
            tail = log.read_text(encoding="utf-8", errors="ignore").splitlines()[-3:]
            err = " ".join(tail).replace('"', '\\"')
        except Exception:
            pass
        _append_record(
            {
                "task": task,
                "stage": stage,
                "config": cfg,
                "status": "fail",
                "best_val_loss": None,
                "best_val_tok_acc": None,
                "run_dir": "",
                "error": err,
            }
        )
        return None, None, ""

    lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
    run_dir = lines[-1].strip() if lines else ""
    best_loss, best_acc = _parse_best(Path(run_dir)) if run_dir else (None, None)
    _append_record(
        {
            "task": task,
            "stage": stage,
            "config": cfg,
            "status": "ok",
            "best_val_loss": best_loss,
            "best_val_tok_acc": best_acc,
            "run_dir": run_dir,
        }
    )
    return best_loss, best_acc, run_dir


def run_task(spec: TaskSpec, env: dict) -> None:
    b_loss, b_acc, _ = run_train(
        env=env,
        task=spec.task,
        script_name=spec.train_script,
        stage="quick",
        cfg="baseline",
        mode="no_fs",
        budget=spec.quick_budget,
        steps=spec.quick_steps,
        base_args=spec.base_args,
        extra_args=[],
    )
    if b_acc is None:
        return

    kept: List[tuple[str, float, List[str]]] = []
    for c in spec.candidates:
        _, a, _ = run_train(
            env=env,
            task=spec.task,
            script_name=spec.train_script,
            stage="quick",
            cfg=c.name,
            mode="prompt_fs",
            budget=spec.quick_budget,
            steps=spec.quick_steps,
            base_args=spec.base_args,
            extra_args=c.args,
        )
        if a is None:
            continue
        d = float(a) - float(b_acc)
        if d >= spec.prune:
            kept.append((c.name, d, c.args))

    kept.sort(key=lambda x: x[1], reverse=True)
    for name, _, args in kept[: spec.topk]:
        run_train(
            env=env,
            task=spec.task,
            script_name=spec.train_script,
            stage="med",
            cfg=name,
            mode="prompt_fs",
            budget=spec.med_budget,
            steps=spec.med_steps,
            base_args=spec.base_args,
            extra_args=args,
        )


def summarize() -> None:
    rows = []
    if RECORDS.exists():
        for line in RECORDS.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    out: List[str] = []
    out.append("=" * 116)
    out.append("Round40 high-value queue summary")
    out.append("=" * 116)
    for task in sorted({r["task"] for r in rows}):
        out.append(f"[{task}]")
        b = [r for r in rows if r["task"] == task and r["stage"] == "quick" and r["config"] == "baseline" and r["status"] == "ok"]
        if not b:
            out.append("  baseline failed")
            out.append("-" * 116)
            continue
        bacc = float(b[0]["best_val_tok_acc"])
        out.append(f"  baseline quick: acc={bacc*100:.2f}%")
        q = [r for r in rows if r["task"] == task and r["stage"] == "quick" and r["config"] != "baseline" and r["status"] == "ok"]
        for r in sorted(q, key=lambda x: float(x["best_val_tok_acc"]) - bacc, reverse=True):
            d = float(r["best_val_tok_acc"]) - bacc
            out.append(f"    {r['config']:22s} d_acc={d*100:+.2f}pp acc={float(r['best_val_tok_acc'])*100:.2f}%")
        m = [r for r in rows if r["task"] == task and r["stage"] == "med" and r["status"] == "ok"]
        if m:
            out.append("  med:")
            for r in sorted(m, key=lambda x: float(x["best_val_tok_acc"]), reverse=True):
                d = float(r["best_val_tok_acc"]) - bacc
                out.append(f"    {r['config']:22s} d_acc={d*100:+.2f}pp acc={float(r['best_val_tok_acc'])*100:.2f}%")
        else:
            out.append("  med skipped (no config passed prune)")
        out.append("-" * 116)
    SUMMARY.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    RECORDS.write_text("", encoding="utf-8")
    env = _set_env()

    tasks = [
        TaskSpec(
            task="punc_hotpot",
            train_script="train_punc_restore_sft.py",
            prune=0.005,
            topk=2,
            quick_budget=80,
            quick_steps=120,
            med_budget=220,
            med_steps=320,
            base_args=[
                "--ds",
                "hotpot_qa",
                "--ds_cfg",
                "distractor",
                "--train_split",
                "train",
                "--val_split",
                "validation",
                "--n_train",
                "800",
                "--n_val",
                "160",
                "--min_chars",
                "48",
                "--max_chars",
                "220",
                "--fill_notes_to_max",
                "--note_pool_size",
                "1024",
                "--max_prompt_tokens",
                "1536",
                "--min_prompt_tokens",
                "512",
                "--max_answer_tokens",
                "128",
                "--eval_every",
                "20",
                "--val_batches",
                "4",
                "--bsz",
                "4",
            ],
            candidates=[
                Candidate("head_l8", ["--alpha_lr", "0", "--alpha_init", "-3", "--fs_variant", "head", "--alpha_head_init", "-3", "--alpha_head_lr", "5e-4", "--fs_layer_start", "8", "--fs_norm", "--fs_detach", "--fs_clip", "1.0"]),
                Candidate("head_l10", ["--alpha_lr", "0", "--alpha_init", "-3", "--fs_variant", "head", "--alpha_head_init", "-3", "--alpha_head_lr", "5e-4", "--fs_layer_start", "10", "--fs_norm", "--fs_detach", "--fs_clip", "1.0"]),
                Candidate("scalar_l8_sched_cos", ["--alpha_lr", "0", "--alpha_init", "-2", "--fs_variant", "scalar", "--fs_layer_start", "8", "--fs_norm", "--fs_detach", "--fs_clip", "1.0", "--fs_alpha_schedule", "cosine", "--fs_alpha_min", "0.4", "--fs_alpha_max", "1.0"]),
                Candidate("scalar_l10_sched_cos", ["--alpha_lr", "0", "--alpha_init", "-2", "--fs_variant", "scalar", "--fs_layer_start", "10", "--fs_norm", "--fs_detach", "--fs_clip", "1.0", "--fs_alpha_schedule", "cosine", "--fs_alpha_min", "0.4", "--fs_alpha_max", "1.0"]),
            ],
        ),
        TaskSpec(
            task="punc_mbpp",
            train_script="train_punc_restore_sft.py",
            prune=0.005,
            topk=2,
            quick_budget=80,
            quick_steps=120,
            med_budget=220,
            med_steps=320,
            base_args=[
                "--ds",
                "mbpp",
                "--ds_cfg",
                "",
                "--train_split",
                "train",
                "--val_split",
                "test",
                "--n_train",
                "320",
                "--n_val",
                "80",
                "--min_chars",
                "32",
                "--max_chars",
                "360",
                "--fill_notes_to_max",
                "--note_pool_size",
                "512",
                "--max_prompt_tokens",
                "1536",
                "--min_prompt_tokens",
                "512",
                "--max_answer_tokens",
                "160",
                "--eval_every",
                "20",
                "--val_batches",
                "4",
                "--bsz",
                "4",
            ],
            candidates=[
                Candidate("head_l8", ["--alpha_lr", "0", "--alpha_init", "-3", "--fs_variant", "head", "--alpha_head_init", "-3", "--alpha_head_lr", "5e-4", "--fs_layer_start", "8", "--fs_norm", "--fs_detach", "--fs_clip", "1.0"]),
                Candidate("scalar_l8_sched_cos", ["--alpha_lr", "0", "--alpha_init", "-2", "--fs_variant", "scalar", "--fs_layer_start", "8", "--fs_norm", "--fs_detach", "--fs_clip", "1.0", "--fs_alpha_schedule", "cosine", "--fs_alpha_min", "0.4", "--fs_alpha_max", "1.0"]),
                Candidate("scalar_l8_trainable", ["--alpha_lr", "2e-4", "--alpha_init", "-2", "--fs_variant", "scalar", "--fs_layer_start", "8", "--fs_norm", "--fs_detach", "--fs_clip", "1.0"]),
            ],
        ),
        TaskSpec(
            task="punc_squad",
            train_script="train_punc_restore_sft.py",
            prune=0.005,
            topk=2,
            quick_budget=80,
            quick_steps=120,
            med_budget=220,
            med_steps=320,
            base_args=[
                "--ds",
                "squad",
                "--ds_cfg",
                "",
                "--train_split",
                "train",
                "--val_split",
                "validation",
                "--n_train",
                "900",
                "--n_val",
                "180",
                "--min_chars",
                "64",
                "--max_chars",
                "260",
                "--fill_notes_to_max",
                "--note_pool_size",
                "1024",
                "--max_prompt_tokens",
                "1536",
                "--min_prompt_tokens",
                "512",
                "--max_answer_tokens",
                "128",
                "--eval_every",
                "20",
                "--val_batches",
                "4",
                "--bsz",
                "4",
            ],
            candidates=[
                Candidate("head_l8", ["--alpha_lr", "0", "--alpha_init", "-3", "--fs_variant", "head", "--alpha_head_init", "-3", "--alpha_head_lr", "5e-4", "--fs_layer_start", "8", "--fs_norm", "--fs_detach", "--fs_clip", "1.0"]),
                Candidate("scalar_l8_sched_cos", ["--alpha_lr", "0", "--alpha_init", "-2", "--fs_variant", "scalar", "--fs_layer_start", "8", "--fs_norm", "--fs_detach", "--fs_clip", "1.0", "--fs_alpha_schedule", "cosine", "--fs_alpha_min", "0.4", "--fs_alpha_max", "1.0"]),
            ],
        ),
        TaskSpec(
            task="hotpot_longctx_qafter",
            train_script="train_hotpot_longctx_sft.py",
            prune=0.003,
            topk=1,
            quick_budget=90,
            quick_steps=120,
            med_budget=240,
            med_steps=320,
            base_args=[
                "--ds",
                "hotpot_qa",
                "--ds_cfg",
                "distractor",
                "--train_split",
                "train",
                "--val_split",
                "validation",
                "--n_train",
                "800",
                "--n_val",
                "160",
                "--max_prompt_tokens",
                "2048",
                "--min_prompt_tokens",
                "768",
                "--max_answer_tokens",
                "24",
                "--eval_every",
                "25",
                "--val_batches",
                "8",
                "--bsz",
                "2",
            ],
            candidates=[
                Candidate("head_l10", ["--alpha_lr", "0", "--alpha_init", "-3", "--fs_variant", "head", "--alpha_head_init", "-3", "--alpha_head_lr", "5e-4", "--fs_layer_start", "10", "--fs_norm", "--fs_detach", "--fs_clip", "1.0"]),
                Candidate("scalar_l10_sched_cos", ["--alpha_lr", "0", "--alpha_init", "-2", "--fs_variant", "scalar", "--fs_layer_start", "10", "--fs_norm", "--fs_detach", "--fs_clip", "1.0", "--fs_alpha_schedule", "cosine", "--fs_alpha_min", "0.4", "--fs_alpha_max", "1.0"]),
            ],
        ),
        TaskSpec(
            task="mbpp_longctx_qafter",
            train_script="train_mbpp_longctx_sft.py",
            prune=0.003,
            topk=1,
            quick_budget=90,
            quick_steps=120,
            med_budget=240,
            med_steps=320,
            base_args=[
                "--ds",
                "mbpp",
                "--ds_cfg",
                "",
                "--train_split",
                "train",
                "--val_split",
                "test",
                "--n_train",
                "320",
                "--n_val",
                "120",
                "--fill_notes_to_max",
                "--note_pool_size",
                "1024",
                "--max_prompt_tokens",
                "3072",
                "--min_prompt_tokens",
                "1024",
                "--max_answer_tokens",
                "160",
                "--eval_every",
                "25",
                "--val_batches",
                "8",
                "--bsz",
                "1",
            ],
            candidates=[
                Candidate("head_l8", ["--alpha_lr", "0", "--alpha_init", "-3", "--fs_variant", "head", "--alpha_head_init", "-3", "--alpha_head_lr", "5e-4", "--fs_layer_start", "8", "--fs_norm", "--fs_detach", "--fs_clip", "1.0"]),
                Candidate("scalar_l8_sched_cos", ["--alpha_lr", "0", "--alpha_init", "-2", "--fs_variant", "scalar", "--fs_layer_start", "8", "--fs_norm", "--fs_detach", "--fs_clip", "1.0", "--fs_alpha_schedule", "cosine", "--fs_alpha_min", "0.4", "--fs_alpha_max", "1.0"]),
            ],
        ),
    ]

    with LOG.open("w", encoding="utf-8") as lf:
        lf.write(f"Round40 started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    for spec in tasks:
        run_task(spec, env)
    summarize()
    with LOG.open("a", encoding="utf-8") as lf:
        lf.write(f"Round40 finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
