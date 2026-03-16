#!/usr/bin/env python3
"""Orchestrate the 9x9 Sudoku in-place repair mainline benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
TRAIN = SCRIPTS / "train_sudoku9_inplace_refine.py"
BUILD = SCRIPTS / "build_sudoku9_inplace_manifests.py"
SUMMARY = SCRIPTS / "summarize_sudoku9_inplace.py"
DEFAULT_ASSETS = REPO / "assets" / "sudoku9_inplace"
DEFAULT_TRAIN_MASKS = "32,36"
DEFAULT_FOCUS_MASKS = "32"
DEFAULT_EVAL_MASKS = "28,32,36,40"

CONFIGS = {
    "baseline": ["--mode", "no_fs", "--fs_variant", "scalar", "--alpha_lr", "0", "--alpha_init", "-2", "--fs_layer_start", "6", "--fs_norm", "--fs_detach", "--fs_clip", "1.0"],
    "fs_scalar_l6_cos": ["--mode", "prompt_fs", "--fs_variant", "scalar", "--alpha_lr", "0", "--alpha_init", "-2", "--fs_layer_start", "6", "--fs_alpha_schedule", "cosine", "--fs_alpha_min", "0.4", "--fs_alpha_max", "1.0", "--fs_norm", "--fs_detach", "--fs_clip", "1.0"],
    "fs_scalar_l8_cos": ["--mode", "prompt_fs", "--fs_variant", "scalar", "--alpha_lr", "0", "--alpha_init", "-2", "--fs_layer_start", "8", "--fs_alpha_schedule", "cosine", "--fs_alpha_min", "0.4", "--fs_alpha_max", "1.0", "--fs_norm", "--fs_detach", "--fs_clip", "1.0"],
}
PHASE_A_CONFIGS = ["baseline", "fs_scalar_l6_cos", "fs_scalar_l8_cos"]
SMOKE_CONFIGS = ["baseline", "fs_scalar_l6_cos"]
DEFAULT_PHASE_B_WINNER = "fs_scalar_l6_cos"


def run(cmd: list[str], *, dry_run: bool) -> None:
    print("RUN", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def parse_int_list(text: str) -> list[int]:
    vals = [int(x) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("expected at least one integer")
    return vals


def manifest_tag(*, smoke: bool, seed: int, masks: list[int]) -> str:
    if smoke:
        return "smoke"
    return "seed" + str(int(seed))


def ensure_manifests(*, assets_dir: Path, smoke: bool, dry_run: bool, workers: int, masks: list[int], seed: int) -> tuple[Path, Path]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    tag = manifest_tag(smoke=smoke, seed=seed, masks=masks)
    val = assets_dir / f"val_{tag}.jsonl"
    test = assets_dir / f"test_{tag}.jsonl"
    if val.exists() and test.exists():
        return val, test
    cmd = [
        sys.executable,
        str(BUILD),
        "--out_dir",
        str(assets_dir),
        "--workers",
        str(workers),
        "--mask_counts",
        ",".join(str(x) for x in masks),
        "--seed",
        str(seed),
    ]
    if smoke:
        cmd.append("--smoke")
    run(cmd, dry_run=dry_run)
    return val, test


def best_metrics_record(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.jsonl"
    best: dict | None = None
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (
                float(row.get("val_focus_board_exact_mean", 0.0)),
                float(row.get("val_focus_masked_acc_mean", 0.0)),
            )
            if best is None:
                best = row
                continue
            best_key = (
                float(best.get("val_focus_board_exact_mean", 0.0)),
                float(best.get("val_focus_masked_acc_mean", 0.0)),
            )
            if key > best_key:
                best = row
    return best or {}


def latest_run_dirs(root: Path, tags: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in root.glob("**/sudoku9_inplace_refine/*/summary.json"):
        tag = path.parent.name
        if tag not in tags:
            continue
        if tag not in out or str(path.parent) > str(out[tag]):
            out[tag] = path.parent
    missing = [tag for tag in tags if tag not in out]
    if missing:
        raise RuntimeError(f"missing summaries for tags: {missing}")
    return out


def pick_winner_fs(root: Path) -> str:
    runs = latest_run_dirs(root, PHASE_A_CONFIGS)
    baseline = best_metrics_record(runs["baseline"])
    base_exact = float(baseline.get("val_focus_board_exact_mean", 0.0))
    base_masked = float(baseline.get("val_focus_masked_acc_mean", 0.0))
    ranked: list[tuple[tuple[float, float], str]] = []
    for tag in PHASE_A_CONFIGS:
        if tag == "baseline":
            continue
        best = best_metrics_record(runs[tag])
        d_exact = float(best.get("val_focus_board_exact_mean", 0.0)) - base_exact
        d_masked = float(best.get("val_focus_masked_acc_mean", 0.0)) - base_masked
        if d_exact < 0.0 and d_masked < 0.0:
            continue
        ranked.append(((float(best.get("val_focus_board_exact_mean", 0.0)), float(best.get("val_focus_masked_acc_mean", 0.0))), tag))
    if not ranked:
        raise RuntimeError("no FS config survived phase A")
    ranked.sort(reverse=True)
    return ranked[0][1]


def launch_phase(*, phase: str, tags: list[str], run_root: Path, val_manifest: Path, test_manifest: Path, dry_run: bool, train_masks: str, focus_masks: str, eval_masks: str, phase_a_max_steps: int, phase_a_eval_every: int, phase_a_eval_examples: int, phase_b_max_steps: int, phase_b_eval_every: int, phase_b_eval_examples: int, phase_b_final_eval_examples: int, refine_steps_train: int, refine_steps_eval: int, consistency_lambda: float, progressive_train: bool, progressive_eval: bool, wrong_digit_corruption_prob: float, decode_legalize: bool, remask_conflicts: bool, remask_low_confidence: float) -> None:
    budgets = {
        "smoke": {"max_steps": 2, "eval_every": 1, "eval_examples": 1, "final_eval": 0},
        "phase_a": {"max_steps": int(phase_a_max_steps), "eval_every": int(phase_a_eval_every), "eval_examples": int(phase_a_eval_examples), "final_eval": 0},
        "phase_b": {"max_steps": int(phase_b_max_steps), "eval_every": int(phase_b_eval_every), "eval_examples": int(phase_b_eval_examples), "final_eval": int(phase_b_final_eval_examples)},
    }
    budget = budgets[phase]
    for tag in tags:
        cmd = [
            sys.executable,
            str(TRAIN),
            "--run_dir", str(run_root),
            "--tag", tag,
            "--train_masks", str(train_masks),
            "--focus_masks", str(focus_masks),
            "--eval_masks", str(eval_masks),
            "--val_manifest", str(val_manifest),
            "--test_manifest", str(test_manifest),
            "--bsz", "16",
            "--max_steps", str(budget["max_steps"]),
            "--eval_every", str(budget["eval_every"]),
            "--eval_examples_per_mask", str(budget["eval_examples"]),
            "--final_eval_examples_per_mask", str(budget["final_eval"]),
            "--refine_steps_train", str(refine_steps_train),
            "--refine_steps_eval", str(refine_steps_eval),
            "--consistency_lambda", str(consistency_lambda),
            "--wrong_digit_corruption_prob", str(wrong_digit_corruption_prob),
            "--remask_low_confidence", str(remask_low_confidence),
        ] + CONFIGS[tag]
        if progressive_train:
            cmd.append("--progressive_train")
        if progressive_eval:
            cmd.append("--progressive_eval")
        if decode_legalize:
            cmd.append("--decode_legalize")
        if remask_conflicts:
            cmd.append("--remask_conflicts")
        run(cmd, dry_run=dry_run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "phase_a", "phase_b", "full"], default="smoke")
    ap.add_argument("--assets_dir", type=str, default=str(DEFAULT_ASSETS))
    ap.add_argument("--run_dir", type=str, default="runs")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--train_masks", type=str, default=DEFAULT_TRAIN_MASKS)
    ap.add_argument("--focus_masks", type=str, default=DEFAULT_FOCUS_MASKS)
    ap.add_argument("--eval_masks", type=str, default=DEFAULT_EVAL_MASKS)
    ap.add_argument("--manifest_seed", type=int, default=20260314)
    ap.add_argument("--phase_a_max_steps", type=int, default=3000)
    ap.add_argument("--phase_a_eval_every", type=int, default=300)
    ap.add_argument("--phase_a_eval_examples", type=int, default=64)
    ap.add_argument("--phase_b_max_steps", type=int, default=25000)
    ap.add_argument("--phase_b_eval_every", type=int, default=1000)
    ap.add_argument("--phase_b_eval_examples", type=int, default=256)
    ap.add_argument("--phase_b_final_eval_examples", type=int, default=2000)
    ap.add_argument("--refine_steps_train", type=int, default=1)
    ap.add_argument("--refine_steps_eval", type=int, default=1)
    ap.add_argument("--consistency_lambda", type=float, default=0.0)
    ap.add_argument("--progressive_train", action="store_true")
    ap.add_argument("--progressive_eval", action="store_true")
    ap.add_argument("--wrong_digit_corruption_prob", type=float, default=0.0)
    ap.add_argument("--decode_legalize", action="store_true")
    ap.add_argument("--remask_conflicts", action="store_true")
    ap.add_argument("--remask_low_confidence", type=float, default=0.0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    assets_dir = Path(args.assets_dir)
    run_root = Path(args.run_dir)
    eval_masks = parse_int_list(args.eval_masks)

    if args.self_test:
        run([sys.executable, str(TRAIN), "--self_test"], dry_run=args.dry_run)
        run([sys.executable, str(BUILD), "--out_dir", str(assets_dir), "--smoke", "--mask_counts", args.eval_masks, "--seed", str(args.manifest_seed)], dry_run=args.dry_run)
        if not args.dry_run:
            run([sys.executable, str(SUMMARY), "--root", str(run_root)], dry_run=False)
        return

    if args.phase == "smoke":
        val_manifest, test_manifest = ensure_manifests(assets_dir=assets_dir, smoke=True, dry_run=args.dry_run, workers=int(args.workers), masks=eval_masks, seed=int(args.manifest_seed))
        launch_phase(
            phase="smoke",
            tags=SMOKE_CONFIGS,
            run_root=run_root,
            val_manifest=val_manifest,
            test_manifest=test_manifest,
            dry_run=args.dry_run,
            train_masks=args.train_masks,
            focus_masks=args.focus_masks,
            eval_masks=args.eval_masks,
            phase_a_max_steps=args.phase_a_max_steps,
            phase_a_eval_every=args.phase_a_eval_every,
            phase_a_eval_examples=args.phase_a_eval_examples,
            phase_b_max_steps=args.phase_b_max_steps,
            phase_b_eval_every=args.phase_b_eval_every,
            phase_b_eval_examples=args.phase_b_eval_examples,
            phase_b_final_eval_examples=args.phase_b_final_eval_examples,
            refine_steps_train=args.refine_steps_train,
            refine_steps_eval=args.refine_steps_eval,
            consistency_lambda=args.consistency_lambda,
            progressive_train=args.progressive_train,
            progressive_eval=args.progressive_eval,
            wrong_digit_corruption_prob=args.wrong_digit_corruption_prob,
            decode_legalize=args.decode_legalize,
            remask_conflicts=args.remask_conflicts,
            remask_low_confidence=args.remask_low_confidence,
        )
        return

    val_manifest, test_manifest = ensure_manifests(assets_dir=assets_dir, smoke=False, dry_run=args.dry_run, workers=int(args.workers), masks=eval_masks, seed=int(args.manifest_seed))
    if args.phase in {"phase_a", "full"}:
        launch_phase(
            phase="phase_a",
            tags=PHASE_A_CONFIGS,
            run_root=run_root,
            val_manifest=val_manifest,
            test_manifest=test_manifest,
            dry_run=args.dry_run,
            train_masks=args.train_masks,
            focus_masks=args.focus_masks,
            eval_masks=args.eval_masks,
            phase_a_max_steps=args.phase_a_max_steps,
            phase_a_eval_every=args.phase_a_eval_every,
            phase_a_eval_examples=args.phase_a_eval_examples,
            phase_b_max_steps=args.phase_b_max_steps,
            phase_b_eval_every=args.phase_b_eval_every,
            phase_b_eval_examples=args.phase_b_eval_examples,
            phase_b_final_eval_examples=args.phase_b_final_eval_examples,
            refine_steps_train=args.refine_steps_train,
            refine_steps_eval=args.refine_steps_eval,
            consistency_lambda=args.consistency_lambda,
            progressive_train=args.progressive_train,
            progressive_eval=args.progressive_eval,
            wrong_digit_corruption_prob=args.wrong_digit_corruption_prob,
            decode_legalize=args.decode_legalize,
            remask_conflicts=args.remask_conflicts,
            remask_low_confidence=args.remask_low_confidence,
        )
    if args.phase in {"phase_b", "full"}:
        winner = DEFAULT_PHASE_B_WINNER if args.dry_run else pick_winner_fs(run_root)
        launch_phase(
            phase="phase_b",
            tags=["baseline", winner],
            run_root=run_root,
            val_manifest=val_manifest,
            test_manifest=test_manifest,
            dry_run=args.dry_run,
            train_masks=args.train_masks,
            focus_masks=args.focus_masks,
            eval_masks=args.eval_masks,
            phase_a_max_steps=args.phase_a_max_steps,
            phase_a_eval_every=args.phase_a_eval_every,
            phase_a_eval_examples=args.phase_a_eval_examples,
            phase_b_max_steps=args.phase_b_max_steps,
            phase_b_eval_every=args.phase_b_eval_every,
            phase_b_eval_examples=args.phase_b_eval_examples,
            phase_b_final_eval_examples=args.phase_b_final_eval_examples,
            refine_steps_train=args.refine_steps_train,
            refine_steps_eval=args.refine_steps_eval,
            consistency_lambda=args.consistency_lambda,
            progressive_train=args.progressive_train,
            progressive_eval=args.progressive_eval,
            wrong_digit_corruption_prob=args.wrong_digit_corruption_prob,
            decode_legalize=args.decode_legalize,
            remask_conflicts=args.remask_conflicts,
            remask_low_confidence=args.remask_low_confidence,
        )


if __name__ == "__main__":
    main()
