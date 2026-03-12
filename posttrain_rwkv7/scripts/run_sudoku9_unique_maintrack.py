#!/usr/bin/env python3
"""Orchestrate the 9x9 unique-solution Sudoku mainline benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
TRAIN = SCRIPTS / "train_sudoku9_unique_sft.py"
BUILD = SCRIPTS / "build_sudoku9_unique_manifests.py"
SUMMARY = SCRIPTS / "summarize_sudoku9_unique.py"
DEFAULT_ASSETS = REPO / "assets" / "sudoku9_unique"

CONFIGS = {
    "baseline": ["--mode", "no_fs", "--fs_variant", "scalar", "--alpha_lr", "0", "--alpha_init", "-2", "--fs_layer_start", "6", "--fs_clip", "1.0"],
    "fs_scalar_l6": ["--mode", "prompt_fs", "--fs_variant", "scalar", "--alpha_lr", "5e-4", "--alpha_init", "-2", "--fs_layer_start", "6", "--fs_norm", "--fs_detach", "--fs_clip", "1.0"],
    "fs_scalar_l8_cos": ["--mode", "prompt_fs", "--fs_variant", "scalar", "--alpha_lr", "0", "--alpha_init", "-2", "--fs_layer_start", "8", "--fs_alpha_schedule", "cosine", "--fs_alpha_min", "0.4", "--fs_alpha_max", "1.0", "--fs_norm", "--fs_detach", "--fs_clip", "1.0"],
    "fs_head_l6": ["--mode", "prompt_fs", "--fs_variant", "head", "--alpha_lr", "0", "--alpha_init", "-3", "--alpha_head_init", "-3", "--alpha_head_lr", "5e-4", "--fs_layer_start", "6", "--fs_norm", "--fs_detach", "--fs_clip", "1.0"],
}
PHASE_A_CONFIGS = ["baseline", "fs_scalar_l6", "fs_scalar_l8_cos", "fs_head_l6"]
SMOKE_CONFIGS = ["baseline", "fs_scalar_l6"]
DEFAULT_PHASE_B_WINNERS = ["fs_scalar_l6", "fs_scalar_l8_cos"]


def run(cmd: list[str], *, dry_run: bool) -> None:
    print("RUN", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def ensure_manifests(*, assets_dir: Path, smoke: bool, dry_run: bool, workers: int) -> tuple[Path, Path]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if smoke else "seed20260312"
    val = assets_dir / f"val_{tag}.jsonl"
    test = assets_dir / f"test_{tag}.jsonl"
    if val.exists() and test.exists():
        return val, test
    cmd = [sys.executable, str(BUILD), "--out_dir", str(assets_dir), "--workers", str(workers)]
    if smoke:
        cmd.append("--smoke")
    run(cmd, dry_run=dry_run)
    return val, test


def read_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


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
                float(row.get("val_focus_exact_mean", 0.0)),
                float(row.get("val_focus_blank_acc_mean", 0.0)),
            )
            if best is None:
                best = row
                continue
            best_key = (
                float(best.get("val_focus_exact_mean", 0.0)),
                float(best.get("val_focus_blank_acc_mean", 0.0)),
            )
            if key > best_key:
                best = row
    return best or {}


def latest_run_dirs(root: Path, tags: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in root.glob("**/sudoku9_unique_sft/*/summary.json"):
        tag = path.parent.name
        if tag not in tags:
            continue
        if tag not in out or str(path.parent) > str(out[tag]):
            out[tag] = path.parent
    missing = [tag for tag in tags if tag not in out]
    if missing:
        raise RuntimeError(f"missing summaries for tags: {missing}")
    return out


def pick_top2_fs(root: Path) -> list[str]:
    runs = latest_run_dirs(root, PHASE_A_CONFIGS)
    baseline = best_metrics_record(runs["baseline"])
    base32 = float(baseline.get("val_exact_32", 0.0))
    base28 = float(baseline.get("val_exact_28", 0.0))
    base_blank = float(baseline.get("val_focus_blank_acc_mean", 0.0))
    nonzero_exact = False
    scored = []
    for tag in PHASE_A_CONFIGS:
        if tag == "baseline":
            continue
        best = best_metrics_record(runs[tag])
        d32 = float(best.get("val_exact_32", 0.0)) - base32
        d28 = float(best.get("val_exact_28", 0.0)) - base28
        if float(best.get("val_focus_exact_mean", 0.0)) > 0.0:
            nonzero_exact = True
        scored.append(
            {
                "tag": tag,
                "focus_exact": float(best.get("val_focus_exact_mean", 0.0)),
                "focus_blank": float(best.get("val_focus_blank_acc_mean", 0.0)),
                "d32": d32,
                "d28": d28,
                "dblank": float(best.get("val_focus_blank_acc_mean", 0.0)) - base_blank,
            }
        )
    ranked: list[tuple[float, str]] = []
    for row in scored:
        if nonzero_exact:
            if row["d32"] < -0.01 and row["d28"] < -0.01:
                continue
            ranked.append((float(row["focus_exact"]), str(row["tag"])))
        else:
            if float(row["dblank"]) < -0.005:
                continue
            ranked.append((float(row["focus_blank"]), str(row["tag"])))
    ranked.sort(reverse=True)
    if not ranked:
        raise RuntimeError("no FS config survived phase A")
    return [tag for _, tag in ranked[:2]]


def launch_phase(
    *,
    phase: str,
    tags: list[str],
    run_root: Path,
    val_manifest: Path,
    test_manifest: Path,
    dry_run: bool,
) -> None:
    phase_budgets = {
        "smoke": {"max_steps": 2, "eval_every": 1, "eval_examples": 1, "final_eval": 0},
        "phase_a": {"max_steps": 3000, "eval_every": 300, "eval_examples": 64, "final_eval": 0},
        "phase_b": {"max_steps": 25000, "eval_every": 1000, "eval_examples": 256, "final_eval": 2000},
    }
    budget = phase_budgets[phase]
    for tag in tags:
        cmd = [
            sys.executable,
            str(TRAIN),
            "--run_dir",
            str(run_root),
            "--tag",
            tag,
            "--train_clues",
            "40,36,32,28",
            "--focus_clues",
            "32,28",
            "--eval_clues",
            "40,36,32,28,24",
            "--val_manifest",
            str(val_manifest),
            "--test_manifest",
            str(test_manifest),
            "--bsz",
            "16",
            "--max_steps",
            str(budget["max_steps"]),
            "--eval_every",
            str(budget["eval_every"]),
            "--eval_examples_per_clue",
            str(budget["eval_examples"]),
            "--final_eval_examples_per_clue",
            str(budget["final_eval"]),
        ] + CONFIGS[tag]
        run(cmd, dry_run=dry_run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "phase_a", "phase_b", "full"], default="smoke")
    ap.add_argument("--assets_dir", type=str, default=str(DEFAULT_ASSETS))
    ap.add_argument("--run_dir", type=str, default="runs")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    assets_dir = Path(args.assets_dir)
    run_root = Path(args.run_dir)

    if args.self_test:
        run([sys.executable, str(TRAIN), "--self_test"], dry_run=args.dry_run)
        run([sys.executable, str(BUILD), "--out_dir", str(assets_dir), "--smoke"], dry_run=args.dry_run)
        if not args.dry_run:
            run([sys.executable, str(SUMMARY), "--root", str(run_root)], dry_run=False)
        return

    if args.phase == "smoke":
        val_manifest, test_manifest = ensure_manifests(assets_dir=assets_dir, smoke=True, dry_run=args.dry_run, workers=int(args.workers))
        launch_phase(phase="smoke", tags=SMOKE_CONFIGS, run_root=run_root, val_manifest=val_manifest, test_manifest=test_manifest, dry_run=args.dry_run)
        return

    val_manifest, test_manifest = ensure_manifests(assets_dir=assets_dir, smoke=False, dry_run=args.dry_run, workers=int(args.workers))
    if args.phase in {"phase_a", "full"}:
        launch_phase(phase="phase_a", tags=PHASE_A_CONFIGS, run_root=run_root, val_manifest=val_manifest, test_manifest=test_manifest, dry_run=args.dry_run)
    if args.phase in {"phase_b", "full"}:
        winners = DEFAULT_PHASE_B_WINNERS if args.dry_run else pick_top2_fs(run_root)
        launch_phase(phase="phase_b", tags=["baseline"] + [tag for tag in winners if tag != "baseline"], run_root=run_root, val_manifest=val_manifest, test_manifest=test_manifest, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
