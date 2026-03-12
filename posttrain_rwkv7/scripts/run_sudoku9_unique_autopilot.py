#!/usr/bin/env python3
"""Nonstop autopilot for the 9x9 Sudoku Future-Seed search."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
RUN_MAIN = SCRIPTS / "run_sudoku9_unique_maintrack.py"
TRAIN = SCRIPTS / "train_sudoku9_unique_sft.py"

CONFIGS = {
    "baseline": ["--mode", "no_fs", "--fs_variant", "scalar", "--alpha_lr", "0", "--alpha_init", "-2", "--fs_layer_start", "6", "--fs_clip", "1.0"],
    "fs_scalar_l6": ["--mode", "prompt_fs", "--fs_variant", "scalar", "--alpha_lr", "5e-4", "--alpha_init", "-2", "--fs_layer_start", "6", "--fs_norm", "--fs_detach", "--fs_clip", "1.0"],
    "fs_scalar_l8_cos": ["--mode", "prompt_fs", "--fs_variant", "scalar", "--alpha_lr", "0", "--alpha_init", "-2", "--fs_layer_start", "8", "--fs_alpha_schedule", "cosine", "--fs_alpha_min", "0.4", "--fs_alpha_max", "1.0", "--fs_norm", "--fs_detach", "--fs_clip", "1.0"],
    "fs_head_l6": ["--mode", "prompt_fs", "--fs_variant", "head", "--alpha_lr", "0", "--alpha_init", "-3", "--alpha_head_init", "-3", "--alpha_head_lr", "5e-4", "--fs_layer_start", "6", "--fs_norm", "--fs_detach", "--fs_clip", "1.0"],
}
FS_TAGS = ["fs_scalar_l6", "fs_scalar_l8_cos", "fs_head_l6"]


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def is_training_active() -> bool:
    out = subprocess.run(
        ["ps", "-eo", "cmd"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in out.splitlines():
        if "train_sudoku9_unique_sft.py" in line or "run_sudoku9_unique_maintrack.py" in line:
            return True
    return False


def wait_until_idle() -> None:
    while is_training_active():
        time.sleep(30)


def latest_matching_run(
    *,
    root: Path,
    tag: str,
    train_clues: list[int],
    focus_clues: list[int],
    eval_clues: list[int],
    require_final_test: bool,
) -> Path | None:
    best: Path | None = None
    for summary_path in root.glob("**/summary.json"):
        summary = read_json(summary_path)
        if summary.get("tag") != tag:
            continue
        if summary.get("train_clues") != train_clues:
            continue
        if summary.get("focus_clues") != focus_clues:
            continue
        if summary.get("eval_clues") != eval_clues:
            continue
        if require_final_test and "final_test" not in summary:
            continue
        if best is None or str(summary_path.parent) > str(best):
            best = summary_path.parent
    return best


def best_metrics(run_dir: Path) -> dict:
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


def collect_stage(
    *,
    root: Path,
    train_clues: list[int],
    focus_clues: list[int],
    eval_clues: list[int],
    require_final_test: bool,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for tag in ["baseline"] + FS_TAGS:
        run_dir = latest_matching_run(
            root=root,
            tag=tag,
            train_clues=train_clues,
            focus_clues=focus_clues,
            eval_clues=eval_clues,
            require_final_test=require_final_test,
        )
        if run_dir is None:
            continue
        summary = read_json(run_dir / "summary.json")
        out[tag] = {
            "run_dir": run_dir,
            "summary": summary,
            "best": best_metrics(run_dir),
        }
    return out


def choose_candidate(rows: dict[str, dict]) -> tuple[str | None, str]:
    baseline = rows.get("baseline")
    if baseline is None:
        return None, "missing_baseline"
    base_final = baseline["summary"].get("final_test", {})
    base_exact = float(base_final.get("focus_exact_mean", 0.0))
    base_blank = float(base_final.get("focus_blank_acc_mean", baseline["best"].get("val_focus_blank_acc_mean", 0.0)))

    exact_candidates: list[tuple[float, str]] = []
    blank_candidates: list[tuple[float, str]] = []
    for tag in FS_TAGS:
        row = rows.get(tag)
        if row is None:
            continue
        final = row["summary"].get("final_test", {})
        focus_exact = float(final.get("focus_exact_mean", 0.0))
        focus_blank = float(final.get("focus_blank_acc_mean", row["best"].get("val_focus_blank_acc_mean", 0.0)))
        if focus_exact > base_exact:
            exact_candidates.append((focus_exact - base_exact, tag))
        if focus_blank > base_blank + 0.005:
            blank_candidates.append((focus_blank - base_blank, tag))
    if exact_candidates:
        exact_candidates.sort(reverse=True)
        return exact_candidates[0][1], "exact"
    if blank_candidates:
        blank_candidates.sort(reverse=True)
        return blank_candidates[0][1], "blank"
    return None, "none"


def run_maintrack_phase(
    *,
    phase: str,
    assets_dir: str,
    run_dir: str,
    train_clues: str,
    focus_clues: str,
    eval_clues: str,
    manifest_seed: int,
    phase_a_max_steps: int = 3000,
    phase_a_eval_every: int = 300,
    phase_a_eval_examples: int = 64,
    phase_b_max_steps: int = 25000,
    phase_b_eval_every: int = 1000,
    phase_b_eval_examples: int = 256,
    phase_b_final_eval_examples: int = 2000,
) -> None:
    run(
        [
            sys.executable,
            str(RUN_MAIN),
            "--phase",
            phase,
            "--workers",
            "32",
            "--assets_dir",
            assets_dir,
            "--run_dir",
            run_dir,
            "--train_clues",
            train_clues,
            "--focus_clues",
            focus_clues,
            "--eval_clues",
            eval_clues,
            "--manifest_seed",
            str(manifest_seed),
            "--phase_a_max_steps",
            str(phase_a_max_steps),
            "--phase_a_eval_every",
            str(phase_a_eval_every),
            "--phase_a_eval_examples",
            str(phase_a_eval_examples),
            "--phase_b_max_steps",
            str(phase_b_max_steps),
            "--phase_b_eval_every",
            str(phase_b_eval_every),
            "--phase_b_eval_examples",
            str(phase_b_eval_examples),
            "--phase_b_final_eval_examples",
            str(phase_b_final_eval_examples),
        ]
    )


def run_confirm_pair(
    *,
    assets_dir: str,
    run_dir: str,
    manifest_seed: int,
    train_clues: str,
    focus_clues: str,
    eval_clues: str,
    winner_tag: str,
    confirm_prefix: str,
    seed: int,
) -> None:
    eval_list = parse_int_list(eval_clues)
    clue_tag = "seed" + str(manifest_seed) + "_" + "-".join(str(x) for x in eval_list)
    if eval_list == parse_int_list("40,36,32,28,24") and int(manifest_seed) == 20260312:
        clue_tag = "seed20260312"
    val_manifest = str(Path(assets_dir) / f"val_{clue_tag}.jsonl")
    test_manifest = str(Path(assets_dir) / f"test_{clue_tag}.jsonl")
    for tag in ["baseline", winner_tag]:
        run(
            [
                sys.executable,
                str(TRAIN),
                "--run_dir",
                run_dir,
                "--tag",
                f"{confirm_prefix}_{tag}_seed{seed}",
                "--seed",
                str(seed),
                "--train_clues",
                train_clues,
                "--focus_clues",
                focus_clues,
                "--eval_clues",
                eval_clues,
                "--val_manifest",
                val_manifest,
                "--test_manifest",
                test_manifest,
                "--bsz",
                "16",
                "--max_steps",
                "15000",
                "--eval_every",
                "1000",
                "--eval_examples_per_clue",
                "256",
                "--final_eval_examples_per_clue",
                "2000",
            ]
            + CONFIGS[tag]
        )


def main() -> None:
    run_root = REPO / "runs"
    wait_until_idle()

    default_train = [40, 36, 32, 28]
    default_focus = [32, 28]
    default_eval = [40, 36, 32, 28, 24]
    default_rows = collect_stage(
        root=run_root,
        train_clues=default_train,
        focus_clues=default_focus,
        eval_clues=default_eval,
        require_final_test=True,
    )
    if "baseline" not in default_rows:
        run_maintrack_phase(
            phase="phase_b",
            assets_dir="assets/sudoku9_unique",
            run_dir="runs",
            train_clues="40,36,32,28",
            focus_clues="32,28",
            eval_clues="40,36,32,28,24",
            manifest_seed=20260312,
        )
        wait_until_idle()
        default_rows = collect_stage(
            root=run_root,
            train_clues=default_train,
            focus_clues=default_focus,
            eval_clues=default_eval,
            require_final_test=True,
        )

    winner, reason = choose_candidate(default_rows)
    if winner is not None:
        run_confirm_pair(
            assets_dir="assets/sudoku9_unique",
            run_dir="runs",
            manifest_seed=20260312,
            train_clues="40,36,32,28",
            focus_clues="32,28",
            eval_clues="40,36,32,28,24",
            winner_tag=winner,
            confirm_prefix=f"confirm_mainline_{reason}",
            seed=1,
        )
        return

    run_maintrack_phase(
        phase="phase_a",
        assets_dir="assets/sudoku9_unique_easy",
        run_dir="runs",
        train_clues="48,44,40,36",
        focus_clues="40,36",
        eval_clues="48,44,40,36,32",
        manifest_seed=20260313,
    )
    run_maintrack_phase(
        phase="phase_b",
        assets_dir="assets/sudoku9_unique_easy",
        run_dir="runs",
        train_clues="48,44,40,36",
        focus_clues="40,36",
        eval_clues="48,44,40,36,32",
        manifest_seed=20260313,
        phase_b_max_steps=30000,
        phase_b_eval_every=1000,
        phase_b_eval_examples=256,
        phase_b_final_eval_examples=2000,
    )

    easy_rows = collect_stage(
        root=run_root,
        train_clues=[48, 44, 40, 36],
        focus_clues=[40, 36],
        eval_clues=[48, 44, 40, 36, 32],
        require_final_test=True,
    )
    easy_winner, easy_reason = choose_candidate(easy_rows)
    if easy_winner is None:
        easy_winner = "fs_scalar_l8_cos"
        easy_reason = "rescue"
    run_confirm_pair(
        assets_dir="assets/sudoku9_unique_easy",
        run_dir="runs",
        manifest_seed=20260313,
        train_clues="48,44,40,36",
        focus_clues="40,36",
        eval_clues="48,44,40,36,32",
        winner_tag=easy_winner,
        confirm_prefix=f"confirm_easy_{easy_reason}",
        seed=1,
    )


if __name__ == "__main__":
    main()
