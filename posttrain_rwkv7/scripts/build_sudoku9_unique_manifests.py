#!/usr/bin/env python3
"""Build deterministic 9x9 unique-solution Sudoku manifests."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

from sudoku9_unique import SudokuExample, generate_example, validate_example, write_manifest


def parse_clues(text: str) -> list[int]:
    vals = [int(x) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("expected at least one clue count")
    return vals


def _worker(args: tuple[str, int, int, int, bool]) -> SudokuExample:
    split, clues, seed, index, validate = args
    ex = generate_example(split=split, clue_target=clues, seed=seed, index=index)
    if validate:
        validate_example(ex)
    return ex


def build_split(
    *,
    split: str,
    clues: list[int],
    count_per_clue: int,
    seed: int,
    workers: int,
    validate: bool,
) -> list[SudokuExample]:
    jobs = []
    split_seed = seed + (0 if split == "val" else 10_000_000)
    for clue in clues:
        for idx in range(count_per_clue):
            jobs.append((split, clue, split_seed, idx, validate))
    if workers <= 1:
        return [_worker(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_worker, jobs))


def write_summary(path: Path, *, clues: Iterable[int], val_per_clue: int, test_per_clue: int, seed: int) -> None:
    row = {
        "clues": list(clues),
        "val_per_clue": int(val_per_clue),
        "test_per_clue": int(test_per_clue),
        "seed": int(seed),
    }
    path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="assets/sudoku9_unique")
    ap.add_argument("--clues", type=str, default="40,36,32,28,24")
    ap.add_argument("--val_per_clue", type=int, default=1000)
    ap.add_argument("--test_per_clue", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260312)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--skip_validate", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="override counts to tiny smoke manifests")
    args = ap.parse_args()

    clues = parse_clues(args.clues)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_per_clue = 2 if args.smoke else int(args.val_per_clue)
    test_per_clue = 2 if args.smoke else int(args.test_per_clue)
    tag = "smoke" if args.smoke else f"seed{int(args.seed)}"
    workers = int(args.workers)
    validate = not bool(args.skip_validate)

    val_examples = build_split(
        split="val",
        clues=clues,
        count_per_clue=val_per_clue,
        seed=int(args.seed),
        workers=workers,
        validate=validate,
    )
    test_examples = build_split(
        split="test",
        clues=clues,
        count_per_clue=test_per_clue,
        seed=int(args.seed),
        workers=workers,
        validate=validate,
    )

    val_path = out_dir / f"val_{tag}.jsonl"
    test_path = out_dir / f"test_{tag}.jsonl"
    summary_path = out_dir / f"manifest_summary_{tag}.json"
    write_manifest(val_path, val_examples)
    write_manifest(test_path, test_examples)
    write_summary(summary_path, clues=clues, val_per_clue=val_per_clue, test_per_clue=test_per_clue, seed=int(args.seed))

    print(json.dumps({
        "val_manifest": str(val_path),
        "test_manifest": str(test_path),
        "summary": str(summary_path),
        "val_examples": len(val_examples),
        "test_examples": len(test_examples),
        "clues": clues,
    }))


if __name__ == "__main__":
    main()
