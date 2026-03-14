#!/usr/bin/env python3
"""Build deterministic 9x9 Sudoku in-place repair manifests."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from sudoku9_inplace import SudokuInplaceExample, generate_example, validate_example, write_manifest


def parse_masks(text: str) -> list[int]:
    vals = [int(x) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("expected at least one mask count")
    return vals


def _worker(args: tuple[str, int, int, int, bool]) -> SudokuInplaceExample:
    split, mask_count, seed, index, validate = args
    ex = generate_example(split=split, mask_target=mask_count, seed=seed, index=index)
    if validate:
        validate_example(ex)
    return ex


def build_split(*, split: str, masks: list[int], count_per_mask: int, seed: int, workers: int, validate: bool) -> list[SudokuInplaceExample]:
    jobs = []
    split_seed = seed + (0 if split == "val" else 10_000_000)
    for mask_count in masks:
        for idx in range(count_per_mask):
            jobs.append((split, mask_count, split_seed, idx, validate))
    if workers <= 1:
        return [_worker(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_worker, jobs))


def write_summary(path: Path, *, masks: list[int], val_per_mask: int, test_per_mask: int, seed: int) -> None:
    path.write_text(json.dumps({
        "mask_counts": list(masks),
        "val_per_mask": int(val_per_mask),
        "test_per_mask": int(test_per_mask),
        "seed": int(seed),
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="assets/sudoku9_inplace")
    ap.add_argument("--mask_counts", type=str, default="28,32,36,40")
    ap.add_argument("--val_per_mask", type=int, default=1000)
    ap.add_argument("--test_per_mask", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260314)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--skip_validate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    masks = parse_masks(args.mask_counts)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_per_mask = 2 if args.smoke else int(args.val_per_mask)
    test_per_mask = 2 if args.smoke else int(args.test_per_mask)
    tag = "smoke" if args.smoke else f"seed{int(args.seed)}"
    workers = int(args.workers)
    validate = not bool(args.skip_validate)

    val_examples = build_split(split="val", masks=masks, count_per_mask=val_per_mask, seed=int(args.seed), workers=workers, validate=validate)
    test_examples = build_split(split="test", masks=masks, count_per_mask=test_per_mask, seed=int(args.seed), workers=workers, validate=validate)

    val_path = out_dir / f"val_{tag}.jsonl"
    test_path = out_dir / f"test_{tag}.jsonl"
    summary_path = out_dir / f"manifest_summary_{tag}.json"
    write_manifest(val_path, val_examples)
    write_manifest(test_path, test_examples)
    write_summary(summary_path, masks=masks, val_per_mask=val_per_mask, test_per_mask=test_per_mask, seed=int(args.seed))

    print(json.dumps({
        "val_manifest": str(val_path),
        "test_manifest": str(test_path),
        "summary": str(summary_path),
        "val_examples": len(val_examples),
        "test_examples": len(test_examples),
        "mask_counts": masks,
    }))


if __name__ == "__main__":
    main()
