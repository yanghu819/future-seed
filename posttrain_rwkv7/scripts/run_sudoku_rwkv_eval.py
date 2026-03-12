#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cache_defaults import apply_cache_env, ensure_cache_dirs
from sudoku9_unique import blank_token_accuracy, is_clue_consistent, is_valid_solution, string_to_board
from sudoku_rwkv_future_seed import FutureSeedConfig, build_future_seed_class, parse_float_list, parse_head_matrix
from sudoku_rwkv_official import CHECKPOINT_FILE, ensure_snapshot

FILL_RE = re.compile(r"^> Fill cell \((\d), (\d)\) (\d)$")


def load_manifest(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def puzzle_to_prompt(puzzle: str) -> str:
    if len(puzzle) != 81:
        raise ValueError("puzzle must have length 81")
    rows = []
    for i in range(9):
        row = puzzle[i * 9 : (i + 1) * 9]
        rows.append(" ".join(row) + " ")
    return "<input>\n" + "\n".join(rows) + "\n</input>\n\n"


def update_board_from_line(board: list[str], line: str) -> bool:
    m = FILL_RE.match(line.strip())
    if not m:
        return False
    row, col, num = [int(x) for x in m.groups()]
    board[row * 9 + col] = str(num)
    return True


def board_metrics(puzzle: str, solution: str, prediction: str) -> dict[str, float]:
    clue_ok = is_clue_consistent(puzzle, prediction)
    valid = is_valid_solution(string_to_board(prediction))
    exact = prediction == solution
    return {
        "exact": float(exact),
        "valid": float(valid),
        "clue": float(clue_ok),
        "blank_acc": float(blank_token_accuracy(puzzle, prediction, solution)),
    }


def build_model(args):
    apply_cache_env()
    ensure_cache_dirs()
    snapshot_dir = ensure_snapshot(root=args.snapshot_root, include_checkpoint=True, verbose=not args.quiet)
    RWKVFutureSeed, _official, root = build_future_seed_class(snapshot_dir)

    import os
    os.environ.setdefault("RWKV_JIT_ON", "1")
    os.environ.setdefault("RWKV_CUDA_ON", "0")

    from rwkv.utils import PIPELINE, PIPELINE_ARGS
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

    model = RWKVFutureSeed(model=str(Path(root) / CHECKPOINT_FILE), strategy=args.strategy, verbose=not args.quiet)
    cfg = FutureSeedConfig(
        enabled=bool(args.future_seed),
        layer_start=int(args.fs_layer_start),
        seed_scale=float(args.seed_scale),
        fs_alpha=parse_float_list(args.fs_alpha),
        fs_alpha_head=parse_head_matrix(args.fs_alpha_head),
        fs_norm=bool(args.fs_norm),
        fs_clip=float(args.fs_clip),
        residual=not bool(args.fs_replace),
    )
    model.set_future_seed(cfg)
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
    pipeline.tokenizer = TRIE_TOKENIZER(str(Path(root) / "sudoku_vocab.txt"))
    gen_args = PIPELINE_ARGS(top_k=int(args.top_k), alpha_frequency=0, alpha_presence=0, token_stop=[105])
    return model, pipeline, gen_args, root


def evaluate_item(pipeline, gen_args, item: dict, *, max_tokens: int, quiet: bool) -> dict:
    prompt = puzzle_to_prompt(item["puzzle"])
    board = list(item["puzzle"])
    token_count = 0
    current_line = ""
    t0 = time.time()

    def callback(text: str):
        nonlocal token_count, current_line
        token_count += 1
        current_line += text
        if text.endswith("\n"):
            line = current_line.strip()
            if line:
                update_board_from_line(board, line)
            current_line = ""

    output = pipeline.generate(prompt, token_count=max_tokens, args=gen_args, callback=callback)
    if current_line.strip():
        update_board_from_line(board, current_line.strip())
    prediction = "".join(board)
    metrics = board_metrics(item["puzzle"], item["solution"], prediction)
    row = {
        "item_id": item["item_id"],
        "clue_count": int(item["clue_count"]),
        "prediction": prediction,
        "tokens": int(token_count),
        "elapsed_sec": float(time.time() - t0),
        "generation_tail": output[-2000:] if isinstance(output, str) else "",
    }
    row.update(metrics)
    if not quiet:
        print(json.dumps({k: row[k] for k in ("item_id", "clue_count", "exact", "valid", "clue", "blank_acc", "tokens")}, ensure_ascii=False))
    return row


def summarize(rows: list[dict]) -> dict:
    by_clue: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_clue[int(row["clue_count"])].append(row)

    buckets = {}
    for clue, group in sorted(by_clue.items()):
        buckets[str(clue)] = {
            "count": len(group),
            "exact": float(statistics.mean(r["exact"] for r in group)),
            "valid": float(statistics.mean(r["valid"] for r in group)),
            "clue": float(statistics.mean(r["clue"] for r in group)),
            "blank_acc": float(statistics.mean(r["blank_acc"] for r in group)),
            "tokens": float(statistics.mean(r["tokens"] for r in group)),
            "elapsed_sec": float(statistics.mean(r["elapsed_sec"] for r in group)),
        }
    all_rows = rows or [{"exact": 0.0, "valid": 0.0, "clue": 0.0, "blank_acc": 0.0, "tokens": 0.0, "elapsed_sec": 0.0}]
    overall = {
        "count": len(rows),
        "exact": float(statistics.mean(r["exact"] for r in all_rows)),
        "valid": float(statistics.mean(r["valid"] for r in all_rows)),
        "clue": float(statistics.mean(r["clue"] for r in all_rows)),
        "blank_acc": float(statistics.mean(r["blank_acc"] for r in all_rows)),
        "tokens": float(statistics.mean(r["tokens"] for r in all_rows)),
        "elapsed_sec": float(statistics.mean(r["elapsed_sec"] for r in all_rows)),
    }
    return {"overall": overall, "buckets": buckets}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate official Sudoku-RWKV baseline or Future-Seed-augmented inference")
    ap.add_argument("--manifest", type=str, required=False, default=str(REPO_ROOT / "posttrain_rwkv7/assets/sudoku9_unique/val_smoke.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=50000)
    ap.add_argument("--strategy", type=str, default="cuda fp16")
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--snapshot_root", type=str, default=None)
    ap.add_argument("--future_seed", action="store_true")
    ap.add_argument("--fs_layer_start", type=int, default=1)
    ap.add_argument("--seed_scale", type=float, default=1.0)
    ap.add_argument("--fs_alpha", type=str, default=None)
    ap.add_argument("--fs_alpha_head", type=str, default=None)
    ap.add_argument("--fs_norm", action="store_true")
    ap.add_argument("--fs_clip", type=float, default=0.0)
    ap.add_argument("--fs_replace", action="store_true")
    ap.add_argument("--output_json", type=str, default=None)
    ap.add_argument("--output_jsonl", type=str, default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--self_test", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        prompt = puzzle_to_prompt("0" * 81)
        assert prompt.startswith("<input>\n")
        board = list("0" * 81)
        assert update_board_from_line(board, "> Fill cell (0, 6) 2")
        assert board[6] == "2"
        assert not update_board_from_line(board, "random")
        print("run_sudoku_rwkv_eval_self_test_ok")
        return

    manifest = load_manifest(Path(args.manifest), limit=int(args.limit))
    _model, pipeline, gen_args, snapshot_root = build_model(args)

    rows = []
    for item in manifest:
        rows.append(evaluate_item(pipeline, gen_args, item, max_tokens=int(args.max_tokens), quiet=bool(args.quiet)))
    summary = summarize(rows)
    summary["config"] = {
        "manifest": str(Path(args.manifest).resolve()),
        "limit": len(rows),
        "max_tokens": int(args.max_tokens),
        "strategy": args.strategy,
        "future_seed": bool(args.future_seed),
        "fs_layer_start": int(args.fs_layer_start),
        "seed_scale": float(args.seed_scale),
        "fs_alpha": parse_float_list(args.fs_alpha),
        "fs_alpha_head": parse_head_matrix(args.fs_alpha_head),
        "fs_norm": bool(args.fs_norm),
        "fs_clip": float(args.fs_clip),
        "fs_replace": bool(args.fs_replace),
        "snapshot_root": str(snapshot_root),
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    if args.output_jsonl:
        out = Path(args.output_jsonl)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
