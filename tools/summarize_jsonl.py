#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize JSONL logs produced by rwkv_diff_future_seed.py (LOG_JSONL=...).

Usage:
  python tools/summarize_jsonl.py path/to/*.jsonl
  python tools/summarize_jsonl.py exp/**/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import glob
from typing import Any, Dict, List, Tuple


def load_records(paths: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="jsonl paths or globs")
    ap.add_argument(
        "--metrics",
        default="val_loss,kvsort_ood_exact,permfill_ood_exact,sudoku_solve,maskacc_val",
        help="comma-separated metric keys to include if present",
    )
    args = ap.parse_args()

    paths: List[str] = []
    for x in args.paths:
        hits = glob.glob(x, recursive=True)
        if hits:
            paths.extend(hits)
        else:
            paths.append(x)

    recs = load_records(paths)
    evals = [r for r in recs if r.get("event") == "eval"]

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    best: Dict[Tuple, Dict[str, Any]] = {}
    for r in evals:
        key = (
            r.get("weights_path", ""),
            r.get("model", ""),
            r.get("future_seed", ""),
            r.get("decode", ""),
            r.get("random_seed", ""),
        )
        it = int(r.get("iter", -1))
        if key not in best or it > int(best[key].get("iter", -1)):
            best[key] = r

    rows = []
    for k, r in best.items():
        row = {
            "weights": k[0],
            "model": k[1],
            "fs": k[2],
            "decode": k[3],
            "seed": k[4],
            "iter": r.get("iter", ""),
        }
        for m in metrics:
            if m in r:
                row[m] = r[m]
        rows.append(row)

    rows.sort(key=lambda x: (x["weights"], x["model"], x["fs"], x["decode"], x["seed"]))

    # markdown table
    cols = ["weights", "model", "fs", "decode", "seed", "iter"] + [m for m in metrics if any(m in r for r in rows)]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        print("| " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main()

