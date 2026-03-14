#!/usr/bin/env python3
"""Summarize sudoku9_inplace_refine runs by board exact solve rate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def best_record(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.jsonl"
    best: dict[str, Any] | None = None
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row_key = (
                    float(row.get("val_focus_board_exact_mean", 0.0)),
                    float(row.get("val_focus_masked_acc_mean", 0.0)),
                )
                best_key = (
                    float(best.get("val_focus_board_exact_mean", 0.0)),
                    float(best.get("val_focus_masked_acc_mean", 0.0)),
                ) if best is not None else None
                if best is None or row_key > best_key:
                    best = row
    summary_path = run_dir / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    return {
        "run_dir": str(run_dir),
        "config": summary.get("tag", run_dir.name),
        "mode": summary.get("mode"),
        "best": best,
        "summary": summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="runs")
    ap.add_argument("--contains", type=str, default="sudoku9_inplace_refine")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for summary_path in root.glob("**/summary.json"):
        if args.contains not in str(summary_path):
            continue
        rows.append(best_record(summary_path.parent))

    if not rows:
        print("No sudoku9_inplace runs found.")
        return

    rows.sort(
        key=lambda row: (
            float((row["best"] or {}).get("val_focus_board_exact_mean", -1.0)),
            float((row["best"] or {}).get("val_focus_masked_acc_mean", -1.0)),
        ),
        reverse=True,
    )
    print("config | mode | best focus exact | best focus masked acc | step | val32 | val36 | val40 | final test focus exact")
    print("---|---|---:|---:|---:|---:|---:|---:|---:")
    for row in rows:
        best = row["best"] or {}
        summary = row["summary"] or {}
        final_test = summary.get("final_test", {})
        focus_test = final_test.get("focus_board_exact_mean")
        focus_test_text = f"{float(focus_test)*100:.2f}%" if focus_test is not None else "-"
        print(
            f"{row['config']} | {row.get('mode','-')} | {float(best.get('val_focus_board_exact_mean', 0.0))*100:.2f}% | "
            f"{float(best.get('val_focus_masked_acc_mean', 0.0))*100:.2f}% | {int(best.get('step', 0))} | "
            f"{float(best.get('val_board_exact_32', 0.0))*100:.2f}% | {float(best.get('val_board_exact_36', 0.0))*100:.2f}% | "
            f"{float(best.get('val_board_exact_40', 0.0))*100:.2f}% | {focus_test_text}"
        )


if __name__ == "__main__":
    main()
