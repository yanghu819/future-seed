#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    txt = path.read_text(encoding="utf-8", errors="ignore")
    line = None
    for ln in txt.splitlines():
        if ln.startswith("L="):
            line = ln.strip()
            break
    if line is None:
        return None
    # Example:
    # L=4096 pairs=3 mean d_acc=+0.0012 std=0.0388 | mean d_loss=-0.0192 std=0.1669 | sign(+/0/-)=2/0/1
    m = re.search(
        r"L=(\d+)\s+pairs=(\d+)\s+mean d_acc=([+-]?\d+\.\d+)\s+std=(\d+\.\d+)\s+\|\s+mean d_loss=([+-]?\d+\.\d+)\s+std=(\d+\.\d+)\s+\|\s+sign\(\+/0/-\)=(\d+)/(\d+)/(\d+)",
        line,
    )
    if not m:
        return None
    return {
        "L": int(m.group(1)),
        "pairs": int(m.group(2)),
        "mean_d_acc": float(m.group(3)),
        "std_d_acc": float(m.group(4)),
        "mean_d_loss": float(m.group(5)),
        "std_d_loss": float(m.group(6)),
        "n_pos": int(m.group(7)),
        "n_zero": int(m.group(8)),
        "n_neg": int(m.group(9)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs")
    args = ap.parse_args()

    runs = Path(args.runs_dir)
    variants = [
        ("R6 baseline (alpha=-2,lstart=6)", "runs/_summary_hotpot_qafter_stabilized_len4096_r6_s012.txt"),
        ("R8 weaker alpha (alpha=-4,lstart=6)", "runs/_summary_hotpot_qafter_stabilized_len4096_r8_alpha_m4_s012.txt"),
        ("R9 linear schedule", "runs/_summary_hotpot_qafter_stabilized_len4096_r9_sched_linear_s012.txt"),
        ("R10 deeper-start (lstart=10)", "runs/_summary_hotpot_qafter_stabilized_len4096_r10_lstart10_s012.txt"),
    ]

    print("Hotpot q-after L=4096 variant comparison")
    print("-" * 92)
    print(
        "{:<34} {:>10} {:>10} {:>11} {:>11} {:>11}".format(
            "variant", "mean_d_acc", "std_d_acc", "mean_d_loss", "std_d_loss", "sign(+/0/-)"
        )
    )
    print("-" * 92)
    for name, rel in variants:
        path = Path(rel)
        if not path.is_absolute():
            path = runs / Path(rel).name
        rec = parse_summary(path)
        if rec is None:
            print("{:<34} {:>10}".format(name, "PENDING"))
            continue
        sgn = f"{rec['n_pos']}/{rec['n_zero']}/{rec['n_neg']}"
        print(
            "{:<34} {:>+10.4f} {:>10.4f} {:>+11.4f} {:>11.4f} {:>11}".format(
                name, rec["mean_d_acc"], rec["std_d_acc"], rec["mean_d_loss"], rec["std_d_loss"], sgn
            )
        )


if __name__ == "__main__":
    main()

