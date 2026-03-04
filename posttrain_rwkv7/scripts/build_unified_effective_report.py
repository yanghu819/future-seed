#!/usr/bin/env python3
"""Build one unified report for all effective experiments across rounds.

Effective experiment definition in this report:
- For a given (round, task), `med` stage exists for both baseline and at least one FS config.
- Best FS `med` token accuracy strictly exceeds baseline (`delta_pp > 0`).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median
from typing import Dict, List, Tuple


ROUND_RE = re.compile(r"_round(\d+)")


@dataclass
class TaskRoundRow:
    round_id: int
    source_file: str
    task: str
    family: str
    baseline_quick: float | None
    best_quick_cfg: str | None
    best_quick_acc: float | None
    best_quick_delta_pp: float | None
    baseline_med: float | None
    best_med_cfg: str | None
    best_med_acc: float | None
    best_med_delta_pp: float | None
    compare_mode: str | None
    anchor_acc: float | None
    best_med_delta_vs_anchor_pp: float | None


def infer_family(task: str) -> str:
    t = task or ""
    if t.startswith("mbpp_longctx"):
        return "mbpp_longctx"
    if t.startswith("mbpp"):
        return "mbpp"
    if t.startswith("arc_mc") or t.startswith("arc_"):
        return "arc_mc"
    if t.startswith("protein_ss"):
        return "protein_ss"
    if t.startswith("protein_contact"):
        return "protein_contact"
    if t.startswith("squad"):
        return "squad"
    if t.startswith("punc"):
        return "punc"
    if t.startswith("hotpot"):
        return "hotpot"
    if t.startswith("wiki") or t.startswith("wikitext"):
        return "wiki"
    if t.startswith("sudoku"):
        return "sudoku"
    if t.startswith("nt") or "nucleotide" in t:
        return "nt"
    if t.startswith("sat"):
        return "sat"
    if t.startswith("tsp"):
        return "tsp"
    return t.split("_seed")[0] if "_seed" in t else t


def parse_round_from_name(path: str) -> int | None:
    m = ROUND_RE.search(os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


def collect_record_files(results_dir: str) -> List[str]:
    candidates = sorted(glob.glob(os.path.join(results_dir, "_round*.jsonl")))
    out: List[str] = []
    for p in candidates:
        b = os.path.basename(p)
        if "records" not in b:
            continue
        if b.endswith(".partial.jsonl"):
            continue
        out.append(p)
    return out


def parse_rows(files: List[str]) -> List[TaskRoundRow]:
    rows: List[TaskRoundRow] = []

    for fp in files:
        round_from_file = parse_round_from_name(fp)
        if round_from_file is None:
            continue

        # (round, task) -> stage -> cfg -> best_acc
        bucket: Dict[Tuple[int, str], Dict[str, Dict[str, float]]] = defaultdict(
            lambda: {"quick": {}, "med": {}}
        )

        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                stage = rec.get("stage")
                if stage not in ("quick", "med"):
                    continue
                if rec.get("status") != "ok":
                    continue

                task = rec.get("task")
                cfg = rec.get("config")
                acc = rec.get("best_val_tok_acc")
                if task is None or cfg is None or acc is None:
                    continue

                round_id = int(rec.get("round", round_from_file))
                key = (round_id, str(task))
                cur = bucket[key][stage].get(str(cfg))
                acc_v = float(acc)
                if cur is None or acc_v > cur:
                    bucket[key][stage][str(cfg)] = acc_v

        for (round_id, task), stages in bucket.items():
            q = stages["quick"]
            m = stages["med"]

            bq = q.get("baseline")
            bm = m.get("baseline")

            best_q_cfg = None
            best_q_acc = None
            if bq is not None:
                for cfg, acc in q.items():
                    if cfg == "baseline":
                        continue
                    if best_q_acc is None or acc > best_q_acc:
                        best_q_cfg = cfg
                        best_q_acc = acc

            best_m_cfg = None
            best_m_acc = None
            for cfg, acc in m.items():
                if cfg == "baseline":
                    continue
                if best_m_acc is None or acc > best_m_acc:
                    best_m_cfg = cfg
                    best_m_acc = acc

            row = TaskRoundRow(
                round_id=round_id,
                source_file=os.path.basename(fp),
                task=task,
                family=infer_family(task),
                baseline_quick=bq,
                best_quick_cfg=best_q_cfg,
                best_quick_acc=best_q_acc,
                best_quick_delta_pp=((best_q_acc - bq) * 100 if (best_q_acc is not None and bq is not None) else None),
                baseline_med=bm,
                best_med_cfg=best_m_cfg,
                best_med_acc=best_m_acc,
                best_med_delta_pp=((best_m_acc - bm) * 100 if (best_m_acc is not None and bm is not None) else None),
                compare_mode=(
                    "med_vs_med"
                    if (best_m_acc is not None and bm is not None)
                    else ("med_vs_quick" if (best_m_acc is not None and bq is not None) else None)
                ),
                anchor_acc=(bm if bm is not None else bq),
                best_med_delta_vs_anchor_pp=(
                    (best_m_acc - bm) * 100
                    if (best_m_acc is not None and bm is not None)
                    else (((best_m_acc - bq) * 100) if (best_m_acc is not None and bq is not None) else None)
                ),
            )
            rows.append(row)

    rows.sort(key=lambda r: (r.round_id, r.task, r.source_file))
    return rows


def fmt_pct(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v*100:.2f}%"


def fmt_pp(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:+.2f}pp"


def write_csv(path: str, rows: List[TaskRoundRow], only_effective: bool) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "round",
                "source_file",
                "task",
                "family",
                "baseline_quick_acc",
                "best_quick_cfg",
                "best_quick_acc",
                "best_quick_delta_pp",
                "baseline_med_acc",
                "best_med_cfg",
                "best_med_acc",
                "best_med_delta_pp",
                "compare_mode",
                "anchor_acc",
                "best_med_delta_vs_anchor_pp",
                "effective",
            ]
        )
        for r in rows:
            eff = r.best_med_delta_vs_anchor_pp is not None and r.best_med_delta_vs_anchor_pp > 0
            if only_effective and not eff:
                continue
            w.writerow(
                [
                    r.round_id,
                    r.source_file,
                    r.task,
                    r.family,
                    "" if r.baseline_quick is None else f"{r.baseline_quick:.10f}",
                    r.best_quick_cfg or "",
                    "" if r.best_quick_acc is None else f"{r.best_quick_acc:.10f}",
                    "" if r.best_quick_delta_pp is None else f"{r.best_quick_delta_pp:.10f}",
                    "" if r.baseline_med is None else f"{r.baseline_med:.10f}",
                    r.best_med_cfg or "",
                    "" if r.best_med_acc is None else f"{r.best_med_acc:.10f}",
                    "" if r.best_med_delta_pp is None else f"{r.best_med_delta_pp:.10f}",
                    r.compare_mode or "",
                    "" if r.anchor_acc is None else f"{r.anchor_acc:.10f}",
                    "" if r.best_med_delta_vs_anchor_pp is None else f"{r.best_med_delta_vs_anchor_pp:.10f}",
                    "1" if eff else "0",
                ]
            )


def build_markdown(
    out_md: str,
    rows: List[TaskRoundRow],
    files_n: int,
) -> None:
    comparable = [r for r in rows if r.best_med_delta_vs_anchor_pp is not None]
    strict_comparable = [r for r in rows if r.best_med_delta_pp is not None]
    effective = [r for r in comparable if r.best_med_delta_vs_anchor_pp is not None and r.best_med_delta_vs_anchor_pp > 0]
    negative = [r for r in comparable if r.best_med_delta_vs_anchor_pp is not None and r.best_med_delta_vs_anchor_pp <= 0]

    rounds = sorted({r.round_id for r in rows})
    round_span = f"{rounds[0]}-{rounds[-1]}" if rounds else "N/A"

    family_rows: Dict[str, List[TaskRoundRow]] = defaultdict(list)
    for r in comparable:
        family_rows[r.family].append(r)

    family_stats = []
    for fam, rs in family_rows.items():
        deltas = [x.best_med_delta_pp for x in rs if x.best_med_delta_pp is not None]
        anchor_deltas = [x.best_med_delta_vs_anchor_pp for x in rs if x.best_med_delta_vs_anchor_pp is not None]
        if not anchor_deltas:
            continue
        pos = [d for d in anchor_deltas if d > 0]
        neg = [d for d in anchor_deltas if d <= 0]
        family_stats.append(
            {
                "family": fam,
                "n": len(rs),
                "pos_n": len(pos),
                "pos_rate": (len(pos) / len(rs)) if rs else 0.0,
                "mean_pp": mean(anchor_deltas),
                "median_pp": median(anchor_deltas),
                "best_pp": max(anchor_deltas),
                "worst_pp": min(anchor_deltas),
            }
        )

    family_stats.sort(key=lambda x: (-x["pos_rate"], -x["mean_pp"], -x["n"]))

    top_effective = sorted(effective, key=lambda r: r.best_med_delta_pp or -1e9, reverse=True)[:50]

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Unified Effective Experiment Report\n\n")
        f.write("This document unifies all **effective** experiments in one place, from earliest logged rounds to latest.\n")
        f.write("\n")
        f.write("## Scope & Rule\n")
        f.write(f"- Parsed record files: **{files_n}**\n")
        f.write(f"- Round span: **{round_span}**\n")
        f.write(f"- Comparable items (best FS med + anchor baseline exists): **{len(comparable)}**\n")
        f.write(f"- Effective items (`best_med_delta_vs_anchor_pp > 0`): **{len(effective)}**\n")
        f.write(f"- Non-effective comparable items: **{len(negative)}**\n")
        if comparable:
            deltas = [r.best_med_delta_vs_anchor_pp for r in comparable if r.best_med_delta_vs_anchor_pp is not None]
            f.write(f"- Comparable med mean/median: **{mean(deltas):+.2f}pp / {median(deltas):+.2f}pp**\n")
        f.write(f"- Strict `med_vs_med` comparable items: **{len(strict_comparable)}**\n")
        f.write(f"- Legacy `med_vs_quick` fallback items: **{len(comparable) - len(strict_comparable)}**\n")
        f.write("\n")
        f.write(
            "Effective = for the same `(round, task)`, best FS `med` token-accuracy is above anchor baseline. "
            "Anchor uses `med baseline` when present; otherwise falls back to `quick baseline`.\n\n"
        )

        f.write("## Family Stability (All Comparable Med)\n\n")
        f.write("| family | comparable_n | effective_n | effective_rate | mean_delta_pp | median_delta_pp | best_pp | worst_pp |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for s in family_stats:
            f.write(
                f"| {s['family']} | {s['n']} | {s['pos_n']} | {s['pos_rate']*100:.1f}% | {s['mean_pp']:+.2f}pp | {s['median_pp']:+.2f}pp | {s['best_pp']:+.2f}pp | {s['worst_pp']:+.2f}pp |\n"
            )

        f.write("\n## Top Effective Runs (Global Top 50 by med delta)\n\n")
        f.write("| round | task | family | best_med_cfg | baseline_med | best_med | delta_pp | source |\n")
        f.write("|---:|---|---|---|---:|---:|---:|---|\n")
        for r in top_effective:
            f.write(
                "| {round} | {task} | {family} | {cfg} | {b} | {a} | {d} | {src} |\n".format(
                    round=r.round_id,
                    task=r.task,
                    family=r.family,
                    cfg=r.best_med_cfg or "-",
                    b=fmt_pct(r.anchor_acc),
                    a=fmt_pct(r.best_med_acc),
                    d=fmt_pp(r.best_med_delta_vs_anchor_pp),
                    src=r.source_file,
                )
            )

        f.write("\n## Full Effective Timeline (Chronological, All Positive med)\n\n")
        f.write("| round | task | family | compare_mode | best_med_cfg | anchor_baseline | best_med | delta_pp | quick_delta_pp | source |\n")
        f.write("|---:|---|---|---|---:|---:|---:|---:|---|\n")
        for r in sorted(effective, key=lambda x: (x.round_id, x.task, x.source_file)):
            f.write(
                "| {round} | {task} | {family} | {mode} | {cfg} | {b} | {a} | {d} | {qd} | {src} |\n".format(
                    round=r.round_id,
                    task=r.task,
                    family=r.family,
                    mode=r.compare_mode or "-",
                    cfg=r.best_med_cfg or "-",
                    b=fmt_pct(r.anchor_acc),
                    a=fmt_pct(r.best_med_acc),
                    d=fmt_pp(r.best_med_delta_vs_anchor_pp),
                    qd=fmt_pp(r.best_quick_delta_pp),
                    src=r.source_file,
                )
            )

        f.write("\n## Reference Files\n")
        f.write("- Full comparable table: `results/_unified_med_comparisons.csv`\n")
        f.write("- Effective-only table: `results/_unified_effective_med_runs.csv`\n")
        f.write("- Existing rolling log: `results/_rolling_round_log.md`\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_md", default="results/UNIFIED_EFFECTIVE_EXPERIMENTS.md")
    ap.add_argument("--out_med_csv", default="results/_unified_med_comparisons.csv")
    ap.add_argument("--out_effective_csv", default="results/_unified_effective_med_runs.csv")
    args = ap.parse_args()

    files = collect_record_files(args.results_dir)
    rows = parse_rows(files)

    write_csv(args.out_med_csv, rows, only_effective=False)
    write_csv(args.out_effective_csv, rows, only_effective=True)
    build_markdown(args.out_md, rows, files_n=len(files))

    comparable = [r for r in rows if r.best_med_delta_vs_anchor_pp is not None]
    effective = [r for r in comparable if r.best_med_delta_vs_anchor_pp is not None and r.best_med_delta_vs_anchor_pp > 0]

    print(f"files={len(files)} rows={len(rows)} comparable_med={len(comparable)} effective={len(effective)}")
    print(f"wrote: {args.out_md}")
    print(f"wrote: {args.out_med_csv}")
    print(f"wrote: {args.out_effective_csv}")


if __name__ == "__main__":
    main()
