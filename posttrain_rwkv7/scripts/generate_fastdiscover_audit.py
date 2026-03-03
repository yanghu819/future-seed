#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def family(task: str) -> str:
    if task.startswith("arc_mc_"):
        return "arc_mc"
    if task.startswith("mbpp_longctx_"):
        return "mbpp_longctx"
    if task.startswith("mbpp_"):
        return "mbpp"
    if task.startswith("protein_ss_"):
        return "protein_ss"
    if task.startswith("protein_contact_"):
        return "protein_contact"
    if task.startswith("hotpot_"):
        return "hotpot"
    if task.startswith("squad_"):
        return "squad"
    if task.startswith("wiki_"):
        return "wiki"
    return "other"


def collect_files(results_dir: Path, round_from: int, round_to: int) -> List[Tuple[int, Path]]:
    pat = re.compile(r"_round(\d+)_fastdiscover_records\.jsonl$")
    out: List[Tuple[int, Path]] = []
    for p in results_dir.glob("_round*_fastdiscover_records.jsonl"):
        m = pat.search(p.name)
        if not m:
            continue
        r = int(m.group(1))
        if round_from <= r <= round_to:
            out.append((r, p))
    out.sort(key=lambda x: x[0])
    return out


def write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, default=Path("results"))
    ap.add_argument("--round_from", type=int, default=77)
    ap.add_argument("--round_to", type=int, default=569)
    ap.add_argument(
        "--out_prefix",
        type=str,
        default="results/_audit_round77_569",
        help="prefix without suffix; script writes _analysis.md/_med_ledger.csv/_quick_ledger.csv/_task_combo.csv",
    )
    args = ap.parse_args()

    files = collect_files(args.results_dir, args.round_from, args.round_to)
    if not files:
        raise RuntimeError("no fastdiscover record files found in range")

    by_stage: Dict[Tuple[int, str, str], List[Tuple[str, float]]] = defaultdict(list)
    prune_counts: Counter[str] = Counter()

    for _, path in files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("status") == "pruned":
                    prune_counts[str(obj.get("gate_reason", "unknown"))] += 1
                acc = obj.get("best_val_tok_acc")
                if acc is None:
                    continue
                by_stage[(int(obj.get("round", -1)), str(obj.get("task", "")), str(obj.get("stage", "")))].append(
                    (str(obj.get("config", "")), float(acc))
                )

    quick_rows: List[dict] = []
    med_rows: List[dict] = []

    for (r, task, stage), rows in by_stage.items():
        base = None
        for cfg, acc in rows:
            if cfg == "baseline":
                base = acc
                break
        if base is None:
            continue
        cands = [(cfg, acc) for cfg, acc in rows if cfg != "baseline"]
        if not cands:
            continue
        best_cfg, best_acc = max(cands, key=lambda x: x[1] - base)
        dpp = (best_acc - base) * 100.0
        rec = {
            "round": r,
            "task": task,
            "family": family(task),
            "baseline_acc": base,
            "best_cfg": best_cfg,
            "best_acc": best_acc,
            "best_delta_pp": dpp,
            "outcome": "positive" if dpp > 0 else ("negative" if dpp < 0 else "flat"),
        }
        if stage == "quick":
            quick_rows.append(rec)
        elif stage == "med":
            med_rows.append(rec)

    qidx = {(x["round"], x["task"]): x for x in quick_rows}
    midx = {(x["round"], x["task"]): x for x in med_rows}
    combo_rows: List[dict] = []
    for k in sorted(set(qidx) | set(midx)):
        q = qidx.get(k)
        m = midx.get(k)
        combo_rows.append(
            {
                "round": k[0],
                "task": k[1],
                "family": family(k[1]),
                "quick_delta_pp": None if q is None else q["best_delta_pp"],
                "quick_cfg": None if q is None else q["best_cfg"],
                "med_delta_pp": None if m is None else m["best_delta_pp"],
                "med_cfg": None if m is None else m["best_cfg"],
                "reversal_qpos_mnonpos": bool(
                    q is not None and m is not None and q["best_delta_pp"] > 0 and m["best_delta_pp"] <= 0
                ),
            }
        )

    prefix = Path(args.out_prefix)
    md_path = Path(str(prefix) + "_analysis.md")
    med_csv = Path(str(prefix) + "_med_ledger.csv")
    quick_csv = Path(str(prefix) + "_quick_ledger.csv")
    combo_csv = Path(str(prefix) + "_task_combo.csv")

    write_csv(
        quick_csv,
        quick_rows,
        ["round", "task", "family", "baseline_acc", "best_cfg", "best_acc", "best_delta_pp", "outcome"],
    )
    write_csv(
        med_csv,
        med_rows,
        ["round", "task", "family", "baseline_acc", "best_cfg", "best_acc", "best_delta_pp", "outcome"],
    )
    write_csv(
        combo_csv,
        combo_rows,
        ["round", "task", "family", "quick_delta_pp", "quick_cfg", "med_delta_pp", "med_cfg", "reversal_qpos_mnonpos"],
    )

    q = [x["best_delta_pp"] for x in quick_rows]
    m = [x["best_delta_pp"] for x in med_rows]
    top_pos = sorted(med_rows, key=lambda x: x["best_delta_pp"], reverse=True)[:25]
    top_neg = sorted(med_rows, key=lambda x: x["best_delta_pp"])[:25]
    revs = [x for x in combo_rows if x["reversal_qpos_mnonpos"]]
    rev_top = sorted(
        revs,
        key=lambda x: (float(x["med_delta_pp"] or 0.0) - float(x["quick_delta_pp"] or 0.0)),
    )[:25]

    fam_stats = []
    for fam in sorted(set([x["family"] for x in quick_rows + med_rows])):
        qf = [x["best_delta_pp"] for x in quick_rows if x["family"] == fam]
        mf = [x["best_delta_pp"] for x in med_rows if x["family"] == fam]
        fam_stats.append((fam, qf, mf))

    lines: List[str] = []
    lines.append(f"# Complete Audit (round{args.round_from}-{args.round_to})")
    lines.append("")
    lines.append(f"- Files parsed: {len(files)}")
    lines.append(f"- Quick rows: {len(quick_rows)}")
    lines.append(f"- Med rows: {len(med_rows)}")
    lines.append(f"- Quick prune decisions: {prune_counts.get('quick_prune', 0)}")
    lines.append(f"- Med skip decisions: {prune_counts.get('med_skip', 0)}")
    lines.append("")
    if q:
        lines.append(f"- Quick mean/median: {statistics.mean(q):+.2f}pp / {statistics.median(q):+.2f}pp")
        lines.append(f"- Quick positive rate: {sum(1 for x in q if x > 0) / len(q) * 100:.1f}%")
    if m:
        lines.append(f"- Med mean/median: {statistics.mean(m):+.2f}pp / {statistics.median(m):+.2f}pp")
        lines.append(f"- Med positive rate: {sum(1 for x in m if x > 0) / len(m) * 100:.1f}%")
        best_row = max(med_rows, key=lambda x: x["best_delta_pp"])
        worst_row = min(med_rows, key=lambda x: x["best_delta_pp"])
        lines.append(
            f"- Best med: round{best_row['round']} `{best_row['task']}` / `{best_row['best_cfg']}` {best_row['best_delta_pp']:+.2f}pp"
        )
        lines.append(
            f"- Worst med: round{worst_row['round']} `{worst_row['task']}` / `{worst_row['best_cfg']}` {worst_row['best_delta_pp']:+.2f}pp"
        )
    lines.append("")
    lines.append("## Family Stats")
    lines.append("")
    lines.append("| Family | Quick n | Quick mean(pp) | Quick pos% | Med n | Med mean(pp) | Med pos% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for fam, qf, mf in fam_stats:
        qn = len(qf)
        mn = len(mf)
        qmean = statistics.mean(qf) if qf else 0.0
        mmean = statistics.mean(mf) if mf else 0.0
        qpos = (sum(1 for x in qf if x > 0) / qn * 100.0) if qn else 0.0
        mpos = (sum(1 for x in mf if x > 0) / mn * 100.0) if mn else 0.0
        lines.append(f"| {fam} | {qn} | {qmean:+.2f} | {qpos:.1f}% | {mn} | {mmean:+.2f} | {mpos:.1f}% |")
    lines.append("")
    lines.append("## Top 25 Positive Med Runs")
    lines.append("")
    for x in top_pos:
        lines.append(f"- round{x['round']} `{x['task']}` / `{x['best_cfg']}`: **{x['best_delta_pp']:+.2f}pp**")
    lines.append("")
    lines.append("## Top 25 Negative Med Runs")
    lines.append("")
    for x in top_neg:
        lines.append(f"- round{x['round']} `{x['task']}` / `{x['best_cfg']}`: **{x['best_delta_pp']:+.2f}pp**")
    lines.append("")
    lines.append("## Top 25 Quick->Med Reversals (quick>0, med<=0)")
    lines.append("")
    for x in rev_top:
        lines.append(
            f"- round{x['round']} `{x['task']}`: quick {float(x['quick_delta_pp']):+.2f}pp -> med {float(x['med_delta_pp']):+.2f}pp"
        )
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(md_path)
    print(med_csv)
    print(quick_csv)
    print(combo_csv)


if __name__ == "__main__":
    main()
