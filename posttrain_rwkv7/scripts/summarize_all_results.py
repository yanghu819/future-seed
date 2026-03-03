#!/usr/bin/env python3
"""
Aggregate all `results/_summary_*.txt` files into machine-readable outputs.

Outputs:
  - JSONL records per (summary_file, L row)
  - markdown table for quick paper updates
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


METRIC_RE = re.compile(
    r"(?:L=(?P<L>\d+)\s+)?pairs=(?P<pairs>\d+)\s+mean d_acc=(?P<d_acc>[+-]?\d+(?:\.\d+)?)\s+std=(?P<d_acc_std>[+-]?\d+(?:\.\d+)?)"
)
MEAN_METRIC_RE = re.compile(
    r"mean\s+(?P<name>d_[a-z_]+)=(?P<value>[+-]?\d+(?:\.\d+)?)\s+std=(?P<std>[+-]?\d+(?:\.\d+)?)"
)
SIGN_RE = re.compile(r"sign\(\+/0/-\)=(?P<pos>\d+)/(?P<zero>\d+)/(?P<neg>\d+)")


def _infer_task(name: str) -> str:
    lowered = name.lower()
    for key in ["arc", "hotpot", "mbpp", "sudoku", "protein", "squad", "longfill"]:
        if key in lowered:
            return key
    return "other"


def _infer_ordering(name: str) -> str:
    lowered = name.lower()
    if "qafter" in lowered:
        return "q_after"
    if "qfirst" in lowered:
        return "q_first"
    if "optionsfirst" in lowered:
        return "options_first"
    if "prefix" in lowered:
        return "prefix"
    if "suffix" in lowered:
        return "suffix"
    return "unspecified"


def _safe_float(v: str | None) -> float | None:
    if v is None:
        return None
    if v == "None":
        return None
    return float(v)


def parse_summary_file(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    filter_obj: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for line in lines:
        if line.startswith("filter:"):
            payload = line.split("filter:", 1)[1].strip()
            try:
                filter_obj = ast.literal_eval(payload)
            except Exception:
                filter_obj = {"_parse_error": payload}
            continue

        if "pairs=" not in line or "mean d_acc=" not in line:
            continue

        m = METRIC_RE.search(line)
        if not m:
            continue

        L = m.group("L")
        if L is None:
            L = filter_obj.get("max_prompt_tokens")
        if L is None:
            L = filter_obj.get("max_prompt")
        if L is None and "base" in filter_obj:
            try:
                b = int(filter_obj["base"])
                L = (b * b) * (b * b)
            except Exception:
                L = None

        record: dict[str, Any] = {
            "summary_file": str(path.as_posix()),
            "summary_name": path.name,
            "task": _infer_task(path.name),
            "ordering": _infer_ordering(path.name),
            "L": int(L) if L is not None else -1,
            "pairs": int(m.group("pairs")),
            "mean_d_acc": float(m.group("d_acc")),
            "std_d_acc": float(m.group("d_acc_std")),
            "mean_d_loss": None,
            "std_d_loss": None,
            "mean_d_seq": None,
            "std_d_seq": None,
            "sign_pos": None,
            "sign_zero": None,
            "sign_neg": None,
            "filter": filter_obj,
        }

        for mm in MEAN_METRIC_RE.finditer(line):
            name = mm.group("name")
            value = float(mm.group("value"))
            std = float(mm.group("std"))
            if name == "d_loss":
                record["mean_d_loss"] = value
                record["std_d_loss"] = std
            elif name == "d_seq":
                record["mean_d_seq"] = value
                record["std_d_seq"] = std

        sm = SIGN_RE.search(line)
        if sm:
            record["sign_pos"] = int(sm.group("pos"))
            record["sign_zero"] = int(sm.group("zero"))
            record["sign_neg"] = int(sm.group("neg"))

        rows.append(record)

    return rows


def load_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    table: dict[str, dict[str, Any]] = {}
    for exp in data.get("experiments", []):
        key = exp.get("summary")
        if isinstance(key, str):
            table[key] = exp
    return table


def to_markdown(rows: list[dict[str, Any]]) -> str:
    hdr = [
        "| task | ordering | summary | L | mean_d_acc | mean_d_loss | mean_d_seq | sign(+/0/-) | role |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    body = []
    for r in rows:
        sign = "-"
        if r["sign_pos"] is not None:
            sign = f"{r['sign_pos']}/{r['sign_zero']}/{r['sign_neg']}"
        role = r.get("paper_role", "")
        body.append(
            "| {task} | {ordering} | `{name}` | {L} | {dacc:+.4f} | {dloss} | {dseq} | {sign} | {role} |".format(
                task=r["task"],
                ordering=r["ordering"],
                name=r["summary_name"],
                L=r["L"],
                dacc=r["mean_d_acc"],
                dloss=f"{r['mean_d_loss']:+.4f}" if r["mean_d_loss"] is not None else "-",
                dseq=f"{r['mean_d_seq']:+.4f}" if r["mean_d_seq"] is not None else "-",
                sign=sign,
                role=role or "-",
            )
        )
    return "\n".join(hdr + body) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, default=Path("results"))
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("paper/exp_manifest.json"),
        help="Optional experiment manifest with paper_role/status metadata.",
    )
    ap.add_argument("--out_jsonl", type=Path, default=Path("results/_aggregate_results.jsonl"))
    ap.add_argument("--out_md", type=Path, default=Path("results/_aggregate_results.md"))
    args = ap.parse_args()

    if not args.results_dir.exists():
        raise FileNotFoundError(f"results dir not found: {args.results_dir}")

    manifest = load_manifest(args.manifest if args.manifest.exists() else None)

    rows: list[dict[str, Any]] = []
    for p in sorted(args.results_dir.glob("_summary_*.txt")):
        parsed = parse_summary_file(p)
        for r in parsed:
            exp = manifest.get(r["summary_file"])
            if exp:
                r["exp_id"] = exp.get("id")
                r["paper_role"] = exp.get("paper_role")
                r["manifest_mean_d_acc"] = _safe_float(
                    str(exp.get("mean_d_acc")) if exp.get("mean_d_acc") is not None else None
                )
                r["manifest_status"] = exp.get("status")
        rows.extend(parsed)

    rows.sort(key=lambda x: (x["task"], x["ordering"], x["summary_name"], x["L"]))

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    args.out_md.write_text(to_markdown(rows), encoding="utf-8")

    task_counts: dict[str, int] = {}
    for r in rows:
        task_counts[r["task"]] = task_counts.get(r["task"], 0) + 1
    print(f"wrote {len(rows)} rows -> {args.out_jsonl} / {args.out_md}")
    print("task_counts:", task_counts)


if __name__ == "__main__":
    main()
