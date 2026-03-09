#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
METRICS_PATH = ROOT / "data" / "metrics.json"
POSTTRAIN_README = REPO / "posttrain_rwkv7" / "README.md"
RUNS_DIR = REPO / "posttrain_rwkv7" / "runs"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_posttrain_readme_table() -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    pattern = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`\+?([-\d.]+)pp`\s*\|\s*`\+?([-\d.]+)pp`\s*\|\s*`(\d+)/(\d+)`\s*\|\s*(.+?)\s*\|$")
    for line in POSTTRAIN_README.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        task, best, median, pos, total, judgment = m.groups()
        rows[task] = {
            "best": float(best),
            "median": float(median),
            "pos": int(pos),
            "total": int(total),
            "judgment": judgment,
        }
    return rows


def load_round_rows(round_id: int) -> list[dict]:
    path = RUNS_DIR / f"_round{round_id}_fastdiscover_records.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def best_stage_row(rows: list[dict], *, task: str, stage: str, config: str) -> dict:
    matches = [
        row for row in rows
        if row.get("task") == task
        and row.get("stage") == stage
        and row.get("config") == config
        and row.get("status") in {"ok", "skip_reuse"}
        and row.get("best_val_tok_acc") is not None
    ]
    if not matches:
        raise AssertionError(f"missing row: task={task} stage={stage} config={config}")
    return max(matches, key=lambda row: float(row["best_val_tok_acc"]))


def assert_close(actual: float, expected: float, *, label: str, atol: float = 0.01) -> None:
    if math.isclose(actual, expected, abs_tol=atol):
        return
    raise AssertionError(f"{label}: expected {expected:.2f}, got {actual:.2f}")


def verify_main_summary() -> list[str]:
    metrics = load_json(METRICS_PATH)["posttrain"]["main_summary"]
    readme_rows = parse_posttrain_readme_table()
    out = []
    for row in metrics:
        task = row["task"]
        if task not in readme_rows:
            raise AssertionError(f"task missing from posttrain README: {task}")
        rr = readme_rows[task]
        assert_close(float(row["best"]), float(rr["best"]), label=f"{task} best")
        assert_close(float(row["median"]), float(rr["median"]), label=f"{task} median")
        if int(row["pos"]) != int(rr["pos"]) or int(row["total"]) != int(rr["total"]):
            raise AssertionError(
                f"{task} positive count mismatch: expected {rr['pos']}/{rr['total']}, "
                f"got {row['pos']}/{row['total']}"
            )
        if str(row["judgment"]) != str(rr["judgment"]):
            raise AssertionError(
                f"{task} judgment mismatch: expected '{rr['judgment']}', got '{row['judgment']}'"
            )
        if "confirm" not in row:
            raise AssertionError(f"{task} missing confirm field in metrics.json")
        out.append(
            f"{task}: main_summary matches README "
            f"(best={row['best']:.2f}pp median={row['median']:.2f}pp pos={row['pos']}/{row['total']})"
        )
    return out


def verify_closure_highlights() -> list[str]:
    metrics = load_json(METRICS_PATH)["posttrain"]["closure_highlights"]
    out = []
    for row in metrics:
        rows = load_round_rows(int(row["round"]))
        baseline = best_stage_row(
            rows,
            task=str(row["task"]),
            stage=str(row["baseline_stage"]),
            config="baseline",
        )
        candidate = best_stage_row(
            rows,
            task=str(row["task"]),
            stage=str(row["stage"]),
            config=str(row["config"]),
        )
        delta = (float(candidate["best_val_tok_acc"]) - float(baseline["best_val_tok_acc"])) * 100.0
        assert_close(delta, float(row["delta"]), label=f"{row['name']} delta")
        out.append(
            f"{row['name']}: verified {delta:+.2f}pp from round {row['round']} "
            f"({row['baseline_stage']} baseline -> {row['stage']} {row['config']})"
        )
    return out


def main() -> None:
    msgs = []
    msgs.extend(verify_main_summary())
    msgs.extend(verify_closure_highlights())
    print("metrics_snapshot_ok")
    for msg in msgs:
        print(f"- {msg}")


if __name__ == "__main__":
    main()
