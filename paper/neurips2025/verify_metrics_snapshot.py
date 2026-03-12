#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable
import statistics


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
METRICS_PATH = ROOT / "data" / "metrics.json"
POSTTRAIN_README = REPO / "posttrain_rwkv7" / "README.md"
RUNS_DIR = REPO / "posttrain_rwkv7" / "runs"
SHIPPED_ROUNDS = [*range(783, 789), *range(799, 809)]
FAMILY_MATCHERS: dict[str, Callable[[object], bool]] = {
    "protein_ss_spot": lambda task: str(task).startswith("protein_ss_"),
    "hotpot_text_restore": lambda task: str(task).startswith("hotpot_seed"),
    "mbpp_longctx_probe": lambda task: str(task).startswith("mbpp_longctx_"),
    "arc_mc_probe": lambda task: str(task).startswith("arc_mc_"),
    "squad_text_restore": lambda task: str(task).startswith("squad_seed"),
    "punc_restore": lambda task: str(task).startswith("punc_"),
}


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
            f"{task}: historical archive summary matches README "
            f"(best={row['best']:.2f}pp median={row['median']:.2f}pp pos={row['pos']}/{row['total']})"
        )
    return out


def family_med_deltas(task_matcher) -> list[float]:
    deltas: list[float] = []
    for round_id in SHIPPED_ROUNDS:
        path = RUNS_DIR / f"_round{round_id}_fastdiscover_records.jsonl"
        if not path.exists():
            continue
        rows = load_round_rows(round_id)
        tasks = sorted({str(row["task"]) for row in rows if task_matcher(row.get("task"))})
        for task in tasks:
            baseline_rows = [
                row for row in rows
                if row.get("task") == task
                and row.get("config") == "baseline"
                and row.get("stage") == "med"
                and row.get("status") in {"ok", "skip_reuse"}
                and row.get("best_val_tok_acc") is not None
            ]
            candidate_rows = [
                row for row in rows
                if row.get("task") == task
                and row.get("config") != "baseline"
                and row.get("stage") == "med"
                and row.get("status") in {"ok", "skip_reuse"}
                and row.get("best_val_tok_acc") is not None
            ]
            if not baseline_rows or not candidate_rows:
                continue
            baseline = max(baseline_rows, key=lambda row: float(row["best_val_tok_acc"]))
            candidate = max(candidate_rows, key=lambda row: float(row["best_val_tok_acc"]))
            deltas.append((float(candidate["best_val_tok_acc"]) - float(baseline["best_val_tok_acc"])) * 100.0)
    return deltas


def verify_snapshot_boundary_summary() -> list[str]:
    metrics = load_json(METRICS_PATH)["posttrain"]["snapshot_boundary_summary"]
    out = []
    for row in metrics:
        task = str(row["task"])
        if task not in FAMILY_MATCHERS:
            raise AssertionError(f"snapshot boundary summary missing matcher: {task}")
        deltas = family_med_deltas(FAMILY_MATCHERS[task])
        expected_best = row.get("best")
        expected_median = row.get("median")
        expected_pos = int(row["pos"])
        expected_total = int(row["total"])
        actual_pos = sum(delta > 0 for delta in deltas)
        actual_total = len(deltas)
        if actual_pos != expected_pos or actual_total != expected_total:
            raise AssertionError(
                f"{task} snapshot boundary count mismatch: expected {expected_pos}/{expected_total}, "
                f"got {actual_pos}/{actual_total}"
            )
        if deltas:
            if expected_best is None or expected_median is None:
                raise AssertionError(f"{task} snapshot boundary unexpectedly has null summary with non-empty deltas")
            assert_close(max(deltas), float(expected_best), label=f"{task} snapshot best")
            assert_close(statistics.median(deltas), float(expected_median), label=f"{task} snapshot median")
            out.append(
                f"{task}: shipped closure/breadth subset recomputes "
                f"(best={float(expected_best):.2f}pp median={float(expected_median):.2f}pp pos={expected_pos}/{expected_total})"
            )
        else:
            if expected_best is not None or expected_median is not None:
                raise AssertionError(f"{task} snapshot boundary expected empty summary but found non-null best/median")
            out.append(f"{task}: shipped closure/breadth subset has no medium-stage rows")
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
    msgs.extend(verify_snapshot_boundary_summary())
    msgs.extend(verify_closure_highlights())
    print("metrics_snapshot_ok")
    for msg in msgs:
        print(f"- {msg}")


if __name__ == "__main__":
    main()
